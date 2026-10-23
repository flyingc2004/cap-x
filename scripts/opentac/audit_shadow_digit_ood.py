"""Offline audit of the frozen four-feature DIGIT model's coverage.

Reads existing expert HDF5 files only. Private labels stay offline; no model,
runtime thresholds, OpenTacBench task, or trajectory is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import h5py
import numpy as np


def load_features():
    # Import the NumPy-only contract without importing simulator registrations.
    path = Path(__file__).resolve().parents[2] / "capx/integrations/opentac_shadow_force.py"
    spec = importlib.util.spec_from_file_location("shadow_force_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_episode(path, sidecar, contract):
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    if result["status"] != "complete" or not result["native_info"]["success"]:
        raise ValueError(f"Incomplete or unsuccessful expert episode: {path}")
    if result["metadata"]["piano_sha256"] != sidecar["sensor_sha256"]:
        raise ValueError(f"Sensor config mismatch: {path}")
    seq = result["native_info"]["piano_sequence"]
    with h5py.File(path / "public.h5") as pub, h5py.File(path / "private_labels.h5") as priv:
        steps = pub["physics_step"][:]
        private_steps = priv["physics_step"][:]
        aligned = np.searchsorted(private_steps, steps)
        if np.any(aligned >= len(private_steps)) or not np.array_equal(private_steps[aligned], steps):
            raise ValueError(f"Step alignment failed: {path}")
        active = priv["score_count"][:][aligned]
        requested = priv["requested_midi"][:][aligned]
        ids = np.flatnonzero((active >= 0) & (active < len(seq["midi"])) &
                            (requested == np.asarray(seq["midi"])[np.clip(active, 0, len(seq["midi"]) - 1)]))
        reference = pub["tactile/index/reference_depth_m"][:]
        features, valid_fraction = [], []
        for frame in ids:
            depth = pub["tactile/index/depth_m"][frame]
            value, quality = contract.public_depth_features(depth, reference)
            if quality != "ok":
                raise ValueError(f"Invalid expert depth: {path} frame={frame} {quality}")
            features.append(value)
            valid_fraction.append(float(np.mean(np.isfinite(depth) & (depth > 0) &
                                                np.isfinite(reference) & (reference > 0))))
        indices = active[ids]
        forces = priv["key_contact_force_N"][:]
        force = forces[aligned[ids], requested[ids] - 21]
        row = dict(x=np.asarray(features), force=force,
                   seed=np.full(len(ids), result["metadata"]["seed"]), frame=ids,
                   steps=steps[ids], note=requested[ids],
                   black=np.asarray(["#" in seq["notes"][i] for i in indices]),
                   heavy=np.asarray([seq["dynamics"][i] == "heavy" for i in indices]),
                   valid_fraction=np.asarray(valid_fraction),
                   hand=pub["hand_state"][:][ids], action=pub["action_target"][:][ids])
    selected = np.zeros(len(ids), bool)
    # Same force-bin sampling as the original model builder, before color split.
    bins = np.floor(force.astype(float) / .1).astype(int)
    for value in np.unique(bins):
        group = np.flatnonzero(bins == value)
        selected[group[np.linspace(0, len(group) - 1, min(len(group), 30)).astype(int)]] = True
    row["selected"] = selected
    return row


def describe(values):
    values = np.asarray(values)
    if not values.size:
        return {"count": 0}
    return dict(count=int(len(values)), min=float(values.min()),
                p05=float(np.percentile(values, 5)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data/ljz/opentacbench/shadow_digit_force_v1/balanced_50seed_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    contract = load_features()
    model_path = args.root / "shadow_digit_key_top_force_4d.v1.json"
    raw = model_path.read_bytes()
    sidecar = json.loads(raw)
    episodes = []
    for seed in range(3100, 3150):
        paths = list((args.root / "episodes").glob(f"{seed}_expert_*"))
        if len(paths) != 1:
            raise ValueError(f"Expected one episode for seed {seed}")
        row = load_episode(paths[0], sidecar, contract)
        episodes.append(row)
        print(f"seed={seed} frames={len(row['x'])}", flush=True)
    data = {key: np.concatenate([e[key] for e in episodes]) for key in episodes[0]}
    report = dict(schema="shadow_digit_ood_coverage_audit.v1", source_root=str(args.root),
                  sidecar_sha256=hashlib.sha256(raw).hexdigest(),
                  feature_names=list(contract.FEATURE_NAMES),
                  exact_historical_capx_features_available=False,
                  runtime_quality_gate_was_applied=True, groups={})
    bounds = np.empty_like(data["x"])
    for color, black in (("white", False), ("black", True)):
        mask = data["black"] == black
        bounds[mask] = np.asarray(sidecar["models"][color]["feature_max"])
    ood_dims = data["x"] > bounds
    # Sidecar lower bounds are zero; features are all nonnegative by definition.
    ood = np.any(ood_dims, axis=1)
    for color, black in (("white", False), ("black", True)):
        color_mask = data["black"] == black
        for dynamic, heavy in (("light", False), ("heavy", True)):
            for split, split_mask in (("train", data["seed"] < 3140), ("holdout", data["seed"] >= 3140)):
                mask = color_mask & (data["heavy"] == heavy) & split_mask
                contact = mask & (data["force"] > .05)
                report["groups"][f"{split}_{color}_{dynamic}"] = dict(
                    frames=int(mask.sum()), contact_frames=int(contact.sum()),
                    ood_frames=int(ood[mask].sum()), ood_contact_frames=int(ood[contact].sum()),
                    ood_by_feature=dict(zip(contract.FEATURE_NAMES, ood_dims[mask].sum(axis=0).tolist())),
                    feature_max_full=data["x"][mask].max(axis=0).tolist(),
                    valid_depth_fraction=describe(data["valid_fraction"][mask]),
                )
        trained = color_mask & data["selected"] & (data["seed"] < 3140)
        report[f"{color}_training_subset"] = dict(frames=int(trained.sum()),
            maxima=data["x"][trained].max(axis=0).tolist(),
            frozen_upper_bounds=sidecar["models"][color]["feature_max"])
        for dynamic, heavy in (("light", False), ("heavy", True)):
            # Compare poses near the private force observed at CaP-X early OOD.
            mask = color_mask & (data["heavy"] == heavy) & (data["force"] >= .15) & (data["force"] <= .20)
            report[f"{color}_{dynamic}_at_0p15_to_0p20_N"] = dict(
                frames=int(mask.sum()), seeds=int(len(np.unique(data["seed"][mask]))),
                features={name: describe(data["x"][mask, j]) for j, name in enumerate(contract.FEATURE_NAMES)},
                command_z_m=describe(data["action"][mask, 2]),
                command_flexion_rad=describe(data["action"][mask, 3]),
                actual_z_m=describe(data["hand"][mask, 2]),
                actual_flexion_rad=describe(data["hand"][mask, 3]))
            # The same force occurs during both pressing and release. Separate
            # these with command motion, without reading expert phase internals.
            delta = np.concatenate([np.diff(e["action"][:, 3], prepend=e["action"][0, 3]) for e in episodes])
            for direction, moving in (("closing", delta > 1e-6), ("opening", delta < -1e-6)):
                phase_mask = mask & moving
                report[f"{color}_{dynamic}_at_0p15_to_0p20_N"][direction] = dict(
                    frames=int(phase_mask.sum()),
                    command_z_m=describe(data["action"][phase_mask, 2]),
                    command_flexion_rad=describe(data["action"][phase_mask, 3]),
                    actual_z_m=describe(data["hand"][phase_mask, 2]),
                    actual_flexion_rad=describe(data["hand"][phase_mask, 3]))
    report["total_frames"] = len(data["x"])
    report["total_ood"] = int(ood.sum())
    report["scope"] = "Existing successful expert paths at 30 Hz; not coverage of new CaP-X contact poses or intermediate 240 Hz states."
    (args.output / "coverage.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/capx-shadow-ood-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for row, (color, black) in enumerate((("white", False), ("black", True))):
        for col, name in enumerate(contract.FEATURE_NAMES):
            ax = axes[row, col]
            for dynamic, heavy, ink in (("light", False, "tab:blue"), ("heavy", True, "tab:orange")):
                ids = np.flatnonzero((data["black"] == black) & (data["heavy"] == heavy) & (data["force"] > .01))
                ids = ids[np.linspace(0, len(ids) - 1, min(1500, len(ids))).astype(int)]
                ax.scatter(data["force"][ids], data["x"][ids, col], s=3, alpha=.3, color=ink, label=dynamic)
            ax.axhline(sidecar["models"][color]["feature_max"][col], color="red", ls="--", label="OOD upper bound")
            ax.set(title=f"{color}: {name}", xlabel="Private key force (N), offline only")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Existing 50 expert episodes: frozen 4D feature coverage")
    figure.savefig(args.output / "expert_feature_coverage.png", dpi=160)
    plt.close(figure)
    print(json.dumps({"frames":report["total_frames"], "ood": report["total_ood"], "output":str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
