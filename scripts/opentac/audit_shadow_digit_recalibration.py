"""Offline/private audit of new expert/probe pairs against the frozen estimator.

Does not fit a model or change task/control files. Public features and estimates
are computed before opening private labels; labels are used for audit only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

from evaluate_shadow_digit_area_rate import read_contract


def metrics(truth, prediction):
    if not len(truth):
        return {"count": 0}
    error = np.asarray(prediction) - np.asarray(truth)
    return dict(count=len(truth), mae_N=float(np.abs(error).mean()),
                rmse_N=float(np.sqrt(np.mean(error ** 2))),
                p95_absolute_error_N=float(np.percentile(np.abs(error), 95)),
                bias_N=float(error.mean()))


def longest_run(mask):
    padded = np.r_[False, np.asarray(mask, bool), False].astype(int)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def audit(path, contract, model_path):
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    meta = result["metadata"]
    if result["status"] != "complete" or meta["action_repeat"] != 4:
        raise ValueError(f"Expected a completed 60 Hz collection: {path}")
    sensor = Path(meta["piano_config"])
    if hashlib.sha256(sensor.read_bytes()).hexdigest() != meta["piano_sha256"]:
        raise ValueError("Sensor provenance mismatch")
    model = contract.PublicDigitKeyForce(model_path, sensor)
    dt = float(meta["physics_dt_s"])
    if not np.isclose(dt, 1 / 240):
        raise ValueError("Expected unchanged 240 Hz physics")
    with h5py.File(path / "public.h5", "r") as public:
        if any(k in public for k in ("score", "reward", "success", "key_contact_force_N")):
            raise ValueError("Private label field in public HDF5")
        steps = public["physics_step"][:].astype(int)
        note_index = public["requested_note_index"][:].astype(int)
        midi = public["requested_midi"][:].astype(int)
        black = np.isin(midi % 12, (1, 3, 6, 8, 10))
        colors = np.where(black, "black", "white")
        time_s = public["time_s"][:]
        if not np.allclose(time_s, steps * dt):
            raise ValueError("Public clock differs from physics ticks")
        spacing = np.diff(steps)
        if steps[0] != 0 or np.any(spacing <= 0) or np.any(spacing[:-1] != 4) or not 0 < spacing[-1] <= 4:
            raise ValueError("Unexpected public sampling interval")
        if (np.any(note_index < 0) or np.any(note_index >= len(meta["notes"])) or
                not np.array_equal(midi, np.asarray(result["native_info"]["piano_sequence"]["midi"])[note_index])):
            raise ValueError("Public note intent does not match the configured phrase")
        reference = public["tactile/index/reference_depth_m"][:]
        x = np.full((len(steps), 5), np.nan)
        quality, bounds, estimates = [], [], np.full(len(steps), np.nan)
        for i, step in enumerate(steps):
            depth = public["tactile/index/depth_m"][i]
            diagnostics = {}
            estimate = model.predict(depth, reference, color=str(colors[i]), sample_step=i,
                                     current_step=i, physics_step=int(step), diagnostics_out=diagnostics)
            if diagnostics.get("features") is not None:
                x[i] = diagnostics["features"]
            quality.append(estimate["quality"])
            bounds.append([r["feature"] for r in diagnostics.get("exceeded_bounds", [])])
            if estimate["quality"] == "ok":
                estimates[i] = estimate["estimated_key_force_N"]
    # No true labels, score or private metadata enter the predictor above.
    with h5py.File(path / "private_labels.h5", "r") as private:
        psteps = private["physics_step"][:].astype(int)
        forces = private["key_contact_force_N"][:]
        angles = private["key_angle_rad"][:]
        if not np.array_equal(psteps, np.arange(steps[-1] + 1)):
            raise ValueError("Private force stream has missing/extra physics ticks")
        if (forces.shape != (len(psteps), 88) or angles.shape != forces.shape or
                not np.isfinite(forces).all() or not np.isfinite(angles).all() or np.any(forces < 0)):
            raise ValueError("Invalid measured labels")
    # A command owns (previous public step, current public step], even if the
    # score advances inside that action. Select the force channel from intent.
    owner = np.searchsorted(steps, psteps, side="left")
    pnote, pmidi = note_index[owner], midi[owner]
    pforce = forces[np.arange(len(psteps)), pmidi - 21]
    pangle = angles[np.arange(len(psteps)), pmidi - 21]
    y = forces[steps, midi - 21]
    initial = np.zeros(len(steps), bool)
    summaries = []
    for index, note in enumerate(meta["notes"]):
        p = np.flatnonzero(pnote == index)
        f = np.flatnonzero(note_index == index)
        down = pforce[p] >= .05
        start = psteps[p[down][0]] if down.any() else None
        if start is not None:
            initial[f] = (steps[f] >= start) & (steps[f] <= start + round(.12 / dt)) & (y[f] >= .05)
        is_black = "#" in note
        heavy = meta["dynamics"][index] == "heavy"
        target = (1. if heavy else .7) if is_black else (.5 if heavy else .35)
        tolerance = .14 if is_black else .07
        force_band = (np.abs(pforce[p] - target) <= tolerance) & (pforce[p] >= .1)
        in_band = force_band & (pangle[p] >= .025)
        non_target = forces[p].copy()
        non_target[:, int(pmidi[p[0]]) - 21] = 0
        accepted = (note_index == index) & np.isfinite(estimates)
        summaries.append(dict(index=index, note=note, dynamic=meta["dynamics"][index],
            public_frames=len(f), contact_frames=int(np.count_nonzero(y[f] >= .05)),
            initial_contact_frames=int(initial[f].sum()),
            private_peak_force_N=float(pforce[p].max()),
            physical_contact_ticks=int(down.sum()),
            longest_force_band_ticks=longest_run(force_band),
            longest_pressed_angle_ticks=longest_run(pangle[p] >= .025),
            longest_native_press_band_ticks=longest_run(in_band),
            longest_native_press_band_seconds=longest_run(in_band) * dt,
            needed_hold_seconds=float(meta["hold_seconds"][index]),
            other_key_contact_ticks=int(np.count_nonzero(np.max(non_target, axis=1) >= .05)),
            quality_counts=dict(Counter(np.asarray(quality)[f])),
            initial_quality_counts=dict(Counter(np.asarray(quality)[f][initial[f]])),
            accepted_estimate_error=metrics(y[accepted], estimates[accepted])))
    quality = np.asarray(quality)
    report = dict(episode=str(path), controller=meta["controller"], seed=meta["seed"],
        status=result["status"], native_notes_completed=result["native_info"]["notes_completed"],
        native_notes_attempted=result["native_info"]["notes_attempted"],
        native_failure=result["native_info"]["failure"], actions=len(steps)-1,
        public_frames=len(steps), private_physics_ticks=len(psteps),
        public_interval_counts={str(k): int(v) for k, v in zip(*np.unique(spacing, return_counts=True))},
        exact_step_alignment=True, physics_gaps=0, sensor_sha256=meta["piano_sha256"],
        private_peak_any_key_force_N=float(forces.max()),
        physical_ticks_at_or_above_2_N=int(np.count_nonzero(forces.max(axis=1) >= 2)),
        quality_counts=dict(Counter(quality)), per_note=summaries,
        exceeded_feature_counts=dict(Counter(b for row in bounds for b in row)),
        contact_quality_counts=dict(Counter(quality[y >= .05])),
        initial_quality_counts=dict(Counter(quality[initial])),
        accepted_contact_error=metrics(y[(y >= .05) & np.isfinite(estimates)],
                                       estimates[(y >= .05) & np.isfinite(estimates)]),
        accepted_initial_error=metrics(y[initial & np.isfinite(estimates)],
                                       estimates[initial & np.isfinite(estimates)]),
        error_scope="Frozen old estimator, accepted frames only; not new training/holdout accuracy")
    arrays = dict(x=x, y=y, estimated=estimates, quality=quality, black=black, steps=steps,
                  initial=initial, note_index=note_index, midi=midi,
                  private_steps=psteps, private_target_force=pforce)
    return report, arrays, model.sidecar


def coverage(runtime_path, records, sidecar):
    report = json.loads(runtime_path.read_text(encoding="utf-8"))
    rows = report["runs"][-1]["ood_rows"]
    combined = {name: np.concatenate([a[name] for _, a in records]) for name in ("x", "y", "black")}
    result = []
    for row in rows:
        black = row["note"].find("#") >= 0
        x, y = np.asarray(row["features"]), row["private_force_N"]
        choose = (combined["black"] == black) & (combined["y"] >= .05) & np.isfinite(combined["x"]).all(axis=1)
        near = choose & (np.abs(combined["y"] - y) <= (.05 if black else .025))
        def within(mask):
            values = combined["x"][mask]
            return dict(count=len(values), inside_feature_box=bool(len(values) and np.all(x >= values.min(axis=0)) and
                                                                   np.all(x <= values.max(axis=0))))
        scale = np.ptp(np.array([sidecar["models"]["black" if black else "white"][k]
                                 for k in ("feature_min", "feature_max")]), axis=0)
        scale[scale < 1e-9] = 1.
        candidates = np.flatnonzero(choose)
        distance = np.linalg.norm((combined["x"][choose] - x) / scale, axis=1)
        closest = candidates[np.argsort(distance)[:5]]
        result.append(dict(note=row["note"], private_force_N=y,
                           full_contact_envelope=within(choose), near_force_envelope=within(near),
                           nearest_5_feature_forces_N=combined["y"][closest].tolist(),
                           nearest_feature_distance=float(distance.min()) if len(distance) else None))
    return result


def plot(records, sidecar, expert_cache, runtime_report, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for record, a in records:
        fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
        axes[0].plot(a["private_steps"] / 240, a["private_target_force"], lw=.7, label="Private measured key force")
        axes[0].plot(a["steps"] / 240, a["estimated"], ".", ms=2, label="Old estimator: ok only")
        axes[0].set_ylabel("Force (N)"); axes[0].legend(fontsize=8)
        axes[1].plot(a["steps"] / 240, a["x"][:, 0]); axes[1].set_ylabel("Mean depth (mm)")
        for dim, threshold in ((1, .01), (2, .05), (3, .2)):
            axes[2].plot(a["steps"] / 240, a["x"][:, dim], label=f"area > {threshold} mm")
        axes[2].set_ylabel("Area fraction"); axes[2].legend(fontsize=8)
        axes[3].plot(a["steps"] / 240, a["x"][:, 4]); axes[3].set_ylabel("Area rate (/s)")
        axes[3].set_xlabel("Physics time (s)")
        for ax in axes:
            for index in range(1, 8):
                ids = np.flatnonzero(a["note_index"] == index)
                if len(ids): ax.axvline(a["steps"][ids[0]] / 240, color="gray", alpha=.25)
            ax.grid(alpha=.15)
        fig.suptitle(f"Seed {record['seed']} {record['controller']} (offline/private labels)")
        fig.savefig(output / f"{record['seed']}_{record['controller']}_alignment.png", dpi=150); plt.close(fig)
    old = []
    for path in sorted(expert_cache.glob("*.private.npz")):
        with np.load(path) as cache:
            old.append({k: cache[k] for k in ("x", "y", "black")})
    if not old:
        raise ValueError("Missing original 50-seed feature cache")
    old = {k: np.concatenate([c[k] for c in old]) for k in old[0]}
    runtime = json.loads(runtime_report.read_text(encoding="utf-8"))["runs"][-1]["ood_rows"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for column, (color, black) in enumerate((("white", False), ("black", True))):
        for r, (dim, label) in enumerate(((0, "Mean depth (mm)"), (2, "Area > .05 mm"), (4, "Area rate (/s)"))):
            ax = axes[r, column]; use = old["black"] == black
            ax.scatter(old["y"][use], old["x"][use, dim], s=2, alpha=.10, color="gray", label="Old 50 expert episodes")
            for record, a in records:
                use = (a["black"] == black) & (a["y"] >= .05)
                ax.scatter(a["y"][use], a["x"][use, dim], s=7, alpha=.4,
                           label=f"New {record['controller']} {record['seed']} 60 Hz")
            rows = [r for r in runtime if ("#" in r["note"]) == black]
            ax.scatter([r["private_force_N"] for r in rows], [r["features"][dim] for r in rows],
                       marker="x", s=45, color="red", label="Previous CaP-X OOD")
            ax.axhline(sidecar["models"][color]["feature_max"][dim], ls="--", lw=.8, color="red")
            ax.set(title=color, xlabel="Measured key-top force (N)", ylabel=label); ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output / "coverage_old_new_runtime.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expert-cache", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    contract = read_contract()
    records, sidecar = [], None
    for episode in args.episode:
        record, arrays, sidecar = audit(episode, contract, args.model)
        records.append((record, arrays))
        np.savez_compressed(args.output / f"{episode.name}.private.npz", **arrays)
    result = dict(schema="shadow_digit_recalibration_private_audit.v1", private_labels=True,
                  model_changed=False, trained=False, records=[r for r, _ in records],
                  previous_runtime_ood_coverage=coverage(args.runtime_report, records, sidecar),
                  coverage_scope="Feature boxes are only a coverage screen, not identifiability or prediction accuracy")
    (args.output / "report.private.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot(records, sidecar, args.expert_cache, args.runtime_report, args.output)
    for record, _ in records:
        print(json.dumps(record))
    print("Private audit:", args.output)


if __name__ == "__main__":
    main()
