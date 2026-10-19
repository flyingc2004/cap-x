"""Audit and (when the required numeric observations exist) calibrate piano key force.

Run with the Isaac-free UniVTAC Python environment.  Force labels are used only
inside this offline program.  A frozen sidecar is emitted only on explicit
request after an episode-level holdout has passed a caller-supplied MAE gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[2]
_feature_module_path = REPO / "capx/integrations/opentac/public_force_calibration.py"
_feature_spec = importlib.util.spec_from_file_location("opentac_public_force_features", _feature_module_path)
if _feature_spec is None or _feature_spec.loader is None:
    raise RuntimeError(f"cannot load committed public feature contract: {_feature_module_path}")
_feature_module = importlib.util.module_from_spec(_feature_spec)
_feature_spec.loader.exec_module(_feature_module)
FEATURE_NAMES = _feature_module.FEATURE_NAMES
extract_public_tactile_force_features = _feature_module.extract_public_tactile_force_features

SCHEMA = "play_piano_public_key_top_force_calibration.v1"
FEATURE_SCHEMA = "play_piano_bilateral_gelsight_depth_marker_26d.v1"
REQUIRED_TACTILE = tuple(
    f"tactile/{hand}/{field}"
    for hand in ("left_tactile", "right_tactile")
    for field in ("depth", "marker")
)
DEFAULT_DATASET = Path(
    "/mnt/data/ljz/opentacbench/datasets/OpenTacBench-tactile-memory/play_piano/episodes"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hdf5_path(episode: Path) -> Path:
    return episode / "hdf5" / f"{episode.name}.hdf5"


def audit_episode(episode: Path) -> dict:
    seed = int(episode.name)
    result = _json(episode / "result.json")
    record = {
        "seed": seed,
        "result": result.get("result"),
        "tool_grasped_native": result.get("piano_result", {}).get("tool_grasped"),
        "hold_seconds": result.get("piano_result", {}).get("sequence", {}).get("hold_seconds", []),
        "notes": result.get("piano_result", {}).get("sequence", {}).get("notes", []),
        "hdf5_status": "missing",
    }
    metadata_path = episode / "metadata.json"
    if metadata_path.exists() and metadata_path.stat().st_size > 1024:
        meta = _json(metadata_path).get(str(seed), {})
        trace = meta.get("piano_pressure_trace", [])
        record["pressure_trace_count"] = len(trace)
        record["pressure_trace_step_min"] = min((int(x["step"]) for x in trace), default=None)
        record["pressure_trace_step_max"] = max((int(x["step"]) for x in trace), default=None)
        record["marker_reference_reset_result"] = meta.get("marker_reference_reset_result")
    h5path = _hdf5_path(episode)
    if not h5path.exists():
        return record
    if h5path.stat().st_size < 1024:
        record["hdf5_status"] = "lfs_pointer_not_checked_out"
        return record
    with h5py.File(h5path) as h5:
        names: list[str] = []
        h5.visititems(lambda name, item: names.append(name) if isinstance(item, h5py.Dataset) else None)
        steps = np.asarray(h5["phase/sim_step"], dtype=np.int64)
        force_steps = np.asarray(h5["piano/physics_step"], dtype=np.int64)
        context = json.loads(h5.attrs["episode_context_json"])
        dt = float(context["timing_contract"]["physics_dt_s"])
        record.update(
            hdf5_status="readable",
            frames=int(len(steps)),
            physics_dt_s=dt,
            observation_step_gaps=dict(Counter(map(int, np.diff(steps)))),
            observed_hz=float(1 / (dt * np.median(np.diff(steps)))),
            force_label_count=int(len(force_steps)),
            force_label_step_aligned=bool(np.array_equal(steps, force_steps)),
            tactile_datasets=[name for name in names if name.startswith("tactile/")],
            missing_numeric_tactile=[name for name in REQUIRED_TACTILE if name not in h5],
            actual_force_max_N=float(np.max(h5["piano/actual_force_N"])),
            contact_frames=int(np.count_nonzero(h5["piano/contact"])),
        )
        if "pressure_trace_count" in record:
            meta = _json(metadata_path)[str(seed)]
            trace_steps = np.asarray([int(x["step"]) for x in meta["piano_pressure_trace"]])
            record["pressure_trace_exact_observation_matches"] = int(
                np.count_nonzero(np.isin(trace_steps, steps))
            )
            record["pressure_trace_nearest_offset_steps_max"] = int(
                max(
                    (min(abs(int(step) - int(steps[np.clip(np.searchsorted(steps, step) + shift, 0, len(steps) - 1)]))
                         for shift in (-1, 0)) for step in trace_steps),
                    default=0,
                )
            )
    return record


def _plot_force_only(episodes: list[Path], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    candidates = [ep for ep in episodes if _hdf5_path(ep).exists() and _hdf5_path(ep).stat().st_size > 1024]
    for episode in (candidates[:2] + candidates[-1:]):
        with h5py.File(_hdf5_path(episode)) as h5:
            t = np.asarray(h5["piano/time_s"])
            force = np.asarray(h5["piano/actual_force_N"])
            target = np.asarray(h5["piano/target_force_N"])
            contact = np.asarray(h5["piano/contact"])
            numeric_missing = any(name not in h5 for name in REQUIRED_TACTILE)
        fig, axes = plt.subplots(3, 1, figsize=(12, 5.8), sharex=True, layout="constrained")
        axes[0].plot(t, force, linewidth=0.8, label="measured max key force (private label)")
        axes[0].plot(t, target, linewidth=0.7, alpha=0.55, label="requested force (not a label)")
        axes[0].fill_between(t, 0, 0.15, where=contact, alpha=0.18, label="native contact")
        axes[0].set_ylabel("N")
        axes[0].legend(loc="upper right", fontsize=7)
        for axis, hand in zip(axes[1:], ("left", "right"), strict=True):
            axis.set_ylim(0, 1)
            axis.set_yticks([])
            axis.set_ylabel(hand)
            if numeric_missing:
                axis.text(0.5, 0.5, "numeric depth / marker coordinates absent from HDF5",
                          ha="center", va="center", transform=axis.transAxes)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"play_piano seed {episode.name}: force clock and unavailable 26D inputs")
        path = output / f"seed_{episode.name}_force_alignment.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _stage(step: int, note: int, force: float, target: float, hold_s: float,
           press_end: dict[int, int], release_end: dict[int, int], dt: float) -> str:
    last_press = press_end.get(note)
    if last_press is not None and last_press < step <= release_end.get(note, last_press):
        return "release"
    if force < 0.1:
        return "no_contact"
    if last_press is not None and step >= last_press - int(math.ceil(hold_s / dt)) and abs(force - target) <= 0.45:
        return "hold_approx"
    return "contact"


def load_samples(episode: Path, far_plane_mm: float) -> dict[str, np.ndarray | list]:
    seed = int(episode.name)
    meta = _json(episode / "metadata.json")[str(seed)]
    trace = {int(x["step"]): x for x in meta.get("piano_pressure_trace", [])}
    press_end: dict[int, int] = {}
    for row in trace.values():
        i = int(row["note"])
        press_end[i] = max(press_end.get(i, -1), int(row["step"]))
    release_end = {
        int(row["previous_note"]): int(row["step"])
        for row in meta.get("piano_contact_events", [])
        if int(row.get("next_note", -1)) > int(row.get("previous_note", -1))
    }
    sequence = meta["episode_context"]["piano_sequence"]
    with h5py.File(_hdf5_path(episode)) as h5:
        missing = [name for name in REQUIRED_TACTILE if name not in h5]
        if missing:
            raise ValueError(f"seed {seed} lacks numeric public tactile datasets: {missing}")
        steps = np.asarray(h5["piano/physics_step"], dtype=np.int64)
        actual = np.asarray(h5["piano/actual_force_N"], dtype=float)
        actual_midi = np.asarray(h5["piano/actual_midi"], dtype=int)
        target_midi = np.asarray(h5["piano/target_midi"], dtype=int)
        target_force = np.asarray(h5["piano/target_force_N"], dtype=float)
        note_index = np.asarray(h5["piano/target_index"], dtype=int)
        dt = float(json.loads(h5.attrs["episode_context_json"])["timing_contract"]["physics_dt_s"])
        xs, ys, keys, holds, stages, times, kept_steps = [], [], [], [], [], [], []
        for frame, step_raw in enumerate(steps):
            step = int(step_raw)
            note = int(note_index[frame])
            if note >= len(sequence["midi"]) or int(target_midi[frame]) < 0:
                continue
            key = int(target_midi[frame])
            row = trace.get(step)
            if row is not None and int(row["key"]) + 21 == key:
                label = float(row["force_N"])
            elif actual[frame] <= 0.01:
                label = 0.0  # the maximum of all 88 key forces is zero
            elif int(actual_midi[frame]) == key:
                label = float(actual[frame])  # the measured maximum is the specified key
            else:
                continue  # a different key carries the load; no target-key label
            if not np.isfinite(label) or label < 0:
                continue
            features = extract_public_tactile_force_features(
                (h5[f"tactile/{hand}/depth"][frame] for hand in ("left_tactile", "right_tactile")),
                (h5[f"tactile/{hand}/marker"][frame] for hand in ("left_tactile", "right_tactile")),
                far_plane_mm=far_plane_mm,
            )
            hold = float(sequence["hold_seconds"][note])
            xs.append(features)
            ys.append(label)
            keys.append(key)
            holds.append(hold)
            stages.append(_stage(step, note, label, float(target_force[frame]), hold,
                                 press_end, release_end, dt))
            times.append(float(step * dt))
            kept_steps.append(step)
    if not xs:
        raise ValueError(f"seed {seed} has no aligned public features and target-key force labels")
    return {
        "seed": seed,
        "x": np.asarray(xs), "y": np.asarray(ys), "key": np.asarray(keys),
        "hold_s": np.asarray(holds), "stage": np.asarray(stages),
        "time_s": np.asarray(times), "step": np.asarray(kept_steps),
        "dt": dt,
    }


def _variant(data: dict, name: str) -> np.ndarray:
    x = data["x"]
    if name == "depth_only":
        return x[:, [0, 13]]
    if name == "marker_only":
        return x[:, [i for i in range(26) if i not in (0, 13)]]
    if name == "full_26d":
        return x
    if name == "full_26d_plus_delta":
        delta = np.zeros_like(x)
        adjacent = (np.diff(data["step"]) <= 2) & (np.diff(data["key"]) == 0)
        delta[1:][adjacent] = (x[1:][adjacent] - x[:-1][adjacent]) / np.maximum(
            np.diff(data["time_s"])[adjacent, None], 1e-9
        )
        return np.concatenate((x, delta), axis=1)
    raise ValueError(name)


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    if not len(y):
        return {"count": 0}
    error = prediction - y
    return {
        "count": int(len(y)),
        "mae_N": float(np.mean(np.abs(error))),
        "rmse_N": float(np.sqrt(np.mean(error**2))),
        "p95_absolute_error_N": float(np.percentile(np.abs(error), 95)),
        "bias_N": float(np.mean(error)),
    }


def _evaluate(episodes: list[dict], predictions: list[np.ndarray]) -> dict:
    y = np.concatenate([ep["y"] for ep in episodes])
    p = np.concatenate(predictions)
    key = np.concatenate([ep["key"] for ep in episodes])
    hold = np.concatenate([ep["hold_s"] for ep in episodes])
    stage = np.concatenate([ep["stage"] for ep in episodes])
    groups = {
        "by_key_midi": {str(k): _metrics(y[key == k], p[key == k]) for k in sorted(set(key))},
        "by_hold_seconds": {str(k): _metrics(y[hold == k], p[hold == k]) for k in sorted(set(hold))},
        "by_stage": {str(k): _metrics(y[stage == k], p[stage == k]) for k in sorted(set(stage))},
        "near_1p5_N": _metrics(y[np.abs(y - 1.5) <= 0.45], p[np.abs(y - 1.5) <= 0.45]),
        "near_3p0_N": _metrics(y[np.abs(y - 3.0) <= 0.45], p[np.abs(y - 3.0) <= 0.45]),
    }
    overloaded = y > 6.0  # native PianoScore's instantaneous key-overload limit
    no_contact = y < 0.1
    groups["overforce_6N"] = {
        "positive_frames": int(overloaded.sum()),
        "missed_frames": int(np.count_nonzero(overloaded & (p <= 6.0))),
        "miss_rate": float(np.mean(p[overloaded] <= 6.0)) if overloaded.any() else None,
    }
    groups["no_contact_false_positive"] = {
        "negative_frames": int(no_contact.sum()),
        "false_positive_frames": int(np.count_nonzero(no_contact & (p >= 0.1))),
        "rate": float(np.mean(p[no_contact] >= 0.1)) if no_contact.any() else None,
    }
    lags = []
    for ep, estimate in zip(episodes, predictions, strict=True):
        truth = ep["y"]
        if len(truth) < 20:
            continue
        scores = {}
        for lag in range(-5, 6):
            yy = truth[max(0, lag):len(truth) + min(0, lag)]
            pp = estimate[max(0, -lag):len(estimate) + min(0, -lag)]
            scores[lag] = float(np.mean((pp - yy) ** 2))
        lags.append(min(scores, key=scores.get) * float(np.median(np.diff(ep["time_s"]))))
    groups["best_alignment_lag_seconds_median"] = float(np.median(lags)) if lags else None
    return {"overall": _metrics(y, p), **groups}


def _balanced_indices(force: np.ndarray) -> np.ndarray:
    """Limit long no-contact plateaus within each training episode/force bin."""
    bins = np.floor(np.asarray(force) / 0.25).astype(int)
    picked = []
    for bin_id in np.unique(bins):
        candidates = np.flatnonzero(bins == bin_id)
        if len(candidates) > 20:
            candidates = candidates[np.linspace(0, len(candidates) - 1, 20).astype(int)]
        picked.append(candidates)
    return np.sort(np.concatenate(picked))


def train(episodes: list[Path], far_plane_mm: float, out_dir: Path,
          emit_sidecar: bool, max_mae_N: float | None) -> dict:
    audits = [audit_episode(ep) for ep in episodes]
    unavailable = [a for a in audits if a["hdf5_status"] != "readable" or a.get("missing_numeric_tactile")]
    if unavailable:
        raise ValueError(
            f"training blocked: {len(unavailable)}/{len(audits)} episodes lack checked-out HDF5 "
            "or bilateral numeric depth/marker; see audit.json. RGB JPEG is not the 26D input"
        )
    data = [load_samples(ep, far_plane_mm) for ep in episodes]
    train_data = [d for d in data if d["seed"] % 5 != 0]
    test_data = [d for d in data if d["seed"] % 5 == 0]
    if len(train_data) < 3 or len(test_data) < 2:
        raise ValueError("need at least three training and two held-out whole episodes")
    report = {
        "schema_version": SCHEMA,
        "split": "whole episode; seed % 5 == 0 held out",
        "training_seeds": [d["seed"] for d in train_data],
        "held_out_seeds": [d["seed"] for d in test_data],
        "training_sampling": "at most 20 frames per seed per 0.25 N measured-force bin; holdout uses all eligible frames",
        "far_plane_mm": far_plane_mm,
        "models": {},
    }
    selected = [_balanced_indices(d["y"]) for d in train_data]
    report["training_sample_count"] = int(sum(len(indices) for indices in selected))
    report["held_out_sample_count"] = int(sum(len(d["y"]) for d in test_data))
    frozen = None
    for name in ("depth_only", "marker_only", "full_26d", "full_26d_plus_delta"):
        xtrain = np.concatenate([_variant(d, name)[indices] for d, indices in zip(train_data, selected, strict=True)])
        ytrain = np.concatenate([d["y"][indices] for d, indices in zip(train_data, selected, strict=True)])
        median = np.median(xtrain, axis=0)
        iqr = np.percentile(xtrain, 75, axis=0) - np.percentile(xtrain, 25, axis=0)
        iqr[iqr < 1e-9] = 1.0
        z = (xtrain - median) / iqr
        design = np.concatenate((z, z * z), axis=1)
        model = Ridge(alpha=0.01).fit(design, ytrain)
        predictions = []
        for d in test_data:
            ztest = (_variant(d, name) - median) / iqr
            predictions.append(model.predict(np.concatenate((ztest, ztest * ztest), axis=1)))
        report["models"][name] = _evaluate(test_data, predictions)
        if name == "full_26d":
            frozen = {
                "schema_version": SCHEMA,
                "calibration_id": "play_piano_public_key_top_force.v1",
                "target": "specified_key_top_contact_force_N",
                "input_contract": {
                    "feature_schema_version": FEATURE_SCHEMA,
                    "feature_transform": "robust_median_iqr_then_linear_and_square",
                    "far_plane_mm": far_plane_mm,
                },
                "feature_names": list(FEATURE_NAMES),
                "scaler": {"median": median.tolist(), "iqr": iqr.tolist()},
                "regression": {
                    "ridge_alpha": 0.01,
                    "bias": float(model.intercept_),
                    "weights": model.coef_.tolist(),
                    "output_clipped": False,
                },
                "training": {
                    "seeds": report["training_seeds"],
                    "held_out_seeds": report["held_out_seeds"],
                    "sampling": report["training_sampling"],
                    "labels": "offline piano_pressure_trace or observed specified-key contact only",
                },
                "held_out_evaluation": report["models"][name],
                "private_label_use": "offline supervision only; no frame labels stored",
            }
    if emit_sidecar:
        if max_mae_N is None or max_mae_N <= 0:
            raise ValueError("--emit-sidecar requires positive --max-mae-N")
        mae = report["models"]["full_26d"]["overall"]["mae_N"]
        if mae > max_mae_N:
            raise ValueError(f"holdout MAE {mae:.3f} N exceeds gate {max_mae_N:.3f} N")
        sidecar = out_dir / "play_piano_public_key_top_force.v1.json"
        sidecar.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        report["sidecar"] = str(sidecar)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--far-plane-mm", type=float)
    parser.add_argument("--emit-sidecar", action="store_true")
    parser.add_argument("--max-mae-N", type=float)
    args = parser.parse_args()
    episodes = sorted((p for p in args.episodes_root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    if not episodes:
        parser.error("no numeric-seed episode directories found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audits = [audit_episode(ep) for ep in episodes]
    report = {
        "dataset_root": str(args.episodes_root.resolve()),
        "episode_count": len(episodes),
        "success_count": sum(a["result"] == "success" for a in audits),
        "failure_count": sum(a["result"] != "success" for a in audits),
        "numeric_depth_marker_ready_count": sum(a["hdf5_status"] == "readable" and not a.get("missing_numeric_tactile") for a in audits),
        "episodes": audits,
    }
    report["private_audit_plots"] = _plot_force_only(episodes, args.output_dir)
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.train:
        print(json.dumps({k: report[k] for k in ("episode_count", "success_count", "failure_count", "numeric_depth_marker_ready_count", "private_audit_plots")}, indent=2))
        return 0
    if args.far_plane_mm is None or not math.isfinite(args.far_plane_mm) or args.far_plane_mm <= 0:
        parser.error("--train requires a verified positive --far-plane-mm from the piano sensor")
    try:
        trained = train(episodes, args.far_plane_mm, args.output_dir, args.emit_sidecar, args.max_mae_N)
    except ValueError as exc:
        print(f"TRAINING BLOCKED: {exc}", file=sys.stderr)
        return 2
    (args.output_dir / "holdout_metrics.json").write_text(json.dumps(trained, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"holdout_metrics": str(args.output_dir / "holdout_metrics.json"), "sidecar": trained.get("sidecar")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
