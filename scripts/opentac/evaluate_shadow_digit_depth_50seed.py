"""Offline depth-only audit of the balanced Shadow DIGIT 50-seed suite.

Private PhysX key forces are labels here only. This script never emits a
runtime sidecar or exposes labels to CaP-X generated code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge

DEFAULT_ROOT = Path("/mnt/data/ljz/opentacbench/shadow_digit_force_v1/balanced_50seed_v1")
FEATURE_NAMES = ("mean_mm", "std_mm", "median_mm", "p90_mm", "p99_mm",
                 "area_gt_0p01mm", "area_gt_0p05mm", "area_gt_0p2mm")


def _features(depth: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.abs(reference[valid] - depth[valid]) * 1000
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Depth frame has no finite valid pixels")
    return np.array((values.mean(), values.std(), *np.percentile(values, (50, 90, 99)),
                     np.mean(values > .01), np.mean(values > .05), np.mean(values > .2)), float)


def _load_episode(path: Path) -> dict:
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("native_info") is None:
        raise ValueError(f"Incomplete episode: {path}")
    meta = result["metadata"]
    if (meta.get("sequence_source") != "balanced_four_groups_8note_v1" or
            meta.get("record_video") is not False or
            meta.get("range_force_N") != [0, 2] or
            not meta.get("require_dynamics")):
        raise ValueError(f"Unexpected collection contract: {path}")
    sequence = result["native_info"]["piano_sequence"]
    if len(sequence["midi"]) != 8 or len(sequence["dynamics"]) != 8:
        raise ValueError(f"Unexpected eight-note sequence: {path}")
    with h5py.File(path / "public.h5") as pub, h5py.File(path / "private_labels.h5") as private:
        steps = np.asarray(pub["physics_step"], np.int64)
        private_steps = np.asarray(private["physics_step"], np.int64)
        if (not len(steps) or not np.all(np.diff(steps) > 0) or
                not np.array_equal(private_steps, np.arange(len(private_steps)))):
            raise ValueError(f"Invalid step order: {path}")
        row = np.searchsorted(private_steps, steps)
        if np.any(row >= len(private_steps)) or not np.array_equal(private_steps[row], steps):
            raise ValueError(f"Public/private step mismatch: {path}")
        active = np.asarray(private["score_count"][row], int)
        requested = np.asarray(private["requested_midi"][row], int)
        force_matrix = np.asarray(private["key_contact_force_N"], np.float32)
        if force_matrix.shape != (len(private_steps), 88):
            raise ValueError(f"Invalid private key force shape: {path}")
        selected = ((active >= 0) & (active < 8) &
                    (requested == np.asarray(sequence["midi"])[np.clip(active, 0, 7)]))
        frame_ids = np.flatnonzero(selected)
        reference = np.asarray(pub["tactile/index/reference_depth_m"], np.float32)
        depth = pub["tactile/index/depth_m"]
        valid = pub["tactile/index/valid_mask"]
        x = np.empty((len(frame_ids), len(FEATURE_NAMES)), float)
        for i, frame in enumerate(frame_ids):
            x[i] = _features(np.asarray(depth[frame]), reference,
                             np.asarray(valid[frame], bool))
        y = force_matrix[row[frame_ids], requested[frame_ids] - 21].astype(float)
        note_indices = active[frame_ids]
        black = np.array(["#" in sequence["notes"][i] for i in note_indices], bool)
        dynamic = np.array([sequence["dynamics"][i] for i in note_indices])
        key = requested[frame_ids]
        if not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError(f"Nonfinite or negative force/depth data: {path}")
    return dict(seed=int(meta["seed"]), x=x, y=y, black=black,
                dynamic=dynamic, key=key, steps=steps[frame_ids],
                success=bool(result["native_info"]["success"]),
                completed=int(result["native_info"]["notes_completed"]),
                sensor_sha256=meta["piano_sha256"], path=str(path))


def _metrics(y: np.ndarray, predicted: np.ndarray) -> dict:
    if not len(y):
        return {"count": 0}
    error = predicted - y
    return {"count": int(len(y)), "mae_N": float(np.mean(np.abs(error))),
            "rmse_N": float(np.sqrt(np.mean(error**2))),
            "p95_absolute_error_N": float(np.percentile(np.abs(error), 95)),
            "bias_N": float(np.mean(error))}


def _balanced_training_rows(data: dict) -> np.ndarray:
    """Keep long zero-force travel periods from dominating the ridge fit."""
    bins = np.floor(data["y"] / .1).astype(int)
    selected = []
    for bin_id in np.unique(bins):
        rows = np.flatnonzero(bins == bin_id)
        selected.append(rows[np.linspace(0, len(rows) - 1, min(len(rows), 30)).astype(int)])
    return np.sort(np.concatenate(selected))


def _fit_predict(train: list[dict], held_out: list[dict], *, variant: str) -> tuple[np.ndarray, int]:
    selections = [_balanced_training_rows(episode) for episode in train]
    xtrain = np.concatenate([episode["x"][rows] for episode, rows in zip(train, selections, strict=True)])
    ytrain = np.concatenate([episode["y"][rows] for episode, rows in zip(train, selections, strict=True)])
    btrain = np.concatenate([episode["black"][rows] for episode, rows in zip(train, selections, strict=True)])
    xtest = np.concatenate([episode["x"] for episode in held_out])
    btest = np.concatenate([episode["black"] for episode in held_out])
    if variant == "mean_depth":
        columns = [0]
    elif variant in ("depth_stats", "depth_stats_by_key_color"):
        columns = list(range(len(FEATURE_NAMES)))
    else:
        raise ValueError(variant)
    output = np.empty(len(xtest), float)
    groups = (False, True) if variant == "depth_stats_by_key_color" else (None,)
    for group in groups:
        fit = np.ones(len(xtrain), bool) if group is None else btrain == group
        predict = np.ones(len(xtest), bool) if group is None else btest == group
        median = np.median(xtrain[fit][:, columns], axis=0)
        iqr = np.percentile(xtrain[fit][:, columns], 75, axis=0) - np.percentile(xtrain[fit][:, columns], 25, axis=0)
        iqr[iqr < 1e-9] = 1.0
        ztrain = (xtrain[fit][:, columns] - median) / iqr
        ztest = (xtest[predict][:, columns] - median) / iqr
        design_train = np.concatenate((ztrain, ztrain * ztrain), axis=1)
        design_test = np.concatenate((ztest, ztest * ztest), axis=1)
        output[predict] = Ridge(alpha=.01).fit(design_train, ytrain[fit]).predict(design_test)
    return output, int(len(ytrain))


def _evaluate(held_out: list[dict], prediction: np.ndarray) -> dict:
    y = np.concatenate([episode["y"] for episode in held_out])
    black = np.concatenate([episode["black"] for episode in held_out])
    dynamic = np.concatenate([episode["dynamic"] for episode in held_out])
    key = np.concatenate([episode["key"] for episode in held_out])
    upper = np.where(black, np.where(dynamic == "light", .84, 1.14),
                     np.where(dynamic == "light", .42, .57))
    target = np.where(black, np.where(dynamic == "light", .7, 1.0),
                      np.where(dynamic == "light", .35, .5))
    tolerance = np.where(black, .14, .07)
    no_contact = y < .05
    overforce = y > upper
    contact = y >= .1
    near_target = np.abs(y - target) <= tolerance
    lags = {}
    offset = 0
    for episode in held_out:
        truth = episode["y"]
        estimate = prediction[offset:offset + len(truth)]
        offset += len(truth)
        errors = {}
        for lag in range(-4, 5):
            actual = truth[:len(truth) - lag] if lag >= 0 else truth[-lag:]
            predicted = estimate[lag:] if lag >= 0 else estimate[:len(estimate) + lag]
            relevant = (actual >= .05) | (predicted >= .05)
            errors[lag] = float(np.mean(np.abs(predicted[relevant] - actual[relevant]))) if relevant.any() else float("inf")
        lags[str(episode["seed"])] = min(errors, key=errors.get) / 30
    return {
        "overall": _metrics(y, prediction),
        "contact": _metrics(y[contact], prediction[contact]),
        "near_target": _metrics(y[near_target], prediction[near_target]),
        "by_group": {f"{color}_{dyn}": _metrics(y[(black == (color == "black")) & (dynamic == dyn)],
                                                   prediction[(black == (color == "black")) & (dynamic == dyn)])
                     for color in ("white", "black") for dyn in ("light", "heavy")},
        "by_group_contact": {f"{color}_{dyn}": _metrics(y[contact & (black == (color == "black")) & (dynamic == dyn)],
                                                           prediction[contact & (black == (color == "black")) & (dynamic == dyn)])
                             for color in ("white", "black") for dyn in ("light", "heavy")},
        "by_key_midi": {str(k): _metrics(y[key == k], prediction[key == k]) for k in np.unique(key)},
        "no_contact_false_positive": {"frames": int(no_contact.sum()),
                                      "predicted_at_least_0p1_N": int(np.count_nonzero(no_contact & (prediction >= .1))),
                                      "rate": float(np.mean(prediction[no_contact] >= .1)) if no_contact.any() else None},
        "overforce_missed": {"frames": int(overforce.sum()),
                             "predicted_within_target": int(np.count_nonzero(overforce & (prediction <= upper))),
                             "rate": float(np.mean(prediction[overforce] <= upper[overforce])) if overforce.any() else None},
        "at_or_above_2_N_frames": int(np.count_nonzero(y >= 2)),
        "best_alignment_lag_seconds_by_seed": lags,
        "best_alignment_lag_seconds_median": float(np.median(list(lags.values()))),
        "predictions_clipped": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, default=DEFAULT_ROOT / "episodes")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "depth_holdout_metrics.json")
    args = parser.parse_args()
    episodes = []
    missing = []
    for seed in range(3100, 3150):
        matches = list(args.episodes_root.glob(f"{seed}_expert_*"))
        if len(matches) != 1 or not (matches[0] / "result.json").is_file():
            missing.append(seed)
        else:
            episodes.append(_load_episode(matches[0]))
    if missing:
        print(json.dumps({"missing_seeds": missing, "loaded_seeds": [ep["seed"] for ep in episodes]}))
        return 2
    hashes = {ep["sensor_sha256"] for ep in episodes}
    if len(hashes) != 1:
        raise ValueError("Mixed TACTO force/image configurations across episodes")
    train = [ep for ep in episodes if ep["seed"] <= 3139]
    held_out = [ep for ep in episodes if ep["seed"] >= 3140]
    report = {"schema": "shadow_digit_depth_force_50seed_holdout.v1",
              "label": "private PhysX specified-key contact force N",
              "input": "public DIGIT simulated depth only; key color only for separate-color variant",
              "sensor_sha256": hashes.pop(),
              "split": "complete episode: 3100..3139 train; 3140..3149 held out",
              "training_seeds": [ep["seed"] for ep in train],
              "held_out_seeds": [ep["seed"] for ep in held_out],
              "native_success_count": sum(ep["success"] for ep in episodes),
              "native_failure_count": sum(not ep["success"] for ep in episodes),
              "completed_notes_by_seed": {str(ep["seed"]): ep["completed"] for ep in episodes},
              "feature_names": list(FEATURE_NAMES), "models": {},
              "runtime_sidecar_emitted": False}
    for variant in ("mean_depth", "depth_stats", "depth_stats_by_key_color"):
        prediction, sample_count = _fit_predict(train, held_out, variant=variant)
        report["models"][variant] = _evaluate(held_out, prediction)
        report["models"][variant]["training_sample_count"] = sample_count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "held_out_seeds": report["held_out_seeds"],
                      "models": {name: value["overall"] for name, value in report["models"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
