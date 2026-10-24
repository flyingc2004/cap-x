"""Freeze the existing paired-seed pilot fit without changing its CV recipe.

Private labels stay in the audit input. The runtime JSON contains aggregate
normalization, coefficients and feature bounds, never per-frame labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from evaluate_shadow_digit_recalibration_pilot import load_audit


def freeze(audit: Path, template: Path, version: str):
    data = load_audit(audit)
    records = json.loads((audit / "report.private.json").read_text(encoding="utf-8"))["records"]
    seeds = sorted(set(int(v) for v in data["seed"]))
    if any(sum(r["seed"] == s and r["controller"] == p for r in records) != 1
           for s in seeds for p in ("expert", "probe")):
        raise ValueError("Each training seed must have one expert and one probe episode")
    original = json.loads(template.read_text(encoding="utf-8"))
    model = {key: original[key] for key in (
        "schema", "feature_names", "label", "sensor_sha256", "sensor_range_force_N", "history_contract")}
    model.update(calibration_version=version, training_seeds=seeds,
                 training_episode_count=len(records), held_out_seeds=[],
                 evaluation="Four-fold leave-one-paired-seed-out development CV; no final independent holdout",
                 runtime_validation="Experimental frozen candidate; online evaluation required",
                 frame_labels_included=False, force_features_include_robot_state=False,
                 ridge_alpha=.01, models={})
    for color, black in (("white", False), ("black", True)):
        rows = data["sampled"] & (data["black"] == black)
        x, y = data["x"][rows], data["y"][rows]
        median = np.median(x, axis=0)
        iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
        iqr[iqr < 1e-9] = 1
        z = (x - median) / iqr
        regression = Ridge(alpha=.01).fit(np.c_[z, z * z], y)
        lower = x.min(axis=0) - .05 * np.ptp(x, axis=0)
        upper = x.max(axis=0) + .05 * np.ptp(x, axis=0)
        lower[:4] = np.maximum(0, lower[:4])
        model["models"][color] = dict(median=median.tolist(), iqr=iqr.tolist(),
            coef=regression.coef_.tolist(), intercept=float(regression.intercept_),
            feature_min=lower.tolist(), feature_max=upper.tolist(), training_frames=int(rows.sum()))
    provenance = dict(schema="shadow_digit_freeze_provenance.private.v1", audit=str(audit),
        calibration_version=version, training_seeds=seeds, training_episode_count=len(records),
        input_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(audit.glob("*.private.npz"))},
        recipe="Per episode 0.1-N bins, at most 30 evenly spaced frames/bin; black/white Median/IQR; linear+square Ridge alpha=0.01; train bounds +/-5%; no output clipping",
        frame_labels_included=False)
    return model, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.provenance.exists():
        raise FileExistsError("Refusing to overwrite a frozen model or provenance")
    model, provenance = freeze(args.audit, args.template, args.version)
    payload = json.dumps(model, indent=2, allow_nan=False) + "\n"
    provenance["model_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    with args.provenance.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"model": str(args.output), "sha256": provenance["model_sha256"],
                      "version": args.version, "training_seeds": model["training_seeds"],
                      "training_frames": {k: v["training_frames"] for k, v in model["models"].items()}}))


if __name__ == "__main__":
    main()
