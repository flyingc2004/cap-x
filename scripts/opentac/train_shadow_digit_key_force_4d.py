"""Freeze a four-feature Shadow DIGIT key-top force sidecar from 50 episodes.

Complete seeds 3100..3139 train; 3140..3149 remain held out. The sidecar
contains no frame-level labels. Output is refused if it already exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from capx.integrations.opentac_shadow_force import FEATURE_NAMES, SCHEMA
from evaluate_shadow_digit_depth_50seed import _balanced_training_rows, _evaluate, _load_episode


def build(root: Path) -> tuple[dict, dict]:
    episodes = []
    for seed in range(3100, 3150):
        paths = list((root / "episodes").glob(f"{seed}_expert_*"))
        if len(paths) != 1:
            raise FileNotFoundError(f"Expected exactly one expert trajectory for {seed}")
        episodes.append(_load_episode(paths[0]))
    hashes = {episode["sensor_sha256"] for episode in episodes}
    if len(hashes) != 1:
        raise ValueError("Mixed TACTO configs in training episodes")
    train, heldout = episodes[:40], episodes[40:]
    xtrain = np.concatenate([episode["x"][_balanced_training_rows(episode)][:, [0, 5, 6, 7]]
                             for episode in train])
    ytrain = np.concatenate([episode["y"][_balanced_training_rows(episode)] for episode in train])
    btrain = np.concatenate([episode["black"][_balanced_training_rows(episode)] for episode in train])
    models = {}
    predictions = []
    for color, is_black in (("white", False), ("black", True)):
        selected = btrain == is_black
        x, y = xtrain[selected], ytrain[selected]
        median = np.median(x, axis=0)
        iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
        iqr[iqr < 1e-9] = 1.
        z = (x - median) / iqr
        regression = Ridge(alpha=.01).fit(np.c_[z, z * z], y)
        models[color] = dict(median=median.tolist(), iqr=iqr.tolist(),
                             coef=regression.coef_.tolist(), intercept=float(regression.intercept_),
                             feature_min=np.maximum(0, np.min(x, axis=0) - .05 * np.ptp(x, axis=0)).tolist(),
                             feature_max=(np.max(x, axis=0) + .05 * np.ptp(x, axis=0)).tolist(),
                             training_frames=int(len(x)))
    for episode in heldout:
        features = episode["x"][:, [0, 5, 6, 7]]
        result = np.empty(len(features), float)
        for color, is_black in (("white", False), ("black", True)):
            selected = episode["black"] == is_black
            group = models[color]
            z = (features[selected] - group["median"]) / group["iqr"]
            result[selected] = np.c_[z, z * z] @ group["coef"] + group["intercept"]
        predictions.append(result)
    metrics = _evaluate(heldout, np.concatenate(predictions))
    sidecar = dict(schema=SCHEMA, feature_names=list(FEATURE_NAMES),
                   label="private PhysX specified-key top contact force N",
                   sensor_sha256=hashes.pop(), sensor_range_force_N=[0, 2],
                   training_seeds=list(range(3100, 3140)), held_out_seeds=list(range(3140, 3150)),
                   models=models, frame_labels_included=False)
    report = dict(schema=SCHEMA, split="complete episode", held_out=metrics,
                  training_frames=sum(model["training_frames"] for model in models.values()),
                  sidecar_has_frame_labels=False,
                  interpretation="TACTO image inversion; no independent fingertip force sensor")
    return sidecar, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data/ljz/opentacbench/shadow_digit_force_v1/balanced_50seed_v1"))
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.sidecar.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite a sidecar or evaluation report")
    sidecar, report = build(args.root)
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sidecar": str(args.sidecar), "report": str(args.report),
                      "contact": report["held_out"]["contact"]}, indent=2))


if __name__ == "__main__":
    main()
