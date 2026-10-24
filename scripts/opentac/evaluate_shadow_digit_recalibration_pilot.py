"""Exploratory leave-one-paired-seed-out validation; never installs a model.

The four pilot seed groups are too small to establish online control accuracy.
Uses only the existing public 4D/5D features; private forces are offline labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from audit_shadow_digit_recalibration import metrics


def load_audit(path):
    report = json.loads((path / "report.private.json").read_text(encoding="utf-8"))
    rows = []
    for record in report["records"]:
        source = Path(record["episode"])
        meta = json.loads((source / "result.json").read_text(encoding="utf-8"))["metadata"]
        with np.load(path / f"{source.name}.private.npz") as data:
            keep = np.isfinite(data["x"]).all(axis=1)
            row = {k: data[k][keep] for k in ("x", "y", "black", "initial", "note_index", "midi")}
        size = len(row["y"])
        row.update(seed=np.full(size, record["seed"], int),
                   expert=np.full(size, record["controller"] == "expert", bool),
                   heavy=np.asarray(meta["dynamics"])[row["note_index"]] == "heavy")
        # Bound no-contact/steady-frame dominance per episode, using training
        # labels only when a fold selects this episode. Test frames stay whole.
        selected = np.zeros(size, bool)
        bins = np.floor(row["y"] / .1).astype(int)
        for value in np.unique(bins):
            ids = np.flatnonzero(bins == value)
            selected[ids[np.linspace(0, len(ids) - 1, min(30, len(ids))).astype(int)]] = True
        row["sampled"] = selected
        rows.append(row)
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def evaluate(data, dimensions):
    seeds = np.unique(data["seed"])
    if len(seeds) < 3:
        raise ValueError("Need at least three whole seed groups")
    prediction = np.full(len(data["y"]), np.nan)
    ood = np.zeros(len(prediction), bool)
    folds = []
    for heldout in seeds:
        test = data["seed"] == heldout
        train = ~test & data["sampled"]
        folds.append(dict(heldout_seed=int(heldout), training_seeds=seeds[seeds != heldout].tolist(),
                          training_frames=int(train.sum()), test_frames=int(test.sum())))
        for black in (False, True):
            fit_rows, predict_rows = train & (data["black"] == black), test & (data["black"] == black)
            x, y = data["x"][fit_rows, :dimensions], data["y"][fit_rows]
            median = np.median(x, axis=0)
            iqr = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
            iqr[iqr < 1e-9] = 1
            z = (x - median) / iqr
            regression = Ridge(alpha=.01).fit(np.c_[z, z * z], y)
            tx = data["x"][predict_rows, :dimensions]
            tz = (tx - median) / iqr
            prediction[predict_rows] = regression.predict(np.c_[tz, tz * tz])
            lower, upper = x.min(axis=0) - .05 * np.ptp(x, axis=0), x.max(axis=0) + .05 * np.ptp(x, axis=0)
            lower[:4] = np.maximum(0, lower[:4])
            ood[predict_rows] = ((tx < lower) | (tx > upper)).any(axis=1)
    depth_no_contact = (data["x"][:, 0] < .005) & (data["x"][:, 1:4] == 0).all(axis=1)
    available = ~ood & ~depth_no_contact & np.isfinite(prediction) & (prediction >= .05)
    contact, probe = data["y"] >= .05, ~data["expert"]
    masks = dict(contact=contact, expert_contact=contact & ~probe,
                 probe_contact=contact & probe, initial=data["initial"],
                 probe_initial=data["initial"] & probe)
    for color, black in (("white", False), ("black", True)):
        for dynamic, heavy in (("light", False), ("heavy", True)):
            group = probe & (data["black"] == black) & (data["heavy"] == heavy)
            masks[f"probe_{color}_{dynamic}_contact"] = group & contact
            masks[f"probe_{color}_{dynamic}_initial"] = group & data["initial"]
    for seed in seeds:
        masks[f"seed_{seed}_probe_initial"] = (data["seed"] == seed) & probe & data["initial"]
    report = {}
    for name, mask in masks.items():
        use = mask & available
        report[name] = dict(total_frames=int(mask.sum()), ood_frames=int((mask & ood).sum()),
                            unavailable_frames=int((mask & ~available).sum()),
                            raw_unclipped_regression_error=metrics(data["y"][mask], prediction[mask]),
                            accepted_error=metrics(data["y"][use], prediction[use]))
    target = np.where(data["black"], np.where(data["heavy"], 1., .7), np.where(data["heavy"], .5, .35))
    upper = target + np.where(data["black"], .14, .07)
    over = data["y"] > upper
    report["safety"] = dict(no_contact_frames=int((~contact).sum()),
        no_contact_false_positive_frames=int((~contact & available & (prediction >= .1)).sum()),
        overforce_frames=int(over.sum()), overforce_unavailable_frames=int((over & ~available).sum()),
        overforce_underestimated_frames=int((over & available & (prediction <= upper)).sum()))
    return dict(dimensions=dimensions, folds=folds, metrics=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_audit(args.audit)
    report = dict(schema="shadow_digit_paired_seed_pilot_cv.v1", private_labels=True,
        exploratory_only=True, frozen_runtime_model_changed=False, force_features_include_robot_state=False,
        split_unit="seed including both expert and probe episodes", ridge_alpha=.01,
        same_history_valid_rows_for_4d_and_5d=True,
        excluded_history_frames="First two frames and partial final frames with no exact 8-tick history",
        variants={f"{d}d": evaluate(data, d) for d in (4, 5)})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    for key, variant in report["variants"].items():
        print(key, json.dumps({k: variant["metrics"][k] for k in ("probe_contact", "probe_initial")}))


if __name__ == "__main__":
    main()
