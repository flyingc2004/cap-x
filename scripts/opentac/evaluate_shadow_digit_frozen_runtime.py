"""Post-run private audit or counterfactual replay of a frozen DIGIT model.

Replay changes neither generated actions nor the historical native score.
No private labels are used to compute a prediction or change model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from audit_shadow_digit_recalibration import metrics, longest_run
from audit_shadow_digit_runtime import jsonl, read_run


def replay(row, model):
    diagnostic = row["depth_diagnostics"]
    features = diagnostic.get("features")
    if features is None or len(features) != 5:
        return "unavailable", None
    if diagnostic["sensor_sha256"] != model["sensor_sha256"]:
        raise ValueError("Sensor mismatch in saved public features")
    x = np.asarray(features, float)
    group = model["models"][diagnostic["color"]]
    if not np.isfinite(x).all():
        return "unavailable", None
    if ((x < group["feature_min"]) | (x > group["feature_max"])).any():
        return "ood", None
    if x[0] < .005 and (x[1:4] == 0).all():
        return "no_contact", None
    z = (x - group["median"]) / group["iqr"]
    force = float(np.r_[z, z*z] @ group["coef"] + group["intercept"])
    if not np.isfinite(force):
        return "invalid_prediction", None
    return ("no_contact", None) if force < .05 else ("ok", force)


def audit(run, model, output):
    base, rows = read_run(run)
    private = Path(base["private_audit"])
    trace = jsonl(private / "shadow_piano_physics_trace.jsonl")
    native = json.loads((private / "shadow_piano_native_audit_summary.json").read_text(encoding="utf-8"))
    predicted = [(replay(row, model) if model else
                  (row["estimate"]["quality"], row["estimate"]["estimated_key_force_N"])) for row in rows]
    valid = np.array([q == "ok" and p is not None for q, p in predicted])
    truth = np.array([r["private_key_top_force_N"] for r in rows])
    estimates = np.array([np.nan if p is None else p for _, p in predicted])
    contact = truth >= .05
    quality = Counter(q for q, _ in predicted)
    report = dict(schema="shadow_digit_frozen_runtime_audit.private.v1", private_labels=True,
        mode="counterfactual_saved_features" if model else "actual_runtime_estimates",
        run=base, calibration_versions=sorted({r["estimate"]["calibration_version"] for r in rows}),
        candidate_version=model["calibration_version"] if model else None,
        quality_counts=dict(quality), accepted_error=metrics(truth[valid], estimates[valid]),
        accepted_contact_error=metrics(truth[valid & contact], estimates[valid & contact]),
        unavailable_contact_queries=int((contact & ~valid).sum()), contact_queries=int(contact.sum()),
        no_contact_false_positives_at_0p1N=int((~contact & valid & (estimates >= .1)).sum()),
        old_ood_replay=[], per_note=[])
    for i, row in enumerate(rows):
        if model and row["estimate"]["quality"] == "ood":
            report["old_ood_replay"].append(dict(note=row["note"], physics_step=row["physics_step"],
                quality=predicted[i][0], estimate_N=predicted[i][1], true_N=truth[i],
                error_N=float(estimates[i]-truth[i]) if valid[i] else None))
    os.environ.setdefault("MPLCONFIGDIR", str(output / "mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
    for index, note in enumerate(native["notes"]):
        physical = [r for r in trace if r["target_index"] == index]
        if not physical:
            continue
        lo, hi = physical[0]["physics_step"], physical[-1]["physics_step"]
        selected = np.array([lo <= r["physics_step"] <= hi for r in rows])
        y = np.array([r["actual_force_N"] if r["actual_midi"] == note["requested_midi"] else 0 for r in physical])
        target = float(physical[0]["target_force_N"])
        tolerance = .14 if note["requested_midi"] % 12 in (1, 3, 6, 8, 10) else .07
        report["per_note"].append(dict(index=index, midi=note["requested_midi"],
            dynamic=note["requested_dynamic"], native_scored=note["native_scored_note"],
            native_failure=note["native_failed_reason"], peak_target_force_N=float(y.max()),
            longest_force_band_ticks=longest_run(np.abs(y-target) <= tolerance),
            band_scope="Force only; native scoring also requires angle, release and event order",
            other_key_force_ticks_in_native_target_interval=sum(r["actual_force_N"] >= .05 and r["actual_midi"] != note["requested_midi"] for r in physical),
            interval_caveat="Native target may advance during release; other-key counts can include the preceding note, not a wrong-key failure",
            estimate_error=metrics(truth[selected & valid], estimates[selected & valid]),
            quality_counts=dict(Counter(predicted[i][0] for i in np.flatnonzero(selected)))))
        ax = axes.flat[index]
        ax.plot([(r["physics_step"]-lo)/240 for r in physical], y, lw=.8, label="Private target-key force / 240 Hz")
        pick = selected & valid
        ax.scatter([(rows[i]["physics_step"]-lo)/240 for i in np.flatnonzero(pick)], estimates[pick], s=8, c="orange", label="Public estimate / queried frames")
        ax.axhspan(target-tolerance, target+tolerance, alpha=.12, color="green")
        ax.set(title=f"{index+1}: MIDI {note['requested_midi']} {note['requested_dynamic']} scored={note['native_scored_note']}", xlabel="Seconds in attempt", ylabel="N")
        ax.grid(alpha=.2)
        if not note["native_scored_note"] and pick.any():
            end = rows[np.flatnonzero(pick)[-1]]["physics_step"]
            start = max(lo, end - 60)
            local = [r for r in physical if start <= r["physics_step"] <= end]
            points = [i for i in np.flatnonzero(pick) if rows[i]["physics_step"] >= start]
            zoom, zx = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
            zx.plot([(r["physics_step"]-start)/240 for r in local],
                    [r["actual_force_N"] if r["actual_midi"] == note["requested_midi"] else 0 for r in local],
                    ".-", ms=3, lw=.8, label="Private physical force / 240 Hz")
            zx.scatter([(rows[i]["physics_step"]-start)/240 for i in points], estimates[points],
                       color="orange", s=20, label="Public sampled force estimate / 60 Hz")
            zx.axhspan(target-tolerance, target+tolerance, color="green", alpha=.12)
            zx.axhline(target-tolerance, color="green", ls="--", lw=.8, label="Native force band")
            zx.axhline(target+tolerance, color="green", ls="--", lw=.8)
            zx.set(title=f"Failed note {index+1}: MIDI {note['requested_midi']} {note['requested_dynamic']} / final 0.25 s before release",
                   xlabel="Seconds", ylabel="N")
            zx.legend(fontsize=8)
            zx.grid(alpha=.2)
            zoom.savefig(output / f"note_{index+1:02d}_force_zoom.private.png", dpi=160)
            plt.close(zoom)
    axes.flat[0].legend(fontsize=7)
    fig.savefig(output / "force_alignment.private.png", dpi=150)
    plt.close(fig)
    (output / "report.private.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Optional counterfactual model; omit to audit actual runtime")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    model = json.loads(args.model.read_text(encoding="utf-8")) if args.model else None
    report = audit(args.run_dir, model, args.output)
    if args.model:
        report["candidate_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
        (args.output / "report.private.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("mode", "quality_counts", "accepted_contact_error", "unavailable_contact_queries")}))
    print("Private audit:", args.output)


if __name__ == "__main__":
    main()
