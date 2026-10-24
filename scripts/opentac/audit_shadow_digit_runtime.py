"""Offline/private audit of DIGIT runtime quality and expert feature coverage.

Reads completed run artifacts only. Labels are used for post-run plots and
metrics, never to change calibration, prompts, or action decisions.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
import re

import numpy as np


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_run(path):
    location = json.loads((path / "shadow_private_audit_location.json").read_text())
    private = Path(location["private_audit_dir"])
    summary = json.loads((private / "shadow_piano_native_audit_summary.json").read_text())
    rows = jsonl(private / "shadow_force_estimates_private_audit.jsonl")
    events = jsonl(private / "shadow_native_events.jsonl")
    log = (path / "run.log").read_text(encoding="utf-8", errors="replace")
    actions = sum(row["kind"] == "action" for row in events)
    clear = re.findall(r"keyboard clear at action_step=(\d+)", log)
    exits = re.findall(r"\[run\] rep 1 exit_code=(\d+)", log)
    accepted = [r for r in rows if r["estimate"]["quality"] == "ok"]
    error = np.array([r["estimate"]["estimated_key_force_N"] - r["private_key_top_force_N"]
                      for r in accepted])
    quality = Counter(r["estimate"]["quality"] for r in rows)
    ood = [r for r in rows if r["estimate"]["quality"] == "ood"]
    bounds = Counter(b["feature"] for r in ood for b in r["depth_diagnostics"]["exceeded_bounds"])
    result = dict(run=path.name, private_audit=str(private),
        native_correct=summary["native_notes_completed"],
        attempted=summary.get("native_notes_attempted"), end=summary["end_submitted"],
        exit_code=int(exits[-1]) if exits else None,
        native_actions=actions, generated_actions=actions-int(clear[-1]) if clear else None,
        queries=len(rows), quality_counts=dict(quality), ood_by_feature=dict(bounds),
        accepted_mae_N=float(np.abs(error).mean()) if error.size else None,
        accepted_samples=int(error.size),
        accepted_error_scope="Queried fresh ok samples only; not contact/hold or full-episode accuracy",
        ood_rows=[dict(note=r["note"], action_step=r["action_step"],
            private_force_N=r["private_key_top_force_N"],
            features=r["depth_diagnostics"]["features"],
            exceeded_bounds=r["depth_diagnostics"]["exceeded_bounds"]) for r in ood])
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--expert-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Never overwrite an existing analysis directory or change run permissions.
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    results, runtime = [], []
    for path in args.run_dir:
        result, rows = read_run(path)
        results.append(result)
        runtime.append(rows)
    expert = []
    for seed in range(3100, 3150):
        with np.load(args.expert_cache / f"{seed}.private.npz") as cache:
            expert.append({k: cache[k] for k in ("x", "y", "black")})
    data = {k: np.concatenate([e[k] for e in expert]) for k in expert[0]}
    report = dict(schema="shadow_digit_runtime_private_audit.v1", private_labels=True,
                  runtime_model_changed=False, expert_frames=len(data["y"]), runs=results)
    (args.output / "report.private.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / "matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    markers = ("x", "+", "1", "2")
    for column, (color, black) in enumerate((("white", False), ("black", True))):
        use = data["black"] == black
        for row, (dimension, name) in enumerate(((0, "Mean depth change (mm)"), (4, "Area change rate (fraction/s)"))):
            ax = axes[row, column]
            ax.scatter(data["y"][use], data["x"][use, dimension], s=3, alpha=.12,
                       color="gray", label="50 expert episodes (30 Hz)", rasterized=True)
            for index, rows in enumerate(runtime):
                selected = [r for r in rows if r["depth_diagnostics"].get("color") == color
                            and r["depth_diagnostics"].get("features") is not None
                            and r["estimate"]["quality"] in ("ok", "ood")]
                if selected:
                    ax.scatter([r["private_key_top_force_N"] for r in selected],
                               [r["depth_diagnostics"]["features"][dimension] for r in selected],
                               s=45, marker=markers[index % len(markers)], label=f"Runtime {index+1} ok/OOD")
            limit = next((r["depth_diagnostics"]["feature_max"][dimension]
                          for rows in runtime for r in rows
                          if r["depth_diagnostics"].get("color") == color
                          and "feature_max" in r["depth_diagnostics"]), None)
            if limit is not None:
                ax.axhline(limit, color="red", linestyle="--", label="Frozen OOD upper bound")
            ax.set(title=f"{color} keys", xlabel="Private same-step key-top force (N)", ylabel=name)
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Offline audit only: expert versus CaP-X feature coverage")
    fig.savefig(args.output / "runtime_expert_coverage.png", dpi=160)
    plt.close(fig)
    for r in results:
        print(json.dumps({k: r[k] for k in ("run", "native_correct", "attempted", "end",
            "exit_code", "native_actions", "generated_actions", "quality_counts", "accepted_mae_N")}))
    print("Private audit output:", args.output)


if __name__ == "__main__":
    main()
