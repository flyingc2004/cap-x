#!/usr/bin/env python
"""Summarize lift-can tactile code memory evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from capx.memory.tactile_code import parse_trial_dir


def main() -> None:
    args = _parser().parse_args()
    root = Path(args.run_root).expanduser().resolve()
    rows = []
    for trial_dir in sorted(root.glob("*/*trial_*")) + sorted(root.glob("trial_*")):
        if not trial_dir.is_dir():
            continue
        example = parse_trial_dir(trial_dir)
        if example is None:
            continue
        rows.append(_trial_metrics(root, trial_dir, example))

    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)

    summary = {
        group: _aggregate(group_rows)
        for group, group_rows in sorted(groups.items())
    }
    output_json = Path(args.output_json) if args.output_json else root / "tactile_code_memory_eval_summary.json"
    output_csv = Path(args.output_csv) if args.output_csv else root / "tactile_code_memory_eval_trials.csv"
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[tactile-code-memory] summary={output_json}")
    print(f"[tactile-code-memory] trials_csv={output_csv}")
    for group, item in summary.items():
        print(
            f"[tactile-code-memory] {group}: "
            f"trials={item['trials']} completed={item['task_completed']} "
            f"avg_reward={item['avg_reward']:.3f} "
            f"memory_trigger_success={item['memory_trigger_success']}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser


def _trial_metrics(root: Path, trial_dir: Path, example) -> dict[str, Any]:
    rel_parent = trial_dir.parent.relative_to(root) if trial_dir.parent != root else Path(".")
    group = str(rel_parent).split("/", maxsplit=1)[0]
    trace = example.tactile_trace
    memory_trace = example.memory_trace
    event_counts = {}
    for row in trace:
        event = str(row.get("event", "unknown"))
        event_counts[event] = event_counts.get(event, 0) + 1
    summary = example.summary_text
    code_blocks = _int_match(summary, r"Num Code Blocks:\s*(\d+)")
    regenerations = _int_match(summary, r"Num Regenerations:\s*(\d+)")
    finishes = _int_match(summary, r"Num Finishes:\s*(\d+)")
    close_ops = sum(1 for row in trace if row.get("operation") == "close" and row.get("iteration") == 0)
    runtime_hits = [
        entry
        for entry in memory_trace
        if entry.get("kind") == "runtime_retrieval" and entry.get("matches")
    ]
    return {
        "group": group,
        "trial_dir": str(trial_dir),
        "trial_index": example.trial_index,
        "sandbox_rc": example.sandbox_rc,
        "reward": example.reward,
        "task_completed": int(example.task_completed),
        "stable_grasp_seen": int(event_counts.get("stable_grasp", 0) > 0),
        "contact_lost_count": event_counts.get("contact_lost", 0),
        "slip_detected_count": event_counts.get("slip_detected", 0),
        "one_hand_contact_count": event_counts.get("one_hand_contact", 0),
        "close_operation_count": close_ops,
        "regrasp_count": max(0, close_ops - 1),
        "memory_trigger_count": len(runtime_hits),
        "memory_trigger_success": int(bool(runtime_hits) and example.task_completed),
        "code_blocks": code_blocks,
        "regenerations": regenerations,
        "finishes": finishes,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(rows))
    return {
        "trials": len(rows),
        "task_completed": sum(int(row["task_completed"]) for row in rows),
        "success_rate": sum(int(row["sandbox_rc"] == 0) for row in rows) / n,
        "avg_reward": sum(float(row["reward"]) for row in rows) / n,
        "stable_initial_grasp_rate": sum(int(row["stable_grasp_seen"]) for row in rows) / n,
        "contact_lost_total": sum(int(row["contact_lost_count"]) for row in rows),
        "slip_detected_total": sum(int(row["slip_detected_count"]) for row in rows),
        "regrasp_total": sum(int(row["regrasp_count"]) for row in rows),
        "memory_trigger_total": sum(int(row["memory_trigger_count"]) for row in rows),
        "memory_trigger_success": sum(int(row["memory_trigger_success"]) for row in rows),
        "avg_code_blocks": sum(int(row["code_blocks"]) for row in rows) / n,
        "avg_regenerations": sum(int(row["regenerations"]) for row in rows) / n,
        "avg_finishes": sum(int(row["finishes"]) for row in rows) / n,
    }


def _int_match(text: str, pattern: str) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else 0


if __name__ == "__main__":
    main()
