#!/usr/bin/env python
"""Compile tactile strategy memory from saved CaP-X run directories."""

from __future__ import annotations

import argparse

from capx.integrations.tactile.strategy_memory import (
    compile_memory_from_run_dir,
    resolve_memory_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory or outputs directory to scan.")
    parser.add_argument(
        "--memory-path",
        default=None,
        help="JSONL memory path. Defaults to .capx_tactile_strategies.jsonl.",
    )
    parser.add_argument("--task", default="cube_stack", help="Task label for memory records.")
    parser.add_argument("--target", default="red cube", help="Target label for memory records.")
    args = parser.parse_args()

    added = compile_memory_from_run_dir(
        args.run_dir,
        memory_path=args.memory_path,
        task=args.task,
        target=args.target,
    )
    print(f"[tactile-memory] Added {len(added)} record(s)")
    print(f"[tactile-memory] Memory: {resolve_memory_path(args.memory_path)}")
    for record in added:
        print(
            "[tactile-memory] "
            f"{record.outcome} {record.failure_type} reward={record.reward:.3f} "
            f"source={record.source_trial_dir}"
        )


if __name__ == "__main__":
    main()
