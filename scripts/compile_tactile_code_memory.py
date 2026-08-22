#!/usr/bin/env python
"""Compile tactile code memory candidates and validated bank records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from capx.memory.tactile_code import (
    DEFAULT_CANDIDATE_PATH,
    DEFAULT_MEMORY_PATH,
    TactileCodeMemoryBank,
    build_candidates_from_examples,
    promote_candidates_with_validation,
    resolve_memory_path,
    scan_trial_examples,
    write_jsonl,
)


def main() -> None:
    args = _parser().parse_args()
    source_run = Path(args.source_run).expanduser().resolve()
    examples = scan_trial_examples(source_run, start=args.train_start, end=args.train_end)
    candidates = build_candidates_from_examples(examples)

    if args.repair_source in {"auto", "llm"}:
        candidates = _maybe_fill_llm_sketches(candidates, force=args.repair_source == "llm")

    candidate_path = resolve_memory_path(args.candidate_output)
    write_jsonl(candidate_path, [record.to_dict() for record in candidates])

    manifest = {
        "source_run": str(source_run),
        "train_range": [args.train_start, args.train_end],
        "candidate_output": str(candidate_path),
        "candidate_count": len(candidates),
        "validation_range": [args.validation_start, args.validation_end],
        "validation_plan": [
            {
                "memory_id": record.id,
                "diagnosis": record.diagnosis,
                "status": record.status,
                "applicable_count": record.evidence.get("applicable_count", 0),
                "strategy_tags": record.strategy_tags,
                "when_to_apply": record.skill.get("when_to_apply", ""),
                "repair_strategy": record.repair_strategy,
            }
            for record in candidates
        ],
        "notes": [
            "Candidates are not formal memory until promoted by simulator validation.",
            "Run validation seeds with a minimal prompt and candidate-path retrieval, then re-run this script with --validation-run.",
        ],
    }
    manifest_path = resolve_memory_path(args.validation_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    promoted = []
    if args.validation_run:
        validation_examples = scan_trial_examples(
            Path(args.validation_run).expanduser().resolve(),
            start=args.validation_start,
            end=args.validation_end,
        )
        promoted = promote_candidates_with_validation(
            [record.to_dict() for record in candidates],
            validation_examples,
            min_applicable=args.min_applicable,
            min_recovery=args.min_recovery,
        )
        bank = TactileCodeMemoryBank(args.bank_output)
        bank.write(promoted)

    print(
        "[tactile-code-memory] "
        f"examples={len(examples)} candidates={len(candidates)} "
        f"candidate_path={candidate_path}"
    )
    print(f"[tactile-code-memory] validation_manifest={manifest_path}")
    if args.validation_run:
        print(
            "[tactile-code-memory] "
            f"promoted={len(promoted)} bank_path={resolve_memory_path(args.bank_output)}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="Existing CaP-X run directory.")
    parser.add_argument("--train-start", type=int, default=1)
    parser.add_argument("--train-end", type=int, default=80)
    parser.add_argument("--validation-start", type=int, default=81)
    parser.add_argument("--validation-end", type=int, default=100)
    parser.add_argument("--validation-run", default=None)
    parser.add_argument("--candidate-output", default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--bank-output", default=DEFAULT_MEMORY_PATH)
    parser.add_argument(
        "--validation-manifest",
        default=".capx_tactile_code_memory/lift_can_v1/validation_manifest.json",
    )
    parser.add_argument("--min-applicable", type=int, default=3)
    parser.add_argument("--min-recovery", type=int, default=2)
    parser.add_argument(
        "--repair-source",
        choices=["template", "auto", "llm"],
        default="template",
        help="template is deterministic; llm/auto can replace code_sketch when a server is configured.",
    )
    return parser


def _maybe_fill_llm_sketches(records, *, force: bool):
    server_url = os.getenv("CAPX_SERVER_URL")
    api_key = os.getenv("CAPX_API_KEY")
    model = os.getenv("CAPX_MODEL", "gpt-4o")
    if not server_url:
        if force:
            raise RuntimeError("CAPX_SERVER_URL is required for --repair-source llm")
        return records
    try:
        from capx.llm.client import ModelQueryArgs, query_model
    except Exception:
        if force:
            raise
        return records

    query_args = ModelQueryArgs(
        model=model,
        server_url=server_url,
        api_key=api_key,
        temperature=0.0,
        max_tokens=2048,
    )
    updated = []
    for record in records:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Generate a short parameterized Python partial-code sketch for tactile "
                    "grasp repair. Use only public APIs: get_object_pose, sample_grasp_pose, "
                    "goto_pose, open_gripper, close_gripper, home_pose, get_tactile_summary, "
                    "is_contacting, is_slipping, is_grasp_stable. Do not use reward, success, "
                    "trial id, private state, fixed coordinates, or task labels."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "diagnosis": record.diagnosis,
                        "failure_signature": record.failure_signature,
                        "repair_strategy": record.repair_strategy,
                    },
                    sort_keys=True,
                ),
            },
        ]
        out = query_model(query_args, prompt)
        sketch = _extract_code(str(out.get("content", ""))).strip()
        if sketch:
            record.code_sketch = sketch
            record.skill = dict(record.skill or {})
            record.skill["code_sketch"] = sketch
        updated.append(record)
    return updated


def _extract_code(text: str) -> str:
    if "```python" in text:
        text = text.split("```python", maxsplit=1)[1]
        text = text.split("```", maxsplit=1)[0]
    elif "```" in text:
        text = text.split("```", maxsplit=1)[1]
        text = text.split("```", maxsplit=1)[0]
    return text.strip()


if __name__ == "__main__":
    main()
