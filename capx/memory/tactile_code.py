"""Generic tactile code memory for CaP-X.

The records here are partial repair skills keyed by tactile failure signatures.
They deliberately avoid task answers, fixed poses, reward values, trial ids in
prompt-facing text, and any simulator-private fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_MEMORY_PATH = ".capx_tactile_code_memory/lift_can_v1/bank.jsonl"
DEFAULT_CANDIDATE_PATH = ".capx_tactile_code_memory/lift_can_v1/candidates.jsonl"
TACTILE_EVENTS = {
    "no_contact",
    "one_hand_contact",
    "stable_grasp",
    "slip_detected",
    "contact_lost",
    "unknown",
}
NON_TACTILE_ERROR_PATTERNS = (
    "Traceback",
    "SyntaxError",
    "NameError",
    "KeyError",
    "RuntimeError",
    "RGB-D detection has only 0 valid depth points",
    "planning failed",
    "motion planning failed",
    "perception failure",
)


@dataclass(slots=True)
class TactileCodeMemoryRecord:
    """One tactile partial-code memory record."""

    id: str
    version: int
    capability: str
    failure_signature: dict[str, Any]
    applicability: dict[str, Any]
    activation_rule: dict[str, Any]
    diagnosis: str
    repair_strategy: str
    code_sketch: str
    trace_summary: dict[str, Any]
    strategy_tags: dict[str, str]
    skill: dict[str, Any]
    evidence: dict[str, Any]
    status: str = "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrialTactileExample:
    """Parsed tactile evidence from a saved CaP-X trial directory."""

    trial_dir: Path
    trial_index: int | None
    reward: float
    task_completed: bool
    sandbox_rc: int
    tactile_trace: list[dict[str, Any]]
    pre_move_timeline: list[dict[str, Any]]
    debug_records: list[dict[str, Any]]
    memory_trace: list[dict[str, Any]]
    code_text: str
    summary_text: str

    @property
    def outcome(self) -> str:
        return "success" if self.task_completed or self.reward >= 1.0 else "failure"


def project_root() -> Path:
    """Return the local cap-x repository root."""
    return Path(__file__).resolve().parents[2]


def resolve_memory_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a memory path relative to the cap-x repository root."""
    raw = path or os.getenv("CAPX_TACTILE_CODE_MEMORY_PATH") or DEFAULT_MEMORY_PATH
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load valid JSONL records from disk."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def write_jsonl(path: str | os.PathLike[str], records: list[dict[str, Any]]) -> None:
    """Write JSONL records with deterministic key order."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


class TactileCodeMemoryBank:
    """Read-only/rebuildable JSONL bank for tactile partial code memory."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = resolve_memory_path(path)

    def load(self, *, statuses: list[str] | None = None) -> list[dict[str, Any]]:
        records = load_jsonl(self.path)
        if statuses is None:
            return records
        allowed = {str(status) for status in statuses}
        return [record for record in records if str(record.get("status", "")) in allowed]

    def write(self, records: list[TactileCodeMemoryRecord | dict[str, Any]]) -> None:
        write_jsonl(
            self.path,
            [
                record.to_dict() if isinstance(record, TactileCodeMemoryRecord) else dict(record)
                for record in records
            ],
        )

    def retrieve(
        self,
        *,
        capability: str = "grasp_stabilization",
        current_signature: dict[str, Any] | None = None,
        required_apis: list[str] | None = None,
        top_k: int = 3,
        statuses: list[str] | None = None,
        include_scores: bool = False,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        records = self.load(statuses=statuses)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for record in records:
            score = score_memory_record(
                record,
                capability=capability,
                current_signature=current_signature,
                required_apis=required_apis,
            )
            if score <= 0.0:
                continue
            item = dict(record)
            if include_scores:
                item["_score"] = round(score, 4)
            scored.append((score, str(record.get("id", "")), item))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item for _, _, item in scored[: int(top_k)]]


def parse_trial_dir(trial_dir: str | os.PathLike[str]) -> TrialTactileExample | None:
    """Parse one saved trial directory into tactile memory evidence."""
    path = Path(trial_dir)
    if not path.is_dir():
        return None
    trace = _read_json_list(path / "tactile_gripper_trace.json")
    pre_move = _read_json_list(path / "pre_move_tactile_timeline.json")
    if not trace and not pre_move:
        return None
    summary_text = _read_text(path / "summary.txt")
    code_text = _read_text(path / "code.py")
    debug_payload = _read_json(path / "univtac_debug.json", default={})
    debug_records = debug_payload.get("records", []) if isinstance(debug_payload, dict) else []
    memory_trace = _read_json_list(path / "tactile_code_memory_trace.json")
    reward, task_completed, sandbox_rc = read_trial_outcome(path, summary_text)
    return TrialTactileExample(
        trial_dir=path,
        trial_index=parse_trial_index(path.name),
        reward=reward,
        task_completed=task_completed,
        sandbox_rc=sandbox_rc,
        tactile_trace=trace,
        pre_move_timeline=pre_move,
        debug_records=debug_records if isinstance(debug_records, list) else [],
        memory_trace=memory_trace,
        code_text=code_text,
        summary_text=summary_text,
    )


def scan_trial_examples(
    run_dir: str | os.PathLike[str],
    *,
    start: int | None = None,
    end: int | None = None,
) -> list[TrialTactileExample]:
    """Scan a run directory for trial evidence, optionally filtering by index."""
    root = Path(run_dir)
    examples: list[TrialTactileExample] = []
    for trial_dir in sorted(root.glob("trial_*")):
        example = parse_trial_dir(trial_dir)
        if example is None:
            continue
        if example.trial_index is not None:
            if start is not None and example.trial_index < start:
                continue
            if end is not None and example.trial_index > end:
                continue
        examples.append(example)
    examples.sort(key=lambda item: (-1 if item.trial_index is None else item.trial_index))
    return examples


def build_signature_from_trace(
    trace: list[dict[str, Any]],
    *,
    phase: str = "grasp_or_lift",
    window: int = 20,
    debug_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a task-agnostic tactile failure signature from gripper trace rows."""
    rows = [row for row in trace if isinstance(row, dict)]
    recent = rows[-max(1, int(window)) :]
    events = event_sequence(rows)
    counts = dict(Counter(str(row.get("event", "unknown")) for row in rows))
    feature_ranges = {
        "normal_force": _range(rows, "normal_force"),
        "left_normal_force": _range(rows, "left_normal_force"),
        "right_normal_force": _range(rows, "right_normal_force"),
        "abs_contact_balance": _range_abs(rows, "contact_balance"),
        "slip_score": _range(rows, "slip_score"),
        "width": _range(rows, "width"),
    }
    feature_trends = {
        "normal_force_delta_recent": _delta(recent, "normal_force"),
        "slip_delta_recent": _delta(recent, "slip_score"),
        "contact_balance_final": _last_float(rows, "contact_balance", 0.0),
        "left_contact_ratio": _bool_ratio(rows, "left_contact"),
        "right_contact_ratio": _bool_ratio(rows, "right_contact"),
        "bilateral_contact_ratio": _bilateral_ratio(rows),
        "contact_lost_after_contact": _has_contact_loss_after_contact(events),
        "stable_then_lost": _has_ordered_subsequence(events, ["stable_grasp", "contact_lost"]),
        "stable_seen": "stable_grasp" in events,
    }
    object_motion = _object_motion_features(debug_records or [])
    if object_motion:
        feature_trends.update(object_motion)
    return {
        "phase": phase,
        "event_sequence": events,
        "event_counts": counts,
        "feature_ranges": feature_ranges,
        "feature_trends": feature_trends,
        "window": int(window),
    }


def build_current_signature_from_env(
    env: Any,
    *,
    window: int = 20,
) -> dict[str, Any] | None:
    """Build the current runtime signature from a CaP-X env if tactile trace exists."""
    low_level = getattr(env, "low_level_env", env)
    trace = getattr(low_level, "_tactile_gripper_trace", None)
    debug_records = getattr(low_level, "_debug_records", None)
    if isinstance(trace, list) and trace:
        return build_signature_from_trace(
            trace[-max(1, int(window)) :],
            phase="grasp_or_lift",
            window=window,
            debug_records=debug_records if isinstance(debug_records, list) else None,
        )

    buffer = getattr(low_level, "tactile_buffer", None)
    if buffer is None or not hasattr(buffer, "frames"):
        return None
    try:
        from capx.integrations.univtac.native_tactile import (
            summarize_native_tactile,
            tactile_event_sequence,
        )
    except Exception:
        return None
    frames = buffer.recent(window) if hasattr(buffer, "recent") else buffer.frames()[-window:]
    if not frames:
        return None
    calibration_fn = getattr(low_level, "get_native_tactile_calibration", None)
    calibration = calibration_fn() if callable(calibration_fn) else {}
    rows: list[dict[str, Any]] = []
    for idx in range(1, len(frames) + 1):
        summary = summarize_native_tactile(frames[:idx], hand="both", **calibration)
        rows.append(
            {
                "event": summary.get("event", "unknown"),
                "left_contact": summary.get("left_contact", False),
                "right_contact": summary.get("right_contact", False),
                "normal_force": summary.get("normal_force", 0.0),
                "left_normal_force": summary.get("left", {}).get("normal_force", 0.0),
                "right_normal_force": summary.get("right", {}).get("normal_force", 0.0),
                "contact_balance": summary.get("contact_balance", 0.0),
                "slip_score": summary.get("slip_score", 0.0),
            }
        )
    signature = build_signature_from_trace(rows, phase="grasp_or_lift", window=window)
    signature["event_sequence"] = tactile_event_sequence(frames, hand="both", **calibration)
    return signature


def code_strategy_tags(code_text: str) -> dict[str, str]:
    """Extract a compact, non-clustering code strategy signature."""
    code = str(code_text or "")
    uses_close = "close_gripper" in code
    uses_sample = "sample_grasp_pose" in code
    uses_pose = "get_object_pose" in code
    uses_perception = uses_sample or uses_pose
    if "is_grasp_stable" in code:
        grasp_verification = "tactile_recheck"
    elif uses_close and re.search(r"\bstable\b", code):
        grasp_verification = "close_stable_only"
    elif uses_close:
        grasp_verification = "close_without_explicit_stable_check"
    else:
        grasp_verification = "unknown"

    if uses_sample:
        recovery_source = "vision_regrasp"
    elif uses_pose:
        recovery_source = "object_pose_regrasp"
    else:
        recovery_source = "none"

    if uses_perception:
        perception_guard = (
            "guarded"
            if re.search(r"\btry\s*:", code) and re.search(r"\bexcept\b", code)
            else "unguarded"
        )
    else:
        perception_guard = "none"

    return {
        "grasp_verification": grasp_verification,
        "recovery_source": recovery_source,
        "perception_guard": perception_guard,
    }


def build_trace_summary(examples: list[TrialTactileExample]) -> dict[str, Any]:
    """Summarize public trace sources without exposing seeds or rewards."""
    trace_types: set[str] = set()
    strategy_counts: dict[str, Counter] = {
        "grasp_verification": Counter(),
        "recovery_source": Counter(),
        "perception_guard": Counter(),
    }
    for example in examples:
        if example.tactile_trace:
            trace_types.add("tactile_gripper_trace")
        if example.pre_move_timeline:
            trace_types.add("pre_move_timeline")
        if example.debug_records:
            trace_types.add("debug_records")
        if example.memory_trace:
            trace_types.add("tactile_code_memory_trace")
        if example.code_text:
            trace_types.add("code")
        if example.summary_text:
            trace_types.add("summary")
        for key, value in code_strategy_tags(example.code_text).items():
            strategy_counts[key][value] += 1
    return {
        "source_count": len(examples),
        "source_trace_types": sorted(trace_types),
        "observed_strategy_counts": {
            key: dict(counter) for key, counter in strategy_counts.items()
        },
        "public_only": True,
    }


def skill_strategy_tags(examples: list[TrialTactileExample]) -> dict[str, str]:
    """Return the compact strategy tags used for retrieval display/ranking."""
    observed = [code_strategy_tags(example.code_text) for example in examples]
    return {
        "grasp_verification": _dominant_tag(observed, "grasp_verification", "tactile_recheck"),
        # Lift-can v1 validates tactile local repair from the official pre-grasp,
        # even if historical code often called sample_grasp_pose for re-grasp.
        "recovery_source": "pregrasp_local_adjustment",
        "perception_guard": _dominant_tag(observed, "perception_guard", "none"),
    }


def build_skill_card(
    *,
    diagnosis: str,
    signature: dict[str, Any],
    activation_rule: dict[str, Any],
    repair_strategy: str,
    code_sketch: str,
) -> dict[str, Any]:
    """Build an ASPIRE-style validated partial-code skill card."""
    rules = list(activation_rule.get("all", [])) + list(activation_rule.get("any", []))
    when_to_apply = "; ".join(str(rule) for rule in rules if str(rule).strip())
    if not when_to_apply:
        when_to_apply = f"when tactile signature resembles {diagnosis}"
    return {
        "when_to_apply": when_to_apply,
        "diagnosis": diagnosis,
        "repair_strategy": repair_strategy,
        "code_sketch": code_sketch,
        "signature_brief": signature_brief(signature),
    }


def _dominant_tag(items: list[dict[str, str]], key: str, default: str) -> str:
    counter = Counter(str(item.get(key, default)) for item in items if item.get(key))
    if not counter:
        return default
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_candidates_from_examples(
    examples: list[TrialTactileExample],
    *,
    capability: str = "grasp_stabilization",
    validated: bool = False,
) -> list[TactileCodeMemoryRecord]:
    """Distill candidate memory records from parsed trial examples."""
    success_signatures: list[dict[str, Any]] = []
    clusters: dict[str, list[tuple[TrialTactileExample, dict[str, Any], str]]] = defaultdict(list)

    for example in examples:
        if example.outcome == "success":
            if example.tactile_trace:
                success_signatures.append(
                    build_signature_from_trace(
                        example.tactile_trace,
                        debug_records=example.debug_records,
                    )
                )
            continue
        if not is_tactile_failure(example):
            continue
        signature = build_signature_from_trace(
            example.tactile_trace,
            debug_records=example.debug_records,
        )
        diagnosis = classify_signature(signature)
        clusters[diagnosis].append((example, signature, diagnosis))

    records: list[TactileCodeMemoryRecord] = []
    for diagnosis, group in sorted(clusters.items()):
        merged_signature = merge_signatures([item[1] for item in group])
        evidence = {
            "source_trials": [_source_trial_hash(item[0].trial_dir) for item in group],
            "validation_seeds": [],
            "applicable_count": len(group),
            "recovery_count": 0,
            "validation_success_rate": 0.0,
        }
        strategy, code_sketch = repair_template(diagnosis)
        activation_rule = activation_rule_for_signature(merged_signature, diagnosis)
        trace_summary = build_trace_summary([item[0] for item in group])
        strategy_tags = skill_strategy_tags([item[0] for item in group])
        record = TactileCodeMemoryRecord(
            id=record_id(capability, merged_signature, strategy),
            version=1,
            capability=capability,
            failure_signature=merged_signature,
            applicability=default_applicability(),
            activation_rule=activation_rule,
            diagnosis=diagnosis,
            repair_strategy=strategy,
            code_sketch=code_sketch,
            trace_summary=trace_summary,
            strategy_tags=strategy_tags,
            skill=build_skill_card(
                diagnosis=diagnosis,
                signature=merged_signature,
                activation_rule=activation_rule,
                repair_strategy=strategy,
                code_sketch=code_sketch,
            ),
            evidence=evidence,
            status="validated" if validated else "candidate",
        )
        records.append(record)

    if success_signatures:
        nominal_signature = merge_signatures(success_signatures)
        strategy, code_sketch = repair_template("maintain_stable_grasp")
        activation_rule = activation_rule_for_signature(
            nominal_signature,
            "maintain_stable_grasp",
        )
        evidence = {
            "source_trials": [],
            "validation_seeds": [],
            "applicable_count": len(success_signatures),
            "recovery_count": 0,
            "validation_success_rate": 0.0,
        }
        records.append(
            TactileCodeMemoryRecord(
                id=record_id(capability, nominal_signature, strategy),
                version=1,
                capability=capability,
                failure_signature=nominal_signature,
                applicability=default_applicability(),
                activation_rule=activation_rule,
                diagnosis="maintain_stable_grasp",
                repair_strategy=strategy,
                code_sketch=code_sketch,
                trace_summary={
                    "source_count": len(success_signatures),
                    "source_trace_types": ["tactile_gripper_trace"],
                    "observed_strategy_counts": {},
                    "public_only": True,
                },
                strategy_tags={
                    "grasp_verification": "tactile_recheck",
                    "recovery_source": "pregrasp_local_adjustment",
                    "perception_guard": "none",
                },
                skill=build_skill_card(
                    diagnosis="maintain_stable_grasp",
                    signature=nominal_signature,
                    activation_rule=activation_rule,
                    repair_strategy=strategy,
                    code_sketch=code_sketch,
                ),
                evidence=evidence,
                status="validated" if validated else "candidate",
            )
        )
    return records


def promote_candidates_with_validation(
    candidates: list[dict[str, Any] | TactileCodeMemoryRecord],
    validation_examples: list[TrialTactileExample],
    *,
    min_applicable: int = 3,
    min_recovery: int = 2,
) -> list[TactileCodeMemoryRecord]:
    """Promote candidates when validation examples provide recovery evidence.

    A validation recovery is counted when a failed training-like signature is
    matched by a candidate and the validation trial completes successfully.
    """
    candidate_dicts = [
        item.to_dict() if isinstance(item, TactileCodeMemoryRecord) else dict(item)
        for item in candidates
    ]
    promoted: list[TactileCodeMemoryRecord] = []
    for candidate in candidate_dicts:
        applicable = 0
        recovered = 0
        validation_refs: list[str] = []
        for example in validation_examples:
            if _memory_was_runtime_matched(example, str(candidate.get("id", ""))):
                applicable += 1
                validation_refs.append(_source_trial_hash(example.trial_dir))
                if example.outcome == "success":
                    recovered += 1
                continue

            # Offline signature matches are useful for applicability audits, but
            # they are not recovery evidence because no simulator repair was
            # actually triggered by this memory.
            signature = build_signature_from_trace(
                example.tactile_trace,
                debug_records=example.debug_records,
            )
            if (
                score_memory_record(
                    candidate,
                    capability=str(candidate.get("capability", "grasp_stabilization")),
                    current_signature=signature,
                    required_apis=["FrankaControlApi", "UniVTACTactileApi"],
                )
                > 0.0
            ):
                applicable += 1
        evidence = dict(candidate.get("evidence", {}))
        evidence["validation_seeds"] = validation_refs
        evidence["applicable_count"] = max(int(evidence.get("applicable_count", 0)), applicable)
        evidence["recovery_count"] = recovered
        evidence["validation_success_rate"] = (
            float(recovered / applicable) if applicable else 0.0
        )
        if applicable >= min_applicable and recovered >= min_recovery:
            candidate["status"] = "validated"
            candidate["evidence"] = evidence
            promoted.append(record_from_dict(candidate))
    return promoted


def record_from_dict(item: dict[str, Any]) -> TactileCodeMemoryRecord:
    diagnosis = str(item.get("diagnosis", "unknown_tactile_failure"))
    failure_signature = dict(item.get("failure_signature", {}))
    activation_rule = dict(item.get("activation_rule", {}))
    repair_strategy = str(item.get("repair_strategy", ""))
    code_sketch = str(item.get("code_sketch", ""))
    skill = dict(item.get("skill", {}))
    if not skill:
        skill = build_skill_card(
            diagnosis=diagnosis,
            signature=failure_signature,
            activation_rule=activation_rule,
            repair_strategy=repair_strategy,
            code_sketch=code_sketch,
        )
    return TactileCodeMemoryRecord(
        id=str(item["id"]),
        version=int(item.get("version", 1)),
        capability=str(item.get("capability", "grasp_stabilization")),
        failure_signature=failure_signature,
        applicability=dict(item.get("applicability", {})),
        activation_rule=activation_rule,
        diagnosis=diagnosis,
        repair_strategy=repair_strategy,
        code_sketch=code_sketch,
        trace_summary=dict(item.get("trace_summary", {})),
        strategy_tags=dict(item.get("strategy_tags", {})),
        skill=skill,
        evidence=dict(item.get("evidence", {})),
        status=str(item.get("status", "candidate")),
    )


def is_tactile_failure(example: TrialTactileExample) -> bool:
    """Return True if a failed example appears tactile-control related."""
    if example.outcome == "success":
        return False
    if not example.tactile_trace:
        return False
    if example.sandbox_rc != 0:
        return False
    if any(pattern in example.summary_text for pattern in NON_TACTILE_ERROR_PATTERNS):
        return False
    return True


def classify_signature(signature: dict[str, Any]) -> str:
    """Classify a tactile signature into a task-agnostic diagnosis."""
    events = [str(event) for event in signature.get("event_sequence", [])]
    trends = signature.get("feature_trends", {})
    ranges = signature.get("feature_ranges", {})
    bilateral_ratio = float(trends.get("bilateral_contact_ratio", 0.0))
    left_ratio = float(trends.get("left_contact_ratio", 0.0))
    right_ratio = float(trends.get("right_contact_ratio", 0.0))
    force_max = _range_max(ranges.get("normal_force"))
    slip_max = _range_max(ranges.get("slip_score"))
    balance_max = _range_max(ranges.get("abs_contact_balance"))

    if _has_ordered_subsequence(events, ["stable_grasp", "contact_lost"]):
        return "contact_lost_after_stable_grasp"
    if "contact_lost" in events and (left_ratio > 0.0 or right_ratio > 0.0):
        return "contact_lost_after_one_hand_contact"
    if bilateral_ratio <= 0.05 and (left_ratio > 0.05 or right_ratio > 0.05):
        return "one_sided_or_off_center_contact"
    if force_max < 0.2:
        return "missed_or_undercompressed_contact"
    if slip_max >= 0.6:
        return "slip_during_lift"
    if balance_max >= 0.75:
        return "contact_imbalance"
    if trends.get("object_lift_height_ok") is False and trends.get("stable_seen"):
        return "tactile_stable_but_object_not_lifted"
    return "unknown_tactile_failure"


def merge_signatures(signatures: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge related signatures into one representative signature."""
    if not signatures:
        return {
            "phase": "grasp_or_lift",
            "event_sequence": [],
            "feature_ranges": {},
            "feature_trends": {},
            "window": 20,
        }
    phase = str(signatures[0].get("phase", "grasp_or_lift"))
    seq_counts: Counter[tuple[str, ...]] = Counter(
        tuple(str(event) for event in sig.get("event_sequence", []))
        for sig in signatures
    )
    event_sequence = list(seq_counts.most_common(1)[0][0])
    range_keys = sorted(
        {
            key
            for sig in signatures
            for key in dict(sig.get("feature_ranges", {})).keys()
        }
    )
    trend_keys = sorted(
        {
            key
            for sig in signatures
            for key in dict(sig.get("feature_trends", {})).keys()
        }
    )
    merged_ranges = {
        key: _merge_range([dict(sig.get("feature_ranges", {})).get(key) for sig in signatures])
        for key in range_keys
    }
    merged_trends = {
        key: _merge_trend([dict(sig.get("feature_trends", {})).get(key) for sig in signatures])
        for key in trend_keys
    }
    counts = Counter()
    for sig in signatures:
        counts.update(dict(sig.get("event_counts", {})))
    return {
        "phase": phase,
        "event_sequence": event_sequence,
        "event_counts": dict(counts),
        "feature_ranges": merged_ranges,
        "feature_trends": merged_trends,
        "window": int(signatures[0].get("window", 20)),
    }


def activation_rule_for_signature(signature: dict[str, Any], diagnosis: str) -> dict[str, Any]:
    """Compile a signature into a simple runtime matching rule."""
    events = [str(event) for event in signature.get("event_sequence", [])]
    ranges = dict(signature.get("feature_ranges", {}))
    trends = dict(signature.get("feature_trends", {}))
    rules_all: list[str] = [f"phase == {signature.get('phase', 'grasp_or_lift')}"]
    rules_any: list[str] = []
    if "contact_lost" in events:
        rules_any.append("event_sequence contains contact_lost")
    if "one_hand_contact" in events:
        rules_any.append("event_sequence contains one_hand_contact")
    if _range_max(ranges.get("abs_contact_balance")) >= 0.75:
        rules_any.append("abs(contact_balance) >= 0.75")
    if _range_max(ranges.get("slip_score")) >= 0.6:
        rules_any.append("slip_score >= 0.6")
    if (
        float(trends.get("bilateral_contact_ratio", 0.0) or 0.0) <= 0.05
        and (
            "contact_lost" in events
            or "one_hand_contact" in events
            or _range_max(ranges.get("abs_contact_balance")) >= 0.75
            or _range_max(ranges.get("slip_score")) >= 0.6
        )
    ):
        rules_any.append("bilateral_contact_ratio <= 0.05")
    if not rules_any:
        rules_any.append(f"diagnosis similar to {diagnosis}")
    return {"all": rules_all, "any": rules_any}


def default_applicability() -> dict[str, Any]:
    return {
        "object_roles": ["grasped_object", "active_object"],
        "required_apis": ["FrankaControlApi", "UniVTACTactileApi"],
        "required_tactile_fields": [
            "event",
            "left_contact",
            "right_contact",
            "normal_force",
            "contact_balance",
            "slip_score",
        ],
    }


def repair_template(diagnosis: str) -> tuple[str, str]:
    """Return a task-independent repair strategy and parameterized code sketch."""
    templates = {
        "one_sided_or_off_center_contact": (
            "Recover an off-center grasp by releasing, making a millimeter-scale pose adjustment, and retrying adaptive close until bilateral tactile contact is stable.",
            _code_sketch_regrasp(one_sided=True),
        ),
        "contact_lost_after_one_hand_contact": (
            "Treat one-hand contact followed by contact loss as an off-center pre-grasp: reopen, reuse the current public pre-grasp anchor, apply a millimeter-scale depth/lateral correction, close adaptively, and only continue after stable bilateral contact.",
            _code_sketch_regrasp(one_sided=True),
        ),
        "contact_lost_after_stable_grasp": (
            "When contact is lost after a stable grasp, lower or pause before continuing, close adaptively again, and lift in short guarded increments while rechecking slip.",
            _code_sketch_guarded_lift(),
        ),
        "missed_or_undercompressed_contact": (
            "If adaptive close ends with little or no compression, move slightly deeper along the public grasp approach and retry with a bounded adaptive close.",
            _code_sketch_regrasp(one_sided=False),
        ),
        "slip_during_lift": (
            "If slip rises during lift, stop lifting, lower a small amount, re-close adaptively, then continue with smaller lift increments and tactile checks.",
            _code_sketch_guarded_lift(),
        ),
        "contact_imbalance": (
            "If force is strongly imbalanced, release and retry the grasp with a small lateral/depth adjustment rather than lifting on a single tactile side.",
            _code_sketch_regrasp(one_sided=True),
        ),
        "tactile_stable_but_object_not_lifted": (
            "Do not trust a single stable tactile reading: confirm stability across several frames, then perform a guarded lift and recheck that contact remains bilateral.",
            _code_sketch_guarded_lift(),
        ),
        "maintain_stable_grasp": (
            "Maintain the successful pattern: confirm bilateral stable contact before lift, keep the lift short and guarded, then release only after the object is supported.",
            _code_sketch_guarded_lift(nominal=True),
        ),
    }
    return templates.get(
        diagnosis,
        (
            "Use native tactile feedback to guard the grasp: require bilateral contact, retry once on contact loss, and avoid lifting on unstable contact.",
            _code_sketch_regrasp(one_sided=False),
        ),
    )


def format_memory_for_prompt(records: list[dict[str, Any]]) -> str:
    """Format retrieved records for initial code generation prompts."""
    lines = [
        "Tactile code memory (compact skill cards):",
        "Use these as short partial-code hints only when the tactile state matches. Candidate skills are unvalidated. Do not read reward, success, trial id, hidden object fields, or private metadata.",
    ]
    if not records:
        lines.append("No validated tactile code memory is available yet.")
        return "\n".join(lines)
    for idx, record in enumerate(records, start=1):
        lines.append(_format_record_for_prompt(idx, record, code_lines=10))
    return "\n".join(lines)


def format_runtime_memory_for_prompt(
    current_signature: dict[str, Any],
    records: list[dict[str, Any]],
) -> str:
    """Format runtime retrieval context for a REGENERATE prompt."""
    lines = [
        "Runtime tactile code memory:",
        f"Current tactile signature: {signature_brief(current_signature)}",
    ]
    if not records:
        lines.append("No matching validated tactile repair memory was retrieved.")
        return "\n".join(lines)
    lines.append(
        "Use a matching partial repair only if it fits the current public APIs and current tactile state. Do not restart a completed or released state just because contact is now absent."
    )
    for idx, record in enumerate(records, start=1):
        lines.append(_format_record_for_prompt(idx, record, code_lines=14))
    return "\n".join(lines)


def signature_brief(signature: dict[str, Any]) -> str:
    events = " -> ".join(str(e) for e in signature.get("event_sequence", [])[:8])
    trends = dict(signature.get("feature_trends", {}))
    ranges = dict(signature.get("feature_ranges", {}))
    return (
        f"phase={signature.get('phase', 'unknown')}; events={events or 'none'}; "
        f"bilateral_ratio={float(trends.get('bilateral_contact_ratio', 0.0) or 0.0):.2f}; "
        f"force_max={_range_max(ranges.get('normal_force')):.2f}; "
        f"balance_max={_range_max(ranges.get('abs_contact_balance')):.2f}; "
        f"slip_max={_range_max(ranges.get('slip_score')):.2f}"
    )


def score_memory_record(
    record: dict[str, Any],
    *,
    capability: str,
    current_signature: dict[str, Any] | None,
    required_apis: list[str] | None,
) -> float:
    """Score a memory record for retrieval."""
    score = 0.0
    if str(record.get("capability", "")) == capability:
        score += 20.0
    required = set(required_apis or [])
    available = set(dict(record.get("applicability", {})).get("required_apis", []))
    if required and not available.issubset(required):
        return 0.0
    if required:
        score += 10.0
    if str(record.get("status", "")) == "validated":
        score += 25.0
    evidence = dict(record.get("evidence", {}))
    score += min(float(evidence.get("validation_success_rate", 0.0)) * 20.0, 20.0)
    score += min(float(evidence.get("recovery_count", 0.0)) * 2.0, 10.0)
    tags = dict(record.get("strategy_tags", {}))
    if tags.get("recovery_source") == "pregrasp_local_adjustment":
        score += 3.0
    if tags.get("grasp_verification") == "tactile_recheck":
        score += 1.0
    if current_signature is None:
        return score
    if not activation_rule_matches(
        dict(record.get("activation_rule", {})),
        current_signature,
    ):
        return 0.0
    score += signature_similarity(
        current_signature,
        dict(record.get("failure_signature", {})),
    ) * 45.0
    return score


def activation_rule_matches(rule: dict[str, Any], signature: dict[str, Any]) -> bool:
    """Return whether a current tactile signature activates a memory record."""
    if not _signature_has_runtime_failure_evidence(signature):
        return False
    all_rules = [str(item) for item in rule.get("all", [])]
    any_rules = [str(item) for item in rule.get("any", [])]
    return all(_match_activation_clause(item, signature) for item in all_rules) and (
        not any_rules or any(_match_activation_clause(item, signature) for item in any_rules)
    )


def _signature_has_runtime_failure_evidence(signature: dict[str, Any]) -> bool:
    events = {str(event) for event in signature.get("event_sequence", [])}
    ranges = dict(signature.get("feature_ranges", {}))
    trends = dict(signature.get("feature_trends", {}))
    return bool(
        {"one_hand_contact", "contact_lost", "slip_detected"} & events
        or bool(trends.get("stable_then_lost", False))
        or _range_max(ranges.get("abs_contact_balance")) >= 0.75
        or _range_max(ranges.get("slip_score")) >= 0.6
    )


def _match_activation_clause(clause: str, signature: dict[str, Any]) -> bool:
    clause = clause.strip()
    events = [str(event) for event in signature.get("event_sequence", [])]
    ranges = dict(signature.get("feature_ranges", {}))
    trends = dict(signature.get("feature_trends", {}))
    if clause.startswith("phase =="):
        return str(signature.get("phase", "")).strip() == clause.split("==", 1)[1].strip()
    if clause.startswith("event_sequence contains"):
        return clause.rsplit(" ", 1)[-1].strip() in events
    if clause.startswith("abs(contact_balance) >="):
        threshold = _parse_clause_threshold(clause, default=0.75)
        return _range_max(ranges.get("abs_contact_balance")) >= threshold
    if clause.startswith("slip_score >="):
        threshold = _parse_clause_threshold(clause, default=0.6)
        return _range_max(ranges.get("slip_score")) >= threshold
    if clause.startswith("bilateral_contact_ratio <="):
        threshold = _parse_clause_threshold(clause, default=0.05)
        return float(trends.get("bilateral_contact_ratio", 0.0) or 0.0) <= threshold
    if clause.startswith("diagnosis similar to"):
        return classify_signature(signature) == clause.rsplit(" ", 1)[-1].strip()
    return False


def _parse_clause_threshold(clause: str, *, default: float) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", clause)
    if match is None:
        return default
    try:
        return float(match.group(0))
    except ValueError:
        return default


def signature_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Return a rough similarity score in [0, 1]."""
    left_events = [str(e) for e in left.get("event_sequence", [])]
    right_events = [str(e) for e in right.get("event_sequence", [])]
    if not left_events or not right_events:
        event_score = 0.0
    else:
        left_set = set(left_events)
        right_set = set(right_events)
        event_score = len(left_set & right_set) / max(1, len(left_set | right_set))
        if _has_ordered_overlap(left_events, right_events):
            event_score = max(event_score, 0.65)
    trend_score = 0.0
    trend_keys = [
        "bilateral_contact_ratio",
        "left_contact_ratio",
        "right_contact_ratio",
        "contact_balance_final",
    ]
    left_trends = dict(left.get("feature_trends", {}))
    right_trends = dict(right.get("feature_trends", {}))
    matched = 0
    for key in trend_keys:
        if key not in left_trends or key not in right_trends:
            continue
        matched += 1
        diff = abs(float(left_trends.get(key, 0.0) or 0.0) - float(right_trends.get(key, 0.0) or 0.0))
        trend_score += max(0.0, 1.0 - diff)
    trend_score = trend_score / matched if matched else 0.0
    return float(max(0.0, min(1.0, 0.65 * event_score + 0.35 * trend_score)))


def read_trial_outcome(trial_path: Path, summary_text: str | None = None) -> tuple[float, bool, int]:
    text = summary_text if summary_text is not None else _read_text(trial_path / "summary.txt")
    reward_match = re.search(r"Reward:\s*([-+]?\d+(?:\.\d+)?)", text)
    completed_match = re.search(r"Task Completed:\s*(True|False|1|0)", text)
    sandbox_match = re.search(r"Sandbox failed:\s*([01])", text)
    reward = float(reward_match.group(1)) if reward_match else 0.0
    completed = _parse_bool(completed_match.group(1)) if completed_match else False
    sandbox_rc = int(sandbox_match.group(1)) if sandbox_match else 0
    name_match = re.search(
        r"trial_(\d+)_sandboxrc_([01])_reward_([-+]?\d+(?:\.\d+)?)_taskcompleted_([01])",
        trial_path.name,
    )
    if name_match:
        sandbox_rc = int(name_match.group(2)) if sandbox_match is None else sandbox_rc
        reward = float(name_match.group(3)) if reward_match is None else reward
        completed = _parse_bool(name_match.group(4)) if completed_match is None else completed
    return reward, completed, sandbox_rc


def parse_trial_index(name: str) -> int | None:
    match = re.search(r"trial_(\d+)", name)
    return int(match.group(1)) if match else None


def record_id(capability: str, signature: dict[str, Any], strategy: str) -> str:
    stable = {
        "capability": capability,
        "events": signature.get("event_sequence", []),
        "phase": signature.get("phase", ""),
        "strategy": strategy,
        "rules": activation_rule_for_signature(signature, classify_signature(signature)),
    }
    return hashlib.sha1(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def event_sequence(rows: list[dict[str, Any]]) -> list[str]:
    events: list[str] = []
    for row in rows:
        event = str(row.get("event", "unknown"))
        if event not in TACTILE_EVENTS:
            event = "unknown"
        if not events or events[-1] != event:
            events.append(event)
    return events


def _format_record_for_prompt(idx: int, record: dict[str, Any], *, code_lines: int) -> str:
    skill = dict(record.get("skill", {}))
    repair_strategy = str(
        skill.get("repair_strategy")
        or record.get("repair_strategy", "")
    )
    lines = [
        (
            f"{idx}. id={record.get('id', 'unknown')} "
            f"status={record.get('status', 'unknown')} "
            f"diagnosis={record.get('diagnosis', 'unknown')}"
        ),
        f"   when: {skill.get('when_to_apply', 'when current tactile signature matches')}",
        f"   repair: {repair_strategy}",
    ]
    sketch = str(skill.get("code_sketch") or record.get("code_sketch", "")).strip()
    if sketch:
        lines.append("   code_hint:")
        lines.extend(f"      {line}" for line in _compact_code_sketch(sketch, max_lines=code_lines))
    return "\n".join(lines)


def _compact_code_sketch(sketch: str, *, max_lines: int) -> list[str]:
    """Keep prompt-facing code hints short while preserving executable shape."""
    kept: list[str] = []
    for raw_line in sketch.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        kept.append(line)
        if len(kept) >= max_lines:
            break
    if len([line for line in sketch.splitlines() if line.strip()]) > len(kept):
        kept.append("    # ...keep this repair local and stop after the guarded action...")
    return kept


def _code_sketch_regrasp(*, one_sided: bool) -> str:
    adjustment = (
        "Use the sign of contact_balance only as a hint; keep the lateral correction millimeter-scale."
        if one_sided
        else "Use a tiny downward correction first, then a tiny lateral correction if contact is still absent."
    )
    return f"""def tactile_pregrasp_repair(object_name="can", max_retries=2, target_force=1.0, close_steps=180, depth_step=0.003, lateral_step=0.003):
    import numpy as np
    anchor_pos, anchor_quat = sample_grasp_pose(object_name)  # public official grasp anchor
    for attempt in range(max_retries):
        summary = get_tactile_summary(window=20, hand="both")
        if summary.get("left_contact") and summary.get("right_contact") and summary.get("slip_score", 1.0) < 0.6:
            return True
        open_gripper(adaptive=True)
        offset = np.array([0.0, 0.0, -abs(float(depth_step))], dtype=float)
        if attempt > 0:
            balance = float(summary.get("contact_balance", 0.0))
            offset = np.array([float(lateral_step) if balance < 0 else -float(lateral_step), 0.0, -0.5 * abs(float(depth_step))], dtype=float)
        # {adjustment}
        goto_pose(np.asarray(anchor_pos, dtype=float) + offset, anchor_quat, z_approach=0.0)
        result = close_gripper(adaptive=True, target_force=target_force, max_steps=close_steps)
        if result.get("stable", False):
            return True
    return False"""


def _code_sketch_guarded_lift(*, nominal: bool = False) -> str:
    prefix = (
        "Use this after a stable tactile close to keep lift guarded."
        if nominal
        else "Use this when lift or post-close tactile feedback suggests slip or contact loss."
    )
    return f"""def guarded_tactile_lift(object_name="can", z_approach=0.08, target_force=1.0, close_steps=160):
    # {prefix}
    summary = get_tactile_summary(window=20, hand="both")
    if not (summary.get("left_contact") and summary.get("right_contact")) or summary.get("slip_score", 1.0) >= 0.6:
        close_gripper(adaptive=True, target_force=target_force, max_steps=max(40, close_steps // 2))
        summary = get_tactile_summary(window=20, hand="both")
    if not (summary.get("left_contact") and summary.get("right_contact")):
        open_gripper(adaptive=True)
        anchor_pos, anchor_quat = sample_grasp_pose(object_name)
        goto_pose(anchor_pos, anchor_quat, z_approach=0.0)
        retry = close_gripper(adaptive=True, target_force=target_force, max_steps=close_steps)
        if not retry.get("stable", False):
            return False
    home_pose()
    post_lift = get_tactile_summary(window=20, hand="both")
    if post_lift.get("slip_score", 1.0) >= 0.6 or not post_lift.get("contact", False):
        close_gripper(adaptive=True, target_force=target_force, max_steps=max(40, close_steps // 2))
    return True"""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _read_json(path: Path, *, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path, default=[])
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _range(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [_to_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _range_abs(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [_to_float(row.get(key)) for row in rows]
    values = [abs(value) for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _merge_range(values: list[Any]) -> dict[str, float | None]:
    mins = [_to_float(dict(v).get("min")) for v in values if isinstance(v, dict)]
    maxs = [_to_float(dict(v).get("max")) for v in values if isinstance(v, dict)]
    means = [_to_float(dict(v).get("mean")) for v in values if isinstance(v, dict)]
    mins = [v for v in mins if v is not None]
    maxs = [v for v in maxs if v is not None]
    means = [v for v in means if v is not None]
    return {
        "min": float(min(mins)) if mins else None,
        "max": float(max(maxs)) if maxs else None,
        "mean": float(sum(means) / len(means)) if means else None,
    }


def _merge_trend(values: list[Any]) -> Any:
    finite = [_to_float(value) for value in values]
    finite = [value for value in finite if value is not None]
    if finite:
        return float(sum(finite) / len(finite))
    bools = [value for value in values if isinstance(value, bool)]
    if bools:
        return sum(1 for value in bools if value) >= (len(bools) / 2.0)
    for value in values:
        if value is not None:
            return value
    return None


def _range_max(value: Any) -> float:
    if isinstance(value, dict):
        parsed = _to_float(value.get("max"))
        return float(parsed) if parsed is not None else 0.0
    parsed = _to_float(value)
    return float(parsed) if parsed is not None else 0.0


def _delta(rows: list[dict[str, Any]], key: str) -> float:
    values = [_to_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return 0.0
    return float(values[-1] - values[0])


def _last_float(rows: list[dict[str, Any]], key: str, default: float) -> float:
    for row in reversed(rows):
        value = _to_float(row.get(key))
        if value is not None:
            return float(value)
    return float(default)


def _bool_ratio(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if bool(row.get(key, False))) / len(rows))


def _bilateral_ratio(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return float(
        sum(
            1
            for row in rows
            if bool(row.get("left_contact", False)) and bool(row.get("right_contact", False))
        )
        / len(rows)
    )


def _object_motion_features(debug_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not debug_records:
        return {}
    last = debug_records[-1]
    output: dict[str, Any] = {}
    if "can_lift_height_ok" in last:
        output["object_lift_height_ok"] = bool(last.get("can_lift_height_ok"))
    if "can_lift_delta" in last:
        value = _to_float(last.get("can_lift_delta"))
        if value is not None:
            output["object_lift_delta"] = value
    return output


def _has_contact_loss_after_contact(events: list[str]) -> bool:
    seen_contact = False
    for event in events:
        if event in {"one_hand_contact", "stable_grasp", "slip_detected"}:
            seen_contact = True
        if seen_contact and event == "contact_lost":
            return True
    return False


def _has_ordered_subsequence(sequence: list[str], expected: list[str]) -> bool:
    if not expected:
        return True
    pos = 0
    for item in sequence:
        if item == expected[pos]:
            pos += 1
            if pos == len(expected):
                return True
    return False


def _has_ordered_overlap(left: list[str], right: list[str]) -> bool:
    if not left or not right:
        return False
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    return _has_ordered_subsequence(longer, shorter[: min(3, len(shorter))])


def _memory_was_runtime_matched(example: TrialTactileExample, memory_id: str) -> bool:
    if not memory_id:
        return False
    for entry in example.memory_trace:
        if entry.get("kind") != "runtime_retrieval":
            continue
        for match in entry.get("matches", []):
            if isinstance(match, dict) and str(match.get("id", "")) == memory_id:
                return True
    return False


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _source_trial_hash(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"trial_sha1:{digest}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
