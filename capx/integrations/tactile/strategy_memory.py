"""Lightweight tactile strategy memory for CaP-X trials."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .summarizer import normalize_target

DEFAULT_MEMORY_FILENAME = ".capx_tactile_strategies.jsonl"
LOW_REWARD_THRESHOLD = 0.5
NO_CONTACT_RATIO_THRESHOLD = 0.75

STRATEGY_TEMPLATES = {
    "successful_execution": (
        "reuse tactile verification after grasping: confirm stable contact, "
        "check for slip, lift safely, and approach placement with z_approach"
    ),
    "missed_grasp": "retry with a small downward or lateral adjustment before lifting",
    "off_center_grasp": "reopen, shift the grasp toward balanced contact, and close again",
    "slip_during_lift": "lower 5mm, close again, lift slower, and recheck slip",
    "placement_error": (
        "keep grasp stable, approach the place pose with z_approach, "
        "and release near the support surface"
    ),
    "unknown_tactile_failure": "retry once with tactile contact and slip checks before lifting",
}

CODE_HINT_TEMPLATES = {
    "successful_execution": (
        "After close_gripper(), call is_grasp_stable() or is_contacting(); "
        "call is_slipping() before and during lift; use z_approach when placing."
    ),
    "missed_grasp": (
        "If not is_contacting(target='red cube'), adjust grasp_pos by about 0.005-0.01m "
        "downward or laterally, then close_gripper() again before lifting."
    ),
    "off_center_grasp": (
        "If get_tactile_summary()['event'] is one_finger_contact, open_gripper(), "
        "shift the grasp toward the missing finger side, and retry the grasp."
    ),
    "slip_during_lift": (
        "If is_slipping(target='red cube'), lower the object slightly, close_gripper(), "
        "then lift more slowly and recheck is_slipping()."
    ),
    "placement_error": (
        "Keep the same grasp quaternion, compute place_z from object half heights, "
        "goto_pose(place_pos, grasp_quat, z_approach=0.1), then open_gripper()."
    ),
    "unknown_tactile_failure": (
        "Use get_tactile_summary(), wait_until_contact(), and one bounded retry "
        "instead of assuming the object is grasped."
    ),
}


@dataclass(slots=True)
class TactileStrategyRecord:
    """One persisted tactile strategy memory item."""

    id: str
    task: str
    target: str
    failure_type: str
    event_sequence: list[str]
    event_counts: dict[str, int]
    strategy: str
    code_hint: str
    source_trial_dir: str
    reward: float
    task_completed: bool
    outcome: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_root() -> Path:
    """Return the CaP-X project root."""
    return Path(__file__).resolve().parents[3]


def resolve_memory_path(path: str | Path | None = None) -> Path:
    """Resolve a tactile memory path relative to the CaP-X project root."""
    if path is None or str(path).strip() == "":
        env_path = os.getenv("CAPX_TACTILE_STRATEGY_MEMORY_READ_PATH", "")
        if not env_path:
            env_path = os.getenv("CAPX_TACTILE_STRATEGY_MEMORY_PATH", "")
        path = env_path or DEFAULT_MEMORY_FILENAME
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def event_sequence_from_timeline(timeline: list[dict[str, Any]]) -> list[str]:
    """Return consecutive-deduplicated tactile events from a timeline."""
    sequence: list[str] = []
    for row in timeline:
        event = str(row.get("event", "unknown"))
        if not sequence or sequence[-1] != event:
            sequence.append(event)
    return sequence


def event_counts_from_timeline(timeline: list[dict[str, Any]]) -> dict[str, int]:
    """Return event counts from a tactile timeline."""
    return dict(Counter(str(row.get("event", "unknown")) for row in timeline))


def classify_failure(
    timeline: list[dict[str, Any]],
    *,
    reward: float,
    task_completed: bool,
) -> str:
    """Classify the first-pass tactile failure mode using rule templates."""
    if task_completed or reward >= LOW_REWARD_THRESHOLD:
        return "successful_execution"

    sequence = event_sequence_from_timeline(timeline)
    counts = event_counts_from_timeline(timeline)
    total = max(1, len(timeline))
    no_contact_ratio = counts.get("no_contact", 0) / total

    if _has_ordered_subsequence(sequence, ["stable_grasp", "slip_detected", "contact_lost"]):
        return "slip_during_lift"
    if counts.get("one_finger_contact", 0) > 0:
        return "off_center_grasp"
    if no_contact_ratio >= NO_CONTACT_RATIO_THRESHOLD:
        return "missed_grasp"
    if counts.get("stable_grasp", 0) > 0:
        return "placement_error"
    return "unknown_tactile_failure"


def build_strategy_record(
    timeline: list[dict[str, Any]],
    *,
    task: str,
    target: str,
    source_trial_dir: str | Path,
    reward: float,
    task_completed: bool,
) -> TactileStrategyRecord:
    """Build a memory record from a tactile timeline and trial metadata."""
    failure_type = classify_failure(
        timeline,
        reward=float(reward),
        task_completed=bool(task_completed),
    )
    source = str(Path(source_trial_dir).resolve())
    record_id = hashlib.sha1(
        f"{task}|{normalize_target(target)}|{source}".encode("utf-8")
    ).hexdigest()[:16]
    return TactileStrategyRecord(
        id=record_id,
        task=task,
        target=target,
        failure_type=failure_type,
        event_sequence=event_sequence_from_timeline(timeline),
        event_counts=event_counts_from_timeline(timeline),
        strategy=STRATEGY_TEMPLATES[failure_type],
        code_hint=CODE_HINT_TEMPLATES[failure_type],
        source_trial_dir=source,
        reward=float(reward),
        task_completed=bool(task_completed),
        outcome="success" if bool(task_completed) or float(reward) >= LOW_REWARD_THRESHOLD else "failure",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def build_strategy_record_from_trial_dir(
    trial_dir: str | Path,
    *,
    task: str = "cube_stack",
    target: str = "red cube",
) -> TactileStrategyRecord | None:
    """Build a memory record from a saved CaP-X trial directory."""
    trial_path = Path(trial_dir)
    timeline_path = trial_path / "tactile_timeline.json"
    if not timeline_path.exists():
        return None

    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if not isinstance(timeline, list) or not timeline:
        return None

    reward, task_completed = _read_trial_outcome(trial_path)
    return build_strategy_record(
        timeline,
        task=task,
        target=target,
        source_trial_dir=trial_path,
        reward=reward,
        task_completed=task_completed,
    )


class TactileStrategyMemory:
    """Append-only JSONL tactile strategy memory."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = resolve_memory_path(path)

    def load(self) -> list[dict[str, Any]]:
        """Load all valid records from disk."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
        return records

    def append(self, record: TactileStrategyRecord | dict[str, Any]) -> bool:
        """Append a record unless its source trial directory was already stored."""
        item = record.to_dict() if isinstance(record, TactileStrategyRecord) else dict(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a", encoding="utf-8") as lock_file:
            _lock_file(lock_file)
            try:
                existing = self.load()
                source = str(item.get("source_trial_dir", ""))
                if source and any(str(row.get("source_trial_dir", "")) == source for row in existing):
                    return False
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(item, sort_keys=True) + "\n")
                return True
            finally:
                _unlock_file(lock_file)

    def append_trial_dir(
        self,
        trial_dir: str | Path,
        *,
        task: str = "cube_stack",
        target: str = "red cube",
    ) -> TactileStrategyRecord | None:
        """Build and append a memory record from a trial directory."""
        record = build_strategy_record_from_trial_dir(trial_dir, task=task, target=target)
        if record is None:
            return None
        return record if self.append(record) else None

    def retrieve(
        self,
        *,
        failure_type: str | None = None,
        target: str = "red cube",
        top_k: int = 3,
        prefer_success: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant strategy records."""
        records = self.load()
        if not records or top_k <= 0:
            return []
        normalized_target = normalize_target(target)

        scored: list[tuple[float, int, dict[str, Any]]] = []
        for idx, record in enumerate(records):
            score = 0.0
            if normalize_target(str(record.get("target", ""))) == normalized_target:
                score += 10.0
            if failure_type and record.get("failure_type") == failure_type:
                score += 40.0
            if prefer_success and record.get("outcome") == "success":
                score += 25.0
            score += min(float(record.get("reward", 0.0)), 1.0)
            scored.append((score, idx, record))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[: max(0, int(top_k))]]


def append_trial_strategy_record(
    trial_dir: str | Path,
    *,
    memory_path: str | Path | None = None,
    task: str = "cube_stack",
    target: str = "red cube",
) -> TactileStrategyRecord | None:
    """Append one trial directory to tactile strategy memory."""
    return TactileStrategyMemory(memory_path).append_trial_dir(
        trial_dir,
        task=task,
        target=target,
    )


def compile_memory_from_run_dir(
    run_dir: str | Path,
    *,
    memory_path: str | Path | None = None,
    task: str = "cube_stack",
    target: str = "red cube",
) -> list[TactileStrategyRecord]:
    """Scan a run directory and append all tactile trial records."""
    root = Path(run_dir)
    memory = TactileStrategyMemory(memory_path)
    added: list[TactileStrategyRecord] = []
    for timeline_path in sorted(root.rglob("tactile_timeline.json")):
        record = memory.append_trial_dir(timeline_path.parent, task=task, target=target)
        if record is not None:
            added.append(record)
    return added


def format_strategies_for_prompt(records: list[dict[str, Any]]) -> str:
    """Format retrieved tactile strategies for prompt injection."""
    lines = [
        "Tactile strategy memory:",
        "Use these as historical strategy hints only. Do not use reward, success, "
        "or trial completion as runtime observations.",
    ]
    if not records:
        lines.append("No learned tactile strategies are available yet.")
        return "\n".join(lines)

    for idx, record in enumerate(records, start=1):
        lines.append(
            f"{idx}. [{record.get('outcome', 'unknown')}; "
            f"{record.get('failure_type', 'unknown')}] {record.get('strategy', '')}"
        )
        code_hint = str(record.get("code_hint", "")).strip()
        if code_hint:
            lines.append(f"   Code hint: {code_hint}")
        events = record.get("event_sequence", [])
        if events:
            lines.append(f"   Tactile events: {' -> '.join(str(event) for event in events[:8])}")
    return "\n".join(lines)


def _has_ordered_subsequence(sequence: list[str], expected: list[str]) -> bool:
    pos = 0
    for event in sequence:
        if event == expected[pos]:
            pos += 1
            if pos == len(expected):
                return True
    return False


def _read_trial_outcome(trial_path: Path) -> tuple[float, bool]:
    summary_path = trial_path / "summary.txt"
    text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    reward_match = re.search(r"Reward:\s*([-+]?\d+(?:\.\d+)?)", text)
    task_match = re.search(r"Task Completed:\s*(True|False|1|0)", text)
    reward = float(reward_match.group(1)) if reward_match else 0.0
    task_completed = _parse_bool(task_match.group(1)) if task_match else False

    name_match = re.search(
        r"reward_([-+]?\d+(?:\.\d+)?)_taskcompleted_([01])",
        trial_path.name,
    )
    if name_match:
        if not reward_match:
            reward = float(name_match.group(1))
        if not task_match:
            task_completed = _parse_bool(name_match.group(2))

    return reward, task_completed


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _lock_file(file: Any) -> None:
    try:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
    except Exception:
        return


def _unlock_file(file: Any) -> None:
    try:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    except Exception:
        return
