"""Trial-local, agent-authored progress files for code-block continuation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "capx_agent_progress.v1"
FILENAME = "agent_progress.json"
MAX_FILE_BYTES = 8192


def progress_path(config: dict[str, Any], trial: int) -> Path | None:
    if not config.get("agent_progress_file", {}).get("enabled", False):
        return None
    output_dir = config.get("output_dir")
    if not output_dir:
        raise ValueError("agent_progress_file requires output_dir")
    return Path(output_dir).resolve() / f"trial_{trial:02d}_in_progress" / FILENAME


def initialize_progress(path: Path) -> None:
    """Reset the file on every attempt, including retries of the same trial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.name + ".tmp").unlink(missing_ok=True)
    path.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "status": "uninitialized"}),
        encoding="utf-8",
    )


def read_progress(path: Path) -> dict[str, Any]:
    """Return only public, bounded fields; never forward arbitrary agent text."""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("oversized")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("schema")
        if raw.get("status") == "uninitialized":
            return {"schema_version": SCHEMA_VERSION, "status": "uninitialized"}
        if raw.get("status") != "active":
            raise ValueError("status")

        grasp = raw["grasp_confirmed"]
        estimator = raw["estimator_started"]
        completed = raw["completed_stages"]
        active = raw["active_stage_index"]
        elapsed = raw["hold_elapsed_s"]
        filtered = raw["filtered_tension_N"]
        step = raw["last_sample_step"]
        if type(grasp) is not bool or type(estimator) is not bool:
            raise ValueError("grasp/estimator flags")
        if not isinstance(completed, list) or completed not in ([], [0], [0, 1]):
            raise ValueError("completed stages")
        if type(active) is not int or active not in (0, 1, 2):
            raise ValueError("active stage")
        if active != len(completed) or (estimator and not grasp):
            raise ValueError("stage order")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
            raise ValueError("hold elapsed")
        if not math.isfinite(elapsed) or not 0 <= elapsed <= 60:
            raise ValueError("hold elapsed")
        if filtered is not None and (
            isinstance(filtered, bool)
            or not isinstance(filtered, (int, float))
            or not math.isfinite(filtered)
        ):
            raise ValueError("filtered tension")
        if step is not None and (type(step) is not int or step < 0):
            raise ValueError("sample step")
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "active",
            "grasp_confirmed": grasp,
            "estimator_started": estimator,
            "completed_stages": completed,
            "active_stage_index": active,
            "hold_elapsed_s": float(elapsed),
            "filtered_tension_N": None if filtered is None else float(filtered),
            "last_sample_step": step,
        }
    except FileNotFoundError:
        return {"schema_version": SCHEMA_VERSION, "status": "missing"}
    except (KeyError, OSError, UnicodeError, ValueError, TypeError):
        return {"schema_version": SCHEMA_VERSION, "status": "invalid"}


def progress_context(path: Path, *, max_chars: int = 1000) -> str:
    snapshot = read_progress(path)
    payload = json.dumps(snapshot, separators=(",", ":"), allow_nan=False)
    prefix = (
        "Public agent progress (not task GT). Read AGENT_PROGRESS_PATH before acting; "
        "missing/invalid means stop, not replay:\n"
    )
    if len(prefix) + len(payload) > max_chars:
        payload = '{"status":"unavailable","reason":"context_budget"}'
    return (prefix + payload)[:max_chars]


def save_progress_artifact(path: Path | None, code_path: str | None) -> None:
    if path is None or code_path is None:
        return
    destination = Path(code_path).parent / FILENAME
    destination.write_text(
        json.dumps(read_progress(path), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
