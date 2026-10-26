"""Load public Shadow piano lessons into code-generation prompts.

The offline failure audit is intentionally not an accepted input to this loader.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml


SCHEMA = "shadow_piano_public_skills.v1"
MAX_FILE_BYTES = 8192
MAX_SKILLS = 16
_PRIVATE_MARKERS = (
    "capx-private-audits", "private_key_top_force", "native_failure",
    "native_notes_completed", "shadow_piano_physics_trace", "shadow_piano_score_events",
    "actual_force_n", "measured_key_top_force_at_press", "score.events",
)


def inject_shadow_piano_public_skills(
    env_factory: dict[str, Any], *, config_path: str | Path, library_path: str | Path,
) -> tuple[Path, str]:
    """Append a bounded, explicitly public library to the initial task prompt.

    The initial message is copied into every Shadow regeneration request.  The
    skill text is therefore included once per model request without duplicating
    it in ``multi_turn_prompt``.
    """
    cfg = env_factory["cfg"]
    if (cfg["low_level"].get("_target_")
            != "capx.envs.simulators.opentac_shadow.ShadowPianoLowLevelEnv"
            or "ShadowPianoVisualForceApi" not in cfg.get("apis", [])):
        raise ValueError("Public Shadow piano skills require the visual-force adapter")
    path = Path(library_path).expanduser()
    if not path.is_absolute():
        path = Path(config_path).expanduser().resolve().parent / path
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError("Public Shadow piano skill library is empty or too large")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or set(data) != {"schema", "skills"} or data["schema"] != SCHEMA:
        raise ValueError("Unsupported public Shadow piano skill library schema")
    skills = data["skills"]
    if not isinstance(skills, list) or not 0 <= len(skills) <= MAX_SKILLS:
        raise ValueError("Public Shadow piano skill library needs 0 to 16 skills")
    lines = [
        "Public Shadow piano skill library. These are reusable lessons from prior PUBLIC observations,",
        ("not observations from this trial. Reuse a validated cached endpoint calibration without images when available; otherwise confirm the initial HEAD endpoints. Use fresh public hand/DIGIT feedback."
         if cfg["low_level"].get("endpoint_calibration", False) else
         "not observations from this trial. Use the initial HEAD image for key mapping; after mapping, use saved coordinates and fresh public hand/DIGIT readings."),
    ]
    if not skills:
        lines.append("No skill entries have been recorded yet; the library starts empty.")
    ids: set[str] = set()
    for skill in skills:
        if not isinstance(skill, dict) or set(skill) != {"id", "trigger", "practice"}:
            raise ValueError("Public Shadow piano skills need id, trigger and practice only")
        identifier, trigger, practice = (skill[k] for k in ("id", "trigger", "practice"))
        if (not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", identifier)
                or identifier in ids or not isinstance(trigger, str) or not isinstance(practice, str)
                or not 1 <= len(trigger) <= 240 or not 1 <= len(practice) <= 600):
            raise ValueError("Invalid or duplicate public Shadow piano skill")
        ids.add(identifier)
        lines.append(f"- {identifier}: When {trigger} Do: {practice}")
    content = "\n".join(lines)
    if any(marker in content.lower() for marker in _PRIVATE_MARKERS):
        raise ValueError("Public Shadow piano skill library contains private audit markers")
    prompt = cfg.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Public Shadow piano skills require a task prompt")
    digest = hashlib.sha256(raw).hexdigest()
    cfg["prompt"] = f"{prompt.rstrip()}\n\n{content}\nLibrary SHA256: {digest}\n"
    return path, digest
