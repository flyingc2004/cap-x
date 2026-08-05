"""Rule-based tactile proxy summarization.

The robosuite MVP does not expose real GelSight/marker arrays. Instead, it
converts MuJoCo gripper-pad contacts and relative object motion into the same
kind of compact state that a real tactile adapter should eventually provide.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ring_buffer import TactileFrame

TARGET_ALIASES = {
    "red cube": "cubeA",
    "primary": "cubeA",
    "cubea": "cubeA",
    "cube a": "cubeA",
    "cubeA": "cubeA",
    "green cube": "cubeB",
    "secondary": "cubeB",
    "cubeb": "cubeB",
    "cube b": "cubeB",
    "cubeB": "cubeB",
}

TARGET_GEOMS = {
    "cubeA": "cubeA_g0",
    "cubeB": "cubeB_g0",
}

EVENTS = {
    "no_contact",
    "one_finger_contact",
    "stable_grasp",
    "slip_detected",
    "contact_lost",
    "unknown",
}


def normalize_target(target: str) -> str:
    """Normalize a target alias to a robosuite cube key."""
    key = target.strip()
    return TARGET_ALIASES.get(key, TARGET_ALIASES.get(key.lower(), key))


def target_geom_name(target: str) -> str:
    """Return the MuJoCo geom name for a supported tactile target."""
    normalized = normalize_target(target)
    return TARGET_GEOMS.get(normalized, normalized)


def default_tactile_summary(event: str = "no_contact") -> dict[str, Any]:
    """Return an empty tactile summary with the public schema."""
    return {
        "contact": False,
        "left_contact": False,
        "right_contact": False,
        "normal_force": 0.0,
        "shear_magnitude": 0.0,
        "slip_score": 0.0,
        "contact_balance": 0.0,
        "max_marker_displacement": 0.0,
        "mean_marker_displacement": 0.0,
        "event": event if event in EVENTS else "unknown",
    }


def summarize_tactile_frames(frames: list[TactileFrame]) -> dict[str, Any]:
    """Summarize recent tactile proxy frames.

    Args:
        frames: Recent frames for one target, ordered oldest to newest.

    Returns:
        Dictionary with contact, force, shear, slip score, balance, displacement,
        and a discrete event.
    """
    if not frames:
        return default_tactile_summary()

    latest = frames[-1]
    left_contact = bool(latest.left_contact)
    right_contact = bool(latest.right_contact)
    contact = left_contact or right_contact
    had_contact = any(frame.contact for frame in frames)

    contact_counts = np.asarray([frame.contact_count for frame in frames], dtype=np.float64)
    penetration_depths = np.asarray(
        [max(0.0, frame.penetration_depth) for frame in frames], dtype=np.float64
    )
    contact_count_score = float(np.clip(contact_counts[-1] / 2.0, 0.0, 1.0))
    penetration_score = float(np.clip(penetration_depths[-1] / 0.01, 0.0, 1.0))
    normal_force = float(np.clip(0.7 * contact_count_score + 0.3 * penetration_score, 0.0, 1.0))

    left_ratio = float(np.mean([frame.left_contact for frame in frames]))
    right_ratio = float(np.mean([frame.right_contact for frame in frames]))
    denom = max(left_ratio + right_ratio, 1e-6)
    contact_balance = float(np.clip((right_ratio - left_ratio) / denom, -1.0, 1.0))

    rel_positions = [frame.relative_pos for frame in frames if frame.relative_pos is not None]
    rel_positions = [np.asarray(pos, dtype=np.float64).reshape(3) for pos in rel_positions]
    if len(rel_positions) >= 2:
        rel_array = np.stack(rel_positions, axis=0)
        rel_disp = np.linalg.norm(rel_array - rel_array[0], axis=1)
        rel_step = np.linalg.norm(np.diff(rel_array, axis=0), axis=1)
        max_marker_displacement = float(rel_disp.max())
        mean_marker_displacement = float(rel_step.mean()) if len(rel_step) else 0.0
    else:
        max_marker_displacement = 0.0
        mean_marker_displacement = 0.0

    movement_score = min(1.0, max_marker_displacement / 0.04) if had_contact else 0.0
    one_finger_score = 0.5 if (left_contact ^ right_contact) else 0.0
    contact_loss_score = 1.0 if had_contact and not contact else 0.0
    slip_score = float(np.clip(max(movement_score, one_finger_score, contact_loss_score), 0.0, 1.0))
    shear_magnitude = float(np.clip(max_marker_displacement / 0.04, 0.0, 1.0))

    if had_contact and not contact:
        event = "contact_lost"
    elif not contact:
        event = "no_contact"
    elif slip_score >= 0.6:
        event = "slip_detected"
    elif left_contact ^ right_contact:
        event = "one_finger_contact"
    elif left_contact and right_contact:
        event = "stable_grasp"
    else:
        event = "unknown"

    return {
        "contact": contact,
        "left_contact": left_contact,
        "right_contact": right_contact,
        "normal_force": normal_force,
        "shear_magnitude": shear_magnitude,
        "slip_score": slip_score,
        "contact_balance": contact_balance,
        "max_marker_displacement": max_marker_displacement,
        "mean_marker_displacement": mean_marker_displacement,
        "event": event,
    }


def tactile_event_sequence(frames: list[TactileFrame], window: int = 8) -> list[str]:
    """Return a compact deduplicated tactile event sequence."""
    events: list[str] = []
    for idx in range(len(frames)):
        start = max(0, idx - max(1, window) + 1)
        event = str(summarize_tactile_frames(frames[start : idx + 1])["event"])
        if not events or events[-1] != event:
            events.append(event)
    return events
