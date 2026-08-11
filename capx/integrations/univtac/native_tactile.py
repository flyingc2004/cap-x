"""Native UniVTAC tactile frame buffering and summarization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


EVENTS = {
    "no_contact",
    "one_hand_contact",
    "stable_grasp",
    "slip_detected",
    "contact_lost",
    "unknown",
}


@dataclass
class UniVTACTactileFrame:
    """Single native tactile observation from UniVTAC."""

    step: int
    timestamp: float
    left_depth: np.ndarray | None
    right_depth: np.ndarray | None
    left_marker: np.ndarray | None
    right_marker: np.ndarray | None
    left_pose: np.ndarray | None
    right_pose: np.ndarray | None


class UniVTACTactileBuffer:
    """Small ring buffer for native UniVTAC tactile frames."""

    def __init__(self, maxlen: int = 500) -> None:
        self._frames: deque[UniVTACTactileFrame] = deque(maxlen=maxlen)

    def clear(self) -> None:
        self._frames.clear()

    def append(self, frame: UniVTACTactileFrame) -> None:
        self._frames.append(frame)

    def recent(self, window: int) -> list[UniVTACTactileFrame]:
        window = max(1, int(window))
        return list(self._frames)[-window:]

    def frames(self) -> list[UniVTACTactileFrame]:
        return list(self._frames)


def frame_from_observation(obs: dict[str, Any], *, step: int, timestamp: float) -> UniVTACTactileFrame:
    """Build a tactile frame from a UniVTAC observation dictionary."""
    tactile = obs.get("tactile", {}) if isinstance(obs, dict) else {}
    left = tactile.get("left_tactile", {}) if isinstance(tactile, dict) else {}
    right = tactile.get("right_tactile", {}) if isinstance(tactile, dict) else {}
    return UniVTACTactileFrame(
        step=step,
        timestamp=timestamp,
        left_depth=_to_numpy(left.get("depth")),
        right_depth=_to_numpy(right.get("depth")),
        left_marker=_to_numpy(left.get("marker")),
        right_marker=_to_numpy(right.get("marker")),
        left_pose=_to_numpy(left.get("pose")),
        right_pose=_to_numpy(right.get("pose")),
    )


def summarize_native_tactile(
    frames: list[UniVTACTactileFrame],
    *,
    hand: str = "both",
    depth_far_plane_mm: float | None = None,
    force_full_scale_mm: float = 2.0,
    depth_contact_margin_mm: float = 0.5,
) -> dict[str, Any]:
    """Summarize recent native UniVTAC tactile frames."""
    if not frames:
        return _empty_summary()

    hand = _normalize_hand(hand)
    current = frames[-1]
    metric_kwargs = {
        "depth_far_plane_mm": depth_far_plane_mm,
        "force_full_scale_mm": force_full_scale_mm,
        "depth_contact_margin_mm": depth_contact_margin_mm,
    }
    baseline = frames[0]
    left_metrics = _hand_metrics(
        current.left_depth,
        current.left_marker,
        baseline_marker=baseline.left_marker,
        **metric_kwargs,
    )
    right_metrics = _hand_metrics(
        current.right_depth,
        current.right_marker,
        baseline_marker=baseline.right_marker,
        **metric_kwargs,
    )

    selected = _select_metrics(hand, left_metrics, right_metrics)
    contact = any(m["contact"] for m in selected)
    one_hand_contact = (left_metrics["contact"] + right_metrics["contact"]) == 1

    normal_force = float(np.mean([m["normal_force"] for m in selected])) if selected else 0.0
    contact_area = float(np.mean([m["contact_area"] for m in selected])) if selected else 0.0
    shear_magnitude = float(np.mean([m["shear_magnitude"] for m in selected])) if selected else 0.0
    depth_delta_mm = float(np.mean([m["depth_delta_mm"] for m in selected])) if selected else 0.0
    marker_centroid_displacement = (
        float(np.mean([m["marker_centroid_displacement"] for m in selected]))
        if selected
        else 0.0
    )

    previous_contact = any(
        any(m["contact"] for m in _select_metrics(
            hand,
            _hand_metrics(frame.left_depth, frame.left_marker, **metric_kwargs),
            _hand_metrics(frame.right_depth, frame.right_marker, **metric_kwargs),
        ))
        for frame in frames[:-1]
    )
    contact_lost = previous_contact and not contact

    marker_growth = _marker_growth(frames, hand, metric_kwargs)
    area_change = _contact_area_change(frames, hand, metric_kwargs)
    balance = _contact_balance(left_metrics, right_metrics)
    slip_score = float(np.clip(0.65 * marker_growth + 0.25 * area_change + 0.10 * abs(balance), 0.0, 1.0))
    if contact_lost:
        slip_score = max(slip_score, 0.75)

    if contact_lost:
        event = "contact_lost"
    elif contact and slip_score >= 0.6:
        event = "slip_detected"
    elif hand == "both" and one_hand_contact:
        event = "one_hand_contact"
    elif contact and normal_force >= 0.2 and slip_score < 0.6:
        event = "stable_grasp" if hand == "both" and left_metrics["contact"] and right_metrics["contact"] else "one_hand_contact"
    elif not contact:
        event = "no_contact"
    else:
        event = "unknown"

    return {
        "contact": bool(contact),
        "left_contact": bool(left_metrics["contact"]),
        "right_contact": bool(right_metrics["contact"]),
        "normal_force": normal_force,
        "contact_area": contact_area,
        "depth_delta_mm": depth_delta_mm,
        "shear_magnitude": shear_magnitude,
        "marker_centroid_displacement": marker_centroid_displacement,
        "slip_score": slip_score,
        "contact_balance": balance,
        "left": left_metrics,
        "right": right_metrics,
        "event": event,
    }


def tactile_event_sequence(
    frames: list[UniVTACTactileFrame],
    *,
    hand: str = "both",
    depth_far_plane_mm: float | None = None,
    force_full_scale_mm: float = 2.0,
    depth_contact_margin_mm: float = 0.5,
) -> list[str]:
    """Return a deduplicated sequence of tactile events over recent frames."""
    events: list[str] = []
    for idx in range(len(frames)):
        event = summarize_native_tactile(
            frames[: idx + 1],
            hand=hand,
            depth_far_plane_mm=depth_far_plane_mm,
            force_full_scale_mm=force_full_scale_mm,
            depth_contact_margin_mm=depth_contact_margin_mm,
        )["event"]
        if not events or events[-1] != event:
            events.append(event)
    return events


def _empty_summary() -> dict[str, Any]:
    return {
        "contact": False,
        "left_contact": False,
        "right_contact": False,
        "normal_force": 0.0,
        "contact_area": 0.0,
        "depth_delta_mm": 0.0,
        "shear_magnitude": 0.0,
        "marker_centroid_displacement": 0.0,
        "slip_score": 0.0,
        "contact_balance": 0.0,
        "left": _empty_hand_metrics(),
        "right": _empty_hand_metrics(),
        "event": "no_contact",
    }


def _empty_hand_metrics() -> dict[str, Any]:
    return {
        "contact": False,
        "normal_force": 0.0,
        "contact_area": 0.0,
        "depth_delta_mm": 0.0,
        "depth_min_mm": None,
        "depth_far_plane_mm": None,
        "shear_magnitude": 0.0,
        "marker_mean_displacement": 0.0,
        "marker_max_displacement": 0.0,
        "marker_centroid_displacement": 0.0,
    }


def _hand_metrics(
    depth: np.ndarray | None,
    marker: np.ndarray | None,
    *,
    baseline_marker: np.ndarray | None = None,
    depth_far_plane_mm: float | None = None,
    force_full_scale_mm: float = 2.0,
    depth_contact_margin_mm: float = 0.5,
) -> dict[str, Any]:
    metrics = _empty_hand_metrics()
    depth_stats = _depth_metrics(
        depth,
        far_plane_mm=depth_far_plane_mm,
        contact_margin_mm=depth_contact_margin_mm,
    )
    depth_delta = depth_stats["depth_delta_mm"]
    marker_mean, marker_max = _marker_displacement(marker)
    marker_centroid_displacement = _marker_centroid_displacement(marker, baseline_marker)
    contact_area = depth_stats["contact_area"]
    force_scale = max(float(force_full_scale_mm), 1e-6)
    # This is a normalized compression proxy, not a Newton estimate. A value
    # of 1.0 means the robot-specific adaptive grasp depth has been reached.
    # Contact area remains a separate stability feature and must not make the
    # controller stop before the calibrated compression target.
    normal_force = float(np.clip(depth_delta / force_scale, 0.0, 1.0))
    shear = float(np.clip(marker_mean / 4.0, 0.0, 1.0))
    depth_contact = bool(
        depth_delta >= max(0.0, float(depth_contact_margin_mm))
        and contact_area > 0.0
    )
    # With a calibrated native depth plane, marker motion is a shear/slip
    # signal only. Treating marker motion alone as contact caused false stops
    # while the GelSight depth maps were still at their no-load far plane.
    if depth_far_plane_mm is not None and depth_stats["depth_available"]:
        contact = depth_contact
    else:
        contact = depth_contact or marker_mean >= 0.35
    metrics.update(
        {
            "contact": bool(contact),
            "normal_force": normal_force,
            "contact_area": float(contact_area),
            "depth_delta_mm": float(depth_delta),
            "depth_min_mm": depth_stats["depth_min_mm"],
            "depth_far_plane_mm": depth_stats["depth_far_plane_mm"],
            "shear_magnitude": shear,
            "marker_mean_displacement": float(marker_mean),
            "marker_max_displacement": float(marker_max),
            "marker_centroid_displacement": float(marker_centroid_displacement),
        }
    )
    return metrics


def _depth_metrics(
    depth: np.ndarray | None,
    *,
    far_plane_mm: float | None,
    contact_margin_mm: float,
) -> dict[str, Any]:
    empty = {
        "depth_available": False,
        "depth_delta_mm": 0.0,
        "depth_min_mm": None,
        "depth_far_plane_mm": far_plane_mm,
        "contact_area": 0.0,
    }
    if depth is None or depth.size == 0:
        return empty
    arr = np.asarray(depth, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return empty

    calibrated = far_plane_mm is not None
    far_plane = (
        float(far_plane_mm)
        if calibrated
        else float(np.nanpercentile(valid, 95))
    )
    near = float(np.nanmin(valid) if calibrated else np.nanpercentile(valid, 5))
    depth_delta = max(0.0, far_plane - near)
    if calibrated:
        threshold = far_plane - max(0.0, float(contact_margin_mm))
    else:
        threshold = far_plane - max(0.25, depth_delta * 0.35)
    contact_area = float(np.mean(valid < threshold)) if depth_delta > 1e-6 else 0.0
    return {
        "depth_available": True,
        "depth_delta_mm": depth_delta,
        "depth_min_mm": near,
        "depth_far_plane_mm": far_plane,
        "contact_area": contact_area,
    }


def _marker_displacement(marker: np.ndarray | None) -> tuple[float, float]:
    if marker is None or marker.size == 0:
        return 0.0, 0.0
    arr = np.asarray(marker, dtype=np.float64)
    if arr.shape[-1] < 2:
        return 0.0, 0.0
    # UniVTAC removes the environment dimension before exposing marker data,
    # leaving [initial/current, num_markers, xy]. Synthetic grids may retain
    # one extra marker-grid dimension, so both 3-D and 4-D layouts use axis 0.
    if arr.ndim >= 3 and arr.shape[0] >= 2:
        disp = arr[-1, ..., :2] - arr[0, ..., :2]
    else:
        disp = arr[..., :2]
    mag = np.linalg.norm(disp.reshape(-1, 2), axis=1)
    mag = mag[np.isfinite(mag)]
    if mag.size == 0:
        return 0.0, 0.0
    # UniVTAC marker_motion is pixel-like in live observations; normalize large
    # values by a conservative GelSight image scale while keeping small
    # synthetic/unit-test displacements unchanged.
    if float(np.nanmax(mag)) > 10.0:
        mag = mag / 320.0
    return float(np.mean(mag)), float(np.max(mag))


def _marker_centroid_displacement(
    marker: np.ndarray | None,
    baseline_marker: np.ndarray | None,
) -> float:
    current = _marker_current_points(marker)
    baseline = _marker_current_points(baseline_marker)
    if current is None or baseline is None or current.size == 0 or baseline.size == 0:
        return 0.0
    return float(np.linalg.norm(np.mean(current, axis=0) - np.mean(baseline, axis=0)))


def _marker_current_points(marker: np.ndarray | None) -> np.ndarray | None:
    if marker is None or marker.size == 0:
        return None
    arr = np.asarray(marker, dtype=np.float64)
    if arr.shape[-1] < 2:
        return None
    if arr.ndim >= 3 and arr.shape[0] >= 2:
        points = arr[-1, ..., :2]
    else:
        points = arr[..., :2]
    points = points.reshape(-1, 2)
    return points[np.all(np.isfinite(points), axis=1)]


def _marker_growth(
    frames: list[UniVTACTactileFrame],
    hand: str,
    metric_kwargs: dict[str, Any],
) -> float:
    if len(frames) < 2:
        return 0.0
    first = frames[0]
    last = frames[-1]
    first_metrics = _select_metrics(
        hand,
        _hand_metrics(first.left_depth, first.left_marker, **metric_kwargs),
        _hand_metrics(first.right_depth, first.right_marker, **metric_kwargs),
    )
    last_metrics = _select_metrics(
        hand,
        _hand_metrics(last.left_depth, last.left_marker, **metric_kwargs),
        _hand_metrics(last.right_depth, last.right_marker, **metric_kwargs),
    )
    first_shear = float(np.mean([m["shear_magnitude"] for m in first_metrics])) if first_metrics else 0.0
    last_shear = float(np.mean([m["shear_magnitude"] for m in last_metrics])) if last_metrics else 0.0
    return float(np.clip(last_shear - first_shear, 0.0, 1.0))


def _contact_area_change(
    frames: list[UniVTACTactileFrame],
    hand: str,
    metric_kwargs: dict[str, Any],
) -> float:
    if len(frames) < 2:
        return 0.0
    areas = []
    for frame in frames:
        metrics = _select_metrics(
            hand,
            _hand_metrics(frame.left_depth, frame.left_marker, **metric_kwargs),
            _hand_metrics(frame.right_depth, frame.right_marker, **metric_kwargs),
        )
        areas.append(float(np.mean([m["contact_area"] for m in metrics])) if metrics else 0.0)
    return float(np.clip(max(areas) - areas[-1], 0.0, 1.0))


def _contact_balance(left: dict[str, Any], right: dict[str, Any]) -> float:
    total = float(left["normal_force"] + right["normal_force"])
    if total <= 1e-9:
        return 0.0
    return float((left["normal_force"] - right["normal_force"]) / total)


def _select_metrics(hand: str, left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    if hand == "left":
        return [left]
    if hand == "right":
        return [right]
    return [left, right]


def _normalize_hand(hand: str) -> str:
    value = str(hand).strip().lower()
    if value in {"left", "left_tactile"}:
        return "left"
    if value in {"right", "right_tactile"}:
        return "right"
    if value in {"both", "all", "two", "hands"}:
        return "both"
    raise ValueError("hand must be 'left', 'right', or 'both'")


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
