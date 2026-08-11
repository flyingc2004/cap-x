"""Backend-independent tactile feedback control for gripper open/close."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class AdaptiveGripperConfig:
    """Normalized controller calibration and tactile decision thresholds."""

    coarse_step: float
    fine_step: float
    contact_debounce_frames: int = 2
    stable_debounce_frames: int = 3
    release_debounce_frames: int = 2
    settle_steps_per_command: int = 1
    contact_force_threshold: float = 0.08
    contact_balance_threshold: float = 0.45
    slip_threshold: float = 0.60
    one_sided_force_limit: float = 0.90
    max_qpos: float = 1.0

    def __post_init__(self) -> None:
        if self.coarse_step <= 0.0 or self.fine_step <= 0.0:
            raise ValueError("adaptive gripper steps must be positive")
        if self.fine_step > self.coarse_step:
            raise ValueError("fine_step must not exceed coarse_step")
        for field_name in (
            "contact_debounce_frames",
            "stable_debounce_frames",
            "release_debounce_frames",
            "settle_steps_per_command",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be at least 1")


class TactileAdaptiveGripperController:
    """Close and open a gripper using native tactile feedback.

    The controller intentionally knows nothing about robot arms, tasks, rewards,
    or simulator internals. Width is normalized to ``[0, 1]`` where zero is
    closed and one is fully open.
    """

    def __init__(
        self,
        *,
        get_width: Callable[[], float],
        command_width: Callable[[float, int], Any],
        read_tactile_summary: Callable[[], dict[str, Any]],
        config: AdaptiveGripperConfig,
    ) -> None:
        self._get_width = get_width
        self._command_width = command_width
        self._read_tactile_summary = read_tactile_summary
        self.config = config
        self.trace: list[dict[str, Any]] = []

    def close(self, *, target_force: float = 0.35, max_steps: int = 80) -> dict[str, Any]:
        """Close until balanced two-sided contact is stable or failure is clear."""
        target_force = float(np.clip(target_force, 0.0, 1.0))
        max_steps = max(1, int(max_steps))
        contact_frames = 0
        stable_frames = 0
        one_sided_high_frames = 0
        ever_contact = False
        phase = "coarse_close"

        for iteration in range(max_steps):
            width = self._width()
            summary = self._summary()
            contact = bool(summary["contact"])
            both_contact = bool(summary["left_contact"] and summary["right_contact"])
            ever_contact = ever_contact or contact
            contact_frames = contact_frames + 1 if contact else 0
            left_force = float(summary["left"].get("normal_force", 0.0))
            right_force = float(summary["right"].get("normal_force", 0.0))

            stable_candidate = bool(
                both_contact
                and min(left_force, right_force) >= target_force
                and abs(summary["contact_balance"]) <= self.config.contact_balance_threshold
                and summary["slip_score"] < self.config.slip_threshold
            )
            stable_frames = stable_frames + 1 if stable_candidate else 0

            hand_force = max(left_force, right_force)
            one_sided_high = bool(contact and not both_contact and hand_force >= self.config.one_sided_force_limit)
            one_sided_high_frames = one_sided_high_frames + 1 if one_sided_high else 0

            if stable_candidate:
                phase = "stable_confirm"
            elif contact:
                phase = (
                    "fine_close"
                    if contact_frames >= self.config.contact_debounce_frames
                    else "contact_debounce"
                )
            elif ever_contact:
                phase = "contact_lost"
            else:
                phase = "coarse_close"
            self._append_trace("close", iteration, phase, width, summary)

            if stable_frames >= self.config.stable_debounce_frames:
                return self._close_result(True, "stable_grasp", iteration + 1, summary)
            if one_sided_high_frames >= self.config.contact_debounce_frames:
                return self._close_result(False, "one_sided_high_force", iteration + 1, summary)
            if width <= 1e-6:
                if not ever_contact:
                    reason = "missed_grasp"
                elif not both_contact:
                    reason = "one_sided_contact" if contact else "contact_lost"
                else:
                    reason = "unstable_at_min_width"
                return self._close_result(False, reason, iteration + 1, summary)

            # Freeze width while confirming stability so debounce cannot add
            # unwanted squeeze. Any contact switches the next command to fine.
            if stable_candidate:
                target_width = width
            else:
                step = self.config.fine_step if (contact or ever_contact) else self.config.coarse_step
                target_width = max(0.0, width - step)
            self._command_width(target_width, self.config.settle_steps_per_command)

        summary = self._summary()
        reason = "contact_lost" if ever_contact and not summary["contact"] else "max_steps"
        self._append_trace("close", max_steps, "timeout", self._width(), summary)
        return self._close_result(False, reason, max_steps, summary)

    def open(self, *, target_width: float = 1.0, max_steps: int = 80) -> dict[str, Any]:
        """Release gently while touching, then open coarsely after contact loss."""
        target_width = float(np.clip(target_width, 0.0, 1.0))
        max_steps = max(1, int(max_steps))
        release_frames = 0
        ever_contact = False

        for iteration in range(max_steps):
            width = self._width()
            summary = self._summary()
            contact = bool(summary["contact"])
            ever_contact = ever_contact or contact
            release_frames = 0 if contact else release_frames + 1

            if contact:
                phase = "fine_release"
            elif ever_contact and release_frames < self.config.release_debounce_frames:
                phase = "release_confirm"
            else:
                phase = "coarse_open"
            self._append_trace("open", iteration, phase, width, summary)

            if width >= target_width - 1e-6:
                released = not contact
                reason = "target_width" if released else "contact_at_target_width"
                return self._open_result(released, reason, iteration + 1, summary, target_width)

            step = self.config.fine_step if phase in {"fine_release", "release_confirm"} else self.config.coarse_step
            self._command_width(
                min(target_width, width + step),
                self.config.settle_steps_per_command,
            )

        summary = self._summary()
        self._append_trace("open", max_steps, "timeout", self._width(), summary)
        return self._open_result(False, "max_steps", max_steps, summary, target_width)

    def _width(self) -> float:
        return float(np.clip(self._get_width(), 0.0, 1.0))

    def _summary(self) -> dict[str, Any]:
        raw = dict(self._read_tactile_summary() or {})
        left = dict(raw.get("left") or {})
        right = dict(raw.get("right") or {})
        raw_left_contact = bool(raw.get("left_contact", left.get("contact", False)))
        raw_right_contact = bool(raw.get("right_contact", right.get("contact", False)))
        left_force = float(left.get("normal_force", 0.0))
        right_force = float(right.get("normal_force", 0.0))
        # Marker displacement is useful for shear/slip, but live UniVTAC data
        # may contain marker-only motion before load-bearing contact. Gripper
        # servo phases therefore require native compression force as well.
        left_contact = bool(
            raw_left_contact and left_force >= self.config.contact_force_threshold
        )
        right_contact = bool(
            raw_right_contact and right_force >= self.config.contact_force_threshold
        )
        return {
            "contact": bool(left_contact or right_contact),
            "left_contact": left_contact,
            "right_contact": right_contact,
            "raw_contact": bool(raw.get("contact", raw_left_contact or raw_right_contact)),
            "raw_left_contact": raw_left_contact,
            "raw_right_contact": raw_right_contact,
            "normal_force": float(raw.get("normal_force", 0.0)),
            "contact_area": float(raw.get("contact_area", 0.0)),
            "depth_delta_mm": float(raw.get("depth_delta_mm", 0.0)),
            "contact_balance": float(raw.get("contact_balance", 0.0)),
            "slip_score": float(raw.get("slip_score", 0.0)),
            "event": str(raw.get("event", "unknown")),
            "left": left,
            "right": right,
        }

    def _append_trace(
        self,
        operation: str,
        iteration: int,
        phase: str,
        width: float,
        summary: dict[str, Any],
    ) -> None:
        self.trace.append(
            {
                "operation": operation,
                "iteration": int(iteration),
                "phase": phase,
                "width": float(width),
                "qpos": float(width * self.config.max_qpos),
                "contact": bool(summary["contact"]),
                "left_contact": bool(summary["left_contact"]),
                "right_contact": bool(summary["right_contact"]),
                "normal_force": float(summary["normal_force"]),
                "left_normal_force": float(summary["left"].get("normal_force", 0.0)),
                "right_normal_force": float(summary["right"].get("normal_force", 0.0)),
                "contact_area": float(summary["contact_area"]),
                "depth_delta_mm": float(summary["depth_delta_mm"]),
                "left_depth_min_mm": summary["left"].get("depth_min_mm"),
                "right_depth_min_mm": summary["right"].get("depth_min_mm"),
                "left_depth_delta_mm": float(
                    summary["left"].get("depth_delta_mm", 0.0)
                ),
                "right_depth_delta_mm": float(
                    summary["right"].get("depth_delta_mm", 0.0)
                ),
                "contact_balance": float(summary["contact_balance"]),
                "slip_score": float(summary["slip_score"]),
                "event": str(summary["event"]),
                "raw_contact": bool(summary["raw_contact"]),
                "raw_left_contact": bool(summary["raw_left_contact"]),
                "raw_right_contact": bool(summary["raw_right_contact"]),
            }
        )

    def _close_result(
        self,
        stable: bool,
        reason: str,
        steps: int,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "ok": bool(stable),
            "stable": bool(stable),
            "reason": reason,
            "steps": int(steps),
            "width": self._width(),
            "normal_force": float(summary["normal_force"]),
            "contact": bool(summary["contact"]),
            "left_contact": bool(summary["left_contact"]),
            "right_contact": bool(summary["right_contact"]),
            "slip_score": float(summary["slip_score"]),
        }

    def _open_result(
        self,
        released: bool,
        reason: str,
        steps: int,
        summary: dict[str, Any],
        target_width: float,
    ) -> dict[str, Any]:
        return {
            "ok": bool(released),
            "released": bool(released),
            "reason": reason,
            "steps": int(steps),
            "width": self._width(),
            "target_width": float(target_width),
            "contact": bool(summary["contact"]),
        }
