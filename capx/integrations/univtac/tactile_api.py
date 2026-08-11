"""Native tactile API for CaP-X on UniVTAC."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase

from .native_tactile import summarize_native_tactile, tactile_event_sequence


class UniVTACTactileApi(ApiBase):
    """Expose UniVTAC native tactile observations to CaP-generated code."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)

    def functions(self) -> dict[str, Any]:
        return {
            "get_tactile_summary": self.get_tactile_summary,
            "is_contacting": self.is_contacting,
            "is_slipping": self.is_slipping,
            "is_grasp_stable": self.is_grasp_stable,
            "wait_until_contact": self.wait_until_contact,
            "get_tactile_image": self.get_tactile_image,
            "get_tactile_depth": self.get_tactile_depth,
            "get_marker_motion_summary": self.get_marker_motion_summary,
            "get_recent_tactile_events": self.get_recent_tactile_events,
        }

    def reset_episode(self) -> None:
        """Clear native tactile memory at the start of a trial."""
        if hasattr(self._env, "reset_tactile_buffer"):
            self._env.reset_tactile_buffer()
        elif hasattr(self._env, "tactile_buffer"):
            self._env.tactile_buffer.clear()

    def get_tactile_summary(self, window: int = 20, hand: str = "both") -> dict:
        """Return native UniVTAC tactile summary over recent frames.

        Args:
            window: Number of recent native tactile frames to aggregate.
            hand: Which tactile sensor to inspect: left, right, or both.

        Returns:
            Dictionary with contact, left_contact, right_contact, normal_force,
            contact_area, depth_delta_mm, shear_magnitude, slip_score,
            contact_balance, per-hand metrics, and event. The summary is based
            only on UniVTAC tactile depth and marker outputs.
        """
        _refresh_native_tactile(self._env, data_types=["rgb", "rgb_marker", "marker", "depth", "pose"])
        frames = self._env.tactile_buffer.recent(window)
        summary = summarize_native_tactile(
            frames,
            hand=hand,
            **_native_tactile_calibration(self._env),
        )
        print(
            "[univtac-tactile] "
            f"hand={hand} contact={summary['contact']} "
            f"left={summary['left_contact']} right={summary['right_contact']} "
            f"force={summary['normal_force']:.3f} depth={summary['depth_delta_mm']:.3f}mm "
            f"shear={summary['shear_magnitude']:.3f} "
            f"slip={summary['slip_score']:.3f} event={summary['event']}"
        )
        return summary

    def is_contacting(self, hand: str = "both", threshold: float = 0.2) -> bool:
        """Return whether native tactile sensing indicates contact.

        Args:
            hand: Which tactile sensor to inspect: left, right, or both.
            threshold: Minimum normal_force proxy for reliable contact.

        Returns:
            True if contact is detected and normal_force exceeds threshold.
        """
        summary = self.get_tactile_summary(hand=hand)
        return bool(summary["contact"] and summary["normal_force"] >= threshold)

    def is_slipping(self, hand: str = "both", threshold: float = 0.6) -> bool:
        """Return whether native tactile marker/depth history indicates slip.

        Args:
            hand: Which tactile sensor to inspect: left, right, or both.
            threshold: Slip score threshold in [0, 1].

        Returns:
            True when slip_score is greater than or equal to threshold.
        """
        summary = self.get_tactile_summary(hand=hand)
        return bool(summary["slip_score"] >= threshold)

    def is_grasp_stable(self, threshold: float = 0.5) -> bool:
        """Return whether both native tactile sensors indicate a stable grasp.

        Args:
            threshold: Minimum normal_force proxy for a stable two-hand grasp.

        Returns:
            True when both sensors have contact, force is sufficient, contact is
            balanced, and slip_score is low.
        """
        summary = self.get_tactile_summary(hand="both")
        return bool(
            summary["left_contact"]
            and summary["right_contact"]
            and summary["normal_force"] >= threshold
            and abs(summary["contact_balance"]) <= 0.45
            and summary["slip_score"] < 0.6
        )

    def wait_until_contact(
        self,
        timeout: float = 2.0,
        hand: str = "both",
        threshold: float = 0.2,
    ) -> dict:
        """Wait until native tactile contact is detected or timeout expires.

        Args:
            timeout: Maximum wall-clock seconds to monitor.
            hand: Which tactile sensor to inspect: left, right, or both.
            threshold: Minimum normal_force proxy for reliable contact.

        Returns:
            Last tactile summary at contact or timeout.
        """
        deadline = time.time() + max(0.0, float(timeout))
        summary = self.get_tactile_summary(hand=hand)
        while time.time() < deadline:
            if summary["contact"] and summary["normal_force"] >= threshold:
                return summary
            self._env.wait_steps(1)
            summary = self.get_tactile_summary(hand=hand)
        return summary

    def get_tactile_image(self, hand: str = "left", image_type: str = "rgb_marker") -> np.ndarray:
        """Return a native UniVTAC tactile image.

        Args:
            hand: left or right.
            image_type: rgb or rgb_marker.

        Returns:
            Tactile image as a numpy array from UniVTAC observation.
        """
        if image_type not in {"rgb", "rgb_marker"}:
            raise ValueError("image_type must be 'rgb' or 'rgb_marker'")

        candidates = [image_type]
        candidates.append("rgb" if image_type == "rgb_marker" else "rgb_marker")
        errors: list[str] = []
        for candidate in candidates:
            try:
                hand_obs = self._hand_observation(
                    hand,
                    data_types=[candidate, "marker", "depth", "pose"],
                )
            except KeyError as exc:
                errors.append(str(exc))
                continue
            if candidate in hand_obs:
                if candidate != image_type:
                    print(
                        "[univtac-tactile] "
                        f"image_type={image_type} unavailable for {hand}; using {candidate}",
                        flush=True,
                    )
                return _to_numpy(hand_obs[candidate])
            errors.append(f"{candidate} absent; available={sorted(hand_obs.keys())}")

        raise KeyError(
            f"{image_type} not available for {hand}; tried={candidates}; details={errors}"
        )

    def get_tactile_depth(self, hand: str = "left") -> np.ndarray:
        """Return a native UniVTAC tactile depth / height map.

        Args:
            hand: left or right.

        Returns:
            Depth or height map as a numpy array from UniVTAC observation.
        """
        hand_obs = self._hand_observation(hand, data_types=["depth", "marker", "pose"])
        if "depth" not in hand_obs:
            raise KeyError(f"depth not available for {hand}; available={sorted(hand_obs.keys())}")
        return _to_numpy(hand_obs["depth"])

    def get_marker_motion_summary(self, hand: str = "both", window: int = 20) -> dict:
        """Return compact marker motion statistics from native UniVTAC tactile data.

        Args:
            hand: Which tactile sensor to inspect: left, right, or both.
            window: Number of recent tactile frames to aggregate.

        Returns:
            Dictionary with marker displacement statistics and shear magnitude.
        """
        summary = self.get_tactile_summary(window=window, hand=hand)
        left_mean = summary["left"]["marker_mean_displacement"]
        right_mean = summary["right"]["marker_mean_displacement"]
        left_max = summary["left"]["marker_max_displacement"]
        right_max = summary["right"]["marker_max_displacement"]
        return {
            "hand": hand,
            "shear_magnitude": summary["shear_magnitude"],
            "mean_displacement": max(left_mean, right_mean),
            "max_displacement": max(left_max, right_max),
            "left_marker_mean_displacement": summary["left"]["marker_mean_displacement"],
            "right_marker_mean_displacement": summary["right"]["marker_mean_displacement"],
            "left_marker_max_displacement": summary["left"]["marker_max_displacement"],
            "right_marker_max_displacement": summary["right"]["marker_max_displacement"],
        }

    def get_recent_tactile_events(self, window: int = 50) -> list[str]:
        """Return recent native tactile event sequence.

        Args:
            window: Number of recent native tactile frames to inspect.

        Returns:
            Deduplicated event list such as no_contact, one_hand_contact,
            stable_grasp, slip_detected, or contact_lost.
        """
        _refresh_native_tactile(self._env, data_types=["rgb", "rgb_marker", "marker", "depth", "pose"])
        events = tactile_event_sequence(
            self._env.tactile_buffer.recent(window),
            **_native_tactile_calibration(self._env),
        )
        print(f"[univtac-tactile] recent_events={events}")
        return events
    def _hand_observation(
        self,
        hand: str,
        *,
        data_types: list[str] | None = None,
    ) -> dict[str, Any]:
        _refresh_native_tactile(self._env, data_types=data_types or ["marker", "depth", "pose"])
        raw = self._env.current_raw_observation()
        key = _hand_key(hand)
        tactile = raw.get("tactile", {})
        if key not in tactile:
            alternate = _find_hand_key(tactile, hand)
            if alternate is None:
                available = sorted(tactile.keys()) if isinstance(tactile, dict) else []
                raise KeyError(
                    f"{key} not available in UniVTAC tactile observation; available={available}"
                )
            key = alternate
        return tactile[key]


def _native_tactile_calibration(env: BaseEnv) -> dict[str, float]:
    calibration_fn = getattr(env, "get_native_tactile_calibration", None)
    if not callable(calibration_fn):
        return {}
    return dict(calibration_fn())


def _hand_key(hand: str) -> str:
    value = str(hand).strip().lower()
    if value in {"left", "left_tactile"}:
        return "left_tactile"
    if value in {"right", "right_tactile"}:
        return "right_tactile"
    raise ValueError("hand must be 'left' or 'right'")


def _find_hand_key(tactile: Any, hand: str) -> str | None:
    if not isinstance(tactile, dict):
        return None
    prefix = str(hand).strip().lower().split("_", maxsplit=1)[0]
    for key in sorted(tactile.keys()):
        if str(key).lower().startswith(prefix):
            return str(key)
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _refresh_native_tactile(env: BaseEnv, data_types: list[str] | None = None) -> None:
    if hasattr(env, "refresh_native_observation"):
        env.refresh_native_observation(
            include_camera=False,
            include_tactile=True,
            tactile_data_types=data_types,
        )
    else:
        env.get_observation()
