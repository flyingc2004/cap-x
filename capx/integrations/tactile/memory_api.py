"""CaP API exposing compact tactile proxy state."""

from __future__ import annotations

import time
import os
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase

from .ring_buffer import TactileFrame, TactileRingBuffer
from .summarizer import (
    normalize_target,
    summarize_tactile_frames,
    tactile_event_sequence,
    target_geom_name,
)

LEFT_PAD_GEOM = "gripper0_right_finger1_pad_collision"
RIGHT_PAD_GEOM = "gripper0_right_finger2_pad_collision"
DEFAULT_TARGETS = ("cubeA", "cubeB")


class TactileMemoryApi(ApiBase):
    """Tactile proxy API backed by MuJoCo contacts and object motion."""

    def __init__(self, env: BaseEnv, buffer_size: int = 500) -> None:
        super().__init__(env)
        self._buffer = TactileRingBuffer(maxlen=buffer_size)
        self._last_recorded_sim_step: int | None = None
        self._last_gripper_width: float | None = None
        self._install_step_observer()

    def functions(self) -> dict[str, Any]:
        functions = {
            "get_tactile_summary": self.get_tactile_summary,
            "is_contacting": self.is_contacting,
            "is_slipping": self.is_slipping,
            "is_grasp_stable": self.is_grasp_stable,
            "wait_until_contact": self.wait_until_contact,
            "get_recent_tactile_events": self.get_recent_tactile_events,
        }
        if os.getenv("CAPX_TACTILE_STRATEGY_MEMORY_ENABLED", "0") == "1":
            functions["retrieve_tactile_strategies"] = self.retrieve_tactile_strategies
        return functions

    def reset_episode(self) -> None:
        """Clear tactile state at the start of a new trial."""
        self._buffer.clear()
        self._last_recorded_sim_step = None
        self._last_gripper_width = None
        self._record_current_step(force=True)

    def export_tactile_frames(self) -> list[dict[str, Any]]:
        """Return recorded tactile proxy frames as JSON-serializable records."""
        records: list[dict[str, Any]] = []
        for frame in self._buffer.frames():
            records.append(
                {
                    "sim_step": frame.sim_step,
                    "timestamp": frame.timestamp,
                    "target": frame.target,
                    "left_contact": frame.left_contact,
                    "right_contact": frame.right_contact,
                    "contact_count": frame.contact_count,
                    "penetration_depth": frame.penetration_depth,
                    "gripper_width": frame.gripper_width,
                    "gripper_velocity": frame.gripper_velocity,
                    "gripper_pos": _array_to_list(frame.gripper_pos),
                    "target_pos": _array_to_list(frame.target_pos),
                }
            )
        return records

    def get_tactile_summary(self, window: int = 20, target: str = "red cube") -> dict:
        """Return a compact tactile summary over the recent window.

        Args:
            window: Number of recent tactile frames to aggregate.
            target: Object to query. Supported aliases are "red cube", "primary",
                "cubeA", "green cube", "secondary", and "cubeB".

        Returns:
            A dictionary with contact, left_contact, right_contact, normal_force,
            shear_magnitude, slip_score, contact_balance, max_marker_displacement,
            mean_marker_displacement, and event. Events are no_contact,
            one_finger_contact, stable_grasp, slip_detected, contact_lost, or unknown.
        """
        normalized = normalize_target(target)
        self._record_current_step()
        frames = self._buffer.recent(window=max(1, int(window)), target=normalized)
        summary = summarize_tactile_frames(frames)
        print(
            "[tactile] "
            f"target={target} contact={summary['contact']} "
            f"left={summary['left_contact']} right={summary['right_contact']} "
            f"force={summary['normal_force']:.3f} slip={summary['slip_score']:.3f} "
            f"balance={summary['contact_balance']:.3f} event={summary['event']}"
        )
        return summary

    def is_contacting(self, threshold: float = 0.2, target: str = "red cube") -> bool:
        """Return whether the gripper is currently in tactile contact with an object.

        Args:
            threshold: Minimum normal_force proxy to count as reliable contact.
            target: Object to query. Defaults to the red/primary cube.

        Returns:
            True if contact is present and the normal_force proxy exceeds threshold.
        """
        summary = self.get_tactile_summary(target=target)
        return bool(summary["contact"] and summary["normal_force"] >= threshold)

    def is_slipping(self, threshold: float = 0.6, target: str = "red cube") -> bool:
        """Return whether recent tactile motion indicates object slip.

        Args:
            threshold: Slip score threshold in [0, 1].
            target: Object to query. Defaults to the red/primary cube.

        Returns:
            True if slip_score is greater than or equal to threshold.
        """
        summary = self.get_tactile_summary(target=target)
        return bool(summary["slip_score"] >= threshold)

    def is_grasp_stable(self, target: str = "red cube") -> bool:
        """Return whether contact is present, balanced, and not slipping.

        Args:
            target: Object to query. Defaults to the red/primary cube.

        Returns:
            True when both fingers are in contact, normal_force is sufficient,
            contact_balance is near zero, and slip_score is low.
        """
        summary = self.get_tactile_summary(target=target)
        return bool(
            summary["contact"]
            and summary["left_contact"]
            and summary["right_contact"]
            and summary["normal_force"] >= 0.2
            and abs(summary["contact_balance"]) <= 0.35
            and summary["slip_score"] < 0.6
        )

    def wait_until_contact(
        self, timeout: float = 2.0, threshold: float = 0.2, target: str = "red cube"
    ) -> dict:
        """Step or monitor the environment until tactile contact is detected.

        Args:
            timeout: Maximum wall-clock seconds to wait.
            threshold: Minimum normal_force proxy to count as reliable contact.
            target: Object to query. Defaults to the red/primary cube.

        Returns:
            The tactile summary at contact or timeout.
        """
        deadline = time.time() + max(0.0, float(timeout))
        summary = self.get_tactile_summary(target=target)
        while time.time() < deadline:
            if summary["contact"] and summary["normal_force"] >= threshold:
                return summary
            if hasattr(self._env, "_step_once"):
                self._env._step_once()
            else:
                time.sleep(0.01)
            summary = self.get_tactile_summary(target=target)
        return summary

    def get_recent_tactile_events(self, window: int = 50) -> list[str]:
        """Return recent discrete tactile events.

        Args:
            window: Number of recent tactile frames to inspect.

        Returns:
            Deduplicated event list such as no_contact, stable_grasp,
            slip_detected, one_finger_contact, or contact_lost.
        """
        self._record_current_step()
        frames = self._buffer.recent(window=max(1, int(window)), target="cubeA")
        events = tactile_event_sequence(frames)
        print(f"[tactile] recent_events={events}")
        return events

    def retrieve_tactile_strategies(
        self,
        failure_type: str | None = None,
        target: str = "red cube",
        top_k: int = 3,
    ) -> list[dict]:
        """Retrieve learned tactile strategy hints from previous trials.

        Args:
            failure_type: Optional tactile failure type such as missed_grasp,
                off_center_grasp, slip_during_lift, placement_error, or
                unknown_tactile_failure. Leave unset to retrieve general hints.
            target: Object to retrieve strategies for. Defaults to the red cube.
            top_k: Maximum number of strategy records to return.

        Returns:
            A list of read-only strategy dictionaries. These are historical hints
            only; generated code must not treat reward, success, or task_completed
            as current runtime observations.
        """
        from .strategy_memory import TactileStrategyMemory

        records = TactileStrategyMemory().retrieve(
            failure_type=failure_type,
            target=target,
            top_k=max(0, int(top_k)),
        )
        print(
            "[tactile-memory] "
            f"retrieved={len(records)} failure_type={failure_type} target={target}"
        )
        return records

    def _install_step_observer(self) -> None:
        if hasattr(self._env, "add_step_observer"):
            self._env.add_step_observer(self._record_current_step)

    def _record_current_step(self, force: bool = False) -> None:
        if not hasattr(self._env, "robosuite_env"):
            return

        sim_step = int(getattr(self._env, "_sim_step_count", 0))
        if (
            self._last_recorded_sim_step is not None
            and sim_step < self._last_recorded_sim_step
        ):
            self._buffer.clear()
            self._last_gripper_width = None
        if not force and self._last_recorded_sim_step == sim_step:
            return

        for target in DEFAULT_TARGETS:
            self._buffer.append(self._read_frame(target=target, sim_step=sim_step))
        self._last_recorded_sim_step = sim_step

    def _read_frame(self, target: str, sim_step: int) -> TactileFrame:
        sim = self._env.robosuite_env.sim
        model = sim.model
        data = sim.data
        left_id = self._geom_id(model, LEFT_PAD_GEOM)
        right_id = self._geom_id(model, RIGHT_PAD_GEOM)
        target_geom = target_geom_name(target)
        target_id = self._geom_id(model, target_geom)

        left_contact = False
        right_contact = False
        contact_count = 0
        penetration_depth = 0.0
        if target_id is not None:
            for idx in range(int(data.ncon)):
                contact = data.contact[idx]
                pair = {int(contact.geom1), int(contact.geom2)}
                if left_id is not None and left_id in pair and target_id in pair:
                    left_contact = True
                    contact_count += 1
                    penetration_depth += max(0.0, -float(contact.dist))
                if right_id is not None and right_id in pair and target_id in pair:
                    right_contact = True
                    contact_count += 1
                    penetration_depth += max(0.0, -float(contact.dist))

        gripper_width = float(getattr(self._env, "_gripper_fraction", 0.0))
        if self._last_gripper_width is None:
            gripper_velocity = 0.0
        else:
            gripper_velocity = gripper_width - self._last_gripper_width
        self._last_gripper_width = gripper_width

        gripper_pos = None
        if hasattr(self._env, "gripper_link_wxyz_xyz"):
            gripper_pos = np.asarray(self._env.gripper_link_wxyz_xyz[-3:], dtype=np.float64).copy()

        target_pos = None
        if target_id is not None:
            target_pos = np.asarray(data.geom_xpos[target_id], dtype=np.float64).copy()

        return TactileFrame(
            sim_step=sim_step,
            timestamp=time.time(),
            target=normalize_target(target),
            left_contact=left_contact,
            right_contact=right_contact,
            contact_count=contact_count,
            penetration_depth=penetration_depth,
            gripper_width=gripper_width,
            gripper_velocity=gripper_velocity,
            gripper_pos=gripper_pos,
            target_pos=target_pos,
        )

    @staticmethod
    def _geom_id(model: Any, name: str) -> int | None:
        try:
            return int(model.geom_name2id(name))
        except Exception:
            return None


def _array_to_list(value: np.ndarray | None) -> list[float] | None:
    if value is None:
        return None
    return [float(x) for x in np.asarray(value, dtype=np.float64).reshape(-1)]
