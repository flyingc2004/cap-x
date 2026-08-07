"""Control API for CaP-X running directly on UniVTAC tasks."""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase


class UniVTACControlApi(ApiBase):
    """Robot control API backed by UniVTAC ``Task.take_action``."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)

    def functions(self) -> dict[str, Any]:
        return {
            "get_task_instruction": self.get_task_instruction,
            "get_robot_state": self.get_robot_state,
            "get_observation_summary": self.get_observation_summary,
            "move_delta_ee": self.move_delta_ee,
            "move_ee": self.move_ee,
            "move_qpos": self.move_qpos,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "wait_steps": self.wait_steps,
            "get_step_status": self.get_step_status,
        }

    def get_task_instruction(self) -> str:
        """Return the current UniVTAC task instruction.

        Returns:
            Task instruction string from the UniVTAC task, or a fallback task
            name if the environment has no natural-language instruction.
        """
        return self._env.get_task_instruction()

    def get_robot_state(self) -> dict:
        """Return compact robot state from UniVTAC observation.

        Returns:
            JSON-serializable dictionary with stable keys:
            ``ee_pos`` (XYZ list), ``ee_quat`` (WXYZ quaternion),
            ``ee_pose`` (XYZ+WXYZ), ``joint``/``qpos`` (robot joint list), and
            ``gripper_qpos`` when available. The raw UniVTAC ``ee`` key is also
            kept for compatibility.
        """
        return self._env.get_robot_state()

    def get_observation_summary(self) -> dict:
        """Return compact non-reward observation metadata.

        Returns:
            Dictionary with task name, instruction, step counters, visible camera
            names, tactile hand names, and optionally scene object names. This
            does not include reward, task success, or trial id.
        """
        obs = self._env.current_raw_observation()
        camera_names = sorted(obs.get("observation", {}).keys())
        tactile_names = sorted(obs.get("tactile", {}).keys())
        actor_names = sorted(self._env.get_actor_poses().keys())
        status = self._env.get_status()
        return {
            "task": status["task"],
            "instruction": status["instruction"],
            "step": status["step"],
            "action_count": status["action_count"],
            "max_steps": status["max_steps"],
            "cameras": camera_names,
            "tactile_sensors": tactile_names,
            "objects": actor_names,
        }

    def move_delta_ee(
        self,
        delta_xyz: list[float],
        delta_rpy: list[float] | None = None,
        delta_gripper: float = 0.0,
    ) -> dict:
        """Move the end effector by a small delta in world coordinates.

        Args:
            delta_xyz: XYZ displacement in meters, length 3.
            delta_rpy: Optional roll/pitch/yaw displacement in radians, length 3.
            delta_gripper: Gripper delta in UniVTAC qpos units. Negative closes,
                positive opens.

        Returns:
            Action result dictionary with ok, step, action_count, and message.
        """
        xyz = np.clip(_vec(delta_xyz, 3, "delta_xyz"), -0.06, 0.06)
        state = self._env.get_robot_state()
        ee_pos = np.asarray(state.get("ee_pos", []), dtype=np.float32).flatten()
        if ee_pos.size >= 3:
            min_safe_z = 0.10
            requested_z = float(ee_pos[2] + xyz[2])
            if requested_z < min_safe_z:
                xyz[2] = np.float32(min_safe_z - float(ee_pos[2]))
        rpy = np.zeros(3, dtype=np.float32) if delta_rpy is None else _vec(delta_rpy, 3, "delta_rpy")
        rpy = np.clip(rpy, -0.35, 0.35)
        action = np.concatenate([xyz, rpy, [float(np.clip(delta_gripper, -0.02, 0.02))]])
        return self._env.take_action(action, action_type="delta_ee")

    def move_ee(
        self,
        position: list[float],
        quaternion: list[float],
        gripper_width: float | None = None,
    ) -> dict:
        """Move to an absolute end-effector pose.

        Args:
            position: XYZ position in UniVTAC robot/world coordinates, length 3.
            quaternion: WXYZ orientation quaternion, length 4.
            gripper_width: Optional normalized gripper opening, where 0 is
                closed and 1 is open.

        Returns:
            Action result dictionary with ok, step, action_count, and message.
        """
        pos = _vec(position, 3, "position")
        quat = _vec(quaternion, 4, "quaternion")
        if gripper_width is None:
            current = self._current_gripper_width()
        else:
            current = _width_to_qpos(gripper_width, self._env)
        action = np.concatenate([pos, quat, [current]])
        return self._env.take_action(action, action_type="ee")

    def move_qpos(self, qpos: list[float], gripper_width: float | None = None) -> dict:
        """Move to absolute robot joint positions.

        Args:
            qpos: Arm joint vector. Length 7 for arm-only or length 8 including
                gripper qpos.
            gripper_width: Optional normalized gripper opening, where 0 is
                closed and 1 is open. Used when qpos has length 7.

        Returns:
            Action result dictionary with ok, step, action_count, and message.
        """
        arr = np.asarray(qpos, dtype=np.float32).flatten()
        if arr.size == 7:
            width = self._current_gripper_width() if gripper_width is None else _width_to_qpos(gripper_width, self._env)
            arr = np.concatenate([arr, [width]])
        if arr.size != 8:
            raise ValueError("qpos must have length 7 or 8")
        return self._env.take_action(arr, action_type="qpos")

    def open_gripper(self, width: float = 1.0) -> dict:
        """Open the gripper to a normalized width.

        Args:
            width: Normalized gripper opening in [0, 1], where 1 is fully open.

        Returns:
            Action result dictionary with ok, step, action_count, and message.
        """
        return self._move_gripper(width)

    def close_gripper(self, width: float = 0.0) -> dict:
        """Close the gripper to a normalized width.

        Args:
            width: Normalized gripper opening in [0, 1], where 0 is closed.

        Returns:
            Action result dictionary with ok, step, action_count, and message.
        """
        return self._move_gripper(width)

    def wait_steps(self, n: int = 1) -> dict:
        """Advance the UniVTAC simulation without changing the command.

        Args:
            n: Number of simulation steps to wait.

        Returns:
            Action result dictionary with ok, step, action_count, and message.
        """
        return self._env.wait_steps(n)

    def get_step_status(self) -> dict:
        """Return current execution status without benchmark success or reward.

        Returns:
            Dictionary with task name, instruction, step counters, action count,
            max steps, elapsed time, plan_success, early_stop, and last action.
        """
        return self._env.get_status()

    def _move_gripper(self, width: float) -> dict:
        robot = self._env.get_robot_state()
        joint = robot.get("joint", [])
        if len(joint) < 7:
            state = self._env.current_raw_observation().get("embodiment", {})
            joint = state.get("joint", [])
        arr = np.asarray(joint, dtype=np.float32).flatten()
        if arr.size >= 7:
            arm = arr[:7]
        else:
            arm = np.zeros(7, dtype=np.float32)
        return self.move_qpos(arm.tolist(), gripper_width=width)

    def _current_gripper_width(self) -> float:
        robot = self._env.get_robot_state()
        joint = np.asarray(robot.get("joint", []), dtype=np.float32).flatten()
        if joint.size >= 8:
            return float(joint[7])
        return _width_to_qpos(1.0, self._env)


def _vec(value: list[float], length: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).flatten()
    if arr.size != length:
        raise ValueError(f"{name} must have length {length}")
    return arr


def _width_to_qpos(width: float, env: Any) -> float:
    clipped = float(np.clip(width, 0.0, 1.0))
    task = getattr(env, "task", None)
    robot_manager = getattr(task, "_robot_manager", None)
    max_qpos = float(getattr(robot_manager, "gripper_max_qpos", 0.039))
    return clipped * max_qpos
