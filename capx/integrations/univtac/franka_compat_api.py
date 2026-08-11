"""CaP-style Franka API compatibility layer for UniVTAC.

This adapter preserves the original high-level Franka control surface used by
CaP-X prompt templates while translating motions into UniVTAC native actions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import BaseEnv
from capx.integrations.base_api import ApiBase
from capx.integrations.tactile.adaptive_gripper import (
    AdaptiveGripperConfig,
    TactileAdaptiveGripperController,
)
from capx.integrations.univtac.native_tactile import summarize_native_tactile


class UniVTACFrankaCompatApi(ApiBase):
    """Franka control API compatible with the original CaP-X surface."""

    def __init__(
        self,
        env: BaseEnv,
        *,
        min_safe_z: float = 0.10,
        max_delta_xyz: float = 0.04,
        max_delta_rpy: float = 0.25,
        max_delta_gripper: float = 0.02,
        default_z_approach: float = 0.10,
        release_hover_height: float = 0.10,
        preserve_landmark_orientation: bool = True,
        use_task_place_actor_for_landmarks: bool = False,
        placement_xy_tolerance: float = 0.08,
        placement_time_dilation_factor: float = 0.5,
        placement_pre_dis: float = 0.0,
        placement_dis: float = 0.0,
        gripper_settle_steps: int = 10,
        use_task_grasp_actor_for_prism: bool = True,
        use_task_grasp_actor_for_objects: bool | None = None,
        grasp_xy_tolerance: float = 0.08,
        grasp_z_tolerance: float = 0.04,
        grasp_pre_dis: float = 0.04,
        grasp_dis: float = 0.0,
        grasp_height: float = 0.04,
        lift_after_close: bool = True,
        lift_after_close_z: float = 0.05,
        close_gripper_qpos: float = 0.007,
        open_gripper_width: float = 1.0,
        object_pose_names: dict[str, str] | None = None,
    ) -> None:
        super().__init__(env)
        cfg = self._runtime_config()
        self.min_safe_z = float(cfg.get("min_safe_z", min_safe_z))
        self.max_delta_xyz = float(cfg.get("max_delta_xyz", max_delta_xyz))
        self.max_delta_rpy = float(cfg.get("max_delta_rpy", max_delta_rpy))
        self.max_delta_gripper = float(cfg.get("max_delta_gripper", max_delta_gripper))
        self.default_z_approach = float(cfg.get("default_z_approach", default_z_approach))
        self.release_hover_height = float(cfg.get("release_hover_height", release_hover_height))
        self.preserve_landmark_orientation = bool(
            cfg.get("preserve_landmark_orientation", preserve_landmark_orientation)
        )
        self.use_task_place_actor_for_landmarks = bool(
            cfg.get("use_task_place_actor_for_landmarks", use_task_place_actor_for_landmarks)
        )
        self.placement_xy_tolerance = float(cfg.get("placement_xy_tolerance", placement_xy_tolerance))
        self.placement_time_dilation_factor = float(
            cfg.get("placement_time_dilation_factor", placement_time_dilation_factor)
        )
        self.placement_pre_dis = float(cfg.get("placement_pre_dis", placement_pre_dis))
        self.placement_dis = float(cfg.get("placement_dis", placement_dis))
        self.gripper_settle_steps = int(cfg.get("gripper_settle_steps", gripper_settle_steps))
        self.use_task_grasp_actor_for_prism = bool(
            cfg.get("use_task_grasp_actor_for_prism", use_task_grasp_actor_for_prism)
        )
        generic_grasp_default = (
            self.use_task_grasp_actor_for_prism
            if use_task_grasp_actor_for_objects is None
            else bool(use_task_grasp_actor_for_objects)
        )
        self.use_task_grasp_actor_for_objects = bool(
            cfg.get("use_task_grasp_actor_for_objects", generic_grasp_default)
        )
        self.grasp_xy_tolerance = float(cfg.get("grasp_xy_tolerance", grasp_xy_tolerance))
        self.grasp_z_tolerance = float(cfg.get("grasp_z_tolerance", grasp_z_tolerance))
        self.grasp_pre_dis = float(cfg.get("grasp_pre_dis", grasp_pre_dis))
        self.grasp_dis = float(cfg.get("grasp_dis", grasp_dis))
        self.grasp_height = float(cfg.get("grasp_height", grasp_height))
        self.lift_after_close = bool(cfg.get("lift_after_close", lift_after_close))
        self.lift_after_close_z = float(cfg.get("lift_after_close_z", lift_after_close_z))
        self.close_gripper_qpos = float(cfg.get("close_gripper_qpos", close_gripper_qpos))
        self.open_gripper_width = float(cfg.get("open_gripper_width", open_gripper_width))
        adaptive_cfg = cfg.get("adaptive_gripper", {})
        self.adaptive_gripper_config = dict(adaptive_cfg) if isinstance(adaptive_cfg, dict) else {}
        self.object_pose_names = dict(cfg.get("object_pose_names", object_pose_names or {})) or {
            "prism": "prism",
            "object": "prism",
            "target object": "prism",
            "orange pad": "orange_pad",
            "orange_pad": "orange_pad",
            "green pad": "green_pad",
            "green_pad": "green_pad",
        }

    def functions(self) -> dict[str, Any]:
        return {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "home_pose": self.home_pose,
        }

    def get_object_pose(
        self,
        object_name: str,
        return_bbox_extent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a public landmark pose from UniVTAC actor observations.

        Args:
            object_name: Public object or landmark name, including the active
                lift-can object under the name "can".
            return_bbox_extent: Whether to also return an approximate extent.

        Returns:
            ``(position, quaternion_wxyz)`` by default. When
            ``return_bbox_extent=True``, returns
            ``(position, quaternion_wxyz, bbox_extent)``.
        """
        key = self._resolve_pose_key(object_name)
        landmarks = self._public_landmarks()
        if key not in landmarks:
            raise KeyError(f"object '{object_name}' not available in UniVTAC public poses")
        pos, quat, extent = landmarks[key]
        if return_bbox_extent:
            return pos, quat, extent
        return pos, quat

    def sample_grasp_pose(
        self,
        object_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a safe grasp pose compatible with original CaP callers.

        Task-specific contact poses are produced by the UniVTAC low-level
        adapter so sampling and native execution use the same geometry.
        """
        tool_pos, tool_quat = self._current_tool_pose()
        key = self._resolve_pose_key(object_name)
        if key in {"prism", "can"}:
            grasp_pose = self._public_grasp_pose(key)
            if grasp_pose is not None:
                return grasp_pose
            if key == "can":
                raise KeyError("object 'can' is not available for UniVTAC grasp sampling")
        return tool_pos, tool_quat

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> None:
        """Move to a target pose using bounded UniVTAC delta actions."""
        pos = np.asarray(position, dtype=np.float32).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=np.float32).reshape(4)

        grasp_result = self._try_approach_public_grasp(pos)
        if grasp_result is not None:
            return

        place_result = self._try_place_on_public_landmark(pos, quat)
        if place_result is not None:
            return

        cur_pos, cur_quat = self._current_tool_pose()
        if self._nearest_public_landmark(pos) is not None and self.preserve_landmark_orientation:
            quat = cur_quat

        # Preserve the original CaP contract: zero means a direct bounded move.
        # Callers request a staged approach explicitly with a positive value.
        approach = max(0.0, float(z_approach))
        if approach > 0.0:
            approach_target = pos.copy()
            approach_target[2] = max(approach_target[2] + approach, self.min_safe_z)
            self._move_to_pose_bounded(approach_target, quat, cur_pos, cur_quat)
            cur_pos, cur_quat = self._current_tool_pose()

        final_target = pos.copy()
        final_target[2] = max(final_target[2], self.min_safe_z)
        self._move_to_pose_bounded(final_target, quat, cur_pos, cur_quat)

    def open_gripper(
        self,
        adaptive: bool = True,
        target_width: float = 1.0,
        max_steps: int = 80,
    ) -> dict[str, Any]:
        """Open the gripper, releasing gently while native contact remains.

        Args:
            adaptive: Use CaP-X tactile feedback control when true.
            target_width: Normalized target width from 0 (closed) to 1 (open).
            max_steps: Maximum tactile servo iterations.

        Returns:
            Result containing ``released``, final width, and stop reason.
        """
        if adaptive:
            controller = self._adaptive_gripper_controller()
            result = controller.open(target_width=target_width, max_steps=max_steps)
            self._save_adaptive_trace(controller.trace)
            print(
                "[univtac-franka] adaptive_open "
                f"released={result['released']} reason={result['reason']} "
                f"width={result['width']:.4f} steps={result['steps']}",
                flush=True,
            )
            return result

        native_gripper = getattr(self._env, "move_gripper_native", None)
        if callable(native_gripper):
            result = native_gripper(
                qpos=self._width_to_qpos(target_width),
                opening=True,
                settle_steps=self.gripper_settle_steps,
            )
            print(
                "[univtac-franka] open_gripper_native "
                f"ok={bool(result.get('ok', False))} message={result.get('message', '')}",
                flush=True,
            )
            return {
                **result,
                "released": bool(result.get("ok", False)),
                "reason": "fixed_open",
                "target_width": float(np.clip(target_width, 0.0, 1.0)),
            }
        self._move_gripper(target_width)
        return {
            "ok": True,
            "released": True,
            "reason": "fixed_open",
            "target_width": float(np.clip(target_width, 0.0, 1.0)),
        }

    def close_gripper(
        self,
        adaptive: bool = True,
        target_force: float = 0.35,
        max_steps: int = 80,
    ) -> dict[str, Any]:
        """Close the gripper using CaP-X native tactile feedback control.

        Args:
            adaptive: Use tactile coarse/fine closing when true.
            target_force: Normalized target force used for stable-contact stop.
            max_steps: Maximum tactile servo iterations.

        Returns:
            Result containing ``stable``, contact state, and stop reason. The
            caller remains responsible for pose adjustment, retry, and lift.
        """
        if adaptive:
            controller = self._adaptive_gripper_controller()
            result = controller.close(target_force=target_force, max_steps=max_steps)
            self._save_adaptive_trace(controller.trace)
            print(
                "[univtac-franka] adaptive_close "
                f"stable={result['stable']} reason={result['reason']} "
                f"force={result['normal_force']:.3f} width={result['width']:.4f} "
                f"steps={result['steps']}",
                flush=True,
            )
            return result

        native_gripper = getattr(self._env, "move_gripper_native", None)
        if callable(native_gripper):
            result = native_gripper(
                qpos=self.close_gripper_qpos,
                opening=False,
                lift_after_close=self.lift_after_close,
                lift_z=self.lift_after_close_z,
                settle_steps=self.gripper_settle_steps,
            )
            print(
                "[univtac-franka] close_gripper_native "
                f"ok={bool(result.get('ok', False))} message={result.get('message', '')}",
                flush=True,
            )
            return {
                **result,
                "stable": False,
                "reason": "fixed_close_requires_tactile_confirmation",
            }
        self._move_gripper(0.0)
        return {
            "ok": True,
            "stable": False,
            "reason": "fixed_close_requires_tactile_confirmation",
        }

    def home_pose(self) -> None:
        """Move to a conservative hover/home pose."""
        tool_pos, tool_quat = self._current_tool_pose()
        target = tool_pos.copy()
        target[2] = max(self.release_hover_height, self.min_safe_z)
        self._move_to_pose_bounded(target, tool_quat, tool_pos, tool_quat)

    def get_robot_state(self) -> dict[str, Any]:
        """Expose the UniVTAC robot state with stable CaP keys."""
        return self._env.get_robot_state()

    def get_step_status(self) -> dict[str, Any]:
        """Expose step status without reward leakage."""
        return self._env.get_status()

    def wait_steps(self, n: int = 1) -> dict[str, Any]:
        """Advance the simulation without changing the command."""
        return self._env.wait_steps(n)

    def _move_gripper(self, width: float) -> None:
        state = self.get_robot_state()
        joint = np.asarray(state.get("joint", []), dtype=np.float32).flatten()
        if joint.size >= 7:
            arm = joint[:7]
        else:
            arm = np.zeros(7, dtype=np.float32)
        target = self._width_to_qpos(width)
        self._env.take_action(np.concatenate([arm, [target]]), action_type="qpos")
        if self.gripper_settle_steps > 0:
            self._env.wait_steps(self.gripper_settle_steps)

    def _adaptive_gripper_controller(self) -> TactileAdaptiveGripperController:
        calibration_fn = getattr(self._env, "get_gripper_calibration", None)
        command_fn = getattr(self._env, "command_gripper_width_step", None)
        if not callable(calibration_fn) or not callable(command_fn):
            raise RuntimeError("UniVTAC environment does not provide adaptive gripper hooks")

        calibration = calibration_fn()
        max_qpos = float(calibration["gripper_max_qpos"])
        if max_qpos <= 0.0:
            raise RuntimeError("gripper_max_qpos must be positive")
        cfg = self.adaptive_gripper_config
        coarse_qpos_step = float(cfg.get("coarse_qpos_step", 0.0005))
        fine_qpos_step = float(cfg.get("fine_qpos_step", 0.00005))
        controller_config = AdaptiveGripperConfig(
            coarse_step=coarse_qpos_step / max_qpos,
            fine_step=fine_qpos_step / max_qpos,
            contact_debounce_frames=int(cfg.get("contact_debounce_frames", 2)),
            stable_debounce_frames=int(cfg.get("stable_debounce_frames", 3)),
            release_debounce_frames=int(cfg.get("release_debounce_frames", 2)),
            settle_steps_per_command=int(cfg.get("settle_steps_per_command", 1)),
            contact_force_threshold=float(cfg.get("contact_force_threshold", 0.08)),
            contact_balance_threshold=float(cfg.get("contact_balance_threshold", 0.45)),
            slip_threshold=float(cfg.get("slip_threshold", 0.60)),
            one_sided_force_limit=float(cfg.get("one_sided_force_limit", 0.90)),
            max_qpos=max_qpos,
        )
        return TactileAdaptiveGripperController(
            get_width=lambda: float(calibration_fn()["current_width"]),
            command_width=lambda width, settle_steps: command_fn(
                width,
                settle_steps=settle_steps,
            ),
            read_tactile_summary=self._read_adaptive_tactile_summary,
            config=controller_config,
        )

    def _read_adaptive_tactile_summary(self) -> dict[str, Any]:
        buffer = getattr(self._env, "tactile_buffer", None)
        if buffer is None:
            raise RuntimeError("UniVTAC native tactile buffer is unavailable")
        window = max(1, int(self.adaptive_gripper_config.get("tactile_window", 5)))
        frames = buffer.recent(window)
        if not frames:
            refresh_fn = getattr(self._env, "refresh_native_observation", None)
            if not callable(refresh_fn):
                raise RuntimeError("UniVTAC native tactile observation is unavailable")
            refresh_fn(
                include_camera=False,
                include_tactile=True,
                include_embodiment=False,
                include_actor=False,
                tactile_data_types=["depth", "marker", "pose"],
            )
            frames = buffer.recent(window)
        calibration_fn = getattr(self._env, "get_native_tactile_calibration", None)
        calibration = calibration_fn() if callable(calibration_fn) else {}
        return summarize_native_tactile(frames, hand="both", **calibration)

    def _save_adaptive_trace(self, trace: list[dict[str, Any]]) -> None:
        save_fn = getattr(self._env, "append_tactile_gripper_trace", None)
        if callable(save_fn):
            save_fn(trace)

    def _move_to_pose_bounded(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
    ) -> None:
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)
        target_quat = np.asarray(target_quat, dtype=np.float32).reshape(4)
        current_pos = np.asarray(current_pos, dtype=np.float32).reshape(3)
        current_quat = np.asarray(current_quat, dtype=np.float32).reshape(4)

        delta_pos = target_pos - current_pos
        dist = float(np.linalg.norm(delta_pos))
        rot_angle = self._rotation_angle(current_quat, target_quat)
        steps = max(
            1,
            int(np.ceil(dist / max(self.max_delta_xyz, 1e-6))),
            int(np.ceil(rot_angle / max(self.max_delta_rpy, 1e-6))),
        )
        pos_points = [current_pos + (delta_pos * (i / steps)) for i in range(1, steps + 1)]

        quat_points = self._slerp_quaternion_path(current_quat, target_quat, steps)
        for idx, pos in enumerate(pos_points):
            prev = current_pos if idx == 0 else pos_points[idx - 1]
            delta_xyz = np.clip(pos - prev, -self.max_delta_xyz, self.max_delta_xyz)
            if pos[2] < self.min_safe_z:
                delta_xyz[2] = max(self.min_safe_z - float(prev[2]), 0.0)

            delta_rpy = quat_points[idx]
            prev_quat = current_quat if idx == 0 else quat_points[idx - 1]
            delta_euler = self._quaternion_delta_to_rpy(prev_quat, delta_rpy)
            delta_euler = np.clip(delta_euler, -self.max_delta_rpy, self.max_delta_rpy)

            result = self._env.take_action(
                np.concatenate([delta_xyz, delta_euler, [0.0]]),
                action_type="delta_ee",
            )
            if not result.get("ok", False):
                break

    def _slerp_quaternion_path(self, q0: np.ndarray, q1: np.ndarray, steps: int) -> list[np.ndarray]:
        q0 = self._normalize_quat(q0)
        q1 = self._normalize_quat(q1)
        if steps <= 1:
            return [q1]
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return [self._normalize_quat((1.0 - t) * q0 + t * q1) for t in np.linspace(0.0, 1.0, steps)]
        theta = float(np.arccos(dot))
        sin_theta = float(np.sin(theta))
        out: list[np.ndarray] = []
        for t in np.linspace(0.0, 1.0, steps):
            w0 = float(np.sin((1.0 - t) * theta) / sin_theta)
            w1 = float(np.sin(t * theta) / sin_theta)
            out.append(self._normalize_quat(w0 * q0 + w1 * q1))
        return out

    def _quaternion_delta_to_rpy(self, q_prev: np.ndarray, q_next: np.ndarray) -> np.ndarray:
        prev = SciRotation.from_quat(self._wxyz_to_xyzw(q_prev))
        nxt = SciRotation.from_quat(self._wxyz_to_xyzw(q_next))
        delta = nxt * prev.inv()
        return delta.as_euler("xyz", degrees=False).astype(np.float32)

    def _width_to_qpos(self, width: float) -> float:
        clipped = float(np.clip(width, 0.0, 1.0))
        task = getattr(self._env, "task", None)
        robot_manager = getattr(task, "_robot_manager", None)
        max_qpos = float(getattr(robot_manager, "gripper_max_qpos", 0.039))
        return clipped * max_qpos

    def _resolve_pose_key(self, object_name: str) -> str:
        key = str(object_name).strip().lower()
        return self.object_pose_names.get(key, key.replace(" ", "_"))

    def _estimate_extent_from_pose_name(self, key: str) -> np.ndarray:
        if "pad" in key:
            return np.array([0.10, 0.10, 0.03], dtype=np.float32)
        if key == "can":
            return np.array([0.06, 0.06, 0.12], dtype=np.float32)
        return np.array([0.03, 0.03, 0.03], dtype=np.float32)

    def _public_landmarks(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        quat = (
            self._current_ee_quat()
            if self.preserve_landmark_orientation
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        landmarks = {
            "orange_pad": (
                np.array([0.40, -0.08, 0.025], dtype=np.float32),
                quat.copy(),
                np.array([0.10, 0.10, 0.03], dtype=np.float32),
            ),
            "green_pad": (
                np.array([0.40, 0.08, 0.025], dtype=np.float32),
                quat.copy(),
                np.array([0.10, 0.10, 0.03], dtype=np.float32),
            ),
        }
        prism_pos = self._public_prism_pose()
        if prism_pos is not None:
            landmarks["prism"] = (
                prism_pos.astype(np.float32),
                self._native_grasp_quat_wxyz(),
                np.array([0.06, 0.03, 0.03], dtype=np.float32),
            )
        can_pose = self._public_actor_pose("can")
        if can_pose is not None:
            can_pos, can_quat = can_pose
            landmarks["can"] = (
                can_pos,
                can_quat,
                self._estimate_extent_from_pose_name("can"),
            )
        return landmarks

    def _current_ee_quat(self) -> np.ndarray:
        _pos, quat = self._current_tool_pose()
        return self._normalize_quat(quat)

    def _current_tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the CaP-facing control pose.

        Original CaP high-level poses are most useful as gripper/tool-center
        targets. UniVTAC's low-level delta_ee action still moves the underlying
        panda_hand frame, but a pure translation delta moves the gripper center
        by the same amount. Keeping the public frame at the gripper center
        prevents pad targets from driving the fingers through the table.
        """
        task = getattr(self._env, "task", None)
        robot_manager = getattr(task, "_robot_manager", None)
        if robot_manager is not None:
            try:
                pose = robot_manager.get_gripper_center_pose()
                pos = np.asarray(pose.p, dtype=np.float32).reshape(3)
                quat = np.asarray(pose.q, dtype=np.float32).reshape(4)
                return pos, self._normalize_quat(quat)
            except Exception:
                pass

        state = self.get_robot_state()
        pos = np.asarray(state.get("ee_pos", [0.0, 0.0, 0.2]), dtype=np.float32).reshape(3)
        quat = np.asarray(state.get("ee_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(4)
        return pos, self._normalize_quat(quat)

    def _nearest_public_landmark(self, position: np.ndarray) -> tuple[str, np.ndarray] | None:
        pos = np.asarray(position, dtype=np.float32).reshape(3)
        best_key = None
        best_landmark = None
        best_dist = float("inf")
        for key, (landmark_pos, _quat, _extent) in self._public_landmarks().items():
            dist = float(np.linalg.norm(pos[:2] - landmark_pos[:2]))
            if dist < best_dist:
                best_key = key
                best_landmark = landmark_pos
                best_dist = dist
        if best_key is None or best_landmark is None or best_dist > self.placement_xy_tolerance:
            return None
        return best_key, best_landmark

    def _nearest_public_grasp_target(self, position: np.ndarray) -> tuple[str, np.ndarray] | None:
        pos = np.asarray(position, dtype=np.float32).reshape(3)
        best: tuple[str, np.ndarray] | None = None
        best_dist = float("inf")
        for key in ("prism", "can"):
            sampled = self._public_grasp_pose(key)
            if sampled is None:
                continue
            grasp_pos, _grasp_quat = sampled
            xy_dist = float(np.linalg.norm(pos[:2] - grasp_pos[:2]))
            z_dist = abs(float(pos[2] - grasp_pos[2]))
            if xy_dist > self.grasp_xy_tolerance or z_dist > self.grasp_z_tolerance:
                continue
            distance = float(np.linalg.norm(pos - grasp_pos))
            if distance < best_dist:
                best = (key, grasp_pos)
                best_dist = distance
        return best

    def _try_approach_public_grasp(self, position: np.ndarray) -> dict[str, Any] | None:
        if not self.use_task_grasp_actor_for_objects:
            return None
        nearest = self._nearest_public_grasp_target(position)
        if nearest is None:
            return None
        key, grasp_pos = nearest
        approach_fn = getattr(self._env, "approach_grasped_actor", None)
        if not callable(approach_fn):
            return None

        result = approach_fn(
            object_name=key,
            position_offset=(np.asarray(position, dtype=np.float32) - grasp_pos),
            pre_dis=self.grasp_pre_dis,
            dis=self.grasp_dis,
            grasp_height=self.grasp_height,
        )
        print(
            "[univtac-franka] approach_grasp_via_task_atom "
            f"object={key} requested={np.array2string(np.asarray(position), precision=3)} "
            f"offset={np.array2string(np.asarray(position) - grasp_pos, precision=3)} "
            f"ok={bool(result.get('ok', False))} message={result.get('message', '')}",
            flush=True,
        )
        return result

    def _try_place_on_public_landmark(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> dict[str, Any] | None:
        if not self.use_task_place_actor_for_landmarks:
            return None
        nearest = self._nearest_public_landmark(position)
        if nearest is None:
            return None
        key, landmark_pos = nearest
        place_fn = getattr(self._env, "place_grasped_actor", None)
        if not callable(place_fn):
            return None

        result = place_fn(
            target_name=key,
            target_position=landmark_pos,
            target_quaternion_wxyz=quaternion_wxyz,
            pre_dis=self.placement_pre_dis,
            dis=self.placement_dis,
            time_dilation_factor=self.placement_time_dilation_factor,
        )
        print(
            "[univtac-franka] place_via_task_atom "
            f"target={key} requested={np.array2string(np.asarray(position), precision=3)} "
            f"motion_ok={bool(result.get('ok', False))} message={result.get('message', '')}",
            flush=True,
        )
        return result

    def _public_prism_pose(self) -> np.ndarray | None:
        pose = self._public_actor_pose("prism")
        return None if pose is None else pose[0]

    def _public_actor_pose(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        task = getattr(self._env, "task", None)
        actor = getattr(task, key, None)
        if actor is None:
            return None
        try:
            pose = actor.get_pose()
            pos = np.asarray(pose.p, dtype=np.float32).reshape(3)
            raw_quat = getattr(pose, "q", None)
            if raw_quat is None:
                if key != "prism":
                    return None
                quat = self._native_grasp_quat_wxyz()
            else:
                quat = self._normalize_quat(np.asarray(raw_quat, dtype=np.float32).reshape(4))
            return pos, quat
        except Exception:
            return None

    def _public_grasp_pose(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        if self._public_actor_pose(key) is None:
            return None
        sample_fn = getattr(self._env, "get_public_grasp_pose", None)
        if callable(sample_fn):
            try:
                sampled = sample_fn(key, grasp_height=self.grasp_height)
            except (KeyError, RuntimeError, ValueError):
                sampled = None
            if sampled is not None:
                pos, quat = sampled
                return (
                    np.asarray(pos, dtype=np.float32).reshape(3),
                    self._normalize_quat(np.asarray(quat, dtype=np.float32).reshape(4)),
                )

        if key == "prism":
            pos = self._public_prism_pose()
            if pos is not None:
                grasp_pos = pos.copy()
                grasp_pos[2] += self.grasp_height
                return grasp_pos.astype(np.float32), self._native_grasp_quat_wxyz()
        return None

    def _native_grasp_quat_wxyz(self) -> np.ndarray:
        try:
            from envs.utils.transforms import construct_grasp_pose

            pose = construct_grasp_pose(np.zeros(3, dtype=np.float32), [0, 0, 1], [1, 0, 0])
            return self._normalize_quat(np.asarray(pose.q, dtype=np.float32).reshape(4))
        except Exception:
            return self._current_ee_quat()

    def _runtime_config(self) -> dict[str, Any]:
        cfg = getattr(self._env, "api_configs", None)
        if not isinstance(cfg, dict):
            return {}
        runtime = cfg.get("franka_control_api", {})
        return runtime if isinstance(runtime, dict) else {}

    def _normalize_quat(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return quat / norm

    def _wxyz_to_xyzw(self, quat: np.ndarray) -> np.ndarray:
        quat = self._normalize_quat(quat)
        return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)

    def _rotation_angle(self, q_prev: np.ndarray, q_next: np.ndarray) -> float:
        q_prev = self._normalize_quat(q_prev)
        q_next = self._normalize_quat(q_next)
        dot = abs(float(np.dot(q_prev, q_next)))
        dot = float(np.clip(dot, -1.0, 1.0))
        return float(2.0 * np.arccos(dot))
