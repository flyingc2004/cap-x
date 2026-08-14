"""CaP-style Franka API compatibility layer for UniVTAC.

This adapter preserves the original high-level Franka control surface used by
CaP-X prompt templates while translating motions into UniVTAC native actions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation

from capx.envs.base import BaseEnv
from capx.envs.tasks.exceptions import RecoverableTaskFailure
from capx.integrations.base_api import ApiBase
from capx.integrations.tactile.adaptive_gripper import (
    AdaptiveGripperConfig,
    TactileAdaptiveGripperController,
)
from capx.integrations.univtac.native_tactile import summarize_native_tactile
from capx.integrations.univtac.rgbd_perception import UniVTACRgbdPerception


class _ComparableTactileLabel(str):
    """String label that tolerates accidental numeric comparisons in LLM code."""

    def __new__(cls, value: str, score: float) -> "_ComparableTactileLabel":
        obj = str.__new__(cls, value)
        obj._score = float(score)
        return obj

    @staticmethod
    def _numeric_value(other: Any) -> float | None:
        if isinstance(other, (str, bytes)):
            return None
        try:
            return float(other)
        except Exception:
            return None

    def __lt__(self, other: Any) -> bool:
        value = self._numeric_value(other)
        return self._score < value if value is not None else str.__lt__(self, other)

    def __le__(self, other: Any) -> bool:
        value = self._numeric_value(other)
        return self._score <= value if value is not None else str.__le__(self, other)

    def __gt__(self, other: Any) -> bool:
        value = self._numeric_value(other)
        return self._score > value if value is not None else str.__gt__(self, other)

    def __ge__(self, other: Any) -> bool:
        value = self._numeric_value(other)
        return self._score >= value if value is not None else str.__ge__(self, other)


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
        home_pose_relative_lift: bool = False,
        home_lift_delta_z: float = 0.10,
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
        tactile_adaptive_gripper_enabled: bool = True,
        object_pose_names: dict[str, str] | None = None,
        rgbd_perception_enabled: bool = False,
        perception_camera: str = "head",
        perception_prompt_map: dict[str, str] | None = None,
        sam3_service_url: str = "http://127.0.0.1:8114",
        graspnet_service_url: str = "http://127.0.0.1:8115",
        perception_timeout_seconds: float = 120.0,
        perception_min_depth_points: int = 32,
        perception_retry_attempts: int = 2,
        perception_prompt_fallbacks: dict[str, list[str]] | None = None,
        grasp_local_z_offset: float = 0.12,
        use_native_pose_planner: bool = False,
        record_perception_diagnostic: bool = False,
        official_anchor_fallback_enabled: bool = False,
        official_anchor_fallback_objects: list[str] | None = None,
        max_goto_pose_actions: int | None = None,
        max_home_pose_actions: int | None = None,
        max_native_pose_actions: int | None = None,
        max_gripper_servo_steps: int | None = None,
        max_gripper_settle_steps: int | None = None,
        max_insert_actions: int | None = None,
        move_relative_max_lateral: float = 0.0005,
        move_relative_max_rotation: float = 0.006,
        move_relative_lateral_budget: float = 0.002,
        move_relative_rotation_budget: float = 0.024,
        move_relative_lock_on_guard_failure: bool = True,
        insert_depth_log_interval: int = 50,
    ) -> None:
        super().__init__(env)
        cfg = self._runtime_config()
        self.task_name = str(cfg.get("task_name", getattr(env, "task_name", "")))
        self.min_safe_z = float(cfg.get("min_safe_z", min_safe_z))
        self.max_delta_xyz = float(cfg.get("max_delta_xyz", max_delta_xyz))
        self.max_delta_rpy = float(cfg.get("max_delta_rpy", max_delta_rpy))
        self.max_delta_gripper = float(cfg.get("max_delta_gripper", max_delta_gripper))
        self.default_z_approach = float(cfg.get("default_z_approach", default_z_approach))
        self.release_hover_height = float(cfg.get("release_hover_height", release_hover_height))
        self.home_pose_relative_lift = bool(
            cfg.get("home_pose_relative_lift", home_pose_relative_lift)
        )
        self.home_lift_delta_z = float(cfg.get("home_lift_delta_z", home_lift_delta_z))
        if not np.isfinite(self.home_lift_delta_z) or self.home_lift_delta_z <= 0.0:
            raise ValueError("home_lift_delta_z must be a positive finite distance")
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
        self.tactile_adaptive_gripper_enabled = bool(
            cfg.get("tactile_adaptive_gripper_enabled", tactile_adaptive_gripper_enabled)
        )
        adaptive_cfg = cfg.get("adaptive_gripper", {})
        self.adaptive_gripper_config = dict(adaptive_cfg) if isinstance(adaptive_cfg, dict) else {}
        guard_cfg = cfg.get("tactile_guard", {})
        self.tactile_guard_config = self._default_tactile_guard_config()
        if isinstance(guard_cfg, dict):
            self.tactile_guard_config.update(guard_cfg)
        self.guard_micro_down_step = float(
            self.tactile_guard_config.get("guard_micro_down_step", 0.0005)
        )
        self.guard_micro_lateral_step = float(
            self.tactile_guard_config.get("guard_micro_lateral_step", 0.0002)
        )
        self.guard_micro_rpy_step = float(
            self.tactile_guard_config.get("guard_micro_rpy_step", 0.002)
        )
        self.object_pose_names = dict(cfg.get("object_pose_names", object_pose_names or {})) or {
            "prism": "prism",
            "object": "prism",
            "target object": "prism",
            "orange pad": "orange_pad",
            "orange_pad": "orange_pad",
            "green pad": "green_pad",
            "green_pad": "green_pad",
        }
        self.rgbd_perception_enabled = bool(
            cfg.get("rgbd_perception_enabled", rgbd_perception_enabled)
        )
        self.perception_camera = str(cfg.get("perception_camera", perception_camera))
        configured_prompts = cfg.get("perception_prompt_map", perception_prompt_map or {})
        self.perception_prompt_map = self._normalize_prompt_map(
            configured_prompts or {"can": "cylindrical can"}
        )
        fallback_cfg = cfg.get("perception_prompt_fallbacks", perception_prompt_fallbacks or {})
        self.perception_prompt_fallbacks = self._normalize_prompt_map(fallback_cfg)
        if "can" not in self.perception_prompt_fallbacks:
            self.perception_prompt_fallbacks["can"] = [
                "can",
                "cylindrical object",
                "small cylinder",
            ]
        self.perception_retry_attempts = max(
            1,
            int(cfg.get("perception_retry_attempts", perception_retry_attempts)),
        )
        self.use_native_pose_planner = bool(
            cfg.get("use_native_pose_planner", use_native_pose_planner)
        )
        self.record_perception_diagnostic = bool(
            cfg.get("record_perception_diagnostic", record_perception_diagnostic)
        )
        self.official_anchor_fallback_enabled = bool(
            cfg.get("official_anchor_fallback_enabled", official_anchor_fallback_enabled)
        )
        fallback_objects = cfg.get(
            "official_anchor_fallback_objects",
            official_anchor_fallback_objects or ["can"],
        )
        self.official_anchor_fallback_objects = self._normalize_object_set(
            fallback_objects,
            object_pose_names=self.object_pose_names,
        )
        self.max_goto_pose_actions = self._optional_positive_int(
            cfg.get("max_goto_pose_actions", max_goto_pose_actions),
        )
        self.max_home_pose_actions = self._optional_positive_int(
            cfg.get("max_home_pose_actions", max_home_pose_actions),
        )
        self.max_native_pose_actions = self._optional_positive_int(
            cfg.get("max_native_pose_actions", max_native_pose_actions),
        )
        self.max_gripper_servo_steps = self._optional_positive_int(
            cfg.get("max_gripper_servo_steps", max_gripper_servo_steps),
        )
        self.max_gripper_settle_steps = self._optional_positive_int(
            cfg.get("max_gripper_settle_steps", max_gripper_settle_steps),
        )
        self.max_insert_actions = self._optional_positive_int(
            cfg.get("max_insert_actions", max_insert_actions),
        )
        self.move_relative_max_lateral = float(
            cfg.get("move_relative_max_lateral", move_relative_max_lateral)
        )
        self.move_relative_max_rotation = float(
            cfg.get("move_relative_max_rotation", move_relative_max_rotation)
        )
        self.move_relative_lateral_budget = float(
            cfg.get("move_relative_lateral_budget", move_relative_lateral_budget)
        )
        self.move_relative_rotation_budget = float(
            cfg.get("move_relative_rotation_budget", move_relative_rotation_budget)
        )
        self.move_relative_lock_on_guard_failure = bool(
            cfg.get("move_relative_lock_on_guard_failure", move_relative_lock_on_guard_failure)
        )
        self.insert_depth_log_interval = max(
            0,
            int(cfg.get("insert_depth_log_interval", insert_depth_log_interval)),
        )
        self._move_relative_lateral_used = 0.0
        self._move_relative_rotation_used = 0.0
        self._move_relative_guard_locked_reason: str | None = None
        self._insert_depth_reference_z: float | None = None
        self._last_insert_depth_log_bucket = 0
        self._perception_diagnostic_reset_serial: int | None = None
        self._rgbd_perception = UniVTACRgbdPerception(
            sam3_url=str(cfg.get("sam3_service_url", sam3_service_url)),
            graspnet_url=str(cfg.get("graspnet_service_url", graspnet_service_url)),
            request_timeout_seconds=float(
                cfg.get("perception_timeout_seconds", perception_timeout_seconds)
            ),
            min_depth_points=int(
                cfg.get("perception_min_depth_points", perception_min_depth_points)
            ),
            grasp_local_z_offset=float(
                cfg.get("grasp_local_z_offset", grasp_local_z_offset)
            ),
        )

    def functions(self) -> dict[str, Any]:
        return {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "home_pose": self.home_pose,
            "move_relative": self.move_relative,
        }

    def reset_episode(self) -> None:
        """Reset per-trial adapter accounting."""
        self._move_relative_lateral_used = 0.0
        self._move_relative_rotation_used = 0.0
        self._move_relative_guard_locked_reason = None
        self._insert_depth_reference_z = None
        self._last_insert_depth_log_bucket = 0

    def get_object_pose(
        self,
        object_name: str,
        return_bbox_extent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate an object pose or read a configured public landmark.

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
        if self.rgbd_perception_enabled:
            estimate, frame, prompt, attempt = self._estimate_rgbd_with_retry(
                key,
                kind="object_pose",
            )
            self._append_perception_artifact(
                {
                    "kind": "object_pose",
                    "source": "rgbd",
                    "object_name": key,
                    "prompt": prompt,
                    "attempt": attempt,
                    "frame": frame,
                    "mask": estimate.mask,
                    "points_world": estimate.points_world,
                    "position": estimate.position,
                    "quaternion_wxyz": estimate.quaternion_wxyz,
                    "extent": estimate.extent,
                    "segmentation_score": estimate.score,
                }
            )
            print(
                "[univtac-franka] pose_source=rgbd "
                f"object={key} points={len(estimate.points_world)} "
                f"score={estimate.score:.3f}",
                flush=True,
            )
            if return_bbox_extent:
                return estimate.position, estimate.quaternion_wxyz, estimate.extent
            return estimate.position, estimate.quaternion_wxyz

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
        key = self._resolve_pose_key(object_name)
        if self.rgbd_perception_enabled:
            try:
                estimate, frame, prompt, attempt = self._estimate_rgbd_with_retry(
                    key,
                    kind="grasp_pose",
                )
            except RecoverableTaskFailure as exc:
                fallback = self._official_anchor_fallback(key, fallback_from=exc)
                if fallback is not None:
                    return fallback
                raise
            self._append_perception_artifact(
                {
                    "kind": "grasp_pose",
                    "source": "rgbd_contact_graspnet",
                    "object_name": key,
                    "prompt": prompt,
                    "attempt": attempt,
                    "frame": frame,
                    "mask": estimate.mask,
                    "points_world": estimate.points_world,
                    "position": estimate.position,
                    "quaternion_wxyz": estimate.quaternion_wxyz,
                    "obb_position": estimate.object_position,
                    "obb_quaternion_wxyz": estimate.object_quaternion_wxyz,
                    "obb_extent": estimate.object_extent,
                    "grasps_camera": estimate.grasps_camera,
                    "grasp_scores": estimate.scores,
                    "selected_index": estimate.selected_index,
                }
            )
            print(
                "[univtac-franka] grasp_source=rgbd_contact_graspnet "
                f"object={key} candidates={len(estimate.scores)} "
                f"selected={estimate.selected_index}",
                flush=True,
            )
            return estimate.position, estimate.quaternion_wxyz

        tool_pos, tool_quat = self._current_tool_pose()
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
    ) -> dict[str, Any]:
        """Move to a target pose using bounded UniVTAC delta actions."""
        success_result = self._success_latched_result("goto_pose")
        if success_result is not None:
            return success_result

        pos = np.asarray(position, dtype=np.float32).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=np.float32).reshape(4)

        if self.use_native_pose_planner:
            return self._goto_pose_native(pos, quat, z_approach=float(z_approach))

        grasp_result = self._try_approach_public_grasp(pos)
        if grasp_result is not None:
            return grasp_result

        place_result = self._try_place_on_public_landmark(pos, quat)
        if place_result is not None:
            return place_result

        cur_pos, cur_quat = self._current_tool_pose()
        if self._nearest_public_landmark(pos) is not None and self.preserve_landmark_orientation:
            quat = cur_quat

        # Preserve the original CaP contract: zero means a direct bounded move.
        # Callers request a staged approach explicitly with a positive value.
        remaining_api_actions = self.max_goto_pose_actions
        approach = max(0.0, float(z_approach))
        if approach > 0.0:
            approach_target = pos.copy()
            approach_target[2] = max(approach_target[2] + approach, self.min_safe_z)
            result = self._call_move_to_pose_bounded(
                approach_target,
                quat,
                cur_pos,
                cur_quat,
                max_actions=remaining_api_actions,
            )
            if isinstance(result, dict) and not bool(result.get("ok", False)):
                return result
            remaining_api_actions = self._consume_api_actions(remaining_api_actions, result)
            cur_pos, cur_quat = self._current_tool_pose()

        final_target = pos.copy()
        final_target[2] = max(final_target[2], self.min_safe_z)
        return self._call_move_to_pose_bounded(
            final_target,
            quat,
            cur_pos,
            cur_quat,
            max_actions=remaining_api_actions,
        )

    def open_gripper(
        self,
        adaptive: bool = True,
        target_width: float = 1.0,
        max_steps: int = 80,
    ) -> dict[str, Any]:
        """Open the gripper, releasing gently while native contact remains.

        Args:
            adaptive: Use feedback-driven coarse/fine control when enabled by config.
            target_width: Normalized target width from 0 (closed) to 1 (open).
            max_steps: Maximum tactile servo iterations.

        Returns:
            Result containing ``released``, final width, and stop reason.
        """
        requested_max_steps = int(max_steps)
        max_steps = self._limit_gripper_servo_steps(requested_max_steps)
        if not self._begin_gripper_action():
            return self._blocked_gripper_result(opening=True)
        if adaptive and self.tactile_adaptive_gripper_enabled:
            controller = self._adaptive_gripper_controller()
            result = controller.open(target_width=target_width, max_steps=max_steps)
            self._annotate_servo_limit(result, requested_max_steps, max_steps)
            self._save_adaptive_trace(controller.trace)
            print(
                "[univtac-franka] adaptive_open "
                f"released={result['released']} reason={result['reason']} "
                f"width={result['width']:.4f} steps={result['steps']}",
                flush=True,
            )
            self._finalize_high_level_action()
            return result

        native_gripper = getattr(self._env, "move_gripper_native", None)
        if callable(native_gripper):
            result = native_gripper(
                qpos=self._width_to_qpos(target_width),
                opening=True,
                settle_steps=self._limited_gripper_settle_steps(),
            )
            print(
                "[univtac-franka] open_gripper_native "
                f"ok={bool(result.get('ok', False))} message={result.get('message', '')}",
                flush=True,
            )
            output = {
                **result,
                "released": bool(result.get("ok", False)),
                "reason": "fixed_open",
                "target_width": float(np.clip(target_width, 0.0, 1.0)),
            }
            self._finalize_high_level_action()
            return output
        self._move_gripper(target_width)
        output = {
            "ok": True,
            "released": True,
            "reason": "fixed_open",
            "target_width": float(np.clip(target_width, 0.0, 1.0)),
        }
        self._finalize_high_level_action()
        return output

    def close_gripper(
        self,
        adaptive: bool = True,
        target_force: float = 0.35,
        max_steps: int = 80,
    ) -> dict[str, Any]:
        """Close the gripper using optional feedback-driven width control.

        Args:
            adaptive: Use feedback-driven coarse/fine closing when enabled by config.
            target_force: Normalized target force used for stable-contact stop.
            max_steps: Maximum tactile servo iterations.

        Returns:
            Result containing ``stable``, contact state, and stop reason. The
            caller remains responsible for pose adjustment, retry, and lift.
        """
        requested_max_steps = int(max_steps)
        max_steps = self._limit_gripper_servo_steps(requested_max_steps)
        if not self._begin_gripper_action():
            return self._blocked_gripper_result(opening=False)
        self._record_rgbd_diagnostic_if_needed()
        if adaptive and self.tactile_adaptive_gripper_enabled:
            controller = self._adaptive_gripper_controller()
            result = controller.close(target_force=target_force, max_steps=max_steps)
            self._annotate_servo_limit(result, requested_max_steps, max_steps)
            self._save_adaptive_trace(controller.trace)
            print(
                "[univtac-franka] adaptive_close "
                f"stable={result['stable']} reason={result['reason']} "
                f"force={result['normal_force']:.3f} width={result['width']:.4f} "
                f"steps={result['steps']}",
                flush=True,
            )
            self._finalize_high_level_action()
            return result

        native_gripper = getattr(self._env, "move_gripper_native", None)
        if callable(native_gripper):
            result = native_gripper(
                qpos=self.close_gripper_qpos,
                opening=False,
                lift_after_close=self.lift_after_close,
                lift_z=self.lift_after_close_z,
                settle_steps=self._limited_gripper_settle_steps(),
            )
            print(
                "[univtac-franka] close_gripper_native "
                f"ok={bool(result.get('ok', False))} message={result.get('message', '')}",
                flush=True,
            )
            output = {
                **result,
                "stable": False,
                "reason": "fixed_close_requires_tactile_confirmation",
            }
            self._finalize_high_level_action()
            return output
        self._move_gripper(0.0)
        output = {
            "ok": True,
            "stable": False,
            "reason": "fixed_close_requires_tactile_confirmation",
        }
        self._finalize_high_level_action()
        return output

    def home_pose(self) -> None:
        """Move to a conservative hover/home pose."""
        tool_pos, tool_quat = self._current_tool_pose()
        target = tool_pos.copy()
        if self.home_pose_relative_lift:
            target[2] = max(
                float(tool_pos[2]) + self.home_lift_delta_z,
                self.min_safe_z,
            )
        else:
            target[2] = max(self.release_hover_height, self.min_safe_z)
        if self.use_native_pose_planner:
            self._goto_pose_native(target, tool_quat, z_approach=0.0)
            return
        self._call_move_to_pose_bounded(
            target,
            tool_quat,
            tool_pos,
            tool_quat,
            max_actions=self.max_home_pose_actions,
        )

    def move_relative(
        self,
        delta_xyz: list[float],
        delta_rpy: list[float] | None = None,
        tactile_guard: bool = True,
    ) -> dict[str, Any]:
        """Move by a small robot-base/world-frame relative displacement.

        The return shape is intentionally compact so generated code can use it
        in simple if/for branches without parsing low-level tactile metrics.

        Args:
            delta_xyz: Relative XYZ displacement in the robot base/world frame.
                For insert_hole, guarded downward progress is executed in
                small micro-steps. Lateral correction requests should stay at
                or below the returned limits["max_lateral_step"].
            delta_rpy: Optional relative roll, pitch, yaw rotation in radians.
                For insert_hole, pitch/roll probes should stay at or below the
                returned limits["max_rotation_step"].
            tactile_guard: If True, use native tactile feedback to preempt or
                interrupt unsafe insertion moves.

        Returns:
            Dictionary with ok, reason, executed_distance, executed_depth,
            remaining_actions, success_latched, episode_stopped, preempted,
            interrupted, clipped, limits, and tactile. Recoverable tactile reasons include
            preempted_by_slip_risk, interrupted_by_slip_warning, and
            completed_with_slip. Correction budget reasons are safe no-ops, not
            code failures. The tactile field contains compact labels such as
            contact, stable, slip, slip_risk, incipient_slip, pressure_side,
            shear_side, force_change, drift_trend, and correction_hint.
        """
        self._ensure_insert_depth_reference()
        before_summary = self._read_move_tactile_summary()
        zero = np.zeros(3, dtype=np.float32)
        success_result = self._success_latched_move_relative_result(before_summary)
        if success_result is not None:
            self._log_move_relative_result(success_result, zero, zero, zero, zero)
            return success_result
        try:
            xyz = np.asarray(delta_xyz, dtype=np.float32).reshape(3)
            rpy = (
                zero.copy()
                if delta_rpy is None
                else np.asarray(delta_rpy, dtype=np.float32).reshape(3)
            )
        except Exception:
            return self._finish_move_relative_result(
                ok=False,
                reason="invalid_move_relative_arguments",
                requested_xyz=zero,
                requested_rpy=zero,
                executed_delta_xyz=zero,
                executed_delta_rpy=zero,
                before_summary=before_summary,
                after_summary=before_summary,
            )
        if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(rpy)):
            return self._finish_move_relative_result(
                ok=False,
                reason="invalid_move_relative_arguments",
                requested_xyz=zero,
                requested_rpy=zero,
                executed_delta_xyz=zero,
                executed_delta_rpy=zero,
                before_summary=before_summary,
                after_summary=before_summary,
            )

        requested_xyz = xyz.copy()
        requested_rpy = rpy.copy()
        locked_reason = self._move_relative_guard_locked_reason
        if bool(tactile_guard) and locked_reason is not None:
            return self._finish_move_relative_result(
                ok=False,
                reason=locked_reason,
                requested_xyz=requested_xyz,
                requested_rpy=requested_rpy,
                executed_delta_xyz=zero,
                executed_delta_rpy=zero,
                before_summary=before_summary,
                after_summary=before_summary,
            )

        xyz, rpy, clip_reason = self._clip_move_relative_request(xyz, rpy)
        if (
            clip_reason in {"move_relative_lateral_budget", "move_relative_rotation_budget"}
            and float(np.linalg.norm(xyz)) <= 1e-9
            and float(np.linalg.norm(rpy)) <= 1e-9
        ):
            return self._finish_move_relative_result(
                ok=True,
                reason=clip_reason,
                requested_xyz=requested_xyz,
                requested_rpy=requested_rpy,
                executed_delta_xyz=zero,
                executed_delta_rpy=zero,
                before_summary=before_summary,
                after_summary=before_summary,
                clipped=True,
                clip_reason=clip_reason,
            )

        downward_guarded = bool(tactile_guard) and self._is_downward_insertion(xyz)
        before_tactile = self._simple_tactile_feedback({}, before_summary)
        if downward_guarded and str(before_tactile.get("slip_risk")) == "high":
            return self._finish_move_relative_result(
                ok=True,
                reason="preempted_by_slip_risk",
                requested_xyz=requested_xyz,
                requested_rpy=requested_rpy,
                executed_delta_xyz=zero,
                executed_delta_rpy=zero,
                before_summary=before_summary,
                after_summary=before_summary,
                preempted=True,
                clipped=clip_reason is not None,
                clip_reason=clip_reason,
            )

        xyz_norm = float(np.linalg.norm(xyz))
        rpy_norm = float(np.linalg.norm(rpy))
        if downward_guarded:
            down_steps = (
                self._ceil_step_count(abs(float(xyz[2])), self.guard_micro_down_step)
                if abs(float(xyz[2])) > 0.0
                else 1
            )
            lateral_norm = float(np.linalg.norm(xyz[:2]))
            lateral_steps = (
                self._ceil_step_count(lateral_norm, self.guard_micro_lateral_step)
                if lateral_norm > 0.0
                else 1
            )
            rpy_steps = (
                self._ceil_step_count(rpy_norm, self.guard_micro_rpy_step)
                if rpy_norm > 0.0
                else 1
            )
            xyz_steps = max(down_steps, lateral_steps)
        else:
            xyz_steps = self._ceil_step_count(xyz_norm, self.max_delta_xyz) if xyz_norm > 0.0 else 1
            rpy_steps = self._ceil_step_count(rpy_norm, self.max_delta_rpy) if rpy_norm > 0.0 else 1
        planned_steps = max(1, xyz_steps, rpy_steps)
        action_limit = self._effective_action_limit(self.max_insert_actions)
        if action_limit is not None and action_limit <= 0:
            success_result = self._success_latched_move_relative_result(before_summary)
            if success_result is not None:
                self._log_move_relative_result(success_result, xyz, rpy, zero, zero)
                return success_result
            return self._finish_move_relative_result(
                ok=False,
                reason="official_protocol_stopped",
                requested_xyz=requested_xyz,
                requested_rpy=requested_rpy,
                executed_delta_xyz=zero,
                executed_delta_rpy=zero,
                before_summary=before_summary,
                after_summary=before_summary,
                clipped=clip_reason is not None,
                clip_reason=clip_reason,
            )
        run_steps = min(planned_steps, action_limit) if action_limit is not None else planned_steps
        if downward_guarded and str(before_tactile.get("slip_risk")) == "medium":
            run_steps = min(run_steps, 1)
        if xyz_norm <= 1e-9 and rpy_norm <= 1e-9:
            return self._finish_move_relative_result(
                ok=True,
                reason="completed",
                requested_xyz=requested_xyz,
                requested_rpy=requested_rpy,
                executed_delta_xyz=zero,
                executed_delta_rpy=zero,
                before_summary=before_summary,
                after_summary=before_summary,
                clipped=clip_reason is not None,
                clip_reason=clip_reason,
            )

        step_xyz = xyz / float(planned_steps)
        step_rpy = rpy / float(planned_steps)
        executed_xyz = zero.copy()
        executed_rpy = zero.copy()
        after_summary = before_summary
        ok = True
        reason = "completed"

        for _step_index in range(run_steps):
            action = np.concatenate(
                [
                    step_xyz.astype(np.float32),
                    step_rpy.astype(np.float32),
                    np.array([0.0], dtype=np.float32),
                ]
            )
            last_action = self._env.take_action(action, action_type="delta_ee")
            if bool(last_action.get("success_latched", False)):
                ok = True
                reason = "native_success"
                break
            action_ok = bool(last_action.get("ok", False))
            if action_ok:
                executed_xyz += step_xyz
                executed_rpy += step_rpy
                self._record_move_relative_usage(step_xyz, step_rpy)
            after_summary = self._read_move_tactile_summary()
            status = self._finalize_motion_action()
            if not action_ok:
                ok = False
                reason = str(last_action.get("reason", "move_relative_action_failed"))
                break
            if bool(tactile_guard):
                event = str(after_summary.get("event", "unknown"))
                contact = bool(after_summary.get("contact", False))
                if not contact or event == "contact_lost":
                    ok = False
                    reason = "contact_lost"
                    self._lock_move_relative_guard(reason)
                    break
                if downward_guarded:
                    after_tactile = self._simple_tactile_feedback(
                        before_summary,
                        after_summary,
                    )
                    if self._should_interrupt_for_slip_risk(before_tactile, after_tactile):
                        ok = True
                        reason = "interrupted_by_slip_warning"
                        break
            if isinstance(status, dict) and bool(status.get("stopped", False)):
                protocol_reason = str(status.get("reason") or "official_protocol_stopped")
                ok = protocol_reason == "native_success"
                reason = protocol_reason
                break

        if (
            ok
            and run_steps < planned_steps
            and reason not in {"interrupted_by_slip_warning", "native_success"}
        ):
            if downward_guarded and str(before_tactile.get("slip_risk")) == "medium":
                reason = "guarded_micro_step"
            else:
                ok = False
                reason = "max_actions"

        return self._finish_move_relative_result(
            ok=ok,
            reason=reason,
            requested_xyz=requested_xyz,
            requested_rpy=requested_rpy,
            executed_delta_xyz=executed_xyz,
            executed_delta_rpy=executed_rpy,
            before_summary=before_summary,
            after_summary=after_summary,
            interrupted=reason == "interrupted_by_slip_warning",
            clipped=clip_reason is not None,
            clip_reason=clip_reason,
        )

    def _finish_move_relative_result(
        self,
        *,
        ok: bool,
        reason: str,
        requested_xyz: np.ndarray,
        requested_rpy: np.ndarray,
        executed_delta_xyz: np.ndarray,
        executed_delta_rpy: np.ndarray,
        before_summary: dict[str, Any] | None,
        after_summary: dict[str, Any] | None,
        preempted: bool = False,
        interrupted: bool = False,
        clipped: bool = False,
        clip_reason: str | None = None,
    ) -> dict[str, Any]:
        result = self._move_relative_result(
            ok=ok,
            reason=reason,
            executed_delta_xyz=executed_delta_xyz,
            executed_delta_rpy=executed_delta_rpy,
            before_summary=before_summary,
            after_summary=after_summary,
            preempted=preempted,
            interrupted=interrupted,
            clipped=clipped,
            clip_reason=clip_reason,
        )
        self._log_move_relative_result(
            result,
            requested_xyz,
            requested_rpy,
            executed_delta_xyz,
            executed_delta_rpy,
        )
        return result

    def _log_move_relative_result(
        self,
        result: dict[str, Any],
        requested_xyz: np.ndarray,
        requested_rpy: np.ndarray,
        executed_xyz: np.ndarray,
        executed_rpy: np.ndarray,
    ) -> None:
        print(
            "[univtac-franka] move_relative "
            f"ok={result['ok']} reason={result['reason']} "
            f"depth={result['executed_depth']:.4f} "
            f"distance={result['executed_distance']:.4f} "
            f"requested_xyz={self._format_vec3(requested_xyz)} "
            f"executed_xyz={self._format_vec3(executed_xyz)} "
            f"requested_rpy={self._format_vec3(requested_rpy)} "
            f"executed_rpy={self._format_vec3(executed_rpy)} "
            f"clipped={result.get('clipped')} "
            f"clip_reason={result.get('clip_reason')} "
            f"preempted={result.get('preempted')} "
            f"interrupted={result.get('interrupted')} "
            f"risk_before={result.get('risk_before')} "
            f"risk_after={result.get('risk_after')} "
            f"action_count={result.get('action_count')} "
            f"max_steps={result.get('max_steps')} "
            f"remaining_actions={result.get('remaining_actions')} "
            f"episode_stopped={result.get('episode_stopped')} "
            f"success_latched={result.get('success_latched')} "
            f"tactile={result['tactile']}",
            flush=True,
        )
        self._log_insert_depth_for_move_relative(result)

    def _is_downward_insertion(self, xyz: np.ndarray) -> bool:
        return bool(float(np.asarray(xyz, dtype=np.float32).reshape(3)[2]) < -1e-9)

    def _should_interrupt_for_slip_risk(
        self,
        before_tactile: dict[str, Any],
        after_tactile: dict[str, Any],
    ) -> bool:
        if bool(after_tactile.get("slip", False)):
            return True
        if bool(after_tactile.get("incipient_slip", False)):
            return True
        before_score = self._risk_label_score(str(before_tactile.get("slip_risk", "low")))
        after_score = self._risk_label_score(str(after_tactile.get("slip_risk", "low")))
        return bool(after_score > before_score and after_score >= 0.5)

    @staticmethod
    def _risk_label_score(label: str) -> float:
        value = str(label)
        if value == "high":
            return 1.0
        if value == "medium":
            return 0.5
        return 0.0

    @staticmethod
    def _ceil_step_count(distance: float, step: float) -> int:
        step = max(float(step), 1e-9)
        ratio = max(0.0, float(distance)) / step
        return max(1, int(np.ceil(max(0.0, ratio - 1e-6))))

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
            self._env.wait_steps(self._limited_gripper_settle_steps())

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
        calibration.update(self._tactile_summary_thresholds())
        return summarize_native_tactile(frames, hand="both", **calibration)

    def _save_adaptive_trace(self, trace: list[dict[str, Any]]) -> None:
        save_fn = getattr(self._env, "append_tactile_gripper_trace", None)
        if callable(save_fn):
            save_fn(trace)

    def _rgbd_frame(self):
        frame_fn = getattr(self._env, "get_rgbd_frame", None)
        if not callable(frame_fn):
            raise RuntimeError("UniVTAC environment does not provide calibrated RGB-D")
        return frame_fn(self.perception_camera)

    def _estimate_rgbd_with_retry(self, key: str, *, kind: str) -> tuple[Any, Any, str, int]:
        prompts = self._perception_prompts(key)
        source = "rgbd_contact_graspnet" if kind == "grasp_pose" else "rgbd"
        errors: list[dict[str, Any]] = []
        for attempt in range(1, self.perception_retry_attempts + 1):
            try:
                frame = self._rgbd_frame()
            except Exception as exc:
                error = self._record_rgbd_failure(
                    key=key,
                    kind=kind,
                    source=source,
                    attempt=attempt,
                    prompt=None,
                    frame=None,
                    exc=exc,
                )
                errors.append(error)
                continue

            for prompt in prompts:
                try:
                    if kind == "grasp_pose":
                        estimate = self._rgbd_perception.estimate_grasp(frame, prompt)
                        extra = f"candidates={len(estimate.scores)}"
                    elif kind == "object_pose":
                        estimate = self._rgbd_perception.estimate_object(frame, prompt)
                        extra = f"points={len(estimate.points_world)}"
                    else:
                        raise ValueError(f"unknown RGB-D estimate kind {kind!r}")
                except Exception as exc:
                    error = self._record_rgbd_failure(
                        key=key,
                        kind=kind,
                        source=source,
                        attempt=attempt,
                        prompt=prompt,
                        frame=frame,
                        exc=exc,
                    )
                    errors.append(error)
                    continue

                print(
                    "[univtac-franka] rgbd_retry "
                    f"kind={kind} object={key} attempt={attempt} "
                    f"prompt={prompt!r} ok=True {extra}",
                    flush=True,
                )
                return estimate, frame, prompt, attempt

        reason = "rgbd_regrasp_unavailable" if kind == "grasp_pose" else "rgbd_pose_unavailable"
        last = errors[-1] if errors else {"reason": "unknown", "message": "no attempts"}
        raise RecoverableTaskFailure(
            reason,
            (
                f"RGB-D {kind} for {key!r} failed after "
                f"{self.perception_retry_attempts} attempt(s): "
                f"{last.get('reason')}: {last.get('message')}"
            ),
            details={
                "object_name": key,
                "kind": kind,
                "attempts": self.perception_retry_attempts,
                "errors": errors,
            },
        )

    def _record_rgbd_failure(
        self,
        *,
        key: str,
        kind: str,
        source: str,
        attempt: int,
        prompt: str | None,
        frame: Any | None,
        exc: Exception,
    ) -> dict[str, Any]:
        diagnostics = dict(getattr(exc, "diagnostics", {}) or {})
        reason = str(getattr(exc, "reason", type(exc).__name__))
        message = str(exc)
        mask = getattr(exc, "mask", None)
        error = {
            "kind": kind,
            "source": source,
            "object_name": key,
            "attempt": int(attempt),
            "prompt": prompt,
            "reason": reason,
            "message": message,
            "diagnostics": diagnostics,
        }
        record: dict[str, Any] = {
            **error,
            "kind": f"{kind}_error",
            "error": repr(exc),
        }
        if frame is not None:
            record["frame"] = frame
        if mask is not None:
            record["mask"] = mask
        self._append_perception_artifact(record)
        print(
            "[univtac-franka] rgbd_retry "
            f"kind={kind} object={key} attempt={attempt} "
            f"prompt={prompt!r} ok=False reason={reason} "
            f"valid_depth_points={diagnostics.get('valid_depth_points', 'n/a')}",
            flush=True,
        )
        return error

    def _official_anchor_fallback(
        self,
        key: str,
        *,
        fallback_from: Exception,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.official_anchor_fallback_enabled:
            return None
        normalized = str(key).strip().lower().replace(" ", "_")
        if normalized not in self.official_anchor_fallback_objects:
            return None
        grasp_pose = self._public_grasp_pose(normalized)
        if grasp_pose is None:
            return None
        pos, quat = grasp_pose
        reason = str(getattr(fallback_from, "reason", type(fallback_from).__name__))
        self._append_perception_artifact(
            {
                "kind": "grasp_pose",
                "source": "official_anchor_fallback",
                "object_name": normalized,
                "position": pos,
                "quaternion_wxyz": quat,
                "fallback_from": reason,
                "used_for_control": True,
            }
        )
        print(
            "[univtac-franka] grasp_source=official_anchor_fallback "
            f"object={normalized} fallback_from={reason}",
            flush=True,
        )
        return pos, quat

    @staticmethod
    def _normalize_prompt_map(raw: Any) -> dict[str, list[str]]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, list[str]] = {}
        for key, value in raw.items():
            normalized = str(key).strip().lower().replace(" ", "_")
            if isinstance(value, (list, tuple)):
                prompts = [str(item).strip() for item in value if str(item).strip()]
            else:
                prompt = str(value).strip()
                prompts = [prompt] if prompt else []
            if prompts:
                out[normalized] = prompts
        return out

    @staticmethod
    def _optional_positive_int(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        out = int(value)
        return out if out > 0 else None

    @staticmethod
    def _normalize_object_set(raw: Any, *, object_pose_names: dict[str, str]) -> set[str]:
        if raw is None:
            return set()
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, (list, tuple, set)):
            values = list(raw)
        else:
            values = [raw]
        normalized: set[str] = set()
        for value in values:
            key = str(value).strip().lower()
            if not key:
                continue
            mapped = object_pose_names.get(key, key.replace(" ", "_"))
            normalized.add(str(mapped).strip().lower().replace(" ", "_"))
        return normalized

    @staticmethod
    def _default_tactile_guard_config() -> dict[str, float]:
        return {
            "slip_warning_threshold": 0.35,
            "slip_high_threshold": 0.55,
            "slip_hard_threshold": 0.60,
            "centroid_warning_delta": 0.5,
            "centroid_high_delta": 1.0,
            "shear_warning_delta": 0.05,
            "guard_micro_down_step": 0.0005,
            "guard_micro_lateral_step": 0.0002,
            "guard_micro_rpy_step": 0.002,
        }

    def _tactile_summary_thresholds(self) -> dict[str, float]:
        return {
            key: float(self.tactile_guard_config[key])
            for key in (
                "slip_warning_threshold",
                "slip_high_threshold",
                "slip_hard_threshold",
                "centroid_warning_delta",
                "centroid_high_delta",
                "shear_warning_delta",
            )
            if key in self.tactile_guard_config
        }

    def _perception_prompts(self, key: str) -> list[str]:
        normalized = str(key).strip().lower().replace(" ", "_")
        prompts: list[str] = []
        prompts.extend(self.perception_prompt_map.get(normalized, []))
        prompts.extend(self.perception_prompt_fallbacks.get(normalized, []))
        unique = list(dict.fromkeys(prompt for prompt in prompts if prompt))
        if not unique:
            raise KeyError(
                f"object '{key}' has no non-privileged RGB-D perception prompt"
            )
        return unique

    def _perception_prompt(self, key: str) -> str:
        return self._perception_prompts(key)[0]

    def _append_perception_artifact(self, record: dict[str, Any]) -> None:
        append_fn = getattr(self._env, "append_perception_artifact", None)
        if callable(append_fn):
            append_fn(record)

    def _record_rgbd_diagnostic_if_needed(self) -> None:
        reset_serial_fn = getattr(self._env, "get_reset_serial", None)
        reset_serial = int(reset_serial_fn()) if callable(reset_serial_fn) else 0
        if (
            self._perception_diagnostic_reset_serial == reset_serial
            or not self.record_perception_diagnostic
            or not self.rgbd_perception_enabled
        ):
            return
        self._perception_diagnostic_reset_serial = reset_serial
        frame = None
        try:
            frame = self._rgbd_frame()
            prompt = self._perception_prompt("can")
            estimate = self._rgbd_perception.estimate_grasp(frame, prompt)
            self._append_perception_artifact(
                {
                    "kind": "diagnostic_grasp_candidates",
                    "source": "rgbd_contact_graspnet_private_diagnostic",
                    "object_name": "can",
                    "prompt": prompt,
                    "frame": frame,
                    "mask": estimate.mask,
                    "points_world": estimate.points_world,
                    "position": estimate.position,
                    "quaternion_wxyz": estimate.quaternion_wxyz,
                    "obb_position": estimate.object_position,
                    "obb_quaternion_wxyz": estimate.object_quaternion_wxyz,
                    "obb_extent": estimate.object_extent,
                    "grasps_camera": estimate.grasps_camera,
                    "grasp_scores": estimate.scores,
                    "selected_index": estimate.selected_index,
                    "used_for_control": False,
                }
            )
            print(
                "[univtac-franka] saved private RGB-D grasp diagnostic "
                f"candidates={len(estimate.scores)} used_for_control=False",
                flush=True,
            )
        except Exception as exc:
            record: dict[str, Any] = {
                "kind": "diagnostic_error",
                "source": "rgbd_private_diagnostic",
                "object_name": "can",
                "error": repr(exc),
                "used_for_control": False,
            }
            if frame is not None:
                record["frame"] = frame
            self._append_perception_artifact(record)
            print(
                f"WARNING: private RGB-D perception diagnostic failed: {exc!r}",
                flush=True,
            )

    def _goto_pose_native(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        z_approach: float,
    ) -> dict[str, Any]:
        move_fn = getattr(self._env, "move_to_tool_pose_native", None)
        if not callable(move_fn):
            raise RuntimeError("UniVTAC environment does not provide native pose planning")
        target = np.asarray(position, dtype=np.float32).reshape(3)
        quat = self._normalize_quat(quaternion_wxyz)
        result: dict[str, Any] = {"ok": True, "message": "no movement requested"}
        approach = max(0.0, float(z_approach))
        required_actions = 2 if approach > 0.0 else 1
        limit_failure = self._api_action_limit_failure(
            "goto_pose_native",
            required_actions,
            self.max_native_pose_actions,
        )
        if limit_failure is not None:
            return limit_failure
        if approach > 0.0:
            approach_target = target.copy()
            approach_target[2] = max(target[2] + approach, self.min_safe_z)
            result = move_fn(approach_target, quat)
            if not bool(result.get("ok", False)):
                self._finalize_high_level_action()
                return result
        result = move_fn(target, quat)
        self._finalize_high_level_action()
        print(
            "[univtac-franka] goto_pose_via_native_planner "
            f"target={np.array2string(target, precision=3)} "
            f"ok={bool(result.get('ok', False))}",
            flush=True,
        )
        return result

    def _begin_gripper_action(self) -> bool:
        begin_fn = getattr(self._env, "begin_high_level_action", None)
        return bool(begin_fn()) if callable(begin_fn) else True

    def _finalize_high_level_action(self) -> None:
        finalize_fn = getattr(self._env, "finalize_high_level_action", None)
        if callable(finalize_fn):
            finalize_fn()

    def _protocol_status(self) -> dict[str, Any]:
        status_fn = getattr(self._env, "get_protocol_status", None)
        status = status_fn() if callable(status_fn) else {}
        return status if isinstance(status, dict) else {}

    def _success_latched_result(self, api_name: str) -> dict[str, Any] | None:
        status = self._protocol_status()
        if not bool(status.get("success_latched", False)):
            return None
        return {
            "ok": True,
            "reason": "native_success",
            "message": f"official success already latched; {api_name} sent no physical command",
            "episode_stopped": True,
            "success_latched": True,
            "action_count": status.get("action_count"),
            "max_steps": status.get("max_steps"),
        }

    def _success_latched_move_relative_result(
        self,
        tactile_summary: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if self._success_latched_result("move_relative") is None:
            return None
        return self._move_relative_result(
            ok=True,
            reason="native_success",
            executed_delta_xyz=np.zeros(3, dtype=np.float32),
            executed_delta_rpy=np.zeros(3, dtype=np.float32),
            before_summary=tactile_summary or {},
            after_summary=tactile_summary or {},
        )

    def _blocked_gripper_result(self, *, opening: bool) -> dict[str, Any]:
        status = self._protocol_status()
        success_latched = (
            bool(status.get("success_latched", False)) if isinstance(status, dict) else False
        )
        reason = (
            str(status.get("reason") or "official_protocol_stopped")
            if isinstance(status, dict)
            else "official_protocol_stopped"
        )
        result: dict[str, Any] = {
            "ok": success_latched,
            "reason": "native_success" if success_latched else reason,
            "message": (
                "official success already latched; no gripper command was sent"
                if success_latched
                else "official protocol has stopped; no physical command was sent"
            ),
            "episode_stopped": (
                bool(status.get("episode_stopped", False)) if isinstance(status, dict) else True
            ),
            "success_latched": success_latched,
        }
        if opening:
            result["released"] = False
        else:
            result.update({"stable": success_latched, "contact": success_latched})
        if isinstance(status, dict):
            result["action_count"] = status.get("action_count")
        return result

    def _move_to_pose_bounded(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
        *,
        max_actions: int | None = None,
    ) -> dict[str, Any]:
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
        limit_failure = self._api_action_limit_failure(
            "goto_pose",
            steps,
            max_actions,
        )
        if limit_failure is not None:
            return limit_failure
        pos_points = [current_pos + (delta_pos * (i / steps)) for i in range(1, steps + 1)]

        quat_points = self._slerp_quaternion_path(current_quat, target_quat, steps)
        last_result: dict[str, Any] = {
            "ok": True,
            "message": "bounded pose movement completed",
            "steps": steps,
        }
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
            last_result = {
                **result,
                "steps": steps,
                "completed_steps": idx + 1,
            }
            if bool(result.get("success_latched", False)):
                return last_result
            if not result.get("ok", False):
                last_result.setdefault("reason", "bounded_pose_action_failed")
                print(
                    "[univtac-franka] goto_pose_bounded_stopped "
                    f"step={idx + 1}/{steps} message={result.get('message', '')}",
                    flush=True,
                )
                return last_result
        return last_result

    def _call_move_to_pose_bounded(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
        *,
        max_actions: int | None,
    ) -> dict[str, Any]:
        if max_actions is None:
            result = self._move_to_pose_bounded(
                target_pos,
                target_quat,
                current_pos,
                current_quat,
            )
        else:
            result = self._move_to_pose_bounded(
                target_pos,
                target_quat,
                current_pos,
                current_quat,
                max_actions=max_actions,
            )
        return result if isinstance(result, dict) else {"ok": True, "message": "movement completed"}

    def _api_action_limit_failure(
        self,
        api_name: str,
        requested_actions: int,
        configured_limit: int | None,
    ) -> dict[str, Any] | None:
        effective_limit = self._effective_action_limit(configured_limit)
        if effective_limit is None or requested_actions <= effective_limit:
            return None
        status_fn = getattr(self._env, "get_protocol_status", None)
        status = status_fn() if callable(status_fn) else {}
        if isinstance(status, dict) and bool(status.get("success_latched", False)):
            return self._success_latched_result(api_name)
        print(
            "[univtac-franka] api_action_limit "
            f"api={api_name} requested={requested_actions} limit={effective_limit}",
            flush=True,
        )
        return {
            "ok": False,
            "reason": "api_action_limit",
            "message": (
                f"{api_name} would require {requested_actions} physical action(s), "
                f"but this API call is limited to {effective_limit}"
            ),
            "requested_actions": int(requested_actions),
            "max_api_actions": int(effective_limit),
            "completed_steps": 0,
            "action_count": status.get("action_count") if isinstance(status, dict) else None,
            "max_steps": status.get("max_steps") if isinstance(status, dict) else None,
        }

    def _effective_action_limit(self, configured_limit: int | None) -> int | None:
        limits: list[int] = []
        if configured_limit is not None:
            limits.append(max(0, int(configured_limit)))
        remaining = self._remaining_protocol_actions()
        if remaining is not None:
            limits.append(max(0, remaining))
        return min(limits) if limits else None

    def _remaining_protocol_actions(self) -> int | None:
        status_fn = getattr(self._env, "get_protocol_status", None)
        if not callable(status_fn):
            return None
        status = status_fn()
        if not isinstance(status, dict) or not bool(status.get("enabled", False)):
            return None
        if bool(status.get("stopped", False)):
            return 0
        max_steps = status.get("max_steps")
        action_count = status.get("action_count")
        if max_steps is None or action_count is None:
            return None
        return max(0, int(max_steps) - int(action_count))

    @staticmethod
    def _consume_api_actions(limit: int | None, result: dict[str, Any]) -> int | None:
        if limit is None:
            return None
        used = int(result.get("completed_steps", result.get("steps", 0)) or 0)
        return max(0, int(limit) - used)

    def _limit_gripper_servo_steps(self, requested_max_steps: int) -> int:
        requested = max(1, int(requested_max_steps))
        if self.max_gripper_servo_steps is None:
            return requested
        return min(requested, self.max_gripper_servo_steps)

    @staticmethod
    def _annotate_servo_limit(
        result: dict[str, Any],
        requested_max_steps: int,
        actual_max_steps: int,
    ) -> None:
        if int(requested_max_steps) == int(actual_max_steps):
            return
        result["requested_max_steps"] = int(requested_max_steps)
        result["max_steps_limit"] = int(actual_max_steps)

    def _limited_gripper_settle_steps(self) -> int:
        requested = max(0, int(self.gripper_settle_steps))
        if self.max_gripper_settle_steps is None:
            return requested
        return min(requested, self.max_gripper_settle_steps)

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

    def _ensure_insert_depth_reference(self) -> None:
        if getattr(self, "task_name", "") != "insert_hole" or self.insert_depth_log_interval <= 0:
            return
        if self._insert_depth_reference_z is not None:
            return
        try:
            pos, _quat = self._current_tool_pose()
        except Exception:
            return
        self._insert_depth_reference_z = float(np.asarray(pos, dtype=np.float32).reshape(3)[2])
        status = self._protocol_status()
        print(
            "[univtac-insert-depth] "
            f"reference_z={self._insert_depth_reference_z:.4f} "
            f"action_count={status.get('action_count')}",
            flush=True,
        )

    def _maybe_log_insert_depth(self, status: dict[str, Any] | None = None) -> None:
        if getattr(self, "task_name", "") != "insert_hole" or self.insert_depth_log_interval <= 0:
            return
        self._ensure_insert_depth_reference()
        reference_z = self._insert_depth_reference_z
        if reference_z is None:
            return
        current_status = status if isinstance(status, dict) else self._protocol_status()
        action_count = current_status.get("action_count")
        if action_count is None:
            return
        try:
            action_count_int = int(action_count)
        except Exception:
            return
        interval = int(self.insert_depth_log_interval)
        bucket = action_count_int // interval
        if bucket <= self._last_insert_depth_log_bucket:
            return
        try:
            pos, _quat = self._current_tool_pose()
        except Exception:
            return
        current_z = float(np.asarray(pos, dtype=np.float32).reshape(3)[2])
        inserted_depth = max(0.0, float(reference_z) - current_z)
        max_steps = current_status.get("max_steps")
        remaining_actions = None
        if max_steps is not None:
            try:
                remaining_actions = int(max_steps) - action_count_int
            except Exception:
                remaining_actions = None
        print(
            "[univtac-insert-depth] "
            f"step_mark={bucket * interval} "
            f"action_count={action_count_int} "
            f"inserted_depth_world={inserted_depth:.4f} "
            f"reference_z={float(reference_z):.4f} "
            f"current_z={current_z:.4f} "
            f"remaining_actions={remaining_actions}",
            flush=True,
        )
        self._last_insert_depth_log_bucket = bucket

    def _log_insert_depth_for_move_relative(self, result: dict[str, Any]) -> None:
        if getattr(self, "task_name", "") != "insert_hole" or self.insert_depth_log_interval <= 0:
            return
        self._ensure_insert_depth_reference()
        reference_z = self._insert_depth_reference_z
        if reference_z is None:
            return
        try:
            pos, _quat = self._current_tool_pose()
        except Exception:
            return
        current_z = float(np.asarray(pos, dtype=np.float32).reshape(3)[2])
        inserted_depth = max(0.0, float(reference_z) - current_z)
        print(
            "[univtac-insert-depth] move_relative_depth "
            f"inserted_depth_world={inserted_depth:.4f} "
            f"reference_z={float(reference_z):.4f} "
            f"current_z={current_z:.4f} "
            f"executed_depth={float(result.get('executed_depth', 0.0)):.4f} "
            f"action_count={result.get('action_count')} "
            f"remaining_actions={result.get('remaining_actions')} "
            f"reason={result.get('reason')}",
            flush=True,
        )

    def _finalize_motion_action(self) -> dict[str, Any]:
        finalize_fn = getattr(self._env, "finalize_high_level_action", None)
        if callable(finalize_fn):
            return finalize_fn()
        status_fn = getattr(self._env, "get_protocol_status", None)
        status = status_fn() if callable(status_fn) else {}
        return status if isinstance(status, dict) else {}

    def _read_move_tactile_summary(self) -> dict[str, Any]:
        try:
            summary = self._read_adaptive_tactile_summary()
        except Exception:
            summary = {}
        return summary if isinstance(summary, dict) else {}

    def _move_relative_limits(self) -> dict[str, float]:
        remaining_lateral = max(
            0.0,
            float(self.move_relative_lateral_budget) - float(self._move_relative_lateral_used),
        )
        remaining_rotation = max(
            0.0,
            float(self.move_relative_rotation_budget) - float(self._move_relative_rotation_used),
        )
        return {
            "recommended_down_step": float(self.guard_micro_down_step),
            "max_lateral_step": float(self.move_relative_max_lateral),
            "max_rotation_step": float(self.move_relative_max_rotation),
            "remaining_lateral_budget": remaining_lateral,
            "remaining_rotation_budget": remaining_rotation,
        }

    def _clip_move_relative_request(
        self,
        xyz: np.ndarray,
        rpy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str | None]:
        clipped_xyz = np.asarray(xyz, dtype=np.float32).reshape(3).copy()
        clipped_rpy = np.asarray(rpy, dtype=np.float32).reshape(3).copy()
        reasons: list[str] = []

        lateral = float(np.linalg.norm(clipped_xyz[:2]))
        max_lateral = max(0.0, float(self.move_relative_max_lateral))
        if lateral > max_lateral + 1e-9:
            if max_lateral <= 0.0:
                clipped_xyz[:2] = 0.0
            else:
                clipped_xyz[:2] *= max_lateral / max(lateral, 1e-9)
            reasons.append("move_relative_lateral_step_limit")

        rotation = float(np.linalg.norm(clipped_rpy))
        max_rotation = max(0.0, float(self.move_relative_max_rotation))
        if rotation > max_rotation + 1e-9:
            if max_rotation <= 0.0:
                clipped_rpy[:] = 0.0
            else:
                clipped_rpy *= max_rotation / max(rotation, 1e-9)
            reasons.append("move_relative_rotation_step_limit")

        lateral = float(np.linalg.norm(clipped_xyz[:2]))
        remaining_lateral = max(
            0.0,
            float(self.move_relative_lateral_budget) - float(self._move_relative_lateral_used),
        )
        if lateral > remaining_lateral + 1e-9:
            if remaining_lateral <= 1e-9:
                clipped_xyz[:2] = 0.0
            else:
                clipped_xyz[:2] *= remaining_lateral / max(lateral, 1e-9)
            reasons.append("move_relative_lateral_budget")

        rotation = float(np.linalg.norm(clipped_rpy))
        remaining_rotation = max(
            0.0,
            float(self.move_relative_rotation_budget) - float(self._move_relative_rotation_used),
        )
        if rotation > remaining_rotation + 1e-9:
            if remaining_rotation <= 1e-9:
                clipped_rpy[:] = 0.0
            else:
                clipped_rpy *= remaining_rotation / max(rotation, 1e-9)
            reasons.append("move_relative_rotation_budget")

        if not reasons:
            return clipped_xyz, clipped_rpy, None
        return clipped_xyz, clipped_rpy, reasons[-1]

    def _record_move_relative_usage(self, xyz: np.ndarray, rpy: np.ndarray) -> None:
        self._move_relative_lateral_used += float(
            np.linalg.norm(np.asarray(xyz, dtype=np.float32).reshape(3)[:2])
        )
        self._move_relative_rotation_used += float(
            np.linalg.norm(np.asarray(rpy, dtype=np.float32).reshape(3))
        )

    def _lock_move_relative_guard(self, reason: str) -> None:
        if self.move_relative_lock_on_guard_failure:
            self._move_relative_guard_locked_reason = str(reason)

    @staticmethod
    def _format_vec3(values: np.ndarray) -> str:
        vec = np.asarray(values, dtype=np.float32).reshape(3)
        return "[" + ", ".join(f"{float(v):.4f}" for v in vec) + "]"

    def _move_relative_result(
        self,
        *,
        ok: bool,
        reason: str,
        executed_delta_xyz: np.ndarray,
        executed_delta_rpy: np.ndarray,
        before_summary: dict[str, Any] | None,
        after_summary: dict[str, Any] | None,
        preempted: bool = False,
        interrupted: bool = False,
        clipped: bool = False,
        clip_reason: str | None = None,
    ) -> dict[str, Any]:
        executed_xyz = np.asarray(executed_delta_xyz, dtype=np.float32).reshape(3)
        before_tactile = self._simple_tactile_feedback({}, before_summary or {})
        tactile = self._simple_tactile_feedback(before_summary or {}, after_summary or {})
        result_reason = str(reason)
        if bool(ok) and result_reason == "completed":
            if bool(tactile.get("slip", False)):
                result_reason = "completed_with_slip"
            elif (
                bool(tactile.get("contact", False))
                and (
                    not bool(tactile.get("stable", False))
                    or str(tactile.get("drift", "unknown")) == "high"
                )
            ):
                result_reason = "completed_unstable"
        status = self._protocol_status()
        result = {
            "ok": bool(ok),
            "reason": result_reason,
            "executed_distance": float(np.linalg.norm(executed_xyz)),
            "executed_depth": float(max(0.0, -float(executed_xyz[2]))),
            "preempted": bool(preempted),
            "interrupted": bool(interrupted),
            "clipped": bool(clipped),
            "clip_reason": clip_reason,
            "risk_before": str(before_tactile.get("slip_risk", "unknown")),
            "risk_after": str(tactile.get("slip_risk", "unknown")),
            "limits": self._move_relative_limits(),
            "tactile": tactile,
        }
        if status:
            action_count = status.get("action_count")
            max_steps = status.get("max_steps")
            remaining_actions = None
            if action_count is not None and max_steps is not None:
                try:
                    remaining_actions = int(max_steps) - int(action_count)
                except Exception:
                    remaining_actions = None
            result.update(
                {
                    "episode_stopped": bool(status.get("episode_stopped", False)),
                    "success_latched": bool(status.get("success_latched", False)),
                    "action_count": action_count,
                    "max_steps": max_steps,
                    "remaining_actions": remaining_actions,
                }
            )
        return result

    def _simple_tactile_feedback(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> dict[str, Any]:
        slip_threshold = float(
            self.tactile_guard_config.get(
                "slip_hard_threshold",
                self.adaptive_gripper_config.get("slip_threshold", 0.60),
            )
        )
        after_slip = self._summary_float(after, "slip_score")
        event = str(after.get("event", "unknown"))
        contact = bool(after.get("contact", False))
        balance = abs(self._summary_float(after, "contact_balance"))
        stable = bool(
            contact
            and bool(after.get("left_contact", False))
            and bool(after.get("right_contact", False))
            and self._summary_float(after, "normal_force") >= 0.5
            and balance <= 0.45
            and after_slip < slip_threshold
        )
        slip_risk = str(after.get("slip_risk", self._risk_from_score(after_slip)))
        pressure_side = str(after.get("pressure_side", self._heavier_side(after)))
        shear_side = str(after.get("shear_side", "unknown"))
        drift_trend = str(after.get("drift_trend", "unknown"))
        correction_hint = str(after.get("correction_hint", "continue"))
        return {
            "contact": contact,
            "stable": stable,
            "slip": bool(event == "slip_detected" or after_slip >= slip_threshold),
            "slip_risk": self._comparable_risk_label(slip_risk),
            "incipient_slip": bool(after.get("incipient_slip", False)),
            "pressure_side": self._side_label(pressure_side),
            "shear_side": self._side_label(shear_side),
            "heavier_side": self._side_label(pressure_side),
            "force_change": self._force_change(before, after),
            "drift_trend": self._trend_label(drift_trend),
            "drift": self._drift_level(after),
            "correction_hint": self._hint_label(correction_hint),
        }

    def _risk_from_score(self, score: float) -> str:
        if float(score) >= float(self.tactile_guard_config.get("slip_high_threshold", 0.55)):
            return "high"
        if float(score) >= float(self.tactile_guard_config.get("slip_warning_threshold", 0.35)):
            return "medium"
        return "low"

    @staticmethod
    def _comparable_risk_label(value: str) -> str:
        normalized = str(value)
        scores = {"low": 0.0, "medium": 0.5, "high": 1.0}
        return _ComparableTactileLabel(normalized, scores.get(normalized, 0.0))

    @staticmethod
    def _side_label(value: str) -> str:
        normalized = str(value)
        scores = {"unknown": 0.0, "balanced": 0.0, "left": 1.0, "right": 1.0}
        return _ComparableTactileLabel(normalized, scores.get(normalized, 0.0))

    @staticmethod
    def _trend_label(value: str) -> str:
        normalized = str(value)
        scores = {
            "unknown": 0.0,
            "stable": 0.0,
            "decreased": -1.0,
            "decreasing": -1.0,
            "increased": 1.0,
            "increasing": 1.0,
        }
        return _ComparableTactileLabel(normalized, scores.get(normalized, 0.0))

    @staticmethod
    def _hint_label(value: str) -> str:
        normalized = str(value)
        scores = {
            "continue": 0.0,
            "reduce_down_step": 0.4,
            "try_pitch_probe": 0.6,
            "try_lateral_probe": 0.6,
            "hold": 1.0,
        }
        return _ComparableTactileLabel(normalized, scores.get(normalized, 0.0))

    @staticmethod
    def _summary_float(summary: dict[str, Any], key: str) -> float:
        try:
            value = summary.get(key, 0.0)
            return float(value) if value is not None else 0.0
        except Exception:
            return 0.0

    def _heavier_side(self, summary: dict[str, Any]) -> str:
        if not summary:
            return _ComparableTactileLabel("unknown", 0.0)
        left = summary.get("left", {}) if isinstance(summary.get("left", {}), dict) else {}
        right = summary.get("right", {}) if isinstance(summary.get("right", {}), dict) else {}
        left_force = self._summary_float(left, "normal_force")
        right_force = self._summary_float(right, "normal_force")
        if left_force <= 0.0 and right_force <= 0.0:
            return _ComparableTactileLabel("unknown", 0.0)
        if abs(left_force - right_force) <= 0.05:
            return _ComparableTactileLabel("balanced", 0.0)
        return _ComparableTactileLabel("left" if left_force > right_force else "right", 1.0)

    def _force_change(self, before: dict[str, Any], after: dict[str, Any]) -> str:
        if not before or not after:
            return _ComparableTactileLabel("unknown", 0.0)
        delta = self._summary_float(after, "normal_force") - self._summary_float(before, "normal_force")
        if delta > 0.05:
            return _ComparableTactileLabel("increased", 1.0)
        if delta < -0.05:
            return _ComparableTactileLabel("decreased", -1.0)
        return _ComparableTactileLabel("stable", 0.0)

    def _drift_level(self, summary: dict[str, Any]) -> str:
        if not summary:
            return _ComparableTactileLabel("unknown", 0.0)
        drift = self._summary_float(summary, "marker_centroid_displacement")
        if drift >= 1.5:
            return _ComparableTactileLabel("high", 1.0)
        if drift >= 0.5:
            return _ComparableTactileLabel("medium", 0.5)
        return _ComparableTactileLabel("low", 0.0)

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
        if key != "prism":
            sampled = self._sample_public_grasp_from_env(key)
            if sampled is not None:
                return sampled

        if self._public_actor_pose(key) is None:
            return None
        sampled = self._sample_public_grasp_from_env(key)
        if sampled is not None:
            return sampled
        if key == "prism":
            pos = self._public_prism_pose()
            if pos is not None:
                grasp_pos = pos.copy()
                grasp_pos[2] += self.grasp_height
                return grasp_pos.astype(np.float32), self._native_grasp_quat_wxyz()
        return None

    def _sample_public_grasp_from_env(self, key: str) -> tuple[np.ndarray, np.ndarray] | None:
        sample_fn = getattr(self._env, "get_public_grasp_pose", None)
        if not callable(sample_fn):
            return None
        try:
            sampled = sample_fn(key, grasp_height=self.grasp_height)
        except (KeyError, RuntimeError, ValueError):
            return None
        if sampled is None:
            return None
        pos, quat = sampled
        return (
            np.asarray(pos, dtype=np.float32).reshape(3),
            self._normalize_quat(np.asarray(quat, dtype=np.float32).reshape(4)),
        )

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
