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
        task_place_landmark_names: list[str] | tuple[str, ...] | None = None,
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
        rgbd_pose_enabled: bool | None = None,
        rgbd_grasp_enabled: bool | None = None,
        rgbd_pose_objects: list[str] | tuple[str, ...] | None = None,
        rgbd_grasp_objects: list[str] | tuple[str, ...] | None = None,
        public_anchor_pose_enabled: bool = True,
        public_pose_fallback_objects: list[str] | tuple[str, ...] | None = None,
        public_grasp_anchor_objects: list[str] | tuple[str, ...] | None = None,
        cache_rgbd_pose_objects: list[str] | tuple[str, ...] | None = None,
        perception_selector_map: dict[str, str] | None = None,
        perception_camera: str = "head",
        perception_prompt_map: dict[str, str] | None = None,
        sam3_service_url: str = "http://127.0.0.1:8114",
        graspnet_service_url: str = "http://127.0.0.1:8115",
        perception_timeout_seconds: float = 120.0,
        perception_min_depth_points: int = 32,
        perception_retry_attempts: int = 2,
        perception_prompt_fallbacks: dict[str, list[str]] | None = None,
        grasp_local_z_offset: float = 0.12,
        current_tool_pose_grasp_objects: list[str] | tuple[str, ...] | None = None,
        use_native_pose_planner: bool = False,
        record_perception_diagnostic: bool = False,
        official_anchor_fallback_enabled: bool = False,
        official_anchor_fallback_objects: list[str] | None = None,
        max_goto_pose_actions: int | None = None,
        max_home_pose_actions: int | None = None,
        max_native_pose_actions: int | None = None,
        max_gripper_servo_steps: int | None = None,
        min_open_gripper_servo_steps: int | None = None,
        max_gripper_settle_steps: int | None = None,
        goto_pose_tactile_guard_force_threshold: float = 0.30,
        goto_pose_tactile_guard_slip_threshold: float = 0.60,
    ) -> None:
        super().__init__(env)
        cfg = self._runtime_config()
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
        self.object_pose_names = dict(cfg.get("object_pose_names", object_pose_names or {})) or {
            "prism": "prism",
            "object": "prism",
            "target object": "prism",
            "orange pad": "orange_pad",
            "orange_pad": "orange_pad",
            "green pad": "green_pad",
            "green_pad": "green_pad",
            "current object": "current_object",
            "current_object": "current_object",
            "current slot": "current_slot",
            "current_slot": "current_slot",
            "object a": "object_a",
            "object_a": "object_a",
            "object b": "object_b",
            "object_b": "object_b",
            "slot a": "slot_a",
            "slot_a": "slot_a",
            "slot b": "slot_b",
            "slot_b": "slot_b",
            "reference": "reference_object",
            "reference_object": "reference_object",
            "candidate left": "candidate_left",
            "candidate_left": "candidate_left",
            "left candidate": "candidate_left",
            "left_candidate": "candidate_left",
            "candidate right": "candidate_right",
            "candidate_right": "candidate_right",
            "right candidate": "candidate_right",
            "right_candidate": "candidate_right",
            "match slot": "match_slot",
            "match_slot": "match_slot",
        }
        configured_place_landmarks = cfg.get(
            "task_place_landmark_names",
            task_place_landmark_names,
        )
        self.task_place_landmark_names = self._optional_normalized_object_set(
            configured_place_landmarks,
            object_pose_names=self.object_pose_names,
        )
        self.rgbd_perception_enabled = bool(
            cfg.get("rgbd_perception_enabled", rgbd_perception_enabled)
        )
        self.rgbd_pose_enabled = bool(
            cfg.get(
                "rgbd_pose_enabled",
                self.rgbd_perception_enabled if rgbd_pose_enabled is None else rgbd_pose_enabled,
            )
        )
        self.rgbd_grasp_enabled = bool(
            cfg.get(
                "rgbd_grasp_enabled",
                self.rgbd_perception_enabled
                if rgbd_grasp_enabled is None
                else rgbd_grasp_enabled,
            )
        )
        pose_objects_raw = cfg.get("rgbd_pose_objects", rgbd_pose_objects)
        grasp_objects_raw = cfg.get("rgbd_grasp_objects", rgbd_grasp_objects)
        public_pose_raw = cfg.get(
            "public_pose_fallback_objects",
            public_pose_fallback_objects,
        )
        public_grasp_raw = cfg.get(
            "public_grasp_anchor_objects",
            public_grasp_anchor_objects,
        )
        cache_pose_raw = cfg.get("cache_rgbd_pose_objects", cache_rgbd_pose_objects)
        self.rgbd_pose_objects = self._optional_normalized_object_set(
            pose_objects_raw,
            object_pose_names=self.object_pose_names,
        )
        self.rgbd_grasp_objects = self._optional_normalized_object_set(
            grasp_objects_raw,
            object_pose_names=self.object_pose_names,
        )
        self.public_anchor_pose_enabled = bool(
            cfg.get("public_anchor_pose_enabled", public_anchor_pose_enabled)
        )
        self.public_pose_fallback_objects = self._normalize_object_set(
            public_pose_raw,
            object_pose_names=self.object_pose_names,
        )
        self.public_grasp_anchor_objects = self._optional_normalized_object_set(
            public_grasp_raw,
            object_pose_names=self.object_pose_names,
        )
        self.cache_rgbd_pose_objects = self._normalize_object_set(
            cache_pose_raw,
            object_pose_names=self.object_pose_names,
        )
        self.perception_selector_map = self._normalize_selector_map(
            cfg.get("perception_selector_map", perception_selector_map or {}),
            object_pose_names=self.object_pose_names,
        )
        self._rgbd_pose_cache: dict[str, Any] = {}
        self._holding_with_tactile = False
        current_tool_pose_objects = cfg.get(
            "current_tool_pose_grasp_objects",
            current_tool_pose_grasp_objects or (),
        )
        if isinstance(current_tool_pose_objects, str):
            current_tool_pose_objects = [current_tool_pose_objects]
        self.current_tool_pose_grasp_objects = {
            self._normalize_name(str(name)) for name in current_tool_pose_objects
        }
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
        self.min_open_gripper_servo_steps = self._optional_positive_int(
            cfg.get("min_open_gripper_servo_steps", min_open_gripper_servo_steps),
        )
        if (
            self.min_open_gripper_servo_steps is not None
            and self.max_gripper_servo_steps is not None
        ):
            self.min_open_gripper_servo_steps = min(
                self.min_open_gripper_servo_steps,
                self.max_gripper_servo_steps,
            )
        self.max_gripper_settle_steps = self._optional_positive_int(
            cfg.get("max_gripper_settle_steps", max_gripper_settle_steps),
        )
        self.goto_pose_tactile_guard_force_threshold = float(
            cfg.get(
                "goto_pose_tactile_guard_force_threshold",
                goto_pose_tactile_guard_force_threshold,
            )
        )
        self.goto_pose_tactile_guard_slip_threshold = float(
            cfg.get(
                "goto_pose_tactile_guard_slip_threshold",
                goto_pose_tactile_guard_slip_threshold,
            )
        )
        self.llm_api_profile = str(cfg.get("llm_api_profile", "")).strip().lower()
        visible_functions = cfg.get("llm_visible_functions")
        if visible_functions is not None and not isinstance(visible_functions, (list, tuple)):
            raise ValueError("llm_visible_functions must be a list when configured")
        self.llm_visible_functions = (
            {str(name).strip() for name in visible_functions}
            if visible_functions is not None
            else None
        )
        self.local_delta_max_m = float(cfg.get("local_delta_max_m", 0.02))
        self.local_delta_segment_m = float(cfg.get("local_delta_segment_m", 0.002))
        self.local_yaw_max_rad = float(cfg.get("local_yaw_max_rad", 0.18))
        self.local_yaw_segment_rad = float(cfg.get("local_yaw_segment_rad", 0.09))
        self.local_wait_max_steps = int(cfg.get("local_wait_max_steps", 20))
        self.profile_open_width = float(cfg.get("profile_open_width", self.open_gripper_width))
        self.profile_open_max_steps = int(
            cfg.get("profile_open_max_steps", self.max_gripper_servo_steps or 120)
        )
        self.transport_close_target_force = float(
            cfg.get("transport_close_target_force", 0.82)
        )
        self.transport_close_max_steps = int(
            cfg.get("transport_close_max_steps", self.max_gripper_servo_steps or 120)
        )
        for name, value in (
            ("local_delta_max_m", self.local_delta_max_m),
            ("local_delta_segment_m", self.local_delta_segment_m),
            ("local_yaw_max_rad", self.local_yaw_max_rad),
            ("local_yaw_segment_rad", self.local_yaw_segment_rad),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite value")
        if self.local_delta_segment_m > self.local_delta_max_m:
            raise ValueError("local_delta_segment_m cannot exceed local_delta_max_m")
        if self.local_yaw_segment_rad > self.local_yaw_max_rad:
            raise ValueError("local_yaw_segment_rad cannot exceed local_yaw_max_rad")
        if self.local_wait_max_steps < 1:
            raise ValueError("local_wait_max_steps must be positive")
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
        full = {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "get_robot_state": self.get_robot_state,
            "home_pose": self.home_pose,
            "get_step_status": self.get_step_status,
            "wait_steps": self.wait_steps,
        }
        if self.llm_api_profile == "tactile_memory_match":
            full = {
                "get_object_pose": self._memory_match_get_object_pose,
                "sample_grasp_pose": self.sample_grasp_pose,
                "goto_pose": self._memory_match_goto_pose,
                "move_delta": self._memory_match_move_delta,
                "rotate_gripper": self._memory_match_rotate_gripper,
                "open_gripper": self._memory_match_open_gripper,
                "close_gripper": self._memory_match_close_gripper,
                "wait_steps": self._memory_match_wait_steps,
            }
        return self._filter_llm_visible_functions(full)

    def _filter_llm_visible_functions(self, functions: dict[str, Any]) -> dict[str, Any]:
        if self.llm_visible_functions is None:
            return functions
        unknown = self.llm_visible_functions.difference(functions)
        if unknown:
            raise ValueError(
                "llm_visible_functions contains unsupported FrankaControlApi functions: "
                f"{sorted(unknown)}"
            )
        return {name: functions[name] for name in functions if name in self.llm_visible_functions}

    def _memory_match_get_object_pose(
        self,
        object_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the configured public pose route for one named object or slot.

        Easy-GT and Hard-SAM choose their source in YAML. Generated code cannot
        override that route or request extra geometry.
        """
        return self.get_object_pose(object_name, source="auto")

    def _memory_match_goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> dict[str, Any]:
        """Move to a semantic grasp, slot, or locally derived target pose.

        The adapter keeps tactile transport guarding active whenever it owns a
        stable grasp. Public task landmarks retain their native staged motion.
        """
        return self.goto_pose(position, quaternion_wxyz)

    def _memory_match_move_delta(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
    ) -> dict[str, Any]:
        """Apply one bounded local translation in world coordinates.

        The requested displacement is limited by the configured local motion
        budget and split into short protected UniVTAC actions. It cannot move
        below the configured gripper-center safety height.
        """
        requested = np.asarray([dx, dy, dz], dtype=np.float32)
        if not np.all(np.isfinite(requested)):
            raise ValueError("move_delta requires finite dx, dy, dz")
        distance = float(np.linalg.norm(requested))
        if distance > self.local_delta_max_m + 1e-8:
            return {
                "ok": False,
                "reason": "local_delta_limit",
                "requested_distance_m": distance,
                "max_distance_m": self.local_delta_max_m,
            }
        current_pos, current_quat = self._current_tool_pose()
        target_pos = current_pos + requested
        target_pos[2] = max(float(target_pos[2]), self.min_safe_z)
        return self._memory_match_execute_relative_pose(
            target_pos,
            current_quat,
            translation_segment_m=self.local_delta_segment_m,
            yaw_segment_rad=self.local_yaw_segment_rad,
            operation="move_delta",
        )

    def _memory_match_rotate_gripper(self, yaw_rad: float) -> dict[str, Any]:
        """Apply a bounded yaw about the current local gripper tool axis.

        Roll and pitch are intentionally unavailable in the tactile-memory
        profile to prevent a local correction from tilting into the table.
        """
        yaw = float(yaw_rad)
        if not np.isfinite(yaw):
            raise ValueError("yaw_rad must be finite")
        if abs(yaw) > self.local_yaw_max_rad + 1e-8:
            return {
                "ok": False,
                "reason": "local_yaw_limit",
                "requested_yaw_rad": yaw,
                "max_yaw_rad": self.local_yaw_max_rad,
            }
        current_pos, current_quat = self._current_tool_pose()
        current_rotation = SciRotation.from_quat(self._wxyz_to_xyzw(current_quat))
        local_yaw = SciRotation.from_rotvec([0.0, 0.0, yaw])
        target_xyzw = (current_rotation * local_yaw).as_quat()
        target_quat = self._normalize_quat(
            np.asarray(
                [target_xyzw[3], target_xyzw[0], target_xyzw[1], target_xyzw[2]],
                dtype=np.float32,
            )
        )
        return self._memory_match_execute_relative_pose(
            current_pos,
            target_quat,
            translation_segment_m=self.local_delta_segment_m,
            yaw_segment_rad=self.local_yaw_segment_rad,
            operation="rotate_gripper",
        )

    def _memory_match_open_gripper(self) -> dict[str, Any]:
        """Release using the task-profile opening width and servo budget."""
        return self.open_gripper(
            adaptive=True,
            target_width=self.profile_open_width,
            max_steps=self.profile_open_max_steps,
        )

    def _memory_match_close_gripper(self, mode: str = "probe") -> dict[str, Any]:
        """Close using the configured ``probe`` or ``transport`` policy.

        ``probe`` reuses the public measurement protocol so reference and
        candidates receive identical excitation. ``transport`` uses the
        independently configured final-grasp policy.
        """
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"probe", "transport"}:
            raise ValueError("close_gripper mode must be 'probe' or 'transport'")
        protocol = self._memory_match_protocol()
        if normalized_mode == "probe":
            target_force = float(protocol["close_target_force"])
            max_steps = int(protocol["close_max_steps"])
            adaptive = bool(protocol["adaptive_close"])
        else:
            target_force = self.transport_close_target_force
            max_steps = self.transport_close_max_steps
            adaptive = True
        result = self.close_gripper(
            adaptive=adaptive,
            target_force=target_force,
            max_steps=max_steps,
        )
        result["mode"] = normalized_mode
        return result

    def _memory_match_wait_steps(self, n: int = 1) -> dict[str, Any]:
        """Advance a bounded number of simulation steps without new motion."""
        steps = int(n)
        if steps < 1 or steps > self.local_wait_max_steps:
            raise ValueError(
                f"wait_steps must be in [1, {self.local_wait_max_steps}] for this task"
            )
        return self.wait_steps(steps)

    def _memory_match_protocol(self) -> dict[str, Any]:
        task_spec_getter = getattr(self._env, "get_public_probe_spec", None)
        if callable(task_spec_getter):
            try:
                task_spec = task_spec_getter()
            except Exception:
                task_spec = None
            if (
                isinstance(task_spec, dict)
                and task_spec.get("schema_version") == "public_probe_spec.v1"
                and "close_target_force" in task_spec
                and "close_max_steps" in task_spec
            ):
                return {
                    "close_target_force": float(
                        np.clip(task_spec["close_target_force"], 0.0, 1.0)
                    ),
                    "close_max_steps": max(1, int(task_spec["close_max_steps"])),
                    "adaptive_close": bool(task_spec.get("adaptive_close", True)),
                }
        configs = getattr(self._env, "api_configs", {})
        source = configs.get("tactile_measurement_protocol", {}) if isinstance(configs, dict) else {}
        if not isinstance(source, dict):
            source = {}
        return {
            "close_target_force": float(np.clip(source.get("close_target_force", 0.82), 0.0, 1.0)),
            "close_max_steps": max(1, int(source.get("close_max_steps", 120))),
            "adaptive_close": bool(source.get("adaptive_close", True)),
        }

    def _memory_match_execute_relative_pose(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        *,
        translation_segment_m: float,
        yaw_segment_rad: float,
        operation: str,
    ) -> dict[str, Any]:
        start_pos, start_quat = self._current_tool_pose()
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)
        target_quat = self._normalize_quat(np.asarray(target_quat, dtype=np.float32).reshape(4))
        translation = float(np.linalg.norm(target_pos - start_pos))
        rotation = self._rotation_angle(start_quat, target_quat)
        # Remove numerical dust before ceil so an exact 6 mm command with a
        # 2 mm segment budget does not gain a fourth, unnecessary action.
        # The action vectors are float32, so allow a tiny relative tolerance
        # at an exact segment boundary rather than emitting an extra command.
        segment_tolerance = 1e-5
        translation_steps = int(
            np.ceil(
                max(0.0, translation / max(translation_segment_m, 1e-6) - segment_tolerance)
            )
        )
        rotation_steps = int(
            np.ceil(
                max(0.0, rotation / max(yaw_segment_rad, 1e-6) - segment_tolerance)
            )
        )
        steps = max(1, translation_steps, rotation_steps)
        points = [start_pos + ((target_pos - start_pos) * (idx / steps)) for idx in range(1, steps + 1)]
        # The generic path includes its start pose; local commands must emit
        # exactly ``steps`` non-noop increments.
        quaternions = self._slerp_quaternion_path(start_quat, target_quat, steps + 1)[1:]
        monitor_tactile = bool(self._holding_with_tactile)
        last_result: dict[str, Any] = {"ok": True, "operation": operation, "steps": steps}
        previous_pos = start_pos
        previous_quat = start_quat
        for index, (position, quaternion) in enumerate(zip(points, quaternions, strict=True), start=1):
            delta_xyz = np.asarray(position - previous_pos, dtype=np.float32)
            delta_rpy = self._quaternion_delta_to_rpy(previous_quat, quaternion)
            result = self._env.take_action(
                np.concatenate([delta_xyz, delta_rpy, [0.0]]),
                action_type="delta_ee",
            )
            last_result = {**result, "operation": operation, "steps": steps, "completed_steps": index}
            if not bool(result.get("ok", False)):
                last_result.setdefault("reason", "local_motion_failed")
                return last_result
            if monitor_tactile:
                guard_failure = self._tactile_guard_failure(
                    abort_on_contact_loss=True,
                    tactile_force_threshold=None,
                    slip_threshold=None,
                )
                if guard_failure is not None:
                    return {
                        **last_result,
                        **guard_failure,
                        "tactile_monitoring": True,
                    }
            previous_pos = position
            previous_quat = quaternion
        return last_result

    def get_object_pose(
        self,
        object_name: str,
        return_bbox_extent: bool = False,
        source: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate an object pose or read a configured public landmark.

        Args:
            object_name: Public object or landmark name, including the active
                lift-can object under the name "can".
            return_bbox_extent: Whether to also return an approximate extent.
            source: ``"auto"`` follows the YAML route, ``"rgbd"`` forces
                configured RGB-D/SAM perception, and ``"anchor"`` forces
                explicitly public reset/slot anchors.

        Returns:
            ``(position, quaternion_wxyz)`` by default. When
            ``return_bbox_extent=True``, returns
            ``(position, quaternion_wxyz, bbox_extent)``.
        """
        key = self._resolve_pose_key(object_name)
        source_key = self._normalize_pose_source(source)
        if source_key == "rgbd" and not self._should_use_rgbd_pose(key):
            raise KeyError(
                f"object '{object_name}' is not available by configured RGB-D pose route"
            )
        if source_key == "anchor":
            if not self.public_anchor_pose_enabled:
                raise KeyError(
                    f"object '{object_name}' is not available by configured public anchor route"
                )
            public_pose = self._public_anchor_pose(key)
            if public_pose is None:
                raise KeyError(f"object '{object_name}' is not available as a public anchor")
            pos, quat, extent = public_pose
            print(
                "[univtac-franka] pose_source=public_anchor "
                f"object={key}",
                flush=True,
            )
            if return_bbox_extent:
                return pos, quat, extent
            return pos, quat

        if source_key == "rgbd" or self._should_use_rgbd_pose(key):
            cached = self._cached_rgbd_pose(key)
            if cached is not None:
                estimate = cached
                print(
                    "[univtac-franka] pose_source=rgbd_cache "
                    f"object={key} points={len(estimate.points_world)} "
                    f"score={estimate.score:.3f}",
                    flush=True,
                )
                if return_bbox_extent:
                    return estimate.position, estimate.quaternion_wxyz, estimate.extent
                return estimate.position, estimate.quaternion_wxyz
            estimate, frame, prompt, attempt = self._estimate_rgbd_with_retry(
                key,
                kind="object_pose",
            )
            if self._normalize_name(key) in self.cache_rgbd_pose_objects:
                self._rgbd_pose_cache[self._normalize_name(key)] = estimate
            selector = self._perception_selector(key)
            self._append_perception_artifact(
                {
                    "kind": "object_pose",
                    "source": "rgbd",
                    "object_name": key,
                    "prompt": prompt,
                    "selector": selector,
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

        if self._rgbd_pose_routes_are_restricted():
            public_pose = self._public_pose_fallback(key)
            if public_pose is None:
                raise KeyError(
                    f"object '{object_name}' is not available by configured "
                    "non-privileged pose route"
                )
            pos, quat, extent = public_pose
            print(
                "[univtac-franka] pose_source=public_fallback "
                f"object={key}",
                flush=True,
            )
            if return_bbox_extent:
                return pos, quat, extent
            return pos, quat

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
        if self._normalize_name(key) in self.current_tool_pose_grasp_objects:
            tool_pos, tool_quat = self._current_tool_pose()
            print(
                "[univtac-franka] grasp_source=current_tool_pose "
                f"object={key} pos={np.round(tool_pos, 4)}",
                flush=True,
            )
            return tool_pos, tool_quat
        if self._should_use_rgbd_grasp(key):
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
                    "selector": self._perception_selector(key),
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

        if not self._allows_public_grasp_anchor(key):
            raise KeyError(
                f"object '{object_name}' is not available for configured public "
                "grasp anchor sampling"
            )
        grasp_pose = self._public_grasp_pose(key)
        if grasp_pose is not None:
            return grasp_pose
        if key == "can":
            raise KeyError("object 'can' is not available for UniVTAC grasp sampling")
        tool_pos, tool_quat = self._current_tool_pose()
        return tool_pos, tool_quat

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
        monitor_tactile: bool = False,
        abort_on_contact_loss: bool = True,
        tactile_force_threshold: float | None = None,
        slip_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Move to a target pose using bounded UniVTAC delta actions.

        ``monitor_tactile=True`` is intended for transport/descent while an
        object is already grasped. The call stops early if native tactile
        feedback indicates contact loss, weak holding force, or slip.
        """
        pos = np.asarray(position, dtype=np.float32).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=np.float32).reshape(4)

        active_tactile_monitor = bool(monitor_tactile or self._holding_with_tactile)
        approach = max(0.0, float(z_approach))
        if not active_tactile_monitor:
            # A sampled grasp pose is a semantic task target, not a generic
            # end-effector pose. Resolve it through the task atom first so
            # UniVTAC applies its object geometry, pre-displacement, and
            # grasp-height convention. This intentionally retains the legacy
            # behavior for callers that supplied a positive approach offset.
            grasp_result = self._try_approach_public_grasp(pos)
            if grasp_result is not None:
                return grasp_result

        # A configured public placement landmark remains meaningful while
        # tactile holding is active. The task bridge performs the safe
        # clearance/horizontal/descent chain internally.
        place_result = self._try_place_on_public_landmark(pos, quat)
        if place_result is not None:
            return place_result

        if self.use_native_pose_planner and not active_tactile_monitor:
            return self._goto_pose_native(pos, quat, z_approach=float(z_approach))

        cur_pos, cur_quat = self._current_tool_pose()
        if self._nearest_public_landmark(pos) is not None and self.preserve_landmark_orientation:
            quat = cur_quat

        # Preserve the original CaP contract: zero means a direct bounded move.
        # Callers request a staged approach explicitly with a positive value.
        # ``max_goto_pose_actions`` is a per-segment safety cap; long motions
        # are split by the adapter instead of failing before transport.
        api_action_limit = self.max_goto_pose_actions
        if approach > 0.0:
            approach_target = pos.copy()
            approach_target[2] = max(approach_target[2] + approach, self.min_safe_z)
            result = self._call_move_to_pose_bounded(
                approach_target,
                quat,
                cur_pos,
                cur_quat,
                max_actions=api_action_limit,
                monitor_tactile=active_tactile_monitor,
                abort_on_contact_loss=abort_on_contact_loss,
                tactile_force_threshold=tactile_force_threshold,
                slip_threshold=slip_threshold,
            )
            if isinstance(result, dict) and not bool(result.get("ok", False)):
                return result
            cur_pos, cur_quat = self._current_tool_pose()

        final_target = pos.copy()
        final_target[2] = max(final_target[2], self.min_safe_z)
        return self._call_move_to_pose_bounded(
            final_target,
            quat,
            cur_pos,
            cur_quat,
            max_actions=api_action_limit,
            monitor_tactile=active_tactile_monitor,
            abort_on_contact_loss=abort_on_contact_loss,
            tactile_force_threshold=tactile_force_threshold,
            slip_threshold=slip_threshold,
        )

    def open_gripper(
        self,
        adaptive: bool = True,
        target_width: float = 1.0,
        max_steps: int = 160,
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
        max_steps = self._limit_open_gripper_servo_steps(requested_max_steps)
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
                f"width={result['width']:.4f} steps={result['steps']} "
                f"holding_cleared={self._release_clears_tactile_holding(result)}",
                flush=True,
            )
            if bool(result.get("holding_cleared", False)):
                self._holding_with_tactile = False
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
            if self._release_clears_tactile_holding(output):
                self._holding_with_tactile = False
            self._finalize_high_level_action()
            return output
        self._move_gripper(target_width)
        output = {
            "ok": True,
            "released": True,
            "reason": "fixed_open",
            "target_width": float(np.clip(target_width, 0.0, 1.0)),
        }
        self._holding_with_tactile = False
        self._finalize_high_level_action()
        return output

    def _release_clears_tactile_holding(self, result: dict[str, Any]) -> bool:
        if bool(result.get("released", False)):
            result["holding_cleared"] = True
            result["holding_clear_reason"] = "released"
            return True
        if result.get("contact") is False:
            result["holding_cleared"] = True
            result["holding_clear_reason"] = "no_tactile_contact"
            return True
        try:
            summary = self._read_adaptive_tactile_summary()
        except Exception as exc:
            result["holding_cleared"] = False
            result["holding_clear_error"] = str(exc)
            return False

        contact = bool(summary.get("contact", False))
        force = float(summary.get("normal_force", 0.0))
        force_threshold = float(
            self.adaptive_gripper_config.get("contact_force_threshold", 0.08)
        )
        result["release_contact"] = contact
        result["release_force"] = force
        if (not contact) or force <= force_threshold:
            result["holding_cleared"] = True
            result["holding_clear_reason"] = "release_tactile_low_contact"
            return True
        result["holding_cleared"] = False
        return False

    def close_gripper(
        self,
        adaptive: bool = True,
        target_force: float = 0.35,
        max_steps: int = 80,
        target_depth_delta_mm: float | None = None,
        min_stable_contact_area: float | None = None,
        post_squeeze_qpos: float | None = None,
        post_squeeze_steps: int | None = None,
        hold_steps: int | None = None,
    ) -> dict[str, Any]:
        """Close the gripper using optional feedback-driven width control.

        Args:
            adaptive: Use feedback-driven coarse/fine closing when enabled by config.
            target_force: Normalized target force used for stable-contact stop.
            max_steps: Maximum tactile servo iterations.
            target_depth_delta_mm: Optional minimum per-pad compression before
                accepting a stable grasp.
            min_stable_contact_area: Optional minimum per-pad contact area.
            post_squeeze_qpos: Optional extra qpos squeeze after first stable contact.
            post_squeeze_steps: Number of extra squeeze commands.
            hold_steps: Number of no-op confirmation commands after squeezing.

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
            result = controller.close(
                target_force=target_force,
                max_steps=max_steps,
                target_depth_delta_mm=target_depth_delta_mm,
                min_stable_contact_area=min_stable_contact_area,
                post_squeeze_qpos=post_squeeze_qpos,
                post_squeeze_steps=post_squeeze_steps,
                hold_steps=hold_steps,
            )
            self._annotate_servo_limit(result, requested_max_steps, max_steps)
            self._save_adaptive_trace(controller.trace)
            print(
                "[univtac-franka] adaptive_close "
                f"stable={result['stable']} reason={result['reason']} "
                f"force={result['normal_force']:.3f} depth={result['depth_delta_mm']:.3f}mm "
                f"area={result['contact_area']:.4f} width={result['width']:.4f} "
                f"post_squeeze={result['post_squeeze_applied']} hold={result['hold_steps']} "
                f"steps={result['steps']}",
                flush=True,
            )
            self._holding_with_tactile = bool(result.get("stable", False))
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
            self._holding_with_tactile = bool(output.get("stable", False))
            self._finalize_high_level_action()
            return output
        self._move_gripper(0.0)
        output = {
            "ok": True,
            "stable": False,
            "reason": "fixed_close_requires_tactile_confirmation",
        }
        self._holding_with_tactile = False
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

    def get_robot_state(self) -> dict[str, Any]:
        """Return public robot state for local tactile servoing.

        The returned dict includes ``ee_pos`` and ``ee_quat`` for the current
        gripper/tool pose, plus public joint state when available. Use this for
        small local adjustments, lift motions, and retreat motions. Do not use
        ``get_object_pose("current_object")`` as a substitute for the current
        gripper pose; object anchors are coarse task anchors, not live tracking.
        """
        state = dict(self._env.get_robot_state())
        task = getattr(self._env, "task", None)
        robot_manager = getattr(task, "_robot_manager", None)
        if robot_manager is None:
            return state

        try:
            pose = robot_manager.get_gripper_center_pose()
            tool_pos = np.asarray(pose.p, dtype=np.float32).reshape(3)
            tool_quat = self._normalize_quat(np.asarray(pose.q, dtype=np.float32).reshape(4))
        except Exception:
            return state

        raw_ee_pos = state.get("ee_pos")
        raw_ee_quat = state.get("ee_quat")
        raw_ee_pose = state.get("ee_pose") or state.get("ee")
        if raw_ee_pos is not None:
            state.setdefault("raw_ee_pos", raw_ee_pos)
        if raw_ee_quat is not None:
            state.setdefault("raw_ee_quat", raw_ee_quat)
        if raw_ee_pose is not None:
            state.setdefault("raw_ee_pose", raw_ee_pose)

        # CaP-facing pose APIs use the gripper center as the control frame.
        # Keep ee_pos/ee_quat in that same frame so relative motions such as
        # ``get_robot_state()["ee_pos"][2] += dz`` remain consistent with
        # ``goto_pose``.
        state["ee_pos"] = tool_pos.tolist()
        state["ee_quat"] = tool_quat.tolist()
        state["ee_pose"] = [*state["ee_pos"], *state["ee_quat"]]
        state["ee"] = state["ee_pose"]
        state["tool_pos"] = state["ee_pos"]
        state["tool_quat"] = state["ee_quat"]
        state["tool_pose"] = state["ee_pose"]
        state["control_frame"] = "gripper_center"
        return state

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
            target_depth_delta_mm=float(cfg.get("target_depth_delta_mm", 0.0)),
            min_stable_contact_area=float(cfg.get("min_stable_contact_area", 0.0)),
            post_squeeze_qpos=float(cfg.get("post_squeeze_qpos", 0.0)),
            post_squeeze_steps=int(cfg.get("post_squeeze_steps", 0)),
            hold_steps=int(cfg.get("hold_steps", 0)),
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

    def _rgbd_frame(self):
        frame_fn = getattr(self._env, "get_rgbd_frame", None)
        if not callable(frame_fn):
            raise RuntimeError("UniVTAC environment does not provide calibrated RGB-D")
        return frame_fn(self.perception_camera)

    def _estimate_rgbd_with_retry(self, key: str, *, kind: str) -> tuple[Any, Any, str, int]:
        prompts = self._perception_prompts(key)
        source = "rgbd_contact_graspnet" if kind == "grasp_pose" else "rgbd"
        selector = self._perception_selector(key)
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
                    selector=selector,
                    frame=None,
                    exc=exc,
                )
                errors.append(error)
                continue

            for prompt in prompts:
                try:
                    if kind == "grasp_pose":
                        if selector == "best_score":
                            estimate = self._rgbd_perception.estimate_grasp(frame, prompt)
                        else:
                            estimate = self._rgbd_perception.estimate_grasp(
                                frame,
                                prompt,
                                selector=selector,
                            )
                        extra = f"candidates={len(estimate.scores)}"
                    elif kind == "object_pose":
                        if selector == "best_score":
                            estimate = self._rgbd_perception.estimate_object(frame, prompt)
                        else:
                            estimate = self._rgbd_perception.estimate_object(
                                frame,
                                prompt,
                                selector=selector,
                            )
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
                        selector=selector,
                        frame=frame,
                        exc=exc,
                    )
                    errors.append(error)
                    continue

                print(
                    "[univtac-franka] rgbd_retry "
                    f"kind={kind} object={key} attempt={attempt} "
                    f"prompt={prompt!r} selector={selector} ok=True {extra}",
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
        selector: str,
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
            "selector": selector,
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
            f"prompt={prompt!r} selector={selector} ok=False reason={reason} "
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
    def _optional_normalized_object_set(
        raw: Any,
        *,
        object_pose_names: dict[str, str],
    ) -> set[str] | None:
        if raw is None:
            return None
        return UniVTACFrankaCompatApi._normalize_object_set(
            raw,
            object_pose_names=object_pose_names,
        )

    @staticmethod
    def _normalize_selector_map(
        raw: Any,
        *,
        object_pose_names: dict[str, str],
    ) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in raw.items():
            raw_key = str(key).strip().lower()
            if not raw_key:
                continue
            mapped = object_pose_names.get(raw_key, raw_key.replace(" ", "_"))
            selector = str(value).strip().lower().replace("-", "_")
            if selector:
                out[str(mapped).strip().lower().replace(" ", "_")] = selector
        return out

    def _should_use_rgbd_pose(self, key: str) -> bool:
        if not self.rgbd_pose_enabled:
            return False
        normalized = self._normalize_name(key)
        return self.rgbd_pose_objects is None or normalized in self.rgbd_pose_objects

    def _should_use_rgbd_grasp(self, key: str) -> bool:
        if not self.rgbd_grasp_enabled:
            return False
        normalized = self._normalize_name(key)
        return self.rgbd_grasp_objects is None or normalized in self.rgbd_grasp_objects

    def _rgbd_pose_routes_are_restricted(self) -> bool:
        return self.rgbd_pose_enabled and self.rgbd_pose_objects is not None

    @staticmethod
    def _normalize_pose_source(source: str | None) -> str:
        raw = "auto" if source is None else str(source).strip().lower()
        aliases = {
            "": "auto",
            "default": "auto",
            "public": "anchor",
            "public_anchor": "anchor",
            "task_anchor": "anchor",
            "sam": "rgbd",
            "rgb_d": "rgbd",
        }
        normalized = aliases.get(raw.replace("-", "_"), raw.replace("-", "_"))
        if normalized not in {"auto", "rgbd", "anchor"}:
            raise ValueError("source must be one of 'auto', 'rgbd', or 'anchor'")
        return normalized

    def _allows_public_grasp_anchor(self, key: str) -> bool:
        normalized = self._normalize_name(key)
        return (
            self.public_grasp_anchor_objects is None
            or normalized in self.public_grasp_anchor_objects
        )

    def _public_pose_fallback(
        self,
        key: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        normalized = self._normalize_name(key)
        if normalized not in self.public_pose_fallback_objects:
            return None
        return self._public_landmarks().get(normalized)

    def _public_anchor_pose(
        self,
        key: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        normalized = self._normalize_name(key)
        return self._public_anchor_landmarks().get(normalized)

    def _cached_rgbd_pose(self, key: str) -> Any | None:
        normalized = self._normalize_name(key)
        if normalized not in self.cache_rgbd_pose_objects:
            return None
        return self._rgbd_pose_cache.get(normalized)

    def _perception_selector(self, key: str) -> str:
        return self.perception_selector_map.get(self._normalize_name(key), "best_score")

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
        requested_z = float(target[2])
        target[2] = max(target[2], self.min_safe_z)
        if requested_z < self.min_safe_z:
            print(
                "[univtac-franka] native_target_z_clamped "
                f"requested={requested_z:.4f} safe_z={self.min_safe_z:.4f}",
                flush=True,
            )
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

    def _blocked_gripper_result(self, *, opening: bool) -> dict[str, Any]:
        status_fn = getattr(self._env, "get_protocol_status", None)
        status = status_fn() if callable(status_fn) else {}
        result: dict[str, Any] = {
            "ok": False,
            "reason": "official_protocol_stopped",
            "message": "official protocol has stopped; no physical command was sent",
        }
        if opening:
            result["released"] = False
        else:
            result.update({"stable": False, "contact": False})
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
        monitor_tactile: bool = False,
        abort_on_contact_loss: bool = True,
        tactile_force_threshold: float | None = None,
        slip_threshold: float | None = None,
    ) -> dict[str, Any]:
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)
        target_quat = np.asarray(target_quat, dtype=np.float32).reshape(4)
        current_pos = np.asarray(current_pos, dtype=np.float32).reshape(3)
        current_quat = np.asarray(current_quat, dtype=np.float32).reshape(4)

        delta_pos = target_pos - current_pos
        steps = self._pose_step_count(current_pos, current_quat, target_pos, target_quat)
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
            if not result.get("ok", False):
                last_result.setdefault("reason", "bounded_pose_action_failed")
                print(
                    "[univtac-franka] goto_pose_bounded_stopped "
                    f"step={idx + 1}/{steps} message={result.get('message', '')}",
                    flush=True,
                )
                return last_result
            if monitor_tactile:
                guard_failure = self._tactile_guard_failure(
                    abort_on_contact_loss=abort_on_contact_loss,
                    tactile_force_threshold=tactile_force_threshold,
                    slip_threshold=slip_threshold,
                )
                if guard_failure is not None:
                    guarded_result = {
                        **last_result,
                        **guard_failure,
                        "steps": steps,
                        "completed_steps": idx + 1,
                        "tactile_monitoring": True,
                    }
                    print(
                        "[univtac-franka] goto_pose_tactile_guard_stopped "
                        f"reason={guarded_result.get('reason')} step={idx + 1}/{steps}",
                        flush=True,
                    )
                    return guarded_result
        return last_result

    def _pose_step_count(
        self,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> int:
        current_pos = np.asarray(current_pos, dtype=np.float32).reshape(3)
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)
        current_quat = np.asarray(current_quat, dtype=np.float32).reshape(4)
        target_quat = np.asarray(target_quat, dtype=np.float32).reshape(4)
        dist = float(np.linalg.norm(target_pos - current_pos))
        rot_angle = self._rotation_angle(current_quat, target_quat)
        return max(
            1,
            int(np.ceil(dist / max(self.max_delta_xyz, 1e-6))),
            int(np.ceil(rot_angle / max(self.max_delta_rpy, 1e-6))),
        )

    def _tactile_guard_failure(
        self,
        *,
        abort_on_contact_loss: bool,
        tactile_force_threshold: float | None,
        slip_threshold: float | None,
    ) -> dict[str, Any] | None:
        try:
            summary = self._read_adaptive_tactile_summary()
        except Exception as exc:
            return {
                "ok": False,
                "reason": "tactile_guard_unavailable",
                "message": f"tactile guard could not read native tactile summary: {exc}",
            }

        force_threshold = (
            self.goto_pose_tactile_guard_force_threshold
            if tactile_force_threshold is None
            else float(tactile_force_threshold)
        )
        slip_limit = (
            self.goto_pose_tactile_guard_slip_threshold
            if slip_threshold is None
            else float(slip_threshold)
        )
        contact_ok = bool(summary.get("contact", False))
        left_ok = bool(summary.get("left_contact", False))
        right_ok = bool(summary.get("right_contact", False))
        force = float(summary.get("normal_force", 0.0))
        slip = float(summary.get("slip_score", 0.0))
        event = str(summary.get("event", "unknown"))

        if slip >= slip_limit or event == "slip_detected":
            return {
                "ok": False,
                "reason": "tactile_guard_slip_detected",
                "message": "native tactile guard detected slip during goto_pose",
                "tactile_summary": summary,
            }
        if abort_on_contact_loss and (not contact_ok or not left_ok or not right_ok):
            return {
                "ok": False,
                "reason": "tactile_guard_contact_lost",
                "message": "native tactile guard detected contact loss during goto_pose",
                "tactile_summary": summary,
            }
        if abort_on_contact_loss and force < force_threshold:
            return {
                "ok": False,
                "reason": "tactile_guard_weak_contact",
                "message": "native tactile guard detected weak holding force during goto_pose",
                "tactile_summary": summary,
            }
        return None

    def _call_move_to_pose_bounded(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
        *,
        max_actions: int | None,
        monitor_tactile: bool = False,
        abort_on_contact_loss: bool = True,
        tactile_force_threshold: float | None = None,
        slip_threshold: float | None = None,
    ) -> dict[str, Any]:
        target_pos = np.asarray(target_pos, dtype=np.float32).reshape(3)
        target_quat = np.asarray(target_quat, dtype=np.float32).reshape(4)
        current_pos = np.asarray(current_pos, dtype=np.float32).reshape(3)
        current_quat = np.asarray(current_quat, dtype=np.float32).reshape(4)
        monitor_kwargs: dict[str, Any] = {}
        if monitor_tactile:
            monitor_kwargs = {
                "monitor_tactile": True,
                "abort_on_contact_loss": abort_on_contact_loss,
                "tactile_force_threshold": tactile_force_threshold,
                "slip_threshold": slip_threshold,
            }
        if max_actions is None:
            result = self._move_to_pose_bounded(
                target_pos,
                target_quat,
                current_pos,
                current_quat,
                **monitor_kwargs,
            )
        else:
            segment_limit = max(0, int(max_actions))
            total_steps = self._pose_step_count(
                current_pos,
                current_quat,
                target_pos,
                target_quat,
            )
            remaining_protocol_actions = self._remaining_protocol_actions()
            if (
                remaining_protocol_actions is not None
                and total_steps > remaining_protocol_actions
            ):
                limit_failure = self._api_action_limit_failure(
                    "goto_pose",
                    total_steps,
                    max_actions,
                )
                if limit_failure is not None:
                    return limit_failure
            if segment_limit <= 0 or total_steps <= segment_limit:
                result = self._move_to_pose_bounded(
                    target_pos,
                    target_quat,
                    current_pos,
                    current_quat,
                    max_actions=max_actions,
                    **monitor_kwargs,
                )
                return result if isinstance(result, dict) else {"ok": True, "message": "movement completed"}

            num_segments = int(np.ceil(total_steps / segment_limit))
            pos_waypoints = [
                current_pos + ((target_pos - current_pos) * (i / num_segments))
                for i in range(1, num_segments + 1)
            ]
            quat_waypoints = self._slerp_quaternion_path(
                current_quat,
                target_quat,
                num_segments + 1,
            )[1:]
            print(
                "[univtac-franka] goto_pose_auto_split "
                f"requested={total_steps} segment_limit={segment_limit} "
                f"segments={num_segments}",
                flush=True,
            )
            total_completed = 0
            last_result: dict[str, Any] = {
                "ok": True,
                "message": "split bounded pose movement completed",
            }
            for segment_idx, (segment_pos, segment_quat) in enumerate(
                zip(pos_waypoints, quat_waypoints, strict=True),
                start=1,
            ):
                segment_current_pos, segment_current_quat = self._current_tool_pose()
                result = self._move_to_pose_bounded(
                    segment_pos,
                    segment_quat,
                    segment_current_pos,
                    segment_current_quat,
                    max_actions=segment_limit,
                    **monitor_kwargs,
                )
                segment_completed = int(
                    result.get("completed_steps", result.get("steps", 0)) or 0
                )
                total_completed += max(0, segment_completed)
                last_result = dict(result)
                if not bool(result.get("ok", False)):
                    last_result.update(
                        {
                            "auto_split": True,
                            "segment_index": segment_idx,
                            "segments": num_segments,
                            "requested_actions": total_steps,
                            "max_api_actions_per_segment": segment_limit,
                            "completed_steps_total": total_completed,
                        }
                    )
                    return last_result
            last_result.update(
                {
                    "ok": True,
                    "auto_split": True,
                    "segments": num_segments,
                    "requested_actions": total_steps,
                    "max_api_actions_per_segment": segment_limit,
                    "steps": total_completed,
                    "completed_steps": total_completed,
                    "completed_steps_total": total_completed,
                    "message": "split bounded pose movement completed",
                }
            )
            return last_result
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

    def _limit_open_gripper_servo_steps(self, requested_max_steps: int) -> int:
        """Apply the configured release floor without exceeding the global cap."""
        bounded = self._limit_gripper_servo_steps(requested_max_steps)
        if self.min_open_gripper_servo_steps is None:
            return bounded
        return max(bounded, self.min_open_gripper_servo_steps)

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

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name).strip().lower().replace(" ", "_")

    def _estimate_extent_from_pose_name(self, key: str) -> np.ndarray:
        normalized = self._normalize_name(key)
        if "pad" in normalized or "slot" in normalized or "target" in normalized:
            return np.array([0.10, 0.10, 0.03], dtype=np.float32)
        if normalized == "can":
            return np.array([0.06, 0.06, 0.12], dtype=np.float32)
        if normalized in {
            "object_a",
            "object_b",
            "current_object",
            "reference",
            "reference_object",
            "candidate_left",
            "candidate_right",
            "left_candidate",
            "right_candidate",
            "candidate_1",
            "candidate_2",
        }:
            return np.array([0.04, 0.04, 0.12], dtype=np.float32)
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
        public_pose_map_fn = getattr(self._env, "get_public_pose_map", None)
        if callable(public_pose_map_fn):
            try:
                public_pose_map = public_pose_map_fn()
            except Exception:
                public_pose_map = {}
            if isinstance(public_pose_map, dict):
                for key, pose_tuple in public_pose_map.items():
                    try:
                        pos, pose_quat, extent = pose_tuple
                        landmarks[str(key)] = (
                            np.asarray(pos, dtype=np.float32).reshape(3),
                            self._normalize_quat(
                                np.asarray(pose_quat, dtype=np.float32).reshape(4)
                            ),
                            np.asarray(extent, dtype=np.float32).reshape(3),
                        )
                    except Exception:
                        continue
        return landmarks

    def _public_anchor_landmarks(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
        public_pose_map_fn = getattr(self._env, "get_public_pose_map", None)
        if callable(public_pose_map_fn):
            try:
                public_pose_map = public_pose_map_fn()
            except Exception:
                public_pose_map = {}
            if isinstance(public_pose_map, dict):
                for key, pose_tuple in public_pose_map.items():
                    try:
                        pos, pose_quat, extent = pose_tuple
                        landmarks[str(key)] = (
                            np.asarray(pos, dtype=np.float32).reshape(3),
                            self._normalize_quat(
                                np.asarray(pose_quat, dtype=np.float32).reshape(4)
                            ),
                            np.asarray(extent, dtype=np.float32).reshape(3),
                        )
                    except Exception:
                        continue
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
        candidate_keys = [
            key
            for key in self._public_landmarks()
            if key
            in {
                "prism",
                "can",
                "object_a",
                "object_b",
                "current_object",
                "reference",
                "reference_object",
                "candidate_left",
                "candidate_right",
                "left_candidate",
                "right_candidate",
                "candidate_1",
                "candidate_2",
            }
        ]
        for key in candidate_keys:
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
        if (
            self.task_place_landmark_names is not None
            and self._normalize_name(key) not in self.task_place_landmark_names
        ):
            return None
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
        normalized = self._normalize_name(key)
        if key != "prism":
            sampled = self._sample_public_grasp_from_env(normalized)
            if sampled is not None:
                return sampled
            if self.public_grasp_anchor_objects is not None:
                return None

        if self._public_actor_pose(normalized) is None:
            return None
        sampled = self._sample_public_grasp_from_env(normalized)
        if sampled is not None:
            return sampled
        if normalized == "prism":
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
