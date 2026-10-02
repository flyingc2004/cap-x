"""UniVTAC low-level environment adapter for CaP-X code execution."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from capx.envs.base import BaseEnv
from capx.envs.tasks.exceptions import HardStopTrial, RecoverableTaskFailure
from capx.integrations.univtac.native_tactile import (
    UniVTACTactileBuffer,
    frame_from_observation,
    summarize_native_tactile,
)


class UniVTACLowLevelEnv(BaseEnv):
    """Wrap a UniVTAC Task as a CaP-X low-level environment.

    This adapter deliberately bypasses UniVTAC policy modules. CaP-X generated
    code controls the task through APIs that call ``task.take_action`` directly.
    """

    record_video_during_reset = True

    def __init__(
        self,
        univtac_root: str = "/mnt/sdc/ljz/UniVTAC",
        task_name: str = "grasp_classify",
        task_config: str = "smoke_capx",
        seed_base: int = 0,
        force_task_seed: int | None = None,
        force_task: bool = False,
        device: str | None = None,
        task_config_overrides: dict[str, Any] | None = None,
        api_configs: dict[str, Any] | None = None,
        expose_actor_pose: bool = True,
        max_steps: int | None = None,
        video_size: tuple[int, int] = (960, 320),
        live_preview_enabled: bool = False,
        live_preview_path: str | os.PathLike[str] | None = (
            "/mnt/sdc/ljz/t-cap/capx-runs/latest_preview.jpg"
        ),
        live_preview_stride: int = 5,
        live_preview_jpeg_quality: int = 80,
        video_renderer: str = "capx_composed",
        tension_response_panel: dict[str, Any] | None = None,
        runtime_preflight: dict[str, Any] | None = None,
        memory_overlay_enabled: bool = False,
        selection_only_audit: bool = False,
        tactile_buffer_size: int = 500,
        lift_success_height_delta: float = 0.10,
        lift_success_require_contact: bool = True,
        privileged: bool = False,
        enable_render: bool = True,
        viser_debug: bool = False,
    ) -> None:
        super().__init__()
        self.univtac_root = Path(univtac_root).expanduser().resolve()
        self.task_name = task_name
        self.task_config_name = task_config
        self.seed_base = int(seed_base)
        self.force_task_seed = (
            int(force_task_seed) if force_task_seed is not None else None
        )
        self._force_task_requested = bool(force_task)
        self._force_task_mode = False
        # Force tasks use a target-position servo.  Keep the commanded target
        # separately from the lagging physical end-effector pose so repeated
        # small public deltas compose like the official expert controller.
        self._force_task_command_ee_pos: torch.Tensor | None = None
        self._force_task_command_ee_quat: torch.Tensor | None = None
        self.device_override = device
        self.task_config_overrides = dict(task_config_overrides or {})
        self.api_configs = api_configs or {}
        self.expose_actor_pose = bool(expose_actor_pose)
        self.max_steps = int(max_steps) if max_steps is not None else 999999
        self.video_size = tuple(video_size)
        self.live_preview_enabled = bool(live_preview_enabled)
        self.live_preview_path = (
            Path(live_preview_path).expanduser() if live_preview_path else None
        )
        self.live_preview_stride = max(1, int(live_preview_stride))
        self.live_preview_jpeg_quality = int(
            np.clip(int(live_preview_jpeg_quality), 1, 95)
        )
        self.video_renderer = str(video_renderer).strip().lower()
        if self.video_renderer not in {"capx_composed", "task_native"}:
            raise ValueError(
                "video_renderer must be 'capx_composed' or 'task_native'"
            )
        self.tension_response_panel = dict(tension_response_panel or {})
        self.tension_response_panel_enabled = bool(
            self.tension_response_panel.get("enabled", False)
        )
        self.tension_response_panel_style = str(
            self.tension_response_panel.get("style", "compact")
        ).strip().lower()
        if self.tension_response_panel_style not in {"compact", "detailed"}:
            raise ValueError(
                "tension_response_panel.style must be 'compact' or 'detailed'"
            )
        self.tension_response_panel_height = max(
            120, int(self.tension_response_panel.get("height", 240))
        )
        self.tension_response_history_points = max(
            30, int(self.tension_response_panel.get("history_points", 180))
        )
        self.runtime_preflight = dict(runtime_preflight or {})
        if self.video_renderer == "task_native" and memory_overlay_enabled:
            raise ValueError(
                "task_native video cannot be combined with memory_overlay_enabled"
            )
        self.memory_overlay_enabled = bool(memory_overlay_enabled)
        self.selection_only_audit = bool(selection_only_audit)
        self._record_action_frames = True
        self._video_frame_stride = 1
        self._record_pre_move_frames = True
        self.enable_render = enable_render
        self.viser_debug = viser_debug
        self.privileged = bool(privileged)

        self._task = None
        self._task_config: dict[str, Any] = {}
        self._current_obs: dict[str, Any] = {}
        self._record_frames = False
        self._record_wrist_camera = False
        self._frame_buffer: list[np.ndarray] = []
        self._wrist_frame_buffer: list[np.ndarray] = []
        self._video_stream_enabled = False
        self._video_stream_dir: Path | None = None
        self._video_stream_combined_writer: Any | None = None
        self._video_stream_turn_writer: Any | None = None
        self._video_stream_turn_index: int | None = None
        self._video_stream_frame_count = 0
        # OpenCV's mp4v stream writer is fast enough for simulation-time
        # recording, but not reliably playable by iPad/browser previews.
        # Native tension videos are converted after writers are closed.
        self._video_stream_h264 = self.video_renderer == "task_native"
        self._tactile_buffer = UniVTACTactileBuffer(maxlen=tactile_buffer_size)
        self.lift_success_height_delta = float(lift_success_height_delta)
        self.lift_success_require_contact = bool(lift_success_require_contact)
        self._initial_lift_object_z: float | None = None
        self._last_recorded_tactile_step: int | None = None
        self._last_recorded_video_step: int | None = None
        self._video_record_failures = 0
        self._live_preview_write_failures = 0
        self._last_action_result: dict[str, Any] = {}
        self._sim_step_count = 0
        self._start_time = time.time()
        self._debug_records: list[dict[str, Any]] = []
        self._tactile_gripper_trace: list[dict[str, Any]] = []
        self._primitive_trace: list[dict[str, Any]] = []
        self._tactile_working_memory_trace: list[dict[str, Any]] = []
        self._tactile_trial_memory_snapshot: dict[str, Any] = {}
        self._public_probe_sessions: dict[str, dict[str, Any]] = {}
        self._public_probe_records: dict[str, dict[str, Any]] = {}
        self._public_probe_serial = 0
        self._pre_move_tactile_timeline: list[dict[str, Any]] = []
        self._pre_move_tactile_last_error: str | None = None
        self._runtime_preflight_diagnostics: dict[str, Any] = {}
        self._tension_response_visualization = _empty_tension_response_visualization()
        self._tension_response_video_frames: list[Any] = []
        self._tension_response_video_history: list[dict[str, Any]] = []
        self._post_step_observers: dict[str, Callable[[], None]] = {}
        self._post_action_observers: dict[
            str, Callable[[str, dict[str, Any]], None]
        ] = {}
        self._post_observer_errors: set[tuple[str, str]] = set()
        self._opentac_tension_estimator_diagnostics: dict[str, Any] = {}
        self._perception_artifacts: list[dict[str, Any]] = []
        self._public_pose_cache: dict[str, dict[str, Any]] = {}
        self._active_public_grasp_object_name: str | None = None
        self._official_task_protocol = False
        self._protocol_stopped = False
        self._protocol_stop_reason: str | None = None
        self._trial_deadline_time: float | None = None
        self._trial_deadline_seconds: float | None = None
        self._code_block_action_start: int | None = None
        self._code_block_action_limit: int | None = None
        self._code_block_index: int | None = None
        self._reset_serial = 0

        self._prepare_import_path()
        self._build_task()

    @property
    def task(self):
        return self._task

    @property
    def tactile_buffer(self) -> UniVTACTactileBuffer:
        return self._tactile_buffer

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        trial = int((options or {}).get("trial", 0) or 0)
        if self._force_task_mode:
            if self.force_task_seed is None:
                raise RuntimeError("force_task requires an explicit force_task_seed")
            if self._reset_serial:
                raise RuntimeError(
                    "force tasks require one CaP-X trial per process; rebuild the environment for another seed"
                )
            actual_seed = self.force_task_seed
        else:
            actual_seed = self.seed_base + int(seed if seed is not None else trial)
        if not self._record_frames:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        self._tactile_buffer.clear()
        self._last_recorded_tactile_step = None
        self._last_recorded_video_step = None
        self._video_record_failures = 0
        self._live_preview_write_failures = 0
        self._last_action_result = {}
        self._sim_step_count = 0
        self._start_time = time.time()
        self._debug_records.clear()
        self._tactile_gripper_trace.clear()
        self._primitive_trace.clear()
        self._tactile_working_memory_trace.clear()
        self._tactile_trial_memory_snapshot.clear()
        self._public_probe_sessions.clear()
        self._public_probe_records.clear()
        self._public_probe_serial = 0
        self._pre_move_tactile_timeline.clear()
        self._pre_move_tactile_last_error = None
        self._tension_response_visualization = _empty_tension_response_visualization()
        self._tension_response_video_frames.clear()
        self._tension_response_video_history.clear()
        # APIs reset immediately after the low-level reset. Clearing callbacks here
        # prevents a previous episode's marker tracker from observing reset motion.
        self._post_step_observers.clear()
        self._post_action_observers.clear()
        self._post_observer_errors.clear()
        self._opentac_tension_estimator_diagnostics.clear()
        self._perception_artifacts.clear()
        self._public_pose_cache.clear()
        self._active_public_grasp_object_name = None
        self._force_task_command_ee_pos = None
        self._force_task_command_ee_quat = None
        self._protocol_stopped = False
        self._protocol_stop_reason = None
        self.clear_trial_deadline()
        self.clear_code_block_action_budget()
        self._reset_serial += 1

        print(
            f"[capx-univtac] reset begin trial={trial} seed={actual_seed}",
            flush=True,
        )
        self._task.reset(seed=actual_seed, instructions=self._instructions())
        self._refresh_public_pose_cache()
        self._initial_lift_object_z = self._active_object_height("can")
        debug = self._append_debug_record("after_reset")
        print(
            "[capx-univtac] reset diagnostic "
            f"pregrasp_ok={debug.get('pregrasp_ok')} "
            f"prism_z={debug.get('prism_pose', {}).get('position', [None, None, None])[2]} "
            f"ee_to_prism={debug.get('ee_to_prism_distance')} "
            f"gripper_qpos={debug.get('gripper_qpos')}",
            flush=True,
        )
        print(
            f"[capx-univtac] task.reset end step={self.get_step_count()}",
            flush=True,
        )
        self.max_steps = int(getattr(self._task.cfg, "step_lim", self.max_steps))
        self._current_obs = self._lightweight_raw_observation()
        print("[capx-univtac] lightweight obs ready", flush=True)
        obs = self._public_observation(self._current_obs)
        info = {"task_prompt": self.get_task_instruction(), "seed": actual_seed}
        print("[capx-univtac] reset return", flush=True)
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs = self.get_observation()
        reward = self.compute_reward()
        completed = self.task_completed()
        truncated = self.get_action_count() >= self.max_steps
        return obs, reward, completed, truncated, {"task_completed": completed}

    def get_observation(self) -> dict[str, Any]:
        if bool(self._task_config.get("defer_auto_observation", False)):
            self._current_obs = self._lightweight_raw_observation()
            return self._public_observation(self._current_obs)
        return self.refresh_native_observation()

    def refresh_native_observation(
        self,
        *,
        include_camera: bool = False,
        include_tactile: bool = True,
        include_embodiment: bool = True,
        include_actor: bool | None = None,
        tactile_data_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Read UniVTAC native observations explicitly for APIs/video capture."""
        raw = self._read_native_observation(
            include_camera=include_camera,
            include_tactile=include_tactile,
            include_embodiment=include_embodiment,
            include_actor=self.expose_actor_pose if include_actor is None else include_actor,
            tactile_data_types=tactile_data_types,
        )
        self._current_obs = raw
        if include_tactile:
            self._record_tactile_frame(force=True)
        return self._public_observation(raw)

    def compute_reward(self) -> float:
        return 1.0 if self.task_completed() else 0.0

    def task_completed(self) -> bool:
        completed = bool(self._task.check_success())
        if completed and bool(getattr(self, "_force_task_mode", False)):
            finalize = getattr(self._task, "_finish_strap_episode", None)
            if callable(finalize):
                finalize()
        return completed

    def _active_object_height(self, object_name: str) -> float | None:
        actor = getattr(self._task, str(object_name), None)
        if actor is None:
            return None
        try:
            return float(actor.get_pose().p[2])
        except Exception:
            return None

    def take_action(self, action: np.ndarray | torch.Tensor | list[float], *, action_type: str) -> dict[str, Any]:
        if not self.protocol_action_allowed():
            return self._protocol_blocked_result()
        tensor = self._to_tensor(action)
        exec_success, eval_success = self._task.take_action(tensor, action_type=action_type)
        self._update_after_action()
        self._append_debug_record(f"after_take_action_{action_type}")
        result = {
            "ok": bool(exec_success),
            "step": self.get_step_count(),
            "action_count": self.get_action_count(),
            "message": "action executed" if exec_success else "UniVTAC action execution failed",
        }
        self._last_action_result = result
        self._notify_post_action_observers(action_type, result)
        return result

    def register_post_step_observer(self, name: str, observer: Callable[[], None]) -> None:
        """Register an adapter-internal observer invoked after each task step."""
        self._post_step_observers[str(name)] = observer

    def unregister_post_step_observer(self, name: str) -> None:
        self._post_step_observers.pop(str(name), None)

    def register_post_action_observer(
        self, name: str, observer: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """Register an adapter-internal observer invoked after each public action."""
        self._post_action_observers[str(name)] = observer

    def unregister_post_action_observer(self, name: str) -> None:
        self._post_action_observers.pop(str(name), None)

    def set_opentac_tension_estimator_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        """Store public OpenTac estimator diagnostics for the trial artifact."""
        self._opentac_tension_estimator_diagnostics = _jsonable(dict(diagnostics))

    def _notify_post_step_observers(self) -> None:
        self._notify_observers(self._post_step_observers, "post_step")

    def _notify_post_action_observers(
        self, action_type: str, result: dict[str, Any]
    ) -> None:
        for name, observer in tuple(self._post_action_observers.items()):
            try:
                observer(str(action_type), _jsonable(dict(result)))
            except Exception as exc:
                self._record_observer_error(name, "post_action", exc)

    def _notify_observers(
        self, observers: dict[str, Callable[[], None]], phase: str
    ) -> None:
        for name, observer in tuple(observers.items()):
            try:
                observer()
            except Exception as exc:
                self._record_observer_error(name, phase, exc)

    def _record_observer_error(self, name: str, phase: str, exc: Exception) -> None:
        key = (str(name), repr(exc))
        if key in self._post_observer_errors:
            return
        self._post_observer_errors.add(key)
        print(
            f"[capx-univtac] internal observer error name={name} phase={phase}: {exc!r}",
            flush=True,
        )

    def move_force_task_vertical_delta(self, *, dz: float) -> dict[str, Any]:
        """Execute a public vertical delta through force-task absolute qpos.

        ViTaForge's final-acceptance force tasks intentionally reject generic
        Cartesian actions.  The adapter uses its native differential-IK helper
        to form a single absolute qpos target while keeping the LLM-visible
        operation restricted to a bounded public Z delta.
        """
        if not self._force_task_mode:
            return {
                "ok": False,
                "reason": "force_task_qpos_bridge_unavailable",
            }
        if not np.isfinite(dz):
            raise ValueError("force-task vertical delta must be finite")
        robot_manager = getattr(self._task, "_robot_manager", None)
        if robot_manager is None:
            return {"ok": False, "reason": "force_task_ik_bridge_unavailable"}
        try:
            qpos, target_pos, motion_path = self._force_task_accumulated_qpos_target(
                robot_manager, float(dz)
            )
            if qpos.numel() != 8 or not bool(torch.isfinite(qpos).all()):
                raise RuntimeError("native IK returned an invalid qpos target")
        except Exception as exc:
            return {
                "ok": False,
                "reason": "force_task_ik_target_failed",
                "message": str(exc),
            }
        result = self.take_action(qpos, action_type="qpos")
        return {
            **result,
            "operation": "opentac_tension_move_delta",
            "motion_path": motion_path,
            "requested_delta_z_m": float(dz),
            "command_target_ee_z_m": float(target_pos.reshape(-1, 3)[0, 2].item()),
        }

    def _force_task_accumulated_qpos_target(
        self, robot_manager: Any, dz: float
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Build the next force-task qpos target using an expert-style servo.

        ViTaForge's expert integrates its desired end-effector target and then
        solves IK from the *current* physical pose.  Reconstructing each
        ``move_delta`` from the physical pose loses target motion whenever the
        compliant strap lags the arm.  This adapter-only bridge keeps a bounded
        command lead while preserving the public local-Z action interface.
        """
        get_pose = getattr(robot_manager, "get_ee_pose_tensor", None)
        ik_controller = getattr(robot_manager, "_ik_controller", None)
        setup_ik = getattr(robot_manager, "_setup_ik_controller", None)
        if ik_controller is None and callable(setup_ik):
            setup_ik()
            ik_controller = getattr(robot_manager, "_ik_controller", None)
        arm_ids = getattr(robot_manager, "_arm_ids", None)
        jacobian = getattr(robot_manager, "jacobian_b", None)
        robot = getattr(robot_manager, "robot", None)
        if not (
            callable(get_pose)
            and ik_controller is not None
            and arm_ids is not None
            and jacobian is not None
            and robot is not None
        ):
            return self._force_task_legacy_delta_qpos_target(robot_manager, dz)

        current_pos, current_quat = get_pose()
        current_pos = current_pos.detach().to(device=self._task.device, dtype=torch.float32)
        current_quat = current_quat.detach().to(device=self._task.device, dtype=torch.float32)
        if self._force_task_command_ee_pos is None:
            commanded_pos = current_pos.clone()
            commanded_quat = current_quat.clone()
        else:
            commanded_pos = self._force_task_command_ee_pos.to(
                device=self._task.device, dtype=torch.float32
            ).clone()
            commanded_quat = self._force_task_command_ee_quat.to(
                device=self._task.device, dtype=torch.float32
            ).clone()

        commanded_pos[:, 2] += float(dz)
        lead = self._force_task_target_lead_m()
        commanded_pos[:, 2] = torch.minimum(
            commanded_pos[:, 2], current_pos[:, 2] + lead
        )
        ik_controller.set_command(torch.cat([commanded_pos, commanded_quat], dim=-1))
        joint_pos = robot.data.joint_pos[:, arm_ids]
        joint_pos_des = ik_controller.compute(
            current_pos,
            current_quat,
            jacobian[:, :, arm_ids],
            joint_pos,
        )
        limits = robot.data.soft_joint_pos_limits[:, arm_ids]
        joint_pos_des = torch.clamp(joint_pos_des, limits[..., 0], limits[..., 1])
        gripper_qpos = robot.data.joint_pos[:, robot_manager._gripper_ids][0, 0]
        qpos = torch.cat([joint_pos_des.reshape(-1)[:7], gripper_qpos.reshape(1)])
        self._force_task_command_ee_pos = commanded_pos.detach().clone()
        self._force_task_command_ee_quat = commanded_quat.detach().clone()
        return qpos, commanded_pos, "force_task_accumulated_ik_to_absolute_qpos"

    def _force_task_legacy_delta_qpos_target(
        self, robot_manager: Any, dz: float
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Keep the old bridge for lightweight tests and older task roots."""
        compute_target = getattr(robot_manager, "compute_delta_ee_rotvec_qpos_target", None)
        if not callable(compute_target):
            raise RuntimeError("force_task_ik_bridge_unavailable")
        action = torch.tensor(
            [0.0, 0.0, float(dz), 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self._task.device,
        )
        arm_qpos, gripper_qpos, target_pos, _target_quat = compute_target(action)
        qpos = torch.cat([arm_qpos.reshape(-1)[:7], gripper_qpos.reshape(-1)[:1]])
        return qpos, target_pos, "force_task_legacy_delta_ik_to_absolute_qpos"

    def _force_task_target_lead_m(self) -> float:
        config = self.api_configs.get("franka_control_api", {})
        configured = config.get("force_task_target_lead_m", 0.001) if isinstance(config, dict) else 0.001
        lead = float(configured)
        if not np.isfinite(lead) or lead <= 0.0:
            raise ValueError("force_task_target_lead_m must be a positive finite distance")
        return lead

    def _task_native_safe_placement_enabled(self) -> bool:
        cfg = self.api_configs.get("franka_control_api", {})
        return bool(cfg.get("task_native_safe_placement", False)) if isinstance(cfg, dict) else False

    def _place_actor_with_safe_transport(
        self,
        actor: Any,
        target_pose: Any,
        *,
        target_name: str,
        time_dilation_factor: float | None,
    ) -> bool:
        """Place through clearance, horizontal transport, then vertical descent."""
        place_pose = self._task.atom.get_place_pose(actor, target_pose=target_pose, pre_dis=0.0)
        if place_pose is None:
            return False

        robot = self._task._robot_manager
        current_gripper = robot.get_gripper_center_pose()
        safe_z = max(
            float(current_gripper.p[2]),
            float(getattr(self._task, "safe_gripper_z", current_gripper.p[2])),
        )
        if safe_z - float(current_gripper.p[2]) >= 0.005:
            clearance_gripper = type(current_gripper)(
                [current_gripper.p[0], current_gripper.p[1], safe_z],
                current_gripper.q,
            )
            clearance_ee = robot.gripper_center_to_ee(clearance_gripper)
            if not self._task.move(
                self._task.atom.move_to_pose(clearance_ee),
                tag=f"capx_place_{target_name}_clearance",
                time_dilation_factor=time_dilation_factor,
            ):
                return False
            self._task.delay(8, is_save=True, force=True)

        place_gripper = robot.ee_to_gripper_center(place_pose)
        current_gripper = robot.get_gripper_center_pose()
        hover_gripper = type(current_gripper)(
            [place_gripper.p[0], place_gripper.p[1], max(float(current_gripper.p[2]), safe_z)],
            place_gripper.q,
        )
        hover_ee = robot.gripper_center_to_ee(hover_gripper)
        if not self._task.move(
            self._task.atom.move_to_pose(hover_ee),
            tag=f"capx_place_{target_name}_horizontal",
            time_dilation_factor=time_dilation_factor,
        ):
            return False
        self._task.delay(8, is_save=True, force=True)
        if not self._task.move(
            self._task.atom.move_to_pose(place_pose),
            tag=f"capx_place_{target_name}_descend",
            time_dilation_factor=time_dilation_factor,
        ):
            return False
        self._task.delay(8, is_save=True, force=True)
        return True

    def get_rgbd_frame(self, camera_name: str = "head"):
        """Return one calibrated RGB-D frame without actor or task metadata."""
        from capx.integrations.univtac.rgbd_perception import RgbdFrame

        raw = self._read_native_observation(
            include_camera=True,
            include_tactile=False,
            include_embodiment=False,
            include_actor=False,
        )
        camera_obs = raw.get("observation", {}).get(str(camera_name), {})
        if not isinstance(camera_obs, dict):
            raise RuntimeError(f"UniVTAC camera {camera_name!r} is unavailable")
        if "rgb" not in camera_obs or "depth" not in camera_obs:
            raise RuntimeError(
                f"UniVTAC camera {camera_name!r} must provide both rgb and depth"
            )

        cameras = getattr(getattr(self._task, "_camera_manager", None), "cameras", {})
        camera = cameras.get(str(camera_name)) if isinstance(cameras, dict) else None
        data = getattr(camera, "data", None)
        if data is None:
            raise RuntimeError(f"calibration for UniVTAC camera {camera_name!r} is unavailable")

        rgb = _to_numpy(camera_obs["rgb"])
        depth = _to_numpy(camera_obs["depth"])
        if np.asarray(rgb).ndim == 4:
            rgb = np.asarray(rgb)[0]
        depth = np.asarray(depth)
        if depth.ndim == 4:
            depth = depth[0]
        depth = np.squeeze(depth)
        frame = RgbdFrame(
            rgb=np.asarray(rgb),
            depth=depth,
            intrinsics=_to_numpy(data.intrinsic_matrices[0]),
            camera_position=_to_numpy(data.pos_w[0]),
            camera_quaternion_wxyz=_to_numpy(data.quat_w_ros[0]),
            camera_name=str(camera_name),
        ).validated()
        self._current_obs = raw
        return frame

    def move_to_tool_pose_native(
        self,
        position: np.ndarray | list[float],
        quaternion_wxyz: np.ndarray | list[float],
    ) -> dict[str, Any]:
        """Plan to a public gripper-center pose without consulting an actor."""
        if not self.protocol_action_allowed():
            return self._protocol_blocked_result()
        from envs.utils.transforms import Pose

        tool_position = np.asarray(position, dtype=np.float32).reshape(3).copy()
        tool_quaternion = np.asarray(quaternion_wxyz, dtype=np.float32).reshape(4)
        tool_pose = Pose(tool_position, tool_quaternion)
        ee_pose = self._task._robot_manager.gripper_center_to_ee(tool_pose)
        api_config = self.api_configs.get("franka_control_api", {})
        native_min_ee_z = float(api_config.get("native_min_ee_z", 0.0))
        ee_z = float(np.asarray(ee_pose.p, dtype=np.float32).reshape(3)[2])
        if ee_z < native_min_ee_z:
            tool_position[2] += native_min_ee_z - ee_z
            tool_pose = Pose(tool_position, tool_quaternion)
            ee_pose = self._task._robot_manager.gripper_center_to_ee(tool_pose)
            print(
                "[capx-univtac] native_ee_z_clamped "
                f"requested_ee_z={ee_z:.4f} safe_ee_z={native_min_ee_z:.4f} "
                f"adjusted_tool_z={tool_position[2]:.4f}",
                flush=True,
            )
        gripper_qpos = float(self._task._robot_manager.get_gripper_qpos())
        action = np.concatenate(
            [
                np.asarray(ee_pose.p, dtype=np.float32).reshape(3),
                np.asarray(ee_pose.q, dtype=np.float32).reshape(4),
                np.array([gripper_qpos], dtype=np.float32),
            ]
        )
        return self.take_action(action, action_type="ee")

    def protocol_action_allowed(self) -> bool:
        """Return whether another official-protocol physical action may run."""
        self._raise_if_hard_stopped("protocol_action_allowed")
        if not self._official_task_protocol:
            return True
        if self._protocol_stopped:
            return False
        if self.get_action_count() >= self.max_steps:
            self._stop_protocol("action_budget")
            return False
        return True

    def begin_high_level_action(self) -> bool:
        """Count a gripper-only action in the official 300-action budget."""
        if not self.protocol_action_allowed():
            return False
        if self._official_task_protocol:
            self._task.take_action_cnt += 1
            logger = getattr(self._task, "logger", None)
            if logger is not None:
                logger.info(
                    f"step: {self.get_action_count()} / {self.max_steps} (CaP gripper action)"
                )
        return True

    def finalize_high_level_action(self) -> dict[str, Any]:
        """Apply native success/early-stop rules after one CaP physical action."""
        if self._official_task_protocol:
            if self.get_action_count() >= self.max_steps:
                self._stop_protocol("action_budget")
            elif not self._protocol_stopped:
                try:
                    if bool(self._task.check_success()):
                        self._task.eval_success = True
                        self._stop_protocol("native_success")
                    elif bool(self._task.check_early_stop()):
                        self._stop_protocol("early_stop")
                except Exception as exc:
                    self._stop_protocol(f"protocol_check_error:{type(exc).__name__}")
        self.refresh_live_preview()
        return self.get_protocol_status()

    def refresh_live_preview(self) -> None:
        """Refresh the local preview after a high-level action attempt.

        Native motion planning can reject a pose before any simulator substep
        occurs.  Recording only from the substep hook leaves ``latest.jpg``
        stale in exactly that failure mode, which is misleading during manual
        debugging.
        """
        if not getattr(self, "_record_frames", False) or not getattr(
            self, "live_preview_enabled", False
        ):
            return
        try:
            self._task._update_render()
            obs = self._read_native_observation(
                include_camera=True,
                include_tactile=True,
                include_embodiment=False,
                include_actor=False,
                tactile_data_types=["rgb", "rgb_marker", "depth", "marker"],
            )
            self._record_frame(obs, force=True)
        except Exception as exc:
            self._live_preview_write_failures += 1
            if self._live_preview_write_failures <= 3:
                print(
                    f"WARNING: failed to refresh UniVTAC live preview: {exc!r}",
                    flush=True,
                )

    def get_protocol_status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self._official_task_protocol),
            "stopped": bool(self._protocol_stopped),
            "reason": self._protocol_stop_reason,
            "action_count": self.get_action_count(),
            "max_steps": self.max_steps,
        }

    def get_reset_serial(self) -> int:
        return int(self._reset_serial)

    def append_perception_artifact(self, record: dict[str, Any]) -> None:
        """Keep non-privileged perception inputs and outputs for trial audit."""
        self._perception_artifacts.append(dict(record))

    def set_trial_deadline(self, timeout_seconds: float) -> None:
        """Set a soft wall-clock deadline checked before physical actions."""
        timeout = max(1.0, float(timeout_seconds))
        self._trial_deadline_seconds = timeout
        self._trial_deadline_time = time.monotonic() + timeout

    def clear_trial_deadline(self) -> None:
        """Clear the current soft wall-clock deadline."""
        self._trial_deadline_time = None
        self._trial_deadline_seconds = None

    def set_code_block_action_budget(
        self,
        max_actions: int | None,
        *,
        block_index: int | None = None,
    ) -> None:
        if max_actions is None or int(max_actions) <= 0:
            self.clear_code_block_action_budget()
            return
        self._code_block_action_start = self.get_action_count()
        self._code_block_action_limit = int(max_actions)
        self._code_block_index = int(block_index) if block_index is not None else None

    def clear_code_block_action_budget(self) -> None:
        self._code_block_action_start = None
        self._code_block_action_limit = None
        self._code_block_index = None

    def _raise_if_hard_stopped(self, where: str) -> None:
        deadline = getattr(self, "_trial_deadline_time", None)
        if deadline is not None and time.monotonic() >= float(deadline):
            if not self._protocol_stopped:
                self._stop_protocol("trial_timeout")
            try:
                self._task.plan_success = False
            except Exception:
                pass
            timeout = float(getattr(self, "_trial_deadline_seconds", None) or 0.0)
            raise HardStopTrial(
                "trial_timeout",
                f"trial deadline reached before {where} after {timeout:.1f}s",
                details={
                    "where": where,
                    "action_count": self.get_action_count(),
                    "max_steps": self.max_steps,
                },
            )

        start = getattr(self, "_code_block_action_start", None)
        limit = getattr(self, "_code_block_action_limit", None)
        if start is None or limit is None:
            return
        used = self.get_action_count() - int(start)
        if used < int(limit):
            return
        reason = "block_action_budget"
        print(
            "CAPX_FAILURE object=current phase=code_block "
            f"reason={reason} action=regenerate stable=false contact=false "
            f"force=0.0000 slip=0.0000 used_actions={used} "
            f"limit={int(limit)} block={self._code_block_index}",
            flush=True,
        )
        raise RecoverableTaskFailure(
            reason,
            (
                f"code block {self._code_block_index} used {used} physical "
                f"action(s), reaching the configured limit {int(limit)} "
                f"before {where}"
            ),
            details={
                "where": where,
                "block_index": self._code_block_index,
                "used_actions": int(used),
                "max_code_block_actions": int(limit),
                "action_count": self.get_action_count(),
                "max_steps": self.max_steps,
            },
        )

    def _stop_protocol(self, reason: str) -> None:
        if self._protocol_stopped:
            return
        self._protocol_stopped = True
        self._protocol_stop_reason = str(reason)
        print(
            "[capx-univtac] official protocol stopped "
            f"reason={self._protocol_stop_reason} action_count={self.get_action_count()}",
            flush=True,
        )

    def _protocol_blocked_result(self) -> dict[str, Any]:
        result = {
            "ok": False,
            "step": self.get_step_count(),
            "action_count": self.get_action_count(),
            "message": f"official protocol stopped: {self._protocol_stop_reason}",
        }
        self._last_action_result = result
        return result

    def place_grasped_actor(
        self,
        *,
        target_name: str,
        target_position: np.ndarray | list[float],
        target_quaternion_wxyz: np.ndarray | list[float] | None = None,
        pre_dis: float = 0.0,
        dis: float = 0.0,
        time_dilation_factor: float | None = 0.5,
    ) -> dict[str, Any]:
        """Place the currently grasped actor with UniVTAC's native placement primitive.

        This method is for the CaP-X UniVTAC compatibility adapter. It does not
        expose private class labels to generated code; the LLM has already chosen
        a public target landmark, and this function only translates that
        placement intent into UniVTAC's task-native motion primitive.
        """
        try:
            from envs.utils.transforms import Pose
        except Exception as exc:
            result = {
                "ok": False,
                "step": self.get_step_count(),
                "action_count": self.get_action_count(),
                "message": f"could not import UniVTAC Pose: {exc!r}",
            }
            self._last_action_result = result
            return result

        actor = None
        active_name = self._active_public_grasp_object_name
        if active_name:
            try:
                actor = self._public_grasp_actor(active_name)
            except Exception:
                actor = None
        if actor is None:
            actor = getattr(self._task, "prism", None)
        if actor is None:
            result = {
                "ok": False,
                "step": self.get_step_count(),
                "action_count": self.get_action_count(),
                "message": "no grasped actor is available for native placement",
            }
            self._last_action_result = result
            return result

        pos = np.asarray(target_position, dtype=np.float32).reshape(3)
        # The target pose is the object pose, not the end-effector pose. Keep
        # public pad targets upright even when the EE orientation is preserved
        # for the LLM-facing get_object_pose contract.
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        target_pose = Pose(pos.tolist(), quat.tolist())

        stages: list[str] = []
        try:
            if self._task_native_safe_placement_enabled():
                exec_success = self._place_actor_with_safe_transport(
                    actor,
                    target_pose,
                    target_name=str(target_name),
                    time_dilation_factor=time_dilation_factor,
                )
                stages = ["clearance", "horizontal", "descend"]
            else:
                actions = self._task.atom.place_actor(
                    actor,
                    target_pose=target_pose,
                    pre_dis=float(pre_dis),
                    dis=float(dis),
                    is_open=False,
                )
                if not actions:
                    exec_success = False
                else:
                    exec_success = bool(
                        self._task.move(
                            actions,
                            tag=f"capx_place_{target_name}",
                            time_dilation_factor=time_dilation_factor,
                        )
                    )
                stages = ["direct"]
        except Exception as exc:
            exec_success = False
            message = f"native placement failed: {exc!r}"
        else:
            message = "native placement executed" if exec_success else "native placement planning failed"

        self._update_after_action()
        self._append_debug_record(f"after_place_{target_name}")
        result = {
            "ok": bool(exec_success),
            "step": self.get_step_count(),
            "action_count": self.get_action_count(),
            "message": message,
            "placement_stages": stages,
        }
        self._last_action_result = result
        return result

    def approach_grasped_actor(
        self,
        *,
        object_name: str = "prism",
        position_offset: np.ndarray | list[float] | None = None,
        pre_dis: float = 0.04,
        dis: float = 0.0,
        grasp_height: float = 0.04,
        time_dilation_factor: float | None = None,
    ) -> dict[str, Any]:
        """Move the gripper to the selected public grasp target.

        This is intentionally an internal adapter helper. Generated code still
        sees only ``sample_grasp_pose`` and ``goto_pose``; this method translates
        that CaP-style request into UniVTAC's native motion planner.
        """
        task_approach = getattr(self._task, "approach_grasped_actor", None)
        if callable(task_approach):
            try:
                result = task_approach(
                    object_name=object_name,
                    position_offset=position_offset,
                    pre_dis=pre_dis,
                    dis=dis,
                    grasp_height=grasp_height,
                    time_dilation_factor=time_dilation_factor,
                )
                if not isinstance(result, dict):
                    result = {
                        "ok": bool(result),
                        "message": "task adapter grasp hook executed",
                    }
            except Exception as exc:
                result = {
                    "ok": False,
                    "message": f"task adapter grasp hook failed: {exc!r}",
                }
            self._update_after_action()
            self._append_debug_record(f"after_approach_grasp_{object_name}")
            result = {
                "step": self.get_step_count(),
                "action_count": self.get_action_count(),
                **result,
            }
            self._last_action_result = result
            if bool(result.get("ok", False)):
                self._active_public_grasp_object_name = str(object_name)
            return result

        actor = self._public_grasp_actor(object_name)
        if actor is None:
            result = {
                "ok": False,
                "step": self.get_step_count(),
                "action_count": self.get_action_count(),
                "message": f"public grasp object '{object_name}' is not available",
            }
            self._last_action_result = result
            return result

        try:
            contact_pose = self._make_public_grasp_pose(
                object_name,
                actor,
                grasp_height=float(grasp_height),
            )
            offset = np.asarray(
                [0.0, 0.0, 0.0] if position_offset is None else position_offset,
                dtype=np.float32,
            ).reshape(3)
            if not np.all(np.isfinite(offset)):
                raise ValueError("native grasp position offset must be finite")
            if float(np.linalg.norm(offset)) > 1e-8:
                contact_pose = contact_pose.add_bias(offset, coord="world")
            contact_id = actor.register_point(contact_pose, type="contact")
            actions = self._task.atom.grasp_actor(
                actor,
                contact_point_id=contact_id,
                pre_dis=float(pre_dis),
                dis=float(dis),
                is_close=False,
            )
            exec_success = bool(
                self._task.move(
                    actions,
                    tag=f"capx_approach_grasp_{object_name}",
                    time_dilation_factor=time_dilation_factor,
                )
            )
        except Exception as exc:
            exec_success = False
            message = f"native grasp approach failed: {exc!r}"
        else:
            message = "native grasp approach executed" if exec_success else "native grasp approach planning failed"

        self._update_after_action()
        self._append_debug_record(f"after_approach_grasp_{object_name}")
        result = {
            "ok": bool(exec_success),
            "step": self.get_step_count(),
            "action_count": self.get_action_count(),
            "message": message,
        }
        self._last_action_result = result
        if bool(result.get("ok", False)):
            self._active_public_grasp_object_name = str(object_name)
        return result

    def move_active_public_grasp_by_displacement(
        self,
        *,
        dz: float,
        time_dilation_factor: float = 0.5,
    ) -> dict[str, Any]:
        """Run a vertical public-probe motion through the task's native atom.

        This is an internal bridge for the tactile-memory ``move_delta`` API.
        It intentionally exposes no new LLM function: callers still issue the
        ordinary bounded ``move_delta(dz=...)`` request.  Once a public
        side-grasp approach and tactile close have succeeded, the benchmark's
        expert uses this same ``move_by_displacement`` route for lift/lower.
        """
        if not self.protocol_action_allowed():
            return self._protocol_blocked_result()
        displacement = float(dz)
        if not np.isfinite(displacement):
            return {
                "ok": False,
                "step": self.get_step_count(),
                "action_count": self.get_action_count(),
                "message": "public probe displacement must be finite",
            }
        object_name = self._active_public_grasp_object_name
        if not object_name:
            return {
                "ok": False,
                "step": self.get_step_count(),
                "action_count": self.get_action_count(),
                "message": "no active public grasp object for native probe motion",
            }

        try:
            resolve_name = getattr(self._task, "_resolve_public_object_name", None)
            public_name = resolve_name(object_name) if callable(resolve_name) else object_name
            if not public_name:
                raise KeyError(f"unknown public object {object_name!r}")
            actions = self._task.atom.move_by_displacement(
                z=displacement,
                xyz_coord="world",
            )
            role_move = getattr(self._task, "_role_move", None)
            if callable(role_move):
                ok = bool(
                    role_move(
                        public_name,
                        actions,
                        tag=f"{public_name}_capx_probe_vertical",
                        time_dilation_factor=float(time_dilation_factor),
                        is_save=True,
                        delay=False,
                    )
                )
            else:
                ok = bool(
                    self._task.move(
                        actions,
                        tag=f"{public_name}_capx_probe_vertical",
                        time_dilation_factor=float(time_dilation_factor),
                        is_save=True,
                        delay=False,
                    )
                )
        except Exception as exc:
            ok = False
            message = f"native public probe displacement failed: {exc!r}"
        else:
            message = (
                "native public probe displacement executed"
                if ok
                else "native public probe displacement planning failed"
            )

        self._update_after_action()
        self._append_debug_record(f"after_public_probe_vertical_{object_name}")
        result = {
            "ok": bool(ok),
            "step": self.get_step_count(),
            "action_count": self.get_action_count(),
            "message": message,
            "object_name": str(object_name),
            "dz": displacement,
            "motion_path": "task_atom_move_by_displacement",
        }
        self._last_action_result = result
        return result

    def get_public_grasp_pose(
        self,
        object_name: str,
        *,
        grasp_height: float = 0.04,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the public pose used by the matching native grasp approach."""
        task_sampler = getattr(self._task, "get_public_grasp_pose", None)
        if callable(task_sampler):
            pos, quat = task_sampler(object_name, grasp_height=float(grasp_height))
            return (
                np.asarray(pos, dtype=np.float32).reshape(3),
                np.asarray(quat, dtype=np.float32).reshape(4),
            )

        actor = self._public_grasp_actor(object_name)
        if actor is None:
            raise KeyError(f"public grasp object '{object_name}' is not available")
        pose = self._make_public_grasp_pose(
            object_name,
            actor,
            grasp_height=float(grasp_height),
        )
        return (
            np.asarray(pose.p, dtype=np.float32).reshape(3),
            np.asarray(pose.q, dtype=np.float32).reshape(4),
        )

    def move_gripper_native(
        self,
        *,
        qpos: float,
        opening: bool,
        lift_after_close: bool = False,
        lift_z: float = 0.05,
        settle_steps: int = 0,
    ) -> dict[str, Any]:
        """Move the UniVTAC gripper through its native gripper planner."""
        try:
            if opening:
                actions = self._task.atom.open_gripper(float(qpos))
                tag = "capx_open_gripper"
            else:
                actions = self._task.atom.close_gripper(float(qpos))
                tag = "capx_close_gripper"
            exec_success = bool(self._task.move(actions, tag=tag))
            if exec_success and lift_after_close:
                exec_success = bool(
                    self._task.move(
                        self._task.atom.move_by_displacement(z=float(lift_z)),
                        tag="capx_lift_after_grasp",
                    )
                )
            if exec_success and settle_steps > 0:
                self._task.delay(int(settle_steps), is_save=True, force=True)
        except Exception as exc:
            exec_success = False
            message = f"native gripper motion failed: {exc!r}"
        else:
            message = "native gripper motion executed" if exec_success else "native gripper motion planning failed"

        self._update_after_action()
        self._append_debug_record("after_open_gripper" if opening else "after_close_gripper")
        result = {
            "ok": bool(exec_success),
            "step": self.get_step_count(),
            "action_count": self.get_action_count(),
            "message": message,
        }
        self._last_action_result = result
        return result

    def get_gripper_calibration(self) -> dict[str, float]:
        """Return the public width conversion for the active robot gripper."""
        robot_manager = self._task._robot_manager
        max_qpos = float(getattr(robot_manager, "gripper_max_qpos", 0.039))
        if not np.isfinite(max_qpos) or max_qpos <= 0.0:
            raise RuntimeError(f"invalid UniVTAC gripper_max_qpos: {max_qpos!r}")
        current_qpos = float(robot_manager.get_gripper_qpos())
        current_qpos = float(np.clip(current_qpos, 0.0, max_qpos))
        return {
            "gripper_max_qpos": max_qpos,
            "current_qpos": current_qpos,
            "current_width": current_qpos / max_qpos,
        }

    def get_native_tactile_calibration(self) -> dict[str, float]:
        """Return robot-native depth calibration for tactile summarization."""
        robot_cfg = self._task.cfg.robot
        far_plane = float(getattr(robot_cfg, "tactile_far_plane", 30.0))
        target_depth = float(
            getattr(
                self._task.cfg,
                "adaptive_grasp_depth_threshold",
                getattr(robot_cfg, "adaptive_grasp_depth_threshold", far_plane - 2.0),
            )
        )
        force_full_scale = far_plane - target_depth
        if not np.isfinite(force_full_scale) or force_full_scale <= 0.0:
            force_full_scale = 2.0
        adaptive_cfg = (
            self.api_configs.get("franka_control_api", {}).get("adaptive_gripper", {})
        )
        contact_margin = float(adaptive_cfg.get("depth_contact_margin_mm", 0.5))
        return {
            "depth_far_plane_mm": far_plane,
            "force_full_scale_mm": force_full_scale,
            "depth_contact_margin_mm": max(0.0, contact_margin),
        }

    def command_gripper_width_step(
        self,
        width: float,
        settle_steps: int = 1,
    ) -> dict[str, Any]:
        """Execute one normalized gripper servo command without a CaP action.

        This hook performs no tactile decision making. It only converts width
        to qpos, sends a non-teleporting target, advances simulation, and
        refreshes the native tactile buffer.
        """
        calibration = self.get_gripper_calibration()
        max_qpos = calibration["gripper_max_qpos"]
        current_qpos = calibration["current_qpos"]
        target_width = float(np.clip(width, 0.0, 1.0))
        target_qpos = target_width * max_qpos
        robot_manager = self._task._robot_manager
        device = getattr(robot_manager, "device", getattr(self._task, "device", "cpu"))
        position = torch.tensor(
            [target_qpos, target_qpos],
            dtype=torch.float32,
            device=device,
        )
        sim_cfg = getattr(getattr(self._task, "cfg", None), "sim", None)
        sim_dt = float(getattr(sim_cfg, "dt", 1.0 / 60.0))
        # Match UniVTAC's native ``plan_gripper`` semantics.  In particular,
        # its articulation path clamps finger velocity and immediately writes
        # the joint position to PhysX.  Leaving this as a target-only command
        # lets the velocity drive run far faster than the official expert and
        # can desynchronise the visible finger meshes from the gripper body.
        velocity = torch.clamp(
            (position - current_qpos) / max(sim_dt, 1e-8),
            min=-0.0001,
            max=0.0001,
        )
        action_count_before = self.get_action_count()
        robot_manager.set_gripper(position, velocity, force=True)
        for _ in range(max(1, int(settle_steps))):
            self._task._step(is_save=True)

        self.refresh_native_observation(
            include_camera=False,
            include_tactile=True,
            include_embodiment=False,
            include_actor=False,
            tactile_data_types=["depth", "marker", "pose"],
        )
        updated = self.get_gripper_calibration()
        result = {
            "ok": True,
            "step": self.get_step_count(),
            "action_count": self.get_action_count(),
            "width": updated["current_width"],
            "qpos": updated["current_qpos"],
            "target_width": target_width,
            "target_qpos": target_qpos,
            "message": "single gripper servo step executed",
        }
        if result["action_count"] != action_count_before:
            raise RuntimeError("gripper micro-step unexpectedly changed the CaP action count")
        self._last_action_result = result
        return result

    def append_tactile_gripper_trace(self, trace: list[dict[str, Any]]) -> None:
        """Store controller-only close/open trace records for trial audit."""
        for item in trace:
            record = dict(item)
            if record.get("operation") not in {"close", "open"}:
                raise ValueError("tactile gripper trace may only contain close/open operations")
            self._tactile_gripper_trace.append(_jsonable(record))

    def reset_primitive_trace(self) -> None:
        """Clear public tactile primitive trace for a fresh trial."""
        if not hasattr(self, "_primitive_trace"):
            self._primitive_trace = []
        self._primitive_trace.clear()

    def append_primitive_trace(self, record: dict[str, Any]) -> None:
        """Store public touch primitive inputs, outputs, and tactile quality."""
        if not hasattr(self, "_primitive_trace"):
            self._primitive_trace = []
        self._primitive_trace.append(_jsonable(dict(record)))

    def append_tactile_working_memory_trace(self, record: dict[str, Any]) -> None:
        """Store public trial-memory capture/write/clear events for audit."""
        if not hasattr(self, "_tactile_working_memory_trace"):
            self._tactile_working_memory_trace = []
        self._tactile_working_memory_trace.append(_jsonable(dict(record)))

    def get_tactile_working_memory_trace(self) -> list[dict[str, Any]]:
        """Return trial-local tactile memory events without private task fields."""
        return list(getattr(self, "_tactile_working_memory_trace", []))

    def set_tactile_trial_memory_snapshot(self, memory: dict[str, Any]) -> None:
        """Store a public, agent-authored memory snapshot for trial artifacts."""
        self._tactile_trial_memory_snapshot = _jsonable(dict(memory))

    def get_tactile_trial_memory_snapshot(self) -> dict[str, Any]:
        """Return the current public trial-memory snapshot for audit only."""
        return _jsonable(getattr(self, "_tactile_trial_memory_snapshot", {}))

    def get_public_probe_spec(self) -> dict[str, Any]:
        """Return the task-declared, label-free tactile probe protocol."""
        getter = getattr(self._task, "get_public_probe_spec", None)
        if not callable(getter):
            raise RuntimeError(
                f"task {self.task_name!r} does not expose a public tactile probe specification"
            )
        spec = getter()
        if not isinstance(spec, dict) or spec.get("schema_version") != "public_probe_spec.v2":
            raise RuntimeError("task returned an invalid public_probe_spec.v2 record")
        return _jsonable(spec)

    def begin_public_probe_capture(self, object_name: str) -> dict[str, Any]:
        """Open a recorder while CaP-X executes one public standard probe.

        This method performs no physical motion and does not consult task
        metadata.  The caller must mark the preload, lift-motion, and hold
        segments around its own FrankaControlApi actions.
        """
        key = self._normalize_public_probe_object_name(object_name)
        self._public_probe_serial += 1
        capture_id = f"probe_{self._public_probe_serial:03d}_{key}"
        self._public_probe_sessions[capture_id] = {
            "schema_version": "capx_public_probe_capture.v1",
            "capture_id": capture_id,
            "object_name": key,
            "protocol": self.get_public_probe_spec(),
            "segments": {"preload": [], "lift_motion": [], "hold": []},
            "aggregates": {},
            "active_segment": None,
            "start_step": self.get_step_count(),
        }
        return {
            "schema_version": "capx_public_probe_capture.v1",
            "capture_id": capture_id,
            "object_name": key,
            "protocol": self.get_public_probe_spec(),
        }

    def begin_public_probe_segment(self, capture_id: str, segment: str) -> dict[str, Any]:
        """Begin recording one public probe segment during caller-owned motion."""
        session = self._public_probe_session(capture_id)
        normalized = self._normalize_public_probe_segment(segment)
        if session["active_segment"] is not None:
            raise RuntimeError(
                f"probe capture {capture_id!r} already records {session['active_segment']!r}"
            )
        if session["segments"][normalized]:
            raise RuntimeError(f"probe segment {normalized!r} was already recorded")
        session["active_segment"] = normalized
        return {
            "ok": True,
            "capture_id": str(capture_id),
            "segment": normalized,
            "start_step": self.get_step_count(),
        }

    def end_public_probe_segment(self, capture_id: str, segment: str) -> dict[str, Any]:
        """Stop one segment and aggregate it with the expert's v4 reducer."""
        session = self._public_probe_session(capture_id)
        normalized = self._normalize_public_probe_segment(segment)
        if session["active_segment"] != normalized:
            raise RuntimeError(
                f"probe capture {capture_id!r} is not recording {normalized!r}"
            )
        session["active_segment"] = None
        aggregator = getattr(self._task, "aggregate_public_probe_window", None)
        if not callable(aggregator):
            raise RuntimeError("task does not expose public tactile probe aggregation")
        frames = session["segments"][normalized]
        aggregate = aggregator(frames) if frames else _empty_public_probe_window()
        session["aggregates"][normalized] = _jsonable(aggregate)
        return {
            "ok": bool(frames),
            "capture_id": str(capture_id),
            "segment": normalized,
            "frame_count": len(frames),
            "aggregate": _jsonable(aggregate),
        }

    def finalize_public_probe_capture(
        self,
        capture_id: str,
        execution: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a canonical public ``tactile_probe.v4`` record.

        ``execution`` must report the caller's own approach, close, lift,
        lower, release, and clearance outcomes.  It changes probe quality but
        cannot introduce labels or select a candidate.
        """
        session = self._public_probe_session(capture_id)
        if session["active_segment"] is not None:
            raise RuntimeError(
                f"probe capture {capture_id!r} still records {session['active_segment']!r}"
            )
        missing = [name for name in ("preload", "lift_motion", "hold") if name not in session["aggregates"]]
        if missing:
            raise RuntimeError(f"probe capture {capture_id!r} is missing segments: {missing}")
        if not isinstance(execution, dict):
            raise TypeError("probe execution must be a dictionary")
        required = {
            "approach_ok",
            "close_ok",
            "bilateral_gate",
            "lift_ok",
            "lower_ok",
            "release_ok",
            "clearance_ok",
        }
        missing_execution = sorted(required.difference(execution))
        if missing_execution:
            raise ValueError(f"probe execution is missing fields: {missing_execution}")
        builder = getattr(self._task, "build_public_probe_record", None)
        if not callable(builder):
            raise RuntimeError("task does not expose public tactile probe construction")
        probe = builder(
            session["object_name"],
            session["aggregates"]["preload"],
            session["aggregates"]["lift_motion"],
            session["aggregates"]["hold"],
            **{key: bool(execution[key]) for key in required},
        )
        record = {
            "schema_version": "capx_public_probe_record.v1",
            "capture_id": str(capture_id),
            "object_name": session["object_name"],
            "protocol": session["protocol"],
            "probe": _jsonable(probe),
            "segments": _jsonable(session["aggregates"]),
            "frame_counts": {
                name: len(session["segments"][name])
                for name in ("preload", "lift_motion", "hold")
            },
            "execution": {key: bool(execution[key]) for key in required},
            "start_step": int(session["start_step"]),
            "end_step": self.get_step_count(),
        }
        self._public_probe_records[str(capture_id)] = record
        self._public_probe_sessions.pop(str(capture_id), None)
        return _jsonable(record)

    def get_public_probe_records(self) -> dict[str, Any]:
        """Return public probe artifacts for output writing, never for LLM context."""
        return _jsonable(getattr(self, "_public_probe_records", {}))

    def _record_active_public_probe_frame(self) -> None:
        active = [
            session
            for session in getattr(self, "_public_probe_sessions", {}).values()
            if session.get("active_segment") is not None
        ]
        if not active:
            return
        capture = getattr(self._task, "capture_public_probe_frame", None)
        if not callable(capture):
            raise RuntimeError("task does not expose public tactile probe frames")
        frame = _jsonable(capture())
        for session in active:
            session["segments"][session["active_segment"]].append(frame)

    @staticmethod
    def _normalize_public_probe_object_name(object_name: str) -> str:
        normalized = str(object_name).strip().lower()
        aliases = {"reference": "reference_object", "reference_object": "reference_object"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"reference_object", "candidate_left", "candidate_right"}:
            raise KeyError(f"unknown public probe object: {object_name!r}")
        return normalized

    @staticmethod
    def _normalize_public_probe_segment(segment: str) -> str:
        normalized = str(segment).strip().lower()
        if normalized not in {"preload", "lift_motion", "hold"}:
            raise ValueError("probe segment must be preload, lift_motion, or hold")
        return normalized

    def _public_probe_session(self, capture_id: str) -> dict[str, Any]:
        session = self._public_probe_sessions.get(str(capture_id))
        if session is None:
            raise KeyError(f"unknown active probe capture: {capture_id!r}")
        return session

    def wait_steps(self, n: int = 1) -> dict[str, Any]:
        steps = max(0, int(n))
        for _ in range(steps):
            self._task.delay(1, is_save=True, force=True)
            self._update_after_action()
        result = {
            "ok": True,
            "step": self.get_step_count(),
            "action_count": self.get_action_count(),
            "message": f"waited {steps} simulation step(s)",
        }
        self._last_action_result = result
        return result

    def get_task_instruction(self) -> str:
        instruction = getattr(self._task, "instruction", "")
        if instruction:
            return str(instruction)
        if self.task_name == "grasp_classify":
            return (
                "Classify the grasped prism using UniVTAC native tactile feedback, "
                "then place a rough prism on the orange pad and a plain prism on the green pad."
            )
        if self.task_name == "lift_can":
            api_configs = getattr(self, "api_configs", {})
            franka_cfg = api_configs.get("franka_control_api", {})
            adaptive_enabled = bool(
                franka_cfg.get("tactile_adaptive_gripper_enabled", True)
            )
            if not adaptive_enabled:
                return (
                    "Grasp and lift the cylindrical can using fixed gripper control, "
                    "then release it upright on the table."
                )
            return (
                "Grasp and lift the cylindrical can using UniVTAC native tactile feedback, "
                "then release it upright on the table."
            )
        if self.task_name == "tactile_memory_match":
            return (
                "Probe the reference cylinder with UniVTAC native tactile feedback, "
                "probe candidate_left and candidate_right, choose the candidate whose "
                "tactile signature best matches the reference, then place the selected "
                "candidate on match_slot. Do not use private labels, reward, success, "
                "density, friction, or task metadata."
            )
        return f"Solve the UniVTAC task: {self.task_name}."

    def get_step_count(self) -> int:
        return int(getattr(self._task, "step_count", 0))

    def get_action_count(self) -> int:
        return int(getattr(self._task, "take_action_cnt", 0))

    def get_robot_state(self) -> dict[str, Any]:
        obs = self._current_obs or self.get_observation()
        embodiment = obs.get("embodiment", {}) if isinstance(obs, dict) else {}
        state = _jsonable(embodiment)

        ee_pose = state.get("ee")
        if not _is_pose_like(ee_pose):
            try:
                ee_pose = self._task._robot_manager.get_ee_pose().tolist()
            except Exception:
                ee_pose = None
        if _is_pose_like(ee_pose):
            ee_pose = [float(x) for x in ee_pose[:7]]
            state["ee"] = ee_pose
            state["ee_pose"] = ee_pose
            state["ee_pos"] = ee_pose[:3]
            state["ee_quat"] = ee_pose[3:7]

        joint = state.get("joint")
        if not _is_sequence(joint):
            try:
                joint = self._task._robot_manager.get_qpos().squeeze(0).tolist()
            except Exception:
                joint = None
        if _is_sequence(joint):
            joint = [float(x) for x in joint]
            state["joint"] = joint
            state["qpos"] = joint
            if len(joint) >= 8:
                state["gripper_qpos"] = float(joint[7])
            else:
                try:
                    state["gripper_qpos"] = float(self._task._robot_manager.get_gripper_qpos())
                except Exception:
                    pass

        return state

    def get_actor_poses(self) -> dict[str, Any]:
        obs = self._current_obs or self.get_observation()
        actor = obs.get("actor", {}) if isinstance(obs, dict) else {}
        return _jsonable(actor) if self.expose_actor_pose else {}

    def get_object_pose(self, name: str) -> dict[str, Any]:
        public_pose = self.get_public_pose(name)
        if public_pose.get("ok"):
            return public_pose
        poses = self.get_actor_poses()
        if name not in poses:
            return {"ok": False, "name": name, "message": f"object '{name}' not found"}
        pose = poses[name]
        return {"ok": True, "name": name, "pose": pose}

    def get_public_pose(self, name: str) -> dict[str, Any]:
        """Return an explicitly public coarse task anchor or target slot pose."""
        if not self._public_pose_cache:
            self._refresh_public_pose_cache()
        try:
            resolved = self._resolve_public_pose_name(name)
        except KeyError as exc:
            return {"ok": False, "name": str(name), "message": str(exc)}
        if resolved in {"object_a", "object_b"}:
            self._begin_capx_role_if_available(resolved)
        record = self._public_pose_cache.get(resolved)
        if record is None:
            return {
                "ok": False,
                "name": str(name),
                "resolved_name": resolved,
                "message": f"public pose '{resolved}' is not available",
            }
        return {
            "ok": True,
            "name": str(name),
            "resolved_name": resolved,
            "position": np.asarray(record["position"], dtype=np.float32).reshape(3).tolist(),
            "quaternion_wxyz": np.asarray(
                record["quaternion_wxyz"],
                dtype=np.float32,
            ).reshape(4).tolist(),
            "extent": np.asarray(record["extent"], dtype=np.float32).reshape(3).tolist(),
            "source": str(record.get("source", "public_anchor")),
        }

    def get_public_pose_map(self) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return public pose anchors for CaP-style Franka compatibility APIs."""
        if not self._public_pose_cache:
            self._refresh_public_pose_cache()
        pose_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for key, record in self._public_pose_cache.items():
            pose_map[key] = (
                np.asarray(record["position"], dtype=np.float32).reshape(3),
                np.asarray(record["quaternion_wxyz"], dtype=np.float32).reshape(4),
                np.asarray(record["extent"], dtype=np.float32).reshape(3),
            )
        for alias in ("current_object", "current_slot"):
            try:
                resolved = self._resolve_public_pose_name(alias)
            except KeyError:
                continue
            if resolved in pose_map:
                pose_map[alias] = tuple(item.copy() for item in pose_map[resolved])  # type: ignore[assignment]
        return pose_map

    def list_public_regions(self) -> list[str]:
        """Return task-declared public coarse region names."""
        return sorted(self.get_public_regions().keys())

    def get_public_region(self, name: str) -> dict[str, Any]:
        """Return one sanitized public coarse region by name."""
        regions = self.get_public_regions()
        try:
            key = self._resolve_public_region_name(name, regions)
        except KeyError as exc:
            return {"ok": False, "name": str(name), "message": str(exc)}
        if key not in regions:
            return {"ok": False, "name": str(name), "message": f"region {name!r} is not available"}
        return {"ok": True, "name": key, **regions[key]}

    def get_public_regions(self) -> dict[str, dict[str, Any]]:
        """Return task-declared public regions without private task state."""
        task_regions = getattr(self._task, "get_public_regions", None)
        if not callable(task_regions):
            return {}
        try:
            raw_regions = task_regions()
        except Exception as exc:
            print(f"[capx-univtac] get_public_regions failed: {exc!r}", flush=True)
            return {}
        if not isinstance(raw_regions, dict):
            return {}
        regions: dict[str, dict[str, Any]] = {}
        for name, raw in raw_regions.items():
            if not isinstance(raw, dict):
                continue
            cleaned = self._sanitize_public_region(str(name), raw)
            if cleaned is not None:
                regions[str(name)] = cleaned
        return regions

    def get_status(self) -> dict[str, Any]:
        return {
            "task": self.task_name,
            "instruction": self.get_task_instruction(),
            "step": self.get_step_count(),
            "action_count": self.get_action_count(),
            "max_steps": self.max_steps,
            "elapsed_time": time.time() - self._start_time,
            "last_action": self._last_action_result,
            "motion_plan_ok": bool(getattr(self._task, "plan_success", True)),
            "protocol": self.get_protocol_status(),
        }

    def current_raw_observation(self) -> dict[str, Any]:
        if not self._current_obs:
            self._current_obs = self._lightweight_raw_observation()
        return self._current_obs

    def reset_tactile_buffer(self) -> None:
        self._tactile_buffer.clear()
        self._last_recorded_tactile_step = None

    def set_tension_response_visualization(
        self,
        response: dict[str, Any],
        stage_memory: dict[str, Any],
    ) -> None:
        """Update the video-only 10D response panel from public OpenTac data."""
        snapshot = _build_tension_response_visualization(response, stage_memory)
        current = snapshot.get("current_values")
        if current is not None:
            self._tension_response_visualization = snapshot
            return

        previous = self._tension_response_visualization
        if previous.get("current_values") is not None:
            previous["status"] = str(snapshot.get("status", "invalid_capture"))
            previous["latest_capture_id"] = snapshot.get("capture_id")
            previous["latest_quality"] = snapshot.get("quality", {})
            previous["stage_values"] = snapshot.get("stage_values", previous.get("stage_values", {}))
            return
        self._tension_response_visualization = snapshot

    def set_tension_stage_memory_visualization(self, stage_memory: dict[str, Any]) -> None:
        """Publish frozen response medians before a live capture exists."""
        previous = self._tension_response_visualization
        previous["stage_values"] = _stage_memory_medians(stage_memory)
        if previous.get("current_values") is None:
            previous["status"] = "awaiting_live_window"

    def set_tension_response_preview(
        self,
        response: dict[str, Any],
        stage_memory: dict[str, Any],
    ) -> None:
        """Update the video-only rolling 10D preview from public observations."""
        snapshot = _build_tension_response_visualization(
            response,
            stage_memory,
            source="rolling_preview",
            allow_partial=True,
        )
        if snapshot.get("current_values") is not None:
            self._tension_response_visualization = snapshot
            return
        previous = self._tension_response_visualization
        previous["stage_values"] = snapshot.get(
            "stage_values", previous.get("stage_values", {})
        )
        previous["status"] = str(snapshot.get("status", "live_warming_up"))
        previous["latest_capture_id"] = snapshot.get("capture_id")
        previous["latest_quality"] = snapshot.get("quality", {})

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
        capture_initial_frame: bool = True,
        stream_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._record_frames = bool(enabled)
        self._record_wrist_camera = bool(wrist_camera)
        use_stream = bool(enabled and self.tension_response_panel_enabled and stream_dir)
        if clear:
            self._close_video_stream_writers()
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
            self._last_recorded_video_step = None
            self._video_record_failures = 0
            self._video_stream_frame_count = 0
        self._video_stream_enabled = use_stream
        self._video_stream_dir = (
            Path(stream_dir).expanduser() if use_stream else None
        )
        if self._video_stream_dir is not None:
            self._video_stream_dir.mkdir(parents=True, exist_ok=True)
        if enabled and capture_initial_frame:
            self._record_frame(force=True)

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        if self._video_stream_enabled:
            return []
        frames = [frame.copy() for frame in self._frame_buffer]
        if frames:
            self._write_live_preview(frames[-1], force=True)
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        if self._video_stream_enabled:
            return self._video_stream_frame_count
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        if self._video_stream_enabled:
            return []
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def begin_video_turn(self, turn_index: int) -> None:
        """Start a streamed per-code-block video when streaming is enabled."""
        if not self._video_stream_enabled:
            return
        self._close_video_stream_turn_writer()
        self._video_stream_turn_index = int(turn_index)

    def end_video_turn(self) -> None:
        """Flush the active streamed per-code-block video."""
        self._close_video_stream_turn_writer()
        self._video_stream_turn_index = None

    def finalize_streamed_video(self, output_dir: str | os.PathLike[str]) -> bool:
        """Close and atomically publish a streamed tension video set."""
        if not self._video_stream_enabled or self._video_stream_dir is None:
            return False
        self._close_video_stream_writers()
        source_dir = self._video_stream_dir
        target_dir = Path(output_dir) / "videos"
        target_dir.mkdir(parents=True, exist_ok=True)
        moved = False
        for source in sorted(source_dir.glob("video_*.mp4")):
            target = target_dir / source.name
            if getattr(self, "_video_stream_h264", False) and self._transcode_streamed_video_h264(source, target):
                pass
            else:
                source.replace(target)
            moved = True
            print(f"Saved interaction video to {target} (streamed)")
        self._video_stream_enabled = False
        self._video_stream_dir = None
        self._video_stream_turn_index = None
        return moved

    @staticmethod
    def _transcode_streamed_video_h264(source: Path, target: Path) -> bool:
        """Publish an H.264/yuv420p copy without risking the source video."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return False
        temporary = target.with_name(f".{target.stem}.h264.tmp.mp4")
        try:
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired):
            temporary.unlink(missing_ok=True)
            return False
        if completed.returncode != 0 or not temporary.exists() or temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            return False
        temporary.replace(target)
        source.unlink(missing_ok=True)
        return True

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._wrist_frame_buffer[start:end]]

    def export_debug_artifacts(self, output_dir: str | os.PathLike[str]) -> str | None:
        """Write private UniVTAC diagnostics for audit, never for LLM prompts."""
        debug_records = getattr(self, "_debug_records", [])
        gripper_trace = getattr(self, "_tactile_gripper_trace", [])
        primitive_trace = getattr(self, "_primitive_trace", [])
        working_memory_trace = getattr(self, "_tactile_working_memory_trace", [])
        trial_memory_snapshot = getattr(self, "_tactile_trial_memory_snapshot", {})
        public_probe_records = getattr(self, "_public_probe_records", {})
        perception_artifacts = getattr(self, "_perception_artifacts", [])
        runtime_preflight = getattr(self, "_runtime_preflight_diagnostics", {})
        if (
            not debug_records
            and not gripper_trace
            and not primitive_trace
            and not working_memory_trace
            and not trial_memory_snapshot
            and not public_probe_records
            and not perception_artifacts
            and not runtime_preflight
        ):
            return None
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        debug_path = output_path / "univtac_debug.json"
        preflight_path = None
        if runtime_preflight:
            preflight_path = output_path / "univtac_runtime_preflight.json"
            with open(preflight_path, "w", encoding="utf-8") as f:
                json.dump(runtime_preflight, f, indent=2, sort_keys=True)
            print(
                f"[capx-univtac] saved runtime preflight diagnostics to {preflight_path}",
                flush=True,
            )
        if debug_records:
            payload = {
                "task": self.task_name,
                "task_config": self.task_config_name,
                "metadata": _jsonable(getattr(self._task, "metadata", {})),
                "records": debug_records,
            }
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            print(
                f"[capx-univtac] saved private debug diagnostics to {debug_path}",
                flush=True,
            )
        if gripper_trace:
            trace_path = output_path / "tactile_gripper_trace.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(gripper_trace, f, indent=2, sort_keys=True)
            print(f"[capx-univtac] saved tactile gripper trace to {trace_path}", flush=True)
        primitive_path = None
        if primitive_trace:
            primitive_path = output_path / "primitive_trace.json"
            with open(primitive_path, "w", encoding="utf-8") as f:
                json.dump(primitive_trace, f, indent=2, sort_keys=True)
            print(f"[capx-univtac] saved primitive trace to {primitive_path}", flush=True)
        working_memory_path = None
        if working_memory_trace:
            working_memory_path = output_path / "tactile_working_memory_trace.json"
            with open(working_memory_path, "w", encoding="utf-8") as f:
                json.dump(working_memory_trace, f, indent=2, sort_keys=True)
            print(
                "[capx-univtac] saved tactile working memory trace "
                f"to {working_memory_path}",
                flush=True,
            )
        trial_memory_path = None
        if working_memory_trace or trial_memory_snapshot:
            trial_memory_path = output_path / "tactile_trial_memory.json"
            with open(trial_memory_path, "w", encoding="utf-8") as f:
                json.dump(trial_memory_snapshot, f, indent=2, sort_keys=True)
            print(
                "[capx-univtac] saved tactile trial memory snapshot "
                f"to {trial_memory_path}",
                flush=True,
            )
        probe_records_path = None
        if public_probe_records:
            probe_records_path = output_path / "tactile_probe_records.json"
            with open(probe_records_path, "w", encoding="utf-8") as f:
                json.dump(public_probe_records, f, indent=2, sort_keys=True)
            print(
                f"[capx-univtac] saved public probe records to {probe_records_path}",
                flush=True,
            )
        selection_path = self._export_selection_summary(output_path, trial_memory_snapshot)
        audit_path = self._export_oracle_audit(output_path, trial_memory_snapshot)
        perception_path = self._export_perception_artifacts(output_path)
        self._export_pre_move_tactile_timeline(output_path)
        if debug_records:
            return str(debug_path)
        if gripper_trace:
            return str(trace_path)
        if primitive_path is not None:
            return str(primitive_path)
        if working_memory_path is not None:
            return str(working_memory_path)
        if trial_memory_path is not None:
            return str(trial_memory_path)
        if probe_records_path is not None:
            return str(probe_records_path)
        if selection_path is not None:
            return str(selection_path)
        if audit_path is not None:
            return str(audit_path)
        if preflight_path is not None:
            return str(preflight_path)
        return str(perception_path) if perception_path is not None else None

    def _export_selection_summary(
        self,
        output_path: Path,
        trial_memory_snapshot: dict[str, Any],
    ) -> Path | None:
        records = (
            trial_memory_snapshot.get("records", {})
            if isinstance(trial_memory_snapshot, dict)
            else {}
        )
        selection = records.get("selection") if isinstance(records, dict) else None
        if not isinstance(selection, dict):
            return None
        path = output_path / "selection_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(selection), f, indent=2, sort_keys=True)
        print(f"[capx-univtac] saved selection summary to {path}", flush=True)
        return path

    def _export_oracle_audit(
        self,
        output_path: Path,
        trial_memory_snapshot: dict[str, Any],
    ) -> Path | None:
        """Persist the hidden-label result only after code execution ends."""
        if self.task_name != "tactile_memory_match":
            return None
        records = (
            trial_memory_snapshot.get("records", {})
            if isinstance(trial_memory_snapshot, dict)
            else {}
        )
        selection = records.get("selection") if isinstance(records, dict) else None
        selected = None
        if isinstance(selection, dict) and isinstance(selection.get("data"), dict):
            selected = selection["data"].get("selected_candidate")
        true_match = getattr(self._task, "match_candidate_public_name", None)
        if selected is None:
            return None
        try:
            task_completed = bool(self._task.check_success())
        except Exception:
            task_completed = False
        audit = {
            "schema_version": "capx_tactile_memory_oracle_audit.v1",
            "audit_stage": "post_execution_only",
            "evaluation_mode": "selection_only" if self.selection_only_audit else "task_completion",
            "selected_candidate": selected,
            "true_match_candidate": true_match,
            "selection_correct": bool(selected is not None and selected == true_match),
            "task_completed": task_completed,
        }
        path = output_path / "oracle_audit.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(audit), f, indent=2, sort_keys=True)
        print(f"[capx-univtac] saved post-execution oracle audit to {path}", flush=True)
        return path

    def _export_perception_artifacts(self, output_path: Path) -> Path | None:
        artifacts = getattr(self, "_perception_artifacts", [])
        if not artifacts:
            return None
        perception_path = output_path / "perception"
        perception_path.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, Any]] = []
        array_keys = {
            "frame",
            "mask",
            "points_world",
            "grasps_camera",
            "grasp_scores",
        }
        for index, record in enumerate(artifacts):
            kind = str(record.get("kind", "estimate")).replace("/", "_")
            stem = f"{index:02d}_{kind}"
            frame = record.get("frame")
            if frame is not None:
                rgb = np.asarray(frame.rgb, dtype=np.uint8)
                depth = np.asarray(frame.depth, dtype=np.float32)
                Image.fromarray(rgb).save(perception_path / f"{stem}_rgb.png")
                Image.fromarray(_depth_visualization(depth)).save(
                    perception_path / f"{stem}_depth.png"
                )
                np.savez_compressed(
                    perception_path / f"{stem}_rgbd.npz",
                    depth=depth,
                    intrinsics=np.asarray(frame.intrinsics, dtype=np.float32),
                    camera_position=np.asarray(frame.camera_position, dtype=np.float32),
                    camera_quaternion_wxyz=np.asarray(
                        frame.camera_quaternion_wxyz,
                        dtype=np.float32,
                    ),
                )
            mask = record.get("mask")
            if mask is not None:
                mask_image = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
                Image.fromarray(mask_image).save(perception_path / f"{stem}_mask.png")
            points_world = record.get("points_world")
            if points_world is not None:
                np.savez_compressed(
                    perception_path / f"{stem}_points_world.npz",
                    points_world=np.asarray(points_world, dtype=np.float32),
                )
            grasps_camera = record.get("grasps_camera")
            if grasps_camera is not None:
                np.savez_compressed(
                    perception_path / f"{stem}_grasps.npz",
                    grasps_camera=np.asarray(grasps_camera, dtype=np.float32),
                    scores=np.asarray(record.get("grasp_scores", []), dtype=np.float32),
                )
            metadata = {
                key: _jsonable(value)
                for key, value in record.items()
                if key not in array_keys
            }
            metadata["index"] = index
            metadata["stem"] = stem
            manifest.append(metadata)

        manifest_path = perception_path / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(
            "[capx-univtac] saved RGB-D perception artifacts "
            f"records={len(manifest)} path={perception_path}",
            flush=True,
        )
        return manifest_path

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        obs = self._read_native_observation(include_camera=True, include_tactile=True)
        self._current_obs = obs
        return self._render_video_frame(obs)

    def render_wrist(self) -> np.ndarray | None:
        obs = self._read_native_observation(include_camera=True, include_tactile=False)
        wrist = obs.get("observation", {}).get("wrist", {}).get("rgb")
        if wrist is None:
            return None
        return _as_uint8_rgb(wrist)

    def close(self) -> None:
        if self._task is not None and hasattr(self._task, "close"):
            self._task.close()

    def _prepare_import_path(self) -> None:
        root_str = str(self.univtac_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

    def _build_task(self) -> None:
        task_config_file = self._task_config_path()
        with open(task_config_file, encoding="utf-8") as f:
            self._task_config = yaml.safe_load(f) or {}
        config_task_overrides = dict(self._task_config.get("task_cfg_overrides", {}) or {})
        runtime_task_overrides = dict(self.task_config_overrides.get("task_cfg_overrides", {}) or {})
        self._task_config.update(self.task_config_overrides)
        if config_task_overrides or runtime_task_overrides:
            self._task_config["task_cfg_overrides"] = {
                **config_task_overrides,
                **runtime_task_overrides,
            }

        task_module = importlib.import_module(f"envs.{self.task_name}")
        env_cfg = task_module.TaskCfg()
        env_cfg.save_dir = (
            Path(self._task_config.get("save_dir", "./data"))
            / self.task_name
            / self.task_config_name
            / "capx"
        )
        env_cfg.tactile_sensor_type = self._task_config.get("sensor_type", "gsmini")
        env_cfg.decimation = self._task_config.get("decimation", env_cfg.decimation)
        obs_data_type = dict(self._task_config.get("observations", {}))
        if not self.expose_actor_pose:
            obs_data_type.pop("actor", None)
        env_cfg.obs_data_type = obs_data_type
        env_cfg.save_frequency = self._task_config.get("save_frequency", env_cfg.save_frequency)
        env_cfg.video_frequency = self._task_config.get("video_frequency", env_cfg.video_frequency)
        env_cfg.render_frequency = self._task_config.get("render_frequency", 0)
        env_cfg.random_texture = self._task_config.get("random_texture", False)
        env_cfg.reset_time_limit = self._task_config.get("reset_time_limit", env_cfg.reset_time_limit)
        env_cfg.step_lim = self._task_config.get("step_lim", env_cfg.step_lim)
        env_cfg.max_save_frames = self._task_config.get("max_save_frames", env_cfg.max_save_frames)
        env_cfg.planner_time_dilation_factor = self._task_config.get(
            "planner_time_dilation_factor",
            env_cfg.planner_time_dilation_factor,
        )
        for key, value in dict(self._task_config.get("task_cfg_overrides", {}) or {}).items():
            if not hasattr(env_cfg, key):
                raise ValueError(
                    f"task_cfg_overrides contains undeclared TaskCfg field {key!r} "
                    f"for task {self.task_name!r}"
                )
            setattr(env_cfg, key, value)
        env_cfg.scene.num_envs = 1
        if self.device_override:
            env_cfg.sim.device = self.device_override
        self._force_task_mode = self._force_task_requested or bool(
            self._task_config.get("force_task", False)
        )
        if self._force_task_mode:
            if self.force_task_seed is None:
                raise ValueError("force_task config requires low_level.force_task_seed")
            try:
                from envs._force_task_utils import prepare_force_task_config
            except Exception as exc:
                raise RuntimeError(
                    "force_task config requires a ViTaForge-compatible task root"
                ) from exc
            prepare_force_task_config(
                env_cfg,
                self._task_config,
                task_config_file,
                seed=self.force_task_seed,
            )
            # Preserve the final public tactile response until generated code
            # returns; the task's physical scorer still remains authoritative.
            env_cfg.capx_defer_terminal_on_success = True
        self._task = task_module.Task(env_cfg, mode="eval")
        self.record_video_during_reset = bool(
            self._task_config.get(
                "record_video_during_reset",
                getattr(self, "record_video_during_reset", True),
            )
        )
        self._official_task_protocol = bool(
            self._task_config.get("official_task_protocol", False)
        )
        self._validate_runtime_preflight()
        self._install_task_runtime_patches()

    def _validate_runtime_preflight(self) -> None:
        """Fail before an LLM query when a force-task TacEx stack is mixed."""
        config = self.runtime_preflight
        if not bool(config.get("enabled", False)):
            return

        expected_root_value = config.get("tacex_root") or os.getenv(
            "TACEX_RUNTIME_ROOT", ""
        )
        expected_root = Path(str(expected_root_value)).expanduser().resolve()
        diagnostics: dict[str, Any] = {
            "schema_version": "capx_univtac_runtime_preflight.v1",
            "task": self.task_name,
            "tacex_root": str(expected_root),
            "modules": {},
            "assets_dir": None,
            "attachments": {},
            "robot_asset": None,
            "errors": [],
        }
        errors: list[str] = diagnostics["errors"]

        if not expected_root.is_dir():
            errors.append(f"TacEx root does not exist: {expected_root}")
        for package in ("tacex", "tacex_uipc", "tacex_assets", "tacex_tasks"):
            try:
                if package == "tacex_tasks":
                    # This is IsaacLab's optional training-task bundle.  Its
                    # package import eagerly loads RL configs and can require
                    # rsl_rl, which the force-task runtime never uses. Verify
                    # its source resolution without executing that import.
                    spec = importlib.util.find_spec(package)
                    if spec is None:
                        raise ModuleNotFoundError(package)
                    origin = spec.origin
                    if origin is None:
                        locations = list(spec.submodule_search_locations or [])
                        if not locations:
                            raise RuntimeError("package has no source location")
                        module_path = Path(locations[0]).resolve()
                    else:
                        module_path = Path(origin).resolve()
                else:
                    module = importlib.import_module(package)
                    module_path = Path(inspect.getfile(module)).resolve()
                diagnostics["modules"][package] = str(module_path)
                expected_package_root = expected_root / "source" / package
                if not _path_within(module_path, expected_package_root):
                    errors.append(
                        f"{package} resolves to {module_path}, expected under {expected_package_root}"
                    )
            except Exception as exc:
                errors.append(f"could not import {package}: {exc!r}")

        if bool(config.get("require_contact_gradient", False)):
            try:
                from tacex_uipc.sim.uipc_sim import UipcSim

                uipc_sim_path = Path(inspect.getfile(UipcSim)).resolve()
                diagnostics["uipc_sim"] = str(uipc_sim_path)
                if not _path_within(
                    uipc_sim_path, expected_root / "source" / "tacex_uipc"
                ):
                    errors.append(
                        "UipcSim resolves outside the requested TacEx runtime: "
                        f"{uipc_sim_path}"
                    )
                if not hasattr(UipcSim, "get_contact_gradient"):
                    errors.append("UipcSim.get_contact_gradient is missing")
            except Exception as exc:
                errors.append(f"could not validate UipcSim contact gradient: {exc!r}")

            task_uipc_sim = getattr(self._task, "uipc_sim", None)
            task_uipc_type = type(task_uipc_sim) if task_uipc_sim is not None else None
            task_uipc_path = None
            if task_uipc_type is not None:
                try:
                    task_uipc_path = Path(inspect.getfile(task_uipc_type)).resolve()
                except (OSError, TypeError):
                    task_uipc_path = None
            diagnostics["task_uipc_sim"] = {
                "class": (
                    f"{task_uipc_type.__module__}.{task_uipc_type.__qualname__}"
                    if task_uipc_type is not None
                    else None
                ),
                "source": str(task_uipc_path) if task_uipc_path is not None else None,
                "has_contact_gradient": bool(
                    callable(getattr(task_uipc_sim, "get_contact_gradient", None))
                ),
            }
            if task_uipc_sim is None:
                errors.append("task did not construct a UipcSim instance")
            elif not callable(getattr(task_uipc_sim, "get_contact_gradient", None)):
                errors.append(
                    "task UipcSim instance has no get_contact_gradient method"
                )
            elif task_uipc_path is not None and not _path_within(
                task_uipc_path, expected_root / "source" / "tacex_uipc"
            ):
                errors.append(
                    "task UipcSim instance resolves outside the requested TacEx runtime: "
                    f"{task_uipc_path}"
                )

        try:
            from tacex_assets import TACEX_ASSETS_DATA_DIR

            assets_dir = Path(str(TACEX_ASSETS_DATA_DIR)).expanduser().resolve()
            diagnostics["assets_dir"] = str(assets_dir)
            if not _path_within(assets_dir, expected_root):
                errors.append(
                    "TacEx asset directory resolves outside the requested runtime: "
                    f"{assets_dir}"
                )
        except Exception as exc:
            errors.append(f"could not validate TacEx asset directory: {exc!r}")

        robot_cfg = getattr(getattr(self._task, "cfg", None), "robot", None)
        spawn = getattr(getattr(robot_cfg, "robot", None), "spawn", None)
        robot_asset = str(getattr(spawn, "usd_path", ""))
        diagnostics["robot_asset"] = robot_asset
        if bool(config.get("require_attached_gelpad_asset", False)) and not robot_asset.endswith(
            "uipc_gelpads_high_res_wrist_attached.usda"
        ):
            errors.append(
                "robot asset is not the ViTaForge attached gelpad USD: "
                f"{robot_asset or '<missing>'}"
            )

        if bool(config.get("require_attachment_points", False)):
            tactiles = getattr(
                getattr(self._task, "_tactile_manager", None), "tactiles", {}
            )
            for name in ("left_tactile", "right_tactile"):
                tactile = tactiles.get(name) if isinstance(tactiles, dict) else None
                attachment = getattr(tactile, "attachment", None)
                count = getattr(attachment, "num_attachment_points_per_obj", None)
                diagnostics["attachments"][name] = count
                if not isinstance(count, int) or count <= 0:
                    errors.append(f"{name} attachment has no points: {count!r}")

        self._runtime_preflight_diagnostics = _jsonable(diagnostics)
        if errors:
            diagnostic_path = self._write_runtime_preflight_failure_diagnostics()
            if diagnostic_path is not None:
                diagnostics["diagnostic_path"] = str(diagnostic_path)
                self._runtime_preflight_diagnostics = _jsonable(diagnostics)
            message = "CAPX_ENVIRONMENT_ERROR: " + " | ".join(errors)
            print(f"[capx-univtac] {message}", flush=True)
            raise RuntimeError(message)
        print(
            "[capx-univtac] runtime preflight passed "
            f"asset={Path(robot_asset).name} "
            f"attachments={diagnostics['attachments']}",
            flush=True,
        )

    def _write_runtime_preflight_failure_diagnostics(self) -> Path | None:
        """Persist an early runtime mismatch when no trial object exists yet."""
        output_value = os.getenv("CAPX_OUTPUT_DIR", "").strip()
        if not output_value:
            return None
        try:
            output_dir = Path(output_value).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            diagnostic_path = output_dir / "univtac_runtime_preflight_error.json"
            with open(diagnostic_path, "w", encoding="utf-8") as f:
                json.dump(
                    self._runtime_preflight_diagnostics,
                    f,
                    indent=2,
                    sort_keys=True,
                )
            print(
                "[capx-univtac] saved runtime preflight failure diagnostics to "
                f"{diagnostic_path}",
                flush=True,
            )
            return diagnostic_path
        except OSError as exc:
            print(
                "WARNING: failed to save runtime preflight diagnostics: "
                f"{exc!r}",
                flush=True,
            )
            return None

    def _task_config_path(self) -> Path:
        path = Path(self.task_config_name)
        if path.suffix in {".yaml", ".yml"}:
            return path if path.is_absolute() else self.univtac_root / path
        return self.univtac_root / "task_config" / f"{self.task_config_name}.yml"

    def _install_task_runtime_patches(self) -> None:
        """Install small CaP-X-only task patches without touching UniVTAC policies."""
        skip_task_pre_move = bool(self._task_config.get("skip_task_pre_move", False))
        skip_pre_move_render = bool(self._task_config.get("skip_pre_move_render", False))
        debug_pre_move = _env_flag("UNIVTAC_DEBUG_PREMOVE") or bool(
            self._task_config.get("debug_pre_move", False)
        )
        self._record_action_frames = bool(self._task_config.get("record_action_frames", True))
        self._record_pre_move_frames = bool(self._task_config.get("record_pre_move_frames", True))
        self._video_frame_stride = max(1, int(self._task_config.get("video_frame_stride", 1)))

        if skip_task_pre_move:
            def _capx_noop_pre_move():
                target = getattr(self._task, "target", None)
                if target is not None:
                    try:
                        self._task.target_pose = target.get_pose().add_bias([0.0, 0.0, 0.015])
                    except Exception:
                        pass
                self._task._capx_pre_move_skipped = True
                print("[capx-univtac] task pre_move skipped for CaP manual grasp", flush=True)

            self._task.pre_move = _capx_noop_pre_move

        original_step = self._task._step

        def _capx_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self._notify_post_step_observers()
            self._record_pre_move_tactile_step()
            # Read the post-step tactile frame so externally executed probes
            # are reduced by the same v3 schema as the task-side expert.
            self._record_active_public_probe_frame()
            self._record_frame_after_task_step()
            return result

        self._task._step = _capx_step

        if skip_pre_move_render:
            original_update_render = self._task._update_render

            def _capx_update_render():
                in_pre_move = bool(getattr(self._task, "in_pre_move", False))
                atom_tag = str(getattr(self._task, "atom_tag", ""))
                if in_pre_move and atom_tag in {"delay", "move"}:
                    if debug_pre_move:
                        print(
                            "[capx-univtac] skip pre_move render "
                            f"atom={atom_tag} step={self.get_step_count()}",
                            flush=True,
                        )
                    return None
                return original_update_render()

            self._task._update_render = _capx_update_render

        if debug_pre_move:
            original_pre_move = self._task.pre_move
            original_move = self._task.move
            original_delay = self._task.delay

            def _capx_pre_move(*args, **kwargs):
                start = time.perf_counter()
                print("[capx-univtac] pre_move begin", flush=True)
                try:
                    return original_pre_move(*args, **kwargs)
                finally:
                    print(
                        "[capx-univtac] pre_move end "
                        f"step={self.get_step_count()} cost={time.perf_counter() - start:.2f}s",
                        flush=True,
                    )

            def _capx_move(*args, **kwargs):
                start = time.perf_counter()
                print(
                    "[capx-univtac] move begin "
                    f"step={self.get_step_count()} actions={len(args[0]) if args else 'unknown'}",
                    flush=True,
                )
                try:
                    return original_move(*args, **kwargs)
                finally:
                    print(
                        "[capx-univtac] move end "
                        f"step={self.get_step_count()} cost={time.perf_counter() - start:.2f}s",
                        flush=True,
                    )

            def _capx_delay(*args, **kwargs):
                steps = args[0] if args else kwargs.get("steps", 20)
                start = time.perf_counter()
                print(
                    "[capx-univtac] delay begin "
                    f"step={self.get_step_count()} n={steps}",
                    flush=True,
                )
                try:
                    return original_delay(*args, **kwargs)
                finally:
                    print(
                        "[capx-univtac] delay end "
                        f"step={self.get_step_count()} cost={time.perf_counter() - start:.2f}s",
                        flush=True,
                    )

            self._task.pre_move = _capx_pre_move
            self._task.move = _capx_move
            self._task.delay = _capx_delay

    def _append_debug_record(self, label: str) -> dict[str, Any]:
        record = self._debug_snapshot(label)
        self._debug_records.append(record)
        return record

    def _record_pre_move_tactile_step(self) -> None:
        if not bool(getattr(self._task, "in_pre_move", False)):
            return
        if not bool(self._task_config.get("record_pre_move_tactile_timeline", True)):
            return
        step = self.get_step_count()
        if step <= 0:
            return
        stride = max(1, int(self._task_config.get("pre_move_tactile_stride", 1)))
        if step % stride != 0:
            return
        try:
            obs = self._read_native_observation(
                include_camera=False,
                include_tactile=True,
                include_embodiment=False,
                include_actor=False,
                tactile_data_types=["depth", "marker", "pose"],
            )
            record = self._pre_move_tactile_record(obs)
            self._pre_move_tactile_timeline.append(record)
        except Exception as exc:
            message = repr(exc)
            if message != self._pre_move_tactile_last_error:
                print(
                    f"WARNING: failed to record UniVTAC pre_move tactile timeline: {message}",
                    flush=True,
                )
                self._pre_move_tactile_last_error = message

    def _pre_move_tactile_record(self, obs: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "step": self.get_step_count(),
            "atom_id": int(getattr(self._task, "atom_id", 0)),
            "atom_tag": str(getattr(self._task, "atom_tag", "")),
            "time_s": float(time.time() - self._start_time),
        }
        task = self._task
        actor = getattr(task, "prism", None)
        robot_manager = getattr(task, "_robot_manager", None)
        if actor is not None:
            try:
                pose = actor.get_pose()
                record["prism_position"] = _jsonable(np.asarray(pose.p, dtype=np.float32).reshape(3))
                record["prism_z"] = float(pose.p[2])
            except Exception as exc:
                record["prism_error"] = repr(exc)
        if robot_manager is not None:
            try:
                gripper_pose = robot_manager.get_gripper_center_pose()
                record["gripper_position"] = _jsonable(np.asarray(gripper_pose.p, dtype=np.float32).reshape(3))
                record["gripper_z"] = float(gripper_pose.p[2])
            except Exception as exc:
                record["gripper_pose_error"] = repr(exc)
            try:
                record["gripper_qpos"] = float(robot_manager.get_gripper_qpos())
            except Exception as exc:
                record["gripper_qpos_error"] = repr(exc)
            if actor is not None:
                try:
                    inhand_pose = robot_manager.get_inhand_pose(actor)
                    record["prism_in_gripper_position"] = _jsonable(
                        np.asarray(inhand_pose.p, dtype=np.float32).reshape(3)
                    )
                except Exception as exc:
                    record["prism_in_gripper_error"] = repr(exc)
        tactile = obs.get("tactile", {})
        hands: dict[str, Any] = {}
        for hand in ("left_tactile", "right_tactile"):
            hand_obs = tactile.get(hand, {})
            hands[hand] = self._summarize_native_tactile_hand(hand_obs)
        record["tactile"] = hands
        left = hands.get("left_tactile", {})
        right = hands.get("right_tactile", {})
        record["contact_balance"] = _balanced_difference(
            float(left.get("contact_area_px", 0.0)),
            float(right.get("contact_area_px", 0.0)),
        )
        record["marker_balance"] = _balanced_difference(
            float(left.get("marker_count", 0.0)),
            float(right.get("marker_count", 0.0)),
        )
        return _jsonable(record)

    def _summarize_native_tactile_hand(self, hand_obs: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        depth = hand_obs.get("depth")
        if depth is not None:
            depth_arr = np.asarray(_to_numpy(depth), dtype=np.float32)
            while depth_arr.ndim > 2 and depth_arr.shape[0] == 1:
                depth_arr = depth_arr[0]
            finite = depth_arr[np.isfinite(depth_arr)]
            if finite.size:
                far_plane = float(getattr(self._task.cfg.robot, "tactile_far_plane", 30.0))
                margin = max(0.1, float(self._task_config.get("pre_move_depth_contact_margin_mm", 0.5)))
                contact_mask = finite < (far_plane - margin)
                indentation = np.clip(far_plane - finite, 0.0, None)
                summary.update(
                    {
                        "depth_min_mm": float(np.min(finite)),
                        "depth_mean_mm": float(np.mean(finite)),
                        "depth_max_mm": float(np.max(finite)),
                        "depth_far_plane_mm": far_plane,
                        "depth_indentation_max_mm": float(np.max(indentation)),
                        "depth_indentation_mean_mm": float(np.mean(indentation)),
                        "contact_area_px": int(np.count_nonzero(contact_mask)),
                        "contact_area_ratio": float(np.count_nonzero(contact_mask) / finite.size),
                    }
                )
        marker = hand_obs.get("marker")
        if marker is not None:
            marker_arr = np.asarray(_to_numpy(marker), dtype=np.float32)
            marker_stats = _marker_motion_stats(marker_arr)
            summary.update(marker_stats)
        return summary

    def _export_pre_move_tactile_timeline(self, output_dir: Path) -> None:
        if not self._pre_move_tactile_timeline:
            return
        json_path = output_dir / "pre_move_tactile_timeline.json"
        csv_path = output_dir / "pre_move_tactile_timeline.csv"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._pre_move_tactile_timeline, f, indent=2, sort_keys=True)
        rows = [_flatten_timeline_record(record) for record in self._pre_move_tactile_timeline]
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(columns) + "\n")
            for row in rows:
                f.write(",".join(_csv_cell(row.get(column, "")) for column in columns) + "\n")
        print(
            "[capx-univtac] saved pre_move tactile timeline "
            f"records={len(self._pre_move_tactile_timeline)} path={json_path}",
            flush=True,
        )

    def _debug_snapshot(self, label: str) -> dict[str, Any]:
        task = self._task
        record: dict[str, Any] = {
            "label": label,
            "step": self.get_step_count(),
            "action_count": self.get_action_count(),
            "plan_success": bool(getattr(task, "plan_success", True)),
            "pre_move_skipped": bool(getattr(task, "_capx_pre_move_skipped", False)),
            "in_pre_move": bool(getattr(task, "in_pre_move", False)),
            "deterministic_inhand_follow": bool(
                getattr(task, "_deterministic_inhand_follow", False)
            ),
        }
        if self._opentac_tension_estimator_diagnostics:
            record["opentac_tension_estimator"] = _jsonable(
                self._opentac_tension_estimator_diagnostics
            )

        target_inhand_pose = getattr(task, "origin_inhand_pose", None)
        if target_inhand_pose is None:
            target_inhand_pose = getattr(task, "_deterministic_inhand_pose", None)
        if target_inhand_pose is not None:
            record["target_inhand_pose"] = _pose_json(target_inhand_pose)

        actor_key = "prism"
        actor = getattr(task, "prism", None)
        if actor is None:
            actor_key = "can"
            actor = getattr(task, "can", None)
        if actor is not None:
            try:
                pose = actor.get_pose()
                pose_json = _pose_json(pose)
                record[f"{actor_key}_pose"] = pose_json
                record["object_pose"] = pose_json
                record["object_height"] = float(pose.p[2])
                record["active_actor_name"] = str(
                    getattr(getattr(actor, "cfg", None), "name", actor_key)
                )
            except Exception as exc:
                record[f"{actor_key}_pose_error"] = repr(exc)

        robot_manager = getattr(task, "_robot_manager", None)
        if robot_manager is not None:
            try:
                record["ee_pose"] = _pose_json(robot_manager.get_ee_pose())
            except Exception as exc:
                record["ee_pose_error"] = repr(exc)
            try:
                record["gripper_center_pose"] = _pose_json(robot_manager.get_gripper_center_pose())
            except Exception as exc:
                record["gripper_center_pose_error"] = repr(exc)
            if actor is not None:
                try:
                    inhand_pose = robot_manager.get_inhand_pose(actor)
                    inhand_json = _pose_json(inhand_pose)
                    record[f"{actor_key}_in_gripper"] = inhand_json
                    record["object_in_gripper"] = inhand_json
                except Exception as exc:
                    record[f"{actor_key}_in_gripper_error"] = repr(exc)
            try:
                record["gripper_qpos"] = float(robot_manager.get_gripper_qpos())
            except Exception as exc:
                record["gripper_qpos_error"] = repr(exc)

        prism_pos = _record_position(record.get("prism_pose"))
        ee_pos = _record_position(record.get("gripper_center_pose") or record.get("ee_pose"))
        if prism_pos is not None and ee_pos is not None:
            record["ee_to_prism_distance"] = float(np.linalg.norm(prism_pos - ee_pos))
            record["ee_to_prism_xy_distance"] = float(np.linalg.norm(prism_pos[:2] - ee_pos[:2]))
            record["ee_to_prism_abs_z"] = float(abs(float(prism_pos[2]) - float(ee_pos[2])))
            record["prism_above_table"] = bool(float(prism_pos[2]) > 0.035)
            inhand_pos = _record_position(record.get("prism_in_gripper"))
            target_inhand_pos = _record_position(record.get("target_inhand_pose"))
            if inhand_pos is not None and target_inhand_pos is not None:
                record["prism_in_gripper_error"] = float(
                    np.linalg.norm(inhand_pos - target_inhand_pos)
                )
            record["pregrasp_ok"] = bool(
                not record["pre_move_skipped"]
                and record["plan_success"]
                and record["prism_above_table"]
                and record["ee_to_prism_xy_distance"] < 0.08
                and record["ee_to_prism_abs_z"] < 0.12
                and record.get("prism_in_gripper_error", 0.0) < 0.012
            )
        else:
            record["pregrasp_ok"] = False
        if actor_key == "can":
            can_pos = _record_position(record.get("can_pose"))
            if can_pos is not None and ee_pos is not None:
                record["ee_to_can_distance"] = float(np.linalg.norm(can_pos - ee_pos))
                record["ee_to_can_xy_distance"] = float(
                    np.linalg.norm(can_pos[:2] - ee_pos[:2])
                )
                record["ee_to_can_abs_z"] = float(abs(float(can_pos[2]) - float(ee_pos[2])))
                record["can_above_table"] = bool(float(can_pos[2]) > 0.06)
                if self._initial_lift_object_z is not None:
                    lift_delta = float(can_pos[2]) - self._initial_lift_object_z
                    record["can_lift_delta"] = lift_delta
                    record["can_lift_height_ok"] = bool(
                        lift_delta >= self.lift_success_height_delta
                    )
        return _jsonable(record)

    def _instructions(self) -> list[str] | None:
        instruction_file = self.univtac_root / "instructions" / f"{self.task_name}.json"
        if not instruction_file.exists():
            return None
        try:
            with open(instruction_file, encoding="utf-8") as f:
                payload = json.load(f)
            values = payload.get("seen") if isinstance(payload, dict) else None
            if isinstance(values, list) and values:
                return [str(v) for v in values]
        except Exception:
            return None
        return None

    def _update_after_action(self) -> None:
        self._sim_step_count = self.get_action_count()
        should_record_frame = self._record_frames and not self._record_action_frames
        should_record_tactile = bool(self._task_config.get("record_action_tactile", False))
        self._current_obs = self._read_native_observation(
            include_camera=should_record_frame,
            include_tactile=should_record_tactile,
            include_embodiment=True,
            include_actor=self.expose_actor_pose,
        )
        if should_record_tactile:
            self._record_tactile_frame(force=True)
        if should_record_frame:
            self._record_frame(self._current_obs)

    def _record_frame_after_task_step(self) -> None:
        if not self._record_frames or not self._record_action_frames:
            return
        if not self._record_pre_move_frames and bool(getattr(self._task, "in_pre_move", False)):
            return
        step = self.get_step_count()
        if step <= 0 or step % self._video_frame_stride != 0:
            return
        try:
            # UniVTAC does not update rendered camera tensors during pre_move
            # unless rendering is requested. CaP-X records reset/pre_move video
            # for diagnosis, so force a render immediately before sampling.
            if bool(getattr(self._task, "in_pre_move", False)):
                self._task._update_render()
            obs = self._read_native_observation(
                include_camera=True,
                include_tactile=True,
                include_embodiment=False,
                include_actor=False,
                # Keep the public depth/marker geometry in this same rendered
                # sample so the dashboard plots and rolling 10D panel advance
                # with every saved video frame.
                tactile_data_types=["rgb", "rgb_marker", "depth", "marker"],
            )
            self._record_frame(obs)
        except Exception as exc:
            self._video_record_failures += 1
            if self._video_record_failures <= 3:
                print(
                    f"WARNING: failed to record UniVTAC action frame: {exc!r}",
                    flush=True,
                )

    def _record_tactile_frame(self, *, force: bool = False) -> None:
        if not self._current_obs:
            return
        step = self.get_step_count()
        if not force and self._last_recorded_tactile_step == step:
            return
        self._tactile_buffer.append(
            frame_from_observation(self._current_obs, step=step, timestamp=time.time())
        )
        self._last_recorded_tactile_step = step

    def _record_frame(self, obs: dict[str, Any] | None = None, *, force: bool = False) -> None:
        if not self._record_frames:
            return
        step = self.get_step_count()
        if not force and self._last_recorded_video_step == step:
            return
        obs = obs or self._read_native_observation(include_camera=True, include_tactile=True)
        self._record_tension_response_video_sample(obs)
        frame = self._render_video_frame(obs)
        if self._video_stream_enabled:
            self._write_streamed_video_frame(frame)
        else:
            self._frame_buffer.append(frame)
        self._last_recorded_video_step = step
        self._write_live_preview(frame, force=force)
        if self._record_wrist_camera:
            wrist = obs.get("observation", {}).get("wrist", {}).get("rgb")
            if wrist is not None:
                self._wrist_frame_buffer.append(_as_uint8_rgb(wrist))

    def _write_streamed_video_frame(self, frame: np.ndarray) -> None:
        if self._video_stream_dir is None:
            return
        rgb = np.ascontiguousarray(_as_uint8_rgb(frame))
        combined_path = self._video_stream_dir / "video_combined.mp4"
        self._video_stream_combined_writer = self._ensure_video_stream_writer(
            self._video_stream_combined_writer, combined_path, rgb
        )
        self._video_stream_combined_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if self._video_stream_turn_index is not None:
            turn_path = self._video_stream_dir / f"video_turn_{self._video_stream_turn_index:02d}.mp4"
            self._video_stream_turn_writer = self._ensure_video_stream_writer(
                self._video_stream_turn_writer, turn_path, rgb
            )
            self._video_stream_turn_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self._video_stream_frame_count += 1

    @staticmethod
    def _ensure_video_stream_writer(writer: Any | None, path: Path, frame: np.ndarray) -> Any:
        if writer is not None:
            return writer
        height, width = frame.shape[:2]
        created = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (int(width), int(height))
        )
        if not created.isOpened():
            raise RuntimeError(f"could not open streamed video writer: {path}")
        return created

    def _close_video_stream_turn_writer(self) -> None:
        writer = self._video_stream_turn_writer
        self._video_stream_turn_writer = None
        if writer is not None:
            writer.release()

    def _close_video_stream_writers(self) -> None:
        self._close_video_stream_turn_writer()
        writer = self._video_stream_combined_writer
        self._video_stream_combined_writer = None
        if writer is not None:
            writer.release()

    def _write_live_preview(self, frame: np.ndarray, *, force: bool = False) -> None:
        if not self.live_preview_enabled or self.live_preview_path is None:
            return
        frame_count = self.get_video_frame_count()
        if not force and frame_count % self.live_preview_stride != 0:
            return
        try:
            out_path = self.live_preview_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = out_path.with_name(f".{out_path.name}.tmp")
            image = Image.fromarray(np.ascontiguousarray(_as_uint8_rgb(frame)))
            image.save(
                tmp_path,
                format="JPEG",
                quality=self.live_preview_jpeg_quality,
            )
            tmp_path.replace(out_path)
        except Exception as exc:
            self._live_preview_write_failures += 1
            if self._live_preview_write_failures <= 3:
                print(
                    f"WARNING: failed to write UniVTAC live preview: {exc!r}",
                    flush=True,
                )

    def _render_video_frame(self, obs: dict[str, Any]) -> np.ndarray:
        if self.video_renderer != "task_native":
            return self._compose_frame(obs)
        if self.tension_response_panel_enabled:
            # The tension task uses the same 1600x1200 public-observation
            # dashboard as the expert replay, not the generic native collage
            # with an extra panel appended underneath.
            return _render_tension_strap_demo_frame(
                obs=obs,
                visualization=self._tension_response_visualization,
                control_diagnostics=self._opentac_tension_estimator_diagnostics,
                history=self._tension_response_video_history,
            )
        native_renderer = getattr(self._task, "get_frame_shot", None)
        if not callable(native_renderer):
            raise RuntimeError(
                f"task {self.task_name!r} does not provide get_frame_shot() for task_native video"
            )
        return np.ascontiguousarray(_as_uint8_rgb(native_renderer(obs)))

    def _record_tension_response_video_sample(self, obs: dict[str, Any]) -> None:
        """Collect display-only public tactile values for the native video panel."""
        if not self.tension_response_panel_enabled:
            return
        tactile = obs.get("tactile") if isinstance(obs, dict) else None
        if not isinstance(tactile, dict):
            return
        left = tactile.get("left_tactile")
        right = tactile.get("right_tactile")
        if not isinstance(left, dict) or not isinstance(right, dict):
            return
        if left.get("depth") is None or right.get("depth") is None:
            return
        frame = frame_from_observation(
            obs,
            step=self.get_step_count(),
            timestamp=time.time(),
        )
        if frame.left_depth is None or frame.right_depth is None:
            return
        self._tension_response_video_frames.append(frame)
        self._tension_response_video_frames = self._tension_response_video_frames[-8:]
        summary = summarize_native_tactile(
            self._tension_response_video_frames,
            hand="both",
        )
        left_summary = summary.get("left", {})
        right_summary = summary.get("right", {})
        diagnostics = self._opentac_tension_estimator_diagnostics
        latest_estimate = diagnostics.get("latest_estimate", {})
        latest_estimate = latest_estimate if isinstance(latest_estimate, dict) else {}
        sample = {
            "step": int(self.get_step_count()),
            "left_depth_mm": float(left_summary.get("depth_delta_mm", 0.0)),
            "right_depth_mm": float(right_summary.get("depth_delta_mm", 0.0)),
            "left_marker_displacement_px": float(
                left_summary.get("marker_displacement_px", 0.0)
            ),
            "right_marker_displacement_px": float(
                right_summary.get("marker_displacement_px", 0.0)
            ),
            "left_marker_coherence": float(left_summary.get("marker_coherence", 0.0)),
            "right_marker_coherence": float(right_summary.get("marker_coherence", 0.0)),
            "estimated_tension_N": _panel_optional_float(
                latest_estimate.get("estimated_tension_N")
            ),
            "stage_index": int(diagnostics.get("current_stage_index", 0) or 0),
            "hold_seconds": _panel_optional_float(
                diagnostics.get("current_stage_in_band_hold_seconds")
            )
            or 0.0,
        }
        if self._tension_response_video_history and (
            self._tension_response_video_history[-1].get("step") == sample["step"]
        ):
            self._tension_response_video_history[-1] = sample
        else:
            self._tension_response_video_history.append(sample)
        self._tension_response_video_history = self._tension_response_video_history[
            -self.tension_response_history_points :
        ]

    def _refresh_public_pose_cache(self) -> None:
        """Cache only task-declared public anchors and slots for LLM APIs."""
        self._public_pose_cache.clear()
        task = self._task
        if task is None:
            return

        task_public_pose_map = getattr(task, "get_public_pose_map", None)
        if callable(task_public_pose_map):
            try:
                raw_map = task_public_pose_map()
            except Exception as exc:
                print(
                    f"[capx-univtac] task public pose map failed: {exc!r}",
                    flush=True,
                )
                raw_map = {}
            if isinstance(raw_map, dict):
                for name, raw_pose in raw_map.items():
                    parsed = self._public_pose_entry_to_record(
                        raw_pose,
                        name=str(name),
                        source="task_public_pose",
                    )
                    if parsed is not None:
                        self._public_pose_cache[str(name)] = parsed

        object_extent = np.array([0.04, 0.04, 0.08], dtype=np.float32)
        slot_extent = np.array([0.10, 0.10, 0.02], dtype=np.float32)
        start_poses = getattr(task, "start_poses", None)
        if isinstance(start_poses, dict):
            for role in ("object_a", "object_b"):
                if role in self._public_pose_cache:
                    continue
                pose = start_poses.get(role)
                parsed = self._pose_to_public_record(
                    pose,
                    extent=object_extent,
                    source="reset_public_anchor",
                )
                if parsed is not None:
                    self._public_pose_cache[role] = parsed

        target_poses = getattr(task, "target_poses", None)
        if isinstance(target_poses, dict):
            for role, slot in (("object_a", "slot_a"), ("object_b", "slot_b")):
                if slot in self._public_pose_cache:
                    continue
                pose = target_poses.get(role)
                parsed = self._pose_to_public_record(
                    pose,
                    extent=slot_extent,
                    source="public_slot",
                )
                if parsed is not None:
                    self._public_pose_cache[slot] = parsed

        # Keep a conservative fallback for task implementations that expose the
        # role dictionaries later than reset but still use the same public names.
        objects = getattr(task, "objects", None)
        if isinstance(objects, dict):
            for role in ("object_a", "object_b"):
                if role in self._public_pose_cache:
                    continue
                actor = objects.get(role)
                get_pose = getattr(actor, "get_pose", None)
                if not callable(get_pose):
                    continue
                try:
                    pose = get_pose()
                except Exception:
                    continue
                parsed = self._pose_to_public_record(
                    pose,
                    extent=object_extent,
                    source="public_actor_fallback",
                )
                if parsed is not None:
                    self._public_pose_cache[role] = parsed

        self._install_public_pose_aliases()

        if self._public_pose_cache:
            print(
                "[capx-univtac] public pose cache "
                f"keys={sorted(self._public_pose_cache)}",
                flush=True,
            )

    def _public_pose_entry_to_record(
        self,
        raw_pose: Any,
        *,
        name: str,
        source: str,
    ) -> dict[str, Any] | None:
        """Parse a task-declared public pose record without task-private fields."""
        if isinstance(raw_pose, dict):
            pos = (
                raw_pose.get("position")
                if "position" in raw_pose
                else raw_pose.get("pos")
            )
            quat = raw_pose.get("quaternion_wxyz", raw_pose.get("quat", None))
            extent = raw_pose.get("extent", raw_pose.get("bbox_extent", None))
            record_source = str(raw_pose.get("source", source))
        elif isinstance(raw_pose, (list, tuple)) and len(raw_pose) >= 2:
            pos = raw_pose[0]
            quat = raw_pose[1]
            extent = raw_pose[2] if len(raw_pose) >= 3 else None
            record_source = source
        else:
            return None
        try:
            position = np.asarray(pos, dtype=np.float32).reshape(3)
            quaternion = np.asarray(
                [1.0, 0.0, 0.0, 0.0] if quat is None else quat,
                dtype=np.float32,
            ).reshape(4)
            bbox_extent = (
                self._estimate_extent_from_public_pose_name(str(name))
                if extent is None
                else np.asarray(extent, dtype=np.float32).reshape(3)
            )
        except Exception:
            return None
        if (
            not np.all(np.isfinite(position))
            or not np.all(np.isfinite(quaternion))
            or not np.all(np.isfinite(bbox_extent))
        ):
            return None
        return {
            "position": position,
            "quaternion_wxyz": quaternion,
            "extent": bbox_extent,
            "source": record_source,
        }

    @staticmethod
    def _estimate_extent_from_public_pose_name(name: str) -> np.ndarray:
        key = str(name).strip().lower().replace(" ", "_")
        if "slot" in key or "pad" in key or "target" in key:
            return np.array([0.10, 0.10, 0.03], dtype=np.float32)
        if key in {
            "reference",
            "reference_object",
            "object_a",
            "candidate_left",
            "candidate_right",
            "left_candidate",
            "right_candidate",
            "candidate_1",
            "candidate_2",
            "current_object",
            "can",
        }:
            return np.array([0.04, 0.04, 0.12], dtype=np.float32)
        return np.array([0.03, 0.03, 0.03], dtype=np.float32)

    def _install_public_pose_aliases(self) -> None:
        """Install public aliases without introducing private task state."""
        aliases = {
            "reference": "reference_object",
            "object_a": "reference_object",
            "a": "reference_object",
            "left_candidate": "candidate_left",
            "left": "candidate_left",
            "candidate_1": "candidate_left",
            "right_candidate": "candidate_right",
            "right": "candidate_right",
            "candidate_2": "candidate_right",
            "slot": "match_slot",
            "target": "match_slot",
            "target_slot": "match_slot",
            "current_slot": "match_slot",
        }
        for alias, target in aliases.items():
            if alias in self._public_pose_cache or target not in self._public_pose_cache:
                continue
            self._public_pose_cache[alias] = self._copy_public_pose_record(
                self._public_pose_cache[target],
                source_alias=target,
            )

    @staticmethod
    def _copy_public_pose_record(
        record: dict[str, Any],
        *,
        source_alias: str | None = None,
    ) -> dict[str, Any]:
        copied = {
            "position": np.asarray(record["position"], dtype=np.float32).reshape(3).copy(),
            "quaternion_wxyz": np.asarray(
                record["quaternion_wxyz"], dtype=np.float32
            ).reshape(4).copy(),
            "extent": np.asarray(record["extent"], dtype=np.float32).reshape(3).copy(),
            "source": str(record.get("source", "public_anchor")),
        }
        if source_alias is not None:
            copied["source_alias"] = str(source_alias)
        return copied

    @staticmethod
    def _pose_to_public_record(
        pose: Any,
        *,
        extent: np.ndarray,
        source: str,
    ) -> dict[str, Any] | None:
        if pose is None:
            return None
        try:
            pos = np.asarray(getattr(pose, "p"), dtype=np.float32).reshape(3)
        except Exception:
            return None
        raw_quat = getattr(pose, "q", None)
        if raw_quat is None:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            quat = np.asarray(raw_quat, dtype=np.float32).reshape(4)
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
            return None
        return {
            "position": pos.astype(np.float32),
            "quaternion_wxyz": quat.astype(np.float32),
            "extent": np.asarray(extent, dtype=np.float32).reshape(3),
            "source": source,
        }

    def _resolve_public_pose_name(self, name: str) -> str:
        key = str(name).strip().lower().replace(" ", "_")
        if key in self._public_pose_cache:
            source_alias = self._public_pose_cache[key].get("source_alias")
            if isinstance(source_alias, str) and source_alias in self._public_pose_cache:
                return source_alias
            return key
        memory_aliases = {
            "reference": "reference_object",
            "reference_object": "reference_object",
            "object_a": "reference_object",
            "a": "reference_object",
            "candidate_left": "candidate_left",
            "left_candidate": "candidate_left",
            "left": "candidate_left",
            "candidate_1": "candidate_left",
            "candidate_right": "candidate_right",
            "right_candidate": "candidate_right",
            "right": "candidate_right",
            "candidate_2": "candidate_right",
            "match_slot": "match_slot",
            "slot": "match_slot",
            "target": "match_slot",
            "target_slot": "match_slot",
            "current_slot": "match_slot",
        }
        resolved_memory = memory_aliases.get(key)
        if resolved_memory in self._public_pose_cache:
            return str(resolved_memory)
        if key in {"current", "current_object", "object", "target_object", "can"}:
            return self._current_transfer_object_name()
        if key in {"current_slot", "slot", "target", "target_slot"}:
            return "slot_b" if self._current_transfer_object_name() == "object_b" else "slot_a"
        aliases = {
            "a": "object_a",
            "first": "object_a",
            "first_object": "object_a",
            "object_a": "object_a",
            "b": "object_b",
            "second": "object_b",
            "second_object": "object_b",
            "object_b": "object_b",
            "slot_a": "slot_a",
            "a_slot": "slot_a",
            "target_a": "slot_a",
            "slot_b": "slot_b",
            "b_slot": "slot_b",
            "target_b": "slot_b",
        }
        resolved = aliases.get(key, key)
        if resolved in self._public_pose_cache:
            return resolved
        if resolved not in {"object_a", "object_b", "slot_a", "slot_b"}:
            raise KeyError(f"unknown public pose {name!r}")
        return resolved

    def _resolve_public_region_name(self, name: str, regions: dict[str, Any]) -> str:
        key = str(name).strip().lower().replace(" ", "_")
        aliases = {
            "pickup_a": "pickup_a_region",
            "object_a": "pickup_a_region",
            "a": "pickup_a_region",
            "pickup_b": "pickup_b_region",
            "object_b": "pickup_b_region",
            "b": "pickup_b_region",
            "slot_a": "slot_a_region",
            "target_a": "slot_a_region",
            "slot_b": "slot_b_region",
            "target_b": "slot_b_region",
            "reference": "reference_region",
            "reference_object": "reference_region",
            "candidate_left": "candidate_left_region",
            "left_candidate": "candidate_left_region",
            "candidate_right": "candidate_right_region",
            "right_candidate": "candidate_right_region",
            "match_slot": "match_slot_region",
            "target_slot": "match_slot_region",
        }
        resolved = aliases.get(key, key)
        if resolved in regions:
            return resolved
        raise KeyError(f"unknown public region {name!r}")

    @staticmethod
    def _sanitize_public_region(name: str, raw: dict[str, Any]) -> dict[str, Any] | None:
        allowed = {
            "kind",
            "center_xy",
            "half_extents",
            "radius",
            "hover_z",
            "search_z_range",
            "release_z",
            "description",
            "source",
        }
        cleaned = {key: _jsonable(value) for key, value in raw.items() if key in allowed}
        if "center_xy" not in cleaned:
            return None
        try:
            center_xy = np.asarray(cleaned["center_xy"], dtype=np.float32).reshape(2)
        except Exception:
            return None
        if not np.all(np.isfinite(center_xy)):
            return None
        cleaned["center_xy"] = center_xy.tolist()
        if "half_extents" in cleaned:
            try:
                half_extents = np.asarray(cleaned["half_extents"], dtype=np.float32).reshape(2)
            except Exception:
                half_extents = np.array([0.03, 0.03], dtype=np.float32)
            cleaned["half_extents"] = np.maximum(half_extents, 0.0).tolist()
        if "search_z_range" in cleaned:
            try:
                z_range = np.asarray(cleaned["search_z_range"], dtype=np.float32).reshape(2)
            except Exception:
                z_range = np.array([0.035, float(cleaned.get("hover_z", 0.16))], dtype=np.float32)
            cleaned["search_z_range"] = [float(np.min(z_range)), float(np.max(z_range))]
        for key in ("hover_z", "release_z", "radius"):
            if key in cleaned:
                try:
                    cleaned[key] = float(cleaned[key])
                except Exception:
                    cleaned.pop(key, None)
        cleaned["name"] = str(name)
        cleaned.setdefault("source", "task_public_region")
        return cleaned

    def _current_transfer_object_name(self) -> str:
        task = self._task
        placed = getattr(task, "object_placed", {}) if task is not None else {}
        if isinstance(placed, dict) and bool(placed.get("object_a", False)):
            return "object_b"
        active_role = getattr(task, "active_role", None) if task is not None else None
        if active_role in {"object_a", "object_b"}:
            return str(active_role)
        return "object_a"

    def _begin_capx_role_if_available(self, role_name: str) -> None:
        begin_fn = getattr(self._task, "begin_capx_role", None)
        if not callable(begin_fn):
            return
        try:
            result = begin_fn(role_name)
            if isinstance(result, dict):
                print(
                    "[capx-univtac] begin_capx_role "
                    f"role={result.get('role', role_name)} "
                    f"vision_disabled={result.get('vision_disabled')}",
                    flush=True,
                )
        except Exception as exc:
            print(
                "[capx-univtac] begin_capx_role failed "
                f"role={role_name} error={exc!r}",
                flush=True,
            )

    def _public_grasp_actor(self, object_name: str):
        task_actor = getattr(self._task, "get_public_grasp_actor", None)
        if callable(task_actor):
            actor = task_actor(object_name)
            if actor is not None:
                return actor

        key = str(object_name).strip().lower().replace("_", " ")
        if key == "can":
            return getattr(self._task, "can", None)
        if key in {"prism", "object", "block", "grasped object", "target object"}:
            actor = getattr(self._task, "prism", None)
            return actor if actor is not None else getattr(self._task, "can", None)
        return None

    def _make_public_grasp_pose(
        self,
        object_name: str,
        actor: Any,
        *,
        grasp_height: float,
    ) -> Any:
        from envs.utils.transforms import construct_grasp_pose

        task_pose = getattr(self._task, "make_public_grasp_pose", None)
        if callable(task_pose):
            return task_pose(object_name, actor=actor, grasp_height=float(grasp_height))

        key = str(object_name).strip().lower().replace("_", " ")
        if key == "can" or actor is getattr(self._task, "can", None):
            target_pose = actor.get_pose().add_bias([-0.065, 0.0, -0.008])
            target_mat = target_pose.to_transformation_matrix()
            x_axis = target_mat[:3, 0].reshape(-1)
            target_mat = np.vstack(
                [
                    x_axis,
                    np.cross(x_axis, [0.0, 0.0, 1.0]),
                    [0.0, 0.0, 1.0],
                ]
            )
            return construct_grasp_pose(
                target_pose.p,
                target_mat[:3, 2],
                target_mat[:3, 0],
            )

        target_pose = actor.get_pose().add_bias([0.0, 0.0, float(grasp_height)])
        return construct_grasp_pose(target_pose.p, [0, 0, 1], [1, 0, 0])

    def _compose_frame(self, obs: dict[str, Any]) -> np.ndarray:
        head = _nested_get(obs, ["observation", "head", "rgb"])
        wrist = _nested_get(obs, ["observation", "wrist", "rgb"])
        left = _first_present(
            _nested_get(obs, ["tactile", "left_tactile", "rgb_marker"]),
            _nested_get(obs, ["tactile", "left_tactile", "rgb"]),
        )
        right = _first_present(
            _nested_get(obs, ["tactile", "right_tactile", "rgb_marker"]),
            _nested_get(obs, ["tactile", "right_tactile", "rgb"]),
        )

        if head is None and wrist is None:
            return np.zeros((320, 1120, 3), dtype=np.uint8)

        head_rgb = _resize_rgb(head if head is not None else wrist, 480, 320)
        wrist_rgb = _resize_rgb(wrist if wrist is not None else head, 480, 320)
        left_rgb = _resize_rgb(left, 160, 160) if left is not None else np.zeros((160, 160, 3), dtype=np.uint8)
        right_rgb = _resize_rgb(right, 160, 160) if right is not None else np.zeros((160, 160, 3), dtype=np.uint8)

        frame = np.zeros((320, 1120, 3), dtype=np.uint8)
        frame[:, :480, :] = head_rgb
        frame[:, 480:960, :] = wrist_rgb
        frame[:160, 960:1120, :] = left_rgb
        frame[160:320, 960:1120, :] = right_rgb
        if self.memory_overlay_enabled:
            frame = self._render_tactile_memory_overlay(frame)
        return np.ascontiguousarray(frame)

    def _render_tactile_memory_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Render agent-authored public memory beside the normal video panel."""
        panel_width = 460
        panel = Image.new("RGB", (panel_width, frame.shape[0]), (18, 24, 35))
        draw = ImageDraw.Draw(panel)
        font = ImageFont.load_default()
        y = 7
        for text, color in self._tactile_memory_overlay_lines():
            draw.text((8, y), text[:76], fill=color, font=font)
            y += 12
            if y >= frame.shape[0] - 10:
                break
        return np.concatenate([frame, np.asarray(panel, dtype=np.uint8)], axis=1)

    def _tactile_memory_overlay_lines(self) -> list[tuple[str, tuple[int, int, int]]]:
        lines: list[tuple[str, tuple[int, int, int]]] = [
            ("TACTILE MEMORY | provisional composable-slot v0", (225, 235, 255)),
        ]
        active = [
            session
            for session in getattr(self, "_public_probe_sessions", {}).values()
            if session.get("active_segment") is not None
        ]
        if active:
            session = active[0]
            lines.append(
                (
                    f"PHASE: {session['object_name']} / {session['active_segment']}",
                    (255, 210, 105),
                )
            )
        else:
            lines.append(("PHASE: memory / transport", (175, 210, 245)))

        snapshot = self.get_tactile_trial_memory_snapshot()
        records = snapshot.get("records", {}) if isinstance(snapshot, dict) else {}
        aliases = {
            "reference": "evidence_reference",
            "left": "evidence_candidate_left",
            "right": "evidence_candidate_right",
        }
        for label, key in aliases.items():
            record = records.get(key) if isinstance(records, dict) else None
            data = record.get("data", {}) if isinstance(record, dict) else {}
            probe = data.get("probe", {}) if isinstance(data, dict) else {}
            quality = probe.get("quality", {}) if isinstance(probe, dict) else {}
            if not probe:
                lines.append((f"{label.upper()}: pending", (145, 155, 170)))
                continue
            valid = bool(quality.get("valid", False))
            ratio = min(
                _safe_overlay_float(quality.get("preload_bilateral_contact_ratio")),
                _safe_overlay_float(quality.get("lift_motion_bilateral_contact_ratio")),
            )
            status = "valid" if valid else "invalid"
            lines.append(
                (f"{label.upper()}: {status} bilateral={ratio:.2f}", (115, 230, 150) if valid else (255, 140, 130))
            )
            slots = data.get("slot_vectors", {}) if isinstance(data, dict) else {}
            for slot in ("weight", "roughness", "hardness"):
                vector = slots.get(slot) if isinstance(slots, dict) else None
                if isinstance(vector, dict):
                    values = vector.get("values", [])
                    confidence = _safe_overlay_float(vector.get("confidence"))
                    compact = ",".join(f"{_safe_overlay_float(value):+.3f}" for value in values[:2])
                    lines.append((f"  {slot[:1].upper()}: [{compact}] q={confidence:.2f}", (195, 205, 218)))

        selection = records.get("selection") if isinstance(records, dict) else None
        selection_data = selection.get("data", {}) if isinstance(selection, dict) else {}
        if isinstance(selection_data, dict) and selection_data:
            lines.append(("SELECTION", (225, 235, 255)))
            slot_scores = selection_data.get("slot_scores", {})
            if isinstance(slot_scores, dict):
                for slot in ("weight", "roughness", "hardness"):
                    score = slot_scores.get(slot)
                    if isinstance(score, dict):
                        lines.append(
                            (
                                f"  {slot[:1].upper()}: L={_safe_overlay_float(score.get('candidate_left')):.3f} "
                                f"R={_safe_overlay_float(score.get('candidate_right')):.3f} "
                                f"q={_safe_overlay_float(score.get('confidence')):.2f}",
                                (205, 215, 230),
                            )
                        )
            lines.append(
                (
                    f"  fused L={_safe_overlay_float(selection_data.get('score_left')):.3f} "
                    f"R={_safe_overlay_float(selection_data.get('score_right')):.3f} "
                    f"margin={_safe_overlay_float(selection_data.get('score_margin')):.3f}",
                    (255, 224, 130),
                )
            )
            lines.append(
                (f"  selected: {selection_data.get('selected_candidate', 'pending')}", (255, 224, 130))
            )
            # This is deliberately a post-transport visual audit. It is not
            # part of the public API, prompt, or generated code context.
            try:
                completed = bool(self._task.check_success())
            except Exception:
                completed = False
            if completed:
                selected = selection_data.get("selected_candidate")
                true_match = getattr(self._task, "match_candidate_public_name", None)
                correct = bool(selected == true_match)
                lines.append(
                    (f"ORACLE AUDIT: {'correct' if correct else 'incorrect'}", (110, 235, 145) if correct else (255, 130, 125))
                )
        return lines

    def _public_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        public = {
            "task_prompt": self.get_task_instruction(),
            "step": int(obs.get("step", self.get_step_count())),
            "observation": {},
            "embodiment": _jsonable(obs.get("embodiment", {})),
            "tactile": {},
        }
        for cam_name, cam_obs in obs.get("observation", {}).items():
            if isinstance(cam_obs, dict) and "rgb" in cam_obs:
                public["observation"][cam_name] = {"rgb": _as_uint8_rgb(cam_obs["rgb"])}
        tactile_public = {}
        for hand, hand_obs in obs.get("tactile", {}).items():
            if not isinstance(hand_obs, dict):
                continue
            tactile_public[hand] = {
                key: _to_numpy(value)
                for key, value in hand_obs.items()
                if key in {"rgb", "rgb_marker", "marker", "depth", "pose"}
            }
        public["tactile"] = tactile_public
        if self.expose_actor_pose:
            public["actor"] = _jsonable(obs.get("actor", {}))
        return public

    def _lightweight_raw_observation(self) -> dict[str, Any]:
        raw = {
            "observation": {},
            "embodiment": {},
            "tactile": {},
            "actor": {},
            "step": self.get_step_count(),
            "atom": {
                "id": int(getattr(self._task, "atom_id", 0)),
                "tag": str(getattr(self._task, "atom_tag", "")),
            },
        }
        if _env_flag("UNIVTAC_LIGHTWEIGHT_ROBOT_STATE"):
            try:
                raw["embodiment"] = self._task._robot_manager.get_observations(["joint", "ee"])
            except Exception:
                raw["embodiment"] = {}
        return raw

    def _read_native_observation(
        self,
        *,
        include_camera: bool,
        include_tactile: bool,
        include_embodiment: bool = True,
        include_actor: bool = False,
        tactile_data_types: list[str] | None = None,
    ) -> dict[str, Any]:
        original = self._task.cfg.obs_data_type
        selected: dict[str, Any] = {}
        if include_embodiment and "embodiment" in original:
            selected["embodiment"] = original["embodiment"]
        if include_camera and "camera" in original:
            selected["camera"] = original["camera"]
        if include_tactile and "tactile" in original:
            selected["tactile"] = tactile_data_types or original["tactile"]
        if include_actor and "actor" in original:
            selected["actor"] = original["actor"]
        debug_observation = _env_flag("UNIVTAC_DEBUG_OBSERVATION")
        if debug_observation:
            print(
                "[capx-univtac] native observation begin "
                f"camera={include_camera} tactile={include_tactile} "
                f"embodiment={include_embodiment} actor={include_actor}",
                flush=True,
            )
        self._task.cfg.obs_data_type = selected
        try:
            obs = self._task._get_observations()
            if debug_observation:
                print(
                    "[capx-univtac] native observation end "
                    f"keys={list(obs.keys())}",
                    flush=True,
                )
            return obs
        finally:
            self._task.cfg.obs_data_type = original

    def _to_tensor(self, action: np.ndarray | torch.Tensor | list[float]) -> torch.Tensor:
        if isinstance(action, torch.Tensor):
            return action.to(self._task.device).float().flatten()
        return torch.as_tensor(action, dtype=torch.float32, device=self._task.device).flatten()


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value
    return value


def _pose_json(pose: Any) -> dict[str, Any]:
    if hasattr(pose, "p") and hasattr(pose, "q"):
        return {
            "position": _jsonable(np.asarray(pose.p, dtype=np.float32).reshape(-1)[:3]),
            "quaternion_wxyz": _jsonable(np.asarray(pose.q, dtype=np.float32).reshape(-1)[:4]),
        }
    values = np.asarray(_to_numpy(pose), dtype=np.float32).reshape(-1)
    out: dict[str, Any] = {"raw": _jsonable(values)}
    if values.size >= 3:
        out["position"] = _jsonable(values[:3])
    if values.size >= 7:
        out["quaternion_wxyz"] = _jsonable(values[3:7])
    return out


def _record_position(pose_record: Any) -> np.ndarray | None:
    if not isinstance(pose_record, dict) or "position" not in pose_record:
        return None
    try:
        return np.asarray(pose_record["position"], dtype=np.float32).reshape(3)
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    value = _to_numpy(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist() if np.issubdtype(value.dtype, np.number) else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


_TENSION_RESPONSE_FIELDS: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    ("normal_static", "L depth", "mm", ("hold", "left", "depth_mm"), "identity"),
    ("normal_static", "R depth", "mm", ("hold", "right", "depth_mm"), "identity"),
    ("normal_dynamic", "L depth delta", "mm", ("end_minus_start", "left", "depth_mm"), "identity"),
    ("normal_dynamic", "R depth delta", "mm", ("end_minus_start", "right", "depth_mm"), "identity"),
    ("surface_spatial_static", "L row gradient", "px", ("hold", "left", "marker_row_gradient_px"), "identity"),
    ("surface_spatial_static", "R row gradient", "px", ("hold", "right", "marker_row_gradient_px"), "identity"),
    ("surface_spatial_static", "L col gradient", "px", ("hold", "left", "marker_col_gradient_px"), "identity"),
    ("surface_spatial_static", "R col gradient", "px", ("hold", "right", "marker_col_gradient_px"), "identity"),
    ("surface_spatial_static", "L log anisotropy", "log", ("hold", "left", "marker_anisotropy_ratio"), "log"),
    ("surface_spatial_static", "R log anisotropy", "log", ("hold", "right", "marker_anisotropy_ratio"), "log"),
)


def _empty_tension_response_visualization() -> dict[str, Any]:
    return {
        "schema_version": "capx_tension_response_panel.v1",
        "status": "waiting_for_valid_capture",
        "capture_id": None,
        "latest_capture_id": None,
        "window": {},
        "quality": {},
        "latest_quality": {},
        "current_values": None,
        "stage_values": {},
    }


def _build_tension_response_visualization(
    response: dict[str, Any],
    stage_memory: dict[str, Any],
    *,
    source: str = "stage_capture",
    allow_partial: bool = False,
) -> dict[str, Any]:
    response = response if isinstance(response, dict) else {}
    stage_memory = stage_memory if isinstance(stage_memory, dict) else {}
    quality = response.get("quality", {})
    quality = quality if isinstance(quality, dict) else {}
    current_values = _flatten_tension_response_values(response)
    current_is_valid = bool(quality.get("valid", False)) and current_values is not None
    show_current = current_values is not None and (current_is_valid or allow_partial)
    if source == "rolling_preview":
        status = "live_valid_window" if current_is_valid else "live_warming_up"
    else:
        status = "valid_capture" if current_is_valid else "invalid_capture"
    return {
        "schema_version": "capx_tension_response_panel.v1",
        "status": status,
        "source": str(source),
        "capture_id": response.get("capture_id"),
        "latest_capture_id": response.get("capture_id"),
        "window": _jsonable(response.get("window", {})),
        "quality": _jsonable(quality),
        "latest_quality": _jsonable(quality),
        "current_values": current_values if show_current else None,
        "stage_values": _stage_memory_medians(stage_memory),
    }


def _flatten_tension_response_values(response: dict[str, Any]) -> list[float] | None:
    values: list[float] = []
    for _block, _label, _unit, path, transform in _TENSION_RESPONSE_FIELDS:
        value: Any = response
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        try:
            numeric = float(value)
            if transform == "log":
                if numeric <= 0.0:
                    return None
                numeric = math.log(numeric)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        values.append(numeric)
    return values


def _stage_memory_medians(memory: dict[str, Any]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    stages = memory.get("stages", []) if isinstance(memory, dict) else []
    if not isinstance(stages, list):
        return values
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        memory_id = str(stage.get("memory_id", ""))
        if memory_id == "tension_strap_hold_12n.v1":
            label = "12N"
        elif memory_id == "tension_strap_hold_18n.v1":
            label = "18N"
        else:
            continue
        flattened: list[float] = []
        for block in stage.get("response_blocks", []):
            scaler = block.get("scaler", {}) if isinstance(block, dict) else {}
            median = scaler.get("median") if isinstance(scaler, dict) else None
            if not isinstance(median, list):
                flattened = []
                break
            try:
                flattened.extend(float(item) for item in median)
            except (TypeError, ValueError):
                flattened = []
                break
        if len(flattened) == len(_TENSION_RESPONSE_FIELDS) and np.isfinite(
            np.asarray(flattened, dtype=np.float64)
        ).all():
            values[label] = flattened
    return values


_TENSION_DEMO_WIDTH = 1600
_TENSION_DEMO_HEIGHT = 1200
_TENSION_DEMO_HEADER_HEIGHT = 70
_TENSION_DEMO_SCENE_HEIGHT = 450
_TENSION_DEMO_TACTILE_HEIGHT = 300
_TENSION_DEMO_RESPONSE_TOP = (
    _TENSION_DEMO_HEADER_HEIGHT + _TENSION_DEMO_SCENE_HEIGHT + _TENSION_DEMO_TACTILE_HEIGHT
)
_TENSION_DEMO_RESPONSE_HEIGHT = 180
_TENSION_DEMO_CHART_TOP = _TENSION_DEMO_RESPONSE_TOP + _TENSION_DEMO_RESPONSE_HEIGHT
_TENSION_DEMO_RESPONSE_LABELS = (
    ("D_L", "mm"), ("D_R", "mm"), ("dD_L", "mm"), ("dD_R", "mm"),
    ("row_L", "px"), ("row_R", "px"), ("col_L", "px"), ("col_R", "px"),
    ("logA_L", ""), ("logA_R", ""),
)


def _render_tension_strap_demo_frame(
    *,
    obs: dict[str, Any],
    visualization: dict[str, Any],
    control_diagnostics: dict[str, Any] | None,
    history: list[dict[str, Any]] | None,
) -> np.ndarray:
    """Render the live counterpart of ``render_tension_strap_demo.py``.

    It deliberately consumes only the already-public camera/tactile observation,
    cached public estimator state, and frozen response medians.
    """
    canvas = np.full(
        (_TENSION_DEMO_HEIGHT, _TENSION_DEMO_WIDTH, 3), (16, 19, 23), dtype=np.uint8
    )
    diagnostics = control_diagnostics if isinstance(control_diagnostics, dict) else {}
    latest = diagnostics.get("latest_estimate", {})
    latest = latest if isinstance(latest, dict) else {}
    estimate = _panel_optional_float(latest.get("estimated_tension_N"))
    stage_index = int(diagnostics.get("current_stage_index", 0) or 0)
    completed_stage_ids = diagnostics.get("completed_stage_ids", [])
    completed_stage_ids = (
        [str(item) for item in completed_stage_ids]
        if isinstance(completed_stage_ids, list)
        else []
    )
    hold_seconds = _panel_optional_float(
        diagnostics.get("current_stage_in_band_hold_seconds")
    ) or 0.0
    action_counts = diagnostics.get("stage_action_counts", [])
    action_count = (
        int(action_counts[stage_index])
        if isinstance(action_counts, list) and stage_index < len(action_counts)
        else 0
    )
    stage_label = "12N target hold" if stage_index == 0 else "18N target hold"
    stage_color = (125, 205, 255) if stage_index == 0 else (255, 205, 125)
    if stage_index == 1 and "tension_strap_hold_12n.v1" in completed_stage_ids:
        stage_label = "12N complete -> 18N target hold"
    if stage_index >= 2:
        stage_label, stage_color = "both target holds complete", (135, 225, 170)

    cv2.rectangle(
        canvas, (0, 0), (_TENSION_DEMO_WIDTH, _TENSION_DEMO_HEADER_HEIGHT), (10, 12, 15), -1
    )
    _tension_demo_text(
        canvas, "CaP-X / OpenTac runtime replay | public GelSight response", 22, 29,
        scale=0.78, thickness=2,
    )
    _tension_demo_text(canvas, stage_label, 22, 58, scale=0.65, color=stage_color, thickness=2)
    estimate_text = "waiting for marker-RGB estimate" if estimate is None else f"estimate {estimate:.3f} N"
    _tension_demo_text(
        canvas,
        f"{estimate_text} | current-stage hold {hold_seconds:.2f}s / 3.00s | control actions {action_count}",
        730,
        56,
        scale=0.44,
        color=(201, 208, 215),
    )

    head = _tension_demo_observation_image(obs, "head", "rgb")
    wrist = _tension_demo_observation_image(obs, "wrist", "rgb")
    _tension_demo_tile(canvas, head, 0, _TENSION_DEMO_HEADER_HEIGHT, 800, _TENSION_DEMO_SCENE_HEIGHT, "Head RGB", "runtime public observation")
    _tension_demo_tile(canvas, wrist, 800, _TENSION_DEMO_HEADER_HEIGHT, 800, _TENSION_DEMO_SCENE_HEIGHT, "Wrist RGB", "runtime public observation")

    last_sample = history[-1] if isinstance(history, list) and history else {}
    for index, hand in enumerate(("left", "right")):
        base_x = index * 800
        rgb = _tension_demo_tactile_image(obs, hand, "rgb")
        marker = _tension_demo_tactile_image(obs, hand, "rgb_marker")
        detail = (
            f"depth {_panel_optional_float(last_sample.get(f'{hand}_depth_mm')) or 0.0:.3f} mm | "
            f"disp {_panel_optional_float(last_sample.get(f'{hand}_marker_displacement_px')) or 0.0:.3f} px | "
            f"coherence {_panel_optional_float(last_sample.get(f'{hand}_marker_coherence')) or 0.0:.3f}"
        )
        tile_y = _TENSION_DEMO_HEADER_HEIGHT + _TENSION_DEMO_SCENE_HEIGHT
        _tension_demo_tile(canvas, rgb, base_x, tile_y, 400, _TENSION_DEMO_TACTILE_HEIGHT, f"{hand.title()} GelSight RGB", detail)
        _tension_demo_tile(canvas, marker, base_x + 400, tile_y, 400, _TENSION_DEMO_TACTILE_HEIGHT, f"{hand.title()} marker stream", "black dots are tracked marker locations")

    state = visualization if isinstance(visualization, dict) else {}
    current = state.get("current_values")
    stage_values = state.get("stage_values", {})
    stage_values = stage_values if isinstance(stage_values, dict) else {}
    _tension_demo_text(canvas, "Current public 10D response and frozen stage memory", 18, _TENSION_DEMO_RESPONSE_TOP + 18, scale=0.48, color=(225, 230, 236))
    card_y = _TENSION_DEMO_RESPONSE_TOP + 28
    card_width, card_height, gap = 517, _TENSION_DEMO_RESPONSE_HEIGHT - 34, 8
    current_title = (
        "Current rolling 10D"
        if state.get("source") == "rolling_preview"
        else "Current captured 10D"
    )
    cards = (
        (current_title, current, (221, 228, 237), bool(current)),
        ("12N target hold memory", stage_values.get("12N"), (117, 194, 255), stage_index == 0),
        ("18N target hold memory", stage_values.get("18N"), (255, 179, 117), stage_index == 1),
    )
    for index, (title, values, color, active) in enumerate(cards):
        _tension_demo_response_card(
            canvas, values, 16 + index * (card_width + gap), card_y, card_width, card_height,
            title, active=active, color=color,
        )

    chart_y = _TENSION_DEMO_CHART_TOP + 16
    chart_height = _TENSION_DEMO_HEIGHT - chart_y - 18
    chart_specs = (
        ("left_depth_mm", "Raw left indentation", "mm", (92, 210, 245)),
        ("right_depth_mm", "Raw right indentation", "mm", (97, 221, 148)),
        ("left_marker_displacement_px", "Raw left marker displacement", "px", (245, 181, 78)),
        ("right_marker_displacement_px", "Raw right marker displacement", "px", (241, 120, 133)),
    )
    for index, (key, title, unit, color) in enumerate(chart_specs):
        _tension_demo_plot(
            canvas, history or [], key, 16 + index * 392, chart_y,
            380 if index < 3 else 392, chart_height, title, unit, color, stage_index,
        )
    return np.ascontiguousarray(canvas)


def _tension_demo_observation_image(obs: dict[str, Any], camera: str, field: str) -> np.ndarray:
    record = obs.get("observation", {}).get(camera, {}) if isinstance(obs, dict) else {}
    image = record.get(field) if isinstance(record, dict) else None
    return _tension_demo_image_or_placeholder(image, f"{camera} {field} unavailable")


def _tension_demo_tactile_image(obs: dict[str, Any], hand: str, field: str) -> np.ndarray:
    tactile = obs.get("tactile", {}) if isinstance(obs, dict) else {}
    record = tactile.get(f"{hand}_tactile", tactile.get(hand, {})) if isinstance(tactile, dict) else {}
    image = record.get(field) if isinstance(record, dict) else None
    return _tension_demo_image_or_placeholder(image, f"{hand} {field} unavailable")


def _tension_demo_image_or_placeholder(image: Any, message: str) -> np.ndarray:
    if image is not None:
        try:
            return _as_uint8_rgb(image)
        except Exception:
            pass
    fallback = np.full((240, 320, 3), (20, 24, 28), dtype=np.uint8)
    _tension_demo_text(fallback, message, 20, 120, scale=0.48, color=(180, 190, 200))
    return fallback


def _tension_demo_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height, width, 3), (20, 24, 28), dtype=np.uint8)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _tension_demo_text(
    image: np.ndarray,
    value: str,
    x: int,
    y: int,
    *,
    scale: float = 0.5,
    color: tuple[int, int, int] = (242, 244, 246),
    thickness: int = 1,
) -> None:
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _tension_demo_tile(
    canvas: np.ndarray,
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    detail: str,
) -> None:
    canvas[y : y + height, x : x + width] = _tension_demo_fit(image, width, height)
    cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), (92, 101, 111), 1)
    cv2.rectangle(canvas, (x, y), (x + width, y + 25), (10, 12, 15), -1)
    _tension_demo_text(canvas, title, x + 8, y + 18, scale=0.47)
    _tension_demo_text(canvas, detail, x + 8, y + height - 10, scale=0.42, color=(180, 214, 255))


def _tension_demo_response_card(
    canvas: np.ndarray,
    values: Any,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    *,
    active: bool,
    color: tuple[int, int, int],
) -> None:
    values = values if isinstance(values, list) and len(values) == 10 else None
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (53, 65, 84) if active else (30, 35, 42), -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), color if active else (92, 101, 111), 2 if active else 1)
    _tension_demo_text(canvas, title, x + 10, y + 21, scale=0.45, color=color, thickness=2 if active else 1)
    _tension_demo_text(canvas, "ACTIVE WINDOW" if active else "REFERENCE", x + width - 132, y + 21, scale=0.34, color=(224, 228, 232))
    if values is None:
        _tension_demo_text(canvas, "waiting for valid 10D capture", x + 12, y + height // 2, scale=0.42, color=(170, 180, 190))
        return
    column_width = (width - 16) // 5
    for index, ((label, unit), value) in enumerate(zip(_TENSION_DEMO_RESPONSE_LABELS, values, strict=True)):
        row, column = divmod(index, 5)
        px, py = x + 8 + column * column_width, y + 48 + row * 42
        _tension_demo_text(canvas, label, px, py, scale=0.34, color=(190, 201, 212))
        _tension_demo_text(canvas, f"{float(value):.3f}{' ' + unit if unit else ''}", px, py + 18, scale=0.37, color=(247, 248, 249))
    _tension_demo_text(canvas, "10D: [normal static | normal dynamic | marker spatial]", x + 10, y + height - 8, scale=0.31, color=(190, 201, 212))


def _tension_demo_plot(
    canvas: np.ndarray,
    history: list[dict[str, Any]],
    key: str,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    unit: str,
    color: tuple[int, int, int],
    stage_index: int,
) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (33, 38, 43), -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (92, 101, 111), 1)
    background = (44, 63, 93) if stage_index == 0 else (90, 59, 42)
    cv2.rectangle(canvas, (x + 1, y + 1), (x + width - 1, y + height - 1), background, -1)
    values = [_panel_optional_float(row.get(key)) for row in history if isinstance(row, dict)]
    values = [value for value in values if value is not None]
    _tension_demo_text(canvas, title, x + 8, y + 19, scale=0.40)
    if len(values) < 2:
        _tension_demo_text(canvas, "waiting for public samples", x + 8, y + height - 7, scale=0.34, color=(190, 201, 212))
        return
    low, high = min(values), max(values)
    span = max(1e-6, high - low)
    low, high = low - 0.08 * span, high + 0.08 * span
    points = []
    for index, value in enumerate(values):
        px = x + 1 + int(index * (width - 2) / max(1, len(values) - 1))
        py = y + height - 18 - int((value - low) * (height - 42) / max(1e-6, high - low))
        points.append((px, py))
    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
    cv2.line(canvas, points[-1], (points[-1][0], y + height - 1), (255, 255, 255), 1, cv2.LINE_AA)
    _tension_demo_text(canvas, f"{low:.3g}-{high:.3g} {unit}", x + 8, y + height - 6, scale=0.32, color=(209, 215, 221))


def _panel_optional_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _safe_overlay_float(value: Any) -> float:
    """Format optional public memory values without breaking video rendering."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0


def _empty_public_probe_window() -> dict[str, Any]:
    """Fallback v3-compatible aggregate for an interrupted empty segment."""
    fields = ("depth_mm", "marker_displacement_px", "marker_coherence", "contact_area")
    return {
        "frame_count": 0,
        "start_step": None,
        "end_step": None,
        "bilateral_contact_ratio": 0.0,
        "left": {field: 0.0 for field in fields},
        "right": {field: 0.0 for field in fields},
        "noise": {
            hand: {
                field: {"value": 0.0, "mad": 0.0, "snr": 0.0}
                for field in fields
            }
            for hand in ("left", "right")
        },
        "gripper_qpos": 0.0,
    }


def _as_uint8_rgb(value: Any) -> np.ndarray:
    arr = _to_numpy(value)
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def _depth_visualization(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    valid = np.isfinite(depth) & (depth > 0.0)
    image = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return image
    low, high = np.percentile(depth[valid], [2.0, 98.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        image[valid] = 127
        return image
    normalized = (depth - float(low)) / float(high - low)
    image[valid] = np.clip(normalized[valid] * 255.0, 0.0, 255.0).astype(np.uint8)
    return image


def _nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _marker_motion_stats(marker: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(marker, dtype=np.float32)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    stats: dict[str, Any] = {}
    if arr.ndim >= 3 and arr.shape[-1] >= 2:
        pts = arr.reshape(-1, arr.shape[-1])[..., :2]
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        if pts.size:
            stats["marker_count"] = int(len(pts))
            stats["marker_centroid_xy"] = _jsonable(np.mean(pts, axis=0))
            stats["marker_spread_xy"] = _jsonable(np.std(pts, axis=0))
    if arr.ndim >= 4 and arr.shape[0] >= 2 and arr.shape[-1] >= 2:
        before = arr[0].reshape(-1, arr.shape[-1])[..., :2]
        after = arr[1].reshape(-1, arr.shape[-1])[..., :2]
        finite_mask = np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
        before = before[finite_mask]
        after = after[finite_mask]
        if before.size:
            disp = np.linalg.norm(after - before, axis=1)
            stats["marker_motion_mean_px"] = float(np.mean(disp))
            stats["marker_motion_max_px"] = float(np.max(disp))
            stats["marker_motion_centroid_delta_xy"] = _jsonable(np.mean(after - before, axis=0))
    return stats


def _balanced_difference(left: float, right: float) -> float:
    denom = abs(left) + abs(right)
    if denom <= 1e-9:
        return 0.0
    return float(abs(left - right) / denom)


def _flatten_timeline_record(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), item)
        elif isinstance(value, list):
            flat[prefix] = " ".join(str(v) for v in value)
        else:
            flat[prefix] = value

    visit("", record)
    return flat


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(ch in text for ch in {",", "\"", "\n"}):
        text = "\"" + text.replace("\"", "\"\"") + "\""
    return text


def _resize_rgb(value: Any, width: int, height: int) -> np.ndarray:
    arr = _as_uint8_rgb(value)
    if arr.shape[0] == height and arr.shape[1] == width:
        return arr
    image = Image.fromarray(arr)
    image = image.resize((width, height), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0


def _is_pose_like(value: Any) -> bool:
    return _is_sequence(value) and len(value) >= 7


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
