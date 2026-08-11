"""UniVTAC low-level environment adapter for CaP-X code execution."""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from capx.envs.base import BaseEnv
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
        device: str | None = None,
        api_configs: dict[str, Any] | None = None,
        expose_actor_pose: bool = True,
        max_steps: int | None = None,
        video_size: tuple[int, int] = (960, 320),
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
        self.device_override = device
        self.api_configs = api_configs or {}
        self.expose_actor_pose = bool(expose_actor_pose)
        self.max_steps = int(max_steps) if max_steps is not None else 999999
        self.video_size = tuple(video_size)
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
        self._tactile_buffer = UniVTACTactileBuffer(maxlen=tactile_buffer_size)
        self.lift_success_height_delta = float(lift_success_height_delta)
        self.lift_success_require_contact = bool(lift_success_require_contact)
        self._initial_lift_object_z: float | None = None
        self._last_recorded_tactile_step: int | None = None
        self._last_recorded_video_step: int | None = None
        self._video_record_failures = 0
        self._last_action_result: dict[str, Any] = {}
        self._sim_step_count = 0
        self._start_time = time.time()
        self._debug_records: list[dict[str, Any]] = []
        self._tactile_gripper_trace: list[dict[str, Any]] = []
        self._pre_move_tactile_timeline: list[dict[str, Any]] = []
        self._pre_move_tactile_last_error: str | None = None

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
        actual_seed = self.seed_base + int(seed if seed is not None else trial)
        if not self._record_frames:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
        self._tactile_buffer.clear()
        self._last_recorded_tactile_step = None
        self._last_recorded_video_step = None
        self._video_record_failures = 0
        self._last_action_result = {}
        self._sim_step_count = 0
        self._start_time = time.time()
        self._debug_records.clear()
        self._tactile_gripper_trace.clear()
        self._pre_move_tactile_timeline.clear()
        self._pre_move_tactile_last_error = None

        print(
            f"[capx-univtac] reset begin trial={trial} seed={actual_seed}",
            flush=True,
        )
        self._task.reset(seed=actual_seed, instructions=self._instructions())
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
        return bool(self._task.check_success())

    def _active_object_height(self, object_name: str) -> float | None:
        actor = getattr(self._task, str(object_name), None)
        if actor is None:
            return None
        try:
            return float(actor.get_pose().p[2])
        except Exception:
            return None

    def take_action(self, action: np.ndarray | torch.Tensor | list[float], *, action_type: str) -> dict[str, Any]:
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

        try:
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
        return result

    def get_public_grasp_pose(
        self,
        object_name: str,
        *,
        grasp_height: float = 0.04,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the public pose used by the matching native grasp approach."""
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
        velocity = (position - current_qpos) / max(sim_dt, 1e-8)
        action_count_before = self.get_action_count()
        robot_manager.set_gripper(position, velocity, force=False)
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
        poses = self.get_actor_poses()
        if name not in poses:
            return {"ok": False, "name": name, "message": f"object '{name}' not found"}
        pose = poses[name]
        return {"ok": True, "name": name, "pose": pose}

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
            "early_stop": bool(self._task.check_early_stop()),
        }

    def current_raw_observation(self) -> dict[str, Any]:
        if not self._current_obs:
            self._current_obs = self._lightweight_raw_observation()
        return self._current_obs

    def reset_tactile_buffer(self) -> None:
        self._tactile_buffer.clear()
        self._last_recorded_tactile_step = None

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
        capture_initial_frame: bool = True,
    ) -> None:
        self._record_frames = bool(enabled)
        self._record_wrist_camera = bool(wrist_camera)
        if clear:
            self._frame_buffer.clear()
            self._wrist_frame_buffer.clear()
            self._last_recorded_video_step = None
            self._video_record_failures = 0
        if enabled and capture_initial_frame:
            self._record_frame(force=True)

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._wrist_frame_buffer]
        if clear:
            self._wrist_frame_buffer.clear()
        return frames

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._wrist_frame_buffer[start:end]]

    def export_debug_artifacts(self, output_dir: str | os.PathLike[str]) -> str | None:
        """Write private UniVTAC diagnostics for audit, never for LLM prompts."""
        if not self._debug_records and not self._tactile_gripper_trace:
            return None
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        debug_path = output_path / "univtac_debug.json"
        if self._debug_records:
            payload = {
                "task": self.task_name,
                "task_config": self.task_config_name,
                "metadata": _jsonable(getattr(self._task, "metadata", {})),
                "records": self._debug_records,
            }
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            print(
                f"[capx-univtac] saved private debug diagnostics to {debug_path}",
                flush=True,
            )
        if self._tactile_gripper_trace:
            trace_path = output_path / "tactile_gripper_trace.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(self._tactile_gripper_trace, f, indent=2, sort_keys=True)
            print(f"[capx-univtac] saved tactile gripper trace to {trace_path}", flush=True)
        self._export_pre_move_tactile_timeline(output_path)
        return str(debug_path if self._debug_records else trace_path)

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        obs = self._read_native_observation(include_camera=True, include_tactile=True)
        self._current_obs = obs
        return self._compose_frame(obs)

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
        env_cfg.scene.num_envs = 1
        if self.device_override:
            env_cfg.sim.device = self.device_override
        self._task = task_module.Task(env_cfg, mode="eval")
        self._install_task_runtime_patches()

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
            self._record_pre_move_tactile_step()
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
                tactile_data_types=["rgb", "rgb_marker"],
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
        self._frame_buffer.append(self._compose_frame(obs))
        self._last_recorded_video_step = step
        if self._record_wrist_camera:
            wrist = obs.get("observation", {}).get("wrist", {}).get("rgb")
            if wrist is not None:
                self._wrist_frame_buffer.append(_as_uint8_rgb(wrist))

    def _public_grasp_actor(self, object_name: str):
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
        return np.ascontiguousarray(frame)

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
