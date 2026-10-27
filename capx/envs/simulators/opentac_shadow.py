"""CaP-X adapter for OpenTacBench's Franka-mounted Shadow Hand piano.

Import the OpenTacBench task only after the Isaac Kit AppLauncher has started.
Native scoring and force measurements stay in the evaluator-side event log.
"""

from __future__ import annotations

import json
import hashlib
import os
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from capx.envs.base import BaseEnv

ACTIVE_SHADOW_ENVS: set[ShadowPianoLowLevelEnv] = set()


class ShadowPianoLowLevelEnv(BaseEnv):
    policy_controls_termination = True
    record_video_during_reset = False

    def __init__(
        self,
        univtac_root: str,
        piano_notes: list[str | int] | None = None,
        piano_dynamics: list[str] | None = None,
        piano_hold_seconds: float | list[float] = 0.12,
        piano_require_dynamics: bool = True,
        piano_continue_on_note_failure: bool = False,
        seed_base: int = 0,
        device: str = "cuda:0",
        tactile_image_width: int = 320,
        action_repeat: int = 8,
        video_frame_stride: int = 3,
        video_width: int = 640,
        task_factory: Callable[["ShadowPianoLowLevelEnv"], Any] | None = None,
        output_dir: str | None = None,
        visual_key_mode: bool = False,
        mirror_head_view: bool = False,
        auto_clear_keyboard_for_visual: bool = False,
        snap_visual_key_pixels: bool = False,
        digit_piano_config: str | None = None,
        digit_force_sidecar: str | None = None,
        private_audit_root: str | None = None,
        piano_note_count: int = 6,
        task_seed: int | None = None,
        endpoint_calibration: bool = False,
        keyboard_calibration_path: str | None = None,
        public_episode_recording: bool = False,
        require_public_release: bool = False,
        third_party_video: bool = False,
        tactile_skill_memory_path: str | None = None,
        tactile_skill_memory_mode: str = 'collect',
    ) -> None:
        super().__init__()
        self.univtac_root = Path(univtac_root).expanduser().resolve()
        self.piano_notes = list(piano_notes) if piano_notes is not None else None
        self.piano_dynamics = list(piano_dynamics) if piano_dynamics is not None else None
        self.piano_note_count = int(piano_note_count)
        self.task_seed = int(task_seed) if task_seed is not None else None
        self.endpoint_calibration = bool(endpoint_calibration)
        self.keyboard_calibration_path = Path(keyboard_calibration_path).expanduser().resolve() if keyboard_calibration_path else None
        self.public_episode_recording = bool(public_episode_recording)
        self.require_public_release = bool(require_public_release)
        self.third_party_video = bool(third_party_video)
        self.tactile_skill_memory_path = Path(tactile_skill_memory_path) if tactile_skill_memory_path else None
        self.tactile_skill_memory_mode = tactile_skill_memory_mode
        self._tactile_skill_memory = None
        self._release_ready_since = None
        self._keyboard_calibration = None
        self._cached_key_targets = {}
        self._keyboard_cache_hit = False
        self._clear_action_steps = 0
        self.piano_hold_seconds = piano_hold_seconds
        self.piano_require_dynamics = bool(piano_require_dynamics)
        self.piano_continue_on_note_failure = bool(piano_continue_on_note_failure)
        self.seed_base = int(seed_base)
        self.device = device
        self.tactile_image_width = int(tactile_image_width)
        if isinstance(action_repeat, bool) or not isinstance(action_repeat, int) or action_repeat < 1:
            raise ValueError("action_repeat must be a positive integer")
        self.action_repeat = action_repeat
        self.video_frame_stride = max(1, int(video_frame_stride))
        self.video_width = int(video_width)
        self.visual_key_mode = bool(visual_key_mode)
        self.mirror_head_view = bool(mirror_head_view)
        self.auto_clear_keyboard_for_visual = bool(auto_clear_keyboard_for_visual)
        self.snap_visual_key_pixels = bool(snap_visual_key_pixels)
        self._visual_key_map_ready = False
        if self.mirror_head_view and not self.visual_key_mode:
            raise ValueError("Mirrored HEAD view requires visual key mode")
        if self.auto_clear_keyboard_for_visual and not self.visual_key_mode:
            raise ValueError("Automatic keyboard clearing requires visual key mode")
        if self.snap_visual_key_pixels and not self.visual_key_mode:
            raise ValueError("Visual key pixel snapping requires visual key mode")
        self.digit_piano_config = Path(digit_piano_config).expanduser().resolve() if digit_piano_config else None
        self.digit_force_sidecar = Path(digit_force_sidecar).expanduser().resolve() if digit_force_sidecar else None
        self.private_audit_root = Path(private_audit_root or "/mnt/data/ljz/opentacbench/capx-private-audits/shadow_piano_visual_force_v1").expanduser().resolve()
        if self.visual_key_mode and (self.digit_piano_config is None or self.digit_force_sidecar is None):
            raise ValueError("Visual Shadow smoke requires the matched DIGIT config and force sidecar")
        self._output_dir = Path(output_dir or os.environ.get("CAPX_OUTPUT_DIR", "")) if (output_dir or os.environ.get("CAPX_OUTPUT_DIR")) else None
        self._task = task_factory(self) if task_factory else self._create_native_task()
        self._third_camera = getattr(self._task, 'capx_third_camera', None)
        if self._third_camera is not None:
            import torch
            self._third_camera.set_world_poses_from_view(
                torch.tensor([[-.58, .85, .66]],device=self._task.sim.device),
                torch.tensor([[.12, 0., .095]],device=self._task.sim.device))
        self._force_estimator = None
        if self.visual_key_mode:
            from capx.integrations.opentac_shadow_force import PublicDigitKeyForce
            self._force_estimator = PublicDigitKeyForce(self.digit_force_sidecar, self.digit_piano_config)
        self._configure_camera()
        native_cfg = getattr(self._task, "cfg", None)
        physics_dt = float(getattr(native_cfg, "physics_dt", 1 / 240))
        native_repeat = int(getattr(native_cfg, "action_repeat", self.action_repeat))
        if self._force_estimator is not None and self._force_estimator.uses_area_rate:
            from capx.integrations.opentac_shadow_force import AREA_RATE_HISTORY
            if (not np.isclose(physics_dt, AREA_RATE_HISTORY["physics_dt_seconds"]) or
                    AREA_RATE_HISTORY["window_physics_steps"] % native_repeat):
                raise ValueError("DIGIT area-rate calibration needs 240 Hz physics and an action repeat dividing 8")
            print(f"[capx-shadow] force calibration={self._force_estimator.sidecar['schema']} "
                  "features=5 area_rate_window_physics_steps=8", flush=True)
        self.video_fps = 1.0 / (physics_dt * native_repeat * self.video_frame_stride)
        self._observation: dict[str, Any] | None = None
        self._native_info: dict[str, Any] = {}
        self._native_done = False
        self._policy_end_submitted = False
        self._policy_end_verdict = False
        self._sim_step_count = 0
        self._public_attempts_finalized = 0
        self._last_native_notes_attempted = 0
        self._last_private_piano_record = 0
        self._last_private_score_event = 0
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._third_frame_buffer: list[np.ndarray] = []
        self._video_physics_steps: list[int] = []
        self._private_audit_dir: Path | None = None
        self._private_audit_streams: dict[str, Any] = {}
        self._action = np.zeros(4, dtype=np.float32)
        self.max_steps = 100000
        self._closed = False
        ACTIVE_SHADOW_ENVS.add(self)

    def _create_native_task(self):
        from env.tasks.arm_shadow_piano import Task, TaskCfg

        if self.third_party_video:
            class RecordedTask(Task):
                def _create_keyboard(task):
                    super()._create_keyboard()
                    # Add a rendering-only sensor before the native first reset.
                    from isaaclab.sensors import Camera, CameraCfg
                    from isaaclab import sim as sim_utils
                    task.capx_third_camera = Camera(CameraCfg(
                        prim_path='/World/CapXThirdPartyCamera', width=960, height=640,
                        data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(
                            focal_length=24.,horizontal_aperture=36.,clipping_range=(.01,10.))))
            Task = RecordedTask

        cfg = TaskCfg(
            device=self.device,
            tactile=True,
            tactile_sensor="digit",
            tactile_image_width=self.tactile_image_width,
            action_repeat=self.action_repeat,
            piano_notes=self.piano_notes,
            piano_dynamics=self.piano_dynamics,
            piano_note_count=self.piano_note_count,
            piano_hold_seconds=self.piano_hold_seconds,
            piano_require_dynamics=self.piano_require_dynamics,
            piano_continue_on_note_failure=self.piano_continue_on_note_failure,
            piano_seed=self.task_seed if self.task_seed is not None else self.seed_base,
            record_video=True,
        )
        if not self.visual_key_mode:
            return Task(cfg=cfg, render_mode="rgb_array")
        import env.tasks._digit_tactile as native_digit
        from capx.integrations.opentac_shadow_runtime_bridge import make_runtime_digit_bridge

        native_class = native_digit.DigitTactile
        native_digit.DigitTactile = make_runtime_digit_bridge(
            native_class, opentac_root=self.univtac_root,
            piano_config=self.digit_piano_config,
            output_dir=self._output_dir,
        )
        try:
            task = Task(cfg=cfg, render_mode="rgb_array")
        finally:
            native_digit.DigitTactile = native_class
        expected = hashlib.sha256(self.digit_piano_config.read_bytes()).hexdigest()
        if task.tactile._geometry.get("config_path") != str(self.digit_piano_config):
            task.close()
            raise RuntimeError("Shadow task used a different TACTO renderer config")
        print(f"[capx-shadow] TACTO config sha256={expected} version={task.tactile._geometry.get('tacto_version')}", flush=True)
        return task

    @property
    def task(self):
        return self._task

    @property
    def visual_key_map_ready(self) -> bool:
        """Whether this episode has an accepted visual key map."""
        return self._visual_key_map_ready

    def get_keyboard_endpoints(self) -> dict[str, Any]:
        """Expose only the static A0/C8 white-key centers and their HEAD pixels."""
        if not self.visual_key_mode:
            raise RuntimeError("Visual keyboard endpoints are not enabled")
        from envs._piano import keyboard_layout
        from capx.integrations.opentac_shadow_vision import head_world_to_display_pixel

        layout = keyboard_layout()
        anchors = {}
        for name, key in (("A0", layout[0]), ("C8", layout[-1])):
            if key.black:
                raise RuntimeError("Keyboard endpoint must be a white key")
            x, y, z = key.position
            point = (float(x), float(y), float(z + key.size[2]))
            anchors[name] = {
                "midi": int(key.midi),
                "world_key_top_center_m": list(point),
                "display_image_center_px": head_world_to_display_pixel(
                    point, mirrored=self.mirror_head_view,
                ),
            }
        white_count = sum(not key.black for key in layout)
        intervals = white_count - 1
        result = {
            "frame": "fixed_HEAD_display_image",
            "mirrored": self.mirror_head_view,
            "display_image_size_px": [960, 662],
            "camera_picture_origin_px": [0, 22],
            "white_key_count": white_count,
            "white_center_intervals": intervals,
            "white_center_spacing_m": (
                anchors["C8"]["world_key_top_center_m"][1]
                - anchors["A0"]["world_key_top_center_m"][1]
            ) / intervals,
            "A0": anchors["A0"],
            "C8": anchors["C8"],
        }
        if getattr(self, "endpoint_calibration", False):
            camera = self._task.camera
            intrinsics = camera.data.intrinsic_matrices[0].detach().cpu().numpy()
            from capx.integrations.opentac_shadow_vision import HEAD_FOCAL_PX, HEAD_WIDTH, HEAD_HEIGHT
            expected_intrinsics = np.array([[HEAD_FOCAL_PX, 0, HEAD_WIDTH / 2], [0, HEAD_FOCAL_PX, HEAD_HEIGHT / 2], [0, 0, 1.]])
            if camera.cfg.width != HEAD_WIDTH or camera.cfg.height != HEAD_HEIGHT or not np.allclose(intrinsics, expected_intrinsics, atol=2.):
                raise ValueError("HEAD camera differs from endpoint-calibration geometry")
            result["observed_camera_intrinsics"] = np.round(intrinsics, 7).tolist()
            for field in ("pos_w", "quat_w_world"):
                value = getattr(camera.data, field, None)
                if value is not None:
                    result["camera_" + field] = np.round(value[0].detach().cpu().numpy(), 7).tolist()
            result.update(calibration_ready=self._visual_key_map_ready,
                          calibration_id=self._keyboard_calibration.get("calibration_id") if self._keyboard_calibration else None,
                          mapped_keys=self._cached_key_targets,
                          localization_mode="public_endpoint_geometry")
        return result

    def _configure_camera(self) -> None:
        if not hasattr(self._task, "camera"):
            return
        if self.visual_key_mode:
            from capx.integrations.opentac_shadow_vision import HEAD_EYE, HEAD_LOOK_AT
            self._set_camera_pose(HEAD_EYE.tolist(), HEAD_LOOK_AT.tolist())
        else:
            target_y = self._task.layout[self._task.target].position[1]
            self._set_camera_pose([-.58, .85, .66], [.12, target_y, .095])

    def _set_camera_pose(self, eye, target) -> None:
        import torch
        self._task.camera.set_world_poses_from_view(
            torch.tensor([eye], dtype=torch.float32, device=self._task.sim.device),
            torch.tensor([target], dtype=torch.float32, device=self._task.sim.device),
        )

    def _write_native_event(self, kind: str) -> None:
        if self._output_dir is None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        state = np.asarray(self._observation["state"]) if self._observation is not None else None
        requested_midi = self._native_info.get("piano_sequence", {}).get("midi", [])
        key_angles = ({str(midi): float(state[6 + midi - 21]) for midi in requested_midi}
                      if state is not None and state.shape == (111,) else {})
        row = {"kind": kind, "action_step": self._sim_step_count,
               "action_target": self._action.tolist(), "key_angles_rad": key_angles,
               "native_done": self._native_done, "native_info": self._native_info}
        if self.visual_key_mode:
            self._append_private_audit("shadow_native_events.jsonl", row)
        else:
            with (self._output_dir / "shadow_native_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")

    def _save_tactile_snapshot(self, filename: str) -> None:
        if self._output_dir is None:
            return
        from PIL import Image
        self._output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.get_index_tactile_rgb()).save(self._output_dir / filename)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._tactile_skill_memory = None
        explicit_seed = getattr(self, "task_seed", None)
        actual_seed = explicit_seed if explicit_seed is not None else (self.seed_base if seed is None else self.seed_base + int(seed))
        self._visual_key_map_ready = False
        self._keyboard_calibration = None
        self._cached_key_targets = {}
        self._keyboard_cache_hit = False
        self._clear_action_steps = 0
        estimator = getattr(self, "_force_estimator", None)
        if estimator is not None:
            estimator.reset_history()
        self._observation, self._native_info = self._task.reset(seed=actual_seed)
        sequence = self._native_info.get("piano_sequence", {})
        if sequence:
            self.piano_notes = list(sequence["notes"])
            self.piano_dynamics = list(sequence["dynamics"])
            self._public_sequence = {k: sequence[k] for k in ("midi", "notes", "dynamics", "hold_seconds")}
        self._native_done = False
        self._policy_end_submitted = False
        self._policy_end_verdict = False
        self._sim_step_count = 0
        self._public_attempts_finalized = 0
        self._release_ready_since = None
        self._last_native_notes_attempted = 0
        self._last_private_piano_record = 0
        self._last_private_score_event = 0
        self._action = np.zeros(4, dtype=np.float32)
        self._frame_buffer.clear()
        if hasattr(self, '_third_frame_buffer'):
            self._third_frame_buffer.clear()
        self._video_physics_steps.clear()
        self._sample_public_force_depth()
        if hasattr(self._task, "camera"):
            # Match the official piano CLI's RTX camera warmup before capture.
            for _ in range(16):
                self._task.sim.render()
                self._task.camera.update(self._task.cfg.physics_dt, force_recompute=True)
        self._write_native_event("reset")
        if self.visual_key_mode:
            self._append_private_audit(
                "shadow_visual_key_maps.jsonl",
                {"kind": "reset", "ready": 0, "action_step": 0, "seed": actual_seed},
            )
            print("[capx-shadow] visual_key_map_ready=0 reset", flush=True)
            self._write_private_force_audit("reset")
            self._write_private_piano_timeline()
        self._save_tactile_snapshot("shadow_digit_initial.png")
        if getattr(self, "endpoint_calibration", False) and self.keyboard_calibration_path and self.keyboard_calibration_path.is_file():
            try:
                from capx.integrations.opentac_shadow_keyboard import validate_calibration, targets_from_calibration
                record = json.loads(self.keyboard_calibration_path.read_text(encoding="utf-8"))
                endpoints = self.get_keyboard_endpoints()
                validate_calibration(record, endpoints)
                self._cached_key_targets = targets_from_calibration(record, endpoints, self.piano_notes)
                self._keyboard_calibration = record
                self._visual_key_map_ready = True
                self._keyboard_cache_hit = True
                print(f"[capx-shadow] keyboard cache hit id={record['calibration_id']} initial image_input=0", flush=True)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                print(f"[capx-shadow] keyboard cache invalid; HEAD required: {exc}", flush=True)
        if self.auto_clear_keyboard_for_visual and not self._keyboard_cache_hit:
            self._clear_keyboard_for_visual()
            self._clear_action_steps = self._sim_step_count
        self._save_public_episode(actual_seed)
        return self.get_observation(), {}

    def public_task_context(self) -> str:
        if not self.endpoint_calibration:
            return ""
        return ("\n\nCURRENT PUBLIC REQUEST (authoritative for this episode): "
                + json.dumps(self._public_sequence) + ". Attempt exactly len(notes) entries in order.\n"
                + "PUBLIC KEYBOARD CALIBRATION: " + json.dumps(self.get_keyboard_endpoints()))

    def _save_public_episode(self, seed: int | None = None) -> None:
        if not getattr(self, "public_episode_recording", False) or self._output_dir is None:
            return
        from capx.integrations.opentac_shadow_keyboard import atomic_json
        if seed is not None:
            self._actual_task_seed = seed
        atomic_json(self._output_dir / "shadow_public_episode.json", {
            "schema": "shadow_piano_public_episode.v1", "seed": self._actual_task_seed,
            "sequence": self._public_sequence, "keyboard_cache_hit": self._keyboard_cache_hit,
            "calibration_id": self._keyboard_calibration.get("calibration_id") if self._keyboard_calibration else None,
            "clear_action_steps": self._clear_action_steps,
            "action_repeat": self.action_repeat, "localization_mode": "public_endpoint_geometry"})

    def _write_public_event(self, kind: str, **fields) -> None:
        if not getattr(self, "public_episode_recording", False) or self._output_dir is None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        row = {"kind": kind, "action_step": self._sim_step_count,
               "physics_step": int(self._task.step_count), **fields}
        with (self._output_dir / "shadow_public_observations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def record_public_code_result(self, stdout: str, stderr: str) -> None:
        import re
        from capx.integrations.opentac_shadow_skills import _PRIVATE_MARKERS
        lines = [line[:8192] if line.startswith('PIANO_POLICY_STATE ') else line[:1200] for line in stdout.splitlines()
                 if line.startswith(("PIANO_NOTE", "PIANO_STOP", "PIANO_ARRIVAL", "PIANO_POLICY_STATE", "CAPX_VISUAL_CHECKPOINT"))]
        errors = [line[:800] for line in stderr.splitlines()
                  if re.match(r"^[A-Za-z]+(?:Error|Exception):", line)]
        safe = [line for line in lines + errors if not any(m in line.lower() for m in _PRIVATE_MARKERS)]
        self._write_public_event("code_execution", public_messages=safe[-24:])

    def _clear_keyboard_for_visual(self) -> None:
        """Move the open index away from the keyboard before the sole initial image."""
        current_y = float(self.get_observation()["hand_state"]["wrist_y_m"])
        # Preserve the old 8-tick clear-view motion duration at other action
        # frequencies. The arm responds in physical time, not action count.
        repeat = int(getattr(self, "action_repeat", 8))
        for y, old_action_steps in ((current_y, 30), (.24, 40), (.24, 40), (.24, 40)):
            remaining = (old_action_steps * 8 + repeat - 1) // repeat
            while remaining:
                batch = min(remaining, 240)
                result = self.command([0., y, .05, 0.], batch)
                if result["native_done"]:
                    raise RuntimeError("Native Shadow episode ended while clearing the keyboard view")
                remaining -= batch
            if y == .24 and abs(
                float(self.get_observation()["hand_state"]["wrist_y_m"]) - .24
            ) <= .025:
                print(f"[capx-shadow] keyboard clear at action_step={self._sim_step_count}", flush=True)
                return
        raise RuntimeError("Shadow wrist did not reach the clear-view position")

    def _validate_action(self, action: Any) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (4,) or not np.isfinite(value).all():
            raise ValueError("Shadow action must contain four finite wrist/index targets")
        bounds = self._task.action_space
        if np.any(value < bounds.low) or np.any(value > bounds.high):
            raise ValueError(f"Shadow action outside native bounds: {value.tolist()}")
        return value

    def _require_xy_arrival_before_press(self, target: np.ndarray) -> None:
        """Keep a moving index clear of adjacent keys in visual-key mode."""
        if not self.visual_key_mode or not self._visual_key_map_ready:
            return
        # Hovering with the index extended remains available while the arm
        # travels. Lowering or curling requires measured, settled wrist X/Y.
        if float(target[2]) >= .035 and float(target[3]) <= .05:
            return
        hand = self.get_observation()["hand_state"]
        actual_xy = np.array([hand["wrist_x_offset_m"], hand["wrist_y_m"]], dtype=float)
        target_xy = np.asarray(target[:2], dtype=float)
        error_m = float(np.linalg.norm(actual_xy - target_xy))
        y_speed_m_s = float(hand["wrist_and_index_velocity"][0])
        if not np.isfinite(error_m) or not np.isfinite(y_speed_m_s):
            raise ValueError("Shadow wrist XY arrival state is not finite")
        if error_m > .002 or abs(y_speed_m_s) > .015:
            raise ValueError(
                "Shadow wrist XY has not settled before lowering/flexion: "
                f"target={target_xy.tolist()}, actual={actual_xy.tolist()}, "
                f"error_m={error_m:.6f}, y_speed_m_s={y_speed_m_s:.6f}; "
                "hold the extended index at hover and recheck get_hand_state()"
            )

    def step(self, action):
        if self._policy_end_submitted or self._native_done:
            raise RuntimeError("Native piano episode has ended; no further action is allowed")
        value = self._validate_action(action)
        self._require_xy_arrival_before_press(value)
        self._observation, reward, terminated, truncated, self._native_info = self._task.step(value)
        self._action = value.copy()
        self._sim_step_count += 1
        self._native_done = bool(terminated or truncated)
        if getattr(self, 'require_public_release', False):
            hand = self.get_observation()['hand_state']
            ready = abs(hand['index_flexion_rad']) <= .02 and abs(hand['wrist_z_m']-.04) <= .002
            if ready:
                if self._release_ready_since is None:
                    self._release_ready_since = int(self._task.step_count)
            else:
                self._release_ready_since = None
        self._sample_public_force_depth()
        if getattr(self, '_tactile_skill_memory', None) is not None:
            self._tactile_skill_memory.observe_action(self._action.tolist(), self._sim_step_count)
        self._write_native_event("action")
        if self.visual_key_mode:
            self._write_private_force_audit("action")
            self._write_private_piano_timeline()
        self._save_attempt_snapshots()
        self._write_public_event("action", target=self._action.tolist(), hand_state=self.get_observation()["hand_state"], native_done=self._native_done)
        if self._native_done:
            self._save_tactile_snapshot("shadow_digit_terminal.png")
        if self._record_frames and self._sim_step_count % self.video_frame_stride == 0:
            self._capture_frame()
        # Keep native reward, success/failure and measured forces out of generated code.
        return self.get_observation(), 0.0, self._native_done, False, {"native_done": self._native_done}

    def command(self, action: Any, steps: int = 1) -> dict[str, Any]:
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 240:
            raise ValueError("steps must be an integer from 1 to 240")
        value = self._validate_action(action)
        completed = 0
        for _ in range(steps):
            self.step(value)
            completed += 1
            if self._native_done:
                break
        return {"ok": True, "steps_executed": completed, "native_done": self._native_done}

    def wait_steps(self, steps: int) -> dict[str, Any]:
        return self.command(self._action, steps)

    def _save_attempt_snapshots(self) -> None:
        attempted = int(self._native_info.get("notes_attempted",
                                              self._native_info.get("notes_completed", 0)))
        for index in range(self._last_native_notes_attempted + 1, attempted + 1):
            self._save_tactile_snapshot(f"shadow_digit_note_{index:02d}.png")
        self._last_native_notes_attempted = attempted

    def finish_piano_note_attempt(self, index: int) -> dict[str, Any]:
        """Advance a released attempt without exposing native scoring to code."""
        if not self.piano_continue_on_note_failure or not self.visual_key_mode:
            raise RuntimeError("Per-note continuation is disabled")
        if self._policy_end_submitted:
            raise RuntimeError("END has already been submitted")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self.piano_notes):
            raise ValueError("Invalid piano attempt index")
        if index != self._public_attempts_finalized:
            raise ValueError("Piano attempts must be finalized once in sequence")
        if getattr(self, 'require_public_release', False) and not self._native_done:
            since = self._release_ready_since
            elapsed = (int(self._task.step_count)-since)/240 if since is not None else 0.
            if elapsed < .12:
                self._write_public_event('release_rejected', index=index,
                    hand_state=self.get_observation()['hand_state'],continuous_release_seconds=elapsed)
                raise ValueError('Public release has not completed: actual index must be <=0.02 rad '
                    'and wrist Z within 0.002 m of 0.04 m continuously for >=0.12 s; keep advancing bounded release actions')
        self._native_info = self._task.finish_piano_note_attempt(index)
        self._public_attempts_finalized += 1
        if getattr(self, '_tactile_skill_memory', None) is not None:
            self._tactile_skill_memory.next_attempt()
        self._write_public_event("attempt_finalized", index=index,
            release_verified=bool(getattr(self, 'require_public_release', False)))
        self._release_ready_since = None
        self._observation = self._task._observation()
        self._native_done = bool(self._task._done)
        self._write_native_event("attempt_end")
        self._write_private_piano_timeline()
        self._save_attempt_snapshots()
        if self._native_done:
            self._save_tactile_snapshot("shadow_digit_terminal.png")
        return {"ok": True, "native_done": self._native_done}

    def get_observation(self) -> dict[str, Any]:
        if self._observation is None:
            return {}
        state = np.asarray(self._observation["state"], dtype=np.float32)
        if state.shape != (111,):
            raise RuntimeError(f"Unexpected native Shadow state shape: {state.shape}")
        native_action_state = np.asarray(self._task._action_state(), dtype=np.float32)
        if native_action_state.shape != (4,):
            raise RuntimeError(f"Unexpected native Shadow action-state shape: {native_action_state.shape}")
        public = {"hand_state": {"wrist_x_offset_m": float(native_action_state[0]),
                               "wrist_y_m": float(state[0]), "wrist_z_m": float(state[1]),
                               "index_flexion_rad": float(state[2]), "wrist_and_index_velocity": state[3:6].tolist(),
                               "arm_joint_position_rad": state[97:104].tolist(),
                               "arm_joint_velocity_rad_s": state[104:111].tolist()}}
        if self.visual_key_mode:
            public["hand_state"]["visual_key_map_ready"] = int(self._visual_key_map_ready)
        if not self.visual_key_mode:
            public["key_angles_rad"] = state[6:94].tolist()
        return public

    def _write_private_force_audit(self, kind: str) -> None:
        if self._output_dir is None or not hasattr(self._task, "tactile"):
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        forces = np.asarray(self._task.tactile.key_forces(), float)
        if forces.shape != (88,) or not np.isfinite(forces).all():
            raise RuntimeError("Invalid private key-force audit")
        row = {"kind": kind, "action_step": self._sim_step_count,
               "physics_step": int(getattr(self._task, "step_count", 0)),
               "key_contact_force_N": forces.tolist()}
        self._append_private_audit("shadow_private_force_audit.jsonl", row)

    def _write_private_piano_timeline(self) -> None:
        """Copy native 240 Hz annotations and score events without altering them."""
        records = self._task.piano_annotations.records
        for row in records[self._last_private_piano_record:]:
            self._append_private_audit("shadow_piano_physics_trace.jsonl", dict(row))
        self._last_private_piano_record = len(records)
        events = self._task.score.events
        for event in events[self._last_private_score_event:]:
            self._append_private_audit("shadow_piano_score_events.jsonl", dict(event))
        self._last_private_score_event = len(events)

    def locate_visual_keys(self, pixels: dict[str, dict]) -> dict[str, dict]:
        if not self.visual_key_mode:
            raise RuntimeError("Visual key location is not enabled")
        if getattr(self, "endpoint_calibration", False) and isinstance(pixels, dict) and set(pixels) == {"A0", "C8"}:
            from capx.integrations.opentac_shadow_keyboard import make_calibration, targets_from_calibration, atomic_json
            endpoints = self.get_keyboard_endpoints()
            record = make_calibration(pixels, endpoints, self._output_dir.name if self._output_dir else "shadow")
            targets = targets_from_calibration(record, endpoints, self.piano_notes)
            if self.keyboard_calibration_path:
                atomic_json(self.keyboard_calibration_path, record)
            self._keyboard_calibration = record
            self._cached_key_targets = targets
            self._visual_key_map_ready = True
            self._write_public_event("endpoint_calibration", calibration_id=record["calibration_id"])
            self._save_public_episode()
            print(f"[capx-shadow] visual_key_map_ready=1 endpoint calibration id={record['calibration_id']}", flush=True)
            return targets
        from capx.integrations.opentac_shadow_vision import (
            HEAD_FOCAL_PX, HEAD_HEIGHT, HEAD_WIDTH, locate_key_pixels,
            snap_key_pixels_to_visible_centers,
        )
        camera = self._task.camera
        if camera.cfg.width != HEAD_WIDTH or camera.cfg.height != HEAD_HEIGHT:
            raise RuntimeError("Head camera geometry differs from visual calibration")
        intrinsics = camera.data.intrinsic_matrices[0].detach().cpu().numpy()
        if (intrinsics.shape != (3, 3) or
                abs(float(intrinsics[0, 0]) - HEAD_FOCAL_PX) > 2 or
                abs(float(intrinsics[1, 1]) - HEAD_FOCAL_PX) > 2 or
                abs(float(intrinsics[0, 2]) - HEAD_WIDTH / 2) > 2 or
                abs(float(intrinsics[1, 2]) - HEAD_HEIGHT / 2) > 2):
            raise RuntimeError("Head camera intrinsics differ from visual calibration")
        selected_pixels = pixels
        snap_audit = None
        if self.snap_visual_key_pixels:
            pixels, snap_audit = snap_key_pixels_to_visible_centers(
                pixels, self.render_head(),
            )
            print(
                "[capx-shadow] visual key pixel snap "
                + " ".join(
                    f"{note}={row['delta_x_px']:+.2f}px"
                    for note, row in snap_audit.items()
                ),
                flush=True,
            )
        targets = locate_key_pixels(
            pixels, expected_notes=self.piano_notes, mirrored=self.mirror_head_view,
        )
        self._append_private_audit(
            "shadow_visual_key_maps.jsonl",
            {"kind": "accepted", "ready": 1, "action_step": self._sim_step_count,
             "targets": targets, "selected_pixels": selected_pixels,
             "pixel_snap": snap_audit},
        )
        os.fsync(self._private_audit_streams["shadow_visual_key_maps.jsonl"].fileno())
        self._visual_key_map_ready = True
        print(f"[capx-shadow] visual_key_map_ready=1 action_step={self._sim_step_count}", flush=True)
        return targets

    def _sample_public_force_depth(self) -> None:
        estimator = getattr(self, "_force_estimator", None)
        if estimator is not None and estimator.uses_area_rate:
            tactile = self._task.tactile
            estimator.observe_frame(tactile.depth_m, tactile.reference_depth_m,
                                    physics_step=self._task.step_count)

    def _ensure_tactile_skill_memory(self):
        if self.tactile_skill_memory_path is None:
            raise RuntimeError('Tactile skill memory is not enabled')
        if self._tactile_skill_memory is None:
            from capx.integrations.opentac_shadow_code_memory import PublicTactileSkillMemory
            signature={'sensor':self._force_estimator.sidecar['sensor_sha256'],
                'force_model':hashlib.sha256(Path(self.digit_force_sidecar).read_bytes()).hexdigest(),
                'action_repeat':self.action_repeat,'physics_hz':240,'features':5}
            session=json.loads((self._output_dir/'piano_code_memory_session.json').read_text(encoding='utf-8'))
            signature['controller_source']=session['executed_version']
            self._tactile_skill_memory=PublicTactileSkillMemory(self.tactile_skill_memory_path,
                signature,self._force_estimator.sidecar['models'],self.tactile_skill_memory_mode,
                self._write_public_event)
        return self._tactile_skill_memory

    def _tactile_skill_context(self, note, dynamics, probe_z_m):
        from capx.integrations.opentac_shadow_vision import parse_note
        i=self._public_attempts_finalized
        if i>=len(self.piano_notes) or parse_note(note)[0]!=parse_note(self.piano_notes[i])[0] or dynamics!=self.piano_dynamics[i]:
            raise ValueError('Tactile skill context must match the current public request')
        if not isinstance(probe_z_m,(int,float)) or not np.isfinite(probe_z_m):
            raise ValueError('Invalid public probe height')
        return 'black' if parse_note(note)[1] else 'white'

    def query_tactile_skill(self, note, dynamics, probe_z_m):
        color=self._tactile_skill_context(note,dynamics,probe_z_m)
        return self._ensure_tactile_skill_memory().query(str(note),dynamics,color,
            float(probe_z_m),self._sim_step_count)

    def record_tactile_skill(self, note, dynamics, probe_z_m, function_name):
        color=self._tactile_skill_context(note,dynamics,probe_z_m)
        session=json.loads((self._output_dir/'piano_code_memory_session.json').read_text(encoding='utf-8'))
        return self._ensure_tactile_skill_memory().record(str(note),dynamics,color,
            float(probe_z_m),session['source'],function_name,self._sim_step_count,self._output_dir.name)

    def estimate_key_force(self, note: str | int) -> dict[str, Any]:
        if not self.visual_key_mode or self._force_estimator is None:
            raise RuntimeError("DIGIT key-force calibration is not enabled")
        from capx.integrations.opentac_shadow_vision import parse_note
        midi, black = parse_note(note)
        tactile = self._task.tactile
        if tactile.depth_m is None or tactile.reference_depth_m is None:
            return {"estimated_key_force_N": None, "sample_step": None,
                    "age_steps": None, "quality": "unavailable",
                    "calibration_version": self._force_estimator.sidecar["schema"]}
        diagnostics = {} if self._output_dir is not None or getattr(self,'tactile_skill_memory_path',None) is not None else None
        estimate = self._force_estimator.predict(
            tactile.depth_m, tactile.reference_depth_m,
            color="black" if black else "white",
            sample_step=self._sim_step_count, current_step=self._sim_step_count,
            diagnostics_out=diagnostics,
            physics_step=int(getattr(self._task, "step_count", 0)),
        )
        self._write_public_event("force_estimate", note=str(note), estimate=estimate,
                                 target=self._action.tolist(), hand_state=self.get_observation()["hand_state"])
        if getattr(self,'tactile_skill_memory_path',None) is not None:
            i=self._public_attempts_finalized
            self._ensure_tactile_skill_memory().observe(str(note),self.piano_dynamics[i],
                estimate,diagnostics.get('features'),self._sim_step_count,self._task.step_count,
                self._action.tolist(),self.get_observation()['hand_state'])
        if self._output_dir is not None:
            truth = float(np.asarray(tactile.key_forces())[midi - 21])
            snapshot = (f"shadow_digit_ood_depth_{self._sim_step_count}_{midi}.npz"
                        if estimate["quality"] == "ood" else None)
            self._append_private_audit(
                "shadow_force_estimates_private_audit.jsonl",
                {"note": str(note), "action_step": self._sim_step_count,
                 "physics_step": int(getattr(self._task, "step_count", 0)),
                 "estimate": estimate, "depth_diagnostics": diagnostics,
                 "public_hand_state": self.get_observation().get("hand_state", {}),
                 "action_target": self._action.tolist(), "depth_sample_file": snapshot,
                 "private_key_top_force_N": truth},
            )
            if snapshot is not None:
                # Save only OOD frames, using the same write-only permissions as
                # private labels. Never return this path or arrays to agent code.
                try:
                    fd = os.open(self._private_audit_dir / snapshot,
                                 os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o200)
                except FileExistsError:
                    pass  # Repeated estimate calls at one step reuse its frame.
                else:
                    with os.fdopen(fd, "wb") as stream:
                        np.savez_compressed(stream, depth_m=tactile.depth_m,
                                            reference_depth_m=tactile.reference_depth_m)
        return estimate

    def _append_private_audit(self, filename: str, row: dict) -> None:
        """Immediately append a private label outside the generated-code artifacts."""
        if self._private_audit_dir is None:
            self.private_audit_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.private_audit_root.chmod(0o300)
            run_id = self._output_dir.name if self._output_dir is not None else "shadow"
            self._private_audit_dir = self.private_audit_root / f"{run_id}_{uuid.uuid4().hex}"
            # During generated-code execution the random subdirectory cannot
            # be listed and the files are write-only. Their path is not
            # included in the public API, prompt or observation.
            self._private_audit_dir.mkdir(mode=0o300)
        stream = self._private_audit_streams.get(filename)
        if stream is None:
            fd = os.open(self._private_audit_dir / filename,
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o200)
            stream = os.fdopen(fd, "w", encoding="utf-8", buffering=1)
            self._private_audit_streams[filename] = stream
        stream.write(json.dumps(row, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")
        stream.flush()

    def _close_private_audit(self) -> None:
        for stream in self._private_audit_streams.values():
            try:
                os.fsync(stream.fileno())
            finally:
                stream.close()
        self._private_audit_streams.clear()
        self.private_audit_root.chmod(0o700)
        if self._private_audit_dir is not None:
            self._private_audit_dir.chmod(0o700)
            for path in self._private_audit_dir.iterdir():
                path.chmod(0o600)
        if self._private_audit_dir is not None and self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            (self._output_dir / "shadow_private_audit_location.json").write_text(
                json.dumps({"private_audit_dir": str(self._private_audit_dir)}) + "\n", encoding="utf-8"
            )

    def get_index_tactile_rgb(self) -> np.ndarray:
        if self._observation is None:
            raise RuntimeError("Reset the Shadow task before reading DIGIT")
        rgb = np.asarray(self._observation["tactile"]["index"]["rgb"])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise RuntimeError(f"Invalid DIGIT RGB frame: shape={rgb.shape}, dtype={rgb.dtype}")
        return rgb.copy()

    def get_note_target(self, note: str | int) -> dict[str, Any]:
        from envs._piano_sequence import parse_notes, note_name
        from envs._piano import keyboard_layout

        midi = parse_notes([note])[0]
        key = keyboard_layout()[midi - 21]
        return {"note": note_name(midi), "midi": midi, "black": bool(key.black),
                "wrist_x_offset_m": 0.07 if key.black else 0.0,
                "wrist_y_m": float(key.position[1] - 0.033),
                "press_lift_z_m": 0.0155 if key.black else 0.0,
                "hover_lift_z_m": 0.04}

    def compute_reward(self) -> float:
        return 1.0 if self.task_completed() else 0.0

    def task_completed(self) -> bool:
        return self._policy_end_submitted and self._policy_end_verdict

    def submit_policy_end(self) -> bool:
        if self._policy_end_submitted:
            raise RuntimeError("END has already been submitted")
        self._policy_end_submitted = True
        self._policy_end_verdict = bool(self._native_info.get("success", False))
        self._write_native_event("END")
        self._save_tactile_snapshot("shadow_digit_end.png")
        return self._policy_end_verdict

    def _capture_frame(self) -> None:
        frame = np.asarray(self._task.render())
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise RuntimeError(f"Invalid native piano video frame: {frame.shape}")
        frame = np.asarray(frame[:, :, :3], dtype=np.uint8)
        if self.mirror_head_view:
            frame = np.ascontiguousarray(frame[:, ::-1])
        if frame.shape[1] > self.video_width:
            from PIL import Image
            width = self.video_width
            height = round(frame.shape[0] * width / frame.shape[1])
            frame = np.asarray(Image.fromarray(frame).resize((width, height)))
        pad_h = -frame.shape[0] % 16
        pad_w = -frame.shape[1] % 16
        if pad_h or pad_w:
            frame = np.pad(frame, ((pad_h // 2, pad_h - pad_h // 2),
                                   (pad_w // 2, pad_w - pad_w // 2), (0, 0)))
        self._frame_buffer.append(frame.copy())
        if getattr(self, '_third_camera', None) is not None:
            self._third_camera.update(self._task.cfg.physics_dt, force_recompute=True)
            third = self._third_camera.data.output['rgb'][0].detach().cpu().numpy()[:,:,:3]
            from PIL import Image
            third = np.asarray(Image.fromarray(third.astype(np.uint8)).resize((frame.shape[1],frame.shape[0])))
            self._third_frame_buffer.append(third.copy())
        self._video_physics_steps.append(int(getattr(self._task, "step_count", self._sim_step_count * self.action_repeat)))

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True,
                             capture_initial_frame: bool = True) -> None:
        self._record_frames = bool(enabled)
        if clear:
            self._frame_buffer.clear()
            if hasattr(self, '_third_frame_buffer'):
                self._third_frame_buffer.clear()
            self._video_physics_steps.clear()
        if enabled and capture_initial_frame and self._observation is not None:
            self._capture_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
            # The trial writer still needs frame timestamps for audio muxing.
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_third_party_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [f.copy() for f in self._third_frame_buffer]
        if clear:
            self._third_frame_buffer.clear()
        return frames

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def add_audio_to_video(self, video_path: str | Path, start: int = 0, end: int | None = None) -> bool:
        """Postprocess a saved video using private, physics-timed contact events."""
        contacts = getattr(self._task, "audio_contacts", None)
        if contacts is None:
            return False
        steps = self._video_physics_steps[start:end]
        expected_frames = (len(self._video_physics_steps) if end is None else end) - start
        if not steps or len(steps) != expected_frames:
            raise ValueError("Piano video frames and physics timestamps are not aligned")
        contacts.finish(int(getattr(self._task, "step_count", steps[-1])))
        from envs._piano_audio import add_piano_audio

        result = add_piano_audio(video_path, contacts.events, steps, fps=self.video_fps)
        print(
            f"[capx-shadow] piano audio video={video_path} wav={result['wav_path']} "
            f"events={len(result['events'])}",
            flush=True,
        )
        return True

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array rendering is supported")
        return self.render_head() if self.visual_key_mode else np.asarray(self._task.render())

    def render_head(self) -> np.ndarray:
        if self.visual_key_mode:
            self._configure_camera()
            self._task.sim.render()
        frame = np.asarray(self._task.render())
        return np.ascontiguousarray(frame[:, ::-1]) if self.mirror_head_view else frame

    def render_wrist(self) -> np.ndarray | None:
        if not self.visual_key_mode:
            return None
        palm = self._task.hand.data.body_pos_w[0, self._task._palm].detach().cpu().numpy()
        eye = palm + np.array([-.18, .0, .20])
        target = palm + np.array([.24, .0, -.035])
        self._set_camera_pose(eye.tolist(), target.tolist())
        try:
            self._task.sim.render()
            frame = np.asarray(self._task.render()).copy()
        finally:
            self._configure_camera()
        return frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.visual_key_mode and self._output_dir is not None:
                try:
                    self._close_private_audit()
                    from capx.integrations.opentac_shadow_audit import save_force_alignment, save_native_piano_summary
                    summary = save_native_piano_summary(self._private_audit_dir)
                    if summary is not None:
                        print(f"[capx-shadow] native piano audit={summary}", flush=True)
                        report = json.loads(summary.read_text(encoding="utf-8"))
                        print(
                            "[capx-shadow] piano score "
                            f"correct={report['native_notes_completed']}/{len(report['notes'])} "
                            f"attempted={report['native_notes_attempted']} "
                            f"failed={report['native_notes_failed']} "
                            f"END={report['end_submitted']}",
                            flush=True,
                        )
                    result = save_force_alignment(self._private_audit_dir)
                    if result is not None:
                        print(f"[capx-shadow] force alignment plot={result}", flush=True)
                except Exception as exc:
                    print(f"[capx-shadow] force audit plot unavailable: {exc}", flush=True)
            self._task.close()
        finally:
            ACTIVE_SHADOW_ENVS.discard(self)
