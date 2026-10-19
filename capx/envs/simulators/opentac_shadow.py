"""CaP-X adapter for OpenTacBench's Franka-mounted Shadow Hand piano.

Import the OpenTacBench task only after the Isaac Kit AppLauncher has started.
Native scoring and force measurements stay in the evaluator-side event log.
"""

from __future__ import annotations

import json
import os
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
        piano_notes: list[str | int],
        piano_dynamics: list[str],
        piano_hold_seconds: float | list[float] = 0.12,
        piano_require_dynamics: bool = True,
        seed_base: int = 0,
        device: str = "cuda:0",
        tactile_image_width: int = 320,
        video_frame_stride: int = 3,
        video_width: int = 640,
        task_factory: Callable[["ShadowPianoLowLevelEnv"], Any] | None = None,
        output_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.univtac_root = Path(univtac_root).expanduser().resolve()
        self.piano_notes = list(piano_notes)
        self.piano_dynamics = list(piano_dynamics)
        self.piano_hold_seconds = piano_hold_seconds
        self.piano_require_dynamics = bool(piano_require_dynamics)
        self.seed_base = int(seed_base)
        self.device = device
        self.tactile_image_width = int(tactile_image_width)
        self.video_frame_stride = max(1, int(video_frame_stride))
        self.video_width = int(video_width)
        self._output_dir = Path(output_dir or os.environ.get("CAPX_OUTPUT_DIR", "")) if (output_dir or os.environ.get("CAPX_OUTPUT_DIR")) else None
        self._task = task_factory(self) if task_factory else self._create_native_task()
        self._configure_camera()
        self.video_fps = float(getattr(self._task, "metadata", {}).get("render_fps", 30)) / self.video_frame_stride
        self._observation: dict[str, Any] | None = None
        self._native_info: dict[str, Any] = {}
        self._native_done = False
        self._policy_end_submitted = False
        self._policy_end_verdict = False
        self._sim_step_count = 0
        self._last_native_notes_completed = 0
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._action = np.zeros(4, dtype=np.float32)
        self.max_steps = 100000
        self._closed = False
        ACTIVE_SHADOW_ENVS.add(self)

    def _create_native_task(self):
        from env.tasks.arm_shadow_piano import Task, TaskCfg

        cfg = TaskCfg(
            device=self.device,
            tactile=True,
            tactile_sensor="digit",
            tactile_image_width=self.tactile_image_width,
            piano_notes=self.piano_notes,
            piano_dynamics=self.piano_dynamics,
            piano_hold_seconds=self.piano_hold_seconds,
            piano_require_dynamics=self.piano_require_dynamics,
            piano_seed=self.seed_base,
            record_video=True,
        )
        return Task(cfg=cfg, render_mode="rgb_array")

    @property
    def task(self):
        return self._task

    def _configure_camera(self) -> None:
        if not hasattr(self._task, "camera"):
            return
        import torch
        target_y = self._task.layout[self._task.target].position[1]
        self._task.camera.set_world_poses_from_view(
            torch.tensor([[-.58, .85, .66]], device=self._task.sim.device),
            torch.tensor([[.12, target_y, .095]], device=self._task.sim.device),
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
        with (self._output_dir / "shadow_native_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")

    def _save_tactile_snapshot(self, filename: str) -> None:
        if self._output_dir is None:
            return
        from PIL import Image
        self._output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.get_index_tactile_rgb()).save(self._output_dir / filename)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        actual_seed = self.seed_base if seed is None else self.seed_base + int(seed)
        self._observation, self._native_info = self._task.reset(seed=actual_seed)
        self._native_done = False
        self._policy_end_submitted = False
        self._policy_end_verdict = False
        self._sim_step_count = 0
        self._last_native_notes_completed = 0
        self._action = np.zeros(4, dtype=np.float32)
        self._frame_buffer.clear()
        if hasattr(self._task, "camera"):
            # Match the official piano CLI's RTX camera warmup before capture.
            for _ in range(16):
                self._task.sim.render()
                self._task.camera.update(self._task.cfg.physics_dt, force_recompute=True)
        self._write_native_event("reset")
        self._save_tactile_snapshot("shadow_digit_initial.png")
        return self.get_observation(), {}

    def _validate_action(self, action: Any) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (4,) or not np.isfinite(value).all():
            raise ValueError("Shadow action must contain four finite wrist/index targets")
        bounds = self._task.action_space
        if np.any(value < bounds.low) or np.any(value > bounds.high):
            raise ValueError(f"Shadow action outside native bounds: {value.tolist()}")
        return value

    def step(self, action):
        if self._policy_end_submitted or self._native_done:
            raise RuntimeError("Native piano episode has ended; no further action is allowed")
        value = self._validate_action(action)
        self._observation, reward, terminated, truncated, self._native_info = self._task.step(value)
        self._action = value.copy()
        self._sim_step_count += 1
        self._native_done = bool(terminated or truncated)
        self._write_native_event("action")
        completed = int(self._native_info.get("notes_completed", 0))
        if completed > self._last_native_notes_completed:
            for index in range(self._last_native_notes_completed + 1, completed + 1):
                self._save_tactile_snapshot(f"shadow_digit_note_{index:02d}.png")
            self._last_native_notes_completed = completed
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

    def get_observation(self) -> dict[str, Any]:
        if self._observation is None:
            return {}
        state = np.asarray(self._observation["state"], dtype=np.float32)
        if state.shape != (111,):
            raise RuntimeError(f"Unexpected native Shadow state shape: {state.shape}")
        native_action_state = np.asarray(self._task._action_state(), dtype=np.float32)
        if native_action_state.shape != (4,):
            raise RuntimeError(f"Unexpected native Shadow action-state shape: {native_action_state.shape}")
        return {"hand_state": {"wrist_x_offset_m": float(native_action_state[0]),
                               "wrist_y_m": float(state[0]), "wrist_z_m": float(state[1]),
                               "index_flexion_rad": float(state[2]), "wrist_and_index_velocity": state[3:6].tolist(),
                               "arm_joint_position_rad": state[97:104].tolist(),
                               "arm_joint_velocity_rad_s": state[104:111].tolist()},
                "key_angles_rad": state[6:94].tolist()}

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

    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True,
                             capture_initial_frame: bool = True) -> None:
        self._record_frames = bool(enabled)
        if clear:
            self._frame_buffer.clear()
        if enabled and capture_initial_frame and self._observation is not None:
            self._capture_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer.clear()
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("Only rgb_array rendering is supported")
        return np.asarray(self._task.render())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._task.close()
        finally:
            ACTIVE_SHADOW_ENVS.discard(self)
