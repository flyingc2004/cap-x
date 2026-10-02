"""Isaac-free rendering coverage for the tension-strap native video panel."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from capx.envs.simulators.univtac import (
    UniVTACLowLevelEnv,
    _build_tension_response_visualization,
    _empty_tension_response_visualization,
    _render_tension_strap_demo_frame,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_PATH = REPO_ROOT / "env_configs/univtac/tactile_response_memory/tension_strap_12n_18n.v1.json"


def _response(*, valid: bool = True) -> dict:
    return {
        "schema_version": "tactile_stage_response.v1",
        "capture_id": "stage_capture_007",
        "window": {"captured_frame_count": 180},
        "quality": {
            "valid": valid,
            "bilateral_contact_ratios": {"hold": 1.0, "start": 1.0, "end": 1.0},
        },
        "hold": {
            "left": {
                "depth_mm": 2.0,
                "marker_row_gradient_px": 0.2,
                "marker_col_gradient_px": 0.3,
                "marker_anisotropy_ratio": 1.4,
            },
            "right": {
                "depth_mm": 2.1,
                "marker_row_gradient_px": 0.22,
                "marker_col_gradient_px": 0.31,
                "marker_anisotropy_ratio": 1.5,
            },
        },
        "end_minus_start": {
            "left": {"depth_mm": 0.4},
            "right": {"depth_mm": 0.45},
        },
    }


def _memory() -> dict:
    return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))


def test_runtime_demo_renders_current_and_both_frozen_medians() -> None:
    visualization = _build_tension_response_visualization(_response(), _memory())

    assert visualization["status"] == "valid_capture"
    assert len(visualization["current_values"]) == 10
    assert set(visualization["stage_values"]) == {"12N", "18N"}
    image = np.full((80, 120, 3), 73, dtype=np.uint8)
    frame = _render_tension_strap_demo_frame(
        obs={
            "observation": {"head": {"rgb": image}, "wrist": {"rgb": image}},
            "tactile": {
                "left_tactile": {"rgb": image, "rgb_marker": image},
                "right_tactile": {"rgb": image, "rgb_marker": image},
            },
        },
        visualization=visualization,
        control_diagnostics={
            "latest_estimate": {"estimated_tension_N": 12.1},
            "current_stage_index": 0,
            "current_stage_in_band_hold_seconds": 1.5,
            "stage_action_counts": [5, 0],
        },
        history=[
            {
                "left_depth_mm": 5.0 + index * 0.1,
                "right_depth_mm": 5.2 + index * 0.1,
                "left_marker_displacement_px": 2.0 + index,
                "right_marker_displacement_px": 2.5 + index,
            }
            for index in range(4)
        ],
    )

    assert frame.shape == (1200, 1600, 3)
    assert frame.dtype == np.uint8
    assert int(frame.std()) > 0


def test_invalid_capture_preserves_the_last_valid_current_response() -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._tension_response_visualization = _empty_tension_response_visualization()
    memory = _memory()

    env.set_tension_response_visualization(_response(), memory)
    valid_values = list(env._tension_response_visualization["current_values"])
    env.set_tension_response_visualization(_response(valid=False), memory)

    assert env._tension_response_visualization["status"] == "invalid_capture"
    assert env._tension_response_visualization["current_values"] == valid_values
    assert env._tension_response_visualization["latest_quality"]["valid"] is False


def test_rolling_preview_shows_partial_10d_and_frozen_memory() -> None:
    visualization = _build_tension_response_visualization(
        _response(valid=False),
        _memory(),
        source="rolling_preview",
        allow_partial=True,
    )

    assert visualization["status"] == "live_warming_up"
    assert visualization["source"] == "rolling_preview"
    assert len(visualization["current_values"]) == 10
    assert set(visualization["stage_values"]) == {"12N", "18N"}


def test_preflight_failure_diagnostics_write_to_current_run(monkeypatch, tmp_path) -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._runtime_preflight_diagnostics = {
        "schema_version": "capx_univtac_runtime_preflight.v1",
        "errors": ["missing contact gradient"],
    }
    monkeypatch.setenv("CAPX_OUTPUT_DIR", str(tmp_path))

    diagnostic_path = env._write_runtime_preflight_failure_diagnostics()

    assert diagnostic_path == tmp_path / "univtac_runtime_preflight_error.json"
    saved = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert saved["errors"] == ["missing contact gradient"]


def test_task_native_tension_renderer_uses_the_runtime_demo_layout() -> None:
    image = np.full((80, 120, 3), 73, dtype=np.uint8)
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env.video_renderer = "task_native"
    env.tension_response_panel_enabled = True
    env._tension_response_visualization = _build_tension_response_visualization(
        _response(), _memory()
    )

    env._opentac_tension_estimator_diagnostics = {
        "latest_estimate": {"estimated_tension_N": 12.1},
        "current_stage_index": 0,
        "current_stage_in_band_hold_seconds": 1.5,
    }
    env._tension_response_video_history = [
        {
            "left_depth_mm": 5.0 + index * 0.1,
            "right_depth_mm": 5.2 + index * 0.1,
            "left_marker_displacement_px": 2.0 + index,
            "right_marker_displacement_px": 2.5 + index,
        }
        for index in range(4)
    ]

    frame = env._render_video_frame(
        {
            "observation": {"head": {"rgb": image}, "wrist": {"rgb": image}},
            "tactile": {
                "left_tactile": {"rgb": image, "rgb_marker": image},
                "right_tactile": {"rgb": image, "rgb_marker": image},
            },
        }
    )

    assert frame.shape == (1200, 1600, 3)
    assert not np.array_equal(frame[:450, :800], image)


def test_tension_video_stream_writes_combined_and_turn_files(tmp_path) -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._video_stream_enabled = True
    env._video_stream_dir = tmp_path / "stream"
    env._video_stream_dir.mkdir()
    env._video_stream_combined_writer = None
    env._video_stream_turn_writer = None
    env._video_stream_turn_index = None
    env._video_stream_frame_count = 0

    env.begin_video_turn(0)
    env._write_streamed_video_frame(np.full((1200, 1600, 3), 73, dtype=np.uint8))
    env.end_video_turn()

    assert env.get_video_frame_count() == 1
    assert env.finalize_streamed_video(tmp_path / "final")
    assert (tmp_path / "final/videos/video_combined.mp4").exists()
    assert (tmp_path / "final/videos/video_turn_00.mp4").exists()


def test_force_task_vertical_delta_bridges_public_z_to_absolute_qpos() -> None:
    class _RobotManager:
        def compute_delta_ee_rotvec_qpos_target(self, action):
            assert action.shape == (7,)
            assert np.isclose(float(action[2]), 0.002)
            arm = torch.arange(7, dtype=torch.float32).reshape(1, 7)
            gripper = torch.tensor([0.015], dtype=torch.float32)
            return arm, gripper, torch.zeros((1, 3)), torch.zeros((1, 4))

    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._force_task_mode = True
    env._task = SimpleNamespace(
        _robot_manager=_RobotManager(),
        device="cpu",
    )
    observed: dict[str, object] = {}

    def take_action(action, *, action_type):
        observed["action"] = action
        observed["action_type"] = action_type
        return {"ok": True}

    env.take_action = take_action
    result = env.move_force_task_vertical_delta(dz=0.002)

    assert result["ok"] is True
    assert result["motion_path"] == "force_task_legacy_delta_ik_to_absolute_qpos"
    assert result["requested_delta_z_m"] == 0.002
    assert observed["action_type"] == "qpos"
    assert torch.equal(
        observed["action"],
        torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.015]),
    )


def test_force_task_vertical_delta_keeps_an_accumulated_command_target() -> None:
    class _Ik:
        def __init__(self) -> None:
            self.commands: list[torch.Tensor] = []

        def set_command(self, command: torch.Tensor) -> None:
            self.commands.append(command.clone())

        def compute(self, *_args) -> torch.Tensor:
            return torch.zeros((1, 7), dtype=torch.float32)

    class _Robot:
        def __init__(self) -> None:
            self.data = SimpleNamespace(
                joint_pos=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0126]]),
                soft_joint_pos_limits=torch.stack(
                    [torch.full((1, 7), -1.0), torch.full((1, 7), 1.0)], dim=-1
                ),
            )

    class _RobotManager:
        def __init__(self) -> None:
            self._ik_controller = _Ik()
            self._arm_ids = list(range(7))
            self._gripper_ids = [7]
            self.jacobian_b = torch.zeros((1, 6, 7), dtype=torch.float32)
            self.robot = _Robot()
            self.position = torch.tensor([[0.5, 0.0, 0.2]], dtype=torch.float32)
            self.quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        def get_ee_pose_tensor(self):
            return self.position, self.quaternion

    manager = _RobotManager()
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._force_task_mode = True
    env._force_task_command_ee_pos = None
    env._force_task_command_ee_quat = None
    env.api_configs = {"franka_control_api": {"force_task_target_lead_m": 0.001}}
    env._task = SimpleNamespace(_robot_manager=manager, device="cpu")
    env.take_action = lambda action, *, action_type: {"ok": True}

    first = env.move_force_task_vertical_delta(dz=0.0001)
    manager.position[:, 2] = 0.20002  # Physical motion lags the command.
    second = env.move_force_task_vertical_delta(dz=0.0001)

    assert first["motion_path"] == "force_task_accumulated_ik_to_absolute_qpos"
    assert second["motion_path"] == "force_task_accumulated_ik_to_absolute_qpos"
    assert first["command_target_ee_z_m"] == pytest.approx(0.2001)
    assert second["command_target_ee_z_m"] == pytest.approx(0.2002)
