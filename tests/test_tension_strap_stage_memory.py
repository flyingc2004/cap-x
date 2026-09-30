"""Isaac-free coverage for the external tension-strap runtime memory bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from capx.integrations.univtac.native_tactile import (
    UniVTACTactileBuffer,
    UniVTACTactileFrame,
    summarize_tactile_stage_response,
)
from capx.integrations.opentac import OpenTacApi
from capx.integrations.univtac.tactile_api import UniVTACTactileApi


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_PATH = REPO_ROOT / "env_configs/univtac/tactile_response_memory/tension_strap_12n_18n.v1.json"
CONFIG_PATH = REPO_ROOT / "env_configs/univtac/tension_strap_stage_memory_control.yaml"


def _depth(indentation_mm: float) -> np.ndarray:
    depth = np.full((16, 16), 34.0, dtype=np.float32)
    depth[4:12, 4:12] = 34.0 - indentation_mm
    return depth


def _marker(row_scale: float, column_scale: float) -> np.ndarray:
    rows, columns = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    initial = np.zeros((8, 8, 2), dtype=np.float32)
    current = initial.copy()
    current[..., 0] = rows * row_scale
    current[..., 1] = columns * column_scale
    return np.stack((initial.reshape(-1, 2), current.reshape(-1, 2)), axis=0)


def _frame(step: int, indentation_mm: float, *, right_contact: bool = True) -> UniVTACTactileFrame:
    return UniVTACTactileFrame(
        step=step,
        timestamp=float(step) / 60.0,
        left_depth=_depth(indentation_mm),
        right_depth=_depth(indentation_mm if right_contact else 0.0),
        left_marker=_marker(0.8 + step * 0.01, 0.4),
        right_marker=_marker(0.7 + step * 0.01, 0.5),
        left_pose=None,
        right_pose=None,
    )


def _memory(window_steps: int = 6, edge_window_steps: int = 2) -> dict:
    memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    memory["capture"]["window_steps"] = window_steps
    memory["capture"]["edge_window_steps"] = edge_window_steps
    return memory


class _Env:
    def __init__(self, frames: list[UniVTACTactileFrame], memory: dict) -> None:
        self.tactile_buffer = UniVTACTactileBuffer(maxlen=500)
        for frame in frames:
            self.tactile_buffer.append(frame)
        self.api_configs = {"opentac_api": {"tactile_stage_memory": memory}}
        self.trace: list[dict] = []
        self.snapshot: dict = {}

    def refresh_native_observation(self, **_kwargs):
        return {}

    def get_native_tactile_calibration(self) -> dict:
        return {
            "depth_far_plane_mm": 34.0,
            "depth_contact_margin_mm": 0.5,
            "force_full_scale_mm": 6.5,
        }

    def get_step_count(self) -> int:
        frames = self.tactile_buffer.frames()
        return frames[-1].step if frames else 0

    def append_tactile_working_memory_trace(self, record: dict) -> None:
        self.trace.append(record)

    def set_tactile_trial_memory_snapshot(self, memory: dict) -> None:
        self.snapshot = memory


def test_runtime_capture_has_public_10d_stage_response_without_stage_label() -> None:
    env = _Env([_frame(step, 6.0 + 0.02 * step) for step in range(6)], _memory())
    api = OpenTacApi(env)

    frozen = api.get_tactile_stage_memory()
    response = api.capture_tactile_stage_response()

    assert frozen["schema_version"] == "tactile_stage_memory.v1"
    assert [stage["memory_id"] for stage in frozen["stages"]] == [
        "tension_strap_hold_12n.v1",
        "tension_strap_hold_18n.v1",
    ]
    assert frozen["retrieval_scaler"]["method"] == "pooled_calibration_iqr"
    assert frozen["retrieval_scaler"]["response_blocks"][0]["iqr"] != frozen["stages"][0]["response_blocks"][0]["scaler"]["iqr"]
    assert response["schema_version"] == "tactile_stage_response.v1"
    assert response["protocol_id"] == frozen["protocol_id"]
    assert response["window"]["captured_frame_count"] == 6
    assert response["quality"]["valid"] is True
    assert response["hold"]["left"]["depth_mm"] > 5.0
    assert response["hold"]["left"]["marker_row_gradient_px"] > 0.0
    assert response["hold"]["right"]["marker_col_gradient_px"] > 0.0
    assert response["end_minus_start"]["left"]["depth_mm"] > 0.0
    assert "target_stage" not in response
    assert "actor" not in repr(response).lower()
    assert "tension" not in repr(response).lower()
    assert any(record["event"] == "stage_response_capture" for record in env.trace)


def test_stage_response_rejects_insufficient_bilateral_contact() -> None:
    frames = [_frame(step, 6.0, right_contact=False) for step in range(6)]
    response = summarize_tactile_stage_response(
        frames,
        window_steps=6,
        edge_window_steps=2,
        min_bilateral_contact_ratio=0.8,
        depth_far_plane_mm=34.0,
        depth_contact_margin_mm=0.5,
        contact_area_threshold=0.001,
    )

    assert response["quality"]["bilateral_contact_ratios"]["hold"] == 0.0
    assert response["quality"]["valid"] is False


def test_stage_response_accepts_flat_or_grid_marker_layouts() -> None:
    frame = _frame(0, 6.0)
    frame.left_marker = frame.left_marker.reshape(2, 8, 8, 2)
    frame.right_marker = frame.right_marker.reshape(2, 8, 8, 2)
    response = summarize_tactile_stage_response(
        [frame, _frame(1, 6.0)],
        window_steps=2,
        edge_window_steps=1,
        min_bilateral_contact_ratio=0.8,
        depth_far_plane_mm=34.0,
    )

    assert response["hold"]["left"]["marker_row_gradient_px"] > 0.0
    assert response["hold"]["right"]["marker_col_gradient_px"] > 0.0


def test_stage_memory_loader_accepts_sidecar_and_has_no_auto_retrieval_function() -> None:
    env = _Env([_frame(step, 6.0) for step in range(6)], _memory())
    env.api_configs = {
        "opentac_api": {
            "tactile_stage_memory_path": "env_configs/univtac/tactile_response_memory/tension_strap_12n_18n.v1.json"
        }
    }
    api = OpenTacApi(env)

    assert api.get_tactile_stage_memory()["protocol_id"] == "external_strap_12n_18n_hold_response.v1"
    assert "capture_tactile_stage_response" in api.functions()
    assert "get_tactile_stage_memory" in api.functions()
    assert "score_tactile_stage_response" not in api.functions()
    assert "select_tactile_stage" not in api.functions()


def test_stage_memory_rejects_private_feature_paths() -> None:
    bad_memory = _memory()
    bad_memory["stages"][0]["response_blocks"][0]["fields"][0]["path"] = "actor.elastic_strap"
    api = OpenTacApi(_Env([_frame(step, 6.0) for step in range(6)], bad_memory))

    with pytest.raises(RuntimeError, match="non-public response field"):
        api.get_tactile_stage_memory()


def test_stage_memory_rejects_stage_specific_distance_scaler() -> None:
    bad_memory = _memory()
    bad_memory["retrieval_scaler"]["method"] = "stage_specific_iqr"
    api = OpenTacApi(_Env([_frame(step, 6.0) for step in range(6)], bad_memory))

    with pytest.raises(RuntimeError, match="pooled_calibration_iqr"):
        api.get_tactile_stage_memory()


def test_tension_strap_control_yaml_uses_opentac_and_frozen_memory() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    low_level = config["env"]["cfg"]["low_level"]
    tactile = low_level["api_configs"]["opentac_api"]

    assert low_level["task_name"] == "tension_strap"
    assert low_level["force_task"] is True
    assert config["env"]["cfg"]["apis"] == ["FrankaControlApi", "OpenTacApi"]
    assert tactile["llm_visible_functions"] == [
        "get_tactile_tension_control_contract",
        "begin_tactile_tension_estimator",
        "get_tactile_tension_estimate",
        "get_tactile_stage_memory",
        "capture_tactile_stage_response",
    ]
    prompt = config["env"]["cfg"]["prompt"]
    assert "true tension" in prompt
    assert "pooled-calibration IQR" in prompt
    assert "physical scorer" in OpenTacApi.__doc__
    assert config["tactile_memory"]["persistent"]["enabled"] is False


def test_univtac_tactile_api_does_not_expose_force_task_stage_memory() -> None:
    functions = UniVTACTactileApi.__new__(UniVTACTactileApi).functions()
    assert "get_tactile_stage_memory" not in functions
    assert "capture_tactile_stage_response" not in functions


def test_opentac_estimator_uses_only_public_marker_rgb_and_calibration(monkeypatch) -> None:
    class _Tracker:
        def __init__(self, reference):
            self.reference = reference

    class _Utilities:
        MarkerFlowTracker = _Tracker

        @staticmethod
        def tracked_flow_rgb_features(reference, current, tracker):
            assert tracker.reference is reference
            return np.asarray([1.0, 2.0]), {"left_tactile": {"matched": 40, "p90_px": 1.5}}

        @staticmethod
        def predict_calibrated(calibration, features):
            assert calibration == {"weights": [0.1, 0.2]}
            assert np.allclose(features, [1.0, 2.0])
            return 13.25

    env = _Env([_frame(step, 6.0) for step in range(6)], _memory())
    api = OpenTacApi(env)
    images = {
        "left_tactile": np.zeros((16, 16, 3), dtype=np.uint8),
        "right_tactile": np.zeros((16, 16, 3), dtype=np.uint8),
    }
    monkeypatch.setattr(api, "_read_marker_images", lambda: images)
    monkeypatch.setattr(api, "_load_tension_calibration", lambda: {"weights": [0.1, 0.2]})
    monkeypatch.setattr(api, "_force_task_utilities", lambda: _Utilities)

    contract = api.get_tactile_tension_control_contract()
    baseline = api.begin_tactile_tension_estimator()
    estimate = api.get_tactile_tension_estimate()

    assert contract["allowed_translation_axes"] == ["z"]
    assert baseline["status"] == "baseline_ready"
    assert estimate["estimated_tension_N"] == 13.25
    assert "true_tension" not in estimate
    assert "reward" not in estimate
    assert "success" not in estimate
