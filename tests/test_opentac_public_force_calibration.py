"""Isaac-free coverage for the frozen public depth/marker tension estimator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from capx.integrations.opentac import OpenTacApi
from capx.integrations.opentac.public_force_calibration import (
    ESTIMATOR_MODE,
    FEATURE_NAMES,
    extract_public_tactile_force_features,
    predict_public_tactile_force,
    validate_public_force_calibration,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SIDECAR = (
    REPO_ROOT
    / "env_configs/univtac/tension_calibration/tension_strap_public_tactile_force_69seed.v1.json"
)


def _marker(scale: float = 1.0) -> np.ndarray:
    rows, columns = np.meshgrid(np.arange(8, dtype=float), np.arange(8, dtype=float), indexing="ij")
    initial = np.stack((columns, rows), axis=-1)
    current = initial.copy()
    current[..., 0] += scale * (0.5 + 0.1 * rows)
    current[..., 1] += scale * (0.2 + 0.05 * columns)
    return np.stack((initial.reshape(-1, 2), current.reshape(-1, 2)), axis=0).astype(np.float32)


def _depth(indentation_mm: float) -> np.ndarray:
    return np.full((16, 16), 34.0 - indentation_mm, dtype=np.float32)


class _PublicObservationEnv:
    def __init__(self) -> None:
        self.step = 0
        self.post_step: dict[str, Any] = {}
        self.trace: list[dict[str, Any]] = []
        self.estimator_trace: list[dict[str, Any]] = []
        self.private_force_audit: list[dict[str, Any]] = []
        self.diagnostics: dict[str, Any] = {}
        self.missing_marker = False
        self.api_configs = {
            "opentac_api": {
                "tension_estimator_mode": ESTIMATOR_MODE,
                "tension_calibration_path": (
                    "env_configs/univtac/tension_calibration/"
                    "tension_strap_public_tactile_force_69seed.v1.json"
                ),
                "stage_targets_N": [12.0, 18.0],
                "stage_tolerance_N": 0.5,
                "observation_dt_s": 1.0 / 60.0,
                "estimator_update_stride": 1,
                "max_estimate_age_steps": 3,
                "estimator_audit_stride": 1,
                "private_force_audit": {"enabled": True, "sample_stride": 1},
            }
        }

    def get_step_count(self) -> int:
        return self.step

    def get_action_count(self) -> int:
        return self.step

    def refresh_native_observation(self, **_kwargs: Any) -> dict[str, Any]:
        return {}

    def current_raw_observation(self) -> dict[str, Any]:
        marker = None if self.missing_marker else _marker(1.0)
        return {
            "tactile": {
                "left_tactile": {"depth": _depth(6.2), "marker": marker},
                "right_tactile": {"depth": _depth(6.6), "marker": marker},
            }
        }

    def register_post_step_observer(self, name: str, observer: Any) -> None:
        self.post_step[name] = observer

    def unregister_post_step_observer(self, name: str) -> None:
        self.post_step.pop(name, None)

    def append_tactile_working_memory_trace(self, record: dict[str, Any]) -> None:
        self.trace.append(record)

    def append_opentac_tension_estimator_trace(self, record: dict[str, Any]) -> None:
        self.estimator_trace.append(record)

    def read_private_tension_for_posthoc_audit(self) -> dict[str, Any]:
        return {"true_tension_N": 12.25, "step": self.step}

    def append_opentac_tension_calibration_audit(self, record: dict[str, Any]) -> None:
        self.private_force_audit.append(record)

    def set_opentac_tension_estimator_diagnostics(self, record: dict[str, Any]) -> None:
        self.diagnostics = record

    def get_robot_state(self) -> dict[str, Any]:
        return {"ee_pos": [0.5, 0.0, 0.2], "gripper_qpos": 0.01}

    def tick(self) -> None:
        self.step += 1
        for observer in tuple(self.post_step.values()):
            observer()


def test_public_feature_order_matches_the_frozen_26d_contract() -> None:
    features = extract_public_tactile_force_features(
        (_depth(7.0), _depth(8.0)),
        (_marker(1.0), _marker(2.0)),
        far_plane_mm=34.0,
    )

    assert features.shape == (26,)
    assert len(FEATURE_NAMES) == 26
    assert np.allclose(features[:5], [7.0, 0.85, 0.375, 0.85, 0.375])
    assert np.allclose(features[8:12], [0.4, 0.0, 0.0, 0.2], atol=1e-6)
    assert np.allclose(features[13:18], [8.0, 1.7, 0.75, 1.7, 0.75])
    assert np.allclose(features[21:25], [0.8, 0.0, 0.0, 0.4], atol=1e-6)


def test_frozen_sidecar_validates_and_uses_the_exact_linear_square_design() -> None:
    payload = json.loads(SIDECAR.read_text(encoding="utf-8"))
    calibration = validate_public_force_calibration(payload)
    features = np.linspace(-2.0, 3.0, 26)

    normalized = (features - calibration["median"]) / calibration["iqr"]
    expected = np.clip(
        np.concatenate((normalized, normalized * normalized)) @ calibration["weights"]
        + calibration["bias"],
        *calibration["force_range_N"],
    )
    assert calibration["calibration_id"] == "tension_strap_public_tactile_force_69seed.v1"
    assert calibration["weights"].shape == (52,)
    assert predict_public_tactile_force(calibration, features) == expected
    text = SIDECAR.read_text(encoding="utf-8")
    assert "private_force_labels" not in text
    assert "true_tension_N" not in text


def test_public_depth_marker_mode_updates_per_step_without_private_force(monkeypatch) -> None:
    env = _PublicObservationEnv()
    api = OpenTacApi(env)
    monkeypatch.setattr(api, "_publish_live_stage_response", lambda: None)

    baseline = api.begin_tactile_tension_estimator()
    env.tick()
    estimate = api.get_tactile_tension_estimate()

    assert baseline["started"] is True
    assert estimate["available"] is True
    assert estimate["measurement"] == {
        "mode": ESTIMATOR_MODE,
        "feature_dimension": 26,
    }
    assert "true_tension" not in repr(estimate).lower()
    assert "private" not in repr(estimate).lower()
    assert env.diagnostics["estimator_mode"] == ESTIMATOR_MODE
    assert env.diagnostics["calibration_id"] == "tension_strap_public_tactile_force_69seed.v1"
    assert env.estimator_trace[-1]["estimator"]["mode"] == ESTIMATOR_MODE
    assert env.estimator_trace[-1]["public_tactile_feature_dimension"] == 26
    audit = env.private_force_audit[-1]
    assert audit["status"] == "paired"
    assert audit["true_tension_N"] == 12.25
    assert audit["estimated_tension_N"] == estimate["estimated_tension_N"]
    assert audit["error_N"] == audit["estimated_tension_N"] - audit["true_tension_N"]
    assert "true_tension" not in api.runtime_memory_context().lower()
    assert "private_force" not in api.functions()

    env.missing_marker = True
    env.tick()
    unavailable = api.get_tactile_tension_estimate()
    assert unavailable["available"] is False
    assert unavailable["estimated_tension_N"] is None
    assert env.diagnostics["private_debug"]["last_failure"]["status"] == "public_tactile_unavailable"
    assert env.private_force_audit[-1]["status"] == "estimate_unavailable"
