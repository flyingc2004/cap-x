"""Isaac-free coverage for OpenTac's per-task-step marker-flow estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from capx.integrations.opentac import OpenTacApi
from capx.integrations.univtac.franka_compat_api import UniVTACFrankaCompatApi


class _ObserverEnv:
    def __init__(self) -> None:
        self.step = 0
        self.tension = 0.0
        self.fail_tracking = False
        self.api_configs = {
            "opentac_api": {
                "stage_targets_N": [12.0, 18.0],
                "stage_tolerance_N": 0.5,
                "observation_dt_s": 0.1,
                "stage_hold_seconds": 0.3,
                "estimator_update_stride": 1,
                "max_estimate_age_steps": 3,
            }
        }
        self.post_step: dict[str, Any] = {}
        self.post_action: dict[str, Any] = {}
        self.trace: list[dict[str, Any]] = []
        self.diagnostics: dict[str, Any] = {}

    def get_step_count(self) -> int:
        return self.step

    def register_post_step_observer(self, name: str, observer: Any) -> None:
        self.post_step[name] = observer

    def unregister_post_step_observer(self, name: str) -> None:
        self.post_step.pop(name, None)

    def register_post_action_observer(self, name: str, observer: Any) -> None:
        self.post_action[name] = observer

    def unregister_post_action_observer(self, name: str) -> None:
        self.post_action.pop(name, None)

    def append_tactile_working_memory_trace(self, record: dict[str, Any]) -> None:
        self.trace.append(record)

    def set_opentac_tension_estimator_diagnostics(self, record: dict[str, Any]) -> None:
        self.diagnostics = record

    def tick(self, tension: float) -> None:
        self.step += 1
        self.tension = float(tension)
        for observer in tuple(self.post_step.values()):
            observer()

    def finish_action(self, action_type: str = "qpos", *, ok: bool = True) -> None:
        result = {"ok": ok, "step": self.step, "action_count": self.step}
        for observer in tuple(self.post_action.values()):
            observer(action_type, result)


class _Tracker:
    def __init__(self, reference: dict[str, np.ndarray]) -> None:
        self.reference = reference
        self.update_count = 0


class _Utilities:
    MarkerFlowTracker = _Tracker

    @staticmethod
    def tracked_flow_rgb_features(
        reference: dict[str, np.ndarray], current: dict[str, np.ndarray], tracker: _Tracker
    ) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
        assert tracker.reference is reference
        tracker.update_count += 1
        value = float(current["left_tactile"][0, 0, 0])
        if value < 0.0:
            raise RuntimeError("marker correspondences unavailable")
        return np.asarray([value], dtype=float), {
            "left_tactile": {"matched": 64, "p90_px": 0.8},
            "right_tactile": {"matched": 64, "p90_px": 0.8},
        }

    @staticmethod
    def predict_calibrated(_calibration: dict[str, Any], features: np.ndarray) -> float:
        return float(features[0])


def _api(monkeypatch, env: _ObserverEnv) -> OpenTacApi:
    api = OpenTacApi(env)

    def marker_images() -> dict[str, np.ndarray]:
        value = -1.0 if env.fail_tracking else env.tension
        image = np.full((4, 4, 3), value, dtype=np.float32)
        return {"left_tactile": image, "right_tactile": image.copy()}

    monkeypatch.setattr(api, "_read_marker_images", marker_images)
    monkeypatch.setattr(api, "_load_tension_calibration", lambda: {"weights": [1.0]})
    monkeypatch.setattr(api, "_force_task_utilities", lambda: _Utilities)
    return api


def test_estimator_updates_every_task_step_without_owning_control_progress(monkeypatch) -> None:
    env = _ObserverEnv()
    api = _api(monkeypatch, env)

    baseline = api.begin_tactile_tension_estimator()
    assert baseline["started"] is True
    assert len(env.post_step) == 1
    assert env.post_action == {}

    for _ in range(3):
        env.tick(12.0)
    estimate = api.get_tactile_tension_estimate()
    assert estimate["available"] is True
    assert estimate["estimated_tension_N"] == 12.0
    assert api._tracker.update_count == 3  # One flow update per task step.
    assert "completed_stage_ids" not in env.diagnostics
    assert "current_stage_index" not in env.diagnostics
    assert "consecutive_invalid_samples" not in env.diagnostics


def test_control_contract_exposes_exact_stage_bands(monkeypatch) -> None:
    env = _ObserverEnv()
    api = _api(monkeypatch, env)

    contract = api.get_tactile_tension_control_contract()

    assert contract["stage_targets_N"] == [12.0, 18.0]
    assert contract["stage_bands_N"] == [[11.5, 12.5], [17.5, 18.5]]
    assert contract["sampling_cadence_steps"] == contract["observation_wait_steps"]
    assert contract["sampling_cadence_seconds"] == pytest.approx(
        contract["observation_wait_steps"] * contract["observation_dt_s"]
    )
    assert "stage_tolerance_N" not in contract
    assert "max_control_actions_per_stage" not in contract
    assert "max_consecutive_invalid_samples" not in contract


def test_estimate_is_cached_and_stale_without_new_task_steps(monkeypatch) -> None:
    env = _ObserverEnv()
    api = _api(monkeypatch, env)
    api.begin_tactile_tension_estimator()
    assert api.get_tactile_tension_estimate()["available"] is False
    assert api._tracker.update_count == 0
    env.tick(12.0)
    tracker_updates = api._tracker.update_count

    estimate = api.get_tactile_tension_estimate()
    assert estimate["available"] is True
    assert api._tracker.update_count == tracker_updates

    env.step += 4
    stale = api.get_tactile_tension_estimate()
    assert stale["available"] is False
    assert stale["estimated_tension_N"] is None
    assert stale["observation_age_steps"] == 4
    assert api._tracker.update_count == tracker_updates


def test_restarting_an_active_estimator_preserves_the_marker_baseline(monkeypatch) -> None:
    env = _ObserverEnv()
    api = _api(monkeypatch, env)
    api.begin_tactile_tension_estimator()
    env.tick(12.0)

    resumed = api.begin_tactile_tension_estimator()

    assert resumed["started"] is True
    assert api._tracker.update_count == 1


def test_multiple_steps_in_one_action_keep_marker_updates_incremental(monkeypatch) -> None:
    env = _ObserverEnv()
    api = _api(monkeypatch, env)
    api.begin_tactile_tension_estimator()

    for tension in (4.0, 8.0, 12.0):
        env.tick(tension)
    estimate = api.get_tactile_tension_estimate()
    assert estimate["available"] is True
    assert estimate["estimated_tension_N"] == 12.0
    assert api._tracker.update_count == 3


def test_tracking_unavailable_is_only_a_temporary_numeric_observation(monkeypatch) -> None:
    env = _ObserverEnv()
    api = _api(monkeypatch, env)
    preview_updates: list[int] = []
    monkeypatch.setattr(api, "_publish_live_stage_response", lambda: preview_updates.append(env.step))
    api.begin_tactile_tension_estimator()
    env.fail_tracking = True

    for _ in range(3):
        env.tick(0.0)
    estimate = api.get_tactile_tension_estimate()
    assert estimate["available"] is False
    assert estimate["estimated_tension_N"] is None
    assert "status" not in estimate
    assert "message" not in estimate

    context = api.runtime_memory_context()
    assert '"available":false' in context
    assert "consecutive_invalid_samples" not in context
    assert "true_tension" not in context
    assert "actor" not in context
    assert preview_updates == [1, 2, 3]
    assert env.diagnostics["private_debug"] == {
        "valid_sample_count": 0,
        "invalid_sample_count": 3,
        "last_failure": {
            "status": "tracking_unavailable",
            "message": "marker correspondences unavailable",
            "step": 3,
        },
    }

    api.reset_episode()
    assert env.post_step == {}
    assert env.diagnostics["private_debug"] == {
        "valid_sample_count": 0,
        "invalid_sample_count": 0,
        "last_failure": None,
    }
    with np.testing.assert_raises_regex(RuntimeError, "begin_tactile_tension_estimator"):
        api.get_tactile_tension_estimate()


def test_force_task_move_splits_one_public_delta_into_marker_trackable_segments() -> None:
    class _ForceTaskEnv:
        def __init__(self) -> None:
            self.deltas: list[float] = []

        def move_force_task_vertical_delta(self, *, dz: float) -> dict[str, Any]:
            self.deltas.append(dz)
            return {"ok": True, "step": len(self.deltas)}

    env = _ForceTaskEnv()
    api = UniVTACFrankaCompatApi.__new__(UniVTACFrankaCompatApi)
    api._env = env
    api.local_delta_max_m = 0.002
    api.local_delta_segment_m = 0.001
    api.min_safe_z = 0.1
    api._current_tool_pose = lambda: (
        np.asarray([0.5, 0.0, 0.2], dtype=float),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    )

    result = api._opentac_tension_move_delta(0.002)

    assert np.allclose(env.deltas, [0.001, 0.001])
    assert result["segment_count"] == 2
    assert result["segments_completed"] == 2
    assert np.isclose(result["executed_delta_z_m"], 0.002)
