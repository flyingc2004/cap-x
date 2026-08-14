from __future__ import annotations

from pathlib import Path
import time
import types

import numpy as np
import pytest
import yaml

from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.envs.tasks.exceptions import HardStopTrial, RecoverableTaskFailure
from capx.envs.runner import _parse_trial_ids
from capx.envs.simulators.univtac import UniVTACLowLevelEnv
from capx.integrations.univtac.franka_compat_api import UniVTACFrankaCompatApi
from capx.integrations.univtac.rgbd_perception import (
    GraspEstimate,
    ObjectEstimate,
    RgbdPerceptionError,
    RgbdFrame,
    UniVTACRgbdPerception,
)


def _frame(*, valid_depth: bool = True) -> RgbdFrame:
    depth = np.ones((8, 8), dtype=np.float32)
    if not valid_depth:
        depth[:] = np.nan
    return RgbdFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=depth,
        intrinsics=np.array(
            [[100.0, 0.0, 3.5], [0.0, 100.0, 3.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        camera_position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        camera_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def _mask() -> np.ndarray:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    return mask


def test_rgbd_mask_deprojection_and_world_obb(monkeypatch) -> None:
    perception = UniVTACRgbdPerception(min_depth_points=8)
    monkeypatch.setattr(perception, "_segment", lambda rgb, prompt: (_mask(), 0.9))

    estimate = perception.estimate_object(_frame(), "cylindrical can")

    assert len(estimate.points_world) == 16
    np.testing.assert_allclose(estimate.position, [1.0, 2.0, 4.0], atol=1e-6)
    assert estimate.extent.shape == (3,)
    assert estimate.quaternion_wxyz.shape == (4,)
    assert estimate.score == pytest.approx(0.9)


def test_parse_selected_trial_ids() -> None:
    assert _parse_trial_ids("8,13,20-22,13") == [8, 13, 20, 21, 22]
    assert _parse_trial_ids("") is None
    with pytest.raises(ValueError):
        _parse_trial_ids("5-3")


def test_rgbd_grasp_camera_to_world_transform(monkeypatch) -> None:
    perception = UniVTACRgbdPerception(
        min_depth_points=8,
        grasp_local_z_offset=0.12,
    )
    monkeypatch.setattr(perception, "_segment", lambda rgb, prompt: (_mask(), 0.8))
    grasps = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    grasps[0, :3, 3] = [0.0, 0.0, 1.0]
    grasps[1, :3, 3] = [0.1, 0.2, 1.0]
    monkeypatch.setattr(
        perception,
        "_request_grasps",
        lambda depth, intrinsics, mask: (grasps, np.array([0.1, 0.9])),
    )

    estimate = perception.estimate_grasp(_frame(), "cylindrical can")

    assert estimate.selected_index == 1
    np.testing.assert_allclose(estimate.position, [1.1, 2.2, 4.12], atol=1e-6)
    np.testing.assert_allclose(estimate.quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert estimate.points_world.shape == (16, 3)
    np.testing.assert_allclose(estimate.object_position, [1.0, 2.0, 4.0], atol=1e-6)


def test_rgbd_perception_failures_do_not_fallback(monkeypatch) -> None:
    perception = UniVTACRgbdPerception(min_depth_points=8)
    monkeypatch.setattr(perception, "_segment", lambda rgb, prompt: (_mask(), 0.8))
    with pytest.raises(RgbdPerceptionError, match="valid depth points"):
        perception.estimate_object(_frame(valid_depth=False), "can")

    monkeypatch.setattr(
        perception,
        "_request_grasps",
        lambda depth, intrinsics, mask: (np.empty((0, 4, 4)), np.empty((0,))),
    )
    with pytest.raises(RgbdPerceptionError, match="invalid grasps"):
        perception.estimate_grasp(_frame(), "can")


def test_rgbd_regrasp_failure_becomes_recoverable_task_failure() -> None:
    class Env:
        def __init__(self) -> None:
            self.api_configs = {
                "franka_control_api": {
                    "rgbd_perception_enabled": True,
                    "object_pose_names": {"can": "can"},
                    "perception_prompt_map": {"can": "cylindrical can"},
                    "perception_prompt_fallbacks": {"can": ["can"]},
                    "perception_retry_attempts": 1,
                }
            }
            self.artifacts = []

        def get_rgbd_frame(self, camera_name):
            return _frame(valid_depth=False)

        def append_perception_artifact(self, record):
            self.artifacts.append(record)

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    with pytest.raises(RecoverableTaskFailure) as exc_info:
        api.sample_grasp_pose("can")

    assert exc_info.value.reason == "rgbd_regrasp_unavailable"
    assert exc_info.value.details["kind"] == "grasp_pose"
    assert env.artifacts
    assert env.artifacts[0]["kind"] == "grasp_pose_error"
    assert "reason" in env.artifacts[0]


def test_rgbd_regrasp_failure_can_use_official_anchor_fallback() -> None:
    class Env:
        def __init__(self) -> None:
            self.api_configs = {
                "franka_control_api": {
                    "rgbd_perception_enabled": True,
                    "official_anchor_fallback_enabled": True,
                    "official_anchor_fallback_objects": ["can"],
                    "object_pose_names": {"can": "can"},
                    "perception_prompt_map": {"can": "cylindrical can"},
                    "perception_prompt_fallbacks": {"can": ["can"]},
                    "perception_retry_attempts": 1,
                }
            }
            self.artifacts = []

        def get_rgbd_frame(self, camera_name):
            return _frame(valid_depth=False)

        def append_perception_artifact(self, record):
            self.artifacts.append(record)

        def get_public_grasp_pose(self, object_name, *, grasp_height):
            assert object_name == "can"
            return (
                np.array([0.55, 0.0, grasp_height], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    pos, quat = api.sample_grasp_pose("can")

    np.testing.assert_allclose(pos, [0.55, 0.0, 0.04], atol=1e-6)
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    assert [record["kind"] for record in env.artifacts[:-1]] == [
        "grasp_pose_error",
        "grasp_pose_error",
    ]
    assert env.artifacts[-1]["kind"] == "grasp_pose"
    assert env.artifacts[-1]["source"] == "official_anchor_fallback"
    assert env.artifacts[-1]["used_for_control"] is True


def test_rgbd_object_pose_failure_does_not_use_official_anchor_fallback() -> None:
    class Env:
        def __init__(self) -> None:
            self.api_configs = {
                "franka_control_api": {
                    "rgbd_perception_enabled": True,
                    "official_anchor_fallback_enabled": True,
                    "official_anchor_fallback_objects": ["can"],
                    "object_pose_names": {"can": "can"},
                    "perception_prompt_map": {"can": "cylindrical can"},
                    "perception_retry_attempts": 1,
                }
            }
            self.artifacts = []

        def get_rgbd_frame(self, camera_name):
            return _frame(valid_depth=False)

        def append_perception_artifact(self, record):
            self.artifacts.append(record)

        def get_public_grasp_pose(self, object_name, *, grasp_height):
            return (
                np.array([0.55, 0.0, grasp_height], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    with pytest.raises(RecoverableTaskFailure) as exc_info:
        api.get_object_pose("can")

    assert exc_info.value.reason == "rgbd_pose_unavailable"
    assert env.artifacts
    assert all(record["source"] != "official_anchor_fallback" for record in env.artifacts)


def test_recoverable_task_failure_does_not_mark_code_execution_failed() -> None:
    env = CodeExecutionEnvBase.__new__(CodeExecutionEnvBase)
    env.low_level_env = types.SimpleNamespace(get_observation=lambda: {})
    env._apis = {}
    env._full_prompt = []
    env._init_exec_globals()

    result = env._exec_user_code(
        "from capx.envs.tasks.exceptions import RecoverableTaskFailure\n"
        "raise RecoverableTaskFailure('rgbd_regrasp_unavailable', 'retry failed')\n"
    )

    assert result["ok"] is True
    assert result["stderr"] == ""
    assert "recoverable_failure reason=rgbd_regrasp_unavailable" in result["stdout"]
    assert result["result"]["recoverable_task_failure"] is True


def test_hard_stop_trial_escapes_code_execution() -> None:
    env = CodeExecutionEnvBase.__new__(CodeExecutionEnvBase)
    env.low_level_env = types.SimpleNamespace(get_observation=lambda: {})
    env._apis = {}
    env._full_prompt = []
    env._init_exec_globals()

    with pytest.raises(HardStopTrial):
        env._exec_user_code(
            "from capx.envs.tasks.exceptions import HardStopTrial\n"
            "raise HardStopTrial('trial_timeout', 'deadline hit')\n"
        )


def test_goto_pose_action_failure_returns_status() -> None:
    class Env:
        api_configs = {
            "franka_control_api": {
                "min_safe_z": 0.0,
                "max_delta_xyz": 1.0,
                "preserve_landmark_orientation": False,
            }
        }
        task = None

        def get_robot_state(self):
            return {
                "ee_pos": [0.0, 0.0, 0.2],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
            }

        def take_action(self, action, *, action_type):
            return {"ok": False, "message": "planner rejected bounded step"}

    api = UniVTACFrankaCompatApi(Env())
    result = api.goto_pose(
        np.array([0.1, 0.0, 0.2], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert result["ok"] is False
    assert result["message"] == "planner rejected bounded step"


def test_goto_pose_api_action_limit_blocks_before_physical_action() -> None:
    class Env:
        api_configs = {
            "franka_control_api": {
                "min_safe_z": 0.0,
                "max_delta_xyz": 0.01,
                "max_goto_pose_actions": 3,
                "preserve_landmark_orientation": False,
            }
        }
        task = None

        def __init__(self) -> None:
            self.actions = []

        def get_robot_state(self):
            return {
                "ee_pos": [0.0, 0.0, 0.2],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
            }

        def take_action(self, action, *, action_type):
            self.actions.append((np.asarray(action), action_type))
            return {"ok": True}

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    result = api.goto_pose(
        np.array([0.10, 0.0, 0.2], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert result["ok"] is False
    assert result["reason"] == "api_action_limit"
    assert result["requested_actions"] > result["max_api_actions"]
    assert result["max_api_actions"] == 3
    assert env.actions == []


def test_adaptive_gripper_max_steps_are_clamped() -> None:
    class Env:
        api_configs = {"franka_control_api": {"max_gripper_servo_steps": 5}}
        task = None

        def begin_high_level_action(self):
            return True

        def finalize_high_level_action(self):
            return None

    class Controller:
        trace = []

        def close(self, *, target_force, max_steps):
            return {
                "ok": True,
                "stable": False,
                "contact": False,
                "reason": "max_steps",
                "normal_force": 0.0,
                "width": 0.0,
                "steps": max_steps,
            }

    api = UniVTACFrankaCompatApi(Env())
    api._adaptive_gripper_controller = lambda: Controller()
    api._save_adaptive_trace = lambda trace: None

    result = api.close_gripper(adaptive=True, max_steps=999)

    assert result["steps"] == 5
    assert result["requested_max_steps"] == 999
    assert result["max_steps_limit"] == 5


def test_official_compat_uses_rgbd_and_never_reads_task_can() -> None:
    class ForbiddenActor:
        def get_pose(self):
            raise AssertionError("task.can ground-truth pose must not be read")

    class Env:
        task = types.SimpleNamespace(can=ForbiddenActor())

        def __init__(self) -> None:
            self.api_configs = {
                "franka_control_api": {
                    "rgbd_perception_enabled": True,
                    "use_native_pose_planner": True,
                    "use_task_grasp_actor_for_objects": False,
                    "object_pose_names": {
                        "can": "can",
                        "object": "can",
                        "target object": "can",
                    },
                    "perception_prompt_map": {"can": "cylindrical can"},
                }
            }
            self.native_moves = []
            self.artifacts = []
            self.finalized = 0

        def get_rgbd_frame(self, camera_name):
            assert camera_name == "head"
            return _frame()

        def append_perception_artifact(self, record):
            self.artifacts.append(record)

        def move_to_tool_pose_native(self, position, quaternion):
            self.native_moves.append((np.asarray(position), np.asarray(quaternion)))
            return {"ok": True}

        def finalize_high_level_action(self):
            self.finalized += 1

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    mask = _mask()
    points = np.zeros((16, 3), dtype=np.float32)
    object_estimate = ObjectEstimate(
        position=np.array([0.7, 0.0, 0.03], dtype=np.float32),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        extent=np.array([0.06, 0.06, 0.12], dtype=np.float32),
        mask=mask,
        points_world=points,
        score=0.9,
        prompt="cylindrical can",
    )
    grasp_estimate = GraspEstimate(
        position=np.array([0.64, 0.0, 0.04], dtype=np.float32),
        quaternion_wxyz=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        mask=mask,
        points_world=points,
        object_position=object_estimate.position,
        object_quaternion_wxyz=object_estimate.quaternion_wxyz,
        object_extent=object_estimate.extent,
        scores=np.array([0.9], dtype=np.float32),
        grasps_camera=np.eye(4, dtype=np.float32)[None],
        selected_index=0,
        prompt="cylindrical can",
    )
    api._rgbd_perception = types.SimpleNamespace(
        estimate_object=lambda frame, prompt: object_estimate,
        estimate_grasp=lambda frame, prompt: grasp_estimate,
    )

    pos, quat = api.get_object_pose("can")
    np.testing.assert_allclose(pos, object_estimate.position)
    np.testing.assert_allclose(quat, object_estimate.quaternion_wxyz)
    grasp_pos, grasp_quat = api.sample_grasp_pose("can")
    np.testing.assert_allclose(grasp_pos, grasp_estimate.position)
    np.testing.assert_allclose(grasp_quat, grasp_estimate.quaternion_wxyz)

    api.goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
    assert len(env.native_moves) == 2
    assert env.finalized == 1
    assert [record["source"] for record in env.artifacts] == [
        "rgbd",
        "rgbd_contact_graspnet",
    ]


def test_official_protocol_budget_and_early_stop() -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._official_task_protocol = True
    env._protocol_stopped = False
    env._protocol_stop_reason = None
    env._protocol_success_latched = False
    env.max_steps = 300
    env._task = types.SimpleNamespace(
        take_action_cnt=0,
        logger=None,
        eval_success=False,
        check_success=lambda: False,
        check_early_stop=lambda: True,
    )

    assert env.begin_high_level_action() is True
    assert env.get_action_count() == 1
    status = env.finalize_high_level_action()
    assert status["stopped"] is True
    assert status["reason"] == "early_stop"
    assert env.protocol_action_allowed() is False


def test_official_success_latches_and_blocks_later_actions() -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._official_task_protocol = True
    env._protocol_stopped = False
    env._protocol_stop_reason = None
    env._protocol_success_latched = False
    env.max_steps = 300
    env._task = types.SimpleNamespace(
        take_action_cnt=300,
        step_count=123,
        logger=None,
        eval_success=False,
        check_success=lambda: True,
        check_early_stop=lambda: True,
    )

    status = env.finalize_high_level_action()

    assert status["stopped"] is True
    assert status["reason"] == "native_success"
    assert status["success_latched"] is True
    assert env._task.eval_success is True

    env._task.check_success = lambda: False
    assert env.task_completed() is True
    assert env.compute_reward() == 1.0

    blocked = env._protocol_blocked_result()
    assert blocked["ok"] is True
    assert blocked["reason"] == "native_success"
    assert blocked["episode_stopped"] is True
    assert blocked["success_latched"] is True


def test_univtac_deadline_hard_stops_before_next_action() -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._official_task_protocol = True
    env._protocol_stopped = False
    env._protocol_stop_reason = None
    env._protocol_success_latched = False
    env._trial_deadline_seconds = 1.0
    env._trial_deadline_time = time.monotonic() - 0.01
    env.max_steps = 300
    env._task = types.SimpleNamespace(
        take_action_cnt=12,
        step_count=456,
        plan_success=True,
    )

    with pytest.raises(HardStopTrial) as exc_info:
        env.protocol_action_allowed()

    assert exc_info.value.reason == "trial_timeout"
    assert env._protocol_stopped is True
    assert env._protocol_stop_reason == "trial_timeout"
    assert env._task.plan_success is False


def test_perception_artifacts_are_saved_under_trial_directory(tmp_path) -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._perception_artifacts = [
        {
            "kind": "diagnostic_grasp_candidates",
            "source": "rgbd_contact_graspnet_private_diagnostic",
            "frame": _frame(),
            "mask": _mask(),
            "points_world": np.zeros((16, 3), dtype=np.float32),
            "grasps_camera": np.eye(4, dtype=np.float32)[None],
            "grasp_scores": np.array([0.9], dtype=np.float32),
            "selected_index": 0,
            "used_for_control": False,
        }
    ]

    manifest = env._export_perception_artifacts(tmp_path)

    assert manifest == tmp_path / "perception/manifest.json"
    assert (tmp_path / "perception/00_diagnostic_grasp_candidates_rgb.png").exists()
    assert (tmp_path / "perception/00_diagnostic_grasp_candidates_depth.png").exists()
    assert (tmp_path / "perception/00_diagnostic_grasp_candidates_mask.png").exists()
    assert (tmp_path / "perception/00_diagnostic_grasp_candidates_points_world.npz").exists()
    assert (tmp_path / "perception/00_diagnostic_grasp_candidates_grasps.npz").exists()


def test_official_yaml_uses_native_protocol_without_privileged_pose() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repo / "env_configs/univtac/lift_can_tactile_official.yaml").read_text()
    )
    cfg = config["env"]["cfg"]
    low = cfg["low_level"]
    franka = low["api_configs"]["franka_control_api"]
    assert low["task_name"] == "lift_can"
    assert low["task_config"] == "smoke_capx_lift_can_official"
    assert low["expose_actor_pose"] is False
    assert low["privileged"] is False
    assert cfg["apis"] == ["FrankaControlApi", "UniVTACTactileApi"]
    assert franka["rgbd_perception_enabled"] is True
    assert franka["perception_retry_attempts"] == 2
    assert franka["use_native_pose_planner"] is False
    assert franka["use_task_grasp_actor_for_objects"] is False
    assert franka["record_perception_diagnostic"] is True
    assert franka["official_anchor_fallback_enabled"] is True
    assert franka["official_anchor_fallback_objects"] == ["can"]
    assert franka["max_goto_pose_actions"] == 40
    assert franka["max_home_pose_actions"] == 15
    assert franka["max_native_pose_actions"] == 2
    assert franka["max_gripper_servo_steps"] == 200
    assert franka["max_gripper_settle_steps"] == 20
    assert franka["home_pose_relative_lift"] is True
    assert franka["home_lift_delta_z"] == pytest.approx(0.10)
    assert franka["max_delta_xyz"] == pytest.approx(0.01)
    assert "api_servers" not in config
    assert "stable=True" in cfg["prompt"]
    assert "official pre-grasp state" in cfg["prompt"]
    assert "RGB-D" in cfg["prompt"]
    assert "sample_grasp_pose(\"can\")" in cfg["prompt"]
    assert "If perception or motion fails" in cfg["prompt"]
    assert "relative=True" not in cfg["prompt"]

    task_config = yaml.safe_load(
        Path(
            "/mnt/sdc/ljz/UniVTAC/task_config/smoke_capx_lift_can_official.yml"
        ).read_text()
    )
    assert task_config["skip_task_pre_move"] is False
    assert task_config["official_task_protocol"] is True
    assert task_config["step_lim"] == 300
    assert task_config["observations"]["camera"] == ["rgb", "depth"]


def test_insert_hole_official_yaml_uses_pre_move_and_move_relative() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repo / "env_configs/univtac/insert_hole_tactile_official.yaml").read_text()
    )
    cfg = config["env"]["cfg"]
    low = cfg["low_level"]
    franka = low["api_configs"]["franka_control_api"]
    prompt = cfg["prompt"]

    assert low["task_name"] == "insert_hole"
    assert low["task_config"] == "smoke_capx_insert_hole_official"
    assert low["expose_actor_pose"] is False
    assert low["privileged"] is False
    assert cfg["apis"] == ["FrankaControlApi", "UniVTACTactileApi"]
    assert franka["rgbd_perception_enabled"] is False
    assert franka["use_task_grasp_actor_for_objects"] is False
    assert franka["max_insert_actions"] == 20
    assert franka["move_relative_lateral_budget"] == pytest.approx(0.006)
    assert franka["move_relative_rotation_budget"] == pytest.approx(1.20)
    assert franka["tactile_guard"]["slip_warning_threshold"] == pytest.approx(0.35)
    assert franka["tactile_guard"]["guard_micro_down_step"] == pytest.approx(0.001)
    assert config["trial_timeout_seconds"] == 600
    assert "move_relative" in prompt
    assert "slip_risk" in prompt
    assert "preempted_by_slip_risk" in prompt
    assert "interrupted_by_slip_warning" in prompt
    assert "correction_hint" in prompt
    assert "remaining_actions" in prompt
    assert "success_latched" in prompt
    assert "episode_stopped" in prompt
    assert "recoverable tactile guard signals" in prompt
    assert "normal guarded downward insertion steps should be 0.001 m" in prompt
    assert "fallback/test downward steps can be 0.0003-0.0005 m" in prompt
    assert "once a useful direction is found" in prompt
    assert "no larger than 0.0005 m" in prompt
    assert "no larger than 0.006 rad" in prompt
    assert 'result["limits"]' in prompt
    assert 'result["risk_before"]' in prompt
    assert 'result["risk_after"]' in prompt
    assert 'result["reason"]' in prompt
    assert 'result["tactile"]' in prompt
    assert "finds succeed" in prompt
    assert "Do not call get_tactile_summary again to override" in prompt
    assert 'reason="completed_with_slip" can still be a useful correction' in prompt
    assert "reduces high risk to medium" in prompt
    assert "Never repeat the same pitch or lateral direction twice" in prompt
    assert 'result["executed_depth"] > 0' in prompt
    assert "protected insertion progress" in prompt
    assert "correction-budget exhaustion" in prompt
    assert 'never to "budget_exhausted"' in prompt
    assert "remaining_actions <= 0" in prompt
    assert "no_progress" in prompt
    assert "SystemExit" in prompt
    assert "insert_along_axis" not in prompt
    assert "open_gripper()" in prompt
    assert "home_pose()" in prompt
    for forbidden in (
        "target_pose",
        "hole_pose",
        "task.",
        "metadata",
        "reward",
        "task_completed",
        "trial id",
    ):
        assert forbidden not in prompt

    task_config = yaml.safe_load(
        Path(
            "/mnt/sdc/ljz/UniVTAC/task_config/smoke_capx_insert_hole_official.yml"
        ).read_text()
    )
    assert task_config["skip_task_pre_move"] is False
    assert task_config["official_task_protocol"] is True
    assert task_config["step_lim"] == 600
    assert task_config["record_pre_move_frames"] is True
    assert task_config["record_action_frames"] is True
    assert task_config["record_action_tactile"] is True
    assert task_config["observations"]["camera"] == ["rgb", "depth"]


def _move_tactile_summary(
    *,
    contact: bool = True,
    event: str = "stable_grasp",
    slip_score: float = 0.0,
    normal_force: float = 0.6,
    left_force: float = 0.6,
    right_force: float = 0.6,
    drift: float = 0.0,
    slip_risk: str = "low",
    incipient_slip: bool = False,
    pressure_side: str | None = None,
    shear_side: str = "balanced",
    drift_trend: str = "stable",
    correction_hint: str = "continue",
) -> dict[str, object]:
    if pressure_side is None:
        if left_force - right_force > 0.05:
            pressure_side = "left"
        elif right_force - left_force > 0.05:
            pressure_side = "right"
        else:
            pressure_side = "balanced"
    return {
        "contact": contact,
        "left_contact": contact,
        "right_contact": contact,
        "normal_force": normal_force,
        "contact_balance": left_force - right_force,
        "event": event,
        "slip_score": slip_score,
        "slip_risk": slip_risk,
        "incipient_slip": incipient_slip,
        "pressure_side": pressure_side,
        "shear_side": shear_side,
        "force_change": "stable",
        "drift_trend": drift_trend,
        "correction_hint": correction_hint,
        "marker_centroid_displacement": drift,
        "left": {"normal_force": left_force},
        "right": {"normal_force": right_force},
    }


def _summary_reader(*summaries: dict[str, object]):
    values = list(summaries) or [_move_tactile_summary()]
    index = 0

    def read() -> dict[str, object]:
        nonlocal index
        value = values[min(index, len(values) - 1)]
        index += 1
        return value

    return read


def test_move_relative_returns_depth_and_simple_tactile_feedback() -> None:
    class Env:
        api_configs = {
            "franka_control_api": {
                "max_delta_xyz": 0.01,
                "max_insert_actions": 20,
            }
        }
        task = None

        def __init__(self) -> None:
            self.actions = []
            self.protocol_reason = None
            self.finalize_reason = None

        def get_robot_state(self):
            return {
                "ee_pos": [0.0, 0.0, 0.2],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
            }

        def take_action(self, action, *, action_type):
            self.actions.append((np.asarray(action, dtype=np.float32), action_type))
            return {"ok": True, "message": "action executed"}

        def finalize_high_level_action(self):
            if self.finalize_reason is not None:
                self.protocol_reason = self.finalize_reason
            stopped = self.protocol_reason is not None
            return {
                "enabled": True,
                "stopped": stopped,
                "reason": self.protocol_reason,
                "episode_stopped": stopped,
                "success_latched": self.protocol_reason == "native_success",
            }

        def get_protocol_status(self):
            return {
                "enabled": True,
                "stopped": self.protocol_reason is not None,
                "reason": self.protocol_reason,
                "episode_stopped": self.protocol_reason is not None,
                "success_latched": self.protocol_reason == "native_success",
                "action_count": len(self.actions),
                "max_steps": 300,
            }

    env = Env()
    api = UniVTACFrankaCompatApi(env)

    api._read_adaptive_tactile_summary = _summary_reader(
        _move_tactile_summary(normal_force=0.5, left_force=0.5, right_force=0.5),
        _move_tactile_summary(normal_force=0.6, left_force=0.7, right_force=0.5),
    )

    result = api.move_relative([0.0, 0.0, -0.003], tactile_guard=True)

    assert result["ok"] is True
    assert result["reason"] == "completed"
    assert result["executed_distance"] == pytest.approx(0.003, abs=1e-6)
    assert result["executed_depth"] == pytest.approx(0.003, abs=1e-6)
    assert len(env.actions) == 6
    action, action_type = env.actions[-1]
    assert action_type == "delta_ee"
    np.testing.assert_allclose(action[:3], [0.0, 0.0, -0.0005], atol=1e-7)
    np.testing.assert_allclose(action[3:], 0.0, atol=1e-7)
    assert result["tactile"] == {
        "contact": True,
        "stable": True,
        "slip": False,
        "slip_risk": "low",
        "incipient_slip": False,
        "pressure_side": "left",
        "shear_side": "balanced",
        "heavier_side": "left",
        "force_change": "increased",
        "drift_trend": "stable",
        "drift": "low",
        "correction_hint": "continue",
    }
    assert result["preempted"] is False
    assert result["interrupted"] is False
    assert result["remaining_actions"] == 294
    assert result["tactile"]["pressure_side"] > 0.3
    assert result["tactile"]["force_change"] > 0.3

    env.actions.clear()
    api._read_adaptive_tactile_summary = _summary_reader(
        _move_tactile_summary(),
        _move_tactile_summary(),
    )

    result = api.move_relative(
        [0.0004, 0.0, 0.0],
        delta_rpy=[0.0, 0.004, 0.0],
        tactile_guard=True,
    )

    assert result["ok"] is True
    assert result["executed_distance"] == pytest.approx(0.0004, abs=1e-6)
    assert result["executed_depth"] == pytest.approx(0.0, abs=1e-6)
    action, action_type = env.actions[0]
    assert action_type == "delta_ee"
    np.testing.assert_allclose(action[:3], [0.0004, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(action[3:6], [0.0, 0.004, 0.0], atol=1e-7)


def test_move_relative_success_latch_is_ok_and_noops() -> None:
    class Env:
        api_configs = {
            "franka_control_api": {
                "max_delta_xyz": 0.01,
                "max_insert_actions": 20,
            }
        }
        task = None

        def __init__(self) -> None:
            self.actions = []

        def get_robot_state(self):
            return {
                "ee_pos": [0.0, 0.0, 0.2],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
            }

        def take_action(self, action, *, action_type):
            self.actions.append((np.asarray(action, dtype=np.float32), action_type))
            return {
                "ok": True,
                "reason": "native_success",
                "episode_stopped": True,
                "success_latched": True,
            }

        def finalize_high_level_action(self):
            return {
                "enabled": True,
                "stopped": True,
                "reason": "native_success",
                "episode_stopped": True,
                "success_latched": True,
                "action_count": len(self.actions),
                "max_steps": 300,
            }

        def get_protocol_status(self):
            return {
                "enabled": True,
                "stopped": True,
                "reason": "native_success",
                "episode_stopped": True,
                "success_latched": True,
                "action_count": len(self.actions),
                "max_steps": 300,
            }

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    api._read_adaptive_tactile_summary = _summary_reader(_move_tactile_summary())

    result = api.move_relative([0.0, 0.0, -0.003], tactile_guard=True)

    assert result["ok"] is True
    assert result["reason"] == "native_success"
    assert result["success_latched"] is True
    assert result["episode_stopped"] is True
    assert result["executed_depth"] == pytest.approx(0.0, abs=1e-6)
    assert env.actions == []


def test_move_relative_guard_stops_on_tactile_and_action_limits() -> None:
    class Env:
        api_configs = {
            "franka_control_api": {
                "max_delta_xyz": 0.01,
                "max_insert_actions": 20,
            }
        }
        task = None

        def __init__(self) -> None:
            self.actions = []
            self.protocol_reason = None
            self.finalize_reason = None

        def get_robot_state(self):
            return {
                "ee_pos": [0.0, 0.0, 0.2],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
            }

        def take_action(self, action, *, action_type):
            self.actions.append((np.asarray(action, dtype=np.float32), action_type))
            return {"ok": True, "message": "action executed"}

        def finalize_high_level_action(self):
            if self.finalize_reason is not None:
                self.protocol_reason = self.finalize_reason
            stopped = self.protocol_reason is not None
            return {
                "enabled": True,
                "stopped": stopped,
                "reason": self.protocol_reason,
            }

        def get_protocol_status(self):
            return {
                "enabled": True,
                "stopped": self.protocol_reason is not None,
                "reason": self.protocol_reason,
                "action_count": len(self.actions),
                "max_steps": 300,
            }

    env = Env()
    api = UniVTACFrankaCompatApi(env)

    api._read_adaptive_tactile_summary = _summary_reader(
        _move_tactile_summary(),
        _move_tactile_summary(contact=False, event="contact_lost", normal_force=0.0),
    )
    result = api.move_relative([0.0, 0.0, -0.003], tactile_guard=True)
    assert result["ok"] is False
    assert result["reason"] == "contact_lost"
    assert result["executed_depth"] == pytest.approx(0.0005, abs=1e-6)
    assert len(env.actions) == 1

    env.actions.clear()
    env.protocol_reason = None
    env.finalize_reason = "action_budget"
    api._read_adaptive_tactile_summary = _summary_reader(
        _move_tactile_summary(),
        _move_tactile_summary(contact=False, event="contact_lost", normal_force=0.0),
    )
    result = api.move_relative([0.0, 0.0, -0.003], tactile_guard=False)
    assert result["ok"] is False
    assert result["reason"] == "action_budget"
    assert len(env.actions) == 1

    env.actions.clear()
    env.protocol_reason = None
    env.finalize_reason = None
    env.api_configs["franka_control_api"]["max_insert_actions"] = 2
    api = UniVTACFrankaCompatApi(env)
    api._read_adaptive_tactile_summary = _summary_reader(_move_tactile_summary())

    result = api.move_relative([0.0, 0.0, -0.03], tactile_guard=True)

    assert result["ok"] is False
    assert result["reason"] == "max_actions"
    assert result["executed_depth"] == pytest.approx(0.001, abs=1e-6)
    assert len(env.actions) == 2


def test_move_relative_limits_corrections_and_allows_slip_recovery() -> None:
    class Env:
        api_configs = {
            "franka_control_api": {
                "max_delta_xyz": 0.01,
                "max_insert_actions": 20,
                "move_relative_max_lateral": 0.0005,
                "move_relative_max_rotation": 0.006,
                "move_relative_lateral_budget": 0.001,
                "move_relative_rotation_budget": 0.012,
            }
        }
        task = None

        def __init__(self) -> None:
            self.actions = []

        def get_robot_state(self):
            return {
                "ee_pos": [0.0, 0.0, 0.2],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
            }

        def take_action(self, action, *, action_type):
            self.actions.append((np.asarray(action, dtype=np.float32), action_type))
            return {"ok": True, "message": "action executed"}

        def finalize_high_level_action(self):
            return {"enabled": True, "stopped": False, "reason": None}

        def get_protocol_status(self):
            return {
                "enabled": True,
                "stopped": False,
                "reason": None,
                "action_count": len(self.actions),
                "max_steps": 300,
            }

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    api._read_adaptive_tactile_summary = lambda: _move_tactile_summary()

    result = api.move_relative([0.0, 0.001, 0.0], tactile_guard=True)
    assert result["ok"] is True
    assert result["reason"] == "completed"
    assert result["clipped"] is True
    assert result["clip_reason"] == "move_relative_lateral_step_limit"
    assert result["executed_distance"] == pytest.approx(0.0005, abs=1e-6)
    assert len(env.actions) == 1
    np.testing.assert_allclose(env.actions[-1][0][:3], [0.0, 0.0005, 0.0], atol=1e-7)

    result = api.move_relative([0.0, 0.0, 0.0], delta_rpy=[0.0, 0.0, 0.02], tactile_guard=True)
    assert result["ok"] is True
    assert result["reason"] == "completed"
    assert result["clipped"] is True
    assert result["clip_reason"] == "move_relative_rotation_step_limit"
    assert len(env.actions) == 2
    np.testing.assert_allclose(env.actions[-1][0][3:6], [0.0, 0.0, 0.006], atol=1e-7)

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    api._read_adaptive_tactile_summary = lambda: _move_tactile_summary()
    result = api.move_relative([0.0, 0.0005, 0.0], tactile_guard=True)
    assert result["ok"] is True
    result = api.move_relative([0.0, 0.0005, 0.0], tactile_guard=True)
    assert result["ok"] is True
    result = api.move_relative([0.0, 0.0005, 0.0], tactile_guard=True)
    assert result["ok"] is True
    assert result["reason"] == "move_relative_lateral_budget"
    assert result["executed_distance"] == pytest.approx(0.0, abs=1e-6)

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    api._read_adaptive_tactile_summary = _summary_reader(
        _move_tactile_summary(),
        _move_tactile_summary(
            event="slip_detected",
            slip_score=0.9,
            drift=2.0,
            slip_risk="high",
            incipient_slip=True,
            drift_trend="increasing",
            correction_hint="try_pitch_probe",
        ),
    )
    result = api.move_relative([0.0, 0.0, -0.002], tactile_guard=True)
    assert result["ok"] is True
    assert result["reason"] == "interrupted_by_slip_warning"
    assert result["interrupted"] is True
    assert result["executed_depth"] == pytest.approx(0.0005, abs=1e-6)
    assert result["tactile"]["slip"] is True
    assert len(env.actions) == 1

    api._read_adaptive_tactile_summary = _summary_reader(
        _move_tactile_summary(
            event="slip_detected",
            slip_score=0.9,
            drift=2.0,
            slip_risk="high",
            incipient_slip=True,
            correction_hint="try_pitch_probe",
        )
    )
    result = api.move_relative([0.0, 0.0, -0.002], tactile_guard=True)
    assert result["ok"] is True
    assert result["reason"] == "preempted_by_slip_risk"
    assert result["preempted"] is True
    assert result["executed_depth"] == pytest.approx(0.0, abs=1e-6)
    assert len(env.actions) == 1

    api._read_adaptive_tactile_summary = _summary_reader(
        _move_tactile_summary(
            event="slip_detected",
            slip_score=0.9,
            drift=2.0,
            slip_risk="high",
            incipient_slip=True,
            correction_hint="try_pitch_probe",
        ),
        _move_tactile_summary(
            event="slip_detected",
            slip_score=0.9,
            drift=2.0,
            slip_risk="high",
            incipient_slip=True,
            correction_hint="try_pitch_probe",
        ),
    )
    result = api.move_relative([0.0, 0.0, 0.0], delta_rpy=[0.0, 0.003, 0.0], tactile_guard=True)
    assert result["ok"] is True
    assert result["reason"] == "completed_with_slip"
    assert len(env.actions) == 2

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    api._read_adaptive_tactile_summary = _summary_reader(
        _move_tactile_summary(),
        _move_tactile_summary(contact=False, event="contact_lost", normal_force=0.0),
    )
    result = api.move_relative([0.0, 0.0, -0.002], tactile_guard=True)
    assert result["ok"] is False
    assert result["reason"] == "contact_lost"
    assert len(env.actions) == 1

    result = api.move_relative([0.0, 0.0, -0.002], tactile_guard=True)
    assert result["ok"] is False
    assert result["reason"] == "contact_lost"
    assert len(env.actions) == 1
