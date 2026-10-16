import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from capx.envs.simulators.univtac import UniVTACLowLevelEnv
from capx.integrations.univtac.franka_compat_api import UniVTACFrankaCompatApi


CONFIG = Path(__file__).resolve().parents[1] / "env_configs/opentac/play_piano_tactile_smoke.yaml"


def test_piano_config_exposes_only_public_tactile_and_control():
    config = yaml.safe_load(CONFIG.read_text())
    spec = config["env"]["cfg"]
    low = spec["low_level"]
    franka = low["api_configs"]["franka_control_api"]
    tactile = low["api_configs"]["univtac_tactile_api"]

    assert spec["apis"] == ["FrankaControlApi", "UniVTACTactileApi"]
    assert low["task_name"] == low["task_config"] == "play_piano"
    assert low["expose_actor_pose"] is False
    assert low["privileged"] is False
    assert low["task_config_overrides"]["skip_task_pre_move"] is False
    assert low["task_config_overrides"]["defer_piano_scoring_failure_until_end"] is True
    assert low["runtime_preflight"]["expected_attachment_points_per_obj"] == 83
    assert franka["llm_api_profile"] == "opentac_piano"
    assert franka["public_grasp_anchor_objects"] == ["striker"]
    assert franka["rgbd_perception_enabled"] is False
    assert franka["use_task_grasp_actor_for_objects"] is False
    assert franka["use_task_place_actor_for_landmarks"] is False
    assert franka["min_safe_z"] < 0.003 + 0.085
    assert franka["local_delta_max_m"] <= 0.000075
    assert franka["local_wait_max_steps"] >= 78
    assert tactile["llm_visible_functions"] == ["get_tactile_summary", "is_grasp_stable"]
    assert config["tactile_memory"]["persistent"]["enabled"] is False
    assert 'open_gripper(adaptive=True, target_width=0.7)' in spec["prompt"]
    assert 'close_gripper(adaptive=True, target_force=0.0, max_steps=240)' in spec["prompt"]
    assert "pre_move only raises the gripper" in spec["prompt"]
    assert "0.000075 m downward" in spec["prompt"]
    assert "30 ticks on short notes and 78 ticks on long notes" in spec["prompt"]
    assert "C4 short, C4 long, G4 short, G4 long, A4 short" in spec["prompt"]
    assert "Do not estimate or control key force" in spec["prompt"]
    assert "capx_stop = True" in spec["prompt"]
    assert "finish_episode()" in spec["prompt"]


class PianoEnv:
    task = object()

    def __init__(self, *, with_striker=True):
        self.center = np.array([.34, .15, .23], dtype=np.float32)
        self.moves = []
        self.with_striker = with_striker
        self.api_configs = {"franka_control_api": {
            "llm_api_profile": "opentac_piano",
            "public_grasp_anchor_objects": ["striker"],
            "max_delta_xyz": .004,
            "max_goto_pose_actions": 120,
            "local_delta_max_m": .00010,
            "local_delta_segment_m": .000075,
            "min_safe_z": .08,
        }}

    def get_public_pose_map(self):
        if not self.with_striker:
            return {}
        return {"striker": (
            np.array([.34, .15, .088], dtype=np.float32),
            np.array([1, 0, 0, 0], dtype=np.float32),
            np.array([.026, .026, .108], dtype=np.float32),
        )}

    def get_piano_gripper_center_pose(self):
        return self.center.copy(), np.array([1, 0, 0, 0], dtype=np.float32)

    def move_piano_gripper_center_pose(self, position):
        self.center = np.asarray(position, dtype=np.float32)
        self.moves.append(self.center.copy())
        return {"ok": True, "motion_path": "piano_center_ik_to_qpos"}

    def get_robot_state(self):
        return {"ee_pos": [.1, .1, .4], "ee_quat": [1, 0, 0, 0], "joint": [0] * 8}


def test_piano_profile_routes_striker_and_bounded_center_motion():
    env = PianoEnv()
    api = UniVTACFrankaCompatApi(env)
    assert set(api.functions()) == {
        "get_object_pose", "sample_grasp_pose", "goto_pose", "move_delta",
        "open_gripper", "close_gripper", "wait_steps",
    }
    position, quat = api.sample_grasp_pose("striker")
    np.testing.assert_allclose(position, [.34, .15, .088])
    public_position, _ = api.functions()["get_object_pose"]("striker")
    np.testing.assert_allclose(public_position, position)
    assert not np.allclose(position, env.get_robot_state()["ee_pos"])
    assert api._piano_goto_pose(np.array([.344, .15, .23]), quat)["ok"]
    assert len(env.moves) == 1
    assert api._piano_move_delta(dz=-.000075)["ok"]
    assert len(env.moves) == 2
    assert api._piano_move_delta(dz=-.001)["reason"] == "local_delta_limit"
    assert len(env.moves) == 2


def test_missing_piano_grasp_anchor_does_not_return_tool_pose():
    with pytest.raises(KeyError, match="striker"):
        UniVTACFrankaCompatApi(PianoEnv(with_striker=False)).sample_grasp_pose("striker")


def test_piano_landmarks_work_without_task_side_public_pose_method():
    class Key:
        def __init__(self, midi, y):
            self.midi = midi
            self.size = (.15, .0225, .0225)
            self.strike_position = (.47, y, .0725)

    class Task:
        layout = (Key(60, -.1), Key(67, .05), Key(69, .1))
        striker_position = (.34, .15, .003)
        striker_grasp_height = .085

    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env.task_name = "play_piano"
    env._task = Task()
    env._public_pose_cache = {}
    env._refresh_public_pose_cache()

    assert set(env._public_pose_cache) == {
        "striker", "key_c4", "key_g4", "key_a4", "piano_safe_hover"
    }
    assert env._public_pose_cache["striker"]["position"] == pytest.approx((.34, .15, .088))
    assert env._public_pose_cache["key_c4"]["position"] == pytest.approx((.47, -.1, .1605))
    assert {record["source"] for record in env._public_pose_cache.values()} == {"piano_static_layout"}


def test_piano_public_grasp_anchor_is_reachable_above_safety_floor():
    env = PianoEnv()
    api = UniVTACFrankaCompatApi(env)
    grasp, orientation = api.sample_grasp_pose("striker")
    assert api._piano_goto_pose(grasp, orientation)["ok"] is True
    np.testing.assert_allclose(env.center, grasp)


def test_piano_center_bridge_uses_qpos_and_checks_ik():
    class Pose:
        def __init__(self, p, q):
            self.p = np.asarray(p)
            self.q = np.asarray(q)

    class Ik:
        def set_command(self, command):
            self.command = command

        def compute(self, *args):
            return torch.zeros((1, 7))

    class Manager:
        _arm_ids = list(range(7))
        _gripper_ids = [7]
        _ik_controller = Ik()
        jacobian_b = torch.zeros((1, 6, 8))
        robot = type("Robot", (), {"data": type("Data", (), {
            "joint_pos": torch.zeros((1, 8)),
            "soft_joint_pos_limits": torch.tensor([[[-1., 1.]] * 8]),
        })()})()

        def get_gripper_center_pose(self):
            return Pose([.34, .15, .2], [1, 0, 0, 0])

        def gripper_center_to_ee(self, pose):
            return pose

        def get_ee_pose_tensor(self):
            return torch.zeros((1, 3)), torch.tensor([[1., 0., 0., 0.]])

    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env.task_name = "play_piano"
    env._force_task_mode = True
    env._task = type("Task", (), {"_robot_manager": Manager(), "device": "cpu"})()
    env._protocol_stopped = False
    env._apply_official_task_protocol = lambda: None
    calls = []
    env.take_action = lambda target, *, action_type: calls.append((target, action_type)) or {"ok": True}

    result = env.move_piano_gripper_center_pose(np.array([.34, .15, .201]))
    assert result["ok"] is True
    assert calls[0][1] == "qpos"
    assert calls[0][0].shape == (8,)
    env._apply_official_task_protocol = lambda: setattr(env, "_protocol_stopped", True)
    env._protocol_stop_reason = "early_stop"
    assert env.move_piano_gripper_center_pose(np.array([.34, .15, .202])) == {
        "ok": False, "reason": "early_stop", "motion_path": "piano_center_ik_to_qpos"
    }
    env.task_name = "lift_can"
    assert env.move_piano_gripper_center_pose(np.array([.34, .15, .201]))["ok"] is False


def test_piano_tactile_summary_timeline_is_bounded():
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env.tactile_summary_timeline_enabled = True
    env.tactile_summary_timeline_max_records = 2
    env._tactile_summary_timeline = []
    env.get_step_count = lambda: len(env._tactile_summary_timeline)
    for value in range(3):
        env.append_tactile_summary_timeline(window=20, hand="both", summary={"normal_force": value})
    assert len(env._tactile_summary_timeline) == 2
    assert [row["summary"]["normal_force"] for row in env._tactile_summary_timeline] == [1, 2]


def test_piano_tactile_summary_timeline_is_exported(tmp_path):
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env.task_name = "play_piano"
    env._tactile_summary_timeline = [{"step": 12, "window": 20, "hand": "both", "summary": {"normal_force": .4}}]
    env._pre_move_tactile_timeline = []
    output = env.export_debug_artifacts(tmp_path)
    assert output == str(tmp_path / "tactile_summary_timeline.json")
    assert json.loads(Path(output).read_text())[0]["summary"]["normal_force"] == .4


def test_piano_smoke_defers_note_scoring_but_not_safety_failure():
    task = type("Task", (), {
        "failure": "note_too_short",
        "plan_success": False,
        "phase_id": "policy",
        "PHASE_TERMINAL": "terminal",
        "check_early_stop": lambda self: bool(self.failure),
    })()
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env.task_name = "play_piano"
    env._task = task
    env._defer_piano_scoring_failure_until_end = True
    env._official_task_protocol = True
    env.policy_controls_termination = True
    env._protocol_stopped = False
    env.max_steps = 300
    env.get_action_count = lambda: 1
    stops = []
    env._stop_protocol = stops.append
    env._install_piano_scoring_failure_deferral()

    assert env._deferred_piano_scoring_failure()
    env._resume_piano_step_after_scoring_failure(was_movable=True)
    assert task.plan_success is True
    assert task.check_early_stop() is False
    env._apply_official_task_protocol()
    assert stops == []
    assert task.failure == "note_too_short"

    for unsafe_reason in ("key_overload", "timeout", "striker_grasp_missing"):
        task.failure = unsafe_reason
        task.plan_success = False
        env._resume_piano_step_after_scoring_failure(was_movable=True)
        assert task.plan_success is False
        assert task.check_early_stop() is True
        env._apply_official_task_protocol()
        assert stops[-1] == "early_stop"


def test_piano_scoring_deferral_does_not_forge_final_success():
    task = type("Task", (), {
        "failure": "wrong_note_force",
        "phase_id": "policy",
        "PHASE_TERMINAL": "terminal",
        "finish_episode": lambda self: False,
    })()
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env.task_name = "play_piano"
    env._task = task
    env._defer_piano_scoring_failure_until_end = True
    env.policy_controls_termination = True
    env._policy_end_submitted = False
    env._protocol_stopped = False
    env._stop_protocol = lambda reason: setattr(env, "_protocol_stopped", True)
    assert env.submit_policy_end() is False
    assert env.task_completed() is False
    task.phase_id = "terminal"
    assert env._deferred_piano_scoring_failure() is False
