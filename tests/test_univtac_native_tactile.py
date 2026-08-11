from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml

from capx.envs.runner import _setup_output_dir
from capx.envs.simulators.univtac import UniVTACLowLevelEnv
from capx.integrations.base_api import list_apis
from capx.integrations.univtac.control_api import UniVTACControlApi
from capx.integrations.univtac.franka_compat_api import UniVTACFrankaCompatApi
from capx.integrations.univtac.native_tactile import (
    UniVTACTactileFrame,
    frame_from_observation,
    summarize_native_tactile,
    tactile_event_sequence,
)
from capx.integrations.univtac.tactile_api import UniVTACTactileApi


def _depth(indented: bool) -> np.ndarray:
    arr = np.full((16, 16), 34.0, dtype=np.float32)
    if indented:
        arr[4:12, 4:12] = 30.0
    return arr


def _marker(displacement: float = 0.0) -> np.ndarray:
    start = np.zeros((8, 8, 2), dtype=np.float32)
    end = start.copy()
    end[..., 0] += displacement
    return np.stack([start, end], axis=0)


def _frame(
    *,
    step: int,
    left_contact: bool = False,
    right_contact: bool = False,
    left_marker: float = 0.0,
    right_marker: float = 0.0,
) -> UniVTACTactileFrame:
    return UniVTACTactileFrame(
        step=step,
        timestamp=float(step),
        left_depth=_depth(left_contact),
        right_depth=_depth(right_contact),
        left_marker=_marker(left_marker),
        right_marker=_marker(right_marker),
        left_pose=None,
        right_pose=None,
    )


def test_native_tactile_summary_events() -> None:
    assert summarize_native_tactile([_frame(step=0)])["event"] == "no_contact"
    assert (
        summarize_native_tactile([_frame(step=0, left_contact=True)])["event"]
        == "one_hand_contact"
    )

    stable = summarize_native_tactile(
        [_frame(step=0, left_contact=True, right_contact=True)]
    )
    assert stable["event"] == "stable_grasp"
    assert stable["left_contact"] is True
    assert stable["right_contact"] is True

    slip = summarize_native_tactile(
        [
            _frame(step=0, left_contact=True, right_contact=True),
            _frame(
                step=1,
                left_contact=True,
                right_contact=True,
                left_marker=4.0,
                right_marker=4.0,
            ),
        ]
    )
    assert slip["event"] == "slip_detected"
    assert slip["slip_score"] >= 0.6

    lost = summarize_native_tactile(
        [_frame(step=0, left_contact=True, right_contact=True), _frame(step=1)]
    )
    assert lost["event"] == "contact_lost"


def test_native_marker_motion_uses_initial_current_axis_for_live_shape() -> None:
    marker = np.zeros((2, 12, 2), dtype=np.float32)
    marker[0, :, 0] = np.linspace(20.0, 300.0, 12)
    marker[0, :, 1] = np.linspace(10.0, 220.0, 12)
    marker[1] = marker[0]
    no_motion = UniVTACTactileFrame(
        step=0,
        timestamp=0.0,
        left_depth=_depth(False),
        right_depth=_depth(False),
        left_marker=marker,
        right_marker=marker,
        left_pose=None,
        right_pose=None,
    )
    summary = summarize_native_tactile([no_motion])
    assert summary["contact"] is False
    assert summary["shear_magnitude"] == 0.0

    displaced = marker.copy()
    displaced[1, :, 0] += 4.0
    motion = UniVTACTactileFrame(
        step=1,
        timestamp=1.0,
        left_depth=_depth(False),
        right_depth=_depth(False),
        left_marker=displaced,
        right_marker=displaced,
        left_pose=None,
        right_pose=None,
    )
    summary = summarize_native_tactile([motion])
    assert summary["contact"] is True
    assert summary["shear_magnitude"] == pytest.approx(1.0)


def test_calibrated_depth_does_not_treat_marker_only_motion_as_contact() -> None:
    frame = _frame(step=0, left_marker=4.0, right_marker=4.0)

    summary = summarize_native_tactile(
        [frame],
        depth_far_plane_mm=34.0,
        force_full_scale_mm=6.5,
        depth_contact_margin_mm=0.5,
    )

    assert summary["contact"] is False
    assert summary["normal_force"] == 0.0
    assert summary["shear_magnitude"] == pytest.approx(1.0)
    assert summary["event"] == "no_contact"


def test_calibrated_depth_uses_robot_far_plane_for_force() -> None:
    summary = summarize_native_tactile(
        [_frame(step=0, left_contact=True, right_contact=True)],
        depth_far_plane_mm=34.0,
        force_full_scale_mm=6.5,
        depth_contact_margin_mm=0.5,
    )

    assert summary["left_contact"] is True
    assert summary["right_contact"] is True
    assert summary["left"]["depth_min_mm"] == pytest.approx(30.0)
    assert summary["left"]["depth_delta_mm"] == pytest.approx(4.0)
    assert summary["normal_force"] == pytest.approx(4.0 / 6.5)
    assert summary["event"] == "stable_grasp"


def test_native_tactile_event_sequence_deduplicates() -> None:
    events = tactile_event_sequence(
        [
            _frame(step=0),
            _frame(step=1),
            _frame(step=2, left_contact=True),
            _frame(step=3, left_contact=True, right_contact=True),
            _frame(
                step=4,
                left_contact=True,
                right_contact=True,
                left_marker=4.0,
                right_marker=4.0,
            ),
        ]
    )
    assert events == ["no_contact", "one_hand_contact", "stable_grasp", "slip_detected"]


def test_frame_from_observation_uses_univtac_native_tactile_keys() -> None:
    obs = {
        "tactile": {
            "left_tactile": {
                "rgb": np.zeros((2, 2, 3)),
                "rgb_marker": np.zeros((2, 2, 3)),
                "marker": _marker(1.0),
                "depth": _depth(True),
                "pose": np.arange(7),
            },
            "right_tactile": {
                "marker": _marker(0.0),
                "depth": _depth(False),
                "pose": np.arange(7),
            },
        }
    }

    frame = frame_from_observation(obs, step=7, timestamp=1.25)
    assert frame.step == 7
    assert frame.left_depth is not None
    assert frame.right_depth is not None
    assert frame.left_marker is not None
    assert frame.right_marker is not None


def test_univtac_control_action_shapes() -> None:
    class Env:
        task = object()

        def __init__(self) -> None:
            self.calls = []

        def take_action(self, action, *, action_type: str):
            self.calls.append((np.asarray(action), action_type))
            return {"ok": True}

        def get_robot_state(self):
            return {"joint": [0.0] * 8}

        def get_task_instruction(self):
            return "test"

        def current_raw_observation(self):
            return {"observation": {}, "tactile": {}}

        def get_actor_poses(self):
            return {}

        def get_status(self):
            return {
                "task": "grasp_classify",
                "instruction": "test",
                "step": 0,
                "action_count": 0,
                "max_steps": 10,
            }

    env = Env()
    api = UniVTACControlApi(env)

    api.move_delta_ee([0.01, 0.0, 0.0])
    api.move_ee([0.3, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0], gripper_width=0.5)
    api.move_qpos([0.0] * 7, gripper_width=1.0)

    assert env.calls[0][1] == "delta_ee"
    assert env.calls[0][0].shape == (7,)
    assert env.calls[1][1] == "ee"
    assert env.calls[1][0].shape == (8,)
    assert env.calls[2][1] == "qpos"
    assert env.calls[2][0].shape == (8,)


def test_univtac_control_robot_state_stable_keys() -> None:
    class Env:
        def get_robot_state(self):
            return {
                "ee": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                "ee_pos": [0.1, 0.2, 0.3],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
                "ee_pose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                "joint": [0.0] * 8,
                "qpos": [0.0] * 8,
                "gripper_qpos": 0.01,
            }

    state = UniVTACControlApi(Env()).get_robot_state()
    assert state["ee_pos"] == [0.1, 0.2, 0.3]
    assert state["ee_quat"] == [1.0, 0.0, 0.0, 0.0]
    assert state["ee_pose"][:3] == state["ee_pos"]
    assert state["qpos"] == state["joint"]
    assert state["gripper_qpos"] == 0.01


def test_univtac_api_registration_and_config_are_native_only() -> None:
    import capx.integrations  # noqa: F401

    assert "UniVTACControlApi" in list_apis()
    assert "UniVTACTactileApi" in list_apis()
    assert "FrankaControlApi" in list_apis()

    franka_functions = UniVTACFrankaCompatApi.__new__(UniVTACFrankaCompatApi).functions()
    assert set(franka_functions) == {
        "get_object_pose",
        "sample_grasp_pose",
        "goto_pose",
        "open_gripper",
        "close_gripper",
        "home_pose",
    }

    functions = UniVTACTactileApi.__new__(UniVTACTactileApi).functions()
    assert "get_tactile_summary" in functions
    assert "retrieve_tactile_strategies" not in functions

    config_path = (
        Path(__file__).resolve().parents[1]
        / "env_configs"
        / "univtac"
        / "grasp_classify_tactile.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    apis = config["env"]["cfg"]["apis"]
    assert apis == ["FrankaControlApi", "UniVTACTactileApi"]
    assert "TactileMemoryApi" not in apis

    prompt = config["env"]["cfg"]["prompt"]
    assert "UniVTAC native" in prompt
    assert "ACT/ViTAL" in prompt
    assert 'sample_grasp_pose("prism") -> goto_pose(grasp)' in prompt
    assert config["env"]["cfg"]["privileged"] is False
    assert config["env"]["cfg"]["low_level"]["expose_actor_pose"] is False
    assert config["env"]["cfg"]["low_level"]["task_config"] == "smoke_capx_manual_grasp"
    assert config["env"]["cfg"]["low_level"]["api_configs"]["franka_control_api"]["min_safe_z"] == 0.10
    assert (
        config["env"]["cfg"]["low_level"]["api_configs"]["franka_control_api"][
            "use_task_place_actor_for_landmarks"
        ]
        is False
    )
    assert (
        config["env"]["cfg"]["low_level"]["api_configs"]["franka_control_api"][
            "use_task_grasp_actor_for_prism"
        ]
        is True
    )

    preplace_config_path = config_path.with_name("grasp_classify_tactile_preplace.yaml")
    preplace_config = yaml.safe_load(preplace_config_path.read_text(encoding="utf-8"))
    assert preplace_config["env"]["cfg"]["low_level"]["task_config"] == "smoke_capx"
    assert (
        preplace_config["env"]["cfg"]["low_level"]["api_configs"]["franka_control_api"][
            "use_task_place_actor_for_landmarks"
        ]
        is False
    )

    lift_config_path = config_path.with_name("lift_can_tactile.yaml")
    lift_config = yaml.safe_load(lift_config_path.read_text(encoding="utf-8"))
    lift_cfg = lift_config["env"]["cfg"]
    lift_low_level = lift_cfg["low_level"]
    lift_franka = lift_low_level["api_configs"]["franka_control_api"]
    assert lift_low_level["task_name"] == "lift_can"
    assert lift_low_level["task_config"] == "smoke_capx_lift_can"
    assert lift_cfg["apis"] == ["FrankaControlApi", "UniVTACTactileApi"]
    assert lift_franka["object_pose_names"] == {
        "can": "can",
        "object": "can",
        "target object": "can",
    }
    assert lift_franka["use_task_grasp_actor_for_objects"] is True
    assert lift_franka["grasp_z_tolerance"] == 0.04
    assert lift_franka["max_delta_xyz"] == 0.01
    assert "lift_success_height_delta" not in lift_low_level
    assert "lift_success_require_contact" not in lift_low_level
    assert "target_force=1.0, max_steps=160" in lift_cfg["prompt"]
    assert "two 0.05 meter increments" in lift_cfg["prompt"]
    assert "call open_gripper(adaptive=True) to release the can" in lift_cfg["prompt"]


def test_lift_can_completion_uses_native_task_check() -> None:
    native_result = {"value": True}
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env.task_name = "lift_can"
    env._task = types.SimpleNamespace(
        check_success=lambda: native_result["value"],
    )

    assert env.task_completed() is True
    native_result["value"] = False
    assert env.task_completed() is False


def test_univtac_franka_compat_respects_config_and_uses_high_level_api() -> None:
    class Env:
        def __init__(self) -> None:
            self.calls = []
            self.place_calls = []
            self.api_configs = {
                "franka_control_api": {
                    "min_safe_z": 0.2,
                    "max_delta_xyz": 0.01,
                    "max_delta_rpy": 0.05,
                    "default_z_approach": 0.12,
                    "release_hover_height": 0.25,
                    "use_task_place_actor_for_landmarks": False,
                    "object_pose_names": {"orange pad": "orange_pad"},
                }
            }

        def take_action(self, action, *, action_type: str):
            self.calls.append((np.asarray(action), action_type))
            return {"ok": True}

        def get_robot_state(self):
            return {"ee_pos": [0.1, 0.1, 0.15], "ee_quat": [0.5, 0.5, 0.5, 0.5], "joint": [0.0] * 8}

        def get_status(self):
            return {"task": "grasp_classify", "instruction": "test", "step": 0, "action_count": 0, "max_steps": 10}

    api = UniVTACFrankaCompatApi(Env())
    assert api.min_safe_z == 0.2
    assert api.default_z_approach == 0.12
    pos, quat = api.get_object_pose("orange pad")
    assert pos.shape == (3,)
    assert quat.shape == (4,)
    np.testing.assert_allclose(quat, [0.5, 0.5, 0.5, 0.5])
    _, _, extent = api.get_object_pose("orange pad", return_bbox_extent=True)
    assert extent.shape == (3,)
    api.goto_pose(np.array([0.1, 0.1, 0.05]), np.array([0.5, 0.5, 0.5, 0.5]))
    assert api._env.calls
    assert all(call[1] == "delta_ee" for call in api._env.calls)
    assert all(float(action[2]) >= 0.0 for action, _ in api._env.calls)


def test_univtac_franka_zero_z_approach_moves_directly() -> None:
    class Env:
        task = None
        api_configs = {
            "franka_control_api": {
                "default_z_approach": 0.12,
                "use_task_grasp_actor_for_objects": False,
                "use_task_place_actor_for_landmarks": False,
            }
        }

        def get_robot_state(self):
            return {
                "ee_pos": [0.1, 0.1, 0.2],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
                "joint": [0.0] * 8,
            }

    api = UniVTACFrankaCompatApi(Env())
    moves = []
    api._move_to_pose_bounded = lambda target, quat, current, current_quat: moves.append(
        np.asarray(target).copy()
    )
    target = np.array([0.2, 0.2, 0.3], dtype=np.float32)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    api.goto_pose(target, quat, z_approach=0.0)
    assert len(moves) == 1
    np.testing.assert_allclose(moves[0], target)

    moves.clear()
    api.goto_pose(target, quat, z_approach=0.1)
    assert len(moves) == 2
    np.testing.assert_allclose(moves[0], target + [0.0, 0.0, 0.1])
    np.testing.assert_allclose(moves[1], target)


def test_univtac_franka_compat_routes_public_pad_to_native_placement() -> None:
    class Env:
        def __init__(self) -> None:
            self.calls = []
            self.place_calls = []
            self.api_configs = {
                "franka_control_api": {
                    "placement_xy_tolerance": 0.08,
                    "use_task_place_actor_for_landmarks": True,
                }
            }

        def take_action(self, action, *, action_type: str):
            self.calls.append((np.asarray(action), action_type))
            return {"ok": True}

        def place_grasped_actor(self, **kwargs):
            self.place_calls.append(kwargs)
            return {"ok": True, "message": "native placement executed"}

        def get_robot_state(self):
            return {"ee_pos": [0.3, 0.0, 0.2], "ee_quat": [0.5, 0.5, 0.5, 0.5], "joint": [0.0] * 8}

        def get_status(self):
            return {"task": "grasp_classify", "instruction": "test", "step": 0, "action_count": 0, "max_steps": 10}

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    api.goto_pose(np.array([0.40, 0.08, 0.125]), np.array([1.0, 0.0, 0.0, 0.0]), z_approach=0.1)
    assert not env.calls
    assert len(env.place_calls) == 1
    assert env.place_calls[0]["target_name"] == "green_pad"
    np.testing.assert_allclose(env.place_calls[0]["target_position"], [0.40, 0.08, 0.025])


def test_univtac_franka_compat_routes_prism_to_native_grasp() -> None:
    class Actor:
        def get_pose(self):
            class Pose:
                p = np.array([0.35, 0.0, 0.01], dtype=np.float32)

            return Pose()

    class Task:
        prism = Actor()

    class Env:
        task = Task()

        def __init__(self) -> None:
            self.approach_calls = []
            self.api_configs = {
                "franka_control_api": {
                    "use_task_grasp_actor_for_prism": True,
                    "use_task_place_actor_for_landmarks": False,
                    "grasp_xy_tolerance": 0.08,
                }
            }

        def approach_grasped_actor(self, **kwargs):
            self.approach_calls.append(kwargs)
            return {"ok": True, "message": "native grasp approach executed"}

        def take_action(self, action, *, action_type: str):
            raise AssertionError("prism grasp should route through native grasp approach")

        def get_robot_state(self):
            return {"ee_pos": [0.3, 0.0, 0.2], "ee_quat": [1.0, 0.0, 0.0, 0.0], "joint": [0.0] * 8}

        def get_status(self):
            return {"task": "grasp_classify", "instruction": "test", "step": 0, "action_count": 0, "max_steps": 10}

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    grasp_pos, grasp_quat = api.sample_grasp_pose("prism")
    np.testing.assert_allclose(grasp_pos, [0.35, 0.0, 0.05], atol=1e-6)
    assert grasp_quat.shape == (4,)
    api.goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
    assert len(env.approach_calls) == 1
    assert env.approach_calls[0]["object_name"] == "prism"


def test_univtac_franka_compat_exposes_and_routes_can_grasp() -> None:
    sampled_pos = np.array([0.635, 0.01, 0.022], dtype=np.float32)
    sampled_quat = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    class Pose:
        p = np.array([0.70, 0.01, 0.03], dtype=np.float32)
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    class Actor:
        def get_pose(self):
            return Pose()

    class Task:
        can = Actor()

    class Env:
        task = Task()

        def __init__(self) -> None:
            self.approach_calls = []
            self.motion_calls = []
            self.api_configs = {
                "franka_control_api": {
                    "use_task_grasp_actor_for_objects": True,
                    "use_task_place_actor_for_landmarks": False,
                    "grasp_xy_tolerance": 0.08,
                    "grasp_z_tolerance": 0.04,
                    "object_pose_names": {
                        "can": "can",
                        "object": "can",
                        "target object": "can",
                    },
                }
            }

        def get_public_grasp_pose(self, object_name, *, grasp_height):
            assert object_name == "can"
            assert grasp_height == 0.04
            return sampled_pos.copy(), sampled_quat.copy()

        def approach_grasped_actor(self, **kwargs):
            self.approach_calls.append(kwargs)
            return {"ok": True, "message": "native grasp approach executed"}

        def take_action(self, action, *, action_type: str):
            self.motion_calls.append((np.asarray(action), action_type))
            return {"ok": True}

        def get_robot_state(self):
            return {
                "ee_pos": [0.30, 0.0, 0.20],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
                "joint": [0.0] * 8,
            }

        def get_status(self):
            return {
                "task": "lift_can",
                "instruction": "test",
                "step": 0,
                "action_count": 0,
                "max_steps": 10,
            }

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    for alias in ("can", "object", "target object"):
        pos, quat = api.get_object_pose(alias)
        np.testing.assert_allclose(pos, Pose.p)
        np.testing.assert_allclose(quat, Pose.q)
    _pos, _quat, extent = api.get_object_pose("can", return_bbox_extent=True)
    assert extent.shape == (3,)

    grasp_pos, grasp_quat = api.sample_grasp_pose("can")
    np.testing.assert_allclose(grasp_pos, sampled_pos)
    np.testing.assert_allclose(grasp_quat, sampled_quat)
    assert not np.allclose(grasp_pos, env.get_robot_state()["ee_pos"])

    api.goto_pose(grasp_pos, grasp_quat, z_approach=0.08)
    assert len(env.approach_calls) == 1
    assert env.approach_calls[0]["object_name"] == "can"
    np.testing.assert_allclose(env.approach_calls[0]["position_offset"], [0.0, 0.0, 0.0])
    assert not env.motion_calls

    adjusted_pos = grasp_pos + np.array([0.01, -0.01, 0.02], dtype=np.float32)
    api.goto_pose(adjusted_pos, grasp_quat, z_approach=0.08)
    assert len(env.approach_calls) == 2
    np.testing.assert_allclose(
        env.approach_calls[1]["position_offset"],
        [0.01, -0.01, 0.02],
        atol=1e-6,
    )

    lift_pos = grasp_pos + np.array([0.0, 0.0, 0.10], dtype=np.float32)
    api.goto_pose(lift_pos, grasp_quat, z_approach=0.0)
    assert len(env.approach_calls) == 2
    assert env.motion_calls
    assert all(action_type == "delta_ee" for _action, action_type in env.motion_calls)


def test_univtac_franka_compat_can_missing_is_explicit() -> None:
    class Env:
        task = object()
        api_configs = {
            "franka_control_api": {
                "object_pose_names": {"can": "can"},
            }
        }

        def get_public_grasp_pose(self, object_name, *, grasp_height):
            raise KeyError(object_name)

        def get_robot_state(self):
            return {
                "ee_pos": [0.30, 0.0, 0.20],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
                "joint": [0.0] * 8,
            }

    api = UniVTACFrankaCompatApi(Env())
    with pytest.raises(KeyError, match="can"):
        api.get_object_pose("can")
    with pytest.raises(KeyError, match="can"):
        api.sample_grasp_pose("can")


def test_univtac_low_level_can_grasp_matches_native_task_geometry(monkeypatch) -> None:
    captured = {}

    class Pose:
        def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)) -> None:
            self.p = np.asarray(p, dtype=np.float32)
            self.q = np.asarray(q, dtype=np.float32)

        def add_bias(self, bias, coord="local"):
            return Pose(self.p + np.asarray(bias, dtype=np.float32), self.q)

        def to_transformation_matrix(self):
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, 3] = self.p
            return matrix

    def construct_grasp_pose(position, z_axis, x_axis):
        captured["position"] = np.asarray(position, dtype=np.float32)
        captured["z_axis"] = np.asarray(z_axis, dtype=np.float32)
        captured["x_axis"] = np.asarray(x_axis, dtype=np.float32)
        return Pose(position, (0.5, 0.5, 0.5, 0.5))

    envs_module = types.ModuleType("envs")
    envs_module.__path__ = []
    utils_module = types.ModuleType("envs.utils")
    utils_module.__path__ = []
    transforms_module = types.ModuleType("envs.utils.transforms")
    transforms_module.construct_grasp_pose = construct_grasp_pose
    monkeypatch.setitem(sys.modules, "envs", envs_module)
    monkeypatch.setitem(sys.modules, "envs.utils", utils_module)
    monkeypatch.setitem(sys.modules, "envs.utils.transforms", transforms_module)

    class Actor:
        def __init__(self) -> None:
            self.registered = []

        def get_pose(self):
            return Pose([0.70, 0.0, 0.03])

        def register_point(self, pose, *, type):
            self.registered.append((pose, type))
            return 7

    actor = Actor()

    class Atom:
        def __init__(self) -> None:
            self.calls = []

        def grasp_actor(self, target_actor, **kwargs):
            self.calls.append((target_actor, kwargs))
            return ["native-grasp-action"]

    class Task:
        can = actor
        atom = Atom()
        step_count = 0
        take_action_cnt = 0

        def move(self, actions, **kwargs):
            self.move_call = (actions, kwargs)
            return True

    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._task = Task()
    env._last_action_result = {}
    env._update_after_action = lambda: None
    env._append_debug_record = lambda label: None

    grasp_pos, grasp_quat = env.get_public_grasp_pose("can")
    np.testing.assert_allclose(grasp_pos, [0.635, 0.0, 0.022], atol=1e-6)
    np.testing.assert_allclose(grasp_quat, [0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(captured["z_axis"], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(captured["x_axis"], [1.0, 0.0, 0.0])

    result = env.approach_grasped_actor(object_name="can")
    assert result["ok"] is True
    assert actor.registered[-1][1] == "contact"
    target_actor, kwargs = env._task.atom.calls[-1]
    assert target_actor is actor
    assert kwargs["contact_point_id"] == 7
    assert kwargs["is_close"] is False

    result = env.approach_grasped_actor(
        object_name="can",
        position_offset=[0.01, -0.01, 0.02],
    )
    assert result["ok"] is True
    shifted_pose = actor.registered[-1][0]
    np.testing.assert_allclose(shifted_pose.p, [0.645, -0.01, 0.042], atol=1e-6)


def test_univtac_lift_can_instruction_is_public_only() -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._task = object()
    env.task_name = "lift_can"

    instruction = env.get_task_instruction()
    assert "cylindrical can" in instruction
    assert "native tactile" in instruction
    assert "release it upright on the table" in instruction
    assert "reward" not in instruction
    assert "success" not in instruction
    assert "metadata" not in instruction


def test_preserve_output_dir_env_flag(monkeypatch, tmp_path) -> None:
    class Args:
        use_oracle_code = False
        model = "qwen/test"

    out = tmp_path / "run"
    config = {"output_dir": str(out)}
    monkeypatch.setenv("CAPX_PRESERVE_OUTPUT_DIR", "1")
    _setup_output_dir(Args(), config)
    assert config["output_dir"] == str(out)
    assert out.exists()


def test_univtac_video_step_hook_records_action_frames() -> None:
    class Task:
        def __init__(self) -> None:
            self.step_count = 0
            self.in_pre_move = False

        def _step(self, *args, **kwargs):
            self.step_count += 1
            return None

    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._task = Task()
    env._task_config = {}
    env._record_frames = True
    env._record_action_frames = True
    env._record_pre_move_frames = True
    env._video_frame_stride = 1
    env._frame_buffer = []
    env._wrist_frame_buffer = []
    env._last_recorded_video_step = None
    env._video_record_failures = 0
    env.get_step_count = lambda: env._task.step_count
    env._read_native_observation = lambda **kwargs: {
        "observation": {
            "head": {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)},
            "wrist": {"rgb": np.ones((4, 4, 3), dtype=np.uint8)},
        },
        "tactile": {
            "left_tactile": {"rgb_marker": np.zeros((4, 4, 3), dtype=np.uint8)},
            "right_tactile": {"rgb_marker": np.ones((4, 4, 3), dtype=np.uint8)},
        },
    }
    env._record_frame = lambda obs=None, force=False: env._frame_buffer.append(obs)

    UniVTACLowLevelEnv._install_task_runtime_patches(env)
    for _ in range(5):
        env._task._step()

    assert len(env._frame_buffer) == 5


def test_univtac_control_api_compat_name_still_registered() -> None:
    import capx.integrations  # noqa: F401

    assert "UniVTACControlApi" in list_apis()


def test_univtac_tactile_image_falls_back_to_rgb(monkeypatch) -> None:
    class Env:
        def __init__(self) -> None:
            self.refresh_calls = []

        def refresh_native_observation(self, *, include_camera, include_tactile, tactile_data_types):
            self.refresh_calls.append(
                {
                    "include_camera": include_camera,
                    "include_tactile": include_tactile,
                    "tactile_data_types": tactile_data_types,
                }
            )

        def current_raw_observation(self):
            return {
                "tactile": {
                    "left_tactile": {
                        "rgb": np.ones((2, 2, 3), dtype=np.uint8),
                        "depth": np.zeros((2, 2), dtype=np.float32),
                        "marker": _marker(0.0),
                        "pose": np.arange(7),
                    }
                }
            }

    api = UniVTACTactileApi(Env())
    image = api.get_tactile_image(hand="left", image_type="rgb_marker")
    assert image.shape == (2, 2, 3)
    assert api._env.refresh_calls[-1]["tactile_data_types"] == ["rgb", "marker", "depth", "pose"]
