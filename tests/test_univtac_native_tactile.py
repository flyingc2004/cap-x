from __future__ import annotations

from pathlib import Path

import numpy as np
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
    pos, quat, extent = api.get_object_pose("orange pad")
    assert pos.shape == (3,)
    assert quat.shape == (4,)
    np.testing.assert_allclose(quat, [0.5, 0.5, 0.5, 0.5])
    assert extent is None
    _, _, extent = api.get_object_pose("orange pad", return_bbox_extent=True)
    assert extent.shape == (3,)
    api.goto_pose(np.array([0.1, 0.1, 0.05]), np.array([0.5, 0.5, 0.5, 0.5]))
    assert api._env.calls
    assert all(call[1] == "delta_ee" for call in api._env.calls)
    assert all(float(action[2]) >= 0.0 for action, _ in api._env.calls)


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
