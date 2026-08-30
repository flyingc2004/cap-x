from __future__ import annotations

from pathlib import Path
import types

import numpy as np
import pytest
import yaml

from capx.envs.tasks.exceptions import RecoverableTaskFailure
from capx.envs.simulators.univtac import UniVTACLowLevelEnv
from capx.integrations.univtac.franka_compat_api import UniVTACFrankaCompatApi
from capx.integrations.univtac.rgbd_perception import (
    GraspEstimate,
    ObjectEstimate,
    RgbdFrame,
    SegmentationCandidate,
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


def _mask_at(row: int, col: int = 2) -> np.ndarray:
    mask = np.zeros((8, 8), dtype=bool)
    mask[row : row + 2, col : col + 2] = True
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


def test_rgbd_selector_chooses_sam_candidate_by_world_axis(monkeypatch) -> None:
    perception = UniVTACRgbdPerception(min_depth_points=4)
    candidates = [
        SegmentationCandidate(mask=_mask_at(1), score=0.2),
        SegmentationCandidate(mask=_mask_at(5), score=0.9),
    ]
    monkeypatch.setattr(
        perception,
        "_segment_candidates",
        lambda rgb, prompt: candidates,
    )

    low_y = perception.estimate_object(_frame(), "placement pad", selector="world_y_min")
    high_y = perception.estimate_object(_frame(), "placement pad", selector="world_y_max")

    assert low_y.score == pytest.approx(0.2)
    assert high_y.score == pytest.approx(0.9)
    assert low_y.position[1] < high_y.position[1]


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
    with pytest.raises(RuntimeError, match="valid depth points"):
        perception.estimate_object(_frame(valid_depth=False), "can")

    monkeypatch.setattr(
        perception,
        "_request_grasps",
        lambda depth, intrinsics, mask: (np.empty((0, 4, 4)), np.empty((0,))),
    )
    with pytest.raises(RuntimeError, match="invalid grasps"):
        perception.estimate_grasp(_frame(), "can")


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
    assert "api_servers" in config
    assert "home_pose()" in cfg["prompt"]
    assert "bounded 0.01 meter steps" in cfg["prompt"]
    assert "Do not command a wrist rotation" in cfg["prompt"]
    assert "official public pre-grasp anchor" in cfg["prompt"]
    assert "Each high-level motion API is bounded" in cfg["prompt"]
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


def test_transfer_easy_gt_yaml_uses_public_anchors_without_sam() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repo / "env_configs/univtac/tactile_transfer_easy_gt.yaml").read_text()
    )
    cfg = config["env"]["cfg"]
    low = cfg["low_level"]
    franka = low["api_configs"]["franka_control_api"]

    assert low["task_name"] == "tactile_transfer_rearrange_clean"
    assert low["task_config"] == "tactile_transfer_clean_smoke"
    assert low["expose_actor_pose"] is False
    assert low["privileged"] is False
    assert cfg["apis"] == ["FrankaControlApi", "UniVTACTactileApi"]
    assert franka["rgbd_perception_enabled"] is False
    assert "api_servers" not in config
    assert "Easy-GT engineering check" in cfg["prompt"]
    assert "get_object_pose(\"object_a\")" in cfg["prompt"]
    assert "UniVTACTouchManipulationApi" not in cfg["apis"]
    assert "search_contact" not in cfg["prompt"]


def test_transfer_hard_sam_yaml_routes_pose_perception_without_gt_slots() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repo / "env_configs/univtac/tactile_transfer_hard_sam.yaml").read_text()
    )
    cfg = config["env"]["cfg"]
    low = cfg["low_level"]
    franka = low["api_configs"]["franka_control_api"]

    assert low["task_name"] == "tactile_transfer_rearrange_clean"
    assert low["task_config"] == "tactile_transfer_clean_smoke"
    assert low["expose_actor_pose"] is False
    assert low["privileged"] is False
    assert cfg["apis"] == ["FrankaControlApi", "UniVTACTactileApi"]
    assert franka["rgbd_perception_enabled"] is True
    assert franka["rgbd_pose_enabled"] is True
    assert franka["rgbd_grasp_enabled"] is False
    assert franka["rgbd_pose_objects"] == ["object_a", "slot_a", "slot_b"]
    assert franka["rgbd_grasp_objects"] == []
    assert franka["public_grasp_anchor_objects"] == [
        "object_a",
        "object_b",
        "current_object",
    ]
    assert franka["public_pose_fallback_objects"] == []
    assert franka["cache_rgbd_pose_objects"] == ["slot_a", "slot_b"]
    assert franka["perception_selector_map"]["slot_a"] == "world_y_min"
    assert franka["perception_selector_map"]["slot_b"] == "world_y_max"
    assert len(config["api_servers"]) == 1
    assert "launch_sam3_server" in config["api_servers"][0]["_target_"]
    assert "launch_contact_graspnet_server" not in str(config["api_servers"])
    assert "get_object_pose(\"object_b\")" in cfg["prompt"]
    assert "do not call get_object_pose(\"object_b\")" in cfg["prompt"].lower()
    assert "0.42" not in cfg["prompt"]
    assert "UniVTACTouchManipulationApi" not in cfg["apis"]
    assert "search_contact" not in cfg["prompt"]


def test_hard_sam_franka_routes_object_and_slot_pose_through_rgbd() -> None:
    calls: list[tuple[str, str | None]] = []
    object_estimates = {
        "object_a": ObjectEstimate(
            position=np.array([0.58, -0.25, 0.04], dtype=np.float32),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            extent=np.array([0.04, 0.04, 0.08], dtype=np.float32),
            mask=_mask(),
            points_world=np.zeros((16, 3), dtype=np.float32),
            score=0.8,
            prompt="cylindrical can",
        ),
        "slot_a": ObjectEstimate(
            position=np.array([0.42, -0.16, 0.025], dtype=np.float32),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            extent=np.array([0.10, 0.10, 0.02], dtype=np.float32),
            mask=_mask(),
            points_world=np.zeros((16, 3), dtype=np.float32),
            score=0.7,
            prompt="placement pad",
        ),
    }

    class Perception:
        def estimate_object(self, frame, prompt, *, selector=None):
            if prompt == "cylindrical can":
                key = "object_a"
            elif selector == "world_y_min":
                key = "slot_a"
            else:
                key = "slot_b"
            calls.append((key, selector))
            return object_estimates.get(
                key,
                ObjectEstimate(
                    position=np.array([0.42, 0.16, 0.025], dtype=np.float32),
                    quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    extent=np.array([0.10, 0.10, 0.02], dtype=np.float32),
                    mask=_mask(),
                    points_world=np.zeros((16, 3), dtype=np.float32),
                    score=0.6,
                    prompt="placement pad",
                ),
            )

        def estimate_grasp(self, frame, prompt, *, selector=None):
            raise AssertionError("Hard-SAM config must not call RGB-D grasp planning")

    class Env:
        def __init__(self) -> None:
            self.api_configs = {
                "franka_control_api": {
                    "rgbd_perception_enabled": True,
                    "rgbd_pose_enabled": True,
                    "rgbd_grasp_enabled": False,
                    "rgbd_pose_objects": ["object_a", "slot_a", "slot_b"],
                    "rgbd_grasp_objects": [],
                    "public_grasp_anchor_objects": [
                        "object_a",
                        "object_b",
                        "current_object",
                    ],
                    "public_pose_fallback_objects": [],
                    "cache_rgbd_pose_objects": ["slot_a", "slot_b"],
                    "object_pose_names": {
                        "object_a": "object_a",
                        "object_b": "object_b",
                        "slot_a": "slot_a",
                        "slot_b": "slot_b",
                    },
                    "perception_prompt_map": {
                        "object_a": "cylindrical can",
                        "slot_a": "placement pad",
                        "slot_b": "placement pad",
                    },
                    "perception_selector_map": {
                        "slot_a": "world_y_min",
                        "slot_b": "world_y_max",
                    },
                    "perception_retry_attempts": 1,
                }
            }

        def get_rgbd_frame(self, camera_name):
            return _frame()

        def append_perception_artifact(self, record):
            pass

        def get_public_pose_map(self):
            raise AssertionError("Hard-SAM pose route must not read public GT poses")

        def get_public_grasp_pose(self, object_name, *, grasp_height):
            assert object_name == "object_b"
            return (
                np.array([0.68, 0.26, grasp_height], dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )

        def get_robot_state(self):
            return {
                "ee_pos": [0.40, 0.0, 0.20],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
                "joint": [0.0] * 8,
            }

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    api._rgbd_perception = Perception()

    pos, _quat = api.get_object_pose("object_a")
    np.testing.assert_allclose(pos, [0.58, -0.25, 0.04])
    slot_pos, _slot_quat = api.get_object_pose("slot_a")
    np.testing.assert_allclose(slot_pos, [0.42, -0.16, 0.025])
    cached_slot_pos, _cached_slot_quat = api.get_object_pose("slot_a")
    np.testing.assert_allclose(cached_slot_pos, slot_pos)
    grasp_pos, grasp_quat = api.sample_grasp_pose("object_b")
    np.testing.assert_allclose(grasp_pos, [0.68, 0.26, 0.04])
    np.testing.assert_allclose(grasp_quat, [1.0, 0.0, 0.0, 0.0])

    with pytest.raises(KeyError, match="configured non-privileged pose route"):
        api.get_object_pose("object_b")
    assert calls == [("object_a", None), ("slot_a", "world_y_min")]
