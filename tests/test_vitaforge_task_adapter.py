from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from capx.envs.simulators.univtac import UniVTACLowLevelEnv
from capx.integrations.univtac.franka_compat_api import UniVTACFrankaCompatApi


class _Pose:
    def __init__(self, position: list[float]) -> None:
        self.p = np.asarray(position, dtype=np.float32)
        self.q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class _Actor:
    def __init__(self, position: list[float]) -> None:
        self._pose = _Pose(position)

    def get_pose(self) -> _Pose:
        return self._pose


class _Task:
    def __init__(self) -> None:
        self.smooth_block = _Actor([0.40, -0.05, 0.002])
        self.rough_block = _Actor([0.40, 0.05, 0.002])
        self.active_cans = {
            "fanta": _Actor([0.50, 0.02, 0.004]),
            "pepsi": _Actor([0.50, 0.20, 0.004]),
        }


def _bare_env(public_entities: dict) -> UniVTACLowLevelEnv:
    env = object.__new__(UniVTACLowLevelEnv)
    env.task_name = "roughness_regrasp"
    env._task = _Task()
    env.public_entities = env._normalize_public_entities(public_entities)
    env._public_pose_cache = {}
    return env


def test_task_cfg_overrides_only_update_declared_fields() -> None:
    class Cfg:
        rough_block_side = "right"

    env = object.__new__(UniVTACLowLevelEnv)
    env.task_name = "roughness_regrasp"
    env.task_cfg_overrides = {"rough_block_side": "random"}
    cfg = Cfg()

    env._apply_task_cfg_overrides(cfg)

    assert cfg.rough_block_side == "random"

    env.task_cfg_overrides = {"metadata": {"leak": True}}
    with pytest.raises(ValueError, match="unknown TaskCfg field"):
        env._apply_task_cfg_overrides(cfg)


def test_public_actor_selector_uses_internal_easy_anchor_only() -> None:
    env = _bare_env(
        {
            "left_block": {
                "source": "easy_actor_anchor",
                "actor_paths": ["smooth_block", "rough_block"],
                "selector_axis": "y",
                "selector_order": "max",
                "extent": [0.04, 0.04, 0.04],
            },
            "can_near": {
                "source": "easy_actor_anchor",
                "actor_paths": ["active_cans.fanta", "active_cans.pepsi"],
                "selector_axis": "y",
                "selector_order": "min",
                "extent": [0.07, 0.07, 0.13],
            },
        }
    )

    env._refresh_public_pose_cache()

    np.testing.assert_allclose(env._public_pose_cache["left_block"]["position"], [0.40, 0.05, 0.002])
    np.testing.assert_allclose(env._public_pose_cache["can_near"]["position"], [0.50, 0.02, 0.004])
    assert env._public_pose_cache["left_block"]["source"] == "easy_actor_anchor"


def test_static_region_has_no_live_actor_dependency_and_exposes_grasp_pose() -> None:
    env = _bare_env(
        {
            "left_probe_region": {
                "source": "public_coarse_region",
                "position": [0.40, 0.05, 0.002],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "extent": [0.05, 0.05, 0.04],
                "grasp_pose": {
                    "position": [0.40, 0.05, 0.040],
                    "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
                },
            }
        }
    )

    env._task = object()
    env._refresh_public_pose_cache()
    position, quaternion = env.get_public_grasp_pose("left_probe_region")

    np.testing.assert_allclose(env._public_pose_cache["left_probe_region"]["position"], [0.40, 0.05, 0.002])
    np.testing.assert_allclose(position, [0.40, 0.05, 0.040])
    np.testing.assert_allclose(quaternion, [0.0, 1.0, 0.0, 0.0])


def test_configured_grasp_targets_are_considered_by_native_approach() -> None:
    class Env:
        api_configs = {
            "franka_control_api": {
                "public_grasp_anchor_objects": ["can_near"],
                "grasp_xy_tolerance": 0.02,
                "grasp_z_tolerance": 0.02,
            }
        }

        def get_public_pose_map(self):
            return {
        "can_near": (
            np.asarray([0.50, 0.02, 0.002], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.07, 0.07, 0.13], dtype=np.float32),
        )
            }

        def get_public_grasp_pose(self, name, *, grasp_height):
            assert name == "can_near"
            return (
                np.asarray([0.50, 0.02, 0.083], dtype=np.float32),
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )

        def get_robot_state(self):
            return {
                "ee_pos": [0.50, 0.02, 0.10],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
                "joint": [0.0] * 8,
            }

    api = UniVTACFrankaCompatApi(Env())

    target = api._nearest_public_grasp_target(np.asarray([0.50, 0.02, 0.083]))

    assert target is not None
    assert target[0] == "can_near"


def test_vitaforge_configs_expose_only_franka_and_tactile_apis() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "env_configs" / "vitaforge"
    expected = {
        "place_cube_on_colored_area_easy_gt.yaml",
        "place_cube_on_colored_area_hard_sam.yaml",
        "roughness_regrasp_easy_gt.yaml",
        "roughness_regrasp_hard_touch.yaml",
        "can_empty_select_easy_gt.yaml",
        "can_empty_select_hard_sam.yaml",
    }

    assert {path.name for path in config_dir.glob("*.yaml")} >= expected
    for name in expected:
        config = yaml.safe_load((config_dir / name).read_text(encoding="utf-8"))
        cfg = config["env"]["cfg"]
        low_level = cfg["low_level"]
        assert cfg["apis"] == ["FrankaControlApi", "UniVTACTactileApi"]
        assert low_level["expose_actor_pose"] is False
        assert low_level["privileged"] is False
        assert cfg["privileged"] is False
        assert low_level["task_config_overrides"]["observations"]["actor"] is False
        assert config["tactile_memory"]["persistent"]["enabled"] is False

    rough_hard = yaml.safe_load(
        (config_dir / "roughness_regrasp_hard_touch.yaml").read_text(encoding="utf-8")
    )
    rough_low_level = rough_hard["env"]["cfg"]["low_level"]
    assert rough_low_level["api_configs"]["franka_control_api"]["rgbd_perception_enabled"] is False
    assert rough_low_level["public_entities"]["left_block"]["source"] == "public_coarse_region"
    assert "actor" not in rough_low_level["public_entities"]["left_block"]
    assert rough_low_level["task_cfg_overrides"] == {
        "rough_block_side": "random",
        "initial_grasp_side": "random",
    }

    can_hard = yaml.safe_load((config_dir / "can_empty_select_hard_sam.yaml").read_text())
    can_hard_low_level = can_hard["env"]["cfg"]["low_level"]
    can_hard_api = can_hard_low_level["api_configs"]["franka_control_api"]
    assert can_hard_low_level["task_cfg_overrides"] == {"empty_can": "random"}
    assert can_hard_api["rgbd_perception_enabled"] is True
    assert can_hard_api["public_grasp_anchor_objects"] == []
    assert set(can_hard_api["rgbd_pose_objects"]) == {"can_near", "can_far"}
