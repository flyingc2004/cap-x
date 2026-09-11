from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml

from capx.envs.simulators.univtac import UniVTACLowLevelEnv
from capx.envs.launch import LaunchArgs
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.envs.trial import (
    _build_trial_working_memory_runtime_context,
    _should_query_multiturn_after_block,
    _was_static_code_rejected_for_full_regeneration,
)
from capx.integrations.univtac.native_tactile import UniVTACTactileFrame, summarize_native_tactile
from capx.integrations.univtac.franka_compat_api import UniVTACFrankaCompatApi
from capx.integrations.univtac.tactile_api import UniVTACTactileApi
from capx.utils.launch_utils import _load_config, _normalize_tactile_memory_config


def _depth(delta_mm: float) -> np.ndarray:
    depth = np.full((16, 16), 34.0, dtype=np.float32)
    if delta_mm > 0.0:
        depth[4:12, 4:12] = 34.0 - delta_mm
    return depth


def _marker(displacement: float) -> np.ndarray:
    initial = np.zeros((8, 8, 2), dtype=np.float32)
    moved = initial.copy()
    moved[..., 0] += displacement
    return np.stack([initial, moved], axis=0)


def _symmetric_marker(displacement: float) -> np.ndarray:
    initial = np.zeros((8, 8, 2), dtype=np.float32)
    moved = initial.copy()
    moved[:, :4, 0] -= displacement
    moved[:, 4:, 0] += displacement
    return np.stack([initial, moved], axis=0)


def _frame(step: int, delta_mm: float = 5.0) -> UniVTACTactileFrame:
    return UniVTACTactileFrame(
        step=step,
        timestamp=float(step),
        left_depth=_depth(delta_mm),
        right_depth=_depth(delta_mm),
        left_marker=_marker(0.1),
        right_marker=_marker(0.1),
        left_pose=None,
        right_pose=None,
    )


class _Buffer:
    def __init__(self) -> None:
        self.frames = [_frame(1)]

    def recent(self, _window: int) -> list[UniVTACTactileFrame]:
        return list(self.frames)

    def clear(self) -> None:
        self.frames = []


class _TactileEnv:
    def __init__(self, api_configs: dict | None = None) -> None:
        self.tactile_buffer = _Buffer()
        self.trace: list[dict] = []
        self.snapshot: dict = {}
        self.api_configs = api_configs or {}

    def refresh_native_observation(self, **_kwargs):
        return {}

    def get_native_tactile_calibration(self) -> dict:
        return {
            "depth_far_plane_mm": 34.0,
            "force_full_scale_mm": 6.5,
            "depth_contact_margin_mm": 0.5,
        }

    def get_step_count(self) -> int:
        return self.tactile_buffer.frames[-1].step if self.tactile_buffer.frames else 0

    def get_robot_state(self) -> dict:
        return {"gripper_qpos": 0.008, "control_frame": "gripper_center"}

    def append_tactile_working_memory_trace(self, record: dict) -> None:
        self.trace.append(record)

    def set_tactile_trial_memory_snapshot(self, memory: dict) -> None:
        self.snapshot = memory


def _record(
    *,
    kind: str = "evidence",
    phase: str = "probe",
    data: dict | None = None,
    capture_ids: list[str] | None = None,
    memory_keys: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "trial_memory.v1",
        "kind": kind,
        "phase": phase,
        "data": data or {},
        "provenance": {
            "capture_ids": capture_ids or [],
            "memory_keys": memory_keys or [],
        },
    }


def test_capture_contains_only_public_tactile_and_embodiment_fields() -> None:
    api = UniVTACTactileApi(_TactileEnv())

    capture = api.capture_tactile_observation(window=20)

    assert set(capture) == {
        "schema_version",
        "capture_id",
        "step",
        "window",
        "tactile",
        "marker_motion",
        "embodiment",
    }
    assert capture["schema_version"] == "tactile_observation.v1"
    assert capture["tactile"]["left_contact"] is True
    assert capture["tactile"]["right_contact"] is True
    assert capture["tactile"]["left_normal_force"] > 0.0
    assert capture["tactile"]["right_normal_force"] > 0.0
    assert capture["tactile"]["left_contact_area"] > 0.0
    assert capture["tactile"]["right_contact_area"] > 0.0
    assert capture["tactile"]["left_depth_mm"] == capture["tactile"]["left_depth_delta_mm"]
    assert capture["tactile"]["right_depth_mm"] == capture["tactile"]["right_depth_delta_mm"]
    assert capture["marker_motion"]["left_marker_displacement_px"] == pytest.approx(0.1)
    assert capture["marker_motion"]["right_marker_displacement_px"] == pytest.approx(0.1)
    assert capture["marker_motion"]["left_marker_coherence"] == pytest.approx(1.0)
    assert capture["marker_motion"]["right_marker_coherence"] == pytest.approx(1.0)
    assert capture["embodiment"] == {
        "gripper_qpos": 0.008,
        "control_frame": "gripper_center",
    }
    serialized = repr(capture).lower()
    for private_name in ("label", "density", "friction", "reward", "success", "metadata"):
        assert private_name not in serialized


def test_capture_exposes_raw_marker_pixels_without_changing_legacy_motion_units() -> None:
    env = _TactileEnv()
    frame = _frame(1)
    frame.left_marker = _marker(16.0)
    frame.right_marker = _marker(16.0)
    env.tactile_buffer.frames = [frame]

    capture = UniVTACTactileApi(env).capture_tactile_observation()

    assert capture["marker_motion"]["left_marker_displacement_px"] == pytest.approx(16.0)
    assert capture["marker_motion"]["right_marker_displacement_px"] == pytest.approx(16.0)
    # Existing motion fields remain normalized for old policies.
    assert capture["marker_motion"]["left_marker_mean_displacement"] == pytest.approx(0.05)


def test_marker_coherence_distinguishes_unidirectional_and_symmetric_flow() -> None:
    aligned = _frame(1)
    symmetric = _frame(2)
    symmetric.left_marker = _symmetric_marker(2.0)
    symmetric.right_marker = _symmetric_marker(2.0)

    aligned_summary = summarize_native_tactile([aligned], depth_far_plane_mm=34.0)
    symmetric_summary = summarize_native_tactile([symmetric], depth_far_plane_mm=34.0)

    assert aligned_summary["left"]["marker_coherence"] == pytest.approx(1.0)
    assert symmetric_summary["left"]["marker_coherence"] == pytest.approx(0.0)


def test_measurement_protocol_is_public_config_and_not_task_state() -> None:
    api = UniVTACTactileApi(
        _TactileEnv(
            {
                "tactile_measurement_protocol": {
                    "capture_window": 11,
                    "settle_steps": 7,
                    "probe_lift_m": 0.012,
                    "probe_hold_steps": 5,
                    "max_attempts_per_object": 2,
                    "adaptive_close": True,
                }
            }
        )
    )

    protocol = api.get_tactile_measurement_protocol()

    assert protocol == {
        "schema_version": "tactile_measurement_protocol.v1",
        "capture_window": 11,
        "settle_steps": 7,
        "probe_lift_m": 0.012,
        "probe_hold_steps": 5,
        "max_attempts_per_object": 2,
        "adaptive_close": True,
    }
    assert "label" not in repr(protocol).lower()


def test_trial_memory_persists_and_records_capture_write_clear_audit() -> None:
    env = _TactileEnv()
    api = UniVTACTactileApi(env)
    observation = api.capture_tactile_observation()

    stored = api.write_trial_memory(
        "reference",
        _record(
            phase="reference_probe",
            data={
                "static": {
                    "capture_id": observation["capture_id"],
                    "chosen_features": ["depth_delta_mm", "contact_area"],
                },
                "rationale": "bilateral indentation is the current hypothesis",
            },
            capture_ids=[observation["capture_id"]],
        ),
    )

    assert stored["schema_version"] == "trial_memory_operation.v1"
    assert stored["operation"] == "write"
    assert stored["ok"] is True
    assert api.read_trial_memory("reference")["data"]["static"]["capture_id"] == observation["capture_id"]
    assert api.list_trial_memory()["schema_version"] == "trial_memory_snapshot.v1"
    assert "reference" in api.list_trial_memory()["records"]
    assert env.snapshot == api.list_trial_memory()
    assert [record["event"] for record in env.trace] == ["capture", "write", "read"]
    assert all(record["schema_version"] == "trial_memory_event.v1" for record in env.trace)

    cleared = api.clear_trial_memory()

    assert cleared["operation"] == "clear"
    assert api.list_trial_memory() == {
        "schema_version": "trial_memory_snapshot.v1",
        "records": {},
    }
    assert env.snapshot == api.list_trial_memory()
    assert [record["event"] for record in env.trace] == ["capture", "write", "read", "clear"]


def test_trial_memory_rejects_images_non_json_and_oversized_payloads() -> None:
    api = UniVTACTactileApi(_TactileEnv())

    with pytest.raises(TypeError, match="numpy.ndarray"):
        api.write_trial_memory("image", _record(data={"raw": np.zeros((8, 8), dtype=np.uint8)}))
    with pytest.raises(TypeError, match="JSON-safe"):
        api.write_trial_memory("bad", _record(data={"values": {1, 2, 3}}))
    with pytest.raises(ValueError, match="entry exceeds"):
        api.write_trial_memory("oversized", _record(data={"text": "x" * (9 * 1024)}))

    for index in range(4):
        api.write_trial_memory(f"entry_{index}", _record(data={"text": "x" * 7000}))
    with pytest.raises(ValueError, match="total limit"):
        api.write_trial_memory("overflow", _record(data={"text": "x" * 7000}))

    with pytest.raises(ValueError, match="schema_version"):
        api.write_trial_memory("wrong_schema", {**_record(), "schema_version": "v0"})
    with pytest.raises(ValueError, match="kind"):
        api.write_trial_memory("wrong_kind", {**_record(), "kind": "profile"})
    with pytest.raises(ValueError, match="provenance"):
        api.write_trial_memory(
            "bad_provenance",
            {**_record(), "provenance": {"capture_ids": []}},
        )


def test_reset_clears_memory_and_resets_capture_identifier() -> None:
    env = _TactileEnv()
    api = UniVTACTactileApi(env)
    capture = api.capture_tactile_observation()
    api.write_trial_memory(
        "reference",
        _record(capture_ids=[capture["capture_id"]], data={"probe": "complete"}),
    )

    api.reset_episode()

    assert api.list_trial_memory() == {
        "schema_version": "trial_memory_snapshot.v1",
        "records": {},
    }
    assert env.snapshot == api.list_trial_memory()
    assert [record["event"] for record in env.trace][-1] == "clear"


def test_public_functions_expose_neutral_agent_owned_memory_only() -> None:
    functions = UniVTACTactileApi.__new__(UniVTACTactileApi).functions()
    api = UniVTACTactileApi(_TactileEnv())

    assert {
        "capture_tactile_observation",
        "get_tactile_measurement_protocol",
        "write_trial_memory",
        "read_trial_memory",
        "list_trial_memory",
        "clear_trial_memory",
    } <= set(functions)
    for legacy_name in (
        "remember_tactile_signature",
        "get_tactile_grasp_profile",
        "compare_tactile_signatures",
        "list_tactile_signatures",
        "clear_tactile_signatures",
    ):
        assert legacy_name not in functions
        assert not hasattr(api, legacy_name)


def test_runtime_memory_context_is_bounded_and_used_only_when_enabled() -> None:
    api = UniVTACTactileApi(_TactileEnv())
    api.write_trial_memory("reference", _record(data={"rationale": "touch"}))
    api.write_trial_memory("candidate_left", _record(data={"rationale": "same protocol"}))
    api.write_trial_memory("candidate_right", _record(data={"rationale": "same protocol"}))

    env = CodeExecutionEnvBase.__new__(CodeExecutionEnvBase)
    env._apis = {"UniVTACTactileApi": api}
    context = env.get_runtime_memory_context(max_chars=1024)

    assert len(context) <= 1024
    assert "trial_memory_snapshot.v1" in context
    assert "records" in context
    assert "reference" in context
    assert _build_trial_working_memory_runtime_context(
        env, {"include_trial_memory_in_multiturn": False}
    ) is None
    injected = _build_trial_working_memory_runtime_context(
        env,
        {
            "include_trial_memory_in_multiturn": True,
            "multiturn_trial_memory_max_chars": 1024,
        },
    )
    assert injected is not None
    assert "agent-authored" in injected
    assert "raw" not in injected.lower()


def test_low_level_env_keeps_public_trial_memory_snapshot() -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    snapshot = {
        "schema_version": "trial_memory_snapshot.v1",
        "records": {"reference": _record(data={"chosen_features": ["area"]})},
    }
    env.set_tactile_trial_memory_snapshot(snapshot)

    assert env.get_tactile_trial_memory_snapshot() == snapshot


def test_tactile_memory_config_normalization() -> None:
    assert _normalize_tactile_memory_config(None) == {
        "configured": False,
        "trial": {"enabled": False, "include_in_multiturn": False, "max_context_chars": 4000},
        "persistent": {"enabled": False},
    }
    assert _normalize_tactile_memory_config(
        {
            "trial": {"enabled": True, "include_in_multiturn": True, "max_context_chars": 4000},
            "persistent": {"enabled": False},
        }
    ) == {
        "configured": True,
        "trial": {"enabled": True, "include_in_multiturn": True, "max_context_chars": 4000},
        "persistent": {"enabled": False},
    }


def test_config_loader_maps_trial_memory_and_disables_persistent_injection() -> None:
    root = Path(__file__).resolve().parents[1]
    env_factory, config, _ = _load_config(
        LaunchArgs(
            config_path=str(root / "env_configs/univtac/tactile_memory_match_easy_sam_gt.yaml")
        )
    )

    assert config["tactile_memory"] == {
        "configured": True,
        "trial": {"enabled": True, "include_in_multiturn": True, "max_context_chars": 4000},
        "persistent": {"enabled": False},
    }
    assert config["include_trial_memory_in_multiturn"] is True
    assert config["multiturn_trial_memory_max_chars"] == 4000
    assert config["tactile_code_memory"]["enabled"] is False
    assert config["tactile_strategy_memory"] is False
    assert "Tactile code memory (compact skill cards):" not in env_factory["cfg"]["prompt"]


def test_native_goto_pose_clamps_target_below_safe_z() -> None:
    class _NativeEnv:
        api_configs = {
            "franka_control_api": {
                "use_native_pose_planner": True,
                "min_safe_z": 0.035,
                "native_min_ee_z": 0.035,
                "use_task_grasp_actor_for_objects": False,
                "max_native_pose_actions": 2,
            }
        }

        def __init__(self) -> None:
            self.targets: list[np.ndarray] = []

        def move_to_tool_pose_native(self, position, quaternion):
            self.targets.append(np.asarray(position, dtype=np.float32))
            return {"ok": True}

    env = _NativeEnv()
    api = UniVTACFrankaCompatApi(env)

    result = api.goto_pose(
        np.array([0.680, 0.242, -0.030], dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert result["ok"] is True
    assert len(env.targets) == 1
    assert env.targets[0][2] == pytest.approx(0.035)


def test_native_planner_resolves_sampled_object_pose_through_task_grasp_atom() -> None:
    grasp_pos = np.array([0.535, -0.174, 0.013], dtype=np.float32)
    grasp_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    class _Env:
        api_configs = {
            "franka_control_api": {
                "use_native_pose_planner": True,
                "use_task_grasp_actor_for_objects": True,
                "grasp_xy_tolerance": 0.08,
                "grasp_z_tolerance": 0.04,
            }
        }

        def __init__(self) -> None:
            self.approach_calls: list[dict[str, object]] = []

        def get_robot_state(self):
            return {
                "ee_pos": [0.4, 0.0, 0.2],
                "ee_quat": [1.0, 0.0, 0.0, 0.0],
                "joint": [0.0] * 8,
            }

        def get_public_grasp_pose(self, name, *, grasp_height):
            assert name == "candidate_right"
            return grasp_pos, grasp_quat

        def get_public_pose_map(self):
            return {
                "candidate_right": (
                    grasp_pos,
                    grasp_quat,
                    np.array([0.04, 0.04, 0.12], dtype=np.float32),
                )
            }

        def approach_grasped_actor(self, **kwargs):
            self.approach_calls.append(kwargs)
            return {"ok": True, "message": "task grasp atom"}

        def move_to_tool_pose_native(self, *_args):
            raise AssertionError("sampled object pose must not use generic native planner")

    env = _Env()
    api = UniVTACFrankaCompatApi(env)
    result = api.goto_pose(grasp_pos, grasp_quat)

    assert result["ok"] is True
    assert len(env.approach_calls) == 1
    assert env.approach_calls[0]["object_name"] == "candidate_right"


def test_low_level_native_move_clamps_converted_ee_target(monkeypatch) -> None:
    class _Pose:
        def __init__(self, p, q) -> None:
            self.p = np.asarray(p, dtype=np.float32)
            self.q = np.asarray(q, dtype=np.float32)

    transforms = types.ModuleType("envs.utils.transforms")
    transforms.Pose = _Pose
    utils = types.ModuleType("envs.utils")
    utils.transforms = transforms
    envs = types.ModuleType("envs")
    envs.utils = utils
    monkeypatch.setitem(sys.modules, "envs", envs)
    monkeypatch.setitem(sys.modules, "envs.utils", utils)
    monkeypatch.setitem(sys.modules, "envs.utils.transforms", transforms)

    class _RobotManager:
        def gripper_center_to_ee(self, pose):
            return _Pose([pose.p[0], pose.p[1], pose.p[2] - 0.045], pose.q)

        def get_gripper_qpos(self):
            return 0.01

    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env.api_configs = {"franka_control_api": {"native_min_ee_z": 0.035}}
    env._task = types.SimpleNamespace(_robot_manager=_RobotManager())
    env.protocol_action_allowed = lambda: True
    actions: list[np.ndarray] = []
    env.take_action = lambda action, *, action_type: (
        actions.append(np.asarray(action, dtype=np.float32)) or {"ok": True}
    )

    result = env.move_to_tool_pose_native(
        [0.680, 0.242, 0.035],
        [1.0, 0.0, 0.0, 0.0],
    )

    assert result["ok"] is True
    assert actions[0][2] == pytest.approx(0.035)


def test_memory_match_configs_use_agent_owned_memory_and_only_differ_in_pose_source() -> None:
    root = Path(__file__).resolve().parents[1]
    configs = {
        "easy": yaml.safe_load(
            (root / "env_configs/univtac/tactile_memory_match_easy_sam_gt.yaml").read_text()
        ),
        "hard": yaml.safe_load(
            (root / "env_configs/univtac/tactile_memory_match_hard_sam.yaml").read_text()
        ),
        "easy_compat": yaml.safe_load(
            (root / "env_configs/univtac/tactile_memory_match_easy_sam_gt_memory.yaml").read_text()
        ),
        "hard_compat": yaml.safe_load(
            (root / "env_configs/univtac/tactile_memory_match_hard_sam_memory.yaml").read_text()
        ),
    }
    legacy_names = (
        "remember_tactile_signature",
        "get_tactile_grasp_profile",
        "compare_tactile_signatures",
    )

    for config in configs.values():
        cfg = config["env"]["cfg"]
        low_level = cfg["low_level"]
        assert cfg["apis"] == ["FrankaControlApi", "UniVTACTactileApi"]
        assert cfg["stream_user_code_output"] is False
        assert low_level["expose_actor_pose"] is False
        assert low_level["privileged"] is False
        assert config["tactile_memory"] == {
            "trial": {"enabled": True, "include_in_multiturn": True, "max_context_chars": 4000},
            "persistent": {"enabled": False},
        }
        assert "tactile_code_memory" not in config
        assert "tactile_strategy_memory" not in config
        assert "include_trial_memory_in_multiturn" not in config
        assert low_level["api_configs"]["franka_control_api"]["native_min_ee_z"] == 0.035
        assert "capture_tactile_observation" in cfg["prompt"]
        assert "get_tactile_measurement_protocol" in cfg["prompt"]
        assert "write_trial_memory" in cfg["prompt"]
        assert "sample_grasp_pose(name)" in cfg["prompt"]
        assert "never a direct gripper contact target" in cfg["prompt"]
        assert 'capture["tactile"]["left_depth_mm"]' in cfg["prompt"]
        assert 'capture["marker_motion"]["left_marker_displacement_px"]' in cfg["prompt"]
        assert 'capture["marker_motion"]["left_marker_coherence"]' in cfg["prompt"]
        assert "trial_memory.v1" in cfg["prompt"]
        assert "score_margin" in cfg["prompt"]
        assert "probe(object_name)" in cfg["prompt"]
        assert "protocol-permitted attempt" in cfg["prompt"]
        assert "retry exactly once" not in cfg["prompt"]
        assert "probe_spec" in cfg["prompt"]
        assert "answer only executable python" in cfg["prompt"].lower()
        assert "StaticCodeError" in cfg["multi_turn_prompt"]
        assert "remaining one symmetric remeasurement" in cfg["multi_turn_prompt"]
        assert "fixed formula" not in cfg["prompt"].lower()
        assert "read-only tactile code memory" not in cfg["prompt"]
        for legacy_name in legacy_names:
            assert legacy_name not in cfg["prompt"]
            assert legacy_name not in cfg["multi_turn_prompt"]
        assert config["regenerate_full_chain_on_static_failure"] is True

    easy_franka = configs["easy"]["env"]["cfg"]["low_level"]["api_configs"]["franka_control_api"]
    hard_franka = configs["hard"]["env"]["cfg"]["low_level"]["api_configs"]["franka_control_api"]
    easy_protocol = configs["easy"]["env"]["cfg"]["low_level"]["api_configs"]["tactile_measurement_protocol"]
    hard_protocol = configs["hard"]["env"]["cfg"]["low_level"]["api_configs"]["tactile_measurement_protocol"]
    assert easy_franka["public_anchor_pose_enabled"] is True
    assert hard_franka["public_anchor_pose_enabled"] is False
    assert easy_protocol == hard_protocol
    assert easy_protocol["max_attempts_per_object"] == 2
    assert easy_protocol["adaptive_close"] is True
    assert 'source="anchor"' in configs["easy"]["env"]["cfg"]["prompt"]
    assert 'source="rgbd"' in configs["hard"]["env"]["cfg"]["prompt"]
    for prefix in ("easy", "hard"):
        assert configs[prefix]["env"]["cfg"]["prompt"] == configs[f"{prefix}_compat"]["env"]["cfg"]["prompt"]


def test_failure_only_multiturn_skips_clean_intermediate_blocks() -> None:
    clean = {
        "sandbox_rc": 0,
        "stdout": "CAPX_EVENT object=reference_object phase=release",
        "stderr": "",
        "task_completed": False,
    }

    assert not _should_query_multiturn_after_block(
        clean,
        code_block_idx=1,
        total_code_blocks=3,
        config={"multi_turn_on_failure_only": True},
    )
    assert _should_query_multiturn_after_block(
        {**clean, "stdout": clean["stdout"] + "\nCAPX_FAILURE reason=ambiguous"},
        code_block_idx=1,
        total_code_blocks=3,
        config={"multi_turn_on_failure_only": True},
    )


def test_static_code_rejection_only_regenerates_full_chain_when_enabled() -> None:
    config = {"regenerate_full_chain_on_static_failure": True}

    assert _was_static_code_rejected_for_full_regeneration(config, "StaticCodeError: bad name")
    assert _was_static_code_rejected_for_full_regeneration(config, "SyntaxError: invalid syntax")
    assert not _was_static_code_rejected_for_full_regeneration(config, "RuntimeError: grasp failed")
    assert not _was_static_code_rejected_for_full_regeneration({}, "StaticCodeError: bad name")
