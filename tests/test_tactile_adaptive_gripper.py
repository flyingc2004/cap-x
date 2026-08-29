from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from capx.envs.simulators.univtac import UniVTACLowLevelEnv
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.integrations.tactile.adaptive_gripper import (
    AdaptiveGripperConfig,
    TactileAdaptiveGripperController,
)
from capx.integrations.univtac.franka_compat_api import UniVTACFrankaCompatApi


def _summary(
    *,
    left: bool = False,
    right: bool = False,
    force: float = 0.0,
    left_force: float | None = None,
    right_force: float | None = None,
    balance: float = 0.0,
    slip: float = 0.0,
) -> dict:
    return {
        "contact": left or right,
        "left_contact": left,
        "right_contact": right,
        "normal_force": force,
        "contact_balance": balance,
        "slip_score": slip,
        "event": "stable_grasp" if left and right and force >= 0.2 else "unknown",
        "left": {
            "contact": left,
            "normal_force": force if left_force is None else left_force,
        },
        "right": {
            "contact": right,
            "normal_force": force if right_force is None else right_force,
        },
    }


class _ControllerRig:
    def __init__(self, summaries: list[dict], *, width: float = 1.0) -> None:
        self.width = width
        self.summaries = list(summaries)
        self.summary_index = 0
        self.commands: list[tuple[float, int]] = []

    def get_width(self) -> float:
        return self.width

    def command_width(self, width: float, settle_steps: int) -> None:
        self.width = float(width)
        self.commands.append((self.width, settle_steps))

    def read_summary(self) -> dict:
        index = min(self.summary_index, len(self.summaries) - 1)
        self.summary_index += 1
        return self.summaries[index]


def _controller(rig: _ControllerRig) -> TactileAdaptiveGripperController:
    return TactileAdaptiveGripperController(
        get_width=rig.get_width,
        command_width=rig.command_width,
        read_tactile_summary=rig.read_summary,
        config=AdaptiveGripperConfig(
            coarse_step=0.20,
            fine_step=0.05,
            contact_debounce_frames=2,
            stable_debounce_frames=3,
            release_debounce_frames=2,
            settle_steps_per_command=1,
            max_qpos=0.04,
        ),
    )


def test_adaptive_close_uses_coarse_then_fine_and_holds_for_stability() -> None:
    no_contact = _summary()
    contact = _summary(left=True, right=True, force=0.15)
    stable = _summary(left=True, right=True, force=0.40)
    rig = _ControllerRig([no_contact, no_contact, contact, stable, stable, stable])
    controller = _controller(rig)

    result = controller.close(target_force=0.35, max_steps=20)

    assert result["stable"] is True
    np.testing.assert_allclose([item[0] for item in rig.commands[:3]], [0.80, 0.60, 0.55])
    assert rig.commands[3][0] == pytest.approx(0.55)
    assert rig.commands[4][0] == pytest.approx(0.55)
    assert {item["operation"] for item in controller.trace} == {"close"}
    assert "coarse_close" in [item["phase"] for item in controller.trace]
    assert "contact_debounce" in [item["phase"] for item in controller.trace]
    assert controller.trace[-1]["phase"] == "stable_confirm"


@pytest.mark.parametrize(
    ("summaries", "width", "reason"),
    [
        ([_summary(), _summary()], 0.20, "missed_grasp"),
        (
            [
                _summary(left=True, left_force=0.95),
                _summary(left=True, left_force=0.95),
            ],
            0.50,
            "one_sided_high_force",
        ),
        (
            [_summary(left=True, right=True, force=0.10), _summary()],
            0.05,
            "contact_lost",
        ),
    ],
)
def test_adaptive_close_reports_failure_modes(
    summaries: list[dict],
    width: float,
    reason: str,
) -> None:
    controller = _controller(_ControllerRig(summaries, width=width))
    result = controller.close(max_steps=6)
    assert result["stable"] is False
    assert result["reason"] == reason


def test_adaptive_close_timeout_does_not_move_arm() -> None:
    rig = _ControllerRig([_summary(left=True, right=True, force=0.10)], width=0.80)
    controller = _controller(rig)
    result = controller.close(max_steps=2)
    assert result["reason"] == "max_steps"
    assert all(item["operation"] == "close" for item in controller.trace)
    assert all("arm" not in item and "lift" not in item for item in controller.trace)


def test_adaptive_close_requires_target_force_on_both_hands() -> None:
    asymmetric = _summary(
        left=True,
        right=True,
        force=0.70,
        left_force=0.90,
        right_force=0.50,
        balance=0.29,
    )
    controller = _controller(_ControllerRig([asymmetric], width=0.60))

    result = controller.close(target_force=0.70, max_steps=3)

    assert result["stable"] is False
    assert result["reason"] == "max_steps"


def test_adaptive_open_releases_fine_then_opens_coarse() -> None:
    contact = _summary(left=True, right=True, force=0.30)
    no_contact = _summary()
    rig = _ControllerRig(
        [contact, contact, no_contact, no_contact, no_contact, no_contact],
        width=0.20,
    )
    controller = _controller(rig)

    result = controller.open(target_width=0.60, max_steps=10)

    assert result["released"] is True
    np.testing.assert_allclose([item[0] for item in rig.commands[:4]], [0.25, 0.30, 0.35, 0.55])
    phases = [item["phase"] for item in controller.trace]
    assert phases[:4] == ["fine_release", "fine_release", "release_confirm", "coarse_open"]
    assert {item["operation"] for item in controller.trace} == {"open"}


def test_marker_only_contact_does_not_force_fine_gripper_motion() -> None:
    marker_only = _summary(right=True, force=0.0, right_force=0.0)
    rig = _ControllerRig([marker_only], width=0.20)
    controller = _controller(rig)

    result = controller.open(target_width=0.60, max_steps=4)

    assert result["released"] is True
    assert rig.commands[0][0] == pytest.approx(0.40)
    assert controller.trace[0]["contact"] is False
    assert controller.trace[0]["raw_contact"] is True
    assert controller.trace[0]["phase"] == "coarse_open"


def test_univtac_api_converts_qpos_calibration_to_normalized_steps() -> None:
    class Env:
        task = object()
        api_configs = {
            "franka_control_api": {
                "adaptive_gripper": {
                    "coarse_qpos_step": 0.0005,
                    "fine_qpos_step": 0.00005,
                }
            }
        }

        def get_gripper_calibration(self):
            return {
                "gripper_max_qpos": 0.039,
                "current_qpos": 0.039,
                "current_width": 1.0,
            }

        def command_gripper_width_step(self, width, *, settle_steps):
            return {"ok": True}

    api = UniVTACFrankaCompatApi(Env())
    api._read_adaptive_tactile_summary = lambda: _summary()
    config = api._adaptive_gripper_controller().config

    assert config.coarse_step == pytest.approx(0.0005 / 0.039)
    assert config.fine_step == pytest.approx(0.00005 / 0.039)


def test_univtac_native_tactile_calibration_uses_robot_depth_range() -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._task = SimpleNamespace(
        cfg=SimpleNamespace(
            adaptive_grasp_depth_threshold=27.75,
            robot=SimpleNamespace(
                tactile_far_plane=34.0,
                adaptive_grasp_depth_threshold=27.5,
            )
        )
    )
    env.api_configs = {
        "franka_control_api": {
            "adaptive_gripper": {"depth_contact_margin_mm": 0.75}
        }
    }

    assert env.get_native_tactile_calibration() == {
        "depth_far_plane_mm": 34.0,
        "force_full_scale_mm": 6.25,
        "depth_contact_margin_mm": 0.75,
    }


def test_adaptive_false_preserves_fixed_native_open_close() -> None:
    class Env:
        task = SimpleNamespace(_robot_manager=SimpleNamespace(gripper_max_qpos=0.039))
        api_configs = {"franka_control_api": {"lift_after_close": False}}

        def __init__(self) -> None:
            self.calls = []

        def move_gripper_native(self, **kwargs):
            self.calls.append(kwargs)
            return {"ok": True, "message": "fixed"}

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    close_result = api.close_gripper(adaptive=False)
    open_result = api.open_gripper(adaptive=False, target_width=0.75)

    assert close_result["reason"] == "fixed_close_requires_tactile_confirmation"
    assert open_result["released"] is True
    assert env.calls[0]["opening"] is False
    assert env.calls[1]["opening"] is True
    assert env.calls[1]["qpos"] == pytest.approx(0.75 * 0.039)


def test_config_can_force_adaptive_requests_to_fixed_native_control() -> None:
    class Env:
        task = SimpleNamespace(_robot_manager=SimpleNamespace(gripper_max_qpos=0.039))
        api_configs = {
            "franka_control_api": {
                "lift_after_close": False,
                "tactile_adaptive_gripper_enabled": False,
            }
        }

        def __init__(self) -> None:
            self.calls = []

        def move_gripper_native(self, **kwargs):
            self.calls.append(kwargs)
            return {"ok": True, "message": "fixed"}

    env = Env()
    api = UniVTACFrankaCompatApi(env)
    api._adaptive_gripper_controller = lambda: pytest.fail(
        "disabled adaptive control must not construct a tactile controller"
    )

    close_result = api.close_gripper(adaptive=True)
    open_result = api.open_gripper(adaptive=True)

    assert api.tactile_adaptive_gripper_enabled is False
    assert close_result["reason"] == "fixed_close_requires_tactile_confirmation"
    assert open_result["reason"] == "fixed_open"
    assert [call["opening"] for call in env.calls] == [False, True]


def test_lift_can_ablation_configs_isolate_tactile_access() -> None:
    config_root = Path(__file__).resolve().parents[1] / "env_configs" / "univtac"
    controller_only = yaml.safe_load(
        (config_root / "lift_can_tactile_controller_only.yaml").read_text(encoding="utf-8")
    )["env"]["cfg"]
    no_tactile = yaml.safe_load(
        (config_root / "lift_can_no_tactile.yaml").read_text(encoding="utf-8")
    )["env"]["cfg"]

    for cfg in (controller_only, no_tactile):
        assert cfg["apis"] == ["FrankaControlApi"]
        assert cfg["low_level"]["task_name"] == "lift_can"
        assert cfg["low_level"]["task_config"] == "smoke_capx_lift_can"
        assert "get_tactile_summary" not in cfg["prompt"]
        assert "get_tactile_image" not in cfg["prompt"]

    controller_franka = controller_only["low_level"]["api_configs"]["franka_control_api"]
    fixed_franka = no_tactile["low_level"]["api_configs"]["franka_control_api"]
    assert controller_franka["tactile_adaptive_gripper_enabled"] is True
    assert "close_gripper(adaptive=True" in controller_only["prompt"]
    assert fixed_franka["tactile_adaptive_gripper_enabled"] is False
    assert "close_gripper(adaptive=False)" in no_tactile["prompt"]
    assert "tactile" not in no_tactile["prompt"].lower()


def test_univtac_gripper_hook_uses_force_false_without_action_count_change() -> None:
    class RobotManager:
        device = "cpu"
        gripper_max_qpos = 0.039

        def __init__(self) -> None:
            self.qpos = 0.020
            self.calls = []

        def get_gripper_qpos(self):
            return self.qpos

        def set_gripper(self, position, velocity, *, force):
            self.calls.append((position.clone(), velocity.clone(), force))
            self.qpos = float(position[0].item())

    class Task:
        device = "cpu"
        cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.01))
        take_action_cnt = 4
        step_count = 10

        def __init__(self) -> None:
            self._robot_manager = RobotManager()

        def _step(self, *, is_save):
            assert is_save is True
            self.step_count += 1

    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._task = Task()
    env._last_action_result = {}
    env.refresh_native_observation = lambda **kwargs: {}

    result = env.command_gripper_width_step(0.50, settle_steps=2)

    position, _velocity, force = env._task._robot_manager.calls[-1]
    assert force is False
    assert float(position[0]) == pytest.approx(0.50 * 0.039)
    assert result["action_count"] == 4
    assert env._task.take_action_cnt == 4
    assert env._task.step_count == 12


def test_trace_accepts_only_gripper_operations() -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._tactile_gripper_trace = []
    env.append_tactile_gripper_trace([{"operation": "close"}, {"operation": "open"}])
    assert [item["operation"] for item in env._tactile_gripper_trace] == ["close", "open"]
    with pytest.raises(ValueError, match="close/open"):
        env.append_tactile_gripper_trace([{"operation": "lift"}])


def test_tactile_gripper_trace_exports_beside_trial_artifacts(tmp_path) -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env.task_name = "lift_can"
    env.task_config_name = "smoke_capx_lift_can"
    env._task = SimpleNamespace(metadata={})
    env._debug_records = []
    env._tactile_gripper_trace = [
        {"operation": "close", "phase": "coarse_close"},
        {"operation": "open", "phase": "fine_release"},
    ]
    env._export_pre_move_tactile_timeline = lambda output_dir: None

    exported = env.export_debug_artifacts(tmp_path)

    path = tmp_path / "tactile_gripper_trace.json"
    assert exported == str(path)
    assert [item["operation"] for item in json.loads(path.read_text())] == ["close", "open"]


def test_lift_can_prompt_and_controller_do_not_use_univtac_adaptive_helper() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "env_configs"
        / "univtac"
        / "lift_can_tactile.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = config["env"]["cfg"]
    franka_cfg = cfg["low_level"]["api_configs"]["franka_control_api"]
    adaptive_cfg = franka_cfg["adaptive_gripper"]

    assert cfg["apis"] == ["FrankaControlApi", "UniVTACTactileApi"]
    assert adaptive_cfg["coarse_qpos_step"] == 0.0005
    assert adaptive_cfg["fine_qpos_step"] == 0.00005
    assert adaptive_cfg["contact_force_threshold"] == 0.08
    assert 'close_gripper(adaptive=True, target_force=1.0, max_steps=160)' in cfg["prompt"]
    assert 'can_pos, can_quat = get_object_pose("can")' in cfg["prompt"]
    assert "Do not call exit, quit" in cfg["prompt"]
    source = inspect.getsource(TactileAdaptiveGripperController)
    hook_source = inspect.getsource(UniVTACLowLevelEnv.command_gripper_width_step)
    assert "adaptive_set_gripper" not in source
    assert "adaptive_set_gripper" not in hook_source
    assert "force=False" in hook_source


def test_code_execution_treats_zero_system_exit_as_normal_completion() -> None:
    env = CodeExecutionEnvBase.__new__(CodeExecutionEnvBase)
    env._exec_globals = {}
    env._apis = {}
    env._get_observation = lambda: {}
    env._exec_env_binding = lambda: {}
    env._exec_apis_binding = lambda: {}

    normal = env._exec_user_code("exit()")
    failed = env._exec_user_code("raise SystemExit(2)")

    assert normal["ok"] is True
    assert normal["stderr"] == ""
    assert failed["ok"] is False
    assert "SystemExit: 2" in failed["stderr"]


def test_code_execution_always_exposes_numpy_alias() -> None:
    env = CodeExecutionEnvBase.__new__(CodeExecutionEnvBase)
    env.low_level_env = object()
    env._apis = {}
    env._init_exec_globals()
    env._get_observation = lambda: {}

    result = env._exec_user_code(
        "if False:\n"
        "    import numpy as np\n"
        "RESULT = np.array([0.0, 0.0, 0.05])\n"
    )

    assert result["ok"] is True
    np.testing.assert_array_equal(result["result"], np.array([0.0, 0.0, 0.05]))


def test_code_execution_auto_calls_new_solve_when_model_forgets_call() -> None:
    env = CodeExecutionEnvBase.__new__(CodeExecutionEnvBase)
    env.low_level_env = SimpleNamespace(get_action_count=lambda: 0, get_step_count=lambda: 0)
    env._apis = {}
    env._get_observation = lambda: {}
    env._exec_globals = {"__name__": "__main__", "calls": []}
    env._exec_env_binding = lambda: env.low_level_env
    env._exec_apis_binding = lambda: {}

    result = env._exec_user_code("def solve():\n    calls.append('ran')\n    return 'done'\n")

    assert result["ok"] is True
    assert result["result"] == "done"
    assert env._exec_globals["calls"] == ["ran"]
    assert "auto-calling generated solve()" in result["stdout"]


def test_code_execution_does_not_double_call_explicit_solve() -> None:
    env = CodeExecutionEnvBase.__new__(CodeExecutionEnvBase)
    env.low_level_env = SimpleNamespace(get_action_count=lambda: 0, get_step_count=lambda: 0)
    env._apis = {}
    env._get_observation = lambda: {}
    env._exec_globals = {"__name__": "__main__", "calls": []}
    env._exec_env_binding = lambda: env.low_level_env
    env._exec_apis_binding = lambda: {}

    result = env._exec_user_code(
        "def solve():\n"
        "    calls.append('ran')\n"
        "    return 'done'\n"
        "solve()\n"
    )

    assert result["ok"] is True
    assert env._exec_globals["calls"] == ["ran"]
    assert "auto-calling generated solve()" not in result["stdout"]


def test_code_execution_propagates_timeout_to_runner() -> None:
    env = CodeExecutionEnvBase.__new__(CodeExecutionEnvBase)
    env.low_level_env = SimpleNamespace(get_action_count=lambda: 0, get_step_count=lambda: 0)
    env._apis = {}
    env._get_observation = lambda: {}
    env._exec_globals = {"__name__": "__main__"}
    env._exec_env_binding = lambda: env.low_level_env
    env._exec_apis_binding = lambda: {}

    with pytest.raises(TimeoutError):
        env._exec_user_code("raise TimeoutError('trial timeout')")
