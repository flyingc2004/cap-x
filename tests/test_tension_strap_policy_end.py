"""Isaac-free checks for the agent-owned OpenTacBench END handshake."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from capx.envs.simulators.univtac import UniVTACLowLevelEnv
from capx.envs.tasks.base import CodeExecutionEnvBase


def _execution_env(*, accepted: bool = True, policy_owned: bool = True):
    calls = {"check": 0, "finish": 0}

    def check_success():
        calls["check"] += 1
        return True

    def finish_episode():
        calls["finish"] += 1
        return accepted

    low = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    low._task = SimpleNamespace(
        check_success=check_success,
        check_early_stop=lambda: False,
        finish_episode=finish_episode,
    )
    low.policy_controls_termination = policy_owned
    low._policy_end_submitted = False
    low._policy_end_verdict = False
    low._force_task_mode = False
    low._official_task_protocol = True
    low._protocol_stopped = False
    low._protocol_stop_reason = None
    low.max_steps = 4000
    low.get_action_count = lambda: 0
    low.get_step_count = lambda: 0

    env = CodeExecutionEnvBase.__new__(CodeExecutionEnvBase)
    env.cfg = SimpleNamespace(stream_user_code_output=False)
    env.low_level_env = low
    env._apis = {}
    env._step_count = 0
    env._task_prompt = "test task"
    env._get_observation = lambda: {}
    env._exec_env_binding = lambda: None
    env._init_exec_globals()
    return env, low, calls


def test_physical_acceptance_does_not_stop_policy_before_agent_end() -> None:
    env, low, calls = _execution_env()

    low._apply_official_task_protocol()
    _, reward, terminated, _, _ = env.step("pass")

    assert calls == {"check": 0, "finish": 0}
    assert low._protocol_stopped is False
    assert reward == 0.0
    assert terminated is False
    assert env._exec_globals["capx_stop"] is False

    _, reward, terminated, _, info = env.step("capx_stop = True")
    assert calls == {"check": 0, "finish": 1}
    assert reward == 1.0
    assert terminated is True
    assert info["task_completed"] is True
    assert low._protocol_stop_reason == "policy_end"
    with pytest.raises(RuntimeError, match="already submitted"):
        low.submit_policy_end()


def test_failed_code_and_stale_stop_flag_cannot_submit_end() -> None:
    env, low, calls = _execution_env()
    env._exec_globals["capx_stop"] = True
    _, reward, terminated, _, _ = env.step("pass")
    assert reward == 0.0 and terminated is False
    assert env._exec_globals["capx_stop"] is False

    _, reward, terminated, _, info = env.step("capx_stop = True\nraise RuntimeError('broken')")
    assert info["sandbox_rc"] == 1
    assert reward == 0.0 and terminated is False
    assert calls["finish"] == 0
    assert low._protocol_stopped is False

    _, reward, terminated, _, info = env.step("capx_stop = True\nraise SystemExit(0)")
    assert info["sandbox_rc"] == 0
    assert reward == 0.0 and terminated is False
    assert calls["finish"] == 0


def test_rejected_end_terminates_without_claiming_success() -> None:
    env, low, calls = _execution_env(accepted=False)

    _, reward, terminated, _, info = env.step("capx_stop = True")

    assert calls["finish"] == 1
    assert reward == 0.0
    assert terminated is True
    assert info["task_completed"] is False
    assert low._protocol_stop_reason == "policy_end"


def test_legacy_task_keeps_native_success_stop() -> None:
    env, low, calls = _execution_env(policy_owned=False)

    low._apply_official_task_protocol()

    assert calls["check"] == 1
    assert calls["finish"] == 0
    assert low._protocol_stop_reason == "native_success"
    assert "capx_stop" not in env._exec_globals
