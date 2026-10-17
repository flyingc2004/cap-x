from __future__ import annotations

import signal
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from capx.envs import trial
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.envs.tasks.exceptions import CodeBlockTimeout


@contextmanager
def _trial_alarm(seconds: float, handler=None):
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, handler or (lambda *_args: None))
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def test_pure_python_loop_times_out_and_restores_trial_clock(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LLM_TIMEOUT_SECONDS", "1")
    code = "while True:\n    pass"
    with _trial_alarm(100):
        previous_trace = sys.gettrace()
        with pytest.raises(CodeBlockTimeout) as caught:
            with trial._code_block_wall_watchdog(
                {"code_block_wall_timeout_seconds": 0.05},
                code,
                1,
                can_regenerate=True,
            ):
                exec(code, {})
        remaining, _ = signal.getitimer(signal.ITIMER_REAL)

    assert sys.gettrace() is previous_trace
    assert caught.value.details["line_number"] in (1, 2)
    assert caught.value.details["source_line"] in ("while True:", "pass")
    assert 90 < remaining < 100


def test_action_loop_timeout_reports_counts_to_multiturn(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LLM_TIMEOUT_SECONDS", "1")
    counts = {"actions": 0, "steps": 0}

    def act() -> None:
        counts["actions"] += 1
        counts["steps"] += 3
        time.sleep(0.005)

    code = "while True:\n    act()"
    with _trial_alarm(100):
        with pytest.raises(CodeBlockTimeout) as caught:
            with trial._code_block_wall_watchdog(
                {"code_block_wall_timeout_seconds": 0.05},
                code,
                2,
                can_regenerate=True,
            ):
                exec(code, {"act": act})

    low_level = SimpleNamespace(
        get_action_count=lambda: counts["actions"],
        get_step_count=lambda: counts["steps"],
    )
    env = SimpleNamespace(
        low_level_env=low_level,
        _interrupted_code_output={"stdout": "probe reached", "stderr": ""},
    )
    info = trial._code_block_timeout_info(
        env,
        caught.value,
        elapsed=0.05,
        action_start=0,
        step_start=0,
    )
    assert counts["actions"] > 0
    assert info["sandbox_rc"] == 1
    assert f"used_actions={counts['actions']}" in info["stdout"]
    assert f"used_sim_steps={counts['steps']}" in info["stdout"]
    assert "generated_line=2" in info["stdout"]
    assert "Check the loop exit condition" in info["stderr"]
    assert trial._should_query_multiturn_after_block(
        info,
        code_block_idx=1,
        total_code_blocks=1,
        config={"multi_turn_on_failure_only": True},
    )


def test_interrupted_code_keeps_partial_stdout() -> None:
    executor = SimpleNamespace(
        cfg=SimpleNamespace(stream_user_code_output=False),
        _apis={},
        _exec_globals={
            "stop": lambda: (_ for _ in ()).throw(
                CodeBlockTimeout("timed out", details={})
            )
        },
        _get_observation=lambda: {},
        _exec_env_binding=lambda: None,
        _exec_apis_binding=lambda: {},
        _find_static_undefined_globals=lambda _code: [],
        _low_level_progress_marker=lambda: (),
    )
    with pytest.raises(CodeBlockTimeout):
        CodeExecutionEnvBase._exec_user_code(executor, "print('probe reached')\nstop()")
    assert "probe reached" in executor._interrupted_code_output["stdout"]


def test_generated_except_exception_cannot_swallow_watchdog(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LLM_TIMEOUT_SECONDS", "1")
    code = "while True:\n    try:\n        pass\n    except Exception:\n        continue"
    with _trial_alarm(100):
        with pytest.raises(CodeBlockTimeout):
            with trial._code_block_wall_watchdog(
                {"code_block_wall_timeout_seconds": 0.05},
                code,
                1,
                can_regenerate=True,
            ):
                exec(code, {})


def test_timeout_diagnostic_is_in_bounded_llm_repair_prompt(monkeypatch) -> None:
    prompts = []

    def fake_query(_args, prompt):
        prompts.append(prompt)
        return {"content": "FINISH", "reasoning": None}

    monkeypatch.setattr(trial, "_query_model", fake_query)
    info = {
        "sandbox_rc": 1,
        "stdout": "CAPX_FAILURE reason=code_block_timeout used_actions=407 generated_line=20",
        "stderr": "CodeBlockTimeout: inspect the loop exit condition",
        "task_completed": False,
    }
    config = {
        "use_visual_feedback": False,
        "use_img_differencing": False,
        "use_video_differencing": False,
        "use_parallel_ensemble": False,
        "filter_multiturn_console": True,
        "multiturn_console_max_lines": 10,
        "multiturn_console_max_chars": 1000,
    }
    decision, *_ = trial._handle_multi_turn_step(
        env=SimpleNamespace(),
        obs={"full_prompt": [{"role": "user", "content": [{"type": "text", "text": "task"}]}]},
        args=SimpleNamespace(model="test", use_legacy_multi_turn_decision_prompt=False),
        config=config,
        visual_differencing_args=SimpleNamespace(model="test"),
        multi_turn_prompt="Executed: {executed_code}\n{console_stdout}\n{console_stderr}",
        code_blocks=["while True:\n    pass"],
        code_block_idx=1,
        info_step=info,
        task_description="",
        visual_feedback_imgs=[],
        visual_feedback_base64_history=[],
        stderr_history=[],
    )
    assert decision == "finish"
    assert len(prompts) == 1
    assert "code_block_timeout" in str(prompts[0])
    assert "used_actions=407" in str(prompts[0])
    assert "while True:" in str(prompts[0])


def test_watchdog_is_disabled_without_repair_budget(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LLM_TIMEOUT_SECONDS", "1")
    with _trial_alarm(100):
        original_handler = signal.getsignal(signal.SIGALRM)
        with trial._code_block_wall_watchdog(
            {"code_block_wall_timeout_seconds": 0.01},
            "time.sleep(0.03)",
            1,
            can_regenerate=False,
        ):
            time.sleep(0.03)
        assert signal.getsignal(signal.SIGALRM) is original_handler


def test_watchdog_preserves_global_hard_limit_when_time_is_short(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LLM_TIMEOUT_SECONDS", "1")
    def hard_stop(*_args):
        raise TimeoutError("global trial timeout")

    with _trial_alarm(0.03, hard_stop):
        original_handler = signal.getsignal(signal.SIGALRM)
        with pytest.raises(TimeoutError, match="global trial timeout"):
            with trial._code_block_wall_watchdog(
                {"code_block_wall_timeout_seconds": 0.01},
                "pass",
                1,
                can_regenerate=True,
            ):
                time.sleep(0.1)
        assert signal.getsignal(signal.SIGALRM) is original_handler


def test_ctrl_c_restores_trial_clock(monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LLM_TIMEOUT_SECONDS", "1")
    with _trial_alarm(100):
        original_handler = signal.getsignal(signal.SIGALRM)
        with pytest.raises(KeyboardInterrupt):
            with trial._code_block_wall_watchdog(
                {"code_block_wall_timeout_seconds": 0.05},
                "raise KeyboardInterrupt()",
                1,
                can_regenerate=True,
            ):
                raise KeyboardInterrupt()
        remaining, _ = signal.getitimer(signal.ITIMER_REAL)
        assert signal.getsignal(signal.SIGALRM) is original_handler
        assert 90 < remaining < 100


def test_trial_regenerates_after_block_timeout_without_reset(monkeypatch, tmp_path) -> None:
    from capx.envs.agent_progress import read_progress

    monkeypatch.setenv("CAPX_LLM_TIMEOUT_SECONDS", "1")
    calls = {"reset": 0, "steps": 0, "repair": 0, "video_turns": 0, "video_saves": 0}

    class FakeEnv:
        def __init__(self):
            self.progress_path = None
            self.low_level_env = SimpleNamespace(
                get_action_count=lambda: 0,
                get_step_count=lambda: 0,
            )

        def reset(self, **_kwargs):
            calls["reset"] += 1
            return {"full_prompt": [{"role": "user", "content": []}]}, {}

        def set_agent_progress_path(self, path):
            self.progress_path = path

        def enable_video_capture(self, *_args, **_kwargs):
            return None

        def get_video_frame_count(self):
            return calls["video_turns"]

        def begin_video_turn(self, _index):
            return None

        def end_video_turn(self):
            calls["video_turns"] += 1

        def step(self, code):
            calls["steps"] += 1
            if calls["steps"] == 1:
                exec(code, {"AGENT_PROGRESS_PATH": self.progress_path})
            return {}, 1.0, True, False, {
                "sandbox_rc": 0,
                "stdout": "",
                "stderr": "",
                "task_completed": True,
            }

    def repair(*args, **kwargs):
        info = args[8]
        calls["repair"] += 1
        assert "code_block_timeout" in info["stderr"]
        assert read_progress(kwargs["agent_progress_path"])["active_stage_index"] == 1
        return "regenerate", "print('recovered')", None, None, []

    monkeypatch.setattr(trial, "_capture_initial_visual_feedback", lambda *_args: ([], [], "", []))
    agent_record = {
        "schema_version": "capx_agent_progress.v1",
        "status": "active",
        "grasp_confirmed": True,
        "estimator_started": True,
        "completed_stages": [0],
        "active_stage_index": 1,
        "hold_elapsed_s": 0.0,
        "filtered_tension_N": 17.7,
        "last_sample_step": 42,
    }
    first_block = (
        "import json, os\n"
        f"record = {agent_record!r}\n"
        "with open(AGENT_PROGRESS_PATH + '.tmp', 'w') as f:\n"
        "    json.dump(record, f)\n"
        "os.replace(AGENT_PROGRESS_PATH + '.tmp', AGENT_PROGRESS_PATH)\n"
        "while True:\n    pass"
    )
    monkeypatch.setattr(trial, "_query_initial_code", lambda *_args: (first_block, None, None))
    monkeypatch.setattr(trial, "_handle_multi_turn_step", repair)
    monkeypatch.setattr(trial, "_save_trial_artifacts", lambda *_args, **_kwargs: str(tmp_path / "code.py"))
    monkeypatch.setattr(
        trial,
        "_save_turn_and_combined_videos",
        lambda *_args, **_kwargs: calls.__setitem__("video_saves", calls["video_saves"] + 1),
    )
    for name in (
        "_save_trial_video",
        "_save_tactile_artifacts",
        "_save_env_debug_artifacts",
        "_save_tactile_code_memory_trace",
        "_save_visual_checkpoint_artifacts",
    ):
        monkeypatch.setattr(trial, name, lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        visual_differencing_model="test",
        visual_differencing_model_server_url="",
        visual_differencing_model_api_key="",
        max_tokens=64,
        temperature=0,
        reasoning_effort=None,
        debug=False,
    )
    config = {
        "trial_timeout_seconds": 100,
        "code_block_wall_timeout_seconds": 0.05,
        "output_dir": str(tmp_path),
        "record_video": True,
        "use_video_differencing": False,
        "use_wrist_camera": False,
        "use_img_differencing": False,
        "use_visual_feedback": False,
        "use_oracle_code": False,
        "save_in_progress_code": False,
        "multi_turn_on_failure_only": True,
        "max_regenerations": 1,
        "stop_multiturn_when_regeneration_exhausted": True,
        "agent_progress_file": {"enabled": True, "max_context_chars": 1000},
    }
    with _trial_alarm(100):
        summary = trial._run_single_trial(
            FakeEnv(), 1, args, config, "{executed_code} {console_stderr}"
        )

    assert calls == {"reset": 1, "steps": 2, "repair": 1, "video_turns": 2, "video_saves": 1}
    assert summary.task_completed is True
    assert summary.num_regenerations == 1
    assert read_progress(tmp_path / "agent_progress.json")["completed_stages"] == [0]


def test_tension_configs_allow_3000_second_blocks_with_3600_second_trials() -> None:
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in (
        "tension_strap_stage_memory_control.yaml",
        "tension_strap_stage_memory_control_visual.yaml",
    ):
        cfg = yaml.safe_load((root / "env_configs" / "univtac" / name).read_text())
        assert cfg["code_block_wall_timeout_seconds"] == 3000
        assert cfg["trial_timeout_seconds"] == 3600


def test_tension_watchdog_reaches_runtime_config() -> None:
    from pathlib import Path

    from capx.envs.launch_univtac import build_parser
    from capx.utils.launch_utils import _load_config

    root = Path(__file__).resolve().parents[1]
    for name in (
        "tension_strap_stage_memory_control.yaml",
        "tension_strap_stage_memory_control_visual.yaml",
    ):
        path = root / "env_configs" / "univtac" / name
        args = build_parser().parse_args(["--config-path", str(path)])
        _, runtime_config, _ = _load_config(args)
        assert runtime_config["code_block_wall_timeout_seconds"] == 3000
        assert runtime_config["trial_timeout_seconds"] == 3600
        assert runtime_config["agent_progress_file"] == {
            "enabled": True,
            "max_context_chars": 1000,
        }
        assert runtime_config["include_trial_memory_in_multiturn"] is False
