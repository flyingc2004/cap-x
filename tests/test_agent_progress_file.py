"""Isaac-free checks for agent-owned progress across generated code blocks."""

from __future__ import annotations

import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from capx.envs.agent_progress import (
    SCHEMA_VERSION,
    initialize_progress,
    progress_context,
    progress_path,
    read_progress,
    save_progress_artifact,
)
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.envs.tasks.exceptions import CodeBlockTimeout
from capx.envs import trial
from capx.envs.trial import _code_block_wall_watchdog, _handle_multi_turn_step


def _record(**overrides) -> dict:
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "active",
        "grasp_confirmed": True,
        "estimator_started": True,
        "completed_stages": [0],
        "active_stage_index": 1,
        "hold_elapsed_s": 1.5,
        "filtered_tension_N": 17.9,
        "last_sample_step": 700,
    }
    return {**record, **overrides}


def _agent_atomic_write(path: Path, record: dict) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)


def test_separate_trials_and_retried_attempt_reset(tmp_path: Path) -> None:
    config = {"output_dir": str(tmp_path), "agent_progress_file": {"enabled": True}}
    first = progress_path(config, 1)
    second = progress_path(config, 2)
    assert first is not None and second is not None and first != second
    initialize_progress(first)
    initialize_progress(second)
    _agent_atomic_write(first, _record())
    assert read_progress(first)["completed_stages"] == [0]
    assert read_progress(second)["status"] == "uninitialized"

    # Runner initializes again before a retry of the same trial.
    initialize_progress(first)
    assert read_progress(first)["status"] == "uninitialized"


def test_interrupted_write_keeps_last_committed_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "agent_progress.json"
    initialize_progress(path)
    _agent_atomic_write(path, _record())
    Path(str(path) + ".tmp").write_text('{"incomplete":', encoding="utf-8")
    assert read_progress(path)["active_stage_index"] == 1
    assert read_progress(path)["hold_elapsed_s"] == 1.5


def test_progress_context_and_artifact_only_include_public_fields(tmp_path: Path) -> None:
    path = tmp_path / "agent_progress.json"
    initialize_progress(path)
    _agent_atomic_write(path, _record(true_tension_N=18.3, private_label="secret"))
    context = progress_context(path, max_chars=1000)
    assert "completed_stages" in context
    assert len(context) <= 1000
    assert "true_tension" not in context
    assert "secret" not in context

    final_dir = tmp_path / "trial_01_sandboxrc_1_reward_0.000_taskcompleted_0"
    final_dir.mkdir()
    save_progress_artifact(path, str(final_dir / "code.py"))
    saved = json.loads((final_dir / "agent_progress.json").read_text())
    assert saved["active_stage_index"] == 1
    assert "private_label" not in saved
    assert "true_tension_N" not in saved


def test_missing_and_corrupt_file_cannot_claim_stage_completion(tmp_path: Path) -> None:
    path = tmp_path / "agent_progress.json"
    assert read_progress(path)["status"] == "missing"
    path.write_text('{"incomplete":', encoding="utf-8")
    assert read_progress(path)["status"] == "invalid"
    assert "completed_stages" not in progress_context(path)
    path.write_text(json.dumps(_record(hold_elapsed_s=float("nan"))), encoding="utf-8")
    assert read_progress(path)["status"] == "invalid"


def test_block_timeout_preserves_file_for_next_generated_block(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CAPX_LLM_TIMEOUT_SECONDS", "1")
    path = tmp_path / "agent_progress.json"
    initialize_progress(path)
    executor = SimpleNamespace(_exec_globals={})
    CodeExecutionEnvBase.set_agent_progress_path(executor, str(path))
    code = (
        "import json, os\n"
        "record = " + repr(_record()) + "\n"
        "with open(AGENT_PROGRESS_PATH + '.tmp', 'w') as f:\n"
        "    json.dump(record, f)\n"
        "os.replace(AGENT_PROGRESS_PATH + '.tmp', AGENT_PROGRESS_PATH)\n"
        "while True:\n"
        "    pass\n"
    )
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, lambda *_args: None)
    signal.setitimer(signal.ITIMER_REAL, 100)
    try:
        with pytest.raises(CodeBlockTimeout):
            with _code_block_wall_watchdog(
                {"code_block_wall_timeout_seconds": 0.05}, code, 1, can_regenerate=True
            ):
                exec(code, executor._exec_globals)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        signal.setitimer(signal.ITIMER_REAL, *old_timer)

    next_globals = {"AGENT_PROGRESS_PATH": str(path)}
    exec(
        "import json\nwith open(AGENT_PROGRESS_PATH) as f:\n"
        "    progress = json.load(f)\ncontinued_stage = progress['active_stage_index']",
        next_globals,
    )
    assert next_globals["continued_stage"] == 1
    assert next_globals["progress"]["hold_elapsed_s"] == 1.5


def test_multiturn_prompt_includes_bounded_file_snapshot(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "agent_progress.json"
    initialize_progress(path)
    _agent_atomic_write(path, _record(true_tension_N=19.0))
    prompts = []

    def fake_query(_args, prompt):
        prompts.append(prompt)
        return {"content": "FINISH", "reasoning": None}

    monkeypatch.setattr("capx.envs.trial._query_model", fake_query)
    config = {
        "use_visual_feedback": False,
        "use_img_differencing": False,
        "use_video_differencing": False,
        "use_parallel_ensemble": False,
        "filter_multiturn_console": True,
        "multiturn_console_max_lines": 10,
        "multiturn_console_max_chars": 1000,
        "agent_progress_file": {"enabled": True, "max_context_chars": 1000},
    }
    decision, *_ = _handle_multi_turn_step(
        env=SimpleNamespace(),
        obs={"full_prompt": [{"role": "user", "content": [{"type": "text", "text": "task"}]}]},
        args=SimpleNamespace(model="test", use_legacy_multi_turn_decision_prompt=False),
        config=config,
        visual_differencing_args=SimpleNamespace(model="test"),
        multi_turn_prompt="{executed_code}\n{console_stdout}\n{console_stderr}",
        code_blocks=["while True: pass"],
        code_block_idx=1,
        info_step={"sandbox_rc": 1, "stdout": "", "stderr": "code_block_timeout"},
        task_description="",
        visual_feedback_imgs=[],
        visual_feedback_base64_history=[],
        stderr_history=[],
        agent_progress_path=path,
    )
    assert decision == "finish"
    prompt = str(prompts[0])
    assert "active_stage_index" in prompt
    assert "code_block_timeout" in prompt
    assert "true_tension_N" not in prompt


def test_runner_fails_closed_when_actions_happen_without_valid_file(tmp_path: Path, monkeypatch) -> None:
    counts = {"steps": 0, "repairs": 0}

    class FakeEnv:
        def __init__(self):
            self.path = None
            self.low_level_env = SimpleNamespace(
                get_action_count=lambda: 0,
                get_step_count=lambda: counts["steps"],
            )

        def reset(self, **_kwargs):
            counts["steps"] = 0
            return {"full_prompt": [{"role": "user", "content": []}]}, {}

        def set_agent_progress_path(self, path):
            self.path = Path(path)

        def step(self, _code):
            counts["steps"] += 1
            self.path.write_text('{"incomplete":', encoding="utf-8")
            return {}, 0.0, False, False, {
                "sandbox_rc": 0,
                "stdout": "",
                "stderr": "",
                "task_completed": False,
            }

    def unexpected_repair(*_args, **_kwargs):
        counts["repairs"] += 1
        raise AssertionError("corrupt progress must not be guessed by a new block")

    monkeypatch.setattr(trial, "_capture_initial_visual_feedback", lambda *_args: ([], [], "", []))
    monkeypatch.setattr(trial, "_query_initial_code", lambda *_args: ("print('start')", None, None))
    monkeypatch.setattr(trial, "_handle_multi_turn_step", unexpected_repair)
    monkeypatch.setattr(trial, "_save_trial_artifacts", lambda *_args, **_kwargs: str(tmp_path / "code.py"))
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
        "output_dir": str(tmp_path),
        "record_video": False,
        "use_video_differencing": False,
        "use_wrist_camera": False,
        "use_img_differencing": False,
        "use_visual_feedback": False,
        "use_oracle_code": False,
        "save_in_progress_code": False,
        "multi_turn_on_failure_only": True,
        "max_regenerations": 1,
        "agent_progress_file": {"enabled": True, "max_context_chars": 1000},
    }
    summary = trial._run_single_trial(FakeEnv(), 1, args, config, "{console_stderr}")
    assert summary.success is False
    assert counts == {"steps": 1, "repairs": 0}
    assert read_progress(tmp_path / "agent_progress.json")["status"] == "invalid"
