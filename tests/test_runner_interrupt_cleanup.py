from __future__ import annotations

import pytest

from capx.envs import runner


class _InterruptEnv:
    def __init__(self) -> None:
        self.deadline: int | None = None

    def set_trial_deadline(self, seconds: int) -> None:
        self.deadline = seconds

    def clear_trial_deadline(self) -> None:
        self.deadline = None


def test_ctrl_c_flushes_partial_artifacts_before_propagating(monkeypatch) -> None:
    env = _InterruptEnv()
    saved: list[tuple] = []

    def raise_interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt()

    def save_partial(*args, **kwargs):
        saved.append((args, kwargs))

    monkeypatch.setattr(runner, "_run_single_trial", raise_interrupt)
    monkeypatch.setattr(runner, "_build_interrupted_summary", save_partial)

    with pytest.raises(KeyboardInterrupt):
        runner._run_single_trial_with_timeout(
            env=env,
            trial=4,
            args=object(),
            config={},
            multi_turn_prompt=None,
            timeout_s=5,
        )

    assert len(saved) == 1
    assert saved[0][0][0] is env
    assert saved[0][0][1] == 4
    assert env.deadline is None


def test_interrupted_trial_writes_video_and_debug_with_interrupt_status(monkeypatch) -> None:
    calls: list[tuple[str, tuple, dict]] = []

    class _VideoEnv:
        def get_video_frame_count(self) -> int:
            return 5

    def record(name: str):
        def _inner(*args, **kwargs):
            calls.append((name, args, kwargs))
            return "/tmp/code.py" if name == "artifacts" else None

        return _inner

    monkeypatch.setattr(runner, "_save_trial_artifacts", record("artifacts"))
    monkeypatch.setattr(runner, "_save_turn_and_combined_videos", record("video"))
    monkeypatch.setattr(runner, "_save_trial_video", record("single_video"))
    monkeypatch.setattr(runner, "_save_tactile_artifacts", record("tactile"))
    monkeypatch.setattr(runner, "_save_env_debug_artifacts", record("debug"))
    monkeypatch.setattr(runner, "_save_tactile_code_memory_trace", record("memory"))

    summary = runner._build_interrupted_summary(
        _VideoEnv(),
        trial=2,
        pa={
            "raw_code": "print('partial')",
            "code_blocks": ["print('partial')"],
            "code_block_metadata": [{}],
            "info_step": {"sandbox_rc": -1, "stdout": "", "stderr": ""},
            "recording_frames": True,
            "current_frame_start": 1,
            "turn_frame_ranges": [],
            "tactile_code_memory_trace": [],
        },
        config={"output_dir": "/tmp", "record_video": True},
        exc=KeyboardInterrupt(),
    )

    assert summary.sandbox_rc == 130
    assert summary.truncated is True
    assert {name for name, _args, _kwargs in calls} == {
        "artifacts",
        "video",
        "tactile",
        "debug",
        "memory",
    }
    artifact_call = next(call for call in calls if call[0] == "artifacts")
    assert artifact_call[2]["sandbox_rc"] == 130
