from __future__ import annotations

import os
import signal

import numpy as np
import pytest

from capx.envs import runner, trial
from capx.envs.simulators.univtac import UniVTACLowLevelEnv


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
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        os.kill(os.getpid(), signal.SIGINT)
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


def test_reset_rearms_adapter_deadline_with_process_alarm(monkeypatch) -> None:
    env = _InterruptEnv()
    calls: list[int] = []

    def fake_alarm(seconds: int) -> int:
        calls.append(int(seconds))
        return 900 if seconds == 0 else 0

    monkeypatch.setattr(trial.signal, "alarm", fake_alarm)

    timeout = trial._restart_action_deadline_after_reset(
        env, {"trial_timeout_seconds": 900}
    )

    assert timeout == 900
    assert calls == [0, 900]
    assert env.deadline == 898


def test_zero_trial_timeout_does_not_arm_adapter_deadline(monkeypatch) -> None:
    env = _InterruptEnv()
    alarm_calls: list[int] = []
    monkeypatch.setattr(runner.signal, "alarm", lambda seconds: alarm_calls.append(seconds) or 0)
    monkeypatch.setattr(runner, "_run_single_trial", lambda *_args, **_kwargs: "complete")

    result = runner._run_single_trial_with_timeout(
        env=env,
        trial=1,
        args=object(),
        config={"trial_timeout_seconds": 0},
        multi_turn_prompt=None,
        timeout_s=0,
    )

    assert result == "complete"
    assert alarm_calls == [0, 0]
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
    assert calls[0][0] == "video"
    artifact_call = next(call for call in calls if call[0] == "artifacts")
    assert artifact_call[2]["sandbox_rc"] == 130


def test_interrupted_trial_closes_video_even_if_metadata_save_fails(monkeypatch) -> None:
    calls: list[str] = []

    class _VideoEnv:
        def get_video_frame_count(self) -> int:
            return 1

    def save_video(*_args, **_kwargs) -> None:
        calls.append("video")

    def fail_metadata(*_args, **_kwargs) -> None:
        calls.append("metadata")
        raise OSError("metadata disk error")

    monkeypatch.setattr(runner, "_save_turn_and_combined_videos", save_video)
    monkeypatch.setattr(runner, "_save_trial_artifacts", fail_metadata)

    summary = runner._build_interrupted_summary(
        _VideoEnv(),
        trial=1,
        pa={
            "info_step": {"sandbox_rc": -1, "stdout": "", "stderr": ""},
            "recording_frames": True,
            "current_frame_start": 0,
            "turn_frame_ranges": [],
        },
        config={"output_dir": "/tmp", "record_video": True},
        exc=KeyboardInterrupt(),
    )

    assert calls == ["video", "metadata"]
    assert summary.sandbox_rc == 130
    assert summary.code_path is None


def test_ctrl_c_finalizes_partial_streamed_mp4(monkeypatch, tmp_path) -> None:
    stream = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    stream._video_stream_enabled = True
    stream._video_stream_dir = tmp_path / ".video_streams" / "trial_01"
    stream._video_stream_dir.mkdir(parents=True)
    stream._video_stream_combined_writer = None
    stream._video_stream_turn_writer = None
    stream._video_stream_turn_index = None
    stream._video_stream_frame_count = 0
    stream._video_stream_h264 = True
    stream.video_stream_fps = 15.0

    class _StreamEnv(_InterruptEnv):
        def get_video_frame_count(self) -> int:
            return stream.get_video_frame_count()

        def get_video_frames(self, *, clear: bool = False) -> list:
            return []

        def finalize_streamed_video(self, output_dir: str) -> bool:
            return stream.finalize_streamed_video(output_dir)

    env = _StreamEnv()

    def interrupt_after_frame(_env, _trial, _args, _config, _prompt, *, partial_artifacts):
        partial_artifacts.update({
            "info_step": {"sandbox_rc": -1, "stdout": "", "stderr": ""},
            "recording_frames": True,
            "current_frame_start": 0,
            "turn_frame_ranges": [],
        })
        stream._write_streamed_video_frame(np.full((64, 64, 3), 73, dtype=np.uint8))
        raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "_run_single_trial", interrupt_after_frame)
    monkeypatch.setattr(runner, "_save_trial_artifacts", lambda *_args, **_kwargs: str(tmp_path / "code.py"))

    with pytest.raises(KeyboardInterrupt):
        runner._run_single_trial_with_timeout(
            env=env,
            trial=1,
            args=object(),
            config={"output_dir": str(tmp_path), "record_video": True},
            multi_turn_prompt=None,
            timeout_s=0,
        )

    video = tmp_path / "trial_01_sandboxrc_130_reward_0.000_taskcompleted_0" / "videos" / "video_combined.mp4"
    assert video.is_file()
    import cv2

    reader = cv2.VideoCapture(str(video))
    try:
        ok, _frame = reader.read()
        assert ok
    finally:
        reader.release()
