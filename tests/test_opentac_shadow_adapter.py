"""GPU-free checks for the Shadow piano bridge's public/private boundary."""

import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import yaml

from capx.envs.simulators.opentac_shadow import ShadowPianoLowLevelEnv
from capx.envs import trial as trial_module
from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
from capx.integrations.base_api import register_api
from capx.integrations.opentac_shadow import ShadowPianoControlApi


CONFIG = Path(__file__).resolve().parents[1] / "env_configs/opentac/play_piano_shadow_digit_smoke.yaml"


class FakeShadowTask:
    def __init__(self, *, stop_after=3, verdict=True):
        self.action_space = gym.spaces.Box(
            np.array([-.01, -.32, -.006, 0.], np.float32),
            np.array([.09, .26, .05, .5], np.float32),
        )
        self.stop_after = stop_after
        self.verdict = verdict
        self.steps = 0
        self.actions = []
        self.closed = False

    def _observation(self):
        state = np.zeros(111, dtype=np.float32)
        state[0:3] = [-.09175, .03, .12]
        state[6 + 60 - 21] = .027
        return {"state": state, "tactile": {"index": {"rgb": np.full((4, 6, 3), 19, np.uint8)}}}

    def _action_state(self):
        return np.array([0., -.09175, .03, .12], np.float32)

    def reset(self, *, seed=None):
        self.seed = seed
        self.steps = 0
        return self._observation(), {"success": False, "actual_force_N": 1.23}

    def step(self, action):
        self.steps += 1
        self.actions.append(np.asarray(action).copy())
        done = self.steps >= self.stop_after
        return self._observation(), 100.0, done, False, {
            "success": bool(done and self.verdict), "failure": "" if self.verdict else "wrong_key",
            "actual_force_N": 1.23,
        }

    def render(self):
        return np.zeros((6, 8, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


def make_adapter(tmp_path, fake):
    return ShadowPianoLowLevelEnv(
        univtac_root="/mnt/a/ljz/cap/OpenTacBench",
        piano_notes=["C#4", "C4", "G4", "A4", "E4", "F4"],
        piano_dynamics=["light", "heavy"] * 3,
        piano_require_dynamics=False,
        task_factory=lambda _: fake,
        output_dir=str(tmp_path),
    )


def test_shadow_smoke_requests_six_notes_and_only_local_dynamics_override():
    config = yaml.safe_load(CONFIG.read_text())
    low = config["env"]["cfg"]["low_level"]
    assert low["piano_notes"] == ["C#4", "C4", "G4", "A4", "E4", "F4"]
    assert low["piano_dynamics"] == ["light", "heavy"] * 3
    assert low["piano_require_dynamics"] is False
    assert "piano_hold_seconds" not in low  # native default: 0.12 s
    assert config["trials"] == config["num_workers"] == 1
    assert config["env"]["cfg"]["apis"] == ["ShadowPianoControlApi"]
    assert "Save the value returned by get_key_angles" in config["env"]["cfg"]["prompt"]


def test_bounded_actions_rgb_private_score_and_end(tmp_path):
    fake = FakeShadowTask(stop_after=3)
    env = make_adapter(tmp_path, fake)
    assert env.video_fps == 10.0
    api = ShadowPianoControlApi(env)
    public, info = env.reset(seed=5)
    assert fake.seed == 5 and info == {}
    assert "success" not in repr(public) and "force" not in repr(public)
    assert public["hand_state"]["wrist_x_offset_m"] == 0.0
    assert api.get_key_angles(["C4"])["C4"] == pytest.approx(.027)
    assert api.get_note_target("C#4")["black"] is True
    assert api.get_note_target("C#4")["wrist_x_offset_m"] == .07
    assert api.get_index_tactile_rgb().shape == (4, 6, 3)
    assert (tmp_path / "shadow_digit_initial.png").is_file()
    with pytest.raises(ValueError, match="outside native bounds"):
        api.move_shadow_index(.5, 0, 0, 0)
    assert fake.steps == 0
    env.enable_video_capture()
    result = api.move_shadow_index(0, -.09, .03, .2, steps=2)
    assert result == {"ok": True, "steps_executed": 2, "native_done": False}
    assert api.wait_steps(4) == {"ok": True, "steps_executed": 1, "native_done": True}
    assert len(fake.actions) == 3
    assert len(env.get_video_frames()) == 2  # initial + stride-three frame
    assert env.get_video_frames()[0].shape == (16, 16, 3)
    with pytest.raises(RuntimeError, match="ended"):
        api.wait_steps(1)
    assert env.submit_policy_end() is True
    assert env.task_completed() and env.compute_reward() == 1.0
    with pytest.raises(RuntimeError, match="already"):
        env.submit_policy_end()
    events = [json.loads(row) for row in (tmp_path / "shadow_native_events.jsonl").read_text().splitlines()]
    assert events[-1]["kind"] == "END"
    assert events[1]["action_target"] == pytest.approx([0, -.09, .03, .2])
    assert events[-1]["native_info"]["actual_force_N"] == 1.23
    assert (tmp_path / "shadow_digit_terminal.png").is_file()
    assert (tmp_path / "shadow_digit_end.png").is_file()
    env.close()
    assert fake.closed


def test_failed_native_verdict_is_returned_only_at_end(tmp_path):
    env = make_adapter(tmp_path, FakeShadowTask(stop_after=1, verdict=False))
    env.reset()
    assert env.command([0, -.09, .03, 0])["native_done"]
    assert not env.task_completed()
    assert env.submit_policy_end() is False
    assert not env.task_completed()


def test_generated_code_end_uses_saved_native_verdict(tmp_path):
    register_api("ShadowPianoControlApi", ShadowPianoControlApi)

    class ClosedShadowCodeEnv(CodeExecutionEnvBase):
        def _exec_env_binding(self):
            return None

        def _exec_apis_binding(self):
            return {}

    low = make_adapter(tmp_path, FakeShadowTask(stop_after=1))
    env = ClosedShadowCodeEnv(CodeExecEnvConfig(
        low_level=low, apis=["ShadowPianoControlApi"], prompt="Play one fake note.",
        stream_user_code_output=False,
    ))
    env.reset()
    _, reward, terminated, _, info = env.step(
        "capx_stop = False\n"
        "result = move_shadow_index(0, -.09, .03, 0)\n"
        "assert result['native_done']\n"
        "capx_stop = True\n"
    )
    assert reward == 1.0 and terminated
    assert info["task_completed"] is True
    assert "force" not in info["stdout"]
    assert (tmp_path / "shadow_native_events.jsonl").read_text().count('"kind": "END"') == 1


def test_shadow_video_writer_receives_decimated_frame_rate(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(trial_module, "_write_video", lambda frames, path, *, suffix, fps: calls.append((suffix, len(frames), fps)))

    class RecordedEnv:
        low_level_env = SimpleNamespace(video_fps=10.0)

        def get_video_frames(self, *, clear=False):
            return [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(7)]

    trial_module._save_turn_and_combined_videos(
        RecordedEnv(), {"record_video": True, "output_dir": str(tmp_path)},
        1, {"sandbox_rc": 0, "task_completed": False}, 0.0, [(1, 7)],
    )
    assert calls == [("turn_00", 6, 10.0), ("combined", 7, 10.0)]
