"""CPU contracts for visual key selection and public DIGIT force inference."""

import hashlib
import json
import os
import stat

import numpy as np
import pytest
import yaml
from PIL import Image

from capx.integrations.opentac_shadow import ShadowPianoVisualForceApi
from capx.integrations.opentac_shadow_audit import save_native_piano_summary
from capx.envs.simulators.opentac_shadow import ShadowPianoLowLevelEnv
from capx.integrations.opentac_shadow_force import FEATURE_NAMES, SCHEMA, PublicDigitKeyForce, public_depth_features
from capx.integrations.opentac_shadow_vision import (
    HEAD_EYE, HEAD_FOCAL_PX, HEAD_HEIGHT, HEAD_LOOK_AT, HEAD_WIDTH,
    head_pixel_to_world, locate_key_pixels, snap_key_pixels_to_visible_centers,
)


def _project(x, y, z):
    forward = HEAD_LOOK_AT - HEAD_EYE
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    ray = np.array([x, y, z]) - HEAD_EYE
    return {"u": .5 + (ray @ right) / (ray @ forward) * HEAD_FOCAL_PX / HEAD_WIDTH,
            "v": .5 - (ray @ up) / (ray @ forward) * HEAD_FOCAL_PX / HEAD_HEIGHT}


def test_overhead_head_fits_both_keyboard_ends():
    assert HEAD_EYE[2] > 1.0
    left_end = _project(.515, -25.5 * .0235, .0725)
    right_end = _project(.515, 25.5 * .0235, .0725)
    # The displayed HEAD is mirrored: A0 must appear left of C8, with room
    # for the piano cheeks at both edges.
    assert .05 < 1 - left_end["u"] < .25
    assert .75 < 1 - right_end["u"] < .95
    assert .1 < left_end["v"] < .9
    assert .1 < right_end["v"] < .9


def test_public_endpoints_match_static_keyboard_and_mirrored_head(monkeypatch):
    from pathlib import Path

    opentac_root = Path(__file__).resolve().parents[3] / "OpenTacBench"
    monkeypatch.syspath_prepend(str(opentac_root))
    env = ShadowPianoLowLevelEnv.__new__(ShadowPianoLowLevelEnv)
    env.visual_key_mode = True
    env.mirror_head_view = True
    endpoints = ShadowPianoVisualForceApi(env).get_keyboard_endpoints()
    assert endpoints["white_key_count"] == 52
    assert endpoints["white_center_intervals"] == 51
    assert endpoints["white_center_spacing_m"] == pytest.approx(.0235)
    assert endpoints["A0"]["display_image_center_px"][0] == pytest.approx(138.35, abs=.1)
    assert endpoints["C8"]["display_image_center_px"][0] == pytest.approx(821.65, abs=.1)
    assert endpoints["A0"]["world_key_top_center_m"][1] == pytest.approx(-.59925)
    assert endpoints["C8"]["world_key_top_center_m"][1] == pytest.approx(.59925)


def test_full_visual_key_map_and_bounds():
    choices = {
        "C#4": {**_project(.52, -.09, .085), "color": "black"},
        "D#4": {**_project(.52, -.043, .085), "color": "black"},
        "E4": {**_project(.47, -.020, .0725), "color": "white"},
        "F4": {**_project(.47, .0035, .0725), "color": "white"},
    }
    targets = locate_key_pixels(choices, expected_notes=["E4", "F4", "C#4", "D#4", "D#4"])
    assert set(targets) == set(choices)
    assert targets["C#4"]["wrist_y_m"] == pytest.approx(-.123)
    assert targets["F4"]["wrist_x_offset_m"] == 0
    assert targets["D#4"]["wrist_x_offset_m"] == .07
    mirrored = {note: {**pixel, "u": 1.0 - pixel["u"]} for note, pixel in choices.items()}
    mirrored_targets = locate_key_pixels(
        mirrored, expected_notes=["E4", "F4", "C#4", "D#4", "D#4"], mirrored=True,
    )
    for note in choices:
        np.testing.assert_allclose(
            mirrored_targets[note]["projected_world_xyz_m"],
            targets[note]["projected_world_xyz_m"], atol=1e-9,
        )
        assert mirrored_targets[note]["head_pixel"] == [mirrored[note]["u"], mirrored[note]["v"]]
    crossed = {
        "E4": {"u": 525/960, "v": (360-22)/640, "color": "white"},
        "F4": {"u": 538/960, "v": (360-22)/640, "color": "white"},
        "C#4": {"u": 518/960, "v": (337-22)/640, "color": "black"},
        "D#4": {"u": 531/960, "v": (337-22)/640, "color": "black"},
    }
    with pytest.raises(ValueError, match=r"D#4.*E4"):
        locate_key_pixels(crossed, expected_notes=list(crossed), mirrored=True)
    with pytest.raises(ValueError):
        locate_key_pixels(mirrored, expected_notes=list(choices))
    with pytest.raises(ValueError, match="every distinct"):
        locate_key_pixels({"E4": choices["E4"]}, expected_notes=list(choices))
    bad = dict(choices)
    bad["E4"] = {**choices["E4"], "color": "black"}
    with pytest.raises(ValueError, match="color"):
        locate_key_pixels(bad, expected_notes=list(choices))
    np.testing.assert_allclose(head_pixel_to_world(**_project(.47, -.020, .0725), black=False),
                               [.47, -.020, .0725], atol=1e-9)


def test_visible_key_snap_stays_within_selected_key():
    image = np.full((640, 960, 3), 190, np.uint8)
    for seam in (466, 480, 493):
        image[350:358, seam-1:seam+2] = 120
    image[305:312, 450:456] = 10
    image[305:312, 464:469] = 10
    selected = {
        "E4": {"u": 469/960, "v": 353/640, "color": "white"},
        "F4": {"u": 483/960, "v": 353/640, "color": "white"},
        "C#4": {"u": 449/960, "v": 308/640, "color": "black"},
        "D#4": {"u": 462/960, "v": 308/640, "color": "black"},
    }
    snapped, audit = snap_key_pixels_to_visible_centers(selected, image)
    assert [snapped[note]["u"] * 960 for note in selected] == pytest.approx([473., 486.5, 452.5, 466.])
    assert all(0 < row["delta_x_px"] <= 4.5 for row in audit.values())
    with pytest.raises(ValueError, match="visible seams|too far"):
        snap_key_pixels_to_visible_centers(
            {"E4": {"u": 468/960, "v": 353/640, "color": "white"}},
            np.roll(image, 8, axis=1),
        )


def test_shadow_head_and_video_mirror(monkeypatch):
    from types import SimpleNamespace

    source = np.zeros((16, 16, 3), dtype=np.uint8)
    source[:, :, 0] = np.arange(16, dtype=np.uint8)
    task = SimpleNamespace(sim=SimpleNamespace(render=lambda: None), render=lambda: source)
    env = object.__new__(ShadowPianoLowLevelEnv)
    env._task = task
    env.visual_key_mode = True
    env.mirror_head_view = True
    env.video_width = 16
    env._frame_buffer = []
    monkeypatch.setattr(env, "_configure_camera", lambda: None)
    np.testing.assert_array_equal(env.render_head(), source[:, ::-1])
    env._capture_frame()
    np.testing.assert_array_equal(env._frame_buffer[0], source[:, ::-1])


def test_force_estimate_quality_and_no_hidden_api(tmp_path):
    config = tmp_path / "digit.yml"
    config.write_text("range_force: [0, 2]\n")
    group = {"median": [0, 0, 0, 0], "iqr": [1, 1, 1, 1],
             "coef": [1, 0, 0, 0, 0, 0, 0, 0], "intercept": 0,
             "feature_min": [0, 0, 0, 0], "feature_max": [10, 1, 1, 1]}
    sidecar = tmp_path / "model.json"
    sidecar.write_text(json.dumps({"schema": SCHEMA, "feature_names": list(FEATURE_NAMES),
                                   "sensor_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                                   "models": {"white": group, "black": group}}))
    estimator = PublicDigitKeyForce(sidecar, config)
    reference = np.ones((20, 20), np.float32)
    depth = reference - .0002
    features, quality = public_depth_features(depth, reference)
    assert quality == "ok" and features[0] == pytest.approx(.2, abs=1e-4)
    reading = estimator.predict(depth, reference, color="white", sample_step=4, current_step=4)
    assert reading["quality"] == "ok" and reading["estimated_key_force_N"] == pytest.approx(.2, abs=1e-4)
    assert estimator.predict(depth, reference, color="white", sample_step=3, current_step=4)["quality"] == "invalid_color_or_stale"
    assert estimator.predict(np.zeros((20, 20)), reference, color="white", sample_step=4, current_step=4)["quality"] == "invalid_depth"
    assert "get_note_target" not in ShadowPianoVisualForceApi(None).functions()
    assert "get_key_angles" not in ShadowPianoVisualForceApi(None).functions()


def test_smoke_uses_current_head_until_map_is_latched_and_strict_score():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "env_configs/opentac/play_piano_shadow_digit_visual_force_smoke.yaml"
    cfg = yaml.safe_load(path.read_text())
    low = cfg["env"]["cfg"]["low_level"]
    assert low["piano_require_dynamics"] is True
    assert low["piano_continue_on_note_failure"] is True
    assert low["mirror_head_view"] is True
    assert low["auto_clear_keyboard_for_visual"] is True
    assert low["snap_visual_key_pixels"] is True
    assert low["piano_notes"] == ["E4", "F4", "C#4", "D#4", "D#4", "E4", "C#4", "F4"]
    assert cfg["visual_feedback_raw_camera_views"] and not cfg["use_wrist_camera"]
    assert cfg["visual_feedback_max_image_side_px"] >= 960
    assert cfg["visual_feedback_context"] == "shadow_piano"
    assert cfg["visual_checkpoint_markers"] == ["finger_alignment"]
    assert cfg["shadow_piano_recover_on_stop"]
    assert cfg["shadow_piano_wrist_on_failure"]
    assert "Generate ONE executable Python program" in cfg["env"]["cfg"]["prompt"]
    assert "finish_piano_note_attempt(index)" in cfg["env"]["cfg"]["prompt"]
    assert "after eight finalized attempts" in cfg["env"]["cfg"]["multi_turn_prompt"]
    assert "get_keyboard_endpoints()" in cfg["env"]["cfg"]["prompt"]
    assert cfg["max_regenerations"] == 1
    assert cfg["trials"] == cfg["num_workers"] == 1
    assert cfg["env"]["cfg"]["apis"] == ["ShadowPianoVisualForceApi"]
    assert "finish_piano_note_attempt" in ShadowPianoVisualForceApi(None).functions()

    from capx.envs.launch import LaunchArgs
    from capx.utils.launch_utils import _load_config
    factory, _, _ = _load_config(LaunchArgs(config_path=str(path)))
    loaded_prompt = factory["cfg"]["prompt"]
    assert loaded_prompt.count("Public Shadow piano skill library.") == 1
    assert "delayed_force_response" in loaded_prompt
    assert "Near-full flexion and two contact samples alone do not establish stable contact" in loaded_prompt
    assert "native_failure" not in loaded_prompt
    assert "1.3704" not in loaded_prompt
    assert "1.5561" not in loaded_prompt

    from capx.utils.launch_utils import _build_multi_turn_decision_prompt
    inherited = _build_multi_turn_decision_prompt(
        {"full_prompt": [{"role": "user", "content": [{"type": "text", "text": loaded_prompt}]}]},
        "Recover from the current state", None,
    )
    assert inherited[0]["content"][0]["text"].count("Public Shadow piano skill library.") == 1

    from capx.envs.trial import _shadow_single_image_decision_obs, _should_query_multiturn_after_block
    from capx.utils.launch_utils import _build_multi_turn_decision_prompt
    previous = {"full_prompt": [
        {"role": "system", "content": "Generate code"},
        {"role": "user", "content": [{"type": "text", "text": "Piano"},
                                      {"type": "image_url", "image_url": {"url": "old"}}]},
    ]}
    cleaned = _shadow_single_image_decision_obs(previous)
    decision = _build_multi_turn_decision_prompt(cleaned, "Check alignment", "new")
    assert sum(part.get("type") == "image_url" for part in decision[-1]["content"] if isinstance(part, dict)) == 1
    mapped_decision = _build_multi_turn_decision_prompt(cleaned, "Align finger using accepted key map", "current")
    assert sum(part.get("type") == "image_url" for part in mapped_decision[-1]["content"] if isinstance(part, dict)) == 1
    assert previous["full_prompt"][-1]["content"][-1]["image_url"]["url"] == "old"
    terminal = {"sandbox_rc": 0, "stdout": "", "stderr": "", "task_completed": False}
    assert not _should_query_multiturn_after_block(
        terminal, code_block_idx=2, total_code_blocks=2, config=cfg,
    )
    assert not _should_query_multiturn_after_block(
        terminal, code_block_idx=2, total_code_blocks=2,
        config={**cfg, "visual_feedback_context": "other"},
    )
    stopped_probe = {**terminal, "stdout": "PIANO_STOP {'completed': 0, 'reason': 'force band'}"}
    assert _should_query_multiturn_after_block(
        stopped_probe, code_block_idx=3, total_code_blocks=3, config=cfg,
    )
    assert not _should_query_multiturn_after_block(
        stopped_probe, code_block_idx=3, total_code_blocks=3,
        config={**cfg, "shadow_piano_recover_on_stop": False},
    )

    _, runtime_cfg, _ = _load_config(LaunchArgs(config_path=str(path)))
    assert runtime_cfg["shadow_piano_recover_on_stop"]
    assert runtime_cfg["shadow_piano_wrist_on_failure"]
    assert _should_query_multiturn_after_block(
        stopped_probe, code_block_idx=1, total_code_blocks=1, config=runtime_cfg,
    )


def test_shadow_public_skill_loader_rejects_private_audit(tmp_path):
    from capx.integrations.opentac_shadow_skills import inject_shadow_piano_public_skills

    factory = {"cfg": {
        "low_level": {"_target_": "capx.envs.simulators.opentac_shadow.ShadowPianoLowLevelEnv"},
        "apis": ["ShadowPianoVisualForceApi"], "prompt": "Play the sequence.",
    }}
    bad = tmp_path / "private.yaml"
    bad.write_text(yaml.safe_dump({
        "schema": "shadow_piano_public_skills.v1",
        "skills": [{"id": "unsafe", "trigger": "After contact.",
                    "practice": "Read native_failure from the audit."}],
    }))
    with pytest.raises(ValueError, match="private audit markers"):
        inject_shadow_piano_public_skills(
            factory, config_path=tmp_path / "smoke.yaml", library_path=bad,
        )
    assert factory["cfg"]["prompt"] == "Play the sequence."


def test_shadow_checkpoint_uses_only_current_head_camera():
    from capx.envs.trial import _capture_raw_visual_snapshot, _compose_checkpoint_image

    class CameraEnv:
        def render_head(self):
            return np.full((640, 960, 3), (0, 128, 0), dtype=np.uint8)

        def render_wrist(self):
            raise AssertionError("Shadow smoke must not request a wrist image")

    previous = {"head": Image.new("RGB", (960, 640), "red")}
    current = _capture_raw_visual_snapshot(CameraEnv(), {"use_wrist_camera": False})
    assert set(current) == {"head"}
    shadow_image = _compose_checkpoint_image(previous, current, "shadow_piano")
    assert shadow_image.size == (960, 662)
    assert shadow_image.getpixel((100, 122)) == (0, 128, 0)
    assert _compose_checkpoint_image(previous, current, "other").size == (960, 1324)


def test_shadow_recovery_can_capture_current_wrist_without_changing_initial_view():
    from capx.envs.trial import _capture_raw_visual_snapshot, _labeled_visual_tile

    class CameraEnv:
        def render_head(self):
            return np.full((640, 960, 3), (0, 128, 0), dtype=np.uint8)

        def render_wrist(self):
            return np.full((640, 960, 3), (20, 40, 180), dtype=np.uint8)

    config = {"use_wrist_camera": False}
    assert set(_capture_raw_visual_snapshot(CameraEnv(), config)) == {"head"}
    recovered = _capture_raw_visual_snapshot(CameraEnv(), config, include_wrist=True)
    assert set(recovered) == {"head", "wrist"}
    wrist_image = _labeled_visual_tile(recovered["wrist"], "Current Wrist RGB")
    assert wrist_image.size == (960, 662)
    assert wrist_image.getpixel((100, 122)) == (20, 40, 180)


def test_initial_shadow_image_prompt_includes_public_endpoints():
    from types import SimpleNamespace
    from capx.envs.trial import _capture_initial_visual_feedback

    endpoints = {"A0": {"display_image_center_px": [138.35, 347.4]},
                 "C8": {"display_image_center_px": [821.65, 347.4]}}
    env = SimpleNamespace(
        low_level_env=SimpleNamespace(get_keyboard_endpoints=lambda: endpoints),
        render_head=lambda: np.full((640, 960, 3), 180, dtype=np.uint8),
    )
    obs = {"full_prompt": [{"role": "user", "content": [{"type": "text", "text": "Piano"}]}]}
    config = {"use_visual_feedback": True, "visual_feedback_allow_unlisted_model": True,
              "visual_feedback_raw_camera_views": True,
              "visual_feedback_context": "shadow_piano",
              "visual_feedback_max_image_side_px": 960,
              "visual_feedback_jpeg_quality": 92}
    _capture_initial_visual_feedback(env, obs, config, SimpleNamespace(model="gpt-6-sol"), None)
    content = obs["full_prompt"][-1]["content"]
    assert "138.35" in content[0]["text"] and "821.65" in content[0]["text"]
    assert sum(part.get("type") == "image_url" for part in content) == 1


def test_visual_keyboard_clear_uses_bounded_native_actions():
    env = ShadowPianoLowLevelEnv.__new__(ShadowPianoLowLevelEnv)
    current_y = [-.09175]
    actions = []
    env._sim_step_count = 0
    env.get_observation = lambda: {"hand_state": {"wrist_y_m": current_y[0]}}

    def command(action, steps):
        actions.append((list(action), steps))
        env._sim_step_count += steps
        current_y[0] = [-.09175, .10, .225][min(len(actions) - 1, 2)]
        return {"native_done": False}

    env.command = command
    env._clear_keyboard_for_visual()
    assert actions == [
        ([0., -.09175, .05, 0.], 30),
        ([0., .24, .05, 0.], 40),
        ([0., .24, .05, 0.], 40),
    ]


def test_visual_press_waits_for_measured_xy_arrival():
    """A commanded key target cannot authorize a press while the arm crosses it."""
    env = ShadowPianoLowLevelEnv.__new__(ShadowPianoLowLevelEnv)
    env.visual_key_mode = True
    env._visual_key_map_ready = True
    hand = {"wrist_x_offset_m": 0., "wrist_y_m": .146,
            "wrist_and_index_velocity": [-.149, 0., 0.]}
    env.get_observation = lambda: {"hand_state": hand}
    e4 = np.array([0., -.04555, 0., .5], dtype=np.float32)

    # Moving at hover stays possible; the same XY target cannot lower yet.
    env._require_xy_arrival_before_press(np.array([0., -.04555, .04, 0.]))
    with pytest.raises(ValueError, match="has not settled"):
        env._require_xy_arrival_before_press(e4)
    hand["wrist_y_m"] = -.04555
    with pytest.raises(ValueError, match="has not settled"):
        env._require_xy_arrival_before_press(e4)
    hand["wrist_and_index_velocity"][0] = 0.
    env._require_xy_arrival_before_press(e4)


def test_visual_key_map_flag_latches_only_after_private_save(tmp_path):
    from types import SimpleNamespace

    class Intrinsics:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.array([[HEAD_FOCAL_PX, 0, HEAD_WIDTH / 2],
                             [0, HEAD_FOCAL_PX, HEAD_HEIGHT / 2], [0, 0, 1]])

    env = ShadowPianoLowLevelEnv.__new__(ShadowPianoLowLevelEnv)
    env.visual_key_mode = True
    env.mirror_head_view = True
    env.auto_clear_keyboard_for_visual = False
    env.snap_visual_key_pixels = False
    env.piano_notes = ["E4"]
    env._sim_step_count = 12
    env._visual_key_map_ready = False
    env._output_dir = tmp_path / "public"
    env.private_audit_root = tmp_path / "private"
    env._private_audit_dir = None
    env._private_audit_streams = {}
    env._task = SimpleNamespace(camera=SimpleNamespace(
        cfg=SimpleNamespace(width=HEAD_WIDTH, height=HEAD_HEIGHT),
        data=SimpleNamespace(intrinsic_matrices=[Intrinsics()]),
    ), _action_state=lambda: np.zeros(4, dtype=np.float32))
    env._observation = {"state": np.zeros(111, dtype=np.float32)}
    pixel = _project(.47, -.02, .0725)
    pixel = {"u": 1 - pixel["u"], "v": pixel["v"], "color": "white"}
    with pytest.raises(ValueError):
        env.locate_visual_keys({"E4": {**pixel, "v": 1.2}})
    assert not env.visual_key_map_ready
    assert env._private_audit_dir is None

    targets = env.locate_visual_keys({"E4": pixel})
    assert targets["E4"]["midi"] == 64
    assert env.visual_key_map_ready
    assert env.get_observation()["hand_state"]["visual_key_map_ready"] == 1
    assert not (env._output_dir / "shadow_visual_key_maps.jsonl").exists()
    env.seed_base = 3149
    env._frame_buffer = []
    env._task.reset = lambda seed: (env._observation, {})
    env._task.sim = SimpleNamespace(render=lambda: None)
    env._task.camera.update = lambda *args, **kwargs: None
    env._task.cfg = SimpleNamespace(physics_dt=1 / 240)
    env._write_native_event = lambda kind: None
    env._write_private_force_audit = lambda kind: None
    env._write_private_piano_timeline = lambda: None
    env._save_tactile_snapshot = lambda filename: None
    env.reset(seed=1)
    assert not env.visual_key_map_ready
    assert env.get_observation()["hand_state"]["visual_key_map_ready"] == 0
    env._close_private_audit()
    private = env._private_audit_dir / "shadow_visual_key_maps.jsonl"
    saved = [json.loads(line) for line in private.read_text().splitlines()]
    assert [row["ready"] for row in saved] == [1, 0]
    assert saved[0]["targets"]["E4"]["midi"] == 64


def test_private_audit_is_written_immediately_outside_public_artifacts(tmp_path):
    env = ShadowPianoLowLevelEnv.__new__(ShadowPianoLowLevelEnv)
    env.private_audit_root = tmp_path / "private"
    env._output_dir = tmp_path / "public"
    env._private_audit_dir = None
    env._private_audit_streams = {}
    env._append_private_audit("labels.jsonl", {"key_force_N": .5})
    private_file = env._private_audit_dir / "labels.jsonl"
    assert private_file.is_file()
    assert stat.S_IMODE(private_file.stat().st_mode) == 0o200
    assert stat.S_IMODE(env.private_audit_root.stat().st_mode) == 0o300
    assert stat.S_IMODE(env._private_audit_dir.stat().st_mode) == 0o300
    assert not env._output_dir.exists()
    if os.geteuid() != 0:
        with pytest.raises(PermissionError):
            private_file.read_text()
    env._close_private_audit()
    assert json.loads(private_file.read_text().strip()) == {"key_force_N": .5}
    locator = json.loads((env._output_dir / "shadow_private_audit_location.json").read_text())
    assert locator["private_audit_dir"] == str(env._private_audit_dir)


def test_native_audit_requires_press_release_pitch_and_dynamic(tmp_path):
    (tmp_path / "shadow_native_events.jsonl").write_text(json.dumps({
        "kind": "END", "native_info": {"success": True, "failure": "", "notes_completed": 1,
                                       "require_dynamics": True,
                                       "piano_sequence": {"midi": [60], "dynamics": ["light"]}},
    }) + "\n")
    (tmp_path / "shadow_piano_physics_trace.jsonl").write_text(json.dumps({
        "physics_step": 29, "target_midi": 60, "actual_midi": 60,
        "target_dynamic": 1, "actual_dynamic": 1, "actual_force_N": .35,
    }) + "\n")
    (tmp_path / "shadow_piano_score_events.jsonl").write_text(
        json.dumps({"kind": "press", "step": 29, "midi": 60, "force_N": .35}) + "\n" +
        json.dumps({"kind": "release", "step": 53, "midi": 60, "repetition": 1}) + "\n"
    )
    result = json.loads(save_native_piano_summary(tmp_path).read_text())
    assert result["native_success"] and result["end_submitted"]
    assert result["physics_trace_missing_steps"] == 0
    assert result["notes"][0]["native_scored_note"]
    assert result["notes"][0]["observed_dynamic_matches"]


def test_partial_score_audit_keeps_later_notes_aligned(tmp_path):
    info = {"success": False, "failure": "partial_score", "notes_completed": 1,
            "notes_attempted": 2, "notes_failed": 1, "require_dynamics": True,
            "piano_sequence": {"midi": [60, 62], "dynamics": ["light", "heavy"]}}
    (tmp_path / "shadow_native_events.jsonl").write_text(
        json.dumps({"kind": "END", "native_info": info}) + "\n")
    (tmp_path / "shadow_piano_physics_trace.jsonl").write_text(json.dumps({
        "physics_step": 20, "target_midi": 62, "actual_midi": 62,
        "target_dynamic": 2, "actual_dynamic": 2, "actual_force_N": .5,
    }) + "\n")
    events = [
        {"kind": "miss", "attempt_index": 0, "step": 10, "midi": 60,
         "reason": "wrong_key"},
        {"kind": "press", "attempt_index": 1, "step": 20, "midi": 62,
         "force_N": .5},
        {"kind": "release", "attempt_index": 1, "step": 45, "midi": 62,
         "repetition": 2},
    ]
    (tmp_path / "shadow_piano_score_events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events))
    result = json.loads(save_native_piano_summary(tmp_path).read_text())
    assert result["native_notes_completed"] == 1
    assert result["native_notes_attempted"] == 2
    assert result["native_notes_failed"] == 1
    assert result["notes"][0]["native_failed_reason"] == "wrong_key"
    assert not result["notes"][0]["native_scored_note"]
    assert result["notes"][1]["native_scored_note"]


def test_finish_note_attempt_exposes_only_terminal_flag(monkeypatch):
    from types import SimpleNamespace

    env = ShadowPianoLowLevelEnv.__new__(ShadowPianoLowLevelEnv)
    env.piano_continue_on_note_failure = True
    env.visual_key_mode = True
    env._policy_end_submitted = False
    env._public_attempts_finalized = 0
    env.piano_notes = ["C4", "D4"]
    env._task = SimpleNamespace(
        finish_piano_note_attempt=lambda index: {"success": False,
            "notes_completed": 0, "notes_attempted": index + 1,
            "actual_force_N": 1.2},
        _observation=lambda: {"state": "private"}, _done=False,
    )
    for name in ("_write_native_event", "_write_private_piano_timeline",
                 "_save_attempt_snapshots", "_save_tactile_snapshot"):
        monkeypatch.setattr(env, name, lambda *args: None)
    public = ShadowPianoVisualForceApi(env).finish_piano_note_attempt(0)
    assert public == {"ok": True, "native_done": False}
    assert env._native_info["actual_force_N"] == 1.2
    assert "actual_force_N" not in repr(public)
    with pytest.raises(ValueError, match="once in sequence"):
        ShadowPianoVisualForceApi(env).finish_piano_note_attempt(0)
