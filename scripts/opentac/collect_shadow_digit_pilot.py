"""Shadow DIGIT force pilot; run only in an Isaac Kit Python env.

No Responses API or CaP-X generated code is involved.  The expert episode
uses the native demonstration policy; the C4 probe uses a fixed public-geometry
trajectory and never reads measured force for action decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from shadow_digit_pilot_core import PilotRecorder, SCHEMA, audit_episode, prepare_piano_config


EXPERT_NOTES = ["C#4", "C4", "G4", "A4", "E4", "F4"]
EXPERT_DYNAMICS = ["light", "heavy"] * 3
WHITE_NOTES = ("C4", "D4", "E4", "F4", "G4", "A4", "B4")
BLACK_NOTES = ("C#4", "D#4", "F#4", "G#4", "A#4")


def _balanced_eight(seed: int) -> tuple[list[str], list[str]]:
    """Pair light/heavy on each of four keys; balance both phrase halves."""
    index = seed - 3100
    white = [WHITE_NOTES[(2 * index + j) % len(WHITE_NOTES)] for j in range(2)]
    black = [BLACK_NOTES[(2 * index + j) % len(BLACK_NOTES)] for j in range(2)]
    first = [(white[0], "light"), (white[1], "heavy"),
             (black[0], "light"), (black[1], "heavy")]
    second = [(white[0], "heavy"), (white[1], "light"),
              (black[0], "heavy"), (black[1], "light")]
    rng = np.random.default_rng(seed)
    rng.shuffle(first)
    rng.shuffle(second)
    pairs = first + second
    return [note for note, _ in pairs], [dynamic for _, dynamic in pairs]


def _expert_request(root: Path, *, seed: int, sequence_config: Path | None,
                    random_note_count: int | None, balanced_eight: bool = False) -> dict:
    """Resolve a phrase with the task's own note/dynamics/hold parser."""
    if sequence_config is None and random_note_count is None and not balanced_eight:
        return dict(notes=EXPERT_NOTES, dynamics=EXPERT_DYNAMICS,
                    holds=[.12] * len(EXPERT_NOTES), strict=True,
                    source="legacy_example", suffix="")

    import yaml
    sys.path.insert(0, str(root))
    from envs._piano_sequence import make_sequence

    if balanced_eight:
        notes, dynamics = _balanced_eight(seed)
        count, holds, strict = 8, .12, True
        source, config_sha256 = "balanced_four_groups_8note_v1", None
    elif sequence_config is not None:
        path = sequence_config.expanduser().resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Piano sequence config must be a YAML mapping")
        notes = config.get("piano_notes")
        count = config.get("piano_note_count", 6)
        dynamics = config.get("piano_dynamics")
        holds = config.get("piano_hold_seconds", .12)
        strict = config.get("piano_require_dynamics", True)
        if not isinstance(strict, bool):
            raise ValueError("piano_require_dynamics must be boolean")
        source = str(path)
        config_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        notes, count, dynamics, holds, strict = None, random_note_count, None, .12, True
        source, config_sha256 = f"seeded_random_{count}", None
    sequence = make_sequence(notes, count=count, dynamics=dynamics, holds=holds,
                             rng=np.random.default_rng(seed))
    normalized = sequence.as_dict()
    identity = dict(sequence=normalized, strict=strict, seed=seed)
    suffix = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    return dict(notes=normalized["notes"], dynamics=normalized["dynamics"],
                holds=normalized["hold_seconds"], strict=strict,
                source=source, source_sha256=config_sha256, suffix=suffix)


def _commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _check_force_ramp(root: Path, config: Path, output_root: Path):
    """Exercise the exact piano renderer before any native collection."""
    from PIL import Image

    ramp_dir = output_root / "fixed_geometry_ramp"
    report_file = ramp_dir / "ramp.json"
    if not ramp_dir.exists():
        temporary = output_root / f"fixed_geometry_ramp_incomplete_{os.getpid()}"
        python = root / ".cache/piano_tactile/digit/venv/bin/python"
        script = Path(__file__).with_name("shadow_digit_renderer.py")
        environment = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME"):
            environment.pop(name, None)
        environment.update(PYTHONNOUSERSITE="1", PYOPENGL_PLATFORM="egl")
        subprocess.run([str(python), str(script), "--ramp",
                        str(root / ".cache/piano_tactile/digit"), str(config),
                        str(temporary)], check=True, env=environment)
        temporary.rename(ramp_dir)
    if not report_file.is_file():
        raise RuntimeError(f"Incomplete fixed-geometry TACTO ramp: {ramp_dir}")
    rows = json.loads(report_file.read_text(encoding="utf-8"))
    values = [0., .35, .5, .7, 1., 1.5, 2., 2.5]
    if [row["force_N"] for row in rows] != values:
        raise RuntimeError("Unexpected TACTO force-ramp sample points")
    if any(row["valid_depth_fraction"] <= 0 or row["depth_delta_m"] is None for row in rows):
        raise RuntimeError("Fixed-geometry TACTO ramp has invalid depth")
    target_images = [np.asarray(Image.open(ramp_dir / f"force_{force:g}_N.png"), dtype=np.int16)
                     for force in (.35, 1., 2., 2.5)]
    if (not np.any(target_images[0] != target_images[1]) or
            not np.array_equal(target_images[2], target_images[3])):
        raise RuntimeError("TACTO ramp does not distinguish target forces or saturate above 2 N")
    if abs(rows[-2]["depth_delta_m"] - rows[-1]["depth_delta_m"]) > 1e-8:
        raise RuntimeError("TACTO depth did not saturate at 2 N")
    print(f"PILOT_RAMP_OK {report_file}", flush=True)


def _configure_camera(task):
    import torch

    target_y = task.layout[task.target].position[1]
    task.camera.set_world_poses_from_view(
        torch.tensor([[-.58, .85, .66]], device=task.sim.device),
        torch.tensor([[.12, target_y, .095]], device=task.sim.device),
    )


def _public_state(task, observation) -> tuple[np.ndarray, np.ndarray]:
    native = np.asarray(observation["state"], np.float32)
    if native.shape != (111,):
        raise ValueError(f"Unexpected native Shadow state shape: {native.shape}")
    command_state = np.asarray(task._action_state(), np.float32)
    if command_state.shape != (4,):
        raise ValueError("Native Shadow action state is not four-dimensional")
    # Native indices 94:97 are score count/armed/active key and stay private.
    hand = np.r_[command_state, native[3:6], native[97:104], native[104:111]].astype(np.float32)
    return hand, native[6:94].copy()


def _probe_actions(task):
    from envs._piano import keyboard_layout

    key = keyboard_layout()[60 - 21]
    y = float(key.position[1] - .033)
    start = np.asarray(task._action_state(), float)
    if start.shape != (4,):
        raise RuntimeError("Unexpected native Shadow wrist/index state")
    stages = [("hover", [0., y, .04, 0.], 45),
              ("descend", [0., y, 0., 0.], 25),
              ("flex", [0., y, 0., task.cfg.press_joint_angle], 20),
              ("hold", [0., y, 0., task.cfg.press_joint_angle], 28),
              ("release", [0., y, 0., 0.], 18),
              ("lift", [0., y, .04, 0.], 25)]
    for phase, target, count in stages:
        target = np.asarray(target, float)
        for index in range(1, count + 1):
            yield phase, np.asarray(start + (target - start) * index / count, np.float32), 60
        start = target


def _run_episode(*, root: Path, capx_root: Path, output_root: Path,
                 config_path: Path, config_provenance: dict,
                 episode_kind: str, seed: int, app, expert_request: dict,
                 record_video: bool = True, action_repeat: int = 8,
                 matched_probe: bool = False, probe_profile: str = "runtime"):
    from env.tasks.arm_shadow_piano import Task, TaskCfg
    import env.tasks._digit_tactile as native_digit
    from shadow_digit_bridge import make_digit_bridge_class

    is_expert = episode_kind == "expert"
    use_phrase = is_expert or matched_probe
    notes = expert_request["notes"] if use_phrase else ["C4"]
    dynamics = expert_request["dynamics"] if use_phrase else ["light"]
    holds = expert_request["holds"] if use_phrase else .6
    strict = expert_request["strict"] if use_phrase else False
    suffix = f"_{expert_request['suffix']}" if use_phrase and expert_request["suffix"] else ""
    if matched_probe and not is_expert:
        suffix += f"_{probe_profile}"
    episode_dir = output_root / "episodes" / f"{seed}_{episode_kind}{suffix}"
    metadata = dict(schema=SCHEMA, seed=seed, controller=episode_kind,
                    notes=notes, dynamics=dynamics, hold_seconds=holds,
                    require_dynamics=strict, physics_dt_s=1 / 240,
                    action_repeat=action_repeat, public_control_frequency_hz=240 / action_repeat,
                    record_video=record_video,
                    opentac_root=str(root), opentac_commit=_commit(root),
                    capx_commit=_commit(capx_root),
                    collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    metadata.update(config_provenance)
    if use_phrase:
        metadata["sequence_source"] = expert_request["source"]
        if expert_request.get("source_sha256") is not None:
            metadata["sequence_source_sha256"] = expert_request["source_sha256"]
    plan = None
    if matched_probe and not is_expert:
        from shadow_digit_matched_probe import probe_plan
        plan = probe_plan(notes, dynamics, seed=seed, profile=probe_profile)
        metadata["probe_plan"] = plan
        metadata["probe_policy_sha256"] = hashlib.sha256(
            Path(__file__).with_name("shadow_digit_matched_probe.py").read_bytes()).hexdigest()
    recorder = PilotRecorder(episode_dir, metadata=metadata)
    task = None
    scene_writer = None
    tactile_writer = None
    video_steps = []
    audio_info = None
    status = "running"
    error = None
    start = time.monotonic()
    try:
        cfg = TaskCfg(device="cuda:0", tactile=True, tactile_sensor="digit",
                      tactile_image_width=320, piano_notes=notes,
                      piano_dynamics=dynamics, piano_hold_seconds=holds,
                      piano_require_dynamics=strict, piano_seed=seed,
                      record_video=record_video, action_repeat=action_repeat,
                      piano_continue_on_note_failure=bool(plan))
        original_class = native_digit.DigitTactile
        native_digit.DigitTactile = make_digit_bridge_class(
            original_class, opentac_root=root, piano_config=config_path,
            episode_dir=episode_dir)
        try:
            task = Task(cfg=cfg, render_mode="rgb_array" if record_video else None)
        finally:
            native_digit.DigitTactile = original_class
        if record_video:
            _configure_camera(task)
        observation, info = task.reset(seed=seed)
        if (list(task.sequence.as_dict()["notes"]) != list(notes) or
                list(task.sequence.dynamics) != list(dynamics) or
                list(task.sequence.holds) != (list(holds) if use_phrase else [holds])):
            raise RuntimeError("Native piano sequence differs from the requested phrase")
        if record_video:
            for _ in range(16):
                task.sim.render()
                task.camera.update(task.cfg.physics_dt, force_recompute=True)
        if not isinstance(observation, dict) or task.tactile.depth_m is None:
            raise RuntimeError("CaP-X DIGIT bridge did not return RGB and depth")
        if task.tactile._geometry["config_path"] != str(config_path):
            raise RuntimeError("The running TACTO renderer used a different config")
        metadata["tacto_version"] = task.tactile._geometry["tacto_version"]
        recorder.metadata["tacto_version"] = metadata["tacto_version"]
        recorder.public.attrs["metadata"] = json.dumps(metadata, sort_keys=True)

        def append_private(step, angles, forces, requested_midi):
            recorder.append_private(
                physics_step=int(step), angles=angles, forces=forces,
                requested_midi=int(requested_midi), score_count=int(task.score.count),
                score_armed=bool(task.score._armed), score_success=bool(task.score.success),
                score_failure=str(task.score.failure))

        angles0, forces0 = task._key_state()
        append_private(0, angles0, forces0, int(task.cfg.melody[0]))
        native_advance = task.score.advance
        def observed_advance(angles, forces, dt, step):
            requested = task.score.active_key + 21
            result = native_advance(angles, forces, dt, step)
            append_private(step, angles, forces, requested)
            return result
        task.score.advance = observed_advance

        def append_public(observation, command, requested_midi, phase, note_index=0):
            hand, key_angles = _public_state(task, observation)
            recorder.append_public(
                physics_step=int(task.step_count), rgb=observation["tactile"]["index"]["rgb"],
                depth_m=task.tactile.depth_m, reference_depth_m=task.tactile.reference_depth_m,
                hand_state=hand, key_angles_rad=key_angles, action_target=command,
                requested_midi=requested_midi, phase=phase, requested_note_index=note_index)

        append_public(observation, np.asarray(task._action_state(), np.float32),
                      int(task.cfg.melody[0]), "expert" if is_expert else "hover")
        if record_video:
            import imageio.v2 as imageio
            fps = metadata["public_control_frequency_hz"]
            scene_writer = imageio.get_writer(episode_dir / "video.mp4", fps=fps, codec="libx264", quality=8)
            tactile_writer = imageio.get_writer(episode_dir / "digit.mp4", fps=fps, codec="libx264", quality=8)
        if is_expert:
            while app.is_running() and not task._done:
                command = np.asarray(task.demonstration_action("demo"), np.float32)
                note_index = min(task._demo_note, len(task.cfg.melody) - 1)
                requested = int(task.cfg.melody[note_index])
                observation, _, terminated, truncated, info = task.step(command)
                append_public(observation, command, requested, "expert", note_index)
                if record_video:
                    scene_writer.append_data(task.render())
                    tactile_writer.append_data(observation["tactile"]["index"]["rgb"])
                    video_steps.append(int(task.step_count))
                if terminated or truncated:
                    break
        else:
            if plan:
                from shadow_digit_matched_probe import matched_probe_actions
                actions = matched_probe_actions(task, plan)
            else:
                actions = ((phase, command, requested, 0, False)
                           for phase, command, requested in _probe_actions(task))
            for phase, command, requested, note_index, finish in actions:
                if not app.is_running() or task._done:
                    break
                if finish:
                    task.finish_piano_note_attempt(note_index)
                    continue
                if not task.action_space.contains(command):
                    raise ValueError(f"Probe command outside native bounds: {command}")
                observation, _, terminated, truncated, info = task.step(command)
                append_public(observation, command, requested, phase, note_index)
                if record_video:
                    scene_writer.append_data(task.render())
                    tactile_writer.append_data(observation["tactile"]["index"]["rgb"])
                    video_steps.append(int(task.step_count))
                if terminated or truncated:
                    break
        task.audio_contacts.finish(task.step_count)
        if scene_writer is not None:
            scene_writer.close()
            scene_writer = None
        if tactile_writer is not None:
            tactile_writer.close()
            tactile_writer = None
        if video_steps:
            from envs._piano_audio import add_piano_audio
            audio_info = add_piano_audio(episode_dir / "video.mp4", task.audio_contacts.events,
                                         video_steps, fps=metadata["public_control_frequency_hz"])
        status = "complete" if task._done else "probe_finished" if not is_expert else "interrupted"
    except BaseException as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if scene_writer is not None:
                scene_writer.close()
            if tactile_writer is not None:
                tactile_writer.close()
            recorder.close()
            result = dict(metadata=metadata, status=status, error=error,
                          wall_seconds=time.monotonic() - start,
                          physics_step=int(task.step_count) if task is not None and hasattr(task, "step_count") else None,
                          audio=audio_info,
                          audio_events=task.audio_contacts.events if task is not None and hasattr(task, "audio_contacts") else [],
                          native_info=task._info() if task is not None and hasattr(task, "score") else None,
                          native_events=task.score.events if task is not None and hasattr(task, "score") else [],
                          tactile_metadata=task.tactile.metadata() if task is not None and task.tactile is not None else None)
            (episode_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            if recorder.public_count:
                audit = audit_episode(episode_dir)
                print(f"PILOT_RESULT {episode_kind} {json.dumps(audit)}", flush=True)
        finally:
            if task is not None:
                task.close()


def main():
    parser = argparse.ArgumentParser(description="Collect Shadow DIGIT force pilot episodes")
    parser.add_argument("--opentac-root", type=Path, required=True)
    parser.add_argument("--capx-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/data/ljz/opentacbench/shadow_digit_force_v1"))
    parser.add_argument("--episode", choices=("expert", "probe", "both"), default="both")
    parser.add_argument("--expert-seed", type=int, default=3000)
    parser.add_argument("--probe-seed", type=int, default=3001)
    parser.add_argument("--action-repeat", type=int, choices=(4, 8), default=8,
                        help="4: 60 Hz public data; 8: legacy 30 Hz. Physics stays 240 Hz")
    parser.add_argument("--matched-probe", action="store_true",
                        help="Use the specified phrase and public-state CaP-X-style probes")
    parser.add_argument("--probe-profile", choices=("runtime", "higher", "lower", "slower", "varied"),
                        default="runtime")
    phrase = parser.add_mutually_exclusive_group()
    phrase.add_argument("--sequence-config", type=Path,
                        help="YAML with piano_notes, piano_dynamics and piano_hold_seconds")
    phrase.add_argument("--random-note-count", type=int,
                        help="Seeded random C4-B4 phrase of this length")
    phrase.add_argument("--balanced-eight", action="store_true",
                        help="Eight notes: two light and two heavy for each key color")
    parser.add_argument("--no-video", action="store_true",
                        help="Keep synchronized RGB/depth/force HDF5 data but omit MP4/WAV")
    args = parser.parse_args()
    if args.probe_profile != "runtime" and not args.matched_probe:
        parser.error("--probe-profile requires --matched-probe")
    root = args.opentac_root.expanduser().resolve()
    capx_root = args.capx_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    expert_request = _expert_request(root, seed=args.expert_seed,
                                     sequence_config=args.sequence_config,
                                     random_note_count=args.random_note_count,
                                     balanced_eight=args.balanced_eight)
    print(f"PILOT_PHRASE {json.dumps(expert_request, ensure_ascii=False, sort_keys=True)}", flush=True)
    original = root / ".cache/piano_tactile/digit/venv/lib/python3.11/site-packages/tacto/config_digit.yml"
    config = output_root / "config" / "config_digit_piano_0_2.yml"
    config_provenance = prepare_piano_config(original, config)
    # TACTO itself lives in an isolated subprocess env, not Isaac's Python.
    if not (root / ".cache/piano_tactile/digit/venv/bin/python").is_file():
        raise RuntimeError("TACTO isolated Python is missing")
    _check_force_ramp(root, config, output_root)
    sys.path.insert(0, str(root))
    from isaaclab.app import AppLauncher
    app = AppLauncher(argparse.Namespace(enable_cameras=not args.no_video,
                                        num_envs=1, headless=True)).app
    try:
        if args.episode in ("expert", "both"):
            _run_episode(root=root, capx_root=capx_root, output_root=output_root,
                         config_path=config, config_provenance=config_provenance,
                         episode_kind="expert", seed=args.expert_seed, app=app,
                         expert_request=expert_request, record_video=not args.no_video,
                         action_repeat=args.action_repeat)
        if args.episode in ("probe", "both"):
            _run_episode(root=root, capx_root=capx_root, output_root=output_root,
                         config_path=config, config_provenance=config_provenance,
                         episode_kind="probe", seed=args.probe_seed, app=app,
                         expert_request=expert_request, record_video=not args.no_video,
                         action_repeat=args.action_repeat, matched_probe=args.matched_probe,
                         probe_profile=args.probe_profile)
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
