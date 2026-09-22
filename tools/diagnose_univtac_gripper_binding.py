#!/usr/bin/env python3
"""Record focused UniVTAC gripper-binding diagnostics on one reset.

The diagnostic is deliberately LLM-free.  It can compare the native and
CaP-X opening paths, or replay the *motion* part of CaP-X's generated first
probe using the same public API calls.  Every simulation step records finger
positions relative to the wrist and uses the task's normal video framing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record expert-vs-CaP-X gripper opening and binding diagnostics."
    )
    parser.add_argument("--univtac-root", required=True)
    parser.add_argument(
        "--task-config",
        default="tactile_memory_match_composable_capx_demo",
        help="UniVTAC task_config name or YAML path.",
    )
    parser.add_argument(
        "--capx-config",
        default=str(REPO_ROOT / "env_configs/univtac/tactile_memory_match_easy_sam_gt.yaml"),
        help="CaP-X YAML used to load the production FrankaControlApi settings.",
    )
    parser.add_argument("--seed", type=int, default=4002)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scenario",
        choices=("open-ab", "capx-first-probe"),
        default="open-ab",
        help=(
            "open-ab compares expert and CaP-X opening. capx-first-probe replays "
            "the generated open/approach/close/preload/lift/hold/lower/release chain."
        ),
    )
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--open-max-steps", type=int, default=120)
    return parser.parse_args()


def _vec(value: Any) -> list[float]:
    return [float(x) for x in np.asarray(value, dtype=np.float64).reshape(-1).tolist()]


def _as_uint8_frame(frame: Any) -> np.ndarray:
    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    if hasattr(frame, "numpy"):
        frame = frame.numpy()
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(array)


def _relative_position(
    hand_pos: np.ndarray,
    hand_quat_wxyz: np.ndarray,
    finger_pos: np.ndarray,
) -> np.ndarray:
    hand_rot = Rotation.from_quat(
        np.asarray(hand_quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    )
    return hand_rot.inv().apply(np.asarray(finger_pos) - np.asarray(hand_pos))


class _TraceRecorder:
    def __init__(self, low_level: Any) -> None:
        self.low_level = low_level
        self.phase = "reset"
        self.records: list[dict[str, Any]] = []
        self.official_frames: list[np.ndarray] = []
        self.wrist_frames: list[np.ndarray] = []
        robot = low_level.task._robot_manager.robot
        self.robot = robot
        self.body_ids = {}
        for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
            ids, _ = robot.find_bodies(name)
            if len(ids) != 1:
                raise RuntimeError(
                    f"expected exactly one body named {name!r}; got {list(ids)} from {robot.body_names}"
                )
            self.body_ids[name] = int(ids[0])

    def capture(self, *, include_frame: bool = True) -> None:
        manager = self.low_level.task._robot_manager
        pos_w = self.robot.data.body_link_pos_w[0].detach().cpu().numpy()
        quat_w = self.robot.data.body_link_quat_w[0].detach().cpu().numpy()
        hand_id = self.body_ids["panda_hand"]
        hand_pos = pos_w[hand_id]
        hand_quat = quat_w[hand_id]
        fingers: dict[str, Any] = {}
        for name in ("panda_leftfinger", "panda_rightfinger"):
            body_id = self.body_ids[name]
            finger_pos = pos_w[body_id]
            fingers[name] = {
                "world_position_m": _vec(finger_pos),
                "relative_to_hand_m": _vec(
                    _relative_position(hand_pos, hand_quat, finger_pos)
                ),
            }
        joint_pos = self.robot.data.joint_pos[0].detach().cpu().numpy()
        gripper_ids = manager._gripper_ids.detach().cpu().numpy().astype(int)
        self.records.append(
            {
                "phase": self.phase,
                "step": int(self.low_level.get_step_count()),
                "gripper_qpos": _vec(joint_pos[gripper_ids]),
                "gripper_qpos_max": float(manager.gripper_max_qpos),
                "hand_world_position_m": _vec(hand_pos),
                "fingers": fingers,
            }
        )
        if not include_frame:
            return
        # Match the exact frame composition used by BaseTask's official video.
        # The preceding native _step already renders during active actions.
        # Do not force a second render immediately after reset: that boundary
        # can block in Isaac/UIPC before any diagnostic action has begun.
        observation = self.low_level.task._get_observations()
        self.official_frames.append(
            _as_uint8_frame(self.low_level.task.get_frame_shot(observation))
        )
        wrist = observation.get("observation", {}).get("wrist", {}).get("rgb")
        if wrist is not None:
            self.wrist_frames.append(_as_uint8_frame(wrist))


def _max_relative_hold_motion(records: list[dict[str, Any]], phase: str) -> float:
    selected = [item for item in records if item["phase"] == phase]
    if len(selected) < 2:
        return float("nan")
    baseline = selected[0]["fingers"]
    max_motion = 0.0
    for item in selected[1:]:
        for name, values in item["fingers"].items():
            delta = np.asarray(values["relative_to_hand_m"]) - np.asarray(
                baseline[name]["relative_to_hand_m"]
            )
            max_motion = max(max_motion, float(np.linalg.norm(delta)))
    return max_motion


def _phase_qpos_span(records: list[dict[str, Any]], phase: str) -> float:
    values = [x for item in records if item["phase"] == phase for x in item["gripper_qpos"]]
    return float(max(values) - min(values)) if values else float("nan")


def _endpoint_finger_delta(records: list[dict[str, Any]]) -> float:
    expert = next((item for item in records if item["phase"] == "expert_open_hold"), None)
    capx = next((item for item in records if item["phase"] == "capx_open_hold"), None)
    if expert is None or capx is None:
        return float("nan")
    return max(
        float(
            np.linalg.norm(
                np.asarray(expert["fingers"][name]["relative_to_hand_m"])
                - np.asarray(capx["fingers"][name]["relative_to_hand_m"])
            )
        )
        for name in ("panda_leftfinger", "panda_rightfinger")
    )


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    threshold_m = 1e-4
    endpoint_threshold_m = 1e-3
    report: dict[str, Any] = {
        "binding_hold_motion_threshold_m": threshold_m,
        "expert_capx_endpoint_threshold_m": endpoint_threshold_m,
    }
    for prefix in ("expert", "capx"):
        hold_phase = f"{prefix}_open_hold"
        qpos_span = _phase_qpos_span(records, hold_phase)
        relative_motion = _max_relative_hold_motion(records, hold_phase)
        report[prefix] = {
            "hold_qpos_span": qpos_span,
            "hold_relative_finger_motion_m": relative_motion,
            "hold_binding_stable": bool(
                np.isfinite(relative_motion) and relative_motion <= threshold_m
            ),
        }

    capx = report["capx"]
    expert = report["expert"]
    endpoint_delta = _endpoint_finger_delta(records)
    report["expert_capx_open_endpoint_delta_m"] = endpoint_delta
    capx_mismatch = bool(
        (not capx["hold_binding_stable"])
        or (np.isfinite(endpoint_delta) and endpoint_delta > endpoint_threshold_m)
    )
    if capx_mismatch and expert["hold_binding_stable"]:
        verdict = "capx_path_binding_or_control_mismatch"
    elif capx_mismatch and not expert["hold_binding_stable"]:
        verdict = "shared_asset_or_sim_render_binding_issue"
    elif capx["hold_binding_stable"]:
        verdict = "no_relative_finger_drift_observed"
    else:
        verdict = "insufficient_trace"
    report["verdict"] = verdict
    return report


def _summarize_probe_binding(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report relative-finger drift only during phases with fixed gripper qpos."""
    threshold_m = 1e-4
    stable_phases = (
        "capx_probe_approach",
        "capx_probe_preload",
        "capx_probe_lift",
        "capx_probe_hold",
        "capx_probe_lower",
        "capx_probe_release_hold",
    )
    phases: dict[str, Any] = {}
    for phase in stable_phases:
        samples = [item for item in records if item["phase"] == phase]
        if not samples:
            continue
        qpos_span = _phase_qpos_span(records, phase)
        relative_motion = _max_relative_hold_motion(records, phase)
        phases[phase] = {
            "sample_count": len(samples),
            "qpos_span": qpos_span,
            "relative_finger_motion_m": relative_motion,
            "binding_stable": bool(
                np.isfinite(relative_motion) and relative_motion <= threshold_m
            ),
        }
    unstable = [name for name, report in phases.items() if not report["binding_stable"]]
    return {
        "binding_hold_motion_threshold_m": threshold_m,
        "phases": phases,
        "verdict": "probe_relative_finger_drift_observed" if unstable else "no_probe_relative_finger_drift_observed",
        "unstable_phases": unstable,
    }


def _load_franka_api_config(path: str) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    try:
        api_configs = config["env"]["cfg"]["low_level"]["api_configs"]
        franka_config = api_configs["franka_control_api"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"could not find env.cfg.low_level.api_configs.franka_control_api in {config_path}"
        ) from exc
    if not isinstance(franka_config, dict):
        raise ValueError("franka_control_api config must be a mapping")
    return {"franka_control_api": dict(franka_config)}


def main() -> None:
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CAPX_UNIVTAC_MINIMAL_IMPORTS", "1")

    univtac_root = Path(args.univtac_root).expanduser().resolve()
    if not univtac_root.is_dir():
        raise FileNotFoundError(f"UniVTAC root not found: {univtac_root}")
    if str(univtac_root) not in sys.path:
        sys.path.insert(0, str(univtac_root))

    original_argv = sys.argv[:]
    app = None
    low_level = None
    try:
        sys.argv = [original_argv[0]]
        from isaaclab.app import AppLauncher

        # Keep camera rendering for the diagnostic videos, without loading GUI
        # extensions that require an X11/desktop window on the remote server.
        app = AppLauncher(
            argparse.Namespace(enable_cameras=True, headless=True, num_envs=1)
        ).app
    finally:
        sys.argv = original_argv

    from capx.envs.simulators.univtac import UniVTACLowLevelEnv
    from capx.integrations.univtac.franka_compat_api import UniVTACFrankaCompatApi
    from capx.utils.video_utils import _write_video

    try:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        low_level = UniVTACLowLevelEnv(
            univtac_root=str(univtac_root),
            task_name="tactile_memory_match",
            task_config=args.task_config,
            seed_base=args.seed,
            device="cuda:0",
            task_config_overrides={
                # Match the normal expert start pose so the diagnostic video
                # uses the same camera framing as official trajectories.
                "skip_task_pre_move": False,
                "skip_pre_move_render": True,
                "record_video_during_reset": False,
                "record_action_frames": False,
                "record_pre_move_frames": False,
                "video_frame_stride": 1,
            },
            api_configs=_load_franka_api_config(args.capx_config),
            expose_actor_pose=False,
            enable_render=True,
        )
        low_level.reset(seed=0)
        recorder = _TraceRecorder(low_level)
        original_step = low_level.task._step

        def traced_step(*step_args, **step_kwargs):
            result = original_step(*step_args, **step_kwargs)
            recorder.capture()
            return result

        low_level.task._step = traced_step
        recorder.capture(include_frame=False)

        # Reproduce the official expert's percentage/adaptive action path.
        def expert_gripper(percent: float, phase: str) -> None:
            recorder.phase = phase
            previous_pre_move = bool(low_level.task.in_pre_move)
            low_level.task.in_pre_move = True
            try:
                low_level.task.move(
                    low_level.task.atom.open_gripper(percent)
                    if percent >= 1.0
                    else low_level.task.atom.close_gripper(percent),
                    tag=phase,
                    is_save=True,
                    delay=False,
                )
            finally:
                low_level.task.in_pre_move = previous_pre_move

        api = UniVTACFrankaCompatApi(low_level)
        stage_results: dict[str, Any] = {}
        if args.scenario == "open-ab":
            expert_gripper(0.0, "expert_prepare_close")
            expert_gripper(1.0, "expert_open")
            recorder.phase = "expert_open_hold"
            low_level.task.delay(args.hold_steps, is_save=True, force=True)

            expert_gripper(0.0, "capx_prepare_close")
            recorder.phase = "capx_open"
            capx_open = api.open_gripper(
                adaptive=True,
                target_width=1.0,
                max_steps=args.open_max_steps,
            )
            stage_results["capx_open"] = capx_open
            recorder.phase = "capx_open_hold"
            api.wait_steps(args.hold_steps)
            summary = _summarize(recorder.records)
            video_suffix = "expert_vs_capx_open"
        else:
            # This is the exact visible motion sequence emitted by the first
            # generated CaP-X program. Capture bookkeeping is intentionally
            # omitted: it never changes the simulator state.
            controls = api.get_callable_functions()
            probe_spec = low_level.get_public_probe_spec()
            object_name = "reference_object"

            def run_probe_stage(phase: str, action):
                recorder.phase = phase
                print(f"[gripper-diagnostic] stage={phase} begin", flush=True)
                result = action()
                ok = result.get("ok", result.get("released", None)) if isinstance(result, dict) else None
                print(
                    f"[gripper-diagnostic] stage={phase} end ok={ok}",
                    flush=True,
                )
                return result

            stage_results["initial_open"] = run_probe_stage(
                "capx_probe_initial_open", controls["open_gripper"]
            )
            pos, quat = controls["sample_grasp_pose"](object_name)
            stage_results["approach"] = run_probe_stage(
                "capx_probe_approach", lambda: controls["goto_pose"](pos, quat)
            )
            stage_results["close"] = run_probe_stage(
                "capx_probe_close", lambda: controls["close_gripper"](mode="probe")
            )

            stage_results["preload"] = run_probe_stage(
                "capx_probe_preload",
                lambda: controls["wait_steps"](int(probe_spec["preload_steps"])),
            )
            stage_results["lift"] = run_probe_stage(
                "capx_probe_lift",
                lambda: controls["move_delta"](dz=float(probe_spec["lift_height"])),
            )
            stage_results["hold"] = run_probe_stage(
                "capx_probe_hold",
                lambda: controls["wait_steps"](int(probe_spec["hold_steps"])),
            )
            stage_results["lower"] = run_probe_stage(
                "capx_probe_lower",
                lambda: controls["move_delta"](dz=-float(probe_spec["lift_height"])),
            )
            stage_results["lower_settle"] = run_probe_stage(
                "capx_probe_lower_settle",
                lambda: controls["wait_steps"](int(probe_spec["lower_settle_steps"])),
            )
            stage_results["release"] = run_probe_stage(
                "capx_probe_release", controls["open_gripper"]
            )
            stage_results["release_hold"] = run_probe_stage(
                "capx_probe_release_hold",
                lambda: controls["wait_steps"](args.hold_steps),
            )
            summary = _summarize_probe_binding(recorder.records)
            video_suffix = "capx_first_probe"

        trace_payload = {
            "schema_version": "univtac_gripper_binding_trace.v2",
            "seed": int(args.seed),
            "task_config": str(args.task_config),
            "scenario": str(args.scenario),
            "stage_results": stage_results,
            "records": recorder.records,
        }
        with open(output_dir / "binding_trace.json", "w", encoding="utf-8") as f:
            json.dump(trace_payload, f, indent=2, sort_keys=True)
        with open(output_dir / "binding_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        if recorder.official_frames:
            _write_video(
                recorder.official_frames,
                str(output_dir),
                suffix=video_suffix,
            )
        if recorder.wrist_frames:
            _write_video(
                recorder.wrist_frames,
                str(output_dir),
                suffix=f"{video_suffix}_wrist",
            )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    finally:
        if low_level is not None:
            low_level.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
