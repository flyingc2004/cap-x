"""Single-trial execution for CaP-X environments.

This module handles single trial execution including code generation,
multi-turn decisions, and visual feedback. It contains the core trial
loop extracted from launch.py, covering:

- Initial code generation and oracle code handling
- Code block execution with multi-turn regeneration
- Visual feedback capture and image/video differencing
- Trial artifact saving (code, logs, per-turn videos, combined video)
"""

from __future__ import annotations

import base64
import copy
import gc
import io
import json
import os
import re
import signal
import time
from typing import Any

import numpy as np
from PIL import Image

from capx.envs.configs.instantiate import instantiate
from capx.envs.tasks.base import CodeExecutionEnvBase

from capx.llm.client import (
    VLM_MODELS,
    ModelQueryArgs,
    query_model as _query_model,
    query_model_ensemble as _query_model_ensemble,
    query_single_model_ensemble as _query_single_model_ensemble,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _build_multi_turn_decision_prompt_legacy,
    _extract_code,
    _get_visual_feedback,
    _normalize_extracted_code,
    _parse_multi_turn_decision,
    _save_in_progress_trial_artifacts,
    _save_trial_artifacts,
)
from capx.utils.video_utils import _encode_video_base64, _write_video

# Use TYPE_CHECKING to avoid circular imports for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs


MULTITURN_LIMIT = 10

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _annotate_code_blocks(
    code_blocks: list[str],
    code_block_metadata: list[dict[str, Any]],
) -> str:
    """Join code blocks into a single string with ``# Code block N`` headers."""
    annotated = []
    for i, (block, metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        annotated.append(f"# Code block {i}\n{block}")
    return "\n\n".join(annotated)


def _extract_configured_code_blocks(content: str, config: dict[str, Any]) -> list[str]:
    """Extract Python code and optionally split explicit stage breakpoints."""
    blocks = _extract_code(content)
    memory_cfg = dict(config.get("tactile_code_memory", {}))
    should_split = bool(
        config.get("split_code_blocks_on_breakpoint", False)
        or memory_cfg.get("split_code_blocks_on_breakpoint", False)
    )
    if not should_split:
        return blocks

    split_blocks: list[str] = []
    for block in blocks:
        parts = re.split(
            r"(?m)^breakpoint_code_block\(\)\s*(?:#.*)?$",
            block,
        )
        split_blocks.extend(_normalize_extracted_code(part) for part in parts if part.strip())
    return split_blocks or blocks


def _build_log_lines(
    final_code: str,
    info_step: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    num_code_blocks: int,
    *,
    prefix: str = "",
    stderr_override: str | None = None,
) -> list[str]:
    """Build the standard log-line list used for both normal and timeout summaries."""
    stderr = stderr_override if stderr_override is not None else info_step.get("stderr", "")
    lines = ["-" * 100]
    if prefix:
        lines.append(prefix)
    lines.extend([
        "Generated program:",
        final_code if final_code else "(no program available)",
        "\n\nEnvironment response:",
        f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {stderr}",
        f"  Reward: {reward}",
        f"  Task Completed: {info_step.get('task_completed', False)}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {num_code_blocks}",
        "-" * 100,
    ])
    return lines


def _is_console_progress_noise(line: str) -> bool:
    """Return whether a console line is progress-rendering noise."""
    text = line.strip()
    if not text:
        return True
    if re.match(r"^Running Trials:\s+", text):
        return True
    if re.match(r"^Step\s+\d+\(", text) and "FPS" in text and "Running" in text:
        return True
    if " FPS " in text and " Running " in text and re.search(r"\b(action|pre moving)\b", text):
        return True
    return False


def _is_console_important_for_multiturn(line: str) -> bool:
    """Return whether a console line should always be kept for regeneration."""
    text = line.strip()
    if not text:
        return False
    important_prefixes = (
        "CAPX_",
        "[univtac-franka]",
        "[univtac-tactile]",
        "[capx-task]",
        "[capx-trial]",
        "[capx-univtac]",
        "Traceback",
        "File ",
    )
    if text.startswith(important_prefixes):
        return True
    important_tokens = (
        "Error",
        "Exception",
        "TimeoutError",
        "RuntimeError",
        "ValueError",
        "KeyError",
        "AttributeError",
        "TypeError",
        "RecoverableTaskFailure",
        "motion planning failed",
        "adaptive_close",
        "adaptive_open",
        "Task completed",
    )
    return any(token in text for token in important_tokens)


def _filter_console_for_multiturn(
    text: str,
    *,
    max_lines: int = 80,
    max_chars: int = 12000,
    keep_recent_other: int = 12,
) -> str:
    """Filter verbose console output before sending it back to the LLM.

    Full logs are still saved as artifacts. This only trims the feedback used
    for agent-level regeneration so progress bars and per-step simulator lines
    do not crowd out tactile failure signatures.
    """
    if not text:
        return ""

    normalized = text.replace("\r", "\n")
    important: list[str] = []
    other: list[str] = []
    progress_dropped = 0
    empty_dropped = 0
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            empty_dropped += 1
            continue
        if _is_console_progress_noise(line):
            progress_dropped += 1
            continue
        if _is_console_important_for_multiturn(line):
            important.append(line)
        else:
            other.append(line)

    max_lines = max(1, int(max_lines))
    keep_recent_other = max(0, int(keep_recent_other))
    max_important = max(1, max_lines - keep_recent_other)
    kept_important = important[-max_important:]
    kept_other = other[-keep_recent_other:] if keep_recent_other else []

    omitted_important = max(0, len(important) - len(kept_important))
    omitted_other = max(0, len(other) - len(kept_other))
    header = (
        "[console-filter] "
        f"dropped_progress={progress_dropped} dropped_empty={empty_dropped} "
        f"omitted_important={omitted_important} omitted_other={omitted_other}"
    )
    sections = [header]
    if kept_important:
        sections.append("[important]")
        sections.extend(kept_important)
    if kept_other:
        sections.append("[recent_other]")
        sections.extend(kept_other)
    filtered = "\n".join(sections)

    max_chars = max(2000, int(max_chars))
    if len(filtered) > max_chars:
        filtered = (
            f"[console-filter] truncated_to_last_chars={max_chars}\n"
            + filtered[-max_chars:]
        )
    return filtered


def _clip_multiturn_code(text: str, max_chars: int, label: str) -> str:
    """Bound code history sent to the repair model.

    A regenerated block runs in the current simulator state. Replaying every
    previous full program adds noise and encourages restarting completed
    phases, so retain only the newest suffix when history is large.
    """
    text = str(text or "")
    limit = max(1000, int(max_chars))
    if len(text) <= limit:
        return text
    return (
        f"# [{label} clipped: omitted {len(text) - limit} chars of older code]\n"
        + text[-limit:]
    )


# ---------------------------------------------------------------------------
# Trial video directory helper
# ---------------------------------------------------------------------------

def _trial_video_dir(
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> str:
    """Return the trial output directory path used for artifact saving."""
    return os.path.join(
        config["output_dir"],
        f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}_reward_{reward:.3f}"
        f"_taskcompleted_{int(info_step.get('task_completed', False))}",
    )


def _save_trial_video(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    num_code_blocks: int,
    *,
    suffix_extra: str = "",
) -> None:
    """Save recorded video frames from the environment, if available."""
    if not config["record_video"] or not hasattr(env, "get_video_frames"):
        return
    frames = env.get_video_frames(clear=True)
    if not frames or not config["output_dir"]:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)
    suffix = f"{reward:.3f}"
    if suffix_extra:
        suffix += f"_{suffix_extra}"

    if isinstance(frames, list):
        _write_video(frames, base_dir, suffix=suffix)
    elif isinstance(frames, dict):
        for key, frame in frames.items():
            _write_video(frame, base_dir, suffix=f"{suffix}_{key}")


def _save_turn_and_combined_videos(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    turn_frame_ranges: list[tuple[int, int]],
) -> None:
    """Save per-turn videos and a combined video of all turns.

    Gets all frames from the environment (clearing the buffer), then writes:
      - ``video_turn_00.mp4``, ``video_turn_01.mp4``, ... for each turn
      - ``video_combined.mp4`` for the full trial
      - If wrist camera is enabled: ``video_turn_00_wrist.mp4``, etc.
    """
    if not config["record_video"] or not config["output_dir"]:
        return
    if not hasattr(env, "get_video_frames"):
        return

    all_frames = env.get_video_frames(clear=True)
    if not all_frames:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)

    # all_frames may be a list (Robosuite) or a dict of lists (R1Pro multi-camera).
    # Normalise to a list for slicing; dict case is handled by _write_multi_video.
    if isinstance(all_frames, dict):
        # Multi-camera: write each camera stream as a combined video
        for key, frames in all_frames.items():
            if frames:
                _write_video(frames, base_dir, suffix=f"combined_{key}")
        return

    # Per-turn videos
    for i, (start, end) in enumerate(turn_frame_ranges):
        turn_frames = all_frames[start:end]
        if turn_frames:
            _write_video(turn_frames, base_dir, suffix=f"turn_{i:02d}")

    # Combined video
    _write_video(all_frames, base_dir, suffix="combined")

    # Wrist camera videos
    if config.get("use_wrist_camera") and hasattr(env, "get_wrist_video_frames"):
        wrist_frames = env.get_wrist_video_frames(clear=True)
        if wrist_frames:
            for i, (start, end) in enumerate(turn_frame_ranges):
                wrist_turn = wrist_frames[start:end]
                if wrist_turn:
                    _write_video(wrist_turn, base_dir, suffix=f"turn_{i:02d}_wrist")
            _write_video(wrist_frames, base_dir, suffix="combined_wrist")


def _save_tactile_artifacts(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> None:
    """Save tactile timeline and overlay artifacts when TactileMemoryApi is active."""
    if not config.get("output_dir"):
        return

    tactile_api = getattr(env, "_apis", {}).get("TactileMemoryApi")
    if tactile_api is None or not hasattr(tactile_api, "export_tactile_frames"):
        return

    try:
        records = tactile_api.export_tactile_frames()
        if not records:
            return

        from capx.integrations.tactile.visualization import save_tactile_artifacts

        trial_dir = _trial_video_dir(config, trial, info_step, reward)
        video_candidates = [
            os.path.join(trial_dir, "videos", "video_combined.mp4"),
            os.path.join(trial_dir, "videos", f"video_{reward:.3f}.mp4"),
        ]
        video_path = next((path for path in video_candidates if os.path.exists(path)), None)
        save_tactile_artifacts(records, trial_dir, target="cubeA", video_path=video_path)
        _save_tactile_strategy_memory(config, trial_dir)
    except Exception as exc:
        print(f"WARNING: Failed to save tactile visualization artifacts: {exc}")


def _save_tactile_strategy_memory(config: dict[str, Any], trial_dir: str) -> None:
    """Append a tactile strategy memory record for a completed trial."""
    if not config.get("tactile_strategy_memory", False):
        return
    try:
        from capx.integrations.tactile.strategy_memory import append_trial_strategy_record

        record = append_trial_strategy_record(
            trial_dir,
            memory_path=config.get("tactile_strategy_memory_path"),
            task=config.get("task_name", "cube_stack"),
            target="red cube",
        )
        if record is None:
            print("[tactile-memory] No new strategy record saved")
        else:
            print(
                "[tactile-memory] "
                f"Saved {record.outcome} strategy {record.failure_type} "
                f"reward={record.reward:.3f}"
            )
    except Exception as exc:
        print(f"WARNING: Failed to save tactile strategy memory: {exc}")


def _save_env_debug_artifacts(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> None:
    """Save private simulator diagnostics beside trial artifacts."""
    if not config.get("output_dir"):
        return
    low_level = getattr(env, "low_level_env", None)
    export_fn = getattr(low_level, "export_debug_artifacts", None)
    if not callable(export_fn):
        return
    try:
        trial_dir = _trial_video_dir(config, trial, info_step, reward)
        export_fn(trial_dir)
    except Exception as exc:
        print(f"WARNING: Failed to save environment debug artifacts: {exc}")


def _save_tactile_code_memory_trace(
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    trace: list[dict[str, Any]],
) -> None:
    """Save per-trial tactile code-memory retrieval/debug trace."""
    memory_cfg = dict(config.get("tactile_code_memory", {}))
    if not memory_cfg.get("enabled", False) or not memory_cfg.get("save_trace", True):
        return
    if not config.get("output_dir"):
        return
    trial_dir = _trial_video_dir(config, trial, info_step, reward)
    try:
        os.makedirs(trial_dir, exist_ok=True)
        path = os.path.join(trial_dir, "tactile_code_memory_trace.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, sort_keys=True)
    except Exception as exc:
        print(f"WARNING: Failed to save tactile code memory trace: {exc}")


def _build_tactile_code_memory_runtime_context(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    *,
    repair_turn_count: int,
    trace: list[dict[str, Any]],
) -> str | None:
    """Retrieve runtime tactile code memories for multi-turn repair."""
    memory_cfg = dict(config.get("tactile_code_memory", {}))
    if not memory_cfg.get("enabled", False):
        return None
    if not memory_cfg.get("runtime_retrieval", True):
        return None
    if repair_turn_count >= int(memory_cfg.get("max_repair_turns", 2)):
        return None

    try:
        from capx.memory.tactile_code import (
            TactileCodeMemoryBank,
            build_current_signature_from_env,
            format_runtime_memory_for_prompt,
        )

        signature = build_current_signature_from_env(
            env,
            window=int(memory_cfg.get("window", 20)),
        )
        if signature is None:
            return None
        required_apis = list(getattr(getattr(env, "cfg", None), "apis", []))
        records = TactileCodeMemoryBank(memory_cfg.get("path")).retrieve(
            capability=str(memory_cfg.get("capability", "grasp_stabilization")),
            current_signature=signature,
            required_apis=required_apis,
            top_k=int(memory_cfg.get("top_k_runtime", 2)),
            statuses=list(memory_cfg.get("statuses", ["validated"])),
            include_scores=True,
        )
        entry = {
            "kind": "runtime_retrieval",
            "repair_turn_count": repair_turn_count,
            "signature": signature,
            "matches": [
                {
                    "id": record.get("id"),
                    "score": record.get("_score"),
                    "diagnosis": record.get("diagnosis"),
                    "status": record.get("status"),
                }
                for record in records
            ],
            "prompt_injected": bool(records),
        }
        trace.append(entry)
        print(
            "[tactile-code-memory] runtime retrieval "
            f"matches={len(records)} repair_turn={repair_turn_count}",
            flush=True,
        )
        if not records:
            return None
        return format_runtime_memory_for_prompt(signature, records)
    except Exception as exc:
        trace.append({"kind": "runtime_retrieval_error", "error": repr(exc)})
        print(f"WARNING: tactile code memory runtime retrieval failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Visual feedback and image differencing
# ---------------------------------------------------------------------------

def _capture_initial_visual_feedback(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    config: dict[str, Any],
    args: LaunchArgs,
    visual_differencing_args: ModelQueryArgs,
) -> tuple[list, list[str], str]:
    """Capture the initial environment image and optionally describe it.

    Returns:
        (visual_feedback_imgs, visual_feedback_base64_history, task_description)
    """
    visual_feedback_imgs: list = []
    visual_feedback_base64_history: list[str] = []
    task_description = ""

    use_wrist = config.get("use_wrist_camera", False)

    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
        or config.get("use_video_differencing", False)
    )
    if not (needs_visual and hasattr(env, "render")):
        return visual_feedback_imgs, visual_feedback_base64_history, task_description

    initial_base64, initial_img = _get_visual_feedback(env)
    visual_feedback_imgs.append(initial_img)
    visual_feedback_base64_history.append(initial_base64)
    task_description = copy.deepcopy(obs["full_prompt"][-1]["content"][0]["text"])

    # Also capture wrist camera image for multiview initial description
    initial_wrist_base64 = None
    if use_wrist and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            initial_wrist_base64 = (
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )
            visual_feedback_imgs.append(pil_wrist)

    # Append image to the prompt for VLM visual feedback
    if config["use_visual_feedback"]:
        obs["full_prompt"][-1]["content"][0]["text"] += (
            "\n\nIncluded below is an image of the initial state of the environment."
        )
        obs["full_prompt"][-1]["content"].append(
            {"type": "image_url", "image_url": {"url": initial_base64}}
        )
        if initial_wrist_base64 is not None:
            obs["full_prompt"][-1]["content"].append(
                {
                    "type": "text",
                    "text": "Included below is an image from the robot's wrist camera.",
                }
            )
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_wrist_base64}}
            )

    # Image differencing: ask a VLM to describe the initial scene
    if config["use_img_differencing"] or config.get("use_video_differencing", False):
        description = _describe_initial_scene(
            visual_differencing_args, task_description, initial_base64,
            wrist_image_base64=initial_wrist_base64,
        )
        feedback = f"The initial state of the environment is described as follows:\n{description}"
        obs["full_prompt"][-1]["content"][0]["text"] += f"\n\n{feedback}"
        if args.debug:
            print(description)

    return visual_feedback_imgs, visual_feedback_base64_history, task_description


def _describe_initial_scene(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    image_base64: str,
    wrist_image_base64: str | None = None,
) -> str:
    """Query a VLM to describe the initial environment state."""
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the initial state of the environment with the goal of the "
                "task in mind. You should try to provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera view:"},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
    if wrist_image_base64 is not None:
        user_content.extend([
            {"type": "text", "text": "Wrist camera view:"},
            {"type": "image_url", "image_url": {"url": wrist_image_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the initial state of the "
                "environment with the goal of the task in mind. You should try to provide "
                "objective information and no assumptions. Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


def _get_visual_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    visual_feedback_base64_history: list[str],
    wrist_base64_history: list[str] | None = None,
) -> str | None:
    """Query a VLM to describe what changed between the two most recent frames.

    Args:
        wrist_base64_history: Optional history of wrist camera images.  When provided
            and has >=2 entries, the before/after wrist images are included in the prompt.
    """
    if len(visual_feedback_base64_history) < 2:
        return None

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the difference between the current state of the "
                "environment and the previous state of the environment with the "
                "goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code.."
            ),
        },
        {"type": "text", "text": "Previous state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-2]}},
        {"type": "text", "text": "Current state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-1]}},
    ]

    if wrist_base64_history and len(wrist_base64_history) >= 2:
        user_content.extend([
            {"type": "text", "text": "Previous state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-2]}},
            {"type": "text", "text": "Current state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-1]}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the difference between the "
                "current state of the environment and the previous state of the environment "
                "with the goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Video differencing
# ---------------------------------------------------------------------------

def _get_video_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    turn_frames: list[np.ndarray],
    wrist_turn_frames: list[np.ndarray] | None = None,
) -> str | None:
    """Query a VLM with a video of the turn execution to describe what happened.

    Args:
        visual_differencing_args: Model query args for the VDM model.
        task_description: The task goal.
        turn_frames: RGB frames from the main camera for this turn.
        wrist_turn_frames: RGB frames from the wrist camera for this turn (optional).

    Returns:
        Text description of the execution, or None if no frames.
    """
    if not turn_frames:
        return None

    video_base64 = _encode_video_base64(turn_frames)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "The following video shows the robot executing code in the "
                "environment from the main camera view. Describe what happened "
                "during execution, including what actions the robot took, how "
                "the objects in the scene changed, and whether the task appears "
                "to have been completed. Provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera video:"},
        {"type": "image_url", "image_url": {"url": video_base64}},
    ]

    if wrist_turn_frames:
        wrist_video_base64 = _encode_video_base64(wrist_turn_frames)
        user_content.extend([
            {
                "type": "text",
                "text": (
                    "The following video shows the same execution from the "
                    "robot's wrist-mounted camera (eye-in-hand view), providing "
                    "a close-up perspective of the gripper and objects being "
                    "manipulated."
                ),
            },
            {"type": "text", "text": "Wrist camera video:"},
            {"type": "image_url", "image_url": {"url": wrist_video_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that analyzes robot execution "
                "videos. You describe what happened during the robot's code "
                "execution, what actions were taken, how the environment "
                "changed, and whether the task appears to have been completed. "
                "Provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return _query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Initial code generation
# ---------------------------------------------------------------------------

def _query_initial_code(
    args: LaunchArgs,
    config: dict[str, Any],
    obs: dict[str, Any],
) -> tuple[str, str | None, dict | None]:
    """Query the model for the initial code generation.

    Returns:
        (raw_code, reasoning, ensemble_data)
    """
    ensemble_data = None
    with _suspend_sigalrm():
        if config["use_parallel_ensemble"]:
            if config.get("use_multimodel", False):
                print("RUNNING MULTIMODEL ENSEMBLE QUERY")
                out = _query_model_ensemble(args, obs["full_prompt"], is_multiturn=False)
            else:
                print("RUNNING SINGLE MODEL ENSEMBLE QUERY")
                out = _query_single_model_ensemble(args, obs["full_prompt"], args.model, is_multiturn=False)
            ensemble_data = {
                "ensemble_candidates_txt": out["ensemble_candidates_txt"],
                "ensemble_synthesis_txt": out["ensemble_synthesis_txt"],
            }
        else:
            out = _query_model(args, obs["full_prompt"])

    return out["content"], out["reasoning"], ensemble_data


class _suspend_sigalrm:
    """Temporarily pause trial SIGALRM while waiting on remote LLM calls."""

    def __enter__(self):
        self.remaining = signal.alarm(0)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.remaining > 0:
            signal.alarm(self.remaining)
        return False


# ---------------------------------------------------------------------------
# Multi-turn decision handling
# ---------------------------------------------------------------------------

def _handle_multi_turn_step(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    args: LaunchArgs,
    config: dict[str, Any],
    visual_differencing_args: ModelQueryArgs,
    multi_turn_prompt: str,
    code_blocks: list[str],
    code_block_idx: int,
    info_step: dict[str, Any],
    task_description: str,
    visual_feedback_imgs: list,
    visual_feedback_base64_history: list[str],
    stderr_history: list[str],
    turn_frames: list[np.ndarray] | None = None,
    wrist_turn_frames: list[np.ndarray] | None = None,
    wrist_base64_history: list[str] | None = None,
    tactile_code_memory_trace: list[dict[str, Any]] | None = None,
    repair_turn_count: int = 0,
) -> tuple[str, str | None, str | None, dict | None, list | None]:
    """Execute one multi-turn decision step.

    Captures visual feedback, builds the decision prompt, queries the model,
    and returns the parsed decision.

    Args:
        turn_frames: Frames from the main camera for this turn (for video differencing).
        wrist_turn_frames: Frames from the wrist camera for this turn (for video differencing).
        wrist_base64_history: History of wrist camera base64 images for image-based
            differencing with multiview.

    Returns:
        (decision, new_code, reasoning, multiturn_ensemble_entry)
        where decision is "regenerate", "finish", or "continue".
    """
    use_wrist = config.get("use_wrist_camera", False)

    executed_code = "\n".join(code_blocks[:code_block_idx])
    remaining_code = "\n\n".join(code_blocks[code_block_idx:])
    executed_code = _clip_multiturn_code(
        executed_code,
        int(config.get("multiturn_executed_code_max_chars", 12000)),
        "executed code history",
    )
    remaining_code = _clip_multiturn_code(
        remaining_code,
        int(config.get("multiturn_remaining_code_max_chars", 8000)),
        "remaining code",
    )
    console_stdout = info_step["stdout"]
    console_stderr = info_step["stderr"]
    if config.get("filter_multiturn_console", False):
        console_stdout = _filter_console_for_multiturn(
            console_stdout,
            max_lines=int(config.get("multiturn_console_max_lines", 80)),
            max_chars=int(config.get("multiturn_console_max_chars", 12000)),
            keep_recent_other=int(config.get("multiturn_console_keep_recent_other", 12)),
        )
        console_stderr = _filter_console_for_multiturn(
            console_stderr,
            max_lines=int(config.get("multiturn_console_max_lines", 80)),
            max_chars=int(config.get("multiturn_console_max_chars", 12000)),
            keep_recent_other=int(config.get("multiturn_console_keep_recent_other", 12)),
        )
    complete_multi_turn_prompt = multi_turn_prompt.format(
        executed_code=executed_code,
        console_stdout=console_stdout,
        console_stderr=console_stderr,
    )
    complete_multi_turn_prompt += (
        "\n\nThe simulator is already at the state reached by the executed block. "
        "For REGENERATE, output only a short recovery/continuation block; do "
        "not replay the whole task or repeat completed object stages."
    )
    if remaining_code.strip():
        complete_multi_turn_prompt = (
            f"{complete_multi_turn_prompt}\n\n"
            "Remaining existing code blocks. Answer CONTINUE to run the next "
            "one unchanged, or REGENERATE to replace all remaining code:\n"
            f"```python\n{remaining_code}\n```"
        )
    if tactile_code_memory_trace is not None:
        memory_context = _build_tactile_code_memory_runtime_context(
            env,
            config,
            repair_turn_count=repair_turn_count,
            trace=tactile_code_memory_trace,
        )
        if memory_context:
            complete_multi_turn_prompt = f"{complete_multi_turn_prompt}\n\n{memory_context}"

    if info_step["stderr"] != "":
        stderr_history.append(info_step["stderr"])

    # Capture visual feedback if applicable
    visual_feedback_base64 = None
    needs_visual = (
        (config["use_visual_feedback"] and args.model in VLM_MODELS)
        or (config["use_img_differencing"] and visual_differencing_args.model in VLM_MODELS)
    )
    if needs_visual and hasattr(env, "render"):
        vf_base64, vf_img = _get_visual_feedback(env)
        visual_feedback_imgs.append(vf_img)
        visual_feedback_base64_history.append(vf_base64)

        # Also capture wrist camera snapshot for image-based multiview
        if use_wrist and hasattr(env, "render_wrist") and wrist_base64_history is not None:
            wrist_result = _get_visual_feedback(env, use_wrist_camera=True)
            if wrist_result[0] is not None and isinstance(wrist_result[0], list) and len(wrist_result[0]) > 1:
                wrist_base64_history.append(wrist_result[0][1])  # index 1 = wrist image

    # Determine differencing feedback
    differencing_feedback = None
    is_video_feedback = False

    if config.get("use_video_differencing") and turn_frames:
        # Video-based differencing: pass video of this turn to VDM
        differencing_feedback = _get_video_differencing_feedback(
            visual_differencing_args, task_description, turn_frames, wrist_turn_frames,
        )
        is_video_feedback = True
    elif config["use_img_differencing"] and len(visual_feedback_base64_history) >= 2:
        # Image-based differencing: pass before/after images to VDM
        differencing_feedback = _get_visual_differencing_feedback(
            visual_differencing_args, task_description, visual_feedback_base64_history,
            wrist_base64_history=wrist_base64_history,
        )

    # Only pass visual feedback to prompt if visual_feedback is enabled
    if not config["use_visual_feedback"]:
        visual_feedback_base64 = None
    elif needs_visual and hasattr(env, "render"):
        visual_feedback_base64 = visual_feedback_base64_history[-1] if visual_feedback_base64_history else None

    # Build decision prompt
    if args.use_legacy_multi_turn_decision_prompt:
        print("Using legacy multi-turn decision prompt")
        decision_prompt = _build_multi_turn_decision_prompt_legacy(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )
    else:
        decision_prompt = _build_multi_turn_decision_prompt(
            obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
            is_video_feedback=is_video_feedback,
        )

    # Query model
    print(
        f"[capx-trial] multi-turn decision query begin after_block={code_block_idx}",
        flush=True,
    )
    multiturn_ensemble_entry = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTITURN MULTIMODEL ENSEMBLE QUERY")
            content = _query_model_ensemble(args, decision_prompt, is_multiturn=True)
        else:
            print("RUNNING MULTITURN SINGLE MODEL ENSEMBLE QUERY")
            content = _query_single_model_ensemble(args, decision_prompt, args.model, is_multiturn=True)
        multiturn_ensemble_entry = {
            "ensemble_candidates_txt": content.get("ensemble_candidates_txt", ""),
            "ensemble_synthesis_txt": content.get("ensemble_synthesis_txt", ""),
        }
    else:
        content = _query_model(args, decision_prompt)

    reasoning = content["reasoning"]
    decision, new_code = _parse_multi_turn_decision(content["content"])
    print(
        f"[capx-trial] multi-turn decision query end decision={decision}",
        flush=True,
    )
    if tactile_code_memory_trace:
        latest = tactile_code_memory_trace[-1]
        if latest.get("kind") == "runtime_retrieval":
            latest["decision"] = decision
            if new_code and decision == "regenerate":
                latest["generated_repair_code"] = new_code

    return decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt


# ---------------------------------------------------------------------------
# Core single-trial execution
# ---------------------------------------------------------------------------

def _run_single_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args: LaunchArgs,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    partial_artifacts: dict[str, Any] | None = None,
) -> TrialSummary:
    """Execute a single trial end-to-end.

    Steps:
        1. Reset the environment.
        2. Capture initial visual feedback (if configured).
        3. Query the model for initial code generation.
        4. Execute code blocks one-by-one, with optional multi-turn regeneration.
        5. Save artifacts (code, logs, per-turn videos, combined video) and return a TrialSummary.
    """
    trial_start_time = time.time()
    print(f"[capx-trial] trial={trial} begin", flush=True)

    use_video_diff = config.get("use_video_differencing", False)
    use_wrist = config.get("use_wrist_camera", False)
    should_record = config["record_video"] or use_video_diff
    record_during_reset = should_record and bool(
        getattr(env, "record_video_during_reset", False)
    )

    # --- 1. Reset environment ---
    print(f"[capx-trial] trial={trial} reset begin", flush=True)
    if record_during_reset and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(
            True,
            clear=True,
            wrist_camera=use_wrist,
            capture_initial_frame=False,
        )
    obs, _ = env.reset(options={"trial": trial}, seed=trial)
    print(f"[capx-trial] trial={trial} reset end", flush=True)
    # Reset the SIGALRM timer AFTER env.reset() so the timeout only covers
    # actual task execution, not scene loading / cuRobo JIT compilation.
    remaining = signal.alarm(0)  # cancel current alarm
    if remaining > 0:
        timeout_seconds = int(config.get("trial_timeout_seconds", remaining))
        signal.alarm(max(1, timeout_seconds))
    obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
    _patch_libero_goal(env, obs)

    if should_record and hasattr(env, "enable_video_capture") and not record_during_reset:
        # Video differencing needs frame recording even without record_video
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)

    # --- Shared trial state ---
    code_blocks: list[str] = []
    code_block_metadata: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    stderr_history: list[str] = []
    num_regenerations = 0
    num_finishes = 0
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    sandbox_rc_override = None
    ensemble_data = None
    multiturn_ensemble_data: list[dict[str, Any]] = []
    tactile_code_memory_trace: list[dict[str, Any]] = []

    # Per-turn frame tracking (for video differencing and per-turn video saving)
    turn_frame_ranges: list[tuple[int, int]] = []

    # Wrist camera base64 history for image-based multiview differencing
    wrist_base64_history: list[str] | None = [] if use_wrist else None

    visual_differencing_args = ModelQueryArgs(
        model=args.visual_differencing_model,
        server_url=args.visual_differencing_model_server_url,
        api_key=args.visual_differencing_model_api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )

    if config["use_img_differencing"] or use_video_diff:
        assert visual_differencing_args.model in VLM_MODELS, (
            "Image/video differencing model must be in the list of VLM models"
        )

    # --- 2. Capture initial visual feedback ---
    print(f"[capx-trial] trial={trial} initial visual begin", flush=True)
    visual_feedback_imgs, visual_feedback_base64_history, task_description = (
        _capture_initial_visual_feedback(env, obs, config, args, visual_differencing_args)
    )
    print(f"[capx-trial] trial={trial} initial visual end", flush=True)

    # Seed wrist base64 history with initial wrist image
    if use_wrist and wrist_base64_history is not None and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            wrist_base64_history.append(
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )

    # --- 3. Initial code generation ---
    if config["use_oracle_code"]:
        raw_code = env.oracle_code
        with open(os.path.join(config["output_dir"], "oracle_code.py"), "w") as f:
            f.write(raw_code)
        reasoning = None
        ensemble_data = None
    else:
        print(f"[capx-trial] trial={trial} initial code query begin", flush=True)
        raw_code, reasoning, ensemble_data = _query_initial_code(args, config, obs)
        print(f"[capx-trial] trial={trial} initial code query end", flush=True)

    # Initialize partial artifacts for timeout recovery
    if partial_artifacts is not None:
        partial_artifacts.update({
            "raw_code": raw_code,
            "code_blocks": code_blocks,
            "code_block_metadata": code_block_metadata,
            "all_responses": all_responses,
            "visual_feedback_imgs": visual_feedback_imgs,
            "info_step": info_step,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "num_regenerations": num_regenerations,
            "num_finishes": num_finishes,
            "num_code_blocks": 0,
            "ensemble_data": ensemble_data,
            "multiturn_ensemble_data": multiturn_ensemble_data,
            "tactile_code_memory_trace": tactile_code_memory_trace,
        })

    # Parse initial code into blocks
    initial_blocks = _extract_configured_code_blocks(raw_code, config)
    code_blocks.extend(initial_blocks)
    code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))
    all_responses.append({
        "block_idx": [0],
        "code_blocks": initial_blocks,
        "decision": "initial",
        "initial_prompt": copy.deepcopy(obs["full_prompt"]),
        "reasoning": reasoning if reasoning is not None else "",
    })
    if config.get("save_in_progress_code", True):
        _save_in_progress_trial_artifacts(
            config,
            trial,
            final_code=_annotate_code_blocks(code_blocks, code_block_metadata),
            raw_code=raw_code,
            all_responses=all_responses,
        )

    # --- 4. Execute code blocks (with optional multi-turn) ---
    info_step = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    code_block_idx = 0

    # Track whether we're recording frames (for video diff or record_video)
    recording_frames = (
        (config["record_video"] or use_video_diff)
        and hasattr(env, "get_video_frame_count")
    )

    while code_block_idx < len(code_blocks) and code_block_idx <= MULTITURN_LIMIT:
        code = code_blocks[code_block_idx]
        code_block_idx += 1

        # Record frame index before step
        frame_start = env.get_video_frame_count() if recording_frames else 0

        obs_next, reward, terminated, truncated, info_step = env.step(code)

        # Record frame index after step
        frame_end = env.get_video_frame_count() if recording_frames else 0
        turn_frame_ranges.append((frame_start, frame_end))

        if partial_artifacts is not None:
            partial_artifacts.update({
                "info_step": info_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            })

        obs = obs_next

        if terminated or bool(info_step.get("task_completed", False)):
            print(
                "[capx-trial] task completed after code block; stopping further blocks",
                flush=True,
            )
            break

        # Multi-turn decision
        if multi_turn_prompt:
            if "terminated episode" in info_step["stderr"]:
                truncated = True
                break

            # Get turn frames for video differencing
            turn_frames = None
            wrist_turn_frames = None
            if use_video_diff and recording_frames:
                turn_frames = env.get_video_frames_range(frame_start, frame_end)
                if use_wrist and hasattr(env, "get_wrist_video_frames_range"):
                    wrist_turn_frames = env.get_wrist_video_frames_range(
                        frame_start, frame_end,
                    )

            decision, new_code, mt_reasoning, mt_ensemble, decision_prompt = _handle_multi_turn_step(
                env, obs, args, config, visual_differencing_args,
                multi_turn_prompt, code_blocks, code_block_idx, info_step,
                task_description, visual_feedback_imgs, visual_feedback_base64_history,
                stderr_history,
                turn_frames=turn_frames,
                wrist_turn_frames=wrist_turn_frames,
                wrist_base64_history=wrist_base64_history,
                tactile_code_memory_trace=tactile_code_memory_trace,
                repair_turn_count=num_regenerations,
            )

            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)

            if decision == "regenerate":
                max_regenerations = int(config.get("max_regenerations", MULTITURN_LIMIT))
                if num_regenerations >= max_regenerations:
                    all_responses.append({
                        "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                        "block_idx": [code_block_idx],
                        "code_blocks": [],
                        "decision": "regenerate_skipped_max",
                        "reasoning": mt_reasoning if mt_reasoning is not None else "",
                    })
                    print(
                        "Model chose to regenerate code, but max_regenerations "
                        f"({max_regenerations}) was reached"
                    )
                    break
                print("Model chose to regenerate code")
                new_blocks = _extract_configured_code_blocks(new_code, config)
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [code_block_idx],
                    "code_blocks": new_blocks,
                    "decision": "regenerate",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                })
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                code_blocks.extend(new_blocks)
                code_block_metadata.extend(
                    [{"generation": num_regenerations + 1, "regenerated": True,
                      "regenerated_at_idx": code_block_idx}]
                    * len(new_blocks)
                )
                num_regenerations += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_regenerations"] = num_regenerations

            elif decision == "finish":
                all_responses.append({
                    "decision": "finish",
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                })
                print("Model chose to finish")
                num_finishes += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_finishes"] = num_finishes
                break
            elif decision == "continue":
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [code_block_idx],
                    "code_blocks": [],
                    "decision": "continue",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                })
                print("Model chose to continue")

        print(f"Code block {code_block_idx} done")
        print(f"Number of code blocks: {len(code_blocks)}")

        # Save intermediate artifacts (code, logs) per code block
        final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
        _save_trial_artifacts(
            config, trial, info_step["sandbox_rc"], reward,
            info_step.get("task_completed", False), final_code, raw_code,
            all_responses, ["-" * 100, "Generated program:", final_code],
            visual_feedback_imgs,
        )

        # Only save intermediate video if NOT doing per-turn saving
        # (per-turn saving is deferred to after the loop to avoid clearing the buffer)
        if not recording_frames:
            _save_trial_video(
                env, config, trial, info_step, reward, len(code_blocks),
                suffix_extra=str(len(code_blocks)),
            )

    print("Code blocks done")

    # --- 5. Build final summary ---
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
    num_code_blocks = len(code_blocks)

    if partial_artifacts is not None:
        partial_artifacts["final_code"] = final_code
        partial_artifacts["num_code_blocks"] = num_code_blocks

    # Override sandbox_rc for terminated-episode stderr
    if "executing action in terminated episode" in info_step["stderr"]:
        sandbox_rc_override = 0
    if sandbox_rc_override is not None:
        info_step["sandbox_rc"] = sandbox_rc_override

    stderr = "\n\n".join(stderr_history) if stderr_history else info_step["stderr"]
    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        stderr_override=stderr,
    )

    code_path = _save_trial_artifacts(
        config, trial, info_step["sandbox_rc"], reward,
        info_step.get("task_completed", False), final_code, raw_code,
        all_responses, log_lines, visual_feedback_imgs,
        ensemble_data=ensemble_data,
        multiturn_ensemble_data=multiturn_ensemble_data,
    )

    # Save per-turn and combined videos
    if recording_frames and turn_frame_ranges:
        _save_turn_and_combined_videos(
            env, config, trial, info_step, reward, turn_frame_ranges,
        )
    else:
        _save_trial_video(env, config, trial, info_step, reward, num_code_blocks)
    _save_tactile_artifacts(env, config, trial, info_step, reward)
    _save_env_debug_artifacts(env, config, trial, info_step, reward)
    for entry in tactile_code_memory_trace:
        if entry.get("kind") == "runtime_retrieval":
            entry["final_reward"] = reward
            entry["task_completed"] = bool(info_step.get("task_completed", False))
            entry["sandbox_rc"] = int(info_step.get("sandbox_rc", 1))
    _save_tactile_code_memory_trace(
        config,
        trial,
        info_step,
        reward,
        tactile_code_memory_trace,
    )

    success = info_step["sandbox_rc"] == 0

    # --- Evolving skill library integration (opt-in) ---
    if config.get("evolve_skill_library", False) and info_step.get("task_completed", False):
        try:
            from capx.skills import SkillLibrary

            skill_lib_path = config.get("skill_library_path", None)
            skill_lib = SkillLibrary(path=skill_lib_path)
            task_name = config.get("task_name", f"trial_{trial}")
            new_skills = skill_lib.extract_from_code(final_code, task_name=task_name)
            skill_lib.save()
            if new_skills:
                print(f"[SkillLibrary] Extracted {len(new_skills)} new skill(s): {new_skills}")
        except Exception as exc:
            print(f"[SkillLibrary] Skill extraction failed: {exc}")

    print(f"Trial {trial} took {time.time() - trial_start_time:.2f} seconds")

    gc.collect()

    return TrialSummary(
        trial=trial,
        success=success,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=info_step["sandbox_rc"],
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", None),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )


def _patch_libero_goal(env: CodeExecutionEnvBase, obs: dict[str, Any]) -> None:
    """Inject the LIBERO task language into the prompt template if applicable."""
    if not hasattr(env.low_level_env, "handle"):
        return
    handle = env.low_level_env.handle
    if (
        hasattr(handle, "task_language")
        and "libero_environment_goal" in obs["full_prompt"][-1]["content"][0]["text"]
    ):
        goal = getattr(handle, "task_language")
        obs["full_prompt"][-1]["content"][0]["text"] = (
            obs["full_prompt"][-1]["content"][0]["text"].format(
                libero_environment_goal=goal
            )
        )
