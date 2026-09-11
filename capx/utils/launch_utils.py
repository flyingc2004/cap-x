"""Utility functions for launch.py."""

from __future__ import annotations

import base64
import copy
import io
import json
import logging
import multiprocessing
import os
import re
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from capx.envs.configs.instantiate import instantiate
from capx.envs.configs.loader import DictLoader

# Re-export LLM client symbols for backward compatibility
from capx.llm.client import (  # noqa: F401
    CLAUDE_MODELS,
    ENSEMBLE_CONFIGS,
    GPT_MODELS,
    OPENROUTER_MODELS,
    OPENROUTER_SERVER_URL,
    OSS_MODELS,
    VLM_MODELS,
    ModelQueryArgs,
    _completions_to_responses_convert_prompt,
    collapse_text_image_inputs,
    is_openrouter_model,
    query_model as _query_model,
    query_model_streaming as _query_model_streaming,
    query_model_ensemble as _query_model_ensemble,
    query_single_model_ensemble as _query_single_model_ensemble,
)

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs

multiprocessing.set_start_method("spawn", force=True)


@dataclass
class TrialSummary:
    """Summary of a single trial execution."""

    trial: int
    success: bool
    reward: float
    terminated: bool
    truncated: bool
    sandbox_rc: int
    log: str
    task_completed: bool | None = None
    code_path: str | None = None
    num_regenerations: int = 0
    num_finishes: int = 0
    num_code_blocks: int = 0


def run_server_proc(api_cfg) -> multiprocessing.Process:
    # Make sure we use spawn for CUDA
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(
        target=instantiate,  # child will call main(**cfg) via Hydra-style instantiate
        args=(api_cfg,),
        daemon=True,
    )
    proc.start()
    return proc


def _load_config(args: LaunchArgs) -> tuple[Any, dict[str, Any], list]:
    """Load YAML configuration and merge with command-line arguments.

    Args:
        args: Command-line arguments

    Returns:
        Tuple of (env_factory, merged_config_dict)
        - env_factory: Dict with _target_ for instantiating the environment
        - merged_config_dict: Execution config with CLI overrides applied
    """
    config_path = os.path.expanduser(args.config_path)
    configs_dict = DictLoader.load([config_path])

    # Extract environment factory (don't instantiate yet - that happens per worker)
    if "env" not in configs_dict:
        raise ValueError(
            f"YAML config {config_path} must contain 'env' key with environment configuration"
        )
    env_factory = configs_dict["env"]

    api_servers = configs_dict.get("api_servers", None)

    # Merge server URLs and VDM model settings from YAML into args
    # (CLI args take priority; YAML fills in when CLI uses defaults)
    _CLI_DEFAULTS = {
        "server_url": "http://127.0.0.1:8110/chat/completions",
        "visual_differencing_model": "google/gemini-3.1-pro-preview",
        "visual_differencing_model_server_url": "http://127.0.0.1:8110/chat/completions",
        "visual_differencing_model_api_key": None,
    }
    for field, cli_default in _CLI_DEFAULTS.items():
        current_value = getattr(args, field, cli_default)
        if current_value == cli_default and field in configs_dict:
            setattr(args, field, configs_dict[field])

    tactile_memory_config = _normalize_tactile_memory_config(
        configs_dict.get("tactile_memory")
    )
    trial_memory_config = tactile_memory_config["trial"]
    if not tactile_memory_config["configured"]:
        trial_memory_config = {
            "enabled": bool(configs_dict.get("include_trial_memory_in_multiturn", False)),
            "include_in_multiturn": bool(
                configs_dict.get("include_trial_memory_in_multiturn", False)
            ),
            "max_context_chars": int(
                configs_dict.get("multiturn_trial_memory_max_chars", 4000)
            ),
        }
        tactile_memory_config["trial"] = trial_memory_config
    persistent_memory_enabled = tactile_memory_config["persistent"]["enabled"]

    tactile_strategy_memory_path = configs_dict.get(
        "tactile_strategy_memory_path",
        ".capx_tactile_strategies.jsonl",
    )
    tactile_strategy_memory_read_path = configs_dict.get(
        "tactile_strategy_memory_read_path",
        tactile_strategy_memory_path,
    )
    tactile_strategy_top_k = int(configs_dict.get("tactile_strategy_top_k", 3))
    tactile_strategy_memory_enabled = bool(configs_dict.get("tactile_strategy_memory", False))
    if tactile_memory_config["configured"]:
        tactile_strategy_memory_enabled = (
            tactile_strategy_memory_enabled and persistent_memory_enabled
        )
    os.environ["CAPX_TACTILE_STRATEGY_MEMORY_ENABLED"] = (
        "1" if tactile_strategy_memory_enabled else "0"
    )
    os.environ["CAPX_TACTILE_STRATEGY_MEMORY_PATH"] = str(tactile_strategy_memory_path)
    os.environ["CAPX_TACTILE_STRATEGY_MEMORY_READ_PATH"] = str(
        tactile_strategy_memory_read_path
    )
    os.environ["CAPX_TACTILE_STRATEGY_TOP_K"] = str(tactile_strategy_top_k)

    tactile_code_memory_raw = configs_dict.get("tactile_code_memory", {})
    if tactile_memory_config["configured"] and not persistent_memory_enabled:
        tactile_code_memory_raw = {"enabled": False}
    tactile_code_memory_config = _normalize_tactile_code_memory_config(tactile_code_memory_raw)

    # Build merged config dict (CLI args override YAML)
    merged_config = {
        "total_trials": args.total_trials
        if args.total_trials is not None
        else configs_dict.get("trials", 10),
        "num_workers": args.num_workers
        if args.num_workers is not None
        else configs_dict.get("num_workers", 1),
        "record_video": args.record_video
        if args.record_video is not None
        else configs_dict.get("record_video", False),
        "output_dir": args.output_dir
        if args.output_dir is not None
        else configs_dict.get("output_dir", None),
        "use_oracle_code": args.use_oracle_code
        if args.use_oracle_code is not None
        else configs_dict.get("use_oracle_code", False),
        "resume_idx": configs_dict.get("resume_idx", None),
        "use_visual_feedback": args.use_visual_feedback
        if args.use_visual_feedback is not None
        else configs_dict.get("use_visual_feedback", False),
        "use_img_differencing": args.use_img_differencing
        if args.use_img_differencing is not None
        else configs_dict.get("use_img_differencing", False),
        "use_parallel_ensemble": args.use_parallel_ensemble
        if args.use_parallel_ensemble is not None
        else configs_dict.get("use_parallel_ensemble", False),
        "use_video_differencing": args.use_video_differencing
        if args.use_video_differencing is not None
        else configs_dict.get("use_video_differencing", False),
        "use_wrist_camera": args.use_wrist_camera
        if args.use_wrist_camera is not None
        else configs_dict.get("use_wrist_camera", False),
        "use_multimodel": args.use_multimodel
        if args.use_multimodel is not None
        else configs_dict.get("use_multimodel", False),
        "web_ui": getattr(args, "web_ui", None)
        if getattr(args, "web_ui", None) is not None
        else configs_dict.get("web_ui", False),
        "web_ui_port": getattr(args, "web_ui_port", None)
        if getattr(args, "web_ui_port", None) is not None
        else configs_dict.get("web_ui_port", 8200),
        "save_multiturn_prompts": configs_dict.get("save_multiturn_prompts", False),
        "save_in_progress_code": configs_dict.get("save_in_progress_code", True),
        "split_code_blocks_on_breakpoint": configs_dict.get(
            "split_code_blocks_on_breakpoint",
            False,
        ),
        "max_code_block_actions": int(
            os.getenv(
                "CAPX_MAX_CODE_BLOCK_ACTIONS",
                configs_dict.get("max_code_block_actions", 0),
            )
        ),
        "tactile_strategy_memory": tactile_strategy_memory_enabled,
        "tactile_strategy_memory_path": tactile_strategy_memory_path,
        "tactile_strategy_memory_read_path": tactile_strategy_memory_read_path,
        "tactile_strategy_top_k": tactile_strategy_top_k,
        "tactile_code_memory": tactile_code_memory_config,
        "tactile_memory": tactile_memory_config,
        "include_trial_memory_in_multiturn": trial_memory_config[
            "enabled"
        ] and trial_memory_config["include_in_multiturn"],
        "multiturn_trial_memory_max_chars": trial_memory_config["max_context_chars"],
        "trial_timeout_seconds": float(
            os.getenv(
                "CAPX_TRIAL_TIMEOUT_SECONDS",
                configs_dict.get("trial_timeout_seconds", 1000),
            )
        ),
        "max_trial_retries": int(
            os.getenv(
                "CAPX_MAX_TRIAL_RETRIES",
                configs_dict.get("max_trial_retries", 3),
            )
        ),
        "max_regenerations": int(
            os.getenv(
                "CAPX_MAX_REGENERATIONS",
                configs_dict.get(
                    "max_regenerations",
                    tactile_code_memory_config.get("max_repair_turns", 10)
                    if tactile_code_memory_config.get("enabled", False)
                    else 10,
                ),
            )
        ),
    }

    if merged_config["tactile_strategy_memory"]:
        _inject_tactile_strategy_memory_prompt(env_factory, merged_config)
    if tactile_code_memory_config.get("enabled", False):
        _inject_tactile_code_memory_prompt(env_factory, merged_config)

    return env_factory, merged_config, api_servers


def _normalize_tactile_memory_config(raw: Any) -> dict[str, Any]:
    """Normalize the UniVTAC trial/persistent tactile-memory switch."""
    configured = isinstance(raw, dict)
    cfg = dict(raw or {}) if configured else {}
    trial_raw = cfg.get("trial", {})
    persistent_raw = cfg.get("persistent", {})
    if not isinstance(trial_raw, dict):
        raise TypeError("tactile_memory.trial must be a mapping")
    if not isinstance(persistent_raw, dict):
        raise TypeError("tactile_memory.persistent must be a mapping")
    return {
        "configured": configured,
        "trial": {
            "enabled": bool(trial_raw.get("enabled", False)),
            "include_in_multiturn": bool(
                trial_raw.get("include_in_multiturn", False)
            ),
            "max_context_chars": int(trial_raw.get("max_context_chars", 4000)),
        },
        "persistent": {"enabled": bool(persistent_raw.get("enabled", False))},
    }


def _normalize_tactile_code_memory_config(raw: Any) -> dict[str, Any]:
    """Normalize the generic tactile code memory config block."""
    cfg = dict(raw or {}) if isinstance(raw, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    mode = str(cfg.get("mode", "read"))
    statuses = cfg.get("statuses")
    if statuses is None:
        statuses = ["candidate", "validated"] if mode == "candidate" else ["validated"]
    elif isinstance(statuses, str):
        statuses = [statuses]
    cfg.update(
        {
            "enabled": enabled,
            "mode": mode,
            "path": cfg.get(
                "path",
                ".capx_tactile_code_memory/lift_can_v1/bank.jsonl",
            ),
            "top_k_initial": int(cfg.get("top_k_initial", 3)),
            "top_k_runtime": int(cfg.get("top_k_runtime", 2)),
            "runtime_retrieval": bool(cfg.get("runtime_retrieval", True)),
            "max_repair_turns": int(cfg.get("max_repair_turns", 2)),
            "statuses": [str(status) for status in statuses],
            "capability": str(cfg.get("capability", "grasp_stabilization")),
            "window": int(cfg.get("window", 20)),
            "split_code_blocks_on_breakpoint": bool(
                cfg.get("split_code_blocks_on_breakpoint", enabled)
            ),
            "save_trace": bool(cfg.get("save_trace", True)),
        }
    )
    os.environ["CAPX_TACTILE_CODE_MEMORY_ENABLED"] = "1" if enabled else "0"
    os.environ["CAPX_TACTILE_CODE_MEMORY_PATH"] = str(cfg["path"])
    return cfg


def _inject_tactile_strategy_memory_prompt(
    env_factory: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Append retrieved tactile strategy memory to the task prompt."""
    try:
        from capx.integrations.tactile.strategy_memory import (
            TactileStrategyMemory,
            format_strategies_for_prompt,
        )

        records = TactileStrategyMemory(
            config.get("tactile_strategy_memory_read_path")
        ).retrieve(
            target="red cube",
            top_k=int(config.get("tactile_strategy_top_k", 3)),
        )
        prompt_addition = format_strategies_for_prompt(records)
        cfg = env_factory.get("cfg", {})
        cfg["prompt"] = f"{cfg.get('prompt', '')}\n\n{prompt_addition}"
        print(f"[tactile-memory] Injected {len(records)} strategy record(s) into prompt")
    except Exception as exc:
        print(f"WARNING: Failed to inject tactile strategy memory into prompt: {exc}")


def _inject_tactile_code_memory_prompt(
    env_factory: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Append retrieved tactile code memory to the initial task prompt."""
    memory_cfg = dict(config.get("tactile_code_memory", {}))
    try:
        from capx.memory.tactile_code import (
            TactileCodeMemoryBank,
            format_memory_for_prompt,
        )

        cfg = env_factory.get("cfg", {})
        required_apis = list(cfg.get("apis", []))
        records = TactileCodeMemoryBank(memory_cfg.get("path")).retrieve(
            capability=str(memory_cfg.get("capability", "grasp_stabilization")),
            required_apis=required_apis,
            top_k=int(memory_cfg.get("top_k_initial", 3)),
            statuses=list(memory_cfg.get("statuses", ["validated"])),
            include_scores=True,
        )
        if not records:
            print("[tactile-code-memory] No code memory records retrieved; prompt unchanged")
            return
        prompt_addition = format_memory_for_prompt(records)
        cfg["prompt"] = f"{cfg.get('prompt', '')}\n\n{prompt_addition}"
        print(
            "[tactile-code-memory] "
            f"Injected {len(records)} code memory record(s) into prompt"
        )
    except Exception as exc:
        print(f"WARNING: Failed to inject tactile code memory into prompt: {exc}")


def _extract_code(content: str) -> list[str]:
    """Extract Python code from Markdown fenced code block.

    Args:
        content: Raw model response

    Returns:
        Extracted Python code list
    """
    fenced = re.search(
        r"```(?:python|py)\s*\n(.*?)```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced is not None:
        content = fenced.group(1)

    content = _normalize_extracted_code(content)
    # NOTE: might gen empty code block at the end
    # content_list = content.split("breakpoint_code_block()")

    return [content]


def _normalize_extracted_code(content: str) -> str:
    """Normalize model-produced Python before sandbox execution."""
    content = "\n".join(
        line
        for line in content.splitlines()
        if not re.match(
            r"^\s*(?:#\s*Code block\s+\d+|```(?:python|py)?|```)\s*$",
            line,
            flags=re.IGNORECASE,
        )
    )
    return textwrap.dedent(content).strip()


def _build_multi_turn_decision_prompt_legacy(
    obs: dict[str, np.ndarray],
    complete_multi_turn_prompt: str,
    visual_feedback: str | None = None,
    visual_differencing_feedback: str | None = None,
    *,
    is_video_feedback: bool = False,
) -> list[dict]:
    """Build the multi-turn decision prompt for the model.

    Args:
        obs: The observation
        complete_multi_turn_prompt: The formatted multi-turn prompt with the executed and remaining code blocks
        visual_feedback: The base64 encoded image of the current state of the environment
        visual_differencing_feedback: Text description from VDM (image or video based)
        is_video_feedback: If True, use video-appropriate framing for the differencing feedback.

    Returns:
        The multi-turn decision prompt as a list of dictionaries
    """
    multi_turn_decision_prompt = copy.deepcopy(obs["full_prompt"])
    # TODO: I think the order of the content is important, unsure if it should be before or after the complete_multi_turn_prompt
    multi_turn_decision_prompt[-1]["content"].append(
        {"type": "text", "text": complete_multi_turn_prompt}
    )
    if visual_feedback is not None:
        multi_turn_decision_prompt[-1]["content"].append(
            {
                "type": "text",
                "text": "Included below is an image of the current state of the environment (after the code above was executed).",
            }
        )
        multi_turn_decision_prompt[-1]["content"].append(
            {"type": "image_url", "image_url": {"url": visual_feedback}}
        )
    if visual_differencing_feedback is not None:
        if is_video_feedback:
            feedback_header = (
                "Included below is a description of the robot's execution based "
                "on video observation of this turn:"
            )
        else:
            feedback_header = (
                "Included below is the observed differences between the current "
                "state of the environment (afer the code above was executed) and "
                "the previous state of the environment (before the code above was "
                "executed):"
            )
        multi_turn_decision_prompt[-1]["content"].append(
            {"type": "text", "text": f"{feedback_header}\n{visual_differencing_feedback}"}
        )
    multi_turn_decision_prompt[-1]["content"].append(
        {
            "type": "text",
            "text": (
                "Based on the code output, potential error messages, and the "
                "observation made above, carefully reason about the following:\n"
                "Please respond with EXACTLY ONE of the following:\n"
                "- The word 'FINISH' if stdout/stderr or the environment status says the task is already complete.\n"
                "- The word 'CONTINUE' if the next existing code block should run unchanged.\n"
                "- The word 'REGENERATE' followed immediately by new Python code in a fenced code block "
                "(```python...```) if you want to replace all remaining code.\n"
                "- The word 'FINISH' if the task appears to be complete or no further action is needed."
            ),
        }
    )
    # collapse the last message
    multi_turn_decision_prompt[-1]["content"] = collapse_text_image_inputs(multi_turn_decision_prompt[-1]["content"])
    return multi_turn_decision_prompt


def _build_multi_turn_decision_prompt(
    obs: dict[str, np.ndarray],
    complete_multi_turn_prompt: str,
    visual_feedback: str | None = None,
    visual_differencing_feedback: str | None = None,
    *,
    is_video_feedback: bool = False,
) -> list[dict]:
    """Build the multi-turn decision prompt for the model.

    Args:
        obs: The observation
        complete_multi_turn_prompt: The formatted multi-turn prompt with the executed and remaining code blocks
        visual_feedback: The base64 encoded image of the current state of the environment
        visual_differencing_feedback: Text description from VDM (image or video based)
        is_video_feedback: If True, use video-appropriate framing for the differencing feedback.

    Returns:
        The multi-turn decision prompt as a list of dictionaries
    """
    multi_turn_decision_prompt = copy.deepcopy(obs["full_prompt"])
    # TODO: I think the order of the content is important, unsure if it should be before or after the complete_multi_turn_prompt
    multi_turn_decision_prompt[-1]["content"].append(
        {"type": "text", "text": complete_multi_turn_prompt}
    )
    if visual_feedback is not None:
        multi_turn_decision_prompt[-1]["content"].append(
            {
                "type": "text",
                "text": "Included below is an image of the current state of the environment (after the code above was executed).",
            }
        )
        multi_turn_decision_prompt[-1]["content"].append(
            {"type": "image_url", "image_url": {"url": visual_feedback}}
        )
    if visual_differencing_feedback is not None:
        if is_video_feedback:
            feedback_header = (
                "Included below is a description of the robot's execution based "
                "on video observation of this turn:"
            )
        else:
            feedback_header = (
                "Included below is the observed differences between the current "
                "state of the environment (afer the code above was executed) and "
                "the previous state of the environment (before the code above was "
                "executed):"
            )
        multi_turn_decision_prompt[-1]["content"].append(
            {"type": "text", "text": f"{feedback_header}\n{visual_differencing_feedback}"}
        )
    multi_turn_decision_prompt[-1]["content"].append(
        {
            "type": "text",
            "text": (
                "Based on the code output, potential error messages, and the "
                "observation made above, carefully reason about the following:\n"
                "Please respond with EXACTLY ONE of the following:\n"
                "- The word 'FINISH' if stdout/stderr or the environment status says the task is already complete.\n"
                "- The word 'CONTINUE' if the next existing code block should run unchanged.\n"
                "- The word 'REGENERATE' followed immediately by new Python code in a fenced code block "
                "(```python...```) if you want to replace all remaining code.\n"
                "- The word 'FINISH' if the task appears to be complete or no further action is needed."
            ),
        }
    )
    # collapse the last message
    multi_turn_decision_prompt[-1]["content"] = collapse_text_image_inputs(multi_turn_decision_prompt[-1]["content"])
    return multi_turn_decision_prompt


def _parse_multi_turn_decision(content: str) -> tuple[str, str | None]:
    """Parses the multi-turn decision response from the model

    Args:
        content: The model's response to the multi-turn decision prompt

    Returns:
        A tuple containing the decision and the new code (if any)
    """
    if content is None:
        return "finish", None

    upper = content.upper()
    if "REGENERATE" in upper:
        match = re.search(r"\bREGENERATE\b", content, flags=re.IGNORECASE)
        payload = content[match.end():] if match is not None else content
        extracted = _extract_code(payload)[0].strip()
        return "regenerate", extracted if extracted else None
    if re.search(r"\bCONTINUE\b", upper):
        return "continue", content
    return "finish", content


def _get_visual_feedback(
    env, use_wrist_camera: bool = False,
) -> tuple[str | list[str] | None, Image.Image | list[Image.Image] | None]:
    """Get visual feedback from the environment.

    Args:
        env: The environment.
        use_wrist_camera: If True, also render from the wrist camera and return
            lists of base64/PIL images (main camera first, wrist camera second).
            When False (default), returns a single base64 string and PIL image
            for backward compatibility.

    Returns:
        When ``use_wrist_camera`` is False (default):
            ``(base64_image, pil_image)`` — a single data-URL string and PIL image,
            or ``(None, None)`` on failure.
        When ``use_wrist_camera`` is True:
            ``(base64_list, pil_list)`` — lists of data-URL strings and PIL images
            (index 0 = main camera, index 1 = wrist camera), or ``(None, None)``
            on failure.
    """
    render_img = env.render()
    if render_img is None:
        return None, None

    def _encode(img_array):
        pil_img = Image.fromarray(img_array)
        buf = io.BytesIO()
        pil_img.save(buf, format="png")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}", pil_img

    main_b64, main_pil = _encode(render_img)

    if not use_wrist_camera:
        return main_b64, main_pil

    # Multiview: collect main + wrist camera images
    b64_list = [main_b64]
    pil_list = [main_pil]

    if hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            wrist_b64, wrist_pil = _encode(wrist_img)
            b64_list.append(wrist_b64)
            pil_list.append(wrist_pil)

    return b64_list, pil_list


def _save_trial_artifacts(
    config: dict[str, Any],
    trial: int,
    sandbox_rc: int,
    reward: float,
    task_completed: bool,
    final_code: str,
    raw_code: str | None,
    all_responses: list[dict],
    log_lines: list[str],
    visual_feedback_imgs: list[Image.Image],
    ensemble_data: dict[str, str] | None = None,
    multiturn_ensemble_data: list[dict[str, str]] | None = None,
) -> str | None:
    """Save trial artifacts (code, logs, images) to the output directory.

    Returns:
        Path to the saved code file, or None if output_dir is not set.
    """
    if not config["output_dir"]:
        return None
    trial_dir = (
        Path(config["output_dir"])
        / f"trial_{trial:02d}_sandboxrc_{sandbox_rc}_reward_{reward:.3f}_taskcompleted_{int(task_completed)}"
    )
    trial_dir.mkdir(parents=True, exist_ok=True)

    code_path_obj = trial_dir / "code.py"
    code_path_obj.write_text(final_code)
    code_path = str(code_path_obj)

    if raw_code:
        (trial_dir / "raw_response.sh").write_text(raw_code)

    (trial_dir / "all_responses.json").write_text(json.dumps(all_responses, indent=2))
    (trial_dir / "summary.txt").write_text("\n".join(log_lines))
    if all_responses:
        try:
            initial_prompt_content = all_responses[0].get("initial_prompt")
            if initial_prompt_content:
                prompt_text = initial_prompt_content[-1]["content"][0]["text"]
                (trial_dir / "initial_prompt.txt").write_text(str(prompt_text))
        except Exception:
            pass

    # Save initial ensemble data if provided
    if ensemble_data:
        if ensemble_data.get("ensemble_candidates_txt"):
            (trial_dir / "ensemble_candidates.txt").write_text(ensemble_data["ensemble_candidates_txt"])
        if ensemble_data.get("ensemble_synthesis_txt"):
            (trial_dir / "ensemble_synthesis.txt").write_text(ensemble_data["ensemble_synthesis_txt"])

    # Save multi-turn ensemble data (one file per regeneration)
    if multiturn_ensemble_data:
        for entry in multiturn_ensemble_data:
            regen_num = entry.get("regeneration", 0)
            if entry.get("ensemble_candidates_txt"):
                (trial_dir / f"ensemble_candidates_regen_{regen_num:02d}.txt").write_text(
                    entry["ensemble_candidates_txt"]
                )
            if entry.get("ensemble_synthesis_txt"):
                (trial_dir / f"ensemble_synthesis_regen_{regen_num:02d}.txt").write_text(
                    entry["ensemble_synthesis_txt"]
                )

    i = 0
    (trial_dir / "prompts_and_responses").mkdir(parents=True, exist_ok=True)
    for response in all_responses:
        if "task_seg_description" in response:
            (trial_dir / "prompts_and_responses" / "task_seg_description.txt").write_text(response["task_seg_description"])
        if "task_seg_prompt" in response:
            (trial_dir / "prompts_and_responses" / "task_seg_prompt.txt").write_text(str(response["task_seg_prompt"]))
        if "initial_prompt" in response:
            try:
                initial_prompt_content = response["initial_prompt"][-1]["content"][0]["text"]
                if isinstance(initial_prompt_content, list):
                    initial_prompt_content = "\n".join(map(str, initial_prompt_content))
                (trial_dir / "prompts_and_responses" / "initial_prompt.txt").write_text(str(initial_prompt_content))
            except Exception as e:
                print(f"Error saving initial_prompt: {e}")
        if "multi_turn_prompt" in response and response["multi_turn_prompt"] is not None:
            try:
                multi_turn_prompt_content = response["multi_turn_prompt"][-1]["content"][0]["text"]
                if isinstance(multi_turn_prompt_content, list):
                    multi_turn_prompt_content = "\n".join(map(str, multi_turn_prompt_content))
                (trial_dir / "prompts_and_responses" / f"multi_turn_prompt_{i:02d}.txt").write_text(str(multi_turn_prompt_content))
            except Exception as e:
                print(f"Error saving multi_turn_prompt_{i}: {e}")
            i += 1
    for i, img in enumerate(visual_feedback_imgs):
        img.save(trial_dir / f"visual_feedback_{i:02d}.png")

    return code_path


def _save_in_progress_trial_artifacts(
    config: dict[str, Any],
    trial: int,
    *,
    final_code: str,
    raw_code: str | None,
    all_responses: list[dict],
) -> str | None:
    """Persist generated code before physical execution completes.

    Isaac/Omniverse can stall inside a simulator call long enough for an
    external watchdog to kill the process. This early copy keeps the generated
    program auditable even when normal trial artifact saving is never reached.
    """
    if not config["output_dir"]:
        return None
    trial_dir = Path(config["output_dir"]) / f"trial_{trial:02d}_in_progress"
    print(f"[capx-trial] in-progress save mkdir path={trial_dir}", flush=True)
    trial_dir.mkdir(parents=True, exist_ok=True)

    code_path = trial_dir / "code.py"
    print(
        f"[capx-trial] in-progress save code.py bytes={len(final_code.encode('utf-8'))}",
        flush=True,
    )
    code_path.write_text(final_code)
    print("[capx-trial] in-progress save code.py done", flush=True)
    if raw_code:
        print(
            f"[capx-trial] in-progress save raw_response bytes={len(raw_code.encode('utf-8'))}",
            flush=True,
        )
        (trial_dir / "raw_response.sh").write_text(raw_code)
        print("[capx-trial] in-progress save raw_response done", flush=True)
    print("[capx-trial] in-progress save all_responses serialize", flush=True)
    serialized_responses = json.dumps(all_responses, indent=2)
    print(
        f"[capx-trial] in-progress save all_responses bytes={len(serialized_responses.encode('utf-8'))}",
        flush=True,
    )
    (trial_dir / "all_responses.json").write_text(serialized_responses)
    print("[capx-trial] in-progress save all_responses done", flush=True)

    if all_responses:
        try:
            initial_prompt_content = all_responses[0].get("initial_prompt")
            if initial_prompt_content:
                print("[capx-trial] in-progress save initial_prompt", flush=True)
                prompt_text = initial_prompt_content[-1]["content"][0]["text"]
                (trial_dir / "initial_prompt.txt").write_text(str(prompt_text))
                print("[capx-trial] in-progress save initial_prompt done", flush=True)
        except Exception:
            pass
    print(f"[capx-trial] in-progress code saved to {code_path}", flush=True)
    return str(code_path)



def _print_and_save_summary(
    summaries: list[TrialSummary], args: LaunchArgs, config: dict[str, Any], start_time: float
) -> None:
    """Compute statistics, print them, and save to a file.

    Args:
        summaries: List of trial summaries
        args: Launch arguments
        config: Configuration dictionary
    """
    # Compute statistics
    success_count = 0
    task_completed_count = 0
    total_reward = 0.0
    total_code_blocks = 0
    total_regenerations = 0
    total_finishes = 0
    executed_trials = len(summaries)

    for summary in summaries:
        print(summary.log)
        if summary.code_path:
            print(f"Code saved to {summary.code_path}")
        success_count += int(summary.success)
        total_reward += summary.reward
        task_completed_count += int(summary.task_completed is not None and summary.task_completed)
        total_code_blocks += summary.num_code_blocks
        total_regenerations += summary.num_regenerations
        total_finishes += summary.num_finishes

    if executed_trials == 0:
        print("No trials completed.")
        return

    success_rate = success_count / executed_trials
    average_reward = total_reward / executed_trials
    average_code_blocks = total_code_blocks / executed_trials
    average_regenerations = total_regenerations / executed_trials
    average_finishes = total_finishes / executed_trials

    # Retrieve Git info
    import subprocess

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )
        git_dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.STDOUT)
            .decode("utf-8")
            .strip()
        )
        is_dirty = bool(git_dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"
        is_dirty = False

    import time

    elapsed_time = time.time() - start_time
    print("\nSummary Statistics:")
    print(f"Model: {args.model}")
    print(f"Visual Differencing Model: {args.visual_differencing_model}") if config[
        "use_img_differencing"
    ] else ""
    print(f"Config Path: {args.config_path}")
    print(f"Git Commit: {git_commit} (Dirty: {is_dirty})")
    print(f"Total number of trials: {executed_trials}")
    print(
        f"Code generation success rate / Average reward / Task completed: \n{success_rate:.3f}/{average_reward:.3f}/{task_completed_count}"
    )
    print(f"Average code blocks: {average_code_blocks:.3f}")
    print(f"Average regenerations: {average_regenerations:.3f}")
    print(f"Average finishes: {average_finishes:.3f}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Write summaries to a txt file
    if config["output_dir"]:
        with open(Path(config["output_dir"]) / "summaries.txt", "w") as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Visual Differencing Model: {args.visual_differencing_model}\n") if config[
                "use_img_differencing"
            ] else ""
            f.write(f"Config Path: {args.config_path}\n")
            f.write(f"Git Commit: {git_commit} (Dirty: {is_dirty})\n")
            f.write(f"Total number of trials: {executed_trials}\n")
            f.write(
                f"Code generation success rate / Average reward / Task completed: \n{success_rate:.3f}/{average_reward:.3f}/{task_completed_count}\n"
            )
            f.write(f"Average code blocks: {average_code_blocks:.3f}\n")
            f.write(f"Average regenerations: {average_regenerations:.3f}\n")
            f.write(f"Average finishes: {average_finishes:.3f}\n")
            f.write(f"Elapsed time: {elapsed_time:.2f} seconds\n")
