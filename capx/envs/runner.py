"""Trial orchestration for CaP-X environments.

Handles batch execution, parallel worker dispatch, retry logic, and
wall-clock timeouts for running code-generation trials.  The actual
single-trial execution lives in :mod:`capx.envs.trial`.
"""

from __future__ import annotations

import functools
import os
import signal
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from capx.envs.configs.instantiate import instantiate
from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.envs.trial import (
    _annotate_code_blocks,
    _build_log_lines,
    _run_single_trial,
    _save_env_debug_artifacts,
    _save_tactile_artifacts,
    _save_tactile_code_memory_trace,
    _save_trial_video,
    _save_turn_and_combined_videos,
)
from capx.utils.launch_utils import (
    TrialSummary,
    _print_and_save_summary,
    _save_trial_artifacts,
    run_server_proc,
)
from capx.utils.parallel_eval import run_parallel_with_setup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRIAL_TIMEOUT_SECONDS = 1000
MAX_TRIAL_RETRIES = 3


# ---------------------------------------------------------------------------
# API server helpers
# ---------------------------------------------------------------------------

def _start_api_servers(
    api_servers: list | None, wait_timeout: float = 120.0
) -> list:
    """Launch any API server sub-processes defined in the config.

    Skips servers whose port is already in use (e.g. started externally).
    After launching, waits until all servers are accepting connections.
    """
    import socket
    import time

    procs = []
    ports_to_wait: list[tuple[str, int]] = []
    if api_servers is not None:
        for api_server in api_servers:
            port = api_server.get("port")
            host = api_server.get("host", "127.0.0.1")
            if port is not None:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex((host, int(port))) == 0:
                        print(f"API server on {host}:{port} already running, skipping")
                        continue
            proc = _run_api_server(api_server)
            procs.append(proc)
            print(f"API server {api_server} started")
            if port is not None:
                ports_to_wait.append((host, int(port)))

    # Wait for all launched servers to accept connections
    if ports_to_wait:
        deadline = time.time() + wait_timeout
        for host, port in ports_to_wait:
            while time.time() < deadline:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex((host, port)) == 0:
                        print(f"API server on {host}:{port} is ready")
                        break
                time.sleep(1.0)
            else:
                print(f"WARNING: API server on {host}:{port} not ready after {wait_timeout}s")

    return procs


def _run_api_server(api_server: dict[str, Any]):
    """Start an API server, optionally in a dedicated Python environment."""
    server_python = os.getenv("CAPX_API_SERVER_PYTHON")
    if not server_python:
        return run_server_proc(api_server)

    target = str(api_server.get("_target_", ""))
    if target.endswith(".main"):
        module = target[: -len(".main")]
    else:
        module = target
    if not module:
        raise ValueError(f"API server config missing _target_: {api_server}")

    cmd = [server_python, "-m", module]
    for key, value in api_server.items():
        if key == "_target_" or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    env = os.environ.copy()
    capx_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = (
        f"{capx_root}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else capx_root
    )
    print(
        "[capx-runner] starting API server with "
        f"python={server_python} module={module}",
        flush=True,
    )
    return subprocess.Popen(cmd, env=env)


def _stop_api_servers(server_procs: list) -> None:
    """Terminate API server sub-processes."""
    for proc in server_procs:
        proc.terminate()
        if hasattr(proc, "join"):
            proc.join(timeout=5.0)
        else:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

def _setup_output_dir(args, config: dict[str, Any]) -> None:
    """Create and normalize the output directory path.

    Inserts the model name into the output path so that results from
    different models are stored separately.
    """
    if args.use_oracle_code:
        args.model = "oracle"
    if config["output_dir"]:
        preserve_output_dir = os.getenv("CAPX_PRESERVE_OUTPUT_DIR", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if preserve_output_dir:
            Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
            return
        parts = config["output_dir"].split("/")
        parts.insert(-1, str(args.model).replace("/", "_"))
        new_out_dir = "/".join(parts)
        Path(new_out_dir).mkdir(parents=True, exist_ok=True)
        config["output_dir"] = new_out_dir


# ---------------------------------------------------------------------------
# Headless trial runner
# ---------------------------------------------------------------------------

def _run_headless_trials(
    args,
    env_factory: dict[str, Any],
    config: dict[str, Any],
    start_time: float,
) -> None:
    """Run all trials in headless CLI mode, then print a summary.

    Dispatches to parallel or sequential execution depending on ``num_workers``.
    """
    if config["total_trials"] <= 0:
        print("No trials requested; exiting.")
        return

    if config["record_video"] and not config["output_dir"]:
        raise ValueError("record_video requires --output-dir to save generated videos")

    _setup_output_dir(args, config)

    # Determine which trials to run (supports resume)
    trial_ids = list(range(1, config["total_trials"] + 1))
    if config.get("resume_idx") is not None:
        trial_ids = list(range(config["resume_idx"], config["total_trials"] + 1))

    # Execute trials
    if config["num_workers"] > 1:
        setup_fn = functools.partial(_worker_setup, env_factory=env_factory)
        trial_fn = functools.partial(_run_single_trial_worker, args=args, config=config)
        summaries = run_parallel_with_setup(
            trial_ids,
            num_workers=config["num_workers"],
            setup_fn=setup_fn,
            trial_fn=trial_fn,
        )
    else:
        summaries = _run_trial_batch(trial_ids, args=args, env_factory=env_factory, config=config)

    summaries.sort(key=lambda s: s.trial)
    _print_and_save_summary(summaries, args, config, start_time)

    # Write completion flag
    with open(os.path.join(config["output_dir"], "aaa_done_flag.txt"), "w") as f:
        f.write("1")


# ---------------------------------------------------------------------------
# Worker setup and trial dispatch
# ---------------------------------------------------------------------------

def _worker_setup(*, env_factory: dict[str, Any]) -> tuple[Any, str | None]:
    """Create the environment once per parallel worker.

    Returns:
        Tuple of (environment, multi_turn_prompt).
    """
    env = instantiate(env_factory)
    multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt", None)
    return (env, multi_turn_prompt)


def _run_single_trial_worker(
    state: tuple[Any, str | None],
    trial: int,
    *,
    args,
    config: dict[str, Any],
) -> TrialSummary:
    """Run a single trial using pre-initialized worker state (for parallel mode)."""
    env, multi_turn_prompt = state
    return _run_trial_with_retries(env, trial, args, config, multi_turn_prompt)


def _run_trial_with_retries(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
) -> TrialSummary:
    """Attempt a trial up to MAX_TRIAL_RETRIES times, retrying on timeout."""
    max_retries = max(1, int(config.get("max_trial_retries", MAX_TRIAL_RETRIES)))
    timeout_s = float(config.get("trial_timeout_seconds", TRIAL_TIMEOUT_SECONDS))
    print(
        f"[capx-runner] trial={trial} timeout_s={timeout_s:g} "
        f"max_retries={max_retries}"
    )
    for attempt in range(max_retries):
        try:
            is_last_attempt = attempt == max_retries - 1
            return _run_single_trial_with_timeout(
                env=env,
                trial=trial,
                args=args,
                config=config,
                multi_turn_prompt=multi_turn_prompt,
                timeout_s=timeout_s,
                raise_on_timeout=not is_last_attempt,
            )
        except TimeoutError:
            print(f"Trial {trial} timed out (attempt {attempt + 1}/{max_retries}). Retrying...")

    # All retries exhausted
    return TrialSummary(
        trial=trial,
        success=False,
        reward=0.0,
        terminated=False,
        truncated=True,
        sandbox_rc=1,
        log=f"Trial {trial} failed after {max_retries} timeout retries",
        task_completed=False,
        code_path=None,
        num_regenerations=0,
        num_finishes=0,
        num_code_blocks=0,
    )


def _run_trial_batch(
    trial_ids: Iterable[int],
    *,
    args,
    env_factory: dict[str, Any],
    config: dict[str, Any],
) -> list[TrialSummary]:
    """Run a batch of trials sequentially (single-worker mode).

    Creates one environment instance and reuses it for all trials.
    """
    trial_indices = list(trial_ids)
    if not trial_indices:
        return []

    # OmniGibson / Isaac Sim reads sys.argv and can crash on unknown flags.
    # Strip our CLI args while instantiating the env.
    original_sys_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        env = instantiate(env_factory)
    finally:
        sys.argv = original_sys_argv

    multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt", None)

    summaries: list[TrialSummary] = []
    for trial in tqdm(trial_indices, desc="Running Trials"):
        summary = _run_trial_with_retries(env, trial, args, config, multi_turn_prompt)
        summaries.append(summary)

    summaries.sort(key=lambda s: s.trial)
    return summaries


# ---------------------------------------------------------------------------
# Timeout wrapper
# ---------------------------------------------------------------------------

def _run_single_trial_with_timeout(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    timeout_s: float,
    raise_on_timeout: bool = False,
) -> TrialSummary:
    """Run a single trial with a wall-clock timeout via SIGALRM."""
    timeout_seconds = max(1, int(timeout_s))
    timed_out = False

    def _timeout_handler(signum: int, frame) -> None:  # type: ignore[override]
        nonlocal timed_out
        timed_out = True
        raise TimeoutError(f"Trial {trial} exceeded {timeout_seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    set_deadline = getattr(env, "set_trial_deadline", None)
    if callable(set_deadline):
        set_deadline(max(1, timeout_seconds - 2))
    partial_artifacts: dict[str, Any] = {}
    try:
        return _run_single_trial(
            env, trial, args, config, multi_turn_prompt, partial_artifacts=partial_artifacts
        )
    except KeyboardInterrupt as exc:
        # A manual interrupt is common while inspecting a long Isaac run.  The
        # simulator has already accumulated frames in memory, so flush them
        # before propagating Ctrl-C and stopping the batch.
        print(
            f"[capx-runner] Ctrl-C received during trial {trial}; "
            "saving partial artifacts before exit...",
            flush=True,
        )
        _build_interrupted_summary(env, trial, partial_artifacts, config, exc)
        print(
            f"[capx-runner] trial {trial} partial artifacts saved; exiting with code 130",
            flush=True,
        )
        raise
    except BaseException as exc:
        is_timeout = timed_out or isinstance(exc, TimeoutError)
        try:
            import requests as _requests
            is_timeout = is_timeout or isinstance(exc, _requests.exceptions.Timeout)
        except Exception:
            pass

        if not is_timeout:
            raise
        if raise_on_timeout:
            raise TimeoutError(f"Trial {trial} timed out") from exc

        print(f"Trial {trial} timed out after {timeout_seconds} seconds")
        return _build_timeout_summary(env, trial, timeout_seconds, partial_artifacts, config, exc)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        clear_deadline = getattr(env, "clear_trial_deadline", None)
        if callable(clear_deadline):
            clear_deadline()


def _build_timeout_summary(
    env: CodeExecutionEnvBase,
    trial: int,
    timeout_seconds: int,
    pa: dict[str, Any],
    config: dict[str, Any],
    exc: BaseException,
) -> TrialSummary:
    """Build a TrialSummary from partial artifacts after a timeout."""
    return _build_aborted_summary(
        env,
        trial,
        pa,
        config,
        exc,
        sandbox_rc=1,
        truncated=False,
        prefix=f"Trial {trial} timed out after {timeout_seconds} seconds.",
        error_label="Timeout Error",
        video_suffix="timeout",
    )


def _build_interrupted_summary(
    env: CodeExecutionEnvBase,
    trial: int,
    pa: dict[str, Any],
    config: dict[str, Any],
    exc: KeyboardInterrupt,
) -> TrialSummary:
    """Flush partial trial output when the user stops the run with Ctrl-C."""
    return _build_aborted_summary(
        env,
        trial,
        pa,
        config,
        exc,
        sandbox_rc=130,
        truncated=True,
        prefix=f"Trial {trial} interrupted by user (Ctrl-C).",
        error_label="KeyboardInterrupt",
        video_suffix="interrupted",
    )


def _build_aborted_summary(
    env: CodeExecutionEnvBase,
    trial: int,
    pa: dict[str, Any],
    config: dict[str, Any],
    exc: BaseException,
    *,
    sandbox_rc: int,
    truncated: bool,
    prefix: str,
    error_label: str,
    video_suffix: str,
) -> TrialSummary:
    """Persist code, video, and diagnostics for an interrupted trial."""
    raw_code = pa.get("raw_code", "")
    code_blocks = pa.get("code_blocks", [])
    code_block_metadata = pa.get("code_block_metadata", [])
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)

    info_step = pa.get("info_step", {"sandbox_rc": 1, "stdout": "", "stderr": str(exc)})
    info_step["sandbox_rc"] = sandbox_rc
    if info_step.get("stderr") == "":
        info_step["stderr"] = str(exc)
    else:
        info_step["stderr"] += f"\n\n{error_label}: {exc}"

    reward = pa.get("reward", 0.0)
    terminated = pa.get("terminated", False)
    truncated = bool(truncated or pa.get("truncated", False))
    num_regenerations = pa.get("num_regenerations", 0)
    num_finishes = pa.get("num_finishes", 0)
    num_code_blocks = pa.get("num_code_blocks", len(code_blocks))

    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        prefix=prefix,
    )

    code_path = _save_trial_artifacts(
        config, trial,
        sandbox_rc=sandbox_rc,
        reward=reward,
        task_completed=info_step.get("task_completed", False),
        final_code=final_code,
        raw_code=raw_code,
        all_responses=pa.get("all_responses", []),
        log_lines=log_lines,
        visual_feedback_imgs=pa.get("visual_feedback_imgs", []),
        ensemble_data=pa.get("ensemble_data"),
        multiturn_ensemble_data=pa.get("multiturn_ensemble_data", []),
    )
    # Ignore repeated Ctrl-C while serializing the buffered frames.  The first
    # interrupt has already been handled; a second should not corrupt the only
    # copy of the partial video.
    previous_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if config.get("record_video", False):
            turn_frame_ranges = list(pa.get("turn_frame_ranges", []))
            current_frame_start = pa.get("current_frame_start")
            if current_frame_start is not None and hasattr(env, "get_video_frame_count"):
                frame_count = int(env.get_video_frame_count())
                start = int(current_frame_start)
                if frame_count > start and (
                    not turn_frame_ranges or turn_frame_ranges[-1] != (start, frame_count)
                ):
                    turn_frame_ranges.append((start, frame_count))

            if pa.get("recording_frames", False) and turn_frame_ranges:
                _save_turn_and_combined_videos(
                    env, config, trial, info_step, reward, turn_frame_ranges,
                )
            else:
                _save_trial_video(
                    env, config, trial, info_step, reward, num_code_blocks,
                    suffix_extra=video_suffix,
                )
        _save_tactile_artifacts(env, config, trial, info_step, reward)
        _save_env_debug_artifacts(env, config, trial, info_step, reward)
        trace = pa.get("tactile_code_memory_trace", [])
        if isinstance(trace, list):
            _save_tactile_code_memory_trace(config, trial, info_step, reward, trace)
    except Exception as artifact_exc:
        print(f"WARNING: Failed to save interrupted trial artifacts: {artifact_exc}")
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

    return TrialSummary(
        trial=trial,
        success=False,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=sandbox_rc,
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", False),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )
