"""Minimal CaP-X launcher for UniVTAC."""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("CAPX_UNIVTAC_MINIMAL_IMPORTS", "1")


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CaP-X UniVTAC launcher")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=20480)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--api-key", default=os.getenv("CAPX_API_KEY"))
    parser.add_argument("--total-trials", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--record-video", type=_str2bool, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--use-oracle-code", action="store_true", default=None)
    parser.add_argument("--use-visual-feedback", type=_str2bool, default=None)
    parser.add_argument("--use-img-differencing", type=_str2bool, default=None)
    parser.add_argument("--use-video-differencing", type=_str2bool, default=None)
    parser.add_argument("--use-wrist-camera", type=_str2bool, default=None)
    parser.add_argument("--use-legacy-multi-turn-decision-prompt", type=_str2bool, default=None)
    parser.add_argument("--use-parallel-ensemble", type=_str2bool, default=None)
    parser.add_argument("--use-multimodel", type=_str2bool, default=None)
    parser.add_argument("--visual-differencing-model", default="google/gemini-3.1-pro-preview")
    parser.add_argument(
        "--visual-differencing-model-server-url",
        default="http://127.0.0.1:8110/chat/completions",
    )
    parser.add_argument("--visual-differencing-model-api-key", default=None)
    parser.add_argument("--web-ui", type=_str2bool, default=None)
    parser.add_argument("--web-ui-port", type=int, default=None)
    parser.add_argument("--univtac-device", default=os.getenv("UNIVTAC_DEVICE", "cuda:0"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _prepare_univtac_import_path()

    app = None
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        from isaaclab.app import AppLauncher

        # UniVTAC tasks import IsaacLab/Omni modules. Launch the Isaac app before
        # importing CaP-X registries that may transitively import those tasks.
        app = AppLauncher(_isaac_launcher_args(args)).app
    except Exception as exc:
        raise RuntimeError("IsaacLab AppLauncher must start before UniVTAC imports") from exc
    finally:
        sys.argv = original_argv

    _register_univtac_components()
    from capx.envs.runner import _run_headless_trials, _start_api_servers, _stop_api_servers
    from capx.utils.launch_utils import _load_config

    start_time = time.time()
    env_factory, config, api_servers = _load_config(args)
    _apply_univtac_env_overrides(env_factory)
    server_procs = _start_api_servers(api_servers)
    run_ok = False
    try:
        _run_headless_trials(args, env_factory, config, start_time)
        run_ok = True
    finally:
        _stop_api_servers(server_procs)
        if app is not None:
            force_exit = os.getenv("CAPX_FORCE_EXIT_AFTER_RUN", "1").strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "on",
            }
            if force_exit and run_ok:
                os._exit(0)
            app.close()


def _prepare_univtac_import_path() -> None:
    """Match UniVTAC's script import context when launched from CaP-X."""
    univtac_root = os.getenv("UNIVTAC_ROOT") or os.getcwd()
    if univtac_root not in sys.path:
        sys.path.insert(0, univtac_root)


def _apply_univtac_env_overrides(env_factory: dict) -> None:
    univtac_root = os.getenv("UNIVTAC_ROOT")
    try:
        low_level = env_factory["cfg"]["low_level"]
    except Exception:
        return
    if univtac_root:
        low_level["univtac_root"] = univtac_root
    univtac_device = os.getenv("UNIVTAC_DEVICE")
    if univtac_device:
        low_level["device"] = univtac_device


def _isaac_launcher_args(args: argparse.Namespace) -> argparse.Namespace:
    """Return a minimal Isaac launcher namespace.

    UniVTAC's official scripts only force cameras and a single environment
    before constructing AppLauncher. Leave headless, livestream, and device
    selection to the same environment variables used by those scripts.
    """
    return argparse.Namespace(enable_cameras=True, num_envs=1)


def _register_univtac_components() -> None:
    """Register only UniVTAC pieces to avoid importing other simulator stacks.

    Isaac/Omni is sensitive to optional robotics libraries imported into the
    same process. The generic CaP-X registries also try Robosuite, LIBERO and
    R1Pro modules; for UniVTAC runs we only need this narrow registration set.
    """
    from capx.envs.base import register_env
    from capx.envs.simulators.univtac import UniVTACLowLevelEnv
    from capx.envs.tasks.base import (
        CodeExecEnvConfig,
        CodeExecutionEnvBase,
        register_config,
        register_exec_env,
    )
    from capx.integrations.base_api import register_api
    from capx.integrations.univtac import (
        UniVTACControlApi,
        UniVTACFrankaCompatApi,
        UniVTACTactileApi,
        UniVTACTouchManipulationApi,
    )

    class UniVTACCodeEnv(CodeExecutionEnvBase):
        """Generic CaP-X code-execution env for UniVTAC tasks."""

        def _exec_env_binding(self):
            return None

        def _exec_apis_binding(self):
            return {
                name: api
                for name, api in self._apis.items()
                if name in {
                    "FrankaControlApi",
                    "UniVTACTactileApi",
                    "UniVTACTouchManipulationApi",
                }
            }

    register_env("univtac_low_level", UniVTACLowLevelEnv)
    register_exec_env("univtac_code_env", UniVTACCodeEnv)
    register_config(
        "univtac_code_env",
        CodeExecEnvConfig(
            low_level="univtac_low_level",
            apis=["FrankaControlApi", "UniVTACTactileApi"],
            privileged=False,
        ),
    )
    register_api("UniVTACControlApi", UniVTACControlApi)
    register_api("FrankaControlApi", UniVTACFrankaCompatApi)
    register_api("UniVTACTactileApi", UniVTACTactileApi)
    register_api("UniVTACTouchManipulationApi", UniVTACTouchManipulationApi)


if __name__ == "__main__":
    main()
