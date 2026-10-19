"""Start Isaac Kit, then run the OpenTacBench Shadow piano through CaP-X."""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("CAPX_UNIVTAC_MINIMAL_IMPORTS", "1")

from capx.envs.launch_univtac import build_parser


def _register_shadow_components() -> None:
    from capx.envs.base import register_env
    from capx.envs.simulators.opentac_shadow import ShadowPianoLowLevelEnv
    from capx.envs.tasks.base import CodeExecutionEnvBase
    from capx.integrations.base_api import register_api
    from capx.integrations.opentac_shadow import ShadowPianoControlApi

    class ShadowPianoCodeEnv(CodeExecutionEnvBase):
        def _exec_env_binding(self):
            return None

        def _exec_apis_binding(self):
            # Generated code receives only the documented function bindings.
            return {}

    register_env("opentac_shadow_piano", ShadowPianoLowLevelEnv)
    register_api("ShadowPianoControlApi", ShadowPianoControlApi)


def main() -> None:
    args = build_parser().parse_args()
    opentac_root = os.environ.get("UNIVTAC_ROOT")
    if not opentac_root:
        raise RuntimeError("UNIVTAC_ROOT must point to OpenTacBench")
    sys.path.insert(0, opentac_root)
    app = None
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]]
        from isaaclab.app import AppLauncher
        app = AppLauncher(argparse.Namespace(enable_cameras=True, num_envs=1)).app
    except Exception as exc:
        raise RuntimeError("IsaacLab AppLauncher must start before the Shadow task imports") from exc
    finally:
        sys.argv = original_argv

    _register_shadow_components()
    from capx.envs.runner import _run_headless_trials, _start_api_servers, _stop_api_servers
    from capx.utils.launch_utils import _load_config

    env_factory, config, api_servers = _load_config(args)
    low_level = env_factory["cfg"]["low_level"]
    if low_level.get("_target_") != "capx.envs.simulators.opentac_shadow.ShadowPianoLowLevelEnv":
        raise ValueError("Shadow launcher requires ShadowPianoLowLevelEnv config")
    low_level["univtac_root"] = opentac_root
    server_procs = _start_api_servers(api_servers)
    run_ok = False
    try:
        _run_headless_trials(args, env_factory, config, time.time())
        run_ok = True
    finally:
        _stop_api_servers(server_procs)
        from capx.envs.simulators.opentac_shadow import ACTIVE_SHADOW_ENVS
        for low_level_env in tuple(ACTIVE_SHADOW_ENVS):
            try:
                low_level_env.close()
            except Exception as exc:
                print(f"[capx-shadow] native task cleanup failed: {exc}", file=sys.stderr, flush=True)
        if app is not None:
            if run_ok and os.getenv("CAPX_FORCE_EXIT_AFTER_RUN", "1").lower() in ("1", "true", "yes"):
                os._exit(0)
            app.close()


if __name__ == "__main__":
    main()
