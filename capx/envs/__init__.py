import os

# Re-export from capx.envs.base and capx.envs.simulators
from .base import BaseEnv, get_env, list_envs, register_env

if os.getenv("CAPX_UNIVTAC_MINIMAL_IMPORTS", "0") != "1":
    from . import simulators  # noqa: F401 -- triggers env registrations
