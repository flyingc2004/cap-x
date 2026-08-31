import ast
import builtins as py_builtins
import contextlib
import io
import sys
import symtable
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, SupportsFloat

import numpy as np
from gymnasium import Env, spaces

from capx.envs.base import BaseEnv, ObsType, get_env
from capx.envs.configs.instantiate import instantiate as cfg_instantiate
from capx.envs.configs.loader import DictLoader
from capx.envs.tasks.exceptions import HardStopTrial, RecoverableTaskFailure
from capx.integrations.base_api import ApiBase, get_api


class Tee(io.TextIOBase):
    """This allows streaming stdout and stderr to both the console and a buffer
    (enables breakpointing for debugging!)
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


@dataclass
class CodeExecEnvConfig:
    """Configuration for a code-execution environment.

    Attributes:
        low_level: A constructed low-level env or a YAML path to its config.
        apis: List of API names to expose to user code (e.g., "graspnet-real").
        prompt: Task instruction for the agent.
        multi_turn_prompt: Instruction for the agent to regenerate the code for multi-turn.
    """

    low_level: Env | str
    apis: list[str]
    prompt: str | None = None
    task_only_prompt: str | None = None
    multi_turn_prompt: str | None = None
    oracle_code: str | None = None
    privileged: bool = False
    enable_render: bool = True
    viser_debug: bool = False


class SimpleExecutor:
    """Minimal in-process code executor with full imports allowed.

    Executes user code with globals: env (low-level env), APIS (name->api), INPUTS, RESULT.
    The user code may import any installed package and can interact with `env` directly
    for closed-loop control.
    """

    def __init__(self, env: BaseEnv, apis: dict[str, ApiBase]) -> None:
        self._env = env
        self._apis = apis

    def run(self, code: str, *, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        g: dict[str, Any] = {
            "__name__": "__main__",
            "np": np,
            "env": self._env,
            "APIS": self._apis,
            "INPUTS": inputs or {},
            "RESULT": None,
        }
        try:
            exec(code, g, g)
            return {"ok": True, "result": g.get("RESULT")}
        except HardStopTrial:
            raise
        except RecoverableTaskFailure as exc:
            return {
                "ok": True,
                "result": {
                    "recoverable_task_failure": True,
                    "reason": exc.reason,
                    "details": exc.details,
                },
            }
        except BaseException as exc:  # defensive; propagate minimal info
            return {"ok": False, "error": repr(exc)}


class CodeExecutionEnvBase(Env):
    """High-level env that runs Python code and interacts with a low-level env."""

    prompt: str | None = None
    regenerate_prompt: str | None = None
    record_video_during_reset: bool = False

    def __init__(self, cfg: CodeExecEnvConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.low_level_env: BaseEnv = self._build_low_level(
            cfg.low_level, cfg.privileged, cfg.enable_render, cfg.viser_debug
        )  # type: ignore[assignment]
        self.record_video_during_reset = bool(
            getattr(self.low_level_env, "record_video_during_reset", False)
        )
        # Create APIs once; maximize sharing inside a worker via lru_cache in get_api
        self._apis: dict[str, ApiBase] = {n: get_api(n)(self.low_level_env) for n in cfg.apis}
        # for api in self._apis.values():
        #     api.set_env(self.low_level_env)
        self._executor = SimpleExecutor(self.low_level_env, self._apis)
        self._step_count = 0
        self.action_space = spaces.Text(max_length=4096)
        self.observation_space = spaces.Dict({"task_prompt": spaces.Text(max_length=4096)})

        # Prompt priority: YAML config (cfg.prompt) overrides the class attribute (self.prompt).
        # The class attribute serves as the single source of truth for the default task prompt.
        # YAML configs should only set prompt when they need to override the class default
        # (e.g., multi-turn variants that add extra instructions).
        self._task_prompt = cfg.prompt if cfg.prompt is not None else self.prompt

        # Oracle code: YAML config overrides class attribute
        if cfg.oracle_code is not None:
            self.oracle_code = cfg.oracle_code
        self._system_prompt = (
            "You are a helpful assistant that generates Python code to directly solve the task. "
            "The submitted code is executed once as a script, so it must perform actions immediately; "
            "if it defines a solve() helper, it must also call solve()."
        )
        self._full_prompt = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": [{"type": "text", "text": self._get_complete_prompt()}]},
        ]

        # Persistent execution namespace to retain variables across steps
        self._exec_globals: dict[str, Any] = {}
        self._init_exec_globals()

    # Functions that can be overridden by subclasses
    def compute_reward(self) -> float:
        """
        Computes the reward for the current state by delegating to the
        low-level environment.

        Returns:
            float: The reward at the current base simulator state.
        """
        return self.low_level_env.compute_reward()

    # ---- Private methods ----
    def _get_complete_prompt(self) -> str:
        """
        Get the complete prompt for the task.
        Returns:
            str: The complete prompt for the task.
        """
        docs = []
        for _name, api in self._apis.items():
            text = api.combined_doc()
            # NOTE: we need to discuss this further down the line
            # docs.append(f"- {name}:\n{text.strip()}")
            docs.append(f"\n{text.strip()}")
        return f"{self._task_prompt}\nAPIs:\n" + "\n".join(docs)

    def _exec_user_code(self, code: str) -> dict[str, Any]:
        obs = self._get_observation()
        # Update dynamic obs while retaining previously defined variables
        self._exec_globals["obs"] = obs
        self._exec_globals["env"] = self._exec_env_binding()
        self._exec_globals["APIS"] = self._exec_apis_binding()
        # Ensure API helper functions remain bound/current
        for api in self._apis.values():
            for fn_name, fn in api.functions().items():
                self._exec_globals[fn_name] = fn

        stdout_buffer = io.StringIO()
        tee_out = Tee(sys.stdout, stdout_buffer)
        stderr_buffer = io.StringIO()
        tee_err = Tee(sys.stderr, stderr_buffer)
        ok = True
        undefined_names = self._find_static_undefined_globals(code)
        if undefined_names:
            ok = False
            print(
                "StaticCodeError: generated code references undefined global "
                f"name(s): {', '.join(undefined_names)}. "
                "Use only documented APIs or define helpers before calling them.",
                file=tee_err,
            )
            return {
                "ok": ok,
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "result": self._exec_globals.get("RESULT"),
            }
        previous_solve = self._exec_globals.get("solve")
        progress_before = self._low_level_progress_marker()
        should_auto_call_solve = not _has_top_level_solve_call(code)
        try:
            with (
                contextlib.redirect_stdout(tee_out),
                contextlib.redirect_stderr(tee_err),
            ):
                exec(code, self._exec_globals, self._exec_globals)
                if should_auto_call_solve:
                    self._auto_call_generated_solve(previous_solve, progress_before)
        except HardStopTrial:
            raise
        except RecoverableTaskFailure as exc:
            self._exec_globals["RESULT"] = {
                "recoverable_task_failure": True,
                "reason": exc.reason,
                "details": exc.details,
            }
            print(
                "[capx-task] recoverable_failure "
                f"reason={exc.reason} message={exc}",
                file=tee_out,
            )
        except TimeoutError:
            raise
        except KeyboardInterrupt:
            raise
        except SystemExit as exc:
            # Generated programs may use exit()/quit() to stop an expected
            # fallback branch. A zero/None status is normal completion; retain
            # non-zero exits as sandbox failures.
            if exc.code not in (None, 0):
                ok = False
                traceback.print_exc(file=tee_err)
        except BaseException:  # defensive; propagate minimal info
            ok = False
            # Always print full traceback to the redirected stderr (tee -> console and buffer)
            traceback.print_exc(file=tee_err)

        return {
            "ok": ok,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "result": self._exec_globals.get("RESULT"),
        }

    def _find_static_undefined_globals(self, code: str) -> list[str]:
        """Find obvious undefined global names before executing generated code."""
        try:
            table = symtable.symtable(code, "<generated-code>", "exec")
        except SyntaxError:
            return []

        allowed_names = set(self._exec_globals.keys())
        allowed_names.update(dir(py_builtins))
        allowed_names.update(self._module_bound_names(table))

        undefined: set[str] = set()

        def visit(scope: symtable.SymbolTable) -> None:
            for name in scope.get_identifiers():
                symbol = scope.lookup(name)
                if not symbol.is_referenced() or not symbol.is_global():
                    continue
                if symbol.is_assigned() or symbol.is_imported():
                    continue
                if name not in allowed_names:
                    undefined.add(name)
            for child in scope.get_children():
                visit(child)

        visit(table)
        return sorted(undefined)

    @staticmethod
    def _module_bound_names(table: symtable.SymbolTable) -> set[str]:
        bound: set[str] = set()
        for name in table.get_identifiers():
            symbol = table.lookup(name)
            if symbol.is_assigned() or symbol.is_imported():
                bound.add(name)
        return bound

    def _auto_call_generated_solve(self, previous_solve: Any, progress_before: tuple[Any, ...]) -> None:
        """Run a newly defined solve() when the model forgot the top-level call."""
        solve_fn = self._exec_globals.get("solve")
        if not callable(solve_fn) or solve_fn is previous_solve:
            return
        progress_after = self._low_level_progress_marker()
        if progress_before and progress_after != progress_before:
            return
        print(
            "[capx-task] auto-calling generated solve(); code defined solve() "
            "without a top-level solve() call",
            flush=True,
        )
        result = solve_fn()
        if result is not None and self._exec_globals.get("RESULT") is None:
            self._exec_globals["RESULT"] = result

    def _low_level_progress_marker(self) -> tuple[Any, ...]:
        env = getattr(self, "low_level_env", None)
        marker: list[Any] = []
        for name in ("get_action_count", "get_step_count", "get_video_frame_count"):
            getter = getattr(env, name, None)
            if not callable(getter):
                continue
            try:
                marker.append((name, getter()))
            except Exception:
                continue
        if hasattr(env, "_sim_step_count"):
            marker.append(("_sim_step_count", getattr(env, "_sim_step_count")))
        return tuple(marker)

    def _init_exec_globals(self) -> None:
        """
        Initialize the persistent globals dictionary for user code execution.
        This is called at construction time and on reset to avoid leakage across episodes.
        """
        g: dict[str, Any] = {
            "__name__": "__main__",
            "np": np,
            "env": self._exec_env_binding(),
            "APIS": self._exec_apis_binding(),
            "breakpoint_code_block": lambda: None,
            # Populated per-step/reset; keep reference stable across execs
            "INPUTS": {},
            # Users can set and reuse RESULT across steps if desired
            "RESULT": None,
        }
        # Bind helper functions from APIs into the global namespace for convenience
        for api in self._apis.values():
            for fn_name, fn in api.functions().items():
                g[fn_name] = fn
        self._exec_globals = g

    def _exec_env_binding(self) -> Any:
        return self.low_level_env

    def set_trial_deadline(self, timeout_seconds: float) -> None:
        """Forward a wall-clock hard deadline to low-level environments."""
        setter = getattr(self.low_level_env, "set_trial_deadline", None)
        if callable(setter):
            setter(timeout_seconds)

    def clear_trial_deadline(self) -> None:
        """Clear a low-level trial deadline after normal or timeout completion."""
        clearer = getattr(self.low_level_env, "clear_trial_deadline", None)
        if callable(clearer):
            clearer()

    def _exec_apis_binding(self) -> dict[str, ApiBase]:
        return self._apis

    def _build_low_level(
        self, src: Env | str, privileged: bool = False, enable_render: bool = True, viser_debug: bool = False
    ) -> BaseEnv:
        """
        Builds the low level environment from the given source.
        Args:
            src: Env | str: the source of the low level environment
        Returns:
            BaseEnv: the low level environment
        """
        if isinstance(src, str):
            if src.endswith(".yaml") or src.endswith(".yml"):
                cfg = DictLoader.load(src)
                if isinstance(cfg, dict) and "_target_" in cfg:
                    return cfg_instantiate(cfg)  # type: ignore[no-any-return]
                return cfg  # type: ignore[return-value]
            else:
                return get_env(src, privileged=privileged, enable_render=enable_render, viser_debug=viser_debug)
        return src

    def _get_observation(self) -> dict[str, Any]:
        """
        Gets the observation of the environment. This should be consistent for all environments, where observation from low level environment
        along with the full prompt is returned
        Returns:
            Dict[str, Any]: The observation of the environment.
        """
        obs = self.low_level_env.get_observation()
        obs.update({"full_prompt": self._full_prompt})
        return obs

    # ---- Public facing methods ----
    # Public facing methods that should be consistent for all environments
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Resets the environment to an initial internal state, returning an initial observation and info.
        Args:
            seed: The seed to reset the environment with.
            options: The options to reset the environment with.
        Returns:
            tuple[ObsType, dict[str, Any]]: A tuple containing the observation and info.
        """
        self._step_count = 0
        obs, info = self.low_level_env.reset(seed=seed, options=options)
        obs.update(self._get_observation())
        # Reinitialize globals for a fresh episode and prime INPUTS with the reset observation
        self._init_exec_globals()
        self._exec_globals["INPUTS"] = obs
        for api in self._apis.values():
            if hasattr(api, "reset_episode"):
                api.reset_episode()
        info.update({"task_prompt": self._task_prompt})
        return obs, info

    def step(self, action: str) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Default implementation: execute code with helpers, report reward and logs.
        Subclasses can override hooks to customize inputs and helper bindings.
        """
        self._step_count += 1
        exec_result = self._exec_user_code(action)
        obs = self._get_observation()
        # Force viser 3D view update after code execution so the scene
        # reflects the final state (sim substep updates may have been skipped).
        if hasattr(self.low_level_env, "viser_debug") and self.low_level_env.viser_debug:
            self.low_level_env._update_viser_server()
        reward = self.compute_reward()
        if hasattr(self.low_level_env, "task_completed"):
            task_completed = self.low_level_env.task_completed()
        else:
            task_completed = None
        terminated = reward == 1.0

        truncated = getattr(self.low_level_env, "_sim_step_count", 0) >= getattr(
            self.low_level_env, "max_steps", 999999
        )  # type: ignore[arg-type]

        if not exec_result["ok"] and exec_result["stderr"] == "":
            print("Uhh we shouldn't be here, sandbox return code 1 but stderr appears empty")
            # import pdb; pdb.set_trace()
            raise RuntimeError("Sandbox return code 1 but stderr appears empty")

        info = {
            "sandbox_rc": 0 if exec_result["ok"] else 1,
            "stdout": exec_result["stdout"],
            "stderr": exec_result["stderr"],
            "task_prompt": self._task_prompt,
            "task_completed": task_completed,
        }
        return obs, reward, bool(terminated), bool(truncated), info

    def render(self, mode: str = "rgb_array"):
        return self.low_level_env.render(mode=mode)

    def render_wrist(self) -> np.ndarray | None:
        if hasattr(self.low_level_env, "render_wrist"):
            return self.low_level_env.render_wrist()
        return None

    # Video passthrough for demo compatibility
    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
        capture_initial_frame: bool = True,
    ) -> None:
        import inspect

        sig = inspect.signature(self.low_level_env.enable_video_capture)
        kwargs = {"clear": clear}
        if "wrist_camera" in sig.parameters:
            kwargs["wrist_camera"] = wrist_camera
        if "capture_initial_frame" in sig.parameters:
            kwargs["capture_initial_frame"] = capture_initial_frame
        self.low_level_env.enable_video_capture(enabled, **kwargs)

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        return self.low_level_env.get_video_frames(clear=clear)

    def get_video_frame_count(self) -> int:
        if hasattr(self.low_level_env, "get_video_frame_count"):
            return self.low_level_env.get_video_frame_count()
        if hasattr(self.low_level_env, "_frame_buffer"):
            return len(self.low_level_env._frame_buffer)
        return 0

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        if hasattr(self.low_level_env, "get_video_frames_range"):
            return self.low_level_env.get_video_frames_range(start, end)
        if hasattr(self.low_level_env, "_frame_buffer"):
            return [f.copy() for f in self.low_level_env._frame_buffer[start:end]]
        return []

    def get_wrist_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        if hasattr(self.low_level_env, "get_wrist_video_frames"):
            return self.low_level_env.get_wrist_video_frames(clear=clear)
        if hasattr(self.low_level_env, "_wrist_frame_buffer"):
            frames = [f.copy() for f in self.low_level_env._wrist_frame_buffer]
            if clear:
                self.low_level_env._wrist_frame_buffer.clear()
            return frames
        return []

    def get_wrist_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        if hasattr(self.low_level_env, "get_wrist_video_frames_range"):
            return self.low_level_env.get_wrist_video_frames_range(start, end)
        if hasattr(self.low_level_env, "_wrist_frame_buffer"):
            return [f.copy() for f in self.low_level_env._wrist_frame_buffer[start:end]]
        return []


# Use user's BaseEnv for low-level envs

_EXEC_ENV_FACTORIES: dict[str, Callable[[], CodeExecutionEnvBase]] = {}


def register_exec_env(name: str, factory: Callable[[], CodeExecutionEnvBase]) -> None:
    _EXEC_ENV_FACTORIES[name] = factory


def get_exec_env(name: str) -> Callable[[], CodeExecutionEnvBase]:
    if name not in _EXEC_ENV_FACTORIES:
        raise KeyError(f"Execution Environment '{name}' not registered")
    return _EXEC_ENV_FACTORIES[name]


def list_exec_envs() -> list[str]:
    return list(_EXEC_ENV_FACTORIES.keys())


_CONFIG_FACTORIES: dict[str, CodeExecEnvConfig] = {}


def register_config(name: str, factory: CodeExecEnvConfig) -> None:
    _CONFIG_FACTORIES[name] = factory


def get_config(name: str) -> CodeExecEnvConfig:
    if name not in _CONFIG_FACTORIES:
        raise KeyError(f"Configuration '{name}' not registered")
    return _CONFIG_FACTORIES[name]


def list_configs() -> list[str]:
    return list(_CONFIG_FACTORIES.keys())


def _has_top_level_solve_call(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    def contains_solve_call(node: ast.AST) -> bool:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "solve":
            return True
        return any(contains_solve_call(child) for child in ast.iter_child_nodes(node))

    return any(contains_solve_call(stmt) for stmt in tree.body)


__all__ = [
    "register_exec_env",
    "get_exec_env",
    "list_exec_envs",
    "register_config",
    "get_config",
    "list_configs",
    "SimpleExecutor",
    "CodeExecEnvConfig",
    "CodeExecutionEnvBase",
]
