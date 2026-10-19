from __future__ import annotations

import sys
import types

import pytest

from capx.envs import launch_univtac
from capx.envs import runner
from capx.utils import launch_utils


def test_run_exception_is_logged_before_kit_close(monkeypatch, capsys) -> None:
    events: list[object] = []
    stderr_before_close: list[str] = []

    class App:
        def close(self) -> None:
            stderr_before_close.append(capsys.readouterr().err)
            events.append("close")
            raise RuntimeError("Kit close failed")

    class AppLauncher:
        def __init__(self, _args) -> None:
            self.app = App()

    isaaclab = types.ModuleType("isaaclab")
    isaaclab.__path__ = []
    isaaclab_app = types.ModuleType("isaaclab.app")
    isaaclab_app.AppLauncher = AppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", isaaclab_app)
    monkeypatch.setattr(launch_univtac, "_prepare_univtac_import_path", lambda: None)
    monkeypatch.setattr(launch_univtac, "_isaac_launcher_args", lambda _args: None)
    monkeypatch.setattr(launch_univtac, "_register_univtac_components", lambda: None)
    monkeypatch.setattr(launch_univtac, "_apply_univtac_env_overrides", lambda _factory: None)
    monkeypatch.setattr(launch_utils, "_load_config", lambda _args: ({}, {}, []))
    monkeypatch.setattr(runner, "_start_api_servers", lambda _servers: [])
    monkeypatch.setattr(runner, "_stop_api_servers", lambda _procs: events.append("stop"))

    def fail_run(*_args) -> None:
        raise ValueError("reset failed")

    monkeypatch.setattr(runner, "_run_headless_trials", fail_run)
    monkeypatch.setattr(
        launch_univtac.faulthandler,
        "dump_traceback_later",
        lambda seconds, **_kwargs: events.append(("dump", seconds)),
    )
    monkeypatch.setattr(
        launch_univtac.faulthandler,
        "cancel_dump_traceback_later",
        lambda: events.append("cancel"),
    )
    monkeypatch.setattr(sys, "argv", ["launch_univtac", "--config-path", "unused"])

    with pytest.raises(ValueError, match="reset failed"):
        launch_univtac.main()

    assert events == ["stop", ("dump", 60), "close", "cancel"]
    assert "run error before Kit shutdown" in stderr_before_close[0]
    assert "ValueError: reset failed" in stderr_before_close[0]
    assert "Kit close error" in capsys.readouterr().err
