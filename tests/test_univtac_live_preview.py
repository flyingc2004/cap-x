from __future__ import annotations

from capx.envs.simulators.univtac import UniVTACLowLevelEnv


def test_finalize_high_level_action_refreshes_preview_when_not_using_official_protocol() -> None:
    env = UniVTACLowLevelEnv.__new__(UniVTACLowLevelEnv)
    env._official_task_protocol = False
    refreshed: list[bool] = []
    env.refresh_live_preview = lambda: refreshed.append(True)
    env.get_protocol_status = lambda: {"enabled": False}

    status = env.finalize_high_level_action()

    assert refreshed == [True]
    assert status["enabled"] is False
