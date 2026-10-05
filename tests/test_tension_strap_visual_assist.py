"""Isaac-free checks for the bounded OpenTac visual-assist path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from capx.envs.trial import (
    _capture_initial_visual_feedback,
    _should_query_multiturn_after_block,
    _visual_feedback_enabled,
)
from capx.llm import client
from capx.llm.client import preflight_image_input
from capx.utils.launch_utils import _extract_code, _load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "env_configs/univtac/tension_strap_stage_memory_control_visual.yaml"


class _RawCameraEnv:
    def __init__(self) -> None:
        self.head_calls = 0
        self.wrist_calls = 0

    def render(self):  # pragma: no cover - must not be reached by this path
        raise AssertionError("task-native dashboard must not enter the visual prompt")

    def render_head(self):
        self.head_calls += 1
        return np.full((24, 40, 3), 30, dtype=np.uint8)

    def render_wrist(self):
        self.wrist_calls += 1
        return np.full((20, 32, 3), 90, dtype=np.uint8)


def _visual_config() -> dict:
    return {
        "use_visual_feedback": True,
        "use_wrist_camera": True,
        "visual_feedback_allow_unlisted_model": True,
        "visual_feedback_raw_camera_views": True,
        "visual_feedback_max_image_side_px": 96,
        "visual_feedback_jpeg_quality": 80,
        "visual_checkpoint_markers": ["grasp_ready", "hold_12n_complete"],
    }


def test_unlisted_gpt_visual_opt_in_uses_raw_head_and_wrist_only() -> None:
    env = _RawCameraEnv()
    obs = {"full_prompt": [{"content": [{"type": "text", "text": "do task"}]}]}
    args = SimpleNamespace(model="gpt-6-sol")

    images, encoded, description, snapshots = _capture_initial_visual_feedback(
        env,
        obs,
        _visual_config(),
        args,
        SimpleNamespace(model="not-used"),
    )

    assert _visual_feedback_enabled(_visual_config(), args)
    assert env.head_calls == 1
    assert env.wrist_calls == 1
    assert len(images) == len(encoded) == len(snapshots) == 1
    assert encoded[0].startswith("data:image/jpeg;base64,")
    assert description == "do task"
    assert "Newton-valued tension" in obs["full_prompt"][-1]["content"][0]["text"]
    assert obs["full_prompt"][-1]["content"][-1]["type"] == "image_url"


def test_visual_checkpoint_is_the_only_successful_failure_only_trigger() -> None:
    clean = {"sandbox_rc": 0, "stdout": "CAPX_EVENT ok", "stderr": "", "task_completed": False}
    config = {"multi_turn_on_failure_only": True, "visual_checkpoint_markers": ["grasp_ready"]}

    assert not _should_query_multiturn_after_block(
        clean, code_block_idx=1, total_code_blocks=3, config=config
    )
    assert _should_query_multiturn_after_block(
        {**clean, "stdout": "CAPX_VISUAL_CHECKPOINT grasp_ready"},
        code_block_idx=1,
        total_code_blocks=3,
        config=config,
    )
    assert _should_query_multiturn_after_block(
        {**clean, "stdout": "Step 9\rCAPX_VISUAL_CHECKPOINT grasp_ready\r"},
        code_block_idx=1,
        total_code_blocks=3,
        config=config,
    )
    assert not _should_query_multiturn_after_block(
        clean,
        code_block_idx=3,
        total_code_blocks=3,
        config={**config, "multiturn_requires_checkpoint_or_failure": True},
    )


def test_image_preflight_uses_responses_input_image(monkeypatch) -> None:
    captured: dict = {}

    class _Response:
        status_code = 200
        headers: dict[str, str] = {}
        text = "{}"

        def raise_for_status(self) -> None:
            return None

    def fake_post(url, *, headers, payload, timeout):
        captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
        return _Response()

    monkeypatch.setenv("CAPX_LLM_PROTOCOL", "responses")
    monkeypatch.setattr(client, "_post_json", fake_post)
    preflight_image_input(
        SimpleNamespace(
            model="gpt-6-sol",
            server_url="https://gateway.example/v1/responses",
            api_key="secret",
        )
    )

    content = captured["payload"]["input"][0]["content"]
    assert captured["url"] == "https://gateway.example/v1/responses"
    assert [item["type"] for item in content] == ["input_text", "input_image"]
    assert content[1]["image_url"].startswith("data:image/png;base64,")

    stats = client._payload_stats(captured["payload"])
    assert stats["image_items"] == 1
    assert stats["image_bytes"] > 0


def test_visual_yaml_keeps_numerical_stage_authority() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    low_level = config["env"]["cfg"]["low_level"]
    prompt = config["env"]["cfg"]["prompt"]
    visible = low_level["api_configs"]["opentac_api"]["llm_visible_functions"]

    assert config["env"]["cfg"]["apis"] == ["FrankaControlApi", "OpenTacApi"]
    assert visible == [
        "get_tactile_tension_control_contract",
        "begin_tactile_tension_estimator",
        "get_tactile_tension_estimate",
        "capture_tactile_stage_response",
    ]
    assert "get_tactile_stage_memory" not in prompt
    assert config["use_visual_feedback"] is True
    assert config["visual_feedback_require_image_input"] is True
    assert config["visual_checkpoint_markers"] == ["grasp_ready", "hold_12n_complete"]
    assert "Do not use an image to estimate Newtons or decide that a stage is complete" in prompt
    assert "Do not invent arbitrary per-stage action-count limits" in prompt
    assert 'contract["stage_hold_seconds"]' in prompt
    assert "do not use hold_duration_seconds" in prompt
    assert "max_code_block_actions" not in config


def test_three_fenced_stage_program_is_not_silently_truncated() -> None:
    response = """
```python
print('setup')
breakpoint_code_block()
```
```python
print('hold 12')
breakpoint_code_block()
```
```python
print('hold 18')
```
"""

    assert _extract_code(response) == [
        "print('setup')\nbreakpoint_code_block()",
        "print('hold 12')\nbreakpoint_code_block()",
        "print('hold 18')",
    ]


def test_config_loader_preserves_visual_and_failure_only_controls() -> None:
    args = SimpleNamespace(
        config_path=str(CONFIG_PATH),
        total_trials=None,
        num_workers=None,
        record_video=None,
        output_dir=None,
        use_oracle_code=None,
        use_visual_feedback=None,
        use_img_differencing=None,
        use_parallel_ensemble=None,
        use_video_differencing=None,
        use_wrist_camera=None,
        use_multimodel=None,
        web_ui=None,
        web_ui_port=None,
        server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model="google/gemini-3.1-pro-preview",
        visual_differencing_model_server_url="http://127.0.0.1:8110/chat/completions",
        visual_differencing_model_api_key=None,
    )

    _env_factory, config, _servers = _load_config(args)

    assert config["multi_turn_on_failure_only"] is True
    assert config["multiturn_requires_checkpoint_or_failure"] is True
    assert config["stop_multiturn_when_regeneration_exhausted"] is True
    assert config["filter_multiturn_console"] is True
    assert config["visual_feedback_allow_unlisted_model"] is True
    assert config["visual_feedback_raw_camera_views"] is True
