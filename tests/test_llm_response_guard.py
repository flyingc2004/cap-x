from __future__ import annotations

import json

import pytest
from PIL import Image

from capx.envs import trial
from capx.llm import client
from capx.utils.launch_utils import _save_trial_artifacts


def test_query_requests_identity_encoding_and_logs_response_stages(monkeypatch, capsys) -> None:
    observed: dict[str, object] = {}
    body = {
        "choices": [{"message": {"content": "print('ok')", "reasoning": None}}],
    }

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = json.dumps(body).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return body

    def post(url, *, headers, data, timeout, stream):
        observed.update(
            url=url,
            headers=headers,
            data=data,
            timeout=timeout,
            stream=stream,
        )
        return Response()

    monkeypatch.setattr(client.requests, "post", post)
    args = client.ModelQueryArgs(model="test-model", server_url="https://example.test/v1")

    result = client.query_model(args, [{"role": "user", "content": "hello"}])

    assert result["content"] == "print('ok')"
    assert result["reasoning"] is None
    assert result["usage"]["usage"]["provider_reported"] is False
    assert observed["stream"] is False
    assert observed["headers"]["Accept-Encoding"] == "identity"
    output = capsys.readouterr().out
    assert "response body materialization begin" in output
    assert "response body materialization end" in output
    assert "response JSON parse begin" in output
    assert "response JSON parse end" in output


def test_query_deadline_remains_bounded_while_action_alarm_is_paused(monkeypatch) -> None:
    alarms = iter([91, 0, 0, 0])
    calls: list[int] = []

    def alarm(seconds: int) -> int:
        calls.append(seconds)
        return next(alarms)

    monkeypatch.setattr(trial.signal, "alarm", alarm)
    monkeypatch.setenv("CAPX_LLM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("CAPX_LLM_HARD_TIMEOUT_GRACE_SECONDS", "7")

    with trial._suspend_sigalrm():
        pass

    assert calls == [0, 37, 0, 91]


def test_query_sends_opted_in_reasoning_effort(monkeypatch) -> None:
    observed: dict[str, object] = {}
    body = {"choices": [{"message": {"content": "print('ok')"}}]}

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = json.dumps(body).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return body

    def post(_url, *, headers, data, timeout, stream):
        observed["payload"] = json.loads(data)
        return Response()

    monkeypatch.setattr(client.requests, "post", post)
    monkeypatch.setenv("CAPX_LLM_SEND_REASONING_EFFORT", "1")
    args = client.ModelQueryArgs(
        model="provider-reasoning-model",
        server_url="https://example.test/v1",
        reasoning_effort="high",
    )

    client.query_model(args, [{"role": "user", "content": "hello"}])

    assert observed["payload"]["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    ("requested_effort", "expected_effort"),
    [("low", "low"), ("medium", "high"), ("high", "high"), ("max", "max")],
)
def test_kimi_k3_uses_its_native_reasoning_contract(
    monkeypatch, requested_effort, expected_effort
) -> None:
    observed: dict[str, object] = {}
    body = {
        "choices": [
            {
                "message": {
                    "content": "print('ok')",
                    "reasoning_content": "bounded reasoning",
                }
            }
        ]
    }

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = json.dumps(body).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return body

    def post(_url, *, headers, data, timeout, stream):
        observed["payload"] = json.loads(data)
        return Response()

    monkeypatch.setattr(client.requests, "post", post)
    monkeypatch.setenv("CAPX_DISABLE_THINKING", "1")
    args = client.ModelQueryArgs(
        model="kimi-k3",
        server_url="https://example.test/v1",
        reasoning_effort=requested_effort,
    )

    result = client.query_model(args, [{"role": "user", "content": "hello"}])

    assert result["content"] == "print('ok')"
    assert result["reasoning"] == "bounded reasoning"
    assert result["usage"]["usage"]["provider_reported"] is False
    assert observed["payload"]["reasoning_effort"] == expected_effort
    assert "enable_thinking" not in observed["payload"]


def test_query_rejects_empty_message_content(monkeypatch) -> None:
    body = {"choices": [{"message": {"content": "", "reasoning_content": "hidden"}}]}

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = json.dumps(body).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return body

    monkeypatch.setattr(client.requests, "post", lambda *_args, **_kwargs: Response())
    args = client.ModelQueryArgs(model="test-model", server_url="https://example.test/v1")

    with pytest.raises(RuntimeError, match="empty message.content"):
        client.query_model(args, [{"role": "user", "content": "hello"}])


def test_query_exposes_provider_rejection_with_status_and_body(monkeypatch) -> None:
    class Response:
        status_code = 429
        headers = {"content-type": "application/json", "retry-after": "30"}
        content = b'{"error":{"message":"rate limit exceeded"}}'

        def raise_for_status(self) -> None:
            raise RuntimeError("HTTP 429")

    monkeypatch.setattr(client.requests, "post", lambda *_args, **_kwargs: Response())
    args = client.ModelQueryArgs(model="kimi-k3", server_url="https://example.test/v1")

    with pytest.raises(client.LLMQueryError, match="status=429") as exc_info:
        client.query_model(args, [{"role": "user", "content": "hello"}])

    assert exc_info.value.retry_after == "30"
    assert "rate limit exceeded" in exc_info.value.response_preview


def test_responses_protocol_routes_and_parses_output_items(monkeypatch) -> None:
    observed: dict[str, object] = {}
    body = {
        "status": "completed",
        "usage": {
            "input_tokens": 120,
            "input_tokens_details": {"cached_tokens": 16},
            "output_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 24},
            "total_tokens": 160,
        },
        "output": [
            {"type": "reasoning", "summary": [{"text": "checked contract"}]},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "print('ok')"}],
            },
        ],
    }

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = json.dumps(body).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return body

    def post(url, *, headers, data, timeout, stream):
        observed.update(url=url, headers=headers, payload=json.loads(data), timeout=timeout, stream=stream)
        return Response()

    monkeypatch.setattr(client.requests, "post", post)
    monkeypatch.setenv("CAPX_LLM_PROTOCOL", "responses")
    monkeypatch.setenv("CAPX_DISABLE_THINKING", "1")
    args = client.ModelQueryArgs(model="gpt-6-sol", server_url="https://example.test/v1/chat/completions")

    result = client.query_model(
        args,
        [
            {"role": "system", "content": "Write code."},
            {"role": "user", "content": [{"type": "text", "text": "Do task."}]},
        ],
    )

    assert result["content"] == "print('ok')"
    assert result["reasoning"] == "checked contract"
    assert result["usage"]["usage"] == {
        "provider_reported": True,
        "input_tokens": 120,
        "output_tokens": 40,
        "reasoning_tokens": 24,
        "cached_input_tokens": 16,
        "total_tokens": 160,
        "provider_usage": body["usage"],
    }
    assert observed["url"] == "https://example.test/v1/responses"
    payload = observed["payload"]
    assert payload["model"] == "gpt-6-sol"
    assert payload["max_output_tokens"] == 4096
    assert "enable_thinking" not in payload
    assert payload["input"][0]["content"] == [{"type": "input_text", "text": "Write code."}]
    assert payload["input"][1]["content"] == [{"type": "input_text", "text": "Do task."}]


def test_responses_protocol_sends_opted_in_nested_reasoning_effort(monkeypatch) -> None:
    observed: dict[str, object] = {}
    body = {"output_text": "print('ok')"}

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = json.dumps(body).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return body

    def post(_url, *, headers, data, timeout, stream):
        observed["payload"] = json.loads(data)
        return Response()

    monkeypatch.setattr(client.requests, "post", post)
    monkeypatch.setenv("CAPX_LLM_PROTOCOL", "responses")
    monkeypatch.setenv("CAPX_RESPONSES_SEND_REASONING_EFFORT", "1")
    args = client.ModelQueryArgs(
        model="gpt-6-sol",
        server_url="https://example.test/v1/responses",
        reasoning_effort="low",
        max_tokens=2048,
    )

    client.query_model(args, [{"role": "user", "content": "Write code."}])

    payload = observed["payload"]
    assert payload["max_output_tokens"] == 2048
    assert payload["reasoning"] == {"effort": "low"}


def test_usage_collector_summarizes_chat_and_responses_tokens(monkeypatch) -> None:
    body = {
        "choices": [{"message": {"content": "print('ok')"}}],
        "usage": {
            "prompt_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens": 12,
            "completion_tokens_details": {"reasoning_tokens": 7},
            "total_tokens": 32,
        },
    }

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = json.dumps(body).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return body

    monkeypatch.setattr(client.requests, "post", lambda *_args, **_kwargs: Response())
    events: list[dict] = []
    args = client.ModelQueryArgs(model="test-model", server_url="https://example.test/v1")
    with client.collect_llm_usage(events, "initial_code"):
        client.query_model(args, [{"role": "user", "content": "hello"}])

    summary = client.summarize_llm_usage(events)
    assert summary["query_count"] == 1
    assert summary["provider_reported_query_count"] == 1
    assert summary["queries"][0]["phase"] == "initial_code"
    assert summary["totals"] == {
        "input_tokens": 20,
        "output_tokens": 12,
        "reasoning_tokens": 7,
        "cached_input_tokens": 4,
        "total_tokens": 32,
    }


def test_trial_artifact_persists_provider_usage(tmp_path) -> None:
    events = [
        {
            "phase": "initial_code",
            "model": "test-model",
            "protocol": "chat_completions",
            "usage": {
                "provider_reported": True,
                "input_tokens": 11,
                "output_tokens": 9,
                "reasoning_tokens": 3,
                "cached_input_tokens": 2,
                "total_tokens": 20,
            },
        }
    ]
    _save_trial_artifacts(
        {"output_dir": str(tmp_path)},
        trial=1,
        sandbox_rc=0,
        reward=1.0,
        task_completed=True,
        final_code="print('ok')",
        raw_code=None,
        all_responses=[],
        log_lines=["summary"],
        visual_feedback_imgs=[Image.new("RGB", (1, 1))],
        llm_usage_events=events,
    )

    usage_path = tmp_path / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1" / "llm_usage.json"
    usage = json.loads(usage_path.read_text())
    assert usage["totals"]["total_tokens"] == 20
    assert usage["queries"][0]["phase"] == "initial_code"
