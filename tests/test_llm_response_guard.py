from __future__ import annotations

import json

import pytest

from capx.envs import trial
from capx.llm import client


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

    assert result == {"content": "print('ok')", "reasoning": None}
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
