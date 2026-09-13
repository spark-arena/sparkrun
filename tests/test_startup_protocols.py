"""Real local SSE servers qualify each readiness wire protocol without a GPU."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

import pytest

from sparkrun.core.readiness import OPENAI_RESPONSES_STREAM, ANTHROPIC_MESSAGES_STREAM
from sparkrun.scripts import startup_probe
from sparkrun.orchestration.startup import validate_observation


PROTOCOLS = [OPENAI_RESPONSES_STREAM, ANTHROPIC_MESSAGES_STREAM]


def events(style, text="OK"):
    if style == OPENAI_RESPONSES_STREAM:
        return [
            {"type": "response.created", "response": {"status": "in_progress"}},
            {"type": "response.reasoning_summary_text.delta", "delta": "thinking"},
            {"type": "response.output_text.delta", "delta": text},
            {"type": "response.completed", "response": {"status": "completed"}},
        ]
    return [
        {"type": "message_start", "message": {"role": "assistant"}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "thinking"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]


def encode(values):
    return b"".join(("event: " + str(value.get("type")) + "\ndata: " + json.dumps(value) + "\n\n").encode() for value in values)


def config(style, **extra):
    return {
        "container": "head",
        "address": "127.0.0.1",
        "port": 8000,
        "port_timeout_s": 1,
        "health_timeout_s": 1,
        "health_path": "/health",
        "inference_timeout_s": 5,
        "inference_style": style,
        "prompt": "Reply OK",
        "model": "served-alias",
        **extra,
    }


@pytest.mark.parametrize("style", PROTOCOLS)
def test_real_native_stream_uses_correct_endpoint_payload_and_auth(style):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(self.path)
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self):
            requests.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))), dict(self.headers)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(encode(events(style)))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with (
            patch.object(startup_probe, "verify_host_observer"),
            patch.object(startup_probe, "port_listening", return_value=True),
            patch.object(startup_probe, "inspect_container", return_value=("current", 1)),
        ):
            result = startup_probe.observe(config(style, port=server.server_port, expected="OK", api_key="fixture-token"))
        assert requests[0] == "/health" and len(requests) == 2  # no OpenAI model discovery required
        path, body, headers = requests[1]
        assert body["model"] == "served-alias" and body["stream"] is True
        assert headers["Authorization"] == "Bearer fixture-token"
        if style == OPENAI_RESPONSES_STREAM:
            assert path == "/v1/responses" and body["input"] == "Reply OK" and body["max_output_tokens"] == 64
            assert body["store"] is False and "messages" not in body
        else:
            assert path == "/v1/messages" and body["messages"] == [{"role": "user", "content": "Reply OK"}]
            assert headers["Anthropic-Version"] == "2023-06-01" and headers["X-Api-Key"] == "fixture-token"
        assert result["response_validated"] and result["content"] == "OK"
        assert result["first_token_field"] == "reasoning" and result["inference_style"] == style
        assert validate_observation(result, expected_style=style, require_inference=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("style", PROTOCOLS)
@pytest.mark.parametrize("case", ["empty", "truncated", "wrong", "error", "invalid_text", "oversized"])
def test_native_exact_validation_rejects_bad_streams(style, case):
    values = events(style)
    if case == "empty":
        values = values[:1] + values[-1:]
    elif case == "truncated":
        values = values[:-1]
    elif case == "wrong":
        values = events(style, "wrong")
    elif case == "error":
        values = [{"type": "error", "error": {"message": "unavailable"}}]
    elif case == "invalid_text":
        values = events(style, ["not a string"])
    raw = b"data: " + b"x" * 65537 if case == "oversized" else encode(values)
    health = io.BytesIO()
    health.status = 200
    http = Mock()
    http.open.side_effect = [health, io.BytesIO(raw)]
    with (
        patch.object(startup_probe, "verify_host_observer"),
        patch.object(startup_probe, "port_listening", return_value=True),
        patch.object(startup_probe, "inspect_container", return_value=("current", 1)),
        patch.object(startup_probe.urllib.request, "build_opener", return_value=http),
    ):
        with pytest.raises((RuntimeError, ValueError)):
            startup_probe.observe(config(style, expected="OK"))


@pytest.mark.parametrize("style", PROTOCOLS)
def test_readiness_returns_at_first_real_delta_without_waiting_for_completion(style):
    raw = encode(events(style)[:2] if style == OPENAI_RESPONSES_STREAM else events(style)[:3])
    health = io.BytesIO()
    health.status = 200
    http = Mock()
    http.open.side_effect = [health, io.BytesIO(raw)]
    with (
        patch.object(startup_probe, "verify_host_observer"),
        patch.object(startup_probe, "port_listening", return_value=True),
        patch.object(startup_probe, "inspect_container", return_value=("current", 1)),
        patch.object(startup_probe.urllib.request, "build_opener", return_value=http),
    ):
        result = startup_probe.observe(config(style))
    assert result["inference_ready"] and not result["response_validated"]
