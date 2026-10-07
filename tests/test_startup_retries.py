"""Bounded startup retries, including the standalone subprocess and real HTTP IO."""

import errno
import io
import json
import logging
import socket
import ssl
import subprocess
import threading
import time
import urllib.error
from email.message import Message
from email.utils import formatdate
from http.client import RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.orchestration import startup
from sparkrun.scripts import startup_probe as probe


STYLES = list(probe.INFERENCE_PROBES)


def config(**extra):
    return {
        "container": "head",
        "address": "127.0.0.1",
        "port": 8000,
        "port_timeout_s": 100,
        "health_timeout_s": 100,
        "inference_timeout_s": 120,
        "model": "test-model",
        "prompt": "Reply OK",
        **extra,
    }


def text_event(style=probe.OPENAI_CHAT_STREAM):
    if style == probe.OPENAI_CHAT_STREAM:
        value = {"choices": [{"delta": {"content": "OK"}}]}
    elif style == probe.OPENAI_RESPONSES_STREAM:
        value = {"type": "response.output_text.delta", "delta": "OK"}
    else:
        value = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "OK"}}
    return ("data: " + json.dumps(value) + "\n\n").encode()


def http_error(code, after=None):
    headers = Message()
    if after is not None:
        headers["Retry-After"] = after
    return urllib.error.HTTPError("http://localhost/inference", code, "unavailable", headers, io.BytesIO(b"private error body"))


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def time(self):
        return 1_700_000_000 + self.now

    def time_ns(self):
        return int(self.time() * 1e9)

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    # Replace only the probe's clock, not the time module used by pytest/threads.
    monkeypatch.setattr(probe, "time", value)
    monkeypatch.setattr(probe, "inspect_container", Mock(return_value=("current", 1)))
    return value


def infer(http, **extra):
    return probe.observe_inference(
        config(**extra), http, "http://localhost", {"container_id": "current", "container_started_unix_ns": 1, "inference_ready": False}
    )


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("code", [408, 429, 502, 503, 504])
def test_transient_http_error_retries_all_protocols(clock, style, code):
    error = http_error(code)
    http = Mock()
    http.open.side_effect = [error, io.BytesIO(text_event(style))]
    result = infer(http, inference_style=style)
    assert result["inference_ready"] and result["inference_attempts"] == 2
    assert result["inference_wait_s"] == 2 and result["request_ttft_seconds"] == 0
    assert error.fp.closed
    assert [call.kwargs["timeout"] for call in http.open.call_args_list] == [120, 118]


@pytest.mark.parametrize(
    "error",
    [
        ConnectionRefusedError(),
        ConnectionResetError(),
        urllib.error.URLError(OSError(errno.ECONNREFUSED, "refused")),
        urllib.error.URLError(OSError(errno.ECONNRESET, "reset")),
        RemoteDisconnected("remote end closed before headers"),
    ],
)
def test_transient_transport_errors_retry(clock, error):
    http = Mock()
    http.open.side_effect = [error, io.BytesIO(text_event())]
    assert infer(http)["inference_attempts"] == 2


@pytest.mark.parametrize(
    "error",
    [http_error(code) for code in (400, 401, 403, 404, 422, 500, 501)]
    + [
        TimeoutError("pending request exceeded its budget"),
        urllib.error.URLError(ssl.SSLCertVerificationError("invalid certificate")),
        urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "DNS lookup failed")),
        urllib.error.URLError("unknown transport failure"),
        BrokenPipeError(),
        PermissionError(),
    ],
)
def test_other_errors_fail_without_retry(clock, error):
    http = Mock()
    http.open.side_effect = error
    with pytest.raises(type(error)):
        infer(http)
    assert http.open.call_count == 1 and clock.now == 0
    if isinstance(error, urllib.error.HTTPError):
        assert error.fp.closed


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("raw", [b"data: invalid json\n\n", b'data: {"error":"busy"}\n\n', b""])
def test_protocol_errors_never_retry(clock, style, raw):
    http = Mock()
    http.open.return_value = io.BytesIO(raw)
    with pytest.raises((RuntimeError, ValueError)):
        infer(http, inference_style=style)
    assert http.open.call_count == 1 and clock.now == 0


class ResetStream(io.BytesIO):
    def readline(self, *args):
        line = super().readline(*args)
        if not line:
            raise ConnectionResetError("stream reset")
        return line


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("partial", [False, True])
def test_stream_reset_retries_only_before_text(clock, style, partial):
    http = Mock()
    http.open.side_effect = [ResetStream(text_event(style) if partial else b": keepalive\n\n"), io.BytesIO(text_event(style))]
    if partial:
        with pytest.raises(ConnectionResetError):
            infer(http, inference_style=style, expected="OK")
        assert http.open.call_count == 1 and clock.now == 0
    else:
        assert infer(http, inference_style=style)["inference_attempts"] == 2


def test_backoff_caps_at_thirty_and_all_attempts_share_budget(clock, capsys):
    http = Mock()

    def respond(*args, **kwargs):
        clock.sleep(1)  # Count time spent in each request, too.
        if http.open.call_count <= 6:
            raise http_error(503)
        return io.BytesIO(text_event())

    http.open.side_effect = respond
    result = infer(http)
    assert result["inference_attempts"] == 7 and result["inference_wait_s"] == 97
    assert result["request_ttft_seconds"] == 1
    assert result["request_started_unix_ns"] == int((1_700_000_000 + 96) * 1e9)
    assert [call.kwargs["timeout"] for call in http.open.call_args_list] == [120, 117, 112, 103, 86, 55, 24]
    output = capsys.readouterr()
    assert not output.out
    for delay in (2, 4, 8, 16, 30):
        assert "backoff %.1fs" % delay in output.err
    assert "private error body" not in output.err


@pytest.mark.parametrize("after", ["5", "date"])
def test_retry_after_seconds_and_http_date(clock, after):
    after = formatdate(clock.time() + 5, usegmt=True) if after == "date" else after
    http = Mock()
    http.open.side_effect = [http_error(429, after), io.BytesIO(text_event())]
    assert infer(http)["inference_wait_s"] == 5


@pytest.mark.parametrize("after", [None, "", "garbage", "-3", "1.5", "1", "past"])
def test_short_or_invalid_retry_after_keeps_backoff(clock, after):
    after = formatdate(clock.time() - 5, usegmt=True) if after == "past" else after
    http = Mock()
    http.open.side_effect = [http_error(503, after), io.BytesIO(text_event())]
    assert infer(http)["inference_wait_s"] == 2


@pytest.mark.parametrize("after,attempts", [(None, 2), ("600", 1)])
def test_budget_exhaustion_bounds_backoff_and_reports_last_error(clock, after, attempts):
    http = Mock()

    def unavailable(*args, **kwargs):
        clock.sleep(1)
        raise http_error(503, after)

    http.open.side_effect = unavailable
    with pytest.raises(TimeoutError, match="after %d attempt.*last transient error: HTTP 503" % attempts):
        infer(http, inference_timeout_s=7)
    assert clock.now == 7 and http.open.call_count == attempts


@pytest.mark.parametrize("replacement", [("other", 1), ("current", 2), RuntimeError("head container is not running")])
def test_dead_or_restarted_container_stops_retries(clock, replacement):
    http = Mock()
    http.open.side_effect = http_error(503)
    probe.inspect_container.side_effect = [("current", 1), replacement]
    with pytest.raises(RuntimeError, match="head container"):
        infer(http)
    assert http.open.call_count == 1 and clock.now == 1


def test_model_discovery_retries_share_the_inference_deadline(clock):
    http = Mock()
    http.open.side_effect = [http_error(503), io.BytesIO(b'{"data":[{"id":"alias"}]}'), io.BytesIO(text_event())]
    result = infer(http, model=None, inference_timeout_s=5)
    assert result["model"] == "alias" and result["inference_attempts"] == 2
    assert [call.kwargs["timeout"] for call in http.open.call_args_list] == [5, 3, 3]


def test_slow_discovery_leaves_only_remaining_budget_for_inference(clock):
    http = Mock()

    def respond(*args, **kwargs):
        if http.open.call_count == 1:
            clock.sleep(4)
            return io.BytesIO(b'{"data":[{"id":"alias"}]}')
        return io.BytesIO(text_event())

    http.open.side_effect = respond
    assert infer(http, model=None, inference_timeout_s=5)["inference_wait_s"] == 4
    assert [call.kwargs["timeout"] for call in http.open.call_args_list] == [5, 1]


@pytest.mark.parametrize("stage", ["open", "read", "socket_timeout"])
def test_late_response_cannot_succeed_or_lose_last_retry_diagnostic(clock, stage):
    http = Mock()

    class LateStream(io.BytesIO):
        def readline(self, *args):
            clock.sleep(4)
            return super().readline(*args)

    response = LateStream(text_event()) if stage == "read" else io.BytesIO(text_event())

    def respond(*args, **kwargs):
        if http.open.call_count == 1:
            raise http_error(503)
        if stage in {"open", "socket_timeout"}:
            clock.sleep(4)
        if stage == "socket_timeout":
            raise urllib.error.URLError(TimeoutError("timed out"))
        return response

    http.open.side_effect = respond
    with pytest.raises(TimeoutError, match="after 2 attempt.*last transient error: HTTP 503"):
        infer(http, inference_timeout_s=5)
    assert http.open.call_count == 2
    if stage != "socket_timeout":
        assert response.closed


@pytest.fixture
def local_probe(monkeypatch):
    """Run the actual shipped script; stub only Docker host observations."""
    source = Path(probe.__file__).read_text()
    source += "\nverify_host_observer = lambda: None\ninspect_container = lambda name: ('current', 1)\n"
    resource = Mock()
    resource.joinpath.return_value.read_text.return_value = source
    monkeypatch.setattr(startup, "files", lambda package: resource)
    monkeypatch.setattr("sparkrun.orchestration.ssh.should_run_locally", lambda *args: True)
    processes = []
    original = subprocess.Popen

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(startup.subprocess, "Popen", spawn)
    yield
    assert processes and all(process.poll() is not None for process in processes)


@pytest.fixture
def server():
    state = SimpleNamespace(mode="retry", posts=[], release=threading.Event())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            state.posts.append(self.path)
            if state.mode == "retry" and len(state.posts) == 1:
                self.send_response(503)
                self.end_headers()
                return
            try:
                if state.mode == "headers":
                    # Never finish the HTTP headers, but keep the socket active.
                    self.wfile.write(b"HTTP/1.0 200 OK\r\n")
                    while not state.release.wait(0.05):
                        self.wfile.write(b"X-Keepalive: 1\r\n")
                        self.wfile.flush()
                    return
                self.send_response(200)
                self.end_headers()
                if state.mode == "drip":
                    # No newline: the child's readline() cannot check its clock.
                    while not state.release.wait(0.05):
                        self.wfile.write(b":")
                        self.wfile.flush()
                    return
                if state.mode == "pending":
                    state.release.wait(0.4)
                self.wfile.write(text_event())
            except (BrokenPipeError, ConnectionResetError):
                pass

    instance = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.port = instance.server_port
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        state.release.set()
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def test_real_http_retry_and_subprocess_progress(local_probe, server, caplog):
    caplog.set_level(logging.INFO, logger=startup.__name__)
    value = startup.run_probe("localhost", config(port=server.port, inference_timeout_s=5))
    assert value["inference_attempts"] == 2 and value["inference_wait_s"] >= 2
    assert len(server.posts) == 2
    assert caplog.text.count("Inference readiness retry:") == 1
    assert "HTTP 503" in caplog.text


def test_pending_inference_is_not_replayed(local_probe, server):
    server.mode = "pending"
    value = startup.run_probe("localhost", config(port=server.port, inference_timeout_s=3))
    assert value["inference_attempts"] == 1 and value["inference_wait_s"] >= 0.4
    assert len(server.posts) == 1


@pytest.mark.parametrize("mode", ["headers", "drip"])
def test_subprocess_bounds_stalled_io_to_inference_budget(local_probe, server, mode):
    server.mode = mode
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="inference readiness exceeded its deadline"):
        startup.run_probe("localhost", config(port=server.port, inference_timeout_s=0.4))
    # The 100s port/health budgets cannot extend inference. Includes process
    # startup, supervisor polling, and cleanup tolerance for shared CI runners.
    assert time.monotonic() - started < 2
    assert len(server.posts) == 1


@pytest.mark.parametrize("mode", ["retry", "drip"])
def test_cancellation_interrupts_backoff_and_pending_stream(local_probe, server, mode):
    server.mode = mode
    cancel = threading.Event()
    timer = threading.Timer(0.6, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(InterruptedError):
            startup.run_probe("localhost", config(port=server.port), cancel=cancel)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 2 and len(server.posts) == 1
