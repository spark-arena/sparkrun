"""Same-host startup measurements: one probe, exact container, strategy reuse."""

import io
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sparkrun.core.launcher import wait_for_serve_ready
from sparkrun.core.config import SparkrunConfig
from sparkrun.orchestration import startup
from sparkrun.scripts import startup_probe
from sparkrun.core.readiness import OPENAI_CHAT_STREAM
from sparkrun.orchestration.executors.docker import DockerExecutor


@pytest.fixture(autouse=True)
def isolated_probe_environment(monkeypatch):
    # Unit tests stub Docker identity below. Subprocess/Docker contract tests
    # execute the packaged source in a fresh interpreter and verify the host.
    monkeypatch.setattr(startup_probe, "verify_host_observer", lambda: None)


def observation():
    return {
        "format": 1,
        "measurement": "rank0-acceptance-v1",
        "observer": "rank0",
        "container_id": "current",
        "container_started_unix_ns": 1_000_000_000,
        "port_open_unix_ns": 2_000_000_000,
        "http_ready_unix_ns": 3_000_000_000,
        "first_token_unix_ns": 4_000_000_000,
        "first_token_field": "content",
        "inference_ready": True,
        "response_validated": True,
    }


def launch():
    settings = SparkrunConfig.__new__(SparkrunConfig)
    settings._data = {"readiness": {"inference_timeout_s": 10, "inference_prompt": "Reply OK"}}
    return SimpleNamespace(
        runtime=SimpleNamespace(
            get_family=lambda: "vllm",
            readiness_styles=(OPENAI_CHAT_STREAM,),
            readiness_health_path="/health",
            get_head_container_name=lambda *a, **kw: "head",
            executor=DockerExecutor(),
        ),
        cluster_id="job",
        host_list=["localhost"],
        is_solo=True,
        serve_port=8000,
        timeline=None,
        runtime_info={},
        startup_observation={},
        config=settings,
        recipe=SimpleNamespace(readiness={}),
        overrides={},
    )


def test_strategy_receipt_does_not_issue_another_request():
    result = launch()
    result.startup_observation = observation()
    with patch.object(startup, "run_probe") as send:
        ready = wait_for_serve_ready(result)
    send.assert_not_called()
    assert ready.ready
    assert ready.startup_observation["first_token_unix_ns"] == 4_000_000_000


def test_normal_launch_runs_one_probe_and_reuses_the_result():
    result = launch()
    with patch.object(startup, "run_probe", return_value=observation()) as send:
        assert wait_for_serve_ready(result).ready
        assert wait_for_serve_ready(result).ready
    assert send.call_count == 1


def test_cancelled_or_failed_probe_never_signals_ready():
    for error, reason in ((InterruptedError(), "cancelled"), (RuntimeError("empty stream"), "inference")):
        with patch.object(startup, "run_probe", side_effect=error):
            ready = wait_for_serve_ready(launch())
        assert not ready.ready and ready.reason == reason
    cancel = threading.Event()
    cancel.set()
    with patch.object(startup, "run_probe") as send:
        assert wait_for_serve_ready(launch(), cancel=cancel).reason == "cancelled"
    send.assert_not_called()


def test_legacy_strategy_acceptance_disables_duplicate_inference():
    result = launch()
    result.runtime_info["inference_readiness"] = "accepted"
    with patch("sparkrun.core.launcher.wait_for_endpoint_ready", return_value="legacy") as legacy:
        assert wait_for_serve_ready(result) == "legacy"
    legacy.assert_called_once()


@pytest.mark.parametrize(
    "key,value",
    [
        ("first_token_unix_ns", 0),
        ("inference_ready", False),
        ("first_token_field", "role"),
        ("measurement", "unknown"),
        ("container_id", ""),
    ],
)
def test_invalid_observation_is_rejected(key, value):
    data = observation()
    data[key] = value
    with pytest.raises(ValueError):
        startup.validate_observation(data)


def test_probe_ignores_empty_deltas_and_counts_reasoning_once():
    config = {
        "container": "head",
        "port": 8000,
        "address": "127.0.0.1",
        "port_timeout_s": 1,
        "health_timeout_s": 1,
        "inference_timeout_s": 1,
        "prompt": "Reply OK",
    }
    health = io.BytesIO()
    health.status = 200
    models = io.BytesIO(b'{"data":[{"id":"served-alias"}]}')
    events = io.BytesIO(
        b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\ndata: {"choices":[{"delta":{"reasoning":"thinking"}}]}\n\n'
    )
    http = Mock()
    http.open.side_effect = [health, models, events]
    with (
        patch.object(startup_probe, "inspect_container", return_value=("current", 1)),
        patch.object(startup_probe, "port_listening", return_value=True),
        patch.object(startup_probe.urllib.request, "build_opener", return_value=http),
    ):
        data = startup_probe.observe(config)
    requests = [call.args[0] for call in http.open.call_args_list if not isinstance(call.args[0], str)]
    assert len(requests) == 1
    assert json.loads(requests[0].data)["model"] == "served-alias"
    assert data["inference_ready"] and data["first_token_field"] == "reasoning"
    assert data["response_validated"] is False
    assert data["container_started_unix_ns"] == 1


def test_precancelled_probe_does_not_spawn_a_process():
    cancel = threading.Event()
    cancel.set()
    with patch.object(startup.subprocess, "Popen") as process:
        with pytest.raises(InterruptedError):
            startup.run_probe("localhost", {}, cancel=cancel)
    process.assert_not_called()


def config(port=8000, **extra):
    return {
        "container": "head",
        "port": port,
        "address": "127.0.0.1",
        "port_timeout_s": 2,
        "health_timeout_s": 2,
        "inference_timeout_s": 2,
        "prompt": "Reply OK",
        **extra,
    }


@pytest.mark.parametrize(
    "events",
    [
        b"data: [DONE]\n\n",
        b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"wrong"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        b'data: {"error":"unavailable"}\n\n',
        b'data: {"choices":[{"delta":42}]}\n\n',
        b"data: " + b"x" * 65536,
    ],
)
def test_full_validation_rejects_empty_truncated_wrong_or_invalid_stream(events):
    health = io.BytesIO()
    health.status = 200
    http = Mock()
    http.open.side_effect = [health, io.BytesIO(b'{"data":[{"id":"alias"}]}'), io.BytesIO(events)]
    with (
        patch.object(startup_probe, "inspect_container", return_value=("current", 1)),
        patch.object(startup_probe, "port_listening", return_value=True),
        patch.object(startup_probe.urllib.request, "build_opener", return_value=http),
    ):
        with pytest.raises(RuntimeError):
            startup_probe.observe(config(expected="OK"))


@pytest.fixture
def streaming_server():
    posts = []
    finished = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            if self.path == "/v1/models":
                self.wfile.write(b'{"data":[{"id":"served-alias"}]}')

        def do_POST(self):
            posts.append((json.loads(self.rfile.read(int(self.headers["Content-Length"]))), self.headers))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                self.wfile.write(b': comment\n\ndata: {"choices":[{"delta":{"content":""}}]}\n\n')
                self.wfile.write(b'data: {"choices":\ndata: [{"delta":{"reasoning_content":"thinking"}}]}\n\n')
                self.wfile.flush()
                time.sleep(0.05)
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\n')
                finished.append(time.time_ns())
                self.wfile.write(b'data: {"choices":[],"usage":{"completion_tokens":2}}\n\ndata: [DONE]\n\n')
            except (BrokenPipeError, ConnectionResetError):
                pass  # Ordinary readiness intentionally closes after first text.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, posts, finished
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_stream_records_first_text_but_validates_complete_response(streaming_server):
    port, posts, finished = streaming_server
    with patch.object(startup_probe, "inspect_container", return_value=("current", 1)):
        data = startup_probe.observe(config(port, expected="OK", api_key="test-secret"))
    assert len(posts) == 1
    assert posts[0][0]["stream"] is True
    assert posts[0][1]["Authorization"] == "Bearer test-secret"
    assert data["first_token_field"] == "reasoning_content"
    assert data["first_token_unix_ns"] < finished[0]
    assert data["response_validated"] and data["content"] == "OK"
    assert data["port_open_unix_ns"] <= data["http_ready_unix_ns"] < data["first_token_unix_ns"]
    assert "test-secret" not in json.dumps(data)


def test_container_replacement_invalidates_observation(streaming_server):
    port, _, _ = streaming_server
    with patch.object(startup_probe, "inspect_container", side_effect=[("original", 1), ("replacement", 2)]):
        with pytest.raises(RuntimeError, match="changed"):
            startup_probe.observe(config(port))


def test_running_probe_cancels_promptly_without_leaving_local_process():
    cancel = threading.Event()
    processes = []
    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    resource = Mock()
    resource.joinpath.return_value.read_text.return_value = "import time\ndef observe(config): time.sleep(30)\nimport json\n"
    timer = threading.Timer(0.3, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with (
            patch.object(startup, "files", return_value=resource),
            patch.object(startup.subprocess, "Popen", side_effect=spawn),
            patch("sparkrun.orchestration.ssh.should_run_locally", return_value=True),
        ):
            with pytest.raises(InterruptedError):
                startup.run_probe("localhost", config(), cancel=cancel)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 3
    assert processes and processes[0].poll() is not None


def test_real_docker_start_and_subprocess_probe(streaming_server):
    image = os.environ.get("SPARKRUN_STARTUP_TEST_DOCKER_IMAGE")
    if not image:
        pytest.skip("set SPARKRUN_STARTUP_TEST_DOCKER_IMAGE for the local Docker contract test")
    container = subprocess.check_output(["docker", "run", "-d", "--network=host", image, "sleep", "60"], text=True).strip()
    try:
        port, posts, finished = streaming_server
        with patch("sparkrun.orchestration.ssh.should_run_locally", return_value=True):
            value = startup.run_probe("localhost", config(port, container=container, expected="OK"))
        assert value["container_id"] == container
        assert value["container_started_unix_ns"] < value["first_token_unix_ns"] < finished[0]
        assert len(posts) == 1 and value["response_validated"]
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=True, capture_output=True)


def test_runtime_api_key_uses_existing_resolution_without_entering_observation():
    result = launch()
    result.recipe, result.overrides = object(), {"api_key": "test-secret"}
    result.runtime.resolve_api_key = Mock(return_value="test-secret")
    with patch.object(startup, "run_probe", return_value=observation()) as send:
        ready = wait_for_serve_ready(result)
    result.runtime.resolve_api_key.assert_called_once_with(result.recipe, result.overrides)
    assert send.call_args.args[1]["api_key"] == "test-secret"
    assert "test-secret" not in json.dumps(ready.startup_observation)


def test_precancelled_remote_launch_performs_no_address_detection():
    result = launch()
    result.host_list = ["remote.invalid"]
    cancel = threading.Event()
    cancel.set()
    with patch("sparkrun.orchestration.primitives.detect_host_ip") as detect:
        assert wait_for_serve_ready(result, cancel=cancel).reason == "cancelled"
    detect.assert_not_called()


def test_startup_spans_are_foreign_overlapping_and_not_duplicated(capsys):
    from sparkrun.cli._run import _echo_endpoint_ready
    from sparkrun.core.timing import Timeline

    result = launch()
    result.timeline = Timeline()
    data = observation()
    data.update(port_wait_s=0.2, health_wait_s=0.3)
    with patch.object(startup, "run_probe", return_value=data):
        ready = wait_for_serve_ready(result)
        wait_for_serve_ready(result)
    spans = result.timeline.export()["spans"]
    assert len(spans) == 3
    assert all(span["clock"] == "host:localhost" for span in spans)
    assert {span["name"] for span in spans} == {"serve.startup_port_open", "serve.startup_http_ready", "serve.startup_ttft"}
    assert ready.total_wait_s == 0.5  # Wait duration, not Docker-start TTFT.
    _echo_endpoint_ready(ready)
    output = capsys.readouterr().err
    assert "TTR port-open 1.000s, HTTP-ready 2.000s, TTFT 3.000s" in output
    assert "rank0-acceptance-v1" in output
    # Overlapping spans are marked so a consumer summing the tree's siblings
    # cannot silently report a 6s startup for a 3s one.
    assert all(span["attrs"]["composition"] == "non_additive" for span in spans)


def test_strategy_receipt_reaches_the_timeline():
    """A receipt sparkrun did not measure is still the launch's own timing.

    Suppressing it left ColdSnap launches with no startup rows at all in the
    timing tree, the diagnostics record or the benchmark artifact.
    """
    from sparkrun.core.timing import Timeline

    result = launch()
    result.timeline = Timeline()
    result.startup_observation = observation()
    with patch.object(startup, "run_probe") as send:
        assert wait_for_serve_ready(result).ready
        wait_for_serve_ready(result)  # Reused, not re-recorded.
    send.assert_not_called()
    spans = result.timeline.export()["spans"]
    assert {span["name"] for span in spans} == {"serve.startup_port_open", "serve.startup_http_ready", "serve.startup_ttft"}
    assert len(spans) == 3


def test_startup_readiness_block_repeats_the_live_line():
    from sparkrun.utils.cli_formatters import format_startup_readiness, startup_readiness_durations

    data = observation()
    block = format_startup_readiness(data, host="spark-01")
    assert "rank 0 on spark-01, elapsed from container start" in block
    assert "measurement: rank0-acceptance-v1" in block
    for label, value in (("TTR port-open", "1.000s"), ("TTR HTTP-ready", "2.000s"), ("TTFT", "3.000s")):
        assert any(line.strip().startswith(label) and line.endswith(value) for line in block.splitlines())
    # Same projection as the live line, so the two cannot disagree.
    assert startup_readiness_durations(data) == {"port_open": "1.000s", "http_ready": "2.000s", "ttft": "3.000s"}
    # A legacy endpoint wait measures nothing from container start.
    assert format_startup_readiness(None) == ""
    assert format_startup_readiness({}) == ""


def test_startup_spans_leave_the_tree_but_stay_in_the_export():
    """Rendered once, in the block built for them; never as tree rows.

    The tree's rows sum to its total and these three are not terms in it, so
    a row here would show the same figure twice and break the one property
    the tree has.
    """
    from sparkrun.core.timing import ROOT, Timeline
    from sparkrun.utils.cli_formatters import STARTUP_SPAN_NAMES, format_launch_timings

    timeline = Timeline()
    with timeline.span("run", label="run"):
        pass
    timeline.add_span("serve.startup_ttft", clock="host:h1", duration_s=46.4, parent=ROOT, composition="non_additive")
    export = timeline.export()
    tree = format_launch_timings(export, omit=STARTUP_SPAN_NAMES)
    assert "serve.startup_ttft" not in tree
    assert "run" in tree
    # Display only: the diagnostics record and benchmark metadata read this.
    assert "serve.startup_ttft" in {span["name"] for span in export["spans"]}
    assert "serve.startup_ttft" in format_launch_timings(export)


def test_startup_readiness_block_reports_unobserved_metrics_honestly():
    from sparkrun.utils.cli_formatters import format_startup_readiness

    endpoint_only = format_startup_readiness(endpoint_observation())
    assert "not applicable (inference disabled)" in endpoint_only
    partial = observation()
    partial.pop("http_ready_unix_ns")
    assert "unavailable" in format_startup_readiness(partial)
    assert "0.000s" not in format_startup_readiness(partial)


def test_inference_configuration_defaults_and_overrides(tmp_path):
    from sparkrun.core.config import SparkrunConfig

    path = tmp_path / "config.yaml"
    path.write_text("{}")
    settings = SparkrunConfig(config_path=path)
    assert settings.readiness_inference_enabled
    assert settings.readiness_inference_timeout_s == 1200
    assert settings.readiness_inference_prompt == "Reply with exactly: sparkrun-ready"
    path.write_text('readiness:\n  inference: false\n  inference_timeout_s: 42\n  inference_prompt: "Reply OK"\n')
    settings = SparkrunConfig(config_path=path)
    assert not settings.readiness_inference_enabled
    assert settings.readiness_inference_timeout_s == 42
    assert settings.readiness_inference_prompt == "Reply OK"


@pytest.mark.parametrize("change", ["other-family", "other-executor", "dry-run"])
def test_unsupported_or_opted_out_launch_uses_legacy_endpoint_check(change):
    result = launch()
    if change == "other-family":
        result.runtime.readiness_styles = ()
    elif change == "other-executor":
        result.runtime.executor = SimpleNamespace(executor_name="k8s")
    with (
        patch("sparkrun.core.launcher.wait_for_endpoint_ready", return_value="legacy") as legacy,
        patch.object(startup, "run_probe") as send,
    ):
        assert wait_for_serve_ready(result, dry_run=(change == "dry-run")) == "legacy"
    send.assert_not_called()
    legacy.assert_called_once()


def endpoint_observation():
    value = observation()
    value.update(
        measurement="sparkrun-rank0-v1", inference_requested=False, endpoint_ready=True, inference_ready=False, response_validated=False
    )
    value.pop("first_token_unix_ns")
    value.pop("first_token_field")
    return value


def test_recipe_disables_inference_but_preserves_ttr_and_reuses_observation(capsys):
    from sparkrun.cli._run import _echo_endpoint_ready
    from sparkrun.core.timing import Timeline

    result = launch()
    result.recipe.readiness = {"inference": False, "health_timeout_s": 35}
    result.timeline = Timeline()
    with patch.object(startup, "run_probe", return_value=endpoint_observation()) as probe:
        ready = wait_for_serve_ready(result)
        assert wait_for_serve_ready(result).ready
    assert probe.call_count == 1
    assert probe.call_args.args[1]["inference"] is False
    assert probe.call_args.args[1]["health_timeout_s"] == 35
    assert probe.call_args.args[1]["inference_timeout_s"] == 10
    assert {span["name"] for span in result.timeline.export()["spans"]} == {"serve.startup_port_open", "serve.startup_http_ready"}
    _echo_endpoint_ready(ready)
    output = capsys.readouterr().err
    assert "Endpoint ready" in output and "Inference ready" not in output
    assert "TTR port-open 1.000s, HTTP-ready 2.000s" in output
    assert "TTFT not applicable (inference disabled)" in output


def test_effective_config_reaches_probe_without_overwriting_global_defaults():
    result = launch()
    result.config.set("readiness", {"inference": False, "port_timeout_s": 2500, "inference_prompt": "global"})
    result.recipe.readiness = {"inference": True, "inference_prompt": "recipe", "inference_timeout_s": 47}
    with patch.object(startup, "run_probe", return_value=observation()) as probe:
        assert wait_for_serve_ready(result).ready
    sent = probe.call_args.args[1]
    assert sent["inference"] is True and sent["prompt"] == "recipe"
    assert sent["port_timeout_s"] == 2500 and sent["health_timeout_s"] == 900
    assert sent["inference_timeout_s"] == 47
    assert result.config.readiness_inference_enabled is False
    assert result.config.readiness_inference_prompt == "global"


@pytest.mark.parametrize(
    "change",
    [{"endpoint_ready": False}, {"response_validated": True}, {"first_token_unix_ns": 4_000_000_000}, {"http_ready_unix_ns": None}],
)
def test_endpoint_only_observation_cannot_claim_inference_or_missing_health(change):
    with pytest.raises(ValueError):
        startup.validate_observation({**endpoint_observation(), **change})


def test_endpoint_only_receipt_cannot_satisfy_required_inference():
    with pytest.raises(ValueError):
        startup.validate_observation(endpoint_observation(), require_inference=True)


def test_real_endpoint_only_probe_sends_no_model_or_chat_request(streaming_server):
    port, posts, _ = streaming_server
    with patch.object(startup_probe, "inspect_container", return_value=("current", 1)):
        value = startup_probe.observe(config(port, inference=False))
    assert startup.validate_observation(value) == value
    assert value["endpoint_ready"] and not value["inference_ready"]
    assert "first_token_unix_ns" not in value and "model" not in value
    assert value["port_open_unix_ns"] <= value["http_ready_unix_ns"]
    assert posts == []


def test_endpoint_only_container_replacement_still_fails(streaming_server):
    port, _, _ = streaming_server
    with patch.object(startup_probe, "inspect_container", side_effect=[("old", 1), ("new", 2)]):
        with pytest.raises(RuntimeError, match="changed"):
            startup_probe.observe(config(port, inference=False))


def test_endpoint_only_probe_does_not_query_model_or_ignore_failed_health():
    health = io.BytesIO()
    health.status = 200
    http = Mock()
    http.open.return_value = health
    with (
        patch.object(startup_probe, "inspect_container", return_value=("current", 1)),
        patch.object(startup_probe, "port_listening", return_value=True),
        patch.object(startup_probe.urllib.request, "build_opener", return_value=http),
    ):
        startup_probe.observe(config(inference=False))
        http.open.assert_called_once_with("http://127.0.0.1:8000/health", timeout=1)
        http.open.side_effect = OSError("not healthy")
        with pytest.raises(TimeoutError, match="health"):
            startup_probe.observe(config(inference=False, health_timeout_s=0))


def test_real_docker_endpoint_only_probe_with_unbounded_budgets(streaming_server):
    image = os.environ.get("SPARKRUN_STARTUP_TEST_DOCKER_IMAGE")
    if not image:
        pytest.skip("set SPARKRUN_STARTUP_TEST_DOCKER_IMAGE for the local Docker contract test")
    container = subprocess.check_output(["docker", "run", "-d", "--network=host", image, "sleep", "60"], text=True).strip()
    try:
        port, posts, _ = streaming_server
        with patch("sparkrun.orchestration.ssh.should_run_locally", return_value=True):
            value = startup.run_probe(
                "localhost", config(port, container=container, inference=False, port_timeout_s=float("inf"), health_timeout_s=float("inf"))
            )
        assert value["container_id"] == container and value["endpoint_ready"]
        assert not value["inference_ready"] and "first_token_unix_ns" not in value
        assert value["container_started_unix_ns"] <= value["port_open_unix_ns"] <= value["http_ready_unix_ns"]
        assert posts == []
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=True, capture_output=True)


def test_disabling_sparkrun_probe_does_not_discard_coldsnap_acceptance():
    result = launch()
    result.recipe.readiness = {"inference": False}
    result.startup_observation = observation()
    with patch.object(startup, "run_probe") as probe:
        ready = wait_for_serve_ready(result)
    assert ready.ready and ready.startup_observation["response_validated"]
    probe.assert_not_called()
