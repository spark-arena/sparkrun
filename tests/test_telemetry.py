"""Tests for anonymous sparkrun telemetry."""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml
from click.testing import CliRunner

from _telemetry_guard import install_telemetry_blocker
import sparkrun.api as api
from sparkrun.cli import main
from sparkrun.core.cluster_manager import ClusterDefinition
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.hardware import AcceleratorSpec, HostHardware
from sparkrun.core.recipe import Recipe
from sparkrun.telemetry.benchmark import build_benchmark_event
from sparkrun.telemetry.client import prepare_event, send_event
from sparkrun.telemetry.config import (
    TELEMETRY_AUTH_HEADER,
    ensure_installation_id,
    telemetry_enabled,
)
from sparkrun.telemetry.events import build_run_event, build_setup_wizard_event
from sparkrun.telemetry.util import int_value, string_value


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        return b""


def test_telemetry_enabled_prefers_env_over_config(tmp_path, monkeypatch):
    monkeypatch.delenv("SPARKRUN_NO_TELEMETRY", raising=False)
    config = SparkrunConfig(tmp_path / "config.yaml")

    assert telemetry_enabled(config) is True

    config.set("telemetry.enabled", False)
    config.save()
    config = SparkrunConfig(tmp_path / "config.yaml")
    assert telemetry_enabled(config) is False

    monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "0")
    assert telemetry_enabled(config) is True


def test_installation_id_is_persistent(tmp_path, monkeypatch):
    monkeypatch.delenv("SPARKRUN_NO_TELEMETRY", raising=False)
    config = SparkrunConfig(tmp_path / "config.yaml")

    first = ensure_installation_id(config)
    second = ensure_installation_id(SparkrunConfig(tmp_path / "config.yaml"))

    assert first == second
    assert first


def test_send_event_posts_json_with_auth_header(tmp_path, monkeypatch):
    monkeypatch.delenv("SPARKRUN_NO_TELEMETRY", raising=False)
    monkeypatch.setenv("SPARKRUN_TELEMETRY_ENDPOINT", "https://telemetry.test/")
    monkeypatch.setenv("SPARKRUN_TELEMETRY_KEY", "secret-key")
    config = SparkrunConfig(tmp_path / "config.yaml")
    calls = []

    def _urlopen(request, timeout):
        calls.append((request, timeout))
        return _FakeResponse()

    monkeypatch.setattr("sparkrun.telemetry.client.urlopen", _urlopen)

    send_event(config, {"event_type": "unit_test", "ok": True})

    assert len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url == "https://telemetry.test/"
    headers = {key.lower(): value for key, value in request.headers.items()}
    assert headers[TELEMETRY_AUTH_HEADER] == "secret-key"
    assert timeout == 0.75
    body = json.loads(request.data.decode("utf-8"))
    assert body["event_type"] == "unit_test"
    assert body["installation_id"]


def test_prepare_event_returns_none_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "1")
    config = SparkrunConfig(tmp_path / "config.yaml")

    assert prepare_event(config, {"event_type": "unit_test"}) is None


def test_telemetry_package_exports_supported_emitters():
    import sparkrun.telemetry as telemetry

    assert telemetry.__all__ == [
        "emit_benchmark_telemetry",
        "emit_run_telemetry",
        "emit_setup_wizard_event",
        "emit_update_event",
    ]


def test_run_event_has_anonymous_source_hardware_and_parallelism():
    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "org/model", "defaults": {"tensor_parallel": 2}})
    recipe.source_registry = "official"
    recipe.source_registry_url = "https://github.com/spark-arena/recipe-registry.git"
    recipe.metadata.update({"quantization": "awq", "quant_bits": 4, "model_dtype": "awq4", "kv_dtype": "fp8"})
    cluster = ClusterDefinition(
        name="lab",
        hosts=["h1", "h2"],
        hosts_hardware={
            "h1": HostHardware([AcceleratorSpec(vendor="nvidia", model="gb10", count=1)]),
            "h2": HostHardware([AcceleratorSpec(vendor="nvidia", model="gb10", count=1)]),
        },
    )
    result = api.RunResult(
        cluster_id="sparkrun_deadbeef_deadbeef",
        host_list=("h1", "h2"),
        placement=None,
        scheduler="greedy",
        runtime="vllm",
        executor="docker",
        started_at=time.time(),
        dry_run=True,
        is_solo=False,
        rc=0,
        metadata={},
    )
    options = api.RunOptions(recipe=recipe, hosts=("h1", "h2"), overrides={})

    with patch("sparkrun.models.vram.fetch_model_visibility", return_value="public"):
        event = build_run_event(result=result, recipe=recipe, cluster=cluster, options=options)

    assert event["event_type"] == "run"
    assert event["model"] == "org/model"
    assert event["parallelism"]["tensor_parallel"] == 2
    assert event["cluster"]["node_count"] == 2
    assert event["cluster"]["gpu_count"] == 2
    assert event["recipe_source"]["from_default_registry"] is True
    assert event["model_quantization"] == {"quantization": "awq", "quant_bits": 4, "model_dtype": "awq4", "kv_dtype": "fp8"}
    assert "h1" not in json.dumps(event)


def test_setup_wizard_event_maps_step_choices():
    event = build_setup_wizard_event(
        wizard_run_kind="initial_setup",
        results={"cluster": "default (2 hosts, default)", "ssh": "skipped", "docker": "failed"},
        cluster_node_count=2,
        dry_run=False,
        cx7_detected=True,
    )

    choices = {entry["step"]: entry["choice"] for entry in event["step_choices"]}
    assert choices["cluster"] == "already_configured"
    assert choices["ssh"] == "opted_out"
    assert choices["docker"] == "failed"
    assert choices["earlyoom"] == "skipped"


def test_benchmark_event_is_anonymous_and_low_cardinality():
    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "org/model"})
    recipe.source_path = "/home/drew/private/recipe.yaml"
    recipe.metadata.update({"quantization": "nvfp4", "quant_bits": 4, "model_dtype": "nvfp4", "kv_dtype": "fp8"})
    result = api.BenchmarkResult(
        success=True,
        benchmark_id="bench_secret",
        category="performance",
        framework="llama-benchy",
        profile="quick",
        results={"throughput": 100, "latency": {"p50": 1.2}},
        outputs={"json": "/home/drew/private/out.json"},
        cluster_id="sparkrun_secret_cluster",
        host_list=("host-a", "host-b"),
        container_image="registry/private:tag",
        container_image_sha="sha256:secret",
        container_image_sha_pinned=True,
        container_image_longterm_ref="registry/private@sha256:secret",
        container_image_longterm_pinned=True,
        metadata={"bench_args": {"depth": 4, "prompt_file": "/home/drew/private/prompts.txt"}},
        state_dir="/home/drew/private/state",
        resumed=True,
        submission_id="sub-secret",
    )
    options = api.BenchmarkOptions(
        recipe=recipe,
        arena=True,
        dry_run=True,
        overrides={"tensor_parallel": 2},
    )

    event = build_benchmark_event(result=result, options=options)

    assert event["event_type"] == "benchmark"
    assert event["success"] is True
    assert event["host_count"] == 2
    assert event["result_keys"] == ["latency", "throughput"]
    assert event["output_formats"] == ["json"]
    assert event["bench_arg_keys"] == ["depth", "prompt_file"]
    assert event["recipe_source"]["from_file"] is True
    assert event["parallelism"]["tensor_parallel"] == 2
    assert event["model_quantization"] == {"quantization": "nvfp4", "quant_bits": 4, "model_dtype": "nvfp4", "kv_dtype": "fp8"}
    assert event["submission_id_present"] is True
    payload = json.dumps(event, sort_keys=True)
    for private_value in ("host-a", "host-b", "bench_secret", "sparkrun_secret", "sub-secret", "/home/drew/private"):
        assert private_value not in payload


class _Flavor(Enum):
    PLAIN = "plain"


class _Opaque:
    def __str__(self):
        return "leaked-repr"


def test_string_value_renders_scalars_only():
    assert string_value("  vllm  ") == "vllm"
    assert string_value(4) == "4"
    assert string_value(0.5) == "0.5"
    assert string_value(True) == "True"
    assert string_value(Path("/tmp/x")) == "/tmp/x"
    assert string_value(_Flavor.PLAIN) == "plain"

    assert string_value(None) is None
    assert string_value("   ") is None
    # Anything else is dropped rather than rendered through ``str()`` --
    # including objects that define a perfectly innocent ``__str__``.
    assert string_value(_Opaque()) is None
    assert string_value(["a"]) is None
    assert string_value({"a": 1}) is None
    assert string_value(len) is None


def test_string_value_never_renders_a_test_double():
    """The regression: mock reprs reached the collector as event dimensions."""
    mock = MagicMock()
    assert string_value(mock) is None
    assert string_value(mock.category) is None
    # A Mock satisfies os.PathLike (it synthesizes __fspath__), so the path case
    # must be typed on PurePath rather than the protocol.
    assert string_value(mock.__fspath__) is None


def test_int_value_does_not_coerce_arbitrary_objects():
    assert int_value("4") == 4
    assert int_value(4.9) == 4
    assert int_value(True) == 1
    assert int_value(None, default=7) == 7
    assert int_value("nope", default=7) == 7
    # ``int(MagicMock())`` is 1 -- a confident, entirely fabricated number.
    assert int_value(MagicMock(), default=None) is None


def test_benchmark_event_drops_mock_dimensions():
    """Reproduces the shape of the events that reached the live collector."""
    mock = MagicMock()

    event = build_benchmark_event(result=mock, options=mock, recipe=mock)

    assert "MagicMock" not in json.dumps(event, sort_keys=True, default=str)
    assert event["category"] is None
    assert event["framework"] is None
    assert event["profile"] is None
    assert event.get("model") is None


def test_run_event_drops_mock_dimensions():
    mock = MagicMock()

    event = build_run_event(result=mock, recipe=mock, cluster=mock, options=mock)

    assert "MagicMock" not in json.dumps(event, sort_keys=True, default=str)
    assert event["runtime"] is None
    assert event["executor"] is None
    assert event["scheduler"] is None


def test_telemetry_fails_closed_on_anything_but_a_real_config(tmp_path, monkeypatch):
    monkeypatch.delenv("SPARKRUN_NO_TELEMETRY", raising=False)

    assert telemetry_enabled(SparkrunConfig(tmp_path / "config.yaml")) is True
    assert telemetry_enabled(MagicMock()) is False
    assert telemetry_enabled(SimpleNamespace(get=lambda key, default=None: None)) is False
    assert telemetry_enabled(None) is False

    # Forcing telemetry on must not resurrect a config we do not recognize.
    monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "0")
    assert telemetry_enabled(MagicMock()) is False


def test_emit_sends_nothing_when_handed_a_mock_config(tmp_path, monkeypatch):
    """The vector behind every event that reached the live collector."""
    from sparkrun.telemetry import emit_benchmark_telemetry

    attempts = install_telemetry_blocker(monkeypatch)
    monkeypatch.delenv("SPARKRUN_NO_TELEMETRY", raising=False)

    emit_benchmark_telemetry(MagicMock(), result=MagicMock(), options=MagicMock())

    assert attempts == []


def test_test_suite_telemetry_blocker_records_despite_emit_swallowing(tmp_path, monkeypatch):
    """The suite-wide guard must survive ``emit_*``'s blanket ``except Exception``.

    Raising from the patched ``urlopen`` alone would be logged at DEBUG and
    forgotten; the recorded attempt is what fails the test in
    ``isolate_stateful``'s teardown.  Installing a fresh blocker here shadows
    that one, so this test does not trip its own guard.
    """
    from sparkrun.telemetry import emit_benchmark_telemetry

    attempts = install_telemetry_blocker(monkeypatch)
    monkeypatch.delenv("SPARKRUN_NO_TELEMETRY", raising=False)

    # A real config on the enabled path -- a mock one no longer sends at all.
    emit_benchmark_telemetry(SparkrunConfig(tmp_path / "config.yaml"), result=MagicMock(), options=MagicMock())

    assert len(attempts) == 1, "the blocker must record the send emit_* swallowed"


def test_setup_telemetry_command_persists_preference(tmp_path, monkeypatch):
    import sparkrun.core.config as config_module

    monkeypatch.delenv("SPARKRUN_NO_TELEMETRY", raising=False)
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_DIR", tmp_path / "config")

    runner = CliRunner()
    disabled = runner.invoke(main, ["setup", "telemetry", "--disable"])
    assert disabled.exit_code == 0, disabled.output
    assert "Telemetry: disabled" in disabled.output

    data = yaml.safe_load((tmp_path / "config" / "config.yaml").read_text())
    assert data["telemetry"]["enabled"] is False

    enabled = runner.invoke(main, ["setup", "telemetry", "--enable"])
    assert enabled.exit_code == 0, enabled.output
    data = yaml.safe_load((tmp_path / "config" / "config.yaml").read_text())
    assert data["telemetry"]["enabled"] is True


def test_api_run_calls_api_level_telemetry():
    recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm", "model": "org/model"})
    options = api.RunOptions(recipe=recipe, hosts=("h1",), solo=True, dry_run=True)

    class _FakeRuntime:
        runtime_name = "vllm"
        executor = None

    fake_runtime = _FakeRuntime()
    fake_result = type(
        "FakeLaunchResult",
        (),
        {
            "rc": 0,
            "cluster_id": "sparkrun_aaaaaaaaaaaa_bbbbbbbbbbbb",
            "host_list": ["h1"],
            "is_solo": True,
            "runtime": fake_runtime,
            "recipe": recipe,
            "overrides": {},
            "container_image": "test:latest",
            "effective_cache_dir": "/tmp/cache",
            "serve_port": 8000,
            "config": None,
            "recipe_ref": None,
            "comm_env": None,
            "ib_ip_map": {},
            "serve_command": "",
            "runtime_info": {},
            "builder": None,
            "backends": {},
            "timeline": None,
        },
    )()

    with (
        patch("sparkrun.core.launcher.launch_inference", return_value=fake_result),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=fake_runtime),
        patch("sparkrun.telemetry.emit_run_telemetry") as emit,
    ):
        result = api.run(options)

    assert result.cluster_id == "sparkrun_aaaaaaaaaaaa_bbbbbbbbbbbb"
    emit.assert_called_once()
    assert emit.call_args.args[0] is not None
    assert emit.call_args.kwargs["result"].cluster_id == result.cluster_id
    assert emit.call_args.kwargs["recipe"] is recipe
    assert emit.call_args.kwargs["options"] is options
