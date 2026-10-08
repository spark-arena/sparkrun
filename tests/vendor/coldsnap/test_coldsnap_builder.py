# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

from __future__ import annotations

import logging
from contextlib import contextmanager
import json
import subprocess
import threading
from types import SimpleNamespace

import pytest

from sparkrun.plugins.coldsnap.builder import (
    ColdSnapBuilder,
    _build_plan,
    _resolve_settings,
    _report_nccl_selection,
    render_build_script,
)
from sparkrun.core.progress import PROGRESS, progress_heartbeat
from sparkrun.plugins.coldsnap.tool import DEFAULT_CONTROLLER_COMMIT, DEFAULT_CONTROLLER_VERSION


PINNED_IMAGE = "registry.example/vllm@sha256:" + "a" * 64
_DETECT_SNAPSHOT_DRIVER = ColdSnapBuilder._detect_snapshot_driver
_DETECT_DOCKER_PLATFORM = ColdSnapBuilder._detect_docker_platform
_PREPARED_SOURCES = ColdSnapBuilder._prepared_sources


def _recipe(**builder_config):
    return SimpleNamespace(builder_config=builder_config)


def _result(*, success=True, stdout="", stderr=""):
    return SimpleNamespace(success=success, returncode=0 if success else 1, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _stable_snapshot_driver(monkeypatch):
    monkeypatch.setattr(ColdSnapBuilder, "_detect_snapshot_driver", lambda *_args, **_kwargs: "n610")
    monkeypatch.setattr(ColdSnapBuilder, "_detect_docker_platform", lambda *_args, **_kwargs: "linux/arm64")

    @contextmanager
    def sources(*_args, **_kwargs):
        yield "/tmp/test-coldsnap-sources"

    monkeypatch.setattr(ColdSnapBuilder, "_prepared_sources", sources)


def test_coldsnap_builder_is_registered_by_plugin_loader():
    from sparkrun.core.bootstrap import get_builder, init_sparkrun, list_builders

    variables = init_sparkrun()
    assert "coldsnap" in list_builders(variables)
    assert isinstance(get_builder("coldsnap", variables), ColdSnapBuilder)


def test_build_plan_requires_digest_pinned_input_and_is_content_addressed():
    settings = _resolve_settings(_recipe(), None)
    with pytest.raises(ValueError, match="digest"):
        _build_plan("registry.example/vllm:latest", settings, "121", docker_platform="linux/arm64")

    first = _build_plan(PINNED_IMAGE, settings, "121", docker_platform="linux/arm64")
    same = _build_plan(PINNED_IMAGE, settings, "121", docker_platform="linux/arm64")
    other_arch = _build_plan(PINNED_IMAGE, settings, "90", docker_platform="linux/arm64")
    other_driver = _build_plan(PINNED_IMAGE, settings, "121", "n580", docker_platform="linux/arm64")
    other_platform = _build_plan(PINNED_IMAGE, settings, "121", docker_platform="linux/amd64")
    assert first == same
    assert first.output_image.startswith("sparkrun/coldsnap-vllm:")
    assert first.output_image != other_arch.output_image
    assert first.output_image != other_driver.output_image
    assert first.output_image != other_platform.output_image


def test_builder_settings_accept_arch_and_source_overrides():
    settings = _resolve_settings(
        _recipe(
            cuda_arch="sm_90",
            output_repository="localhost:5500/example/coldsnap",
            coldsnap_ref="refs/heads/staging",
            coldsnap_revision="b" * 40,
            nccl_policy="exact",
            criu_image="registry.example/criu@sha256:" + "c" * 64,
        ),
        None,
    )
    assert settings.cuda_arch == "90"
    assert settings.output_repository == "localhost:5500/example/coldsnap"
    assert settings.sources[0].ref == "refs/heads/staging"
    assert settings.sources[0].revision == "b" * 40
    assert settings.nccl_policy == "exact"
    assert settings.criu_image == "registry.example/criu@sha256:" + "c" * 64


def test_builder_requires_digest_pinned_criu_image():
    with pytest.raises(ValueError, match="criu_image must be digest-pinned"):
        _resolve_settings(_recipe(criu_image="ghcr.io/sparksq/criu:latest"), None)


def test_sparkrun_builder_defaults_to_latest_qualified_same_major_nccl():
    settings = _resolve_settings(_recipe(), None)
    assert settings.nccl_policy == "match-or-latest-qualified"
    with pytest.raises(ValueError, match="nccl_policy"):
        _resolve_settings(_recipe(nccl_policy="latest"), None)


def test_builder_sources_use_public_https_where_available():
    settings = _resolve_settings(_recipe(), None)
    private = {source.name: source.url for source in settings.sources}

    assert private == {
        "coldsnap": "https://github.com/sparksq/coldsnap.git",
        "go_criu": "https://github.com/sparksq/go-criu.git",
        "cuda_checkpoint": "https://github.com/sparksq/cuda-checkpoint.git",
    }
    assert settings.sources[0].ref == "refs/tags/v%s" % DEFAULT_CONTROLLER_VERSION
    assert settings.sources[0].revision == DEFAULT_CONTROLLER_COMMIT
    assert len(DEFAULT_CONTROLLER_COMMIT) == 40 and all(char in "0123456789abcdef" for char in DEFAULT_CONTROLLER_COMMIT)
    assert settings.sources[1].ref == settings.sources[1].revision == "29a4f2f8e8374d38319a9851d9c1ef880dd0a0e8"
    assert settings.sources[2].ref == settings.sources[2].revision == "00d5cce84c628088d6caa203fc4af40c1538b6f7"
    assert settings.criu_image == ("ghcr.io/sparksq/criu@sha256:2ff53a61af48e7e676bd4d64747394ca7c7622840ef0c740e719e6ecadb0d07c")


def test_rendered_build_uses_canonical_coldsnap_dockerfiles_and_pins():
    settings = _resolve_settings(_recipe(), None)
    plan = _build_plan(PINNED_IMAGE, settings, "121", docker_platform="linux/arm64")
    script = render_build_script(plan)

    assert "deploy/nccl/Dockerfile.payload" in script
    assert "deploy/nccl/Dockerfile.provider" in script
    assert "deploy/vllm/Dockerfile" in script
    assert '--build-context "nccl_source=$COLDSNAP_SOURCE_NCCL_DIR"' in script
    assert '--build-context "nccl_release=$COLDSNAP_NCCL_RELEASE_DIR"' in script
    assert "COLDSNAP_NCCL_BUILD_IMAGE=" in script
    assert "NCCL recipe payload_build_image must be digest-pinned" in script
    assert '--build-arg "NCCL_BUILD_IMAGE=$COLDSNAP_NCCL_BUILD_IMAGE"' in script
    assert '--build-arg "NCCL_PAYLOAD_IMAGE=$COLDSNAP_NCCL_PAYLOAD_IMAGE"' in script
    assert "resolving published NCCL payload" in script
    assert "published NCCL payload unavailable" in script
    assert "io.sparksq.coldsnap.cuda.gencode" in script
    assert "published NCCL payload lacks sm_$COLDSNAP_CUDA_ARCH" in script
    assert "published NCCL payload architecture $COLDSNAP_NCCL_PAYLOAD_ARCH does not match $COLDSNAP_NCCL_HOST_ARCH" in script
    assert "provider_revision" in script
    assert "payload_repository" in script
    assert 'COLDSNAP_NCCL_PAYLOAD_REPODIGEST_REPOSITORY="${COLDSNAP_NCCL_PAYLOAD_REPOSITORY#docker.io/}"' in script
    assert '--build-context "nccl_provider=docker-image://$COLDSNAP_NCCL_IMAGE"' in script
    assert '--build-context "criu_image=docker-image://$COLDSNAP_CRIU_IMAGE"' in script
    assert '--build-arg "CRIU_IMAGE=$COLDSNAP_CRIU_IMAGE"' in script
    assert "COLDSNAP_SOURCE_CRIU_DIR" not in script
    assert "source.lock" in script
    assert "libnccl.so.2" in script
    assert "use_reproducible_nvcc" in script
    assert "NCCL_REPRODUCIBLE_NVCC=$COLDSNAP_NCCL_REPRODUCIBLE_NVCC" in script
    assert '["build"].get("tls_backend", "")' in script
    assert '--build-arg "NCCL_TLS_BACKEND=$COLDSNAP_NCCL_TLS_BACKEND"' in script
    assert "strip_unneeded" in script
    assert "NCCL_STRIP_OUTPUTS=$COLDSNAP_NCCL_STRIP_OUTPUTS" in script
    assert "COLDSNAP_NCCL_POLICY=match-or-latest-qualified" in script
    assert "using latest qualified same-major provider" in script
    assert '--build-arg "COLDSNAP_NCCL_POLICY=$COLDSNAP_NCCL_POLICY"' in script
    assert "io.sparksq.coldsnap.nccl.selection" in script
    assert "compute_${COLDSNAP_CUDA_ARCH},code=sm_${COLDSNAP_CUDA_ARCH}" in script
    for source in settings.sources:
        assert source.revision in script
        assert "source cache hit:" in script
        assert "fetching $coldsnap_source_name" in script
    subprocess.run(("bash", "-n"), input=script, text=True, check=True)


def test_sglang_build_plan_uses_engine_specific_runtime_for_both_drivers():
    settings = _resolve_settings(_recipe(), None, "sglang")
    plan = _build_plan(PINNED_IMAGE, settings, "121", "n610", "sglang", docker_platform="linux/arm64")
    script = render_build_script(plan)

    assert plan.output_image.startswith("sparkrun/coldsnap-sglang:")
    assert "deploy/sglang/Dockerfile" in script
    assert '--build-arg "SGLANG_IMAGE=$COLDSNAP_INPUT_IMAGE"' in script
    assert "io.sparksq.coldsnap.snapshot-drivers=n580,n610" in script
    assert 'group="sglang.srt.plugins"' in script
    subprocess.run(("bash", "-n"), input=script, text=True, check=True)

    n580 = _build_plan(PINNED_IMAGE, settings, "121", "n580", "sglang", docker_platform="linux/arm64")
    assert n580.snapshot_driver == "n580"
    assert n580.output_image != plan.output_image


def test_rendered_selector_uses_latest_qualified_same_major_even_with_exact_available(tmp_path):
    settings = _resolve_settings(_recipe(), None)
    script = render_build_script(_build_plan(PINNED_IMAGE, settings, "121", docker_platform="linux/arm64"))
    selector = script.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    releases = tmp_path / "releases"
    releases.mkdir()

    def add(release, revision, *, state="accepted"):
        directory = releases / (release + "-1")
        directory.mkdir()
        version = tuple(int(part) for part in release.split("."))
        recipe = {
            "format": 2,
            "provider_id": f"nccl-{release}-1+coldsnap.{revision}",
            "provider_revision": revision,
            "nccl_release": release,
            "nccl_version_code": version[0] * 10000 + version[1] * 100 + version[2],
            "capabilities": ["full-network-reset"],
        }
        qualification = {
            "format": 2,
            "provider_id": recipe["provider_id"],
            "state": state,
            "policy": "production",
            "capabilities": recipe["capabilities"],
            "transports": ["ib-roce"],
            "checks": ["socket-restore"],
        }
        (directory / "recipe.json").write_text(json.dumps(recipe))
        (directory / "qualification.json").write_text(json.dumps(qualification))

    add("2.30.7", 1)
    add("2.31.1", 2)
    add("2.31.2", 3)

    latest = subprocess.run(
        ("python3", "-", str(releases), "2.30.7", "match-or-latest-qualified"),
        input=selector,
        text=True,
        check=True,
        capture_output=True,
    ).stdout.splitlines()
    assert latest[1:] == ["2.31.2", "latest-qualified-same-major"]

    exact = subprocess.run(
        ("python3", "-", str(releases), "2.30.7", "exact"),
        input=selector,
        text=True,
        check=True,
        capture_output=True,
    ).stdout.splitlines()
    assert exact[1:] == ["2.30.7", "exact"]

    (releases / "2.30.7-1").rename(releases / "2.30.6-removed")
    (releases / "2.30.6-removed" / "qualification.json").unlink()
    latest_without_exact = subprocess.run(
        ("python3", "-", str(releases), "2.30.7", "match-or-latest-qualified"),
        input=selector,
        text=True,
        check=True,
        capture_output=True,
    ).stdout.splitlines()
    assert latest_without_exact[1:] == ["2.31.2", "latest-qualified-same-major"]

    rejected = subprocess.run(
        ("python3", "-", str(releases), "2.30.7", "exact"),
        input=selector,
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0
    assert "no qualified NCCL provider" in rejected.stderr


def test_prepare_returns_existing_coldsnap_image_without_build(monkeypatch):
    builder = ColdSnapBuilder()
    calls = []

    def run(_host, script, **_kwargs):
        calls.append(script)
        if "snapshot-drivers" in script:
            return _result(stdout="n580,n610\n")
        return _result(stdout="vllm-cuda-criu-v1\n")

    monkeypatch.setattr(builder, "_run", run)
    monkeypatch.setattr(
        builder,
        "_run_streaming",
        lambda _host, script, **_kwargs: calls.append(script) or _result(),
    )
    assert builder.prepare_image(PINNED_IMAGE, _recipe(), ["node-a"], transfer_mode="delegated") == PINNED_IMAGE
    assert len(calls) == 3
    assert all("docker image inspect" in call for call in calls)


def test_prepare_reuses_arch_specific_cached_conversion(monkeypatch):
    builder = ColdSnapBuilder()
    calls = []

    def run(_host, script, **_kwargs):
        calls.append(script)
        if "io.sparksq.coldsnap.runtime" in script:
            return _result(stdout="<no value>\n")
        if "torch.cuda.get_device_capability" in script:
            return _result(stdout="121\n")
        if "image inspect" in script:
            return _result()
        raise AssertionError("unexpected build invocation")

    monkeypatch.setattr(builder, "_run", run)
    monkeypatch.setattr(
        builder,
        "_run_streaming",
        lambda _host, script, **_kwargs: calls.append(script) or _result(),
    )
    result = builder.prepare_image(PINNED_IMAGE, _recipe(), ["node-a"], transfer_mode="delegated")
    assert result.startswith("sparkrun/coldsnap-vllm:")
    assert len(calls) == 4


def test_prepare_dry_run_does_not_probe_or_build(monkeypatch):
    builder = ColdSnapBuilder()
    monkeypatch.setattr(builder, "_run", lambda *_args, **_kwargs: pytest.fail("dry run executed a command"))
    monkeypatch.setattr(builder, "_run_streaming", lambda *_args, **_kwargs: pytest.fail("dry run streamed a command"))
    result = builder.prepare_image(PINNED_IMAGE, _recipe(), ["node-a"], dry_run=True)
    assert result.startswith("sparkrun/coldsnap-vllm:")


def test_shared_builder_context_selects_driver_without_a_second_probe(monkeypatch):
    builder = ColdSnapBuilder()
    monkeypatch.setattr(builder, "_run", lambda *_args, **_kwargs: pytest.fail("dry run probed a host"))
    n580 = builder.prepare(
        PINNED_IMAGE,
        _recipe(),
        ["node-a"],
        dry_run=True,
        builder_context={"snapshot_driver": "n580"},
    )
    n610 = builder.prepare(
        PINNED_IMAGE,
        _recipe(),
        ["node-a"],
        dry_run=True,
        builder_context={"snapshot_driver": "n610"},
    )
    assert n580 != n610


def test_shared_builder_context_selects_sglang_without_a_second_probe(monkeypatch):
    builder = ColdSnapBuilder()
    monkeypatch.setattr(builder, "_run", lambda *_args, **_kwargs: pytest.fail("dry run probed a host"))
    result = builder.prepare(
        PINNED_IMAGE,
        _recipe(),
        ["node-a"],
        dry_run=True,
        builder_context={"snapshot_driver": "n610", "engine": "sglang"},
    )
    assert result.startswith("sparkrun/coldsnap-sglang:")


@pytest.mark.parametrize(("version", "selected"), [("580.126.09", "n580"), ("610.22.03", "n610")])
def test_builder_detects_snapshot_driver_family(monkeypatch, version, selected):
    builder = ColdSnapBuilder()
    monkeypatch.setattr(builder, "_run", lambda *_args, **_kwargs: _result(stdout=version + "\n"))
    # The module-level fixture installs a class method; an instance override
    # exercises the real selector directly.
    assert _DETECT_SNAPSHOT_DRIVER(builder, "node-a", _resolve_settings(_recipe(), None), None) == selected


def test_prepare_surfaces_build_failure_tail(monkeypatch):
    builder = ColdSnapBuilder()

    def run(_host, script, **_kwargs):
        if "io.sparksq.coldsnap.runtime" in script:
            return _result(stdout="<no value>\n")
        if "torch.cuda.get_device_capability" in script:
            return _result(stdout="121\n")
        if "image inspect" in script:
            return _result(success=False)
        raise AssertionError("unexpected captured command")

    monkeypatch.setattr(builder, "_run", run)

    def stream(_host, script, **_kwargs):
        if "pulling input image" in script:
            return _result()
        return _result(success=False, stderr="build exploded")

    monkeypatch.setattr(builder, "_run_streaming", stream)
    with pytest.raises(RuntimeError, match="build exploded"):
        builder.prepare_image(PINNED_IMAGE, _recipe(rebuild=True), ["node-a"], transfer_mode="delegated")


def test_builder_reports_default_progress_and_detail(caplog, monkeypatch):
    builder = ColdSnapBuilder()
    monkeypatch.setattr(builder, "_run_streaming", lambda *_args, **_kwargs: _result())

    def run(_host, script, **_kwargs):
        if "io.sparksq.coldsnap.runtime" in script:
            return _result(stdout="<no value>\n")
        if "torch.cuda.get_device_capability" in script:
            return _result(stdout="121\n")
        return _result()

    monkeypatch.setattr(builder, "_run", run)
    with caplog.at_level(PROGRESS):
        builder.prepare_image(PINNED_IMAGE, _recipe(), ["node-a"], transfer_mode="delegated")

    messages = [record.getMessage() for record in caplog.records]
    assert any("preparing pinned input image" in message for message in messages)
    assert any("detecting target CUDA architecture" in message for message in messages)
    assert any("reusing cached image" in message for message in messages)


def test_builder_progress_heartbeat_is_visible_at_default_level(caplog, monkeypatch):
    observed = threading.Event()
    original_emit = caplog.handler.emit

    def record_heartbeat(record):
        original_emit(record)
        if record.name == "test.coldsnap.heartbeat" and "still running" in record.getMessage():
            observed.set()

    monkeypatch.setattr(caplog.handler, "emit", record_heartbeat)
    with caplog.at_level(PROGRESS):
        with progress_heartbeat(logging.getLogger("test.coldsnap.heartbeat"), "ColdSnap builder: test operation", interval=0.01):
            assert observed.wait(timeout=5), "heartbeat was not logged at the default level"

    assert any("still running" in record.getMessage() for record in caplog.records)


def test_builder_reports_latest_nccl_selection_at_progress_level(caplog):
    result = _result(stderr="[coldsnap-builder] base NCCL 2.30.7; using latest qualified same-major provider 2.31.2\n")
    with caplog.at_level(PROGRESS):
        _report_nccl_selection(result)
    assert "used latest qualified same-major provider 2.31.2" in caplog.text


@pytest.mark.parametrize(("level", "expected_quiet"), [(PROGRESS, True), (logging.INFO, False)])
def test_builder_verbosity_controls_native_output(level, expected_quiet, caplog, monkeypatch):
    builder = ColdSnapBuilder()
    calls = []
    monkeypatch.setattr(
        "sparkrun.plugins.coldsnap.builder.run_script_on_host_streaming",
        lambda *_args, **kwargs: calls.append(kwargs) or _result(),
    )

    with caplog.at_level(level):
        builder._run_streaming(
            "node-a",
            "echo build",
            ssh_kwargs={},
            timeout=60,
            progress_label="ColdSnap builder: test operation",
        )

    assert calls[0]["quiet"] is expected_quiet


def test_builder_command_diagnostics_are_debug_only(caplog, monkeypatch):
    builder = ColdSnapBuilder()
    monkeypatch.setattr("sparkrun.plugins.coldsnap.builder.run_script_on_host_streaming", lambda *_args, **_kwargs: _result())

    with caplog.at_level(logging.DEBUG):
        builder._run_streaming(
            "node-a",
            "echo build",
            ssh_kwargs={},
            timeout=60,
            progress_label="ColdSnap builder: test operation",
        )

    assert any("ColdSnap builder command" in record.getMessage() for record in caplog.records)


def test_builder_metadata_probes_do_not_require_container_network():
    settings = _resolve_settings(_recipe(), None, "sglang")
    plan = _build_plan(PINNED_IMAGE, settings, "121", "n580", "sglang", docker_platform="linux/arm64")
    script = render_build_script(plan)
    probes = [line for line in script.splitlines() if '"$COLDSNAP_DOCKER" run ' in line]
    assert len(probes) == 2
    assert all("--network none" in line for line in probes)


def test_sglang_deepep_provider_check_runs_with_gpu_after_build():
    settings = _resolve_settings(_recipe(), None, "sglang")
    plan = _build_plan(PINNED_IMAGE, settings, "121", "n580", "sglang", docker_platform="linux/arm64")
    script = render_build_script(plan)
    assert '--gpus all -e EP_SUPPRESS_NCCL_CHECK=0 --entrypoint python3 "$COLDSNAP_OUTPUT_IMAGE"' in script
    assert "import deep_ep" in script
    assert "samefile" in script
