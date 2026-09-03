"""CLI tests for ``sparkrun cluster inspect`` — head-node hardware reporting."""

from __future__ import annotations

import json
from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.orchestration.disk_info import CacheStatus
from sparkrun.orchestration.distribution import TransferModeResult
from sparkrun.orchestration.ssh import RemoteResult


@pytest.fixture
def runner():
    return CliRunner()


def _diag_stdout() -> str:
    lines = [
        "DIAG_HOSTNAME=spark-01",
        "DIAG_PRODUCT_NAME=NVIDIA DGX Spark",
        "DIAG_BOARD_NAME=DGX Spark",
        "DIAG_BIOS_VERSION=1.2.3",
        "DIAG_OS_PRETTY=Ubuntu 24.04 LTS",
        "DIAG_KERNEL=6.11.0-1008-nvidia",
        "DIAG_ARCH=aarch64",
        "DIAG_CPU_MODEL=ARM Neoverse",
        "DIAG_CPU_CORES=20",
        "DIAG_CPU_THREADS=20",
        "DIAG_RAM_TOTAL_KB=131072000",
        "DIAG_GPU_NAME=NVIDIA GB10",
        "DIAG_GPU_MEMORY_MB=122880",
        "DIAG_GPU_DRIVER=580.95.05",
        "DIAG_CUDA_VERSION=13.0",
        "DIAG_JETPACK_VERSION=6.2",
        "DIAG_DOCKER_VERSION=28.1.1",
        "DIAG_DOCKER_STORAGE=overlay2",
        "DIAG_DOCKER_NVIDIA_RUNTIME=true",
        "DIAG_NET_COUNT=0",
        "DIAG_COMPLETE=1",
    ]
    return "\n".join(lines) + "\n"


class _InspectMocks:
    """Patch every SSH-touching call ``cluster inspect`` makes."""

    def __init__(self, diag_results):
        self._diag_results = diag_results
        self._patches = []

    def __enter__(self):
        targets = [
            mock.patch("sparkrun.core.launcher.resolve_effective_cache_dir", return_value="/home/u/.cache/huggingface"),
            mock.patch(
                "sparkrun.orchestration.distribution.resolve_auto_transfer_mode",
                return_value=TransferModeResult(mode="delegated"),
            ),
            mock.patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts", return_value=None),
            mock.patch(
                "sparkrun.orchestration.disk_info.probe_cache_status",
                return_value={"10.0.0.1": CacheStatus(host="10.0.0.1")},
            ),
            mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho ok"),
            mock.patch(
                "sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel",
                return_value=self._diag_results,
            ),
        ]
        for p in targets:
            p.start()
            self._patches.append(p)
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


def test_inspect_reports_head_node_hardware(runner, v):
    results = [RemoteResult(host="10.0.0.1", returncode=0, stdout=_diag_stdout(), stderr="")]
    with _InspectMocks(results):
        result = runner.invoke(main, ["cluster", "inspect", "--hosts", "10.0.0.1"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "Head Node Hardware (10.0.0.1):" in result.output
    assert "NVIDIA DGX Spark" in result.output
    assert "Ubuntu 24.04 LTS (kernel 6.11.0-1008-nvidia, aarch64)" in result.output
    assert "580.95.05" in result.output
    assert "13.0" in result.output
    assert "28.1.1 (storage: overlay2, nvidia runtime: yes)" in result.output


def test_inspect_head_hardware_json(runner, v):
    results = [RemoteResult(host="10.0.0.1", returncode=0, stdout=_diag_stdout(), stderr="")]
    with _InspectMocks(results):
        result = runner.invoke(main, ["cluster", "inspect", "--hosts", "10.0.0.1", "--json"], catch_exceptions=False)

    assert result.exit_code == 0
    data = json.loads(result.output)
    head = data["head_node"]
    assert head["host"] == "10.0.0.1"
    assert head["hardware"]["gpu_driver"] == "580.95.05"
    assert head["hardware"]["gpu_memory_gb"] == 120.0
    assert head["hardware"]["os"] == "Ubuntu 24.04 LTS"


def test_inspect_survives_failed_hardware_probe(runner, v):
    """A failed probe must not take down the rest of the report."""
    results = [RemoteResult(host="10.0.0.1", returncode=255, stdout="", stderr="connection refused")]
    with _InspectMocks(results):
        result = runner.invoke(main, ["cluster", "inspect", "--hosts", "10.0.0.1"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "hardware probe failed" in result.output
    assert "Cache Paths:" in result.output


def test_inspect_dry_run_skips_hardware_probe(runner, v):
    with _InspectMocks([]) as mocks:  # noqa: F841
        result = runner.invoke(main, ["cluster", "inspect", "--hosts", "10.0.0.1", "--dry-run"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "[dry-run] Would probe head node hardware on 10.0.0.1" in result.output
    assert "Head Node Hardware" not in result.output
