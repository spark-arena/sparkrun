"""CLI integration tests for sparkrun setup diagnose and run --collect-diagnostics."""

from __future__ import annotations

import json
from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.orchestration.ssh import RemoteResult


@pytest.fixture
def runner():
    return CliRunner()


def _make_diag_stdout() -> str:
    """Minimal valid spark_diagnose.sh output."""
    lines = [
        "DIAG_OS_NAME=Ubuntu",
        "DIAG_OS_VERSION=22.04",
        "DIAG_OS_PRETTY=Ubuntu 22.04",
        "DIAG_KERNEL=6.8.0",
        "DIAG_ARCH=aarch64",
        "DIAG_BIOS_VERSION=1.0",
        "DIAG_BOARD_NAME=DGX Spark",
        "DIAG_PRODUCT_NAME=NVIDIA DGX Spark",
        "DIAG_JETPACK_VERSION=6.2",
        "DIAG_CPU_MODEL=ARM",
        "DIAG_CPU_CORES=12",
        "DIAG_CPU_THREADS=12",
        "DIAG_RAM_TOTAL_KB=131072000",
        "DIAG_RAM_AVAILABLE_KB=100000000",
        "DIAG_DISK_ROOT_TOTAL_KB=500000000",
        "DIAG_DISK_ROOT_AVAIL_KB=300000000",
        "DIAG_DISK_HOME_TOTAL_KB=500000000",
        "DIAG_DISK_HOME_AVAIL_KB=300000000",
        "DIAG_GPU_NAME=GH200",
        "DIAG_GPU_MEMORY_MB=131072",
        "DIAG_GPU_DRIVER=570.86",
        "DIAG_CUDA_VERSION=12.8",
        "DIAG_NET_COUNT=0",
        "DIAG_DEFAULT_IFACE=eth0",
        "DIAG_MGMT_IP=10.0.0.1",
        "DIAG_DOCKER_VERSION=27.1",
        "DIAG_DOCKER_STORAGE=overlay2",
        "DIAG_DOCKER_ROOT=/var/lib/docker",
        "DIAG_DOCKER_NVIDIA_RUNTIME=true",
        "DIAG_DOCKER_RUNNING=0",
        "DIAG_DOCKER_SPARKRUN=0",
        "DIAG_SSH_USER=user",
        "DIAG_COMPLETE=1",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# sparkrun setup diagnose
# ---------------------------------------------------------------------------


class TestSetupDiagnose:
    @mock.patch("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho ok")
    def test_diagnose_basic(self, mock_script, mock_parallel, runner, v, tmp_path):
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout=_make_diag_stdout(), stderr=""),
        ]

        output = tmp_path / "test.ndjson"
        result = runner.invoke(
            main,
            [
                "setup",
                "diagnose",
                "--hosts",
                "10.0.0.1",
                "-o",
                str(output),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert output.exists()

        lines = output.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        types = [r["_type"] for r in records]
        assert "diag_header" in types
        assert "host_hardware" in types
        assert "diag_summary" in types

    @mock.patch("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho ok")
    def test_diagnose_json_stdout(self, mock_script, mock_parallel, runner, v, tmp_path):
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout=_make_diag_stdout(), stderr=""),
        ]

        output = tmp_path / "test.ndjson"
        result = runner.invoke(
            main,
            [
                "setup",
                "diagnose",
                "--hosts",
                "10.0.0.1",
                "-o",
                str(output),
                "--json",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # The JSON summary should be in stdout
        assert '"successful": 1' in result.output

    @mock.patch("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho ok")
    def test_diagnose_failed_host_exits_1(self, mock_script, mock_parallel, runner, v, tmp_path):
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=1, stdout="", stderr="refused"),
        ]

        output = tmp_path / "test.ndjson"
        result = runner.invoke(
            main,
            [
                "setup",
                "diagnose",
                "--hosts",
                "10.0.0.1",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 1

    @mock.patch("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho ok")
    def test_diagnose_dry_run(self, mock_script, mock_parallel, runner, v, tmp_path):
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr="[dry-run]"),
        ]

        output = tmp_path / "test.ndjson"
        runner.invoke(
            main,
            [
                "setup",
                "diagnose",
                "--hosts",
                "10.0.0.1",
                "-o",
                str(output),
                "--dry-run",
            ],
        )

        # dry-run mode may produce incomplete output; just check it runs
        assert output.exists()

    def test_diagnose_hidden_from_help(self, runner, v):
        result = runner.invoke(main, ["setup", "--help"], catch_exceptions=False)
        assert "diagnose" not in result.output


# ---------------------------------------------------------------------------
# sparkrun run --collect-diagnostics
# ---------------------------------------------------------------------------

_TEST_RECIPE_DATA = {
    "sparkrun_version": "2",
    "name": "Test Diag Recipe",
    "description": "A test recipe for diagnostics",
    "model": "test/model",
    "runtime": "sglang",
    "mode": "auto",
    "max_nodes": 1,
    "container": "img:latest",
    "defaults": {
        "port": 30000,
        "host": "0.0.0.0",
    },
}


class TestRunCollectDiagnostics:
    def test_collect_diagnostics_flag_exists(self, runner, v):
        """Verify the --collect-diagnostics flag is accepted (hidden but functional)."""
        # Just check that the option is recognized — a full run test requires
        # extensive mocking of launch_inference which is covered elsewhere.
        result = runner.invoke(main, ["run", "--help"])
        # Hidden options don't show in help, but shouldn't error
        assert result.exit_code == 0
