"""Unit tests for the diagnostics package: NDJSONWriter, spark_collector, run_collector."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from sparkrun.diagnostics.ndjson_writer import NDJSONWriter
from sparkrun.diagnostics.spark_collector import (
    _extract_keys,
    _extract_firmware_devices,
    _extract_firmware_history,
    _extract_network,
    collect_spark_diagnostics,
    collect_sudo_diagnostics,
)
from sparkrun.diagnostics.run_collector import RunDiagnosticsCollector
from sparkrun.orchestration.ssh import RemoteResult


# ---------------------------------------------------------------------------
# NDJSONWriter
# ---------------------------------------------------------------------------

class TestNDJSONWriter:
    def test_emit_creates_valid_ndjson(self, tmp_path: Path):
        path = tmp_path / "out.ndjson"
        with NDJSONWriter(path) as w:
            w.emit("test_record", {"foo": "bar"})
            w.emit("another", {"num": 42})

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

        r1 = json.loads(lines[0])
        assert r1["_type"] == "test_record"
        assert r1["_seq"] == 1
        assert "_ts" in r1
        assert r1["foo"] == "bar"

        r2 = json.loads(lines[1])
        assert r2["_type"] == "another"
        assert r2["_seq"] == 2
        assert r2["num"] == 42

    def test_seq_monotonic(self, tmp_path: Path):
        path = tmp_path / "seq.ndjson"
        with NDJSONWriter(path) as w:
            for i in range(5):
                rec = w.emit("t", {})
                assert rec["_seq"] == i + 1

    def test_default_str_serialization(self, tmp_path: Path):
        path = tmp_path / "ser.ndjson"
        with NDJSONWriter(path) as w:
            w.emit("paths", {"p": Path("/some/path")})

        line = json.loads(path.read_text().strip())
        assert line["p"] == "/some/path"

    def test_emit_without_data(self, tmp_path: Path):
        path = tmp_path / "nodata.ndjson"
        with NDJSONWriter(path) as w:
            rec = w.emit("empty")

        assert rec["_type"] == "empty"
        assert rec["_seq"] == 1
        line = json.loads(path.read_text().strip())
        assert line["_type"] == "empty"

    def test_emit_returns_record(self, tmp_path: Path):
        path = tmp_path / "ret.ndjson"
        with NDJSONWriter(path) as w:
            rec = w.emit("t", {"k": "v"})
        assert rec["_type"] == "t"
        assert rec["k"] == "v"

    def test_append_mode(self, tmp_path: Path):
        path = tmp_path / "append.ndjson"
        path.write_text('{"existing": true}\n')

        with NDJSONWriter(path) as w:
            w.emit("new", {})

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["existing"] is True
        assert json.loads(lines[1])["_type"] == "new"

    def test_close_idempotent(self, tmp_path: Path):
        path = tmp_path / "close.ndjson"
        w = NDJSONWriter(path)
        w.open()
        w.emit("t", {})
        w.close()
        w.close()  # should not raise

    def test_emit_without_open_warns(self, tmp_path: Path, caplog):
        path = tmp_path / "noop.ndjson"
        w = NDJSONWriter(path)
        with caplog.at_level("WARNING"):
            w.emit("dropped", {})
        assert "not open" in caplog.text


# ---------------------------------------------------------------------------
# spark_collector helpers
# ---------------------------------------------------------------------------

class TestSparkCollectorHelpers:
    def test_extract_keys(self):
        kv = {
            "DIAG_CPU_MODEL": "ARM Neoverse",
            "DIAG_CPU_CORES": "12",
            "UNRELATED": "skip",
        }
        result = _extract_keys(kv, ("DIAG_CPU_MODEL", "DIAG_CPU_CORES"))
        assert result == {"cpu_model": "ARM Neoverse", "cpu_cores": "12"}
        assert "unrelated" not in result

    def test_extract_network(self):
        kv = {
            "DIAG_NET_COUNT": "2",
            "DIAG_NET_0_NAME": "eth0",
            "DIAG_NET_0_STATE": "up",
            "DIAG_NET_0_IP": "10.0.0.1",
            "DIAG_NET_1_NAME": "ib0",
            "DIAG_NET_1_STATE": "up",
            "DIAG_NET_1_IP": "192.168.1.1",
            "DIAG_DEFAULT_IFACE": "eth0",
            "DIAG_MGMT_IP": "10.0.0.1",
        }
        result = _extract_network(kv)
        assert len(result["interfaces"]) == 2
        assert result["interfaces"][0]["name"] == "eth0"
        assert result["interfaces"][1]["name"] == "ib0"
        assert result["default_iface"] == "eth0"
        assert result["mgmt_ip"] == "10.0.0.1"

    def test_extract_network_empty(self):
        result = _extract_network({})
        assert result["interfaces"] == []
        assert result["default_iface"] == ""

    def test_extract_firmware_devices(self):
        kv = {
            "DIAG_FWUPD_DEV_COUNT": "2",
            "DIAG_FWUPD_DEV_0": "NVIDIA GPU|570.86|abc-123",
            "DIAG_FWUPD_DEV_1": "System Firmware|1.0|def-456,ghi-789",
        }
        devices = _extract_firmware_devices(kv)
        assert len(devices) == 2
        assert devices[0]["name"] == "NVIDIA GPU"
        assert devices[0]["version"] == "570.86"
        assert devices[0]["guid"] == "abc-123"
        assert devices[1]["guid"] == "def-456,ghi-789"

    def test_extract_firmware_devices_empty(self):
        assert _extract_firmware_devices({}) == []
        assert _extract_firmware_devices({"DIAG_FWUPD_DEV_COUNT": "0"}) == []

    def test_extract_firmware_history(self):
        kv = {
            "DIAG_FWUPD_HIST_COUNT": "1",
            "DIAG_FWUPD_HIST_0": "System Firmware|2.0|2025-01-15",
        }
        history = _extract_firmware_history(kv)
        assert len(history) == 1
        assert history[0]["name"] == "System Firmware"
        assert history[0]["version"] == "2.0"
        assert history[0]["date"] == "2025-01-15"

    def test_extract_firmware_history_empty(self):
        assert _extract_firmware_history({}) == []


# ---------------------------------------------------------------------------
# collect_spark_diagnostics
# ---------------------------------------------------------------------------

def _make_diag_stdout(host: str = "10.0.0.1") -> str:
    """Build realistic spark_diagnose.sh stdout."""
    lines = [
        "DIAG_HOSTNAME=dgx-spark-01",
        "DIAG_HOSTNAME_FQDN=dgx-spark-01.local",
        "DIAG_OS_NAME=Ubuntu",
        "DIAG_OS_VERSION=22.04",
        "DIAG_OS_PRETTY=Ubuntu 22.04.5 LTS",
        "DIAG_KERNEL=6.8.0-nvidia",
        "DIAG_ARCH=aarch64",
        "DIAG_BIOS_VERSION=1.0",
        "DIAG_BOARD_NAME=DGX Spark",
        "DIAG_PRODUCT_NAME=NVIDIA DGX Spark",
        "DIAG_JETPACK_VERSION=6.2",
        "DIAG_CPU_MODEL=ARM Neoverse-V2",
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
        "DIAG_GPU_DRIVER=570.86.15",
        "DIAG_GPU_PSTATE=P0",
        "DIAG_GPU_TEMP_C=45",
        "DIAG_GPU_POWER_W=100",
        "DIAG_GPU_SERIAL=ABC123",
        "DIAG_GPU_UUID=GPU-xxxxx",
        "DIAG_CUDA_VERSION=12.8",
        "DIAG_NET_COUNT=1",
        "DIAG_NET_0_NAME=eth0",
        "DIAG_NET_0_STATE=up",
        "DIAG_NET_0_MTU=9000",
        "DIAG_NET_0_MAC=aa:bb:cc:dd:ee:ff",
        "DIAG_NET_0_SPEED=100000",
        "DIAG_NET_0_IP=10.0.0.1",
        "DIAG_DEFAULT_IFACE=eth0",
        "DIAG_MGMT_IP=10.0.0.1",
        "DIAG_DOCKER_VERSION=27.1.1",
        "DIAG_DOCKER_STORAGE=overlay2",
        "DIAG_DOCKER_ROOT=/var/lib/docker",
        "DIAG_DOCKER_NVIDIA_RUNTIME=true",
        "DIAG_DOCKER_RUNNING=2",
        "DIAG_DOCKER_SPARKRUN=1",
        "DIAG_FWUPD_DEV_COUNT=1",
        "DIAG_FWUPD_DEV_0=System Firmware|1.0|abc-123",
        "DIAG_FWUPD_HIST_COUNT=1",
        "DIAG_FWUPD_HIST_0=System Firmware|2.0|2025-01-15",
        "DIAG_SSH_USER=user",
        "DIAG_COMPLETE=1",
    ]
    return "\n".join(lines) + "\n"


def _make_sudo_diag_stdout() -> str:
    """Build realistic spark_diagnose_sudo.sh stdout."""
    lines = [
        "DIAG_DMI_BIOS_VENDOR=NVIDIA",
        "DIAG_DMI_BIOS_VERSION=1.2.3",
        "DIAG_DMI_BIOS_DATE=01/15/2025",
        "DIAG_DMI_SYS_MANUFACTURER=NVIDIA",
        "DIAG_DMI_SYS_PRODUCT=DGX Spark",
        "DIAG_DMI_SYS_VERSION=1.0",
        "DIAG_DMI_SYS_SERIAL=SN12345",
        "DIAG_DMI_SYS_UUID=uuid-abcdef",
        "DIAG_DMI_BOARD_MANUFACTURER=NVIDIA",
        "DIAG_DMI_BOARD_PRODUCT=DGX Spark Board",
        "DIAG_DMI_BOARD_VERSION=A01",
        "DIAG_DMI_BOARD_SERIAL=BSN67890",
        "DIAG_DMI_MEM_SLOTS=4",
        "DIAG_DMI_MEM_POPULATED=4",
        "DIAG_DMI_MEM_MAX=128 GB",
        "DIAG_SUDO_COMPLETE=1",
    ]
    return "\n".join(lines) + "\n"


class TestCollectSparkDiagnostics:
    @mock.patch("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho test")
    def test_successful_collection(self, mock_script, mock_parallel, tmp_path: Path):
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout=_make_diag_stdout(), stderr=""),
        ]

        path = tmp_path / "diag.ndjson"
        with NDJSONWriter(path) as writer:
            result = collect_spark_diagnostics(
                hosts=["10.0.0.1"],
                ssh_kwargs={"ssh_user": "test"},
                writer=writer,
            )

        assert "10.0.0.1" in result
        assert result["10.0.0.1"]["DIAG_COMPLETE"] == "1"

        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        types = [r["_type"] for r in records]
        assert "host_hardware" in types
        assert "host_firmware" in types
        assert "host_network" in types
        assert "host_docker" in types
        assert "diag_summary" in types

    @mock.patch("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho test")
    def test_failed_host(self, mock_script, mock_parallel, tmp_path: Path):
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=1, stdout="", stderr="connection refused"),
        ]

        path = tmp_path / "diag.ndjson"
        with NDJSONWriter(path) as writer:
            result = collect_spark_diagnostics(
                hosts=["10.0.0.1"],
                ssh_kwargs={},
                writer=writer,
            )

        assert result["10.0.0.1"] == {}

        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        types = [r["_type"] for r in records]
        assert "host_error" in types
        assert "diag_summary" in types
        summary = next(r for r in records if r["_type"] == "diag_summary")
        assert summary["failed"] == 1

    @mock.patch("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho test")
    def test_incomplete_sentinel(self, mock_script, mock_parallel, tmp_path: Path):
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="DIAG_OS_NAME=Ubuntu\n", stderr=""),
        ]

        path = tmp_path / "diag.ndjson"
        with NDJSONWriter(path) as writer:
            collect_spark_diagnostics(
                hosts=["10.0.0.1"],
                ssh_kwargs={},
                writer=writer,
            )

        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        assert any(r["_type"] == "host_error" for r in records)

    @mock.patch("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho test")
    def test_no_writer(self, mock_script, mock_parallel):
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout=_make_diag_stdout(), stderr=""),
        ]
        result = collect_spark_diagnostics(
            hosts=["10.0.0.1"],
            ssh_kwargs={},
            writer=None,
        )
        assert result["10.0.0.1"]["DIAG_COMPLETE"] == "1"

    @mock.patch("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho test")
    def test_multi_host(self, mock_script, mock_parallel, tmp_path: Path):
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout=_make_diag_stdout(), stderr=""),
            RemoteResult(host="10.0.0.2", returncode=1, stdout="", stderr="timeout"),
        ]

        path = tmp_path / "diag.ndjson"
        with NDJSONWriter(path) as writer:
            result = collect_spark_diagnostics(
                hosts=["10.0.0.1", "10.0.0.2"],
                ssh_kwargs={},
                writer=writer,
            )

        assert result["10.0.0.1"]["DIAG_COMPLETE"] == "1"
        assert result["10.0.0.2"] == {}

        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        summary = next(r for r in records if r["_type"] == "diag_summary")
        assert summary["successful"] == 1
        assert summary["failed"] == 1

    @mock.patch("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho test")
    def test_firmware_updates_emitted(self, mock_script, mock_parallel, tmp_path: Path):
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout=_make_diag_stdout(), stderr=""),
        ]

        path = tmp_path / "diag.ndjson"
        with NDJSONWriter(path) as writer:
            collect_spark_diagnostics(hosts=["10.0.0.1"], ssh_kwargs={}, writer=writer)

        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        fw_rec = next((r for r in records if r["_type"] == "host_firmware_updates"), None)
        assert fw_rec is not None
        assert len(fw_rec["devices"]) == 1
        assert fw_rec["devices"][0]["name"] == "System Firmware"
        assert len(fw_rec["history"]) == 1

    @mock.patch("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho test")
    def test_hostname_in_firmware(self, mock_script, mock_parallel, tmp_path: Path):
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout=_make_diag_stdout(), stderr=""),
        ]

        path = tmp_path / "diag.ndjson"
        with NDJSONWriter(path) as writer:
            collect_spark_diagnostics(hosts=["10.0.0.1"], ssh_kwargs={}, writer=writer)

        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        fw_rec = next(r for r in records if r["_type"] == "host_firmware")
        assert fw_rec["hostname"] == "dgx-spark-01"
        assert fw_rec["hostname_fqdn"] == "dgx-spark-01.local"


# ---------------------------------------------------------------------------
# collect_sudo_diagnostics
# ---------------------------------------------------------------------------

class TestCollectSudoDiagnostics:
    @mock.patch("sparkrun.orchestration.sudo.run_sudo_script_on_host")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho test")
    def test_sudo_collection(self, mock_script, mock_sudo, tmp_path: Path):
        mock_sudo.return_value = RemoteResult(
            host="10.0.0.1", returncode=0, stdout=_make_sudo_diag_stdout(), stderr="",
        )

        path = tmp_path / "sudo.ndjson"
        with NDJSONWriter(path) as writer:
            result = collect_sudo_diagnostics(
                hosts=["10.0.0.1"],
                ssh_kwargs={},
                sudo_password="test",
                writer=writer,
            )

        assert result["10.0.0.1"]["DIAG_SUDO_COMPLETE"] == "1"

        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        dmi = next(r for r in records if r["_type"] == "host_dmi")
        assert dmi["dmi_bios_vendor"] == "NVIDIA"
        assert dmi["dmi_sys_product"] == "DGX Spark"
        assert dmi["dmi_board_serial"] == "BSN67890"
        assert dmi["dmi_mem_max"] == "128 GB"

    @mock.patch("sparkrun.orchestration.sudo.run_sudo_script_on_host")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho test")
    def test_sudo_failure(self, mock_script, mock_sudo, tmp_path: Path):
        mock_sudo.return_value = RemoteResult(
            host="10.0.0.1", returncode=1, stdout="", stderr="auth failed",
        )

        path = tmp_path / "sudo.ndjson"
        with NDJSONWriter(path) as writer:
            result = collect_sudo_diagnostics(
                hosts=["10.0.0.1"],
                ssh_kwargs={},
                sudo_password="wrong",
                writer=writer,
            )

        assert result["10.0.0.1"] == {}
        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        assert any(r["_type"] == "host_error" for r in records)


# ---------------------------------------------------------------------------
# RunDiagnosticsCollector
# ---------------------------------------------------------------------------

class TestRunDiagnosticsCollector:
    @mock.patch("sparkrun.diagnostics.spark_collector.run_remote_scripts_parallel")
    @mock.patch("sparkrun.diagnostics.spark_collector.read_script", return_value="#!/bin/bash\necho test")
    def test_full_lifecycle(self, mock_script, mock_parallel, tmp_path: Path):
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout=_make_diag_stdout(), stderr=""),
        ]

        path = tmp_path / "run_diag.ndjson"
        with RunDiagnosticsCollector(str(path), ["10.0.0.1"], {}) as diag:
            diag.emit_header(command="sparkrun run test")
            diag.collect_spark_diagnostics()

            # Simulate recipe
            recipe = mock.MagicMock()
            recipe.qualified_name = "test-recipe"
            recipe.model = "test/model"
            recipe.runtime = "vllm"
            recipe.container = "img:latest"
            recipe.defaults = {"port": 8000}
            diag.emit_recipe(recipe, {"tp": 2})

            diag.emit_config(hosts=["10.0.0.1"], is_solo=True)

            diag.phase_start("launch")
            diag.phase_end("launch")

            result = mock.MagicMock()
            result.rc = 0
            result.cluster_id = "sparkrun_abc123"
            result.runtime_info = {"vllm": "0.8.0"}
            result.nccl_env = None
            diag.emit_launch_result(result)
            diag.emit_serve_command("vllm serve test", "img:latest")

            diag.emit_summary()

        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        types = [r["_type"] for r in records]

        assert "diag_header" in types
        assert "host_hardware" in types
        assert "run_recipe" in types
        assert "run_config" in types
        assert "run_phase" in types
        assert "run_launch_result" in types
        assert "run_serve_command" in types
        assert "run_summary" in types

        summary = next(r for r in records if r["_type"] == "run_summary")
        assert summary["success"] is True

    def test_phase_timing(self, tmp_path: Path):
        path = tmp_path / "timing.ndjson"
        with RunDiagnosticsCollector(str(path), [], {}) as diag:
            diag.phase_start("test_phase")
            diag.phase_end("test_phase")
            diag.emit_summary()

        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        phases = [r for r in records if r["_type"] == "run_phase"]
        assert len(phases) == 2
        assert phases[0]["status"] == "start"
        assert phases[1]["status"] == "end"
        assert "duration_seconds" in phases[1]

    def test_phase_error(self, tmp_path: Path):
        path = tmp_path / "err.ndjson"
        with RunDiagnosticsCollector(str(path), [], {}) as diag:
            diag.phase_start("failing")
            diag.phase_end("failing", error="something broke")
            diag.emit_summary()

        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        summary = next(r for r in records if r["_type"] == "run_summary")
        assert summary["success"] is False

    def test_emit_error(self, tmp_path: Path):
        path = tmp_path / "error.ndjson"
        with RunDiagnosticsCollector(str(path), [], {}) as diag:
            diag.emit_error("launch", ValueError("test error"))
            diag.emit_summary()

        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        err = next(r for r in records if r["_type"] == "run_error")
        assert "test error" in err["error"]
        assert err["traceback"] is not None

    def test_context_manager_on_exception(self, tmp_path: Path):
        path = tmp_path / "exc.ndjson"
        try:
            with RunDiagnosticsCollector(str(path), [], {}) as diag:
                diag.phase_start("crashing")
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        lines = path.read_text().strip().splitlines()
        records = [json.loads(ln) for ln in lines]
        types = [r["_type"] for r in records]
        assert "run_error" in types
        assert "run_summary" in types

    def test_emit_health_check(self, tmp_path: Path):
        path = tmp_path / "hc.ndjson"
        with RunDiagnosticsCollector(str(path), [], {}) as diag:
            diag.emit_health_check("http://host:8000/v1/models", 1, 200, True)

        lines = path.read_text().strip().splitlines()
        rec = json.loads(lines[0])
        assert rec["_type"] == "run_health_check"
        assert rec["success"] is True

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_capture_container_logs(self, mock_remote, tmp_path: Path):
        mock_remote.return_value = RemoteResult(
            host="10.0.0.1", returncode=0, stdout="log line 1\nlog line 2\n", stderr="",
        )

        path = tmp_path / "logs.ndjson"
        with RunDiagnosticsCollector(str(path), [], {}) as diag:
            diag.capture_container_logs("10.0.0.1", "sparkrun_abc_solo", {})

        lines = path.read_text().strip().splitlines()
        rec = json.loads(lines[0])
        assert rec["_type"] == "run_container_logs"
        assert len(rec["lines"]) == 2
