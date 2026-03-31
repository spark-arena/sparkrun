# """Tests for nv-monitor orchestration module."""
#
# from __future__ import annotations
#
# from unittest.mock import MagicMock, patch
#
# from sparkrun.orchestration.nv_monitor import (
#     ensure_nv_monitor,
#     scrape_metrics,
#     start_nv_monitor_ssh,
#     stop_nv_monitor_ssh,
#     stop_nv_monitor_remote,
# )
#
#
# class TestEnsureNvMonitor:
#     @patch("sparkrun.orchestration.nv_monitor.run_rsync_parallel")
#     @patch("sparkrun.orchestration.nv_monitor.run_remote_scripts_parallel")
#     @patch("sparkrun.bin.get_binary_resource")
#     @patch("sparkrun.bin.get_binary_checksum")
#     def test_all_hosts_up_to_date(self, mock_checksum, mock_resource, mock_parallel, mock_rsync):
#         """When all hosts have matching checksums, no deploy needed."""
#         mock_checksum.return_value = "abc123"
#         # Simulate remote checksum check returning matching hash
#         mock_result = MagicMock()
#         mock_result.success = True
#         mock_result.stdout = "abc123\n"
#         mock_result.host = "host1"
#         mock_parallel.return_value = [mock_result]
#
#         result = ensure_nv_monitor(["host1"], {"ssh_user": "user"})
#         assert result == {"host1": True}
#         mock_rsync.assert_not_called()
#
#     @patch("sparkrun.orchestration.nv_monitor.run_rsync_parallel")
#     @patch("sparkrun.orchestration.nv_monitor.run_remote_scripts_parallel")
#     @patch("sparkrun.bin.get_binary_resource")
#     @patch("sparkrun.bin.get_binary_checksum")
#     def test_missing_binary_triggers_deploy(self, mock_checksum, mock_resource, mock_parallel, mock_rsync):
#         """When binary is missing, should deploy via rsync."""
#         mock_checksum.return_value = "abc123"
#
#         # First call: checksum check returns MISSING
#         check_result = MagicMock()
#         check_result.success = True
#         check_result.stdout = "MISSING\n"
#         check_result.host = "host1"
#
#         # Second call: mkdir (script outputs OK on success)
#         mkdir_result = MagicMock()
#         mkdir_result.success = True
#         mkdir_result.stdout = "OK\n"
#         mkdir_result.host = "host1"
#
#         mock_parallel.side_effect = [[check_result], [mkdir_result]]
#
#         # Mock binary resource context manager
#         import tempfile
#         from pathlib import Path
#         with tempfile.NamedTemporaryFile(delete=False, suffix="-nv-monitor") as tmp:
#             tmp.write(b"fake binary")
#             tmp_path = Path(tmp.name)
#
#         from contextlib import contextmanager
#         @contextmanager
#         def fake_resource(name):
#             yield tmp_path
#
#         mock_resource.side_effect = fake_resource
#
#         # Mock rsync success
#         rsync_result = MagicMock()
#         rsync_result.success = True
#         rsync_result.host = "host1"
#         mock_rsync.return_value = [rsync_result]
#
#         result = ensure_nv_monitor(["host1"], {"ssh_user": "user"})
#         assert result["host1"] is True
#         mock_rsync.assert_called_once()
#
#         # Cleanup
#         tmp_path.unlink(missing_ok=True)
#
#     @patch("sparkrun.orchestration.nv_monitor.run_remote_scripts_parallel")
#     @patch("sparkrun.bin.get_binary_checksum")
#     def test_checksum_mismatch_triggers_deploy(self, mock_checksum, mock_parallel):
#         """When checksums don't match, should redeploy."""
#         mock_checksum.return_value = "abc123"
#         check_result = MagicMock()
#         check_result.success = True
#         check_result.stdout = "different_hash\n"
#         check_result.host = "host1"
#         mock_parallel.return_value = [check_result]
#
#         # This will fail at the deploy step (no rsync mock), but we verify
#         # it identified the mismatch
#         with patch("sparkrun.orchestration.nv_monitor.run_rsync_parallel") as mock_rsync, \
#              patch("sparkrun.bin.get_binary_resource") as mock_resource:
#             import tempfile
#             from pathlib import Path
#             from contextlib import contextmanager
#
#             with tempfile.NamedTemporaryFile(delete=False, suffix="-nv-monitor") as tmp:
#                 tmp.write(b"fake")
#                 tmp_path = Path(tmp.name)
#
#             @contextmanager
#             def fake_resource(name):
#                 yield tmp_path
#
#             mock_resource.side_effect = fake_resource
#
#             rsync_result = MagicMock()
#             rsync_result.success = True
#             rsync_result.host = "host1"
#             mock_rsync.return_value = [rsync_result]
#
#             # Need mkdir call too (script outputs OK on success)
#             mkdir_result = MagicMock(success=True, stdout="OK\n", host="host1")
#             mock_parallel.side_effect = [[check_result], [mkdir_result]]
#
#             result = ensure_nv_monitor(["host1"], {"ssh_user": "user"})
#             assert result["host1"] is True
#             mock_rsync.assert_called_once()
#
#             tmp_path.unlink(missing_ok=True)
#
#     def test_dry_run(self):
#         result = ensure_nv_monitor(["host1", "host2"], {}, dry_run=True)
#         assert result == {"host1": True, "host2": True}
#
#
# class TestStartStopNvMonitor:
#     @patch("sparkrun.orchestration.nv_monitor.subprocess.Popen")
#     @patch("sparkrun.orchestration.nv_monitor.build_ssh_cmd")
#     def test_start_returns_popen(self, mock_build_ssh, mock_popen):
#         mock_build_ssh.return_value = ["ssh", "host1"]
#         mock_proc = MagicMock()
#         mock_popen.return_value = mock_proc
#
#         result = start_nv_monitor_ssh("host1", {}, port=29110, local_forward_port=30110)
#         assert result is mock_proc
#         # Verify port forwarding args
#         call_args = mock_popen.call_args[0][0]
#         assert "-L" in call_args
#         assert "30110:localhost:29110" in call_args
#
#     @patch("sparkrun.orchestration.nv_monitor.subprocess.Popen")
#     @patch("sparkrun.orchestration.nv_monitor.build_ssh_cmd")
#     def test_start_failure_returns_none(self, mock_build_ssh, mock_popen):
#         mock_build_ssh.return_value = ["ssh", "host1"]
#         mock_popen.side_effect = OSError("connection failed")
#
#         result = start_nv_monitor_ssh("host1", {})
#         assert result is None
#
#     def test_stop_none_is_safe(self):
#         stop_nv_monitor_ssh(None)  # Should not raise
#
#     def test_stop_terminates_process(self):
#         mock_proc = MagicMock()
#         mock_proc.wait.return_value = 0
#         stop_nv_monitor_ssh(mock_proc)
#         mock_proc.terminate.assert_called_once()
#
#     @patch("sparkrun.orchestration.nv_monitor.run_remote_command")
#     def test_stop_remote(self, mock_cmd):
#         stop_nv_monitor_remote("host1", {"ssh_user": "user"}, port=29110)
#         mock_cmd.assert_called_once()
#         # Verify pkill command contains the port
#         call_args = mock_cmd.call_args
#         assert "29110" in call_args[0][1]
#
#
# class TestScrapeMetrics:
#     @patch("socket.create_connection")
#     def test_successful_scrape(self, mock_conn):
#         mock_sock = MagicMock()
#         # Simulate HTTP response with headers + body
#         mock_sock.recv.side_effect = [
#             b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\nsimple_metric 42.0\n",
#             b"",
#         ]
#         mock_conn.return_value = mock_sock
#
#         result = scrape_metrics("http://localhost:29110/metrics")
#         assert result["simple_metric"] == 42.0
#         mock_sock.close.assert_called_once()
#
#     @patch("socket.create_connection")
#     def test_failed_scrape_returns_empty(self, mock_conn):
#         mock_conn.side_effect = Exception("connection refused")
#         result = scrape_metrics("http://localhost:29110/metrics")
#         assert result == {}
