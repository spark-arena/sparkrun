"""Tests for container and model distribution plumbing.

Covers:
- SSH transfer primitives (build_ssh_opts_string, run_pipeline_to_remote,
  run_rsync, and their parallel variants)
- Container image hash checking (_check_remote_image_ids, _filter_hosts_needing_image)
- Container distribution (distribute_image_from_local, distribute_image_from_head)
- Model distribution (distribute_model_from_local, distribute_model_from_head)
- InfiniBand detection helpers (IBDetectionResult, extract_ib_ips, detect_ib_for_hosts)
- Transfer host routing (IB IPs for fast transfers)
"""

from __future__ import annotations

from unittest import mock

from sparkrun.orchestration.ssh import (
    RemoteResult,
    build_ssh_opts_string,
    run_pipeline_to_remote,
    run_pipeline_to_remotes_parallel,
    run_rsync,
    run_rsync_parallel,
)


# ---------------------------------------------------------------------------
# build_ssh_opts_string
# ---------------------------------------------------------------------------

class TestBuildSshOptsString:
    """Test SSH options string builder."""

    def test_defaults_only(self):
        """Minimal call produces BatchMode and ConnectTimeout."""
        result = build_ssh_opts_string()
        assert "-o BatchMode=yes" in result
        assert "-o ConnectTimeout=10" in result

    def test_custom_timeout(self):
        result = build_ssh_opts_string(connect_timeout=30)
        assert "ConnectTimeout=30" in result

    def test_with_key(self):
        result = build_ssh_opts_string(ssh_key="/path/to/key")
        assert "-i /path/to/key" in result

    def test_with_options(self):
        result = build_ssh_opts_string(ssh_options=["-o", "StrictHostKeyChecking=no"])
        assert "StrictHostKeyChecking=no" in result

    def test_with_all(self):
        result = build_ssh_opts_string(
            ssh_key="/key", ssh_options=["-v"], connect_timeout=5,
        )
        assert "-i /key" in result
        assert "-v" in result
        assert "ConnectTimeout=5" in result

    def test_user_not_in_string(self):
        """ssh_user is accepted but not embedded in the options string."""
        result = build_ssh_opts_string(ssh_user="admin")
        assert "admin" not in result


# ---------------------------------------------------------------------------
# run_pipeline_to_remote
# ---------------------------------------------------------------------------

class TestRunPipelineToRemote:
    """Test local-to-remote shell pipeline."""

    def test_dry_run(self):
        result = run_pipeline_to_remote(
            "host1", "echo hello", "cat",
            dry_run=True,
        )
        assert result.success
        assert result.host == "host1"
        assert "[dry-run]" in result.stdout

    @mock.patch("sparkrun.orchestration.ssh.subprocess.run")
    def test_constructs_pipeline(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0, stdout="ok", stderr="")
        run_pipeline_to_remote(
            "host1", "docker save img | gzip", "gunzip | docker load",
            ssh_user="user", ssh_key="/key",
        )
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        pipeline_str = call_args[0][0]
        assert "docker save img | gzip" in pipeline_str
        assert "ssh" in pipeline_str
        assert "user@host1" in pipeline_str
        assert "gunzip | docker load" in pipeline_str
        assert call_args[1]["shell"] is True

    @mock.patch("sparkrun.orchestration.ssh.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0, stdout="loaded", stderr="")
        result = run_pipeline_to_remote("host1", "echo hi", "cat")
        assert result.success
        assert result.stdout == "loaded"

    @mock.patch("sparkrun.orchestration.ssh.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="error")
        result = run_pipeline_to_remote("host1", "echo hi", "cat")
        assert not result.success
        assert result.returncode == 1

    @mock.patch("sparkrun.orchestration.ssh.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=10)
        result = run_pipeline_to_remote("host1", "echo hi", "cat", timeout=10)
        assert not result.success
        assert "timed out" in result.stderr.lower()

    @mock.patch("sparkrun.orchestration.ssh.subprocess.run")
    def test_no_user(self, mock_run):
        """Without ssh_user, host is used directly (not user@host)."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        run_pipeline_to_remote("myhost", "echo", "cat")
        pipeline_str = mock_run.call_args[0][0]
        assert "myhost" in pipeline_str
        assert "@myhost" not in pipeline_str


# ---------------------------------------------------------------------------
# run_rsync
# ---------------------------------------------------------------------------

class TestRunRsync:
    """Test rsync wrapper."""

    def test_dry_run(self):
        result = run_rsync("/src/path", "host1", "/dst/path", dry_run=True)
        assert result.success
        assert result.host == "host1"
        assert "[dry-run]" in result.stdout

    @mock.patch("sparkrun.orchestration.ssh.subprocess.run")
    def test_constructs_command(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        run_rsync(
            "/src/path", "host1", "/dst/path",
            ssh_user="user", ssh_key="/key",
        )
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "rsync"
        assert "-az" in cmd
        assert "--partial" in cmd
        assert "--links" in cmd
        # -e with ssh options
        e_idx = cmd.index("-e")
        ssh_arg = cmd[e_idx + 1]
        assert "ssh" in ssh_arg
        assert "-i /key" in ssh_arg
        # source has trailing slash
        assert "/src/path/" in cmd
        # destination is user@host:path
        assert "user@host1:/dst/path" in cmd

    @mock.patch("sparkrun.orchestration.ssh.subprocess.run")
    def test_trailing_slash_dedup(self, mock_run):
        """Trailing slash on source is not doubled."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        run_rsync("/src/path/", "host1", "/dst/path")
        cmd = mock_run.call_args[0][0]
        # Should end with single trailing slash, not double
        src_items = [c for c in cmd if c.startswith("/src/path")]
        assert src_items[0] == "/src/path/"

    @mock.patch("sparkrun.orchestration.ssh.subprocess.run")
    def test_custom_rsync_options(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        run_rsync(
            "/src", "host1", "/dst",
            rsync_options=["-avz", "--progress"],
        )
        cmd = mock_run.call_args[0][0]
        assert "-avz" in cmd
        assert "--progress" in cmd
        # Default options should NOT be present
        assert "--partial" not in cmd

    @mock.patch("sparkrun.orchestration.ssh.subprocess.run")
    def test_no_user(self, mock_run):
        """Without ssh_user, destination is host:path (not user@host:path)."""
        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        run_rsync("/src", "myhost", "/dst")
        cmd = mock_run.call_args[0][0]
        dest = [c for c in cmd if "myhost" in c][0]
        assert dest == "myhost:/dst"

    @mock.patch("sparkrun.orchestration.ssh.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=23, stdout="", stderr="rsync error")
        result = run_rsync("/src", "host1", "/dst")
        assert not result.success
        assert result.returncode == 23

    @mock.patch("sparkrun.orchestration.ssh.subprocess.run")
    def test_timeout(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="rsync", timeout=10)
        result = run_rsync("/src", "host1", "/dst", timeout=10)
        assert not result.success
        assert "timed out" in result.stderr.lower()


# ---------------------------------------------------------------------------
# Parallel variants
# ---------------------------------------------------------------------------

class TestParallelPipeline:
    """Test run_pipeline_to_remotes_parallel."""

    @mock.patch("sparkrun.orchestration.ssh.run_pipeline_to_remote")
    def test_calls_per_host(self, mock_pipe):
        mock_pipe.return_value = RemoteResult(host="h", returncode=0, stdout="", stderr="")
        hosts = ["h1", "h2", "h3"]
        results = run_pipeline_to_remotes_parallel(hosts, "cmd1", "cmd2")
        assert mock_pipe.call_count == 3
        called_hosts = {call.args[0] for call in mock_pipe.call_args_list}
        assert called_hosts == {"h1", "h2", "h3"}
        assert len(results) == 3

    def test_dry_run(self):
        results = run_pipeline_to_remotes_parallel(
            ["h1", "h2"], "echo", "cat", dry_run=True,
        )
        assert len(results) == 2
        assert all(r.success for r in results)


class TestParallelRsync:
    """Test run_rsync_parallel."""

    @mock.patch("sparkrun.orchestration.ssh.run_rsync")
    def test_calls_per_host(self, mock_rsync):
        mock_rsync.return_value = RemoteResult(host="h", returncode=0, stdout="", stderr="")
        hosts = ["h1", "h2"]
        results = run_rsync_parallel("/src", hosts, "/dst")
        assert mock_rsync.call_count == 2
        called_hosts = {call.args[1] for call in mock_rsync.call_args_list}
        assert called_hosts == {"h1", "h2"}
        assert len(results) == 2

    def test_dry_run(self):
        results = run_rsync_parallel("/src", ["h1", "h2"], "/dst", dry_run=True)
        assert len(results) == 2
        assert all(r.success for r in results)


# ---------------------------------------------------------------------------
# Image hash checking
# ---------------------------------------------------------------------------

class TestCheckRemoteImageIds:
    """Test _check_remote_image_ids."""

    @mock.patch("sparkrun.containers.distribute.run_remote_command")
    def test_returns_host_id_map(self, mock_cmd):
        """Returns mapping of host → image ID for hosts that have the image."""
        mock_cmd.side_effect = [
            RemoteResult(host="h1", returncode=0, stdout="sha256:abc123\n", stderr=""),
            RemoteResult(host="h2", returncode=0, stdout="sha256:def456\n", stderr=""),
        ]
        from sparkrun.containers.distribute import _check_remote_image_ids
        result = _check_remote_image_ids("img:latest", ["h1", "h2"])
        assert result == {"h1": "sha256:abc123", "h2": "sha256:def456"}

    @mock.patch("sparkrun.containers.distribute.run_remote_command")
    def test_skips_hosts_without_image(self, mock_cmd):
        """Hosts where the image is absent (empty stdout) are omitted."""
        mock_cmd.side_effect = [
            RemoteResult(host="h1", returncode=0, stdout="sha256:abc123\n", stderr=""),
            RemoteResult(host="h2", returncode=0, stdout="", stderr=""),
        ]
        from sparkrun.containers.distribute import _check_remote_image_ids
        result = _check_remote_image_ids("img:latest", ["h1", "h2"])
        assert result == {"h1": "sha256:abc123"}

    @mock.patch("sparkrun.containers.distribute.run_remote_command")
    def test_skips_failed_commands(self, mock_cmd):
        """Hosts where the SSH command failed are omitted."""
        mock_cmd.side_effect = [
            RemoteResult(host="h1", returncode=0, stdout="sha256:abc123\n", stderr=""),
            RemoteResult(host="h2", returncode=1, stdout="", stderr="error"),
        ]
        from sparkrun.containers.distribute import _check_remote_image_ids
        result = _check_remote_image_ids("img:latest", ["h1", "h2"])
        assert result == {"h1": "sha256:abc123"}

    def test_dry_run_returns_empty(self):
        from sparkrun.containers.distribute import _check_remote_image_ids
        result = _check_remote_image_ids("img:latest", ["h1", "h2"], dry_run=True)
        assert result == {}

    def test_empty_hosts_returns_empty(self):
        from sparkrun.containers.distribute import _check_remote_image_ids
        result = _check_remote_image_ids("img:latest", [])
        assert result == {}


class TestFilterHostsNeedingImage:
    """Test _filter_hosts_needing_image."""

    @mock.patch("sparkrun.containers.distribute._check_remote_image_ids")
    def test_all_hosts_up_to_date(self, mock_check):
        """Hosts with matching image ID are filtered out."""
        mock_check.return_value = {"h1": "sha256:abc", "h2": "sha256:abc"}
        from sparkrun.containers.distribute import _filter_hosts_needing_image
        result = _filter_hosts_needing_image("img", ["h1", "h2"], "sha256:abc")
        assert result == []

    @mock.patch("sparkrun.containers.distribute._check_remote_image_ids")
    def test_all_hosts_need_transfer(self, mock_check):
        """Hosts with missing or mismatched IDs are included."""
        mock_check.return_value = {"h1": "sha256:old"}  # h2 not present
        from sparkrun.containers.distribute import _filter_hosts_needing_image
        result = _filter_hosts_needing_image("img", ["h1", "h2"], "sha256:new")
        assert result == ["h1", "h2"]

    @mock.patch("sparkrun.containers.distribute._check_remote_image_ids")
    def test_partial_match(self, mock_check):
        """Only hosts with mismatched IDs are returned."""
        mock_check.return_value = {"h1": "sha256:abc", "h2": "sha256:old"}
        from sparkrun.containers.distribute import _filter_hosts_needing_image
        result = _filter_hosts_needing_image("img", ["h1", "h2"], "sha256:abc")
        assert result == ["h2"]

    def test_dry_run_returns_all(self):
        """Dry run skips checking and returns all hosts."""
        from sparkrun.containers.distribute import _filter_hosts_needing_image
        result = _filter_hosts_needing_image("img", ["h1", "h2"], "sha256:abc", dry_run=True)
        assert result == ["h1", "h2"]

    def test_no_local_id_returns_all(self):
        """When local image ID is None, all hosts need transfer."""
        from sparkrun.containers.distribute import _filter_hosts_needing_image
        result = _filter_hosts_needing_image("img", ["h1", "h2"], None)
        assert result == ["h1", "h2"]


# ---------------------------------------------------------------------------
# Container distribution
# ---------------------------------------------------------------------------

class TestDistributeImageFromLocal:
    """Test distribute_image_from_local."""

    @mock.patch("sparkrun.containers.distribute.run_pipeline_to_remotes_parallel")
    @mock.patch("sparkrun.containers.distribute.get_image_id", return_value=None)
    @mock.patch("sparkrun.containers.distribute.ensure_image")
    def test_dry_run(self, mock_ensure, mock_id, mock_parallel):
        mock_ensure.return_value = 0
        mock_parallel.return_value = [
            RemoteResult(host="h1", returncode=0, stdout="[dry-run]", stderr=""),
        ]
        from sparkrun.containers.distribute import distribute_image_from_local
        failed = distribute_image_from_local("img:latest", ["h1"], dry_run=True)
        assert failed == []

    @mock.patch("sparkrun.containers.distribute.run_pipeline_to_remotes_parallel")
    @mock.patch("sparkrun.containers.distribute.get_image_id", return_value=None)
    @mock.patch("sparkrun.containers.distribute.ensure_image")
    def test_success(self, mock_ensure, mock_id, mock_parallel):
        mock_ensure.return_value = 0
        mock_parallel.return_value = [
            RemoteResult(host="h1", returncode=0, stdout="ok", stderr=""),
            RemoteResult(host="h2", returncode=0, stdout="ok", stderr=""),
        ]
        from sparkrun.containers.distribute import distribute_image_from_local
        failed = distribute_image_from_local("img:latest", ["h1", "h2"])
        assert failed == []
        # Verify pipeline commands
        mock_parallel.assert_called_once()
        call_kwargs = mock_parallel.call_args
        assert "docker save img:latest" in call_kwargs[0][1]
        assert "docker load" in call_kwargs[0][2]

    @mock.patch("sparkrun.containers.distribute.run_pipeline_to_remotes_parallel")
    @mock.patch("sparkrun.containers.distribute.get_image_id", return_value=None)
    @mock.patch("sparkrun.containers.distribute.ensure_image")
    def test_ensure_fails(self, mock_ensure, mock_id, mock_parallel):
        """If ensure_image fails, all hosts are returned as failed."""
        mock_ensure.return_value = 1
        from sparkrun.containers.distribute import distribute_image_from_local
        failed = distribute_image_from_local("img:latest", ["h1", "h2"])
        assert failed == ["h1", "h2"]
        mock_parallel.assert_not_called()

    @mock.patch("sparkrun.containers.distribute.run_pipeline_to_remotes_parallel")
    @mock.patch("sparkrun.containers.distribute.get_image_id", return_value=None)
    @mock.patch("sparkrun.containers.distribute.ensure_image")
    def test_partial_failure(self, mock_ensure, mock_id, mock_parallel):
        """Only failed hosts are returned."""
        mock_ensure.return_value = 0
        mock_parallel.return_value = [
            RemoteResult(host="h1", returncode=0, stdout="ok", stderr=""),
            RemoteResult(host="h2", returncode=1, stdout="", stderr="err"),
        ]
        from sparkrun.containers.distribute import distribute_image_from_local
        failed = distribute_image_from_local("img:latest", ["h1", "h2"])
        assert failed == ["h2"]

    @mock.patch("sparkrun.containers.distribute.run_pipeline_to_remotes_parallel")
    @mock.patch("sparkrun.containers.distribute.get_image_id", return_value=None)
    @mock.patch("sparkrun.containers.distribute.ensure_image")
    def test_empty_hosts(self, mock_ensure, mock_id, mock_parallel):
        mock_ensure.return_value = 0
        from sparkrun.containers.distribute import distribute_image_from_local
        failed = distribute_image_from_local("img:latest", [])
        assert failed == []
        mock_parallel.assert_not_called()

    @mock.patch("sparkrun.containers.distribute.run_pipeline_to_remotes_parallel")
    @mock.patch("sparkrun.containers.distribute._filter_hosts_needing_image")
    @mock.patch("sparkrun.containers.distribute.get_image_id", return_value="sha256:abc")
    @mock.patch("sparkrun.containers.distribute.ensure_image")
    def test_hash_check_skips_up_to_date(self, mock_ensure, mock_id, mock_filter, mock_parallel):
        """When all hosts are up to date, no transfer happens."""
        mock_ensure.return_value = 0
        mock_filter.return_value = []  # all hosts already have the image
        from sparkrun.containers.distribute import distribute_image_from_local
        failed = distribute_image_from_local("img:latest", ["h1", "h2"])
        assert failed == []
        mock_parallel.assert_not_called()

    @mock.patch("sparkrun.containers.distribute.run_pipeline_to_remotes_parallel")
    @mock.patch("sparkrun.containers.distribute._filter_hosts_needing_image")
    @mock.patch("sparkrun.containers.distribute.get_image_id", return_value="sha256:abc")
    @mock.patch("sparkrun.containers.distribute.ensure_image")
    def test_hash_check_transfers_stale_hosts(self, mock_ensure, mock_id, mock_filter, mock_parallel):
        """Only hosts that need the image get the transfer."""
        mock_ensure.return_value = 0
        mock_filter.return_value = ["h2"]  # only h2 needs transfer
        mock_parallel.return_value = [
            RemoteResult(host="h2", returncode=0, stdout="ok", stderr=""),
        ]
        from sparkrun.containers.distribute import distribute_image_from_local
        failed = distribute_image_from_local("img:latest", ["h1", "h2"])
        assert failed == []
        # Only h2 should be in the transfer list
        transferred = mock_parallel.call_args[0][0]
        assert transferred == ["h2"]


class TestDistributeImageTransferHosts:
    """Test transfer_hosts routing in distribute_image_from_local."""

    @mock.patch("sparkrun.containers.distribute.run_pipeline_to_remotes_parallel")
    @mock.patch("sparkrun.containers.distribute.get_image_id", return_value=None)
    @mock.patch("sparkrun.containers.distribute.ensure_image")
    def test_transfer_hosts_used_for_pipeline(self, mock_ensure, mock_id, mock_parallel):
        """When transfer_hosts is provided, pipeline uses those IPs."""
        mock_ensure.return_value = 0
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="ok", stderr=""),
            RemoteResult(host="10.0.0.2", returncode=0, stdout="ok", stderr=""),
        ]
        from sparkrun.containers.distribute import distribute_image_from_local
        failed = distribute_image_from_local(
            "img:latest", ["h1", "h2"],
            transfer_hosts=["10.0.0.1", "10.0.0.2"],
        )
        assert failed == []
        # Pipeline should receive IB IPs, not management hosts
        transferred = mock_parallel.call_args[0][0]
        assert "10.0.0.1" in transferred
        assert "10.0.0.2" in transferred

    @mock.patch("sparkrun.containers.distribute.run_pipeline_to_remotes_parallel")
    @mock.patch("sparkrun.containers.distribute.get_image_id", return_value=None)
    @mock.patch("sparkrun.containers.distribute.ensure_image")
    def test_transfer_hosts_failure_maps_back(self, mock_ensure, mock_id, mock_parallel):
        """Failures on transfer IPs are reported as management hostnames."""
        mock_ensure.return_value = 0
        mock_parallel.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="ok", stderr=""),
            RemoteResult(host="10.0.0.2", returncode=1, stdout="", stderr="err"),
        ]
        from sparkrun.containers.distribute import distribute_image_from_local
        failed = distribute_image_from_local(
            "img:latest", ["mgmt1", "mgmt2"],
            transfer_hosts=["10.0.0.1", "10.0.0.2"],
        )
        # Failure should be reported using management hostname
        assert failed == ["mgmt2"]


class TestDistributeImageFromHead:
    """Test distribute_image_from_head."""

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_single_host(self, mock_run):
        """Single host: pull only, no distribution."""
        mock_run.return_value = RemoteResult(
            host="head", returncode=0, stdout="pulled", stderr="",
        )
        from sparkrun.containers.distribute import distribute_image_from_head
        failed = distribute_image_from_head("img:latest", ["head"])
        assert failed == []
        assert mock_run.call_count == 1

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_multi_host(self, mock_run):
        """Multi host: pull on head, then distribute script."""
        mock_run.return_value = RemoteResult(
            host="head", returncode=0, stdout="ok", stderr="",
        )
        from sparkrun.containers.distribute import distribute_image_from_head
        failed = distribute_image_from_head("img:latest", ["head", "w1", "w2"])
        assert failed == []
        # Called twice: once for pull, once for distribute
        assert mock_run.call_count == 2
        # Second call should contain the distribution script
        dist_script = mock_run.call_args_list[1][0][1]
        assert "w1" in dist_script
        assert "w2" in dist_script

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_pull_fails(self, mock_run):
        """If pull on head fails, all hosts are returned as failed."""
        mock_run.return_value = RemoteResult(
            host="head", returncode=1, stdout="", stderr="pull failed",
        )
        from sparkrun.containers.distribute import distribute_image_from_head
        failed = distribute_image_from_head("img:latest", ["head", "w1"])
        assert failed == ["head", "w1"]

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_distribute_fails(self, mock_run):
        """If distribution script fails, target hosts are returned as failed."""
        mock_run.side_effect = [
            RemoteResult(host="head", returncode=0, stdout="pulled", stderr=""),
            RemoteResult(host="head", returncode=1, stdout="", stderr="dist failed"),
        ]
        from sparkrun.containers.distribute import distribute_image_from_head
        failed = distribute_image_from_head("img:latest", ["head", "w1", "w2"])
        assert set(failed) == {"w1", "w2"}

    def test_empty_hosts(self):
        from sparkrun.containers.distribute import distribute_image_from_head
        failed = distribute_image_from_head("img:latest", [])
        assert failed == []

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_ssh_params_forwarded(self, mock_run):
        """SSH parameters are forwarded to remote script calls."""
        mock_run.return_value = RemoteResult(
            host="head", returncode=0, stdout="ok", stderr="",
        )
        from sparkrun.containers.distribute import distribute_image_from_head
        distribute_image_from_head(
            "img:latest", ["head"],
            ssh_user="admin", ssh_key="/mykey",
        )
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["ssh_user"] == "admin"
        assert call_kwargs["ssh_key"] == "/mykey"

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_worker_transfer_hosts(self, mock_run):
        """worker_transfer_hosts are used for distribution targets."""
        mock_run.return_value = RemoteResult(
            host="head", returncode=0, stdout="ok", stderr="",
        )
        from sparkrun.containers.distribute import distribute_image_from_head
        failed = distribute_image_from_head(
            "img:latest", ["head", "w1", "w2"],
            worker_transfer_hosts=["10.0.0.1", "10.0.0.2"],
        )
        assert failed == []
        # Distribution script should contain IB IPs, not management hosts
        dist_script = mock_run.call_args_list[1][0][1]
        assert "10.0.0.1" in dist_script
        assert "10.0.0.2" in dist_script


# ---------------------------------------------------------------------------
# Model distribution
# ---------------------------------------------------------------------------

class TestDistributeModelFromLocal:
    """Test distribute_model_from_local."""

    @mock.patch("sparkrun.models.distribute.run_rsync_parallel")
    @mock.patch("sparkrun.models.distribute.download_model")
    def test_dry_run(self, mock_dl, mock_rsync):
        mock_dl.return_value = 0
        mock_rsync.return_value = [
            RemoteResult(host="h1", returncode=0, stdout="[dry-run]", stderr=""),
        ]
        from sparkrun.models.distribute import distribute_model_from_local
        failed = distribute_model_from_local("org/model", ["h1"], dry_run=True)
        assert failed == []

    @mock.patch("sparkrun.models.distribute.run_rsync_parallel")
    @mock.patch("sparkrun.models.distribute.download_model")
    def test_success(self, mock_dl, mock_rsync):
        mock_dl.return_value = 0
        mock_rsync.return_value = [
            RemoteResult(host="h1", returncode=0, stdout="ok", stderr=""),
            RemoteResult(host="h2", returncode=0, stdout="ok", stderr=""),
        ]
        from sparkrun.models.distribute import distribute_model_from_local
        failed = distribute_model_from_local("org/model", ["h1", "h2"])
        assert failed == []
        # Verify rsync source path matches HF cache convention
        call_args = mock_rsync.call_args[0]
        assert "models--org--model" in call_args[0]

    @mock.patch("sparkrun.models.distribute.run_rsync_parallel")
    @mock.patch("sparkrun.models.distribute.download_model")
    def test_download_fails(self, mock_dl, mock_rsync):
        """If download fails, all hosts are returned as failed."""
        mock_dl.return_value = 1
        from sparkrun.models.distribute import distribute_model_from_local
        failed = distribute_model_from_local("org/model", ["h1", "h2"])
        assert failed == ["h1", "h2"]
        mock_rsync.assert_not_called()

    @mock.patch("sparkrun.models.distribute.run_rsync_parallel")
    @mock.patch("sparkrun.models.distribute.download_model")
    def test_partial_failure(self, mock_dl, mock_rsync):
        mock_dl.return_value = 0
        mock_rsync.return_value = [
            RemoteResult(host="h1", returncode=0, stdout="ok", stderr=""),
            RemoteResult(host="h2", returncode=1, stdout="", stderr="err"),
        ]
        from sparkrun.models.distribute import distribute_model_from_local
        failed = distribute_model_from_local("org/model", ["h1", "h2"])
        assert failed == ["h2"]

    @mock.patch("sparkrun.models.distribute.run_rsync_parallel")
    @mock.patch("sparkrun.models.distribute.download_model")
    def test_custom_cache_dir(self, mock_dl, mock_rsync):
        mock_dl.return_value = 0
        mock_rsync.return_value = [
            RemoteResult(host="h1", returncode=0, stdout="ok", stderr=""),
        ]
        from sparkrun.models.distribute import distribute_model_from_local
        distribute_model_from_local("org/model", ["h1"], cache_dir="/custom/cache")
        mock_dl.assert_called_once_with(
            "org/model", cache_dir="/custom/cache", token=None, revision=None, dry_run=False,
        )
        # Rsync path should use custom cache
        assert "/custom/cache/hub/models--org--model" in mock_rsync.call_args[0][0]

    @mock.patch("sparkrun.models.distribute.run_rsync_parallel")
    @mock.patch("sparkrun.models.distribute.download_model")
    def test_empty_hosts(self, mock_dl, mock_rsync):
        mock_dl.return_value = 0
        from sparkrun.models.distribute import distribute_model_from_local
        failed = distribute_model_from_local("org/model", [])
        assert failed == []
        mock_rsync.assert_not_called()

    @mock.patch("sparkrun.models.distribute.run_rsync_parallel")
    @mock.patch("sparkrun.models.distribute.download_model")
    def test_transfer_hosts_used(self, mock_dl, mock_rsync):
        """When transfer_hosts provided, rsync targets use IB IPs."""
        mock_dl.return_value = 0
        mock_rsync.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="ok", stderr=""),
            RemoteResult(host="10.0.0.2", returncode=0, stdout="ok", stderr=""),
        ]
        from sparkrun.models.distribute import distribute_model_from_local
        failed = distribute_model_from_local(
            "org/model", ["h1", "h2"],
            transfer_hosts=["10.0.0.1", "10.0.0.2"],
        )
        assert failed == []
        # Rsync should receive IB IPs
        rsynced_hosts = mock_rsync.call_args[0][1]
        assert rsynced_hosts == ["10.0.0.1", "10.0.0.2"]

    @mock.patch("sparkrun.models.distribute.run_rsync_parallel")
    @mock.patch("sparkrun.models.distribute.download_model")
    def test_transfer_hosts_failure_maps_back(self, mock_dl, mock_rsync):
        """Failures on transfer IPs are reported as management hostnames."""
        mock_dl.return_value = 0
        mock_rsync.return_value = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="ok", stderr=""),
            RemoteResult(host="10.0.0.2", returncode=1, stdout="", stderr="err"),
        ]
        from sparkrun.models.distribute import distribute_model_from_local
        failed = distribute_model_from_local(
            "org/model", ["mgmt1", "mgmt2"],
            transfer_hosts=["10.0.0.1", "10.0.0.2"],
        )
        assert failed == ["mgmt2"]


class TestDistributeModelFromHead:
    """Test distribute_model_from_head."""

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_single_host(self, mock_run):
        """Single host: download only, no distribution."""
        mock_run.return_value = RemoteResult(
            host="head", returncode=0, stdout="downloaded", stderr="",
        )
        from sparkrun.models.distribute import distribute_model_from_head
        failed = distribute_model_from_head("org/model", ["head"])
        assert failed == []
        assert mock_run.call_count == 1

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_multi_host(self, mock_run):
        """Multi host: download on head, then distribute script."""
        mock_run.return_value = RemoteResult(
            host="head", returncode=0, stdout="ok", stderr="",
        )
        from sparkrun.models.distribute import distribute_model_from_head
        failed = distribute_model_from_head("org/model", ["head", "w1", "w2"])
        assert failed == []
        assert mock_run.call_count == 2
        # Second call should contain the distribution script with targets
        dist_script = mock_run.call_args_list[1][0][1]
        assert "w1" in dist_script
        assert "w2" in dist_script
        assert "models--org--model" in dist_script

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_download_fails(self, mock_run):
        """If download on head fails, all hosts are returned as failed."""
        mock_run.return_value = RemoteResult(
            host="head", returncode=1, stdout="", stderr="dl failed",
        )
        from sparkrun.models.distribute import distribute_model_from_head
        failed = distribute_model_from_head("org/model", ["head", "w1"])
        assert failed == ["head", "w1"]

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_distribute_fails(self, mock_run):
        """If distribution script fails, target hosts are returned as failed."""
        mock_run.side_effect = [
            RemoteResult(host="head", returncode=0, stdout="ok", stderr=""),
            RemoteResult(host="head", returncode=1, stdout="", stderr="rsync failed"),
        ]
        from sparkrun.models.distribute import distribute_model_from_head
        failed = distribute_model_from_head("org/model", ["head", "w1", "w2"])
        assert set(failed) == {"w1", "w2"}

    def test_empty_hosts(self):
        from sparkrun.models.distribute import distribute_model_from_head
        failed = distribute_model_from_head("org/model", [])
        assert failed == []

    @mock.patch("sparkrun.orchestration.ssh.run_remote_script")
    def test_worker_transfer_hosts(self, mock_run):
        """worker_transfer_hosts are used for distribution targets."""
        mock_run.return_value = RemoteResult(
            host="head", returncode=0, stdout="ok", stderr="",
        )
        from sparkrun.models.distribute import distribute_model_from_head
        failed = distribute_model_from_head(
            "org/model", ["head", "w1", "w2"],
            worker_transfer_hosts=["10.0.0.1", "10.0.0.2"],
        )
        assert failed == []
        # Distribution script should contain IB IPs
        dist_script = mock_run.call_args_list[1][0][1]
        assert "10.0.0.1" in dist_script
        assert "10.0.0.2" in dist_script


# ---------------------------------------------------------------------------
# InfiniBand detection helpers
# ---------------------------------------------------------------------------

class TestIBDetectionResult:
    """Test IBDetectionResult dataclass."""

    def test_defaults(self):
        from sparkrun.orchestration.infiniband import IBDetectionResult
        result = IBDetectionResult()
        assert result.nccl_env == {}
        assert result.ib_ip_map == {}

    def test_with_data(self):
        from sparkrun.orchestration.infiniband import IBDetectionResult
        result = IBDetectionResult(
            nccl_env={"NCCL_NET": "IB"},
            ib_ip_map={"h1": "10.0.0.1"},
        )
        assert result.nccl_env == {"NCCL_NET": "IB"}
        assert result.ib_ip_map == {"h1": "10.0.0.1"}


class TestExtractIbIps:
    """Test extract_ib_ips."""

    def test_multiple_ips(self):
        from sparkrun.orchestration.infiniband import extract_ib_ips
        ib_info = {"DETECTED_IB_IPS": "10.0.0.1,10.0.0.2"}
        assert extract_ib_ips(ib_info) == ["10.0.0.1", "10.0.0.2"]

    def test_single_ip(self):
        from sparkrun.orchestration.infiniband import extract_ib_ips
        ib_info = {"DETECTED_IB_IPS": "10.0.0.1"}
        assert extract_ib_ips(ib_info) == ["10.0.0.1"]

    def test_empty_string(self):
        from sparkrun.orchestration.infiniband import extract_ib_ips
        ib_info = {"DETECTED_IB_IPS": ""}
        assert extract_ib_ips(ib_info) == []

    def test_missing_key(self):
        from sparkrun.orchestration.infiniband import extract_ib_ips
        assert extract_ib_ips({}) == []

    def test_whitespace_handling(self):
        from sparkrun.orchestration.infiniband import extract_ib_ips
        ib_info = {"DETECTED_IB_IPS": " 10.0.0.1 , 10.0.0.2 "}
        assert extract_ib_ips(ib_info) == ["10.0.0.1", "10.0.0.2"]


class TestDetectIbForHosts:
    """Test detect_ib_for_hosts."""

    @mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel")
    def test_returns_nccl_env_from_head(self, mock_parallel):
        """NCCL env is derived from the head host's detection."""
        mock_parallel.return_value = [
            RemoteResult(
                host="h1", returncode=0,
                stdout="IB_DETECTED=1\nDETECTED_GID_INDEX=3\nDETECTED_HCA_LIST=mlx5_0\n"
                       "DETECTED_NET_LIST=ib0\nDETECTED_IB_IPS=10.0.0.1\n",
                stderr="",
            ),
            RemoteResult(
                host="h2", returncode=0,
                stdout="IB_DETECTED=1\nDETECTED_IB_IPS=10.0.0.2\n",
                stderr="",
            ),
        ]
        from sparkrun.orchestration.infiniband import detect_ib_for_hosts
        result = detect_ib_for_hosts(["h1", "h2"])
        # NCCL env from head
        assert result.nccl_env.get("NCCL_NET") == "IB"
        assert result.nccl_env.get("NCCL_IB_GID_INDEX") == "3"
        # IB IPs for both hosts
        assert result.ib_ip_map == {"h1": "10.0.0.1", "h2": "10.0.0.2"}

    @mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel")
    def test_no_ib_detected(self, mock_parallel):
        """When no IB is found, returns empty results."""
        mock_parallel.return_value = [
            RemoteResult(host="h1", returncode=0, stdout="IB_DETECTED=0\n", stderr=""),
        ]
        from sparkrun.orchestration.infiniband import detect_ib_for_hosts
        result = detect_ib_for_hosts(["h1"])
        assert result.nccl_env == {}
        assert result.ib_ip_map == {}

    def test_empty_hosts(self):
        from sparkrun.orchestration.infiniband import detect_ib_for_hosts
        result = detect_ib_for_hosts([])
        assert result.nccl_env == {}
        assert result.ib_ip_map == {}

    @mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel")
    def test_partial_ib_detection(self, mock_parallel):
        """Only hosts with IB get IPs; all get NCCL from head."""
        mock_parallel.return_value = [
            RemoteResult(
                host="h1", returncode=0,
                stdout="IB_DETECTED=1\nDETECTED_HCA_LIST=mlx5_0\n"
                       "DETECTED_NET_LIST=ib0\nDETECTED_IB_IPS=10.0.0.1\n",
                stderr="",
            ),
            RemoteResult(
                host="h2", returncode=0,
                stdout="IB_DETECTED=0\n",
                stderr="",
            ),
        ]
        from sparkrun.orchestration.infiniband import detect_ib_for_hosts
        result = detect_ib_for_hosts(["h1", "h2"])
        assert result.nccl_env.get("NCCL_NET") == "IB"
        assert "h1" in result.ib_ip_map
        assert "h2" not in result.ib_ip_map

    @mock.patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel")
    def test_failed_detection_on_host(self, mock_parallel):
        """Hosts where detection fails are skipped gracefully."""
        mock_parallel.return_value = [
            RemoteResult(host="h1", returncode=1, stdout="", stderr="ssh error"),
            RemoteResult(
                host="h2", returncode=0,
                stdout="IB_DETECTED=1\nDETECTED_HCA_LIST=mlx5_0\n"
                       "DETECTED_NET_LIST=ib0\nDETECTED_IB_IPS=10.0.0.2\n",
                stderr="",
            ),
        ]
        from sparkrun.orchestration.infiniband import detect_ib_for_hosts
        result = detect_ib_for_hosts(["h1", "h2"])
        # h1 failed, so NCCL env comes from h2 (not the head)
        # But since h1 is the head and failed, we may get empty NCCL
        # unless h2's result is used as fallback
        assert "h1" not in result.ib_ip_map
        assert result.ib_ip_map.get("h2") == "10.0.0.2"


# ---------------------------------------------------------------------------
# Push-mode distribution helpers
# ---------------------------------------------------------------------------

class TestDistributeImagePush:
    """Test _distribute_image_push helper."""

    @mock.patch("sparkrun.containers.distribute.distribute_image_from_head")
    @mock.patch("sparkrun.containers.distribute.distribute_image_from_local")
    def test_single_host_push(self, mock_local, mock_head):
        """Single host: push to head only, no head-to-worker distribution."""
        mock_local.return_value = []
        from sparkrun.orchestration.distribution import _distribute_image_push
        failed = _distribute_image_push(
            "img:latest", ["head"],
            worker_transfer_hosts=None,
            ssh_kwargs={}, dry_run=False,
        )
        assert failed == []
        mock_local.assert_called_once()
        mock_head.assert_not_called()

    @mock.patch("sparkrun.containers.distribute.distribute_image_from_head")
    @mock.patch("sparkrun.containers.distribute.distribute_image_from_local")
    def test_multi_host_push(self, mock_local, mock_head):
        """Multi host: push to head, then head distributes to workers."""
        mock_local.return_value = []
        mock_head.return_value = []
        from sparkrun.orchestration.distribution import _distribute_image_push
        failed = _distribute_image_push(
            "img:latest", ["head", "w1", "w2"],
            worker_transfer_hosts=["10.0.0.1", "10.0.0.2"],
            ssh_kwargs={}, dry_run=False,
        )
        assert failed == []
        # Local push should target only the head
        local_hosts = mock_local.call_args[0][1]
        assert local_hosts == ["head"]
        # Head distribution should use worker_transfer_hosts
        mock_head.assert_called_once()
        head_kwargs = mock_head.call_args[1]
        assert head_kwargs["worker_transfer_hosts"] == ["10.0.0.1", "10.0.0.2"]

    @mock.patch("sparkrun.containers.distribute.distribute_image_from_head")
    @mock.patch("sparkrun.containers.distribute.distribute_image_from_local")
    def test_head_push_fails(self, mock_local, mock_head):
        """If push to head fails, all hosts are returned as failed."""
        mock_local.return_value = ["head"]
        from sparkrun.orchestration.distribution import _distribute_image_push
        failed = _distribute_image_push(
            "img:latest", ["head", "w1"],
            worker_transfer_hosts=None,
            ssh_kwargs={}, dry_run=False,
        )
        assert failed == ["head", "w1"]
        mock_head.assert_not_called()


class TestDistributeModelPush:
    """Test _distribute_model_push helper."""

    @mock.patch("sparkrun.models.distribute.distribute_model_from_head")
    @mock.patch("sparkrun.models.distribute.distribute_model_from_local")
    def test_single_host_push(self, mock_local, mock_head):
        """Single host: push to head only."""
        mock_local.return_value = []
        from sparkrun.orchestration.distribution import _distribute_model_push
        failed = _distribute_model_push(
            "org/model", ["head"],
            cache_dir="/cache",
            worker_transfer_hosts=None,
            ssh_kwargs={}, dry_run=False,
        )
        assert failed == []
        mock_local.assert_called_once()
        mock_head.assert_not_called()

    @mock.patch("sparkrun.models.distribute.distribute_model_from_head")
    @mock.patch("sparkrun.models.distribute.distribute_model_from_local")
    def test_multi_host_push(self, mock_local, mock_head):
        """Multi host: push to head, then head distributes to workers."""
        mock_local.return_value = []
        mock_head.return_value = []
        from sparkrun.orchestration.distribution import _distribute_model_push
        failed = _distribute_model_push(
            "org/model", ["head", "w1", "w2"],
            cache_dir="/cache",
            worker_transfer_hosts=["10.0.0.1", "10.0.0.2"],
            ssh_kwargs={}, dry_run=False,
        )
        assert failed == []
        local_hosts = mock_local.call_args[0][1]
        assert local_hosts == ["head"]
        mock_head.assert_called_once()

    @mock.patch("sparkrun.models.distribute.distribute_model_from_head")
    @mock.patch("sparkrun.models.distribute.distribute_model_from_local")
    def test_head_push_fails(self, mock_local, mock_head):
        """If push to head fails, all hosts are returned as failed."""
        mock_local.return_value = ["head"]
        from sparkrun.orchestration.distribution import _distribute_model_push
        failed = _distribute_model_push(
            "org/model", ["head", "w1"],
            cache_dir="/cache",
            worker_transfer_hosts=None,
            ssh_kwargs={}, dry_run=False,
        )
        assert failed == ["head", "w1"]
        mock_head.assert_not_called()


# ---------------------------------------------------------------------------
# distribute_resources transfer_mode routing
# ---------------------------------------------------------------------------

class TestDistributeResourcesTransferMode:
    """Test that distribute_resources routes to the correct distribution
    functions based on transfer_mode."""

    def _make_config(self):
        """Create a minimal mock SparkrunConfig."""
        cfg = mock.MagicMock()
        cfg.cache_dir = "/tmp/cache"
        cfg.ssh_user = None
        cfg.ssh_key = None
        cfg.ssh_options = None
        return cfg

    @mock.patch("sparkrun.containers.distribute.distribute_image_from_local")
    @mock.patch("sparkrun.orchestration.infiniband.validate_ib_connectivity", return_value={})
    @mock.patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
    @mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
    def test_local_mode_uses_from_local(self, mock_ssh, mock_ib, mock_validate, mock_img):
        """Local mode calls distribute_image_from_local."""
        mock_ib.return_value = mock.MagicMock(nccl_env={}, ib_ip_map={}, mgmt_ip_map={})
        mock_img.return_value = []
        from sparkrun.orchestration.distribution import distribute_resources
        distribute_resources(
            "img:latest", "", ["h1", "h2"], "/cache",
            self._make_config(), dry_run=True,
            transfer_mode="local",
        )
        mock_img.assert_called_once()
        mock_validate.assert_called_once()

    @mock.patch("sparkrun.orchestration.distribution._distribute_image_push")
    @mock.patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
    @mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
    def test_push_mode_uses_push_helper(self, mock_ssh, mock_ib, mock_push):
        """Push mode calls _distribute_image_push."""
        mock_ib.return_value = mock.MagicMock(nccl_env={}, ib_ip_map={}, mgmt_ip_map={})
        mock_push.return_value = []
        from sparkrun.orchestration.distribution import distribute_resources
        distribute_resources(
            "img:latest", "", ["h1", "h2"], "/cache",
            self._make_config(), dry_run=True,
            transfer_mode="push",
        )
        mock_push.assert_called_once()

    @mock.patch("sparkrun.containers.distribute.distribute_image_from_head")
    @mock.patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
    @mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
    def test_delegated_mode_uses_from_head(self, mock_ssh, mock_ib, mock_head):
        """Delegated mode calls distribute_image_from_head."""
        mock_ib.return_value = mock.MagicMock(nccl_env={}, ib_ip_map={}, mgmt_ip_map={})
        mock_head.return_value = []
        from sparkrun.orchestration.distribution import distribute_resources
        distribute_resources(
            "img:latest", "", ["h1", "h2"], "/cache",
            self._make_config(), dry_run=True,
            transfer_mode="delegated",
        )
        mock_head.assert_called_once()

    @mock.patch("sparkrun.orchestration.distribution._distribute_image_push", return_value=[])
    @mock.patch("sparkrun.orchestration.infiniband.validate_ib_connectivity", return_value={})
    @mock.patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
    @mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
    def test_push_mode_skips_ib_validation(self, mock_ssh, mock_ib, mock_validate, mock_push):
        """Push mode does not call validate_ib_connectivity."""
        mock_ib.return_value = mock.MagicMock(nccl_env={}, ib_ip_map={}, mgmt_ip_map={})
        from sparkrun.orchestration.distribution import distribute_resources
        distribute_resources(
            "img:latest", "", ["h1", "h2"], "/cache",
            self._make_config(), dry_run=True,
            transfer_mode="push",
        )
        mock_validate.assert_not_called()

    @mock.patch("sparkrun.orchestration.distribution._distribute_model_push")
    @mock.patch("sparkrun.orchestration.distribution._distribute_image_push")
    @mock.patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
    @mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
    def test_push_mode_with_model(self, mock_ssh, mock_ib, mock_img_push, mock_mdl_push):
        """Push mode routes both image and model through push helpers."""
        mock_ib.return_value = mock.MagicMock(nccl_env={}, ib_ip_map={}, mgmt_ip_map={})
        mock_img_push.return_value = []
        mock_mdl_push.return_value = []
        from sparkrun.orchestration.distribution import distribute_resources
        distribute_resources(
            "img:latest", "org/model", ["h1", "h2"], "/cache",
            self._make_config(), dry_run=True,
            transfer_mode="push",
        )
        mock_img_push.assert_called_once()
        mock_mdl_push.assert_called_once()

    @mock.patch("sparkrun.models.distribute.distribute_model_from_head")
    @mock.patch("sparkrun.containers.distribute.distribute_image_from_head")
    @mock.patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
    @mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
    def test_delegated_mode_with_model(self, mock_ssh, mock_ib, mock_img_head, mock_mdl_head):
        """Delegated mode routes both image and model through from_head."""
        mock_ib.return_value = mock.MagicMock(nccl_env={}, ib_ip_map={}, mgmt_ip_map={})
        mock_img_head.return_value = []
        mock_mdl_head.return_value = []
        from sparkrun.orchestration.distribution import distribute_resources
        distribute_resources(
            "img:latest", "org/model", ["h1", "h2"], "/cache",
            self._make_config(), dry_run=True,
            transfer_mode="delegated",
        )
        mock_img_head.assert_called_once()
        mock_mdl_head.assert_called_once()

    @mock.patch("sparkrun.orchestration.distribution._distribute_image_push", return_value=[])
    @mock.patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
    @mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
    def test_push_mode_computes_worker_transfer_hosts(self, mock_ssh, mock_ib, mock_push):
        """Push mode builds worker_transfer_hosts from IB IPs for workers."""
        mock_ib.return_value = mock.MagicMock(
            nccl_env={}, mgmt_ip_map={},
            ib_ip_map={"h1": "10.0.0.1", "h2": "10.0.0.2", "h3": "10.0.0.3"},
        )
        from sparkrun.orchestration.distribution import distribute_resources
        distribute_resources(
            "img:latest", "", ["h1", "h2", "h3"], "/cache",
            self._make_config(), dry_run=True,
            transfer_mode="push",
        )
        call_kwargs = mock_push.call_args[1]
        assert call_kwargs["worker_transfer_hosts"] == ["10.0.0.2", "10.0.0.3"]

    @mock.patch("sparkrun.containers.distribute.distribute_image_from_local")
    @mock.patch("sparkrun.orchestration.infiniband.validate_ib_connectivity")
    @mock.patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
    @mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
    def test_auto_resolves_to_local_with_ib(self, mock_ssh, mock_ib, mock_validate, mock_img):
        """Auto mode resolves to local when IB is reachable from control node."""
        mock_ib.return_value = mock.MagicMock(
            nccl_env={}, mgmt_ip_map={},
            ib_ip_map={"h1": "10.0.0.1", "h2": "10.0.0.2"},
        )
        mock_validate.return_value = {"h1": "10.0.0.1", "h2": "10.0.0.2"}
        mock_img.return_value = []
        from sparkrun.orchestration.distribution import distribute_resources
        distribute_resources(
            "img:latest", "", ["h1", "h2"], "/cache",
            self._make_config(), dry_run=True,
            transfer_mode="auto",
        )
        mock_validate.assert_called_once()
        mock_img.assert_called_once()

    @mock.patch("sparkrun.orchestration.distribution._distribute_image_push")
    @mock.patch("sparkrun.orchestration.infiniband.validate_ib_connectivity")
    @mock.patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
    @mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
    def test_auto_resolves_to_push_without_ib(self, mock_ssh, mock_ib, mock_validate, mock_push):
        """Auto mode resolves to push when IB is not reachable from control node."""
        mock_ib.return_value = mock.MagicMock(
            nccl_env={}, mgmt_ip_map={},
            ib_ip_map={"h1": "10.0.0.1", "h2": "10.0.0.2"},
        )
        mock_validate.return_value = {}
        mock_push.return_value = []
        from sparkrun.orchestration.distribution import distribute_resources
        distribute_resources(
            "img:latest", "", ["h1", "h2"], "/cache",
            self._make_config(), dry_run=True,
            transfer_mode="auto",
        )
        mock_validate.assert_called_once()
        mock_push.assert_called_once()

    @mock.patch("sparkrun.orchestration.distribution._distribute_image_push")
    @mock.patch("sparkrun.orchestration.infiniband.validate_ib_connectivity")
    @mock.patch("sparkrun.orchestration.infiniband.detect_ib_for_hosts")
    @mock.patch("sparkrun.orchestration.primitives.build_ssh_kwargs", return_value={})
    def test_auto_resolves_to_push_no_ib_hardware(self, mock_ssh, mock_ib, mock_validate, mock_push):
        """Auto mode resolves to push when no IB hardware detected."""
        mock_ib.return_value = mock.MagicMock(
            nccl_env={}, mgmt_ip_map={}, ib_ip_map={},
        )
        mock_validate.return_value = {}
        mock_push.return_value = []
        from sparkrun.orchestration.distribution import distribute_resources
        distribute_resources(
            "img:latest", "", ["h1", "h2"], "/cache",
            self._make_config(), dry_run=True,
            transfer_mode="auto",
        )
        mock_validate.assert_called_once()
        mock_push.assert_called_once()
