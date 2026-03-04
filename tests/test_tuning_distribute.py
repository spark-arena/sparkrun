"""Tests for sparkrun.tuning.distribute module."""

from __future__ import annotations

from unittest.mock import patch

from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.tuning.distribute import distribute_tuning_to_hosts


class TestNoopWhenNoTuningDir:
    """distribute_tuning_to_hosts is a no-op when there is no local tuning dir."""

    def test_noop_when_no_tuning_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tmp_path / "nonexistent",
        )
        with patch("sparkrun.orchestration.ssh.run_rsync_parallel") as mock_rsync:
            result = distribute_tuning_to_hosts("sglang", ["10.0.0.1"])
            mock_rsync.assert_not_called()
        assert result == []


class TestNoopWhenTuningDirEmpty:
    """distribute_tuning_to_hosts is a no-op when dir has no .json files."""

    def test_noop_when_tuning_dir_empty(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )
        with patch("sparkrun.orchestration.ssh.run_rsync_parallel") as mock_rsync:
            result = distribute_tuning_to_hosts("sglang", ["10.0.0.1"])
            mock_rsync.assert_not_called()
        assert result == []

    def test_noop_when_dir_has_non_json_files(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "readme.txt").write_text("not a config")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )
        with patch("sparkrun.orchestration.ssh.run_rsync_parallel") as mock_rsync:
            result = distribute_tuning_to_hosts("sglang", ["10.0.0.1"])
            mock_rsync.assert_not_called()
        assert result == []


class TestNoopWhenAllHostsLocal:
    """distribute_tuning_to_hosts is a no-op when all hosts are localhost."""

    def test_noop_when_all_hosts_local(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )
        with patch("sparkrun.orchestration.ssh.run_rsync_parallel") as mock_rsync:
            result = distribute_tuning_to_hosts("sglang", ["localhost"])
            mock_rsync.assert_not_called()
        assert result == []

    def test_noop_with_127_0_0_1(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )
        with patch("sparkrun.orchestration.ssh.run_rsync_parallel") as mock_rsync:
            result = distribute_tuning_to_hosts("sglang", ["127.0.0.1"])
            mock_rsync.assert_not_called()
        assert result == []


class TestDistributesToRemoteHosts:
    """distribute_tuning_to_hosts calls rsync with correct args."""

    def test_distributes_to_remote_hosts(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "E=128_N=256.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )

        mock_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr=""),
            RemoteResult(host="10.0.0.2", returncode=0, stdout="", stderr=""),
        ]
        with patch(
            "sparkrun.orchestration.ssh.run_rsync_parallel",
            return_value=mock_results,
        ) as mock_rsync:
            result = distribute_tuning_to_hosts(
                "sglang", ["10.0.0.1", "10.0.0.2"],
                ssh_user="testuser",
            )
            mock_rsync.assert_called_once()
            call_kwargs = mock_rsync.call_args
            # source and dest are both the tuning dir path
            assert call_kwargs[0][0] == str(tuning_dir)  # source_path
            assert call_kwargs[0][1] == ["10.0.0.1", "10.0.0.2"]  # hosts
            assert call_kwargs[0][2] == str(tuning_dir)  # dest_path
            assert "--delete" in call_kwargs[1]["rsync_options"]
            assert "-az" in call_kwargs[1]["rsync_options"]
            assert "--partial" in call_kwargs[1]["rsync_options"]
            assert call_kwargs[1]["ssh_user"] == "testuser"
        assert result == []

    def test_distributes_with_nested_json(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "vllm"
        nested = tuning_dir / "triton_3_6_0"
        nested.mkdir(parents=True)
        (nested / "E=128_N=256.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )

        mock_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr=""),
        ]
        with patch(
            "sparkrun.orchestration.ssh.run_rsync_parallel",
            return_value=mock_results,
        ) as mock_rsync:
            result = distribute_tuning_to_hosts("vllm", ["10.0.0.1"])
            mock_rsync.assert_called_once()
        assert result == []


class TestReturnsFailedHosts:
    """distribute_tuning_to_hosts returns failed hostnames."""

    def test_returns_failed_hosts(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )

        mock_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr=""),
            RemoteResult(host="10.0.0.2", returncode=1, stdout="", stderr="rsync error"),
            RemoteResult(host="10.0.0.3", returncode=1, stdout="", stderr="timeout"),
        ]
        with patch(
            "sparkrun.orchestration.ssh.run_rsync_parallel",
            return_value=mock_results,
        ):
            result = distribute_tuning_to_hosts(
                "sglang", ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
            )
        assert result == ["10.0.0.2", "10.0.0.3"]


class TestDryRunPassthrough:
    """distribute_tuning_to_hosts forwards dry_run to rsync."""

    def test_dry_run_passthrough(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )

        mock_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr=""),
        ]
        with patch(
            "sparkrun.orchestration.ssh.run_rsync_parallel",
            return_value=mock_results,
        ) as mock_rsync:
            distribute_tuning_to_hosts(
                "sglang", ["10.0.0.1"], dry_run=True,
            )
            assert mock_rsync.call_args[1]["dry_run"] is True


class TestRuntimeNormalization:
    """vllm-ray resolves to the vllm tuning directory."""

    def test_vllm_ray_resolves_to_vllm(self, tmp_path, monkeypatch):
        """Exercises _get_local_tuning_dir indirectly via the real function."""
        from sparkrun.tuning.sync import _get_local_tuning_dir
        vllm_dir = _get_local_tuning_dir("vllm-ray")
        assert str(vllm_dir).endswith("sparkrun/tuning/vllm")

    def test_vllm_distributed_resolves_to_vllm(self, tmp_path, monkeypatch):
        from sparkrun.tuning.sync import _get_local_tuning_dir
        vllm_dir = _get_local_tuning_dir("vllm-distributed")
        assert str(vllm_dir).endswith("sparkrun/tuning/vllm")

    def test_eugr_vllm_resolves_to_vllm(self, tmp_path, monkeypatch):
        from sparkrun.tuning.sync import _get_local_tuning_dir
        vllm_dir = _get_local_tuning_dir("eugr-vllm")
        assert str(vllm_dir).endswith("sparkrun/tuning/vllm")


class TestTransferModePush:
    """distribute_tuning_to_hosts with push/delegated mode uses two-hop distribution."""

    def test_push_mode_two_hop(self, tmp_path, monkeypatch):
        """Push mode rsyncs to head, then runs dist script on head."""
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )

        rsync_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr=""),
        ]
        dist_result = RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr="")

        with patch(
            "sparkrun.orchestration.ssh.run_rsync_parallel",
            return_value=rsync_results,
        ) as mock_rsync, patch(
            "sparkrun.orchestration.ssh.run_remote_script",
            return_value=dist_result,
        ) as mock_script:
            result = distribute_tuning_to_hosts(
                "sglang", ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
                transfer_mode="push",
            )
            # Should rsync to head only
            mock_rsync.assert_called_once()
            rsync_hosts = mock_rsync.call_args[0][1]
            assert rsync_hosts == ["10.0.0.1"]
            # Should run distribution script on head
            mock_script.assert_called_once()
            script_host = mock_script.call_args[0][0]
            assert script_host == "10.0.0.1"
            # Script should reference worker hosts
            script_content = mock_script.call_args[0][1]
            assert "10.0.0.2" in script_content
            assert "10.0.0.3" in script_content
        assert result == []

    def test_push_mode_single_host_falls_back_to_direct(self, tmp_path, monkeypatch):
        """Push mode with single remote host uses direct rsync (no two-hop)."""
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )

        mock_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr=""),
        ]
        with patch(
            "sparkrun.orchestration.ssh.run_rsync_parallel",
            return_value=mock_results,
        ) as mock_rsync:
            result = distribute_tuning_to_hosts(
                "sglang", ["10.0.0.1"],
                transfer_mode="push",
            )
            # Single host: direct rsync, no distribution script
            mock_rsync.assert_called_once()
        assert result == []

    def test_push_mode_head_rsync_fails(self, tmp_path, monkeypatch):
        """Push mode returns all hosts as failed if head rsync fails."""
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )

        rsync_results = [
            RemoteResult(host="10.0.0.1", returncode=1, stdout="", stderr="err"),
        ]
        with patch(
            "sparkrun.orchestration.ssh.run_rsync_parallel",
            return_value=rsync_results,
        ):
            result = distribute_tuning_to_hosts(
                "sglang", ["10.0.0.1", "10.0.0.2"],
                transfer_mode="push",
            )
        assert result == ["10.0.0.1", "10.0.0.2"]

    def test_push_mode_dist_script_fails(self, tmp_path, monkeypatch):
        """Push mode returns workers as failed if distribution script fails."""
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )

        rsync_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr=""),
        ]
        dist_result = RemoteResult(host="10.0.0.1", returncode=1, stdout="", stderr="rsync err")

        with patch(
            "sparkrun.orchestration.ssh.run_rsync_parallel",
            return_value=rsync_results,
        ), patch(
            "sparkrun.orchestration.ssh.run_remote_script",
            return_value=dist_result,
        ):
            result = distribute_tuning_to_hosts(
                "sglang", ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
                transfer_mode="push",
            )
        assert result == ["10.0.0.2", "10.0.0.3"]

    def test_delegated_mode_same_as_push(self, tmp_path, monkeypatch):
        """Delegated mode uses the same two-hop path as push mode."""
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )

        rsync_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr=""),
        ]
        dist_result = RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr="")

        with patch(
            "sparkrun.orchestration.ssh.run_rsync_parallel",
            return_value=rsync_results,
        ) as mock_rsync, patch(
            "sparkrun.orchestration.ssh.run_remote_script",
            return_value=dist_result,
        ) as mock_script:
            result = distribute_tuning_to_hosts(
                "sglang", ["10.0.0.1", "10.0.0.2"],
                transfer_mode="delegated",
            )
            mock_rsync.assert_called_once()
            mock_script.assert_called_once()
        assert result == []

    def test_local_mode_uses_direct_rsync(self, tmp_path, monkeypatch):
        """Local mode (default) uses direct rsync to all hosts."""
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )

        mock_results = [
            RemoteResult(host="10.0.0.1", returncode=0, stdout="", stderr=""),
            RemoteResult(host="10.0.0.2", returncode=0, stdout="", stderr=""),
        ]
        with patch(
            "sparkrun.orchestration.ssh.run_rsync_parallel",
            return_value=mock_results,
        ) as mock_rsync:
            result = distribute_tuning_to_hosts(
                "sglang", ["10.0.0.1", "10.0.0.2"],
                transfer_mode="local",
            )
            mock_rsync.assert_called_once()
            rsync_hosts = mock_rsync.call_args[0][1]
            assert rsync_hosts == ["10.0.0.1", "10.0.0.2"]
        assert result == []


class TestFiltersLocalhostFromHosts:
    """distribute_tuning_to_hosts filters localhost and only rsyncs to remotes."""

    def test_filters_localhost_from_hosts(self, tmp_path, monkeypatch):
        tuning_dir = tmp_path / "tuning" / "sglang"
        tuning_dir.mkdir(parents=True)
        (tuning_dir / "config.json").write_text("{}")
        monkeypatch.setattr(
            "sparkrun.tuning.distribute._get_local_tuning_dir",
            lambda runtime: tuning_dir,
        )

        mock_results = [
            RemoteResult(host="10.0.0.2", returncode=0, stdout="", stderr=""),
            RemoteResult(host="10.0.0.3", returncode=0, stdout="", stderr=""),
        ]
        with patch(
            "sparkrun.orchestration.ssh.run_rsync_parallel",
            return_value=mock_results,
        ) as mock_rsync:
            result = distribute_tuning_to_hosts(
                "sglang", ["localhost", "10.0.0.2", "127.0.0.1", "10.0.0.3"],
            )
            mock_rsync.assert_called_once()
            # Only remote hosts should be passed to rsync
            assert mock_rsync.call_args[0][1] == ["10.0.0.2", "10.0.0.3"]
        assert result == []
