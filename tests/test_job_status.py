"""Unit tests for check_job_running() in orchestration.job_metadata."""

from __future__ import annotations

from unittest import mock

import pytest

from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload
from sparkrun.orchestration.job_metadata import check_job_running


@pytest.fixture
def mock_recipe():
    """Create a mock recipe with attributes needed for derive_cluster_id."""
    r = mock.MagicMock()
    r.runtime = "sglang"
    r.model = "Qwen/Qwen3-1.7B"
    r.defaults = {"port": 30000}
    r.qualified_name = "test-recipe"
    return r


def _status_running(cluster_id: str, hosts: list[str], executor_name: str = "docker") -> ClusterStatus:
    """Build a ClusterStatus snapshot where *cluster_id* is running on every host."""
    workload = RunningWorkload(cluster_id=cluster_id)
    return ClusterStatus(
        hosts=tuple(HostOccupancy(host=h, workloads=(workload,), used_slots=1) for h in hosts),
        executor=executor_name,
    )


def _status_empty(hosts: list[str], executor_name: str = "docker") -> ClusterStatus:
    """Build a ClusterStatus snapshot with no workloads on any host."""
    return ClusterStatus(
        hosts=tuple(HostOccupancy(host=h) for h in hosts),
        executor=executor_name,
    )


class TestCheckJobRunning:
    """Tests for check_job_running()."""

    def test_check_running_solo_up(self):
        """Solo container running on head host."""
        cid = "sparkrun_aabbccdd0011"
        hosts = ["10.0.0.1"]
        with (
            mock.patch(
                "sparkrun.orchestration.executor.resolve_executor",
            ) as mock_resolve,
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value={"recipe": "test", "hosts": hosts},
            ),
        ):
            mock_resolve.return_value.query_status.return_value = _status_running(cid, hosts)
            status = check_job_running(cluster_id=cid, hosts=hosts)
        assert status.running is True
        assert status.cluster_id == cid
        assert status.container_statuses["sparkrun_aabbccdd0011_solo"] is True

    def test_check_running_solo_down(self):
        """Solo container not running."""
        cid = "sparkrun_aabbccdd0011"
        hosts = ["10.0.0.1"]
        with (
            mock.patch(
                "sparkrun.orchestration.executor.resolve_executor",
            ) as mock_resolve,
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value=None,
            ),
        ):
            mock_resolve.return_value.query_status.return_value = _status_empty(hosts)
            status = check_job_running(cluster_id=cid, hosts=hosts)
        assert status.running is False
        assert status.container_statuses["sparkrun_aabbccdd0011_solo"] is False

    def test_check_running_multinode_up(self):
        """Multi-node: cluster has a workload on the head host -> running."""
        cid = "sparkrun_aabbccdd0011"
        hosts = ["10.0.0.1", "10.0.0.2"]
        with (
            mock.patch(
                "sparkrun.orchestration.executor.resolve_executor",
            ) as mock_resolve,
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value=None,
            ),
        ):
            mock_resolve.return_value.query_status.return_value = _status_running(cid, hosts)
            status = check_job_running(cluster_id=cid, hosts=hosts)
        assert status.running is True
        # Candidate names for multi-node mode are both marked True when
        # the cluster has a workload on the head host (we can't recover
        # exact container names from a RunningWorkload).
        assert status.container_statuses["sparkrun_aabbccdd0011_node_0"] is True
        assert status.container_statuses["sparkrun_aabbccdd0011_head"] is True

    def test_check_running_multinode_down(self):
        """Multi-node: no workload on the head host -> not running."""
        cid = "sparkrun_aabbccdd0011"
        hosts = ["10.0.0.1", "10.0.0.2"]
        with (
            mock.patch(
                "sparkrun.orchestration.executor.resolve_executor",
            ) as mock_resolve,
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value=None,
            ),
        ):
            mock_resolve.return_value.query_status.return_value = _status_empty(hosts)
            status = check_job_running(cluster_id=cid, hosts=hosts)
        assert status.running is False
        assert status.container_statuses["sparkrun_aabbccdd0011_node_0"] is False
        assert status.container_statuses["sparkrun_aabbccdd0011_head"] is False

    def test_check_running_no_hosts_loads_metadata(self):
        """When hosts not provided, loads them from job metadata."""
        cid = "sparkrun_aabbccdd0011"
        hosts = ["10.0.0.1"]
        with (
            mock.patch(
                "sparkrun.orchestration.executor.resolve_executor",
            ) as mock_resolve,
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value={"recipe": "test", "hosts": hosts},
            ),
        ):
            mock_resolve.return_value.query_status.return_value = _status_running(cid, hosts)
            status = check_job_running(cluster_id=cid)
        assert status.running is True
        assert status.hosts == hosts

    def test_check_running_no_hosts_no_metadata(self):
        """No hosts provided and no metadata — returns not running."""
        with mock.patch(
            "sparkrun.orchestration.job_metadata.load_job_metadata",
            return_value=None,
        ):
            status = check_job_running(cluster_id="sparkrun_aabbccdd0011")
        assert status.running is False
        assert status.hosts == []

    def test_check_running_by_recipe(self, mock_recipe):
        """Derive cluster_id from recipe + hosts."""
        from sparkrun.orchestration.job_metadata import derive_cluster_id

        hosts = ["10.0.0.1", "10.0.0.2"]
        expected_cid = derive_cluster_id(mock_recipe, hosts)

        with (
            mock.patch(
                "sparkrun.orchestration.executor.resolve_executor",
            ) as mock_resolve,
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value=None,
            ),
        ):
            mock_resolve.return_value.query_status.return_value = _status_running(expected_cid, hosts)
            status = check_job_running(recipe=mock_recipe, hosts=hosts)
        assert status.cluster_id == expected_cid
        assert status.running is True

    def test_check_running_health_check_healthy(self):
        """Container running + health check passes."""
        cid = "sparkrun_aabbccdd0011"
        hosts = ["10.0.0.1"]
        with (
            mock.patch(
                "sparkrun.orchestration.executor.resolve_executor",
            ) as mock_resolve,
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value={"port": 9000},
            ),
            mock.patch(
                "sparkrun.orchestration.primitives.wait_for_healthy",
                return_value=True,
            ) as mock_health,
        ):
            mock_resolve.return_value.query_status.return_value = _status_running(cid, hosts)
            status = check_job_running(
                cluster_id=cid,
                hosts=hosts,
                check_http_models=True,
            )
        assert status.running is True
        assert status.healthy is True
        mock_health.assert_called_once()
        # Should use port from metadata
        call_args = mock_health.call_args
        assert "9000" in call_args[0][0]

    def test_check_running_health_check_unhealthy(self):
        """Container running but health check fails."""
        cid = "sparkrun_aabbccdd0011"
        hosts = ["10.0.0.1"]
        with (
            mock.patch(
                "sparkrun.orchestration.executor.resolve_executor",
            ) as mock_resolve,
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value=None,
            ),
            mock.patch(
                "sparkrun.orchestration.primitives.wait_for_healthy",
                return_value=False,
            ),
        ):
            mock_resolve.return_value.query_status.return_value = _status_running(cid, hosts)
            status = check_job_running(
                cluster_id=cid,
                hosts=hosts,
                check_http_models=True,
            )
        assert status.running is True
        assert status.healthy is False

    def test_check_running_health_check_not_running(self):
        """Container down — health check not attempted, healthy stays None."""
        cid = "sparkrun_aabbccdd0011"
        hosts = ["10.0.0.1"]
        with (
            mock.patch(
                "sparkrun.orchestration.executor.resolve_executor",
            ) as mock_resolve,
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value=None,
            ),
        ):
            mock_resolve.return_value.query_status.return_value = _status_empty(hosts)
            status = check_job_running(
                cluster_id=cid,
                hosts=hosts,
                check_http_models=True,
            )
        assert status.running is False
        assert status.healthy is None

    def test_check_running_explicit_port_overrides_metadata(self):
        """Explicit port param takes priority over metadata port."""
        cid = "sparkrun_aabbccdd0011"
        hosts = ["10.0.0.1"]
        with (
            mock.patch(
                "sparkrun.orchestration.executor.resolve_executor",
            ) as mock_resolve,
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value={"port": 9000},
            ),
            mock.patch(
                "sparkrun.orchestration.primitives.wait_for_healthy",
                return_value=True,
            ) as mock_health,
        ):
            mock_resolve.return_value.query_status.return_value = _status_running(cid, hosts)
            check_job_running(
                cluster_id=cid,
                hosts=hosts,
                check_http_models=True,
                port=7777,
            )
        call_url = mock_health.call_args[0][0]
        assert "7777" in call_url

    def test_requires_cluster_id_or_recipe_and_hosts(self):
        """Raises ValueError if neither cluster_id nor recipe+hosts given."""
        with pytest.raises(ValueError, match="cluster_id or both recipe and hosts"):
            check_job_running()

    def test_check_running_uses_metadata_executor(self):
        """When metadata records an executor, the override flows through resolve_executor."""
        cid = "sparkrun_aabbccdd0011"
        hosts = ["10.0.0.1"]
        with (
            mock.patch(
                "sparkrun.orchestration.executor.resolve_executor",
            ) as mock_resolve,
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value={
                    "hosts": hosts,
                    "executor": "local",
                    "executor_config": {"working_dir": "/tmp/x"},
                },
            ),
        ):
            mock_resolve.return_value.query_status.return_value = _status_empty(hosts)
            check_job_running(cluster_id=cid)
        kwargs = mock_resolve.call_args.kwargs
        assert kwargs["cli_overrides"] == {"executor": "local", "working_dir": "/tmp/x"}
