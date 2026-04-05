"""Unit tests for check_job_running() in orchestration.job_metadata."""

from __future__ import annotations

from unittest import mock

import pytest

from sparkrun.orchestration.job_metadata import check_job_running


@pytest.fixture
def mock_recipe():
    """Create a mock recipe with attributes needed for generate_cluster_id."""
    r = mock.MagicMock()
    r.runtime = "sglang"
    r.model = "Qwen/Qwen3-1.7B"
    r.defaults = {"port": 30000}
    r.qualified_name = "test-recipe"
    return r


class TestCheckJobRunning:
    """Tests for check_job_running()."""

    def test_check_running_solo_up(self):
        """Solo container running on head host."""
        with (
            mock.patch(
                "sparkrun.orchestration.primitives.is_container_running",
                side_effect=lambda host, name, **kw: name.endswith("_solo"),
            ),
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value={"recipe": "test", "hosts": ["10.0.0.1"]},
            ),
        ):
            status = check_job_running(
                cluster_id="sparkrun_aabbccdd0011",
                hosts=["10.0.0.1"],
            )
        assert status.running is True
        assert status.cluster_id == "sparkrun_aabbccdd0011"
        assert status.container_statuses["sparkrun_aabbccdd0011_solo"] is True

    def test_check_running_solo_down(self):
        """Solo container not running."""
        with (
            mock.patch(
                "sparkrun.orchestration.primitives.is_container_running",
                return_value=False,
            ),
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value=None,
            ),
        ):
            status = check_job_running(
                cluster_id="sparkrun_aabbccdd0011",
                hosts=["10.0.0.1"],
            )
        assert status.running is False

    def test_check_running_multinode_native(self):
        """Multi-node native: _node_0 running on head."""

        def _is_running(host, name, **kw):
            return name.endswith("_node_0")

        with (
            mock.patch(
                "sparkrun.orchestration.primitives.is_container_running",
                side_effect=_is_running,
            ),
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value=None,
            ),
        ):
            status = check_job_running(
                cluster_id="sparkrun_aabbccdd0011",
                hosts=["10.0.0.1", "10.0.0.2"],
            )
        assert status.running is True
        assert status.container_statuses["sparkrun_aabbccdd0011_node_0"] is True
        assert status.container_statuses["sparkrun_aabbccdd0011_head"] is False

    def test_check_running_multinode_ray(self):
        """Multi-node Ray: _head running (not _node_0)."""

        def _is_running(host, name, **kw):
            return name.endswith("_head")

        with (
            mock.patch(
                "sparkrun.orchestration.primitives.is_container_running",
                side_effect=_is_running,
            ),
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value=None,
            ),
        ):
            status = check_job_running(
                cluster_id="sparkrun_aabbccdd0011",
                hosts=["10.0.0.1", "10.0.0.2"],
            )
        assert status.running is True
        assert status.container_statuses["sparkrun_aabbccdd0011_head"] is True

    def test_check_running_no_hosts_loads_metadata(self):
        """When hosts not provided, loads them from job metadata."""
        with (
            mock.patch(
                "sparkrun.orchestration.primitives.is_container_running",
                return_value=True,
            ),
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value={"recipe": "test", "hosts": ["10.0.0.1"]},
            ),
        ):
            status = check_job_running(cluster_id="sparkrun_aabbccdd0011")
        assert status.running is True
        assert status.hosts == ["10.0.0.1"]

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
        """Generate cluster_id from recipe + hosts."""
        from sparkrun.orchestration.job_metadata import generate_cluster_id

        hosts = ["10.0.0.1", "10.0.0.2"]
        expected_cid = generate_cluster_id(mock_recipe, hosts)

        with (
            mock.patch(
                "sparkrun.orchestration.primitives.is_container_running",
                return_value=True,
            ),
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value=None,
            ),
        ):
            status = check_job_running(recipe=mock_recipe, hosts=hosts)
        assert status.cluster_id == expected_cid
        assert status.running is True

    def test_check_running_health_check_healthy(self):
        """Container running + health check passes."""
        with (
            mock.patch(
                "sparkrun.orchestration.primitives.is_container_running",
                return_value=True,
            ),
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value={"port": 9000},
            ),
            mock.patch(
                "sparkrun.orchestration.primitives.wait_for_healthy",
                return_value=True,
            ) as mock_health,
        ):
            status = check_job_running(
                cluster_id="sparkrun_aabbccdd0011",
                hosts=["10.0.0.1"],
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
        with (
            mock.patch(
                "sparkrun.orchestration.primitives.is_container_running",
                return_value=True,
            ),
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value=None,
            ),
            mock.patch(
                "sparkrun.orchestration.primitives.wait_for_healthy",
                return_value=False,
            ),
        ):
            status = check_job_running(
                cluster_id="sparkrun_aabbccdd0011",
                hosts=["10.0.0.1"],
                check_http_models=True,
            )
        assert status.running is True
        assert status.healthy is False

    def test_check_running_health_check_not_running(self):
        """Container down — health check not attempted, healthy stays None."""
        with (
            mock.patch(
                "sparkrun.orchestration.primitives.is_container_running",
                return_value=False,
            ),
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value=None,
            ),
        ):
            status = check_job_running(
                cluster_id="sparkrun_aabbccdd0011",
                hosts=["10.0.0.1"],
                check_http_models=True,
            )
        assert status.running is False
        assert status.healthy is None

    def test_check_running_explicit_port_overrides_metadata(self):
        """Explicit port param takes priority over metadata port."""
        with (
            mock.patch(
                "sparkrun.orchestration.primitives.is_container_running",
                return_value=True,
            ),
            mock.patch(
                "sparkrun.orchestration.job_metadata.load_job_metadata",
                return_value={"port": 9000},
            ),
            mock.patch(
                "sparkrun.orchestration.primitives.wait_for_healthy",
                return_value=True,
            ) as mock_health,
        ):
            check_job_running(
                cluster_id="sparkrun_aabbccdd0011",
                hosts=["10.0.0.1"],
                check_http_models=True,
                port=7777,
            )
        call_url = mock_health.call_args[0][0]
        assert "7777" in call_url

    def test_requires_cluster_id_or_recipe_and_hosts(self):
        """Raises ValueError if neither cluster_id nor recipe+hosts given."""
        with pytest.raises(ValueError, match="cluster_id or both recipe and hosts"):
            check_job_running()
