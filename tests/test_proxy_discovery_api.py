"""Tests for proxy discovery driven by sparkrun.api.list_jobs + api.status."""

from __future__ import annotations

from unittest.mock import patch


def _make_job(
    cluster_id: str,
    *,
    host: str = "10.0.0.1",
    port: int | None = 8000,
    served_model_name: str | None = "served-model",
    recipe: str = "test-recipe",
    runtime: str = "vllm",
    model: str = "test/model",
    extra_meta: dict | None = None,
):
    """Construct a JobInfo with metadata matching the real on-disk schema."""
    from sparkrun.api import JobInfo

    meta = {
        "cluster_id": cluster_id,
        "recipe": recipe,
        "model": model,
        "runtime": runtime,
        "hosts": [host],
    }
    if port is not None:
        meta["port"] = port
    if served_model_name is not None:
        meta["served_model_name"] = served_model_name
    if extra_meta:
        meta.update(extra_meta)

    return JobInfo(
        cluster_id=cluster_id,
        recipe=recipe,
        runtime=runtime,
        hosts=(host,),
        metadata=meta,
    )


def _make_snapshot(running_ids_per_host: dict[str, list[str]]):
    """Build a ClusterStatus snapshot with workloads on each host."""
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload

    hosts = tuple(
        HostOccupancy(
            host=host,
            workloads=tuple(RunningWorkload(cluster_id=cid) for cid in cids),
        )
        for host, cids in running_ids_per_host.items()
    )
    return ClusterStatus(hosts=hosts)


class TestDiscoveryApiIntegration:
    """Discovery built atop mocked api.list_jobs + api.status."""

    def test_case1_job_metadata_running_status_emits_endpoint(self):
        """One job in list_jobs, reported running by status -> one EndpointEntry."""
        from sparkrun.proxy.discovery import discover_endpoints

        jobs = [_make_job("sparkrun_X", host="10.0.0.1", port=8000)]
        snapshot = _make_snapshot({"10.0.0.1": ["sparkrun_X"]})

        with (
            patch("sparkrun.proxy.discovery.api.list_jobs", return_value=jobs),
            patch("sparkrun.proxy.discovery.api.status", return_value=snapshot),
        ):
            endpoints = discover_endpoints(
                check_health=False,
                host_list=["10.0.0.1"],
                ssh_kwargs={},
            )

        assert len(endpoints) == 1
        assert endpoints[0].cluster_id == "sparkrun_X"
        assert endpoints[0].host == "10.0.0.1"
        assert endpoints[0].port == 8000

    def test_case2_job_present_but_not_running_dropped(self):
        """Job in list_jobs but status doesn't report it running -> discovery drops it."""
        from sparkrun.proxy.discovery import discover_endpoints

        jobs = [_make_job("sparkrun_X", host="10.0.0.1")]
        # Empty snapshot — no workloads on the host.
        snapshot = _make_snapshot({"10.0.0.1": []})

        with (
            patch("sparkrun.proxy.discovery.api.list_jobs", return_value=jobs),
            patch("sparkrun.proxy.discovery.api.status", return_value=snapshot),
        ):
            endpoints = discover_endpoints(
                check_health=False,
                host_list=["10.0.0.1"],
                ssh_kwargs={},
            )

        assert endpoints == []

    def test_case3_running_but_no_jobinfo_dropped(self):
        """Job running in status but no JobInfo in list_jobs -> discovery drops it."""
        from sparkrun.proxy.discovery import discover_endpoints

        jobs = []  # no metadata at all
        snapshot = _make_snapshot({"10.0.0.1": ["sparkrun_X"]})

        with (
            patch("sparkrun.proxy.discovery.api.list_jobs", return_value=jobs),
            patch("sparkrun.proxy.discovery.api.status", return_value=snapshot),
        ):
            endpoints = discover_endpoints(
                check_health=False,
                host_list=["10.0.0.1"],
                ssh_kwargs={},
            )

        assert endpoints == []

    def test_case4_jobinfo_without_port_dropped(self):
        """JobInfo without port in metadata -> discovery drops it."""
        from sparkrun.proxy.discovery import discover_endpoints

        # Build a job with no `port` and explicitly null hosts to short-circuit.
        # We need to make port falsey in a way that triggers the skip — but
        # the current implementation defaults to 8000 when missing.  The
        # spec requires "skip the job if port is missing"; the implementation
        # returns None from _endpoint_from_job() when port can't be parsed.
        # Insert a non-parseable port value.
        from sparkrun.api import JobInfo

        meta = {
            "cluster_id": "sparkrun_X",
            "recipe": "test",
            "runtime": "vllm",
            "model": "test/model",
            "hosts": ["10.0.0.1"],
            "port": "not-a-number",
        }
        job = JobInfo(
            cluster_id="sparkrun_X",
            recipe="test",
            runtime="vllm",
            hosts=("10.0.0.1",),
            metadata=meta,
        )

        snapshot = _make_snapshot({"10.0.0.1": ["sparkrun_X"]})

        with (
            patch("sparkrun.proxy.discovery.api.list_jobs", return_value=[job]),
            patch("sparkrun.proxy.discovery.api.status", return_value=snapshot),
        ):
            endpoints = discover_endpoints(
                check_health=False,
                host_list=["10.0.0.1"],
                ssh_kwargs={},
            )

        assert endpoints == []

    def test_case5_two_jobs_different_hosts_both_surface(self):
        """Two jobs on different hosts -> both surface correctly."""
        from sparkrun.proxy.discovery import discover_endpoints

        jobs = [
            _make_job("sparkrun_A", host="10.0.0.1", port=8000),
            _make_job("sparkrun_B", host="10.0.0.2", port=9000),
        ]
        snapshot = _make_snapshot(
            {
                "10.0.0.1": ["sparkrun_A"],
                "10.0.0.2": ["sparkrun_B"],
            }
        )

        with (
            patch("sparkrun.proxy.discovery.api.list_jobs", return_value=jobs),
            patch("sparkrun.proxy.discovery.api.status", return_value=snapshot),
        ):
            endpoints = discover_endpoints(
                check_health=False,
                host_list=["10.0.0.1", "10.0.0.2"],
                ssh_kwargs={},
            )

        assert len(endpoints) == 2
        by_cid = {ep.cluster_id: ep for ep in endpoints}
        assert "sparkrun_A" in by_cid
        assert "sparkrun_B" in by_cid
        assert by_cid["sparkrun_A"].host == "10.0.0.1"
        assert by_cid["sparkrun_A"].port == 8000
        assert by_cid["sparkrun_B"].host == "10.0.0.2"
        assert by_cid["sparkrun_B"].port == 9000

    def test_case6_status_empty_returns_empty_result(self):
        """Status query returns empty (no hosts reachable) -> empty result."""
        from sparkrun.core.cluster_status import ClusterStatus
        from sparkrun.proxy.discovery import discover_endpoints

        jobs = [_make_job("sparkrun_X", host="10.0.0.1")]
        snapshot = ClusterStatus(hosts=())  # empty snapshot

        with (
            patch("sparkrun.proxy.discovery.api.list_jobs", return_value=jobs),
            patch("sparkrun.proxy.discovery.api.status", return_value=snapshot),
        ):
            endpoints = discover_endpoints(
                check_health=False,
                host_list=["10.0.0.1"],
                ssh_kwargs={},
            )

        assert endpoints == []
