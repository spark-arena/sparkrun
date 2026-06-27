"""Tests for native-cluster distributed-init network selection."""

from unittest import mock

from sparkrun.orchestration.comm_env import ClusterCommEnv
from sparkrun.runtimes import _cluster_ops, _init_network


def _make_ctx(hosts: list[str], *, dry_run: bool = False) -> _cluster_ops.ClusterContext:
    return _cluster_ops.ClusterContext(
        hosts=list(hosts),
        head_host=hosts[0],
        worker_hosts=list(hosts[1:]),
        num_nodes=len(hosts),
        ssh_kwargs={},
        volumes={},
        all_env={},
        cluster_id="test-cluster",
        image="test:image",
        dry_run=dry_run,
        config=None,
    )


class TestSelectInitNetwork:
    """Verify native init address selection keeps mgmt first, then falls back."""

    def test_management_path_stays_selected_when_reachable(self, monkeypatch):
        """Given reachable mgmt, when IB candidates exist, then mgmt remains selected."""
        ctx = _make_ctx(["node-1", "node-2"])
        candidates = _init_network.InitNetworkCandidates(
            management_head_ip="192.168.128.10",
            management_hosts=("192.168.128.10", "192.168.96.114"),
            ib_ip_map={"node-1": "192.168.100.10", "node-2": "192.168.100.11"},
        )
        monkeypatch.setattr(_init_network, "workers_can_reach", lambda *_args, **_kwargs: True)

        selection = _init_network.select_init_network(ctx, candidates)

        assert selection.network == "management"
        assert selection.head_ip == "192.168.128.10"
        assert selection.hosts == ("192.168.128.10", "192.168.96.114")

    def test_reachable_ib_path_selected_when_management_fails(self, monkeypatch):
        """Given unreachable mgmt and reachable IB, then CX7 addresses are selected."""
        ctx = _make_ctx(["node-1", "node-2"])
        candidates = _init_network.InitNetworkCandidates(
            management_head_ip="192.168.128.10",
            management_hosts=("192.168.128.10", "192.168.96.114"),
            ib_ip_map={"node-1": "192.168.100.10", "node-2": "192.168.100.11"},
        )
        reachable = {
            "192.168.128.10": False,
            "192.168.100.10": True,
        }
        monkeypatch.setattr(
            _init_network,
            "workers_can_reach",
            lambda _ctx, target_ip: reachable[target_ip],
        )

        selection = _init_network.select_init_network(ctx, candidates)

        assert selection.network == "ib"
        assert selection.head_ip == "192.168.100.10"
        assert selection.hosts == ("192.168.100.10", "192.168.100.11")

    def test_management_path_kept_when_ib_map_is_incomplete(self, monkeypatch):
        """Given unreachable mgmt but partial IB data, then unsafe substitution is skipped."""
        ctx = _make_ctx(["node-1", "node-2"])
        candidates = _init_network.InitNetworkCandidates(
            management_head_ip="192.168.128.10",
            management_hosts=("192.168.128.10", "192.168.96.114"),
            ib_ip_map={"node-1": "192.168.100.10"},
        )
        monkeypatch.setattr(_init_network, "workers_can_reach", lambda *_args, **_kwargs: False)

        selection = _init_network.select_init_network(ctx, candidates)

        assert selection.network == "management"
        assert selection.head_ip == "192.168.128.10"
        assert selection.hosts == ("192.168.128.10", "192.168.96.114")


def test_native_cluster_threads_reachable_ib_selection_into_node_commands(monkeypatch):
    """Given mgmt failure, when native cluster launches, then commands receive IB init hosts."""
    ctx = _make_ctx(["node-1", "node-2"])
    monkeypatch.setattr(_cluster_ops, "cleanup_ranked_containers", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        _cluster_ops,
        "detect_ib_with_ips",
        lambda *_args, **_kwargs: (
            ClusterCommEnv.empty(),
            {"node-1": "192.168.100.10", "node-2": "192.168.100.11"},
        ),
    )
    monkeypatch.setattr(_cluster_ops, "detect_head_ip", lambda _ctx: "192.168.128.10")
    monkeypatch.setattr(
        _cluster_ops,
        "resolve_hosts_for_init",
        lambda _ctx, _head_ip: ["192.168.128.10", "192.168.96.114"],
    )
    monkeypatch.setattr(_cluster_ops, "find_port", lambda _ctx, _host, port: port)
    monkeypatch.setattr(_cluster_ops, "launch_containers_parallel", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        _init_network,
        "workers_can_reach",
        lambda _ctx, target_ip: target_ip == "192.168.100.10",
    )

    runtime = mock.MagicMock()
    runtime._resolve_executor.return_value.node_container_name = lambda cid, rank: "%s_node_%d" % (cid, rank)
    runtime.generate_node_command = mock.MagicMock(return_value="serve")
    runtime.get_extra_docker_opts = lambda: []
    runtime._print_cluster_banner = mock.MagicMock()

    rc = _cluster_ops.run_native_cluster(runtime=runtime, ctx=ctx)

    assert rc == 1
    call = runtime.generate_node_command.call_args
    assert call.kwargs["head_ip"] == "192.168.100.10"
    assert call.kwargs["hosts"] == ["192.168.100.10", "192.168.100.11"]
