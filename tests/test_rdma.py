"""Tests for RDMA fabric verification (``sparkrun setup rdma-test``).

The perftest parser tests run against output captured verbatim from real
hardware (the two-Spark procedure in the eugr NETWORKING.md), because the
whole value of the parsers is that they survive perftest's actual banner.
"""

from __future__ import annotations

from unittest import mock

import pytest
from click.testing import CliRunner

from sparkrun.api.setup import RdmaTestError
from sparkrun.api.setup._rdma import (
    CONTAINER_NAME,
    STATUS_FAIL,
    STATUS_OK,
    STATUS_SKIP,
    STATUS_WARN,
    rdma_test,
)
from sparkrun.cli import main
from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.orchestration.mpi import build_rsh_wrapper
from sparkrun.orchestration.networking import CX7HostDetection, CX7Interface
from sparkrun.orchestration.rdma import (
    RdmaLink,
    RdmaPair,
    build_nccl_test_cmd,
    build_perftest_cmd,
    derive_link_pairs,
    parse_device_facts,
    parse_framed_runs,
    parse_nccl_busbw,
    parse_perftest_bw,
    parse_perftest_lat,
    rate_to_gbps,
)
from sparkrun.orchestration.ssh import RemoteResult

# --- Captured from real hardware (eugr spark-vllm-docker NETWORKING.md) -----

REAL_BW_OUTPUT = """\
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF          Device         : rocep1s0f1
 Number of qps   : 4            Transport type : IB
 Connection type : RC           Using SRQ      : OFF
 TX depth        : 128
 Mtu             : 1024[B]
 Link type       : IB
 rdma_cm QPs     : ON
---------------------------------------------------------------------------------------
 local address: LID 0000 QPN 0x03ec PSN 0xb680ae
 remote address: LID 0000 QPN 0x03eb PSN 0x75f6ee
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 65536      20000            111.72             111.71             0.213070
---------------------------------------------------------------------------------------
"""

REAL_LAT_OUTPUT = """\
---------------------------------------------------------------------------------------
                    RDMA_Write Latency Test
 Dual-port       : OFF          Device         : rocep1s0f1
 Number of qps   : 1            Transport type : IB
 Max inline data : 220[B]
 rdma_cm QPs     : ON
---------------------------------------------------------------------------------------
 local address: LID 0000 QPN 0x02ee PSN 0xb0c21c
 remote address: LID 0000 QPN 0x02ee PSN 0x14568b
---------------------------------------------------------------------------------------
 #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    t_avg[usec] \
   t_stdev[usec]   99% percentile[usec]   99.9% percentile[usec]
 2       1000          1.42           1.93         1.47                1.47   \
          0.00            1.57                    1.93
---------------------------------------------------------------------------------------
"""

REAL_NCCL_OUTPUT = """\
#                                                              out-of-place       in-place
#       size         count      type   redop    root     time   algbw   busbw
  17179869184    4294967296     float    none      -1  730123   23.53   23.53
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 23.5312
#
"""


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_parse_perftest_bw_real_output():
    """Real ib_write_bw output yields the documented 111.71 Gb/s average."""
    s = parse_perftest_bw(REAL_BW_OUTPUT)
    assert s is not None
    assert s.bytes_ == 65536
    assert s.iterations == 20000
    assert s.peak_gbps == pytest.approx(111.72)
    assert s.avg_gbps == pytest.approx(111.71)


def test_parse_perftest_lat_real_output():
    """t_typical is the headline figure, and p99 survives the wide header."""
    s = parse_perftest_lat(REAL_LAT_OUTPUT)
    assert s is not None
    assert s.t_min == pytest.approx(1.42)
    assert s.t_max == pytest.approx(1.93)
    assert s.t_typical == pytest.approx(1.47)
    assert s.p99 == pytest.approx(1.57)


def test_parse_perftest_lat_older_build_without_percentiles():
    """A short row still yields t_typical; p99 degrades to None, not to 0."""
    out = " #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]\n 2  1000  1.42  1.93  1.47\n"
    s = parse_perftest_lat(out)
    assert s is not None
    assert s.t_typical == pytest.approx(1.47)
    assert s.p99 is None


def test_parse_nccl_busbw_real_output():
    s = parse_nccl_busbw(REAL_NCCL_OUTPUT)
    assert s is not None
    assert s.avg_bus_bw == pytest.approx(23.5312)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "Unable to connect to remote host\n",
        "Couldn't connect to 192.168.11.14\n",
        "---------------\n only a banner \n---------------\n",
        " #bytes #iterations BW peak\n",  # header, no data row
    ],
)
def test_parsers_return_none_rather_than_fabricating(bad):
    """A failed run must never be reported as a measurement."""
    assert parse_perftest_bw(bad) is None
    assert parse_perftest_lat(bad) is None
    assert parse_nccl_busbw(bad) is None


def test_parse_bw_ignores_commentary_and_rules():
    """The data row is found by shape, not by line offset."""
    noisy = "# a comment with 1 2 3 4 5\n" + REAL_BW_OUTPUT
    assert parse_perftest_bw(noisy).avg_gbps == pytest.approx(111.71)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("100 Gb/sec (4X EDR)", 100.0),
        ("200 Gb/sec (2X NDR)", 200.0),
        ("25 Gb/sec (1X DDR)", 25.0),
        ("", None),
        ("unknown", None),
    ],
)
def test_rate_to_gbps(raw, expected):
    """The bandwidth expectation is derived from the hardware, not hardcoded."""
    assert rate_to_gbps(raw) == expected


def test_parse_framed_runs_separates_concurrent_output():
    stdout = (
        "SPARKRUN_RDMA_RUN_0_BEGIN\nalpha 1 2\nSPARKRUN_RDMA_RUN_0_END\nSPARKRUN_RDMA_RUN_0_RC=0\n"
        "SPARKRUN_RDMA_RUN_1_BEGIN\nbeta\nSPARKRUN_RDMA_RUN_1_END\nSPARKRUN_RDMA_RUN_1_RC=7\n"
        "RDMA_RUN_COUNT=2\nRDMA_COMPLETE=1\n"
    )
    runs = parse_framed_runs(stdout)
    assert runs == {0: ("alpha 1 2", 0), 1: ("beta", 7)}


def test_parse_framed_runs_missing_rc_is_not_success():
    """A truncated stream must not read as rc=0."""
    runs = parse_framed_runs("SPARKRUN_RDMA_RUN_0_BEGIN\nx\nSPARKRUN_RDMA_RUN_0_END\n")
    assert runs[0][1] == -1


def test_parse_device_facts():
    kv = {
        "RDMA_COMPLETE": "1",
        "RDMA_PERFTEST": "1",
        "RDMA_DOCKER": "0",
        "RDMA_DEV_COUNT": "2",
        "RDMA_DEV_0_NAME": "rocep1s0f1",
        "RDMA_DEV_0_STATE": "4: ACTIVE",
        "RDMA_DEV_0_RATE": "100 Gb/sec (4X EDR)",
        "RDMA_DEV_1_NAME": "rocep1s0f0",
        "RDMA_DEV_1_STATE": "1: DOWN",
        "RDMA_DEV_1_RATE": "100 Gb/sec (4X EDR)",
    }
    facts = parse_device_facts("h1", kv)
    assert facts.complete and facts.has_perftest and not facts.has_docker
    assert facts.device("rocep1s0f1").active
    assert not facts.device("rocep1s0f0").active
    assert facts.device("rocep1s0f1").rate_gbps == 100.0
    assert facts.device("nope") is None


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def test_build_perftest_cmd_matches_documented_form():
    """Client form reproduces the published procedure's flags."""
    cmd = build_perftest_cmd("ib_write_bw", "rocep1s0f1", peer_ip="192.168.11.14", queue_pairs=4, duration=10)
    assert cmd.startswith("ib_write_bw 192.168.11.14 -d rocep1s0f1")
    assert "--report_gbits" in cmd
    assert "-R" in cmd
    assert "--force-link IB" in cmd
    assert "-q 4" in cmd
    assert "-D 10" in cmd


def test_build_perftest_cmd_server_form_has_no_peer():
    cmd = build_perftest_cmd("ib_write_lat", "rocep1s0f1")
    assert cmd.startswith("ib_write_lat -d rocep1s0f1")
    assert "192." not in cmd


def test_build_perftest_cmd_link_type_and_gid_are_optional():
    cmd = build_perftest_cmd("ib_write_bw", "mlx5_0", link_type=None)
    assert "--force-link" not in cmd
    assert "-x " not in cmd
    assert "-x 3" in build_perftest_cmd("ib_write_bw", "mlx5_0", gid_index=3)


def test_build_nccl_test_cmd():
    cmd = build_nccl_test_cmd(
        ["10.0.0.1", "10.0.0.2"],
        rsh_agent="/tmp/agent.sh",
        msg_size="16G",
        env={"NCCL_IB_HCA": "rocep1s0f1"},
    )
    assert "--mca plm_rsh_agent /tmp/agent.sh" in cmd
    assert "-H 10.0.0.1:1,10.0.0.2:1" in cmd
    assert "-x NCCL_IB_HCA" in cmd
    assert "all_gather_perf -b 16G -e 16G" in cmd


# ---------------------------------------------------------------------------
# Shared mpirun rsh agent
# ---------------------------------------------------------------------------


def test_build_rsh_wrapper_maps_hosts_to_containers():
    w = build_rsh_wrapper({"10.0.0.1": "c0", "10.0.0.2": "c1"})
    assert '10.0.0.1) CONTAINER="c0"' in w
    assert '10.0.0.2) CONTAINER="c1"' in w
    assert 'docker exec "$CONTAINER" "$@"' in w


def test_trtllm_still_uses_the_shared_wrapper():
    """The extraction must not change trtllm's generated agent."""
    from sparkrun.runtimes.trtllm import TrtllmRuntime

    assert TrtllmRuntime._generate_rsh_wrapper({"10.0.0.1": "c0"}, "cid") == build_rsh_wrapper({"10.0.0.1": "c0"})


@pytest.mark.parametrize(
    "host_map",
    [
        {"10.0.0.1; rm -rf /": "c0"},
        {"10.0.0.1": 'c0" ; rm -rf / ; "'},
        {"$(whoami)": "c0"},
    ],
)
def test_build_rsh_wrapper_refuses_uninterpolatable_values(host_map):
    """Values are emitted bare or double-quoted, so they cannot be shell-quoted."""
    with pytest.raises(ValueError):
        build_rsh_wrapper(host_map)


def test_build_rsh_wrapper_refuses_unsafe_key_path():
    with pytest.raises(ValueError):
        build_rsh_wrapper({"10.0.0.1": "c0"}, ssh_key_path="/tmp/$(id).key")


# ---------------------------------------------------------------------------
# Topology derivation
# ---------------------------------------------------------------------------


def _iface(name, ip, subnet, hca):
    return CX7Interface(name=name, ip=ip, prefix=24, subnet=subnet, mtu=9000, state="UP", hca=hca)


def _det(host, ifaces, mgmt_iface="enp0s1"):
    return CX7HostDetection(host=host, interfaces=list(ifaces), detected=True, mgmt_iface=mgmt_iface)


def _two_spark_direct():
    """The common case: one cable, two RDMA devices, two /24s."""
    return {
        "h1": _det(
            "h1",
            [
                _iface("enp1s0f1np1", "192.168.177.11", "192.168.177.0/24", "rocep1s0f1"),
                _iface("enP2p1s0f1np1", "192.168.178.11", "192.168.178.0/24", "roceP2p1s0f1"),
            ],
        ),
        "h2": _det(
            "h2",
            [
                _iface("enp1s0f1np1", "192.168.177.12", "192.168.177.0/24", "rocep1s0f1"),
                _iface("enP2p1s0f1np1", "192.168.178.12", "192.168.178.0/24", "roceP2p1s0f1"),
            ],
        ),
    }


def test_derive_pairs_two_host_direct_yields_one_pair_two_links():
    """A QSFP cable is one pair carrying both of its RDMA devices."""
    pairs = derive_link_pairs(_two_spark_direct(), ["h1", "h2"])
    assert len(pairs) == 1
    assert pairs[0].host_a == "h1" and pairs[0].host_b == "h2"
    assert len(pairs[0].links) == 2
    assert {link.hca_a for link in pairs[0].links} == {"rocep1s0f1", "roceP2p1s0f1"}
    for link in pairs[0].links:
        assert link.usable
        # Each end names its OWN device and the PEER's address.
        assert link.ip_b.endswith(".12")


def test_derive_pairs_ring_yields_neighbour_links_only():
    """A 3-host ring gets its three cables, not three-choose-two host pairs."""
    dets = {
        "h1": _det("h1", [_iface("e0", "10.1.0.1", "10.1.0.0/24", "hca0"), _iface("e1", "10.3.0.1", "10.3.0.0/24", "hca1")]),
        "h2": _det("h2", [_iface("e0", "10.1.0.2", "10.1.0.0/24", "hca0"), _iface("e1", "10.2.0.2", "10.2.0.0/24", "hca1")]),
        "h3": _det("h3", [_iface("e0", "10.2.0.3", "10.2.0.0/24", "hca0"), _iface("e1", "10.3.0.3", "10.3.0.0/24", "hca1")]),
    }
    pairs = derive_link_pairs(dets, ["h1", "h2", "h3"])
    assert {p.key for p in pairs} == {("h1", "h2"), ("h2", "h3"), ("h1", "h3")}
    assert all(len(p.links) == 1 for p in pairs)


def test_derive_pairs_switched_segment_chains_rather_than_meshes():
    """4 hosts on one subnet is O(N) chained links, not 6 meshed ones."""
    dets = {h: _det(h, [_iface("e0", "10.0.0.%d" % (i + 1), "10.0.0.0/24", "hca0")]) for i, h in enumerate(["h1", "h2", "h3", "h4"])}
    pairs = derive_link_pairs(dets, ["h1", "h2", "h3", "h4"])
    assert {p.key for p in pairs} == {("h1", "h2"), ("h2", "h3"), ("h3", "h4")}


def test_derive_pairs_ignores_unconfigured_and_undetected_hosts():
    dets = {
        "h1": _det("h1", [_iface("e0", "", "", "hca0")]),  # no address yet
        "h2": CX7HostDetection(host="h2", detected=False),
        "h3": _det("h3", [_iface("e0", "10.0.0.3", "10.0.0.0/24", "hca0")]),  # alone on its subnet
    }
    assert derive_link_pairs(dets, ["h1", "h2", "h3"]) == []


def test_derive_pairs_respects_caller_host_order():
    """Pair ordering follows the host list so output is stable across runs."""
    dets = _two_spark_direct()
    assert derive_link_pairs(dets, ["h2", "h1"])[0].key == ("h2", "h1")


def test_derive_pairs_only_considers_requested_hosts():
    dets = _two_spark_direct()
    dets["h3"] = _det("h3", [_iface("e0", "192.168.177.13", "192.168.177.0/24", "rocep1s0f1")])
    pairs = derive_link_pairs(dets, ["h1", "h2"])
    assert all("h3" not in p.key for p in pairs)


# ---------------------------------------------------------------------------
# api.setup.rdma_test
# ---------------------------------------------------------------------------


class _Sctx:
    def __init__(self, config=None):
        self.config = config or _Config()


class _Config:
    """Minimal config stub.

    Implements the feature-resolution surface because ``rdma_test`` gates the
    collective on ``cli.setup.rdma_test.nccl``. Defaults to *enabled*: the
    subject of these tests is what each suite does, not whether it is
    permitted — the gate has its own tests below.
    """

    def __init__(self, nccl: bool = True):
        self._nccl = nccl

    def rdma_test_settings(self):
        return {}

    def feature_override(self, name):
        return self._nccl if name == "cli.setup.rdma_test.nccl" else None

    @property
    def feature_channel(self):
        return "stable"


def _facts_kv(perftest="1", docker="1"):
    return (
        "RDMA_COMPLETE=1\nRDMA_PERFTEST=%s\nRDMA_DOCKER=%s\n"
        "RDMA_DEV_COUNT=2\n"
        "RDMA_DEV_0_NAME=rocep1s0f1\nRDMA_DEV_0_STATE=4: ACTIVE\nRDMA_DEV_0_RATE=100 Gb/sec (4X EDR)\n"
        "RDMA_DEV_1_NAME=roceP2p1s0f1\nRDMA_DEV_1_STATE=4: ACTIVE\nRDMA_DEV_1_RATE=100 Gb/sec (4X EDR)\n" % (perftest, docker)
    )


def _patch_probe(perftest="1"):
    return mock.patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        side_effect=lambda hosts, script, **kw: [RemoteResult(h, 0, _facts_kv(perftest), "") for h in hosts],
    )


def test_rdma_test_dry_run_runs_nothing():
    """A dry run resolves and plans, but starts no container and sends no traffic."""
    with (
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_two_spark_direct()),
        mock.patch("sparkrun.containers.sync.sync_image_to_hosts") as sync,
        mock.patch("sparkrun.orchestration.ssh.run_remote_script") as run_one,
    ):
        # suite is explicit: the default is `perftest`, which plans no
        # collective, and this asserts the dry run covers both halves.
        report = rdma_test(_Sctx(), ["h1", "h2"], {}, suite="all", dry_run=True)

    assert report.dry_run
    assert len(report.pairs) == 1
    assert report.pairs[0].status == STATUS_SKIP
    assert report.nccl is not None and report.nccl.status == STATUS_SKIP
    sync.assert_not_called()
    run_one.assert_not_called()


def test_dry_run_claims_nothing_about_tooling_it_did_not_probe():
    """A dry run does not probe, so it must not report perftest as missing.

    The probe returns empty facts under --dry-run; reading those as "perftest
    absent" is a statement about hosts nobody looked at.
    """
    with (
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_two_spark_direct()),
        mock.patch("sparkrun.containers.sync.sync_image_to_hosts"),
    ):
        report = rdma_test(_Sctx(), ["h1", "h2"], {}, suite="perftest", dry_run=True)

    assert report.warnings == []
    assert report.used_container is False  # perftest suite requires no image


def test_dry_run_lists_the_links_it_would_test():
    """Showing the plan is the point of a dry run."""
    with mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_two_spark_direct()):
        report = rdma_test(_Sctx(), ["h1", "h2"], {}, dry_run=True)

    assert len(report.pairs[0].links) == 2
    assert {link.link.hca_a for link in report.pairs[0].links} == {"rocep1s0f1", "roceP2p1s0f1"}


def test_dry_run_reports_the_image_when_the_suite_requires_it():
    with mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_two_spark_direct()):
        assert rdma_test(_Sctx(), ["h1", "h2"], {}, suite="nccl", dry_run=True).used_container is True


def test_rdma_test_without_links_raises_pointing_at_setup_cx7():
    dets = {"h1": CX7HostDetection(host="h1", detected=False), "h2": CX7HostDetection(host="h2", detected=False)}
    with (
        _patch_probe(),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=dets),
    ):
        with pytest.raises(RdmaTestError, match="setup cx7"):
            rdma_test(_Sctx(), ["h1", "h2"], {})


def test_default_suite_is_perftest_and_needs_no_image():
    """The default answers "is the fabric working" without pulling anything.

    ``all`` used to be the default, so the common check dragged a ~1 GB image
    onto every host and ran a multi-minute collective before saying anything.
    """
    with (
        _patch_probe(perftest="1"),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_two_spark_direct()),
        mock.patch("sparkrun.containers.sync.sync_image_to_hosts") as sync,
        mock.patch("sparkrun.api.setup._rdma._run_nccl") as nccl,
        mock.patch("sparkrun.api.setup._rdma._run_pair", return_value=mock.Mock(status=STATUS_OK, links=(), detail="")),
    ):
        report = rdma_test(_Sctx(), ["h1", "h2"], {})

    assert report.suite == "perftest"
    assert report.used_container is False
    assert report.nccl is None
    sync.assert_not_called()
    nccl.assert_not_called()


def test_cli_default_suite_matches_the_api_default():
    """A CLI default that drifts from the api's is a silent behaviour split."""
    from sparkrun.api.setup._rdma import SUITE_PERFTEST
    from sparkrun.cli._setup._rdma import setup_rdma_test

    opt = next(p for p in setup_rdma_test.params if p.name == "suite")
    assert opt.default == SUITE_PERFTEST


def test_rdma_test_unknown_suite_raises():
    with pytest.raises(RdmaTestError, match="unknown suite"):
        rdma_test(_Sctx(), ["h1", "h2"], {}, suite="bogus")


def test_perftest_suite_skips_the_container_when_hosts_have_perftest():
    """DGX OS ships perftest, so the common check costs no image pull."""
    with (
        _patch_probe(perftest="1"),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_two_spark_direct()),
        mock.patch("sparkrun.containers.sync.sync_image_to_hosts") as sync,
        mock.patch("sparkrun.api.setup._rdma._run_pair", return_value=mock.Mock(status=STATUS_OK, links=(), detail="")),
    ):
        report = rdma_test(_Sctx(), ["h1", "h2"], {}, suite="perftest")

    assert report.used_container is False
    sync.assert_not_called()


def test_perftest_suite_falls_back_to_the_image_when_perftest_is_missing():
    with (
        _patch_probe(perftest="0"),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_two_spark_direct()),
        mock.patch("sparkrun.api.setup._rdma._prepare_container") as prep,
        mock.patch("sparkrun.api.setup._rdma._stop_containers") as stop,
        mock.patch("sparkrun.api.setup._rdma._run_pair", return_value=mock.Mock(status=STATUS_OK, links=(), detail="")),
    ):
        report = rdma_test(_Sctx(), ["h1", "h2"], {}, suite="perftest")

    assert report.used_container is True
    prep.assert_called_once()
    stop.assert_called_once()
    assert any("perftest not found" in w for w in report.warnings)


def test_nccl_suite_always_needs_the_container_even_with_host_perftest():
    """The collective lives only in the image."""
    with (
        _patch_probe(perftest="1"),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_two_spark_direct()),
        mock.patch("sparkrun.api.setup._rdma._prepare_container") as prep,
        mock.patch("sparkrun.api.setup._rdma._stop_containers"),
        mock.patch("sparkrun.api.setup._rdma._run_nccl", return_value=mock.Mock(status=STATUS_OK)),
    ):
        report = rdma_test(_Sctx(), ["h1", "h2"], {}, suite="nccl")

    assert report.used_container is True
    prep.assert_called_once()


def test_containers_are_torn_down_when_a_test_raises():
    """Teardown runs from a finally — an interrupted run must not leak containers."""
    with (
        _patch_probe(perftest="0"),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_two_spark_direct()),
        mock.patch("sparkrun.api.setup._rdma._prepare_container"),
        mock.patch("sparkrun.api.setup._rdma._stop_containers") as stop,
        mock.patch("sparkrun.api.setup._rdma._run_pair", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            rdma_test(_Sctx(), ["h1", "h2"], {}, suite="perftest")

    stop.assert_called_once()


def test_keep_containers_skips_teardown_and_says_so():
    with (
        _patch_probe(perftest="0"),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_two_spark_direct()),
        mock.patch("sparkrun.api.setup._rdma._prepare_container"),
        mock.patch("sparkrun.api.setup._rdma._stop_containers") as stop,
        mock.patch("sparkrun.api.setup._rdma._run_pair", return_value=mock.Mock(status=STATUS_OK, links=(), detail="")),
    ):
        report = rdma_test(_Sctx(), ["h1", "h2"], {}, suite="perftest", keep_containers=True)

    stop.assert_not_called()
    assert any(CONTAINER_NAME in w for w in report.warnings)


def test_probe_never_drops_a_host():
    """A failed probe is recorded, not omitted — those read the same otherwise."""
    from sparkrun.api.setup._rdma import _run_probe

    with mock.patch(
        "sparkrun.orchestration.ssh.run_remote_scripts_parallel",
        return_value=[RemoteResult("h1", 0, _facts_kv(), ""), RemoteResult("h2", 1, "", "no route to host")],
    ):
        facts = _run_probe(["h1", "h2"], {}, dry_run=False)

    assert set(facts) == {"h1", "h2"}
    assert facts["h1"].has_perftest
    assert facts["h2"].error


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def test_grade_underperformance_warns_but_a_dead_link_fails():
    """Severity split: slow is a warning, broken is a failure."""
    from sparkrun.api.setup._rdma import _grade_link
    from sparkrun.orchestration.rdma import BwSample, LatSample

    good_lat = LatSample(bytes_=2, iterations=1000, t_min=1.4, t_max=1.9, t_typical=1.47)

    status, _ = _grade_link(good_lat, BwSample(65536, 20000, 111.7, 111.7), 100.0, [], 0.75, 10.0)
    assert status == STATUS_OK

    status, detail = _grade_link(good_lat, BwSample(65536, 20000, 12.0, 10.0), 100.0, [], 0.75, 10.0)
    assert status == STATUS_WARN
    assert "below" in detail

    status, detail = _grade_link(None, None, 100.0, ["bandwidth test produced no result (rc=1)"], 0.75, 10.0)
    assert status == STATUS_FAIL
    assert "no result" in detail


def test_grade_flags_latency_that_is_not_really_rdma():
    from sparkrun.api.setup._rdma import _grade_link
    from sparkrun.orchestration.rdma import BwSample, LatSample

    slow = LatSample(bytes_=2, iterations=1000, t_min=40.0, t_max=90.0, t_typical=55.0)
    status, detail = _grade_link(slow, BwSample(65536, 20000, 111.7, 111.7), 100.0, [], 0.75, 10.0)
    assert status == STATUS_WARN
    assert "latency" in detail


_SPARK_GUID = "4cbb:4703:002c:7beb"


def _dev_facts(host, devices):
    """devices: [(name, phys_port_name, rate_gbps)] all on one adapter."""
    kv = {"RDMA_COMPLETE": "1", "RDMA_DEV_COUNT": str(len(devices))}
    for i, (name, port, rate) in enumerate(devices):
        kv["RDMA_DEV_%d_NAME" % i] = name
        kv["RDMA_DEV_%d_STATE" % i] = "4: ACTIVE"
        kv["RDMA_DEV_%d_RATE" % i] = "%d Gb/sec" % rate
        kv["RDMA_DEV_%d_SYS_IMAGE_GUID" % i] = _SPARK_GUID
        kv["RDMA_DEV_%d_PHYS_PORT_NAME" % i] = port
    return parse_device_facts(host, kv)


def _links(hcas):
    return [RdmaLink("10.%d.0.0/24" % i, "h1", "e", h, "10.%d.0.1" % i, "h2", "e", h, "10.%d.0.2" % i) for i, h in enumerate(hcas)]


def test_devices_on_one_adapter_and_port_are_detected_as_sharing():
    """DGX Spark exposes two PCIe functions of one adapter on ONE port.

    Same sys_image_guid AND same phys_port_name means one wire, so the two
    devices' bandwidths are not additive.
    """
    from sparkrun.api.setup._rdma import _port_share

    facts = {h: _dev_facts(h, [("rocep1s0f0", "p0", 200), ("roceP2p1s0f0", "p0", 200)]) for h in ("h1", "h2")}
    links = _links(["rocep1s0f0", "roceP2p1s0f0"])
    assert _port_share("h1", "rocep1s0f0", links, facts) == 2


def test_dual_port_card_with_two_cables_is_not_shared():
    """Same card, different physical ports — additive, and must stay so."""
    from sparkrun.api.setup._rdma import _port_share

    facts = {h: _dev_facts(h, [("mlx5_0", "p0", 200), ("mlx5_1", "p1", 200)]) for h in ("h1", "h2")}
    links = _links(["mlx5_0", "mlx5_1"])
    assert _port_share("h1", "mlx5_0", links, facts) == 1


def test_per_device_expectation_is_the_ports_fair_share():
    """A healthy Spark link measures ~112 Gb/s on a 200 Gb/s shared port.

    Expecting the port's full rate reported working hardware as
    underperforming — observed live.
    """
    from sparkrun.api.setup._rdma import _expected_gbps

    facts = {h: _dev_facts(h, [("rocep1s0f0", "p0", 200), ("roceP2p1s0f0", "p0", 200)]) for h in ("h1", "h2")}
    links = _links(["rocep1s0f0", "roceP2p1s0f0"])
    expected = _expected_gbps(links[0], facts, links)
    assert expected == 100.0
    assert 111.7 >= expected * 0.75, "a healthy measured link must not warn"


def test_aggregate_expectation_sums_ports_not_links():
    """195.7 Gb/s is ~98% of the cable, not 49% of a fictitious 400."""
    from sparkrun.api.setup._rdma import _run_aggregate

    facts = {h: _dev_facts(h, [("rocep1s0f0", "p0", 200), ("roceP2p1s0f0", "p0", 200)]) for h in ("h1", "h2")}
    links = _links(["rocep1s0f0", "roceP2p1s0f0"])
    pair = RdmaPair(host_a="h1", host_b="h2", links=tuple(links))

    with mock.patch(
        "sparkrun.api.setup._rdma._run_round",
        return_value={0: (REAL_BW_OUTPUT, 0), 1: (REAL_BW_OUTPUT, 0)},
    ):
        total, expected, _detail = _run_aggregate(
            pair, {}, facts, container=None, duration=5, queue_pairs=4, gid_index=None, link_type="IB", timeout=60
        )
    assert expected == 200.0, "two functions on one port must not sum to 400"
    assert total == pytest.approx(223.42)  # 2 x the fixture's 111.71


def test_aggregate_expectation_is_additive_across_real_ports():
    from sparkrun.api.setup._rdma import _run_aggregate

    facts = {h: _dev_facts(h, [("mlx5_0", "p0", 200), ("mlx5_1", "p1", 200)]) for h in ("h1", "h2")}
    links = _links(["mlx5_0", "mlx5_1"])
    pair = RdmaPair(host_a="h1", host_b="h2", links=tuple(links))

    with mock.patch(
        "sparkrun.api.setup._rdma._run_round",
        return_value={0: (REAL_BW_OUTPUT, 0), 1: (REAL_BW_OUTPUT, 0)},
    ):
        _total, expected, _detail = _run_aggregate(
            pair, {}, facts, container=None, duration=5, queue_pairs=4, gid_index=None, link_type="IB", timeout=60
        )
    assert expected == 400.0


def test_port_key_falls_back_when_sysfs_fields_are_absent():
    """Older probes report neither field; each device is then its own port."""
    facts = parse_device_facts(
        "h1", {"RDMA_COMPLETE": "1", "RDMA_DEV_COUNT": "1", "RDMA_DEV_0_NAME": "mlx5_0", "RDMA_DEV_0_RATE": "200 Gb/sec"}
    )
    assert facts.device("mlx5_0").port_key == ("mlx5_0", "")


def test_expected_bandwidth_takes_the_slower_end():
    from sparkrun.api.setup._rdma import _expected_gbps

    facts = {
        "h1": parse_device_facts("h1", {"RDMA_DEV_COUNT": "1", "RDMA_DEV_0_NAME": "a", "RDMA_DEV_0_RATE": "200 Gb/sec"}),
        "h2": parse_device_facts("h2", {"RDMA_DEV_COUNT": "1", "RDMA_DEV_0_NAME": "b", "RDMA_DEV_0_RATE": "100 Gb/sec"}),
    }
    link = derive_link_pairs(
        {
            "h1": _det("h1", [_iface("e", "10.0.0.1", "10.0.0.0/24", "a")]),
            "h2": _det("h2", [_iface("e", "10.0.0.2", "10.0.0.0/24", "b")]),
        },
        ["h1", "h2"],
    )[0].links[0]
    assert _expected_gbps(link, facts) == 100.0


# ---------------------------------------------------------------------------
# setup check integration
# ---------------------------------------------------------------------------


def _check_ctx(multi_host=True):
    from sparkrun.cli._setup._check import CheckContext

    return CheckContext(cluster_name=None, multi_host=multi_host)


def _check_state(**kv):
    from sparkrun.cli._setup._check import HostState

    return HostState(host="h1", rdma=parse_device_facts("h1", {"RDMA_COMPLETE": "1", **kv}))


def test_check_rdma_is_skipped_for_single_host_clusters():
    """RDMA is inter-node; reporting it on one host is noise."""
    from sparkrun.cli._setup._check import _check_rdma

    assert _check_rdma(_check_state(), _check_ctx(multi_host=False)) is None


def test_check_rdma_reports_ok_with_the_link_rate():
    from sparkrun.cli._setup._check import OK, _check_rdma

    item = _check_rdma(
        _check_state(
            RDMA_PERFTEST="1",
            RDMA_DEV_COUNT="2",
            RDMA_DEV_0_NAME="rocep1s0f1",
            RDMA_DEV_0_STATE="4: ACTIVE",
            RDMA_DEV_0_RATE="100 Gb/sec (4X EDR)",
            RDMA_DEV_1_NAME="roceP2p1s0f1",
            RDMA_DEV_1_STATE="4: ACTIVE",
            RDMA_DEV_1_RATE="100 Gb/sec (4X EDR)",
        ),
        _check_ctx(),
    )
    assert item.status == OK
    assert "2 device(s) ACTIVE" in item.detail
    assert "100 Gb/s" in item.detail


def test_check_rdma_warns_when_no_device_is_active():
    from sparkrun.cli._setup._check import WARN, _check_rdma

    item = _check_rdma(
        _check_state(RDMA_DEV_COUNT="1", RDMA_DEV_0_NAME="rocep1s0f1", RDMA_DEV_0_STATE="1: DOWN"),
        _check_ctx(),
    )
    assert item.status == WARN
    assert "setup cx7" in item.guidance


def test_check_rdma_missing_perftest_is_a_note_not_a_warning():
    """It is one command away and rdma-test has a container fallback."""
    from sparkrun.cli._setup._check import OK, _check_rdma

    item = _check_rdma(
        _check_state(
            RDMA_PERFTEST="0",
            RDMA_DEV_COUNT="1",
            RDMA_DEV_0_NAME="rocep1s0f1",
            RDMA_DEV_0_STATE="4: ACTIVE",
            RDMA_DEV_0_RATE="100 Gb/sec",
        ),
        _check_ctx(),
    )
    assert item.status == OK
    assert "perftest not installed" in item.detail


def test_check_rdma_skips_rather_than_failing_when_it_cannot_tell():
    """'Could not probe' and 'no devices' are both SKIP, never FAIL."""
    from sparkrun.cli._setup._check import SKIP, HostState, _check_rdma

    assert _check_rdma(HostState(host="h1", rdma=None), _check_ctx()).status == SKIP
    assert _check_rdma(_check_state(RDMA_DEV_COUNT="0"), _check_ctx()).status == SKIP


def test_check_rdma_is_registered_in_order():
    from sparkrun.cli._setup._check import SETUP_CHECKS

    keys = [c.key for c in SETUP_CHECKS]
    assert "rdma" in keys
    # After cx7: you configure the fabric, then verify the devices came up.
    assert keys.index("rdma") > keys.index("cx7")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def patched_cluster_mgr(tmp_path):
    root = tmp_path / "rdma_config"
    root.mkdir(parents=True, exist_ok=True)
    mgr = ClusterManager(root)
    with (
        mock.patch("sparkrun.cli._common._get_cluster_manager", return_value=mgr),
        mock.patch("sparkrun.cli._setup._get_cluster_manager", return_value=mgr),
    ):
        yield mgr


def test_cli_is_gated_off_by_default(runner, v, patched_cluster_mgr, monkeypatch):
    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_RDMA_TEST", "0")
    result = runner.invoke(main, ["setup", "rdma-test", "--hosts", "h1,h2"])
    assert result.exit_code != 0
    assert "features enable cli.setup.rdma_test" in result.output


def test_cli_requires_two_hosts(runner, v, patched_cluster_mgr, monkeypatch):
    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_RDMA_TEST", "1")
    result = runner.invoke(main, ["setup", "rdma-test", "--hosts", "h1"])
    assert result.exit_code != 0
    assert "at least two hosts" in result.output


def test_cli_dry_run_reports_the_plan(runner, v, patched_cluster_mgr, monkeypatch):
    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_RDMA_TEST", "1")
    with mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_two_spark_direct()):
        result = runner.invoke(main, ["setup", "rdma-test", "--hosts", "h1,h2", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "[dry-run]" in result.output


def test_cli_exits_one_on_failure_and_zero_on_warning(runner, v, patched_cluster_mgr, monkeypatch):
    """WARN does not fail the command; FAIL does — the setup check convention."""
    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_RDMA_TEST", "1")

    def _report(status):
        from sparkrun.api.setup._rdma import LinkTestResult, PairTestResult, RdmaTestReport
        from sparkrun.orchestration.rdma import BwSample, LatSample

        pair = derive_link_pairs(_two_spark_direct(), ["h1", "h2"])[0]
        link = LinkTestResult(
            link=pair.links[0],
            latency=LatSample(2, 1000, 1.4, 1.9, 1.47),
            bandwidth=BwSample(65536, 20000, 111.7, 111.7),
            expected_gbps=100.0,
            status=status,
            detail="synthetic",
        )
        return RdmaTestReport(hosts=("h1", "h2"), pairs=(PairTestResult(pair=pair, links=(link,), status=status),))

    for status, expected_code in ((STATUS_WARN, 0), (STATUS_FAIL, 1)):
        with mock.patch("sparkrun.api.setup.rdma_test", return_value=_report(status)):
            result = runner.invoke(main, ["setup", "rdma-test", "--hosts", "h1,h2"])
        assert result.exit_code == expected_code, (status, result.output)


def test_cli_json_output_carries_the_numbers(runner, v, patched_cluster_mgr, monkeypatch):
    import json

    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_RDMA_TEST", "1")

    from sparkrun.api.setup._rdma import LinkTestResult, PairTestResult, RdmaTestReport
    from sparkrun.orchestration.rdma import BwSample, LatSample

    pair = derive_link_pairs(_two_spark_direct(), ["h1", "h2"])[0]
    link = LinkTestResult(
        link=pair.links[0],
        latency=LatSample(2, 1000, 1.42, 1.93, 1.47, p99=1.57),
        bandwidth=BwSample(65536, 20000, 111.72, 111.71),
        expected_gbps=100.0,
        status=STATUS_OK,
    )
    report = RdmaTestReport(
        hosts=("h1", "h2"),
        pairs=(PairTestResult(pair=pair, links=(link,), aggregate_gbps=196.4, status=STATUS_OK),),
    )

    with mock.patch("sparkrun.api.setup.rdma_test", return_value=report):
        result = runner.invoke(main, ["setup", "rdma-test", "--hosts", "h1,h2", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    entry = payload["pairs"][0]
    assert entry["aggregate_gbps"] == pytest.approx(196.4)
    assert entry["links"][0]["bandwidth_gbps_avg"] == pytest.approx(111.71)
    assert entry["links"][0]["latency_us_typical"] == pytest.approx(1.47)
    assert entry["links"][0]["hca_a"] == "rocep1s0f1"


# ---------------------------------------------------------------------------
# Suite maturity gating
# ---------------------------------------------------------------------------
#
# The two flags gate *maturity*, not blast radius. perftest is proven and
# ships everywhere; the collective rides alpha. Deliberately NOT split on
# "needs the container" — the perftest suite falls back to the image on a host
# without perftest, and that path stays open (asserted below).


def test_channel_defaults_ship_perftest_everywhere_and_nccl_on_alpha_only():
    from sparkrun.core.features import FEATURE_CLI_SETUP_RDMA_TEST, FEATURE_CLI_SETUP_RDMA_TEST_NCCL

    for channel in ("stable", "beta", "alpha"):
        assert FEATURE_CLI_SETUP_RDMA_TEST.default_for_channel(channel) is True, channel

    assert FEATURE_CLI_SETUP_RDMA_TEST_NCCL.default_for_channel("stable") is False
    assert FEATURE_CLI_SETUP_RDMA_TEST_NCCL.default_for_channel("beta") is False
    assert FEATURE_CLI_SETUP_RDMA_TEST_NCCL.default_for_channel("alpha") is True


def test_available_suites_tracks_the_nccl_gate():
    from sparkrun.api.setup._rdma import ALL_SUITES, SUITE_PERFTEST, available_suites

    assert available_suites(_Config(nccl=False)) == (SUITE_PERFTEST,)
    assert available_suites(_Config(nccl=True)) == ALL_SUITES


@pytest.mark.parametrize("suite", ["nccl", "all"])
def test_a_gated_suite_is_refused_before_anything_is_probed(suite):
    """The refusal must precede the probe, not follow a fan-out of SSH."""
    with mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts") as detect:
        with pytest.raises(RdmaTestError) as exc:
            rdma_test(_Sctx(_Config(nccl=False)), ["h1", "h2"], {}, suite=suite)

    assert "cli.setup.rdma_test.nccl" in str(exc.value)
    detect.assert_not_called()


@pytest.mark.parametrize("suite", ["nccl", "all"])
def test_a_dry_run_cannot_plan_a_suite_the_real_run_would_refuse(suite):
    with pytest.raises(RdmaTestError, match="cli.setup.rdma_test.nccl"):
        rdma_test(_Sctx(_Config(nccl=False)), ["h1", "h2"], {}, suite=suite, dry_run=True)


def test_perftest_is_ungated_and_runs_with_the_nccl_flag_off():
    report = rdma_test(_Sctx(_Config(nccl=False)), ["h1", "h2"], {}, suite="perftest", dry_run=True)
    assert report.suite == "perftest"


def test_the_perftest_container_fallback_survives_the_gate():
    """The gate is maturity, not the image.

    Refusing this fallback would break the *proven* suite on any host without
    perftest installed, which is the opposite of what the split is for.
    """
    with (
        _patch_probe(perftest="0"),
        mock.patch("sparkrun.orchestration.networking.detect_cx7_for_hosts", return_value=_two_spark_direct()),
        mock.patch("sparkrun.api.setup._rdma._prepare_container") as prep,
        mock.patch("sparkrun.api.setup._rdma._stop_containers"),
        mock.patch("sparkrun.api.setup._rdma._run_pair", return_value=mock.Mock(status=STATUS_OK, links=(), detail="")),
    ):
        report = rdma_test(_Sctx(_Config(nccl=False)), ["h1", "h2"], {}, suite="perftest")

    prep.assert_called_once()
    assert report.used_container is True
    assert any("perftest not found" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Channel-aware help
# ---------------------------------------------------------------------------


def test_help_omits_a_gated_suite_entirely():
    """Omitted, not annotated: help describes what this install does."""
    from sparkrun.api.setup._rdma import SUITE_PERFTEST
    from sparkrun.cli._setup._rdma import _rdma_help

    text = _rdma_help((SUITE_PERFTEST,))

    assert "perftest" in text
    assert "nccl" not in text
    assert "--suite all" not in text
    # The one-line summary is what `setup --help` shows, so it must not
    # advertise the suite either.
    assert text.splitlines()[0] == "Measure RDMA bandwidth and latency across the fabric."


def test_help_lists_every_suite_once_enabled():
    from sparkrun.api.setup._rdma import ALL_SUITES
    from sparkrun.cli._setup._rdma import _rdma_help

    text = _rdma_help(ALL_SUITES)

    for suite in ALL_SUITES:
        assert suite in text
    assert "--suite all" in text
    assert "NCCL throughput" in text.splitlines()[0]


def test_help_advertises_exactly_what_the_api_would_accept():
    """Help and refusal read one function, so they cannot name different sets."""
    from sparkrun.api.setup._rdma import ALL_SUITES, available_suites
    from sparkrun.cli._setup._rdma import _rdma_help

    for nccl in (True, False):
        config = _Config(nccl=nccl)
        allowed = available_suites(config)
        text = _rdma_help(allowed)
        for suite in ALL_SUITES:
            listed = ("\n  %-9s " % suite) in text
            assert listed is (suite in allowed), (suite, nccl)


def test_completion_offers_only_the_available_suites():
    from sparkrun.api.setup._rdma import ALL_SUITES, SUITE_PERFTEST
    from sparkrun.cli._setup._rdma import _SuiteChoice

    narrowed = _SuiteChoice((SUITE_PERFTEST,))
    assert [i.value for i in narrowed.shell_complete(None, None, "")] == [SUITE_PERFTEST]

    wide = _SuiteChoice(ALL_SUITES)
    assert [i.value for i in wide.shell_complete(None, None, "")] == list(ALL_SUITES)


def test_a_gated_suite_still_parses_so_the_error_can_name_the_flag(runner, monkeypatch):
    """Click's "'nccl' is not one of 'perftest'" names no way forward.

    Keeping the full choice list is what lets the actionable message through,
    and the CLI fails before its "Testing RDMA fabric..." banner can claim
    work that will not happen.
    """
    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_RDMA_TEST_NCCL", "0")

    with mock.patch("sparkrun.api.setup.rdma_test") as api_call:
        result = runner.invoke(main, ["setup", "rdma-test", "--hosts", "h1,h2", "--suite", "nccl", "--dry-run"])

    assert result.exit_code != 0
    assert "sparkrun setup features enable cli.setup.rdma_test.nccl" in result.output
    assert "is not one of" not in result.output
    assert "Testing RDMA fabric" not in result.output
    api_call.assert_not_called()
