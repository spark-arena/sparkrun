"""CX7 apply ordering and route recovery around a local ``netplan apply`` (issue #311).

The control node's own ``netplan apply`` can briefly drop its management
routes; SSH attempts made in that window fail instantly with "Network is
unreachable".  So the local host is configured last, routes are awaited
after it, and a remote attempt that still finds no route is retried once.
"""

from __future__ import annotations

import socket
from unittest import mock

import pytest

from sparkrun.orchestration.networking import CX7ClusterPlan, CX7HostPlan, CX7InterfaceAssignment, apply_cx7_plan
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.utils import net

LOCAL = "192.168.68.41"
REMOTES = ["192.168.68.42", "192.168.68.43", "192.168.68.44"]
UNREACHABLE = "ssh: connect to host %s port 22: Network is unreachable"


def _plan(hosts, needs_change=True):
    def host_plan(host):
        octet = host.rsplit(".", 1)[-1]
        assignments = [CX7InterfaceAssignment("if%d" % n, "10.%d.0.%s" % (n, octet), "10.%d.0.0/24" % n) for n in range(2)]
        return CX7HostPlan(host=host, assignments=assignments, needs_change=needs_change)

    return CX7ClusterPlan(host_plans=[host_plan(h) for h in hosts])


@pytest.fixture
def events():
    return []


@pytest.fixture
def patched(events):
    """Record configure calls and route waits in one ordered log."""
    outcomes: dict[str, list[RemoteResult]] = {}

    def configure(hp, mtu, prefix_len, ssh_kwargs=None, dry_run=False, sudo_password=None):
        events.append(("configure", hp.host))
        queue = outcomes.get(hp.host)
        return queue.pop(0) if queue else RemoteResult(hp.host, 0, "CX7_CONFIGURED=1", "")

    def wait(hosts, timeout_s=30.0, **_kw):
        events.append(("wait", list(hosts)))
        return []

    with (
        mock.patch("sparkrun.orchestration.networking.ROUTE_RETRY_DELAY_S", 0),
        mock.patch("sparkrun.orchestration.networking.configure_cx7_host", side_effect=configure),
        mock.patch("sparkrun.utils.net.is_local_host", side_effect=lambda h: h == LOCAL),
        mock.patch("sparkrun.utils.net.wait_for_routes", side_effect=wait),
    ):
        yield outcomes


def test_local_host_is_configured_last_then_routes_are_awaited(patched, events):
    results = apply_cx7_plan(_plan([LOCAL, *REMOTES]))

    assert events == [("configure", h) for h in REMOTES] + [("configure", LOCAL), ("wait", REMOTES)]
    assert [r.host for r in results] == [*REMOTES, LOCAL]
    assert all(r.success for r in results)


def test_routes_are_awaited_for_hosts_not_being_changed_too(patched, events):
    """The caller's next step (host-key distribution) reaches every host, configured or not."""
    plan = _plan([LOCAL, *REMOTES])
    plan.host_plans[1].needs_change = False

    apply_cx7_plan(plan)

    assert events[-1] == ("wait", REMOTES)


def test_no_route_wait_without_a_local_apply(patched, events):
    apply_cx7_plan(_plan(REMOTES))
    assert all(kind == "configure" for kind, _ in events)


def test_no_route_wait_under_dry_run(patched, events):
    apply_cx7_plan(_plan([LOCAL, *REMOTES]), dry_run=True)
    assert all(kind == "configure" for kind, _ in events)


def test_remote_route_failure_is_retried_once_after_the_route_returns(patched, events):
    host = REMOTES[0]
    patched[host] = [RemoteResult(host, 255, "", UNREACHABLE % host)]

    results = apply_cx7_plan(_plan(REMOTES))

    assert events[:3] == [("configure", host), ("wait", [host]), ("configure", host)]
    assert results[0].success


def test_route_failure_retry_happens_only_once(patched, events):
    host = REMOTES[0]
    patched[host] = [RemoteResult(host, 255, "", UNREACHABLE % host)] * 2

    results = apply_cx7_plan(_plan([host]))

    assert events == [("configure", host), ("wait", [host]), ("configure", host)]
    assert not results[0].success


@pytest.mark.parametrize(
    "result",
    [
        RemoteResult("h", 255, "", "ssh: connect to host h port 22: Connection refused"),
        RemoteResult("h", 1, "", UNREACHABLE % "h"),  # the script's own failure, not ssh's
        # A session that opened and then died: part of the script ran, so it is not re-run.
        RemoteResult("h", 255, "Configuring CX7 interfaces:", "client_loop: send disconnect: Network is unreachable"),
    ],
)
def test_other_failures_are_not_retried(patched, events, result):
    patched["h"] = [result]
    apply_cx7_plan(_plan(["h"]))
    assert events == [("configure", "h")]


def test_local_route_failure_is_not_retried(patched, events):
    """A local apply never went through SSH; re-running it would re-apply netplan."""
    patched[LOCAL] = [RemoteResult(LOCAL, 255, "", UNREACHABLE % LOCAL)]
    apply_cx7_plan(_plan([LOCAL]))
    assert events == [("configure", LOCAL)]


def test_failures_are_left_to_callers_to_render(patched, caplog):
    """The CLI and wizard print each failure; logging it at error printed it twice."""
    patched["h"] = [RemoteResult("h", 1, "", "netplan exploded")]
    with caplog.at_level("INFO", logger="sparkrun.orchestration.networking"):
        apply_cx7_plan(_plan(["h"]))
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


# ---------------------------------------------------------------------------
# utils.net route primitives
# ---------------------------------------------------------------------------


def test_has_route_to_loopback():
    assert net.has_route_to("127.0.0.1")


def test_has_route_to_cannot_tell_for_an_unresolvable_name():
    """An ~/.ssh/config alias does not resolve here; that is not "no route"."""
    with mock.patch("socket.getaddrinfo", side_effect=socket.gaierror("no name")):
        assert net.has_route_to("spark-02") is None


def test_wait_for_routes_does_not_wait_for_hosts_it_cannot_check():
    sleeps = []
    with mock.patch("sparkrun.utils.net.has_route_to", return_value=None):
        assert net.wait_for_routes(["spark-02"], timeout_s=30, sleep=sleeps.append) == []
    assert sleeps == []


def test_has_route_to_is_false_when_connect_finds_no_route():
    sock = mock.MagicMock()
    sock.__enter__.return_value.connect.side_effect = OSError(101, "Network is unreachable")
    with (
        mock.patch("socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_DGRAM, 0, "", ("10.9.9.9", 22))]),
        mock.patch("socket.socket", return_value=sock),
    ):
        assert net.has_route_to("10.9.9.9") is False


def test_wait_for_routes_returns_once_routes_recover():
    answers = iter([False, False, True])
    sleeps = []
    with mock.patch("sparkrun.utils.net.has_route_to", side_effect=lambda h: next(answers)):
        missing = net.wait_for_routes(["h"], timeout_s=60, sleep=sleeps.append)
    assert missing == []
    assert len(sleeps) == 2


def test_wait_for_routes_reports_what_never_recovered():
    clock = iter(range(0, 1000, 10))
    with (
        mock.patch("sparkrun.utils.net.has_route_to", side_effect=lambda h: h == "up"),
        mock.patch("sparkrun.utils.net.time.monotonic", side_effect=lambda: next(clock)),
    ):
        missing = net.wait_for_routes(["up", "down", "down"], timeout_s=30, sleep=lambda _s: None)
    assert missing == ["down"]
