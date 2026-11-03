"""Management-interface reporting: probe keys → record → findings → CLI.

The shell half (which interface, which rule) is exercised under real bash in
``test_mgmt_iface.py``; this covers what sparkrun does with the report.
"""

from __future__ import annotations

import logging
from unittest import mock

from click.testing import CliRunner

from sparkrun.core.hardware import HostHardware
from sparkrun.core.setup_checks import _check_mgmt_interface
from sparkrun.core.setup_models import OK, WARN, CheckContext, HostState
from sparkrun.orchestration.infiniband import ib_detection_from_observations
from sparkrun.orchestration.mgmt_iface import ManagementInterface, from_probe, management_findings


def _facts(iface="enP7s7", ip="10.24.68.102", rule="peers", kind="wired", **extra):
    return {
        "IB_DETECTED": "0",
        "DETECTED_MGMT_IFACE": iface,
        "DETECTED_MGMT_IP": ip,
        "DETECTED_MGMT_RULE": rule,
        "DETECTED_MGMT_KIND": kind,
        **extra,
    }


def test_from_probe_reads_the_report():
    mi = from_probe("h1", _facts())
    assert mi == ManagementInterface("h1", "enP7s7", "10.24.68.102", "peers", "wired")
    assert mi.decided_by == "reaches the other hosts"


def test_from_probe_distinguishes_unreported_from_unresolved():
    """An older probe says nothing; a probe that found nothing says so."""
    assert from_probe("h1", {"IB_DETECTED": "0"}) is None
    assert from_probe("h1", None) is None
    unresolved = from_probe("h1", _facts(iface="", ip="", rule="", kind=""))
    assert unresolved is not None and unresolved.interface == ""


def test_findings_flag_wireless_and_fallback_rules():
    interfaces = {
        "wired": from_probe("wired", _facts()),
        "wifi": from_probe("wifi", _facts(iface="wlP9s9", kind="wireless")),
        "fallback": from_probe("fallback", _facts(rule="default-route")),
        "pinned": from_probe("pinned", _facts(rule="pinned")),
    }
    findings = management_findings(interfaces, multi_host=True)
    assert set(findings) == {"wifi", "fallback"}
    assert "wireless" in findings["wifi"][0]
    assert "no interface routes to the other hosts" in findings["fallback"][0]
    assert "default route (fallback)" in findings["fallback"][0]


def test_fallback_rule_is_expected_without_peers():
    """Single host: nothing to measure against, so a fallback is not a finding."""
    findings = management_findings({"h": from_probe("h", _facts(rule="default-route"))}, multi_host=False)
    assert findings == {}


def test_detection_result_carries_the_report_and_logs_findings(caplog):
    observations = {"h1": _facts(), "h2": _facts(iface="wlP9s9", ip="192.168.1.50", kind="wireless")}
    with caplog.at_level(logging.WARNING, logger="sparkrun.orchestration.infiniband"):
        result = ib_detection_from_observations(["h1", "h2"], observations)
    assert result.mgmt_interfaces["h2"].kind == "wireless"
    # The address now comes from the report even on a host without IB.
    assert result.mgmt_ip_map == {"h1": "10.24.68.102", "h2": "192.168.1.50"}
    assert [r.getMessage() for r in caplog.records if "wireless" in r.getMessage()]


def test_detection_result_can_leave_findings_to_the_caller(caplog):
    observations = {"h1": _facts(), "h2": _facts(iface="wlP9s9", kind="wireless")}
    with caplog.at_level(logging.WARNING, logger="sparkrun.orchestration.infiniband"):
        ib_detection_from_observations(["h1", "h2"], observations, log_findings=False)
    assert not [r for r in caplog.records if "wireless" in r.getMessage()]


def _state(facts):
    return HostState(host="h1", hardware=HostHardware(ib_info=facts))


def _ctx(multi_host=True):
    return CheckContext(cluster_name="quad", multi_host=multi_host)


def test_setup_check_reports_the_choice():
    item = _check_mgmt_interface(_state(_facts()), _ctx())
    assert item.status == OK
    assert item.detail == "enP7s7 10.24.68.102 (wired, reaches the other hosts)"


def test_setup_check_warns_on_wireless_with_a_pin_command():
    item = _check_mgmt_interface(_state(_facts(iface="wlP9s9", kind="wireless")), _ctx())
    assert item.status == WARN
    assert "wireless" in item.detail
    assert item.guidance == "sparkrun cluster update quad --mgmt-interface wlP9s9"


def test_setup_check_warns_when_nothing_resolved():
    item = _check_mgmt_interface(_state(_facts(iface="", ip="", rule="", kind="")), _ctx())
    assert item.status == WARN
    assert "no management interface" in item.detail


def test_setup_check_is_silent_for_single_host_or_unreported():
    assert _check_mgmt_interface(_state(_facts()), _ctx(multi_host=False)) is None
    assert _check_mgmt_interface(_state({"IB_DETECTED": "0"}), _ctx()) is None


def test_inspect_renders_the_management_network(v):
    from sparkrun.cli import main
    from sparkrun.orchestration.disk_info import CacheStatus
    from sparkrun.orchestration.distribution import TransferModeResult
    from sparkrun.orchestration.infiniband import IBDetectionResult

    interfaces = {
        "10.0.0.1": ManagementInterface("10.0.0.1", "enP7s7", "10.0.0.1", "peers", "wired"),
        "10.0.0.2": ManagementInterface("10.0.0.2", "wlP9s9", "192.168.1.50", "default-route", "wireless"),
    }
    with (
        mock.patch("sparkrun.core.launcher.resolve_effective_cache_dir", return_value="/home/u/.cache/huggingface"),
        mock.patch(
            "sparkrun.orchestration.distribution.resolve_auto_transfer_mode",
            return_value=TransferModeResult(mode="delegated", ib_result=IBDetectionResult(mgmt_interfaces=interfaces)),
        ),
        mock.patch("sparkrun.orchestration.disk_info.probe_cache_status", return_value={"10.0.0.1": CacheStatus(host="10.0.0.1")}),
        mock.patch("sparkrun.diagnostics.collect_spark_diagnostics", return_value={}),
    ):
        result = CliRunner().invoke(main, ["cluster", "inspect", "--hosts", "10.0.0.1,10.0.0.2"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "Management Network:" in result.output
    out = result.output
    assert out.index("Cluster Configuration:") < out.index("Management Network:") < out.index("Head Node Hardware")
    assert "reaches the other hosts" in result.output
    assert "[WARN] 10.0.0.2: management interface wlP9s9 is wireless" in result.output
    assert "--mgmt-interface" in result.output
