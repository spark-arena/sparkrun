"""Management-interface choices reported by the host probes.

``scripts/_mgmt_iface.sh`` decides each host's management interface (the one
behind ``NODE_IP``, the gloo/TP socket env and the native head address) and,
through ``sparkrun_mgmt_report``, says which rule decided and what kind of
interface won.  This module turns those ``DETECTED_MGMT_*`` probe keys into a
typed record and the findings worth telling an operator about, so that
``cluster inspect``, ``setup check`` and the launch log cannot word them
differently.  Console-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

#: What each ``DETECTED_MGMT_RULE`` means, worded for display.
RULE_DESCRIPTIONS: dict[str, str] = {
    "pinned": "pinned by the cluster's mgmt_interface",
    "peers": "reaches the other hosts",
    "ssh": "the SSH session arrived on it",
    "default-route": "default route (fallback)",
    "ssh-virtual": "the SSH session arrived on it (virtual interface)",
    "scan": "first active NIC (fallback)",
}

#: Rules that are evidence about the cluster rather than a fallback.
_DECIDED_BY_CLUSTER = frozenset({"pinned", "peers"})


@dataclass(frozen=True)
class ManagementInterface:
    """One host's management interface, as its probe resolved it."""

    host: str
    interface: str
    """Empty when the probe could not establish one (see ``_mgmt_iface.sh``)."""
    address: str
    rule: str
    """Which rule chose it; a key of :data:`RULE_DESCRIPTIONS`, or empty."""
    kind: str
    """``wired``, ``wireless``, ``virtual`` or ``fabric``; empty when unknown."""

    @property
    def decided_by(self) -> str:
        """:attr:`rule` in words."""
        return RULE_DESCRIPTIONS.get(self.rule, self.rule or "unknown")

    def to_dict(self) -> dict[str, str]:
        return {**asdict(self), "decided_by": self.decided_by}


def from_probe(host: str, facts: Mapping[str, str] | None) -> ManagementInterface | None:
    """Read the ``DETECTED_MGMT_*`` keys of one host's probe output.

    Returns ``None`` when the probe did not report them (an older probe, or a
    failed one), which is different from reporting that nothing resolved.
    """
    if not facts or "DETECTED_MGMT_RULE" not in facts:
        return None
    return ManagementInterface(
        host=host,
        interface=(facts.get("DETECTED_MGMT_IFACE") or facts.get("DETECTED_SOCKET_IFNAME") or "").strip(),
        address=(facts.get("DETECTED_MGMT_IP") or "").strip(),
        rule=(facts.get("DETECTED_MGMT_RULE") or "").strip(),
        kind=(facts.get("DETECTED_MGMT_KIND") or "").strip(),
    )


def management_findings(interfaces: Mapping[str, ManagementInterface], *, multi_host: bool) -> dict[str, list[str]]:
    """What an operator should hear about these choices, per host.

    *multi_host* says whether the probes were given the other hosts to measure
    against; without peers a fallback rule is the expected outcome and not
    worth a finding.  A host with no interface at all is left to the caller,
    whose consequence (fabric-bound control traffic) it knows how to word.
    """
    findings: dict[str, list[str]] = {}
    for host, mi in interfaces.items():
        notes: list[str] = []
        if not mi.interface:
            continue
        if mi.kind == "wireless":
            notes.append("management interface %s is wireless" % mi.interface)
        if multi_host and mi.rule not in _DECIDED_BY_CLUSTER:
            notes.append(
                "no interface routes to the other hosts; %s was chosen by %s — check that the cluster's "
                "host addresses resolve on this host" % (mi.interface, mi.decided_by)
            )
        if notes:
            findings[host] = notes
    return findings


#: Shared remedy line, so the CLI surfaces cannot drift.
PIN_GUIDANCE = "pin the right NIC with the cluster's mgmt_interface (cluster update <name> --mgmt-interface <iface>)"
