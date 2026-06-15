"""Import a legacy spark-vllm-docker ``.env`` into a sparkrun cluster.

The spark-vllm-docker world configured a single implicit cluster via an
``.env`` file (``CLUSTER_NODES``, ``ETH_IF``, ``CONTAINER_*`` …).  This
module maps that file onto a sparkrun :class:`ClusterDefinition`, carrying
only the non-derivable intent and deliberately dropping everything sparkrun
re-derives at runtime.  Pure mapping logic — no I/O beyond reading the file,
no cluster mutation; the CLI layer performs the upsert.

Mapping:
  CLUSTER_NODES  -> hosts (as-is; not normalized to mgmt IPs)
  ETH_IF         -> fabric_interfaces (port suffix ``npN`` -> ``*npN``)
  CONTAINER_*    -> env as ``${ORIGINAL_KEY}`` references resolved from the
                    env file at launch (secrets never enter cluster YAML)

Dropped (with reason):
  NCCL mesh vars       -> sparkrun derives these from detected topology
  LOCAL_IP             -> auto-detected
  COPY_HOSTS           -> distribution targets are derived
  MASTER_PORT          -> run-level (--init-port)
  CONTAINER_NAME       -> run-level container identity
  IB_IF                -> corroborates the port; fabric comes from ETH_IF
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from sparkrun.core.cluster_manager import ClusterError, _parse_env_file

SYNC_SOURCE_TYPE = "spark_vllm_docker"

# Container NCCL vars that sparkrun derives itself from detected topology —
# carrying them would conflict with the collectives backend.
_DERIVED_NCCL_VARS = frozenset(
    {
        "NCCL_NET_PLUGIN",
        "NCCL_IB_SUBNET_AWARE_ROUTING",
        "NCCL_IB_MERGE_NICS",
    }
)

_NP_SUFFIX_RE = re.compile(r"(np\d+)$")
_NAME_SANITIZE_RE = re.compile(r"[^a-z0-9_-]+")


@dataclass
class SvdImport:
    """Result of mapping a spark-vllm-docker ``.env`` to cluster fields."""

    hosts: list[str]
    fabric_interfaces: list[str]
    env: dict[str, str]
    env_file: str
    sync_source: str
    default_name: str
    carried: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def _derive_default_name(path: Path) -> str:
    """Stable cluster name from the env-file path.

    Uses the file stem when meaningful, else the parent directory name
    (the canonical svd file is literally ``.env``).  Sanitized to the
    cluster-name charset.
    """
    stem = path.stem  # ".env" -> "" ; "prod-4x.env" -> "prod-4x"
    base = stem if stem and not path.name.startswith(".") else path.parent.name
    name = _NAME_SANITIZE_RE.sub("-", base.lower()).strip("-")
    return name or "imported-cluster"


def _fabric_from_eth_if(eth_if: str) -> str | None:
    """Map an ETH_IF netdev name to a fabric_interfaces glob.

    ``enp1s0f1np1`` -> ``*np1`` (the #203 port-pin format).  Falls back to
    the exact interface name when no ``npN`` suffix is present.
    """
    eth_if = eth_if.strip()
    if not eth_if:
        return None
    m = _NP_SUFFIX_RE.search(eth_if)
    return "*%s" % m.group(1) if m else eth_if


def build_svd_import(path: str) -> SvdImport:
    """Parse a spark-vllm-docker ``.env`` and map it to cluster fields.

    Raises :class:`ClusterError` if the file is unreadable or has no
    ``CLUSTER_NODES``.
    """
    abs_path = str(Path(path).expanduser().resolve())
    raw = _parse_env_file(abs_path)

    nodes = [h.strip() for h in raw.get("CLUSTER_NODES", "").split(",") if h.strip()]
    if not nodes:
        raise ClusterError("spark-vllm-docker env %s has no CLUSTER_NODES" % path)

    carried: list[str] = ["hosts (from CLUSTER_NODES): %s" % ", ".join(nodes)]
    dropped: list[str] = []

    fabric: list[str] = []
    eth_if = raw.get("ETH_IF", "")
    glob = _fabric_from_eth_if(eth_if) if eth_if else None
    if glob:
        fabric = [glob]
        carried.append("fabric_interfaces (from ETH_IF %s): %s" % (eth_if, glob))

    # CONTAINER_* -> env references (minus CONTAINER_NAME and derived NCCL vars).
    env: dict[str, str] = {}
    for key in sorted(raw):
        if not key.startswith("CONTAINER_"):
            continue
        target = key[len("CONTAINER_") :]
        if not target or key == "CONTAINER_NAME":
            dropped.append("%s (run-level container identity)" % key)
            continue
        if target in _DERIVED_NCCL_VARS:
            dropped.append("%s (NCCL mesh var — sparkrun derives from topology)" % key)
            continue
        env[target] = "${%s}" % key
    if env:
        carried.append("env references: %s" % ", ".join(sorted(env)))

    # Report the rest of the dropped/derived keys.
    for key, reason in (
        ("COPY_HOSTS", "distribution targets are derived"),
        ("LOCAL_IP", "auto-detected"),
        ("MASTER_PORT", "run-level (--init-port)"),
        ("IB_IF", "corroborates port; fabric comes from ETH_IF"),
    ):
        if raw.get(key):
            dropped.append("%s (%s)" % (key, reason))

    return SvdImport(
        hosts=nodes,
        fabric_interfaces=fabric,
        env=env,
        env_file=abs_path,
        sync_source="%s:%s" % (SYNC_SOURCE_TYPE, abs_path),
        default_name=_derive_default_name(Path(abs_path)),
        carried=carried,
        dropped=dropped,
    )
