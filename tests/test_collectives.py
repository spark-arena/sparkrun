"""Tests for the collective backend abstraction (Phase 5)."""

from __future__ import annotations

import pytest

from sparkrun.orchestration.collectives import (
    HcclBackend,
    NcclBackend,
    RcclBackend,
    UnsupportedCollectiveError,
    get_backend,
)
from sparkrun.orchestration.infiniband import generate_nccl_env, generate_ring_nccl_overrides


# --------------------------------------------------------------------------
# Vendor → backend factory
# --------------------------------------------------------------------------


@pytest.mark.parametrize("vendor", [None, "nvidia", "NVIDIA"])
def test_get_backend_defaults_to_nccl(vendor):
    """None and any-case "nvidia" both yield NCCL — the legacy default."""
    backend = get_backend(vendor)
    assert isinstance(backend, NcclBackend)
    assert backend.name == "nccl"
    assert backend.vendor == "nvidia"


def test_get_backend_amd_returns_rccl_scaffold():
    backend = get_backend("amd")
    assert isinstance(backend, RcclBackend)
    assert backend.name == "rccl"


def test_get_backend_intel_returns_hccl_scaffold():
    backend = get_backend("intel")
    assert isinstance(backend, HcclBackend)
    assert backend.name == "hccl"


@pytest.mark.parametrize("vendor", ["apple", "cpu", "weird-fpga", ""])
def test_get_backend_unknown_vendor_raises(vendor):
    """Apple/CPU/unknown vendors raise rather than silently producing NCCL env."""
    if vendor == "":
        # empty string falls through to "nvidia" default
        assert isinstance(get_backend(vendor), NcclBackend)
        return
    with pytest.raises(UnsupportedCollectiveError, match="No collective backend"):
        get_backend(vendor)


# --------------------------------------------------------------------------
# NCCL backend parity with legacy generator (byte-for-byte)
# --------------------------------------------------------------------------


_IB_DETECTED = {
    "IB_DETECTED": "1",
    "DETECTED_HCA_LIST": "mlx5_0,mlx5_1",
    "DETECTED_NET_LIST": "ibp1s0f0,ibp2s0f0",
    "DETECTED_SOCKET_IFNAME": "enp1s0f0",
    "DETECTED_GID_INDEX": "3",
    "DETECTED_UCX_LIST": "mlx5_0:1,mlx5_1:1",
    "DETECTED_MGMT_IP": "10.0.0.5",
}


def test_nccl_backend_matches_legacy_generator():
    """NcclBackend.env_for_host produces output identical to generate_nccl_env."""
    legacy = generate_nccl_env(_IB_DETECTED, topology=None)
    via_backend = get_backend("nvidia").env_for_host(_IB_DETECTED, topology=None)
    assert via_backend == legacy


def test_nccl_backend_matches_resolve_ib_env_legacy():
    """A1 back-compat contract: NcclBackend on a canonical NVIDIA+IB host
    emits the exact env dict that ``resolve_ib_env`` would produce.

    The legacy resolve_ib_env path routes detection output through
    :func:`generate_nccl_env`, so for NVIDIA hosts both code paths
    must produce a byte-identical env block.  Asserts both the *keys*
    and *values* match — including every optional NCCL tunable
    (HCA list, GID index, socket interface, UCX devices).
    """
    legacy_env = generate_nccl_env(_IB_DETECTED, topology=None)
    new_env = get_backend("nvidia").env_for_host(_IB_DETECTED, topology=None)

    # Sanity: legacy actually emits the expected NCCL tunables for this fixture.
    assert legacy_env["NCCL_IB_HCA"] == _IB_DETECTED["DETECTED_HCA_LIST"]
    assert legacy_env["NCCL_IB_GID_INDEX"] == _IB_DETECTED["DETECTED_GID_INDEX"]
    assert "NCCL_SOCKET_IFNAME" in legacy_env  # head-of-list mgmt + IB nets
    assert legacy_env["UCX_NET_DEVICES"] == _IB_DETECTED["DETECTED_UCX_LIST"]

    # Byte-identical to the legacy generator
    assert new_env == legacy_env
    # Same key set (nothing dropped, nothing added)
    assert set(new_env.keys()) == set(legacy_env.keys())
    # Same values for every key
    for k in legacy_env:
        assert new_env[k] == legacy_env[k], "Parity mismatch on %s: legacy=%r backend=%r" % (k, legacy_env[k], new_env[k])


def test_nccl_backend_ring_topology_matches_legacy():
    legacy = generate_nccl_env(_IB_DETECTED, topology="ring")
    via_backend = get_backend("nvidia").env_for_host(_IB_DETECTED, topology="ring")
    assert via_backend == legacy
    # Sanity: ring topology actually overrides plugin
    assert via_backend["NCCL_NET_PLUGIN"] == "none"


def test_nccl_backend_empty_ib_returns_empty_dict():
    """No IB detected -> no env (matches legacy)."""
    assert get_backend("nvidia").env_for_host({}, topology=None) == {}


def test_nccl_backend_ring_overrides_matches_legacy():
    assert NcclBackend().ring_overrides(_IB_DETECTED) == generate_ring_nccl_overrides(_IB_DETECTED)


# --------------------------------------------------------------------------
# AMD / Intel scaffolds raise clear NotImplementedError
# --------------------------------------------------------------------------


def test_rccl_env_raises_not_implemented_with_actionable_message():
    backend = get_backend("amd")
    with pytest.raises(NotImplementedError, match="RCCL backend is not yet implemented"):
        backend.env_for_host(_IB_DETECTED)


def test_rccl_ring_overrides_raises():
    with pytest.raises(NotImplementedError, match="RCCL ring-topology"):
        RcclBackend().ring_overrides(_IB_DETECTED)


def test_hccl_env_raises_not_implemented_with_actionable_message():
    backend = get_backend("intel")
    with pytest.raises(NotImplementedError, match="HCCL backend is not yet implemented"):
        backend.env_for_host(_IB_DETECTED)


def test_hccl_ring_overrides_raises():
    with pytest.raises(NotImplementedError, match="HCCL ring-topology"):
        HcclBackend().ring_overrides(_IB_DETECTED)


# --------------------------------------------------------------------------
# Heterogeneous cluster: caller can resolve per-host backend explicitly
# --------------------------------------------------------------------------


def test_heterogeneous_cluster_resolves_per_host_backend():
    """A mixed cluster yields different backends per host via get_backend."""
    per_host = {h: get_backend(v) for h, v in [("spark-01", "nvidia"), ("mi300x-box", "amd")]}
    assert per_host["spark-01"].name == "nccl"
    assert per_host["mi300x-box"].name == "rccl"
    # NVIDIA host returns real env; AMD raises (as expected for a scaffold).
    assert per_host["spark-01"].env_for_host(_IB_DETECTED) == generate_nccl_env(_IB_DETECTED)
    with pytest.raises(NotImplementedError):
        per_host["mi300x-box"].env_for_host(_IB_DETECTED)
