"""Tests for sparkrun.core.fingerprint (Phase 6)."""

from __future__ import annotations

from sparkrun.core.fingerprint import (
    build_host_hardware,
    compute_fingerprint_hash,
    generate_fingerprint_script,
    parse_fingerprint_output,
)
from sparkrun.core.hardware import AcceleratorSpec


# --------------------------------------------------------------------------
# Probe script
# --------------------------------------------------------------------------


def test_generate_fingerprint_script_is_bash_with_emit():
    """The script is well-formed bash and emits KEY=VALUE pairs."""
    script = generate_fingerprint_script()
    assert script.startswith("#!/bin/bash")
    assert "emit()" in script
    assert "NVIDIA_PRESENT" in script
    assert "AMD_PRESENT" in script
    assert "INTEL_PRESENT" in script
    assert "APPLE_PRESENT" in script
    assert "IB_PRESENT" in script


# --------------------------------------------------------------------------
# parse_fingerprint_output
# --------------------------------------------------------------------------


def test_parse_fingerprint_output_basic():
    out = parse_fingerprint_output("NVIDIA_PRESENT=1\nNVIDIA_GPU_COUNT=1\nNVIDIA_GPU_0_NAME=NVIDIA GB10\nNVIDIA_GPU_0_MEMORY_MIB=131072\n")
    assert out["NVIDIA_PRESENT"] == "1"
    assert out["NVIDIA_GPU_0_NAME"] == "NVIDIA GB10"


def test_parse_fingerprint_output_ignores_blank_and_comments():
    out = parse_fingerprint_output("\n# comment line\nNVIDIA_PRESENT=0\n   \nBAD LINE WITHOUT EQ\nIB_PRESENT=1\n")
    assert out == {"NVIDIA_PRESENT": "0", "IB_PRESENT": "1"}


def test_parse_fingerprint_output_handles_equals_in_value():
    out = parse_fingerprint_output("KEY=val=ue\n")
    assert out == {"KEY": "val=ue"}


# --------------------------------------------------------------------------
# build_host_hardware — captured-fixture style probes
# --------------------------------------------------------------------------


def _probe_dgx_spark() -> str:
    return (
        "NVIDIA_PRESENT=1\n"
        "NVIDIA_GPU_COUNT=1\n"
        "NVIDIA_GPU_0_NAME=NVIDIA GB10\n"
        "NVIDIA_GPU_0_MEMORY_MIB=131072\n"
        "AMD_PRESENT=0\n"
        "AMD_GPU_COUNT=0\n"
        "INTEL_PRESENT=0\n"
        "INTEL_GAUDI_COUNT=0\n"
        "APPLE_PRESENT=0\n"
        "IB_PRESENT=1\n"
        "OS=Linux\n"
        "ARCH=aarch64\n"
    )


def test_build_host_hardware_dgx_spark():
    hw = build_host_hardware(parse_fingerprint_output(_probe_dgx_spark()))
    assert len(hw.accelerators) == 1
    a = hw.accelerators[0]
    assert a.vendor == "nvidia"
    assert a.model == "gb10"
    assert a.count == 1
    assert a.memory_gb == 128.0  # 131072 MiB / 1024 = 128 GiB
    assert "cuda" in a.capabilities
    assert "rdma:roce-v2" in a.capabilities
    assert hw.fingerprint and len(hw.fingerprint) == 16
    assert "Linux/aarch64" in hw.notes


def test_build_host_hardware_h200_8gpu():
    parsed = parse_fingerprint_output(
        "NVIDIA_PRESENT=1\n"
        "NVIDIA_GPU_COUNT=8\n"
        + "".join("NVIDIA_GPU_%d_NAME=NVIDIA H200\nNVIDIA_GPU_%d_MEMORY_MIB=144384\n" % (i, i) for i in range(8))
        + "AMD_GPU_COUNT=0\nINTEL_GAUDI_COUNT=0\nAPPLE_PRESENT=0\nIB_PRESENT=1\nOS=Linux\nARCH=x86_64\n"
    )
    hw = build_host_hardware(parsed)
    # 8 identical GPUs compact into a single AcceleratorSpec with count=8.
    assert len(hw.accelerators) == 1
    assert hw.accelerators[0] == AcceleratorSpec(
        vendor="nvidia",
        model="h200",
        count=8,
        memory_gb=141.0,
        capabilities=frozenset({"cuda", "rdma:roce-v2"}),
    )


def test_build_host_hardware_mi300x_4gpu_no_ib():
    parsed = parse_fingerprint_output(
        "NVIDIA_GPU_COUNT=0\n"
        "AMD_PRESENT=1\n"
        "AMD_GPU_COUNT=4\n"
        + "".join("AMD_GPU_%d_NAME=Instinct MI300X\nAMD_GPU_%d_MEMORY_MIB=196608\n" % (i, i) for i in range(4))
        + "INTEL_GAUDI_COUNT=0\nAPPLE_PRESENT=0\nIB_PRESENT=0\nOS=Linux\nARCH=x86_64\n"
    )
    hw = build_host_hardware(parsed)
    assert len(hw.accelerators) == 1
    spec = hw.accelerators[0]
    assert spec.vendor == "amd"
    assert spec.model == "mi300x"
    assert spec.count == 4
    assert spec.memory_gb == 192.0
    assert "rocm" in spec.capabilities
    # No IB on this box -> no rdma cap tag.
    assert "rdma:roce-v2" not in spec.capabilities


def test_build_host_hardware_apple_m5():
    parsed = parse_fingerprint_output(
        "NVIDIA_GPU_COUNT=0\nAMD_GPU_COUNT=0\nINTEL_GAUDI_COUNT=0\n"
        "APPLE_PRESENT=1\nAPPLE_MODEL=Apple M5\n"
        "IB_PRESENT=0\nOS=Darwin\nARCH=arm64\n"
    )
    hw = build_host_hardware(parsed)
    assert len(hw.accelerators) == 1
    assert hw.accelerators[0].vendor == "apple"
    assert hw.accelerators[0].model == "m5"
    assert "mlx" in hw.accelerators[0].capabilities


def test_build_host_hardware_mixed_nvidia_plus_apple():
    """Laptop with discrete NVIDIA + Apple integrated GPU (theoretical) -> two specs."""
    parsed = parse_fingerprint_output(
        "NVIDIA_PRESENT=1\nNVIDIA_GPU_COUNT=1\n"
        "NVIDIA_GPU_0_NAME=NVIDIA RTX PRO 6000\nNVIDIA_GPU_0_MEMORY_MIB=98304\n"
        "AMD_GPU_COUNT=0\nINTEL_GAUDI_COUNT=0\n"
        "APPLE_PRESENT=1\nAPPLE_MODEL=Apple M5\n"
        "IB_PRESENT=0\nOS=Darwin\nARCH=arm64\n"
    )
    hw = build_host_hardware(parsed)
    vendors = {a.vendor for a in hw.accelerators}
    assert vendors == {"nvidia", "apple"}


def test_build_host_hardware_intel_gaudi():
    parsed = parse_fingerprint_output(
        "NVIDIA_GPU_COUNT=0\nAMD_GPU_COUNT=0\n"
        "INTEL_PRESENT=1\nINTEL_GAUDI_COUNT=8\n"
        + "".join("INTEL_GAUDI_%d_NAME=HL-225\nINTEL_GAUDI_%d_MEMORY_MIB=98304\n" % (i, i) for i in range(8))
        + "APPLE_PRESENT=0\nIB_PRESENT=1\nOS=Linux\nARCH=x86_64\n"
    )
    hw = build_host_hardware(parsed)
    assert hw.accelerators[0].vendor == "intel"
    assert hw.accelerators[0].count == 8
    assert "habana" in hw.accelerators[0].capabilities


def test_build_host_hardware_no_accelerator_returns_empty():
    """Probe returns no accelerator -> empty list, fingerprint still computable."""
    parsed = parse_fingerprint_output(
        "NVIDIA_GPU_COUNT=0\nAMD_GPU_COUNT=0\nINTEL_GAUDI_COUNT=0\nAPPLE_PRESENT=0\nIB_PRESENT=0\nOS=Linux\nARCH=x86_64\n"
    )
    hw = build_host_hardware(parsed)
    assert hw.accelerators == []
    assert hw.fingerprint  # deterministic hash of []


def test_build_host_hardware_heterogeneous_nvidia_counts_split():
    """Mixed NVIDIA models on one host stay as separate AcceleratorSpec groups."""
    parsed = parse_fingerprint_output(
        "NVIDIA_PRESENT=1\nNVIDIA_GPU_COUNT=4\n"
        "NVIDIA_GPU_0_NAME=NVIDIA H100\nNVIDIA_GPU_0_MEMORY_MIB=81920\n"
        "NVIDIA_GPU_1_NAME=NVIDIA H100\nNVIDIA_GPU_1_MEMORY_MIB=81920\n"
        "NVIDIA_GPU_2_NAME=NVIDIA A100\nNVIDIA_GPU_2_MEMORY_MIB=81920\n"
        "NVIDIA_GPU_3_NAME=NVIDIA A100\nNVIDIA_GPU_3_MEMORY_MIB=81920\n"
        "AMD_GPU_COUNT=0\nINTEL_GAUDI_COUNT=0\nAPPLE_PRESENT=0\nIB_PRESENT=0\nOS=Linux\nARCH=x86_64\n"
    )
    hw = build_host_hardware(parsed)
    models = [a.model for a in hw.accelerators]
    counts = [a.count for a in hw.accelerators]
    assert models == ["h100", "a100"]
    assert counts == [2, 2]


# --------------------------------------------------------------------------
# Fingerprint hash stability
# --------------------------------------------------------------------------


def test_compute_fingerprint_hash_is_deterministic():
    accels = [AcceleratorSpec(vendor="nvidia", model="gb10", memory_gb=121.0, capabilities=frozenset({"cuda"}))]
    h1 = compute_fingerprint_hash(accels)
    h2 = compute_fingerprint_hash(list(accels))  # different list identity, same content
    assert h1 == h2
    assert len(h1) == 16


def test_compute_fingerprint_hash_differs_by_count():
    one = [AcceleratorSpec(vendor="nvidia", model="h200", count=1)]
    eight = [AcceleratorSpec(vendor="nvidia", model="h200", count=8)]
    assert compute_fingerprint_hash(one) != compute_fingerprint_hash(eight)


def test_compute_fingerprint_hash_empty_is_stable():
    assert compute_fingerprint_hash([]) == compute_fingerprint_hash([])


# --------------------------------------------------------------------------
# fingerprint_host SSH wrapper (mocked)
# --------------------------------------------------------------------------


def test_fingerprint_host_success(monkeypatch):
    """fingerprint_host wires SSH stdout into the parser pipeline."""
    from sparkrun.core import fingerprint as fp_mod

    class _Result:
        success = True
        stdout = _probe_dgx_spark()
        stderr = ""

    def _fake_run(host, script, timeout=None, **kwargs):
        assert host == "test-host"
        return _Result()

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", _fake_run)
    hw = fp_mod.fingerprint_host("test-host")
    assert hw.accelerators[0].model == "gb10"
    assert hw.fingerprint


def test_fingerprint_host_failure_returns_empty_with_note(monkeypatch):
    """A failed probe surfaces a clear note rather than raising."""
    from sparkrun.core import fingerprint as fp_mod

    class _Result:
        success = False
        stdout = ""
        stderr = "ssh: connection refused"

    def _fake_run(host, script, timeout=None, **kwargs):
        return _Result()

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", _fake_run)
    hw = fp_mod.fingerprint_host("dead-host")
    assert hw.accelerators == []
    assert "fingerprint probe failed" in hw.notes
