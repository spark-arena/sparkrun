"""The RDMA shell halves, exercised under real bash.

Both scripts are run for real — ``rdma_probe.sh`` against a fixture sysfs tree
and ``rdma_perftest.sh`` with stub commands — because their whole job is shell
semantics (concurrent job control, exit-status propagation, output framing)
that no amount of Python mocking would exercise. Same approach as
``tests/test_mgmt_iface.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from sparkrun.orchestration.rdma import parse_device_facts, parse_framed_runs
from sparkrun.scripts import inject_shell_vars, read_script
from sparkrun.utils import parse_kv_output

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")

# name: (state, rate, link_layer, netdev)
_FIXTURE_DEVICES = {
    "rocep1s0f1": ("4: ACTIVE", "100 Gb/sec (4X EDR)", "InfiniBand", "enp1s0f1np1"),
    "roceP2p1s0f1": ("4: ACTIVE", "100 Gb/sec (4X EDR)", "InfiniBand", "enP2p1s0f1np1"),
    "rocep1s0f0": ("1: DOWN", "100 Gb/sec (4X EDR)", "InfiniBand", "enp1s0f0np0"),
}


@pytest.fixture(autouse=True)
def fixture_gid_command(tmp_path, monkeypatch):
    """The real show_gids scans host hardware even with fixture sysfs."""
    bin_dir = tmp_path / "gid-bin"
    bin_dir.mkdir()
    show_gids = bin_dir / "show_gids"
    show_gids.write_text("#!/bin/sh\nprintf 'fixture 1 3 192.0.2.1 v2\\n'\n")
    show_gids.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])


@pytest.fixture
def ib_sysfs(tmp_path):
    """Materialize a fixture ``/sys/class/infiniband`` tree."""
    root = tmp_path / "infiniband"
    for dev, (state, rate, link_layer, netdev) in _FIXTURE_DEVICES.items():
        port = root / dev / "ports" / "1"
        port.mkdir(parents=True)
        (port / "state").write_text(state)
        (port / "rate").write_text(rate)
        (port / "link_layer").write_text(link_layer)
        (root / dev / "device" / "net" / netdev).mkdir(parents=True)
    return root


def _run(script: str, env_extra: dict | None = None, timeout: int = 60):
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, **(env_extra or {})},
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# rdma_probe.sh
# ---------------------------------------------------------------------------


@needs_bash
def test_probe_reports_devices_from_sysfs(ib_sysfs):
    r = _run(read_script("rdma_probe.sh"), {"SPARKRUN_IB_SYSFS": str(ib_sysfs)})
    assert r.returncode == 0, r.stderr

    facts = parse_device_facts("h1", parse_kv_output(r.stdout))
    assert facts.complete
    assert {d.name for d in facts.devices} == set(_FIXTURE_DEVICES)
    assert facts.device("rocep1s0f1").active
    assert facts.device("rocep1s0f1").rate_gbps == 100.0
    assert facts.device("rocep1s0f1").netdev == "enp1s0f1np1"
    # A down port is reported, not hidden: "present but DOWN" is the finding.
    assert not facts.device("rocep1s0f0").active


@needs_bash
def test_probe_on_a_host_with_no_rdma_hardware(tmp_path):
    """No sysfs directory is a clean zero-device answer, not an error."""
    r = _run(read_script("rdma_probe.sh"), {"SPARKRUN_IB_SYSFS": str(tmp_path / "absent")})
    assert r.returncode == 0
    facts = parse_device_facts("h1", parse_kv_output(r.stdout))
    assert facts.complete
    assert facts.devices == []


@needs_bash
def test_probe_emits_the_completion_sentinel_last(ib_sysfs):
    """rc 0 does not prove the script finished; the sentinel does."""
    r = _run(read_script("rdma_probe.sh"), {"SPARKRUN_IB_SYSFS": str(ib_sysfs)})
    assert r.stdout.strip().splitlines()[-1] == "RDMA_COMPLETE=1"


@needs_bash
def test_probe_never_writes_to_stdout_except_key_values(ib_sysfs):
    r = _run(read_script("rdma_probe.sh"), {"SPARKRUN_IB_SYSFS": str(ib_sysfs)})
    for line in r.stdout.strip().splitlines():
        assert "=" in line, "non key=value line on stdout: %r" % line


# ---------------------------------------------------------------------------
# rdma_perftest.sh
# ---------------------------------------------------------------------------


def _runner(cmds: list[str], **vars_):
    script = read_script("rdma_perftest.sh")
    return inject_shell_vars(script, RDMA_CMDS="\n".join(cmds), **vars_)


@needs_bash
def test_runner_frames_each_command_separately():
    r = _run(_runner(["echo alpha", "echo beta"], RDMA_TIMEOUT="10"))
    assert r.returncode == 0, r.stderr
    runs = parse_framed_runs(r.stdout)
    assert runs == {0: ("alpha", 0), 1: ("beta", 0)}


@needs_bash
def test_runner_propagates_nonzero_exit_status():
    """A failed test must be distinguishable from one that produced no output."""
    r = _run(_runner(["exit 7"], RDMA_TIMEOUT="10"))
    assert parse_framed_runs(r.stdout)[0][1] == 7


@needs_bash
def test_runner_captures_stderr_into_the_frame():
    """perftest reports connection failures on stderr; those must survive."""
    r = _run(_runner(["echo oops >&2; exit 1"], RDMA_TIMEOUT="10"))
    output, rc = parse_framed_runs(r.stdout)[0]
    assert rc == 1
    assert "oops" in output


@needs_bash
def test_runner_actually_runs_commands_concurrently():
    """Concurrency is the feature: two 2s sleeps must not take 4s.

    On DGX Spark the aggregate bandwidth figure only appears when both RDMA
    devices are driven at once, so a runner that serialized would silently
    halve the headline number.
    """
    import time

    t0 = time.monotonic()
    r = _run(_runner(["sleep 2; echo a", "sleep 2; echo b"], RDMA_TIMEOUT="20"), timeout=60)
    elapsed = time.monotonic() - t0

    assert r.returncode == 0, r.stderr
    assert parse_framed_runs(r.stdout) == {0: ("a", 0), 1: ("b", 0)}
    assert elapsed < 3.5, "commands ran serially (%.1fs)" % elapsed


@needs_bash
def test_runner_enforces_its_timeout():
    """A server whose client never arrives must die on its own."""
    import time

    t0 = time.monotonic()
    r = _run(_runner(["sleep 30"], RDMA_TIMEOUT="2"), timeout=60)
    elapsed = time.monotonic() - t0

    assert elapsed < 20, "timeout did not fire (%.1fs)" % elapsed
    assert parse_framed_runs(r.stdout)[0][1] != 0


@needs_bash
def test_runner_retries_a_failing_command(tmp_path):
    """The client's retry IS the rendezvous — perftest has no wait-for-peer."""
    marker = tmp_path / "attempts"
    cmd = "echo x >> %s; test $(wc -l < %s) -ge 3" % (marker, marker)

    # Assert actual attempts and exit propagation, without two 2s backoffs.
    r = _run("sleep() { :; };\n" + _runner([cmd], RDMA_TIMEOUT="10", RDMA_RETRIES="4"), timeout=10)
    assert parse_framed_runs(r.stdout)[0][1] == 0
    assert marker.read_text().count("x") == 3


@needs_bash
def test_runner_gives_up_after_the_retry_budget(tmp_path):
    marker = tmp_path / "attempts"
    cmd = "echo x >> %s; false" % marker

    r = _run("sleep() { :; };\n" + _runner([cmd], RDMA_TIMEOUT="5", RDMA_RETRIES="2"), timeout=10)
    assert parse_framed_runs(r.stdout)[0][1] != 0
    # initial attempt + 2 retries
    assert marker.read_text().count("x") == 3


@needs_bash
def test_runner_honours_the_pre_sleep_head_start():
    import time

    t0 = time.monotonic()
    r = _run(_runner(["echo go"], RDMA_TIMEOUT="10", RDMA_PRE_SLEEP="2"), timeout=60)
    assert time.monotonic() - t0 >= 1.8
    assert parse_framed_runs(r.stdout)[0][0] == "go"


@needs_bash
def test_runner_survives_output_that_looks_like_a_perftest_table():
    """Real payload passes through the framing untouched."""
    row = " 65536      20000            111.72             111.71             0.213070"
    r = _run(_runner(["printf '%s\\n' %s" % ("%s", "'" + row + "'")], RDMA_TIMEOUT="10"))
    from sparkrun.orchestration.rdma import parse_perftest_bw

    output, rc = parse_framed_runs(r.stdout)[0]
    assert rc == 0
    assert parse_perftest_bw(output).avg_gbps == pytest.approx(111.71)


@needs_bash
def test_runner_emits_the_completion_sentinel():
    r = _run(_runner(["true"], RDMA_TIMEOUT="10"))
    assert "RDMA_COMPLETE=1" in r.stdout


# ---------------------------------------------------------------------------
# Container start script
# ---------------------------------------------------------------------------


def _render_start(gpu_opts="--gpus all", mounts=""):
    from sparkrun.api.setup._rdma import _START_CONTAINER, CONTAINER_NAME
    from sparkrun.utils.shell import quote

    return _START_CONTAINER.format(name=CONTAINER_NAME, image=quote("img:tag"), gpu_opts=gpu_opts, mounts=mounts)


@pytest.fixture
def ib_dev_dir(tmp_path):
    """A /dev/infiniband lookalike: real char devices beside two directories.

    ``by-ibdev`` and ``by-path`` are directories of symlinks that the kernel
    creates alongside the device nodes — the exact thing that broke this.
    Char devices are faked by symlinking to /dev/null, which ``[ -c ]``
    follows, since tests cannot mknod.
    """
    d = tmp_path / "infiniband"
    (d / "by-ibdev").mkdir(parents=True)
    (d / "by-path").mkdir(parents=True)
    for node in ("uverbs0", "uverbs1", "rdma_cm"):
        (d / node).symlink_to("/dev/null")
    (d / "README").write_text("a regular file")
    return d


@needs_bash
def test_start_script_passes_only_character_devices(ib_dev_dir, tmp_path):
    """`--device=<a directory>` fails the whole `docker run`.

    /dev/infiniband contains `by-ibdev` and `by-path` directories next to the
    device nodes. Testing `-e` (exists) instead of `-c` (character device)
    passed a directory to --device and docker rejected the entire run with
    "not a device node", taking every real device with it — observed live on
    a DGX Spark.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "docker"
    # To stderr: the script sends `docker run` stdout to /dev/null, so args
    # echoed on stdout would be discarded. `docker rm` has both streams
    # suppressed, so only the `run` invocation is observed here.
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >&2\nexit 0\n')
    stub.chmod(0o755)

    r = _run(
        _render_start(),
        {"PATH": "%s:%s" % (bindir, os.environ.get("PATH", "/usr/bin:/bin")), "SPARKRUN_IB_DEV_DIR": str(ib_dev_dir)},
    )
    assert r.returncode == 0, r.stderr

    devices = [line.split("=", 1)[1] for line in r.stderr.splitlines() if line.startswith("--device=")]
    assert sorted(os.path.basename(d) for d in devices) == ["rdma_cm", "uverbs0", "uverbs1"]
    for excluded in ("by-ibdev", "by-path", "README"):
        assert not any(excluded in d for d in devices), "%s must not be passed to --device" % excluded


@needs_bash
def test_start_script_warns_when_no_devices_exist(tmp_path):
    """A host with no RDMA nodes still starts the container, with a warning."""
    empty = tmp_path / "empty"
    empty.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "docker").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bindir / "docker").chmod(0o755)

    r = _run(
        _render_start(),
        {"PATH": "%s:%s" % (bindir, os.environ.get("PATH", "/usr/bin:/bin")), "SPARKRUN_IB_DEV_DIR": str(empty)},
    )
    assert r.returncode == 0
    assert "no character devices" in r.stderr
    assert "RDMA_CONTAINER_STARTED=1" in r.stdout


@needs_bash
def test_start_script_reports_docker_failure_rc(tmp_path):
    """A failed `docker run` must exit non-zero, not silently continue."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "docker").write_text('#!/usr/bin/env bash\ncase "$1" in rm) exit 0;; esac\necho "boom" >&2\nexit 125\n')
    (bindir / "docker").chmod(0o755)

    r = _run(
        _render_start(),
        {"PATH": "%s:%s" % (bindir, os.environ.get("PATH", "/usr/bin:/bin")), "SPARKRUN_IB_DEV_DIR": str(tmp_path)},
    )
    assert r.returncode == 125
    assert "RDMA_CONTAINER_STARTED=1" not in r.stdout


# ---------------------------------------------------------------------------
# Formatting contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rdma_probe.sh", "rdma_perftest.sh"])
def test_scripts_are_valid_bash(name):
    assert subprocess.run(["bash", "-n"], input=read_script(name), text=True, capture_output=True).returncode == 0


@pytest.mark.parametrize("name", ["rdma_probe.sh", "rdma_perftest.sh"])
def test_scripts_survive_inject_shell_vars(name):
    """These take parameters by injection, never through str.format()."""
    injected = inject_shell_vars(read_script(name), RDMA_CMDS="echo hi", RDMA_TIMEOUT="5")
    assert subprocess.run(["bash", "-n"], input=injected, text=True, capture_output=True).returncode == 0
