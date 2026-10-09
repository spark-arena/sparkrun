"""Foreground gateway supervision across managed restarts, stops and crashes.

``proxy start --foreground`` used to ``wait()`` on its own child only.  The
auto-discover daemon replaces the gateway on every model change, so the first
change returned that wait, and the cleanup that followed erased the
replacement's state.  These tests drive real (sleeping) processes through the
state-file protocol the daemon and ``proxy stop`` use.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from sparkrun.api.proxy._ops import _foreground_exit_status
from sparkrun.proxy._supervisor import SUPERVISOR_ENV, exit_status, supervisor_from_env
from sparkrun.proxy.engine import ProxyEngine


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


def _exiter(code: int) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import sys, time; time.sleep(0.3); sys.exit(%d)" % code])


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.delenv(SUPERVISOR_ENV, raising=False)
    return ProxyEngine(state_dir=tmp_path / "proxy")


@pytest.fixture
def procs():
    """Track spawned processes so a failing test cannot leak sleepers."""
    spawned: list[subprocess.Popen] = []
    yield spawned
    for proc in spawned:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


class _Supervised:
    """Run ``supervise_foreground`` on a worker thread and collect its status."""

    def __init__(self, engine: ProxyEngine, proc: subprocess.Popen) -> None:
        self.status: int | None = None
        self._thread = threading.Thread(target=self._run, args=(engine, proc), daemon=True)
        self._thread.start()

    def _run(self, engine, proc):
        self.status = engine.supervise_foreground(proc)

    def join(self, timeout: float = 30.0) -> int | None:
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "supervisor did not return"
        return self.status

    @property
    def running(self) -> bool:
        return self._thread.is_alive()


def _managed_restart(engine: ProxyEngine, old: subprocess.Popen, procs: list, *, delay: float = 0.0) -> subprocess.Popen:
    """Replay ``ProxyEngine._restart_proxy``'s state protocol by hand."""
    engine._mark_state(restarting=True)
    old.send_signal(signal.SIGTERM)
    time.sleep(delay)
    new = _sleeper()
    procs.append(new)
    engine._save_state(new.pid)
    return new


def test_follows_managed_restart_then_stops_cleanly(engine, procs):
    first = _sleeper()
    procs.append(first)
    engine._save_state(first.pid)
    sup = _Supervised(engine, first)

    second = _managed_restart(ProxyEngine(state_dir=engine.state_dir), first, procs)
    time.sleep(1.0)
    assert sup.running, "supervisor exited on a managed restart"
    assert engine.current_pid() == second.pid

    # A second restart, of a process that is not the supervisor's child.
    third = _managed_restart(ProxyEngine(state_dir=engine.state_dir), second, procs)
    time.sleep(1.0)
    assert sup.running
    assert engine.current_pid() == third.pid

    # `proxy stop` from another process.
    assert ProxyEngine(state_dir=engine.state_dir).stop() is True
    assert sup.join() == 0
    assert engine.get_state() is None


def test_restart_window_is_not_a_crash(engine, procs):
    """Old PID dead, replacement not yet recorded: keep waiting."""
    first = _sleeper()
    procs.append(first)
    engine._save_state(first.pid)
    sup = _Supervised(engine, first)

    second = _managed_restart(ProxyEngine(state_dir=engine.state_dir), first, procs, delay=1.5)
    time.sleep(0.5)
    assert sup.running
    second.kill()
    second.wait()
    # Killed with no intent recorded: a crash. The replacement is not the
    # supervisor's child in production, so its status is unknown: 1.
    assert sup.join() == 1


def test_failed_restart_is_a_crash(engine, procs):
    """``_restart_proxy`` records ``restart_failed`` when the replacement fails to start."""
    first = _sleeper()
    procs.append(first)
    engine._save_state(first.pid)
    sup = _Supervised(engine, first)

    other = ProxyEngine(state_dir=engine.state_dir)
    other._mark_state(restarting=True)
    first.send_signal(signal.SIGTERM)
    time.sleep(0.5)
    other._mark_state(restarting=False, restart_failed=True)
    # A crash, carrying the old child's own status (it died of the SIGTERM).
    assert sup.join() == 128 + signal.SIGTERM


def test_stop_during_restart_is_a_stop(engine, procs):
    """A stop that clears the record mid-restart wins: exit 0, no crash."""
    first = _sleeper()
    procs.append(first)
    engine._save_state(first.pid)
    sup = _Supervised(engine, first)

    other = ProxyEngine(state_dir=engine.state_dir)
    other._mark_state(restarting=True)
    first.send_signal(signal.SIGTERM)
    time.sleep(0.5)
    other._clear_state()
    assert sup.join() == 0


def test_stop_of_own_child_returns_zero(engine, procs):
    gateway = _sleeper()
    procs.append(gateway)
    engine._save_state(gateway.pid)
    sup = _Supervised(engine, gateway)
    time.sleep(0.2)

    other = ProxyEngine(state_dir=engine.state_dir)
    other.stop()
    assert sup.join() == 0
    assert engine.get_state() is None


@pytest.mark.parametrize("code", [1, 3])
def test_crash_returns_gateway_status_and_clears_state(engine, procs, code):
    gateway = _exiter(code)
    procs.append(gateway)
    engine._save_state(gateway.pid)
    assert engine.supervise_foreground(gateway) == code
    assert engine.get_state() is None


def test_unrequested_clean_exit_is_still_a_failure(engine, procs):
    """A gateway that exits 0 on its own was not asked to: keep it restartable."""
    gateway = _exiter(0)
    procs.append(gateway)
    engine._save_state(gateway.pid)
    assert engine.supervise_foreground(gateway) == 1


def test_crash_stops_autodiscover(engine, procs, monkeypatch):
    calls = []
    monkeypatch.setattr(engine, "stop_autodiscover", lambda: calls.append("autodiscover"))
    gateway = _exiter(2)
    procs.append(gateway)
    engine._save_state(gateway.pid)
    engine.supervise_foreground(gateway)
    assert calls == ["autodiscover"]


def test_interrupt_stops_daemon_before_gateway(engine, procs, monkeypatch):
    """SIGTERM to the supervisor (systemctl stop) arrives as KeyboardInterrupt."""
    gateway = _sleeper()
    procs.append(gateway)
    engine._save_state(gateway.pid)

    order = []
    monkeypatch.setattr(engine, "stop_autodiscover", lambda: order.append("autodiscover"))
    real_terminate = engine._terminate_gateway

    def terminate(pid, child):
        order.append("gateway")
        real_terminate(pid, child)

    monkeypatch.setattr(engine, "_terminate_gateway", terminate)
    # A real signal: interrupt_main() cannot break a blocking waitpid().
    timer = threading.Timer(0.5, os.kill, args=(os.getpid(), signal.SIGINT))
    timer.start()
    try:
        status = engine.supervise_foreground(gateway)
    finally:
        timer.cancel()

    assert status == 0
    assert order == ["autodiscover", "gateway"]
    assert gateway.poll() is not None
    assert engine.get_state() is None


def test_foreign_start_after_stop_is_left_alone(engine, procs, monkeypatch):
    """State recorded by a start we did not supervise is not ours to clear."""
    monkeypatch.setenv(SUPERVISOR_ENV, "systemd:system:sparkrun-proxy.service")
    gateway = _sleeper()
    procs.append(gateway)
    engine._save_state(gateway.pid)
    sup = _Supervised(engine, gateway)
    time.sleep(0.2)

    # Someone else's proxy, started outside the unit, has taken over the state
    # by the time ours exits.
    monkeypatch.delenv(SUPERVISOR_ENV)
    foreign = _sleeper()
    procs.append(foreign)
    ProxyEngine(state_dir=engine.state_dir)._save_state(foreign.pid)
    gateway.send_signal(signal.SIGTERM)

    assert sup.join() == 0
    assert engine.current_pid() == foreign.pid
    assert foreign.poll() is None


def test_stop_records_intent_before_signalling(engine, procs):
    gateway = _sleeper()
    procs.append(gateway)
    engine._save_state(gateway.pid)
    other = ProxyEngine(state_dir=engine.state_dir)
    other.stop()
    state = engine.get_state()
    # Not our child, so stop() keeps the record while shutdown is pending.
    if state is not None:
        assert state["stop_requested"] is True
    gateway.wait(timeout=10)


def test_restart_proxy_marks_and_clears_restarting(engine, procs, monkeypatch):
    old = _sleeper()
    procs.append(old)
    engine._save_state(old.pid)
    seen = {}

    def launch(cmd, env):
        seen["state"] = engine.get_state()
        new = _sleeper()
        procs.append(new)
        return new.pid

    monkeypatch.setattr(engine, "_build_command", lambda config_path=None: ["true"])
    monkeypatch.setattr(engine, "_launch_background", launch)
    new_pid = engine._restart_proxy()

    assert seen["state"]["restarting"] is True
    state = engine.get_state()
    assert state["pid"] == new_pid
    assert "restarting" not in state


def test_supervisor_record_follows_env(engine, procs, monkeypatch):
    monkeypatch.setenv(SUPERVISOR_ENV, "systemd:user:sparkrun-proxy.service")
    engine._save_state(12345)
    assert engine.get_state()["supervisor"] == {"kind": "systemd", "scope": "user", "unit": "sparkrun-proxy.service"}
    monkeypatch.delenv(SUPERVISOR_ENV)
    engine._save_state(12345)
    assert "supervisor" not in engine.get_state()


@pytest.mark.parametrize(
    "value, expected",
    [
        ("systemd:system:sparkrun-proxy.service", {"kind": "systemd", "scope": "system", "unit": "sparkrun-proxy.service"}),
        ("", None),
        ("systemd:sparkrun-proxy.service", None),
        ("systemd::unit", None),
    ],
)
def test_supervisor_from_env(value, expected):
    assert supervisor_from_env({SUPERVISOR_ENV: value}) == expected


@pytest.mark.parametrize("rc, expected", [(0, 0), (3, 3), (-signal.SIGKILL, 137), (None, 1)])
def test_exit_status(rc, expected):
    assert exit_status(rc) == expected


@pytest.mark.parametrize("rc, expected", [(-signal.SIGTERM, 0), (-signal.SIGKILL, 137), (0, 0), (2, 2)])
def test_foreground_exit_status_treats_sigterm_as_requested(rc, expected):
    assert _foreground_exit_status(rc) == expected


# -- Review follow-ups: stale boots, stop/restart races, signal deferral -------


def test_record_from_a_previous_boot_names_no_process(engine, procs, monkeypatch):
    """A reused PID after a reboot must not read as our running proxy."""
    alive = _sleeper()
    procs.append(alive)
    engine._save_state(alive.pid)
    assert engine.is_running()
    monkeypatch.setattr("sparkrun.proxy._supervisor._boot_id", lambda: "another-boot")
    assert engine.current_pid() is None
    assert not engine.is_running()
    assert engine.stop() is False  # nothing to signal: the stranger is left alone
    assert alive.poll() is None


def test_record_without_boot_id_is_still_honoured(engine, procs, monkeypatch):
    """Records written before boot_id existed keep their old meaning."""
    alive = _sleeper()
    procs.append(alive)
    monkeypatch.setattr("sparkrun.proxy._supervisor._boot_id", lambda: None)
    engine._save_state(alive.pid)
    monkeypatch.setattr("sparkrun.proxy._supervisor._boot_id", lambda: "now")
    assert engine.is_running()


def _restart_harness(engine, procs, monkeypatch, *, during_launch=None):
    old = _sleeper()
    procs.append(old)
    engine._save_state(old.pid)
    spawned = []

    def launch(cmd, env):
        if during_launch:
            during_launch()
        new = _sleeper()
        procs.append(new)
        spawned.append(new)
        return new.pid

    monkeypatch.setattr(engine, "_build_command", lambda config_path=None: ["true"])
    monkeypatch.setattr(engine, "_launch_background", launch)
    return old, spawned


def test_stop_during_launch_terminates_the_replacement(engine, procs, monkeypatch):
    other = ProxyEngine(state_dir=engine.state_dir)
    old, spawned = _restart_harness(engine, procs, monkeypatch, during_launch=lambda: other._mark_state(stop_requested=True))
    assert engine._restart_proxy() is None
    (new,) = spawned
    new.wait(timeout=10)
    assert engine.get_state()["pid"] == old.pid  # never replaced by the stopped one


def test_stop_before_launch_spawns_nothing(engine, procs, monkeypatch):
    old, spawned = _restart_harness(engine, procs, monkeypatch)
    real_await = engine._await_exit

    def await_then_stop(pid, timeout):
        ok = real_await(pid, timeout)
        ProxyEngine(state_dir=engine.state_dir)._clear_state()
        return ok

    monkeypatch.setattr(engine, "_await_exit", await_then_stop)
    assert engine._restart_proxy() is None
    assert spawned == []


def test_failed_launch_records_restart_failed(engine, procs, monkeypatch):
    old, _ = _restart_harness(engine, procs, monkeypatch)
    monkeypatch.setattr(engine, "_launch_background", lambda cmd, env: None)
    assert engine._restart_proxy() is None
    state = engine.get_state()
    assert state["restart_failed"] is True and state["restarting"] is False
    assert not engine.is_running()


def test_restart_keeps_the_supervisor_record(engine, procs, monkeypatch):
    """A restart from a shell (no supervisor env) must not drop the unit's record."""
    record = {"kind": "systemd", "scope": "system", "unit": "sparkrun-proxy.service"}
    monkeypatch.setenv(SUPERVISOR_ENV, "systemd:system:sparkrun-proxy.service")
    old, _ = _restart_harness(engine, procs, monkeypatch)
    monkeypatch.delenv(SUPERVISOR_ENV)
    # No such unit is installed for us, so the direct path runs.
    new_pid = engine._restart_proxy()
    assert engine.get_state()["pid"] == new_pid
    assert engine.get_state()["supervisor"] == record


def test_restart_outside_the_unit_goes_through_it(engine, procs, monkeypatch):
    from sparkrun.proxy import _systemd

    monkeypatch.setenv(SUPERVISOR_ENV, "systemd:system:sparkrun-proxy.service")
    old, spawned = _restart_harness(engine, procs, monkeypatch)
    monkeypatch.delenv(SUPERVISOR_ENV)
    spec = _systemd.UnitSpec("system", "sparkrun-proxy.service", _systemd.SYSTEM_UNIT_DIR / "sparkrun-proxy.service", "alice")
    monkeypatch.setattr(_systemd, "unit_for_record", lambda record: spec)
    replacement = _sleeper()
    procs.append(replacement)

    def control(unit, action):
        assert (unit, action) == (spec, "restart")
        old.terminate()
        ProxyEngine(state_dir=engine.state_dir)._save_state(replacement.pid)
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(_systemd, "control", control)
    assert engine._restart_proxy() == replacement.pid
    assert spawned == []


def test_replacement_dying_on_arrival_is_a_crash(engine, procs):
    first = _sleeper()
    procs.append(first)
    engine._save_state(first.pid)
    sup = _Supervised(engine, first)
    other = ProxyEngine(state_dir=engine.state_dir)
    other._mark_state(restarting=True)
    first.send_signal(signal.SIGTERM)
    dead = _exiter(0)
    procs.append(dead)
    dead.wait()
    other._save_state(dead.pid)
    assert sup.join() != 0


def test_failed_stop_signal_clears_intent(engine, procs, monkeypatch):
    gateway = _sleeper()
    procs.append(gateway)
    engine._save_state(gateway.pid)
    real_kill = os.kill

    def deny(pid, sig):
        if pid == gateway.pid and sig == signal.SIGTERM:
            raise PermissionError
        return real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", deny)
    assert engine.stop() is False
    assert engine.get_state()["stop_requested"] is False


def test_cleanup_ignores_repeated_signals():
    from sparkrun.proxy._supervisor import _signals_deferred

    before = signal.getsignal(signal.SIGINT)
    with _signals_deferred():
        os.kill(os.getpid(), signal.SIGINT)  # would raise KeyboardInterrupt otherwise
        time.sleep(0.05)
    assert signal.getsignal(signal.SIGINT) is before
