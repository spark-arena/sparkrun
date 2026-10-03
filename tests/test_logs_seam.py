"""Tests for the log seam: LogSource → read_logs_cmd → read_log_sources → api.logs.

The seam replaced an ``api.logs`` that hardcoded ``{cid}_solo`` / ``{cid}_node_0``
and read them with ``docker logs``.  Both halves of that were wrong:

* Ray runtimes name their head ``{cid}_head``, so the container didn't exist.
* Every sleep-infinity + exec runtime writes serve output to a file *inside*
  the container while PID 1 is ``sleep infinity`` — ``docker logs`` shows PID
  1's stdout, i.e. nothing.  ``scripts/exec_serve_detached.sh`` documents this
  at the point of the redirect.

These tests pin both, plus the ordering contract and the composition.
"""

from __future__ import annotations

from unittest import mock

import pytest

import sparkrun.api as api
from sparkrun.core.log_source import (
    MODE_FILE,
    MODE_STDOUT,
    SCOPE_ALL,
    SERVE_LOG_PATH,
    LogSource,
)
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.logs import build_read_command, read_log_sources


@pytest.fixture(autouse=True)
def _skip_logs_liveness_precheck(monkeypatch):
    """Keep ``api.logs``' liveness precheck off the real network.

    The precheck sweeps the job's hosts through ``executor.query_status`` —
    a real SSH connect-timeout against every fabricated host here. These
    tests assert source derivation and reader plumbing; the precheck's own
    decision tree has dedicated coverage (``test_api_run_stop_logs``). An
    *empty* snapshot leaves every host inconclusive, which is the precheck's
    documented skip path — nothing this file asserts changes.
    """
    from sparkrun.core.cluster_status import ClusterStatus

    monkeypatch.setattr(DockerExecutor, "query_status", lambda self, hosts, **kw: ClusterStatus(hosts=(), executor="docker"))


# --------------------------------------------------------------------------
# Executor: how a source is read on the docker substrate
# --------------------------------------------------------------------------


class TestDockerReadLogsCmd:
    def test_file_source_execs_into_the_container(self):
        """A serve-log source must be read with ``docker exec … tail``.

        ``docker logs`` cannot see it: PID 1 is ``sleep infinity``.
        """
        source = LogSource(host="h1", container="cid_head", mode=MODE_FILE, path=SERVE_LOG_PATH)
        cmd = DockerExecutor().read_logs_cmd(source, follow=True, tail=100)

        assert cmd == "docker exec cid_head tail -F -n 100 /tmp/sparkrun_serve.log"

    def test_file_source_without_tail_reads_whole_file(self):
        source = LogSource(host="h1", container="cid_solo", mode=MODE_FILE)
        cmd = DockerExecutor().read_logs_cmd(source, follow=False, tail=None)

        assert cmd == "docker exec cid_solo tail -n +1 /tmp/sparkrun_serve.log"

    def test_follow_uses_capital_f_to_await_a_missing_file(self):
        """``-F`` (not ``-f``): the serve log is created by the exec'd process
        slightly after the container starts, so ``-f`` can race and error."""
        source = LogSource(host="h1", container="cid_solo", mode=MODE_FILE)
        assert " -F " in DockerExecutor().read_logs_cmd(source, follow=True, tail=None)

    def test_stdout_source_uses_docker_logs(self):
        """Ray workers / TRT-LLM cluster genuinely put output on container stdout."""
        source = LogSource(host="h1", container="cid_worker", mode=MODE_STDOUT, path=None)
        cmd = DockerExecutor().read_logs_cmd(source, follow=True, tail=50)

        assert cmd == "docker logs -f --tail 50 cid_worker"

    def test_custom_path_is_honoured(self):
        source = LogSource(host="h1", container="c", mode=MODE_FILE, path="/var/log/custom.log")
        assert "/var/log/custom.log" in DockerExecutor().read_logs_cmd(source)

    def test_base_default_ignores_file_indirection(self):
        """Substrates without in-container filesystem indirection read their own
        stream — the ``local`` executor already tails the host logfile its
        ``run_cmd`` redirects to, and k8s uses ``kubectl logs``."""
        from sparkrun.orchestration.executors.local import LocalExecutor

        source = LogSource(host="h1", container="cid_solo", mode=MODE_FILE)
        cmd = LocalExecutor().read_logs_cmd(source, follow=True, tail=10)

        assert "docker exec" not in cmd
        assert cmd.startswith("tail")


# --------------------------------------------------------------------------
# Dispatch: local vs SSH
# --------------------------------------------------------------------------


class TestBuildReadCommand:
    def test_remote_host_goes_through_ssh(self):
        source = LogSource(host="10.0.0.5", container="c", mode=MODE_FILE)
        cmd = build_read_command(DockerExecutor(), source, ssh_kwargs={"ssh_user": "nick"})

        assert cmd[0] == "ssh"
        assert "nick@10.0.0.5" in cmd
        assert cmd[-2] == "-c"

    def test_local_host_skips_ssh(self):
        """Reading a workload on the control machine shouldn't round-trip through
        sshd — which also means it works when SSH-to-self isn't configured."""
        source = LogSource(host="localhost", container="c", mode=MODE_FILE)
        cmd = build_read_command(DockerExecutor(), source, ssh_kwargs={})

        assert cmd[0] == "bash"

    def test_remote_command_survives_ssh_argv_flattening(self):
        """The command must still be one word after the remote shell re-splits it.

        `ssh` does not preserve argv: it joins its trailing arguments into a
        single string and the remote login shell splits that on whitespace.
        An unquoted `["bash", "-c", command]` therefore arrives as
        `bash -c docker exec … tail …`, where `bash -c` consumes only
        `docker` and the rest become $0, $1, … — the remote runs a bare
        `docker` and prints its help instead of the logs.

        This models that round-trip rather than asserting on the local argv,
        because the local argv looked perfectly correct while the feature was
        completely broken over SSH.
        """
        import shlex

        source = LogSource(host="10.0.0.5", container="c", mode=MODE_FILE, path="/tmp/serve.log")
        expected = DockerExecutor().read_logs_cmd(source, follow=True, tail=100)

        cmd = build_read_command(DockerExecutor(), source, follow=True, tail=100, ssh_kwargs={})

        # What sshd hands the remote shell, and what that shell makes of it.
        remote_line = " ".join(cmd[cmd.index("bash") :])
        assert shlex.split(remote_line) == ["bash", "-c", expected]

    def test_local_command_is_not_double_quoted(self):
        """Locally the argv reaches execve untouched, so quoting would break it.

        The two branches must quote differently; a shared "just quote it"
        would turn the local path into `bash -c "'docker exec …'"`.
        """
        source = LogSource(host="localhost", container="c", mode=MODE_FILE, path="/tmp/serve.log")
        expected = DockerExecutor().read_logs_cmd(source, follow=True, tail=100)

        cmd = build_read_command(DockerExecutor(), source, follow=True, tail=100, ssh_kwargs={})
        assert cmd == ["bash", "-c", expected]

    def test_remote_stdout_mode_also_survives(self):
        """The docker-logs (non-file) read path goes through the same quoting."""
        import shlex

        source = LogSource(host="10.0.0.5", container="c", mode=MODE_STDOUT)
        expected = DockerExecutor().read_logs_cmd(source, follow=False, tail=50)

        cmd = build_read_command(DockerExecutor(), source, follow=False, tail=50, ssh_kwargs={})
        remote_line = " ".join(cmd[cmd.index("bash") :])
        assert shlex.split(remote_line) == ["bash", "-c", expected]


# --------------------------------------------------------------------------
# Reader: ordering contract
# --------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, lines):
        from io import StringIO

        self.stdout = StringIO("".join(lines))
        self._done = False

    def poll(self):
        return 0

    def terminate(self):
        self._done = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._done = True


def _fake_spawn(mapping):
    """Return a _spawn stub that keys canned output off the container name."""

    def _spawn(cmd):
        joined = " ".join(cmd)
        for container, lines in mapping.items():
            if container in joined:
                return _FakeProc(lines)
        return _FakeProc([])

    return _spawn


class TestReadOrdering:
    SOURCES = [
        LogSource(host="h1", container="c_node_0", role="node_0", rank=0, mode=MODE_FILE),
        LogSource(host="h2", container="c_node_1", role="node_1", rank=1, mode=MODE_FILE),
    ]

    def test_non_follow_is_rank_grouped(self, monkeypatch):
        """Dumps are grouped head-first, then workers by rank — each source in full.

        Cross-host time interleaving isn't possible here: the serve log has no
        per-line capture timestamps, and a dump delivers everything at once so
        arrival order carries no information either.
        """
        monkeypatch.setattr(
            "sparkrun.orchestration.logs._spawn",
            _fake_spawn({"c_node_0": ["h0-a\n", "h0-b\n"], "c_node_1": ["h1-a\n", "h1-b\n"]}),
        )

        lines = list(read_log_sources(DockerExecutor(), self.SOURCES, follow=False))

        assert [line.text for line in lines] == ["h0-a", "h0-b", "h1-a", "h1-b"]
        assert [line.host for line in lines] == ["h1", "h1", "h2", "h2"]

    def test_non_follow_lines_carry_role_and_rank(self, monkeypatch):
        monkeypatch.setattr("sparkrun.orchestration.logs._spawn", _fake_spawn({"c_node_1": ["only\n"]}))

        lines = list(read_log_sources(DockerExecutor(), self.SOURCES, follow=False))

        (line,) = lines
        assert (line.role, line.rank, line.container) == ("node_1", 1, "c_node_1")

    def test_non_follow_has_no_timestamps(self, monkeypatch):
        """Nothing to stamp: there is no capture time and arrival time is meaningless."""
        monkeypatch.setattr("sparkrun.orchestration.logs._spawn", _fake_spawn({"c_node_0": ["x\n"]}))

        (line,) = list(read_log_sources(DockerExecutor(), self.SOURCES[:1], follow=False))
        assert line.timestamp is None

    def test_follow_interleaves_and_stamps_on_arrival(self, monkeypatch):
        """Following yields lines as they arrive — time order for a live stream."""
        monkeypatch.setattr(
            "sparkrun.orchestration.logs._spawn",
            _fake_spawn({"c_node_0": ["a\n", "b\n"], "c_node_1": ["c\n", "d\n"]}),
        )

        lines = list(read_log_sources(DockerExecutor(), self.SOURCES, follow=True))

        # Both sources are represented; per-source order is preserved even
        # though the cross-source interleaving is arrival-dependent.
        assert sorted(line.text for line in lines) == ["a", "b", "c", "d"]
        by_host = {}
        for line in lines:
            by_host.setdefault(line.host, []).append(line.text)
        assert by_host["h1"] == ["a", "b"]
        assert by_host["h2"] == ["c", "d"]
        assert all(line.timestamp is not None for line in lines)
        # Arrival stamps are monotonic in yield order — that IS the ordering.
        stamps = [line.timestamp for line in lines]
        assert stamps == sorted(stamps)

    def test_follow_survives_one_source_failing(self, monkeypatch):
        """A dead worker must not take the head's stream down with it."""

        def _spawn(cmd):
            if "c_node_1" in " ".join(cmd):
                raise OSError("ssh: connect failed")
            return _FakeProc(["alive\n"])

        monkeypatch.setattr("sparkrun.orchestration.logs._spawn", _spawn)

        lines = list(read_log_sources(DockerExecutor(), self.SOURCES, follow=True))
        assert [line.text for line in lines] == ["alive"]

    def test_no_sources_yields_nothing(self):
        assert list(read_log_sources(DockerExecutor(), [], follow=True)) == []

    def test_follow_terminates_readers_when_consumer_stops(self, monkeypatch):
        """Closing the generator early must reap the reader subprocesses."""
        procs = []

        def _spawn(cmd):
            proc = _FakeProc(["x\n"] * 100)
            proc.poll = lambda: None  # pretend still running so terminate() is exercised
            procs.append(proc)
            return proc

        monkeypatch.setattr("sparkrun.orchestration.logs._spawn", _spawn)
        monkeypatch.setattr("sparkrun.orchestration.logs._terminate", lambda proc: proc.terminate())

        stream = read_log_sources(DockerExecutor(), self.SOURCES, follow=True)
        next(stream)
        stream.close()

        assert procs and all(p._done for p in procs)


# --------------------------------------------------------------------------
# api.logs composition
# --------------------------------------------------------------------------


def _write_meta(tmp_path, cluster_id, runtime_name, hosts):
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import save_job_metadata

    recipe = Recipe({"sparkrun_version": "2", "runtime": runtime_name, "model": "test/m"})
    save_job_metadata(cluster_id, recipe, list(hosts), cache_dir=str(tmp_path))
    return recipe


class TestApiLogsComposition:
    CID = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"

    def _captured_sources(self, tmp_path, runtime_name, hosts, **kwargs):
        captured = {}

        def _capture(executor, sources, **kw):
            captured["sources"] = list(sources)
            captured["kwargs"] = kw
            return iter(())

        _write_meta(tmp_path, self.CID, runtime_name, hosts)
        with mock.patch("sparkrun.orchestration.logs.read_log_sources", _capture):
            list(api.logs(self.CID, hosts=tuple(hosts), cache_dir=str(tmp_path), **kwargs))
        return captured

    def test_ray_job_reads_the_head_container_not_node_0(self, tmp_path):
        """The headline bug: a Ray job's head is ``{cid}_head``.

        The old implementation asked for ``{cid}_node_0`` — a container that
        does not exist — so this returned "No such container".
        """
        captured = self._captured_sources(tmp_path, "vllm-ray", ["h1", "h2"])

        (source,) = captured["sources"]
        assert source.container == "%s_head" % self.CID
        assert source.mode == MODE_FILE

    def test_native_job_reads_node_0(self, tmp_path):
        captured = self._captured_sources(tmp_path, "sglang", ["h1", "h2"])

        (source,) = captured["sources"]
        assert source.container == "%s_node_0" % self.CID

    def test_solo_job_reads_the_serve_log_not_container_stdout(self, tmp_path):
        """The second half of the old bug: ``docker logs`` on a sleep-infinity
        container is empty, because PID 1 is ``sleep infinity``."""
        captured = self._captured_sources(tmp_path, "sglang", ["h1"])

        (source,) = captured["sources"]
        assert source.container == "%s_solo" % self.CID
        assert source.mode == MODE_FILE
        assert source.path == SERVE_LOG_PATH

    def test_scope_all_fans_out_to_workers(self, tmp_path):
        captured = self._captured_sources(tmp_path, "sglang", ["h1", "h2", "h3"], scope=SCOPE_ALL)

        assert [s.container for s in captured["sources"]] == [
            "%s_node_0" % self.CID,
            "%s_node_1" % self.CID,
            "%s_node_2" % self.CID,
        ]

    def test_follow_and_tail_reach_the_reader(self, tmp_path):
        captured = self._captured_sources(tmp_path, "sglang", ["h1"], follow=True, tail=42)

        assert captured["kwargs"]["follow"] is True
        assert captured["kwargs"]["tail"] == 42

    def test_invalid_scope_rejected(self, tmp_path):
        with pytest.raises(api.SparkrunError):
            api.logs(self.CID, hosts=("h1",), scope="nonsense", cache_dir=str(tmp_path))


class TestApiLogsRecipeForm:
    """The recipe form resolves through live intent discovery, like api.stop."""

    CID = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"

    def test_recipe_form_discovers_by_intent(self, tmp_path):
        """Not derived: a load-aware scheduler randomizes the placement token, so
        the running id can't be computed from (recipe, hosts)."""
        from sparkrun.core.recipe import Recipe

        recipe = Recipe({"sparkrun_version": "2", "runtime": "sglang", "model": "test/m"})
        captured = {}

        def _capture(executor, sources, **kw):
            captured["sources"] = list(sources)
            return iter(())

        with (
            mock.patch("sparkrun.api._resolve.discover_cluster_id_by_intent", return_value=self.CID) as discover,
            mock.patch("sparkrun.orchestration.logs.read_log_sources", _capture),
        ):
            list(api.logs(recipe=recipe, hosts=("h1", "h2"), cache_dir=str(tmp_path)))

        discover.assert_called_once()
        assert captured["sources"][0].container == "%s_node_0" % self.CID

    def test_recipe_form_works_without_job_metadata(self, tmp_path):
        """The recipe carries its own runtime, so a missing metadata cache (job
        launched from another control machine, cache cleared) doesn't break it."""
        from sparkrun.core.recipe import Recipe

        recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-ray", "model": "test/m"})
        captured = {}

        def _capture(executor, sources, **kw):
            captured["sources"] = list(sources)
            return iter(())

        with (
            mock.patch("sparkrun.api._resolve.discover_cluster_id_by_intent", return_value=self.CID),
            mock.patch("sparkrun.orchestration.logs.read_log_sources", _capture),
        ):
            list(api.logs(recipe=recipe, hosts=("h1", "h2"), cache_dir=str(tmp_path)))

        assert captured["sources"][0].container == "%s_head" % self.CID

    def test_overrides_participate_in_the_intent(self, tmp_path):
        """Port is hashed into the intent, so it must reach the discovery call."""
        from sparkrun.core.recipe import Recipe
        from sparkrun.orchestration.job_metadata import generate_intent_id

        recipe = Recipe({"sparkrun_version": "2", "runtime": "sglang", "model": "test/m"})

        with (
            mock.patch("sparkrun.api._resolve.discover_cluster_id_by_intent", return_value=self.CID) as discover,
            mock.patch("sparkrun.orchestration.logs.read_log_sources", lambda *a, **k: iter(())),
        ):
            list(api.logs(recipe=recipe, hosts=("h1",), overrides={"port": 8123}, cache_dir=str(tmp_path)))

        assert discover.call_args.args[0] == generate_intent_id(recipe, overrides={"port": 8123})

    def test_neither_target_raises(self):
        with pytest.raises(api.SparkrunError):
            api.logs()


@pytest.mark.parametrize(
    "command",
    [
        "exec sleep 60",
        "while true; do echo startup; done",
        "sleep 60 & echo child:$!; wait",
        "python3 -c 'import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        'print("child:" + str(os.getpid()), flush=True); time.sleep(60)\' & wait',
    ],
)
def test_cancel_reaps_real_quiet_busy_and_child_readers(monkeypatch, command):
    import threading
    import time
    from sparkrun.orchestration import logs

    executor = mock.Mock()
    executor.read_logs_cmd.return_value = command
    source = LogSource(host="localhost", container="reader-test", mode=MODE_STDOUT)
    spawned = []
    original = logs._spawn
    started = threading.Event()

    def spawn(cmd):
        proc = original(cmd)
        spawned.append(proc)
        started.set()
        return proc

    monkeypatch.setattr(logs, "_spawn", spawn)
    child = []
    child_seen = threading.Event()

    def on_line(line):
        if line.text.startswith("child:"):
            child.append(int(line.text.split(":")[1]))
            child_seen.set()
        # Slow consumer exercises cancellation with a full producer queue.
        time.sleep(0.001)

    follower = logs.LogFollower(executor, [source], on_line=on_line).start()
    try:
        assert started.wait(5)
        if "child:" in command:
            assert child_seen.wait(5)
    finally:
        follower.stop()
    assert follower.done.is_set()
    assert follower.error is None
    assert spawned and all(proc.poll() is not None for proc in spawned)
    assert not any(t.name == "sparkrun-logs-" + source.label for t in threading.enumerate())
    from pathlib import Path

    for pid in child:
        status = Path("/proc/%d/stat" % pid)
        deadline = time.monotonic() + 2
        while True:
            try:
                if status.read_text().split()[2] == "Z":
                    break
            except (FileNotFoundError, ProcessLookupError):
                # Reaping may remove /proc before or during the read.
                break
            assert time.monotonic() < deadline, "reader child is still running"
            time.sleep(0.01)


def test_checked_follow_reports_reader_exit_code():
    import subprocess

    executor = mock.Mock()
    executor.read_logs_cmd.return_value = "exit 7"
    source = LogSource(host="localhost", container="reader-test")
    with pytest.raises(subprocess.CalledProcessError) as error:
        list(read_log_sources(executor, [source], follow=True, check=True))
    assert error.value.returncode == 7
