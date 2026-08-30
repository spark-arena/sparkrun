"""Tests for ``Executor.describe_terminated`` — the post-mortem seam.

``query_status`` reports what is *running*; ``describe_terminated`` reports what
became of something that isn't.  It exists because the answer — are there
remains, what state are they in, what should the operator run next — is
substrate-specific, and answering it in ``api.logs`` forced a ``docker ps -a``
onto every executor (PR #243).
"""

from __future__ import annotations

import base64
from unittest.mock import patch

import pytest

from sparkrun.orchestration.executors._base import ExecutorConfig
from sparkrun.orchestration.executors.docker import (
    POST_MORTEM_LOG_MARKER,
    DockerExecutor,
    _parse_post_mortem_logs,
    _parse_terminated_probe,
)
from sparkrun.orchestration.executors.k8s import K8sExecutor
from sparkrun.orchestration.executors.local import LocalExecutor
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.core.log_source import MODE_FILE, MODE_STDOUT, SERVE_LOG_PATH, LogSource
from tests.test_log_diagnostics import ISSUE_280_LOG


def _source(host: str, container: str) -> LogSource:
    return LogSource(host=host, container=container, role="solo")


def _b64(text: str) -> str:
    """Encode as the remote probe does, so the parser is tested on real framing."""
    return base64.b64encode(text.encode("utf-8")).decode()


class TestBaseDefault:
    """The base contract degrades to "cannot tell", never to "gone"."""

    def test_returns_empty_mapping(self):
        """A provider executor that hasn't implemented it contributes no verdict.

        Callers read an absent entry as "cannot tell", so such an executor never
        triggers the metadata deletion that only a confirmed ``exists=False``
        should.
        """
        from sparkrun.orchestration.executors._base import Executor

        assert Executor.describe_terminated(DockerExecutor(ExecutorConfig()), [_source("h1", "c1")]) == {}

    def test_empty_sources_short_circuits(self):
        assert DockerExecutor(ExecutorConfig()).describe_terminated([]) == {}


class TestDockerProbe:
    def _run(self, executor, sources, results):
        with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results) as fan_out:
            return executor.describe_terminated(sources), fan_out

    def test_stopped_container_is_inspectable(self):
        ex = DockerExecutor(ExecutorConfig())
        results = [RemoteResult(host="h1", returncode=0, stdout="c1\tExited (137) 3 minutes ago\n", stderr="")]
        found, _ = self._run(ex, [_source("h1", "c1")], results)

        info = found[("h1", "c1")]
        assert info.exists is True
        assert info.detail == "Exited (137) 3 minutes ago"
        assert info.investigate_hints == ("docker logs c1", "docker inspect c1")

    def test_absent_container_under_auto_remove_says_so(self):
        """`--rm` is the default, so absence is the normal outcome of a crash.

        Reporting a bare "gone" invites the caller to treat the most
        interesting failure as stale bookkeeping.
        """
        ex = DockerExecutor(ExecutorConfig(auto_remove=True))
        results = [RemoteResult(host="h1", returncode=0, stdout="", stderr="")]
        found, _ = self._run(ex, [_source("h1", "c1")], results)

        info = found[("h1", "c1")]
        assert info.exists is False
        assert "auto-removed on exit" in info.detail
        assert any("auto_remove=false" in h for h in info.investigate_hints)

    def test_absent_container_without_auto_remove_is_plainly_gone(self):
        ex = DockerExecutor(ExecutorConfig(auto_remove=False))
        results = [RemoteResult(host="h1", returncode=0, stdout="", stderr="")]
        found, _ = self._run(ex, [_source("h1", "c1")], results)

        info = found[("h1", "c1")]
        assert info.exists is False
        assert "auto-removed" not in (info.detail or "")
        assert info.investigate_hints == ()

    @pytest.mark.parametrize("rc", [255, 127, -1])
    def test_probe_failure_yields_no_verdict(self, rc):
        """Not "gone" — absent.  "Gone" is what deletes cached job metadata."""
        ex = DockerExecutor(ExecutorConfig())
        results = [RemoteResult(host="h1", returncode=rc, stdout="", stderr="boom")]
        found, _ = self._run(ex, [_source("h1", "c1")], results)
        assert found == {}

    def test_fan_out_exception_yields_no_verdict(self):
        ex = DockerExecutor(ExecutorConfig())
        with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=RuntimeError("ssh exploded")):
            assert ex.describe_terminated([_source("h1", "c1")]) == {}

    def test_shared_container_name_is_attributed_per_host(self):
        """Ray worker containers share one name across nodes.

        Keying by name alone would let one host's verdict overwrite another's.
        """
        ex = DockerExecutor(ExecutorConfig())
        sources = [_source("h1", "w"), _source("h2", "w")]
        results = [
            RemoteResult(host="h1", returncode=0, stdout="w\tExited (0) 1 minute ago\n", stderr=""),
            RemoteResult(host="h2", returncode=0, stdout="", stderr=""),
        ]
        found, _ = self._run(ex, sources, results)

        assert found[("h1", "w")].exists is True
        assert found[("h2", "w")].exists is False

    def test_one_parallel_sweep_regardless_of_source_count(self):
        """The old implementation did a second, sequential SSH for the head."""
        ex = DockerExecutor(ExecutorConfig())
        sources = [_source("h1", "a"), _source("h2", "b"), _source("h3", "c")]
        results = [RemoteResult(host=h, returncode=0, stdout="", stderr="") for h in ("h1", "h2", "h3")]
        _, fan_out = self._run(ex, sources, results)
        assert fan_out.call_count == 1

    def test_name_filter_is_anchored(self):
        """``foo`` must not match ``foo_solo`` — same anchoring as ``status_cmd``."""
        script = DockerExecutor(ExecutorConfig())._terminated_probe_script(["sparkrun_a_b_solo"])
        assert "'name=^sparkrun_a_b_solo$'" in script

    def test_container_name_cannot_reach_the_shell_raw(self):
        """Checked with the shell's own parser rather than by eyeballing quotes.

        The precheck runs before anything is known to be healthy, so it must not
        be the one place a name is interpolated unquoted.
        """
        import shlex

        nasty = "evil; rm -rf /"
        line = DockerExecutor(ExecutorConfig())._terminated_probe_script([nasty]).splitlines()[0]
        tokens = shlex.split(line)
        # The whole filter survives as exactly one argv token — no word
        # splitting, and the `;` never becomes a command separator.
        assert "name=^%s$" % nasty in tokens
        assert "rm" not in tokens


class TestParseProbe:
    def test_parses_name_and_status(self):
        assert _parse_terminated_probe("a\tUp 2 hours\nb\tExited (1) ago\n") == {"a": "Up 2 hours", "b": "Exited (1) ago"}

    @pytest.mark.parametrize("raw", ["", "   \n", "no-tab-here\n"])
    def test_ignores_unparseable_output(self, raw):
        assert _parse_terminated_probe(raw) == {}

    def test_skips_the_post_mortem_log_lines(self):
        """Both halves share one stdout and one tab-separated shape."""
        raw = "a\tUp 2 hours\n%s\ta\t%s\n" % (POST_MORTEM_LOG_MARKER, _b64("hi"))
        assert _parse_terminated_probe(raw) == {"a": "Up 2 hours"}


class TestParsePostMortemLogs:
    def test_decodes_per_container(self):
        raw = "c1\tExited (1) ago\n%s\tc1\t%s\n" % (POST_MORTEM_LOG_MARKER, _b64("boom\n"))
        assert _parse_post_mortem_logs(raw) == {"c1": "boom\n"}

    def test_payload_cannot_be_confused_for_framing(self):
        """A container is free to print anything, including our own marker.

        Base64 is what makes the payload unable to inject a line the parser
        would read as another container's log.
        """
        hostile = "%s\tother\tZmFrZQ==\n" % POST_MORTEM_LOG_MARKER
        raw = "%s\tc1\t%s\n" % (POST_MORTEM_LOG_MARKER, _b64(hostile))
        parsed = _parse_post_mortem_logs(raw)
        assert set(parsed) == {"c1"}
        assert parsed["c1"] == hostile

    def test_undecodable_payload_is_dropped_not_raised(self):
        """The payload came off a crashing workload; a mangled one costs a hint."""
        raw = "%s\tc1\tnot~valid~base64\n" % POST_MORTEM_LOG_MARKER
        assert _parse_post_mortem_logs(raw) == {}

    def test_truncated_utf8_survives(self):
        """``tail -c`` cuts mid-sequence, which is the normal case."""
        import base64

        payload = base64.b64encode("café".encode("utf-8")[:-1]).decode()
        raw = "%s\tc1\t%s\n" % (POST_MORTEM_LOG_MARKER, payload)
        assert _parse_post_mortem_logs(raw)["c1"].startswith("caf")

    @pytest.mark.parametrize("raw", ["", "c1\tUp 2 hours\n", "%s\tc1\t\n" % POST_MORTEM_LOG_MARKER])
    def test_nothing_to_decode(self, raw):
        assert _parse_post_mortem_logs(raw) == {}


class TestPostMortemAttribution:
    """A launcher decision that kills the workload should say so.

    sparkrun runs containers as the invoking uid; an image that JIT-compiles
    into its own ``site-packages`` dies with a traceback that points at the
    image and never at the ``--user`` flag that caused it (issue #280).
    """

    def _probe_output(self, container: str, status: str, log_text: str) -> str:
        return "%s\t%s\n%s\t%s\t%s\n" % (container, status, POST_MORTEM_LOG_MARKER, container, _b64(log_text))

    def test_recognised_failure_names_the_fix(self):
        ex = DockerExecutor(ExecutorConfig(auto_remove=False))
        results = [RemoteResult(host="h1", returncode=0, stdout=self._probe_output("c1", "Exited (1) ago", ISSUE_280_LOG), stderr="")]
        with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results):
            info = ex.describe_terminated([_source("h1", "c1")])[("h1", "c1")]

        assert info.exists is True
        assert any("-o user=root" in h for h in info.investigate_hints)
        # ``--rootful`` also adds --privileged; the image needs to write inside
        # itself, not to own the host.
        assert not any(h.startswith("relaunch with `--rootful`") for h in info.investigate_hints)

    def test_attribution_precedes_the_generic_hints(self):
        """It names a *likely* cause — the operator still wants the raw log."""
        ex = DockerExecutor(ExecutorConfig(auto_remove=False))
        results = [RemoteResult(host="h1", returncode=0, stdout=self._probe_output("c1", "Exited (1) ago", ISSUE_280_LOG), stderr="")]
        with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results):
            hints = ex.describe_terminated([_source("h1", "c1")])[("h1", "c1")].investigate_hints

        assert hints[-2:] == ("docker logs c1", "docker inspect c1")
        assert "user=root" in hints[1]

    def test_unrecognised_crash_adds_nothing(self):
        ex = DockerExecutor(ExecutorConfig(auto_remove=False))
        boring = "ValueError: No available memory for the cache blocks\n"
        results = [RemoteResult(host="h1", returncode=0, stdout=self._probe_output("c1", "Exited (1) ago", boring), stderr="")]
        with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results):
            hints = ex.describe_terminated([_source("h1", "c1")])[("h1", "c1")].investigate_hints

        assert hints == ("docker logs c1", "docker inspect c1")

    def test_gone_container_offers_no_attribution(self):
        """Under ``--rm`` there is no log left to read.

        Which is exactly what the ``auto_remove=false`` hint exists to change —
        it must not be displaced by a guess.
        """
        ex = DockerExecutor(ExecutorConfig(auto_remove=True))
        results = [RemoteResult(host="h1", returncode=0, stdout="", stderr="")]
        with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results):
            info = ex.describe_terminated([_source("h1", "c1")])[("h1", "c1")]

        assert info.exists is False
        assert info.investigate_hints == ("relaunch with `-o auto_remove=false` to keep the container for inspection",)

    def test_a_broken_detector_never_breaks_the_post_mortem(self):
        ex = DockerExecutor(ExecutorConfig(auto_remove=False))
        results = [RemoteResult(host="h1", returncode=0, stdout=self._probe_output("c1", "Exited (1) ago", ISSUE_280_LOG), stderr="")]
        with (
            patch("sparkrun.utils.log_diagnostics.detect_in_place_write_failure", side_effect=RuntimeError("boom")),
            patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results),
        ):
            hints = ex.describe_terminated([_source("h1", "c1")])[("h1", "c1")].investigate_hints

        assert hints == ("docker logs c1", "docker inspect c1")


class TestPostMortemLogScript:
    """Retrieval is mode-aware for the same reason ``read_logs_cmd`` is."""

    def test_file_mode_reads_the_in_container_serve_log(self):
        """``docker logs`` is empty for sleep-infinity + exec workloads.

        That is sparkrun's dominant launch pattern, so a ``docker logs``-only
        probe would find nothing to attribute on most recipes.  ``docker cp``
        rather than ``read_logs_cmd``'s ``docker exec``: the container has
        already exited and ``exec`` needs a running one.
        """
        script = DockerExecutor(ExecutorConfig())._post_mortem_log_script(
            [LogSource(host="h1", container="c1", role="solo", mode=MODE_FILE, path=SERVE_LOG_PATH)]
        )
        assert "docker cp c1:%s -" % SERVE_LOG_PATH in script
        assert "tar -xO" in script
        assert "docker logs" not in script

    def test_stdout_mode_reads_docker_logs_with_stderr(self):
        """``docker logs`` demultiplexes — the traceback arrives on *its* stderr."""
        script = DockerExecutor(ExecutorConfig())._post_mortem_log_script(
            [LogSource(host="h1", container="w", role="worker", mode=MODE_STDOUT)]
        )
        assert "docker logs --tail" in script
        assert "2>&1" in script
        assert "docker cp" not in script

    def test_one_line_per_distinct_source(self):
        """Ray workers share a name across hosts; one script serves every host."""
        sources = [
            LogSource(host="h1", container="w", role="worker", mode=MODE_STDOUT),
            LogSource(host="h2", container="w", role="worker", mode=MODE_STDOUT),
            LogSource(host="h1", container="head", role="head", mode=MODE_FILE),
        ]
        script = DockerExecutor(ExecutorConfig())._post_mortem_log_script(sources)
        assert len([ln for ln in script.splitlines() if ln.strip()]) == 2

    def test_no_sources_emits_nothing(self):
        assert DockerExecutor(ExecutorConfig())._post_mortem_log_script([]) == ""

    def test_container_name_and_path_cannot_reach_the_shell_raw(self):
        """Same rule as the ``ps`` half — this runs on an already-broken workload.

        Checked with the shell's own parser: neither the ``;`` in the name nor
        the space in the path may become word-splitting or a command separator.
        """
        import shlex

        nasty = "evil; rm -rf /"
        script = DockerExecutor(ExecutorConfig())._post_mortem_log_script(
            [LogSource(host="h1", container=nasty, role="solo", mode=MODE_FILE, path="/tmp/a b.log")]
        )
        # A token may carry the `;` that legitimately ends the printf command;
        # strip it so the comparison is about word-splitting, not punctuation.
        tokens = [token.rstrip(";") for token in shlex.split(script)]

        assert "rm" not in tokens
        # Both interpolation sites survive as exactly one argv token each.
        assert nasty in tokens
        assert "%s:/tmp/a b.log" % nasty in tokens

    def test_script_runs_under_real_bash(self):
        """The framing is generated by ``printf`` in a ``%``-formatted string.

        A mis-escaped ``%%`` yields a script that runs but frames nothing, which
        no amount of substring-asserting would catch.
        """
        import os
        import subprocess
        import tempfile

        ex = DockerExecutor(ExecutorConfig())
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = os.path.join(tmp, "bin")
            os.mkdir(bin_dir)
            fake = os.path.join(bin_dir, "docker")
            with open(fake, "w") as fh:
                fh.write('#!/usr/bin/env bash\n[ "$1" = logs ] && printf %s "crash output" || true\n')
            os.chmod(fake, 0o755)

            script = ex._post_mortem_log_script([LogSource(host="h1", container="c1", role="solo", mode=MODE_STDOUT)])
            proc = subprocess.run(
                ["bash", "-s"],
                input=script,
                capture_output=True,
                text=True,
                env=dict(os.environ, PATH=bin_dir + os.pathsep + os.environ["PATH"]),
            )

        assert proc.returncode == 0
        assert _parse_post_mortem_logs(proc.stdout) == {"c1": "crash output"}


class TestLocalProbe:
    """No containers here at all — `docker logs` would be meaningless advice."""

    def test_surviving_logfile_is_the_thing_to_read(self):
        ex = LocalExecutor(ExecutorConfig(log_dir="/var/log/sr"))
        results = [RemoteResult(host="h1", returncode=0, stdout="c1\t/var/log/sr/c1.log\n", stderr="")]
        with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results):
            found = ex.describe_terminated([_source("h1", "c1")])

        info = found[("h1", "c1")]
        assert info.exists is True
        assert info.investigate_hints == ("cat /var/log/sr/c1.log",)
        assert not any("docker" in h for h in info.investigate_hints)

    def test_missing_logfile_is_gone(self):
        ex = LocalExecutor(ExecutorConfig(log_dir="/var/log/sr"))
        results = [RemoteResult(host="h1", returncode=0, stdout="", stderr="")]
        with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results):
            found = ex.describe_terminated([_source("h1", "c1")])
        assert found[("h1", "c1")].exists is False

    def test_probe_script_names_no_container_engine(self):
        ex = LocalExecutor(ExecutorConfig(log_dir="/var/log/sr"))
        captured = {}

        def _capture(hosts, script, **kwargs):
            captured["script"] = script
            return [RemoteResult(host="h1", returncode=0, stdout="", stderr="")]

        with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=_capture):
            ex.describe_terminated([_source("h1", "c1")])
        assert "docker" not in captured["script"]
        assert "kubectl" not in captured["script"]


class TestK8sProbe:
    def test_terminal_pod_phase_is_inspectable(self):
        ex = K8sExecutor(ExecutorConfig(k8s_namespace="ns"))
        results = [RemoteResult(host="h1", returncode=0, stdout="c1\tFailed\n", stderr="")]
        with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results):
            found = ex.describe_terminated([_source("h1", "c1")])

        info = found[("h1", "c1")]
        assert info.exists is True
        assert info.detail == "pod phase Failed"
        assert info.investigate_hints == ("kubectl logs c1", "kubectl describe pod c1")
        assert not any("docker" in h for h in info.investigate_hints)

    def test_missing_pod_is_gone(self):
        ex = K8sExecutor(ExecutorConfig(k8s_namespace="ns"))
        results = [RemoteResult(host="h1", returncode=0, stdout="c1\t\n", stderr="")]
        with patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", return_value=results):
            found = ex.describe_terminated([_source("h1", "c1")])
        assert found[("h1", "c1")].exists is False


class TestApiLogsIsSubstrateAgnostic:
    """The regression this whole seam exists to prevent.

    Before, ``api.logs`` built its own ``docker ps -a``, so the precheck issued
    a docker command no matter which executor was in play — meaningless for a
    ``local`` job (a native process with a logfile) and wrong on k8s.
    """

    def _job(self, tmp_path):
        from sparkrun.core.recipe import Recipe
        from sparkrun.orchestration.job_metadata import save_job_metadata

        recipe = Recipe({"sparkrun_version": "2", "runtime": "vllm-distributed", "model": "test/m"})
        cluster_id = "sparkrun_aaaaaaaaaaaaaaaa_111111111111"
        save_job_metadata(cluster_id, recipe, ["h1"], cache_dir=str(tmp_path))
        return cluster_id

    def test_local_executor_precheck_issues_no_docker_command(self, tmp_path):
        from sparkrun import api
        from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy
        from sparkrun.orchestration.job_metadata import load_job_metadata

        cluster_id = self._job(tmp_path)
        executor = LocalExecutor(ExecutorConfig(executor_type="local", log_dir="/var/log/sr"))
        empty = ClusterStatus(hosts=(HostOccupancy(host="h1"),), executor="local")

        scripts: list[str] = []

        def _capture(hosts, script, **kwargs):
            scripts.append(script)
            # The logfile survives, so the workload is inspectable.
            return [RemoteResult(host="h1", returncode=0, stdout="%s_solo\t/var/log/sr/x.log\n" % cluster_id, stderr="")]

        with (
            patch("sparkrun.orchestration.executor.resolve_executor", return_value=executor),
            patch.object(LocalExecutor, "query_status", return_value=empty),
            patch("sparkrun.orchestration.ssh.run_remote_scripts_parallel", side_effect=_capture),
        ):
            with pytest.raises(api.JobNotFound) as exc:
                api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)

        assert scripts, "the precheck never probed the substrate"
        for script in scripts:
            assert "docker" not in script

        msg = str(exc.value)
        assert "docker" not in msg
        assert "cat /var/log/sr/x.log" in msg
        # Remains exist → metadata preserved.
        assert load_job_metadata(cluster_id, cache_dir=str(tmp_path)) is not None

    def test_executor_without_post_mortem_support_preserves_metadata(self, tmp_path):
        """An executor that can't answer must not look like "confirmed gone"."""
        from sparkrun import api
        from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy
        from sparkrun.orchestration.job_metadata import load_job_metadata

        cluster_id = self._job(tmp_path)
        executor = DockerExecutor(ExecutorConfig())
        empty = ClusterStatus(hosts=(HostOccupancy(host="h1"),), executor="docker")

        with (
            patch("sparkrun.orchestration.executor.resolve_executor", return_value=executor),
            patch.object(DockerExecutor, "query_status", return_value=empty),
            patch.object(DockerExecutor, "describe_terminated", return_value={}),
        ):
            with pytest.raises(api.JobNotFound) as exc:
                api.logs(cluster_id, hosts=("h1",), cache_dir=str(tmp_path), tail=10)

        assert "nothing remains to read" not in str(exc.value)
        assert load_job_metadata(cluster_id, cache_dir=str(tmp_path)) is not None
