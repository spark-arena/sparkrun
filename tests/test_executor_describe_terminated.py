"""Tests for ``Executor.describe_terminated`` — the post-mortem seam.

``query_status`` reports what is *running*; ``describe_terminated`` reports what
became of something that isn't.  It exists because the answer — are there
remains, what state are they in, what should the operator run next — is
substrate-specific, and answering it in ``api.logs`` forced a ``docker ps -a``
onto every executor (PR #243).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sparkrun.orchestration.executors._base import ExecutorConfig
from sparkrun.orchestration.executors.docker import DockerExecutor, _parse_terminated_probe
from sparkrun.orchestration.executors.k8s import K8sExecutor
from sparkrun.orchestration.executors.local import LocalExecutor
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.core.log_source import LogSource


def _source(host: str, container: str) -> LogSource:
    return LogSource(host=host, container=container, role="solo")


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
