"""Recipe hooks reach a workload through its executor, not an assumed ``docker exec``."""

from __future__ import annotations

import base64
import subprocess
from unittest import mock

import pytest

from _hook_scripts import hook_payload
from sparkrun.orchestration.executors._base import ExecutorConfig
from sparkrun.orchestration.executors.docker import DockerExecutor
from sparkrun.orchestration.executors.local import LocalExecutor
from sparkrun.orchestration.hooks import run_post_exec, run_pre_exec
from sparkrun.orchestration.ssh import RemoteResult


def _ok(*args, **kwargs):
    return RemoteResult("h", 0, "", "")


@mock.patch("sparkrun.orchestration.primitives.run_script_on_host", side_effect=_ok)
def test_without_an_executor_hooks_keep_their_docker_root_exec(mock_run):
    run_pre_exec([("h1", "ctr")], ["echo hi"], {}, trust=True)

    script = mock_run.call_args[0][1]
    assert script.startswith("docker exec --user root ")
    assert " ctr bash -c " in script
    assert base64.b64encode(b"echo hi").decode() in hook_payload(script)


@mock.patch("sparkrun.orchestration.primitives.run_script_on_host", side_effect=_ok)
def test_each_host_uses_its_own_executor(mock_run):
    docker = DockerExecutor()
    local = LocalExecutor(ExecutorConfig(working_dir="/srv/w"))
    docker._host_executors = {"h1": docker, "h2": local}

    run_pre_exec([("h1", "c1"), ("h2", "c2")], ["true"], {}, trust=True, executor=docker)

    first, second = (call[0][1] for call in mock_run.call_args_list)
    assert first.startswith("docker exec --user root ")
    assert not second.startswith("docker") and "cd -- /srv/w" in second


@mock.patch("sparkrun.orchestration.primitives.run_script_on_host", side_effect=_ok)
def test_copy_entries_are_refused_up_front_without_a_container(mock_run):
    with pytest.raises(RuntimeError, match="need a container; the local executor on h1 has none"):
        run_pre_exec([("h1", "c1")], ["echo first", {"copy": "/tmp/x"}], {}, trust=True, executor=LocalExecutor())

    assert not mock_run.called, "nothing may run before the refusal"


def test_a_local_workloads_hooks_run_natively_in_its_environment(tmp_path):
    """The case that used to ``docker exec`` into a container that does not exist."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "venv.env").write_text("VIRTUAL_ENV=%s/venv\n" % tmp_path)
    executor = LocalExecutor(ExecutorConfig(working_dir=str(work), env_file="venv.env"))
    out = tmp_path / "hook.out"
    scripts = []

    def run_here(host, script, **kwargs):
        scripts.append(script)
        done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30, cwd=tmp_path)
        return RemoteResult(host, done.returncode, done.stdout, done.stderr)

    with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", side_effect=run_here):
        run_pre_exec(
            [("localhost", "sparkrun_" + "a" * 16 + "_" + "b" * 12 + "_solo")],
            ['echo "pre $PWD $VIRTUAL_ENV $SPARKRUN_HOOK" > %s' % out],
            {},
            trust=True,
            executor=executor,
        )
        assert out.read_text().split() == ["pre", str(work), "%s/venv" % tmp_path, "pre_exec"]

        run_post_exec("localhost", "ctr", ['echo "post $SPARKRUN_HOOK" >> %s' % out], {}, trust=True, executor=executor)

    assert out.read_text().splitlines()[-1] == "post post_exec"
    assert not any(script.startswith("docker") for script in scripts)


def test_a_local_hook_still_has_its_deadline(tmp_path):
    executor = LocalExecutor(ExecutorConfig(working_dir=str(tmp_path)))

    def run_here(host, script, **kwargs):
        done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        return RemoteResult(host, done.returncode, done.stdout, done.stderr)

    with mock.patch("sparkrun.orchestration.primitives.run_script_on_host", side_effect=run_here):
        with pytest.raises(RuntimeError, match="timed out after"):
            run_pre_exec([("localhost", "c")], [{"command": "sleep 30", "timeout": 1}], {}, trust=True, executor=executor)
