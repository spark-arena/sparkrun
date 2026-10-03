"""Real process and filesystem acceptance for native lifecycle boundaries."""

import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import pytest

from sparkrun.core.application_profile import get_application_profile
from sparkrun.orchestration.executors._base import ExecutorConfig
from sparkrun.orchestration.executors.local import LocalExecutor
from sparkrun.orchestration.teardown import parse_teardown_removed

NAME = "sparkrun_aaaabbbbccccdddd_eeeeffff0000_solo"


def bash(script, *, cwd=None, env=None):
    return subprocess.run(["bash", "-c", script], cwd=cwd, env=env, capture_output=True, text=True, timeout=20)


def await_file(path):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return path.read_text().strip()
        time.sleep(0.02)
    raise AssertionError("workload never became ready")


def alive(pid):
    r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip()) and not r.stdout.strip().startswith("Z")


def cleanup_group(pid):
    if pid is not None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize("prefix", ["~", "$HOME", "${HOME}"])
def test_explicit_literal_relative_home_spelling_is_preserved(tmp_path, prefix):
    actual_dir = tmp_path / prefix
    actual_dir.mkdir()
    remote_home = tmp_path / "remote-home"
    remote_home.mkdir()
    (actual_dir / "serve.log").write_text("literal relative log\n")
    (remote_home / "serve.log").write_text("wrong home log\n")
    configured = "./" + prefix + "/serve.log"
    env = dict(os.environ, HOME=str(remote_home))
    assert (tmp_path / configured).read_text() == "literal relative log\n"
    executor = LocalExecutor(ExecutorConfig(log_file=configured))
    result = bash(executor.logs_cmd(NAME), cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "literal relative log\n"
    home_target = LocalExecutor(ExecutorConfig(pid_dir=prefix + "/pids")).resolve_target()
    assert home_target.config["pid_dir"] == "$HOME/pids"
    with pytest.raises(ValueError, match="pid_dir requires an absolute or home-relative"):
        LocalExecutor(ExecutorConfig(pid_dir="./" + prefix + "/pids")).resolve_target()


def test_working_dir_preserves_absolute_control_state(tmp_path):
    work = tmp_path / "work"
    for root in (tmp_path, work):
        (root / "pids").mkdir(parents=True)
        (root / "logs").mkdir()
    ready = tmp_path / "ready"
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path / "pids"), log_dir=str(tmp_path / "logs"), working_dir=str(work)))
    assert executor.resolve_target().config["pid_dir"] == str(tmp_path / "pids")
    pid = None
    try:
        command = "printf ready > %s; exec sleep 300" % shlex.quote(str(ready))
        launched = bash(executor.run_cmd("", command, NAME), cwd=tmp_path)
        assert launched.returncode == 0, launched.stderr
        await_file(ready)
        record = tmp_path / "pids" / (NAME + ".pid")
        pid = int(record.read_text())
        assert alive(pid)
        assert not (work / "pids" / (NAME + ".pid")).exists()
        assert bash(executor.status_cmd(NAME), cwd=tmp_path).returncode == 0
        result = bash(executor.teardown_script([NAME]), cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        assert parse_teardown_removed(result.stdout) == 1
        assert not record.exists() and not alive(pid)
    finally:
        cleanup_group(pid)


@pytest.mark.parametrize("leader_exits_first", [False, True])
def test_native_teardown_accounts_for_workers_after_leader_exit(tmp_path, leader_exits_first):
    ready = tmp_path / "worker.pid"
    child = (
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); "
        "time.sleep(300)"
    )
    parent = f'import subprocess, sys, time; subprocess.Popen([sys.executable, "-c", {child!r}]); time.sleep(300)'
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path / "pids"), log_dir=str(tmp_path / "logs")))
    record = tmp_path / "pids" / (NAME + ".pid")
    pid = None
    try:
        launched = bash(executor.run_cmd("", shlex.quote(sys.executable) + " -c " + shlex.quote(parent), NAME))
        assert launched.returncode == 0, launched.stderr
        pid = int(record.read_text())
        worker = int(await_file(ready))
        assert os.getpgid(worker) == pid
        assert alive(pid) and alive(worker)
        if leader_exits_first:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while alive(pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not alive(pid) and alive(worker)
            assert bash(executor.status_cmd(NAME)).returncode == 0
            assert executor.query_status(["localhost"]).running_cluster_ids()
            replacement = bash(executor.run_cmd("", "true", NAME))
            assert replacement.returncode != 0
            assert record.read_text() == str(pid) + "\n"
        # These workers deliberately ignore TERM. Advance only the ten
        # one-second grace sleeps; keep real process checks, signals, and the
        # post-KILL polling delay so scheduler latency cannot make this flaky.
        waits = tmp_path / "grace-waits"
        sleep_stub = (
            "sleep() { if [ \"$1\" = 1 ]; then printf 'wait\\n' >> " + shlex.quote(str(waits)) + '; else command sleep "$@"; fi; };\n'
        )
        result = bash(sleep_stub + executor.teardown_script([NAME]))
        assert waits.read_text().splitlines() == ["wait"] * 10
        assert result.returncode == 0, result.stderr
        assert parse_teardown_removed(result.stdout) == 1
        assert not record.exists() and not Path(str(record) + ".owner").exists()
        assert not alive(pid) and not alive(worker)
        assert bash(executor.status_cmd(NAME)).returncode == 1
    finally:
        cleanup_group(pid)


@pytest.mark.parametrize("problem", ["working_dir", "env_file"])
@pytest.mark.parametrize("operation", ["launch", "exec"])
def test_native_setup_failure_prevents_payload(tmp_path, problem, operation):
    marker = tmp_path / "payload-ran"
    executor = LocalExecutor(
        ExecutorConfig(
            pid_dir=str(tmp_path / "pids"),
            log_dir=str(tmp_path / "logs"),
            **{problem: str(tmp_path / "missing")},
        )
    )
    payload = "printf executed > " + shlex.quote(str(marker))
    script = (
        executor.generate_exec_serve_script(NAME, payload + "; exec sleep 300")
        if operation == "launch"
        else executor.exec_cmd(NAME, payload)
    )
    pid = None
    try:
        result = bash(script, cwd=tmp_path)
        if operation == "launch":
            record = tmp_path / "pids" / (NAME + ".pid")
            if record.exists():
                pid = int(record.read_text())
        assert result.returncode != 0
        assert "No such file or directory" in result.stderr
        assert not marker.exists()
        assert not (tmp_path / "pids" / (NAME + ".pid")).exists()
        assert not (tmp_path / "pids" / (NAME + ".pid.owner")).exists()
    finally:
        cleanup_group(pid)


def test_native_environment_home_preserves_default_control_state(tmp_path):
    initial_home, workload_home = tmp_path / "initial", tmp_path / "workload"
    suffix = ".cache/" + get_application_profile().cache_namespace + "/local"
    for root in (initial_home, workload_home):
        (root / suffix / "pids").mkdir(parents=True)
        (root / suffix / "logs").mkdir()
    marker = tmp_path / "payload-ran"
    executor = LocalExecutor(ExecutorConfig())
    env = dict(os.environ, HOME=str(initial_home))
    pid = None
    try:
        script = executor.run_cmd(
            "", "printf ready > " + shlex.quote(str(marker)) + "; exec sleep 300", NAME, env={"HOME": str(workload_home)}
        )
        result = bash(script, cwd=tmp_path, env=env)
        assert result.returncode == 0, result.stderr
        await_file(marker)
        record = initial_home / suffix / "pids" / (NAME + ".pid")
        pid = int(record.read_text())
        assert alive(pid)
        assert bash(executor.status_cmd(NAME), cwd=tmp_path, env=env).returncode == 0
        stopped = bash(executor.teardown_script([NAME]), cwd=tmp_path, env=env)
        assert stopped.returncode == 0 and parse_teardown_removed(stopped.stdout) == 1
        assert not record.exists() and not alive(pid)
    finally:
        cleanup_group(pid)


@pytest.mark.parametrize("state", ["live_legacy", "zombie_group"])
def test_legacy_pid_and_zombie_group_recovery(tmp_path, state):
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path)))
    child = subprocess.Popen(["sleep", "300" if state == "live_legacy" else ".01"], start_new_session=state == "zombie_group")
    record = tmp_path / (NAME + ".pid")
    record.write_text(str(child.pid))
    try:
        if state == "live_legacy":
            assert os.getpgid(child.pid) != child.pid
            assert bash(executor.status_cmd(NAME)).returncode == 0
        else:
            deadline = time.monotonic() + 5
            while alive(child.pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not alive(child.pid)
            assert bash(executor.status_cmd(NAME)).returncode == 1
        result = bash(executor.teardown_script([NAME]))
        assert result.returncode == 0, result.stderr
        assert parse_teardown_removed(result.stdout) == (1 if state == "live_legacy" else 0)
        assert not record.exists()
        child.wait(timeout=5)
    finally:
        child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize("problem", ["unsearchable_directory", "unreadable_env", "nonzero_env", "readonly_env", "readonly_gpu"])
@pytest.mark.parametrize("operation", ["launch", "exec"])
def test_setup_rejection_preserves_existing_claim(tmp_path, problem, operation):
    path = tmp_path / "setup"
    if problem == "unsearchable_directory":
        path.mkdir()
        key = "working_dir"
    else:
        source = {
            "nonzero_env": "return 7\n",
            "readonly_env": "readonly TEST_SETUP=locked\n",
            "readonly_gpu": "readonly CUDA_VISIBLE_DEVICES=locked\n",
        }.get(problem, "export TEST_SETUP=yes\n")
        path.write_text(source)
        key = "env_file"
    if problem in {"unsearchable_directory", "unreadable_env"}:
        if os.geteuid() == 0:
            pytest.skip("Permission tests require an unprivileged account")
        path.chmod(0)
    record = tmp_path / (NAME + ".pid")
    marker = Path(str(record) + ".owner")
    record.write_text("999999999")
    marker.write_text(get_application_profile().id)
    payload = tmp_path / "payload"
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path), log_dir=str(tmp_path), gpus="device=0", **{key: str(path)}))
    command = "printf ran > " + shlex.quote(str(payload))
    try:
        options = {"env": {"TEST_SETUP": "explicit"}}
        script = executor.run_cmd("", command, NAME, **options) if operation == "launch" else executor.exec_cmd(NAME, command, **options)
        result = bash(script)
        assert result.returncode != 0
        assert not payload.exists()
        assert record.read_text() == "999999999"
        assert marker.read_text() == get_application_profile().id
    finally:
        path.chmod(0o700 if key == "working_dir" else 0o600)


@pytest.mark.parametrize("path_kind", ["absolute", "relative", "home"])
@pytest.mark.parametrize("operation", ["launch", "exec"])
def test_activation_environment_and_directory_do_not_relocate_control_state(tmp_path, path_kind, operation):
    home, work, activated = (tmp_path / name for name in ("home", "work", "activated"))
    for root in (home, work, activated):
        root.mkdir()
    unrelated = tmp_path / "unrelated-bin"
    unrelated.mkdir()
    (unrelated / "activate.sh").write_text("return 23\n")
    env = dict(os.environ, HOME=str(home), PATH=str(unrelated) + os.pathsep + os.environ["PATH"])
    (work / "activate.sh").write_text(
        "HOME=%s\ncd -- %s\nTEST_SETUP=activation\n" % (shlex.quote(str(activated)), shlex.quote(str(activated)))
    )
    if path_kind == "absolute":
        root = tmp_path / "state"
        config_root = str(root)
    elif path_kind == "home":
        root = home / "state"
        config_root = "$HOME/state"
    else:
        root = tmp_path / "state"
        config_root = "state"
    executor = LocalExecutor(
        ExecutorConfig(
            pid_dir=config_root + "/pids",
            log_dir=config_root + "/logs",
            working_dir=str(work),
            env_file="activate.sh",
        )
    )
    ready = tmp_path / "ready"
    content = tmp_path / "observed-environment"
    payload = 'printf "%s|%s|%s\\n" "$HOME" "$PWD" "$TEST_SETUP" | tee %s; printf ready > %s' % (
        "%s",
        "%s",
        "%s",
        shlex.quote(str(content)),
        shlex.quote(str(ready)),
    )
    if path_kind == "relative" and operation == "launch":
        with pytest.raises(ValueError, match="pid_dir requires an absolute or home-relative"):
            executor.run_cmd("", payload, NAME)
        assert not ready.exists() and not root.exists()
        return
    pid = None
    try:
        script = (
            executor.run_cmd("", payload + "; exec sleep 300", NAME, env={"TEST_SETUP": "explicit"})
            if operation == "launch"
            else executor.exec_cmd(NAME, payload, env={"TEST_SETUP": "explicit"})
        )
        result = bash(script, cwd=tmp_path, env=env)
        assert result.returncode == 0, result.stderr
        await_file(ready)
        assert content.read_text() == f"{activated}|{activated}|explicit\n"
        if operation == "launch":
            record = root / "pids" / (NAME + ".pid")
            pid = int(record.read_text())
            assert alive(pid)
            log = bash(executor.logs_cmd(NAME), cwd=tmp_path, env=env)
            assert log.returncode == 0 and log.stdout == content.read_text()
            assert bash(executor.status_cmd(NAME), cwd=tmp_path, env=env).returncode == 0
            stopped = bash(executor.teardown_script([NAME]), cwd=tmp_path, env=env)
            assert stopped.returncode == 0 and parse_teardown_removed(stopped.stdout) == 1
            assert not alive(pid) and not record.exists()
            assert not (work / "state").exists() and not (activated / "state").exists()
    finally:
        cleanup_group(pid)


@pytest.mark.parametrize("failure", ["command", "empty", "invalid"])
def test_group_inspection_failure_preserves_recovery_state(tmp_path, monkeypatch, failure):
    from sparkrun.orchestration.ssh import RemoteResult

    child = subprocess.Popen(["sleep", "300"], start_new_session=True)
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path)))
    record = tmp_path / (NAME + ".pid")
    marker = Path(str(record) + ".owner")
    record.write_text(str(child.pid))
    marker.write_text(get_application_profile().id)
    bodies = {
        "command": "return 2",
        "empty": "return 0",
        "invalid": "printf '%s %s ?\\n'" % (child.pid, child.pid),
    }
    inject = "ps() { %s; };\n" % bodies[failure]

    def remote(hosts, script, **kwargs):
        result = bash(inject + script)
        return [RemoteResult(host, result.returncode, result.stdout, result.stderr) for host in hosts]

    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_scripts_parallel", remote)
    try:
        assert executor.query_status(["localhost"]).errors
        for script in (executor.status_cmd(NAME), executor.run_cmd("", "true", NAME), executor.teardown_script([NAME])):
            result = bash(inject + script)
            assert result.returncode == 2, result.stderr
            assert record.read_text() == str(child.pid)
            assert marker.exists() and child.poll() is None
    finally:
        child.kill()
        child.wait(timeout=5)
