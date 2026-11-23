"""The in-tree ``connect`` plugin: ``sparkrun ssh`` / ``exec`` targets, CLI wiring and gating."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from click.testing import CliRunner

from sparkrun.core.cluster_manager import DEFAULT_CLUSTER_ENV, ClusterManager
from sparkrun.plugins.connect.target import ConnectTargetError, NodeTarget, parse_cluster_target, remote_command, split_node_suffix


# ---------------------------------------------------------------------------
# Target parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (None, NodeTarget(None, 0)),
        ("", NodeTarget(None, 0)),
        ("2", NodeTarget(None, 2)),
        ("lab", NodeTarget("lab", 0)),
        ("lab/", NodeTarget("lab", 0)),
        ("lab/3", NodeTarget("lab", 3)),
        ("2/", NodeTarget("2", 0)),
        ("@official/qwen", NodeTarget("@official/qwen", 0)),
        ("@official/qwen/1", NodeTarget("@official/qwen", 1)),
        ("./recipes/x.yaml", NodeTarget("./recipes/x.yaml", 0)),
        ("/1", NodeTarget("/1", 0)),
    ],
)
def test_split_node_suffix(target, expected):
    assert split_node_suffix(target) == expected


@pytest.mark.parametrize(
    ("target", "option", "expected"),
    [
        (None, None, NodeTarget(None, 0)),
        ("1", "lab", NodeTarget("lab", 1)),
        (None, "lab", NodeTarget("lab", 0)),
        ("lab/2", None, NodeTarget("lab", 2)),
        ("2/1", None, NodeTarget("2", 1)),
    ],
)
def test_parse_cluster_target(target, option, expected):
    assert parse_cluster_target(target, option) == expected


@pytest.mark.parametrize(
    ("target", "option"),
    [("lab/x", None), ("lab/-1", None), ("lab/1/2", None), ("../lab", None), ("-lab", None), ("/1", None), ("²", None), ("lab", "other")],
)
def test_parse_cluster_target_rejects(target, option):
    with pytest.raises(ConnectTargetError):
        parse_cluster_target(target, option)


def test_remote_command_single_word_is_shell_syntax_several_are_arguments():
    assert remote_command([]) is None
    assert remote_command(["ps aux | grep vllm"]) == "ps aux | grep vllm"
    assert remote_command(["grep", "two words", "$HOME"]) == "grep 'two words' '$HOME'"


# ---------------------------------------------------------------------------
# Shared ssh pieces
# ---------------------------------------------------------------------------


def test_interactive_ssh_argv_drops_batch_mode_only():
    from sparkrun.orchestration.ssh import build_ssh_cmd, interactive_ssh_argv

    argv = interactive_ssh_argv("10.0.0.1", "uptime", ssh_user="ops", ssh_key="/k/id", ssh_options=["-p", "2222"], tty=True)

    assert "BatchMode=yes" not in argv
    assert argv[:2] == ["ssh", "-t"]
    assert argv[-2:] == ["ops@10.0.0.1", "uptime"]
    for pair in (["-l", "ops"], ["-i", "/k/id"], ["-p", "2222"]):
        assert pair == argv[argv.index(pair[0]) : argv.index(pair[0]) + 2]
    assert "BatchMode=yes" in build_ssh_cmd("h")


@pytest.mark.parametrize("host", ["-oProxyCommand=touch /tmp/pwn", "", "a b", "h\n-oX=y"])
def test_option_shaped_hosts_never_reach_ssh(host):
    from sparkrun.orchestration.ssh import build_ssh_cmd, interactive_ssh_argv, run_rsync

    for call in (lambda: build_ssh_cmd(host), lambda: interactive_ssh_argv(host), lambda: run_rsync("/src", host, "/dst", dry_run=True)):
        with pytest.raises(ValueError, match="Refusing host"):
            call()


def test_ssh_user_may_contain_a_space_but_not_an_option():
    from sparkrun.orchestration.ssh import build_ssh_cmd, ssh_destination

    assert ssh_destination("h", "dgx user") == "dgx user@h"
    assert ssh_destination("h", "") == "h"  # empty has always meant the client's default
    assert build_ssh_cmd("h", ssh_user="")[-1] == "h"
    with pytest.raises(ValueError, match="ssh user"):
        ssh_destination("h", "-oProxyCommand=x")


# ---------------------------------------------------------------------------
# Executor hook
# ---------------------------------------------------------------------------


def test_docker_exec_argv():
    from sparkrun.orchestration.executors.docker import DockerExecutor

    argv = DockerExecutor().interactive_exec_argv("h1", "ctr", "nvidia-smi -L", tty=True, ssh_kwargs={"ssh_user": "ops"})

    assert argv[:2] == ["ssh", "-t"] and "BatchMode=yes" not in argv
    assert argv[-2] == "ops@h1"
    assert argv[-1] == "docker exec -i -t ctr sh -c 'nvidia-smi -L'"


def test_docker_exec_shell_prefers_bash_and_skips_tty_when_asked():
    from sparkrun.orchestration.executors.docker import DockerExecutor

    argv = DockerExecutor().interactive_exec_argv("h1", "ctr")

    assert "-t" not in argv
    assert argv[-1].startswith("docker exec -i ctr sh -c ")
    assert "command -v bash" in argv[-1] and "exec sh" in argv[-1]


def test_k8s_exec_argv_runs_on_the_control_machine():
    from sparkrun.core.log_source import LogResource
    from sparkrun.plugins.k8s.executor import K8sExecutor

    executor = K8sExecutor()

    class _Client:
        def base_args(self):
            return ["kubectl", "--context", "c", "-n", "default"]

    executor._client = lambda: _Client()
    plain = executor.interactive_exec_argv("ignored", "pod-0", "ls", tty=True, ssh_kwargs={"ssh_user": "x"})
    assert plain == ["kubectl", "--context", "c", "-n", "default", "exec", "-i", "-t", "pod-0", "--", "sh", "-c", "ls"]

    resource = LogResource(kind="pod", namespace="ml", name="job-0-abc", uid="u", instance="i")
    native = executor.interactive_exec_argv("ignored", "serve", None, resource=resource)
    assert native[native.index("exec") - 2 : native.index("exec")] == ["-n", "ml"]
    assert native[native.index("exec") :][:5] == ["exec", "-i", "job-0-abc", "-c", "serve"]


def test_base_executor_has_no_interactive_exec():
    from sparkrun.orchestration.executors._base import Executor

    class _Bare(Executor):
        executor_name = "bare"
        run_cmd = exec_cmd = stop_cmd = logs_cmd = None

    with pytest.raises(NotImplementedError, match="bare"):
        Executor.interactive_exec_argv(_Bare.__new__(_Bare), "h", "c")


@pytest.fixture
def native_workload(tmp_path):
    """A real native workload launched by LocalExecutor, stopped afterwards."""
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    work = tmp_path / "work"
    venv = tmp_path / "venv" / "bin"
    work.mkdir()
    venv.mkdir(parents=True)
    (venv / "python").write_text("#!/bin/sh\necho venv-python\n")
    (venv / "python").chmod(0o755)
    (work / "venv.env").write_text("VIRTUAL_ENV=%s\nPATH=%s:/usr/bin:/bin\n" % (venv.parent, venv))
    executor = LocalExecutor(
        ExecutorConfig(pid_dir=str(tmp_path / "pids"), log_dir=str(tmp_path / "logs"), working_dir=str(work), env_file="venv.env")
    )
    name = "sparkrun_" + "a" * 16 + "_" + "b" * 12 + "_solo"
    launched = subprocess.run(
        ["bash", "-c", executor.run_cmd("", "sleep 60", container_name=name, env={"LAUNCH_ENV": "from-launch"})],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert launched.returncode == 0, launched.stderr
    yield executor, name, work, venv.parent
    subprocess.run(["bash", "-c", executor.stop_cmd(name)], capture_output=True, cwd=tmp_path)


def _host_side(argv):
    """Run the host-side half of an ssh argv locally (the ssh hop is ssh's business)."""
    assert argv[0] == "ssh"
    return argv[-1]


def test_local_exec_enters_the_workload_environment(native_workload, tmp_path):
    executor, name, work, venv = native_workload
    remote = _host_side(executor.interactive_exec_argv("localhost", name, "echo $LAUNCH_ENV $VIRTUAL_ENV; pwd; python"))

    result = subprocess.run(["bash", "-c", remote], capture_output=True, text=True, cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["from-launch", str(venv), str(work), "venv-python"]


def test_local_exec_shell_keeps_the_venv_ahead_of_bashrc(native_workload, tmp_path, monkeypatch):
    executor, name, _work, _venv = native_workload
    home = tmp_path / "home"
    home.mkdir()
    (home / ".bashrc").write_text("PATH=/usr/bin:$PATH\n")
    remote = _host_side(executor.interactive_exec_argv("localhost", name))

    result = subprocess.run(
        ["bash", "-c", remote],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        input="python; exit 7\n",
        env={**os.environ, "HOME": str(home)},
    )

    assert result.returncode == 7
    assert "venv-python" in result.stdout


def test_local_exec_refuses_a_stopped_workload(native_workload, tmp_path):
    executor, name, _work, _venv = native_workload
    subprocess.run(["bash", "-c", executor.stop_cmd(name)], capture_output=True, cwd=tmp_path)

    result = subprocess.run(
        ["bash", "-c", _host_side(executor.interactive_exec_argv("localhost", name, "true"))], capture_output=True, text=True
    )

    assert result.returncode == 1
    assert "is not running" in result.stderr


# ---------------------------------------------------------------------------
# api.open_exec_target
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_access(monkeypatch):
    """open_workload resolved to a two-node docker workload; the liveness sweep is recorded."""
    import sparkrun.api._logs as logs_mod
    import sparkrun.api._workload as workload_mod
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.log_source import LogSource
    from sparkrun.orchestration.executors.docker import DockerExecutor
    from sparkrun.api._workload import WorkloadAccess

    class _Runtime:
        runtime_name = "sglang"

        def log_sources(self, cluster_id, hosts, *, is_solo, scope):
            return [LogSource(host=h, container="%s_node_%d" % (cluster_id, i), role="node_%d" % i, rank=i) for i, h in enumerate(hosts)]

    access = WorkloadAccess(
        cluster_id="sparkrun_abc",
        meta={"runtime": "sglang"},
        hosts=["h0", "h1"],
        cluster=ClusterDefinition(name="lab", hosts=["h0", "h1"]),
        runtime=_Runtime(),
        executor=DockerExecutor(),
        ssh_kwargs={"ssh_user": "ops"},
        sctx=None,
        cache_dir="/tmp",
    )
    seen = {}
    monkeypatch.setattr(workload_mod, "open_workload", lambda *a, **k: seen.setdefault("open", (a, k)) and access)
    monkeypatch.setattr(logs_mod, "_verify_log_source_alive", lambda executor, sources, *a, **k: seen.setdefault("alive", list(sources)))
    return seen


def test_open_exec_target_selects_node_and_checks_only_it(fake_access):
    import sparkrun.api as api

    target = api.open_exec_target("sparkrun_abc", node=1)

    assert (target.host, target.container, target.node) == ("h1", "sparkrun_abc_node_1", 1)
    assert [s.container for s in fake_access["alive"]] == ["sparkrun_abc_node_1"]
    assert fake_access["open"][1]["operation"] == "Exec resolution"
    argv = target.argv("uptime")
    assert argv[-2:] == ["ops@h1", "docker exec -i sparkrun_abc_node_1 sh -c uptime"]


def test_open_exec_target_out_of_range_lists_nodes(fake_access):
    import sparkrun.api as api

    with pytest.raises(api.SparkrunError, match=r"has no node 2 \(nodes: 0=h0/sparkrun_abc_node_0, 1=h1/sparkrun_abc_node_1\)"):
        api.open_exec_target("sparkrun_abc", node=2)


def test_open_exec_target_wraps_native_resolution_errors(fake_access, monkeypatch):
    import dataclasses

    import sparkrun.api as api
    import sparkrun.api._workload as workload_mod

    class _NativeExecutor:
        def native_log_sources(self, *a, **k):
            raise RuntimeError("no readable pods yet")

    access = workload_mod.open_workload()  # the fixture's stub returns its access
    native = dataclasses.replace(access, meta={"native_resource": {"kind": "JobSet"}}, executor=_NativeExecutor())
    monkeypatch.setattr(workload_mod, "open_workload", lambda *a, **k: native)

    with pytest.raises(api.SparkrunError, match="Native workload resolution failed: no readable pods yet"):
        api.open_exec_target("sparkrun_abc")


@pytest.mark.parametrize(("explicit_hosts", "expected"), [(None, ["h3", "h4"]), (["h1", "h2", "h3", "h4"], ["h1", "h2", "h3", "h4"])])
def test_recipe_form_addresses_the_placed_hosts(monkeypatch, explicit_hosts, expected):
    """Logs and exec address rank i on the i-th *placed* host, not the discovery range."""
    import sparkrun.api._resolve as resolve_mod
    from sparkrun.api._resolve import WorkloadTarget
    from sparkrun.api._workload import open_workload
    from sparkrun.core.cluster_manager import ClusterDefinition

    cluster = ClusterDefinition(name="lab", hosts=["h1", "h2", "h3", "h4"])
    recipe = type("R", (), {"runtime": "sglang"})()
    target = WorkloadTarget("sparkrun_x", {"hosts": ["h3", "h4"], "runtime": "sglang"}, list(cluster.hosts), cluster, recipe, True)
    monkeypatch.setattr(resolve_mod, "resolve_workload_target", lambda **k: target)

    access = open_workload(
        None, recipe=recipe, hosts=explicit_hosts, overrides=None, cluster="lab", cache_dir=None, operation="t", sctx=None
    )

    assert access.hosts == expected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def clusters():
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.config import get_config_root

    mgr = ClusterManager(get_config_root(init_sparkrun()))
    mgr.create("lab-a", ["10.0.0.1", "10.0.0.2"], user="ops")
    mgr.create("lab-b", ["10.0.1.1"])
    mgr.set_default("lab-a")
    return mgr


@pytest.fixture
def handed(monkeypatch):
    """Record what would be handed the terminal instead of exec'ing it."""
    import sparkrun.plugins.connect.cli as cli_mod

    calls: list[list[str]] = []
    monkeypatch.setattr(cli_mod, "_hand_over", calls.append)
    return calls


def _invoke(command, args, env=None):
    return CliRunner().invoke(command, list(args), env=env)


@pytest.fixture
def ssh_cmd():
    from sparkrun.plugins.connect.cli import build_ssh_command

    return build_ssh_command("ssh")


def test_ssh_defaults_to_node_zero_of_default_cluster(clusters, handed, ssh_cmd):
    result = _invoke(ssh_cmd, [])

    assert result.exit_code == 0, result.output
    assert handed[-1][-1] == "ops@10.0.0.1"
    assert "-t" not in handed[-1]
    assert "Connecting to lab-a node 0 (10.0.0.1)" in result.output


@pytest.mark.parametrize(
    ("args", "tail"),
    [
        (["1"], ["ops@10.0.0.2"]),
        (["lab-b"], ["10.0.1.1"]),
        (["lab-b/0"], ["10.0.1.1"]),
        (["--cluster", "lab-a", "1"], ["ops@10.0.0.2"]),
        (["lab-b", "--", "ps | grep x"], ["10.0.1.1", "ps | grep x"]),
        (["--", "uptime"], ["ops@10.0.0.1", "uptime"]),
    ],
)
def test_ssh_targets(clusters, handed, ssh_cmd, args, tail):
    result = _invoke(ssh_cmd, args)

    assert result.exit_code == 0, result.output
    assert handed[-1][-len(tail) :] == tail


def test_ssh_follows_env_default(clusters, handed, ssh_cmd):
    result = _invoke(ssh_cmd, [], env={DEFAULT_CLUSTER_ENV: "lab-b"})

    assert result.exit_code == 0, result.output
    assert handed[-1][-1] == "10.0.1.1"


@pytest.mark.parametrize(
    ("args", "code", "message"),
    [
        (["lab-a/2"], 1, "has no node 2 (nodes: 0=10.0.0.1, 1=10.0.0.2)"),
        (["nope"], 1, "Cluster 'nope' not found"),
        (["lab-a", "--cluster", "lab-b"], 2, "cluster given twice"),
        (["lab-a/x"], 2, "is not a node number or CLUSTER[/NODE]"),
        (["lab-b", "nvidia-smi"], 2, "unexpected extra argument"),
    ],
)
def test_ssh_errors(clusters, handed, ssh_cmd, args, code, message):
    result = _invoke(ssh_cmd, args)

    assert result.exit_code == code
    assert message in result.output
    assert not handed


def test_ssh_refuses_an_option_shaped_host(handed, ssh_cmd):
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.config import get_config_root

    mgr = ClusterManager(get_config_root(init_sparkrun()))
    mgr.create("evil", ["10.0.0.1", "-oProxyCommand=touch /tmp/pwn"])

    result = _invoke(ssh_cmd, ["evil/1", "--", "x"])

    assert result.exit_code == 1
    assert "Refusing host" in result.output
    assert not handed


def test_ssh_without_any_cluster_explains_how_to_pick_one(handed, ssh_cmd):
    result = _invoke(ssh_cmd, [])

    assert result.exit_code == 1
    assert "No cluster given and no default cluster set" in result.output and DEFAULT_CLUSTER_ENV in result.output


def test_ssh_dry_run_prints_without_connecting(clusters, handed, ssh_cmd):
    result = _invoke(ssh_cmd, ["-n", "lab-a/1"])

    assert result.exit_code == 0, result.output
    assert not handed
    assert result.output.strip().startswith("ssh ") and result.output.strip().endswith("ops@10.0.0.2")


def test_missing_ssh_binary_is_a_clean_error(clusters, ssh_cmd, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)

    result = _invoke(ssh_cmd, [])

    assert result.exit_code == 1
    assert "ssh not found on PATH" in result.output and "OpenSSH" in result.output
    assert not isinstance(result.exception, FileNotFoundError)


@pytest.mark.parametrize(
    ("stdin", "stdout", "command", "expected"),
    [(True, True, None, True), (True, False, None, True), (True, False, "x", False), (False, True, "x", False), (True, True, "x", True)],
)
def test_tty_needs_both_ends_for_a_command(monkeypatch, stdin, stdout, command, expected):
    import sparkrun.plugins.connect.cli as cli_mod

    monkeypatch.setattr(sys.stdin, "isatty", lambda: stdin, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout, raising=False)
    assert cli_mod._wants_tty(command) is expected


def test_hand_over_execs_on_posix_and_passes_status_on_windows(monkeypatch):
    import sparkrun.plugins.connect.cli as cli_mod

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    execs = []
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(os, "execvp", lambda f, a: execs.append((f, a)))
    cli_mod._hand_over(["ssh", "h"])
    assert execs == [("ssh", ["ssh", "h"])]

    import signal

    # Recorded, not applied: ignoring SIGINT in the pytest process would make
    # every later test that interrupts something hang.
    handlers = []
    monkeypatch.setattr(signal, "signal", lambda sig, handler: handlers.append((sig, handler)))
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(subprocess, "call", lambda argv: 5)
    with pytest.raises(SystemExit) as exit_info:
        cli_mod._hand_over(["ssh", "h"])
    assert exit_info.value.code == 5
    assert handlers == [(signal.SIGINT, signal.SIG_IGN)]  # the child, not Popen's unwinding, owns Ctrl-C


def test_completion_offers_clusters_and_nodes_but_not_after_double_dash(clusters):
    from sparkrun.plugins.connect.cli import build_ssh_command

    command = build_ssh_command("ssh")
    target = next(p for p in command.params if p.name == "target")

    def complete(args, incomplete):
        ctx = command.make_context("ssh", list(args), resilient_parsing=True)
        return [item.value for item in target.type.shell_complete(ctx, target, incomplete)]

    assert complete([], "lab") == ["lab-a/", "lab-b/"]
    assert complete([], "lab-a/") == ["lab-a/0", "lab-a/1"]
    assert complete(["--"], "") == []


@pytest.fixture
def exec_cmd():
    from sparkrun.plugins.connect.cli import build_exec_command

    return build_exec_command()


@pytest.fixture
def fake_exec_api(monkeypatch):
    """Record open_exec_target calls; status reports *running* workloads."""
    import sparkrun.api as api
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy, RunningWorkload

    calls = {"running": {"sparkrun_aaaaaaaaaaaaaaaa_bbbbbbbbbbbb": "qwen"}}

    class _Target:
        workload_id, node, host, container = "sparkrun_aaaaaaaaaaaaaaaa_bbbbbbbbbbbb", 0, "10.0.0.1", "ctr"

        def argv(self, command, *, tty):
            calls["argv"] = (command, tty)
            return ["ssh", "10.0.0.1", "docker exec -i ctr sh -c %s" % (command or "shell")]

    def _open(**kwargs):
        calls["open"] = kwargs
        return _Target()

    def _status(hosts, **kwargs):
        workloads = tuple(RunningWorkload(cluster_id=c, recipe_name=n) for c, n in calls["running"].items())
        return ClusterStatus(hosts=(HostOccupancy(host=hosts[0], workloads=workloads),))

    monkeypatch.setattr(api, "open_exec_target", _open)
    monkeypatch.setattr(api, "status", _status)
    return calls


def test_exec_without_target_enters_the_sole_running_workload(clusters, handed, exec_cmd, fake_exec_api):
    result = _invoke(exec_cmd, [])

    assert result.exit_code == 0, result.output
    assert fake_exec_api["open"]["workload_id"] == "sparkrun_aaaaaaaaaaaaaaaa_bbbbbbbbbbbb"
    assert fake_exec_api["open"]["node"] == 0
    # No metadata recorded here, so the cluster's hosts are the best answer.
    assert fake_exec_api["open"]["hosts"] == ["10.0.0.1", "10.0.0.2"]
    assert "Entering sparkrun_aaaaaaaaaaaaaaaa_bbbbbbbbbbbb node 0 (ctr on 10.0.0.1)" in result.output
    assert handed


def test_exec_digits_select_a_node_of_the_sole_workload(clusters, handed, exec_cmd, fake_exec_api):
    result = _invoke(exec_cmd, ["1", "--", "nvidia-smi", "-L"])

    assert result.exit_code == 0, result.output
    assert fake_exec_api["open"]["node"] == 1
    assert fake_exec_api["argv"][0] == "nvidia-smi -L"
    assert "Entering" not in result.output


@pytest.mark.parametrize("running", [{}, {"sparkrun_a": "qwen", "sparkrun_b": "glm"}])
def test_exec_without_target_refuses_unless_exactly_one_runs(clusters, handed, exec_cmd, fake_exec_api, running):
    fake_exec_api["running"] = running

    result = _invoke(exec_cmd, [])

    assert result.exit_code == 1
    assert ("No running workloads" if not running else "Several workloads are running") in result.output
    assert "Cluster: lab-a" in result.output
    assert not handed


@pytest.mark.parametrize("args", [[], ["--cluster", "lab-a"]])
def test_exec_prefers_the_jobs_placed_hosts_over_the_cluster(clusters, handed, fake_exec_api, monkeypatch, args):
    """A job placed on a subset is addressed on that subset unless hosts are typed."""
    import sparkrun.orchestration.job_metadata as job_metadata
    from sparkrun.plugins.connect.cli import build_exec_command

    monkeypatch.setattr(job_metadata, "load_job_metadata", lambda cid, cache_dir=None: {"hosts": ["10.0.0.2"], "runtime": "sglang"})
    exec_cmd = build_exec_command()

    result = _invoke(exec_cmd, args)

    assert result.exit_code == 0, result.output
    assert fake_exec_api["open"]["hosts"] is None  # resolve_workload_target then uses the recorded hosts
    assert fake_exec_api["open"]["cluster"] == ("lab-a" if args else None)

    explicit = _invoke(exec_cmd, ["aaaaaaaaaaaaaaaa_bbbbbbbbbbbb", "--hosts", "10.0.0.1"])
    assert explicit.exit_code == 0, explicit.output
    assert fake_exec_api["open"]["hosts"] == ["10.0.0.1"]


def test_exec_workload_id_target(clusters, handed, exec_cmd, fake_exec_api):
    result = _invoke(exec_cmd, ["aaaaaaaaaaaaaaaa_bbbbbbbbbbbb/1", "-n"])

    assert result.exit_code == 0, result.output
    assert fake_exec_api["open"]["workload_id"] == "sparkrun_aaaaaaaaaaaaaaaa_bbbbbbbbbbbb"
    assert fake_exec_api["open"]["node"] == 1
    assert result.output.strip().startswith("ssh ")
    assert not handed


def test_exec_recipe_target(clusters, handed, exec_cmd, fake_exec_api, tmp_path, monkeypatch):
    recipe = tmp_path / "tiny.yaml"
    recipe.write_text("model: org/tiny\nruntime: vllm\ncontainer: img:1\n")

    result = _invoke(exec_cmd, ["%s/1" % recipe, "--port", "9000"])

    assert result.exit_code == 0, result.output
    assert fake_exec_api["open"]["recipe"].model == "org/tiny"
    assert fake_exec_api["open"]["node"] == 1
    assert fake_exec_api["open"]["cluster"] == "lab-a"
    assert fake_exec_api["open"]["overrides"]["port"] == 9000


def test_exec_api_errors_are_clean(clusters, handed, exec_cmd, monkeypatch):
    import sparkrun.api as api

    def _missing(**kwargs):
        raise api.JobNotFound("Workload x is not currently running on h0.")

    monkeypatch.setattr(api, "open_exec_target", _missing)
    result = _invoke(exec_cmd, ["aaaaaaaaaaaaaaaa_bbbbbbbbbbbb"])

    assert result.exit_code == 1
    assert "is not currently running" in result.output
    assert not handed


# ---------------------------------------------------------------------------
# Gating and wiring through the real root group
# ---------------------------------------------------------------------------


_GATING_CODE = """
import json, sys
from sparkrun.core.bootstrap import init_sparkrun
import sparkrun.core.registry as registry
registry.BOOTSTRAP_REGISTRY_URLS = []
registry.RegistryManager._clone_or_pull = lambda *args: False
init_sparkrun()
assert 'click' not in sys.modules
from click.testing import CliRunner
from sparkrun.cli import main
from sparkrun.core.cluster_manager import ClusterManager
from sparkrun.core.config import get_config_root
ClusterManager(get_config_root(init_sparkrun())).create('lab', ['10.0.0.1'])
runner = CliRunner()
root = runner.invoke(main, ['--help'])
codes = {name: runner.invoke(main, [name, '--help']).exit_code for name in ('ssh', 'connect', 'exec')}
dry = runner.invoke(main, ['ssh', '-n', 'lab', '--', 'echo', 'a b'])
alias = runner.invoke(main, ['connect', '-n', 'lab'])
print(json.dumps(dict(root=root.output, codes=codes, dry=dry.output, alias=alias.output)))
"""


@pytest.mark.parametrize(
    ("config", "ssh", "exec_"),
    [
        ("features:\\n  channel: stable\\n", False, False),
        ("features:\\n  channel: beta\\n", True, True),
        ("features:\\n  channel: alpha\\n", True, True),
        ("features:\\n  channel: stable\\n  cli.ssh: true\\n", True, False),
        ("features:\\n  channel: alpha\\n  cli.exec: false\\n", True, False),
        ("features:\\n  channel: alpha\\nplugins:\\n  enabled:\\n    connect: false\\n", False, False),
    ],
)
def test_gating_and_root_group_wiring(tmp_path, config, ssh, exec_):
    root = tmp_path / ".config/sparkrun"
    root.mkdir(parents=True)
    (root / "config.yaml").write_text(config.replace("\\n", "\n"))
    (root / "registries.yaml").write_text("registries: []\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SPARKRUN_", "ALTERNATE_", "XDG_", "STATEFUL_"))}
    env.update(HOME=str(tmp_path), SPARKRUN_NO_TELEMETRY="1", SPARKRUN_NO_EXTERNAL_PLUGINS="1", SPARKRUN_NO_INSTALLED_PLUGINS="1")
    proc = subprocess.run([sys.executable, "-c", _GATING_CODE], env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)

    assert ("  ssh " in result["root"]) is ssh
    assert "  connect " not in result["root"]  # the alias is hidden
    assert result["codes"] == {"ssh": 0 if ssh else 2, "connect": 0 if ssh else 2, "exec": 0 if exec_ else 2}
    if ssh:
        # '--' survives the root group and both words stay one argument each.
        assert result["dry"].strip().endswith("10.0.0.1 'echo '\"'\"'a b'\"'\"''")
        assert result["alias"].strip().endswith("10.0.0.1")
