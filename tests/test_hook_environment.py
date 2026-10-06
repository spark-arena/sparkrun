"""Behavioral coverage for hook node metadata, mod inheritance, and cache paths."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.core.runtime_cache import RuntimeCacheSettings, build_runtime_cache_mounts
from sparkrun.orchestration import hooks
from sparkrun.orchestration.ssh import RemoteResult
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime


@pytest.fixture
def launch(tmp_path):
    recipe = Recipe.from_dict(
        {
            "name": "hook-fixture",
            "model": "org/model",
            "model_revision": "revision-1",
            "runtime": "vllm-distributed",
            "defaults": {"port": 8000},
        }
    )
    recipe.name = "hook-fixture"
    mounts = build_runtime_cache_mounts(
        runtime=VllmDistributedRuntime(),
        recipe=recipe,
        settings=RuntimeCacheSettings(key_by_image=True),
        root=str(tmp_path / "compilation cache"),
        image="fixture:v1",
        image_identity="sha256:abc",
    )
    return hooks.build_hook_launch_context(
        ["z-head", "a-worker", "b-worker"],
        recipe.build_config_chain({"port": 9000}),
        recipe=recipe,
        runtime="vllm-distributed",
        cluster_id="fixture-cluster",
        runtime_cache=mounts,
        volumes=mounts.volumes,
    )


def _docker_exec(script):
    args = shlex.split(script)
    assert args[:4] == ["docker", "exec", "--user", "root"]
    env, i = {}, 4
    while args[i] == "--env":
        key, value = args[i + 1].split("=", 1)
        env[key] = value
        i += 2
    return args[i], env, args[i + 1 :]


@pytest.fixture
def container_shell(monkeypatch):
    """Execute the actual bash/base64 payload with an isolated image env."""
    calls = []
    image_env = {key: val for key, val in os.environ.items() if key not in hooks.HOOK_ENV_KEYS}

    def run(host, script, *, dry_run=False, **kwargs):
        container, overlay, command = _docker_exec(script)
        calls.append((host, container, overlay))
        if dry_run:
            return RemoteResult(host, 0, "", "")
        result = subprocess.run(command, env={**image_env, **overlay}, capture_output=True, text=True, timeout=10)
        return RemoteResult(host, result.returncode, result.stdout, result.stderr)

    monkeypatch.setattr("sparkrun.orchestration.primitives.run_script_on_host", run)
    return image_env, calls


@pytest.mark.parametrize(
    "host,rank,role",
    [
        ("z-head", "0", "head"),
        ("a-worker", "1", "worker"),
        ("b-worker", "2", "worker"),
    ],
)
def test_node_order_and_full_launch_context(launch, host, rank, role):
    env = hooks.build_hook_env(launch, "pre_exec", host=host, container_name="worker")
    expected = dict(
        NODE_RANK=rank,
        NODE_ROLE=role,
        IS_HEAD="1" if rank == "0" else "0",
        NUM_NODES="3",
        NODE_HOST=host,
        HEAD_HOST="z-head",
        CONTAINER_NAME="worker",
        MODEL="org/model",
        MODEL_REVISION="revision-1",
        RECIPE_NAME="hook-fixture",
        RUNTIME="vllm-distributed",
        PORT="9000",
    )
    for key, value in expected.items():
        assert env["SPARKRUN_" + key] == value
    assert set(env) == hooks.HOOK_ENV_KEYS


def test_solo_is_head_but_control_is_not(launch):
    solo = replace(launch, hosts=("localhost",))
    env = hooks.build_hook_env(solo, "pre_exec", host="localhost", container_name="c")
    assert (env["SPARKRUN_NODE_RANK"], env["SPARKRUN_NODE_ROLE"], env["SPARKRUN_IS_HEAD"]) == ("0", "solo", "1")
    control = hooks.build_hook_env(solo, "post_commands", host="localhost", container_name="c")
    assert (control["SPARKRUN_NODE_RANK"], control["SPARKRUN_NODE_ROLE"], control["SPARKRUN_IS_HEAD"]) == ("", "control", "0")
    assert control["SPARKRUN_NODE_HOST"] == control["SPARKRUN_CONTAINER_NAME"] == ""


def test_post_cache_namespaces_and_endpoint(launch):
    launch = replace(launch, head_ip="10.0.0.1", base_url="http://10.0.0.1:9000/v1")
    inside = hooks.build_hook_env(launch, "post_exec", host="z-head", container_name="c")
    outside = hooks.build_hook_env(launch, "post_commands")
    assert inside["SPARKRUN_NUM_NODES"] == outside["SPARKRUN_NUM_NODES"] == "3"
    assert inside["SPARKRUN_RUNTIME_CACHE_DIR"] == "/cache/runtime"
    assert outside["SPARKRUN_RUNTIME_CACHE_DIR"] == ""
    for env in (inside, outside):
        assert env["SPARKRUN_RUNTIME_CACHE_ENABLED"] == "1"
        assert env["SPARKRUN_RUNTIME_CACHE_HOST_DIR"] == launch.runtime_cache.leaf
        assert env["SPARKRUN_RUNTIME_CACHE_HOST"] == "z-head"
        assert env["SPARKRUN_HEAD_IP"] == "10.0.0.1"
        assert env["SPARKRUN_BASE_URL"] == "http://10.0.0.1:9000/v1"
    pre = hooks.build_hook_env(launch, "pre_exec", host="a-worker")
    assert pre["SPARKRUN_HEAD_IP"] == pre["SPARKRUN_BASE_URL"] == ""
    assert pre["SPARKRUN_RUNTIME_CACHE_HOST"] == "a-worker"


@pytest.mark.parametrize("mount_change", ["disabled", "moved", "conflicting"])
def test_unavailable_cache_is_explicit(launch, mount_change):
    if mount_change == "disabled":
        launch = replace(launch, runtime_cache=None)
    elif mount_change == "moved":
        launch = replace(launch, volumes={launch.runtime_cache.leaf: "/elsewhere"})
    else:
        launch = replace(launch, volumes={**launch.volumes, "/other": "/cache/runtime"})
    env = hooks.build_hook_env(launch, "pre_exec", host="z-head")
    assert env["SPARKRUN_RUNTIME_CACHE_ENABLED"] == "0"
    assert env["SPARKRUN_RUNTIME_CACHE_DIR"] == env["SPARKRUN_RUNTIME_CACHE_HOST_DIR"] == env["SPARKRUN_RUNTIME_CACHE_HOST"] == ""


@pytest.mark.parametrize(
    "settings",
    [
        RuntimeCacheSettings(),
        RuntimeCacheSettings(key_by_model=False),
        RuntimeCacheSettings(key_by_image=True),
        RuntimeCacheSettings(key_by_image=True, key_by_model=False),
    ],
)
def test_cache_keys_are_reused_without_reconstruction(tmp_path, settings):
    recipe = Recipe.from_dict({"model": "org/model", "model_revision": "pinned"})
    mounts = build_runtime_cache_mounts(
        runtime=VllmDistributedRuntime(),
        recipe=recipe,
        settings=settings,
        root=str(tmp_path / "custom root"),
        image="fixture:v2",
        image_identity="sha256:123",
    )
    context = hooks.build_hook_launch_context(["h"], runtime_cache=mounts)
    env = hooks.build_hook_env(context, "pre_exec", host="h")
    assert env["SPARKRUN_RUNTIME_CACHE_HOST_DIR"] == mounts.leaf
    assert mounts.volumes[env["SPARKRUN_RUNTIME_CACHE_HOST_DIR"]] == env["SPARKRUN_RUNTIME_CACHE_DIR"]


def test_target_subset_keeps_rank_and_repeated_container_names(launch, container_shell):
    hooks.run_pre_exec([("b-worker", "worker"), ("a-worker", "worker")], ["true"], {}, trust=True, launch_context=launch)
    assert [c[2]["SPARKRUN_NODE_RANK"] for c in container_shell[1]] == ["2", "1"]
    assert all(c[2]["SPARKRUN_NUM_NODES"] == "3" for c in container_shell[1])


def test_direct_caller_deduplicates_hosts(container_shell):
    hooks.run_pre_exec([("h", "c1"), ("h", "c2"), ("w", "c")], ["true"], {}, trust=True)
    assert [c[2]["SPARKRUN_NODE_RANK"] for c in container_shell[1]] == ["0", "0", "1"]
    assert all(c[2]["SPARKRUN_NUM_NODES"] == "2" for c in container_shell[1])


def _capture_env_command(output):
    return "python3 -c %s" % shlex.quote(
        f"import json, os; from pathlib import Path; Path({str(output)!r}).write_text(json.dumps(dict(os.environ)))"
    )


def test_metadata_round_trips_as_data(launch, container_shell, tmp_path):
    output, injected = tmp_path / "captured.json", tmp_path / "must-not-exist"
    value = "quote ' \" space\n$(touch %s) %ctouch %s%c {model} = end" % (injected, 96, injected, 96)
    context = replace(launch, cluster_id=value, model=value)
    hooks.run_pre_exec([("z-head", "c")], [_capture_env_command(output)], {}, trust=True, launch_context=context)
    observed = json.loads(output.read_text())
    assert observed["SPARKRUN_CLUSTER_ID"] == observed["SPARKRUN_MODEL"] == value
    assert not injected.exists()


@pytest.mark.parametrize(
    "expression",
    [
        "${SPARKRUN_NODE_RANK}",
        "${SPARKRUN_NODE_RANK:-unset}",
        "${SPARKRUN_NODE_RANK:?missing}",
        "${#SPARKRUN_NODE_RANK}",
        "${SPARKRUN_NODE_RANK%0}",
        "${!SPARKRUN_NODE_RANK}",
    ],
)
def test_shell_metadata_survives_colliding_template_keys(expression):
    context = {expression[2:-1]: "WRONG", "model": "org/model", "nested": expression}
    result = hooks.render_hook_command(f'echo {expression} {{nested}} \'{{"model":"{{model}}"}}\'', context)
    assert result == f'echo {expression} {expression} \'{{"model":"org/model"}}\''


def test_plain_config_placeholder_and_nested_shell_default_still_render():
    context = {"SPARKRUN_MODEL": "config-model", "model": "default-model"}
    result = hooks.render_hook_command('echo {SPARKRUN_MODEL} "${SPARKRUN_MODEL:-{model}}"', context)
    assert result == 'echo config-model "${SPARKRUN_MODEL:-default-model}"'


def test_rank_selected_mod_inherits_cache_and_child_environment(launch, container_shell, monkeypatch, tmp_path):
    from sparkrun.core import mods

    image_env, calls = container_shell
    cache_path = str(tmp_path / "persistent cache")
    mounts = replace(launch.runtime_cache, leaf=cache_path, volumes={cache_path: cache_path})
    context = replace(launch, runtime_cache=mounts, volumes=mounts.volumes)
    image_env.update(TRITON_CACHE_DIR=cache_path + "/custom-triton", SPARKRUN_NODE_RANK="stale")
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.sh").write_text(
        "#!/bin/bash\nset -eu\n"
        'if [ "$SPARKRUN_NODE_RANK" != "1" ]; then exit 0; fi\n'
        'mkdir -p "$SPARKRUN_RUNTIME_CACHE_DIR/mods/example"\n'
        'bash -c \'printf "%s:%s" "$SPARKRUN_NODE_RANK" "$TRITON_CACHE_DIR" '
        '> "$SPARKRUN_RUNTIME_CACHE_DIR/mods/example/marker"\'\n'
    )
    monkeypatch.setattr(mods, "_CONTAINER_MODS_BASE", str(tmp_path / "workload-mods"))
    monkeypatch.setattr(
        hooks, "_run_copy_command", lambda host, container, cmd, *a, **kw: shutil.copytree(cmd["copy"], cmd["dest"], dirs_exist_ok=True)
    )
    entries = mods._build_pre_exec_entries(mods.ResolvedMod("example", str(source), None))
    hooks.run_pre_exec([(h, "worker") for h in context.hosts], entries, {}, trust=True, launch_context=context)
    assert (tmp_path / "persistent cache/mods/example/marker").read_text() == "1:" + image_env["TRITON_CACHE_DIR"]
    assert [c[2]["SPARKRUN_NODE_RANK"] for c in calls] == ["0", "1", "2"]
    assert image_env["SPARKRUN_NODE_RANK"] == "stale"
    assert all("TRITON_CACHE_DIR" not in c[2] for c in calls)
    hooks.run_pre_exec([("other", "solo")], ["true"], {}, trust=True)
    assert calls[-1][2]["SPARKRUN_NODE_RANK"] == "0"
    assert calls[-1][2]["SPARKRUN_CLUSTER_ID"] == ""
    assert calls[-1][2]["SPARKRUN_RUNTIME_CACHE_ENABLED"] == "0"


def test_control_hook_keeps_parent_env_and_clears_stale_metadata(launch, monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKRUN_NODE_RANK", "stale")
    monkeypatch.setenv("SPARKRUN_RUNTIME_CACHE_DIR", "/wrong")
    monkeypatch.setenv("SPARKRUN_CACHE_DIR", "/control-cache-setting")
    output = tmp_path / "control.json"
    hooks.run_post_commands([_capture_env_command(output)], {}, trust=True, launch_context=launch)
    observed = json.loads(output.read_text())
    assert observed["SPARKRUN_NODE_RANK"] == observed["SPARKRUN_RUNTIME_CACHE_DIR"] == ""
    assert observed["SPARKRUN_RUNTIME_CACHE_HOST_DIR"] == launch.runtime_cache.leaf
    assert observed["SPARKRUN_RUNTIME_CACHE_HOST"] == "z-head"
    assert observed["SPARKRUN_CACHE_DIR"] == "/control-cache-setting"
    assert os.environ["SPARKRUN_NODE_RANK"] == "stale"
    assert os.environ["SPARKRUN_RUNTIME_CACHE_DIR"] == "/wrong"


def test_dry_run_reports_target_without_execution(launch, container_shell, caplog, tmp_path):
    caplog.set_level("INFO", logger=hooks.__name__)
    container_shell[0]["PRIVATE_TOKEN"] = "do-not-log-me"
    marker = tmp_path / "must-not-exist"
    command = "touch " + shlex.quote(str(marker))
    hooks.run_pre_exec([("b-worker", "worker")], [command], {}, trust=True, dry_run=True, launch_context=launch)
    hooks.run_post_commands([command], {}, trust=True, dry_run=True, launch_context=launch)
    assert not marker.exists()
    assert "host=b-worker rank=2/3" in caplog.text
    assert launch.runtime_cache.leaf in caplog.text
    assert "do-not-log-me" not in caplog.text


def test_scope_restores_after_nested_failure_and_isolates_threads(launch):
    assert hooks.current_hook_launch_context() is None
    with hooks.hook_launch_scope(launch):
        with pytest.raises(RuntimeError), hooks.hook_launch_scope(replace(launch, cluster_id="inner")):
            assert hooks.current_hook_launch_context().cluster_id == "inner"
            raise RuntimeError("fixture")
        assert hooks.current_hook_launch_context() is launch
        barrier = Barrier(2)

        def worker(name):
            assert hooks.current_hook_launch_context() is None
            with hooks.hook_launch_scope(replace(launch, cluster_id=name)):
                barrier.wait(timeout=5)
                return hooks.current_hook_launch_context().cluster_id

        with ThreadPoolExecutor(max_workers=2) as pool:
            assert set(pool.map(worker, ["one", "two"])) == {"one", "two"}
        assert hooks.current_hook_launch_context() is launch
    assert hooks.current_hook_launch_context() is None


def test_legacy_runtime_override_receives_scoped_context(launch, container_shell):
    from sparkrun.core.scheduler import RankAssignment, RankSlot
    from _runtime_fixtures import StubRuntime
    from sparkrun.runtimes._cluster_ops import ClusterContext, run_pre_serve_hooks

    class LegacyRuntime(StubRuntime):
        def _pre_serve(self, hosts_containers, ssh_kwargs, dry_run, recipe=None, config_chain=None, trust=False, cache_dir=None):
            super()._pre_serve(hosts_containers, ssh_kwargs, dry_run, recipe, config_chain, trust, cache_dir)

    runtime = LegacyRuntime()
    recipe = Recipe.from_dict({"model": "m", "pre_exec": ["true"]})
    ctx = ClusterContext.build(
        runtime,
        list(launch.hosts),
        "image",
        "legacy-cluster",
        {},
        "/hf",
        None,
        False,
        recipe=recipe,
        runtime_cache=launch.runtime_cache,
        placement=RankAssignment(
            by_rank=tuple(RankSlot(host=h, local_gpu=gpu) for h in launch.hosts for gpu in (0, 1)),
            hosts_used=launch.hosts,
        ),
    )
    run_pre_serve_hooks(runtime, ctx, [("b-worker", "c")], recipe, {}, trust=True)
    env = container_shell[1][0][2]
    assert env["SPARKRUN_NODE_RANK"] == "2"
    assert env["SPARKRUN_CLUSTER_ID"] == "legacy-cluster"
    assert env["SPARKRUN_RUNTIME_CACHE_HOST_DIR"] == launch.runtime_cache.leaf
    assert ctx.runtime_cache is launch.runtime_cache
    assert hooks.current_hook_launch_context() is None


@pytest.mark.parametrize(
    "runtime_name", ["solo", "vllm-distributed", "sglang", "vllm-ray", "llama-cpp", "trtllm", "vllm-distributed-hybrid", "sglang-hybrid"]
)
def test_every_launch_route_injects_node_and_cache_context(runtime_name, launch, monkeypatch):
    from _runtime_fixtures import StubRuntime
    from sparkrun.orchestration.comm_env import ClusterCommEnv
    from sparkrun.runtimes import _cluster_ops
    from sparkrun.runtimes.llama_cpp import LlamaCppRuntime
    from sparkrun.runtimes.sglang import SglangRuntime
    from sparkrun.runtimes.trtllm import TrtllmRuntime
    from sparkrun.runtimes.vllm_ray import VllmRayRuntime

    factories = {
        "solo": StubRuntime,
        "vllm-distributed": VllmDistributedRuntime,
        "sglang": SglangRuntime,
        "vllm-ray": VllmRayRuntime,
        "llama-cpp": LlamaCppRuntime,
        "trtllm": TrtllmRuntime,
    }
    hybrid = runtime_name.endswith("-hybrid")
    runtime = factories[runtime_name.removesuffix("-hybrid")]()
    hosts = list(launch.hosts[:1] if runtime_name == "solo" else launch.hosts)
    if hybrid:
        hosts.append("c-worker")
    recipe = Recipe.from_dict(
        {
            "model": "org/model",
            "runtime": runtime.runtime_name,
            "defaults": {"port": 9000, "tensor_parallel": 2 if hybrid else len(hosts), "data_parallel": 2 if hybrid else 1},
            "pre_exec": ["echo node-setup"],
        }
    )
    calls = []

    def remote(host, script, **kwargs):
        if script.startswith("docker exec --user root"):
            calls.append((host, *_docker_exec(script)))
        return RemoteResult(host, 0, "10.0.0.1\n", "")

    for name in ("cleanup_ranked_containers", "cleanup_named_containers"):
        monkeypatch.setattr(_cluster_ops, name, lambda *a, **k: None)
    monkeypatch.setattr(_cluster_ops, "launch_containers_parallel", lambda *a, **k: 0)
    monkeypatch.setattr(_cluster_ops, "find_port", lambda ctx, host, port: port)
    monkeypatch.setattr(_cluster_ops, "detect_head_ip", lambda ctx: "10.0.0.1")
    monkeypatch.setattr(_cluster_ops, "resolve_hosts_for_init", lambda ctx, head_ip: ctx.hosts)
    monkeypatch.setattr(_cluster_ops, "detect_ib_with_ips", lambda *a, **k: (ClusterCommEnv.empty(), {}, {}))
    monkeypatch.setattr(_cluster_ops, "resolve_comm_env", lambda *a, **k: ClusterCommEnv.empty())
    monkeypatch.setattr("sparkrun.orchestration.primitives.run_script_on_host", remote)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_script", remote)
    monkeypatch.setattr("sparkrun.orchestration.ssh.run_remote_command", remote)
    monkeypatch.setattr("sparkrun.orchestration.primitives.detect_host_ip", lambda *a, **k: "10.0.0.1")
    monkeypatch.setattr("sparkrun.orchestration.primitives.wait_for_port", lambda *a, **k: True)
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda *a, **k: {})

    rc = runtime.run(
        hosts=hosts,
        image="fixture",
        serve_command="echo serve",
        recipe=recipe,
        overrides={"port": 9001},
        cluster_id="fixture-run",
        env={"TRITON_CACHE_DIR": "/custom"},
        dry_run=True,
        trust=True,
        comm_env=ClusterCommEnv.empty(),
        runtime_cache=launch.runtime_cache,
        backends={},
    )
    assert rc == 0
    assert len(calls) == len(hosts)
    assert {c[0]: c[2]["SPARKRUN_NODE_RANK"] for c in calls} == {h: str(i) for i, h in enumerate(hosts)}
    for _, _, env, _ in calls:
        assert env["SPARKRUN_CLUSTER_ID"] == "fixture-run"
        assert env["SPARKRUN_RUNTIME"] == runtime.runtime_name
        assert env["SPARKRUN_PORT"] == "9001"
        assert env["SPARKRUN_RUNTIME_CACHE_HOST_DIR"] == launch.runtime_cache.leaf
        assert env["SPARKRUN_RUNTIME_CACHE_DIR"] == "/cache/runtime"
        assert "TRITON_CACHE_DIR" not in env


def test_capture_scope_does_not_leak_between_launches(launch):
    with hooks.capture_hook_launch_contexts() as outer:
        with hooks.hook_launch_scope(launch):
            with pytest.raises(RuntimeError), hooks.capture_hook_launch_contexts() as inner:
                with hooks.hook_launch_scope(replace(launch, cluster_id="inner")):
                    raise RuntimeError("inner launch failed")
    assert outer == [launch]
    assert [ctx.cluster_id for ctx in inner] == ["inner"]
    with hooks.hook_launch_scope(replace(launch, cluster_id="unrecorded")):
        pass
    assert outer == [launch]
    with hooks.capture_hook_launch_contexts() as fresh:
        assert fresh == []


def test_dry_run_endpoint_placeholders_are_not_exported(launch):
    env = hooks.build_hook_env(
        replace(launch, head_ip="<HEAD_IP>", base_url="http://<HEAD_IP>:9000/v1"),
        "post_commands",
    )
    assert env["SPARKRUN_HEAD_IP"] == env["SPARKRUN_BASE_URL"] == ""
