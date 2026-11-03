"""One effective port override reaches identity, launch, handlers, and ensure."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from sparkrun import api
from sparkrun.api._run import plan, run
from sparkrun.core.run_handlers import RunHandler
from sparkrun.orchestration.job_metadata import generate_intent_id, derive_recipe_fingerprint
from test_benchmark_startup_collection import bench_env as bench_env


# What the combined hardware probe reports for every host in these tests: a
# detected DGX Spark with known fabric and management facts.  api.plan probes
# the candidate hosts, and with "localhost" the real probe runs on the machine
# executing the suite -- its result (this box's CX7 addresses, its GPU) would
# leak into launch metadata and vary by developer.
CANNED_PROBE_IB_INFO = {
    "IB_DETECTED": "1",
    "DETECTED_HCA_LIST": "rocep1s0f0",
    "DETECTED_NET_LIST": "enp1s0f0np0",
    "DETECTED_UCX_LIST": "rocep1s0f0:1",
    "DETECTED_IB_IPS": "192.168.99.13",
    "DETECTED_GID_INDEX": "3",
    "DETECTED_SOCKET_IFNAME": "enP7s7",
    "DETECTED_MGMT_IFACE": "enP7s7",
    "DETECTED_MGMT_IP": "10.99.0.13",
    "DETECTED_MGMT_RULE": "ssh",
    "DETECTED_MGMT_KIND": "wired",
}


def canned_probe_hosts(hosts, ssh_kwargs=None, *, dry_run=False, mgmt_interface=None):
    from sparkrun.core.hardware import default_dgx_spark_hardware

    return {host: replace(default_dgx_spark_hardware(), source="detected", ib_info=dict(CANNED_PROBE_IB_INFO)) for host in hosts}


@pytest.fixture
def run_env(bench_env, monkeypatch):
    env = bench_env
    monkeypatch.setattr("sparkrun.core.hardware_probe.probe_hosts", canned_probe_hosts)
    monkeypatch.setattr("sparkrun.api._hosts.resolve_effective_hosts", lambda *a, **kw: (["localhost"], True, [], None))
    monkeypatch.setattr("sparkrun.core.run_handlers.registered_run_handlers", lambda _: {})
    return env


@pytest.mark.parametrize("supplied_plan", [False, True])
@pytest.mark.parametrize("handler", [False, True])
def test_port_override_reaches_plan_command_and_metadata(run_env, monkeypatch, supplied_plan, handler):
    env = run_env
    options = api.RunOptions(recipe=env.recipe, hosts=("localhost",), solo=True, overrides={"port": 9001}, dry_run=True)
    planned = plan(options, sctx=env.sctx)
    default_plan = plan(replace(options, overrides={}), sctx=env.sctx)
    assert planned.intent_id == generate_intent_id(env.recipe, options.overrides) != default_plan.intent_id
    assert planned.recipe_fingerprint == derive_recipe_fingerprint(env.recipe, options.overrides) != default_plan.recipe_fingerprint
    seen = []

    def command(recipe, overrides):
        seen.append(dict(overrides))
        return env.launch.runtime.generate_command(recipe, overrides, is_cluster=False, num_nodes=1)

    def launch(**kwargs):
        env.launch.cluster_id = kwargs["cluster_id_override"]
        env.launch.serve_port = kwargs["overrides"]["port"]
        env.launch.serve_command = command(kwargs["recipe"], kwargs["overrides"])
        return env.launch

    def plugin(options, sctx, *, plan, started_at, before_start):
        return api.RunResult(
            workload_id=plan.workload_id,
            host_list=plan.host_list,
            placement=plan.placement,
            scheduler=plan.scheduler,
            runtime=plan.runtime.runtime_name,
            executor="docker",
            started_at=started_at,
            dry_run=True,
            is_solo=True,
            serve_port=options.overrides["port"],
            serve_command=command(plan.recipe, options.overrides),
            intent_id=plan.intent_id,
        )

    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", launch)
    if handler:
        monkeypatch.setattr("sparkrun.core.run_handlers.registered_run_handlers", lambda _: {"docker": RunHandler("docker", plugin)})
    result = run(options, sctx=env.sctx, plan=planned if supplied_plan else None)
    assert seen == [{"port": 9001}]
    assert result.serve_port == 9001 and "--port 9001" in result.serve_command
    assert result.intent_id == planned.intent_id
    assert result.placement_token == planned.placement_token
    assert result.recipe_fingerprint == planned.recipe_fingerprint
    assert result.timeline is not None


@pytest.mark.parametrize("running_port", [8000, 9001])
def test_ensure_matches_the_requested_port_only(run_env, monkeypatch, running_port):
    env = run_env
    options = api.RunOptions(recipe=env.recipe, hosts=("localhost",), solo=True, overrides={"port": 9001}, dry_run=True, ensure=True)
    running_intent = generate_intent_id(env.recipe, {"port": running_port})
    from sparkrun.api._intent import IntentMatch

    match = IntentMatch(intent_id=running_intent, workload_id="sparkrun_%s_aaaaaaaaaaaa" % running_intent, hosts=("localhost",))
    find = Mock(side_effect=lambda intent, *a, **kw: match if intent == running_intent else None)
    monkeypatch.setattr("sparkrun.api._intent.find_running_intent", find)
    launch = Mock(return_value=env.launch)
    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", launch)
    result = run(options, sctx=env.sctx)
    assert find.call_args.args[0] == generate_intent_id(env.recipe, {"port": 9001})
    assert result.already_running is (running_port == 9001)
    assert launch.call_count == int(running_port != 9001)
    if result.already_running:
        assert result.workload_id == match.workload_id and result.placement_token == "aaaaaaaaaaaa"
        assert result.timeline is None and result.recipe_fingerprint == ""
        assert result.launch_result is None and result.placement is None


@pytest.mark.parametrize("hook", ["pre_exec", "mods", "post_exec", "post_commands"])
@pytest.mark.parametrize("tty", [False, True])
def test_untrusted_hooks_fail_before_preparation_without_prompting(run_env, monkeypatch, capsys, hook, tty):
    env = run_env
    env.recipe.is_url_sourced = True
    setattr(env.recipe, hook, ["echo reviewed-hook"])
    options = api.RunOptions(recipe=env.recipe, hosts=("localhost",), solo=True, dry_run=True)
    # Planned dry (no host probe); the launch itself is real, since a dry run
    # executes nothing and is reported rather than refused (test below).
    planned = plan(options, sctx=env.sctx)
    options = replace(options, dry_run=False)
    prompt, preparation, launch = Mock(), Mock(), Mock()
    monkeypatch.setattr("sys.stdin.isatty", lambda: tty)
    monkeypatch.setattr("click.confirm", prompt)
    monkeypatch.setattr("sparkrun.core.execution.resolve_recipe_execution", preparation)
    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", launch)
    capsys.readouterr()
    with pytest.raises(api.TrustRejected, match="explicit authorization"):
        run(options, sctx=env.sctx, plan=planned)
    for call in (prompt, preparation, launch):
        call.assert_not_called()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("hook", ["pre_exec", "mods", "post_exec", "post_commands"])
def test_untrusted_hooks_in_a_dry_run_are_reported_not_refused(run_env, monkeypatch, capsys, caplog, hook):
    env = run_env
    env.recipe.is_url_sourced = True
    setattr(env.recipe, hook, ["echo reviewed-hook"])
    options = api.RunOptions(recipe=env.recipe, hosts=("localhost",), solo=True, dry_run=True)
    prompt, launch = Mock(), Mock(return_value=env.launch)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("click.confirm", prompt)
    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", launch)
    assert run(options, sctx=env.sctx).rc == 0
    prompt.assert_not_called()
    assert launch.call_args.kwargs["recipe_trusted"] is False
    assert "%s would require authorization (--trust)" % hook in caplog.text


@pytest.mark.parametrize("local,trust", [(True, False), (False, True)])
def test_automatic_and_explicit_recipe_trust_keep_launching(run_env, monkeypatch, local, trust):
    env = run_env
    env.recipe.is_url_sourced = not local
    env.recipe.pre_exec = ["echo reviewed-hook"]
    options = api.RunOptions(recipe=env.recipe, hosts=("localhost",), solo=True, dry_run=True, trust=trust)
    launch = Mock(return_value=env.launch)
    monkeypatch.setattr("sparkrun.core.launcher.launch_inference", launch)
    assert run(options, sctx=env.sctx).rc == 0
    assert launch.call_args.kwargs["trust"] is trust
