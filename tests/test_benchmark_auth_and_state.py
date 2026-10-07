"""Credentials stay ephemeral and checkpoints fail closed across real API flows."""

from dataclasses import replace
import logging
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
import yaml

from sparkrun.api import benchmark, resume_benchmark, BenchmarkFailed, SparkrunError, ResumeMode
from sparkrun.benchmarking.run_state import BenchmarkRunState, DEFAULT_BENCHMARK_TIMEOUT
from sparkrun.core.benchmark_integrations import BenchmarkIntegration, register_benchmark_integration
from test_benchmark_startup_collection import bench_env as bench_env
from test_benchmark_api_contract import scheduled_env as scheduled_env


@pytest.fixture(autouse=True)
def integration_registry(monkeypatch):
    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("source", ["environment", "recipe", "auth_token", "literal_api_key", "literal_auth_token"])
def test_credentials_only_reach_command_execution(request, scheduled, source, monkeypatch, caplog):
    env = request.getfixturevalue("scheduled_env" if scheduled else "bench_env")
    secret = "review-only-secret-84712"
    if source == "environment":
        env.sctx.variables.set("TEST_BENCH_KEY", secret)
    else:
        monkeypatch.setattr(env.launch.runtime, "resolve_api_key", lambda *a: secret)
        if source.startswith("literal_"):
            flag = "--api-key" if source == "literal_api_key" else "--auth-token"
            env.recipe.command = "serve --model {model} %s='%s'" % (flag, secret)
        else:
            env.recipe.defaults[source if source == "auth_token" else "api_key"] = secret
    env.recipe.metadata.update(model_params=123, num_layers=2)
    snapshots, received = [], []
    register_benchmark_integration(BenchmarkIntegration("audit", on_complete=lambda ctx: snapshots.append(ctx.result)))
    original = env.fw.build_benchmark_command.side_effect

    def command(*args, **kwargs):
        values = args[2] if len(args) > 2 else kwargs["args"]
        received.append(dict(values))
        cmd = original(*args, **kwargs)
        return [*cmd[:-1], cmd[-1] + "; import sys; print(%r, file=sys.stderr)" % secret]

    env.fw.build_benchmark_command.side_effect = command
    caplog.set_level(logging.INFO)
    events = []
    result = benchmark(
        replace(
            env.options,
            api_key_env="TEST_BENCH_KEY" if source == "environment" else None,
            integrations={"audit": {}},
            progress_callback=events.append,
        ),
        sctx=env.sctx,
    )
    assert received and received[0]["api_key"] == secret
    assert "api_key" not in env.fw.build_task_list.call_args.args[0]
    for view in (result.metadata, result.results, snapshots[0].benchmark_args, snapshots[0].provenance, snapshots[0].recipe_yaml, events):
        assert secret not in str(view)
    assert secret not in caplog.text
    exported = yaml.safe_load(Path(result.outputs["yaml"]).read_text())
    assert snapshots[0].provenance["model"] == exported["sparkrun_benchmark"]["model"]
    assert exported["sparkrun_benchmark"]["model"]["params"] == 123
    assert secret not in Path(result.outputs["yaml"]).read_text()
    if scheduled:
        state = BenchmarkRunState.load(result.benchmark_id, str(env.sctx.config.cache_dir), strict=True)
        assert state.api_key_required
        assert state.api_key_env == ("TEST_BENCH_KEY" if source == "environment" else None)
        for path in Path(result.state_dir).rglob("*"):
            if path.is_file():
                assert secret not in path.read_text(), path


def _interrupt_measurement(env, **options):
    original = env.fw.build_benchmark_command.side_effect
    env.fw.build_benchmark_command.side_effect = lambda *a, **kw: [sys.executable, "-c", "raise SystemExit(1)"]
    with pytest.raises(BenchmarkFailed, match="incomplete"):
        benchmark(replace(env.options, export_files=False, **options), sctx=env.sctx)
    env.fw.build_benchmark_command.side_effect = original
    return next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))


@pytest.mark.parametrize("override", [False, True])
def test_resume_reacquires_auth_and_preserves_or_overrides_execution_policy(scheduled_env, monkeypatch, override):
    import sparkrun.benchmarking.scheduler as scheduler

    env = scheduled_env
    env.sctx.variables.set("TEST_BENCH_KEY", "old-review-secret")
    path = _interrupt_measurement(env, api_key_env="TEST_BENCH_KEY", timeout=7, exit_on_first_fail=True)
    env.sctx.variables.set("TEST_BENCH_KEY", "new-review-secret")
    original = scheduler.run_schedule
    policies = []

    def observe(**kwargs):
        policies.append((kwargs["timeout"], kwargs["exit_on_first_fail"]))
        return original(**kwargs)

    monkeypatch.setattr(scheduler, "run_schedule", observe)
    original_command = env.fw.build_benchmark_command.side_effect
    received = []

    def command(*args, **kwargs):
        received.append(args[2]["api_key"])
        return original_command(*args, **kwargs)

    env.fw.build_benchmark_command.side_effect = command
    kwargs = {"timeout": 13, "exit_on_first_fail": False} if override else {}
    result = resume_benchmark(path.parent.name, sctx=env.sctx, export_files=False, **kwargs)
    assert result.success and received == ["new-review-secret"]
    assert policies == ([(13, False)] if override else [(7, True)])
    saved = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir), strict=True)
    assert (saved.timeout, saved.exit_on_first_fail) == policies[0]
    assert "review-secret" not in path.read_text()


@pytest.mark.parametrize("source", ["environment", "recipe", "overrides", "auth_token", "literal_api_key", "literal_auth_token"])
def test_credential_rotation_does_not_change_measurement_identity(scheduled_env, monkeypatch, source):
    env = scheduled_env
    options = replace(env.options, export_files=False)

    def rotate(secret):
        nonlocal options
        if source == "environment":
            env.sctx.variables.set("TEST_BENCH_KEY", secret)
            options = replace(options, api_key_env="TEST_BENCH_KEY")
        else:
            monkeypatch.setattr(env.launch.runtime, "resolve_api_key", lambda *a: secret)
            if source == "recipe":
                env.recipe.defaults["api_key"] = secret
                env.recipe.env["OPENAI_API_KEY"] = secret
            elif source == "auth_token":
                env.recipe.defaults["auth_token"] = secret
            elif source.startswith("literal_"):
                flag = "--api-key" if source == "literal_api_key" else "--auth-token"
                env.recipe.command = "serve --model {model} %s %s" % (flag, secret)
            else:
                options = replace(options, overrides={**options.overrides, "api_key": secret})

    rotate("old-review-secret")
    first = benchmark(options, sctx=env.sctx)
    rotate("new-review-secret")
    second = benchmark(replace(options, resume=ResumeMode.IF_EXISTS), sctx=env.sctx)
    assert second.benchmark_id == first.benchmark_id and second.resumed
    env.fw.build_benchmark_command.assert_called_once()
    if source == "recipe":
        assert env.recipe.defaults["api_key"] == "new-review-secret"
        assert env.recipe.env["OPENAI_API_KEY"] == "new-review-secret"


def test_legacy_publication_state_is_sanitized_without_requiring_auth(scheduled_env, monkeypatch):
    env = scheduled_env
    first = benchmark(replace(env.options, export_files=False), sctx=env.sctx)
    path = Path(first.state_dir) / "state.yaml"
    data = yaml.safe_load(path.read_text())
    secret = "legacy-review-secret"
    data["base_args"]["api_key"] = secret
    data["extras"]["benchmark_integrations"] = {
        "arena": {
            "settings": {"local_test": True},
            "data": {
                "metadata_json": {"benchmark": {"args": {"api_key": secret}}},
                "effective_recipe_text": "defaults:\n  api_key: " + secret + "\n  port: 8000\ncommand: serve --auth-token=" + secret,
            },
        }
    }
    path.write_text(yaml.safe_dump(data))
    views = []
    from sparkrun.plugins.sparkarena.integration import bind

    monkeypatch.setattr("sparkrun.core.benchmark_integrations._INTEGRATIONS", {})
    register_benchmark_integration(
        BenchmarkIntegration("arena", on_bind=bind, on_complete=lambda ctx: views.append((ctx.data, ctx.result)))
    )
    result = resume_benchmark(first.benchmark_id, sctx=env.sctx)
    assert result.success and secret not in str(views) and secret not in str(result.metadata)
    assert secret not in path.read_text()


def test_legacy_authenticated_measurement_requires_a_current_credential(scheduled_env):
    env = scheduled_env
    path = _interrupt_measurement(env)
    data = yaml.safe_load(path.read_text())
    data["base_args"]["api_key"] = "legacy-review-secret"
    for key in ("timeout", "exit_on_first_fail", "api_key_env", "api_key_required", "schema_version"):
        data.pop(key, None)
    path.write_text(yaml.safe_dump(data))
    original = path.read_bytes()
    with pytest.raises(BenchmarkFailed, match="requires authentication"):
        resume_benchmark(path.parent.name, sctx=env.sctx)
    assert path.read_bytes() == original
    env.sctx.variables.set("TEST_BENCH_KEY", "current-review-secret")
    assert resume_benchmark(path.parent.name, sctx=env.sctx, api_key_env="TEST_BENCH_KEY", export_files=False).success
    state = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir), strict=True)
    assert state.timeout == DEFAULT_BENCHMARK_TIMEOUT and not state.exit_on_first_fail
    assert "review-secret" not in path.read_text()


@pytest.mark.parametrize("mode", ["implicit", "by_id"])
@pytest.mark.parametrize("bad", ["yaml", "empty", "schedule", "extras", "hosts", "indices", "version", "identity", "read_error"])
def test_unusable_checkpoint_fails_before_work_and_is_preserved(scheduled_env, monkeypatch, mode, bad):
    env = scheduled_env
    first = benchmark(replace(env.options, export_files=False), sctx=env.sctx)
    path = Path(first.state_dir) / "state.yaml"
    data = yaml.safe_load(path.read_text())
    changes = {
        "schedule": {"schedule": "bad"},
        "extras": {"extras": []},
        "hosts": {"host_list": "bad"},
        "indices": {"completed_indices": [True]},
        "version": {"schema_version": 999},
        "identity": {"benchmark_id": "other"},
    }
    data.update(changes.get(bad, {}))
    path.write_text("[broken" if bad == "yaml" else "" if bad == "empty" else yaml.safe_dump(data))
    before = {p: p.read_bytes() for p in Path(first.state_dir).rglob("*") if p.is_file()}
    if bad == "read_error":
        import builtins

        original = builtins.open

        def deny(file, *a, **kw):
            if Path(file) == path:
                raise PermissionError("cannot read saved benchmark")
            return original(file, *a, **kw)

        monkeypatch.setattr(builtins, "open", deny)
    env.run.reset_mock()
    env.fw.build_benchmark_command.reset_mock()
    bound = Mock()
    register_benchmark_integration(BenchmarkIntegration("audit", on_bind=bound))
    with pytest.raises(SparkrunError, match="Cannot read benchmark state"):
        if mode == "implicit":
            benchmark(replace(env.options, export_files=False, resume=ResumeMode.IF_EXISTS, integrations={"audit": {}}), sctx=env.sctx)
        else:
            resume_benchmark(first.benchmark_id, sctx=env.sctx, integrations={"audit": {}})
    env.run.assert_not_called()
    env.fw.build_benchmark_command.assert_not_called()
    bound.assert_not_called()
    assert before == {p: p.read_bytes() for p in Path(first.state_dir).rglob("*") if p.is_file()}


@pytest.mark.parametrize("dry_run", [False, True])
def test_explicit_fresh_can_recover_without_decoding_bad_state(scheduled_env, dry_run):
    env = scheduled_env
    first = benchmark(replace(env.options, export_files=False), sctx=env.sctx)
    path = Path(first.state_dir) / "state.yaml"
    path.write_text("[broken")
    result = benchmark(replace(env.options, export_files=False, resume=ResumeMode.FRESH, dry_run=dry_run), sctx=env.sctx)
    assert result.success
    if dry_run:
        assert path.read_text() == "[broken"
    else:
        assert BenchmarkRunState.load(first.benchmark_id, str(env.sctx.config.cache_dir), strict=True)


def test_short_legacy_keys_do_not_change_state_identity_or_measurement_data(scheduled_env):
    env = scheduled_env
    first = benchmark(replace(env.options, export_files=False), sctx=env.sctx)
    path = Path(first.state_dir) / "state.yaml"
    data = yaml.safe_load(path.read_text())
    data["base_args"]["api_key"] = "a"
    path.write_text(yaml.safe_dump(data))
    loaded = BenchmarkRunState.load(first.benchmark_id, str(env.sctx.config.cache_dir), strict=True)
    assert loaded.benchmark_id == first.benchmark_id
    assert loaded.recipe_qualified_name == data["recipe_qualified_name"]
    assert "api_key" not in loaded.base_args
    assert resume_benchmark(first.benchmark_id, sctx=env.sctx).results == first.results


def test_preview_does_not_resolve_credentials(scheduled_env, monkeypatch):
    env = scheduled_env
    lookup = Mock(side_effect=AssertionError("preview cannot require an execution credential"))
    monkeypatch.setattr("sparkrun.api._benchmark.resolve_credentials", lookup)
    assert benchmark(replace(env.options, dry_run=True, api_key_env="UNSET_KEY"), sctx=env.sctx).success
    lookup.assert_not_called()


@pytest.mark.parametrize("failure", ["timeout", "interrupt", "log_write"])
def test_authenticated_subprocess_failures_close_redacted_log_and_remain_resumable(scheduled_env, monkeypatch, failure):
    import builtins
    import subprocess

    env = scheduled_env
    secret = "review-subprocess-secret"
    env.sctx.variables.set("TEST_BENCH_KEY", secret)
    env.fw.build_benchmark_command.side_effect = lambda *a, **kw: [
        sys.executable,
        "-c",
        "import time; print(%r, flush=True); time.sleep(%d)" % (secret, 0 if failure == "log_write" else 30),
    ]
    if failure == "interrupt":
        original = subprocess.Popen

        def spawn(*args, **kwargs):
            proc = original(*args, **kwargs)
            poll = proc.poll
            first = True

            def interrupt_once(*a, **kw):
                nonlocal first
                if first:
                    first = False
                    raise KeyboardInterrupt()
                return poll(*a, **kw)

            proc.poll = interrupt_once
            return proc

        monkeypatch.setattr("sparkrun.benchmarking.scheduler.subprocess.Popen", spawn)
    elif failure == "log_write":
        original_open = builtins.open

        class FailedLog:
            def __init__(self, handle):
                self.handle = handle

            def write(self, value):
                raise OSError("log storage unavailable")

            def flush(self):
                pass

            def close(self):
                self.handle.close()

        def open_log(path, *a, **kw):
            handle = original_open(path, *a, **kw)
            return FailedLog(handle) if str(path).endswith(".log") else handle

        monkeypatch.setattr(builtins, "open", open_log)
    error = KeyboardInterrupt if failure == "interrupt" else BenchmarkFailed
    with pytest.raises(error):
        benchmark(replace(env.options, timeout=1, api_key_env="TEST_BENCH_KEY", export_files=False), sctx=env.sctx)
    path = next(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))
    state = BenchmarkRunState.load(path.parent.name, str(env.sctx.config.cache_dir), strict=True)
    assert not state.completed_indices
    assert not (path.parent / "run.lock").exists()
    env.stop.assert_called_once()
    for artifact in path.parent.rglob("*"):
        if artifact.is_file():
            assert secret not in artifact.read_text()


def test_authenticated_log_preserves_non_utf8_subprocess_output_without_thread_errors(scheduled_env):
    env = scheduled_env
    secret = "review-log-secret"
    env.sctx.variables.set("TEST_BENCH_KEY", secret)
    original = env.fw.build_benchmark_command.side_effect

    def command(*args, **kwargs):
        cmd = original(*args, **kwargs)
        return [*cmd[:-1], cmd[-1] + "; import os; os.write(1, bytes([255])); print(%r)" % secret]

    env.fw.build_benchmark_command.side_effect = command
    result = benchmark(replace(env.options, api_key_env="TEST_BENCH_KEY", export_files=False), sctx=env.sctx)
    log = next(Path(result.state_dir).glob("runs/*.log")).read_text()
    assert result.success and "\ufffd" in log and "[REDACTED]" in log
    assert secret not in log


@pytest.mark.parametrize("timeout", [0, -1, True, "7"])
def test_invalid_initial_timeout_is_rejected_before_state_or_launch(scheduled_env, timeout):
    env = scheduled_env
    with pytest.raises(BenchmarkFailed, match="positive integer timeout"):
        benchmark(replace(env.options, timeout=timeout), sctx=env.sctx)
    env.run.assert_not_called()
    assert not list(env.sctx.config.cache_dir.glob("benchmarks/bench_*/state.yaml"))


@pytest.mark.parametrize("snapshot", [False, True])
def test_environment_template_credentials_do_not_change_benchmark_fingerprint(snapshot):
    from copy import deepcopy

    from sparkrun.benchmarking.metadata import benchmark_recipe_fingerprint
    from sparkrun.core.recipe import Recipe
    from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint

    def recipe(secret, cache="engram"):
        result = Recipe(
            {
                "model": "org/model",
                "runtime": "sglang",
                "container": "image:test",
                "env": {
                    "SERVICE_API_KEY": secret,
                    "CACHE_PATH": "{launch.runtime_cache_dir}/" + cache,
                },
            }
        )
        if snapshot:
            result.snapshot_declared_values()
        return result

    first, second = recipe("secret-one"), recipe("secret-two")
    original = deepcopy(first.__getstate__())
    assert derive_recipe_fingerprint(first) != derive_recipe_fingerprint(second)
    assert benchmark_recipe_fingerprint(first) == benchmark_recipe_fingerprint(second)
    assert benchmark_recipe_fingerprint(first) != benchmark_recipe_fingerprint(recipe("secret-one", "other"))
    assert first.__getstate__() == original


def test_published_recipe_redacts_template_credentials_from_state_and_snapshot():
    from sparkrun.benchmarking.metadata import public_recipe_text
    from sparkrun.core.recipe import Recipe

    recipe = Recipe(
        {
            "model": "org/model",
            "runtime": "sglang",
            "env": {
                "SERVICE_AUTH_TOKEN": "private-auth-value",
                "MODEL_PATH": "{launch.model_path}",
            },
        }
    )
    recipe.snapshot_declared_values()
    published = public_recipe_text(recipe._serialize_yaml())
    assert "private-auth-value" not in published
    data = yaml.safe_load(published)
    assert data["env"] == {"MODEL_PATH": "{launch.model_path}"}
    assert data["_declared"]["env"] == {"MODEL_PATH": "{launch.model_path}"}
    assert recipe.env["SERVICE_AUTH_TOKEN"] == "private-auth-value"
