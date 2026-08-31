"""Tests for :mod:`sparkrun.core.validation`.

The shapes exercised here are the ones a real ``@spark-arena`` recipe carried
past ``sparkrun recipe validate`` without a word: an unresolvable ``builder:``,
NCCL device names hardcoded over sparkrun's own detection, a bind mount only
its author had, and a serve flag pinned in ``command:`` where the config chain
could not see it.
"""

from __future__ import annotations

import pytest

from sparkrun.core.bootstrap import init_sparkrun
from sparkrun.core.recipe import Recipe
from sparkrun.core.validation import (
    ERROR,
    SUGGESTION,
    WARNING,
    RecipeIssue,
    check_hardcoded_serve_flags,
    check_managed_comm_env,
    check_mount_portability,
    validate_recipe,
)


@pytest.fixture
def v():
    return init_sparkrun()


def _recipe(**overrides) -> Recipe:
    data = {
        "name": "test-recipe",
        "model": "Qwen/Qwen3-1.7B",
        "runtime": "vllm-distributed",
        "container": "vllm/vllm-openai:latest",
    }
    data.update(overrides)
    return Recipe(data)


def _codes(issues) -> set[str]:
    return {i.code for i in issues}


# --------------------------------------------------------------------------
# Builder resolution
# --------------------------------------------------------------------------


def test_unknown_builder_is_an_error(v):
    """The @spark-arena case: `builder: ursuciprian` used to validate clean."""
    issues = validate_recipe(_recipe(builder="ursuciprian"), v=v)
    assert "builder-unknown" in _codes(issues)
    assert [i for i in issues if i.code == "builder-unknown"][0].severity == ERROR


def test_known_builder_is_not_reported(v):
    assert "builder-unknown" not in _codes(validate_recipe(_recipe(builder="docker-pull"), v=v))


def test_builder_alias_resolves(v):
    """An alias is a spelling of a real builder, not an unknown one."""
    from sparkrun.core.bootstrap import get_builder

    try:
        get_builder("venv", v)
    except Exception:
        pytest.skip("uv-venv builder gated off on this channel")
    assert "builder-unknown" not in _codes(validate_recipe(_recipe(builder="venv"), v=v))


def test_no_builder_declared_is_silent(v):
    assert not {"builder-unknown", "builder-disabled"} & _codes(validate_recipe(_recipe(), v=v))


def test_unknown_builder_aborts_the_launch(v):
    """``launch_inference`` must not warn-and-skip: nothing would build."""
    from sparkrun.core import launcher
    from sparkrun.core.bootstrap import get_runtime
    from sparkrun.core.config import SparkrunConfig

    recipe = _recipe(builder="ursuciprian")
    recipe.resolve({})
    with pytest.raises(ValueError, match="Unknown builder"):
        launcher.launch_inference(
            recipe=recipe,
            runtime=get_runtime("vllm-distributed", v),
            host_list=["host1"],
            overrides={},
            config=SparkrunConfig(),
            dry_run=True,
            v=v,
        )


# --------------------------------------------------------------------------
# Executor resolution
# --------------------------------------------------------------------------


def test_unknown_executor_is_an_error(v):
    issues = validate_recipe(_recipe(executor="nonexistent-executor"), v=v)
    assert "executor-unavailable" in _codes(issues)


def test_default_executor_is_not_reported(v):
    assert "executor-unavailable" not in _codes(validate_recipe(_recipe(), v=v))


# --------------------------------------------------------------------------
# Sparkrun-managed communication env
# --------------------------------------------------------------------------


def test_managed_comm_env_is_warned():
    recipe = _recipe(
        env={
            "NCCL_IB_HCA": "rocep1s0f1,roceP2p1s0f1",
            "NCCL_IB_GID_INDEX": "3",
            "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "600",
        }
    )
    issues = check_managed_comm_env(recipe)
    assert len(issues) == 1
    assert issues[0].severity == WARNING
    assert "NCCL_IB_HCA" in issues[0].message
    assert "NCCL_IB_GID_INDEX" in issues[0].message
    # Unmanaged vars are the recipe's own business.
    assert "VLLM_HTTP_TIMEOUT_KEEP_ALIVE" not in issues[0].message


def test_unmanaged_env_is_silent():
    assert check_managed_comm_env(_recipe(env={"VLLM_PLE_CPU_OFFLOAD": "0"})) == []


def test_managed_key_set_covers_what_the_generator_emits():
    """The constant is hand-maintained; this is what keeps it honest.

    ``MANAGED_COMM_ENV_KEYS`` lives next to the generators precisely so the two
    move together — but nothing enforces that at import time, so assert it.
    """
    from sparkrun.orchestration.infiniband import (
        MANAGED_COMM_ENV_KEYS,
        generate_nccl_env,
        generate_ring_nccl_overrides,
    )

    ib_info = {
        "IB_DETECTED": "1",
        "DETECTED_HCA_LIST": "rocep1s0f0",
        "DETECTED_SOCKET_IFNAME": "enp1s0f0np0",
        "DETECTED_NET_LIST": "enp1s0f1np1",
        "DETECTED_UCX_LIST": "rocep1s0f0:1",
        "DETECTED_GID_INDEX": "3",
        "DETECTED_MGMT_IP": "10.0.0.1",
    }
    for topology in (None, "ring"):
        emitted = set(generate_nccl_env(ib_info, topology=topology))
        assert emitted <= MANAGED_COMM_ENV_KEYS, sorted(emitted - MANAGED_COMM_ENV_KEYS)
    assert set(generate_ring_nccl_overrides(ib_info)) <= MANAGED_COMM_ENV_KEYS

    # And the fallback branch (no mgmt interface, IB nets only).
    fallback = dict(ib_info)
    del fallback["DETECTED_SOCKET_IFNAME"]
    assert set(generate_nccl_env(fallback)) <= MANAGED_COMM_ENV_KEYS


# --------------------------------------------------------------------------
# Bind-mount portability
# --------------------------------------------------------------------------


def test_author_local_mount_is_warned():
    recipe = _recipe(
        executor_config={
            "volumes": ["/home/nvidia/GEN-AI/patches/ple_layer.py:/usr/local/lib/python3.12/x.py:ro"],
        }
    )
    issues = check_mount_portability(recipe)
    assert len(issues) == 1
    assert issues[0].severity == WARNING
    assert "/home/nvidia/GEN-AI/patches/ple_layer.py" in issues[0].message


@pytest.mark.parametrize(
    "source",
    ["/dev/infiniband", "/sys/class/infiniband", "/etc/localtime", "/tmp/scratch", "my-named-volume"],
)
def test_portable_mounts_are_silent(source):
    assert check_mount_portability(_recipe(executor_config={"volumes": ["%s:/x" % source]})) == []


def test_sparkrun_managed_cache_mount_is_silent():
    from sparkrun.core.config import DEFAULT_CACHE_DIR

    recipe = _recipe(executor_config={"volumes": ["%s/tuning:/cache/tuning" % DEFAULT_CACHE_DIR]})
    assert check_mount_portability(recipe) == []


def test_mount_dict_form_is_handled():
    recipe = _recipe(executor_config={"volumes": {"/opt/site-data": "/data"}})
    assert _codes(check_mount_portability(recipe)) == {"non-portable-mount"}


def test_no_volumes_is_silent():
    assert check_mount_portability(_recipe(executor_config={"cap_add": ["SYS_PTRACE"]})) == []


# --------------------------------------------------------------------------
# Hardcoded serve flags
# --------------------------------------------------------------------------


def test_hardcoded_kv_cache_dtype_is_warned(v):
    from sparkrun.core.bootstrap import get_runtime

    runtime = get_runtime("vllm-distributed", v)
    recipe = _recipe(command="vllm serve {model} --kv-cache-dtype auto --port {port}")
    issues = check_hardcoded_serve_flags(recipe, runtime)
    assert _codes(issues) == {"hardcoded-serve-flag"}
    assert "kv_cache_dtype" in issues[0].message


def test_placeholder_reference_is_not_a_finding(v):
    """The documented pattern must stay silent or the check is unusable."""
    from sparkrun.core.bootstrap import get_runtime

    runtime = get_runtime("vllm-distributed", v)
    recipe = _recipe(
        defaults={"kv_cache_dtype": "fp8", "tensor_parallel": 2},
        command="vllm serve {model} --kv-cache-dtype {kv_cache_dtype} --tensor-parallel-size {tensor_parallel}",
    )
    assert check_hardcoded_serve_flags(recipe, runtime) == []


def test_declared_default_suppresses_the_finding(v):
    """Declaring the key makes it visible to the chain even with a literal."""
    from sparkrun.core.bootstrap import get_runtime

    runtime = get_runtime("vllm-distributed", v)
    recipe = _recipe(defaults={"kv_cache_dtype": "fp8"}, command="vllm serve {model} --kv-cache-dtype fp8")
    assert check_hardcoded_serve_flags(recipe, runtime) == []


def test_taskset_dash_c_is_not_read_as_context_length(v):
    """``-c`` is llama.cpp's --ctx-size *and* taskset's cpu-list flag."""
    from sparkrun.core.bootstrap import get_runtime

    runtime = get_runtime("vllm-distributed", v)
    recipe = _recipe(command="taskset -c 5-9,15-19 vllm serve {model} --port {port}")
    assert check_hardcoded_serve_flags(recipe, runtime) == []


def test_line_continuation_is_not_read_as_a_value(v):
    from sparkrun.core.bootstrap import get_runtime

    runtime = get_runtime("vllm-distributed", v)
    recipe = _recipe(
        defaults={"port": 8000},
        command="vllm serve {model} \\\n  --port {port} \\\n  --kv-cache-dtype fp8\n",
    )
    issues = check_hardcoded_serve_flags(recipe, runtime)
    assert _codes(issues) == {"hardcoded-serve-flag"}
    assert "--kv-cache-dtype fp8" in issues[0].message


def test_equals_form_is_detected(v):
    from sparkrun.core.bootstrap import get_runtime

    runtime = get_runtime("vllm-distributed", v)
    recipe = _recipe(command="vllm serve {model} --kv-cache-dtype=fp8")
    assert _codes(check_hardcoded_serve_flags(recipe, runtime)) == {"hardcoded-serve-flag"}


def test_no_command_template_is_silent(v):
    from sparkrun.core.bootstrap import get_runtime

    assert check_hardcoded_serve_flags(_recipe(), get_runtime("vllm-distributed", v)) == []


def test_runtime_without_a_flag_map_still_uses_aliases():
    """A runtime that declares no map must not disable the check outright."""

    class _Bare:
        runtime_name = "bare"

        def serve_flag_map(self):
            return None

    recipe = _recipe(command="serve --kv-cache-dtype fp8")
    assert _codes(check_hardcoded_serve_flags(recipe, _Bare())) == {"hardcoded-serve-flag"}


# --------------------------------------------------------------------------
# Severity contract
# --------------------------------------------------------------------------


def test_metadata_problems_are_suggestions(v):
    """Real published recipes carry prose here; aborting on it strands them.

    A suggestion rather than a warning because the cost — a dropped VRAM
    estimate — is identical on every cluster, so it fails the "does this
    behave differently elsewhere?" test the warning tier is for.
    """
    recipe = _recipe(metadata={"quantization": "NVFP4 (compressed-tensors, mixed precision)"})
    issues = validate_recipe(recipe, v=v)
    meta = [i for i in issues if i.code == "recipe-metadata"]
    assert meta and all(i.severity == SUGGESTION for i in meta)


def test_missing_model_is_an_error(v):
    issues = validate_recipe(_recipe(model=""), v=v)
    assert any(i.code == "recipe-field" and i.severity == ERROR for i in issues)


def test_runtime_hook_can_declare_an_error(v):
    """A runtime that says "I cannot serve this" must block the launch."""
    from sparkrun.core.bootstrap import get_runtime

    recipe = _recipe(runtime="vllm-ray", defaults={"data_parallel": 2})
    runtime = get_runtime("vllm-ray", v)
    hook = [i for i in validate_recipe(recipe, runtime=runtime, v=v) if i.code == "runtime-field"]
    assert hook and any(i.severity == ERROR and "data_parallel" in i.message for i in hook)


def test_undeclared_string_findings_are_suggestions(v):
    """The legacy return form lands at the *bottom* of the ladder.

    Severity can't be read out of prose, and guessing up is the direction that
    breaks working launches — so the unknown is never fatal by default.
    """

    class _Legacy:
        runtime_name = "legacy"

        def validate_recipe(self, recipe):
            return ["[legacy] a plain string finding"]

        def serve_flag_map(self):
            return None

        def known_config_keys(self):
            return None

    issues = validate_recipe(_recipe(), runtime=_Legacy(), v=v)
    hook = [i for i in issues if i.code == "runtime-field"]
    assert hook and all(i.severity == SUGGESTION for i in hook)


def test_coerce_issues_preserves_a_declared_code():
    """A plugin naming its own check keeps that name in the report."""
    from sparkrun.core.validation import coerce_issues

    mixed = ["bare string", RecipeIssue(ERROR, "my-own-check", "declared"), RecipeIssue(ERROR, "", "no code")]
    out = coerce_issues(mixed, "runtime-field")
    assert [(i.severity, i.code) for i in out] == [
        (SUGGESTION, "runtime-field"),
        (ERROR, "my-own-check"),
        (ERROR, "runtime-field"),
    ]


def test_builder_hook_findings_are_coerced(v):
    """The eugr advice is a bare string, so it stays advisory."""
    recipe = _recipe(builder="eugr", command="")
    issues = validate_recipe(recipe, v=v)
    hook = [i for i in issues if i.code == "builder-field"]
    assert hook and all(i.severity == SUGGESTION for i in hook)


def test_validate_splits_into_structure_and_metadata():
    """``Recipe.validate()`` stays the concatenation of its two halves."""
    recipe = _recipe(model="", metadata={"quantization": "not-a-method"})
    assert recipe.validate() == recipe.validate_structure() + recipe.validate_metadata()
    assert any("model" in m for m in recipe.validate_structure())
    assert any("quantization" in m for m in recipe.validate_metadata())


def test_errors_sort_before_warnings(v):
    issues = validate_recipe(
        _recipe(
            builder="ursuciprian",
            env={"NCCL_IB_HCA": "rocep1s0f1"},
        ),
        v=v,
    )
    severities = [i.severity for i in issues]
    assert severities == sorted(severities, key=lambda s: 0 if s == ERROR else 1)
    assert ERROR in severities and WARNING in severities


def test_issue_to_dict_shape():
    issue = RecipeIssue(WARNING, "some-code", "some message")
    assert issue.to_dict() == {
        "severity": "warning",
        "code": "some-code",
        "message": "some message",
        "summary": "some message",
        "fix": "",
    }
    assert not issue.is_error
    assert str(issue) == "some message"


# --------------------------------------------------------------------------
# mounts.missing_source policy
# --------------------------------------------------------------------------


def _config_with(data):
    from sparkrun.core.config import SparkrunConfig

    cfg = SparkrunConfig.__new__(SparkrunConfig)
    cfg._data = data
    return cfg


@pytest.mark.parametrize("raw,expected", [("fail", "fail"), ("warn", "warn"), ("ignore", "ignore"), ("WARN", "warn"), (" warn ", "warn")])
def test_missing_mount_source_policy_parses(raw, expected):
    assert _config_with({"mounts": {"missing_source": raw}}).missing_mount_source_policy == expected


def test_missing_mount_source_policy_defaults_to_fail():
    assert _config_with({}).missing_mount_source_policy == "fail"
    assert _config_with({"mounts": {}}).missing_mount_source_policy == "fail"


def test_unrecognized_policy_falls_back_to_fail(caplog):
    """A typo must not quietly disable a safety check."""
    import logging

    with caplog.at_level(logging.WARNING):
        assert _config_with({"mounts": {"missing_source": "yolo"}}).missing_mount_source_policy == "fail"
    assert "missing_source" in caplog.text


# --------------------------------------------------------------------------
# Executor bind_mount_sources seam
# --------------------------------------------------------------------------


def test_docker_reports_its_bind_sources():
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.docker import DockerExecutor

    ex = DockerExecutor(ExecutorConfig(volumes=["/opt/a", "/opt/b:/x", "/opt/c:/y:ro"]))
    assert ex.bind_mount_sources() == ["/opt/a", "/opt/b", "/opt/c"]


def test_docker_excludes_named_volumes_and_relative_paths():
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.docker import DockerExecutor

    ex = DockerExecutor(ExecutorConfig(volumes=["my-named-vol:/x", "./rel:/y", "/opt/real:/z"]))
    assert ex.bind_mount_sources() == ["/opt/real"]


def test_local_executor_binds_nothing():
    """``volumes:`` is inert for a native run — verifying it would block a launch."""
    from sparkrun.orchestration.executors._base import ExecutorConfig
    from sparkrun.orchestration.executors.local import LocalExecutor

    ex = LocalExecutor(ExecutorConfig(volumes=["/opt/a:/x"]))
    assert ex.bind_mount_sources() == []


def test_base_executor_binds_nothing():
    from sparkrun.orchestration.executors._base import Executor

    assert Executor.bind_mount_sources(object()) == []


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------


def test_issue_message_joins_summary_and_fix():
    issue = RecipeIssue(WARNING, "c", "Something is wrong.", "Do this instead.")
    assert issue.message == "Something is wrong. Do this instead."
    assert issue.to_dict() == {
        "severity": "warning",
        "code": "c",
        "message": "Something is wrong. Do this instead.",
        "summary": "Something is wrong.",
        "fix": "Do this instead.",
    }


def test_issue_without_a_fix_is_just_the_summary():
    issue = RecipeIssue(ERROR, "c", "model is required")
    assert issue.message == "model is required"
    assert issue.fix == ""


def test_coerce_preserves_the_fix_when_recoding():
    """Re-coding an issue must not collapse its two halves into one."""
    from sparkrun.core.validation import coerce_issues

    (out,) = coerce_issues([RecipeIssue(ERROR, "", "diagnosis", "remedy")], "runtime-field")
    assert (out.code, out.summary, out.fix) == ("runtime-field", "diagnosis", "remedy")


def test_report_separates_findings_and_fixes():
    from sparkrun.utils.cli_formatters import format_validation_report

    issues = [
        RecipeIssue(ERROR, "builder-unknown", "Unknown builder: 'x'."),
        RecipeIssue(WARNING, "non-portable-mount", "Binds a host path.", "Package it as a mod."),
    ]
    import click

    # click.style always emits colour; click.echo is what strips it for a
    # non-TTY, so assert against the unstyled form.
    text = click.unstyle(format_validation_report("r", issues))
    lines = text.splitlines()

    assert lines[0] == "Recipe 'r': 1 error, 1 warning"
    # A blank line before each finding keeps the count scannable.
    assert text.count("\n\n") == 3  # two findings + one fix block
    assert "ERROR  builder-unknown" in text
    assert "warning  non-portable-mount" in text
    # The fix is indented deeper than the summary it belongs to.
    summary_line = next(ln for ln in lines if "Binds a host path" in ln)
    fix_line = next(ln for ln in lines if "Package it as a mod" in ln)
    assert len(fix_line) - len(fix_line.lstrip()) > len(summary_line) - len(summary_line.lstrip())


def test_report_pluralizes_counts():
    from sparkrun.utils.cli_formatters import format_validation_report

    import click

    one = click.unstyle(format_validation_report("r", [RecipeIssue(ERROR, "c", "m")]))
    assert one.splitlines()[0] == "Recipe 'r': 1 error"
    two = click.unstyle(format_validation_report("r", [RecipeIssue(WARNING, "c", "m"), RecipeIssue(WARNING, "c", "m")]))
    assert two.splitlines()[0] == "Recipe 'r': 2 warnings"


def test_report_wraps_long_messages(monkeypatch):
    import shutil

    from sparkrun.utils.cli_formatters import format_validation_report

    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=(80, 24): __import__("os").terminal_size((80, 24)))
    import click

    text = click.unstyle(format_validation_report("r", [RecipeIssue(WARNING, "c", "word " * 80)]))
    assert all(len(line) <= 80 for line in text.splitlines())
    assert len(text.splitlines()) > 3


# --------------------------------------------------------------------------
# Severity ladder / --fail-on threshold
# --------------------------------------------------------------------------


def _ladder():
    return [
        RecipeIssue(ERROR, "e", "err"),
        RecipeIssue(WARNING, "w", "warn"),
        RecipeIssue(SUGGESTION, "s", "sugg"),
    ]


@pytest.mark.parametrize(
    "fail_on,expected",
    [("error", True), ("warning", True), ("suggestion", True), ("none", False)],
)
def test_should_fail_with_an_error_present(fail_on, expected):
    from sparkrun.core.validation import should_fail

    assert should_fail(_ladder(), fail_on) is expected


@pytest.mark.parametrize(
    "fail_on,expected",
    [("error", False), ("warning", True), ("suggestion", True), ("none", False)],
)
def test_should_fail_on_a_warning_only(fail_on, expected):
    from sparkrun.core.validation import should_fail

    assert should_fail([RecipeIssue(WARNING, "w", "warn")], fail_on) is expected


@pytest.mark.parametrize(
    "fail_on,expected",
    [("error", False), ("warning", False), ("suggestion", True), ("none", False)],
)
def test_should_fail_on_a_suggestion_only(fail_on, expected):
    """The whole point of the third tier: --strict must not fail on advice."""
    from sparkrun.core.validation import should_fail

    assert should_fail([RecipeIssue(SUGGESTION, "s", "sugg")], fail_on) is expected


def test_default_threshold_is_error_only():
    from sparkrun.core.validation import DEFAULT_FAIL_ON, should_fail

    assert DEFAULT_FAIL_ON == ERROR
    assert should_fail([RecipeIssue(WARNING, "w", "w")]) is False
    assert should_fail([RecipeIssue(ERROR, "e", "e")]) is True


def test_issues_sort_most_severe_first(v):
    issues = validate_recipe(
        _recipe(
            builder="ursuciprian",
            env={"NCCL_IB_HCA": "rocep1s0f1"},
            command="vllm serve {model} --kv-cache-dtype auto",
        ),
        v=v,
    )
    from sparkrun.core.validation import rank

    assert [i.severity for i in issues] == sorted((i.severity for i in issues), key=rank)
    assert {ERROR, WARNING, SUGGESTION} <= {i.severity for i in issues}


# --------------------------------------------------------------------------
# Launch-path display: suggestions withheld, but never failed-on silently
# --------------------------------------------------------------------------


def test_launch_withholds_suggestions(v):
    """A hardcoded serve flag is advice for the author, not the launcher."""
    from sparkrun.core.validation import validate_for_launch

    recipe = _recipe(command="vllm serve {model} --kv-cache-dtype auto")
    shown, failed = validate_for_launch(recipe, v=v)
    assert "hardcoded-serve-flag" not in _codes(shown)
    assert failed is False


def test_launch_shows_warnings(v):
    from sparkrun.core.validation import validate_for_launch

    shown, failed = validate_for_launch(_recipe(env={"NCCL_IB_HCA": "x"}), v=v)
    assert "managed-comm-env" in _codes(shown)
    assert failed is False  # default threshold is errors only


def test_launch_aborts_on_an_error(v):
    from sparkrun.core.validation import validate_for_launch

    shown, failed = validate_for_launch(_recipe(builder="ursuciprian"), v=v)
    assert "builder-unknown" in _codes(shown)
    assert failed is True


def test_launch_never_fails_on_something_it_did_not_show(v):
    """A threshold strict enough to fail on suggestions must also print them."""
    from sparkrun.core.validation import validate_for_launch

    recipe = _recipe(command="vllm serve {model} --kv-cache-dtype auto")
    shown, failed = validate_for_launch(recipe, fail_on="suggestion", v=v)
    assert failed is True
    assert "hardcoded-serve-flag" in _codes(shown)


def test_display_threshold_only_widens_for_suggestion_level():
    from sparkrun.core.validation import display_threshold

    assert display_threshold("error") == WARNING
    assert display_threshold("warning") == WARNING
    assert display_threshold("none") == WARNING
    assert display_threshold("suggestion") == SUGGESTION


def test_launch_threshold_comes_from_config(v):
    """`validation.fail_on` tightens every launch path at once."""
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.validation import validate_for_launch

    cfg = SparkrunConfig.__new__(SparkrunConfig)
    cfg._data = {"validation": {"fail_on": "warning"}}
    _shown, failed = validate_for_launch(_recipe(env={"NCCL_IB_HCA": "x"}), config=cfg, v=v)
    assert failed is True


# --------------------------------------------------------------------------
# validation.fail_on config parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["error", "warning", "suggestion", "none", "WARNING", " none "])
def test_validation_fail_on_parses(raw):
    assert _config_with({"validation": {"fail_on": raw}}).validation_fail_on == raw.strip().lower()


def test_validation_fail_on_defaults_to_error():
    assert _config_with({}).validation_fail_on == ERROR
    assert _config_with({"validation": {}}).validation_fail_on == ERROR


def test_unrecognized_fail_on_falls_back_to_default(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        assert _config_with({"validation": {"fail_on": "yolo"}}).validation_fail_on == ERROR
    assert "fail_on" in caplog.text


# --------------------------------------------------------------------------
# Plugin severity sugar
# --------------------------------------------------------------------------


def test_runtime_severity_sugar(v):
    from sparkrun.core.bootstrap import get_runtime

    rt = get_runtime("vllm-distributed", v)
    assert rt.recipe_error("boom").severity == ERROR
    assert rt.recipe_warning("hmm").severity == WARNING
    # Both tag the message with the runtime so a report names its source.
    assert rt.recipe_warning("hmm").message.startswith("[vllm-distributed]")


def test_builder_severity_sugar(v):
    from sparkrun.core.bootstrap import get_builder

    b = get_builder("docker-pull", v)
    assert b.recipe_error("boom").severity == ERROR
    assert b.recipe_warning("hmm").severity == WARNING
    assert b.recipe_warning("hmm").message.startswith("[docker-pull]")


# --------------------------------------------------------------------------
# Launch-path presentation
# --------------------------------------------------------------------------


def _launch_report(issues, failed, ref="my-recipe"):
    """Capture what a launch prints. Findings go to stderr (stdout stays free
    for the launch's own output), so read that stream specifically."""
    import click
    from click.testing import CliRunner

    from sparkrun.cli._common import report_launch_validation

    @click.command()
    def _cmd():
        report_launch_validation(ref, issues, failed)

    return click.unstyle(CliRunner().invoke(_cmd).stderr)


def test_launch_report_names_validation_as_the_source():
    """Amid a launch, unlabelled findings read as failures of whatever ran last."""
    text = _launch_report([RecipeIssue(WARNING, "managed-comm-env", "pinned devices")], False)
    assert text.startswith("Recipe validation for 'my-recipe': 1 warning")


def test_launch_report_states_the_verdict_when_fatal():
    text = _launch_report([RecipeIssue(ERROR, "builder-unknown", "no such builder")], True)
    assert "Cannot launch" in text
    # Points at the one place the withheld suggestions can be seen.
    assert "sparkrun recipe validate my-recipe" in text


def test_launch_report_states_the_verdict_when_advisory():
    """ "Did this stop my launch?" is the reader's only real question."""
    text = _launch_report([RecipeIssue(WARNING, "managed-comm-env", "pinned devices")], False)
    assert "Nothing above blocks the launch" in text
    assert "Cannot launch" not in text


def test_launch_report_is_silent_with_no_findings():
    assert _launch_report([], False) == ""


def test_launch_report_uses_the_block_layout():
    """Same renderer as `recipe validate` — the two must not drift."""
    text = _launch_report([RecipeIssue(WARNING, "non-portable-mount", "binds a path", "Use a mod.")], False)
    assert "warning  non-portable-mount" in text
    lines = text.splitlines()
    summary = next(ln for ln in lines if "binds a path" in ln)
    fix = next(ln for ln in lines if "Use a mod." in ln)
    assert len(fix) - len(fix.lstrip()) > len(summary) - len(summary.lstrip())


def test_report_title_override():
    from sparkrun.utils.cli_formatters import format_validation_report

    import click

    text = click.unstyle(format_validation_report("r", [RecipeIssue(ERROR, "c", "m")], title="Custom heading"))
    assert text.splitlines()[0] == "Custom heading: 1 error"
