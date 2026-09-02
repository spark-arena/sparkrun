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
        "deprecation": False,
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
        "deprecation": False,
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


# --------------------------------------------------------------------------
# Deprecated recipe features
#
# The gap these close: two of these notices existed only as a ``logger.warning``
# on a path ``recipe validate`` never takes — one inside ``render_command``
# (reached only by an actual launch) and one inside ``EugrVllmRayRuntime.prepare``
# (reached only *after* image distribution).  So the command whose job is to
# report what is wrong with a recipe said nothing, while ``sparkrun run`` did.
# --------------------------------------------------------------------------


def test_v2_command_brace_escape_is_reported(v):
    """The reported case: `{{` in `command:` validated clean but warned at launch."""
    issues = validate_recipe(_recipe(command="vllm serve {model} --x '{{\"a\":1}}'"), v=v)
    found = [i for i in issues if i.code == "deprecated-brace-escape"]
    assert len(found) == 1
    assert found[0].severity == WARNING
    assert found[0].deprecation is True
    assert "command:" in found[0].summary


def test_v2_defaults_brace_escape_is_reported(v):
    """Read off ``_raw``: the resolver collapses the escape in ``defaults`` at load."""
    recipe = _recipe(defaults={"spec": '{{"method":"mtp"}}'}).resolve()
    # Precondition — the parsed copy no longer carries the evidence, which is
    # exactly why the check cannot use it.
    assert recipe.defaults["spec"] == '{"method":"mtp"}'

    found = [i for i in validate_recipe(recipe, v=v) if i.code == "deprecated-brace-escape"]
    assert len(found) == 1
    assert "defaults.spec" in found[0].summary


def test_v1_brace_escapes_are_not_deprecated(v):
    """v1 is the convention's home; advising against it there is advice to write
    something the format does not support."""
    recipe = Recipe(
        {
            "name": "v1",
            "recipe_version": "1",
            "model": "m",
            "runtime": "vllm",
            "container": "c",
            "command": "vllm serve m --x '{{\"a\":1}}'",
            "defaults": {"spec": '{{"a":1}}'},
        }
    ).resolve()
    assert "deprecated-brace-escape" not in _codes(validate_recipe(recipe, v=v))


def test_plain_json_is_not_reported_as_an_escape(v):
    """`}}` closes nested plain JSON as often as it escapes a brace — `{{` is the
    only reliable marker, and the idiomatic v2 spelling must stay silent."""
    issues = validate_recipe(_recipe(command='vllm serve {model} --x \'{"a":{"b":1}}\''), v=v)
    assert "deprecated-brace-escape" not in _codes(issues)


def test_v1_recipe_format_is_reported(v):
    recipe = Recipe({"name": "v1", "recipe_version": "1", "model": "m", "runtime": "vllm", "container": "c"}).resolve()
    found = [i for i in validate_recipe(recipe, v=v) if i.code == "deprecated-recipe-format"]
    assert len(found) == 1
    assert found[0].deprecation is True


def test_deprecated_runtime_is_reported(v):
    """`eugr-vllm` announced itself only from prepare(), i.e. mid-launch."""
    found = [i for i in validate_recipe(_recipe(runtime="eugr-vllm"), v=v) if i.code == "deprecated-runtime"]
    assert len(found) == 1
    assert "vllm-ray" in found[0].fix


def test_deprecated_build_arg_is_reported(v):
    recipe = _recipe(builder="eugr", runtime_config={"build_args": ["--cleanup", "--tf5"]})
    found = [i for i in validate_recipe(recipe, v=v) if i.code == "deprecated-build-arg"]
    assert len(found) == 1
    assert "--tf5" in found[0].summary


def test_live_build_args_are_not_reported(v):
    recipe = _recipe(builder="eugr", runtime_config={"build_args": ["--cleanup"]})
    assert "deprecated-build-arg" not in _codes(validate_recipe(recipe, v=v))


def test_a_clean_v2_recipe_reports_no_deprecations(v):
    assert not [i for i in validate_recipe(_recipe(), v=v) if i.deprecation]


# --------------------------------------------------------------------------
# Launch-path deprecation summary
# --------------------------------------------------------------------------


def test_launch_collapses_deprecations_to_one_line(v):
    """The migration is the author's work; at launch you want one line and a pointer."""
    from sparkrun.core.validation import validate_for_launch

    recipe = _recipe(command="vllm serve {model} --x '{{\"a\":1}}'", defaults={"spec": '{{"a":1}}'})
    assert len([i for i in validate_recipe(recipe, v=v) if i.deprecation]) == 2

    shown, _failed = validate_for_launch(recipe, v=v, recipe_ref="my-recipe")
    summaries = [i for i in shown if i.deprecation]
    assert len(summaries) == 1
    assert summaries[0].code == "deprecated-feature"
    assert "2 deprecated recipe features" in summaries[0].summary
    assert "sparkrun recipe validate my-recipe" in summaries[0].fix


def test_launch_summary_does_not_soften_the_verdict(v):
    """Collapsing is display-only: --strict still fails on a deprecation it
    described in one line rather than five."""
    from sparkrun.core.validation import validate_for_launch

    recipe = _recipe(command="vllm serve {model} --x '{{\"a\":1}}'")
    shown, failed = validate_for_launch(recipe, fail_on=WARNING, v=v)
    assert failed is True
    assert [i.code for i in shown if i.deprecation] == ["deprecated-feature"]


def test_launch_summary_is_absent_without_deprecations(v):
    from sparkrun.core.validation import validate_for_launch

    shown, _ = validate_for_launch(_recipe(env={"NCCL_IB_HCA": "x"}), v=v)
    assert "deprecated-feature" not in _codes(shown)


def test_recipe_validate_keeps_the_full_deprecation_detail(v):
    """The collapse belongs to the launch path only — `recipe validate` is where
    the migration is meant to be read."""
    recipe = _recipe(command="vllm serve {model} --x '{{\"a\":1}}'", defaults={"spec": '{{"a":1}}'})
    codes = _codes(validate_recipe(recipe, v=v))
    assert "deprecated-brace-escape" in codes
    assert "deprecated-feature" not in codes


# --------------------------------------------------------------------------
# Inferred builder
# --------------------------------------------------------------------------


def test_inferred_builder_is_reported(v):
    """`build_args` alone sets `builder: eugr` — a rule that versions with
    sparkrun rather than with the recipe."""
    recipe = _recipe(runtime="vllm", runtime_config={"build_args": ["--cleanup"]}).resolve()
    assert recipe.builder == "eugr"
    found = [i for i in validate_recipe(recipe, v=v) if i.code == "implicit-builder"]
    assert len(found) == 1
    assert found[0].severity == SUGGESTION
    assert "builder: eugr" in found[0].fix


def test_declared_builder_is_not_reported(v):
    recipe = _recipe(runtime="vllm", builder="eugr", runtime_config={"build_args": ["--cleanup"]}).resolve()
    assert "implicit-builder" not in _codes(validate_recipe(recipe, v=v))


def test_v1_does_not_double_report_the_inferred_builder(v):
    """`deprecated-recipe-format` already names it; saying it twice teaches skimming."""
    recipe = Recipe(
        {"name": "v1", "recipe_version": "1", "model": "m", "runtime": "vllm", "container": "c", "build_args": ["--cleanup"]}
    ).resolve()
    assert recipe.builder == "eugr"
    assert "implicit-builder" not in _codes(validate_recipe(recipe, v=v))


# --------------------------------------------------------------------------
# Sparkrun-managed cache env
#
# The two tiers sit on opposite sides of ``recipe.env`` in ``merge_env``, so the
# same-looking mistake has opposite outcomes and needs opposite advice.
# --------------------------------------------------------------------------


def test_recipe_hf_cache_env_is_reported_as_overridden(v):
    """`get_extra_env` is merged last, so the recipe's value is discarded."""
    found = [i for i in validate_recipe(_recipe(env={"HF_HOME": "/elsewhere"}), v=v) if i.code == "overridden-cache-env"]
    assert len(found) == 1
    assert found[0].severity == WARNING
    assert "discarded" in found[0].summary


def test_recipe_xdg_cache_env_is_reported_as_winning(v):
    """The runtime-cache tier is merged first, so the recipe's value wins and the
    compile caches land off the mount sparkrun persists."""
    found = [i for i in validate_recipe(_recipe(env={"XDG_CACHE_HOME": "/elsewhere"}), v=v) if i.code == "managed-cache-env"]
    assert len(found) == 1
    assert "XDG_CACHE_HOME" in found[0].summary


def test_the_two_cache_env_findings_do_not_overlap(v):
    """A key cannot be both overridden and winning."""
    issues = validate_recipe(_recipe(env={"HF_HOME": "/a", "XDG_CACHE_HOME": "/b"}), v=v)
    overridden = next(i for i in issues if i.code == "overridden-cache-env")
    wins = next(i for i in issues if i.code == "managed-cache-env")
    assert overridden.summary.startswith("env: sets HF_HOME,")
    assert wins.summary.startswith("env: sets XDG_CACHE_HOME,")


def test_unrelated_env_is_not_reported(v):
    assert not [i for i in validate_recipe(_recipe(env={"VLLM_USE_V1": "1"}), v=v) if "cache-env" in i.code]


# --------------------------------------------------------------------------
# Unknown top-level keys
# --------------------------------------------------------------------------


def test_misplaced_serve_key_is_reported(v):
    """Only `defaults:` feeds the config chain. At the top level the key is
    absorbed into runtime_config, reaches nothing, and the rendered command
    shows nothing missing."""
    found = [i for i in validate_recipe(_recipe(max_model_len=8192), v=v) if i.code == "misplaced-config-key"]
    assert len(found) == 1
    assert found[0].severity == WARNING
    assert "max_model_len" in found[0].summary


def test_unknown_top_level_key_is_a_suggestion(v):
    """A key this build does not know is routinely a *newer* recipe."""
    found = [i for i in validate_recipe(_recipe(totally_bogus=1), v=v) if i.code == "unknown-top-level-key"]
    assert len(found) == 1
    assert found[0].severity == SUGGESTION


def test_v1_build_args_at_top_level_are_not_unknown(v):
    """`build_args` at the top level is the v1 spelling and is read by name —
    flagging it would fire on every v1 recipe ever published."""
    recipe = Recipe(
        {"name": "v1", "recipe_version": "1", "model": "m", "runtime": "vllm", "container": "c", "build_args": ["--cleanup"]}
    ).resolve()
    assert "unknown-top-level-key" not in _codes(validate_recipe(recipe, v=v))


def test_explicit_runtime_config_is_not_reported(v):
    """An explicit `runtime_config:` mapping is a deliberate statement about a
    runtime that reads it by name, not an accident of the sweep."""
    recipe = _recipe(runtime_config={"some_engine_knob": 1})
    assert "unknown-top-level-key" not in _codes(validate_recipe(recipe, v=v))


def test_known_top_level_keys_are_not_reported(v):
    issues = validate_recipe(_recipe(description="d", min_nodes=1, capabilities=["tools"]), v=v)
    assert "unknown-top-level-key" not in _codes(issues)
    assert "misplaced-config-key" not in _codes(issues)


# --------------------------------------------------------------------------
# Deprecated topology surface (mode / solo_only / cluster_only)
# --------------------------------------------------------------------------


def _topo(**overrides) -> Recipe:
    data = {"name": "t", "model": "m", "runtime": "vllm-distributed", "container": "c"}
    data.update(overrides)
    return Recipe(data).resolve()


@pytest.mark.parametrize(
    "declared,expected_migration",
    [
        ({"mode": "solo"}, "`max_nodes: 1`"),
        ({"mode": "cluster"}, "`min_nodes: 2`"),
        ({"solo_only": True}, "`max_nodes: 1`"),
        ({"cluster_only": True}, "`min_nodes: 2`"),
    ],
)
def test_declared_topology_key_is_reported(v, declared, expected_migration):
    found = [i for i in validate_recipe(_topo(**declared), v=v) if i.code == "deprecated-topology"]
    assert len(found) == 1
    assert found[0].severity == WARNING
    assert found[0].deprecation is True
    assert expected_migration in found[0].fix


def test_node_range_recipes_are_not_reported(v):
    """The load-bearing one: ``Recipe.__init__`` *derives* ``mode`` from the node
    range, so reporting off the parsed value would fire on exactly the recipes
    this advises people to write."""
    recipe = _topo(min_nodes=2)
    assert recipe.mode == "cluster"  # derived, never declared
    assert "deprecated-topology" not in _codes(validate_recipe(recipe, v=v))

    solo = _topo(max_nodes=1)
    assert solo.mode == "solo"
    assert "deprecated-topology" not in _codes(validate_recipe(solo, v=v))


def test_a_recipe_declaring_no_topology_is_not_reported(v):
    assert "deprecated-topology" not in _codes(validate_recipe(_topo(), v=v))


def test_falsy_topology_key_is_not_told_to_pin_a_range(v):
    """`solo_only: false` constrains nothing; advising `max_nodes: 1` would be
    the opposite instruction."""
    found = next(i for i in validate_recipe(_topo(solo_only=False), v=v) if i.code == "deprecated-topology")
    # The general advice still names the replacement fields; what must not
    # appear is a concrete range to pin.
    assert "`max_nodes: 1`" not in found.fix
    assert "`solo_only` → nothing" in found.fix
    assert "delete the key" in found.fix


def test_multiple_topology_keys_are_one_finding(v):
    """They are one concept and their migrations interact."""
    found = [i for i in validate_recipe(_topo(mode="solo", solo_only=True), v=v) if i.code == "deprecated-topology"]
    assert len(found) == 1
    assert "`mode: solo`" in found[0].summary
    assert "`solo_only`" in found[0].summary


def test_topology_is_reported_for_v1_too(v):
    """Unlike the brace escape, this is not a spelling v1 requires — the
    replacement has always worked in both."""
    recipe = Recipe(
        {"name": "v1", "recipe_version": "1", "model": "m", "runtime": "vllm", "container": "c", "cluster_only": True}
    ).resolve()
    assert "deprecated-topology" in _codes(validate_recipe(recipe, v=v))


def test_first_party_container_inference_is_not_reported(v):
    """sparkrun owns the image *and* the rule, so keeping them in step is its
    job. This is 40 of the 47 registry recipes with an inferred builder —
    reporting them would make the finding mostly noise."""
    from sparkrun.core.recipe import EUGR_CONTAINER_PREFIX

    recipe = _recipe(runtime="vllm", container=EUGR_CONTAINER_PREFIX + "-b12x:latest").resolve()
    assert recipe.builder == "eugr"  # still inferred — only the *advice* is withheld
    assert "implicit-builder" not in _codes(validate_recipe(recipe, v=v))


def test_third_party_container_with_build_args_is_reported(v):
    """The 7 that remain: an inference over an artifact sparkrun does not
    publish, which is the case the finding is actually about."""
    recipe = _recipe(runtime="vllm", container="myorg/vllm:latest", runtime_config={"build_args": ["--cleanup"]}).resolve()
    assert recipe.builder == "eugr"
    found = next(i for i in validate_recipe(recipe, v=v) if i.code == "implicit-builder")
    assert found.severity == SUGGESTION
    assert "`build_args`" in found.summary


def test_mods_alone_does_not_infer_a_builder(v):
    """`mods` is part of the v2 spec and works with any builder, so it is not a
    signal — and must not produce an `implicit-builder` finding."""
    recipe = _recipe(runtime="vllm", mods=["mods/some-patch"]).resolve()
    assert recipe.builder == ""
    assert "implicit-builder" not in _codes(validate_recipe(recipe, v=v))


# --------------------------------------------------------------------------
# Restated substitutions — values sparkrun would have supplied anyway
# --------------------------------------------------------------------------


def test_literal_model_id_as_positional_serve_arg_is_reported(v):
    """The dominant shape: 39 of the 118 cached registry recipes with a
    `command:` write `vllm serve <literal id>` where `{model}` belongs."""
    recipe = _recipe(command="vllm serve Qwen/Qwen3-1.7B --port 8000").resolve()
    found = next(i for i in validate_recipe(recipe, v=v) if i.code == "restated-model-arg")
    assert found.severity == SUGGESTION
    assert "Qwen/Qwen3-1.7B" in found.summary
    assert "{model}" in found.fix


def test_literal_model_id_as_flag_value_is_reported(v):
    recipe = _recipe(runtime="sglang", command="sglang serve --model-path Qwen/Qwen3-1.7B").resolve()
    assert "restated-model-arg" in _codes(validate_recipe(recipe, v=v))


def test_model_placeholder_is_not_reported(v):
    """The working spelling. A finding that fires here would fire on the
    two-thirds of the corpus that is doing the right thing."""
    recipe = _recipe(command="vllm serve {model} --port {port}", defaults={"port": 8000}).resolve()
    assert "restated-model-arg" not in _codes(validate_recipe(recipe, v=v))


def test_model_id_embedded_in_another_argument_is_not_reported(v):
    """Whole-token equality, never substring: measured across the cached
    registries the id appears *only* as the model argument, so substring
    matching would buy nothing and claim this."""
    recipe = _recipe(command="vllm serve {model} --served-model-name Qwen/Qwen3-1.7B-chat").resolve()
    assert "restated-model-arg" not in _codes(validate_recipe(recipe, v=v))


def test_absolute_model_path_is_not_reported(v):
    """`model:` is already the on-disk path, so `{model}` renders the same
    string and none of the three rewrite mechanisms applies."""
    recipe = _recipe(model="/mnt/weights/qwen3", command="vllm serve /mnt/weights/qwen3").resolve()
    assert "restated-model-arg" not in _codes(validate_recipe(recipe, v=v))


def test_managed_container_path_in_command_is_reported(v):
    recipe = _recipe(command="vllm serve /cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/abc").resolve()
    found = next(i for i in validate_recipe(recipe, v=v) if i.code == "restated-managed-path")
    assert found.severity == SUGGESTION
    assert "/cache/huggingface" in found.summary
    assert "models--Qwen--Qwen3-1.7B" in found.summary


def test_managed_container_path_in_hook_is_reported(v):
    """The one instance in the cached corpus is a `pre_exec`, not a `command:`
    — and it searches a cache root sparkrun never populates."""
    recipe = _recipe(
        command="vllm serve {model}",
        pre_exec=['cp "$(find /root/.cache/huggingface/ -name parser.py | head -1)" /tmp/p.py'],
    ).resolve()
    found = next(i for i in validate_recipe(recipe, v=v) if i.code == "restated-managed-path")
    assert found.summary.startswith("pre_exec:")


def test_ordinary_paths_are_not_managed_paths(v):
    recipe = _recipe(command="vllm serve {model} --chat-template /opt/templates/chat.jinja").resolve()
    assert "restated-managed-path" not in _codes(validate_recipe(recipe, v=v))


# --------------------------------------------------------------------------
# Sparkrun-owned config keys
# --------------------------------------------------------------------------


def test_internal_config_key_in_defaults_is_reported(v):
    """Nothing reported this before: `launcher._is_internal_config_key`
    excludes `_`-prefixed keys from the unmapped-key report, which left a
    recipe *declaring* one with no diagnostic at all."""
    recipe = _recipe(
        model="unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
        runtime="llama-cpp",
        defaults={"_gguf_model_path": "/cache/huggingface/hub/x.gguf"},
    ).resolve()
    found = next(i for i in validate_recipe(recipe, v=v) if i.code == "internal-config-key")
    assert found.severity == WARNING
    assert "_gguf_model_path" in found.summary


def test_ordinary_defaults_keys_are_not_internal(v):
    recipe = _recipe(defaults={"port": 8000, "max_model_len": 4096}).resolve()
    assert "internal-config-key" not in _codes(validate_recipe(recipe, v=v))


# --------------------------------------------------------------------------
# Per-launch coordination flags
# --------------------------------------------------------------------------


def test_hardcoded_rendezvous_flags_are_reported(v):
    recipe = _recipe(command="vllm serve {model} --nnodes 2 --node-rank 0 --master-addr 10.0.0.5").resolve()
    found = next(i for i in validate_recipe(recipe, v=v) if i.code == "hardcoded-rendezvous-flag")
    assert found.severity == WARNING
    assert "'--nnodes'" in found.summary and "'--master-addr'" in found.summary


def test_rendezvous_flag_list_is_per_runtime(v):
    """Atlas spells the world `--rank`/`--world-size`; vLLM's `--nnodes` is not
    its vocabulary and must not be reported against it. The lists are declared
    per runtime precisely so they can drift apart."""
    from sparkrun.core.bootstrap import get_runtime
    from sparkrun.core.validation import check_hardcoded_rendezvous_flags

    atlas = get_runtime("atlas", v)
    vllm = get_runtime("vllm-distributed", v)

    nnodes = _recipe(command="spark serve {model} --nnodes 2")
    assert check_hardcoded_rendezvous_flags(nnodes, atlas) == []
    assert check_hardcoded_rendezvous_flags(nnodes, vllm) != []

    world = _recipe(command="spark serve {model} --world-size 2")
    assert check_hardcoded_rendezvous_flags(world, atlas) != []
    assert check_hardcoded_rendezvous_flags(world, vllm) == []


def test_runtime_declaring_no_rendezvous_flags_disables_the_check(v):
    """`vllm-ray` rendezvouses through Ray and `trtllm` through `mpirun -H`, so
    neither has a serve flag to protect. The empty base default is also what an
    out-of-tree runtime built against an older base class gets."""
    from sparkrun.core.bootstrap import get_runtime
    from sparkrun.core.validation import check_hardcoded_rendezvous_flags

    for name in ("vllm-ray", "trtllm"):
        runtime = get_runtime(name, v)
        assert runtime.managed_rendezvous_flags() == ()
        recipe = _recipe(runtime=name, command="vllm serve {model} --nnodes 2")
        assert check_hardcoded_rendezvous_flags(recipe, runtime) == []


def test_data_parallel_size_is_not_a_rendezvous_flag(v):
    """The one flag in vLLM's block injected only when the template did not
    supply it — writing it is a supported choice, not a collision."""
    recipe = _recipe(command="vllm serve {model} --data-parallel-size 2").resolve()
    assert "hardcoded-rendezvous-flag" not in _codes(validate_recipe(recipe, v=v))


def test_clean_recipe_produces_none_of_the_new_findings(v):
    """The corpus posture: all 160 cached registry recipes produce zero of the
    two new warnings, and a recipe using placeholders throughout produces none
    of the four."""
    recipe = _recipe(
        command="vllm serve {model} --port {port} --tensor-parallel-size {tensor_parallel}",
        defaults={"port": 8000, "tensor_parallel": 2},
    ).resolve()
    codes = _codes(validate_recipe(recipe, v=v))
    assert not codes & {
        "restated-model-arg",
        "restated-managed-path",
        "internal-config-key",
        "hardcoded-rendezvous-flag",
    }


def test_managed_container_path_in_a_defaults_value_is_reported(v):
    """A serve flag can name a path sparkrun owns just as a command can."""
    recipe = _recipe(
        command="vllm serve {model} --chat-template {chat_template}",
        defaults={"chat_template": "/cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/a/chat.jinja"},
    ).resolve()
    found = next(i for i in validate_recipe(recipe, v=v) if i.code == "restated-managed-path")
    assert found.summary.startswith("defaults.chat_template:")


def test_internal_key_holding_a_managed_path_is_reported_once(v):
    """`internal-config-key` already owns that line and says strictly more;
    a second finding on the same assignment is the noise this module avoids."""
    recipe = _recipe(defaults={"_gguf_model_path": "/cache/huggingface/hub/x.gguf"}).resolve()
    codes = [i.code for i in validate_recipe(recipe, v=v)]
    assert codes.count("internal-config-key") == 1
    assert "restated-managed-path" not in codes


# --------------------------------------------------------------------------
# Inline scripting and patching
# --------------------------------------------------------------------------


def test_heredoc_program_in_command_is_reported(v):
    """The canonical shape: an `@spark-arena` recipe whose `command:` carries a
    ~280-line Python program that rewrites vLLM's source before the serve
    line."""
    recipe = _recipe(
        command="set -euo pipefail\npython3 - <<'PY'\nprint('patching')\nPY\nexec vllm serve {model}",
    ).resolve()
    found = next(i for i in validate_recipe(recipe, v=v) if i.code == "inline-script")
    assert found.severity == SUGGESTION
    assert found.summary.startswith("command:")
    assert "heredoc" in found.summary
    assert "`mods:`" in found.fix


def test_in_place_patching_in_pre_exec_is_reported(v):
    recipe = _recipe(
        command="vllm serve {model}",
        pre_exec=["sed -i 's/foo/bar/' /usr/lib/python3/site-packages/vllm/config.py"],
    ).resolve()
    found = next(i for i in validate_recipe(recipe, v=v) if i.code == "inline-script")
    assert found.summary.startswith("pre_exec:")


def test_launch_time_package_install_is_reported(v):
    """The only instance in the cached corpus (two recipes), and the one case
    where the image is the better fix than a mod — so the advice names both."""
    recipe = _recipe(command="vllm serve {model}", pre_exec=["python3 -m pip install cohere_melody"]).resolve()
    found = next(i for i in validate_recipe(recipe, v=v) if i.code == "inline-script")
    assert "package install" in found.summary
    assert "container image" in found.fix


def test_ordinary_shell_in_a_command_is_not_an_inline_script(v):
    """Shape is deliberately not a signal. A serve command wrapped in a couple
    of shell lines is the common idiom and must stay silent — a line-count
    threshold would have reported the recipes with the most flags."""
    recipe = _recipe(
        command="set -euo pipefail\nexport VLLM_USE_V1=1\nexec vllm serve {model} --port {port}",
        defaults={"port": 8000},
    ).resolve()
    assert "inline-script" not in _codes(validate_recipe(recipe, v=v))


def test_a_recipe_using_mods_properly_is_not_reported(v):
    """The shape the finding recommends. `mods:` resolution happens inside
    `launch_inference`, long after validation, so the entries seen here are the
    author's own and never sparkrun's injected mod-runner."""
    recipe = _recipe(command="vllm serve {model}", mods=["mods/glm53-sm121-patch"]).resolve()
    assert "inline-script" not in _codes(validate_recipe(recipe, v=v))


def test_sed_without_in_place_is_not_reported(v):
    """`sed -n`/`sed -e` filtering a stream edits nothing; only `-i` does."""
    recipe = _recipe(command="vllm serve {model} --port $(sed -n 1p /etc/portfile)").resolve()
    assert "inline-script" not in _codes(validate_recipe(recipe, v=v))


def test_inline_script_in_a_defaults_value_is_reported(v):
    """A default reaches the command through its placeholder, so a program can
    be written there too — one remove further from where it runs."""
    recipe = _recipe(
        command="vllm serve {model} && {setup}",
        defaults={"setup": "python3 - <<'PY'\nimport vllm\nPY"},
    ).resolve()
    found = next(i for i in validate_recipe(recipe, v=v) if i.code == "inline-script")
    assert found.summary.startswith("defaults.setup:")


def test_site_packages_path_in_a_defaults_value_is_not_a_script(v):
    """The one signal that names a *location* rather than a verb. In a command
    it is nearly always the target of a write; in a serve-flag value it is as
    likely to be a plain path, and calling that an inline script is wrong."""
    recipe = _recipe(
        command="vllm serve {model} --chat-template {chat_template}",
        defaults={"chat_template": "/usr/lib/python3.12/site-packages/vllm/templates/tool.jinja"},
    ).resolve()
    assert "inline-script" not in _codes(validate_recipe(recipe, v=v))


def test_site_packages_write_in_a_command_is_still_a_script(v):
    """Narrowed for `defaults:` only — the command scan keeps the full set."""
    recipe = _recipe(command="cp /tmp/fix.py /usr/lib/python3/site-packages/vllm/x.py && vllm serve {model}").resolve()
    assert "inline-script" in _codes(validate_recipe(recipe, v=v))


def test_internal_key_holding_a_script_is_reported_by_both_checks(v):
    """Unlike the managed-path scan, the script scan does *not* skip
    `_`-prefixed defaults: "you declared a key sparkrun owns" and "you put a
    program in it" are different problems, and the first message covers neither
    the second's diagnosis nor its fix."""
    recipe = _recipe(defaults={"_gguf_model_path": "$(python3 -c 'print(1)')"}).resolve()
    codes = _codes(validate_recipe(recipe, v=v))
    assert "internal-config-key" in codes
    assert "inline-script" in codes
