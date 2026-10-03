"""Forgiven recipe mistakes still receive actionable authoring suggestions."""

from __future__ import annotations

import json
from unittest import mock

import pytest
import yaml
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.core.bootstrap import init_sparkrun
from sparkrun.core.recipe import Recipe
from sparkrun.core.validation import SUGGESTION, check_forgiven_command_shapes, validate_for_launch, validate_recipe
from sparkrun.orchestration.hooks import render_hook_commands, run_post_commands

HOOKS = ("pre_exec", "post_exec", "post_commands")
BLOCK = "value=hello\nprintf '%s\\n' \"$value\"\n"
DATA = {"model": "org/model", "runtime": "vllm-distributed", "container": "vllm/vllm-openai:v1", "builder": "docker-pull"}


@pytest.mark.parametrize("hook", HOOKS)
def test_scalar_hook_is_one_command_and_keeps_its_authoring_suggestion(hook):
    recipe = Recipe.from_dict({**DATA, hook: BLOCK})
    for candidate in (recipe, Recipe._deserialize(recipe.__getstate__())):
        assert getattr(candidate, hook) == [BLOCK]
        (issue,) = check_forgiven_command_shapes(candidate)
        assert issue.code == "scalar-hook-command-list"
        assert issue.severity == SUGGESTION
        assert hook in issue.summary
        assert "'- |'" in issue.fix
        assert not issue.deprecation
        assert candidate._raw[hook] == BLOCK


@pytest.mark.parametrize("hook", HOOKS)
def test_scalar_state_is_normalized_when_restored(hook):
    # A deepcopy of an already-normalized recipe would not exercise this branch.
    state = Recipe.from_dict(DATA).__getstate__()
    state[hook] = BLOCK
    state["_raw"][hook] = BLOCK
    recipe = Recipe._deserialize(state)
    assert getattr(recipe, hook) == [BLOCK]
    assert check_forgiven_command_shapes(recipe)[0].code == "scalar-hook-command-list"


@pytest.mark.parametrize("hook", HOOKS)
@pytest.mark.parametrize("value", [None, [], [BLOCK]])
def test_canonical_hooks_and_null_stay_quiet(hook, value):
    recipe = Recipe.from_dict({**DATA, hook: value})
    assert getattr(recipe, hook) == (value or [])
    assert check_forgiven_command_shapes(recipe) == []
    if isinstance(value, list):
        assert getattr(recipe, hook) is not value


def test_absent_hooks_stay_quiet():
    assert check_forgiven_command_shapes(Recipe.from_dict(DATA)) == []


def test_pre_exec_copy_entries_and_readonly_sequences_are_preserved():
    entries = ({"copy": "/source/{model}", "dest": "/target"}, "echo {model}")
    assert render_hook_commands(entries, {"model": "m"}) == [{"copy": "/source/m", "dest": "/target"}, "echo m"]
    assert render_hook_commands("echo {model}", {"model": "m"}) == ["echo m"]
    assert entries[0]["copy"] == "/source/{model}"


def test_scalar_post_commands_keeps_one_shell_invocation():
    with mock.patch("sparkrun.orchestration.hooks.subprocess.run", return_value=mock.Mock(returncode=0, stdout="")) as run:
        run_post_commands(BLOCK, {}, trust=True)
    run.assert_called_once()
    assert run.call_args.args[0] == BLOCK


def test_scalar_post_commands_still_requires_trust():
    with mock.patch("sys.stdin.isatty", return_value=False), mock.patch("sparkrun.orchestration.hooks.subprocess.run") as run:
        with pytest.raises(RuntimeError, match="--trust"):
            run_post_commands(BLOCK, {})
    run.assert_not_called()


@pytest.mark.parametrize("suffix", ["\\", "\\\n", "\\  \n", "\\\n  \\\n"])
def test_dangling_continuation_is_reported_without_mutating_the_source(suffix):
    command = "vllm serve {model} " + suffix
    recipe = Recipe.from_dict({**DATA, "command": command})
    for candidate in (recipe, Recipe._deserialize(recipe.__getstate__())):
        (issue,) = check_forgiven_command_shapes(candidate)
        assert (issue.severity, issue.code) == (SUGGESTION, "dangling-line-continuation")
        assert "Remove" in issue.fix
        assert candidate.command == command


@pytest.mark.parametrize(
    "command",
    [
        "vllm serve {model}",
        "vllm serve {model} \\\n  --port {port}\n",
        "vllm serve {model} --pattern value" + "\\" * 2 + "\n",
        "vllm serve {model} --pattern 'value\\'",
        "vllm serve {model} --pattern value; # comment " + "\\",
        'vllm serve {model} --pattern "value\\\\"',
        "vllm serve {model} # comment \\\n",
        "vllm serve {model} --pattern 'unfinished\\",
        'vllm serve {model} --pattern "unfinished\\',
    ],
)
def test_valid_literals_and_unfinished_quotes_are_not_diagnosed_as_continuations(command):
    assert check_forgiven_command_shapes(Recipe.from_dict({**DATA, "command": command})) == []


def test_included_scalar_hook_retains_advice(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({**DATA, "pre_exec": BLOCK}))
    child = tmp_path / "child.yaml"
    child.write_text("include: base\n")
    recipe = Recipe.load(child)
    assert recipe.pre_exec == [BLOCK]
    assert check_forgiven_command_shapes(recipe)[0].code == "scalar-hook-command-list"


def test_normal_launch_withholds_advice_and_explicit_author_threshold_shows_it():
    v = init_sparkrun()
    recipe = Recipe.from_dict({**DATA, "pre_exec": BLOCK, "command": "vllm serve {model} \\\n"})
    expected = {"scalar-hook-command-list", "dangling-line-continuation"}
    issues = validate_recipe(recipe, v=v)
    assert expected <= {issue.code for issue in issues}
    shown, failed = validate_for_launch(recipe, v=v)
    assert not failed
    assert not expected & {issue.code for issue in shown}
    shown, failed = validate_for_launch(recipe, v=v, fail_on=SUGGESTION)
    assert failed
    assert expected <= {issue.code for issue in shown}


@pytest.mark.parametrize("options,exit_code", [([], 0), (["--strict"], 0), (["--fail-on", "suggestion"], 1)])
@pytest.mark.parametrize("as_json", [False, True])
def test_recipe_validate_reports_forgiveness_and_honors_thresholds(tmp_path, options, exit_code, as_json):
    path = tmp_path / "forgiven.yaml"
    path.write_text(yaml.safe_dump({**DATA, "pre_exec": BLOCK, "command": "vllm serve {model} \\\n"}))
    argv = ["recipe", "validate", str(path), *options]
    if as_json:
        argv.append("--json")
    result = CliRunner().invoke(main, argv)
    assert result.exit_code == exit_code, result.output
    if as_json:
        report = json.loads(result.stdout)
        assert report["valid"] is True
        assert report["failed"] is bool(exit_code)
        issues = {issue["code"]: issue for issue in report["issues"]}
        for code in ("scalar-hook-command-list", "dangling-line-continuation"):
            assert issues[code]["severity"] == SUGGESTION
            assert issues[code]["fix"]
    else:
        assert "scalar-hook-command-list" in result.output
        assert "dangling-line-continuation" in result.output
        assert "Wrap the whole pre_exec block" in result.output
        assert "Remove the final continuation" in result.output
