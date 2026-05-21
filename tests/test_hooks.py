"""Tests for sparkrun.orchestration.hooks module."""

from __future__ import annotations

import base64
from unittest import mock

import pytest

from sparkrun.orchestration.hooks import (
    build_hook_context,
    render_hook_command,
    render_hook_commands,
    run_pre_exec,
    run_post_exec,
    run_post_commands,
)
from sparkrun.orchestration.ssh import RemoteResult


# ---------------------------------------------------------------------------
# build_hook_context
# ---------------------------------------------------------------------------


class TestBuildHookContext:
    """Tests for build_hook_context()."""

    def test_build_hook_context_basic(self):
        """Builds context with head_host, head_ip, port, cluster_id."""
        ctx = build_hook_context(
            {},
            head_host="spark-01",
            head_ip="192.168.1.10",
            port=8000,
            cluster_id="sparkrun_abc123",
        )
        assert ctx["head_host"] == "spark-01"
        assert ctx["head_ip"] == "192.168.1.10"
        assert ctx["port"] == "8000"
        assert ctx["cluster_id"] == "sparkrun_abc123"

    def test_build_hook_context_derives_base_url(self):
        """When head_ip and port are given, base_url is http://{head_ip}:{port}/v1."""
        ctx = build_hook_context(
            {},
            head_ip="10.0.0.5",
            port=30000,
        )
        assert ctx["base_url"] == "http://10.0.0.5:30000/v1"

    def test_build_hook_context_includes_config_chain(self):
        """Values from config_chain (dict with .keys()) are included in context."""
        config_chain = {"model": "meta-llama/Llama-2-7b-hf", "port": 8080, "empty": None}
        ctx = build_hook_context(config_chain, head_host="spark-01")
        assert ctx["model"] == "meta-llama/Llama-2-7b-hf"
        assert ctx["port"] == "8080"
        # None values are excluded
        assert "empty" not in ctx
        assert ctx["head_host"] == "spark-01"

    def test_build_hook_context_empty(self):
        """No kwargs produces minimal (empty) context."""
        ctx = build_hook_context({})
        assert ctx == {}

    def test_build_hook_context_no_base_url_without_port(self):
        """base_url is not derived when port is missing."""
        ctx = build_hook_context({}, head_ip="10.0.0.1")
        assert "base_url" not in ctx

    def test_build_hook_context_no_base_url_without_ip(self):
        """base_url is not derived when head_ip is missing."""
        ctx = build_hook_context({}, port=8000)
        assert "base_url" not in ctx

    def test_build_hook_context_optional_fields(self):
        """container_name and cache_dir are included when provided."""
        ctx = build_hook_context(
            {},
            container_name="sparkrun_abc_solo",
            cache_dir="/cache/huggingface",
        )
        assert ctx["container_name"] == "sparkrun_abc_solo"
        assert ctx["cache_dir"] == "/cache/huggingface"

    def test_build_hook_context_port_coerced_to_str(self):
        """Port value (int) is coerced to string."""
        ctx = build_hook_context({}, port=9000)
        assert ctx["port"] == "9000"

    def test_build_hook_context_config_chain_overridden_by_kwargs(self):
        """Explicit kwargs override config_chain values for the same key."""
        config_chain = {"head_host": "old-host"}
        ctx = build_hook_context(config_chain, head_host="new-host")
        assert ctx["head_host"] == "new-host"


# ---------------------------------------------------------------------------
# render_hook_command
# ---------------------------------------------------------------------------


class TestRenderHookCommand:
    """Tests for render_hook_command()."""

    def test_render_simple_substitution(self):
        """{model} placeholder is replaced with context value."""
        ctx = {"model": "meta-llama/Llama-2-7b-hf"}
        result = render_hook_command("echo {model}", ctx)
        assert result == "echo meta-llama/Llama-2-7b-hf"

    def test_render_multiple_substitutions(self):
        """Multiple placeholders are all replaced."""
        ctx = {"host": "10.0.0.1", "port": "8000", "model": "llama-7b"}
        result = render_hook_command("curl http://{host}:{port}/v1/models?model={model}", ctx)
        assert result == "curl http://10.0.0.1:8000/v1/models?model=llama-7b"

    def test_render_no_placeholders(self):
        """Command with no placeholders passes through unchanged."""
        ctx = {"model": "llama-7b"}
        result = render_hook_command("echo hello world", ctx)
        assert result == "echo hello world"

    def test_render_unknown_placeholder_unchanged(self):
        """Unknown placeholders are left as-is (not substituted)."""
        ctx = {"model": "llama-7b"}
        result = render_hook_command("echo {model} and {unknown}", ctx)
        assert "llama-7b" in result
        assert "{unknown}" in result

    def test_render_chained_substitution(self):
        """Substitution iterates until stable (handles nested substitutions)."""
        ctx = {"base_url": "http://10.0.0.1:8000/v1"}
        result = render_hook_command("curl {base_url}/models", ctx)
        assert result == "curl http://10.0.0.1:8000/v1/models"


# ---------------------------------------------------------------------------
# render_hook_commands
# ---------------------------------------------------------------------------


class TestRenderHookCommands:
    """Tests for render_hook_commands()."""

    def test_render_string_entries(self):
        """List of strings are rendered with context."""
        ctx = {"model": "llama-7b", "port": "8000"}
        commands = ["echo {model}", "curl http://localhost:{port}/v1"]
        result = render_hook_commands(commands, ctx)
        assert result == ["echo llama-7b", "curl http://localhost:8000/v1"]

    def test_render_dict_entries(self):
        """Dict entries have their string values rendered, keys preserved."""
        ctx = {"src": "/local/mods", "dest": "/workspace/mods"}
        commands = [{"copy": "{src}", "dest": "{dest}"}]
        result = render_hook_commands(commands, ctx)
        assert len(result) == 1
        rendered = result[0]
        assert isinstance(rendered, dict)
        assert rendered["copy"] == "/local/mods"
        assert rendered["dest"] == "/workspace/mods"

    def test_render_mixed_entries(self):
        """Mix of strings and dicts are both rendered."""
        ctx = {"model": "llama-7b", "path": "/tmp/mods"}
        commands = [
            "echo {model}",
            {"copy": "{path}"},
            "ls /workspace",
        ]
        result = render_hook_commands(commands, ctx)
        assert result[0] == "echo llama-7b"
        assert isinstance(result[1], dict)
        assert result[1]["copy"] == "/tmp/mods"
        assert result[2] == "ls /workspace"

    def test_render_empty_list(self):
        """Empty command list returns empty list."""
        result = render_hook_commands([], {"model": "llama-7b"})
        assert result == []

    def test_render_preserves_dict_non_string_values(self):
        """Dict entries with non-string values are preserved unchanged."""
        ctx = {"model": "llama-7b"}
        commands = [{"copy": "{model}", "count": 5}]
        result = render_hook_commands(commands, ctx)
        assert result[0]["copy"] == "llama-7b"
        assert result[0]["count"] == 5


# ---------------------------------------------------------------------------
# run_pre_exec
# ---------------------------------------------------------------------------


class TestRunPreExec:
    """Tests for run_pre_exec()."""

    def _make_success(self, host="spark-01"):
        return RemoteResult(host=host, returncode=0, stdout="ok", stderr="")

    def _make_failure(self, host="spark-01"):
        return RemoteResult(host=host, returncode=1, stdout="", stderr="command failed")

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_pre_exec_string_commands(self, mock_run):
        """String commands produce docker exec calls on each container."""
        mock_run.return_value = self._make_success()
        run_pre_exec(
            hosts_containers=[("spark-01", "sparkrun_abc_solo")],
            commands=["echo hello"],
            config_chain={},
            trust=True,
        )
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == "spark-01"
        script = call_args[0][1]
        assert "docker exec" in script
        assert "sparkrun_abc_solo" in script
        expected = base64.b64encode(b"echo hello").decode("utf-8")
        assert expected in script

    @mock.patch("sparkrun.orchestration.primitives.should_run_locally", return_value=True)
    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_pre_exec_copy_commands(self, mock_run, mock_should_run_locally):
        """Dict with 'copy' key produces docker cp calls."""
        mock_run.return_value = self._make_success()
        run_pre_exec(
            hosts_containers=[("spark-01", "sparkrun_abc_solo")],
            commands=[{"copy": "/tmp/mods/patch.sh"}],
            config_chain={},
            trust=True,
        )
        mock_run.assert_called()
        script = mock_run.call_args[0][1]
        assert "docker cp" in script or "docker exec" in script

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_pre_exec_fail_fast(self, mock_run):
        """RuntimeError raised on non-zero exit from any command."""
        mock_run.return_value = self._make_failure()
        with pytest.raises(RuntimeError, match="pre_exec"):
            run_pre_exec(
                hosts_containers=[("spark-01", "sparkrun_abc_solo")],
                commands=["bad-command"],
                config_chain={},
                trust=True,
            )

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_pre_exec_dry_run(self, mock_run):
        """In dry_run mode no actual execution occurs."""
        mock_run.return_value = self._make_success()
        run_pre_exec(
            hosts_containers=[("spark-01", "sparkrun_abc_solo")],
            commands=["echo hello"],
            config_chain={},
            dry_run=True,
            trust=True,
        )
        # run_script_on_host is still called in dry_run (it handles the dry_run flag itself)
        mock_run.assert_called()
        call_args = mock_run.call_args
        assert call_args[1].get("dry_run") is True

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_pre_exec_empty_commands(self, mock_run):
        """No-op when commands list is empty."""
        run_pre_exec(
            hosts_containers=[("spark-01", "sparkrun_abc_solo")],
            commands=[],
            config_chain={},
        )
        mock_run.assert_not_called()

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_pre_exec_multiple_containers(self, mock_run):
        """Commands run on all containers in the hosts_containers list."""
        mock_run.return_value = self._make_success()
        run_pre_exec(
            hosts_containers=[
                ("spark-01", "sparkrun_abc_node_0"),
                ("spark-02", "sparkrun_abc_node_1"),
            ],
            commands=["echo hello", "ls /workspace"],
            config_chain={},
            trust=True,
        )
        # 2 containers * 2 commands = 4 calls
        assert mock_run.call_count == 4

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_pre_exec_renders_config_chain(self, mock_run):
        """Commands have config_chain values substituted before execution."""
        mock_run.return_value = self._make_success()
        run_pre_exec(
            hosts_containers=[("spark-01", "sparkrun_abc_solo")],
            commands=["echo {model}"],
            config_chain={"model": "llama-7b"},
            trust=True,
        )
        mock_run.assert_called_once()
        script = mock_run.call_args[0][1]
        expected = base64.b64encode(b"echo llama-7b").decode("utf-8")
        assert expected in script

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_pre_exec_skips_unrecognized_entry(self, mock_run):
        """Unrecognized entries (not str or dict with 'copy') are skipped."""
        mock_run.return_value = self._make_success()
        # A dict without a 'copy' key — the code logs a warning and skips it
        run_pre_exec(
            hosts_containers=[("spark-01", "sparkrun_abc_solo")],
            commands=["echo ok", {"unknown_key": "value"}],
            config_chain={},
            trust=True,
        )
        # Only the string command triggers a call
        assert mock_run.call_count == 1

    @mock.patch("sys.stdin")
    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_pre_exec_trust_false_refuses_without_tty(self, mock_run, mock_stdin):
        """When trust=False and stdin is not a TTY, RuntimeError is raised."""
        mock_stdin.isatty.return_value = False
        mock_run.return_value = self._make_success()
        with pytest.raises(RuntimeError, match="pre_exec require confirmation"):
            run_pre_exec(
                hosts_containers=[("spark-01", "sparkrun_abc_solo")],
                commands=["echo hello"],
                config_chain={},
                trust=False,
            )
        # SSH call must NOT have happened — the gate rejects before execution
        mock_run.assert_not_called()

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_pre_exec_trust_true_skips_prompt(self, mock_run):
        """When trust=True, no confirmation prompt is required and the command runs."""
        mock_run.return_value = self._make_success()
        # We don't patch sys.stdin / click.confirm — neither should be consulted.
        with mock.patch("click.confirm") as mock_confirm:
            run_pre_exec(
                hosts_containers=[("spark-01", "sparkrun_abc_solo")],
                commands=["echo hello"],
                config_chain={},
                trust=True,
            )
            mock_confirm.assert_not_called()
        mock_run.assert_called_once()

    @mock.patch("sys.stdin")
    def test_run_pre_exec_trust_false_user_decline_raises(self, mock_stdin):
        """Interactive TTY but user declines — RuntimeError is raised."""
        mock_stdin.isatty.return_value = True
        with mock.patch("click.confirm", return_value=False):
            with pytest.raises(RuntimeError, match="pre_exec execution cancelled"):
                run_pre_exec(
                    hosts_containers=[("spark-01", "sparkrun_abc_solo")],
                    commands=["echo hello"],
                    config_chain={},
                    trust=False,
                )

    def test_run_pre_exec_trust_defaults_to_false(self):
        """Default trust value is False — third-party recipes must opt in."""
        import inspect

        sig = inspect.signature(run_pre_exec)
        assert sig.parameters["trust"].default is False


# ---------------------------------------------------------------------------
# run_post_exec
# ---------------------------------------------------------------------------


class TestRunPostExec:
    """Tests for run_post_exec()."""

    def _make_success(self, host="spark-01"):
        return RemoteResult(host=host, returncode=0, stdout="ok", stderr="")

    def _make_failure(self, host="spark-01"):
        return RemoteResult(host=host, returncode=1, stdout="", stderr="exec failed")

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_post_exec_runs_on_head(self, mock_run):
        """Commands run inside the head container via docker exec."""
        mock_run.return_value = self._make_success()
        run_post_exec(
            head_host="spark-01",
            container_name="sparkrun_abc_solo",
            commands=["curl http://localhost:8000/health"],
            context={"port": "8000"},
            trust=True,
        )
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == "spark-01"
        script = call_args[0][1]
        assert "docker exec" in script
        assert "sparkrun_abc_solo" in script
        expected = base64.b64encode(b"curl http://localhost:8000/health").decode("utf-8")
        assert expected in script

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_post_exec_fail_fast(self, mock_run):
        """RuntimeError raised when command exits with non-zero status."""
        mock_run.return_value = self._make_failure()
        with pytest.raises(RuntimeError, match="post_exec"):
            run_post_exec(
                head_host="spark-01",
                container_name="sparkrun_abc_solo",
                commands=["bad-command"],
                context={},
                trust=True,
            )

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_post_exec_empty(self, mock_run):
        """No-op when commands list is empty."""
        run_post_exec(
            head_host="spark-01",
            container_name="sparkrun_abc_solo",
            commands=[],
            context={},
        )
        mock_run.assert_not_called()

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_post_exec_renders_context(self, mock_run):
        """Commands have context values substituted before execution."""
        mock_run.return_value = self._make_success()
        run_post_exec(
            head_host="spark-01",
            container_name="sparkrun_abc_solo",
            commands=["curl {base_url}/models"],
            context={"base_url": "http://10.0.0.1:8000/v1"},
            trust=True,
        )
        mock_run.assert_called_once()
        script = mock_run.call_args[0][1]
        expected = base64.b64encode(b"curl http://10.0.0.1:8000/v1/models").decode("utf-8")
        assert expected in script

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_post_exec_multiple_commands(self, mock_run):
        """Multiple commands each trigger a separate run_script_on_host call."""
        mock_run.return_value = self._make_success()
        run_post_exec(
            head_host="spark-01",
            container_name="sparkrun_abc_solo",
            commands=["echo step1", "echo step2", "echo step3"],
            context={},
            trust=True,
        )
        assert mock_run.call_count == 3

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_post_exec_dry_run(self, mock_run):
        """In dry_run mode the call is forwarded with dry_run=True."""
        mock_run.return_value = self._make_success()
        run_post_exec(
            head_host="spark-01",
            container_name="sparkrun_abc_solo",
            commands=["echo hello"],
            context={},
            dry_run=True,
            trust=True,
        )
        mock_run.assert_called_once()
        assert mock_run.call_args[1].get("dry_run") is True

    @mock.patch("sys.stdin")
    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_post_exec_trust_false_refuses_without_tty(self, mock_run, mock_stdin):
        """When trust=False and stdin is not a TTY, RuntimeError is raised."""
        mock_stdin.isatty.return_value = False
        mock_run.return_value = self._make_success()
        with pytest.raises(RuntimeError, match="post_exec require confirmation"):
            run_post_exec(
                head_host="spark-01",
                container_name="sparkrun_abc_solo",
                commands=["echo hello"],
                context={},
                trust=False,
            )
        # SSH call must NOT have happened — the gate rejects before execution
        mock_run.assert_not_called()

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_run_post_exec_trust_true_skips_prompt(self, mock_run):
        """When trust=True, no confirmation prompt is required and the command runs."""
        mock_run.return_value = self._make_success()
        with mock.patch("click.confirm") as mock_confirm:
            run_post_exec(
                head_host="spark-01",
                container_name="sparkrun_abc_solo",
                commands=["echo hello"],
                context={},
                trust=True,
            )
            mock_confirm.assert_not_called()
        mock_run.assert_called_once()

    @mock.patch("sys.stdin")
    def test_run_post_exec_trust_false_user_decline_raises(self, mock_stdin):
        """Interactive TTY but user declines — RuntimeError is raised."""
        mock_stdin.isatty.return_value = True
        with mock.patch("click.confirm", return_value=False):
            with pytest.raises(RuntimeError, match="post_exec execution cancelled"):
                run_post_exec(
                    head_host="spark-01",
                    container_name="sparkrun_abc_solo",
                    commands=["echo hello"],
                    context={},
                    trust=False,
                )

    def test_run_post_exec_trust_defaults_to_false(self):
        """Default trust value is False — third-party recipes must opt in."""
        import inspect

        sig = inspect.signature(run_post_exec)
        assert sig.parameters["trust"].default is False


# ---------------------------------------------------------------------------
# run_post_commands
# ---------------------------------------------------------------------------


class TestRunPostCommands:
    """Tests for run_post_commands()."""

    @mock.patch("subprocess.run")
    def test_run_post_commands_success(self, mock_subproc):
        """subprocess.run is called with rendered commands."""
        mock_subproc.return_value = mock.Mock(returncode=0, stdout="done\n")
        run_post_commands(
            commands=["echo hello"],
            context={"model": "llama-7b"},
            trust=True,
        )
        mock_subproc.assert_called_once()
        call_kwargs = mock_subproc.call_args
        assert call_kwargs[0][0] == "echo hello"
        assert call_kwargs[1]["shell"] is True

    @mock.patch("subprocess.run")
    def test_run_post_commands_renders_context(self, mock_subproc):
        """Commands have context values substituted before execution."""
        mock_subproc.return_value = mock.Mock(returncode=0, stdout="")
        run_post_commands(
            commands=["curl {base_url}/health"],
            context={"base_url": "http://10.0.0.1:8000/v1"},
            trust=True,
        )
        mock_subproc.assert_called_once()
        assert mock_subproc.call_args[0][0] == "curl http://10.0.0.1:8000/v1/health"

    @mock.patch("subprocess.run")
    def test_run_post_commands_fail_fast(self, mock_subproc):
        """RuntimeError raised when subprocess exits with non-zero status."""
        mock_subproc.return_value = mock.Mock(returncode=1, stdout="error\n")
        with pytest.raises(RuntimeError, match="post_commands"):
            run_post_commands(
                commands=["bad-command"],
                context={},
                trust=True,
            )

    @mock.patch("subprocess.run")
    def test_run_post_commands_dry_run(self, mock_subproc):
        """In dry_run mode no subprocess calls are made."""
        run_post_commands(
            commands=["echo hello", "ls /tmp"],
            context={},
            dry_run=True,
            trust=True,
        )
        mock_subproc.assert_not_called()

    @mock.patch("subprocess.run")
    def test_run_post_commands_empty(self, mock_subproc):
        """No-op when commands list is empty."""
        run_post_commands(commands=[], context={})
        mock_subproc.assert_not_called()

    @mock.patch("subprocess.run")
    def test_run_post_commands_multiple(self, mock_subproc):
        """Each command produces one subprocess.run call in order."""
        mock_subproc.return_value = mock.Mock(returncode=0, stdout="")
        run_post_commands(
            commands=["echo step1", "echo step2"],
            context={},
            trust=True,
        )
        assert mock_subproc.call_count == 2
        first_cmd = mock_subproc.call_args_list[0][0][0]
        second_cmd = mock_subproc.call_args_list[1][0][0]
        assert first_cmd == "echo step1"
        assert second_cmd == "echo step2"

    @mock.patch("subprocess.run")
    def test_run_post_commands_fail_fast_stops_after_first_failure(self, mock_subproc):
        """On failure of first command, subsequent commands are not executed."""
        mock_subproc.return_value = mock.Mock(returncode=1, stdout="")
        with pytest.raises(RuntimeError):
            run_post_commands(
                commands=["bad-command", "echo should-not-run"],
                context={},
                trust=True,
            )
        # Only one call made before fail-fast
        assert mock_subproc.call_count == 1

    @mock.patch("subprocess.run")
    def test_run_post_commands_skips_non_string(self, mock_subproc):
        """Non-string entries in commands list are skipped (logged as warning)."""
        mock_subproc.return_value = mock.Mock(returncode=0, stdout="")
        run_post_commands(
            commands=["echo valid", {"not": "a string"}],
            context={},
            trust=True,
        )
        # Only the string command runs
        assert mock_subproc.call_count == 1
        assert mock_subproc.call_args[0][0] == "echo valid"


# ---------------------------------------------------------------------------
# _run_copy_command — delegated source_host support
# ---------------------------------------------------------------------------


class TestRunCopyCommandDelegated:
    """Tests for _run_copy_command with source_host (delegated mode)."""

    def _make_success(self, host="spark-01"):
        return RemoteResult(host=host, returncode=0, stdout="ok", stderr="")

    def _make_failure(self, host="spark-01"):
        return RemoteResult(host=host, returncode=1, stdout="", stderr="copy failed")

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_copy_source_host_same_as_target(self, mock_run):
        """When source_host == target host, docker cp runs directly (no rsync)."""
        from sparkrun.orchestration.hooks import _run_copy_command

        mock_run.return_value = self._make_success()
        cmd = {"copy": "/home/user/mods/patch", "dest": "/workspace/mods/patch", "source_host": "spark-01"}
        _run_copy_command("spark-01", "container1", cmd, ssh_kwargs={})
        mock_run.assert_called_once()
        script = mock_run.call_args[0][1]
        assert "docker cp" in script
        # No rsync needed when files are already on the target host
        assert "rsync" not in script

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_copy_source_host_different_from_target(self, mock_run):
        """When source_host != target host, rsync from source then docker cp."""
        from sparkrun.orchestration.hooks import _run_copy_command

        mock_run.return_value = self._make_success()
        cmd = {"copy": "/home/user/mods/patch", "dest": "/workspace/mods/patch", "source_host": "head-node"}
        _run_copy_command("worker-node", "container1", cmd, ssh_kwargs={"ssh_user": "user"})
        mock_run.assert_called_once()
        script = mock_run.call_args[0][1]
        assert "rsync" in script
        assert "head-node" in script
        assert "docker cp" in script

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_copy_source_host_with_ssh_key(self, mock_run):
        """rsync ssh options include the SSH key when provided."""
        from sparkrun.orchestration.hooks import _run_copy_command

        mock_run.return_value = self._make_success()
        cmd = {"copy": "/mods/patch", "dest": "/workspace/mods/patch", "source_host": "head"}
        _run_copy_command("worker", "container1", cmd, ssh_kwargs={"ssh_user": "u", "ssh_key": "/path/key"})
        script = mock_run.call_args[0][1]
        assert "-i /path/key" in script

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_copy_source_host_failure_raises(self, mock_run):
        """TransferError raised when delegated copy fails."""
        from sparkrun.orchestration.hooks import _run_copy_command
        from sparkrun.orchestration.transfer import TransferError

        mock_run.return_value = self._make_failure()
        cmd = {"copy": "/mods/patch", "dest": "/workspace/mods/patch", "source_host": "head"}
        with pytest.raises(TransferError, match="copy failed"):
            _run_copy_command("head", "container1", cmd, ssh_kwargs={})

    def test_copy_source_host_dry_run(self):
        """In dry_run mode, no SSH calls are made for delegated copy."""
        from sparkrun.orchestration.hooks import _run_copy_command

        cmd = {"copy": "/mods/patch", "dest": "/workspace/mods/patch", "source_host": "head"}
        with mock.patch("sparkrun.orchestration.primitives.run_script_on_host") as mock_run:
            _run_copy_command("head", "container1", cmd, ssh_kwargs={}, dry_run=True)
        mock_run.assert_not_called()

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_copy_no_source_host_still_works(self, mock_run):
        """Without source_host, original behavior is preserved (rsync from local)."""
        from sparkrun.orchestration.hooks import _run_copy_command

        mock_run.return_value = self._make_success()
        cmd = {"copy": "/local/mods/patch", "dest": "/workspace/mods/patch"}

        with mock.patch("sparkrun.orchestration.ssh.run_rsync_parallel"):
            _run_copy_command("remote-host", "container1", cmd, ssh_kwargs={"ssh_user": "u"})

        # run_script_on_host called for mkdir + docker cp
        assert mock_run.call_count >= 1


# ---------------------------------------------------------------------------
# _run_delegated_copy — input validation / injection hardening
# ---------------------------------------------------------------------------


class TestRunCopyCommandDelegatedSecurity:
    """Verify that malicious source_host and dest values are rejected."""

    def test_source_host_semicolon_raises(self):
        """source_host with a semicolon is rejected before any SSH call."""
        from sparkrun.orchestration.hooks import _run_copy_command

        cmd = {"copy": "/mods/patch", "dest": "/workspace/mods/patch", "source_host": "evil; rm -rf $HOME #"}
        with pytest.raises(ValueError, match="hostname"):
            _run_copy_command("worker", "container1", cmd, ssh_kwargs={})

    def test_source_host_backtick_raises(self):
        """source_host with backticks is rejected."""
        from sparkrun.orchestration.hooks import _run_copy_command

        cmd = {"copy": "/mods/patch", "dest": "/workspace/mods/patch", "source_host": "host`whoami`"}
        with pytest.raises(ValueError, match="hostname"):
            _run_copy_command("worker", "container1", cmd, ssh_kwargs={})

    def test_source_host_command_substitution_raises(self):
        """source_host with $(...) is rejected."""
        from sparkrun.orchestration.hooks import _run_copy_command

        cmd = {"copy": "/mods/patch", "dest": "/workspace/mods/patch", "source_host": "host$(id)"}
        with pytest.raises(ValueError, match="hostname"):
            _run_copy_command("worker", "container1", cmd, ssh_kwargs={})

    def test_source_host_newline_raises(self):
        """source_host with a newline is rejected."""
        from sparkrun.orchestration.hooks import _run_copy_command

        cmd = {"copy": "/mods/patch", "dest": "/workspace/mods/patch", "source_host": "host\nevil"}
        with pytest.raises(ValueError, match="hostname"):
            _run_copy_command("worker", "container1", cmd, ssh_kwargs={})

    def test_dest_injection_raises(self):
        """dest with shell metacharacters is rejected by safe_remote_path."""
        from sparkrun.orchestration.hooks import _run_copy_command

        cmd = {"copy": "/mods/patch", "dest": "/workspace; rm -rf /", "source_host": "head"}
        with pytest.raises(ValueError):
            _run_copy_command("worker", "container1", cmd, ssh_kwargs={})

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    def test_valid_hostname_and_ip_pass(self, mock_run):
        """Valid DNS hostname and IPv4 address pass validation and reach run_script_on_host."""
        from sparkrun.orchestration.hooks import _run_copy_command
        from sparkrun.orchestration.ssh import RemoteResult

        mock_run.return_value = RemoteResult(host="worker", returncode=0, stdout="ok", stderr="")

        # DNS hostname
        cmd = {"copy": "/mods/patch", "dest": "/workspace/mods/patch", "source_host": "host01.example.com"}
        _run_copy_command("worker", "container1", cmd, ssh_kwargs={})
        mock_run.assert_called()

        mock_run.reset_mock()

        # IPv4 address
        cmd["source_host"] = "10.0.0.5"
        _run_copy_command("worker", "container1", cmd, ssh_kwargs={})
        mock_run.assert_called()


# ---------------------------------------------------------------------------
# Error classification for _run_copy_command (rsync path)
# ---------------------------------------------------------------------------


class TestRunCopyCommandErrorClassification:
    """Verify _run_copy_command raises TransferError with classified reasons."""

    def _make_result(self, host: str, returncode: int, stderr: str = "", stdout: str = "") -> RemoteResult:
        return RemoteResult(host=host, returncode=returncode, stdout=stdout, stderr=stderr)

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    @mock.patch("sparkrun.orchestration.ssh.run_rsync_parallel")
    def test_disk_space_raises_transfer_error(self, mock_rsync, mock_run_script):
        """rsync failure with 'No space left on device' raises TransferError with classified reason."""
        from sparkrun.orchestration.hooks import _run_copy_command
        from sparkrun.orchestration.transfer import TransferError

        # mkdir succeeds
        mock_run_script.return_value = self._make_result("spark-01", 0, stdout="")
        # rsync fails with disk-space error
        mock_rsync.return_value = [
            self._make_result(
                "spark-01",
                11,
                stderr='rsync: [Receiver] mkdir "/tmp/sparkrun_hook_patch/" failed: No space left on device (28)',
            )
        ]

        cmd = {"copy": "/local/mods/patch"}
        with pytest.raises(TransferError) as exc_info:
            _run_copy_command("spark-01", "ctr1", cmd, ssh_kwargs={})

        err = str(exc_info.value)
        assert "out of disk space" in err
        assert "spark-01" in err
        # Must NOT be a bare RuntimeError
        assert type(exc_info.value) is TransferError

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    @mock.patch("sparkrun.orchestration.ssh.run_rsync_parallel")
    def test_first_failure_wins_over_downstream_lstat(self, mock_rsync, mock_run_script):
        """When mkdir fails (disk space) AND a later step would fail, the FIRST failure (mkdir) is reported."""
        from sparkrun.orchestration.hooks import _run_copy_command
        from sparkrun.orchestration.transfer import TransferError

        # mkdir fails — this is the root cause
        mock_run_script.return_value = self._make_result(
            "spark-01",
            1,
            stderr="mkdir: cannot create directory '/tmp/sparkrun_hook_patch': No space left on device",
        )
        # rsync would not even be called, but set a downstream lstat error to be safe
        mock_rsync.return_value = [
            self._make_result(
                "spark-01",
                1,
                stderr="lstat /tmp/sparkrun_hook_patch: no such file or directory",
            )
        ]

        cmd = {"copy": "/local/mods/patch"}
        with pytest.raises(TransferError) as exc_info:
            _run_copy_command("spark-01", "ctr1", cmd, ssh_kwargs={})

        err = str(exc_info.value)
        # Should report the classified reason from mkdir stderr (root cause),
        # NOT the lstat message from the downstream step.
        # "No space left on device" → classified as "out of disk space"
        assert "out of disk space" in err
        assert "lstat" not in err
        # rsync should never have been called because we failed early
        mock_rsync.assert_not_called()

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    @mock.patch("sparkrun.orchestration.ssh.run_rsync_parallel")
    def test_transfer_error_not_runtime_error(self, mock_rsync, mock_run_script):
        """Raised exception is TransferError, not bare RuntimeError."""
        from sparkrun.orchestration.hooks import _run_copy_command
        from sparkrun.orchestration.transfer import TransferError

        mock_run_script.return_value = self._make_result("spark-01", 0)
        mock_rsync.return_value = [self._make_result("spark-01", 23, stderr="permission denied")]

        cmd = {"copy": "/local/mods/patch"}
        with pytest.raises(TransferError):
            _run_copy_command("spark-01", "ctr1", cmd, ssh_kwargs={})

        # Confirm it is NOT a plain RuntimeError (TransferError != RuntimeError)
        with pytest.raises(TransferError):
            _run_copy_command("spark-01", "ctr1", cmd, ssh_kwargs={})

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    @mock.patch("sparkrun.orchestration.ssh.run_rsync_parallel")
    def test_permission_denied_classified(self, mock_rsync, mock_run_script):
        """'permission denied' in rsync stderr is classified correctly."""
        from sparkrun.orchestration.hooks import _run_copy_command
        from sparkrun.orchestration.transfer import TransferError

        mock_run_script.return_value = self._make_result("spark-02", 0)
        mock_rsync.return_value = [
            self._make_result("spark-02", 23, stderr="rsync: [sender] send_files failed to open ... permission denied")
        ]

        cmd = {"copy": "/local/mods/weights"}
        with pytest.raises(TransferError) as exc_info:
            _run_copy_command("spark-02", "ctr2", cmd, ssh_kwargs={})

        assert "permission denied" in str(exc_info.value)
        assert "spark-02" in str(exc_info.value)

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    @mock.patch("sparkrun.orchestration.ssh.run_rsync_parallel")
    def test_run_copy_command_emits_cache_status_table_on_oos(self, mock_rsync, mock_run_script):
        """When rsync fails with disk-space, format_cache_status_table is invoked."""
        from sparkrun.orchestration.hooks import _run_copy_command
        from sparkrun.orchestration.transfer import TransferError

        # mkdir succeeds, rsync fails with OOS
        mock_run_script.return_value = self._make_result("spark-01", 0)
        mock_rsync.return_value = [
            self._make_result(
                "spark-01",
                11,
                stderr="rsync: write failed: No space left on device (28)",
            )
        ]

        fake_status = {"spark-01": object()}
        fake_table = "  Host   Free Space\n  spark-01   0"

        cmd = {"copy": "/local/mods/patch"}
        with mock.patch("sparkrun.orchestration.disk_info.probe_cache_status", return_value=fake_status) as mock_probe:
            with mock.patch("sparkrun.utils.cli_formatters.format_cache_status_table", return_value=fake_table) as mock_fmt:
                with pytest.raises(TransferError) as exc_info:
                    _run_copy_command("spark-01", "ctr1", cmd, ssh_kwargs={})

                # The cache-status table must have been generated
                mock_probe.assert_called_once()
                mock_fmt.assert_called_once()

        err = str(exc_info.value)
        assert "out of disk space" in err
        assert "spark-01" in err

    @mock.patch("sparkrun.orchestration.primitives.run_script_on_host")
    @mock.patch("sparkrun.orchestration.ssh.run_rsync_parallel")
    def test_run_copy_command_uses_threaded_cache_dir(self, mock_rsync, mock_run_script):
        """A custom cache_dir threaded into _run_copy_command flows to probe_cache_status."""
        from sparkrun.orchestration.hooks import _run_copy_command
        from sparkrun.orchestration.transfer import TransferError

        mock_run_script.return_value = self._make_result("spark-01", 0)
        mock_rsync.return_value = [
            self._make_result(
                "spark-01",
                11,
                stderr="rsync: write failed: No space left on device (28)",
            )
        ]

        cmd = {"copy": "/local/mods/patch"}
        with mock.patch("sparkrun.orchestration.disk_info.probe_cache_status", return_value={}) as mock_probe:
            with pytest.raises(TransferError):
                _run_copy_command(
                    "spark-01",
                    "ctr1",
                    cmd,
                    ssh_kwargs={},
                    cache_dir="/opt/custom-hf-cache",
                )
            mock_probe.assert_called_once()
            assert mock_probe.call_args.kwargs.get("hf_cache_dir") == "/opt/custom-hf-cache"
