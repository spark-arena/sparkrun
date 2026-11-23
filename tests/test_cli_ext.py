"""Tests for the plugin CLI-command extension point (``sparkrun.cli.ext``)."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from sparkrun.cli import ext
from sparkrun.core import cli_registry


@pytest.fixture
def clean_registry(monkeypatch):
    """Isolate the module-global CLI-extension registry per test.

    The registry lives in ``core.cli_registry`` (Click-free, so plugins can
    register without dragging the CLI onto the ``api`` import path);
    ``cli.ext`` re-exports its API and owns attachment.
    """
    monkeypatch.setattr(cli_registry, "_CLI_EXTENSIONS", [])
    yield


def _leaf(name: str) -> click.Command:
    @click.command(name)
    def _cmd():
        click.echo("ran %s" % name)

    return _cmd


def test_register_is_idempotent(clean_registry):
    cmd = _leaf("foo")
    ext.register_cli_command(cmd)
    ext.register_cli_command(cmd)  # same (parent, name) -> no dup
    ext.register_cli_command(_leaf("foo"))  # same name, different object -> still no dup
    assert len(ext.registered_cli_commands()) == 1


def test_attach_top_level(clean_registry):
    root = ext.PluggableGroup(name="root")
    ext.register_cli_command(_leaf("foo"))

    ext.attach_cli_extensions(root)

    assert "foo" in root.commands


def test_attach_nested(clean_registry):
    root = click.Group("root")
    parent = click.Group("cluster")
    child = click.Group("import")
    parent.add_command(child)
    root.add_command(parent)

    ext.register_cli_command(_leaf("thunder"), parent=("cluster", "import"))
    ext.attach_cli_extensions(root)

    assert "thunder" in child.commands


def test_attach_missing_parent_is_skipped_not_fatal(clean_registry):
    root = click.Group("root")
    ext.register_cli_command(_leaf("x"), parent=("nope",))

    ext.attach_cli_extensions(root)  # must not raise

    assert "x" not in root.commands


def test_attach_does_not_clobber_builtin(clean_registry):
    root = click.Group("root")
    builtin = _leaf("dup")
    root.add_command(builtin)
    ext.register_cli_command(_leaf("dup"))

    ext.attach_cli_extensions(root)

    assert root.commands["dup"] is builtin  # built-in wins


def test_pluggable_group_attaches_on_resolution(clean_registry, monkeypatch):
    # ensure_cli_extensions initializes the application; stub it to isolate the
    # lazy-attach behavior of PluggableGroup itself.
    monkeypatch.setattr(ext, "ensure_cli_extensions", lambda root, **kw: ext.attach_cli_extensions(root))

    root = ext.PluggableGroup(name="root")
    ext.register_cli_command(_leaf("plugincmd"))

    result = CliRunner().invoke(root, ["plugincmd"])

    assert result.exit_code == 0
    assert "ran plugincmd" in result.output


def test_pluggable_group_ensure_runs_once(clean_registry, monkeypatch):
    calls = []
    monkeypatch.setattr(ext, "ensure_cli_extensions", lambda root, **kw: calls.append(root))

    root = ext.PluggableGroup(name="root")
    root.add_command(_leaf("a"))
    runner = CliRunner()
    runner.invoke(root, ["a"])
    runner.invoke(root, ["--help"])

    assert len(calls) == 1  # guarded per-instance


def test_resource_busy_renders_as_an_error_not_a_traceback(clean_registry, monkeypatch):
    from sparkrun.core.coordination import ResourceBusy

    monkeypatch.setattr(ext, "ensure_cli_extensions", lambda root, **kw: ext.attach_cli_extensions(root))
    root = ext.PluggableGroup(name="root")

    @root.command("busy")
    def busy():
        raise ResourceBusy("Execution resources are busy:\n  host:h1: held by running pid 1")

    result = CliRunner().invoke(root, ["busy"])

    assert result.exit_code == 1
    assert "Error: Execution resources are busy:" in result.output
    assert "host:h1: held by running pid 1" in result.output
    assert "Traceback" not in result.output
