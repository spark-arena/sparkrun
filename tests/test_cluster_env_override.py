"""``SPARKRUN_CLUSTER`` overrides the default cluster for one shell.

The override lives in ``ClusterManager.get_default``, the one place every
default-cluster consumer reads, so these tests pin the contract there and
check that the CLI and the console-free api both follow it.
"""

from __future__ import annotations

import logging

import pytest
from click.testing import CliRunner

import sparkrun.core.cluster_manager as cm_mod
from sparkrun.cli import main
from sparkrun.core.cluster_manager import DEFAULT_CLUSTER_ENV, ClusterManager


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(cm_mod, "_warned_env_overrides", set())


@pytest.fixture
def mgr(tmp_path):
    manager = ClusterManager(tmp_path)
    manager.create("lab-a", ["10.0.0.1"])
    manager.create("lab-b", ["10.0.0.2"])
    manager.set_default("lab-a")
    return manager


@pytest.fixture
def cli_mgr():
    """A ClusterManager rooted where the CLI's own context will look."""
    from sparkrun.core.bootstrap import init_sparkrun
    from sparkrun.core.config import get_config_root

    manager = ClusterManager(get_config_root(init_sparkrun()))
    manager.create("lab-a", ["10.0.0.1"])
    manager.create("lab-b", ["10.0.0.2"])
    manager.set_default("lab-a")
    return manager


def test_env_overrides_persisted_default(mgr, monkeypatch):
    monkeypatch.setenv(DEFAULT_CLUSTER_ENV, "lab-b")

    assert mgr.get_default() == "lab-b"
    assert mgr.default_override() == "lab-b"
    assert mgr.get_persisted_default() == "lab-a"
    # The marker file is untouched.
    assert mgr.default_file.read_text().strip() == "lab-a"


def test_env_override_applies_without_a_persisted_default(mgr, monkeypatch):
    mgr.unset_default()
    monkeypatch.setenv(DEFAULT_CLUSTER_ENV, "lab-b")

    assert mgr.get_default() == "lab-b"


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_env_is_unset(mgr, monkeypatch, value):
    monkeypatch.setenv(DEFAULT_CLUSTER_ENV, value)

    assert mgr.get_default() == "lab-a"
    assert mgr.default_override() is None


@pytest.mark.parametrize("value", ["no-such-cluster", "../lab-b", "-lab-b"])
def test_unknown_or_invalid_env_warns_once_and_falls_back(mgr, monkeypatch, caplog, value):
    monkeypatch.setenv(DEFAULT_CLUSTER_ENV, value)

    with caplog.at_level(logging.WARNING, logger=cm_mod.__name__):
        assert mgr.get_default() == "lab-a"
        assert mgr.get_default() == "lab-a"

    warnings = [r for r in caplog.records if DEFAULT_CLUSTER_ENV in r.getMessage()]
    assert len(warnings) == 1
    assert value in warnings[0].getMessage()


def test_delete_of_env_cluster_keeps_persisted_default(mgr, monkeypatch):
    """Deleting the env-selected cluster must not clear a marker naming another."""
    monkeypatch.setenv(DEFAULT_CLUSTER_ENV, "lab-b")

    mgr.delete("lab-b")

    assert mgr.get_persisted_default() == "lab-a"
    assert mgr.get_default() == "lab-a"


def test_api_resolve_cluster_follows_env(mgr, monkeypatch):
    from sparkrun.api._resolve import _default_cluster

    monkeypatch.setenv(DEFAULT_CLUSTER_ENV, "lab-b")

    cluster = _default_cluster(None, mgr)
    assert cluster is not None
    assert cluster.name == "lab-b"
    assert cluster.hosts == ["10.0.0.2"]


def test_core_resolve_hosts_follows_env(mgr, monkeypatch):
    from sparkrun.core.hosts import resolve_hosts

    monkeypatch.setenv(DEFAULT_CLUSTER_ENV, "lab-b")

    assert resolve_hosts(cluster_manager=mgr) == ["10.0.0.2"]


def test_explicit_cluster_beats_env(mgr, monkeypatch):
    from sparkrun.core.hosts import resolve_hosts

    monkeypatch.setenv(DEFAULT_CLUSTER_ENV, "lab-b")

    assert resolve_hosts(cluster_name="lab-a", cluster_manager=mgr) == ["10.0.0.1"]


def test_cli_cluster_default_reports_env_source(cli_mgr, monkeypatch):
    monkeypatch.setenv(DEFAULT_CLUSTER_ENV, "lab-b")
    runner = CliRunner()

    result = runner.invoke(main, ["cluster", "default"])
    assert result.exit_code == 0, result.output
    assert "lab-b (from SPARKRUN_CLUSTER)" in result.output

    result = runner.invoke(main, ["cluster", "default", "--json"])
    assert result.exit_code == 0, result.output
    assert '"default_source": "env"' in result.output


def test_cli_cluster_list_marks_env_default(cli_mgr, monkeypatch):
    monkeypatch.setenv(DEFAULT_CLUSTER_ENV, "lab-b")

    result = CliRunner().invoke(main, ["cluster", "list"])

    assert result.exit_code == 0, result.output
    assert "* lab-b" in result.output
    assert "* lab-a" not in result.output
    assert "default cluster (from SPARKRUN_CLUSTER)" in result.output


def test_cli_set_default_notes_shadowing_override(cli_mgr, monkeypatch):
    monkeypatch.setenv(DEFAULT_CLUSTER_ENV, "lab-b")

    result = CliRunner().invoke(main, ["cluster", "set-default", "lab-a"])

    assert result.exit_code == 0, result.output
    assert "SPARKRUN_CLUSTER=lab-b overrides the default cluster in this shell" in result.output


def test_cli_set_default_silent_without_override(cli_mgr):
    result = CliRunner().invoke(main, ["cluster", "set-default", "lab-b"])

    assert result.exit_code == 0, result.output
    assert "SPARKRUN_CLUSTER" not in result.output
