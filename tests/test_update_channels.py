"""Tests for update-channel selection, version display, and self-update wiring."""

from __future__ import annotations

from unittest import mock

import click
import pytest
from click.testing import CliRunner

from sparkrun.cli import main
from sparkrun.cli._self_update import channel_from_flags, describe_change, update_argv
from sparkrun.core import channels
from sparkrun.core.config import SparkrunConfig
from sparkrun.core.version import display_version


# --- channel vocabulary ---


def test_normalize_channel_defaults_and_aliases():
    assert channels.normalize_channel(None) == "stable"
    assert channels.normalize_channel("") == "stable"
    assert channels.normalize_channel("bogus") == "stable"
    assert channels.normalize_channel("BETA") == "beta"
    assert channels.normalize_channel("yolo") == "alpha"
    assert channels.normalize_channel("alpha") == "alpha"


def test_channel_requirements_map():
    assert channels.channel_requirement("stable") == "sparkrun"
    assert channels.channel_requirement("beta").endswith("@develop")
    assert channels.channel_requirement("alpha").endswith("@develop-next")
    assert channels.channel_requirement("yolo").endswith("@develop-next")


def test_is_git_channel():
    assert not channels.is_git_channel("stable")
    assert channels.is_git_channel("beta")
    assert channels.is_git_channel("alpha")


def test_channel_suffix():
    assert channels.channel_suffix("stable") == ""
    assert channels.channel_suffix("beta") == "-beta"
    assert channels.channel_suffix("alpha") == "-alpha"


# --- config accessors ---


def test_config_channel_defaults_stable(tmp_path):
    assert SparkrunConfig(tmp_path / "config.yaml").self_update_channel == "stable"


def test_config_set_channel_persists(tmp_path):
    path = tmp_path / "config.yaml"
    SparkrunConfig(path).set_self_update_channel("beta")
    reloaded = SparkrunConfig(path)
    assert reloaded.self_update_channel == "beta"
    assert reloaded.get("self_update.source") == "git"
    assert reloaded.get("self_update.requirement").endswith("@develop")


def test_config_set_channel_yolo_normalizes_to_alpha(tmp_path):
    path = tmp_path / "config.yaml"
    SparkrunConfig(path).set_self_update_channel("yolo")
    assert SparkrunConfig(path).self_update_channel == "alpha"


def test_config_unknown_channel_reads_stable(tmp_path):
    path = tmp_path / "config.yaml"
    config = SparkrunConfig(path)
    config.set("self_update.channel", "wat")
    config.save()
    assert SparkrunConfig(path).self_update_channel == "stable"


# --- version display ---


def test_display_version_stable_no_suffix(tmp_path):
    assert display_version(SparkrunConfig(tmp_path / "config.yaml"), base="0.2.40") == "0.2.40"


def test_display_version_beta_with_commit(tmp_path, monkeypatch):
    config = SparkrunConfig(tmp_path / "config.yaml")
    config.set_self_update_channel("beta")
    monkeypatch.setattr("sparkrun.core.version.installed_commit", lambda: "1a2b3c4d5e6f")
    assert display_version(config, base="0.2.40") == "0.2.40-beta+g1a2b3c4"


def test_display_version_alpha_without_commit(tmp_path, monkeypatch):
    config = SparkrunConfig(tmp_path / "config.yaml")
    config.set_self_update_channel("alpha")
    monkeypatch.setattr("sparkrun.core.version.installed_commit", lambda: None)
    assert display_version(config, base="0.3.0") == "0.3.0-alpha"


# --- flag resolution + change messaging ---


def test_channel_from_flags_none():
    assert channel_from_flags(False, False, False, False) is None


def test_channel_from_flags_yolo_is_alpha():
    assert channel_from_flags(False, False, False, True) == "alpha"


def test_channel_from_flags_alpha_and_yolo_agree():
    assert channel_from_flags(False, False, True, True) == "alpha"


def test_channel_from_flags_conflict_raises():
    with pytest.raises(click.ClickException):
        channel_from_flags(True, True, False, False)


def test_update_argv_stable_uses_upgrade():
    assert update_argv("uv", "stable") == ["uv", "tool", "upgrade", "sparkrun"]


def test_update_argv_git_uses_force_install():
    argv = update_argv("uv", "beta")
    assert argv[1:3] == ["tool", "install"] and argv[-1] == "--force" and argv[3].endswith("@develop")


def test_describe_change_stable_same():
    assert "already the latest" in describe_change("stable", ("0.2.40", None), ("0.2.40", None))


def test_describe_change_stable_updated():
    assert describe_change("stable", ("0.2.39", None), ("0.2.40", None)) == "sparkrun updated: 0.2.39 -> 0.2.40"


def test_describe_change_git_same_commit():
    msg = describe_change("beta", ("0.2.40", "abcdef123456"), ("0.2.40", "abcdef123456"))
    assert "already on the latest commit" in msg and "gabcdef1" in msg


def test_describe_change_git_new_commit():
    msg = describe_change("alpha", ("0.3.0", "aaaaaaa1"), ("0.3.0", "bbbbbbb2"))
    assert "-> gbbbbbbb" in msg


# --- CLI integration ---


def _isolate_config(monkeypatch, tmp_path):
    import sparkrun.core.config as config_module

    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_DIR", tmp_path / "config")


def _run(monkeypatch, tmp_path, args, *, stored=None, new_json='{"version": "9.9.9", "channel": "x", "commit": "deadbeefcafe"}'):
    _isolate_config(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    if stored:
        SparkrunConfig(tmp_path / "config" / "config.yaml").set_self_update_channel(stored)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:3] == ["tool", "list"]:
            return mock.Mock(returncode=0, stdout="sparkrun v1\n", stderr="")
        if cmd == ["sparkrun", "setup", "version", "--json"]:
            return mock.Mock(returncode=0, stdout=new_json, stderr="")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("subprocess.run", side_effect=fake_run):
        result = CliRunner().invoke(main, args)
    return result, calls


def _stored_channel(tmp_path):
    return SparkrunConfig(tmp_path / "config" / "config.yaml").self_update_channel


def test_top_update_beta_uses_git_and_persists(monkeypatch, tmp_path):
    result, calls = _run(monkeypatch, tmp_path, ["update", "--beta"])
    assert result.exit_code == 0, result.output
    installs = [c for c in calls if c[1:3] == ["tool", "install"]]
    assert installs and installs[0][3].endswith("@develop") and "--force" in installs[0]
    assert _stored_channel(tmp_path) == "beta"
    assert "Updating recipe registries" in result.output


def test_top_update_no_flag_uses_stored_alpha(monkeypatch, tmp_path):
    result, calls = _run(monkeypatch, tmp_path, ["update"], stored="alpha")
    assert result.exit_code == 0, result.output
    installs = [c for c in calls if c[1:3] == ["tool", "install"]]
    assert installs and installs[0][3].endswith("@develop-next")


def test_top_update_stable_uses_uv_upgrade(monkeypatch, tmp_path):
    result, calls = _run(monkeypatch, tmp_path, ["update"])
    assert result.exit_code == 0, result.output
    assert any(c[1:3] == ["tool", "upgrade"] for c in calls)


def test_top_update_switch_to_stable_warns_downgrade(monkeypatch, tmp_path):
    result, calls = _run(monkeypatch, tmp_path, ["update", "--stable"], stored="alpha")
    assert result.exit_code == 0, result.output
    assert "may downgrade" in result.output
    installs = [c for c in calls if c[1:3] == ["tool", "install"]]
    assert installs and installs[0][3] == "sparkrun"
    assert _stored_channel(tmp_path) == "stable"


def test_setup_update_beta_uses_git(monkeypatch, tmp_path):
    result, calls = _run(monkeypatch, tmp_path, ["setup", "update", "--beta"])
    assert result.exit_code == 0, result.output
    installs = [c for c in calls if c[1:3] == ["tool", "install"]]
    assert installs and installs[0][3].endswith("@develop")
    assert _stored_channel(tmp_path) == "beta"


def test_build_update_event_includes_channel():
    from sparkrun.telemetry.events import build_update_event

    event = build_update_event(
        command="sparkrun update",
        old_version="0.3.0",
        new_version="0.3.0",
        upgraded=True,
        registries=[],
        channel="alpha",
        requested_channel="alpha",
    )
    assert event["channel"] == "alpha"
    assert event["requested_channel"] == "alpha"


def test_build_update_event_omits_channel_when_none():
    from sparkrun.telemetry.events import build_update_event

    event = build_update_event(
        command="sparkrun update",
        old_version="0.3.0",
        new_version="0.3.0",
        upgraded=False,
        registries=[],
    )
    assert "channel" not in event
    assert "requested_channel" not in event


def test_top_update_survives_telemetry_failure(monkeypatch, tmp_path):
    """A telemetry emit failure (e.g. a cross-version downgrade) must not crash update."""
    import sparkrun.telemetry.emit as emit_mod

    def boom(*args, **kwargs):
        raise RuntimeError("telemetry shape mismatch across versions")

    monkeypatch.setattr(emit_mod, "emit_update_event", boom)
    result, _ = _run(monkeypatch, tmp_path, ["update", "--beta"])
    assert result.exit_code == 0, result.output


def test_setup_install_alpha_uses_git_and_persists(monkeypatch, tmp_path):
    _isolate_config(monkeypatch, tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch("subprocess.run", side_effect=fake_run), mock.patch("sparkrun.cli._setup._commands.setup_completion", mock.MagicMock()):
        result = CliRunner().invoke(main, ["setup", "install", "--alpha", "--no-update-registries", "--shell", "bash"])
    assert result.exit_code == 0, result.output
    installs = [c for c in calls if c[1:3] == ["tool", "install"]]
    assert installs and installs[0][3].endswith("@develop-next")
    assert _stored_channel(tmp_path) == "alpha"
