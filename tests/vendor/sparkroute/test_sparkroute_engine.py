# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Tests for the SparkRoute engine — bootstrap, launch, and reconcile.

The contract is documented in ``docs/SPARKROUTE_BRIDGE.md`` and SparkRoute's
public ``docs/SPARKRUN_ON_DEMAND.md`` guide. These tests cover:

* SQLite configuration mode with the credential minted **before** first start,
  because the gateway refuses that mode without one;
* ``-sparkrun-command`` is a single executable — the gateway runs it as
  ``exec.Command(command, "gateway-bridge")``, appending exactly one argument;
* reconcile is whole-set replacement under CAS, idempotent by content, and a
  409 is ordinary concurrency rather than a failure;
* sparkrun never writes a gateway config file and never restarts the gateway
  to apply a change.
"""

from __future__ import annotations

import stat
from types import SimpleNamespace
from unittest import mock

import pytest

from sparkrun.plugins.sparkroute import engine as engine_mod
from sparkrun.plugins.sparkroute.admin import AdminError, RevisionConflict
from sparkrun.plugins.sparkroute.engine import SparkrouteConfigError, SparkrouteEngine

REV_A = "a" * 64
REV_B = "b" * 64


@pytest.fixture(autouse=True)
def _gateway_feature_enabled(monkeypatch):
    """Enable the integration and register it.

    Registration is normally done by the in-tree plugin loader at bootstrap.
    Doing it here keeps this file self-contained: the gateway registry is
    process-global, so without it these tests would silently depend on some
    other test module having loaded plugins first.
    """
    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_SPARKROUTE", "1")

    from sparkrun.plugins.sparkroute import register

    register(None)


def _proxy_config(**changes):
    data = {
        "bindings": [{"recipe": "@local/qwen", "cluster": "spark-a"}],
        "aliases": {},
        "gateway_admin_port": 8081,
        "gateway_admin_host": "127.0.0.1",
        "gateway_admin_configured": True,
        "capability_policy": "permissive",
        "gateway_config": None,
        "discovered_models": [],
    }
    data.update(changes)
    return SimpleNamespace(**data)


class _MutableProxyConfig(SimpleNamespace):
    """Small proxy.yaml double for binding persistence tests."""

    def __init__(self, **changes):
        base = vars(_proxy_config()).copy()
        base.update(changes)
        super().__init__(**base)
        self.save_calls = 0

    def set_bindings(self, bindings):
        self.bindings = list(bindings)

    def set_discovered_models(self, models):
        self.discovered_models = list(models)

    def save(self):
        self.save_calls += 1


@pytest.fixture
def binary(tmp_path):
    path = tmp_path / "llm-gateway"
    path.write_bytes(b"binary")
    return path


@pytest.fixture
def engine(tmp_path):
    return SparkrouteEngine(host="127.0.0.1", port=4000, state_dir=tmp_path, proxy_config=_proxy_config())


def _document(deployments=("sparkrun:abc",)):
    return {
        "providers": [{"name": "sparkrun:openai-compatible", "type": "openai_compatible"}],
        "deployments": [{"name": name} for name in deployments],
        "virtual_models": [{"name": "qwen3-8b", "pools": []}],
    }


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def _parse_argv(cmd: list[str]) -> tuple[dict[str, str], set[str]]:
    """Split argv into ``-flag value`` pairs and bare boolean flags.

    Deliberately not a positional ``zip`` over alternating tokens: Go's flag
    package takes a bare ``-boolflag``, so one valueless flag would silently
    shift every later pair and turn a real assertion into a KeyError.
    """
    pairs: dict[str, str] = {}
    bare: set[str] = set()
    index = 1
    while index < len(cmd):
        token = cmd[index]
        if index + 1 < len(cmd) and not cmd[index + 1].startswith("-"):
            pairs[token] = cmd[index + 1]
            index += 2
        else:
            bare.add(token)
            index += 1
    return pairs, bare


def test_command_selects_sqlite_mode_with_a_loopback_admin_listener(engine, binary):
    with mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="/opt/venv/bin/sparkrun"):
        cmd = engine.build_command(binary)
    assert cmd[0] == str(binary)
    assert "-config" not in cmd, "-config is rejected in SQLite mode"
    pairs, _ = _parse_argv(cmd)
    assert pairs["-config-source"] == "sqlite"
    assert pairs["-config-sqlite"] == str(engine.credential.config_db)
    assert "-client-credentials-sqlite" not in pairs
    assert pairs["-admin-auth-mode"] == "token-file"
    assert pairs["-admin-token-file"] == str(engine.credential.admin_token_file)
    assert pairs["-data-address"] == "127.0.0.1:4000"
    # The admin listener accepts a credential that can rewrite the served
    # model set, so it is loopback regardless of the data bind host.
    assert pairs["-admin-address"] == "127.0.0.1:8081"
    assert pairs["-sparkrun-command"] == "/opt/venv/bin/sparkrun"
    assert "-sparkrun" in cmd


def test_aliases_are_discoverable_not_just_addressable(engine, binary):
    """Under LiteLLM an alias is an ordinary ``model_name`` and always shows up
    in ``GET /v1/models``.  The gateway defaults ``-list-aliases`` off, so
    sparkrun passes it or a client that enumerates models never sees one."""
    with mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"):
        _, bare = _parse_argv(engine.build_command(binary))
    assert "-list-aliases" in bare


def test_operator_document_is_passed_as_a_bootstrap_seed(tmp_path, binary):
    seed = tmp_path / "operator.yaml"
    seed.write_text("providers: []\n")
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=_proxy_config(gateway_config=str(seed)))
    with mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"):
        cmd = engine.build_command(binary)
    assert cmd[cmd.index("-config-bootstrap") + 1] == str(seed)


def test_master_key_uses_a_private_caller_token_file(engine, binary):
    """The secret stays out of argv while the gateway still requires it."""
    engine.master_key = "sk-secret"
    with mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"):
        command = engine.build_command(binary)
    pairs, _ = _parse_argv(command)
    assert "sk-secret" not in " ".join(command)
    assert pairs["-caller-auth-mode"] == "token-file"
    assert pairs["-caller-token-file"] == str(engine.credential.caller_token_file)
    assert engine.data_plane_authenticated is True


def test_admin_listener_stays_loopback_even_when_data_binds_wide(tmp_path):
    engine = SparkrouteEngine(host="0.0.0.0", port=4000, state_dir=tmp_path, proxy_config=_proxy_config())
    assert engine.data_address == "0.0.0.0:4000"
    assert engine.admin_address.startswith("127.0.0.1:")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_master_key_seeds_data_and_admin_tokens_before_launch(tmp_path, binary):
    engine = SparkrouteEngine(
        host="127.0.0.1",
        port=4000,
        master_key="sk-master",
        state_dir=tmp_path,
        proxy_config=_proxy_config(),
    )
    commands = []
    with (
        mock.patch.object(engine_mod, "ensure_binary", return_value=binary),
        mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"),
        mock.patch.object(engine, "_launch_background", side_effect=lambda command, _env: commands.append(command) or 4242),
        mock.patch.object(engine, "_await_admin_ready"),
        mock.patch.object(engine, "reconcile", return_value=(0, 0)),
    ):
        assert engine.start() == 0

    assert engine.credential.caller_token_file.read_text().strip() == "sk-master"
    assert engine.credential.read_admin_token() == "sk-master"
    assert "sk-master" not in " ".join(commands[0])
    assert engine.admin_auth_required is True


def test_admin_token_cannot_be_cleared_while_master_key_is_active(tmp_path):
    engine = SparkrouteEngine(
        master_key="sk-master",
        state_dir=tmp_path,
        proxy_config=_proxy_config(),
    )
    engine.credential.ensure_admin_token(preferred="sk-master")

    with pytest.raises(SparkrouteConfigError, match="master key"):
        engine.admin_token(clear=True)


def test_start_refuses_when_the_feature_flag_is_off(monkeypatch, engine):
    from sparkrun.proxy.gateway import GatewayUnavailableError

    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_SPARKROUTE", "0")
    with mock.patch.object(engine_mod, "ensure_binary") as acquire:
        with pytest.raises(GatewayUnavailableError):
            engine.start()
    acquire.assert_not_called()


def test_open_admin_start_does_not_create_a_token(engine, binary):
    """An ordinary start preserves the open-by-default admin state."""
    order = []
    with (
        mock.patch.object(engine_mod, "ensure_binary", return_value=binary),
        mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"),
        mock.patch.object(engine.credential, "ensure_admin_token") as ensure_admin,
        mock.patch.object(engine, "_launch_background", side_effect=lambda *_a: order.append("launch") or 4242),
        mock.patch.object(engine, "_await_admin_ready"),
        mock.patch.object(engine, "reconcile", side_effect=lambda **_k: order.append("reconcile") or (1, 0)),
    ):
        assert engine.start() == 0
    ensure_admin.assert_not_called()
    assert order == ["launch", "reconcile"]
    assert engine.admin_token() is None
    assert engine.get_state()["gateway"] == "sparkroute"


def test_start_launches_autodiscover_after_initial_reconcile(engine, binary):
    order = []
    with (
        mock.patch.object(engine_mod, "ensure_binary", return_value=binary),
        mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"),
        mock.patch.object(engine.credential, "ensure", return_value="sk"),
        mock.patch.object(engine.credential, "ensure_admin", return_value="admin"),
        mock.patch.object(engine, "_launch_background", return_value=4242),
        mock.patch.object(engine, "_await_admin_ready"),
        mock.patch.object(engine, "reconcile", side_effect=lambda **_kwargs: order.append("reconcile") or (0, 0)),
        mock.patch.object(
            engine,
            "start_autodiscover",
            side_effect=lambda **_kwargs: order.append("autodiscover") or 4343,
        ) as start_autodiscover,
    ):
        assert (
            engine.start(
                autodiscover_kwargs={
                    "interval": 30,
                    "removal_grace_sweeps": 2,
                    "host_list": ["10.0.0.1"],
                    "ssh_kwargs": {"ssh_user": "drew"},
                }
            )
            == 0
        )

    assert order == ["reconcile", "autodiscover"]
    start_autodiscover.assert_called_once_with(
        proxy_pid=4242,
        interval=30,
        removal_grace_sweeps=2,
        host_list=["10.0.0.1"],
        ssh_kwargs={"ssh_user": "drew"},
    )
    assert engine.get_state()["autodiscover_pid"] == 4343


def test_a_wide_data_bind_warns_even_though_the_admin_port_is_safe(tmp_path, binary, caplog):
    """The loopback admin listener protects the *config*, not the models.

    This gateway is launched without ``-caller-auth-mode``, so its data
    listener is unauthenticated; an unchosen ``0.0.0.0`` therefore exposes
    every served model network-wide and has to be said out loud, exactly as
    the LiteLLM engine does.
    """
    import logging

    engine = SparkrouteEngine(host="0.0.0.0", port=4000, state_dir=tmp_path, proxy_config=_proxy_config(), host_configured=False)
    with (
        mock.patch.object(engine_mod, "ensure_binary", return_value=binary),
        mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"),
        mock.patch.object(engine.credential, "ensure", return_value="sk"),
        mock.patch.object(engine.credential, "ensure_admin", return_value="admin"),
        mock.patch.object(engine, "_launch_background", return_value=4242),
        mock.patch.object(engine, "_await_admin_ready"),
        mock.patch.object(engine, "reconcile", return_value=(1, 0)),
        caplog.at_level(logging.WARNING),
    ):
        assert engine.start() == 0
    assert "SECURITY WARNING" in caplog.text
    assert "NO authentication" in caplog.text
    # A master key is forwarded through a private token file, not argv.
    caplog.clear()
    engine.master_key = "sk-secret"
    assert engine.data_plane_authenticated is True


def test_an_explicit_bind_host_silences_the_warning(tmp_path, binary, caplog):
    import logging

    engine = SparkrouteEngine(host="0.0.0.0", port=4000, state_dir=tmp_path, proxy_config=_proxy_config(), host_configured=True)
    with (
        mock.patch.object(engine_mod, "ensure_binary", return_value=binary),
        mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"),
        mock.patch.object(engine.credential, "ensure", return_value="sk"),
        mock.patch.object(engine.credential, "ensure_admin", return_value="admin"),
        mock.patch.object(engine, "_launch_background", return_value=4242),
        mock.patch.object(engine, "_await_admin_ready"),
        mock.patch.object(engine, "reconcile", return_value=(1, 0)),
        caplog.at_level(logging.WARNING),
    ):
        assert engine.start() == 0
    assert "SECURITY WARNING" not in caplog.text


def test_a_failed_reconcile_does_not_report_a_failed_start(engine, binary):
    """The process is up and serving whatever the database held; a non-zero
    exit here would leave a running gateway behind a failure."""
    with (
        mock.patch.object(engine_mod, "ensure_binary", return_value=binary),
        mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"),
        mock.patch.object(engine.credential, "ensure", return_value="sk"),
        mock.patch.object(engine.credential, "ensure_admin", return_value="admin"),
        mock.patch.object(engine, "_launch_background", return_value=4242),
        mock.patch.object(engine, "_await_admin_ready"),
        mock.patch.object(engine, "reconcile", side_effect=AdminError("unreachable")),
    ):
        assert engine.start() == 0
    assert engine.get_state()["pid"] == 4242


def test_dry_run_builds_the_desired_set_without_provisioning(engine, binary):
    """A dry run should surface an unresolvable recipe now, not on first start."""
    with (
        mock.patch.object(engine_mod, "ensure_binary", return_value=binary),
        mock.patch.object(engine, "build_desired_set", return_value=_document()) as build,
        mock.patch.object(engine.credential, "ensure") as ensure,
        mock.patch.object(engine, "_launch_background") as launch,
    ):
        assert engine.start(dry_run=True) == 0
    build.assert_called_once()
    ensure.assert_not_called()
    launch.assert_not_called()


def test_reset_refuses_while_the_gateway_is_running(engine):
    with mock.patch.object(engine, "is_running", return_value=True):
        with pytest.raises(SparkrouteConfigError, match="Stop the gateway"):
            engine.reset()


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def _client(active=REV_A, current=None, replace=None):
    client = mock.Mock()
    client.active_revision.return_value = active
    client.status.return_value = {"config_revision": active}
    client.get_set.return_value = {"document": current if current is not None else {}}
    client.replace.return_value = replace if replace is not None else {"changed": True}
    return client


def test_reconcile_replaces_the_whole_set_under_cas(engine):
    client = _client()
    with (
        mock.patch.object(engine, "admin_client", return_value=client),
        mock.patch.object(engine, "build_desired_set", return_value=_document()),
    ):
        added, removed = engine.reconcile(reason="test")
    document, expected, reason = client.replace.call_args.args
    assert expected == REV_A
    assert reason == "test"
    assert set(document) == {"providers", "deployments", "virtual_models"}
    assert (added, removed) == (3, 0)


def test_unchanged_content_is_a_no_op(engine):
    """A steady-state sweep must not create a revision or an activation."""
    client = _client(current=_document(), replace={"changed": False})
    with (
        mock.patch.object(engine, "admin_client", return_value=client),
        mock.patch.object(engine, "build_desired_set", return_value=_document()),
    ):
        assert engine.reconcile() == (0, 0)


def test_conflict_retries_with_a_fresh_revision_not_the_stale_one(engine):
    """A 409 means an operator wrote between our read and our write; retrying
    with the stale token would only conflict again."""
    client = mock.Mock()
    client.active_revision.side_effect = [REV_A, REV_B, REV_B]
    client.status.return_value = {"config_revision": REV_B}
    client.get_set.return_value = {"document": {}}
    client.replace.side_effect = [RevisionConflict("moved", status=409, code="revision_conflict"), {"changed": True}]
    with (
        mock.patch.object(engine, "admin_client", return_value=client),
        mock.patch.object(engine, "build_desired_set", return_value=_document()),
        mock.patch.object(engine_mod.time, "sleep"),
    ):
        engine.reconcile()
    assert [call.args[1] for call in client.replace.call_args_list] == [REV_A, REV_B]


def test_reconcile_waits_for_the_runtime_revision_to_catch_storage(engine):
    client = _client(active=REV_B)
    client.status.side_effect = [
        {"config_revision": REV_A},
        {"config_revision": REV_B},
    ]
    with (
        mock.patch.object(engine, "admin_client", return_value=client),
        mock.patch.object(engine, "build_desired_set", return_value=_document()),
        mock.patch.object(engine_mod.time, "sleep") as sleep,
    ):
        engine.reconcile()
    assert client.status.call_count == 2
    sleep.assert_called_once_with(engine_mod.SERVING_READY_POLL_SECONDS)


def test_reconcile_fails_boundedly_when_runtime_never_catches_storage(engine):
    client = _client(active=REV_B)
    client.status.return_value = {"config_revision": REV_A}
    with (
        mock.patch.object(engine, "admin_client", return_value=client),
        mock.patch.object(engine, "build_desired_set", return_value=_document()),
        mock.patch.object(engine_mod.time, "monotonic", side_effect=[0.0, 31.0]),
    ):
        with pytest.raises(AdminError, match="did not reach the serving runtime"):
            engine.reconcile()


def test_persistent_conflict_gives_up_rather_than_spinning(engine):
    client = _client()
    client.replace.side_effect = RevisionConflict("moved", status=409, code="revision_conflict")
    with (
        mock.patch.object(engine, "admin_client", return_value=client),
        mock.patch.object(engine, "build_desired_set", return_value=_document()),
        mock.patch.object(engine_mod.time, "sleep"),
    ):
        with pytest.raises(AdminError, match="gave up"):
            engine.reconcile()
    assert client.replace.call_count == engine_mod.RECONCILE_ATTEMPTS


def test_invalid_configuration_surfaces_the_gateways_own_diagnostic(engine):
    detail = 'alias "fast" owned by "sparkrun" conflicts with virtual model "fast" owned by "operator"'
    client = _client()
    client.replace.side_effect = AdminError(detail, status=400, code="invalid_configuration")
    with (
        mock.patch.object(engine, "admin_client", return_value=client),
        mock.patch.object(engine, "build_desired_set", return_value=_document()),
    ):
        with pytest.raises(SparkrouteConfigError, match="owned by"):
            engine.reconcile()


def test_alias_and_sync_commands_both_reconcile(engine):
    engine.proxy_config = _MutableProxyConfig(bindings=[])
    endpoint = SimpleNamespace(
        healthy=True,
        actual_models=["deepseek-ai/DeepSeek-V4-Flash-0731"],
        served_model_name=None,
        model="ignored/fallback",
    )
    with mock.patch.object(engine, "reconcile", return_value=(1, 0)) as reconcile:
        assert engine.sync_aliases({"fast": "qwen3-8b"}) == (1, 0)
        assert engine.sync_models([endpoint], {"fast": "qwen3-8b"}) == (1, 0)
    assert [call.args[0] for call in reconcile.call_args_list] == [{"fast": "qwen3-8b"}, {"fast": "qwen3-8b"}]
    assert engine.proxy_config.discovered_models == ["deepseek-ai/DeepSeek-V4-Flash-0731"]


def test_sync_persists_only_healthy_actual_models_and_fallbacks(tmp_path):
    config = _MutableProxyConfig(bindings=[], discovered_models=["stale/model"])
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    endpoints = [
        SimpleNamespace(healthy=True, actual_models=["served/a", "served/b"], served_model_name=None, model="metadata/a"),
        SimpleNamespace(healthy=True, actual_models=[], served_model_name="served/fallback", model="metadata/b"),
        SimpleNamespace(healthy=False, actual_models=["unhealthy/model"], served_model_name=None, model="metadata/c"),
    ]
    with mock.patch.object(engine, "reconcile", return_value=(3, 1)):
        assert engine.sync_models(endpoints) == (3, 1)
    assert config.discovered_models == ["served/a", "served/b", "served/fallback"]
    assert config.save_calls == 1


def test_sync_does_not_rewrite_unchanged_discovery_snapshot(tmp_path):
    config = _MutableProxyConfig(bindings=[], discovered_models=["served/a"])
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    endpoint = SimpleNamespace(healthy=True, actual_models=["served/a"], served_model_name=None, model="metadata/a")
    with mock.patch.object(engine, "reconcile", return_value=(0, 0)):
        assert engine.sync_models([endpoint]) == (0, 0)
    assert config.save_calls == 0


def test_sync_does_not_remember_a_model_that_already_has_a_binding(tmp_path):
    config = _MutableProxyConfig(bindings=[{"recipe": "@official/qwen"}], discovered_models=["served/qwen"])
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    endpoint = SimpleNamespace(healthy=True, actual_models=["served/qwen"], served_model_name=None, model="metadata/qwen")
    bound = SimpleNamespace(recipe="@official/qwen", recipe_revision="revision-a", virtual_model="served/qwen")

    with (
        mock.patch.object(engine_mod, "resolve_bindings", return_value=[bound]),
        mock.patch.object(engine, "reconcile", return_value=(0, 0)),
    ):
        assert engine.sync_models([endpoint]) == (0, 0)

    assert config.discovered_models == []
    assert config.save_calls == 1


@pytest.mark.parametrize("bindings", [[], [{"recipe": "@official/qwen", "cluster": "spark-a"}]])
def test_manual_load_requests_discovery_without_creating_or_changing_bindings(tmp_path, bindings):
    config = _MutableProxyConfig(bindings=bindings, discovered_models=["served/qwen"])
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    with mock.patch.object(engine, "reconcile") as reconcile:
        assert engine.register_loaded_model("@official/qwen", {"tensor_parallel": 2}, "spark-a") is None
    assert config.bindings == bindings
    assert config.discovered_models == ["served/qwen"]
    assert config.save_calls == 0
    reconcile.assert_not_called()


def test_manual_load_and_unload_use_host_discovery_and_remove_a_stopped_workload(tmp_path):
    from sparkrun.api.proxy import _ops

    config = _MutableProxyConfig(bindings=[])
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    endpoint = SimpleNamespace(healthy=True, actual_models=["served/qwen"], served_model_name=None, model="metadata/qwen")
    with (
        mock.patch.object(_ops, "_running_engine", return_value=engine),
        mock.patch.object(engine, "is_running", return_value=True),
        mock.patch.object(_ops, "_discover", side_effect=[[endpoint], []]),
        mock.patch.object(engine, "reconcile", return_value=(1, 0)),
    ):
        assert _ops.register_loaded_model("@official/qwen").proxy_running
        assert config.bindings == []
        assert config.discovered_models == ["served/qwen"]
        assert _ops.unregister_loaded_model("@official/qwen").proxy_running
        assert config.bindings == []
        assert config.discovered_models == []


def test_manual_unload_removes_all_override_variants(tmp_path):
    bindings = [
        {"recipe": "@alias/qwen", "overrides": {"tensor_parallel": "1"}},
        {"recipe": "@official/qwen", "overrides": {"tensor_parallel": "2"}},
        {"recipe": "@official/other"},
    ]
    config = _MutableProxyConfig(bindings=bindings)
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)

    def resolve(reference, **kwargs):
        path = "/recipes/other.yaml" if reference == "@official/other" else "/recipes/qwen.yaml"
        return SimpleNamespace(source_path=path), {}

    with (
        mock.patch("sparkrun.api.resolve_catalog_recipe", side_effect=resolve),
        mock.patch.object(engine, "reconcile") as reconcile,
    ):
        assert engine.unregister_loaded_model("@official/qwen") is None

    assert config.bindings == [{"recipe": "@official/other"}]
    assert config.save_calls == 1
    reconcile.assert_not_called()


@pytest.mark.parametrize("target", ["@official/qwen", "source-path"])
def test_unload_matches_registry_reference_and_cached_path(tmp_path, target):
    recipe_path = tmp_path / "qwen.yaml"
    other_path = tmp_path / "other.yaml"
    for path in (recipe_path, other_path):
        path.write_text('sparkrun_version: "2"\nmodel: same/model\nruntime: sglang\n')
    config = _MutableProxyConfig(
        bindings=[{"recipe": str(recipe_path)}, {"recipe": "@official/qwen", "overrides": {"port": 30001}}, {"recipe": str(other_path)}]
    )
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    from sparkrun.api._context import resolve_sctx

    sctx = resolve_sctx(None)
    engine.sctx = sctx
    with mock.patch.object(sctx.registry_manager, "find_recipe_in_registries", return_value=[("official", recipe_path)]):
        engine.unregister_loaded_model(str(recipe_path) if target == "source-path" else target)
    # Same model/content is not enough to remove an independent source file.
    assert config.bindings == [{"recipe": str(other_path)}]
    assert config.save_calls == 1


def test_unload_resolution_failure_does_not_partially_remove_bindings(tmp_path):
    bindings = [{"recipe": "/recipes/qwen.yaml"}, {"recipe": "missing"}]
    config = _MutableProxyConfig(bindings=bindings)
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    import sparkrun.api as api

    with mock.patch(
        "sparkrun.api.resolve_catalog_recipe",
        side_effect=[
            (SimpleNamespace(source_path="/recipes/qwen.yaml"), {}),
            (SimpleNamespace(source_path="/recipes/qwen.yaml"), {}),
            api.SparkrunError("missing recipe"),
        ],
    ):
        with pytest.raises(SparkrouteConfigError, match="missing recipe"):
            engine.unregister_loaded_model("@official/qwen")
    assert config.bindings == bindings
    assert config.save_calls == 0


def test_admin_client_tracks_live_token_changes_without_reconstruction(tmp_path):
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=_proxy_config())
    assert engine.admin_client()._token is None

    token = engine.admin_token(rotate=True)
    assert token
    assert engine.admin_client()._token == token

    assert engine.admin_token(clear=True) is None
    assert engine.admin_client()._token is None


# ---------------------------------------------------------------------------
# Read-back
# ---------------------------------------------------------------------------


def test_models_come_from_status_so_operator_entries_are_included(engine):
    """sparkrun's reconcile role cannot read the operator document, so
    /v1/status is the only complete answer to "what can I call?"."""
    client = mock.Mock()
    client.status.return_value = {"served_model_names": ["fast", "operator-model", "qwen3-8b"]}
    with mock.patch.object(engine, "admin_client", return_value=client):
        models = engine.list_models_via_api()
    assert [m["model_name"] for m in models] == ["fast", "operator-model", "qwen3-8b"]
    assert {m["api_base"] for m in models} == {"http://127.0.0.1:4000/v1"}


def test_models_degrade_to_empty_when_the_gateway_is_unreachable(engine):
    client = mock.Mock()
    client.status.side_effect = AdminError("unreachable")
    with mock.patch.object(engine, "admin_client", return_value=client):
        assert engine.list_models_via_api() == []
    assert engine.model_query_error == "unreachable"


def test_models_reject_a_status_response_without_the_contract_field(engine):
    client = mock.Mock()
    client.status.return_value = {"config_revision": REV_B}
    with mock.patch.object(engine, "admin_client", return_value=client):
        assert engine.list_models_via_api() == []
    assert "missing served_model_names" in engine.model_query_error


def test_serving_revision_is_read_from_status(engine):
    client = mock.Mock()
    client.status.return_value = {"config_revision": REV_B}
    with mock.patch.object(engine, "admin_client", return_value=client):
        assert engine.serving_revision() == REV_B


def test_prepare_config_writes_no_file_and_previews_aliases(engine):
    engine.proxy_config = _MutableProxyConfig(bindings=[])
    with mock.patch.object(engine, "build_desired_set", return_value=_document()):
        path, applied, pending = engine.prepare_config([], {"fast": "qwen3-8b", "orphan": "absent"})
    assert path is None
    assert (applied, pending) == ({"fast"}, {"orphan"})


def test_prepare_config_imports_the_startup_discovery_snapshot(tmp_path):
    config = _MutableProxyConfig(bindings=[])
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    endpoint = SimpleNamespace(
        healthy=True,
        actual_models=["deepseek-ai/DeepSeek-V4-Flash-0731"],
        served_model_name=None,
        model="metadata/fallback",
    )

    _path, applied, pending = engine.prepare_config([endpoint], {}, write=True)

    assert config.discovered_models == ["deepseek-ai/DeepSeek-V4-Flash-0731"]
    assert config.save_calls == 1
    assert applied == set()
    assert pending == set()


def test_prepare_config_dry_run_does_not_persist_discovery(tmp_path):
    config = _MutableProxyConfig(bindings=[])
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    endpoint = SimpleNamespace(
        healthy=True,
        actual_models=["served/model"],
        served_model_name=None,
        model="metadata/fallback",
    )

    engine.prepare_config([endpoint], {}, write=False)

    assert config.discovered_models == []
    assert config.save_calls == 0
    assert not engine._discovery_labels_path.exists()


def test_autodiscover_is_safe_for_the_warm_only_snapshot(engine):
    assert engine.supports_autodiscover is True


def test_provider_rename_migrates_atomically_in_one_replacement(engine):
    """The old provider identity must never exist alongside deployments that
    still point at it. Whole-set replacement makes that structural: there is no
    intermediate document in which a deployment references a removed provider."""
    from sparkrun.plugins.sparkroute.projection import ProjectedBinding, build_sparkrun_set

    stale = {
        "providers": [{"name": "sparkrun:openai-compatible", "type": "openai_compatible"}],
        "deployments": [{"name": "sparkrun:abc", "provider": "sparkrun:openai-compatible", "model": "m"}],
        "virtual_models": [],
    }
    desired = build_sparkrun_set([ProjectedBinding(recipe="@local/x", recipe_revision="abc", model="m", virtual_model="m")])
    client = _client(current=stale)
    with (
        mock.patch.object(engine, "admin_client", return_value=client),
        mock.patch.object(engine, "build_desired_set", return_value=desired),
    ):
        engine.reconcile()

    assert client.replace.call_count == 1
    document = client.replace.call_args.args[0]
    assert [p["name"] for p in document["providers"]] == ["sparkrun"]
    assert {d["provider"] for d in document["deployments"]} == {"sparkrun"}


# ---------------------------------------------------------------------------
# Management paths resolve their engine from the state file
# ---------------------------------------------------------------------------
#
# Both of these were found by running the gateway for real, and both are the
# same shape: `_running_engine` constructs an engine from the state file with
# no proxy.yaml, so anything the engine needs from config has to arrive some
# other way or it silently degrades to a default.


def test_admin_port_falls_back_to_the_running_state(tmp_path):
    """Without this, `proxy models` talks to the default admin port and reports
    nothing while the gateway is serving on a configured one."""
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=_proxy_config(gateway_admin_port=8091))
    engine._save_state(4242)

    unconfigured = SparkrouteEngine(state_dir=tmp_path)
    assert unconfigured.proxy_config is None
    assert unconfigured.admin_port == 8091

    # Configured still wins: `start` must bind where the user asked, and at
    # that point the state file still describes the previous run.
    configured = SparkrouteEngine(state_dir=tmp_path, proxy_config=_proxy_config(gateway_admin_port=9000))
    # PID 4242 can exist on a busy test host; this assertion is specifically
    # about a stale state file, so make that precondition deterministic.
    with mock.patch.object(configured, "is_running", return_value=False):
        assert configured.admin_port == 9000


def test_management_engines_are_given_proxy_config(tmp_path, monkeypatch):
    """A reconcile without proxy.yaml computes an *empty* desired state, so
    `proxy alias add` would replace the running configuration with nothing —
    deleting every deployment it was not told about."""
    from sparkrun.api.proxy import _ops

    monkeypatch.setenv("SPARKRUN_FEATURE_GATEWAY_SPARKROUTE", "1")
    proxy_config = _proxy_config()
    sctx = SimpleNamespace(proxy_config=proxy_config)
    with mock.patch.object(_ops, "resolve_sctx", return_value=sctx):
        kwargs = _ops._engine_config_kwargs("sparkroute", None)
    assert kwargs["proxy_config"] is proxy_config

    # LiteLLM does not declare the capability, so it is left alone.
    assert _ops._engine_config_kwargs("litellm", None) == {}


def test_an_engine_without_bindings_would_wipe_the_managed_set(tmp_path):
    """The failure the test above prevents, stated directly."""
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=None)
    assert engine.build_desired_set({})["deployments"] == []


# ---------------------------------------------------------------------------
# Admin console
# ---------------------------------------------------------------------------


def test_console_url_follows_the_admin_listener(tmp_path):
    """There is no enable/disable: the console lives on the admin listener,
    which sparkrun requires in order to reconcile at all."""
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=_proxy_config(gateway_admin_port=8091))
    assert engine.ui_url == "http://127.0.0.1:8091/admin"


def test_default_console_is_colocated_on_the_data_listener(tmp_path, binary):
    from sparkrun.proxy.config import ProxyConfig

    config = ProxyConfig(tmp_path / "proxy.yaml")
    engine = SparkrouteEngine(host="0.0.0.0", port=4000, state_dir=tmp_path / "state", proxy_config=config)
    with mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"):
        command = engine.build_command(binary)

    assert engine.admin_separate is False
    assert engine.admin_address == "0.0.0.0:4000"
    assert engine.ui_url == "http://127.0.0.1:4000/admin"
    assert "-admin-address" not in command


def test_explicit_admin_host_or_port_keeps_a_separate_listener(tmp_path, binary):
    from sparkrun.proxy.config import ProxyConfig

    config = ProxyConfig(tmp_path / "proxy.yaml")
    config.set_proxy(gateway_admin_port=8091)
    engine = SparkrouteEngine(host="0.0.0.0", port=4000, state_dir=tmp_path / "state", proxy_config=config)
    with mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"):
        command = engine.build_command(binary)

    assert engine.admin_separate is True
    assert engine.admin_address == "127.0.0.1:8091"
    index = command.index("-admin-address")
    assert command[index + 1] == "127.0.0.1:8091"


def test_open_admin_mode_needs_no_credential_database(tmp_path, binary):
    config = _proxy_config(gateway_admin_configured=False)
    engine = SparkrouteEngine(host="127.0.0.1", port=4000, state_dir=tmp_path, proxy_config=config)
    with mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"):
        command = engine.build_command(binary)

    assert command[command.index("-admin-auth-mode") + 1] == "token-file"
    assert command[command.index("-admin-token-file") + 1] == str(engine.credential.admin_token_file)
    assert "-client-credentials-sqlite" not in command
    assert engine.admin_token() is None


def test_open_admin_nonloopback_override_is_forwarded(tmp_path, binary):
    config = _proxy_config(
        gateway_admin_configured=False,
        gateway_allow_insecure_admin_nonloopback=True,
    )
    engine = SparkrouteEngine(host="0.0.0.0", port=4000, state_dir=tmp_path, proxy_config=config)
    with mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"):
        command = engine.build_command(binary)

    assert "-allow-insecure-admin-nonloopback" in command


def test_open_admin_nonloopback_warns_loudly_but_starts(tmp_path, binary, caplog):
    """Unauthenticated admin is the default, so refusing to start would refuse
    the default configuration. The exposure is announced instead — including on
    a dry run, which is where an operator should be able to see it without
    opening the listener first."""
    config = _proxy_config(gateway_admin_configured=False)
    engine = SparkrouteEngine(host="0.0.0.0", port=4000, state_dir=tmp_path, proxy_config=config)

    with (
        mock.patch.object(engine_mod, "ensure_binary", return_value=binary),
        mock.patch.object(engine, "build_desired_set", return_value={"deployments": []}),
    ):
        assert engine.start(dry_run=True) == 0
    assert "DANGER" in caplog.text
    assert "admin-token set" in caplog.text


def test_default_admin_auth_needs_no_token(tmp_path, binary):
    """The whole point: a stock install serves its console on the data port
    with no credential and no credential database."""
    from sparkrun.proxy.config import ProxyConfig

    config = ProxyConfig(tmp_path / "proxy.yaml")
    engine = SparkrouteEngine(host="127.0.0.1", port=4000, state_dir=tmp_path / "state", proxy_config=config)
    with mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"):
        command = engine.build_command(binary)

    assert engine.admin_auth_required is False
    assert engine.ui_url == "http://127.0.0.1:4000/admin"
    assert command[command.index("-admin-auth-mode") + 1] == "token-file"
    assert "-client-credentials-sqlite" not in command
    assert engine.admin_token() is None


def test_default_open_auth_implies_the_nonloopback_permission(tmp_path, binary):
    """The gateway rejects unauthenticated admin off-loopback unless told
    otherwise, so with the legacy 0.0.0.0 bind the default configuration would
    otherwise spawn a process that refuses its own arguments."""
    from sparkrun.proxy.config import ProxyConfig

    config = ProxyConfig(tmp_path / "proxy.yaml")
    engine = SparkrouteEngine(host="0.0.0.0", port=4000, state_dir=tmp_path / "state", proxy_config=config)
    with mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"):
        command = engine.build_command(binary)

    assert config.gateway_allow_insecure_admin_nonloopback is False
    assert "-allow-insecure-admin-nonloopback" in command


def test_public_token_file_gets_nonloopback_permission_even_when_currently_secured(tmp_path, binary):
    """The live token can be cleared, so the process must remain valid afterward."""
    config = _proxy_config(gateway_admin_host="0.0.0.0")
    engine = SparkrouteEngine(host="0.0.0.0", port=4000, state_dir=tmp_path, proxy_config=config)
    engine.credential.ensure_admin_token()
    with mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"):
        command = engine.build_command(binary)

    assert engine.admin_exposed is True
    assert engine.admin_auth_required is True
    assert "-allow-insecure-admin-nonloopback" in command


def test_console_credential_enables_live_auth_and_is_stored_owner_only(tmp_path):
    engine = SparkrouteEngine(
        state_dir=tmp_path,
        proxy_config=_proxy_config(),
    )

    token = engine.issue_ui_credential()

    assert token.startswith("sr_admin_v1.")
    assert engine.admin_auth_required is True
    assert engine.credential.read_admin_token() == token
    assert stat.S_IMODE(engine.credential.admin_token_file.stat().st_mode) == 0o600


def test_admin_token_rotation_replaces_the_live_token_without_a_binary(tmp_path):
    engine = SparkrouteEngine(
        state_dir=tmp_path,
        proxy_config=_proxy_config(),
    )
    first = engine.issue_ui_credential()

    second = engine.admin_token(rotate=True)

    assert second and second != first
    assert engine.credential.read_admin_token() == second


def test_live_admin_token_state_persists_across_engine_reconstruction(tmp_path):
    state_dir = tmp_path / "state"
    first = SparkrouteEngine(state_dir=state_dir, proxy_config=_proxy_config())
    token = first.issue_ui_credential()

    second = SparkrouteEngine(state_dir=state_dir, proxy_config=_proxy_config())
    assert second.admin_token() == token
    assert second.admin_auth_required is True

    second.admin_token(clear=True)
    third = SparkrouteEngine(state_dir=state_dir, proxy_config=_proxy_config())
    assert third.admin_token() is None
    assert third.admin_auth_required is False


def test_litellm_reports_no_console_rather_than_failing():
    """LiteLLM simply has no admin console — an answer, not an error."""
    from sparkrun.api.proxy import ProxyUnsupported
    from sparkrun.api.proxy import _ops
    from sparkrun.proxy.engine import ProxyEngine

    assert not hasattr(ProxyEngine, "ui_url")
    with mock.patch.object(_ops, "_running_engine", return_value=ProxyEngine()):
        with pytest.raises(ProxyUnsupported, match="does not serve an admin console"):
            _ops.ui()


def test_admin_listener_is_loopback_until_explicitly_widened(tmp_path):
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=_proxy_config())
    assert engine.admin_bind_host == "127.0.0.1"
    assert engine.admin_exposed is False


def test_a_wildcard_bind_still_prints_a_connectable_url(tmp_path):
    """`0.0.0.0` is a bind address, not somewhere a browser can go — and the
    URL sparkrun prints has to be one a human can actually open."""
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=_proxy_config(gateway_admin_host="0.0.0.0"))
    assert engine.admin_address == "0.0.0.0:8081"
    assert engine.ui_url == "http://127.0.0.1:8081/admin"
    assert engine.admin_exposed is True


def test_a_specific_bind_is_used_for_both_bind_and_connect(tmp_path):
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=_proxy_config(gateway_admin_host="10.0.0.5"))
    assert engine.admin_address == "10.0.0.5:8081"
    assert engine.ui_url == "http://10.0.0.5:8081/admin"
    assert engine.admin_exposed is True


def test_the_bind_host_is_recorded_so_management_can_follow_it(tmp_path):
    """Same reason the port is recorded: `_running_engine` has no proxy.yaml."""
    SparkrouteEngine(state_dir=tmp_path, proxy_config=_proxy_config(gateway_admin_host="10.0.0.5"))._save_state(7)
    assert SparkrouteEngine(state_dir=tmp_path).admin_bind_host == "10.0.0.5"


def test_exposing_the_console_warns(tmp_path, binary, caplog):
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=_proxy_config(gateway_admin_host="0.0.0.0"))
    engine.credential.ensure_admin_token()
    with (
        mock.patch.object(engine_mod, "ensure_binary", return_value=binary),
        mock.patch.object(engine_mod, "resolve_sparkrun_executable", return_value="sparkrun"),
        mock.patch.object(engine, "_launch_background", return_value=1),
        mock.patch.object(engine, "_await_admin_ready"),
        mock.patch.object(engine, "reconcile", return_value=(0, 0)),
        caplog.at_level("WARNING"),
    ):
        engine.start()
    assert "reachable off this host" in caplog.text.lower()


def test_discovered_titles_follow_healthy_clusters_and_survive_cli_restart(tmp_path):
    config = _MutableProxyConfig(bindings=[])
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    jobs = [
        SimpleNamespace(cluster_id="opaque-a", metadata={"cluster": "spark-a"}),
        SimpleNamespace(cluster_id="opaque-b", metadata={"cluster": "spark-b"}),
        SimpleNamespace(cluster_id="stale-job", metadata={"cluster": "old-cluster", "model": "served-model"}),
    ]
    endpoints = [
        SimpleNamespace(cluster_id="opaque-a", healthy=True, actual_models=["served-model"]),
        SimpleNamespace(cluster_id="opaque-b", healthy=True, actual_models=["served-model"]),
        SimpleNamespace(cluster_id="stale-job", healthy=False, actual_models=["served-model"]),
    ]
    with mock.patch("sparkrun.api.list_jobs", return_value=jobs), mock.patch.object(engine, "reconcile", return_value=(1, 0)):
        engine.sync_models(endpoints)
        first = engine.build_desired_set()
        assert first["deployments"][0]["title"] == "sparkrun:spark-a,spark-b:served-model"
        restarted = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
        assert restarted.build_desired_set() == first
        engine.sync_models(endpoints[1:])
        second = engine.build_desired_set()
        assert second["deployments"][0]["title"] == "sparkrun:spark-b:served-model"
        assert second["deployments"][0]["name"] == first["deployments"][0]["name"]
        engine.sync_models([])
        assert SparkrouteEngine(state_dir=tmp_path, proxy_config=config)._read_discovery_labels() == {}


def test_discovered_title_fallbacks_and_corrupt_optional_cache(tmp_path):
    config = _MutableProxyConfig(bindings=[])
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    endpoint = SimpleNamespace(cluster_id="opaque-a", healthy=True, actual_models=["served-model"])
    with mock.patch("sparkrun.api.list_jobs", side_effect=OSError("metadata unavailable")):
        engine.prepare_config([endpoint], {})
    document = engine.build_desired_set()
    assert document["deployments"][0]["title"] == "sparkrun:opaque-a:served-model"
    engine._discovery_labels_path.write_text("{broken")
    restarted = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    fallback = restarted.build_desired_set()
    assert fallback["deployments"][0]["name"] == document["deployments"][0]["name"]
    assert fallback["deployments"][0]["title"] == "sparkrun:discovered:served-model"


def test_bound_titles_use_live_job_fingerprints_even_without_configured_clusters(tmp_path):
    from sparkrun.plugins.sparkroute.projection import ProjectedBinding

    config = _MutableProxyConfig(bindings=[{"recipe": "@official/deepseek"}])
    engine = SparkrouteEngine(state_dir=tmp_path, proxy_config=config)
    bound = ProjectedBinding(recipe="@official/deepseek", recipe_revision="5813420d8cac", model="deepseek", virtual_model="deepseek")
    jobs = [
        SimpleNamespace(cluster_id="opaque-job", metadata={"cluster": "real-cluster", "recipe_fingerprint": bound.recipe_revision}),
        SimpleNamespace(cluster_id="same-model-other-recipe", metadata={"cluster": "other-cluster", "recipe_fingerprint": "other-recipe"}),
        SimpleNamespace(cluster_id="stale", metadata={"cluster": "stale-cluster", "recipe_fingerprint": bound.recipe_revision}),
    ]
    endpoints = [SimpleNamespace(cluster_id=job.cluster_id, healthy=job.cluster_id != "stale", actual_models=["deepseek"]) for job in jobs]
    with mock.patch.object(engine_mod, "resolve_bindings", return_value=[bound]), mock.patch("sparkrun.api.list_jobs", return_value=jobs):
        before = engine.build_desired_set()["deployments"][0]
        engine.prepare_config(endpoints, {}, write=False)
        assert not engine._discovery_labels_path.exists()
        engine.prepare_config(endpoints, {}, write=True)
        current = engine.build_desired_set()["deployments"][0]
        assert current["title"] == "sparkrun:real-cluster:deepseek"
        assert current["name"] == before["name"] == "sparkrun:5813420d8cac"
        assert current["endpoint_source"] == before["endpoint_source"]
        assert SparkrouteEngine(state_dir=tmp_path, proxy_config=config).build_desired_set()["deployments"][0] == current
        engine.prepare_config([], {}, write=True)
        assert engine.build_desired_set()["deployments"][0]["title"] == "sparkrun:unassigned:deepseek"


def test_discovery_projects_shared_native_apis_without_conflating_model_names(engine):
    endpoints = [
        SimpleNamespace(
            healthy=True, actual_models=["shared", "modern"], native_protocols=["openai", "anthropic"], capabilities=["responses"]
        ),
        SimpleNamespace(healthy=True, actual_models=["shared", "legacy"], native_protocols=["openai"], capabilities=[]),
    ]
    engine._persist_discovered_apis(endpoints)
    cached = engine._read_discovered_apis()
    assert cached["shared"] == {"native_protocols": ["openai"], "capabilities": []}
    assert cached["modern"] == {"native_protocols": ["openai", "anthropic"], "capabilities": ["responses"]}
    with mock.patch.object(engine_mod, "resolve_bindings", return_value=[]):
        document = engine.build_desired_set(discovered_models=["modern", "shared"])
    by_model = {deployment["model"]: deployment for deployment in document["deployments"]}
    assert by_model["modern"]["capabilities"] == ["responses"]
    assert by_model["shared"]["native_protocols"] == ["openai"]


def test_discovery_persists_saved_model_size_and_observed_context(engine):
    from sparkrun.core.recipe import Recipe

    recipe = Recipe(
        {
            "model": "fixture",
            "runtime": "vllm",
            "container": "fixture:latest",
            "metadata": {"model_params": 8_000_000_000},
            "defaults": {"max_model_len": 65536},
        }
    )
    job = SimpleNamespace(cluster_id="fixture-job", metadata={"recipe_state": recipe.__getstate__()})
    endpoint = SimpleNamespace(
        cluster_id="fixture-job", healthy=True, actual_models=["coding"], native_protocols=["openai"], max_model_len=8192
    )
    with mock.patch("sparkrun.api.list_jobs", return_value=[job]):
        engine._persist_discovered_apis([endpoint])
    with mock.patch.object(engine_mod, "resolve_bindings", return_value=[]):
        deployment = engine.build_desired_set(discovered_models=["coding"])["deployments"][0]
    assert deployment["model_metadata"] == {"size_b": 8, "context": 8192, "input_price": 0, "output_price": 0, "tags": ["local", "vllm"]}


def test_bound_metadata_does_not_borrow_from_a_different_recipe(engine):
    from sparkrun.core.recipe import Recipe
    from sparkrun.plugins.sparkroute.projection import ProjectedBinding

    jobs = []
    endpoints = []
    for revision, size, context in [("matching", 8, 8192), ("other", 70, 4096)]:
        recipe = Recipe({"model": "fixture", "runtime": "vllm", "container": "fixture:latest", "metadata": {"model_params": size * 1e9}})
        jobs.append(SimpleNamespace(cluster_id=revision, metadata={"recipe_state": recipe.__getstate__(), "recipe_fingerprint": revision}))
        endpoints.append(SimpleNamespace(cluster_id=revision, healthy=True, actual_models=["coding"], max_model_len=context))
    with mock.patch("sparkrun.api.list_jobs", return_value=jobs):
        engine._persist_discovered_apis(endpoints)
    bound = ProjectedBinding(recipe="fixture.yaml", recipe_revision="matching", model="coding", virtual_model="coding")
    with mock.patch.object(engine_mod, "resolve_bindings", return_value=[bound]):
        fields = engine.build_desired_set()["deployments"][0]["model_metadata"]
    assert fields["size_b"] == 8
    assert fields["context"] == 8192
