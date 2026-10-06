"""Optional provider dispatch must preserve core image policy and host identity."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace

import pytest

from sparkrun.core import image_distribution as api
from sparkrun.core.registration import registry_transaction
from sparkrun.transports.session import SshHostSession


@pytest.fixture(autouse=True)
def isolated_provider(monkeypatch):
    monkeypatch.setattr(api, "_PROVIDERS", {})
    token = api._CONFIG.set(None)
    yield
    api._CONFIG.reset(token)


def invoke(**overrides):
    arguments = dict(
        image="test:tag",
        source_host=None,
        targets=["node-a", "node-b"],
        transfer_hosts=["192.0.2.1", "192.0.2.2"],
        ssh_user="worker",
        ssh_key=None,
        ssh_options=None,
        timeout=20,
        dry_run=True,
        offline=True,
    )
    arguments.update(overrides)
    return api.try_image_copy(**arguments)


def test_builtin_path_never_opens_provider_session(monkeypatch):
    monkeypatch.setattr(SshHostSession, "__init__", lambda *a, **k: pytest.fail("session created"))
    assert invoke() is None
    api.register_image_distribution_provider("test", SimpleNamespace(copy=lambda request: pytest.fail("provider invoked")))
    api._CONFIG.set({"container_distribution_provider": "builtin"})
    assert invoke() is None


def test_logical_targets_and_transfer_addresses_remain_aligned():
    requests = []

    def copy(request):
        requests.append(request)
        return api.ImageCopyResult({"node-a": "already_present", "node-b": "failed"})

    api.register_image_distribution_provider("test", SimpleNamespace(copy=copy))
    assert invoke() == ["node-b"]
    assert requests[0].transfer_hosts == ("192.0.2.1", "192.0.2.2")
    assert requests[0].session is None and requests[0].offline
    with pytest.raises(ValueError, match="aligned"):
        invoke(transfer_hosts=["one"])


@pytest.mark.parametrize("outcomes", [{"node-a": "complete"}, {"node-a": "complete", "node-b": "invented"}])
def test_provider_cannot_omit_or_invent_results(outcomes):
    api.register_image_distribution_provider("test", SimpleNamespace(copy=lambda request: api.ImageCopyResult(outcomes)))
    with pytest.raises(ValueError):
        invoke()


def test_fallback_is_explicit_and_only_before_transfer():
    def unsupported(request):
        raise api.ImageDistributionUnsupported("source unsupported")

    api.register_image_distribution_provider("test", SimpleNamespace(copy=unsupported))
    assert invoke() == ["node-a", "node-b"]
    api._CONFIG.set({"container_distribution_fallback": True})
    assert invoke() is None
    api._CONFIG.set({"container_distribution_provider": "test", "container_distribution_fallback": True})
    assert invoke() == ["node-a", "node-b"]


def test_operation_config_survives_worker_context_and_is_restored():
    config = {"container_distribution_provider": "test"}
    observed = []
    api.register_image_distribution_provider(
        "test",
        SimpleNamespace(
            copy=lambda request: observed.append(request.config) or api.ImageCopyResult(dict.fromkeys(request.targets, "complete"))
        ),
    )

    @api.image_distribution_operation
    def operation(config):
        with ThreadPoolExecutor() as pool:
            return pool.submit(copy_context().run, invoke).result()

    assert operation(config) == []
    assert observed == [config]
    assert api._CONFIG.get() is None


def test_registry_rolls_back_partial_plugin_registration():
    with pytest.raises(RuntimeError):
        with registry_transaction():
            api.register_image_distribution_provider("test", SimpleNamespace(copy=lambda request: None))
            raise RuntimeError("failed registration")
    assert not api._PROVIDERS


def test_managed_local_process_closes_and_rejects_reuse():
    import sys

    session = SshHostSession()
    process = session.open_process("localhost", [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(60)"])
    assert process.stdout.readline() == b"ready\n"
    session.close()
    assert process.poll() is not None
    process.close()


def test_forward_owns_its_connection_even_with_user_multiplexing(monkeypatch):
    session = SshHostSession(ssh_options=["-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/user-master"])
    commands = []
    monkeypatch.setattr(session, "_start_stream", lambda command: commands.append(command))
    session.open_forward("remote", listen_port=0, target_host="127.0.0.1", target_port=9443, reverse=True)
    assert commands[0][:5] == ["ssh", "-o", "ControlMaster=no", "-o", "ControlPath=none"]
    assert "127.0.0.1:0:127.0.0.1:9443" in commands[0]
    assert "ExitOnForwardFailure=yes" in commands[0]
    session.close()


def invoke_pull(**overrides):
    arguments = dict(
        image="registry.test/image:tag",
        source_host=None,
        targets=["a", "b"],
        transfer_hosts=["192.0.2.1", "192.0.2.2"],
        ssh_user=None,
        ssh_key=None,
        ssh_options=None,
        timeout=20,
        dry_run=True,
        offline=False,
        force_pull=False,
    )
    arguments.update(overrides)
    return api.try_image_pull(**arguments)


def test_pre_pull_is_optional_and_offline_never_invokes_it():
    calls = []
    provider = SimpleNamespace(copy=lambda r: None)
    api.register_image_distribution_provider("test", provider)
    assert invoke_pull() is None
    provider.pull = lambda r: calls.append(r) or api.ImageCopyResult(dict.fromkeys(r.targets, "complete"))
    assert invoke_pull(offline=True) is None and not calls
    assert invoke_pull(force_pull=True) == []
    assert calls[0].force_pull and calls[0].source_host is None
    assert calls[0].transfer_hosts == ("192.0.2.1", "192.0.2.2")


def test_pre_pull_decline_and_partial_failure_are_distinct():
    provider = SimpleNamespace(copy=lambda r: None, pull=lambda r: None)
    api.register_image_distribution_provider("test", provider)
    assert invoke_pull() is None
    provider.pull = lambda r: api.ImageCopyResult({"a": "complete", "b": "failed"})
    with pytest.raises(api.ImageDistributionFailed):
        invoke_pull()
    provider.pull = lambda r: api.ImageCopyResult({"a": "complete"})
    with pytest.raises(ValueError, match="every requested target"):
        invoke_pull()


def test_pre_pull_unsupported_fallback_requires_policy():
    def unsupported(request):
        raise api.ImageDistributionUnsupported("unsupported source")

    api.register_image_distribution_provider("test", SimpleNamespace(copy=lambda r: None, pull=unsupported))
    with pytest.raises(api.ImageDistributionFailed):
        invoke_pull()
    api._CONFIG.set({"container_distribution_fallback": True})
    assert invoke_pull() is None


@pytest.mark.parametrize("head", [False, True])
def test_pre_pull_runs_before_builtin_source_pull(monkeypatch, head):
    from sparkrun.containers import distribute
    from sparkrun.orchestration import distribution

    calls = []
    api.register_image_distribution_provider(
        "test",
        SimpleNamespace(
            copy=lambda r: pytest.fail("late copy invoked"),
            pull=lambda r: calls.append(r) or api.ImageCopyResult(dict.fromkeys(r.targets, "complete")),
        ),
    )
    monkeypatch.setattr(distribute, "ensure_image", lambda *a, **k: pytest.fail("source pulled first"))
    monkeypatch.setattr(distribution, "_distribute_from_head", lambda *a, **k: pytest.fail("head pulled first"))
    monkeypatch.setattr(distribute, "_check_remote_image_identities", lambda *a, **k: pytest.fail("head precheck ran first"))
    if head:
        assert (
            distribute.distribute_image_from_head("registry.test/image:tag", ["a", "b"], worker_transfer_hosts=["192.0.2.2"], dry_run=True)
            == []
        )
        assert calls[0].source_host == "a" and calls[0].targets == ("a", "b")
    else:
        assert distribute.distribute_image_from_local("registry.test/image:tag", ["a", "b"], dry_run=True) == []
        assert calls[0].source_host is None and calls[0].targets == ("a", "b")
