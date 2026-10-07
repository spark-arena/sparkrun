# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

"""Restore image callbacks use the host provider and preserve per-unit identities."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.core import image_distribution as distribution
from sparkrun.plugins.coldsnap.host_provider import ColdSnapHostProvider
from sparkrun.plugins.coldsnap.manager_runtime import DockerManagerRuntime
from sparkrun.plugins.coldsnap.service import ColdSnapService
from sparkrun.transports.session import HostCommandResult, HostSessionError
from test_coldsnap_host_provider import _request, _rpc, _Session

PIN_A = "registry.test/unit-a@sha256:" + "a" * 64
PIN_B = "registry.test/unit-b@sha256:" + "b" * 64
ID_A = "sha256:" + "c" * 64
ID_B = "sha256:" + "d" * 64


class Config(dict):
    ssh_user = "operator"
    ssh_key = "/key"
    ssh_options = ["-o", "BatchMode=yes"]


class Images(_Session):
    def __init__(self):
        super().__init__()
        self.resident = set()

    def execute(self, host, arguments, **kwargs):
        self.calls.append(("exec", host, list(arguments)))
        if arguments[:3] == ["docker", "image", "inspect"]:
            image = arguments[3]
            if (host, image) not in self.resident:
                return HostCommandResult(host, 1, b"", b"Error: No such image: absent")
            return HostCommandResult(host, 0, json.dumps({"id": image, "repo_digests": [], "size": 123}).encode())
        if arguments[:2] == ["docker", "run"]:
            assert arguments[arguments.index("--pull") + 1] == "never"
            assert any((host, value) in self.resident for value in arguments)
            return HostCommandResult(host, 0, b"container-id")
        return HostCommandResult(host, 0, b"")

    def docker_registry(self, host, operation, reference):
        super().docker_registry(host, operation, reference)
        if operation == "pull":
            self.resident.add((host, reference))


class Provider:
    supports_offline_pull = True

    def __init__(self, session):
        self.session = session
        self.pulls = []
        self.lookups = []
        self.bindings = {}

    def copy(self, request):
        pytest.fail("typed registry pulls should use pre-pull API")

    def pull(self, request):
        assert request.session is self.session and not self.session.closed
        self.pulls.append(request)
        reference = ID_A if request.image == PIN_A else ID_B
        for host in request.targets:
            self.bindings[host, request.image] = reference
            self.session.resident.add((host, reference))
        return distribution.ImageCopyResult(
            dict.fromkeys(request.targets, "complete"), runtime_images=dict.fromkeys(request.targets, reference)
        )

    def local_image(self, request):
        assert request.session is self.session and not self.session.closed
        self.lookups.append(request)
        ref = self.bindings.get((request.source_host, request.image))
        return ref if (request.source_host, ref) in self.session.resident else None


@pytest.fixture(autouse=True)
def isolate_distribution(monkeypatch):
    monkeypatch.setattr(distribution, "_PROVIDERS", {})
    monkeypatch.setattr("sparkrun.transports.session.SshHostSession", Mock(side_effect=AssertionError("bypassed prepared transport")))
    config = distribution._CONFIG.set(None)
    bindings = distribution._RUNTIME_IMAGES.set(None)
    yield
    distribution._CONFIG.reset(config)
    distribution._RUNTIME_IMAGES.reset(bindings)


def _runtime(session, provider, *, config=None, offline=False):
    distribution.register_image_distribution_provider("relay-test", provider)
    return DockerManagerRuntime(session, config=config if config is not None else Config(), offline=offline)


def test_threaded_restore_keeps_each_capsule_and_host_session():
    session = Images()
    relay = Provider(session)
    distribution.register_image_distribution_provider("relay-test", relay)
    request = _request()
    request["launch"]["units"].append({"id": "unit-2", "host": "node-a"})
    original = deepcopy(request)
    selected = Config(container_distribution_provider="relay-test")
    root_config = Config(container_distribution_provider="builtin")
    root_config.for_cluster = lambda cluster: selected
    sctx = SimpleNamespace(config=root_config)
    session_factory = Mock(return_value=session)
    distribution._CONFIG.set(Config(container_distribution_provider="builtin"))
    assignments = [("node-a", PIN_A, ID_A), ("node-b", PIN_A, ID_A), ("node-a", PIN_B, ID_B)]
    with ColdSnapHostProvider(request, sctx=sctx, cluster=SimpleNamespace(), session_factory=session_factory) as manager:

        def prepare(assignment):
            host, image, _reference = assignment
            absent = _rpc(manager, operation="runtime", host=host, runtime={"action": "image-inspect", "image": image})
            assert not absent["ok"]
            pulled = _rpc(manager, operation="runtime", host=host, runtime={"action": "image-pull", "image": image})
            assert pulled["ok"], pulled
            inspected = _rpc(manager, operation="runtime", host=host, runtime={"action": "image-inspect", "image": image})
            assert inspected["runtime"]["image"]["id"] == _reference

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(prepare, assignments))
        assert not session.closed
    assert session.closed
    # Activation gets a new callback scope, so it must recover verified pins
    # from the provider, rather than depend on contextvars from preparation.
    session.closed = False
    with ColdSnapHostProvider(request, sctx=sctx, cluster=SimpleNamespace(), session_factory=session_factory) as manager:
        for host, image, _reference in assignments:
            result = _rpc(
                manager,
                operation="runtime",
                host=host,
                runtime={"action": "workload-run", "workload": {"image": image, "pull_policy": "never", "command": ["serve"]}},
            )
            assert result["ok"], result
    assert request == original
    assert {(r.image, r.targets) for r in relay.pulls} == {(image, (host,)) for host, image, _ref in assignments}
    assert all(r.config is selected and r.ssh_user == "operator" and r.ssh_key == "/key" for r in relay.pulls)
    assert all(r.transfer_hosts == r.targets for r in relay.pulls)
    runs = [call for call in session.calls if call[0] == "exec" and call[2][:2] == ["docker", "run"]]
    assert len(runs) == 3
    assert all(any(ref in call[2] for ref in (ID_A, ID_B)) for call in runs)
    assert not any(call[0] == "docker" for call in session.calls)


@pytest.mark.parametrize("selected", ["auto", "builtin"])
def test_disabled_provider_uses_existing_credentialed_docker_pull(selected):
    session = Images()
    if selected == "builtin":
        distribution.register_image_distribution_provider("relay-test", Provider(session))
    runtime = DockerManagerRuntime(session, config=Config(container_distribution_provider=selected))
    runtime.invoke("node-a", {"action": "image-pull", "image": PIN_A})
    assert session.calls == [("docker", "node-a", "pull", PIN_A)]
    assert not session.closed


@pytest.mark.parametrize("selected,fallback,allowed", [("auto", True, True), ("auto", False, False), ("relay-test", True, False)])
def test_only_configured_pretransfer_fallback_uses_docker(selected, fallback, allowed):
    session = Images()
    provider = Provider(session)
    provider.pull = Mock(side_effect=distribution.ImageDistributionUnsupported("unavailable before transfer"))
    runtime = _runtime(session, provider, config=Config(container_distribution_provider=selected, container_distribution_fallback=fallback))
    if allowed:
        runtime.invoke("node-a", {"action": "image-pull", "image": PIN_A})
        assert session.calls == [("docker", "node-a", "pull", PIN_A)]
    else:
        with pytest.raises(distribution.ImageDistributionFailed, match="unsupported"):
            runtime.invoke("node-a", {"action": "image-pull", "image": PIN_A})
        assert session.calls == []
    assert not session.closed


@pytest.mark.parametrize("mode", ["partial", "exception", "bad_id"])
def test_failed_transfer_never_falls_back(mode):
    session = Images()
    provider = Provider(session)
    if mode == "exception":
        provider.pull = Mock(side_effect=RuntimeError("interrupted transfer"))
    elif mode == "partial":
        provider.pull = lambda request: distribution.ImageCopyResult({"node-a": "failed"})
    else:
        provider.pull = lambda request: distribution.ImageCopyResult({"node-a": "complete"}, runtime_images={"node-a": "mutable:tag"})
    runtime = _runtime(session, provider, config=Config(container_distribution_fallback=True))
    with pytest.raises((RuntimeError, ValueError)):
        runtime.invoke("node-a", {"action": "image-pull", "image": PIN_A})
    assert session.calls == [] and not session.closed


@pytest.mark.parametrize("provider_kind", ["capable", "incapable", "absent"])
def test_offline_never_falls_back_to_registry(provider_kind):
    session = Images()
    provider = Provider(session)
    if provider_kind != "absent":
        provider.supports_offline_pull = provider_kind == "capable"
        distribution.register_image_distribution_provider("relay-test", provider)
    runtime = DockerManagerRuntime(session, config=Config(), offline=True)
    if provider_kind == "capable":
        runtime.invoke("node-a", {"action": "image-pull", "image": PIN_A})
        assert provider.pulls[0].offline
    else:
        with pytest.raises(HostSessionError, match="offline"):
            runtime.invoke("node-a", {"action": "image-pull", "image": PIN_A})
    assert not any(call[0] == "docker" for call in session.calls)


def test_stale_binding_is_rechecked_before_inspection():
    session = Images()
    provider = Provider(session)
    runtime = _runtime(session, provider)
    runtime.invoke("node-a", {"action": "image-pull", "image": PIN_A})
    assert runtime.invoke("node-a", {"action": "image-inspect", "image": PIN_A})["image"]["id"] == ID_A
    session.resident.clear()
    with pytest.raises(RuntimeError, match="No such image"):
        runtime.invoke("node-a", {"action": "image-inspect", "image": PIN_A})
    assert session.calls[-1][2][3] == PIN_A
    runtime.invoke("node-a", {"action": "image-pull", "image": PIN_A})
    assert runtime.invoke("node-a", {"action": "image-inspect", "image": PIN_A})["image"]["id"] == ID_A


@pytest.mark.parametrize("policy", ["missing", "always", "never"])
def test_workload_pull_policy_uses_provider_then_launches_verified_image(policy):
    session = Images()
    provider = Provider(session)
    runtime = _runtime(session, provider)
    if policy == "never":
        runtime.invoke("node-a", {"action": "image-pull", "image": PIN_A})
        provider.pulls.clear()
    workload = {"image": PIN_A, "pull_policy": policy, "command": ["serve"]}
    runtime.invoke("node-a", {"action": "workload-run", "workload": workload})
    assert len(provider.pulls) == (0 if policy == "never" else 1)
    if provider.pulls:
        assert provider.pulls[0].force_pull == (policy == "always")
    args = session.calls[-1][2]
    assert args[-2:] == [ID_A, "serve"]
    assert args[args.index("--pull") + 1] == "never"
    assert workload["image"] == PIN_A


def test_local_only_image_is_never_pulled():
    session = Images()
    provider = Provider(session)
    runtime = _runtime(session, provider)
    session.resident.add(("node-a", ID_A))
    runtime.invoke("node-a", {"action": "image-inspect", "image": ID_A})
    runtime.invoke("node-a", {"action": "workload-run", "workload": {"image": ID_A, "pull_policy": "never"}})
    with pytest.raises(HostSessionError, match="local-only"):
        runtime.invoke("node-a", {"action": "image-pull", "image": ID_A})
    assert not provider.pulls and not provider.lookups


@pytest.mark.parametrize("operation", ["image-pull", "oci-pull"])
def test_registry_capability_remains_required(operation):
    session = Images()
    provider = Provider(session)
    distribution.register_image_distribution_provider("relay-test", provider)
    with ColdSnapHostProvider(
        _request(), sctx=SimpleNamespace(config=Config()), session_factory=lambda *_a, **_kw: session, capabilities=("runtime-v1",)
    ) as manager:
        values = (
            {"operation": "runtime", "runtime": {"action": operation, "image": PIN_A}}
            if operation == "image-pull"
            else {"operation": operation, "image": PIN_A}
        )
        response = _rpc(manager, host="node-a", **values)
        assert not response["ok"] and "oci-pull" in response["error"]
        assert not provider.pulls


@pytest.mark.parametrize("override,expected", [(None, True), (False, False), (True, True)])
def test_manager_honors_cli_offline_override(override, expected):
    session = Images()
    with ColdSnapHostProvider(
        _request(),
        sctx=SimpleNamespace(config=Config()),
        cluster=SimpleNamespace(offline=True),
        offline=override,
        session_factory=lambda *_a, **_kw: session,
    ) as manager:
        assert manager.runtime.offline is expected


@pytest.mark.parametrize("offline", [True, False])
def test_service_passes_image_offline_policy_to_callback_provider(offline):
    from contextlib import contextmanager
    from test_coldsnap_host_provider import _context

    calls = []

    @contextmanager
    def factory(request, **kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(environment={})

    service = ColdSnapService(
        "/opt/coldsnap/bin/coldsnap",
        host_provider_factory=factory,
        run_command=lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    service._invoke(
        _request("status"),
        prepare_only=False,
        capture_output=False,
        sctx=_context(),
        cluster=SimpleNamespace(sparkrun_cache_dir="/cache/sparkrun", plugins={}, user=None),
        offline=offline,
    )
    assert calls[0]["offline"] is offline


@pytest.mark.parametrize("cached", [False, True])
def test_real_oci_relay_pin_receipt_bridges_pull_inspect_and_run(monkeypatch, cached):
    import hashlib

    relay_module = pytest.importorskip("sparkrun.plugins.oci_relay.provider")
    receipts = {}

    class RelayImages(Images):
        def execute(self, host, arguments, **kwargs):
            if arguments[:1] == ["head"]:
                data = receipts.get((host, arguments[-1]))
                return HostCommandResult(host, 0 if data else 1, json.dumps(data).encode() if data else b"")
            if any(value.startswith("--format={{.Id}}|") for value in arguments):
                image = arguments[-1]
                found = (host, image) in self.resident
                return HostCommandResult(host, 0 if found else 1, (image + "|linux|arm64").encode() if found else b"")
            return super().execute(host, arguments, **kwargs)

    session = RelayImages()

    class Runner:
        def __init__(self, actual_session, settings):
            assert actual_session is session
            self.settings = settings

        def connection(self, host):
            return session

        def cache_directory(self, host):
            return "/relay"

        def architecture(self, host):
            return "arm64"

        def close(self):
            pass

    def imported(host):
        session.resident.add((host, ID_A))
        path = "/relay/pins/" + hashlib.sha256(PIN_A.encode()).hexdigest() + ".json"
        receipts[host, path] = {
            "version": 1,
            "image": PIN_A,
            "registry_digest": "sha256:" + "a" * 64,
            "config_digest": ID_A,
            "runtime_image": ID_A,
        }

    relay = relay_module.RelayProvider()
    transfers = []

    def transfer(request, *, registry=False):
        transfers.append(request)
        assert registry and request.targets == ("node-a",) and request.session is session
        imported("node-a")
        return distribution.ImageCopyResult({"node-a": "complete"}, runtime_images={"node-a": ID_A})

    # The network/import boundary is simulated. Provider selection, pinned
    # pre-pull handling, persistent receipt validation and manager image handling are real.
    monkeypatch.setattr(relay_module, "Runner", Runner)
    monkeypatch.setattr(relay, "copy", transfer)
    distribution.register_image_distribution_provider("oci-relay", relay)
    config = Config(container_distribution_provider="oci-relay")
    config.plugin_settings = lambda name: {"source_mode": "registry"} if name == "oci-relay" else {}
    if cached:
        imported("node-a")
    runtime = DockerManagerRuntime(session, config=config)
    runtime.invoke("node-a", {"action": "image-pull", "image": PIN_A})
    assert len(transfers) == (0 if cached else 1)
    info = runtime.invoke("node-a", {"action": "image-inspect", "image": PIN_A})["image"]
    assert info["id"] == ID_A and info["repo_digests"] == []
    runtime.invoke("node-a", {"action": "workload-run", "workload": {"image": PIN_A, "pull_policy": "never"}})
    assert session.calls[-1][2][-1] == ID_A
    assert not any(call[0] == "docker" for call in session.calls)
    assert not session.closed


def test_push_retains_original_reference_and_image_tag_resolves_only_source():
    session = Images()
    relay = Provider(session)
    runtime = _runtime(session, relay)
    runtime.invoke("node-a", {"action": "image-pull", "image": PIN_A})
    runtime.invoke("node-a", {"action": "image-tag", "source": PIN_A, "target": "registry.test/published:new"})
    assert session.calls[-1][2] == ["docker", "tag", ID_A, "registry.test/published:new"]
    runtime.invoke("node-a", {"action": "image-push", "image": "registry.test/published:new"})
    assert session.calls[-1] == ("docker", "node-a", "push", "registry.test/published:new")


def test_offline_workload_cannot_pull_implicitly():
    session = Images()
    session.resident.add(("node-a", ID_A))
    runtime = DockerManagerRuntime(session, config=Config(), offline=True)
    runtime.invoke("node-a", {"action": "workload-run", "workload": {"image": ID_A}})
    args = session.calls[-1][2]
    assert args[args.index("--pull") + 1] == "never"
    with pytest.raises(HostSessionError, match="offline"):
        runtime.invoke("node-a", {"action": "workload-run", "workload": {"image": PIN_A, "pull_policy": "always"}})
    assert not any(call[0] == "docker" for call in session.calls)
