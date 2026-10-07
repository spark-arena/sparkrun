# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/coldsnap/LICENSE_EXCEPTION.

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from sparkrun.plugins.coldsnap.host_provider import CAPABILITIES, PROTOCOL_FORMAT, ColdSnapHostProvider
from sparkrun.plugins.coldsnap.service import ColdSnapService
from sparkrun.transports.session import HostCommandResult, SshHostSession


def _request(operation="restore"):
    return {
        "format": 1,
        "id": "operation-one",
        "operation": operation,
        "launch": {
            "engine": "vllm",
            "units": [
                {"id": "unit-0", "host": "node-a"},
                {"id": "unit-1", "host": "node-b"},
            ],
            "execution": {"workers": [{"id": "worker-0"}, {"id": "worker-1"}]},
        },
        "snapshot_driver": {"id": "n610", "abi": 1},
    }


def _context():
    return SimpleNamespace(
        config=SimpleNamespace(ssh_user=None, ssh_key=None, ssh_options=None, get=lambda _key, default=None: default),
    )


class _Session:
    provider_name = "test-session"

    def __init__(self):
        self.calls = []
        self.closed = False

    def execute(self, host, arguments, *, input_data=None, combined=False, timeout=None):
        self.calls.append(("exec", host, arguments, input_data, combined, timeout))
        return HostCommandResult(host, 7 if arguments[0] == "fail" else 0, b"stdout", b"stderr")

    def upload(self, host, sources, destination, *, recursive=False):
        self.calls.append(("upload", host, sources, destination, recursive))

    def docker_registry(self, host, operation, reference):
        self.calls.append(("docker", host, operation, reference))

    def close(self):
        self.closed = True


def _rpc(provider, **values):
    request = {
        "format": PROTOCOL_FORMAT,
        "id": values.pop("id", "rpc-one"),
        "token": values.pop("_token", provider.token),
        "session": provider.request_id,
        **values,
    }
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(provider.socket))
    try:
        client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
        client.shutdown(socket.SHUT_WR)
        payload = b""
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            payload += chunk
    finally:
        client.close()
    return json.loads(payload)


def test_provider_socket_fits_even_with_a_long_macos_tmpdir(tmp_path, monkeypatch):
    import tempfile

    long_directory = tmp_path / ("macos-private-temp-" * 8)
    long_directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(long_directory))
    session = _Session()
    with ColdSnapHostProvider(_request(), sctx=_context(), session_factory=lambda *a, **kw: session) as provider:
        assert len(os.fsencode(provider.socket)) < 104
        assert provider.socket.parent.stat().st_mode & 0o777 == 0o700
        assert provider.socket.stat().st_mode & 0o777 == 0o600
        assert _rpc(provider, operation="exec", host="node-a", arguments=["true"])["ok"]
        directory = provider.socket.parent
    assert session.closed and not directory.exists()


def test_provider_round_trips_exact_argv_input_and_exit_status():
    session = _Session()
    with ColdSnapHostProvider(
        _request(),
        sctx=_context(),
        session_factory=lambda _cluster, **_kwargs: session,
    ) as provider:
        capabilities = _rpc(provider, operation="capabilities")
        response = _rpc(
            provider,
            operation="exec",
            host="node-a",
            arguments=["fail", "space separated", "*.json", ""],
            input=base64.b64encode(b"input\x00bytes").decode(),
            combined=True,
        )

    assert capabilities["provider"] == "test-session"
    assert capabilities["capabilities"] == list(CAPABILITIES)
    assert response == {
        "format": PROTOCOL_FORMAT,
        "id": "rpc-one",
        "ok": True,
        "exit_code": 7,
        "output": base64.b64encode(b"stdout").decode(),
        "error_output": base64.b64encode(b"stderr").decode(),
    }
    assert session.calls == [("exec", "node-a", ["fail", "space separated", "*.json", ""], b"input\x00bytes", True, None)]
    assert session.closed


def test_provider_rejects_hosts_and_tokens_outside_operation_scope():
    session = _Session()
    with ColdSnapHostProvider(
        _request(),
        sctx=_context(),
        session_factory=lambda _cluster, **_kwargs: session,
    ) as provider:
        outside = _rpc(provider, operation="exec", host="node-c", arguments=["true"])
        unauthorized = _rpc(
            provider,
            operation="exec",
            host="node-a",
            arguments=["true"],
            _token="different",
        )

    assert not outside["ok"] and "outside this request" in outside["error"]
    assert not unauthorized["ok"] and "unauthorized" in unauthorized["error"]
    assert session.calls == []


def test_provider_routes_upload_and_controller_authenticated_registry_calls():
    session = _Session()
    with ColdSnapHostProvider(
        _request(),
        sctx=_context(),
        session_factory=lambda _cluster, **_kwargs: session,
    ) as provider:
        upload = _rpc(
            provider,
            operation="upload",
            host="node-b",
            sources=["/tmp/source one", "/tmp/source-two"],
            destination="/var/lib/coldsnap/target",
            recursive=True,
        )
        pull = _rpc(provider, operation="oci-pull", host="node-b", image="registry/model@sha256:abc")
        push = _rpc(provider, operation="oci-push", host="node-a", image="registry/model:tag")

    assert upload["ok"] and pull["ok"] and push["ok"]
    assert session.calls == [
        ("upload", "node-b", ["/tmp/source one", "/tmp/source-two"], "/var/lib/coldsnap/target", True),
        ("docker", "node-b", "pull", "registry/model@sha256:abc"),
        ("docker", "node-a", "push", "registry/model:tag"),
    ]


def test_provider_keeps_hugging_face_token_on_manager_side(monkeypatch):
    session = _Session()
    monkeypatch.setattr("sparkrun.plugins.coldsnap.host_provider.resolve_hf_token", lambda: "hf_manager_secret")
    with ColdSnapHostProvider(
        _request("publish-native"),
        sctx=_context(),
        session_factory=lambda _cluster, **_kwargs: session,
    ) as provider:
        published = _rpc(
            provider,
            operation="huggingface-publish",
            host="node-a",
            image="capsule@sha256:abc",
            repository="org/model-native",
            revision="staging",
            source="/packs/worker-0.pack",
            destination="objects/worker-0.pack",
        )
        resolved = _rpc(
            provider,
            operation="huggingface-resolve",
            host="node-a",
            image="capsule@sha256:abc",
            repository="org/model-native",
            revision="staging",
        )

    assert published["ok"]
    assert resolved["ok"] and resolved["value"] == "stdout"
    assert len(session.calls) == 2
    for call in session.calls:
        assert call[0:2] == ("exec", "node-a")
        assert call[2][:2] == ["docker", "run"]
        assert all(flag in call[2] for flag in ("--rm", "-i", "--entrypoint"))
        assert call[3] == b"hf_manager_secret"
    assert "/packs/worker-0.pack:/coldsnap-upload/native.pack:ro" in session.calls[0][2]


def test_service_scopes_provider_to_subprocess_and_scrubs_hf_credentials(monkeypatch):
    events = []

    class Provider:
        def __init__(self):
            self.environment = {
                "COLDSNAP_HOST_PROVIDER": "external",
                "COLDSNAP_HOST_PROVIDER_SOCKET": "/tmp/provider.sock",
                "COLDSNAP_HOST_PROVIDER_TOKEN": "secret",
            }

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args):
            events.append("exit")

    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "must-not-leak-either")

    def invoke(_arguments, **kwargs):
        events.append("invoke")
        assert kwargs["env"]["COLDSNAP_HOST_PROVIDER"] == "external"
        assert "HF_TOKEN" not in kwargs["env"]
        assert "HUGGING_FACE_HUB_TOKEN" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    service = ColdSnapService(
        "/opt/coldsnap/bin/coldsnap",
        run_command=invoke,
        host_provider_factory=lambda *_args, **_kwargs: Provider(),
        target_tool_resolver=lambda **_kwargs: SimpleNamespace(environment={}),
    )
    service._invoke(
        _request("publish-native"),
        prepare_only=False,
        capture_output=False,
        sctx=_context(),
        cluster=SimpleNamespace(sparkrun_cache_dir="/cache/sparkrun", plugins={}, user=None),
    )

    assert events == ["enter", "invoke", "exit"]


def test_service_closes_provider_when_controller_is_interrupted():
    events = []

    class Provider:
        def __init__(self):
            self.environment = {
                "COLDSNAP_HOST_PROVIDER": "external",
                "COLDSNAP_HOST_PROVIDER_SOCKET": "/tmp/provider.sock",
                "COLDSNAP_HOST_PROVIDER_TOKEN": "secret",
            }

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args):
            events.append("exit")

    def interrupt(_arguments, **_kwargs):
        events.append("invoke")
        raise KeyboardInterrupt

    service = ColdSnapService(
        "/opt/coldsnap/bin/coldsnap",
        run_command=interrupt,
        host_provider_factory=lambda *_args, **_kwargs: Provider(),
        target_tool_resolver=lambda **_kwargs: SimpleNamespace(environment={}),
    )
    with pytest.raises(KeyboardInterrupt):
        service._invoke(
            _request(),
            prepare_only=False,
            capture_output=False,
            sctx=_context(),
            cluster=SimpleNamespace(sparkrun_cache_dir="/cache/sparkrun", plugins={}, user=None),
        )

    assert events == ["enter", "invoke", "exit"]


def test_ssh_host_session_preserves_local_argv_and_binary_input():
    session = SshHostSession()
    try:
        result = session.execute(
            "localhost",
            [
                os.environ.get("PYTHON", "python3"),
                "-c",
                "import sys; print(repr(sys.argv[1:])); sys.stdout.buffer.write(sys.stdin.buffer.read())",
                "space separated",
                "*.json",
                "",
            ],
            input_data=b"binary\x00input\n",
            combined=True,
        )
    finally:
        session.close()

    assert result.returncode == 0
    assert b"binary\x00input\n" in result.stdout
    assert b"['space separated', '*.json', '']" in result.stdout


def test_provider_rejects_malformed_input_base64():
    session = _Session()
    with ColdSnapHostProvider(
        _request(),
        sctx=_context(),
        session_factory=lambda _cluster, **_kwargs: session,
    ) as provider:
        response = _rpc(
            provider,
            operation="exec",
            host="node-a",
            arguments=["true"],
            input="%%%",
        )
    assert not response["ok"] and "input is invalid" in response["error"]
    assert session.calls == []


def test_provider_advertises_and_enforces_selected_capabilities():
    session = _Session()
    with ColdSnapHostProvider(
        _request(),
        sctx=_context(),
        session_factory=lambda _cluster, **_kwargs: session,
        capabilities=("exec",),
    ) as provider:
        capabilities = _rpc(provider, operation="capabilities")
        rejected = _rpc(
            provider,
            operation="oci-pull",
            host="node-a",
            image="registry/model@sha256:abc",
        )

    assert capabilities["capabilities"] == ["exec"]
    assert not rejected["ok"] and "unsupported" in rejected["error"]
    assert session.calls == []


def test_provider_rejects_unknown_fields_and_trailing_messages():
    session = _Session()
    with ColdSnapHostProvider(
        _request(),
        sctx=_context(),
        session_factory=lambda _cluster, **_kwargs: session,
    ) as provider:
        unknown = _rpc(
            provider,
            operation="exec",
            host="node-a",
            arguments=["true"],
            injected="value",
        )
        request = {
            "format": PROTOCOL_FORMAT,
            "id": "rpc-trailing",
            "token": provider.token,
            "session": provider.request_id,
            "operation": "exec",
            "host": "node-a",
            "arguments": ["true"],
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5)
            client.connect(str(provider.socket))
            client.sendall(json.dumps(request).encode() + b"\n{}\n")
            # Trailing data is enough to reject the request without EOF.
            # A half-close here races with that rejection on macOS (ENOTCONN).
            payload = b""
            while chunk := client.recv(65536):
                payload += chunk
        trailing = json.loads(payload)

    assert not unknown["ok"] and "unknown fields" in unknown["error"]
    assert not trailing["ok"] and "trailing data" in trailing["error"]
    assert session.calls == []


def test_ssh_host_session_close_terminates_inflight_child():
    session = SshHostSession()
    result = []

    def invoke():
        result.append(
            session.execute(
                "localhost",
                [os.environ.get("PYTHON", "python3"), "-c", "import time; time.sleep(60)"],
            )
        )

    thread = Thread(target=invoke)
    thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with session._lock:
            if session._processes:
                break
        time.sleep(0.01)
    else:
        pytest.fail("host-session child did not start")
    session.close()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert result and result[0].returncode != 0


def test_go_client_and_python_provider_cross_language_contract():
    source_value = os.environ.get("COLDSNAP_SOURCE_ROOT", "")
    source = Path(source_value).resolve() if source_value else None
    go = os.environ.get("COLDSNAP_GO") or shutil.which("go")
    if source is None or not (source / "go.mod").is_file() or not go:
        pytest.skip("set COLDSNAP_SOURCE_ROOT and COLDSNAP_GO for cross-language contract")

    class ContractSession(_Session):
        def execute(self, host, arguments, *, input_data=None, combined=False, timeout=None):
            self.calls.append(("exec", host, arguments, input_data, combined, timeout))
            return HostCommandResult(host, 0, b"contract:" + (input_data or b""), b"")

    session = ContractSession()
    with ColdSnapHostProvider(
        _request(),
        sctx=_context(),
        session_factory=lambda _cluster, **_kwargs: session,
        capabilities=("exec",),
    ) as provider:
        environment = {
            **os.environ,
            "COLDSNAP_TEST_HOST_PROVIDER_SOCKET": str(provider.socket),
            "COLDSNAP_TEST_HOST_PROVIDER_TOKEN": provider.token,
            "COLDSNAP_TEST_HOST_PROVIDER_SESSION": provider.request_id,
            "COLDSNAP_TEST_HOST_PROVIDER_HOST": "node-a",
        }
        completed = subprocess.run(
            [
                go,
                "test",
                "./internal/hostprovider",
                "-run",
                "^TestManagerProviderCrossLanguageContract$",
                "-count=1",
            ],
            cwd=source,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert session.calls == [
        (
            "exec",
            "node-a",
            ["contract-command", "space separated", "*.json", ""],
            b"cross-language\x00payload\n",
            False,
            None,
        )
    ]
