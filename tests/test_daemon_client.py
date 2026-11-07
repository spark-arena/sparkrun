"""Explicit CLI endpoints exercise the same authenticated API as applications."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from click.testing import CliRunner

from sparkrun.cli._daemon import daemon
from sparkrun.daemon.state import read_connection
from test_daemon import binding as binding, controlled_plugin as controlled_plugin, real_server, wait_state


@pytest.fixture
def no_local_binding(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Explicit endpoint must never inspect or fall back to a local binding")

    monkeypatch.setattr("sparkrun.cli._daemon._binding", forbidden)
    monkeypatch.delenv("SPARKRUN_DAEMON_TOKEN", raising=False)


def invoke(*args, input=None):
    result = CliRunner().invoke(daemon, list(args), input=input)
    assert result.exit_code == 0, result.output
    return result.output


def test_tcp_cli_operations_jobs_prompts_events_and_cancel(controlled_plugin, no_local_binding, monkeypatch):
    binding = controlled_plugin
    with real_server(binding, extra=("--no-unix", "--tcp", "127.0.0.1:0")) as (_, client):
        connection = read_connection(binding.connection)
        endpoint = ("--url", connection["url"])
        token_file = binding.directory / "client.token"
        token_file.write_text(connection["token"] + "\n")
        credentials = (*endpoint, "--token-file", str(token_file))
        monkeypatch.setenv("SPARKRUN_DAEMON_TOKEN", "invalid-environment-token")
        assert json.loads(invoke(*credentials, "status"))["binding"] == binding.key
        assert json.loads(invoke(*endpoint, "--token", connection["token"], "status"))["running"]
        rejected = CliRunner().invoke(daemon, [*endpoint, "status"])
        assert rejected.exit_code != 0
        monkeypatch.delenv("SPARKRUN_DAEMON_TOKEN")
        assert CliRunner().invoke(daemon, [*endpoint, "status"]).exit_code != 0
        monkeypatch.setenv("SPARKRUN_DAEMON_TOKEN", connection["token"])
        assert json.loads(invoke(*endpoint, "status"))["protocol"] == 1

        job = json.loads(invoke(*credentials, "call", "plugins.daemon-fixture.wait", "--idempotency-key", "remote-cli"))["job_id"]
        wait_state(client, job, {"waiting_input"})
        assert any(row["id"] == job for row in json.loads(invoke(*credentials, "jobs", "list")))
        assert json.loads(invoke(*credentials, "jobs", "show", job))["prompt"]["kind"] == "secret"
        secret = "remote-client-secret"
        assert secret not in invoke(*credentials, "jobs", "answer", job, input=secret + "\n")
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            row = client.request("GET", "/v1/jobs/" + job)
            if row["prompt"] and row["prompt"]["kind"] == "approval":
                break
            time.sleep(0.05)
        else:
            pytest.fail("Approval prompt did not arrive")
        invoke(*credentials, "jobs", "answer", job, input="y\n")
        assert client.wait(job, timeout=15) == {"accepted": True, "echo": "[redacted]"}
        events = [json.loads(line) for line in invoke(*credentials, "jobs", "events", job).splitlines()]
        assert events[-1]["state"] == "succeeded"
        after = events[-2]["id"]
        resumed = [json.loads(line) for line in invoke(*credentials, "jobs", "events", job, "--after", str(after)).splitlines()]
        assert resumed == events[-1:]
        assert secret not in json.dumps(events)
        assert json.loads(invoke(*credentials, "call", "plugins.daemon-fixture.wait", "--idempotency-key", "remote-cli"))["job_id"] == job
        output = invoke(*credentials, "call", "plugins.daemon-fixture.wait", "--wait", input=secret + "\ny\n")
        assert '"accepted": true' in output and secret not in output
        cancelled = json.loads(invoke(*credentials, "call", "plugins.daemon-fixture.wait"))["job_id"]
        wait_state(client, cancelled, {"waiting_input"})
        invoke(*credentials, "jobs", "cancel", cancelled)
        wait_state(client, cancelled, {"cancelled"})

        # The public CLI entry point also bypasses local plugins and feature gates.
        from sparkrun.core.application_profile import child_environment

        local_config = binding.directory / "unused-config.yaml"
        local_config.write_text("features:\n  daemon: false\n")
        result = subprocess.run(
            [sys.executable, "-m", "sparkrun", "daemon", *credentials, "status"],
            env={**os.environ, **child_environment(config_path=local_config)},
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["binding"] == binding.key


@pytest.mark.skipif(sys.platform != "linux", reason="Linux peer-authenticated Unix listener")
def test_explicit_unix_socket_ignores_token_environment(binding, no_local_binding, monkeypatch):
    monkeypatch.setenv("SPARKRUN_DAEMON_TOKEN", "irrelevant-environment-token")
    with real_server(binding):
        options = ("--socket", str(binding.socket))
        assert json.loads(invoke(*options, "status"))["binding"] == binding.key
        assert json.loads(invoke(*options, "jobs", "list")) == []
        invoke(*options, "call", "catalog.clusters")


@pytest.fixture
def incompatible_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"protocol": 999, "binding": "remote"}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize(
    "command",
    [
        ["status"],
        ["call", "cluster.create"],
        ["jobs", "list"],
        ["jobs", "show", "job"],
        ["jobs", "cancel", "job"],
        ["jobs", "events", "job"],
        ["jobs", "answer", "job"],
    ],
)
def test_protocol_mismatch_stops_before_operation(command, incompatible_server, no_local_binding):
    url, requests = incompatible_server
    result = CliRunner().invoke(daemon, ["--url", url, *command])
    assert result.exit_code != 0 and "protocol" in result.output
    assert requests == ["/v1/info"]


@pytest.mark.parametrize(
    "options,command,message",
    [
        (["--url", "http://host", "--socket", "api.sock"], ["status"], "Choose either --url or --socket"),
        (["--url", "http://host", "--config", "config.yaml"], ["status"], "Choose either --config"),
        (["--token", "dummy"], ["status"], "require --url"),
        *[
            (["--url", "http://host"], command, "apply only to call, status, jobs, and help")
            for command in (
                ["start"],
                ["stop"],
                ["restart"],
                ["serve"],
                ["token", "list"],
                ["systemd", "install"],
                ["resources", "list"],
                ["connection-status", "unused"],
            )
        ],
    ],
)
def test_invalid_or_local_only_combinations(options, command, message, no_local_binding):
    result = CliRunner().invoke(daemon, [*options, *command])
    assert result.exit_code == 2 and message in result.output


@pytest.mark.parametrize("content", ["", "first\nsecond", "x" * 4097, "not-ascii-\N{SNOWMAN}"])
def test_invalid_token_file_fails_without_echoing_credentials(tmp_path, content, no_local_binding):
    path = tmp_path / "token"
    path.write_text(content, encoding="utf-8")
    result = CliRunner().invoke(daemon, ["--url", "http://127.0.0.1:1", "--token-file", str(path), "status"])
    assert result.exit_code != 0 and "token" in result.output.lower()
    if content:
        assert content not in result.output


def test_token_and_token_file_are_exclusive(tmp_path, no_local_binding):
    path = tmp_path / "token"
    path.write_text("file-token")
    result = CliRunner().invoke(daemon, ["--url", "http://host", "--token", "inline-token", "--token-file", str(path), "status"])
    assert result.exit_code == 2 and "Choose either --token or --token-file" in result.output
    assert "inline-token" not in result.output and "file-token" not in result.output


@pytest.mark.parametrize(
    "url", ["ftp://host", "http://name:secret@host", "http://host/prefix", "http://host?query=1", "http://host#fragment"]
)
def test_invalid_origin_is_rejected(url, no_local_binding):
    result = CliRunner().invoke(daemon, ["--url", url, "status"])
    assert result.exit_code != 0 and "HTTP(S) daemon origin" in result.output
    assert "secret" not in result.output


def test_explicit_connection_failure_never_falls_back(incompatible_server, no_local_binding):
    url, _ = incompatible_server
    # Port zero is never a daemon listener; no host/config discovery is allowed.
    result = CliRunner().invoke(daemon, ["--url", url.rsplit(":", 1)[0] + ":0", "status"])
    assert result.exit_code != 0 and "running" not in result.output


def test_tls_cli_custom_ca_preserves_certificate_and_hostname_checks(binding, no_local_binding, monkeypatch):
    if not shutil.which("openssl"):
        pytest.skip("openssl unavailable")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    cert, key = binding.directory / "cert.pem", binding.directory / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=daemon.example.invalid",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    with real_server(binding, extra=("--no-unix", "--tcp", "127.0.0.1:0", "--tls-cert", str(cert), "--tls-key", str(key))):
        token = read_connection(binding.connection)["token"]
        url = binding.read_state()["url"]
        credentials = ("--url", url, "--token", token)
        failed = CliRunner().invoke(daemon, [*credentials, "status"])
        assert failed.exit_code != 0 and "CERTIFICATE_VERIFY_FAILED" in failed.output
        assert json.loads(invoke(*credentials, "--ca-cert", str(cert), "status"))["binding"] == binding.key
        wrong_host = CliRunner().invoke(
            daemon, ["--url", url.replace("127.0.0.1", "localhost"), "--token", token, "--ca-cert", str(cert), "status"]
        )
        assert wrong_host.exit_code != 0 and "CERTIFICATE_VERIFY_FAILED" in wrong_host.output
        plaintext = CliRunner().invoke(daemon, ["--url", "http://127.0.0.1:1", "--ca-cert", str(cert), "status"])
        assert plaintext.exit_code != 0 and "requires an HTTPS" in plaintext.output


def test_token_environment_does_not_change_local_discovery(binding, monkeypatch):
    monkeypatch.setenv("SPARKRUN_DAEMON_TOKEN", "ignored")
    with real_server(binding, extra=("--no-unix", "--tcp", "127.0.0.1:0")):
        assert json.loads(invoke("--config", str(binding.config_path), "status"))["binding"] == binding.key


def test_explicit_socket_accepts_manager_token(monkeypatch, no_local_binding):
    seen = []

    def info(client):
        seen.append((client.socket_path, client.token))
        return {"protocol": 1, "manager": "sparkroute"}

    monkeypatch.setattr("sparkrun.daemon.client.Client.info", info)
    result = CliRunner().invoke(daemon, ["--socket", "api.sock", "--token", "manager-token", "status"])
    assert result.exit_code == 0, result.output
    assert seen == [("api.sock", "manager-token")]
