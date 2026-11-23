"""Thin daemon CLI. Importing command metadata never imports the HTTP server."""

from __future__ import annotations

import json

import click


def _binding(ctx):
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.daemon.state import Binding

    return Binding.from_config(SparkrunConfig(ctx.find_object(dict).get("daemon_config")))


def _client(ctx):
    from sparkrun.daemon.client import Client

    options = ctx.find_object(dict)
    if options.get("daemon_endpoint"):
        return Client(**options["daemon_endpoint"])
    return Client(binding=_binding(ctx))


def _json(value):
    click.echo(json.dumps(value, indent=2, default=str))


def _enabled(binding):
    from sparkrun.core.config import SparkrunConfig

    if not SparkrunConfig(binding.config_path).is_feature_enabled("daemon"):
        raise click.ClickException("Daemon feature is disabled; enable it with 'sparkrun setup features enable daemon'")


class DaemonGroup(click.Group):
    def invoke(self, ctx):
        from sparkrun.daemon.client import DaemonError
        import subprocess

        try:
            return super().invoke(ctx)
        except click.exceptions.Exit:
            raise
        except (DaemonError, RuntimeError, OSError, ValueError, LookupError, subprocess.SubprocessError) as exc:
            raise click.ClickException(str(exc)) from exc


@click.group(cls=DaemonGroup, help="Serve the local API, manage durable jobs, and connect automation clients.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False), help="Configuration file binding for this daemon instance.")
@click.option("--url", help="HTTP(S) daemon origin for call, status, jobs, and help (e.g. https://controller:8443).")
@click.option(
    "--socket", "socket_path", type=click.Path(dir_okay=False), help="Explicit local Unix socket for call, status, jobs, and help."
)
@click.option(
    "--token", help="Bearer token for --url or --socket; defaults to SPARKRUN_DAEMON_TOKEN. Prefer --token-file to avoid shell history."
)
@click.option(
    "--token-file", type=click.Path(exists=True, dir_okay=False), help="Read the bearer token for --url or --socket from a plain-text file."
)
@click.option(
    "--ca-cert", type=click.Path(exists=True, dir_okay=False), help="CA certificate bundle for an HTTPS --url; verification stays enabled."
)
@click.pass_context
def daemon(ctx, config_path, url, socket_path, token, token_file, ca_cert):
    ctx.ensure_object(dict)
    ctx.obj["daemon_config"] = config_path
    ctx.obj["daemon_endpoint"] = None
    if any(value is not None for value in (url, socket_path, token, token_file, ca_cert)):
        if ctx.invoked_subcommand not in {"call", "status", "jobs", "help"}:
            raise click.UsageError("Endpoint and authentication options apply only to call, status, jobs, and help.")
        if url is not None and socket_path is not None:
            raise click.UsageError("Choose either --url or --socket.")
        if config_path is not None and (url is not None or socket_path is not None):
            raise click.UsageError("Choose either --config for local discovery or an explicit --url/--socket.")
        if token is not None and token_file is not None:
            raise click.UsageError("Choose either --token or --token-file.")
        if url is None and socket_path is None and any(value is not None for value in (token, token_file)):
            raise click.UsageError("--token and --token-file require --url or --socket.")
        if ca_cert is not None and url is None:
            raise click.UsageError("--ca-cert requires --url.")
        if token_file is not None:
            from pathlib import Path

            with Path(token_file).open(encoding="utf-8") as stream:
                token = stream.read(4097)
            if len(token) > 4096:
                raise click.UsageError("Token file exceeds 4096 characters; expected a plain-text bearer token.")
        elif token is None and url is not None:
            import os

            token = os.environ.get("SPARKRUN_DAEMON_TOKEN")
        ctx.obj["daemon_endpoint"] = {
            "url": url,
            "socket_path": socket_path,
            "token": token.strip() if token is not None else None,
            "ca_cert": ca_cert,
        }


@daemon.command(help="Run the API in the foreground. Workers continue if this process exits.")
@click.option("--unix/--no-unix", default=None, help="Enable Linux peer-authenticated Unix socket (default: Linux only).")
@click.option("--tcp", metavar="HOST:PORT", help="Enable authenticated TCP (e.g. 127.0.0.1:0 for a dynamic port).")
@click.option(
    "--allow-insecure",
    is_flag=True,
    help="Allow non-loopback TCP without TLS. Bearer tokens and traffic are transmitted unencrypted.",
)
@click.option(
    "--allow-request-trust", is_flag=True, default=None, help="Explicitly allow operate clients to authorize recipe hooks over TCP."
)
@click.option(
    "--tls-cert",
    type=click.Path(exists=True, dir_okay=False),
    help="TLS certificate for TCP; required for remote binds unless --allow-insecure is set.",
)
@click.option("--tls-key", type=click.Path(exists=True, dir_okay=False), help="Private key matching the TCP TLS certificate.")
@click.option("--conn-file", type=click.Path(dir_okay=False), help="Exclusively create a private connection file with an ephemeral token.")
@click.option("--parent-pid", type=click.IntRange(min=1), help="Stop this GUI-owned API when the verified parent exits.")
@click.option(
    "--worker-supervisor",
    type=click.Choice(["process", "systemd"]),
    default="process",
    show_default=True,
    help="Use independent systemd user units when running as a service.",
)
@click.pass_context
def serve(ctx, tcp, unix, tls_cert, tls_key, conn_file, parent_pid, worker_supervisor, allow_request_trust, allow_insecure):
    binding = _binding(ctx)
    _enabled(binding)
    from sparkrun.daemon.server import serve as run_server
    import asyncio

    address = None
    if tcp:
        host, separator, port = tcp.rpartition(":")
        if not separator:
            host, port = "127.0.0.1", tcp
        try:
            address = (host.strip("[]"), int(port))
        except ValueError as exc:
            raise click.BadParameter("Expected HOST:PORT", param_hint="--tcp") from exc
    try:
        asyncio.run(
            run_server(
                binding,
                tcp=address,
                unix=unix,
                cert=tls_cert,
                key=tls_key,
                conn_file=conn_file,
                parent_pid=parent_pid,
                worker_supervisor=worker_supervisor,
                allow_request_trust=allow_request_trust,
                allow_insecure=allow_insecure,
            )
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc


@daemon.command(help="Start a background API for this configuration binding.")
@click.pass_context
def start(ctx, launch_options=None):
    from sparkrun.daemon.management import ensure_daemon

    _json(ensure_daemon(_binding(ctx), launch_options))


@daemon.command(help="Inspect identity and runtime state without loading plugins.")
@click.pass_context
def status(ctx):
    if ctx.find_object(dict).get("daemon_endpoint"):
        _json({"running": True, **_client(ctx).info()})
        return
    from sparkrun.core.process_identity import alive

    binding = _binding(ctx)
    state = binding.read_state()
    if not state or not alive(state.get("process")):
        _json({"running": False, **binding.info()})
        return
    try:
        _json({"running": True, **_client(ctx).info()})
    except OSError:
        _json({"running": True, "responsive": False, **state})


@daemon.command(help="Stop the API; active workers continue unless cancellation is requested.")
@click.option("--cancel-jobs", is_flag=True, help="Request cancellation of all active jobs before stopping the API.")
@click.option("--timeout", type=click.FloatRange(min=1), default=30, show_default=True, help="Seconds to wait for drain/shutdown.")
@click.pass_context
def stop(ctx, cancel_jobs, timeout):
    import time
    from sparkrun.core.process_identity import alive

    binding = _binding(ctx)
    state = binding.read_state()
    if not state or not alive(state.get("process")):
        click.echo("Daemon is not running.")
        return
    client = _client(ctx)
    if cancel_jobs:
        client.request("POST", "/v1/maintenance", {"drain": True})
        from sparkrun.daemon.jobs import TERMINAL

        for job in client.request("GET", "/v1/jobs?limit=1000"):
            if job["state"] not in TERMINAL:
                client.request("DELETE", "/v1/jobs/" + job["id"])
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(j["state"] in TERMINAL for j in client.request("GET", "/v1/jobs?limit=1000")):
                break
            time.sleep(0.2)
        else:
            raise click.ClickException("Jobs are still cancelling; the API remains running")
    if state.get("service_unit"):
        import subprocess
        from sparkrun.daemon.supervisor import unit_name

        subprocess.run(["systemctl", "--user", "stop", unit_name(binding)], check=True, timeout=timeout)
    elif alive(state["process"]):
        client.request("POST", "/v1/shutdown", {})
    deadline = time.monotonic() + timeout
    while alive(state["process"]) and time.monotonic() < deadline:
        time.sleep(0.1)
    if alive(state["process"]):
        raise click.ClickException("Daemon did not stop before the deadline")
    click.echo("Daemon stopped. Uncancelled workers continue.")


@daemon.command(help="Restart the API and adopt existing workers.")
@click.pass_context
def restart(ctx):
    state = _binding(ctx).read_state() or {}
    ctx.invoke(stop, cancel_jobs=False, timeout=30)
    ctx.invoke(start, launch_options=state.get("launch"))


@daemon.command("help")
@click.argument("operation", required=False)
@click.option("--json", "as_json", is_flag=True, help="Print the catalog entries as JSON (the GET /v1/ops shape).")
@click.pass_context
def help_(ctx, operation, as_json):
    """List the operations 'daemon call' accepts, or describe one.

    \b
      sparkrun daemon help                  every operation, grouped
      sparkrun daemon help workloads        one namespace
      sparkrun daemon help workloads.run    fields, permissions and an example

    Reads this installation's catalog; with --url or --socket, the catalog of
    that daemon (what its token may call).
    """
    import shutil

    from sparkrun.cli._daemon_help import render_list, render_operation, suggestions

    endpoint = ctx.find_object(dict).get("daemon_endpoint")
    if endpoint:
        catalog = _client(ctx).request("GET", "/v1/ops")
        if not isinstance(catalog, list) or not all(isinstance(op, dict) and isinstance(op.get("name"), str) for op in catalog):
            raise click.ClickException("Unexpected GET /v1/ops response from the daemon; is that a sparkrun daemon of this version?")
        source = "on " + (endpoint.get("url") or endpoint.get("socket_path"))
    else:
        catalog = _local_catalog(ctx)
        source = "in this installation"
    width = shutil.get_terminal_size((100, 20)).columns
    if operation is None or not any(op["name"] == operation for op in catalog):
        members = [op for op in catalog if operation is None or op["name"].startswith(operation + ".")]
        if operation is not None and not members:
            close = suggestions(catalog, operation)
            hint = (" Did you mean: %s?" % ", ".join(close)) if close else ""
            raise click.ClickException("No daemon operation %r.%s List them with: sparkrun daemon help" % (operation, hint))
        if as_json:
            _json(members)
        else:
            click.echo(render_list(members, width=width, source=source, prefix=operation))
        return
    op = next(entry for entry in catalog if entry["name"] == operation)
    if as_json:
        _json(op)
        return
    from sparkrun.daemon.op_docs import operation_docs

    # Argument notes come from this installation's library; a remote daemon's
    # catalog is still authoritative for which fields exist.
    children = sorted(entry["name"] for entry in catalog if entry["name"].startswith(operation + "."))
    click.echo(render_operation(op, width=width, argument_docs=operation_docs(operation)[1], children=children))


def _local_catalog(ctx):
    """The operations a daemon started from this configuration would serve."""
    from sparkrun.application import initialize
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.daemon_registry import operation_catalog
    from sparkrun.daemon.ops import register_core_ops

    config_path = ctx.find_object(dict).get("daemon_config")
    initialize(config_path=config_path)
    register_core_ops()
    return operation_catalog(SparkrunConfig(config_path))


@daemon.command("call", help="Submit a named API operation. Mutations return a durable job ID. List them: 'sparkrun daemon help'.")
@click.argument("operation")
@click.option("--data", default="{}", show_default=True, help="JSON request object (secrets must use job prompts).")
@click.option("--idempotency-key", help="Stable retry key; reuse only for identical arguments.")
@click.option("--wait", is_flag=True, help="Wait for completion and render approval/credential prompts.")
@click.pass_context
def call(ctx, operation, data, idempotency_key, wait):
    try:
        arguments = json.loads(data)
    except ValueError as exc:
        raise click.BadParameter("Expected JSON", param_hint="--data") from exc
    client = _client(ctx)
    result = client.call(operation, arguments, key=idempotency_key)
    if wait and "job_id" in result:
        click.echo("Job: " + result["job_id"], err=True)

        def prompt(value):
            if value["kind"] in {"approval", "decision"}:
                return click.confirm(value["message"], default=False)
            return click.prompt(value["message"], hide_input=value["kind"] == "secret")

        result = client.wait(result["job_id"], prompt=prompt)
    _json(result)


@daemon.group(help="Inspect jobs, replay events, answer prompts, and request cancellation.")
@click.pass_context
def jobs(ctx):
    _client(ctx).info()


@jobs.command("list", help="List durable jobs accessible to the authenticated caller.")
@click.pass_context
def jobs_list(ctx):
    _json(_client(ctx).request("GET", "/v1/jobs"))


@jobs.command("show", help="Show a job's state, result, error, and pending prompt.")
@click.argument("job_id")
@click.pass_context
def jobs_show(ctx, job_id):
    _json(_client(ctx).request("GET", "/v1/jobs/" + job_id))


@jobs.command("cancel", help="Request graceful cancellation; uncertain forced termination is reported as lost.")
@click.argument("job_id")
@click.pass_context
def jobs_cancel(ctx, job_id):
    _json(_client(ctx).request("DELETE", "/v1/jobs/" + job_id))


@jobs.command("events", help="Replay retained events and follow a job through completion.")
@click.argument("job_id")
@click.option("--after", type=click.IntRange(min=0), default=0, help="Resume after this event sequence number.")
@click.pass_context
def jobs_events(ctx, job_id, after):
    for event in _client(ctx).events(job_id, after=after):
        click.echo(json.dumps(event))


@jobs.command("answer", help="Interactively answer a pending prompt; secret input is hidden.")
@click.argument("job_id")
@click.pass_context
def jobs_answer(ctx, job_id):
    client = _client(ctx)
    job = client.request("GET", "/v1/jobs/" + job_id)
    prompt = job.get("prompt")
    if not prompt:
        raise click.ClickException("Job has no pending prompt")
    value = (
        click.confirm(prompt["message"], default=False)
        if prompt["kind"] in {"approval", "decision"}
        else click.prompt(prompt["message"], hide_input=prompt["kind"] == "secret")
    )
    _json(client.request("POST", f"/v1/jobs/{job_id}/input", {"prompt_id": prompt["prompt_id"], "value": value}))


@daemon.group(help="Create, inspect, and revoke hashed bearer-token credentials.")
def token():
    pass


@token.command("create", help="Create a token and print its secret once.")
@click.argument("name")
@click.option(
    "--scope", type=click.Choice(["read", "operate", "admin"]), default="operate", show_default=True, help="Maximum operation permission."
)
@click.option("--secret", is_flag=True, help="Allow answers to credential prompts as well as ordinary operations.")
@click.option("--expires-in", type=click.FloatRange(min=1), help="Expire the credential after this many seconds; omit for no time limit.")
@click.option(
    "--manager",
    default="daemon",
    show_default=True,
    help="Stable workload management identity granted to this credential; use sparkroute for a gateway client.",
)
@click.pass_context
def token_create(ctx, name, scope, secret, expires_in, manager):
    from sparkrun.daemon.auth import TokenStore

    import time

    identifier, value = TokenStore(_binding(ctx)).create(
        name,
        scope=scope,
        secret=secret,
        expires=time.time() + expires_in if expires_in else None,
        manager=manager,
        principal_id="manager:" + manager if manager != "daemon" else None,
    )
    _json({"id": identifier, "token": value})


@token.command("list", help="List token metadata without revealing credentials or hashes.")
@click.pass_context
def token_list(ctx):
    from sparkrun.daemon.auth import TokenStore

    _json(TokenStore(_binding(ctx)).list())


@token.command("revoke", help="Immediately revoke a token by its stable ID.")
@click.argument("token_id")
@click.pass_context
def token_revoke(ctx, token_id):
    from sparkrun.daemon.auth import TokenStore

    TokenStore(_binding(ctx)).revoke(token_id)
    click.echo("Token revoked.")


@daemon.group(help="Install a user service with independently supervised workers.")
def systemd():
    pass


@systemd.command("install", help="Install and enable this binding's systemd user service.")
@click.option("--now", is_flag=True, help="Start the API service immediately after installing.")
@click.pass_context
def systemd_install(ctx, now):
    from sparkrun.daemon.supervisor import install_service

    _enabled(_binding(ctx))
    click.echo(install_service(_binding(ctx), now=now))
    click.echo("For startup before login, enable user lingering with: loginctl enable-linger")


@systemd.command("status", help="Show systemd status for this configuration binding.")
@click.pass_context
def systemd_status(ctx):
    import subprocess
    from sparkrun.daemon.supervisor import unit_name

    result = subprocess.run(["systemctl", "--user", "status", "--no-pager", unit_name(_binding(ctx))], check=False)
    ctx.exit(result.returncode)


@systemd.command("uninstall", help="Stop and remove the API unit; independent worker units remain.")
@click.pass_context
def systemd_uninstall(ctx):
    from sparkrun.daemon.supervisor import uninstall_service

    uninstall_service(_binding(ctx))


@daemon.group(help="Inspect execution reservations and reconcile claims left by lost workers.")
def resources():
    pass


@resources.command("list", help="Show resources, process identities, and whether each owner is alive.")
@click.pass_context
def resource_list(ctx):
    from sparkrun.core.coordination import reservations

    _json(reservations(_binding(ctx).config_path.parent))


@resources.command("reconcile", help="Release a dead owner's claims after inspecting remaining remote effects.")
@click.argument("owner")
@click.option("--acknowledge-effects", is_flag=True, help="Confirm remote workloads/effects have been reconciled.")
@click.pass_context
def resource_reconcile(ctx, owner, acknowledge_effects):
    from sparkrun.core.coordination import reconcile_reservation

    reconcile_reservation(_binding(ctx).config_path.parent, owner, acknowledge_effects=acknowledge_effects)
    click.echo("Reservation released.")


@daemon.command("connection-status", hidden=True)
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def connection_status(path):
    """Validate desktop discovery using the same native security/identity rules."""
    from sparkrun.daemon.state import read_connection
    from sparkrun.core.process_identity import alive

    data = read_connection(path)
    _json({"alive": alive(data["process"]), "process": data["process"], "binding": data.get("binding")})
