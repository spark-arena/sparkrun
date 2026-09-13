# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""SparkRoute engine — supervise the Go binary and reconcile its config.

Peer of :class:`sparkrun.proxy.engine.ProxyEngine`. Both inherit process and
state-file handling from
:class:`~sparkrun.proxy._supervisor.GatewaySupervisor`; what differs is how the
gateway is acquired (a verified GitHub release asset rather than ``uvx``), how
it is configured, and — uniquely here — that the gateway calls *back* into
sparkrun through ``sparkrun gateway-bridge`` to activate workloads.

**Sparkrun never writes a gateway config file.** The gateway runs in SQLite
configuration mode and owns two managed sets, ``operator`` and ``sparkrun``.
Sparkrun replaces its own set whole over the optionally token-protected admin API under
compare-and-swap; it cannot read or write the operator's, so an operator's
routing policy is safe by construction rather than by convention. Nothing here
restarts the gateway to apply a change: the gateway builds a new runtime
generation and drains the old one, so activation leases survive.

``proxy.gateway_config`` is therefore a **one-time bootstrap seed**, passed as
``-config-bootstrap`` and imported into the ``operator`` set only while the
configuration database is empty. ``-config`` is rejected in SQLite mode.

Desired state has two deliberately separate inputs in ``proxy.yaml``: the
durable **binding catalog** for activatable workloads, and the last explicit
``proxy sync`` snapshot for already-running workloads. Discovery can replace
only the latter; an offline workload can therefore never delete the
``activatable`` deployment the gateway needs in order to start it again.

Contract: ``docs/SPARKROUTE_BRIDGE.md`` and the public SparkRoute integration guide:
https://github.com/sparksq/sparkroute/blob/v0.0.2/docs/SPARKRUN_ON_DEMAND.md.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from sparkrun.proxy import DEFAULT_MASTER_KEY, DEFAULT_PROXY_HOST, DEFAULT_PROXY_PORT
from sparkrun.proxy._supervisor import GatewayOperationError, GatewaySupervisor, _restrict_dir_permissions
from sparkrun.proxy.gateway import require_gateway_enabled
from sparkrun.plugins.sparkroute.admin import AdminClient, AdminError, RevisionConflict
from sparkrun.plugins.sparkroute.credentials import CredentialError, ReconcilerCredential
from sparkrun.plugins.sparkroute.projection import ProjectionError, build_sparkrun_set, dedupe_bindings, resolve_bindings
from sparkrun.plugins.sparkroute.release import GatewayReleaseError, ensure_binary, resolve_sparkrun_executable

logger = logging.getLogger(__name__)

#: Defaults for an explicitly split admin listener when only one component is
#: configured. With neither component configured, admin is co-located on data.
DEFAULT_ADMIN_PORT = 8081
DEFAULT_ADMIN_HOST = "127.0.0.1"

#: Bind values that mean "every interface" — not connectable as written.
WILDCARD_BIND_HOSTS = frozenset({"0.0.0.0", "::", "[::]", ""})

#: Bind values that keep the listener on this machine.
LOOPBACK_BIND_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

#: How long to wait for the admin listener after spawning the gateway.
ADMIN_READY_TIMEOUT_SECONDS = 30.0
ADMIN_READY_POLL_SECONDS = 0.5

#: Attempts for the CAS reconcile loop. A conflict means an operator wrote
#: between our read and our write, so re-reading and rebuilding is the correct
#: response — but not forever.
RECONCILE_ATTEMPTS = 5
RECONCILE_BACKOFF_SECONDS = 0.5

# Managed-set replacement commits storage before the watcher swaps the serving
# runtime generation. CLI/API reconciliation is synchronous from the user's
# point of view, so wait for those two revisions to meet instead of reporting a
# model as registered while an immediate status call still sees the old set.
SERVING_READY_TIMEOUT_SECONDS = 30.0
SERVING_READY_POLL_SECONDS = 0.05


class SparkrouteConfigError(GatewayOperationError):
    """The gateway rejected sparkrun's configuration, or none could be built."""


class SparkrouteEngine(GatewaySupervisor):
    """Acquire, supervise, and reconcile the SparkRoute binary."""

    #: Selector this implementation answers to (``proxy.gateway`` in proxy.yaml).
    gateway_name = "sparkroute"

    #: Feature flag gating this gateway; off on every channel.
    required_feature_flag = "gateway.sparkroute"

    log_name = "sparkroute.log"

    #: Periodic discovery updates only the warm-only snapshot. Durable recipe
    #: bindings remain the activatable desired state, so the two cannot fight.
    supports_autodiscover = True

    #: Needs the whole ``proxy.yaml`` view (bindings, aliases, admin port,
    #: capability policy), not just one path. Declared as a capability so the
    #: API layer adapts without a per-gateway name check.
    wants_proxy_config = True

    def __init__(
        self,
        host: str = DEFAULT_PROXY_HOST,
        port: int = DEFAULT_PROXY_PORT,
        master_key: str | None = DEFAULT_MASTER_KEY,
        state_dir: Path | None = None,
        host_configured: bool = False,
        proxy_config: Any = None,
        sctx: Any = None,
    ):
        super().__init__(state_dir)
        self.host = host
        self.port = port
        # Retained for the shared state payload and parity with ProxyEngine.
        # NOT forwarded: the gateway authenticates callers via
        # ``-caller-auth-mode`` and managed credentials, so presenting a shared
        # bearer token as equivalent would misrepresent what guards the
        # listener.
        self.master_key = master_key
        self.host_configured = host_configured
        self.proxy_config = proxy_config
        self.sctx = sctx
        self._discovery_labels: dict[str, Any] | None = None
        self.credential = ReconcilerCredential(self.state_dir / "gateway")

    def _state_payload(self) -> dict[str, Any]:
        return {
            "port": self.port,
            "host": self.host,
            "master_key": self.master_key,
            "admin_port": self.admin_port,
            "admin_host": self.admin_bind_host,
            "admin_separate": self.admin_separate,
        }

    # -- Addresses ----------------------------------------------------------

    @property
    def data_address(self) -> str:
        """``-data-address`` — where clients send inference requests."""
        return "%s:%d" % (self.host, self.port)

    @property
    def data_plane_authenticated(self) -> bool:
        """A supplied master key protects SparkRoute's inference listener."""
        return bool(self.master_key)

    @property
    def _master_key_requires_admin_auth(self) -> bool:
        state = self.get_state() or {}
        return bool(self.master_key or state.get("master_key"))

    @property
    def admin_auth_required(self) -> bool:
        """Whether the live admin token currently protects the admin surface."""
        if self._master_key_requires_admin_auth:
            return True
        try:
            return self.credential.read_admin_token() is not None
        except CredentialError:
            # SparkRoute fails closed for malformed or unreadable token files.
            return True

    @property
    def allow_insecure_admin_nonloopback(self) -> bool:
        """Whether the operator explicitly acknowledged off-loopback admin."""
        return bool(getattr(self.proxy_config, "gateway_allow_insecure_admin_nonloopback", False))

    @property
    def admin_nonloopback_permitted(self) -> bool:
        """Token-file auth can be cleared live, so public binds need opt-in."""
        return self.admin_exposed or self.allow_insecure_admin_nonloopback

    @property
    def admin_separate(self) -> bool:
        """Whether admin has its own listener instead of sharing data.

        Running state wins for management commands so a newly edited
        ``proxy.yaml`` cannot make the CLI contact a port the current process
        was not started with. A stopped process follows current configuration.
        """
        state = self.get_state() or {}
        if state and (self.proxy_config is None or self.is_running()):
            if "admin_separate" in state:
                return bool(state["admin_separate"])
            # Migration for state files written before the explicit marker.
            if state.get("admin_port") is not None or state.get("admin_host") is not None:
                return True
        configured = getattr(self.proxy_config, "gateway_admin_configured", None)
        if configured is not None:
            return bool(configured)
        # Older/double configs have no marker; the presence of either value
        # preserves their historical dedicated-listener behavior.
        return self.proxy_config is not None and (
            hasattr(self.proxy_config, "gateway_admin_port") or hasattr(self.proxy_config, "gateway_admin_host")
        )

    @property
    def admin_port(self) -> int:
        """Admin listener port: configured first, else whatever is running.

        Management paths (``proxy status`` / ``models`` / ``alias``) resolve
        their engine from the *state file* via ``_running_engine``, which
        constructs it without a ``proxy_config`` — so without the state
        fallback they would talk to the default port and silently fail to see
        a gateway started on a configured one.

        Configured wins because ``start`` must bind where the user asked, and
        at that point the state file still describes the *previous* run.
        """
        state = self.get_state() or {}
        if state and (self.proxy_config is None or self.is_running()) and state.get("admin_port") is not None:
            return int(state["admin_port"])
        if not self.admin_separate:
            return self.port
        return int(getattr(self.proxy_config, "gateway_admin_port", DEFAULT_ADMIN_PORT) or DEFAULT_ADMIN_PORT)

    @property
    def admin_bind_host(self) -> str:
        """Where admin binds, following data when it is co-located."""
        state = self.get_state() or {}
        if state and (self.proxy_config is None or self.is_running()) and state.get("admin_host") is not None:
            return str(state["admin_host"])
        if not self.admin_separate:
            return self.host
        return str(getattr(self.proxy_config, "gateway_admin_host", DEFAULT_ADMIN_HOST) or DEFAULT_ADMIN_HOST)

    @property
    def admin_address(self) -> str:
        """Effective bind address, shared with data unless explicitly split."""
        return "%s:%d" % (self.admin_bind_host, self.admin_port)

    @property
    def admin_connect_host(self) -> str:
        """Host sparkrun and a local browser should *connect* to.

        A wildcard bind is not a connectable address, so it resolves back to
        loopback — sparkrun runs on the same machine as the gateway, and a URL
        printed for a human has to be one they can actually open.
        """
        host = self.admin_bind_host
        return "127.0.0.1" if host in WILDCARD_BIND_HOSTS else host

    @property
    def admin_exposed(self) -> bool:
        """True when the admin listener is reachable beyond this machine."""
        return self.admin_bind_host not in LOOPBACK_BIND_HOSTS

    @property
    def admin_url(self) -> str:
        return "http://%s:%d" % (self.admin_connect_host, self.admin_port)

    @property
    def ui_url(self) -> str:
        """Where the gateway's admin console is served.

        There is no toggle for this: the console lives on the admin listener,
        which sparkrun requires in order to reconcile at all, so it is always
        up. By default it needs no sign-in. ``sparkrun proxy admin-token set``
        creates one owner-only live token, and ``admin-token get`` displays it.
        """
        return "%s/admin" % self.admin_url

    def issue_ui_credential(self) -> str:
        """Create or return the live admin token without restarting SparkRoute."""
        try:
            return self.credential.ensure_admin_token()
        except CredentialError as exc:
            raise SparkrouteConfigError(str(exc)) from exc

    def admin_token(self, *, rotate: bool = False, clear: bool = False) -> str | None:
        """Read, rotate, or clear the live Sparkrun-managed admin token."""
        try:
            if clear:
                if self._master_key_requires_admin_auth:
                    raise SparkrouteConfigError("Admin authentication cannot be cleared while a proxy master key is configured")
                self.credential.clear_admin_token()
                return None
            if rotate:
                return self.credential.ensure_admin_token(force=True)
            return self.credential.read_admin_token()
        except CredentialError as exc:
            raise SparkrouteConfigError(str(exc)) from exc

    # -- Configuration ------------------------------------------------------

    def _permissive(self) -> bool:
        return str(getattr(self.proxy_config, "capability_policy", "permissive")) != "strict"

    def _aliases(self) -> dict[str, str]:
        return dict(getattr(self.proxy_config, "aliases", {}) or {})

    def build_desired_set(
        self,
        aliases: dict[str, str] | None = None,
        *,
        discovered_models: list[str] | None = None,
        discovered_clusters: dict[str, list[str]] | None = None,
        binding_clusters: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Project ``proxy.yaml``'s bindings into the complete sparkrun set.

        Raises:
            SparkrouteConfigError: A binding names a recipe that cannot be
                resolved, or declares a contradictory capability.
        """
        bindings = list(getattr(self.proxy_config, "bindings", []) or ())
        models = list(getattr(self.proxy_config, "discovered_models", []) or ()) if discovered_models is None else discovered_models
        try:
            entries = dedupe_bindings(resolve_bindings(bindings, sctx=self.sctx))
            return build_sparkrun_set(
                entries,
                self._aliases() if aliases is None else aliases,
                permissive=self._permissive(),
                discovered_models=models,
                discovered_apis=self._read_discovered_apis(),
                discovered_clusters=self._read_discovery_labels() if discovered_clusters is None else discovered_clusters,
                binding_clusters=self._read_display_labels()["recipes"] if binding_clusters is None else binding_clusters,
            )
        except ProjectionError as exc:
            raise SparkrouteConfigError(str(exc)) from exc

    def prepare_config(self, endpoints: list, aliases: dict[str, str], *, write: bool = True) -> tuple[Path | None, set[str], set[str]]:
        """Compute the alias preview; see the base method.

        No gateway document is written and no request is made — configuration
        lives in the gateway's database and is reconciled once it is running
        (see :meth:`start`). Healthy endpoints from the startup discovery pass
        do update the separate warm-only snapshot when *write* is true; they
        never replace the durable binding catalog.
        """
        discovered = self._discovered_model_names(endpoints)
        clusters, binding_clusters = self._discovered_cluster_names(endpoints, discovered)
        if write:
            self._persist_discovered_apis(endpoints)
            self._persist_discovered_models(discovered)
            self._persist_discovery_labels(clusters, binding_clusters)
        document = self.build_desired_set(
            aliases, discovered_models=discovered, discovered_clusters=clusters, binding_clusters=binding_clusters
        )
        bound = {model["name"] for model in document["virtual_models"]}
        applied = {name for name, target in aliases.items() if target in bound}
        return None, applied, set(aliases) - applied

    # -- Lifecycle ----------------------------------------------------------

    def build_command(self, binary: Path) -> list[str]:
        """Argv for the gateway process.

        ``-sparkrun-command`` must be a single executable: the gateway runs it
        as ``exec.Command(command, "gateway-bridge")``, appending exactly one
        argument.

        ``-list-aliases`` is passed rather than left at the gateway's default
        because under LiteLLM an alias is an ordinary ``model_name`` entry and
        always appears in ``GET /v1/models``. Omitting it here would leave
        sparkrun's aliases addressable but not *discoverable*, which breaks the
        drop-in promise for any client that enumerates models instead of being
        told them.
        """
        cmd = [
            str(binary),
            "-config-source",
            "sqlite",
            "-config-sqlite",
            str(self.credential.config_db),
            "-admin-auth-mode",
            "token-file",
            "-admin-token-file",
            str(self.credential.admin_token_file),
            "-data-address",
            self.data_address,
            "-list-aliases",
            "-sparkrun",
            "-sparkrun-command",
            resolve_sparkrun_executable(),
        ]
        if self.master_key:
            cmd += [
                "-caller-auth-mode",
                "token-file",
                "-caller-token-file",
                str(self.credential.caller_token_file),
            ]
        if self.admin_nonloopback_permitted:
            cmd += ["-allow-insecure-admin-nonloopback"]
        if self.admin_separate:
            cmd += ["-admin-address", self.admin_address]
        bootstrap = getattr(self.proxy_config, "gateway_config", None)
        if bootstrap:
            # Seeds the operator set on a first, empty database; ignored after.
            cmd += ["-config-bootstrap", str(Path(bootstrap).expanduser())]
        return cmd

    def start(
        self,
        config_path: Path | None = None,
        foreground: bool = False,
        dry_run: bool = False,
        autodiscover_kwargs: dict | None = None,
    ) -> int:
        """Provision, launch, and reconcile.

        Args:
            config_path: Ignored — configuration lives in the gateway's
                database. Accepted for signature parity with ``ProxyEngine``.
            foreground: Run blocking rather than detached. No reconcile happens
                in this mode; the caller owns the process's lifetime.
            dry_run: Report what would happen and change nothing.
            autodiscover_kwargs: Shared discovery-sidecar settings. Periodic
                sweeps reconcile only warm discovered routes.

        Returns:
            0 on success, non-zero on failure.

        Raises:
            GatewayUnavailableError: this gateway's feature flag is off.
        """
        # The one enforcement point, checked before --dry-run so a dry run
        # cannot advertise a start that would be refused.
        require_gateway_enabled(self.gateway_name)
        try:
            binary = ensure_binary()
        except GatewayReleaseError as exc:
            logger.error("%s", exc)
            return 1

        if dry_run:
            self._warn_admin_exposure()
            # Build the desired set too, so a dry run surfaces an unresolvable
            # recipe rather than deferring it to the first real start.
            try:
                document = self.build_desired_set()
            except SparkrouteConfigError as exc:
                logger.error("%s", exc)
                return 1
            logger.info(
                "[dry-run] Would run SparkRoute on %s (admin %s) and reconcile %d deployment(s)",
                self.data_address,
                self.admin_address,
                len(document["deployments"]),
            )
            return 0

        if self.is_running():
            logger.warning("Gateway already running (PID %s)", self._read_pid())
            return 1

        self.state_dir.mkdir(parents=True, exist_ok=True)
        _restrict_dir_permissions(self.state_dir)

        try:
            self._prepare_live_tokens()
            cmd = self.build_command(binary)
        except (CredentialError, GatewayReleaseError) as exc:
            logger.error("%s", exc)
            return 1

        self._warn_admin_exposure()
        self._warn_insecure_bind()

        env = os.environ.copy()

        if foreground:
            proc = subprocess.Popen(cmd, env=env)
            self._save_state(proc.pid)
            if autodiscover_kwargs:
                autodiscover_pid = self.start_autodiscover(
                    proxy_pid=proc.pid,
                    **autodiscover_kwargs,
                )
                if autodiscover_pid:
                    self.update_autodiscover_pid(autodiscover_pid)
            try:
                return proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                return 130
            finally:
                self.stop_autodiscover()
                self._clear_state()

        pid = self._launch_background(cmd, env)
        if pid is None:
            return 1
        self._save_state(pid)
        logger.info("Gateway started (PID %d) on %s (admin %s)", pid, self.data_address, self.admin_address)
        logger.info("Log: %s", self.log_path)

        try:
            self._await_admin_ready()
            added, removed = self.reconcile(reason="gateway start")
            if added or removed:
                logger.info("Reconciled gateway configuration: +%d, -%d", added, removed)
        except (AdminError, SparkrouteConfigError) as exc:
            # The process is up and serving whatever the database already held.
            # A failed reconcile is not a failed start, and reporting it as one
            # would leave a running gateway behind a non-zero exit code.
            logger.error("Gateway started but its configuration could not be reconciled: %s", exc)
        if autodiscover_kwargs:
            autodiscover_pid = self.start_autodiscover(
                proxy_pid=pid,
                **autodiscover_kwargs,
            )
            if autodiscover_pid:
                self.update_autodiscover_pid(autodiscover_pid)
        return 0

    def _prepare_live_tokens(self) -> None:
        """Apply master-key auth without exposing secrets in process argv."""
        if self.master_key:
            # Seed admin with the master key only when no independently rotated
            # admin token exists. Later admin-token rotations remain durable.
            self.credential.ensure_admin_token(preferred=self.master_key)
        self.credential.set_caller_token(self.master_key)

    def _warn_admin_exposure(self) -> None:
        """Say out loud what the admin surface is open to.

        Unauthenticated admin is the default, so this is the only thing
        standing between a convenient local install and an unauthenticated
        config-rewrite/workload-activation API on the LAN. It is a warning
        rather than a refusal by explicit choice — the data listener is
        already unauthenticated on the same legacy ``0.0.0.0`` bind, and a
        gateway that refuses to start is not a security boundary anyone keeps.
        """
        if self.admin_auth_required:
            if self.admin_exposed:
                logger.warning(
                    "Admin listener bound to %s — the console and managed-configuration API are reachable off this host. "
                    "Access needs a gateway credential; treat one that can write configuration like any other secret.",
                    self.admin_bind_host,
                )
            return

        if not self.admin_exposed:
            logger.warning("Admin authentication is DISABLED on %s (loopback only)", self.admin_address)
            return

        logger.warning(
            "\n"
            "============================================================\n"
            "  DANGER: sparkrun gateway admin is UNAUTHENTICATED on %s\n"
            "  (ALL network interfaces). Anyone who can reach it can\n"
            "  rewrite the served model set and activate workloads on\n"
            "  your cluster.\n"
            "  To close it, either bind to localhost:\n"
            "      sparkrun proxy start --host 127.0.0.1\n"
            "  or require a token:\n"
            "      sparkrun proxy admin-token set\n"
            "  (the token takes effect immediately; no restart is needed)\n"
            "============================================================",
            self.admin_address,
        )

    def _await_admin_ready(self) -> None:
        """Poll the admin listener until it answers."""
        deadline = time.monotonic() + ADMIN_READY_TIMEOUT_SECONDS
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self.admin_client().active_revision()
                return
            except AdminError as exc:
                if not exc.retryable:
                    raise
                last = exc
            time.sleep(ADMIN_READY_POLL_SECONDS)
        raise AdminError("gateway admin API did not become ready within %.0fs" % ADMIN_READY_TIMEOUT_SECONDS) from last

    def reset(self) -> list[Path]:
        """Delete the gateway's configuration database (gateway must be stopped).

        The credential database is left alone: it is independent, and removing
        it would strand the stored secret while leaving the gateway unable to
        authenticate sparkrun.

        Raises:
            SparkrouteConfigError: The gateway is still running.
        """
        if self.is_running():
            raise SparkrouteConfigError("Stop the gateway before resetting its configuration database")
        return self.credential.reset_configuration()

    # -- Reconciliation -----------------------------------------------------

    def admin_client(self) -> AdminClient:
        """Client bound to the admin listener with the current live token.

        The token is reread for every new client so live rotation cannot leave
        Sparkrun using a credential cached at process startup.
        """
        try:
            return AdminClient(self.admin_url, self.credential.read_admin_token())
        except CredentialError as exc:
            raise SparkrouteConfigError(str(exc)) from exc

    def reconcile(self, aliases: dict[str, str] | None = None, *, reason: str = "") -> tuple[int, int]:
        """Replace the sparkrun managed set with the current desired state.

        Idempotent by content: when the desired set already matches, the
        gateway reports ``changed: false`` and creates no revision, activation
        or audit event — so a periodic sweep costs two reads and never disturbs
        serving.

        A 409 means an operator wrote between our read and our write. That is
        ordinary control-plane behaviour rather than an error, so we re-read
        the active revision, rebuild, and retry — never retry with the stale
        token, which would only conflict again.

        Returns:
            ``(added, removed)`` entity counts relative to the previous set.

        Raises:
            SparkrouteConfigError: The gateway rejected the document.
            AdminError: Transport, auth, or exhausted retries.
        """
        client = self.admin_client()
        document = self.build_desired_set(aliases)
        last_conflict: Exception | None = None

        for attempt in range(RECONCILE_ATTEMPTS):
            active = client.active_revision()
            current = client.get_set().get("document") or {}
            added, removed = _entity_delta(current, document)
            try:
                result = client.replace(document, active, reason or "sparkrun reconcile")
            except RevisionConflict as exc:
                last_conflict = exc
                # Rebuild rather than resubmit: our own desired state may also
                # have moved while we were waiting.
                document = self.build_desired_set(aliases)
                time.sleep(RECONCILE_BACKOFF_SECONDS * (attempt + 1))
                continue
            except AdminError as exc:
                if exc.code == "invalid_configuration":
                    # The gateway's message names the conflicting entity and
                    # its owner — something sparkrun cannot determine itself,
                    # since it cannot read the operator set. Surface verbatim.
                    raise SparkrouteConfigError(str(exc)) from exc
                raise
            self._await_serving_config(client)
            if not result.get("changed", True):
                return 0, 0
            return added, removed

        raise AdminError(
            "gateway configuration kept changing under us; gave up after %d attempts" % RECONCILE_ATTEMPTS,
            code="revision_conflict",
        ) from last_conflict

    def _await_serving_config(self, client: AdminClient) -> None:
        """Wait until storage and the serving runtime expose one revision.

        The active revision may advance again while we wait (the operator set
        has an independent writer), so compare the two live values on every
        poll rather than pinning the revision returned by our own PUT.
        """
        deadline = time.monotonic() + SERVING_READY_TIMEOUT_SECONDS
        last_error: AdminError | None = None
        while True:
            try:
                active = client.active_revision()
                payload = client.status()
                if "config_revision" not in payload or not isinstance(payload["config_revision"], str):
                    raise AdminError("gateway status response is missing config_revision")
                serving = payload["config_revision"]
                if active and serving == active:
                    return
                last_error = None
            except AdminError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
            if time.monotonic() >= deadline:
                raise AdminError(
                    "gateway stored configuration did not reach the serving runtime within %.0fs" % SERVING_READY_TIMEOUT_SECONDS
                ) from last_error
            time.sleep(SERVING_READY_POLL_SECONDS)

    # -- Model management (the sparkrun proxy commands) ---------------------

    def _discovered_cluster_names(self, endpoints: list, models: list[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """Label healthy endpoints and recipe bindings from the same job metadata."""
        from sparkrun import api

        live = [endpoint for endpoint in endpoints if bool(getattr(endpoint, "healthy", False))]
        if not live:
            return {}, {}
        jobs: dict[str, Any] = {}
        try:
            jobs = {job.cluster_id: job.metadata or {} for job in api.list_jobs(sctx=self.sctx)}
        except Exception:
            logger.debug("Job display metadata unavailable; using discovery fields", exc_info=True)
        names: dict[str, set[str]] = {}
        recipes: dict[str, set[str]] = {}
        for endpoint in live:
            cluster_id = str(getattr(endpoint, "cluster_id", "") or "")
            metadata = jobs.get(cluster_id, {})
            cluster_name = getattr(endpoint, "cluster_name", None) or metadata.get("cluster")
            cluster = str(cluster_name or cluster_id or "discovered")
            revision = getattr(endpoint, "recipe_revision", "") or metadata.get("recipe_fingerprint")
            if revision and cluster_name:
                # Match bindings by launch fingerprint, never just upstream model name.
                recipes.setdefault(str(revision), set()).add(str(cluster_name))
            candidates = getattr(endpoint, "actual_models", None) or [
                getattr(endpoint, "served_model_name", None) or getattr(endpoint, "model", None)
            ]
            for candidate in candidates:
                model = str(candidate or "").strip()
                if model in models:
                    names.setdefault(model, set()).add(cluster)
        return (
            {model: sorted(clusters) for model, clusters in names.items()},
            {revision: sorted(clusters) for revision, clusters in recipes.items()},
        )

    @property
    def _discovery_labels_path(self) -> Path:
        return self.state_dir / "sparkroute-discovery-labels.json"

    def _read_display_labels(self) -> dict[str, Any]:
        """Optional presentation cache; never a source of routing or lifecycle IDs."""
        if self._discovery_labels is not None:
            return self._discovery_labels

        def valid_labels(value: Any) -> bool:
            return isinstance(value, dict) and all(
                isinstance(names, list) and all(isinstance(name, str) for name in names) for names in value.values()
            )

        try:
            value = json.loads(self._discovery_labels_path.read_text(encoding="utf-8"))
            if (
                isinstance(value, dict)
                and value.get("schema") == 2
                and valid_labels(value.get("models"))
                and valid_labels(value.get("recipes"))
            ):
                return {"schema": 2, "models": value["models"], "recipes": value["recipes"]}
            if valid_labels(value):  # Prior cache held discovery-only model labels.
                return {"schema": 2, "models": value, "recipes": {}}
        except (OSError, ValueError):
            pass
        return {"schema": 2, "models": {}, "recipes": {}}

    def _read_discovery_labels(self) -> dict[str, list[str]]:
        return self._read_display_labels()["models"]

    def _persist_discovery_labels(self, labels: dict[str, list[str]], binding_labels: dict[str, list[str]]) -> None:
        """Keep labels across CLI invocations without rewriting unchanged snapshots."""
        previous = self._read_display_labels()
        snapshot = {"schema": 2, "models": labels, "recipes": binding_labels}
        self._discovery_labels = snapshot
        if snapshot == previous:
            return
        temporary: Path | None = None
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            _restrict_dir_permissions(self.state_dir)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.state_dir, prefix=".sparkroute-labels-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(snapshot, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._discovery_labels_path)
        except OSError:
            logger.debug("Could not cache display labels; discovery still applies", exc_info=True)
        finally:
            if temporary is not None:
                with contextlib.suppress(OSError):
                    temporary.unlink(missing_ok=True)

    def _read_discovered_apis(self) -> dict[str, dict[str, Any]]:
        try:
            value = json.loads((self.state_dir / "sparkroute-discovery-apis.json").read_text())
            if isinstance(value, dict) and all(
                isinstance(entry, dict)
                and all(
                    isinstance(entry.get(key), list) and all(isinstance(item, str) for item in entry[key])
                    for key in ("native_protocols", "capabilities")
                )
                for entry in value.values()
            ):
                return value
        except (OSError, ValueError):
            pass
        return {}

    def _persist_discovered_apis(self, endpoints: list) -> None:
        from .recipe_config import parse_sparkroute, SparkrouteRecipeError

        from sparkrun import api
        from .metadata import build_model_metadata, reduce_model_metadata
        from .operations import _recipe_of

        jobs = {}
        if endpoints:
            try:
                jobs = {job.cluster_id: job for job in api.list_jobs(sctx=self.sctx)}
            except Exception:
                logger.debug("Job model metadata unavailable", exc_info=True)
        snapshot: dict[str, dict[str, Any]] = {}
        declared_profiles: dict[str, dict[str, Any]] = {}
        for endpoint in endpoints:
            if not getattr(endpoint, "healthy", False):
                continue
            protocols = list(getattr(endpoint, "native_protocols", None) or ["openai"])
            try:
                items = getattr(endpoint, "plugin_items", {}) or {}
                settings = parse_sparkroute(items.get("sparkroute", {}), source="discovered recipe")
            except SparkrouteRecipeError as error:
                raise SparkrouteConfigError(str(error)) from error
            capabilities = sorted(set(getattr(endpoint, "capabilities", None) or []) | set(settings.get("capabilities", [])))
            profiles = settings.get("request_profiles", {})
            for model in getattr(endpoint, "actual_models", None) or [
                getattr(endpoint, "served_model_name", None) or getattr(endpoint, "model", "")
            ]:
                if not model:
                    continue
                declared = declared_profiles.setdefault(model, {})
                for selector, parameters in profiles.items():
                    if selector in declared and declared[selector] != parameters:
                        raise SparkrouteConfigError(f"Discovered recipes serving {model!r} define conflicting request profiles")
                    declared[selector] = parameters
                if model in snapshot:
                    # One discovered model can cover multiple jobs: advertise
                    # only APIs shared by all candidates for that model.
                    shared_protocols = [p for p in protocols if p in snapshot[model]["native_protocols"]]
                    shared_capabilities = [c for c in capabilities if c in snapshot[model]["capabilities"]]
                    previous_profiles = snapshot[model].get("request_profiles", {})
                    common = set(profiles) & set(previous_profiles)
                    shared_profiles = {key: profiles[key] for key in sorted(common)}
                else:
                    shared_protocols, shared_capabilities, shared_profiles = protocols, capabilities, profiles
                job = jobs.get(getattr(endpoint, "cluster_id", ""))
                fields = build_model_metadata(_recipe_of(job), [model]).get(model, {})
                context = getattr(endpoint, "max_model_len", None)
                if isinstance(context, int) and not isinstance(context, bool) and context > 0:
                    fields["context"] = context
                recipe_metadata = dict(snapshot.get(model, {}).get("recipe_metadata", {}))
                revision = getattr(endpoint, "recipe_revision", "") or (getattr(job, "metadata", None) or {}).get("recipe_fingerprint")
                if revision and fields:
                    recipe_metadata[revision] = reduce_model_metadata(recipe_metadata.get(revision, {}), fields)
                fields = reduce_model_metadata(snapshot.get(model, {}).get("model_metadata", {}), fields)
                snapshot[model] = {"native_protocols": shared_protocols, "capabilities": shared_capabilities}
                if fields:
                    snapshot[model]["model_metadata"] = fields
                if recipe_metadata:
                    snapshot[model]["recipe_metadata"] = recipe_metadata
                if shared_profiles:
                    snapshot[model]["request_profiles"] = shared_profiles
        if snapshot == self._read_discovered_apis():
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        _restrict_dir_permissions(self.state_dir)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.state_dir, prefix=".sparkroute-apis-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(snapshot, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_dir / "sparkroute-discovery-apis.json")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _discovered_model_names(self, endpoints: list) -> list[str]:
        """Normalize a live endpoint snapshot and exclude bound models."""
        models: set[str] = set()
        for endpoint in endpoints:
            if not bool(getattr(endpoint, "healthy", False)):
                continue
            candidates = list(getattr(endpoint, "actual_models", None) or ())
            if not candidates:
                fallback = getattr(endpoint, "served_model_name", None) or getattr(endpoint, "model", None)
                if fallback:
                    candidates = [fallback]
            for candidate in candidates:
                model = str(candidate).strip()
                if model:
                    models.add(model)

        # A bound model is already projected as activatable and the
        # controller adopts its warm endpoint directly. Do not also remember
        # it as a discovery import: if the binding is later unloaded, that
        # stale import must not make the stopped model reappear.
        try:
            bound = dedupe_bindings(resolve_bindings(list(self.proxy_config.bindings), sctx=self.sctx))
        except ProjectionError as exc:
            raise SparkrouteConfigError(str(exc)) from exc
        models.difference_update(entry.virtual_model for entry in bound)
        return sorted(models)

    def _persist_discovered_models(self, models: list[str]) -> bool:
        """Persist a changed discovery snapshot; return whether it changed."""
        current = set(getattr(self.proxy_config, "discovered_models", []) or ())
        if set(models) == current:
            return False
        self.proxy_config.set_discovered_models(models)
        self.proxy_config.save()
        return True

    def sync_models(self, endpoints: list, aliases: dict[str, str] | None = None) -> tuple[int, int]:
        """Import the live discovery snapshot, then reconcile the managed set.

        Imported models render as ``discovered`` sources. They are kept apart
        from recipe bindings so a later sync can remove a stopped warm-only
        workload without deleting an activatable cold route.
        """
        if self.proxy_config is None:
            raise SparkrouteConfigError("SparkRoute model sync requires proxy.yaml")

        models = self._discovered_model_names(endpoints)
        self._persist_discovered_apis(endpoints)
        self._persist_discovered_models(models)
        self._persist_discovery_labels(*self._discovered_cluster_names(endpoints, models))
        return self.reconcile(aliases, reason="sparkrun sync")

    def sync_aliases(self, aliases: dict[str, str]) -> tuple[int, int]:
        """Apply *aliases* by reconciling the managed set."""
        return self.reconcile(aliases, reason="sparkrun alias update")

    def register_loaded_model(
        self,
        recipe: str,
        overrides: dict[str, Any] | None = None,
        cluster: str | None = None,
    ) -> None:
        """Request the host's normal discovery sync after a manual load.

        Loading a workload is not an opt-in to future automatic activation.
        Returning None uses api.proxy's discovery path and leaves explicit
        recipe bindings unchanged. On-demand recipes belong in the UI or an
        explicitly authored proxy.yaml binding.
        """
        return None

    def unregister_loaded_model(self, recipe: str) -> None:
        """Remove every managed binding resolving to *recipe*.

        A recipe may have several override variants. ``proxy unload`` already
        addresses the recipe rather than one binding revision, so removing all
        of those variants keeps workload teardown and route teardown aligned.
        """
        if self.proxy_config is None:
            raise SparkrouteConfigError("SparkRoute model removal requires proxy.yaml")

        bindings = list(self.proxy_config.bindings)
        if not bindings:
            return None
        import sparkrun.api as api

        try:
            target_recipe, _ = api.resolve_catalog_recipe(str(recipe), sctx=self.sctx)
            target = Path(target_recipe.source_path).resolve()
            retained = []
            for binding in bindings:
                # Compare the resolved source, not @registry/name against a
                # cached YAML path. Overrides do not change recipe identity.
                current, _ = api.resolve_catalog_recipe(str(binding.get("recipe") or ""), sctx=self.sctx)
                if Path(current.source_path).resolve() != target:
                    retained.append(binding)
        except api.SparkrunError as exc:
            raise SparkrouteConfigError(str(exc)) from exc
        if len(retained) != len(bindings):
            self.proxy_config.set_bindings(retained)
            self.proxy_config.save()
        # Rescan even when there was no binding: the stopped workload may have
        # been discovered, and its last warm snapshot must not survive unload.
        return None

    def list_models_via_api(self) -> list[dict[str, Any]]:
        """Every model the gateway serves, aliases included.

        Read from ``GET /v1/status``, whose ``served_model_names`` spans *both*
        owner sets — sparkrun's reconcile role cannot read the operator
        document, so this is the only way to answer "what can I call?"
        completely. The API base is the gateway's own listener, which is where
        a client actually sends the request.
        """
        self.model_query_error = ""
        try:
            payload = self.admin_client().status()
            if "served_model_names" not in payload:
                raise AdminError("gateway status response is missing served_model_names")
            names = payload["served_model_names"]
            if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
                raise AdminError("gateway status response has invalid served_model_names")
        except (AdminError, SparkrouteConfigError) as exc:
            self.model_query_error = str(exc)
            logger.debug("Could not read served model names from the gateway", exc_info=True)
            return []
        base = "http://%s/v1" % self.data_address
        return [{"model_name": str(name), "api_base": base} for name in names]

    def serving_revision(self) -> str:
        """Revision currently serving requests, for convergence checks.

        Differs briefly from the storage ``active_revision`` after a write,
        while the gateway builds a new runtime generation.
        """
        try:
            return str(self.admin_client().status().get("config_revision") or "")
        except (AdminError, SparkrouteConfigError):
            return ""


def _entity_delta(current: dict[str, Any], desired: dict[str, Any]) -> tuple[int, int]:
    """Count entities added and removed between two managed-set documents."""

    def keys(document: dict[str, Any]) -> set[tuple[str, str]]:
        result: set[tuple[str, str]] = set()
        for kind in ("providers", "deployments", "virtual_models"):
            for entity in document.get(kind) or ():
                name = str(entity.get("name", ""))
                result.add((kind, name))
                for alias in entity.get("aliases") or ():
                    result.add((kind, "%s/%s" % (name, alias)))
        return result

    current_keys = keys(current)
    desired_keys = keys(desired)
    return len(desired_keys - current_keys), len(current_keys - desired_keys)


__all__ = [
    "ADMIN_READY_TIMEOUT_SECONDS",
    "DEFAULT_ADMIN_HOST",
    "DEFAULT_ADMIN_PORT",
    "RECONCILE_ATTEMPTS",
    "SERVING_READY_TIMEOUT_SECONDS",
    "SparkrouteConfigError",
    "SparkrouteEngine",
]
