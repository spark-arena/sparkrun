"""Proxy configuration management.

Reads/writes ``~/.config/sparkrun/proxy.yaml`` for proxy settings,
model aliases, and (future) default recipe mappings.

**Two processes write this file.** The auto-discover daemon re-reads it every
sweep, and a ``sparkrun proxy`` command may save an alias or listener setting
at any moment.  A whole-document last-writer-wins save therefore silently
discards the other's change, so :meth:`ProxyConfig.save` locks, re-reads, and
merges only the sections *this* instance modified.
"""

from __future__ import annotations

import copy
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from sparkrun.proxy import (
    DEFAULT_DISCOVER_INTERVAL,
    DEFAULT_DISCOVER_REMOVAL_GRACE_SWEEPS,
    DEFAULT_MASTER_KEY,
    DEFAULT_PROXY_HOST,
    DEFAULT_PROXY_PORT,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised via a masked-import test
    # POSIX-only. This module is reached from SparkrunContext, i.e. essentially
    # every invocation, so a hard import would make sparkrun unimportable on a
    # Windows control node. See _write_lock for what degrades.
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_UNCHANGED = object()
_DELETE = object()


class ProxyConfig:
    """Manages proxy configuration stored in ``proxy.yaml``.

    The canonical access path is :attr:`sparkrun.core.context.SparkrunContext.proxy_config`
    (or :meth:`sparkrun.core.config.SparkrunConfig.get_proxy_config` for direct
    callers without a ``sctx``).  Direct construction is reserved for tests
    that need to point at a temporary config path.
    """

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            from sparkrun.core.config import DEFAULT_CONFIG_DIR

            config_path = DEFAULT_CONFIG_DIR / "proxy.yaml"
        self.config_path = config_path
        self._data: dict[str, Any] = {}
        self._reset_pending()
        self._load()

    def _reset_pending(self) -> None:
        """Forget which sections this instance has modified."""
        self._pending_proxy: dict[str, Any] = {}
        self._pending_aliases: dict[str, Any] = {}
        self._pending_bindings: object = _UNCHANGED
        self._pending_discovered_models: object = _UNCHANGED

    def _read(self) -> dict[str, Any]:
        """Parse the file as it is on disk right now ({} when absent/bad)."""
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path) as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            logger.debug("Failed to load proxy config: %s", self.config_path, exc_info=True)
            return {}

    def _load(self) -> None:
        self._data = self._read()
        self._reset_pending()

    @property
    def _lock_path(self) -> Path:
        return Path(str(self.config_path) + ".lock")

    @contextmanager
    def _write_lock(self):
        """Serialize read-modify-write transactions across processes.

        A stable sidecar is locked rather than ``proxy.yaml`` itself, because
        the config is replaced by rename — a lock held on the old inode would
        not be seen by the next writer.

        Degrades to no locking where ``fcntl`` is unavailable (Windows), which
        is **strictly weaker**: concurrent writers can still interleave their
        read-modify-write.  Correct for the single-writer case a Windows
        control node has today — the auto-discover daemon is a POSIX-only fork
        path — and the atomic replace below means the file is never truncated
        or half-written either way.
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None:
            yield
            return
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self._lock_path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "r+") as lock_file:
                descriptor = -1
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                yield
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _write_atomic(self, data: dict[str, Any]) -> None:
        """Replace the config in one step, never leaving a partial file."""
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".%s." % self.config_path.name,
            dir=self.config_path.parent,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                descriptor = -1
                yaml.safe_dump(data, stream, default_flow_style=False, sort_keys=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.config_path)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def save(self) -> None:
        """Atomically merge this instance's pending changes into the file.

        Writers lock a stable sidecar, re-read the newest document, and apply
        only the sections changed through this instance.  An auto-discovery
        sweep therefore cannot overwrite an alias or listener setting saved
        concurrently by a CLI process, and vice versa.

        With no pending changes this still rewrites the document held in
        memory, preserving the historical whole-file semantics for a caller
        that mutated ``_data`` directly.
        """
        has_pending = bool(self._pending_proxy or self._pending_aliases)
        has_pending = has_pending or self._pending_bindings is not _UNCHANGED
        has_pending = has_pending or self._pending_discovered_models is not _UNCHANGED

        with self._write_lock():
            latest = self._read() if has_pending else copy.deepcopy(self._data)

            if self._pending_proxy:
                latest.setdefault("proxy", {}).update(copy.deepcopy(self._pending_proxy))

            if self._pending_aliases:
                aliases = latest.setdefault("aliases", {})
                for alias, target in self._pending_aliases.items():
                    if target is _DELETE:
                        aliases.pop(alias, None)
                    else:
                        aliases[alias] = target
                if not aliases:
                    latest.pop("aliases", None)

            if self._pending_bindings is not _UNCHANGED:
                latest["bindings"] = copy.deepcopy(self._pending_bindings)
            if self._pending_discovered_models is not _UNCHANGED:
                models = copy.deepcopy(self._pending_discovered_models)
                if models:
                    latest["discovered_models"] = models
                else:
                    latest.pop("discovered_models", None)

            self._write_atomic(latest)

        self._data = latest
        self._reset_pending()
        logger.debug("Saved proxy config to %s", self.config_path)

    # -- Proxy settings --

    @property
    def port(self) -> int:
        return int(self._data.get("proxy", {}).get("port", DEFAULT_PROXY_PORT))

    @property
    def host(self) -> str:
        return str(self._data.get("proxy", {}).get("host", DEFAULT_PROXY_HOST))

    @property
    def host_configured(self) -> bool:
        """True when a bind host has been explicitly persisted.

        Distinguishes "user chose a bind host" from "fell back to the legacy
        ``0.0.0.0`` default".  When False, the proxy keeps the legacy 0.0.0.0
        bind for backward compatibility but emits a loud security warning.
        """
        return "host" in self._data.get("proxy", {})

    @property
    def gateway(self) -> str | None:
        """Explicitly pinned gateway implementation, or ``None``.

        The selector half of :mod:`sparkrun.proxy.gateway` — availability is a
        feature flag in ``config.yaml``, *which* gateway to use is this key.
        ``None`` (the normal case) means "resolve the default"; pinning a
        gateway that is disabled is an error rather than a silent fallback.
        """
        val = self._data.get("proxy", {}).get("gateway")
        return str(val) if val else None

    @property
    def master_key(self) -> str | None:
        val = self._data.get("proxy", {}).get("master_key", DEFAULT_MASTER_KEY)
        return str(val) if val is not None else None

    @property
    def auto_discover(self) -> bool:
        return bool(self._data.get("proxy", {}).get("auto_discover", True))

    @property
    def discover_interval(self) -> int:
        return int(self._data.get("proxy", {}).get("discover_interval", DEFAULT_DISCOVER_INTERVAL))

    @property
    def discover_removal_grace_sweeps(self) -> int:
        """Consecutive absent discovery sweeps required before removal.

        Clamped to at least 1 — zero would mean "remove before you have looked".
        """
        return max(
            1,
            int(
                self._data.get("proxy", {}).get(
                    "discover_removal_grace_sweeps",
                    DEFAULT_DISCOVER_REMOVAL_GRACE_SWEEPS,
                )
            ),
        )

    @property
    def enable_ui(self) -> bool:
        """True when an obsolete ``enable_ui`` is still set in proxy.yaml.

        The LiteLLM ``/ui`` is not supported — it is DB-backed and its
        ``schema.prisma`` requires PostgreSQL.  This survives only so
        ``proxy start`` can warn about (and ignore) a stale key rather than
        silently dropping a setting the user believes is active.
        """
        return bool(self._data.get("proxy", {}).get("enable_ui", False))

    def set_proxy(self, **kwargs: Any) -> None:
        """Update proxy settings (port, host, master_key, etc.).

        Recorded as pending so :meth:`save` merges exactly these keys into the
        newest document rather than replacing the whole ``proxy`` section.
        """
        proxy = self._data.setdefault("proxy", {})
        proxy.update(kwargs)
        self._pending_proxy.update(copy.deepcopy(kwargs))

    # -- Alias management --

    @property
    def aliases(self) -> dict[str, str]:
        """Return alias -> model group mapping."""
        return dict(self._data.get("aliases", {}))

    def add_alias(self, alias: str, target: str) -> None:
        """Add or update an alias mapping."""
        aliases = self._data.setdefault("aliases", {})
        aliases[alias] = target
        self._pending_aliases[alias] = target

    def remove_alias(self, alias: str) -> bool:
        """Remove an alias. Returns True if it existed."""
        aliases = self._data.get("aliases", {})
        if alias in aliases:
            del aliases[alias]
            # Recorded as an explicit deletion rather than by absence: save()
            # merges into the newest document, where the alias may well still
            # be present, and "not in my copy" is not an instruction to delete.
            self._pending_aliases[alias] = _DELETE
            return True
        return False

    def list_aliases(self) -> list[tuple[str, str]]:
        """Return list of (alias, target) pairs."""
        return list(self._data.get("aliases", {}).items())

    # -- Default recipes (schema only, not wired up) --

    @property
    def default_recipes(self) -> dict[str, dict]:
        """Return default recipe mappings (future use)."""
        return dict(self._data.get("default_recipes", {}))

    @property
    def gateway_config(self) -> str | None:
        """Path to an operator-authored gateway document, or ``None``.

        For the ``sparkroute`` backend this is a **one-time bootstrap seed**,
        not a document sparkrun keeps running: the gateway rejects ``-config``
        in SQLite mode, so it is passed as ``-config-bootstrap`` and imported
        into the ``operator`` managed set only when the configuration database
        is empty.  Ignored by LiteLLM.
        """
        val = self._data.get("proxy", {}).get("gateway_config")
        return str(val) if val else None

    @property
    def gateway_admin_port(self) -> int:
        """Port for an explicitly split gateway admin listener."""
        return int(self._data.get("proxy", {}).get("gateway_admin_port", 8081))

    @property
    def gateway_admin_host(self) -> str:
        """Bind address for an explicitly split gateway admin listener.

        Loopback by default, because this listener accepts a credential that
        can rewrite the served model set.  Set it to ``0.0.0.0`` or a specific
        address to reach the admin console from another machine.

        Widening it is defensible rather than reckless: the listener is
        authenticated, so exposure costs a credential rather than the whole
        config.  It stays opt-in because a default nobody chose should never be
        the reason something is on the network.
        """
        return str(self._data.get("proxy", {}).get("gateway_admin_host", "127.0.0.1"))

    @property
    def gateway_admin_configured(self) -> bool:
        """Whether admin was explicitly isolated from the data listener.

        With neither key present SparkRoute mounts ``/admin`` and its admin API
        routes on the data address. Setting either legacy host/port key opts
        into a dedicated listener; the other component keeps its safe legacy
        default.
        """
        proxy = self._data.get("proxy", {})
        return "gateway_admin_host" in proxy or "gateway_admin_port" in proxy

    @property
    def capability_policy(self) -> str:
        """``permissive`` (default) or ``strict`` capability generation.

        Under ``permissive`` sparkrun emits ``capability_policy.unknown =
        "try"`` on generated deployments: an undeclared optional capability is
        attempted and the upstream error decides.  That is right while sparkrun
        cannot determine model capabilities — the alternative is every recipe
        silently losing tool-calling until someone declares it.  Capabilities
        the gateway treats as fail-closed (embeddings, files, responses, …)
        always require a positive declaration regardless.
        """
        value = str(self._data.get("proxy", {}).get("capability_policy", "permissive")).strip().lower()
        return value if value in ("permissive", "strict") else "permissive"

    @property
    def gateway_allow_insecure_admin_nonloopback(self) -> bool:
        """Persisted acknowledgment of unauthenticated non-loopback admin.

        Required for a non-loopback token-file listener because its token can
        be cleared while the process remains running. Sparkrun implies it for
        public binds and still warns loudly whenever the live token is absent.
        """
        return bool(self._data.get("proxy", {}).get("gateway_allow_insecure_admin_nonloopback", False))

    @property
    def bindings(self) -> list[dict[str, Any]]:
        """Recipe bindings under ``bindings:`` — sparkrun's desired state.

        A **catalog**, deliberately not "whatever is running now": generating
        deployments from live discovery would deadlock, since a workload going
        down would delete the very deployment needed to activate it again.
        """
        value = self._data.get("bindings", [])
        return [dict(entry) for entry in value if isinstance(entry, dict)]

    def set_bindings(self, bindings: list[dict[str, Any]]) -> None:
        """Replace the binding list."""
        self._data["bindings"] = [dict(entry) for entry in bindings]
        self._pending_bindings = copy.deepcopy(self._data["bindings"])

    @property
    def discovered_models(self) -> list[str]:
        """Models imported by the last explicit discovery sync.

        These are deliberately separate from :attr:`bindings`. A binding is
        durable, activatable desired state; a discovered model is only a route
        to a workload observed running during ``proxy sync``. A later explicit
        sync may therefore remove a discovered model without deleting the
        cold-start route for a bound recipe.
        """
        value = self._data.get("discovered_models", [])
        if not isinstance(value, list):
            return []
        return sorted({str(model).strip() for model in value if str(model).strip()})

    def set_discovered_models(self, models: list[str]) -> None:
        """Replace the explicit-discovery snapshot deterministically."""
        normalized = sorted({str(model).strip() for model in models if str(model).strip()})
        if normalized:
            self._data["discovered_models"] = normalized
        else:
            self._data.pop("discovered_models", None)
        self._pending_discovered_models = normalized
