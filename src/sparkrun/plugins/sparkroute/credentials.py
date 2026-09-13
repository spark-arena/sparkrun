# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Owner-only credentials for Sparkrun's local SparkRoute integration.

The OSS integration uses live token files: SparkRoute rereads the admin token
for every request, so an atomic create/replace/remove enables, rotates, or
disables admin authentication without restarting the listener. The older
managed-credential helpers remain for migration compatibility, but normal
Sparkrun startup no longer needs a credential database or a subprocess just to
secure the local gateway.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import subprocess
import tempfile
from pathlib import Path

from sparkrun.proxy._supervisor import _restrict_dir_permissions, _restrict_file_permissions

logger = logging.getLogger(__name__)

#: Roles sparkrun needs: replace its own managed set, and read the serving
#: revision plus the content-free served-model list that backs
#: ``sparkrun proxy models``.
RECONCILER_ROLES = "config_reconcile:sparkrun,status_read"

RECONCILER_PRINCIPAL = "sparkrun-reconciler"

#: Roles for a *human* operator signing into the gateway's admin UI.
#:
#: Deliberately different from the reconciler's: sparkrun's own credential is
#: scoped to its managed set and cannot read the operator's, so handing it to a
#: browser would show a half-empty console. An operator credential can read and
#: write the operator set — which is why issuing one is an explicit act rather
#: than something ``proxy start`` does on its own.
OPERATOR_ROLES = "config_read,config_write,status_read,credentials_read,credentials_write"

OPERATOR_PRINCIPAL = "sparkrun-operator"

CREATE_TIMEOUT_SECONDS = 30


class CredentialError(RuntimeError):
    """Provisioning or reading the reconciler credential failed."""


class ReconcilerCredential:
    """The bearer secret and the databases it lives beside."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.credentials_db = self.state_dir / "client-credentials.db"
        self.config_db = self.state_dir / "config.db"
        self.secret_file = self.state_dir / "reconciler.secret"
        self.admin_secret_file = self.state_dir / "admin.secret"
        self.admin_token_file = self.state_dir / "admin-token.secret"
        self.caller_token_file = self.state_dir / "caller-token.secret"

    # -- Secret ------------------------------------------------------------

    def read_secret(self) -> str | None:
        """Return the stored secret, or ``None`` when not provisioned."""
        if not self.secret_file.is_file():
            return None
        try:
            return self.secret_file.read_text().strip() or None
        except OSError:
            logger.debug("Could not read the reconciler secret", exc_info=True)
            return None

    def _store_secret(self, secret: str, path: Path | None = None) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        _restrict_dir_permissions(self.state_dir)
        # Create restricted, then write: a create-then-chmod would leave the
        # secret briefly readable.
        path = path or self.secret_file
        path.touch(mode=0o600, exist_ok=True)
        _restrict_file_permissions(path)
        path.write_text(secret)

    def read_admin_secret(self) -> str | None:
        """Return Sparkrun's locally stored UI/admin token, if provisioned."""
        if not self.admin_secret_file.is_file():
            return None
        try:
            return self.admin_secret_file.read_text().strip() or None
        except OSError:
            logger.debug("Could not read the admin secret", exc_info=True)
            return None

    def read_admin_token(self) -> str | None:
        """Return the live admin bearer token, or None when auth is open."""
        return self._read_live_token(self.admin_token_file, "admin")

    def ensure_admin_token(self, preferred: str | None = None, *, force: bool = False) -> str:
        """Return the live admin token, creating or rotating it atomically."""
        if not force:
            existing = self.read_admin_token()
            if existing:
                return existing
        token = preferred or ("sr_admin_v1." + secrets.token_urlsafe(32))
        self._store_live_token(self.admin_token_file, token)
        return token

    def clear_admin_token(self) -> None:
        """Open the admin API immediately by atomically removing its token."""
        self._remove_live_token(self.admin_token_file, "admin")

    def set_caller_token(self, token: str | None) -> None:
        """Set or remove the data-plane token used for the next gateway run."""
        if token:
            self._store_live_token(self.caller_token_file, token)
        else:
            self._remove_live_token(self.caller_token_file, "caller")

    def _read_live_token(self, path: Path, label: str) -> str | None:
        try:
            raw = path.read_text()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise CredentialError("Could not read the gateway %s token: %s" % (label, exc)) from exc
        token = raw.strip()
        if not token:
            return None
        _validate_live_token(token, label)
        return token

    def _store_live_token(self, path: Path, token: str) -> None:
        _validate_live_token(token, path.name)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        _restrict_dir_permissions(self.state_dir)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % path.name, dir=self.state_dir)
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            stream = os.fdopen(descriptor, "w")
            descriptor = -1
            with stream:
                stream.write(token + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _restrict_file_permissions(path)
        except OSError as exc:
            raise CredentialError("Could not store the gateway token: %s" % exc) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _remove_live_token(path: Path, label: str) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise CredentialError("Could not remove the gateway %s token: %s" % (label, exc)) from exc

    # -- Provisioning ------------------------------------------------------

    def ensure(self, binary: Path, *, force: bool = False) -> str:
        """Return sparkrun's bearer secret, minting one if needed.

        Args:
            binary: Verified SparkRoute binary.
            force: Mint a replacement even when one is stored (rotation).

        Raises:
            CredentialError: The credential command failed or returned no
                secret.
        """
        if not force:
            existing = self.read_secret()
            if existing:
                return existing

        secret = self._create(binary, name=RECONCILER_PRINCIPAL, principal=RECONCILER_PRINCIPAL, roles=RECONCILER_ROLES)
        self._store_secret(secret)
        logger.info("Provisioned the gateway reconciler credential (roles: %s)", RECONCILER_ROLES)
        return secret

    def ensure_admin(self, binary: Path, *, force: bool = False) -> str:
        """Return the single stored UI/admin token, creating or rotating it.

        When upgrading from the former show-once behavior, ``ensure`` adopts
        and rotates the single active ``sparkrun-operator`` credential in the
        database. That invalidates the unrecoverable old token instead of
        silently leaving two administrators active.
        """
        existing = self.read_admin_secret()
        if existing and not force:
            return existing

        if existing and force:
            credential_id = _credential_id(existing)
            if not credential_id:
                raise CredentialError("Stored gateway admin token has an invalid format")
            secret = self._rotate(binary, credential_id)
        else:
            secret = self._ensure_operator(binary)
        self._store_secret(secret, self.admin_secret_file)
        logger.info("Provisioned the gateway admin credential (roles: %s)", OPERATOR_ROLES)
        return secret

    def issue_operator_credential(self, binary: Path, *, name: str = OPERATOR_PRINCIPAL) -> str:
        """Compatibility alias returning the single stored admin token."""
        if name != OPERATOR_PRINCIPAL:
            raise CredentialError("Sparkrun manages exactly one admin credential")
        return self.ensure_admin(binary)

    def _ensure_operator(self, binary: Path) -> str:
        return self._run_credential_command(
            binary,
            "ensure",
            "-name",
            OPERATOR_PRINCIPAL,
            "-principal-id",
            OPERATOR_PRINCIPAL,
            "-roles",
            OPERATOR_ROLES,
        )

    def _rotate(self, binary: Path, credential_id: str) -> str:
        return self._run_credential_command(binary, "rotate", "-credential-id", credential_id)

    def _create(self, binary: Path, *, name: str, principal: str, roles: str) -> str:
        """Run the offline credential command and return its one-time secret."""
        return self._run_credential_command(
            binary,
            "create",
            "-name",
            name,
            "-principal-id",
            principal,
            "-roles",
            roles,
        )

    def _run_credential_command(self, binary: Path, verb: str, *arguments: str) -> str:
        """Run an offline credential mutation and return its one-time secret."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        _restrict_dir_permissions(self.state_dir)

        cmd = [
            str(binary),
            "client-credentials",
            verb,
            "-database",
            str(self.credentials_db),
            *arguments,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=CREATE_TIMEOUT_SECONDS)  # noqa: S603 - argv list, no shell
        except (OSError, subprocess.SubprocessError) as exc:
            raise CredentialError("Could not run the gateway credential command: %s" % exc) from exc
        if result.returncode != 0:
            # stderr describes the *command*; the secret only ever reaches
            # stdout on success, so this is safe to surface.
            raise CredentialError(
                "Gateway credential provisioning failed: %s" % ((result.stderr or "").strip() or "exit code %d" % result.returncode)
            )
        secret = _extract_secret(result.stdout)
        if not secret:
            raise CredentialError("Gateway credential command returned no secret")
        return secret

    def reset_configuration(self) -> list[Path]:
        """Delete the configuration database and its SQLite sidecars.

        Only safe with the gateway stopped, which is the caller's
        responsibility. The credential database is deliberately left alone: it
        is independent, and removing it would strand the stored secret.

        Returns:
            The paths actually removed.
        """
        removed: list[Path] = []
        for path in (self.config_db, Path(str(self.config_db) + "-wal"), Path(str(self.config_db) + "-shm")):
            try:
                if path.exists():
                    path.unlink()
                    removed.append(path)
            except OSError:
                logger.warning("Could not remove %s", path, exc_info=True)
        return removed


MAX_LIVE_TOKEN_BYTES = 4096


def _validate_live_token(token: str, label: str) -> None:
    if not token or not token.isascii() or any(character.isspace() or character == "\x00" for character in token):
        raise CredentialError("Gateway %s token must be non-empty ASCII without whitespace" % label)
    if len(token.encode()) > MAX_LIVE_TOKEN_BYTES:
        raise CredentialError("Gateway %s token is too large" % label)


#: JSON field carrying the one-time secret (``IssuedCredential.APIKey``).
SECRET_FIELD = "api_key"


def _extract_secret(stdout: str) -> str:
    """Pull the one-time secret out of the credential command's JSON.

    Reads exactly :data:`SECRET_FIELD` rather than trying a list of plausible
    names: guessing would silently pick the wrong field if the payload ever
    gained another string-valued one, and a credential is the last place to
    accept a near-miss.
    """
    try:
        payload = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        logger.debug("Credential command output was not JSON")
        return ""
    if not isinstance(payload, dict):
        return ""
    value = payload.get(SECRET_FIELD)
    return value if isinstance(value, str) else ""


def _credential_id(api_key: str) -> str:
    """Extract the non-secret credential ID from a SparkRoute bearer token."""
    parts = api_key.split(".", 2)
    if len(parts) != 3 or parts[0] != "llmgw_v1" or not parts[1].startswith("cc_") or not parts[2]:
        return ""
    return parts[1]


__all__ = [
    "CREATE_TIMEOUT_SECONDS",
    "OPERATOR_PRINCIPAL",
    "OPERATOR_ROLES",
    "RECONCILER_PRINCIPAL",
    "RECONCILER_ROLES",
    "CredentialError",
    "ReconcilerCredential",
]
