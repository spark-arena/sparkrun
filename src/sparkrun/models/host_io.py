"""Owned host sessions and cooperative model-cache publication leases."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import PurePosixPath
import uuid

from sparkrun.models.artifacts import ModelArtifactError, ModelArtifactManifest
from sparkrun.models.manifest import STATE_DIRECTORY
from sparkrun.models.observation import observation_script, parse_observations, path_assignment
from sparkrun.transports.session import HostCommandResult, HostSession, HostSessionError, SshHostSession, StreamingHostSession
from sparkrun.utils.shell import quote


class ModelHostIO:
    def __init__(self, ssh_kwargs: dict | None = None, *, session: HostSession | None = None):
        kw = ssh_kwargs or {}
        self.session = session or SshHostSession(**{key: kw[key] for key in ("ssh_user", "ssh_key", "ssh_options") if key in kw})
        self.local = SshHostSession()
        self.owns_session = session is None
        self._stack = ExitStack()
        self._operation = uuid.uuid4().hex
        self._locks: set[tuple[str | None, str]] = set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self.local.close()
        if self.owns_session:
            self.session.close()

    def execute(self, host: str | None, script: str, *, timeout: float = 30) -> HostCommandResult:
        session = self.local if host is None else self.session
        try:
            return session.execute(host or "localhost", ["bash", "-s"], input_data=script.encode(), timeout=timeout)
        except (HostSessionError, OSError) as exc:
            raise ModelArtifactError("model host session unavailable on " + (host or "control")) from exc

    def checked(self, host: str | None, script: str, *, timeout: float = 30) -> str:
        result = self.execute(host, script, timeout=timeout)
        if result.returncode:
            raise ModelArtifactError(
                "model operation on %s failed (rc=%d): %s"
                % (host or "control", result.returncode, result.stderr.decode(errors="replace")[-1500:])
            )
        return result.stdout.decode()

    def root(self, host: str | None, cache: str) -> str:
        text = self.checked(host, path_assignment("root", cache) + 'realpath -m -- "$root"\n')
        # realpath prints one terminating newline; keep newlines inside paths.
        path = text.removesuffix("\n")
        if not PurePosixPath(path).is_absolute() or "\x00" in path:
            raise ModelArtifactError("host returned an invalid model cache path")
        return path

    def observe(self, host: str | None, manifest: ModelArtifactManifest, cache: str, *, checksums: bool = False):
        result = self.execute(
            host,
            observation_script(manifest, str(PurePosixPath(cache) / manifest.relative_snapshot), checksums=checksums),
            timeout=1800 if checksums else 60,
        )
        return parse_observations(
            host or "control", manifest, result.stdout.decode(errors="replace"), returncode=result.returncode, checksums=checksums
        )

    def read(self, host: str | None, path: str) -> str | None:
        result = self.execute(host, path_assignment("p", path) + 'if [ ! -e "$p" ]; then exit 3; fi\ncat -- "$p"\n')
        if result.returncode == 3:
            return None
        if result.returncode:
            raise ModelArtifactError("could not read model manifest on " + (host or "control"))
        return result.stdout.decode()

    def write(self, host: str | None, path: str, contents: str) -> None:
        # Exclusive temp + atomic publication. stdin remains the script, so the
        # payload is safely quoted shell data rather than an interpolated heredoc.
        self.checked(
            host,
            "set -eu\n" + path_assignment("p", path) + 'mkdir -p -- "$(dirname -- "$p")"\n'
            'tmp=$(mktemp "$(dirname -- "$p")/.manifest-XXXXXXXX")\n'
            "trap 'rm -f -- \"$tmp\"' EXIT\n"
            f'printf %s {quote(contents)} > "$tmp"\n'
            'mv -f -- "$tmp" "$p"\n',
        )

    def lock(self, host: str | None, cache: str, repo: str) -> None:
        """One lease per shared repo write domain, held across provider callbacks.

        The nonce in the flock file detects a lease we already hold through
        another host/mount alias. Never unlink flock files: that would allow
        two writers to lock different inodes under the same pathname.
        """
        path = str(PurePosixPath(cache) / STATE_DIRECTORY / (repo.replace("/", "--") + ".lock"))
        key = host, path
        if key in self._locks:
            return
        existing = self.read(host, path)
        if existing == self._operation:
            self._locks.add(key)
            return
        session = self.local if host is None else self.session
        if not isinstance(session, StreamingHostSession):
            raise ModelArtifactError("model publication requires a host session with supervised streaming leases")
        inner = f'printf %s {quote(self._operation)} > {quote(path)}; printf "LOCKED\\n"; cat >/dev/null'
        script = "set -eu\nmkdir -p -- %s\nexec flock -w 30 %s bash -c %s\n" % (
            quote(str(PurePosixPath(path).parent)),
            quote(path),
            quote(inner),
        )
        process = session.open_process(host or "localhost", ["bash", "-c", script])
        self._stack.callback(process.close)
        if process.stdout.readline() != b"LOCKED\n":
            raise ModelArtifactError("could not acquire model cache publication lease on " + (host or "control"))
        self._locks.add(key)
