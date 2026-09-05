"""Pinned third-party tooling that sparkrun installs onto cluster hosts.

Everything here describes software fetched from the internet and executed on
someone else's machines, so it lives in one greppable place rather than inline
at the point of use.  Two call sites reach for ``uv`` today — the model ensure
scripts (``scripts/model_sync*.sh``, when no HuggingFace client is present) and
the ``uv-venv`` builder — and both previously ran::

    curl -LsSf https://astral.sh/uv/install.sh | sh

which is whatever Astral published this morning, fetched onto every node, with
no record of what was installed and nothing to bump when it needs changing.

Three properties are deliberate:

- **A pin, not a floor.** The URL names an exact version, so two hosts
  provisioned a month apart get the same binary and a `uv` release cannot
  change a launch under you.  A host that already has *any* ``uv`` keeps it —
  this governs what sparkrun *installs*, never what it demands.
- **Version-only, for now.** The pinned URL removes the "latest silently
  changed" risk, which is the one that actually bit.  A digest is strictly
  stronger and there is a slot for it below; it is a second thing to keep in
  step, so it is a deliberate later step rather than a half-done one now.
- **Not user-configurable.** A supply-chain pin is not a preference. Someone
  who wants a different ``uv`` installs it on the hosts, which this respects.

Bumping: change :data:`UV_VERSION`, confirm ``https://astral.sh/uv/<v>/install.sh``
resolves, run the suite.  Prefer a release with a week or two of soak over the
newest one — this is installed across a whole cluster.
"""

from __future__ import annotations

#: Pinned ``uv`` release.  See the module docstring before changing this.
UV_VERSION = "0.12.6"

#: Astral's versioned standalone installer.  The unversioned
#: ``https://astral.sh/uv/install.sh`` tracks latest and must not be used.
UV_INSTALL_URL = "https://astral.sh/uv/%s/install.sh" % UV_VERSION

#: Placeholder for a future integrity check on the installer script.  Left
#: unset rather than omitted so the decision is visible: see the module
#: docstring on why the pin ships before the digest.
UV_INSTALL_SHA256: str | None = None

#: Where Astral's installer puts the binary.  Needed because a script that just
#: installed ``uv`` has to find it without a fresh login shell.
UV_INSTALL_BIN_DIR = "$HOME/.local/bin"


def uv_pip_spec() -> str:
    """Return the pinned requirement specifier for ``pip install``.

    The ``uv-venv`` builder prefers a wheel from the host's index before
    falling back to the installer, and that path has to honor the same pin.
    """
    return "uv==%s" % UV_VERSION
