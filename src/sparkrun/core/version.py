"""User-facing version rendering and installed-build identity.

Kept separate from ``sparkrun.__version__`` (the raw package metadata version):
the display string carries a channel suffix and git commit, while the metadata
version stays clean for machine comparison.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution, version

from sparkrun.core.channels import CHANNEL_STABLE, channel_suffix, normalize_channel


def installed_commit() -> str | None:
    """Return the git commit the installed sparkrun was built from, if any.

    Reads PEP 610 ``direct_url.json`` (written by uv/pip for VCS installs) and
    returns ``vcs_info.commit_id``. Returns ``None`` for PyPI installs or when
    the metadata is missing/malformed. Never shells out to git — installed uv
    tool environments are not guaranteed to contain a checkout.
    """
    try:
        raw = distribution("sparkrun").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not raw:
        return None
    try:
        vcs_info = json.loads(raw).get("vcs_info") or {}
    except (ValueError, TypeError):
        return None
    commit = vcs_info.get("commit_id")
    return commit if isinstance(commit, str) and commit else None


def base_version() -> str:
    """Return the raw package metadata version (no channel suffix)."""
    try:
        return version("sparkrun")
    except PackageNotFoundError:
        return "0.0.0-dev"


def installed_identity() -> tuple[str, str | None]:
    """Return ``(base_version, commit_id|None)`` for the installed sparkrun."""
    return base_version(), installed_commit()


def display_version(config=None, base: str | None = None) -> str:
    """Return the user-facing version string for the configured channel.

    Stable returns the base version unchanged. Beta/alpha append ``-beta`` /
    ``-alpha`` plus ``+g<short-sha>`` when the installed git commit is known,
    falling back to the channel-only suffix otherwise.
    """
    base = base if base is not None else base_version()
    channel = CHANNEL_STABLE
    if config is not None:
        try:
            channel = normalize_channel(config.self_update_channel)
        except Exception:  # pragma: no cover — display must never crash the CLI
            channel = CHANNEL_STABLE
    suffix = channel_suffix(channel)
    if not suffix:
        return base
    commit = installed_commit()
    if commit:
        return "%s%s+g%s" % (base, suffix, commit[:7])
    return "%s%s" % (base, suffix)
