"""User-facing version rendering and installed-build identity.

Kept separate from ``sparkrun.__version__`` (the raw package metadata version):
the display string carries a channel suffix and git commit, while the metadata
version stays clean for machine comparison.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, distribution, version
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version

from sparkrun.core.application_profile import get_application_profile
from sparkrun.core.channels import CHANNEL_STABLE, channel_suffix, normalize_channel


if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe


def parse_min_sparkrun_version(value: object) -> str | None:
    """Validate an optional minimum core version, preserving its spelling."""
    if value is None:
        return None
    message = 'min_sparkrun_version must be a version string such as "0.4.0" (not a version range)'
    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    try:
        Version(value)
    except InvalidVersion as error:
        raise ValueError(message) from error
    return value.strip()


def require_recipe_version(recipe: Recipe) -> None:
    """Reject incompatible execution, while allowing recipes to be inspected.

    Compare the core package metadata, not a branded application version or the
    display channel suffix. Recheck supplied/serialized plans at execution.
    """
    minimum = parse_min_sparkrun_version(getattr(recipe, "min_sparkrun_version", None))
    if minimum is None:
        return
    installed = base_version("sparkrun")
    upgrade = "Upgrade to Sparkrun %s or newer with `%s setup update`." % (minimum, get_application_profile().command)
    try:
        current = Version(installed)
    except InvalidVersion as error:
        raise ValueError(
            "Cannot verify min_sparkrun_version %s: installed Sparkrun version %r is invalid. %s" % (minimum, installed, upgrade)
        ) from error
    if current < Version(minimum):
        raise ValueError(
            "Recipe %r requires Sparkrun >= %s (min_sparkrun_version); installed version is %s. %s"
            % (recipe.name, minimum, installed, upgrade)
        )


def installed_commit(package: str | None = None) -> str | None:
    """Return the git commit the installed sparkrun was built from, if any.

    Reads PEP 610 ``direct_url.json`` (written by uv/pip for VCS installs) and
    returns ``vcs_info.commit_id``. Returns ``None`` for PyPI installs or when
    the metadata is missing/malformed. Never shells out to git — installed uv
    tool environments are not guaranteed to contain a checkout.
    """
    try:
        raw = distribution(package or get_application_profile().package).read_text("direct_url.json")
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


def base_version(package: str | None = None) -> str:
    """Return the raw package metadata version (no channel suffix)."""
    try:
        return version(package or get_application_profile().package)
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


def version_diagnostics(config) -> dict:
    from sparkrun.core.plugin_inventory import list_plugins

    base, commit = installed_identity()
    profile = get_application_profile()
    return {
        "version": base,
        "commit": commit,
        "channel": config.self_update_channel,
        "distribution": {"id": profile.id, "package": profile.package, "version": base, "commit": commit},
        "core": {"package": "sparkrun", "version": base_version("sparkrun"), "commit": installed_commit("sparkrun")},
        "plugins": [p.to_dict() for p in list_plugins(config) if p.loaded],
    }
