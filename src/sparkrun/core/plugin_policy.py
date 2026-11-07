"""One loading policy for bundled, directory, managed and installed plugins.

Plugin state is independent of the capability flags registered by a plugin.
Old whole-plugin feature gates and installed ``integrations`` settings are
migration inputs; an explicit ``plugins.enabled`` override takes precedence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sparkrun.core.plugin_identity import normalize_plugin_id


class PluginSource(StrEnum):
    """Where a plugin comes from, as far as loading policy is concerned.

    ``EXTERNAL`` covers configured directories and managed plugins: both need
    ``core.external_plugins``. ``INSTALLED`` entry points are off unless
    selected. ``DEFAULT_MANAGED`` is a persisted source-trusted default; it
    honors explicit global opt-outs without enabling arbitrary directories.
    An unknown spelling is an error rather than a silent bypass of those gates.
    """

    IN_TREE = "in-tree"
    EXTERNAL = "external"
    INSTALLED = "installed"
    DEFAULT_MANAGED = "default-managed"


def policy_source(kind: str) -> PluginSource:
    """Policy source for a registration identity's source kind.

    Programmatic registrations (``module``, ``application``) have no source
    gate of their own, which is the in-tree rule.
    """
    try:
        return PluginSource(kind)
    except ValueError:
        return PluginSource.IN_TREE


@dataclass(frozen=True)
class KnownPlugin:
    """What core knows about a first-party plugin before importing it."""

    module: str
    """Subpackage name under ``sparkrun.plugins``."""
    title: str
    legacy_flag: str | None = None
    """Whole-plugin feature flag that gated loading before ``plugins.enabled``."""
    flag_is_capability: bool = False
    """The legacy flag is now the plugin's own capability flag. It then gates
    loading only for an in-tree snapshot that declares no manifest (one that
    predates the split); for anything else a capability toggle would disable
    the plugin itself."""
    bundled: bool = False
    """Ships and is tested with this release."""
    requires_profile_api: bool = False
    """Releases exist that predate application-profile support, so under an
    alternate application profile the plugin must declare
    ``application_profile_api`` to load."""


#: The one table of first-party plugins, keyed by normalized plugin ID. The
#: in-tree loader's inventory and the legacy flag fallbacks derive from it.
KNOWN_PLUGINS: dict[str, KnownPlugin] = {
    "litellm": KnownPlugin("litellm", "LiteLLM", bundled=True),
    "sparkroute": KnownPlugin("sparkroute", "SparkRoute", "gateway.sparkroute", flag_is_capability=True, requires_profile_api=True),
    "coldsnap": KnownPlugin("coldsnap", "ColdSnap", "plugins.coldsnap"),
    "oci-relay": KnownPlugin("oci_relay", "OCI Relay", "plugins.oci_relay", bundled=True),
    "sparkarena": KnownPlugin("sparkarena", "Spark Arena", "integration.arena", bundled=True),
    "k8s": KnownPlugin("k8s", "Kubernetes", "integration.k8s", bundled=True),
    "lil": KnownPlugin("lil", "lil", "registry.lil", bundled=True),
    "connect": KnownPlugin("connect", "Connect", bundled=True),
}

#: Plugin ID -> whole-plugin flag still honored as a loading fallback for every source.
LEGACY_PLUGIN_FLAGS = {
    plugin_id: known.legacy_flag for plugin_id, known in KNOWN_PLUGINS.items() if known.legacy_flag and not known.flag_is_capability
}


def known_plugin(name: object) -> KnownPlugin | None:
    try:
        return KNOWN_PLUGINS.get(normalize_plugin_id(name))
    except ValueError:
        return None


def declared_application_profile_api(origin) -> int | None:
    """``application_profile_api`` from the manifest beside a plugin's entry module.

    Reads the source manifest (``plugin.toml``) or vendoring record
    (``VENDORED.toml``) in the package directory, without importing anything.
    Managed plugins carry the same value in their install metadata instead.
    """
    from sparkrun.core.plugin_metadata import sidecar_manifest

    value = sidecar_manifest(origin).get("application_profile_api")
    return value if type(value) is int else None


def application_profile_failure(plugin_id: object, declared_api: object, *, profile_id: str | None = None) -> str | None:
    """Why a plugin cannot load under the active application profile, or ``None``.

    A declared ``application_profile_api`` must equal the host's, for every
    plugin source. An undeclared one is profile-agnostic, except for a plugin
    whose older releases predate profile support (``requires_profile_api``):
    those must declare it before running under an alternate profile.
    """
    from sparkrun.core.application_profile import APPLICATION_PROFILE_API_VERSION, get_application_profile

    if declared_api is not None:
        if type(declared_api) is not int or declared_api != APPLICATION_PROFILE_API_VERSION:
            return "%s declares application profile API %r; this host implements %s" % (
                plugin_id,
                declared_api,
                APPLICATION_PROFILE_API_VERSION,
            )
        return None
    known = known_plugin(plugin_id)
    # Resolve the active profile only when it matters: reading it selects the
    # default profile, which must not happen before an application chooses one.
    if (
        known is not None
        and known.requires_profile_api
        and requires_profile_api_declaration(plugin_id, profile_id or get_application_profile().id)
    ):
        return (
            "The pinned %s plugin does not declare support for this host's application profile API; "
            "import a compatible plugin snapshot before using an alternate application profile" % known.title
        )
    return None


def requires_profile_api_declaration(plugin_id: object, profile_id: str) -> bool:
    """True when *plugin_id* must declare ``application_profile_api`` under *profile_id*."""
    known = known_plugin(plugin_id)
    return profile_id != "sparkrun" and known is not None and known.requires_profile_api


@dataclass(frozen=True)
class PluginState:
    enabled: bool
    source: str
    reason: str | None = None
    required: bool = False


def plugin_overrides(config) -> dict[str, bool]:
    raw = config.get("plugins.enabled", {})
    if not isinstance(raw, dict) or any(type(value) is not bool for value in raw.values()):
        raise ValueError("plugins.enabled must map plugin IDs to true or false")
    result = {}
    for name, value in raw.items():
        identifier = normalize_plugin_id(name)
        if identifier in result:
            raise ValueError("Repeated canonical plugin ID: %s" % identifier)
        result[identifier] = value
    return result


def plugin_state(
    name,
    config=None,
    *,
    source: PluginSource | str = PluginSource.IN_TREE,
    legacy_feature=None,
    env=None,
    channel=None,
    external_gate: bool = True,
) -> PluginState:
    """Resolve whether plugin *name* loads.

    ``external_gate=False`` is only for explicitly named directories (a
    programmatic ``load_external_plugins(paths=...)``), which bypass the
    ``core.external_plugins`` flag by contract.
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.features import _raw_feature_enabled, feature_source
    from sparkrun.core.external_plugins import _external_plugins_disabled
    import os

    source = PluginSource(source)
    config = config if config is not None else SparkrunConfig()
    env = os.environ if env is None else env
    name = normalize_plugin_id(name)
    profile = config.profile
    required = name in {normalize_plugin_id(item) for item in profile.required_integrations}
    selected = name in {normalize_plugin_id(item) for item in (*profile.integrations, *profile.required_integrations)}
    overrides = plugin_overrides(config)
    integrations = config.get("integrations", {})
    if not isinstance(integrations, dict) or any(type(value) is not bool for value in integrations.values()):
        raise ValueError("integrations must map plugin IDs to true or false")
    integrations = {normalize_plugin_id(key): value for key, value in integrations.items()}
    legacy = LEGACY_PLUGIN_FLAGS.get(name)
    # Older plugins can still declare one loading feature. New declarations
    # contain capabilities and are independent of this migration fallback.
    legacy = legacy or legacy_feature
    if name in overrides:
        enabled, origin = overrides[name], "config"
    elif name in integrations:
        enabled, origin = integrations[name], "integrations (legacy)"
    elif selected:
        enabled, origin = True, "distribution"
    elif source is PluginSource.DEFAULT_MANAGED and (
        not legacy or feature_source(legacy, config=config, env=env, channel=channel) in {"channel", "unset"}
    ):
        enabled, origin = True, "default plugin"
    elif legacy:
        enabled = _raw_feature_enabled(legacy, config=config, env=env, channel=channel)
        origin = "legacy feature: " + feature_source(legacy, config=config, env=env, channel=channel)
    else:
        enabled, origin = source is not PluginSource.INSTALLED, "default"
    reason = None if enabled else "plugin disabled"
    if (
        source is PluginSource.EXTERNAL
        and external_gate
        and (_external_plugins_disabled() or not _raw_feature_enabled("core.external_plugins", config=config, env=env))
    ):
        enabled, reason = False, "external plugin loading disabled"
    if source is PluginSource.INSTALLED and env.get("SPARKRUN_NO_INSTALLED_PLUGINS", "").lower() in {"1", "true", "yes"}:
        enabled, reason = False, "installed plugin loading disabled"
    if source is PluginSource.DEFAULT_MANAGED:
        # The release's scoped trust permits only this managed payload. An
        # explicit global opt-out still wins; never activate plugins.paths.
        explicit_gate = feature_source("core.external_plugins", config=config, env=env) not in {"channel", "unset"}
        if _external_plugins_disabled() or (explicit_gate and not _raw_feature_enabled("core.external_plugins", config=config, env=env)):
            enabled, reason = False, "external plugin loading disabled"
    return PluginState(enabled, origin, reason, required)


def set_plugin_override(config, name, enabled):
    from copy import deepcopy

    overrides = deepcopy(plugin_overrides(config))
    name = normalize_plugin_id(name)
    if enabled is None:
        overrides.pop(name, None)
    else:
        overrides[name] = enabled
    config.set("plugins.enabled", overrides)
