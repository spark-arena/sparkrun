"""Feature flags — channel-aware gating for experimental plugins and behavior.

A feature flag is a named boolean whose *default* value can vary per release
channel (``stable`` / ``beta`` / ``alpha``, see :mod:`sparkrun.core.channels`).
This lets an experimental capability ship on by default for the ``alpha`` train
while staying off for ``stable`` users, without a code change per release.

Resolution precedence (highest first):

1. Environment override — ``SPARKRUN_FEATURE_<NAME>`` (dots/dashes → underscores,
   upper-cased). Handy for CI and one-off debugging.
2. Explicit config override — ``features.<name>: true|false`` in ``config.yaml``.
3. Per-channel default declared on the flag (``channel_defaults``).
4. The flag's baseline ``default`` (used when the active channel isn't listed).
5. Unknown flag → ``False`` (fail closed).

The active channel defaults to the persisted self-update channel and can be
overridden per-config via ``features.channel`` (see
:attr:`sparkrun.core.config.SparkrunConfig.feature_channel`).

Plugins opt into gating by declaring ``required_feature_flag = "<flag-name>"``; the
bootstrap discovery loop skips registering a plugin whose flag resolves off (see
:func:`sparkrun.core.bootstrap.init_sparkrun`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, TYPE_CHECKING

from scitrera_app_framework import ext_parse_bool

from sparkrun.core.channels import CHANNEL_ALPHA, CHANNEL_BETA, CHANNEL_STABLE, normalize_channel

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig

_ENV_PREFIX = "SPARKRUN_FEATURE_"


def _env_key(name: str) -> str:
    """Return the environment-variable key that overrides feature *name*."""
    return _ENV_PREFIX + name.upper().replace(".", "_").replace("-", "_")


@dataclass(frozen=True)
class FeatureFlag:
    """Definition of a single feature flag.

    Args:
        name: Dotted flag identifier (e.g. ``"executor.k8s"``).
        description: Human-facing one-liner shown by ``setup features list``.
        channel_defaults: Per-channel default overrides, keyed by canonical
            channel name. Channels omitted here fall back to ``default``.
        default: Baseline default used when the active channel is not present
            in ``channel_defaults``.
    """

    name: str
    description: str
    channel_defaults: Mapping[str, bool] = field(default_factory=dict)
    default: bool = False

    def default_for_channel(self, channel: str | None) -> bool:
        """Return the default value of this flag for *channel*."""
        return bool(self.channel_defaults.get(normalize_channel(channel), self.default))


# --------------------------------------------------------------------------
# Registry — the single source of truth for known flags and their defaults.
# --------------------------------------------------------------------------

FEATURE_FLAGS: dict[str, FeatureFlag] = {}


def register_feature(flag: FeatureFlag) -> FeatureFlag:
    """Register *flag* in the global registry and return it.

    Raises:
        ValueError: if a different flag is already registered under the name.
    """
    existing = FEATURE_FLAGS.get(flag.name)
    if existing is not None and existing != flag:
        raise ValueError("Feature flag %r already registered with a different definition" % flag.name)
    FEATURE_FLAGS[flag.name] = flag
    return flag


def get_feature(name: str) -> FeatureFlag | None:
    """Return the :class:`FeatureFlag` registered under *name*, or ``None``."""
    return FEATURE_FLAGS.get(name)


def all_features() -> list[FeatureFlag]:
    """Return all registered flags, sorted by name."""
    return [FEATURE_FLAGS[n] for n in sorted(FEATURE_FLAGS)]


# --------------------------------------------------------------------------
# Resolution.
# --------------------------------------------------------------------------


def _env_override(name: str, env: Mapping[str, str]) -> bool | None:
    """Return the env-var override for *name*, or ``None`` when unset."""
    raw = env.get(_env_key(name))
    if raw is None:
        return None
    try:
        return bool(ext_parse_bool(raw))
    except Exception:  # pragma: no cover - ext_parse_bool is permissive
        return None


def is_feature_enabled(
    name: str,
    *,
    config: "SparkrunConfig | None" = None,
    channel: str | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Resolve whether feature *name* is enabled.

    See the module docstring for the full precedence order. Unknown flags
    (no registered definition and no explicit override) resolve to ``False``.
    """
    env = os.environ if env is None else env

    override = _env_override(name, env)
    if override is not None:
        return override

    if config is not None:
        cfg_override = config.feature_override(name)
        if cfg_override is not None:
            return bool(cfg_override)

    if channel is None:
        channel = config.feature_channel if config is not None else CHANNEL_STABLE

    flag = get_feature(name)
    if flag is not None:
        return flag.default_for_channel(channel)

    # Unknown flag, no override anywhere — fail closed.
    return False


def feature_gate_enabled(feature: str, v=None) -> bool:
    """Resolve *feature* for a plugin self-gate at registration time.

    Called from ``Plugin.is_multi_extension`` (which SAF evaluates when a
    plugin registers, before any ``SparkrunContext`` exists), so the config
    is loaded from the standard path rather than an ``sctx``. Environment
    overrides (``SPARKRUN_FEATURE_*``) short-circuit before the file is read.
    """
    config = _gate_config(v)
    return is_feature_enabled(feature, config=config)


def _gate_config(v=None):
    """Load a :class:`SparkrunConfig` for gate resolution (best-effort).

    Read-only — this must never write. Returns ``None`` when a config can't
    be built, in which case resolution falls back to env + channel defaults.
    """
    try:
        from sparkrun.core.config import SparkrunConfig, get_config_root

        return SparkrunConfig(get_config_root(v) / "config.yaml")
    except Exception:  # pragma: no cover - defensive; gate degrades to defaults
        return None


def feature_source(
    name: str,
    *,
    config: "SparkrunConfig | None" = None,
    channel: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return where the effective value of *name* comes from.

    One of ``"env"``, ``"config"``, ``"channel"``, or ``"unset"``. Used by the
    ``setup features list`` command to explain why a flag is on or off.
    """
    env = os.environ if env is None else env
    if _env_override(name, env) is not None:
        return "env"
    if config is not None and config.feature_override(name) is not None:
        return "config"
    if get_feature(name) is not None:
        return "channel"
    return "unset"


# --------------------------------------------------------------------------
# Built-in flags.
# --------------------------------------------------------------------------

# The docker executor is the default backend and ships ENABLED on every
# channel (``default=True``, no channel overrides). It carries a flag purely for
# uniformity — so every executor self-gates the same way — and to leave the door
# open to disabling docker on hosts/clusters that don't need it. A user can turn
# it off explicitly via ``features.executor.docker: false`` (or the env
# override); unlike the experimental executors below, it is never off by default.
FEATURE_EXECUTOR_DOCKER = register_feature(
    FeatureFlag(
        name="executor.docker",
        description="Default Docker executor (enabled on all channels; disable to drop docker where it isn't needed)",
        default=True,
    )
)

# The experimental executors are off by default on every channel; users opt
# in explicitly via ``features.executor.*`` in config.yaml (or the
# ``SPARKRUN_FEATURE_*`` env override). The per-channel default mechanism
# (``channel_defaults``) remains available for future flags that want it.
FEATURE_EXECUTOR_LOCAL = register_feature(
    FeatureFlag(
        name="executor.local",
        description="Experimental native (no-container) executor",
        default=False,
    )
)

FEATURE_EXECUTOR_K8S = register_feature(
    FeatureFlag(
        name="executor.k8s",
        description="Experimental Kubernetes (kubectl) executor draft",
        default=False,
    )
)

# The uv-venv builder provisions a Python venv on the target hosts (running
# `uv` and installing packages there) instead of preparing a container image.
# That is real host-side mutation outside a container, so stable keeps it off;
# beta/alpha get it by default since it is the only way to run vllm-class
# runtimes where nested `docker run` doesn't work.
FEATURE_BUILDER_UV_VENV = register_feature(
    FeatureFlag(
        name="builder.uv_venv",
        description="uv-venv environment builder (provisions a host-side Python venv; pairs with executor: local)",
        channel_defaults={CHANNEL_BETA: True, CHANNEL_ALPHA: True},
        default=False,
    )
)

FEATURE_CLI_SETUP_K8S = register_feature(
    FeatureFlag(
        name="cli.setup.k8s",
        description="Experimental 'sparkrun setup k8s' command group",
        default=False,
    )
)

FEATURE_API_RUN_K8S = register_feature(
    FeatureFlag(
        name="api.run.k8s",
        description="Experimental: route 'sparkrun run' with executor=k8s through the JobSet launch path",
        default=False,
    )
)

FEATURE_CLI_SETUP_TAILSCALE = register_feature(
    FeatureFlag(
        name="cli.setup.tailscale",
        description="Experimental 'sparkrun setup tailscale' command group (join nodes / publish endpoint on a tailnet)",
        default=False,
    )
)

# These two gate *maturity*, not blast radius — which is why they are split at
# the suite boundary rather than at "does it need the container".
#
# The perftest suite has field mileage: it is the common "did my cable work?"
# check, and gating it was friction exactly when a user's networking is already
# broken. So it ships on every channel, and the flag remains only as a kill
# switch (the ``executor.docker`` shape: ``default=True``, no channel
# overrides).
FEATURE_CLI_SETUP_RDMA_TEST = register_feature(
    FeatureFlag(
        name="cli.setup.rdma_test",
        description="'sparkrun setup rdma-test' command (per-link RDMA bandwidth and latency)",
        default=True,
    )
)

# The collective does not have that mileage yet: mpirun across containers, the
# purpose-built nccl-tests image, and a verdict derived from bus bandwidth are
# all young. It rides alpha until they are proven. Note this *narrows* beta,
# which had the collective under the single flag.
#
# Deliberately not gated on "pulls the image": the perftest suite falls back to
# the container when a host lacks perftest, and refusing that would break the
# proven check on non-DGX-OS hosts for no gain. Maturity is the axis; the
# image is incidental to it.
FEATURE_CLI_SETUP_RDMA_TEST_NCCL = register_feature(
    FeatureFlag(
        name="cli.setup.rdma_test.nccl",
        description="Experimental NCCL collective suite for 'setup rdma-test' (--suite nccl/all)",
        channel_defaults={CHANNEL_ALPHA: True},
        default=False,
    )
)

# Stable and beta use LiteLLM; alpha exercises the bundled SparkRoute plugin.
# These remain ordinary feature defaults, so explicit config/env overrides win.
# Gateway selection uses the existing resolver after applying these gates.
FEATURE_GATEWAY_LITELLM = register_feature(
    FeatureFlag(
        name="gateway.litellm",
        description="LiteLLM gateway behind 'sparkrun proxy' (enabled by default on stable and beta)",
        channel_defaults={CHANNEL_ALPHA: False},
        default=True,
    )
)

# Loading out-of-tree plugins from ``plugins.paths`` executes user-supplied
# Python at startup; keep it off by default on every channel so a stock install
# never reads (let alone imports) external plugin directories unless the user
# opts in explicitly. See ``sparkrun.core.external_plugins``.
FEATURE_CORE_EXTERNAL_PLUGINS = register_feature(
    FeatureFlag(
        name="core.external_plugins",
        description="Load out-of-tree plugins from 'plugins.paths' in config.yaml at startup",
        default=False,
    )
)

# Visibility-only flag: the 'setup features' group is always functional, but is
# hidden from `setup --help` unless this resolves on. On by default for the beta
# and alpha channels (where poking at flags is expected), off for stable.
FEATURE_CLI_SETUP_FEATURES = register_feature(
    FeatureFlag(
        name="cli.setup.features",
        description="Show the 'sparkrun setup features' group in --help (always functional; on by default for beta/alpha)",
        channel_defaults={CHANNEL_BETA: True, CHANNEL_ALPHA: True},
        default=False,
    )
)


FEATURE_GATEWAY_SPARKROUTE = register_feature(
    FeatureFlag(
        name="gateway.sparkroute",
        description="SparkRoute gateway and workload bridge (enabled by default on alpha)",
        channel_defaults={CHANNEL_ALPHA: True},
        default=False,
    )
)
