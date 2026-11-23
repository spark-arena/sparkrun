"""Bootstrap sparkrun plugin system using SAF's lightweight test harness init."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from scitrera_app_framework import Variables, register_plugin, get_extensions
from scitrera_app_framework.util import find_types_in_modules

from sparkrun.runtimes.base import EXT_RUNTIME
from sparkrun.builders.base import EXT_BUILDER

if TYPE_CHECKING:
    from sparkrun.runtimes.base import RuntimePlugin
    from sparkrun.benchmarking.base import BenchmarkingPlugin
    from sparkrun.builders.base import BuilderPlugin

logger = logging.getLogger(__name__)

EXT_BENCHMARKING_FRAMEWORKS = "sparkrun.benchmarking"

# Module-level singleton for the sparkrun Variables instance
_variables: Variables | None = None
_initialization_error: BaseException | None = None


def init_sparkrun(v: Variables | None = None, log_level: str = "WARNING", *, profile=None, config=None) -> Variables:
    """Initialize sparkrun's plugin system.

    Uses SAF's desktop initialization with application-specific state and no
    fault handler or shutdown hooks. The first call binds plugin discovery to
    one canonical configuration path for the lifetime of the process. After
    plugin registration fails, this process cannot return an initialized
    context; fix the cause and start a new process.

    Args:
        v: Optional pre-existing Variables instance to reuse.
        log_level: SAF log level (default WARNING to reduce verbosity).

    Returns:
        The initialized Variables instance.
    """
    global _variables, _initialization_error

    if _initialization_error is not None:
        raise RuntimeError("Application initialization previously failed; restart the process before retrying") from _initialization_error

    from sparkrun.core.application_profile import select_application_profile, get_application_profile

    active = select_application_profile(profile) if profile is not None else get_application_profile()
    import sparkrun.core.config as config_module

    if config is not None and config.profile != active:
        raise RuntimeError("Configuration belongs to another distribution")
    config_path = (config.config_path if config is not None else config_module.resolve_config_path(v)).expanduser().resolve()
    # CLI dispatch can pin the path before plugin discovery. Implicit API
    # initialization shares that same binding and cannot replace it.
    config_module._bind_config_path(config_path)

    from sparkrun.core.plugin_store import PluginStore

    PluginStore(config if config is not None else config_module.SparkrunConfig(config_path)).snapshot()

    if _variables is not None:
        if v is not None and v is not _variables:
            raise RuntimeError("Application is already initialized; Variables can only be supplied on first initialization")
        return _variables

    from sparkrun.core.managed_plugins import validating_candidate
    from sparkrun.core.plugin_defaults import reconcile_default_plugins

    if not validating_candidate():
        try:
            results = reconcile_default_plugins(config if config is not None else config_module.SparkrunConfig(config_path))
            if config is not None and any(result["status"] == "installed" for result in results):
                config._load()  # Installation hooks may have committed plugin-owned settings.
            for result in results:
                if result["status"] == "failed":
                    logger.warning("Cannot install default plugin %s: %s", result["name"], result["error"])
        except (ValueError, OSError) as exc:
            logger.warning("Cannot reconcile default plugins: %s", exc)

    if v is None:
        from scitrera_app_framework import init_framework_desktop

        from sparkrun.core.config import get_config_root

        root = get_config_root()
        # Pin alternate SAF identity locally, independently of generic SAF env.
        framework_options = {}
        if active.id != "sparkrun":
            from sparkrun.core.application_profile import product_path

            framework_v = Variables()
            framework_v.set("APP_NAME", active.id)
            framework_v.set("SAF_SETUP_STATEFUL", False)
            framework_options["v"] = framework_v
        v = init_framework_desktop(
            active.id,
            **framework_options,
            log_level=log_level,
            fault_handler=False,
            shutdown_hooks=False,
            fixed_logger=logger,
            default_stateful_root=str(root.parent),
            default_run_id=None,
            stateful_root_env_key="STATEFUL_ROOT" if active.id == "sparkrun" else active.env_prefix + "_STATEFUL_ROOT",
        )

        if active.id != "sparkrun":
            from scitrera_app_framework.core.core import _init_stateful_root

            state_root = product_path("state")
            state_root.parent.mkdir(parents=True, exist_ok=True)
            v.set("RUN_ID", None)
            v.set("RUN_SERIAL", None)
            _init_stateful_root(
                v,
                local_name=state_root.name,
                default_stateful_root=str(state_root.parent),
                default_chdir=False,
                stateful_root_env_key=active.env_prefix + "_STATEFUL_ROOT",
            )

        # suppress noisy loggers (separate from our logging level)
        from sparkrun.utils import suppress_noisy_loggers

        suppress_noisy_loggers()

    _variables = v

    try:
        _register_plugins(v, config=config)
    except BaseException as exc:
        # Registration has process-global side effects, so a partial bootstrap
        # cannot safely be retried or returned as an API-ready context.
        _initialization_error = exc
        raise
    return v


def _register_plugins(v: Variables, *, config=None) -> None:
    from sparkrun.core.installed_plugins import claim_implementation

    # Import here to avoid circular imports
    from sparkrun.runtimes.base import RuntimePlugin

    # Auto-discover all RuntimePlugin subclasses in sparkrun.runtimes.
    # Plugins gated off by a feature flag still register here; they hide
    # themselves as extensions via ``is_multi_extension`` (see
    # ``Plugin.is_multi_extension`` overrides / ``core.features``).
    discovered = list(find_types_in_modules("sparkrun.runtimes", RuntimePlugin))
    for runtime_cls in discovered:
        try:
            claim_implementation(runtime_cls, v)
            register_plugin(runtime_cls, v=v)
            logger.debug("Registered runtime: %s", runtime_cls.__name__)
        except (ValueError, TypeError) as e:
            logger.debug("Skipping runtime %s: %s", runtime_cls.__name__, e)

    # Auto-discover all BenchmarkingPlugin subclasses in sparkrun.benchmarking
    from sparkrun.benchmarking.base import BenchmarkingPlugin as _BenchPlugin

    discovered_bench = list(find_types_in_modules("sparkrun.benchmarking", _BenchPlugin))
    for bench_cls in discovered_bench:
        try:
            claim_implementation(bench_cls, v)
            register_plugin(bench_cls, v=v)
            logger.debug("Registered benchmarking framework: %s", bench_cls.__name__)
        except (ValueError, TypeError) as e:
            logger.debug("Skipping benchmarking framework %s: %s", bench_cls.__name__, e)

    # Auto-discover all BuilderPlugin subclasses in sparkrun.builders
    from sparkrun.builders.base import BuilderPlugin as _BuilderPlugin

    discovered_builders = list(find_types_in_modules("sparkrun.builders", _BuilderPlugin))
    for builder_cls in discovered_builders:
        try:
            claim_implementation(builder_cls, v)
            register_plugin(builder_cls, v=v)
            _record_builder_gate(builder_cls)
            logger.debug("Registered builder: %s", builder_cls.__name__)
        except (ValueError, TypeError) as e:
            logger.debug("Skipping builder %s: %s", builder_cls.__name__, e)

    # Auto-discover all Executor subclasses in sparkrun.orchestration.executors
    from sparkrun.orchestration.executors._base import Executor as _ExecutorPlugin

    discovered_executors = list(find_types_in_modules("sparkrun.orchestration.executors", _ExecutorPlugin))
    for executor_cls in discovered_executors:
        try:
            claim_implementation(executor_cls, v)
            register_plugin(executor_cls, v=v)
            logger.debug("Registered executor: %s", executor_cls.__name__)
        except (ValueError, TypeError) as e:
            logger.debug("Skipping executor %s: %s", executor_cls.__name__, e)

    # Auto-discover TelemetryProvider subclasses in sparkrun.orchestration.telemetry
    # (substrate resource sampling, keyed by scope — the telemetry peer of the
    # executors' query_status). Skip the abstract base (blank scope).
    from sparkrun.orchestration.telemetry._base import TelemetryProvider as _TelemetryPlugin

    discovered_telemetry = list(find_types_in_modules("sparkrun.orchestration.telemetry", _TelemetryPlugin))
    for telemetry_cls in discovered_telemetry:
        if not getattr(telemetry_cls, "scope", ""):
            logger.debug("Skipping unnamed telemetry provider: %s", telemetry_cls.__name__)
            continue
        try:
            claim_implementation(telemetry_cls, v)
            register_plugin(telemetry_cls, v=v)
            logger.debug("Registered telemetry provider: %s", telemetry_cls.__name__)
        except (ValueError, TypeError) as e:
            logger.debug("Skipping telemetry provider %s: %s", telemetry_cls.__name__, e)

    # Auto-discover all Scheduler subclasses in sparkrun.schedulers
    from sparkrun.core.scheduler import Scheduler as _SchedulerPlugin

    discovered_schedulers = list(find_types_in_modules("sparkrun.schedulers", _SchedulerPlugin))
    for scheduler_cls in discovered_schedulers:
        # Skip private base classes that intentionally leave scheduler_name
        # blank — they exist only to share logic between concrete plugins.
        if not getattr(scheduler_cls, "scheduler_name", ""):
            logger.debug("Skipping abstract/base scheduler: %s", scheduler_cls.__name__)
            continue
        try:
            claim_implementation(scheduler_cls, v)
            register_plugin(scheduler_cls, v=v)
            logger.debug("Registered scheduler: %s", scheduler_cls.__name__)
        except (ValueError, TypeError) as e:
            logger.debug("Skipping scheduler %s: %s", scheduler_cls.__name__, e)

    # Auto-discover all Transport subclasses in sparkrun.transports. Imported
    # here (function-local) so ``core`` only touches ``transports`` at
    # discovery time — no module-level cycle (transports imports core).
    from sparkrun.transports.base import Transport as _TransportPlugin

    discovered_transports = list(find_types_in_modules("sparkrun.transports", _TransportPlugin))
    for transport_cls in discovered_transports:
        # Skip the abstract-ish base (blank transport_name) shared by concrete
        # transports; only named selectors are registered.
        if not getattr(transport_cls, "transport_name", ""):
            logger.debug("Skipping unnamed transport: %s", transport_cls.__name__)
            continue
        try:
            claim_implementation(transport_cls, v)
            register_plugin(transport_cls, v=v)
            logger.debug("Registered transport: %s", transport_cls.__name__)
        except (ValueError, TypeError) as e:
            logger.debug("Skipping transport %s: %s", transport_cls.__name__, e)

    # Establish all selected identities before any cross-cutting plugin imports.
    # Ambiguous providers cannot be resolved by import order: contested IDs are
    # skipped and recorded (require_integrations blocks dependent launches), so
    # one conflict or malformed source cannot fail initialization for every command.
    from sparkrun.core.plugin_identity import preflight_plugin_sources

    preflight_plugin_sources(v, config=config)
    from sparkrun.core.in_tree_plugins import load_in_tree_plugins

    try:
        load_in_tree_plugins(v)
    except Exception:  # noqa: BLE001 - a broken integration must not kill the CLI
        logger.exception("In-tree plugin loading failed")

    from sparkrun.core.external_plugins import load_external_plugins

    try:
        load_external_plugins(v)
    except Exception:  # noqa: BLE001 - a broken plugin dir must not kill the CLI
        logger.exception("External plugin loading failed")

    from sparkrun.core.managed_plugins import load_managed_plugins, validating_candidate

    try:
        load_managed_plugins(v, config=config)
    except Exception:  # noqa: BLE001 - a broken plugin store must not kill the CLI
        if validating_candidate():
            raise  # candidate validation reports through initialization failure
        logger.exception("Managed plugin loading failed")

    from sparkrun.core.installed_plugins import load_installed_plugins

    try:
        load_installed_plugins(v, config=config)
    except Exception:  # noqa: BLE001 - required integrations still fail closed at launch
        logger.exception("Installed plugin loading failed")


def get_variables() -> Variables:
    """Get the sparkrun Variables instance, initializing if needed."""
    global _variables
    if _variables is None or _initialization_error is not None:
        return init_sparkrun()
    return _variables


def get_runtime(name: str, v: Variables | None = None) -> RuntimePlugin:
    """Get a specific runtime by name.

    Args:
        name: Runtime name (e.g. "vllm", "sglang", "eugr-vllm")
        v: Optional Variables instance; uses singleton if not provided

    Raises:
        RuntimeUnavailableError: The runtime exists but its feature flag is off.
        ValueError: If the runtime is not found
    """
    if v is None:
        v = get_variables()

    all_runtimes = get_extensions(EXT_RUNTIME, v=v)
    for _plugin_name, runtime in all_runtimes.items():
        if runtime.runtime_name == name:
            return runtime

    from sparkrun.runtimes.base import GATED_OFF_RUNTIMES, RuntimeUnavailableError

    gate = GATED_OFF_RUNTIMES.get(name)
    if gate is not None:
        raise RuntimeUnavailableError(
            "Runtime %r is disabled by feature flag %r. Enable it with `sparkrun setup features enable %s`." % (name, gate, gate)
        )
    available = [r.runtime_name for r in all_runtimes.values()]
    raise ValueError("Unknown runtime: %r. Available: %s" % (name, available))


def list_runtimes(v: Variables | None = None) -> list[str]:
    """List all registered runtime names."""
    if v is None:
        v = get_variables()

    all_runtimes = get_extensions(EXT_RUNTIME, v=v)
    return sorted(r.runtime_name for r in all_runtimes.values())


def get_benchmarking_framework(name: str, v: Variables | None = None) -> "BenchmarkingPlugin":
    """Get a specific benchmarking framework by name.

    Args:
        name: Benchmarking framework name (e.g. "llama-benchy",...)
        v: Optional Variables instance; uses singleton if not provided

    Raises:
        ValueError: If the runtime is not found
    """
    if v is None:
        v = get_variables()

    all_frameworks = get_extensions(EXT_BENCHMARKING_FRAMEWORKS, v=v)
    for _plugin_name, runtime in all_frameworks.items():
        if runtime.framework_name == name:
            return runtime

    available = [r.framework_name for r in all_frameworks.values()]
    raise ValueError("Unknown benchmarking framework: %r. Available: %s" % (name, available))


def list_benchmarking_frameworks(v: Variables | None = None) -> list[str]:
    """List all registered benchmarking framework names."""
    if v is None:
        v = get_variables()

    all_frameworks = get_extensions(EXT_BENCHMARKING_FRAMEWORKS, v=v)
    return sorted(r.framework_name for r in all_frameworks.values())


def list_benchmark_categories(v: Variables | None = None) -> list[str]:
    """List all benchmark categories implemented by registered frameworks.

    The union of every plugin's ``categories`` tuple, deduplicated and sorted.
    Used by the CLI to decide which ``sparkrun benchmark <category>``
    subcommands to register.
    """
    if v is None:
        v = get_variables()

    all_frameworks = get_extensions(EXT_BENCHMARKING_FRAMEWORKS, v=v)
    seen: set[str] = set()
    for fw in all_frameworks.values():
        for cat in getattr(fw, "categories", ()) or ():
            if cat:
                seen.add(cat)
    return sorted(seen)


def get_benchmarking_frameworks_for_category(
    category: str,
    v: Variables | None = None,
) -> list["BenchmarkingPlugin"]:
    """Return all registered frameworks whose ``categories`` includes *category*."""
    if v is None:
        v = get_variables()

    all_frameworks = get_extensions(EXT_BENCHMARKING_FRAMEWORKS, v=v)
    return [fw for fw in all_frameworks.values() if category in (getattr(fw, "categories", ()) or ())]


class AmbiguousCategoryError(ValueError):
    """Raised when a category has multiple frameworks and no default is set."""


class CategoryNotFoundError(ValueError):
    """Raised when a category has no registered frameworks."""


def get_default_framework_for_category(
    category: str,
    config: Any = None,
    v: Variables | None = None,
) -> "BenchmarkingPlugin":
    """Return the framework to use by default for *category*.

    Resolution order:

    1. If *config* declares ``default_benchmark_framework`` and that
       framework belongs to *category*, use it.
    2. If exactly one registered framework belongs to *category*, use it.
    3. Otherwise raise :class:`AmbiguousCategoryError` (multiple matches) or
       :class:`CategoryNotFoundError` (no matches).
    """
    candidates = get_benchmarking_frameworks_for_category(category, v=v)
    if not candidates:
        raise CategoryNotFoundError(
            "No benchmarking framework registered for category %r. Registered categories: %s" % (category, list_benchmark_categories(v=v))
        )

    if config is not None:
        configured_name = getattr(config, "default_benchmark_framework", None)
        if configured_name:
            for fw in candidates:
                if fw.framework_name == configured_name:
                    return fw

    if len(candidates) == 1:
        return candidates[0]

    names = sorted(fw.framework_name for fw in candidates)
    raise AmbiguousCategoryError(
        "Category %r has multiple frameworks (%s). "
        "Pin one via `defaults.benchmark_framework` in config.yaml or "
        "pass --framework explicitly." % (category, names)
    )


#: Builder name/alias -> the feature flag gating it, recorded at discovery.
#:
#: A gated builder hides itself from ``get_extensions`` (see
#: ``BuilderPlugin.is_multi_extension``), so by the time :func:`get_builder`
#: fails to find one it can no longer tell "disabled" from "never existed".
#: Discovery runs *before* that gate, which is the one place both are visible.
_BUILDER_GATES: dict[str, str] = {}


def _record_builder_gate(builder_cls: "type[BuilderPlugin]") -> None:
    """Record *builder_cls*'s feature flag under its name and every alias."""
    flag = getattr(builder_cls, "required_feature_flag", None)
    if not flag:
        return
    for alias in (getattr(builder_cls, "builder_name", ""), *getattr(builder_cls, "builder_aliases", ())):
        if alias:
            _BUILDER_GATES[alias] = flag


def get_builder(name: str, v: Variables | None = None) -> "BuilderPlugin":
    """Get a specific builder by canonical name or alias.

    Args:
        name: Builder name or alias (e.g. "docker-pull", "eugr", "uv-venv", "venv")
        v: Optional Variables instance; uses singleton if not provided

    Raises:
        BuilderUnavailableError: The builder exists but its feature flag is off.
        ValueError: No builder claims *name*.
    """
    if v is None:
        v = get_variables()

    all_builders = get_extensions(EXT_BUILDER, v=v)
    for _plugin_name, builder in all_builders.items():
        if builder.matches_name(name):
            return builder

    gate = _BUILDER_GATES.get(name)
    if gate is not None:
        from sparkrun.builders.base import BuilderUnavailableError

        raise BuilderUnavailableError(
            "Builder %r is disabled by feature flag %r. Enable it with `sparkrun setup features enable %s`." % (name, gate, gate)
        )

    available = [b.builder_name for b in all_builders.values()]
    raise ValueError("Unknown builder: %r. Available: %s" % (name, available))


def list_builders(v: Variables | None = None) -> list[str]:
    """List canonical names of all enabled builders (aliases excluded)."""
    if v is None:
        v = get_variables()

    all_builders = get_extensions(EXT_BUILDER, v=v)
    return sorted(b.builder_name for b in all_builders.values())
