"""Bootstrap sparkrun plugin system using SAF's lightweight test harness init."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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


def init_sparkrun(v: Variables | None = None, log_level: str = "WARNING") -> Variables:
    """Initialize sparkrun's plugin system.

    Uses SAF's init_framework_test_harness for a lightweight framework
    initialization that properly sets up the plugin registry without
    heavy-weight features (no fault handler, no shutdown hooks, no stateful).

    Args:
        v: Optional pre-existing Variables instance to reuse.
        log_level: SAF log level (default WARNING to reduce verbosity).

    Returns:
        The initialized Variables instance.
    """
    global _variables

    if _variables is not None and v is None:
        return _variables

    if v is None:
        from scitrera_app_framework import init_framework_desktop

        v = init_framework_desktop("sparkrun", log_level=log_level, fault_handler=False, shutdown_hooks=False, fixed_logger=logger)

        # suppress noisy loggers (separate from our logging level)
        from sparkrun.utils import suppress_noisy_loggers

        suppress_noisy_loggers()

    _variables = v

    # Import here to avoid circular imports
    from sparkrun.runtimes.base import RuntimePlugin

    # Auto-discover all RuntimePlugin subclasses in sparkrun.runtimes
    discovered = list(find_types_in_modules("sparkrun.runtimes", RuntimePlugin))
    for runtime_cls in discovered:
        try:
            register_plugin(runtime_cls, v=v)
            logger.debug("Registered runtime: %s", runtime_cls.__name__)
        except (ValueError, TypeError) as e:
            logger.debug("Skipping runtime %s: %s", runtime_cls.__name__, e)

    # Auto-discover all BenchmarkingPlugin subclasses in sparkrun.benchmarking
    from sparkrun.benchmarking.base import BenchmarkingPlugin as _BenchPlugin

    discovered_bench = list(find_types_in_modules("sparkrun.benchmarking", _BenchPlugin))
    for bench_cls in discovered_bench:
        try:
            register_plugin(bench_cls, v=v)
            logger.debug("Registered benchmarking framework: %s", bench_cls.__name__)
        except (ValueError, TypeError) as e:
            logger.debug("Skipping benchmarking framework %s: %s", bench_cls.__name__, e)

    # Auto-discover all BuilderPlugin subclasses in sparkrun.builders
    from sparkrun.builders.base import BuilderPlugin as _BuilderPlugin

    discovered_builders = list(find_types_in_modules("sparkrun.builders", _BuilderPlugin))
    for builder_cls in discovered_builders:
        try:
            register_plugin(builder_cls, v=v)
            logger.debug("Registered builder: %s", builder_cls.__name__)
        except (ValueError, TypeError) as e:
            logger.debug("Skipping builder %s: %s", builder_cls.__name__, e)

    return v


def get_variables() -> Variables:
    """Get the sparkrun Variables instance, initializing if needed."""
    global _variables
    if _variables is None:
        init_sparkrun()
    return _variables


def get_runtime(name: str, v: Variables | None = None) -> RuntimePlugin:
    """Get a specific runtime by name.

    Args:
        name: Runtime name (e.g. "vllm", "sglang", "eugr-vllm")
        v: Optional Variables instance; uses singleton if not provided

    Raises:
        ValueError: If the runtime is not found
    """
    if v is None:
        v = get_variables()

    all_runtimes = get_extensions(EXT_RUNTIME, v=v)
    for _plugin_name, runtime in all_runtimes.items():
        if runtime.runtime_name == name:
            return runtime

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


def get_builder(name: str, v: Variables | None = None) -> "BuilderPlugin":
    """Get a specific builder by name.

    Args:
        name: Builder name (e.g. "docker-pull", "eugr")
        v: Optional Variables instance; uses singleton if not provided

    Raises:
        ValueError: If the builder is not found
    """
    if v is None:
        v = get_variables()

    all_builders = get_extensions(EXT_BUILDER, v=v)
    for _plugin_name, builder in all_builders.items():
        if builder.builder_name == name:
            return builder

    available = [b.builder_name for b in all_builders.values()]
    raise ValueError("Unknown builder: %r. Available: %s" % (name, available))


def list_builders(v: Variables | None = None) -> list[str]:
    """List all registered builder names."""
    if v is None:
        v = get_variables()

    all_builders = get_extensions(EXT_BUILDER, v=v)
    return sorted(b.builder_name for b in all_builders.values())
