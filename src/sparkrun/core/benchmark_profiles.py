from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vpd import read_yaml

from sparkrun.utils.shell import render_args_as_flags
from sparkrun.core.recipe import Recipe

# Keys in the benchmark: block that are NOT framework args
_KNOWN_BENCHMARK_KEYS = {"framework", "args", "metadata", "timeout"}


class ProfileError(Exception):
    """Raised when a benchmark profile cannot be found."""


class ProfileAmbiguousError(ProfileError):
    """Raised when a profile name matches multiple registries."""

    def __init__(self, name: str, matches: list[tuple[str, Path]]):
        self.name = name
        self.matches = matches
        reg_names = [reg for reg, _ in matches]
        super().__init__(
            "Benchmark profile '%s' found in multiple registries: %s. "
            "Use @registry/%s to disambiguate." % (name, ", ".join(reg_names), name)
        )


def find_benchmark_profile(
    name: str,
    config,
    registry_manager=None,
    include_hidden: bool = False,
) -> Path:
    """Find a benchmark profile by name.

    Resolution chain:
    1. Direct file path (contains / or .yaml/.yml extension)
    2. @registry/name scoped lookup
    3. Local benchmarking directory (~/.config/sparkrun/benchmarking/)
    4. Registry search with ambiguity detection

    Args:
        name: Profile name, path, or @registry/name
        config: SparkrunConfig instance
        registry_manager: Optional RegistryManager for registry search
        include_hidden: If True, include hidden registries

    Returns:
        Path to the profile YAML file.

    Raises:
        ProfileError: If profile not found.
        ProfileAmbiguousError: If bare name matches multiple registries.
    """
    # Parse @registry/ prefix
    from sparkrun.utils import parse_scoped_name

    scoped_registry, lookup_name = parse_scoped_name(name)

    # 1. Direct file path
    if "/" in name and not name.startswith("@"):
        direct = Path(name)
        if direct.exists():
            return direct
        # Try with extension
        for ext in (".yaml", ".yml"):
            candidate = Path(name + ext)
            if candidate.exists():
                return candidate
        raise ProfileError("Benchmark profile file not found: %s" % name)

    # 2. Scoped registry lookup
    if scoped_registry and registry_manager:
        matches = registry_manager.find_benchmark_profile_in_registries(
            lookup_name,
            include_hidden=True,
        )
        scoped_matches = [(reg, path) for reg, path in matches if reg == scoped_registry]
        if scoped_matches:
            return scoped_matches[0][1]
        raise ProfileError("Benchmark profile '%s' not found in registry '%s'" % (lookup_name, scoped_registry))

    # 3. Local benchmarking directory
    local_dir = config.config_path.parent / "benchmarking"
    if local_dir.is_dir():
        for ext in (".yaml", ".yml"):
            candidate = local_dir / (lookup_name + ext)
            if candidate.exists():
                return candidate

    # 4. Registry search with ambiguity detection
    if registry_manager:
        matches = registry_manager.find_benchmark_profile_in_registries(
            lookup_name,
            include_hidden=include_hidden,
        )
        if len(matches) == 1:
            return matches[0][1]
        elif len(matches) > 1:
            raise ProfileAmbiguousError(lookup_name, matches)

    raise ProfileError("Benchmark profile '%s' not found" % lookup_name)


class BenchmarkError(Exception):
    """Raised when a benchmark spec is invalid or cannot be loaded."""


@dataclass
class BenchmarkSpec:
    """Standalone benchmark YAML definition.

    Supports two formats for the ``benchmark:`` block:

    Nested (explicit args)::

        benchmark:
          framework: llama-benchy
          args:
            pp: [2048]
            depth: [0]

    Flat (unknown keys swept into args)::

        benchmark:
          framework: llama-benchy
          pp: [2048]
          depth: [0]
    """

    source_path: str
    framework: str
    args: dict[str, Any]
    timeout: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkSpec:
        """Load and validate a benchmark YAML file.

        Supports two formats:

        1. Embedded in a recipe YAML (``benchmark:`` key wraps the block).
        2. Standalone profile file (top-level keys *are* the benchmark config).
        """
        p = Path(path)
        if not p.exists():
            raise BenchmarkError("Benchmark file not found: %s" % p)

        data = read_yaml(str(p))
        if not isinstance(data, dict):
            raise BenchmarkError("Benchmark file must contain a YAML mapping: %s" % p)

        block = data.get("benchmark")
        if not isinstance(block, dict):
            # Standalone profile file: top-level keys *are* the benchmark block
            if "framework" in data:
                block = data
            else:
                raise BenchmarkError("Benchmark file missing required 'benchmark' mapping")

        framework = block.get("framework")
        if not framework or not isinstance(framework, str):
            raise BenchmarkError("benchmark.framework is required and must be a string")

        # Explicit args take priority; unknown keys are swept in as well
        args = dict(block.get("args") or {})
        for k, v in block.items():
            if k not in _KNOWN_BENCHMARK_KEYS and k not in args:
                args[k] = v

        if not isinstance(args, dict):
            raise BenchmarkError("benchmark.args must be a mapping")

        timeout = block.get("timeout")
        if timeout is not None:
            timeout = int(timeout)

        return cls(
            source_path=str(p),
            framework=framework,
            args=args,
            timeout=timeout,
        )

    @classmethod
    def from_recipe(cls, recipe: Recipe) -> BenchmarkSpec | None:
        """Extract benchmark config from a recipe's raw data.

        Returns None if the recipe has no ``benchmark:`` block.
        """
        block = recipe._raw.get("benchmark")
        if not isinstance(block, dict):
            return None

        framework = block.get("framework", "llama-benchy")

        # Explicit args take priority; unknown keys swept in
        args = dict(block.get("args") or {})
        for k, v in block.items():
            if k not in _KNOWN_BENCHMARK_KEYS and k not in args:
                args[k] = v

        timeout = block.get("timeout")
        if timeout is not None:
            timeout = int(timeout)

        return cls(
            source_path=recipe.source_path or "",
            framework=str(framework),
            args=args,
            timeout=timeout,
        )

    def build_command(self, extra_args: dict[str, Any] | None = None) -> list[str]:
        """Render a shell argv list for this benchmark spec.

        Command shape: ``framework --kebab-case-key VALUE ...``

        Booleans become bare flags; lists emit repeated flags.
        """
        merged_args = dict(self.args)
        if extra_args:
            merged_args.update(extra_args)

        cmd: list[str] = [self.framework]
        cmd.extend(render_args_as_flags(merged_args))

        return cmd
