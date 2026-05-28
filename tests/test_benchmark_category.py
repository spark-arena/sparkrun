"""Tests for the benchmark category taxonomy (step 1 of the redesign).

Covers:
- Plugin-level ``categories`` / ``primary_category`` declarations and
  ``__init_subclass__`` defaulting.
- ``BenchmarkSpec.category`` loading from YAML, both embedded under
  ``benchmark:`` and at top-level for standalone profiles.
- ``BenchmarkSpec.resolved_category()`` fallback to the framework's
  ``primary_category`` when the spec doesn't declare one.
- Bootstrap discovery helpers (``list_benchmark_categories``,
  ``get_benchmarking_frameworks_for_category``,
  ``get_default_framework_for_category``) including the
  ``AmbiguousCategoryError`` and ``CategoryNotFoundError`` paths.
- ``find_benchmark_profile`` category filter (registry + local + scoped).
- ``RegistryManager.find_benchmark_profile_in_registries`` /
  ``list_benchmark_profiles`` category filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sparkrun.benchmarking.base import BenchmarkingPlugin
from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework
from sparkrun.benchmarking.tool_eval_bench import ToolEvalBenchFramework
from sparkrun.core.benchmark_profiles import (
    BenchmarkSpec,
    ProfileError,
    find_benchmark_profile,
)
from sparkrun.core.bootstrap import (
    AmbiguousCategoryError,
    CategoryNotFoundError,
    get_benchmarking_frameworks_for_category,
    get_default_framework_for_category,
    init_sparkrun,
    list_benchmark_categories,
)


# ---------------------------------------------------------------------------
# Plugin contract
# ---------------------------------------------------------------------------


class _PerfDefault(BenchmarkingPlugin):
    """Plugin that does NOT declare categories — should inherit defaults."""

    framework_name = "perf-default"

    def check_prerequisites(self) -> list[str]:
        return []

    def build_benchmark_command(self, target_url: str, model: str, args: dict[str, Any], result_file: str | None = None) -> list[str]:
        return ["/usr/bin/true"]

    def parse_results(self, stdout: str, stderr: str, result_file: str | None = None) -> dict[str, Any]:
        return {}


class _ToolsOnly(BenchmarkingPlugin):
    """Plugin that declares categories=("tools",) without primary_category."""

    framework_name = "tools-only"
    categories = ("tools",)

    def check_prerequisites(self) -> list[str]:
        return []

    def build_benchmark_command(self, target_url: str, model: str, args: dict[str, Any], result_file: str | None = None) -> list[str]:
        return ["/usr/bin/true"]

    def parse_results(self, stdout: str, stderr: str, result_file: str | None = None) -> dict[str, Any]:
        return {}


class _MultiCategory(BenchmarkingPlugin):
    """Plugin in two categories, with explicit primary_category."""

    framework_name = "multi-cat"
    categories = ("tools", "evals")
    primary_category = "evals"

    def check_prerequisites(self) -> list[str]:
        return []

    def build_benchmark_command(self, target_url: str, model: str, args: dict[str, Any], result_file: str | None = None) -> list[str]:
        return ["/usr/bin/true"]

    def parse_results(self, stdout: str, stderr: str, result_file: str | None = None) -> dict[str, Any]:
        return {}


def test_default_categories_are_performance():
    """A plugin that doesn't declare categories inherits ('performance',)."""
    assert _PerfDefault.categories == ("performance",)
    assert _PerfDefault.primary_category == "performance"


def test_tools_only_plugin_inherits_primary_from_categories():
    """When a subclass declares categories but not primary_category, the first wins."""
    assert _ToolsOnly.categories == ("tools",)
    assert _ToolsOnly.primary_category == "tools"


def test_explicit_primary_overrides_first_category():
    """When primary_category is explicit, it wins regardless of order in categories."""
    assert _MultiCategory.categories == ("tools", "evals")
    assert _MultiCategory.primary_category == "evals"


def test_bundled_plugins_have_expected_categories():
    """Sanity: the two real plugins declare what step 1 expects."""
    assert LlamaBenchyFramework.categories == ("performance",)
    assert LlamaBenchyFramework.primary_category == "performance"
    assert ToolEvalBenchFramework.categories == ("tools",)
    assert ToolEvalBenchFramework.primary_category == "tools"


# ---------------------------------------------------------------------------
# BenchmarkSpec.category loading
# ---------------------------------------------------------------------------


def _write_profile(path: Path, data: dict[str, Any]) -> Path:
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh)
    return path


def test_spec_load_reads_explicit_category_from_benchmark_block(tmp_path: Path):
    profile = _write_profile(
        tmp_path / "p.yaml",
        {"benchmark": {"framework": "llama-benchy", "category": "performance", "pp": [2048]}},
    )
    spec = BenchmarkSpec.load(profile)
    assert spec.category == "performance"


def test_spec_load_reads_explicit_category_from_standalone_profile(tmp_path: Path):
    profile = _write_profile(
        tmp_path / "p.yaml",
        {"framework": "tool-eval-bench", "category": "tools", "scenarios": ["a"]},
    )
    spec = BenchmarkSpec.load(profile)
    assert spec.category == "tools"


def test_spec_load_leaves_category_none_when_unset(tmp_path: Path):
    profile = _write_profile(
        tmp_path / "p.yaml",
        {"benchmark": {"framework": "llama-benchy", "pp": [2048]}},
    )
    spec = BenchmarkSpec.load(profile)
    assert spec.category is None


def test_spec_load_rejects_non_string_category(tmp_path: Path):
    profile = _write_profile(
        tmp_path / "p.yaml",
        {"benchmark": {"framework": "llama-benchy", "category": 42}},
    )
    with pytest.raises(Exception):
        BenchmarkSpec.load(profile)


def test_resolved_category_falls_back_to_framework_primary():
    init_sparkrun()
    spec = BenchmarkSpec(source_path="", framework="tool-eval-bench", args={})
    # category=None → tool-eval-bench's primary_category
    assert spec.resolved_category() == "tools"

    spec_perf = BenchmarkSpec(source_path="", framework="llama-benchy", args={})
    assert spec_perf.resolved_category() == "performance"


def test_resolved_category_prefers_explicit():
    spec = BenchmarkSpec(source_path="", framework="llama-benchy", args={}, category="evals")
    assert spec.resolved_category() == "evals"


def test_resolved_category_default_when_framework_unknown():
    spec = BenchmarkSpec(source_path="", framework="not-registered-xyz", args={})
    # No framework lookup → caller's default
    assert spec.resolved_category("performance") == "performance"
    assert spec.resolved_category("safety") == "safety"


# ---------------------------------------------------------------------------
# Bootstrap discovery
# ---------------------------------------------------------------------------


def test_list_benchmark_categories_includes_bundled():
    init_sparkrun()
    cats = list_benchmark_categories()
    assert "performance" in cats
    assert "tools" in cats
    assert cats == sorted(cats)  # deterministic ordering


def test_get_frameworks_for_category_returns_only_matching():
    init_sparkrun()
    perf = [fw.framework_name for fw in get_benchmarking_frameworks_for_category("performance")]
    tools = [fw.framework_name for fw in get_benchmarking_frameworks_for_category("tools")]
    assert "llama-benchy" in perf
    assert "tool-eval-bench" not in perf
    assert "tool-eval-bench" in tools
    assert "llama-benchy" not in tools


def test_get_frameworks_for_unknown_category_returns_empty():
    init_sparkrun()
    assert get_benchmarking_frameworks_for_category("hallucinations") == []


def test_default_framework_for_singleton_category():
    """When exactly one framework registers a category, it is the default."""
    init_sparkrun()
    fw = get_default_framework_for_category("tools")
    assert fw.framework_name == "tool-eval-bench"


def test_default_framework_respects_config_pin():
    """default_benchmark_framework on config wins when it matches the category."""
    init_sparkrun()

    class _FakeConfig:
        default_benchmark_framework = "llama-benchy"

    fw = get_default_framework_for_category("performance", config=_FakeConfig())
    assert fw.framework_name == "llama-benchy"


def test_default_framework_falls_back_when_config_pin_wrong_category():
    """When config pins a framework that isn't in the category, fall back to candidates."""
    init_sparkrun()

    class _FakeConfig:
        # tools-only plugin name; performance category should ignore this.
        default_benchmark_framework = "tool-eval-bench"

    fw = get_default_framework_for_category("performance", config=_FakeConfig())
    assert fw.framework_name == "llama-benchy"


def test_default_framework_category_not_found_raises():
    init_sparkrun()
    with pytest.raises(CategoryNotFoundError):
        get_default_framework_for_category("hallucinations")


def test_default_framework_ambiguous_raises(monkeypatch):
    """When >1 framework belongs to a category and no config pin, raise."""
    init_sparkrun()

    def _two_candidates(category, v=None):
        class _A:
            framework_name = "a-fw"

        class _B:
            framework_name = "b-fw"

        return [_A(), _B()]

    monkeypatch.setattr(
        "sparkrun.core.bootstrap.get_benchmarking_frameworks_for_category",
        _two_candidates,
    )
    with pytest.raises(AmbiguousCategoryError):
        get_default_framework_for_category("performance")


# ---------------------------------------------------------------------------
# find_benchmark_profile category filter
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, config_dir: Path) -> None:
        self.config_path = config_dir / "config.yaml"


def test_find_benchmark_profile_local_dir_filters_by_category(tmp_path: Path):
    init_sparkrun()
    config_dir = tmp_path / "config"
    bench_dir = config_dir / "benchmarking"
    bench_dir.mkdir(parents=True)
    _write_profile(bench_dir / "fast.yaml", {"framework": "llama-benchy", "pp": [2048]})

    cfg = _FakeConfig(config_dir)

    # No filter → resolves
    assert find_benchmark_profile("fast", cfg, registry_manager=None) == bench_dir / "fast.yaml"

    # Performance filter → resolves (derived from framework primary)
    assert find_benchmark_profile("fast", cfg, registry_manager=None, category="performance") == bench_dir / "fast.yaml"

    # Tools filter → not found
    with pytest.raises(ProfileError):
        find_benchmark_profile("fast", cfg, registry_manager=None, category="tools")


def test_find_benchmark_profile_direct_path_ignores_category_filter(tmp_path: Path):
    """Explicit file paths are honored regardless of declared category."""
    init_sparkrun()
    p = _write_profile(tmp_path / "explicit.yaml", {"framework": "llama-benchy"})
    cfg = _FakeConfig(tmp_path)

    # str(p) reliably contains a "/" so the resolver takes the direct-path branch
    found = find_benchmark_profile(str(p), cfg, registry_manager=None, category="tools")
    assert found == p


# ---------------------------------------------------------------------------
# RegistryManager category filtering
# ---------------------------------------------------------------------------


class _FakeEntry:
    def __init__(self, name: str, benchmark_subpath: str, enabled: bool = True, visible: bool = True):
        self.name = name
        self.benchmark_subpath = benchmark_subpath
        self.enabled = enabled
        self.visible = visible


class _FakeRegistryManager:
    """Bare-bones RegistryManager substitute exposing only what these tests need."""

    def __init__(self, root: Path, entries: list[_FakeEntry]) -> None:
        self._root = root
        self._entries = entries

    def _load_registries(self):
        return self._entries

    def _cache_dir(self, name: str) -> Path:
        return self._root / name

    # Reuse the real implementations — they only depend on the methods above.
    from sparkrun.core.registry import RegistryManager as _Real

    _benchmark_dir = _Real._benchmark_dir
    find_benchmark_profile_in_registries = _Real.find_benchmark_profile_in_registries
    list_benchmark_profiles = _Real.list_benchmark_profiles


def _make_registry_fixture(tmp_path: Path) -> tuple[_FakeRegistryManager, dict[str, Path]]:
    root = tmp_path / "registries"
    perf_dir = root / "perf-reg" / "benchmarks"
    tools_dir = root / "tools-reg" / "benchmarks"
    perf_dir.mkdir(parents=True)
    tools_dir.mkdir(parents=True)

    perf_profile = _write_profile(
        perf_dir / "throughput.yaml",
        {"framework": "llama-benchy", "category": "performance", "pp": [2048]},
    )
    tools_profile = _write_profile(
        tools_dir / "throughput.yaml",
        {"framework": "tool-eval-bench"},  # category derived from framework
    )
    other_perf_profile = _write_profile(
        perf_dir / "latency.yaml",
        {"framework": "llama-benchy"},  # derived performance
    )

    rm = _FakeRegistryManager(
        root,
        [
            _FakeEntry("perf-reg", "benchmarks"),
            _FakeEntry("tools-reg", "benchmarks"),
        ],
    )
    return rm, {
        "perf-throughput": perf_profile,
        "tools-throughput": tools_profile,
        "perf-latency": other_perf_profile,
    }


def test_find_in_registries_category_filter_narrows_matches(tmp_path: Path):
    init_sparkrun()
    rm, files = _make_registry_fixture(tmp_path)

    # No filter: two matches for the colliding "throughput" stem.
    matches = rm.find_benchmark_profile_in_registries("throughput")
    paths = sorted(p for _, p in matches)
    assert paths == sorted([files["perf-throughput"], files["tools-throughput"]])

    # performance filter narrows to one.
    matches_perf = rm.find_benchmark_profile_in_registries("throughput", category="performance")
    assert [p for _, p in matches_perf] == [files["perf-throughput"]]

    # tools filter narrows to the other one.
    matches_tools = rm.find_benchmark_profile_in_registries("throughput", category="tools")
    assert [p for _, p in matches_tools] == [files["tools-throughput"]]


def test_list_benchmark_profiles_emits_category_and_filters(tmp_path: Path):
    init_sparkrun()
    rm, _files = _make_registry_fixture(tmp_path)

    # Unfiltered: each profile carries a category (declared or derived).
    all_profiles = rm.list_benchmark_profiles()
    by_path = {p["path"]: p for p in all_profiles}
    assert all(p["category"] in {"performance", "tools"} for p in all_profiles)

    # Filtered to performance: only performance profiles appear.
    perf_only = rm.list_benchmark_profiles(category="performance")
    assert {p["registry"] for p in perf_only} == {"perf-reg"}
    assert all(p["category"] == "performance" for p in perf_only)

    # Filtered to tools: only tools profiles appear.
    tools_only = rm.list_benchmark_profiles(category="tools")
    assert {p["registry"] for p in tools_only} == {"tools-reg"}
    assert all(p["category"] == "tools" for p in tools_only)

    # Sanity: by_path keys all distinct (no duplicate entries).
    assert len(by_path) == len(all_profiles)
