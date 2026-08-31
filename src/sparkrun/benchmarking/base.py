"""Benchmarking plugin base class, spec loading, and result export."""

from __future__ import annotations

import hashlib
import logging
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Any, Callable, ClassVar, TYPE_CHECKING, Optional

import yaml
from scitrera_app_framework import Plugin, Variables

from sparkrun.core.bootstrap import EXT_BENCHMARKING_FRAMEWORKS

if TYPE_CHECKING:
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.launcher import LaunchResult, ServeReadiness
    from sparkrun.benchmarking.scheduler import BenchTask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProgressColumn:
    """One column definition for the progress UI table."""

    name: str
    justify: str = "right"  # "left" | "right" | "center"
    style: str | None = None


def _identity_row_key(row: tuple[Any, ...]) -> Any:
    return row


def _default_rows_from_consolidated(consolidated: dict[str, Any]) -> list[tuple[Any, ...]]:
    """Default: emit one row per entry in ``consolidated["runs"]`` with its index."""
    runs = consolidated.get("runs") or []
    return [(i,) for i in range(len(runs))]


@dataclass(frozen=True)
class ProgressTableSpec:
    """Description of the per-completion progress table for a benchmarking plugin."""

    columns: list[ProgressColumn]
    rows_from_consolidated: Callable[[dict[str, Any]], list[tuple[Any, ...]]]
    row_key: Callable[[tuple[Any, ...]], Any] = _identity_row_key
    format_cell: Callable[[Any, ProgressColumn], str] | None = None


_DEFAULT_PROGRESS_TABLE_SPEC = ProgressTableSpec(
    columns=[ProgressColumn(name="idx", justify="right")],
    rows_from_consolidated=_default_rows_from_consolidated,
)


class BenchmarkingPlugin(Plugin):
    """Abstract base for benchmarking frameworks (SAF multi-extension plugin).

    Mirrors :class:`~sparkrun.runtimes.base.RuntimePlugin` in structure.
    Subclasses register via entry points under ``sparkrun.benchmarking``.
    """

    eager = False
    framework_name: str = ""
    default_args: dict[str, Any] = {}
    passthrough_args: set[str] = set()

    # Benchmark category taxonomy: the kind of question this framework answers
    # ("performance", "tools", "hallucinations", ...). A plugin may belong to
    # more than one category, but ``primary_category`` is the one used when a
    # spec/profile does not pin a category explicitly.
    categories: ClassVar[tuple[str, ...]] = ("performance",)
    primary_category: ClassVar[str] = "performance"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # If the subclass declared ``categories`` but not ``primary_category``,
        # default the primary to the first declared category. Without this,
        # tools-only plugins would inherit ``primary_category = "performance"``.
        if "categories" in cls.__dict__ and "primary_category" not in cls.__dict__:
            cats = cls.__dict__["categories"]
            if cats:
                cls.primary_category = cats[0]

    # --- SAF Plugin interface ---

    def name(self) -> str:
        return "sparkrun.benchmarking.%s" % self.framework_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_BENCHMARKING_FRAMEWORKS

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger: Logger) -> BenchmarkingPlugin:
        return self

    # --- Framework interface ---

    @abstractmethod
    def check_prerequisites(self) -> list[str]:
        """Check and return missing prerequisites (empty list = ready).

        For example, llama-benchy checks for uvx availability.
        """
        ...

    @abstractmethod
    def build_benchmark_command(
        self,
        target_url: str,
        model: str,
        args: dict[str, Any],
        result_file: str | None = None,
    ) -> list[str]:
        """Build the benchmark command argv list.

        Args:
            target_url: The inference endpoint URL (e.g. http://host:8000/v1).
            model: Model name for the --model flag.
            args: Merged profile + CLI override args.
            result_file: Path where the framework should save results.

        Returns:
            Command argv list suitable for subprocess.
        """
        ...

    def interpret_arg(self, key: str, value: str) -> Any:
        """Interpret a CLI string arg into the correct type.

        Default: comma-separated strings become lists, else coerce_value().
        Subclasses override for framework-specific type handling.
        """
        from sparkrun.utils import coerce_value

        if "," in value:
            return [coerce_value(v.strip()) for v in value.split(",")]
        return coerce_value(value)

    @abstractmethod
    def parse_results(self, stdout: str, stderr: str, result_file: str | None = None) -> dict[str, Any]:
        """Parse benchmark output into structured results dict.

        Args:
            stdout: Captured standard output.
            stderr: Captured standard error.
            result_file: Path to saved results file (if any).

        Returns:
            Structured results dict.
        """
        ...

    def get_default_args(self) -> dict[str, Any]:
        """Return default benchmark args when no profile is provided."""
        return dict(self.default_args)

    def prepare_benchmark_args(
        self,
        recipe: "Recipe",
        config_chain: dict[str, Any],
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Framework-specific args derived from the recipe/config chain.

        Called by the common benchmark path after profile + CLI overrides
        have been resolved, before ``build_benchmark_command``.  Each
        framework returns a dict of extra args it wants injected; the
        common code merges this in *without overwriting* keys already
        present in ``bench_args``.

        Default: nothing extra.  Frameworks override to opt in to recipe
        fields they care about (e.g. ``llama-benchy`` consumes
        ``served_model_name``).
        """
        return {}

    def detect_version(self) -> str | None:
        """Resolve the framework tool version that will be used for execution.

        Frameworks that pin a tool version for reproducibility (e.g. via
        ``uvx pkg@version``, or ``--from git+url@ref``) should implement this
        so the version is captured once and threaded back into every task's
        ``run_args`` via the ``framework_pinned_version`` sentinel.

        Default: framework does not support pinning.  Returns ``None``.
        """
        return None

    def apply_session_warmup_state(self, run_args: dict[str, Any], *, is_first_task: bool) -> dict[str, Any]:
        """Return a possibly-mutated copy of ``run_args`` reflecting the framework's
        one-time-per-session vs per-task split for warmup / coherence checks.
        Default: no change.  Frameworks that do warmup once per session at the
        start should set their warmup/coherence-disabling args on tasks beyond
        the first.
        """
        return dict(run_args)

    def estimate_test_count(self, args: dict[str, Any]) -> int | None:
        """Estimate the number of test combinations from the args.

        Returns None if the count cannot be determined.  Subclasses should
        override this when the framework's test matrix is predictable.
        """
        return None

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def measured_nothing(self, parsed: dict[str, Any]) -> bool:
        """Whether *parsed* results contain no actual measurement.

        A benchmark tool that exits 0 having failed every request is not a
        hypothetical: llama-benchy prints each failure and carries on, writing a
        result row per test with every metric ``null``.  sparkrun then exported
        that, rendered a table of ``…``, and said "Benchmark complete" — a run
        indistinguishable from a real one to anything downstream.  The tool's
        exit code cannot be the only check.

        Default ``False`` — "no opinion", so a framework that has not
        implemented this keeps the previous behaviour rather than being failed
        on a shape this base class cannot read.
        """
        return False

    def build_task_list(
        self,
        base_args: dict[str, Any],
        schedule: list[dict[str, Any]] | None,
    ) -> list["BenchTask"] | None:
        """Build a list of scheduled benchmark tasks or return None for legacy single-call path.

        If ``schedule`` is provided (non-None), the framework should validate each entry
        and raise :class:`~sparkrun.core.benchmark_profiles.BenchmarkError` on invalid entries.
        If ``schedule`` is None, the framework may optionally build a default task list
        from ``base_args`` (e.g. cartesian product). Returning None opts out of the
        batched execution path and falls back to the legacy single-call flow.

        Default: framework does not support batched/scheduled execution.
        """
        return None

    # --- Scheduler / aggregator hooks (optional, with safe defaults) -----

    def result_filename_suffix(self, task: "BenchTask") -> str:
        """Per-task suffix appended after the index in scheduler artifact filenames.

        The scheduler writes ``{idx:03d}{suffix}.json`` / ``{idx:03d}{suffix}.log``.
        Default: empty string (use index-only filenames).
        """
        return ""

    def consolidate_per_task_results(self, per_task_jsons: list[dict[str, Any]]) -> dict[str, Any]:
        """Consolidate per-task JSON dicts into a single framework-shaped dict.

        Default: ``{"runs": list(per_task_jsons)}``.
        """
        return {"runs": list(per_task_jsons)}

    def task_coverage_key(self, task: "BenchTask") -> Any:
        """Identifier used to detect whether a task is represented in consolidated results.

        Default: the task's index.
        """
        return task.index

    def consolidated_coverage_keys(self, consolidated: dict[str, Any]) -> set[Any]:
        """Return the set of coverage keys present in the consolidated dict.

        Default: indices ``[0, len(consolidated["runs"]))``.
        """
        return set(range(len(consolidated.get("runs") or [])))

    def progress_table_spec(self) -> ProgressTableSpec:
        """Return the column spec / row generator the progress UI uses for live display.

        Default: a minimal one-column ``idx`` table populated from
        ``consolidated["runs"]``.
        """
        return _DEFAULT_PROGRESS_TABLE_SPEC

    def __repr__(self) -> str:
        return "%s(framework_name=%r)" % (self.__class__.__name__, self.framework_name)


def redact_hosts(hosts) -> list[str]:
    """Return stable pseudonyms (``node-<8hex>``) for *hosts*.

    Published results (Spark Arena) must not carry a user's LAN addresses or
    internal hostnames, but "which node produced this row" is exactly the
    question :func:`host_meta` exists to answer — so the identity is
    preserved as a digest rather than dropped.  Two rows measured on the same
    node still match; nothing else is recoverable.
    """
    return ["node-%s" % hashlib.sha256(str(h).encode("utf-8")).hexdigest()[:8] for h in hosts]


def host_meta(hosts, *, redact: bool = False) -> dict[str, Any]:
    """Build the ``hosts`` / ``node_count`` half of a result's cluster block.

    Recording *which* hosts were measured is what makes a per-node comparison
    auditable after the fact: without it a table of identical numbers from
    different nodes is indistinguishable from a table of genuinely identical
    nodes (issue #267).  ``hosts_redacted`` marks the pseudonymised form so a
    consumer never mistakes a digest for a reachable address.
    """
    hosts = list(hosts or [])
    meta: dict[str, Any] = {"node_count": len(hosts)}
    if redact:
        meta["hosts"] = redact_hosts(hosts)
        meta["hosts_redacted"] = True
    else:
        meta["hosts"] = [str(h) for h in hosts]
    return meta


def _build_cluster_meta(recipe, overrides, cluster_id, host_list, *, redact: bool = False):
    """Build cluster metadata dict with only non-default parallelism values."""
    from sparkrun.core.parallelism import extract_parallelism_meta

    config_chain = recipe.build_config_chain(overrides)
    meta = {
        "cluster_id": cluster_id,
    }
    meta.update(host_meta(host_list, redact=redact))
    meta.update(extract_parallelism_meta(config_chain))
    return meta


@dataclass
class BenchmarkResult:
    """Result of a benchmark run with output file paths."""

    # benchmark results
    success: bool = False
    results: dict[str, Any] = None
    outputs: Optional[dict[str, Any]] = None
    start_time: datetime = None
    end_time: datetime = None

    # recipe/launch info
    recipe_name: Optional[str] = None
    launch_result: Optional["LaunchResult"] = None

    # store detailed recipe info for "non-launch" scenarios
    recipe: Optional["Recipe"] = None
    overrides: Optional[dict[str, Any]] = None
    cluster_id: Optional[str] = None
    host_list: Optional[list[str]] = None
    container_image: Optional[str] = None

    # benchmark info
    framework: Optional["BenchmarkingPlugin"] = None
    profile: Optional[str] = None
    benchmark_args: Optional[dict[str, Any]] = None

    # Pre-resolved long-term archival reference. When set (e.g. from a
    # persisted resume state), generate_metadata uses these instead of calling
    # builder.resolve_long_term_image again — keeps result-yaml provenance
    # identical across sessions of the same resumable run.
    longterm_image_ref: Optional[str] = None
    longterm_image_pinned: bool = False

    # Provenance of the numbers themselves.  ``timing`` records *this
    # invocation*, which for a run that re-emitted recorded results is not
    # when anything was measured — so a stale result was indistinguishable
    # from a fresh measurement in the exported artifact (issue #267).
    # ``measured_at`` carries the reused state's last-write time.
    resumed: bool = False
    measured_at: Optional[str] = None

    # Launch-stage timing.  ``readiness`` carries the two-stage
    # containers-running → serving wait; the span timeline comes off
    # ``launch_result``.  Both are absent under ``--skip-run`` (nothing was
    # launched) and are omitted from the metadata rather than zeroed — a
    # recorded 0.0s would read as an instantaneous launch.
    readiness: Optional["ServeReadiness"] = None

    @property
    def output_csv(self):
        return self.outputs.get("csv") if self.outputs else None

    @output_csv.setter
    def output_csv(self, value):
        if self.outputs is None:
            self.outputs = {}
        self.outputs["csv"] = value

    @property
    def output_json(self):
        return self.outputs.get("json") if self.outputs else None

    @output_json.setter
    def output_json(self, value):
        if self.outputs is None:
            self.outputs = {}
        self.outputs["json"] = value

    @property
    def output_yaml(self):
        return self.outputs.get("yaml") if self.outputs else None

    @output_yaml.setter
    def output_yaml(self, value):
        if self.outputs is None:
            self.outputs = {}
        self.outputs["yaml"] = value

    def _launch_timing_meta(self) -> dict[str, Any]:
        """Launch-stage timing for ``metadata["timing"]``.

        Returns ``{}`` when nothing was launched in this invocation, which
        keeps the key absent rather than present-and-zero.  A **resumed**
        run is the case that matters: its numbers come from a launch it did
        not perform (and under ``--skip-run`` from one it never saw), so
        emitting them beside freshly-measured throughput would misattribute
        a stale launch to this result — the same confusion ``measured_at``
        exists to prevent for the benchmark numbers themselves (issue #267).
        """
        if self.resumed:
            return {}

        meta: dict[str, Any] = {}
        if self.readiness is not None:
            meta["serve_ready"] = {
                "port_open_s": round(self.readiness.port_wait_s, 2),
                "health_ok_s": round(self.readiness.health_wait_s, 2),
                "total_s": round(self.readiness.total_wait_s, 2),
            }
        timeline = getattr(self.launch_result, "timeline", None)
        if timeline is not None:
            meta["launch"] = timeline.export()
        return meta

    def generate_metadata(self, *, redact_hosts: bool = True):
        """Build the provenance mapping for this result.

        *redact_hosts* pseudonymises the recorded host set.  It defaults to
        ``True`` because this mapping's consumer is the Spark Arena
        submission — a published artifact must fail closed on host identity,
        so a caller that genuinely wants real addresses has to ask.
        """
        from sparkrun.models.download import parse_gguf_model_spec
        from sparkrun.utils.cli_formatters import RUNTIME_DISPLAY as _RUNTIME_DISPLAY

        # Use launch_result fields when available, fall back to direct fields
        # (e.g. when --skip-run was used and no launch occurred).
        if launch_result := self.launch_result:
            recipe = launch_result.recipe
            overrides = launch_result.overrides
            cluster_id = launch_result.cluster_id
            host_list = launch_result.host_list
            container_image = launch_result.container_image
            runtime_info = launch_result.runtime_info
        else:
            recipe = self.recipe
            overrides = self.overrides or {}
            cluster_id = self.cluster_id
            host_list = self.host_list or []
            container_image = self.container_image
            runtime_info = {}

        framework = self.framework
        profile = self.profile
        benchmark_args = self.benchmark_args

        # Resolve container image to a pinned long-term reference when possible
        container_pinned = False
        recipe_container = container_image or recipe.container
        if self.longterm_image_ref:
            # Pre-resolved (and persisted) on first launch of this resumable run.
            recipe_container = self.longterm_image_ref
            container_pinned = self.longterm_image_pinned
        elif launch_result and launch_result.builder:
            try:
                resolved_image, pinned = launch_result.builder.resolve_long_term_image(
                    container_image=launch_result.container_image,
                    runtime_info=launch_result.runtime_info,
                    recipe=recipe,
                )
                if pinned:
                    recipe_container = resolved_image
                    container_pinned = True
                    logger.info("Pinned container image: %s", recipe_container)
            except Exception:
                logger.debug("Long-term image resolution failed", exc_info=True)

        recipe_hash = hashlib.sha256(recipe.export(overrides=None).encode("utf-8")).hexdigest()

        hf_model = parse_gguf_model_spec(recipe.model)[0]
        model_meta: dict[str, Any] = {}
        if recipe.metadata.get("model_dtype"):
            model_meta["dtype"] = recipe.metadata["model_dtype"]
        if recipe.model_revision:
            model_meta["revision"] = recipe.model_revision
        if recipe.metadata.get("model_params"):
            model_meta["params"] = recipe.metadata["model_params"]
        if recipe.metadata.get("num_layers"):
            model_meta["num_layers"] = recipe.metadata["num_layers"]
        if recipe.metadata.get("num_kv_heads"):
            model_meta["num_kv_heads"] = recipe.metadata["num_kv_heads"]
        if recipe.metadata.get("head_dim"):
            model_meta["head_dim"] = recipe.metadata["head_dim"]
        if recipe.metadata.get("quantization"):
            model_meta["quantization"] = recipe.metadata["quantization"]
        if recipe.metadata.get("quant_bits"):
            model_meta["quant_bits"] = recipe.metadata["quant_bits"]
        if recipe.metadata.get("kv_dtype"):
            model_meta["kv_dtype"] = recipe.metadata["kv_dtype"]

        metadata = {
            "recipe": {
                "name": recipe.name,
                "qualified_name": recipe.qualified_name,
                "type": "sparkrun",
                "model": recipe.model,  # will include quant if applicable
                "hf_model": hf_model,  # will exclude quant if applicable
                "raw_container": recipe.container,
                "container": recipe_container,
                "container_pinned": container_pinned,
                "runtime": _RUNTIME_DISPLAY.get(recipe.runtime, recipe.runtime),
                "runtime_full": recipe.runtime,
                "registry": recipe.source_registry,
                "registry_git": recipe.source_registry_url or "",
                "hash": recipe_hash,
            },
            "timing": {
                "start": self.start_time.isoformat(),
                "end": self.end_time.isoformat(),
                "duration": (self.end_time - self.start_time).total_seconds(),
                **self._launch_timing_meta(),
            },
            "cluster": _build_cluster_meta(recipe, overrides, cluster_id, host_list, redact=redact_hosts),
            "benchmark": {
                "framework": framework.framework_name if framework else "unknown",
                "profile": profile,
                "args": benchmark_args,
                "resumed": bool(self.resumed),
                **({"measured_at": self.measured_at} if self.measured_at else {}),
            },
            "model": model_meta,
            "runtime_info": runtime_info,
        }

        # TODO: more care for sparkrun version and/or other metadata
        # noinspection PyBroadException
        try:
            from sparkrun import __version__ as sparkrun_version

            metadata["sparkrun"] = {
                "version": sparkrun_version,
            }
        except Exception:
            pass

        return metadata


def export_results(
    *,
    recipe: Recipe,
    hosts: list[str],
    tp: int,
    cluster_id: str,
    framework_name: str,
    profile_name: str | None,
    args: dict[str, Any],
    results: dict[str, Any],
    output_path: str | Path,
    runtime_info: dict[str, str] | None = None,
    resumed: bool = False,
    measured_at: str | None = None,
) -> Path:
    """Export benchmark results to a YAML file.

    Args:
        recipe: The recipe that was benchmarked.
        hosts: Hosts used for inference.  Recorded under ``cluster.hosts`` as
            **pseudonyms** — a measurement is *of* a node set, and without any
            node identity a per-node comparison cannot be audited after the
            fact (issue #267), but this file embeds the full recipe and is the
            natural thing to attach to a bug report, so it must not carry LAN
            addresses.  The digests still distinguish node A from node B,
            which is the property the audit needs; the real host list lives in
            the local-only ``BenchmarkRunState``.
        tp: Tensor parallelism value.
        cluster_id: Cluster ID from the inference run.
        framework_name: Name of the benchmarking framework.
        profile_name: Profile name (or None if defaults used).
        args: Benchmark args that were used.
        results: Structured results from the framework.
        output_path: Path to write the YAML file.
        resumed: ``True`` when these numbers came from recorded state rather
            than from requests sent during this invocation.
        measured_at: When the recorded results were actually measured.
            ``timestamp`` below is when this file was *written*, which for a
            resumed run is not the same thing.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    # noinspection PyProtectedMember
    recipe_text = recipe.export(path=None)
    recipe_hash = hashlib.sha256(recipe_text.encode("utf-8")).hexdigest()

    # Build model metadata from recipe metadata (includes auto-detected
    # values written back by Recipe.estimate_vram).
    model_meta: dict[str, Any] = {}
    if recipe.metadata.get("model_dtype"):
        model_meta["dtype"] = recipe.metadata["model_dtype"]
    if recipe.model_revision:
        model_meta["revision"] = recipe.model_revision
    if recipe.metadata.get("model_params"):
        model_meta["params"] = recipe.metadata["model_params"]
    if recipe.metadata.get("num_layers"):
        model_meta["num_layers"] = recipe.metadata["num_layers"]
    if recipe.metadata.get("num_kv_heads"):
        model_meta["num_kv_heads"] = recipe.metadata["num_kv_heads"]
    if recipe.metadata.get("head_dim"):
        model_meta["head_dim"] = recipe.metadata["head_dim"]
    if recipe.metadata.get("quantization"):
        model_meta["quantization"] = recipe.metadata["quantization"]
    if recipe.metadata.get("quant_bits"):
        model_meta["quant_bits"] = recipe.metadata["quant_bits"]
    if recipe.metadata.get("kv_dtype"):
        model_meta["kv_dtype"] = recipe.metadata["kv_dtype"]

    data = {
        "sparkrun_benchmark": {
            "version": "1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recipe": {
                "name": recipe.name,
                "qualified_name": recipe.qualified_name,
                "type": "sparkrun",
                "model": recipe.model,
                "container": recipe.container,
                "runtime": recipe.runtime,
                "registry": recipe.source_registry,
                "registry_git": recipe.source_registry_url or "",
                "text": recipe_text,
                "hash": recipe_hash,
            },
            "model": model_meta,
            "cluster": {
                "tp": tp,
                "cluster_id": cluster_id,
                **host_meta(hosts, redact=True),
                "runtime_info": runtime_info or {},
            },
            "benchmark": {
                "framework": framework_name,
                "profile": profile_name,
                "args": args,
                "resumed": bool(resumed),
                **({"measured_at": measured_at} if measured_at else {}),
            },
            "results": results,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)

    logger.info("Results exported to %s", output_path)
    return output_path
