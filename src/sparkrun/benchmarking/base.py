"""Benchmarking plugin base class, spec loading, and result export."""

from __future__ import annotations

from collections.abc import Mapping

from sparkrun.benchmarking.metadata import public_benchmark_data, model_metadata, public_recipe_text

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Any, Callable, ClassVar, TYPE_CHECKING, Optional

import yaml
from scitrera_app_framework import Plugin, Variables

from sparkrun.core.bootstrap import EXT_BENCHMARKING_FRAMEWORKS

if TYPE_CHECKING:
    from sparkrun.api._models import RunResult
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


class BenchmarkingPlugin(Plugin, ABC):
    """Abstract base for benchmarking frameworks (SAF multi-extension plugin).

    Mirrors :class:`~sparkrun.runtimes.base.RuntimePlugin` in structure.
    Installed modules declare a ``sparkrun.plugins`` entry point; its loader
    scans their concrete subclasses. ``sparkrun.benchmarking`` is the internal
    SAF extension point, not an installed-package entry-point group.
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

    # What the ``model`` argument of ``build_benchmark_command`` *means* to this
    # framework.  See :func:`resolve_request_model` — the default is the name
    # the endpoint answers to, because that is what an OpenAI-compatible client
    # has to send.  A framework whose ``--model`` identifies the *weights*
    # (llama-benchy, which tokenizes locally and takes the request name
    # separately) declares ``True``.
    model_argument_is_model_id: ClassVar[bool] = False

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
            args: Detached measurement args, with ephemeral api_key when authenticated.
                Do not persist or log the credential; task definitions omit it.
            result_file: Path where the framework should save results.

        Returns:
            Raw command argv suitable for subprocess, without shell quoting.
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
            Structured results dict. API results support string-keyed mappings,
            lists/tuples, and primitive scalars. Date/datetime and pathlib paths
            are normalized to strings before export, persistence, and hooks;
            other application objects are rejected at the framework boundary.
        """
        ...

    def get_default_args(self) -> dict[str, Any]:
        """Return default benchmark args when no profile is provided."""
        return dict(self.default_args)

    def prepare_benchmark_args(
        self,
        recipe: "Recipe",
        config_chain: Variables | Mapping[str, Any],
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

    @abstractmethod
    def build_task_list(
        self,
        base_args: dict[str, Any],
        schedule: list[dict[str, Any]] | None,
    ) -> list["BenchTask"]:
        """Build a nonempty, ordered task list, with contiguous zero-based indices.

        Frameworks implement this directly, including those that run one task.
        Persist each task's overrides in ``schedule_entry`` so the same list
        can be reconstructed on resume. Invalid schedules raise BenchmarkError.
        """
        raise NotImplementedError("Benchmark frameworks must implement build_task_list")

    # --- Scheduler / aggregator hooks (optional, with safe defaults) -----

    def result_filename_suffix(self, task: "BenchTask") -> str:
        """Per-task suffix appended after the index in scheduler artifact filenames.

        The scheduler writes ``{idx:03d}{suffix}.json`` / ``{idx:03d}{suffix}.log``.
        Default: empty string (use index-only filenames). The suffix must be
        deterministic for the saved task across resumes. The host invalidates
        a pending task's previous output before command construction.
        """
        return ""

    def consolidate_per_task_results(self, per_task_jsons: list[dict[str, Any]]) -> dict[str, Any]:
        """Consolidate per-task JSON dicts into a single framework-shaped dict.

        Inputs contain only usable JSON objects from successful task attempts,
        in schedule order. Failed/interrupted artifacts are excluded.
        Default: ``{"runs": list(per_task_jsons)}``.
        """
        return {"runs": list(per_task_jsons)}

    def task_coverage_key(self, task: "BenchTask") -> Any:
        """Identifier used to detect whether a task is represented in consolidated results.

        Default: the task's index.
        """
        return task.index

    def consolidated_coverage_keys(self, consolidated: dict[str, Any]) -> set[Any] | None:
        """Return the set of coverage keys present in the consolidated dict.

        Default: ``None`` — successful task artifacts provide coverage. The
        scheduler retains their original indices even when failures leave gaps.
        Override together with ``task_coverage_key`` to require framework-specific
        measurements in addition to successful task execution. An empty set means
        no measurements are covered, whereas ``None`` adds no semantic check.
        """
        return None

    def progress_table_spec(self) -> ProgressTableSpec:
        """Return the column spec / row generator the progress UI uses for live display.

        Default: a minimal one-column ``idx`` table populated from
        ``consolidated["runs"]``.
        """
        return _DEFAULT_PROGRESS_TABLE_SPEC

    def __repr__(self) -> str:
        return "%s(framework_name=%r)" % (self.__class__.__name__, self.framework_name)


def resolve_request_model(framework: BenchmarkingPlugin, recipe: "Recipe", config_chain: Variables | Mapping[str, Any]) -> str:
    """The ``model`` value handed to ``build_benchmark_command``.

    Two different things are called "the model" and a benchmark framework wants
    exactly one of them.  ``recipe.model`` identifies the *weights* — an HF repo
    id, the thing a local tokenizer is loaded from.  The served name is what the
    endpoint answers to, which is what goes in the ``"model"`` field of every
    request.  They differ whenever a recipe declares ``served_model_name``, and
    sending the repo id to a server that answers only to the alias is a 404 on
    every request.

    sparkrun passed ``recipe.model`` to every framework, which is right for
    llama-benchy (its ``--model`` is the tokenizer identity and the request name
    is a separate ``--served-model-name``, issue #257) and wrong for anything
    whose ``--model`` is simply the name it sends — tool-eval-bench, and by
    default any framework written against an OpenAI-compatible endpoint
    (issue #298).  So the **served name is the default** and the model id is the
    declared exception: the two coincide for every recipe without an alias, and
    where they don't, defaulting to the id fails every request rather than at
    worst mislabelling a tokenizer.

    The ``command:`` fallback inside :func:`resolve_served_model_name` matters
    here for the same reason it does everywhere else: a recipe that hardcodes
    ``--served-model-name`` in its template is invisible to the config chain.
    """
    from sparkrun.core.recipe import resolve_served_model_name
    from sparkrun.models.download import parse_gguf_model_spec

    if getattr(framework, "model_argument_is_model_id", False):
        return recipe.model
    name = resolve_served_model_name(recipe, config_chain.get("served_model_name"))
    # Only the model-id fallback can carry sparkrun's GGUF ``repo:quant``
    # suffix, which no server answers to. A *declared* alias is a free-form
    # name and may legitimately contain a colon ("qwen3:8b"), so it is never
    # split.
    return parse_gguf_model_spec(name)[0] if name == recipe.model else name


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


def startup_timing_metadata(readiness: ServeReadiness | None, *, resumed: bool = False) -> dict[str, Any]:
    """Project a fresh, successful rank-local observation into public metadata.

    Never substitute endpoint wait durations for Docker-start measurements.
    Allowlist provenance fields: the raw observation may contain prompts,
    responses, container names, or future private extensions.
    """
    if resumed or readiness is None or readiness.ready is not True:
        return {}
    observation = readiness.startup_observation
    if not isinstance(observation, dict) or not observation:
        return {}
    from sparkrun.orchestration.startup import validate_observation

    try:
        observation = validate_observation(observation, normalize=True)
    except (TypeError, ValueError):
        logger.warning("Omitting invalid benchmark startup observation")
        return {}

    start = observation["container_started_unix_ns"]
    inference = observation.get("inference_requested") is not False
    meta: dict[str, Any] = {
        "format": 1,
        "measurement": observation["measurement"],
        "observer": "rank0",
        "start_boundary": observation["start_boundary"],
        "executor": observation["executor"],
        "observer_location": observation["observer_location"],
        "inference_style": observation["inference_style"],
        "ttft_status": "measured" if inference else "not_applicable",
        "inference_requested": inference,
        "inference_ready": observation["inference_ready"],
    }
    for source, target in (
        ("port_open_unix_ns", "ttr_port_open_s"),
        ("http_ready_unix_ns", "ttr_http_ready_s"),
        ("first_token_unix_ns", "ttft_s"),
    ):
        if source in observation:
            meta[target] = round((observation[source] - start) / 1e9, 6)
    observer_start = observation.get("observer_started_unix_ns")
    if type(observer_start) is int and observer_start >= start:
        meta["observer_start_delay_s"] = round((observer_start - start) / 1e9, 6)
    if inference:
        meta["first_token_field"] = observation["first_token_field"]
    if type(observation.get("response_validated")) is bool:
        meta["response_validated"] = observation["response_validated"]
    if observation.get("http_ready_path") in ("/health", "/v1/models"):
        meta["http_ready_path"] = observation["http_ready_path"]
    if inference:
        prompt_hash = observation.get("prompt_sha256")
        if isinstance(prompt_hash, str) and len(prompt_hash) == 64 and all(c in "0123456789abcdef" for c in prompt_hash):
            meta["prompt_sha256"] = prompt_hash
        max_tokens = observation.get("max_tokens")
        if type(max_tokens) is int and max_tokens > 0:
            meta["max_tokens"] = max_tokens
        for source, target in (("temperature", "temperature"), ("request_ttft_seconds", "request_ttft_s")):
            value = observation.get(source)
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                meta[target] = value
    return meta


@dataclass
class BenchmarkExecution:
    """Mutable orchestration record, distinct from public API BenchmarkResult.

    Integrations receive a detached BenchmarkMeasurement snapshot. Recipe,
    framework and launch_result can be absent during a completed-measurement
    publication retry. This record remains private orchestration state.
    """

    # benchmark results
    success: bool = False
    results: dict[str, Any] = field(default_factory=dict)
    outputs: Optional[dict[str, Any]] = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    # recipe/launch info
    recipe_name: Optional[str] = None
    launch_result: Optional["LaunchResult"] = None

    # store detailed recipe info for "non-launch" scenarios
    recipe: Optional["Recipe"] = None
    overrides: Optional[dict[str, Any]] = None
    cluster_id: Optional[str] = None
    host_list: Optional[list[str]] = None
    container_image: Optional[str] = None
    image_context_known: bool = False
    runtime_info: dict[str, Any] = field(default_factory=dict)

    benchmark_id: str = ""
    state_dir: Optional[str] = None
    integration_results: Optional[dict[str, dict[str, Any]]] = None

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
    # ``measured_at`` is the first measurement time (a pinned fallback for legacy state).
    resumed: bool = False
    already_complete: bool = False
    measured_at: Optional[str] = None
    measurement_completed_at: Optional[str] = None

    # Launch-stage timing. ``readiness`` carries endpoint wait durations and,
    # when supported, a separate Docker-start observation. The span timeline
    # comes off ``launch_result``. Both are absent under ``--skip-run`` (nothing was
    # launched) and are omitted from the metadata rather than zeroed — a
    # recorded 0.0s would read as an instantaneous launch.
    readiness: Optional["ServeReadiness"] = None

    run_result: Optional["RunResult"] = None
    framework_name: str = ""
    category: str = ""
    integration_errors: dict[str, str] = field(default_factory=dict)
    container_image_sha: str | None = None
    container_image_sha_pinned: bool = False

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
        if startup := startup_timing_metadata(self.readiness):
            meta["startup"] = startup
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

    def recipe_provenance(self, *, resolve_image: bool = False) -> dict[str, Any]:
        """One effective recipe projection; hashes distinguish declared/effective input."""
        launch_result = self.launch_result
        recipe = self.recipe or (launch_result.recipe if launch_result else None)
        if recipe is None:
            raise ValueError("Benchmark recipe provenance requires a captured recipe")
        overrides = self.overrides if self.overrides is not None else (launch_result.overrides if launch_result else {})
        container_image = self.container_image
        if not self.image_context_known and launch_result:
            container_image = launch_result.container_image
        # Resolve container image to a pinned long-term reference when possible
        container_pinned = False
        recipe_container = container_image
        if not recipe_container and not self.resumed and not self.image_context_known:
            recipe_container = recipe.container
        if self.longterm_image_ref:
            # Pre-resolved (and persisted) on first launch of this resumable run.
            recipe_container = self.longterm_image_ref
            container_pinned = self.longterm_image_pinned
        elif self.container_image_sha:
            recipe_container = self.container_image_sha
            container_pinned = self.container_image_sha_pinned
        elif resolve_image and launch_result and launch_result.builder and launch_result.container_image:
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

        declared = recipe.export(overrides=None)
        if not isinstance(declared, str):
            raise ValueError("Benchmark recipe export must return text when no path is supplied")
        declared_text = public_recipe_text(declared)
        effective = recipe.to_dict(overrides=overrides, effective=True)
        # None is explicit unknown here; Recipe.export(None) would reuse the declaration.
        effective["container"] = recipe_container
        text = public_recipe_text(yaml.safe_dump(effective, sort_keys=False))
        return {
            "raw_container": recipe.container,
            "container": recipe_container,
            "container_pinned": container_pinned,
            "text": text,
            "hash": hashlib.sha256(declared_text.encode("utf-8")).hexdigest(),
            "effective_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    def generate_metadata(self, *, redact_hosts: bool = True, resolve_image: bool = True) -> dict[str, Any]:
        """Build the provenance mapping for this result.

        *redact_hosts* pseudonymises the recorded host set.  It defaults to
        ``True`` because this mapping's consumer is the Spark Arena
        submission — a published artifact must fail closed on host identity,
        so callers wanting real addresses must ask. ``resolve_image=False``
        builds provenance solely from recorded values and does no image
        resolution; plugin snapshots use that mode. Legacy direct callers may
        retain best-effort builder resolution with the default True.
        """
        from sparkrun.models.download import parse_gguf_model_spec
        from sparkrun.utils.runtime_display import RUNTIME_DISPLAY as _RUNTIME_DISPLAY

        # Recorded measurement context stays authoritative on resumed runs.
        # Only legacy callers without captured context use the launch fallback.
        if (launch_result := self.launch_result) and not self.resumed and not self.image_context_known:
            recipe = launch_result.recipe
            overrides = launch_result.overrides
            cluster_id = launch_result.cluster_id
            host_list = launch_result.host_list
            runtime_info = launch_result.runtime_info
        else:
            recipe = self.recipe
            overrides = self.overrides or {}
            cluster_id = self.cluster_id
            host_list = self.host_list or []
            runtime_info = self.runtime_info

        if recipe is None:
            raise ValueError("Benchmark metadata requires a captured recipe")

        framework = self.framework
        profile = self.profile
        benchmark_args = self.benchmark_args

        recipe_projection = self.recipe_provenance(resolve_image=resolve_image)
        hf_model = parse_gguf_model_spec(recipe.model)[0]
        model_meta = model_metadata(recipe)

        metadata = {
            "recipe": {
                "name": recipe.name,
                "qualified_name": recipe.qualified_name,
                "type": "sparkrun",
                "model": recipe.model,  # will include quant if applicable
                "hf_model": hf_model,  # will exclude quant if applicable
                **recipe_projection,
                "runtime": _RUNTIME_DISPLAY.get(recipe.runtime, recipe.runtime),
                "runtime_full": recipe.runtime,
                "registry": recipe.source_registry,
                "registry_git": recipe.source_registry_url or "",
            },
            "timing": {
                "start": self.start_time.isoformat() if self.start_time else None,
                "end": self.end_time.isoformat() if self.end_time else None,
                "duration": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
                **self._launch_timing_meta(),
            },
            "cluster": _build_cluster_meta(recipe, overrides, cluster_id, host_list, redact=redact_hosts),
            "benchmark": {
                "framework": self.framework_name or (framework.framework_name if framework else "unknown"),
                "profile": profile,
                "args": benchmark_args,
                "resumed": bool(self.resumed),
                **({"measured_at": self.measured_at} if self.measured_at else {}),
                **({"completed_at": self.measurement_completed_at} if self.measurement_completed_at else {}),
                "category": self.category,
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

        return public_benchmark_data(metadata)


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
    readiness: ServeReadiness | None = None,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Export benchmark results to a YAML file.

    Args:
        recipe: The recipe that was benchmarked.
        overrides: Effective serving overrides to include in the exported recipe.
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
        readiness: Successful readiness from this launch, with optional
            Docker-start TTR/TTFT. Omitted for resumed results and absent
            observations; legacy endpoint wait durations are not substituted.

    Returns:
        Path to the written file.
    """
    execution = BenchmarkExecution(
        recipe=recipe,
        host_list=hosts,
        cluster_id=cluster_id,
        framework_name=framework_name,
        profile=profile_name,
        benchmark_args=args,
        results=results,
        runtime_info=runtime_info or {},
        resumed=resumed,
        measured_at=measured_at,
        readiness=readiness,
        overrides=overrides,
        container_image=recipe.container,
    )
    return _write_measurement(execution, tp=tp, output_path=output_path)


def _write_measurement(execution: BenchmarkExecution, *, tp: int, output_path: str | Path) -> Path:
    """Write the same effective description supplied to integration snapshots."""
    output_path = Path(output_path)
    metadata = execution.generate_metadata(resolve_image=False)
    recipe = dict(metadata["recipe"])
    recipe["declared_hash"] = recipe["hash"]
    recipe["hash"] = recipe.pop("effective_hash")
    recipe["runtime"] = recipe.pop("runtime_full")
    data = {
        "sparkrun_benchmark": {
            "version": "1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recipe": recipe,
            "model": metadata["model"],
            "cluster": {
                "tp": tp,
                "cluster_id": execution.cluster_id,
                **host_meta(execution.host_list or [], redact=True),
                "runtime_info": metadata["runtime_info"],
            },
            "benchmark": {**metadata["benchmark"], "framework": execution.framework_name},
            "results": execution.results,
        },
    }
    if startup := startup_timing_metadata(execution.readiness, resumed=execution.resumed):
        data["sparkrun_benchmark"]["timing"] = {"startup": startup}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(public_benchmark_data(data), f, default_flow_style=False, sort_keys=False, indent=2)
    logger.info("Results exported to %s", output_path)
    return output_path
