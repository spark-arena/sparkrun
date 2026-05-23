"""Centralized parallelism configuration and extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParallelismConfig:
    """Parallelism dimensions for an inference workload.

    Carries both the recipe-extracted parallelism dimensions
    (:attr:`tensor_parallel` etc.) and an optional runtime-derived
    :attr:`total_ranks` override.  Callers asking "how many ranks does
    the scheduler pack?" should call :meth:`world_size` — which returns
    the override when set, else falls back to the :attr:`total_gpus`
    formula.

    The override is produced by ``Runtime.world_size(...)`` and baked
    in via ``dataclasses.replace(parallelism, total_ranks=N)`` before
    the config is handed to the scheduler.  Runtimes whose semantics
    differ from the default formula (e.g., Atlas's ``tp * ep`` MoE
    mesh) use this mechanism to express their actual slot count.
    """

    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    expert_parallel: int = 1
    context_parallel: int = 1
    total_ranks: int | None = None
    """Runtime-derived rank-count override.  ``None`` (default) means
    "use the :attr:`total_gpus` formula".  Set via
    :func:`dataclasses.replace` after consulting
    ``Runtime.world_size(...)``."""

    @property
    def total_gpus(self) -> int:
        """Total GPU count from the standard formula = tp * pp * dp.

        Each DP replica uses ``tp * pp`` GPUs; with ``dp`` replicas the
        full job consumes ``tp * pp * dp`` GPUs.  This is the formula
        only — for the canonical "ranks the scheduler packs" call
        :meth:`world_size`, which honours runtime-derived overrides.
        """
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel

    def world_size(self) -> int:
        """Canonical rank count to schedule.

        Returns :attr:`total_ranks` when explicitly set (runtime
        adjustment baked in); otherwise falls back to the
        :attr:`total_gpus` formula.  Higher layers (scheduler, launcher)
        should ask this method rather than reading :attr:`total_gpus`
        directly so runtime-specific math (e.g., Atlas's MoE mesh) is
        honoured uniformly.
        """
        return self.total_ranks if self.total_ranks is not None else self.total_gpus

    @property
    def model_shard_factor(self) -> int:
        """Factor by which model weights are sharded across GPUs = tp * pp.

        Also the GPU count for a *single* DP replica.  Use this (not
        :attr:`total_gpus`) for VRAM-per-GPU estimation.
        """
        return self.tensor_parallel * self.pipeline_parallel

    @property
    def total_nodes(self) -> int:
        """Total nodes required *assuming 1 GPU per host* = tp * pp * dp.

        .. deprecated:: phase-2
            Equivalent to :attr:`total_gpus`; correct only for clusters
            where every host has exactly one accelerator (e.g. DGX Spark).
            For multi-GPU hosts or heterogeneous clusters use
            :func:`sparkrun.core.placement.compute_placement` and read
            ``len(assignment.hosts_used)``.
        """
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel


# Canonical list of parallelism keys and their short aliases.
PARALLELISM_KEYS: list[tuple[str, str]] = [
    ("tensor_parallel", "tp"),
    ("pipeline_parallel", "pp"),
    ("data_parallel", "dp"),
    ("expert_parallel", "ep"),
    ("context_parallel", "cp"),
]

_FIELD_NAMES = {long for long, _ in PARALLELISM_KEYS}


def extract_parallelism(config_chain: Any) -> ParallelismConfig:
    """Extract parallelism values from a config chain (SAF Variables, dict, etc.).

    Accepts anything with a ``.get(key)`` method (SAF Variables, plain dict, etc.).
    Missing or ``None`` values default to 1.
    """
    kwargs: dict[str, int] = {}
    for long_key, _ in PARALLELISM_KEYS:
        val = config_chain.get(long_key)
        if val is not None:
            kwargs[long_key] = int(val)
    return ParallelismConfig(**kwargs)


def extract_parallelism_meta(config_chain: Any) -> dict[str, int]:
    """Build metadata dict with only non-default parallelism values (short keys).

    Returns e.g. ``{"tp": 2, "pp": 2}`` — omits dimensions that are 1.
    """
    meta: dict[str, int] = {}
    for long_key, short_key in PARALLELISM_KEYS:
        val = config_chain.get(long_key)
        if val is not None:
            int_val = int(val)
            if int_val != 1:
                meta[short_key] = int_val
    return meta
