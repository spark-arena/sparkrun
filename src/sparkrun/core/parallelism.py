"""Centralized parallelism configuration and extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParallelismConfig:
    """Parallelism dimensions for an inference workload."""

    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    data_parallel: int = 1
    expert_parallel: int = 1
    context_parallel: int = 1

    @property
    def total_gpus(self) -> int:
        """Total GPU count across the whole job = tp * pp * dp.

        Each DP replica uses ``tp * pp`` GPUs; with ``dp`` replicas the full
        job consumes ``tp * pp * dp`` GPUs.  On DGX Spark (1 GPU per node)
        this equals :attr:`total_nodes`.
        """
        return self.tensor_parallel * self.pipeline_parallel * self.data_parallel

    @property
    def model_shard_factor(self) -> int:
        """Factor by which model weights are sharded across GPUs = tp * pp.

        Also the GPU count for a *single* DP replica.  Use this (not
        :attr:`total_gpus`) for VRAM-per-GPU estimation.
        """
        return self.tensor_parallel * self.pipeline_parallel

    @property
    def total_nodes(self) -> int:
        """Total nodes required on DGX Spark (1 GPU per node) = tp * pp * dp."""
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
