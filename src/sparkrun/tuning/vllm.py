"""vLLM fused MoE kernel tuning for DGX Spark.

Launches a container, clones vLLM benchmark scripts, and runs Triton
kernel tuning for each requested TP size.  Results are saved to the host
and auto-mounted in future ``sparkrun run`` invocations.
"""

from __future__ import annotations

from pathlib import Path

from sparkrun.tuning._common import (
    BaseTuner,
    DEFAULT_TP_SIZES,  # noqa: F401 — re-exported for public API
    _format_duration,  # noqa: F401 — re-exported for public API
    _get_tuning_dir,
    _get_tuning_env,
    _get_tuning_volumes,
)
from sparkrun.utils.shell import quote

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VLLM_TUNING_CACHE_SUBDIR = "tuning/vllm"
VLLM_TUNING_CONTAINER_PATH = "/tuning/vllm"
VLLM_TUNING_CONTAINER_OUTPUT_PATH = "/tuning_output"

TUNE_VLLM_CONTAINER_NAME = "sparkrun_tune_vllm"
VLLM_CLONE_DIR = "/tmp/vllm_src"


# ---------------------------------------------------------------------------
# Host-side helpers (used by runtimes for auto-mounting)
# ---------------------------------------------------------------------------


def get_vllm_tuning_dir() -> Path:
    """Return the host-side directory for vLLM tuning configs.

    Default: ``~/.cache/sparkrun/tuning/vllm/``
    """
    return _get_tuning_dir(VLLM_TUNING_CACHE_SUBDIR)


def get_vllm_tuning_volumes() -> dict[str, str] | None:
    """Return volume mapping for vLLM tuning configs if they exist.

    Returns:
        Dict mapping host dir to container dir, or ``None`` if no
        tuning configs are available.
    """
    return _get_tuning_volumes(get_vllm_tuning_dir, VLLM_TUNING_CONTAINER_PATH)


def get_vllm_tuning_env() -> dict[str, str] | None:
    """Return env vars for vLLM tuning configs if they exist.

    Returns:
        Dict with ``VLLM_TUNED_CONFIG_FOLDER`` set, or ``None`` if no
        tuning configs are available.
    """
    return _get_tuning_env(
        get_vllm_tuning_volumes,
        "VLLM_TUNED_CONFIG_FOLDER",
        VLLM_TUNING_CONTAINER_PATH,
    )


# ---------------------------------------------------------------------------
# VllmTuner
# ---------------------------------------------------------------------------


class VllmTuner(BaseTuner):
    """Orchestrates vLLM fused MoE kernel tuning on a single host.

    Args:
        host: Target host for tuning.
        image: Container image to use.
        model: Model name (HuggingFace repo ID).
        config: SparkrunConfig for SSH settings.
        cache_dir: HuggingFace cache directory.
        output_dir: Override for tuning output directory on host.
        skip_clone: Skip cloning vLLM repo (scripts already in image).
        dry_run: Show commands without executing.
    """

    runtime_label = "vLLM"
    container_name = TUNE_VLLM_CONTAINER_NAME
    output_path = VLLM_TUNING_CONTAINER_OUTPUT_PATH
    clone_script = "vllm_clone_benchmarks.sh"

    def _default_output_dir(self) -> Path:
        return get_vllm_tuning_dir()

    def _build_tune_command(self, tp_size: int, triton_version: str) -> str:
        return build_vllm_tuning_command(self.model, tp_size)


def build_vllm_tuning_command(model: str, tp_size: int) -> str:
    """Build the vLLM tuning command string for display/testing.

    Args:
        model: Model name.
        tp_size: Tensor parallel size.

    Returns:
        The tuning command string.

    Note:
        vLLM's ``benchmark_moe.py`` accepts ``--save-dir`` to control
        where tuning JSON files are written (defaults to ``"./"``).
        We point it at the mounted output directory so configs survive
        container cleanup.
    """
    config_dir = VLLM_TUNING_CONTAINER_OUTPUT_PATH
    return (
        "cd %s && "
        "mkdir -p %s && "
        "VLLM_TUNED_CONFIG_FOLDER=%s "
        "python3 benchmarks/kernels/benchmark_moe.py "
        "--model %s --tp-size %d --tune --save-dir %s"
    ) % (VLLM_CLONE_DIR, config_dir, config_dir, quote(model), tp_size, config_dir)
