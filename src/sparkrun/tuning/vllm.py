"""vLLM fused MoE kernel tuning for DGX Spark.

Launches a container, clones vLLM benchmark scripts, and runs Triton
kernel tuning for each requested TP size.  Results are saved to the host
and auto-mounted in future ``sparkrun run`` invocations.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from sparkrun.tuning._common import (
    BaseTuner,
    DEFAULT_TP_SIZES,  # noqa: F401 — re-exported for public API
    _format_duration,  # noqa: F401 — re-exported for public API
    _get_tuning_dir,
    _get_tuning_env,
    _get_tuning_volumes,
)

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
        get_vllm_tuning_volumes, "VLLM_TUNED_CONFIG_FOLDER", VLLM_TUNING_CONTAINER_PATH,
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

    def _pre_check_tp(self, tp_size: int, triton_version: str) -> bool:
        """Check if vLLM tuning configs already exist for this TP size.

        Runs a lightweight script inside the container that loads the model
        config to determine MoE shape params (E, N), then checks whether
        matching config files already exist in the output directory.

        Returns ``True`` if configs exist (skip tuning), ``False`` otherwise.
        On any error, returns ``False`` (safe default — tune anyway).
        """
        import logging
        from sparkrun.orchestration.primitives import run_command_on_host
        from sparkrun.orchestration.docker import docker_exec_cmd

        logger = logging.getLogger(__name__)

        if self.dry_run:
            return False

        config_dir = VLLM_TUNING_CONTAINER_OUTPUT_PATH

        check_script = (
            "python3 -c \""
            "import sys, os, glob; "
            "from transformers import AutoConfig; "
            "c = AutoConfig.from_pretrained('%s', trust_remote_code=True); "
            "E = getattr(c, 'num_local_experts', getattr(c, 'num_experts', 0)); "
            "I = getattr(c, 'intermediate_size', getattr(c, 'moe_intermediate_size', 0)); "
            "N = (I * 2) // %d; "
            "pattern = os.path.join('%s', 'E=%%d,N=%%d,*' %% (E, N)); "
            "matches = glob.glob(pattern); "
            "sys.exit(0 if matches else 1)"
            "\""
        ) % (self.model, tp_size, config_dir)

        exec_cmd = docker_exec_cmd(self.container_name, check_script)
        try:
            result = run_command_on_host(
                self.host, exec_cmd,
                ssh_kwargs=self.ssh_kwargs, timeout=60, dry_run=False,
            )
            return result.success
        except Exception:
            logger.debug("Pre-check failed for TP=%d, will proceed with tuning", tp_size)
            return False

    def _run_tune_for_tp(self, tp_size: int, triton_version: str) -> int:
        """Step 4 (per-TP): Run the tuning script for a given TP size."""
        import logging
        import time
        from sparkrun.orchestration.primitives import run_command_on_host
        from sparkrun.orchestration.docker import docker_exec_cmd

        logger = logging.getLogger(__name__)
        t0 = time.monotonic()

        tune_cmd = build_vllm_tuning_command(self.model, tp_size)

        exec_cmd = docker_exec_cmd(self.container_name, tune_cmd)

        # Tuning can take many hours (e.g. 4+ hours for TP=4 on large
        # models).  Use an 8-hour timeout so remote SSH sessions aren't
        # killed prematurely.
        result = run_command_on_host(
            self.host, exec_cmd,
            ssh_kwargs=self.ssh_kwargs, timeout=28800, dry_run=self.dry_run,
        )

        if self.dry_run:
            logger.info("  [dry-run] Would run tuning for TP=%d", tp_size)
            return 0

        elapsed = time.monotonic() - t0
        if not result.success:
            logger.error(
                "  Tuning for TP=%d failed (exit %d, %.1fs)",
                tp_size, result.returncode, elapsed,
            )
            if result.stdout and result.stdout.strip():
                logger.error("  stdout:\n%s", result.stdout.rstrip())
            if result.stderr and result.stderr.strip():
                logger.error("  stderr:\n%s", result.stderr.rstrip())
            return result.returncode

        logger.info("  TP=%d tuning complete (%.1fs)", tp_size, elapsed)
        return 0


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
    ) % (VLLM_CLONE_DIR, config_dir, config_dir, shlex.quote(model), tp_size, config_dir)
