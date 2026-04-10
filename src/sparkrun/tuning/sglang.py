"""SGLang fused MoE kernel tuning for DGX Spark.

Launches a container, clones SGLang benchmark scripts, and runs Triton
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

TUNING_CACHE_SUBDIR = "tuning/sglang"
TUNING_CONTAINER_PATH = "/tuning/sglang/configs"
TUNING_ENV_PATH = "/tuning/sglang"
TUNING_CONTAINER_OUTPUT_PATH = "/tuning_output"

TUNE_CONTAINER_NAME = "sparkrun_tune"
SGLANG_CLONE_DIR = "/tmp/sglang_src"


# ---------------------------------------------------------------------------
# Host-side helpers (used by runtimes for auto-mounting)
# ---------------------------------------------------------------------------


def get_sglang_tuning_dir() -> Path:
    """Return the host-side directory for SGLang tuning configs.

    Default: ``~/.cache/sparkrun/tuning/sglang/``
    """
    return _get_tuning_dir(TUNING_CACHE_SUBDIR)


def get_sglang_tuning_volumes() -> dict[str, str] | None:
    """Return volume mapping for tuning configs if they exist.

    Returns:
        Dict mapping host dir to container dir, or ``None`` if no
        tuning configs are available.
    """
    return _get_tuning_volumes(get_sglang_tuning_dir, TUNING_CONTAINER_PATH)


def get_sglang_tuning_env() -> dict[str, str] | None:
    """Return env vars for tuning configs if they exist.

    The env var points to the *parent* of the mount target so that
    SGLang's internal ``$SGLANG_MOE_CONFIG_DIR/configs/triton_X_Y_Z/``
    lookup resolves correctly.

    Returns:
        Dict with ``SGLANG_MOE_CONFIG_DIR`` set, or ``None`` if no
        tuning configs are available.
    """
    return _get_tuning_env(
        get_sglang_tuning_volumes,
        "SGLANG_MOE_CONFIG_DIR",
        TUNING_ENV_PATH,
    )


# ---------------------------------------------------------------------------
# SglangTuner
# ---------------------------------------------------------------------------


class SglangTuner(BaseTuner):
    """Orchestrates SGLang fused MoE kernel tuning on a single host.

    Args:
        host: Target host for tuning.
        image: Container image to use.
        model: Model name (HuggingFace repo ID).
        config: SparkrunConfig for SSH settings.
        cache_dir: HuggingFace cache directory.
        output_dir: Override for tuning output directory on host.
        skip_clone: Skip cloning SGLang repo (scripts already in image).
        dry_run: Show commands without executing.
    """

    runtime_label = "SGLang"
    container_name = TUNE_CONTAINER_NAME
    output_path = TUNING_CONTAINER_OUTPUT_PATH
    clone_script = "sglang_clone_benchmarks.sh"

    def _default_output_dir(self) -> Path:
        return get_sglang_tuning_dir()

    def _apply_patches(self) -> None:
        """Patch known issues in cloned SGLang benchmark scripts.

        Pipes ``sglang_patch_common_utils.py`` into the container to fix:

        * ``config.architectures`` being ``None`` (Qwen3.5 MoE /
          transformers >= 5.x).
        * MoE attribute name variations (``num_experts`` vs
          ``num_local_experts`` etc.) across model families.
        """
        import logging
        from sparkrun.scripts import read_script
        from sparkrun.orchestration.primitives import run_script_on_host

        logger = logging.getLogger(__name__)

        patch_py = read_script("sglang_patch_common_utils.py")

        # Pipe the Python patch script into docker exec -i via heredoc
        # to avoid all shell quoting issues.
        patch_script = ("#!/bin/bash\ndocker exec -i %s python3 << 'PYEOF'\n%s\nPYEOF\n") % (self.container_name, patch_py)

        result = run_script_on_host(
            self.host,
            patch_script,
            ssh_kwargs=self.ssh_kwargs,
            timeout=15,
            dry_run=self.dry_run,
        )
        if result.success or self.dry_run:
            logger.debug("  Patched common_utils.py for MoE config compatibility")
        else:
            logger.debug("  Patch skipped (file may not need it): %s", result.stderr[:100])

    def _pre_check_output_dir(self, tp_size: int, triton_version: str) -> str:
        """SGLang uses versioned subdirectories for tuning configs."""
        config_dir = TUNING_CONTAINER_OUTPUT_PATH
        if triton_version and triton_version != "unknown":
            versioned = "triton_%s" % triton_version.replace(".", "_")
            return "%s/%s" % (config_dir, versioned)
        return config_dir

    def _build_tune_command(self, tp_size: int, triton_version: str) -> str:
        return build_tuning_command(self.model, tp_size, triton_version=triton_version)


def build_tuning_command(model: str, tp_size: int, triton_version: str | None = None) -> str:
    """Build the tuning command string for display/testing.

    Args:
        model: Model name.
        tp_size: Tensor parallel size.
        triton_version: Triton version string (e.g. ``"3.6.0"``).  When
            provided, the versioned config subdirectory is pre-created so
            SGLang's ``save_configs()`` can write into it (it doesn't create
            directories itself).

    Returns:
        The tuning command string.

    Note:
        SGLang's ``get_config_file_name()`` returns only the basename
        (e.g. ``E=64,N=2560,...json``).  The benchmark ``save_configs()``
        writes to that basename via ``open(filename, "w")``, which
        resolves relative to the working directory.  We therefore ``cd``
        into the versioned output subdirectory so the JSON files land in
        the mounted volume rather than being lost inside the container.

        ``SGLANG_MOE_CONFIG_DIR`` is still set for any runtime code that
        reads configs via the standard loader.
    """
    config_dir = TUNING_CONTAINER_OUTPUT_PATH
    # Build the versioned output subdirectory path.  save_configs() writes
    # to CWD, so we cd into this directory before running the script.
    if triton_version and triton_version != "unknown":
        versioned = "triton_%s" % triton_version.replace(".", "_")
        output_subdir = "%s/%s" % (config_dir, versioned)
    else:
        output_subdir = config_dir
    return (
        "mkdir -p %s && "
        "cd %s && "
        "SGLANG_MOE_CONFIG_DIR=%s "
        "python3 %s/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py "
        "--model %s --tp-size %d --tune"
    ) % (output_subdir, output_subdir, config_dir, SGLANG_CLONE_DIR, quote(model), tp_size)
