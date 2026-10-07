"""SGLang fused MoE and dense block-FP8 kernel tuning for DGX Spark.

MoE tuning uses upstream benchmarks for each TP size. Dense FP8 tuning uses
explicit per-rank shapes and the kernels installed in the serving image.
Results in the tuning cache are reused by future ``sparkrun run`` invocations.
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


def install_sglang_fp8_configs(hosts_containers, executor, ssh_kwargs, dry_run=False):
    """Install matching dense configs without masking the image's bundled ones."""
    import logging
    from sparkrun.orchestration.primitives import run_command_on_host
    from sparkrun.scripts import read_script

    if executor.executor_name != "docker" or not any((get_sglang_tuning_dir() / "fp8").rglob("*.json")):
        return
    command = "python3 - --install --output %s <<'SPARKRUN_FP8_PY'\n%s\nSPARKRUN_FP8_PY\n" % (
        quote(TUNING_CONTAINER_PATH),
        read_script("sglang_tune_fp8.py"),
    )
    logger = logging.getLogger(__name__)
    for host, container in hosts_containers:
        result = run_command_on_host(
            host,
            executor.exec_cmd(container, command, env={"PYTHONPATH": ""}),
            ssh_kwargs=ssh_kwargs,
            timeout=120,
            dry_run=dry_run,
        )
        if not result.success and not dry_run:
            logger.warning("Could not install SGLang FP8 tuning configs on %s: %s", host, result.stderr[-500:])
        elif result.stdout.strip():
            logger.info("%s: %s", host, result.stdout.strip())


# ---------------------------------------------------------------------------
# SglangTuner
# ---------------------------------------------------------------------------


class SglangTuner(BaseTuner):
    """Orchestrates SGLang MoE or dense block-FP8 tuning on a single host.

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

    @property
    def container_name(self):
        from sparkrun.core.application_profile import resource_name

        return resource_name("_tune")

    output_path = TUNING_CONTAINER_OUTPUT_PATH
    clone_script = "sglang_clone_benchmarks.sh"

    def __init__(self, *args, mode="moe", shapes=(), block_shape=(128, 128), batch_sizes=(), **kwargs):
        if mode not in {"moe", "fp8"}:
            raise ValueError("Unknown SGLang tuning mode: " + mode)
        if mode == "fp8" and not shapes:
            raise ValueError("FP8 tuning requires explicit per-rank weight shapes")
        self.mode, self.shapes = mode, tuple(shapes)
        self.block_shape, self.batch_sizes = tuple(block_shape), tuple(batch_sizes)
        if mode == "fp8":
            kwargs["skip_clone"] = True  # Use kernels installed in the image.
        super().__init__(*args, **kwargs)

    def _pre_check_tp(self, tp_size, triton_version):
        # An MoE filename cannot establish that dense shapes/batches are tuned.
        return False if self.mode == "fp8" else super()._pre_check_tp(tp_size, triton_version)

    def _run_tune_for_tp(self, tp_size, triton_version):
        if self.mode == "moe":
            return super()._run_tune_for_tp(tp_size, triton_version)
        from sparkrun.orchestration.primitives import run_script_on_host_streaming
        from sparkrun.scripts import read_script
        from sparkrun.tuning._common import resolve_tuning_timeout

        args = ["--output", self.output_path, "--block-shape", *map(str, self.block_shape)]
        for shape in self.shapes:
            args.extend(["--shape", *map(str, shape)])
        for batch in self.batch_sizes:
            args.extend(["--batch-size", str(batch)])
        script = "docker exec -i -e PYTHONPATH= %s python3 -u - %s <<'SPARKRUN_FP8_PY'\n%s\nSPARKRUN_FP8_PY\n" % (
            quote(self.container_name),
            " ".join(quote(arg) for arg in args),
            read_script("sglang_tune_fp8.py"),
        )
        result = run_script_on_host_streaming(
            self.host, script, ssh_kwargs=self.ssh_kwargs, timeout=resolve_tuning_timeout(self.timeout), dry_run=self.dry_run
        )
        return 0 if self.dry_run else result.returncode

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
            # A fixed step guard, deliberately not --timeout: this is a
            # sub-second best-effort `docker exec`, and the job budget is the
            # ceiling for the tuning run in _run_tune_for_tp.  Sharing one knob
            # would mean the default ("no ceiling") lets a wedged container
            # hang the patch step forever.
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
