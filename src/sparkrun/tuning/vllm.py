"""vLLM kernel tuning for DGX Spark, backed by the vllm-tune CLI.

Delegates the actual tuning work to https://github.com/SeraphimSerapis/vllm-tune
(Apache-2.0), shelled out over SSH on the target host.  vllm-tune covers both
fused MoE Triton kernels and FP8 dense GEMM kernels, handles its own
standalone container lifecycle, and writes results into sparkrun's tuning
cache via its ``--export-sparkrun`` integration point.

Sparkrun's role:
  * Resolve the pinned vllm-tune git URL/ref from :class:`SparkrunConfig`.
  * Ensure the pinned ref is checked out on the remote host.
  * Invoke ``vllm-tune.sh`` once per TP size with ``--standalone --foreground``.
  * Trigger the post-tune ``--export-sparkrun`` step that flattens configs into
    the cache directory the vLLM runtimes auto-mount.
  * Rsync the flattened cache back to the control machine.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from sparkrun.tuning._common import (
    DEFAULT_TP_SIZES,  # noqa: F401 — re-exported for public API
    _format_duration,
    _get_tuning_dir,
    _get_tuning_env,
    _get_tuning_volumes,
)
from sparkrun.utils.shell import quote, safe_remote_path, validate_git_url

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (public — also consumed by tuning/sync.py and the runtime mixin)
# ---------------------------------------------------------------------------

VLLM_TUNING_CACHE_SUBDIR = "tuning/vllm"
VLLM_TUNING_CONTAINER_PATH = "/tuning/vllm"

# Per-TP tuning timeout: vllm-tune's MoE phase can take 1.5-3 hours, FP8 adds
# another 15-25 minutes; allow 8 hours per invocation to match the prior
# BaseTuner budget.
_TUNE_TIMEOUT_SEC = 28800

# Subdir on the remote host where the pinned vllm-tune checkout lives.
_VLLM_TUNE_REMOTE_PARENT = "$HOME/.cache/sparkrun/vllm-tune"

VALID_MODES = ("moe", "fp8", "all")


# ---------------------------------------------------------------------------
# Host-side helpers (used by runtimes for auto-mounting)
# ---------------------------------------------------------------------------


def get_vllm_tuning_dir() -> Path:
    """Return the host-side directory for vLLM tuning configs.

    Default: ``~/.cache/sparkrun/tuning/vllm/``
    """
    return _get_tuning_dir(VLLM_TUNING_CACHE_SUBDIR)


def get_vllm_tuning_volumes() -> dict[str, str] | None:
    """Return volume mapping for vLLM tuning configs if they exist."""
    return _get_tuning_volumes(get_vllm_tuning_dir, VLLM_TUNING_CONTAINER_PATH)


def get_vllm_tuning_env() -> dict[str, str] | None:
    """Return env vars for vLLM tuning configs if they exist."""
    return _get_tuning_env(
        get_vllm_tuning_volumes,
        "VLLM_TUNED_CONFIG_FOLDER",
        VLLM_TUNING_CONTAINER_PATH,
    )


# ---------------------------------------------------------------------------
# Command builders (also used by tests)
# ---------------------------------------------------------------------------


def build_vllm_tune_invocation(
    install_path: str,
    model: str,
    tp_size: int,
    mode: str,
    image: str,
    sparkrun_dir: str,
) -> str:
    """Build the shell command that runs vllm-tune for a single TP size.

    The command runs vllm-tune in foreground mode so output streams over SSH,
    in standalone mode so vllm-tune launches its own dedicated tuning
    container, with --image flowing the recipe's container image through.
    SPARKRUN_TUNING_DIR is exported so a follow-up ``--export-sparkrun`` call
    writes to the same flat cache directory the vLLM runtimes auto-mount.
    """
    return ("SPARKRUN_TUNING_DIR=%s bash %s %s --tp %d --mode %s --standalone --image %s --foreground") % (
        quote(sparkrun_dir),
        quote(install_path),
        quote(model),
        tp_size,
        quote(mode),
        quote(image),
    )


def build_vllm_tune_export(
    install_path: str,
    model: str,
    tp_size: int,
    mode: str,
    sparkrun_dir: str,
) -> str:
    """Build the shell command that copies vllm-tune outputs into the sparkrun
    flat cache.  Cheap (just ``cp``) and idempotent."""
    return ("SPARKRUN_TUNING_DIR=%s bash %s %s --tp %d --mode %s --export-sparkrun --sparkrun-dir %s") % (
        quote(sparkrun_dir),
        quote(install_path),
        quote(model),
        tp_size,
        quote(mode),
        quote(sparkrun_dir),
    )


# ---------------------------------------------------------------------------
# VllmTuner
# ---------------------------------------------------------------------------


class VllmTuner:
    """Drive ``vllm-tune`` against a single remote host.

    Args:
        host: Target host (where Docker runs).
        image: Container image (passed to vllm-tune ``--image``).
        model: HuggingFace model ID to tune for.
        config: SparkrunConfig (used for SSH kwargs and vllm-tune pin).
        cache_dir: Unused for now — vllm-tune resolves HF cache via HF_HOME
            inside the standalone container.
        output_dir: Optional override for the host-side flat cache directory.
        mode: ``"moe"`` / ``"fp8"`` / ``"all"`` (default ``"all"``).
        vllm_tune_ref: Override the git ref pinned in ``SparkrunConfig``.
        dry_run: Print commands without executing.
    """

    def __init__(
        self,
        host: str,
        image: str,
        model: str,
        config: SparkrunConfig | None = None,
        cache_dir: str | None = None,
        output_dir: str | None = None,
        mode: str = "all",
        vllm_tune_ref: str | None = None,
        dry_run: bool = False,
    ):
        if mode not in VALID_MODES:
            raise ValueError("mode must be one of %s, got %r" % (VALID_MODES, mode))

        from sparkrun.orchestration.primitives import build_ssh_kwargs

        self.host = host
        self.image = image
        self.model = model
        self.config = config
        self.cache_dir = cache_dir
        self.mode = mode
        self.dry_run = dry_run

        self._custom_output_dir = output_dir is not None
        self.output_dir = output_dir or str(get_vllm_tuning_dir())

        self.ssh_kwargs = build_ssh_kwargs(config)
        self.remote_output_dir = _resolve_remote_output_dir(
            self.output_dir,
            self._custom_output_dir,
            self.ssh_kwargs.get("ssh_user"),
        )

        # Resolve the vllm-tune pin (CLI override → config → built-in default).
        self.vllm_tune_repo, self.vllm_tune_ref = _resolve_vllm_tune_pin(config, vllm_tune_ref)

    # ----- public entry point -----

    def run_tuning(
        self,
        tp_sizes: tuple[int, ...] = DEFAULT_TP_SIZES,
        parallel: int = 1,
    ) -> int:
        """Run vllm-tune for each TP size and rsync results back."""
        logger.info("=" * 60)
        logger.info("sparkrun vLLM Kernel Tuner (vllm-tune backend)")
        logger.info("=" * 60)
        logger.info("Host:           %s", self.host)
        logger.info("Image:          %s", self.image)
        logger.info("Model:          %s", self.model)
        logger.info("TP sizes:       %s", ", ".join(str(t) for t in tp_sizes))
        logger.info("Mode:           %s", self.mode)
        logger.info("Parallel:       %d", parallel)
        logger.info("Output:         %s", self.output_dir)
        logger.info("vllm-tune ref:  %s @ %s", self.vllm_tune_repo, self.vllm_tune_ref)
        logger.info("Run mode:       %s", "DRY-RUN" if self.dry_run else "LIVE")
        logger.info("=" * 60)

        t_total = time.monotonic()
        tp_timings: list[tuple[int, float]] = []

        # Step 1: install/update the pinned vllm-tune checkout on the host.
        install_path = self._install_vllm_tune()
        if install_path is None:
            return 1

        # Step 2: preflight (jq, docker on PATH on the remote).
        rc = self._preflight()
        if rc != 0:
            return rc

        # Step 3: ensure the export target directory exists on the remote.
        rc = self._ensure_remote_output_dir()
        if rc != 0:
            return rc

        # Step 4: run tuning + export, per TP.
        if parallel > 1 and len(tp_sizes) > 1:
            rc = self._run_tp_parallel(install_path, tp_sizes, parallel, tp_timings)
        else:
            rc = self._run_tp_sequential(install_path, tp_sizes, tp_timings)
        if rc != 0:
            return rc

        # Step 5: rsync the flat cache back to the control machine.
        self._sync_back_configs()

        total_elapsed = time.monotonic() - t_total
        self._print_timing_summary(tp_timings, total_elapsed)
        return 0

    # ----- steps -----

    def _install_vllm_tune(self) -> str | None:
        """Ensure the pinned vllm-tune ref is checked out on the remote host.

        Returns the absolute path to ``vllm-tune.sh`` on the remote, or
        ``None`` on failure.
        """
        from sparkrun.orchestration.primitives import run_script_on_host
        from sparkrun.scripts import read_script

        # Derive a stable subdir from the ref.  Use a sanitized form so an
        # accidental slash in the ref name can't escape the parent dir.
        safe_ref = self.vllm_tune_ref.replace("/", "_")
        dest = "%s/%s" % (_VLLM_TUNE_REMOTE_PARENT, safe_ref)
        install_script = read_script("vllm_tune_install.sh")

        # Prefix env-var assignments to the script body so they're available to
        # the script regardless of how the remote shell sources them.
        prelude = "export VLLM_TUNE_REPO=%s\nexport VLLM_TUNE_REF=%s\nexport VLLM_TUNE_DEST=%s\n" % (
            quote(self.vllm_tune_repo),
            quote(self.vllm_tune_ref),
            quote(dest),
        )
        # The script's shebang stays at the top.
        body = install_script
        if body.startswith("#!"):
            shebang, _, rest = body.partition("\n")
            body = shebang + "\n" + prelude + rest
        else:
            body = "#!/bin/bash\n" + prelude + body

        logger.info("Step 1/5: Ensuring vllm-tune %s is installed on %s...", self.vllm_tune_ref, self.host)
        result = run_script_on_host(
            self.host,
            body,
            ssh_kwargs=self.ssh_kwargs,
            timeout=300,
            dry_run=self.dry_run,
        )

        if self.dry_run:
            return "%s/vllm-tune.sh" % dest

        if not result.success:
            logger.error("Failed to install vllm-tune on %s: %s", self.host, result.stderr[:500])
            return None

        # The install script echoes the resolved vllm-tune.sh path on the last line.
        path = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
        if not path:
            path = "%s/vllm-tune.sh" % dest
        logger.info("Step 1/5: vllm-tune ready at %s:%s", self.host, path)
        return path

    def _preflight(self) -> int:
        """Verify jq, docker, and git are on PATH on the remote."""
        from sparkrun.orchestration.primitives import run_command_on_host

        logger.info("Step 2/5: Preflight (jq, docker on remote)...")
        # `command -v` returns non-zero if the binary is missing; report which.
        cmd = 'for bin in jq docker git; do command -v "$bin" >/dev/null 2>&1 || { echo "MISSING:$bin"; exit 1; }; done; echo OK'
        result = run_command_on_host(
            self.host,
            cmd,
            ssh_kwargs=self.ssh_kwargs,
            timeout=30,
            dry_run=self.dry_run,
        )
        if self.dry_run:
            return 0
        if not result.success:
            missing = ""
            for line in (result.stdout or "").splitlines():
                if line.startswith("MISSING:"):
                    missing = line.split(":", 1)[1].strip()
                    break
            if missing:
                logger.error(
                    "Preflight failed on %s: '%s' is not on PATH. vllm-tune requires jq, docker, and git.",
                    self.host,
                    missing,
                )
            else:
                logger.error("Preflight failed on %s: %s", self.host, result.stderr[:300])
            return 1
        return 0

    def _ensure_remote_output_dir(self) -> int:
        """Create the remote flat cache dir so --export-sparkrun has a target."""
        from sparkrun.orchestration.primitives import run_script_on_host

        script = '#!/bin/bash\nset -uo pipefail\nmkdir -p "%s"\n' % safe_remote_path(self.remote_output_dir)
        result = run_script_on_host(
            self.host,
            script,
            ssh_kwargs=self.ssh_kwargs,
            timeout=30,
            dry_run=self.dry_run,
        )
        if not result.success and not self.dry_run:
            logger.error(
                "Failed to create output directory %s on %s: %s",
                self.remote_output_dir,
                self.host,
                result.stderr,
            )
            return 1
        return 0

    def _run_tp_sequential(
        self,
        install_path: str,
        tp_sizes: tuple[int, ...],
        tp_timings: list[tuple[int, float]],
    ) -> int:
        for i, tp in enumerate(tp_sizes):
            logger.info("Step 4/5: Tuning TP=%d (%d/%d)...", tp, i + 1, len(tp_sizes))
            t0 = time.monotonic()
            rc = self._run_one_tp(install_path, tp)
            tp_timings.append((tp, time.monotonic() - t0))
            if rc != 0:
                logger.error("Tuning failed for TP=%d (exit %d)", tp, rc)
                return rc
        return 0

    def _run_tp_parallel(
        self,
        install_path: str,
        tp_sizes: tuple[int, ...],
        max_workers: int,
        tp_timings: list[tuple[int, float]],
    ) -> int:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        effective_workers = min(max_workers, len(tp_sizes))
        logger.info(
            "Step 4/5: Tuning %d TP sizes with %d parallel workers...",
            len(tp_sizes),
            effective_workers,
        )
        failed: list[tuple[int, int]] = []

        def _tune_one(tp: int) -> tuple[int, int, float]:
            t0 = time.monotonic()
            rc = self._run_one_tp(install_path, tp)
            return tp, rc, time.monotonic() - t0

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_tune_one, tp): tp for tp in tp_sizes}
            for future in as_completed(futures):
                tp, rc, elapsed = future.result()
                tp_timings.append((tp, elapsed))
                if rc != 0:
                    logger.error("Tuning failed for TP=%d (exit %d)", tp, rc)
                    failed.append((tp, rc))
                else:
                    logger.info("  TP=%d done (%s)", tp, _format_duration(elapsed))

        tp_timings.sort(key=lambda x: x[0])
        if failed:
            logger.error("Tuning failed for TP size(s): %s", ", ".join(str(tp) for tp, _ in failed))
            return failed[0][1]
        return 0

    def _run_one_tp(self, install_path: str, tp_size: int) -> int:
        """Tune for a single TP size, then export to the sparkrun flat cache."""
        from sparkrun.orchestration.primitives import run_command_on_host

        tune_cmd = build_vllm_tune_invocation(
            install_path=install_path,
            model=self.model,
            tp_size=tp_size,
            mode=self.mode,
            image=self.image,
            sparkrun_dir=self.remote_output_dir,
        )
        t0 = time.monotonic()
        result = run_command_on_host(
            self.host,
            tune_cmd,
            ssh_kwargs=self.ssh_kwargs,
            timeout=_TUNE_TIMEOUT_SEC,
            dry_run=self.dry_run,
        )
        if self.dry_run:
            logger.info("  [dry-run] Would run: %s", tune_cmd)
            return 0
        if not result.success:
            logger.error(
                "  Tune for TP=%d failed (exit %d, %s)",
                tp_size,
                result.returncode,
                _format_duration(time.monotonic() - t0),
            )
            if result.stdout and result.stdout.strip():
                logger.error("  stdout:\n%s", result.stdout.rstrip())
            if result.stderr and result.stderr.strip():
                logger.error("  stderr:\n%s", result.stderr.rstrip())
            return result.returncode

        # Tuning succeeded; flatten outputs into the sparkrun cache so the
        # runtime mixin can mount them.  vllm-tune's --export-sparkrun runs as
        # a fast early-exit code path (just cp).
        export_cmd = build_vllm_tune_export(
            install_path=install_path,
            model=self.model,
            tp_size=tp_size,
            mode=self.mode,
            sparkrun_dir=self.remote_output_dir,
        )
        export_result = run_command_on_host(
            self.host,
            export_cmd,
            ssh_kwargs=self.ssh_kwargs,
            timeout=300,
            dry_run=False,
        )
        if not export_result.success:
            logger.warning(
                "  Export to sparkrun cache failed for TP=%d: %s",
                tp_size,
                export_result.stderr[:300],
            )
            # Don't fail the tuning run — configs still exist in vllm-tune's
            # own nested layout on the remote; surface the warning.
        return 0

    def _sync_back_configs(self) -> None:
        """Rsync the remote flat tuning cache back to the control machine."""
        from sparkrun.utils import is_local_host

        if is_local_host(self.host):
            return
        if self.dry_run:
            logger.info("  [dry-run] Would rsync %s:%s -> %s", self.host, self.remote_output_dir, self.output_dir)
            return

        from sparkrun.orchestration.ssh import run_rsync_from_remote

        logger.info("Step 5/5: Syncing tuning configs back from %s...", self.host)
        result = run_rsync_from_remote(
            host=self.host,
            source_path=self.remote_output_dir,
            dest_path=self.output_dir,
            ssh_user=self.ssh_kwargs.get("ssh_user"),
            ssh_key=self.ssh_kwargs.get("ssh_key"),
            ssh_options=self.ssh_kwargs.get("ssh_options"),
            rsync_options=["-az", "--mkpath", "--partial", "--links"],
            timeout=300,
        )
        if result.success:
            logger.info("  Tuning configs synced to local %s", self.output_dir)
        else:
            logger.warning(
                "  Failed to sync tuning configs back from %s: %s",
                self.host,
                result.stderr[:300],
            )

    def _print_timing_summary(
        self,
        tp_timings: list[tuple[int, float]],
        total_elapsed: float,
    ) -> None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Tuning Summary")
        logger.info("=" * 60)
        logger.info("  %-8s  %s", "TP Size", "Duration")
        logger.info("  %-8s  %s", "-------", "--------")
        for tp, elapsed in tp_timings:
            logger.info("  %-8d  %s", tp, _format_duration(elapsed))
        logger.info("  %-8s  %s", "-------", "--------")
        logger.info("  %-8s  %s", "Total", _format_duration(total_elapsed))
        logger.info("")
        logger.info("Tuning configs saved to: %s", self.output_dir)
        logger.info("These will be auto-mounted in future 'sparkrun run' invocations.")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_vllm_tune_pin(
    config: SparkrunConfig | None,
    cli_ref_override: str | None,
) -> tuple[str, str]:
    """Resolve (repo, ref) from CLI override → config → built-in defaults."""
    from sparkrun.core.config import DEFAULT_VLLM_TUNE_REF, DEFAULT_VLLM_TUNE_REPO

    repo = config.vllm_tune_repo if config is not None else DEFAULT_VLLM_TUNE_REPO
    ref = cli_ref_override or (config.vllm_tune_ref if config is not None else DEFAULT_VLLM_TUNE_REF)
    # Reject URLs that could be misread as git CLI options.
    repo = validate_git_url(repo)
    return repo, ref


def _resolve_remote_output_dir(
    output_dir: str,
    custom_output_dir: bool,
    ssh_user: str | None,
) -> str:
    """Map the local output dir to the remote-host equivalent.

    Mirrors the cross-OS / cross-user logic from BaseTuner: when the control
    machine is non-Linux or the SSH user differs from the local user, the
    local cache prefix doesn't exist on the remote.  Rewrites
    ``DEFAULT_CACHE_DIR/...`` to ``/home/<ssh_user>/.cache/sparkrun/...``.
    """
    import os
    import sys

    if custom_output_dir:
        return output_dir

    from sparkrun.core.config import DEFAULT_CACHE_DIR

    local_user = os.environ.get("USER")
    if (ssh_user and ssh_user != local_user) or sys.platform != "linux":
        _user = ssh_user or local_user or "user"
        local_prefix = str(DEFAULT_CACHE_DIR)
        if output_dir.startswith(local_prefix):
            suffix = output_dir[len(local_prefix) :]
            return "/home/%s/.cache/sparkrun%s" % (_user, suffix)
        return output_dir
    return output_dir
