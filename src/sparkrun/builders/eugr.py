"""eugr builder: container image building and mod injection for eugr-vllm recipes."""

from __future__ import annotations

import logging
import subprocess
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING

from scitrera_app_framework import Variables, get_working_path

from sparkrun.builders.base import BuilderPlugin

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

EUGR_REPO_URL = "https://github.com/eugr/spark-vllm-docker.git"


class EugrBuilder(BuilderPlugin):
    """Builder for eugr-style container images with mod support.

    Handles:
    - Building container images via eugr's ``build-and-copy.sh``
    - Converting recipe ``mods`` into ``pre_exec`` entries for the
      hooks system to execute at container launch time.
    """

    builder_name = "eugr"

    _v: Variables = None
    _repo_dir: Path | None = None

    def initialize(self, v: Variables, logger_arg: Logger) -> EugrBuilder:
        """Initialize the eugr builder plugin."""
        self._v = v
        return self

    def prepare_image(
            self,
            image: str,
            recipe: Recipe,
            hosts: list[str],
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
            transfer_mode: str = "local",
            ssh_kwargs: dict | None = None,
    ) -> str:
        """Build container image and inject mod pre_exec commands.

        If the recipe has ``build_args`` in runtime_config, builds
        the container via eugr's ``build-and-copy.sh``.  If the recipe
        has ``mods``, converts them to ``pre_exec`` entries that the
        hook system will execute at container launch time.

        In delegated mode (``transfer_mode="delegated"``), the build
        and repo clone happen on the **head node** via SSH rather than
        on the local control machine.

        Args:
            image: Target image name.
            recipe: The loaded recipe.
            hosts: Target host list (first element is head).
            config: SparkrunConfig for cache dir resolution.
            dry_run: Show what would be done without executing.
            transfer_mode: ``"local"`` or ``"delegated"``.
            ssh_kwargs: SSH connection kwargs (needed for delegated mode).

        Returns:
            Final image name (may be unchanged).
        """
        delegated = transfer_mode == "delegated"
        head = hosts[0] if hosts else "localhost"
        build_args = recipe.runtime_config.get("build_args", [])
        mods = recipe.runtime_config.get("mods", [])
        has_mods = bool(mods)

        # Determine if we need to build the image.
        needs_build = bool(build_args)
        if not needs_build:
            if delegated:
                if not self._image_exists_on_host(image, head, ssh_kwargs):
                    logger.info("eugr image '%s' not found on head '%s'; will build remotely", image, head)
                    needs_build = True
            else:
                from sparkrun.containers.registry import image_exists_locally
                if not image_exists_locally(image):
                    logger.info("eugr image '%s' not found locally; will build", image)
                    needs_build = True

        if not needs_build and not has_mods:
            return image  # nothing eugr-specific to prepare

        # Ensure repo is available (for build script and/or mods)
        if delegated:
            remote_repo = self._ensure_repo_remote(head, ssh_kwargs, dry_run=dry_run)
            self._repo_dir = Path(remote_repo)
        else:
            registry_cache_root = None
            if config is not None:
                registry_cache_root = Path(config.cache_dir) / "registries"
            self._repo_dir = self.ensure_repo(registry_cache_root=registry_cache_root)

        # Build image if needed
        if needs_build:
            if delegated:
                self._build_image_remote(image, build_args, head, ssh_kwargs, dry_run)
            else:
                self._build_image(image, build_args, dry_run)

        # Convert mods to pre_exec entries
        if has_mods:
            source_host = head if delegated else None
            self._inject_mod_pre_exec(recipe, mods, source_host=source_host)

        return image

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate eugr-specific recipe fields."""
        issues = []
        if not recipe.command:
            issues.append("[eugr] command template is recommended for eugr recipes")
        return issues

    # --- Mod -> pre_exec conversion ---

    def _inject_mod_pre_exec(
            self,
            recipe: Recipe,
            mods: list[str],
            source_host: str | None = None,
    ) -> None:
        """Convert mod entries to pre_exec commands on the recipe.

        Each mod is a directory containing a ``run.sh`` script.  The
        conversion produces two pre_exec entries per mod:
        1. A dict with ``copy`` key — file injection via docker cp
        2. A string — execute run.sh inside the container

        Args:
            recipe: Recipe instance (pre_exec list is mutated).
            mods: List of mod directory names relative to repo root.
            source_host: When set (delegated mode), the ``copy`` entry
                gets a ``source_host`` key so the hooks system knows
                where the source files live.
        """
        if not self._repo_dir:
            logger.warning("Cannot inject mods without a repo dir")
            return

        for mod_name in mods:
            # Strip leading 'mods/' if present — recipes may specify either
            # "fix-something" or "mods/fix-something"; normalise to avoid
            # doubling the path component.
            clean_name = mod_name.removeprefix('mods/')
            mod_path = self._repo_dir / 'mods' / clean_name
            mod_basename = Path(clean_name).name
            dest = "/workspace/mods/%s" % mod_basename

            # Add copy entry (docker cp source into container)
            copy_entry: dict[str, str] = {
                "copy": str(mod_path),
                "dest": dest,
            }
            if source_host is not None:
                copy_entry["source_host"] = source_host
            recipe.pre_exec.append(copy_entry)
            # Add exec entry (run the mod script with WORKSPACE_DIR set)
            recipe.pre_exec.append(
                "export WORKSPACE_DIR=$PWD && cd %s && chmod +x run.sh && ./run.sh" % dest
            )

        logger.info("Injected %d mod(s) as pre_exec entries", len(mods))

    # --- Image building ---

    def _build_image(self, image: str, build_args: list[str], dry_run: bool = False) -> None:
        """Build container image via eugr's build-and-copy.sh.

        Args:
            image: Target image name (passed as ``-t``).
            build_args: Additional build arguments forwarded to the script.
            dry_run: Show what would be done without executing.
        """
        build_script = self._repo_dir / "build-and-copy.sh"
        if not build_script.exists():
            raise RuntimeError("build-and-copy.sh not found at %s" % build_script)

        cmd = [str(build_script), "-t", image] + build_args
        logger.info("Building eugr container: %s", " ".join(cmd))

        if dry_run:
            return

        result = subprocess.run(cmd, cwd=str(self._repo_dir))
        if result.returncode != 0:
            raise RuntimeError("eugr container build failed (exit %d)" % result.returncode)

    # --- Remote / delegated helpers ---

    @staticmethod
    def _image_exists_on_host(image: str, host: str, ssh_kwargs: dict | None = None) -> bool:
        """Check whether *image* exists on a remote host.

        Runs ``docker image inspect <image>`` via SSH and returns
        ``True`` when the exit code is 0.
        """
        from sparkrun.orchestration.primitives import run_script_on_host
        script = "docker image inspect %s >/dev/null 2>&1" % image
        result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=30)
        return result.success

    def _ensure_repo_remote(
            self,
            head: str,
            ssh_kwargs: dict | None = None,
            dry_run: bool = False,
    ) -> str:
        """Clone or update the eugr repo on the head node.

        Returns the remote path where the repo lives.
        """
        from sparkrun.orchestration.primitives import run_script_on_host

        remote_path = "~/.cache/sparkrun/eugr-spark-vllm-docker"
        script = (
                     "set -e\n"
                     "REPO_DIR=%(path)s\n"
                     "if [ -d \"$REPO_DIR/.git\" ]; then\n"
                     "  git -C \"$REPO_DIR\" pull --ff-only || true\n"
                     "else\n"
                     "  mkdir -p \"$(dirname \"$REPO_DIR\")\"\n"
                     "  git clone %(url)s \"$REPO_DIR\"\n"
                     "fi\n"
                     "echo \"$REPO_DIR\"\n"
                 ) % {"path": remote_path, "url": EUGR_REPO_URL}

        logger.info("Ensuring eugr repo on head node %s...", head)

        if dry_run:
            return remote_path

        result = run_script_on_host(head, script, ssh_kwargs=ssh_kwargs, timeout=120)
        if not result.success:
            raise RuntimeError(
                "Failed to ensure eugr repo on %s: %s"
                % (head, result.stderr.strip() if result.stderr else "(no output)")
            )

        return remote_path

    def _build_image_remote(
            self,
            image: str,
            build_args: list[str],
            head: str,
            ssh_kwargs: dict | None = None,
            dry_run: bool = False,
    ) -> None:
        """Build container image on the head node via SSH.

        Args:
            image: Target image name (passed as ``-t``).
            build_args: Additional build arguments forwarded to the script.
            head: Head node hostname.
            ssh_kwargs: SSH connection kwargs.
            dry_run: Show what would be done without executing.
        """
        from sparkrun.orchestration.primitives import run_script_on_host

        remote_path = "~/.cache/sparkrun/eugr-spark-vllm-docker"
        args_str = " ".join(build_args) if build_args else ""
        script = (
                     "set -e\n"
                     "cd %(path)s\n"
                     "chmod +x build-and-copy.sh\n"
                     "./build-and-copy.sh -t %(image)s %(args)s\n"
                 ) % {"path": remote_path, "image": image, "args": args_str}

        logger.info("Building eugr container on %s: %s -t %s %s", head, remote_path, image, args_str)

        if dry_run:
            return

        result = run_script_on_host(head, script, ssh_kwargs=ssh_kwargs, timeout=1800)
        if not result.success:
            raise RuntimeError(
                "eugr remote container build failed on %s (exit %d)"
                % (head, result.returncode)
            )

    # --- Repo management ---

    def ensure_repo(
            self,
            cache_dir: Path | None = None,
            registry_cache_root: Path | None = None,
    ) -> Path:
        """Clone or update the eugr repo in sparkrun's cache.

        If the registry system already has a cached clone of the eugr-vllm
        repo (from recipe syncing), reuses it instead of cloning a second
        copy.  Sparse checkout is disabled on the registry clone so that
        scripts like ``build-and-copy.sh`` are available.
        """
        # Check if registry already has this repo cloned
        if registry_cache_root is not None:
            registry_repo = registry_cache_root / "eugr-vllm"
            if (registry_repo / ".git").exists():
                logger.info("Reusing eugr repo from registry cache: %s", registry_repo)
                self._ensure_full_checkout(registry_repo)
                self._update_repo(registry_repo)
                return registry_repo

        if cache_dir is None:
            if self._v is not None:
                cache_dir = Path(get_working_path(v=self._v)) / "cache"
            else:
                from sparkrun.core.config import resolve_cache_dir
                cache_dir = Path(resolve_cache_dir(None))
        repo_dir = cache_dir / "eugr-spark-vllm-docker"

        if repo_dir.exists() and (repo_dir / ".git").exists():
            self._update_repo(repo_dir)
        else:
            logger.info("Cloning eugr/spark-vllm-docker...")
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["git", "clone", EUGR_REPO_URL, str(repo_dir)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Failed to clone eugr repo: %s" % result.stderr.strip()
                )

        return repo_dir

    @staticmethod
    def _ensure_full_checkout(repo_dir: Path) -> None:
        """Disable sparse checkout so all repo files are available."""
        sparse_file = repo_dir / ".git" / "info" / "sparse-checkout"
        if not sparse_file.exists():
            return  # not sparse, nothing to do
        logger.debug("Disabling sparse checkout on %s", repo_dir)
        subprocess.run(
            ["git", "-C", str(repo_dir), "sparse-checkout", "disable"],
            capture_output=True, text=True,
        )

    @staticmethod
    def _update_repo(repo_dir: Path) -> None:
        """Pull latest changes for an existing repo clone."""
        logger.info("Updating eugr/spark-vllm-docker repo...")
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "pull", "--ff-only"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "Failed to update eugr repo (continuing with existing): %s",
                result.stderr.strip(),
            )
