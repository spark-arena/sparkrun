"""eugr-vllm runtime: extends VllmRuntime with eugr container builds and mods."""

from __future__ import annotations

import logging
import subprocess
from logging import Logger
from pathlib import Path
from typing import Any, TYPE_CHECKING

from scitrera_app_framework import Variables, get_working_path

from sparkrun.runtimes.vllm_ray import VllmRayRuntime

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

EUGR_REPO_URL = "https://github.com/eugr/spark-vllm-docker.git"


class EugrVllmRayRuntime(VllmRayRuntime):
    """eugr-vllm runtime extending native vLLM with eugr build and mod support.

    Inherits all Ray-based orchestration from VllmRuntime (container launch,
    cluster management, log following, stop).  Adds eugr-specific features:

    - ``prepare()``: builds container images via eugr's ``build-and-copy.sh``
      when the recipe specifies ``build_args``.
    - ``_pre_serve()``: applies eugr mods (``docker cp`` + ``run.sh``) to
      containers after launch but before the serve command starts.
    """

    _v: Variables = None
    _repo_dir: Path | None = None
    _mods: list[str] = []

    runtime_name = "eugr-vllm"
    default_image_prefix = ""  # eugr uses local builds

    def initialize(self, v: Variables, logger_arg: Logger) -> EugrVllmRayRuntime:
        """Initialize the eugr-vllm runtime plugin."""
        self._v = v
        return self

    def resolve_container(self, recipe: Recipe, overrides: dict[str, Any] | None = None) -> str:
        """Resolve container -- eugr images use plain names, not prefix:tag."""
        return recipe.container or "vllm-node"

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate eugr-vllm-specific recipe fields."""
        issues = super().validate_recipe(recipe)
        if not recipe.command:
            issues.append("[eugr-vllm] command template is recommended for eugr recipes")
        return issues

    # --- Pre-launch preparation ---

    def prepare(
            self,
            recipe: Recipe,
            hosts: list[str],
            config: SparkrunConfig | None = None,
            dry_run: bool = False,
    ) -> None:
        """Build container image and cache mod info for _pre_serve.

        Called by the CLI before resource distribution.  If the recipe
        has ``build_args``, calls eugr's ``build-and-copy.sh`` to build
        the image locally.  If the recipe has ``mods``, caches the mod
        list and repo path for later application in ``_pre_serve()``.
        """
        build_args = recipe.runtime_config.get("build_args", [])
        has_mods = bool(recipe.runtime_config.get("mods"))

        # Determine if we need to build the image.
        # eugr images are locally built (not pulled from a registry), so if
        # the image is missing we must build it even without explicit build_args.
        image = recipe.container or "vllm-node"
        needs_build = bool(build_args)
        if not needs_build:
            from sparkrun.containers.registry import image_exists_locally
            if not image_exists_locally(image):
                logger.info("eugr image '%s' not found locally; will build", image)
                needs_build = True

        if not needs_build and not has_mods:
            return  # nothing eugr-specific to prepare

        # Ensure repo is available (for build script and/or mods)
        registry_cache_root = None
        if config is not None:
            registry_cache_root = Path(config.cache_dir) / "registries"
        self._repo_dir = self.ensure_repo(registry_cache_root=registry_cache_root)

        # Cache mod names for _pre_serve hook
        self._mods = recipe.runtime_config.get("mods", [])

        if not needs_build:
            return

        # Build image using eugr's build-and-copy.sh
        self._build_image(image, build_args, dry_run)

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

    # --- Pre-serve mod application ---

    def _pre_serve(
            self,
            hosts_containers: list[tuple[str, str]],
            ssh_kwargs: dict,
            dry_run: bool,
    ) -> None:
        """Apply eugr mods to containers after launch, before serve command.

        Each mod is a directory containing a ``run.sh`` script.  The mod
        directory is copied into the container at ``/workspace/mods/<name>``
        and ``run.sh`` is executed inside the container.
        """
        mods = getattr(self, "_mods", [])
        repo_dir = getattr(self, "_repo_dir", None)
        if not mods or not repo_dir:
            return

        logger.info("Applying %d mod(s) to %d container(s)...", len(mods), len(hosts_containers))
        for host, container_name in hosts_containers:
            for mod_name in mods:
                self._apply_mod(host, container_name, mod_name, repo_dir, ssh_kwargs, dry_run)

    def _apply_mod(
            self,
            host: str,
            container_name: str,
            mod_name: str,
            repo_dir: Path,
            ssh_kwargs: dict,
            dry_run: bool,
    ) -> None:
        """Copy a single mod into a container and execute its run.sh.

        For local hosts, uses ``docker cp`` and ``docker exec`` directly.
        For remote hosts, rsyncs the mod to a temp directory, then uses
        ``docker cp`` and ``docker exec`` via SSH.

        Args:
            host: Target hostname.
            container_name: Docker container name.
            mod_name: Mod directory name (relative to repo root).
            repo_dir: Path to the eugr repo clone.
            ssh_kwargs: SSH connection kwargs.
            dry_run: Show what would be done without executing.
        """
        mod_path = repo_dir / mod_name
        if not mod_path.is_dir() or not (mod_path / "run.sh").exists():
            logger.warning("Mod '%s' not found or missing run.sh at %s", mod_name, mod_path)
            return

        logger.info("Applying mod '%s' to %s on %s...", mod_name, container_name, host)
        if dry_run:
            return

        mod_basename = Path(mod_name).name
        container_dest = "/workspace/mods/%s" % mod_basename

        from sparkrun.core.hosts import is_local_host

        if is_local_host(host):
            # Local: docker cp directly
            subprocess.run(
                ["docker", "exec", container_name, "mkdir", "-p", container_dest],
                check=True,
            )
            subprocess.run(
                ["docker", "cp", "%s/." % mod_path, "%s:%s/" % (container_name, container_dest)],
                check=True,
            )
            subprocess.run(
                ["docker", "exec", container_name, "bash", "-c",
                 "cd %s && chmod +x run.sh && ./run.sh" % container_dest],
                check=True,
            )
        else:
            # Remote: rsync mod to temp dir, docker cp, docker exec, cleanup
            from sparkrun.orchestration.ssh import run_rsync_parallel
            from sparkrun.orchestration.primitives import run_script_on_host

            remote_tmp = "/tmp/sparkrun_mod_%s" % mod_basename
            run_script_on_host(
                host, "mkdir -p %s" % remote_tmp,
                ssh_kwargs=ssh_kwargs, timeout=30,
            )
            # rsync mod contents to remote
            kw = ssh_kwargs or {}
            run_rsync_parallel(
                str(mod_path) + "/", [host], remote_tmp + "/",
                ssh_user=kw.get("ssh_user"),
                ssh_key=kw.get("ssh_key"),
                ssh_options=kw.get("ssh_options"),
            )
            # docker cp into container and run
            script = (
                "docker exec {c} mkdir -p {dest}\n"
                "docker cp {tmp}/. {c}:{dest}/\n"
                "docker exec {c} bash -c 'cd {dest} && chmod +x run.sh && ./run.sh'\n"
                "rm -rf {tmp}\n"
            ).format(c=container_name, dest=container_dest, tmp=remote_tmp)
            run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=300)

    # --- Repo management (kept from original) ---

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
            cache_dir = Path(get_working_path(v=self._v)) / "cache"
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
