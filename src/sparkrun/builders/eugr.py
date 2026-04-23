"""eugr builder: container image building and mod injection for eugr-vllm recipes."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING

from scitrera_app_framework import Variables, get_working_path

from sparkrun.builders.base import BuilderPlugin, _flatten_dict, PULLABLE_REGISTRY_PREFIXES
from sparkrun.utils.shell import quote, quote_list, args_list_to_shell_str

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe

logger = logging.getLogger(__name__)

EUGR_REPO_URL = "https://github.com/eugr/spark-vllm-docker.git"

# Build-index URL for eugr nightly builds (used for long-term image resolution)
EUGR_BUILD_INDEX_URL = "https://raw.githubusercontent.com/spark-arena/dgx-vllm/refs/heads/main/build-index.json"
EUGR_BUILD_INDEX_CACHE_NAME = "eugr-vllm-build-index.json"

# GHCR image names for standard eugr nightly variants
GHCR_EUGR_NIGHTLY = "ghcr.io/spark-arena/dgx-vllm-eugr-nightly"
GHCR_EUGR_NIGHTLY_TF5 = "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5"

# GHCR package paths (without registry prefix) for API calls
GHCR_EUGR_PKG = "spark-arena/dgx-vllm-eugr-nightly"
GHCR_EUGR_PKG_TF5 = "spark-arena/dgx-vllm-eugr-nightly-tf5"

# Local image names produced by prepare_image() for nightly builds
LOCAL_EUGR_NIGHTLY = "sparkrun-eugr-vllm"
LOCAL_EUGR_NIGHTLY_TF5 = "sparkrun-eugr-vllm-tf5"

# Build cache file name (stored under cache_dir)
EUGR_BUILD_CACHE_NAME = "eugr-build-cache.json"

# GitHub release tags used by build-and-copy.sh to fetch prebuilt wheels
_GITHUB_RELEASE_URL = "https://api.github.com/repos/eugr/spark-vllm-docker/releases/tags/%s"
_VLLM_RELEASE_TAG = "prebuilt-vllm-current"
_FLASHINFER_RELEASE_TAG = "prebuilt-flashinfer-current"

# Regexes to extract commit hashes from GitHub release names
_RE_VLLM_COMMIT = re.compile(r"\+g([0-9a-f]{6,})\.")
_RE_FLASHINFER_COMMIT = re.compile(r"\([\d.]+\w*-([0-9a-f]{6,})-d\d{8}\)")

# build_args values that are eligible for cache skip checks
_CACHEABLE_BUILD_ARGS: list[list[str]] = [[], ["--tf5"]]


def _load_build_cache(cache_dir: Path) -> dict:
    """Load the eugr build cache from disk."""
    cache_file = cache_dir / EUGR_BUILD_CACHE_NAME
    if not cache_file.exists():
        return {}
    try:
        return json.loads(cache_file.read_text())
    except Exception:
        logger.debug("Failed to read build cache %s", cache_file, exc_info=True)
        return {}


def _save_build_cache(cache_dir: Path, cache: dict) -> None:
    """Persist the eugr build cache to disk."""
    cache_file = cache_dir / EUGR_BUILD_CACHE_NAME
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=2))
    except Exception:
        logger.debug("Failed to write build cache %s", cache_file, exc_info=True)


def _fetch_upstream_wheel_hashes() -> dict[str, str]:
    """Fetch current upstream vLLM and FlashInfer commit hashes from GitHub releases.

    Returns a dict with ``vllm_commit`` and ``flashinfer_commit`` keys,
    or an empty dict on any failure.
    """
    result: dict[str, str] = {}
    try:
        for tag, regex, key in [
            (_VLLM_RELEASE_TAG, _RE_VLLM_COMMIT, "vllm_commit"),
            (_FLASHINFER_RELEASE_TAG, _RE_FLASHINFER_COMMIT, "flashinfer_commit"),
        ]:
            url = _GITHUB_RELEASE_URL % tag
            req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            name = data.get("name", "")
            m = regex.search(name)
            if not m:
                logger.debug("Could not parse commit hash from release name: %s", name)
                return {}
            result[key] = m.group(1)
    except Exception:
        logger.debug("Failed to fetch upstream wheel hashes", exc_info=True)
        return {}
    return result


class EugrBuilder(BuilderPlugin):
    """Builder for eugr-style container images with mod support.

    Handles:
    - Building container images via eugr's ``build-and-copy.sh``
    - Converting recipe ``mods`` into ``pre_exec`` entries for the
      hooks system to execute at container launch time.
    """

    builder_name = "eugr"

    _v: Variables | None = None
    _repo_dir: Path | None = None

    def initialize(self, v: Variables, logger: Logger) -> EugrBuilder:
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
        logger.debug("eugr prepare_image: transfer_mode=%s, delegated=%s", transfer_mode, delegated)
        head = hosts[0] if hosts else "localhost"
        build_args = recipe.runtime_config.get("build_args", [])
        mods = recipe.runtime_config.get("mods", [])
        has_mods = bool(mods)
        needs_build = False  # assume False at first

        # ~~ SPECIAL CASES: map pullable eugr nightly images to use direct build ~~~
        if image.strip() == GHCR_EUGR_NIGHTLY_TF5 + ":latest" and (build_args == ["--tf5"] or not build_args):
            # use sparkrun prefixed names to avoid collisions with other user images
            image = LOCAL_EUGR_NIGHTLY_TF5
            build_args = ["--tf5"]
            needs_build = True
            logger.info("Mapped eugr nightly tf5 image to use direct build via container name '%s' (build_args managed)", image)
        elif image.strip() == GHCR_EUGR_NIGHTLY + ":latest" and not build_args:
            # use sparkrun prefixed names to avoid collisions with other user images
            image = LOCAL_EUGR_NIGHTLY
            needs_build = True
            logger.info("Mapped eugr nightly image to container name '%s'", image)
        # NOTE: if not :latest, then we do want to use the given container image

        # Determine if we need to build the image.
        # If the image references a known public registry, it's pullable — never build it.
        is_pullable = any(image.startswith(prefix) for prefix in PULLABLE_REGISTRY_PREFIXES)
        if is_pullable and not needs_build:
            logger.info("image '%s' is from a known registry; skipping build (will be pulled at runtime)", image)
            needs_build = False
        elif not needs_build:
            # Check if image already exists — skip build if specifically named and already present
            if delegated:
                image_found = self._image_exists_on_host(image, head, ssh_kwargs)
            else:
                from sparkrun.containers.registry import image_exists_locally

                image_found = image_exists_locally(image)

            if not image_found:
                needs_build = True
                if delegated:
                    logger.info("image '%s' not found on head '%s'; will build remotely", image, head)
                else:
                    logger.info("image '%s' not found locally; will build", image)

        # nothing eugr-specific to prepare -- no build, no mods
        if not needs_build and not has_mods:
            return image

        # Resolve optional branch override from builder_config
        branch = recipe.builder_config.get("branch") if recipe.builder_config else None

        # Ensure repo is available (for build script and/or mods)
        if delegated:
            remote_repo = self._ensure_repo_remote(head, ssh_kwargs, dry_run=dry_run, branch=branch)
            self._repo_dir = Path(remote_repo)
        else:
            registry_cache_root = None
            if config is not None:
                registry_cache_root = Path(config.cache_dir) / "registries"
            self._repo_dir = self.ensure_repo(registry_cache_root=registry_cache_root, branch=branch)

        # Check build cache — skip rebuild if upstream wheels haven't changed
        if needs_build and not dry_run:
            skip_host = head if delegated else None
            skip_ssh = ssh_kwargs if delegated else None
            if self._can_skip_build(image, build_args, config, host=skip_host, ssh_kwargs=skip_ssh):
                logger.info(
                    "Build cache hit — skipping rebuild of '%s' (upstream wheels unchanged)",
                    image,
                )
                needs_build = False

        # Build image if needed
        if needs_build:
            if delegated:
                self._build_image_remote(image, build_args, head, ssh_kwargs, dry_run)
                if not dry_run:
                    self._save_build_metadata(image, build_args, config, host=head, ssh_kwargs=ssh_kwargs)
            else:
                self._build_image(image, build_args, dry_run)
                if not dry_run:
                    self._save_build_metadata(image, build_args, config)

        # Convert mods to pre_exec entries
        if has_mods:
            source_host = head if delegated else None
            self._inject_mod_pre_exec(recipe, mods, source_host=source_host)

        # TODO: potentially inject metadata flags as needed into recipe?
        return image

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Validate eugr-specific recipe fields."""
        issues = []
        if not recipe.command:
            issues.append("[eugr] command template is recommended for eugr recipes")
        return issues

    def version_info_commands(self) -> dict[str, str]:
        return {
            "build_metadata": "cat /workspace/build-metadata.yaml 2>/dev/null || true",
        }

    @staticmethod
    def _strip_container_banner(text: str) -> str:
        """Strip container startup banners that precede YAML content.

        Docker images emit CUDA/PyTorch banners to stdout before
        command output. We find the first line that looks like a
        top-level YAML key (``word_chars: ...``) and discard
        everything above it.
        """
        import re

        for m in re.finditer(r"^[a-z]\w*: ", text, re.MULTILINE):
            # Skip URL-like lines (https: //, http: //)
            if text[m.start() : m.start() + 8].startswith(("https://", "http://")):
                continue
            return text[m.start() :]
        return text

    def process_version_info(self, raw: dict[str, str]) -> dict[str, str]:
        """Parse build-metadata.yaml and flatten with 'build_' prefix."""
        content = raw.get("build_metadata", "").strip()
        if not content:
            return {}
        try:
            import yaml

            content = self._strip_container_banner(content)
            data = yaml.safe_load(content)
            if not isinstance(data, dict):
                return {}
            return _flatten_dict(data, prefix="build")
        except Exception:
            logger.debug("Failed to parse build-metadata.yaml", exc_info=True)
            return {}

    # --- Long-term image resolution ---

    def resolve_long_term_image(
        self,
        container_image: str,
        runtime_info: dict[str, str],
        recipe: Recipe,
    ) -> tuple[str, bool]:
        """Resolve an eugr container image to a pinned GHCR nightly tag.

        Matches source hashes from the running container's
        ``build-metadata.yaml`` (available in *runtime_info* with
        ``build_`` prefix) against published GHCR nightly builds.

        Only attempts resolution for standard variants (no custom
        build_args beyond ``--tf5``).
        """
        # Determine the target GHCR image and package path
        ghcr_image, ghcr_pkg = self._resolve_ghcr_target(container_image, recipe)
        if not ghcr_image:
            return container_image, False

        # Extract the source commit hash — this is the primary match key
        repo_commit = runtime_info.get("build_build_script_commit", "").strip()
        if not repo_commit:
            logger.debug("No build_build_script_commit in runtime_info, cannot resolve long-term image")
            return container_image, False

        # Optional secondary hashes for tighter matching
        vllm_hash = runtime_info.get("build_vllm_commit", "").strip()
        flashinfer_hash = runtime_info.get("build_flashinfer_commit", "").strip()

        # Try build-index.json first (fast, cached)
        resolved = self._match_via_build_index(
            ghcr_image,
            repo_commit,
            vllm_hash,
            flashinfer_hash,
            recipe,
        )
        if resolved:
            return resolved, True

        assert ghcr_pkg is not None
        # Fall back to GHCR API tag enumeration
        resolved = self._match_via_ghcr_api(
            ghcr_image,
            ghcr_pkg,
            repo_commit,
            vllm_hash,
            flashinfer_hash,
        )
        if resolved:
            return resolved, True

        logger.debug("No matching GHCR tag found for commit %s", repo_commit)
        return container_image, False

    def _resolve_ghcr_target(
        self,
        container_image: str,
        recipe: Recipe,
    ) -> tuple[str, str] | tuple[None, None]:
        """Determine the GHCR image name and package path for resolution.

        Returns ``(None, None)`` if the image is not a standard eugr
        nightly variant eligible for resolution.
        """
        build_args = recipe.runtime_config.get("build_args", [])

        # Only resolve standard variants
        is_tf5 = build_args == ["--tf5"] or ("-tf5" in container_image)
        is_plain = not build_args

        if not (is_tf5 or is_plain):
            logger.debug("Custom build_args %r — skipping long-term resolution", build_args)
            return None, None

        # Match by local image name or GHCR :latest reference
        img_stripped = container_image.split(":")[0].strip()

        if is_tf5 or img_stripped in (LOCAL_EUGR_NIGHTLY_TF5, GHCR_EUGR_NIGHTLY_TF5):
            return GHCR_EUGR_NIGHTLY_TF5, GHCR_EUGR_PKG_TF5
        if img_stripped in (LOCAL_EUGR_NIGHTLY, GHCR_EUGR_NIGHTLY):
            return GHCR_EUGR_NIGHTLY, GHCR_EUGR_PKG

        # Image doesn't match any known eugr nightly pattern
        return None, None

    def _match_via_build_index(
        self,
        ghcr_image: str,
        repo_commit: str,
        vllm_hash: str,
        flashinfer_hash: str,
        recipe: Recipe,
    ) -> str | None:
        """Try to match via the spark-arena build-index.json."""
        from sparkrun.builders._ghcr import fetch_build_index

        # Determine variant suffix for filtering index entries
        variant = "nightly-tf5" if ghcr_image == GHCR_EUGR_NIGHTLY_TF5 else "nightly"

        # Resolve cache dir for index caching
        cache_dir = None
        try:
            from sparkrun.core.config import resolve_sparkrun_cache_dir

            cache_dir = resolve_sparkrun_cache_dir()
        except Exception:
            pass

        entries = fetch_build_index(
            EUGR_BUILD_INDEX_URL,
            cache_dir=cache_dir,
            cache_name=EUGR_BUILD_INDEX_CACHE_NAME,
        )
        if not entries:
            return None

        for entry in reversed(entries):  # newest first (append-only)
            if entry.get("variant") != variant:
                continue
            if entry.get("repo_commit", "").startswith(repo_commit[:12]):
                # Primary match on repo commit
                tag = entry.get("tag", "")
                if not tag:
                    continue
                # Optional secondary hash checks
                if vllm_hash and entry.get("vllm_hash") and not entry["vllm_hash"].startswith(vllm_hash[:12]):
                    continue
                if flashinfer_hash and entry.get("flashinfer_hash") and not entry["flashinfer_hash"].startswith(flashinfer_hash[:12]):
                    continue
                resolved = "%s:%s" % (ghcr_image, tag)
                logger.info("Resolved long-term image via build-index: %s", resolved)
                return resolved

        return None

    def _match_via_ghcr_api(
        self,
        ghcr_image: str,
        ghcr_pkg: str,
        repo_commit: str,
        vllm_hash: str,
        flashinfer_hash: str,
    ) -> str | None:
        """Fall back to GHCR API: enumerate tags and check OCI labels."""
        from sparkrun.builders._ghcr import ghcr_list_tags, ghcr_get_labels

        tags = ghcr_list_tags(ghcr_pkg)
        if not tags:
            return None

        # Check most recent tags first (YYYYMMDDNN sorts lexicographically)
        for tag in sorted(tags, reverse=True)[:10]:
            labels = ghcr_get_labels(ghcr_pkg, tag)
            if not labels:
                continue
            label_commit = labels.get("dev.sparkrun.repo-commit", "")
            if not label_commit or not label_commit.startswith(repo_commit[:12]):
                continue
            # Primary match — check optional secondary hashes
            if vllm_hash:
                label_vllm = labels.get("dev.sparkrun.vllm-hash", "")
                if label_vllm and not label_vllm.startswith(vllm_hash[:12]):
                    continue
            if flashinfer_hash:
                label_fi = labels.get("dev.sparkrun.flashinfer-hash", "")
                if label_fi and not label_fi.startswith(flashinfer_hash[:12]):
                    continue
            resolved = "%s:%s" % (ghcr_image, tag)
            logger.info("Resolved long-term image via GHCR labels: %s", resolved)
            return resolved

        return None

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
            clean_name = mod_name.removeprefix("mods/")
            mod_path = self._repo_dir / "mods" / clean_name
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
            recipe.pre_exec.append("export WORKSPACE_DIR=$PWD && cd %s && chmod +x run.sh && ./run.sh" % dest)

        logger.info("Injected %d mod(s) as pre_exec entries", len(mods))

    # --- Build cache ---

    @staticmethod
    def _cache_key(image: str, host: str | None = None) -> str:
        """Return the build-cache lookup key — host-qualified for delegated builds."""
        return "%s:%s" % (host, image) if host else image

    @staticmethod
    def _get_image_id_on_host(image: str, host: str, ssh_kwargs: dict | None = None) -> str | None:
        """Return the Docker image ID on a remote host, or None on failure."""
        from sparkrun.orchestration.primitives import run_script_on_host

        # TODO: should route via executor
        script = "docker image inspect --format '{{.Id}}' %s 2>/dev/null" % quote(image)
        result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=30)
        if result.success and result.stdout.strip():
            return result.stdout.strip()
        return None

    @staticmethod
    def _get_repo_head_on_host(repo_path: str, host: str, ssh_kwargs: dict | None = None) -> str | None:
        """Return ``git rev-parse HEAD`` on a remote host, or None on failure."""
        from sparkrun.orchestration.primitives import run_script_on_host

        script = "git -C %s rev-parse HEAD 2>/dev/null" % repo_path
        result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=30)
        if result.success and result.stdout.strip():
            return result.stdout.strip()
        return None

    def _resolve_cache_dir(self, config: SparkrunConfig | None) -> Path | None:
        """Resolve the cache directory for build cache storage."""
        try:
            from sparkrun.core.config import resolve_sparkrun_cache_dir

            return resolve_sparkrun_cache_dir(config.cache_dir if config else None)
        except Exception:
            return None

    def _can_skip_build(
        self,
        image: str,
        build_args: list[str],
        config: SparkrunConfig | None = None,
        host: str | None = None,
        ssh_kwargs: dict | None = None,
    ) -> bool:
        """Check whether a build can be skipped based on cached metadata.

        Returns True only when:
        - The build_args are cacheable (empty or ``["--tf5"]``)
        - A cache entry exists for this image with matching build_args
        - The Docker image still exists with the same image ID
        - The repo HEAD commit hasn't changed
        - Upstream wheel hashes (vLLM + FlashInfer) match the cached values

        When *host* is provided (delegated mode), Docker and git checks
        are performed on the remote host via SSH.  The cache file itself
        always lives on the control machine.
        """
        if build_args not in _CACHEABLE_BUILD_ARGS:
            return False

        cache_dir = self._resolve_cache_dir(config)
        if not cache_dir:
            return False

        cache = _load_build_cache(cache_dir)
        key = self._cache_key(image, host)
        entry = cache.get(key)
        if not entry:
            return False

        if entry.get("build_args", []) != build_args:
            return False

        # Verify image still exists with same ID
        if host:
            current_id = self._get_image_id_on_host(image, host, ssh_kwargs)
        else:
            from sparkrun.containers.registry import get_image_id

            current_id = get_image_id(image)
        if not current_id or current_id != entry.get("image_id"):
            return False

        # Check repo commit
        if not self._repo_dir:
            return False
        if host:
            current_commit = self._get_repo_head_on_host(str(self._repo_dir), host, ssh_kwargs)
            if not current_commit or current_commit != entry.get("repo_commit"):
                return False
        else:
            try:
                result = subprocess.run(
                    ["git", "-C", str(self._repo_dir), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    return False
                current_commit = result.stdout.strip()
                if current_commit != entry.get("repo_commit"):
                    return False
            except Exception:
                return False

        # Fetch upstream wheel hashes and compare
        upstream = _fetch_upstream_wheel_hashes()
        if not upstream:
            return False  # can't verify — safer to rebuild

        if upstream.get("vllm_commit") != entry.get("vllm_commit"):
            return False
        if upstream.get("flashinfer_commit") != entry.get("flashinfer_commit"):
            return False

        return True

    def _save_build_metadata(
        self,
        image: str,
        build_args: list[str],
        config: SparkrunConfig | None = None,
        host: str | None = None,
        ssh_kwargs: dict | None = None,
    ) -> None:
        """Save build metadata to the cache after a successful build.

        When *host* is provided (delegated mode), Docker and git checks
        are performed on the remote host via SSH.  The cache entry is
        stored under a host-qualified key.
        """
        cache_dir = self._resolve_cache_dir(config)
        if not cache_dir:
            return

        # Get repo HEAD commit
        repo_commit = None
        if self._repo_dir:
            if host:
                repo_commit = self._get_repo_head_on_host(str(self._repo_dir), host, ssh_kwargs)
            else:
                try:
                    result = subprocess.run(
                        ["git", "-C", str(self._repo_dir), "rev-parse", "HEAD"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        repo_commit = result.stdout.strip()
                except Exception:
                    pass

        # Extract build-metadata.yaml from the built image
        vllm_commit = None
        flashinfer_commit = None
        if host:
            try:
                from sparkrun.orchestration.primitives import run_script_on_host

                # TODO: inline script / should be on executor
                script = "docker run --rm %s cat /workspace/build-metadata.yaml" % quote(image)
                r = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=30)
                if r.success and r.stdout.strip():
                    info = self.process_version_info({"build_metadata": r.stdout})
                    vllm_commit = info.get("build_vllm_commit")
                    flashinfer_commit = info.get("build_flashinfer_commit")
            except Exception:
                logger.debug("Failed to extract build-metadata.yaml from %s on %s", image, host, exc_info=True)
        else:
            try:
                result = subprocess.run(
                    ["docker", "run", "--rm", quote(image), "cat", "/workspace/build-metadata.yaml"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    info = self.process_version_info({"build_metadata": result.stdout})
                    vllm_commit = info.get("build_vllm_commit")
                    flashinfer_commit = info.get("build_flashinfer_commit")
            except Exception:
                logger.debug("Failed to extract build-metadata.yaml from %s", image, exc_info=True)

        # Get image ID
        if host:
            image_id = self._get_image_id_on_host(image, host, ssh_kwargs)
        else:
            from sparkrun.containers.registry import get_image_id

            image_id = get_image_id(image)

        key = self._cache_key(image, host)
        cache = _load_build_cache(cache_dir)
        cache[key] = {
            "build_args": build_args,
            "repo_commit": repo_commit,
            "vllm_commit": vllm_commit,
            "flashinfer_commit": flashinfer_commit,
            "image_id": image_id,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_build_cache(cache_dir, cache)
        logger.debug("Saved build cache entry for '%s'", key)

    # --- Image building ---

    def _build_image(self, image: str, build_args: list[str], dry_run: bool = False) -> None:
        """Build container image via eugr's build-and-copy.sh.

        Args:
            image: Target image name (passed as ``-t``).
            build_args: Additional build arguments forwarded to the script.
            dry_run: Show what would be done without executing.
        """
        if not self._repo_dir:
            raise RuntimeError("Repository not initialized")
        build_script = self._repo_dir / "build-and-copy.sh"
        if not build_script.exists():
            raise RuntimeError("build-and-copy.sh not found at %s" % build_script)

        cmd = [str(build_script), "-t", quote(image)] + quote_list(build_args)
        logger.info("Building eugr container: %s", " ".join(cmd))

        if dry_run:
            return

        # Stream build output only at INFO+ (i.e. -v); at default verbosity capture silently
        stream = logging.getLogger().isEnabledFor(logging.INFO)
        if stream:
            result = subprocess.run(cmd, cwd=str(self._repo_dir))
        else:
            result = subprocess.run(cmd, cwd=str(self._repo_dir), capture_output=True, text=True)
            if result.stdout and isinstance(result.stdout, str):
                logger.debug("Build stdout:\n%s", result.stdout[-2000:])
            if result.stderr and isinstance(result.stderr, str):
                logger.debug("Build stderr:\n%s", result.stderr[-2000:])
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

        script = "docker image inspect %s >/dev/null 2>&1" % quote(image)
        result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=30)
        return result.success

    def _ensure_repo_remote(
        self,
        head: str,
        ssh_kwargs: dict | None = None,
        dry_run: bool = False,
        branch: str | None = None,
    ) -> str:
        """Clone or update the eugr repo on the head node.

        Args:
            head: Head node hostname.
            ssh_kwargs: SSH connection kwargs.
            dry_run: Show what would be done without executing.
            branch: Optional git branch to checkout (for developer builds).

        Returns the remote path where the repo lives.
        """
        from sparkrun.orchestration.primitives import run_script_on_host

        # TODO: hard-coded inline script
        remote_path = "~/.cache/sparkrun/eugr-spark-vllm-docker"

        if branch:
            # Clone with specific branch or fetch+checkout if already cloned
            script = (
                "set -e\n"
                "REPO_DIR=%(path)s\n"
                'if [ -d "$REPO_DIR/.git" ]; then\n'
                '  git -C "$REPO_DIR" fetch origin\n'
                '  git -C "$REPO_DIR" checkout %(branch)s\n'
                '  git -C "$REPO_DIR" pull --ff-only || true\n'
                "else\n"
                '  mkdir -p "$(dirname "$REPO_DIR")"\n'
                '  git clone -b %(branch)s %(url)s "$REPO_DIR"\n'
                "fi\n"
                'echo "$REPO_DIR"\n'
            ) % {"path": remote_path, "url": EUGR_REPO_URL, "branch": quote(branch)}
        else:
            script = (
                "set -e\n"
                "REPO_DIR=%(path)s\n"
                'if [ -d "$REPO_DIR/.git" ]; then\n'
                '  git -C "$REPO_DIR" pull --ff-only || true\n'
                "else\n"
                '  mkdir -p "$(dirname "$REPO_DIR")"\n'
                '  git clone %(url)s "$REPO_DIR"\n'
                "fi\n"
                'echo "$REPO_DIR"\n'
            ) % {"path": remote_path, "url": EUGR_REPO_URL}

        if branch:
            logger.info("Ensuring eugr repo on head node %s (branch: %s)...", head, branch)
        else:
            logger.info("Ensuring eugr repo on head node %s...", head)

        if dry_run:
            return remote_path

        result = run_script_on_host(head, script, ssh_kwargs=ssh_kwargs, timeout=120)
        if not result.success:
            raise RuntimeError("Failed to ensure eugr repo on %s: %s" % (head, result.stderr.strip() if result.stderr else "(no output)"))

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
        from sparkrun.orchestration.ssh import run_remote_script_streaming

        # TODO: hard-coded inline script
        remote_path = "~/.cache/sparkrun/eugr-spark-vllm-docker"
        args_str = args_list_to_shell_str(build_args)
        script = "set -e\ncd %(path)s\nchmod +x build-and-copy.sh\n./build-and-copy.sh -t %(image)s %(args)s\n" % {
            "path": remote_path,
            "image": quote(image),
            "args": args_str,
        }

        logger.info("Building eugr container on %s: %s -t %s %s", head, remote_path, image, args_str)

        if dry_run:
            return

        kw = ssh_kwargs or {}
        result = run_remote_script_streaming(
            head,
            script,
            ssh_user=kw.get("ssh_user"),
            ssh_key=kw.get("ssh_key"),
            ssh_options=kw.get("ssh_options"),
            timeout=1800,
            quiet=not logging.getLogger().isEnabledFor(logging.INFO),
        )
        if not result.success:
            raise RuntimeError("eugr remote container build failed on %s (exit %d)" % (head, result.returncode))

    # --- Repo management ---

    def ensure_repo(
        self,
        cache_dir: Path | None = None,
        registry_cache_root: Path | None = None,
        branch: str | None = None,
    ) -> Path:
        """Clone or update the eugr repo in sparkrun's cache.

        If the registry system already has a cached clone of the eugr-vllm
        repo (from recipe syncing), reuses it instead of cloning a second
        copy.  Sparse checkout is disabled on the registry clone so that
        scripts like ``build-and-copy.sh`` are available.

        Args:
            cache_dir: Override cache directory.
            registry_cache_root: Registry cache root to check for existing clones.
            branch: Optional git branch to checkout (for developer builds).
        """
        # Check if registry already has this repo cloned
        if registry_cache_root is not None:
            registry_repo = registry_cache_root / "eugr-vllm"
            if (registry_repo / ".git").exists():
                logger.info("Reusing eugr repo from registry cache: %s", registry_repo)
                self._ensure_full_checkout(registry_repo)
                self._update_repo(registry_repo, branch=branch)
                return registry_repo

        if cache_dir is None:
            if self._v is not None:
                cache_dir = Path(get_working_path(v=self._v)) / "cache"
            else:
                from sparkrun.core.config import resolve_sparkrun_cache_dir

                cache_dir = resolve_sparkrun_cache_dir()
        repo_dir = cache_dir / "eugr-spark-vllm-docker"

        if repo_dir.exists() and (repo_dir / ".git").exists():
            self._update_repo(repo_dir, branch=branch)
        else:
            if branch:
                logger.info("Cloning eugr/spark-vllm-docker (branch: %s)...", branch)
            else:
                logger.info("Cloning eugr/spark-vllm-docker...")
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["git", "clone"]
            if branch:
                cmd += ["-b", quote(branch)]
            cmd += [EUGR_REPO_URL, str(repo_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError("Failed to clone eugr repo: %s" % result.stderr.strip())

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
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _update_repo(repo_dir: Path, branch: str | None = None) -> None:
        """Pull latest changes for an existing repo clone.

        When *branch* is specified, fetches and checks out that branch
        before pulling — useful for developer builds from feature branches.
        """
        if branch:
            logger.info("Updating eugr/spark-vllm-docker repo (branch: %s)...", branch)
            subprocess.run(
                ["git", "-C", str(repo_dir), "fetch", "origin"],
                capture_output=True,
                text=True,
            )
            checkout = subprocess.run(
                ["git", "-C", str(repo_dir), "checkout", quote(branch)],
                capture_output=True,
                text=True,
            )
            if checkout.returncode != 0:
                logger.warning(
                    "Failed to checkout branch '%s' (continuing with current): %s",
                    branch,
                    checkout.stderr.strip(),
                )
                return
        else:
            logger.info("Updating eugr/spark-vllm-docker repo...")

        result = subprocess.run(
            ["git", "-C", str(repo_dir), "pull", "--ff-only"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "Failed to update eugr repo (continuing with existing): %s",
                result.stderr.strip(),
            )
