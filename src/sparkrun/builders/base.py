"""Base class for sparkrun builders."""

from __future__ import annotations

import logging
from logging import Logger
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

from scitrera_app_framework import Plugin, Variables

from sparkrun.utils.shell import quote

if TYPE_CHECKING:
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.validation import RecipeIssue

EXT_BUILDER = "sparkrun.builder"

# Fully qualified image prefixes that indicate a pullable registry image
# (no build needed even if the image isn't present locally).
#
# DEPRECATED as a decision procedure -- kept only because it is importable API.
# Prefer :func:`sparkrun.utils.images.is_pullable_image_ref`, which implements
# Docker's actual reference grammar.  Matching on host prefixes alone rejects
# the canonical short form of every Docker Hub image (``vllm/vllm-openai:tag``
# carries no host), which the eugr builder then treated as a local image that
# had gone missing and silently replaced with its own nightly.
PULLABLE_REGISTRY_PREFIXES = (
    "docker.io/",
    "ghcr.io/",
    "nvcr.io/",
    "quay.io/",
    "registry.hub.docker.com/",
    "public.ecr.aws/",
    "gcr.io/",
)

logger = logging.getLogger(__name__)


class BuilderUnavailableError(ValueError):
    """A recipe named a real builder that is gated off by a feature flag.

    Distinct from the plain ``ValueError`` raised for an *unknown* builder,
    which callers may reasonably warn about and skip. This one must not be
    skipped: the user named a builder that exists, and for an environment
    builder (a venv the serve command depends on) continuing without it would
    launch the workload under the wrong interpreter. Mirrors
    ``ExecutorUnavailableError`` — an explicitly-requested but unavailable
    plugin fails loudly rather than silently downgrading.
    """


def _flatten_dict(d: dict, prefix: str = "", sep: str = "_", normalize: bool = False) -> dict[str, str]:
    """Recursively flatten a nested dict, joining keys with *sep*.

    When *normalize* is True, dots, slashes, and dashes in keys are
    replaced with *sep* — useful for OCI labels that use dotted names
    like ``org.opencontainers.image.version``.

    >>> _flatten_dict({"version": "1.0", "git": {"commit": "abc"}}, prefix="build")
    {'build_version': '1.0', 'build_git_commit': 'abc'}
    """
    result: dict[str, str] = {}
    for k, v in d.items():
        if normalize:
            k = k.replace(".", sep).replace("/", sep).replace("-", sep)
        full_key = "%s%s%s" % (prefix, sep, k) if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, prefix=full_key, sep=sep, normalize=normalize))
        else:
            result[full_key] = str(v)
    return result


class BuilderPlugin(Plugin):
    """Abstract base class for sparkrun image builders.

    Each builder is an SAF Plugin that registers as a multi-extension
    under the 'sparkrun.builder' extension point.

    Subclasses must define:
        - builder_name: str identifier (e.g. "docker-pull", "eugr")

    Optionally:
        - builder_aliases: alternate spellings ``builder:`` accepts
        - required_feature_flag: gate this builder behind a feature flag
    """

    eager = False
    builder_name: str = ""

    #: Alternate names :func:`~sparkrun.core.bootstrap.get_builder` also
    #: accepts.  Deliberately *not* surfaced by ``list_builders`` — an alias is
    #: another spelling of one builder, and listing it would imply a second one
    #: exists.
    builder_aliases: ClassVar[tuple[str, ...]] = ()

    #: Feature flag gating this builder, or ``None`` for always-available.
    #: Mirrors :class:`~sparkrun.orchestration.executors._base.Executor` and
    #: :class:`~sparkrun.transports.base.Transport`.
    required_feature_flag: ClassVar[str | None] = None

    #: Whether :meth:`prepare` rewrites the image reference it is handed.
    #:
    #: ``prepare()`` is single-valued, so a recipe declaring per-machine images
    #: (``containers:``) cannot be served by a builder that transforms one — and
    #: calling it once per image is not an option when a build is minutes long.
    #: The launcher fails fast on that combination.  An *environment* builder
    #: (``uv-venv``) returns the ref untouched and sets this ``False``, so it
    #: composes with per-machine images freely.  Precedent:
    #: :attr:`~sparkrun.orchestration.executors._base.Executor.needs_image`.
    transforms_image: ClassVar[bool] = True

    def name(self) -> str:
        return "sparkrun.builder.%s" % self.builder_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_BUILDER

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        # SAF only exposes a multi-extension plugin (via get_extensions) when
        # this returns True at registration.  A gated builder hides itself here
        # — it stays in the plugin registry but is absent from get_extensions /
        # list_builders / resolution, so naming it fails closed
        # (BuilderUnavailableError).  See core.features.
        if self.required_feature_flag:
            from sparkrun.core.features import feature_gate_enabled

            return feature_gate_enabled(self.required_feature_flag, v)
        return True

    def matches_name(self, name: str) -> bool:
        """True when *name* is this builder's canonical name or an alias."""
        return name == self.builder_name or name in self.builder_aliases

    def initialize(self, v: Variables, logger: Logger) -> BuilderPlugin:
        return self

    def prepare(
        self,
        image: str,
        recipe: Recipe,
        hosts: list[str],
        config: SparkrunConfig | None = None,
        dry_run: bool = False,
        transfer_mode: str = "local",
        ssh_kwargs: dict | None = None,
        builder_context: Mapping[str, Any] | None = None,
    ) -> str:
        """Prepare the execution environment. Returns the final image name.

        This is the canonical hook the launcher calls in its builder
        phase. The default implementation delegates to
        :meth:`prepare_image`, so image builders keep overriding
        ``prepare_image`` unchanged (back-compat). Environment builders
        (e.g. a host-side python venv) override ``prepare`` directly to do
        their host-side setup and return the image ref (usually unchanged).
        """
        return self.prepare_image(
            image,
            recipe,
            hosts,
            config=config,
            dry_run=dry_run,
            transfer_mode=transfer_mode,
            ssh_kwargs=ssh_kwargs,
        )

    def default_env_file(self, recipe: Recipe) -> str | None:
        """Return a shell env_file this builder produces, or ``None``.

        Environment builders that produce a shell env_file (sourced before
        the serve command, e.g. a venv activation script) return its path;
        the local executor auto-populates its ``env_file`` from this when the
        recipe doesn't set one explicitly.
        """
        return None

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
        """Ensure image is available. Returns final image name.

        Called before the distribution phase. After this returns,
        the image should exist locally (or on the head node when
        *transfer_mode* is ``"delegated"``) so distribution can sync
        it to remote hosts.

        Args:
            image: Target image name.
            recipe: The loaded recipe.
            hosts: Target host list (first element is head).
            config: SparkrunConfig for cache dir resolution.
            dry_run: Show what would be done without executing.
            transfer_mode: ``"local"`` (build locally) or
                ``"delegated"`` (build on head node via SSH).
            ssh_kwargs: SSH connection kwargs (needed for delegated mode).
        """
        return image

    def version_info_commands(self) -> dict[str, str]:
        """Return label→shell command pairs for raw data capture from container.

        Unlike runtime version_commands (single value per command), these
        commands can produce multi-line output. Raw stdout is passed to
        process_version_info() for Python-side processing.
        """
        return {}

    def process_version_info(self, raw: dict[str, str]) -> dict[str, str]:
        """Process raw command outputs into flat key-value pairs.

        Args:
            raw: {label: raw_stdout} from version_info_commands().
        Returns:
            Flat dict to merge into runtime_info.
        """
        return {}

    def collect_container_labels(
        self,
        container_name: str,
        host: str,
        ssh_kwargs: dict,
    ) -> dict[str, str]:
        """Inspect container labels and return as flat dict with 'container_' prefix.

        Default implementation uses ``docker inspect``. Subclasses may
        override for alternative container engines (podman, etc.).
        Fails silently — label collection never blocks a launch.
        """
        try:
            import json
            from sparkrun.orchestration.primitives import run_script_on_host

            script = "docker inspect --format '{{json .Config.Labels}}' %s 2>/dev/null || echo '{}'" % quote(container_name)
            result = run_script_on_host(host, script, ssh_kwargs=ssh_kwargs, timeout=15)
            if result.returncode != 0 or not result.stdout.strip():
                return {}
            raw = result.stdout.strip()
            data = json.loads(raw)
            if not isinstance(data, dict) or not data:
                return {}
            return _flatten_dict(data, prefix="container", normalize=True)
        except Exception:
            logger.debug("Container label collection failed", exc_info=True)
            return {}

    def resolve_long_term_image(
        self,
        container_image: str,
        runtime_info: dict[str, str],
        recipe: Recipe,
    ) -> tuple[str, bool]:
        """Resolve a container image to a long-term pinned reference.

        Builders that publish to public registries with dated tags can
        override this to map an ephemeral image (e.g. ``:latest`` or a
        locally-built name) to a reproducible ``YYYYMMDDNN`` tag by
        matching source hashes from *runtime_info* against known builds.

        Returns:
            ``(image_ref, pinned)`` where *pinned* is ``True`` when the
            image was successfully resolved to a stable tag.
        """
        return container_image, False

    def validate_recipe(self, recipe: Recipe) -> list["str | RecipeIssue"]:
        """Return builder-specific findings for *recipe*.

        Same two return forms as
        :meth:`~sparkrun.runtimes.base.RuntimePlugin.validate_recipe`: a plain
        ``str`` leaves severity undeclared and is reported as a *suggestion*
        (never fatal by default, not shown at launch), a
        :class:`~sparkrun.core.validation.RecipeIssue` declares it.  Use
        :meth:`recipe_error` for a recipe this builder cannot build — that
        aborts the launch rather than letting ``prepare()`` fail after image
        and model distribution have already run — and :meth:`recipe_warning`
        for one that builds but not reproducibly elsewhere.
        """
        return []

    def recipe_error(self, message: str, code: str = "builder-field") -> "RecipeIssue":
        """Build a launch-blocking :class:`RecipeIssue` tagged with this builder."""
        from sparkrun.core.validation import ERROR, RecipeIssue

        return RecipeIssue(ERROR, code, "[%s] %s" % (self.builder_name, message))

    def recipe_warning(self, message: str, code: str = "builder-field") -> "RecipeIssue":
        """Build a portability-class :class:`RecipeIssue` tagged with this builder."""
        from sparkrun.core.validation import WARNING, RecipeIssue

        return RecipeIssue(WARNING, code, "[%s] %s" % (self.builder_name, message))

    def __repr__(self) -> str:
        return "%s(builder_name=%r)" % (self.__class__.__name__, self.builder_name)
