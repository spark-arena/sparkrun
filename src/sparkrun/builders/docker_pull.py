"""Default docker-pull builder — no-op (distribution handles pulling)."""

from __future__ import annotations

from sparkrun.builders.base import BuilderPlugin


class DockerPullBuilder(BuilderPlugin):
    """Default builder that relies on the distribution phase for image pulling.

    This is the fallback builder when no explicit builder is specified
    in the recipe. It does nothing in prepare_image() because sparkrun's
    distribution layer already handles docker pull/save/load.
    """

    builder_name = "docker-pull"
