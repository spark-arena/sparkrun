"""Docker image reference parsing (``sparkrun.utils.images``).

These encode the two grammar rules sparkrun previously got wrong:
a registry host is not implied by a slash, and a colon is not always a tag.
"""

from __future__ import annotations

import pytest

from sparkrun.utils.images import (
    image_has_explicit_version,
    is_pullable_image_ref,
    parse_image_ref,
)


class TestParseImageRef:
    @pytest.mark.parametrize(
        "ref,repository,tag,digest",
        [
            # Docker Hub short form -- the shape that regressed.
            ("vllm/vllm-openai:qwen38-flash-next", "vllm/vllm-openai", "qwen38-flash-next", None),
            ("vllm/vllm-openai", "vllm/vllm-openai", None, None),
            # Explicit registry host.
            ("docker.io/vllm/vllm-openai:x", "docker.io/vllm/vllm-openai", "x", None),
            ("ghcr.io/org/img:latest", "ghcr.io/org/img", "latest", None),
            # A colon in the FIRST component is a port, not a tag.
            ("myreg.io:5000/foo", "myreg.io:5000/foo", None, None),
            ("myreg.io:5000/foo:v1", "myreg.io:5000/foo", "v1", None),
            ("localhost:5000/foo", "localhost:5000/foo", None, None),
            # Bare names.
            ("vllm-node", "vllm-node", None, None),
            ("my-image:latest", "my-image", "latest", None),
            # Digests.
            ("ghcr.io/o/i@sha256:abc123", "ghcr.io/o/i", None, "sha256:abc123"),
        ],
    )
    def test_parse(self, ref, repository, tag, digest):
        assert parse_image_ref(ref) == (repository, tag, digest)

    def test_empty(self):
        assert parse_image_ref("") == ("", None, None)


class TestIsPullableImageRef:
    @pytest.mark.parametrize(
        "ref",
        [
            "vllm/vllm-openai:qwen38-flash-next",
            "docker.io/vllm/vllm-openai:x",
            "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:latest",
            "nvcr.io/nvidia/vllm:x",
            "eugr/spark-vllm:latest",
            "myreg.io:5000/foo:v1",
            "localhost:5000/foo",
        ],
    )
    def test_pullable(self, ref):
        assert is_pullable_image_ref(ref) is True

    @pytest.mark.parametrize(
        "ref",
        [
            # Single-component names stay local: these are the eugr build tags,
            # and treating them as pullable would skip the wheels build.
            "vllm-node",
            "vllm-node-tf5",
            "my-image",
            "my-image:latest",
            "sparkrun-eugr-vllm",
            "",
        ],
    )
    def test_not_pullable(self, ref):
        assert is_pullable_image_ref(ref) is False

    def test_docker_io_prefix_is_not_required(self):
        """The user-facing workaround must stop being necessary."""
        bare = "vllm/vllm-openai:qwen38-flash-next"
        assert is_pullable_image_ref(bare) == is_pullable_image_ref("docker.io/" + bare)


class TestVersionHelpers:
    @pytest.mark.parametrize(
        "ref,expected",
        [
            ("vllm/vllm-openai:qwen38-flash-next", True),
            ("ghcr.io/o/i@sha256:abc", True),
            ("vllm/vllm-openai", False),
            # A registry port is not a version pin.
            ("myreg.io:5000/foo", False),
        ],
    )
    def test_has_explicit_version(self, ref, expected):
        assert image_has_explicit_version(ref) is expected
