"""Drift guard between the TensorFold image pin and the platform default.

``docker/tensorfold/image.env`` is the one place the engine version (and so the
image tag) is recorded.  A platform default naming a tag nobody built fails
minutes into a launch on the cluster as an unexplained pull error, which is
what this test exists to catch in the repository instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sparkrun.platforms.dgx_spark import DgxSparkPlatform

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_ENV = REPO_ROOT / "docker" / "tensorfold" / "image.env"
DOCKERFILE = REPO_ROOT / "docker" / "tensorfold" / "Dockerfile"
DSV41_ENV = REPO_ROOT / "docker" / "tensorfold-dsv41" / "image.env"
DSV41_DOCKERFILE = REPO_ROOT / "docker" / "tensorfold-dsv41" / "Dockerfile"
DSV41_PATCHES = REPO_ROOT / "docker" / "tensorfold-dsv41" / "patches"

needs_repo = pytest.mark.skipif(not IMAGE_ENV.is_file(), reason="requires the repo checkout")


def _load_env_from(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _load_env() -> dict[str, str]:
    return _load_env_from(IMAGE_ENV)


@needs_repo
def test_platform_default_matches_the_pin():
    env = _load_env()
    expected = "%s:%s" % (env["IMAGE_REPO"], env["IMAGE_TAG"])
    actual = DgxSparkPlatform().default_image("tensorfold")
    assert actual == expected, "DgxSparkPlatform.default_image('tensorfold') disagrees with docker/tensorfold/image.env; bump both together"


@needs_repo
def test_image_tag_is_the_engine_version():
    env = _load_env()
    assert env["IMAGE_TAG"] == env["TF_VERSION"]
    assert not env["IMAGE_TAG"].endswith(":latest")


@needs_repo
def test_dockerfile_pins_release_and_nccl():
    text = DOCKERFILE.read_text()
    assert "ARG TF_VERSION" in text
    assert "ARG NCCL_APT_VERSION" in text
    assert "sha256sum -c" in text, "the release archive must be verified by its published checksum"
    assert "NCCL_APT_VERSION}" in text


@needs_repo
def test_nccl_pin_is_at_least_the_engine_minimum():
    """TensorFold 1.1.x refuses NCCL < 2.30 for native tensor-parallel ranks."""
    env = _load_env()
    package = env["NCCL_APT_VERSION"]
    version = package.split("-", 1)[0]
    major, minor = (int(part) for part in version.split(".")[:2])
    assert (major, minor) >= (2, 30), "NCCL %s is below TensorFold's 2.30 minimum" % version


@needs_repo
def test_dsv41_platform_default_matches_the_pin():
    env = _load_env_from(DSV41_ENV)
    expected = "%s:%s" % (env["IMAGE_REPO"], env["IMAGE_TAG"])
    actual = DgxSparkPlatform().default_image("tensorfold-dsv41")
    assert actual == expected, "the tensorfold-dsv41 platform default disagrees with docker/tensorfold-dsv41/image.env"


@needs_repo
def test_dsv41_image_tag_is_the_fork_commit():
    env = _load_env_from(DSV41_ENV)
    assert env["IMAGE_TAG"] == env["TF_FORK_SHA"][:7]


@needs_repo
def test_dsv41_dockerfile_pins_fork_and_patches():
    text = DSV41_DOCKERFILE.read_text()
    assert "codeload.github.com/bertholomus/TensorFold" in text
    assert "ARG TF_SHA" in text
    assert "patch -p1" in text
    patches = sorted(path.name for path in DSV41_PATCHES.glob("*.patch"))
    assert len(patches) == 3, "the pinned TP2 build applies three patches; got %r" % patches


@needs_repo
def test_dsv41_bundle_helpers_are_shipped():
    assert (REPO_ROOT / "docker" / "tensorfold-dsv41" / "chat_template.jinja").is_file()
    assert (REPO_ROOT / "scripts" / "tensorfold-dsv41-prepare.sh").is_file()
