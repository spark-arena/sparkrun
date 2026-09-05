"""The ``uv`` supply-chain pin (``core/tooling.py``) reaches every install site.

sparkrun fetches ``uv`` from the internet and runs it on *cluster hosts*, from
two places: the model ensure scripts (when a host has no HuggingFace client)
and the ``uv-venv`` builder.  Both used the unversioned installer::

    curl -LsSf https://astral.sh/uv/install.sh | sh

which is whatever Astral published that morning, with nothing recording what
landed on the nodes and nothing to bump when it needs changing.

The value of centralizing the pin is entirely in nothing being able to drift
back, so that is what is asserted here rather than the constant's value.
"""

from __future__ import annotations

import re
import types

import pytest

from sparkrun.builders.uv_venv import _provision_script, _resolve_spec
from sparkrun.core.tooling import (
    UV_INSTALL_BIN_DIR,
    UV_INSTALL_URL,
    UV_VERSION,
    uv_pip_spec,
)
from sparkrun.models.distribute import _build_model_ensure_script

#: Every rendered script that may install uv.
_RENDERED = {
    "model_sync.sh": lambda: _build_model_ensure_script("org/model", "/hf"),
    "model_sync_gguf.sh": lambda: _build_model_ensure_script("org/repo-GGUF:Q4_K_M", "/hf"),
}

#: The unversioned installer — tracks latest, must appear nowhere.
_UNVERSIONED = "https://astral.sh/uv/install.sh"


class TestPinShape:
    def test_version_is_concrete(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+", UV_VERSION), UV_VERSION

    def test_url_is_versioned(self):
        assert UV_INSTALL_URL == f"https://astral.sh/uv/{UV_VERSION}/install.sh"
        assert UV_INSTALL_URL != _UNVERSIONED

    def test_pip_spec_is_an_equality_pin(self):
        """A floor (``>=``) would let two hosts get different builds."""
        assert uv_pip_spec() == f"uv=={UV_VERSION}"


class TestModelEnsureScripts:
    @pytest.mark.parametrize("name", sorted(_RENDERED))
    def test_carries_the_pinned_url(self, name):
        script = _RENDERED[name]()

        assert UV_INSTALL_URL in script
        assert UV_VERSION in script

    @pytest.mark.parametrize("name", sorted(_RENDERED))
    def test_never_the_unversioned_url(self, name):
        assert _UNVERSIONED not in _RENDERED[name]()

    @pytest.mark.parametrize("name", sorted(_RENDERED))
    def test_failure_guidance_is_actionable(self, name):
        """An air-gapped host is a fact about the host, not a broken recipe.

        The message must name a way out; the previous "failed to install uv"
        named none.
        """
        script = _RENDERED[name]()

        assert "no outbound network access" in script
        assert "transfer_mode" in script
        assert "pre-place the weights" in script


class TestUvVenvBuilder:
    """The builder shares the pin — it is the other host-mutating install site."""

    def _script(self):
        recipe = types.SimpleNamespace(
            builder="uv-venv",
            builder_config={"venv_path": "/v", "requirements": ["vllm"]},
            name="demo",
            source_path=None,
        )
        return _provision_script(_resolve_spec(recipe))

    def test_carries_the_pinned_url_and_spec(self):
        script = self._script()

        assert UV_INSTALL_URL in script
        assert uv_pip_spec() in script

    def test_never_the_unversioned_url(self):
        assert _UNVERSIONED not in self._script()

    def test_fails_with_guidance_when_uv_cannot_be_installed(self):
        """Previously `uv venv` just reported "command not found" under set -e."""
        script = self._script()

        assert "could not be installed" in script
        assert "no outbound network access" in script

    def test_path_uses_the_central_bin_dir(self):
        assert UV_INSTALL_BIN_DIR in self._script()
