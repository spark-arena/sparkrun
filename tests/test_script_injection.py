"""Recipe-sourced values must not reach a generated shell script as code.

The model ensure/distribute scripts are built with ``str.format()`` and run on
every cluster host.  Several of the interpolated values are **recipe content**
— the model id, a GGUF quant suffix, ``model_revision``,
``cluster_config.remote_cache_dir`` — and recipes come from registries.  They
were substituted as bare text inside double quotes, where ``$(...)`` still
evaluates; the ``revision`` case was confirmed to execute (see
``test_model_cache_check``).

Two tools, and picking the wrong one silently breaks things:

* :func:`quote` for values with no expansion to preserve (ids, quants).
* :func:`validate_interpolated_path` for cache paths, which are emitted
  double-quoted **so that** a leading ``~/`` or ``$HOME/`` expands on the
  target.  Quoting those would point the cache at a literal ``$HOME`` and
  re-download the weights every launch.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from sparkrun.models.distribute import _build_model_ensure_script
from sparkrun.utils.shell import ShellSafetyError, validate_interpolated_path

needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="requires a POSIX shell",
)


def _run(script: str, bindir=None) -> subprocess.CompletedProcess:
    env = {"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"}
    if bindir is not None:
        env["PATH"] = f"{bindir}:{env['PATH']}"
    return subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=30, env=env)


def _stub_bin(tmp_path):
    bindir = tmp_path / "stub-bin"
    bindir.mkdir()
    stub = bindir / "huggingface-cli"
    stub.write_text('#!/bin/bash\necho "STUB-DOWNLOAD $*"\n')
    stub.chmod(0o755)
    return bindir


# ---------------------------------------------------------------------------
# validate_interpolated_path
# ---------------------------------------------------------------------------


class TestValidateInterpolatedPath:
    @pytest.mark.parametrize(
        "value",
        [
            "/mnt/huggingface",
            "$HOME/.cache/huggingface",
            "${HOME}/.cache/huggingface",
            "~/.cache/huggingface",
            # Double quotes make these literal, and a cache dir with a space
            # works today — rejecting it would be a regression for no gain.
            "/mnt/my models/hf",
            "/mnt/a;b&c|d",
        ],
    )
    def test_accepts(self, value):
        assert validate_interpolated_path(value, field_name="cache_dir") == value

    @pytest.mark.parametrize(
        "value",
        [
            "/mnt/$(id -u)",
            "/mnt/`id -u`",
            '/mnt/"; id; "',
            "/mnt/a\\b",
            "/mnt/a\nid\n",
            "/mnt/$USER/hf",
        ],
    )
    def test_rejects(self, value):
        with pytest.raises(ShellSafetyError):
            validate_interpolated_path(value, field_name="cache_dir")

    def test_home_expansion_survives_into_the_script(self):
        """The whole reason these are validated rather than quoted."""
        script = _build_model_ensure_script("org/model", "$HOME/.cache/huggingface")

        assert 'CACHE_PATH="$HOME/.cache/huggingface/hub/models--org--model"' in script


# ---------------------------------------------------------------------------
# Recipe-sourced values, executed for real
# ---------------------------------------------------------------------------


def _payload(tmp_path, name):
    """A value that leaves a file behind iff the shell evaluates it."""
    return "$(touch " + str(tmp_path / name) + ")"


@needs_bash
class TestRefusedOutright:
    """Values that also form the cache directory cannot be quoted away.

    ``model_cache_path`` folds the repo id into ``models--<org>--<name>``, so a
    hostile id reaches a path that must stay expandable.  There is no quoting
    that both neutralizes it and preserves ``$HOME`` — so it is rejected before
    a script exists at all, which is the stronger outcome.
    """

    def test_model_id(self, tmp_path):
        with pytest.raises(ShellSafetyError):
            _build_model_ensure_script(_payload(tmp_path, "pwn_model"), "/hf")

        assert not (tmp_path / "pwn_model").exists()

    def test_gguf_repo_id(self, tmp_path):
        with pytest.raises(ShellSafetyError):
            _build_model_ensure_script(_payload(tmp_path, "pwn_repo") + "-GGUF:Q4_K_M", "/hf")

        assert not (tmp_path / "pwn_repo").exists()

    def test_cache_dir(self, tmp_path):
        with pytest.raises(ShellSafetyError):
            _build_model_ensure_script("org/model", _payload(tmp_path, "pwn_cache"))

        assert not (tmp_path / "pwn_cache").exists()


@needs_bash
class TestNeutralizedByQuoting:
    """Values with no expansion to preserve are quoted and run harmlessly.

    These do *not* reach the cache path, so the script is built and executed —
    the payload has to survive as inert text rather than be refused.
    """

    def test_gguf_quant(self, tmp_path):
        """`quant` lands inside a glob (`-name "*<quant>*.gguf"`), still evaluated."""
        model = "org/repo-GGUF:" + _payload(tmp_path, "pwn_quant")

        proc = _run(_build_model_ensure_script(model, str(tmp_path)), _stub_bin(tmp_path))

        assert not (tmp_path / "pwn_quant").exists()
        assert proc.returncode == 0

    def test_revision(self, tmp_path):
        script = _build_model_ensure_script("org/model", str(tmp_path), revision=_payload(tmp_path, "pwn_rev"))

        proc = _run(script, _stub_bin(tmp_path))

        assert not (tmp_path / "pwn_rev").exists()
        assert proc.returncode == 0


@needs_bash
class TestQuotingDidNotBreakBehaviour:
    """Quoting is only correct if the ordinary values still work end to end."""

    def test_ordinary_model_still_hits_its_cache(self, tmp_path):
        snap = tmp_path / "hub" / "models--org--model" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").write_text("w")

        proc = _run(_build_model_ensure_script("org/model", str(tmp_path)), _stub_bin(tmp_path))

        assert "already cached" in proc.stdout

    def test_quant_still_matches_as_a_glob_fragment(self, tmp_path):
        """`quote()` on the quant would yield "*'Q4_K_M'*.gguf" and match nothing."""
        snap = tmp_path / "hub" / "models--org--repo-GGUF" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "repo-Q4_K_M-00001.gguf").write_text("w")

        proc = _run(_build_model_ensure_script("org/repo-GGUF:Q4_K_M", str(tmp_path)), _stub_bin(tmp_path))

        assert "already cached" in proc.stdout

    def test_download_receives_the_literal_id(self, tmp_path):
        proc = _run(_build_model_ensure_script("org/model", str(tmp_path)), _stub_bin(tmp_path))

        assert "STUB-DOWNLOAD download org/model" in proc.stdout
