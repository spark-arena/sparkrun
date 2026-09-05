"""Remote model cache check (``model_sync.sh`` / ``model_sync_gguf.sh``) — issue #291.

Both scripts used to mangle the HuggingFace repo id themselves::

    SAFE_NAME=$(echo "{model_id}" | tr '/' '--')

``tr`` truncates SET2 to the length of SET1, so ``/`` became a *single*
hyphen: ``org/model`` probed ``models--org-model`` while the weights (and the
rsync destination built from :func:`model_cache_path`) live at
``models--org--model``.  Every ``org/model`` recipe therefore missed its own
cache and fell through to the download path — a wasted Hub round-trip at best,
and a hard launch failure on a host with no ``huggingface-cli``/``uvx``, no
egress, or no token for a gated repo, despite the weights being present.

The mangling now has one implementation (``model_cache_path``) and is rendered
into the script as ``{cache_path}``.  The shell half is exercised for real
against a fixture cache tree: the defect lived entirely in shell string
semantics, which no amount of Python mocking would have caught.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from sparkrun.models.distribute import _build_model_ensure_script
from sparkrun.models.download import model_cache_path
from sparkrun.scripts import read_script

needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="requires a POSIX shell",
)

# The reporter's model: an ``owner/model`` id whose name also contains hyphens
# and a dot, so a fix that merely special-cased the separator would still be
# visibly wrong here.
_MODEL = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
_EXPECTED_DIR = "models--RadixArk--Qwen3.8-Flash-Next-NVFP4"


def _render(model_id: str, cache: str = "/mnt/huggingface") -> str:
    return _build_model_ensure_script(model_id, cache)


def _stub_bin(tmp_path):
    """A PATH whose ``huggingface-cli`` downloads nothing.

    The miss path in these scripts really does reach for the network — and
    ``curl | sh``-installs ``uv`` when no client is found — so the stub is what
    keeps the suite hermetic while still letting us observe *that* the miss
    path was taken.
    """
    bindir = tmp_path / "stub-bin"
    bindir.mkdir()
    stub = bindir / "huggingface-cli"
    stub.write_text('#!/bin/bash\necho "STUB-DOWNLOAD $*"\n')
    stub.chmod(0o755)
    return bindir


def _run(script: str, bindir=None) -> subprocess.CompletedProcess:
    env = {"HOME": "/nonexistent", "PATH": "/usr/bin:/bin"}
    if bindir is not None:
        env["PATH"] = f"{bindir}:{env['PATH']}"
    return subprocess.run(
        ["bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


# ---------------------------------------------------------------------------
# Rendering: owner/model resolves to models--owner--model
# ---------------------------------------------------------------------------


class TestEnsureScriptCachePath:
    """The rendered script must probe the real HF cache directory."""

    def test_owner_model_uses_double_hyphen(self):
        script = _render(_MODEL)

        assert f'CACHE_PATH="/mnt/huggingface/hub/{_EXPECTED_DIR}"' in script
        # The single-hyphen form is what the tr-based derivation produced.
        assert "models--RadixArk-Qwen3.8" not in script

    def test_gguf_spec_strips_quant_and_uses_double_hyphen(self):
        script = _render("Qwen/Qwen3-1.7B-GGUF:Q4_K_M")

        assert 'CACHE_PATH="/mnt/huggingface/hub/models--Qwen--Qwen3-1.7B-GGUF"' in script
        # The quant selects files inside the snapshot, never the directory.
        assert "models--Qwen--Qwen3-1.7B-GGUF:Q4_K_M" not in script

    def test_matches_model_cache_path(self):
        """The check and the rsync destination are the same string.

        They are built from one function now; asserting it here is what keeps
        a future 'quick fix' in the shell from re-forking the rule.
        """
        for model_id in (_MODEL, "meta-llama/Llama-3-8B", "Qwen/Qwen3-1.7B-GGUF:Q4_K_M", "gpt2"):
            script = _build_model_ensure_script(model_id, "/hf")
            assert f'CACHE_PATH="{model_cache_path(model_id, "/hf")}"' in script

    def test_bare_model_id_unchanged(self):
        """A slash-free id was the one case the old derivation got right."""
        assert 'CACHE_PATH="/hf/hub/models--gpt2"' in _build_model_ensure_script("gpt2", "/hf")

    def test_no_bash_side_mangling_remains(self):
        """The scripts must not re-derive the name — that is the whole defect."""
        for name in ("model_sync.sh", "model_sync_gguf.sh"):
            raw = read_script(name)
            assert "SAFE_NAME" not in raw
            assert "tr '/'" not in raw


# ---------------------------------------------------------------------------
# Behaviour under bash: cached weights skip the download path
# ---------------------------------------------------------------------------


def _make_cache(tmp_path, dirname: str, *filenames: str):
    """Materialize ``<tmp>/hub/<dirname>/snapshots/abc123/<files>``."""
    snapshot = tmp_path / "hub" / dirname / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    for filename in filenames:
        (snapshot / filename).write_text("weights")
    return snapshot


@needs_bash
class TestEnsureScriptUnderBash:
    """Run the rendered script for real against a fixture cache tree.

    A cache hit must ``exit 0`` before reaching the downloader.  These scripts
    curl-install ``uv`` when no HF client is found, so a missed hit is not
    merely wasted work on a host without one — it fails the launch.
    """

    def test_cache_hit_skips_download(self, tmp_path):
        _make_cache(tmp_path, _EXPECTED_DIR, "model-00001-of-00002.safetensors")

        proc = _run(_render(_MODEL, cache=str(tmp_path)), _stub_bin(tmp_path))

        assert proc.returncode == 0
        assert "already cached" in proc.stdout
        assert "STUB-DOWNLOAD" not in proc.stdout

    def test_single_hyphen_directory_is_not_a_hit(self, tmp_path):
        """Guards against a fix that swaps one wrong mangling for another."""
        _make_cache(tmp_path, "models--RadixArk-Qwen3.8-Flash-Next-NVFP4", "model.safetensors")

        proc = _run(_render(_MODEL, cache=str(tmp_path)), _stub_bin(tmp_path))

        assert "already cached" not in proc.stdout
        assert "STUB-DOWNLOAD" in proc.stdout

    def test_config_only_snapshot_is_not_a_hit(self, tmp_path):
        """A VRAM auto-detect fetch leaves config.json and no weights."""
        _make_cache(tmp_path, _EXPECTED_DIR, "config.json")

        proc = _run(_render(_MODEL, cache=str(tmp_path)), _stub_bin(tmp_path))

        assert "already cached" not in proc.stdout
        assert "STUB-DOWNLOAD" in proc.stdout

    def test_missing_cache_enters_download_path(self, tmp_path):
        proc = _run(_render(_MODEL, cache=str(tmp_path)), _stub_bin(tmp_path))

        assert f"STUB-DOWNLOAD download {_MODEL}" in proc.stdout

    def test_gguf_cache_hit_skips_download(self, tmp_path):
        _make_cache(tmp_path, "models--Qwen--Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf")

        proc = _run(_render("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", cache=str(tmp_path)), _stub_bin(tmp_path))

        assert proc.returncode == 0
        assert "already cached" in proc.stdout
        assert "STUB-DOWNLOAD" not in proc.stdout

    def test_gguf_other_quant_is_not_a_hit(self, tmp_path):
        """The directory is keyed by repo; the quant still has to match a file."""
        _make_cache(tmp_path, "models--Qwen--Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf")

        proc = _run(_render("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", cache=str(tmp_path)), _stub_bin(tmp_path))

        assert "already cached" not in proc.stdout
        assert "STUB-DOWNLOAD" in proc.stdout
