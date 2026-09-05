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
into the script as ``{cache_path}``.

The same check was also **revision-blind**: ``{revision_flag}`` reached only the
downloader below it, while the check scanned every snapshot.  A host holding a
different revision therefore reported a hit, the pinned revision was never
fetched, and the workload served weights the recipe had explicitly pinned
against — silently.  Snapshot resolution reads the *target's* filesystem, so
unlike the mangling it cannot move control-side; it lives in one shared helper,
``_hf_snapshots.sh``, whose contract mirrors ``is_model_cached``.

The shell half is exercised for real against a fixture cache tree: both defects
lived entirely in shell semantics, which no amount of Python mocking would have
caught.
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


def _render(model_id: str, cache: str = "/mnt/huggingface", revision: str | None = None) -> str:
    return _build_model_ensure_script(model_id, cache, revision=revision)


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


def _make_cache(tmp_path, dirname: str, *filenames: str, commit: str = "abc123", ref: str | None = None):
    """Materialize ``<tmp>/hub/<dirname>/snapshots/<commit>/<files>``.

    *ref* additionally writes ``refs/<ref>`` pointing at *commit*, which is how
    a real HF cache records a branch or tag.
    """
    model_cache = tmp_path / "hub" / dirname
    snapshot = model_cache / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        (snapshot / filename).write_text("weights")
    if ref is not None:
        refs = model_cache / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / ref).write_text(commit + "\n")
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


# ---------------------------------------------------------------------------
# Revision pinning
# ---------------------------------------------------------------------------


@needs_bash
class TestRevisionPinning:
    """A pinned revision must gate the cache check, not just the download.

    ``{revision_flag}`` used to reach only the downloader while the check
    scanned every snapshot, so a host holding a *different* revision reported
    a hit and the pin was silently ignored — the workload then served weights
    the recipe had explicitly pinned against.
    """

    def test_other_revision_cached_is_not_a_hit(self, tmp_path):
        _make_cache(tmp_path, _EXPECTED_DIR, "model.safetensors", commit="oldsha", ref="main")

        proc = _run(_render(_MODEL, cache=str(tmp_path), revision="newsha"), _stub_bin(tmp_path))

        assert "already cached" not in proc.stdout
        assert "STUB-DOWNLOAD" in proc.stdout
        assert "--revision newsha" in proc.stdout

    def test_matching_ref_is_a_hit(self, tmp_path):
        _make_cache(tmp_path, _EXPECTED_DIR, "model.safetensors", commit="deadbeef", ref="v2.0")

        proc = _run(_render(_MODEL, cache=str(tmp_path), revision="v2.0"), _stub_bin(tmp_path))

        assert proc.returncode == 0
        assert "already cached" in proc.stdout
        assert "STUB-DOWNLOAD" not in proc.stdout

    def test_revision_as_commit_hash_is_a_hit(self, tmp_path):
        """A revision may name the snapshot directory directly, with no ref."""
        _make_cache(tmp_path, _EXPECTED_DIR, "model.safetensors", commit="deadbeef")

        proc = _run(_render(_MODEL, cache=str(tmp_path), revision="deadbeef"), _stub_bin(tmp_path))

        assert "already cached" in proc.stdout

    def test_pinned_ref_without_weights_is_not_a_hit(self, tmp_path):
        """No fallback: another snapshot's weights are not the pinned ones."""
        _make_cache(tmp_path, _EXPECTED_DIR, "config.json", commit="wanted", ref="v2.0")
        _make_cache(tmp_path, _EXPECTED_DIR, "model.safetensors", commit="other")

        proc = _run(_render(_MODEL, cache=str(tmp_path), revision="v2.0"), _stub_bin(tmp_path))

        assert "already cached" not in proc.stdout
        assert "STUB-DOWNLOAD" in proc.stdout

    def test_unpinned_prefers_refs_main(self, tmp_path):
        _make_cache(tmp_path, _EXPECTED_DIR, "model.safetensors", commit="mainsha", ref="main")

        proc = _run(_render(_MODEL, cache=str(tmp_path)), _stub_bin(tmp_path))

        assert "already cached" in proc.stdout
        assert "--revision" not in proc.stdout

    def test_unpinned_falls_back_to_any_snapshot(self, tmp_path):
        """A hand-placed cache has no refs/ at all and must still count."""
        _make_cache(tmp_path, _EXPECTED_DIR, "model.safetensors", commit="whatever")

        proc = _run(_render(_MODEL, cache=str(tmp_path)), _stub_bin(tmp_path))

        assert "already cached" in proc.stdout

    def test_gguf_honors_revision(self, tmp_path):
        _make_cache(
            tmp_path,
            "models--Qwen--Qwen3-1.7B-GGUF",
            "Qwen3-1.7B-Q4_K_M.gguf",
            commit="oldsha",
            ref="main",
        )

        proc = _run(
            _render("Qwen/Qwen3-1.7B-GGUF:Q4_K_M", cache=str(tmp_path), revision="newsha"),
            _stub_bin(tmp_path),
        )

        assert "already cached" not in proc.stdout
        assert "--revision newsha" in proc.stdout

    def test_revision_is_not_shell_injected(self, tmp_path):
        """`revision` is recipe content and reaches a script run on every host.

        Exercised on a cache **miss** deliberately: the value used to be
        interpolated into the download command as bare text, which a hit would
        short-circuit past.  It now travels in positional parameters.
        """
        evil = "$(touch " + str(tmp_path / "pwned") + ")"

        proc = _run(_render(_MODEL, cache=str(tmp_path), revision=evil), _stub_bin(tmp_path))

        assert not (tmp_path / "pwned").exists()
        # Reached the downloader, and handed it the value verbatim.
        assert "STUB-DOWNLOAD" in proc.stdout
        assert f"--revision {evil}" in proc.stdout


# ---------------------------------------------------------------------------
# Parity with the control-side peer
# ---------------------------------------------------------------------------


@needs_bash
class TestParityWithIsModelCached:
    """The remote check must agree with ``is_model_cached``.

    They are two implementations of one question — the drift between them is
    the whole subject of this file — so the agreement is asserted directly
    rather than left to matching prose in two docstrings.
    """

    # (label, files, commit, ref, queried revision)
    CASES = [
        ("unpinned hit via refs/main", ("model.safetensors",), "sha1", "main", None),
        ("unpinned hit, no refs at all", ("model.safetensors",), "sha1", None, None),
        ("unpinned miss, config only", ("config.json",), "sha1", "main", None),
        ("pinned hit via ref", ("model.safetensors",), "sha1", "v2", "v2"),
        ("pinned hit via commit hash", ("model.safetensors",), "sha1", None, "sha1"),
        ("pinned miss, other revision", ("model.safetensors",), "sha1", "main", "v9"),
        ("pinned miss, ref has no weights", ("config.json",), "sha1", "v2", "v2"),
        ("hit on .bin weights", ("pytorch_model.bin",), "sha1", "main", None),
    ]

    @pytest.mark.parametrize("label,files,commit,ref,revision", CASES, ids=[c[0] for c in CASES])
    def test_agrees(self, tmp_path, label, files, commit, ref, revision):
        from sparkrun.models.download import is_model_cached

        _make_cache(tmp_path, _EXPECTED_DIR, *files, commit=commit, ref=ref)

        proc = _run(_render(_MODEL, cache=str(tmp_path), revision=revision), _stub_bin(tmp_path))
        shell_hit = "already cached" in proc.stdout
        python_hit = is_model_cached(_MODEL, cache_dir=str(tmp_path), revision=revision)

        assert shell_hit == python_hit, f"{label}: shell={shell_hit} python={python_hit}"

    def test_known_divergence_is_deliberate(self, tmp_path):
        """Sharded-into-a-subdirectory repos: the shell scan stays recursive.

        ``is_model_cached`` globs the snapshot's top level only.  Narrowing the
        remote check to match would re-download those repos on every launch,
        which is the failure this file exists to prevent — so the shell is
        deliberately the more permissive of the two.
        """
        from sparkrun.models.download import is_model_cached

        snapshot = _make_cache(tmp_path, _EXPECTED_DIR, commit="sha1", ref="main")
        nested = snapshot / "shards"
        nested.mkdir()
        (nested / "model-00001.safetensors").write_text("weights")

        proc = _run(_render(_MODEL, cache=str(tmp_path)), _stub_bin(tmp_path))

        assert "already cached" in proc.stdout
        assert is_model_cached(_MODEL, cache_dir=str(tmp_path)) is False
