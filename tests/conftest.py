"""Shared pytest fixtures for sparkrun tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from _telemetry_guard import describe_escapes, install_telemetry_blocker
from sparkrun.core.bootstrap import init_sparkrun
from sparkrun.core.registry import RegistryManager

#: Captured before ``isolate_stateful`` stubs it out, so the ``real_registry_git``
#: opt-out fixture can hand the genuine implementation back.
_REAL_CLONE_OR_PULL = RegistryManager._clone_or_pull


@pytest.fixture(autouse=True)
def isolate_stateful(tmp_path: Path, monkeypatch):
    """Redirect SAF stateful root to temp dir for test isolation.

    Prevents tests from writing to the real ~/.config/sparkrun/.
    Also resets the bootstrap singleton between tests.
    """
    monkeypatch.setenv("STATEFUL_ROOT", str(tmp_path / "stateful"))
    monkeypatch.setenv("SPARKRUN_NO_TELEMETRY", "1")
    # The HuggingFace Hub metadata budget, its breaker and its negative memo are
    # process-global (one budget per command, by design). Without a reset a test
    # that exhausts the budget or memoises an unavailable repo silently disables
    # Hub lookups for every test that runs after it -- an ordering-dependent
    # failure, which is the expensive kind.
    from sparkrun.models.hub import reset_hub_state

    reset_hub_state()
    # ...and block the send itself, because the env var above is only policy.
    # Any test can drop it (test_telemetry.py does, on purpose), and telemetry
    # fails *open*: a MagicMock config makes `telemetry_enabled` return True,
    # which is how mock objects reached the production collector as a
    # benchmark's category/framework/profile. Checked at teardown below.
    telemetry_attempts = install_telemetry_blocker(monkeypatch)
    # Hard-disable external-plugin auto-loading during tests. The feature flag
    # alone is not enough: pytest reads the developer's REAL ~/.config/sparkrun
    # (the SAF stateful root isn't "ready"), so a developer who enabled
    # core.external_plugins would otherwise load their real plugins mid-suite.
    # Loader tests pass explicit paths (which bypass this) or delenv it.
    monkeypatch.setenv("SPARKRUN_NO_EXTERNAL_PLUGINS", "1")
    # The experimental local/k8s executors gate themselves off by default on
    # the stable channel (via is_multi_extension). Most of the suite exercises
    # their behavior directly and predates the feature-flag gating, so enable
    # them here to preserve that contract. tests/test_features.py exercises the
    # gating itself in clean subprocesses that strip these env overrides.
    monkeypatch.setenv("SPARKRUN_FEATURE_EXECUTOR_LOCAL", "1")
    monkeypatch.setenv("SPARKRUN_FEATURE_EXECUTOR_K8S", "1")
    # The `setup k8s` command group is likewise gated off by default; enable it
    # so the CLI tests that exercise it keep passing (the gate itself is tested
    # explicitly in test_k8s_setup with the env override cleared).
    monkeypatch.setenv("SPARKRUN_FEATURE_CLI_SETUP_K8S", "1")
    # Same for the uv-venv builder (off on stable, on for beta/alpha): SAF
    # decides is_multi_extension once at registration, so a test cannot un-hide
    # a builder the process already registered as gated. Enable it here and
    # exercise the gate itself in the clean subprocesses of test_uv_venv.py.
    monkeypatch.setenv("SPARKRUN_FEATURE_BUILDER_UV_VENV", "1")
    # Point the user config dir at the sandbox too. STATEFUL_ROOT alone does
    # not cover it: DEFAULT_CONFIG_DIR is computed from Path.home() at import
    # time, so without this a test silently reads the developer's real
    # ~/.config/sparkrun -- its default cluster, its saved clusters, its
    # registries. Such a test passes on the machine that wrote that config and
    # fails everywhere else, which is the most expensive way to find out.
    import sparkrun.core.config as _config_module

    monkeypatch.setattr(_config_module, "DEFAULT_CONFIG_DIR", tmp_path / "config", raising=False)

    # Same treatment for the cache dir, and for the same reason -- but the
    # stakes are higher here because the cache holds *live* state, not just
    # preferences. Unpatched, ProxyEngine defaults its state_dir to
    # DEFAULT_CACHE_DIR/"proxy", so `api.proxy.status()` in a test reads the
    # developer's really-running proxy and `api.proxy.stop()` SIGTERMs it.
    # Job metadata, pending-op lock files and cached remote recipes land in the
    # real cache the same way.
    # Keep the trailing "sparkrun" segment: the real cache is ~/.cache/sparkrun
    # and callers derive subpaths from it, so tests that assert on the shape of
    # a derived path (".../sparkrun/tuning/vllm") stay meaningful.
    _cache = tmp_path / "cache" / "sparkrun"
    monkeypatch.setattr(_config_module, "DEFAULT_CACHE_DIR", _cache, raising=False)
    # Two modules bind the symbol at import time, so patching core.config alone
    # does not reach them. Keep this list in sync with:
    #   grep -rn '^from sparkrun.core.config import.*DEFAULT_CACHE_DIR' src/
    import sparkrun.core.pending_ops as _pending_ops
    import sparkrun.tuning._common as _tuning_common

    monkeypatch.setattr(_pending_ops, "DEFAULT_CACHE_DIR", _cache, raising=False)
    monkeypatch.setattr(_tuning_common, "DEFAULT_CACHE_DIR", _cache, raising=False)

    # No network from the test suite. The sandboxed config dir never has a
    # registries.yaml, so every RegistryManager falls into first-run bootstrap
    # (_default_registries -> _init_defaults_from_manifests) and git-clones each
    # BOOTSTRAP_REGISTRY_URLS entry. That was previously masked by the cache dir
    # leaking out to the developer's real, already-populated
    # ~/.cache/sparkrun/registries; with the cache sandboxed it becomes a full
    # clone of three GitHub repos, and a single CLI test went from 1.4s to 6.6s.
    # Emptying the list takes the documented offline path: discovery yields
    # nothing and FALLBACK_DEFAULT_REGISTRIES supplies the defaults.
    # Tests that exercise discovery set their own URLs or mock subprocess.run.
    import sparkrun.core.registry as _registry_module

    monkeypatch.setattr(_registry_module, "BOOTSTRAP_REGISTRY_URLS", [], raising=False)

    # ...and no `git clone` / `git fetch` either. `_clone_or_pull` is the single
    # choke point for every registry git operation (registry.py + core/mods.py);
    # it is documented best-effort and already returns False on failure, so
    # stubbing it takes a path callers handle. Profiling one CLI test showed
    # 5.3s of its 5.9s inside `_sync_url` / `_clone_or_pull_single` -- real
    # network round-trips, in a suite that is supposed to be hermetic.
    monkeypatch.setattr(_registry_module.RegistryManager, "_clone_or_pull", lambda self, entry: False, raising=False)

    import sparkrun.core.bootstrap

    sparkrun.core.bootstrap._variables = None
    yield
    sparkrun.core.bootstrap._variables = None

    if telemetry_attempts:
        pytest.fail(describe_escapes(telemetry_attempts), pytrace=False)


@pytest.fixture
def real_registry_git(isolate_stateful, monkeypatch):
    """Restore ``RegistryManager._clone_or_pull`` for tests that mock git themselves.

    ``isolate_stateful`` stubs the method so no test can reach the network by
    accident. Tests that assert on the git argv (``update`` calls clone/pull,
    ``--`` before the URL) patch ``subprocess.run`` in their own body, so the
    real implementation is hermetic for them — they just need it back.

    Depends on ``isolate_stateful`` so the ordering is explicit: the stub is
    installed first, this undoes it.
    """
    import sparkrun.core.registry as registry_module

    monkeypatch.setattr(registry_module.RegistryManager, "_clone_or_pull", _REAL_CLONE_OR_PULL)


@pytest.fixture
def cluster_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for cluster definitions."""
    d = tmp_path / "clusters"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def hosts_file(tmp_path: Path) -> Path:
    """Create a temporary hosts file with sample hosts."""
    f = tmp_path / "hosts.txt"
    f.write_text("10.0.0.1\n10.0.0.2\n10.0.0.3\n")
    return f


@pytest.fixture
def tmp_recipe_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample YAML recipe files.

    Creates both v1 (eugr-style) and v2 format recipes for testing.

    Returns:
        Path to temporary directory containing recipe files.
    """
    recipe_dir = tmp_path / "recipes"
    recipe_dir.mkdir()

    # v2 vllm recipe. No ``name:`` — it is the v1 spelling, ignored on load
    # (the name comes from the filename), and reported as a deprecation for v2.
    v2_vllm = {
        "sparkrun_version": "2",
        "description": "A test recipe for vLLM",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "mode": "auto",
        "container": "scitrera/dgx-spark-vllm:latest",
        "defaults": {
            "port": 8000,
            "host": "0.0.0.0",
            "tensor_parallel": 1,
            "gpu_memory_utilization": 0.9,
        },
        "env": {
            "VLLM_BATCH_INVARIANT": "1",
        },
        "command": "vllm serve {model} --port {port} --host {host}",
    }
    with open(recipe_dir / "test-vllm.yaml", "w") as f:
        yaml.dump(v2_vllm, f)

    # v2 sglang recipe
    v2_sglang = {
        "sparkrun_version": "2",
        "description": "A test recipe for SGLang",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
        "mode": "cluster",
        "min_nodes": 2,
        "container": "scitrera/dgx-spark-sglang:latest",
        "defaults": {
            "port": 30000,
            "host": "0.0.0.0",
            "tensor_parallel": 2,
        },
    }
    with open(recipe_dir / "test-sglang.yaml", "w") as f:
        yaml.dump(v2_sglang, f)

    # v1 recipe with mods (should auto-set eugr builder).
    # v1 default values are strings (template-substituted via {port}); the
    # migration's `.replace("{{", "{")` step assumes str values.
    v1_eugr = {
        "recipe_version": "1",
        "name": "Test EUGR Recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "build_args": ["ARG1=value1"],
        "mods": ["mod1.patch"],
        "defaults": {
            "port": "8000",
        },
    }
    with open(recipe_dir / "test-eugr.yaml", "w") as f:
        yaml.dump(v1_eugr, f)

    # v1 recipe without mods (should auto-set eugr builder, resolve to vllm-distributed)
    v1_plain = {
        "recipe_version": "1",
        "name": "Test Plain v1 Recipe",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "defaults": {
            "port": "8000",
        },
    }
    with open(recipe_dir / "test-plain-v1.yaml", "w") as f:
        yaml.dump(v1_plain, f)

    return recipe_dir


@pytest.fixture
def v(tmp_path: Path) -> Any:
    """Initialize sparkrun and return the Variables instance.

    Uses WARNING log level to reduce test output noise.
    Resets the global _variables singleton for test isolation.

    Returns:
        Initialized Variables instance.
    """
    # Reset global singleton to ensure test isolation
    import sparkrun.core.bootstrap

    sparkrun.core.bootstrap._variables = None

    return init_sparkrun(log_level="WARNING")


@pytest.fixture
def sample_v2_recipe_data() -> dict[str, Any]:
    """Return a dict for a v2 vllm recipe.

    Returns:
        Dictionary containing a valid v2 recipe.
    """
    return {
        "sparkrun_version": "2",
        "description": "A sample vLLM recipe for testing",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "mode": "auto",
        "min_nodes": 1,
        "max_nodes": 4,
        "container": "scitrera/dgx-spark-vllm:0.16.0",
        "defaults": {
            "port": 8000,
            "host": "0.0.0.0",
            "tensor_parallel": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "VLLM_BATCH_INVARIANT": "1",
        },
        "command": "vllm serve {model} --port {port} -tp {tensor_parallel}",
    }


@pytest.fixture
def sample_v1_recipe_data() -> dict[str, Any]:
    """Return a dict for a v1 eugr-style recipe with mods and build_args.

    Returns:
        Dictionary containing a valid v1 recipe that should auto-set eugr builder.
    """
    return {
        "recipe_version": "1",
        "name": "Sample EUGR Recipe",
        "description": "A v1 recipe with custom build",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "vllm",
        "cluster_only": True,
        "build_args": [
            "VLLM_VERSION=0.5.0",
            "CUSTOM_FLAG=true",
        ],
        "mods": [
            "custom_attention.patch",
            "performance_tweaks.patch",
        ],
        "defaults": {
            # v1 defaults are template-substituted as strings ({port} → "8000")
            "port": "8000",
            "tensor_parallel": "2",
        },
        "env": {
            "NCCL_DEBUG": "INFO",
        },
        "command": "python -m vllm.entrypoints.openai.api_server --model {model}",
    }


@pytest.fixture
def sample_sglang_recipe_data() -> dict[str, Any]:
    """Return a dict for a v2 sglang recipe.

    Returns:
        Dictionary containing a valid v2 SGLang recipe.
    """
    return {
        "sparkrun_version": "2",
        "description": "A sample SGLang recipe for testing",
        "model": "meta-llama/Llama-2-7b-hf",
        "runtime": "sglang",
        "mode": "cluster",
        "min_nodes": 2,
        "max_nodes": 8,
        "container": "scitrera/dgx-spark-sglang:0.5.8",
        "defaults": {
            "port": 30000,
            "host": "0.0.0.0",
            "tensor_parallel": 2,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 32768,
        },
        "env": {
            "NCCL_CUMEM_ENABLE": "0",
        },
        "command": "python3 -m sglang.launch_server --model-path {model} --port {port}",
    }


@pytest.fixture
def log_sources_spy(monkeypatch):
    """Capture what ``RuntimePlugin.follow_logs`` hands to the log reader.

    ``follow_logs`` is a printing shim over
    :meth:`~sparkrun.runtimes.base.RuntimePlugin.log_sources` +
    :func:`~sparkrun.orchestration.logs.print_log_sources`, so intercepting
    the latter records the resolved sources — host, container name, and
    file-vs-stdout mode.  That is the observable behaviour the older
    ``stream_container_file_logs`` / ``stream_remote_logs`` mocks stood in
    for, asserted directly instead of via which helper got called (and
    without spawning a reader subprocess).

    Yields a list of captured calls, each with ``.sources`` plus the
    ``follow`` / ``tail`` / ``dry_run`` / ``ssh_kwargs`` keywords.
    """
    from types import SimpleNamespace

    calls: list[Any] = []

    def _capture(executor, sources, **kwargs):
        calls.append(SimpleNamespace(executor=executor, sources=list(sources), **kwargs))

    monkeypatch.setattr("sparkrun.orchestration.logs.print_log_sources", _capture)
    return calls


@pytest.fixture
def deepseek_v4_config() -> dict[str, Any]:
    """Return an abridged DeepSeek-V4-Flash-0731 ``config.json``.

    Carries the fields the VRAM estimator reads for a Multi-head Latent
    Attention model: ``head_dim`` holds the compressed latent (V2/V3 name it
    ``kv_lora_rank`` instead), and ``compress_ratios`` gives 21 layers at ratio
    4 and 20 at ratio 128.  The ratio-0 layers are sliding-window and hold no
    latent cache.

    Returns:
        Dictionary of HuggingFace config fields.
    """
    return {
        "model_type": "deepseek_v4",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "hidden_size": 4096,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "sliding_window": 128,
        "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0, 0, 0],
        # Sparse attention: a second cache the estimator does not size.
        "index_head_dim": 128,
        "index_topk": 512,
    }


@pytest.fixture
def deepseek_v3_config() -> dict[str, Any]:
    """Return an abridged DeepSeek-V3 ``config.json``.

    The other MLA shape: V2/V3 name the compressed latent ``kv_lora_rank``
    explicitly and cache ``qk_rope_head_dim`` *in addition* to it, so the KV
    width is ``512 + 64`` elements.  There is no top-level ``head_dim`` — it is
    derived as ``hidden_size // num_attention_heads`` — and no
    ``compress_ratios``.

    Returns:
        Dictionary of HuggingFace config fields.
    """
    return {
        "model_type": "deepseek_v3",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "hidden_size": 7168,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
    }


@pytest.fixture
def deepseek_v32_config() -> dict[str, Any]:
    """Return an abridged DeepSeek-V3.2-Exp ``config.json``.

    The shape that exposed the auxiliary-cache warning gap: sparse attention
    (``index_head_dim``) with **no** ``compress_ratios``, so a warning keyed on
    per-layer compression alone would never fire for it — even though its
    indexer cache is a full KV peer worth roughly 132 B per token per layer.

    Returns:
        Dictionary of HuggingFace config fields.
    """
    return {
        "model_type": "deepseek_v32",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "hidden_size": 7168,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "index_n_heads": 64,
        "index_topk": 2048,
    }
