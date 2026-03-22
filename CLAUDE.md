# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sparkrun** is a CLI tool for launching, managing, and stopping Docker-based LLM inference workloads on NVIDIA DGX
Spark systems. It orchestrates containers over SSH — no Slurm or Kubernetes required. The control machine doesn't need
to be a cluster member; it coordinates DGX Sparks remotely.

Each DGX Spark has one GPU with 128 GB unified memory, so tensor parallelism maps directly to node count (`--tp 2` = 2
hosts).

## Common Commands

```bash
# Install in development mode (editable)
uv pip install -e ".[dev]"

# Run full test suite
.venv/bin/python -m pytest tests/ -v

# Run a single test file
.venv/bin/python -m pytest tests/test_recipe.py -v

# Run a specific test
.venv/bin/python -m pytest tests/test_cli.py::test_run_command_basic -v

# Run with coverage
.venv/bin/python -m pytest tests/ --cov=sparkrun --cov-report=term-missing

# Lint (ruff, line-length 140, target py312)
ruff check src/ tests/
ruff format src/ tests/

# Run the CLI directly during development
.venv/bin/sparkrun --help
.venv/bin/sparkrun run --dry-run qwen3-1.7b-vllm

# Sync versions across packages (pyproject.toml + sparkrun-cc-plugin)
python scripts/update-versions.py
python scripts/update-versions.py --check   # CI-friendly verify
```

Versions are tracked in `versions.yaml` at the repo root and synced to all package files via
`scripts/update-versions.py`.

## Architecture

### Source Layout

```
src/sparkrun/
├── cli/                # Click CLI package (see CLI Architecture below)
├── core/               # Core data models, bootstrap, and business logic (see below)
├── runtimes/           # Runtime plugins (see below)
├── orchestration/      # SSH, Docker, InfiniBand, script execution primitives
├── models/             # HuggingFace model download, distribution, and VRAM estimation
├── containers/         # Container image distribution (docker save/load over SSH)
├── tuning/             # Triton fused MoE kernel tuning for SGLang and vLLM
├── builders/           # Container image builder plugins (docker-pull, eugr)
├── diagnostics/        # Host and run diagnostic collection (NDJSON output)
├── proxy/              # LiteLLM-based inference proxy gateway
├── benchmarking/       # Benchmark framework plugins and result export (llama-benchy)
├── utils/              # Shared helpers (coerce_value, suppress_noisy_loggers, etc.)
└── scripts/            # Embedded bash scripts (IB detection, container launch, etc.)
```

### Core Data Models (`core/`)

Core domain logic extracted from the top-level package. All imports use `sparkrun.core.*` (e.g.,
`from sparkrun.core.config import SparkrunConfig`).

| Module                  | Purpose                                                                              |
|-------------------------|--------------------------------------------------------------------------------------|
| `bootstrap.py`          | SAF plugin initialization, runtime and benchmarking framework discovery              |
| `config.py`             | `SparkrunConfig` — reads `~/.config/sparkrun/config.yaml`, cache dir resolution      |
| `registry.py`           | `RegistryManager` — git-based recipe registry system (see Registry System below)     |
| `recipe.py`             | `Recipe` loading, validation, v1→v2 migration, config chain via SAF Variables         |
| `cluster_manager.py`    | `ClusterManager` — named cluster CRUD (YAML files in `~/.config/sparkrun/clusters/`) |
| `hosts.py`              | Host resolution priority chain (CLI → file → cluster → default)                      |
| `pending_ops.py`        | PID-based lock files for in-progress operations                                      |
| `benchmark_profiles.py` | Benchmark profile discovery, resolution, and rendering across registries             |

### CLI Architecture (`cli/`)

The CLI was split from a single `cli.py` into a package for maintainability. The `__init__.py` defines the top-level
`main` Click group, registers all subcommands, and provides top-level aliases (`list`, `show`, `search`, `status`).

| Module            | Purpose                                                                                                                                                                                                                                                           |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `__init__.py`     | `main` Click group, command registration, top-level aliases                                                                                                                                                                                                       |
| `_common.py`      | Shared infrastructure: logging setup, Click parameter types (`RECIPE_NAME`, `REGISTRY_NAME`, `RUNTIME_NAME`, `CLUSTER_NAME`, `PROFILE_NAME`), decorators (`host_options`, `dry_run_option`), and reusable helpers (host resolution, recipe loading, VRAM display) |
| `_run.py`         | `run` command — launch inference workloads                                                                                                                                                                                                                        |
| `_stop_logs.py`   | `stop` and `logs` commands — stop workloads and stream container logs                                                                                                                                                                                             |
| `_setup.py`       | `setup` command group — shell completion, SSH mesh, model/container sync, permissions, cache, networking                                                                                                                                                          |
| `_cluster.py`     | `cluster` command group — create/list/show/delete/update saved cluster definitions, cluster status                                                                                                                                                                |
| `_recipe.py`      | `recipe` command group — list/show/search recipes across registries                                                                                                                                                                                               |
| `_registry.py`    | `registry` command group — add/remove/enable/disable/update registries, list/show benchmark profiles                                                                                                                                                              |
| `_benchmark.py`   | `benchmark` command group — run benchmark profiles against inference workloads                                                                                                                                                                                    |
| `_tune.py`        | `tune` command group — run Triton fused MoE kernel tuning (SGLang and vLLM)                                                                                                                                                                                       |
| `_wizard.py`      | `setup wizard` command — guided cluster setup                                                                                                                                                                                                                     |
| `_proxy.py`       | `proxy` command group — LiteLLM inference proxy management                                                                                                                                                                                                        |
| `_monitor_tui.py` | Textual TUI for `cluster monitor`                                                                                                                                                                                                                                 |

### Plugin System (SAF)

sparkrun uses [scitrera-app-framework](https://github.com/scitrera/python-app-framework) (SAF) for plugin discovery and
lifecycle. Runtimes and benchmarking frameworks register as multi-extension plugins. Discovery happens via Python entry
points defined in `pyproject.toml`.

Key bootstrap flow: `cli/__init__.py` → `core.bootstrap.init_sparkrun()` → SAF `init_framework_desktop()` →
`find_types_in_modules("sparkrun.runtimes", RuntimePlugin)` +
`find_types_in_modules("sparkrun.benchmarking", BenchmarkingPlugin)` → `register_plugin()` for each discovered plugin.

### Runtime Architecture

All runtimes extend `RuntimePlugin` (in `runtimes/base.py`), which itself extends SAF's `Plugin` class. The base class
provides solo-mode orchestration; runtimes override `run()`/`stop()`/`follow_logs()` for multi-node support.

| Runtime              | File                           | Entry Point              | Clustering         | Strategy                                                                                    |
|----------------------|--------------------------------|--------------------------|--------------------|---------------------------------------------------------------------------------------------|
| **vllm-ray**         | `runtimes/vllm_ray.py`         | `VllmRayRuntime`         | Ray head/worker    | `"ray"` — starts Ray cluster, exec serve on head                                            |
| **vllm-distributed** | `runtimes/vllm_distributed.py` | `VllmDistributedRuntime` | Native distributed | `"native"` — each node runs serve independently (no Ray)                                    |
| **sglang**           | `runtimes/sglang.py`           | `SglangRuntime`          | Native distributed | `"native"` — each node runs serve with `--node-rank`                                        |
| **llama-cpp**        | `runtimes/llama_cpp.py`        | `LlamaCppRuntime`        | Experimental RPC   | `"native/rpc"` — workers run `rpc-server`, head connects via `--rpc`                        |
| **trtllm**           | `runtimes/trtllm.py`           | `TrtllmRuntime`          | MPI (native)       | `"native"` — sleep infinity containers + mpirun on head                                     |
| **eugr-vllm**        | `runtimes/eugr_vllm_ray.py`    | `EugrVllmRayRuntime`     | Ray (inherited)    | Extends VllmRayRuntime with eugr container builds and mods (v1 recipe support) (deprecated) |

Runtimes must implement `generate_command()` and `resolve_container()`. The `cluster_strategy()` return value determines
which orchestration path the base class uses.

### Orchestration Layer (`orchestration/`)

All remote operations use **SSH stdin piping** — scripts are generated as Python strings and piped to `ssh <host> bash -s`. No files are ever copied to remote hosts.

- **`ssh.py`** — `RemoteResult` dataclass, `build_ssh_cmd()`, `run_remote_script()`, `run_remote_scripts_parallel()`, `run_rsync_parallel()`, `stream_remote_logs()`
- **`sudo.py`** — `run_with_sudo_fallback()` — tries non-interactive sudo in parallel, then falls back to password-based sudo for failures
- **`docker.py`** — Pure command-string generators (`docker_run_cmd`, `docker_exec_cmd`, etc.), cluster ID generation
- **`distribution.py`** — High-level resource distribution: IB detection, container image and model syncing to target hosts (orchestrates `models/`, `containers/`, and IB detection)
- **`infiniband.py`** — IB detection script generation, NCCL env var computation, IB IP mapping for fast transfers
- **`networking.py`** — ConnectX-7 NIC detection, IP assignment planning, CX7 configuration script generation, host key distribution
- **`primitives.py`** — Higher-level composition: `build_ssh_kwargs()`, `build_volumes()`, `merge_env()`, `detect_infiniband()`, `run_script_on_host()`, `cleanup_containers()`
- **`job_metadata.py`** — Persistent job metadata (cluster_id → recipe mapping) stored in `~/.cache/sparkrun/jobs/`

### Recipe System

Recipes are YAML files with fields: `model`, `runtime`, `container`, `command`, `defaults`, `env`, `metadata`,
`min_nodes`, `max_nodes`. The `Recipe` class (`core/recipe.py`) uses SAF `Variables` for config chain resolution —
CLI overrides → recipe defaults → runtime defaults.

Recipe resolution: CLI → `find_recipe()` (module-level function in `core/recipe.py`) → searches bundled recipes, local
`./recipes/`, user config recipes, and git-cloned registries.

Two recipe format versions exist: v1 (eugr-style, auto-detected by `recipe_version: "1"` or presence of `build_args`/
`mods`) and v2 (sparkrun native). vLLM recipes are resolved to either `vllm-ray` (if Ray hints are present) or
`vllm-distributed` (default). See `RECIPES.md` for the full specification.

### Registry System

The `RegistryManager` (`core/registry.py`) tracks recipe collections from remote git repos using sparse checkouts.
Registries are stored in `~/.config/sparkrun/registries.yaml`; cached clones live under `~/.cache/sparkrun/registries/`.

**Default registry initialization** (first run, no `registries.yaml`):

1. `_load_registries()` → no file → `_default_registries()`
2. `_default_registries()` calls `_init_defaults_from_manifests()` which clones each URL in `DEFAULT_REGISTRIES_GIT` and
   reads its `.sparkrun/registry.yaml` manifest via `_discover_manifest_entries()`. URLs that fail are skipped
   individually (partial success).
3. Discovered manifest entries are merged with `FALLBACK_DEFAULT_REGISTRIES`: manifest entries take priority by name;
   non-conflicting fallback entries are appended to fill gaps in registry coverage.
4. The combined list is saved to `registries.yaml` for subsequent loads.
5. If all manifest URLs fail, pure `FALLBACK_DEFAULT_REGISTRIES` is returned (offline/no-git safety net).

**Manifest format** (`.sparkrun/registry.yaml` in a git repo): supports both canonical keys (`subpath`,
`tuning_subpath`, `benchmark_subpath`) and short keys (`recipes`, `tuning`, `benchmarks`). Canonical keys take
precedence when both are present.

**Shared clones**: When multiple registries point to the same git URL, a single shared clone is used at `_url_<hash>/`
with per-registry symlinks. Sparse checkout paths are the union of all subpaths for that URL.

**Reserved name prefixes**: Names starting with reserved prefixes (`sparkrun`, `official`, `arena`, etc.) can only be
used by repos hosted under allowed GitHub organizations (`spark-arena`, `scitrera`, `eugr`, `dbotwinick`,
`raphaelamorim`). Enforced via `validate_registry_name()`.

**Tab completion**: `RecipeNameType.shell_complete()` in `_common.py` supports `@registry/recipe` syntax — `@` prefix
lists registries, `@registry/` lists recipes from that registry. Falls back to showing registry names when recipe cache
isn't populated.

### Model & Container Distribution

Before launching, sparkrun can pre-sync models and container images from the control machine to target hosts:

- **Models** (`models/`): Downloads from HuggingFace Hub locally via `snapshot_download` (`models/download.py`), then
  rsyncs to targets (`models/distribute.py`, `models/sync.py`). GGUF models use colon syntax (`repo:quant`) for
  selective quant-file download.
- **Containers** (`containers/`): Pulls image locally (`containers/registry.py`), then streams via
  `docker save | ssh docker load` (`containers/distribute.py`, `containers/sync.py`). Checks image IDs to skip hosts
  that already have the correct image.
- **VRAM estimation** (`models/vram.py`): Estimates VRAM usage based on model parameter count, dtype, and quantization.
  Supports HuggingFace model auto-detection to resolve parameter counts.

### Kernel Tuning (`tuning/`)

Provides utilities for running Triton fused MoE kernel tuning on DGX Spark and auto-mounting the resulting configs in
inference runs. Common tuning internals live in `tuning/_common.py`; runtime-specific helpers are in `tuning/sglang.py`
and `tuning/vllm.py`. `tuning/sync.py` handles syncing tuning configs from registries to local cache and runtime name
normalization.

### Utilities (`utils/`)

Shared helpers used across multiple modules to avoid circular imports:

- `coerce_value()` — type coercion for CLI string inputs (to int, float, bool)
- `suppress_noisy_loggers()` — silences verbose HTTP/transport loggers
- `resolve_ssh_user()` — SSH user resolution (cluster → config → env → fallback)
- `is_valid_ip()`, `parse_kv_output()`, `load_yaml()` — parsing helpers
- `cli_formatters.py` — Presentation-layer formatting for recipe tables and CLI output

### Config & State Paths

| Path                                 | Purpose                                           |
|--------------------------------------|---------------------------------------------------|
| `~/.config/sparkrun/config.yaml`     | User configuration                                |
| `~/.config/sparkrun/clusters/*.yaml` | Named cluster definitions                         |
| `~/.config/sparkrun/registries.yaml` | Custom recipe registry list                       |
| `~/.cache/sparkrun/registries/`      | Git-cloned recipe registries                      |
| `~/.cache/sparkrun/jobs/`            | Job metadata (cluster_id → recipe mapping)        |
| `~/.cache/sparkrun/pending/`         | PID lock files for in-progress operations         |
| `~/.cache/huggingface/`              | HuggingFace model cache (mounted into containers) |

### Testing Patterns

Tests use pytest with `pytest-asyncio`. The `conftest.py` provides an `isolate_stateful` autouse fixture that redirects
SAF's stateful root to `tmp_path`, preventing tests from touching `~/.config/sparkrun/`. The bootstrap singleton (
`_variables`) is reset between tests. All core module imports in tests use `sparkrun.core.*` paths (e.g.,
`from sparkrun.core.registry import RegistryManager`).

All SSH/Docker operations in tests are mocked — no real hosts are needed. Common fixtures: `tmp_recipe_dir` (creates
sample v1/v2 recipes), `cluster_dir`, `hosts_file`, `v` (initialized SAF Variables instance).

Test files cover: benchmarking, bootstrap, CLI commands, CLI recipe integration, cluster manager, config, distribution,
Docker command generation, GGUF handling, host resolution, InfiniBand, networking, orchestration primitives, recipes,
registry (including manifest discovery, fallback merging, shared clones, and reserved name enforcement), runtimes,
embedded scripts, SSH execution, kernel tuning, and VRAM estimation.

### Companion Packages

- **`sparkrun-cc-plugin/`** — Claude Code plugin providing slash commands (`/sparkrun:run`, `/sparkrun:stop`, `/sparkrun:status`, `/sparkrun:list`, `/sparkrun:setup`) and skills for AI-assisted inference management (`run`, `setup`, `registry`).
- **`website/`** — Documentation site built with Astro (Starlight theme), deployed to Cloudflare Pages.

## Key Dependencies

- **`scitrera-app-framework`** (SAF) — Plugin system, lifecycle, variables/config management
- **`vpd`** — YAML reading (`read_yaml`) and command template placeholder substitution (`arg_substitute`)
- **`click`** — CLI framework
- **`huggingface_hub`** — Model downloading (`snapshot_download`)
- **`pyyaml`** — YAML parsing for recipes, clusters, registries
