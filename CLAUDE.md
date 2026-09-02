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
uv sync

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
├── orchestration/      # SSH, Docker, InfiniBand, executors, collectives (see below)
├── transports/         # Cluster connectivity seam — how hosts are reached/prepared (ssh default + thunder)
├── platforms/          # HardwarePlatformPlugin registry (DGX Spark + generic NVIDIA today)
├── models/             # HuggingFace model download, distribution, and VRAM estimation
├── containers/         # Container image distribution (docker save/load over SSH)
├── tuning/             # Triton fused MoE kernel tuning for SGLang and vLLM
├── builders/           # Image + environment builder plugins (docker-pull, eugr, uv-venv)
├── diagnostics/        # Host and run diagnostic collection (NDJSON output)
├── plugins/            # In-tree cross-cutting integrations (see docs/PLUGINS.md)
├── proxy/              # Inference gateway (LiteLLM engine + gateway selection seam)
├── benchmarking/       # Benchmark framework plugins and result export (llama-benchy)
├── utils/              # Shared helpers (coerce_value, suppress_noisy_loggers, etc.)
└── scripts/            # Embedded bash scripts (IB detection, container launch, etc.)
```

### Core Data Models (`core/`)

Core domain logic extracted from the top-level package. All imports use `sparkrun.core.*` (e.g.,
`from sparkrun.core.config import SparkrunConfig`).

| Module                  | Purpose                                                                              |
|-------------------------|--------------------------------------------------------------------------------------|
| `bootstrap.py`          | SAF plugin initialization, runtime / benchmarking / builder / executor discovery     |
| `config.py`             | `SparkrunConfig` — reads `~/.config/sparkrun/config.yaml`, cache dir resolution      |
| `registry.py`           | `RegistryManager` — git-based recipe registry system (see Registry System below)     |
| `recipe.py`             | `Recipe` loading, validation, v1→v2 migration, config chain via SAF Variables         |
| `cluster_manager.py`    | `ClusterManager` — named cluster CRUD (YAML files in `~/.config/sparkrun/clusters/`) |
| `hosts.py`              | Host resolution priority chain (CLI → file → cluster → default)                      |
| `pending_ops.py`        | PID-based lock files for in-progress operations                                      |
| `benchmark_profiles.py` | Benchmark profile discovery, resolution, and rendering across registries             |
| `hardware.py`           | `AcceleratorSpec` / `HostHardware` / `default_dgx_spark_hardware()`                  |
| `hardware_probe.py`     | `probe_host` / `probe_hosts` — combined accelerator + InfiniBand SSH probe           |
| `fingerprint.py`        | Thin shim — accelerator-only parsing on top of the combined probe                    |
| `backend_select.py`     | `select_backends(HostHardware) -> BackendBundle`, `NoMatchingBackendError`           |
| `placement.py`          | `compute_placement()` — rank → (host, local-GPU) honoring `RecipeLayout`             |
| `layout.py`             | `RecipeLayout` / `Placement` dataclasses parsed from recipe `layout:` block          |
| `images.py`             | `ImagePlan` / `resolve_image_plan()` — per-machine `containers:` resolution           |
| `image_preparation.py`  | `prepare_images()` / `stage_prepared_images()` — builder + image plan as one step     |
| `recipe_items.py`       | `register_recipe_item()` — plugin-owned top-level recipe keys                        |
| `execution.py`          | `RecipeExecutionStrategy`, `PreparationStep` DAG, `LaunchAssetPolicy`                |
| `launcher.py`           | `launch_inference()`, `resolve_per_host_backends()`, `resolve_recipe_trust()`        |
| `validation.py`         | `RecipeIssue`, `validate_recipe()`, `validate_for_launch()` (see Recipe Validation)  |

### CLI Architecture (`cli/`)

The CLI was split from a single `cli.py` into a package for maintainability. The `__init__.py` defines the top-level
`main` Click group, registers all subcommands, and provides top-level aliases (`list`, `show`, `search`, `status`).

| Module            | Purpose                                                                                                                                                                                                                                                           |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `__init__.py`     | `main` Click group, command registration, top-level aliases                                                                                                                                                                                                       |
| `_common.py`      | Shared infrastructure: logging setup, Click parameter types (`RECIPE_NAME`, `REGISTRY_NAME`, `RUNTIME_NAME`, `CLUSTER_NAME`, `PROFILE_NAME`), decorators (`host_options`, `dry_run_option`), and reusable helpers (host resolution, recipe loading, VRAM display) |
| `_run.py`         | `run` command — launch inference workloads                                                                                                                                                                                                                        |
| `_stop_logs.py`   | `stop` and `logs` commands — stop workloads and stream container logs                                                                                                                                                                                             |
| `_setup.py`       | `setup` command group — shell completion, SSH mesh, model/container sync, permissions, cache, networking, GPU clock throttling                                                                                                                                                          |
| `_cluster.py`     | `cluster` command group — create/list/show/delete/update saved cluster definitions, cluster status                                                                                                                                                                |
| `_recipe.py`      | `recipe` command group — list/show/search recipes across registries                                                                                                                                                                                               |
| `_registry.py`    | `registry` command group — add/remove/enable/disable/update registries, list/show benchmark profiles                                                                                                                                                              |
| `_benchmark.py`   | `benchmark` command group — run benchmark profiles against inference workloads                                                                                                                                                                                    |
| `_tune.py`        | `tune` command group — run Triton fused MoE kernel tuning (SGLang and vLLM)                                                                                                                                                                                       |
| `_wizard.py`      | `setup wizard` command — guided cluster setup                                                                                                                                                                                                                     |
| `_check.py`       | `setup check` command — non-destructive readiness probe of a cluster's hosts against the wizard's setup steps (ordered `SETUP_CHECKS` registry; seed of a future per-platform step system with paired check/apply stages)                                          |
| `_proxy.py`       | `proxy` command group — thin renderer over `api.proxy` (see Inference Gateway below)                                                                                                                                                                              |
| `_monitor_tui.py` | Textual TUI for `cluster monitor`                                                                                                                                                                                                                                 |
| `ext.py`          | Plugin CLI-command extension point — `register_cli_command(cmd, parent=…)` + `PluggableGroup` (see below)                                                                                                                                                          |

**Plugin CLI commands** (`cli/ext.py`): external plugins add Click commands via
`register_cli_command(command, parent=(...))` (`parent=()` → top-level;
`parent=("cluster","import")` → nested). The top-level `main` is a
`PluggableGroup`: on first command resolution it runs `ensure_cli_extensions`
(→ `init_sparkrun`, which imports plugins so they register, → attach), so plugin
commands appear in `--help` and dispatch like built-ins even though the command
tree is built at import time (before plugins load). Attachment is idempotent and
never clobbers a built-in; command↔plugin mapping is intentionally free-form
(not tied to the transport/executor abstractions). Per-command gating is the
command's own concern.

### Plugin System (SAF)

sparkrun uses [scitrera-app-framework](https://github.com/scitrera/python-app-framework) (SAF) for plugin discovery and
lifecycle. Six extension points are registered:

| Extension point        | Constant       | Module scanned                       | Base class              |
|------------------------|----------------|--------------------------------------|-------------------------|
| `sparkrun.runtime`     | `EXT_RUNTIME`  | `sparkrun.runtimes`                  | `RuntimePlugin`         |
| `sparkrun.builder`     | `EXT_BUILDER`  | `sparkrun.builders`                  | `BuilderPlugin`         |
| `sparkrun.benchmarking`| `EXT_BENCHMARKING` | `sparkrun.benchmarking`          | `BenchmarkingPlugin`    |
| `sparkrun.executor`    | `EXT_EXECUTOR` | `sparkrun.orchestration.executors`   | `Executor`              |
| `sparkrun.scheduler`   | (scheduler)    | `sparkrun.schedulers`                | `Scheduler`             |
| `sparkrun.transport`   | `EXT_TRANSPORT`| `sparkrun.transports`                | `Transport`             |

Key bootstrap flow: `cli/__init__.py` → `core.bootstrap.init_sparkrun()` → SAF `init_framework_desktop()` →
`find_types_in_modules(...)` over each scanned module above → `register_plugin()` for each discovered plugin (schedulers
and transports skip base classes with a blank `scheduler_name` / `transport_name`). Finally
`load_external_plugins(v)` loads any out-of-tree plugins (see External Plugins below).

The `EXT_PLATFORM` constant is defined in `platforms/base.py` for future SAF
entry-point discovery; today `platforms/__init__.py` keeps an ordered
in-process registry that callers iterate via `resolve_platform()`. Platforms
stay in-process (not SAF-scanned) because their resolution is **order-sensitive**
(most-specific `matches()` first) — transports, which select by exact name, moved
onto SAF; platforms did not.

**Platform default tiers.** A platform publishes hardware-conditional defaults
through five hooks, each folded in at a different layer so anything the user
wrote always wins:

| Hook                                   | Scope                        | Folded in by                                              |
|----------------------------------------|------------------------------|-----------------------------------------------------------|
| `default_image(runtime)`               | container image              | `RuntimePlugin.default_image_for()`                        |
| `default_runtime_flags(runtime, accel)`| recipe `defaults` (serve flags) | `launcher.apply_platform_runtime_flag_defaults` (`setdefault`) |
| `default_env(runtime, accel, family=)` | container env                | `launcher.resolve_platform_env_defaults` (lowest env tier) |
| `default_executor_config(executor)`    | `ExecutorConfig`             | `executor.resolve_executor(host_hardware=…)`, layer 9      |
| `default_max_gpu_memory_utilization`   | usable-memory cap            | `core.limits`                                              |

All are keyed off the **head host's** hardware — one image / serve command /
executor is built per launch, so a representative host is the right scope.
`default_env` receives the runtime *family* (`get_family()`, e.g. `"vllm"` for
vllm-ray / vllm-distributed / eugr-vllm) alongside the exact name, so a platform
can target a family without enumerating variants; exact name wins over family.
Today DGX Spark uses this for `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
on vllm/sglang and `gpu_access_mode: gpus` (classic `--gpus` rather than CDI,
whose `/etc/cdi/nvidia.yaml` goes stale across driver upgrades).

### In-Tree Plugins (`plugins/` + `core/in_tree_plugins.py`)

`sparkrun.plugins` is the **mate of the out-of-tree plugin system**: same
registration path (`external_plugins.load_plugin_module` — SAF subclass scan,
then the `register(v)` hook), different source. A first-party integration
therefore has no capability an external one lacks.

It is for **cross-cutting integrations only** — things that span several
extension points and are only coherent as one removable unit: an integration
contributing a backend implementation, a hidden CLI command and the wire
protocol that backend calls back through, none of which is "a runtime" or "an
executor" on its own. Packages that map cleanly onto one extension point
(`runtimes`, `transports`, `executors`, `schedulers`, `builders`,
`benchmarking`) stay where they are with their own `find_types_in_modules` scan
in `core/bootstrap.py`.

**Every in-tree plugin is gated by a feature flag** — the same reason
`executor.docker` and `gateway.litellm` carry one despite shipping on. The flag
is checked *before* the import, so turning a plugin off costs nothing at all:
no import, no commands, no registrations. It is the plugin's **own** flag, not
a separate presence flag — a dual-flag scheme was considered and rejected as
incoherent (no point loading a plugin whose capability won't be used, and none
in enabling that capability without the plugin).

The binding lives in `in_tree_plugins.IN_TREE_PLUGIN_FEATURES` (data, not a
plugin attribute) because the flag must resolve *without importing* the module
it gates. A plugin missing an entry or a registered flag is skipped loudly,
since unknown flags fail closed and the silent version would put the symptom
far from its cause.

One consequence worth knowing: `PluggableGroup` attaches plugin commands once
per process, so the gate is effectively read once per CLI invocation (the test
suite resets `_cli_ext_loaded`).

**What a cross-cutting plugin gets.** `sparkrun.plugins` re-exports the two
registration entry points — `register_recipe_item` (own a top-level recipe key,
optionally with an execution strategy and preparation steps) and
`register_cli_command`. Together with `Transport.open_host_session`,
`api.materialize` and `SparkrunConfig.plugin_settings`, those are the seams that
let an integration participate in a launch without forking it. See
`docs/PLUGINS.md` for the author-facing version, and the Plugin-Owned Recipe
Items / Recipe Execution Strategies / Launch Materialization sections below for
the rationale.

**Layering trap.** `init_sparkrun` runs on the console-free `sparkrun.api` path,
and plugin scanning imports *every* submodule of a plugin package. So the CLI
registry lives in `core/cli_registry.py` (Click-free; `cli/ext.py` re-exports it
and owns attachment), and a plugin registers a **loader** that *builds* its
command with `import click` inside the function. Registering lazily is not
enough on its own. `test_api_sctx_threading.py::test_sctx_layer_does_not_import_click`
guards this.

### External Plugins (`core/external_plugins.py`)

Out-of-tree plugins (private executors, transports, runtimes, …) load from
directories listed under `plugins.paths` in `config.yaml`
(`SparkrunConfig.external_plugin_paths`). `load_external_plugins(v)`, called at
the end of `init_sparkrun`, prepends each dir to `sys.path`, imports every
top-level module/package in it, scans each for the six SAF plugin base types and
`register_plugin`s them, then calls an optional module-level `register(v)` hook —
the escape hatch for the still-in-process registries (`platforms`,
`collectives`), which register via `register_platform()` rather than SAF subclass
discovery. Loading is trusted by definition (the config + dirs are user-owned);
a broken plugin logs and is skipped, never breaking startup.

**Gated off by default** behind the `core.external_plugins` feature flag (off on
every channel). When the flag resolves off, the config-driven path returns
immediately **without reading `plugins.paths`** (let alone importing anything) —
the gate uses the same context-free `feature_gate_enabled` resolution as the
executor/transport gates, so no init cycle. Enable via `sparkrun setup features
enable core.external_plugins`. The flag gates only the auto-load (`paths=None`)
path; an explicit `paths=` argument (programmatic / a plugin's own tests)
bypasses it.

Test isolation uses a separate hard kill-switch, `SPARKRUN_NO_EXTERNAL_PLUGINS`
(set by `conftest.isolate_stateful`): the feature flag alone is insufficient
because pytest reads the developer's **real** `~/.config/sparkrun` (the SAF
stateful root isn't "ready" under pytest), so a developer who enabled the flag
would otherwise load their real plugins mid-suite. The kill-switch short-circuits
the auto-load path before the flag is even consulted.

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

**Node-command template** (`RuntimePlugin._make_node_command_args`): native
multi-node runtimes (`vllm-distributed`, `sglang`, `trtllm`) emit rank-specific
argv via this template rather than ad-hoc per-runtime construction. Subclasses
override the hook methods (`_node_rank_args`, `_master_args`, etc.) and inherit
the assembly.

**Data parallelism is not one thing** (`runtimes/sglang.py:_SglangTopology`).
sparkrun's `world_size = tp*pp*dp` says dp *replicates*, but each engine
spells that differently and some refuse it in the shape sparkrun asks for.
SGLang asserts both of these **before binding a port**, so getting it wrong
is an abort, not a degradation:

```python
assert (tp_size * pp_size) % nnodes == 0
assert not (dp_size > 1 and nnodes != 1 and not enable_dp_attention)
```

Which means `--nnodes`/`--node-rank` cannot be injected unconditionally, and
whether `--dp-size` is legal is a property of the **launch**, not of the
recipe — so it is emitted by the caller (`_append_dp_size`), never from the
flag map. Four regimes, and the boundary between them is `dp > 1 and not
dp_attention` (`_SglangTopology.independent_replicas`):

| Regime | World | Flags |
|---|---|---|
| `dp == 1` | one, spanning the cluster | `--nnodes N --node-rank r --dist-init-addr head` |
| DP attention | one, spanning the cluster | the above **plus** `--dp-size` |
| `dp > 1`, `tp*pp > 1` | one **per replica** | `--nnodes tp*pp`, intra-replica rank, rendezvous at *that replica's* head |
| `dp > 1`, `tp*pp == 1` | none | nothing — a standalone server |
| solo (`num_nodes == 1`), `dp > 1` | one | `--dp-size dp` (replicas share the launch) |

Three consequences that are each silently wrong if missed:

- **DP attention is not a multiplier.** There `dp` partitions the tensor world
  (`dp == tp`), so the job costs `tp*pp` GPUs, not `tp*pp*dp` — hence
  `SglangRuntime.world_size`. Without it a 2-node `tp 16 / dp 16` DeepSeek
  layout is scheduled at 256 ranks.
- **`_dp_attention_enabled` reads the `command:` template too.** A literal
  `--enable-dp-attention` is invisible to the config chain, and misreading a
  working MoE recipe as "independent replicas" would strip the rendezvous
  flags that make it work.
- **The global `node_rank` goes into `_make_node_command_args` paired with
  `replica_size=world_nodes`** — that pairing is what selects the world
  (`_resolve_master_addr` floor-divides them). Passing the intra-world rank
  collapses every replica onto `hosts[0]`; passing `replica_size=1` points
  each node at itself ("1/N clients joined").

**Independent replicas are N endpoints, not one.** `validate_recipe` warns
and points at `sglang_router` / `sparkrun proxy`; proxy discovery still
registers only the head (`proxy/discovery.py`), so the other replicas are
live but unrouted. See issue #284.

**Rendezvous gate** (`RuntimePlugin.native_rendezvous_port`): the native
cluster path starts the head, waits for its port, then starts the workers —
the wait exists so a worker cannot race the head's distributed store, and
`None` says this launch has no store. `None` deliberately does **not** mean
"wait on the serve port instead": a serve port opens only after weight load
and graph capture, which this gate's budget (60×2s) does not cover, and
endpoint readiness already has its own budgeted watcher
(`launcher.wait_for_endpoint_ready`). Reusing it here would time out a
healthy launch.

**Executor resolution** (`RuntimePlugin._resolve_executor`): runtimes do not
construct executors directly. The base helper delegates to
`orchestration.executor:resolve_executor()` — a single layered chain (CLI →
recipe → `runtime.default_executor()` → per-executor adjustments →
`SparkrunConfig` → per-executor defaults → dataclass field defaults). The
previously-hardcoded `_KNOWN_EXECUTORS` set has been retired; selector
validation queries SAF via `get_extensions(EXT_EXECUTOR)`.

**Unmapped-key reporting** (`RuntimePlugin.known_config_keys` +
`launcher.py:report_unmapped_config_keys`): a structured runtime builds its
serve command by iterating a flag map, so a `defaults:` key — or a
`-o key=value` — the map doesn't list is **dropped**: no error, no warning,
nothing in the rendered command, and the engine quietly uses its own default.
That is how an `@atlas` recipe's `lm_head_dtype: bf16` correctness pin served
weeks of traffic at NVFP4 (#276), and the same gap as `--disable-tool-grammar`
in #221.

A runtime declares the keys it understands — its flag map **plus what it
consumes outside it** (`prepare()`, parallelism, builder, executor); a key
handled elsewhere reported as dropped is noise that trains people to ignore
the report. `BASE_CONSUMED_CONFIG_KEYS` (`runtimes/base.py`) covers what the
shared layers read for every runtime. `None` — the base default — means "not
declared" and disables the check, which is what `eugr-vllm` returns: it
inherits vLLM's map but routes v1 `defaults` through eugr's `build_args` /
`mods`, so the inherited answer would be wrong.

Three things are deliberately *not* reported, and each was load-bearing in
getting the false-positive rate to zero across all 292 cached registry
recipes: keys referenced as a `{placeholder}` in `command:` **or in another
default's value** (`render_template` iterates, so one default may exist only
to feed another), `_`-prefixed keys sparkrun injects mid-launch, and dotted
keys routed by prefix (`-o env.KEY=VALUE`). It warns rather than raises
because registries version independently of sparkrun — an unknown key is
routinely a *newer* recipe, and hard-failing would strand a user between two
published artifacts. It runs from `launch_inference` after the platform
default tier (so a platform contributing an unmapped flag is caught too) and
before anything starts, so `--dry-run` reports it.

The atlas flag map is now exhaustive against Atlas's own machine-readable
`vendor/serve-options.v1.json`, which also distinguishes the two boolean
shapes: presence-only (`_ATLAS_BOOL_FLAGS`, bare flag) versus `Option<bool>`
(`_ATLAS_VALUE_BOOL_FLAGS`, explicit lowercase `true`/`false`, where *absent*
defers to MODEL.toml but `false` overrides it — so dropping the flag for a
falsy value hands the decision back to the engine).

**Trust gating** (`launcher.py:resolve_recipe_trust`): each launch resolves a
single trust verdict shared by `pre_exec` (inside `runtime.run()`) and
`post_exec` / `post_commands` (inside `post_launch_lifecycle`). Local recipes
and default-registry recipes are auto-trusted; third-party registry recipes
prompt unless `--trust` is passed. See `docs/SECURITY.md`.

**Backend bundle**: `RuntimePlugin.run()` accepts a keyword-only
`backends: dict[str, BackendBundle] | None` resolved by
`launcher.resolve_per_host_backends()`. Runtimes route per-host env emission
through `_cluster_ops.resolve_comm_env(ctx, comm_env, backends)`. When
`backends` is `None`, `resolve_comm_env` falls back to the legacy NCCL
generator (byte-identical for NVIDIA hosts).

### Environment Builders (`builders/uv_venv.py`)

A builder's canonical hook is `prepare()`, and it does not have to prepare an
*image*. `uv-venv` is the first **environment** builder: it provisions a
`uv`-created Python venv on each target host and writes a shell `env_file` that
activates it, returning the image ref untouched. It pairs with `executor: local`
(native, no container) — the answer to hosts where nested `docker run` doesn't
work (Thunder's proot/fastvfs sandbox).

The builder→executor coupling is a **core seam, not a special case**:
`BuilderPlugin.default_env_file(recipe)` contributes `{"env_file": …}` as one
layer of `resolve_executor`'s chain (`orchestration/executor.py:_builder_exec_dict`).
It sits below the recipe layer (an explicit `executor_config.env_file` wins) but
*above* cluster/runtime/config, because an environment builder's env_file is
essential to running the workload at all — a cluster's generic one must not
silently suppress it and leave the serve command under the wrong interpreter.

Four properties are load-bearing and easy to break:

- **Idempotency is content-addressed.** `dep_hash()` covers python,
  torch_backend, inline requirements *and the contents* of any
  `requirements_file`/`pyproject` — which are read control-side and embedded in
  the provisioning script, never transferred. With no `venv_path`, the venv
  lives at `$HOME/.cache/sparkrun/uv-venv/<dep-hash>`, so recipes with identical
  deps share one venv and editing a requirements file re-provisions.
- **The marker guards the venv, not the env_file.** `cuda_home` deliberately
  stays *out* of `dep_hash` (it doesn't change the installed packages, and
  including it would rebuild a multi-GB venv to edit one `export`), so the
  env_file is rewritten on every run while the expensive half stays behind the
  marker. Guarding the whole script on the marker — as the original did — made
  adding `cuda_home` to a recipe whose venv already existed a *silent no-op*.
- **Quoting is split deliberately, and what can't be quoted is validated.**
  `venv_path`/`env_file`/`cuda_home` are emitted double-quoted so bash expands
  `$HOME` *on the host*, which rules out `shlex`-quoting them — so they are
  validated against a strict charset instead (`$HOME`/`~` prefix + `[A-Za-z0-9_./+-]`).
  This is not optional: `builder_config` is recipe content, recipes come from
  registries, and unlike `executor_config` it has **no trust gate**. Requirements
  and python are `shlex`-quoted; a requirement may not begin with `-`, because
  `--index-url=…` survives `shlex.quote` untouched and would reach `uv` as a
  flag that repoints the package index.
- **Heredoc delimiters are content-derived.** Staged requirement files are
  written through a quoted heredoc whose delimiter is seeded from a hash of the
  content (and extended on collision). A fixed delimiter is an injection
  vector — a requirements file containing that line closes the heredoc early
  and everything after it executes as shell.

`prepare()` fans out with `run_remote_scripts_parallel(..., allow_local=True,
session_guard=True)`. The guard is not optional for this payload: a first-time
vllm+torch install is minutes of network per host, and without it a Ctrl-C on
the control node leaves `uv pip install` running on every host with nothing left
to observe or stop it (issue #240).

**Gating** is `builder.uv_venv` — off on `stable`, on for `beta`/`alpha` via
`channel_defaults`. Unlike an image builder it mutates the host (creating venvs,
installing packages), so stable requires an explicit opt-in.
`BuilderPlugin.required_feature_flag` + the `is_multi_extension` self-gate is
the same mechanism `Executor`/`Transport`/`TelemetryProvider` use.

Two consequences of gating a *builder* specifically:

- `get_builder` must distinguish **disabled** from **unknown**, or a stable user
  running an alpha recipe is told their recipe is wrong when their channel is.
  A gated builder is hidden from `get_extensions`, so the distinction is
  recorded at discovery in `bootstrap._BUILDER_GATES` (the one point where both
  are visible) and raises `BuilderUnavailableError`.
- The launcher's builder phase **must not skip it**. Phase 2 warns-and-continues
  for an *unknown* builder only; `BuilderUnavailableError` is re-raised, and the
  `try` now wraps `get_builder` alone — a `ValueError` out of `prepare()` is a
  build failure, and reporting it as "builder not found, skipping" launched the
  workload without the environment it asked for.

**Aliases**: `builder_aliases` lets one builder answer to several spellings
(`venv` → `uv-venv`). `list_builders()` returns canonical names only — an alias
is another spelling of one builder, and listing it would imply a second exists
and put a phantom name in every "Available: […]".

### Orchestration Layer (`orchestration/`)

All remote operations use **SSH stdin piping** — scripts are generated as Python strings and piped to `ssh <host> bash -s`. No files are ever copied to remote hosts.

- **`ssh.py`** — `RemoteResult` dataclass, `build_ssh_cmd()`, `run_remote_script()`, `run_remote_scripts_parallel()`, `run_rsync_parallel()`, `stream_remote_logs()`
- **`sudo.py`** — `run_with_sudo_fallback()` — tries non-interactive sudo in parallel, then falls back to password-based sudo for failures
- **`docker.py`** — Pure command-string generators (`docker_run_cmd`, `docker_exec_cmd`, etc.), cluster ID generation
- **`distribution.py`** — High-level resource distribution: IB detection, container image and model syncing to target hosts (orchestrates `models/`, `containers/`, and IB detection)
- **`infiniband.py`** — IB detection script generation, NCCL env var computation, IB IP mapping for fast transfers
- **`networking.py`** — ConnectX-7 NIC detection, IP assignment planning, CX7 configuration script generation, host key distribution
- **`primitives.py`** — Higher-level composition: `build_ssh_kwargs()`, `build_volumes()`, `merge_env()`, `detect_infiniband()`, `run_script_on_host()`, `cleanup_containers()`
- **`job_metadata.py`** — Persistent job metadata (cluster_id → recipe mapping) stored in `~/.cache/sparkrun/jobs/` (see Job Metadata Lifecycle below)
- **`executor.py`** — Public facade. Re-exports `Executor`, `ExecutorConfig`, `EXT_EXECUTOR`. `resolve_executor()` is the single sanctioned executor entry point; `query_status_for_cluster()` is the single status source (see Status Discovery below).
- **`executors/`** — Executor plugin package. `_base.py` (ABC + dataclass), `docker.py` (default), `local.py` (experimental, no container), `k8s.py` (experimental draft, `kubectl run`-driven). Discovered via SAF. Each declares a `status_scope` (default `"host"`).
- **`collectives/`** — `CollectiveBackend` ABC + implementations: `nccl.py` (default; wraps `infiniband.py`), `rccl.py` (AMD scaffold), `hccl.py` (Intel Gaudi scaffold). `get_backend(vendor)` is the lookup.
- **`hooks.py`** — `pre_exec` / `post_exec` / `post_commands` runners. Trust gating via `_confirm_hook_execution(trust=...)`.

**Management-interface detection** (`scripts/_mgmt_iface.sh`): every host probe
needs to know which NIC is the *management* interface — it becomes
`DETECTED_SOCKET_IFNAME` and from there `GLOO_SOCKET_IFNAME` /
`TP_SOCKET_IFNAME` / `MN_IF_NAME`, the head of `NCCL_SOCKET_IFNAME`, and
`NODE_IP`. Seven probes each answered it with `ip route get 8.8.8.8 || echo
"eth0"`, which on an air-gapped Spark (no default route, by design in `push`
mode) emitted an interface that does not exist on the hardware and killed the
launch at gloo init (issue #275).

They now share one helper with a four-step chain — operator pin
(`SPARKRUN_MGMT_IFACE`) → default route → the interface holding the local end
of our own `SSH_CONNECTION` → first physical NIC that is up with a global IPv4
— and **print nothing when none of them resolves**. Empty is the load-bearing
part: `generate_nccl_env` already falls back to `DETECTED_NET_LIST` (the
fabric adapters, which exist), whereas a guessed name is unrecoverable. Every
candidate is checked against sysfs first, so a pin naming an absent device
warns and falls through rather than being passed along.

Two details are easy to break. The `<sysfs>/<if>/device` test is what
separates real NICs from `docker0` / `br-*` / `veth*` / `tailscale0` (several
hold a global IPv4 and sort *ahead* of the mgmt NIC), and `device/infiniband`
is what stops the scan claiming a CX7 port as "management" — pinning the
fabric is a deliberate act (`pin_comm_env_to_ib`), never a fallback. The
helper is also written **brace-free, including its comments**: `ray_head.sh` /
`ray_worker.sh` / the combined probe run it through `str.format()`, so one
brace raises `KeyError` inside a Ray launch. `tests/test_mgmt_iface.py` runs
the helper under real bash against a fixture sysfs tree and guards both.

Scripts compose via a `# sparkrun:include <file>` directive resolved in
`scripts/read_script` (`load_script_resource` is the *raw* loader and does not
expand it); `scripts.inject_shell_vars` passes config into a script piped to
`bash -s`, which takes no arguments. An included helper must not default a
var `inject_shell_vars` sets — it is included partway down and would clobber
the injected value. `ClusterDefinition.mgmt_interface` is the persistent
override, threaded through `detect_ib_for_hosts` / `distribute_from_config` /
solo `_run_solo`.

**Address-persistence attribution** (`scripts/_net_persist.sh`): "will this CX7
address survive a reboot?" was answered by testing for
`/etc/netplan/40-cx7.yaml`, which actually answers "did *sparkrun* write it?".
Those are different questions, and `setup check` reported every host
configured another way as a defect — pointing at a `sparkrun setup cx7` that
then correctly did nothing, because `plan_cluster_cx7` already reads the live
IPs and returns `"already configured"`. On Ubuntu 24.04 / DGX OS 7 the false
positive is the *common* case: netplan renders through NetworkManager, and
`nmcli con add` writes its own `90-NM-<uuid>.yaml`.

The helper attributes the live address to whatever owns it, first hit wins:
`netplan status --format=json` (an interface netplan owns carries an `id`;
this is netplan's *merged* view, so any filename counts) → the active
NetworkManager profile (`autoconnect` + `ipv4.method` + a matching
`ipv4.addresses`) → `networkctl`'s `Network File:` → `/etc/network/interfaces`.
Every probe is read-only and works unprivileged — deliberately, since `netplan
get` does not (the files are mode 600) and detection runs as the SSH user.

`CX7Persistence` is tri-state and that is the load-bearing part: **`UNKNOWN`
(no probe available) must never render as "won't survive reboot"** — the rule
`TerminationInfo.exists=None` follows. A failing `nmcli` counts as *unprobed*
rather than as "NM says no", so a half-interrogated host degrades to `UNKNOWN`
rather than to `EPHEMERAL`. `EPHEMERAL` is only claimed when a probe actually
answered and nothing owned the address. `CX7HostDetection.netplan_exists`
survives, but now means only "sparkrun's own file is present", which is what
the *uninstall* path needs.

Two consequences beyond the check. `plan_cluster_cx7` / `plan_ring_cx7` warn
(never refuse) when they would reconfigure a device a non-netplan source
persists — writing `CX7_NETPLAN_FILE` there leaves two owners, and which wins
is a property of the host's renderer. And `cx7_unconfigure.sh` reports
`FOREIGN:` for interfaces it cannot release instead of printing `SKIPPED` and
letting `setup uninstall` claim a teardown it did not perform; `_teardown_cx7`
renders those as `[WARN]`. `CX7_NETPLAN_FILE` is the one Python spelling of
the path, drift-guarded against the two bash scripts by a test (they include
brace-using helpers, so they cannot take it as a `.format()` placeholder).

**Session guard** (`ssh.py:wrap_with_session_guard` + `scripts/session_guard.sh`):
remote payloads run via `ssh <host> bash -s`, i.e. **without a PTY**, so on
disconnect sshd's session process exits without signalling its child (the
SIGHUP-on-disconnect path is PTY-only) and the payload is merely reparented —
a killed `sparkrun` would leave `hf download` / `docker pull` / rsync fan-outs
running on the cluster, invisible from the control node and stacking across
retries (issue #240). The guard backgrounds the payload in its own process
group (`set -m` outside, `set +m` *inside*, so a payload that backgrounds its
own jobs stays in one killable group) and polls its own PPID; on reparent it
TERMs then KILLs the group. Transparent otherwise — stdout, stderr and rc pass
through.

It is **opt-in** per call site via `session_guard=True` on `run_remote_script`,
`run_remote_scripts_parallel` and `run_remote_script_streaming` (mirroring the
`allow_local` precedent), and is on for the long, expensive payloads only:
`_distribute_from_head` (both steps), `sync_resource_to_hosts` (model + image
parallel sync), and the eugr container build. Short status probes stay
unguarded — an orphaned `docker ps` is harmless. Never applied on the
local-dispatch path (no session to lose) or under `--dry-run`. Kill switch:
`SPARKRUN_NO_SESSION_GUARD=1`.

The control-node half is `cli/_common.py:install_termination_handlers`, called
from the `main` group callback: SIGTERM/SIGHUP raise `KeyboardInterrupt`, so
`subprocess.run`'s existing kill-on-exception cleanup drops the SSH client (and
every `KeyboardInterrupt` state-preservation path already in the codebase runs).
`SIGKILL` remains unreachable — it leaves the ssh client alive, the session
healthy, and the guard correctly dormant.

### Launch Timing (`core/timing.py`)

Launch-stage timings are collected as a **span timeline**, not a
`{stage: seconds}` mapping: phases nest (phase 5 → runtime steps), fan-out
spans repeat their name (one per host / distribution entry) so names are not
keys, and consumers want the tree — a flattened summary is derivable from it,
not the reverse.

Almost no new call sites were needed, because the elapsed times were already
being computed and thrown away. `LaunchProgress.phase()` / `step()` timed
every bracket only to log `done (%.1fs)`; they now also record spans when a
`Timeline` is attached. `PHASE_SLUGS` gives the phases stable span names
(`launch.distribute`) deliberately *not* derived from `PHASE_LABELS` —
rewording a display string must not rename a metric already in published
artifacts. Step spans slugify the runtime's label and keep the raw label in
`attrs`.

| Piece | Role |
|-------|------|
| `Timeline.begin/end` | explicit brackets, for begin/end that aren't lexically paired |
| `Timeline.span` / `timing.timed` | context managers; `timed` tolerates `timeline=None` |
| `LaunchResult.timeline` / `RunResult.timeline` | the live collector, not a snapshot |
| `SparkrunContext.timing` | caller-owned timeline, widening the window past one launch |
| `format_launch_timings` | the `run` timing tree (on by default; `--no-timings`) |
| `launcher.ReadinessWatcher` | the readiness poll, run alongside `run`'s log stream |

`None` means "not collecting" everywhere and is byte-identical to the
behaviour before the module existed — the convention `progress` and
`backends` already use.

Five things are load-bearing:

- **Closing a span closes its open children, carrying its status.** Runtimes
  call `step()` and never `step_done()`, so the phase boundary is what ends
  the last step; and if the phase failed, the step that was running is *where*
  it failed — reporting it `ok` points the reader at the wrong stage.
- **Phases parent to an explicit root**, not to the timeline's open-span
  stack. A phase skipped while the previous one is still open would otherwise
  nest inside its predecessor — invisible at the call site, wrong in every
  consumer.
- **`skipped` is not a zero-duration success.** "Distribution found the image
  already resident" and "distribution never ran" are different data points in
  a benchmark artifact.
- **`export()` reports unclosed spans as `open` without closing them.** A
  launch that raised is when the breakdown is worth most, and the open spans
  are exactly the path to the failure. But `LaunchResult` is never built on
  that path, so **a caller that wants the timeline on failure must own it** —
  set `sctx.timing` (what `run` / `--collect-diagnostics` do) rather than
  letting `launch_inference` create one internally.
- **Clocks are never mixed in a derived figure.** A measured span is on the
  control node's `time.monotonic()` (`CLOCK_CONTROL`), which is what makes
  durations subtractable. Spans recovered from a remote engine's own log
  timestamps (`remote_clock(host)`) are a *different* clock, and with NTP
  skew "container start → weights loaded" across the two comes out negative.
  Two things enforce the rule rather than leaving it to discipline:
  `Timeline.add_span` — the only way in for a foreign clock — **rejects
  `CLOCK_CONTROL`** (a control-clock span must be *measured*, never asserted,
  or a parsed duration lands in the set that gets summed), and
  `Timeline.total` filters by clock and defaults to the control one. A
  foreign span's `t_start` is a placement estimate derived through our
  origin, so it keeps its unconverted `wall_start`; `export()` adds a
  `clocks` key **only** when mixed, so a consumer that never sees it is
  reading a single-clock timeline and may sum freely, and
  `format_launch_timings` annotates foreign rows `[remote:h1]` so the tree
  does not read as one arithmetic whole. Nothing produces a foreign-clock
  span yet — this is the seam a log probe plugs into.

**Time to first inference** is `serve.port_open` + `serve.health_ok`, recorded
by `wait_for_serve_ready`. Note which stage is the long pole: sglang and vLLM
V1 start their HTTP server **after** engine init, weight load *and*
CUDA-graph capture have all finished, so `serve.port_open` absorbs nearly the
whole startup and `serve.health_ok` is seconds. The original budgets assumed
the opposite (port `120×2s`, health `120×5s`) and a 30B NVFP4 spec-decode
model on 2 Sparks — 775s to bind, 570s of it capturing target-verify graphs —
was reported as an endpoint that never came up.

Two things follow, and both are in `DEFAULT_PORT_READY_TIMEOUT_S`'s docstring:

- **Budgets are wall-clock, not retry counts.** A retry count is a poor proxy
  for time because every attempt also pays a probe: `120 × 2s` bought 240s of
  *sleeping* but ran 321s, and the "%ds elapsed" progress line understated it
  by the same third. `wait_for_port` / `wait_for_healthy` take `timeout_s`
  (superseding `max_retries`, which stays for the callers that predate it);
  `math.inf` polls until cancelled. Overridable via `readiness.port_timeout_s`
  / `readiness.health_timeout_s`, where `0` means unbounded.
- **A generous budget is safe, because the budget is not what detects
  failure.** `wait_for_port` re-checks container liveness on every attempt and
  `wait_for_healthy` gives up after `max_consecutive_refused`, so a workload
  that actually died is caught within one interval either way. The budget's
  only job is to bound a *hang* — which is why the background watcher on `run`
  uses `math.inf` (it is cancelled by the log stream ending, so a fixed budget
  there could only ever expire early and mislabel a slow engine).

`benchmark` used to reimplement that two-stage wait inline
with its own retry budgets, which would have made the figure incomparable
between `run` and `benchmark`; both now go through
`launcher.wait_for_endpoint_ready` (the field-based form — the `--ensure`
already-running path has no `LaunchResult` but still has to wait). That
unification also fixed a live bug: the wait reconstructed the head container
name as `<id>_node_0`, but Ray runtimes head `<id>_head`, and `wait_for_port`
reads "that container isn't running" as proof the workload died — so every
Ray launch reached via `proxy load` or a post-hook recipe aborted one retry
interval in. It now asks `runtime.get_head_container_name()`, which also
routes through the resolved executor.

**On `run`, that wait is concurrent with the log stream** (`ReadinessWatcher`).
`sparkrun run` attaches to the container logs immediately after launching, so
the two obvious placements are both wrong: waiting *before* attaching blanks
the screen for the most informative minutes of the launch, and not waiting at
all is what the original `--timings` did — `wait_for_serve_ready` was reachable
only from `post_launch_lifecycle`, so for any recipe **without** post hooks the
tree reported distribution and container start while weight load and graph
capture scrolled past unrecorded. The watcher polls on a background thread and
reports through an `on_ready` callback; the CLI skips it when the recipe *has*
post hooks (that path already waited synchronously and would double-record
`serve.*`).

Four properties are load-bearing:

- **The live half is one line, the tree is not.** The callback fires on the
  watcher thread while `docker logs -f` writes to the same terminal, so it
  emits a single short line in one write. The multi-line table is printed by
  the CLI's finalize step, after the stream has stopped — which is also the
  fix for the original cosmetic bug, where the tree was printed and then
  immediately buried under `--tail 100` plus a live stream.
- **Cancelled is not failed.** `wait_for_port` / `wait_for_healthy` report
  cancellation and genuine failure with the same `False`, so
  `wait_for_endpoint_ready` checks the event before closing the span and
  returns `reason="cancelled"` with the span left **open** — rendered "did not
  finish" rather than an `error` claiming the endpoint never came up. Ctrl-C
  must not write a verdict about the cluster into a benchmark artifact.
- **A worker-thread span states its parent.** `parent=None` means "inherit
  from the open-span stack", which is a sequential-control-flow convenience
  and the wrong default off-thread: `Timeline.end` closes everything open
  *above* its target, so a main-thread `end()` would close the watcher's span
  too, stamp it with the main thread's status, and turn the watcher's own
  `end()` into a silent no-op — defeating the open-span property above *and*
  the ok/error verdict. `timing.ROOT` is the sentinel that lets a worker say
  "root" rather than depend on the stack happening to be empty. The same
  function nests under phase 6 when `post_launch_lifecycle` is the caller,
  which is why the parent is the caller's to state rather than a constant.
- **Cancellation reaches into the sleep.** Both waiters take a
  `cancel: threading.Event` and wait *on it* between polls
  (`health._interruptible_sleep`). At the readiness defaults the health stage
  sleeps 5s at a time up to 120 times, so a plain `time.sleep` would leave a
  cancelled watch polling over SSH for minutes after its caller exited — the
  same orphaned-work failure the session guard exists to prevent.
- **`serve.serving` closes the accounting.** A root-level sibling of `run`
  and the two readiness stages, opened when the endpoint answers and closed
  when the watch stops. Without it the tree stopped accounting at readiness
  while the total kept running — a launch watched for two hours rendered
  ~775s of rows under a 7695s total. It is **measured** (endpoint answered →
  we stopped watching), not the arithmetic gap between the rows and the
  total: a derived row would be invisible to `export()`, so the diagnostics
  record and the tree would disagree, and it would put an unmeasured number
  in the tree that `Timeline.add_span`'s refusal of `CLOCK_CONTROL` exists to
  keep out. The post-hook path opens its own (it waited synchronously and has
  no watcher), so both follow paths account the same.
- **The watch is observational.** It runs on every launch now, so it never
  touches the exit code: a model that outlasts the poll budget would otherwise
  start failing everything scripted around `sparkrun run`. It warns instead,
  and stays silent for `cancelled`.

**Timings are on by default.** `--no-timings` (hidden) suppresses the *table
only* — the readiness watch and its "endpoint ready" line still run, because
"the endpoint is up now" is worth having while logs scroll whether or not you
want a breakdown afterwards. `--collect-diagnostics` keeps the timeline for its
own record regardless.

**Sinks**: `run` (tree, on by default), `--collect-diagnostics` (`run_timeline`
NDJSON record — additive to that collector's own phases, which bracket a
strictly wider window including host diagnostics and `api.run`'s planning),
and `BenchmarkResult.generate_metadata()` under `metadata["timing"]` as
`serve_ready` + the full `launch` span list. That mapping is the Spark Arena
submission, so the existing `start`/`end`/`duration` keys are untouched. A
**resumed** run emits neither: its launch numbers came from a launch it did
not perform, and reporting them beside freshly-measured throughput
misattributes a stale launch — the same confusion `measured_at` exists to
prevent for the benchmark numbers themselves (issue #267).

### Launch Placement (`api.plan` → `api.run`)

A launch **decides once**. The launch path is split at the point where it stops
deciding and starts acting:

| Function | Does | Returns |
|----------|------|---------|
| `api.plan(options)` | resolve recipe/cluster/runtime, `prepare_transport`, one `api.status` sweep, **one** `api.schedule`, compose intent/token/cluster_id | `RunPlan` |
| `api.run(options, plan=…)` | `launch_inference` (evicting superseded deployments just before containers start), build result | `RunResult` |

**Eviction timing is load-bearing.** `_evict_superseded_deployments` tears down
a *serving* workload, so it runs from `launch_inference`'s `before_start` hook
— fired at phase 5, after image distribution, model download and tuning sync
have all succeeded — not at the top of `api.run`. Those steps take minutes and
are routinely interrupted; evicting up front meant a `sparkrun run` killed with
Ctrl-C during distribution left the cluster with *neither* the old deployment
nor the new one. `api.run` passes `before_start=None` under `--dry-run` (the
launcher guards too, so a dry run stays read-only regardless). The experimental
k8s path returns before `launch_inference` and so evicts explicitly, without
that guarantee.

`run(options)` with no plan plans internally, so the split is invisible to
callers that render nothing. It exists for the ones that don't: **anything that
shows the target hosts before launching must plan, render, then
`run(options, plan=plan)`.**

The alternative — schedule yourself, then pass the winners as `options.hosts` —
is what the CLI and the benchmark flow used to do, and it is a trap: `RunOptions.hosts`
is the *candidate* set, so narrowing it makes the display pass authoritative over
placement and leaves `run` unable to reach any host the display pass discarded.
That is how a `tp 2` launch on a 4-host `occupancy-sparse` cluster with 2 idle
hosts died with *"cluster has insufficient free capacity for 2 node(s)"*: the CLI
trimmed with the greedy default (it never forwarded the resolved scheduler), then
`api.run` re-placed with `occupancy-sparse` over only the busy pair it had been
handed.

`RunPlan` therefore keeps **both** host sets. They are not interchangeable:

- `candidate_hosts` — what placement could choose from. Feeds
  `derive_placement_token_from_hosts` (the deterministic/greedy cluster_id) and
  `_evict_superseded_deployments`, so that `stop` / `status` — which only know
  the cluster's full host list — derive the same cluster_id the launch used.
- `host_list` — what it chose. What actually runs.

`sparkrun run` previously scheduled three times (trim, a statusless
display-placement for the per-host VRAM fit table, then the launch) across two
SSH occupancy sweeps. Both are now one. The fit table renders `plan.placement`,
which is why it can no longer show every target as `[OK]` above the capacity
error that rejected them. Regression tests: `tests/test_api_plan.py`,
`tests/test_cli_run_single_placement.py`.

`api/_hosts.py:resolve_effective_hosts()` remains the single placement authority
underneath; `plan` is its one caller on the launch path.

**Which scheduler is in effect** is reported by
`core/scheduler.py:describe_effective_scheduler()` — the display peer of
`resolve_scheduler_selector` (that returns the *selector*, `None` when nothing
named one; this resolves it to the scheduler that would run, plus a `defaulted`
flag). Used by `cluster show`, `cluster inspect`, and the `run` banner so none
can disagree with the launch. A cluster predating the `scheduler` field stores
`None` and silently resolves to greedy; `cluster show` previously printed
nothing at all in that case, leaving no way to tell a greedy cluster from an
occupancy-aware one without launching something.

### `--ensure` ("is this workload already up?")

`--ensure` matches on the launch **intent** (see below) via
`api.find_running_intent(intent_id, hosts) -> IntentMatch | None`, never on a
cluster_id. A cluster_id encodes *placement* as well as intent, so the old
`derive_cluster_id(recipe, host_list)` lookup could only match a job the greedy
scheduler had put on exactly the host set being asked about — under an
`occupancy-*` scheduler (random placement token) it matched nothing, ever, and
`--ensure` launched a duplicate on every invocation.

- `core/cluster_status.py:workload_matches_intent()` is the shared predicate.
  Placement subtracts its own intent's workloads from the occupancy snapshot
  (`exclude_intent_id`) while `--ensure` looks for exactly those; the two must
  agree or `--ensure` would decline to launch something placement had already
  decided to replace.
- Pass the cluster's **full** host list, not a placement subset — a deployment
  that landed elsewhere still counts as running.
- A failed status probe means "not running" and the launch proceeds. Refusing
  to launch because we couldn't tell is the worse failure.
- The CLI queries before planning (skip → no scheduling at all); `api.run`
  honors `RunOptions.ensure` after planning and returns a `RunResult` with
  `already_running=True`, `launch_result=None`, describing the *pre-existing*
  deployment. Both go through `find_running_intent`.

### Per-Machine Container Images (`core/images.py`)

A recipe normally names one `container:` and every node runs it. A recipe
serving **pre-optimized, machine-tuned** builds instead declares `containers:`,
binding an image to a **hostname**:

```yaml
container: nvcr.io/nvidia/vllm:25.09      # fallback for unlisted machines
containers:
  - {host: spark-01, image: myorg/vllm-spark:node-01}
  - {host: spark-02, image: myorg/vllm-spark:node-02}
```

Keyed by hostname, not rank or node index: the image is a property of the
*machine*, so a rank-indexed map would be silently wrong the moment the
scheduler ordered hosts differently — and "silently wrong image" is the entire
failure mode this has to avoid. It is deliberately **not** spelled inside
`layout:`, whose `placements` every scheduler honors verbatim; putting the
image there would pin placement as a side effect of naming an image, wrong when
the tuned images exist on every machine and a `--tp 2` launch should still be
free to pick the two idlest. A recipe wanting both declares `layout:` as well.

`ImagePlan` keeps **two** maps because they answer different questions:

| Field | What it is | Consumers |
|-------|------------|-----------|
| `declared` | sorted `(host, image)` as the recipe wrote it | `generate_intent_id`, `derive_recipe_fingerprint` |
| `images_by_node` | image per node, aligned with the resolved host list | container launch, distribution derivation |

Hashing the *resolved* map would make the intent placement-dependent, so
`stop` / `logs` / `--ensure` would stop matching a workload the moment the
scheduler picked a different host subset. Both digests append their part only
when a `containers:` block exists, so every recipe predating the feature hashes
byte-identically (the same rule the fingerprint's `layout=` part could not
follow, which is why `containers=` sits outside the unconditional attr loop).

Because a wrong image on a tuned machine fails confusingly rather than loudly,
resolution is strict and every guard fires **before any side effect** — before
the builder runs, the image is pulled, or the model is synced:

- A `host:` not in the cluster raises; a duplicate `host:` raises; a selected
  host with neither an entry nor a `container:` fallback raises.
- A host falling back to `container:` is logged **by name at default
  verbosity** — on a machine-tuned cluster that is a material difference.
- **The runtime must opt in** (`RuntimePlugin.supports_heterogeneous_images`),
  fail-closed. Anything with a wire protocol between ranks breaks as a hang or
  a cryptic deserialization error rather than a clean failure: Ray needs one
  build across head and workers, MPI ranks must share an ABI. `sglang`,
  `vllm-distributed` and `llama-cpp` opt in. Note `vllm-ray` is a *sibling* of
  `vllm-distributed` (both are `VllmMixin` + `RuntimePlugin`), not a subclass,
  so it does not inherit the opt-in.
- **No image-transforming builder.** `prepare()` is single-valued so it cannot
  serve N images, and calling it once per image is not an option when a build
  is minutes long. `BuilderPlugin.transforms_image` separates those from
  *environment* builders (`uv-venv`) and `docker-pull`, which return the ref
  untouched and compose fine. Precedent: `Executor.needs_image`.
- `--image` clears the block entirely (`core/resolve.py`, logged). Anything
  subtler — override the fallback but keep the machine-specific images — would
  leave most nodes on the image the user just replaced.

Distribution is **derived** from the launch plan (`derive_container_entries`)
rather than declared beside it, so "what to ship" and "what to run" cannot
disagree. A hand-written `distribution_config.containers` still wins, which is
what `DistributionResourceConfig.explicit` exists to detect — the whole-config
`externally_provided` flag is too coarse, since a recipe customizing only
`models` still receives an auto-generated container entry. Two consequences in
`orchestration/distribution.py`:

- `_distribute_image_plan` fans out **across** images concurrently (each
  per-image call already fans out over its own targets under
  `resolve_parallel_cap`). A single image is dispatched inline so the
  overwhelmingly common case keeps its exact previous call shape and log order.
- `delegated` is redirected to per-node `pull` when the launch is
  heterogeneous: its head-pull-then-`docker save | ssh docker load` fan-out
  would make the head pull images it does not run and copy a machine-tuned
  image onto the wrong machine.

The ENTRYPOINT preflight probes once per **distinct** image, on a host that
actually runs it. The old single probe was correct only because distribution
had established every host carried the same one.

Job metadata records `effective_container_images` **alongside** — never instead
of — the scalar `effective_container_image` (the head's), because proxy
discovery, `logs` and the desktop sidecar all read the scalar. All three
`save_job_metadata` call sites must forward it: each rewrites the file
wholesale, so an omission is an erasure.

### Transfer mode `pull`

The fourth transfer mode. The other three route bytes through *somewhere* — the
control machine (`local` / `push`) or the head (`delegated`); `pull` routes them
through nobody: every node fetches from origin itself, concurrently. Faster than
a serialized head-pull-and-fan-out when per-node egress is good, and the only
correct strategy for heterogeneous images. Opt-in, because it costs N× egress
and needs registry / HF credentials reachable on every node.

- **A shared model cache overrides it.** With `prefs.skip_fan_out` the workers
  already mount the head's copy, and N nodes downloading into one NFS path
  concurrently is waste at best and a corrupted snapshot at worst.
- **`--rebuild` must reach the side that actually pulls**, which here is every
  node — `sync_image_to_hosts(force_pull=…)`. Its presence check is
  metadata-only, so an image re-pushed under the same tag is otherwise never
  refreshed.
- An *inferred* mode (auto → delegated) still falls back to a control-machine
  push for the hosts that failed; an explicitly-named `pull` is honored
  literally, the rule `delegated` already follows.

`models/distribute.py:_build_model_ensure_script` is shared by the head and
per-node paths — a second copy would be free to drift on GGUF handling or token
injection, and the drift would only surface on gated or quant-selected models.

### Plugin-Owned Recipe Items (`core/recipe_items.py`)

`register_recipe_item(key, handler, owner=…)` lets a cross-cutting plugin claim
a **top-level recipe key** exclusively, without `Recipe` learning its schema and
without hiding the settings in `metadata` (untyped, unvalidated, invisible to
the fingerprint). The core owns lifecycle only; the handler owns
`parse` / `validate` / `export`. Registration is idempotent for the same owner
and handler; a second owner raises, so nothing can silently reinterpret an
existing recipe surface.

Four properties are load-bearing:

- **Owned keys are excluded from the `runtime_config` sweep.** Unknown
  top-level keys are otherwise absorbed into `runtime_config`, which feeds the
  serve command — so without the exclusion a plugin's settings reach the engine
  as flags.
- **Items round-trip at the same top level** through `__getstate__`, registry
  caching and `to_yaml_dict`. A plugin key is recipe content; re-exporting it
  elsewhere would break the next load.
- **A raw item survives its plugin being unavailable.** `_plugin_item_raw`
  preserves it verbatim, so disabling a plugin never silently rewrites recipes.
- **Items participate in `derive_recipe_fingerprint`** via the handler's
  canonical export, appended only when present.

Note the deliberate contrast with `capabilities:` / `unsupported_capabilities:`,
which are core keys parsed as real attributes *specifically* to stay **out** of
the fingerprint: describing what a deployment can do must not change what it is.
A plugin item is the opposite — it changes how the workload is produced.

`SparkrunConfig.plugin_settings(name)` reads `plugins.<name>` for site-local
operational policy that does not belong in a portable recipe. `plugins.paths`
stays reserved for external plugin discovery. This is the *only* sanctioned
home for per-plugin config — a bespoke top-level block plus a property on
`SparkrunConfig` would put plugin-specific knowledge back in core.

### Recipe Execution Strategies (`core/execution.py`)

An integration that changes *how* a workload is brought up — not which image or
which model, but the act of starting it — would otherwise have to fork
`launch_inference` and lose distribution, placement, preflight and job
metadata. `RecipeExecutionStrategy` is the seam: a plugin's recipe item may opt
its recipes into **one** strategy, which supplies preparation steps and replaces
`runtime.run()` at the end. Everything else in the launcher still runs.

| Hook | Runs | Returns |
|------|------|---------|
| `preparation_steps(ctx)` | in `api.run`, before `launch_inference` | `PreparationStep`s |
| `finalize_preparation(ctx, receipts)` | after those steps | `PreparedExecution` |
| `prepare_activation(ctx)` | phase 5, assets resident, **before** eviction | opaque receipt |
| `activate(ctx, receipt)` | in place of `runtime.run()` | `ActivationResult` |

Six rules:

- **Recipe-local.** A strategy is reachable only through a top-level key its
  plugin owns, so *installing* a plugin never changes what `sparkrun run` does
  for a recipe that omits the key.
- **One strategy, or an error.** Two things claiming to launch the workload
  have no correct arbitration, so this is not a precedence rule.
- **The replacement barrier stays core-owned.** The launcher completes plugin
  preparation, normal image/model preparation *and* `prepare_activation` before
  firing `before_start`. A strategy never decides when the deployment it
  replaces is torn down — and by that point everything slow and interruptible
  is behind it, the same reason eviction moved into that hook (see Launch
  Placement above).
- **Preparation is a named DAG with compensation.** Globally unique step names,
  explicit `requires`, completed steps cleaned up in reverse on failure. Naming
  beats positional ordering because two plugins contributing steps share no
  list to order themselves within.
- **`LaunchAssetPolicy` declines, it does not replace.** A strategy opts out of
  builder / model / image distribution / entrypoint probe / tuning sync / page
  cache and may supply `images_by_node`; whatever it does not decline still
  runs, so it inherits the pipeline rather than reimplementing it.
- **`RunOptions.strategy_options` is not workload identity.** Per-invocation
  choices stay out of the fingerprint and intent id — the same rule that keeps
  serve flags out of `generate_intent_id`.

The activation path writes job metadata itself and must write **all** of it
(cluster, ssh_user, fingerprint, owner): `save_job_metadata` rewrites the file
wholesale, so an omission is an erasure whose symptom — a teardown that
authenticates as the wrong user and reports success — looks nothing like its
cause. The experimental k8s path returns before `launch_inference`, so it has
neither the asset pipeline nor the barrier; a strategy there is **refused**
rather than silently ignored.

### Launch Materialization (`api.materialize`)

`api.plan` stops at placement. `api.materialize(options, plan=…) ->
ResolvedLaunchSpec` resolves the rest — per-unit argv, env, mounts, devices and
the worker/rank topology — **read-only**: no image pull, no model distribution,
no cache creation, no network probing. It reuses a `RunPlan`'s placement rather
than re-deciding, so it cannot disagree with the launch it describes.

The shape separates three things a flat "one container per host" model
conflates, each of which silently produces a wrong launch:

- **Launch units vs. workers.** A unit is a container / process tree; a worker
  owns an accelerator. Scheduler ranks number *workers*, vLLM's `--node-rank`
  numbers *process trees*, and the two diverge as soon as a host owns more than
  one GPU. `generate_node_command` cannot express that (it takes one rank and
  infers the rest), so `generate_launch_unit_command` takes both namespaces
  explicitly; `_generate_parallel_command` is the shared renderer so the
  single- and multi-worker paths cannot drift.
- **Service domains.** A unit never crosses a service boundary, which permits
  several DP/PD service containers on one host.
- **Adapter topology.** Runtime-specific parallelism is opaque payload guarded
  by a canonical digest, rather than schema fields growing per runtime.

Two details: the command is a **bash argv** (`bash --noprofile --norc -c …`)
because sparkrun executes generated serve commands as scripts, so pipes,
redirects and expansions must survive the boundary; and executor mounts are
resolved and included, since a strategy creating its own containers would
otherwise simply not have what `ExecutorConfig.volumes` would have added. The
HF cache and pinned model inputs are marked read-only — a privileged controller
in the workload container must not leak ownership back into the shared cache.
`image_digest` is populated only from an already-pinned `name@sha256:` ref;
resolving one would need a remote probe, which is what "read-only" rules out.

### Workload identity — intent vs. fingerprint

Three different questions, two digests. Getting them confused is destructive,
because a matching intent is `api.run`'s licence to **destroy**:

| Question | Key | Consumers |
|----------|-----|-----------|
| Which served endpoint is this? | `generate_intent_id` | `stop` / `logs` / `status` / `--ensure` |
| May I replace that? | `generate_intent_id` | occupancy exclusion + eviction |
| Is this *exactly* this configuration? | `derive_recipe_fingerprint` | benchmark identity, provenance |

`generate_intent_id` hashes runtime + model + **container** + port +
served-model-name + non-default parallelism. The container image is in there
because the intent is the destroy key: two recipes serving the same model on
the same port through different images (a stable build and a nightly) are
workloads a user runs side by side, and while the image was excluded, launching
the second silently evicted the first (observed live on a 4-host cluster).
`--image` writes through to `recipe.container`, so overrides are covered.

**The served name has two resolutions, deliberately.** The supported spelling is
`defaults.served_model_name`, which every runtime reconciles into the rendered
command via `RuntimePlugin._augment_served_model_name`. A recipe may instead
hardcode `--served-model-name` in its `command:` template, which bypasses that
and is invisible to the config chain — so
`core/recipe.py:resolve_served_model_name` (declared → `command:` →
model id, via `extract_served_model_name_from_command`) is the shared last
resort for everything that needs the name for **display or routing**: the
benchmark's request target (this was issue #257 — llama-benchy asked for the
model id, HTTP 404, whole sweep dead), proxy-discovery metadata, and container
labels. `generate_intent_id` is the **non**-consumer: it still hashes only the
*declared* name, because widening it would change the intent id of every recipe
that hardcodes the flag and orphan already-running workloads from `stop` /
`logs` / `--ensure`, which recompute it. Precedent for parsing the template at
all: `kv_cache_dtype` (issue #248).

It stays narrow otherwise — serve arguments are **not** hashed — because
`stop` / `logs` / `--ensure` recompute it from the recipe without the flags the
user typed at launch. Three reasons not to widen it to the fingerprint:

1. Discovery breaks: `run r --gpu-mem 0.9` then `stop r` would not match.
2. Eviction stops working for the common relaunch (tweak one flag → a
   different intent → the old deployment is never replaced and keeps the GPUs).
3. The fingerprint is not stable across the launch boundary:
   `launch_inference` calls `apply_platform_runtime_flag_defaults`, which
   `setdefault`s platform flags into `recipe.defaults` keyed off the **head
   host's** hardware — so the config chain it hashes differs before vs. after
   launch, and is placement-dependent. Fine as a provenance digest derived once
   and threaded down; unusable as a lookup key.

### Status Discovery ("what's running where?")

All workload-status discovery flows through **one source**, `api.status`, in two
tiers:

- **`api.status(hosts, cluster=…) -> ClusterStatus`** — the lean *occupancy*
  snapshot (per-host `used_slots`/`free_slots`/`workloads`, `errors`). Consumed
  by the occupancy schedulers, `api/_hosts.py` placement, proxy discovery, and
  `api/_stop.py` teardown (intent→cluster_id). Data-only shape in
  `core/cluster_status.py`.
- **`api.status_report(hosts, cluster=…, cache_dir=…) -> ClusterStatusResult`** —
  the *display* tier: `classify_cluster_status(status(...))` shapes the snapshot
  into groups/solo/idle/pending + cached job-metadata enrichment (the CLI-facing
  aggregate in `core/cluster_manager.py`). Used by `cluster status` and `stop
  --all`.

**Free ≠ idle.** A host that a pending model download or image distribution is
staging onto is minutes from taking its whole GPU, so the display tier splits
the reachable/zero-container hosts into `idle_hosts` and `preparing_hosts`
using the pending-op locks (`core/pending_ops.py`), which have always recorded
their `hosts` — the report simply discarded them and reported the targets as
idle. Attribution is literal (case + `user@` normalized, no DNS): an op that
matches nothing in the queried host list is dropped as another cluster's, while
one that recorded *no* hosts is still shown but pinned to none — "unknown
scope" must never read as "affects every host". Each op carries derived
`matched_hosts` / `other_hosts`. This stays **display-only**: pending locks are
control-node-local and best-effort, so `preparing_hosts` is a lower bound, and
feeding it into the occupancy snapshot would let a stale lock refuse a launch
onto a genuinely free host. Locks also carry `job_cluster_id` / `cluster`, so a
pending op can name the job it is preparing (its lock *key* remains the
image/model/host hash — distribution runs before the launch commits).

**Which cluster is being reported** is `cli/_common.py:resolve_host_context()`
→ `HostContext`. `resolve_hosts` consults the default cluster, so a flagless
invocation ends up with a concrete host list; handing that to `api.status_report`
with `cluster=None` made `resolve_cluster` short-circuit to an *anonymous*
`ClusterDefinition` (`api/_resolve.py`: explicit hosts are checked **before**
the default-cluster step) and silently dropped that cluster's executor pin,
`executor_config`, `hosts_hardware` and transport from the sweep. Any command
that pairs a resolved host list with an `api.*` cluster argument must forward
`HostContext.cluster_name`; `HostContext.describe()` is the matching banner, so
the report also *says* what it covers.

Under the hood, `api.status` calls
`orchestration/executor.py:query_status_for_cluster(cluster, hosts, …)`, which
**sweeps every enabled executor on the cluster's status substrate and merges**:

- **`Executor.status_scope`** (ClassVar, default `"host"`) is the substrate an
  executor's `query_status` inspects. Executors sharing a scope inspect
  *disjoint* state on the *same* substrate (docker containers vs `local`
  pidfiles on the SSH hosts) and are merged (`ClusterStatus.merge` — N-way
  fold, first snapshot authoritative). A provider executor declares its own
  scope (`k8s`, `modal`) and is queried alone.
- The **cluster's scope** = `status_scope` of the executor it would launch with
  (`resolve_executor_name`, i.e. explicit override → cluster pin →
  config/default). So an SSH/Thunder cluster → `"host"` (docker + local); a
  Modal cluster → `"modal"`; a k8s cluster → `"k8s"`. The scope's default
  executor is queried first (wins per-`cluster_id` collisions); a single failing
  executor is skipped; an unresolvable executor (gated-off provider plugin)
  degrades to an empty snapshot rather than raising.

Each `Executor.query_status(hosts, …)` inspects its own backend (docker `docker
ps`, local pidfile scan, k8s/modal control plane) and returns a `ClusterStatus`.
There is no separate status extension point.

**Post-mortem** (`Executor.describe_terminated(sources, …) -> {(host, container):
TerminationInfo}`) is the *dead* peer of `query_status`: that reports what is
running; this reports, for something that is **not**, whether its remains are
still on the substrate and how the operator inspects them. `query_status` runs
`docker ps` (running only), so it structurally cannot make that distinction —
which is why `api.logs`'s liveness precheck needs a second question rather than
a wider answer to the first.

Every part of the answer is substrate-specific, including the remediation
wording: `docker logs` is wrong advice on a k8s cluster and meaningless for a
`local` job (a native process whose output is a host logfile). So
`TerminationInfo` carries `investigate_hints` supplied by the executor;
`api/_logs.py` lays them out and never authors them. Keyed by `(host,
container)` because a container name is unique only per host — Ray worker
containers share one name across nodes.

Three rules, all of which exist because a `False` verdict is what **deletes**
cached job metadata:

- `exists=None` / an absent entry means *cannot tell* (unreachable host, no
  container engine, an executor with no post-mortem support) and must never be
  read as "gone". The base-class default is `{}`, so an unimplemented executor
  degrades to preserving metadata.
- Best-effort throughout: it never raises, exactly like `query_status` and the
  `verify_*` preflights.
- **`--rm` is why this is not a plain existence check.** `auto_remove` defaults
  to `True`, so the daemon deletes a container the moment it exits: absence is
  the *normal* outcome of a crash, not evidence a workload never ran. The Docker
  executor knows its own config, so it reports *which* it was and hints at `-o
  auto_remove=false` for the next attempt, instead of reporting the most
  interesting failure as stale bookkeeping.

### Job Metadata Lifecycle ("what have I launched?")

`~/.cache/sparkrun/jobs/<digest>.yaml` is written by every launch and read by
`stop` / `logs` / proxy discovery / `--ensure`. It is **append-only in
practice**: only an explicit `stop` (or the `logs` staleness path) removes an
entry, and a job that crashed under `auto_remove` is gone from docker before
anything asks about it. Left alone this reaches hundreds of dead entries
against a couple of dozen live intents.

**A job records how to *reach* it, not just where it ran.** `hosts` answers
"where"; `cluster` + `ssh_user` answer "as what". Without them, `stop <cluster_id>`
— which recovers its hosts from this file and so names no cluster — resolved to
an *anonymous* `ClusterDefinition`, dropping the cluster's SSH user, executor
pin, `executor_config` and transport, and SSH-ing as the control node's own
login. On a cluster whose `user:` differs from that login, every teardown
connection was refused while `stop` still printed a success line and the
workload kept serving (issue #277). `api._resolve.resolve_cluster_for_job` is
the read side, in strict order: an explicitly named cluster → the one the job
recorded → the recorded `ssh_user` alone.

Three properties are load-bearing:

- The recorded cluster supplies **connection identity only**; hosts stay the
  job's own, or the user's `--hosts`. A load-aware scheduler may have placed the
  workload on a subset, and widening teardown back to the cluster's full host
  list is not a fix, it is a different command.
- `ssh_user` is recorded **separately** from the cluster name, so a job that
  outlives its cluster (renamed, deleted, read on another control node) keeps
  the part that decides whether the hosts are reachable at all. It only ever
  *fills* a gap — a resolved cluster's own `user` wins, since that is current
  configuration while the recorded value is history.
- Both are **omitted when unknown**, never written empty: the read side has to
  tell "anonymous `--hosts` launch" from "written before sparkrun recorded
  this", and an empty `ssh_user` would be applied as a real username.

All three `save_job_metadata` call sites in `launch_inference` must forward
them, because each rewrites the file wholesale — a site that forgot would erase
them a phase later, and the symptom (a teardown that cannot authenticate) looks
nothing like the cause.

Three more pieces keep the cache usable:

- **`started_at`** is stamped at launch. The read side (`api.list_jobs`) always
  looked for it but nothing wrote it, so every job was untimed and the
  documented "most recent first" ordering silently degraded to alphabetical by
  hex digest. `_resolve_started_at` backfills from file mtime for entries
  predating the field — on any existing cache that is nearly all of them, so
  without the backfill the fix would do nothing for anyone.
- **`list_jobs(limit=N)`** pre-ranks by mtime (a stat per file) and parses only
  the top N. Each file embeds a full serialized recipe state, so an unpruned
  cache is ~1.7 MB and ~1.5 s to load — fine for a report, far too slow for
  shell completion, which runs on every TAB.
- **`prune_job_metadata`** keeps an entry only when it is *both* among the
  newest `keep_per_intent` (3) for its `intent_id` *and* younger than
  `max_age_days` (30). The per-intent window is deliberately not a global
  count: "keep newest N overall" would erase every trace of a rarely-run
  workload. Nothing in `protected_cluster_ids` is ever deleted.

**Pruning runs from `run`, and only from `run`,** because that is the one
command that both grows the cache and already holds a live occupancy snapshot —
the eviction sweep's, which `_evict_superseded_deployments` now returns
alongside the evictions. Age alone is *not* a sufficient guard: a long-lived
server easily outlives the cutoff, and deleting its metadata strands the
deployment. Where that snapshot is absent (`--dry-run`, or a failed status
query) the prune is skipped entirely — `None` vs. an empty set is the
difference between "couldn't look" and "looked, nothing there". Opt out with
`jobs.autoprune: false`; sweep manually with the advanced-only
`sparkrun setup prune-job-metadata-cache` (which has no snapshot, and says so).

**`running.json`** is the completion half. Completion filters to what is
*actually running*, which means a live `api.status` sweep — a deliberate
choice: a list of dozens of dead hex digests is not worth having. The cost is
managed rather than avoided:

- **Cache first.** A recorded snapshot is reused for `COMPLETION_CACHE_TTL_S`
  (60s), so a burst of TABs costs one sweep, not one per keystroke. Measured:
  1.15s cold, 0.08s warm. The TTL is far shorter than the file's own
  `RUNNING_SNAPSHOT_MAX_AGE_S` (600s) because the point here is to make a burst
  cheap — the longer window would keep offering a workload for ten minutes
  after it was stopped. The long window is still honoured as a *fallback* when
  a sweep fails: stale beats nothing.
- **A snapshot is only reused when it covers the target's hosts.** One left by
  a sweep of some other cluster says nothing about this one, and treating its
  hosts as unobserved would put every dead job back in the list.
- **Hard-bounded.** `COMPLETION_STATUS_TIMEOUT_S` (5s) is a per-host subprocess
  timeout with the hosts swept in parallel, so an unreachable host costs that
  ceiling once instead of hanging the shell. `completion.status_timeout_s: 0`
  disables the sweep entirely for anyone on a flaky link.
- `api.status` records every sweep it performs, so the sweep completion
  triggers is what makes the next TAB instant — and any other command's sweep
  primes it too.

The recorded *hosts* matter as much as the cluster_ids. A sweep is frequently
partial (placement queries a candidate subset), and without knowing what was
covered a reader cannot tell "not running" from "not looked at" — it would hide
live workloads on unswept hosts. So an unswept host's workloads are *unknown*
and kept — **except** when the job lies entirely outside the target cluster.
Those are the leftovers of clusters that no longer exist (a torn-down cloud
instance keeps its jobs forever and its hostnames stop resolving, so they can
never be verified); verifying them would mean sweeping every cluster any recent
job touched, paying a full connect timeout for each dead one. Naming that
cluster sweeps it. A stale or absent snapshot still means show everything.

**Completion targets** (`cli/_common.py:_complete_targets`) offer **recipe
names first, cluster_ids second**, because on bash there is no way to annotate
a completion — `BashComplete.format_completion` emits `type,value` and drops
the help text entirely (zsh and fish render it). The value has to carry the
meaning, and a recipe name does while a hex digest does not. The split is
forced by how each form resolves: `logs <recipe>` goes through
`resolve_cluster`, so recipe names are scoped to the cluster the invocation
targets; `logs <cluster_id>` reads its hosts from the metadata and so stays
valid regardless.

The **target** is the `--cluster` / `--hosts` already on the command line
(completion runs after Click has parsed the options it has seen), else whatever
`resolve_cluster()` returns with no arguments. It is never *guessed* — inferring
one from, say, the most recent job would point completion's SSH sweep at a
cluster the user never named. With no target nothing can be confirmed dead, so
everything cached is offered.

That makes `resolve_cluster`'s chain load-bearing here, and it was missing a
step. **The default cluster** (`sparkrun cluster set-default`, stored in a
marker file the `ClusterManager` owns rather than in `config.yaml`) is now
consulted between `hosts_input` and `config.default_hosts` — the ordering
`core/hosts.py:resolve_hosts` has always used. The two resolvers disagreed, so
a user whose only host source was a default cluster got `HostsUnreachable` from
every `api.*` entry point that resolves without an explicit cluster. It returns
the whole definition rather than just the hosts, so the cluster's SSH user /
executor / scheduler come with it; a dangling default (naming a deleted
cluster) falls through to the next source instead of raising.

**Mount-source preflight** (`Executor.verify_mount_sources(paths, hosts, …)`) is
the substrate peer of `query_status` on the *write* path: "do these identity-mount
sources already exist where the workload will run?" It validates pre-placed model
weights (an absolute-path `model:` or `cluster_config.resolved_model_path`, which
skip download+distribution) *before* the launch commits to that skip. Host-substrate
executors (docker/local) override it to SSH `test -e` the hosts via the shared
`ssh.verify_host_paths` helper; provider executors (k8s/modal) probe their own
volumes; the base default is a safe no-op (`{}`). Wired at the launch choke point
by `launcher._verify_pre_placed_model` (skipped on `--dry-run`), which raises a
`RecipeError` listing host→missing-path gaps. Best-effort like `query_status`: an
unresolvable executor or unreachable host degrades to "couldn't verify" and never
blocks — only a *confirmed*-missing path fails the launch. This is why an
absolute-path model works from a **remote control machine** that isn't a cluster
member: the check runs on the *targets*, not the control node.

**ENTRYPOINT preflight** (`Executor.verify_command_passthrough(image, hosts, …)`)
is the second write-path preflight: "will the command I append actually *run* on
this image?" sparkrun always emits its launcher as CMD **arguments** (`docker run
<image> bash -c <b64 cmd>`), so the image's ENTRYPOINT decides their fate — and
the two idioms in wide use are indistinguishable by `docker image inspect`:
a **passthrough** wrapper (`/opt/nvidia/nvidia_entrypoint.sh` → `exec "$@"`,
inherited by nearly every NGC image, so this is the *common* case) versus a
**consuming** one (`ENTRYPOINT ["vllm","serve"]`, which parses sparkrun's
`bash -c …` as its own flags and never starts the workload). Warning on "non-empty
ENTRYPOINT" would fire on every working image; auto-clearing would strip the NGC
wrapper's setup from all of them.

So the verdict is established *empirically* (`containers/entrypoint.py` +
`scripts/image_entrypoint_probe.sh`): no ENTRYPOINT → `absent`, no container
started; else run the real argv shape and look for a **computed** sentinel on
stdout (computed because a consuming entrypoint typically echoes the argv it
rejected — an echo can reproduce a literal token, never the evaluated one) →
`pass`; else re-run with `--entrypoint ''` and only call it `fail` if *that*
succeeds, so a stale CDI spec / absent GPU / missing bash degrades to `unknown`
rather than a bogus verdict. That second run is also what lets the error claim
the fix is *verified*.

Wired via `distribute_from_config`'s `after_container_sync` hook — the image is
resident everywhere but the long, interruptible model sync hasn't started — and
`launcher._verify_image_command_passthrough` raises a `RecipeError` naming both
fixes (`executor_config: {entrypoint: ""}` or `-o entrypoint=''`) rather than
auto-clearing: the probe proves clearing *works*, not that it's *harmless*.
`DockerExecutor` skips the probe entirely when `config.entrypoint is not None` —
otherwise it would reject the very fix it recommends. One host is probed (the
verdict is a property of the image; distribution already established it matches
everywhere), ~0.7s for the passthrough case. Fail-open throughout; kill switch
`SPARKRUN_NO_IMAGE_PROBE=1`. `entrypoint` is in `cli/_run.py`'s
`_EXECUTOR_OVERRIDE_KEYS`, which is what makes `-o entrypoint=''` reach
`ExecutorConfig` instead of the serve command.

### Live monitoring (telemetry + occupancy)

Monitoring has a second axis alongside occupancy — **telemetry** (per-host/node
util/mem/temp) — abstracted the same way status is, by **substrate scope**:

- **`TelemetryProvider`** (`orchestration/telemetry/`, SAF ext point
  `EXT_TELEMETRY`) is the telemetry peer of `Executor.query_status`, selected by
  `scope` (matches `status_scope`). Telemetry is a substrate property, not
  per-executor — docker and local share one host source — so there is one
  provider per scope. Core ships `HostTelemetryProvider` (scope `"host"`,
  wrapping `ClusterMonitor` / `NvMonitorClusterMonitor`); k8s/modal providers
  ship in their plugins. `get_telemetry_provider(scope)` returns the stateless
  singleton (or `None` → occupancy-only monitoring). A live collection's state
  lives on the `TelemetrySession` from `provider.open(...)`.
- **`api.open_telemetry`** — raw telemetry session for a cluster's scope.
- **`api.open_live_monitor` / `api.live_monitor`** — compose the telemetry
  stream with a background `api.status` occupancy poll into
  `MonitorFrame` snapshots (per-host `HostActivity` = telemetry + the workloads
  occupying the host). Substrate-agnostic: a host cluster combines
  `host_monitor.sh` + docker/local occupancy, a k8s cluster would combine k8s
  metrics + k8s occupancy — clients see the same shape.

The `cluster monitor` TUI drives a `LiveMonitorSession`, so its Jobs column and
detail pane show **all** executors' workloads (docker + local + provider), not
just the docker containers `host_monitor.sh`'s embedded `docker ps` used to
report. The `--simple` / `--json` telemetry-only paths still use `ClusterMonitor`
directly.

### Transport Layer (`transports/`)

The **transport** is the connectivity seam: *how sparkrun reaches / prepares a
cluster's hosts* before the generic SSH machinery runs. It is orthogonal to the
**executor** (`orchestration/executors/`, *how the workload runs on the host*) —
a provider-transport cluster still uses the docker executor.

- **`base.py`** — `Transport` SAF `Plugin` (selector `transport_name`, extension point `EXT_TRANSPORT`) with `prepare(cluster, *, dry_run=…)` (default no-op) and its delete-time counterpart `cleanup_cluster(cluster, *, dry_run=…)` (release out-of-band state — ssh alias/key — on cluster delete). A transport self-gates via `required_feature_flag` (like `Executor`). Discovered via `find_types_in_modules("sparkrun.transports", Transport)` in `core.bootstrap` — mirrors `Executor`.
- **`ssh.py`** — `SshTransport` (`transport_name = "ssh"`), the default; `prepare()` is a no-op, so every existing cluster is byte-identical to before transports existed.
- **`session.py`** — `HostSession` / `SshHostSession`, the *runtime* peer of `prepare`. `prepare` makes a host reachable; a session executes **exact argv** on it (plus `upload` and `docker_registry` against the host's own daemon). Everything else sparkrun runs remotely is a generated *script* piped to `bash -s`, which is wrong for a managed binary invoked with structured arguments — re-quoting through a generated shell script is a correctness hazard there, not a convenience. Local dispatch bypasses the shell entirely so the two paths cannot disagree about quoting; a non-zero command is *data* (only an unusable session raises); and `close()` is the cancellation handle, killing in-flight process groups so an interrupted caller does not orphan remote work. `Transport.open_host_session` defaults to `SshHostSession`, so every existing cluster has one without declaring anything.
- **`__init__.py`** — SAF-backed resolution: `resolve_transport(name)` / `list_transports()` query `get_extensions(EXT_TRANSPORT)` by `transport_name` (returning the stateless SAF singleton). `prepare_cluster_transport(cluster)` (run/status/logs/stop), `open_cluster_host_session(cluster)` (executable session, gated the same way) and `cleanup_cluster_transport(cluster)` (delete) are the single call-site helpers. `_require_transport_enabled` reads the resolved transport's `required_feature_flag` and fails closed at the `prepare` call site (never a silent SSH downgrade, never SAF `is_multi_extension` hiding) — a gated selector yields a clear "enable it with …" error rather than "unknown transport". `cleanup_cluster_transport` is deliberately **ungated** (teardown must succeed even if the flag was later disabled) and tolerant of an absent transport plugin.
- **Thunder Compute** (`transport: thunder`) is **no longer in core** — it was externalized to the out-of-tree `sparkrun_thunder` plugin (the reference example for the plugin system). It registers `ThunderTransport` (SAF), its own `transports.thunder` feature flag, and the `sparkrun cluster import thunder` command (via `register_cli_command`). Core keeps only the generic seam; a `transport: thunder` cluster fails closed unless the plugin is loaded (`core.external_plugins` + `transports.thunder`).

`ClusterDefinition.transport: str = "ssh"` + `provider_ref` select the transport
(serialized only when non-default). The single wiring is
`api/_resolve.py:prepare_transport(cluster_def)` — called by `api.run` / `api.status`
/ `api.logs` / `api.stop` right after `resolve_cluster`, before any SSH — which
translates `TransportError` → `SparkrunError` for clean CLI errors. **Layering:
`cli/api → transports → {core, orchestration}`; `orchestration` never imports
`transports`.** `sparkrun cluster import thunder` attaches one single-host cluster
per RUNNING instance (attach-only; multi-node out of scope, multi-GPU-per-node
handled by the probe).

### Tailscale Endpoint Publishing (`setup tailscale`)

`sparkrun setup tailscale` (gated behind `cli.setup.tailscale`, off by default) joins cluster
hosts to a **tailnet** and surfaces the inference HTTP endpoint to the rest of the user's network.
It is a **control-plane / endpoint-publishing** feature — orthogonal to the transport seam and NOT
a data-plane path (NCCL stays on InfiniBand/CX7). Auth is via a Tailscale **OAuth client** that mints
a short-lived, pre-authorized, **tagged** auth key per join batch (never a long-lived key in config);
exposure is a **raw tailnet port** (`http://<ip>:<port>/v1`), not `tailscale serve`.

3-layer split mirroring `setup k8s`: `cli/_setup/_tailscale.py` (thin Click group, self-gates like
`_k8s.py`) → `api/tailscale/` (console-free `join`/`status`/`expose`/`down` + dataclasses + errors)
→ `orchestration/tailscale/` (`api.py` stdlib OAuth + key-mint + device REST client, `scripts.py` +
`scripts/tailscale_join*.sh` join scripts driven through `run_with_sudo_fallback`, `local.py` for
control-machine `tailscale ip` probes used by `expose --proxy`). Layering: `cli → api.tailscale →
orchestration.tailscale → {orchestration.ssh/sudo, core.config}`. Design spec: `.slop/tailscale-setup.md`.

### Inference Gateway (`proxy/` + `api/proxy/`)

The **gateway** is the process fronting every discovered inference endpoint
behind one OpenAI-compatible API. Core ships one implementation — `ProxyEngine`
(LiteLLM) — and everything a second needs is in place, including for one living
outside the `sparkrun.proxy` tree entirely. One word throughout: **gateway** is
the pluggable family, `proxy` is the user-facing command.

Three mechanisms, deliberately separate:

- **Registration** — `proxy/gateway.py:register_gateway(name, feature_flag=,
  loader=)`; `gateway_class(name)` is the one place a name becomes an
  implementation. The registry is **in-process** rather than a SAF extension
  point because an engine is *constructed with arguments* (host, port, master
  key, state dir) rather than resolved as a stateless singleton — the same
  reason `platforms` and `models/kv` stayed in-process. Registration carries a
  **loader**, not the class, so `proxy.engine` can import this module without a
  cycle and registering costs nothing at import time. Idempotent by name, which
  is what lets an out-of-tree plugin substitute an in-tree implementation.
  litellm registers in core, not from a plugin: `proxy` must resolve to
  *something* with every plugin absent.
- **Availability** — `gateway.<name>` feature flag. `gateway.litellm` ships
  **enabled on every channel** (`default=True`, like `executor.docker`); a
  plugin-contributed gateway would ship off.
- **Selection** — exactly one gateway is used at a time, arbitrated in
  `resolve_gateway()`: an explicit name (`proxy.gateway:` in `proxy.yaml`, or
  `--gateway`) must be known *and* enabled; with no name, the default wins when
  enabled, else the single remaining enabled gateway, else
  `AmbiguousGatewayError`. Enabling a second gateway's flag does **not** switch
  to it. The flag registry has **no** notion of mutually-exclusive flags —
  nothing stops a user enabling two, so resolution refuses to guess (mirrors
  `_default_executor_name`).

"Unregistered" and "disabled" are deliberately distinct errors: a name can be
known to the flag registry while its plugin failed to load, and telling that
user to enable a flag that is already on is a dead end.

**`GatewaySupervisor` (`proxy/_supervisor.py`)** holds everything a gateway
does *as a local process* — spawn, startup grace, SIGTERM/reap/zombie
detection, the `state.yaml` format, 0600/0700 permissions, the auto-discover
sidecar, the insecure-bind warning. `ProxyEngine` subclasses it, so LiteLLM's
argv, environment and config format are all that remain gateway-specific. Two
implementations writing that state format independently would eventually
disagree about it, and the symptom is a proxy nobody can stop.
`GatewayState` is the read-only half, split out so a caller that only needs
"what is running" (the auto-discover daemon, a status probe) doesn't construct
a supervisor whose `gateway_name` is blank.

Capabilities are **declared** rather than branched on by name, which is what
keeps `gateway_class` the only place a name resolves to an implementation:

| Attribute | Default | Why it exists |
|-----------|---------|---------------|
| `supports_autodiscover` | `True` | A gateway owning its own desired state would fight sparkrun's daemon. `start()` warns and disables rather than silently dropping a configured setting. |
| `wants_proxy_config` | `False` | Management paths resolve their engine from the *state file*. A config-driven gateway without `proxy.yaml` computes an **empty** desired state, so `proxy alias add` would delete every deployment it wasn't told about. |
| `data_plane_authenticated` | `False` | The safe assumption. A gateway that authenticates says so, rather than every gateway being trusted to have opted out of the warning by accident. |
| `model_query_error` | `""` | Empty means the query succeeded, *including* an empty model list — collapsing the two reports an authenticated management failure as "no models registered". |

`prepare_config(endpoints, aliases, write=)` puts config generation on the
engine: what a gateway's config *is* — a rendering of discovered endpoints, or
a list of desired bindings — is the implementation's business. `write=False`
is not a convenience; a dry run reports which aliases *would* apply, and
answering that from the same code that renders the real config is what keeps
the preview honest.

The model-management surface (`sync_models`, `sync_aliases`,
`list_models_via_api`, `register_loaded_model`, `unregister_loaded_model`)
lives on the base **because `api.proxy` resolves an engine from the state
file** — a LiteLLM-only method reached against another running gateway was an
`AttributeError` far from its cause. The first three raise
`NotImplementedError` naming the gateway; the last two return `None`, meaning
"discovery-driven, do the ordinary endpoint sync", which makes them a true
no-op seam for LiteLLM while giving `proxy load` / `unload` somewhere to hook.

**Gate placement**: `ProxyEngine.start()` is the *one* enforcement point —
bringing a gateway up, checked before `--dry-run` so a dry run can't advertise
a start that would be refused. `stop` / `status` / `models` / `sync` /
`alias_*` and the auto-discover daemon's `_restart_proxy` path are **ungated**:
a proxy started while the flag was on must stay manageable (and stoppable)
after it is turned off, and the daemon keeps driving the engine it was started
with. Same rule `cleanup_cluster_transport` follows for transports. That rule
extends to a *missing* implementation: when the state file names a gateway
whose plugin is not loaded, `_running_engine` falls back to a bare
`GatewaySupervisor` bound to the recorded name rather than raising — state
reading and SIGTERM are gateway-independent, and raising would strand a live
process that nothing could then describe or kill.

**The auto-discover daemon is gateway-neutral.** It is handed `{gateway,
state_dir, interval, removal_grace_sweeps}` and reconciles through
`api.proxy.sync`, which re-resolves the implementation from the state file on
every sweep — so a `proxy start --restart` that swaps gateways is followed
rather than fought. It therefore no longer carries the master key: the
credential belongs to whichever engine the state file names. A previously
healthy endpoint survives `discover_removal_grace_sweeps - 1` consecutive
misses (default 2, `proxy.discover_removal_grace_sweeps`, `1` restores
remove-on-first-miss) — one timed-out probe is not evidence a workload is gone,
and evicting it costs a restart plus a window of 404s for a model that is
serving fine. Identity is `cluster_id` when present, since an address is not
stable across a relaunch.

**`proxy.yaml` has two writers** — the daemon and any `sparkrun proxy` command
— so `ProxyConfig.save()` locks a stable sidecar, re-reads, and merges only the
sections *that instance* modified; a whole-document save silently discards the
other's change. An alias removal is recorded as an explicit deletion, because
"not in my copy" is not an instruction to delete a document this instance never
saw. `fcntl` is imported **guarded**: `proxy/config.py` is reached from
`SparkrunContext`, i.e. essentially every invocation, and a hard import would
make sparkrun unimportable on a Windows control node. Without it the lock
degrades to none (`os.replace` is still atomic, so the file is never
half-written) — strictly weaker, and correct only because the daemon is a
POSIX-only fork path.

`api/proxy/` is the console-free facade (mirrors `api/tailscale/`): `start`,
`stop`, `status`, `models`, `sync`, `register_loaded_model` /
`unregister_loaded_model`, `add_alias` / `remove_alias` / `list_aliases`, plus
`resolve_gateway` / `list_gateways`. `cli/_proxy.py` is a renderer over it, and
the desktop sidecar calls it directly. `ProxyUnsupported` is the "this gateway
has no such capability" *answer*, distinct from a failure. `ProxyUpdateFailed`
wraps `GatewayOperationError` (of which `ProxyRestartError` is the LiteLLM
member) — a base every gateway's management failures derive from, so `sync`
translates one exception type instead of catching bare `RuntimeError` and
reporting an unrelated engine bug as a routine update failure. The state file
records `gateway`, so management paths bind to *what is running* rather than to
what is currently configured. Layering: `cli → api.proxy → sparkrun.proxy →
{core, orchestration}`; `sparkrun.proxy` imports of `api` stay deferred
(`proxy.discovery` imports `sparkrun.api`, so module-level would be circular).

Two declarative seams a gateway consumes, both deliberately **outside** the
workload's identity (`derive_recipe_fingerprint`) — describing what a
deployment can do must not change what it *is*, or declaring a capability on a
running deployment would force it to be re-admitted:

- `RuntimePlugin.native_protocols(recipe)` — API dialects served natively, most
  preferred first. **Fail-closed** (base returns `["openai"]`): a protocol
  selects the upstream URL, headers, parser, streaming framing, error
  vocabulary and retry classification, so under-claiming costs a translation
  while over-claiming sends wrong-shaped bytes to a server that cannot parse
  them. A runtime that gained a dialect at some version must gate on the
  recipe's resolved container tag, not on the runtime name.
- Recipe `capabilities:` / `unsupported_capabilities:` — declared in
  `_KNOWN_KEYS` and parsed as real attributes precisely so they stay out of
  `runtime_config` (which would put them in the fingerprint *and* the serve
  command). sparkrun never infers them; nothing in a recipe reveals them.

`RunOptions.owner` tags the component that launched a workload (persisted to
job metadata, omitted when unset) so an automated supervisor can tell its own
jobs from identically-configured ones a human started, and refuse to tear the
latter down. `RunPlan.recipe_fingerprint` is derived in `api.plan` and threaded
through `launch_inference` to **all three** `save_job_metadata` call sites —
each rewrites the file wholesale, so a site that forgot would erase it a phase
later. It must be pre-launch: `apply_platform_runtime_flag_defaults` mutates
`recipe.defaults` keyed off the head host's hardware *before* metadata is
saved, so a digest taken inside the launcher is placement-dependent and the
caller that later matches on it can never reproduce the value.

### SSH Access Bootstrap (`api/setup/`)

Every setup phase — CX7 detection, shared-cache detection, the mesh itself —
assumes passwordless control→host SSH. `api/setup/` is the console-free layer
that *establishes* that, and it is the first piece of the wizard written to be
GUI-drivable (the desktop app's sidecar can call it directly instead of
shelling out to a terminal wizard).

| Function                          | Purpose                                                                              |
|-----------------------------------|--------------------------------------------------------------------------------------|
| `probe_ssh_access`                | `BatchMode=yes` reachability sweep → `SshProbe` per host                              |
| `ensure_local_key`                | Find an existing identity, else generate `~/.ssh/sparkrun_ed25519`                    |
| `install_public_key_interactive`  | Install the pubkey on one host via password auth                                      |
| `mesh_ssh_keys_native`            | host↔host key mesh with **no local shell**                                            |

**Everything here runs on a bare Windows control machine.** The only external
binaries are `ssh` and `ssh-keygen` (Windows 10+ ships both); there is no
`bash`, no paramiko. Two design rules follow from that and should not be
relaxed:

- **Key material travels over stdin, never argv.** Scripts are generated with
  the key embedded (single-quoted, or a quoted heredoc for the mesh) and piped
  to `bash -s`, so nothing depends on the *local* platform's command-line
  quoting — the difference between working and mangling a key on Windows.
- **`probe_ssh_access` adds `StrictHostKeyChecking=accept-new`.** `build_ssh_cmd`
  already forces `BatchMode=yes`; with the stock `ask` policy a first-contact
  host fails host-key verification and would be misreported as *unreachable*
  rather than merely unknown.

`SshProbe` distinguishes **auth failure** (host answered, rejected us → a
bootstrap candidate), **host-key failure** (changed key → operator must resolve;
never auto-fixed), and **unreachable** (network/sshd). Only the first is
offered a key install. The install's exit code is treated as a hint —
success is confirmed by re-probing, which is the only trustworthy signal.

**CLI wiring** (`cli/_setup/_ssh.py`): `_ensure_ssh_access` is the wizard's gate
(prints, prompts, persists a generated key as `ssh.key`); `_default_ssh_user`
replaces the old `os.environ.get("USER", "root")` — POSIX-only, so on Windows it
made every first connection as `root`. `_run_ssh_mesh` prefers
`scripts/mesh_ssh_keys.sh` and falls back to `mesh_ssh_keys_native` when local
`bash` is absent. The wizard runs the gate **once**, after the cluster-name and
SSH-username prompts and before any other probe.

> Note: other `os.environ.get("USER", "root")` call sites remain outside setup
> (`cli/_common.py`, `orchestration/primitives.py:is_local_user`,
> `orchestration/distribution.py`, `core/launcher.py`). They affect *launch-time*
> cross-user decisions and are still POSIX-only.

### Recipe System

Recipes are YAML files with fields: `model`, `runtime`, `container`, `containers`, `command`, `defaults`, `env`,
`metadata`, `min_nodes`, `max_nodes`. A plugin may also own additional top-level keys (see Plugin-Owned Recipe Items
above). The `Recipe` class (`core/recipe.py`) uses SAF `Variables` for config chain resolution —
CLI overrides → recipe defaults → runtime defaults.

Recipe resolution: CLI → `find_recipe()` (module-level function in `core/recipe.py`) → searches bundled recipes, local
`./recipes/`, user config recipes, and git-cloned registries.

Two recipe format versions exist: v1 (eugr-style, auto-detected by `recipe_version: "1"` or presence of `build_args`/
`mods`) and v2 (sparkrun native). vLLM recipes are resolved to either `vllm-ray` (if Ray hints are present) or
`vllm-distributed` (default). See `RECIPES.md` for the full specification.

### Recipe Validation (`core/validation.py`)

The single aggregator behind `sparkrun recipe validate`, and — through
`validate_for_launch` — behind what `run` / `benchmark` / `proxy launch` print
and refuse, so the four cannot drift.

Three severities, separated by one question: *if this recipe runs on a cluster
that isn't the author's — or on a later sparkrun — does it break or behave
differently?* **Errors** are things sparkrun cannot honor (always fatal).
**Warnings** are yes. **Suggestions** are no: it works as written and merely
gives something up, so they are for authors and registry CI and `run` does not
print them at all. Only errors are fatal by default; `--strict` is
`--fail-on warning`.

The **"or on a later sparkrun"** clause is the deprecation axis, added
deliberately rather than filed under suggestions. Read with place alone, a
deprecation answers "no" — it behaves identically everywhere *today* — which
would make `sparkrun run` silent about it. It does **not** add a fourth
severity: `--fail-on` still ranks three.

**Deprecations are collapsed at launch, not downgraded.** `RecipeIssue.deprecation`
is a *display* flag; `summarize_deprecations` replaces every deprecation finding
with one line naming the count and `sparkrun recipe validate <ref>`. The
migration is the author's work and at launch you are usually running someone
else's recipe — three deprecations printed in full is three paragraphs between
you and your logs. It runs **after** `should_fail`, on the display list only, so
`--strict` still fails on a deprecation it described in one line. `run` threads
the reference the user typed (`recipe_ref`) so the suggested command is pasteable.

**Two checks must read `recipe._raw`, and both would be silently wrong without
it** — the parsed recipe is lossy exactly where they look:

- `_resolve_brace_escapes` collapses `{{`→`{` in `defaults` **in place** at
  load, so by validation time the evidence is gone from `recipe.defaults`.
  (`command:` is untouched — masking happens inside `render_command`.)
- `Recipe.__init__` **derives** `mode` from the node range (`auto` → `cluster`
  when `min_nodes > 1`). Reporting the deprecated `mode:` key off the parsed
  value would fire on every recipe that correctly uses `min_nodes`/`max_nodes`
  and never wrote `mode` at all — i.e. on exactly what the finding advises.

**Deprecation notices belong here, not at their point of use.** Two of them
previously lived only as a `logger.warning` on a path `recipe validate` never
takes: one inside `render_command` (reached only by an actual launch) and one
inside `EugrVllmRayRuntime.prepare` (reached only *after* image distribution).
So the command whose entire job is to report what is wrong with a recipe was
silent while `sparkrun run` was not. Both are now findings; the original sites
keep a `debug` line.

Catalogue notes worth knowing before adding a check:

- **A finding that fires on correct recipes is how a report teaches people to
  skim it.** `_CONSUMED_RUNTIME_CONFIG_KEYS` exists solely for this:
  `build_args` at the top level is the *v1 spelling* and is read by name, so
  without the allowlist `unknown-top-level-key` would fire on every v1 recipe
  ever published. New `runtime_config.get()` call sites must join that list.
  `implicit-builder` is the same lesson measured: it reported all 47 registry
  recipes with an inferred builder until it stopped reporting the 40 inferred
  from a **first-party** `ghcr.io/spark-arena/` image. The concern is "sparkrun
  guessed about an artifact it does not control"; for its own images sparkrun
  owns both the image and the rule, so keeping them in step is its job.
- **`mods` is not a builder signal**, and `resolve_builder` (the catalog peer of
  the resolver chain, feeding `sparkrun list` / `api.search_recipes`) thought it
  was. The two disagreed in *both* directions — the display path counted `mods`
  (so a v2 recipe with `mods:` and an ordinary container was listed as
  `builder: eugr` while resolving to no builder at all) and missed the
  `container:` prefix (so a recipe that really does get an eugr build was listed
  as having none). They now share `_has_eugr_signal`, with a test asserting they
  agree rather than trusting the helper. A catalog that disagrees with the
  launch about what will be built is worse than one that says nothing, because
  it is read as an answer.
- **`EUGR_CONTAINER_PREFIX` is narrower than `FIRST_PARTY_CONTAINER_PREFIX` on
  purpose.** Being first-party does not make an image an eugr build
  (`dgx-spark-sglang` is neither), so the *inference* stays pinned to the
  `dgx-vllm-eugr-nightly*` families while the *advice* is suppressed for the
  whole org. Widening the inference would change what gets built for recipes
  that never asked for it. `core` cannot import `builders`, so the eugr prefix
  is spelled twice and drift-guarded by a test.
- **A finding names the signal it actually found**, not the catalogue of
  signals it might have. A list reads as a guess and sends the reader to the
  wrong entry — which is exactly how a recipe flagged for its `container:` was
  diagnosed as having been flagged for its `mods:`.
- **`misplaced-config-key` vs `unmapped-config-key`.** The latter
  (`launcher.report_unmapped_config_keys`) reads `defaults` and `-o` overrides —
  precisely the set an absorbed *top-level* key is not in. Nothing reported that
  shape before: `max_model_len: 8192` at the top level lands in `runtime_config`,
  which nothing consumes generically, and the rendered command shows nothing
  missing.
- **The two cache-env findings are opposite on purpose.** `get_extra_env()`
  (`HF_HOME`/`HF_HUB_CACHE`) is merged **last** so a recipe setting them is
  *discarded* (`overridden-cache-env`); the runtime-cache tier
  (`XDG_CACHE_HOME`) is merged **first** so a recipe setting it *wins* and the
  compile caches land off the mount (`managed-cache-env`). Same-looking
  mistake, opposite outcome, opposite advice.
- **`hardcoded-serve-flag` and `restated-model-arg` are mirror images.** The
  first is about values sparkrun *reads* from the config chain, the second
  about values it *writes* into the command. Both are suggestions for the same
  reason — the rendered command serves exactly what it says, identically
  everywhere — but the mechanisms defeated differ, so the advice does too.
  `recipe.model` is rewritten mid-launch by three mechanisms that all reach the
  command only through `{model}`: `resolved_model_path` (which also sets
  `_skip_model_distribution`), an absolute `model:`, and a pre-synced GGUF
  (whose raw value still carries the `:quant` suffix no runtime parses). None
  is the author's to control. Measured: 39 of the 118 cached registry recipes
  with a `command:` restate the id, and the id appears **nowhere else** in any
  of them — which is why matching is whole-token and never substring.
- **`restated-managed-path` reads `command:`, the hook lists and `defaults:`
  values, but deliberately not `env:`.** `env` already has two dedicated checks
  whose subject is a recipe touching sparkrun's cache wiring; a third message
  about the same assignment is the noise this catalogue exists to avoid.
  `_`-prefixed defaults are excluded for the same reason —
  `internal-config-key` owns that line and says strictly more.
- **`internal-config-key` fills a hole `_is_internal_config_key` left.** That
  helper excludes `_`-prefixed keys from the unmapped-key report — correctly,
  since they *are* consumed — which left a recipe **declaring** one
  (`_gguf_model_path`, `_mmproj_path`) with no diagnostic at all. The injected
  value outranks `defaults`, so the literal is normally dead; it wins under
  `--dry-run` (the command you review is not the one that runs), for pre-placed
  weights, and when the GGUF was not pre-synced here. Warning, not error: a
  requirement sparkrun's resolution cannot express is a reason to keep the key.
- **`deprecated-recipe-name` is a deprecation for something already inert.**
  `name:` is the v1 spelling; `Recipe.__init__` assigns the filename stem
  unconditionally (the `data.get("name", …)` beside it is commented out), so a
  declared name is *discarded on load* rather than merely scheduled to stop
  working — which is worth saying plainly, since a recipe that names itself one
  thing and resolves as another is only visible by listing it. It must read
  `_raw` (the parsed attribute is never the declared value — the `mode:` trap)
  via `getattr`, since `__setstate__` does not restore `_raw` and a recipe
  revived from the registry cache has no attribute at all. Gated on **not** v1,
  the same way the brace escape is: every v1 recipe in the cached corpus
  declares one and already carries `deprecated-recipe-format`, whose migration
  subsumes it; three v2 recipes declare one, which is the population this is
  for. Note the test fixtures had to drop `name:` too — `conftest`'s v2
  samples and `test_cli`'s carried it, which is how widely the v1 idiom had
  spread.
- **`inline-script` names tools, never shape.** A recipe carrying a program —
  a heredoc'd patch script, `sed -i` over the installed package, a launch-time
  `pip install` — is doing what `mods:` exists for, and the argument is
  sparkrun-specific rather than stylistic: the three homes render differently
  and only one leaves the script alone. `command:` goes through
  `render_command`, whose escape mode is inferred *from the template*
  (`uses_brace_escapes`), so an embedded program must double every literal
  brace and drags the whole template into v1 escape mode — one missed brace
  mis-renders it silently. A `pre_exec` string goes through
  `render_hook_command`, which does not collapse `{{` but still substitutes
  `{name}`, so a program is exposed to a collision with any config key it
  spells. A mod's `run.sh` is `docker cp`'d verbatim and never rendered. Shape
  is deliberately *not* a signal: the corpus clusters at 10–19 command lines
  with nothing over 40, so a line-count threshold would report the recipes with
  the most flags rather than the ones carrying code. Every signal measures zero
  across the corpus except `pip install` (2 recipes), which is why the check
  can afford to be broad. Suggestion, because an inline patch runs identically
  everywhere — what is lost is reviewability, not portability. It reads
  `defaults:` values too (a default reaches the command through its
  placeholder, so a program can be written one remove away), but with
  `site-packages` withheld there: every other signal names a *verb*, which is
  unambiguous wherever it appears, while that one names a *location* — the
  target of a write in a command, but as likely a plain path in
  `defaults.chat_template`.
- **`hardcoded-rendezvous-flag`'s flag list is the runtime's own**
  (`RuntimePlugin.managed_rendezvous_flags`), with **no shared core** even
  where two runtimes spell a flag identically. Which flags coordinate a launch
  is a property of the engine — Atlas says `--rank`/`--world-size` where vLLM
  says `--node-rank`/`--nnodes`, llama.cpp dials workers with `--rpc`, and
  `vllm-ray` (Ray) and `trtllm` (`mpirun -H`) rendezvous outside the serve
  command and so declare nothing. Declaring nothing is the base default and
  disables the check, which is what an out-of-tree runtime built against an
  older base class gets. It is a **warning** because the *single-node*
  direction breaks: these flags are appended unconditionally by
  `generate_node_command` (none of the `reconcile_flag_in_command` guarding
  that `--served-model-name` and `--distributed-executor-backend` get), so
  multi-node merely emits them twice and argparse drops the recipe's — but
  under `world_nodes <= 1` sparkrun appends *nothing*, and the recipe's
  `--nnodes 2` plus the author's head IP survive verbatim into a launch that
  rendezvouses with a host that is not there, or trips sglang's
  `(tp*pp) % nnodes == 0` assert before binding a port (issue #284).
  `--data-parallel-size` and sglang's `--dp-size` are excluded: they are
  injected only when the template did not supply them, so writing them is a
  supported choice rather than a collision.
- Every check runs through `_safe`, so a check that trips over an unexpected
  recipe shape costs its own finding and nothing else — it must not be able to
  block a launch it was only meant to describe.

### Registry System

The `RegistryManager` (`core/registry.py`) tracks recipe collections from remote git repos using sparse checkouts.
Registries are stored in `~/.config/sparkrun/registries.yaml`; cached clones live under `~/.cache/sparkrun/registries/`.

**Registry assets** — recipes, benchmark profiles, tuning configs and mods are all "a named file under a per-registry
subdirectory", so the shape is data, not four code paths. `RegistryAsset` (`RECIPE_ASSET`, `BENCHMARK_ASSET`,
`TUNING_ASSET`, `MODS_ASSET`) names the subpath field, whether the scan recurses, and the extension precedence; the
generic machinery hangs off it:

| Function                   | Role                                                                                     |
|----------------------------|------------------------------------------------------------------------------------------|
| `_iter_registries`         | the one enabled / visibility / name filter (every scan routes through it)                  |
| `asset_dir`                | `<cache>/<entry.<subpath_field>>` when it exists — the four `_*_dir` accessors wrap this   |
| `find_asset_in_registries` | resolve one name; per-registry flat-then-recursive, optional `accept` predicate            |
| `iter_asset_files`         | the *catalog* peer — same rules, so listing and lookup can never disagree                  |
| `qualified_asset_name`     | the typeable `@registry/<relpath>` label used to disambiguate                              |

Two rules are shared by every asset kind and are the reason the scan is not a plain `rglob`:

- **Flat beats nested, per registry.** A flat `<dir>/<name>.yaml` wins and suppresses that registry's recursive scan —
  but never another registry's (the bug fixed in #227).
- **`.yaml` beats a same-stem `.yml`, per directory.** They are one asset spelled two ways; treating them as two would
  produce an "ambiguous" error no name could resolve. The same stem in *different* subdirectories stays two distinct
  assets, so the catalog is never deduped by name.

Ambiguity therefore means "genuinely several assets", and both `RecipeAmbiguousError` and `ProfileAmbiguousError` carry
path-qualified `labels` (shared wording via `format_ambiguity`). Tuning configs and mods share only `asset_dir` —
tuning lookup is by runtime and returns a collection, so it is deliberately not routed through
`find_asset_in_registries`.

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

**Path containment** is a *separate* question from namespace legitimacy, and a name can be perfectly safe and still be an
impersonation. Names and subpaths come from remote `.sparkrun/registry.yaml` manifests and both become real paths — a
name is a directory under the cache root (`_cache_dir` is `cache_root / name`, and `_link_registry_to_shared` `rmtree`s
a non-link cache dir, so an escaping name is a *delete* primitive), and a subpath is resolved inside it (`asset_dir` is
`_cache_dir(name) / subpath`, whose contents `find_recipe` offers as runnable recipes, so an escaping subpath is a
*read* primitive feeding the recipe loader). `assert_safe_registry_name` / `assert_safe_registry_subpath` /
`assert_safe_registry_entry` contain this; both charsets require each path component to start alphanumeric, which rules
out `.`/`..`, dotfiles, a leading `-` (git would read it as an option) and the `_url_<hash>` shared-clone prefix in one
rule. `validate_registry_name` runs the name check **first** — `../sparkrun-x` doesn't *start with* a reserved prefix, so
the namespace rule alone would pass it. Enforcement differs by entry point on purpose: `add_registry` raises,
`_discover_manifest_entries` drops the bad entry and keeps the rest (raising only when nothing survives, so a hostile
manifest is never a successful no-op add), and `_load_registries_from_file` skips-with-warning — narrower than the
enclosing `except`, which reverts to the shipped defaults and would let one hand-edited entry discard every registry the
user has. `SUBPATH_FIELDS` is the list any new path-forming field must join, or it escapes validation entirely. See
`docs/SECURITY.md`.

**Tab completion**: `RecipeNameType.shell_complete()` in `_common.py` supports `@registry/recipe` syntax — `@` prefix
lists registries, `@registry/` lists recipes from that registry. Falls back to showing registry names when recipe cache
isn't populated.

### Recipe Catalog ("what recipes exist?")

All recipe *enumeration* flows through one function, **`api.search_recipes(query, …) -> list[RecipeSummary]`**
(`api/_recipes.py`) — the console-free peer of `api.status` for the catalog rather than the cluster. It merges the
configured registries with working-directory recipes, applies the registry/runtime filters, and returns typed
`RecipeSummary` rows (`api/_models.py`; `.to_dict()` yields the legacy `core.recipe.recipe_summary` mapping the CLI
formatters and `--json` consume). `sparkrun list` / `sparkrun recipe search` and their aliases are thin renderers over
it, and the desktop sidecar calls it directly instead of reaching into `RegistryManager`.

Two knobs carry the semantic difference between the commands:

| Knob            | `list`  | `search` | Meaning                                                                       |
|-----------------|---------|----------|-------------------------------------------------------------------------------|
| `unique_names`  | `True`  | `False`  | One row per unqualified name (locals first) vs every copy                     |
| `include_local` | `True`  | `True`   | Include CWD recipes (dropped anyway when a registry filter is set)            |

`unique_names` is why `list` shows a name once while `search` shows all of its variants: a registry's recipe dir is
scanned with `rglob`, so `3x-spark-cluster/foo.yaml` and `4x-spark-cluster/foo.yaml` are *different recipes sharing a
qualified name*. Only literal repeats (same resolved path, e.g. via shared-clone symlinks) are dropped unconditionally
— never dedupe the catalog by name.

**Implicit registry scope**: the positional QUERY of `recipe list` / `recipe search` (and the top-level `list` /
`search` aliases) accepts the same `@registry` syntax recipe *names* use — `@community` and `@community/` mean
`--registry community`, and anything after the `/` becomes the remaining query (`@community/qwen`).
`core/registry.py:resolve_registry_filter()` is the single resolver for both spellings (exposed as
`api.resolve_recipe_filter` for callers that need to name the resolved filter, e.g. in an empty-result message): it
strips the scope, rejects a scope that conflicts with an explicit `--registry`, then validates the resulting name —
unknown or disabled registries raise `RegistryFilterError` (→ `api.InvalidRegistryFilter` → `click.UsageError`)
listing the available ones, rather than silently yielding "No recipes found", whether the name arrived via the
shorthand or `--registry` (matching `registry list-benchmark-profiles`, which already validated upfront). A registry
filter implies `include_hidden` — naming a registry outranks its visibility default. The `RECIPE_QUERY` param type
completes `@` into `@registry/` and delegates to `RECIPE_NAME` past the slash.

`core.recipe.recipe_matches_query()` is the one matching predicate (substring over name / file / model /
description), shared by `RegistryManager.search_recipes` and the CWD scan so a local recipe is found on the same
terms as a registry one.

### Model & Container Distribution

Before launching, sparkrun can pre-sync models and container images from the control machine to target hosts:

- **Models** (`models/`): Downloads from HuggingFace Hub locally via `snapshot_download` (`models/download.py`), then
  rsyncs to targets (`models/distribute.py`, `models/sync.py`). GGUF models use colon syntax (`repo:quant`) for
  selective quant-file download.
- **Containers** (`containers/`): Pulls image locally (`containers/registry.py`), then streams via
  `docker save | ssh docker load` (`containers/distribute.py`, `containers/sync.py`). Checks image IDs to skip hosts
  that already have the correct image.
- **VRAM estimation** (`models/vram.py`): Model weights, the GPU memory budget, and the arithmetic that combines them
  with a KV cache estimate. Supports HuggingFace model auto-detection to resolve parameter counts.
  `Recipe.estimate_vram()` writes every detected field back into `metadata`, which is what lets later calls skip the
  HF fetch. `kv_cache_dtype` is resolved CLI → metadata → HF quant config → `defaults.kv_cache_dtype` →
  `--kv-cache-dtype` parsed from the `command:` template (last resort, with a warning). When no dtype is resolved, the
  estimator computes with `bfloat16` but leaves `VRAMEstimate.kv_dtype` as `None` so the CLI formatter can show
  `bfloat16 (default)` rather than silently reporting a guessed value. Element widths live in the `models/dtypes.py`
  leaf (weights and KV are separate tables — NVFP4's KV packing carries block scales its weight packing does not).

#### Hub Metadata Budget (`models/hub.py`)

Everything that asks the HuggingFace Hub for **metadata** — `fetch_model_config`, `fetch_hf_quant_config`,
`fetch_safetensors_size`, `fetch_safetensors_params`, `fetch_model_visibility` — routes through
`hub_metadata_call`. Weight downloads do **not**: a 200 GB pull legitimately takes hours and must not be
budgeted. The distinction is the whole design. Everything behind this seam is *advisory* — the launch already
degrades to "no memory claim" when an estimate is unavailable (`api/_hosts.py`) — so none of it is worth a
second of unexplained hang (issue #278, where `sparkrun run` sat for 10+ minutes printing nothing).

**The guarantee: the advisory phase costs at most `hub.metadata_budget_s`** (default 30 s; `0` = unbounded).
Four levers, because four separate things were unbounded and no one of them subsumes the rest:

| Lever | Bounds | Why nothing simpler works |
|-------|--------|---------------------------|
| `configure_hub_client` | every httpx call in the library | `huggingface_hub` builds its shared client with `timeout=None`; `list_repo_tree` accepts no `timeout` argument, so the client factory is the only reachable knob |
| `_align_download_timeouts` | `hf_hub_download`'s own ceilings | it passes `HF_HUB_ETAG_TIMEOUT` / `HF_HUB_DOWNLOAD_TIMEOUT` explicitly, which *override* the client — so the client default silently missed the calls sparkrun makes most |
| `without_xet` | metadata transfers on Xet repos | `hf_xet` is a Rust HTTP stack honouring none of the Python timeouts. Metadata is kilobytes of JSON, where Xet's chunk dedup buys nothing; weight downloads keep it |
| `_run_with_deadline` | one lookup, whatever it does inside | `http_backoff` retries 5× with exponential backoff at fixed call sites. Measured: ~40 s for one `hf_hub_download` with the client bounded at 3 s. **A per-request ceiling does not bound a lookup** |

The last one is the only structural layer, and it is why the guarantee survives a `huggingface_hub` upgrade.
It abandons a daemon thread rather than cancelling (Python cannot interrupt a blocked socket read); the
breaker bounds that to one leaked thread per command, and a parked thread burns no CPU and cannot hold the
interpreter open.

Three rules are load-bearing:

- **Only time trips the breaker.** A 404 for `hf_quant_config.json` is the normal outcome for most repos and
  costs milliseconds; classifying library exceptions would report a missing optional file as an outage.
- **Negatives are memoised, successes are not.** `Recipe.estimate_vram` writes *successful* detection back
  into `metadata` and re-runs detection when it is absent — so an unreachable Hub failed `needs_detection`
  every time and refetched on all three estimates a single `run` performs. Successes are already carried by
  that write-back; duplicating it here would hand callers a shared mutable dict.
- **`hub_degraded_message` returns its string once.** Fifteen lookups share one breaker. `cli/_run.py` calls
  it from both points where the phase can run out (the plan, and the VRAM table after it) precisely because
  the once-only contract makes that correct rather than merely tolerable.

Escape hatches, in the order the message offers them: `HF_HUB_OFFLINE=1`, `--no-auto-detect` (which calls
`disable_hub_metadata()` — process-wide rather than an `auto_detect=False` threaded down, because
`estimate_vram` runs from host resolution, the banner, the scheduling pass *and* telemetry, and a flag
reaching three of those four would look like it worked), and raising `hub.metadata_budget_s`.

The other half of #278 was that **nothing was printed before `api.plan`**, so the pause was unattributable.
`cli/_run.py` now prints the identity lines (version / runtime / image / model) above the plan — they need
nothing from it — followed by a `Planning: ...` line. Everything placement-dependent still renders from
`run_plan`, so the display cannot disagree with the launch.

#### KV Cache Sizing (`models/kv/`)

Sizing the KV cache is architecture-specific, so it is a seam rather than a branch. `vram.py` names no attention
architecture; it resolves a strategy and asks it. See `.slop/kv-cache-sizing-design.md` for the rationale.

A `KVCacheStrategy` (`kv/_base.py`) answers three questions — *how big at length L* (`size`), *how big per token*
(`KVSizing.per_token_bytes`, for display), and *how long fits in a budget* (`tokens_for_budget`). **Sizing is a total
for a requested length, not a per-token figure multiplied out.** Per-token is the special case: it holds for dense and
MLA but not for a sliding-window layer (caches `min(max_model_len, window)`) or a Mamba/SSM layer (state is per
*sequence*). Making the total primitive is what keeps those addable.

| Strategy | Module | Priority | Sizing |
|----------|--------|----------|--------|
| `mla`    | `kv/mla.py`   | 10   | DeepSeek V2/V3/V4 compressed latent — one latent per token per layer, **replicated across TP ranks** (only PP divides it). Selected by `kv_lora_rank`/`qk_rope_head_dim`, by an `*_ds_mla` packed layout in `kv_dtype`, or by a `deepseek_v*`/`kimiko*` `model_type`. `mla_latent_dim()` normalizes V2/V3 (latent + RoPE tail on top) and V4 (both folded into `head_dim`) to the NoPE width, so the tail counts once. |
| `dense`  | `kv/dense.py` | 1000 | `2 * layers * kv_heads * head_dim * bytes`. Claims everything, so resolution always terminates. |

Resolution is **order-sensitive** (most specific `detect()` first), which is why the registry is in-process rather
than SAF — the `platforms/` precedent. Out-of-tree plugins call `register_kv_strategy()` from their `register(v)`
hook.

Two rules the seam exists to enforce:

- **Detection is separate from sizing.** A strategy that recognises a model but cannot size it returns
  `KVSizing(total_bytes=None, unsizable_reason=…)` — it does *not* fall through to dense, whose formula would
  overestimate a latent cache by ~100x. `VRAMEstimate.kv_arch` therefore reports the detected architecture even when
  the estimate is unavailable; relabelling it dense would misreport it to `to_dict()` (benchmark export) and flip the
  replication rule a `kv_vram_per_token` override still depends on.
- **A strategy *declares* the architecture fields it reads** (`ArchField`: canonical name, HF config keys, kind,
  validation, docs). That declaration is the single source of truth for HF extraction (`extract_arch_fields`), recipe
  `metadata` keys, their validation, and the post-estimate write-back — four sites that were four hand-maintained
  copies of the same list, where omitting one silently reverted the estimate on the path that decides placement.
  `estimate_vram(arch={...})` takes them as a mapping, never as per-architecture keyword arguments.

`tests/test_kv_strategies.py` registers a sliding-window strategy at runtime and asserts it reaches extraction,
estimation, recipe write-back and validation with **no core edit** — the executable form of that claim.

### Runtime Cache (`core/runtime_cache.py` + `orchestration/runtime_cache.py`)

Containers are `--rm`, so every launch discarded its compilation and autotune output — torch.compile /
Inductor graphs, Triton cubins, FlashInfer JIT modules, and the TRT-LLM autotuner, which was never
written at all (issue #256). A host directory is now mounted at `/cache/runtime`, sibling of
`/cache/huggingface` and the same convention.

**The load-bearing invariant is that directory keying is hygiene, never correctness.** The host path
is `<root>/<family>/[<image-key>/][<model-key>/]`, and `key_by_image` defaults **off** — so nothing
may depend on the key to avoid loading a stale artifact. torch.compile / Inductor / Triton /
FlashInfer are content-addressed internally and are correct (and far more useful) in a tree shared
across image versions. TRT-LLM's autotuner is the exception in both directions: it holds tactics for
exactly one configuration, so it carries `derive_recipe_fingerprint` **in its filename** regardless
of the directory key; and it validates neither the version nor the GPU it records, so `trtllm`
returns `{"key_by_image": True}` from `runtime_cache_defaults()` — the runtime tier, so every TRT-LLM
recipe including a user's own is safe without a `runtime_cache:` block.

The container path is **constant**; all keying is host-side, so recipes and serve commands never
spell a key.

| Piece | Role |
|-------|------|
| `RuntimePlugin.runtime_cache_paths(fingerprint=)` | env var → `CachePath` **relative** to the mount (`file=True` ⇒ only the parent is created) |
| `RuntimePlugin.runtime_cache_defaults()` | runtime tier of the settings chain — same slot `default_executor()` holds in `resolve_executor` |
| `resolve_runtime_cache_settings` | recipe → CLI → cluster → config → runtime → baseline; `SPARKRUN_NO_RUNTIME_CACHE` beats all |
| `build_runtime_cache_mounts` | volumes + env + dirs, or `None` (⇒ byte-identical to pre-feature) |
| `Executor.ensure_runtime_cache` | write-path peer of `verify_mount_sources`; base no-op, docker/local share the SSH impl |

Two env rules are load-bearing and fail *silently* if broken (both have regression tests):

- **`XDG_CACHE_HOME` is the catch-all, so `HF_HOME`/`HF_HUB_CACHE` must stay explicitly set** —
  `huggingface_hub` honors XDG, and losing that ordering relocates the model cache off its own mount
  and re-downloads the weights on every launch.
- **The cache env is injected at the *lowest* tier**, not through `get_extra_env` (which wins over
  `recipe.env` and would clobber a recipe that points `VLLM_CACHE_ROOT` itself).

`ensure_runtime_cache` does mkdir + marker-touch + prune in **one** SSH round-trip. The `mkdir` is not
optional — Docker materializes a missing `-v` source **root-owned**, breaking rootless and breaking
`local` outright. Pruning ages trees by the `.sparkrun-last-used` marker rather than directory mtime,
because reading a cache never touches the directory: an mtime-aged sweep would delete exactly the warm
trees it should keep. Best-effort throughout — a cache that could not be prepared costs a recompile,
never a launch. Manual sweep: `sparkrun setup prune-runtime-cache`. Design: `.slop/runtime-cache-design.md`.

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

Readiness budgets live under `readiness:` in `config.yaml`
(`port_timeout_s` / `health_timeout_s`, `0` = unbounded) — raise
`port_timeout_s` for engines with long graph-capture phases; see Launch
Timing above.

HuggingFace Hub budgets live under `hub:` (`timeout_s` per request,
`metadata_budget_s` for the whole advisory phase, `0` = unbounded — see Hub
Metadata Budget above). Note the asymmetry with `readiness.*`: a non-positive
`timeout_s` falls back to the default rather than meaning "unbounded", because
an unbounded Hub client is the defect and has no spelling.

### Feature Flags (`core/features.py`)

Channel-aware gating for experimental plugins and behavior. Each `FeatureFlag`
(registered in the module-level `FEATURE_FLAGS` registry) carries a
`description`, per-channel `channel_defaults`, and a baseline `default`.
`is_feature_enabled(name, config=…)` resolves with precedence: env override
(`SPARKRUN_FEATURE_<NAME>`, dots→underscores) → `features.<name>` in
`config.yaml` → per-channel default → baseline → fail-closed for unknown flags.

The active channel reuses the release channel from `core/channels.py`
(`stable`/`beta`/`alpha`): `SparkrunConfig.feature_channel` reads
`features.channel`, falling back to `self_update.channel`. Via `channel_defaults`
a flag can be on-by-default for `alpha` while off for `stable`/`beta`. The
built-in flags — `executor.local` and `executor.k8s` (gating the corresponding
experimental executors), `cli.setup.k8s` (gating the entire `sparkrun setup
k8s` command group), `cli.setup.tailscale` (gating the `sparkrun setup
tailscale` group), and `transports.thunder` (gating the Thunder Compute
transport + `cluster import thunder`) — are off by default on **every** channel;
enable them explicitly per-flag. `builder.uv_venv` is the first flag to use
`channel_defaults` for a *plugin* rather than for visibility: off on `stable`,
on for `beta`/`alpha` (see Environment Builders below). The `setup k8s` group self-gates in its Click
callback (raises pointing at `setup features enable cli.setup.k8s`) and hides
itself from `setup --help` unless the flag resolves on at import; `setup
tailscale` and `cluster import thunder` gate the same way
(`cli.setup.tailscale` / `transports.thunder`), and the Thunder transport also
fails closed at use in `transports.prepare_cluster_transport` so an
already-imported Thunder cluster can't run once the flag is off.

**Docker gate (`executor.docker`)**: the default executor gates like every other
one — for uniformity and a future opt-out — but ships **enabled on every channel**
(`default=True`, no channel overrides). Disabling it (`features.executor.docker:
false`) removes docker from `list_executors()` / explicit selection, and the
baseline-default resolution honors that: `_resolve_executor_name` no longer
hard-codes `"docker"` when no layer names an executor — `_default_executor_name`
returns docker when enabled, else the sole enabled executor, else raises "name
one / set `default_executor`" (never silently runs on a disabled backend).

**Gateway gate (`gateway.litellm`)**: same shape as the docker gate — ships
enabled on every channel, exists so an alternate inference gateway can be added
as a peer. Exclusivity ("one gateway at a time") is arbitrated at *resolution*,
not by the flag registry. See Inference Gateway above.

**Visibility-only gate**: `cli.setup.features` (via `channel_defaults`, **on for
`beta`/`alpha`, off for `stable`**) is different — it does NOT gate execution.
The `setup features` group is always functional; the flag only decides whether it
appears in `setup --help` (`@setup.group("features", hidden=not
_setup_features_visible_at_import())`, no callback raise). So a stable user can
still run `setup features enable <flag>` even though the group is hidden.

**Plugin gating**: a plugin opts in by setting `required_feature_flag = "<flag>"`
(e.g. on `LocalExecutor`/`K8sExecutor`) and self-gates via `is_multi_extension` —
SAF only exposes a multi-extension plugin through `get_extensions` when that hook
returns True, so a gated plugin stays in the plugin registry but is absent from
`list_executors()`, tab-completion, and resolution. `core.bootstrap` stays a pure
discovery loop (no config reads); the gate resolves config itself via
`features.feature_gate_enabled` at registration time (env overrides
short-circuit before any file read). The decision is frozen per-process, which
is fine for the one-shot CLI. An explicitly-requested but gated/unknown executor
raises `ExecutorUnavailableError` (never a silent docker fallback); teardown of a
job whose executor was later disabled will also fail — the accepted cost of
relying on an experimental, opt-in feature.

Manage via `sparkrun setup features {list,enable,disable,reset}` (advanced, under
`setup`). Example `config.yaml`:

```yaml
features:
  channel: alpha          # optional; defaults to self_update.channel
  executor.k8s: true      # explicit per-flag override (beats channel default)
  executor.local: true
```

Tests: the `isolate_stateful` conftest fixture force-enables the two executor
flags via env so the legacy executor suite (which predates gating) keeps
passing. `tests/test_features.py` unit-tests the gate directly
(`K8sExecutor().is_multi_extension(...)`) and exercises exclusion end-to-end in a
clean subprocess (SAF exposes `is_multi_extension` once at registration and the
registry is process-global, so a plugin can't be re-hidden mid-process).

### Testing Patterns

Tests use pytest with `pytest-asyncio`. The `conftest.py` provides an `isolate_stateful` autouse fixture that redirects
SAF's stateful root to `tmp_path`, preventing tests from touching `~/.config/sparkrun/`. The bootstrap singleton (
`_variables`) is reset between tests. All core module imports in tests use `sparkrun.core.*` paths (e.g.,
`from sparkrun.core.registry import RegistryManager`).

All SSH/Docker operations in tests are mocked — no real hosts are needed. Common fixtures: `tmp_recipe_dir` (creates
sample v1/v2 recipes), `cluster_dir`, `hosts_file`, `v` (initialized SAF Variables instance).

**The suite is hermetic — it touches neither the developer's state nor the network.** Both properties are enforced in
`isolate_stateful`, and both were once broken in ways that hid for a long time:

| Guard                                                          | What it prevents                                                                                    |
|----------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `DEFAULT_CONFIG_DIR` → `tmp_path` (+ `STATEFUL_ROOT`)          | reading the developer's clusters / registries / default hosts                                        |
| `DEFAULT_CACHE_DIR` → `tmp_path` (+ `pending_ops`, `tuning._common`, which bind it at import) | reading *live* state — `ProxyEngine` defaults `state_dir` to `DEFAULT_CACHE_DIR/proxy`, so a test would report on (and `stop()` would SIGTERM) a really-running proxy |
| `BOOTSTRAP_REGISTRY_URLS` → `[]`                               | first-run manifest discovery git-cloning three GitHub repos                                          |
| `RegistryManager._clone_or_pull` → stub                        | every other registry `git clone` / `fetch`                                                           |

The last two are why the suite runs in ~70s rather than 30+ minutes: registry git was costing seconds *per test*, masked
for years by the cache dir leaking out to an already-populated `~/.cache/sparkrun/registries`. Sandboxing the cache
turned those pulls into full clones, which is how it surfaced.

Consequences for writing tests:

- **Never assume a registry recipe exists.** Recipes must be created locally — a flat `*.yaml` in a `monkeypatch.chdir`
  target (needs `model` + `container` + a resolvable `runtime` to pass `is_recipe_file`), or a direct path passed to a
  command (`find_recipe` resolves paths before registries).
- Tests that assert on **git argv** take the `real_registry_git` fixture, which restores the real `_clone_or_pull`;
  they stay hermetic by mocking `subprocess.run` themselves.
- Tests that exercise **manifest discovery** supply their own URLs (`bootstrap_urls` in `test_registry.py`) and mock
  `_discover_manifest_entries`.
- Assert against `<module>.DEFAULT_CACHE_DIR` rather than an import-time copy, which would be the real path.

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
