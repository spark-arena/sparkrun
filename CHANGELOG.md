# Changelog

All notable changes to sparkrun are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows semantic versioning.

For the long-form 0.3.0 narrative, see [`docs/RELEASE_NOTES.md`](docs/RELEASE_NOTES.md).

## [Unreleased]

### Highlights — Benchmarking redesign

- Per-category CLI: `sparkrun benchmark performance <recipe>` (alias `perf`),
  `sparkrun benchmark tools <recipe>`. Bare `sparkrun benchmark <recipe>`
  remains and falls through to `performance`. `sparkrun benchmark run` is
  preserved as a legacy entry that imposes no category. New categories
  appear automatically when a plugin registers them.
- Non-interactive `--resume` flag (mutually exclusive with `--fresh`) on
  both `benchmark run` / category subcommands and `arena benchmark run`.
  Existing TTY prompt is unchanged for default invocations.
- Spark Arena via flag: `sparkrun benchmark perf <recipe> --arena` runs
  the same opinionated flow as `sparkrun arena benchmark`; both entry
  points share new `preflight_arena` / `finalize_arena` helpers.
- Container image pinning for resumable runs: content-addressable SHA
  (`container_image_sha`) captured on first launch and used to override
  `overrides["image"]` on resume, so re-pushed tags or rebuilt local
  images cannot silently change the bits between sessions. The builder's
  long-term archival reference (`container_image_longterm_ref`) is
  persisted separately so resumed sessions emit identical archive
  provenance.
- Public library API: `sparkrun.api.benchmark(BenchmarkOptions)` returns
  a typed `BenchmarkResult`; orchestration lifted into
  `sparkrun.api._benchmark._execute_benchmark`. CLI becomes a thin shell
  that wires a `_CliEmitter` and translates typed exceptions back into
  `click.echo` + `sys.exit`. All `sys.exit()` paths inside the
  orchestration are typed exceptions; `KeyboardInterrupt` re-raises after
  state preservation.

### Added

- `BenchmarkingPlugin.categories` / `primary_category` class attrs; default
  `("performance",)`. `tool-eval-bench` declares `("tools",)`.
- `BenchmarkSpec.category` + `resolved_category()` helper.
- `find_benchmark_profile(..., category=...)` filter; same kwarg on
  `RegistryManager.find_benchmark_profile_in_registries` and
  `list_benchmark_profiles`. Per-process cache keyed on (path, mtime, size).
- `sparkrun.core.bootstrap`: `list_benchmark_categories()`,
  `get_benchmarking_frameworks_for_category(category)`,
  `get_default_framework_for_category(category, config)`, plus
  `AmbiguousCategoryError` / `CategoryNotFoundError`.
- `sparkrun.api`: `benchmark()`, `BenchmarkOptions`, `BenchmarkResult`,
  `ResumeMode` (`AUTO`/`IF_EXISTS`/`FRESH`/`REQUIRED`), `ProgressEvent`,
  plus `BenchmarkFailed`, `NoResumableState`, `CategoryNotFound`,
  `AmbiguousCategoryError`, `FrameworkCategoryMismatch` typed errors.
- `sparkrun.orchestration.primitives.resolve_image_sha()` — captures the
  content-addressable image ID from a target host.
- `sparkrun.cli._arena_flow`: `preflight_arena`, `finalize_arena`,
  `persist_arena_extras`, plus `ARENA_BENCHMARK_PROFILE` constant.
- `BenchmarkResult.longterm_image_ref` / `longterm_image_pinned` fields;
  `generate_metadata()` prefers the persisted ref when set.
- `--resume`, `--arena`, `--local-test` flags on the category subcommands
  via the new `_shared_run_options` decorator.

### Changed

- `BenchmarkRunState.extras` now carries `container_image_sha`,
  `container_image_longterm_ref`, and `container_image_longterm_pinned`
  on resumable runs (additive — schema unchanged).
- `_run_benchmark` (CLI) shrunk from ~1000 to ~175 lines; orchestration
  body moved to `sparkrun.api._benchmark._execute_benchmark`. Behavior
  preserved: same banners, same exit codes, same error formatting.
- `click.confirm` resume prompt replaced with a `ResumeMode` decision
  tree + injectable `on_prompt_required` callback.

## [0.3.0] — 2026-05-20

### Highlights

- Multiplatform foundations (`HostHardware`, `CollectiveBackend`,
  `HardwarePlatformPlugin`) with the DGX Spark path preserved byte-for-byte.
- Unified executor resolution (`resolve_executor()`) and two experimental
  executors: `LocalExecutor` (native subprocess) and `K8sExecutor` (kubectl
  draft).
- Security hardening across recipe hooks, git URLs, sudoers interpolation,
  delegated copies, and trtllm SSH host-keys.

### Added

- `core/hardware.py`: `AcceleratorSpec`, `HostHardware`, `default_dgx_spark_hardware()`.
- `core/hardware_probe.py`: combined accelerator + InfiniBand SSH probe
  (`probe_host` / `probe_hosts`) — one round-trip instead of two.
- `core/backend_select.py`: `select_backends(HostHardware) -> BackendBundle`,
  `NoMatchingBackendError`.
- `core/placement.py` / `core/layout.py`: explicit rank → (host, local-GPU)
  placement, auto-pack for single-vendor clusters.
- `core/launcher.py`: `resolve_per_host_backends()`, centralized compatibility
  check, recipe trust resolution, platform validation pass.
- `orchestration/executor.py` (facade) + `orchestration/executors/` package
  (`_base.py`, `docker.py`, `local.py`, `k8s.py`).
- `orchestration/collectives/` package: `base.py`, `nccl.py` (default),
  `rccl.py` scaffold, `hccl.py` scaffold.
- `platforms/` package: `base.py` (ABC + `validate_host` hook), `dgx_spark.py`,
  `nvidia_generic.py`.
- Recipe fields: `executor`, `executor_config` (Local: `working_dir`,
  `log_dir`, `pid_dir`, `env_file`, `command_prefix`; K8s: `k8s_namespace`,
  `k8s_context`, `k8s_node_selector`, `k8s_image_pull_policy`, `kubeconfig`).
- `SparkrunConfig` fields: `default_executor`, `executor_config`.
- `utils/resource_loader.py`: unified embedded-resource loader (used by
  `scripts/`).
- `utils/shell.py:quote()`, `validate_unix_username()`, base64-bash wrappers.

### Changed

- `RuntimePlugin.run()` accepts a new keyword-only `backends:
  dict[str, BackendBundle] | None`. Threaded through every native multi-node
  runtime (vllm-distributed, sglang, trtllm, llama-cpp).
- vLLM command generation centralized in `runtimes/_vllm_mixin.py:VllmMixin._build_command`.
- `RuntimePlugin._make_node_command_args` template used by every native
  multi-node runtime instead of per-runtime ad-hoc argv emission.
- `RuntimePlugin.executor` lazy property replaced with `_resolve_executor()`
  helper that delegates to `orchestration.executor:resolve_executor()`.
- Executor selector validation queries the SAF registry via
  `get_extensions(EXT_EXECUTOR)`; the static `{docker, local, k8s}` set is a
  test-only fallback.
- `core/fingerprint.py:fingerprint_host` is now a thin shim over
  `core/hardware_probe.probe_host`.
- CLI host-context decorator (`cli/_common.py:@with_host_context`) replaces
  duplicated host/cluster resolution boilerplate in six CLI modules.

### Removed

- `RuntimePlugin.executor` property.
- Hardcoded `_KNOWN_EXECUTORS` set (replaced by SAF discovery).

### Deprecated

- `runtimes/_cluster_ops.py:resolve_ib_env(ctx, comm_env)` — emits
  `DeprecationWarning`. Consumers should switch to `resolve_comm_env(ctx,
  comm_env, backends)` and pass per-host `BackendBundle`.

### Security

- Trust gating on `pre_exec`, `post_exec`, and `post_commands`. Local /
  default-registry recipes auto-trusted; third-party registry recipes prompt
  or require `--trust`.
- `orchestration/transfer.py:_run_delegated_copy` validates `source_host` and
  `dest` paths.
- `utils/shell.py:validate_unix_username()` gates every sudoers interpolation
  in `cli/_setup/`.
- `runtimes/trtllm.py` removes SSH host-key relaxation in MPI rsh wrapper.
- `core/registry.py:_validate_git_url` allowlists `https://`, `git@`,
  `ssh://`, `file://` schemes only.
- OAuth callback CORS allowlist restricted to `AUTH_PROXY_BASE`.
- Token-prefix logging removed from debug paths.

[0.3.0]: https://github.com/spark-arena/sparkrun/releases/tag/v0.3.0
