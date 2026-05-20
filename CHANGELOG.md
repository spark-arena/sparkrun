# Changelog

All notable changes to sparkrun are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows semantic versioning.

For the long-form 0.3.0 narrative, see [`docs/RELEASE_NOTES.md`](docs/RELEASE_NOTES.md).

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
