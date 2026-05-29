# sparkrun 0.3.0 — Release Notes

Multiplatform foundations, unified executor resolution, and the B-workstream
security tightening land in this release. The shipped DGX Spark path is
unchanged byte-for-byte (NCCL output, container launch, post-launch lifecycle);
the surface area beneath it has been factored so AMD, Intel Gaudi, Local, and
Kubernetes hooks plug in without re-plumbing runtimes.

Repository: <https://github.com/spark-arena/sparkrun>

## Highlights

- **Multiplatform seams**: `HostHardware` / `AcceleratorSpec` /
  `CollectiveBackend` / `HardwarePlatformPlugin` are the abstractions every
  future vendor implementation builds against. NVIDIA stays the only fully
  wired vendor; AMD (RCCL) and Intel Gaudi (HCCL) ship as scaffolds.
- **Unified executor resolution**: a single `resolve_executor()` chain
  (CLI → recipe → runtime → SAF defaults → fallback) replaces ad-hoc executor
  construction. Two experimental executors land: `LocalExecutor` (native
  subprocess, no container) and `K8sExecutor` (`kubectl run`-driven draft).
- **Security hardening**: URL recipes never auto-trusted + hardened recipe
  fetch, explicit/persisted proxy bind host, arena OAuth state binding, trust
  gating on every recipe hook surface (`pre_exec` / `post_exec` /
  `post_commands`), git URL allowlist, validated sudo usernames and cache
  paths, strict trtllm host-key checking, delegated-copy host/path validation.
- **Scales to larger clusters**: bounded SSH/rsync fan-out, parallel
  head→worker distribution, execution timeouts on the launch path, and
  parallel per-host cleanup keep 8/16/32-host launches from saturating the
  control node or stalling on one slow host.
- **Console-free library API**: the `sparkrun.api` package no longer imports
  the CLI; `api.run` / `api.benchmark` / `api.resume_benchmark` share a single
  placement authority and raise typed errors instead of writing to stdout.

## New

### Multiplatform foundations

- `core/hardware.py` introduces `AcceleratorSpec` and `HostHardware`, the data
  model every multiplatform path consumes. Hosts without explicit metadata
  default to DGX Spark via `default_dgx_spark_hardware()`.
- `core/hardware_probe.py:probe_host()` / `probe_hosts()` collapse the
  accelerator fingerprint and InfiniBand detection into a single SSH
  round-trip. Output is split by sentinel markers and routed to existing
  parsers.
- `core/fingerprint.py:fingerprint_host` is now a thin shim retained for
  callers that don't pay for the IB section.
- `core/backend_select.py:select_backends(host_hardware)` returns a
  `BackendBundle(accelerator_vendor, collective)`. `launcher.py` calls this
  per host and threads the result through `runtime.run(..., backends=...)`.
- `core/placement.py:compute_placement()` maps `ParallelismConfig` onto hosts
  honoring an optional `RecipeLayout`. Auto-packs single-vendor clusters,
  raises `LayoutRequiredError` for multi-vendor clusters.

### `CollectiveBackend` abstraction

- `orchestration/collectives/base.py` defines the ABC.
- `orchestration/collectives/nccl.py` wraps `infiniband.generate_nccl_env` /
  `generate_ring_nccl_overrides`. Byte-identical to legacy DGX output.
- `orchestration/collectives/rccl.py` and `hccl.py` are scaffolds that raise
  `NotImplementedError` from `env_for_host` — surfaces the missing
  implementation rather than silently emitting NCCL.
- `orchestration/collectives/__init__.py:get_backend(vendor)` is the lookup;
  `UnsupportedCollectiveError` covers vendors with no scaffold (Apple, CPU,
  ...).

### `HardwarePlatformPlugin` + `validate_host` hook

- `platforms/base.py` defines the ABC. Built-ins in `platforms/dgx_spark.py`
  (`DgxSparkPlatform`) and `platforms/nvidia_generic.py`
  (`GenericNvidiaPlatform`).
- `DgxSparkPlatform.validate_host()` warns when the GB10 accelerator is
  missing the `rdma:roce-v2` capability — multi-node collective health
  concern.
- `GenericNvidiaPlatform.validate_host()` warns when `matches()` was called on
  a host without any NVIDIA accelerator.
- `launcher.py` runs `validate_host` per placed host after backend resolution.
  Warnings are logged; not raised.

### Executor subsystem

- `orchestration/executor.py` is the public facade. Exports `Executor`,
  `ExecutorConfig`, `EXT_EXECUTOR`, `get_executor()`, `list_executors()`,
  `resolve_executor()`.
- `orchestration/executors/_base.py` holds the ABC and `ExecutorConfig`
  dataclass; `from_chain()` parses CLI/recipe/runtime layers.
- `orchestration/executors/docker.py` owns `DOCKER_DEFAULTS` and the
  `rootless` / `auto_user` adjustment lever (`apply_runtime_adjustments`).
- `orchestration/executors/local.py` (`LocalExecutor`): native subprocess via
  `setsid`, pid/log-file lifecycle, no images. See `docs/EXECUTORS.md` for
  limitations.
- `orchestration/executors/k8s.py` (`K8sExecutor`): `kubectl run`-driven
  draft. Drops Docker-only options (`--privileged`, `--shm-size`, ...).
- `core/bootstrap.py` discovers executors via
  `find_types_in_modules("sparkrun.orchestration.executors", Executor)`.

### Recipe / config surface

- New recipe field `executor: docker | local | k8s` (default `docker`).
- New recipe block `executor_config:` for per-executor knobs (Local:
  `working_dir`, `log_dir`, `pid_dir`, `env_file`, `command_prefix`; K8s:
  `k8s_namespace`, `k8s_context`, `k8s_node_selector`, `kubeconfig`, ...).
- `SparkrunConfig` accepts `default_executor` + `executor_config` to set the
  fleet-wide default below recipe overrides.

### `core/launcher.py`

- `resolve_per_host_backends(host_list, cluster=...)` builds the per-host
  `BackendBundle` map.
- Centralized compatibility check (`runtimes/compatibility.py:
  check_runtime_host_compatibility`) walks every host before any side effects
  (container pull, model sync, ...). Raises `IncompatibleHardwareError`,
  surfaced cleanly by the CLI.
- `resolve_recipe_trust(recipe, trust_cli)` returns one trust verdict per
  recipe, shared by `pre_exec` and `post_exec`/`post_commands`.

### Scalability & robustness

- `SparkrunConfig.max_parallel_ssh` (config key `ssh.max_parallel_ssh`, default
  20) caps every parallel SSH/rsync fan-out; small clusters are unchanged.
- Head→worker image/model distribution (`image_distribute.sh` /
  `model_distribute.sh`) runs transfers with bounded concurrency instead of
  serially.
- Launch-path parallel SSH calls carry explicit execution timeouts so a host
  that connects then hangs no longer blocks the whole batch.
- Container cleanup runs in parallel and reports per-host success/failure,
  naming hosts that may still hold VRAM after a failed launch.
- `pending_ops` locks record host + token + timestamp, prune by max age (guards
  PID reuse / NFS-shared caches), and verify ownership before removal.

### Library API

- `sparkrun.api` is a console-free facade over `core/`: it no longer imports
  from `sparkrun.cli`. Shared resolution moved to `core/resolve.py` and
  `api/_hosts.py:resolve_effective_hosts` (the single placement authority used
  by `run`, `benchmark`, and the CLI).
- New `api.resume_benchmark(...)` entry point; benchmark resume is no longer
  CLI-only.

## Breaking changes

- **`RuntimePlugin.run()` signature** gains a keyword-only `backends:
  dict[str, BackendBundle] | None = None`. Existing runtimes that subclass
  `RuntimePlugin.run()` must accept and forward the new kwarg. Solo-mode and
  every in-tree native multi-node runtime are migrated.
- **`RuntimePlugin.executor` property removed.** Replaced by
  `_resolve_executor()` which delegates to
  `orchestration.executor:resolve_executor()`. Subclasses that previously read
  `self.executor` directly should call `self._resolve_executor()`.
- **`_KNOWN_EXECUTORS` set removed.** Executor selector validation now queries
  the SAF plugin registry via `get_extensions(EXT_EXECUTOR)`; the static set
  remains as an in-process fallback for test harnesses that bypass
  `init_sparkrun()`.
- **Native node-command generation** now goes through
  `RuntimePlugin._make_node_command_args` (template). Subclasses that emitted
  rank-specific argv lists by hand should switch to the template; the
  signatures and overall shape are documented in `runtimes/base.py`.
- **vLLM `_build_command`** moved out of `runtimes/vllm_distributed.py` and
  `runtimes/vllm_ray.py` into the new `VllmMixin._build_command()`
  (`runtimes/_vllm_mixin.py`). External subclasses inheriting the previous
  per-runtime method need to inherit from the mixin instead.

## Security fixes

- **URL-sourced recipes are never auto-trusted.** Recipes fetched from a URL
  (e.g. `@spark-arena/<uuid>` links) previously carried no `source_registry`
  and were treated as local — so their `pre_exec`/`post_exec`/`post_commands`
  hooks ran with **no confirmation**. `Recipe.is_url_sourced` now forces these
  through `--trust` / interactive confirmation, closing a "run this link" code-
  execution path.
- **Recipe fetch hardening.** `fetch_and_cache_recipe` enforces `https://`
  only (no plaintext-HTTP tampering), an allowed-host list (`spark-arena.com`;
  off-allowlist hosts require confirmation), redirect re-validation, and a
  response size cap.
- **Proxy bind host is now explicit and persisted.** A new `proxy.host` setting
  is saved and reused (recommended `127.0.0.1`). For backward compatibility an
  unconfigured proxy still binds `0.0.0.0`, but now emits a loud warning each
  start — escalated when no master key is set (unauthenticated gateway exposed
  to the network). Proxy `state.yaml` / `autodiscover.yaml` are written `0600`.
- **Arena OAuth callback** binds to a per-flow random `state` nonce and rejects
  mismatched/missing state before storing the token (token-fixation / CSRF).
- **Sudoers `cache_dir` validation.** `validate_sudoers_path()` (absolute path,
  conservative charset) gates `cache_dir` before it is interpolated into the
  passwordless `chown` sudoers rule.
- **Recipe env no longer expanded.** Control-machine environment variables are
  no longer substituted into recipe `env` values, preventing a third-party
  recipe from exfiltrating host secrets (`$AWS_SECRET_ACCESS_KEY`, ...) into a
  container.
- **Secret masking in logs.** `token`/`key`/`password`/`secret` env values are
  masked in the docker executor DEBUG output (per-var dump and full command).
- Trust gating on `pre_exec` / `post_exec` / `post_commands` (single decision
  via `resolve_recipe_trust`). Local recipes and default-registry recipes are
  auto-trusted; third-party registry recipes require `--trust` or prompt.
- `orchestration/transfer.py:_run_delegated_copy` validates `source_host`
  against the validated host list and rejects `dest` paths that escape the
  cache root.
- `cli/_setup/_sudo.py`, `_phases.py`, `_uninstall.py`, `_commands.py` all
  call `utils.shell.validate_unix_username()` before interpolating user
  identifiers into sudoers fragments.
- `runtimes/trtllm.py` no longer relaxes SSH host-key checking in the
  MPI rsh wrapper.
- `core/registry.py:_validate_git_url` allowlists `https://`, `git@`,
  `ssh://`, `file://` schemes before any `git clone` invocation.
- OAuth callback CORS allowlist restricted to `AUTH_PROXY_BASE` (no wildcard).
- Token-prefix logging removed from debug paths.
- `utils/shell.py:quote()` wraps `shlex.quote()`; every shell-command
  construction in-tree routes through it.

## Deprecations

- The legacy `resolve_ib_env(ctx, comm_env)` wrapper has been **removed**.
  Per-host communication env now flows exclusively through
  `runtimes/_cluster_ops.py:resolve_comm_env(ctx, comm_env, backends)` with a
  `backends: dict[host, BackendBundle]` map; NVIDIA-only call sites continue to
  emit identical output.
- Recipe topology fields `mode`, `solo_only`, `cluster_only` remain
  deprecated in favor of `min_nodes` / `max_nodes` (carry-over from earlier
  releases; documented in `RECIPES.md`).
