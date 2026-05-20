# Architecture (Multiplatform)

Contributor-facing map of the layers introduced for multiplatform support. Pair
with `MULTIPLATFORM.md`, `EXECUTORS.md`, and `SECURITY.md` for the deeper dives.

## Layer diagram

```
                       +---------------------------+
                       | cli.* / launcher entry    |
                       +-------------+-------------+
                                     |
                                     v
+--------------------+   +-----------+-----------+   +--------------------+
| core/recipe.py     |   | core/launcher.py      |   | core/config.py     |
| (executor field)   |-->| launch_inference()    |<--| SparkrunConfig     |
+--------------------+   |  - resolve_recipe_..  |   +--------------------+
                         |  - resolve_per_host_  |
                         |    backends           |
                         |  - resolve_executor   |
                         |  - platform validate  |
                         +-----------+-----------+
                                     |
                +--------------------+--------------------+
                v                                         v
   +-----------------------+                  +-----------------------+
   | core/hardware_probe.py|                  | core/backend_select.py|
   | probe_host (combined  |                  | select_backends() ->  |
   |  accel + IB SSH probe)|                  | BackendBundle         |
   +-----------+-----------+                  +-----------+-----------+
               |                                          |
               v                                          v
   +-----------------------+         +---------------------------------+
   | core/hardware.py      |         | orchestration/collectives/      |
   | HostHardware / Accel  |         |  base.py  (CollectiveBackend)   |
   |  Spec / capabilities  |         |  nccl.py  (NVIDIA, default)     |
   +-----------+-----------+         |  rccl.py  (AMD scaffold)        |
               |                     |  hccl.py  (Intel scaffold)      |
               v                     +---------------------------------+
   +-----------------------+
   | core/fingerprint.py   |         +---------------------------------+
   | core/placement.py     |         | orchestration/executors/        |
   | core/layout.py        |         |  _base.py (Executor ABC)        |
   +-----------------------+         |  docker.py (default)            |
                                     |  local.py  (experimental)       |
                                     |  k8s.py    (experimental)       |
                                     +---------------------------------+
                                                  ^
                                                  |
                                     +---------------------------------+
                                     | platforms/                      |
                                     |  base.py  (HardwarePlatformPlug)|
                                     |  dgx_spark.py                   |
                                     |  nvidia_generic.py              |
                                     +---------------------------------+
```

## Module map (new + reshuffled layers)

### `core/`

| Module               | Purpose                                                                                                                                      |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `hardware.py`        | `AcceleratorSpec`, `HostHardware`, `default_dgx_spark_hardware()`. Vendor/model/capability records consumed by every multiplatform path.     |
| `fingerprint.py`     | Thin shim around the combined probe. Parses accelerator KV output into a `HostHardware`. Standalone fingerprint script kept for legacy use.  |
| `hardware_probe.py`  | `probe_host()` / `probe_hosts()` — single SSH script that emits both the fingerprint section and the IB section, split by sentinel markers.  |
| `backend_select.py`  | `select_backends(host_hardware) -> BackendBundle`. Raises `NoMatchingBackendError` when no collective matches.                               |
| `placement.py`       | `compute_placement()` — maps a `ParallelismConfig` onto hosts, honoring explicit `RecipeLayout` or auto-packing single-vendor clusters.      |
| `layout.py`          | `RecipeLayout` / `Placement` dataclasses (per-rank `(host, local_gpu)`). Parsed from recipe `layout:` block.                                 |
| `launcher.py`        | `launch_inference()`. Resolves trust, per-host backends, runs the central compatibility check, threads `BackendBundle` to `runtime.run()`.   |

### `orchestration/`

| Subpath                       | Purpose                                                                                                                                  |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `executor.py`                 | Public facade. Exports `Executor`, `ExecutorConfig`, `EXT_EXECUTOR`, `get_executor()`, `list_executors()`, `resolve_executor()`.         |
| `executors/_base.py`          | `Executor` ABC, `ExecutorConfig` dataclass, `_registered_executor_names()` SAF lookup, `accelerator_vendor_for()` helper.                |
| `executors/docker.py`         | `DockerExecutor` (default). Owns `DOCKER_DEFAULTS` and the `rootless` / `auto_user` adjustment layer.                                    |
| `executors/local.py`          | `LocalExecutor` — experimental, native subprocess via `setsid`, pid/log-file lifecycle. No images.                                       |
| `executors/k8s.py`            | `K8sExecutor` — experimental draft, `kubectl run`-based. Drops Docker-only options.                                                      |
| `collectives/base.py`         | `CollectiveBackend` ABC, `UnsupportedCollectiveError`.                                                                                   |
| `collectives/nccl.py`         | `NcclBackend` — wraps `infiniband.generate_nccl_env` / `generate_ring_nccl_overrides`. Byte-identical to legacy DGX output.               |
| `collectives/rccl.py`         | `RcclBackend` scaffold (AMD). Raises `NotImplementedError`.                                                                              |
| `collectives/hccl.py`         | `HcclBackend` scaffold (Intel Gaudi). Raises `NotImplementedError`.                                                                      |
| `comm_env.py`                 | `ClusterCommEnv` dataclass. Carries per-host env produced by collective backends through `runtime.run()`.                                |
| `infiniband.py`               | Legacy Mellanox IB probe parser + NCCL env emitter. Still authoritative; `NcclBackend` delegates here.                                   |

### `platforms/`

| Module               | Purpose                                                                                                                                            |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `base.py`            | `HardwarePlatformPlugin` ABC, `EXT_PLATFORM` constant. `matches()`, `accelerator_vendor()`, `collective_backend()`, `default_image()`, `validate_host()`. |
| `__init__.py`        | `_REGISTRY` ordered list (most-specific first). `register_platform()`, `iter_platforms()`, `resolve_platform()`.                                   |
| `dgx_spark.py`       | `DgxSparkPlatform` — matches `vendor=nvidia, model=gb10`. Publishes Spark Arena image defaults. `validate_host` warns on missing `rdma:roce-v2`.   |
| `nvidia_generic.py`  | `GenericNvidiaPlatform` — catch-all for any NVIDIA accelerator. Publishes upstream images. `validate_host` warns when no NVIDIA accelerator found. |

## Resolution chains

### Executor

`orchestration/executor.py:resolve_executor()` walks (highest priority first):

1. `cli_overrides` (e.g. `-o executor=local`, `-o k8s_namespace=...`).
2. `recipe.executor` + `recipe.executor_config`.
3. `runtime.default_executor()`.
4. `executor_cls.apply_runtime_adjustments(rootless=, auto_user=)` — Docker reads
   these here; Local/K8s ignore.
5. `config.default_executor` + `config.executor_config` (`SparkrunConfig`).
6. `executor_cls.default_config()` (e.g. `DOCKER_DEFAULTS`).
7. `ExecutorConfig` dataclass field defaults.

The selected class comes from `get_executor(name)` which queries the SAF plugin
registry (`EXT_EXECUTOR = "sparkrun.executor"`) and falls back to a static
`{docker, local, k8s}` map for test paths that bypass `init_sparkrun()`.

Unknown selectors log a warning and degrade to `"docker"` (see
`ExecutorConfig.from_chain` and `_resolve_executor_name`).

### Collective backend

`orchestration/collectives/__init__.py:get_backend(vendor)` maps:

- `vendor in {"nvidia", None}` → `NcclBackend()`
- `vendor == "amd"` → `RcclBackend()` (scaffold; raises on `env_for_host`)
- `vendor == "intel"` → `HcclBackend()` (scaffold; raises on `env_for_host`)
- everything else → `UnsupportedCollectiveError`

`core/backend_select.py:select_backends(HostHardware)` derives the vendor via
`accelerator_vendor_for(...)`, calls `get_backend()`, and packages the result
into a `BackendBundle(accelerator_vendor, collective)`. `launcher.py`
materializes this once per host before calling `runtime.run(..., backends=...)`.

### Platform

`platforms/resolve_platform(host_hardware)` iterates `_REGISTRY` in order and
returns the first plugin whose `matches()` returns true. Ordering is
most-specific first: `DgxSparkPlatform` (`model="gb10"`) pre-empts
`GenericNvidiaPlatform`. `launcher.py` calls `validate_host()` per host and
logs warnings (does not raise).

## Launch flow (multiplatform additions)

`core/launcher.py:launch_inference()` order:

1. Resolve trust (`resolve_recipe_trust`).
2. SSH kwargs + transfer mode + serve port.
3. Generate `cluster_id`.
4. Resolve container image (delegated to `runtime.resolve_container`).
5. Expand `recipe.mods` into `pre_exec`.
6. Builder phase (optional).
7. **`resolve_per_host_backends(host_list, cluster=...)`** — returns
   `dict[host, BackendBundle]`. Hosts that fail to resolve are dropped from the
   dict; runtimes fall back to legacy `resolve_ib_env`.
8. **Compatibility check** — for runtimes with `requires_capability`, walk every
   host, accumulate errors, raise `IncompatibleHardwareError` if any.
9. **Platform validation** — `resolve_platform(hw).validate_host(hw)` per host;
   warnings logged, never raised.
10. Persist job metadata (includes serialized `backends`).
11. Distribution phase (container image + model + IB probe).
12. Tuning sync.
13. GGUF resolution.
14. `runtime.generate_command()`.
15. Page-cache clear.
16. `resolve_executor(...)` — single sanctioned entry point.
17. `runtime.run(..., backends=backends, executor=executor, trust=recipe_trusted)`.

## Key reference files

- `src/sparkrun/core/launcher.py:130-170` — `resolve_per_host_backends`.
- `src/sparkrun/core/launcher.py:381-422` — backend resolution + compatibility
  + platform validation block.
- `src/sparkrun/orchestration/executor.py:260-309` — `resolve_executor` chain.
- `src/sparkrun/orchestration/executors/_base.py:59-207` — `ExecutorConfig`
  dataclass and `from_chain` parsing.
- `src/sparkrun/core/backend_select.py:42-83` — `BackendBundle` + `select_backends`.
- `src/sparkrun/core/hardware_probe.py:299-416` — `probe_host` / `probe_hosts`
  and the combined script + sentinel splitter.
- `src/sparkrun/platforms/__init__.py:31-80` — registry + `resolve_platform`.

## Where new abstractions plug in

| Adding…                          | Touch                                                                                                                                   |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| A new executor                   | `orchestration/executors/<name>.py` (subclass `Executor`, set `executor_name`). SAF discovers it via `find_types_in_modules`.           |
| A new collective backend         | `orchestration/collectives/<name>.py` (subclass `CollectiveBackend`); register in `collectives/__init__._BY_VENDOR`.                    |
| A new platform                   | `platforms/<name>.py` (subclass `HardwarePlatformPlugin`); register via `register_platform(MyPlatform(), prepend=True)` if it must win. |
| A new accelerator vendor probe   | Extend `core/hardware_probe.py` (combined script), `core/fingerprint.py` (parsing), and `core/hardware.py` (vendor in `vendors`).       |
| A new runtime                    | `runtimes/<name>.py` (subclass `RuntimePlugin`); use `_make_node_command_args` template for native multi-node, set `requires_capability`. |
