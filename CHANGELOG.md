# Changelog

All notable changes to sparkrun are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows semantic versioning.

For the long-form 0.3.0 narrative, see [`docs/RELEASE_NOTES_0.3.0.md`](docs/RELEASE_NOTES_0.3.0.md).
The 0.3.x entries below follow release-tag ancestry: changes are listed in the
first tagged release containing them, regardless of their original commit date.

## [Unreleased]

### 0.4.0 application and API changes

- Fixed model distribution transferring no weights when a host's HF cache uses
  huggingface_hub's shared blob store (1.32 and later, on by default): both
  transfer paths now pass `--copy-unsafe-links`, materialising the blob links
  that leave the model directory while keeping the in-tree snapshot links as
  links, with one copy per repo blob on the destination. Weight-existence
  checks reject dangling links, and GGUF weights/projectors are filtered before
  fallback or precision selection. Re-sync repairs previously broken targets
  without deleting the cache. These presence checks do not establish complete
  snapshots or read permissions; cross-repo deduplication remains source-side
  (#299, #300).
- Honor explicit vLLM `kv_cache_memory_bytes` as a per-GPU allocation, separate
  from context demand. Hybrid linear/MLA caches report unknown context capacity
  instead of applying MLA sizing to every layer (#297).
- Raise the default DGX Spark memory cap to 90%. Occupancy schedulers reserve
  exclusive GPUs for ordinary runs and utilization claims of at least 80%,
  leaving estimated model fit to runtime readiness. Shared allocations retain
  hard compute and memory admission limits, including solo and explicit layouts.
- Persist GPU reservations on Docker and native workers, recover them across
  status merging and partial jobs, and bind launches to their assigned devices.
  Fit summaries share the per-host budgets and report why placement was rejected.
- Identified GB10 hosts use the platform's 121 GB planning capacity when recorded
  probes lack memory measurements. Scheduling and fit share this fallback and
  utilization caps; explicit inventory capacity takes precedence.
- Fixed runtime-cache pruning to resolve named/default clusters and their SSH
  users through the shared host context. Empty monitor tables render correctly;
  selecting the dormant `nv-monitor` backend without assets fails before startup.
- Tightened profile defaults, required recipe/setup inputs, Kubernetes config and
  log-stream contracts, and shared read-only inputs. SparkRoute now passes launch
  overrides explicitly instead of attaching private fields to recipes. Reduced
  the full-source type baseline without adding suppressions or private imports.

- Enforced required runtime command hooks and nonempty generated commands;
  builders now use one `prepare()` hook. Removed the unused image-staging API
  with competing transport inputs.
- Typed model/container distribution entries and cluster update sentinels;
  readiness and job host validation preserve their resolved types. Added a
  full-source Pyright baseline gate to prevent renewed type debt.
- RDMA readiness treats extra inactive ports as informational when active
  links are available; inactive RDMA on an up, configured interface still warns.
  The check distinguishes link state from connectivity/performance validation.
- NaN Hub metadata budgets retain the default deadline. Directory plugin API
  mismatches produce concise update instructions, with tracebacks in debug logs; completion
  suppresses plugin-loading diagnostics.

- Corrected public API, benchmark process, placement, and sudo result type
  contracts. Core launch and indirect sudo reject missing required context
  before execution; configuration files require a top-level mapping.

- Completed native benchmark asset handling: image provenance comes from launch
  or captured deployment context. `materialize()` explicitly targets containers.
- Local and remote distribution honor the same resource entries and node targets,
  including required helper images and prepared policies.
- Tool-eval-bench rejects infrastructure-only/empty measurements while preserving
  real zero scores and partial completion. Unmeasured tasks remain resumable.
  Text arguments retain their values; standalone modes without scheduled JSON
  artifacts fail before launch. Benchmark task-building hooks are now abstract.

This is a breaking Python API release. See the
[0.4.0 migration guide](docs/DISTRIBUTION_API_MIGRATION.md) for removed imports,
fields, and downstream migration steps. The existing library API, in-tree plugin
system, gateway registry, and execution-strategy extension points shipped in
0.3.x; the changes below build on those foundations.

- Removed obsolete recipe/fingerprint/networking/teardown aliases and private
  benchmark wrappers. Benchmark dispatch resolves one resume mode; scheduled
  progress uses task events. Plugin modules now declare API version 1 across
  every loading source, and IB probes accept per-host candidate lists only.
- Unified runtime image selection across preparation and materialization, using
  per-host platform defaults while preserving explicit recipe images and
  homogeneous-runtime/builder constraints. SparkRoute uses required 0.4 host
  initialization directly. Published a separate [CLI and saved-data migration
  plan](docs/LEGACY_MIGRATION_PLAN.md) before retiring recovery support.
- Added immutable application profiles and profile-aware, console-free
  initialization for CLI, daemon, desktop, and other Python frontends. Plugins
  can read application and controller identities; one controller represents an
  application/config directory.
- Unified installed plugin discovery under `sparkrun.plugins`, with API version
  checks, explicit contribution ownership, and registration rollback on failure.
- Separated benchmark frameworks, publication integrations, immutable measurement
  snapshots, and resumable state/finalization. Completed measurements can retry
  publication without relaunching inference; execution provenance and credentials
  have separate persistence rules. Exported artifact references are absolute and
  remain usable when recovery runs from another working directory.
- Moved Kubernetes-specific APIs and configuration into `sparkrun.plugins.k8s`.
  Hardened destination-aware run/status/stop, setup, and monitoring contracts.
- Added typed gateway model queries, a public supervisor import, and optional
  console/token protocols with specific operational errors. Failed enumeration
  raises `ProxyQueryFailed` and makes `proxy models` exit nonzero, while status
  retains process diagnostics. Maintained providers implement these contracts directly. Gateway selection
  without an explicit context initializes plugins and honors the saved pin.

### Added

- Bundled the SparkRoute gateway integration, pinned to the verified 0.1.1
  snapshot. Alpha defaults to SparkRoute; stable/beta retain LiteLLM. An explicit
  enabled gateway pin takes precedence. The integration includes an admin
  console, managed credentials, recipe catalogs, and durable on-demand activation.
- Added cached recipe catalogs, registry/capacity operations, and richer proxy
  endpoint metadata for controller clients, including named clusters, recipe
  revisions, and native runtime APIs. Recipe plugins can register passive items.
- Added `setup plugins list` and JSON output for plugin/feature inventory.

### Changed

- Upgraded tool-eval-bench to v2.6.0, with native JSON-file results, current
  scenario/argument handling, and one resumable task per suite. All benchmark
  frameworks now use scheduled execution; removed the single-call fallback and
  internal `BenchmarkResult` alias. Published typed progress event fields.
- Prepared images supersede unused defaults and obey final runtime constraints.
  Local executors no longer require image defaults; generated image transfers
  stay local to each preparation instead of mutating reusable recipes.

- Made proxy CLI help and documentation gateway-neutral. Load/unload use the
  shared run/stop APIs, preserve cluster targeting, and update recipe bindings;
  restart waits for the preceding process to exit.
- Docker defaults to a bundled seccomp profile allowing io_uring. Controller-local
  custom profiles travel with generated launch commands to every node; explicit
  policy choices remain supported. vLLM no longer supplies `OMP_NUM_THREADS=4`.
- RDMA `perftest` is enabled on every feature channel. NCCL and `all` suites now
  have their own `cli.setup.rdma_test.nccl` gate, enabled by default only on alpha;
  help and API refusal follow the same suite-availability rules.
- Hardened native executor ownership, process-group, and PID persistence
  contracts, including rollback after failed record writes. Managed native
  launch requires detached mode and anchored control paths.
- Added offline documentation/import/help checks and installed downstream
  application/plugin compatibility tests. Documentation tests also collect from
  source archives. Generated CI runs pinned Ruff lint and Python formatting checks.
  Pinned static consumer checks verify catalog types in editable and installed-wheel
  environments, including diagnostics for invalid API usage.

### Fixed

- Embedded CLI invocations validate configuration binding independently of cached
  plugin discovery. Conflicting paths fail before dispatch or writes.
- Job APIs and metadata helpers honor the configured cache with omitted contexts,
  sharing one lookup policy without requiring plugin bootstrap for passive reads.
- LiteLLM uses typed gateway model queries directly. Removed the legacy provider
  dictionary hook and mutable query-error side channel; external gateways use
  `query_models()` and `GatewayQueryError`. Public status diagnostics remain.

- Gateway updates and stop hooks consistently translate declared provider errors,
  including alias reconciliation. The updated upstream SparkRoute plugin implements
  the shared error and typed model-query contracts directly, without a host adapter.
- Process-only gateway recovery refuses model mutations before discovery with
  `GatewayUnavailable`. Alias commands report persisted edits separately from
  failed live application; stopped follow/sync operations retain their no-op behavior.
- Gateway management initializes application plugins before resolving the running
  implementation, while preserving process status/stop if bootstrap fails or a
  plugin is unavailable. Recorded gateway identity takes precedence over a saved
  preference for the next start. CLI status/stop reach the same recovery path,
  and unavailable model queries fail explicitly after bootstrap errors.
- Gateway start translates declared provider operational failures into
  `ProxyStartFailed` across construction, preparation, shutdown, and launch,
  preserving causes and normal CLI diagnostics.
- CLI context creation and extension discovery use the shared application
  initializer and bind custom configuration before plugin loading. Quiet mode
  no longer supplies an invalid negative progress-verbosity value.
- Rejected proxy starts and timed-out restarts preserve generated configuration
  and discovery snapshots. Successful fileless gateway starts return `None` for
  `config_path`, matching dry-run results.
- Preserved managed-import distrust across catalog, run API, CLI, and benchmark
  loaders, including symlinked paths. Shared source tagging retains selected
  registry identity. Automatic trust requires the recorded registry URL to match
  the enabled, trusted entry; older recipes without that provenance need reloading
  or explicit trust.
- Public recipe planning and execution wrap known recipe-decoding/resolution
  failures in `SparkrunError`, retaining their cause and preserving interrupts.
- Repaired tab-broken shell continuations, including recipe hook commands.
- Status reports without an explicit context use the application's configured
  cache for pending operations and job metadata. Intent matching respects the
  requested hosts and their order when given a wider status snapshot.
- Offline recipe searches and catalog listings skip first-run manifest discovery.
  Search and filter operations reuse the context's registry manager, and an
  offline read preserves later explicit initialization or refresh.
- Explicit registry refresh retries unfinished bootstrap discovery, including
  after partial success or a local configuration edit, and reports failure when
  no inventory can be established. Configuration edits stay offline and retain
  their precedence over late discovery.
- Catalog preview resolves a selection once and uses the context's registry
  manager for trust. Preview, resolution, import, and retention skip bootstrap
  discovery. Unrepresentable numeric metadata is omitted without aborting browsing.
- Shared-clone catalog selections retain registry provenance and trust across
  qualified names, references, and paths. Disabled, ambiguous, and orphaned cache
  sources no longer fall back to trusted local identity. Catalog-resolved recipes
  retain registry URLs for benchmark attribution.
- Catalog operations reuse one inventory snapshot per call. Malformed core recipe
  fields are isolated per file during browsing and reported as typed errors during
  exact resolution.

## [0.3.8] — 2026-09-07

### Added

- Added rank-local Docker-start readiness measurements: port-open and HTTP-ready
  time-to-ready plus first-token time. Runtime styles and executor observation
  capabilities determine when these measurements are applicable; recipes can
  override readiness policy.
- Benchmarks collect and export the measured startup timings. End-of-launch
  summaries repeat Docker-start TTR/TTFT and avoid duplicating those spans in the
  timing tree.
- Added experimental `setup rdma-test` for per-link latency/bandwidth and an NCCL
  collective. Host pairs derive from configured CX-7 subnets; links are tested
  separately and concurrently. Shared physical ports are detected from sysfs so
  two PCIe functions on one cable do not double the bandwidth expectation.
- RDMA tests default to host-native `perftest` where available. NCCL uses a
  maintained image with pinned NCCL/nccl-tests refs, built natively for arm64 and
  amd64. Its slim runtime keeps the required CUDA library and collective binaries.
  Underperformance warns; inability to run fails. In this release the entire
  command is gated off on stable and enabled on beta/alpha.
- `setup check` reports RDMA device presence and ACTIVE state, using the same
  hardware probe as the performance test. The MPI rsh-wrapper helper is shared
  by TRT-LLM and RDMA testing, with validation of interpolated values.

### Fixed

- Remote Hugging Face cache checks inspect the actual configured cache directory
  and honor the requested model revision.
- Pinned the uv version installed on cluster hosts for repeatable tooling setup.
- Eugr's successful pull-first image path no longer reports failure.
- Runtime metadata reports the NCCL library that executes collectives.
- Pure data-parallel vLLM no longer waits for a port that no process binds.

### Security

- Quoted and validated recipe-controlled values in generated shell scripts so
  they are passed as data instead of executable shell fragments.

## [0.3.7] — 2026-09-04

### Added

- Made the inference gateway an implementable plugin extension point, with native
  protocol and recipe-capability declarations. This is the gateway foundation;
  the bundled SparkRoute implementation follows in 0.4.0.
- Added executable host sessions for transports, per-machine container images,
  shared image preparation, recipe-local execution strategies, and the
  `api.materialize()` view of resolved launch units. Plugins can own top-level
  recipe items and declare default registries; clusters can carry local plugin policy.
- Added per-node pull distribution. Model transfers retain each model's own
  revision, and tuning configs follow the same transfer preferences as models.
- Added structured launch-stage timing, time-to-first-inference observation
  alongside log streaming, serving intervals, and verbosity-aware timing detail.
  Progress supports heartbeats and dynamic labels for long-running host scripts.
- Added three-tier recipe validation with launch-time mount checks, deprecation
  and advisory reporting, unused override diagnostics, and warnings for inline
  patching, pinned launch flags, and model revisions that do not reach the engine.
  V2 recipe top-level `name` is deprecated. Arena validates recipes and requests
  confirmation before submission.
- Accelerator driver versions are detected and persisted; `cluster inspect`
  shows head-node hardware. VRAM estimation recognizes EXL2/EXL3 weight formats.

### Changed

- Docker defaults to `ipc=shareable` instead of host IPC. More permissive IPC
  choices are trust-gated according to their value.
- SGLang expresses data parallelism through its runtime configuration. Runtime
  cache coverage includes torch and TVM-FFI; SGLang draft-model revisions are pinned.
- Enforced the selected Ruff bugbear rules after fixing the existing violations.

### Fixed

- Concurrent proxy/auto-discovery writes merge their changes to `proxy.yaml`
  instead of overwriting one another.
- Launch records the recipe fingerprint before execution and preserves it when
  injecting resolved revisions. Strict replacement refuses to proceed when the
  prior workload cannot be stopped as required.
- Stop/log operations retain the launching cluster and connection context;
  status preserves the default cluster and attributes pending operations to hosts.
- Successful rsync transfers no longer report failure. Attribute-related errors
  retry once with relaxed attributes; destructive tuning synchronization is
  guarded, and remote tuning directories are created before Docker can own them.
- CX-7 address persistence follows the service managing it; missing default routes
  no longer fall back to a nonexistent `eth0`.
- Hugging Face metadata lookup is bounded so launch cannot hang indefinitely.
  Rootless write failures get useful log diagnostics and transport sessions can
  perform ownership repair.
- Atlas forwards recipe options not enumerated by its flag map. Eugr recognizes
  Docker Hub short references as pullable. vLLM preserves served model names when
  rewriting model references to local paths.

## [0.3.6] — 2026-08-24

### Added

- Proxy discovery advertises each served model's context window and preserves it
  across config rewrites.
- vLLM, SGLang, and TRT-LLM receive the `SYS_PTRACE` capability by default.

### Changed

- Migrated the Atlas registry to `Atlas-Inf/sparkrun-recipes`, preserving user
  enabled/visible/trust settings. Updated the reserved organization and fallback
  image to `azeezish/atlas-gb10:latest`.
- Manifest-only discovery uses blob-filtered sparse clones of `.sparkrun`.
  Sparse-checkout failures are reported rather than mistaken for missing manifests.
- Benchmark identity and state ownership include the measured node set. Hid the
  `--exit-on-first-fail` CLI option pending further policy work.

### Fixed

- Port readiness checks require a LISTEN socket, including their fallback path,
  instead of treating other TCP states as a ready inference service.
- FlashInfer JIT and CuTeDSL caches persist across launches.
- Self-update determines whether an upgrade occurred from installation identity,
  rather than treating every successful uv exit as an upgrade.

### Security

- Confined manifest-derived registry names and asset subpaths to the registry
  cache. Rejects traversal, reserved-prefix collisions, and option-like names
  before clone/link/delete or recipe discovery; invalid persisted entries are
  skipped without discarding the remaining registry configuration.

## [0.3.5] — 2026-08-18

### Added

- Introduced the in-tree plugin system for cross-cutting integrations and the
  gated `uv-venv` environment builder.
- Added cluster GPU SM-clock reporting/capping and per-architecture KV-cache
  sizing extensions. MLA estimates use compressed latent dimensions, including
  DeepSeek V4 and `nvfp4_ds_mla`; command-template KV dtype is recognized.
- Added live workload completion and stale-job detection for `logs`/`stop`,
  readable completion targets, launch timestamps, and bounded metadata retention.
- Added `--timeout SECONDS` to vLLM/SGLang tuning. The default is now `0`
  (unbounded), replacing the eight-hour cap; tunable recipes are selected by
  runtime family.
- Persisted runtime compile/autotune caches, including SGLang's own cache root.
  Added `run --rebuild` support for a fresh pull of registry images and expanded
  vLLM/SGLang flag maps so more recipes can omit custom commands.

### Changed

- Registry configuration has explicit two-way trust and a versioned migration
  marker. Git URLs are canonicalized without mutating shared default settings.
- FE system updates run hosts in parallel, update the control node last, and
  report progress while the system updater works.

### Fixed

- Benchmark runs honor host targeting, serving overrides such as `--tp`, and
  hook trust on Arena runs. `--skip-run` targets the running workload rather than
  every configured cluster host.
- Finalized benchmark arguments before schedule creation; omit unsupported
  SGLang streaming `return_token_ids` and fail measurements that produce no data.
- Stop and log post-mortems dispatch through the executor that launched the job.
  NOPASSWD sudo runs noninteractively over SSH; failed remote scripts surface
  stdout when stderr is empty.
- Recipe served-model-name is resolved from command templates. MLA estimates are
  idempotent and handle compressed-cache edge cases correctly.

## [0.3.4] — 2026-08-09

### Added

- Detect and refuse image ENTRYPOINT configurations that would swallow the
  requested serving command.

### Fixed

- Plan placement once, delay eviction until the replacement is ready to launch,
  and include the container image in workload intent identity.
- `proxy load` waits for the inference endpoint to answer before synchronizing it
  into the gateway.
- Telemetry dimensions use scalar values instead of object representations and
  fail closed for invalid configuration objects. Tests detect collector access.

## [0.3.3] — 2026-08-06

### Added

- Added `run -e/--env KEY=VALUE` for container environment overrides.
- Platforms can supply executor configuration and container environment defaults.
- Added `fp4` model dtype support.

### Changed

- Moved the Eugr registry to the Sparkrun mirror with migration of existing
  configurations. Renamed the registry `Visible` column to `Stemless` to explain
  its effect on unqualified recipe lookup.

### Fixed

- Resolved scheduler placement onto the configured initialization network.
- CDI setup diagnostics use the cluster's GPU access mode to choose severity.
- Self-update commit messages no longer carry a bare `g` prefix.
- Isolated tests from live user state and network access.

## [0.3.2] — 2026-08-05

### Added

- The `run-recipe.sh` compatibility shim accepts repeatable `-v/--volume`
  mappings for solo and multi-node launches.
- Eugr supports `--exp-b12x` / `--experimental-b12x`, including the B12x nightly
  mirror, explicit variant images, custom-build tags, and long-term image pinning.

### Changed

- The compatibility shim warns and ignores Ray flags in solo mode, regardless of
  argument order. Unsupported earlyoom flags use the normal migration guidance.
- Pinned the repository CI tooling revision.

### Fixed

- Recipe substitution handles placeholders inside legacy brace escapes and
  literal JSON braces in V2 commands/hooks. Escape conventions are detected from
  templates rather than inferred from the recipe version.
- Cancelling a launch no longer leaves its remote SSH work orphaned.

## [0.3.1] — 2026-07-30

### Added

- Detect stale NVIDIA CDI specifications and explain CDI-related launch failures.
- Version tags create GitHub releases with built distribution artifacts.

### Fixed

- Quoted remote log-reading commands so SSH preserves the intended shell command.

## [0.3.0] — 2026-07-30

The largest release since 0.1: multiplatform foundations, a console-free
library API, pluggable placement, and a security pass across recipe trust.
The shipped DGX Spark path is preserved — NCCL output, container launch, and
the post-launch lifecycle are unchanged for existing recipes.

### Highlights

- **`sparkrun.api`** — a console-free public library API. It never writes to
  stdout/stderr, never calls `sys.exit()`, and does not import `sparkrun.cli`.
  The CLI is now a renderer over it, so anything the CLI does is reachable
  from Python.
- **Pluggable schedulers.** `occupancy-sparse` (the new default for new
  clusters) spreads workloads onto least-loaded hosts using live cluster
  occupancy; `occupancy-dense` bin-packs; `greedy` is the 0.2.x behavior.
  Clusters with no `scheduler` key keep `greedy`, so upgrading never silently
  moves existing workloads.
- **Unified status.** All "what's running where?" flows through `api.status`,
  which sweeps every enabled executor on a cluster's substrate and merges. A
  native `local` workload and a docker container are invisible to each other's
  introspection; both now appear in `status`, `cluster monitor`, and
  `stop --all`.
- **Multiplatform seams.** `HostHardware` / `AcceleratorSpec` /
  `CollectiveBackend` / `HardwarePlatformPlugin` are the abstractions future
  vendor support builds against. NVIDIA remains the only fully wired vendor;
  AMD (RCCL) and Intel Gaudi (HCCL) ship as scaffolds.
- **Feature flags.** Channel-aware gating (`stable` / `beta` / `alpha`) for
  experimental capabilities, managed with `sparkrun setup features`.
- **Security.** URL-sourced recipes are never auto-trusted, recipe `env` is no
  longer expanded, and container-escape surfaces (bind-mounts, privileged
  `executor_config` keys, executor selection) are trust-gated.

### Added

#### Library API

- `sparkrun.api`: `run`, `stop`, `stop_all`, `logs`, `status`, `status_report`,
  `schedule`, `list_jobs`, `search_recipes`, `resolve_recipe_filter`,
  `benchmark`, `resume_benchmark`, `live_monitor` / `open_live_monitor` /
  `open_telemetry`.
- Sub-namespaces `api.proxy`, `api.setup`, `api.tailscale`, `api.k8s`, each
  with `_ops.py` + `_errors.py` + a re-exporting `__init__.py`.
- Typed error hierarchy under `SparkrunError`: `InsufficientCapacity`,
  `LayoutRequired`, `RecipeNotFound`, `InvalidRegistryFilter`,
  `HostsUnreachable`, `JobNotFound`, `AmbiguousWorkload`, `TrustRejected`,
  plus the benchmark family. Several carry structured detail (e.g.
  `InsufficientCapacity.status` / `.host_list` / `.required`).
- `SparkrunContext` (`sctx`) threaded through every entry point so chained
  calls share config, registry manager, and cluster manager.
- `api/_hosts.py:resolve_effective_hosts` — the single placement authority,
  used by `api.run`, the benchmark flow, and the CLI.

#### Scheduling and placement

- `schedulers/` package: `greedy`, `occupancy-sparse`, `occupancy-dense`,
  discovered via SAF (`EXT_SCHEDULER`).
- `--scheduler` on `run` and `benchmark`; `scheduler:` on recipes; `--scheduler`
  on `cluster create` / `cluster update`.
- `core/limits.py` — per-accelerator usable-memory cap for scheduling and fit
  (`--max-gpu-mem-util`, cluster `accelerator_memory_limits`, platform default;
  GB10 → 0.85). Distinct from the serving `--gpu-memory-utilization`.
- `core/layout.py` — recipe `layout:` block pinning ranks to hosts and local
  accelerator indices, honored verbatim by every scheduler and **required** on
  multi-vendor clusters. Experimental.
- `core/cluster_status.py:ClusterStatus` — data-only occupancy snapshot.

#### Multiplatform

- `core/hardware.py`, `core/hardware_probe.py` (combined accelerator + IB probe
  in one SSH round-trip), `core/backend_select.py`, `core/placement.py`.
- `orchestration/collectives/`: `CollectiveBackend` ABC, `nccl` (default,
  byte-identical to legacy DGX output), `rccl` / `hccl` scaffolds.
- `platforms/`: `HardwarePlatformPlugin` + `dgx_spark`, `nvidia_generic`.
  Order-sensitive in-process registry (most-specific `matches()` wins).
- `runtimes/compatibility.py` — `requires_capability` gate, raising
  `IncompatibleHardwareError` before any side effects.

#### Executors, transports, plugins

- `orchestration/executor.py` facade + `orchestration/executors/`:
  `docker` (default), `local` (experimental), `k8s` (experimental draft).
- `resolve_executor()` — one layered chain (CLI → recipe → cluster → runtime →
  config → baseline). Recipe `executor:` / `executor_config:`;
  `SparkrunConfig.default_executor` / `executor_config`; `cluster create
  --executor` / `--executor-opt`.
- `transports/` — connectivity seam (`Transport`, `EXT_TRANSPORT`), `ssh`
  default with a no-op `prepare()`, plus `cleanup_cluster` for teardown.
- `core/external_plugins.py` — out-of-tree plugin loading from `plugins.paths`,
  scanning the SAF plugin bases and calling an optional `register(v)` hook.
- `cli/ext.py` — `register_cli_command()` + `PluggableGroup`, letting plugins
  attach Click commands to the built-in tree.
- `orchestration/telemetry/` — `TelemetryProvider` per status scope, powering
  `api.live_monitor`.

#### Feature flags and telemetry

- `core/features.py` — `FeatureFlag` registry with per-channel defaults;
  resolution order env → config → channel → baseline → fail-closed.
  `SPARKRUN_FEATURE_<NAME>` env override.
- `sparkrun setup features list|enable|disable|reset`.
- Plugin self-gating via `required_feature_flag`.
- Anonymous, opt-out telemetry (`telemetry/`), `sparkrun setup telemetry`,
  `SPARKRUN_NO_TELEMETRY` per-process override.

#### CLI

- `sparkrun setup check` — non-destructive readiness probe (docker, CDI,
  earlyoom, sudoers, SSH mesh, CX7), with `--json`.
- `sparkrun setup tailscale` (gated) — join hosts to a tailnet and publish an
  inference endpoint, via JIT-minted tagged OAuth keys.
- `sparkrun setup k8s` (gated) — kubectl acquisition, cluster info, least-priv
  service account, Kueue install, JobSet launch.
- `sparkrun cluster import svd|eugr`.
- `sparkrun registry trust|untrust`, `registry add --trust`.
- `sparkrun recipe update`; `--json` on recipe list/search and status.
- `@registry` scope accepted in `list` / `search` queries.
- `logs`: `-f/--follow`, `-a/--all-sources`, `-n/--lines` (default: all).
- `run`: hidden `--scheduler`, `--rebuild`, `--executor-args`, `--dp`.
- `benchmark` categories (`performance` / `perf`, `tools`), `--resume`,
  `--arena`, `--api-key-env`.
- `run-recipe.sh` compatibility shim for the spark-vllm-docker workflow.

#### Runtimes and builders

- `tokenary` runtime — native multi-node TP over its own NCCL bootstrap;
  embedding-only and NER workloads.
- `modular-max` runtime — single-node; `tensor_parallel` maps to local
  `--devices` rather than to host count.
- Recipe-settable `command_binary` for self-dispatching images.
- `--rebuild` / `builder_config.rebuild`.

#### Other

- Absolute-path `model:` plus `Executor.verify_mount_sources()` — pre-placed
  weights are verified on the *targets* before the launch commits to skipping
  download and distribution.
- Cluster-level `env` + `env_file` with `${VAR}` references resolved at launch
  from that file only, plus `sync_source` import provenance.
- `ssh.max_parallel_ssh` (default 20) capping every parallel SSH/rsync fan-out.
- NVIDIA CDI spec generation in the wizard; GPUs requested via CDI rather than
  `--gpus`.
- SSH access bootstrap in the wizard (`api/setup/`), written to run on a bare
  Windows control machine — no local `bash`, no paramiko, key material over
  stdin rather than argv.

### Changed

- **Status discovery** is cluster-scoped and cross-executor
  (`query_status_for_cluster`); `Executor.status_scope` names the substrate.
  `stop --all` and the proxy's endpoint discovery use the same source.
- **Teardown reports honestly.** `docker rm -f … || true` made stop exit codes
  meaningless; teardown now verifies and counts what it removed.
- **The proxy applies changes by config regeneration + managed restart.**
  LiteLLM's `/model/new` / `/model/delete` need a DB-backed model store; the
  management API is used read-only. The restart is skipped when the desired
  model set already matches disk. Gateway selection added
  (`gateway.litellm` flag, `proxy.gateway` pin) behind `api/proxy`.
- **Registry assets are data, not four code paths.** Recipes, benchmark
  profiles, tuning configs, and mods share `RegistryAsset` +
  `find_asset_in_registries` / `iter_asset_files`, so listing and lookup can
  never disagree. Flat beats nested *per registry*; `.yaml` beats a same-stem
  `.yml` *per directory*.
- **All built-in default registries ship trusted**, including `eugr` and
  `atlas`. The legacy-config trust migration derives from
  `FALLBACK_DEFAULT_REGISTRIES` rather than `BOOTSTRAP_REGISTRY_URLS`, so a
  newly-trusted default reaches upgrading users and not just fresh installs.
- **The `eugr` builder is pull-first**, following upstream eugr's July 2026
  move to published images. Sentinel `:latest` refs resolve to the
  authoritative nightly and are pulled; `build-and-copy.sh` runs only when
  `build_args` request a wheels or custom build.
- `RuntimePlugin.run()` gains a keyword-only `backends:` map. Native
  multi-node runtimes emit rank-specific argv via
  `RuntimePlugin._make_node_command_args`.
- vLLM command generation centralized in `runtimes/_vllm_mixin.py:VllmMixin`.
- Recipe log locations are data (`core/log_source.py`), so `sparkrun logs`
  finds output that a plain `docker logs` would miss for
  sleep-infinity + `docker exec` workloads.
- Thunder Compute transport moved out of core into an out-of-tree plugin.
- Contributions target `develop-next` (see `DEVELOPERS.md`).

### Fixed

- **Windows control machines**: scripts piped to a remote shell were
  CRLF-mangled; job metadata and proxy configs were never written; registry
  shared-clone symlinks failed without elevation.
- `auto_port` no longer moves a workload's identity, so `stop` / `logs` still
  resolve it.
- A relaunch evicts the deployment it told the scheduler it replaces.
- Occupancy schedulers no longer count the launching workload's own intent
  against its capacity.
- `cluster monitor`: a host that hangs before its first sample reconnects, and
  the last sample is dropped when a host disconnects.
- Recipe resolution never silently guesses between same-stem recipes;
  ambiguity errors list path-qualified candidates.
- Benchmark identity is tied to recipe content, so an edited recipe starts a
  new run rather than silently resuming state measured against other settings.
- Per-host comm env is pinned to the init network's verdict.
- `HF_HUB_CACHE` is set alongside `HF_HOME` so hf-hub clients find the mounted
  cache.
- The detached serve launches via `docker exec -d` rather than in-shell
  `nohup`.
- CX7 setup pins a coherent port pair on multi-port hosts.
- `-o distributed_executor_backend` overrides a literal command flag.
- v1 recipe brace escapes are collapsed; non-string defaults are guarded.

### Removed

- `resolve_ib_env()` — per-host communication env flows exclusively through
  `resolve_comm_env(ctx, comm_env, backends)`.
- `RuntimePlugin.executor` property and the hardcoded `_KNOWN_EXECUTORS` set.
- The deprecated `placement` module (migrated to `scheduler`).
- The Atlas single-node `validate_recipe` restriction. It was never load
  bearing — `recipe.validate()` issues surface as warnings — so it only
  printed a false claim on working multi-node launches.

### Deprecated

- Recipe topology fields `mode`, `solo_only`, `cluster_only` — use `min_nodes`
  / `max_nodes`.
- The `eugr-vllm` runtime, superseded by the `eugr` builder.

### Security

- **URL-sourced recipes are never auto-trusted.** Recipes fetched from a URL
  (including `@spark-arena/<uuid>` links) previously carried no
  `source_registry` and were treated as local, so their hooks ran with no
  confirmation — a "run this link" code-execution path.
- **Recipe fetch hardening**: https-only, host allowlist, redirect
  re-validation, response size cap.
- **Container-escape surfaces are trust-gated**
  (`_enforce_recipe_mount_trust`): host bind-mounts, the `executor_config`
  privilege keys (`privileged`, `cap_add`, `security_opt`, `devices`, `user`,
  `volumes`), and executor selection (untrusted → `docker` only). A denylist
  refuses the host root, the docker socket, SSH keys, and kernel
  pseudo-filesystems regardless of trust.
- **Recipe `env` is no longer expanded**, preventing a third-party recipe from
  exfiltrating control-machine secrets into a container it controls.
- **Telemetry sends a model identifier only for a confirmed-public HF repo.**
  Everything else becomes a coarse placeholder that records why it was
  withheld — `<hf-private>`, `<local-path>`, `<unknown-visibility>` — so a
  private repo id or a path to local weights never leaves the machine. The
  verdict reads the Hub's `private` / `gated` flags rather than inferring from
  whether a fetch succeeded, since an ambient `HF_TOKEN` resolves a user's own
  private repos. It fails closed: offline, rate-limited, and skipped-detection
  all yield `<unknown-visibility>`. Opted-out users never trigger the lookup —
  the enablement check moved ahead of event construction in
  `emit_run_telemetry` / `emit_benchmark_telemetry`.
- **Trust is a per-registry local decision** stored in the user's
  `registries.yaml`; a repository manifest cannot grant itself trust.
- Proxy bind host is explicit and persisted; an unconfigured proxy still binds
  `0.0.0.0` for compatibility but warns loudly, escalating when no master key
  is set. Proxy state files are `0600` and refuse to follow symlinks.
- Arena OAuth callback bound to a per-flow `state` nonce; CORS allowlist
  restricted to `AUTH_PROXY_BASE`.
- `validate_unix_username()` and `validate_sudoers_path()` gate every sudoers
  interpolation.
- Token/key/password/secret env values masked in docker executor DEBUG output.
- `runtimes/trtllm.py` no longer relaxes SSH host-key checking.
- `_validate_git_url` allowlists `https://`, `git@`, `ssh://`, `file://`.
- `utils/shell.py:quote()` wraps `shlex.quote()`; in-tree shell construction
  routes through it.

[0.3.0]: https://github.com/spark-arena/sparkrun/releases/tag/v0.3.0

[Unreleased]: https://github.com/spark-arena/sparkrun/compare/v0.3.8...develop-next
[0.3.1]: https://github.com/spark-arena/sparkrun/releases/tag/v0.3.1
[0.3.2]: https://github.com/spark-arena/sparkrun/releases/tag/v0.3.2
[0.3.3]: https://github.com/spark-arena/sparkrun/releases/tag/v0.3.3
[0.3.4]: https://github.com/spark-arena/sparkrun/releases/tag/v0.3.4
[0.3.5]: https://github.com/spark-arena/sparkrun/releases/tag/v0.3.5
[0.3.6]: https://github.com/spark-arena/sparkrun/releases/tag/v0.3.6
[0.3.7]: https://github.com/spark-arena/sparkrun/releases/tag/v0.3.7
[0.3.8]: https://github.com/spark-arena/sparkrun/releases/tag/v0.3.8
