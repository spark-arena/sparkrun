# Changelog

All notable changes to sparkrun are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows semantic versioning.

For the long-form 0.3.0 narrative, see [`docs/RELEASE_NOTES.md`](docs/RELEASE_NOTES.md).

## [Unreleased]

### Fixed

- Fixed model distribution transferring no weights when a host's HF cache uses
  huggingface_hub's shared blob store (1.32 and later, on by default): both
  transfer paths now pass `--copy-unsafe-links`, materialising the blob links
  that leave the model directory while keeping the in-tree snapshot links as
  links, so each blob is still stored once. Weight-existence checks require a
  readable file rather than a matching name, so a bytes-less skeleton of
  dangling symlinks is a cache miss instead of a permanent false hit that
  skipped both the download and the repair (#299).

### Added

- `sparkrun setup rdma-test` verifies the high-speed fabric that `setup cx7`
  configures — RDMA latency and bandwidth per link, plus an NCCL collective
  across the cluster. Until now this was a manual copy-paste procedure, and
  nothing in sparkrun exercised the fabric at all: `validate_ib_connectivity`
  only proves an IB IP answers SSH, which a link silently running over the
  management NIC would also do.

  Host pairs are **derived from the configured CX-7 subnets**, so only links
  that physically exist are tested (a switched segment is chained, not meshed,
  keeping coverage O(N)). Each link is measured on its own and then **every
  link between a host pair is driven concurrently**: a DGX Spark QSFP112 cable
  presents as two RDMA devices, so a per-device figure is about half the
  cable's real throughput.

  The **`perftest` suite is the default** — host-native where `ib_write_bw`
  exists, no image pull, seconds to run. `--suite all` adds the NCCL
  collective, which fetches the image onto every host and takes minutes.

  **Devices that share a physical port are detected, not assumed additive.**
  DGX Spark exposes two PCIe functions of one adapter on a single 200 Gb/s
  port, so summing their rates claims a ceiling the wire cannot carry — a
  measured 195.7 Gb/s, ~98% of the cable, was reported as "of 400 Gb/s" and
  each healthy ~112 Gb/s device warned for being under 75% of 200. Devices
  reporting the same `sys_image_guid` *and* `phys_port_name` are treated as
  sharing one wire: the aggregate expectation sums distinct ports, and each
  device is compared against its fair share of its port. Both fields come from
  sysfs, so a conventional dual-port card with two cables remains additive
  without special-casing any platform.

  Progress is reported through the shared `sparkrun.progress` logger (visible
  at default verbosity, and not a console write, so the api layer stays
  console-free): previously the command sat silent through an image pull and
  minutes of transfers.

  Severity follows the `setup check` convention — underperformance is a
  warning and exits 0; only a test that could not run at all fails. The
  bandwidth expectation is derived from the link's own reported rate rather
  than hardcoded, so the verdict is meaningful on hardware that is not a DGX
  Spark. Parsers return no measurement rather than a fabricated number when a
  run is refused or times out.

  The `perftest` suite runs host-native where `ib_write_bw` exists (DGX OS
  ships it), so the common "did my cable work?" check pulls no image. The
  `nccl` suite uses a new maintained image, `ghcr.io/spark-arena/sparkrun-rdma-test`
  (built from `docker/rdma-test/`), because building NCCL and nccl-tests takes
  ~10 minutes per node. Its tag tracks the NCCL release it was built from —
  that is the image's rebuild cadence — and the default is pinned to that tag
  rather than `:latest` so two runs a month apart stay comparable. Override
  with `--image` or `rdma_test.image` in `config.yaml`.

  Experimental, gated behind `cli.setup.rdma_test` (off on `stable`, on for
  `beta`/`alpha`) because it pulls a multi-GB image and starts containers.

  Each architecture of the image is built on a **native** machine and the
  results are combined into one multi-arch manifest, rather than
  cross-building under QEMU: the image compiles NCCL and nccl-tests with nvcc
  across four GPU architectures, which emulation turns from a ~20 minute build
  into hours. CI builds on native arm64 and amd64 runners in parallel and
  pushes by digest, then merges; `docker/rdma-test/build.sh --push` /
  `--merge` is the same flow by hand across two machines.

  The runtime stage is plain Ubuntu rather than the NGC CUDA runtime image —
  only the builder needs the toolchain. Measured on arm64: **4.3 GB → 739 MB**
  for identical contents, because the CUDA base carries ~1.9 GB of cuBLAS /
  cuFFT / cuSPARSE / cuSolver / cuRAND while the only CUDA library anything
  here links is `libcudart` (768 KB). `libnccl_static.a` and nccl-tests' `.o`
  objects are dropped in the builder, where they never reach a runtime layer.
  The full nccl-tests binary set is kept so `rdma_test(nccl_binary=…)` accepts
  any collective it names. Leaving the CUDA base means the image now declares
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility` itself; without it the
  container runtime defaults to `utility`, CUDA is absent, and the collective
  fails with nothing explaining why.

  Both NCCL and nccl-tests are pinned by git ref, and the build args are named
  `NCCL_GIT_REF` / `NCCL_TESTS_GIT_REF` rather than `NCCL_VERSION`: the NGC
  CUDA images set `ENV NCCL_VERSION` to the libnccl2 they bundle, and an
  inherited `ENV` outranks a same-named `ARG` under the legacy builder though
  not under BuildKit — so the obvious name built our pinned NCCL under `buildx`
  and NVIDIA's bundled one under `docker build`, from the same Dockerfile. The
  build now verifies the ref it cloned and that `all_gather_perf` exists, so a
  mis-resolved pin fails loudly instead of shipping a version nobody asked for.

- `sparkrun setup check` gained an `rdma` check reporting whether RDMA devices
  are present and `ACTIVE`, from the same probe `setup rdma-test` uses so the
  two cannot disagree about what hardware is there. It sends nothing over the
  fabric — "the link is configured" and "the link performs" are different
  questions — and points at `setup rdma-test` for the second. Absent
  `perftest` is reported as a note rather than a warning: it is one command
  away and the test has a container fallback, so flagging it would fire on
  hosts where nothing is wrong.

### Changed

- The `mpirun`-across-containers rsh agent moved from
  `TrtllmRuntime._generate_rsh_wrapper` to
  `sparkrun.orchestration.mpi.build_rsh_wrapper`, shared with the new RDMA
  NCCL suite. Output is byte-identical; the values it interpolates are now
  validated (they are emitted bare or double-quoted, so they cannot be
  shell-quoted without changing what bash sees).

### Security

- Registry names and asset subpaths are now contained to the registry cache.
  Both arrive from `.sparkrun/registry.yaml` manifests in **remote**
  repositories (`sparkrun registry add <url>`, bootstrap discovery) and both
  become real filesystem paths, with no charset validation anywhere before this:
  `_cache_dir` is `cache_root / name`, so an escaping name resolved outside the
  cache root and `_link_registry_to_shared` would then `shutil.rmtree` it (a
  delete primitive), while `asset_dir` is `_cache_dir(name) / subpath`, so an
  escaping subpath had `iter_asset_files` `rglob` a directory outside the clone
  and `find_recipe` offer whatever YAML it found there as a runnable recipe (a
  read primitive feeding the recipe loader). New
  `assert_safe_registry_name` / `assert_safe_registry_subpath` /
  `assert_safe_registry_entry` enforce this at every entry point: `add_registry`
  and `validate_registry_name` raise, `_discover_manifest_entries` drops the
  offending entry and keeps the rest (raising only when nothing survives, so a
  wholly hostile manifest is never reported as a successful no-op add), and
  `_load_registries_from_file` skips-with-warning — narrower than the enclosing
  `except`, which reverts to the shipped defaults and would let one hand-edited
  entry discard every registry the user has. Requiring each path component to
  start alphanumeric also rules out `.`/`..`, dotfiles, a leading `-` (git would
  read it as an option) and the `_url_<hash>` shared-clone prefix, whose
  collision would have deleted the checkout every registry on that URL shares.
  See `docs/SECURITY.md`.

### Changed

- The `atlas` registry now points at `Atlas-Inf/sparkrun-recipes`; the recipes
  moved there from `Avarok-Cybersecurity/atlas-recipes` along with the project
  itself (source now at <https://github.com/Atlas-Inf/atlas>). Existing installs
  follow the move via `MIGRATED_REGISTRY_URLS`, which rewrites the URL in place
  and keeps each user's `enabled`/`visible`/trust state — the stale clone is
  re-fetched by `_drop_cache_if_url_changed` on the next sync. The layout is
  identical on both sides (a `recipes` subpath), so nothing else changes.
  Correspondingly, `atlas-inf` replaces `avarok-cybersecurity` in
  `EXTERNAL_RESERVED_NAMES`: the reserved `atlas` registry name now belongs to
  the org that publishes the recipes, and a registry claiming it from the former
  org is rejected like any other impersonation.
- The default Atlas container image is now `azeezish/atlas-gb10:latest`
  (previously `avarok/atlas-gb10:latest`), matching what every recipe in the
  new Atlas-Inf registry pins. Only affects `atlas` recipes that do not set
  `container:` themselves.

- Manifest discovery clones blob-filtered and sparse (`--filter=blob:none
  --sparse` + `sparse-checkout set .sparkrun`) instead of pulling the whole
  repository: only `.sparkrun/registry.yaml` is ever read, so the recipe trees
  were wasted transfer on every bootstrap URL. A failed sparse-checkout raises
  rather than falling through to "No `.sparkrun/registry.yaml` manifest found",
  so a clone whose manifest directory was never materialized is not mistaken for
  a repo that declares nothing.

### Added

- `run-recipe.sh` shim: `-v/--volume LOCAL:CONTAINER` (repeatable), matching
  spark-vllm-docker upstream. It maps to `--executor-args "-v ..."`, which the
  docker executor shlex-splits back into the `docker run` argv. As upstream, it
  applies to both solo and multi-node runs (unlike `-p/--publish`, solo-only).
- `tests/test_run_recipe_shim.py` — argv-mapping coverage for the shim, driven
  through its `RUN_RECIPE_DEBUG=1` hook.
- eugr builder: support for `build-and-copy.sh`'s `--exp-b12x` /
  `--experimental-b12x` preset. As upstream, it does not set
  `CUSTOM_BUILD_REQUESTED`, so on its own it only changes *which* prebuilt image
  is pulled: a nightly `:latest` sentinel (or a missing non-pullable eugr image)
  resolves to `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:latest` — sparkrun's
  mirror of upstream's `eugr/spark-vllm-b12x:latest` — instead of the standard
  nightly. Naming a b12x prebuilt image selects the variant the same way. With a
  custom build flag alongside it the build runs under the `sparkrun-eugr-vllm-b12x`
  local tag (upstream tags `vllm-node-b12x`) and the flag is forwarded verbatim.
  Long-term image pinning resolves against the `nightly-b12x` variant.

### Added

- `sparkrun tune vllm` / `sparkrun tune sglang`: `--timeout SECONDS`, a per-TP
  ceiling on the tuning job itself. `0` (the default) means no ceiling.

### Changed

- Tuning jobs are **unbounded by default**, replacing the fixed 8-hour cap on
  both the vLLM and SGLang paths. A tuning run that is killed at the wire loses
  all of its work, and the normal runtime is measured in hours (vllm-tune's MoE
  phase alone is 1.5-3h), so completion now outranks bounded runtime. Pass
  `--timeout` to get a ceiling back.
- `run-recipe.sh` shim: `--ray`/`--no-ray` combined with `--solo` now warn and
  are ignored instead of erroring, matching upstream's
  `use_ray = args.ray and not is_solo`. The flags are recorded during parsing
  and resolved afterwards, so a trailing `--solo` suppresses them too.
- `run-recipe.sh` shim: `--earlyoom` / `--earlyoom-args` are now rejected with
  the standard "not supported" pointer rather than a bare "unknown option"
  error. sparkrun runs the server as the container foreground process, so there
  is no earlyoom supervisor to substitute.

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
