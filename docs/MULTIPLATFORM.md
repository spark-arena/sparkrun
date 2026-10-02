# Multiplatform

How sparkrun models heterogeneous accelerator fleets and what it takes to add a
new platform or collective backend.

## The model

### `HostHardware` (`core/hardware.py`)

A single host's accelerator metadata:

```python
@dataclass
class HostHardware:
    accelerators: list[AcceleratorSpec]
    fingerprint: str | None
    notes: str
    ib_info: dict | None
    driver_versions: dict[str, str]
```

- `accelerators` is a list of `AcceleratorSpec(vendor, model, count, memory_gb,
  capabilities: frozenset[str])`. Multiple entries cover heterogeneous-on-one-host
  scenarios (e.g. an Apple M5 host with a discrete NVIDIA GPU).
- `vendors` returns the set of distinct vendors across `accelerators`. Used by
  the placement engine to decide whether a host is single-vendor.
- `has_capability("rdma:roce-v2")` walks every accelerator on the host.
- `ib_info` is the raw KV output from the IB section of the combined probe.
  Populated by `hardware_probe.probe_host`. Not serialized to cluster YAML.
- `driver_versions` stores detected host driver versions by vendor (currently
  NVIDIA) and is serialized when present. It is software compatibility
  metadata and therefore is not part of the stable hardware fingerprint.

Missing inventory is resolved centrally by `resolve_hardware()` through the
application's `hardware_fallback` policy. Built-in Sparkrun retains the explicit
`dgx-spark` compatibility policy; alternate applications default to
`require-metadata`. A platform may provide `assumed_hardware()` for a qualified
legacy assumption. Such hardware has `source: assumed`, no fingerprint, and must
not advertise unobserved fabric capabilities. The Spark assumption is one GB10
with CUDA and unified memory; it does not claim RoCE availability. Detected
inventory has `source: detected`; older/configured inventory defaults to
`source: inventory`. See [application hardware policy](#application-profiles-and-hardware-integrations).

`default_dgx_spark_hardware()` and `resolve_fallback_hardware()` remain compatibility
entry points. Generic operational code uses the central resolver.

### Memory capacity and fit

Scheduling and per-host fit resolve capacity from `AcceleratorSpec.memory_gb`
first, then the matched platform's `default_accelerator_memory_gb(accelerator)`.
GB10's `nvidia-smi` reports memory as unavailable, so for a unified-memory
device that is the host's only accelerator the probe measures capacity from the
host's `MemTotal` instead (119.7 GiB on a 128 GB DGX Spark, about 62 GiB on a
64 GB part). DGX Spark's 121 GiB planning capacity applies only when nothing was
measured (assumed hardware, no probe). The default 90% scheduling/fit cap is
applied on top of either.

Explicit inventory capacity takes precedence. Generic or unidentified devices
remain unknown unless their platform provides a qualified capacity. Resolution
also applies to previously saved probe records; it does not rewrite raw
inventory or its fingerprint. Scheduler inputs receive temporary resolved copies,
and fit uses the same resolver and utilization-cap precedence.

Model requirements are independent of target capacity. `estimate_vram()` leaves
capacity, available automatic KV memory, and maximum context unknown unless the
caller supplies a target. An explicit `kv_cache_memory_bytes` remains a per-GPU
budget and can be evaluated without target hardware. Standalone `recipe vram`
reports target capacity unknown; placement-aware reporting uses assigned GPUs.

Capacity provenance and scheduling-cap provenance travel with resolved inputs.
`check_fit()` and its JSON output expose `status: fits | exceeds | unknown`, with
per-host details. Unknown or assumed hardware and incomplete model estimates
cannot establish a verified fit. The legacy `ok` boolean remains permissive for
unknown memory; consumers requiring verification should use `status`. Fit is a
memory estimate, separate from the allocation admission rules below.

Spark's capacity and 0.90 policy constants live in its platform plugin. Generic
estimator JSON no longer contains `fits_dgx_spark` or `dgx_spark_fit_budget_gb`.
The old Python `fits_dgx_spark` property and capacity constant imports remain
through deprecated compatibility adapters; internal code must not use them.

### GPU allocation admission

The occupancy schedulers (`occupancy-sparse` and `occupancy-dense`) treat normal
recipe launches as exclusive GPU allocations. A `ResourceRequest` with
`util_fraction >= 0.8` also reserves the whole GPU. An exclusive allocation needs
an unoccupied GPU; the model/KV estimate does not veto it. Runtime startup and
readiness determine whether the recipe actually fits in memory.

Explicit shared requests (`util_fraction < 0.8`) must fit both the remaining
compute fraction and the capped memory budget. `memory_gb` is a per-rank GiB
reservation; when omitted it is derived from utilization times nominal capacity.
Unknown memory capacity cannot establish a shared fit. Shared jobs cannot enter
an exclusively owned GPU, and an exclusive job cannot displace shared jobs.
The same checks apply to solo requests and explicit layouts in the occupancy
schedulers.

The legacy `greedy` scheduler ignores existing occupancy and reservations. It
packs ranks into the first available hardware slots in host order, treating busy
GPUs as available, so separate runs can be placed on the same GPU. It does not
enforce the shared-allocation admission limits above. Separately, its API rejects
an explicit fractional `ResourceRequest` (`util_fraction < 0.8`); this is a request
type restriction, not an occupancy check. Ordinary recipe runs are not rejected
because another job already occupies their chosen GPU.

Reservations are recovered from worker records, not current GPU activity. An
idle serving job still owns its allocation. Missing legacy allocation records
are treated conservatively; a partially occupied multi-GPU host needs known GPU
indices before further placement. Failed status queries do not establish idle
capacity. Rejections include host/GPU identifiers and reasons such as exclusive
ownership, exhausted compute, insufficient memory, or incomplete observations.

Fit reports are separate from admission. They use the assigned accelerators,
resolved scheduling cap, and runtime memory budget. Explicit
`kv_cache_memory_bytes` supplies a per-GPU KV allocation and supersedes
`gpu_memory_utilization` for KV sizing. Automatic KV fit uses the smaller of the
runtime and scheduling budgets. Partial architecture estimates are marked
unverified. Inventory and cluster cap overrides retain their precedence.

### `Capability` tags

Free-form strings on `AcceleratorSpec.capabilities`. Conventions in use:

- `cuda` / `rocm` / `gaudi` — software stack hint.
- `unified-memory` — DGX Spark / Apple Silicon.
- `nvlink` — present where applicable.
- `rdma:roce-v2` — multi-node collective over Mellanox.

Runtimes declare `requires_capability: frozenset[str]` (e.g.
`{"rdma:roce-v2"}` for TRT-LLM multi-node) and the central compatibility check
(`launcher.py`, `runtimes/compatibility.py`) verifies every placed host
satisfies the set.

## `probe_host` — combined SSH probe

`core/hardware_probe.py` runs a single bash script per host that emits two
sections:

```
SPARKRUN_PROBE_ACCEL_START
NVIDIA_GPU_0_NAME=NVIDIA GB10
NVIDIA_GPU_0_MEMORY_MIB=124032
NVIDIA_GPU_COUNT=1
NVIDIA_DRIVER_VERSION=610.22.03
...
IB_PRESENT=1
SPARKRUN_PROBE_ACCEL_END
SPARKRUN_PROBE_IB_START
IB_DETECTED=1
DETECTED_GID_INDEX=3
DETECTED_HCA_LIST=mlx5_0
DETECTED_SOCKET_IFNAME=enP2p1s0
...
SPARKRUN_PROBE_IB_END
```

`split_probe_output()` slices the two sections by sentinel; the accel section
is parsed by `fingerprint.parse_fingerprint_output` and the IB section by
`utils.parse_kv_output`. The resulting `HostHardware.ib_info` is the IB dict
verbatim, and `HostHardware.accelerators` is materialized through
`build_host_hardware`.

`probe_hosts()` parallelizes via `ssh.run_remote_scripts_parallel` — one SSH
connection per host, all concurrent. Single-host callers use `probe_host()`;
both APIs return accelerator and InfiniBand information together.

For host executors (Docker and local processes), `api.plan()` runs this combined
probe before occupancy scheduling and the CLI memory-fit summary. Candidate hosts
are probed concurrently once per plan, using the prepared transport, cluster SSH
user and management-interface pin. Dry runs skip discovery; provider-managed
executors such as Kubernetes retain their own inventory path.

`RunPlan.host_hardware` retains the raw, operation-local observations. Successful
observations are overlaid onto an ephemeral cluster view while preserving
configured capacity ceilings, utilization limits and physical GPU indices. Saved
cluster inventory is unchanged. A failed observation leaves existing inventory
or the application's hardware assumption in use, with a warning; the failed
result is retained so stricter consumers can reject it without a second probe.

`run(plan=...)` carries the observations into launch and reuses interface data
for transfer selection. ColdSnap checks its driver requirements against the same
results; direct ColdSnap operations probe when no planning observation exists.
Reports distinguish detected GPU/driver identity, measured or estimated capacity,
and discovered interfaces. GB10's 121 GiB platform fallback remains an estimate,
so detecting the GPU does not establish verified memory fit. Interface discovery
also does not establish peer connectivity or validate the RDMA data path.

### Plugin-owned hardware probes

Selected integrations can call
`core.hardware_probe_extensions.register_hardware_probe(name, script=..., enrich=...)`
in their `register(v)` hook. The script is a read-only Bash fragment that emits
namespaced facts using core's `emit` function. Both the standalone fingerprint
script and combined accelerator/IB script include these fragments. Each fragment
runs in a subshell; the combined path places it inside the accelerator section
without adding another SSH connection.

`enrich(parsed, hardware)` returns a `HostHardware`, preserving unrelated
accelerators and metadata. Core computes the final accelerator fingerprint after
all enrichers run. Parser errors propagate rather than silently accepting partial
hardware. Registration is idempotent for identical providers, rejects conflicting
names, and participates in installed-plugin transaction rollback. Applications
must initialize selected integrations before generating or parsing probes.
Registration itself never detects controller hardware.

## `select_backends` → `BackendBundle`

`core/backend_select.py`:

```python
@dataclass(frozen=True)
class BackendBundle:
    accelerator_vendor: str   # "nvidia" / "amd" / "intel"
    collective: CollectiveBackend
```

`select_backends(host_hardware)` resolves providers for the selected accelerators.
It uses the matched platform's `collective_backend()`, otherwise the vendor's
registered provider, and verifies a common vendor and communication protocol.
An unresolved provider raises `NoMatchingBackendError` with detected hardware.

`launcher.py:resolve_per_host_backends(...)` resolves all participating hosts.
Required groups reject missing, incompatible, or unimplemented providers before
preparation. `_cluster_ops.resolve_comm_env()` resolves an omitted map through
the same platform mechanism; an incomplete explicit map cannot fall back to NCCL.

The bundle is threaded all the way through to `runtime.run(..., backends=...)`
and persisted in `orchestration/job_metadata.py` so post-launch commands can
recover the same view.

## Platform plugins

`platforms/base.py:HardwarePlatformPlugin` binds a vendor to executor/collective
choices and image defaults. The hook surface:

| Method                   | Purpose                                                                                       |
|--------------------------|-----------------------------------------------------------------------------------------------|
| `matches(hw)`            | Predicate. First plugin that returns true wins.                                               |
| `accelerator_vendor()`   | Vendor string for `ExecutorConfig.accelerator_vendor`.                                        |
| `collective_backend()`   | A `CollectiveBackend` instance.                                                               |
| `default_image(runtime)` | Per-runtime container default. `None` means "recipe must set `container:`".                   |
| `validate_host(hw)`      | List of human-readable warning strings. `launcher.py` logs them; does not raise.              |
| `default_accelerator_memory_gb(accelerator)` | Known capacity for a qualified device, or `None`; used only when inventory capacity is absent. |
| `default_max_gpu_memory_utilization(accelerator)` | Platform scheduling/fit cap, or `None`; host and cluster settings take precedence. |

### Built-in plugins

- **`DgxSparkPlatform`** — matches `vendor=nvidia, model=gb10`. Defaults to
  Spark Arena images (`ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest`,
  etc.). `validate_host` warns when:
  - The accelerator vendor/model drifted from `gb10`.
  - No `rdma:roce-v2` capability is present on the GB10 (multi-node fabric
    health concern).
- **`GenericNvidiaPlatform`** — catch-all for any NVIDIA. Upstream images
  (`vllm/vllm-openai:latest`, `lmsysorg/sglang:latest`, ...). `validate_host`
  only warns when `matches()` was called on a host with no NVIDIA accelerator.

Registration order in `platforms/__init__.py` is most-specific first;
`DgxSparkPlatform` always wins on GB10 hosts.

Device-specific policy uses `resolve_accelerator_platform()` with the selected
accelerator and preserved host facts. Unassigned devices cannot select an image,
capacity cap, runtime setting, or collective provider for the workload.

Images must be qualified by the matched platform or declared explicitly in the
recipe. A platform returning `None` does not fall through to a Spark runtime
image prefix. Assigned devices sharing a container must agree on its default
image. Executor settings and environment defaults resolve per host; common serve
flags must agree across the group unless an explicit recipe/CLI setting resolves
the conflict. Platform defaults remain below explicit user settings. Policy hook
errors are surfaced instead of silently substituting another platform's defaults.

Collectives use the platform's `collective_backend()` hook, with the explicit
vendor provider as fallback when no platform matches. Required groups must have
complete, compatible, implemented providers. Unknown providers cannot silently
receive NCCL settings. Single-rank execution may use an explicit no-collective
provider. RCCL/HCCL remain scaffolds; this policy does not enable their runtime
implementations.

### Adding a new platform

1. Subclass `HardwarePlatformPlugin` in `platforms/<name>.py`:
   ```python
   class MyAmdPlatform(HardwarePlatformPlugin):
       platform_name = "amd-mi300x"
       vendors = frozenset({"amd"})
       def matches(self, hw): return any(a.vendor=="amd" and a.model=="mi300x" for a in hw.accelerators)
       def accelerator_vendor(self): return "amd"
       def collective_backend(self): return RcclBackend()
       def default_image(self, runtime): return _MI300X_DEFAULTS.get(runtime)
       def validate_host(self, hw): return [...]
   ```
2. Call `register_platform(MyAmdPlatform(), prepend=True)` from your plugin
   registration hook when it must match before built-ins. Use the registrar so
   provider conflicts and transaction rollback remain enforced.
3. Installed packages use the `sparkrun.plugins` entry-point group;
   `EXT_PLATFORM` is not an installed-package discovery route. See
   [plugin registration](PLUGINS.md).

## Collective backends

`orchestration/collectives/` houses `CollectiveBackend` implementations. The
ABC requires:

- `name` / `vendor` class attrs.
- `env_for_host(ib_info, *, topology=None) -> dict[str, str]`.
- `ring_overrides(ib_info) -> dict[str, str]` (used by 3-node ring topologies).

### Filling in `RcclBackend` (AMD)

`collectives/rccl.py` is a stub. To implement:

1. Map the existing Mellanox IB probe into ROCm Communication Collectives env
   vars (`RCCL_IB_HCA`, `RCCL_NET_SHARED_BUFFERS`, etc.). The probe shape is
   already vendor-agnostic — Mellanox HCAs serve AMD hosts too.
2. Consider whether HIP env (`HSA_OVERRIDE_GFX_VERSION`, `HIP_VISIBLE_DEVICES`)
   should travel with the collective env block or live in the runtime container
   env.
3. Implement `ring_overrides()` only if 3-node ring topology is in scope.

### Filling in `HcclBackend` (Intel Gaudi)

`collectives/hccl.py` is a stub. To implement:

1. Translate IB detection into HCCL env (`HCCL_OVER_OFI=1`,
   `HCCL_SOCKET_IFNAME=<DETECTED_SOCKET_IFNAME>`, etc.).
2. Add Gaudi-specific env (`HABANA_VISIBLE_DEVICES`) where appropriate.

Either backend implementation will start producing env immediately for every
host whose vendor resolves via `select_backends`. No runtime wiring changes
needed — `runtime.run(..., backends=...)` already consumes
`backends[host].collective.env_for_host(...)` through
`_cluster_ops.resolve_comm_env()`.

## Heterogeneity boundaries

- **Per host, single vendor**: `accelerator_vendor_for()` returns `None` when
  a host advertises >1 vendor (e.g. Apple M5 + discrete NVIDIA). Such hosts
  fall through `select_backends`; the placement engine refuses to auto-pack
  them and requires an explicit `recipe.layout`.
- **Across hosts, single vendor**: auto-packing works (e.g. mixed RTX + H200
  cluster as long as both are NVIDIA).
- **Across selected hosts, multi-vendor**: `schedulers/greedy.py:pack()` raises
  `core.scheduler.LayoutRequiredError`. Recipes must declare a `RecipeLayout`
  mapping ranks to (host, local-GPU) explicitly. Unselected hosts do not impose
  a vendor constraint on the placement.

## Apple / CPU

Apple MLX and pure-CPU hosts have `accelerator_vendor_for(...) in {"apple",
"cpu"}` (or `None` if the metadata reads ambiguous). They fail
`select_backends` with `NoMatchingBackendError`. To support them:

1. Add an `AppleSiliconPlatform` that returns the right `collective_backend`
   (likely a CPU-only no-op) and `accelerator_vendor()` string.
2. Add Apple/CPU to `backend_select._KNOWN_VENDORS`.
3. Route those hosts to a non-Docker executor (`LocalExecutor`) since MLX
   doesn't run inside Docker.

This work is unstarted — the seams exist, the scaffolds don't.


## Application profiles and hardware integrations

Application identity does not detect hardware. An integration installed into
ordinary Sparkrun must provide the same target-host behavior it provides under
an alternate launcher. Application profile defaults only supply site policy beneath
explicit recipe, cluster and invocation choices.

Operational missing-hardware paths use `resolve_hardware()`. Built-in
Sparkrun retains its legacy DGX Spark fallback; alternate application profiles default
to requiring metadata. The scheduler, resource limits, cluster hardware access,
runtime fallback and executor status paths must not invent GB10 capabilities
for an unidentified alternate target.

Hardware integrations own device-specific detection, measured memory policies,
software-stack diagnostics, platform registration and compatibility validation.
`core.hardware_probe_extensions.register_hardware_probe` lets a selected plugin
contribute target facts and enrich the shared hardware model. The same plugin
can be selected by Sparkrun or required by an alternate application.

Readiness and wizard behavior use explicit `SetupPlan` declarations owned by each
hardware platform, separately for each executor. Plans compose selected core and
plugin steps; new registered steps remain inactive until a plan includes them.
Constraints can further restrict the selection. Application defaults select policy; they do not
establish hardware or container qualification. See [SETUP_STEPS.md](SETUP_STEPS.md)
for setup extensions and [APPLICATION_PROFILES.md](APPLICATION_PROFILES.md) for
the application composition contract.
