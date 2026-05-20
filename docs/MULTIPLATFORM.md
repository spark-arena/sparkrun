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
```

- `accelerators` is a list of `AcceleratorSpec(vendor, model, count, memory_gb,
  capabilities: frozenset[str])`. Multiple entries cover heterogeneous-on-one-host
  scenarios (e.g. an Apple M5 host with a discrete NVIDIA GPU).
- `vendors` returns the set of distinct vendors across `accelerators`. Used by
  the placement engine to decide whether a host is single-vendor.
- `has_capability("rdma:roce-v2")` walks every accelerator on the host.
- `ib_info` is the raw KV output from the IB section of the combined probe.
  Populated by `hardware_probe.probe_host`. Not serialized to cluster YAML.

The convenience default is `default_dgx_spark_hardware()` — 1 × GB10, 121 GB,
capabilities `{cuda, unified-memory, rdma:roce-v2}`. Every code path that lacks
explicit metadata falls back here so existing DGX clusters keep working.

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
connection per host, all concurrent. Replaces the older two-trip pattern
(`fingerprint_host` then `detect_ib_for_hosts`).

`fingerprint_host` (`core/fingerprint.py`) is now a thin shim retained for
callers that only want accelerator data and don't pay for the IB section.

## `select_backends` → `BackendBundle`

`core/backend_select.py`:

```python
@dataclass(frozen=True)
class BackendBundle:
    accelerator_vendor: str   # "nvidia" / "amd" / "intel"
    collective: CollectiveBackend
```

`select_backends(host_hardware)`:

1. `accelerator_vendor_for(host_hardware)` — single shared vendor across the
   host's accelerators, else `None`.
2. If the vendor is unknown or `None`, raise `NoMatchingBackendError` with both
   detected and known vendor lists.
3. Otherwise call `collectives.get_backend(vendor)` and wrap the result.

`launcher.py:resolve_per_host_backends(host_list, cluster=...)` runs this per
host, silently dropping failures so partial-vendor coverage still launches and
runtimes fall back to legacy `resolve_ib_env` per missing host.

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
2. Register either by appending to `_REGISTRY` in `platforms/__init__.py`, or
   from external packages via `register_platform(MyAmdPlatform(), prepend=True)`
   when the plugin must match before built-ins.
3. (Future) An `EXT_PLATFORM` entry-point hookup will replace the manual
   registration list. The constant is already defined in `platforms/base.py`.

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
- **Across hosts, multi-vendor**: `core/placement.py` raises
  `LayoutRequiredError`. Recipes must declare a `RecipeLayout` mapping ranks
  to (host, local-GPU) explicitly.

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
