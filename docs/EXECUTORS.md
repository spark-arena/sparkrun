# Executors

How sparkrun selects and configures an executor for a launch. Three are shipped:

| Selector | Class            | Status       | Notes                                                                   |
|----------|------------------|--------------|-------------------------------------------------------------------------|
| `docker` | `DockerExecutor` | Stable       | Default. Used by every previously-released launch path.                 |
| `local`  | `LocalExecutor`  | Alpha        | Native subprocess (no container). Hand-coded process-group lifecycle.    |
| `k8s`    | `K8sExecutor`    | Experimental | `kubectl run`-driven. Drops Docker-specific options.                    |

## Resolution chain

`orchestration/executor.py:resolve_executor()` is the single sanctioned entry
point. It layers (highest priority first):

1. **CLI overrides** — `cli_overrides` dict (`-o executor=local`, `-o
   k8s_namespace=...`, etc.).
2. **Recipe** — `recipe.executor` (selector) + `recipe.executor_config` (dict).
3. **Cluster** — `cluster.executor` (selector) + `cluster.executor_config` (dict).
4. **Runtime executor selector** — `runtime.default_executor()` (`None` by default; runtimes can force a non-Docker executor).
5. **Per-executor adjustments** — `cls.apply_runtime_adjustments(rootless=,
   auto_user=)`. Docker reads these here; Local/K8s ignore.
6. **Runtime executor-config defaults** — `runtime.default_executor_config()` (`{}` by default; runtimes can set overridable executor defaults).
7. **`SparkrunConfig`** — `config.default_executor` + `config.executor_config`.
8. **Platform** — `platform.default_executor_config(<name>)` for the platform
   resolved from the launching host's hardware (`host_hardware=`, the head
   node's). Hardware-conditional container plumbing that should still lose to
   anything the user wrote — e.g. DGX Spark pins `gpu_access_mode: gpus`.
   Dropped entirely when no hardware is threaded (naming / teardown / log paths).
9. **Per-executor defaults** — `cls.default_config()` (e.g. `DOCKER_DEFAULTS`).
10. **Dataclass field defaults** — `ExecutorConfig` declares the floor.

A selector that is unknown — or names a real executor whose feature flag is
off — raises `ExecutorUnavailableError` naming the flag to enable. Resolution
never silently degrades to `"docker"`: running an explicitly-requested workload
on the wrong backend is worse than failing loudly.

When *no* layer names an executor, `_default_executor_name` returns `docker`
when enabled, else the sole remaining enabled executor, else raises. So
disabling `executor.docker` is honored rather than being overridden by a
hardcoded baseline.

The set of known selectors is queried from SAF via
`get_extensions(EXT_EXECUTOR, v=v)`; the hardcoded `_KNOWN_EXECUTORS` set was
retired in 0.3.0.

## SAF discovery

```python
EXT_EXECUTOR = "sparkrun.executor"
```

`core/bootstrap.py` calls
`find_types_in_modules("sparkrun.orchestration.executors", Executor)` and
registers each discovered subclass. Subclasses must set:

- `executor_name: ClassVar[str]` — the selector string (must be unique).
- `is_multi_extension(v)` → `True` and `is_enabled(v)` → `False` (inherited;
  prevents SAF's single-extension cache from short-circuiting).

Look-up helpers:

- `get_executor(name, v=None) -> type[Executor]` — returns the class, not an
  instance. Falls back to a static map (`docker`/`local`/`k8s`) when SAF isn't
  initialized (test paths).
- `list_executors(v=None) -> list[str]` — sorted selectors.

## `ExecutorConfig.from_chain` field reference

Every field is parsed from a chain layer with the same name (`chain.get(key)`).
Bool fields use `ext_parse_bool`. List fields promote bare strings to single-item
lists. Falsy values fall through to the dataclass defaults.

### Common (Docker + K8s read; Local ignores most)

| Field                 | Type        | Consumed by         | Default       | Notes                                                                                          |
|-----------------------|-------------|---------------------|---------------|------------------------------------------------------------------------------------------------|
| `executor`            | str         | resolver            | `"docker"`    | Selector. Also accepts `executor_type` as an alias for forward compat.                         |
| `auto_remove`         | bool        | Docker              | `True`        | Adds `--rm`. Force-flipped to `False` when `restart_policy` is set.                            |
| `restart_policy`      | str?        | Docker              | `None`        | Docker `--restart` value.                                                                      |
| `privileged`          | bool        | Docker              | `True`        | Off in rootless mode.                                                                          |
| `gpus`                | str         | Docker, Local, K8s  | `"all"`       | The GPU spec. Docker spells it per `gpu_access_mode`. Local translates `device=0,2` → `CUDA_VISIBLE_DEVICES`. K8s extracts a count for `nvidia.com/gpu`. |
| `gpu_access_mode`     | str         | Docker              | `"cdi"`       | `cdi` → `--device nvidia.com/gpu=<id>`; `gpus` → `--gpus <gpus>`. Platform-defaulted (DGX Spark pins `gpus`). |
| `ipc`                 | str         | Docker              | `"shareable"` | `--ipc`. Own IPC namespace + own `/dev/shm`, joinable via `--ipc=container:<name>`. **Not `host`**: see below. K8s drops. |
| `shm_size`            | str         | Docker              | `"32gb"`      | `--shm-size`. Only applies when `ipc` is not `host` (Docker ignores it otherwise). K8s drops.  |
| `network`             | str         | Docker              | `"host"`      | `--network`. K8s drops.                                                                        |
| `user`                | str?        | Docker              | `None`        | `--user`. Sentinel `"$SHELL_USER"` expands to `$(id -u):$(id -g)` + bind-mounts passwd/group.  |
| `security_opt`        | list[str]?  | Docker              | `None`        | Repeated `--security-opt`. Defaults to `["no-new-privileges"]` in rootless mode.               |
| `cap_add`             | list[str]?  | Docker              | `None`        | Repeated `--cap-add`.                                                                          |
| `ulimit`              | list[str]?  | Docker              | `None`        | Repeated `--ulimit`. Rootless mode sets `memlock=-1:-1`, `stack=67108864`.                     |
| `devices`             | list[str]?  | Docker              | `None`        | Repeated `--device`. Rootless mode adds `/dev/infiniband`.                                     |
| `memory_limit`        | str?        | Docker, K8s         | `None`        | Docker `--memory`; K8s `--limits=memory=...`.                                                  |
| `labels`              | list[str]?  | Docker, K8s         | `None`        | Repeated `--label` / `--labels`.                                                               |
| `entrypoint`          | str?        | Docker, K8s         | `None`        | Docker emits `--entrypoint`; `""` clears the image ENTRYPOINT. K8s emits `--command`; `""` uses `bash -c`. |
| `accelerator_vendor`  | str?        | Docker              | `None`        | `nvidia` / `amd` / `intel` / `apple` / `cpu`. Drives accelerator-flag emission.                |

### Local-only (Docker / K8s ignore)

| Field             | Default                                        | Notes                                                                                                |
|-------------------|------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `working_dir`     | `None`                                         | `cd <working_dir>` before launch.                                                                    |
| `log_dir`         | `$HOME/.cache/sparkrun/local/logs`             | Per-container `<log_dir>/<container_name>.log`.                                                      |
| `log_file`        | `None`                                         | Overrides `<log_dir>/...` entirely.                                                                  |
| `pid_dir`         | `$HOME/.cache/sparkrun/local/pids`             | Per-container `<pid_dir>/<container_name>.pid`.                                                      |
| `pid_file`        | `None`                                         | Overrides `<pid_dir>/...` entirely.                                                                  |
| `env_file`        | `None`                                         | Sourced via `set -a; . <env_file>; set +a` before launch.                                            |
| `command_prefix`  | `None`                                         | Prepended verbatim (e.g. `nice -n 10 ionice -c2`).                                                   |

### K8s-only (Docker / Local ignore)

| Field                    | Default | Notes                                                                                       |
|--------------------------|---------|---------------------------------------------------------------------------------------------|
| `k8s_namespace`          | `None`  | `kubectl -n <ns>`.                                                                          |
| `k8s_context`            | `None`  | `kubectl --context <ctx>`.                                                                  |
| `k8s_node_selector`      | `None`  | `key=value[,key=value]`. Emitted as `--overrides` JSON because `--node-selector` was removed. |
| `k8s_image_pull_policy`  | `None`  | `--image-pull-policy`.                                                                      |
| `kubeconfig`             | `None`  | `--kubeconfig`.                                                                             |

## `DockerExecutor` (default)

`orchestration/executors/docker.py`. Owns `DOCKER_DEFAULTS` and the
`apply_runtime_adjustments(rootless=, auto_user=)` lever:

- `rootless=True` (default) → flips `privileged=False`, adds
  `no-new-privileges` / `memlock` ulimit / `/dev/infiniband` device, sets
  `auto_user="$SHELL_USER"` when paired with `auto_user=True`.
- `auto_user=True` (default) → `--user $(id -u):$(id -g)` + bind-mounts
  `/etc/passwd:/etc/passwd:ro` and `/etc/group:/etc/group:ro` (kernel reads UID
  names) and sets `HOME=/tmp`.

`_accelerator_opts()` emits device flags based on `accelerator_vendor`:

| Vendor      | Flags                                                              |
|-------------|--------------------------------------------------------------------|
| `nvidia`/None | per `gpu_access_mode` — see below                                |
| `amd`         | `--device /dev/kfd --device /dev/dri --group-add video`           |
| `intel`       | `--device /dev/accel`                                             |
| `apple`/`cpu` | (none — route to a non-Docker executor)                           |

### ENTRYPOINT preflight

`Executor.verify_command_passthrough(image, hosts)` is the second write-path
preflight alongside `verify_mount_sources` — that one asks "do the paths I will
mount exist on this substrate?", this one asks "will the command I append
actually run on this image?".

sparkrun composes every workload as `docker run <opts> <image> bash -c <b64 cmd>`,
so the command is always CMD *arguments* and the image's ENTRYPOINT decides what
happens to them. Two opposite idioms are in wide use and `docker image inspect`
cannot tell them apart — both are simply "a non-empty ENTRYPOINT":

- **passthrough** — `/opt/nvidia/nvidia_entrypoint.sh` and friends: do setup,
  then `exec "$@"`. Inherited by nearly every NGC-derived image, so this is the
  *common* case. Clearing it would skip the setup.
- **consuming** — `ENTRYPOINT ["vllm","serve"]`: the appended `bash -c …` is
  parsed as that program's own flags, so the workload never starts.

The verdict is therefore established empirically (`containers/entrypoint.py` +
`scripts/image_entrypoint_probe.sh`) rather than by inspection or an allowlist:

1. No ENTRYPOINT → `absent`, no container started (the cheap exit for most images).
2. Run the real argv shape and look for a **computed** sentinel on stdout. Found
   → `pass`. The sentinel is computed, not literal, because a consuming
   entrypoint typically echoes the argv it rejected — an echo can reproduce a
   literal token but never the evaluated one.
3. Not found → re-run byte-identically with `--entrypoint ''`. Only if *that*
   succeeds is the entrypoint provably the cause (`fail`); otherwise the fault
   is elsewhere — stale CDI, no GPU, no bash — and the verdict is `unknown`.

Only `fail` blocks, and `launcher._verify_image_command_passthrough` raises a
`RecipeError` naming both fixes (`executor_config: {entrypoint: ""}` or
`-o entrypoint=''`) rather than auto-clearing: the probe shows that clearing
*works*, not that it is *harmless* — a consuming entrypoint may also be doing
setup the workload needs.

It runs from `distribute_from_config`'s `after_container_sync` hook — the image
is resident on every target but the long, routinely-interrupted model sync has
not started yet. One host is probed (the verdict is a property of the image, and
distribution already established the image matches everywhere), under the
launch's own accelerator flags. Everything is fail-open: unreachable host,
timeout, unresolvable executor, or `SPARKRUN_NO_IMAGE_PROBE=1` all read as
"proceed". The base implementation returns `None`, so container-less (`local`)
and provider executors never block a launch.

### IPC namespace (`ipc`) and `shm_size`

The default is `shareable`, **not** `host`. The two settings are coupled: Docker
ignores `--shm-size` whenever `--ipc=host`, because the container then simply
gets the host's `/dev/shm` (typically 50% of RAM). Change one and you change
what the other means.

`ipc: host` is unsafe in sparkrun's default configuration. The container runs as
the SSH user (`--user $(id -u):$(id -g)`), so under a host IPC namespace every
POSIX semaphore and shared-memory segment the workload creates is a host file
owned by a regular UID. systemd-logind with `RemoveIPC=yes` — the Ubuntu 24.04 /
DGX OS default — deletes exactly those objects `UserStopDelaySec` (10s) after
that user's last login session ends. sparkrun launches detached (`docker exec -d`)
and then closes its SSH session, so a workload still in Python's multiprocessing
bootstrap when the reaper runs dies with:

```text
File "/usr/lib/python3.12/multiprocessing/synchronize.py", line 115, in __setstate__
    self._semlock = _multiprocessing.SemLock._rebuild(*state)
FileNotFoundError: [Errno 2] No such file or directory
```

A workload that gets past that point first survives indefinitely — unlinking a
name does not invalidate an already-open handle — which is why the failure is
intermittent, why longer startups (large models, graph capture) fail more often
than short ones, and why holding an unrelated SSH session open makes it vanish.
Running as root also masks it, since logind never reaps system UIDs.

A container-owned namespace is immune: logind walks `/dev/shm` itself, not
arbitrary mounts. `shareable` is preferred over `private` because it costs
nothing and still lets a second container on the same host join with
`--ipc=container:<name>`.

Keep `ipc: host` only for cross-*container* shared memory on one host (e.g.
intra-node NCCL between separate containers). When you do, either enable
lingering for the SSH user (`sudo loginctl enable-linger <user>`) or set
`RemoveIPC=no` in `/etc/systemd/logind.conf` on every host; `sparkrun setup
check` reports this combination, and the executor warns at launch. Note the
`local` executor has the same exposure by construction — it runs natively, with
no namespace to hide in.

**`ipc` is trust-gated on its value.** A recipe may narrow the namespace freely
(`private`, `shareable`, `none`), but an untrusted recipe — one from a
non-default registry or a URL — cannot select `host` or `container:<name>`
without `--trust`. Both reach *outside* the container: the host's `/dev/shm`
holds every other tenant's POSIX shared memory and semaphores, and a container
name is derivable from a cluster_id. See [SECURITY.md](SECURITY.md).

### NVIDIA GPU access mode

There are two ways to ask Docker for NVIDIA GPUs and neither works everywhere,
so `gpu_access_mode` selects between them:

| Mode           | Flags for `gpus: all`          | Flags for `gpus: device=0,1`                                       |
|----------------|--------------------------------|---------------------------------------------------------------------|
| `cdi` (default) | `--device nvidia.com/gpu=all` | `--device nvidia.com/gpu=0 --device nvidia.com/gpu=1`               |
| `gpus`          | `--gpus all`                  | `--gpus device=0,1` (value passed through verbatim)                 |

A falsy `gpus` still means "no GPU request" in both modes. An unrecognised mode
warns and falls back to `cdi`.

CDI (Container Device Interface, Docker >= 25) is the portable path and is
*required* on daemons that reject `--gpus` (e.g. Thunder Compute), but it
depends on a present, non-stale `/etc/cdi/nvidia.yaml` — the spec pins versioned
absolute paths, so a driver upgrade can leave it dangling and containers then
fail to start. `--gpus` resolves through the container runtime at launch instead.

`sparkrun setup check` reports on the spec at a severity that follows this
setting: it resolves each host's effective `gpu_access_mode` through the real
executor chain, so a missing spec is a **FAIL** only for a cluster that would
actually read it. Under `gpus` the finding drops to **SKIP** (it doesn't count
as a gap or affect the exit code) while still naming the staleness, because it
becomes real the moment the mode changes. A mode that can't be resolved fails
safe to "CDI required".

The default comes from the resolved hardware platform
(`HardwarePlatformPlugin.default_executor_config("docker")`), which sits just
above `DOCKER_DEFAULTS` in the chain: **DGX Spark / GB10 pins `gpus`**, every
other platform inherits `cdi`. Override anywhere higher — recipe, cluster, or
`config.yaml`:

```yaml
executor_config:
  gpu_access_mode: cdi
```

## `LocalExecutor` (experimental)

`orchestration/executors/local.py`. Native subprocess; no container.

### What it does

- `run_cmd`: `mkdir -p <pid_dir> <log_dir>` → optional `cd working_dir` →
  optional `set -a; . env_file; set +a` → translate `gpus` →
  `CUDA_VISIBLE_DEVICES` → export explicit env → `setsid bash -c <base64 cmd>
  >>log 2>&1 </dev/null &` → write `$!` to pidfile.
- `stop_cmd`: reads pidfile, sends `SIGTERM` to the negative PID (process
  group), polls up to ~10 s, then `SIGKILL`. Removes pidfile.
- `status_cmd`: pidfile + `kill -0`.
- `logs_cmd`: `tail [-F] [-n N] <log_file>`.

### Known limitations

- **No images**: `image`, `volumes`, `extra_opts` are ignored. `pull_cmd` /
  `inspect_exists_cmd` are no-ops returning `true`.
- **Hand-coded process-group lifecycle**: relies on `setsid` (present on every
  modern Linux), `kill -- -<pgid>`, and pidfile parsing. No supervisor, no
  systemd unit, no restart on crash.
- **No Ray strategy**: `generate_ray_head_script` and
  `generate_ray_worker_script` raise `NotImplementedError`. Use a native
  runtime (`vllm-distributed`, `sglang`) or fall back to Docker.
- **GPU visibility is best-effort**: only `gpus="all"` / `gpus="device=0,2"`
  are translated. `count=2` and capability filters log a warning and leave
  visibility to the workload.

Multi-host native cluster runtimes work because each host's
`container_name` becomes `<cluster_id>_node_<rank>`, which is the basename for
per-rank pid/log files. The Ray restriction means `vllm-ray` is the only
runtime that can't pair with Local.

## `K8sExecutor` (experimental draft)

`orchestration/executors/k8s.py`. `kubectl run`-based — every lifecycle
operation is a `kubectl` invocation.

### What it does

- `run_cmd`: `kubectl [--kubeconfig …] [--context …] [-n …] run <name>
  --image=<image> --restart=Never [--image-pull-policy=…] [--overrides=<JSON
  nodeSelector>] [--limits=nvidia.com/gpu=N] [--limits=memory=…] [--env=K=V …]
  [--labels=…] [--command when `entrypoint` is set] -- bash -c <base64 cmd>`.
- `exec_cmd`: `kubectl exec` (`-d`-like behavior synthesized via nohup).
- `stop_cmd`: `kubectl delete pod --ignore-not-found [--grace-period=0 --force]`.
- `logs_cmd`: `kubectl logs [-f] [--tail=N]`.
- `status_cmd`: `kubectl get pod -o jsonpath='{.status.phase}'` equality test
  vs `"Running"`.
- `inspect_exists_cmd` / `pull_cmd`: no-ops (the cluster pulls on Pod creation).

### Known limitations

- **`kubectl run`, not manifests**: init containers, sidecars, custom scheduler
  hints, volume claims, services — all unreachable. Use `k8s_node_selector`
  and any `--overrides`-compatible `extra_opts` entries to wedge in extras.
- **Docker-specific options dropped silently**: `--privileged`, `--shm-size`,
  `--ipc`, `--network` aren't translated.
- **GPU mapping is conservative**: `gpus="all"` → `nvidia.com/gpu=1`,
  `device=0,1` → `nvidia.com/gpu=2`. Anything fancier logs a warning and ships
  no GPU resource request.
- **No Ray strategy**: same restriction as Local; runtime authors can either
  target `vllm-distributed` / `sglang` or stay on Docker.
- **Image plumbing**: `generate_exec_serve_script` requires the runtime to put
  the image into the env block under `SPARKRUN_K8S_IMAGE`. Without it the
  generated command points at `sparkrun-k8s-image-not-configured` to fail
  loudly.
- **No StatefulSet / Job**: one Pod per host_list entry; multi-host
  orchestration is still loop-driven from `runtime.run()`.

## Quick recipe field cheat sheet

```yaml
# Local executor (no container)
executor: local
executor_config:
  working_dir: /opt/myproject
  log_dir: /var/log/sparkrun
  env_file: /etc/sparkrun.env
  command_prefix: nice -n 10
  gpus: "device=0,2"     # → CUDA_VISIBLE_DEVICES=0,2

# K8s executor
executor: k8s
executor_config:
  k8s_namespace: ml-prod
  k8s_context: prod-east
  k8s_node_selector: nodepool=dgx-spark
  k8s_image_pull_policy: IfNotPresent
  kubeconfig: /etc/k8s/admin.conf
  memory_limit: 128Gi
  entrypoint: ""        # emits --command -- bash -c ...
```

Docker fields (the existing ones — `privileged`, `cap_add`, `devices`, etc.)
keep their previous behavior and ship under the same `executor_config:` block
when `executor: docker` (or unset).
