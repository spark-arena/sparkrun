# sparkrun proxy

A unified OpenAI-compatible gateway that discovers running sparkrun inference endpoints and exposes them through a
single API powered by [LiteLLM](https://docs.litellm.ai/).

## Overview

The proxy sits in front of one or more inference workloads launched by sparkrun and provides:

- **Live endpoint discovery** using same mechanism as `sparkrun cluster status` and `sparkrun cluster monitor`
- **Auto-discovery** background process that periodically re-scans and syncs models (in case of drift)
- **Health checking** via `GET /v1/models` on each discovered endpoint
- **Deduplication** of endpoints reachable on multiple network interfaces (e.g. management IP vs ConnectX-7 IP)
- **Model aliases** so clients can address a model by a friendly name
- **Load/unload** models through `sparkrun proxy load` or `sparkrun proxy unload` to keep the proxy in sync (although
  autodiscovery should also ensure models are available)

The proxy runs LiteLLM via `uvx --from 'litellm[proxy]==1.82.6' litellm` — no permanent installation required.

> **How changes are applied.** The generated config file is the single source of truth for the model list. LiteLLM's
> runtime mutation endpoints (`/model/new`, `/model/delete`) require a DB-backed model store — PostgreSQL plus a
> generated prisma client — which sparkrun does not provision, so against a sparkrun-launched proxy they answer
> `500 No DB Connected`. Applying a change therefore means **rewriting the config and restarting the process**
> (`ProxyEngine.apply_desired_state`). The restart is skipped entirely when the desired model set already matches
> what is on disk, so a steady-state auto-discover sweep costs nothing. The management API is still used read-only,
> to report what the proxy is currently serving.

## Quick Start

```bash
# Launch some inference workloads first
sparkrun run qwen3-1.7b-vllm --cluster mylab

# Start the proxy (discovers endpoints automatically)
sparkrun proxy start --cluster mylab

# Query models through the unified API
curl http://localhost:4000/v1/models
```

--- OR ---

```bash
# Start the proxy (discovers endpoints automatically if relevant)
sparkrun proxy start

# Load a new model
sparkrun proxy load qwen3.5-0.8b-bf16-sglang

# Query models through the unified API
curl http://localhost:4000/v1/models
```

## Commands

### `sparkrun proxy start`

Discovers running endpoints, generates a LiteLLM config, and launches the proxy. A background auto-discover process
periodically re-scans and syncs models with the proxy. Starting when a proxy is already running is an error; pass
`--restart` to replace it with one carrying the new settings.

```bash
sparkrun proxy start --host 127.0.0.1         # recommended bind (unset keeps legacy 0.0.0.0 + warning)
sparkrun proxy start --port 8080              # custom port
sparkrun proxy start --cluster mylab          # discover from cluster hosts (live SSH)
sparkrun proxy start --hosts 10.0.0.1,10.0.0.2  # explicit host list
sparkrun proxy start --foreground             # run in foreground (blocking)
sparkrun proxy start --master-key sk-mykey    # enable LiteLLM auth
sparkrun proxy start --no-auto-discover       # disable periodic re-scanning
sparkrun proxy start --discover-interval 60   # re-scan every 60s (default: 30)
sparkrun proxy start --discover-removal-grace-sweeps 1  # remove on first miss (default: 2)
sparkrun proxy start --gateway litellm        # pin the gateway implementation
sparkrun proxy start --dry-run                # show what would be done
```

By default, the proxy daemonizes in the background. Logs are written to `~/.cache/sparkrun/proxy/litellm.log`.

### `sparkrun proxy stop`

Sends SIGTERM to the running proxy and its auto-discover process using the stored PIDs.

```bash
sparkrun proxy stop
```

### `sparkrun proxy status`

Shows whether the proxy is running, its PID, bind address, gateway, auto-discover status, and the models it is
currently serving (read via the LiteLLM management API).

```bash
sparkrun proxy status
```

### `sparkrun proxy sync`

Reconciles the proxy's model list with the workloads actually running. Equivalent to the reconciliation an
auto-discover sweep performs, on demand.

```bash
sparkrun proxy sync
```

### `sparkrun proxy models`

Lists models currently registered with the running proxy. With `--refresh`, re-discovers endpoints and syncs the proxy —
adding newly available models and removing stale entries whose backends are no longer healthy.

```bash
sparkrun proxy models
sparkrun proxy models --refresh
```

### `sparkrun proxy load <recipe>`

Launches an inference workload via `sparkrun run` (detached) and registers it with the running proxy.

Unlike plain `sparkrun run`, `proxy load` automatically avoids port conflicts. When no `--port` is specified, it loads
the recipe to determine the desired port (e.g. 8000), then checks the head host over SSH (using `nc -z`, the same
mechanism as `sparkrun benchmark`) to find the first available port. If the desired port is occupied, it increments
until a free port is found:

```
$ sparkrun proxy load qwen3-1.7b-vllm
# Uses port 8000

$ sparkrun proxy load qwen3.5-35b-a3b-fp8-sglang
# Note: port 8000 in use on 10.24.11.13, using 8001 instead
```

This is intentionally different from `sparkrun run`, which uses exactly the port specified (or the recipe default) and
fails if it's occupied — preserving the user's explicit intent. The proxy's `load` command is designed for managing
multiple concurrent models where automatic port assignment is expected.

### `sparkrun proxy unload <recipe>`

Stops the inference workload containers directly (same logic as `sparkrun stop`) and syncs the proxy to remove the
now-stale model entry.

```bash
sparkrun proxy unload qwen3-1.7b-vllm --cluster mylab
```

### `sparkrun proxy alias`

Manage model aliases so clients can reference models by friendly names. An alias is saved to `proxy.yaml`
immediately; if a proxy is running, the config is regenerated and the proxy restarted so the alias takes effect. An
alias whose target has no healthy backend is saved but skipped in the generated config, and starts working as soon as
the target is loaded.

```bash
sparkrun proxy alias add qwen3-small "Qwen/Qwen3-1.7B"
sparkrun proxy alias remove qwen3-small
sparkrun proxy alias list
```

## How Discovery Works

1. `api.list_jobs` enumerates persisted job metadata (`~/.cache/sparkrun/jobs/*.yaml`) — every cluster_id, recipe,
   runtime, hosts, port, served_model_name, api_key
2. `api.status` produces the live snapshot; its `running_cluster_ids` is the authoritative liveness filter. This is
   the cross-executor status source, so native (`local`) workloads are visible too — not only docker containers.
   When no host list is available the liveness step is skipped (metadata-only mode)
3. Normalises host IPs to management IPs (prefers management IPs over InfiniBand IPs)
4. Performs parallel health checks via `GET /v1/models` (3-second timeout)
5. Returns only healthy endpoints

### Auto-discovery

When the proxy starts with auto-discover enabled (the default), a background process runs alongside the proxy:

- Periodically calls `discover_endpoints` at the configured interval (default: 30 seconds)
- Reconciles models **and** aliases in one gateway-neutral `api.proxy.sync` call — for a config-file gateway they
  share a single file, so applying them separately would rewrite and restart the proxy twice per sweep
- Applies the change only when the desired model set differs. *How* is the gateway's business: LiteLLM rewrites its
  config and restarts; another implementation may update its control plane in place
- Keeps a previously healthy endpoint through `discover_removal_grace_sweeps - 1` consecutive misses (default 2). One
  timed-out health probe is not evidence a workload is gone, and evicting it costs a restart plus a window of 404s for
  a model that is serving fine. Set it to `1` for the historical remove-on-first-miss behaviour
- Re-reads the proxy PID from the state file each sweep, so it follows a restart instead of mistaking it for a
  shutdown, and exits automatically when the proxy is really gone
- Re-resolves the gateway from the state file every sweep, so a `proxy start --restart` that swaps implementations is
  followed rather than fought. It is handed `{gateway, state_dir, interval, removal_grace_sweeps}` and therefore never
  carries the master key — the credential belongs to whichever engine the state file names
- Runs as a detached subprocess (`python -m sparkrun.proxy.autodiscover`)

Disable with `--no-auto-discover` or set `auto_discover: false` in `proxy.yaml`. A gateway that owns its own desired
state declares `supports_autodiscover = False`; `proxy start` then warns and disables it rather than letting two
components fight over the same endpoints.

## Configuration

Persistent proxy settings are stored in `~/.config/sparkrun/proxy.yaml`:

```yaml
proxy:
  port: 4000
  host: 127.0.0.1         # recommended; unset keeps the legacy 0.0.0.0 + warning
  master_key: null        # set to require a bearer token (stateless; no DB)
  gateway: litellm        # optional; pins the gateway implementation
  auto_discover: true
  discover_interval: 30   # seconds between re-scans
  discover_removal_grace_sweeps: 2  # missed sweeps before removing an endpoint (1 = remove on first miss)

aliases:
  my-model: "Qwen/Qwen3-1.7B"
  gpt-4: "Qwen/Qwen3-30B-A3B"
```

CLI flags override config file values for a given invocation, and explicitly
supplied values are persisted back (`api/proxy/_ops.py:_persist_overrides`).

Two processes write this file — the auto-discover daemon and any `sparkrun
proxy` command — so `save()` locks a sidecar, re-reads the newest document, and
merges only the sections that instance changed. A whole-document save would
silently discard the other writer's alias or listener change.

## Gateway Selection

The *gateway* is the pluggable family; `proxy` is the user-facing command.
Core ships LiteLLM, enabled on every channel via the `gateway.litellm` feature
flag (`default=True`, like `executor.docker`). A plugin may contribute another,
including one living outside the `sparkrun.proxy` tree.

Three mechanisms, deliberately separate (`proxy/gateway.py`):

- **Registration** — `register_gateway(name, feature_flag=, loader=)`, with
  `gateway_class(name)` the one place a name becomes an implementation. The
  registry is in-process (an engine is *constructed with arguments*, not
  resolved as a stateless singleton) and carries a **loader** rather than the
  class, so registering imports nothing. Idempotent by name, which lets an
  out-of-tree plugin substitute an in-tree implementation.
- **Availability** — the `gateway.<name>` feature flag.
- **Selection** — exactly one gateway at a time, arbitrated by
  `resolve_gateway()`: an explicit pin (`proxy.gateway`, or `--gateway`) must
  be known *and* enabled; with no pin the default wins when enabled, else the
  sole remaining enabled gateway, else `AmbiguousGatewayError`. Enabling a
  second gateway's flag does **not** switch to it. The flag registry has no
  notion of mutually-exclusive flags, so resolution refuses to guess rather
  than picking.

"Unregistered" and "disabled" are distinct errors: a name can be known to the
flag registry while its plugin failed to load, and telling that user to enable
a flag that is already on is a dead end.

An implementation subclasses `GatewaySupervisor` (`proxy/_supervisor.py`),
which owns the process, state-file and auto-discover machinery every gateway
shares, and declares its capabilities — `supports_autodiscover`,
`wants_proxy_config`, `data_plane_authenticated` — rather than being
special-cased by name anywhere above it.

**Gate placement.** `ProxyEngine.start()` is the single enforcement point —
bringing a gateway *up*, checked before the `--dry-run` branch so a dry run
cannot advertise a start that would be refused. `stop` / `status` / model sync
/ alias mutation / the auto-discover daemon's `_restart_proxy` path are
ungated: a proxy started while the flag was on must stay manageable and
stoppable after it is turned off, and the daemon keeps driving the engine it
was started with. The state file records `gateway`, so management paths bind to
what is *running* rather than to what is configured — and when that gateway's
implementation is not loaded at all, they fall back to the base supervisor
rather than raising, since state reading and SIGTERM are gateway-independent
and a live process must not be left undescribable and unkillable.

`api/proxy/` is the console-free facade (`start`, `stop`, `status`, `models`,
`sync`, `register_loaded_model` / `unregister_loaded_model`, `add_alias` /
`remove_alias` / `list_aliases`, `resolve_gateway`, `list_gateways`);
`cli/_proxy.py` renders it. `register_loaded_model` returning `None` from the
engine means "discovery-driven, do the ordinary sync", so `proxy load` behaves
identically under LiteLLM while a catalog-driven gateway can persist an
activatable binding instead.

## State & Files

| Path                                          | Purpose                                                        |
|-----------------------------------------------|----------------------------------------------------------------|
| `~/.config/sparkrun/proxy.yaml`               | Persistent proxy settings and aliases                          |
| `~/.cache/sparkrun/proxy/litellm_config.yaml` | Generated LiteLLM config                                       |
| `~/.cache/sparkrun/proxy/state.yaml`          | Running proxy state (PID, port, auto-discover PID, start time) |
| `~/.cache/sparkrun/proxy/litellm.log`         | Proxy process stdout/stderr                                    |
| `~/.cache/sparkrun/proxy/autodiscover.yaml`   | Auto-discover process config (written by engine)               |
| `~/.cache/sparkrun/proxy/autodiscover.log`    | Auto-discover process stdout/stderr                            |
| `~/.cache/sparkrun/jobs/*.yaml`               | Job metadata used for endpoint discovery                       |

## Architecture

```
sparkrun proxy start
        │
        ▼
┌──────────────┐     ┌──────────────────┐
│  Discovery   │────▶│  Health Check    │
│  (SSH/meta)  │     │  (GET /v1/models)│
└──────────────┘     └────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ LiteLLM Config   │
                    │   Generation     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐     ┌──────────────────┐
                    │  uvx litellm     │     │  Auto-discover   │
                    │  (subprocess)    │◀───▶│  (background)    │
                    └──────────────────┘     └──────────────────┘
                             │
                             ▼
                    OpenAI-compatible API
                    on localhost:4000

Clients ──▶ localhost:4000/v1/... ──▶ LiteLLM ──▶ backend endpoints
```

The proxy package consists of four modules:

- **`discovery.py`** — Live (SSH) and metadata-based endpoint discovery, health checks, deduplication
- **`config.py`** — `ProxyConfig` class for reading/writing `proxy.yaml` (settings and aliases)
- **`engine.py`** — `ProxyEngine` class managing the LiteLLM subprocess lifecycle, auto-discover, and management API
- **`autodiscover.py`** — Background auto-discovery loop (runs as `python -m sparkrun.proxy.autodiscover`)
