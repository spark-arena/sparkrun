# sparkrun proxy

`sparkrun proxy` manages an OpenAI-compatible inference gateway. Gateway
implementations are pluggable: Sparkrun includes LiteLLM and the SparkRoute
integration, and installed plugins can register others. The selected gateway
owns its model configuration, update mechanism, and supported management features.

## Choose a gateway

For built-in Sparkrun, the channel defaults are:

| Feature channel | Default gateway |
| --- | --- |
| stable / beta | LiteLLM |
| alpha | SparkRoute |

An existing `proxy.gateway` pin takes precedence over channel defaults. Enable a
gateway before selecting it; enabling a second gateway alone does not switch the
selection. When both bundled gateways are enabled and there is no pin, LiteLLM
wins. Alternate application profiles can supply their own feature defaults.

```bash
sparkrun setup features list
sparkrun setup features enable gateway.sparkroute
sparkrun proxy start --gateway sparkroute --host 127.0.0.1
```

`--gateway <name>` persists the selection in `proxy.yaml`. Use `--restart` to
replace a running gateway with new settings or another enabled implementation.
Management commands bind to the gateway recorded in running state, so editing
the pin or disabling a feature does not redirect `stop`, `status`, or updates to
a different implementation. See [SparkRoute channel and update details](SPARKROUTE.md).

## Quick start

Start the gateway, then load a model through it:

```bash
sparkrun proxy start --host 127.0.0.1 --cluster mylab
sparkrun proxy load qwen3-1.7b-vllm --cluster mylab
sparkrun proxy models
curl http://localhost:4000/v1/models
```

You can also start the gateway after launching workloads with `sparkrun run`;
discovery finds healthy endpoints. By default the gateway runs in the background.
Use `--foreground` to keep its process attached to the terminal. To start it at
boot, [run it as a systemd service](#run-as-a-systemd-service).

## Commands

### Start and stop

```bash
sparkrun proxy start --host 127.0.0.1 --port 8080
sparkrun proxy start --hosts 10.0.0.1,10.0.0.2
sparkrun proxy start --gateway litellm --restart
sparkrun proxy start --master-key sk-mykey
sparkrun proxy start --no-auto-discover
sparkrun proxy start --discover-interval 60
sparkrun proxy start --discover-removal-grace-sweeps 1
sparkrun proxy start --cluster mylab        # saved as the discovery cluster
sparkrun proxy start --clear-cluster        # forget it; follow the default cluster
sparkrun proxy start --dry-run
sparkrun proxy stop
```

Start discovers endpoints, asks the selected gateway to prepare configuration,
and launches its process. An already-running proxy is an error unless `--restart`
is set. Explicit settings are saved even when an existing process prevents the
start; the dry run computes a preview without persisting settings.

`--master-key` configures an inference API bearer token. Both bundled gateways
support it; their admin authentication differs. With a key configured, clients
send `Authorization: Bearer <key>`. See [authentication](#authentication-and-admin-console).

The unconfigured legacy bind is `0.0.0.0` and emits a warning. Set the bind address
explicitly for the intended access scope. Stop sends SIGTERM to the recorded
proxy and auto-discover processes; when the proxy runs as a systemd unit, start
and stop go through the unit instead.

`--foreground` follows the gateway across the restarts auto-discovery performs
when the model list changes (LiteLLM reloads its config by restarting). It exits
0 when the proxy is stopped on request (`proxy stop`, Ctrl-C, SIGTERM) and
non-zero when the gateway dies on its own, so a supervisor can restart it.

### Status, models, and sync

```bash
sparkrun proxy status
sparkrun proxy status --json
sparkrun proxy models
sparkrun proxy models --refresh
sparkrun proxy sync
```

Status reports the running implementation, PID, bind address, discovery process,
and models returned by the gateway. It preserves `model_query_error` when
observation fails. `proxy models` returns a nonzero exit status on query failure
in both text and JSON modes; it never reports a successful empty list for that
case. `api.proxy.models()` raises `ProxyQueryFailed`; callers already holding a
`ProxyStatus` can use `require_models()` for the same checked result.

Sync reconciles discovered endpoints and saved settings through the running
gateway. `models --refresh` syncs before listing. Update behavior depends on the
implementation: LiteLLM regenerates a file and restarts when it changes;
SparkRoute reconciles its control plane. Saved activation bindings and model
aliases are interpreted by the selected gateway.

### Load and unload

```bash
sparkrun proxy load qwen3-1.7b-vllm --cluster mylab
sparkrun proxy unload qwen3-1.7b-vllm --cluster mylab
```

Load uses the ordinary workload planning/run API in detached mode. It selects an
available port when the recipe's default is occupied, waits for serving readiness,
and registers with the running gateway. A gateway can save an activation binding;
a discovery-driven implementation can simply sync the new endpoint. If readiness
fails, the workload may still be running, but registration has not completed.

Unload uses the shared stop API. Failed or ambiguous stops retain the proxy
registration. Successful stops remove the registration and sync models when the
proxy is running; if it is stopped, its saved registration remains unchanged.

### Aliases

```bash
sparkrun proxy alias add qwen3-small "Qwen/Qwen3-1.7B"
sparkrun proxy alias remove qwen3-small
sparkrun proxy alias list
```

Aliases are saved to `proxy.yaml` and applied through the running gateway. An
alias whose target is unavailable can remain pending. Persistence can succeed
while applying to the running gateway fails; the command reports that failure
so a later sync can retry. A restart is specific to the implementation.

### Authentication and admin console

```bash
sparkrun proxy ui
sparkrun proxy ui --json
sparkrun proxy admin-token get
sparkrun proxy admin-token set
sparkrun proxy admin-token clear
```

These commands require the corresponding gateway capabilities. SparkRoute
supports an admin console and managed admin token. `ui` prints its URL; it does
not launch a browser. `get` reads the token, `set` generates a replacement, and
`clear` removes the requirement when gateway policy permits. These operations
are not an inference API key rotation API. `ui --issue-token` is a compatibility
option that asks the gateway to create or return console credentials. It can
enable admin authentication; use `admin-token get` for a read-only query.

Sparkrun's LiteLLM integration does not provision the database required for its
admin UI. Its master key provides stateless authentication. SparkRoute has its
own admin listener and credential policy; a master key also requires admin
authentication there. Consult the
[SparkRoute integration reference](../src/sparkrun/plugins/sparkroute/README.md)
for implementation-specific configuration. Unsupported capabilities report an
error instead of pretending a console or token exists.

## Discovery

Start prints which hosts discovery covers and which setting decided it, for
example `Discovery: cluster mylab (from default cluster)`. The first match wins:

1. `--hosts` / `--hosts-file`, for this invocation only.
2. `--cluster NAME`, saved to `proxy.yaml` as `proxy.cluster`.
3. `proxy.cluster` from `proxy.yaml`. If that cluster was deleted, start warns
   and continues with the next source.
4. The default cluster (`sparkrun cluster set-default`), read at every start, so
   changing the default takes effect on the next start.
5. `cluster.hosts` from `config.yaml`.
6. None: metadata-only discovery.

A cluster brings its SSH user and the rest of its definition, not only its
hosts. `--clear-cluster` removes a saved `proxy.cluster`.

Discovery reads saved jobs through `api.list_jobs`, obtains live cross-executor
status through `api.status` when hosts are available, normalizes endpoint
addresses, and checks `/v1/models`. Healthy native workloads can participate
alongside Docker workloads. Metadata-only discovery is possible without a host
list; it does not establish live executor coverage.

When auto-discovery is enabled and supported, a detached process periodically
calls `api.proxy.sync`:

- The default interval is 30 seconds.
- Models and aliases reconcile together.
- Two consecutive misses are required to remove a previously healthy endpoint
  by default. `discover_removal_grace_sweeps: 1` removes it on the first miss.
- Each sweep follows the gateway and PID recorded in state, including restarts.
- Credentials are resolved by the gateway; the sidecar configuration does not
  carry the master key.

A plugin can declare `supports_autodiscover = False` if it manages desired state
itself. Start then warns and disables this sidecar. Both currently bundled
implementations support Sparkrun auto-discovery.

## Run as a systemd service

On a Linux control machine with systemd, install the proxy as a service so it
starts at boot:

```bash
sparkrun proxy systemd install --now          # system unit (sudo once)
sparkrun proxy systemd install --user --now   # user unit (no sudo)
sparkrun proxy systemd status
sparkrun proxy systemd remove
```

The unit runs `sparkrun proxy start --foreground` with no other options, so it
serves whatever `proxy.yaml` says each time it starts. Change settings with
`sparkrun proxy start --port 4100 --restart` (the setting is saved and the unit
restarted) or by editing `proxy.yaml` and restarting. `install --cluster NAME`
saves the discovery cluster; without it the default cluster is used at each start.

- **System unit (default).** `/etc/systemd/system/sparkrun-proxy.service`, run
  as you (`User=`). It starts at boot whether or not anyone is logged in.
  Installing needs sudo once and also writes `/etc/sudoers.d/sparkrun-proxy`,
  which lets you start, stop, and restart exactly that unit without a password.
  If another user already owns `sparkrun-proxy.service` on the machine, yours
  is `sparkrun-proxy-<user>.service`. `status` shows the name in use.
- **User unit (`--user`).** `~/.config/systemd/user/sparkrun-proxy.service`. No
  sudo, but it starts at boot only while logind lingering is enabled; `install`
  tries `loginctl enable-linger` and reports whether it is on.

Once a unit is installed, `proxy start` starts it (or restarts it with
`--restart`) instead of launching a separate process, `proxy stop` stops it, and
`proxy status` shows `Managed by: systemd (<unit>)`. Stopping does not disable
the unit: it starts again at the next boot. Use `proxy systemd remove` to remove it.
`install --now` replaces a proxy that was started outside the unit.

The service runs without an `ssh-agent`, so discovery needs a key that loads
without a passphrase. Set `ssh.key` in `config.yaml` if your default key is not
usable that way. Logs go to the journal (`journalctl -u sparkrun-proxy`, or
`journalctl --user-unit sparkrun-proxy` for a user unit).

With the SparkRoute gateway, routes are not reconciled when the unit starts the
gateway; they are applied at the first discovery sweep.

## Configuration and state

For built-in Sparkrun, settings live in `~/.config/sparkrun/proxy.yaml`:

```yaml
proxy:
  port: 4000
  host: 127.0.0.1
  master_key: null
  # gateway: sparkroute  # optional pin; the named gateway must be enabled
  # cluster: mylab        # saved by an explicit --cluster; otherwise the default cluster
  auto_discover: true
  discover_interval: 30
  discover_removal_grace_sweeps: 2

aliases:
  my-model: "Qwen/Qwen3-1.7B"
```

CLI overrides persist for subsequent invocations. Writers lock and merge changed
sections so a discovery sweep does not overwrite a concurrent alias or listener
change. Gateway-specific settings also belong to `proxy.yaml`; see the relevant
integration guide.

Default paths below are relative to `~/.cache/sparkrun/`. Other application
profiles and cache settings change the root.

| Path | Purpose |
| --- | --- |
| `proxy/state.yaml` | Process identity, gateway selector, listener and discovery state, and the supervising systemd unit (if any) |
| `proxy/.distribution` | Application ownership of the gateway state directory |
| `proxy/litellm.log` | LiteLLM process output |
| `proxy/litellm_config.yaml` | Generated LiteLLM model configuration |
| `proxy/sparkroute.log` | SparkRoute process output |
| `proxy/autodiscover.yaml` | Discovery sidecar configuration |
| `proxy/autodiscover.log` | Discovery sidecar output |
| `jobs/*.yaml` | Job metadata used for discovery |

Each state directory manages one gateway process. Different applications must
use separate managed gateway directories. Application/controller descriptors
provide groundwork for shared external services; they do not yet make one
managed SparkRoute instance reconcile multiple application controllers. See
[application identity](APPLICATION_PROFILES.md#plugin-identity-and-shared-services).

## LiteLLM implementation

The LiteLLM engine runs the pinned `litellm[proxy]` package through `uvx`; it does
not require a permanent LiteLLM installation. The pin and launch command live in
`src/sparkrun/proxy/engine.py`.

Sparkrun uses generated configuration for model mutations, because it does not
provision LiteLLM's database-backed model store. Changes regenerate the config
and restart the process; unchanged desired state skips the restart. Management
queries report what the process is serving. These are LiteLLM implementation
details, not requirements for other gateway plugins.

## Python API and gateway plugins

`sparkrun.api.proxy` is the frontend-independent API: `start`, `stop`, `status`,
`models`, `sync`, `register_loaded_model`, `unregister_loaded_model`, alias
operations, `ui`, `admin_token`, `resolve_gateway`, and `list_gateways`.
The CLI only renders these results. For example:

```python
from sparkrun.application import initialize
from sparkrun import api

context = initialize()
enabled = api.proxy.list_gateways(sctx=context)
selected = api.proxy.resolve_gateway(sctx=context)
```

Omitting `sctx` initializes the default application context, including installed
plugins. Selection and start use the saved gateway pin unless an explicit
name overrides it. Management operations use the gateway recorded in process
state, even when the saved preference has changed. Low-level registry selection in `sparkrun.proxy.gateway` does not
initialize the application and remains usable during plugin bootstrap.

Gateway plugins register a deferred class loader with
`register_gateway(name, feature_flag=..., loader=...)` in
`sparkrun.proxy.gateway`. Repeated registration of the same provider is
idempotent; distinct providers claiming the same name raise `PluginConflictError`.
Select installed plugins through [the plugin contract](PLUGINS.md).

| Component | Responsibility |
| --- | --- |
| `api/proxy/` | Options, results, selection, discovery and gateway dispatch |
| `proxy/gateway.py` | Registration, availability and selection |
| `proxy/_supervisor.py` | Shared process/state ownership and discovery lifecycle |
| `proxy/discovery.py` | Job/status discovery, health checks and deduplication |
| `proxy/config.py` | Persistent settings and aliases |
| `proxy/engine.py` | LiteLLM implementation |
| `plugins/sparkroute/` | Vendored SparkRoute integration |
| `proxy/autodiscover.py` | Optional reconciliation sidecar |

`sparkrun.proxy.supervisor.GatewaySupervisor` is the supported lifecycle base.
It supplies the common process contract. Implementations
provide configuration, model reconciliation and management capabilities such as
`supports_autodiscover`, `wants_proxy_config`, and `data_plane_authenticated`.
Start checks availability even for a dry run. Existing processes remain
manageable after a feature is disabled; when an implementation is unavailable,
the base supervisor can still inspect state and stop the recorded process.
This fallback also covers plugin initialization failures and declared operational
failures while constructing the provider: the failure is logged,
status reports model enumeration as unavailable, and process stop remains usable.
The CLI's `proxy status`, `proxy stop`, and `proxy models` reach this same recovery
path without requiring successful plugin bootstrap; models fail explicitly when
enumeration is unavailable.
An invalid application profile still raises instead of accessing another
application's state.

Process recovery cannot change the live model configuration. `sync`,
`models --refresh`, and registration updates raise `GatewayUnavailable` before
endpoint discovery if the provider is unavailable. `sync(require_running=True)`
and registration updates still return a no-op when the process is stopped.
Alias add/remove save local settings first; if the live provider is unavailable,
they then raise `GatewayUnavailable` and the CLI reports both the saved edit and
the failed application. Restore the plugin/configuration and sync to retry.
The same saved-edit policy applies to `ProxyUpdateFailed`. Initialization failures
occur before alias persistence and do not produce a saved-edit message.
Console and token operations retain `ProxyUnsupported` when the resolved supervisor lacks those
optional capabilities.

Start previews configuration before stopping an existing process and writes
generated configuration or discovery snapshots only after shutdown succeeds.
Duplicate-start rejection and shutdown timeout preserve those files. Explicit
settings still persist to `proxy.yaml` before the lifecycle decision unless
`persist=False` or `dry_run=True`. `ProxyStartResult.config_path` is a string for
a generated file, or `None` for a fileless gateway or dry run.
Expected provider configuration, preparation, shutdown, and launch failures raise
`ProxyStartFailed` with the original cause. Availability failures remain
`GatewayUnavailable`; interrupts and unrelated programming errors propagate.

### Gateway plugin contract

`sparkrun.proxy.contracts` is the supported import for `ProxyModel`,
`GatewayOperationError`, `GatewayQueryError`, and optional console/credential protocols. `ProxyModel` is
the same immutable class exported by `api.proxy`; providers and callers do not
need separate copies of the model record.

```python
from sparkrun.proxy.supervisor import GatewaySupervisor
from sparkrun.proxy.contracts import ProxyModel, GatewayQueryError

class ExampleGateway(GatewaySupervisor):
    gateway_name = "example"

    def query_models(self) -> tuple[ProxyModel, ...]:
        # Fetch and validate the provider's response here. Translate a failed
        # query to GatewayQueryError with a diagnostic containing no secrets.
        return (ProxyModel("example-model", "http://worker:8000/v1", 32768),)
```

`query_models()` returns a tuple, with an empty tuple meaning a successful empty
observation. It raises `GatewayQueryError` when enumeration is unavailable.
`status()` preserves that diagnostic alongside process state; `models()` exposes
it as `ProxyQueryFailed`. Provider wire dictionaries stay below the API boundary.

Both bundled gateways implement this typed contract directly. LiteLLM translates
its `/model/info` response inside its provider; SparkRoute translates `/v1/status`
in its upstream plugin. `list_models_via_api()` and the supervisor's mutable
`model_query_error` side channel are removed in 0.4. Providers must implement
`query_models()`; the default supervisor raises `GatewayQueryError` when it cannot
enumerate models. `api.proxy.status().model_query_error` remains the public
snapshot diagnostic, populated from that exception.

Optional capabilities are independent structural protocols:

| Protocol | Contract |
| --- | --- |
| `GatewayConsole` | Read-only `ui_url`, `admin_bind_host`, `admin_exposed`, and `admin_auth_required` properties describing the live console. |
| `GatewayConsoleCredentials` | `issue_ui_credential() -> str` creates or returns a console credential; it may enable authentication. |
| `GatewayAdminToken` | `admin_token(*, rotate=False, clear=False) -> str \| None`: read, replace, or disable authentication when provider policy permits. `None` means open access. |

A gateway can offer a read-only console without credential issuance, or token
management independently of a console. Missing capabilities raise
`ProxyUnsupported`. Reconciliation, credential operations, and console properties
raise `GatewayOperationError` for expected failures; the API translates management
failures to `ProxyUpdateFailed`. Model enumeration raises its subtype
`GatewayQueryError`, which becomes `ProxyQueryFailed` for model-list callers.
Diagnostics must not include secrets. Other exceptions, including a bare
`RuntimeError` or provider-thrown `NotImplementedError`, propagate as provider
bugs. Stop hooks also use `GatewayOperationError` and become `ProxyUpdateFailed`.

The bundled SparkRoute plugin implements these contracts directly in its upstream
source. Registry resolution returns that same class; no host adapter changes its
errors or behavior. Transport/authentication/retry failures retain their upstream
metadata and causes. Applications should use `api.proxy`; providers import the
supported `supervisor` and `contracts` modules. Legacy `_supervisor` imports remain
aliases for the shared classes.

Passing both `rotate=True` and `clear=True` is invalid and raises `ValueError`
before dispatch. Admin-token operations do not rotate the inference API key.
