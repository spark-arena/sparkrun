# Sparkrun daemon

The daemon exposes `sparkrun.api` through authenticated JSON and SSE. It keeps
catalog queries and monitor sessions warm and runs each mutation or long operation
in a fresh, independently supervised process. One job retains its context and live
plan through approval, execution, readiness, and post-launch hooks.

The local daemon supports HTTP/TCP on Linux, macOS, and Windows. This is a
single-user controller, bound to one OS account, application profile, and canonical
config file. It controls Linux/Spark targets over SSH; local Linux setup and GPU
workloads retain their existing platform requirements. Optional gateway discovery
and workload lifecycle integrations are described below. The full operation and
route inventory is in [DAEMON_API.md](DAEMON_API.md). Native macOS/Windows execution is covered by the
portable-daemon CI matrix; those results must pass before releasing platform support.

## Starting and connecting

The `daemon` feature is enabled by default on alpha, disabled on stable/beta. To
opt in explicitly:

```bash
sparkrun setup features enable daemon
sparkrun daemon start
sparkrun daemon status
sparkrun daemon call catalog.clusters
sparkrun daemon stop
```

Use `sparkrun daemon --config /absolute/config.yaml ...` for another binding.
`serve` runs in the foreground; `restart` reconnects existing workers. `stop`
stops the API only. `stop --cancel-jobs` first drains admission and waits for
cancellation; if it times out, the API remains available for recovery.

On macOS and Windows, the default listener is authenticated loopback TCP on an
OS-assigned port. Linux retains its private Unix socket by default; use
`--no-unix` for the same TCP-only mode. Same-UID Unix peers have local admin and
credential-prompt permissions. CLI discovery reads the binding's protected local
connection file when no Unix listener is present. `daemon status` and recovery
commands do not load external plugins.

TCP is bearer-authenticated, including loopback:

```bash
sparkrun daemon serve --no-unix --tcp 127.0.0.1:0
# Optional discovery file for another local client:
sparkrun daemon serve --no-unix --tcp 127.0.0.1:0 --conn-file /private/path/daemon.json
sparkrun daemon token create automation --scope operate --expires-in 3600
sparkrun daemon token list
sparkrun daemon token revoke TOKEN_ID
```

TCP serving automatically writes a private connection file for local CLI discovery.
`--conn-file` additionally publishes a file for clients and refuses to overwrite an
existing one. Files contain URL, token, protocol, config binding, and process
identity. Credentials use mode 0600 on POSIX and protected owner-only DACLs on
Windows; Windows storage must support ACLs, and reparse points are rejected.
A credential is valid only while its API process lives; normal shutdown removes
it. After a crash, desktop clients use core's native process identity checks to
verify stale state before reclaiming it. Local tokens rotate on restart while
keeping the same account principal, preserving job ownership and idempotency.
Manually created tokens survive API restart until revoked/expired; only hashes
are stored server-side.

Remote TCP requires a certificate and matching key by default:

```bash
sparkrun daemon serve --tcp 0.0.0.0:8443 \
  --tls-cert /private/server.crt --tls-key /private/server.key
```

To explicitly allow remote TCP without TLS, pass `--allow-insecure`:

```bash
sparkrun daemon serve --no-unix --tcp 0.0.0.0:8080 --allow-insecure
# IPv6 wildcard bind:
sparkrun daemon serve --no-unix --tcp '[::]:8080' --allow-insecure
```

Bearer authentication is still required; tokens and request/response content are
transmitted unencrypted. The flag permits plaintext but does not disable TLS when
both certificate and key are provided, or accept an incomplete TLS configuration.
`daemon restart` preserves the explicit opt-in with the other listener settings.
There is no config/environment setting that implicitly permits remote plaintext.
When connecting remotely to a wildcard bind, use the host's reachable address.

A Linux Unix listener remains plaintext and peer-authenticated. TLS and remote
binds also keep a separate bearer-authenticated loopback HTTP listener for local
CLI recovery, independent of the remote address and certificate names/trust stores.
All listeners share one application lifecycle. Remote plaintext binds without
`--allow-insecure` are refused. Proxy headers do not affect transport or auth policy.

## Explicit client endpoints

Applications can use the HTTP/SSE API or `sparkrun.daemon.client.Client` directly.
The CLI is also a reference client for debugging those requests. Put shared
connection options before `call`, `status`, or `jobs`:

```bash
sparkrun daemon --url https://controller:8443 --token-file /private/daemon.token status
sparkrun daemon --url https://controller:8443 --token-file /private/daemon.token \
  call catalog.clusters
sparkrun daemon --url https://controller:8443 --token-file /private/daemon.token \
  jobs events JOB_ID --after 12
sparkrun daemon --socket /private/path/api.sock jobs list
```

`--url` takes an HTTP(S) origin, including an optional port; URL credentials,
path prefixes, queries, and fragments are rejected. `--socket` selects a local
Unix socket, not a socket on another machine. Windows clients use TCP. Endpoint
options are mutually exclusive with each other and with `--config`. Without an
explicit endpoint, existing local discovery applies.

Supply either `--token TOKEN` or `--token-file PATH`. The file contains just the
bearer token, with an optional trailing newline, not a connection JSON document.
If neither is provided, `--url` uses `SPARKRUN_DAEMON_TOKEN` when set. Explicit
credentials take precedence over that environment variable. The variable does
not affect local discovery or Unix connections. Token files avoid putting a
credential in command arguments or shell history. Create a token on the daemon
host with `daemon token create`, then provision it to the client.

HTTPS uses certificate and hostname verification. Use `--ca-cert /path/ca.pem`
with `--url https://...` for a private CA or trusted self-signed certificate;
there is no option to disable verification. Explicit `http://...` addresses use
plaintext; a non-loopback server requires `serve --allow-insecure` as described
above. The Python reference client supports the same CA selection through
`Client(url=..., token=..., ca_cert=...)`.

Explicit clients check the server protocol without loading local plugins,
requiring the local daemon feature, comparing local config bindings, or checking
the server's PID on the client host. Unreachable endpoints, authentication
failures, and protocol mismatches exit with an error; they never fall back to a
local daemon or resubmit an operation. `call --wait` and all `jobs` subcommands
retain their normal prompt, event replay, and cancellation behavior.

`serve`, `start`, `stop`, `restart`, `token`, `systemd`, and `resources` remain
local commands and reject client connection options.

## Operations, jobs, and prompts

Authenticated `GET /v1/ops` publishes operation names, permissions, execution
classes, ownership (`owner`, `local_name`), HTTP alias paths
(`http_path`), and closed request/result schemas.
`sparkrun daemon help` renders the same catalog in the terminal: every
operation grouped by namespace, and with a name (`daemon help workloads.run`)
its kind, required scope, HTTP alias, request fields with notes, result shape
and a ready-to-edit `daemon call` example. A namespace (`daemon help setup`)
filters the list; `--json` prints the `/v1/ops` entries. It reads this
installation's catalog, or with `--url` / `--socket` that daemon's (limited to
what the token may call).
`/openapi.json` and `/docs` describe the wire API. `/v1/health` reveals only
health/protocol; `/v1/info` requires auth.

```bash
sparkrun daemon call cluster.create \
  --data '{"name":"lab","hosts":["spark-a","spark-b"]}' --wait
sparkrun daemon call workloads.run \
  --data '{"recipe":"registry/recipe","cluster":"lab","confirm":true}' \
  --idempotency-key launch-lab-001 --wait
sparkrun daemon jobs list
sparkrun daemon jobs show JOB_ID
sparkrun daemon jobs events JOB_ID --after 12
sparkrun daemon jobs answer JOB_ID
sparkrun daemon jobs cancel JOB_ID
```

`POST /v1/ops/{name}` returns a result directly for audited queries, or HTTP 202
with `job_id` and `events_url` for jobs. Use `Idempotency-Key` for retries of the
same request; different arguments with the same principal/key return 409.
Inspect the existing job after an uncertain response. Clients must not fall back
to local execution or submit a new mutation merely because a connection failed.

Every core query or job also has a literal `POST /v1/<operation>` alias,
with dots replaced by slashes. Plugin operations use
`POST /v1/plugins/<plugin-id>/ops/<local-operation-path>`. Aliases use the same
JSON body and dispatcher as `/v1/ops/{name}` on both TCP and Unix sockets:

| HTTP alias | Operation | Response |
| --- | --- | --- |
| `POST /v1/cluster/create` | `cluster.create` | 202, job acknowledgement |
| `POST /v1/cluster/update` | `cluster.update` | 202, job acknowledgement |
| `POST /v1/cluster/delete` | `cluster.delete` | 202, job acknowledgement |
| `POST /v1/catalog/clusters` | `catalog.clusters` | 200, cluster listing |
| `POST /v1/workloads/run` | `workloads.run` | 202, job acknowledgement |
| `POST /v1/plugins/k8s/ops/list_nodes` | `plugins.k8s.list_nodes` | 202, job acknowledgement |

For example, with a daemon URL and admin token in your environment:

```bash
curl --fail-with-body "$SPARKRUN_DAEMON_URL/v1/cluster/create" \
  -H "Authorization: Bearer $SPARKRUN_DAEMON_TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: create-lab-001' \
  --data '{"name":"lab","hosts":["spark-a","spark-b"]}'
```

Retrying through either URL with the same principal, arguments, and idempotency
key returns the same job. Job inspection, cancellation, prompts, event streams,
log tails, and monitoring keep their existing endpoints. Queries also use POST
and accept a JSON object (`{}` for no arguments). The generic operation endpoint
and `sparkrun daemon call` remain supported.

Each alias has a stable OpenAPI operation ID, request schema, permission metadata,
and response schema. A job's HTTP response describes its acknowledgement;
`x-sparkrun-operation.result` references its final result schema, available after
completion through the job endpoint. Queries return their declared result directly.
Bearer authentication is described for API tooling, including the Swagger UI's
Authorize control. `/v1/ops` filters discovery by permissions and feature flags;
OpenAPI describes all registered operations, whose permissions and flags are still
enforced when called.

A run preview is display-only. An approved run executes the same resolved Python
plan, not a reconstruction of its JSON preview. Jobs requiring input enter
`waiting_input`; answers go to the authenticated live worker. Approval/decision
answers are booleans; secret answers are strings. Prompts time out after 300
seconds by default and fail closed. Secret values stay in worker memory and are
redacted from subsequent wire events/results; prompt metadata and answered state
are durable. Do not put credentials in request arguments, recipe overrides, or
freeform log messages. Named secret fields are rejected/redacted; arbitrary text
is not a general-purpose secret detector.

Permissions are hierarchical (`read`, `operate`, `admin`) with independent
`--secret` permission for credential prompts. Non-admin tokens access only their
own jobs and need the operation's scope to inspect or answer them. Token revocation
also ends existing streams. TCP recipe-hook trust requires server opt-in via
`serve --allow-request-trust` (or `daemon.allow_request_trust`) and a request with
explicit `trust: true`; same-UID Unix callers can make that decision locally.

Terminal states are `succeeded`, `failed`, `cancelled`, and `lost`. Cancellation
means execution ended, not that remote effects were rolled back. A worker lost
after authorization is never rerun automatically. After reconciling remote effects:

```bash
sparkrun daemon resources list
sparkrun daemon resources reconcile OWNER_ID --acknowledge-effects
```

Core resource reservations also cover standalone mutating API/CLI operations.
They serialize shared controller/documents and overlapping host sets. Unresolved
destinations use a conservative controller-wide reservation. Host keys normalize
user prefixes and case; different SSH aliases for the same machine are not yet
resolved through DNS/SSH config, so use consistent destination names. Dead-owner
reservations remain until explicit reconciliation.

## Streams, restart, and storage

- `GET /v1/jobs/{id}/events`: durable SSE, sequence IDs, `Last-Event-ID` replay;
  HTTP 410 when the cursor precedes retained history. Fetch the job, then resume
  from the available history. Results over the event limit are fetched separately.
- `GET /v1/events`: bounded job-state snapshots for reconnecting clients.
- `GET /v1/streams/logs`: cancellable log tail; bounded buffering with gap events.
- `GET /v1/streams/monitor`: reference-counted monitor sessions, latest frame,
  five-second teardown linger. Disconnect/cancellation closes quiet reads too.
  The first event is a `history` backfill (`series`: per-host recent samples from
  the shared session, `monitor.history_points` deep); `frame` events follow.

Jobs, events, prompt metadata, and idempotency records live in
`<config-root>/daemon/<binding>/jobs.sqlite3` (WAL). Local discovery and any Unix
socket use the private runtime directory, with a short private OS temporary path
on POSIX if Unix socket length requires it.
Directories are private and unsafe ancestors/symlinks are rejected. Unknown
exceptions produce a generic wire error and private stack locations, without
exception values or local-variable dumps.

Storage has not been migrated wholesale to SQLite:

| State | Storage | Callers |
| --- | --- | --- |
| Workload placement, post-launch effect and lifecycle receipts | `<config-root>/workload-receipts.sqlite3` | Shared workload APIs and plugin adapters |
| Durable operation jobs, events, results, prompts, retry keys | `<config-root>/daemon/<binding>/jobs.sqlite3` | Daemon and its workers |
| Resource reservations and owner identities | `<config-root>/operations.sqlite3` | Shared by daemon and standalone mutating API/CLI operations |
| Workload metadata (recipe, hosts, deployment identity) | `<cache-root>/jobs/*.yaml` | Existing shared workload APIs, including daemon calls |
| Configuration, cluster definitions, benchmark state | Existing YAML/JSON files | Existing core APIs |

A daemon `job_id` identifies an operation invocation. A `workload_id`
identifies the deployment produced or affected by an operation. For example,
a daemon `workloads.run` job may finish successfully while the deployed inference service
keeps running; its result includes the workload identity. `daemon jobs list`
shows operation history, whereas `daemon call workloads.list` and Python
`sparkrun.api.list_jobs()` list saved workload metadata. Ordinary standalone CLI
operations do not create daemon job-history rows. The daemon reuses core workload
storage; its durable operation journal is additional state, not a replacement
workload catalog. Audited queries return directly without creating job rows.

API restart adopts workers through native process creation identities and a nonce
challenge over bounded loopback TCP. Linux retains boot ID + `/proc` start time;
macOS/Windows use process creation time. PID-only checks are never sufficient.
Jobs write events/results directly while the API is down. Host reboot, worker
crash, and replacing a Python environment do not preserve an in-memory operation.

`POST /v1/shutdown` requires admin authorization and stops only the API. CLI and
desktop use it on every OS. Cancellation first requests cooperative cleanup;
unresponsive workers become `lost` after forced termination. POSIX process mode
tracks verified descendants, including nested sessions; arbitrary orphaned or
malicious children are outside that tree guarantee. Linux systemd uses independent
per-worker cgroups. Windows workers own named Job Objects: stopping the API does
not close their containment handles, while worker crashes and forced cancellation
terminate contained children. Successful jobs release crash cleanup so intentional
persistent services can outlive the workflow. Windows Job Object assignment must
succeed before execution; an incompatible outer supervisor fails closed.

## Services and software updates

Foreground serving, background `start`/`stop`/`restart`, and GUI-owned lifetimes are
portable. Native macOS LaunchAgent and Windows service/autostart installers are a
separate follow-up; they must preserve independently running workers across API
service restarts. Linux has the following systemd installer:

```bash
sparkrun daemon systemd install --now
sparkrun daemon systemd status
sparkrun daemon restart
sparkrun daemon systemd uninstall
```

The installer validates a user unit, binds it to this interpreter/config, and
refuses to replace unrelated units. API and per-job units have separate cgroups;
restarting/stopping the API service does not kill workers. Service mode requires
`--worker-supervisor systemd`; the installer sets it. User lingering is an
explicit administrator/user choice; installation does not enable it. For a remote
service, add the TCP and TLS flags (or explicit `--allow-insecure`) to its
`ExecStart` override. Certificates and tokens
are provisioned separately.

Daemon and worker runtime leases pin core identity and managed-plugin generation
IDs. Plugin activation requires API restart; admission enters maintenance when
runtime identity changes. In-use generations cannot be removed or pruned, and
retention never exceeds `plugins.retain_generations` (default **2**, current/previous).
An update that would evict a leased generation waits for the user to drain/stop
its owners. Unmanaged external directories cannot provide immutable-generation
guarantees.

`sparkrun update`, `setup update`, and `setup install` refuse to replace an interpreter installation
while any daemon/worker lease in that installation is live, including another
config binding. Installation checks the destination tool environment even when
invoked from a temporary `uvx` environment. Stop/drain all such instances first,
update, then restart. Coordination is stored under `.sparkrun-runtime` in the
OS account's home/profile directory, keyed by user and canonical interpreter
installation. It is shared across application/config bindings and does not follow
`HOME`, `USERPROFILE`, XDG, or temporary-directory overrides. The directory uses
private permissions (an owner-only ACL on Windows).

When upgrading from a 0.4.0 pre-release that stored runtime leases in temporary
directories, stop and restart all daemons/workers. Updates check old records in
known temp roots and the current config as a transition safeguard, but cannot
discover every custom temp root used by old processes on other config bindings.

Normal plugin compatibility evaluation stays in plugin management/update; daemon startup
does not add a compatibility audit. The existing default plugin-update behavior
and `--no-update-plugins` remain intact.

## Configuration and bounds

Settings are under `daemon:` in the application config. Restart the API after
changing admission/retention settings.

| Key | Default | Meaning |
| --- | ---: | --- |
| `max_services` | 2 | Separate capacity for gateway/discovery service jobs; use at least 2 for both |
| `max_jobs` | 4 | Concurrent ordinary jobs, including waiting prompts; service workers have a separate budget |
| `max_queued_jobs` | 128 | Accepted queued jobs |
| `max_query_threads` | 8 | Audited query threads |
| `max_queued_queries` | 32 | Total admitted active/waiting queries |
| `query_timeout` | 30 | Cooperative query deadline, seconds |
| `max_streams` | 64 | SSE subscriptions |
| `max_body_bytes` | 1048576 | HTTP request-body limit |
| `max_events` | 4096 | Retained events per job |
| `max_event_bytes` | 8388608 | Retained event bytes per job |
| `cancel_grace` | 15 | Seconds before forced worker cancellation |
| `worker_start_timeout` | 60 | Seconds allowed for worker registration |
| `prompt_timeout` | 300 | Seconds allowed for a prompt answer |
| `job_retention_days` | 7 | Terminal result/event retention |
| `retained_jobs` | 1000 | Maximum retained terminal jobs |
| `discovery_interval` | 0 | Opt-in endpoint snapshots; 0 disables, minimum 5 seconds |

Count and byte-limit settings require positive integers (`max_event_bytes` must
be at least 1024); fractional counts and booleans are rejected at startup.
Worker/query deadlines and retention durations accept positive finite fractions.
Limits are normalized at startup.

Results have a 2 MiB cap and individual events a 256 KiB cap. Log subscription
queues hold 128 lines and emit gaps on overflow. Query caches hold 128 entries
for at most two seconds and invalidate on config revision/job events. Query
cancellation does not release a thread slot until underlying work ends; blocking
third-party calls still need their own timeout. Idempotency tombstones remain
for 30 days and active jobs retain their keys beyond that window. Never blindly
retry an older uncertain mutation after its retry window expires.

## Anonymous usage telemetry

Daemon `workloads.run` and `benchmarks.run` operations retain the existing core API telemetry
emitters and payloads. All usage events now include `via_daemon`: `true` when
emitted inside daemon-managed operation execution, `false` for direct CLI/Python
execution. Attribution follows nested API work, worker processes, query threads,
and stream producers through operation context. It is not inferred from a
client-supplied header, job name, or process-wide environment flag.

Each serving process emits one `daemon_serve` event after all its listeners are
ready. This covers foreground `serve`, background `start`, `restart`, systemd,
and desktop-owned serving. Starting an already-running daemon does not produce
another event; rejected startup does not report successful serving. Multiple
listeners share one event. Its fields are `unix_socket`, `tcp`, `tls`,
`remote_tcp`, `worker_supervisor`, `parent_managed`, `startup_ms`, and the usual
system/version/installation envelope. Serving is not an operation request, so
this event has `via_daemon: false`. There are no periodic heartbeat events.

No addresses, ports, socket/config paths, bearer tokens, principal IDs, job IDs,
or raw client headers are added to usage events. The durable SQLite job journal
is not uploaded. Host/GPU monitoring streams remain separate from anonymous
usage reporting.

Policy and installation identity come from the daemon host's application profile
and bound config. `telemetry.enabled: false` disables reporting unless an
explicit environment override forces it on. Set `SPARKRUN_NO_TELEMETRY=1` in the
serving process environment to opt out; remote client environment settings are
not transmitted to the server. Ordinary process workers inherit the environment;
systemd workers now explicitly receive telemetry opt-out, endpoint, key, and
timeout overrides, including application-profile variable names and aliases.
For installed services, configure the daemon unit environment or bound config;
an environment variable on the client running `daemon start` does not configure
the systemd service. New workers read the config; already-running jobs retain
their existing context.

Serving-event delivery runs outside the API event loop and is best-effort, using
the existing telemetry timeout and failure handling. Run/benchmark events keep
their existing emission points; a run event can precede readiness checks or
post-launch hooks and is not a report of final daemon job success.

## Optional clients and services

`SPARKRUN_DAEMON=auto` enables selective read-only CLI passthrough for recipe
listing/search and JSON cluster status. Discovery failures can fall back before
submission; submitted operations are not retried locally. Normal CLI behavior
remains the default.

`daemon.discovery_interval` enables advisory endpoint snapshots at
`/v1/snapshots/discovery.snapshot`; it defaults to zero to avoid an unused duplicate
sweep. A lost periodic snapshot is replaced after the configured interval once
its registered worker is confirmed gone. The lost job remains available for
diagnosis; an uncertain or still-live worker blocks the retry. This recovery
applies only to the periodic read-only observation, never to mutating jobs.
Gateway discovery instead uses `proxy.discovery_backend: daemon` and the
proxy's own scope, interval and removal grace. The gateway and watcher run as
separate service jobs, share the ordinary job/event machinery, and survive an
API-process restart. See [gateway discovery and workload transport](#gateway-discovery-and-workload-recovery)
below. The old `daemon.proxy_autodiscover` switch no longer schedules a watcher.

MCP support is disabled. Its adapter implementation is retained for possible
future reuse, but no MCP command or service entry point is registered.

The desktop application now uses this API, with an Operations view for reconnect,
cancellation, and prompts. It attaches without owning an existing verified daemon;
a GUI-owned API exits with the GUI while durable jobs continue. Local spawning
uses TCP-only mode on Linux, macOS, and Windows. GUI shutdown never requests a
process-tree kill. Native builds and the core process-lifetime contracts are
covered by macOS/Windows CI in addition to Linux.

Reusable worker pools remain deferred. Warm queries and shared monitoring provide
the initial reuse without retaining plugin state between independent jobs.

## Extending the API

Plugins register framework-free `DaemonOp` metadata through `sparkrun.plugins`.
Each plugin declares local operation names. The loader assigns ownership before
importing the module, covering both module-level declarations and `register(v)`.
For a plugin identified as `example`, this registers `plugins.example.status`
and mounts `POST /v1/plugins/example/ops/status`:

```python
from sparkrun.plugins import DaemonOp, register_plugin_daemon_op

SPARKRUN_PLUGIN_API_VERSION = 1


def load_status():
    from .handlers import status
    return status


def register(v):
    register_plugin_daemon_op(DaemonOp(
        name="status",
        handler_loader=load_status,
        kind="query",
        scope="read",
        request_schema={"type": "object", "properties": {}, "additionalProperties": False},
        result_schema={"type": "object", "properties": {"ready": {"type": "boolean"}},
                       "required": ["ready"], "additionalProperties": False},
        description="Report readiness without changing state.",
    ))
```

Handlers receive decoded JSON, an operation-local context, and a prompt callback.
Declare the appropriate execution kind, permission, feature flag, resource policy,
and result schema. Do not import FastAPI/Pydantic/uvicorn from scanned plugin
modules. The in-tree k8s plugin provides a job example.

A local name uses lowercase identifier segments separated by dots, such as
`workloads.activate`. The canonical operation becomes
`plugins.<plugin-id>.workloads.activate`; the HTTP suffix is `workloads/activate`.
Two plugins can each declare `status`, `cluster.create`, or `jobs.list` without
colliding with each other or with core. `plugins` is reserved as a local root.
Stream operations keep dedicated endpoints and have `http_path: null`.

The ID comes from existing metadata: the managed manifest name, installed
entry-point name, or bundled/directory discovery name. A nearby `plugin.toml`
that names the exact module can supply its logical name; for an installed entry
point it must agree with that entry point. ASCII case and runs of `_`, `.`, and
`-` normalize to lowercase hyphens, so `oci_relay` becomes `oci-relay`. IDs start
with a letter, end with a letter/digit, and contain at most 100 input characters.
Paths, traversal-like names, `core`, and `plugins` are rejected. Versions and
installation generations are excluded from IDs. `sparkrun plugins list --json`
includes `plugin_id`; existing store paths and config keys retain their names.
Duplicate selected IDs across sources fail before plugin imports, with both
origins in the error.

`owner` is optional in declarations; an explicit value must match the loader's
ID. Registration outside a loader context fails. Embedded applications can use the
public `application_daemon_registration()` context manager, which assigns
`app-<application-profile-id>`. Core operations use a separate internal context.
Equivalent declarations from the same provider can be repeated; conflicting
handlers or metadata raise `PluginConflictError`, the rule every plugin registry
follows. Failed plugin imports/hooks roll back daemon
registries alongside the other plugin registries. These are contribution
ownership rules for trusted Python extensions, not a sandbox for plugin code.

### Plugin JSON adapters

`DaemonRoute` supplies a bounded JSON envelope decoder. Register a relative path
through `register_plugin_daemon_route`; `example` declaring `compat/status`
receives `POST /v1/plugins/example/routes/compat/status`:

```python
from sparkrun.plugins import DaemonRoute, register_plugin_daemon_route


def load_decoder():
    return lambda envelope: ("status", {})


def register_adapter():  # Call from register(v).
    register_plugin_daemon_route(DaemonRoute(
        path="compat/status",
        decode_loader=load_decoder,
        operations=("status",),
        request_schema={"type": "object", "properties": {}, "additionalProperties": False},
    ))
```

The decoder must return one of the declared target spellings plus its arguments.
Local targets resolve within the plugin. Use `core:cluster.create` for an explicit
core target, or `plugins.other.status` for a cross-plugin target. The selected
operation's validation, permissions, feature gates, secret/trust policy, and job
dispatch still apply. The adapter namespace grants no extra authority.

Only literal relative paths are accepted: no absolute paths, traversal, encoded
separators, parameters, wildcards, queries, or fragments. Adapters cannot install
arbitrary ASGI handlers. Their separate `routes` branch prevents them from
shadowing generated `ops` aliases. All targets must exist when the daemon starts;
path collisions fail startup. Changing plugins or their declarations requires a
daemon restart. Discovery and OpenAPI use canonical operation names and separate
schema components/tags for each owner; clients should use the advertised
`http_path` rather than constructing it themselves.

### Kubernetes operations

The bundled Kubernetes plugin uses the same canonical naming convention:

```bash
sparkrun daemon call plugins.k8s.list_nodes --data '{"context":"lab"}' --wait
```

This dispatches the same operation as `POST /v1/plugins/k8s/ops/list_nodes`.
Discovery, persisted jobs, and retries all use `plugins.k8s.list_nodes`.

`daemon/coverage.py` accounts for public callable API entry points. Exclusions
include live Python/session helpers, TTY `ssh-copy-id`, arbitrary low-level k8s
launcher specifications, and credential-bearing proxy UI/token helpers. Readiness
and hooks are exposed as steps of `run`, not serializable live handles. Explicit
DTO allowlists/schemas live in `api/_wire*.py`; unsupported objects fail closed.

Operation request/result schemas and route request schemas are checked when their
declarations are created. The shared validator supports `type` (including a list
of types), `properties`, `required`, `additionalProperties` (boolean or schema),
`items` (schema), `anyOf`, `enum`, `minimum`, `maximum`, `minLength`, `maxLength`,
`minItems`, `maxItems`, and `pattern` (Python regular-expression syntax).
Constraints alongside `anyOf` still apply. Documentation annotations `title`,
`description`, `default`, `examples`, `deprecated`, `readOnly`, `writeOnly`, and
`$comment` are accepted without changing validation or filling in defaults.
Unsupported keywords, including `$ref`, `const`, `oneOf`, `allOf`, and `format`,
are rejected rather than published as unenforced constraints. Use inline schemas
and single-value `enum` declarations where appropriate. Requests also have implicit
size/depth bounds; results use their declared limits and the serialization depth bound.

Log inspection is read-only even when no container exists: a launch can have
persisted metadata before creating its containers. Logs preserve workload metadata
and model-cache claims; explicit stop/reconciliation owns cleanup.

## Plugin feedback

Job event streams include versioned `type: "feedback"` events from the public
[plugin feedback API](FEEDBACK.md), plus operation-scoped legacy `type: "log"`
messages. Both expose a top-level `message`; structured feedback also identifies
activities, stages, hosts, counters, and metrics. These describe the operation
(e.g. image transfer or ColdSnap restore), not container log output.

## Gateway discovery and workload recovery

Both LiteLLM and the external SparkRoute engine use the same discovery session,
coverage/removal-grace logic, and owner fencing. Keep the existing standalone
process backend, or select daemon discovery explicitly:

```bash
sparkrun proxy start --discovery-backend daemon --cluster lab
sparkrun proxy status
sparkrun daemon call proxy.discovery.status
```

The daemon feature must already be enabled. Startup attaches to or starts the
matching local controller binding; dry runs do neither. A private session pins the
effective host/cluster selection, SSH context, interval and removal grace. A changed
default cluster is picked up on the next explicit start, not midway through a
sweep. `proxy.auto_discover: false` disables periodic discovery for either backend.

Daemon mode uses a dedicated `proxy.serve` job for the gateway lifetime and a
separate `proxy.watch` job for discovery. Configure at least two service slots
when both are selected. Foreground attachment waits after its startup reservation
has been released. LiteLLM replacements are performed by its own supervisor;
SparkRoute keeps its admin-API configuration reconciliation. Finishing/cancelling
discovery does not own the gateway process. Stopping the proxy stops its matching
discovery session even if the gateway has already exited.

Discovery status includes backend, generation/job, scope, last attempt,
last complete successful observation, last application and sanitized error.
`running` means the loop is active; with metadata-only scope, `last_success` may
remain null because there is no complete host inventory. Partial/failed coverage
never counts as absence on an unobserved host. Failed or fenced configuration
application never commits removal-grace changes. An uncertain/dead owner requires
explicit stop/restart; it is not replaced merely because the API process restarted.
Do not repeatedly submit raw `proxy.watch` jobs; lifecycle controls manage ownership.

For the new SparkRoute HTTP/SSE client, configure `proxy.yaml` explicitly:

```yaml
proxy:
  gateway: sparkroute
  discovery_backend: daemon
  discover_interval: 30
  discover_removal_grace_sweeps: 2

gateways:
  sparkroute:
    transport: daemon
```

`gateways.sparkroute.transport` defaults to `bridge`. Selecting daemon transport defaults
discovery to daemon unless a discovery backend was explicitly saved. The managed
plugin verifies that both its daemon operations and the gateway binary support
this transport. Use coordinated host/plugin/gateway builds; older release pins do
not implement it. There is no automatic fallback after an uncertain mutation.
The Go lifecycle, admin catalog and profile-refresh clients all use the selected
transport. Managed startup writes a private connection descriptor containing local
binding/state/credential-file references; tokens stay out of gateway arguments
and exports. Local TCP token rotation is followed without per-request Python.

A separately managed Go gateway can use `-sparkrun-daemon-connection /private/client.json`
(or `SPARKROUTE_SPARKRUN_DAEMON_CONNECTION`) with an explicit remote descriptor:

```json
{
  "protocol": 1,
  "binding": "BINDING_FROM_DAEMON_INFO",
  "profile": "sparkrun",
  "manager": "sparkroute",
  "url": "https://controller:8443",
  "token_file": "/private/daemon.token",
  "ca_file": "/private/controller-ca.pem",
  "action_directory": "/private/gateway-actions"
}
```

The descriptor/token/action directory must be private. Omit `ca_file` for normal
system trust. An HTTP URL is explicit plaintext; the remote server still requires
its `--allow-insecure` opt-in. Remote connections attach only; they do not start a
remote daemon or make that controller own a locally spawned gateway. Verify the
binding/profile via the existing authenticated daemon tooling before provisioning
the descriptor.

Public `sparkrun.api.workloads` provides binding resolution, live endpoint
observations, inventory, ensure-ready, guarded stop/provider controls, and explicit
receipt reconciliation. A launch records placement before effects and uses the
same context through readiness and hooks. Healthy external workloads can be
adopted; automatic stop/sleep/wake require the integration's exact saved owner,
workload ID and recipe revision. The owner comes from the credential’s stable
manager identity, is persisted at job admission, and is never accepted from HTTP arguments or client labels.
`/v1/info` reports the current `manager`. Locally issue a remote gateway credential
with `daemon token create gateway --scope admin --manager sparkroute`; keep its
secret in a private token file. Token scope and management ownership are separate.
The local integration provisions/rotates this credential automatically and uses
it over both UDS and TCP. Ordinary local admin calls default to manager `daemon`.
ColdSnap registers the public version-1 lifecycle provider for its existing exact capture/activation receipt checks.

Interrupted effects are kept in `<config-root>/workload-receipts.sqlite3`, separate
from daemon job retention. A failed/lost job does not authorize a duplicate launch
or replay of uncertain post-launch hooks. Inspect the workload first. Release any
verified-dead execution reservations using `daemon resources reconcile`, then use
`workloads.reconcile` with a credential for the workload’s manager. For SparkRoute,
use its private token file and the controller URL (or `--socket` in place of `--url`):

```bash
sparkrun daemon --url http://127.0.0.1:8376 --token-file /private/sparkroute.token \
  call workloads.reconcile --data \
  '{"recipe":"@registry/recipe","recipe_revision":"REVISION","workload_id":"WORKLOAD_ID","acknowledge_effects":true}' --wait
```

Reconciliation checks complete placement evidence and matching live identity; it
does not execute hooks again. Stopping a confirmed exact workload also resolves its
old activation receipts. Switching SparkRoute transports refuses active legacy
workers or active daemon jobs admitted under manager `sparkroute`. Legacy
placement/provider records remain readable and are conservatively imported on daemon cutover; an uncertain old
placement requires reconciliation. Switching back to the bridge checks jobs and revokes SparkRoute credentials under
the same lock used for daemon admission, closing the cutover/submission race.
