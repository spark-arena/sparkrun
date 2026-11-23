# Daemon operation and HTTP reference

Reviewed: 2026-10-13, for the unreleased Sparkrun 0.4.0 daemon/discovery migration.
This inventory was rendered from the actual operation declarations in the local
host and SparkRoute plugin worktrees, without running handlers or a live daemon.
Operational setup and examples are in [DAEMON.md](DAEMON.md).

Gateway operations also honor the default-on `gateway` feature. Disabling it
rejects new gateway actions while status, stop, service status and service removal
remain available for recovery. Plugin loading state and capability flags are
separate: disabling a plugin removes its contributions in new daemon processes;
restart an existing daemon to apply loading or gateway selection changes.

## Contract and terminology

Optional request fields are omitted unless their schema explicitly accepts null.
Core adapters use reviewed reusable field contracts; they do not infer a type
from a parameter name. Query and worker results are encoded, redacted and checked
against the published result schema before success is cached or stored. Invalid
results produce `ResultContractError` (`invalid_operation_result`), not success.
OpenAPI includes Job, Prompt, Error, Event and Info envelopes. Successful job
results discriminate by operation name and version. Request safety limits are
separate from result constraints: large inventories follow their declared result
schemas. Stored job results still have a 2 MiB encoded-size limit.

`proxy.serve` and `proxy.watch` are internal service executors. They use the same
registry and durable job queue, but have no public operation alias, catalog entry
or HTTP submission path. Applications use public proxy/discovery controls.


- HTTP protocol: **1** (`/v1`); every operation listed here currently has operation **version 1**.
  The old SparkRoute stdio bridge protocol **5** is a separate, explicitly selected transport.
- The canonical operation name selects a handler. Both `POST /v1/ops/{name}`
  and the literal alias listed below call the same dispatcher, authorization,
  schema validation, idempotency and worker admission path. They take the same
  JSON object. A query still uses POST; these aliases are RPC conveniences.
- `query` means bounded inline execution, including administrative controls; it
  does not imply read-only behavior. Only operations explicitly declaring
  `cacheable: true` opt into the short query cache. It returns the declared result with HTTP **200**. `job` returns HTTP **202**
  with `{"job_id":"…","events_url":"/v1/jobs/…/events"}`. The final result is
  `GET /v1/jobs/{id}` → `result`; acceptance is never execution success.
- `read < operate < admin` are hierarchical permissions. Secret-prompt permission
  is separate. Same-user Unix peers have local admin access when no bearer is supplied; an
  explicit bearer selects its own scope/manager on UDS too. TCP always requires a token. `/v1/health` alone is unauthenticated.
- A non-admin principal sees only its own jobs and must retain the job's operation
  permission. `X-Sparkrun-Client` is an advisory label, never an identity or owner.
- `Idempotency-Key` applies to **job submissions**, scoped to the authenticated
  principal. The same key/operation/version/arguments/manager returns the same job;
  changed arguments conflict. Retention bounds this guarantee. A disconnect does
  not cancel the job. A later intentional action requires a new key.
- `auto` infers execution reservations; unresolved mutation destinations use `*`.
  `controller` requests that conservative reservation explicitly. `none` skips
  the worker's resource reservation, but a handler can still take domain locks
  or public-API reservations. It does not mean authorization is skipped.
- `proxy.watch` and `proxy.serve` are host-owned persistent service jobs, admitted
  against `daemon.max_services` (default 2), separately from `daemon.max_jobs`
  (default 4). They use the same durable job/event/cancellation infrastructure.
- Top-level request objects reject unknown fields. `!` marks a required field
  below; all others can be omitted. `null` means the published schema accepts
  it, not that every underlying API normalizes it identically.
  Maps, nested objects, bounds and final result schemas are published in `/v1/ops`
  and OpenAPI. This compact table is not a replacement for those schemas.

## Workload management identity

Lifecycle belongs to core `workloads.*`, for SparkRoute and other automation clients.
The daemon authenticates a stable **manager** from the locally issued credential;
`/v1/info` reports it, job admission persists it, and the isolated worker reuses it.
The default is `daemon`; managed SparkRoute credentials use `sparkroute`. A caller
cannot select ownership through operation arguments or `X-Sparkrun-Client`.
Local provisioning (`daemon token create NAME --manager sparkroute`) is the grant.
Non-default manager tokens share stable principal `manager:<name>` when provisioned
through the CLI/integration helper, so replacement credentials can resume retained
jobs and retry keys. Revocation prevents new submissions; accepted jobs keep their
admitted identity. Integration cutover drains those jobs before revoking credentials.

`workloads.run` and `workloads.ensure_ready` use this manager for new workload
ownership. The shared workload row (inventory, lifecycle control and the
`workload` field of ensure-ready) reports deployed `workload_id`, revision, owner,
`owned` relative to this manager, provider, permitted actions, and recorded state.
`ensure_ready` separately returns a verified endpoint, original `observed_at` and
`adopted`. Adoption never rewrites an existing owner. A different manager may route
to a healthy endpoint and inspect its provider status, but cannot use the guarded
stop/sleep/wake/reconcile operations to manage it.

This is an automation guard within the existing single-account permission model,
not tenant isolation. `workloads.stop_matching`/`stop_all` deliberately retain broad
operator control. Managed gateway credentials retain admin scope for the admin
catalog (without secret-prompt permission); manager identity does not narrow scope.
Namespaced plugin `owner` in `/v1/ops` means declaration ownership, distinct from a
workload's management owner and a daemon job's authenticated principal.

SparkRoute registers only `endpoints.discover`, `endpoints.status`, and
`endpoints.metadata` projections. The first two add gateway model-card fields to
verified host observations; metadata is an advisory, revision-checked read of saved
recipe fields, never readiness evidence. Activation and lifecycle DTOs remain
core; the Go client adapts them to its internal gateway types. No duplicate
`plugins.sparkroute.workloads.*`, binding-resolution or capability routes remain.
ColdSnap continues implementing the common lifecycle provider without HTTP routes.

## Infrastructure routes (not operation aliases)

| Method | Path | Permission | Behavior / input |
| --- | --- | --- | --- |
| GET | `/v1/health` | public | Liveness and protocol; not worker readiness. |
| GET | `/v1/info` | read | Profile/config binding, runtime identity, restart-required state. |
| GET | `/v1/ops` | read | Authorized, feature-filtered operation metadata and request/result schemas. |
| POST | `/v1/ops/{name}` | selected operation | Generic dispatcher; JSON arguments, query 200 or job 202. |
| GET | `/v1/jobs` | read + job visibility | Optional `state`, `limit` (default 200; clamped 1–1000); bounded list, no cursor. |
| GET | `/v1/jobs/{id}` | job owner/admin + operation scope | Durable state, result/error, public prompt metadata; no raw arguments, worker nonce or runtime lease. |
| DELETE | `/v1/jobs/{id}` | job access + operate | Requests cancellation; does **not** delete history or promise remote rollback. |
| POST | `/v1/jobs/{id}/input` | job access + operation scope; secret if needed | Exactly `prompt_id`, `value`; answer a live worker prompt. |
| GET | `/v1/jobs/{id}/artifacts/diagnostics` | job access | Download a finished job's bounded NDJSON bundle; expires with job history. |
| GET | `/v1/jobs/{id}/events` | job access | Durable SSE; `Last-Event-ID`; expired cursor → 410, later expiry → `gap` event. |
| GET | `/v1/events` | read + job visibility | SSE snapshots of visible jobs every 2 seconds; no inventory stream or replay cursor. |
| GET | `/v1/streams/logs` | read | SSE; optional `workload_id`, `recipe`, comma-separated `hosts`, `cluster`, `host`, `container`, `source_id`, `scope=head/all`, nonnegative `tail`, `follow=true/false`. Includes saved native Kubernetes workloads. Records carry source identity/resource metadata. Disconnect cancels finite/follow readers. |
| GET | `/v1/streams/monitor` | read | SSE; optional comma-separated `hosts`, `cluster`, `interval` (1–60 seconds). Shared latest frames, not durable history. The first event is `{"type": "history", "series": [{host, points: [{recorded_at, sample}]}]}` (the shared session's in-memory window); `frame` events follow. |
| GET | `/v1/reservations` | admin | Execution claims and verified owner liveness; dead claims remain until reconciled. |
| POST | `/v1/reservations/{owner}/reconcile` | admin | `{"acknowledge_effects":true}`; releases only a verified dead owner's claims. |
| GET | `/v1/snapshots/{name}` | read | Advisory snapshot of `discovery.snapshot`; no other selector is accepted. |
| POST | `/v1/maintenance` | admin | `{"drain":true/false}`; admission control, not worker cancellation. |
| POST | `/v1/shutdown` | admin | `{}`; stops API process, workers continue. |
| GET | `/openapi.json` | authenticated | Full registered operation schemas, including feature-disabled or higher-scope declarations; not the filtered capability view. |
| GET | `/docs` | read | Swagger UI; same authorization boundary. |

No MCP route is mounted. No combined logs+monitor route exists. Tokens are issued,
listed and revoked through local CLI tooling, not an HTTP token-administration API.
Raw recipe-trust and credentials are not accepted as arbitrary request fields;
trust and secret prompts retain their separate server policies.

Log `host`/`container` filters select from the resolved full workload topology;
both filters intersect and override head-only scope. `hosts` still supplies the
full topology, not a source filter. Empty filters and invalid query parameters
are rejected before streaming. Read failures after streaming starts produce an
SSE `error` record. Each request resolves its target anew; there is no target-ID
registry or paging route. See [LOGGING_API.md](LOGGING_API.md) for matching,
cancellation, ordering, and Python paging contracts.

This snapshot contains **98 operations: 87 core, 8 K8s and 3 SparkRoute**.

## Domain vocabulary

Names use `domain.action`, with underscores inside multiword actions. These are
unreleased canonical operations; retired flat names have no compatibility aliases.
Python library names are independent of the daemon operation vocabulary.

| Surface | Meaning |
| --- | --- |
| `/v1/jobs` | Durable daemon invocation history, prompts, results and events. |
| `workloads.list` | Saved deployment metadata; does not probe current health. |
| `workloads.inventory` | Recorded provider lifecycle/ownership/action inventory. |
| `cluster.observe` / `cluster.report` | Raw host occupancy and coverage / grouped, metadata-enriched display report. |
| `workloads.plan` / `workloads.run` | Resolve placement / launch and optionally await readiness and post-launch hooks. |
| `workloads.ensure_ready` | Adopt, recover or launch with durable effect receipts and exact revision identity. |
| `workloads.stop` | Exact-ID/revision teardown of an integration-owned workload. |
| `workloads.stop_matching` / `workloads.stop_all` | Explicit selector-based teardown / host-wide teardown; distinct from ownership-guarded automation. |
| `workloads.materialize` | Resolve container launch units without launching. |
| `recipes.plan_build` / `recipes.build` | Plan / prepare native or image environments and stage assets without starting inference. |
| `recipes.validate` / `recipes.realize` | Validate a recipe / export a placed, pinned realization. |
| `catalog.search` / `catalog.recipes` | Simple recipe summaries / paginated catalog browsing and facets. |
| `catalog.registries.*` | Named admin actions: add, remove, enable, disable, trust, untrust; no action-dispatch field. |
| `benchmarks.run` / `benchmarks.resume` | Execute a benchmark / resume by benchmark ID, separate from the daemon job ID. |
| `discovery.snapshot` | Scoped endpoint observations with original timestamp, host coverage and errors. |

Registry `add` requires name and URL and never grants trust. `trust` requires
`acknowledge_trust: true`. Each registry action rejects irrelevant fields. Advisory
snapshots use `/v1/snapshots/discovery.snapshot`, with the same coverage-bearing
result as the corresponding operation. Plugin endpoint projections retain their
own namespaces and routing-policy fields.

<!-- BEGIN GENERATED DAEMON OPERATIONS -->

## Core operations

### benchmarks

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `benchmarks.resume` | `/v1/benchmarks/resume` | job / operate | auto | `benchmark_id!: string`, `dry_run: boolean`, `integrations: object`, `export_files: boolean`, `output_file: string`, `timeout: integer`, `exit_on_first_fail: boolean`, `api_key_env: string`, `collect_diagnostics: boolean` |
| `benchmarks.run` | `/v1/benchmarks/run` | job / operate | auto | `recipe!: string`, `category: string`, `framework: string`, `profile: string`, `bench_args: object`, `hosts: string[]`, `cluster: string`, `overrides: object`, `resume: string`, `skip_run: boolean`, `no_stop: boolean`, `exit_on_first_fail: boolean`, `timeout: integer`, `api_key_env: string`, `integrations: object`, `output_file: string`, `export_files: boolean`, `solo: boolean`, `dry_run: boolean`, `scheduler: string`, `rootful: boolean`, `trust: boolean`, `offline: boolean`, `sync_tuning: boolean`, `extra_docker_opts: string[]`, `collect_diagnostics: boolean` |

### catalog

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `catalog.capacity` | `/v1/catalog/capacity` | job / read | auto | `cluster!: string` |
| `catalog.cleanup` | `/v1/catalog/cleanup` | job / operate | auto | `{}` |
| `catalog.clusters` | `/v1/catalog/clusters` | query / read | auto | `{}` |
| `catalog.import` | `/v1/catalog/import` | job / operate | auto | `content!: string` |
| `catalog.recipe` | `/v1/catalog/recipe` | job / read | auto | `reference!: string`, `overrides: object`, `expected_revision: string` |
| `catalog.recipes` | `/v1/catalog/recipes` | query / read | auto | `query: string`, `registry: string`, `runtime: string`, `local_only: boolean`, `offset: integer`, `limit: integer`, `filters: object` |
| `catalog.refresh` | `/v1/catalog/refresh` | job / operate | auto | `{}` |
| `catalog.registries` | `/v1/catalog/registries` | query / read | auto | `{}` |
| `catalog.registries.add` | `/v1/catalog/registries/add` | job / admin | auto | `name!: string`, `url!: string`, `subpath: string` |
| `catalog.registries.disable` | `/v1/catalog/registries/disable` | job / admin | auto | `name!: string` |
| `catalog.registries.enable` | `/v1/catalog/registries/enable` | job / admin | auto | `name!: string` |
| `catalog.registries.remove` | `/v1/catalog/registries/remove` | job / admin | auto | `name!: string` |
| `catalog.registries.trust` | `/v1/catalog/registries/trust` | job / admin | auto | `name!: string`, `acknowledge_trust!: boolean` |
| `catalog.registries.untrust` | `/v1/catalog/registries/untrust` | job / admin | auto | `name!: string` |
| `catalog.retain` | `/v1/catalog/retain` | job / operate | auto | `reference!: string` |
| `catalog.search` | `/v1/catalog/search` | query / read | auto | `query: string`, `registry: string`, `runtime: string`, `include_hidden: boolean`, `include_local: boolean`, `unique_names: boolean` |

### cluster

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `cluster.create` | `/v1/cluster/create` | job / admin | auto | `name!: string`, `hosts!: string[]`, `description: string`, `user: string` |
| `cluster.default` | `/v1/cluster/default` | job / admin | auto | `name: string/null` |
| `cluster.delete` | `/v1/cluster/delete` | job / admin | auto | `name!: string` |
| `cluster.observe` | `/v1/cluster/observe` | job / read | auto | `hosts: string[]`, `cluster: string`, `executor: string` |
| `cluster.report` | `/v1/cluster/report` | job / read | auto | `hosts: string[]`, `cluster: string`, `executor: string` |
| `cluster.update` | `/v1/cluster/update` | job / admin | auto | `name!: string`, `hosts: string[]`, `description: string`, `user: string` |

### diagnostics

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `diagnostics.collect` | `/v1/diagnostics/collect` | job / read | auto | `hosts: string[]`, `cluster: string`, `workload_id: string`, `log_tail: integer`, `max_bytes: integer`, `dry_run: boolean` |

### discovery

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `discovery.snapshot` | `/v1/discovery/snapshot` | job / read | auto | `hosts: string[]`, `cluster: string`, `workload_ids: string[]`, `metadata_only: boolean` |

### model_cache

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `model_cache.adopt` | `/v1/model_cache/adopt` | job / admin | auto | `model: string`, `revision: string`, `endpoint: string`, `cache_dir!: string`, `hosts: string[]`, `all_models: boolean`, `dry_run: boolean`, `offline: boolean`, `last_used: string`, `pinned: boolean` |
| `model_cache.delete` | `/v1/model_cache/delete` | job / admin | auto | `model!: string`, `revision: string`, `cache_dir!: string`, `hosts: string[]`, `dry_run: boolean`, `force: boolean`, `allowed_revisions: string[]` |
| `model_cache.import` | `/v1/model_cache/import` | job / operate | auto | `model!: string`, `revision: string`, `endpoint: string`, `projector: string`, `file_selection: string`, `manifest: object`, `cache_dir!: string`, `hosts: string[]`, `checksums: boolean`, `fetch_manifest: boolean`, `offline: boolean` |
| `model_cache.inspect` | `/v1/model_cache/inspect` | job / read | auto | `model!: string`, `revision: string`, `endpoint: string`, `projector: string`, `file_selection: string`, `cache_dir!: string`, `hosts: string[]`, `checksums: boolean` |
| `model_cache.list` | `/v1/model_cache/list` | job / read | auto | `cache_dir!: string`, `hosts: string[]` |
| `model_cache.pin` | `/v1/model_cache/pin` | job / operate | auto | `model!: string`, `revision!: string`, `endpoint: string`, `cache_dir!: string`, `hosts: string[]`, `pinned: boolean` |
| `model_cache.prune` | `/v1/model_cache/prune` | job / admin | auto | `cache_dir!: string`, `hosts: string[]`, `dry_run: boolean`, `min_age_days: number`, `max_bytes: integer`, `min_free_bytes: integer` |
| `model_cache.reconcile` | `/v1/model_cache/reconcile` | job / admin | auto | `model!: string`, `revision!: string`, `endpoint: string`, `cache_dir!: string`, `hosts: string[]`, `claim!: string`, `confirm_inactive!: boolean` |
| `model_cache.repair` | `/v1/model_cache/repair` | job / operate | auto | `model!: string`, `revision: string`, `endpoint: string`, `projector: string`, `file_selection: string`, `cache_dir!: string`, `hosts: string[]`, `local_cache_dir: string`, `transfer_mode: string`, `checksums: boolean`, `offline: boolean`, `dry_run: boolean` |
| `model_cache.resolve` | `/v1/model_cache/resolve` | job / read | auto | `model!: string`, `revision: string`, `endpoint: string`, `projector: string`, `file_selection: string`, `cache_dir: string` |

`model_cache.import` accepts a supplied `manifest` without origin access. Omitting
it requires explicit `fetch_manifest: true`; the daemon resolves metadata once
using server-side HF credentials, then imports the same immutable inventory on
all targets without a prompt. `offline: true` always requires a supplied
manifest. No model weights are downloaded, and origin failures never fall back
to inferred local inventories. Specify `revision` to check an existing cached
commit; omission resolves the origin's current default revision. Call
`model_cache.resolve` first and submit its returned manifest to review the
inventory before importing. Incomplete imports return validation reports with
an incomplete status, even though the manifest has been saved for later repair.

### proxy

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `proxy.add_alias` | `/v1/proxy/add_alias` | job / admin | none | `alias!: string`, `target!: string` |
| `proxy.discovery.ensure` | `/v1/proxy/discovery/ensure` | query / admin | none | `generation!: string` |
| `proxy.discovery.status` | `/v1/proxy/discovery/status` | query / admin | none | `{}` |
| `proxy.discovery.stop` | `/v1/proxy/discovery/stop` | query / admin | none | `{}` |
| `proxy.install_service` | `/v1/proxy/install_service` | job / admin | none | `scope: string`, `cluster: string`, `now: boolean`, `dry_run: boolean` |
| `proxy.list_aliases` | `/v1/proxy/list_aliases` | query / read | none | `{}` |
| `proxy.list_gateways` | `/v1/proxy/list_gateways` | query / read | none | `{}` |
| `proxy.models` | `/v1/proxy/models` | job / read | none | `{}` |
| `proxy.register_loaded_model` | `/v1/proxy/register_loaded_model` | job / operate | none | `recipe!: string`, `overrides: object`, `cluster: string` |
| `proxy.remove_alias` | `/v1/proxy/remove_alias` | job / admin | none | `alias!: string` |
| `proxy.resolve_gateway` | `/v1/proxy/resolve_gateway` | query / read | none | `name: string` |
| `proxy.service_status` | `/v1/proxy/service_status` | job / admin | none | `scope: string`, `journal_lines: integer` |
| `proxy.start` | `/v1/proxy/start` | job / operate | none | `host: string`, `port: integer`, `gateway: string`, `host_filter: string[]`, `cluster: string`, `clear_cluster: boolean`, `auto_discover: boolean`, `discovery_backend: string`, `discover_interval: integer`, `discover_removal_grace_sweeps: integer`, `restart: boolean`, `persist: boolean`, `dry_run: boolean` |
| `proxy.status` | `/v1/proxy/status` | job / read | none | `{}` |
| `proxy.stop` | `/v1/proxy/stop` | job / operate | none | `dry_run: boolean` |
| `proxy.sync` | `/v1/proxy/sync` | job / operate | none | `aliases: object`, `host_filter: string[]` |
| `proxy.uninstall_service` | `/v1/proxy/uninstall_service` | job / admin | none | `scope: string`, `dry_run: boolean` |
| `proxy.unregister_loaded_model` | `/v1/proxy/unregister_loaded_model` | job / operate | none | `recipe!: string` |

### recipes

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `recipes.build` | `/v1/recipes/build` | job / operate | auto | `recipe!: string`, `hosts: string[]`, `cluster: string`, `overrides: object`, `dry_run: boolean`, `trust: boolean`, `rebuild: boolean`, `offline: boolean`, `executor: string`, `executor_config: object`, `cache_dir: string`, `local_cache_dir: string`, `transfer_mode: string`, `transfer_interface: string`, `topology: string`, `sync_tuning: boolean`, `preserve_model_perms: boolean`, `skip_model_fan_out: boolean`, `builder_context: object` |
| `recipes.plan_build` | `/v1/recipes/plan_build` | job / read | auto | `recipe!: string`, `hosts: string[]`, `cluster: string`, `overrides: object`, `dry_run: boolean`, `trust: boolean`, `rebuild: boolean`, `offline: boolean`, `executor: string`, `executor_config: object`, `cache_dir: string`, `local_cache_dir: string`, `transfer_mode: string`, `transfer_interface: string`, `topology: string`, `sync_tuning: boolean`, `preserve_model_perms: boolean`, `skip_model_fan_out: boolean`, `builder_context: object` |
| `recipes.realize` | `/v1/recipes/realize` | job / operate | auto | `recipe!: string`, `hosts: string[]`, `cluster: string`, `overrides: object`, `strategy_options: object`, `solo: boolean`, `dry_run: boolean`, `trust: boolean`, `ensure: boolean`, `scheduler: string`, `transfer_mode: string`, `cache_dir: string`, `offline: boolean`, `runtime_cache: boolean`, `auto_port: boolean`, `ray_port: integer`, `dashboard_port: integer`, `dashboard: boolean`, `init_port: integer`, `executor: string`, `executor_config: object`, `rootful: boolean`, `workload_id_override: string`, `transfer_interface: string`, `local_cache_dir: string`, `sync_tuning: boolean`, `extra_docker_opts: string[]`, `topology: string`, `preserve_model_perms: boolean`, `skip_model_fan_out: boolean`, `recipe_ref: string` |
| `recipes.validate` | `/v1/recipes/validate` | job / read | auto | `recipe!: string`, `overrides: object` |

### setup

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `setup.ensure_local_key` | `/v1/setup/ensure_local_key` | job / admin | auto | `preferred: string`, `generate: boolean` |
| `setup.mesh_ssh_keys` | `/v1/setup/mesh_ssh_keys` | job / admin | auto | `hosts!: string[]`, `user: string`, `key: string`, `timeout: integer`, `dry_run: boolean` |
| `setup.plan` | `/v1/setup/plan` | job / read | auto | `hosts: string[]`, `cluster: string` |
| `setup.probe_hosts` | `/v1/setup/probe_hosts` | job / read | auto | `hosts: string[]`, `cluster: string` |
| `setup.probe_ssh_access` | `/v1/setup/probe_ssh_access` | job / read | auto | `hosts!: string[]`, `user: string`, `key: string`, `timeout: integer`, `max_workers: integer` |
| `setup.rdma_test` | `/v1/setup/rdma_test` | job / operate | auto | `hosts: string[]`, `cluster: string`, `suite: string`, `parallel_pairs: integer`, `full: boolean`, `image: string`, `duration: integer`, `queue_pairs: integer`, `msg_size: string`, `nccl_binary: string`, `gid_index: integer`, `link_type: string`, `force_container: boolean`, `keep_containers: boolean`, `dry_run: boolean` |
| `setup.resolve_control_key` | `/v1/setup/resolve_control_key` | job / admin | auto | `host: string`, `user: string`, `generate: boolean` |
| `setup.run_steps` | `/v1/setup/run_steps` | job / admin | auto | `hosts: string[]`, `cluster: string`, `dry_run: boolean`, `only_steps: string[]` |
| `setup.undo` | `/v1/setup/undo` | job / admin | auto | `cluster!: string`, `dry_run: boolean`, `only_steps: string[]` |

### tailscale

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `tailscale.down` | `/v1/tailscale/down` | job / admin | auto | `hosts: string[]`, `cluster: string`, `remove: boolean`, `dry_run: boolean`, `ask_sudo: boolean` |
| `tailscale.expose` | `/v1/tailscale/expose` | job / admin | auto | `cluster: string`, `proxy: boolean`, `head_host: string`, `port: integer`, `set_proxy_host: boolean`, `ask_sudo: boolean` |
| `tailscale.join` | `/v1/tailscale/join` | job / admin | auto | `hosts: string[]`, `cluster: string`, `tag: string`, `ephemeral: boolean`, `enable_ssh: boolean`, `hostname: string`, `dry_run: boolean`, `ask_sudo: boolean` |
| `tailscale.status` | `/v1/tailscale/status` | job / read | auto | `hosts: string[]`, `cluster: string` |

### workloads

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `workloads.endpoints` | `/v1/workloads/endpoints` | job / read | none | `recipe_revision: string`, `cluster_candidates: string[]`, `workload_id: string` |
| `workloads.ensure_ready` | `/v1/workloads/ensure_ready` | job / operate | controller | `recipe!: string`, `recipe_revision: string`, `cluster_candidates: string[]`, `overrides: object`, `timeout: number` |
| `workloads.find_intent` | `/v1/workloads/find_intent` | job / read | auto | `intent_id!: string`, `hosts!: string[]`, `cluster: string` |
| `workloads.inventory` | `/v1/workloads/inventory` | job / read | none | `{}` |
| `workloads.list` | `/v1/workloads/list` | query / read | auto | `limit: integer` |
| `workloads.materialize` | `/v1/workloads/materialize` | job / operate | auto | `recipe!: string`, `hosts: string[]`, `cluster: string`, `overrides: object`, `strategy_options: object`, `solo: boolean`, `dry_run: boolean`, `trust: boolean`, `ensure: boolean`, `scheduler: string`, `transfer_mode: string`, `cache_dir: string`, `offline: boolean`, `runtime_cache: boolean`, `auto_port: boolean`, `ray_port: integer`, `dashboard_port: integer`, `dashboard: boolean`, `init_port: integer`, `executor: string`, `executor_config: object`, `rootful: boolean`, `workload_id_override: string`, `transfer_interface: string`, `local_cache_dir: string`, `sync_tuning: boolean`, `extra_docker_opts: string[]`, `topology: string`, `preserve_model_perms: boolean`, `skip_model_fan_out: boolean`, `recipe_ref: string` |
| `workloads.plan` | `/v1/workloads/plan` | job / read | auto | `recipe!: string`, `hosts: string[]`, `cluster: string`, `overrides: object`, `strategy_options: object`, `solo: boolean`, `dry_run: boolean`, `trust: boolean`, `ensure: boolean`, `scheduler: string`, `transfer_mode: string`, `cache_dir: string`, `offline: boolean`, `runtime_cache: boolean`, `auto_port: boolean`, `ray_port: integer`, `dashboard_port: integer`, `dashboard: boolean`, `init_port: integer`, `executor: string`, `executor_config: object`, `rootful: boolean`, `workload_id_override: string`, `transfer_interface: string`, `local_cache_dir: string`, `sync_tuning: boolean`, `extra_docker_opts: string[]`, `topology: string`, `preserve_model_perms: boolean`, `skip_model_fan_out: boolean`, `recipe_ref: string` |
| `workloads.plugins` | `/v1/workloads/plugins` | job / read | none | `{}` |
| `workloads.reconcile` | `/v1/workloads/reconcile` | job / admin | controller | `recipe!: string`, `recipe_revision!: string`, `cluster_candidates: string[]`, `overrides: object`, `workload_id!: string`, `acknowledge_effects!: boolean` |
| `workloads.run` | `/v1/workloads/run` | job / operate | auto | `recipe!: string`, `hosts: string[]`, `cluster: string`, `overrides: object`, `strategy_options: object`, `solo: boolean`, `dry_run: boolean`, `trust: boolean`, `ensure: boolean`, `scheduler: string`, `transfer_mode: string`, `cache_dir: string`, `offline: boolean`, `runtime_cache: boolean`, `auto_port: boolean`, `ray_port: integer`, `dashboard_port: integer`, `dashboard: boolean`, `init_port: integer`, `executor: string`, `executor_config: object`, `rootful: boolean`, `workload_id_override: string`, `transfer_interface: string`, `local_cache_dir: string`, `sync_tuning: boolean`, `extra_docker_opts: string[]`, `topology: string`, `preserve_model_perms: boolean`, `skip_model_fan_out: boolean`, `recipe_ref: string`, `collect_diagnostics: boolean`, `confirm: boolean`, `wait_ready: boolean`, `post_launch: boolean` |
| `workloads.sleep` | `/v1/workloads/sleep` | job / operate | controller | `recipe!: string`, `recipe_revision!: string`, `cluster_candidates: string[]`, `overrides: object`, `workload_id!: string` |
| `workloads.status` | `/v1/workloads/status` | job / read | none | `recipe!: string`, `recipe_revision!: string`, `cluster_candidates: string[]`, `overrides: object`, `workload_id!: string` |
| `workloads.stop` | `/v1/workloads/stop` | job / operate | controller | `recipe!: string`, `recipe_revision!: string`, `cluster_candidates: string[]`, `overrides: object`, `workload_id!: string` |
| `workloads.stop_all` | `/v1/workloads/stop_all` | job / operate | auto | `hosts: string[]`, `cluster: string`, `dry_run: boolean`, one of `hosts` / `cluster` required |
| `workloads.stop_matching` | `/v1/workloads/stop_matching` | job / operate | auto | `workload_id: string`, `recipe: string`, `hosts: string[]`, `cluster: string`, `overrides: object` |
| `workloads.wake` | `/v1/workloads/wake` | job / operate | controller | `recipe!: string`, `recipe_revision!: string`, `cluster_candidates: string[]`, `overrides: object`, `workload_id!: string` |

## Bundled Kubernetes plugin

Requires the loaded `k8s` integration and `integration.k8s`.

### operations

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `plugins.k8s.cluster_info` | `/v1/plugins/k8s/ops/cluster_info` | job / operate | auto | `kubeconfig: string`, `context: string`, `namespace: string`, `pin: boolean` |
| `plugins.k8s.configure_service_account` | `/v1/plugins/k8s/ops/configure_service_account` | job / admin | auto | `name: string`, `namespace: string`, `kubeconfig: string`, `context: string`, `create_namespace: boolean`, `token_duration: string`, `write_kubeconfig: boolean`, `dry_run: boolean` |
| `plugins.k8s.ensure_kubectl` | `/v1/plugins/k8s/ops/ensure_kubectl` | job / admin | auto | `version: string`, `context: string`, `download: boolean` |
| `plugins.k8s.jobset_status` | `/v1/plugins/k8s/ops/jobset_status` | job / operate | auto | `name!: string`, `namespace: string`, `kubeconfig: string`, `context: string` |
| `plugins.k8s.kueue_status` | `/v1/plugins/k8s/ops/kueue_status` | job / operate | auto | `kubeconfig: string`, `context: string` |
| `plugins.k8s.list_nodes` | `/v1/plugins/k8s/ops/list_nodes` | job / operate | auto | `kubeconfig: string`, `context: string`, `selector: string`, `gpu_only: boolean` |
| `plugins.k8s.setup_kueue` | `/v1/plugins/k8s/ops/setup_kueue` | job / admin | auto | `install: boolean`, `kueue_version: string`, `jobset_version: string`, `namespace: string`, `kubeconfig: string`, `context: string`, `dry_run: boolean` |
| `plugins.k8s.stop_jobset` | `/v1/plugins/k8s/ops/stop_jobset` | job / operate | auto | `name!: string`, `namespace: string`, `kubeconfig: string`, `context: string` |

## External SparkRoute plugin

Requires the external SparkRoute plugin and `gateway.sparkroute`; it is not bundled with Sparkrun.

### endpoints

| Operation | POST alias | Kind / permission | Reservation | Arguments (`!` required) |
| --- | --- | --- | --- | --- |
| `plugins.sparkroute.endpoints.discover` | `/v1/plugins/sparkroute/ops/endpoints/discover` | job / read | none | `binding: object` |
| `plugins.sparkroute.endpoints.metadata` | `/v1/plugins/sparkroute/ops/endpoints/metadata` | query / read | none | `workload_id!: string`, `recipe_revision!: string`, `served_models!: string[]` |
| `plugins.sparkroute.endpoints.status` | `/v1/plugins/sparkroute/ops/endpoints/status` | job / read | none | `binding!: object`, `workload_id!: string` |

<!-- END GENERATED DAEMON OPERATIONS -->

## Result families and interpretation

| Family | Final result / meaning |
| --- | --- |
| Catalog | Bounded recipe listings/facets, registry or cluster arrays, recipe details with stable `source_kind`, normalized overrides, revision, plugin items and issues; refresh/capacity/import mutations have their own schemas. |
| Planning/execution | Explicit Plan/Run/Build/Benchmark DTOs. Workload `workload_id` is distinct from the daemon `job_id`. A serialized plan is a preview, not a resumable execution handle. |
| Generic workloads | `ensure_ready`: ready state, exact workload ID, credential-free endpoint, original observation timestamp, adoption flag and shared workload row. Guarded stop: offline state/IDs; selector/all teardown retain their separate Stop/StopAll DTOs. Inventory: recorded state/providers/actions. Control: the same workload row after provider observation/action. Reconcile: ID/state/receipt count. |
| Discovery | `discovery.snapshot` and `workloads.endpoints`: sweep ID/time, live/metadata mode, hosts/coverage/errors/completeness, safe endpoints. A partial or metadata-only sweep is not authoritative absence. |
| Proxy | State/models/start/stop/sync/alias/service DTOs; status includes discovery owner/backend/job/scope/health. `proxy.discovery.ensure` returns an existing/new watch job ID, not that watch's result. |
| SparkRoute | Gateway-only endpoint/model-card projections; optional metadata enrichment never changes readiness or observation time. Lifecycle and plugin/provider inventory use core `workloads.*`; binding resolution uses `catalog.recipe`. |
| Setup/network/cache/Kubernetes | Operation-specific result schemas; some are maps/arrays or booleans, not a universal envelope. Check `/v1/ops` rather than guessing from operation name. |

Query results and successful job results do not have the same HTTP envelope.
Transport errors use `detail` and usually a typed `error` record; job failures are
stored in the job's `error`. Terminal job states are `succeeded`, `failed`,
`cancelled`, `lost`; nonterminal states are `queued`, `starting`, `running`,
`waiting_input`, `cancelling`. Lost/uncertain effects require reconciliation;
`retryable` does not mean that replaying a mutation with a fresh key is safe.

## Extension model and current limits

Plugins register **local** names inside their loader-assigned identity. The host
binds `endpoints.discover` from SparkRoute to
`plugins.sparkroute.endpoints.discover`, with alias
`/v1/plugins/sparkroute/ops/endpoints/discover`. Registering another owner's
name, conflicting paths, or reserved infrastructure roots fails registration.
An explicit plugin `DaemonRoute` can adapt a closed envelope into declared
operations; authorization is still checked on the selected target. No production
custom adapter routes are currently registered by the reviewed core/K8s/SparkRoute set.

ColdSnap contributes the version-1 workload lifecycle **provider** interface;
it does not currently contribute `/v1/plugins/coldsnap/...` daemon operations.
Its receipt-backed status/sleep/wake are reached through core `workloads.*`
operations. The bundled LiL, OCI Relay and Sparkarena integrations do
not register independent daemon operations in this snapshot. Their effects and
structured feedback flow through the enclosing public API operation/job.

The live `/v1/ops` response is authoritative for an installation: disabled or
missing plugins change the available set. Compatibility pins for the coordinated
migration must be updated to committed, tested host/gateway revisions before
release; this worktree inventory is not a declaration that old pins support it.

## Updating this reference

The operation tables come from declarations, without executing handlers. From the
host checkout, with the matching reference plugin checkout available:

```sh
python scripts/render-daemon-api.py --sparkroute-checkout /path/to/sparkrun-sparkroute-plugin
```

Use `--check` to detect drift without writing. The prose/infrastructure table is
reviewed separately when routes or semantics change. `/v1/ops` and OpenAPI expose
full request/result schemas for the running installation.

Model-cache operations use explicit bindings in
`src/sparkrun/daemon/model_cache_ops.py`. Each binding declares forwarded fields,
request-object adaptation, server-supplied parameters and intentional exclusions;
registration, execution and API coverage consume that declaration. Adding a Python
parameter requires classifying it in the binding and its contract tests. Signature
inspection does not automatically expose new network inputs. Operation-specific
schema constraints and the existing result allowlists remain explicit.

## Diagnostic bundles

`diagnostics.collect` produces a standalone, read-only diagnostic bundle. Its
result includes section status, `partial`/`truncated`, and an artifact descriptor;
clients download it at `/v1/jobs/{job_id}/artifacts/diagnostics` after completion.
The route enforces the job's ownership and operation scope and expires with job
retention. Requests cannot select filesystem paths or include credentials.

`workloads.run`, `benchmarks.run`, and `benchmarks.resume` accept optional
`collect_diagnostics: true`. Capture surrounds the full adapter, including run
readiness/hooks, without altering the normal operation result. An `artifact`
event advertises the download URL, which also works for failed/cancelled jobs
when a bundle was created. Capture defaults to off.

See [DIAGNOSTICS_API.md](DIAGNOSTICS_API.md) for Python/CLI use, the versioned NDJSON
format, redaction, bounds, partial sections, and current collection limits.
