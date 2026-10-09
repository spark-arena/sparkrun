# API migration to 0.4.0

`0.4.0` is a breaking Python API release. Update downstream callers and pin
compatible core/integration versions together. Removed imports and model fields
have no deprecation shims. Existing recipe and configuration YAML keys remain
shared unless noted below. The tables also cover pre-release plugin contracts.

## Supported imports

| Consumer | Supported surface |
| --- | --- |
| Application authors | `sparkrun.application`: `initialize`, `run_cli`, `ApplicationProfile`, `UpdateSource`, `APPLICATION_PROFILE_API_VERSION`, and application/controller identity helpers. |
| Workload and benchmark callers | `sparkrun.api`: operation functions, options, results, decisions/events, and operational errors. |
| Catalog/front-end callers | `sparkrun.api`: catalog functions and `Catalog*` dictionary types; see the [headless catalog guide](CATALOG_API.md#browse-preview-and-retain). |
| Setup callers | `sparkrun.api.setup`: checks/models, probe/plan helpers, apply/undo runners, manifest manager, statuses, and `SetupFailed`. |
| Benchmark integration authors | `sparkrun.core.benchmark_integrations`: registration, `BenchmarkIntegration`, `BenchmarkDefaults`, context and immutable measurement/state snapshots. |
| Setup extension authors | `sparkrun.core.setup_steps`: step/constraint registration; `sparkrun.core.setup_plans.SetupPlan`: hardware-owned, opt-in executor plans; caller-facing types are also exported by `api.setup`. |
| Other extension authors | Documented registries such as `core.cli_registry`, `core.run_handlers`, `core.hardware_probe_extensions`, and `core.features`; shared transactions and `PLUGIN_API_VERSION` in `core.registration`. |
| Gateway plugin authors | `sparkrun.proxy.contracts`: shared `ProxyModel`, `GatewayOperationError`, `GatewayQueryError`, and optional console/credential protocols; `proxy.supervisor.GatewaySupervisor` for lifecycle and `proxy.gateway` for registration. |
| Kubernetes callers | `sparkrun.plugins.k8s.api`, `plugins.k8s.config`, and `plugins.k8s.executor`. |

Documented core extension modules remain supported imports. Private API/CLI
modules are implementation details; installed plugins should not depend on the
bundled Arena adapter's private command helpers.

## Application initialization and identity

```python
from sparkrun.application import initialize, ApplicationProfile, UpdateSource

profile = ApplicationProfile(
    id="example-app", display_name="Example App", command="example-app",
    package="example-app", profile_ref="example_app.profile:PROFILE",
    update_sources={"stable": UpdateSource("example-app")},
)
context = initialize(profile, config_path="/path/to/example-app/config.yaml")
```

An application can be a CLI, daemon, desktop app, or another Python frontend.
Select the profile/config before reading settings or initializing plugins. One
process binds to one application/profile and canonical config file. Repeated
initialization reuses its plugin registry; switching the binding raises an error.
`api.default_sctx()` delegates to `initialize()` and creates a fresh context view
of that same binding. Pass an existing context to share its cached managers.

`initialize(variables=...)` supports injection on the first call only. Later calls
may omit it or pass the same instance. A fatal bootstrap error poisons the
initialization; repair the cause and restart the process. Optional plugin failures
are contained and reported in inventory. Required integration failures prevent
launch and raise `api.IntegrationUnavailable`.

One controller identity represents an application using a canonical config
**directory**. A private seed and the directory derive its opaque ID. Files in
that directory share it; copying/moving to another directory changes it; restoring
the seed at the original path preserves it. There is no configurable controller
selector. Child processes inherit the selected profile/config. Shared services
can use `(application.id, controller_id, resource_id)` for reconciliation without
switching the core process's profile. Existing executor ownership enforcement is
application-scoped; emitting a controller envelope does not expand that policy.
See [application profiles](APPLICATION_PROFILES.md).

Cluster SSH overrides are operation-local. `SparkrunContext.for_cluster()` and
`SparkrunConfig.for_cluster()` preserve the original configured fallback user;
inspect the resolved cluster/operation context rather than expecting a previous
operation to mutate `sctx.config.ssh_user`.

## Optional integration moves and selection

| Previous surface | 0.4.0 surface |
| --- | --- |
| `sparkrun.arena` | `sparkrun.plugins.sparkarena` and corresponding auth/upload modules. |
| `sparkrun.api.k8s` | `sparkrun.plugins.k8s.api`. |
| `sparkrun.orchestration.k8s` | `sparkrun.plugins.k8s.orchestration`. |
| `sparkrun.orchestration.executors.k8s` | `sparkrun.plugins.k8s.executor`. |
| Kubernetes accessors on `SparkrunConfig` | `K8sSettings(config)` from `plugins.k8s.config`. |
| Kubernetes fields on generic executor config | `K8sExecutorConfig`; normal executor resolution selects it. |
| Run-handler callback with individual resolved arguments | `(options, sctx, *, plan: RunPlan, started_at: float) -> RunResult`. Inspect public `rc`; the private launch handle is optional. |
| `core.installed_plugins.registration_transaction(v)` | `core.registration.registry_transaction(v)`. All loaders use one shared import/registration boundary. |

Arena's `integration.arena` gate registers its CLI and benchmark hooks; Sparkrun
enables it by default. Kubernetes requires the parent `integration.k8s` gate.
On stable/beta, enabling only `executor.k8s` is insufficient. On alpha, the parent
and its `executor.k8s`, `cli.setup.k8s`, and `api.run.k8s` children default on.
Explicit profile/user policy overrides defaults. See the
[Kubernetes guide](../src/sparkrun/plugins/k8s/README.md).

Installed integrations are selected by stable ID; installation alone does not
load them. All plugin sources use registration rollback, including import-time
contributions to enlisted registries. Failed plugins are not reported as loaded
and independent plugins may continue. Inventory retains import/registration failure
messages for all three sources, with directory/package provenance. Listing does
not import disabled modules; a successful reload clears a prior failure for that
source. Provider conflicts remain launch blockers. Directory plugins with duplicate
top-level names are rejected before import; pre-imported modules from another
origin cannot satisfy a configured source. Directory inventory reads versions
only from the module at that exact path and no longer borrows a same-named
installed distribution's version. Installed entry-point versions remain metadata-based.
Arbitrary plugin I/O is outside rollback. The SAF adapter depends on
`scitrera-app-framework==0.0.69`, including private state-root/registry internals;
changing that pin requires profile, rollback, and installed-wheel checks.
Plugin modules declare literal integer `SPARKRUN_PLUGIN_API_VERSION = 1`, checked
against `core.registration.PLUGIN_API_VERSION`, independently of the application
profile API version. Installed, directory, bundled, and direct module loaders all
require it. A missing or incompatible declaration rejects that plugin before
registration; inventory retains the reason and enlisted import-time contributions
are rolled back. Upgrade directory plugins before enabling them on 0.4. See [framework registration and identifier terminology](PLUGINS.md#installing-a-benchmark-framework).

Run-handler callbacks now receive `before_start` as a required keyword argument.
For real launches call this core-owned, idempotent callback after validation and
staging, immediately before submission; let failures abort the launch. It is `None`
in dry-run. Core no longer replaces a running deployment before dispatching to
the plugin. Controller-native handlers pass their resolved executor via
`before_start(executor=...)`; core uses its exact target for confirmed replacement,
including a prior deployment with the same portable ID. Kubernetes performs this
after manifest and feasibility prechecks.

The 0.4.0 `RunOptions` cleanup removes two ineffective fields:

- Replace `RunOptions(port=9001)` with `RunOptions(overrides={"port": 9001})`.
  This single input participates in planning, intent/fingerprint generation,
  ensure matching, command generation, and handler dispatch.
- `diagnostics_path` is removed from library options. The CLI's
  `--collect-diagnostics PATH` still owns its collector; embedding applications
  manage diagnostics collection separately.

`RunOptions.trust` is a boolean, defaulting to `False`. It never requests an
interactive prompt. False retains automatic trust for local recipes and trusted
registries; `True` explicitly authorizes hooks and other trust-gated execution.
Frontends must complete any prompt before calling the API. API execution now
rejects untrusted hooks before preparation/launch even with a TTY attached, instead
of reaching legacy interactive prompts. The CLI uses `--trust` for this explicit
authorization.

Preloaded registry recipes must carry the source URL recorded by their loader
for automatic trust. A missing URL or a mismatch with the currently configured
registry requires reloading or explicit `trust=True`. Staged catalog imports keep
their external-source flag when loaded by path through any frontend.

`RunOptions.follow` now defaults to `False`. It controls only legacy foreground
runtime attachment (`detached=False`); detached API launches return without
following. Read detached logs through `api.logs()` or a plugin's explicit iterator.
The CLI selects its own log policy. Native handlers can return only `RunResult`;
the CLI renders that public result without requiring a private launch handle or
running the legacy host lifecycle.

Kubernetes `launch_jobset()` and `run_launcher_job()` no longer accept `follow`.
Both return after submission. Use `plugins.k8s.api.logs(name=..., kind="jobset")`
or `kind="job"`, optionally with `follow=True`, to consume structured `LogLine`
records. Close the iterator when stopping early. `setup k8s launch --follow` and
`setup k8s run-job --follow` render that iterator in the CLI. Native `run` JobSet
submission returns without attachment; use the explicit plugin log API.
Native `RunResult.cluster_id` remains the portable workload ID. Use
`api.stop(cluster_id=result.cluster_id)` for shared teardown; the saved native
resource reference and resolved Kubernetes target survive a new API session and
changes to defaults. `api.status(..., executor="k8s")` reports submitted JobSets
for intent discovery and replacement. Use `result.metadata["k8s_jobset"]` for the
plugin's direct resource/log API. Common `api.logs()` explicitly rejects native
resource records until that surface supports controller-owned log sources.
The manifest retains portable identity, fingerprint, and host scope as annotations.
Explicit JobSet names and generated suffixes are validated before replacement.
Kubernetes lifecycle operations consistently translate operational failures into
`SparkrunError`, while malformed resource arguments remain `ValueError`.

Core completes common fresh-launch `RunResult` identity, fingerprint, timing and
preview fields for native handlers too. Handlers still return actual substrate
outcomes and need no private launch handle. Ensure hits keep unknown fields empty
instead of borrowing the new plan's fingerprint or timeline. Native ports now
resolve recipe defaults as well as explicit overrides and reject invalid values
before preparation; custom command templates must honor that configured port.

Run handlers reuse the supplied plan and standard executor resolver.
`RunOptions.executor_overrides()` supplies the caller layer. Do not independently
merge recipe/cluster/default layers or repeat placement in a handler.

The unused per-kind package entry-point groups were removed from host metadata;
installed plugins use `sparkrun.plugins` module entry points. Setup-step graphs
must be valid when module registration completes. Same-module forward references
work; cross-module providers must already be registered. Failed validation rolls
back the module, including when the affected step's feature is disabled.

## Benchmark callers

| Previous surface | 0.4.0 surface |
| --- | --- |
| `BenchmarkOptions(arena=True)` | `BenchmarkOptions(integrations={"arena": {}})`. Omit or use `{}` to disable explicit selection. |
| `resume_benchmark(id)` returning a raw dict | `BenchmarkResult`; measurements in `.results`, publication outcomes in `.integration_results`. |
| `resume_benchmark(id, emitter=...)` | `progress_callback=...` and `decision_callback=...`. Emitters are private frontend adapters. |
| `BenchmarkOptions(on_prompt_required=..., on_complete_state=...)` | One `decision_callback(BenchmarkDecision) -> bool`. The old fields are removed. |
| `BenchmarkResult.submission_id` | `result.integration_results.get("arena", {}).get("submission_id")`. |
| `BenchmarkFailed(exit_code=0)` for completed resume | Successful saved `BenchmarkResult` with `already_complete=True`. |

Benchmark authentication is now execution-only. Use `api_key_env` rather than
persisting API keys in task args or publication data. Resume saves/reuses the
variable name and resolves its current value; it accepts `api_key_env`, `timeout`,
and `exit_on_first_fail` keyword overrides. Completed publication retries need no
inference credential, and implicit completed-result reuse performs no inference
launch or cleanup. Legacy structured keys are removed from loaded snapshots;
plugin-owned serialized documents are handled by their owner at binding.

Incomplete resumes restore a credential-free saved recipe specification and
verify it against measurement identity and running-job metadata. Legacy state
without a specification must reproduce its original identity from current inputs;
otherwise start fresh. Old incomplete IDs whose hashes included credentials
cannot be verified after removing those credentials. Completed results remain
loadable by ID.

Checkpoints now distinguish absent state from unreadable/invalid state. Implicit
resume raises a typed operational error for the latter; only explicit fresh
execution replaces it. See [benchmark authentication and recovery](BENCHMARK_API.md#authentication-and-checkpoint-recovery).

Preloaded `Recipe` and `ClusterDefinition` objects retain their in-memory edits;
recipe resolution may mutate the supplied recipe. Use a separate instance when
you need to preserve a template. Benchmark run/resume are silent unless a
callback is supplied. `BenchmarkDecision` distinguishes `resume_incomplete`,
`remeasure_complete`, and `integration_confirmation`; callbacks see a frozen
value, not mutable persistence state. Missing callbacks use each decision's
default. Arena quality confirmation defaults to declining in headless use. Recipe
trust remains a separate option.

`resume_benchmark(id, export_files=False)` suppresses optional file exports after
resumed measurement; `output_file=...` selects their base path. Complete result
loading/publication retry never remeasures or regenerates exports. A completed ID
with no available integration work returns its saved result without changing its
state. The CLI renders its own “nothing to resume” message.

Validated scheduled measurements commit before optional exports. Failures after
validation raise `BenchmarkFinalizationFailed` with `.result`, `.stage`, `.errors`,
and original cause. `BenchmarkIntegrationFailed` remains its subtype for plugin
and shared-state errors; shared persistence uses `integration="<state>"`.
Successful partial output paths remain attached. Publication retry requires saved
measurements; storage failures may require retaining `.result` directly.

Owned inference cleanup runs on all exits after launch, including early plugin,
framework and frontend callback errors. `no_stop`, `skip_run`, and resume-by-ID
retain their ownership policies. Secondary cleanup errors preserve the original
failure. Publication retry does not repair failed optional exports.

## Benchmark integration authors

| Previous contract | 0.4.0 contract |
| --- | --- |
| `BenchmarkIntegration(on_ready=...)` | `on_bind`, before launch; endpoint readiness is a different event. |
| Settings validation in `prepare()` | `validate(context)` sees merged saved/explicit settings on every invocation. |
| `prepare()` returning `BenchmarkOptions` | Return `BenchmarkDefaults`: category, framework, profile, read-only bench args only. |
| Live `context.result: BenchmarkExecution` | Read-only `BenchmarkMeasurement`, including recipe YAML and redacted provenance. |
| Mutable `context.state: BenchmarkRunState` | Detached `BenchmarkStateInfo`; persist private JSON values in `context.data`. |
| Writing other integrations' result mappings | Write your own `context.outcome`; core produces detached public results. |

Normalization converts dates/datetimes to ISO strings, pathlib paths to strings,
and tuples to lists, and rejects other arbitrary framework result objects.
Measurement snapshots are recursively read-only. `measured_at` and `completed_at`
use a persisted interval; publication retries do not advance it. Legacy states
pin a one-time timestamp fallback. Publication-only hooks may lack recipe YAML
and provenance and should use data saved by their earlier hooks.

See [benchmark API behavior and reserved state keys](BENCHMARK_API.md) and
[plugin contracts](PLUGINS.md).

## Setup recording and undo

`api.setup` exports caller-facing types, statuses, probe/plan helpers,
`run_setup_steps`, and `run_setup_undo`. Existing documented core imports remain
valid. Built-in reversals live in core; new frontends should use the public runner
rather than private CLI teardown helpers.

Use `ok`, `warn`, `fail`, and `skip` consistently. Per-host outcomes retain details;
step summaries aggregate severity. Recording needs one consistent cluster name
and a `ManifestManager`; omitting the manager explicitly disables recording.
Strict manifest validation rejects invalid shapes/unsupported versions before
changes, while accepting supported version-1 legacy defaults. Detached undo
inputs receive the same ownership/schema checks even without a manager.
`SetupUndoResult.manifest` now returns the detached updated state for caller-owned
persistence; with a manager it matches saved state. The input remains unchanged.

Only confirmed `ok` undo clears a host's record. Failed, skipped, declined,
filtered, and unknown plugin changes remain recorded. A prerequisite cannot be
removed while a recorded dependent remains on the same host, including transitive
dependents outside the selected filter. Successful independent hosts continue.
The CLI deletes cluster state only after complete reversible teardown, subject to
`--keep-cluster`; the runner itself never deletes cluster definitions.

Hard prerequisites use `FAIL` checks; `WARN` is advisory. Reprobes, rather than an
action's success flag alone, determine whether dependent apply actions are ready.
See [the complete headless example](SETUP_STEPS.md#complete-headless-example).

## Errors and release ownership

Operational API failures use `SparkrunError` subtypes. Implicit API initialization
wraps bootstrap failures with their original cause; explicit `initialize()` exposes
bootstrap errors directly. Malformed recipe files and known recipe-resolution
failures in `plan()` / `run()` raise `SparkrunError` with the original cause;
missing recipe selections use `RecipeNotFound`. Pure model/plan validation,
including unsupported `materialize()` layouts, may raise `ValueError` or
`TypeError`. Setup runner orchestration errors use `SetupFailed`; per-host
action failures are returned
outcomes. Benchmark result-bearing failures are described above. Interrupts and
`SystemExit` propagate unchanged.

Core and bundled K8s/Arena declarations share the owning `sparkrun` distribution
version. Independently released plugins, including bundled OCI Relay, retain
their own versions. Downstream applications should declare a compatible core
range or exact pin rather than infer compatibility from their own version.

Installed plugin inventory rows are frozen reporting snapshots; entry-point
handles are private. Replace any mutation of `InstalledIntegration` rows with
explicit configuration before initialization. Profile integration lists,
bootstrap URL lists, and environment-alias lists require list/tuple values;
use `("arena",)` instead of `"arena"` for a single integration.

Benchmark data inputs (`overrides`, `bench_args`, `integrations`, `state_extras`)
now consistently accept string-keyed mappings and copy serializable nested data
before hooks. Invalid shapes raise field-specific errors instead of silently
falling back to empty overrides. Resume integration settings use the same rules.
Incomplete resumes now enforce the same framework prerequisites as initial
measurements; preview and completed-result/publication paths remain independent
of those tools. Custom setup undo mappings now derive reverse dependency order
from the shared graph, independent of insertion order.

## Measurement failure and setup targeting

A nonzero task exit cannot contribute measurements, even when partial output
parses. `exit_on_first_fail=False` attempts remaining tasks once per invocation;
failed tasks stay resumable and retry on the next run. An incomplete schedule
raises `BenchmarkFailed` without successful-completion or publication hooks.
Timeouts follow the same rule. The bounded measurement-gap pass remains available.
Scheduled retries clear their previous task artifact before command construction.
Only usable JSON from successful attempts is aggregated, in schedule order;
unrelated or failed files cannot satisfy measurement coverage. A zero exit without
usable output and coverage still missing after the bounded gap pass remain
incomplete and cannot publish. Successful tasks retain their artifacts on resume.
The internal aggregator now accepts selected paths instead of scanning a directory;
the unused `gap_analysis(expected_per_task=...)` argument was removed.

`run_setup_steps()` requires nonempty host keys and matching `HostState.host`
values. Invalid initial mappings fail with `SetupFailed` before callbacks or
manifest access. Reprobes must satisfy the same invariant and return only requested
hosts; invalid results stop further actions while retaining changes already recorded.

## Sparse tasks and interrupted result processing

Base framework coverage now defaults to `None` (artifact success), rather than
reconstructing task indices from consolidated row positions. Frameworks can still
provide semantic coverage keys. Internal `gap_analysis()` requires
`completed_indices`, and `run_schedule(skip_run=...)` is removed. The public
`BenchmarkOptions.skip_run` option remains unchanged; warmup follows measurement
sessions, including the first task of a resumed session.

Resume distinguishes successful commands from committed measurements. Interrupted
consolidation/parsing/commit resumes from the saved task artifacts and original
measurement specification, without live inference, credentials, or command tools.
Only genuinely missing/invalid artifacts become pending. Committed results still
use the existing publication-only retry path.


### Docker policy and vLLM thread defaults

Docker now uses the bundled `io-uring` seccomp profile. Locally supplied
`executor_config.security_opt: ["seccomp=/path/policy.json"]` files are
snapshotted and carried to every Docker execution node in its launch script.
Explicit `seccomp=builtin` and `seccomp=unconfined` remain available. See
[executor configuration](EXECUTORS.md#docker-seccomp-profiles-04) for precedence,
path resolution, and preview behavior. Atlas no longer forces unconfined mode.

`Executor.prepare_launch(extra_opts=...)` is a default no-op hook for local
input validation before replacement. `apply_runtime_adjustments()` now receives
`defaults`, a read-only lower-priority `Variables` chain; plugins accepting the
documented `**kwargs` need no change. Docker uses it to preserve configured
security policies when adding its rootless defaults.

`vllm-distributed` no longer defines `OMP_NUM_THREADS=4`. The image/runtime can
choose its own thread count. Explicit `recipe.env.OMP_NUM_THREADS` values still
pass through unchanged, including clustered runs.

## Destination and recovery contracts

`RunPlan.executor_target` now records an `ExecutorTarget` (exported from
`sparkrun.orchestration.executor`). Native plugins derive destination identity
and snapshot connection settings through `Executor.resolve_target()`. Planning,
ensure, and replacement use the same target as launch. Native Kubernetes IDs
now distinguish namespaces, contexts, and canonical kubeconfig paths; use the
ID returned by `run()` for durable lifecycle calls. Existing saved native IDs
remain stoppable using their recorded targets. Controller identity remains one
per configuration directory, with no new configurable IDs.

Native Kubernetes does not expose a serving endpoint, so new benchmarks,
skip-run, and pending-task resumes reject it before using control-plane hosts as
an inference URL. Finished-artifact recovery and publication-only resume-by-ID
remain supported. Control-plane-only providers should set
`Executor.supports_host_endpoint = False` until they can supply a usable endpoint.

Initial job-metadata persistence is now required before submission. Preparation
and replacement-callback failures preserve the active record; writes are atomic.
A failed initial write aborts launch. Submission failure after replacement can
still leave the old workload stopped; the recorded new target supports recovery.

Benchmark recovery preserves the selected category and original measurement
interval through partial execution, consolidation, decoding, and final commit
failures. Legacy records retain their documented timestamp fallback. The private
`BenchmarkExecution` record uses `outputs` exclusively; the unused
`output_csv`/`output_json`/`output_yaml` aliases are removed. Public
`BenchmarkResult` is unchanged.

### Scoped observations and effective measurement context

Low-level `load_running_snapshot()` now returns `RunningSnapshot | None`, with
`cluster_ids` and per-executor `coverage`, instead of an `(ids, hosts)` tuple.
`save_running_snapshot(observation, ...)` accepts that same value; use
`ClusterStatus.observation` from a real status query. Do not reconstruct absence
coverage from hostnames alone. Existing unscoped cache files are ignored.
`prune_job_metadata(observation=...)` restricts pruning to confirmed absences;
omitting it is deliberate age-based cleanup and is unsuitable for automatic use.

Custom local PID directories now contribute to destination identity and therefore
to deterministic workload IDs. User-scoped targets also include the resolved SSH
user, so local default-path IDs change in 0.4.0. Docker's default IDs are unchanged.
Implicit principals are resolved before a new user-scoped launch: the OS user
for local dispatch, or OpenSSH's effective configuration (`ssh -G`) for remote
hosts. A cluster must resolve to one user; unresolved or mixed defaults require
an explicit `cluster.user`/`ssh.user` before submission. Existing jobs remain
addressable using their saved IDs, executor configuration, and recorded user.

Benchmark results add `measured_at` and `completed_at`. Runtime-default and
builder-selected images now survive interrupted result processing. Exports use
the same effective image/recipe projection as integrations; raw declarations are
retained separately. Exported `recipe.hash` remains a hash of exported text,
with `declared_hash` added for declared inputs. See the
[provenance contract](BENCHMARK_API.md#effective-measurement-provenance).


### Observed names during eviction

`api.stop(..., discovered=snapshot)` accepts the `ClusterStatus` returned by
`api.status()`. Eviction passes its existing snapshot so a surviving `node_1`
is stopped by its observed name, even when only one host remains. Container
names stay scoped to their observed hosts and executor; recorded destination
coverage must match the teardown target. Missing names or mismatched coverage
fail without deleting job metadata. Native controller teardown retains its
executor-owned lifecycle.

### Operation transport and preview results

`api.status()`, `status_report()`, and `stop_all()` now honor application SSH
configuration and the selected cluster's user when `ssh_kwargs` is omitted.
Explicit kwargs override individual keys; `None` or an empty value explicitly
clears that key. Resolving an operation does not mutate the shared context or
carry a cluster user into the next operation. Launch planning, intent discovery,
stop, and logs use the same operation-scoping helper.

`stop_all(discovered=...)` keeps the observed connection when defaults change.
Callers may override authentication settings (for example a rotated SSH key),
but an explicit different SSH user requires a fresh discovery. A manually built
snapshot without coverage still needs its matching caller-supplied context.
The observation cache format is now version 2; old caches are discarded rather
than treated as evidence about another remote user.

`StopAllResult.dry_run` explicitly identifies preview counts/host lists. Those
fields describe planned work when true and confirmed outcomes otherwise.
Discovery errors remain errors in previews, including when some workloads were
found; `success` is false whenever discovery or requested teardown failed.

`ClusterStatus.observation_errors` is the shared incomplete-discovery view for
classification, occupancy scheduling, strict recipe-based lifecycle discovery,
and strict replacement checks. `find_running_intent()` and ensure remain best
effort; `None` can also mean discovery failed. For verified absence, acquire
status and reject `observation_errors` before interpreting a miss. Custom
occupancy consumers should inspect it before using partial host observations.
See [executor coverage](EXECUTORS.md#status-observation-coverage) and the
[measurement provenance contract](BENCHMARK_API.md#effective-measurement-provenance).


Saved local jobs retain their recorded SSH principal even after their named
cluster changes user. Explicit incompatible user overrides and legacy records
with no principal fail without deleting metadata; use fresh status discovery to
inspect the intended namespace. Authentication-key rotation remains supported.
Metadata now persists `executor_user_scoped`; old local records derive that policy
from the executor. Placement, observation matching, and lifecycle recovery share
one private destination definition.

Pending benchmark resume validates effective serving configuration on automatic,
direct, skip-run, and artifact-gap paths. The initial job fingerprint is immutable;
a new launch cannot replace it to accept changed preparation output. Verified image
pins can change reference spelling while the remaining serving inputs stay equal.
Publication continues to use recorded measurement context, including runtime info.


Plans now retain the resolved user-scoped principal through execution even when
defaults or the nested cluster definition change after preview. Credential rotation
remains independent of identity. New local jobs resolve implicit users and record
them for stop/log/liveness recovery; unverifiable old records remain conservative.

Raw telemetry and live monitors now share status's per-key transport override
contract, including omitted context and explicit clears. A monitor with a failed
peer executor preserves known workloads but reports incomplete observation and
zero confirmed free slots.

Benchmark baseline recording consumes the same normalized deployment evidence as
resume validation. API-only run results retain their image-equivalence evidence
without requiring a private launch handle or a redundant image field in metadata.


### Effective transport, local paths, and deployment evidence

Explicit SSH users now take precedence over user settings in `ssh.options`, as
well as SSH config files. This applies to saved-job lifecycle operations and new
launches, and to direct SSH, embedded transfer scripts, pipelines, and rsync.
`build_ssh_opts_string(ssh_user=...)` now includes that user instead of ignoring it.
Implicit users still resolve from OpenSSH configuration or the local OS account;
rotating authentication keys does not change the selected namespace.

Local execution now uses one remote-path normalization/rendering policy for target
identity, launch, status, logs, and teardown. Leading remote-home prefixes are
equivalent; spaces and shell metacharacters are literal. Managed `pid_file` overrides
are rejected before submission; configure a per-workload `pid_dir` instead. Existing
fixed-file jobs retain only command-level recovery support, and failed discovery
cannot authorize automatic metadata deletion. See the [local path and recovery
contract](EXECUTORS.md#local-only-docker--k8s-ignore) before migrating an old job.
`log_file` remains an intentional shared append log; `log_dir` keeps logs separate.

Benchmark candidate assembly checks all available actual-image evidence for
agreement, including metadata when a private launch handle is present. Missing
sources retain their fallback behavior. Conflicting nonempty references fail before
new measurement commands; existing artifacts and historical provenance remain
unchanged. Equivalent digest aliases are accepted. This adds no public result or
context type and preserves complete-artifact recovery and publication-only retries.


### Native filesystem state and recovery

The [native path, state, and lifecycle contract](EXECUTORS.md#native-paths-state-and-lifecycle)
is the reference for local execution and legacy fixed-file/relative-path recovery.

Managed `pid_dir`, `log_dir`, and `log_file` now require absolute or home-relative
paths, including direct launch/preflight script generation. This prevents another
invocation directory from becoming the same saved destination. Legacy relative-path
metadata and cached coverage cannot authorize absence/pruning; managed stop rejects
the unresolved destination and retains metadata. Recover with the low-level helpers
from the original execution directory before migrating the configuration.

PID/log paths retain symlink-parent traversal. Low-level recovery preserves literal
relative home-like names. Locations bind before workload setup, so changing
`working_dir` or workload `HOME` no longer relocates state. Workload `working_dir`
and `env_file` may still be relative.
Missing/unreadable working directories or environment files, and nonzero activation
results, now abort launch/exec before the payload and leave existing claims intact.

State acquisition errors cannot establish free capacity or authorize metadata
pruning. Empty, invalid, or unreadable owner markers do not receive the confirmed
missing legacy-owner fallback. PID 1 is rejected as invalid workload state.

Native status and teardown include surviving process-group members after their
leader exits. Teardown retains records on failed inspection or surviving workers;
legacy standalone PID recovery remains supported. Low-level `status_cmd()` returns
0 for a live PID/group, 1 for absent/dead, and 2 for acquisition failure. Do not
interpret every nonzero result as absence. Public status/stop types and the managed
`pid_file` rejection remain unchanged.

Native launch now commits complete owner/PID records atomically. Failed PID
persistence rolls back the newly spawned workload using the shared verified group
shutdown; it never reports launch success with a partial PID. Unconfirmed rollback
identifies the PID/group for manual recovery. Successful detached submission and
serving-endpoint readiness remain distinct.

Local `run_cmd(detach=False)`, `generate_launch_script(detach=False)`, and
`generate_exec_serve_script(detached=False)` now reject unsupported foreground
workload execution before generating a script. Use detached mode for serving;
`exec_cmd` remains the separate foreground-hook operation.


## Benchmark artifact paths

New benchmark export references are absolute in API results, plugin snapshots,
and saved state. Recovery omits legacy relative output references because their
original working directory was not recorded; it still returns saved measurements.
See [benchmark artifact recovery](BENCHMARK_API.md#results-and-publication-retries).

## Gateway model queries and optional management

`api.proxy.models()` retains its tuple result on success and when the gateway is
stopped. Failed enumeration now raises `api.proxy.ProxyQueryFailed` instead of
returning an empty tuple. `proxy models` returns nonzero in both text and JSON
modes on failure. `api.proxy.status()` remains diagnostic and preserves
`model_query_error`; `ProxyStatus.require_models()` applies the same failure
policy to an existing snapshot.

Gateway plugins implement `query_models() -> tuple[ProxyModel, ...]` and raise
`GatewayQueryError` for unavailable observations. Import those from
`sparkrun.proxy.contracts`; `api.proxy.ProxyModel` remains the same public class.
LiteLLM and SparkRoute implement this contract directly; legacy dictionary
providers must migrate their decoding in the provider. SparkRoute changes belong
in its upstream checkout and are imported as a committed vendor snapshot.

Console and credential support are optional structural protocols in that module.
See [the gateway contract](PROXY.md#gateway-plugin-contract). `ui(issue_token=True)`
can create credentials and enable authentication; `admin_token()` without flags
only reads. Passing both `rotate=True` and `clear=True` raises `ValueError`.

Gateway selection helpers now initialize the default context when `sctx` is
omitted and honor the saved gateway pin, just like start. Plugins should import
`GatewaySupervisor` from `sparkrun.proxy.supervisor` and operational errors from
`sparkrun.proxy.contracts`. The older `_supervisor` imports preserve class identity
for existing providers. Optional credential and console operations must raise
`GatewayOperationError` for expected failures; arbitrary `RuntimeError` is no
longer treated as a routine management refusal.

Management operations also initialize an omitted context before plugin lookup,
then use the gateway recorded in process state rather than the saved pin. If
plugin bootstrap fails, generic process status/stop remain available through both
the API and CLI; model queries report unavailability. Invalid application profiles
still raise. Normal CLI context creation and extension discovery now use the
shared application initializer, binding the config path before plugin loading.

`prepare_config(write=False)` now previews ordinary starts and restarts as well
as dry runs. It must not write files, persist discovery snapshots, or update a
live gateway. The API calls `prepare_config(write=True)` after any previous
process has exited. `ProxyStartResult.config_path` is `None` for fileless
gateways and dry runs; callers must not expect the old string `"None"`.
Declared `GatewayOperationError` failures during construction, preparation,
shutdown, or process start are exposed as `ProxyStartFailed` with their original
cause. Gateway availability errors retain their separate API type. Providers
should not rely on callers catching their private configuration exception classes.

Gateway updates (including alias changes) now translate declared operational
failures consistently to `ProxyUpdateFailed`. SparkRoute now declares its
transport/authentication failures through that contract in the upstream plugin.
Preserve the cause when reporting these errors; callers should catch public API
types rather than import provider exceptions.

Recovery-only supervisors refuse model/alias updates with `GatewayUnavailable`
before discovery. Alias changes still persist before attempting to update a
running gateway, so this error can mean the local edit succeeded. CLI alias
commands report that distinction. Stopped `sync(require_running=True)` and
registration operations retain their no-op behavior; process status/stop remain
available after plugin loss or declared provider-construction failure.

## SparkRoute provider update

The bundled plugin now implements the 0.4 gateway contracts in its canonical
repository. `AdminError` derives from `GatewayOperationError`, preserving its
status, code, retryability, and transport cause. `SparkrouteEngine.query_models()`
returns `tuple[ProxyModel, ...]` and raises `GatewayQueryError` for an unavailable
or malformed status response; it no longer exposes provider rows through
`list_models_via_api()`. Callers should use `api.proxy.models()` or the supported
typed gateway method. Gateway registration returns the upstream class directly;
the host has no SparkRoute-specific adaptation. LiteLLM now implements the same
typed method directly. The legacy `list_models_via_api()` hook and supervisor
`model_query_error` attribute are removed; external providers must migrate to
`query_models()` and `GatewayQueryError`. `ProxyStatus.model_query_error` remains
available to API callers as an immutable observation diagnostic.

## Configuration binding and metadata storage

Each CLI invocation validates its explicit configuration path even when command
extensions are already loaded or a context is cached. A conflicting binding is a
CLI usage error before dispatch. Plugin discovery failure still permits process
recovery under the same configuration; it cannot redirect that operation to a
different config path. Applications continue to bind one profile/configuration
per process.

Job listing, stop, logs, and low-level metadata I/O now share configured-cache
precedence: an explicit `cache_dir`, then the supplied context's configuration or
the current application's configuration, whose missing setting uses profile/env
defaults. Omitting `sctx` no longer ignores a `cache_dir` setting in `config.yaml`.
Passive metadata reads do not initialize plugins, including after a failed plugin
bootstrap. Invalid configuration or config-property errors propagate instead of
silently selecting a different cache. An explicit cache permits metadata recovery
without reading a broken config file.


## Legacy helper and signature removals

| Previous import or input | 0.4 replacement |
| --- | --- |
| `core.recipe.ClusterConfig` | `core.recipe.LaunchOverrides`. The persisted `cluster_config` field remains supported. |
| `models.vram._resolve_quant_dtype` | `models.quantization.resolve_quantization(hf_config={"quantization_config": ...})`, then read `weight_dtype` if a result exists. |
| `core.fingerprint.fingerprint_host` | `core.hardware_probe.probe_host`; it returns accelerator and InfiniBand information together. |
| `orchestration.networking.distribute_cx7_host_keys` | `distribute_host_keys` in the same module. |
| `orchestration.docker.parse_teardown_removed` | `orchestration.teardown.parse_teardown_removed`. |
| Scheduler `run_schedule(progress_ui=...)` | `run_schedule(task_events=...)`; the task-event protocol is unchanged. |
| `detect_infiniband(..., head_host=...)`, `resolve_nccl_env(..., head_host=...)` | Omit the unused head-host argument; results remain per host. |
| `validate_ib_connectivity({host: ip})` | Pass `{host: [ip, ...]}`. The return value still maps each reachable host to one verified IP. |
| `prepare_images(..., overrides, source_image=...)` | Pass the already resolved recipe; source selection belongs to the shared image planner. |
| `RuntimePlugin.resolve_container(recipe, overrides)` | `resolve_container(recipe, host_hardware=...)`; apply image overrides to the recipe before resolution. Customize defaults with `default_image_for`, not a recipe-selection override. |

Private benchmark shells no longer accept both `fresh` and a resolved mode, or
unused runtime/config cleanup arguments. Frontends should call the public
benchmark API with `BenchmarkOptions.resume_mode` and consume its events. CLI
`--fresh` and `--resume` keep their existing meanings.

### Runtime image policy

`resolve_container()` selects a single host's image: explicit `recipe.container`,
then `default_image_for(host_hardware)`. The latter is the runtime customization
hook: a matching platform default takes precedence over `default_image_prefix`.
Without hardware, only the runtime default is used. No prefix/default yields an
empty selection, which the image planner rejects unless a per-host image exists.

`core.images.resolve_runtime_image_plan()` applies that policy to selected hosts
and overlays explicit `containers:` entries. Launch preparation, API
materialization, Kubernetes launch, tuning, and benchmark image summaries use this shared
plan. The planner does not rewrite recipe image declarations or workload identity.
A runtime that requires one build rejects differing per-host defaults; set an
explicit common image. An image-transforming builder also requires one source
image. Builder output remains authoritative for the subsequent launch plan.

The SparkRoute plugin now requires the host's 0.4 application APIs directly.
Workers use `sparkrun.application.initialize(config_path=...)`; child processes
preserve the selected application/config identity and inherited process settings.
The canonical plugin checkout owns these changes and the bundled snapshot is
imported from its committed source.

### CLI and saved-data boundaries

See the [CLI retirement and stored-data migration plan](LEGACY_MIGRATION_PLAN.md)
for the 0.4.x/0.5.x support windows and required conversion/recovery work. Legacy
ownership, benchmark results, Arena submission IDs, and recipe parsing remain
readable until their migration paths are implemented and verified.

### Internal benchmark result name

The compatibility alias `sparkrun.benchmarking.base.BenchmarkResult` was removed.
Internal orchestration tests use `BenchmarkExecution`; application callers use
`sparkrun.api.BenchmarkResult`, and integrations consume `BenchmarkMeasurement`.
The private execution record is not an integration or framework API.

### Benchmark task and image preparation contracts

Frameworks now implement the abstract `build_task_list` hook directly and return a nonempty list
of contiguous, zero-based `BenchTask` records. The single-call `None` fallback
has been removed. Commands return raw argv and write a JSON object to the supplied
`result_file`; do not shell-quote individual arguments. See the framework example
in [PLUGINS.md](PLUGINS.md) and [benchmark API details](BENCHMARK_API.md).

`tool-eval-bench` defaults to v2.6.0 and uses its native JSON-file command. Its
removed `experimental_async`, `llm_judge`, `perf_legacy`, and `perf_legacy_only`
options have no translation aliases. Backend detection and the 120-second request
timeout now follow upstream; explicit supported arguments still take precedence.
Text arguments retain their supplied spelling; only declared booleans become
flags. Infrastructure-only or missing measurements remain failed and resumable.
Partial completion is accepted if at least one scenario was graded. Standalone
modes that do not emit the tool-suite JSON artifact (including `perf_only`) are
rejected before launch; use `llama-benchy` for performance measurements.

Image preparation selects prepared overrides before unused defaults, and applies
the same alignment/string/runtime validation to both. Image-less executors have
no `PreparedImageSet.image_plan`; the image accessors return `None`/an empty tuple.
Pass its `container_distribution` to `distribute_from_config` when staging images;
generated transfer entries no longer mutate the recipe. Standalone integrations
such as ColdSnap use `stage_prepared_images()` and its `StagedImageSet` receipt;
this delegates to the same distribution function with the prepared container
policy and one operation-scoped configuration (`config.for_cluster(cluster)`).
`require_content_ids=True` verifies every host after image transfer and before
model transfer. Mutable tags become local Docker content IDs, while already
pinned references retain their identity. Missing or invalid host identities abort
staging; dry runs perform no live inspection. `resolve_content_images()` exposes
that strict per-host check independently. An optional `ssh_kwargs` mapping must
match the operation config rather than select a second transport. The unused `prepare_images(strategy_name=...)` diagnostic input is removed.
`ImagePlan.image_for_node` now raises `IndexError` for invalid resolved-node indices.


Local and remote `distribute_from_config()` calls now stage the same resolved
container/model entries, including explicit extra images and node targeting.
Prepared policies remain caller-owned; staging does not mutate them. Disabled
and skipped resource classes remain skipped on both transports.

Native `run()` and `benchmark()` calls require no unused image defaults.
`materialize()` is container-specific and rejects native executor targets.
`BenchmarkOptions.timeout=None` uses the benchmark-spec timeout, then 14,400
seconds. Framework request deadlines (tool-eval-bench: 120 seconds by default)
are independent; see [BENCHMARK_API.md](BENCHMARK_API.md) for a combined example.

### Typed configuration and operation results

Configuration paths accept `str` or `Path`. A present configuration file must
contain a mapping (an empty file is valid). The core `launch_inference()` helper
requires `config` or an `sctx` that supplies it; application callers normally use
`api.run()`. Omitting a registry manager skips registry tuning synchronization.
Indirect sudo requires a password for its selected sudo user and rejects an
absent password before transport.

Placement resolution advertises `RankAssignment | None`, and sudo orchestration
returns `RemoteResult` objects. Benchmark process handling and catalog/Tailscale
result annotations now match their runtime shapes. Recipe validation hooks may
return both plain suggestion strings and structured `RecipeIssue` records.


## Required runtime hooks and builder preparation

`RuntimePlugin` is an abstract base. Implement `generate_command()` before a
runtime can be instantiated or discovered. Command generation must return a
nonempty string; invalid values fail before runtime submission. Direct users of
`run_native_cluster()` must supply a recipe; omitted overrides still mean an
empty override mapping.

Builders implement `prepare(..., builder_context=None)` for both image and host
environment preparation. The `prepare_image()` hook and its forwarding adapter
are removed. Migrate overrides and direct calls to `prepare`; the base remains a
no-op, and registry pulls belong to distribution. Eugr implements the same hook.

`DistributionResourceConfig` is now generic in its entry type. Model resources
contain `DistributionModelEntry`; container resources contain
`DistributionContainerEntry`. `DistributionConfig` validates kinds at
construction and again before resolution, including resources edited by plugins.
Mismatched entry kinds raise `RecipeError` instead of being silently skipped.

Plugin API mismatches retain registration rollback and inventory failure details.
Normal CLI calls show a concise update instruction; debug logs include the
traceback. Shell completion suppresses plugin diagnostics for that invocation.


## Concrete provider inputs and results

`ApplicationProfile` materializes namespace and environment-prefix defaults into
string fields. Omit `config_namespace`, `cache_namespace`, `state_namespace`,
`resource_namespace`, and `env_prefix` when using defaults; do not pass `None`.
The [profile field reference](APPLICATION_PROFILES.md#profile-fields) describes
which other fields still accept `None`.

The shared `core.resolve.apply_recipe_overrides()` helper requires a keyword-only
`recipe` and returns `(Recipe, overrides)`. Resolve a recipe before calling it;
there is no recipe-free override mode. Application consumers normally pass
`RunOptions.overrides` to `api.run()` instead. SparkRoute passes normalized launch
overrides explicitly through its activation path; it no longer stores private
launch options on the recipe object or checks for host APIs required by 0.4.

Read-only hook command inputs and SSH host sequences accept `Sequence[str]`.
Kubernetes executor construction requires `K8sExecutorConfig`; passing an
unrelated executor config raises `TypeError` before operations begin. Kubernetes
log APIs declare their closeable generator return contract. Setup execution
requires configuration before approval callbacks or host actions run; dry-run
planning may still omit it.

Hardware platforms may implement
`default_accelerator_memory_gb(accelerator) -> float | None` for devices with a
qualified fixed capacity. Return a positive finite number or `None`; malformed
capacity defaults raise `ValueError`. Inventory measurements take precedence.
Both scheduling and per-host fit use this policy without changing stored probes.
See [memory capacity and fit](MULTIPLATFORM.md#memory-capacity-and-fit).
