# Cross-cutting plugins

`sparkrun.plugins` is the in-tree half of the plugin system; `plugins.paths`
loads the out-of-tree half. Both register the same way, so a first-party
integration has no capability an external one lacks. See `CLAUDE.md` for how
each is discovered.

This document covers the seams a *cross-cutting* integration uses — one that is
not "a runtime" or "an executor" and so has no single extension point of its
own.

## Owning a top-level recipe key

A plugin may own a top-level recipe key without adding its schema to `Recipe`
or hiding the settings in `metadata`. Ownership is exclusive and registered at
plugin bootstrap:

```python
from sparkrun.plugins import register_recipe_item


class SnapshotHandler:
    def parse(self, value, recipe):
        # Return a plugin-owned typed value.
        ...

    def validate(self, value, recipe):
        # Return issue strings relative to the owned key.
        return []

    def export(self, value, recipe):
        # Return YAML/JSON-compatible canonical data.
        ...


def register(variables):
    register_recipe_item("snapshot", SnapshotHandler(), owner=__name__)
```

The key must be lowercase and cannot conflict with a core recipe key or a key
owned by another plugin — a second owner must not be able to silently
reinterpret an existing recipe surface. Parsing failures name both the owner
and the key; validation issues are namespaced as `snapshot.<issue>`.

Parsed items are read with `recipe.plugin_item("snapshot")`. Use
`recipe.export_plugin_items()` for detached canonical values keyed by their
registered recipe keys. Catalog previews and discovered endpoints carry these
values in their generic `plugin_items` mapping; integrations interpret their own
entries and adapt them to their own wire protocol.

Four properties are load-bearing:

- **The key is excluded from the `runtime_config` sweep.** Unknown top-level
  keys are otherwise swept into `runtime_config`, which feeds the serve
  command — so without this a plugin's settings would be handed to the engine
  as flags.
- **Items round-trip at the same top level** through serialization, registry
  caching, and recipe export. A plugin key is recipe content, not a runtime
  detail, and re-exporting it somewhere else would break the next load.
- **A raw item survives its plugin being unavailable.** Reading a serialized
  recipe with the plugin disabled preserves the item verbatim rather than
  discarding it, so disabling a plugin never silently rewrites recipes.
- **Items participate in `derive_recipe_fingerprint` by default**, using the
  handler's canonical export. Existing execution plugins retain their identity
  behavior. A part is appended only when an item is present, so recipes predating
  the seam hash byte-identically.

A plugin supplying passive annotations can opt out of workload identity:

```python
register_recipe_item(
    "annotations", AnnotationHandler(), owner=__name__, affects_fingerprint=False
)
```

This keeps annotation edits from changing the workload fingerprint while retaining
normal parsing, validation and transport. Items contributing an execution strategy
or preparation steps cannot opt out. The per-item identity policy is saved with
the recipe so it remains stable when the plugin is unavailable. Older saved plugin
items default to participating unless their registered owner declares otherwise.

The owning plugin must be enabled to recognize its key when loading fresh recipe
YAML. With no registration, unknown top-level keys retain the existing runtime
configuration behavior and validation diagnostics. Already serialized plugin
items remain preserved even without their owner. A registered owner may also
recover its key from raw recipe data in older saved state; core does not maintain
integration-specific migrations or schemas.

## Owning how a recipe is executed

An owned item may also opt its recipes into **one** execution strategy and
contribute typed preparation steps:

```python
register_recipe_item(
    "snapshot",
    SnapshotHandler(),
    owner=__name__,
    execution_strategy=SnapshotExecutionStrategy(),
    preparation_steps=contribute_snapshot_preparation,
)
```

Both are **recipe-local**: installing the plugin has no effect on a recipe that
omits its key, so merely having it on disk never changes what `sparkrun run`
does. More than one active strategy is an error rather than a precedence rule —
two things claiming to launch the workload have no correct arbitration.

A strategy implements four hooks:

| Hook | Runs | Returns |
|---|---|---|
| `preparation_steps(ctx)` | before the launcher | `PreparationStep`s to schedule |
| `finalize_preparation(ctx, receipts)` | after those steps | `PreparedExecution` (asset policy + state) |
| `prepare_activation(ctx)` | assets resident, **before** eviction | an opaque receipt |
| `activate(ctx, receipt)` | in place of `runtime.run()` | `ActivationResult` |

Preparation steps form a small deterministic DAG — globally unique names,
explicit `requires` — and completed steps are compensated in reverse order if a
later one fails. Naming beats ordering here because two plugins contributing
steps have no shared list to order themselves within.

`LaunchAssetPolicy` is how a strategy declines parts of the shared pipeline
(builder, model, image distribution, entrypoint probe, tuning sync, page-cache
clear) and how it supplies `images_by_node` when it prepared the images itself.
Everything it does not decline still runs, so a strategy inherits distribution,
placement and preflight rather than reimplementing them.

Three boundaries are not negotiable:

- **The replacement barrier stays core-owned.** sparkrun completes plugin
  preparation, normal image/model preparation, *and* the strategy's
  prepare-only `prepare_activation` before it fires the `before_start` eviction
  hook. A strategy never decides when the deployment it replaces is torn down,
  and by the time eviction happens everything slow and interruptible is behind
  it.
- **The launcher still records job metadata**, with the same identity a normal
  launch records — cluster, SSH user, fingerprint, owner. `save_job_metadata`
  rewrites the file wholesale, so an omission here is an erasure, and the
  symptom (a teardown that cannot authenticate) looks nothing like the cause.
- **`RunOptions.strategy_options` is not workload identity.** Per-invocation
  choices belong there and are deliberately excluded from the recipe
  fingerprint and intent ID, the same way serve flags are.

## Runtime pre-serve hooks

A runtime overriding `RuntimePlugin._pre_serve` should call
`super()._pre_serve(...)` to retain recipe hooks and mods. The existing method
signature is unchanged. Built-in launch paths scope the full launch context
around that call, so the base implementation supplies the
[hook environment](../RECIPES.md#hook-environment-variables), including node
rank and resolved compilation-cache paths.

Context is isolated per invocation and reset on errors. Plugins that dispatch
hooks directly can pass `launch_context=HookLaunchContext(...)` to
`run_pre_exec`; supply the full selected host order even for a subset of
targets. Direct callers without launch context receive ranks derived from
their supplied host list and empty values for unavailable launch metadata.
Hooks dispatched in another thread need their context passed explicitly.

## Recipe environment templates

The standard launch pipeline detects templates inside recipe `env` and prepares
them after assets and cache
mounts are available, before the replacement barrier. Built-in solo and cluster
paths consume the scoped plan while assembling per-host environment, and pass
the same rendered values to container creation and serve execution. They also
validate that final mounts still match the prepared context. The runtime plugin
instance and recipe template source are not mutated with rendered values.

Custom runtimes using these shared paths inherit interpolation. A runtime
implementing its own launch path must call
`core.env_templates.render_launch_env(recipe, host, volumes, runtime_cache, executor)`
inside the standard launch scope and merge the result at the recipe environment
tier for both creation and serving. Raw calls to `runtime.run()` without that
prepared context reject active templates. Separate threads need explicit
context propagation.

`--transfer-mode delegated` remains supported: it changes download/distribution
and mod staging, while the standard pipeline still resolves target-host assets,
constructs mounts, and creates containers. It is independent of
`RecipeExecutionStrategy` and `RuntimePlugin.is_delegating_runtime()`. Those
plugin-owned execution paths bypass standard launch preparation and currently
reject active templates, as does `api.materialize()` without a prepared
environment contract. A future strategy integration should own its environment
lifecycle: provide actual prepared facts for fresh launches/capture, or use the
captured environment for snapshot restore. ColdSnap should own model/cache
logistics and compatible rebinding on restore; the core must not impose normal
Docker cache probes on that path. No strategy should silently drop templates or
pass unresolved source to an engine.

See [Environment templates](../RECIPES.md#environment-templates) for syntax and
availability, including MPI's restriction on differing per-node values.

## Preparation-only build hooks

`sparkrun build` and `api.build()` run builders and asset staging without the
execution strategy or its preparation DAG. A plugin that needs to contribute
build inputs can register a separate, recipe-local callback:

~~~python
from sparkrun.core.build_preparation import BuildContext


def prepare_build(context: BuildContext):
    context.recipe.distribution_config.add_model("org/auxiliary", revision="commit-sha")
    return {"engine": "vllm"}


register_recipe_item(
    "snapshot",
    SnapshotHandler(),
    owner=__name__,
    build_preparation=prepare_build,
)
~~~

The hook receives an operation-local recipe copy, resolved build plan, options,
and session context. It may declare distribution resources and return builder
context defaults (a mapping or `None`). Core supplies the runtime family as
`engine`; explicit `BuildOptions.builder_context` values override defaults.
Conflicting context values from two plugins are rejected.

Hooks must honor `context.options.dry_run`. They must not start inference,
capture or restore snapshots, reserve capacity, or evict workloads. Build hooks
require the same recipe trust authorization as executable recipe content, and
their items must participate in the recipe fingerprint. Installing a plugin
does not run its hook unless the recipe contains its registered key.

## Plugin settings that do not belong in a recipe

A recipe is portable; operational policy for a given site is not. `plugins.<name>`
in `config.yaml` is the per-plugin mapping for that, read with
`SparkrunConfig.plugin_settings(name)`:

```yaml
plugins:
  paths:                      # reserved for external plugin discovery
    - ~/src/sparkrun-plugins
  snapshot:
    artifact_generations: 2
```

`plugins.paths` stays reserved for discovery. Everything else under `plugins`
is a plugin's own namespace, so a plugin never needs a bespoke top-level config
block or a property on `SparkrunConfig`.

## Contributing a recipe registry

A plugin whose recipes are inert without it can declare the registry that holds
them, so `@<registry>/<recipe>` resolves for anyone with the plugin enabled and
nobody has to `sparkrun registry add` anything:

```python
from sparkrun.plugins import RegistryEntry, register_default_registry


def register(v):
    register_default_registry(
        RegistryEntry(
            name="coldsnap",
            url="https://github.com/sparksq/sparkrun-recipes.git",
            subpath="coldsnap-recipes",
            description="Qualified ColdSnap recipes",
            visible=False,       # stays out of `sparkrun list`; @coldsnap/… still resolves
        ),
        owner="coldsnap",
    )
```

Call it once per registry. `enabled` and `visible` are honored, so a control or
opt-in lane can ship disabled.

**Declarations are an overlay, not a config edit.** They are merged into the
list `RegistryManager` loads and are never written to `registries.yaml`, so
disabling or uninstalling the plugin takes its registries with it and leaves
nothing to clean up. The user's own file always wins on a name collision, which
is how they repoint a declared registry at an internal mirror.

User decisions still stick. `registry disable` / `trust` / `untrust` on a
declared registry **materializes** it — writes it into `registries.yaml` as an
ordinary entry the user now owns — and `registry remove` records a tombstone so
the declaration cannot put it back on the next launch. `sparkrun registry list`
grows a `Source` column showing `plugin:<owner>` when anything is declared.

Four rules worth knowing before you rely on this:

- **Registration does no I/O.** It records intent and validates; it does not
  clone, read or fetch. `init_sparkrun` runs on every shell completion, so
  anything expensive here lands on the interactive path — which is also why a
  plugin cannot point at a remote `.sparkrun/registry.yaml` and have its
  *names* come from there. Declare them in code; the repo manifest stays for
  users who add the registry by URL by hand.
- **Trust depends on where the plugin came from.** An in-tree plugin may ship
  `trusted=True` (it arrives through sparkrun's own review and release gate).
  For an out-of-tree plugin it is forced to `False` and the user grants it with
  `sparkrun registry trust <name>` — installing a plugin says "I want this
  capability", not "I grant its recipe repo standing permission to run
  lifecycle hooks from whatever it contains next month". Document that step in
  your install instructions if your recipes use `pre_exec` / `post_exec` /
  `post_commands`. The tier is set by the loader; a plugin cannot claim it.
- **Names are validated at declaration and raise.** `validate_registry_name`
  checks the *registry URL's* GitHub org against the reserved-name tables, and
  `assert_safe_registry_entry` checks the name and every subpath as filesystem
  paths. A reserved name from the wrong org fails at bootstrap, not later — so
  if you want a name reserved to your org, that entry belongs in
  `EXTERNAL_RESERVED_NAMES` in core.
- **You cannot redefine a shipped default.** Declaring `official` is refused
  with a warning naming your plugin, rather than silently ignored. A new name
  is the supported use; shadowing a curated one is not.

Nothing is fetched until something clones it, so suggest `sparkrun registry
update` after your plugin is first enabled rather than paying a clone inside
the next `sparkrun run`.

## Declaring the module API contract

Every top-level plugin module/package, whether bundled, loaded from a directory,
installed through an entry point, or registered directly, must declare:

```python
SPARKRUN_PLUGIN_API_VERSION = 1
```

This literal integer declares compatibility with the host plugin contract. It is
separate from the plugin's optional release `__version__` below and from the
application-profile schema version. Missing, boolean, string, or incompatible
values reject the plugin before its registration hook runs. The loader records
the failure and rolls back enlisted import-time contributions. Optional failures
allow other plugins to load; required integrations still block dependent work.
Update old directory plugins and add the declaration after checking their APIs
against the [0.4 migration guide](DISTRIBUTION_API_MIGRATION.md).

## Declaring a version

Set `__version__` on your plugin's top-level module or package:

```python
# sparkrun_thunder/__init__.py
__version__ = "0.2.0"
```

`sparkrun setup plugins list` reports it beside the plugin's source and gate
state:

```
NAME              VERSION  SOURCE    STATE  FLAG
sparkroute        0.1.0    in-tree   on     gateway.sparkroute
sparkrun_thunder  0.2.0    external  off    core.external_plugins
```

It is optional, and a plugin that declares nothing is reported `unknown` rather
than being given a version it did not claim. Two consequences of that rule:

- **Directory versions require matching source provenance.** Only the loaded
  module at that configured directory supplies its `__version__`; a same-named
  installed package does not supply a fallback. Installed entry-point rows use
  their owning package's metadata version. Bundled modules use their declared
  version, never the host package version.
- **Listing never imports a plugin for its version.** An unattempted or failed
  source has an unknown module version. A previously loaded source can still
  report its version after its gate is disabled.

Duplicate top-level module names in different configured plugin directories are
rejected before those modules are imported. Unambiguous plugins still load. A
module already imported from outside its configured directory is also rejected;
use distinct module names or separate application processes. Inventory attributes
loaded status and version to the actual source path and retains each rejected
source's diagnostic. Listing itself does not import plugins.

The `STATE` column separates the gate from the outcome. `on (load failed)`
means the flag resolves on but the import raised — run with `-v` for the
traceback. `off` means the plugin was never imported at all, which is the
point of the gate.

`--json` emits the same rows as an array, where an unknown version is `null`
rather than the string `"unknown"` — the latter is a display rendering, and
would be indistinguishable from a plugin that declared it.


## Installed integrations (API version 1)

Distributions and ordinary Sparkrun can explicitly select installed packages via
one shared entry-point group:

```toml
[project]
name = "example-hardware"
version = "0.1.0"
dependencies = ["sparkrun>=0.4.0,<0.5"]

[project.entry-points."sparkrun.plugins"]
example-hardware = "example_hardware"
```

The entry point targets a **module**, with `SPARKRUN_PLUGIN_API_VERSION = 1`.
Select it using `integrations: {example-hardware: true}` in user config or an
`ApplicationProfile` integration list. Installation alone never imports it.
Selected installed integrations use the existing class scan and `register(v)`
hook. They do not require the development-only `core.external_plugins` flag.
Configured directory loading and its existing kill switch remain available.

A module can expose `FEATURE_DEFINITIONS`, a sequence of `FeatureFlag` objects;
these register before its implementations are gated. For CLI contributions,
import `register_cli_command` from `sparkrun.core.cli_registry` and pass a lazy
loader with an explicit name to keep API initialization console-free. Registration
must be local and inexpensive: no dependency installation, network, registry
clones or hardware probes. Directory, bundled, and installed loaders roll back
supported registry mutations from a failed import or registration hook. Arbitrary
plugin I/O is outside this contract. The rollback adapter targets the project's pinned SAF version.

Discovery is deterministic. Repeated registration of the same provider is
idempotent. Distinct integration IDs/providers or concrete runtime, builder,
executor, scheduler, transport, framework, telemetry scope or gateway selector claims cannot silently
replace each other; diagnostics name both owners. Ordered platform matching is
preserved, with duplicate platform names rejected. Required missing, disabled,
incompatible or failed integrations block launch while help/version/inventory
remain usable. Optional load failures are reported as failures, not loaded
plugins. Ambiguous selected implementation claims also block launch.

`discover_installed_plugins()` and `installed_plugin_inventory()` return lists of
frozen `InstalledIntegration` reporting values. They expose package metadata and
load outcomes without entry-point handles. A retained snapshot does not change as
loading proceeds; callers cannot change required-integration enforcement through
inventory rows.

`setup plugins list --json` includes installed package/version, source,
selected/enabled, loaded, required, selection source and failure. A profile's
selected package still declares registries at the external trust tier; reviewed
profile registry declarations are a separate explicit policy.

The module's `SPARKRUN_PLUGIN_API_VERSION` declares its tested plugin contract
as a literal integer, currently `1`. The host exposes the supported version as
`sparkrun.core.registration.PLUGIN_API_VERSION`. Do not copy that constant into a
plugin declaration dynamically: a newer host does not establish compatibility
for an older plugin. Booleans, floats and strings are not version declarations.
Every installed, directory, and bundled plugin module must declare this API
version; all sources use the same compatibility check.

An absent or unsupported declaration prevents registration. For directory
plugins, the loader emits a concise update instruction and keeps the traceback
in debug logs. Failure information remains in inventory. Shell completion
suppresses host plugin-loading diagnostics.

Plugin API compatibility, `APPLICATION_PROFILE_API_VERSION`, and the serialized
controller `schema_version` are independent contracts. Also declare compatible
core package ranges in project requirements. See
[the application profile guide](APPLICATION_PROFILES.md).

These identifiers serve different purposes:

| Identifier | Example | Purpose |
| --- | --- | --- |
| Python distribution/package | `example-framework` | Installation and dependency versions. |
| Installed plugin ID | `example-framework` in `sparkrun.plugins` | Selection in `ApplicationProfile.integrations` or config `integrations`. |
| Feature gate | `integration.arena` | Enable a capability or bundled plugin; it is not an installed plugin ID. |
| Benchmark framework | `llama-benchy` | Measurement implementation selected with `framework`. |
| Benchmark integration | `arena` | Publication hooks selected with `BenchmarkOptions.integrations`. |

Only `sparkrun.plugins` is consumed for installed plugin discovery. Core
implementations are scanned from bundled packages. The unused per-kind package
entry-point groups (`sparkrun.runtimes`, `sparkrun.benchmarking`,
`sparkrun.builders`, and `sparkrun.executors`) were removed from host metadata in
0.4.0; they are not alternate registration routes. Inventory includes bundled,
directory, and installed modules, with selection/load/failure fields.

Import and registration errors are retained in `PluginInfo.failure` for every
source, even after contribution rollback. Failure identity includes the source
and directory/package; a failure in one directory does not appear against an
unattempted module with the same name elsewhere. Listing imports no plugin, even
when reporting an earlier failure for a currently disabled source. Successful
loading clears the prior failure for that source. This is process-local diagnostic
state, not a persistent retry service.

### Installing a benchmark framework

The package entry point targets the module that exports the framework class:

```toml
[project.entry-points."sparkrun.plugins"]
example-framework = "example_framework"
```

For example, `example_framework/__init__.py` can adapt a command-line measurement
tool that writes one JSON result artifact:

```python
import json
import shutil
from copy import deepcopy
from pathlib import Path
from sparkrun.benchmarking.base import BenchmarkingPlugin
from sparkrun.benchmarking.scheduler import BenchTask

SPARKRUN_PLUGIN_API_VERSION = 1

class ExampleFramework(BenchmarkingPlugin):
    framework_name = "example-rate"

    def check_prerequisites(self):
        return [] if shutil.which("example-bench") else ["Install example-bench"]

    def build_task_list(self, base_args, schedule):
        entries = [{}] if schedule is None else schedule
        return [BenchTask(i, "measurement", {**deepcopy(base_args), **deepcopy(entry)}, deepcopy(entry))
                for i, entry in enumerate(entries)]

    def build_benchmark_command(self, target_url, model, args, result_file=None):
        return ["example-bench", "--url", target_url, "--model", model,
                "--result-file", result_file]

    def parse_results(self, stdout, stderr, result_file=None):
        return json.loads(Path(result_file).read_text())
```

Select the installed plugin with `integrations: {example-framework: true}`, then
use `BenchmarkOptions(recipe="...", framework="example-rate")`. The loader scans
the concrete class; no `register(v)` hook is needed here. `sparkrun.benchmarking`
is the internal SAF extension point, not a Python package entry-point group.
Every framework implements `build_task_list`, including single-invocation tools.
The base is abstract: incomplete subclasses cannot be instantiated and are not
discovered as concrete framework plugins.
Return a nonempty list of `BenchTask` records with contiguous zero-based indices;
returning `None` no longer selects a separate execution path. Each successful
command must write a JSON object to `result_file`. Arguments are raw subprocess
argv, without shell quoting. The scheduler handles deadlines, process cleanup,
artifact validation, and retries; resume reconstructs tasks from saved entries.
Implement `consolidated_coverage_keys()` against the raw consolidated artifact
and `measured_nothing()` against the parsed result when exit zero does not ensure
usable measurements. Coverage must leave unmeasured tasks retryable; the final
empty-measurement check alone cannot provide recovery. Tool-eval-bench uses these
hooks to distinguish infrastructure failures from graded model-quality failures.

Image-less executors are supported by `run()` and `benchmark()` without default
image lookup. Image provenance comes from launch/captured deployment context.
`materialize()` is specifically a container-unit API and rejects native targets.


### Application and controller identity

Plugins can import `get_application_profile()`, `get_application_identity()` and
`get_controller_identity()` from `sparkrun.application`. Context-bearing hooks
can use `sctx.application_profile`, `sctx.application_identity`, and
`sctx.controller_identity`. A controller is one application/profile using one
canonical config directory; every process and config file in that directory
shares its opaque ID. There is no configurable controller ID. The full profile
is local policy; the serializable identity descriptors are suitable for shared
services. Key ownership by
`(application.id, controller_id)` and use `controller.labels()` to tag resources.
See [identity, persistence and shared-service contracts](APPLICATION_PROFILES.md#plugin-identity-and-shared-services)
for the wire format, lazy ID creation and migration behavior.

### Hardware inventory extensions

An installed integration can register a platform with `register_platform` and
extend the target probe through
`core.hardware_probe_extensions.register_hardware_probe`. Device-specific shell
facts and the hardware enricher stay in the plugin. The normal fingerprint and
combined accelerator/IB probes include selected providers, and failed installed
plugin registration rolls back their probe registrations. See
[MULTIPLATFORM.md](MULTIPLATFORM.md#plugin-owned-hardware-probes) for the contract.
Plugin-specific devices and qualification tests belong to the integration repository.

Framework task definitions contain measurement arguments only. The scheduler
injects `args["api_key"]` into the detached dictionary passed to
`build_benchmark_command()` at execution time. Frameworks must not depend on
finding it in `build_task_list()`, persist it in result artifacts, or log it.
Publication hooks receive credential-free argument and recipe projections.

## Benchmark integrations

A `BenchmarkingPlugin` implements a measurement framework: command building,
result parsing, and scheduled tasks. A **benchmark integration** adds behavior
around that measurement, such as publishing results. Integrations use the same
execution path from the CLI and Python API.

Register `BenchmarkIntegration` with
`sparkrun.core.benchmark_integrations.register_benchmark_integration` in the
plugin's `register(v)` hook. Its callbacks are:

- `prepare(defaults, context) -> BenchmarkDefaults`: select measurement defaults
  before resolving a new invocation's recipe/profile/framework. `BenchmarkDefaults`
  contains only `category`, `framework`, `profile`, and read-only `bench_args`.
  Return a new value with `dataclasses.replace`; lifecycle, targets, dry-run mode,
  callbacks, and integration selection remain caller-owned. Returning full
  `BenchmarkOptions` is rejected. This hook is not called by `resume_benchmark(id)`;
  saved measurements keep their original configuration.
- `validate(context)`: validate the effective integration settings before any
  binding or launch. This runs for fresh runs, incomplete resumes, and completed
  publication retries. Saved settings are merged first; explicit settings win.
  Move settings validation out of `prepare` into this hook.
- `on_bind(context)`: authenticate and establish persistent state before launching
  or resuming measurements. Saved settings and data have already been restored.
- `on_checkpoint(context)`: capture provenance after launch and before framework
  execution. This callback can run more than once and must be idempotent.
- `on_complete(context)`: finalize a successful benchmark, preview a dry run, or
  retry finalization for saved, validated results. It never runs after a failed
  measurement and must tolerate repeated calls for the same state.

`context` carries `sctx`, `settings`, an emitter, mutable integration-private
`data` and `outcome` mappings, optional `BenchmarkMeasurement` and
`BenchmarkStateInfo` snapshots, and `dry_run`/`resumed`. `BenchmarkReporter`
defines supported `info`, `warning`, `error`, `event`, and `confirm` methods.
`sctx` exposes application and controller identities for shared services.

`prepare` has no result/state; `validate` has restored settings/data/state but no
measurement snapshot. The other hooks see a frozen `BenchmarkMeasurement`,
built only for hooks that are present and refreshed before each hook. It contains measurement identity, framework/category,
profile, results, output paths, hosts/image, benchmark arguments, and resume
provenance. `measured_at` and `completed_at` describe the measurement interval
and stay fixed during publication-only retries. Nested result/provenance mappings
and sequences are read-only.
The snapshot exposes `recipe_yaml` and a redacted `provenance` mapping when a
recipe is available. It contains no live Recipe, runtime, launcher, or framework
plugin object. Snapshot construction does not resolve builder images: provenance
uses an already recorded archival reference, or the recorded container reference.
`framework` is a name. If a plugin needs to validate the recipe,
it can construct its own Recipe from the YAML snapshot.

Write publication outcomes into `context.outcome`; the host copies this into
`api.BenchmarkResult.integration_results[<your integration name>]`. Each plugin
can write only its own outcome. Persist values needed for retries in
`context.data`; `outcome` describes the current invocation and is not itself
persisted. Publication-only retries can have `recipe_yaml=None` and empty
`provenance`, so save original provenance during binding/checkpointing.

`on_bind` precedes inference launch. By `on_checkpoint`, launch/readiness has
finished (or was skipped). `on_complete` sees successful measurements or a dry-run
preview. The mutable internal `BenchmarkExecution` remains orchestration-only.
Use `api.BenchmarkResult` for API outcomes and `BenchmarkMeasurement` for
integration observations. The old `benchmarking.base.BenchmarkResult` alias was
removed in 0.4; frameworks do not receive the private execution record.
`BenchmarkStateInfo` exposes the benchmark ID, creation/update timestamps, and a
detached, top-level read-only `extras` mapping for legacy migration. It has no
`save()` method. New plugins should persist only through `context.data`; internal
state-file keys outside their data are not a stable plugin API.

State callbacks run while the benchmark state-directory lock is held. The host
persists each integration's settings and data,
including when a binding/checkpoint/completion callback fails. Rejected settings
from `validate` are not persisted and do not overwrite the last accepted settings.
Store only JSON-compatible values, never credentials. Core copies and filters
structured credential fields without interpreting serialized documents by key
name. Plugins own validation and sanitization of documents they persist; use
`public_recipe_text()` from `sparkrun.benchmarking.metadata` at an actual recipe
boundary. Arena sanitizes its saved `effective_recipe_text` before publication;
malformed legacy content produces a fresh-run diagnostic without parser excerpts.
Unavailable plugins' saved documents remain opaque until their owner is loaded.
Ordinary caller-provided
`BenchmarkOptions.state_extras` is also copied into newly created state; see
[caller metadata and reserved keys](BENCHMARK_API.md#caller-metadata).

Every framework saves validated task results for completion retries;
`benchmark resume <ID>` can retry publication after inference has stopped.
Dry runs invoke preview hooks without persisting
integration data; each plugin must also honor `context.dry_run` for its own I/O.

To contribute flags, register a `CliOptionSpec` from the Click-free
`sparkrun.core.cli_registry`, targeting `benchmark.run` (generic and category
commands) or `benchmark.resume`. Supply a lazy loader returning `click.Option`
objects and a decoder returning a settings dict, or `None` when not selected.
The host resolves options before parsing/help/completion, rejects parameter name
and flag collisions, and forwards the decoded mapping to `BenchmarkOptions`.
Use the same `feature_flag` on the lifecycle and option specs. Option and lifecycle
registrations participate in rollback for every plugin loader.

The in-tree [sparkarena plugin](../src/sparkrun/plugins/sparkarena/README.md) provides a
complete example. `integration.arena` gates its commands, flags, and lifecycle.
Sparkrun enables it by default; application profiles can override that default.
Saved integrations resume automatically when available. Disabling or uninstalling
one leaves its saved data intact and permits local benchmarking to continue;
explicitly selecting an unavailable integration fails with a clear error.

Python callers now select Arena with
`BenchmarkOptions(recipe='...', integrations={'arena': {}})` or use
`{'arena': {'local_test': True}}` for a rehearsal. This replaces the former
`arena=True` field. The result exposes
`result.integration_results['arena']['submission_id']` and `uploaded`, replacing
the Arena-specific top-level `submission_id` field. Telemetry records integration
names only, never their settings, saved data, or outcome details.

## Kubernetes plugin and executor-owned launches

The in-tree [Kubernetes plugin](../src/sparkrun/plugins/k8s/README.md) is loaded
through `integration.k8s`: off for Sparkrun stable/beta and on for alpha.
Application profiles can override that policy. Its `FEATURE_DEFINITIONS` register
`executor.k8s`, `cli.setup.k8s`, and `api.run.k8s` only after the loading gate is
on. All three child features default on inside the enabled plugin. Explicit
child overrides do not load a disabled parent.

An executor may set `config_class` to an `ExecutorConfig` subclass with a
`from_chain()` implementation for its own fields. The shared resolution chain
selects this class instead of keeping plugin-specific fields in core.

Executors with destinations beyond the host list implement `resolve_target()`
and return `ExecutorTarget` from `sparkrun.orchestration.executor`. Its derived
`destination_key` participates in deterministic workload placement identity;
its read-only `config` carries destination and connection settings. Core retains
this as `RunPlan.executor_target` and uses the scoped cluster for occupancy,
early ensure, and replacement. `options.executor_overrides()` passed to handlers
already includes the pinned target. Do not re-read current-context or other
mutable destination defaults. See [executor targets](EXECUTORS.md#operation-targets-and-endpoint-support).

Plugins with a native control-plane launch may register a
`RunHandler(executor, callback, feature_flag=...)` through
`sparkrun.core.run_handlers.register_run_handler()`. The typed callback receives
`(options, sctx, *, plan: RunPlan, started_at: float, before_start)` and returns `RunResult`.
The plan is the existing resolved recipe/cluster/placement/identity decision;
handlers must not independently repeat placement. Core enforces execution-strategy
compatibility before dispatch. For real launches, `before_start` is an idempotent
callback owned by core. Call it after validation and staging succeed,
immediately before starting/submitting the new workload. Controller-native handlers
call `before_start(executor=resolved_executor)` so replacement uses the exact
connection settings, includes a prior deployment with the same ID, and requires
confirmed teardown. Container launchers call it without arguments and retain
their own per-container cleanup. Let any callback failure
abort submission. During dry-run it is `None`; no replacement or submission occurs.
Core does not stop the previous deployment merely by dispatching a handler.
Kubernetes uses the same private preparation/submission implementation as
`launch_jobset()`. It invokes the callback after manifest construction and
feasibility checks, then saves the resource reference before submission. A submission
failure after replacement can still leave the old deployment stopped.
Both default and plugin
launches preserve typed `SparkrunError` errors and translate other exceptions with
their cause; interrupts propagate unchanged. A runtime failure may instead return
a nonzero `RunResult.rc`; consumers must inspect that public status. A plugin need
not create the private `launch_result` handle. Core completes shared result fields
for both launch paths: identity components from the returned portable cluster ID
(with the plan as fallback), the plan's recipe fingerprint, operation start time,
preview flag, and an operation timeline when the handler omits one. Handlers supply
actual substrate outcomes such as hosts, command, image, and configured serve port.
Ensure hits describe an existing deployment and keep unavailable fields empty;
they do not inherit the new plan's fingerprint or a launch timeline. This also
applies when the handler discovers reuse: return `already_running=True`, preserve
verified existing metadata, and do not invoke the replacement callback.

Native handlers persist `save_job_metadata(..., executor=resolved_executor,
native_resource={...})` before submission. The executor owns the reference schema;
core stores it separately from the portable ID. Implement
`Executor.stop_workload(cluster_id, *, metadata=None)` to confirm controller deletion
and return a removed-resource count (zero when already absent). Raise on errors;
return `None` only to select ordinary container teardown. A recorded native
reference cannot fall back silently. Implement `query_status()` with the same
portable IDs so shared discovery, ensure, and replacement see the workloads.
Common `api.logs()` currently rejects recorded native resources; provide a plugin
log API rather than addressing controller children with container-style names.
Executors whose hosts represent a control plane rather than a serving endpoint
set `supports_host_endpoint = False`. Benchmarking rejects them before launch,
including skip-run discovery and resumes that need more measurements. Completed
artifact processing and publication-only resume-by-ID remain available without
an inference endpoint. Benchmarking also rejects nonzero real launch status
before endpoint waits, checkpoint hooks, or measurement.
The Kubernetes plugin uses the same
executor configuration chain for its target settings. `options.executor_overrides()`
returns the caller layer for `resolve_executor()`; do not independently merge the
recipe/cluster/default layers.

Setup plugins can contribute read-only probes, per-host readiness checks, and
optional apply/undo actions through the [shared setup step API](SETUP_STEPS.md).


Failures in completion hooks, snapshot/outcome conversion, or completion-state
persistence raise `api.BenchmarkIntegrationFailed`, a subclass of `BenchmarkFailed`.
Its `result` retains successful measurements and output paths; errors describe
finalization separately from measurement success. `integration` names the failing
plugin, or the reserved `<state>` marker for shared persistence failures. Secondary
persistence failures are retained without masking the original hook error.
When completed state was saved, retry using `resume_benchmark(id)`.

Python frontends pass `BenchmarkOptions.decision_callback` (or the same argument
to `resume_benchmark`) for integration confirmations. Without a callback,
`emitter.confirm(message, default=...)` uses the supplied default. The removed
`on_prompt_required` and `on_complete_state` options are replaced by the same
`BenchmarkDecision` callback for resume choices. Recipe trust remains separate.
See [the benchmark API guide](BENCHMARK_API.md) for events and decision kinds.

Arena's in-tree CLI adapter is versioned with core and intentionally uses private
CLI helpers to reuse its command layout. Those helpers are not a supported API
for installed plugins. Use `CliOptionSpec` for contributions to existing commands,
or register your own lazy command and call the public benchmark APIs. No import
from `sparkrun.cli` is needed during plugin registration or library initialization.


### Registry rollback state

A registry that stores plugin contributions enlists its containers beside their
module-level definitions:

```python
from sparkrun.core.registration import enlist_registry_state, register_unique

_ENTRIES = {}
enlist_registry_state(globals(), "_ENTRIES")

# Inside the registry's registration function, after domain-specific validation:
# register_unique(_ENTRIES, name, spec, description="Example extension")
```

Installed, directory, and bundled loaders share
`core.registration.load_and_register_plugin()` for import, compatibility checks
and registration in one transaction. Direct `load_plugin_module()` registration
uses the same boundary. `core.registration.registry_transaction(v)` is available
for explicit transactions over enlisted containers and SAF state. Only completed
registration is reported as loaded. A failed hook rolls back its contributions
before optional-plugin error handling continues to independent plugins.
Setup dependency graphs are checked before committing module registration.
Same-module forward references are supported; dependencies on another module
must already be registered. Unknown providers and cycles roll back the module
instead of breaking all later setup planning. See [setup dependency policy](SETUP_STEPS.md).
Nested mutable containers and registry imports during registration participate
in rollback. Identical registrations are allowed; conflicting providers fail.
Enlistment tracks the namespace entry, so deliberately replacing a container
does not leave rollback pointing at its old value. Registration finishes before
application workers start. It does not roll back arbitrary plugin I/O, and the
SAF adapter remains tied to the pinned framework version.


### Finalization and setup undo contracts

`api.BenchmarkFinalizationFailed` is the common result-bearing exception after
measurement validation. `BenchmarkIntegrationFailed` remains its subtype for
plugin and shared-state failures. Export, inference cleanup, and final frontend
notification failures now retain measurements too; inspect `stage` and `errors`.
See [the benchmark API guide](BENCHMARK_API.md#results-and-publication-retries).

Setup apply and undo callbacks share result validation. Return `ok` from undo
only when the recorded change has been removed or is already absent. `warn`,
`skip`, `fail`, exceptions, and invalid statuses leave the host's record available
for retry. Return useful detail explaining incomplete cleanup. Successful hosts
are cleared independently before frontend notification; other hosts and
unselected phases remain recorded. Unresolved dependents block prerequisite
undo on the same host, even through transitive or filtered dependencies.
Built-in and plugin undo run through the
same console-free API. See [shared setup](SETUP_STEPS.md).

### Scheduled artifact coverage and processing recovery

The host tracks successful artifacts by stable task index, including sparse
successes. `consolidated_coverage_keys()` defaults to `None`, meaning no additional
semantic check; override it together with `task_coverage_key()` to verify requested
measurements. An empty set means no measurement coverage. Framework result data
remains framework-owned; the host does not insert bookkeeping into it.
The internal `gap_analysis()` receives `completed_indices` explicitly.

Completed commands and committed measurements are distinct recovery stages.
After interrupted consolidation, parsing, or result persistence, resume rebuilds
results from accepted artifacts using the saved framework/specification, without
inference or command prerequisites. Missing/invalid artifacts make only those
tasks pending again. Publication requires validated, committed measurements.
`run_schedule()` no longer accepts the unused `skip_run` argument. Session warmup
is independent of whether the caller launched inference; public
`BenchmarkOptions.skip_run` still controls inference launch.


Executor launch preparation may validate/snapshot local inputs through
`prepare_launch(extra_opts=...)` before core's replacement barrier. The hook is
also called for previews and must not launch workloads or mutate remote state.
Docker uses it to freeze local seccomp policies and embeds those contents in
all generated node scripts. `apply_runtime_adjustments(..., defaults=...)`
receives the lower-priority `Variables` chain as read-only input; its returned
mapping retains its existing position in the resolution chain.

### Effective benchmark provenance

Benchmark checkpoints now capture effective image, serving options, runtime
information, and recipe state before commands run. Exported YAML and integration
snapshots share the effective recipe projection; `recipe_yaml` corresponds to
`provenance.recipe.effective_hash`, while `provenance.recipe.hash` retains its
meaning as the declared-input hash. `raw_container` preserves the declaration.
A missing historical image remains unknown during processing-only recovery.
Publication-only retry payloads remain the integration's responsibility in
`context.data`, as described above.


## Gateway extensions

Use `sparkrun.proxy.gateway.register_gateway` for deferred engine registration.
New engines implement the typed model-query contract in
[`sparkrun.proxy.contracts`](PROXY.md#gateway-plugin-contract), using the same
`ProxyModel` class returned by the public API. Console, console-credential
issuance, and admin-token management are separate optional protocols. Existing
dictionary-returning providers must migrate their own response decoding to this
typed contract; the supervisor supplies no legacy model adapter.

## OCI Relay image distribution

The bundled OCI Relay adapter defaults on for alpha and off for stable/beta.
See [OCI Relay](OCI_RELAY.md) for enablement, platform support, and the
commit-pinned vendoring workflow.
