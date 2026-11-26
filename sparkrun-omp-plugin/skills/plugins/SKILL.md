---
name: plugins
description: "Author, package, test, and publish out-of-tree sparkrun plugins: entry-point discovery, extension points, recipe items, execution strategies, and trust."
---

# Authoring sparkrun plugins

A sparkrun plugin is a Python package that registers runtimes, builders,
executors, schedulers, transports, telemetry, benchmark frameworks, gateways,
platforms, recipe keys, or other cross-cutting behavior through
`sparkrun.plugins`. `sparkrun plugins add` installs an existing plugin; it is not
an authoring step.

Choose the right home first:

- **Out-of-tree**: a separate distribution with a `sparkrun.plugins` entry point
  for integrations that map onto one extension point.
- **In-tree `src/sparkrun/plugins/`**: only for cross-cutting integrations that
  must ship with core. A single-kind runtime/executor/builder belongs in its own
  top-level package instead.
- **Public plugin imports**: `sparkrun.plugins.__all__` is authoritative for
  names from that module; use a dedicated public facade where the extension
  table lists one. Operational calls belong to the separately documented
  `sparkrun.api` and `sparkrun.application`. Do not import private implementation
  modules as a workaround for a base that is not publicly exported.

## Package skeleton

```toml
[project]
name = "example-plugin"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["sparkrun>=0.4,<0.5"]

[project.entry-points."sparkrun.plugins"]
example-plugin = "example_plugin"
```

The entry point implements `register(v)`. The complete, typed copyable module
example appears under “Owning a recipe key”; other snippets are explicitly
identified as fragments or pseudocode.

`sparkrun.plugins` is the **only** installed-package entry-point group. The
per-kind groups (`sparkrun.runtimes`, `sparkrun.builders`,
`sparkrun.executors`, `sparkrun.benchmarking`) were removed in 0.4.0 and are not
alternate routes. Installation alone never imports a package: select it with
`sparkrun plugins enable example-plugin`, a `plugins.enabled` config entry, or
an application profile integration list.

A managed-Git install (`sparkrun plugins add <url>`) reads a `plugin.toml`
beside `pyproject.toml`:

```toml
schema = 1
name = "example-plugin"
module = "example_plugin"
source = "src/example_plugin"
repository = "https://github.com/example/example-plugin"
sparkrun = ">=0.4,<0.5"
# feature = "example.enabled"            # optional parent gate
# application_profile_api = 1            # only for alternate profiles
```

The source-copy installer requires static `pyproject.toml` metadata, a pure
Python package with a literal `SPARKRUN_PLUGIN_API_VERSION` in its entry module,
explicit Python and Sparkrun ranges, and dependencies already satisfied on the
host. Symlinks, submodules, compiled payloads, conflicting providers, and
unsupported layouts are rejected. Both top-level packages and
`sparkrun.plugins.<package>` layouts work.

The copyable reference package is
`tests/fixtures/application_profiles/plugin/`.

## Entry-module contract

Every plugin entry module declares a **literal integer**. This is a declaration
fragment, not a standalone module:

```text
SPARKRUN_PLUGIN_API_VERSION = 1
```

Missing, boolean, string, float, or incompatible values reject the plugin before
`register(v)` runs. Do not compute it from a host constant; a newer host does not
establish compatibility for an older plugin.

Optional declaration fragment (not a complete manifest):

```text
__version__ = "0.1.0"                 # release version; omitted means "unknown"

SPARKRUN_PLUGIN_MANIFEST = {
    "api": 1,                          # plugin contract version
    "features": {...},                 # static feature definitions
    "hooks": {...},                    # lifecycle event -> "module:function"
}
```

`FEATURE_DEFINITIONS` (a sequence of `FeatureFlag`) is the dynamic alternative.
The loader registers static features, then dynamic features, then scans concrete
SAF subclasses, then calls `register(v)`. Registration must be local and cheap:
no network, dependency installation, registry clones, or hardware probes.
`SPARKRUN_PLUGIN_SCAN_SUBMODULES = False` keeps the loader from importing
unrelated submodules.

## Extension points

| Base class | Public import | Extension point | Selector / discovery |
| --- | --- | --- | --- |
| `RuntimePlugin` | `sparkrun.plugins` | `sparkrun.runtime` | `runtime_name`; class scan |
| `BuilderPlugin` | `sparkrun.plugins` | `sparkrun.builder` | `builder_name`; class scan |
| `Executor` | `sparkrun.plugins` | `sparkrun.executor` | `executor_name`; class scan |
| `Scheduler` | No public base export in 0.4.3 | `sparkrun.scheduler` | `scheduler_name`; class scan |
| `Transport` | `sparkrun.transports` | `sparkrun.transport` | `transport_name`; class scan |
| `TelemetryProvider` | `sparkrun.orchestration.telemetry` | `sparkrun.telemetry` | scope/feature gate; class scan |
| `BenchmarkingPlugin` | `sparkrun.plugins` | `sparkrun.benchmarking` | `framework_name`; class scan |
| `HardwarePlatformPlugin` | `sparkrun.plugins` | `sparkrun.platform` | `platform_name`; explicit `register_platform` only |

`Transport` and `TelemetryProvider` are exported by their listed package
facades, not by `sparkrun.plugins`. In this checkout, `Scheduler` is defined in
`sparkrun.core.scheduler` but has no supported public base-class export; do not
import that internal module for an out-of-tree plugin. Ask for a public scheduler
seam before implementing one.

The Sparkrun base classes already implement the SAF multi-extension hooks;
subclasses normally inherit them and must not add unconditional overrides.
`RuntimePlugin`, `BuilderPlugin`, `Executor`, `Scheduler`, `TelemetryProvider`,
`BenchmarkingPlugin`, `Transport`, and `HardwarePlatformPlugin` provide
`is_enabled` / `is_multi_extension` behavior. In particular, runtime, builder,
executor, and telemetry bases apply their `required_feature_flag` in the
inherited gate; replacing that method with `return True` exposes a gated class
when its feature is disabled. Transport checks its feature at use time instead.
If implementing SAF hooks directly, preserve the multi-extension behavior
(`is_enabled` must be false) and any base-class feature gate. `sparkrun.platform`
is declared but deliberately not scanned: platform resolution is order-sensitive,
so register it through `register_platform`.

## Registry-style registrations

These are one-line registrations from `sparkrun.plugins`, typically called in
`register(v)`:

| Call | Purpose |
| --- | --- |
| `register_recipe_item` | Own a top-level recipe key. |
| `register_recipe_format` | Teach registries another launcher's manifest format. |
| `register_default_registry` | Declare a registry a plugin needs. |
| `register_gateway` | Add a deferrable inference-gateway engine. |
| `register_benchmark_integration` | Add publication/integration behavior to benchmarks. |
| `register_cli_command` / `register_cli_options` | Add lazy-loaded CLI commands/options. |
| `register_plugin_daemon_op` / `register_plugin_daemon_route` | Expose daemon operations/routes. |
| `register_image_distribution_provider` | Move images with a custom transport. |
| `register_model_distribution_provider` | Move model files with a custom transport. |
| `register_kv_strategy` | Add a KV-cache strategy. |
| `register_workload_lifecycle` | Observe workload lifecycle events. |
| `register_setup_step` / `register_setup_constraint` | Add host setup actions or preconditions. |
| `register_plugin_diagnostics` | Add a diagnostics section. |
| `register_run_handler` | Handle a run mode. |
| `register_hardware_probe` | Extend the target hardware probe. |
| `enlist_registry_state(globals(), "_ENTRIES")` | Roll back a module registry on failed load. |
| `register_unique(_ENTRIES, name, value, description=...)` | Conflict-checked insert into that registry. |

Every registration is conflict-checked and rolled back with a failed plugin
load:

- Registering an **equal** value again is a no-op.
- Anything else (different value, owner, or a changed definition under the same
  owner) raises `PluginConflictError`.
- Registries never silently replace an entry; register a new name instead.

Equality compares callables by definition, so a re-imported module's functions
are equal to the originals.

## Owning a recipe key

```python
"""Copyable, Pyright-checked example for src/example_plugin/__init__.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from scitrera_app_framework import Variables
from sparkrun.plugins import ExecutionContext, PreparationStep, Recipe, register_recipe_item

# pyright-checked copyable example
SPARKRUN_PLUGIN_API_VERSION = 1
__version__ = "0.1.0"
SPARKRUN_PLUGIN_MANIFEST = {
    "api": 1,
    "features": {"example.enabled": {"description": "Example capability", "default": False}},
    "hooks": {},
}


@dataclass(frozen=True)
class SnapshotConfig:
    size: int


class SnapshotHandler:
    def parse(self, value: Any, recipe: Recipe) -> SnapshotConfig:
        if not isinstance(value, dict):
            raise TypeError("snapshot must be a mapping")
        size = value.get("size")
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError("snapshot.size must be an integer")
        return SnapshotConfig(size=size)

    def validate(self, value: Any, recipe: Recipe) -> list[str]:
        if not isinstance(value, SnapshotConfig):
            return ["snapshot must be parsed before validation"]
        return [] if value.size > 0 else ["size must be positive"]

    def export(self, value: Any, recipe: Recipe) -> dict[str, int]:
        if not isinstance(value, SnapshotConfig):
            raise TypeError("snapshot must be parsed before export")
        return {"size": value.size}


def stage_snapshot(_context: ExecutionContext, _receipts: Mapping[str, Any]) -> None:
    return None


def contribute_snapshot_preparation(_context: ExecutionContext) -> tuple[PreparationStep, ...]:
    return (PreparationStep(name="snapshot.prepare", action=stage_snapshot),)


def register(v: Variables) -> None:
    register_recipe_item("snapshot", SnapshotHandler(), owner=__name__, preparation_steps=contribute_snapshot_preparation)
```

The key must match `^[a-z][a-z0-9_-]*$` and cannot collide with a core recipe key
or another owner. Validation issues are namespaced as `snapshot.<issue>`.
Parsed values are read with `recipe.plugin_item("snapshot")`; canonical values
are collected with `recipe.export_plugin_items()`.

Four properties matter:

- The key is excluded from the `runtime_config` sweep, so plugin settings are
  not passed to the engine as serve flags.
- Items round-trip at the same top level through serialization, registry
  caching, and export.
- A raw item survives its plugin being unavailable, so disabling a plugin never
  rewrites recipes.
- Items participate in `derive_recipe_fingerprint` by default, using the
  handler's canonical export.

Passive annotations may opt out of workload identity. This is an illustrative
call fragment; define `AnnotationHandler` in the plugin before using it:

```text
register_recipe_item(
    "annotations", AnnotationHandler(), owner=__name__, affects_fingerprint=False
)
```

Items contributing an execution strategy, preparation steps, or a build hook
cannot opt out. A registered owner may also recover its key from raw data in
older saved state.

`FunctionalRecipeItemHandler` wraps plain functions when a class is overkill.

## Execution strategies

An owned item may opt its recipes into **one** execution strategy and typed
preparation steps. This illustrative call fragment assumes a strategy class
defined by the plugin; the copyable example above shows the complete
`preparation_steps` callback contract:

```text
register_recipe_item(
    "snapshot",
    SnapshotHandler(),
    owner=__name__,
    execution_strategy=SnapshotExecutionStrategy(),
    preparation_steps=contribute_snapshot_preparation,
)
```

Both are recipe-local: installing the plugin has no effect on a recipe that
omits the key. More than one active strategy is an error, never a precedence
rule.

| Hook | Runs | Returns |
| --- | --- | --- |
| `preparation_steps(ctx)` | Before the launcher | `PreparationStep`s to schedule |
| `finalize_preparation(ctx, receipts)` | After those steps | `PreparedExecution` |
| `prepare_activation(ctx)` | Assets resident, before eviction | An opaque receipt |
| `activate(ctx, receipt)` | In place of `runtime.run()` | `ActivationResult` |

Preparation steps form a deterministic DAG with globally unique names and
explicit `requires`. Completed steps are compensated in reverse order when a
later step fails. `PreparationStep` carries `name`, `action`, `requires`, and an
optional `cleanup`.

`LaunchAssetPolicy` declines shared work: `images_by_node`, `distribute_images`,
`prepare_model`, `run_builder`, `prepare_runtime`, `probe_images`,
`sync_tuning`, `clear_page_cache`. Everything not declined still runs, so a
strategy inherits distribution, placement, and preflight instead of
reimplementing them. `PreparedExecution` carries `strategy`, `assets`, `state`,
`receipts`, and `host_hardware`. `ActivationContext` exposes the final
`cluster_id`, `hosts`, images, cache dir, serve port/command, and communication
maps; `ActivationResult` returns `rc`, `runtime_info`, and optional
`startup_observation`.

Three boundaries stay core-owned:

- The replacement barrier fires after plugin preparation, normal image/model
  preparation, and `prepare_activation`. A strategy never decides when the
  deployment it replaces is torn down.
- The launcher records job metadata with the normal identity. `save_job_metadata`
  rewrites the file wholesale, so an omitted field is an erasure.
- `RunOptions.strategy_options` is per-invocation, not workload identity.

A strategy that supports recipe `env` templates must declare
`owns_env_templates = True` and own preparation, rendering, and restore
compatibility. `build_preparation(ctx)` contributes preparation-only build
inputs; it must honor dry-run, recipe trust, and fingerprint participation.

## Registries and foreign formats

A plugin whose recipes are inert without it declares its registry in code. This
typed function fragment belongs in the plugin module:

```python
"""Copyable function fragment; keep this in the plugin entry module."""

# pyright-checked copyable fragment
from sparkrun.plugins import RegistryEntry, register_default_registry
from scitrera_app_framework import Variables


def register(v: Variables) -> None:
    register_default_registry(
        RegistryEntry(
            name="example",
            url="https://github.com/example/recipes.git",
            subpath="recipes",
            description="Example recipes",
            visible=False,
        ),
        owner="example",
    )
```

Declarations are an overlay: they are merged into the loaded registry list and
never written to `registries.yaml`. The user's file wins on a name collision.
`registry disable`/`trust`/`untrust` materializes a declaration into the user's
file; `registry remove` leaves a tombstone. Registration does no I/O, so declare
names in code and let the user run `sparkrun registry update` to clone.

Trust depends on origin. An in-tree plugin may ship `trusted=True`; an
out-of-tree declaration is forced untrusted and the user grants it with
`sparkrun registry trust <name>`. A plugin cannot claim its own trust, and
`official` cannot be redefined.

For another launcher's manifests, register a `RecipeFormat`. This is pseudocode
for the adapter-specific discovery and translation logic, not copyable Python:

```text
from sparkrun.plugins import RecipeFormat, RegistryEntry, register_default_registry, register_recipe_format


def iter_files(root):
    return sorted(root.glob("*/manifest.yaml"))


def name_of(path, root):
    return path.parent.name


def load(path, *, registry_manager, offline, allow_local=True):
    return translate_to_v2(path)


def register(v):
    register_recipe_format(
        RecipeFormat(name="example", owner=__name__, iter_files=iter_files, name_of=name_of, load=load)
    )
    register_default_registry(
        RegistryEntry(name="example", url="https://github.com/example/catalog.git", subpath="/", format="example"),
        owner=__name__,
    )
```

`iter_files(root)` returns every launchable manifest under a registry directory,
sorted. `name_of(path, root)` returns the name typed after `@registry/`; `root` is
`None` for a manifest loaded by path outside a registry. `load(...)` returns v2
recipe data, must not touch the network when `offline=True`, and must not read
files beside `path` when `allow_local=False`. Optional `claims(path, data)`
recognizes a manifest by content for a local path outside every registry.
Listing and lookup share `iter_files`, so anything listed is runnable. An
unregistered format is inert, and a foreign format is honored only for a trusted
registry or a plugin-declared one.

## Trust, features, and configuration

Plugin trust has tiers:

- **In-tree** plugins ship through sparkrun's review and release gate and may
  declare trusted registries.
- **Out-of-tree** plugins run with your permissions after an install-time trust
  prompt; `--trust` is explicit noninteractive consent, not `--yes`.
- **Installed entry-point** integrations are selected by
  `plugins.enabled`, an application profile, or `sparkrun plugins enable`; their
  registries remain external-tier until trusted.

Configuration layers:

```yaml
plugins:
  enabled:
    example-plugin: true
  paths:
    - /opt/sparkrun-plugins
features:
  example.enabled: true
```

`plugins.enabled` wins over application integration defaults and legacy
integration flags. `plugins.paths` requires `core.external_plugins`; installed
entry-point plugins do not. Per-plugin settings live under `plugins.<name>` and
are read with the plugin's settings namespace, not in a recipe: a recipe is
portable, site policy is not.

Kill switches: `SPARKRUN_NO_EXTERNAL_PLUGINS`, `SPARKRUN_NO_INSTALLED_PLUGINS`,
and `SPARKRUN_NO_DEFAULT_PLUGINS`.

## Test and verify

```bash
sparkrun plugins list --all --json
sparkrun plugins enable example-plugin
sparkrun plugins list --check
sparkrun plugins disable example-plugin
sparkrun recipe validate my-recipe.yaml --strict
sparkrun run my-recipe.yaml --cluster mylab --dry-run
```

- `plugins list --json` reports source, version, enabled/loaded, compatibility,
  features, and failures without importing plugin code.
- `plugins list --check` loads managed plugins in a bounded child process.
- Dry runs must exercise every mutating command before a real launch.
- Unit tests can use `FunctionalRecipeItemHandler` for terse handlers and an
  explicit `paths=` argument to bypass the external-plugin environment guard.
  Test isolation may call `sparkrun.core.recipe_items.unregister_recipe_item`
  (internal, tests only; not part of the plugin import surface).
- Reference tests: `tests/test_recipe_plugin_items.py`,
  `tests/test_execution_strategies.py`, and the wheel fixture test.
- Test with the plugin absent, disabled, and loaded; an unregistered key must
  retain the runtime-config behavior while serialized items stay preserved.

## Pitfalls

1. Registration must be local and inexpensive; it runs on help, completion, and
   inventory paths.
2. One conflict rule for every registry: equal is a no-op, anything else raises
   and rolls back the plugin's load.
3. `SPARKRUN_PLUGIN_API_VERSION` must be a literal integer, not a computed value.
4. An unknown top-level recipe key becomes `runtime_config` and may reach the
   engine as a flag; claim the key to prevent that.
5. One execution strategy per recipe; arbitration is an error, not precedence.
6. Opt out of the fingerprint only for passive annotations.
7. Do not reimplement distribution, placement, or preflight; decline only what
   the strategy actually owns.
8. Recipe `env` templates are core-prepared; a custom launch path must call
   `render_launch_env(...)` or declare `owns_env_templates`.
9. Lifecycle hooks are planners: return proposed changes, do not write config or
   start processes.
10. Never import `sparkrun.core.*` or other internals; use `sparkrun.plugins` and
    the documented `sparkrun.api` seams.
11. An executor's `resolve_target` must be pure: no side effects, no probes.
12. Image providers and model providers are separate registrations; one does not
    imply the other.
13. A model provider must move exactly `target.required_files`, respect offline
    and cancellation, and never delete unrelated shared blobs.
14. Benchmark integrations must not log or persist API keys; credentials arrive
    through the documented accessors.
15. Directory plugins with duplicate top-level module names are rejected; use
    distinct names or separate processes.
16. Managed installs reject symlinks, submodules, compiled payloads, dynamic
    metadata, and unsatisfied dependencies.

<Tool_Usage>
Run these commands with the `bash` tool. The default 300-second deadline is too
short for many downloads, readiness waits, and benchmarks; raise `timeout` within
the host's configured maximum (documented maximum: 3600 seconds). `async: true`
requests a background job when the host's `async.enabled` setting permits it; it
does not change the deadline. `timeout: 0` disables the tool-imposed deadline but
does not itself request background execution. Background jobs continue until
completion, cancellation, or their deadline (unless `timeout: 0` disables it).
Cancellation or timeout can stop the local CLI while leaving remote workloads
running; inspect status before retrying.
Interactive prompts (`setup wizard`, SSH/sudo, trust confirmations, Arena login)
need the user's own terminal. Never claim a timed-out launch was rolled back.
</Tool_Usage>

## References

- [Plugin guide](https://github.com/spark-arena/sparkrun/blob/main/docs/PLUGINS.md)
- [Executor guide](https://github.com/spark-arena/sparkrun/blob/main/docs/EXECUTORS.md)
- [Multi-platform guide](https://github.com/spark-arena/sparkrun/blob/main/docs/MULTIPLATFORM.md)
- [Security and trust model](https://github.com/spark-arena/sparkrun/blob/main/docs/SECURITY.md)
- [Plugin import surface](https://github.com/spark-arena/sparkrun/blob/main/src/sparkrun/plugins/__init__.py)
- Fixture package: `tests/fixtures/application_profiles/plugin/`
- The `configuration` skill for install/enable commands and feature flags.