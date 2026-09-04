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

Parsed items are read with `recipe.plugin_item("snapshot")`.

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
- **Items participate in `derive_recipe_fingerprint`**, using the handler's
  canonical export. They are declared configuration; omitting them would make
  two recipes with different extension policy share every cache and provenance
  record keyed off that digest. The fingerprint part is appended only when an
  item is present, so recipes predating the seam hash byte-identically.

Note the contrast with `capabilities:` / `unsupported_capabilities:`, which are
core keys parsed as real attributes *specifically* to stay out of the
fingerprint: describing what a deployment can do must not change what it is.
A plugin item is the opposite — it changes how the workload is produced.

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
