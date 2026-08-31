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
