<!--
SPDX-FileCopyrightText: 2026 Scitrera LLC
SPDX-FileCopyrightText: 2026 Fox Engine Ltd
SPDX-License-Identifier: AGPL-3.0-only
-->

# Vendored SparkRoute integration

Canonical source: https://github.com/sparksq/sparkrun-sparkroute-plugin

Imported as `sparkrun.plugins.sparkroute`; gated by `gateway.sparkroute`.
This package is AGPL-3.0-only with the sparkrun combination permission in
`LICENSE_EXCEPTION`. Both license notices must accompany redistribution.

sparkrun's packaged `VENDORED.toml` identifies the immutable upstream source
and content hashes. Develop changes in the canonical repository and update
the host vendor pin rather than editing this copy.

## Recipe defaults

Recipes may carry an optional top-level `sparkroute` block:

```yaml
model: example/model
runtime: vllm
container: example/vllm:latest
defaults:
  served_model_name: coding
sparkroute:
  capabilities: [vision]
  request_profiles:
    low:
      chat_completions:
        temperature: 0.2
        chat_template_kwargs:
          enable_thinking: false
      responses:
        reasoning:
          effort: low
    xhigh:
      chat_completions:
        reasoning_effort: xhigh
```

Parameter values are examples: choose values supported by the particular model
and runtime. YAML mappings (or inline JSON objects) are accepted; JSON strings
containing encoded objects are not. `request_profiles` maps selector suffixes to
API operations, then parameter objects, matching SparkRoute's `request_overrides`.

- `capabilities` adds declarations to the deployment, alongside existing top-level
  recipe capabilities and native runtime APIs. It does not require vision on every
  request. The UI displays the shared Vision / Files choices; other supported
  capability names and `x-` extensions remain available in YAML/JSON.
- The example creates `coding`, `coding:low`, and `coding:xhigh`. All use the same
  deployment, cluster, cold-start wait, and idle policy. Public aliases get the
  same suffixes (for example `code:low`). Profiles appear in the parent model's
  Request profiles table, not as additional deployment entries.
- Overrides are keyed by ingress API, such as `chat_completions`, `responses`, or
  `messages`. They override caller parameters before translation. Nested objects
  merge recursively; scalar/array/null values replace the corresponding value.
  An API absent from a profile receives no override from that profile.
- Request structure and routing fields such as `model`, `messages`, `input`,
  `stream`, `system`, and `tools` cannot be overridden. Selectors use 1–64 ASCII
  letters/digits, dots, underscores, or hyphens, starting with a letter/digit.
  At most 64 profiles and 256 KiB of settings are accepted. Malformed JSON values,
  non-finite numbers, unknown operations, and ambiguous names are rejected.

Explicit recipe bindings follow their recipe on subsequent `proxy sync`. Normal
`sparkrun run` and `sparkrun proxy load` workloads are discovery-only and carry
the settings saved at launch;
editing the source file alone does not alter that saved recipe state. Multiple
recipes serving the same public model may share an identical profile, but its
routing pool only includes recipes declaring it. Conflicting definitions fail
reconciliation. A discovery-only model aggregates its jobs and exposes only
identical profiles common to every candidate. Capability declarations likewise
intersect across discovery candidates.

The on-demand UI previews the recipe defaults and imports profiles under the
chosen public model name when adding it to a draft, including when reusing a
deployment. They then become operator configuration: edits in the profile table
are preserved. Applying recipe/lifecycle edits to an existing deployment does
not refresh or overwrite its profiles. Generated profiles remain read-only.
A generated profile cannot silently replace an operator model/profile with the
same name: resolve the reported collision by renaming or removing one definition.

These settings stay outside the workload fingerprint and binding identity.
The plugin declares `sparkroute` through `register_recipe_item` with
`affects_fingerprint=False`; core carries generic `plugin_items` data without
knowing this schema. The plugin owns parsing, validation, canonical export and
bridge projection. It must be enabled to recognize the key in fresh recipe YAML.
Saved plugin items and their fingerprint policy survive serialization/export
when the plugin is unavailable.
