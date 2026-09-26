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
  An API absent from a shorthand profile is not supported by that implementation.
- Request structure and routing fields such as `model`, `messages`, `input`,
  `stream`, `system`, and `tools` cannot be overridden. Selectors use 1–64 ASCII
  letters/digits, dots, underscores, or hyphens, starting with a letter/digit.
  At most 64 profiles and 256 KiB of settings are accepted. Malformed JSON values,
  non-finite numbers, unknown operations, and ambiguous names are rejected.

Profiles are public children of a virtual model. Each deployment keeps its own
implementation, so two recipes can both define `low` with different parameters.
A profile inherits parent routing pools, weights, aliases, limits, and policies;
only deployments implementing the requested profile/API are eligible. Missing
support never silently falls back to the unprofiled model. Conflicting definitions
for the same deployment identity remain an error.

The shorthand above means incoming API parameters, before translation. For a
profile implemented by runtime-native parameters after translation, use:

```yaml
sparkroute:
  request_profiles:
    low:
      supported_operations: [responses, chat_completions]
      upstream_overrides:
        chat_completions:
          chat_template_kwargs:
            enable_thinking: false
```

`ingress_overrides` and `upstream_overrides` may be combined. Upstream keys name
operations, not protocol families. `supported_operations` declares incoming APIs;
without it, the ingress override keys define support. An explicit supported
operation with no overrides is an intentional no-op. Contradictory controls at
both stages are rejected when their overlap is known. State fields `store` and
`background` are protected in these deployment implementations.

Explicit bindings follow their recipe on `proxy sync`. Discovery-only workloads
use saved launch recipe settings and preserve job identity, so different runtimes
serving the same model can expose different implementations. Without reliable job
identity, only the base model is exposed. Editing the source YAML alone does not
change a discovery-only job's saved state. The historical aggregate deployment
remains available to saved operator references; generated profile routes use the
job-specific targets. Excluding the aggregate still excludes that model's jobs.

Recipe-picker deployments automatically follow their resolved recipe source at
startup, after configuration/catalog updates, and every 60 seconds (gateway flag
`-sparkrun-profile-refresh-interval`). This reads the local catalog; registry
network refresh policy is unchanged. Uploaded YAML is an immutable snapshot until
explicitly replaced. A recipe change that changes workload identity requires a
deployment update before its profiles can apply to the old runtime.

In **Virtual Models / Aliases → Request profiles**, choose **Use recipe profile**
and edit common or deployment-specific overrides. Recipe values and operator
patches are stored separately. A recipe update changes inherited values while
preserving your patches. **Reset to inherited** removes a patch for the selected
scope, stage, and API. Suppression uses JSON pointers such as `/temperature` and
removes an inherited override, allowing caller values through; JSON null remains
a literal value. New recipe selectors become available without automatically
publishing new public model names.

The gateway persists refreshed sources in its read-only `recipe_profiles` managed
set. Source errors retain last-known-good values and appear in the editor.
Confirmed selector removal makes that implementation unavailable; operator patches
remain available if the selector returns. Profile digests are separate from
workload fingerprints, so profile-only edits do not restart workloads.

Existing flat `request_overrides` virtual models keep their historical behavior,
including no override for an absent API. **Review migration** previews the routing
change before linking one to its parent. Existing values become operator patches;
select **Follow deployment recipes** to subscribe a recipe-backed deployment.
Generated entries remain read-only. Public model/alias collisions remain errors.

This implementation requires a gateway built from the matching bridge-v5 source.
The currently published gateway pin is not a release of this change; use a local
build via `SPARKROUTE_BINARY` until coordinated gateway/plugin releases are made.

These settings stay outside the workload fingerprint and binding identity.
The plugin declares `sparkroute` through `register_recipe_item` with
`affects_fingerprint=False`; core carries generic `plugin_items` data without
knowing this schema. The plugin owns parsing, validation, canonical export and
bridge projection. It must be enabled to recognize the key in fresh recipe YAML.
Saved plugin items and their fingerprint policy survive serialization/export
when the plugin is unavailable.

## Application profile API compatibility

The plugin requires Sparkrun `>=0.4,<0.5` and declares module contract
`SPARKRUN_PLUGIN_API_VERSION = 1`. Workers initialize through the host application
API directly; hosts without that API are unsupported.

The source manifest declares `application_profile_api = 1`. A compatible host records
that declaration in verified vendor metadata before enabling the integration for
an alternate application profile. Cache roots follow the active profile;
`SPARKROUTE_BINARY` is the only binary override across profiles. Profile-prefixed
and historical override names are not supported. Gateway and worker children
preserve the profile's installed reference and
same-controller config file. An alternate callback must exist in the current
Python environment. This branch requires Sparkrun `>=0.4,<0.5`.
The `sparkrun` protocol/provider identifiers and `sparkroute` extension IDs remain
shared. See the repository README for integration and testing details.

## Gateway API contract

`AdminError` implements `sparkrun.proxy.contracts.GatewayOperationError` directly,
including transport, authentication and exhausted revision-conflict failures.
Its status, code, retryability and exception causes remain available to callers.
`query_models()` returns a tuple of `ProxyModel` records. Failed or malformed
status responses raise `GatewayQueryError`; a successful empty response returns
an empty tuple. No provider-specific host adapter is required.
