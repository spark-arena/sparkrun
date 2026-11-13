---
description: Manage the selected inference gateway and its models
argument-hint: "<action> [options]"
---

# /sparkrun:proxy

Use the workloads/configuration skills. LiteLLM is bundled and the default on all channels.
SparkRoute must be installed as an external Sparkrun plugin. Inspect selection
and feature state before changing providers:

```bash
sparkrun plugins list
sparkrun setup features
sparkrun proxy status --json
sparkrun setup features enable gateway.litellm
sparkrun proxy start --host 127.0.0.1 --cluster <name>
sparkrun proxy models
sparkrun proxy models --refresh
sparkrun proxy load <recipe> --cluster <name>
sparkrun proxy unload <recipe> --cluster <name>
sparkrun proxy sync
sparkrun proxy stop
```

Run only the requested actions, not this entire example sequence. Explicitly
enabling `gateway.<provider>` selects it; enabling/installing a plugin alone does
not. Both provider flags may stay on. Use `proxy start --restart` to apply a
switch to a running gateway. Disabling shared `gateway` blocks new activity but
does not stop an existing service; status/stop remain available for recovery.

Model queries resolve gateway credentials (including `LITELLM_MASTER_KEY` for
LiteLLM). Do not invent a `models --master-key` flag or print tokens. `proxy ui`
and `proxy admin-token` depend on provider capabilities. Model/alias updates may
restart LiteLLM; report errors even when settings were saved successfully.
`proxy systemd` manages Linux services; sudo/login prompts need a user terminal.

User request: $ARGUMENTS
