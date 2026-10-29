---
description: Stop a selected inference workload
argument-hint: "[target|--all] [options]"
---

# /sparkrun:stop

Use the workloads skill. If no target is supplied, inspect status and identify the
user's intended workload before stopping it. Prefer the recorded workload ID or
exact stop command printed by status:

```bash
sparkrun stop <workload_id> --dry-run
sparkrun stop <workload_id>
sparkrun stop <recipe> --cluster <name> --tp 2
sparkrun stop --all --cluster <name>
```

Only use `--all` for an explicitly authorized whole-scope stop. Recipe-based targets
must preserve relevant launch overrides (`--tp`, `--port`, `--served-model-name`);
use a recorded workload ID to avoid ambiguity. Do not derive hashes or assume
all configured hosts belong to that workload. After stopping, report failures or
remaining workloads from status instead of claiming unconditional success.

User request: $ARGUMENTS
