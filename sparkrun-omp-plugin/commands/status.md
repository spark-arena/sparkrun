---
description: Inspect current workload placement and health
argument-hint: "[--cluster name] [options]"
---

# /sparkrun:status

Use the workloads skill. `status` aliases `cluster status`; run only one with the
requested host/cluster scope:

```bash
sparkrun status --cluster <name>
sparkrun cluster check-job <workload_id> --json
sparkrun cluster check-job <workload_id> --check-http-models --json
```

Report actual placement, workload IDs, runtime/container state, pending operations,
and idle hosts as available. Reuse the exact log/stop commands printed by status.
A workload ID (`workload_id` in JSON) is not a saved host cluster.
Do not guess IDs or container naming. `check-job` returns 0 for running and 1 for
not running; distinguish configuration/transport errors from an absent workload.
Container presence and `/v1/models` are narrower than inference readiness.

User request: $ARGUMENTS
