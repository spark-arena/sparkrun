---
description: Observe cluster metrics using a cancellable stream
argument-hint: "[--cluster name] [options]"
---

# /sparkrun:monitor

Use the workloads skill. For a finite state check, prefer `sparkrun status`. If metrics
are requested, choose a supported cancellable background session or an explicit
short timeout:

```bash
sparkrun cluster monitor --cluster <name> --simple --interval 5
sparkrun cluster monitor --cluster <name> --json --interval 5
```

Both forms stream indefinitely; neither is one-shot. Default TUI mode requires
an interactive terminal. Stop the local monitor after collecting the requested
samples. Report observed CPU/RAM/GPU metrics and the sampling period; do not
interpret termination of the monitor as termination of inference workloads.

User request: $ARGUMENTS
