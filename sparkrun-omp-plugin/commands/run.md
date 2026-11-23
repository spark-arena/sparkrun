---
description: Launch an inference workload with readiness-aware defaults
argument-hint: "<recipe> [options]"
---

# /sparkrun:run

Use the workloads skill for the Sparkrun 0.4.0 reference. Search for an unspecified
recipe with `sparkrun recipe search <query>`; otherwise inspect the supplied
recipe. Preserve requested hosts/cluster and overrides. Preview with `--dry-run`
when useful, then launch within the user's authorized scope:

```bash
sparkrun run <recipe> --cluster <name> --no-follow
```

`--no-follow` suppresses model logs but **still waits for readiness**. Allow time
for downloads/startup. If the user wants submission only, use `--no-ready-wait`
instead; preparation still runs, and report readiness as unchecked. Do not add
it to recipes with readiness-dependent hooks. `--ensure` reuses a matching
workload and still waits. Avoid `--follow`/`--foreground` in a finite tool call.

Use the reported placement/workload ID, not a guessed hash or first cluster host.
TP × PP describes GPU demand on one-GPU Sparks; the scheduler selects hosts.

```bash
sparkrun status --cluster <name>
sparkrun cluster check-job <workload_id> --check-http-models --json
sparkrun logs <workload_id> --all-sources --tail 200
```

A container or models endpoint check is not a full inference readiness probe.
For troubleshooting, `run --collect-diagnostics <new_file>` records DEBUG and
structured feedback without extra console verbosity. Report the endpoint, actual
readiness outcome, and exact stop/log commands. Do not treat a tool timeout as
proof the remote workload stopped.

User request: $ARGUMENTS
