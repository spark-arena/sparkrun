# /sparkrun:stop

Stop a running inference workload.

## Usage

```
/sparkrun:stop [target] [options]
```

## Examples

```
/sparkrun:stop glm-4.7-flash-awq
/sparkrun:stop glm-4.7-flash-awq --tp 2
/sparkrun:stop glm-4.7-flash-awq --cluster mylab
/sparkrun:stop e5f6a7b8
/sparkrun:stop sparkrun_e5f6a7b8
/sparkrun:stop --all --cluster mylab
```

## Behavior

When this command is invoked:

1. If no target is specified and `--all` is not used, run `sparkrun cluster status` first to see what's running and ask the user which job to stop.
2. Stop the workload:

```bash
# By recipe name
sparkrun stop <recipe> [options]

# By cluster ID (from sparkrun status output)
sparkrun stop <cluster_id> [options]

# Stop all sparkrun containers
sparkrun stop --all --cluster <name>
```

3. After stopping, optionally run `sparkrun status` to confirm containers are gone.

## Common Options

| Option | Description |
|--------|-------------|
| `--hosts, -H` | Comma-separated host list |
| `--cluster` | Use a saved cluster |
| `--tp N` | Must match the value used in `run` |
| `--port N` | Must match the port override used in `run` |
| `--served-model-name` | Must match the served model name used in `run` |
| `--all, -a` | Stop all sparkrun containers (no target needed) |
| `--dry-run` | Show what would be done |

## Notes

- TARGET can be a recipe name OR a cluster ID (hex string from `sparkrun status` output)
- The stop command identifies containers by a hash derived from runtime + model + sorted hosts + overrides
- If `--tp`, `--port`, or `--served-model-name` was used during `run`, they must also be passed to `stop`
- Use `sparkrun status` to see the exact stop commands for running jobs
- Use `--all` to discover and stop all sparkrun containers without specifying a target
