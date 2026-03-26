# /sparkrun:status

Check the status of running sparkrun inference workloads.

## Usage

```
/sparkrun:status [options]
```

## Examples

```
/sparkrun:status
/sparkrun:status --cluster mylab
/sparkrun:status --hosts 192.168.11.13,192.168.11.14
```

## Behavior

When this command is invoked:

1. Assume `sparkrun` is in PATH and run `sparkrun cluster status` directly:

```bash
# Preferred -- uses default cluster
sparkrun cluster status

# With explicit targets
sparkrun cluster status --cluster <name>
sparkrun cluster status --hosts <ip1>,<ip2>,...
```

If sparkrun is not found, suggest installation via the setup skill.

2. Report the output to the user. The status command shows:
   - Grouped containers by job (with recipe name if cached)
   - Container role, host, status, and image
   - Ready-to-use `sparkrun logs` and `sparkrun stop` commands for each job
   - **Pending operations**: any in-progress model downloads or container image distributions
   - **Idle hosts**: hosts with no sparkrun containers

3. If the user wants to act on a specific job (view logs, stop it), use the commands shown in the status output.

4. For checking if a specific job is running (with optional health check), use:

```bash
sparkrun cluster check-job <recipe_or_cluster_id> --cluster <name>
sparkrun cluster check-job <recipe_or_cluster_id> --check-http-models  # also verify /v1/models
sparkrun cluster check-job <recipe_or_cluster_id> --json               # JSON output
```

## Notes

- Uses the default cluster if no `--hosts` or `--cluster` is specified
- `sparkrun status` is a top-level alias for `sparkrun cluster status` — they are identical. Only run one.
- Container names follow the pattern `sparkrun_{hash}_{role}`
- Pending operations (model downloads, image distributions) are tracked via lock files and shown in status output
- `check-job` exits with code 0 if running, 1 if not — useful for scripting
