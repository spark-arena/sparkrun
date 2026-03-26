# /sparkrun:monitor

Live-monitor CPU, RAM, and GPU metrics across cluster hosts.

## Usage

```
/sparkrun:monitor [options]
```

## Examples

```
/sparkrun:monitor --cluster mylab
/sparkrun:monitor --hosts 192.168.11.13,192.168.11.14
/sparkrun:monitor --cluster mylab --simple
/sparkrun:monitor --cluster mylab --json
```

## Behavior

When this command is invoked:

1. Run the cluster monitor:

```bash
# Interactive TUI (default)
sparkrun cluster monitor --cluster <name>

# Plain-text output (better for agent context)
sparkrun cluster monitor --cluster <name> --simple

# JSON streaming (for automation)
sparkrun cluster monitor --cluster <name> --json
```

**IMPORTANT:** When running from an agent context, prefer `--simple` or `--json` mode since the TUI requires an interactive terminal.

2. Report metrics to the user: CPU usage, RAM usage, GPU utilization, GPU memory, and running containers.

## Common Options

| Option | Description |
|--------|-------------|
| `--hosts, -H` | Comma-separated host list |
| `--cluster` | Use a saved cluster |
| `--interval N` | Sampling interval in seconds (default: 2) |
| `--simple` | Use plain-text output instead of TUI |
| `--json` | Stream updates as newline-delimited JSON |
| `--dry-run` | Show what would be done |

## Notes

- The default TUI mode requires an interactive terminal — use `--simple` or `--json` when running non-interactively
- Press `q` (TUI) or Ctrl-C to stop monitoring
- Metrics include CPU, RAM, GPU utilization, GPU memory, and temperature
