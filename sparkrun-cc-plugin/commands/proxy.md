# /sparkrun:proxy

Manage the inference proxy through a pluggable gateway, such as LiteLLM or SparkRoute.

## Usage

```
/sparkrun:proxy <action> [options]
```

## Examples

```
/sparkrun:proxy start --cluster mylab
/sparkrun:proxy status
/sparkrun:proxy load qwen3-1.7b-vllm --cluster mylab
/sparkrun:proxy models
/sparkrun:proxy stop
```

## Behavior

The proxy provides a single unified OpenAI-compatible API in front of multiple inference endpoints.

### Start the proxy

```bash
sparkrun proxy start
sparkrun proxy start --cluster mylab --port 4000
sparkrun proxy start --foreground
sparkrun proxy start --gateway sparkroute --host 127.0.0.1
```

Auto-discovers running inference endpoints and registers them.

### Check proxy status

```bash
sparkrun proxy status
```

Shows running state, registered models, and auto-discover status.

### List registered models

```bash
sparkrun proxy models
sparkrun proxy models --refresh   # re-discover and sync
```

### Load a model (launch + register)

```bash
sparkrun proxy load <recipe> --cluster <name>
sparkrun proxy load <recipe> --tp 1 --gpu-mem 0.8
```

Launches inference via `sparkrun run` and registers the endpoint with the proxy.

### Unload a model (stop + deregister)

```bash
sparkrun proxy unload <recipe> --cluster <name>
```

### Manage model aliases

```bash
sparkrun proxy alias add my-model "Qwen/Qwen3-1.7B"
sparkrun proxy alias remove my-model
sparkrun proxy alias list
```

### Run the proxy as a systemd service

```bash
sparkrun proxy systemd install --now          # system unit, starts at boot (sudo once)
sparkrun proxy systemd install --user --now   # user unit, no sudo (needs lingering for boot start)
sparkrun proxy systemd status
sparkrun proxy systemd remove
```

The unit runs `proxy start --foreground` and reads `proxy.yaml` each time it starts. Once installed, `proxy start` / `proxy stop` go through the unit. Installing a system unit prompts for a sudo password, so ask the user to run it themselves (`! sparkrun proxy systemd install --now`).

### Stop the proxy

```bash
sparkrun proxy stop
```

## Notes

- `--gateway <name>` selects and persists an enabled implementation. Built-in Sparkrun defaults to LiteLLM on stable/beta and SparkRoute on alpha; a saved gateway pin takes precedence.
- Inspect availability with `sparkrun setup features list`; enable a gateway with `sparkrun setup features enable gateway.<name>`. Enabling alone does not change an existing selection.
- `sparkrun proxy ui` and `sparkrun proxy admin-token get|set|clear` require gateway support (currently SparkRoute).
- Model and alias updates use the running gateway; a restart is implementation-specific.
- See [the proxy guide](../../docs/PROXY.md) for configuration, authentication, and gateway selection.
- Auto-discover periodically scans for new/removed endpoints (configurable interval)
- The proxy API is available at `http://localhost:<port>/v1` (default port: 4000)
- Aliases let clients use friendly names instead of full model paths
- `proxy load` auto-selects ports to avoid conflicts with running instances
