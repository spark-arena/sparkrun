# /sparkrun:proxy

Manage the LiteLLM-based inference proxy gateway.

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

### Stop the proxy

```bash
sparkrun proxy stop
```

## Notes

- The proxy uses LiteLLM under the hood (installed via `uvx litellm`)
- Auto-discover periodically scans for new/removed endpoints (configurable interval)
- The proxy API is available at `http://localhost:<port>/v1` (default port: 4000)
- Aliases let clients use friendly names instead of full model paths
- `proxy load` auto-selects ports to avoid conflicts with running instances
