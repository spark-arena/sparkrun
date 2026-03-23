# /sparkrun:run

Run an inference workload on DGX Spark using a sparkrun recipe.

## Usage

```
/sparkrun:run <recipe> [options]
```

## Examples

```
/sparkrun:run qwen3-1.7b-vllm
/sparkrun:run glm-4.7-flash-awq --tp 2
/sparkrun:run qwen3-1.7b-llama-cpp --tp 1
/sparkrun:run @spark-arena/some-recipe --cluster mylab
```

## Behavior

When this command is invoked:

1. If no recipe is specified, run `sparkrun list` to show available recipes. If the user describes what they're looking for, use `sparkrun recipe search <query>` to find matching recipes. Ask the user to pick one.
2. Determine the target hosts:
   - If the user specifies `--hosts`, `--cluster`, or `--tp`, use those.
   - Otherwise check if a default cluster is configured (`sparkrun cluster default`).
   - If no hosts can be resolved, ask the user for hosts or to create a cluster first.
3. Optionally run `sparkrun show <recipe> --tp <N>` to preview VRAM estimation before launching.
4. Run the workload:

```bash
sparkrun run <recipe> [options] --no-follow
```

**CRITICAL: Always use `--no-follow`** to avoid blocking on log streaming. After launch, use `sparkrun cluster status` or `sparkrun logs` separately.

5. After launching, run `sparkrun cluster status` to confirm containers are running.

## Common Options

| Option | Description |
|--------|-------------|
| `--hosts, -H` | Comma-separated host list |
| `--cluster` | Use a saved cluster |
| `--tp N` | Tensor parallelism (= number of nodes) |
| `--pp N` | Pipeline parallelism |
| `--port N` | Override serve port |
| `--gpu-mem F` | GPU memory utilization (0.0-1.0) |
| `--max-model-len N` | Override maximum model context length |
| `--served-model-name` | Override the served model name |
| `--image` | Override container image |
| `-o KEY=VALUE` | Override any recipe default (repeatable) |
| `--ensure` | Only launch if not already running; exit 0 if already up |
| `--no-rm` | Don't auto-remove containers on exit |
| `--restart POLICY` | Docker restart policy (no, always, unless-stopped, on-failure[:N]) |
| `--transfer-mode MODE` | Resource transfer mode (auto, local, push, delegated) |
| `--dry-run` | Show what would be done |

## Notes

- Each DGX Spark has 1 GPU, so `--tp N` means N hosts
- `--solo` is deprecated; use `--tp 1` instead
- `sparkrun stop` and `sparkrun logs` need the same `--hosts`/`--cluster`/`--tp` flags as `run`, or can use the cluster ID from status output
- Ctrl+C while following logs detaches safely — it never kills the inference job
- `--ensure` is useful for idempotent scripts — it checks if the job is already running before launching
- Recipe names support `@registry/name` syntax for explicit registry selection
