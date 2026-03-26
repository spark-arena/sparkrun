# /sparkrun:benchmark

Run benchmarks against an inference workload.

## Usage

```
/sparkrun:benchmark <recipe> [options]
```

## Examples

```
/sparkrun:benchmark qwen3-1.7b-sglang --tp 1
/sparkrun:benchmark qwen3-1.7b-sglang --tp 2 --profile spark-arena-v1
/sparkrun:benchmark qwen3-1.7b-sglang --skip-run --tp 1
/sparkrun:benchmark qwen3-1.7b-sglang --no-stop --cluster mylab
/sparkrun:benchmark qwen3-1.7b-sglang -b depth=0,2048,4096 -b tg=32,128
```

## Behavior

When this command is invoked:

1. If no recipe is specified, run `sparkrun list` to show available recipes and ask the user to pick one.
2. Determine the target hosts (same as `/sparkrun:run`).
3. Run the full benchmark flow:

```bash
# Full flow: launch inference -> benchmark -> stop
sparkrun benchmark <recipe> [options]
```

The benchmark command handles the complete lifecycle:
- **Step 1/3**: Launches the inference server (unless `--skip-run`)
- **Step 2/3**: Runs the benchmark against the server
- **Step 3/3**: Stops the inference server (unless `--no-stop`)

4. Results are saved to YAML, JSON, and CSV files.

## Common Options

| Option | Description |
|--------|-------------|
| `--hosts, -H` | Comma-separated host list |
| `--cluster` | Use a saved cluster |
| `--tp N` | Tensor parallelism (= number of nodes) |
| `--pp N` | Pipeline parallelism |
| `--port N` | Override serve port |
| `--gpu-mem F` | GPU memory utilization (0.0-1.0) |
| `--max-model-len N` | Override max context length |
| `--image` | Override container image |
| `--profile` | Benchmark profile name or file path |
| `--framework` | Override benchmarking framework (default: llama-benchy) |
| `-o KEY=VALUE` | Override recipe default (repeatable) |
| `-b KEY=VALUE` | Override benchmark arg (repeatable) |
| `--exit-on-first-fail` | Abort on first failure, skip saving results (default: on) |
| `--no-stop` | Keep inference running after benchmarking |
| `--skip-run` | Skip launching inference (benchmark existing instance) |
| `--sync-tuning` | Sync tuning configs from registries before benchmarking |
| `--timeout` | Benchmark timeout in seconds (default: 14400) |
| `--out, --output` | Output file for results YAML |
| `--dry-run` | Show what would be done |

## Benchmark Profiles

Browse available profiles from registries:

```bash
sparkrun registry list-benchmark-profiles
sparkrun registry show-benchmark-profile <name>
```

## Notes

- The benchmark command auto-detects available ports to avoid collisions with running instances
- Use `--skip-run` to benchmark an already-running inference server
- Use `--no-stop` to keep the inference server running after the benchmark completes
- Use `-b key=value` for benchmark-specific args; use `-o key=value` for recipe overrides
- Results are saved as YAML, JSON, and CSV when available
- Profiles support `@registry/name` syntax for explicit registry selection
