# sparkrun Plugin for Claude Code

AI-assisted inference on NVIDIA DGX Spark -- run, manage, and stop LLM workloads with Claude.

## What It Does

This plugin teaches Claude Code how to use [sparkrun](https://github.com/scitrera/sparkrun) to manage LLM inference
workloads on NVIDIA DGX Spark systems. It provides:

- **Slash Commands** -- Quick actions for running, stopping, benchmarking, monitoring, and managing inference jobs
- **Skills** -- Detailed reference that Claude uses automatically when working with sparkrun

## Installation

### From the Marketplace

```bash
# Add the marketplace (one-time setup)
claude plugin marketplace add scitrera/sparkrun

# Install the plugin
claude plugin install sparkrun@sparkrun
```

### Manual Installation

Clone or copy the plugin directory:

```bash
# Global (available in all projects)
cp -r sparkrun-cc-plugin ~/.claude/plugins/sparkrun

# Or project-local
cp -r sparkrun-cc-plugin .claude/plugins/sparkrun
```

### Local Development

```bash
claude --plugin-dir ./sparkrun-cc-plugin
```

## Prerequisites

### sparkrun CLI

The plugin requires sparkrun to be installed:

```bash
# Install via uvx (recommended)
uvx sparkrun setup install

# Or via uv
uv tool install sparkrun
```

### DGX Spark Cluster

You need SSH access to one or more NVIDIA DGX Spark systems. The fastest way to get started:

```bash
# Interactive setup wizard (handles everything)
sparkrun setup wizard

# Or manual cluster creation
sparkrun cluster create mylab --hosts 192.168.11.13,192.168.11.14 -d "My DGX Spark lab"
sparkrun cluster set-default mylab
sparkrun setup ssh --cluster mylab
```

## Slash Commands

| Command                        | Description                                                      |
|--------------------------------|------------------------------------------------------------------|
| `/sparkrun:run <recipe>`       | Launch an inference workload                                     |
| `/sparkrun:stop [target]`      | Stop a running workload (by recipe name, cluster ID, or `--all`) |
| `/sparkrun:status`             | Check status of running workloads                                |
| `/sparkrun:list [query]`       | Browse and search available recipes                              |
| `/sparkrun:benchmark <recipe>` | Run benchmarks against an inference workload                     |
| `/sparkrun:monitor`            | Live-monitor CPU, RAM, and GPU metrics across cluster hosts      |
| `/sparkrun:proxy <action>`     | Manage the LiteLLM-based inference proxy gateway                 |
| `/sparkrun:setup`              | Guided setup for sparkrun and cluster config                     |

## Skills (Automatic)

Claude automatically uses these skills when the task context matches:

| Skill      | Activates When                                                                                            |
|------------|-----------------------------------------------------------------------------------------------------------|
| `run`      | Running, monitoring, stopping, benchmarking, tuning, or managing inference workloads and proxy            |
| `setup`    | Installing sparkrun, configuring clusters, SSH setup, CX7 networking, Docker group, permissions, earlyoom |
| `registry` | Managing recipe registries, browsing benchmark profiles, creating/editing recipes                         |

## Usage Examples

You can use the slash commands directly:

```
/sparkrun:run qwen3-1.7b-vllm --tp 2
/sparkrun:status
/sparkrun:list qwen3
/sparkrun:benchmark qwen3-1.7b-sglang --tp 1 --profile spark-arena-v1
/sparkrun:monitor --cluster mylab --simple
/sparkrun:proxy start --cluster mylab
```

Or just describe what you want in natural language -- Claude will use the skills automatically:

- "Run the Qwen3 1.7B model on my cluster"
- "What inference jobs are running?"
- "Stop all inference jobs on my cluster"
- "Show me available recipes for llama models"
- "Benchmark the sglang recipe on a single node"
- "Set up sparkrun on my DGX Spark cluster"
- "Configure CX7 networking on my cluster"
- "Create a recipe for Mistral 7B on vLLM"
- "Monitor my cluster's GPU usage"
- "Start the inference proxy and load a model"
- "Check if my job is healthy"

## Key Concepts

- **Recipes** are YAML files describing an inference workload (model, runtime, container, defaults)
- **Runtimes** are inference engines: vLLM, SGLang, llama.cpp, TensorRT-LLM
- **Clusters** are named groups of DGX Spark hosts
- **Registries** are git-based collections of recipes and benchmark profiles
- **Benchmark profiles** define standardized benchmark configurations from registries
- **Proxy** is a unified OpenAI-compatible gateway in front of multiple inference endpoints
- Each DGX Spark has 1 GPU, so `--tp N` (tensor parallelism) = N hosts
- sparkrun launches detached containers -- Ctrl+C detaches from logs, never kills the job
- Recipe names support `@registry/name` syntax for explicit registry selection

## Links

- [sparkrun Documentation](https://github.com/scitrera/sparkrun)
- [Recipe Format Specification](https://github.com/scitrera/sparkrun/blob/main/RECIPES.md)

## License

Apache 2.0 License -- see [LICENSE](../LICENSE) for details.
