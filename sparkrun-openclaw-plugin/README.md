# sparkrun Plugin for OpenClaw

AI-assisted inference on NVIDIA DGX Spark -- run, manage, and stop LLM workloads with OpenClaw.

## What It Does

This plugin teaches OpenClaw how to use [sparkrun](https://github.com/scitrera/sparkrun) to manage LLM inference
workloads on NVIDIA DGX Spark systems. It provides:

- **Skills** -- Detailed reference that OpenClaw uses automatically when working with sparkrun
- **sparkrun_exec Tool** -- A dedicated tool for executing sparkrun CLI commands

## Installation

### From npm

```bash
openclaw plugins install @sparkarena/sparkrun-openclaw
```

### From Local Directory

```bash
# Clone the repo
git clone https://github.com/spark-arena/sparkrun.git
cd sparkrun

# Install from local path
openclaw plugins install ./sparkrun-openclaw-plugin

# Or link for development
openclaw plugins install ./sparkrun-openclaw-plugin --link
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

## Skills (Automatic)

OpenClaw automatically uses these skills when the task context matches:

| Skill      | Activates When                                                                                            |
|------------|-----------------------------------------------------------------------------------------------------------|
| `run`      | Running, monitoring, stopping, benchmarking, tuning, or managing inference workloads and proxy            |
| `setup`    | Installing sparkrun, configuring clusters, SSH setup, CX7 networking, Docker group, permissions, earlyoom |
| `registry` | Managing recipe registries, browsing benchmark profiles, creating/editing recipes                         |

## Usage Examples

Describe what you want in natural language -- OpenClaw will use the skills automatically:

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

- [sparkrun Documentation](https://sparkrun.dev)
- [sparkrun github repo](https://github.com/spark-arena/sparkrun)
- [Recipe Format Specification](https://sparkrun.dev/recipes/format/)

## License

Apache 2.0 License -- see [LICENSE](./LICENSE) for details.
