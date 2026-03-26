<p align="center">
  <img src="assets/sparkrun-banner.svg" alt="sparkrun — Part of the Spark Arena ecosystem" width="480" />
</p>

<p align="center">
  <a href="https://pypi.org/project/sparkrun/"><img src="https://img.shields.io/pypi/v/sparkrun?color=76b900" alt="PyPI version" /></a>
  <a href="https://github.com/spark-arena/sparkrun/blob/main/LICENSE"><img src="https://img.shields.io/github/license/spark-arena/sparkrun" alt="License" /></a>
  <a href="https://sparkrun.dev"><img src="https://img.shields.io/badge/docs-sparkrun.dev-1e40af" alt="Documentation" /></a>
  <a href="https://spark-arena.com"><img src="https://img.shields.io/badge/Spark_Arena-community-76b900" alt="Spark Arena" /></a>
</p>

<h3 align="center">One command to rule them all</h3>

<p align="center">
  Launch, manage, and stop LLM inference workloads on one or more NVIDIA DGX Spark systems — no Slurm, no Kubernetes, no fuss.
</p>

<p align="center">
  <a href="https://sparkrun.dev">Documentation</a> &middot;
  <a href="https://sparkrun.dev/getting-started/quick-start/">Quick Start</a> &middot;
  <a href="https://sparkrun.dev/recipes/overview/">Recipes</a> &middot;
  <a href="https://spark-arena.com">Spark Arena</a>
</p>

---

## Install

```bash
uvx sparkrun setup
```

One command — installs sparkrun, then launches the guided setup wizard to create a cluster, configure SSH mesh, detect ConnectX-7 NICs, set up sudoers, and enable earlyoom.

## Quick Start

```bash
# Run an inference workload
sparkrun run qwen3-1.7b-vllm

# Multi-node tensor parallelism (TP maps to node count on DGX Spark)
sparkrun run qwen3-1.7b-vllm --tp 2

# Re-attach to logs, stop a workload, check status
sparkrun logs qwen3-1.7b-vllm
sparkrun stop qwen3-1.7b-vllm
sparkrun status
```

Ctrl+C detaches from logs — it never kills your inference job. Your model keeps serving.

See the [full CLI reference](https://sparkrun.dev/cli/overview/) for all commands and options.

## Highlights

- **Multi-runtime** — vLLM, SGLang, llama.cpp out of the box
- **Multi-node tensor parallelism** — `--tp 2` = 2 hosts, automatic InfiniBand/RDMA detection
- **VRAM estimation** — know if your model fits before you launch (`sparkrun show <recipe>`)
- **Git-based recipe registries** — we publish official recipes, community recipes, and benchmarked recipes via [Spark Arena](https://spark-arena.com), plus you can add your own registries.
- **Guided setup wizard** — cluster creation, SSH mesh, CX7 auto-detection, sudoers, earlyoom
- **Model & container distribution** — syncs models and images to cluster nodes over SSH automatically

## Spark Arena
[Spark Arena](https://spark-arena.com) is the community hub for DGX Spark recipe benchmarks — browse benchmark results, then run them directly with sparkrun.

## Official Recipes
[Official Recipes](https://github.com/spark-arena/recipe-registry) are maintained by the Spark Arena team and hosted on GitHub. They are tested and optimized for NVIDIA DGX Spark systems.

## Community Recipes
[Community Recipes](https://github.com/spark-arena/community-recipe-registry) are contributed by the community and hosted on GitHub.



## Sponsored by

<a href="https://scitrera.ai"><img src="https://scitrera.com/logo2.png" alt="scitrera.ai" height="40" /></a>

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
