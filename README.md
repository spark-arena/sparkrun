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

Watched launches also report Docker-start TTR and TTFT using a rank-local
streaming readiness check for Docker vLLM/SGLang launches. By default, `run`
follows startup logs and exits when ready; use `--follow` to stay attached,
`--no-follow` to wait without model logs, or `--no-ready-wait` to return after launch. See
[startup readiness](docs/STARTUP_READINESS.md) for configuration, timing
boundaries, per-recipe overrides, and execution-strategy integration.

See the [full CLI reference](https://sparkrun.dev/cli/overview/) for all commands and options.

## Updating

```bash
sparkrun update
```

Upgrades sparkrun (when installed via `uv tool`) and refreshes recipe registries.

### Update channels (advanced)

Opt into preview builds installed from git instead of PyPI:

```bash
sparkrun update --stable   # PyPI stable release (default)
sparkrun update --beta     # develop branch preview
sparkrun update --alpha    # develop-next branch (bleeding edge)
sparkrun update --yolo     # alias for --alpha
```

`sparkrun update` with no flag stays on your current channel; a channel flag switches and is remembered for future updates. The same flags work with `sparkrun setup install` and `sparkrun setup update`. Stable prints a plain version; beta/alpha add a channel suffix and commit (for example, `0.4.0-alpha+g1a2b3c4`). Switching from a preview channel back to `--stable` may downgrade.

## Highlights

- **Multi-runtime** — vLLM, SGLang, llama.cpp out of the box
- **Multi-node tensor parallelism** — `--tp 2` = 2 hosts, automatic InfiniBand/RDMA detection
- **VRAM estimation** — know if your model fits before you launch (`sparkrun show <recipe>`)
- **Git-based recipe registries** — we publish official recipes, community recipes, and benchmarked recipes via [Spark Arena](https://spark-arena.com), plus you can add your own registries.
- **Guided setup wizard** — cluster creation, SSH mesh, CX7 auto-detection, sudoers, earlyoom
- **Model & container distribution** — syncs models and images to cluster nodes over SSH automatically
- **Manifest model caches** — automatic [expected-file validation, selective weight downloads and pluggable model transfer](docs/MODEL_CACHE.md)

## Python API and applications (0.4.0)

The 0.4.0 branch introduces breaking Python API changes. Start with the
[migration guide](docs/DISTRIBUTION_API_MIGRATION.md), including the supported
imports and option/result contracts. [Application profiles](docs/APPLICATION_PROFILES.md)
let CLIs, daemons, and desktop applications share the core with their own identity,
paths, defaults, and plugins. Use `sparkrun.application.initialize()` for Python
API use without the Sparkrun CLI. The [catalog API](docs/CATALOG_API.md) supports
headless browsing, previews, and persistent recipe selection.

[Preparation-only builds](docs/BUILD.md) build images or native Python environments
and stage model assets before launch. Set `SPARKRUN_ADVANCED=1` to show
`sparkrun build` in CLI help.

[Benchmarking](docs/BENCHMARK_API.md) separates measurement frameworks from
publication integrations. [Plugin authors](docs/PLUGINS.md) register installed
integrations through `sparkrun.plugins`.

The [proxy CLI](docs/PROXY.md) supports pluggable gateways, including LiteLLM and
SparkRoute. Docker now stages a custom seccomp profile allowing io_uring on each
launch node; see [executor configuration](docs/EXECUTORS.md#docker-seccomp-profiles-04).
vLLM no longer sets `OMP_NUM_THREADS` by default; recipes can set it explicitly.

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

The bundled [SparkRoute integration](src/sparkrun/plugins/sparkroute/README.md)
is AGPL-3.0-only with an [additional permission](src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION)
for combining and distributing it with SparkRun. Its notices and immutable
source provenance ship in the installed plugin package. The independently
acquired SparkRoute executable is AGPL-3.0-only.
SparkRoute defaults on for alpha; LiteLLM defaults on for stable and beta.
See [channel defaults and managed plugin updates](docs/SPARKROUTE.md).

## Anonymous Telemetry

sparkrun sends basic anonymous usage telemetry to `https://telemetry.sparkrun.dev` by default. Events include a random installation id stored in `~/.config/sparkrun/config.yaml`, sparkrun version, OS/version, system architecture, and command-specific metadata such as run runtime/model/parallelism/source/hardware counts, benchmark category/framework/profile/result keys, update version and registry counts, and setup-wizard step choices.

The data allows us to make informed decisions about new features for sparkrun or the greater DGX Spark ecosystem. 

Telemetry very specifically does not include personally identifiable information or information that may reveal trade secrets. Telemetry does not include hostnames, usernames, local file paths, tokens, secrets, logs, private HF or local models, or full command arguments. Disable it persistently with `sparkrun setup telemetry --disable`, re-enable with `sparkrun setup telemetry --enable`, or opt out for one process with `SPARKRUN_NO_TELEMETRY=1`.
