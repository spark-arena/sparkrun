# Developer Guide

## Quick Start

```bash
git clone https://github.com/scitrera/sparkrun.git -b develop
cd sparkrun
source dev.sh
sparkrun --help
```

`source dev.sh` creates a `.venv`, installs sparkrun in editable mode with dev dependencies, and activates the venv in your current shell. After sourcing, `sparkrun` in that shell runs the code from your checkout — edits take effect immediately.

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

## Branch Model

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases. Protected — no direct pushes |
| `develop` | Integration branch. **PRs target here** |
| `feature/*` | Feature branches off `develop` |

**All PRs should target `develop`**, not `main`. Releases are merged from `develop` → `main`.

## Running Tests

```bash
# Full suite
.venv/bin/python -m pytest tests/ -v

# Single file
.venv/bin/python -m pytest tests/test_recipe.py -v

# Single test
.venv/bin/python -m pytest tests/test_cli.py::test_run_command_basic -v

# With coverage
.venv/bin/python -m pytest tests/ --cov=sparkrun --cov-report=term-missing
```

All tests are self-contained — no real hosts, SSH, or Docker needed. SSH/Docker operations are mocked via `conftest.py` fixtures.

## Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

Config: `pyproject.toml` — line-length 140, target Python 3.12.

## Project Layout

```
src/sparkrun/
├── cli/              # Click CLI (one module per command group)
├── core/             # Config, recipe, registry, launcher, cluster manager
├── runtimes/         # Runtime plugins (vllm, sglang, llama-cpp, trtllm)
├── orchestration/    # SSH, Docker, executor, InfiniBand, scripts
├── builders/         # Builder plugins (eugr, docker-pull)
├── models/           # HuggingFace download, distribution, VRAM estimation
├── containers/       # Container image distribution
├── tuning/           # Triton kernel tuning (SGLang, vLLM)
├── benchmarking/     # Benchmark framework plugins
├── utils/            # Shared helpers
└── scripts/          # Embedded bash scripts (*.sh)
tests/                # pytest tests (mirrors src/ structure)
```

## Key Patterns

### Plugin System (SAF)

Runtimes, builders, and benchmarking frameworks are SAF multi-extension plugins discovered via Python entry points in `pyproject.toml`. Bootstrap flow:

```
cli/__init__.py → core/bootstrap.py → SAF init → find_types_in_modules() → register_plugin()
```

### Config Chain (vpd)

sparkrun uses `vpd_chain` for cascading config resolution throughout the codebase:

```python
from vpd.legacy.yaml_dict import vpd_chain
config = vpd_chain(cli_overrides, recipe_defaults, runtime_defaults)
value = config.get("port")  # resolves through the chain
```

Priority: CLI → recipe → runtime defaults. The same pattern is used for executor config, recipe defaults, and benchmark profiles.

### Executor Abstraction

Container engine operations go through the `Executor` ABC (`orchestration/executor.py`). `DockerExecutor` is the current implementation. Runtimes use `self.executor.*` instead of importing `docker.py` directly:

```python
# In a runtime:
self.executor.run_cmd(image, command, container_name=name, env=env)
self.executor.stop_cmd(container_name)
self.executor.generate_launch_script(image, container_name, command, ...)
self.executor.container_name(cluster_id, "solo")
self.executor.node_container_name(cluster_id, rank)
```

`ExecutorConfig` is built from a vpd chain (CLI flags → recipe `executor_config` → `EXECUTOR_DEFAULTS`) in `launcher.py` and passed to the executor.

### SSH Execution Model

All remote operations use **SSH stdin piping** — scripts are generated as Python strings and piped to `ssh host bash -s`. No files are ever copied to remote hosts for execution.

```python
from sparkrun.orchestration.ssh import run_remote_script
result = run_remote_script(host, script_string, timeout=120, **ssh_kwargs)
```

### Runtime Architecture

All runtimes extend `RuntimePlugin` (`runtimes/base.py`):

- `generate_command()` — produce the serve command from recipe + overrides
- `resolve_container()` — resolve the container image
- `run()` / `stop()` — solo/cluster dispatch (base class handles solo; subclasses implement `_run_cluster`)
- `cluster_strategy()` — `"ray"` or `"native"` determines orchestration path
- `get_extra_docker_opts()`, `get_extra_volumes()`, `get_extra_env()` — runtime-specific hooks

### Test Isolation

`conftest.py` provides an `isolate_stateful` autouse fixture that redirects SAF's stateful root to `tmp_path`. Tests never touch `~/.config/sparkrun/`. The bootstrap singleton is reset between tests.

## Adding a New Runtime

1. Create `src/sparkrun/runtimes/my_runtime.py` extending `RuntimePlugin`
2. Set `runtime_name = "my-runtime"` and `default_image_prefix`
3. Implement `generate_command()` and optionally `resolve_container()`
4. For multi-node: implement `_run_cluster()` and `_stop_cluster()`
5. Register the entry point in `pyproject.toml`:
   ```toml
   [project.entry-points."sparkrun.runtimes"]
   my_runtime = "sparkrun.runtimes.my_runtime:MyRuntime"
   ```
6. Add tests in `tests/test_my_runtime.py`

## Adding a New Builder

1. Create `src/sparkrun/builders/my_builder.py` extending `BuilderPlugin`
2. Set `builder_name = "my-builder"`
3. Implement `prepare_image()` — must return the final image name
4. Register in `pyproject.toml` under `sparkrun.builders`
5. Recipes reference it as `builder: my-builder`

## Version Management

Versions are tracked in `versions.yaml` at the repo root:

```bash
# Sync versions across pyproject.toml and companion packages
python scripts/update-versions.py

# CI check (verify, don't write)
python scripts/update-versions.py --check
```

## Commit Guidelines

- Target `develop` branch for all PRs
- Keep commits atomic — one logical change per commit
- Run `pytest` and `ruff check` before pushing
- Use `--dry-run` to verify CLI changes produce correct Docker commands
