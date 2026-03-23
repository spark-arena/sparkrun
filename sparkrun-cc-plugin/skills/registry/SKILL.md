---
name: registry
description: Manage recipe registries and create inference recipes
---

<Purpose>
Provides complete reference for managing sparkrun recipe registries, browsing and searching recipes, managing benchmark profiles, validating recipes, and understanding the recipe file format for NVIDIA DGX Spark inference workloads.
</Purpose>

<Use_When>
- User wants to add, remove, enable, disable, or update recipe registries
- User wants to browse or search for available recipes
- User wants to create or edit a recipe YAML file
- User wants to validate a recipe or check VRAM requirements
- User wants to browse or inspect benchmark profiles
- User asks about recipe format or fields
</Use_When>

<Do_Not_Use_When>
- User wants to run, stop, or monitor workloads -- use the run skill instead
- User wants to install sparkrun or set up clusters -- use the setup skill instead
</Do_Not_Use_When>

<Steps>

## Registry Commands

```bash
# List configured registries (enabled only by default)
sparkrun registry list
sparkrun registry list --show-disabled
sparkrun registry list --only-show-visible

# Add registries from a git repo's .sparkrun/registry.yaml manifest
sparkrun registry add <git_url>

# Remove a registry
sparkrun registry remove <name>

# Enable a disabled registry
sparkrun registry enable <name>

# Disable a registry (recipes will not appear in searches)
sparkrun registry disable <name>

# Update all enabled registries from git (fetches latest recipes)
sparkrun registry update

# Update a specific registry
sparkrun registry update <name>

# Update sparkrun + registries in one command
sparkrun update
```

## Browsing Recipes

```bash
# List all recipes across all registries (no filter)
sparkrun list

# List with filters
sparkrun list --all                         # include hidden registry recipes
sparkrun list --registry <name>             # filter by registry
sparkrun list --runtime vllm                # filter by runtime (vllm, sglang, llama-cpp)
sparkrun list <query>                       # filter by name

# Search for recipes by name, model, runtime, or description (contains-match)
sparkrun recipe search <query>
sparkrun recipe search <query> --registry <name> --runtime sglang

# Inspect a specific known recipe (by exact name or file path)
sparkrun recipe show <recipe> [--tp N]

# Export a normalized recipe
sparkrun recipe export <recipe>
sparkrun recipe export <recipe> --json
sparkrun recipe export <recipe> --save out.yaml
```

Use `sparkrun recipe search` as the first attempt when looking for a particular recipe. Use `sparkrun recipe show` when given a specific recipe name or file -- it may not appear in search results.

Recipe names support `@registry/name` syntax for explicit registry selection (e.g. `@spark-arena/qwen3-1.7b-vllm`).

## Benchmark Profiles

```bash
# List available benchmark profiles across registries
sparkrun registry list-benchmark-profiles
sparkrun registry list-benchmark-profiles --registry <name>
sparkrun registry list-benchmark-profiles --all    # include hidden registries

# Show detailed benchmark profile information
sparkrun registry show-benchmark-profile <name>
```

## Validating Recipes

```bash
# Check a recipe for issues
sparkrun recipe validate <recipe>

# Estimate VRAM usage with overrides
sparkrun recipe vram <recipe> [--tp N] [--max-model-len 32768] [--gpu-mem 0.9]
```

## Recipe File Format

Recipes are YAML files defining an inference workload:

```yaml
model: org/model-name                  # HuggingFace model ID (required)
runtime: vllm | sglang | llama-cpp     # Inference runtime (required)
container: registry/image:tag          # Docker image (required)
min_nodes: 1                           # Minimum hosts needed
max_nodes: 4                           # Maximum hosts (optional)
model_revision: abc123                 # Pin to specific HF revision (optional)

metadata:
  description: Human-readable description
  maintainer: name <email>
  model_params: 7B
  model_dtype: fp16
  category: general                    # Recipe category

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 2
  pipeline_parallel: 1
  gpu_memory_utilization: 0.9
  max_model_len: 32768
  served_model_name: my-model
  tokenizer_path: org/base-model       # Required for GGUF models on SGLang

# Optional: explicit command template (overrides auto-generation)
command: |
  python3 -m vllm.entrypoints.openai.api_server \
      --model {model} \
      --tensor-parallel-size {tensor_parallel} \
      --port {port}

# Optional: environment variables passed to the container
env:
  NCCL_DEBUG: INFO

# Optional: post-launch hooks
post_exec:                             # Commands to run inside the head container
  - "echo 'Model loaded'"
post_commands:                         # Commands to run on the control machine
  - "curl http://{head_ip}:{port}/v1/models"
stop_after_post: false                 # Stop workload after post hooks (default: false)
```

**Key fields:**
- `{placeholder}` in `command:` templates are substituted from defaults + CLI overrides
- `model_revision` pins downloads to a specific HuggingFace commit/tag
- `tokenizer_path` is required for GGUF models on SGLang (points to base non-GGUF model)
- `min_nodes` / `max_nodes` control cluster size validation
- Shell variable references like `${HF_TOKEN}` in `env:` are expanded from the control machine's environment
- `post_exec` and `post_commands` run after the server is healthy (port listening + /v1/models check)
- `pipeline_parallel` enables pipeline parallelism (total nodes = TP * PP)

</Steps>

<Tool_Usage>
All sparkrun commands are executed via the Bash tool. No MCP tools are required.
</Tool_Usage>

<Important_Notes>
- Run `sparkrun registry update` or `sparkrun update` periodically to get the latest community recipes
- Use `sparkrun recipe validate` before publishing custom recipes
- Use `sparkrun recipe vram` to check if a model fits on DGX Spark before trying to run it
- When creating GGUF + SGLang recipes, always set `tokenizer_path` in defaults
- Custom command templates should include all relevant `{placeholder}` references to pick up defaults and CLI overrides
- Registries are cached at `~/.cache/sparkrun/registries/` and updated with `sparkrun registry update`
- sparkrun ships with built-in recipes; additional registries point to any git repo containing `.yaml` recipe files
- Use `sparkrun registry list-benchmark-profiles` to discover available benchmark profiles from registries
- Recipe names support `@registry/name` syntax for explicit registry selection
- Use `sparkrun recipe export` to get a normalized view of a recipe (useful for debugging)
</Important_Notes>

Task: {{ARGUMENTS}}
