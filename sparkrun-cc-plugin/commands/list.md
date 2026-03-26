# /sparkrun:list

Browse and search available inference recipes.

## Usage

```
/sparkrun:list [query]
```

## Examples

```
/sparkrun:list
/sparkrun:list qwen3
/sparkrun:list llama-cpp
```

## Behavior

When this command is invoked:

1. **Browse all recipes** (no filter):

```bash
sparkrun list
```

2. **Browse with filters:**

```bash
sparkrun list --all                         # include hidden registry recipes
sparkrun list --registry <name>             # filter by registry
sparkrun list --runtime vllm                # filter by runtime (vllm, sglang, llama-cpp)
sparkrun list <query>                       # filter by name (contains-match)
```

3. **Search for recipes** by name, model, runtime, or description (contains-match):

```bash
sparkrun recipe search <query>
sparkrun recipe search <query> --registry <name> --runtime sglang
```

Use `sparkrun recipe search` as the first attempt when the user wants to find a particular recipe. Only fall back to other approaches if it doesn't return useful results.

4. **Inspect a specific recipe** (by exact name or recipe file):

```bash
sparkrun recipe show <recipe>
sparkrun recipe show <recipe> --tp <N>   # include VRAM estimate for N nodes
```

Use `sparkrun recipe show` when given a specific recipe name or file path -- these may not appear in search results.

5. If the user wants to validate or check VRAM for a recipe:

```bash
sparkrun recipe validate <recipe>
sparkrun recipe vram <recipe> --tp <N> --max-model-len 32768 --gpu-mem 0.9
```

6. To export a normalized recipe:

```bash
sparkrun recipe export <recipe>             # print to stdout
sparkrun recipe export <recipe> --json      # JSON format
sparkrun recipe export <recipe> --save out.yaml  # save to file
```

## Notes

- Recipes come from built-in and custom registries
- Run `sparkrun registry update` or `sparkrun update` to fetch the latest recipes
- Use `sparkrun registry list` to see configured registries
- `sparkrun list` shows everything; `sparkrun recipe search` filters by query; `sparkrun recipe show` inspects a known recipe
- Recipe names support `@registry/name` syntax for explicit registry selection
