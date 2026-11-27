# Sparkrun plugin for oh-my-pi (omp)

Skills and slash commands for the **Sparkrun 0.4.3 CLI in this checkout**: recipe discovery and
authoring, readiness-aware launches, bounded logs, performance/tool benchmarks,
gateway management, host setup, and plugin authoring. Sparkrun must be installed
on the machine executing omp's `bash` tool, with access to the target hosts.

## Install

```bash
omp plugin marketplace add spark-arena/sparkrun
omp plugin install sparkrun@sparkrun
```

For a local checkout, load the plugin directly:

```bash
omp --plugin-dir ./sparkrun-omp-plugin
```

Run `/reload-plugins` after updating a checkout. Use `omp plugin list` to confirm
the plugin and its commands are loaded. This assistant plugin is separate from
Python extensions installed with `sparkrun plugins add`.

## Sparkrun prerequisites

```bash
uvx sparkrun setup install
sparkrun --version
sparkrun setup check --cluster mylab
```

The CLI requires Python 3.12+. These references target core 0.4.3; check installed
`--version` and `--help` because older releases may not expose every capability.
Keep the user's selected update channel. Interactive setup, SSH/sudo, trust confirmations,
and Arena login require a user terminal, because omp's `bash` tool has no
interactive stdin.

## Commands and skills

| Command | Purpose |
| --- | --- |
| `/sparkrun:run <recipe>` | Launch and report actual readiness/placement |
| `/sparkrun:stop <target>` | Stop the selected workload |
| `/sparkrun:status` | Inspect workloads and pending operations |
| `/sparkrun:list [query]` | Browse, search, and inspect recipes |
| `/sparkrun:benchmark [perf\|tools] <recipe>` | Measure performance or tool use; optionally publish |
| `/sparkrun:monitor` | Observe a cancellable metrics stream |
| `/sparkrun:proxy <action>` | Manage the selected gateway |
| `/sparkrun:setup` | Install, check, or configure hosts and plugins |

| Skill | Scope |
| --- | --- |
| workloads | Workloads, logs, benchmarks, gateways, optional daemon clients |
| configuration | Install/update, host checks, SSH identities, plugins/features, cache retention |
| registry | Recipe catalogs, benchmark profiles, registry trust |
| recipes | Recipe schema, runtime defaults, topology, hooks, includes/overrides, publishing |
| plugins | Plugin packages, extension points, recipe items, execution strategies, trust |

Reference skill names differ from the action commands so they do not shadow
`/sparkrun:run` or `/sparkrun:setup`. Describe tasks naturally or use a command:

```text
/sparkrun:run @official/my-recipe --cluster mylab
/sparkrun:benchmark tools my-recipe --skip-run
/sparkrun:benchmark my-recipe --arena
/sparkrun:proxy status
```

Use recipe names returned by your registry; the names above illustrate syntax.

## Execution notes

omp's `bash` tool has a 300-second default deadline; raise `timeout` for downloads,
readiness waits, and benchmarks, respecting the host's configured maximum
(documented maximum: 3600 seconds). `async: true` requests a background job only
when the session's `async.enabled` setting permits it, and does not change the
deadline. `timeout: 0` disables the tool-imposed deadline; it does not itself
request background execution. Background jobs continue until completion,
cancellation, or their deadline (`timeout: 0` disables that deadline).

For a requested long-running stream such as `logs --follow` or cluster monitor
`--simple`/`--json`, use `async: true` with an appropriate deadline and stop or
cancel it when enough samples are collected. Cancellation or timeout stops the
local command, but remote workloads may continue; inspect status before retrying
and never claim a timed-out launch was rolled back. Interactive prompts need the
user's own terminal.

`--no-follow` suppresses model logs but waits for readiness; `--no-ready-wait`
returns after preparation/launch without claiming readiness. Bare `logs` is
finite; monitor modes and `logs --follow` stream.

LiteLLM is the bundled gateway default on every channel. SparkRoute is installed
separately; explicitly enabling its gateway feature selects it. Plugin loading,
feature activation, and gateway selection are distinct operations. Prefer
`sparkrun benchmark --arena` over the hidden `arena benchmark` alias.

Ordinary CLI use does not need the optional daemon and is not automatically routed
through it. No MCP server is required.

## Maintaining these references

The Claude Code skills are the source for all three assistant packages. After
editing one, run these commands from the repository root:

```bash
python3 scripts/sync-assistant-plugins.py
python3 scripts/sync-assistant-plugins.py --check
```

The generator replaces tool guidance for OpenClaw and omp, and copies the
commands for omp. Python CI checks generated parity and parses documented
Sparkrun examples without running workloads. Update package versions through
`versions.yaml` and `python3 scripts/update-versions.py`.

See the [Sparkrun guides](https://github.com/spark-arena/sparkrun#readme) and
[recipe specification](https://github.com/spark-arena/sparkrun/blob/main/RECIPES.md).
Licensed under [Apache 2.0](../LICENSE).