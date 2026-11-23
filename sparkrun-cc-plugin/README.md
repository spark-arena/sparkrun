# Sparkrun plugin for Claude Code

Skills and slash commands for the **Sparkrun 0.4.0 CLI**: recipe discovery,
readiness-aware launches, bounded logs, performance/tool benchmarks, gateway
management, host setup, and plugin administration. Sparkrun must be installed on
the machine executing Claude Code's commands, with access to the target hosts.

## Install

```bash
claude plugin marketplace add spark-arena/sparkrun
claude plugin install sparkrun@sparkrun
```

For a local checkout, use the supported development loader:

```bash
claude --plugin-dir ./sparkrun-cc-plugin
```

Use Claude Code's plugin manager for installed-plugin updates. Copying files into
an arbitrary `.claude/plugins` directory is not an installation workflow. See
[Claude Code installation](https://code.claude.com/docs/en/discover-plugins).
This assistant plugin is separate from Python extensions installed with
`sparkrun plugins add`.

## Sparkrun prerequisites

```bash
uvx sparkrun setup install
sparkrun --version
sparkrun setup wizard
```

The CLI requires Python 3.12+. These instructions target 0.4.0, including its
preview builds; inspect installed `--help` when using an older release. Keep the
user's selected update channel. Interactive setup, SSH/sudo, trust confirmations,
and Arena login require a user terminal, not a captured agent command.

For an existing cluster, start with the read-only check:

```bash
sparkrun setup check --cluster mylab
```

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

The workloads, configuration, registry, recipes, and plugins skills supply detailed references automatically when
relevant. Reference skill names differ from the action commands so they do not
shadow `/sparkrun:run` or `/sparkrun:setup`. Describe tasks naturally or use a command:

```text
/sparkrun:run @official/my-recipe --cluster mylab
/sparkrun:benchmark tools my-recipe --skip-run
/sparkrun:benchmark my-recipe --arena
/sparkrun:proxy status
```

Use recipe names returned by your registry; the names above illustrate syntax.
`--no-follow` suppresses model logs but waits for readiness; `--no-ready-wait`
returns after preparation/launch without claiming readiness. Bare `logs` is
finite; monitor modes and `logs --follow` stream. Allow sufficient execution time
for downloads and benchmarks.

LiteLLM is the bundled gateway default on every channel. SparkRoute is installed
separately; explicitly enabling its gateway feature selects it. Plugin loading,
feature activation, and gateway selection are distinct operations. Prefer
`sparkrun benchmark --arena` over the hidden `arena benchmark` alias.

Ordinary CLI use does not need the optional daemon and is not automatically routed
through it. No MCP server is required.

## Maintaining these references

The Claude Code skills are the source for all three assistant packages. After editing
one, run these commands from the repository root:

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
Licensed under [Apache 2.0](https://github.com/spark-arena/sparkrun/blob/main/LICENSE).
