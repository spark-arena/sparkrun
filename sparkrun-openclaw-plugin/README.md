# Sparkrun plugin for OpenClaw

Skills and a `sparkrun_exec` tool for the **Sparkrun 0.4.0 CLI**: launches,
readiness, bounded logs, performance/tool benchmarks, gateways, host setup, plugins,
and model caches. Sparkrun must be installed on the OpenClaw host's PATH, with
access to the target hosts and the intended environment/configuration.

## Install

```bash
openclaw plugins install @sparkarena/sparkrun-openclaw
```

Or install/link from a checkout:

```bash
openclaw plugins install ./sparkrun-openclaw-plugin
openclaw plugins install ./sparkrun-openclaw-plugin --link
```

Use OpenClaw's plugin manager for updates. The host must support
`openclaw/plugin-sdk/plugin-entry` and the registered tool callback contract.
See [OpenClaw plugin entrypoints](https://docs.openclaw.ai/plugins/sdk-entrypoints/define-plugin-entry)
and [tool plugins](https://docs.openclaw.ai/plugins/tool-plugins).
This assistant plugin is separate from Python extensions installed with
`sparkrun plugins add`.

## Sparkrun prerequisites

```bash
uvx sparkrun setup install
sparkrun --version
sparkrun setup check --cluster mylab
```

Sparkrun requires Python 3.12+. References target 0.4.0, including preview builds;
check installed `--help` for older releases and preserve the user's update channel.
Initial `sparkrun setup wizard`, SSH/sudo, trust confirmations, and Arena login
need an interactive user terminal. The execution tool has no interactive stdin.

## Execution tool

Pass CLI **arguments as an array**, excluding the executable name:

```json
{"args": ["run", "my-recipe", "--cluster", "mylab", "--no-follow"], "timeout": 1800000}
```

```json
{"args": ["logs", "sparkrun_actual_workload_id", "--all-sources", "--tail", "200"]}
```

```json
{"args": ["benchmark", "tools", "my-recipe", "--skip-run"], "timeout": 3600000}
```

Replace illustrative recipe/ID values with those returned by Sparkrun. As of
plugin 0.0.2, `args` replaces the old `command` string input. Each array element
is one literal argument; do not include shell quotes, pipelines, redirects,
environment assignments, or `$(...)` expansion. Configure environment values in
the OpenClaw host instead. This is a CLI adapter, not a restriction on what an
authorized Sparkrun command can do.

Execution is asynchronous and reports stdout, stderr, errors, exit status,
cancellation, and timeouts. The default timeout is 120000 ms; the maximum is
3600000 ms. Output capture is limited to 4 MiB per stream; exceeding it is an
error, so prefer bounded log reads. The tool closes stdin to avoid waiting for
answers it cannot supply. Cancellation/timeout stops the local CLI process;
remote workloads may still be running. Check status before retrying.

`--no-follow` still waits for model readiness. `--no-ready-wait` skips the final
readiness wait, not downloads or preparation. Bare `logs` reads and exits;
`logs --follow`, monitor `--simple`, and monitor `--json` stream indefinitely.
Use a cancellable background facility for continuous observation or choose a
short timeout for a deliberate sample. Never claim a timed-out run was rolled back.

## Skills

| Skill | Scope |
| --- | --- |
| workloads | Workloads, logs, benchmarks, gateways, optional daemon clients |
| configuration | Install/update, host checks, SSH identities, plugins/features, cache retention |
| registry | Recipe catalogs, benchmark profiles, registry trust |
| recipes | Recipe schema, runtime defaults, topology, hooks, includes/overrides, publishing |
| plugins | Plugin packages, extension points, recipe items, execution strategies, trust |

Examples of natural-language tasks:

- Find a recipe for my model and preview placement.
- Show the last 200 lines from every worker of this workload.
- Run the tool benchmark against my already-running server.
- Run a benchmark and submit its results to Arena.
- Check my hosts for duplicate SSH keys without changing them.
- Preview model-cache cleanup on these hosts.

LiteLLM is bundled/default on all channels; SparkRoute is a separately installed
plugin. Explicit gateway feature enabling selects the implementation. Prefer
`benchmark --arena` for publication. Ordinary CLI commands do not require the
daemon or route through it automatically; MCP remains disabled.

## Development

Run `npm test` here for offline child-process tests. No OpenClaw host, inference
cluster, credentials, or network are needed for those tests. They cover the
registered tool callback signature, output/error handling, literal arguments,
timeouts, cancellation, and closed stdin. Live OpenClaw loading requires the host.

Skills are generated from `sparkrun-cc-plugin/skills` and commands for oh-my-pi
from `sparkrun-cc-plugin/commands`; edit the Claude Code source and run
`python3 scripts/sync-assistant-plugins.py` from the repository root. Python CI
checks parity for both generated packages and documented CLI syntax. Update
independent package versions in `versions.yaml` and run
`python3 scripts/update-versions.py`.

See the [Sparkrun guides](https://github.com/spark-arena/sparkrun#readme).
Licensed under [Apache 2.0](./LICENSE).
