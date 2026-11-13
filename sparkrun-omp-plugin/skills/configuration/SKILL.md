---
name: configuration
description: Install/update Sparkrun 0.4.0, configure host clusters and SSH, check setup, manage plugins/features, and inspect or clean model caches.
---

# Installation and host setup

Check `sparkrun --version` and installed `--help` before assuming optional commands
are available. These instructions target 0.4.0. Use the workloads skill for execution
and registry for recipes. Keep changes within the user's requested hosts and scope.

## Install and update

```bash
uv --version
uvx sparkrun setup install
sparkrun --version
sparkrun update
sparkrun update --no-update-plugins
sparkrun setup update --no-update-registries --no-update-plugins
sparkrun update --alpha
sparkrun update --stable
```

`uvx sparkrun setup` is the guided install/setup entrypoint; `setup install`
installs the CLI. Python 3.12+ is required. Ordinary `update` upgrades core,
updates trusted managed plugins to compatible versions, and refreshes registries.
`--no-update-plugins` skips plugin fetch/update but audits compatibility. Skipping
registry updates alone does not skip plugin updates.

Stable uses PyPI; beta uses develop; alpha (`--yolo`) uses develop-next. No channel
flag keeps the current channel; switching to stable may downgrade. Do not switch
channels merely to expose optional commands. macOS/Windows controller support and
Linux inference host support are different: Linux setup and systemd steps are not
local macOS/Windows services.

## Wizard and read-only checks

```bash
sparkrun setup wizard
sparkrun setup wizard --hosts <host1>,<host2> --cluster mylab
sparkrun setup wizard --hosts <host1>,<host2> --dry-run
sparkrun setup check --cluster <name>
sparkrun setup check --cluster <name> --json
```

`setup` opens the wizard when no default cluster is configured and its feature is
on. Registered steps/hardware probes cover SSH mesh/identities, Docker, NVIDIA CDI,
CX7, sudoers, IPC reaping, and earlyoom where applicable; do not assume a fixed
sequence. `--yes` accepts defaults, not SSH/sudo passwords; it does not guarantee
an unattended run.

`setup check` is read-only and detects duplicate SSH host keys and machine IDs.
The wizard can offer repair. Rotation is a separate confirmed action:

```bash
sparkrun setup ssh-rotate-keys --cluster <name> --dry-run
sparkrun setup ssh-rotate-keys --cluster <name>
sparkrun setup ssh-rotate-keys --cluster <name> --machine-id
```

This rotates **server host keys**, not login private keys, and refreshes trust/SSH
mesh for the same hosts. `--machine-id` also changes OS identities; reboot afterward.
Other clients must trust new keys. Do not add `--yes` without existing authorization.
If incomplete, report the recovery record; do not discard key mismatch warnings.

## Clusters and host steps

```bash
sparkrun cluster create mylab --hosts <host1>,<host2> --user <user> -d "My lab"
sparkrun cluster set-default mylab
sparkrun cluster list
sparkrun cluster show mylab
sparkrun cluster default
sparkrun cluster update mylab --add-host <host3>
sparkrun cluster update mylab --remove-host <host3>
sparkrun setup ssh --cluster mylab
sparkrun setup ssh --cluster mylab --extra-hosts <controller_ip>
sparkrun setup ssh --cluster mylab --no-include-self --dry-run
sparkrun setup cx7 --cluster mylab --dry-run
sparkrun setup docker-group --cluster mylab
sparkrun setup fix-permissions --cluster mylab --cache-dir /data/huggingface --dry-run
sparkrun setup clear-cache --cluster mylab --dry-run
sparkrun setup earlyoom --cluster mylab
sparkrun setup diagnose --cluster mylab --output diagnostics.ndjson
```

Clusters hold candidate hosts; actual placement may select a subset/different
head. Incremental changes use `--add-host`/`--remove-host` (repeatable, comma lists).
SSH and shared sudo fallback can prompt: use a user terminal. `fix-permissions`
and `clear-cache` offer `--save-sudo` for scoped sudoers entries. `clear-cache`
drops Linux **page cache**, not model files. Diagnostics are NDJSON;
`setup diagnose --json` prints a JSON summary as well as writing the bundle.

## Plugins and features

```bash
sparkrun plugins
sparkrun plugins list --all --json
sparkrun plugins add https://github.com/sparksq/sparkrun-sparkroute-plugin
sparkrun plugins add https://github.com/sparksq/sparkrun-coldsnap-plugin
sparkrun plugins update --all
sparkrun plugins update <plugin_name>
sparkrun plugins rollback <plugin_name>
sparkrun plugins enable <plugin_name>
sparkrun plugins disable <plugin_name>
sparkrun plugins remove <plugin_name>
sparkrun setup features
sparkrun setup features list --all --json
sparkrun setup features enable gateway.sparkroute
sparkrun setup features disable gateway
sparkrun setup features reset gateway
```

Prefer top-level `plugins`; `setup plugins` is a hidden alias. No subcommand lists
plugins/features. Installation asks for trust in remote executable code and may
enable external loading; do not silently add `--trust`. Managed generations are
independent of core installs, compatibility checked, and retain current + previous
by default (configurable).

Plugin enable/disable controls whole-plugin loading. Feature flags control its
capabilities; a flag cannot load a disabled plugin. Installation/update is not
explicit feature activation. Plugin list shows declared flags, not on/off values;
`setup features` shows effective state. Both list only enabled entries by
default (text and JSON, with a "N disabled ... hidden" hint); add `--all` to
include disabled plugins/flags (and `setup.steps.*` flags). OCI Relay is bundled and enabled on all
channels; use `plugins enable/disable oci_relay`, not a legacy integration flag.

Gateway support defaults on; LiteLLM is bundled/default on every channel. After
installing/enabling SparkRoute, explicitly enabling `gateway.sparkroute` selects
it; enabling `gateway.litellm` selects LiteLLM. Both provider flags may stay on.
Disabling shared `gateway` blocks new activity but does not stop a running proxy.
Reset removes an override without activation hooks or changing saved selection.

`SPARKRUN_ADVANCED=1` reveals advanced help (`build`, etc.); it does
not enable feature flags or install plugins.

## Models, transfers, and optional cleanup

```bash
sparkrun model-cache list --cluster <cluster_name>
sparkrun model-cache inspect org/model-name --hosts <host1>,<host2>
sparkrun model-cache repair org/model-name --cluster <cluster_name> --dry-run
sparkrun model-cache prune --cluster <cluster_name> --max-bytes 100G     # preview; --apply to remove
sparkrun model-cache pin org/model-name --revision <commit_or_ref> --cluster <cluster_name>
sparkrun model-cache delete org/model-name --cluster <cluster_name> --dry-run
sparkrun model-cache adopt org/model-name --cluster <cluster_name>   # preview; --apply to track
sparkrun model-cache reconcile org/model-name --cluster <cluster_name>  # preview; --apply to release
```

Targets resolve like other commands: `--cluster`/`--hosts`/`--hosts-file`, else the
default cluster, else localhost. Output is one table per cache (hosts sharing one cache,
such as an NFS export, are listed together); add `--json` to parse it.
Prune previews by default; `--apply` needs authorization for cleanup. It uses LRU
among eligible managed revisions, respects pins/live claims, leaves untracked snapshots
alone, and removes only blobs no other revision uses. Usage records cover Sparkrun lifecycle activity,
not every external read. To remove a specific model regardless of pins, use
`model-cache delete` (preview first; it needs explicit confirmation or `--yes`,
and `--force` for revisions a workload claims). A claim whose workload is gone
(nothing in `sparkrun status`) is released by `model-cache reconcile <model>` or
`--all`; it previews by default and keeps claims it cannot prove orphaned (unreachable
hosts, younger than an hour). Prefer it to `--force`. Do not replace either with
recursive cache deletion. Revisions downloaded outside sparkrun are untracked until
`model-cache adopt` (preview by default; name a model or pass `--all`): each file is
checked against the model origin, or with `--offline` only Hub-layout snapshots qualify;
adoption starts the retention clock now unless `--last-used mtime`.

Transfer modes: `auto`, `local`, `push`, `delegated`, `pull` (per-node origin fetches).
Delegated acquisition downloads on the chosen head and distributes to workers;
it is not independent worker downloads. Credentials must exist where acquisition
occurs. Manifest validation and optional offline preparation apply across modes.
Default rsync concurrency is two streams per destination, sixteen total. Diagnose
before overriding transfer/file-selection policy.

<Tool_Usage>
Run these commands with the `bash` tool. Raise `timeout` (up to 3600 seconds) for
downloads, readiness waits, and benchmarks; the default 300-second deadline is too
short for most launches. Start indefinite streams (`logs --follow`, `cluster
monitor --simple`, `cluster monitor --json`) as background jobs (`async: true` or
`timeout: 0`) and stop them when enough samples are collected. A background job or
a timed-out bash call kills only the local CLI process, never the remote workload.
Interactive prompts (`setup wizard`, SSH/sudo, trust confirmations, Arena login)
need the user's own terminal. Never claim a timed-out launch was rolled back.
</Tool_Usage>

References: [upgrade guide](https://github.com/spark-arena/sparkrun/blob/main/docs/UPGRADING_0_4.md),
[plugins](https://github.com/spark-arena/sparkrun/blob/main/docs/PLUGINS.md),
[model caches](https://github.com/spark-arena/sparkrun/blob/main/docs/MODEL_CACHE.md).
