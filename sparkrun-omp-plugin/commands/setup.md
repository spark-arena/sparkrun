---
description: Install or check Sparkrun and configure hosts, plugins, and features
argument-hint: "[task or options]"
---

# /sparkrun:setup

Use the configuration skill for 0.4.0 command details. Check the installed version before
changing anything. Preserve the user's chosen cluster/hosts and update channel.

```bash
sparkrun --version
uvx sparkrun setup install
sparkrun setup check --cluster <name> --json
sparkrun setup wizard --hosts <host1>,<host2> --cluster mylab --dry-run
```

Install only if needed. Prefer read-only `setup check` when diagnosing an existing
cluster. The wizard uses hardware-aware steps; interactive SSH/sudo prompts need
a user terminal. `--yes` accepts defaults but cannot provide passwords. A cluster
stores candidate hosts; launch placement can use a subset and choose its head.

Duplicate server keys/machine IDs are reported by `setup check`; the wizard can
offer repair. `setup ssh-rotate-keys --cluster <name>` requires confirmation;
`--machine-id` also requires reboot afterward. Do not bypass trust/rotation
confirmation without the user's existing authorization.

```bash
sparkrun plugins
sparkrun setup features
sparkrun update
sparkrun update --no-update-plugins
sparkrun registry update
```

Ordinary update includes compatible updates for trusted managed plugins as well
as core/registries. Registry-only refresh uses `registry update`. Plugin loading
and feature enabling are distinct; explicit gateway feature enabling selects
that provider. See the configuration skill for external plugin installation, SSH repair,
model-cache retention, and the advanced help switch.

User request: $ARGUMENTS
