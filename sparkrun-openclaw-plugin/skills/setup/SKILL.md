---
name: setup
description: Install sparkrun and configure DGX Spark clusters
---

<Purpose>
Provides complete reference for installing sparkrun, creating and managing cluster configurations, setting up SSH mesh for multi-node inference, configuring CX7 networking, Docker group membership, file permissions, page cache, earlyoom OOM protection, and diagnostics on NVIDIA DGX Spark systems.
</Purpose>

<Use_When>
- User wants to install or update sparkrun
- User wants to create, modify, or manage cluster configurations
- User wants to set up SSH for multi-node inference
- User wants to configure CX7 network interfaces
- User wants to fix file permissions on cluster hosts
- User wants to clear page cache on cluster hosts
- User wants to set up Docker group or earlyoom
- User asks about sparkrun configuration or setup
- User is getting started with DGX Spark inference for the first time
</Use_When>

<Do_Not_Use_When>
- User wants to run, stop, or monitor workloads -- use the run skill instead
- User wants to manage recipe registries or create recipes -- use the registry skill instead
</Do_Not_Use_When>

<Steps>

## Installation

```bash
# Ensure that uv is installed
uv --version

# uv can be installed with (IF NEEDED)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install sparkrun as a CLI tool
uvx sparkrun setup install

# Update sparkrun + registries (top-level shortcut)
sparkrun update

# Update to latest version (and registries)
sparkrun setup update

# Update sparkrun only (skip registry sync)
sparkrun setup update --no-update-registries
```

## Setup Wizard (Recommended for First-Time Setup)

The wizard handles all setup steps in a single guided flow:

```bash
# Interactive wizard (auto-launches when no default cluster exists)
sparkrun setup wizard

# Pre-populate hosts and cluster name
sparkrun setup wizard --hosts <ip1>,<ip2> --cluster mylab

# Non-interactive (accept all defaults)
sparkrun setup wizard --yes --hosts <ip1>,<ip2>

# Dry-run preview
sparkrun setup wizard --dry-run --hosts <ip1>,<ip2>
```

The wizard performs these phases:
1. **Cluster Setup** -- detects CX7 peers, creates cluster, sets default
2. **SSH Mesh** -- passwordless SSH across all hosts + control machine
3. **CX7 Configuration** -- high-speed networking (if CX7 detected)
4. **Docker Group** -- ensures user can run Docker without sudo
5. **Sudoers Entries** -- scoped sudoers for fix-permissions and clear-cache
6. **earlyoom** -- OOM protection to prevent system hangs

Running `sparkrun setup` with no subcommand auto-launches the wizard when no default cluster is configured.

## Cluster Management

Clusters are named host groups saved in `~/.config/sparkrun/clusters/`.

```bash
# Create a cluster (first host = head node)
sparkrun cluster create <name> --hosts <ip1>,<ip2>,... [-d "description"] [--user <ssh_user>]
sparkrun cluster create <name> --hosts <ips> --transfer-mode push --transfer-interface cx7

# Set as default (used when --hosts/--cluster not specified)
sparkrun cluster set-default <name>

# View clusters
sparkrun cluster list
sparkrun cluster show <name>
sparkrun cluster default

# Modify
sparkrun cluster update <name> --hosts <new_hosts> [--user <user>] [-d "desc"]
sparkrun cluster update <name> --add-host 10.0.0.5
sparkrun cluster update <name> --add-host 10.0.0.5,10.0.0.6
sparkrun cluster update <name> --remove-host 10.0.0.2
sparkrun cluster update <name> --transfer-mode push --transfer-interface cx7
sparkrun cluster delete <name>
sparkrun cluster unset-default
```

### Cluster Options

| Option | Description |
|--------|-------------|
| `--hosts, -H` | Comma-separated host list |
| `--hosts-file` | File with hosts (one per line) |
| `--user, -u` | SSH username for this cluster |
| `--cache-dir` | HuggingFace cache directory for this cluster |
| `--transfer-mode` | Resource transfer mode (auto, local, push, delegated) |
| `--transfer-interface` | Network interface for transfers (auto, cx7, mgmt) |
| `--add-host` | Add host(s) to the cluster (repeatable, comma-ok) |
| `--remove-host` | Remove host(s) from the cluster (repeatable, comma-ok) |

## SSH Setup

Multi-node inference requires passwordless SSH between all hosts. sparkrun bundles a mesh setup script.

```bash
# Set up SSH mesh across cluster hosts (interactive -- prompts for passwords)
sparkrun setup ssh --cluster <name>
sparkrun setup ssh --hosts <ip1>,<ip2> [--user <username>]

# Include extra hosts (e.g. control machine) in the mesh
sparkrun setup ssh --cluster <name> --extra-hosts <control_ip>

# Exclude the local machine from the mesh
sparkrun setup ssh --cluster <name> --no-include-self

# Dry-run to see what would happen
sparkrun setup ssh --cluster <name> --dry-run
```

**IMPORTANT:** The SSH setup script runs interactively (prompts for passwords on first connection). Do NOT capture its output -- let it pass through to the terminal.

## CX7 Networking

Configure ConnectX-7 network interfaces on cluster hosts for high-speed transfers.

```bash
# Auto-detect CX7 interfaces and configure with defaults
sparkrun setup cx7 --cluster <name>
sparkrun setup cx7 --hosts <ip1>,<ip2>

# Override subnets
sparkrun setup cx7 --cluster <name> --subnet1 192.168.11.0/24 --subnet2 192.168.12.0/24

# Force reconfiguration and set MTU
sparkrun setup cx7 --cluster <name> --force --mtu 9000

# Dry-run
sparkrun setup cx7 --cluster <name> --dry-run
```

Requires passwordless sudo on target hosts. Will prompt for sudo password if needed.

## Docker Group Membership

Ensure the SSH user can run Docker commands without sudo.

```bash
sparkrun setup docker-group --cluster <name>
sparkrun setup docker-group --hosts <ip1>,<ip2> [--user <username>]
sparkrun setup docker-group --cluster <name> --dry-run
```

## Fix File Permissions

Fix file ownership in HuggingFace cache directories on cluster hosts.

```bash
# Fix permissions on default cache directory
sparkrun setup fix-permissions --cluster <name>

# Custom cache directory
sparkrun setup fix-permissions --cluster <name> --cache-dir /data/hf-cache

# Install sudoers entry for passwordless future runs
sparkrun setup fix-permissions --cluster <name> --save-sudo

# Dry-run
sparkrun setup fix-permissions --cluster <name> --dry-run
```

## Clear Page Cache

Drop the Linux page cache on cluster hosts to free memory for inference.

```bash
sparkrun setup clear-cache --cluster <name>
sparkrun setup clear-cache --cluster <name> --save-sudo
sparkrun setup clear-cache --cluster <name> --dry-run
```

## earlyoom OOM Protection

Install earlyoom on cluster hosts to prevent system hangs from memory pressure.

```bash
sparkrun setup earlyoom --cluster <name>
sparkrun setup earlyoom --hosts <ip1>,<ip2> [--user <username>]
sparkrun setup earlyoom --cluster <name> --dry-run
```

## Diagnostics

Collect diagnostic information from cluster hosts (hidden command, useful for debugging).

```bash
sparkrun setup diagnose --cluster <name>
sparkrun setup diagnose --cluster <name> --output diag.json
sparkrun setup diagnose --cluster <name> --json   # JSON to stdout
sparkrun setup diagnose --cluster <name> --sudo   # include sudo-level checks
```

## Configuration

Config file: `~/.config/sparkrun/config.yaml`

Key settings:
- `cluster.hosts`: Default host list (used when no --hosts/--cluster given)
- `ssh.user`: Default SSH username
- `ssh.key`: Path to SSH private key
- `ssh.options`: Additional SSH options list
- `cache_dir`: sparkrun cache directory (default: `~/.cache/sparkrun`)
- `hf_cache_dir`: HuggingFace cache directory (default: `~/.cache/huggingface`)

</Steps>

<Tool_Usage>
Use the `sparkrun_exec` tool for all sparkrun commands.

When running SSH setup, the command is interactive and must be run with inherited stdio -- use the shell tool directly for `sparkrun setup ssh` instead of `sparkrun_exec`, as it requires terminal interaction.
</Tool_Usage>

<Important_Notes>
- The **setup wizard** is recommended for first-time setup -- it handles all steps automatically
- Always create a cluster and set it as default for the user's lab setup
- The first host in a cluster is the **head node** for multi-node jobs
- SSH mesh must be set up before multi-node inference will work
- `sparkrun setup ssh` is interactive -- let it pass through to the terminal
- DGX Spark systems have 1 GPU per host, so `tensor_parallel` maps to node count
- `uv` is the recommended Python package manager; install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- `sparkrun setup cx7` requires passwordless sudo; use `--force` to reconfigure already-valid hosts
- `sparkrun setup fix-permissions` and `clear-cache` try non-interactive sudo first, then prompt if needed
- Use `--save-sudo` to install scoped sudoers entries for passwordless future runs
- `sparkrun update` is a top-level shortcut that upgrades sparkrun (if uv-installed) and updates registries
- Cluster `--transfer-mode` options: `auto` (default), `local` (no transfer), `push` (head pushes to workers), `delegated` (workers pull)
- Cluster `--transfer-interface` options: `auto` (default), `cx7` (use CX7 IPs), `mgmt` (use management IPs)
- Use `--add-host` / `--remove-host` for incremental cluster changes instead of replacing the full host list
</Important_Notes>

Task: {{ARGUMENTS}}
