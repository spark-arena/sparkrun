# /sparkrun:setup

Install sparkrun and configure a DGX Spark cluster.

## Usage

```
/sparkrun:setup
```

## Behavior

When this command is invoked, walk the user through setup:

### Step 1: Check if sparkrun is installed

```bash
which sparkrun && sparkrun --version
```

If not installed, install it:

```bash
uvx sparkrun setup install
```

This creates a managed virtual environment, installs sparkrun, and sets up shell tab-completion.

### Step 2: Run the setup wizard (recommended)

The wizard handles cluster creation, SSH mesh, CX7 networking, Docker group, sudoers, and earlyoom in one guided flow:

```bash
sparkrun setup wizard
sparkrun setup wizard --hosts <ip1>,<ip2> --cluster mylab
sparkrun setup wizard --yes --hosts <ip1>,<ip2>    # non-interactive
sparkrun setup wizard --dry-run                     # preview
```

The wizard auto-detects CX7 interfaces and discovers peer DGX Sparks on the local network.

Running `sparkrun setup` with no subcommand also auto-launches the wizard when no default cluster is configured.

### Alternative: Manual step-by-step setup

If the user prefers manual control, each step can be run individually:

#### Create a cluster

```bash
sparkrun cluster create <name> --hosts <ip1>,<ip2>,... [-d "description"] [--user <ssh_user>]
sparkrun cluster set-default <name>
```

#### SSH mesh

```bash
sparkrun setup ssh --cluster <name>
```

**IMPORTANT:** This command is interactive (prompts for passwords). Do NOT capture its output.

#### CX7 networking (optional)

```bash
sparkrun setup cx7 --cluster <name>
```

#### Docker group membership

```bash
sparkrun setup docker-group --cluster <name>
```

Ensures the SSH user can run Docker without sudo.

#### Fix file permissions (optional)

```bash
sparkrun setup fix-permissions --cluster <name>
sparkrun setup fix-permissions --cluster <name> --save-sudo
```

#### Clear page cache (optional)

```bash
sparkrun setup clear-cache --cluster <name>
sparkrun setup clear-cache --cluster <name> --save-sudo
```

#### earlyoom OOM protection (optional)

```bash
sparkrun setup earlyoom --cluster <name>
```

Installs earlyoom on cluster hosts to prevent system hangs from memory pressure.

### Step 3: Verify

```bash
sparkrun cluster show <name>
sparkrun list
sparkrun show <recipe> --tp <N>
```

### Step 4: Update

```bash
# Update sparkrun + registries in one command
sparkrun update

# Or update just registries
sparkrun registry update

# Or update sparkrun only
sparkrun setup update --no-update-registries
```

## Notes

- `uv` is the recommended Python package manager; install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- The first host in a cluster is the head node for multi-node jobs
- DGX Spark has 1 GPU per host, so tensor_parallel = number of hosts
- SSH user defaults to current OS user; override with `--user`
- Use `sparkrun setup update` to update sparkrun and registries to the latest version
- `sparkrun update` is a top-level alias that upgrades sparkrun (if installed via uv) and updates registries
