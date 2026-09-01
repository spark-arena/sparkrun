# Security

The trust model for recipes, hooks, and registries; what the B-workstream
security fixes actually changed.

## Recipe trust model

Trust is a **per-registry** local decision, stored in
`~/.config/sparkrun/registries.yaml` as a boolean `trusted:` field on each
entry (see `RegistryEntry.trusted` in `core/registry.py`).

A recipe is **trusted** when any of the following holds (see
`core/launcher.py:resolve_recipe_trust`):

1. The user passed `--trust` on the CLI (hidden flag, default off).
2. The recipe was loaded from a local path (no `source_registry` recorded —
   files passed on the CLI, `./recipes/`, `~/.config/sparkrun/recipes/`).
3. The recipe came from a registry whose `trusted` flag is `true` in the
   user's local `registries.yaml`.

A recipe is **untrusted** otherwise — typically a third-party registry the user
added via `sparkrun registry add <url>` without the `--trust` flag, or any
registry whose name cannot be resolved against the local `registries.yaml`.

### Where the trust bit comes from

- **Default registries**: every entry shipped in
  `core/registry.py:FALLBACK_DEFAULT_REGISTRIES` declares `trusted=True` on
  the entry itself.  All built-in defaults are first-party recipe sources, so
  all of them ship trusted — including `eugr` and `atlas`, which are not
  bootstrap-discovery URLs.

  Trust is declared **per entry**, not derived from `BOOTSTRAP_REGISTRY_URLS`
  (that list exists for bootstrap-time manifest discovery and deliberately
  differs).  `FALLBACK_DEFAULT_REGISTRIES` is the single source of truth for
  "which registries ship trusted"; `_default_trusted_urls()` exposes it to the
  migration below.

- **Bootstrap manifest discovery**: when `_init_defaults_from_manifests`
  successfully clones a bootstrap URL and reads its
  `.sparkrun/registry.yaml`, **sparkrun** marks the discovered entries
  `trusted=True` because they came in via the curated bootstrap path.
  The manifest YAML itself **cannot** grant trust — only the local
  decision (curated bootstrap URL list, explicit user opt-in) does.

- **User-added registries**: `sparkrun registry add <url>` lands new
  entries with `trusted=False`.  Pass `--trust` (or run
  `sparkrun registry trust <name>` afterwards) to opt in.

- **Migration**: when an existing `registries.yaml` predates the
  `trusted` field, sparkrun performs a one-time migration on next load,
  marking entries whose (normalized) URL matches a registry that ships
  trusted — `_default_trusted_urls()` — and leaving the rest `trusted=False`.
  Comparison strips a trailing `/` and `.git`, since `eugr`'s default URL
  carries no `.git` suffix while the others do.

  Deriving this from the default list rather than `BOOTSTRAP_REGISTRY_URLS`
  is deliberate: otherwise marking a registry trusted would reach only fresh
  installs, and anyone upgrading from a pre-trust config would silently keep
  it untrusted.  A user who has **already** migrated keeps whatever their
  `registries.yaml` says — re-trusting a registry their own config marks
  untrusted is not a decision the migration makes for them.

### CLI surface

| Command                                       | Effect                                                |
|-----------------------------------------------|-------------------------------------------------------|
| `sparkrun registry add <url>`                 | Add registries from a manifest (lands `trusted=False`)|
| `sparkrun registry add --trust <url>`         | Add and immediately mark `trusted=True`               |
| `sparkrun registry trust <name>`              | Flip an existing registry to `trusted=True`           |
| `sparkrun registry untrust <name>`            | Flip back to `trusted=False`                          |
| `sparkrun registry list`                      | Includes a `Trusted` column                           |
| `sparkrun registry show <name>`               | Includes a `Trusted:` line                            |

## What trust gates

Three hook surfaces consult the trust flag (all in
`orchestration/hooks.py:_confirm_hook_execution`):

| Hook            | Where it runs                                            | Trust behavior                                              |
|-----------------|----------------------------------------------------------|-------------------------------------------------------------|
| `pre_exec`      | Inside the head container, before the serve command.     | Trusted: runs. Untrusted: interactive confirmation prompt.  |
| `post_exec`     | Inside the head container, after the port is healthy.    | Trusted: runs. Untrusted: interactive confirmation prompt.  |
| `post_commands` | On the **control machine**, after the port is healthy.   | Trusted: runs. Untrusted: interactive confirmation prompt.  |

`launcher.py:launch_inference` computes `recipe_trusted` once and passes it to
`runtime.run(...)` (which gates `pre_exec`) and to
`post_launch_lifecycle(trust=...)` (which gates `post_exec` + `post_commands`).
The same recipe gets the same answer for every surface.

## What trust gates beyond hooks

`core/launcher.py:_enforce_recipe_mount_trust` refuses these
container-escape surfaces for an **untrusted** recipe, at the single launch
choke point:

- `executor_config` privilege keys (`_TRUST_GATED_EXECUTOR_KEYS`):
  `privileged`, `cap_add`, `security_opt`, `devices`, `user`, `volumes`.
  Each maps to a `docker run` flag that defeats the rootless hardening or
  exposes host state, and each sits *above* the executor's rootless
  `apply_runtime_adjustments` layer in the resolution chain — so a recipe
  setting them would otherwise win over the hardening.
- **Executor selection** (`_TRUSTED_DEFAULT_EXECUTORS`): restricted to
  `docker`.  The rootless, namespaced container is the sandbox that justifies
  running a registry/URL recipe's serve `command` without a prompt; `local`
  runs it natively via `setsid bash -c` and `k8s` wedges it into
  `kubectl run`, either of which is arbitrary host code execution.
- The undocumented `cluster_config` launch overrides
  (`resolved_model_path` / `remote_cache_dir` / `local_cache_dir`), which
  identity-mount a host directory and repoint the serve argument at it.
- **`executor_config.ipc`** (`_UNTRUSTED_IPC_MODES`), gated on its *value*
  rather than its presence: an untrusted recipe may pick any mode that gives
  the container a fresh IPC namespace (`private`, `shareable`, `none`, or an
  empty value), but not one that reaches outside it. `host` shares the host's
  IPC namespace and `/dev/shm` — read/write access to every other tenant's
  POSIX shared memory and semaphores on that machine, including the ability to
  delete them — and `container:<name>` joins one specific other workload's
  namespace, which is a *targeted* lateral read since sparkrun container names
  are derivable from a cluster_id. The allowlist shape means an unrecognised
  or non-string value fails closed too.

  This knob was ungated while `host` was the default, when setting it changed
  nothing. It became an escalation when the default moved to a container-owned
  namespace (see [EXECUTORS.md](EXECUTORS.md#ipc-namespace-ipc-and-shm_size)):
  an untrusted recipe asking for `host` is asking to leave the sandbox that
  justifies running its serve `command` without a prompt.

Innocuous resource knobs are deliberately **not** gated: `shm_size`,
`network`, `memory_limit`, `ulimit`, `restart_policy`, `auto_remove`,
`labels`.

`utils/shell.py:assert_safe_mount_source` applies **regardless of trust**:
the host root, the Docker control socket, SSH keys, and kernel
pseudo-filesystems are refused outright, even for a trusted recipe.  It
validates the *literal* path shape (absolute, no `..`, not under a forbidden
subtree) rather than trusting control-machine `realpath`, because the mount
happens on a remote host whose symlink layout differs.

## What trust does *not* gate

Once a recipe **is** trusted — local path, a registry marked `trusted`,
`sparkrun registry trust`, or `--trust` — all of the above become available
to it with no second prompt for `cap_add: SYS_ADMIN`.

Adding and trusting a third-party registry **implies trusting its recipes'
privileged fields**. If you don't trust a registry, don't trust it.

## Git URL hardening

`core/registry.py:_validate_git_url` accepts only four URL schemes for
`sparkrun registry add` and the default-registry clone path:

- `https://...`
- `git@host:org/repo`
- `ssh://...`
- `file://...`

Anything else (e.g. `http://`, `ext::`, `--upload-pack=...`) is rejected before
`git clone` is invoked, preventing argument-injection through URL parsing.

## Reserved registry name prefixes

`core/registry.py:RESERVED_NAME_PREFIXES` (`arena`, `spark-arena`,
`sparkarena`, `sparkrun`, `official`, ...) may only be used by URLs hosted
under approved GitHub orgs (`spark-arena`, `scitrera`, `eugr`, `dbotwinick`,
`raphaelamorim`). `validate_registry_name()` enforces this — preventing
third-party repositories from impersonating an official source by claiming a
look-alike name.

## Registry name / subpath path containment

Registry names and asset subpaths arrive from `.sparkrun/registry.yaml`
manifests in **remote repositories** (`sparkrun registry add <url>`, bootstrap
manifest discovery), and both are then turned into real filesystem paths. Two
distinct primitives result if they are not contained:

- A **name** is used verbatim as a directory under the registry cache root
  (`RegistryManager._cache_dir` is `cache_root / name`). An escaping name
  (`../…`, `a/b`) resolves outside that root, and
  `_link_registry_to_shared` goes on to `shutil.rmtree` a cache dir that is not
  a link — so this is a *delete* primitive, not merely an untidy path. A name
  matching the `_url_<hash>` form reserved by `_clone_dir_for_url` is the same
  hazard aimed at the shared clone its siblings on that URL depend on.
- A **subpath** is resolved against the registry's cache dir
  (`asset_dir` is `_cache_dir(name) / subpath`) and handed to
  `git sparse-checkout set`. An escaping subpath makes `iter_asset_files`
  `rglob` a directory outside the clone, and `find_recipe` then offers whatever
  YAML it finds there as a runnable recipe — a *read* primitive that feeds the
  recipe loader.

Three validators in `core/registry.py` contain this. Both charsets require the
first character of every path component to be alphanumeric, which rules out
`.`/`..`, dotfiles, a leading `-` (which git would read as an option) and the
`_url_` prefix in one rule:

| Function | Guards |
|---|---|
| `assert_safe_registry_name(name)` | non-empty, ≤100 chars, `[A-Za-z0-9][A-Za-z0-9._-]*` |
| `assert_safe_registry_subpath(subpath, field=…)` | relative, no backslash, every `/`-segment in the same charset; empty means "asset kind not declared" |
| `assert_safe_registry_entry(entry)` | the single chokepoint — name plus all four fields in `SUBPATH_FIELDS` |

Enforcement points, and why each behaves differently:

- `validate_registry_name()` runs `assert_safe_registry_name` **first**, so an
  unsafe name is rejected on containment grounds before the namespace rule is
  consulted — `../sparkrun-x` does not *start with* a reserved prefix, so the
  namespace check alone would pass it.
- `add_registry()` additionally runs `assert_safe_registry_entry`, since
  `validate_registry_name` only sees the name and this is the public
  programmatic entry point.
- `_discover_manifest_entries()` validates every declared entry and **drops**
  unsafe ones with a warning, keeping the rest (per-entry partial success,
  matching the per-URL behavior of `_init_defaults_from_manifests`). A manifest
  with nothing left raises, so a wholly hostile manifest is never reported as a
  successful no-op add.
- `_load_registries_from_file()` **skips** unsafe entries with a warning rather
  than raising. This is deliberately narrower than the enclosing `except` in
  `_load_registries`, which discards the file and reverts to the shipped
  defaults: one bad entry — a hand-edit, a merge, a manifest read by an older
  build with no charset check — must not take the user's other registries with
  it. The *namespace* check is deliberately not applied on load, since it gates
  adding a registry and would otherwise invalidate an existing config
  retroactively.

Manifest discovery clones blob-filtered and sparse (`--filter=blob:none
--sparse` + `sparse-checkout set .sparkrun`): only the manifest is ever read, so
the recipe trees are never fetched. A failed sparse-checkout raises rather than
being reported as "no manifest found", so a clone whose manifest directory was
never materialized cannot be mistaken for a repo that declares nothing.

## SSH / shell command construction

`utils/shell.py` is the canonical place for shell-string assembly:

- `quote()` — wraps `shlex.quote()`. Every command string interpolation passes
  through this.
- `validate_unix_username(user)` — used before any `sudoers` / sudo script
  interpolation in `cli/_setup/`. Rejects strings outside POSIX usernames so
  installer scripts can't be steered into arbitrary file paths.
- `b64_encode_cmd()` / `b64_wrap_bash()` — base64 wrappers for serve commands
  that contain embedded newlines, single quotes, or unicode.

## Delegated copy validation

`orchestration/transfer.py:_run_delegated_copy` validates both ends of
delegated rsync transfers:

- `source_host` is matched against the validated host list before any SSH
  invocation runs against it.
- `dest` is rejected when the resolved path escapes the cache root (basic
  traversal containment).

## trtllm host-key strictness

`runtimes/trtllm.py` no longer relaxes SSH host-key checking inside the rsh
wrapper for MPI multi-node. Operators are expected to seed `known_hosts` via
`sparkrun setup ssh-mesh` (or equivalent) — strict checking now applies to
every leg.

## Sudo-user validation

`cli/_setup/_sudo.py` + `_phases.py` + `_uninstall.py` call
`validate_unix_username()` on every `sudo_user` value before it is interpolated
into sudoers fragments or script bodies. Combined with the `auth_proxy` CORS
tightening (limited to `AUTH_PROXY_BASE`, no wildcards) and the removal of
prefix token logging from debug paths, the setup surface no longer trusts
operator-supplied identifiers verbatim.

## CORS / OAuth proxy

The OAuth callback CORS allowlist is restricted to `AUTH_PROXY_BASE`. Token
prefixes (the first N chars of a bearer token) are no longer emitted in debug
logs.

## Operator checklist

When adding a third-party registry:

1. Inspect `recipes/*.yaml` for `pre_exec`, `post_exec`, `post_commands`,
   `executor_config.cap_add`, `devices`, `security_opt`.
2. Confirm the registry URL matches one of the approved schemes
   (`https://...`, `git@...`, `ssh://...`, `file://...`).
3. Run untrusted recipes with `--dry-run` first; the interactive trust prompt
   makes the per-launch posture explicit.
4. Use `--trust` only when you've reviewed the recipe and intend to run its
   privileged content.
