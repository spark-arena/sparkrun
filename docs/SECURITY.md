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

- **Default registries**: bootstrap-curated entries shipped via
  `core/registry.py:FALLBACK_DEFAULT_REGISTRIES` are marked `trusted=True`
  when their URL is in `BOOTSTRAP_REGISTRY_URLS`:
  - `https://github.com/dbotwinick/sparkrun-recipe-registry.git`
  - `https://github.com/spark-arena/recipe-registry.git`
  - `https://github.com/spark-arena/community-recipe-registry.git`

  This preserves out-of-the-box behavior for first-run installs.  Default
  entries whose URLs are not in `BOOTSTRAP_REGISTRY_URLS` (currently
  `eugr`, `atlas`) ship `trusted=False`.

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
  marking entries whose URL is in `BOOTSTRAP_REGISTRY_URLS` as
  `trusted=True` and leaving the rest `trusted=False`.

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

## What trust does *not* gate

Recipe-driven privileged fields are **not** allowlisted by the trust model:

- `executor_config.cap_add`, `devices`, `security_opt`, `privileged`,
  `ulimit`, `user`, `network` — Docker executor pass-through.
- `executor_config.extra_opts` (Local), `k8s_*` fields (K8s).
- Bind-mounts derived from `volumes` / cache-dir resolution.

Adding a third-party registry **implies trusting its recipes' privileged
fields**. There is no second prompt for `cap_add: SYS_ADMIN`. If you don't
trust a registry, don't add it.

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
