# Security

The trust model for recipes, hooks, and registries; what the B-workstream
security fixes actually changed.

## Recipe trust model

A recipe is **trusted** when any of the following holds (see
`core/launcher.py:resolve_recipe_trust`):

1. The user passed `--trust` on the CLI (hidden flag, default off).
2. The recipe was loaded from a local path (no `source_registry` recorded —
   files passed on the CLI, `./recipes/`, `~/.config/sparkrun/recipes/`).
3. The recipe came from a registry whose URL is in
   `core/registry.py:DEFAULT_REGISTRIES_GIT`:
   - `https://github.com/dbotwinick/sparkrun-recipe-registry.git`
   - `https://github.com/spark-arena/recipe-registry.git`
   - `https://github.com/spark-arena/community-recipe-registry.git`

A recipe is **untrusted** otherwise — typically a third-party registry the user
added via `sparkrun registry add <url>`.

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
