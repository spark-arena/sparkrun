"""All setup subcommand definitions for sparkrun."""

from __future__ import annotations

from sparkrun.core.application_profile import render_identity_text

import sys

import click

from sparkrun.core.application_profile import get_application_profile

from .._common import (
    _detect_shell,
    _get_cluster_manager,
    _require_uv,
    _get_context,
    resolve_cluster_config,
    _resolve_setup_context,
    _shell_rc_file,
    dry_run_option,
    host_options,
    json_option,
    print_json,
    HIDE_ADVANCED_OPTIONS,
)
from . import setup
from ._phases import (
    EARLYOOM_PREFER_PATTERNS,
    EARLYOOM_AVOID_PATTERNS,
    _build_earlyoom_regex,
    _earlyoom_summary,
)
from ._ssh import _run_ssh_mesh, _check_ssh_host_identities
from ._sudo import _record_setup_phase, host_password_prompt, interactive_sudo_passwords, run_sudo_action, save_sudoers_entry
from sparkrun.orchestration.job_metadata import PRUNE_KEEP_PER_INTENT, PRUNE_MAX_AGE_DAYS
from sparkrun.utils.text import excerpt


@setup.command("completion", hidden=True)
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), default=None, help="Shell type (auto-detected if not specified)")
@click.pass_context
def setup_completion(ctx, shell):
    """Install shell tab-completion for sparkrun.

    Detects your current shell and appends the completion setup to
    your shell config file (~/.bashrc, ~/.zshrc, or ~/.config/fish/...).

    Examples:

      {app_command} setup completion

      {app_command} setup completion --shell bash
    """
    if not shell:
        shell, rc_file = _detect_shell()
    else:
        rc_file = _shell_rc_file(shell)

    # Resolve the absolute path to the installed `sparkrun` so the completion
    # snippet doesn't depend on PATH ordering in the shell rc file (issue #198).
    # shutil.which() returns what the shell itself resolves -- typically the
    # stable ~/.local/bin/sparkrun shim -- rather than an internal uv
    # version-specific venv path. Fall back to the bare command if unresolved.
    import shutil

    _install_completion(shell, rc_file, shutil.which(get_application_profile().command))


def _install_completion(shell, rc_file, executable, *, replace_existing=False):
    """Install completion, optionally refreshing previously generated entries.

    *executable* is embedded as given (quoted); ``None`` embeds the bare command.
    Only callers with a persistent executable should replace existing entries;
    the standalone completion command's PATH lookup may resolve into uvx's cache.
    """
    import shlex

    from sparkrun.utils.shell import quote

    completion_var = "_%s_COMPLETE" % get_application_profile().command.upper().replace("-", "_")
    sparkrun_cmd = quote(str(executable) if executable else get_application_profile().command)

    if shell in ("bash", "zsh"):
        prefix, suffix = 'eval "$(%s=%s_source ' % (completion_var, shell), ')"'
    elif shell == "fish":
        prefix, suffix = "%s=fish_source " % completion_var, " | source"
    else:
        raise click.BadParameter("Unsupported shell: %s" % shell, param_hint="--shell")
    snippet = prefix + sparkrun_cmd + suffix

    if rc_file.exists():
        contents = rc_file.read_text()
        if replace_existing and executable:
            lines = contents.splitlines(keepends=True)
            for index, line in enumerate(lines):
                stripped = line.strip()
                if not (stripped.startswith(prefix) and stripped.endswith(suffix)):
                    continue
                old_command = stripped[len(prefix) : -len(suffix)]
                try:
                    words = shlex.split(old_command)
                except ValueError:
                    continue
                # Match the exact single-executable form we generate. Leave
                # custom arguments, expansions, and compound shell code alone.
                if len(words) == 1 and quote(words[0]) == old_command:
                    lines[index] = line.replace(stripped, snippet, 1)
            updated = "".join(lines)
            if updated != contents:
                rc_file.write_text(updated)
                click.echo("Completion updated for %s in %s" % (shell, rc_file))
                click.echo("Restart your shell or run: source %s" % rc_file)
                return
        if completion_var in contents:
            click.echo("Completion already configured in %s" % rc_file)
            return

    # Ensure parent directory exists (for fish)
    rc_file.parent.mkdir(parents=True, exist_ok=True)

    with open(rc_file, "a") as f:
        f.write("\n# %s tab-completion\n" % get_application_profile().command)
        f.write(snippet + "\n")

    click.echo("Completion installed for %s in %s" % (shell, rc_file))
    click.echo("Restart your shell or run: source %s" % rc_file)


@setup.command("install")
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), default=None, help="Shell type (auto-detected if not specified)")
@click.option("--no-update-registries", is_flag=True, help="Skip updating recipe registries after installation")
@click.option("--stable", is_flag=True, help="Install the stable channel from its configured source")
@click.option("--beta", is_flag=True, help="Install the beta channel from its configured source")
@click.option("--alpha", is_flag=True, help="Install the alpha channel from its configured source")
@click.option("--yolo", is_flag=True, help="Alias for --alpha")
@click.pass_context
def setup_install(ctx, shell, no_update_registries, stable, beta, alpha, yolo):
    """Install {app_command} and tab-completion.

    Requires uv (https://docs.astral.sh/uv/).  Typical usage:

    \b
      uvx {app_command} setup install

    This installs {app_command} as a uv tool (real binary on PATH), cleans up
    any old aliases/functions from previous installs, configures
    tab-completion, and updates recipe registries.

    Channel flags select a configured distribution source and persist the
    choice for future updates. No flag keeps the configured channel.
    """
    import subprocess

    from sparkrun.cli._self_update import (
        channel_from_flags,
        install_argv,
        installation_prefix,
        environment_executable,
        update_registries_after_core,
    )
    from sparkrun.core.channels import CHANNEL_STABLE

    config = _get_context(ctx).config
    requested = channel_from_flags(stable, beta, alpha, yolo)
    channel = requested if requested is not None else config.self_update_channel

    if not shell:
        shell, rc_file = _detect_shell()
    else:
        rc_file = _shell_rc_file(shell)

    # Step 1: Install sparkrun via uv tool
    uv = _require_uv()

    label = "" if channel == CHANNEL_STABLE else "%s channel " % channel
    click.echo(f"Installing {get_application_profile().command} %svia uv tool install..." % label)
    from sparkrun.core.runtime_lease import guarded_update

    try:
        target_prefix = installation_prefix(uv)
        result = guarded_update(
            config,
            subprocess.run,
            install_argv(uv, channel),
            target_prefix=target_prefix,
            capture_output=True,
            text=True,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    if result.returncode != 0:
        click.echo(f"Error installing {get_application_profile().command}: %s" % result.stderr.strip(), err=True)
        sys.exit(1)
    config.set_self_update_channel(channel)
    click.echo(f"{get_application_profile().command} %sinstalled on PATH" % label)

    # Step 2: Clean up old aliases/functions from previous installs
    if rc_file.exists():
        old_markers = [
            "alias %s=" % get_application_profile().command,
            "alias %s " % get_application_profile().command,
            "function %s" % get_application_profile().command,
            "%s()" % get_application_profile().command,
        ]
        contents = rc_file.read_text()
        lines = contents.splitlines(keepends=True)
        cleaned = [line for line in lines if not any(m in line for m in old_markers)]
        if len(cleaned) != len(lines):
            rc_file.write_text("".join(cleaned))
            click.echo(f"Cleaned up old {get_application_profile().command} aliases from %s" % rc_file)

    # Step 3: Install tab-completion.  Embed the executable just installed, not
    # the PATH lookup setup_completion does: under uvx the first `sparkrun` on
    # PATH is uvx's throwaway environment in the uv cache.
    _install_completion(shell, rc_file, environment_executable(target_prefix, get_application_profile().command), replace_existing=True)

    # Use the resulting environment, which may differ from a uvx caller.
    if not no_update_registries:
        update_registries_after_core(
            config.config_path,
            executable=environment_executable(target_prefix, get_application_profile().command),
        )


@setup.command("update")
@click.option("--no-update-plugins", is_flag=True, help="Audit compatibility without fetching or updating plugins")
@click.option("--no-update-registries", is_flag=True, help="Skip updating recipe registries")
@click.option("--stable", is_flag=True, help="Switch to and update the stable channel from its configured source")
@click.option("--beta", is_flag=True, help="Switch to and update the beta channel from its configured source")
@click.option("--alpha", is_flag=True, help="Switch to and update the alpha channel from its configured source")
@click.option("--yolo", is_flag=True, help="Alias for --alpha")
@click.pass_context
def setup_update(ctx, no_update_registries, stable, beta, alpha, yolo, no_update_plugins=False):
    """Update {app_command}, managed plugins, and recipe registries (same as 'update').

    With no channel flag, updates the currently configured channel; a channel
    flag switches channels. A non-uv install skips the self-upgrade; a failed
    upgrade exits non-zero.

    \b
      {app_command} setup update
      {app_command} setup update --beta
    """
    from sparkrun.cli._plugins import management_config
    from sparkrun.cli._self_update import channel_from_flags, run_update

    config = management_config(ctx)
    status = run_update(
        config,
        command=f"{get_application_profile().command} setup update",
        requested=channel_from_flags(stable, beta, alpha, yolo),
        update_plugins=not no_update_plugins,
        update_registries=not no_update_registries,
    )
    ctx.exit(status)


@setup.command("version", hidden=True)
@json_option(help="Emit machine-readable application, core, and plugin version information")
@click.pass_context
def setup_version(ctx, output_json):
    """Show application, core, and discovered plugin versions without loading plugins.

    Includes the application profile, release channel, and installed Git
    commits when available. The self-update flow uses --json to compare builds.
    """
    from sparkrun.core.version import version_diagnostics

    from sparkrun.cli._plugins import management_config

    data = version_diagnostics(management_config(ctx))
    if output_json:
        print_json(data)
        return

    def package_identity(identity):
        label = "%s %s" % (identity["package"], identity["version"])
        if identity.get("commit"):
            label += " (commit %s)" % identity["commit"]
        return label

    click.echo("Application: %s" % package_identity(data["distribution"]))
    click.echo("Profile:     %s" % data["distribution"]["id"])
    click.echo("Channel:     %s" % data["channel"])
    click.echo("Core:        %s" % package_identity(data["core"]))
    click.echo()
    click.echo("Plugins:")
    if not data["plugins"]:
        click.echo("  (none)")
    for plugin in data["plugins"]:
        details = [value for value in (plugin.get("package") or plugin.get("module"), plugin.get("source")) if value]
        if plugin.get("required"):
            details.append("required")
        suffix = " (%s)" % "; ".join(details) if details else ""
        click.echo("  %s: %s%s" % (plugin["name"], plugin.get("version") or "unknown", suffix))


@setup.command("telemetry")
@click.option("--enable", "enable", is_flag=True, help="Enable anonymous telemetry in config")
@click.option("--disable", "disable", is_flag=True, help="Disable anonymous telemetry in config")
@click.pass_context
def setup_telemetry(ctx, enable, disable):
    """View or change anonymous telemetry settings."""
    from sparkrun.telemetry.config import (
        NO_TELEMETRY_ENV,
        env_telemetry_override,
        persistent_telemetry_setting,
        set_persistent_telemetry,
        telemetry_enabled,
    )

    if enable and disable:
        raise click.ClickException("Choose only one of --enable or --disable.")

    sctx = _get_context(ctx)
    config = sctx.config
    if enable:
        set_persistent_telemetry(config, True)
        click.echo("Telemetry enabled in config.")
    elif disable:
        set_persistent_telemetry(config, False)
        click.echo("Telemetry disabled in config.")

    effective = telemetry_enabled(config)
    persisted = persistent_telemetry_setting(config)
    env_override = env_telemetry_override()

    persisted_text = "default enabled" if persisted is None else ("enabled" if persisted else "disabled")
    click.echo("Telemetry: %s" % ("enabled" if effective else "disabled"))
    click.echo("Config:    %s" % persisted_text)
    if env_override is not None:
        click.echo("Env:       %s" % ("enabled" if env_override else "disabled"))
    else:
        click.echo("Env:       set %s=1 to opt out for one process" % NO_TELEMETRY_ENV)


SETUP_FEATURES_FLAG = "cli.setup.features"


def _setup_features_visible_at_import() -> bool:
    """Resolve --help visibility for the 'setup features' group.

    Visibility-only: the group is always functional, this just decides whether
    it appears in ``setup --help`` (on by default for beta/alpha channels).
    """
    try:
        from sparkrun.core.config import SparkrunConfig

        return SparkrunConfig().is_feature_enabled(SETUP_FEATURES_FLAG)
    except Exception:  # noqa: BLE001 — never let a config read break CLI import
        return False


@setup.group("features", invoke_without_command=True, hidden=not _setup_features_visible_at_import())
@click.pass_context
def setup_features(ctx):
    """View and toggle advanced feature flags.

    With no subcommand, list feature flags and their effective state.

    Feature flags gate experimental capabilities (e.g. the ``local`` and
    ``k8s`` executors). Each flag has a per-channel default; explicit
    overrides written here take precedence. See ``setup features list``.

    Always functional; hidden from ``setup --help`` unless the
    ``cli.setup.features`` flag resolves on (default: on for beta/alpha).
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(setup_features_list)


def _feature_state(config, flag):
    """Return ``(enabled, source, override)`` for *flag* under *config*."""
    from sparkrun.core.features import feature_source

    enabled = config.is_feature_enabled(flag.name)
    source = feature_source(flag.name, config=config)
    override = config.feature_override(flag.name)
    return enabled, source, override


@setup_features.command("list")
@json_option()
@click.option("--all", "include_all", is_flag=True, help="Include disabled flags and setup-step flags (default: enabled flags only).")
@click.pass_context
def setup_features_list(ctx, output_json, include_all):
    """List enabled feature flags and their effective state.

    Flags that resolve off, and the ``setup.steps.*`` flags, are omitted unless
    --all is given (in --json output too).
    """
    from sparkrun.core.features import all_features

    config = _get_context(ctx).config
    channel = config.feature_channel
    from sparkrun.core.features import FEATURE_OWNERS

    states = {flag.name: _feature_state(config, flag) for flag in all_features()}
    flags = [flag for flag in all_features() if include_all or (states[flag.name][0] and not flag.name.startswith("setup.steps."))]
    hidden_off = (
        0 if include_all else sum(1 for flag in all_features() if not states[flag.name][0] and not flag.name.startswith("setup.steps."))
    )

    if output_json:
        # Preserve `channel` as core maturity; application release policy has
        # its own channel and provenance in the same diagnostic row.
        print_json(
            [
                {
                    "name": flag.name,
                    "owner": getattr(FEATURE_OWNERS.get(flag.name), "plugin_id", "core"),
                    "description": flag.description,
                    "enabled": enabled,
                    "source": source,
                    "override": override,
                    "channel": channel,
                    "application_channel": config.self_update_channel,
                }
                for flag, (enabled, source, override) in ((f, states[f.name]) for f in flags)
            ]
        )
        return

    hidden_note = "%d disabled feature(s) hidden; use --all to include them." % hidden_off if hidden_off else None
    if not flags:
        click.echo("No enabled feature flags." if hidden_off else "No feature flags registered.")
        if hidden_note:
            click.echo(hidden_note)
        return

    click.echo("Feature channel: %s" % channel)
    if config.profile.feature_channel_defaults:
        click.echo("Application channel: %s" % config.self_update_channel)
    click.echo("")
    table = [("FEATURE", "STATE", "SOURCE", "OWNER", "DESCRIPTION")]
    for flag in flags:
        enabled, source, _override = states[flag.name]
        state = "on" if enabled else "off"
        owner = getattr(FEATURE_OWNERS.get(flag.name), "plugin_id", "core")
        table.append((flag.name, state, source, owner, flag.description))
    widths = [max(len(cell) for cell in column) for column in zip(*table, strict=True)]
    for cells in table:
        click.echo("  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip())
    if hidden_note:
        click.echo("")
        click.echo(hidden_note)


def _require_known_flag(name):
    """Return the :class:`FeatureFlag` for *name* or raise a ClickException."""
    from sparkrun.core.features import all_features, get_feature

    flag = get_feature(name)
    if flag is None:
        known = ", ".join(f.name for f in all_features()) or "(none)"
        raise click.ClickException("Unknown feature flag %r. Known flags: %s" % (name, known))
    return flag


def _change_feature(ctx, name, enabled):
    from sparkrun.cli._plugins import management_config, _show_result
    from sparkrun.core.plugin_lifecycle import change_feature
    from sparkrun.core.plugin_manifest import PluginError

    try:
        _show_result(change_feature(management_config(ctx), name, enabled))
    except (PluginError, OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@setup_features.command("enable")
@click.argument("name")
@click.pass_context
def setup_features_enable(ctx, name):
    """Enable capability NAME and run its explicit enable hook.

    A gateway feature also selects that provider for subsequent starts.
    Repeating the command repeats the idempotent hook. The owning plugin must
    already be enabled; environment overrides cannot be bypassed.
    """
    _change_feature(ctx, name, True)


@setup_features.command("disable")
@click.argument("name")
@click.pass_context
def setup_features_disable(ctx, name):
    """Disable capability NAME without disabling its plugin or stopping services."""
    _change_feature(ctx, name, False)


@setup_features.command("reset")
@click.argument("name")
@click.pass_context
def setup_features_reset(ctx, name):
    """Clear NAME's override; restore defaults without activation hooks or selection changes."""
    _change_feature(ctx, name, None)


def _run_ssh_diagnose(host_list, user, local_user):
    """Run SSH diagnostics on each host and render them (``api.setup.diagnose_ssh``).

    Uses interactive password-based SSH (ControlMaster) to connect, then
    collects permission info, sshd settings, and tests pubkey auth
    independently.  Prints structured pass/fail results with remediation.
    """
    from sparkrun.api.setup import (
        SSH_DIAG_COLLECT_START,
        SSH_DIAG_CONNECT_FAILED,
        SSH_DIAG_CONNECT_START,
        SSH_DIAG_HOST_DONE,
        SSH_DIAG_KEY_CHECK_DONE,
        SSH_DIAG_KEY_CHECK_SKIPPED,
        SSH_DIAG_KEY_CHECK_START,
        SSH_DIAG_LOCAL_KEY,
        SSH_DIAG_PUBKEY_DONE,
        SSH_DIAG_PUBKEY_START,
        diagnose_ssh,
    )

    cross_user = user != local_user

    click.echo("=== SSH Diagnostics ===")
    click.echo("SSH user: %s" % user)
    click.echo("Local user: %s" % local_user)
    click.echo("Cross-user: %s" % ("yes" if cross_user else "no"))
    click.echo("Hosts: %s" % ", ".join(host_list))
    click.echo()

    def _render(kind, report, d):
        if kind == SSH_DIAG_LOCAL_KEY:
            key = report.local_pubkey
            if key:
                click.echo("Local public key (%s): %s ...%s" % (report.local_pubkey_source, key[:30], key[-20:]))
            else:
                click.echo("WARNING: No local SSH public key found!")
            click.echo()
        elif kind == SSH_DIAG_PUBKEY_START:
            click.echo("--- Diagnosing %s@%s ---" % (user, d.host))
            click.echo("  [1/4] Testing pubkey authentication (BatchMode)...")
        elif kind == SSH_DIAG_PUBKEY_DONE:
            click.echo("        %s" % ("PASS" if d.pubkey_ok else "FAIL"))
            if d.ssh_debug_lines:
                click.echo("        SSH debug (key-related):")
                for line in d.ssh_debug_lines:
                    click.echo("          %s" % line)
        elif kind == SSH_DIAG_CONNECT_START:
            click.echo("  [2/4] Establishing authenticated connection (may prompt for password)...")
        elif kind == SSH_DIAG_CONNECT_FAILED:
            click.echo("        FAIL — cannot connect to %s (even with password)" % d.host)
            click.echo()
        elif kind == SSH_DIAG_COLLECT_START:
            click.echo("  [3/4] Collecting remote diagnostics...")
        elif kind == SSH_DIAG_KEY_CHECK_START:
            click.echo("  [4/4] Checking if local pubkey is in authorized_keys...")
        elif kind == SSH_DIAG_KEY_CHECK_DONE:
            click.echo("        Key present: %s" % ("yes" if d.key_installed else "NO"))
        elif kind == SSH_DIAG_KEY_CHECK_SKIPPED:
            click.echo("  [4/4] Key check: %s" % ("skipped (same user)" if not cross_user else "skipped (no local key)"))
        elif kind == SSH_DIAG_HOST_DONE:
            _render_ssh_host_diagnosis(d, report.local_pubkey_source)

    report = diagnose_ssh(host_list, user, local_user=local_user, on_event=_render)

    click.echo("=== Diagnostics Complete ===")
    if report.all_passed:
        click.echo("All hosts passed pubkey authentication.")
    else:
        click.echo("Some hosts failed. See remediation suggestions above.")


def _render_ssh_host_diagnosis(d, local_pubkey_source):
    """Results and remediation for one diagnosed host (see :func:`_run_ssh_diagnose`)."""
    user, h = d.user, d.host
    click.echo()
    click.echo("  Results for %s@%s:" % (user, h))

    home_perms = d.fact("home_perms", "unknown")
    click.echo(
        "    Home dir:           %s (perms: %s) %s"
        % (d.fact("home_dir", "?"), home_perms, "PASS" if d.home_ok else "FAIL — must not be group/world-writable")
    )
    ssh_perms = d.fact("ssh_perms", "missing")
    click.echo(
        "    .ssh/ dir:          perms %s %s" % (ssh_perms, "PASS" if d.ssh_dir_ok else ("FAIL" if ssh_perms != "missing" else "MISSING"))
    )
    ak_perms = d.fact("ak_perms", "missing")
    click.echo(
        "    authorized_keys:    perms %s, %s line(s) %s"
        % (ak_perms, d.fact("ak_lines", "0"), "PASS" if d.authorized_keys_ok else ("FAIL" if ak_perms != "missing" else "MISSING"))
    )
    ak_file = d.fact("sshd_ak_file", "default")
    click.echo("    AuthorizedKeysFile: %s %s" % (ak_file, "PASS" if d.authorized_keys_file_ok else "NOTE — non-default location"))
    click.echo("    StrictModes:        %s" % d.fact("sshd_strict", "default"))
    click.echo(
        "    PubkeyAuth:         %s %s"
        % (d.fact("sshd_pubkey", "default"), "PASS" if d.pubkey_setting_ok else "FAIL — pubkey auth disabled!")
    )

    allow_users = d.fact("sshd_allow_users", "none")
    allow_groups = d.fact("sshd_allow_groups", "none")
    if allow_users != "none":
        click.echo("    AllowUsers:         %s — verify '%s' is listed" % (allow_users, user))
    if allow_groups != "none":
        click.echo("    AllowGroups:        %s — verify '%s' is in a listed group" % (allow_groups, user))

    click.echo("    AcceptedAlgorithms: %s" % d.fact("sshd_accepted_algs", "default"))
    ak_key_types = d.fact("ak_key_types", "none")
    if ak_key_types and ak_key_types != "none":
        click.echo("    AK key types:       %s" % ak_key_types.rstrip(","))

    dropin_ak = d.dropin("ak_file")
    dropin_pubkey = d.dropin("pubkey")
    dropin_accepted = d.dropin("accepted_algs")
    dropin_allow_users = d.dropin("allow_users")
    dropin_allow_groups = d.dropin("allow_groups")
    dropin_files = d.fact("sshd_dropin_files", "none")
    if dropin_files != "none":
        click.echo("    sshd drop-ins:      %s" % dropin_files)
        if dropin_ak:
            click.echo("    ↳ AuthorizedKeysFile override: %s" % dropin_ak)
        if dropin_pubkey:
            click.echo("    ↳ PubkeyAuthentication override: %s" % dropin_pubkey)
        if dropin_accepted:
            click.echo("    ↳ AcceptedAlgorithms override: %s" % dropin_accepted)
        if dropin_allow_users:
            click.echo("    ↳ AllowUsers override: %s — verify '%s' is listed" % (dropin_allow_users, user))
        if dropin_allow_groups:
            click.echo("    ↳ AllowGroups override: %s — verify '%s' is in a listed group" % (dropin_allow_groups, user))

    click.echo("    Pubkey auth test:   %s" % ("PASS" if d.pubkey_ok else "FAIL"))
    if d.cross_user:
        click.echo("    Local key installed: %s" % ("PASS" if d.key_installed else "FAIL"))
    click.echo()

    if d.pubkey_ok:
        return
    click.echo("  Suggested fixes for %s:" % h)
    if not d.home_ok and home_perms != "unknown":
        click.echo("    chmod go-w ~%s    # Fix home dir permissions (most likely cause)" % user)
    if not d.authorized_keys_ok and ak_perms != "missing":
        click.echo("    chmod 600 ~%s/.ssh/authorized_keys" % user)
    if not d.ssh_dir_ok and ssh_perms != "missing":
        click.echo("    chmod 700 ~%s/.ssh" % user)
    if not d.authorized_keys_file_ok:
        click.echo("    Check sshd_config: keys may need to go in %s" % ak_file)
    if not d.pubkey_setting_ok:
        click.echo("    Enable PubkeyAuthentication in /etc/ssh/sshd_config and restart sshd")
    if d.cross_user and not d.key_installed:
        click.echo(render_identity_text("    Re-run '{app_command} setup ssh' to install your public key"))

    if dropin_ak:
        click.echo("    ** Drop-in override detected: AuthorizedKeysFile = %s" % dropin_ak)
        click.echo("       Your key may need to go in that location instead of ~/.ssh/authorized_keys")
        click.echo("       Check files in /etc/ssh/sshd_config.d/ on the remote host")
    if dropin_pubkey and "no" in dropin_pubkey.lower():
        click.echo("    ** Drop-in override DISABLES pubkey auth: %s" % dropin_pubkey)
        click.echo("       Check files in /etc/ssh/sshd_config.d/ on the remote host")
    if dropin_accepted:
        click.echo("    ** Drop-in restricts accepted key algorithms: %s" % dropin_accepted)
        click.echo("       Ensure your key type is in the allowed list")
    if dropin_allow_users:
        click.echo("    ** Drop-in restricts AllowUsers: %s — verify '%s' is listed" % (dropin_allow_users, user))
    if dropin_allow_groups:
        click.echo("    ** Drop-in restricts AllowGroups: %s — verify '%s' is in a listed group" % (dropin_allow_groups, user))

    if d.key_mismatch:
        click.echo("    ** KEY MISMATCH: SSH is offering '%s'" % d.offered_key_path)
        click.echo(render_identity_text("       but {app_command} detected '%s' as the local key." % local_pubkey_source))
        click.echo("       Check your ~/.ssh/config IdentityFile settings.")

    # Everything looks correct but it still fails: point at the verbose output.
    if d.standard_checks_pass:
        click.echo("    All standard checks passed but pubkey auth still fails.")
        click.echo("    The SSH debug output above may reveal the cause.")
        click.echo("    Common hidden causes:")
        click.echo("      - ~/.ssh/config IdentityFile points to a non-standard key")
        click.echo("        (the mesh may have installed a different key than SSH uses)")
        click.echo("      - sshd_config 'Include' loading a drop-in that overrides settings")
        click.echo("      - SELinux/AppArmor blocking access to authorized_keys")
        click.echo("      - authorized_keys owned by wrong user (must be owned by %s)" % user)
        click.echo("      - Host key changed (check ~/.ssh/known_hosts on control machine)")
    click.echo()


@setup.command("ssh")
@host_options
@click.option("--extra-hosts", default=None, help="Additional comma-separated hosts to include (e.g. control machine)")
@click.option("--include-self/--no-include-self", default=True, show_default=True, help="Include this machine's hostname in the mesh")
@click.option("--user", "-u", default=None, help="SSH username (default: current user)")
@click.option(
    "--discover-ips/--no-discover-ips", default=True, show_default=True, help="After meshing, discover IB/CX7 IPs and distribute host keys"
)
@click.option("--diagnose", is_flag=True, default=False, help="Run SSH diagnostics instead of mesh setup")
@dry_run_option
@click.pass_context
def setup_ssh(ctx, hosts, hosts_file, cluster_name, extra_hosts, include_self, user, discover_ips, diagnose, dry_run):
    """Set up passwordless SSH mesh across cluster hosts.

    Ensures every host can SSH to every other host without password prompts.
    Creates ed25519 keys if missing and distributes public keys.

    After the mesh is established, {app_command} automatically discovers
    additional network IPs (InfiniBand, CX7) on cluster hosts and
    distributes their host keys so inter-node SSH works over all
    networks. Use --no-discover-ips to skip this phase.

    By default, the machine running {app_command} is included in the mesh
    (--include-self). Use --no-include-self to exclude it.

    You will be prompted for passwords on first connection to each host.

    Examples:

      {app_command} setup ssh --hosts 192.168.11.13,192.168.11.14

      {app_command} setup ssh --cluster mylab --user ubuntu

      {app_command} setup ssh --cluster mylab --extra-hosts 10.0.0.1

      {app_command} setup ssh --hosts 10.0.0.1,10.0.0.2 --no-discover-ips

      {app_command} setup ssh --cluster mylab --diagnose
    """
    import os

    from sparkrun.core.hosts import resolve_hosts
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.utils.net import local_ip_for

    config = SparkrunConfig()

    # Resolve hosts and look up cluster user if applicable
    cluster_mgr = _get_cluster_manager()
    host_list = resolve_hosts(
        hosts=hosts,
        hosts_file=hosts_file,
        cluster_name=cluster_name,
        cluster_manager=cluster_mgr,
        config_default_hosts=config.default_hosts,
    )

    # Determine the cluster's configured user (if hosts came from a cluster)
    cluster_user = resolve_cluster_config(cluster_name, hosts, hosts_file, cluster_mgr).user

    # Resolve 127.0.0.1 to a routable IP so other nodes can SSH back
    _resolved_loopback = False
    for i, h in enumerate(host_list):
        if h == "127.0.0.1":
            # Find first non-loopback host to determine routing
            other = next((x for x in host_list if x != "127.0.0.1"), None)
            if other:
                resolved = local_ip_for(other)
                if resolved and resolved != "127.0.0.1":
                    click.echo("Resolved 127.0.0.1 -> %s (routable IP)" % resolved)
                    host_list[i] = resolved
                    _resolved_loopback = True
                else:
                    click.echo("Warning: Could not resolve 127.0.0.1 to a routable IP, dropping it.", err=True)
                    host_list[i] = ""  # mark for removal
            else:
                click.echo("Warning: All hosts are 127.0.0.1, cannot resolve.", err=True)
                host_list[i] = ""
    host_list = [h for h in host_list if h]

    # Resolve effective user early (needed for self-inclusion decision)
    local_user = os.environ.get("USER", "root")
    if user is None:
        user = cluster_user or config.ssh_user or local_user

    # --diagnose: run SSH diagnostics instead of the mesh
    if diagnose:
        if not host_list:
            click.echo("Error: No hosts specified. Use --hosts, --hosts-file, or --cluster.", err=True)
            sys.exit(1)
        _run_ssh_diagnose(host_list, user, local_user)
        sys.exit(0)

    # Track original cluster hosts before extras/self are appended
    cluster_hosts = list(host_list)
    seen = set(host_list)
    added: list[str] = []
    if extra_hosts:
        for h in extra_hosts.split(","):
            h = h.strip()
            if h and h not in seen:
                host_list.append(h)
                seen.add(h)
                added.append(h)

    # Include the control machine unless opted out.
    self_host: str | None = None
    cross_user = user != local_user
    if include_self and host_list:
        self_host = local_ip_for(host_list[0])
        if self_host and self_host in seen and cross_user:
            click.echo(
                "Note: SSH user '%s' differs from local user '%s'. "
                "The mesh script will handle cross-user key exchange for %s automatically." % (user, local_user, self_host)
            )
        elif self_host and self_host not in seen and not cross_user:
            host_list.append(self_host)
            seen.add(self_host)
            added.append("%s (this machine)" % self_host)
        elif self_host and self_host not in seen and cross_user:
            click.echo(
                "Note: Skipping control machine (%s) in mesh — user '%s' differs from "
                "local user '%s'. Control→cluster SSH is handled automatically by the mesh script." % (self_host, user, local_user)
            )

    if not host_list:
        click.echo("Error: No hosts specified. Use --hosts, --hosts-file, or --cluster.", err=True)
        sys.exit(1)

    if len(host_list) == 1 and not cross_user:
        click.echo("Single host with same user — no SSH setup needed.")
        sys.exit(0)

    if not dry_run:
        click.echo("Setting up SSH mesh for user '%s' across %d hosts..." % (user, len(host_list)))
        click.echo("Cluster Hosts: %s" % ", ".join(sorted(cluster_hosts)))
        if added:
            click.echo("Added: %s" % ", ".join(added))
        click.echo()

    # Without a local bash (Windows) the mesh runs natively, which logs in with
    # keys from its first connection; ``mesh_ssh_keys.sh`` bootstraps with
    # passwords instead.  So the native path needs the same access gate the
    # wizard runs, or a fresh cluster fails before anything is installed.  The
    # user is the one this command resolved, so the gate may not change it.
    from ._ssh import _ensure_ssh_access, _mesh_uses_native

    if _mesh_uses_native():
        outcome = _ensure_ssh_access(host_list, user, config, dry_run=dry_run, offer_user_change=False)
        if not outcome.all_ok:
            click.echo("Error: key-based SSH is required on every host before the mesh can run.", err=True)
            sys.exit(1)

    ok = _run_ssh_mesh(
        host_list,
        user,
        cluster_hosts=cluster_hosts,
        ssh_key=config.ssh_key,
        discover_ips=discover_ips,
        dry_run=dry_run,
        control_is_member=_resolved_loopback or (self_host is not None and self_host in set(cluster_hosts)),
        config=config,
    )
    if ok:
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        _check_ssh_host_identities(
            cluster_hosts,
            {**build_ssh_kwargs(config), "ssh_user": user},
            cluster_name=cluster_name,
            dry_run=dry_run,
        )
    if ok and not dry_run:
        _record_setup_phase(cluster_name, user, host_list, "ssh_mesh", mesh_hosts=list(host_list))
    sys.exit(0 if ok else 1)


@setup.command("cx7")
@host_options
@click.option("--user", "-u", default=None, help="SSH username (default: from config or current user)")
@dry_run_option
@click.option("--force", is_flag=True, help="Reconfigure even if existing config is valid")
@click.option("--mtu", default=9000, show_default=True, type=int, help="MTU for CX7 interfaces")
@click.option("--subnet1", default=None, help="Override subnet for CX7 partition 1 (e.g. 192.168.11.0/24)", hidden=True)
@click.option("--subnet2", default=None, help="Override subnet for CX7 partition 2 (e.g. 192.168.12.0/24)", hidden=True)
@click.option(
    "--topology",
    type=click.Choice(["auto", "direct", "switch", "ring"]),
    default="auto",
    show_default=True,
    help="CX7 topology (auto-detected by default)",
    hidden=True,
)
@click.option(
    "--interfaces",
    default=None,
    help="Limit which CX7 interfaces are used (comma-separated names or globs, e.g. '*np1'). "
    "Saved to the cluster and reused on subsequent runs.",
)
@click.option(
    "--port",
    type=click.Choice(["0", "1"]),
    default=None,
    help="Convenience selector for the Nth physical CX7 port pair (equivalent to --interfaces '*npN').",
)
@click.pass_context
def setup_cx7(ctx, hosts, hosts_file, cluster_name, user, dry_run, force, mtu, subnet1, subnet2, topology, interfaces, port):
    """Configure CX7 network interfaces on cluster hosts.

    Detects ConnectX-7 interfaces, assigns static IPs with jumbo frames
    (MTU 9000), and applies netplan configuration.

    Supports three topologies:

    \b
      direct  — 2 nodes, 2 subnets (default for 2-node clusters)
      switch  — N nodes via switch, 2 subnets
      ring    — 3 nodes in ring, 6 subnets (auto-detected with 3 nodes)

    Existing valid configurations are preserved unless --force is used.
    IP addresses are derived from each host's management IP last octet.

    Requires passwordless sudo on target hosts.

    Examples:

      {app_command} setup cx7 --hosts 10.24.11.13,10.24.11.14

      {app_command} setup cx7 --cluster mylab --dry-run

      {app_command} setup cx7 --cluster mylab --topology ring

      {app_command} setup cx7 --cluster mylab --force

      {app_command} setup cx7 --cluster two --interfaces '*np1'   # pin a port pair

      {app_command} setup cx7 --cluster two --port 1              # same, by port index
    """
    import os

    from sparkrun.api.setup import (
        CX7_EVENT_DETECTED,
        CX7_EVENT_DETECTED_TOPOLOGY,
        CX7_EVENT_DETECTING_TOPOLOGY,
        CX7_EVENT_INTERFACES,
        CX7_EVENT_RING_DETECTING,
        CX7_EVENT_SUBNETS,
        CX7_EVENT_TOPOLOGY,
        Cx7PlanError,
        apply_cx7,
        plan_cx7,
        validate_cx7_options,
    )
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.networking import CX7_NETPLAN_FILE
    from sparkrun.orchestration.sudo import SudoPasswords
    from ._sudo import ensure_sudo_password

    try:
        validate_cx7_options(interfaces=interfaces, port=port, subnet1=subnet1, subnet2=subnet2)
    except Cx7PlanError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)
    explicit_interfaces: list[str] | None = None
    if interfaces:
        explicit_interfaces = [s.strip() for s in interfaces.split(",") if s.strip()]

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)
    from ._step_runner import require_setup_step_targets

    require_setup_step_targets(
        "cx7",
        host_list,
        ssh_kwargs,
        config,
        cluster_name=cluster_name,
        explicit_hosts=bool(hosts or hosts_file),
        dry_run=dry_run,
    )

    cluster_mgr = _get_cluster_manager()

    def _render(kind, p):
        if kind == CX7_EVENT_DETECTED:
            if p.missing_cx7:
                click.echo("Warning: No CX7 interfaces on: %s" % ", ".join(p.missing_cx7), err=True)
        elif kind == CX7_EVENT_INTERFACES:
            if p.interfaces:
                click.echo("Interface filter: %s" % ", ".join(p.interfaces))
        elif kind == CX7_EVENT_DETECTING_TOPOLOGY:
            click.echo("Detecting topology via neighbor discovery...")
        elif kind == CX7_EVENT_DETECTED_TOPOLOGY:
            click.echo("Detected topology: %s" % p.topology.value)
        elif kind == CX7_EVENT_RING_DETECTING:
            click.echo("Running topology detection for ring configuration...")
        elif kind == CX7_EVENT_TOPOLOGY:
            click.echo()
            # Two hosts on auto cannot be told apart at L2 — show both.
            click.echo("Topology: switch/direct" if p.topology_ambiguous else "Topology: %s" % p.topology.value)
        elif kind == CX7_EVENT_SUBNETS:
            click.echo("Subnets: %s" % ", ".join(str(s) for s in p.subnets))

    try:
        plan = plan_cx7(
            host_list,
            ssh_kwargs=ssh_kwargs,
            dry_run=dry_run,
            cluster_name=cluster_name,
            cluster_manager=cluster_mgr,
            topology=topology,
            interfaces=explicit_interfaces,
            port=int(port) if port is not None else None,
            subnet1=subnet1,
            subnet2=subnet2,
            mtu=mtu,
            force=force,
            on_event=_render,
        )
    except Cx7PlanError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    click.echo("MTU: %d" % mtu)
    click.echo()

    cluster_plan = plan.cluster_plan
    assert cluster_plan is not None
    for hp in cluster_plan.host_plans:
        det = plan.detections.get(hp.host)
        mgmt_label = " (%s)" % det.mgmt_ip if det and det.mgmt_ip else ""
        click.echo("  %s%s" % (hp.host, mgmt_label))
        for a in hp.assignments:
            status = "OK" if not hp.needs_change else "configure"
            click.echo("    %-20s -> %s/%d  MTU %d  [%s]" % (a.iface_name, a.ip, cluster_plan.prefix_len, cluster_plan.mtu, status))
        if not hp.assignments and hp.needs_change:
            click.echo("    %s" % hp.reason)
        click.echo()

    for w in cluster_plan.warnings:
        click.echo("Warning: %s" % w, err=True)

    # Plan-level errors (e.g. insufficient ports for ring)
    if plan.plan_failed:
        for e in cluster_plan.errors:
            click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    if plan.all_valid and not force:
        click.echo("All hosts already configured. Use --force to reconfigure.")
        return

    needs_config = plan.configurable_count
    already_ok = plan.already_valid_count
    has_errors = plan.blocked_count

    if needs_config == 0:
        if has_errors:
            click.echo("Error: %d host(s) have issues that prevent configuration." % has_errors, err=True)
            for e in cluster_plan.errors:
                click.echo("  %s" % e, err=True)
            sys.exit(1)
        click.echo("No hosts need configuration changes.")
        return

    if dry_run:
        click.echo("[dry-run] Would configure %d host(s), %d already valid." % (needs_config, already_ok))
        return

    if not force and not click.confirm("Apply changes to %d host(s)?" % needs_config, default=True):
        click.echo("Aborted.")
        return

    # Sudo: ensure_sudo_password() tests NOPASSWD, prompts, verifies, and
    # supports cross-user fallback.  sudo_ssh_kwargs differs from ssh_kwargs
    # when the cluster user lacks sudo but another user has it (indirect sudo).
    sudo_ssh_kwargs = dict(ssh_kwargs)
    sudo_hosts_needing_pw: set[str] = set()
    sudo_password, indirect_user = ensure_sudo_password(
        host_list,
        user,
        ssh_kwargs,
        sudo_ssh_kwargs=sudo_ssh_kwargs,
        dry_run=dry_run,
        allow_indirect=True,
        default_user=os.environ.get("USER", ""),
    )
    if sudo_password is not None:
        if indirect_user:
            sudo_ssh_kwargs = dict(ssh_kwargs, ssh_user=indirect_user)
        sudo_hosts_needing_pw.update(h for h, d in plan.hosts_with_cx7.items() if not d.sudo_ok)

    click.echo("Applying configuration to %d host(s)..." % needs_config)
    applied = apply_cx7(
        plan,
        ssh_kwargs=sudo_ssh_kwargs,
        dry_run=dry_run,
        sudo_password=sudo_password,
        sudo_hosts=sudo_hosts_needing_pw,
        passwords=SudoPasswords(prompt_host=host_password_prompt(sudo_ssh_kwargs.get("ssh_user", user))),
        cluster_manager=cluster_mgr,
    )
    configured = applied.configured
    failed = applied.failed

    for r in applied.failures:
        click.echo("  [FAIL] %s: %s" % (r.host, r.stderr.strip()[:100]), err=True)

    click.echo()
    parts = []
    if configured:
        parts.append("%d configured" % configured)
    if already_ok:
        parts.append("%d already valid" % already_ok)
    if failed:
        parts.append("%d failed" % failed)
    if has_errors:
        parts.append("%d skipped (errors)" % has_errors)
    click.echo("Results: %s." % ", ".join(parts))

    saved_topology = applied.topology_save
    if saved_topology is not None and saved_topology.refused:
        click.echo(
            "Topology not saved to cluster '%s': %d host(s) failed. Re-run once they are fixed."
            % (saved_topology.cluster_name, saved_topology.failed_hosts),
            err=True,
        )
    elif saved_topology is not None and saved_topology.saved:
        click.echo("Saved topology '%s' to cluster '%s'." % (saved_topology.topology, saved_topology.cluster_name))
    elif saved_topology is not None and saved_topology.error:
        click.echo("Warning: could not save topology to cluster: %s" % saved_topology.error, err=True)

    saved_ifaces = applied.interfaces_save
    if saved_ifaces is not None and saved_ifaces.saved:
        click.echo("Saved interface filter %s to cluster '%s'." % (saved_ifaces.interfaces, saved_ifaces.cluster_name))
    elif saved_ifaces is not None and saved_ifaces.error:
        click.echo("Warning: could not save interface filter to cluster: %s" % saved_ifaces.error, err=True)

    if configured and not dry_run:
        _record_setup_phase(
            cluster_name,
            user,
            host_list,
            "cx7",
            subnets=[str(s) for s in plan.subnets],
            cx7_ips=plan.assigned_ips,
            netplan_file=CX7_NETPLAN_FILE,
        )

    if failed:
        sys.exit(1)


@setup.command("docker-group")
@host_options
@click.option("--user", "-u", default=None, help="Target user (default: SSH user)")
@dry_run_option
@click.pass_context
def setup_docker_group(ctx, hosts, hosts_file, cluster_name, user, dry_run):
    """Ensure user is a member of the docker group on cluster hosts.

    Runs ``usermod -aG docker <user>`` on each host so that Docker
    commands work without sudo.  Requires sudo on target hosts.

    A re-login (or ``newgrp docker``) is needed on hosts where the
    user was newly added.

    Examples:

      {app_command} setup docker-group --hosts 10.24.11.13,10.24.11.14

      {app_command} setup docker-group --cluster mylab

      {app_command} setup docker-group --cluster mylab --user ubuntu
    """
    from ._step_runner import run_setup_step_command

    run_setup_step_command("docker_group", hosts, hosts_file, cluster_name, user, dry_run)


@setup.command("fix-permissions")
@host_options
@click.option("--user", "-u", default=None, help="Target owner (default: SSH user)")
@click.option("--cache-dir", default=None, help="Cache directory (default: ~/.cache/huggingface)")
@click.option("--save-sudo", is_flag=True, default=False, help="Install sudoers entry for passwordless chown (requires sudo once)")
@dry_run_option
@click.pass_context
def setup_fix_permissions(ctx, hosts, hosts_file, cluster_name, user, cache_dir, save_sudo, dry_run):
    """Fix file ownership in HuggingFace cache on cluster hosts.

    Docker containers create files as root in ~/.cache/huggingface/,
    leaving the normal user unable to manage or clean the cache.
    This command runs chown on all target hosts to restore ownership.

    Tries non-interactive sudo first on all hosts in parallel, then
    falls back to password-based sudo for any that fail.

    Use --save-sudo to install a scoped sudoers entry so future runs
    never need a password. The entry only permits chown on the cache
    directory — no broader privileges are granted.

    Examples:

      {app_command} setup fix-permissions --hosts 10.24.11.13,10.24.11.14

      {app_command} setup fix-permissions --cluster mylab

      {app_command} setup fix-permissions --cluster mylab --cache-dir /data/hf-cache

      {app_command} setup fix-permissions --cluster mylab --save-sudo

      {app_command} setup fix-permissions --cluster mylab --dry-run
    """
    from sparkrun.core.config import SparkrunConfig

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

    # Resolve cache path
    cache_path = cache_dir  # None means use getent-based home detection

    click.echo("Fixing file permissions for user '%s' on %d host(s)..." % (user, len(host_list)))
    if cache_path:
        click.echo("Cache directory: %s" % cache_path)
    click.echo()

    from sparkrun.scripts import read_script

    passwords = interactive_sudo_passwords(user)

    # --save-sudo: install the scoped chown entry where it is not in effect yet
    if save_sudo:
        save_sudoers_entry(
            "chown",
            host_list,
            user,
            ssh_kwargs,
            passwords,
            config=config,
            cluster_name=cluster_name,
            explicit_hosts=bool(hosts or hosts_file),
            dry_run=dry_run,
            cache_dir=cache_path or "",
        )

    # sudo -n first (works once the entry is in place), then the password
    # fallback; a host rejecting the shared password is asked for its own.
    chown_script = read_script("fix_permissions.sh").format(user=user, cache_dir=cache_path or "")
    fallback_script = read_script("fix_permissions_fallback.sh").format(user=user, cache_dir=cache_path or "")
    result_map = run_sudo_action(host_list, chown_script, fallback_script, ssh_kwargs, passwords, dry_run=dry_run)

    # Report results
    ok_count = 0
    skip_count = 0
    fail_count = 0
    for h in host_list:
        r = result_map.get(h)
        if r is None:
            continue
        if r.success:
            if "SKIP:" in r.stdout:
                skip_count += 1
                click.echo("  [SKIP] %s: %s" % (h, r.stdout.strip()))
            else:
                ok_count += 1
                click.echo("  [OK]   %s: %s" % (h, r.stdout.strip()))
        else:
            fail_count += 1
            click.echo("  [FAIL] %s: %s" % (h, excerpt(r.stderr, 200)), err=True)

    click.echo()
    parts = []
    if ok_count:
        parts.append("%d fixed" % ok_count)
    if skip_count:
        parts.append("%d skipped (no cache)" % skip_count)
    if fail_count:
        parts.append("%d failed" % fail_count)
    click.echo("Results: %s." % ", ".join(parts) if parts else "No hosts processed.")

    if fail_count:
        sys.exit(1)


@setup.command("clear-cache")
@host_options
@click.option("--user", "-u", default=None, help="Target user for sudoers entry (default: SSH user)")
@click.option("--save-sudo", is_flag=True, default=False, help="Install sudoers entry for passwordless cache clearing (requires sudo once)")
@dry_run_option
@click.pass_context
def setup_clear_cache(ctx, hosts, hosts_file, cluster_name, user, save_sudo, dry_run):
    """Drop the Linux page cache on cluster hosts.

    Runs 'sync' followed by writing 3 to /proc/sys/vm/drop_caches on
    each target host.  This frees cached file data so inference
    containers have maximum available memory on DGX Spark's unified
    CPU/GPU memory.

    Tries non-interactive sudo first on all hosts in parallel, then
    falls back to password-based sudo for any that fail.

    Use --save-sudo to install a scoped sudoers entry so future runs
    never need a password. The entry only permits writing to
    /proc/sys/vm/drop_caches — no broader privileges are granted.

    Examples:

      {app_command} setup clear-cache --hosts 10.24.11.13,10.24.11.14

      {app_command} setup clear-cache --cluster mylab

      {app_command} setup clear-cache --cluster mylab --save-sudo

      {app_command} setup clear-cache --cluster mylab --dry-run
    """
    from sparkrun.core.config import SparkrunConfig

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

    click.echo("Clearing page cache on %d host(s)..." % len(host_list))
    click.echo()

    from sparkrun.scripts import read_script

    passwords = interactive_sudo_passwords(user)

    # --save-sudo: install only the drop_caches entry, where not yet in effect
    if save_sudo:
        save_sudoers_entry(
            "dropcaches",
            host_list,
            user,
            ssh_kwargs,
            passwords,
            config=config,
            cluster_name=cluster_name,
            explicit_hosts=bool(hosts or hosts_file),
            dry_run=dry_run,
        )

    result_map = run_sudo_action(
        host_list, read_script("clear_cache.sh"), read_script("clear_cache_fallback.sh"), ssh_kwargs, passwords, dry_run=dry_run
    )

    # Report results
    ok_count = 0
    fail_count = 0
    for h in host_list:
        r = result_map.get(h)
        if r is None:
            continue
        if r.success:
            ok_count += 1
            click.echo("  [OK]   %s: %s" % (h, r.stdout.strip()))
        else:
            fail_count += 1
            click.echo("  [FAIL] %s: %s" % (h, excerpt(r.stderr, 200)), err=True)

    click.echo()
    parts = []
    if ok_count:
        parts.append("%d cleared" % ok_count)
    if fail_count:
        parts.append("%d failed" % fail_count)
    click.echo("Results: %s." % ", ".join(parts) if parts else "No hosts processed.")

    if fail_count:
        sys.exit(1)


@setup.command("earlyoom")
@host_options
@click.option("--user", "-u", default=None, help="SSH username (default: from config or current user)")
@click.option("--prefer", "extra_prefer", default=None, help="Additional comma-separated process patterns to prefer killing on OOM")
@click.option("--avoid", "extra_avoid", default=None, help="Additional comma-separated process patterns to avoid killing on OOM")
@dry_run_option
@click.pass_context
def setup_earlyoom(ctx, hosts, hosts_file, cluster_name, user, extra_prefer, extra_avoid, dry_run):
    """Install and configure earlyoom OOM killer on cluster hosts.

    earlyoom monitors available memory and proactively kills processes
    before the kernel OOM killer triggers. This prevents system hangs
    when large inference models exhaust memory on DGX Spark.

    By default, {app_command} configures earlyoom to prefer killing inference
    workload processes (vllm, sglang, llama-server, trtllm, python) and
    to avoid killing system services (sshd, systemd, dockerd).

    Use --prefer/--avoid to add additional process patterns.

    Requires sudo on target hosts (apt-get install, systemctl).

    Examples:

      {app_command} setup earlyoom --hosts 192.168.11.13,192.168.11.14

      {app_command} setup earlyoom --cluster mylab

      {app_command} setup earlyoom --cluster mylab --prefer "my-app,worker"

      {app_command} setup earlyoom --cluster mylab --dry-run
    """
    from sparkrun.core.config import SparkrunConfig

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)
    from ._step_runner import require_setup_step_targets

    require_setup_step_targets(
        "earlyoom",
        host_list,
        ssh_kwargs,
        config,
        cluster_name=cluster_name,
        explicit_hosts=bool(hosts or hosts_file),
        dry_run=dry_run,
    )

    # Build prefer/avoid pattern lists
    prefer = list(EARLYOOM_PREFER_PATTERNS)
    if extra_prefer:
        prefer.extend(p.strip() for p in extra_prefer.split(",") if p.strip())

    avoid = list(EARLYOOM_AVOID_PATTERNS)
    if extra_avoid:
        avoid.extend(p.strip() for p in extra_avoid.split(",") if p.strip())

    prefer_regex = _build_earlyoom_regex(prefer)
    avoid_regex = _build_earlyoom_regex(avoid)

    click.echo("Installing earlyoom on %d host(s)..." % len(host_list))
    click.echo("  Prefer (kill first): %s" % prefer_regex)
    click.echo("  Avoid (protect):     %s" % avoid_regex)
    click.echo()
    click.echo("Note: first install may take ~1 minute per host (apt-get update + install).")
    click.echo()

    from sparkrun.scripts import read_script

    # Generate install scripts with the prefer/avoid patterns
    install_script = read_script("earlyoom_install.sh").format(
        prefer=prefer_regex,
        avoid=avoid_regex,
    )
    fallback_script = read_script("earlyoom_install_fallback.sh").format(
        prefer=prefer_regex,
        avoid=avoid_regex,
    )

    def report(host, result):
        if result is None:
            click.echo("  %s: configuring with sudo password..." % host)
        elif result.success:
            click.echo("  [OK]   %s: %s" % (host, _earlyoom_summary(result.stdout)))
        else:
            click.echo("  %s: FAILED (details below)" % host)

    # sudo -n first; hosts that need a password are configured one at a time
    # with progress, each re-asked for its own password if sudo rejects it.
    result_map = run_sudo_action(
        host_list, install_script, fallback_script, ssh_kwargs, interactive_sudo_passwords(user), dry_run=dry_run, progress=report
    )

    # Final summary
    ok_count = sum(1 for h in host_list if result_map.get(h) and result_map[h].success)
    fail_count = sum(1 for h in host_list if result_map.get(h) and not result_map[h].success)

    for h in host_list:
        r = result_map.get(h)
        if r and not r.success:
            click.echo("  [FAIL] %s: %s" % (h, excerpt(r.stderr, 200)), err=True)

    click.echo()
    parts = []
    if ok_count:
        parts.append("%d configured" % ok_count)
    if fail_count:
        parts.append("%d failed" % fail_count)
    click.echo("Results: %s." % ", ".join(parts) if parts else "No hosts processed.")

    if ok_count and not dry_run:
        installed_pkg = any("INSTALLING:" in (result_map.get(h) and result_map[h].stdout or "") for h in host_list)
        _record_setup_phase(
            cluster_name,
            user,
            host_list,
            "earlyoom",
            installed_package=installed_pkg,
        )

    if ok_count:
        click.echo()
        click.echo("Thanks to @shahizat for posting this idea on the DGX forums!")

    if fail_count:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------


@setup.command("diagnose", hidden=HIDE_ADVANCED_OPTIONS)
@host_options
@dry_run_option
@click.option(
    "-o",
    "--output",
    "output_file",
    default=None,
    type=click.Path(),
    help="Output NDJSON file path (default: spark_diag_<timestamp>.ndjson)",
)
@json_option(help="Also print summary to stdout as JSON")
@click.option("--sudo", "use_sudo", is_flag=True, default=False, help="Also collect sudo-only diagnostics (dmidecode)")
@click.pass_context
def setup_diagnose(ctx, hosts, hosts_file, cluster_name, dry_run, output_file, output_json, use_sudo):
    """Collect hardware, firmware, network, and Docker diagnostics from hosts.

    Collects OS, kernel, CPU, memory, disk, GPU, network, Docker, and
    firmware device information without requiring elevated privileges.

    Use --sudo to also collect dmidecode data (BIOS, system, baseboard,
    memory details) which requires a sudo password.
    """
    from datetime import datetime, timezone

    from sparkrun import api

    from .._common import _get_context, _resolve_setup_context

    sctx = _get_context(ctx)
    config = sctx.config

    host_list, user, _ = _resolve_setup_context(hosts, hosts_file, cluster_name, config)

    if not output_file:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = "spark_diag_%s.ndjson" % ts

    # Prompt for sudo password upfront if --sudo
    sudo_password = None
    if use_sudo and not dry_run:
        sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)

    click.echo("Collecting diagnostics from %d host(s)..." % len(host_list))
    click.echo("Output: %s" % output_file)
    click.echo()

    try:
        bundle = api.collect_diagnostics(
            output_file,
            hosts=host_list,
            cluster=cluster_name,
            sctx=sctx,
            sudo_password=sudo_password,
            dry_run=dry_run,
        )
    except OSError as exc:
        raise click.ClickException("Cannot open diagnostic bundle: %s" % exc) from exc
    host_data = bundle.hosts

    # Display summary
    ok = sum(host_data.values())
    fail = 0 if dry_run else len(host_list) - ok

    click.echo()
    click.echo("Diagnostics complete: %d/%d hosts OK" % (ok, len(host_list)))
    if fail:
        click.echo("  %d host(s) failed — see %s for details" % (fail, output_file), err=True)

    for section in bundle.sections:
        if section["status"] != "ok":
            click.echo("  %s: %s — %s" % (section["section"], section["status"], section.get("detail", "")), err=True)

    if output_json:
        from .._common import print_json

        summary = {
            "output_file": output_file,
            "total_hosts": len(host_list),
            "successful": ok,
            "failed": fail,
            "hosts": host_data,
            "partial": bundle.partial,
            "sections": bundle.sections,
        }
        print_json(summary)

    if fail or (bundle.partial and not dry_run):
        sys.exit(1)


@setup.command("prune-runtime-cache", hidden=HIDE_ADVANCED_OPTIONS)
@host_options
@click.option(
    "--older-than",
    "older_than_days",
    type=int,
    default=None,
    help="Age cutoff in days (default: the resolved runtime_cache.prune.max_age_days).",
)
@click.option("--all", "purge_all", is_flag=True, default=False, help="Remove every cached tree, ignoring the age cutoff.")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be removed, with sizes, without removing it.")
@click.pass_context
def setup_prune_runtime_cache(ctx, hosts, hosts_file, cluster_name, older_than_days, purge_all, dry_run):
    """Reclaim disk from the compilation/autotune cache on cluster hosts.

    The runtime cache (~/.cache/{app_command}/runtime-cache/ on each host) holds
    torch.compile, Inductor, Triton, FlashInfer and TRT-LLM autotuner output so
    a relaunch skips minutes of recompilation.  It lives on the TARGETS, not on
    this machine, so this command fans out over SSH.

    '{app_command} run' already sweeps the trees it can see on every launch (disable
    with 'runtime_cache.prune.enabled: false').  This is the manual sweep, for
    reclaiming space without launching something — and the only way to clear
    trees for a model or image you no longer run at all.

    A tree is aged by its last-used marker, which every launch re-stamps, so a
    cache you use daily never ages out no matter how long ago it was written.

    Examples:

      {app_command} setup prune-runtime-cache --cluster mylab --dry-run

      {app_command} setup prune-runtime-cache --cluster mylab --older-than 7

      {app_command} setup prune-runtime-cache --cluster mylab --all
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.cli._common import resolve_host_context
    from sparkrun.core.runtime_cache import (
        resolve_runtime_cache_root,
        resolve_runtime_cache_settings,
    )
    from sparkrun.orchestration.primitives import build_ssh_kwargs, probe_remote_sparkrun_cache
    from sparkrun.orchestration.runtime_cache import generate_runtime_cache_sweep_script
    from sparkrun.orchestration.ssh import run_remote_scripts_parallel

    config = SparkrunConfig()
    hctx = resolve_host_context(hosts, hosts_file, cluster_name, config)
    host_list = hctx.host_list
    cluster_def = hctx.cluster_mgr.get(hctx.cluster_name) if hctx.cluster_name else None

    settings = resolve_runtime_cache_settings(config=config, cluster=cluster_def)
    max_age = older_than_days if older_than_days is not None else settings.prune_max_age_days
    ssh_kwargs = build_ssh_kwargs(config)

    if purge_all and not dry_run:
        click.confirm("Remove ALL runtime cache trees on %d host(s)?" % len(host_list), abort=True)

    total_kb = 0
    total_trees = 0
    for host in host_list:
        # Each host resolves its own root: $HOME differs, and an explicit
        # runtime_cache.dir applies everywhere.
        try:
            root = resolve_runtime_cache_root(settings, probe_remote_sparkrun_cache(host, **ssh_kwargs))
            script = generate_runtime_cache_sweep_script(root, max_age_days=max_age, dry_run=dry_run, purge_all=purge_all)
        except Exception as exc:
            click.echo("  %s: skipped (%s)" % (host, exc))
            continue

        results = run_remote_scripts_parallel(
            [host],
            script,
            ssh_user=ssh_kwargs.get("ssh_user"),
            ssh_key=ssh_kwargs.get("ssh_key"),
            ssh_options=ssh_kwargs.get("ssh_options"),
            timeout=120,
            quiet=True,
            allow_local=True,
        )
        for r in results:
            if r.returncode != 0:
                click.echo("  %s: unreachable or sweep failed" % host)
                continue
            for line in r.stdout.splitlines():
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                verb, size_kb, tree = parts
                total_trees += 1
                total_kb += int(size_kb) if size_kb.isdigit() else 0
                click.echo("  %s  %6.1f GB  %s:%s" % (verb, int(size_kb) / 1048576 if size_kb.isdigit() else 0.0, host, tree))

    click.echo()
    if not total_trees:
        click.echo("Nothing to prune.")
        return
    click.echo("%s %d tree(s), %.1f GB." % ("Would remove" if dry_run else "Removed", total_trees, total_kb / 1048576))
    if dry_run:
        click.echo("Re-run without --dry-run to apply.")


@setup.command("prune-job-metadata-cache", hidden=HIDE_ADVANCED_OPTIONS)
@click.option(
    "--older-than",
    "older_than_days",
    type=int,
    default=None,
    help="Age cutoff in days (default: %d). Use 0 to ignore age." % PRUNE_MAX_AGE_DAYS,
)
@click.option(
    "--keep-per-intent",
    type=int,
    default=None,
    help="Keep this many most recent jobs per workload identity (default: %d)." % PRUNE_KEEP_PER_INTENT,
)
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be removed without removing it.")
def setup_prune_job_metadata_cache(older_than_days, keep_per_intent, dry_run):
    """Delete stale entries from the local job metadata cache.

    Job metadata (~/.cache/{app_command}/jobs/) records the recipe, hosts and
    container image behind every launch, so stop/logs/proxy discovery can find
    a workload later.  Only an explicit 'stop' removes an entry, so a job that
    crashed — the norm when auto_remove is on and the container is gone before
    anything asks about it — stays cached forever.  Left alone this reaches
    hundreds of dead entries and makes 'logs <TAB>' unusable.

    '{app_command} run' prunes automatically using the occupancy snapshot it
    already has (disable with 'jobs.autoprune: false' in config.yaml).  This
    command is the manual sweep, for when you want to reclaim the cache
    without launching something.

    An entry is KEPT when it is both among the most recent for its workload
    identity AND newer than the age cutoff.

    WARNING: this command has no live cluster snapshot to check against, so —
    unlike the automatic prune — it cannot protect a running workload whose
    metadata has aged out.  Run '{app_command} status' first if you have long-lived
    deployments, or pass --dry-run to review the list.

    Examples:

      {app_command} setup prune-job-metadata-cache --dry-run

      {app_command} setup prune-job-metadata-cache --older-than 90

      {app_command} setup prune-job-metadata-cache --keep-per-intent 1
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.job_metadata import prune_job_metadata

    config = SparkrunConfig()
    kwargs = {"cache_dir": str(config.cache_dir), "dry_run": dry_run}
    if older_than_days is not None:
        kwargs["max_age_days"] = older_than_days
    if keep_per_intent is not None:
        kwargs["keep_per_intent"] = keep_per_intent

    removed = prune_job_metadata(**kwargs)
    if not removed:
        click.echo("Nothing to prune.")
        return

    for cluster_id in removed:
        click.echo("  %s %s" % ("would remove" if dry_run else "removed", cluster_id))
    click.echo()
    click.echo("%s %d job metadata entrie(s)." % ("Would remove" if dry_run else "Removed", len(removed)))
    if dry_run:
        click.echo("Re-run without --dry-run to apply.")
