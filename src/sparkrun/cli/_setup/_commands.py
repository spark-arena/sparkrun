"""All setup subcommand definitions for sparkrun."""

from __future__ import annotations

import sys

import click

from .._common import (
    _detect_shell,
    _get_cluster_manager,
    _require_uv,
    resolve_cluster_config,
    _resolve_setup_context,
    _shell_rc_file,
    dry_run_option,
    host_options, json_option,
)
from . import setup
from ._phases import (
    EARLYOOM_PREFER_PATTERNS,
    EARLYOOM_AVOID_PATTERNS,
    _build_earlyoom_regex,
    _earlyoom_summary,
    _DOCKER_GROUP_SCRIPT,
    _DOCKER_GROUP_FALLBACK_SCRIPT,
    _docker_group_summary,
)
from ._ssh import _run_ssh_mesh
from ._sudo import _record_setup_phase


@setup.command("completion", hidden=True)
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), default=None, help="Shell type (auto-detected if not specified)")
@click.pass_context
def setup_completion(ctx, shell):
    """Install shell tab-completion for sparkrun.

    Detects your current shell and appends the completion setup to
    your shell config file (~/.bashrc, ~/.zshrc, or ~/.config/fish/...).

    Examples:

      sparkrun setup completion

      sparkrun setup completion --shell bash
    """
    if not shell:
        shell, rc_file = _detect_shell()
    else:
        rc_file = _shell_rc_file(shell)

    completion_var = "_SPARKRUN_COMPLETE"

    if shell == "bash":
        snippet = 'eval "$(%s=bash_source sparkrun)"' % completion_var
    elif shell == "zsh":
        snippet = 'eval "$(%s=zsh_source sparkrun)"' % completion_var
    elif shell == "fish":
        snippet = "%s=fish_source sparkrun | source" % completion_var

    # Check if already installed
    if rc_file.exists():
        contents = rc_file.read_text()
        if completion_var in contents:
            click.echo("Completion already configured in %s" % rc_file)
            return

    # Ensure parent directory exists (for fish)
    rc_file.parent.mkdir(parents=True, exist_ok=True)

    with open(rc_file, "a") as f:
        f.write("\n# sparkrun tab-completion\n")
        f.write(snippet + "\n")

    click.echo("Completion installed for %s in %s" % (shell, rc_file))
    click.echo("Restart your shell or run: source %s" % rc_file)


@setup.command("install")
@click.option("--shell", type=click.Choice(["bash", "zsh", "fish"]), default=None, help="Shell type (auto-detected if not specified)")
@click.option("--no-update-registries", is_flag=True, help="Skip updating recipe registries after installation")
@click.pass_context
def setup_install(ctx, shell, no_update_registries):
    """Install sparkrun and tab-completion.

    Requires uv (https://docs.astral.sh/uv/).  Typical usage:

    \b
      uvx sparkrun setup install

    This installs sparkrun as a uv tool (real binary on PATH), cleans up
    any old aliases/functions from previous installs, configures
    tab-completion, and updates recipe registries.
    """
    import subprocess

    if not shell:
        shell, rc_file = _detect_shell()
    else:
        rc_file = _shell_rc_file(shell)

    # Step 1: Install sparkrun via uv tool
    uv = _require_uv()

    click.echo("Installing sparkrun via uv tool install...")
    result = subprocess.run(
        [uv, "tool", "install", "sparkrun", "--force"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo("Error installing sparkrun: %s" % result.stderr.strip(), err=True)
        sys.exit(1)
    click.echo("sparkrun installed on PATH")

    # Step 2: Clean up old aliases/functions from previous installs
    if rc_file.exists():
        old_markers = [
            "alias sparkrun=",
            "alias sparkrun ",
            "function sparkrun",
            "sparkrun()",
        ]
        contents = rc_file.read_text()
        lines = contents.splitlines(keepends=True)
        cleaned = [line for line in lines if not any(m in line for m in old_markers)]
        if len(cleaned) != len(lines):
            rc_file.write_text("".join(cleaned))
            click.echo("Cleaned up old sparkrun aliases from %s" % rc_file)

    # Step 3: Install tab-completion
    ctx.invoke(setup_completion, shell=shell)

    # Step 4: Update recipe registries
    if not no_update_registries:
        click.echo()
        click.echo("Updating recipe registries...")
        reg_result = subprocess.run(
            ["sparkrun", "registry", "update"],
            capture_output=False,
        )
        if reg_result.returncode != 0:
            click.echo("Warning: registry update failed (non-fatal).", err=True)


@setup.command("update")
@click.option("--no-update-registries", is_flag=True, help="Skip updating recipe registries")
@click.pass_context
def setup_update(ctx, no_update_registries):
    """Update sparkrun to the latest version.

    Requires sparkrun to have been installed via ``uv tool install``.
    After upgrading, recipe registries are also updated.

    \b
      sparkrun setup update
    """
    import shutil
    import subprocess

    from sparkrun import __version__ as old_version

    uv = shutil.which("uv")
    if not uv:
        click.echo("Error: uv not found. Install uv first: https://docs.astral.sh/uv/", err=True)
        sys.exit(1)

    check = subprocess.run(
        [uv, "tool", "list"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0 or "sparkrun" not in check.stdout:
        click.echo(
            "Error: sparkrun was not installed via 'uv tool install'.\n"
            "Cannot safely upgrade — manage updates through your package manager instead.",
            err=True,
        )
        sys.exit(1)

    click.echo("Checking for updates (current: %s)..." % old_version)
    result = subprocess.run(
        [uv, "tool", "upgrade", "sparkrun"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo("Error updating sparkrun: %s" % result.stderr.strip(), err=True)
        sys.exit(1)

    # The running process still has the old module cached, and reload
    # won't help because uv tool installs into a separate virtualenv.
    # Ask the newly installed binary instead.
    ver_result = subprocess.run(
        ["sparkrun", "--version"],
        capture_output=True,
        text=True,
    )
    if ver_result.returncode == 0:
        new_version = ver_result.stdout.strip().rsplit(None, 1)[-1]
        if new_version == old_version:
            click.echo("sparkrun %s is already the latest version." % old_version)
        else:
            click.echo("sparkrun updated: %s -> %s" % (old_version, new_version))
    else:
        click.echo("sparkrun updated (could not determine new version)")

    # Update recipe registries from the newly installed binary — the
    # running process still has old code, so we must shell out.
    if not no_update_registries:
        click.echo()
        click.echo("Updating recipe registries...")
        reg_result = subprocess.run(
            ["sparkrun", "registry", "update"],
            capture_output=False,
        )
        if reg_result.returncode != 0:
            click.echo("Warning: registry update failed (non-fatal).", err=True)


def _run_ssh_diagnose(host_list, user, local_user):
    """Run comprehensive SSH diagnostics on each host.

    Uses interactive password-based SSH (ControlMaster) to connect, then
    collects permission info, sshd settings, and tests pubkey auth
    independently.  Prints structured pass/fail results with remediation.
    """
    import subprocess
    import tempfile

    cross_user = user != local_user

    click.echo("=== SSH Diagnostics ===")
    click.echo("SSH user: %s" % user)
    click.echo("Local user: %s" % local_user)
    click.echo("Cross-user: %s" % ("yes" if cross_user else "no"))
    click.echo("Hosts: %s" % ", ".join(host_list))
    click.echo()

    # Find local public key — use ssh -G to respect ~/.ssh/config IdentityFile
    import os

    local_pubkey = None
    local_pubkey_source = None

    # Ask SSH which identity files it would use for the first host
    try:
        ssh_g_result = subprocess.run(
            ["ssh", "-G", "%s@%s" % (user, host_list[0])],
            capture_output=True, text=True, timeout=5,
        )
        if ssh_g_result.returncode == 0:
            for line in ssh_g_result.stdout.splitlines():
                if line.startswith("identityfile "):
                    idf = os.path.expanduser(line.split(None, 1)[1])
                    pub = idf + ".pub"
                    if os.path.isfile(pub):
                        with open(pub) as f:
                            local_pubkey = f.read().strip()
                        local_pubkey_source = pub
                        break
    except Exception:
        pass

    # Fallback: check standard key filenames
    if not local_pubkey:
        for kf in ("id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"):
            path = os.path.join(os.path.expanduser("~"), ".ssh", kf)
            if os.path.isfile(path):
                with open(path) as f:
                    local_pubkey = f.read().strip()
                local_pubkey_source = path
                break

    if local_pubkey:
        click.echo("Local public key (%s): %s ...%s" % (local_pubkey_source, local_pubkey[:30], local_pubkey[-20:]))
    else:
        click.echo("WARNING: No local SSH public key found!")
    click.echo()

    ssh_opts = [
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=3",
    ]

    control_dir = tempfile.mkdtemp(prefix="sparkrun-diag-")
    os.chmod(control_dir, 0o700)

    all_passed = True

    for h in host_list:
        click.echo("--- Diagnosing %s@%s ---" % (user, h))
        control_path = os.path.join(control_dir, "cm-%%r@%%h:%%p")

        # Step 1: Test pubkey auth (no password, no ControlMaster)
        # Use -v (verbose) so we can show WHY the key was rejected on failure.
        click.echo("  [1/4] Testing pubkey authentication (BatchMode)...")
        pubkey_result = subprocess.run(
            ["ssh", "-v"] + ssh_opts + [
                "-o", "ControlPath=none",
                "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=5",
                "%s@%s" % (user, h), "true",
            ],
            capture_output=True, text=True, timeout=15,
        )
        pubkey_ok = pubkey_result.returncode == 0
        click.echo("        %s" % ("PASS" if pubkey_ok else "FAIL"))

        # Parse verbose SSH output for key diagnostic lines
        ssh_verbose_lines = []
        if not pubkey_ok and pubkey_result.stderr:
            for line in pubkey_result.stderr.splitlines():
                line_lower = line.lower()
                # Capture lines about key offers, rejections, auth methods, and errors
                if any(kw in line_lower for kw in (
                        "offering", "trying", "authentications that can continue",
                        "no more authentication", "permission denied",
                        "key_load", "identity file", "will attempt",
                        "server accepts key", "authentication refused",
                        "pubkey_prepare", "sign_and_send",
                )):
                    ssh_verbose_lines.append(line.strip())
            if ssh_verbose_lines:
                click.echo("        SSH debug (key-related):")
                for vl in ssh_verbose_lines:
                    click.echo("          %s" % vl)

        if not pubkey_ok:
            all_passed = False

        # Step 2: Establish ControlMaster (interactive — may prompt for password)
        click.echo("  [2/4] Establishing authenticated connection (may prompt for password)...")
        cm_result = subprocess.run(
            ["ssh"] + ssh_opts + [
                "-o", "ControlMaster=auto",
                "-o", "ControlPersist=2m",
                "-o", "ControlPath=%s" % control_path,
                      "%s@%s" % (user, h), "true",
            ],
            timeout=60,
        )
        if cm_result.returncode != 0:
            click.echo("        FAIL — cannot connect to %s (even with password)" % h)
            click.echo()
            all_passed = False
            continue

        cm_ssh = [
                     "ssh"] + ssh_opts + [
                     "-o", "ControlMaster=auto",
                     "-o", "ControlPersist=2m",
                     "-o", "ControlPath=%s" % control_path,
                           "%s@%s" % (user, h),
                 ]

        # Step 3: Collect remote diagnostics
        click.echo("  [3/4] Collecting remote diagnostics...")
        diag_script = r"""set -eu
printf 'home_dir=%s\n' "$(eval echo ~)"
printf 'home_perms=%s\n' "$(stat -c '%a' ~ 2>/dev/null || echo unknown)"
printf 'home_owner=%s\n' "$(stat -c '%U' ~ 2>/dev/null || echo unknown)"
printf 'ssh_dir_exists=%s\n' "$(test -d ~/.ssh && echo yes || echo no)"
printf 'ssh_perms=%s\n' "$(stat -c '%a' ~/.ssh 2>/dev/null || echo missing)"
printf 'ak_exists=%s\n' "$(test -f ~/.ssh/authorized_keys && echo yes || echo no)"
printf 'ak_perms=%s\n' "$(stat -c '%a' ~/.ssh/authorized_keys 2>/dev/null || echo missing)"
printf 'ak_lines=%s\n' "$(wc -l < ~/.ssh/authorized_keys 2>/dev/null || echo 0)"
printf 'sshd_ak_file=%s\n' "$(grep -i '^AuthorizedKeysFile' /etc/ssh/sshd_config 2>/dev/null || echo default)"
printf 'sshd_pubkey=%s\n' "$(grep -i '^PubkeyAuthentication' /etc/ssh/sshd_config 2>/dev/null || echo default)"
printf 'sshd_strict=%s\n' "$(grep -i '^StrictModes' /etc/ssh/sshd_config 2>/dev/null || echo default)"
printf 'sshd_allow_users=%s\n' "$(grep -i '^AllowUsers' /etc/ssh/sshd_config 2>/dev/null || echo none)"
printf 'sshd_allow_groups=%s\n' "$(grep -i '^AllowGroups' /etc/ssh/sshd_config 2>/dev/null || echo none)"

# Check sshd drop-in config files (Ubuntu 22.04+ uses Include)
_dropin_dir="/etc/ssh/sshd_config.d"
if [ -d "$_dropin_dir" ]; then
    _dropin_files="$(ls -1 "$_dropin_dir"/*.conf 2>/dev/null | tr '\n' ',' || echo none)"
    printf 'sshd_dropin_files=%s\n' "${_dropin_files%,}"

    # Check for overrides in drop-in files that affect pubkey auth
    _dropin_ak="$(grep -rhi '^AuthorizedKeysFile' "$_dropin_dir"/ 2>/dev/null || echo none)"
    printf 'sshd_dropin_ak_file=%s\n' "$_dropin_ak"
    _dropin_pubkey="$(grep -rhi '^PubkeyAuthentication' "$_dropin_dir"/ 2>/dev/null || echo none)"
    printf 'sshd_dropin_pubkey=%s\n' "$_dropin_pubkey"
    _dropin_accepted="$(grep -rhi '^PubkeyAcceptedAlgorithms\|^PubkeyAcceptedKeyTypes' "$_dropin_dir"/ 2>/dev/null || echo none)"
    printf 'sshd_dropin_accepted_algs=%s\n' "$_dropin_accepted"
    _dropin_allow_users="$(grep -rhi '^AllowUsers' "$_dropin_dir"/ 2>/dev/null || echo none)"
    printf 'sshd_dropin_allow_users=%s\n' "$_dropin_allow_users"
    _dropin_allow_groups="$(grep -rhi '^AllowGroups' "$_dropin_dir"/ 2>/dev/null || echo none)"
    printf 'sshd_dropin_allow_groups=%s\n' "$_dropin_allow_groups"
else
    printf 'sshd_dropin_files=none\n'
    printf 'sshd_dropin_ak_file=none\n'
    printf 'sshd_dropin_pubkey=none\n'
    printf 'sshd_dropin_accepted_algs=none\n'
    printf 'sshd_dropin_allow_users=none\n'
    printf 'sshd_dropin_allow_groups=none\n'
fi

# PubkeyAcceptedAlgorithms in main sshd_config
printf 'sshd_accepted_algs=%s\n' "$(grep -i '^PubkeyAcceptedAlgorithms\|^PubkeyAcceptedKeyTypes' /etc/ssh/sshd_config 2>/dev/null || echo default)"

# Key types present in authorized_keys
printf 'ak_key_types=%s\n' "$(awk '{print $1}' ~/.ssh/authorized_keys 2>/dev/null | sort -u | tr '\n' ',' || echo none)"
"""
        diag_result = subprocess.run(
            cm_ssh + [diag_script],
            capture_output=True, text=True, timeout=15,
        )

        diag = {}
        if diag_result.returncode == 0:
            for line in diag_result.stdout.strip().splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    diag[k.strip()] = v.strip()

        # Step 4: Check if local pubkey is in authorized_keys
        key_installed = False
        if local_pubkey and cross_user:
            click.echo("  [4/4] Checking if local pubkey is in authorized_keys...")
            # Extract just the key type + key data (no comment) for matching
            key_parts = local_pubkey.split()
            if len(key_parts) >= 2:
                key_match = "%s %s" % (key_parts[0], key_parts[1])
                grep_result = subprocess.run(
                    cm_ssh + ["grep -cF '%s' ~/.ssh/authorized_keys 2>/dev/null || echo 0" % key_match],
                    capture_output=True, text=True, timeout=10,
                )
                count = grep_result.stdout.strip()
                key_installed = count != "0"
                click.echo("        Key present: %s" % ("yes" if key_installed else "NO"))
        else:
            click.echo("  [4/4] Key check: %s" % ("skipped (same user)" if not cross_user else "skipped (no local key)"))

        # Print structured results
        click.echo()
        click.echo("  Results for %s@%s:" % (user, h))

        home_perms = diag.get("home_perms", "unknown")
        home_ok = home_perms in ("700", "750", "755")
        click.echo("    Home dir:           %s (perms: %s) %s" % (
            diag.get("home_dir", "?"),
            home_perms,
            "PASS" if home_ok else "FAIL — must not be group/world-writable",
        ))

        ssh_perms = diag.get("ssh_perms", "missing")
        ssh_ok = ssh_perms == "700"
        click.echo("    .ssh/ dir:          perms %s %s" % (
            ssh_perms,
            "PASS" if ssh_ok else ("FAIL" if ssh_perms != "missing" else "MISSING"),
        ))

        ak_perms = diag.get("ak_perms", "missing")
        ak_ok = ak_perms == "600"
        click.echo("    authorized_keys:    perms %s, %s line(s) %s" % (
            ak_perms,
            diag.get("ak_lines", "0"),
            "PASS" if ak_ok else ("FAIL" if ak_perms != "missing" else "MISSING"),
        ))

        ak_file = diag.get("sshd_ak_file", "default")
        ak_file_ok = ak_file == "default"
        click.echo("    AuthorizedKeysFile: %s %s" % (
            ak_file,
            "PASS" if ak_file_ok else "NOTE — non-default location",
        ))

        sshd_strict = diag.get("sshd_strict", "default")
        click.echo("    StrictModes:        %s" % sshd_strict)

        sshd_pubkey = diag.get("sshd_pubkey", "default")
        pubkey_setting_ok = sshd_pubkey == "default" or "yes" in sshd_pubkey.lower()
        click.echo("    PubkeyAuth:         %s %s" % (
            sshd_pubkey,
            "PASS" if pubkey_setting_ok else "FAIL — pubkey auth disabled!",
        ))

        allow_users = diag.get("sshd_allow_users", "none")
        allow_groups = diag.get("sshd_allow_groups", "none")
        if allow_users != "none":
            click.echo("    AllowUsers:         %s — verify '%s' is listed" % (allow_users, user))
        if allow_groups != "none":
            click.echo("    AllowGroups:        %s — verify '%s' is in a listed group" % (allow_groups, user))

        # Accepted key algorithms (main config)
        accepted_algs = diag.get("sshd_accepted_algs", "default")
        click.echo("    AcceptedAlgorithms: %s" % accepted_algs)

        # Key types in authorized_keys
        ak_key_types = diag.get("ak_key_types", "none")
        if ak_key_types and ak_key_types != "none":
            click.echo("    AK key types:       %s" % ak_key_types.rstrip(","))

        # Drop-in config overrides
        dropin_files = diag.get("sshd_dropin_files", "none")
        if dropin_files != "none":
            click.echo("    sshd drop-ins:      %s" % dropin_files)
            dropin_ak = diag.get("sshd_dropin_ak_file", "none")
            if dropin_ak != "none":
                click.echo("    ↳ AuthorizedKeysFile override: %s" % dropin_ak)
            dropin_pubkey = diag.get("sshd_dropin_pubkey", "none")
            if dropin_pubkey != "none":
                click.echo("    ↳ PubkeyAuthentication override: %s" % dropin_pubkey)
            dropin_accepted = diag.get("sshd_dropin_accepted_algs", "none")
            if dropin_accepted != "none":
                click.echo("    ↳ AcceptedAlgorithms override: %s" % dropin_accepted)
            dropin_allow_users = diag.get("sshd_dropin_allow_users", "none")
            if dropin_allow_users != "none":
                click.echo("    ↳ AllowUsers override: %s — verify '%s' is listed" % (dropin_allow_users, user))
            dropin_allow_groups = diag.get("sshd_dropin_allow_groups", "none")
            if dropin_allow_groups != "none":
                click.echo("    ↳ AllowGroups override: %s — verify '%s' is in a listed group" % (dropin_allow_groups, user))

        click.echo("    Pubkey auth test:   %s" % ("PASS" if pubkey_ok else "FAIL"))
        if cross_user:
            click.echo("    Local key installed: %s" % ("PASS" if key_installed else "FAIL"))
        click.echo()

        # Remediation suggestions
        if not pubkey_ok:
            click.echo("  Suggested fixes for %s:" % h)
            if not home_ok and home_perms != "unknown":
                click.echo("    chmod go-w ~%s    # Fix home dir permissions (most likely cause)" % user)
            if not ak_ok and ak_perms != "missing":
                click.echo("    chmod 600 ~%s/.ssh/authorized_keys" % user)
            if not ssh_ok and ssh_perms != "missing":
                click.echo("    chmod 700 ~%s/.ssh" % user)
            if not ak_file_ok:
                click.echo("    Check sshd_config: keys may need to go in %s" % ak_file)
            if not pubkey_setting_ok:
                click.echo("    Enable PubkeyAuthentication in /etc/ssh/sshd_config and restart sshd")
            if cross_user and not key_installed:
                click.echo("    Re-run 'sparkrun setup ssh' to install your public key")

            # Drop-in specific remediation
            dropin_ak = diag.get("sshd_dropin_ak_file", "none")
            if dropin_ak != "none":
                click.echo("    ** Drop-in override detected: AuthorizedKeysFile = %s" % dropin_ak)
                click.echo("       Your key may need to go in that location instead of ~/.ssh/authorized_keys")
                click.echo("       Check files in /etc/ssh/sshd_config.d/ on the remote host")
            dropin_pubkey = diag.get("sshd_dropin_pubkey", "none")
            if dropin_pubkey != "none" and "no" in dropin_pubkey.lower():
                click.echo("    ** Drop-in override DISABLES pubkey auth: %s" % dropin_pubkey)
                click.echo("       Check files in /etc/ssh/sshd_config.d/ on the remote host")
            dropin_accepted = diag.get("sshd_dropin_accepted_algs", "none")
            if dropin_accepted != "none":
                click.echo("    ** Drop-in restricts accepted key algorithms: %s" % dropin_accepted)
                click.echo("       Ensure your key type is in the allowed list")
            dropin_allow_users = diag.get("sshd_dropin_allow_users", "none")
            if dropin_allow_users != "none":
                click.echo("    ** Drop-in restricts AllowUsers: %s — verify '%s' is listed" % (dropin_allow_users, user))
            dropin_allow_groups = diag.get("sshd_dropin_allow_groups", "none")
            if dropin_allow_groups != "none":
                click.echo("    ** Drop-in restricts AllowGroups: %s — verify '%s' is in a listed group" % (dropin_allow_groups, user))

            # Check for key mismatch: SSH verbose output shows which key was offered
            if not pubkey_ok and ssh_verbose_lines and local_pubkey_source:
                offered_key_path = None
                for vl in ssh_verbose_lines:
                    if "offering public key:" in vl.lower() or "will attempt key:" in vl.lower():
                        # Extract path from lines like "debug1: Offering public key: /home/me/.ssh/id_ed25519_shared ..."
                        parts = vl.split(":", 2)
                        if len(parts) >= 3:
                            tokens = parts[2].strip().split()
                            if tokens:
                                offered_key_path = tokens[0]
                                break
                if offered_key_path:
                    # Compare the key SSH tried with what we'd install
                    expected_private = local_pubkey_source.removesuffix(".pub") if local_pubkey_source else None
                    if expected_private and offered_key_path != expected_private:
                        click.echo("    ** KEY MISMATCH: SSH is offering '%s'" % offered_key_path)
                        click.echo("       but sparkrun detected '%s' as the local key." % local_pubkey_source)
                        click.echo("       Check your ~/.ssh/config IdentityFile settings.")

            # If everything looks correct but still fails, point to verbose output
            if (home_ok and ssh_ok and ak_ok and ak_file_ok and pubkey_setting_ok
                    and (not cross_user or key_installed)
                    and dropin_ak == "none" and dropin_pubkey == "none"):
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

        # Clean up ControlMaster
        subprocess.run(
            ["ssh"] + ssh_opts + [
                "-o", "ControlPath=%s" % control_path,
                "-O", "exit",
                      "%s@%s" % (user, h),
            ],
            capture_output=True, timeout=5,
        )

    # Summary
    click.echo("=== Diagnostics Complete ===")
    if all_passed:
        click.echo("All hosts passed pubkey authentication.")
    else:
        click.echo("Some hosts failed. See remediation suggestions above.")

    # Cleanup
    import shutil
    shutil.rmtree(control_dir, ignore_errors=True)


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

    After the mesh is established, sparkrun automatically discovers
    additional network IPs (InfiniBand, CX7) on cluster hosts and
    distributes their host keys so inter-node SSH works over all
    networks. Use --no-discover-ips to skip this phase.

    By default, the machine running sparkrun is included in the mesh
    (--include-self). Use --no-include-self to exclude it.

    You will be prompted for passwords on first connection to each host.

    Examples:

      sparkrun setup ssh --hosts 192.168.11.13,192.168.11.14

      sparkrun setup ssh --cluster mylab --user ubuntu

      sparkrun setup ssh --cluster mylab --extra-hosts 10.0.0.1

      sparkrun setup ssh --hosts 10.0.0.1,10.0.0.2 --no-discover-ips

      sparkrun setup ssh --cluster mylab --diagnose
    """
    import os

    from sparkrun.core.hosts import resolve_hosts
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.primitives import local_ip_for

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

    ok = _run_ssh_mesh(
        host_list,
        user,
        cluster_hosts=cluster_hosts,
        ssh_key=config.ssh_key,
        discover_ips=discover_ips,
        dry_run=dry_run,
        control_is_member=_resolved_loopback or (self_host is not None and self_host in set(cluster_hosts)),
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
@click.pass_context
def setup_cx7(ctx, hosts, hosts_file, cluster_name, user, dry_run, force, mtu, subnet1, subnet2, topology):
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

      sparkrun setup cx7 --hosts 10.24.11.13,10.24.11.14

      sparkrun setup cx7 --cluster mylab --dry-run

      sparkrun setup cx7 --cluster mylab --topology ring

      sparkrun setup cx7 --cluster mylab --subnet1 192.168.11.0/24 --subnet2 192.168.12.0/24

      sparkrun setup cx7 --cluster mylab --force
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.networking import (
        CX7Topology,
        configure_cx7_host,
        detect_cx7_for_hosts,
        detect_topology,
        select_subnets,
        select_subnets_for_topology,
        plan_cluster_cx7,
        plan_ring_cx7,
        apply_cx7_plan,
    )

    # Validate subnet pair
    if (subnet1 is None) != (subnet2 is None):
        click.echo("Error: --subnet1 and --subnet2 must be specified together.", err=True)
        sys.exit(1)

    import os

    from ._sudo import ensure_sudo_password

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

    # Step 1: Detect CX7 interfaces (with MACs)
    detections = detect_cx7_for_hosts(host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run)

    # Check all hosts have CX7
    no_cx7 = [h for h, d in detections.items() if not d.detected]
    if no_cx7:
        click.echo("Warning: No CX7 interfaces on: %s" % ", ".join(no_cx7), err=True)

    hosts_with_cx7 = {h: d for h, d in detections.items() if d.detected}
    if not hosts_with_cx7:
        click.echo("Error: No CX7 interfaces detected on any host.", err=True)
        sys.exit(1)

    # Lazy sudo handling — uses ensure_sudo_password() which tests NOPASSWD,
    # prompts, verifies, and supports cross-user fallback.  The sudo_ssh_kwargs
    # may differ from ssh_kwargs when the cluster user doesn't have sudo but
    # another user does (indirect sudo).
    sudo_ssh_kwargs = dict(ssh_kwargs)
    sudo_password = None
    sudo_hosts_needing_pw: set[str] = set()

    def _ensure_sudo() -> str | None:
        """Acquire sudo password lazily.  Returns password or None (NOPASSWD works)."""
        nonlocal sudo_password, sudo_ssh_kwargs
        if sudo_password is not None:
            return sudo_password
        if dry_run:
            return None

        default_user = os.environ.get("USER", "")
        pw, indirect_user = ensure_sudo_password(
            host_list,
            user,
            ssh_kwargs,
            sudo_ssh_kwargs=sudo_ssh_kwargs,
            dry_run=dry_run,
            allow_indirect=True,
            default_user=default_user,
        )
        if pw is None:
            return None  # NOPASSWD works

        sudo_password = pw
        if indirect_user:
            sudo_ssh_kwargs = dict(ssh_kwargs, ssh_user=indirect_user)

        # Mark which hosts actually need the password
        sudo_hosts_needing_pw.update(
            h for h, d in detections.items()
            if d.detected and not d.sudo_ok
        )
        return sudo_password

    # Step 2: Topology determination
    effective_topology = CX7Topology.SWITCH  # default
    topology_result = None

    if topology == "ring":
        effective_topology = CX7Topology.RING
        # Fail fast: ring requires exactly 3 hosts with 2 ports each
        if len(hosts_with_cx7) != 3:
            click.echo(
                "Error: ring topology requires exactly 3 hosts with CX7, found %d" % len(hosts_with_cx7),
                err=True,
            )
            sys.exit(1)
        from sparkrun.orchestration.networking import _group_interfaces_by_port
        for h, det in hosts_with_cx7.items():
            port_groups = _group_interfaces_by_port(det.interfaces)
            if len(port_groups) < 2:
                click.echo(
                    "Error: %s: ring topology requires 2 physical ports (4 interfaces), "
                    "but only %d port group(s) found (%d interfaces)"
                    % (h, len(port_groups), len(det.interfaces)),
                    err=True,
                )
                sys.exit(1)
    elif topology == "direct":
        effective_topology = CX7Topology.DIRECT
    elif topology == "switch":
        effective_topology = CX7Topology.SWITCH
    elif topology == "auto":
        n_hosts = len(hosts_with_cx7)
        # Check if hosts have >= 4 interfaces (ring candidate)
        has_4_ifaces = all(len(d.interfaces) >= 4 for d in hosts_with_cx7.values())

        if n_hosts == 3 and has_4_ifaces:
            # Run topology detection via MAC/ARP — ring candidate
            click.echo("Detecting topology via neighbor discovery...")
            topology_result = detect_topology(detections, host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
            effective_topology = topology_result.topology
            click.echo("Detected topology: %s" % effective_topology.value)
        else:
            # 2-node or switch: can't reliably distinguish direct vs switch
            # at L2 without LLDP.  Use --topology direct to override.
            effective_topology = CX7Topology.SWITCH

    # For explicit ring topology without detection, create empty result
    if effective_topology == CX7Topology.RING and topology_result is None:
        if not dry_run:
            click.echo("Running topology detection for ring configuration...")
            topology_result = detect_topology(detections, host_list, ssh_kwargs=ssh_kwargs, dry_run=dry_run)
        else:
            from sparkrun.orchestration.networking import CX7TopologyResult
            topology_result = CX7TopologyResult(topology=CX7Topology.RING)

    click.echo()
    # For 2-node auto, we can't distinguish direct vs switch — show both
    if effective_topology == CX7Topology.SWITCH and topology == "auto" and len(hosts_with_cx7) <= 2:
        click.echo("Topology: switch/direct")
    else:
        click.echo("Topology: %s" % effective_topology.value)

    # Step 3: Select subnets
    try:
        if effective_topology == CX7Topology.RING:
            all_subnets = select_subnets_for_topology(detections, effective_topology)
            click.echo("Subnets: %s" % ", ".join(str(s) for s in all_subnets))
        else:
            s1, s2 = select_subnets(detections, override1=subnet1, override2=subnet2)
            all_subnets = [s1, s2]
            click.echo("Subnets: %s, %s" % (s1, s2))
    except RuntimeError as e:
        click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    click.echo("MTU: %d" % mtu)
    click.echo()

    # Step 4: Plan
    if effective_topology == CX7Topology.RING:
        plan = plan_ring_cx7(detections, topology_result, all_subnets, mtu=mtu, force=force)
    else:
        plan = plan_cluster_cx7(detections, all_subnets[0], all_subnets[1], mtu=mtu, force=force)

    # Display plan
    for hp in plan.host_plans:
        det = detections.get(hp.host)
        mgmt_label = " (%s)" % det.mgmt_ip if det and det.mgmt_ip else ""
        click.echo("  %s%s" % (hp.host, mgmt_label))
        for a in hp.assignments:
            status = "OK" if not hp.needs_change else "configure"
            click.echo("    %-20s -> %s/%d  MTU %d  [%s]" % (a.iface_name, a.ip, plan.prefix_len, plan.mtu, status))
        if not hp.assignments and hp.needs_change:
            click.echo("    %s" % hp.reason)
        click.echo()

    # Show warnings
    for w in plan.warnings:
        click.echo("Warning: %s" % w, err=True)

    # Plan-level errors (e.g. insufficient ports for ring)
    if plan.errors and not plan.host_plans:
        for e in plan.errors:
            click.echo("Error: %s" % e, err=True)
        sys.exit(1)

    # Step 5: Check if all valid
    if plan.all_valid and not force:
        click.echo("All hosts already configured. Use --force to reconfigure.")
        return

    # Count
    needs_config = sum(1 for hp in plan.host_plans if hp.needs_change and len(hp.assignments) >= 2)
    already_ok = sum(1 for hp in plan.host_plans if not hp.needs_change)
    has_errors = sum(1 for hp in plan.host_plans if hp.needs_change and len(hp.assignments) < 2)

    if needs_config == 0:
        if has_errors:
            click.echo("Error: %d host(s) have issues that prevent configuration." % has_errors, err=True)
            for e in plan.errors:
                click.echo("  %s" % e, err=True)
            sys.exit(1)
        click.echo("No hosts need configuration changes.")
        return

    if dry_run:
        click.echo("[dry-run] Would configure %d host(s), %d already valid." % (needs_config, already_ok))
        return

    # Confirm before applying
    if not force and not click.confirm("Apply changes to %d host(s)?" % needs_config, default=True):
        click.echo("Aborted.")
        return

    # Step 7: Apply — acquire sudo now if not already prompted
    _ensure_sudo()

    click.echo("Applying configuration to %d host(s)..." % needs_config)
    results = apply_cx7_plan(
        plan,
        ssh_kwargs=sudo_ssh_kwargs,
        dry_run=dry_run,
        sudo_password=sudo_password,
        sudo_hosts=sudo_hosts_needing_pw,
    )

    # Build a map of host -> result for easy lookup
    result_map = {r.host: r for r in results}

    # Check for sudo failures and retry with per-host passwords
    if sudo_hosts_needing_pw and not dry_run:
        failed_sudo_hosts = [r.host for r in results if not r.success and r.host in sudo_hosts_needing_pw]
        if failed_sudo_hosts:
            click.echo()
            click.echo("Sudo authentication failed on %d host(s). Retrying individually..." % len(failed_sudo_hosts))
            host_plan_map = {hp.host: hp for hp in plan.host_plans}
            for fhost in failed_sudo_hosts:
                hp = host_plan_map.get(fhost)
                if not hp:
                    continue
                per_host_pw = click.prompt("[sudo] password for %s @ %s" % (user, fhost), hide_input=True)
                retry_result = configure_cx7_host(
                    hp,
                    mtu=plan.mtu,
                    prefix_len=plan.prefix_len,
                    ssh_kwargs=sudo_ssh_kwargs,
                    dry_run=dry_run,
                    sudo_password=per_host_pw,
                )
                result_map[fhost] = retry_result

    # Collect final results in plan order
    final_results = [result_map[hp.host] for hp in plan.host_plans if hp.host in result_map]
    configured = sum(1 for r in final_results if r.success)
    failed = sum(1 for r in final_results if not r.success)

    for r in final_results:
        if not r.success:
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

    # Persist topology to cluster YAML (explicit --cluster or default cluster)
    effective_cluster = cluster_name
    if not effective_cluster:
        try:
            mgr = _get_cluster_manager()
            effective_cluster = mgr.get_default() if mgr else None
        except Exception:
            pass
    if effective_cluster and effective_topology != CX7Topology.UNKNOWN:
        try:
            mgr = _get_cluster_manager()
            if mgr:
                mgr.update(effective_cluster, topology=effective_topology.value)
                click.echo("Saved topology '%s' to cluster '%s'." % (effective_topology.value, effective_cluster))
        except Exception as e:
            click.echo("Warning: could not save topology to cluster: %s" % e, err=True)

    subnet_strs = [str(s) for s in all_subnets]
    if configured and not dry_run:
        cx7_ips = [a.ip for hp in plan.host_plans for a in hp.assignments if a.ip]
        _record_setup_phase(
            cluster_name, user, host_list, "cx7",
            subnets=subnet_strs, cx7_ips=cx7_ips,
            netplan_file="/etc/netplan/40-cx7.yaml",
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

      sparkrun setup docker-group --hosts 10.24.11.13,10.24.11.14

      sparkrun setup docker-group --cluster mylab

      sparkrun setup docker-group --cluster mylab --user ubuntu
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.sudo import run_with_sudo_fallback, run_sudo_script_on_host

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

    click.echo("Ensuring user '%s' is in the docker group on %d host(s)..." % (user, len(host_list)))
    click.echo()

    script = _DOCKER_GROUP_SCRIPT.format(user=user)
    fallback = _DOCKER_GROUP_FALLBACK_SCRIPT.format(user=user)

    result_map, still_failed = run_with_sudo_fallback(
        host_list,
        script,
        fallback,
        ssh_kwargs,
        dry_run=dry_run,
    )

    # Report immediate successes
    for h in host_list:
        r = result_map.get(h)
        if r and r.success:
            click.echo("  [OK]   %s: %s" % (h, _docker_group_summary(r.stdout, user=user)))

    # Prompt and retry if needed
    if still_failed and not dry_run:
        sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
        for h in still_failed:
            r = run_sudo_script_on_host(
                h,
                fallback,
                sudo_password,
                ssh_kwargs=ssh_kwargs,
                timeout=30,
                dry_run=dry_run,
            )
            result_map[h] = r

    # Final summary
    ok_count = sum(1 for h in host_list if result_map.get(h) and result_map[h].success)
    fail_count = sum(1 for h in host_list if result_map.get(h) and not result_map[h].success)

    for h in host_list:
        r = result_map.get(h)
        if r and not r.success:
            click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)
        elif r and r.success and h in (still_failed or []):
            click.echo("  [OK]   %s: %s" % (h, _docker_group_summary(r.stdout, user=user)))

    click.echo()
    parts = []
    if ok_count:
        parts.append("%d OK" % ok_count)
    if fail_count:
        parts.append("%d failed" % fail_count)
    click.echo("Results: %s." % ", ".join(parts) if parts else "No hosts processed.")

    if ok_count and not dry_run:
        _record_setup_phase(cluster_name, user, host_list, "docker_group")

    if ok_count and any("added" in (result_map.get(h).stdout if result_map.get(h) else "") for h in host_list):
        click.echo()
        click.echo("Note: Users newly added to the docker group must re-login")
        click.echo("(or run 'newgrp docker') for the change to take effect.")

    if fail_count:
        sys.exit(1)


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

      sparkrun setup fix-permissions --hosts 10.24.11.13,10.24.11.14

      sparkrun setup fix-permissions --cluster mylab

      sparkrun setup fix-permissions --cluster mylab --cache-dir /data/hf-cache

      sparkrun setup fix-permissions --cluster mylab --save-sudo

      sparkrun setup fix-permissions --cluster mylab --dry-run
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.sudo import run_with_sudo_fallback, run_sudo_script_on_host

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

    # Resolve cache path
    cache_path = cache_dir  # None means use getent-based home detection

    click.echo("Fixing file permissions for user '%s' on %d host(s)..." % (user, len(host_list)))
    if cache_path:
        click.echo("Cache directory: %s" % cache_path)
    click.echo()

    sudo_password = None

    from sparkrun.scripts import read_script

    # --save-sudo: install scoped sudoers entry on each host
    if save_sudo:
        click.echo("Installing sudoers entry for passwordless chown...")
        from sparkrun.utils.shell import validate_unix_username
        import re as _re

        validate_unix_username(user)
        safe_cache_dir = cache_path or ""
        if safe_cache_dir and not _re.fullmatch(r"[/a-zA-Z0-9_.~-]+", safe_cache_dir):
            raise click.UsageError("cache_dir contains unsafe characters: %r" % safe_cache_dir)
        sudoers_script = read_script("fix_permissions_sudoers.sh").format(
            user=user,
            cache_dir=safe_cache_dir,
        )

        if dry_run:
            click.echo("  [dry-run] Would install sudoers entry on %d host(s):" % len(host_list))
            for h in host_list:
                click.echo("    %s: /etc/sudoers.d/sparkrun-chown-%s" % (h, user))
            click.echo()
        else:
            sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
            sudoers_ok = 0
            sudoers_fail = 0
            for h in host_list:
                r = run_sudo_script_on_host(
                    h,
                    sudoers_script,
                    sudo_password,
                    ssh_kwargs=ssh_kwargs,
                    timeout=300,
                    dry_run=False,
                )
                if r.success:
                    sudoers_ok += 1
                    click.echo("  [OK]   %s: %s" % (h, r.stdout.strip()))
                else:
                    sudoers_fail += 1
                    click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)
            click.echo("Sudoers install: %d OK, %d failed." % (sudoers_ok, sudoers_fail))
            if sudoers_ok:
                _record_setup_phase(
                    cluster_name, user, host_list, "sudoers",
                    files=["/etc/sudoers.d/sparkrun-chown-%s" % user],
                )
            click.echo()

    # Generate the chown script with sudo -n (non-interactive).
    chown_script = read_script("fix_permissions.sh").format(
        user=user,
        cache_dir=cache_path or "",
    )

    # Password-based fallback script (no sudo prefix — run_remote_sudo_script runs as root)
    fallback_script = read_script("fix_permissions_fallback.sh").format(
        user=user,
        cache_dir=cache_path or "",
    )

    # Try non-interactive sudo, then password-based fallback
    result_map, still_failed = run_with_sudo_fallback(
        host_list,
        chown_script,
        fallback_script,
        ssh_kwargs,
        dry_run=dry_run,
        sudo_password=sudo_password,
    )

    # If hosts failed without a password, prompt and retry
    if still_failed and not dry_run:
        if sudo_password is None:
            sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
            # Re-run fallback with the password for failed hosts
            result_map, still_failed = run_with_sudo_fallback(
                still_failed,
                chown_script,
                fallback_script,
                ssh_kwargs,
                dry_run=dry_run,
                sudo_password=sudo_password,
            )

        # Retry individually on per-host sudo failures
        if still_failed and sudo_password:
            click.echo()
            click.echo("Sudo authentication failed on %d host(s). Retrying individually..." % len(still_failed))
            for fhost in still_failed:
                per_host_pw = click.prompt("[sudo] password for %s @ %s" % (user, fhost), hide_input=True)
                retry_result = run_sudo_script_on_host(
                    fhost,
                    fallback_script,
                    per_host_pw,
                    ssh_kwargs=ssh_kwargs,
                    timeout=300,
                    dry_run=dry_run,
                )
                result_map[fhost] = retry_result

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
            click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)

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

      sparkrun setup clear-cache --hosts 10.24.11.13,10.24.11.14

      sparkrun setup clear-cache --cluster mylab

      sparkrun setup clear-cache --cluster mylab --save-sudo

      sparkrun setup clear-cache --cluster mylab --dry-run
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.sudo import run_with_sudo_fallback, run_sudo_script_on_host

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

    click.echo("Clearing page cache on %d host(s)..." % len(host_list))
    click.echo()

    sudo_password = None

    from sparkrun.scripts import read_script

    # --save-sudo: install scoped sudoers entry on each host
    if save_sudo:
        click.echo("Installing sudoers entry for passwordless cache clearing...")
        from sparkrun.utils.shell import validate_unix_username

        validate_unix_username(user)
        sudoers_script = read_script("clear_cache_sudoers.sh").format(user=user)

        if dry_run:
            click.echo("  [dry-run] Would install sudoers entry on %d host(s):" % len(host_list))
            for h in host_list:
                click.echo("    %s: /etc/sudoers.d/sparkrun-dropcaches-%s" % (h, user))
            click.echo()
        else:
            sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
            sudoers_ok = 0
            sudoers_fail = 0
            for h in host_list:
                r = run_sudo_script_on_host(
                    h,
                    sudoers_script,
                    sudo_password,
                    ssh_kwargs=ssh_kwargs,
                    timeout=300,
                    dry_run=False,
                )
                if r.success:
                    sudoers_ok += 1
                    click.echo("  [OK]   %s: %s" % (h, r.stdout.strip()))
                else:
                    sudoers_fail += 1
                    click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)
            click.echo("Sudoers install: %d OK, %d failed." % (sudoers_ok, sudoers_fail))
            if sudoers_ok:
                _record_setup_phase(
                    cluster_name, user, host_list, "sudoers",
                    files=["/etc/sudoers.d/sparkrun-dropcaches-%s" % user],
                )
            click.echo()

    # Generate the drop_caches script with sudo -n (non-interactive).
    drop_script = read_script("clear_cache.sh")

    # Password-based fallback script (no sudo — run_remote_sudo_script runs as root)
    fallback_script = read_script("clear_cache_fallback.sh")

    # Try non-interactive sudo, then password-based fallback
    result_map, still_failed = run_with_sudo_fallback(
        host_list,
        drop_script,
        fallback_script,
        ssh_kwargs,
        dry_run=dry_run,
        sudo_password=sudo_password,
    )

    # If hosts failed without a password, prompt and retry
    if still_failed and not dry_run:
        if sudo_password is None:
            sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)
            result_map, still_failed = run_with_sudo_fallback(
                still_failed,
                drop_script,
                fallback_script,
                ssh_kwargs,
                dry_run=dry_run,
                sudo_password=sudo_password,
            )

        # Retry individually on per-host sudo failures
        if still_failed and sudo_password:
            click.echo()
            click.echo("Sudo authentication failed on %d host(s). Retrying individually..." % len(still_failed))
            for fhost in still_failed:
                per_host_pw = click.prompt("[sudo] password for %s @ %s" % (user, fhost), hide_input=True)
                retry_result = run_sudo_script_on_host(
                    fhost,
                    fallback_script,
                    per_host_pw,
                    ssh_kwargs=ssh_kwargs,
                    timeout=300,
                    dry_run=dry_run,
                )
                result_map[fhost] = retry_result

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
            click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)

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

    By default, sparkrun configures earlyoom to prefer killing inference
    workload processes (vllm, sglang, llama-server, trtllm, python) and
    to avoid killing system services (sshd, systemd, dockerd).

    Use --prefer/--avoid to add additional process patterns.

    Requires sudo on target hosts (apt-get install, systemctl).

    Examples:

      sparkrun setup earlyoom --hosts 192.168.11.13,192.168.11.14

      sparkrun setup earlyoom --cluster mylab

      sparkrun setup earlyoom --cluster mylab --prefer "my-app,worker"

      sparkrun setup earlyoom --cluster mylab --dry-run
    """
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.orchestration.sudo import run_with_sudo_fallback, run_sudo_script_on_host

    config = SparkrunConfig()
    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)

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

    # Try non-interactive sudo, then password-based fallback
    result_map, still_failed = run_with_sudo_fallback(
        host_list,
        install_script,
        fallback_script,
        ssh_kwargs,
        dry_run=dry_run,
    )

    # Report hosts that succeeded immediately
    for h in host_list:
        r = result_map.get(h)
        if r and r.success:
            click.echo("  [OK]   %s: %s" % (h, _earlyoom_summary(r.stdout)))

    # If hosts failed without a password, prompt and retry
    if still_failed and not dry_run:
        sudo_password = click.prompt("[sudo] password for %s" % user, hide_input=True)

        # Run fallback with progress — report each host as it completes
        click.echo("Configuring %d host(s)..." % len(still_failed))
        remaining = list(still_failed)
        for h in remaining:
            click.echo("  %-20s ..." % h, nl=False)
            r = run_sudo_script_on_host(
                h,
                fallback_script,
                sudo_password,
                ssh_kwargs=ssh_kwargs,
                timeout=300,
                dry_run=dry_run,
            )
            result_map[h] = r
            if r.success:
                click.echo(" %s" % _earlyoom_summary(r.stdout))
            else:
                click.echo(" FAILED")

        still_failed = [h for h in remaining if not result_map.get(h) or not result_map[h].success]

        # Retry individually on per-host sudo failures
        if still_failed and sudo_password:
            click.echo()
            click.echo("Sudo authentication failed on %d host(s). Retrying individually..." % len(still_failed))
            for fhost in still_failed:
                per_host_pw = click.prompt("[sudo] password for %s @ %s" % (user, fhost), hide_input=True)
                click.echo("  %-20s ..." % fhost, nl=False)
                retry_result = run_sudo_script_on_host(
                    fhost,
                    fallback_script,
                    per_host_pw,
                    ssh_kwargs=ssh_kwargs,
                    timeout=300,
                    dry_run=dry_run,
                )
                result_map[fhost] = retry_result
                if retry_result.success:
                    click.echo(" %s" % _earlyoom_summary(retry_result.stdout))
                else:
                    click.echo(" FAILED")

    # Final summary
    ok_count = sum(1 for h in host_list if result_map.get(h) and result_map[h].success)
    fail_count = sum(1 for h in host_list if result_map.get(h) and not result_map[h].success)

    for h in host_list:
        r = result_map.get(h)
        if r and not r.success:
            click.echo("  [FAIL] %s: %s" % (h, r.stderr.strip()[:200]), err=True)

    click.echo()
    parts = []
    if ok_count:
        parts.append("%d configured" % ok_count)
    if fail_count:
        parts.append("%d failed" % fail_count)
    click.echo("Results: %s." % ", ".join(parts) if parts else "No hosts processed.")

    if ok_count and not dry_run:
        installed_pkg = any(
            "INSTALLING:" in (result_map.get(h) and result_map[h].stdout or "")
            for h in host_list
        )
        _record_setup_phase(
            cluster_name, user, host_list, "earlyoom",
            installed_package=installed_pkg,
        )

    if ok_count:
        click.echo()
        click.echo("Thanks to @shahizat for posting this idea on the DGX forums!")

    if fail_count:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Founders Edition System Update
# ---------------------------------------------------------------------------


_FE_UPDATE_STEPS = [
    ("Updating package lists", "apt update"),
    ("Upgrading packages", "DEBIAN_FRONTEND=noninteractive apt dist-upgrade -y"),
    ("Refreshing firmware metadata", "fwupdmgr refresh --force"),
    ("Upgrading firmware", "fwupdmgr upgrade -y --no-reboot-check"),
]


@setup.command("fe-system-update", hidden=True)
@host_options
@click.option("--user", default=None, help="SSH user (default: cluster user or $USER)")
@dry_run_option
@click.pass_context
def setup_fe_system_update(ctx, hosts, hosts_file, cluster_name, user, dry_run):
    """Run a full system update on DGX Spark Founders Edition hosts.

    Updates system packages (apt), firmware (fwupdmgr), and reboots.
    Can target the local machine, cluster hosts, or both.

    \b
    Steps performed (as root):
      1. apt update
      2. apt dist-upgrade
      3. fwupdmgr refresh
      4. fwupdmgr upgrade
      5. reboot
    """
    from sparkrun.core.config import SparkrunConfig

    from .._common import _resolve_setup_context
    from ._sudo import ensure_sudo_password

    # TODO: should use same sctx init as we typically use
    config = SparkrunConfig()

    # --- Step 1: Determine target hosts ---
    # If explicit hosts/cluster provided, use those directly
    explicit_hosts = hosts or hosts_file or cluster_name
    if explicit_hosts:
        host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config, user)
    else:
        # Interactive: ask local vs cluster
        click.echo("Where would you like to run the system update?")
        click.echo()
        click.echo("  1) Local machine only")

        # Try to list cluster hosts
        mgr = _get_cluster_manager()
        default_cluster = mgr.get_default()
        cluster_hosts = []
        if default_cluster:
            try:
                cdata = mgr.get_cluster(default_cluster)
                cluster_hosts = cdata.get("hosts", [])
            except Exception:
                pass

        if cluster_hosts:
            click.echo("  2) Cluster hosts (%s): %s" % (default_cluster, ", ".join(cluster_hosts)))
            click.echo("  3) All (local + cluster hosts)")
            choice = click.prompt("Selection", type=click.IntRange(1, 3), default=2)
        else:
            click.echo("  (No default cluster configured — cluster option unavailable)")
            choice = click.prompt("Selection", type=click.IntRange(1, 1), default=1)

        click.echo()

        import socket

        if choice == 1:
            host_list = [socket.gethostname()]
        elif choice == 2:
            host_list = list(cluster_hosts)
        else:
            local = socket.gethostname()
            host_list = [local] + [h for h in cluster_hosts if h != local]

        import os

        if user is None:
            user = config.ssh_user or os.environ.get("USER", "root")
        from sparkrun.orchestration.primitives import build_ssh_kwargs

        ssh_kwargs = build_ssh_kwargs(config)
        if user:
            ssh_kwargs["ssh_user"] = user

    # TODO: guard to detect non-founders edition hosts and block them

    # --- Step 2: Confirm the activity ---
    click.echo("Founders Edition System Update")
    click.echo("=" * 40)
    click.echo("Target hosts: %s" % ", ".join(host_list))
    click.echo()
    click.echo("The following will be executed as root:")
    for desc, cmd in _FE_UPDATE_STEPS:
        click.echo("  - %s  (%s)" % (desc, cmd))
    click.echo("  - Reboot")
    click.echo()

    if not dry_run:
        if not click.confirm("Proceed?", default=False):
            click.echo("Aborted.")
            return

    # --- Step 3: Get sudo access ---
    sudo_password, indirect_user = ensure_sudo_password(
        host_list,
        user,
        ssh_kwargs,
        dry_run=dry_run,
        allow_indirect=True,
        default_user=user,
    )
    sudo_ssh_kwargs = dict(ssh_kwargs)
    if indirect_user:
        sudo_ssh_kwargs["ssh_user"] = indirect_user

    # --- Step 4: Run update steps ---
    from sparkrun.orchestration.sudo import run_sudo_script_on_host

    failed_hosts: set[str] = set()
    for desc, cmd in _FE_UPDATE_STEPS:
        active_hosts = [h for h in host_list if h not in failed_hosts]
        if not active_hosts:
            click.echo("All hosts failed — aborting remaining steps.")
            break

        click.echo()
        click.echo("[%s]" % desc)
        for h in active_hosts:
            click.echo("  %-30s ..." % h, nl=False)
            r = run_sudo_script_on_host(
                h,
                cmd,
                password=sudo_password,
                ssh_kwargs=sudo_ssh_kwargs,
                timeout=600,
                dry_run=dry_run,
            )
            if r.success:
                click.echo(" OK")
                if r.stdout.strip():
                    # Show last few lines of output for visibility
                    for line in r.stdout.strip().splitlines()[-3:]:
                        click.echo("    %s" % line)
            else:
                click.echo(" FAILED")
                click.echo("    %s" % r.stderr.strip()[:200], err=True)
                failed_hosts.add(h)

    # --- Step 5: Reboot ---
    reboot_hosts = [h for h in host_list if h not in failed_hosts]
    if reboot_hosts:
        click.echo()
        click.echo("[Reboot]")
        if dry_run:
            click.echo("  [dry-run] Would reboot: %s" % ", ".join(reboot_hosts))
        else:
            if not click.confirm("Updates complete. Reboot %d host(s) now?" % len(reboot_hosts), default=True):
                click.echo("Skipping reboot. Remember to reboot manually for updates to take effect.")
            else:
                for h in reboot_hosts:
                    click.echo("  %-30s rebooting..." % h)
                    run_sudo_script_on_host(
                        h,
                        "nohup bash -c 'sleep 2 && reboot' &>/dev/null &",
                        password=sudo_password,
                        ssh_kwargs=sudo_ssh_kwargs,
                        timeout=10,
                        dry_run=dry_run,
                    )
                click.echo()
                click.echo("Reboot initiated on %d host(s)." % len(reboot_hosts))

    # --- Summary ---
    click.echo()
    ok = len([h for h in host_list if h not in failed_hosts])
    fail = len(failed_hosts)
    if fail:
        click.echo("Results: %d updated, %d failed (%s)" % (ok, fail, ", ".join(sorted(failed_hosts))))
        sys.exit(1)
    else:
        click.echo("Results: %d host(s) updated successfully." % ok)


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------


@setup.command("diagnose", hidden=True)
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

    from sparkrun.diagnostics import (
        NDJSONWriter,
        collect_config_diagnostics,
        collect_spark_diagnostics,
        collect_sudo_diagnostics,
    )

    from .._common import _get_context, _resolve_setup_context

    sctx = _get_context(ctx)
    config = sctx.config

    host_list, user, ssh_kwargs = _resolve_setup_context(hosts, hosts_file, cluster_name, config)

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

    with NDJSONWriter(output_file) as writer:
        try:
            from sparkrun import __version__
        except Exception:
            __version__ = "unknown"

        writer.emit(
            "diag_header",
            {
                "sparkrun_version": __version__,
                "hosts": host_list,
                "command": "sparkrun setup diagnose",
                "sudo": use_sudo,
            },
        )

        # Local config: clusters and registries
        cluster_mgr = _get_cluster_manager()
        registry_mgr = config.get_registry_manager()
        collect_config_diagnostics(
            writer,
            config=config,
            cluster_mgr=cluster_mgr,
            registry_mgr=registry_mgr,
        )

        host_data = collect_spark_diagnostics(
            hosts=host_list,
            ssh_kwargs=ssh_kwargs,
            writer=writer,
            dry_run=dry_run,
        )

        # Sudo pass: dmidecode
        if use_sudo and sudo_password:
            click.echo("Collecting sudo diagnostics (dmidecode)...")
            collect_sudo_diagnostics(
                hosts=host_list,
                ssh_kwargs=ssh_kwargs,
                sudo_password=sudo_password,
                writer=writer,
                dry_run=dry_run,
            )

    # Display summary
    ok = sum(1 for v in host_data.values() if v.get("DIAG_COMPLETE") == "1")
    fail = len(host_list) - ok

    click.echo()
    click.echo("Diagnostics complete: %d/%d hosts OK" % (ok, len(host_list)))
    if fail:
        click.echo("  %d host(s) failed — see %s for details" % (fail, output_file), err=True)

    if output_json:
        from .._common import print_json

        summary = {
            "output_file": output_file,
            "total_hosts": len(host_list),
            "successful": ok,
            "failed": fail,
            "hosts": {h: bool(d.get("DIAG_COMPLETE") == "1") for h, d in host_data.items()},
        }
        print_json(summary)

    if fail:
        sys.exit(1)
