"""Shared setup phase operations for sparkrun setup commands and wizard."""

from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Docker group membership scripts (inline — too short for separate .sh files)
# ---------------------------------------------------------------------------

_DOCKER_GROUP_SCRIPT = """\
#!/bin/bash
set -uo pipefail
TARGET_USER="{user}"
if id -nG "$TARGET_USER" 2>/dev/null | grep -qw docker; then
    echo "DOCKER_GROUP=already_member"
else
    sudo -n usermod -aG docker "$TARGET_USER" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "DOCKER_GROUP=added"
    else
        echo "DOCKER_GROUP=needs_sudo"
        exit 1
    fi
fi
"""

_DOCKER_GROUP_FALLBACK_SCRIPT = """\
#!/bin/bash
set -uo pipefail
TARGET_USER="{user}"
usermod -aG docker "$TARGET_USER"
echo "DOCKER_GROUP=added"
"""


def _docker_group_summary(stdout: str, user: str | None = None) -> str:
    """Extract status from docker-group script output."""
    label = "'%s' " % user if user else ""
    for line in stdout.strip().splitlines():
        if line.startswith("DOCKER_GROUP="):
            val = line.split("=", 1)[1]
            if val == "already_member":
                return "%salready a member" % label
            elif val == "added":
                return "added %sto docker group" % label
    return stdout.strip()[:80]


# ---------------------------------------------------------------------------
# Earlyoom process patterns — used to build --prefer / --avoid arguments
# ---------------------------------------------------------------------------

# Processes that should be killed first on OOM (inference workloads).
EARLYOOM_PREFER_PATTERNS = [
    "vllm",
    "VLLM",
    "sglang",
    "llama-server",
    "llama-cli",
    "trtllm",
    "tritonserver",
    "ray",
    "python3",
    "python",
]

# Processes to protect from OOM kill (system services).
EARLYOOM_AVOID_PATTERNS = [
    "systemd",
    "sshd",
    "dockerd",
    "containerd",
    "dbus-daemon",
    "NetworkManager",
]


def _earlyoom_summary(stdout: str) -> str:
    """Extract key status lines from earlyoom install output.

    Filters out noisy apt-get progress (Reading database ...) and
    returns only the meaningful status lines (INSTALLED/PRESENT/CONFIGURED/OK/ERROR).
    """
    keywords = ("INSTALLING:", "INSTALLED:", "PRESENT:", "CONFIGURED:", "OK:", "ERROR:")
    lines = [line.strip() for line in stdout.strip().splitlines() if any(line.strip().startswith(kw) for kw in keywords)]
    return "; ".join(lines) if lines else stdout.strip()[:100]


def _build_earlyoom_regex(patterns: list[str]) -> str:
    """Build a regex pattern string for earlyoom --prefer/--avoid.

    earlyoom uses POSIX extended regex matching against ``/proc/pid/comm``.
    Wraps patterns in ``(...)`` so matching is unanchored and can match
    anywhere in the process name.
    """
    return "(%s)" % "|".join(patterns)


# ---------------------------------------------------------------------------
# Shared apply functions
# ---------------------------------------------------------------------------


def apply_docker_group(host_list, user, ssh_kwargs, dry_run, sudo_password=None, sudo_dispatch_fn=None):
    """Apply docker group membership on hosts.

    When *sudo_dispatch_fn* is provided (wizard indirect sudo case), it is
    called as ``sudo_dispatch_fn(host, script, password, timeout=...)``
    instead of :func:`run_with_sudo_fallback`.

    Returns:
        Tuple of ``(result_map, ok_count)`` where *result_map* maps
        host → RemoteResult.
    """
    from sparkrun.orchestration.sudo import run_with_sudo_fallback

    script = _DOCKER_GROUP_SCRIPT.format(user=user)
    fallback = _DOCKER_GROUP_FALLBACK_SCRIPT.format(user=user)

    if dry_run:
        click.echo("  [dry-run] Would ensure docker group on %d host(s)." % len(host_list))
        return {}, 0

    if sudo_dispatch_fn:
        # Indirect sudo path: run fallback script per-host via dispatch function
        result_map = {}
        for h in host_list:
            r = sudo_dispatch_fn(h, fallback, sudo_password, timeout=30)
            result_map[h] = r
    else:
        result_map, still_failed = run_with_sudo_fallback(
            host_list,
            script,
            fallback,
            ssh_kwargs,
            dry_run=dry_run,
        )
        if still_failed and sudo_password:
            for h in still_failed:
                r = (
                    sudo_dispatch_fn(h, fallback, sudo_password, timeout=30)
                    if sudo_dispatch_fn
                    else _sudo_script_on_host(h, fallback, sudo_password, ssh_kwargs, timeout=30, dry_run=dry_run)
                )
                result_map[h] = r

    ok_count = sum(1 for h in host_list if result_map.get(h) and result_map[h].success)
    return result_map, ok_count


def apply_sudoers(host_list, user, dry_run, sudo_password=None, sudo_dispatch_fn=None):
    """Apply scoped sudoers entries (fix-permissions + clear-cache) on hosts.

    Returns:
        Tuple of ``(ok_per_label, failed_any)`` where *ok_per_label* is a
        list of ``(label, ok_count)`` tuples.
    """
    from sparkrun.scripts import read_script
    from sparkrun.utils.shell import validate_unix_username

    validate_unix_username(user)
    sudoers_scripts = [
        (
            "fix-permissions",
            read_script("fix_permissions_sudoers.sh").format(
                user=user,
                cache_dir="",
            ),
        ),
        (
            "clear-cache",
            read_script("clear_cache_sudoers.sh").format(
                user=user,
            ),
        ),
    ]

    if dry_run:
        for label, _ in sudoers_scripts:
            click.echo("  [dry-run] Would install %s sudoers on %d host(s)." % (label, len(host_list)))
        return [("fix-permissions", 0), ("clear-cache", 0)], False

    failed_any = False
    ok_per_label = []
    for label, script in sudoers_scripts:
        label_ok = 0
        for h in host_list:
            r = sudo_dispatch_fn(h, script, sudo_password or "", timeout=300)
            if r.success:
                label_ok += 1
            else:
                logger.debug(
                    "Sudoers %s failed on %s: %s",
                    label,
                    h,
                    r.stderr[:100],
                )
        if label_ok < len(host_list):
            failed_any = True
        click.echo("  %s: %d/%d host(s)" % (label, label_ok, len(host_list)))
        ok_per_label.append((label, label_ok))
    return ok_per_label, failed_any


def apply_earlyoom(host_list, ssh_kwargs, prefer_regex, avoid_regex, dry_run, sudo_password=None, sudo_dispatch_fn=None):
    """Apply earlyoom installation on hosts.

    When *sudo_dispatch_fn* is provided (wizard indirect sudo case), it is
    called per-host instead of :func:`run_with_sudo_fallback`.

    Returns:
        Tuple of ``(result_map, ok_count, installed_pkg)`` where
        *installed_pkg* indicates whether any host freshly installed
        the earlyoom package.
    """
    from sparkrun.orchestration.sudo import run_with_sudo_fallback
    from sparkrun.scripts import read_script

    install_script = read_script("earlyoom_install.sh").format(
        prefer=prefer_regex,
        avoid=avoid_regex,
    )
    fallback_script = read_script("earlyoom_install_fallback.sh").format(
        prefer=prefer_regex,
        avoid=avoid_regex,
    )

    if sudo_dispatch_fn:
        result_map = {}
        still_failed = []
        for h in host_list:
            r = sudo_dispatch_fn(h, fallback_script, sudo_password, timeout=300)
            result_map[h] = r
            if not r.success:
                still_failed.append(h)
    else:
        result_map, still_failed = run_with_sudo_fallback(
            host_list,
            install_script,
            fallback_script,
            ssh_kwargs,
            dry_run=dry_run,
            sudo_password=sudo_password,
        )
        if still_failed and sudo_password and not dry_run:
            for h in still_failed:
                r = _sudo_script_on_host(h, fallback_script, sudo_password, ssh_kwargs, timeout=300, dry_run=dry_run)
                result_map[h] = r

    ok_count = sum(1 for h in host_list if result_map.get(h) and result_map[h].success)
    installed_pkg = any("INSTALLING:" in (result_map.get(h) and result_map[h].stdout or "") for h in host_list)
    return result_map, ok_count, installed_pkg


def _sudo_script_on_host(host, script, password, ssh_kwargs, timeout=300, dry_run=False):
    """Helper: run sudo script on a single host (used when no dispatch_fn)."""
    from sparkrun.orchestration.sudo import run_sudo_script_on_host

    return run_sudo_script_on_host(host, script, password, ssh_kwargs=ssh_kwargs, timeout=timeout, dry_run=dry_run)
