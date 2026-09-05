#!/bin/bash
# Non-destructive setup-readiness probe for `sparkrun setup check`.
# Emits key=value pairs on stdout; NEVER modifies host state. Diagnostic
# noise goes to stderr. Mirrors the parse style of spark_diagnose.sh.
#
# Params: {peers}  — space-separated peer hosts for the SSH-mesh probe.
set -uo pipefail

WHO=$(id -un 2>/dev/null || echo unknown)
echo "CHECK_USER=$WHO"
echo "CHECK_UID=$(id -u 2>/dev/null || echo unknown)"

# --- systemd-logind IPC reaping ---
# `RemoveIPC=yes` (the Ubuntu 24.04 / DGX OS default) makes logind delete every
# POSIX semaphore, shared-memory segment and message queue owned by a regular
# UID once that user's last session ends. That reaps a detached workload's own
# IPC whenever it shares the host's namespace -- an `ipc: host` container run
# as the SSH user, or the `local` executor, which has no namespace at all.
# Enabled lingering keeps the user manager alive and suppresses the reap.
#
# Read the *effective* value from logind over D-Bus first (config lives across
# logind.conf + logind.conf.d and defaults to yes when unset, so parsing one
# file answers a different question); fall back to the config, then to
# "unknown". Every probe here is read-only and works unprivileged.
_REMOVE_IPC=unknown
if command -v busctl >/dev/null 2>&1; then
    # NOTE no awk/sed braces anywhere in this script -- it is delivered through
    # str.format() (for {peers}), so a literal brace raises KeyError at render.
    _RI=$(busctl get-property org.freedesktop.login1 /org/freedesktop/login1 \
              org.freedesktop.login1.Manager RemoveIPC 2>/dev/null | cut -d' ' -f2 | tr -d '[:space:]')
    case "$_RI" in
        true) _REMOVE_IPC=yes ;;
        false) _REMOVE_IPC=no ;;
    esac
fi
if [ "$_REMOVE_IPC" = unknown ] && [ -d /run/systemd/system ]; then
    # Last uncommented RemoveIPC= across the drop-in set wins for our purposes;
    # absent means the built-in default, which is yes.
    _RI=$(grep -hiE '^[[:space:]]*RemoveIPC[[:space:]]*=' \
              /etc/systemd/logind.conf /etc/systemd/logind.conf.d/*.conf \
              /run/systemd/logind.conf.d/*.conf /usr/lib/systemd/logind.conf.d/*.conf 2>/dev/null \
              | tail -n 1 | cut -d= -f2 | tr -d '[:space:]' | tr 'A-Z' 'a-z')
    case "$_RI" in
        no|false|0|off) _REMOVE_IPC=no ;;
        yes|true|1|on) _REMOVE_IPC=yes ;;
        *) _REMOVE_IPC=yes ;;
    esac
fi
echo "CHECK_LOGIND_REMOVE_IPC=$_REMOVE_IPC"

_LINGER=unknown
if command -v loginctl >/dev/null 2>&1; then
    _LG=$(loginctl show-user "$WHO" --property=Linger --value 2>/dev/null)
    case "$_LG" in
        yes) _LINGER=1 ;;
        no) _LINGER=0 ;;
    esac
fi
if [ "$_LINGER" = unknown ] && [ -d /var/lib/systemd/linger ]; then
    # The marker directory is world-readable, so this works without a session.
    if [ -e "/var/lib/systemd/linger/$WHO" ]; then _LINGER=1; else _LINGER=0; fi
fi
echo "CHECK_LOGIND_LINGER=$_LINGER"

# --- Docker ---
if command -v docker >/dev/null 2>&1; then
    echo "CHECK_DOCKER_INSTALLED=1"
    if docker info >/dev/null 2>&1; then
        echo "CHECK_DOCKER_USABLE=1"
    else
        echo "CHECK_DOCKER_USABLE=0"
    fi
else
    echo "CHECK_DOCKER_INSTALLED=0"
    echo "CHECK_DOCKER_USABLE=0"
fi

if id -nG "$WHO" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
    echo "CHECK_DOCKER_GROUP=1"
else
    echo "CHECK_DOCKER_GROUP=0"
fi

# --- NVIDIA GPU / Container Toolkit / CDI ---
command -v nvidia-smi >/dev/null 2>&1 && echo "CHECK_GPU_PRESENT=1" || echo "CHECK_GPU_PRESENT=0"
command -v nvidia-ctk >/dev/null 2>&1 && echo "CHECK_NVIDIA_CTK=1" || echo "CHECK_NVIDIA_CTK=0"
if [ -s /etc/cdi/nvidia.yaml ]; then
    echo "CHECK_CDI_SPEC=1"
    # Staleness. A CDI spec pins absolute host paths -- versioned driver
    # libraries (libnvidia-ml.so.<driver>) and device nodes -- captured when it
    # was generated. A driver upgrade replaces those files, leaving a spec that
    # resolves to nothing; containers then fail to start with a CDI error that
    # says nothing about the driver having moved. Counting how many referenced
    # paths still exist detects that directly, and catches any other cause of a
    # spec drifting from the host, rather than inferring it from a version
    # string. Bounded to keep the probe cheap on a spec with many devices.
    if [ -r /etc/cdi/nvidia.yaml ]; then
        _cdi_checked=0
        _cdi_missing=0
        for _cdi_path in $(grep -oE '(hostPath|path): *"?/[^ "]+' /etc/cdi/nvidia.yaml 2>/dev/null \
                | sed 's/.*: *"\?//' | sort -u | head -60); do
            _cdi_checked=$((_cdi_checked + 1))
            [ -e "$_cdi_path" ] || _cdi_missing=$((_cdi_missing + 1))
        done
        echo "CHECK_CDI_PATHS_CHECKED=$_cdi_checked"
        echo "CHECK_CDI_PATHS_MISSING=$_cdi_missing"
    fi
else
    echo "CHECK_CDI_SPEC=0"
fi

# --- earlyoom ---
command -v earlyoom >/dev/null 2>&1 && echo "CHECK_EARLYOOM_INSTALLED=1" || echo "CHECK_EARLYOOM_INSTALLED=0"
if systemctl is-active --quiet earlyoom 2>/dev/null; then
    echo "CHECK_EARLYOOM_ACTIVE=1"
else
    echo "CHECK_EARLYOOM_ACTIVE=0"
fi

# --- Sudoers entries (best-effort) ---
# Only inspect when passwordless sudo is available so the probe never blocks
# on a password prompt; otherwise report "unknown".
if sudo -n true 2>/dev/null; then
    if sudo -n test -e "/etc/sudoers.d/sparkrun-chown-$WHO" 2>/dev/null; then
        echo "CHECK_SUDOERS_CHOWN=1"
    else
        echo "CHECK_SUDOERS_CHOWN=0"
    fi
    if sudo -n test -e "/etc/sudoers.d/sparkrun-dropcaches-$WHO" 2>/dev/null; then
        echo "CHECK_SUDOERS_DROPCACHES=1"
    else
        echo "CHECK_SUDOERS_DROPCACHES=0"
    fi
else
    echo "CHECK_SUDOERS_CHOWN=unknown"
    echo "CHECK_SUDOERS_DROPCACHES=unknown"
fi

# --- SSH mesh (non-destructive) ---
# Attempt a BatchMode SSH to each peer; write no known_hosts entries.
# NOTE: `ssh -n` (stdin from /dev/null) is REQUIRED here. This whole script
# is delivered to the host via `ssh <host> bash -s`, so the script body IS the
# remote bash's stdin. Without -n the inner ssh would slurp the rest of that
# stdin (the remainder of this script), truncating execution so CHECK_COMPLETE
# never prints and the host is falsely reported unreachable.
PEERS="{peers}"
MESH_TOTAL=0
MESH_OK=0
for peer in $PEERS; do
    MESH_TOTAL=$((MESH_TOTAL + 1))
    if ssh -n -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 "$peer" true 2>/dev/null; then
        MESH_OK=$((MESH_OK + 1))
    fi
done
echo "CHECK_MESH_TOTAL=$MESH_TOTAL"
echo "CHECK_MESH_OK=$MESH_OK"

echo "CHECK_COMPLETE=1"
