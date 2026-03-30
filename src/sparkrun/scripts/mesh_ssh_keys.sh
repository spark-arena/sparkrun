#!/usr/bin/env bash
set -euo pipefail

# Mesh passwordless SSH among N hosts for the same username.
# Creates per-host ed25519 key if missing, then appends each host's pubkey
# into all other hosts' authorized_keys (deduped).
#
# Usage:
#   ./mesh-ssh-keys.sh <username> <host1> <host2> <host3> <host4> [more hosts...]
#
# Example:
#   ./mesh-ssh-keys.sh ubuntu node1 node2 node3 node4
#
# Notes:
# - You will be prompted for passwords as needed (no sshpass/expect required).
# TODO: add support for giving key file reference instead of passwords.
# - Requires: bash, ssh, ssh-keygen (standard on Ubuntu w/ OpenSSH client)
# - Assumes the same username exists on all hosts.

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <username> <host1> <host2> <host3> [more hosts...]"
  exit 1
fi

USER_NAME="$1"
shift
HOSTS=("$@")

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$HOME/.ssh/known_hosts"
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
)

# SSH connection multiplexing to avoid repeated password prompts per host
CONTROL_DIR="${TMPDIR:-/tmp}/mesh-ssh-keys-$USER_NAME"
mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"

control_path_for() {
  local host="$1"
  # ControlPath must be short; %h and %p expansions help.
  echo "$CONTROL_DIR/cm-%r@%h:%p"
}

ssh_cmd() {
  local host="$1"
  shift
  local cp
  cp="$(control_path_for "$host")"

  ssh "${SSH_OPTS[@]}" \
    -o ControlMaster=auto \
    -o ControlPersist=10m \
    -o ControlPath="$cp" \
    "$USER_NAME@$host" "$@"
}

ssh_stdin_cmd() {
  local host="$1"
  shift
  local cp
  cp="$(control_path_for "$host")"

  ssh "${SSH_OPTS[@]}" \
    -o ControlMaster=auto \
    -o ControlPersist=10m \
    -o ControlPath="$cp" \
    "$USER_NAME@$host" "$@"
}

echo "=== Phase 1: Connectivity check ==="
for h in "${HOSTS[@]}"; do
  echo "[*] Checking SSH connectivity to $USER_NAME@$h ..."
  ssh_cmd "$h" "true"
done
echo

# If the caller (local OS user) differs from the mesh user, install the
# caller's public key on every host so the control machine can later SSH
# as USER_NAME without password prompts (BatchMode=yes).
CALLER="$(whoami)"
if [[ "$CALLER" != "$USER_NAME" ]]; then
  LOCAL_PUBKEY=""
  for kf in "$HOME/.ssh/id_ed25519.pub" "$HOME/.ssh/id_rsa.pub" "$HOME/.ssh/id_ecdsa.pub"; do
    if [[ -f "$kf" ]]; then
      LOCAL_PUBKEY="$(cat "$kf")"
      break
    fi
  done
  if [[ -z "$LOCAL_PUBKEY" ]]; then
    echo "[*] No SSH key found for local user '$CALLER'. Generating ed25519 key..."
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    ssh-keygen -t ed25519 -N '' -f "$HOME/.ssh/id_ed25519" >/dev/null
    LOCAL_PUBKEY="$(cat "$HOME/.ssh/id_ed25519.pub")"
  fi
  echo "=== Installing control machine key for remote access ==="
  echo "Caller '$CALLER' differs from mesh user '$USER_NAME'."
  echo "Installing caller's public key so the control machine can SSH as '$USER_NAME'..."
  for h in "${HOSTS[@]}"; do
    echo "[*] Installing caller key on $h ..."
    printf '%s\n' "$LOCAL_PUBKEY" | ssh_stdin_cmd "$h" "set -eu
      umask 077
      chmod go-w ~ 2>/dev/null || true
      mkdir -p ~/.ssh
      chmod 700 ~/.ssh
      touch ~/.ssh/authorized_keys
      chmod 600 ~/.ssh/authorized_keys

      read -r incoming_key
      grep -qxF \"\$incoming_key\" ~/.ssh/authorized_keys || echo \"\$incoming_key\" >> ~/.ssh/authorized_keys
    "
  done
  echo

  # Verify cross-user pubkey auth works without ControlMaster
  echo "=== Verifying cross-user SSH authentication ==="
  _verify_failures=0
  for h in "${HOSTS[@]}"; do
    if ssh "${SSH_OPTS[@]}" \
         -o ControlPath=none \
         -o BatchMode=yes \
         -o ConnectTimeout=5 \
         "$USER_NAME@$h" "true" 2>/dev/null; then
      echo "[+] Pubkey auth OK: $USER_NAME@$h"
    else
      echo "[!] Pubkey auth FAILED: $USER_NAME@$h"
      _verify_failures=$((_verify_failures + 1))
      # Collect diagnostics over the still-active ControlMaster
      echo "    Collecting diagnostics via existing connection..."
      _diag="$(ssh_cmd "$h" "set -eu
        echo \"home_perms=\$(stat -c '%a' ~ 2>/dev/null || echo unknown)\"
        echo \"ssh_perms=\$(stat -c '%a' ~/.ssh 2>/dev/null || echo unknown)\"
        echo \"ak_perms=\$(stat -c '%a' ~/.ssh/authorized_keys 2>/dev/null || echo unknown)\"
        echo \"ak_lines=\$(wc -l < ~/.ssh/authorized_keys 2>/dev/null || echo 0)\"
        echo \"ak_file=\$(grep -i '^AuthorizedKeysFile' /etc/ssh/sshd_config 2>/dev/null || echo 'default (~/.ssh/authorized_keys)')\"
        echo \"strict_modes=\$(grep -i '^StrictModes' /etc/ssh/sshd_config 2>/dev/null || echo 'StrictModes yes (default)')\"
      " 2>/dev/null || echo "(diagnostics unavailable)")"
      echo "    $USER_NAME@$h diagnostics:"
      while IFS= read -r line; do
        echo "      $line"
      done <<< "$_diag"
      echo
      echo "    Common causes:"
      echo "      1. Home directory (~$USER_NAME) is group/world-writable (sshd StrictModes rejects this)"
      echo "         Fix: ssh $USER_NAME@$h 'chmod go-w ~'"
      echo "      2. sshd_config AuthorizedKeysFile points elsewhere (e.g. /etc/ssh/authorized_keys/%u)"
      echo "      3. AllowUsers/AllowGroups in sshd_config restricts $USER_NAME"
      echo
    fi
  done
  if [[ $_verify_failures -gt 0 ]]; then
    echo "[!] WARNING: Pubkey authentication failed for $_verify_failures host(s)."
    echo "    The inter-node mesh will still be set up, but control-machine SSH"
    echo "    to these hosts may require a password. See diagnostics above."
    echo "    Run: sparkrun setup ssh --cluster <name> --diagnose"
    echo
  fi
fi

echo "=== Phase 2: Ensure SSH key exists on each host ==="
for h in "${HOSTS[@]}"; do
  echo "[*] Ensuring ~/.ssh and id_ed25519 exist on $h ..."
  ssh_cmd "$h" "set -eu
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    if [ ! -f ~/.ssh/id_ed25519 ]; then
      ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519 >/dev/null
    fi
    chmod 600 ~/.ssh/id_ed25519
    chmod 644 ~/.ssh/id_ed25519.pub
  "
done
echo

echo "=== Phase 3: Fetch each host's public key ==="
declare -A PUBKEY
for h in "${HOSTS[@]}"; do
  echo "[*] Reading public key from $h ..."
  PUBKEY["$h"]="$(ssh_cmd "$h" "cat ~/.ssh/id_ed25519.pub")"
  if [[ -z "${PUBKEY[$h]}" ]]; then
    echo "[!] Failed to read pubkey from $h"
    exit 1
  fi
done
echo

echo "=== Phase 4: Install keys so every host trusts every other host ==="
for src in "${HOSTS[@]}"; do
  key="${PUBKEY[$src]}"
  echo "[*] Installing key from $src onto all other hosts ..."
  for dst in "${HOSTS[@]}"; do
    if [[ "$src" == "$dst" ]]; then
      continue
    fi

    echo "    - $src -> $dst"
    # Send the key over stdin and append only if not already present.
    printf '%s\n' "$key" | ssh_stdin_cmd "$dst" "set -eu
      umask 077
      chmod go-w ~ 2>/dev/null || true
      mkdir -p ~/.ssh
      chmod 700 ~/.ssh
      touch ~/.ssh/authorized_keys
      chmod 600 ~/.ssh/authorized_keys

      read -r incoming_key
      # Deduplicate: add only if exact line not present.
      grep -qxF \"\$incoming_key\" ~/.ssh/authorized_keys || echo \"\$incoming_key\" >> ~/.ssh/authorized_keys
    "
  done
done
echo

echo "=== Done ==="
echo "All hosts should now be able to SSH to each other as '$USER_NAME' without passwords."
echo
echo "Quick test examples (run from any host):"
echo "  ssh $USER_NAME@${HOSTS[0]}"
echo "  ssh $USER_NAME@${HOSTS[1]}"

