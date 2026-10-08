# SSH mesh reachability (non-destructive). Input: PEERS, space-separated.
# Each peer is dialed with BatchMode, the host's own known_hosts and the
# default host-key policy -- the way head-to-worker distribution connects
# (build_ssh_opts_string) -- so a peer whose key was never recorded fails here
# exactly as it would fail a transfer, and is counted in MESH_HOSTKEY.
# BatchMode never adds a known_hosts entry.
# NOTE: `ssh -n` (stdin from /dev/null) is REQUIRED. Scripts reach the host via
# `ssh <host> bash -s`, so the script body IS the remote bash's stdin; without
# -n the inner ssh would slurp the rest of it and truncate the probe.
# NOTE brace-free, comments included: setup_check.sh includes this and is
# rendered with str.format().
MESH_TOTAL=0
MESH_OK=0
MESH_HOSTKEY=0
for peer in $PEERS; do
    MESH_TOTAL=$((MESH_TOTAL + 1))
    if _MESH_ERR="$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$peer" true 2>&1 >/dev/null)"; then
        MESH_OK=$((MESH_OK + 1))
    else
        case "$_MESH_ERR" in
            *"Host key verification failed"*) MESH_HOSTKEY=$((MESH_HOSTKEY + 1)) ;;
        esac
    fi
done
echo "CHECK_MESH_TOTAL=$MESH_TOTAL"
echo "CHECK_MESH_OK=$MESH_OK"
echo "CHECK_MESH_HOSTKEY=$MESH_HOSTKEY"
