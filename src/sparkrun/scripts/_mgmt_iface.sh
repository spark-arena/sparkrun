# shellcheck shell=bash
# Shared management-interface detection for sparkrun's remote probe scripts.
#
# Not executable on its own: included into the other scripts through the
# "# sparkrun:include" directive resolved by sparkrun.scripts.read_script.
#
# WHY THIS EXISTS
# ---------------
# Seven probes each spelled the same one-liner:
#
#     DEFAULT_IF=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' || echo "eth0")
#
# On a host with no default route -- an intentionally air-gapped DGX Spark --
# the lookup fails and the fallback names an interface that does not exist on
# this hardware.  That name flows into GLOO_SOCKET_IFNAME / TP_SOCKET_IFNAME /
# MN_IF_NAME and the head of NCCL_SOCKET_IFNAME, and kills the workload at init
# with "Unable to find address for: eth0" (issue #275).
#
# A guessed name is strictly worse than no answer.  Every caller already
# handles "could not tell" and degrades correctly; none can recover from a
# confident lie.  So this helper prints a name only after confirming it exists,
# and otherwise prints nothing.
#
# STYLE CONSTRAINTS -- both load-bearing, do not "tidy up":
#
#   * No curly braces anywhere -- not even in these comments.  That rules out
#     the braced parameter form (write "$VAR", never the dollar-brace one),
#     awk programs, and the braced function-body syntax, which is why the
#     function below is defined with a ( subshell ) body instead.  Several of
#     the including scripts are passed through Python's str.format(), which
#     raises KeyError on any brace appearing here.
#   * The subshell body is also what isolates "set --" and the temporaries, so
#     the helper cannot disturb the including script's state.

# Operator override: SPARKRUN_MGMT_IFACE, injected at the top of the including
# script by sparkrun.scripts.inject_shell_vars when the cluster pins
# `mgmt_interface`.  Deliberately NOT defaulted to "" here -- this file is
# included partway down the script, so a default would clobber the injected
# value.  The reader below runs under "set +u", which is what makes the
# unset case read as empty.
#
# Usage: IFACE=$(sparkrun_mgmt_iface "<comma-separated interfaces to exclude>")
#
# Prints the resolved management interface, or nothing when none can be
# established.  The exclude list is optional and may be empty; it is typically
# the detected fabric adapters, which are not management interfaces.
sparkrun_mgmt_iface() (
    # SSH_CONNECTION is legitimately unset on the local-dispatch path and the
    # argument is optional, so -u (set by every including script) must go.
    set +u
    set -o pipefail

    _excl="$1"

    # Interface sysfs root.  Overridable only so the resolution chain can be
    # exercised against a fixture tree in the test suite; nothing in
    # production sets it.
    _netdir="$SPARKRUN_NET_SYSFS"
    if [ -z "$_netdir" ]; then
        _netdir=/sys/class/net
    fi

    # (0) Operator override.  A pinned interface outranks anything we could
    #     detect, but still may not name a device that isn't there.
    if [ -n "$SPARKRUN_MGMT_IFACE" ]; then
        if [ -e "$_netdir/$SPARKRUN_MGMT_IFACE" ]; then
            printf '%s\n' "$SPARKRUN_MGMT_IFACE"
            exit 0
        fi
        echo "sparkrun: pinned mgmt_interface '$SPARKRUN_MGMT_IFACE' is not present on this host; detecting instead" >&2
    fi

    # (1) The default route.  Identical to the historical behaviour, so every
    #     internet-connected cluster resolves exactly as it always has.
    _cand=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' | head -1)
    if [ -n "$_cand" ] && [ -e "$_netdir/$_cand" ]; then
        printf '%s\n' "$_cand"
        exit 0
    fi

    # (2) The local end of our own SSH connection.  SSH_CONNECTION is
    #     "<client ip> <client port> <server ip> <server port>", so field 3 is
    #     the address on THIS host that the control machine reached -- the
    #     management address by construction.  sparkrun always arrives over
    #     SSH, which is what makes this reliable with no default route.
    #     An IPv6 session finds no IPv4 match and falls through.
    _ssh_ip=""
    if [ -n "$SSH_CONNECTION" ]; then
        set -- $SSH_CONNECTION
        if [ "$#" -ge 3 ]; then
            _ssh_ip="$3"
        fi
    fi
    if [ -n "$_ssh_ip" ]; then
        for _path in "$_netdir"/*; do
            [ -e "$_path" ] || continue
            _cand=$(basename "$_path")
            if ip -4 -o addr show dev "$_cand" 2>/dev/null | grep -qF "inet $_ssh_ip/"; then
                printf '%s\n' "$_cand"
                exit 0
            fi
        done
    fi

    # (3) First real NIC that is up and carries a global IPv4.
    #
    #     The <sysfs>/<if>/device test is what separates physical NICs
    #     from docker0, br-*, veth*, tailscale0 and lo -- several of which hold
    #     a global IPv4 on a normal Spark and would otherwise look plausible.
    #
    #     RDMA-backed NICs (device/infiniband present) are never selected: the
    #     CX7 fabric is not the management network, and picking one here would
    #     silently pin control traffic to the fabric.  Callers that want the
    #     fabric ask for it deliberately -- see pin_comm_env_to_ib -- and the
    #     empty answer is what routes them there.
    for _path in "$_netdir"/*; do
        [ -e "$_path/device" ] || continue
        if [ -e "$_path/device/infiniband" ]; then
            continue
        fi
        _cand=$(basename "$_path")
        case ",$_excl," in
            *",$_cand,"*) continue ;;
        esac
        if [ "$(cat "$_path/operstate" 2>/dev/null)" != "up" ]; then
            continue
        fi
        if ip -4 -o addr show dev "$_cand" scope global 2>/dev/null | grep -q 'inet '; then
            printf '%s\n' "$_cand"
            exit 0
        fi
    done

    # (4) Nothing established.  Print nothing.
    exit 0
)
