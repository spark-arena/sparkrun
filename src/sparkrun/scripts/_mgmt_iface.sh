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
# The management network is the one the cluster's hosts reach each other on,
# so that is what is measured first.  The default route says where *internet*
# traffic goes, which is a different question: a host whose wired NIC reaches
# its peers but has no gateway, and whose WiFi holds the only default route,
# used to resolve to the WiFi and rendezvous over it.  The default route is
# now only a fallback, for when there are no peers to measure against.
#
# STYLE CONSTRAINTS -- both load-bearing, do not "tidy up":
#
#   * No curly braces anywhere -- not even in these comments.  That rules out
#     the braced parameter form (write "$VAR", never the dollar-brace one),
#     arrays (whose expansion needs braces), awk programs, and the braced
#     function-body syntax, which is why the functions below are defined with
#     a ( subshell ) body instead.  Several of the including scripts are
#     passed through Python's str.format(), which raises KeyError on any brace
#     appearing here.
#   * The subshell body is also what isolates "set --" and the temporaries, so
#     the helper cannot disturb the including script's state.


# Inputs, injected at the top of the including script by
# sparkrun.scripts.inject_mgmt_vars.  Deliberately NOT defaulted here -- this
# file is included partway down the script, so a default would clobber the
# injected value.  The readers run under "set +u", which is what makes the
# unset case read as empty.
#
#   SPARKRUN_MGMT_IFACE  the cluster's pinned `mgmt_interface` (operator override)
#   SPARKRUN_MGMT_PEERS  comma-separated addresses of the cluster's hosts, as
#                        sparkrun reaches them; this host's own entry is skipped
#
# Entry points:
#
#   IFACE=$(sparkrun_mgmt_iface "<excludes>")     the management interface
#   read -r IFACE RULE <<< "$(sparkrun_mgmt_resolve "<excludes>")"
#                                                 ... and the rule that chose it
#   IP=$(sparkrun_mgmt_ip "$IFACE")               its address facing the peers
#   KIND=$(sparkrun_mgmt_kind "$IFACE")           wired / wireless / virtual / fabric
#   sparkrun_mgmt_report "<excludes>"             all of the above as key=value lines
#
# Each prints nothing when it cannot establish an answer.  <excludes> is an
# optional comma-separated interface list, typically the detected fabric
# adapters; it applies only to the last-resort scan: every earlier rule is
# evidence about this host, which outranks a guess about which adapters are
# fabric.
#
# RULE is one of: pinned, peers, ssh, default-route, ssh-virtual, scan.

sparkrun_mgmt_iface() (
    set +u
    _resolved=$(sparkrun_mgmt_resolve "$1")
    printf '%s\n' "$_resolved" | cut -d' ' -f1 | grep -v '^$'
    exit 0
)

sparkrun_mgmt_resolve() (
    # SSH_CONNECTION is legitimately unset on the local-dispatch path and the
    # argument is optional, so -u (set by every including script) must go.
    set +u
    set -o pipefail

    _excl="$1"
    _netdir=$(_sparkrun_mgmt_netdir)

    # (0) Operator override.  A pinned interface outranks anything we could
    #     detect, but still may not name a device that isn't there.
    if [ -n "$SPARKRUN_MGMT_IFACE" ]; then
        if [ -e "$_netdir/$SPARKRUN_MGMT_IFACE" ]; then
            echo "$SPARKRUN_MGMT_IFACE pinned"
            exit 0
        fi
        echo "sparkrun: pinned mgmt_interface '$SPARKRUN_MGMT_IFACE' is not present on this host; detecting instead" >&2
    fi

    _ssh_if=$(_sparkrun_mgmt_ssh_iface)

    # (1) Peer coverage: the interface that reaches the most of the cluster's
    #     other hosts, and reaches them most directly.
    _best=$(_sparkrun_mgmt_by_peers "$_netdir" "$_ssh_if")
    if [ -n "$_best" ]; then
        echo "$_best peers"
        exit 0
    fi

    # (2) The SSH arrival interface, when it is a real NIC (one with a sysfs
    #     device).  Physical-only so that a control machine connecting over a
    #     VPN does not move the cluster's traffic onto tailscale0 / wg0.
    if [ -n "$_ssh_if" ] && [ -e "$_netdir/$_ssh_if/device" ]; then
        echo "$_ssh_if ssh"
        exit 0
    fi

    # (3) The default route -- a fallback, not evidence of the management
    #     network (an offline or air-gapped cluster may have none at all).
    _cand=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' | head -1)
    if [ -n "$_cand" ] && [ -e "$_netdir/$_cand" ]; then
        echo "$_cand default-route"
        exit 0
    fi

    # (4) The SSH arrival interface even when virtual: management may really
    #     run over a VPN, and nothing above contradicted it.
    if [ -n "$_ssh_if" ]; then
        echo "$_ssh_if ssh-virtual"
        exit 0
    fi

    # (5) First real NIC that is up and carries a global IPv4.
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
    #
    #     Wired NICs are tried before wireless ones.
    for _pass in wired wireless; do
        for _path in "$_netdir"/*; do
            [ -e "$_path/device" ] || continue
            if [ -e "$_path/device/infiniband" ]; then
                continue
            fi
            if _sparkrun_mgmt_is_wireless "$_path"; then
                [ "$_pass" = wireless ] || continue
            else
                [ "$_pass" = wired ] || continue
            fi
            _cand=$(basename "$_path")
            case ",$_excl," in
                *",$_cand,"*) continue ;;
            esac
            if [ "$(cat "$_path/operstate" 2>/dev/null)" != "up" ]; then
                continue
            fi
            if ip -4 -o addr show dev "$_cand" scope global 2>/dev/null | grep -q 'inet '; then
                echo "$_cand scan"
                exit 0
            fi
        done
    done

    # (6) Nothing established.  Print nothing.
    exit 0
)

# The probes' report: the resolved management interface with its address, the
# rule that chose it and its kind, as DETECTED_MGMT_IFACE / _IP / _RULE / _KIND
# lines (empty values when nothing resolved).  sparkrun reads these to show
# and warn about the choice -- see orchestration/mgmt_iface.py.
sparkrun_mgmt_report() (
    set +u
    read -r _if _rule <<< "$(sparkrun_mgmt_resolve "$1")"
    echo "DETECTED_MGMT_IFACE=$_if"
    echo "DETECTED_MGMT_IP=$(sparkrun_mgmt_ip "$_if")"
    echo "DETECTED_MGMT_RULE=$_rule"
    echo "DETECTED_MGMT_KIND=$(sparkrun_mgmt_kind "$_if")"
)

# The IPv4 address on interface $1 that the cluster should use for this host
# (NODE_IP, the native head address).  An interface may hold several, and its
# first is not necessarily the one on the peers' network -- e.g. a DHCP address
# plus a static cluster address on one NIC.  In order:
#   1. the source address of the most direct route to a peer through $1
#   2. the address the SSH session arrived at, when it is on $1
#   3. the first address on $1
sparkrun_mgmt_ip() (
    set +u
    _if="$1"
    [ -n "$_if" ] || exit 0
    _addrs=$(ip -4 -o addr show dev "$_if" 2>/dev/null | grep -oP 'inet \K[0-9.]+')
    [ -n "$_addrs" ] || exit 0

    _best=""
    _best_grade=0
    for _peer in $(_sparkrun_mgmt_peers); do
        while read -r _dev _grade _src; do
            [ "$_dev" = "$_if" ] || continue
            [ "$_grade" -gt "$_best_grade" ] || continue
            printf '%s\n' "$_addrs" | grep -qxF -- "$_src" || continue
            _best="$_src"
            _best_grade="$_grade"
        done < <(_sparkrun_mgmt_routes "$_peer")
    done
    if [ -n "$_best" ]; then
        printf '%s\n' "$_best"
        exit 0
    fi

    if [ -n "$SSH_CONNECTION" ]; then
        set -f
        set -- $SSH_CONNECTION
        set +f
        if [ "$#" -ge 3 ] && printf '%s\n' "$_addrs" | grep -qxF -- "$3"; then
            printf '%s\n' "$3"
            exit 0
        fi
    fi

    printf '%s\n' "$_addrs" | head -1
)

# What kind of interface $1 is: fabric (RDMA-backed), wireless, wired
# (another physical NIC) or virtual (no sysfs device: VPN, bond, bridge).
sparkrun_mgmt_kind() (
    set +u
    [ -n "$1" ] || exit 0
    _path="$(_sparkrun_mgmt_netdir)/$1"
    [ -e "$_path" ] || exit 0
    if [ -e "$_path/device/infiniband" ]; then
        echo fabric
    elif _sparkrun_mgmt_is_wireless "$_path"; then
        echo wireless
    elif [ -e "$_path/device" ]; then
        echo wired
    else
        echo virtual
    fi
)

# Interface sysfs root.  Overridable only so the resolution chain can be
# exercised against a fixture tree in the test suite; nothing in production
# sets it.
_sparkrun_mgmt_netdir() (
    set +u
    if [ -n "$SPARKRUN_NET_SYSFS" ]; then
        echo "$SPARKRUN_NET_SYSFS"
    else
        echo /sys/class/net
    fi
)

# True when the sysfs interface at $1 is a wireless NIC.
_sparkrun_mgmt_is_wireless() (
    [ -e "$1/wireless" ] || [ -e "$1/phy80211" ]
)

# The interface the SSH session arrived on.
# SSH_CONNECTION is "<client ip> <client port> <server ip> <server port>", so
# field 3 is this host's address on the path used.  Empty on the
# local-dispatch path and for an IPv6 session.  On local dispatch it may be
# the operator's own login rather than sparkrun's connection; the rules that
# read it are ordered with that in mind.
_sparkrun_mgmt_ssh_iface() (
    set +u
    [ -n "$SSH_CONNECTION" ] || exit 0
    set -f
    set -- $SSH_CONNECTION
    set +f
    [ "$#" -ge 3 ] || exit 0
    _ip="$3"
    for _path in "$(_sparkrun_mgmt_netdir)"/*; do
        [ -e "$_path" ] || continue
        _cand=$(basename "$_path")
        if ip -4 -o addr show dev "$_cand" 2>/dev/null | grep -qF "inet $_ip/"; then
            echo "$_cand"
            exit 0
        fi
    done
)

# The cluster's other hosts as IPv4 addresses, space-separated: hostnames
# resolved here, this host's own addresses and loopback dropped.
_sparkrun_mgmt_peers() (
    set +u
    [ -n "$SPARKRUN_MGMT_PEERS" ] || exit 0
    _out=""
    _own=$(ip -4 -o addr show 2>/dev/null)
    for _peer in $(printf '%s' "$SPARKRUN_MGMT_PEERS" | tr ',' ' '); do
        case "$_peer" in
            # Bounded: an air-gapped host may have a resolver it cannot reach.
            *[!0-9.]*) _peer=$(timeout 2 getent ahostsv4 "$_peer" 2>/dev/null | head -1 | cut -d' ' -f1) ;;
        esac
        [ -n "$_peer" ] || continue
        case "$_peer" in
            127.*) continue ;;
        esac
        if printf '%s\n' "$_own" | grep -qF "inet $_peer/"; then
            continue
        fi
        case " $_out " in
            *" $_peer "*) continue ;;
        esac
        _out="$_out $_peer"
    done
    printf '%s\n' "$_out"
)

# Every route that could carry traffic to IPv4 address $1, one line per
# route: "<dev> <grade> <src>", where src is the route's preferred source
# address ("-" when it names none) and grade is
#   3  the address is on one of that interface's own subnets (no gateway)
#   2  through a gateway named by a specific route
#   1  only through that interface's default route
#
# Judged from the routing tables alone -- no packet is sent, so this works
# offline.  "ip route show to match" lists every route covering the address,
# which is what lets one call answer for all interfaces.  "ip route get ...
# oif" cannot be used instead: given an interface without a route it assumes
# the address is on-link and answers anyway.  "table all" brings in policy
# tables (Tailscale keeps its routes in table 52); local and special-purpose
# entries are not ways to reach a peer.
#
# A multipath (ECMP) route prints a header line without a device, then one
# "nexthop via X dev Y" line per path; each nexthop inherits whether its
# header was a default route, and its source address.
_sparkrun_mgmt_routes() (
    set -f
    _header=3
    _hsrc=-
    ip -4 route show table all to match "$1" 2>/dev/null | while read -r _line; do
        set -- $_line
        case "$1" in
            ""|local|broadcast|anycast|multicast|unreachable|blackhole|prohibit|throw|nat) continue ;;
            nexthop)
                _grade="$_header"
                _src="$_hsrc"
                ;;
            default)
                _grade=1
                _src=-
                ;;
            *)
                _grade=3
                _src=-
                ;;
        esac
        _first="$1"
        _dev=""
        while [ "$#" -gt 0 ]; do
            case "$1" in
                via) [ "$_grade" -eq 1 ] || _grade=2 ;;
                dev) [ "$#" -lt 2 ] || _dev="$2" ;;
                src) [ "$#" -lt 2 ] || _src="$2" ;;
            esac
            shift
        done
        if [ "$_first" != nexthop ]; then
            _header="$_grade"
            _hsrc="$_src"
        fi
        [ -z "$_dev" ] || echo "$_dev $_grade $_src"
    done
)

# Rule (1) of sparkrun_mgmt_resolve: score every interface that routes to the
# cluster's other hosts and print the best, or nothing.  $1 is the sysfs root,
# $2 the SSH arrival interface (possibly empty).
#
# An interface qualifies when some route to a peer uses it, it is not down and
# it holds a global IPv4 -- virtual ones included, since a VPN or bond can
# genuinely be the management network.  The ranking is lexicographic; the
# first difference wins:
#
#   1. peers on the interface's own subnet
#   2. peers reached through a specific route
#   3. peers reached only through the interface's default route
#   4. physical (sysfs device) over virtual -- a VPN wins only by reaching
#      peers the physical NICs do not, as when hosts are listed by VPN address
#   5. wired over wireless
#   6. it is the interface the SSH session arrived on
#   7. not a CX7 fabric port -- the fabric wins only by reaching peers the
#      management network does not, as when the cluster lists fabric addresses
#   8. how many peers the kernel itself routes through it
#
# then sysfs order.  No internet route appears in it: where the default route
# points says nothing about the cluster.  Wired outranks the SSH arrival
# interface because on the local-dispatch path SSH_CONNECTION is the
# operator's own login, not sparkrun's connection: logging in over WiFi must
# not pull the cluster onto it.
_sparkrun_mgmt_by_peers() (
    _netdir="$1"
    _ssh_if="$2"
    _peers=$(_sparkrun_mgmt_peers)
    [ -n "$_peers" ] || exit 0

    # One pass over the peers, collecting "<peer#> <dev> <grade>" for every
    # route to each, and the interface the kernel would actually pick for it.
    _routes=""
    _kdevs=" "
    _n=0
    for _peer in $_peers; do
        _n=$((_n + 1))
        while read -r _dev _grade _src; do
            [ -z "$_dev" ] || _routes="$_routes$_n $_dev $_grade
"
        done < <(_sparkrun_mgmt_routes "$_peer")
        _kdev=$(ip route get "$_peer" 2>/dev/null | grep -oP 'dev \K\S+' | head -1)
        _kdevs="$_kdevs$_kdev "
    done
    [ -n "$_routes" ] || exit 0

    _best=""
    _best_key=""
    for _path in "$_netdir"/*; do
        [ -e "$_path" ] || continue
        _cand=$(basename "$_path")
        [ "$_cand" != lo ] || continue
        case "$_routes" in
            *" $_cand "*) ;;
            *) continue ;;
        esac
        [ "$(cat "$_path/operstate" 2>/dev/null)" != "down" ] || continue
        ip -4 -o addr show dev "$_cand" scope global 2>/dev/null | grep -q 'inet ' || continue

        # Best grade per peer for this interface.  Lines for one peer are
        # contiguous, so a running maximum suffices.
        _direct=0
        _routed=0
        _defaulted=0
        _cur=""
        _max=0
        while read -r _pn _dev _grade; do
            if [ "$_pn" != "$_cur" ]; then
                case "$_max" in
                    3) _direct=$((_direct + 1)) ;;
                    2) _routed=$((_routed + 1)) ;;
                    1) _defaulted=$((_defaulted + 1)) ;;
                esac
                _cur="$_pn"
                _max=0
            fi
            if [ "$_dev" = "$_cand" ] && [ "$_grade" -gt "$_max" ]; then
                _max="$_grade"
            fi
        done <<< "$_routes
end - 0"
        [ $((_direct + _routed + _defaulted)) -gt 0 ] || continue

        _phys=0
        [ ! -e "$_path/device" ] || _phys=1
        _wired=1
        if _sparkrun_mgmt_is_wireless "$_path"; then
            _wired=0
        fi
        _ctl=0
        [ "$_cand" != "$_ssh_if" ] || _ctl=1
        _nonfab=1
        [ ! -e "$_path/device/infiniband" ] || _nonfab=0
        _kernel=0
        for _kdev in $_kdevs; do
            [ "$_kdev" != "$_cand" ] || _kernel=$((_kernel + 1))
        done

        # Fixed-width key, compared as a string: three-digit counts (a cluster
        # is never 1000 hosts) and one-digit flags, in ranking order.
        _key=$(printf '%03d%03d%03d%d%d%d%d%03d' "$_direct" "$_routed" "$_defaulted" "$_phys" "$_wired" "$_ctl" "$_nonfab" "$_kernel")
        if [ -z "$_best" ] || [[ "$_key" > "$_best_key" ]]; then
            _best="$_cand"
            _best_key="$_key"
        fi
    done
    [ -z "$_best" ] || echo "$_best"
)
