# Persistence attribution for a network interface's IPv4 address.
#
# Answers "will this address still be here after a reboot?" by asking the
# config sources that could own the interface, rather than by looking for one
# specific file.  A file check answers "did sparkrun write this?", which is a
# different question: on Ubuntu 24.04 / DGX OS 7 `nmcli con add` writes its own
# netplan file (90-NM-<uuid>.yaml), a hand-rolled netplan file can carry any
# number, and NetworkManager / systemd-networkd / ifupdown persist an address
# with no netplan file at all.
#
# Usage:
#   result=$(sparkrun_net_persistence <iface> [<ipv4-addr>])
#   -> "<state>|<source>|<detail>"
#      state:  persistent | ephemeral | unknown
#      source: netplan | networkmanager | systemd-networkd | ifupdown | ""
#      detail: netplan id, NM profile, .network path (may contain spaces)
#
#   sparkrun_net_is_dhcp <iface>   -> 1 when the live address is a DHCP lease
#
# "unknown" means *no probe was available* and must never be reported as "not
# persistent" -- the same rule TerminationInfo.exists=None follows.  Every
# probe is read-only and works unprivileged.
#
# NOTE: this helper uses ${...} parameter expansion, so it must NOT be
# included in a script that is later passed through Python's str.format().

_SPARKRUN_PERSIST_INIT=0
_SPARKRUN_NETPLAN_MAP=""
_SPARKRUN_NETPLAN_PROBED=0
_SPARKRUN_NM_LIST=""
_SPARKRUN_NM_PROBED=0
_SPARKRUN_NETWORKD_PROBED=0

_sparkrun_persist_probe_init() {
    if [ "$_SPARKRUN_PERSIST_INIT" = "1" ]; then
        return 0
    fi
    _SPARKRUN_PERSIST_INIT=1

    # netplan reports the *merged* view of /etc/netplan, /run/netplan and
    # /lib/netplan, so any filename counts.  `netplan status` works
    # unprivileged; `netplan get` does not (the files are mode 600).
    # Interfaces netplan owns carry an "id"; everything else has none.
    if command -v netplan >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
        local raw=""
        raw=$(netplan status --format=json 2>/dev/null) || raw=""
        if [ -n "$raw" ]; then
            _SPARKRUN_NETPLAN_MAP=$(printf '%s' "$raw" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)
if not isinstance(data, dict):
    raise SystemExit(0)
for name, info in data.items():
    if isinstance(info, dict) and info.get("id"):
        sys.stdout.write("%s\t%s\n" % (name, info["id"]))
' 2>/dev/null) || _SPARKRUN_NETPLAN_MAP=""
            _SPARKRUN_NETPLAN_PROBED=1
        fi
    fi

    # A failing nmcli means NetworkManager is not running -- treat that as
    # "probe unavailable" rather than "NM does not own it", so a host we
    # could not fully interrogate degrades to unknown, not to ephemeral.
    if command -v nmcli >/dev/null 2>&1; then
        local nm=""
        if nm=$(nmcli -t -f DEVICE,NAME connection show --active 2>/dev/null); then
            _SPARKRUN_NM_LIST="$nm"
            _SPARKRUN_NM_PROBED=1
        fi
    fi

    if command -v networkctl >/dev/null 2>&1; then
        _SPARKRUN_NETWORKD_PROBED=1
    fi
}

sparkrun_net_persistence() {
    local ifc="${1:-}"
    local addr="${2:-}"
    if [ -z "$ifc" ]; then
        printf 'unknown||\n'
        return 0
    fi
    _sparkrun_persist_probe_init

    # --- 1. netplan (any file, any renderer) ---
    if [ "$_SPARKRUN_NETPLAN_PROBED" = "1" ]; then
        local np_id=""
        np_id=$(printf '%s\n' "$_SPARKRUN_NETPLAN_MAP" | awk -F'\t' -v i="$ifc" '$1 == i { print $2; exit }') || np_id=""
        if [ -n "$np_id" ]; then
            printf 'persistent|netplan|%s\n' "$np_id"
            return 0
        fi
    fi

    # --- 2. NetworkManager ---
    # Device names cannot contain ":", so split on the first one only: an NM
    # profile name can, and nmcli -t escapes it rather than omitting it.
    if [ "$_SPARKRUN_NM_PROBED" = "1" ]; then
        local prof=""
        prof=$(printf '%s\n' "$_SPARKRUN_NM_LIST" | awk -F: -v i="$ifc" '$1 == i { print substr($0, index($0, ":") + 1); exit }') || prof=""
        if [ -n "$prof" ]; then
            local props="" autoconnect="" method="" addrs=""
            props=$(nmcli -t -f connection.autoconnect,ipv4.method,ipv4.addresses connection show "$prof" 2>/dev/null) || props=""
            autoconnect=$(printf '%s\n' "$props" | sed -n 's/^connection\.autoconnect://p' | head -1) || autoconnect=""
            method=$(printf '%s\n' "$props" | sed -n 's/^ipv4\.method://p' | head -1) || method=""
            addrs=$(printf '%s\n' "$props" | sed -n 's/^ipv4\.addresses://p' | head -1) || addrs=""

            if [ "$autoconnect" = "no" ]; then
                printf 'ephemeral|networkmanager|%s (autoconnect disabled)\n' "$prof"
                return 0
            fi
            if [ "$method" = "manual" ]; then
                if [ -z "$addr" ] || printf '%s' "$addrs" | grep -Fq -- "$addr/"; then
                    printf 'persistent|networkmanager|%s\n' "$prof"
                else
                    printf 'ephemeral|networkmanager|%s pins %s, not %s\n' "$prof" "${addrs:-nothing}" "$addr"
                fi
                return 0
            fi
            if [ -n "$method" ] && [ "$method" != "disabled" ]; then
                printf 'persistent|networkmanager|%s (ipv4.method=%s)\n' "$prof" "$method"
                return 0
            fi
        fi
    fi

    # --- 3. systemd-networkd ---
    if [ "$_SPARKRUN_NETWORKD_PROBED" = "1" ]; then
        local netfile=""
        netfile=$(SYSTEMD_COLORS=0 networkctl status --no-pager "$ifc" 2>/dev/null | sed -n 's/.*Network File: *//p' | head -1) || netfile=""
        netfile=$(printf '%s' "$netfile" | tr -d '\r')
        if [ -n "$netfile" ] && [ "$netfile" != "n/a" ]; then
            printf 'persistent|systemd-networkd|%s\n' "$netfile"
            return 0
        fi
    fi

    # --- 4. ifupdown ---
    local f
    for f in /etc/network/interfaces /etc/network/interfaces.d/*; do
        [ -f "$f" ] || continue
        if grep -Eq "^[[:space:]]*iface[[:space:]]+$ifc([[:space:]]|\$)" "$f" 2>/dev/null; then
            printf 'persistent|ifupdown|%s\n' "$f"
            return 0
        fi
    done

    # --- 5. Nothing claims it ---
    # Only call that ephemeral if we actually got to ask someone.
    if [ "$_SPARKRUN_NETPLAN_PROBED" = "1" ] || [ "$_SPARKRUN_NM_PROBED" = "1" ] || [ "$_SPARKRUN_NETWORKD_PROBED" = "1" ]; then
        printf 'ephemeral||\n'
    else
        printf 'unknown||\n'
    fi
    return 0
}

sparkrun_net_is_dhcp() {
    local ifc="${1:-}"
    if [ -n "$ifc" ] && ip -4 addr show "$ifc" 2>/dev/null | grep -m1 'inet ' | grep -qw dynamic; then
        echo 1
    else
        echo 0
    fi
}
