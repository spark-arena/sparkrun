# shellcheck shell=bash
# Shared RoCE GID-index selection for sparkrun's InfiniBand probes.
#
# Not executable on its own: included into ib_detect.sh through the
# "# sparkrun:include" directive, and substituted into the combined hardware
# probe (core/hardware_probe.py) verbatim.
#
# WHAT IT ANSWERS
# ---------------
# The index becomes NCCL_IB_GID_INDEX.  Peers address each other with
# IPv4-mapped GIDs (::ffff:a.b.c.d), so the only correct pin is the RoCE v2 GID
# carrying the port's IPv4 address.  The chain, first hit wins:
#
#   1. show_gids: the first "v2" row with an IPv4 address for this HCA.  This
#      is what a stock DGX OS host answers, unchanged from before this helper.
#   2. sysfs: the first RoCE v2 entry whose GID is IPv4-mapped.  For hosts
#      without show_gids.  The old fallback took the first RoCE v2 entry of any
#      kind -- on ConnectX that is index 1, the link-local fe80:: GID, which
#      bootstraps and then fails QP setup with "ibv_modify_qp: 22" (#327).
#   3. -1: NCCL's own dynamic selection (RoCE v2 + AF_INET by default, NCCL
#      2.21+; 2.32+ keeps -1 as an alias for AUTO).  The old last resort was a
#      hardcoded 3, which is right only on a port laid out like a Spark's.
#      When we cannot find the IPv4 GID ourselves, NCCL searching for the same
#      thing is strictly better than a guess.
#
# STYLE CONSTRAINT -- load-bearing, do not "tidy up":
#
#   No curly braces anywhere, not even in these comments.  The combined probe
#   is passed through Python's str.format(), which raises KeyError on any brace
#   appearing here.  That rules out awk programs (show_gids is parsed with
#   "read" instead) and the braced function-body syntax, which is why the
#   function is defined with a ( subshell ) body.
#
# Usage: IDX=$(sparkrun_rocev2_gid_index "<hca name>" "<hca sysfs dir>")
#
# Always prints an index.  Diagnostics go to stderr.
sparkrun_rocev2_gid_index() (
    set +u
    _hca="$1"
    _port="$2/ports/1"

    if command -v show_gids >/dev/null 2>&1; then
        # Columns: DEV PORT INDEX GID IPv4 VER DEV.  A row without an IPv4
        # address has an empty column, so every later column shifts left: the
        # netdev name lands in _ver and the row fails the v2 test -- the same
        # field shift the original awk match relied on.
        _idx=$(show_gids 2>/dev/null | while read -r _dev _p _i _gid _ipv4 _ver _rest; do
            if [ "$_dev" = "$_hca" ] && [ "$_ver" = "v2" ]; then
                case "$_ipv4" in
                    [0-9]*.[0-9]*.[0-9]*.[0-9]*) echo "$_i"; break ;;
                esac
            fi
        done)
        if [ -n "$_idx" ]; then
            echo "Device $_hca: Active (RoCEv2 GID Index: $_idx)" >&2
            echo "$_idx"
            exit 0
        fi
    fi

    if [ -d "$_port/gid_attrs/types" ]; then
        # Numeric order: a glob would visit 10 before 2.
        for _i in $(ls "$_port/gid_attrs/types" 2>/dev/null | sort -n); do
            case "$_i" in
                *[!0-9]*) continue ;;
            esac
            # Unpopulated entries fail to read (EINVAL); skip them.
            _type=$(cat "$_port/gid_attrs/types/$_i" 2>/dev/null) || continue
            case "$_type" in
                *"RoCE v2"*|*RoCEv2*) ;;
                *) continue ;;
            esac
            _gid=$(cat "$_port/gids/$_i" 2>/dev/null) || continue
            case "$_gid" in
                0000:0000:0000:0000:0000:ffff:*)
                    echo "Device $_hca: Active (RoCEv2 GID Index: $_i via sysfs)" >&2
                    echo "$_i"
                    exit 0
                    ;;
            esac
        done
    fi

    echo "Device $_hca: Active (no IPv4 RoCEv2 GID found; leaving GID selection to NCCL with -1)" >&2
    echo "-1"
)

# Usage: IDX=$(sparkrun_gid_index_consensus "<index>" ["<index>" ...])
# Reduce a nonempty per-HCA index list; disagreement defers selection to NCCL.
sparkrun_gid_index_consensus() (
    if [ "$#" -eq 0 ]; then
        echo "GID index consensus requires at least one index" >&2
        exit 2
    fi

    _first="$1"
    shift
    _mismatch=0
    for _index in "$@"; do
        if [ "$_index" != "$_first" ]; then
            _mismatch=1
        fi
    done

    if [ "$_mismatch" -ne 0 ]; then
        echo "GID index disagreement across active HCAs; leaving GID selection to NCCL with -1" >&2
        echo "-1"
    else
        echo "$_first"
    fi
)
