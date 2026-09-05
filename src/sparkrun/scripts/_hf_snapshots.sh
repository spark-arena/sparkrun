# shellcheck shell=bash
# Shared HuggingFace snapshot resolution for sparkrun's model ensure scripts.
#
# Not executable on its own: included into the other scripts through the
# "# sparkrun:include" directive resolved by sparkrun.scripts.read_script.
#
# WHY THIS EXISTS
# ---------------
# model_sync.sh and model_sync_gguf.sh answered "are the weights already here?"
# with a single recursive scan of EVERY snapshot:
#
#     find "$CACHE_PATH/snapshots" -name '*.safetensors' -print -quit
#
# while the recipe's revision reached only the downloader below it.  So a host
# holding some OTHER revision of the repo reported a cache hit, the pinned
# revision was never fetched, and the workload served weights the recipe had
# explicitly pinned against -- with no error and nothing in the log.  That is
# the exact failure a revision pin exists to prevent.
#
# The control-side peer, models.download.is_model_cached, gets this right, and
# its contract is what is mirrored here:
#
#   * an explicit revision checks that ref or commit hash ONLY, with no
#     fallback -- "cached, but not the revision you asked for" is a miss;
#   * an unpinned lookup prefers refs/main (snapshot_download's own default)
#     and falls back to any snapshot, which is what makes a hand-placed cache
#     directory work.
#
# Unlike the models--org--name mangling -- pure string work on a known input,
# so it moved to Python and is rendered in as CACHE_PATH -- this resolution
# reads the TARGET host's filesystem and cannot be computed control-side.  It
# has to be bash; one copy, exercised under real bash against a fixture tree
# in tests/test_model_cache_check.py, is the mitigation.
#
# STYLE CONSTRAINTS -- both load-bearing, do not "tidy up":
#
#   * No curly braces anywhere -- not even in these comments.  Both including
#     scripts are passed through Python's str.format(), which raises KeyError
#     on any brace appearing here.  That rules out the braced parameter form
#     (write "$VAR", never the dollar-brace one) and the braced function-body
#     syntax, which is why the function below has a ( subshell ) body.
#   * The subshell body also isolates the temporaries, so the helper cannot
#     disturb the including script's state.

# Usage: DIRS=$(sparkrun_hf_snapshot_dirs "<model cache path>" "<revision or empty>")
#
# Prints the snapshot directories to probe, one per line, most specific first.
# Prints nothing when none resolve -- for a pinned revision that means "not
# cached", never "check whatever else happens to be lying around".
sparkrun_hf_snapshot_dirs() (
    # The revision argument is legitimately empty on an unpinned launch.
    set +u

    _cache="$1"
    _rev="$2"
    _snaps="$_cache/snapshots"
    [ -d "$_snaps" ] || exit 0

    # An unpinned lookup asks for refs/main, matching snapshot_download.
    _want="$_rev"
    if [ -z "$_want" ]; then
        _want=main
    fi

    # refs/<name> holds the commit hash the ref points at; the revision may
    # also BE that hash, naming the snapshot directory directly.  Both are
    # checked, and the second is suppressed when it would repeat the first --
    # mirroring _snapshot_dirs_for_revision, de-duplication included.
    _hash=""
    if [ -f "$_cache/refs/$_want" ]; then
        _hash=$(tr -d '[:space:]' < "$_cache/refs/$_want" 2>/dev/null)
    fi

    _found=""
    if [ -n "$_hash" ] && [ -d "$_snaps/$_hash" ]; then
        printf '%s\n' "$_snaps/$_hash"
        _found=yes
    fi
    if [ "$_want" != "$_hash" ] && [ -d "$_snaps/$_want" ]; then
        printf '%s\n' "$_snaps/$_want"
        _found=yes
    fi

    if [ -n "$_found" ]; then
        exit 0
    fi

    # Nothing resolved.  A pinned revision stops here: falling back to another
    # snapshot is precisely the bug this helper exists to remove.
    if [ -n "$_rev" ]; then
        exit 0
    fi

    for _d in "$_snaps"/*; do
        [ -d "$_d" ] || continue
        printf '%s\n' "$_d"
    done
    exit 0
)
