# Scoped sudoers readiness: which of sparkrun's NOPASSWD entries already work
# for $WHO. Read-only and never prompts. Shared by setup_check.sh (the
# `sudoers` check) and the standalone --save-sudo commands, so the check and
# the install decision cannot disagree.
#
# Inputs: WHO (the probed user); optional SPARKRUN_SUDOERS_CACHE_DIR, the
# directory the chown entry permits (default: WHO's ~/.cache/huggingface).
# NOTE brace-free, comments included: setup_check.sh includes this and is
# rendered with str.format().
_SUDOERS_CHOWN_DIR="$(getent passwd "$WHO" 2>/dev/null | cut -d: -f6)/.cache/huggingface"
if declare -p SPARKRUN_SUDOERS_CACHE_DIR >/dev/null 2>&1; then
    _SUDOERS_CHOWN_DIR="$SPARKRUN_SUDOERS_CACHE_DIR"
fi
if sudo -n true 2>/dev/null; then
    # Unrestricted passwordless sudo: look for the entry files themselves.
    if sudo -n test -e "/etc/sudoers.d/@RESOURCE_NAMESPACE@-chown-$WHO" 2>/dev/null; then
        echo "CHECK_SUDOERS_CHOWN=1"
    else
        echo "CHECK_SUDOERS_CHOWN=0"
    fi
    if sudo -n test -e "/etc/sudoers.d/@RESOURCE_NAMESPACE@-dropcaches-$WHO" 2>/dev/null; then
        echo "CHECK_SUDOERS_DROPCACHES=1"
    else
        echo "CHECK_SUDOERS_DROPCACHES=0"
    fi
else
    # Password sudo -- the hosts these entries exist for. The rule listing
    # needs no password once any NOPASSWD rule exists (sudoers' default
    # listpw=any); with none, it fails, which also means ours is absent.
    # Match our exact command on a NOPASSWD line: `sudo -l <command>` alone
    # would also accept a password-requiring rule such as (ALL) ALL. A site
    # setting listpw=always reads as missing; the install is idempotent, so
    # that costs a password prompt, never a skipped install.
    _SUDOERS_NOPASSWD="$(sudo -n -l 2>/dev/null | grep NOPASSWD || true)"
    case "$_SUDOERS_NOPASSWD" in
        *"/usr/bin/chown -R $WHO $_SUDOERS_CHOWN_DIR"*) echo "CHECK_SUDOERS_CHOWN=1" ;;
        *) echo "CHECK_SUDOERS_CHOWN=0" ;;
    esac
    case "$_SUDOERS_NOPASSWD" in
        *"/usr/bin/tee /proc/sys/vm/drop_caches"*) echo "CHECK_SUDOERS_DROPCACHES=1" ;;
        *) echo "CHECK_SUDOERS_DROPCACHES=0" ;;
    esac
fi
