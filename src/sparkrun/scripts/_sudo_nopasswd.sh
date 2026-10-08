# Does the probed user have unrestricted passwordless sudo? Read-only.
# Sets SUDO_NOPASSWD and emits CHECK_SUDO_NOPASSWD: a host that answers 1 must
# be driven with `sudo -n`, never with a password on `sudo -S` stdin (sudo does
# not read it there, so bash would run the password line as a command).
# NOTE brace-free: included by setup_check.sh, rendered with str.format().
if sudo -n true 2>/dev/null; then
    SUDO_NOPASSWD=1
else
    SUDO_NOPASSWD=0
fi
echo "CHECK_SUDO_NOPASSWD=$SUDO_NOPASSWD"
