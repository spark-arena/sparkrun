#!/bin/bash
# Uninstall earlyoom configured by sparkrun setup.
# Params: {remove_package} — "true" to apt-get remove, "false" to leave installed
set -euo pipefail

REMOVE_PACKAGE="{remove_package}"

# Stop and disable earlyoom service
if systemctl is-active --quiet earlyoom 2>/dev/null; then
    sudo -n systemctl stop earlyoom
    echo "STOPPED: earlyoom service"
else
    echo "SKIPPED: earlyoom service not running"
fi

if systemctl is-enabled --quiet earlyoom 2>/dev/null; then
    sudo -n systemctl disable earlyoom
    echo "DISABLED: earlyoom service"
fi

# Remove sparkrun configuration
if [ -f /etc/default/earlyoom ]; then
    sudo -n rm -f /etc/default/earlyoom
    echo "REMOVED: /etc/default/earlyoom"
fi

# Remove systemd override
OVERRIDE_DIR="/etc/systemd/system/earlyoom.service.d"
if [ -d "$OVERRIDE_DIR" ]; then
    sudo -n rm -rf "$OVERRIDE_DIR"
    echo "REMOVED: $OVERRIDE_DIR"
fi

# Reload systemd
sudo -n systemctl daemon-reload
echo "RELOADED: systemd daemon"

# Optionally remove the package
if [ "$REMOVE_PACKAGE" = "true" ]; then
    if command -v earlyoom >/dev/null 2>&1; then
        sudo -n DEBIAN_FRONTEND=noninteractive apt-get remove -y -qq earlyoom
        echo "UNINSTALLED: earlyoom package"
    else
        echo "SKIPPED: earlyoom package not installed"
    fi
else
    echo "KEPT: earlyoom package (was already installed before sparkrun)"
fi

echo "OK: earlyoom uninstall complete"
