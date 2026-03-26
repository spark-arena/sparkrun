#!/bin/bash
# Sudo-only diagnostic collection for DGX Spark.
# Collects data that requires root privileges (dmidecode).
# Outputs key=value pairs on stdout (parsed by sparkrun).
# Diagnostic messages go to stderr so they don't pollute parsing.
#
# This script is run via sudo as a second pass after spark_diagnose.sh.

set -uo pipefail

echo "Running sudo diagnostics..." >&2

# --- dmidecode ---
if command -v dmidecode &>/dev/null; then
    echo "  Collecting DMI/SMBIOS data..." >&2

    # BIOS Information
    DIAG_DMI_BIOS_VENDOR=$(dmidecode -s bios-vendor 2>/dev/null || echo "")
    DIAG_DMI_BIOS_VERSION=$(dmidecode -s bios-version 2>/dev/null || echo "")
    DIAG_DMI_BIOS_DATE=$(dmidecode -s bios-release-date 2>/dev/null || echo "")
    echo "DIAG_DMI_BIOS_VENDOR=$DIAG_DMI_BIOS_VENDOR"
    echo "DIAG_DMI_BIOS_VERSION=$DIAG_DMI_BIOS_VERSION"
    echo "DIAG_DMI_BIOS_DATE=$DIAG_DMI_BIOS_DATE"

    # System Information
    DIAG_DMI_SYS_MANUFACTURER=$(dmidecode -s system-manufacturer 2>/dev/null || echo "")
    DIAG_DMI_SYS_PRODUCT=$(dmidecode -s system-product-name 2>/dev/null || echo "")
    DIAG_DMI_SYS_VERSION=$(dmidecode -s system-version 2>/dev/null || echo "")
    DIAG_DMI_SYS_SERIAL=$(dmidecode -s system-serial-number 2>/dev/null || echo "")
    DIAG_DMI_SYS_UUID=$(dmidecode -s system-uuid 2>/dev/null || echo "")
    echo "DIAG_DMI_SYS_MANUFACTURER=$DIAG_DMI_SYS_MANUFACTURER"
    echo "DIAG_DMI_SYS_PRODUCT=$DIAG_DMI_SYS_PRODUCT"
    echo "DIAG_DMI_SYS_VERSION=$DIAG_DMI_SYS_VERSION"
    echo "DIAG_DMI_SYS_SERIAL=$DIAG_DMI_SYS_SERIAL"
    echo "DIAG_DMI_SYS_UUID=$DIAG_DMI_SYS_UUID"

    # Baseboard Information
    DIAG_DMI_BOARD_MANUFACTURER=$(dmidecode -s baseboard-manufacturer 2>/dev/null || echo "")
    DIAG_DMI_BOARD_PRODUCT=$(dmidecode -s baseboard-product-name 2>/dev/null || echo "")
    DIAG_DMI_BOARD_VERSION=$(dmidecode -s baseboard-version 2>/dev/null || echo "")
    DIAG_DMI_BOARD_SERIAL=$(dmidecode -s baseboard-serial-number 2>/dev/null || echo "")
    echo "DIAG_DMI_BOARD_MANUFACTURER=$DIAG_DMI_BOARD_MANUFACTURER"
    echo "DIAG_DMI_BOARD_PRODUCT=$DIAG_DMI_BOARD_PRODUCT"
    echo "DIAG_DMI_BOARD_VERSION=$DIAG_DMI_BOARD_VERSION"
    echo "DIAG_DMI_BOARD_SERIAL=$DIAG_DMI_BOARD_SERIAL"

    # Memory summary (total slots and populated)
    DIAG_DMI_MEM_SLOTS=$(dmidecode -t memory 2>/dev/null | grep -c "Memory Device" || echo "0")
    DIAG_DMI_MEM_POPULATED=$(dmidecode -t memory 2>/dev/null | grep "Size:" | grep -cv "No Module Installed" || echo "0")
    DIAG_DMI_MEM_MAX=$(dmidecode -t memory 2>/dev/null | grep "Maximum Capacity:" | head -1 | sed 's/.*: *//' || echo "")
    echo "DIAG_DMI_MEM_SLOTS=$DIAG_DMI_MEM_SLOTS"
    echo "DIAG_DMI_MEM_POPULATED=$DIAG_DMI_MEM_POPULATED"
    echo "DIAG_DMI_MEM_MAX=$DIAG_DMI_MEM_MAX"
else
    echo "  dmidecode not found" >&2
fi

# --- Completion sentinel ---
echo "DIAG_SUDO_COMPLETE=1"
echo "Sudo diagnostics complete." >&2
