#!/bin/bash
# Comprehensive host diagnostic collection for DGX Spark.
# Outputs key=value pairs on stdout (parsed by sparkrun).
# Diagnostic messages go to stderr so they don't pollute parsing.
#
# Follows the same pattern as ib_detect.sh / cx7_detect.sh.

set -uo pipefail

echo "Running spark diagnostics..." >&2

# --- Hostname ---
echo "DIAG_HOSTNAME=$(hostname 2>/dev/null || echo unknown)"
echo "DIAG_HOSTNAME_FQDN=$(hostname -f 2>/dev/null || echo unknown)"

# --- OS / Kernel ---
if [ -f /etc/os-release ]; then
    DIAG_OS_NAME=$(. /etc/os-release && echo "${NAME:-unknown}")
    DIAG_OS_VERSION=$(. /etc/os-release && echo "${VERSION_ID:-unknown}")
    DIAG_OS_PRETTY=$(. /etc/os-release && echo "${PRETTY_NAME:-unknown}")
else
    DIAG_OS_NAME="unknown"
    DIAG_OS_VERSION="unknown"
    DIAG_OS_PRETTY="unknown"
fi
echo "DIAG_OS_NAME=$DIAG_OS_NAME"
echo "DIAG_OS_VERSION=$DIAG_OS_VERSION"
echo "DIAG_OS_PRETTY=$DIAG_OS_PRETTY"
echo "DIAG_KERNEL=$(uname -r)"
echo "DIAG_ARCH=$(uname -m)"

# --- Firmware / Board ---
DIAG_BIOS_VERSION=""
if [ -f /sys/class/dmi/id/bios_version ]; then
    DIAG_BIOS_VERSION=$(cat /sys/class/dmi/id/bios_version 2>/dev/null || echo "")
fi
echo "DIAG_BIOS_VERSION=$DIAG_BIOS_VERSION"

DIAG_BOARD_NAME=""
if [ -f /sys/class/dmi/id/board_name ]; then
    DIAG_BOARD_NAME=$(cat /sys/class/dmi/id/board_name 2>/dev/null || echo "")
fi
echo "DIAG_BOARD_NAME=$DIAG_BOARD_NAME"

DIAG_PRODUCT_NAME=""
if [ -f /sys/class/dmi/id/product_name ]; then
    DIAG_PRODUCT_NAME=$(cat /sys/class/dmi/id/product_name 2>/dev/null || echo "")
fi
echo "DIAG_PRODUCT_NAME=$DIAG_PRODUCT_NAME"

# --- JetPack / L4T ---
DIAG_JETPACK_VERSION=""
if command -v dpkg &>/dev/null; then
    DIAG_JETPACK_VERSION=$(dpkg -l nvidia-jetpack 2>/dev/null | awk '/^ii/{print $3}' || echo "")
fi
if [ -z "$DIAG_JETPACK_VERSION" ] && [ -f /etc/nv_tegra_release ]; then
    DIAG_JETPACK_VERSION=$(head -1 /etc/nv_tegra_release 2>/dev/null || echo "")
fi
echo "DIAG_JETPACK_VERSION=$DIAG_JETPACK_VERSION"

# --- CPU ---
DIAG_CPU_MODEL=$(lscpu 2>/dev/null | grep -i "model name" | head -1 | sed 's/.*:\s*//' || echo "unknown")
DIAG_CPU_CORES=$(nproc 2>/dev/null || echo "0")
DIAG_CPU_THREADS=$(lscpu 2>/dev/null | grep -i "^CPU(s):" | head -1 | awk '{print $2}' || echo "0")
echo "DIAG_CPU_MODEL=$DIAG_CPU_MODEL"
echo "DIAG_CPU_CORES=$DIAG_CPU_CORES"
echo "DIAG_CPU_THREADS=$DIAG_CPU_THREADS"

# --- Memory ---
DIAG_RAM_TOTAL_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")
DIAG_RAM_AVAILABLE_KB=$(grep MemAvailable /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")
echo "DIAG_RAM_TOTAL_KB=$DIAG_RAM_TOTAL_KB"
echo "DIAG_RAM_AVAILABLE_KB=$DIAG_RAM_AVAILABLE_KB"

# --- Disk ---
DIAG_DISK_ROOT_TOTAL=$(df -k / 2>/dev/null | awk 'NR==2{print $2}' || echo "0")
DIAG_DISK_ROOT_AVAIL=$(df -k / 2>/dev/null | awk 'NR==2{print $4}' || echo "0")
DIAG_DISK_HOME_TOTAL=$(df -k /home 2>/dev/null | awk 'NR==2{print $2}' || echo "0")
DIAG_DISK_HOME_AVAIL=$(df -k /home 2>/dev/null | awk 'NR==2{print $4}' || echo "0")
echo "DIAG_DISK_ROOT_TOTAL_KB=$DIAG_DISK_ROOT_TOTAL"
echo "DIAG_DISK_ROOT_AVAIL_KB=$DIAG_DISK_ROOT_AVAIL"
echo "DIAG_DISK_HOME_TOTAL_KB=$DIAG_DISK_HOME_TOTAL"
echo "DIAG_DISK_HOME_AVAIL_KB=$DIAG_DISK_HOME_AVAIL"

# --- GPU ---
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version,pstate,temperature.gpu,power.draw,gpu_serial,gpu_uuid --format=csv,noheader,nounits 2>/dev/null || echo "")
    if [ -n "$GPU_INFO" ]; then
        IFS=',' read -r gpu_name gpu_mem gpu_driver gpu_pstate gpu_temp gpu_power gpu_serial gpu_uuid <<< "$GPU_INFO"
        echo "DIAG_GPU_NAME=$(echo "$gpu_name" | xargs)"
        echo "DIAG_GPU_MEMORY_MB=$(echo "$gpu_mem" | xargs)"
        echo "DIAG_GPU_DRIVER=$(echo "$gpu_driver" | xargs)"
        echo "DIAG_GPU_PSTATE=$(echo "$gpu_pstate" | xargs)"
        echo "DIAG_GPU_TEMP_C=$(echo "$gpu_temp" | xargs)"
        echo "DIAG_GPU_POWER_W=$(echo "$gpu_power" | xargs)"
        echo "DIAG_GPU_SERIAL=$(echo "$gpu_serial" | xargs)"
        echo "DIAG_GPU_UUID=$(echo "$gpu_uuid" | xargs)"
    else
        echo "DIAG_GPU_NAME="
        echo "DIAG_GPU_MEMORY_MB="
        echo "DIAG_GPU_DRIVER="
    fi
else
    echo "DIAG_GPU_NAME="
    echo "DIAG_GPU_MEMORY_MB="
    echo "DIAG_GPU_DRIVER="
    echo "  nvidia-smi not found" >&2
fi

# --- CUDA ---
DIAG_CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
    DIAG_CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' || echo "")
fi
echo "DIAG_CUDA_VERSION=$DIAG_CUDA_VERSION"

# --- Network interfaces ---
NET_IDX=0
for netdev in /sys/class/net/*; do
    [ -e "$netdev" ] || continue
    iface=$(basename "$netdev")
    # Skip loopback
    [ "$iface" = "lo" ] && continue

    state=$(cat "$netdev/operstate" 2>/dev/null || echo "unknown")
    mtu=$(cat "$netdev/mtu" 2>/dev/null || echo "0")
    mac=$(cat "$netdev/address" 2>/dev/null || echo "")
    speed=$(cat "$netdev/speed" 2>/dev/null || echo "")
    ip_addr=$(ip -4 addr show "$iface" 2>/dev/null | grep -oP 'inet \K[0-9.]+' | head -1 || echo "")

    echo "DIAG_NET_${NET_IDX}_NAME=$iface"
    echo "DIAG_NET_${NET_IDX}_STATE=$state"
    echo "DIAG_NET_${NET_IDX}_MTU=$mtu"
    echo "DIAG_NET_${NET_IDX}_MAC=$mac"
    echo "DIAG_NET_${NET_IDX}_SPEED=$speed"
    echo "DIAG_NET_${NET_IDX}_IP=$ip_addr"

    NET_IDX=$((NET_IDX + 1))
done
echo "DIAG_NET_COUNT=$NET_IDX"

# --- Default route ---
DIAG_DEFAULT_IFACE=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'dev \K\S+' || echo "")
DIAG_MGMT_IP=$(ip -4 addr show "$DIAG_DEFAULT_IFACE" 2>/dev/null | grep -oP 'inet \K[0-9.]+' | head -1 || echo "")
echo "DIAG_DEFAULT_IFACE=$DIAG_DEFAULT_IFACE"
echo "DIAG_MGMT_IP=$DIAG_MGMT_IP"

# --- Docker ---
if command -v docker &>/dev/null; then
    DIAG_DOCKER_VERSION=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown")
    DIAG_DOCKER_STORAGE=$(docker info --format '{{.Driver}}' 2>/dev/null || echo "unknown")
    DIAG_DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo "unknown")

    # Check nvidia runtime
    DIAG_DOCKER_NVIDIA_RUNTIME="false"
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
        DIAG_DOCKER_NVIDIA_RUNTIME="true"
    fi

    DIAG_DOCKER_RUNNING=$(docker ps -q 2>/dev/null | wc -l | xargs)
    DIAG_DOCKER_SPARKRUN=$(docker ps --filter "label=sparkrun" -q 2>/dev/null | wc -l | xargs)
else
    DIAG_DOCKER_VERSION=""
    DIAG_DOCKER_STORAGE=""
    DIAG_DOCKER_ROOT=""
    DIAG_DOCKER_NVIDIA_RUNTIME="false"
    DIAG_DOCKER_RUNNING="0"
    DIAG_DOCKER_SPARKRUN="0"
    echo "  docker not found" >&2
fi
echo "DIAG_DOCKER_VERSION=$DIAG_DOCKER_VERSION"
echo "DIAG_DOCKER_STORAGE=$DIAG_DOCKER_STORAGE"
echo "DIAG_DOCKER_ROOT=$DIAG_DOCKER_ROOT"
echo "DIAG_DOCKER_NVIDIA_RUNTIME=$DIAG_DOCKER_NVIDIA_RUNTIME"
echo "DIAG_DOCKER_RUNNING=$DIAG_DOCKER_RUNNING"
echo "DIAG_DOCKER_SPARKRUN=$DIAG_DOCKER_SPARKRUN"

# --- Firmware updates (fwupdmgr) ---
# get-devices and get-history work without sudo on most systems
if command -v fwupdmgr &>/dev/null; then
    echo "  Collecting firmware device inventory..." >&2
    # Compact single-line per device: Name | Version | GUID
    FW_DEV_IDX=0
    while IFS= read -r line; do
        echo "DIAG_FWUPD_DEV_${FW_DEV_IDX}=$line"
        FW_DEV_IDX=$((FW_DEV_IDX + 1))
    done < <(fwupdmgr get-devices --no-unreported-check 2>/dev/null \
        | awk '
            /^[^ ]/ { if (name != "") print name "|" ver "|" guid; name=$0; ver=""; guid="" }
            /Version:/ { sub(/.*Version: */, ""); ver=$0 }
            /Guid:/ { sub(/.*Guid: */, ""); if (guid != "") guid=guid","; guid=guid $0 }
            END { if (name != "") print name "|" ver "|" guid }
        ' || true)
    echo "DIAG_FWUPD_DEV_COUNT=$FW_DEV_IDX"

    # Firmware update history (recent updates)
    FW_HIST_IDX=0
    while IFS= read -r line; do
        echo "DIAG_FWUPD_HIST_${FW_HIST_IDX}=$line"
        FW_HIST_IDX=$((FW_HIST_IDX + 1))
    done < <(fwupdmgr get-history --no-unreported-check 2>/dev/null \
        | awk '
            /^[^ ]/ { if (name != "") print name "|" ver "|" date; name=$0; ver=""; date="" }
            /Version:/ { sub(/.*Version: */, ""); ver=$0 }
            /Updated:/ { sub(/.*Updated: */, ""); date=$0 }
            END { if (name != "") print name "|" ver "|" date }
        ' || true)
    echo "DIAG_FWUPD_HIST_COUNT=$FW_HIST_IDX"
else
    echo "DIAG_FWUPD_DEV_COUNT=0"
    echo "DIAG_FWUPD_HIST_COUNT=0"
    echo "  fwupdmgr not found" >&2
fi

# --- SSH user ---
echo "DIAG_SSH_USER=$(whoami)"

# --- Completion sentinel ---
echo "DIAG_COMPLETE=1"
echo "Diagnostics collection complete." >&2
