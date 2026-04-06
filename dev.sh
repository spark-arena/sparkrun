#!/bin/bash
# Dev setup script for sparkrun
#
# Usage (must be sourced so the venv activates in your shell):
#   source dev.sh
#
# Quick start:
#   git clone https://github.com/scitrera/sparkrun.git -b develop
#   cd sparkrun
#   source dev.sh
#   sparkrun run --help

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# Check for uv
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. Install it via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    return 1 2>/dev/null || exit 1
fi

# Sync the environment (creates venv and installs project + dev dependencies)
echo "Syncing environment with uv ..."
if ! uv sync; then
    echo "Error: failed to sync environment"
    return 1 2>/dev/null || exit 1
fi

# Activate the venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install pre-commit hooks using uv run
echo "Installing pre-commit hooks ..."
if ! uv run pre-commit install; then
    echo "Warning: failed to install pre-commit hooks"
fi

echo ""
echo "Done! sparkrun is ready to use in this shell."
echo "  sparkrun --help"
