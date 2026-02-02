#!/bin/bash
# FlashAI Portable Launcher for Unix/Linux/macOS
#
# This script launches FlashAI from a flash drive, automatically
# detecting the installation location and setting up the environment.

set -e

# Determine script location (flash drive root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FLASHAI_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     FlashAI Launcher                         ║"
echo "║        Portable AI System with Energy-Based Reasoning        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Installation: $FLASHAI_ROOT"
echo ""

# Set environment variables
export FLASHAI_BASE="$FLASHAI_ROOT"
export FLASHAI_PORTABLE=1
export FLASHAI_DATA="$FLASHAI_ROOT/data"
export FLASHAI_CONFIG="$FLASHAI_ROOT/config"

# Check for portable Python
PORTABLE_PYTHON="$FLASHAI_ROOT/python/bin/python3"
if [ -f "$PORTABLE_PYTHON" ]; then
    PYTHON="$PORTABLE_PYTHON"
    echo "Using portable Python: $PYTHON"
else
    # Use system Python
    if command -v python3 &> /dev/null; then
        PYTHON="python3"
    elif command -v python &> /dev/null; then
        PYTHON="python"
    else
        echo "Error: Python not found. Please install Python 3.9+ or bundle portable Python."
        exit 1
    fi
    echo "Using system Python: $PYTHON"
fi

# Check Python version
PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PY_VERSION"

# Check if FlashAI is installed
if ! $PYTHON -c "import flashai" 2>/dev/null; then
    echo ""
    echo "FlashAI not installed. Installing..."
    cd "$FLASHAI_ROOT"
    $PYTHON -m pip install -e . --quiet
    echo "Installation complete."
fi

echo ""
echo "Starting FlashAI..."
echo ""

# Parse arguments
ACTION="${1:-serve}"
shift 2>/dev/null || true

case "$ACTION" in
    serve)
        $PYTHON -m flashai.cli serve --base-path "$FLASHAI_ROOT" "$@"
        ;;
    reason)
        $PYTHON -m flashai.cli reason --base-path "$FLASHAI_ROOT" "$@"
        ;;
    reset)
        $PYTHON -m flashai.cli reset --base-path "$FLASHAI_ROOT" "$@"
        ;;
    restore)
        $PYTHON -m flashai.cli restore --base-path "$FLASHAI_ROOT" "$@"
        ;;
    status)
        $PYTHON -m flashai.cli status --base-path "$FLASHAI_ROOT" "$@"
        ;;
    init)
        $PYTHON -m flashai.cli init --base-path "$FLASHAI_ROOT" "$@"
        ;;
    help|--help|-h)
        echo "Usage: launch.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  serve     Start the FlashAI API server (default)"
        echo "  reason    Perform reasoning on a query"
        echo "  reset     Save and reset to initial state"
        echo "  restore   Restore from an export file"
        echo "  status    Show system status"
        echo "  init      Initialize FlashAI"
        echo "  help      Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./launch.sh serve --port 8080"
        echo "  ./launch.sh reason 'What is 2+2?'"
        echo "  ./launch.sh reset --name backup"
        ;;
    *)
        # Pass through to CLI
        $PYTHON -m flashai.cli "$ACTION" --base-path "$FLASHAI_ROOT" "$@"
        ;;
esac
