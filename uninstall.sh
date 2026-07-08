#!/bin/bash
set -euo pipefail

echo "=== Vox Uninstaller ==="

# Kill running instance
if pgrep -f "vox/dictation.py" >/dev/null 2>&1; then
    echo "Stopping running instance..."
    pkill -f "vox/dictation.py" || true
fi

echo "Removing files..."
rm -rf "$HOME/.local/share/vox"
rm -f "$HOME/.local/bin/vox"
rm -f "$HOME/.config/autostart/vox.desktop"

echo "Done. Vox has been removed."
