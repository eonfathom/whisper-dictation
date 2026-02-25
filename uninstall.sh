#!/bin/bash
set -euo pipefail

echo "=== Whisper Dictation Uninstaller ==="

# Kill running instance
if pgrep -f "whisper-dictation/dictation.py" >/dev/null 2>&1; then
    echo "Stopping running instance..."
    pkill -f "whisper-dictation/dictation.py" || true
fi

echo "Removing files..."
rm -rf "$HOME/.local/share/whisper-dictation"
rm -f "$HOME/.local/bin/whisper-dictation"
rm -f "$HOME/.config/autostart/whisper-dictation.desktop"

echo "Done. Whisper Dictation has been removed."
