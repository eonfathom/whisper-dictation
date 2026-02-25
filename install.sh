#!/bin/bash
set -euo pipefail

# Whisper Dictation installer
# Sets up venv, installs dependencies, creates launcher and autostart entry.

INSTALL_DIR="$HOME/.local/share/whisper-dictation"
LAUNCHER="$HOME/.local/bin/whisper-dictation"
AUTOSTART="$HOME/.config/autostart/whisper-dictation.desktop"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Whisper Dictation Installer ==="
echo ""

# --- Check system dependencies ---
echo "Checking system dependencies..."
missing=()
for cmd in xdotool xclip xprop python3; do
    if ! command -v "$cmd" &>/dev/null; then
        missing+=("$cmd")
    fi
done

if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing system packages: ${missing[*]}"
    echo "Install them with:"
    echo "  sudo apt install ${missing[*]}"
    exit 1
fi
echo "  All system dependencies found."

# --- Check input group ---
if ! groups | grep -qw input; then
    echo ""
    echo "WARNING: You are not in the 'input' group."
    echo "  Global hotkeys require it. Run:"
    echo "    sudo usermod -aG input \$USER"
    echo "  Then log out and log back in."
    echo ""
fi

# --- Create install directory ---
echo "Installing to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
cp "$SCRIPT_DIR/dictation.py" "$INSTALL_DIR/dictation.py"

# --- Create virtual environment ---
if [ ! -d "$INSTALL_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$INSTALL_DIR/venv"
else
    echo "Virtual environment already exists."
fi

echo "Installing Python dependencies..."
"$INSTALL_DIR/venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install --quiet \
    faster-whisper \
    sounddevice \
    numpy \
    evdev

# --- Detect CUDA ---
SITE="$INSTALL_DIR/venv/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
CUBLAS_DIR="$SITE/nvidia/cublas/lib"
CUDNN_DIR="$SITE/nvidia/cudnn/lib"

if [ -d "$CUBLAS_DIR" ] && [ -d "$CUDNN_DIR" ]; then
    LD_EXTRA="$CUBLAS_DIR:$CUDNN_DIR:"
    echo "  CUDA libraries found."
else
    LD_EXTRA=""
    echo "  No CUDA libraries found (will use CPU)."
    echo "  For GPU support: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
fi

# --- Create launcher script ---
echo "Creating launcher at $LAUNCHER..."
mkdir -p "$(dirname "$LAUNCHER")"
cat > "$LAUNCHER" <<LAUNCHER_EOF
#!/bin/bash
VENV=$INSTALL_DIR/venv
SITE="\$VENV/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
export LD_LIBRARY_PATH="\$SITE/nvidia/cublas/lib:\$SITE/nvidia/cudnn/lib:\${LD_LIBRARY_PATH:-}"
exec "\$VENV/bin/python3" $INSTALL_DIR/dictation.py "\$@"
LAUNCHER_EOF
chmod +x "$LAUNCHER"

# --- Create autostart desktop entry ---
echo "Creating autostart entry..."
mkdir -p "$(dirname "$AUTOSTART")"
cat > "$AUTOSTART" <<DESKTOP_EOF
[Desktop Entry]
Type=Application
Name=Whisper Dictation
Comment=Hold Ctrl+Alt to dictate with Whisper
Exec=$LAUNCHER
Hidden=false
X-GNOME-Autostart-enabled=true
DESKTOP_EOF

echo ""
echo "=== Installation complete ==="
echo ""
echo "  Start now:   $LAUNCHER"
echo "  Autostart:   enabled (runs on login)"
echo "  Hotkey:      Hold Ctrl+Alt to record, release to type"
echo ""
echo "  Configuration (environment variables):"
echo "    WHISPER_MODEL=base     (tiny, base, small, medium, large-v3)"
echo "    WHISPER_LANG=en        (language code or empty for auto-detect)"
echo "    WHISPER_DEVICE=cuda    (cuda or cpu)"
echo "    WHISPER_COMPUTE=float16 (float16, int8, etc.)"
echo ""
