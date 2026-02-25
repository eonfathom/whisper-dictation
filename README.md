# Whisper Dictation

Push-to-talk dictation for Linux. Hold **Ctrl+Alt** to record, release to transcribe and type the result wherever your cursor is.

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for local, offline speech-to-text with GPU acceleration. Inspired by [Whisper Writer](https://github.com/savbell/whisper-writer) but built as a lightweight, single-file alternative for Linux/X11.

## Features

- **Global push-to-talk** — works in any window (uses evdev for hotkey capture)
- **GPU-accelerated** — runs on CUDA via CTranslate2 for fast transcription
- **Clipboard paste** — uses clipboard + paste instead of keystroke simulation for instant output
- **Terminal-aware** — detects terminal emulators and uses Ctrl+Shift+V automatically
- **Filler word removal** — strips "um", "uh", "you know", "I mean", etc.
- **Punctuation** — uses initial prompt to encourage proper punctuation from Whisper
- **Autostart** — installs a desktop entry to run on login

## Requirements

- **Linux with X11** (Wayland is not supported — xdotool requires X11)
- **Python 3.10+**
- **NVIDIA GPU with CUDA** (optional — falls back to CPU)
- System packages: `xdotool`, `xclip`, `xprop`
- User must be in the `input` group (for global hotkey capture via evdev)

## Installation

```bash
# Install system dependencies
sudo apt install xdotool xclip x11-utils python3-venv

# Add yourself to the input group (required for global hotkeys)
sudo usermod -aG input $USER
# Log out and back in for group change to take effect

# Clone and install
git clone https://github.com/YOUR_USERNAME/whisper-dictation.git
cd whisper-dictation
chmod +x install.sh
./install.sh
```

The installer will:
1. Create a virtual environment at `~/.local/share/whisper-dictation/venv/`
2. Install Python dependencies (faster-whisper, sounddevice, numpy, evdev)
3. Create a launcher script at `~/.local/bin/whisper-dictation`
4. Set up autostart on login

## Usage

```bash
# Start manually
whisper-dictation

# Or it starts automatically on login
```

1. Hold **Ctrl+Alt** — recording starts
2. Speak
3. Release **Ctrl+Alt** — audio is transcribed and typed at your cursor

## Configuration

Configuration is done via environment variables. Set them in your shell profile or before launching:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_LANG` | `en` | Language code (`en`, `es`, `fr`, etc.) or empty for auto-detect |
| `WHISPER_DEVICE` | `cuda` | `cuda` for GPU, `cpu` for CPU-only |
| `WHISPER_COMPUTE` | `float16` | Compute type: `float16`, `int8`, `float32` |

Example — use the tiny model on CPU:

```bash
WHISPER_MODEL=tiny WHISPER_DEVICE=cpu WHISPER_COMPUTE=int8 whisper-dictation
```

### Model size guide

| Model | Speed | Accuracy | VRAM |
|---|---|---|---|
| `tiny` | Fastest | Good for clear speech | ~1 GB |
| `base` | Fast | Good balance (recommended) | ~1 GB |
| `small` | Moderate | Better accuracy | ~2 GB |
| `medium` | Slow | High accuracy | ~5 GB |
| `large-v3` | Slowest | Best accuracy | ~10 GB |

## Troubleshooting

### No text appears

- **Check your default audio input.** If you have multiple audio devices (e.g., a capture card), the wrong one may be selected:
  ```bash
  # List audio sources
  wpctl status
  # Set the correct source (replace ID with your mic's ID)
  wpctl set-default <source-id>
  ```

### "No keyboards found"

- Add yourself to the `input` group:
  ```bash
  sudo usermod -aG input $USER
  ```
- Log out and back in for it to take effect.

### Slow transcription

- Use a smaller model: `WHISPER_MODEL=tiny`
- Make sure CUDA is being used: check for `device: cuda` in the startup output
- The first transcription after launch may be slower as CUDA warms up

### Doesn't work in terminals

The script detects common terminal emulators and uses Ctrl+Shift+V for pasting. If your terminal isn't recognized, add its WM_CLASS to the `TERMINAL_CLASSES` set in `dictation.py`.

Find your terminal's WM_CLASS:
```bash
xprop WM_CLASS  # then click your terminal window
```

## Uninstall

```bash
chmod +x uninstall.sh
./uninstall.sh
```

## How it works

1. **evdev** captures raw keyboard events globally (no window focus required)
2. When Ctrl+Alt are both held, audio recording starts via **sounddevice**
3. On release, the audio buffer is passed to **faster-whisper** for transcription
4. Filler words are stripped and the result is copied to the clipboard
5. **xdotool** simulates a paste keystroke (Ctrl+V or Ctrl+Shift+V for terminals)

## Comparison with Whisper Writer

This project was inspired by [Whisper Writer](https://github.com/savbell/whisper-writer). Here's how they compare:

| | Whisper Dictation | Whisper Writer |
|---|---|---|
| **Architecture** | Single file (278 lines) | ~12 source files, full PyQt5 GUI |
| **Dependencies** | 4 Python packages + 3 system tools | 69 Python packages |
| **Text output** | Clipboard paste (instant) | Keystroke simulation (slow, known bugs with duplicated/missing chars) |
| **Post-processing** | Filler word removal, punctuation cleanup | Trailing space/period removal only |
| **Recording modes** | Hold-to-record | Continuous, VAD, press-to-toggle, hold-to-record |
| **Hotkey** | Ctrl+Alt (evdev) | Configurable chord (evdev or pynput) |
| **Configuration** | 4 environment variables | YAML config + settings GUI |
| **STT backend** | faster-whisper (local) | faster-whisper (local) + OpenAI API |
| **Platforms** | Linux/X11 only | Linux, macOS, Windows |
| **GPU** | CUDA | CUDA with auto-fallback to CPU |

**Why this project exists:** Whisper Writer is a full-featured desktop application that supports multiple platforms and recording modes. Whisper Dictation trades that flexibility for simplicity and reliability — clipboard paste is faster and more reliable than keystroke simulation, the single-file design has no complex dependency chains, and the filler word removal produces cleaner output. If you're on Linux/X11 and want something that just works with minimal setup, this is for you.

## License

MIT
