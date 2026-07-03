# Whisper Dictation

Push-to-talk dictation for **Linux and Windows**. Hold **Ctrl+Alt** to record, release to transcribe and paste the result wherever your cursor is.

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for local, offline speech-to-text with GPU acceleration. Inspired by [Whisper Writer](https://github.com/savbell/whisper-writer) but built as a lightweight, single-file alternative that just works with minimal setup.

## Features

- **Global push-to-talk** — works in any window (evdev on Linux, pynput on Windows)
- **Cross-platform** — one script runs on Linux/X11 and Windows
- **Auto device detection** — CUDA if an NVIDIA GPU is present, otherwise CPU (`int8`) — no config needed
- **Clipboard paste** — pastes instead of simulating keystrokes, for instant output
- **Personal dictionary** — biases Whisper toward your vocabulary and auto-corrects misheard terms (`dictionary.json`)
- **Filler word removal** — strips "um", "uh", "you know", "I mean", etc.
- **Punctuation** — uses an initial prompt to encourage proper punctuation from Whisper
- **Terminal-aware (Linux)** — detects terminal emulators and uses Ctrl+Shift+V automatically
- **Autostart** — run on login (desktop entry on Linux, Startup shortcut on Windows)

## Requirements

**All platforms:** Python 3.10+. An NVIDIA GPU with CUDA is optional — it's auto-detected, and the app falls back to CPU otherwise.

**Linux:**
- X11 (Wayland is not supported — xdotool requires X11)
- System packages: `xdotool`, `xclip`, `xprop`
- User must be in the `input` group (for global hotkey capture via evdev)

**Windows:**
- Windows 10/11 — no special permissions or groups required
- Python packages `pynput` and `pyperclip` (installed automatically)

## Installation (Linux)

```bash
# Install system dependencies
sudo apt install xdotool xclip x11-utils python3-venv

# Add yourself to the input group (required for global hotkeys)
sudo usermod -aG input $USER
# Log out and back in for group change to take effect

# Clone and install
git clone https://github.com/eonfathom/whisper-dictation.git
cd whisper-dictation
chmod +x install.sh
./install.sh
```

The installer creates a venv at `~/.local/share/whisper-dictation/venv/`, installs dependencies, creates a launcher at `~/.local/bin/whisper-dictation`, and sets up autostart on login.

## Installation (Windows)

```powershell
# Clone
git clone https://github.com/eonfathom/whisper-dictation.git
cd whisper-dictation

# Install: creates a venv, installs deps, adds GPU libraries if an NVIDIA GPU is present
.\install-windows.ps1

# ...or install and also run automatically on login:
.\install-windows.ps1 -Autostart
```

If PowerShell blocks the script, allow it for your user once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

The installer auto-detects your GPU. On an NVIDIA machine it installs the CUDA libraries (cuBLAS, cuDNN, cudart) and runs on the GPU; on an Intel-only laptop it installs the CPU build and the app uses `int8` on the CPU automatically — the same code, no config changes.

## Usage

**Linux:**
```bash
whisper-dictation          # or it starts automatically on login
```

**Windows:**
```powershell
.\run-windows.ps1          # or starts on login if installed with -Autostart
```

Then, on either platform:
1. Hold **Ctrl+Alt** — recording starts
2. Speak
3. Release **Ctrl+Alt** — audio is transcribed and pasted at your cursor

## Configuration

Configuration is via environment variables.

- Linux: set them in your shell profile or before launching.
- Windows: set them before launching, e.g. `$env:WHISPER_MODEL = "large-v3"` then `.\run-windows.ps1`.

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_LANG` | `en` | Language code (`en`, `es`, `fr`, …) or empty for auto-detect |
| `WHISPER_DEVICE` | auto | Auto-detected: `cuda` if a GPU is present, else `cpu`. Override with `cuda`/`cpu`. |
| `WHISPER_COMPUTE` | auto | `float16` on GPU, `int8` on CPU. Override with `float16`/`int8`/`float32`. |
| `WHISPER_DICT` | `dictionary.json` | Path to the personal dictionary file |

Example — force the tiny model on CPU:

```bash
# Linux
WHISPER_MODEL=tiny WHISPER_DEVICE=cpu whisper-dictation
```
```powershell
# Windows
$env:WHISPER_MODEL="tiny"; $env:WHISPER_DEVICE="cpu"; .\run-windows.ps1
```

### Model size guide

| Model | Speed | Accuracy | VRAM | Good for |
|---|---|---|---|---|
| `tiny` | Fastest | Good for clear speech | ~1 GB | Weak CPUs |
| `base` | Fast | Good balance (recommended) | ~1 GB | Default; CPU laptops |
| `small` | Moderate | Better accuracy | ~2 GB | CPU laptops wanting accuracy |
| `medium` | Slow | High accuracy | ~5 GB | Mid-range GPUs |
| `large-v3` | Slowest | Best accuracy | ~10 GB | Strong GPUs (12 GB+) |

On a CPU-only laptop, stick to `base` or `small`. On a GPU with 10 GB+ VRAM, `large-v3` gives the best accuracy at no real speed cost.

## Personal dictionary

Names, jargon, and product names that Whisper mishears go in `dictionary.json`:

```json
{
  "hotwords": ["Eon", "Obsidian", "Claude Code"],
  "corrections": { "clod code": "Claude Code", "obsidion": "Obsidian" }
}
```

- **`hotwords`** bias Whisper *during* recognition so it's more likely to hear your terms correctly.
- **`corrections`** are a whole-word, case-insensitive find/replace applied *after* transcription — a safety net for anything it still gets wrong.

Edit the file and restart the app to reload.

## Troubleshooting

### No text appears (Linux)

Check your default audio input — if you have multiple devices, the wrong one may be selected:
```bash
wpctl status                    # list audio sources
wpctl set-default <source-id>   # set the correct mic
```

### "No keyboards found" (Linux)

Add yourself to the `input` group, then log out and back in:
```bash
sudo usermod -aG input $USER
```

### Nothing happens / CUDA errors (Windows)

- **First run is slow** — it downloads the model (~140 MB for `base`) and warms up CUDA.
- **`cublas64_12.dll ... cannot be loaded`** — the CUDA libraries aren't installed. Run `.\install-windows.ps1` again, or `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12` into the venv. The app adds these to the DLL search path automatically at startup.
- **Pastes into the wrong place** — the text is pasted into whatever window has focus when you release Ctrl+Alt. Keep your cursor where you want the text.
- **No GPU?** That's fine — it runs on CPU automatically. Use `WHISPER_MODEL=base` or `small` for reasonable speed.

### Slow transcription

- Use a smaller model: `WHISPER_MODEL=tiny` (or `base`/`small` on CPU).
- Confirm the device: the startup banner prints `Device: cuda` or `Device: cpu`.

## Uninstall

**Linux:** `./uninstall.sh`

**Windows:** delete the repo folder and remove the `Whisper Dictation` shortcut from your Startup folder (`shell:startup`) if you enabled autostart.

## How it works

1. Keyboard events are captured globally — **evdev** on Linux, **pynput** on Windows (no window focus required)
2. When Ctrl+Alt are both held, audio recording starts via **sounddevice**
3. On release, the audio buffer is passed to **faster-whisper** for transcription (CUDA if available, else CPU)
4. The transcript is cleaned: filler words stripped, then personal-dictionary corrections applied
5. The result is copied to the clipboard and pasted — **xdotool** on Linux (Ctrl+V, or Ctrl+Shift+V in terminals), **pynput** on Windows (Ctrl+V)

## Comparison with Whisper Writer

This project was inspired by [Whisper Writer](https://github.com/savbell/whisper-writer). Here's how they compare:

| | Whisper Dictation | Whisper Writer |
|---|---|---|
| **Architecture** | Single file | ~12 source files, full PyQt5 GUI |
| **Dependencies** | A handful of Python packages | 69 Python packages |
| **Text output** | Clipboard paste (instant) | Keystroke simulation (slow, known bugs with duplicated/missing chars) |
| **Post-processing** | Filler removal, punctuation cleanup, personal dictionary | Trailing space/period removal only |
| **Recording modes** | Hold-to-record | Continuous, VAD, press-to-toggle, hold-to-record |
| **Hotkey** | Ctrl+Alt (evdev on Linux, pynput on Windows) | Configurable chord (evdev or pynput) |
| **Configuration** | Environment variables + `dictionary.json` | YAML config + settings GUI |
| **STT backend** | faster-whisper (local) | faster-whisper (local) + OpenAI API |
| **Platforms** | Linux/X11 + Windows | Linux, macOS, Windows |
| **GPU** | CUDA with auto-fallback to CPU | CUDA with auto-fallback to CPU |

**Why this project exists:** Whisper Dictation trades flexibility for simplicity and reliability — clipboard paste is faster and more reliable than keystroke simulation, the single-file design has no complex dependency chains, and filler-word removal plus a personal dictionary produce cleaner output. If you want something that just works on Linux or Windows with minimal setup, this is for you.

## License

MIT
