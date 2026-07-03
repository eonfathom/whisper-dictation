#!/usr/bin/env python3
"""
Whisper Dictation - Hold Ctrl+Alt, speak, release to type.

Cross-platform push-to-talk dictation (Linux/X11 + Windows). Hold Ctrl+Alt to
record, release to stop, transcribe, and paste the result wherever your cursor is.

  Linux   : evdev for global hotkeys, xclip/xdotool for clipboard paste.
            User must be in the 'input' group.
  Windows : pynput for global hotkeys + paste, pyperclip for clipboard.
            No special permissions required.

Speech-to-text is local and offline via faster-whisper. The device is
auto-detected: CUDA if a GPU is available, otherwise CPU (int8).

Requires: faster-whisper, sounddevice, numpy
  Linux:   evdev            (+ system: xdotool, xclip, xprop)
  Windows: pynput, pyperclip
"""

import inspect
import json
import os
import re
import sys
import threading
import time

import numpy as np
import sounddevice as sd

IS_WINDOWS = sys.platform == "win32"

# --- Platform-specific imports ---
if IS_WINDOWS:
    import pyperclip
    from pynput import keyboard as pynput_keyboard
    from pynput.keyboard import Controller as KeyController, Key as PKey
else:
    import select
    import subprocess
    import evdev
    from evdev import ecodes

# --- Configuration (override via environment variables) ---
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
LANGUAGE = os.environ.get("WHISPER_LANG", "en")
SAMPLE_RATE = 16000
CHANNELS = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.environ.get(
    "WHISPER_DICT", os.path.join(SCRIPT_DIR, "dictionary.json")
)

# Prompt that encourages proper punctuation from whisper
INITIAL_PROMPT = "Hello, how are you? I'm doing well. Let's discuss the project."

# Filler words/phrases to strip (order matters - longer phrases first)
FILLERS = [
    "you know what I mean", "you know what i mean",
    "I mean,", "i mean,", "I mean", "i mean",
    "you know,", "You know,", "you know", "You know",
    ", like,", ", Like,",
    "like,", "Like,",
    ", um,", ", Um,", ", uh,", ", Uh,",
    "um,", "Um,", "uh,", "Uh,",
    " um ", " Um ", " uh ", " Uh ",
    " um.", " uh.",
]


def _add_cuda_dll_dirs():
    """Windows: make the bundled cuBLAS/cuDNN/cudart DLLs discoverable.

    CTranslate2's runtime CUDA loader uses the legacy DLL search order, which
    consults PATH but ignores os.add_dll_directory() dirs - so we prepend the
    nvidia-*-cu12 package bin directories to PATH. This is the Windows analogue
    of the LD_LIBRARY_PATH export the Linux launcher does. Must run before
    ctranslate2 first loads a CUDA kernel, so it is called at import (below),
    ahead of detect_device(). No-op on Linux or CPU-only machines where the
    nvidia packages aren't installed.
    """
    if not IS_WINDOWS:
        return
    import site
    roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(user_site)
    dirs = []
    for base in roots:
        nvidia = os.path.join(base, "nvidia")
        if not os.path.isdir(nvidia):
            continue
        for sub in sorted(os.listdir(nvidia)):
            bindir = os.path.join(nvidia, sub, "bin")
            if os.path.isdir(bindir) and bindir not in dirs:
                dirs.append(bindir)
                try:
                    os.add_dll_directory(bindir)
                except (OSError, AttributeError):
                    pass
    if dirs:
        os.environ["PATH"] = (
            os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
        )


_add_cuda_dll_dirs()  # before ctranslate2 is imported in detect_device()


def detect_device():
    """Pick (device, compute_type): honor env override, else CUDA if available, else CPU.

    Set WHISPER_DEVICE / WHISPER_COMPUTE to override. Auto-detection means the
    same code runs on a CUDA desktop and an Intel laptop with no config changes.
    """
    dev = os.environ.get("WHISPER_DEVICE")
    compute = os.environ.get("WHISPER_COMPUTE")
    if dev:
        if not compute:
            compute = "float16" if dev == "cuda" else "int8"
        return dev, compute
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", compute or "float16"
    except Exception:
        pass
    return "cpu", compute or "int8"


DEVICE, COMPUTE_TYPE = detect_device()


# --- Personal dictionary ------------------------------------------------------
def load_dictionary():
    """Load hotwords + corrections from dictionary.json (if present).

    hotwords    : terms fed to Whisper to bias recognition toward your vocabulary
                  (names, jargon, product names) so it hears them correctly.
    corrections : {misheard: correct} applied after transcription as a safety net
                  for words Whisper still gets wrong.
    """
    hotwords, corrections = [], {}
    try:
        with open(DICT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        hotwords = data.get("hotwords") or []
        corrections = data.get("corrections") or {}
    except FileNotFoundError:
        pass
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: could not read dictionary {DICT_PATH}: {e}", flush=True)
    return hotwords, corrections


HOTWORDS, CORRECTIONS = load_dictionary()
_CORRECTION_RES = [
    (re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE), right)
    for wrong, right in CORRECTIONS.items()
]


def apply_corrections(text):
    """Replace known misrecognitions (whole-word, case-insensitive)."""
    for pattern, right in _CORRECTION_RES:
        text = pattern.sub(right, text)
    return text


def clean_text(text):
    """Remove filler words and clean up spacing/punctuation."""
    for filler in FILLERS:
        text = text.replace(filler, " ")
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.!?])\s{2,}", r"\1 ", text)
    return text.strip()


def post_process(text):
    """Transform a raw transcript into final text via an ordered pipeline.

    Stage order matters. New stages (e.g. a local-LLM formatting pass, or
    app-aware context) slot in at the marked point below.
    """
    text = clean_text(text)          # 1. strip fillers, tidy spacing
    text = apply_corrections(text)   # 2. personal-dictionary corrections
    # --- Phase 2 insertion point -------------------------------------------
    # text = llm_format(text, context=active_app())   # formatting + commands
    # -----------------------------------------------------------------------
    return text


# --- Global state ---
recording = False
audio_frames = []
stream = None
model = None
supports_hotwords = False
lock = threading.Lock()
target_window = None


# --- Platform I/O: active window + text output --------------------------------
if IS_WINDOWS:
    _kbd_controller = KeyController()

    def get_active_window():
        """Windows pastes into the focused control; no window handle needed."""
        return None

    def type_text(text, window=None):
        """Copy to clipboard and paste (Ctrl+V) into the foreground window."""
        if not text.strip():
            return
        pyperclip.copy(text + " ")
        time.sleep(0.02)  # let the clipboard settle before pasting
        with _kbd_controller.pressed(PKey.ctrl):
            _kbd_controller.press("v")
            _kbd_controller.release("v")

else:
    TERMINAL_CLASSES = {
        "gnome-terminal", "gnome-terminal-server", "kitty", "alacritty",
        "konsole", "xterm", "urxvt", "st-256color", "terminator",
        "tilix", "xfce4-terminal", "mate-terminal", "foot",
    }

    def get_active_window():
        """Get the currently focused window ID (X11)."""
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, check=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    def _is_terminal(window_id):
        """Check if a window is a terminal emulator (needs Ctrl+Shift+V)."""
        if not window_id:
            return False
        try:
            result = subprocess.run(
                ["xprop", "-id", window_id, "WM_CLASS"],
                capture_output=True, text=True, check=True,
            )
            for cls in TERMINAL_CLASSES:
                if cls.lower() in result.stdout.lower():
                    return True
        except Exception:
            pass
        return False

    def type_text(text, window=None):
        """Paste text via clipboard into the target window (X11)."""
        if not text.strip():
            return
        subprocess.run(
            ["xclip", "-selection", "clipboard"],
            input=(text + " ").encode(), check=False,
        )
        if window:
            subprocess.run(
                ["xdotool", "windowfocus", "--sync", window], check=False,
            )
        paste_key = "ctrl+shift+v" if _is_terminal(window) else "ctrl+v"
        subprocess.run(
            ["xdotool", "key", "--clearmodifiers", paste_key], check=False,
        )


# --- Model + recording --------------------------------------------------------
def load_model():
    """Load the Whisper model."""
    global model, supports_hotwords
    from faster_whisper import WhisperModel

    print(
        f"Loading model: {MODEL_SIZE} (device={DEVICE}, compute={COMPUTE_TYPE})",
        flush=True,
    )
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    try:
        supports_hotwords = "hotwords" in inspect.signature(
            model.transcribe
        ).parameters
    except (ValueError, TypeError):
        supports_hotwords = False
    print("Model loaded.", flush=True)


def start_recording():
    """Start recording audio from the default microphone."""
    global recording, audio_frames, stream, target_window

    audio_frames = []
    target_window = get_active_window()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}", file=sys.stderr, flush=True)
        audio_frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
    )
    stream.start()
    recording = True
    print(">> RECORDING - speak now...", flush=True)


def stop_and_transcribe():
    """Stop recording, transcribe, and type the result."""
    global recording, stream

    if stream is not None:
        stream.stop()
        stream.close()
        stream = None
    recording = False

    if not audio_frames:
        print(">> No audio captured.", flush=True)
        return

    audio = np.concatenate(audio_frames, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE
    print(f">> Transcribing {duration:.1f}s of audio...", flush=True)

    kwargs = {}
    if LANGUAGE:
        kwargs["language"] = LANGUAGE
    if HOTWORDS and supports_hotwords:
        kwargs["hotwords"] = " ".join(HOTWORDS)
    segments, info = model.transcribe(
        audio, beam_size=1, vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=INITIAL_PROMPT, **kwargs,
    )
    text = " ".join(seg.text for seg in segments).strip()
    text = post_process(text)

    if text:
        print(f">> Result: {text}", flush=True)
        type_text(text, target_window)
    else:
        print(">> No speech detected.", flush=True)


def _set_recording(active):
    """Thread-safe transition of recording state, shared by both backends."""
    global recording
    with lock:
        if active and not recording:
            start_recording()
        elif not active and recording:
            recording = False
            threading.Thread(target=stop_and_transcribe, daemon=True).start()


# --- Hotkey backends ----------------------------------------------------------
if IS_WINDOWS:
    _held = set()  # tokens currently held: {"ctrl", "alt"}

    def _token(key):
        if key in (PKey.ctrl_l, PKey.ctrl_r, PKey.ctrl):
            return "ctrl"
        if key in (PKey.alt_l, PKey.alt_r, PKey.alt_gr):
            return "alt"
        return None

    def _both_held():
        return "ctrl" in _held and "alt" in _held

    def monitor_keys():
        """Monitor keyboard globally via pynput (works in any window)."""
        def on_press(key):
            tok = _token(key)
            if tok:
                _held.add(tok)
                _set_recording(_both_held())

        def on_release(key):
            tok = _token(key)
            if tok:
                _held.discard(tok)
                _set_recording(_both_held())

        with pynput_keyboard.Listener(
            on_press=on_press, on_release=on_release
        ) as listener:
            listener.join()

else:
    CTRL_CODES = {ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL}
    ALT_CODES = {ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT}
    keys_held = set()

    def _both_held():
        return bool(keys_held & CTRL_CODES) and bool(keys_held & ALT_CODES)

    def find_keyboards():
        """Find keyboard input devices (evdev)."""
        keyboards = []
        for path in evdev.list_devices():
            dev = evdev.InputDevice(path)
            caps = dev.capabilities()
            if ecodes.EV_KEY in caps:
                key_codes = caps[ecodes.EV_KEY]
                if ecodes.KEY_A in key_codes and ecodes.KEY_Z in key_codes:
                    keyboards.append(dev)
                    print(f"  Found keyboard: {dev.name} ({dev.path})", flush=True)
        return keyboards

    def monitor_keys(keyboards):
        """Monitor keyboard events using evdev (works globally)."""
        devices = {kb.fd: kb for kb in keyboards}
        while True:
            r, _, _ = select.select(devices.values(), [], [])
            for dev in r:
                try:
                    for event in dev.read():
                        if event.type != ecodes.EV_KEY:
                            continue
                        if event.value == 1:
                            keys_held.add(event.code)
                        elif event.value == 0:
                            keys_held.discard(event.code)
                        else:
                            continue  # ignore key-repeat
                        _set_recording(_both_held())
                except OSError:
                    pass


def main():
    load_model()

    print("", flush=True)
    print("=== Whisper Dictation ready ===", flush=True)
    print(f"  Model:  {MODEL_SIZE}", flush=True)
    print(f"  Device: {DEVICE} ({COMPUTE_TYPE})", flush=True)
    if HOTWORDS or CORRECTIONS:
        print(f"  Dictionary: {len(HOTWORDS)} hotwords, "
              f"{len(CORRECTIONS)} corrections", flush=True)
    print("", flush=True)

    if IS_WINDOWS:
        print("  Hold Ctrl+Alt to record, release to transcribe & type.", flush=True)
        print("  Works in any window. Ctrl+C in this window to quit.", flush=True)
        print("", flush=True)
        monitor_keys()
    else:
        keyboards = find_keyboards()
        if not keyboards:
            print("ERROR: No keyboards found. Are you in the 'input' group?", flush=True)
            print("  Run: sudo usermod -aG input $USER", flush=True)
            print("  Then log out and log back in.", flush=True)
            sys.exit(1)
        print("", flush=True)
        print("  Hold Ctrl+Alt to record, release to transcribe & type.", flush=True)
        print("  Works in any window. Ctrl+C to quit.", flush=True)
        print("", flush=True)
        monitor_keys(keyboards)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
