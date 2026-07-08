#!/usr/bin/env python3
"""
Vox - Hold Ctrl+Win, speak, release to type.

Cross-platform push-to-talk dictation (Linux/X11 + Windows). Hold Ctrl+Win to
record, release to stop, transcribe, and paste the result wherever your cursor is.
The chord is configurable via WHISPER_HOTKEY (e.g. "ctrl+alt", "ctrl+shift").

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
# MODEL_SIZE is resolved below, after the device is detected (it is device-aware).
LANGUAGE = os.environ.get("WHISPER_LANG", "en")
SAMPLE_RATE = 16000
CHANNELS = 1

# Push-to-talk chord: hold all these modifiers together to record. Override via
# WHISPER_HOTKEY, e.g. "ctrl+alt" or "ctrl+shift". Supported modifiers:
# ctrl, alt, win, shift  ("win" is the Windows/Super key).
HOTKEY_MODS = tuple(
    m for m in (t.strip().lower() for t in
                os.environ.get("WHISPER_HOTKEY", "ctrl+win").split("+")) if m
) or ("ctrl", "win")
HOTKEY_LABEL = "+".join(m.capitalize() for m in HOTKEY_MODS)

# Trailing-audio robustness (seconds, override via env): capture a short grace
# tail after release so the last word isn't clipped, and pad the buffer with
# silence so Whisper reliably finalizes the final segment.
RELEASE_TAIL_SEC = float(os.environ.get("WHISPER_RELEASE_TAIL", "0.2"))
TRAILING_PAD_SEC = float(os.environ.get("WHISPER_PAD", "0.4"))

# Decoding beam width. Higher = more accurate, a bit slower. 5 is Whisper's
# standard default; drop to 1 on a slow CPU if latency matters more than accuracy.
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM", "5"))

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

# Model default is device-aware, so one zero-config setup works everywhere:
# large-v3 on a CUDA GPU (best accuracy, and the GPU keeps it real-time), but
# base on a CPU-only machine (large-v3 is far too slow on CPU for live
# dictation). Override anytime with WHISPER_MODEL.
MODEL_SIZE = os.environ.get("WHISPER_MODEL") or (
    "large-v3" if DEVICE == "cuda" else "base"
)


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
        """Paste the text (Ctrl+V), then leave the clean text on the clipboard.

        The paste uses a trailing space so back-to-back dictations stay
        separated; afterwards the clipboard is reset to the clean transcription
        (no trailing space) so you can re-paste it cleanly - like Wispr Flow.
        """
        if not text.strip():
            return
        pyperclip.copy(text + " ")
        time.sleep(0.02)  # let the clipboard settle before pasting
        with _kbd_controller.pressed(PKey.ctrl):
            _kbd_controller.press("v")
            _kbd_controller.release("v")
        time.sleep(0.05)  # let the target app consume the paste first
        pyperclip.copy(text)  # leave the clean transcription on the clipboard

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

    # Grace tail: keep capturing for a beat after release so trailing words
    # aren't clipped by release timing or the final audio block being dropped
    # when the stream stops. The recording callback keeps appending while we wait.
    local_stream = stream
    if RELEASE_TAIL_SEC > 0:
        time.sleep(RELEASE_TAIL_SEC)
    if local_stream is not None:
        local_stream.stop()
        local_stream.close()
        if stream is local_stream:
            stream = None
    recording = False

    frames = list(audio_frames)
    if not frames:
        print(">> No audio captured.", flush=True)
        return

    audio = np.concatenate(frames, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE
    # Pad with trailing silence so Whisper reliably finalizes the last segment.
    if TRAILING_PAD_SEC > 0:
        audio = np.concatenate(
            [audio, np.zeros(int(SAMPLE_RATE * TRAILING_PAD_SEC), dtype=audio.dtype)]
        )
    print(f">> Transcribing {duration:.1f}s of audio...", flush=True)

    kwargs = {}
    if LANGUAGE:
        kwargs["language"] = LANGUAGE
    if HOTWORDS and supports_hotwords:
        kwargs["hotwords"] = " ".join(HOTWORDS)
    segments, info = model.transcribe(
        audio, beam_size=BEAM_SIZE, vad_filter=True,
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
    _held = set()  # modifier tokens currently held, e.g. {"ctrl", "win"}

    _WIN_TOKENS = {
        "ctrl": (PKey.ctrl_l, PKey.ctrl_r, PKey.ctrl),
        "alt": (PKey.alt_l, PKey.alt_r, PKey.alt_gr),
        "win": (PKey.cmd, PKey.cmd_l, PKey.cmd_r),  # Windows/Super key
        "shift": (PKey.shift_l, PKey.shift_r, PKey.shift),
    }
    _WIN_KEY_TO_TOKEN = {
        k: tok for tok, keys in _WIN_TOKENS.items() for k in keys
    }

    def _token(key):
        return _WIN_KEY_TO_TOKEN.get(key)

    def _combo_held():
        return all(m in _held for m in HOTKEY_MODS)

    def monitor_keys():
        """Monitor keyboard globally via pynput (works in any window)."""
        def on_press(key):
            tok = _token(key)
            if tok:
                _held.add(tok)
                _set_recording(_combo_held())

        def on_release(key):
            tok = _token(key)
            if tok:
                _held.discard(tok)
                _set_recording(_combo_held())

        with pynput_keyboard.Listener(
            on_press=on_press, on_release=on_release
        ) as listener:
            listener.join()

else:
    _LINUX_TOKENS = {
        "ctrl": {ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL},
        "alt": {ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT},
        "win": {ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA},  # Super key
        "shift": {ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT},
    }
    keys_held = set()

    def _combo_held():
        return all(
            bool(keys_held & _LINUX_TOKENS.get(m, set())) for m in HOTKEY_MODS
        )

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
                        _set_recording(_combo_held())
                except OSError:
                    pass


def main():
    load_model()

    print("", flush=True)
    print("=== Vox ready ===", flush=True)
    print(f"  Model:  {MODEL_SIZE}", flush=True)
    print(f"  Device: {DEVICE} ({COMPUTE_TYPE})", flush=True)
    if HOTWORDS or CORRECTIONS:
        print(f"  Dictionary: {len(HOTWORDS)} hotwords, "
              f"{len(CORRECTIONS)} corrections", flush=True)
    print("", flush=True)

    if IS_WINDOWS:
        print(f"  Hold {HOTKEY_LABEL} to record, release to transcribe & type.", flush=True)
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
        print(f"  Hold {HOTKEY_LABEL} to record, release to transcribe & type.", flush=True)
        print("  Works in any window. Ctrl+C to quit.", flush=True)
        print("", flush=True)
        monitor_keys(keyboards)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
