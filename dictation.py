#!/usr/bin/env python3
"""
Whisper Dictation - Hold Ctrl+Alt, speak, release to type.

Push-to-talk: hold Ctrl+Alt to record, release either to
stop recording, transcribe, and type the result wherever your cursor is.

Uses evdev for global hotkey capture (works in any window).
Requires: faster-whisper, sounddevice, numpy, evdev, xdotool, xclip, xprop
User must be in the 'input' group.
"""

import re
import subprocess
import sys
import threading
import time
import os
import select

import numpy as np
import sounddevice as sd
import evdev
from evdev import ecodes

# --- Configuration (override via environment variables) ---
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
LANGUAGE = os.environ.get("WHISPER_LANG", "en")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE", "float16")
SAMPLE_RATE = 16000
CHANNELS = 1

CTRL_CODES = {ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL}
ALT_CODES = {ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT}

# Prompt that encourages proper punctuation from whisper
INITIAL_PROMPT = "Hello, how are you? I'm doing well. Let's discuss the project."

# Filler words/phrases to strip (order matters — longer phrases first)
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


def clean_text(text):
    """Remove filler words and clean up spacing/punctuation."""
    for filler in FILLERS:
        text = text.replace(filler, " ")
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.!?])\s{2,}", r"\1 ", text)
    return text.strip()


# --- Global state ---
recording = False
audio_frames = []
stream = None
model = None
lock = threading.Lock()
keys_held = set()
target_window = None


def get_active_window():
    """Get the currently focused window ID."""
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


TERMINAL_CLASSES = {
    "gnome-terminal", "gnome-terminal-server", "kitty", "alacritty",
    "konsole", "xterm", "urxvt", "st-256color", "terminator",
    "tilix", "xfce4-terminal", "mate-terminal", "foot",
}


def _is_terminal(window_id):
    """Check if a window is a terminal emulator."""
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


def type_text(text, window_id=None):
    """Paste text via clipboard into the target window."""
    if not text.strip():
        return
    subprocess.run(
        ["xclip", "-selection", "clipboard"],
        input=(text + " ").encode(), check=False,
    )
    if window_id:
        subprocess.run(
            ["xdotool", "windowfocus", "--sync", window_id],
            check=False,
        )
    paste_key = "ctrl+shift+v" if _is_terminal(window_id) else "ctrl+v"
    subprocess.run(
        ["xdotool", "key", "--clearmodifiers", paste_key],
        check=False,
    )


def load_model():
    """Load the Whisper model."""
    global model
    from faster_whisper import WhisperModel

    print(f"Loading model: {MODEL_SIZE} (device={DEVICE}, compute={COMPUTE_TYPE})", flush=True)
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
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
    segments, info = model.transcribe(
        audio, beam_size=1, vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=INITIAL_PROMPT, **kwargs,
    )
    text = " ".join(seg.text for seg in segments).strip()
    text = clean_text(text)

    if text:
        print(f">> Result: {text}", flush=True)
        type_text(text, target_window)
    else:
        print(">> No speech detected.", flush=True)


def _both_held():
    return bool(keys_held & CTRL_CODES) and bool(keys_held & ALT_CODES)


def find_keyboards():
    """Find keyboard input devices."""
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
    global recording

    devices = {kb.fd: kb for kb in keyboards}

    while True:
        r, _, _ = select.select(devices.values(), [], [])
        for dev in r:
            try:
                for event in dev.read():
                    if event.type != ecodes.EV_KEY:
                        continue

                    # value: 1=press, 0=release, 2=hold/repeat
                    if event.value == 1:
                        keys_held.add(event.code)
                        with lock:
                            if not recording and _both_held():
                                start_recording()
                    elif event.value == 0:
                        keys_held.discard(event.code)
                        with lock:
                            if recording and not _both_held():
                                recording = False
                                threading.Thread(
                                    target=stop_and_transcribe, daemon=True
                                ).start()
            except OSError:
                pass


def main():
    load_model()

    print("", flush=True)
    print("=== Whisper Dictation ready ===", flush=True)
    print(f"  Model:  {MODEL_SIZE}", flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print("", flush=True)

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

    try:
        monitor_keys(keyboards)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
