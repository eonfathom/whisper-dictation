#!/usr/bin/env python3
"""
Vox - Hold Ctrl+Win, speak, release to type.

Cross-platform push-to-talk dictation (Linux/X11 + Windows). Hold Ctrl+Win to
record, release to stop, transcribe, and paste the result wherever your cursor is.
The chord is configurable via VOX_HOTKEY (e.g. "ctrl+alt", "ctrl+shift").

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
LANGUAGE = os.environ.get("VOX_LANG", "en")
SAMPLE_RATE = 16000
CHANNELS = 1

# Optional LLM cleanup pass (Wispr-style). Off by default. Set VOX_LLM=anthropic
# (with an ANTHROPIC_API_KEY) to polish the raw transcript with Claude before it's
# typed. Any failure/offline falls back to the raw text - a dictation is never lost.
# Model defaults to fast Haiku for low latency; override with VOX_LLM_MODEL.
LLM_BACKEND = os.environ.get("VOX_LLM", "off").lower()
LLM_MODEL = os.environ.get("VOX_LLM_MODEL", "claude-haiku-4-5")
LLM_TIMEOUT = float(os.environ.get("VOX_LLM_TIMEOUT", "8"))

# Push-to-talk chord: hold all these modifiers together to record. Override via
# VOX_HOTKEY, e.g. "ctrl+alt" or "ctrl+shift". Supported modifiers:
# ctrl, alt, win, shift  ("win" is the Windows/Super key).
HOTKEY_MODS = tuple(
    m for m in (t.strip().lower() for t in
                os.environ.get("VOX_HOTKEY", "ctrl+win").split("+")) if m
) or ("ctrl", "win")
HOTKEY_LABEL = "+".join(m.capitalize() for m in HOTKEY_MODS)

# Trailing-audio robustness (seconds, override via env): capture a short grace
# tail after release so the last word isn't clipped, and pad the buffer with a
# little silence so Whisper reliably finalizes the final segment. Keep the pad
# small - a long silent tail invites Whisper to hallucinate the primed hotwords.
RELEASE_TAIL_SEC = float(os.environ.get("VOX_RELEASE_TAIL", "0.2"))
TRAILING_PAD_SEC = float(os.environ.get("VOX_PAD", "0.15"))

# Strip a hallucinated trailing run of hotwords/proper nouns that Whisper can
# regurgitate over the silent tail after you stop speaking. On by default;
# set VOX_STRIP_PHANTOMS=0 to disable.
STRIP_PHANTOMS = os.environ.get("VOX_STRIP_PHANTOMS", "1").lower() not in (
    "0", "false", "off", "no",
)

# Decoding beam width. Higher = more accurate, a bit slower. 5 is Whisper's
# standard default; drop to 1 on a slow CPU if latency matters more than accuracy.
BEAM_SIZE = int(os.environ.get("VOX_BEAM", "5"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.environ.get(
    "VOX_DICT", os.path.join(SCRIPT_DIR, "dictionary.json")
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

    Set VOX_DEVICE / VOX_COMPUTE to override. Auto-detection means the
    same code runs on a CUDA desktop and an Intel laptop with no config changes.
    """
    dev = os.environ.get("VOX_DEVICE")
    compute = os.environ.get("VOX_COMPUTE")
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
# dictation). Override anytime with VOX_MODEL.
MODEL_SIZE = os.environ.get("VOX_MODEL") or (
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
# Individual words from every hotword entry ("Claude Code" -> claude, code),
# used to detect a hallucinated trailing run of hotwords.
_HOTWORD_TOKENS = {
    w for hw in HOTWORDS for w in re.split(r"[\s\-]+", hw.lower()) if w
}
_NAMELIKE_RE = re.compile(r"^[A-Z][\w'\-]*$")
# Space-joined runs of >=2 consecutive dictionary entries, longest first
# (e.g. "OBS Andregg Rokid"). Whisper is prompted with " ".join(HOTWORDS), so
# when its decoder echoes the prompt it reproduces exactly such a run:
# dictionary order, original casing, nothing but spaces between entries.
_HOTWORD_RUNS = sorted(
    (
        " ".join(HOTWORDS[i:j])
        for i in range(len(HOTWORDS))
        for j in range(i + 2, len(HOTWORDS) + 1)
    ),
    key=len,
    reverse=True,
)


def apply_corrections(text):
    """Replace known misrecognitions (whole-word, case-insensitive)."""
    for pattern, right in _CORRECTION_RES:
        text = pattern.sub(right, text)
    return text


def strip_trailing_hotword_run(text):
    """Remove a verbatim echo of the hotword prompt from the end of the text.

    Whisper sees the dictionary as one space-joined prompt string on every
    decode window; when the final window is mostly silence or a garbled tail,
    the decoder can copy a chunk of that prompt instead of transcribing the
    audio - even mid-sentence, replacing real speech: "...feedback I've given
    you, OBS Andregg Rokid". The echo's fingerprint is >=2 dictionary entries
    in dictionary order with prompt casing and only spaces between them; real
    speech that mentions a hotword ("I met Andregg"), or several with natural
    punctuation ("Claude Code, Anthropic"), never takes that exact shape.
    Disable with VOX_STRIP_PHANTOMS=0.
    """
    if not STRIP_PHANTOMS or not _HOTWORD_RUNS or not text:
        return text
    trail = text.rstrip().rstrip(".,!?;: ")
    for run in _HOTWORD_RUNS:
        if not trail.endswith(run):
            continue
        head = trail[: -len(run)]
        if head and not (head[-1].isspace() or head[-1] in ",;:.!?"):
            continue  # run must start on a word boundary
        head = head.rstrip().rstrip(",;: ")
        print(f">> Stripped hotword echo from tail: {run!r}", flush=True)
        return head
    return text


def strip_trailing_phantoms(text):
    """Remove a hallucinated trailing run of hotwords/proper nouns.

    When you stop talking, Whisper can regurgitate the words it was primed with
    (hotwords + initial prompt) over the trailing silence, appending phantoms
    like "Fathom Claude" after your real sentence. This strips that tail, but
    only in the telltale shape: a completed sentence (ends in . ! ?) followed by
    nothing but capitalized / hotword tokens, at least one of which is a hotword.
    So real endings survive - "I love Claude." (hotword before the period),
    "email Michael" (no terminator), and any sentence containing a normal
    lowercase word are all left untouched. Disable with VOX_STRIP_PHANTOMS=0.
    """
    if not STRIP_PHANTOMS or not HOTWORDS or not text:
        return text
    stripped = text.rstrip()
    idx = max((stripped.rfind(c) for c in ".!?"), default=-1)
    if idx < 0 or idx == len(stripped) - 1:
        return text  # no completed sentence, or nothing trails it
    head = stripped[: idx + 1]
    tail = stripped[idx + 1:].strip()
    if not head.strip(".!? ") or not tail:
        return text  # need real words before the terminator, and a tail to test
    has_hotword = False
    for tok in re.split(r"[,\s]+", tail):
        if not tok:
            continue
        word = tok.strip(".,!?;:'\"").lower()
        if word in _HOTWORD_TOKENS:
            has_hotword = True
        elif not _NAMELIKE_RE.match(tok):
            return text  # a normal lowercase word -> this is real speech, keep it
    return head if has_hotword else text


def clean_text(text):
    """Remove filler words and clean up spacing/punctuation."""
    for filler in FILLERS:
        text = text.replace(filler, " ")
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.!?])\s{2,}", r"\1 ", text)
    return text.strip()


def llm_format(text):
    """Optional Wispr-style cleanup: polish the raw transcript with an LLM.

    Enabled only when VOX_LLM="anthropic". Fixes grammar/punctuation/capitalization,
    drops fillers and false starts, and obeys spoken commands ("new paragraph",
    "scratch that"). On a missing key, offline, timeout, or any error it returns the
    input unchanged, so a dictation is never lost. Uses the official Anthropic SDK.
    """
    if LLM_BACKEND != "anthropic" or not text.strip():
        return text
    try:
        import anthropic
        client = anthropic.Anthropic(max_retries=0, timeout=LLM_TIMEOUT)
        keep = ""
        if HOTWORDS:
            keep = (
                " Reference spellings, to be used ONLY where the transcript"
                " already contains these terms: " + ", ".join(HOTWORDS) + "."
                " Never insert or append them anywhere else; if the transcript"
                " ends abruptly, leave the ending exactly as it is."
            )
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=4000,
            system=(
                "You are a dictation cleanup engine. Turn raw speech-to-text into "
                "polished written text: fix capitalization, punctuation, and obvious "
                "mis-transcriptions; remove fillers and false starts; honor spoken "
                "commands (new paragraph, new line, scratch that). Output ONLY the "
                "cleaned text - no preamble, quotes, or commentary." + keep
            ),
            messages=[{"role": "user", "content": text}],
        )
        cleaned = "".join(b.text for b in msg.content if b.type == "text").strip()
        return cleaned or text
    except Exception as e:
        print(f">> LLM cleanup skipped ({e.__class__.__name__}); using raw text",
              flush=True)
        return text


def post_process(text):
    """Transform a raw transcript into final text via an ordered pipeline.

    Order matters: clean, drop hallucinated trailing hotwords (before the LLM, so
    it can't "preserve" them), then the optional LLM pass, then personal-dictionary
    corrections LAST so they always win (e.g. Rokid) even over the LLM's rewrite.
    """
    text = clean_text(text)                  # 1. strip fillers, tidy spacing
    text = strip_trailing_hotword_run(text)  # 2. drop verbatim hotword-prompt echo
    text = strip_trailing_phantoms(text)     # 3. drop hallucinated trailing hotwords
    text = llm_format(text)                  # 4. optional LLM cleanup (off by default)
    text = apply_corrections(text)           # 5. personal-dictionary corrections (final say)
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
    if LLM_BACKEND != "off":
        print(f"  LLM cleanup: {LLM_BACKEND} ({LLM_MODEL})", flush=True)
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
