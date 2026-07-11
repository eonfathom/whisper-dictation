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
import logging
import logging.handlers
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import wave

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

# Optional LLM cleanup pass (Wispr-style). Off by default. Fixes punctuation,
# capitalization, fillers, false starts, and honors spoken commands. Any
# failure/offline/timeout falls back to the raw text - a dictation is never lost.
#   VOX_LLM=local     -> a LOCAL OpenAI-compatible server (Ollama by default);
#                        fully offline, no API key, low latency. RECOMMENDED.
#   VOX_LLM=anthropic -> Claude via the Anthropic API (needs ANTHROPIC_API_KEY).
# VOX_LLM_URL points at the local server's OpenAI-compatible base (default
# Ollama). VOX_LLM_MODEL picks the model; the local default is a small, fast
# instruct model. VOX_LLM_KEEPALIVE keeps the model resident between dictations
# so there's no reload latency (Ollama-specific, ignored elsewhere).
LLM_BACKEND = os.environ.get("VOX_LLM", "off").lower()
_LOCAL_BACKENDS = ("local", "ollama", "openai-compatible")
_DEFAULT_MODEL = (
    "llama3.2:3b" if LLM_BACKEND in _LOCAL_BACKENDS else "claude-haiku-4-5"
)
LLM_MODEL = os.environ.get("VOX_LLM_MODEL", _DEFAULT_MODEL)
LLM_URL = os.environ.get("VOX_LLM_URL", "http://localhost:11434/v1").rstrip("/")
LLM_KEEPALIVE = os.environ.get("VOX_LLM_KEEPALIVE", "30m")
# Generous default: the fallback on timeout is the raw transcript (never lost),
# so the only cost of a high ceiling is how long a genuinely stuck request waits.
# A warm local 3B cleans a normal dictation in ~2s; long ones need headroom.
LLM_TIMEOUT = float(os.environ.get("VOX_LLM_TIMEOUT", "20"))

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

# Optional daily transcript record: point VOX_TRANSCRIPT_DIR at a directory
# (e.g. a folder inside an Obsidian vault) and the final text of every
# dictation is appended to a per-day markdown file there ("YYYY-MM-DD vox.md",
# one "- **HH:MM** text" bullet per dictation). Unset = off. Writing is
# best-effort: a failure is logged and never blocks the paste.
TRANSCRIPT_DIR = os.environ.get("VOX_TRANSCRIPT_DIR", "")

# Diagnostic audio retention: keep the raw audio of recent dictations as WAV
# files so a bad transcription can be replayed and re-transcribed after the
# fact - the transcript alone can't distinguish "Whisper mis-decoded the tail"
# from "the microphone never delivered the speech" (e.g. another app grabbed
# the device); the WAV answers that decisively. Default keeps the last 20
# under %LOCALAPPDATA%\vox\audio (Linux: ~/.local/state/vox/audio).
# VOX_SAVE_AUDIO=0 disables; VOX_SAVE_AUDIO=<dir> overrides the location.
SAVE_AUDIO = os.environ.get("VOX_SAVE_AUDIO", "1")
AUDIO_KEEP = int(os.environ.get("VOX_AUDIO_KEEP", "20"))

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


# --- Diagnostic log -------------------------------------------------------------
def _make_logger():
    """File logger so windowless runs are diagnosable after the fact.

    Under pythonw.exe there is no stdout - every print() is silently discarded -
    so a mangled dictation used to leave no trace. A small rotating file
    (512 KB x 2) captures per-dictation diagnostics: duration/samples captured,
    the raw transcript, anything stripped or recovered, and the final text.
    Default %LOCALAPPDATA%\\vox\\vox.log (Linux: ~/.local/state/vox/vox.log);
    override the path with VOX_LOG, disable with VOX_LOG=0.
    """
    dest = os.environ.get("VOX_LOG", "")
    if dest.lower() in ("0", "false", "off", "no"):
        return None
    if not dest:
        base = (os.environ.get("LOCALAPPDATA") if IS_WINDOWS else None
                ) or os.path.expanduser("~/.local/state")
        dest = os.path.join(base, "vox", "vox.log")
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            dest, maxBytes=512 * 1024, backupCount=2, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger = logging.getLogger("vox")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger
    except OSError:
        return None


_LOGGER = _make_logger()


def _state_dir():
    """Per-user state directory (same base the diagnostic log uses)."""
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get(
            "XDG_STATE_HOME"
        ) or os.path.expanduser("~/.local/state")
    return os.path.join(base, "vox")


def log(msg):
    """Print to the console (visible when run via python) and the log file
    (the only record under pythonw, which has no stdout)."""
    print(msg, flush=True)
    if _LOGGER is not None:
        _LOGGER.info(msg)


def level_profile(audio, win_sec=5):
    """RMS level (dBFS) of each win_sec window, e.g. '-33 -31 -29 -78'.

    Makes a dead or hijacked microphone visible in the log without listening
    to anything: speech sits around -35..-15, a run of values below about -60
    where speech should be means the device delivered silence.
    """
    n = max(1, int(SAMPLE_RATE * win_sec))
    out = []
    for i in range(0, len(audio), n):
        chunk = audio[i:i + n].astype(np.float64)
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        out.append(f"{20 * np.log10(rms):.0f}" if rms > 1e-9 else "-inf")
    return " ".join(out)


def save_debug_audio(audio):
    """Keep the raw audio of recent dictations for after-the-fact diagnosis.

    Best-effort and rotating (newest AUDIO_KEEP kept): a failure is logged and
    never blocks transcription. Returns the saved path or None.
    """
    if SAVE_AUDIO.strip().lower() in ("0", "false", "off", "no", ""):
        return None
    try:
        dest = SAVE_AUDIO if SAVE_AUDIO != "1" else os.path.join(
            _state_dir(), "audio"
        )
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(
            dest, time.strftime("rec-%Y%m%d-%H%M%S.wav")
        )
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        stale = sorted(
            f for f in os.listdir(dest)
            if f.startswith("rec-") and f.endswith(".wav")
        )[:-AUDIO_KEEP]
        for f in stale:
            os.remove(os.path.join(dest, f))
        return path
    except Exception as e:
        log(f">> WARNING: could not save debug audio: {e}")
        return None


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
        log(f"WARNING: could not read dictionary {DICT_PATH}: {e}")
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


def strip_trailing_hotword_run(text, quiet=False):
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
        if not quiet:
            log(f">> Stripped hotword echo from tail: {run!r}")
        return head
    return text


def strip_trailing_phantoms(text, quiet=False):
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
    if has_hotword:
        if not quiet:
            log(f">> Stripped trailing phantoms: {tail!r}")
        return head
    return text


def strip_phantoms(text, quiet=False):
    """Apply both echo strippers (verbatim run, then trailing phantoms)."""
    return strip_trailing_phantoms(
        strip_trailing_hotword_run(text, quiet=quiet), quiet=quiet
    )


def clean_text(text):
    """Remove filler words and clean up spacing/punctuation."""
    for filler in FILLERS:
        text = text.replace(filler, " ")
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.!?])\s{2,}", r"\1 ", text)
    return text.strip()


def _cleanup_system_prompt():
    """Shared instruction for both the local and Anthropic cleanup backends."""
    keep = ""
    if HOTWORDS:
        keep = (
            " Reference spellings, to be used ONLY where the transcript"
            " already contains these terms: " + ", ".join(HOTWORDS) + "."
            " Never insert or append them anywhere else; if the transcript"
            " ends abruptly, leave the ending exactly as it is."
        )
    return (
        "You are a dictation cleanup engine. Turn raw speech-to-text into "
        "polished written text: fix capitalization, punctuation, and obvious "
        "mis-transcriptions; remove fillers and false starts; honor spoken "
        "commands (new paragraph, new line, scratch that). Output ONLY the "
        "cleaned text - no preamble, quotes, or commentary." + keep
    )


def _llm_format_local(text, system):
    """Cleanup via a LOCAL OpenAI-compatible chat endpoint (Ollama etc.).

    Stdlib-only (urllib), so no extra dependency and nothing to install in the
    venv. Deterministic (temperature 0) and offline. keep_alive keeps the model
    resident on Ollama so there's no per-dictation reload; harmless on servers
    that ignore the field.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "stream": False,
        "keep_alive": LLM_KEEPALIVE,
    }
    req = urllib.request.Request(
        LLM_URL + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def _llm_format_anthropic(text, system):
    """Cleanup via the Anthropic API (needs ANTHROPIC_API_KEY). Uses the SDK."""
    import anthropic
    client = anthropic.Anthropic(max_retries=0, timeout=LLM_TIMEOUT)
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": text}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def warm_llm():
    """Pre-load the local cleanup model so the FIRST dictation isn't slow.

    A cold Ollama model load is many seconds (weights into VRAM); without this
    the first real dictation would block or time out and fall back to raw text.
    Runs in a background thread at startup so it never delays readiness, and is
    best-effort - if the server isn't up yet it just logs and moves on (the
    first dictation pays the load cost instead). Local backend only.
    """
    if LLM_BACKEND not in _LOCAL_BACKENDS:
        return

    def _warm():
        try:
            t0 = time.monotonic()
            _llm_format_local("ok", _cleanup_system_prompt())
            log(f">> LLM warm-up done in {time.monotonic() - t0:.1f}s "
                f"({LLM_MODEL} resident)")
        except Exception as e:
            log(f">> LLM warm-up skipped ({e.__class__.__name__}); "
                "first dictation will load the model")

    threading.Thread(target=_warm, daemon=True).start()


def llm_format(text):
    """Optional Wispr-style cleanup: polish the raw transcript with an LLM.

    Off unless VOX_LLM is set: "local" (default) uses a local OpenAI-compatible
    server - offline, no API key; "anthropic" uses Claude. Fixes
    grammar/punctuation/capitalization, drops fillers and false starts, and obeys
    spoken commands. On offline/timeout/any error it returns the input unchanged,
    so a dictation is never lost.
    """
    if LLM_BACKEND in ("off", "") or not text.strip():
        return text
    t0 = time.monotonic()
    try:
        system = _cleanup_system_prompt()
        if LLM_BACKEND in _LOCAL_BACKENDS:
            cleaned = _llm_format_local(text, system)
        elif LLM_BACKEND == "anthropic":
            cleaned = _llm_format_anthropic(text, system)
        else:
            log(f">> Unknown VOX_LLM={LLM_BACKEND!r}; using raw text")
            return text
        log(f">> LLM cleanup ({LLM_BACKEND}) in {time.monotonic() - t0:.2f}s")
        return cleaned or text
    except Exception as e:
        log(f">> LLM cleanup skipped ({e.__class__.__name__}: {e}); using raw text")
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


def record_transcript(text):
    """Append the pasted text to a per-day markdown file (VOX_TRANSCRIPT_DIR).

    Creates "<dir>/YYYY-MM-DD vox.md" on the first dictation of the day, with
    a header linking the day's sibling notes so Obsidian backlinks tie them
    together. Best-effort by design: any failure is logged and the dictation
    itself is never delayed or lost (this runs after the paste).
    """
    if not TRANSCRIPT_DIR:
        return
    try:
        now = time.localtime()
        day = time.strftime("%Y-%m-%d", now)
        path = os.path.join(TRANSCRIPT_DIR, f"{day} vox.md")
        entry = f"- **{time.strftime('%H:%M', now)}** {text}\n"
        if not os.path.exists(path):
            entry = (
                f"# Vox transcripts — {day}\n\n"
                f"*Dictated via [vox](https://github.com/eonfathom/vox) · "
                f"[[Daily transcripts/{day} transcript|Day transcript]] · "
                f"[[Daily summaries/{day} story|Day story]]*\n\n"
            ) + entry
        os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception as e:
        log(f">> Transcript record failed ({e.__class__.__name__}: {e})")


# --- Global state ---
recording = False
audio_frames = []
overflow_count = 0
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

    log(f"Loading model: {MODEL_SIZE} (device={DEVICE}, compute={COMPUTE_TYPE})")
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    try:
        supports_hotwords = "hotwords" in inspect.signature(
            model.transcribe
        ).parameters
    except (ValueError, TypeError):
        supports_hotwords = False
    log("Model loaded.")


def start_recording():
    """Start recording audio from the default microphone."""
    global recording, audio_frames, overflow_count, stream, target_window

    audio_frames = []
    overflow_count = 0
    target_window = get_active_window()

    def callback(indata, frames, time_info, status):
        # Count under/overflow flags instead of printing: printing from the
        # audio thread can itself glitch capture, and under pythonw stderr is
        # None so the old message vanished anyway. Reported at stop.
        global overflow_count
        if status:
            overflow_count += 1
        audio_frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
    )
    stream.start()
    recording = True
    # Name the device each recording actually opened: if Windows switched the
    # default input (e.g. a Bluetooth headset connected), it shows up here.
    try:
        mic = sd.query_devices(sd.default.device[0])["name"]
    except Exception:
        mic = "unknown"
    log(f">> RECORDING - speak now... (mic: {mic})")


def transcribe_audio(audio, with_hotwords=True):
    """One Whisper pass over the buffer; returns the raw joined transcript."""
    kwargs = {}
    if LANGUAGE:
        kwargs["language"] = LANGUAGE
    if with_hotwords and HOTWORDS and supports_hotwords:
        kwargs["hotwords"] = " ".join(HOTWORDS)
    segments, info = model.transcribe(
        audio, beam_size=BEAM_SIZE, vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=INITIAL_PROMPT, **kwargs,
    )
    return " ".join(seg.text for seg in segments).strip()


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
        log(">> No audio captured.")
        return

    audio = np.concatenate(frames, axis=0).flatten()
    duration = len(audio) / SAMPLE_RATE
    # Pad with trailing silence so Whisper reliably finalizes the last segment.
    if TRAILING_PAD_SEC > 0:
        audio = np.concatenate(
            [audio, np.zeros(int(SAMPLE_RATE * TRAILING_PAD_SEC), dtype=audio.dtype)]
        )
    log(f">> Transcribing {duration:.1f}s of audio ({len(frames)} blocks"
        + (f", {overflow_count} overflow flags" if overflow_count else "")
        + ")...")
    log(f">> Levels (dBFS per 5s): {level_profile(audio)}")
    saved = save_debug_audio(audio)
    if saved:
        log(f">> Audio kept: {saved}")

    text = transcribe_audio(audio)
    log(f">> Raw: {text}")

    # Hotword-prompt echo can REPLACE speech, not just trail it: past ~30s of
    # speech Whisper decodes in multiple windows, and a window whose decode
    # derails into the echo consumes its whole span of audio - the last
    # sentence or two come back as hotword junk. Stripping the junk leaves the
    # text CLEAN but silently TRUNCATED, so detection alone isn't enough. When
    # either echo guard would strip something, retranscribe once WITHOUT hotword
    # priming (the echo text is no longer in the prompt, so the tail decodes as
    # real speech) and keep whichever result preserved more speech. Measured on
    # the length AFTER stripping, so we compare real content, not echo noise.
    # Dictionary corrections still run on the winner via post_process.
    cleaned = clean_text(text)
    stripped = strip_phantoms(cleaned, quiet=True)
    if stripped != cleaned:
        retry = transcribe_audio(audio, with_hotwords=False)
        retry_stripped = strip_phantoms(clean_text(retry), quiet=True)
        log(f">> Echo detected; retried without hotwords "
            f"({len(stripped)} -> {len(retry_stripped)} chars of real speech)")
        if len(retry_stripped) > len(stripped):
            log(">> Recovered lost speech via no-hotword retry")
            text = retry

    text = post_process(text)

    if text:
        log(f">> Result: {text}")
        type_text(text, target_window)
        record_transcript(text)
    else:
        log(">> No speech detected.")


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
    warm_llm()  # background pre-load of the local cleanup model (no-op if off)

    print("", flush=True)
    log("=== Vox ready ===")
    log(f"  Model:  {MODEL_SIZE}")
    log(f"  Device: {DEVICE} ({COMPUTE_TYPE})")
    if LLM_BACKEND != "off":
        log(f"  LLM cleanup: {LLM_BACKEND} ({LLM_MODEL})")
    if HOTWORDS or CORRECTIONS:
        log(f"  Dictionary: {len(HOTWORDS)} hotwords, "
            f"{len(CORRECTIONS)} corrections")
    if TRANSCRIPT_DIR:
        log(f"  Transcript dir: {TRANSCRIPT_DIR}")
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
