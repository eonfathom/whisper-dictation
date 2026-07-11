"""Floating dictation HUD for Vox (Windows).

A tiny always-on-top overlay that follows the mouse cursor while you hold the
push-to-talk chord: a pulsing record dot, a live elapsed timer (m:ss.mmm), and
a mic level meter driven by the actual audio callback. On release it switches
to a "transcribing" spinner, then flashes the result stats (words / seconds /
LLM tokens) for a moment and disappears.

Design constraints, in order:
  - NEVER interfere with dictation. The window is click-through
    (WS_EX_TRANSPARENT), never takes focus (WS_EX_NOACTIVATE), skips the
    Alt-Tab list (WS_EX_TOOLWINDOW), and every entry point is wrapped so a HUD
    failure can only cost the overlay, not a transcription.
  - No dependencies: tkinter + ctypes, both in the standard library.
  - Thread-safe by construction: tkinter runs in its own daemon thread and is
    the ONLY thread that touches widgets; the rest of the app just assigns
    plain attributes on the Hud object (atomic under the GIL) that the tk
    thread polls at ~30 fps.

Text is drawn twice (black shadow + bright fill) so it stays readable over
both light and dark backgrounds without an opaque panel.

Disable with VOX_HUD=0. VOX_HUD_ANCHOR=corner pins it to the bottom-right of
the primary screen instead of following the cursor.
"""

import ctypes
import math
import os
import time

IS_WINDOWS = os.name == "nt"

ENABLED = os.environ.get("VOX_HUD", "1").strip().lower() not in (
    "0", "false", "off", "no",
)
# Placement: "caret" anchors to the text insertion point where available and
# falls back to a static position captured at record-start (never chases the
# mouse); "cursor" follows the mouse live (old behavior); "corner" pins to the
# primary screen's bottom-right.
ANCHOR = os.environ.get("VOX_HUD_ANCHOR", "caret").strip().lower()

# Thin by design: Segoe UI Light, small, normal weight, no shadow. All tunable
# live via env so styling doesn't need a code change.
FONT = os.environ.get("VOX_HUD_FONT", "Segoe UI Light")
try:
    SIZE = int(os.environ.get("VOX_HUD_SIZE", "8"))   # pixels (see negative-size note)
except ValueError:
    SIZE = 8


def _opacity():
    """Whole-overlay opacity, 0.05..1.0 (VOX_HUD_OPACITY, default 1.0).

    Applied as a layered-window alpha over everything drawn. The thin 8px light
    text is already subtle; lower this if you want it fainter. Clamped so a typo
    can't make it invisible or fully opaque-and-heavy."""
    try:
        return max(0.05, min(1.0, float(os.environ.get("VOX_HUD_OPACITY", "1.0"))))
    except ValueError:
        return 1.0

# The transparent-color key: every pixel painted this exact color becomes a
# hole in the window. Chosen to be a color no HUD element ever uses.
_KEY = "#010203"
_FPS_MS = 33          # ~30 fps poll/redraw
_DONE_SECS = 1.6      # how long the result stats linger
_W, _H = 240, 44      # canvas size
_OFFSET = (18, 26)    # HUD position relative to the mouse cursor (cursor mode)
_CARET_OFFSET = (14, 6)   # below-right of the text caret, so it clears the line

# Windows extended-style bits for a ghost window.
_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_NOACTIVATE = 0x08000000
_WS_EX_TOOLWINDOW = 0x00000080


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _GuiThreadInfo(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("flags", ctypes.c_ulong),
                ("hwndActive", ctypes.c_void_p), ("hwndFocus", ctypes.c_void_p),
                ("hwndCapture", ctypes.c_void_p), ("hwndMenuOwner", ctypes.c_void_p),
                ("hwndMoveSize", ctypes.c_void_p), ("hwndCaret", ctypes.c_void_p),
                ("rcCaret", _Rect)]


def _cursor_pos():
    pt = _Point()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def _caret_pos():
    """Screen (x, y) of the focused window's text caret, or None.

    Works in native Win32 controls via GetGUIThreadInfo. Chromium/Electron apps
    (browsers, Obsidian, VS Code) draw their own caret the OS can't see, so this
    returns None there and the caller falls back to a static anchor. Bottom-left
    of the caret rect is used so the HUD sits just under the current line."""
    try:
        u = ctypes.windll.user32
        fg = u.GetForegroundWindow()
        if not fg:
            return None
        tid = u.GetWindowThreadProcessId(fg, None)
        gti = _GuiThreadInfo()
        gti.cbSize = ctypes.sizeof(gti)
        if not u.GetGUIThreadInfo(tid, ctypes.byref(gti)) or not gti.hwndCaret:
            return None
        pt = _Point(gti.rcCaret.left, gti.rcCaret.bottom)
        u.ClientToScreen(gti.hwndCaret, ctypes.byref(pt))
        if pt.x == 0 and pt.y == 0:
            return None
        return pt.x, pt.y
    except Exception:
        return None


def _virtual_screen():
    """(left, top, width, height) of the full multi-monitor desktop."""
    m = ctypes.windll.user32.GetSystemMetrics
    return m(76), m(77), m(78), m(79)  # SM_[XY]VIRTUALSCREEN, SM_C[XY]VIRTUALSCREEN


class Hud:
    """State mailbox + tk thread. See module docstring for the contract."""

    def __init__(self, logger=None):
        self._log = logger or (lambda msg: None)
        # -- written by the app threads, read by the tk thread --
        self.phase = "idle"        # idle | rec | busy | done
        self.t0 = 0.0              # monotonic start of recording
        self.level_db = -120.0     # latest mic RMS in dBFS
        self.stats = ""            # e.g. "42 w · 7.4s · 128 tok"
        self._done_at = 0.0
        self._anchor = (0, 0)      # static fallback position (mouse at rec start)
        # Session generation: each recording() bumps it; busy()/done()/idle()
        # from a superseded session (a slow transcription finishing after the
        # user re-pressed the chord) carry an older gen and are ignored, so a
        # stale result can't stomp or prematurely hide the live recording.
        self._gen = 0

    # ---- API called from Vox (any thread) ----------------------------------

    def start(self):
        """Spawn the tk thread. Failure disables the HUD, nothing else."""
        if not (ENABLED and IS_WINDOWS):
            return self
        import threading
        threading.Thread(target=self._run, daemon=True, name="vox-hud").start()
        return self

    def recording(self):
        """Begin a new session; returns its generation for busy()/done()/idle()."""
        self.t0 = time.monotonic()
        self.level_db = -120.0
        # Static fallback anchor: where the mouse is at record-start. Used only
        # when no caret is available, and captured ONCE so the HUD holds still
        # instead of chasing the mouse.
        try:
            self._anchor = _cursor_pos() if IS_WINDOWS else (0, 0)
        except Exception:
            self._anchor = (0, 0)
        self._gen += 1
        self.phase = "rec"
        return self._gen

    def session(self):
        """Current generation, captured by the release path to tag its updates."""
        return self._gen

    def busy(self, gen=None):
        if gen is not None and gen != self._gen:
            return
        if self.phase == "rec":
            self.phase = "busy"

    def done(self, words, secs, tokens=None, gen=None):
        if gen is not None and gen != self._gen:
            return
        parts = [f"{words} w", f"{secs:.1f}s"]
        if tokens:
            parts.append(f"{tokens} tok")
        self.stats = " · ".join(parts)
        self._done_at = time.monotonic()
        self.phase = "done"

    def idle(self, gen=None):
        if gen is not None and gen != self._gen:
            return
        self.phase = "idle"

    def feed_level(self, rms):
        """Record the latest mic RMS (0..1 float). Called from the audio
        callback, so it must stay allocation-free and never raise."""
        self.level_db = -120.0 if rms <= 1e-9 else 20.0 * math.log10(rms)

    # ---- tk thread ----------------------------------------------------------

    def _run(self):
        try:
            import tkinter as tk

            self._root = root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.config(bg=_KEY)
            root.attributes("-transparentcolor", _KEY)
            # Whole-window alpha on top of the color-key: keyed pixels stay fully
            # clickable-through holes, everything drawn dims to a faint ghost.
            root.attributes("-alpha", _opacity())
            self._canvas = tk.Canvas(
                root, width=_W, height=_H, bg=_KEY, highlightthickness=0,
            )
            self._canvas.pack()
            root.withdraw()
            root.update_idletasks()
            self._ghost(root)
            self._visible = False
            self._spin = 0
            self._alive = True
            root.after(_FPS_MS, self._tick)
            root.mainloop()
        except Exception as e:
            self._alive = False
            self._log(f">> HUD disabled ({e.__class__.__name__}: {e})")

    def _ghost(self, root):
        """Make the window click-through, non-activating, and Alt-Tab-less."""
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id()
        get = ctypes.windll.user32.GetWindowLongPtrW
        put = ctypes.windll.user32.SetWindowLongPtrW
        style = get(hwnd, _GWL_EXSTYLE)
        put(hwnd, _GWL_EXSTYLE,
            style | _WS_EX_LAYERED | _WS_EX_TRANSPARENT
            | _WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW)

    def _tick(self):
        try:
            phase = self.phase
            if phase == "done" and time.monotonic() - self._done_at > _DONE_SECS:
                self.phase = phase = "idle"
            if phase == "idle":
                if self._visible:
                    self._root.withdraw()
                    self._visible = False
            else:
                self._place()
                if not self._visible:
                    self._root.deiconify()
                    self._root.attributes("-topmost", True)
                    self._visible = True
                self._draw(phase)
        except Exception:
            pass  # a bad frame must never kill the HUD loop
        self._root.after(_FPS_MS, self._tick)

    def _place(self):
        if ANCHOR == "corner":
            sw = self._root.winfo_screenwidth()
            sh = self._root.winfo_screenheight()
            x, y = sw - _W - 24, sh - _H - 64
        elif ANCHOR == "cursor":
            cx, cy = _cursor_pos()
            x, y = cx + _OFFSET[0], cy + _OFFSET[1]
        else:  # caret: track the text insertion point, static fallback
            caret = _caret_pos()
            if caret:
                x, y = caret[0] + _CARET_OFFSET[0], caret[1] + _CARET_OFFSET[1]
            else:
                x, y = self._anchor[0] + _OFFSET[0], self._anchor[1] + _OFFSET[1]
        vl, vt, vw, vh = _virtual_screen()
        x = max(vl, min(x, vl + vw - _W))
        y = max(vt, min(y, vt + vh - _H))
        self._root.geometry(f"{_W}x{_H}+{x}+{y}")

    # ---- drawing ------------------------------------------------------------

    def _text(self, x, y, s, fill, size=None, anchor="w"):
        """Thin light text: configured font, normal weight, no shadow. Negative
        tk font size = pixels, so 8 renders as 8px regardless of screen DPI."""
        self._canvas.create_text(
            x, y, text=s, fill=fill, font=(FONT, -(size or SIZE)), anchor=anchor)

    def _tabular(self, x, y, s, fill, cell):
        """Draw each glyph centered in a fixed-width cell, so a digit changing
        (1 -> 2) never shifts the string or anything after it. The even cell
        also gives the numbers breathing room. Returns the end x."""
        for i, ch in enumerate(s):
            self._canvas.create_text(
                x + i * cell + cell / 2.0, y, text=ch, fill=fill,
                font=(FONT, -SIZE), anchor="c")
        return x + len(s) * cell

    def _draw(self, phase):
        c = self._canvas
        c.delete("all")
        u = float(SIZE)
        y = _H // 2
        if phase == "rec":
            # gently breathing record dot (small amplitude = unobtrusive)
            pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 3.2)
            r = u * 0.42 + u * 0.13 * pulse
            cx = u * 1.3
            c.create_oval(cx - r, y - r, cx + r, y + r,
                          fill="#d1605b", outline="")
            # elapsed m:ss.t  (tenths only - calmer than milliseconds)
            el = max(0.0, time.monotonic() - self.t0)
            stamp = f"{int(el // 60)}:{int(el % 60):02d}.{int(el * 10 % 10)}"
            cell = u * 0.78
            end = self._tabular(u * 2.5, y, stamp, "#f2f3f5", cell)
            # mic meter: 5 bars over -55..-15 dBFS
            mx = end + u * 0.9
            step = u * 0.55
            lit = max(0, min(5, int((self.level_db + 55.0) / 40.0 * 5 + 0.5)))
            for i in range(5):
                bx = mx + i * step
                bh = u * 0.4 + i * (u * 0.3)
                color = "#1d9e75" if i < lit else "#2c2f36"
                c.create_rectangle(bx, y + u * 0.75, bx + max(1.0, u * 0.3),
                                   y + u * 0.75 - bh, fill=color, outline="")
            if self.level_db > -100:
                self._text(mx + 5 * step + u * 0.8, y, f"{self.level_db:.0f}",
                           "#8fa3ad", size=max(7, int(u * 0.85)))
        elif phase == "busy":
            rr = u * 1.0
            self._spin = (self._spin + 24) % 360
            c.create_arc(u * 0.6, y - rr, u * 0.6 + 2 * rr, y + rr,
                         start=self._spin, extent=250, style="arc",
                         outline="#e0a83a", width=max(1, int(u * 0.22)))
            self._text(u * 3.0, y, "transcribing…", "#e0d7c4")
        elif phase == "done":
            self._text(u * 0.8, y, "✓", "#5dcaa5")
            self._text(u * 2.6, y, self.stats, "#f2f3f5")


class NullHud:
    """Inert stand-in so call sites never need None checks."""

    def start(self):
        return self

    def recording(self):
        return 0

    def session(self):
        return 0

    def busy(self, gen=None):
        pass

    def done(self, words, secs, tokens=None, gen=None):
        pass

    def idle(self, gen=None):
        pass

    def feed_level(self, rms):
        pass


def create(logger=None):
    """Return a started Hud, or a NullHud when disabled/unsupported/broken."""
    try:
        if ENABLED and IS_WINDOWS:
            return Hud(logger).start()
    except Exception as e:
        if logger:
            logger(f">> HUD disabled ({e.__class__.__name__}: {e})")
    return NullHud()


if __name__ == "__main__":
    # Demo: python hud.py  — plays through the three phases near the cursor.
    import random
    hud = Hud(print).start()
    time.sleep(0.6)
    hud.recording()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.5:
        hud.feed_level(random.uniform(0.003, 0.12))
        time.sleep(0.05)
    hud.busy()
    time.sleep(1.2)
    hud.done(words=42, secs=3.5, tokens=128)
    time.sleep(2.2)
