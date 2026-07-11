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
ANCHOR = os.environ.get("VOX_HUD_ANCHOR", "cursor").strip().lower()


def _opacity():
    """Whole-overlay opacity, 0.05..1.0 (VOX_HUD_OPACITY, default 0.5).

    Applied as a layered-window alpha over everything drawn, so the meter,
    timer, and dot all dim together and the HUD reads as a faint ghost rather
    than a solid widget. Clamped so a typo can't make it invisible or opaque."""
    try:
        return max(0.05, min(1.0, float(os.environ.get("VOX_HUD_OPACITY", "0.5"))))
    except ValueError:
        return 0.5

# The transparent-color key: every pixel painted this exact color becomes a
# hole in the window. Chosen to be a color no HUD element ever uses.
_KEY = "#010203"
_FPS_MS = 33          # ~30 fps poll/redraw
_DONE_SECS = 1.6      # how long the result stats linger
_W, _H = 240, 44      # canvas size; roomy enough for m:ss.mmm + meter
_OFFSET = (18, 26)    # HUD position relative to the mouse cursor

# Windows extended-style bits for a ghost window.
_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_NOACTIVATE = 0x08000000
_WS_EX_TOOLWINDOW = 0x00000080


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _cursor_pos():
    pt = _Point()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


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
        else:
            cx, cy = _cursor_pos()
            x, y = cx + _OFFSET[0], cy + _OFFSET[1]
            vl, vt, vw, vh = _virtual_screen()
            x = max(vl, min(x, vl + vw - _W))
            y = max(vt, min(y, vt + vh - _H))
        self._root.geometry(f"{_W}x{_H}+{x}+{y}")

    # ---- drawing ------------------------------------------------------------

    def _text(self, x, y, s, fill, size=13, anchor="w"):
        font = ("Consolas", size, "bold")
        c = self._canvas
        c.create_text(x + 1, y + 1, text=s, fill="#000000",
                      font=font, anchor=anchor)
        c.create_text(x, y, text=s, fill=fill, font=font, anchor=anchor)

    def _draw(self, phase):
        c = self._canvas
        c.delete("all")
        y = _H // 2
        if phase == "rec":
            # gently breathing record dot (small amplitude = unobtrusive)
            pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 3.2)
            r = 4.5 + 1.0 * pulse
            c.create_oval(12 - r, y - r, 12 + r, y + r,
                          fill="#d1605b", outline="")
            # elapsed m:ss.mmm
            el = max(0.0, time.monotonic() - self.t0)
            stamp = f"{int(el // 60)}:{int(el % 60):02d}.{int(el * 1000 % 1000):03d}"
            self._text(26, y, stamp, "#f2f3f5", size=14)
            # mic meter: 7 bars over -55..-15 dBFS
            lit = max(0, min(7, int((self.level_db + 55.0) / 40.0 * 7 + 0.5)))
            for i in range(7):
                bx = 138 + i * 9
                bh = 5 + i * 2.4
                color = "#1d9e75" if i < lit else "#2c2f36"
                c.create_rectangle(bx, y + 9, bx + 6, y + 9 - bh,
                                   fill=color, outline="")
            db = f"{self.level_db:.0f}" if self.level_db > -100 else "-inf"
            self._text(206, y, db, "#8fa3ad", size=10)
        elif phase == "busy":
            # rotating spinner arc
            self._spin = (self._spin + 24) % 360
            c.create_arc(6, y - 8, 22, y + 8, start=self._spin, extent=250,
                         style="arc", outline="#e0a83a", width=3)
            self._text(30, y, "transcribing…", "#e0d7c4", size=12)
        elif phase == "done":
            self._text(8, y, "✓", "#5dcaa5", size=14)
            self._text(24, y, self.stats, "#f2f3f5", size=13)


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
