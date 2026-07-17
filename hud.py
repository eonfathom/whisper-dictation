"""Floating dictation HUD for Vox (Windows).

A tiny always-on-top overlay pinned to the bottom-right of the monitor you
are dictating into - the one holding the focused window at chord-press
(Wispr Flow style; VOX_HUD_ANCHOR chooses caret/cursor tracking instead) while
you hold the push-to-talk chord: a red plus, a live elapsed timer (seconds)
and a mic level meter. On release the timer turns into a yellow
release-to-paste latency clock, then freezes that latency plus the word/token
count for a moment and disappears.

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

Thin light text (Segoe UI Light), no panel. Everything is env-tunable - see the
VOX_HUD_* config block below. Disable entirely with VOX_HUD=0.
"""

import ctypes
import math
import os
import time

IS_WINDOWS = os.name == "nt"

# Make the whole process per-monitor-DPI-aware BEFORE any window exists.
# Without this, Windows DPI-virtualizes tkinter's coordinates while the caret
# and cursor positions arrive in (differently scaled) physical pixels - on a
# mixed-DPI multi-monitor setup the crosshair lands visibly off target, and by
# a different amount per monitor. Aware = one physical coordinate space
# everywhere. Drawing sizes are compensated by the DPI scale factor (self._k).
if IS_WINDOWS:
    try:
        # -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 (Win10 1703+)
        ctypes.windll.user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(-4))
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass

ENABLED = os.environ.get("VOX_HUD", "1").strip().lower() not in (
    "0", "false", "off", "no",
)
# Placement: "corner" (default) pins the HUD to a fixed spot at the
# bottom-right of the monitor being dictated into (the focused window's
# monitor, sampled at chord-press), just above the taskbar - Wispr style. One
# predictable place, zero per-frame position tracking (the caret/cursor modes
# visibly lagged the pointer, and the caret is invisible to the OS in
# Electron/Chromium apps anyway, so "anchored to the text" mostly wasn't).
# "caret" anchors to the text insertion point where available, falling back to
# a static position captured at record-start; "cursor" follows the mouse live.
ANCHOR = os.environ.get("VOX_HUD_ANCHOR", "corner").strip().lower()

# Thin by design: Segoe UI Light, small, normal weight, no shadow. All tunable
# live via env so styling doesn't need a code change.
FONT = os.environ.get("VOX_HUD_FONT", "Segoe UI Light")
try:
    SIZE = int(os.environ.get("VOX_HUD_SIZE", "13"))  # pixels (tk negative size)
except ValueError:
    SIZE = 13
# Leading marker: a red plus/crosshair that lands on the caret (VOX_HUD_PLUS=1,
# default). VOX_HUD_DOT=1 uses a small static dot instead; set both off for none.
PLUS = os.environ.get("VOX_HUD_PLUS", "1").strip().lower() in ("1", "true", "on", "yes")
SHOW_DOT = os.environ.get("VOX_HUD_DOT", "0").strip().lower() in ("1", "true", "on", "yes")
try:
    PLUS_SIZE = int(os.environ.get("VOX_HUD_PLUS_SIZE", "9"))    # full arm length, px
except ValueError:
    PLUS_SIZE = 9
try:
    PLUS_THICK = int(os.environ.get("VOX_HUD_PLUS_THICK", "3"))  # line thickness, px
except ValueError:
    PLUS_THICK = 3


def _int_env(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Two INDEPENDENT offsets, both relative to the caret (0,0 = right on it):
# the plus marker, and the timer/meter group. So you can pin the crosshair on
# the caret and place the readout wherever you like, separately.
PLUS_DX = _int_env("VOX_HUD_PLUS_DX", 0)
PLUS_DY = _int_env("VOX_HUD_PLUS_DY", 0)
CARET_DX = _int_env("VOX_HUD_DX", 16)   # timer group, right of the caret
CARET_DY = _int_env("VOX_HUD_DY", 0)

# Corner-mode margins (96dpi px, DPI-scaled): where the plus sits relative to
# the primary work area's bottom-right corner. MX leaves room to the RIGHT of
# the plus for the timer/meter/result readout (~150px worst case) so nothing
# runs off screen; MY floats it just above the taskbar (the work area already
# excludes the taskbar itself).
CORNER_MX = _int_env("VOX_HUD_CORNER_MX", 200)
CORNER_MY = _int_env("VOX_HUD_CORNER_MY", 24)


def _opacity():
    """Whole-overlay opacity, 0.05..1.0 (VOX_HUD_OPACITY, default 1.0).

    Applied as a layered-window alpha over everything drawn. The thin light text
    is already subtle; lower this if you want it fainter. Clamped so a typo can't
    make it invisible or fully opaque-and-heavy."""
    try:
        return max(0.05, min(1.0, float(os.environ.get("VOX_HUD_OPACITY", "0.9"))))
    except ValueError:
        return 0.9

# The transparent-color key: every pixel painted this exact color becomes a
# hole in the window. Chosen to be a color no HUD element ever uses.
_KEY = "#010203"
_FPS_MS = 33          # ~30 fps poll/redraw
# How long the frozen result readout (latency + word count) stays on screen
# after the paste, then auto-clears. NOT the latency itself - just the linger.
# VOX_HUD_DONE_SECS=0 clears it immediately (no result flash).
try:
    _DONE_SECS = max(0.0, float(os.environ.get("VOX_HUD_DONE_SECS", "0.5")))
except ValueError:
    _DONE_SECS = 0.5
# The window is a generous transparent canvas (click-through, so the empty
# space costs nothing) with the caret mapped to a fixed interior point; both
# the plus and the timer are then drawn at their own caret-relative offsets.
_W, _H = 320, 80
_CAX, _CAY = 64, 40   # where the caret maps inside the window
# No shift in the static fallback: when there's no OS caret (Electron apps -
# browsers, Obsidian, Claude), the plus should sit ON the mouse cursor, not
# beside it. Cursor-follow mode keeps a small offset so it clears the pointer.
_OFFSET = (0, 0)
_CURSOR_OFFSET = (18, 26)

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
        mid_y = (gti.rcCaret.top + gti.rcCaret.bottom) // 2
        pt = _Point(gti.rcCaret.left, mid_y)
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


def _work_area():
    """Primary monitor work area (left, top, right, bottom) - the desktop
    minus the taskbar, so corner placement clears it at any taskbar size or
    position instead of guessing with a fixed offset."""
    r = _Rect()
    if ctypes.windll.user32.SystemParametersInfoW(
            0x0030, 0, ctypes.byref(r), 0):  # SPI_GETWORKAREA
        return r.left, r.top, r.right, r.bottom
    m = ctypes.windll.user32.GetSystemMetrics
    return 0, 0, m(0), m(1)  # SM_CXSCREEN, SM_CYSCREEN


class _MonitorInfo(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", _Rect),
                ("rcWork", _Rect), ("dwFlags", ctypes.c_ulong)]


def _focus_work_area():
    """Work area of the monitor holding the FOCUSED window - i.e. the one
    being dictated into - so on a multi-monitor desktop the corner HUD shows
    up where you're actually looking, not on the primary screen. Falls back
    to the mouse's monitor (no foreground window), then the primary."""
    try:
        u = ctypes.windll.user32
        hmon = None
        fg = u.GetForegroundWindow()
        if fg:
            hmon = u.MonitorFromWindow(fg, 2)  # MONITOR_DEFAULTTONEAREST
        if not hmon:
            pt = _Point()
            u.GetCursorPos(ctypes.byref(pt))
            hmon = u.MonitorFromPoint(pt, 2)
        if hmon:
            mi = _MonitorInfo()
            mi.cbSize = ctypes.sizeof(mi)
            if u.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                r = mi.rcWork
                return r.left, r.top, r.right, r.bottom
    except Exception:
        pass
    return _work_area()


class Hud:
    """State mailbox + tk thread. See module docstring for the contract."""

    def __init__(self, logger=None):
        self._log = logger or (lambda msg: None)
        # -- written by the app threads, read by the tk thread --
        self.phase = "idle"        # idle | rec | busy | done
        self.t0 = 0.0              # monotonic start of recording
        self.level_db = -120.0     # latest mic RMS in dBFS
        self._busy_t0 = 0.0        # monotonic when transcription began (latency clock)
        self._latency = 0.0        # measured release->paste seconds, shown on done
        self._words = 0
        self._tokens = None
        self._done_at = 0.0
        self._anchor = (0, 0)      # static fallback position (mouse at rec start)
        self._corner_wa = None     # work area of the dictated-into monitor
        # DPI scale + scaled geometry; real values set in _run once the window
        # exists. 1.0/base defaults keep direct _draw use (tests, demo) working.
        self._k = 1.0
        self._w, self._h = _W, _H
        self._cax, self._cay = _CAX, _CAY
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
        # instead of chasing the mouse. The corner work area is likewise
        # captured once per session: the monitor focused at chord-press is the
        # one being dictated into, and a mid-dictation focus change must not
        # make the HUD jump screens.
        try:
            self._anchor = _cursor_pos() if IS_WINDOWS else (0, 0)
            self._corner_wa = _focus_work_area() if IS_WINDOWS else None
        except Exception:
            self._anchor = (0, 0)
            self._corner_wa = None
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
            self._busy_t0 = time.monotonic()  # start the release->paste latency clock
            self.phase = "busy"

    def done(self, words, secs, tokens=None, gen=None):
        if gen is not None and gen != self._gen:
            return
        self._latency = (time.monotonic() - self._busy_t0) if self._busy_t0 else 0.0
        self._words = words
        self._tokens = tokens
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
            root.withdraw()
            root.update_idletasks()
            # DPI scale for this window's monitor: the process is per-monitor
            # aware (coordinates are physical), so drawing sizes must scale up
            # by dpi/96 to keep the visual size that was tuned at 96dpi. One
            # factor, sampled at startup - a monitor-to-monitor DPI change
            # mid-recording keeps position exact and only mis-sizes slightly.
            try:
                hwnd = ctypes.windll.user32.GetParent(
                    root.winfo_id()) or root.winfo_id()
                self._k = ctypes.windll.user32.GetDpiForWindow(hwnd) / 96.0
            except Exception:
                self._k = 1.0
            if not (0.5 <= self._k <= 4.0):
                self._k = 1.0
            self._w = int(_W * self._k)
            self._h = int(_H * self._k)
            self._cax = int(_CAX * self._k)
            self._cay = int(_CAY * self._k)
            self._canvas = tk.Canvas(
                root, width=self._w, height=self._h, bg=_KEY,
                highlightthickness=0,
            )
            self._canvas.pack()
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
            # Fixed spot: the plus lands CORNER_MX/MY (scaled) inside the
            # bottom-right corner of the dictated-into monitor's work area
            # (captured at record-start); timer/meter grow rightward into that
            # reserved margin. Static by design - nothing to track, no lag.
            _, _, wr, wb = self._corner_wa or _focus_work_area()
            ax = wr - int(CORNER_MX * self._k)
            ay = wb - int(CORNER_MY * self._k)
            x, y = ax - self._cax, ay - self._cay
        elif ANCHOR == "cursor":
            cx, cy = _cursor_pos()
            ax = cx + int(_CURSOR_OFFSET[0] * self._k)
            ay = cy + int(_CURSOR_OFFSET[1] * self._k)
            x, y = ax - self._cax, ay - self._cay
        else:  # caret: track the text insertion point, static fallback
            caret = _caret_pos()
            ax, ay = caret if caret else (self._anchor[0] + _OFFSET[0],
                                          self._anchor[1] + _OFFSET[1])
            # Map the caret to the window's interior anchor; the plus and timer
            # are drawn at their own offsets from there.
            x, y = ax - self._cax, ay - self._cay
        vl, vt, vw, vh = _virtual_screen()
        x = max(vl, min(x, vl + vw - self._w))
        y = max(vt, min(y, vt + vh - self._h))
        self._root.geometry(f"{self._w}x{self._h}+{x}+{y}")

    # ---- drawing ------------------------------------------------------------

    def _text(self, x, y, s, fill, size=None, anchor="w"):
        """Thin light text: configured font, normal weight, no shadow. Negative
        tk font size = PHYSICAL pixels (scaled by the DPI factor so the visual
        size matches what was tuned at 96dpi). Returns the right edge x."""
        px = int((size or SIZE) * self._k)
        item = self._canvas.create_text(
            x, y, text=s, fill=fill, font=(FONT, -px), anchor=anchor)
        return self._canvas.bbox(item)[2]

    def _tabular(self, x, y, s, fill, cell):
        """Draw each glyph centered in a fixed-width cell, so a digit changing
        (1 -> 2) never shifts the string or anything after it, and the number
        gets even spacing. Callers reserve a fixed field width so downstream
        elements never move even when the number gains a digit."""
        px = int(SIZE * self._k)
        for i, ch in enumerate(s):
            self._canvas.create_text(
                x + i * cell + cell / 2.0, y, text=ch, fill=fill,
                font=(FONT, -px), anchor="c")

    def _plus(self):
        """Draw the caret marker at its own offset: a red plus (default), a
        static dot, or nothing. Independent of the timer group."""
        k = self._k
        cx = self._cax + PLUS_DX * k
        cy = self._cay + PLUS_DY * k
        if PLUS:
            half = PLUS_SIZE * k / 2.0
            w = max(1, round(PLUS_THICK * k))
            c = self._canvas
            c.create_line(cx, cy - half, cx, cy + half, fill="#e2504a",
                          width=w, capstyle="projecting")
            c.create_line(cx - half, cy, cx + half, cy, fill="#e2504a",
                          width=w, capstyle="projecting")
        elif SHOW_DOT:
            r = SIZE * k * 0.34
            self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                     fill="#d1605b", outline="")

    def _draw(self, phase):
        c = self._canvas
        c.delete("all")
        u = float(SIZE) * self._k
        cell = u * 0.66
        self._plus()
        # timer/meter group at its own independent caret-relative offset
        tx = self._cax + CARET_DX * self._k
        ty = self._cay + CARET_DY * self._k
        if phase == "rec":
            # seconds only, tenths, no leading zero / no minutes: 7.4, 92.6, 128.1
            el = max(0.0, time.monotonic() - self.t0)
            self._tabular(tx, ty, f"{el:.1f}", "#f2f3f5", cell)
            # meter at a FIXED offset (5-cell field) so it never shifts as the
            # number grows past 10s / 100s
            mx = tx + 5 * cell + u * 0.6
            step = u * 0.5
            lit = max(0, min(5, int((self.level_db + 55.0) / 40.0 * 5 + 0.5)))
            for i in range(5):
                bx = mx + i * step
                bh = u * 0.36 + i * (u * 0.28)
                color = "#1d9e75" if i < lit else "#2c2f36"
                c.create_rectangle(bx, ty + u * 0.7, bx + max(1.0, u * 0.28),
                                   ty + u * 0.7 - bh, fill=color, outline="")
            if self.level_db > -100:
                self._text(mx + 5 * step + u * 0.7, ty, f"{self.level_db:.0f}",
                           "#8fa3ad", size=max(7, int(SIZE * 0.82)))
        elif phase == "busy":
            lat = max(0.0, time.monotonic() - self._busy_t0)
            self._tabular(tx, ty, f"{lat:.1f}", "#e0a83a", cell)
        elif phase == "done":
            xend = self._text(tx, ty, f"{self._latency:.1f}s", "#e0a83a")
            rest = f"  {self._words} w"
            if self._tokens:
                rest += f"  {self._tokens} tok"
            self._text(xend + u * 0.3, ty, rest, "#c9ccd1")


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
