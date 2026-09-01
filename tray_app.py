"""
System Tray Application for Vision-Based Virtual Mouse.

This is the main entry point for the desktop-app version.
It creates a system-tray icon with controls to start/stop
camera tracking and open the settings window.

Usage:
    python tray_app.py          (run from source)
    VirtualMouse.exe            (after PyInstaller build)

The camera tracking runs in a background thread so the tray
stays responsive.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pystray
from PIL import Image, ImageDraw

from virtual_mouse import Config, VirtualMouse
from settings_ui import SettingsWindow


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Icon generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _load_icon() -> Image.Image:
    """
    Load the project logo for the tray icon.

    Falls back to a generated icon if logo.png is not found.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ("logo.png", "logo.ico"):
        path = os.path.join(script_dir, name)
        if os.path.exists(path):
            try:
                img = Image.open(path)
                img = img.resize((64, 64), Image.LANCZOS)
                return img
            except Exception:
                pass

    # Fallback: generate a simple icon
    return _generate_icon()


def _generate_icon(color: str = "#7c6ff0") -> Image.Image:
    """Generate a simple hand-cursor icon as fallback."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Circle background
    draw.ellipse([4, 4, 60, 60], fill=color, outline="#ffffff", width=2)
    # Simple hand shape
    draw.rectangle([24, 14, 30, 34], fill="white")  # index finger
    draw.rectangle([32, 18, 38, 34], fill="white")  # middle
    draw.rectangle([16, 18, 22, 34], fill="white")  # ring
    draw.ellipse([18, 32, 42, 50], fill="white")     # palm
    return img


def _generate_icon_active() -> Image.Image:
    """Green icon for active tracking state."""
    return _generate_icon("#50c878")


def _generate_icon_inactive() -> Image.Image:
    """Gray icon for inactive state."""
    return _generate_icon("#666680")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TrayApp
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TrayApp:
    """
    System tray application that manages camera tracking.

    Responsibilities:
    - System tray icon with right-click menu
    - Start / Stop camera tracking in a background thread
    - Open the Settings window
    - Quit the application
    """

    def __init__(self) -> None:
        # Load saved settings (or defaults)
        self.config: Config = Config.load()

        # Tracking state
        self._mouse: VirtualMouse | None = None
        self._tracking_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._tracking_active: bool = False

        # Settings window reference (so we only create one)
        self._settings_window: SettingsWindow | None = None
        self._settings_thread: threading.Thread | None = None

        # Tray icon
        self._icon_default = _load_icon()
        self._icon_active = _generate_icon_active()
        self._icon_inactive = _generate_icon_inactive()

        self._icon = pystray.Icon(
            name="VirtualMouse",
            title="Virtual Mouse (Stopped)",
            icon=self._icon_inactive,
            menu=self._build_menu(),
        )

    # ── menu ─────────────────────────────────────────────────────────

    def _build_menu(self) -> pystray.Menu:
        """Build the right-click context menu."""
        return pystray.Menu(
            pystray.MenuItem(
                text=lambda _: "Stop Tracking" if self._tracking_active
                               else "Start Tracking",
                action=self._toggle_tracking,
                default=True,  # double-click action
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings", self._open_settings),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )

    # ── tracking control ─────────────────────────────────────────────

    def _start_tracking(self) -> None:
        """Start camera tracking in a background thread."""
        if self._tracking_active:
            return

        # Reload config in case settings were changed
        self.config = Config.load()
        self._stop_event.clear()

        self._mouse = VirtualMouse(self.config)
        self._tracking_thread = threading.Thread(
            target=self._run_tracking,
            daemon=True,
            name="TrackingThread",
        )
        self._tracking_active = True
        self._tracking_thread.start()

        # Update icon
        self._icon.icon = self._icon_active
        self._icon.title = "Virtual Mouse (Tracking)"
        self._icon.update_menu()

        print("  [TRAY] Tracking started")

    def _stop_tracking(self) -> None:
        """Stop camera tracking."""
        if not self._tracking_active:
            return

        self._stop_event.set()

        if self._mouse:
            self._mouse.request_stop()

        # Wait for the thread to finish (with timeout)
        if self._tracking_thread and self._tracking_thread.is_alive():
            self._tracking_thread.join(timeout=5.0)

        self._tracking_active = False
        self._mouse = None
        self._tracking_thread = None

        # Update icon
        self._icon.icon = self._icon_inactive
        self._icon.title = "Virtual Mouse (Stopped)"
        self._icon.update_menu()

        print("  [TRAY] Tracking stopped")

    def _toggle_tracking(self, icon=None, item=None) -> None:
        """Toggle tracking on/off."""
        if self._tracking_active:
            self._stop_tracking()
        else:
            self._start_tracking()

    def _run_tracking(self) -> None:
        """Worker function for the tracking thread."""
        try:
            if self._mouse:
                self._mouse.run(stop_event=self._stop_event)
        except Exception as e:
            print(f"  [TRAY] Tracking error: {e}")
        finally:
            self._tracking_active = False
            # Update icon from the thread
            try:
                self._icon.icon = self._icon_inactive
                self._icon.title = "Virtual Mouse (Stopped)"
                self._icon.update_menu()
            except Exception:
                pass

    # ── settings ─────────────────────────────────────────────────────

    def _open_settings(self, icon=None, item=None) -> None:
        """Open the settings window in a separate thread."""
        # Run tkinter in a new thread to avoid blocking the tray
        if (self._settings_thread is not None
                and self._settings_thread.is_alive()):
            # Settings window already open — try to bring it to front
            return

        self._settings_thread = threading.Thread(
            target=self._run_settings_window,
            daemon=True,
            name="SettingsThread",
        )
        self._settings_thread.start()

    def _run_settings_window(self) -> None:
        """Create and show the settings window (runs in its own thread)."""
        cfg = Config.load()
        win = SettingsWindow(
            config=cfg,
            on_save=self._on_settings_saved,
        )
        win.show()

    def _on_settings_saved(self, new_cfg: Config) -> None:
        """Called when settings are saved from the settings window."""
        self.config = new_cfg
        print(f"  [TRAY] Settings updated: smoothing={new_cfg.smoothing_alpha}, "
              f"sensitivity={new_cfg.pinch_threshold}, "
              f"scroll_speed={new_cfg.scroll_speed}")

    # ── quit ─────────────────────────────────────────────────────────

    def _quit(self, icon=None, item=None) -> None:
        """Stop everything and exit."""
        print("  [TRAY] Quitting...")
        self._stop_tracking()
        self._icon.stop()

    # ── run ──────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the tray application.

        If 'start_tracking_on_launch' is enabled in settings,
        tracking starts automatically.
        """
        print()
        print("  +=============================================+")
        print("  |    Virtual Mouse  v2.0  (System Tray)       |")
        print("  +=============================================+")
        print("  |  Right-click the tray icon for options      |")
        print("  |  Double-click to start/stop tracking        |")
        print("  +=============================================+")
        print()

        # Auto-start tracking if configured
        if self.config.start_tracking_on_launch:
            # Small delay so the tray icon appears first
            def _delayed_start():
                time.sleep(1.0)
                self._start_tracking()

            threading.Thread(target=_delayed_start, daemon=True).start()

        # This blocks — runs the tray icon event loop
        self._icon.run()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """Entry point for the tray application."""
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0

    app = TrayApp()
    app.run()


if __name__ == "__main__":
    main()
