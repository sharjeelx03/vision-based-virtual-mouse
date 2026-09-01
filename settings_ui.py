"""
Settings UI for Vision-Based Virtual Mouse.

A modern tkinter-based settings window that lets users configure
camera, gesture, display, and app behaviour settings. Settings are
persisted to %APPDATA%/VirtualMouse/settings.json.

This module is imported by tray_app.py but can also be run standalone
for testing:  python settings_ui.py
"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable

# Import Config from the main module
from virtual_mouse import Config, SETTINGS_FILE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Colour theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Theme:
    """Dark-mode colour constants for the settings window."""
    BG          = "#1e1e2e"
    BG_CARD     = "#2a2a3c"
    BG_INPUT    = "#363650"
    FG          = "#e0e0e0"
    FG_DIM      = "#a0a0b0"
    ACCENT      = "#7c6ff0"
    ACCENT_HOVER = "#9585ff"
    SUCCESS     = "#50c878"
    DANGER      = "#ff6b6b"
    BORDER      = "#3a3a50"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Windows-specific: start with Windows toggle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_REG_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_REG_APP = "VirtualMouse"


def _get_startup_enabled() -> bool:
    """Check if Virtual Mouse is set to start with Windows."""
    if sys.platform != "win32":
        return False
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REG_KEY, 0, winreg.KEY_READ)
        winreg.QueryValueEx(key, _REG_APP)
        winreg.CloseKey(key)
        return True
    except (FileNotFoundError, OSError):
        return False


def _set_startup_enabled(enabled: bool, exe_path: str | None = None) -> None:
    """Enable or disable start with Windows via the registry."""
    if sys.platform != "win32":
        return
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REG_KEY, 0,
                             winreg.KEY_SET_VALUE)
        if enabled and exe_path:
            winreg.SetValueEx(key, _REG_APP, 0, winreg.REG_SZ, exe_path)
        else:
            try:
                winreg.DeleteValue(key, _REG_APP)
            except FileNotFoundError:
                pass
        winreg.CloseKey(key)
    except OSError:
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SettingsWindow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SettingsWindow:
    """
    Modern dark-themed settings GUI.

    Parameters
    ----------
    config : Config
        The current configuration to populate fields from.
    on_save : callable, optional
        Called with the new Config when the user clicks Save.
    on_close : callable, optional
        Called when the window is closed.
    """

    def __init__(self, config: Config,
                 on_save: Callable[[Config], None] | None = None,
                 on_close: Callable[[], None] | None = None) -> None:
        self.config = config
        self._on_save = on_save
        self._on_close = on_close

        self.root = tk.Tk()
        self.root.title("Virtual Mouse  -  Settings")
        self.root.geometry("520x680")
        self.root.resizable(False, False)
        self.root.configure(bg=Theme.BG)

        # Try to set icon
        try:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            ico_path = os.path.join(script_dir, "logo.ico")
            if os.path.exists(ico_path):
                self.root.iconbitmap(ico_path)
        except Exception:
            pass

        # Apply dark-mode ttk theme
        self._configure_style()

        # Build UI
        self._build_ui()

        # Handle close button
        self.root.protocol("WM_DELETE_WINDOW", self._handle_close)

    # ── ttk styling ──────────────────────────────────────────────────

    def _configure_style(self) -> None:
        """Apply a dark theme to ttk widgets."""
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(".", background=Theme.BG, foreground=Theme.FG,
                         font=("Segoe UI", 10))
        style.configure("TFrame", background=Theme.BG)
        style.configure("Card.TFrame", background=Theme.BG_CARD)
        style.configure("TLabel", background=Theme.BG, foreground=Theme.FG,
                         font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=Theme.BG_CARD)
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"),
                         foreground=Theme.ACCENT, background=Theme.BG)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"),
                         foreground=Theme.FG, background=Theme.BG)
        style.configure("TCheckbutton", background=Theme.BG_CARD,
                         foreground=Theme.FG, font=("Segoe UI", 10))
        style.map("TCheckbutton",
                  background=[("active", Theme.BG_CARD)])
        style.configure("TScale", background=Theme.BG_CARD,
                         troughcolor=Theme.BG_INPUT)
        style.configure("TCombobox", fieldbackground=Theme.BG_INPUT,
                         background=Theme.BG_INPUT, foreground=Theme.FG)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"),
                         background=Theme.ACCENT, foreground="white",
                         padding=(16, 8))
        style.map("Accent.TButton",
                  background=[("active", Theme.ACCENT_HOVER)])
        style.configure("Danger.TButton", font=("Segoe UI", 10),
                         background=Theme.DANGER, foreground="white",
                         padding=(16, 8))
        style.map("Danger.TButton",
                  background=[("active", "#ff8888")])

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Build the entire settings form."""
        # Main container with padding
        container = ttk.Frame(self.root, padding=20)
        container.pack(fill="both", expand=True)

        # Title
        ttk.Label(container, text="Virtual Mouse Settings",
                  style="Title.TLabel").pack(anchor="w", pady=(0, 16))

        # Scrollable area
        canvas = tk.Canvas(container, bg=Theme.BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical",
                                   command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # ── Camera section ───────────────────────────────────────────
        self._add_section(scroll_frame, "Camera")
        cam_card = self._add_card(scroll_frame)

        self.cam_index_var = tk.IntVar(value=self.config.cam_index)
        self._add_combobox_row(cam_card, "Camera Device:",
                               self.cam_index_var, [0, 1, 2, 3])

        self.resolution_var = tk.StringVar(
            value=f"{self.config.cam_width}x{self.config.cam_height}")
        self._add_combobox_row(cam_card, "Resolution:",
                               self.resolution_var,
                               ["640x480", "1280x720", "320x240"])

        self.start_tracking_var = tk.BooleanVar(
            value=self.config.start_tracking_on_launch)
        self._add_checkbox_row(cam_card, "Start tracking on launch",
                               self.start_tracking_var)

        # ── Gesture section ──────────────────────────────────────────
        self._add_section(scroll_frame, "Gesture Tuning")
        gesture_card = self._add_card(scroll_frame)

        self.smoothing_var = tk.DoubleVar(value=self.config.smoothing_alpha)
        self._add_slider_row(gesture_card, "Smoothing (EMA alpha):",
                             self.smoothing_var, 0.1, 1.0, 0.05,
                             fmt="{:.2f}",
                             desc="Low = smooth but slow, High = responsive but jittery")

        self.sensitivity_var = tk.IntVar(value=self.config.pinch_threshold)
        self._add_slider_row(gesture_card, "Pinch Sensitivity:",
                             self.sensitivity_var, 20, 80, 1,
                             desc="Lower = easier to trigger clicks")

        self.scroll_speed_var = tk.IntVar(value=self.config.scroll_speed)
        self._add_slider_row(gesture_card, "Scroll Speed:",
                             self.scroll_speed_var, 5, 50, 1,
                             desc="Higher = faster scrolling")

        # ── Display section ──────────────────────────────────────────
        self._add_section(scroll_frame, "Display")
        display_card = self._add_card(scroll_frame)

        self.show_guide_var = tk.BooleanVar(value=self.config.show_guide)
        self._add_checkbox_row(display_card, "Show gesture guide panel",
                               self.show_guide_var)

        self.mirror_var = tk.BooleanVar(value=self.config.mirror)
        self._add_checkbox_row(display_card, "Mirror webcam feed",
                               self.mirror_var)

        # ── General section ──────────────────────────────────────────
        self._add_section(scroll_frame, "General")
        general_card = self._add_card(scroll_frame)

        self.minimize_tray_var = tk.BooleanVar(
            value=self.config.minimize_to_tray)
        self._add_checkbox_row(general_card, "Minimize to system tray on close",
                               self.minimize_tray_var)

        self.start_windows_var = tk.BooleanVar(
            value=self.config.start_with_windows)
        self._add_checkbox_row(general_card, "Start with Windows",
                               self.start_windows_var)

        # ── Buttons ──────────────────────────────────────────────────
        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill="x", pady=(16, 0))

        ttk.Button(btn_frame, text="Reset Defaults",
                   style="Danger.TButton",
                   command=self._reset_defaults).pack(side="left")

        ttk.Button(btn_frame, text="Save Settings",
                   style="Accent.TButton",
                   command=self._save).pack(side="right")

    # ── helper widgets ───────────────────────────────────────────────

    def _add_section(self, parent: ttk.Frame, title: str) -> None:
        """Add a section header label."""
        ttk.Label(parent, text=title,
                  style="Header.TLabel").pack(anchor="w", pady=(12, 4),
                                               padx=4)

    def _add_card(self, parent: ttk.Frame) -> ttk.Frame:
        """Add a card container with rounded-corner styling."""
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.pack(fill="x", pady=(0, 4), padx=2)
        return card

    def _add_combobox_row(self, parent: ttk.Frame, label: str,
                          var: tk.Variable,
                          values: list) -> None:
        """Add a label + combobox row."""
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, style="Card.TLabel").pack(side="left")
        combo = ttk.Combobox(row, textvariable=var,
                             values=values, width=14, state="readonly")
        combo.pack(side="right")

    def _add_checkbox_row(self, parent: ttk.Frame, label: str,
                          var: tk.BooleanVar) -> None:
        """Add a checkbox row."""
        ttk.Checkbutton(parent, text=label,
                        variable=var).pack(anchor="w", pady=3)

    def _add_slider_row(self, parent: ttk.Frame, label: str,
                        var: tk.Variable,
                        from_: float, to: float, resolution: float,
                        fmt: str = "{}", desc: str = "") -> None:
        """Add a label + slider + value display row."""
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=4)

        ttk.Label(row, text=label, style="Card.TLabel").pack(anchor="w")

        if desc:
            ttk.Label(row, text=desc, style="Card.TLabel",
                      foreground=Theme.FG_DIM,
                      font=("Segoe UI", 8)).pack(anchor="w")

        slider_row = ttk.Frame(row, style="Card.TFrame")
        slider_row.pack(fill="x", pady=(2, 0))

        value_label = ttk.Label(slider_row,
                                text=fmt.format(var.get()),
                                style="Card.TLabel", width=6)
        value_label.pack(side="right")

        scale = ttk.Scale(slider_row, from_=from_, to=to,
                          variable=var, orient="horizontal",
                          command=lambda v: value_label.configure(
                              text=fmt.format(float(v) if "." in fmt else int(float(v)))))
        scale.pack(side="left", fill="x", expand=True, padx=(0, 8))

    # ── actions ──────────────────────────────────────────────────────

    def _build_config(self) -> Config:
        """Read all UI fields and return a Config."""
        res = self.resolution_var.get()
        try:
            w, h = res.split("x")
            cam_w, cam_h = int(w), int(h)
        except ValueError:
            cam_w, cam_h = 640, 480

        return Config(
            cam_index=self.cam_index_var.get(),
            cam_width=cam_w,
            cam_height=cam_h,
            smoothing_alpha=round(self.smoothing_var.get(), 2),
            pinch_threshold=int(self.sensitivity_var.get()),
            scroll_speed=int(self.scroll_speed_var.get()),
            show_guide=self.show_guide_var.get(),
            mirror=self.mirror_var.get(),
            start_tracking_on_launch=self.start_tracking_var.get(),
            minimize_to_tray=self.minimize_tray_var.get(),
            start_with_windows=self.start_windows_var.get(),
        )

    def _save(self) -> None:
        """Save current settings and optionally notify caller."""
        new_cfg = self._build_config()
        new_cfg.save()

        # Handle start-with-Windows registry
        _set_startup_enabled(new_cfg.start_with_windows, sys.executable)

        if self._on_save:
            self._on_save(new_cfg)

        messagebox.showinfo("Settings Saved",
                            "Your settings have been saved successfully.\n\n"
                            "Some changes will take effect next time you "
                            "start tracking.",
                            parent=self.root)

    def _reset_defaults(self) -> None:
        """Reset all fields to default values."""
        defaults = Config()
        self.cam_index_var.set(defaults.cam_index)
        self.resolution_var.set(f"{defaults.cam_width}x{defaults.cam_height}")
        self.smoothing_var.set(defaults.smoothing_alpha)
        self.sensitivity_var.set(defaults.pinch_threshold)
        self.scroll_speed_var.set(defaults.scroll_speed)
        self.show_guide_var.set(defaults.show_guide)
        self.mirror_var.set(defaults.mirror)
        self.start_tracking_var.set(defaults.start_tracking_on_launch)
        self.minimize_tray_var.set(defaults.minimize_to_tray)
        self.start_windows_var.set(defaults.start_with_windows)

    def _handle_close(self) -> None:
        """Handle window close event."""
        if self._on_close:
            self._on_close()
        self.root.destroy()

    # ── public API ───────────────────────────────────────────────────

    def show(self) -> None:
        """Show the settings window (blocking mainloop)."""
        self.root.mainloop()

    def show_nonblocking(self) -> None:
        """Make the window visible without starting a new mainloop."""
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def hide(self) -> None:
        """Hide the window without destroying it."""
        self.root.withdraw()

    def destroy(self) -> None:
        """Destroy the window."""
        try:
            self.root.destroy()
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Standalone test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    cfg = Config.load()

    def on_save(new_cfg: Config):
        print(f"Settings saved: {new_cfg.to_dict()}")

    win = SettingsWindow(cfg, on_save=on_save)
    win.show()
