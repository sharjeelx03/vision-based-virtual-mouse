"""
Vision-Based Virtual Mouse
===========================
Control your computer cursor with hand gestures using your webcam.

Supported Gestures
------------------
• MOVE          – Index finger up            → move cursor
• LEFT CLICK    – Thumb + Index pinch        → left click
• RIGHT CLICK   – Thumb + Middle pinch       → right click
• DOUBLE CLICK  – Thumb + Ring pinch         → double click
• SCROLL        – Index + Middle up together → scroll by moving hand
• DRAG & DROP   – Pinch & hold > 0.5 s      → hold mouse, release to drop
• PAUSE         – All five fingers open      → toggle tracking on/off

Architecture
------------
• HandDetector  – wraps MediaPipe Tasks HandLandmarker; returns pixel-space
                  landmarks and finger-up state.
• GestureEngine – maps finger states to a discrete gesture enum.
• VirtualMouse  – main loop: reads frames, runs gesture engine, executes
                  OS actions, draws the full HUD overlay.

Performance target: >= 30 FPS on an Intel i7 with a 640x480 webcam feed.

Author : Sharjeel
Python : 3.10+
License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import math
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    HandLandmarksConnections,
    RunningMode,
    drawing_utils,
)
import pyautogui


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Gesture enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Gesture(Enum):
    """All recognised hand gestures."""
    NONE        = auto()   # no hand or unrecognised pose
    MOVE        = auto()   # index finger only  → cursor movement
    LEFT_CLICK  = auto()   # thumb + index pinch
    RIGHT_CLICK = auto()   # thumb + middle pinch
    DOUBLE_CLICK = auto()  # thumb + ring pinch
    SCROLL      = auto()   # index + middle up   → scroll mode
    DRAG        = auto()   # pinch held > threshold
    PAUSE       = auto()   # all 5 fingers open  → toggle pause


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@dataclass
class Config:
    """Every tunable parameter in one place."""

    # ── Camera ───────────────────────────────────────────────────────
    cam_width: int = 640
    cam_height: int = 480
    cam_index: int = 0

    # ── Model ────────────────────────────────────────────────────────
    model_path: str = "hand_landmarker.task"

    # ── Reduction box ────────────────────────────────────────────────
    frame_margin_x: int = 110
    frame_margin_y: int = 80

    # ── Smoothing (EMA) ──────────────────────────────────────────────
    smoothing_alpha: float = 0.35

    # ── Click / gesture thresholds ───────────────────────────────────
    pinch_threshold: int = 40         # px – pinch distance
    click_cooldown: float = 0.40      # seconds between clicks
    drag_hold_time: float = 0.50      # seconds pinch must hold for drag
    scroll_speed: int = 15            # pyautogui scroll units per tick
    pause_cooldown: float = 1.0       # seconds between pause toggles

    # ── Display ──────────────────────────────────────────────────────
    show_guide: bool = True           # show gesture guide panel
    mirror: bool = True               # mirror the webcam feed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Colour palette (BGR)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Colors:
    """Centralised colour constants (BGR format for OpenCV)."""
    CYAN        = (255, 255, 0)
    GREEN       = (0, 255, 0)
    RED         = (0, 0, 255)
    ORANGE      = (0, 140, 255)
    MAGENTA     = (255, 0, 200)
    YELLOW      = (0, 255, 255)
    WHITE       = (255, 255, 255)
    LIGHT_GRAY  = (200, 200, 200)
    DARK_GRAY   = (60, 60, 60)
    BLUE        = (255, 160, 50)
    PURPLE      = (200, 80, 200)
    TEAL        = (200, 200, 0)

    # Per-gesture accent colours
    GESTURE = {
        Gesture.MOVE:         (255, 255, 0),     # cyan
        Gesture.LEFT_CLICK:   (0, 255, 0),       # green
        Gesture.RIGHT_CLICK:  (0, 0, 255),       # red
        Gesture.DOUBLE_CLICK: (0, 140, 255),     # orange
        Gesture.SCROLL:       (255, 0, 200),      # magenta
        Gesture.DRAG:         (0, 255, 255),      # yellow
        Gesture.PAUSE:        (200, 200, 200),    # gray
        Gesture.NONE:         (100, 100, 100),    # dim gray
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HandDetector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HandDetector:
    """
    Wraps MediaPipe Tasks HandLandmarker (v1.0+).

    Returns pixel-space landmarks and per-finger up/down state.
    Uses the synchronous VIDEO running mode with monotonically
    increasing timestamps.
    """

    # ── Landmark indices ─────────────────────────────────────────────
    # Tips
    THUMB_TIP   = 4
    INDEX_TIP   = 8
    MIDDLE_TIP  = 12
    RING_TIP    = 16
    PINKY_TIP   = 20

    # PIP joints (one joint below the tip)
    THUMB_IP    = 3      # for thumb we compare tip.x vs IP.x
    INDEX_PIP   = 6
    MIDDLE_PIP  = 10
    RING_PIP    = 14
    PINKY_PIP   = 18

    # MCP (knuckle) for thumb direction
    THUMB_MCP   = 2

    def __init__(self, model_path: str,
                 detection_confidence: float = 0.70,
                 tracking_confidence: float = 0.70) -> None:
        # Resolve model path relative to this script's directory.
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, model_path)

        if not os.path.exists(model_path):
            print(f"\n  ERROR: Model file not found at:\n  {model_path}\n")
            print("  Download it by running:\n")
            if sys.platform == "win32":
                print('  Invoke-WebRequest -Uri "https://storage.googleapis.com/'
                      'mediapipe-models/hand_landmarker/hand_landmarker/float16/'
                      'latest/hand_landmarker.task" -OutFile "hand_landmarker.task"')
            else:
                print('  curl -L -o hand_landmarker.task "https://storage.googleapis.com/'
                      'mediapipe-models/hand_landmarker/hand_landmarker/float16/'
                      'latest/hand_landmarker.task"')
            sys.exit(1)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_ts_ms: int = 0
        self._last_result: HandLandmarkerResult | None = None

    # ── detection ────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int]]:
        """
        Run detection on a BGR frame.

        Returns list of (landmark_id, pixel_x, pixel_y) for the first
        detected hand, or an empty list.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._frame_ts_ms += 33   # ~30 FPS monotonic step
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        self._last_result = result

        landmarks: list[tuple[int, int, int]] = []
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            for idx, lm in enumerate(hand):
                px, py = int(lm.x * w), int(lm.y * h)
                landmarks.append((idx, px, py))
        return landmarks

    # ── finger state ─────────────────────────────────────────────────

    @staticmethod
    def fingers_up(lms: list[tuple[int, int, int]]) -> list[bool]:
        """
        Determine which fingers are extended (up).

        Returns [thumb, index, middle, ring, pinky] as booleans.

        For fingers 1-4 we compare tip.y < pip.y (since y grows
        downward in image space, a lower y means the fingertip is
        above the knuckle → finger is up).

        For the thumb we compare tip.x vs IP.x. Because the image
        is mirrored, thumb-out means tip.x < ip.x for a right hand
        (which appears on the left side of the mirrored frame).
        We use the simple heuristic: abs(tip.x - ip.x) > 20 and
        tip.x != ip.x, checking which side the thumb extends.
        """
        if len(lms) < 21:
            return [False] * 5

        fingers: list[bool] = []

        # Thumb: use x-axis distance between tip and IP joint.
        # In a mirrored feed the thumb sticks out laterally.
        thumb_tip_x = lms[HandDetector.THUMB_TIP][1]
        thumb_ip_x  = lms[HandDetector.THUMB_IP][1]
        thumb_mcp_x = lms[HandDetector.THUMB_MCP][1]
        # Thumb is "up" if tip extends away from palm center.
        fingers.append(abs(thumb_tip_x - thumb_ip_x) > 20 and
                       abs(thumb_tip_x - thumb_mcp_x) > abs(thumb_ip_x - thumb_mcp_x))

        # Index, Middle, Ring, Pinky: tip.y < pip.y means finger is up.
        tip_ids = [HandDetector.INDEX_TIP, HandDetector.MIDDLE_TIP,
                   HandDetector.RING_TIP,  HandDetector.PINKY_TIP]
        pip_ids = [HandDetector.INDEX_PIP, HandDetector.MIDDLE_PIP,
                   HandDetector.RING_PIP,  HandDetector.PINKY_PIP]

        for tip_id, pip_id in zip(tip_ids, pip_ids):
            fingers.append(lms[tip_id][2] < lms[pip_id][2])

        return fingers

    # ── drawing ──────────────────────────────────────────────────────

    def draw_skeleton(self, frame: np.ndarray) -> None:
        """Draw the MediaPipe hand skeleton overlay."""
        if self._last_result and self._last_result.hand_landmarks:
            for hand_lms in self._last_result.hand_landmarks:
                drawing_utils.draw_landmarks(
                    frame, hand_lms,
                    HandLandmarksConnections.HAND_CONNECTIONS,
                )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GestureEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class GestureEngine:
    """
    Maps finger-up states and pinch distances to a Gesture enum.

    Priority order (first match wins):
    1. All 5 up          → PAUSE toggle
    2. Index+Middle up   → SCROLL
    3. Thumb+Index pinch → LEFT_CLICK / DRAG (if held)
    4. Thumb+Middle pinch→ RIGHT_CLICK
    5. Thumb+Ring pinch  → DOUBLE_CLICK
    6. Index only up     → MOVE
    7. Otherwise         → NONE
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._pinch_start_time: float = 0.0
        self._is_pinching: bool = False

    @staticmethod
    def _dist(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        """Euclidean pixel distance between two landmarks."""
        return math.hypot(a[1] - b[1], a[2] - b[2])

    def classify(self, lms: list[tuple[int, int, int]],
                 fingers: list[bool]) -> tuple[Gesture, dict]:
        """
        Classify the current hand pose into a Gesture.

        Returns (gesture, metadata) where metadata contains extra info
        like pinch distances for the HUD.
        """
        meta: dict = {"pinch_dist": 0.0, "scroll_dy": 0}

        if not lms or len(lms) < 21:
            self._is_pinching = False
            return Gesture.NONE, meta

        thumb  = lms[HandDetector.THUMB_TIP]
        index  = lms[HandDetector.INDEX_TIP]
        middle = lms[HandDetector.MIDDLE_TIP]
        ring   = lms[HandDetector.RING_TIP]

        pinch_index  = self._dist(thumb, index)
        pinch_middle = self._dist(thumb, middle)
        pinch_ring   = self._dist(thumb, ring)

        meta["pinch_dist"] = pinch_index

        # ── 1. PAUSE: all five fingers open ──────────────────────────
        if all(fingers):
            self._is_pinching = False
            return Gesture.PAUSE, meta

        # ── 2. SCROLL: index + middle up, others down ────────────────
        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            self._is_pinching = False
            return Gesture.SCROLL, meta

        # ── 3. Thumb + Index pinch → LEFT_CLICK or DRAG ─────────────
        if pinch_index < self.cfg.pinch_threshold:
            now = time.time()
            if not self._is_pinching:
                self._is_pinching = True
                self._pinch_start_time = now

            hold_duration = now - self._pinch_start_time
            if hold_duration >= self.cfg.drag_hold_time:
                return Gesture.DRAG, meta
            else:
                return Gesture.LEFT_CLICK, meta

        # ── 4. Thumb + Middle pinch → RIGHT_CLICK ────────────────────
        if pinch_middle < self.cfg.pinch_threshold:
            self._is_pinching = False
            return Gesture.RIGHT_CLICK, meta

        # ── 5. Thumb + Ring pinch → DOUBLE_CLICK ─────────────────────
        if pinch_ring < self.cfg.pinch_threshold:
            self._is_pinching = False
            return Gesture.DOUBLE_CLICK, meta

        # Reset pinch state when fingers separate.
        self._is_pinching = False

        # ── 6. MOVE: index finger only ───────────────────────────────
        if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
            return Gesture.MOVE, meta

        return Gesture.NONE, meta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HUD (Heads-Up Display) renderer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class HUD:
    """Draws all visual overlays on the webcam frame."""

    # Gesture guide entries: (label, description, colour)
    GUIDE_ENTRIES = [
        ("INDEX UP",         "Move cursor",   Colors.CYAN),
        ("THUMB+INDEX",      "Left click",    Colors.GREEN),
        ("THUMB+MIDDLE",     "Right click",   Colors.RED),
        ("THUMB+RING",       "Double click",  Colors.ORANGE),
        ("INDEX+MIDDLE UP",  "Scroll",        Colors.MAGENTA),
        ("PINCH & HOLD",     "Drag & drop",   Colors.YELLOW),
        ("ALL FINGERS OPEN", "Pause/Resume",  Colors.LIGHT_GRAY),
    ]

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _draw_translucent_rect(frame: np.ndarray,
                                x1: int, y1: int, x2: int, y2: int,
                                color: tuple, alpha: float = 0.45) -> None:
        """Draw a semi-transparent filled rectangle."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    @staticmethod
    def _put_text(frame: np.ndarray, text: str, pos: tuple,
                  color: tuple, scale: float = 0.55,
                  thickness: int = 1,
                  font: int = cv2.FONT_HERSHEY_SIMPLEX) -> None:
        """Put text with a subtle dark shadow for readability."""
        x, y = pos
        cv2.putText(frame, text, (x + 1, y + 1), font, scale,
                    (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale,
                    color, thickness, cv2.LINE_AA)

    # ── drawing methods ──────────────────────────────────────────────

    def draw_reduction_box(self, frame: np.ndarray) -> None:
        """Draw the reduction-box rectangle with corner accents."""
        cfg = self.cfg
        x1, y1 = cfg.frame_margin_x, cfg.frame_margin_y
        x2 = cfg.cam_width - cfg.frame_margin_x
        y2 = cfg.cam_height - cfg.frame_margin_y
        color = Colors.BLUE

        # Main rectangle (thin)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Corner accents (thicker L-shaped marks at each corner)
        corner_len = 20
        t = 2
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, t)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, t)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, t)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, t)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, t)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, t)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, t)

    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """FPS counter in the top-left."""
        self._put_text(frame, f"FPS: {int(fps)}", (12, 28),
                       Colors.GREEN, scale=0.65, thickness=2)

    def draw_gesture_label(self, frame: np.ndarray, gesture: Gesture,
                           is_paused: bool) -> None:
        """Large gesture label in the top-centre."""
        if is_paused:
            label = "PAUSED"
            color = Colors.LIGHT_GRAY
        elif gesture == Gesture.NONE:
            label = "NO HAND"
            color = Colors.DARK_GRAY
        else:
            label = gesture.name.replace("_", " ")
            color = Colors.GESTURE.get(gesture, Colors.WHITE)

        # Background pill
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        tw, th = text_size
        cx = self.cfg.cam_width // 2
        pill_x1 = cx - tw // 2 - 14
        pill_y1 = 8
        pill_x2 = cx + tw // 2 + 14
        pill_y2 = 8 + th + 16
        self._draw_translucent_rect(frame, pill_x1, pill_y1, pill_x2, pill_y2,
                                     (30, 30, 30), alpha=0.6)
        cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), color, 1)

        self._put_text(frame, label, (cx - tw // 2, pill_y2 - 10),
                       color, scale=0.7, thickness=2)

    def draw_finger_dots(self, frame: np.ndarray,
                         lms: list[tuple[int, int, int]],
                         fingers: list[bool],
                         gesture: Gesture) -> None:
        """Draw colored dots on active fingertips."""
        if not lms:
            return

        tip_ids = [HandDetector.THUMB_TIP, HandDetector.INDEX_TIP,
                   HandDetector.MIDDLE_TIP, HandDetector.RING_TIP,
                   HandDetector.PINKY_TIP]
        tip_colors = [Colors.YELLOW, Colors.CYAN, Colors.RED,
                      Colors.ORANGE, Colors.PURPLE]

        gesture_color = Colors.GESTURE.get(gesture, Colors.CYAN)

        for i, (tip_id, base_color) in enumerate(zip(tip_ids, tip_colors)):
            x, y = lms[tip_id][1], lms[tip_id][2]
            if fingers[i]:
                # Active finger: filled circle + ring
                cv2.circle(frame, (x, y), 10, gesture_color, cv2.FILLED)
                cv2.circle(frame, (x, y), 14, gesture_color, 2)
            else:
                # Inactive finger: small dim dot
                cv2.circle(frame, (x, y), 5, Colors.DARK_GRAY, cv2.FILLED)

    def draw_pinch_bar(self, frame: np.ndarray, pinch_dist: float,
                       threshold: int) -> None:
        """
        Pinch proximity indicator bar at the bottom of the frame.

        Shows how close thumb and index are to triggering a pinch.
        Bar fills up and turns green as they get closer.
        """
        bar_w = 200
        bar_h = 12
        x1 = self.cfg.cam_width // 2 - bar_w // 2
        y1 = self.cfg.cam_height - 30
        x2 = x1 + bar_w
        y2 = y1 + bar_h

        # Background
        self._draw_translucent_rect(frame, x1 - 2, y1 - 18, x2 + 2, y2 + 4,
                                     (20, 20, 20), alpha=0.5)
        cv2.rectangle(frame, (x1, y1), (x2, y2), Colors.DARK_GRAY, 1)

        # Fill ratio: 1.0 when fingers touching, 0.0 when far apart
        max_dist = threshold * 4
        ratio = max(0.0, min(1.0, 1.0 - (pinch_dist / max_dist)))
        fill_w = int(bar_w * ratio)

        # Colour gradient: gray → yellow → green
        if ratio < 0.5:
            color = Colors.LIGHT_GRAY
        elif ratio < 0.75:
            color = Colors.YELLOW
        else:
            color = Colors.GREEN

        if fill_w > 0:
            cv2.rectangle(frame, (x1, y1), (x1 + fill_w, y2), color, cv2.FILLED)

        self._put_text(frame, "PINCH", (x1, y1 - 4), Colors.LIGHT_GRAY, scale=0.4)

    def draw_scroll_indicator(self, frame: np.ndarray,
                              direction: int) -> None:
        """Draw up/down arrow when scrolling."""
        cx = self.cfg.cam_width - 40
        cy = self.cfg.cam_height // 2
        arrow_color = Colors.MAGENTA

        if direction < 0:   # scroll up
            pts = np.array([[cx, cy - 30], [cx - 15, cy], [cx + 15, cy]])
            cv2.fillPoly(frame, [pts], arrow_color)
            self._put_text(frame, "UP", (cx - 12, cy + 20), arrow_color, scale=0.5)
        elif direction > 0:  # scroll down
            pts = np.array([[cx, cy + 30], [cx - 15, cy], [cx + 15, cy]])
            cv2.fillPoly(frame, [pts], arrow_color)
            self._put_text(frame, "DOWN", (cx - 18, cy - 10), arrow_color, scale=0.5)

    def draw_drag_indicator(self, frame: np.ndarray,
                            lms: list[tuple[int, int, int]]) -> None:
        """Draw a pulsing ring around the index finger during drag."""
        if not lms:
            return
        ix, iy = lms[HandDetector.INDEX_TIP][1:3]
        # Pulsing effect using time
        pulse = int(8 * abs(math.sin(time.time() * 5)))
        radius = 18 + pulse
        cv2.circle(frame, (ix, iy), radius, Colors.YELLOW, 2)
        cv2.circle(frame, (ix, iy), radius + 5, Colors.YELLOW, 1)
        self._put_text(frame, "DRAG", (ix - 20, iy - radius - 8),
                       Colors.YELLOW, scale=0.5)

    def draw_click_flash(self, frame: np.ndarray,
                         lms: list[tuple[int, int, int]],
                         gesture: Gesture) -> None:
        """Flash effect at the index finger on click."""
        if not lms:
            return
        ix, iy = lms[HandDetector.INDEX_TIP][1:3]
        color = Colors.GESTURE.get(gesture, Colors.GREEN)
        cv2.circle(frame, (ix, iy), 22, color, 3)
        cv2.circle(frame, (ix, iy), 30, color, 1)

        # Draw line from thumb to relevant finger
        tx, ty = lms[HandDetector.THUMB_TIP][1:3]
        cv2.line(frame, (tx, ty), (ix, iy), color, 2)

    def draw_pause_overlay(self, frame: np.ndarray) -> None:
        """Darken the frame and show PAUSED text when tracking is off."""
        self._draw_translucent_rect(frame, 0, 0,
                                     self.cfg.cam_width, self.cfg.cam_height,
                                     (0, 0, 0), alpha=0.55)
        label = "PAUSED"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        tw, th = text_size
        cx = self.cfg.cam_width // 2 - tw // 2
        cy = self.cfg.cam_height // 2 + th // 2
        self._put_text(frame, label, (cx, cy), Colors.WHITE, scale=1.5, thickness=3)
        self._put_text(frame, "Show open palm to resume",
                       (self.cfg.cam_width // 2 - 120, cy + 35),
                       Colors.LIGHT_GRAY, scale=0.5)

    def draw_no_hand(self, frame: np.ndarray) -> None:
        """Show a message when no hand is detected."""
        label = "Show your hand to start"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        tw, _ = text_size
        cx = self.cfg.cam_width // 2 - tw // 2
        self._put_text(frame, label, (cx, self.cfg.cam_height // 2),
                       Colors.LIGHT_GRAY, scale=0.6)

    def draw_guide_panel(self, frame: np.ndarray) -> None:
        """
        Semi-transparent gesture guide panel in the bottom-left.

        Shows all available gestures with their trigger description
        and a colour-coded dot.
        """
        if not self.cfg.show_guide:
            return

        panel_w = 220
        line_h = 18
        padding = 8
        num_entries = len(self.GUIDE_ENTRIES)
        panel_h = num_entries * line_h + padding * 2 + 20

        x1 = 6
        y1 = self.cfg.cam_height - panel_h - 6
        x2 = x1 + panel_w
        y2 = self.cfg.cam_height - 6

        self._draw_translucent_rect(frame, x1, y1, x2, y2,
                                     (20, 20, 20), alpha=0.55)
        cv2.rectangle(frame, (x1, y1), (x2, y2), Colors.DARK_GRAY, 1)

        # Title
        self._put_text(frame, "GESTURE GUIDE",
                       (x1 + padding, y1 + padding + 12),
                       Colors.WHITE, scale=0.45, thickness=1)

        # Entries
        for i, (label, desc, color) in enumerate(self.GUIDE_ENTRIES):
            ey = y1 + padding + 28 + i * line_h
            # Color dot
            cv2.circle(frame, (x1 + padding + 5, ey - 4), 4, color, cv2.FILLED)
            # Label + description
            self._put_text(frame, f"{label}: {desc}",
                           (x1 + padding + 16, ey),
                           Colors.LIGHT_GRAY, scale=0.35)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VirtualMouse — main application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class VirtualMouse:
    """
    Main application class.

    Captures webcam frames, detects hand gestures, maps them to
    OS-level cursor actions, and draws a rich visual HUD.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.scr_w, self.scr_h = pyautogui.size()

        self.detector = HandDetector(
            model_path=config.model_path,
            detection_confidence=0.70,
            tracking_confidence=0.70,
        )
        self.gesture_engine = GestureEngine(config)
        self.hud = HUD(config)

        # ── State ────────────────────────────────────────────────────
        self._prev_x: float = 0.0
        self._prev_y: float = 0.0
        self._ema_init: bool = False

        self._last_click_time: float = 0.0
        self._last_rclick_time: float = 0.0
        self._last_dclick_time: float = 0.0
        self._last_pause_time: float = 0.0

        self._is_paused: bool = False
        self._is_dragging: bool = False
        self._was_drag: bool = False

        # Scroll tracking: previous Y of middle finger
        self._scroll_prev_y: int | None = None

        self._prev_time: float = 0.0

    # ── coordinate mapping ───────────────────────────────────────────

    def _map_to_screen(self, x: int, y: int) -> tuple[float, float]:
        """
        Map webcam pixel position inside the reduction box to the
        full screen resolution using numpy.interp.

        numpy.interp automatically clamps values outside the input
        range, so positions outside the reduction box map to the
        screen edges.
        """
        sx = float(np.interp(
            x,
            (self.cfg.frame_margin_x,
             self.cfg.cam_width - self.cfg.frame_margin_x),
            (0, self.scr_w),
        ))
        sy = float(np.interp(
            y,
            (self.cfg.frame_margin_y,
             self.cfg.cam_height - self.cfg.frame_margin_y),
            (0, self.scr_h),
        ))
        return sx, sy

    # ── EMA smoothing ────────────────────────────────────────────────

    def _smooth(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        """
        Exponential Moving Average (EMA) — first-order IIR low-pass
        filter to remove high-frequency jitter.

        Formula:
            smoothed_t = alpha * raw_t + (1 - alpha) * smoothed_{t-1}

        alpha close to 1.0 → responsive but jittery
        alpha close to 0.0 → very smooth but sluggish
        alpha ~ 0.35       → good balance at 30+ FPS

        EMA beats a simple moving average because it has O(1) memory
        (only stores previous value) and introduces less lag — recent
        samples are weighted exponentially more than older ones.
        """
        a = self.cfg.smoothing_alpha

        if not self._ema_init:
            self._prev_x, self._prev_y = raw_x, raw_y
            self._ema_init = True
            return raw_x, raw_y

        # Core EMA: weighted blend of new sample and previous output
        sx = a * raw_x + (1.0 - a) * self._prev_x
        sy = a * raw_y + (1.0 - a) * self._prev_y
        self._prev_x, self._prev_y = sx, sy
        return sx, sy

    # ── action executors ─────────────────────────────────────────────

    def _move_cursor(self, lms: list[tuple[int, int, int]]) -> None:
        """Map index finger to screen and move the OS cursor."""
        _, ix, iy = lms[HandDetector.INDEX_TIP]
        raw_sx, raw_sy = self._map_to_screen(ix, iy)
        sx, sy = self._smooth(raw_sx, raw_sy)

        # Clamp away from edges to avoid triggering PyAutoGUI failsafe
        cx = int(np.clip(sx, 2, self.scr_w - 3))
        cy = int(np.clip(sy, 2, self.scr_h - 3))
        pyautogui.moveTo(cx, cy, _pause=False)

    def _do_left_click(self) -> bool:
        """Fire a left click with debounce. Returns True if fired."""
        now = time.time()
        if now - self._last_click_time > self.cfg.click_cooldown:
            pyautogui.click()
            self._last_click_time = now
            return True
        return False

    def _do_right_click(self) -> bool:
        """Fire a right click with debounce."""
        now = time.time()
        if now - self._last_rclick_time > self.cfg.click_cooldown:
            pyautogui.rightClick()
            self._last_rclick_time = now
            return True
        return False

    def _do_double_click(self) -> bool:
        """Fire a double click with debounce."""
        now = time.time()
        if now - self._last_dclick_time > self.cfg.click_cooldown:
            pyautogui.doubleClick()
            self._last_dclick_time = now
            return True
        return False

    def _do_scroll(self, lms: list[tuple[int, int, int]]) -> int:
        """
        Scroll based on the vertical movement of the middle finger.
        Returns -1 (up), +1 (down), or 0 (stationary).
        """
        my = lms[HandDetector.MIDDLE_TIP][2]

        if self._scroll_prev_y is None:
            self._scroll_prev_y = my
            return 0

        delta = my - self._scroll_prev_y
        self._scroll_prev_y = my

        if abs(delta) > 8:   # dead zone to avoid micro-scrolls
            direction = 1 if delta > 0 else -1
            # pyautogui.scroll: positive = up, negative = down
            # delta > 0 means finger moved DOWN → scroll down (negative)
            pyautogui.scroll(-direction * self.cfg.scroll_speed, _pause=False)
            return direction
        return 0

    def _do_drag_start(self, lms: list[tuple[int, int, int]]) -> None:
        """Start a drag operation (mouse button down)."""
        if not self._is_dragging:
            self._is_dragging = True
            pyautogui.mouseDown(_pause=False)
        # Keep moving cursor during drag
        self._move_cursor(lms)

    def _do_drag_end(self) -> None:
        """End drag operation (mouse button up)."""
        if self._is_dragging:
            self._is_dragging = False
            pyautogui.mouseUp(_pause=False)

    def _toggle_pause(self) -> None:
        """Toggle pause state with cooldown."""
        now = time.time()
        if now - self._last_pause_time > self.cfg.pause_cooldown:
            self._is_paused = not self._is_paused
            self._last_pause_time = now
            state = "PAUSED" if self._is_paused else "RESUMED"
            print(f"  [{state}]")

    # ── main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the virtual mouse.  Press 'q' to quit."""
        cap = cv2.VideoCapture(self.cfg.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.cam_height)

        if not cap.isOpened():
            print(f"\n  ERROR: Cannot open webcam (device {self.cfg.cam_index}).")
            print("  Try a different camera index with --cam 1\n")
            sys.exit(1)

        print()
        print("  +=============================================+")
        print("  |    Vision-Based Virtual Mouse  v2.0         |")
        print("  +=============================================+")
        print("  |  Press 'q' in the window to quit            |")
        print("  |  Press 'g' to toggle gesture guide          |")
        print("  |  Show open palm to pause/resume             |")
        print("  +=============================================+")
        print()

        scroll_dir = 0
        fired_action = False

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                if self.cfg.mirror:
                    frame = cv2.flip(frame, 1)

                # ── Detect ───────────────────────────────────────────
                lms = self.detector.detect(frame)
                fingers = HandDetector.fingers_up(lms) if lms else [False] * 5
                gesture, meta = self.gesture_engine.classify(lms, fingers)

                scroll_dir = 0
                fired_action = False

                # ── Handle PAUSE toggle ──────────────────────────────
                if gesture == Gesture.PAUSE:
                    self._toggle_pause()

                # ── Execute gesture actions (only when not paused) ───
                if not self._is_paused and lms:
                    if gesture == Gesture.MOVE:
                        self._do_drag_end()
                        self._move_cursor(lms)

                    elif gesture == Gesture.LEFT_CLICK:
                        self._do_drag_end()
                        self._move_cursor(lms)
                        fired_action = self._do_left_click()

                    elif gesture == Gesture.RIGHT_CLICK:
                        self._do_drag_end()
                        fired_action = self._do_right_click()

                    elif gesture == Gesture.DOUBLE_CLICK:
                        self._do_drag_end()
                        fired_action = self._do_double_click()

                    elif gesture == Gesture.SCROLL:
                        self._do_drag_end()
                        scroll_dir = self._do_scroll(lms)

                    elif gesture == Gesture.DRAG:
                        self._do_drag_start(lms)

                    else:
                        self._do_drag_end()
                        self._scroll_prev_y = None
                else:
                    self._do_drag_end()
                    self._scroll_prev_y = None

                # ── Draw ─────────────────────────────────────────────
                self.detector.draw_skeleton(frame)
                self.hud.draw_reduction_box(frame)

                if self._is_paused:
                    self.hud.draw_pause_overlay(frame)
                elif not lms:
                    self.hud.draw_no_hand(frame)
                else:
                    self.hud.draw_finger_dots(frame, lms, fingers, gesture)
                    self.hud.draw_pinch_bar(frame, meta["pinch_dist"],
                                            self.cfg.pinch_threshold)

                    if gesture == Gesture.SCROLL and scroll_dir != 0:
                        self.hud.draw_scroll_indicator(frame, scroll_dir)

                    if gesture == Gesture.DRAG:
                        self.hud.draw_drag_indicator(frame, lms)

                    if fired_action and gesture in (
                        Gesture.LEFT_CLICK, Gesture.RIGHT_CLICK,
                        Gesture.DOUBLE_CLICK
                    ):
                        self.hud.draw_click_flash(frame, lms, gesture)

                self.hud.draw_gesture_label(frame, gesture, self._is_paused)
                self.hud.draw_guide_panel(frame)

                # FPS
                now = time.time()
                fps = 1.0 / (now - self._prev_time) if self._prev_time else 0.0
                self._prev_time = now
                self.hud.draw_fps(frame, fps)

                cv2.imshow("Virtual Mouse", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("g"):
                    self.cfg.show_guide = not self.cfg.show_guide

        except pyautogui.FailSafeException:
            print("\n  [FAILSAFE] Mouse reached screen corner — exiting safely.\n")
        except KeyboardInterrupt:
            print("\n  [INTERRUPTED] Ctrl+C — exiting.\n")
        finally:
            self._do_drag_end()
            self.detector.close()
            cap.release()
            cv2.destroyAllWindows()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI & entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        prog="virtual_mouse",
        description="Vision-Based Virtual Mouse — control your cursor with "
                    "hand gestures via your webcam.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
gestures:
  INDEX finger up          Move cursor
  THUMB + INDEX pinch      Left click
  THUMB + MIDDLE pinch     Right click
  THUMB + RING pinch       Double click
  INDEX + MIDDLE up        Scroll (move hand up/down)
  Pinch & hold (0.5 s)     Drag & drop
  All 5 fingers open       Pause / Resume

examples:
  python virtual_mouse.py
  python virtual_mouse.py --cam 1 --smoothing 0.5
  python virtual_mouse.py --no-guide --sensitivity 45
""",
    )
    p.add_argument("-c", "--cam", type=int, default=0,
                   help="Camera device index (default: 0)")
    p.add_argument("--width", type=int, default=640,
                   help="Capture width in pixels (default: 640)")
    p.add_argument("--height", type=int, default=480,
                   help="Capture height in pixels (default: 480)")
    p.add_argument("-s", "--smoothing", type=float, default=0.35,
                   help="EMA smoothing alpha, 0.1-1.0 (default: 0.35)")
    p.add_argument("--sensitivity", type=int, default=40,
                   help="Pinch distance threshold in px (default: 40)")
    p.add_argument("--scroll-speed", type=int, default=15,
                   help="Scroll speed units (default: 15)")
    p.add_argument("--no-guide", action="store_true",
                   help="Hide the gesture guide panel")
    p.add_argument("--no-mirror", action="store_true",
                   help="Disable webcam mirror flip")
    p.add_argument("--model", type=str, default="hand_landmarker.task",
                   help="Path to MediaPipe hand_landmarker.task model")
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    # PyAutoGUI settings
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0

    config = Config(
        cam_index=args.cam,
        cam_width=args.width,
        cam_height=args.height,
        model_path=args.model,
        smoothing_alpha=args.smoothing,
        pinch_threshold=args.sensitivity,
        scroll_speed=args.scroll_speed,
        show_guide=not args.no_guide,
        mirror=not args.no_mirror,
    )

    mouse = VirtualMouse(config)
    mouse.run()


if __name__ == "__main__":
    main()
