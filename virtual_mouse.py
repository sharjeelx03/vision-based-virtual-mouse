"""
Vision-Based Virtual Mouse
===========================
Real-time hand-tracking virtual mouse using MediaPipe, OpenCV, and PyAutoGUI.

Architecture
------------
• HandDetector  – wraps MediaPipe Hands; returns landmark positions.
• VirtualMouse  – main loop: reads frames, maps coordinates, moves cursor,
                  detects clicks, draws overlays.

Performance target: ≥ 30 FPS on an Intel i7 with a 640×480 webcam feed.

Author : Sharj
Date   : 2026-08-30
Python : 3.10+
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp
import pyautogui


# ──────────────────────────────────────────────────────────────────────
#  Configuration dataclass – single source of truth for every tunable.
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    """All tuneable parameters collected in one place."""

    # ── Camera ───────────────────────────────────────────────────────
    cam_width: int = 640                 # capture width  (px)
    cam_height: int = 480                # capture height (px)
    cam_index: int = 0                   # default webcam device id

    # ── Reduction box (margin from each edge of the frame) ──────────
    #    The user only needs to move their hand inside this smaller
    #    rectangle rather than reaching the extreme edges of the cam.
    frame_margin_x: int = 110            # left / right padding (px)
    frame_margin_y: int = 80             # top / bottom padding (px)

    # ── Smoothing (Exponential Moving Average) ───────────────────────
    #    α (alpha) controls how quickly the filter responds.
    #      α close to 1  → fast response, more jitter
    #      α close to 0  → very smooth, but sluggish
    #    0.35 is a good trade-off for a 640×480 feed at 30+ FPS.
    smoothing_alpha: float = 0.35

    # ── Click detection ──────────────────────────────────────────────
    click_distance_threshold: int = 35   # px – thumb-to-index distance
    click_cooldown_sec: float = 0.40     # debounce window (seconds)

    # ── Visual overlays ──────────────────────────────────────────────
    index_circle_radius: int = 14        # highlight circle around index tip
    color_index: tuple = (0, 255, 255)   # yellow-ish (BGR)
    color_click: tuple = (0, 255, 0)     # green flash on click
    color_box: tuple = (255, 180, 0)     # reduction-box colour
    color_fps: tuple = (0, 255, 128)     # FPS text colour


# ──────────────────────────────────────────────────────────────────────
#  HandDetector – thin wrapper around MediaPipe Hands
# ──────────────────────────────────────────────────────────────────────
class HandDetector:
    """
    Detects a single hand and exposes pixel-space landmark positions.

    MediaPipe returns *normalised* coordinates (0-1).  This class converts
    them to pixel coordinates using the frame dimensions so the rest of
    the pipeline works in a consistent integer pixel space.
    """

    # Landmark indices we care about (MediaPipe hand model)
    INDEX_TIP = 8    # tip of the index finger
    THUMB_TIP = 4    # tip of the thumb

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.75,
        tracking_confidence: float = 0.75,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        # Cache the last result so the caller can draw later.
        self._results = None

    # ── public API ───────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int]]:
        """
        Run detection on a BGR frame.

        Returns
        -------
        landmarks : list[(id, x, y)]
            Pixel-space landmarks for the first detected hand,
            or an empty list if no hand is found.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._results = self._hands.process(rgb)

        landmarks: list[tuple[int, int, int]] = []

        if self._results.multi_hand_landmarks:
            hand = self._results.multi_hand_landmarks[0]
            for idx, lm in enumerate(hand.landmark):
                # Convert normalised (0-1) coords → pixel coords.
                px, py = int(lm.x * w), int(lm.y * h)
                landmarks.append((idx, px, py))

        return landmarks

    def draw(self, frame: np.ndarray) -> None:
        """Draw MediaPipe hand skeleton on the frame (in-place)."""
        if self._results and self._results.multi_hand_landmarks:
            for hand_lms in self._results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._mp_styles.get_default_hand_landmarks_style(),
                    self._mp_styles.get_default_hand_connections_style(),
                )


# ──────────────────────────────────────────────────────────────────────
#  VirtualMouse – main application loop
# ──────────────────────────────────────────────────────────────────────
class VirtualMouse:
    """
    Captures webcam frames, tracks the index finger, and maps its
    position to the OS cursor via PyAutoGUI.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config or Config()

        # Screen dimensions (primary monitor)
        self.scr_w, self.scr_h = pyautogui.size()

        # Initialise hand detector
        self.detector = HandDetector()

        # ── Smoothing state ──────────────────────────────────────────
        #    We store the *previous* smoothed screen-space position.
        #    On the very first frame we seed it with the raw value.
        self._prev_x: float = 0.0
        self._prev_y: float = 0.0
        self._initialised: bool = False

        # ── Click debounce ───────────────────────────────────────────
        self._last_click_time: float = 0.0

        # ── FPS counter ──────────────────────────────────────────────
        self._prev_time: float = 0.0

    # ── coordinate mapping ───────────────────────────────────────────

    def _map_to_screen(self, x: int, y: int) -> tuple[float, float]:
        """
        Map a pixel position from the webcam's *reduction box* to the
        full screen resolution.

        The reduction box is the webcam frame shrunk by `frame_margin`
        pixels on each side.  `numpy.interp` performs a linear
        interpolation (and clamps automatically) from one range to
        another:

            screen_x = interp(cam_x,
                              [margin_left,  cam_w - margin_right],
                              [0,            screen_w])

        This means the user only has to move their hand within the
        inner rectangle of the webcam view to reach all four corners
        of the monitor.
        """
        screen_x = np.interp(
            x,
            (self.cfg.frame_margin_x, self.cfg.cam_width - self.cfg.frame_margin_x),
            (0, self.scr_w),
        )
        screen_y = np.interp(
            y,
            (self.cfg.frame_margin_y, self.cfg.cam_height - self.cfg.frame_margin_y),
            (0, self.scr_h),
        )
        return float(screen_x), float(screen_y)

    # ── EMA smoothing ────────────────────────────────────────────────

    def _smooth(self, raw_x: float, raw_y: float) -> tuple[float, float]:
        """
        Exponential Moving Average (EMA) – a first-order IIR low-pass
        filter that removes high-frequency jitter from the landmark
        coordinates while keeping the cursor responsive.

        Mathematics
        -----------
        Given a smoothing factor α ∈ (0, 1]:

            smoothed_t = α · raw_t  +  (1 − α) · smoothed_{t-1}

        • When α = 1 the filter is disabled (output = raw input).
        • When α → 0 the output barely moves (extreme smoothing).
        • α ≈ 0.3-0.4 works well at 30 FPS for cursor control.

        The filter is *causal* (only uses past values) and has O(1)
        memory – we only store the previous smoothed value.

        Why EMA instead of a simple moving average?
        ────────────────────────────────────────────
        A simple N-sample moving average would introduce N/2 frames of
        lag.  EMA reacts faster to genuine direction changes because
        recent samples are weighted exponentially more than older ones.
        """
        alpha = self.cfg.smoothing_alpha

        if not self._initialised:
            # First frame: seed the filter with the raw value so we
            # don't see the cursor jump from (0, 0).
            self._prev_x = raw_x
            self._prev_y = raw_y
            self._initialised = True
            return raw_x, raw_y

        # Core EMA equations:
        #   smoothed = α * new_sample + (1 − α) * previous_smoothed
        smooth_x = alpha * raw_x + (1.0 - alpha) * self._prev_x
        smooth_y = alpha * raw_y + (1.0 - alpha) * self._prev_y

        # Store for next iteration.
        self._prev_x = smooth_x
        self._prev_y = smooth_y

        return smooth_x, smooth_y

    # ── click detection ──────────────────────────────────────────────

    @staticmethod
    def _distance(
        lm_a: tuple[int, int, int],
        lm_b: tuple[int, int, int],
    ) -> float:
        """Euclidean distance between two landmarks in pixel space."""
        return math.hypot(lm_a[1] - lm_b[1], lm_a[2] - lm_b[2])

    def _try_click(self, landmarks: list[tuple[int, int, int]]) -> bool:
        """
        Check if the thumb tip and index tip are close enough to
        constitute a 'pinch' (click).  Returns True if a click was
        actually fired (respects debounce cooldown).
        """
        thumb = landmarks[HandDetector.THUMB_TIP]
        index = landmarks[HandDetector.INDEX_TIP]

        dist = self._distance(thumb, index)

        if dist < self.cfg.click_distance_threshold:
            now = time.time()
            if now - self._last_click_time > self.cfg.click_cooldown_sec:
                pyautogui.click()
                self._last_click_time = now
                return True
        return False

    # ── HUD / overlay drawing ────────────────────────────────────────

    def _draw_hud(
        self,
        frame: np.ndarray,
        landmarks: list[tuple[int, int, int]],
        clicked: bool,
        fps: float,
    ) -> None:
        """Draw reduction box, index-finger highlight, and FPS text."""
        cfg = self.cfg

        # --- Reduction box (the area the user should keep their hand in)
        cv2.rectangle(
            frame,
            (cfg.frame_margin_x, cfg.frame_margin_y),
            (cfg.cam_width - cfg.frame_margin_x,
             cfg.cam_height - cfg.frame_margin_y),
            cfg.color_box,
            2,
        )

        if landmarks:
            # --- Circle around index-finger tip
            ix, iy = landmarks[HandDetector.INDEX_TIP][1:3]
            colour = cfg.color_click if clicked else cfg.color_index
            cv2.circle(frame, (ix, iy), cfg.index_circle_radius, colour, cv2.FILLED)
            cv2.circle(frame, (ix, iy), cfg.index_circle_radius + 4, colour, 2)

            # --- If clicking, draw a line between thumb and index
            if clicked:
                tx, ty = landmarks[HandDetector.THUMB_TIP][1:3]
                cv2.line(frame, (tx, ty), (ix, iy), cfg.color_click, 3)

        # --- FPS counter (top-left)
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            cfg.color_fps,
            2,
        )

    # ── main loop ────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the virtual-mouse main loop.  Press 'q' to quit."""
        cap = cv2.VideoCapture(self.cfg.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.cam_height)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam (device index {self.cfg.cam_index})."
            )

        print("──────────────────────────────────────────")
        print("  Vision-Based Virtual Mouse — Running")
        print("  Press 'q' in the OpenCV window to quit.")
        print("──────────────────────────────────────────")

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                # Mirror the frame so the cursor moves in the intuitive
                # direction (move hand right → cursor goes right).
                frame = cv2.flip(frame, 1)

                # ── Hand detection ───────────────────────────────────
                landmarks = self.detector.detect(frame)

                clicked = False

                if landmarks:
                    # Index finger tip position (pixel-space).
                    _, ix, iy = landmarks[HandDetector.INDEX_TIP]

                    # Map cam coords → screen coords via the reduction box.
                    raw_sx, raw_sy = self._map_to_screen(ix, iy)

                    # Apply EMA smoothing to remove jitter.
                    smooth_sx, smooth_sy = self._smooth(raw_sx, raw_sy)

                    # Move the OS cursor.
                    pyautogui.moveTo(
                        int(smooth_sx),
                        int(smooth_sy),
                        _pause=False,           # skip PyAutoGUI's built-in pause
                    )

                    # Check for pinch → click.
                    clicked = self._try_click(landmarks)

                # ── Draw overlays ────────────────────────────────────
                self.detector.draw(frame)

                # FPS calculation.
                now = time.time()
                fps = 1.0 / (now - self._prev_time) if self._prev_time else 0.0
                self._prev_time = now

                self._draw_hud(frame, landmarks, clicked, fps)

                # ── Show the frame ───────────────────────────────────
                cv2.imshow("Virtual Mouse", frame)

                # 'q' to quit.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # PyAutoGUI failsafe: moving the physical mouse to any screen corner
    # raises pyautogui.FailSafeException and aborts the script.
    pyautogui.FAILSAFE = True

    # Disable PyAutoGUI's default 0.1 s pause after every call — it
    # would halve our effective FPS.
    pyautogui.PAUSE = 0.0

    config = Config()
    mouse = VirtualMouse(config)
    mouse.run()


if __name__ == "__main__":
    main()
