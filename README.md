# Vision-Based Virtual Mouse

Real-time hand-tracking virtual mouse using **MediaPipe**, **OpenCV**, and **PyAutoGUI**.  
Control your cursor with your index finger and click by pinching your thumb and index finger together.

## Quick Start

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python virtual_mouse.py
```

Press **`q`** in the OpenCV window to quit.  
Move your **physical mouse to any screen corner** to trigger the PyAutoGUI failsafe and abort immediately.

## How It Works

| Component | Role |
|---|---|
| `HandDetector` | Wraps MediaPipe Hands; returns pixel-space landmarks. |
| `VirtualMouse` | Main loop — reads frames, maps coordinates, smooths, moves cursor, detects clicks, draws HUD. |
| `Config` | Single dataclass holding every tunable parameter. |

### Coordinate Mapping (Reduction Box)

The webcam frame is logically shrunk by a configurable margin on each side.  
`numpy.interp` linearly maps positions inside this inner rectangle to the full screen resolution, so you don't need to stretch your arm to the extreme edges of the camera.

### Anti-Jitter (EMA Smoothing)

Raw MediaPipe landmarks jitter ±3-5 px per frame. An **Exponential Moving Average** filter is applied:

```
smoothed_t = α · raw_t  +  (1 − α) · smoothed_{t-1}
```

`α = 0.35` balances responsiveness with jitter suppression at ≥ 30 FPS.

### Click Detection

Euclidean distance between **Thumb tip (Landmark 4)** and **Index tip (Landmark 8)** is computed each frame.  
If the distance drops below `35 px`, a single left click is fired with a `0.4 s` debounce cooldown to prevent accidental double-clicks.

## Tuning

Edit the `Config` dataclass at the top of `virtual_mouse.py`:

| Parameter | Default | Description |
|---|---|---|
| `smoothing_alpha` | `0.35` | EMA weight (↑ = more responsive, ↓ = smoother) |
| `click_distance_threshold` | `35` | Pinch distance in pixels to trigger a click |
| `click_cooldown_sec` | `0.40` | Minimum seconds between consecutive clicks |
| `frame_margin_x / y` | `110 / 80` | Reduction-box padding from each edge |

## Requirements

- Python 3.10+
- Webcam
- Windows / macOS / Linux (PyAutoGUI supports all three)
