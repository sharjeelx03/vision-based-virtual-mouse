"""
Generate app icons at various sizes from logo.png.

Creates the required icon files for:
  - Windows .ico file (for the EXE and installer)
  - MS Store tile assets (various PNG sizes)

Usage:
    python assets/generate_icons.py

Requires: Pillow (pip install Pillow)
"""

import os
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("ERROR: Pillow is required. Run: pip install Pillow")
    sys.exit(1)


# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
LOGO_PATH = PROJECT_DIR / "logo.png"
ASSETS_DIR = SCRIPT_DIR


def generate_fallback_logo() -> Image.Image:
    """Generate a simple logo if logo.png doesn't exist."""
    size = 512
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background circle
    margin = 20
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill="#7c6ff0",
        outline="#ffffff",
        width=8,
    )

    # Hand shape (simplified)
    cx, cy = size // 2, size // 2
    # Palm
    draw.ellipse([cx - 80, cy + 20, cx + 80, cy + 160], fill="white")
    # Fingers
    finger_w = 28
    for i, offset in enumerate([-60, -20, 20, 60]):
        x = cx + offset
        top = cy - 100 + abs(i - 1.5) * 15
        draw.rounded_rectangle(
            [x - finger_w // 2, int(top), x + finger_w // 2, cy + 60],
            radius=14,
            fill="white",
        )
    # Thumb
    draw.rounded_rectangle(
        [cx - 100, cy - 20, cx - 60, cy + 80],
        radius=14,
        fill="white",
    )

    return img


def load_logo() -> Image.Image:
    """Load the project logo or generate a fallback."""
    if LOGO_PATH.exists():
        print(f"  [OK] Using logo: {LOGO_PATH}")
        img = Image.open(LOGO_PATH).convert("RGBA")
        return img
    else:
        print(f"  [WARN] logo.png not found, generating fallback icon")
        return generate_fallback_logo()


def create_ico(logo: Image.Image) -> None:
    """Create a Windows .ico file with multiple sizes."""
    ico_sizes = [16, 24, 32, 48, 64, 128, 256]
    ico_path = ASSETS_DIR / "app_icon.ico"

    # Resize to each size
    images = []
    for s in ico_sizes:
        resized = logo.copy()
        resized = resized.resize((s, s), Image.LANCZOS)
        images.append(resized)

    # Save as ICO (first image with append_images for multi-size)
    images[0].save(
        ico_path,
        format="ICO",
        append_images=images[1:],
        sizes=[(s, s) for s in ico_sizes],
    )
    print(f"  [OK] Created: {ico_path}")


def create_store_assets(logo: Image.Image) -> None:
    """Create PNG assets required for MS Store."""
    store_sizes = {
        "Square44x44Logo.png": (44, 44),
        "Square71x71Logo.png": (71, 71),
        "Square150x150Logo.png": (150, 150),
        "Square310x310Logo.png": (310, 310),
        "Wide310x150Logo.png": (310, 150),
        "StoreLogo.png": (50, 50),
    }

    for name, (w, h) in store_sizes.items():
        path = ASSETS_DIR / name
        # For non-square tiles, center the logo on a transparent background
        if w != h:
            resized = logo.copy()
            s = min(w, h)
            resized = resized.resize((s, s), Image.LANCZOS)
            canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            offset_x = (w - s) // 2
            offset_y = (h - s) // 2
            canvas.paste(resized, (offset_x, offset_y), resized)
            canvas.save(path)
        else:
            resized = logo.copy()
            resized = resized.resize((w, h), Image.LANCZOS)
            resized.save(path)

        print(f"  [OK] Created: {name} ({w}x{h})")


def main() -> None:
    print()
    print("  +================================================+")
    print("  |    Icon Generator for Virtual Mouse              |")
    print("  +================================================+")
    print()

    logo = load_logo()

    print()
    print("  Creating Windows .ico file...")
    create_ico(logo)

    print()
    print("  Creating MS Store assets...")
    create_store_assets(logo)

    print()
    print("  Done! All icons generated in assets/ folder.")
    print()


if __name__ == "__main__":
    main()
