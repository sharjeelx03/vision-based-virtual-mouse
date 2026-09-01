"""
Build script for creating a Windows executable (.exe) of Virtual Mouse.

Usage:
    python build_exe.py

This will:
  1. Install PyInstaller if not already installed
  2. Build the executable using PyInstaller (--onedir mode)
  3. Bundle the hand_landmarker.task model alongside the EXE

Output: dist/VirtualMouse/VirtualMouse.exe
"""

import os
import sys
import shutil
import subprocess

# ── Configuration ────────────────────────────────────────────────────
APP_NAME = "VirtualMouse"
MAIN_SCRIPT = "virtual_mouse.py"
MODEL_FILE = "hand_landmarker.task"
DIST_DIR = "dist"
BUILD_DIR = "build"


def check_model_file() -> bool:
    """Ensure the MediaPipe model file exists."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"\n  ERROR: '{MODEL_FILE}' not found in project folder.")
        print(f"  Expected path: {model_path}")
        print()
        print("  Download it first:")
        print()
        print('  Invoke-WebRequest -Uri "https://storage.googleapis.com/'
              'mediapipe-models/hand_landmarker/hand_landmarker/float16/'
              f'latest/{MODEL_FILE}" -OutFile "{MODEL_FILE}"')
        print()
        return False
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  [OK] Model file found: {MODEL_FILE} ({size_mb:.1f} MB)")
    return True


def install_pyinstaller() -> None:
    """Install PyInstaller if it's not already available."""
    try:
        import PyInstaller  # noqa: F401
        print("  [OK] PyInstaller is already installed")
    except ImportError:
        print("  --> Installing PyInstaller...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "pyinstaller",
            "--quiet"
        ])
        print("  [OK] PyInstaller installed")


def clean_previous_build() -> None:
    """Remove previous build artifacts."""
    for folder in [BUILD_DIR, os.path.join(DIST_DIR, APP_NAME)]:
        if os.path.exists(folder):
            print(f"  --> Cleaning {folder}/")
            shutil.rmtree(folder)
    # Remove .spec if it exists
    spec_file = f"{APP_NAME}.spec"
    if os.path.exists(spec_file):
        os.remove(spec_file)


def build() -> bool:
    """Run PyInstaller to build the EXE."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = os.path.join(script_dir, MAIN_SCRIPT)
    model_path = os.path.join(script_dir, MODEL_FILE)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--onedir",
        "--noconsole",
        "--noconfirm",
        # Bundle the model file next to the EXE
        "--add-data", f"{model_path};.",
        main_script,
    ]

    print()
    print(f"  --> Building {APP_NAME}.exe ...")
    print(f"  --> Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=script_dir)

    if result.returncode != 0:
        print(f"\n  ERROR: PyInstaller exited with code {result.returncode}")
        return False

    return True


def post_build() -> None:
    """Show summary after successful build."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exe_path = os.path.join(script_dir, DIST_DIR, APP_NAME, f"{APP_NAME}.exe")

    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print()
        print("  +================================================+")
        print("  |          BUILD SUCCESSFUL!                      |")
        print("  +================================================+")
        print(f"  |  EXE:  dist/{APP_NAME}/{APP_NAME}.exe")
        print(f"  |  Size: {size_mb:.1f} MB")
        print("  |                                                  |")
        print("  |  To run:                                         |")
        print(f"  |    .\\dist\\{APP_NAME}\\{APP_NAME}.exe")
        print("  +================================================+")
        print()
    else:
        print(f"\n  WARNING: EXE not found at {exe_path}")
        print("  Check the dist/ folder manually.\n")


def main() -> None:
    print()
    print("  +================================================+")
    print("  |    Virtual Mouse -- EXE Builder                  |")
    print("  +================================================+")
    print()

    # Step 1: Check model file
    if not check_model_file():
        sys.exit(1)

    # Step 2: Install PyInstaller
    install_pyinstaller()

    # Step 3: Clean previous build
    clean_previous_build()

    # Step 4: Build
    if not build():
        sys.exit(1)

    # Step 5: Summary
    post_build()


if __name__ == "__main__":
    main()
