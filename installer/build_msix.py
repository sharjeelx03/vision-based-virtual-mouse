"""
Build an MSIX package for Microsoft Store submission.

This script takes the PyInstaller output (dist/VirtualMouse/) and
packages it into a .msix file that can be submitted to the
Microsoft Store via Partner Center.

Prerequisites:
    1. Build the EXE first:       python build_exe.py
    2. Generate icon assets:      python assets/generate_icons.py
    3. Windows 10 SDK installed (for MakeAppx.exe and SignTool.exe)
       Download: https://developer.microsoft.com/windows/downloads/windows-sdk/

Usage:
    python installer/build_msix.py

Output:
    installer/Output/VirtualMouse.msix
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


# ── Configuration ────────────────────────────────────────────────────
APP_NAME = "VirtualMouse"
VERSION = "2.0.0.0"

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DIST_DIR = PROJECT_DIR / "dist" / APP_NAME
ASSETS_DIR = PROJECT_DIR / "assets"
OUTPUT_DIR = SCRIPT_DIR / "Output"
STAGING_DIR = SCRIPT_DIR / "_msix_staging"

MANIFEST_TEMPLATE = SCRIPT_DIR / "AppxManifest.xml"


def find_makeappx() -> str | None:
    """Find MakeAppx.exe on the system."""
    # Check PATH first
    result = shutil.which("MakeAppx.exe")
    if result:
        return result

    # Check common Windows SDK locations
    sdk_base = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    sdk_dir = sdk_base / "Windows Kits" / "10" / "bin"

    if sdk_dir.exists():
        # Find the latest version
        versions = sorted(
            [d for d in sdk_dir.iterdir() if d.is_dir() and d.name.startswith("10.")],
            reverse=True,
        )
        for ver_dir in versions:
            makeappx = ver_dir / "x64" / "MakeAppx.exe"
            if makeappx.exists():
                return str(makeappx)

    return None


def check_prerequisites() -> bool:
    """Verify all prerequisites are met."""
    ok = True

    # Check PyInstaller output exists
    if not DIST_DIR.exists():
        print(f"  [ERROR] PyInstaller output not found: {DIST_DIR}")
        print(f"          Run 'python build_exe.py' first.")
        ok = False
    else:
        print(f"  [OK] PyInstaller output found: {DIST_DIR}")

    # Check icon assets
    required_assets = [
        "Square44x44Logo.png",
        "Square150x150Logo.png",
        "StoreLogo.png",
    ]
    for asset in required_assets:
        path = ASSETS_DIR / asset
        if not path.exists():
            print(f"  [ERROR] Missing asset: {path}")
            print(f"          Run 'python assets/generate_icons.py' first.")
            ok = False

    if ok:
        print(f"  [OK] Icon assets found in: {ASSETS_DIR}")

    # Check manifest
    if not MANIFEST_TEMPLATE.exists():
        print(f"  [ERROR] AppxManifest.xml not found: {MANIFEST_TEMPLATE}")
        ok = False
    else:
        # Check if publisher ID has been filled in
        content = MANIFEST_TEMPLATE.read_text(encoding="utf-8")
        if "INSERT-YOUR-PUBLISHER-ID" in content:
            print(f"  [WARN] AppxManifest.xml still has placeholder Publisher ID.")
            print(f"         Update it with your Partner Center Publisher ID before")
            print(f"         submitting to the Microsoft Store.")
        else:
            print(f"  [OK] AppxManifest.xml found")

    # Check MakeAppx.exe
    makeappx = find_makeappx()
    if not makeappx:
        print(f"  [ERROR] MakeAppx.exe not found.")
        print(f"          Install the Windows 10 SDK:")
        print(f"          https://developer.microsoft.com/windows/downloads/windows-sdk/")
        ok = False
    else:
        print(f"  [OK] MakeAppx.exe found: {makeappx}")

    return ok


def create_staging() -> None:
    """Create a staging directory with the MSIX layout."""
    # Clean staging
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True)

    print(f"\n  --> Copying application files...")

    # Copy all PyInstaller output directly to staging root
    shutil.copytree(DIST_DIR, STAGING_DIR, dirs_exist_ok=True)

    print(f"  --> Copying manifest...")
    shutil.copy2(MANIFEST_TEMPLATE, STAGING_DIR / "AppxManifest.xml")

    print(f"  --> Copying icon assets...")
    assets_dest = STAGING_DIR / "Assets"
    assets_dest.mkdir(exist_ok=True)

    for png in ASSETS_DIR.glob("*.png"):
        shutil.copy2(png, assets_dest / png.name)

    print(f"  [OK] Staging directory created: {STAGING_DIR}")


def build_msix() -> bool:
    """Run MakeAppx.exe to create the .msix package."""
    makeappx = find_makeappx()
    if not makeappx:
        return False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{APP_NAME}.msix"

    # Remove existing package
    if output_path.exists():
        output_path.unlink()

    cmd = [
        makeappx,
        "pack",
        "/d", str(STAGING_DIR),
        "/p", str(output_path),
        "/o",  # overwrite
    ]

    print(f"\n  --> Building MSIX package...")
    print(f"  --> Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\n  [ERROR] MakeAppx failed (code {result.returncode})")
        if result.stderr:
            print(f"  {result.stderr[:500]}")
        return False

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  [OK] MSIX package created: {output_path} ({size_mb:.1f} MB)")

    return True


def cleanup() -> None:
    """Remove staging directory."""
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR, ignore_errors=True)


def main() -> None:
    print()
    print("  +================================================+")
    print("  |    Virtual Mouse -- MSIX Builder                 |")
    print("  +================================================+")
    print()

    if not check_prerequisites():
        print("\n  [FAILED] Prerequisites not met. Fix errors above and retry.")
        sys.exit(1)

    try:
        create_staging()

        if build_msix():
            print()
            print("  +================================================+")
            print("  |          MSIX BUILD SUCCESSFUL!                  |")
            print("  +================================================+")
            print(f"  |  Package: installer/Output/{APP_NAME}.msix")
            print("  |                                                  |")
            print("  |  Next steps:                                     |")
            print("  |  1. Go to https://partner.microsoft.com          |")
            print("  |  2. Create a new app submission                  |")
            print("  |  3. Upload the .msix package                     |")
            print("  +================================================+")
            print()
        else:
            print("\n  [FAILED] MSIX build failed.")
            sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
