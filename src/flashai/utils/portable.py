"""
Portable Mode Utilities

Helpers for running FlashAI in portable mode from a flash drive.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Optional


def get_script_directory() -> Path:
    """Get the directory containing the current script."""
    if getattr(sys, "frozen", False):
        # Running as compiled executable
        return Path(sys.executable).parent
    return Path(__file__).parent.parent.parent.parent


def is_portable_mode() -> bool:
    """
    Check if running in portable mode.

    Portable mode is detected by:
    1. FLASHAI_PORTABLE environment variable
    2. Presence of .flashai_portable marker file
    3. Running from removable media
    """
    # Check environment variable
    if os.environ.get("FLASHAI_PORTABLE", "").lower() in ("1", "true", "yes"):
        return True

    # Check for marker file
    script_dir = get_script_directory()
    marker_file = script_dir / ".flashai_portable"
    if marker_file.exists():
        return True

    # Check if on removable media (platform-specific)
    if platform.system() == "Windows":
        return _is_removable_windows(script_dir)
    elif platform.system() == "Darwin":
        return _is_removable_macos(script_dir)
    else:  # Linux and others
        return _is_removable_linux(script_dir)


def _is_removable_windows(path: Path) -> bool:
    """Check if path is on removable drive (Windows)."""
    try:
        import ctypes
        drive = str(path.resolve())[:3]  # e.g., "E:\\"
        drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive)
        # 2 = DRIVE_REMOVABLE, 6 = DRIVE_RAMDISK
        return drive_type in (2, 6)
    except Exception:
        return False


def _is_removable_macos(path: Path) -> bool:
    """Check if path is on removable drive (macOS)."""
    try:
        resolved = path.resolve()
        # Check if under /Volumes (typical for external drives)
        return str(resolved).startswith("/Volumes/")
    except Exception:
        return False


def _is_removable_linux(path: Path) -> bool:
    """Check if path is on removable drive (Linux)."""
    try:
        resolved = path.resolve()
        # Check common removable mount points
        removable_paths = ["/media/", "/mnt/", "/run/media/"]
        return any(str(resolved).startswith(p) for p in removable_paths)
    except Exception:
        return False


def get_flash_drive_path() -> Optional[Path]:
    """
    Get the root path of the flash drive if in portable mode.

    Returns:
        Path to flash drive root, or None if not in portable mode.
    """
    if not is_portable_mode():
        return None

    script_dir = get_script_directory()

    # Walk up to find flash drive root
    # Look for marker file or volume root
    current = script_dir
    while current != current.parent:
        marker = current / ".flashai_root"
        if marker.exists():
            return current

        # Check if we've reached a volume boundary
        if platform.system() == "Windows":
            if len(str(current)) <= 3:  # e.g., "E:\\"
                return current
        elif str(current).startswith("/Volumes/"):
            parts = str(current).split("/")
            if len(parts) <= 3:  # /Volumes/DriveName
                return current
        elif str(current).startswith(("/media/", "/mnt/", "/run/media/")):
            parts = str(current).split("/")
            if len(parts) <= 4:
                return current

        current = current.parent

    return script_dir


def setup_portable_environment() -> dict[str, str]:
    """
    Set up environment variables for portable mode.

    Returns:
        Dictionary of environment variables set.
    """
    flash_drive = get_flash_drive_path()
    if flash_drive is None:
        return {}

    env_vars = {}

    # Set FLASHAI_BASE to flash drive root
    env_vars["FLASHAI_BASE"] = str(flash_drive)
    os.environ["FLASHAI_BASE"] = str(flash_drive)

    # Set data directory
    data_dir = flash_drive / "data"
    env_vars["FLASHAI_DATA"] = str(data_dir)
    os.environ["FLASHAI_DATA"] = str(data_dir)

    # Set config directory
    config_dir = flash_drive / "config"
    env_vars["FLASHAI_CONFIG"] = str(config_dir)
    os.environ["FLASHAI_CONFIG"] = str(config_dir)

    # Set portable flag
    env_vars["FLASHAI_PORTABLE"] = "1"
    os.environ["FLASHAI_PORTABLE"] = "1"

    return env_vars


def create_portable_marker(path: Path) -> None:
    """Create portable mode marker file."""
    marker = path / ".flashai_portable"
    marker.touch()

    root_marker = path / ".flashai_root"
    root_marker.touch()


def get_portable_python_path() -> Optional[Path]:
    """
    Get path to portable Python if available.

    For fully portable installations, a Python runtime can be
    bundled with FlashAI.
    """
    flash_drive = get_flash_drive_path()
    if flash_drive is None:
        return None

    # Check for portable Python installations
    python_paths = [
        flash_drive / "python" / "python.exe",  # Windows
        flash_drive / "python" / "bin" / "python3",  # Unix
        flash_drive / "python" / "python",  # Alternative
    ]

    for python_path in python_paths:
        if python_path.exists():
            return python_path

    return None
