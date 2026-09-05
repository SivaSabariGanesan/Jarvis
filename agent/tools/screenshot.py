"""
Screenshot capture tool for JARVIS V3.
Safely captures desktop screen and stores artifacts within data/screenshots/ in the workspace.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional
from PIL import Image

from agent.config import settings
from agent.tools.security import security_validator, SecurityAuditLogger

logger = logging.getLogger("jarvis.tools.screenshot")


def _capture_screen_win32() -> Image.Image:
    """Capture desktop display using Windows GDI."""
    import ctypes
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    w = user32.GetSystemMetrics(0)
    h = user32.GetSystemMetrics(1)

    hdc_screen = user32.GetDC(0)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
    hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
    gdi32.SelectObject(hdc_mem, hbmp)
    gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, 0, 0, 0x00CC0020)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32),
            ("biWidth", ctypes.c_int32),
            ("biHeight", ctypes.c_int32),
            ("biPlanes", ctypes.c_uint16),
            ("biBitCount", ctypes.c_uint16),
            ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32),
            ("biXPelsPerMeter", ctypes.c_int32),
            ("biYPelsPerMeter", ctypes.c_int32),
            ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h  # top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0
    buffer = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(hdc_mem, hbmp, 0, h, buffer, ctypes.byref(bmi), 0)

    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(0, hdc_screen)

    return Image.frombuffer("RGBA", (w, h), buffer, "raw", "BGRA", 0, 1).convert("RGB")


def prune_old_screenshots(screenshots_dir: Path, max_keep: int):
    """Prune oldest screenshot files when exceeding retention limit."""
    if max_keep <= 0:
        return

    try:
        files = list(screenshots_dir.glob("screenshot_*.png")) + list(screenshots_dir.glob("*.png"))
        files = list(set(files))
        if len(files) > max_keep:
            # Sort by modification time ascending (oldest first)
            files.sort(key=lambda p: p.stat().st_mtime)
            to_delete = files[: len(files) - max_keep]
            for f in to_delete:
                try:
                    f.unlink()
                    logger.debug(f"Pruned old screenshot: {f.name}")
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Error during screenshot pruning: {e}")


def capture_screen_image() -> Image.Image:
    """Capture current desktop image into a PIL Image object."""
    if sys.platform == "win32":
        return _capture_screen_win32()
    else:
        from PIL import ImageGrab
        return ImageGrab.grab()


def take_screenshot(filename: Optional[str] = None) -> str:
    """
    Capture a screenshot of the current screen and save to the workspace data/screenshots directory.
    Enforces retention count to prevent unlimited disk usage.
    """
    screenshots_dir = Path(settings.jarvis_workspace) / "data" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        clean_name = Path(filename).name
        if not clean_name.endswith(".png"):
            clean_name += ".png"
        target_path = screenshots_dir / clean_name
    else:
        timestamp = int(time.time())
        target_path = screenshots_dir / f"screenshot_{timestamp}.png"

    # Validate destination inside workspace
    validated_path = security_validator.validate_path(str(target_path))

    try:
        img = capture_screen_image()

        if getattr(settings, "screenshot_save_enabled", True):
            img.save(str(validated_path), "PNG")
            # Enforce retention pruning
            prune_old_screenshots(
                screenshots_dir,
                getattr(settings, "screenshot_retention_count", 20),
            )
            summary = f"Screenshot captured ({img.width}x{img.height}) and saved to data/screenshots/{validated_path.name}, sir."
        else:
            summary = f"Screenshot captured in memory ({img.width}x{img.height}), sir."

        SecurityAuditLogger.log_execution("take_screenshot", True, summary)
        return summary
    except Exception as e:
        SecurityAuditLogger.log_execution("take_screenshot", False, str(e))
        return f"Unable to capture screenshot: {e}"

