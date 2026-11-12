"""Portable resources and installed entry points for the Windows application."""

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
MODEL_SIZE = 7_819_105


def data_dir():
    """Keep writable files outside the checkout or installed package."""
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "handmouse"


def ffmpeg_exe():
    from imageio_ffmpeg import get_ffmpeg_exe

    return get_ffmpeg_exe()


def list_cameras():
    result = subprocess.run(
        [ffmpeg_exe(), "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True, encoding="utf-8", errors="replace", timeout=8,
    )
    # Device enumeration exits nonzero even when it succeeds.
    if "Unknown input format" in result.stderr:
        raise RuntimeError("FFmpeg lacks DirectShow support; unset IMAGEIO_FFMPEG_EXE to use the bundled build.")
    return re.findall(r'"([^"\r\n]+)"\s+\(video\)', result.stderr + result.stdout)


def ensure_model():
    """Download the pinned model once; never publish a partial/unverified file."""
    target = data_dir() / "hand_landmarker.task"
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == MODEL_SHA256:
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading MediaPipe hand model (~8 MB)...", flush=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            content = response.read(MODEL_SIZE + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError("Could not download the hand model. Check your internet connection and retry.") from exc
    if len(content) != MODEL_SIZE or hashlib.sha256(content).hexdigest() != MODEL_SHA256:
        raise RuntimeError("Hand model failed integrity verification; retry the download.")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(target)


def require_windows():
    if sys.platform != "win32":
        raise SystemExit("handmouse requires native Windows 10/11 (not WSL).")


def gui():
    require_windows()
    from handmouse_gui import HandmouseGUI

    HandmouseGUI().mainloop()


def track():
    require_windows()
    from handmouse_win import main

    main()
