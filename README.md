# handmouse

Control your Windows mouse with your hand using a webcam. No special camera needed.
Fork of [AidenMoyer/handmouse](https://github.com/AidenMoyer/handmouse), with portable uv installation.

## Install and run (Windows)

Requires **64-bit Windows 10/11**, a webcam, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
No WSL, system FFmpeg, winget, administrator access, or separate pip setup is needed.
Use uv-managed Python to get a matching interpreter with Tkinter.

Download this repository as a ZIP and extract it, or clone it with Git:

```powershell
git clone https://github.com/Rogue-King/handmouse.git
cd handmouse
uv sync --managed-python
uv run handmouse
```

`uv` installs Python 3.11 and the dependencies declared in `pyproject.toml`.
FFmpeg is supplied by the `imageio-ffmpeg` wheel; the Windows C++ runtime DLLs
are supplied by `msvc-runtime`. There are no hard-coded user or installation paths.
The MediaPipe hand model (~8 MB) downloads automatically when tracking starts,
with SHA-256 verification. Internet access is required for installation and the
first tracking session; the model is cached for later offline use.

The project targets Windows x64; it does not add Linux, WSL, macOS, or native ARM64 support.
Moving the source folder is fine: run `uv sync --managed-python` at the destination
to recreate the environment. Virtual environments themselves are not relocatable.

### Install as a standalone command

From an extracted checkout, install with:

```powershell
uv tool install --python 3.11 --managed-python .
handmouse
```

This installs a wheel into uv's tool environment; the checkout is no longer needed.
If `handmouse` is not on PATH, run `uv tool update-shell` and reopen PowerShell.
Developers can build a distributable wheel with `uv build`; recipients install it
with `uv tool install --python 3.11 --managed-python .\dist\handmouse-0.1.0-py3-none-any.whl`.

### Command line

```powershell
uv run handmouse-track --list
uv run handmouse-track --camera "USB Camera" --show
uv run handmouse-track --help
```

Without `--camera`, the first detected camera is selected. An installed tool also
provides `handmouse-track` directly. The old `handmouse_win.bat` and
`setup_windows.ps1` are compatibility wrappers around uv; setup no longer changes
system settings or creates a Desktop shortcut. Legacy relay scripts are not part
of the installed application.

## Gestures

| Gesture | Action |
|---|---|
| Open hand (all fingers extended) | Move mouse |
| Thumb + index finger pinch | Left click / drag |
| Thumb + middle finger pinch | Scroll (move hand up/down) |
| Thumb + ring finger pinch | Middle click (scroll-wheel click) |
| Thumb + pinky finger pinch | Right click |
| Closed fist | Pause — cursor freezes |

**Tracking modes** (selectable in the GUI):

- **Relative (trackpad)** *(default)* — fist to pause, open hand to resume from the same spot. Cursor moves freely across all monitors.
- **Absolute** — hand position in the camera frame maps directly to the selected monitor.

Overhead cameras and palm-down gestures are **not hardware-validated**. Keep fingers
visible and enable the preview to check tracking; do not assume desk-resting pinches
will be detected reliably.

## Settings and resources

Settings and the downloaded model live in `%LOCALAPPDATA%\handmouse`, not beside
the installed scripts. A source checkout's existing `settings.json` is read as a
fallback and saved to the new location when settings change; the original is not deleted.
Settings survive upgrades and can be shared between source and tool installations.

| Setting | Description |
|---|---|
| Camera | Which webcam to use |
| Control monitor | Which monitor to control (absolute mode only) |
| Resolution | Camera capture resolution (higher = more CPU; inference uses at most 640×480) |
| FPS | Camera frame rate |
| Track hand | Any, left, or right |
| Tracking mode | Relative (trackpad) or Absolute |
| Sensitivity | Cursor movement per hand movement |
| Scroll speed | Scroll movement multiplier |
| Mirror | Flip horizontal movement to match your perspective |
| Preview window | Show camera feed with landmark overlay |

The standard MediaPipe Windows wheel runs on CPU+XNNPACK; selecting GPU may fall
back to CPU. A GPU is not required.

## Troubleshooting

- **No cameras / no frames:** Enable camera access for desktop apps in Windows
  Settings, close other applications using the webcam, refresh the camera list,
  and try 640×480 at 30 FPS. `handmouse-track --list` lists device names.
- **FFmpeg/DirectShow error:** Unset `IMAGEIO_FFMPEG_EXE` to use the bundled build.
  If needed, set it to an alternate FFmpeg executable with DirectShow support.
- **Tkinter missing:** Recreate the environment with `uv sync --managed-python`
  using the pinned Python 3.11 instead of a stripped-down system Python.
- **Model download failed:** Check your connection/proxy and retry. A bad or
  incomplete download is never installed. The next launch retries automatically.
- **Cursor drift:** Lower sensitivity; high sensitivity amplifies tracking noise.
- **Hand not detected:** Improve lighting and contrast; keep the whole hand visible.
- **Scroll not working:** Pinch thumb + middle finger. The closest fingertip wins,
  so avoid bringing adjacent fingers close to the thumb.

## Development checks

```powershell
uv sync --locked --managed-python
uv run python -m unittest discover -s tests -v
uv run python test_protocol.py
uv build
```

The Windows CI workflow checks Python 3.11/3.12, native dependency imports,
bundled DirectShow support, unit tests, and wheel installation from outside the checkout.
Real camera accuracy, latency, and overhead positioning still need manual testing.
