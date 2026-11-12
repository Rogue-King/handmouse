"""Low-latency DirectShow capture with a single latest-frame slot."""

import subprocess
import threading
import time

import numpy as np

from handmouse_runtime import ffmpeg_exe


def processing_size(width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Camera width and height must be positive")
    scale = min(1.0, 640 / width, 480 / height)
    return max(1, int(width * scale)), max(1, int(height * scale))


class FFmpegCamera:
    def __init__(self, device_name, width, height, fps):
        self.width, self.height = processing_size(width, height)
        if fps <= 0:
            raise ValueError("Camera FPS must be positive")
        self.frame_bytes = self.width * self.height * 3
        cmd = [
            ffmpeg_exe(),
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-probesize", "32", "-analyzeduration", "0",
            "-loglevel", "error", "-f", "dshow", "-rtbufsize", "16M",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", f"video={device_name}",
            # Resize before the pipe: 1080p/60 otherwise sends 373 MB/s to Python.
            "-vf", f"scale={self.width}:{self.height}:flags=fast_bilinear",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-flush_packets", "1", "-",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        self._frame = None
        self._seq = 0
        self._received_at = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        buf = bytearray()
        while self._running:
            chunk = self._proc.stdout.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            complete = len(buf) // self.frame_bytes
            if not complete:
                continue
            # Drop every older complete frame in this read, not just at inference time.
            end = complete * self.frame_bytes
            raw = bytes(buf[end - self.frame_bytes:end])
            del buf[:end]
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
            with self._lock:
                self._frame = frame
                self._seq += complete
                self._received_at = time.monotonic()

    def read_seq(self):
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return self._seq, frame, self._received_at

    def release(self):
        self._running = False
        self._proc.terminate()
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._thread.join(timeout=3)
        self._proc.stdout.close()
