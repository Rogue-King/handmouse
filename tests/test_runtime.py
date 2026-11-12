import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import handmouse_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="handmouse test ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data_patch = patch.dict(os.environ, {"LOCALAPPDATA": str(self.root)})
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)

    def test_user_data_is_outside_installation(self):
        self.assertEqual(runtime.data_dir(), self.root / "handmouse")

    def model_patches(self, content):
        size = patch.object(runtime, "MODEL_SIZE", len(content))
        digest = patch.object(runtime, "MODEL_SHA256", hashlib.sha256(content).hexdigest())
        size.start()
        digest.start()
        self.addCleanup(size.stop)
        self.addCleanup(digest.stop)

    def test_model_download_is_cached_and_corruption_is_repaired(self):
        content = b"verified model"
        self.model_patches(content)
        with patch.object(runtime.urllib.request, "urlopen", return_value=io.BytesIO(content)) as download:
            target = Path(runtime.ensure_model())
            self.assertEqual(target.read_bytes(), content)
            download.assert_called_once_with(runtime.MODEL_URL, timeout=60)
        with patch.object(runtime.urllib.request, "urlopen", side_effect=AssertionError("network used")):
            self.assertEqual(Path(runtime.ensure_model()), target)
        target.write_bytes(b"corrupt")
        with patch.object(runtime.urllib.request, "urlopen", return_value=io.BytesIO(content)):
            runtime.ensure_model()
        self.assertEqual(target.read_bytes(), content)
        self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_invalid_model_is_never_installed(self):
        self.model_patches(b"correct")
        for invalid in (b"", b"wrong!!", b"too much data"):
            with self.subTest(invalid=invalid):
                with patch.object(runtime.urllib.request, "urlopen", return_value=io.BytesIO(invalid)):
                    with self.assertRaisesRegex(RuntimeError, "integrity"):
                        runtime.ensure_model()
                self.assertFalse((runtime.data_dir() / "hand_landmarker.task").exists())
                self.assertEqual(list(runtime.data_dir().glob("*.tmp")), [])

    def test_network_error_is_actionable(self):
        with patch.object(runtime.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaisesRegex(RuntimeError, "internet connection"):
                runtime.ensure_model()

    def test_failed_replace_preserves_existing_file_and_removes_temporary(self):
        content = b"correct"
        self.model_patches(content)
        runtime.data_dir().mkdir()
        target = runtime.data_dir() / "hand_landmarker.task"
        target.write_bytes(b"old")
        with patch.object(runtime.urllib.request, "urlopen", return_value=io.BytesIO(content)):
            with patch.object(runtime.os, "replace", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    runtime.ensure_model()
        self.assertEqual(target.read_bytes(), b"old")
        self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_camera_enumeration_uses_bundled_executable_and_accepts_nonzero_exit(self):
        result = subprocess.CompletedProcess([], 1, "", '[dshow] "USB Camera" (video)\n[dshow] "Microphone" (audio)\n[dshow] "Other Camera" (video)')
        with patch.object(runtime, "ffmpeg_exe", return_value="C:/path with spaces/ffmpeg.exe"):
            with patch.object(runtime.subprocess, "run", return_value=result) as run:
                self.assertEqual(runtime.list_cameras(), ["USB Camera", "Other Camera"])
                self.assertEqual(run.call_args.args[0][0], "C:/path with spaces/ffmpeg.exe")
                self.assertEqual(run.call_args.kwargs["timeout"], 8)

    def test_missing_camera_does_not_invent_a_device(self):
        with patch.object(runtime, "ffmpeg_exe", return_value="ffmpeg"):
            with patch.object(runtime.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "")):
                self.assertEqual(runtime.list_cameras(), [])

    def test_incompatible_ffmpeg_is_reported(self):
        with patch.object(runtime, "ffmpeg_exe", return_value="ffmpeg"):
            result = subprocess.CompletedProcess([], 1, "", "Unknown input format: 'dshow'")
            with patch.object(runtime.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "DirectShow"):
                    runtime.list_cameras()

    def test_non_windows_entrypoints_fail_before_importing_desktop_dependencies(self):
        with patch.object(sys, "platform", "linux"):
            for entrypoint in (runtime.gui, runtime.track):
                with self.assertRaisesRegex(SystemExit, "native Windows"):
                    entrypoint()


if __name__ == "__main__":
    unittest.main()
