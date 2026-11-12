import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

spec = importlib.util.spec_from_file_location("handmouse_win_test", Path(__file__).resolve().parents[1] / "handmouse_win.py")
tracker = importlib.util.module_from_spec(spec)
# Never import a live desktop input backend in a headless/model test.
with patch.dict(sys.modules, {"pyautogui": Mock(), "screeninfo": Mock()}):
    spec.loader.exec_module(tracker)


class TrackingLoopTests(unittest.TestCase):
    def run_tracker(self, backend, camera_error=None, stalled=False):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        result = SimpleNamespace(
            hand_landmarks=[[SimpleNamespace(x=0.5, y=0.5, z=0) for _ in range(21)]],
            handedness=[[SimpleNamespace(category_name="Left", score=0.9)]],
        )
        detector = Mock()
        detector.detect.return_value = detector.detect_for_video.return_value = result
        camera = Mock(width=640, height=360)
        frames = [(1, frame, 0), (1, frame, 0), (1, frame, 0) if stalled else KeyboardInterrupt]
        camera.read_seq.side_effect = frames
        monitor = SimpleNamespace(width=1920, height=1080, x=0, y=0, is_primary=True)
        arguments = ["handmouse-track", "--camera", "USB Camera", "--backend", backend, "--hand", "right", "--no-mirror"]
        with patch.object(sys, "argv", arguments), patch.object(tracker, "sorted_monitors", return_value=[monitor]):
            with patch.object(tracker, "FFmpegCamera", return_value=camera, side_effect=camera_error):
                with patch.object(tracker, "MouseController") as controller, patch.object(tracker, "classify", return_value=("move", {})):
                    with patch.object(tracker, "build_hands", return_value=detector) as cpu, patch("handmouse_openvino.OpenVINOHands", return_value=detector) as intel:
                        with patch.object(tracker.time, "monotonic", return_value=3), patch.object(tracker, "mp", Mock()):
                            if camera_error or stalled:
                                with self.assertRaises((OSError, RuntimeError)):
                                    tracker.main()
                            else:
                                tracker.main()
                        if backend == "intel":
                            intel.assert_called_once_with()
                            cpu.assert_not_called()
                        else:
                            cpu.assert_called_once_with(False)
                            intel.assert_not_called()
        return detector, camera, controller.return_value

    def test_each_backend_drives_existing_controller_and_cleans_up(self):
        for backend in ["mediapipe", "intel"]:
            with self.subTest(backend=backend):
                detector, camera, controller = self.run_tracker(backend)
                if backend == "intel":
                    detector.detect.assert_called_once()
                    detector.detect_for_video.assert_not_called()
                else:
                    detector.detect_for_video.assert_called_once()
                    detector.detect.assert_not_called()
                controller.update.assert_called_once()
                controller.cleanup.assert_called_once()
                camera.release.assert_called_once()
                detector.close.assert_called_once()

    def test_failed_camera_start_closes_detector(self):
        detector, _, controller = self.run_tracker("intel", camera_error=OSError("camera failed"))
        controller.cleanup.assert_called_once()
        detector.close.assert_called_once()

    def test_stalled_camera_releases_controller_and_detector(self):
        detector, camera, controller = self.run_tracker("intel", stalled=True)
        controller.cleanup.assert_called_once()
        camera.release.assert_called_once()
        detector.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
