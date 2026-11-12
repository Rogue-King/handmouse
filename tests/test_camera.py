import io
import subprocess
import unittest
from unittest.mock import Mock, patch

from handmouse_camera import FFmpegCamera, processing_size


class CameraTests(unittest.TestCase):
    def make_camera(self, width=4, height=2, content=b""):
        process = Mock()
        process.stdout = io.BytesIO(content)
        with patch("handmouse_camera.ffmpeg_exe", return_value="ffmpeg"):
            with patch("handmouse_camera.subprocess.Popen", return_value=process) as spawn:
                with patch("handmouse_camera.threading.Thread"):
                    camera = FFmpegCamera("USB Camera", width, height, 60)
        return camera, process, spawn.call_args.args[0]

    def test_resize_preserves_aspect_ratio_and_never_upscales(self):
        for source, expected in [((1920, 1080), (640, 360)), ((1080, 1920), (270, 480)),
                                 ((640, 480), (640, 480)), ((320, 240), (320, 240))]:
            self.assertEqual(processing_size(*source), expected)
        for source in [(0, 480), (640, -1)]:
            with self.assertRaises(ValueError):
                processing_size(*source)

    def test_downscales_before_pipe_without_changing_requested_capture_mode(self):
        camera, _, command = self.make_camera(1920, 1080)
        self.assertEqual(command[command.index("-video_size") + 1], "1920x1080")
        self.assertEqual(command[command.index("-framerate") + 1], "60")
        self.assertEqual(command[command.index("-vf") + 1], "scale=640:360:flags=fast_bilinear")
        self.assertLess(command.index("-i"), command.index("-vf"))
        self.assertEqual(camera.frame_bytes, 640 * 360 * 3)

    def test_older_complete_frames_are_dropped_and_partial_frames_are_not_published(self):
        first, second = bytes([1]) * 24, bytes([2]) * 24
        camera, _, _ = self.make_camera(content=first + second + bytes([3]) * 12)
        camera._reader()
        sequence, frame, received_at = camera.read_seq()
        self.assertEqual(sequence, 2)
        self.assertEqual(frame.shape, (2, 4, 3))
        self.assertTrue((frame == 2).all())
        self.assertGreater(received_at, 0)
        frame[:] = 0
        self.assertTrue((camera.read_seq()[1] == 2).all())

    def test_fragmented_reads_reconstruct_complete_frames(self):
        frame = bytes(range(24))
        camera, process, _ = self.make_camera()
        process.stdout = Mock()
        process.stdout.read.side_effect = [frame[:5], frame[5:11], frame[11:], b""]
        camera._reader()
        sequence, result, _ = camera.read_seq()
        self.assertEqual(sequence, 1)
        self.assertEqual(result.tobytes(), frame)

    def test_no_frame_returns_empty_slot(self):
        camera, _, _ = self.make_camera()
        self.assertEqual(camera.read_seq(), (0, None, 0.0))

    def test_release_kills_process_if_termination_times_out(self):
        camera, process, _ = self.make_camera()
        process.wait.side_effect = [subprocess.TimeoutExpired("ffmpeg", 3), 0]
        camera.release()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        camera._thread.join.assert_called_once_with(timeout=3)
        self.assertTrue(process.stdout.closed)


if __name__ == "__main__":
    unittest.main()
