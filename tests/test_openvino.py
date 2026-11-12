import os
import sys
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from handmouse_openvino import OpenVINOHands, crop_and_pad, integrated_gpu, palm_anchors


class IntelBackendTests(unittest.TestCase):
    def test_selects_integrated_gpu_not_discrete_or_cpu(self):
        core = Mock()
        core.available_devices = ["CPU", "GPU.0", "GPU.1"]
        core.get_property.side_effect = lambda device, _: "Type.INTEGRATED" if device == "GPU.1" else "Type.DISCRETE"
        self.assertEqual(integrated_gpu(core), "GPU.1")
        core.available_devices = ["CPU", "GPU.0"]
        with self.assertRaisesRegex(RuntimeError, "No Intel integrated GPU"):
            integrated_gpu(core)

    def test_anchors_match_model_order(self):
        anchors = palm_anchors()
        self.assertEqual(anchors.shape, (2016, 2))
        np.testing.assert_allclose(anchors[0], [0.5 / 24, 0.5 / 24])
        np.testing.assert_array_equal(anchors[0], anchors[1])
        np.testing.assert_allclose(anchors[2], [1.5 / 24, 0.5 / 24])
        np.testing.assert_allclose(anchors[1152:1158], np.tile([0.5 / 12, 0.5 / 12], (6, 1)))

    def test_empty_crop_is_rejected(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop, _, _ = crop_and_pad(frame, np.array([[200, 200], [210, 210]]))
        self.assertIsNone(crop)

    def test_palm_nms_suppresses_duplicates_without_losing_second_hand(self):
        detector = OpenVINOHands.__new__(OpenVINOHands)
        detector.anchors = palm_anchors()
        deltas = np.zeros((1, 2016, 18), dtype=np.float32)
        logits = np.full((1, 2016, 1), -20, dtype=np.float32)
        for index, center in [(0, [80, 80]), (1, [80, 80]), (3, [240, 120])]:
            offset = (np.array(center) / 320 - detector.anchors[index]) * 192
            deltas[0, index, :2] = offset
            deltas[0, index, 2:4] = [32 / 320 * 192, 64 / 320 * 192]
            deltas[0, index, 4:] = np.tile(offset, 7)
            logits[0, index, 0] = 10
        detector._infer = Mock(return_value=[deltas, logits])
        palms = detector._palms(np.zeros((320, 320, 3), dtype=np.uint8))
        self.assertEqual(len(palms), 2)
        np.testing.assert_allclose(sorted(float((p[0] + p[2]) / 2) for p in palms), [80, 240], atol=0.001)
        tensor = detector._infer.call_args.args[1]
        self.assertEqual(tensor.shape, (1, 192, 192, 3))
        self.assertEqual(tensor.dtype, np.float32)

    def test_projection_normalizes_pixels_and_preserves_hand_rotation(self):
        detector = OpenVINOHands.__new__(OpenVINOHands)
        points = np.tile([112, 112, 0], (21, 1)).astype(np.float32)
        points[8] = [112, 56, 10]
        detector._infer = Mock(return_value=[points[None].reshape(1, 63), np.array([[0.9]]), np.array([[0.8]]), points[None].reshape(1, 63)])
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        palm_points = np.array([[200, 220], [180, 200], [200, 180], [210, 190], [220, 200], [190, 210], [185, 205]])
        palm = np.r_[[180, 180, 220, 220], palm_points.reshape(-1)]
        landmarks, handedness = detector._pose(frame, palm)
        self.assertEqual(len(landmarks), 21)
        self.assertEqual(handedness[0].category_name, "Right")
        self.assertAlmostEqual(handedness[0].score, 0.8)
        self.assertAlmostEqual(landmarks[0].x, 0.5)
        self.assertAlmostEqual(landmarks[0].y, 184 / 400)
        self.assertAlmostEqual(landmarks[8].y, 154 / 400, places=5)
        rotation = cv2.getRotationMatrix2D((200, 200), 90, 1)
        rotated_points = np.c_[palm_points, np.ones(7)] @ rotation.T
        rotated_palm = np.r_[[180, 180, 220, 220], rotated_points.reshape(-1)]
        rotated, _ = detector._pose(frame, rotated_palm)
        for original, actual in zip(landmarks, rotated):
            expected = rotation @ [original.x * 400, original.y * 400, 1]
            np.testing.assert_allclose([actual.x, actual.y], expected / 400, atol=0.003)
            self.assertAlmostEqual(actual.z, original.z, places=5)

    def test_empty_detections_do_not_move_cursor(self):
        detector = OpenVINOHands.__new__(OpenVINOHands)
        detector._palms = Mock(return_value=[])
        result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(result.hand_landmarks, [])
        self.assertEqual(result.handedness, [])

    def test_low_or_invalid_confidence_pose_is_rejected(self):
        detector = OpenVINOHands.__new__(OpenVINOHands)
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        palm_points = np.array([[200, 220], [180, 200], [200, 180], [210, 190], [220, 200], [190, 210], [185, 205]])
        palm = np.r_[[180, 180, 220, 220], palm_points.reshape(-1)]
        for confidence in [0.1, float("nan")]:
            detector._infer = Mock(return_value=[np.zeros((1, 63)), np.array([[confidence]]), np.array([[0.8]]), np.zeros((1, 63))])
            self.assertIsNone(detector._pose(frame, palm))


@unittest.skipUnless(os.environ.get("HANDMOUSE_OPENVINO_SMOKE") == "1", "opt-in real model/network test")
class OpenVINOModelSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import urllib.request

        with urllib.request.urlopen("https://storage.googleapis.com/mediapipe-assets/right_hands.jpg", timeout=30) as response:
            cls.frame = cv2.imdecode(np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if cls.frame is None:
            raise RuntimeError("Could not decode reference image")
        detector = OpenVINOHands(device="CPU")
        cls.addClassCleanup(detector.close)
        cls.result = detector.detect(cls.frame)

    def test_real_models_on_cpu(self):
        self.assertGreaterEqual(len(self.result.hand_landmarks), 1)
        for landmarks in self.result.hand_landmarks:
            self.assertEqual(len(landmarks), 21)
            coordinates = np.array([[point.x, point.y, point.z] for point in landmarks])
            self.assertTrue(np.isfinite(coordinates).all())
            self.assertTrue(((coordinates[:, :2] >= -0.1) & (coordinates[:, :2] <= 1.1)).all())

    @unittest.skipUnless(sys.platform == "win32", "MediaPipe native comparison runs on Windows CI")
    def test_matches_mediapipe_reference(self):
        from mediapipe.tasks.python import BaseOptions, vision
        import mediapipe as mp
        from handmouse_runtime import ensure_model

        options = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=ensure_model()), num_hands=2)
        with vision.HandLandmarker.create_from_options(options) as reference:
            expected = reference.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)))
        self.assertGreaterEqual(len(expected.hand_landmarks), 1)
        self.assertGreaterEqual(len(self.result.hand_landmarks), 1)
        for landmarks, handedness in zip(self.result.hand_landmarks, self.result.handedness):
            coordinates = np.array([[point.x, point.y] for point in landmarks])
            distances = [np.linalg.norm(coordinates - np.array([[point.x, point.y] for point in other]), axis=1).mean()
                         for other in expected.hand_landmarks]
            best = int(np.argmin(distances))
            self.assertLess(distances[best], 0.08, "OpenVINO projection differs from MediaPipe")
            self.assertEqual(handedness[0].category_name, expected.handedness[best][0].category_name)


if __name__ == "__main__":
    unittest.main()
