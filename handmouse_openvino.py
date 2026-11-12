"""Intel GPU inference using the two TFLite models in the existing hand task.

Crop/projection geometry adapted from OpenCV Zoo's mp_handpose.py (Apache-2.0):
https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe
Changes: OpenVINO inference, generated anchors, xywh NMS, normalized results.
See licenses/opencv-zoo.txt.
"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np

from handmouse_runtime import ensure_model


def integrated_gpu(core):
    for device in core.available_devices:
        if device.startswith("GPU") and str(core.get_property(device, "DEVICE_TYPE")).upper().endswith("INTEGRATED"):
            return device
    raise RuntimeError("No Intel integrated GPU found by OpenVINO. Update your Intel graphics driver or select CPU (MediaPipe).")


def palm_anchors():
    return np.array([((x + 0.5) / grid, (y + 0.5) / grid)
                     for grid, repeats in ((24, 2), (12, 6))
                     for y in range(grid) for x in range(grid) for _ in range(repeats)], dtype=np.float32)


def crop_and_pad(image, box, for_rotation=False):
    size = box[1] - box[0]
    shifted = box + np.array([0, 0 if for_rotation else -0.4]) * size
    center = shifted.mean(axis=0)
    half = size * (4 if for_rotation else 3) / 2
    box = np.array([center - half, center + half]).astype(np.int32)
    box[:, 0] = np.clip(box[:, 0], 0, image.shape[1])
    box[:, 1] = np.clip(box[:, 1], 0, image.shape[0])
    crop = image[box[0, 1]:box[1, 1], box[0, 0]:box[1, 0]]
    if crop.size == 0:
        return None, box, None
    side = int(np.linalg.norm(crop.shape[:2])) if for_rotation else max(crop.shape[:2])
    pad_h, pad_w = side - crop.shape[0], side - crop.shape[1]
    left, top = pad_w // 2, pad_h // 2
    crop = cv2.copyMakeBorder(crop, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT)
    return crop, box, box[0] - [left, top]


class OpenVINOHands:
    def __init__(self, device=None):
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("Intel GPU support is not installed. Run: uv sync --extra intel --managed-python") from exc
        core = ov.Core()
        self.device = integrated_gpu(core) if device is None else device
        config = {"PERFORMANCE_HINT": "LATENCY"}
        self._models = []
        self._requests = []
        with zipfile.ZipFile(ensure_model()) as archive, tempfile.TemporaryDirectory() as directory:
            for name in ("hand_detector.tflite", "hand_landmarks_detector.tflite"):
                path = Path(directory) / name
                path.write_bytes(archive.read(name))
                try:
                    model = core.read_model(str(path))
                    compiled = core.compile_model(model, self.device, config)
                except RuntimeError as exc:
                    raise RuntimeError(f"Could not compile {name} on {self.device}. Update your Intel graphics driver or select CPU (MediaPipe). {exc}") from exc
                self._models.append(compiled)
                self._requests.append(compiled.create_infer_request())
        self.anchors = palm_anchors()
        print(f"Hand detector: OpenVINO {self.device} ({core.get_property(self.device, 'FULL_DEVICE_NAME')})", flush=True)

    def _infer(self, index, tensor):
        results = self._requests[index].infer({0: np.ascontiguousarray(tensor, dtype=np.float32)})
        model = self._models[index]
        names = ("Identity", "Identity_1") if index == 0 else ("Identity", "Identity_1", "Identity_2", "Identity_3")
        return [results[model.output(name)] for name in names]

    def _palms(self, frame):
        height, width = frame.shape[:2]
        ratio = 192 / max(width, height)
        new_w, new_h = max(1, int(width * ratio)), max(1, int(height * ratio))
        image = cv2.resize(frame, (new_w, new_h))
        left, top = (192 - new_w) // 2, (192 - new_h) // 2
        image = cv2.copyMakeBorder(image, top, 192 - new_h - top, left, 192 - new_w - left, cv2.BORDER_CONSTANT)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        deltas, logits = self._infer(0, image[None])
        scores = 1 / (1 + np.exp(-np.clip(logits[0, :, 0], -80, 80)))
        centers = (deltas[0, :, :2] / 192 + self.anchors) * max(width, height)
        sizes = deltas[0, :, 2:4] / 192 * max(width, height)
        bias = np.array([left, top]) / ratio
        boxes = np.c_[centers - sizes / 2 - bias, centers + sizes / 2 - bias]
        landmarks = (deltas[0, :, 4:].reshape(-1, 7, 2) / 192 + self.anchors[:, None]) * max(width, height) - bias
        valid = (scores >= 0.6) & np.isfinite(boxes).all(axis=1) & np.isfinite(landmarks).all(axis=(1, 2)) & (sizes > 0).all(axis=1)
        boxes, landmarks, scores = boxes[valid], landmarks[valid], scores[valid]
        if not len(scores):
            return []
        # OpenCV NMS expects x/y/width/height, not x1/y1/x2/y2.
        xywh = np.c_[boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]
        keep = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.6, 0.3)).reshape(-1)[:2]
        return [np.r_[boxes[i], landmarks[i].reshape(-1)] for i in keep]

    def _pose(self, frame, palm):
        image, box, bias = crop_and_pad(frame, palm[:4].reshape(2, 2), True)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        box -= bias
        palm_points = palm[4:18].reshape(7, 2) - bias
        wrist, middle = palm_points[0], palm_points[2]
        radians = np.pi / 2 - np.arctan2(-(middle[1] - wrist[1]), middle[0] - wrist[0])
        radians -= 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = np.rad2deg(radians)
        rotation = cv2.getRotationMatrix2D(box.mean(axis=0), angle, 1)
        image = cv2.warpAffine(image, rotation, (image.shape[1], image.shape[0]))
        rotated = np.c_[palm_points, np.ones(7)] @ rotation.T
        rotated_box = np.array([rotated.min(axis=0), rotated.max(axis=0)])
        crop, rotated_box, _ = crop_and_pad(image, rotated_box)
        if crop is None:
            return None
        tensor = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        points, confidence, handedness, _ = self._infer(1, tensor[None])
        confidence = float(confidence.ravel()[0])
        if not np.isfinite(confidence) or confidence < 0.6:
            return None
        points = points.reshape(21, 3).copy()
        scale = max(rotated_box[1] - rotated_box[0]) / 224
        points[:, :2] = (points[:, :2] - 112) * scale
        points[:, 2] *= scale
        points[:, :2] = points[:, :2] @ cv2.getRotationMatrix2D((0, 0), angle, 1)[:, :2]
        center = cv2.invertAffineTransform(rotation) @ np.r_[rotated_box.mean(axis=0), 1]
        points[:, :2] += center + bias
        points /= np.array([frame.shape[1], frame.shape[0], frame.shape[1]])
        right_probability = float(handedness.ravel()[0])
        if not np.isfinite(points).all() or not np.isfinite(right_probability):
            return None
        label = "Right" if right_probability >= 0.5 else "Left"
        score = right_probability if label == "Right" else 1 - right_probability
        landmarks = [SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x, y, z in points]
        return landmarks, [SimpleNamespace(category_name=label, score=score)]

    def detect(self, frame):
        result = SimpleNamespace(hand_landmarks=[], handedness=[])
        for palm in self._palms(frame):
            pose = self._pose(frame, palm)
            if pose is not None:
                landmarks, handedness = pose
                result.hand_landmarks.append(landmarks)
                result.handedness.append(handedness)
        return result

    def close(self):
        self._requests.clear()
        self._models.clear()
