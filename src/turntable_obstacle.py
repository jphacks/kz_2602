"""YOLOv8 segmentation adapter for turntable obstacle masks."""

from __future__ import annotations

import cv2
import numpy as np


class YoloObstacleDetector:
    """Combine all YOLO segmentation instances into one image-sized mask."""

    def __init__(
        self,
        model_path: str,
        *,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ValueError("obstacle detection requires ultralytics") from error
        self._model = YOLO(model_path)
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold

    def predict(self, image: np.ndarray) -> np.ndarray:
        image_array = np.asarray(image)
        if image_array.ndim not in (2, 3) or image_array.size == 0:
            raise ValueError("image must be a non-empty array")
        height, width = image_array.shape[:2]
        try:
            result = self._model(
                image_array,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                verbose=False,
            )[0]
        except (AttributeError, RuntimeError, ValueError) as error:
            raise ValueError("obstacle model prediction failed") from error

        combined = np.zeros((height, width), dtype=bool)
        if result.masks is None:
            return combined
        for mask in result.masks.data.cpu().numpy():
            resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            combined |= resized > 0.5
        return combined
