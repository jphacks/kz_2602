"""YOLOv8 segmentation adapter for turntable obstacle masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from yolo_segmentation import (
    YoloSegmentationEngine,
    YoloSegmentationError,
    YoloSegmentationResult,
)


class YoloObstacleDetector:
    """Combine all YOLO segmentation instances into one image-sized mask."""

    def __init__(
        self,
        model_path: str | Path | None,
        class_names: Sequence[str],
        *,
        obstacle_class_names: Sequence[str] | None = None,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        session: Any | None = None,
        engine: YoloSegmentationEngine | None = None,
    ) -> None:
        self._class_names = tuple(class_names)
        self._obstacle_class_names = frozenset(
            self._class_names if obstacle_class_names is None else obstacle_class_names
        )
        if not self._obstacle_class_names.issubset(self._class_names):
            raise ValueError("obstacle_class_names must be model class names")
        if engine is not None and session is not None:
            raise ValueError("engine and session cannot both be provided")
        try:
            self._engine = engine or YoloSegmentationEngine(
                model_path,
                self._class_names,
                input_size=input_size,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                session=session,
            )
        except YoloSegmentationError as error:
            raise ValueError(str(error)) from error

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference for an RGB image and combine obstacle masks."""
        image_array = np.asarray(image)
        if (
            image_array.ndim != 3
            or image_array.shape[2] != 3
            or image_array.size == 0
            or image_array.dtype != np.uint8
        ):
            raise ValueError("image must be a non-empty array")
        try:
            return self.select(self._engine.predict(image_array))
        except YoloSegmentationError as error:
            raise ValueError("obstacle model prediction failed") from error

    def select(self, result: YoloSegmentationResult) -> np.ndarray:
        """Combine obstacle-class masks from a shared segmentation result."""
        combined = np.zeros(result.image_shape, dtype=bool)
        for detection in result.detections:
            if detection.class_name in self._obstacle_class_names:
                combined |= detection.mask
        return combined
