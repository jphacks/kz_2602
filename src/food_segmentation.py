"""Food segmentation and classification with a YOLO segmentation model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from yolo_segmentation import (
    YoloSegmentationEngine,
    YoloSegmentationError,
    YoloSegmentationResult,
)


class FoodSegmentationError(ValueError):
    """Raised when food segmentation input, output, or configuration is invalid."""


@dataclass(frozen=True, slots=True)
class FoodDetection:
    """A single food detection in the original image coordinate system."""

    mask: np.ndarray
    class_id: str
    confidence: float
    y_base_px: int


class FoodSegmenter:
    """Run a YOLOv8 segmentation ONNX model for one food item per image."""

    def __init__(
        self,
        model_path: str | Path | None,
        class_names: Sequence[str],
        *,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_aliases: Mapping[str, str] | None = None,
        session: Any | None = None,
        engine: YoloSegmentationEngine | None = None,
    ) -> None:
        if len(class_names) == 0 or any(not name for name in class_names):
            raise FoodSegmentationError("class_names must contain non-empty names")
        if len(input_size) != 2 or any(size <= 0 for size in input_size):
            raise FoodSegmentationError("input_size must contain positive dimensions")
        if not 0.0 < confidence_threshold <= 1.0:
            raise FoodSegmentationError("confidence_threshold must be in (0, 1]")
        if not 0.0 < iou_threshold <= 1.0:
            raise FoodSegmentationError("iou_threshold must be in (0, 1]")
        if engine is None and session is None and model_path is None:
            raise FoodSegmentationError("model_path is required without a session")

        self._class_names = tuple(class_names)
        self._input_height, self._input_width = input_size
        self._confidence_threshold = float(confidence_threshold)
        self._iou_threshold = float(iou_threshold)
        self._class_aliases = dict(class_aliases or {})
        if any(
            not source or not target for source, target in self._class_aliases.items()
        ):
            raise FoodSegmentationError("class_aliases must contain non-empty names")
        if engine is not None and session is not None:
            raise FoodSegmentationError("engine and session cannot both be provided")
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
            raise FoodSegmentationError(str(error)) from error

    def predict(self, image: np.ndarray) -> FoodDetection:
        """Return the only food detection in ``image``."""
        image_array = self._validate_image(image)
        try:
            result = self._engine.predict(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
            return self.select(result)
        except YoloSegmentationError as error:
            raise FoodSegmentationError(str(error)) from error

    def select(self, result: YoloSegmentationResult) -> FoodDetection:
        """Select and classify the only configured food detection."""
        allowed_names = set(self._class_aliases) or set(self._class_names)
        detections = [
            detection
            for detection in result.detections
            if detection.class_name in allowed_names
        ]
        if len(detections) != 1:
            raise FoodSegmentationError(
                "expected exactly one food detection, " f"but found {len(detections)}"
            )
        detection = detections[0]
        if not detection.mask.any():
            raise FoodSegmentationError("the decoded food mask is empty")
        return FoodDetection(
            mask=detection.mask,
            class_id=self._class_aliases.get(
                detection.class_name, detection.class_name
            ),
            confidence=detection.confidence,
            y_base_px=int(np.flatnonzero(detection.mask.any(axis=1)).max()),
        )

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        image_array = np.asarray(image)
        if (
            image_array.ndim != 3
            or image_array.shape[2] != 3
            or image_array.shape[0] == 0
            or image_array.shape[1] == 0
        ):
            raise FoodSegmentationError("image must be a non-empty BGR image")
        if image_array.dtype != np.uint8:
            raise FoodSegmentationError("image must have uint8 dtype")
        return image_array

    def _parse_outputs(
        self, outputs: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            return self._engine._parse_outputs(outputs)
        except YoloSegmentationError as error:
            raise FoodSegmentationError(str(error)) from error
