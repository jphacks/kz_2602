"""Shared ONNX Runtime inference for YOLOv8 segmentation models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


class YoloSegmentationError(ValueError):
    """Raised when YOLO segmentation input, output, or configuration is invalid."""


@dataclass(frozen=True, slots=True)
class YoloDetection:
    """One decoded YOLO segmentation detection in image coordinates."""

    mask: np.ndarray
    class_name: str
    confidence: float
    box: np.ndarray


@dataclass(frozen=True, slots=True)
class YoloSegmentationResult:
    """All decoded detections for one image."""

    detections: tuple[YoloDetection, ...]
    image_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _LetterboxInfo:
    scale: float
    pad_x: int
    pad_y: int
    width: int
    height: int


class YoloSegmentationEngine:
    """Run and decode a YOLOv8 segmentation ONNX model once per image."""

    def __init__(
        self,
        model_path: str | Path | None,
        class_names: Sequence[str],
        *,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        session: Any | None = None,
    ) -> None:
        if len(class_names) == 0 or any(not name for name in class_names):
            raise YoloSegmentationError("class_names must contain non-empty names")
        if len(input_size) != 2 or any(size <= 0 for size in input_size):
            raise YoloSegmentationError("input_size must contain positive dimensions")
        if not 0.0 < confidence_threshold <= 1.0:
            raise YoloSegmentationError("confidence_threshold must be in (0, 1]")
        if not 0.0 < iou_threshold <= 1.0:
            raise YoloSegmentationError("iou_threshold must be in (0, 1]")
        if session is None and model_path is None:
            raise YoloSegmentationError("model_path is required without a session")

        self._class_names = tuple(class_names)
        self._input_height, self._input_width = input_size
        self._confidence_threshold = float(confidence_threshold)
        self._iou_threshold = float(iou_threshold)
        self._session = (
            session if session is not None else self._load_session(model_path)
        )

    @staticmethod
    def _load_session(model_path: str | Path | None) -> Any:
        if model_path is None:
            raise YoloSegmentationError("model_path is required")
        try:
            import onnxruntime

            return onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            raise YoloSegmentationError("could not load the ONNX model") from error

    def predict(self, image_rgb: np.ndarray) -> YoloSegmentationResult:
        """Run inference for an RGB uint8 image and decode all detections."""
        image_array = self._validate_image(image_rgb)
        model_input, letterbox = self._preprocess(image_array)
        try:
            outputs = self._session.run(None, {self._input_name(): model_input})
        except (RuntimeError, ValueError, TypeError) as error:
            raise YoloSegmentationError("YOLO inference failed") from error

        predictions, prototypes = self._parse_outputs(outputs)
        candidates = self._select_candidates(predictions)
        detections = tuple(
            YoloDetection(
                mask=self._decode_mask(
                    prototypes,
                    coefficients,
                    box,
                    letterbox,
                    image_array.shape[:2],
                ),
                class_name=self._class_names[class_index],
                confidence=confidence,
                box=box,
            )
            for box, class_index, confidence, coefficients in candidates
        )
        return YoloSegmentationResult(detections, image_array.shape[:2])

    def _input_name(self) -> str:
        try:
            return self._session.get_inputs()[0].name
        except (AttributeError, IndexError, TypeError) as error:
            raise YoloSegmentationError("the ONNX session has no input") from error

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        image_array = np.asarray(image)
        if (
            image_array.ndim != 3
            or image_array.shape[2] != 3
            or image_array.shape[0] == 0
            or image_array.shape[1] == 0
        ):
            raise YoloSegmentationError("image must be a non-empty RGB image")
        if image_array.dtype != np.uint8:
            raise YoloSegmentationError("image must have uint8 dtype")
        return image_array

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, _LetterboxInfo]:
        image_height, image_width = image.shape[:2]
        scale = min(self._input_width / image_width, self._input_height / image_height)
        resized_width = max(1, round(image_width * scale))
        resized_height = max(1, round(image_height * scale))
        resized = cv2.resize(image, (resized_width, resized_height))
        canvas = np.full(
            (self._input_height, self._input_width, 3), 114, dtype=np.uint8
        )
        pad_x = (self._input_width - resized_width) // 2
        pad_y = (self._input_height - resized_height) // 2
        canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
        model_input = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return model_input, _LetterboxInfo(
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            width=resized_width,
            height=resized_height,
        )

    def _parse_outputs(
        self, outputs: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(outputs) != 2:
            raise YoloSegmentationError(
                "YOLO segmentation requires prediction and prototype outputs"
            )
        prediction, prototype = (np.asarray(output) for output in outputs)
        if prediction.ndim != 3 or prototype.ndim != 4:
            raise YoloSegmentationError("unexpected YOLO output dimensions")
        if prediction.shape[0] != 1 or prototype.shape[0] != 1:
            raise YoloSegmentationError("only batch size one is supported")
        prediction = prediction[0]
        prototype = prototype[0]
        if (
            prototype.shape[0] <= prototype.shape[1]
            and prototype.shape[0] <= prototype.shape[2]
        ):
            prototype = prototype.transpose(1, 2, 0)
        expected_channels = 4 + len(self._class_names) + prototype.shape[2]
        if prediction.shape[0] == expected_channels:
            prediction = prediction.T
        elif prediction.shape[1] != expected_channels:
            raise YoloSegmentationError(
                "YOLO model output does not match class_names: "
                f"output shape={prediction.shape}, prototypes="
                f"{prototype.shape[2]}, configured classes={len(self._class_names)}"
            )
        return prediction, prototype

    def _select_candidates(
        self, predictions: np.ndarray
    ) -> list[tuple[np.ndarray, int, float, np.ndarray]]:
        class_count = len(self._class_names)
        coefficient_count = predictions.shape[1] - 4 - class_count
        if coefficient_count <= 0:
            raise YoloSegmentationError("prediction output has no mask coefficients")

        boxes = predictions[:, :4]
        scores = predictions[:, 4 : 4 + class_count]
        coefficients = predictions[:, 4 + class_count :]
        class_indices = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_indices]
        valid = np.isfinite(boxes).all(axis=1) & np.isfinite(confidences)
        valid &= np.isfinite(coefficients).all(axis=1)
        valid &= confidences >= self._confidence_threshold
        if not valid.any():
            return []

        boxes = self._xywh_to_xyxy(boxes[valid])
        confidences = confidences[valid]
        class_indices = class_indices[valid]
        coefficients = coefficients[valid]
        selected: list[tuple[np.ndarray, int, float, np.ndarray]] = []
        for class_index in np.unique(class_indices):
            class_mask = class_indices == class_index
            kept = cv2.dnn.NMSBoxes(
                boxes[class_mask].tolist(),
                confidences[class_mask].astype(float).tolist(),
                self._confidence_threshold,
                self._iou_threshold,
            )
            indices = (
                np.asarray(kept).reshape(-1) if len(kept) else np.empty(0, dtype=int)
            )
            class_boxes = boxes[class_mask]
            class_confidences = confidences[class_mask]
            class_coefficients = coefficients[class_mask]
            for index in indices:
                selected.append(
                    (
                        class_boxes[index],
                        int(class_index),
                        float(class_confidences[index]),
                        class_coefficients[index],
                    )
                )
        return selected

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        result = boxes.copy()
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return result

    @staticmethod
    def _decode_mask(
        prototypes: np.ndarray,
        coefficients: np.ndarray,
        box: np.ndarray,
        letterbox: _LetterboxInfo,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        prototype_height, prototype_width, coefficient_count = prototypes.shape
        if coefficients.shape[0] != coefficient_count:
            raise YoloSegmentationError(
                "mask coefficient count does not match prototypes"
            )
        logits = prototypes.reshape(-1, coefficient_count) @ coefficients
        mask = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        mask = mask.reshape(prototype_height, prototype_width)
        scale_x = prototype_width / (letterbox.width + 2 * letterbox.pad_x)
        scale_y = prototype_height / (letterbox.height + 2 * letterbox.pad_y)
        x1, y1, x2, y2 = box * np.array([scale_x, scale_y, scale_x, scale_y])
        mask[: max(0, math.floor(y1))] = 0.0
        mask[min(prototype_height, math.ceil(y2)) :] = 0.0
        mask[:, : max(0, math.floor(x1))] = 0.0
        mask[:, min(prototype_width, math.ceil(x2)) :] = 0.0
        input_width = letterbox.width + 2 * letterbox.pad_x
        input_height = letterbox.height + 2 * letterbox.pad_y
        resized = cv2.resize(mask, (input_width, input_height))
        content = resized[
            letterbox.pad_y : letterbox.pad_y + letterbox.height,
            letterbox.pad_x : letterbox.pad_x + letterbox.width,
        ]
        restored = cv2.resize(content, (image_shape[1], image_shape[0]))
        return restored >= 0.5
