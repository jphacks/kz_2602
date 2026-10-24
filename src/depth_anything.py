from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class DepthEstimationError(ValueError):
    """Raised when Depth Anything cannot produce a valid depth map."""


class DepthAnythingV2SmallEstimator:
    """ONNX Runtime adapter for the Depth Anything V2 Small model."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: str | Path | None,
        *,
        input_size: tuple[int, int] = (518, 518),
        session: Any | None = None,
        output_is_inverse_depth: bool = True,
    ) -> None:
        if len(input_size) != 2 or any(
            not isinstance(size, int) or size <= 0 for size in input_size
        ):
            raise DepthEstimationError("input_size must contain two positive integers")
        if session is None and model_path is None:
            raise DepthEstimationError("model_path is required when session is omitted")

        self.input_size = input_size
        self.output_is_inverse_depth = output_is_inverse_depth
        self.session = session or self._load_session(model_path)
        inputs = self.session.get_inputs()
        if not inputs:
            raise DepthEstimationError("Depth Anything model has no inputs")
        self._input_name = inputs[0].name

    @staticmethod
    def _load_session(model_path: str | Path | None) -> Any:
        try:
            import onnxruntime as ort

            return ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise DepthEstimationError(
                f"could not load Depth Anything model: {exc}"
            ) from exc

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        self._validate_image(image_rgb)
        original_height, original_width = image_rgb.shape[:2]
        model_input = self._preprocess(image_rgb)

        try:
            outputs = self.session.run(None, {self._input_name: model_input})
        except Exception as exc:
            raise DepthEstimationError(
                f"Depth Anything inference failed: {exc}"
            ) from exc

        depth_map = self._parse_output(outputs)
        if self.output_is_inverse_depth:
            finite_min = float(np.min(depth_map))
            finite_max = float(np.max(depth_map))
            depth_map = finite_max - depth_map
            if finite_min == finite_max:
                depth_map.fill(0.0)

        depth_map = cv2.resize(
            depth_map,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.asarray(depth_map, dtype=np.float32)

    @staticmethod
    def _validate_image(image_rgb: np.ndarray) -> None:
        if not isinstance(image_rgb, np.ndarray):
            raise DepthEstimationError("image_rgb must be a numpy array")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise DepthEstimationError("image_rgb must be an RGB image")
        if image_rgb.dtype != np.uint8:
            raise DepthEstimationError("image_rgb must have uint8 dtype")
        if image_rgb.shape[0] == 0 or image_rgb.shape[1] == 0:
            raise DepthEstimationError("image_rgb must not be empty")

    def _preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            image_rgb,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - self._MEAN) / self._STD
        return np.transpose(normalized, (2, 0, 1))[None, ...]

    @staticmethod
    def _parse_output(outputs: list[np.ndarray]) -> np.ndarray:
        if len(outputs) != 1:
            raise DepthEstimationError(f"expected one depth output, got {len(outputs)}")

        output = np.asarray(outputs[0])
        if output.ndim == 4 and output.shape[0] == 1 and output.shape[1] == 1:
            output = output[0, 0]
        elif output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        elif output.ndim != 2:
            raise DepthEstimationError(
                f"unsupported depth output shape: {output.shape}"
            )

        if output.size == 0 or not np.all(np.isfinite(output)):
            raise DepthEstimationError("depth output must contain finite values")
        return np.asarray(output, dtype=np.float32)
