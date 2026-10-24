"""End-to-end orchestration for food measurement and heating estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from food_heating import (
    FoodMaterialProperties,
    HeatingParameters,
    estimate_heating_parameters,
    lookup_food_properties,
)
from food_measurement import FoodMeasurement, estimate_food_measurement
from food_segmentation import FoodDetection
from turntable import (
    ObstacleMaskDetectorProtocol,
    TurntableCalibration,
    calibrate_turntable,
    fit_ellipse_from_image,
    fit_ellipse_from_image_with_obstacles,
)


class PipelineError(ValueError):
    """Raised when an end-to-end pipeline stage cannot be completed."""


class FoodSegmenterProtocol(Protocol):
    """Minimal food segmenter contract required by the pipeline."""

    def predict(self, image: np.ndarray) -> FoodDetection: ...


class DepthEstimatorProtocol(Protocol):
    """Minimal depth estimator contract required by the pipeline."""

    def predict(self, image: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Combined result from calibration through heating estimation."""

    calibration: TurntableCalibration
    detection: FoodDetection
    material: FoodMaterialProperties
    measurement: FoodMeasurement
    heating: HeatingParameters


def run_pipeline(
    image_rgb: np.ndarray,
    diameter_cm: float,
    food_segmenter: FoodSegmenterProtocol,
    depth_estimator: DepthEstimatorProtocol,
    *,
    obstacle_detector: ObstacleMaskDetectorProtocol | None = None,
) -> PipelineResult:
    """Run turntable calibration, food detection, measurement, and heating.

    The public image contract is RGB ``uint8``. Existing OpenCV-based food
    segmentation receives a converted BGR copy, while the depth estimator
    receives the original RGB image.
    """
    image_array = _validate_rgb_image(image_rgb)
    try:
        if obstacle_detector is None:
            ellipse = fit_ellipse_from_image(image_array)
        else:
            try:
                obstacle_mask = np.asarray(obstacle_detector.predict(image_array))
                if (
                    obstacle_mask.ndim != 2
                    or obstacle_mask.shape != image_array.shape[:2]
                ):
                    raise ValueError("obstacle mask must match the image dimensions")
                ellipse = fit_ellipse_from_image_with_obstacles(
                    image_array, obstacle_mask
                )
            except (AttributeError, TypeError, ValueError):
                ellipse = fit_ellipse_from_image(image_array)
        calibration = calibrate_turntable(ellipse, diameter_cm)
    except ValueError as error:
        raise PipelineError(f"turntable calibration failed: {error}") from error

    try:
        detection = food_segmenter.predict(cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    except (AttributeError, TypeError, ValueError) as error:
        raise PipelineError(f"food segmentation failed: {error}") from error

    try:
        material = lookup_food_properties(detection.class_id)
    except ValueError as error:
        raise PipelineError(f"food material lookup failed: {error}") from error

    try:
        depth_map = np.asarray(depth_estimator.predict(image_array))
        if depth_map.ndim != 2 or depth_map.shape != detection.mask.shape:
            raise PipelineError("depth_map must match the food mask dimensions")
        measurement = estimate_food_measurement(
            depth_map,
            detection,
            calibration,
            material.density_g_per_cm3,
        )
        heating = estimate_heating_parameters(detection.class_id, measurement)
    except PipelineError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise PipelineError(f"food measurement or heating failed: {error}") from error

    return PipelineResult(
        calibration=calibration,
        detection=detection,
        material=material,
        measurement=measurement,
        heating=heating,
    )


def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image)
    if (
        image_array.ndim != 3
        or image_array.shape[2] != 3
        or image_array.shape[0] == 0
        or image_array.shape[1] == 0
    ):
        raise PipelineError("image_rgb must be a non-empty RGB image")
    if image_array.dtype != np.uint8:
        raise PipelineError("image_rgb must have uint8 dtype")
    return image_array
