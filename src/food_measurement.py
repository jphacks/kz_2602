"""Food height, area, volume, and weight estimation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from food_segmentation import FoodDetection
from turntable import TurntableCalibration


class FoodMeasurementError(ValueError):
    """Raised when food measurement input or geometry is invalid."""


@dataclass(frozen=True, slots=True)
class FoodMeasurement:
    """Estimated dimensions and mass for one detected food item."""

    top_x_px: int
    top_y_px: int
    height_cm: float
    area_cm2: float
    volume_cm3: float
    weight_g: float


def find_top_vertex(depth_map: np.ndarray, food_mask: np.ndarray) -> tuple[int, int]:
    """Return the nearest finite-depth pixel inside ``food_mask``."""
    depth_array, mask_array = _validate_maps(depth_map, food_mask)
    candidate_mask = mask_array & np.isfinite(depth_array)
    if not candidate_mask.any():
        raise FoodMeasurementError(
            "food mask must contain at least one finite depth value"
        )

    candidate_depths = np.where(candidate_mask, depth_array, np.inf)
    top_y_px, top_x_px = np.unravel_index(
        int(np.argmin(candidate_depths)), candidate_depths.shape
    )
    return int(top_x_px), int(top_y_px)


def estimate_food_measurement(
    depth_map: np.ndarray,
    detection: FoodDetection,
    calibration: TurntableCalibration,
    density_g_per_cm3: float,
) -> FoodMeasurement:
    """Estimate food dimensions and weight from depth, mask, and calibration."""
    depth_array, mask_array = _validate_maps(depth_map, detection.mask)
    _validate_calibration(calibration)
    if not math.isfinite(density_g_per_cm3) or density_g_per_cm3 <= 0:
        raise FoodMeasurementError("density_g_per_cm3 must be a positive finite value")
    if not 0 <= detection.y_base_px < mask_array.shape[0]:
        raise FoodMeasurementError("y_base_px must be inside the food mask")

    top_x_px, top_y_px = find_top_vertex(depth_array, mask_array)
    pixel_height = detection.y_base_px - top_y_px
    if pixel_height < 0:
        raise FoodMeasurementError("top vertex must not be below the food base")

    theta = calibration.theta_rad
    sine_theta = math.sin(theta)
    if abs(sine_theta) < 1e-8:
        raise FoodMeasurementError(
            "height cannot be estimated for a near-top-down view"
        )

    alpha = calibration.alpha_cm_per_px
    height_cm = pixel_height * alpha / sine_theta
    area_cm2 = int(np.count_nonzero(mask_array)) * alpha**2 * math.cos(theta)
    volume_cm3 = (2.0 / 3.0) * area_cm2 * height_cm
    weight_g = volume_cm3 * density_g_per_cm3
    values = (height_cm, area_cm2, volume_cm3, weight_g)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise FoodMeasurementError("measurement result must be finite and non-negative")

    return FoodMeasurement(
        top_x_px=top_x_px,
        top_y_px=top_y_px,
        height_cm=float(height_cm),
        area_cm2=float(area_cm2),
        volume_cm3=float(volume_cm3),
        weight_g=float(weight_g),
    )


def _validate_maps(
    depth_map: np.ndarray, food_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    depth_array = np.asarray(depth_map)
    mask_array = np.asarray(food_mask)
    if depth_array.ndim != 2 or mask_array.ndim != 2:
        raise FoodMeasurementError("depth_map and food_mask must be 2D arrays")
    if depth_array.shape != mask_array.shape:
        raise FoodMeasurementError("depth_map and food_mask must have the same shape")
    if depth_array.size == 0 or not mask_array.any():
        raise FoodMeasurementError("food_mask must be non-empty")
    return depth_array, mask_array.astype(bool, copy=False)


def _validate_calibration(calibration: TurntableCalibration) -> None:
    alpha = calibration.alpha_cm_per_px
    theta = calibration.theta_rad
    if not math.isfinite(alpha) or alpha <= 0:
        raise FoodMeasurementError("alpha_cm_per_px must be positive and finite")
    if not math.isfinite(theta) or not 0 <= theta <= math.pi / 2:
        raise FoodMeasurementError("theta_rad must be finite in [0, pi/2]")
