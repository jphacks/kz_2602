"""Turntable ellipse fitting and geometric calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

import cv2
import numpy as np


class TurntableCalibrationError(ValueError):
    """Raised when turntable calibration input is invalid."""


class TurntableDetectionError(TurntableCalibrationError):
    """Raised when a turntable contour cannot be detected."""


class ObstacleMaskDetectorProtocol(Protocol):
    """Minimal detector contract used by the turntable image pipeline."""

    def predict(self, image: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class EllipseParameters:
    """Ellipse parameters with major/minor axes normalized."""

    center_px: tuple[float, float]
    major_axis_px: float
    minor_axis_px: float
    angle_deg: float


@dataclass(frozen=True, slots=True)
class TurntableCalibration:
    """Absolute scale and camera-angle calibration for a turntable."""

    ellipse: EllipseParameters
    diameter_cm: float
    alpha_cm_per_px: float
    theta_rad: float


def fit_ellipse(points: Sequence[Sequence[float]] | np.ndarray) -> EllipseParameters:
    """Fit an ellipse to at least five finite ``(x, y)`` points."""
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.size == 0:
        raise TurntableCalibrationError("at least five points are required")
    if points_array.ndim == 3 and points_array.shape[1:] == (1, 2):
        points_array = points_array[:, 0, :]
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise TurntableCalibrationError("points must have shape (n, 2)")
    if len(points_array) < 5:
        raise TurntableCalibrationError("at least five points are required")
    if not np.isfinite(points_array).all():
        raise TurntableCalibrationError("points must be finite")

    try:
        (center_x, center_y), (axis_a, axis_b), angle_deg = cv2.fitEllipse(
            points_array.reshape(-1, 1, 2)
        )
    except cv2.error as error:
        raise TurntableCalibrationError("could not fit an ellipse") from error

    if axis_a >= axis_b:
        major_axis_px, minor_axis_px = axis_a, axis_b
    else:
        major_axis_px, minor_axis_px = axis_b, axis_a
        angle_deg += 90.0

    if not math.isfinite(major_axis_px) or major_axis_px <= 0:
        raise TurntableCalibrationError("ellipse major axis must be positive")
    if not math.isfinite(minor_axis_px) or minor_axis_px <= 0:
        raise TurntableCalibrationError("ellipse minor axis must be positive")

    return EllipseParameters(
        center_px=(float(center_x), float(center_y)),
        major_axis_px=float(major_axis_px),
        minor_axis_px=float(minor_axis_px),
        angle_deg=float(angle_deg % 180.0),
    )


def calibrate_turntable(
    ellipse: EllipseParameters, diameter_cm: float
) -> TurntableCalibration:
    """Convert a fitted ellipse and real diameter into calibration values."""
    if not math.isfinite(diameter_cm) or diameter_cm <= 0:
        raise TurntableCalibrationError("diameter_cm must be a positive finite value")
    if not math.isfinite(ellipse.major_axis_px) or ellipse.major_axis_px <= 0:
        raise TurntableCalibrationError("ellipse major axis must be positive")
    if not math.isfinite(ellipse.minor_axis_px) or ellipse.minor_axis_px <= 0:
        raise TurntableCalibrationError("ellipse minor axis must be positive")

    axis_ratio = min(max(ellipse.minor_axis_px / ellipse.major_axis_px, 0.0), 1.0)
    return TurntableCalibration(
        ellipse=ellipse,
        diameter_cm=float(diameter_cm),
        alpha_cm_per_px=float(diameter_cm / ellipse.major_axis_px),
        theta_rad=float(math.acos(axis_ratio)),
    )


def calibrate_from_points(
    points: Sequence[Sequence[float]] | np.ndarray, diameter_cm: float
) -> TurntableCalibration:
    """Fit an ellipse from points and return its geometric calibration."""
    return calibrate_turntable(fit_ellipse(points), diameter_cm)


def fit_ellipse_from_mask(mask: np.ndarray) -> EllipseParameters:
    """Fit the largest external contour in a binary turntable mask."""
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2 or mask_array.size == 0:
        raise TurntableDetectionError("mask must be a non-empty 2D array")

    binary_mask = np.where(mask_array > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise TurntableDetectionError("no turntable contour was found")

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        raise TurntableDetectionError("turntable contour needs at least five points")
    return fit_ellipse(contour)


def fit_ellipse_from_image(image: np.ndarray) -> EllipseParameters:
    """Find and fit the strongest ellipse-like contour in an image.

    This lightweight fallback is intended for controlled images. A supplied
    segmentation mask should be preferred when the turntable is occluded.
    """
    image_array = np.asarray(image)
    if image_array.size == 0:
        raise TurntableDetectionError("image must not be empty")
    if image_array.ndim == 2:
        grayscale = image_array
    elif image_array.ndim == 3 and image_array.shape[2] in (3, 4):
        conversion = (
            cv2.COLOR_BGR2GRAY if image_array.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        )
        grayscale = cv2.cvtColor(image_array, conversion)
    else:
        raise TurntableDetectionError("image must be grayscale, BGR, or BGRA")

    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = [contour for contour in contours if len(contour) >= 5]
    if not candidates:
        raise TurntableDetectionError("no ellipse-like contour was found")

    contour = max(candidates, key=cv2.contourArea)
    return fit_ellipse(contour)


def fit_ellipse_from_image_with_obstacles(
    image: np.ndarray, obstacle_mask: np.ndarray
) -> EllipseParameters:
    """Fit a horizontally aligned turntable ellipse after obstacle removal."""
    image_array = np.asarray(image)
    if image_array.size == 0:
        raise TurntableDetectionError("image must not be empty")
    if image_array.ndim == 2:
        grayscale = image_array
    elif image_array.ndim == 3 and image_array.shape[2] in (3, 4):
        conversion = (
            cv2.COLOR_BGR2GRAY if image_array.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        )
        grayscale = cv2.cvtColor(image_array, conversion)
    else:
        raise TurntableDetectionError("image must be grayscale, BGR, or BGRA")

    mask_array = np.asarray(obstacle_mask)
    if mask_array.ndim != 2 or mask_array.shape != grayscale.shape:
        raise TurntableDetectionError("obstacle_mask must match the image dimensions")

    height, width = grayscale.shape
    roi_y_start = int(height * 0.40)
    roi_y_end = int(height * 0.78)
    if roi_y_end <= roi_y_start:
        raise TurntableDetectionError("image is too small for the turntable ROI")

    roi = grayscale[roi_y_start:roi_y_end, :]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(roi)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3,
    )

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel_clean)
    lines = cv2.HoughLinesP(
        cleaned,
        1,
        np.pi / 180.0,
        threshold=80,
        minLineLength=50,
        maxLineGap=10,
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.ravel()
            cv2.line(cleaned, (int(x1), int(y1)), (int(x2), int(y2)), 0, 5)

    cleaned_full = np.zeros((height, width), dtype=np.uint8)
    cleaned_full[roi_y_start:roi_y_end, :] = cleaned
    dilated_obstacles = cv2.dilate(
        np.where(mask_array > 0, 255, 0).astype(np.uint8),
        np.ones((15, 15), dtype=np.uint8),
        iterations=1,
    )
    candidate_mask = cleaned_full.copy()
    candidate_mask[dilated_obstacles > 0] = 0
    candidate_mask[:, : int(width * 0.12)] = 0
    candidate_mask[:, int(width * 0.88) :] = 0

    y_points, x_points = np.where(candidate_mask > 0)
    if len(x_points) <= 50:
        raise TurntableDetectionError("not enough points for turntable fitting")

    distance = np.abs(x_points - width / 2.0)
    valid = (distance > width * 0.10) & (distance < width * 0.42)
    points = np.column_stack((x_points[valid], y_points[valid])).astype(np.float32)
    if len(points) <= 20:
        raise TurntableDetectionError(
            "not enough filtered points for turntable fitting"
        )

    try:
        (center_x, center_y), (axis_a, axis_b), _ = cv2.fitEllipseDirect(
            points.reshape(-1, 1, 2)
        )
    except cv2.error as error:
        raise TurntableDetectionError(
            "could not fit the filtered turntable ellipse"
        ) from error

    major_axis, minor_axis = sorted((float(axis_a), float(axis_b)), reverse=True)
    if (
        not math.isfinite(center_y)
        or not math.isfinite(major_axis)
        or not math.isfinite(minor_axis)
        or major_axis <= 0
        or minor_axis <= 0
    ):
        raise TurntableDetectionError("filtered ellipse parameters are invalid")
    return EllipseParameters(
        center_px=(width / 2.0, float(center_y)),
        major_axis_px=major_axis,
        minor_axis_px=minor_axis,
        angle_deg=0.0,
    )
