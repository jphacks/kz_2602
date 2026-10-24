from __future__ import annotations

import math

import numpy as np
import pytest

import cv2
from turntable import (
    EllipseParameters,
    TurntableCalibrationError,
    calibrate_turntable,
    fit_ellipse_from_image,
    fit_ellipse_from_mask,
    fit_ellipse,
)


def ellipse_points(
    major_axis: float, minor_axis: float, angle_deg: float = 0.0
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, 80, endpoint=False)
    angle_rad = math.radians(angle_deg)
    x = major_axis * np.cos(angles)
    y = minor_axis * np.sin(angles)
    return np.c_[
        x * math.cos(angle_rad) - y * math.sin(angle_rad) + 100.0,
        x * math.sin(angle_rad) + y * math.cos(angle_rad) + 80.0,
    ]


def test_fit_ellipse_normalizes_axes_and_preserves_center() -> None:
    ellipse = fit_ellipse(ellipse_points(80.0, 40.0, 25.0))

    assert ellipse.center_px == pytest.approx((100.0, 80.0), abs=0.1)
    assert ellipse.major_axis_px == pytest.approx(160.0, abs=0.1)
    assert ellipse.minor_axis_px == pytest.approx(80.0, abs=0.1)


def test_calibration_computes_scale_and_viewing_angle() -> None:
    ellipse = fit_ellipse(ellipse_points(80.0, 40.0))

    calibration = calibrate_turntable(ellipse, 20.0)

    assert calibration.alpha_cm_per_px == pytest.approx(0.125)
    assert calibration.theta_rad == pytest.approx(math.acos(0.5))


def test_circle_projection_has_zero_viewing_angle() -> None:
    ellipse = fit_ellipse(ellipse_points(60.0, 60.0))

    calibration = calibrate_turntable(ellipse, 24.0)

    assert calibration.theta_rad == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("points", "diameter_cm"),
    [
        (np.empty((0, 2)), 20.0),
        (np.ones((4, 2)), 20.0),
        (np.full((5, 2), np.nan), 20.0),
        (ellipse_points(80.0, 40.0), 0.0),
        (ellipse_points(80.0, 40.0), -1.0),
    ],
)
def test_invalid_inputs_raise_calibration_error(
    points: np.ndarray, diameter_cm: float
) -> None:
    if len(points) < 5 or not np.isfinite(points).all():
        with pytest.raises(TurntableCalibrationError):
            fit_ellipse(points)
        return

    ellipse = fit_ellipse(points)
    with pytest.raises(TurntableCalibrationError):
        calibrate_turntable(ellipse, diameter_cm)


def test_invalid_ellipse_is_rejected() -> None:
    ellipse = EllipseParameters(
        center_px=(0.0, 0.0),
        major_axis_px=0.0,
        minor_axis_px=1.0,
        angle_deg=0.0,
    )

    with pytest.raises(TurntableCalibrationError):
        calibrate_turntable(ellipse, 20.0)


def test_fit_ellipse_from_mask_uses_largest_external_contour() -> None:
    mask = np.zeros((220, 260), dtype=np.uint8)
    cv2.ellipse(mask, (130, 110), (80, 40), 25.0, 0.0, 360.0, 255, -1)

    ellipse = fit_ellipse_from_mask(mask)

    assert ellipse.center_px == pytest.approx((130.0, 110.0), abs=1.0)
    assert ellipse.major_axis_px == pytest.approx(160.0, abs=1.0)
    assert ellipse.minor_axis_px == pytest.approx(80.0, abs=1.0)


def test_fit_ellipse_from_image_detects_basic_ellipse() -> None:
    image = np.zeros((220, 260), dtype=np.uint8)
    cv2.ellipse(image, (130, 110), (80, 40), 0.0, 0.0, 360.0, 255, -1)

    ellipse = fit_ellipse_from_image(image)

    assert ellipse.center_px == pytest.approx((130.0, 110.0), abs=1.0)
    assert ellipse.major_axis_px == pytest.approx(160.0, abs=2.0)
    assert ellipse.minor_axis_px == pytest.approx(80.0, abs=2.0)


@pytest.mark.parametrize("invalid_input", [np.empty((0, 0)), np.zeros((10, 10, 2))])
def test_invalid_detection_input_raises_calibration_error(
    invalid_input: np.ndarray,
) -> None:
    with pytest.raises(TurntableCalibrationError):
        fit_ellipse_from_mask(invalid_input)

    with pytest.raises(TurntableCalibrationError):
        fit_ellipse_from_image(invalid_input)
