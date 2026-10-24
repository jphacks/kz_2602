from __future__ import annotations

import math

import numpy as np
import pytest

from food_measurement import (
    FoodMeasurementError,
    estimate_food_measurement,
    find_top_vertex,
)
from food_segmentation import FoodDetection
from turntable import EllipseParameters, calibrate_turntable


def make_detection(mask: np.ndarray, y_base_px: int) -> FoodDetection:
    return FoodDetection(
        mask=mask,
        class_id="rice",
        confidence=0.9,
        y_base_px=y_base_px,
    )


def make_calibration(alpha: float = 0.5, theta: float = math.pi / 6):
    ellipse = EllipseParameters(
        center_px=(0.0, 0.0),
        major_axis_px=20.0,
        minor_axis_px=20.0 * math.cos(theta),
        angle_deg=0.0,
    )
    calibration = calibrate_turntable(ellipse, alpha * ellipse.major_axis_px)
    return calibration


def test_find_top_vertex_chooses_minimum_finite_depth() -> None:
    depth_map = np.array([[5.0, np.nan], [3.0, 1.0]])
    food_mask = np.array([[True, True], [True, True]])

    assert find_top_vertex(depth_map, food_mask) == (1, 1)


def test_find_top_vertex_uses_row_major_order_for_ties() -> None:
    depth_map = np.array([[2.0, 1.0], [1.0, 4.0]])
    food_mask = np.ones((2, 2), dtype=bool)

    assert find_top_vertex(depth_map, food_mask) == (1, 0)


def test_estimate_food_measurement_computes_height_area_volume_and_weight() -> None:
    depth_map = np.array(
        [
            [9.0, 9.0, 9.0],
            [9.0, 1.0, 9.0],
            [9.0, 2.0, 9.0],
            [9.0, 3.0, 9.0],
        ]
    )
    food_mask = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, True, False],
            [False, True, False],
        ]
    )
    measurement = estimate_food_measurement(
        depth_map,
        make_detection(food_mask, y_base_px=3),
        make_calibration(),
        density_g_per_cm3=1.0,
    )

    expected_area = 3.0 * 0.5**2 * math.cos(math.pi / 6)
    expected_height = 2.0 * 0.5 / math.sin(math.pi / 6)
    expected_volume = (2.0 / 3.0) * expected_area * expected_height
    assert (measurement.top_x_px, measurement.top_y_px) == (1, 1)
    assert measurement.height_cm == pytest.approx(expected_height)
    assert measurement.area_cm2 == pytest.approx(expected_area)
    assert measurement.volume_cm3 == pytest.approx(expected_volume)
    assert measurement.weight_g == pytest.approx(expected_volume)


@pytest.mark.parametrize(
    ("depth_map", "food_mask"),
    [
        (np.ones((2, 2)), np.zeros((2, 2), dtype=bool)),
        (np.ones((2, 2)), np.ones((3, 3), dtype=bool)),
        (np.full((2, 2), np.nan), np.ones((2, 2), dtype=bool)),
    ],
)
def test_find_top_vertex_rejects_invalid_maps(
    depth_map: np.ndarray, food_mask: np.ndarray
) -> None:
    with pytest.raises(FoodMeasurementError):
        find_top_vertex(depth_map, food_mask)


@pytest.mark.parametrize(
    "density",
    [0.0, -1.0, math.inf, math.nan],
)
def test_estimate_rejects_invalid_density(density: float) -> None:
    mask = np.array([[True]])
    with pytest.raises(FoodMeasurementError, match="density"):
        estimate_food_measurement(
            np.array([[1.0]]),
            make_detection(mask, 0),
            make_calibration(theta=math.pi / 6),
            density,
        )


def test_estimate_rejects_near_top_down_view() -> None:
    mask = np.array([[True], [True]])
    with pytest.raises(FoodMeasurementError, match="top-down"):
        estimate_food_measurement(
            np.array([[1.0], [2.0]]),
            make_detection(mask, 1),
            make_calibration(theta=0.0),
            1.0,
        )


def test_estimate_rejects_vertex_below_base() -> None:
    mask = np.array([[True], [True]])
    with pytest.raises(FoodMeasurementError, match="below"):
        estimate_food_measurement(
            np.array([[2.0], [1.0]]),
            make_detection(mask, 0),
            make_calibration(),
            1.0,
        )
