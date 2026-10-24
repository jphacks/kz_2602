from __future__ import annotations

import cv2
import numpy as np
import pytest

from food_segmentation import FoodDetection
from pipeline import PipelineError, run_pipeline


class FakeSegmenter:
    def __init__(self, detection: FoodDetection) -> None:
        self.detection = detection
        self.image: np.ndarray | None = None

    def predict(self, image: np.ndarray) -> FoodDetection:
        self.image = image
        return self.detection


class FakeDepthEstimator:
    def __init__(self, depth_map: np.ndarray) -> None:
        self.depth_map = depth_map
        self.image: np.ndarray | None = None

    def predict(self, image: np.ndarray) -> np.ndarray:
        self.image = image
        return self.depth_map


class FakeObstacleDetector:
    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        self.image: np.ndarray | None = None

    def predict(self, image: np.ndarray) -> np.ndarray:
        self.image = image
        return self.mask


def make_turntable_image() -> np.ndarray:
    image = np.zeros((120, 120), dtype=np.uint8)
    cv2.ellipse(image, (60, 60), (40, 20), 0.0, 0.0, 360.0, 255, -1)
    return image


def make_detection() -> FoodDetection:
    mask = np.zeros((120, 120), dtype=bool)
    mask[30:61, 50:71] = True
    return FoodDetection(mask=mask, class_id="rice", confidence=0.9, y_base_px=60)


def test_run_pipeline_connects_all_stages_and_keeps_rgb_contract() -> None:
    image_rgb = make_turntable_image()
    image_rgb = np.repeat(image_rgb[:, :, None], 3, axis=2)
    image_rgb[0, 0] = [10, 20, 30]
    segmenter = FakeSegmenter(make_detection())
    depth = np.full((120, 120), 10.0)
    depth[30, 60] = 1.0
    estimator = FakeDepthEstimator(depth)

    result = run_pipeline(
        image_rgb,
        20.0,
        segmenter,
        estimator,
        obstacle_detector=FakeObstacleDetector(np.zeros((120, 120), dtype=bool)),
    )

    assert result.detection.class_id == "rice"
    assert result.material.density_g_per_cm3 == pytest.approx(1.05)
    assert result.measurement.top_x_px == 60
    assert result.measurement.top_y_px == 30
    assert result.measurement.weight_g > 0
    assert result.heating.heating_time_s > 0
    assert segmenter.image is not None
    assert tuple(segmenter.image[0, 0]) == (30, 20, 10)
    assert estimator.image is image_rgb


def test_run_pipeline_can_detect_turntable_from_image_without_mask() -> None:
    image_rgb = make_turntable_image()
    image_rgb = np.repeat(image_rgb[:, :, None], 3, axis=2)
    detection = make_detection()
    segmenter = FakeSegmenter(detection)
    depth = np.full((120, 120), 10.0)
    depth[30, 60] = 1.0

    result = run_pipeline(
        image_rgb,
        20.0,
        segmenter,
        FakeDepthEstimator(depth),
    )

    assert result.calibration.diameter_cm == pytest.approx(20.0)


@pytest.mark.parametrize(
    "invalid_image",
    [np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.float32)],
)
def test_run_pipeline_rejects_invalid_rgb_images(invalid_image: np.ndarray) -> None:
    with pytest.raises(PipelineError, match="image_rgb"):
        run_pipeline(
            invalid_image,
            20.0,
            FakeSegmenter(make_detection()),
            FakeDepthEstimator(np.zeros((10, 10))),
        )


def test_run_pipeline_rejects_depth_shape_mismatch() -> None:
    with pytest.raises(PipelineError, match="depth_map"):
        run_pipeline(
            np.repeat(make_turntable_image()[:, :, None], 3, axis=2),
            20.0,
            FakeSegmenter(make_detection()),
            FakeDepthEstimator(np.zeros((10, 10))),
            obstacle_detector=FakeObstacleDetector(np.zeros((120, 120), dtype=bool)),
        )
