from __future__ import annotations

import numpy as np

from turntable_obstacle import YoloObstacleDetector
from yolo_segmentation import YoloDetection, YoloSegmentationResult


class FakeEngine:
    def __init__(self, result: YoloSegmentationResult) -> None:
        self.result = result
        self.image: np.ndarray | None = None

    def predict(self, image: np.ndarray) -> YoloSegmentationResult:
        self.image = image
        return self.result


def test_detector_combines_all_obstacle_masks() -> None:
    result = YoloSegmentationResult(
        detections=(
            YoloDetection(
                mask=np.array([[True, True, False, False]] * 2 + [[False] * 4] * 2),
                class_name="obstacle",
                confidence=0.9,
                box=np.array([0, 0, 2, 2], dtype=np.float32),
            ),
            YoloDetection(
                mask=np.array([[False] * 4] * 2 + [[False, False, True, True]] * 2),
                class_name="obstacle",
                confidence=0.8,
                box=np.array([2, 2, 4, 4], dtype=np.float32),
            ),
        ),
        image_shape=(4, 4),
    )
    engine = FakeEngine(result)
    detector = YoloObstacleDetector(
        None,
        ["obstacle"],
        engine=engine,
    )

    mask = detector.predict(np.zeros((4, 4, 3), dtype=np.uint8))

    assert mask.dtype == bool
    assert mask.shape == (4, 4)
    assert mask[0, 0]
    assert mask[-1, -1]
    assert mask.sum() == 8


def test_detector_ignores_non_obstacle_classes() -> None:
    result = YoloSegmentationResult(
        detections=(
            YoloDetection(
                mask=np.ones((3, 3), dtype=bool),
                class_name="food",
                confidence=0.9,
                box=np.array([0, 0, 3, 3], dtype=np.float32),
            ),
        ),
        image_shape=(3, 3),
    )
    detector = YoloObstacleDetector(
        None,
        ["food", "obstacle"],
        obstacle_class_names=["obstacle"],
        engine=FakeEngine(result),
    )

    mask = detector.predict(np.zeros((3, 3, 3), dtype=np.uint8))

    assert not mask.any()
