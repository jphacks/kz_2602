from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from turntable_obstacle import YoloObstacleDetector


class FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakeModel:
    def __init__(self, masks: np.ndarray) -> None:
        self.masks = masks

    def __call__(self, image: np.ndarray, **kwargs: object) -> list[object]:
        return [SimpleNamespace(masks=SimpleNamespace(data=FakeTensor(self.masks)))]


def test_detector_combines_all_resized_masks(monkeypatch) -> None:
    model = FakeModel(
        np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
            ],
            dtype=np.float32,
        )
    )
    detector = object.__new__(YoloObstacleDetector)
    detector._model = model
    detector._confidence_threshold = 0.25
    detector._iou_threshold = 0.45

    mask = detector.predict(np.zeros((4, 4, 3), dtype=np.uint8))

    assert mask.dtype == bool
    assert mask.shape == (4, 4)
    assert mask[0, 0]
    assert mask[-1, -1]
    assert mask.sum() == 8
