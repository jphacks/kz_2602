from __future__ import annotations

import numpy as np
import pytest

from food_segmentation import FoodSegmentationError, FoodSegmenter


class FakeInput:
    name = "images"


class FakeSession:
    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = outputs
        self.input = None

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]

    def run(self, _: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.input = inputs["images"]
        return self.outputs


class FailingSession(FakeSession):
    def run(self, _: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        raise RuntimeError("inference failed")


def segmentation_outputs(
    *, class_score: float = 0.9, second_detection: bool = False
) -> list[np.ndarray]:
    predictions = np.zeros((1, 6, 2 if second_detection else 1), dtype=np.float32)
    predictions[0, :4, 0] = [10.0, 10.0, 10.0, 10.0]
    predictions[0, 4, 0] = class_score
    if second_detection:
        predictions[0, :4, 1] = [4.0, 4.0, 4.0, 4.0]
        predictions[0, 4, 1] = class_score - 0.1

    prototype = np.full((1, 1, 4, 4), -10.0, dtype=np.float32)
    prototype[0, 0, 1:3, 1:3] = 10.0
    return [predictions, prototype]


def make_segmenter(
    outputs: list[np.ndarray], **kwargs: object
) -> tuple[FoodSegmenter, FakeSession]:
    session = FakeSession(outputs)
    segmenter = FoodSegmenter(
        None,
        ["rice"],
        input_size=(20, 20),
        session=session,
        **kwargs,
    )
    return segmenter, session


def test_predict_returns_original_size_mask_and_base_coordinate() -> None:
    segmenter, session = make_segmenter(segmentation_outputs())
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    detection = segmenter.predict(image)

    assert detection.class_id == "rice"
    assert detection.confidence == pytest.approx(0.9)
    assert detection.mask.shape == image.shape[:2]
    assert detection.mask.dtype == np.bool_
    assert detection.y_base_px == int(np.flatnonzero(detection.mask.any(axis=1)).max())
    assert session.input is not None
    assert session.input.shape == (1, 3, 20, 20)
    assert session.input.dtype == np.float32


def test_predict_maps_configured_class_alias() -> None:
    segmenter, _ = make_segmenter(
        segmentation_outputs(), class_aliases={"rice": "bread"}
    )

    detection = segmenter.predict(np.zeros((10, 20, 3), dtype=np.uint8))

    assert detection.class_id == "bread"


def test_predict_rejects_multiple_food_detections() -> None:
    segmenter, _ = make_segmenter(segmentation_outputs(second_detection=True))

    with pytest.raises(FoodSegmentationError, match="exactly one"):
        segmenter.predict(np.zeros((20, 20, 3), dtype=np.uint8))


def test_predict_returns_no_detection_below_threshold() -> None:
    segmenter, _ = make_segmenter(
        segmentation_outputs(class_score=0.2), confidence_threshold=0.5
    )

    with pytest.raises(FoodSegmentationError, match="exactly one"):
        segmenter.predict(np.zeros((20, 20, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    ("prediction_shape", "prototype_shape"),
    [((1, 7, 3), (1, 2, 4, 4)), ((1, 3, 7), (1, 4, 4, 2))],
)
def test_parse_outputs_normalizes_common_yolov8_axes(
    prediction_shape: tuple[int, ...], prototype_shape: tuple[int, ...]
) -> None:
    segmenter, _ = make_segmenter(
        [
            np.zeros(prediction_shape, dtype=np.float32),
            np.zeros(prototype_shape, dtype=np.float32),
        ]
    )

    predictions, prototypes = segmenter._parse_outputs(
        [
            np.zeros(prediction_shape, dtype=np.float32),
            np.zeros(prototype_shape, dtype=np.float32),
        ]
    )

    assert predictions.shape == (3, 7)
    assert prototypes.shape == (4, 4, 2)


def test_predict_wraps_inference_failures() -> None:
    segmenter = FoodSegmenter(
        None,
        ["rice"],
        input_size=(20, 20),
        session=FailingSession([]),
    )

    with pytest.raises(FoodSegmentationError, match="inference failed"):
        segmenter.predict(np.zeros((20, 20, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "image",
    [np.empty((0, 10, 3), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8)],
)
def test_predict_rejects_invalid_images(image: np.ndarray) -> None:
    segmenter, _ = make_segmenter(segmentation_outputs())

    with pytest.raises(FoodSegmentationError, match="BGR image"):
        segmenter.predict(image)
