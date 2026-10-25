from __future__ import annotations

import numpy as np
import pytest

from depth_anything import DepthAnythingV2SmallEstimator, DepthEstimationError


class FakeInput:
    name = "pixel_values"


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.input: np.ndarray | None = None

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]

    def run(self, _: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.input = inputs["pixel_values"]
        return [self.output]


class FailingSession(FakeSession):
    def run(self, _: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        raise RuntimeError("inference failed")


def make_estimator(
    output: np.ndarray, **kwargs: object
) -> tuple[DepthAnythingV2SmallEstimator, FakeSession]:
    session = FakeSession(output)
    estimator = DepthAnythingV2SmallEstimator(
        None,
        input_size=(4, 6),
        session=session,
        **kwargs,
    )
    return estimator, session


def test_predict_returns_original_size_and_nchw_normalized_input() -> None:
    output = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    estimator, session = make_estimator(output)
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    depth_map = estimator.predict(image)

    assert depth_map.shape == image.shape[:2]
    assert depth_map.dtype == np.float32
    assert np.all(np.isfinite(depth_map))
    assert session.input is not None
    assert session.input.shape == (1, 3, 4, 6)
    assert session.input.dtype == np.float32
    assert session.input[0, 0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229)


def test_predict_converts_inverse_depth_to_small_is_near() -> None:
    output = np.array([[[[1.0, 4.0], [2.0, 3.0]]]], dtype=np.float32)
    estimator, _ = make_estimator(output)

    depth_map = estimator.predict(np.zeros((2, 2, 3), dtype=np.uint8))

    assert depth_map[0, 1] == pytest.approx(0.0)
    assert depth_map[0, 0] == pytest.approx(3.0)


def test_predict_can_keep_normal_depth_direction() -> None:
    output = np.array([[[[1.0, 4.0], [2.0, 3.0]]]], dtype=np.float32)
    estimator, _ = make_estimator(output, output_is_inverse_depth=False)

    depth_map = estimator.predict(np.zeros((2, 2, 3), dtype=np.uint8))

    assert depth_map[0, 0] == pytest.approx(1.0)
    assert depth_map[0, 1] == pytest.approx(4.0)


def test_predict_wraps_inference_failures() -> None:
    estimator = DepthAnythingV2SmallEstimator(
        None,
        input_size=(4, 6),
        session=FailingSession(np.zeros((1, 1, 2, 2), dtype=np.float32)),
    )

    with pytest.raises(DepthEstimationError, match="inference failed"):
        estimator.predict(np.zeros((8, 10, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "image",
    [
        np.empty((0, 10, 3), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
    ],
)
def test_predict_rejects_invalid_images(image: np.ndarray) -> None:
    estimator, _ = make_estimator(np.zeros((1, 1, 2, 2), dtype=np.float32))

    with pytest.raises(DepthEstimationError, match="image_rgb"):
        estimator.predict(image)


@pytest.mark.parametrize(
    "output",
    [
        [np.zeros((1, 2, 2), dtype=np.float32), np.zeros((1, 2, 2))],
        [np.zeros((1, 2, 2, 2), dtype=np.float32)],
        [np.full((1, 1, 2, 2), np.nan, dtype=np.float32)],
    ],
)
def test_parse_output_rejects_invalid_outputs(output: list[np.ndarray]) -> None:
    estimator, _ = make_estimator(np.zeros((1, 1, 2, 2), dtype=np.float32))

    with pytest.raises(DepthEstimationError):
        estimator._parse_output(output)
