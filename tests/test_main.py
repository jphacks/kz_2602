from __future__ import annotations

from pathlib import Path

import pytest

import main


class FakeSegmenter:
    def __init__(
        self, model_path: str, class_names: list[str], **kwargs: object
    ) -> None:
        self.model_path = model_path
        self.class_names = class_names
        self.options = kwargs


class FakeDepthEstimator:
    def __init__(self, model_path: str, **kwargs: object) -> None:
        self.model_path = model_path
        self.options = kwargs


def test_create_food_segmenter_passes_json_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "FoodSegmenter", FakeSegmenter)
    monkeypatch.setattr(main, "ensure_model_file", lambda model_path, *args: model_path)
    model_path = tmp_path / "food.onnx"

    segmenter = main.create_food_segmenter(
        {
            "model_path": str(model_path),
            "class_names": ["rice", "meat"],
            "input_size": [320, 416],
            "confidence_threshold": 0.4,
            "iou_threshold": 0.5,
        }
    )

    assert segmenter.model_path == str(model_path)
    assert segmenter.class_names == ["rice", "meat"]
    assert segmenter.options == {
        "input_size": (320, 416),
        "confidence_threshold": 0.4,
        "iou_threshold": 0.5,
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"class_names": ["rice"]}, "model_path"),
        ({"model_path": "food.onnx", "class_names": []}, "class_names"),
        (
            {"model_path": "food.onnx", "class_names": ["rice"], "input_size": [640]},
            "input_size",
        ),
    ],
)
def test_create_food_segmenter_rejects_invalid_settings(
    config: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        main.create_food_segmenter(config)


def test_create_depth_estimator_passes_json_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "DepthAnythingV2SmallEstimator", FakeDepthEstimator)
    monkeypatch.setattr(
        main, "ensure_onnx_model_file", lambda model_path, *args, **kwargs: model_path
    )
    model_path = tmp_path / "depth.onnx"

    estimator = main.create_depth_estimator(
        {
            "model_path": str(model_path),
            "input_size": [384, 384],
            "output_is_inverse_depth": False,
        }
    )

    assert estimator.model_path == str(model_path)
    assert estimator.options == {
        "input_size": (384, 384),
        "output_is_inverse_depth": False,
    }
