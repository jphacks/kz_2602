from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import server
import uvicorn

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


def test_cli_reads_rgb_passes_mask_and_writes_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    output_path = tmp_path / "result.json"
    image_bgr = np.zeros((4, 5, 3), dtype=np.uint8)
    image_bgr[0, 0] = [10, 20, 30]
    assert cv2.imwrite(str(image_path), image_bgr)
    assert cv2.imwrite(str(mask_path), np.array([[255] * 5] * 4, dtype=np.uint8))

    captured: dict[str, object] = {}

    def fake_segmenter(config: dict[str, object]) -> object:
        captured["segmenter_config"] = config
        return object()

    def fake_depth(config: dict[str, object]) -> object:
        captured["depth_config"] = config
        return object()

    def fake_obstacle(config: dict[str, object]) -> object:
        captured["obstacle_config"] = config
        return object()

    def fake_run_pipeline(
        image_rgb: np.ndarray,
        diameter_cm: float,
        segmenter: object,
        depth_estimator: object,
        *,
        obstacle_detector: object,
    ) -> object:
        captured["image_rgb"] = image_rgb
        captured["diameter_cm"] = diameter_cm
        captured["segmenter"] = segmenter
        captured["depth_estimator"] = depth_estimator
        captured["obstacle_detector"] = obstacle_detector
        return object()

    monkeypatch.setattr(main, "create_food_segmenter", fake_segmenter)
    monkeypatch.setattr(main, "create_depth_estimator", fake_depth)
    monkeypatch.setattr(main, "create_obstacle_detector", fake_obstacle)
    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(main, "pipeline_result_to_dict", lambda result: {"ok": True})

    exit_code = main.main(
        [
            "--image",
            str(image_path),
            "--diameter-cm",
            "20",
            "--yolo-model",
            "food.onnx",
            "--class-names",
            "rice",
            "--depth-model",
            "depth.onnx",
            "--obstacle-model",
            "obstacle.onnx",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert captured["image_rgb"][0, 0].tolist() == [30, 20, 10]
    assert captured["diameter_cm"] == 20.0
    assert captured["obstacle_detector"] is not None
    assert captured["segmenter_config"]["class_names"] == ["rice"]
    assert captured["depth_config"]["model_path"] == "depth.onnx"
    assert captured["obstacle_config"]["model_path"] == "obstacle.onnx"
    assert output_path.read_text(encoding="utf-8") == '{\n  "ok": true\n}\n'
    assert capsys.readouterr().out == '{\n  "ok": true\n}\n'


def test_cli_rejects_unsupported_model_classes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main.main(
        [
            "--image",
            "input.png",
            "--diameter-cm",
            "20",
            "--yolo-model",
            "yolov8n-seg.onnx",
            "--class-names",
            "person",
        ]
    )

    assert exit_code == 2
    assert "unsupported: person" in capsys.readouterr().err


def test_cli_can_start_http_server_without_an_input_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(main, "create_food_segmenter", lambda config: config)
    monkeypatch.setattr(main, "create_depth_estimator", lambda config: config)
    monkeypatch.setattr(main, "create_obstacle_detector", lambda config: config)

    def fake_create_app(factory: object) -> object:
        captured["components"] = factory()
        return object()

    monkeypatch.setattr(server, "create_app", fake_create_app)
    monkeypatch.setattr(
        uvicorn,
        "run",
        lambda app, **kwargs: captured.update({"app": app, "kwargs": kwargs}),
    )

    exit_code = main.main(
        [
            "--serve",
            "--yolo-model",
            "food.onnx",
            "--class-names",
            "rice",
            "--depth-model",
            "depth.onnx",
            "--obstacle-model",
            "obstacle.onnx",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
        ]
    )

    assert exit_code == 0
    assert captured["kwargs"] == {
        "host": "0.0.0.0",
        "port": 9000,
        "workers": 1,
    }
    components = captured["components"]
    assert components.food_segmenter["model_path"] == "food.onnx"
    assert components.food_segmenter["class_names"] == ["rice"]
    assert components.depth_estimator["model_path"] == "depth.onnx"
    assert components.obstacle_detector["model_path"] == "obstacle.onnx"
