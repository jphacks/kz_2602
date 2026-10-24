from __future__ import annotations

import hashlib
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


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _: int) -> bytes:
        content, self.content = self.content, b""
        return content


def test_create_food_segmenter_passes_json_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "FoodSegmenter", FakeSegmenter)
    model_path = tmp_path / "food.onnx"
    model_path.write_bytes(b"existing model")

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


def test_create_food_segmenter_downloads_missing_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main, "FoodSegmenter", FakeSegmenter)
    model_path = tmp_path / "downloaded.onnx"
    model_bytes = b"downloaded model"
    monkeypatch.setattr(main, "urlopen", lambda url, timeout: FakeResponse(model_bytes))

    segmenter = main.create_food_segmenter(
        {
            "model_path": str(model_path),
            "model_url": "https://example.com/model.onnx",
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "class_names": ["rice"],
        }
    )

    assert model_path.read_bytes() == model_bytes
    assert segmenter.model_path == str(model_path)


def test_create_food_segmenter_rejects_download_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main, "FoodSegmenter", FakeSegmenter)
    model_path = tmp_path / "invalid.onnx"
    monkeypatch.setattr(main, "urlopen", lambda url, timeout: FakeResponse(b"model"))

    with pytest.raises(ValueError, match="could not download YOLO model"):
        main.create_food_segmenter(
            {
                "model_path": str(model_path),
                "model_url": "https://example.com/model.onnx",
                "model_sha256": "0" * 64,
                "class_names": ["rice"],
            }
        )

    assert not model_path.exists()


def test_create_food_segmenter_converts_pt_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main, "FoodSegmenter", FakeSegmenter)
    pt_path = tmp_path / "food.pt"
    onnx_path = tmp_path / "food.onnx"
    pt_path.write_bytes(b"pt model")
    export_calls: list[tuple[Path, Path, tuple[int, int]]] = []

    def fake_export(source: Path, target: Path, input_size: tuple[int, int]) -> None:
        export_calls.append((source, target, input_size))
        target.write_bytes(b"onnx model")

    monkeypatch.setattr(main, "_export_pt_to_onnx", fake_export)

    segmenter = main.create_food_segmenter(
        {
            "model_path": str(onnx_path),
            "model_pt_path": str(pt_path),
            "class_names": ["rice"],
            "input_size": [320, 416],
        }
    )

    assert export_calls == [(pt_path, onnx_path, (320, 416))]
    assert segmenter.model_path == str(onnx_path)
    assert onnx_path.read_bytes() == b"onnx model"


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
