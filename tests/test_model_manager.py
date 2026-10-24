from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import model_manager


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


def test_ensure_model_file_downloads_missing_onnx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "downloaded.onnx"
    model_bytes = b"downloaded model"
    monkeypatch.setattr(
        model_manager, "urlopen", lambda url, timeout: FakeResponse(model_bytes)
    )

    result = model_manager.ensure_model_file(
        str(model_path),
        "https://example.com/model.onnx",
        hashlib.sha256(model_bytes).hexdigest(),
        None,
        None,
        None,
        (640, 640),
    )

    assert result == str(model_path)
    assert model_path.read_bytes() == model_bytes


def test_ensure_model_file_rejects_download_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "invalid.onnx"
    monkeypatch.setattr(
        model_manager, "urlopen", lambda url, timeout: FakeResponse(b"model")
    )

    with pytest.raises(ValueError, match="could not download YOLO model"):
        model_manager.ensure_model_file(
            str(model_path),
            "https://example.com/model.onnx",
            "0" * 64,
            None,
            None,
            None,
            (640, 640),
        )

    assert not model_path.exists()


def test_ensure_model_file_converts_pt_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pt_path = tmp_path / "food.pt"
    onnx_path = tmp_path / "food.onnx"
    pt_path.write_bytes(b"pt model")
    export_calls: list[tuple[Path, Path, tuple[int, int]]] = []

    def fake_export(source: Path, target: Path, input_size: tuple[int, int]) -> None:
        export_calls.append((source, target, input_size))
        target.write_bytes(b"onnx model")

    monkeypatch.setattr(model_manager, "_export_pt_to_onnx", fake_export)

    result = model_manager.ensure_model_file(
        str(onnx_path),
        None,
        None,
        str(pt_path),
        None,
        None,
        (320, 416),
    )

    assert result == str(onnx_path)
    assert export_calls == [(pt_path, onnx_path, (320, 416))]
    assert onnx_path.read_bytes() == b"onnx model"


def test_ensure_onnx_model_file_downloads_external_data_and_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "depth.onnx"
    external_path = tmp_path / "model.onnx_data"
    model_bytes = b"onnx model"
    external_bytes = b"external weights"
    responses = iter([FakeResponse(model_bytes), FakeResponse(external_bytes)])
    monkeypatch.setattr(model_manager, "urlopen", lambda url, timeout: next(responses))

    result = model_manager.ensure_onnx_model_file(
        str(model_path),
        "https://example.com/model.onnx",
        hashlib.sha256(model_bytes).hexdigest(),
        external_data_path=str(external_path),
        external_data_url="https://example.com/model.onnx_data",
        external_data_sha256=hashlib.sha256(external_bytes).hexdigest(),
    )

    assert result == str(model_path)
    assert model_path.read_bytes() == model_bytes
    assert external_path.read_bytes() == external_bytes


def test_ensure_onnx_model_file_rejects_external_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "depth.onnx"
    external_path = tmp_path / "model.onnx_data"
    monkeypatch.setattr(
        model_manager,
        "urlopen",
        lambda url, timeout: FakeResponse(b"invalid external weights"),
    )

    with pytest.raises(ValueError, match="could not download ONNX external data model"):
        model_manager.ensure_onnx_model_file(
            str(model_path),
            "https://example.com/model.onnx",
            hashlib.sha256(b"invalid external weights").hexdigest(),
            external_data_path=str(external_path),
            external_data_url="https://example.com/model.onnx_data",
            external_data_sha256="0" * 64,
        )

    assert not external_path.exists()
