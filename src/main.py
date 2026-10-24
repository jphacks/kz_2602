"""Application entry-point helpers for the food estimation pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from importlib import import_module
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from urllib.request import urlopen

from food_segmentation import FoodSegmenter
from pipeline import PipelineResult


def _verify_model_hash(model_path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    digest = hashlib.sha256()
    with model_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch for model: {model_path}")


def _ensure_model_file(
    model_path: str,
    model_url: Any,
    model_sha256: Any,
    model_pt_path: Any,
    model_pt_url: Any,
    model_pt_sha256: Any,
    input_size: tuple[int, int],
) -> str:
    path = Path(model_path)
    if model_sha256 is not None:
        if (
            not isinstance(model_sha256, str)
            or len(model_sha256) != 64
            or any(
                character not in "0123456789abcdefABCDEF" for character in model_sha256
            )
        ):
            raise ValueError("model_sha256 must be a 64-character hexadecimal string")

    if path.exists():
        _verify_model_hash(path, model_sha256)
        return str(path)

    if model_pt_path is not None or model_pt_url is not None:
        if model_pt_path is None:
            model_pt_path = str(path.with_suffix(".pt"))
        if not isinstance(model_pt_path, str) or not model_pt_path:
            raise ValueError("model_pt_path must be a non-empty string")
        pt_path = Path(model_pt_path)
        if not pt_path.exists():
            if not isinstance(model_pt_url, str) or not model_pt_url:
                raise ValueError(
                    "model_pt_url is required when model_pt_path does not exist"
                )
            _download_model_file(model_pt_url, pt_path, model_pt_sha256)
        else:
            _verify_model_hash(pt_path, model_pt_sha256)
        _export_pt_to_onnx(pt_path, path, input_size)
        _verify_model_hash(path, model_sha256)
        return str(path)

    if not isinstance(model_url, str) or not model_url:
        raise ValueError(
            "model_url or model_pt_path is required when model_path does not exist"
        )

    _download_model_file(model_url, path, model_sha256)
    return str(path)


def _download_model_file(url: str, path: Path, expected_sha256: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".download", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as output_file:
            with urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output_file.write(chunk)
        _verify_model_hash(temporary_path, expected_sha256)
        os.replace(temporary_path, path)
    except (OSError, ValueError) as error:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(f"could not download YOLO model: {url}") from error


def _export_pt_to_onnx(
    pt_path: Path, onnx_path: Path, input_size: tuple[int, int]
) -> None:
    try:
        YOLO = import_module("ultralytics").YOLO
    except ImportError as error:
        raise ValueError(
            "PT-to-ONNX conversion requires ultralytics; install it with "
            "'uv add ultralytics'"
        ) from error

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        exported_path = YOLO(str(pt_path)).export(
            format="onnx",
            imgsz=list(input_size),
            nms=False,
            device="cpu",
        )
        source_path = (
            Path(exported_path) if exported_path else pt_path.with_suffix(".onnx")
        )
        if not source_path.exists():
            raise FileNotFoundError(f"exported ONNX file was not found: {source_path}")
        if source_path.resolve() != onnx_path.resolve():
            temporary_path = onnx_path.with_suffix(f"{onnx_path.suffix}.tmp")
            shutil.copyfile(source_path, temporary_path)
            os.replace(temporary_path, onnx_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"could not convert YOLO PT model: {pt_path}") from error


def create_food_segmenter(config: Mapping[str, Any]) -> FoodSegmenter:
    """Build a YOLOv8 segmentation adapter from JSON-compatible settings."""
    try:
        model_path = config["model_path"]
        class_names = config["class_names"]
    except KeyError as error:
        raise ValueError(f"missing segmenter setting: {error.args[0]}") from error

    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_path must be a non-empty string")
    if (
        not isinstance(class_names, Sequence)
        or isinstance(class_names, (str, bytes))
        or len(class_names) == 0
        or not all(isinstance(name, str) and name for name in class_names)
    ):
        raise ValueError("class_names must be a sequence of non-empty strings")

    input_size = config.get("input_size", [640, 640])
    if (
        not isinstance(input_size, Sequence)
        or isinstance(input_size, (str, bytes))
        or len(input_size) != 2
        or not all(
            isinstance(size, int) and not isinstance(size, bool) and size > 0
            for size in input_size
        )
    ):
        raise ValueError("input_size must contain two positive integers")

    model_path = _ensure_model_file(
        model_path,
        config.get("model_url"),
        config.get("model_sha256"),
        config.get("model_pt_path"),
        config.get("model_pt_url"),
        config.get("model_pt_sha256"),
        (input_size[0], input_size[1]),
    )

    return FoodSegmenter(
        model_path,
        class_names,
        input_size=(input_size[0], input_size[1]),
        confidence_threshold=config.get("confidence_threshold", 0.25),
        iou_threshold=config.get("iou_threshold", 0.45),
    )


def pipeline_result_to_dict(result: PipelineResult) -> dict[str, Any]:
    """Convert a pipeline result into JSON-serializable builtin values."""
    return {
        "calibration": {
            "center_px": list(result.calibration.ellipse.center_px),
            "major_axis_px": result.calibration.ellipse.major_axis_px,
            "minor_axis_px": result.calibration.ellipse.minor_axis_px,
            "angle_deg": result.calibration.ellipse.angle_deg,
            "diameter_cm": result.calibration.diameter_cm,
            "alpha_cm_per_px": result.calibration.alpha_cm_per_px,
            "theta_rad": result.calibration.theta_rad,
        },
        "detection": {
            "class_id": result.detection.class_id,
            "confidence": result.detection.confidence,
            "y_base_px": result.detection.y_base_px,
        },
        "material": {
            "class_id": result.material.class_id,
            "density_g_per_cm3": result.material.density_g_per_cm3,
            "epsilon_prime": result.material.epsilon_prime,
            "epsilon_double_prime": result.material.epsilon_double_prime,
            "tan_delta": result.material.tan_delta,
            "specific_heat_j_per_g_k": result.material.specific_heat_j_per_g_k,
        },
        "measurement": {
            "top_x_px": result.measurement.top_x_px,
            "top_y_px": result.measurement.top_y_px,
            "height_cm": result.measurement.height_cm,
            "area_cm2": result.measurement.area_cm2,
            "volume_cm3": result.measurement.volume_cm3,
            "weight_g": result.measurement.weight_g,
        },
        "heating": {
            "class_id": result.heating.class_id,
            "penetration_depth_cm": result.heating.penetration_depth_cm,
            "needs_turning_or_stirring": result.heating.needs_turning_or_stirring,
            "required_energy_j": result.heating.required_energy_j,
            "heating_time_s": result.heating.heating_time_s,
        },
    }
