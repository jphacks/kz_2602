"""Application entry-point helpers for the food estimation pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from depth_anything import DepthAnythingV2SmallEstimator
from food_segmentation import FoodSegmenter
from model_manager import ensure_model_file, ensure_onnx_model_file
from pipeline import PipelineResult

DEFAULT_DEPTH_MODEL_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small-ONNX/"
    "resolve/main/onnx/model.onnx?download=true"
)
DEFAULT_DEPTH_EXTERNAL_DATA_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small-ONNX/"
    "resolve/main/onnx/model.onnx_data?download=true"
)


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

    model_path = ensure_model_file(
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


def create_depth_estimator(config: Mapping[str, Any]) -> DepthAnythingV2SmallEstimator:
    """Build and cache a Depth Anything V2 Small ONNX adapter."""
    model_path = config.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("model_path must be a non-empty string")

    input_size = config.get("input_size", [518, 518])
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

    default_external_path = str(Path(model_path).with_name("model.onnx_data"))
    cached_model_path = ensure_onnx_model_file(
        model_path,
        config.get("model_url", DEFAULT_DEPTH_MODEL_URL),
        config.get("model_sha256"),
        external_data_path=config.get("external_data_path", default_external_path),
        external_data_url=config.get(
            "external_data_url", DEFAULT_DEPTH_EXTERNAL_DATA_URL
        ),
        external_data_sha256=config.get("external_data_sha256"),
    )

    return DepthAnythingV2SmallEstimator(
        cached_model_path,
        input_size=(input_size[0], input_size[1]),
        output_is_inverse_depth=config.get("output_is_inverse_depth", True),
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
