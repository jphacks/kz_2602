"""Application entry-point helpers for the food estimation pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

from depth_anything import DepthAnythingV2SmallEstimator
from food_segmentation import FoodSegmenter
from model_manager import ensure_model_file, ensure_onnx_model_file
from pipeline import PipelineError, PipelineResult, run_pipeline

DEFAULT_DEPTH_MODEL_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small-ONNX/"
    "resolve/main/onnx/model.onnx?download=true"
)
DEFAULT_DEPTH_EXTERNAL_DATA_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small-ONNX/"
    "resolve/main/onnx/model.onnx_data?download=true"
)
SUPPORTED_FOOD_CLASSES = frozenset({"rice", "meat", "vegetable", "bread", "bowl"})
COCO_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
COCO_FOOD_CLASS_ALIASES = {
    "bowl": "bowl",
    "banana": "vegetable",
    "apple": "vegetable",
    "sandwich": "bread",
    "orange": "vegetable",
    "broccoli": "vegetable",
    "carrot": "vegetable",
    "hot dog": "meat",
    "pizza": "bread",
    "donut": "bread",
    "cake": "bread",
}


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

    segmenter_options: dict[str, Any] = {
        "input_size": (input_size[0], input_size[1]),
        "confidence_threshold": config.get("confidence_threshold", 0.25),
        "iou_threshold": config.get("iou_threshold", 0.45),
    }
    if config.get("class_aliases") is not None:
        segmenter_options["class_aliases"] = config["class_aliases"]
    return FoodSegmenter(
        model_path,
        class_names,
        **segmenter_options,
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


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run food measurement and heating estimation on one image."
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument(
        "--diameter-cm", required=True, type=float, help="Turntable diameter in cm"
    )
    parser.add_argument(
        "--yolo-model", required=True, help="Food-trained YOLO segmentation ONNX path"
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="YOLO class names in model index order",
    )
    parser.add_argument(
        "--coco",
        action="store_true",
        help="Use the standard COCO 80-class names and food class mapping",
    )
    parser.add_argument(
        "--yolo-model-url", help="URL used when --yolo-model does not exist"
    )
    parser.add_argument("--yolo-model-sha256")
    parser.add_argument(
        "--yolo-model-pt", help="Optional YOLO PT source for conversion"
    )
    parser.add_argument("--yolo-model-pt-url")
    parser.add_argument("--yolo-model-pt-sha256")
    parser.add_argument(
        "--depth-model",
        default="models/depth-anything-v2-small.onnx",
        help="Depth Anything ONNX path",
    )
    parser.add_argument("--depth-model-url")
    parser.add_argument("--depth-model-sha256")
    parser.add_argument("--depth-external-data")
    parser.add_argument("--depth-external-data-url")
    parser.add_argument("--depth-external-data-sha256")
    parser.add_argument(
        "--turntable-mask",
        help="Optional grayscale mask image; non-zero pixels define the turntable",
    )
    parser.add_argument("--output", help="Optional JSON output path; stdout by default")
    return parser


def _read_rgb_image(image_path: str) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _read_turntable_mask(mask_path: str, image_shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"could not read turntable mask: {mask_path}")
    if mask.shape != image_shape:
        raise ValueError(
            "turntable mask dimensions must match the input image "
            f"({image_shape[1]}x{image_shape[0]})"
        )
    return mask > 0


def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
    if args.coco:
        if args.class_names:
            raise ValueError("--coco cannot be combined with --class-names")
        class_names = list(COCO_CLASS_NAMES)
        class_aliases = COCO_FOOD_CLASS_ALIASES
    else:
        if not args.class_names:
            raise ValueError("--class-names is required unless --coco is used")
        class_names = list(args.class_names)
        class_aliases = None
        unsupported_classes = sorted(set(class_names) - SUPPORTED_FOOD_CLASSES)
        if unsupported_classes:
            raise ValueError(
                "class-names must use supported food classes only; "
                f"unsupported: {', '.join(unsupported_classes)}"
            )
        if len(set(class_names)) != len(class_names):
            raise ValueError("class-names must not contain duplicates")

    image_rgb = _read_rgb_image(args.image)
    turntable_mask = (
        _read_turntable_mask(args.turntable_mask, image_rgb.shape[:2])
        if args.turntable_mask
        else None
    )
    segmenter = create_food_segmenter(
        {
            "model_path": args.yolo_model,
            "class_names": class_names,
            "model_url": args.yolo_model_url,
            "model_sha256": args.yolo_model_sha256,
            "model_pt_path": args.yolo_model_pt,
            "model_pt_url": args.yolo_model_pt_url,
            "model_pt_sha256": args.yolo_model_pt_sha256,
            "class_aliases": class_aliases,
        }
    )
    depth_estimator = create_depth_estimator(
        {
            "model_path": args.depth_model,
            "model_url": args.depth_model_url or DEFAULT_DEPTH_MODEL_URL,
            "model_sha256": args.depth_model_sha256,
            "external_data_path": args.depth_external_data,
            "external_data_url": args.depth_external_data_url
            or DEFAULT_DEPTH_EXTERNAL_DATA_URL,
            "external_data_sha256": args.depth_external_data_sha256,
        }
    )
    result = run_pipeline(
        image_rgb,
        args.diameter_cm,
        segmenter,
        depth_estimator,
        turntable_mask=turntable_mask,
    )
    return pipeline_result_to_dict(result)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the end-to-end pipeline CLI."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    try:
        output = json.dumps(_run_cli(args), ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(output)
    except (OSError, PipelineError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
