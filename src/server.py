"""HTTP server for the food estimation pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
import logging
import math
from typing import Any

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import numpy as np

from food_heating import DEFAULT_TARGET_TEMPERATURE_C
from pipeline import PipelineError, run_pipeline

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
DEFAULT_INITIAL_TEMPERATURE_C = 20.0

logger = logging.getLogger(__name__)


class ServerComponents:
    """Models required by one application process."""

    def __init__(
        self,
        food_segmenter: Any,
        depth_estimator: Any,
        obstacle_detector: Any,
    ) -> None:
        self.food_segmenter = food_segmenter
        self.depth_estimator = depth_estimator
        self.obstacle_detector = obstacle_detector


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _decode_image(image_bytes: bytes) -> np.ndarray | JSONResponse:
    if not image_bytes:
        return _error_response(400, "invalid_image", "image must not be empty")
    image_bgr = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image_bgr is None:
        return _error_response(400, "invalid_image", "image could not be decoded")
    height, width = image_bgr.shape[:2]
    if max(height, width) > MAX_IMAGE_DIMENSION:
        return _error_response(
            413,
            "image_too_large",
            f"image dimensions must not exceed {MAX_IMAGE_DIMENSION}px",
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _parse_diameter(value: str | None) -> float | JSONResponse:
    if value is None or not value.strip():
        return _error_response(400, "invalid_diameter", "diameter_cm is required")
    try:
        diameter_cm = float(value)
    except ValueError:
        return _error_response(400, "invalid_diameter", "diameter_cm must be a number")
    if not math.isfinite(diameter_cm) or diameter_cm <= 0:
        return _error_response(
            400, "invalid_diameter", "diameter_cm must be a positive finite number"
        )
    return diameter_cm


def _parse_initial_temperature(value: str | None) -> float | JSONResponse:
    if value is None or not value.strip():
        return DEFAULT_INITIAL_TEMPERATURE_C
    try:
        initial_temperature_c = float(value)
    except ValueError:
        return _error_response(
            400,
            "invalid_initial_temperature",
            "initial_temperature_c must be a number",
        )
    if not math.isfinite(initial_temperature_c):
        return _error_response(
            400,
            "invalid_initial_temperature",
            "initial_temperature_c must be finite",
        )
    if initial_temperature_c >= DEFAULT_TARGET_TEMPERATURE_C:
        return _error_response(
            400,
            "invalid_initial_temperature",
            f"initial_temperature_c must be below {DEFAULT_TARGET_TEMPERATURE_C:g} C",
        )
    return initial_temperature_c


def create_app(
    component_factory: Callable[[], ServerComponents],
    *,
    pipeline_runner: Callable[..., Any] = run_pipeline,
    result_serializer: Callable[[Any], Mapping[str, Any]] | None = None,
) -> FastAPI:
    """Create an application and initialize models once during startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.components = component_factory()
            app.state.ready = True
            app.state.initialization_error = None
        except Exception as error:
            logger.exception("model initialization failed")
            app.state.components = None
            app.state.ready = False
            app.state.initialization_error = error
        yield

    app = FastAPI(
        title="Food Estimation API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        if not getattr(app.state, "ready", False):
            return _error_response(503, "not_ready", "models are not ready")
        return JSONResponse(content={"status": "ok"})

    @app.post("/v1/estimate")
    def estimate(
        image: UploadFile | None = File(default=None),
        diameter_cm: str | None = Form(default=None),
        initial_temperature_c: str | None = Form(default=None),
    ) -> JSONResponse:
        if image is None:
            return _error_response(400, "missing_image", "image is required")
        if not getattr(app.state, "ready", False):
            return _error_response(503, "not_ready", "models are not ready")

        diameter = _parse_diameter(diameter_cm)
        if isinstance(diameter, JSONResponse):
            return diameter
        initial_temperature = _parse_initial_temperature(initial_temperature_c)
        if isinstance(initial_temperature, JSONResponse):
            return initial_temperature

        image_bytes = image.file.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            return _error_response(
                413,
                "image_too_large",
                f"image file must not exceed {MAX_IMAGE_BYTES} bytes",
            )
        image_rgb = _decode_image(image_bytes)
        if isinstance(image_rgb, JSONResponse):
            return image_rgb

        components: ServerComponents = app.state.components
        try:
            result = pipeline_runner(
                image_rgb,
                diameter,
                components.food_segmenter,
                components.depth_estimator,
                obstacle_detector=components.obstacle_detector,
                initial_temperature_c=initial_temperature,
            )
        except PipelineError as error:
            return _error_response(422, "pipeline_error", str(error))
        except Exception:
            logger.exception("unexpected pipeline failure")
            return _error_response(500, "internal_error", "internal server error")

        if result_serializer is None:
            from main import pipeline_result_to_dict

            payload = pipeline_result_to_dict(result)
        else:
            payload = result_serializer(result)
        return JSONResponse(content=payload)

    return app
