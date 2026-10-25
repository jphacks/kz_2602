from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from pipeline import PipelineError
from server import MAX_IMAGE_BYTES, ServerComponents, create_app


class FakeModel:
    pass


def make_image_bytes() -> bytes:
    image_bgr = np.zeros((4, 5, 3), dtype=np.uint8)
    image_bgr[0, 0] = [10, 20, 30]
    success, encoded = cv2.imencode(".png", image_bgr)
    assert success
    return encoded.tobytes()


def make_app(
    pipeline_runner: object | None = None,
    component_factory: object | None = None,
):
    components = ServerComponents(FakeModel(), FakeModel(), FakeModel())
    return create_app(
        component_factory or (lambda: components),
        pipeline_runner=pipeline_runner or (lambda *args, **kwargs: {"ok": True}),
        result_serializer=lambda result: result,
    )


def test_healthz_and_estimate_return_result_and_rgb_image() -> None:
    captured: dict[str, object] = {}

    def run_pipeline(
        image_rgb: np.ndarray,
        diameter_cm: float,
        segmenter: object,
        depth_estimator: object,
        *,
        obstacle_detector: object,
        initial_temperature_c: float,
    ) -> dict[str, bool]:
        captured["image_rgb"] = image_rgb
        captured["diameter_cm"] = diameter_cm
        captured["segmenter"] = segmenter
        captured["depth_estimator"] = depth_estimator
        captured["obstacle_detector"] = obstacle_detector
        captured["initial_temperature_c"] = initial_temperature_c
        return {"ok": True}

    with TestClient(make_app(run_pipeline)) as client:
        health = client.get("/healthz")
        response = client.post(
            "/v1/estimate",
            files={"image": ("input.png", io.BytesIO(make_image_bytes()), "image/png")},
            data={"diameter_cm": "20"},
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert captured["diameter_cm"] == 20.0
    assert captured["initial_temperature_c"] == 20.0
    assert captured["image_rgb"][0, 0].tolist() == [30, 20, 10]


def test_estimate_rejects_missing_and_invalid_inputs() -> None:
    with TestClient(make_app()) as client:
        missing_image = client.post("/v1/estimate", data={"diameter_cm": "20"})
        missing_diameter = client.post(
            "/v1/estimate",
            files={"image": ("input.png", io.BytesIO(make_image_bytes()), "image/png")},
        )
        invalid_image = client.post(
            "/v1/estimate",
            files={"image": ("input.txt", io.BytesIO(b"not an image"), "text/plain")},
            data={"diameter_cm": "20"},
        )
        invalid_diameter = client.post(
            "/v1/estimate",
            files={"image": ("input.png", io.BytesIO(make_image_bytes()), "image/png")},
            data={"diameter_cm": "0"},
        )
        invalid_initial_temperature = client.post(
            "/v1/estimate",
            files={"image": ("input.png", io.BytesIO(make_image_bytes()), "image/png")},
            data={"diameter_cm": "20", "initial_temperature_c": "hot"},
        )
        too_high_initial_temperature = client.post(
            "/v1/estimate",
            files={"image": ("input.png", io.BytesIO(make_image_bytes()), "image/png")},
            data={"diameter_cm": "20", "initial_temperature_c": "75"},
        )

    assert missing_image.status_code == 400
    assert missing_image.json()["error"]["code"] == "missing_image"
    assert missing_diameter.status_code == 400
    assert missing_diameter.json()["error"]["code"] == "invalid_diameter"
    assert invalid_image.status_code == 400
    assert invalid_image.json()["error"]["code"] == "invalid_image"
    assert invalid_diameter.status_code == 400
    assert invalid_diameter.json()["error"]["code"] == "invalid_diameter"
    assert invalid_initial_temperature.status_code == 400
    assert (
        invalid_initial_temperature.json()["error"]["code"]
        == "invalid_initial_temperature"
    )
    assert too_high_initial_temperature.status_code == 400
    assert (
        too_high_initial_temperature.json()["error"]["code"]
        == "invalid_initial_temperature"
    )


def test_estimate_rejects_oversized_upload() -> None:
    with TestClient(make_app()) as client:
        response = client.post(
            "/v1/estimate",
            files={
                "image": (
                    "large.bin",
                    io.BytesIO(b"x" * (MAX_IMAGE_BYTES + 1)),
                    "application/octet-stream",
                )
            },
            data={"diameter_cm": "20"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "image_too_large"


def test_pipeline_error_is_returned_as_unprocessable_entity() -> None:
    def run_pipeline(*args: object, **kwargs: object) -> None:
        raise PipelineError("food segmentation failed")

    with TestClient(make_app(run_pipeline)) as client:
        response = client.post(
            "/v1/estimate",
            files={"image": ("input.png", io.BytesIO(make_image_bytes()), "image/png")},
            data={"diameter_cm": "20"},
        )

    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "pipeline_error", "message": "food segmentation failed"}
    }


def test_healthz_returns_service_unavailable_when_models_fail() -> None:
    def fail_to_create_components() -> ServerComponents:
        raise RuntimeError("model unavailable")

    with TestClient(make_app(component_factory=fail_to_create_components)) as client:
        response = client.get("/healthz")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_ready"
