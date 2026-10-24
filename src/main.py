"""Application entry-point helpers for the food estimation pipeline."""

from __future__ import annotations

from typing import Any

from pipeline import PipelineResult


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
