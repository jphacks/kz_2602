"""Heating parameter estimation from food properties and measurements."""

from __future__ import annotations

from dataclasses import dataclass
import math

from food_measurement import FoodMeasurement

DEFAULT_FREQUENCY_HZ = 2.45e9
DEFAULT_INITIAL_TEMPERATURE_C = 20.0
DEFAULT_TARGET_TEMPERATURE_C = 75.0
DEFAULT_POWER_W = 500.0
DEFAULT_EFFICIENCY = 0.7
_LIGHT_SPEED_M_PER_S = 299_792_458.0


class FoodHeatingError(ValueError):
    """Raised when food heating input or calculation is invalid."""


@dataclass(frozen=True, slots=True)
class FoodMaterialProperties:
    """Thermal and dielectric properties for a supported food category."""

    class_id: str
    density_g_per_cm3: float
    epsilon_prime: float
    epsilon_double_prime: float
    tan_delta: float
    specific_heat_j_per_g_k: float


@dataclass(frozen=True, slots=True)
class HeatingParameters:
    """Estimated heating parameters for one measured food item."""

    class_id: str
    material: FoodMaterialProperties
    penetration_depth_cm: float
    needs_turning_or_stirring: bool
    required_energy_j: float
    heating_time_s: float


_FOOD_PROPERTIES = {
    "rice": FoodMaterialProperties("rice", 1.05, 40.0, 12.0, 0.30, 2.5),
    "meat": FoodMaterialProperties("meat", 1.00, 50.0, 20.0, 0.40, 3.2),
    "vegetable": FoodMaterialProperties("vegetable", 0.95, 60.0, 15.0, 0.25, 3.8),
    "bread": FoodMaterialProperties("bread", 0.35, 5.0, 1.0, 0.20, 1.8),
}


def lookup_food_properties(class_id: str) -> FoodMaterialProperties:
    """Return dielectric and thermal properties for ``class_id``."""
    if not isinstance(class_id, str) or not class_id:
        raise FoodHeatingError("class_id must be a non-empty string")
    try:
        return _FOOD_PROPERTIES[class_id]
    except KeyError as error:
        raise FoodHeatingError(f"unsupported food class: {class_id}") from error


def calculate_penetration_depth(
    properties: FoodMaterialProperties,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
) -> float:
    """Calculate electric-field 1/e penetration depth in centimeters."""
    _validate_properties(properties)
    _validate_positive_finite(frequency_hz, "frequency_hz")
    loss_term = math.sqrt(1.0 + properties.tan_delta**2) - 1.0
    attenuation_per_m = (
        2.0
        * math.pi
        * frequency_hz
        / _LIGHT_SPEED_M_PER_S
        * math.sqrt(properties.epsilon_prime * loss_term / 2.0)
    )
    if not math.isfinite(attenuation_per_m) or attenuation_per_m <= 0:
        raise FoodHeatingError("penetration depth could not be calculated")
    penetration_depth_cm = 100.0 / attenuation_per_m
    if not math.isfinite(penetration_depth_cm) or penetration_depth_cm <= 0:
        raise FoodHeatingError("penetration depth must be positive and finite")
    return penetration_depth_cm


def estimate_heating_parameters(
    class_id: str,
    measurement: FoodMeasurement,
    *,
    initial_temperature_c: float = DEFAULT_INITIAL_TEMPERATURE_C,
    target_temperature_c: float = DEFAULT_TARGET_TEMPERATURE_C,
    power_w: float = DEFAULT_POWER_W,
    efficiency: float = DEFAULT_EFFICIENCY,
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
) -> HeatingParameters:
    """Estimate penetration, mixing guidance, energy, and heating time."""
    properties = lookup_food_properties(class_id)
    _validate_measurement(measurement)
    _validate_finite(initial_temperature_c, "initial_temperature_c")
    _validate_finite(target_temperature_c, "target_temperature_c")
    if target_temperature_c <= initial_temperature_c:
        raise FoodHeatingError(
            "target_temperature_c must be greater than initial_temperature_c"
        )
    _validate_positive_finite(power_w, "power_w")
    if not math.isfinite(efficiency) or not 0 < efficiency <= 1:
        raise FoodHeatingError("efficiency must be finite and in (0, 1]")

    penetration_depth_cm = calculate_penetration_depth(properties, frequency_hz)
    temperature_delta_c = target_temperature_c - initial_temperature_c
    required_energy_j = (
        measurement.weight_g * properties.specific_heat_j_per_g_k * temperature_delta_c
    )
    heating_time_s = required_energy_j / (power_w * efficiency)
    values = (required_energy_j, heating_time_s)
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise FoodHeatingError("heating result must be finite and non-negative")

    return HeatingParameters(
        class_id=class_id,
        material=properties,
        penetration_depth_cm=penetration_depth_cm,
        needs_turning_or_stirring=measurement.height_cm > penetration_depth_cm,
        required_energy_j=required_energy_j,
        heating_time_s=heating_time_s,
    )


def _validate_measurement(measurement: FoodMeasurement) -> None:
    for field_name in ("height_cm", "weight_g"):
        value = getattr(measurement, field_name, None)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise FoodHeatingError(
                f"measurement.{field_name} must be finite and non-negative"
            )


def _validate_properties(properties: FoodMaterialProperties) -> None:
    if not isinstance(properties, FoodMaterialProperties):
        raise FoodHeatingError("properties must be FoodMaterialProperties")
    positive_fields = (
        "density_g_per_cm3",
        "epsilon_prime",
        "epsilon_double_prime",
        "tan_delta",
        "specific_heat_j_per_g_k",
    )
    for field_name in positive_fields:
        _validate_positive_finite(getattr(properties, field_name), field_name)


def _validate_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise FoodHeatingError(f"{name} must be finite")


def _validate_positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise FoodHeatingError(f"{name} must be positive and finite")
