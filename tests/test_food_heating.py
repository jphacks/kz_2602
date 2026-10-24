from __future__ import annotations

import math

import pytest

from food_heating import (
    DEFAULT_FREQUENCY_HZ,
    FoodHeatingError,
    FoodMaterialProperties,
    calculate_penetration_depth,
    estimate_heating_parameters,
    lookup_food_properties,
)
from food_measurement import FoodMeasurement


def make_measurement(
    height_cm: float = 1.0, weight_g: float = 100.0
) -> FoodMeasurement:
    return FoodMeasurement(
        top_x_px=0,
        top_y_px=0,
        height_cm=height_cm,
        area_cm2=10.0,
        volume_cm3=100.0,
        weight_g=weight_g,
    )


@pytest.mark.parametrize(
    ("class_id", "density", "epsilon_prime", "epsilon_double_prime", "tan_delta", "cp"),
    [
        ("rice", 1.05, 40.0, 12.0, 0.30, 2.5),
        ("meat", 1.00, 50.0, 20.0, 0.40, 3.2),
        ("vegetable", 0.95, 60.0, 15.0, 0.25, 3.8),
        ("bread", 0.35, 5.0, 1.0, 0.20, 1.8),
        ("bowl", 1.20, 4.0, 0.20, 0.05, 1.00),
    ],
)
def test_lookup_food_properties_returns_design_values(
    class_id: str,
    density: float,
    epsilon_prime: float,
    epsilon_double_prime: float,
    tan_delta: float,
    cp: float,
) -> None:
    properties = lookup_food_properties(class_id)

    assert properties == FoodMaterialProperties(
        class_id,
        density,
        epsilon_prime,
        epsilon_double_prime,
        tan_delta,
        cp,
    )


def test_lookup_rejects_unknown_class() -> None:
    with pytest.raises(FoodHeatingError, match="unsupported"):
        lookup_food_properties("soup")


def test_calculate_penetration_depth_uses_electric_field_one_over_e() -> None:
    properties = lookup_food_properties("rice")
    loss_term = math.sqrt(1.0 + 0.30**2) - 1.0
    attenuation = (
        2.0
        * math.pi
        * DEFAULT_FREQUENCY_HZ
        / 299_792_458.0
        * math.sqrt(40.0 * loss_term / 2.0)
    )

    assert calculate_penetration_depth(properties) == pytest.approx(100.0 / attenuation)


def test_estimate_marks_food_thicker_than_penetration_depth() -> None:
    parameters = estimate_heating_parameters(
        "meat",
        make_measurement(
            height_cm=calculate_penetration_depth(lookup_food_properties("meat")) + 0.01
        ),
    )

    assert parameters.needs_turning_or_stirring is True


def test_estimate_does_not_mark_equal_thickness() -> None:
    penetration_depth = calculate_penetration_depth(lookup_food_properties("bread"))

    parameters = estimate_heating_parameters(
        "bread", make_measurement(height_cm=penetration_depth)
    )

    assert parameters.needs_turning_or_stirring is False


def test_estimate_computes_energy_and_time() -> None:
    parameters = estimate_heating_parameters("rice", make_measurement(weight_g=100.0))

    assert parameters.required_energy_j == pytest.approx(100.0 * 2.5 * 55.0)
    assert parameters.heating_time_s == pytest.approx(
        100.0 * 2.5 * 55.0 / (500.0 * 0.7)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_temperature_c": 20.0},
        {"power_w": 0.0},
        {"efficiency": 0.0},
        {"efficiency": 1.1},
        {"frequency_hz": math.nan},
    ],
)
def test_estimate_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(FoodHeatingError):
        estimate_heating_parameters("rice", make_measurement(), **kwargs)


@pytest.mark.parametrize("field", ["height_cm", "weight_g"])
@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan])
def test_estimate_rejects_invalid_measurement(field: str, value: float) -> None:
    measurement_values = {
        "height_cm": 1.0,
        "weight_g": 100.0,
    }
    measurement_values[field] = value
    measurement = make_measurement(**measurement_values)

    with pytest.raises(FoodHeatingError, match="measurement"):
        estimate_heating_parameters("rice", measurement)
