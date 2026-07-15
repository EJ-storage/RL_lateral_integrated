"""STBW-specific vehicle body wrapper backed by stbw.yaml."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from vehicle_sim.models.vehicle_body.vehicle_body import (
    StbwVehicleBody as BaseStbwVehicleBody,
    StbwVehicleBodyParameters,
    StbwVehicleBodyState,
)
from vehicle_sim.utils.config_loader import load_param

from ..steering.steering_model import resolve_stbw_config_path


def _float_value(mapping: Mapping, key: str, default: float = 0.0) -> float:
    return float(mapping.get(key, default))


def _build_axle_offsets(geometry: Mapping) -> Dict[str, Dict[str, float]]:
    raw_offsets = geometry.get("axle_offsets") or {}
    lf = _float_value(geometry, "lf")
    lr = _float_value(geometry, "lr")

    return {
        "F": {
            "x": _float_value(raw_offsets.get("F", {}), "x", lf),
            "y": _float_value(raw_offsets.get("F", {}), "y"),
        },
        "R": {
            "x": _float_value(raw_offsets.get("R", {}), "x", -lr),
            "y": _float_value(raw_offsets.get("R", {}), "y"),
        },
    }


def _build_corner_offsets(
    geometry: Mapping,
    axle_offsets: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    raw_offsets = geometry.get("corner_offsets") or {}
    half_track = 0.5 * _float_value(
        geometry,
        "L_track",
        _float_value(geometry, "track_width"),
    )

    return {
        "FL": {
            "x": _float_value(raw_offsets.get("FL", {}), "x", axle_offsets["F"]["x"]),
            "y": _float_value(raw_offsets.get("FL", {}), "y", half_track),
        },
        "FR": {
            "x": _float_value(raw_offsets.get("FR", {}), "x", axle_offsets["F"]["x"]),
            "y": _float_value(raw_offsets.get("FR", {}), "y", -half_track),
        },
        "RL": {
            "x": _float_value(raw_offsets.get("RL", {}), "x", axle_offsets["R"]["x"]),
            "y": _float_value(raw_offsets.get("RL", {}), "y", half_track),
        },
        "RR": {
            "x": _float_value(raw_offsets.get("RR", {}), "x", axle_offsets["R"]["x"]),
            "y": _float_value(raw_offsets.get("RR", {}), "y", -half_track),
        },
    }


def _build_parameters(
    vehicle_body: Mapping,
    physics: Mapping,
    axle_offsets: Mapping[str, Mapping[str, float]],
) -> StbwVehicleBodyParameters:
    inertia = vehicle_body.get("inertia") or {}

    return StbwVehicleBodyParameters(
        m_total=_float_value(vehicle_body, "m"),
        Izz=_float_value(inertia, "Izz"),
        a=abs(float(axle_offsets["F"]["x"])),
        b=abs(float(axle_offsets["R"]["x"])),
        h_CG=_float_value(vehicle_body, "h_CG"),
        g=_float_value(physics, "g"),
    )


class StbwVehicleBody(BaseStbwVehicleBody):
    def __init__(
        self,
        parameters: Optional[StbwVehicleBodyParameters] = None,
        config_path: Optional[str] = None,
        drive_axles: str = "R",
    ):
        resolved_config_path = resolve_stbw_config_path(config_path)
        vehicle_spec = load_param("vehicle_spec", resolved_config_path)
        vehicle_body = load_param("vehicle_body", resolved_config_path)
        physics = load_param("physics", resolved_config_path)
        geometry = vehicle_spec.get("geometry") or {}

        axle_offsets = _build_axle_offsets(geometry)
        corner_offsets = _build_corner_offsets(geometry, axle_offsets)
        if parameters is None:
            parameters = _build_parameters(vehicle_body, physics, axle_offsets)

        super().__init__(
            parameters=parameters,
            config_path=str(resolved_config_path),
            drive_axles=drive_axles,
            axle_offsets=axle_offsets,
            corner_offsets=corner_offsets,
        )


VehicleBody = StbwVehicleBody


__all__ = [
    "BaseStbwVehicleBody",
    "StbwVehicleBody",
    "StbwVehicleBodyParameters",
    "StbwVehicleBodyState",
    "VehicleBody",
]
