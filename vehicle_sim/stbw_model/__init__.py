"""STBW wheel model compatibility package."""

from .drive import (
    StbwBrakeModel,
    StbwBrakeParameters,
    StbwBrakeState,
    StbwBrakeTorqueParameters,
    StbwDriveModel,
    StbwDriveParameters,
    StbwDriveState,
)
from .stbw import Stbw, StbwParameters, StbwState
from .steering import (
    StbwSteeringModel,
    StbwSteeringParameters,
    StbwSteeringState,
    SteeringModel,
    SteeringParameters,
    SteeringState,
    resolve_stbw_config_path,
)
from .tire import (
    LateralTireModel,
    LongitudinalTireModel,
    ModifiedDugoff4WTireModel,
    ModifiedDugoff4WTireParameters,
    ModifiedDugoff4WTireState,
    ModifiedDugoffTireModel,
    ModifiedDugoffWheelState,
    StbwLateralTireModel,
    StbwLateralTireParameters,
    StbwLateralTireState,
    StbwLongitudinalTireModel,
    StbwLongitudinalTireParameters,
    StbwLongitudinalTireState,
)

__all__ = [
    "LateralTireModel",
    "LongitudinalTireModel",
    "ModifiedDugoff4WTireModel",
    "ModifiedDugoff4WTireParameters",
    "ModifiedDugoff4WTireState",
    "ModifiedDugoffTireModel",
    "ModifiedDugoffWheelState",
    "Stbw",
    "StbwBrakeModel",
    "StbwBrakeParameters",
    "StbwBrakeState",
    "StbwBrakeTorqueParameters",
    "StbwDriveModel",
    "StbwDriveParameters",
    "StbwDriveState",
    "StbwLateralTireModel",
    "StbwLateralTireParameters",
    "StbwLateralTireState",
    "StbwLongitudinalTireModel",
    "StbwLongitudinalTireParameters",
    "StbwLongitudinalTireState",
    "StbwParameters",
    "StbwState",
    "StbwSteeringModel",
    "StbwSteeringParameters",
    "StbwSteeringState",
    "StbwVehicleBody",
    "StbwVehicleBodyParameters",
    "StbwVehicleBodyState",
    "VehicleBody",
    "resolve_stbw_config_path",
]


def __getattr__(name):
    if name in {
        "StbwVehicleBody",
        "StbwVehicleBodyParameters",
        "StbwVehicleBodyState",
        "VehicleBody",
    }:
        from .vehicle_body import (
            StbwVehicleBody,
            StbwVehicleBodyParameters,
            StbwVehicleBodyState,
            VehicleBody,
        )

        mapping = {
            "StbwVehicleBody": StbwVehicleBody,
            "StbwVehicleBodyParameters": StbwVehicleBodyParameters,
            "StbwVehicleBodyState": StbwVehicleBodyState,
            "VehicleBody": VehicleBody,
        }
        return mapping[name]
    raise AttributeError(f"module 'vehicle_sim.stbw_model' has no attribute {name!r}")
