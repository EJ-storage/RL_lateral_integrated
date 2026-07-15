"""STBW tire compatibility aliases."""

from .dugoff import (
    ModifiedDugoff4WTireModel,
    ModifiedDugoff4WTireParameters,
    ModifiedDugoff4WTireState,
    ModifiedDugoffTireModel,
    ModifiedDugoffWheelState,
)
from .lateral import (
    FialaLateralTireModel,
    FialaLateralTireParameters,
    FialaLateralTireState,
    LateralTireModel,
    StbwLateralTireModel,
    StbwLateralTireParameters,
    StbwLateralTireState,
)
from .longitudinal import (
    FialaLongitudinalTireModel,
    FialaLongitudinalTireParameters,
    FialaLongitudinalTireState,
    LongitudinalTireModel,
    StbwLongitudinalTireModel,
    StbwLongitudinalTireParameters,
    StbwLongitudinalTireState,
)

__all__ = [
    "FialaLateralTireModel",
    "FialaLateralTireParameters",
    "FialaLateralTireState",
    "FialaLongitudinalTireModel",
    "FialaLongitudinalTireParameters",
    "FialaLongitudinalTireState",
    "LateralTireModel",
    "LongitudinalTireModel",
    "ModifiedDugoff4WTireModel",
    "ModifiedDugoff4WTireParameters",
    "ModifiedDugoff4WTireState",
    "ModifiedDugoffTireModel",
    "ModifiedDugoffWheelState",
    "StbwLateralTireModel",
    "StbwLateralTireParameters",
    "StbwLateralTireState",
    "StbwLongitudinalTireModel",
    "StbwLongitudinalTireParameters",
    "StbwLongitudinalTireState",
]
