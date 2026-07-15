"""Combined longitudinal/lateral tire models."""

from .dugoff import (
    ModifiedDugoff4WTireModel,
    ModifiedDugoff4WTireParameters,
    ModifiedDugoff4WTireState,
    ModifiedDugoffTireModel,
    ModifiedDugoffWheelState,
)

__all__ = [
    "ModifiedDugoff4WTireModel",
    "ModifiedDugoff4WTireParameters",
    "ModifiedDugoff4WTireState",
    "ModifiedDugoffTireModel",
    "ModifiedDugoffWheelState",
]
