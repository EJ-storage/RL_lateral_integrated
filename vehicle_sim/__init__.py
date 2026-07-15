"""Vehicle dynamics simulation package."""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "VehicleBody",
    "StbwVehicleBody",
    "ECorner",
    "Stbw",
    "ActiveAntiRollBarController",
    "ActiveAntiRollBarGains",
    "scenarios",
]


def __getattr__(name):
    if name in {"VehicleBody", "StbwVehicleBody", "Stbw"}:
        from . import stbw_model

        return getattr(stbw_model, name)
    if name in {"ECorner"}:
        from . import models

        return getattr(models, name)
    if name in {"ActiveAntiRollBarController", "ActiveAntiRollBarGains"}:
        from . import controllers

        return getattr(controllers, name)
    if name == "scenarios":
        from . import scenarios

        return scenarios
    raise AttributeError(f"module 'vehicle_sim' has no attribute {name!r}")
