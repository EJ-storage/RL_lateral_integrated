"""E-corner models."""

__all__ = ["ECorner", "ECornerState", "ECornerParameters", "Stbw", "StbwState", "StbwParameters"]


def __getattr__(name):
    if name in {"ECorner", "ECornerState", "ECornerParameters", "Stbw", "StbwState", "StbwParameters"}:
        from .e_corner import ECorner, ECornerParameters, ECornerState, Stbw, StbwParameters, StbwState

        mapping = {
            "ECorner": ECorner,
            "ECornerState": ECornerState,
            "ECornerParameters": ECornerParameters,
            "Stbw": Stbw,
            "StbwState": StbwState,
            "StbwParameters": StbwParameters,
        }
        return mapping[name]
    raise AttributeError(f"module 'vehicle_sim.models.e_corner' has no attribute {name!r}")
