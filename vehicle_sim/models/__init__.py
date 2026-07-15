"""Vehicle dynamics models."""

__all__ = ["VehicleBody", "StbwVehicleBody", "ECorner", "Stbw"]


def __getattr__(name):
    if name in {"VehicleBody", "StbwVehicleBody"}:
        from .vehicle_body.vehicle_body import StbwVehicleBody, VehicleBody

        mapping = {
            "VehicleBody": VehicleBody,
            "StbwVehicleBody": StbwVehicleBody,
        }
        return mapping[name]
    if name in {"ECorner", "Stbw"}:
        from .e_corner.e_corner import ECorner, Stbw

        mapping = {
            "ECorner": ECorner,
            "Stbw": Stbw,
        }
        return mapping[name]
    raise AttributeError(f"module 'vehicle_sim.models' has no attribute {name!r}")
