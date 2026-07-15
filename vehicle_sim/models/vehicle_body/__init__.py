"""Vehicle body dynamics module."""

__all__ = ["StbwVehicleBody", "VehicleBody"]


def __getattr__(name):
    if name in {"StbwVehicleBody", "VehicleBody"}:
        from .vehicle_body import StbwVehicleBody, VehicleBody

        mapping = {
            "StbwVehicleBody": StbwVehicleBody,
            "VehicleBody": VehicleBody,
        }
        return mapping[name]
    raise AttributeError(f"module 'vehicle_sim.models.vehicle_body' has no attribute {name!r}")
