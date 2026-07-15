"""STBW drive model compatibility aliases."""

from pathlib import Path

from vehicle_sim.models.drive.drive_model import (
    BrakeTorqueParameters,
    DriveModel,
    DriveParameters,
    DriveState,
)
from vehicle_sim.utils.config_loader import load_param


STBW_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "stbw.yaml"


def _resolve_config_path(config_path):
    if config_path is None or str(config_path).lower() in {"stbw", "stbw.yaml"}:
        return STBW_CONFIG_PATH
    return config_path


def _float_from(mapping, key, default=0.0):
    try:
        value = float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value


def _axle_group(axle_id):
    label = str(axle_id or "").upper()
    if label.startswith("F"):
        return "F"
    if label.startswith("R"):
        return "R"
    return ""


def _build_drive_parameters(config_path, axle_id):
    vehicle_spec = load_param("vehicle_spec", config_path)
    wheel_spec = vehicle_spec.get("wheel", {})
    drive_param = load_param("drive", config_path)
    group = _axle_group(axle_id)

    if group == "F":
        j_wheel = _float_from(
            wheel_spec,
            "J_wheel_front",
            _float_from(wheel_spec, "J_wheel"),
        )
        b_wheel = _float_from(
            wheel_spec,
            "B_wheel_front",
            _float_from(wheel_spec, "B_wheel"),
        )
    elif group == "R":
        j_wheel = _float_from(
            wheel_spec,
            "J_wheel_rear",
            _float_from(wheel_spec, "J_wheel"),
        )
        b_wheel = _float_from(
            wheel_spec,
            "B_wheel_rear",
            _float_from(wheel_spec, "B_wheel"),
        )
    else:
        j_wheel = _float_from(wheel_spec, "J_wheel")
        b_wheel = _float_from(wheel_spec, "B_wheel")

    return DriveParameters(
        J_wheel=j_wheel,
        R_wheel=_float_from(wheel_spec, "R_eff"),
        B_wheel=max(b_wheel, 0.0),
        max_wheel_speed=_float_from(drive_param, "max_wheel_speed"),
    )


def _build_brake_torque_parameters(config_path):
    brake_param = load_param("brake", config_path)
    return BrakeTorqueParameters(
        mu_pad=_float_from(brake_param, "mu_pad"),
        R_rotor=_float_from(brake_param, "R_rotor"),
    )


class StbwDriveModel(DriveModel):
    def __init__(self, config_path=None, axle_id=None):
        resolved_config_path = _resolve_config_path(config_path)
        super().__init__(
            parameters=_build_drive_parameters(resolved_config_path, axle_id),
            brake_parameters=_build_brake_torque_parameters(resolved_config_path),
        )
        self.state.wheel_speed = 0.0

    def reset(self) -> None:
        super().reset()
        self.state.wheel_speed = 0.0


StbwDriveParameters = DriveParameters
StbwBrakeTorqueParameters = BrakeTorqueParameters
StbwDriveState = DriveState

__all__ = [
    "STBW_CONFIG_PATH",
    "StbwBrakeTorqueParameters",
    "StbwDriveModel",
    "StbwDriveParameters",
    "StbwDriveState",
]
