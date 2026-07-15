from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from vehicle_sim.utils.config_loader import load_param


DEFAULT_SPEED_PI_CONFIG_PATH = "stbw"

@dataclass
class SpeedPIConfig:
    kp: Optional[float] = None
    ki: Optional[float] = None
    min_target_accel_mps2: Optional[float] = None
    max_target_accel_mps2: Optional[float] = None
    integrator_limit_mps: Optional[float] = None
    speed_deadband_mps: Optional[float] = None
    config_path: Optional[Union[str, Path]] = DEFAULT_SPEED_PI_CONFIG_PATH

    def __post_init__(self) -> None:
        yaml_config = load_param("speed_pi", self.config_path) if self.config_path is not None else {}
        self.kp = self._resolve_float("kp", yaml_config)
        self.ki = self._resolve_float("ki", yaml_config)
        self.min_target_accel_mps2 = self._resolve_float("min_target_accel_mps2", yaml_config)
        self.max_target_accel_mps2 = self._resolve_float("max_target_accel_mps2", yaml_config)
        self.integrator_limit_mps = abs(self._resolve_float("integrator_limit_mps", yaml_config))
        self.speed_deadband_mps = abs(self._resolve_float("speed_deadband_mps", yaml_config))
        if self.max_target_accel_mps2 <= self.min_target_accel_mps2:
            raise ValueError("max_target_accel_mps2 must be greater than min_target_accel_mps2.")

    def _resolve_float(self, name: str, yaml_config: dict) -> float:
        explicit_value = getattr(self, name)
        if explicit_value is not None:
            return float(explicit_value)
        return float(yaml_config.get(name))


@dataclass(frozen=True)
class SpeedPIOutput:
    target_speed_mps: float
    current_speed_mps: float
    speed_error_mps: float
    p_accel_mps2: float
    i_accel_mps2: float
    raw_target_accel_mps2: float
    target_accel_mps2: float
    integrator_state_mps: float
    saturated: bool


class SpeedPIController:
    def __init__(
        self,
        config: Optional[SpeedPIConfig] = None,
        config_path: Optional[Union[str, Path]] = DEFAULT_SPEED_PI_CONFIG_PATH,
    ) -> None:
        self.config = config if config is not None else SpeedPIConfig(config_path=config_path)
        self.integrator_state_mps = 0.0

    def reset(self, integrator_state_mps: float = 0.0) -> None:
        self.integrator_state_mps = float(integrator_state_mps)

    def update(
        self,
        *,
        target_speed_mps: float,
        current_speed_mps: float,
        dt: float,
    ) -> SpeedPIOutput:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        target_speed_mps = float(target_speed_mps)
        current_speed_mps = float(current_speed_mps)
        speed_error_mps = target_speed_mps - current_speed_mps
        if abs(speed_error_mps) < self.config.speed_deadband_mps:
            speed_error_for_integral = 0.0
        else:
            speed_error_for_integral = speed_error_mps

        candidate_integrator = self.integrator_state_mps + speed_error_for_integral * float(dt)
        candidate_integrator = float(
            np.clip(
                candidate_integrator,
                -self.config.integrator_limit_mps,
                self.config.integrator_limit_mps,
            )
        )

        output = self._build_output(
            target_speed_mps=target_speed_mps,
            current_speed_mps=current_speed_mps,
            speed_error_mps=speed_error_mps,
            integrator_state_mps=candidate_integrator,
        )
        saturation_pushes_farther = (
            output.raw_target_accel_mps2 > self.config.max_target_accel_mps2
            and speed_error_for_integral > 0.0
        ) or (
            output.raw_target_accel_mps2 < self.config.min_target_accel_mps2
            and speed_error_for_integral < 0.0
        )

        if output.saturated and saturation_pushes_farther:
            output = self._build_output(
                target_speed_mps=target_speed_mps,
                current_speed_mps=current_speed_mps,
                speed_error_mps=speed_error_mps,
                integrator_state_mps=self.integrator_state_mps,
            )
        else:
            self.integrator_state_mps = candidate_integrator

        return output

    def _build_output(
        self,
        *,
        target_speed_mps: float,
        current_speed_mps: float,
        speed_error_mps: float,
        integrator_state_mps: float,
    ) -> SpeedPIOutput:
        p_accel_mps2 = self.config.kp * speed_error_mps
        i_accel_mps2 = self.config.ki * integrator_state_mps
        raw_target_accel_mps2 = p_accel_mps2 + i_accel_mps2
        target_accel_mps2 = float(
            np.clip(
                raw_target_accel_mps2,
                self.config.min_target_accel_mps2,
                self.config.max_target_accel_mps2,
            )
        )
        saturated = not math.isclose(
            target_accel_mps2,
            raw_target_accel_mps2,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        return SpeedPIOutput(
            target_speed_mps=float(target_speed_mps),
            current_speed_mps=float(current_speed_mps),
            speed_error_mps=float(speed_error_mps),
            p_accel_mps2=float(p_accel_mps2),
            i_accel_mps2=float(i_accel_mps2),
            raw_target_accel_mps2=float(raw_target_accel_mps2),
            target_accel_mps2=float(target_accel_mps2),
            integrator_state_mps=float(integrator_state_mps),
            saturated=bool(saturated),
        )


@dataclass(frozen=True)
class SpeedHoldTorqueCommand:
    drive_torque_nm: float
    brake_motor_torque: float
    requested_accel_mps2: float


def target_accel_to_axle_torque(
    *,
    target_accel_mps2: float,
    vehicle_mass_kg: float,
    wheel_radius_m: float,
    brake_clamp_gain: float,
    brake_torque_gain: float,
    brake_axle_count: int,
) -> SpeedHoldTorqueCommand:
    target_accel_mps2 = float(target_accel_mps2)
    vehicle_mass_kg = max(float(vehicle_mass_kg), 1e-9)
    wheel_radius_m = max(float(wheel_radius_m), 1e-9)

    total_wheel_torque_nm = vehicle_mass_kg * abs(target_accel_mps2) * wheel_radius_m
    if target_accel_mps2 >= 0.0:
        return SpeedHoldTorqueCommand(
            drive_torque_nm=float(total_wheel_torque_nm),
            brake_motor_torque=0.0,
            requested_accel_mps2=float(target_accel_mps2),
        )

    brake_gain = max(float(brake_clamp_gain) * float(brake_torque_gain), 1e-9)
    axle_count = max(int(brake_axle_count), 1)
    brake_motor_torque = total_wheel_torque_nm / (brake_gain * axle_count)
    return SpeedHoldTorqueCommand(
        drive_torque_nm=0.0,
        brake_motor_torque=float(brake_motor_torque),
        requested_accel_mps2=float(target_accel_mps2),
    )
