"""STBW steering model compatibility aliases and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from vehicle_sim.models.steering.steering_model import (
    SteeringModel as _SteeringModel,
    SteeringParameters,
    SteeringState,
)


STBW_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "stbw.yaml"


def resolve_stbw_config_path(config_path):
    if config_path is None or str(config_path).lower() in {"stbw", "stbw.yaml"}:
        return STBW_CONFIG_PATH
    return config_path


class SteeringModel(_SteeringModel):
    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        corner_id: Optional[str] = None,
        side: Optional[str] = None,
    ):
        super().__init__(
            config=config,
            config_path=resolve_stbw_config_path(config_path),
            corner_id=corner_id,
            side=side,
        )
        self._reset_state_defaults()

    def reset(self) -> None:
        super().reset()
        self._reset_state_defaults()

    def _reset_state_defaults(self) -> None:
        self.state.steering_angle = 0.0
        self.state.steering_rate = 0.0
        self.state.steering_torque = 0.0
        self.state.self_aligning_torque = 0.0


class StbwSteeringModel(SteeringModel):
    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        axle_id: Optional[str] = None,
    ):
        super().__init__(
            config=config,
            config_path=resolve_stbw_config_path(config_path),
            corner_id=axle_id,
        )

    def _safe_steering_ratio(self) -> float:
        return max(abs(float(self.params.steering_ratio)), 1e-6)

    def road_wheel_angle_to_steering_wheel_angle(self, road_wheel_angle: float) -> float:
        return float(road_wheel_angle) * self._safe_steering_ratio()

    def steering_wheel_angle_to_road_wheel_angle(self, steering_wheel_angle: float) -> float:
        return float(steering_wheel_angle) / self._safe_steering_ratio()

    def road_wheel_rate_to_steering_wheel_rate(self, road_wheel_rate: float) -> float:
        return float(road_wheel_rate) * self._safe_steering_ratio()

    def steering_wheel_rate_to_road_wheel_rate(self, steering_wheel_rate: float) -> float:
        return float(steering_wheel_rate) / self._safe_steering_ratio()

    def road_wheel_ddot_to_steering_wheel_ddot(self, road_wheel_ddot: float) -> float:
        return float(road_wheel_ddot) * self._safe_steering_ratio()

    def steering_wheel_ddot_to_road_wheel_ddot(self, steering_wheel_ddot: float) -> float:
        return float(steering_wheel_ddot) / self._safe_steering_ratio()

    def get_road_wheel_angle(self) -> float:
        return float(self.state.steering_angle)

    def get_road_wheel_rate(self) -> float:
        return float(self.state.steering_rate)

    def get_steering_wheel_angle(self) -> float:
        return self.road_wheel_angle_to_steering_wheel_angle(self.state.steering_angle)

    def get_steering_wheel_rate(self) -> float:
        return self.road_wheel_rate_to_steering_wheel_rate(self.state.steering_rate)

    def update(self, dt: float, T_str: float, T_align: float = 0.0) -> Tuple[float, float]:
        angle = super().update(dt, T_str, T_align)
        return float(angle), float(self.state.steering_rate)

    def update_front_ddot(self, delta_ddot_cmd: float, T_align: Optional[float] = None) -> float:
        if T_align is None:
            T_align = self.state.self_aligning_torque

        T_str = (
            self.params.J_cq * float(delta_ddot_cmd)
            + self.params.B_cq * float(self.state.steering_rate)
            + float(T_align)
        ) / self.params.gear_ratio
        self.state.steering_torque = float(T_str) * self.params.gear_ratio
        self.state.self_aligning_torque = float(T_align)
        return float(T_str)

    def update_from_ddot(
        self,
        dt: float,
        delta_ddot_cmd: float,
        T_align: float = 0.0,
    ) -> Tuple[float, float]:
        T_str = self.update_front_ddot(delta_ddot_cmd, T_align)
        return self.update(dt, T_str, T_align)

    def get_state(self) -> Dict:
        state = super().get_state()
        state.update(
            {
                "road_wheel_angle": self.get_road_wheel_angle(),
                "road_wheel_rate": self.get_road_wheel_rate(),
                "steering_wheel_angle": self.get_steering_wheel_angle(),
                "steering_wheel_rate": self.get_steering_wheel_rate(),
            }
        )
        return state


StbwSteeringParameters = SteeringParameters
StbwSteeringState = SteeringState

__all__ = [
    "SteeringModel",
    "SteeringParameters",
    "SteeringState",
    "STBW_CONFIG_PATH",
    "StbwSteeringModel",
    "StbwSteeringParameters",
    "StbwSteeringState",
    "resolve_stbw_config_path",
]
