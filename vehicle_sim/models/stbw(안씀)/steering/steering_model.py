"""Steering actuator models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from vehicle_sim.utils.config_loader import load_param


@dataclass
class SteeringParameters:
    J_cq: float = 0.05
    B_cq: float = 0.5
    gear_ratio: float = 118.0
    steering_ratio: float = 1.0
    max_angle_pos: float = 0.0
    max_angle_neg: float = 0.0
    max_rate: float = np.deg2rad(360.0)


@dataclass
class SteeringState:
    steering_angle: float = 0.0
    steering_rate: float = 0.0
    steering_torque: float = 0.0
    self_aligning_torque: float = 0.0


@dataclass
class StbwSteeringParameters:
    J_cq: float = 0.0738
    B_cq: float = 85.61
    gear_ratio: float = 118.0
    steering_ratio: float = 15.0
    max_angle_pos: float = 0.0
    max_angle_neg: float = 0.0
    max_rate: float = np.deg2rad(360.0)


@dataclass
class StbwSteeringState:
    steering_angle: float = 0.0
    steering_rate: float = 0.0
    steering_torque: float = 0.0
    self_aligning_torque: float = 0.0

class StbwSteeringModel:
    """Steer-by-wire steering actuator model."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        axle_id: Optional[str] = None,
    ):
        del axle_id
        steering_param = config if config is not None else load_param("steering", config_path)
        self.params = StbwSteeringParameters()
        self.params.J_cq = float(steering_param.get("J_cq", self.params.J_cq))
        self.params.B_cq = float(steering_param.get("B_cq", self.params.B_cq))
        self.params.gear_ratio = float(steering_param.get("gear_ratio", self.params.gear_ratio))
        self.params.steering_ratio = float(steering_param.get("steering_ratio", self.params.steering_ratio))
        self.params.max_rate = float(steering_param.get("max_rate", self.params.max_rate))
        self.params.max_angle_pos = self._get_angle_limit(steering_param, "max_angle_pos")
        self.params.max_angle_neg = self._get_angle_limit(steering_param, "max_angle_neg")
        if self.params.max_angle_pos == 0.0 and self.params.max_angle_neg == 0.0:
            raise ValueError("Steering max_angle_pos/max_angle_neg must be provided via config")

        self.state = StbwSteeringState()

    @staticmethod
    def _get_angle_limit(steering_param: Dict, key: str) -> float:
        if isinstance(steering_param, dict) and key in steering_param:
            return float(steering_param.get(key, 0.0))
        if isinstance(steering_param, dict):
            front_cfg = steering_param.get("front", {})
            if isinstance(front_cfg, dict) and key in front_cfg:
                return float(front_cfg.get(key, 0.0))
        return 0.0

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
        self.state.steering_torque = float(T_str) * self.params.gear_ratio
        self.state.self_aligning_torque = float(T_align)

        delta = self.state.steering_angle
        delta_dot = self.state.steering_rate
        numerator = (
            float(T_str) * self.params.gear_ratio
            - float(T_align)
            - self.params.B_cq * delta_dot
        )
        delta_ddot = numerator / self.params.J_cq

        lower = min(self.params.max_angle_neg, self.params.max_angle_pos)
        upper = max(self.params.max_angle_neg, self.params.max_angle_pos)
        at_upper_and_outward = delta >= upper and delta_dot >= 0.0 and delta_ddot >= 0.0
        at_lower_and_outward = delta <= lower and delta_dot <= 0.0 and delta_ddot <= 0.0
        if at_upper_and_outward or at_lower_and_outward:
            delta_ddot = 0.0
            delta_dot = 0.0

        limited_rate = self.apply_rate_limits(delta_dot + dt * delta_ddot)
        limited_angle = self.apply_angle_limits(delta + dt * limited_rate)

        self.state.steering_rate = limited_rate
        self.state.steering_angle = limited_angle
        return self.state.steering_angle, self.state.steering_rate

    def update_front_ddot(self, delta_ddot_cmd: float, T_align: Optional[float] = None) -> float: 
        if T_align is None:
            T_align = self.state.self_aligning_torque

        delta_dot = self.state.steering_rate
        T_str = (
            self.params.J_cq * float(delta_ddot_cmd)
            + self.params.B_cq * delta_dot
            + float(T_align)
        ) / self.params.gear_ratio
        self.state.steering_torque = float(T_str) * self.params.gear_ratio 
        self.state.self_aligning_torque = float(T_align)
        return float(T_str) # action으로 나온 road wheel ddot을 조향 토크로 역산해서 차량 모델 입력으로 사용 

    def update_from_ddot(self, dt: float, delta_ddot_cmd: float, T_align: float = 0.0) -> Tuple[float, float]:
        T_str = self.update_front_ddot(delta_ddot_cmd, T_align)
        return self.update(dt, T_str, T_align) # 여기서 차량 업데이트, 차량 모델 내부에서 다시 road wheel ddot이 계산되어 사용됨, 차량 모델 입력이 토크로 돼있어서 그렇게 할 수밖에 없음

    def apply_angle_limits(self, angle: float) -> float:
        lower = min(self.params.max_angle_neg, self.params.max_angle_pos)
        upper = max(self.params.max_angle_neg, self.params.max_angle_pos)
        return float(np.clip(angle, lower, upper))

    def apply_rate_limits(self, desired_rate: float) -> float:
        return float(np.clip(desired_rate, -self.params.max_rate, self.params.max_rate))

    def get_state(self) -> Dict:
        return {
            "steering_angle": self.state.steering_angle,
            "steering_rate": self.state.steering_rate,
            "road_wheel_angle": self.get_road_wheel_angle(),
            "road_wheel_rate": self.get_road_wheel_rate(),
            "steering_wheel_angle": self.get_steering_wheel_angle(),
            "steering_wheel_rate": self.get_steering_wheel_rate(),
            "steering_torque": self.state.steering_torque,
            "self_aligning_torque": self.state.self_aligning_torque,
        }

    def reset(self) -> None:
        self.state = StbwSteeringState()
