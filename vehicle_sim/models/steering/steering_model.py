"""Steering actuator model."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from vehicle_sim.utils.config_loader import load_param


@dataclass
class SteeringParameters:
    J_cq: float = 0.0
    B_cq: float = 0.0
    gear_ratio: float = 0.0
    steering_ratio: float = 0.0
    max_angle_pos: float = 0.0
    max_angle_neg: float = 0.0
    max_rate: float = 0.0


@dataclass
class SteeringState:
    steering_angle: float = 0.0
    steering_rate: float = 0.0
    steering_torque: float = 0.0
    self_aligning_torque: float = 0.0


class SteeringModel:
    """Electric steering actuator model."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        corner_id: Optional[str] = None,
        side: Optional[str] = None,
    ):
        steering_param = config if config is not None else load_param("steering", config_path)

        self.params = SteeringParameters()
        self.params.J_cq = float(steering_param.get("J_cq", self.params.J_cq))
        self.params.B_cq = float(steering_param.get("B_cq", self.params.B_cq))
        self.params.gear_ratio = float(steering_param.get("gear_ratio", self.params.gear_ratio))
        self.params.steering_ratio = float(
            steering_param.get("steering_ratio", self.params.steering_ratio)
        )
        self.params.max_rate = float(steering_param.get("max_rate", self.params.max_rate))

        axle_key = None
        side_key = None
        if side:
            side_key = "left" if side.lower().startswith("l") else "right"
        elif corner_id:
            corner_key = str(corner_id).upper()
            axle_key = "front" if corner_key.startswith("F") else "rear"
            if corner_key in ["FL", "RL"]:
                side_key = "left"
            elif corner_key in ["FR", "RR"]:
                side_key = "right"

        def get_angle_limit(key: str) -> float:
            if isinstance(steering_param, dict) and key in steering_param:
                return float(steering_param.get(key, 0.0))
            if axle_key and isinstance(steering_param, dict):
                axle_cfg = steering_param.get(axle_key, {})
                if isinstance(axle_cfg, dict) and key in axle_cfg:
                    return float(axle_cfg.get(key, 0.0))
            if side_key and isinstance(steering_param, dict):
                side_cfg = steering_param.get(side_key, {})
                if isinstance(side_cfg, dict) and key in side_cfg:
                    return float(side_cfg.get(key, 0.0))
            return 0.0

        self.params.max_angle_pos = get_angle_limit("max_angle_pos")
        self.params.max_angle_neg = get_angle_limit("max_angle_neg")

        if self.params.max_angle_pos == 0.0 and self.params.max_angle_neg == 0.0:
            raise ValueError("Steering max_angle_pos/max_angle_neg must be provided via config")

        self.state = SteeringState()

    def update(self, dt: float, T_str: float, T_align: float = 0.0) -> float:
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

        return self.state.steering_angle

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
            "steering_torque": self.state.steering_torque,
            "self_aligning_torque": self.state.self_aligning_torque,
        }

    def reset(self) -> None:
        self.state = SteeringState()
