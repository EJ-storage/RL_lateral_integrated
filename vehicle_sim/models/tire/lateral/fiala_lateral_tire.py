#!/bin/python3
"""Shared Fiala lateral tire model.

This module is model math only. Concrete vehicle packages are responsible for
loading YAML configuration and passing ``FialaLateralTireParameters``.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class FialaLateralTireParameters:
    C_alpha: float = 0.0
    mu: float = 0.0
    trail: float = 0.0
    vx_epsilon: float = 0.0


@dataclass
class FialaLateralTireState:
    slip_angle: float = 0.0
    slip_angle_dyn: float = 0.0
    lateral_force: float = 0.0
    aligning_torque: float = 0.0


class FialaLateralTireModel:
    def __init__(
        self,
        parameters: Optional[FialaLateralTireParameters] = None,
        config_path: Optional[str] = None,
    ):
        del config_path
        self.params = parameters if parameters is not None else FialaLateralTireParameters()
        self.state = FialaLateralTireState()

    def update(self, dt: float, V_wheel_x: float, V_wheel_y: float, F_tire: float) -> float:
        del dt
        alpha = self.calculate_slip_angle(V_wheel_x, V_wheel_y)
        Fy = self.calculate_force(alpha, F_tire)
        M_z = self.calculate_aligning_torque(Fy)

        self.state.slip_angle = alpha
        self.state.slip_angle_dyn = alpha
        self.state.lateral_force = Fy
        self.state.aligning_torque = M_z

        return Fy

    def calculate_slip_angle(self, V_wheel_x: float, V_wheel_y: float) -> float:
        alpha = np.arctan2(V_wheel_y, V_wheel_x)
        return float(alpha)

    def calculate_force(self, alpha: float, F_tire: float) -> float:
        F_z = abs(float(F_tire))
        mu = float(self.params.mu)
        C_alpha = float(self.params.C_alpha)

        if F_z <= 1e-6:
            return 0.0

        tan_alpha = np.tan(alpha)
        alpha_sl = np.arctan((3.0 * mu * F_z) / C_alpha)

        if abs(alpha) < alpha_sl:
            Fy = (
                -C_alpha * tan_alpha
                + (C_alpha**2 / (3.0 * mu * F_z)) * abs(tan_alpha) * tan_alpha
                - (C_alpha**3 / (27.0 * mu**2 * F_z**2)) * (tan_alpha**3)
            )
        else:
            Fy = -mu * F_z * np.sign(alpha)

        return float(Fy)

    def calculate_aligning_torque(self, Fy: float) -> float:
        M_z = self.params.trail * Fy
        return float(M_z)

    def get_state(self) -> Dict:
        return {
            "slip_angle": self.state.slip_angle,
            "slip_angle_dyn": self.state.slip_angle_dyn,
            "lateral_force": self.state.lateral_force,
            "aligning_torque": self.state.aligning_torque,
        }

    def reset(self) -> None:
        self.state = FialaLateralTireState()
