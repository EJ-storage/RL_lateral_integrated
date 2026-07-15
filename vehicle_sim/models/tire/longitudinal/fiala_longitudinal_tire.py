#!/bin/python3
"""Shared Fiala longitudinal tire model.

This module is model math only. Concrete vehicle packages are responsible for
loading YAML configuration and passing ``FialaLongitudinalTireParameters``.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class FialaLongitudinalTireParameters:
    C_x: float = 0.0
    mu: float = 0.0
    v_min: float = 0.0
    R_eff: float = 0.0
    epsilon: float = 0.0


@dataclass
class FialaLongitudinalTireState:
    slip_ratio_cmd: float = 0.0
    slip_ratio: float = 0.0
    longitudinal_force: float = 0.0
    V_wheel_x: float = 0.0
    V_wheel: float = 0.0
    V_wheel_x_minus_V_wheel: float = 0.0
    slip_ratio_denom: float = 0.0


class FialaLongitudinalTireModel:
    MAX_ABS_SLIP_RATIO = 5.0

    def __init__(
        self,
        parameters: Optional[FialaLongitudinalTireParameters] = None,
        config_path: Optional[str] = None,
    ):
        del config_path
        self.params = (
            parameters
            if parameters is not None
            else FialaLongitudinalTireParameters()
        )
        self.state = FialaLongitudinalTireState()

    def _sanitize_slip_ratio(self, value: float, fallback: float = 0.0) -> float:
        if not np.isfinite(value):
            return float(fallback)
        return float(np.clip(value, -self.MAX_ABS_SLIP_RATIO, self.MAX_ABS_SLIP_RATIO))

    def calculate_slip_ratio(self, omega_wheel: float, V_wheel_x: float) -> float:
        if not np.isfinite(omega_wheel) or not np.isfinite(V_wheel_x):
            self.state.V_wheel_x = 0.0
            self.state.V_wheel = 0.0
            self.state.V_wheel_x_minus_V_wheel = 0.0
            self.state.slip_ratio_denom = float(self.params.v_min)
            self.state.slip_ratio_cmd = 0.0
            self.state.slip_ratio = 0.0
            return 0.0

        V_wheel = omega_wheel * self.params.R_eff
        denom = max(abs(V_wheel_x), self.params.v_min)
        kappa = (V_wheel - V_wheel_x) / denom

        kappa = self._sanitize_slip_ratio(kappa)
        self.state.V_wheel_x = float(V_wheel_x)
        self.state.V_wheel = float(V_wheel)
        self.state.V_wheel_x_minus_V_wheel = float(V_wheel_x - V_wheel)
        self.state.slip_ratio_denom = float(denom)
        self.state.slip_ratio_cmd = float(kappa)
        self.state.slip_ratio = float(kappa)
        return float(kappa)

    def calculate_force(self, kappa: float, F_z_tire: float) -> float:
        F_z = abs(float(F_z_tire))
        C_x = float(self.params.C_x)
        mu = float(self.params.mu)

        if F_z <= 1e-8:
            self.state.longitudinal_force = 0.0
            return 0.0

        kappa = self._sanitize_slip_ratio(kappa)
        k_abs = abs(float(kappa))
        k_sign = np.sign(kappa)

        kappa_sl = (3.0 * mu * F_z) / C_x

        if k_abs <= kappa_sl:
            Fx = k_sign * (
                C_x * k_abs
                - (C_x**2 / (3.0 * mu * F_z)) * (k_abs**2)
                + (C_x**3 / (27.0 * (mu * F_z) ** 2) * (k_abs**3))
            )
        else:
            Fx = k_sign * mu * F_z

        if not np.isfinite(Fx):
            Fx = 0.0
        self.state.longitudinal_force = float(Fx)
        return float(Fx)

    def update(self, dt: float, omega_wheel: float, V_wheel_x: float, F_z_tire: float) -> float:
        del dt
        kappa = self.calculate_slip_ratio(omega_wheel, V_wheel_x)
        Fx = self.calculate_force(kappa, F_z_tire)
        return Fx

    def reset(self) -> None:
        self.state = FialaLongitudinalTireState()

    def get_state(self) -> Dict:
        return {
            "slip_ratio_cmd": self.state.slip_ratio_cmd,
            "slip_ratio": self.state.slip_ratio,
            "longitudinal_force": self.state.longitudinal_force,
            "V_wheel_x": self.state.V_wheel_x,
            "V_wheel": self.state.V_wheel,
            "V_wheel_x_minus_V_wheel": self.state.V_wheel_x_minus_V_wheel,
            "slip_ratio_denom": self.state.slip_ratio_denom,
        }
