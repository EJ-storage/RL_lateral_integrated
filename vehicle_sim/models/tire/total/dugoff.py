#!/bin/python3
"""Shared Modified Dugoff combined-slip tire model.

This module owns only the tire math. Concrete vehicle packages should load
their own configuration and pass ``ModifiedDugoff4WTireParameters`` explicitly.
All parameter defaults are zero so an unconfigured shared model produces no
tire force instead of silently using vehicle-specific constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


WHEEL_LABELS = ("FL", "FR", "RL", "RR")


@dataclass
class ModifiedDugoff4WTireParameters:
    Re_FL: float = 0.0
    Re_FR: float = 0.0
    Re_RL: float = 0.0
    Re_RR: float = 0.0

    Ckappa_FL: float = 0.0
    Ckappa_FR: float = 0.0
    Ckappa_RL: float = 0.0
    Ckappa_RR: float = 0.0

    Calpha_FL: float = 0.0
    Calpha_FR: float = 0.0
    Calpha_RL: float = 0.0
    Calpha_RR: float = 0.0

    muX_FL: float = 0.0
    muX_FR: float = 0.0
    muX_RL: float = 0.0
    muX_RR: float = 0.0

    muY_FL: float = 0.0
    muY_FR: float = 0.0
    muY_RL: float = 0.0
    muY_RR: float = 0.0

    Veps: float = 0.0
    FzMin: float = 0.0
    kappaMin: float = 0.0
    kappaMax: float = 0.0
    alphaMax: float = 0.0


@dataclass
class ModifiedDugoffWheelState:
    fx: float = 0.0
    fy: float = 0.0
    fx_linear: float = 0.0
    fy_linear: float = 0.0
    fx_dugoff: float = 0.0
    fy_dugoff: float = 0.0
    kappa: float = 0.0
    alpha: float = 0.0
    delta_v: float = 0.0
    wheel_linear_speed: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    fz: float = 0.0
    fx_limit: float = 0.0
    fy_limit: float = 0.0
    combined_demand: float = 0.0
    dugoff_scale: float = 1.0
    ellipse_value: float = 0.0
    ellipse_scale: float = 1.0


@dataclass
class ModifiedDugoff4WTireState:
    FL: ModifiedDugoffWheelState
    FR: ModifiedDugoffWheelState
    RL: ModifiedDugoffWheelState
    RR: ModifiedDugoffWheelState

    @classmethod
    def zero(cls) -> "ModifiedDugoff4WTireState":
        return cls(
            FL=ModifiedDugoffWheelState(),
            FR=ModifiedDugoffWheelState(),
            RL=ModifiedDugoffWheelState(),
            RR=ModifiedDugoffWheelState(),
        )


class ModifiedDugoff4WTireModel:
    """Four-wheel Modified Dugoff tire model with optional one-wheel updates."""

    def __init__(
        self,
        parameters: Optional[ModifiedDugoff4WTireParameters] = None,
        config_path: Optional[str] = None,
    ):
        del config_path
        self.params = (
            parameters
            if parameters is not None
            else ModifiedDugoff4WTireParameters()
        )
        self.state = ModifiedDugoff4WTireState.zero()

    @staticmethod
    def _safe_float(value: float, fallback: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        if not np.isfinite(result):
            return float(fallback)
        return result

    @staticmethod
    def _validate_wheel_label(wheel_label: str) -> str:
        label = str(wheel_label).upper()
        if label not in WHEEL_LABELS:
            raise ValueError(
                f"Invalid wheel_label {wheel_label!r}; expected one of {WHEEL_LABELS}."
            )
        return label

    def _wheel_parameters(
        self,
        wheel_label: str,
        road_mu: Optional[float] = None,
    ) -> Tuple[float, float, float, float, float]:
        label = self._validate_wheel_label(wheel_label)
        params = self.params
        re = getattr(params, f"Re_{label}")
        ckappa = getattr(params, f"Ckappa_{label}")
        calpha = getattr(params, f"Calpha_{label}")
        mux = getattr(params, f"muX_{label}")
        muy = getattr(params, f"muY_{label}")

        if road_mu is not None and np.isfinite(float(road_mu)):
            mux = float(road_mu)
            muy = float(road_mu)

        return float(re), float(ckappa), float(calpha), float(mux), float(muy)

    def _calculate_tire(
        self,
        wheelspeed: float,
        vx: float,
        vy: float,
        fz: float,
        re: float,
        ckappa: float,
        calpha: float,
        mux: float,
        muy: float,
    ) -> ModifiedDugoffWheelState:
        wheelspeed = self._safe_float(wheelspeed)
        vx = self._safe_float(vx)
        vy = self._safe_float(vy)
        fz = self._safe_float(fz)

        re = max(self._safe_float(re), 0.0)
       # print(re)
        ckappa = max(self._safe_float(ckappa), 0.0)
        calpha = max(self._safe_float(calpha), 0.0)
        mux = max(self._safe_float(mux), 0.0)
        muy = max(self._safe_float(muy), 0.0)

        veps = max(self._safe_float(self.params.Veps), 1.0e-9)
        fz_min = max(self._safe_float(self.params.FzMin), 0.0)
        kappa_min = self._safe_float(self.params.kappaMin)
        kappa_max = self._safe_float(self.params.kappaMax)
        alpha_max = max(self._safe_float(self.params.alphaMax), 0.0)

        fz_effective = max(fz, 0.0)
        wheel_linear_speed = re * wheelspeed
        delta_v = wheel_linear_speed - vx
        fx_limit = mux * fz_effective
        fy_limit = muy * fz_effective

        if (
            fz_effective <= fz_min
            or ckappa <= 0.0
            or calpha <= 0.0
            or mux <= 0.0
            or muy <= 0.0
            or re <= 0.0
        ):
            return ModifiedDugoffWheelState(
                delta_v=float(delta_v),
                wheel_linear_speed=float(wheel_linear_speed),
                vx=float(vx),
                vy=float(vy),
                fz=float(fz_effective),
                fx_limit=float(fx_limit),
                fy_limit=float(fy_limit),
            )

        vref = max(abs(vx), veps)
        kappa = delta_v / vref
        if kappa_min < kappa_max:
            kappa = float(np.clip(kappa, kappa_min, kappa_max))
        else:
            kappa = float(kappa)

        alpha = float(np.arctan2(vy, vref))
        if alpha_max > 0.0:
            alpha = float(np.clip(alpha, -alpha_max, alpha_max))

        one_plus_kappa = max(1.0 + kappa, 1.0e-9)
        sigma_x = kappa / one_plus_kappa
        sigma_y = np.tan(alpha) / one_plus_kappa

        fx_linear = ckappa * sigma_x
        fy_linear = -calpha * sigma_y
        fx_limit_safe = max(fx_limit, 1.0e-9)
        fy_limit_safe = max(fy_limit, 1.0e-9)

        combined_demand = float(
            np.sqrt(
                (fx_linear / fx_limit_safe) ** 2
                + (fy_linear / fy_limit_safe) ** 2
            )
        )

        if combined_demand <= 1.0e-12:
            dugoff_scale = 1.0
        else:
            lambda_dugoff = 1.0 / (2.0 * combined_demand)
            dugoff_scale = (
                1.0
                if lambda_dugoff >= 1.0
                else lambda_dugoff * (2.0 - lambda_dugoff)
            )

        fx_dugoff = dugoff_scale * fx_linear
        fy_dugoff = dugoff_scale * fy_linear

        ellipse_value = float(
            np.sqrt(
                (fx_dugoff / fx_limit_safe) ** 2
                + (fy_dugoff / fy_limit_safe) ** 2
            )
        )
        ellipse_scale = 1.0 / ellipse_value if ellipse_value > 1.0 else 1.0

        fx = ellipse_scale * fx_dugoff
        fy = ellipse_scale * fy_dugoff
        if not np.isfinite(fx):
            fx = 0.0
        if not np.isfinite(fy):
            fy = 0.0

        return ModifiedDugoffWheelState(
            fx=float(fx),
            fy=float(fy),
            fx_linear=float(fx_linear),
            fy_linear=float(fy_linear),
            fx_dugoff=float(fx_dugoff),
            fy_dugoff=float(fy_dugoff),
            kappa=float(kappa),
            alpha=float(alpha),
            delta_v=float(delta_v),
            wheel_linear_speed=float(wheel_linear_speed),
            vx=float(vx),
            vy=float(vy),
            fz=float(fz_effective),
            fx_limit=float(fx_limit),
            fy_limit=float(fy_limit),
            combined_demand=float(combined_demand),
            dugoff_scale=float(dugoff_scale),
            ellipse_value=float(ellipse_value),
            ellipse_scale=float(ellipse_scale),
        )

    def calculate_wheel(
        self,
        wheel_label: str,
        wheelspeed: float,
        vx: float,
        vy: float,
        fz: float,
        road_mu: Optional[float] = None,
    ) -> ModifiedDugoffWheelState:
        re, ckappa, calpha, mux, muy = self._wheel_parameters(wheel_label, road_mu)
        return self._calculate_tire(
            wheelspeed=wheelspeed,
            vx=vx,
            vy=vy,
            fz=fz,
            re=re,
            ckappa=ckappa,
            calpha=calpha,
            mux=mux,
            muy=muy,
        )

    def update_wheel(
        self,
        dt: float,
        wheel_label: str,
        wheelspeed: float,
        vx: float,
        vy: float,
        fz: float,
        road_mu: Optional[float] = None,
    ) -> Tuple[float, float]:
        del dt
        label = self._validate_wheel_label(wheel_label)
        wheel_state = self.calculate_wheel(label, wheelspeed, vx, vy, fz, road_mu)
        setattr(self.state, label, wheel_state)
        return wheel_state.fx, wheel_state.fy

    def update(
        self,
        dt: float,
        wheelspeed_FL: float,
        wheelspeed_FR: float,
        wheelspeed_RL: float,
        wheelspeed_RR: float,
        vx_FL: float,
        vx_FR: float,
        vx_RL: float,
        vx_RR: float,
        vy_FL: float,
        vy_FR: float,
        vy_RL: float,
        vy_RR: float,
        fz_FL: float,
        fz_FR: float,
        fz_RL: float,
        fz_RR: float,
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        del dt
        inputs = {
            "FL": (wheelspeed_FL, vx_FL, vy_FL, fz_FL),
            "FR": (wheelspeed_FR, vx_FR, vy_FR, fz_FR),
            "RL": (wheelspeed_RL, vx_RL, vy_RL, fz_RL),
            "RR": (wheelspeed_RR, vx_RR, vy_RR, fz_RR),
        }
        for label, wheel_input in inputs.items():
            setattr(self.state, label, self.calculate_wheel(label, *wheel_input))

        return (
            self.state.FL.fx,
            self.state.FR.fx,
            self.state.RL.fx,
            self.state.RR.fx,
            self.state.FL.fy,
            self.state.FR.fy,
            self.state.RL.fy,
            self.state.RR.fy,
        )

    def reset(self) -> None:
        self.state = ModifiedDugoff4WTireState.zero()

    def get_wheel_state(self, wheel_label: str) -> Dict[str, float]:
        wheel_state = getattr(self.state, self._validate_wheel_label(wheel_label))
        return dict(wheel_state.__dict__)

    def get_state(self) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for label in WHEEL_LABELS:
            wheel_state = getattr(self.state, label)
            result[f"fx_{label}"] = wheel_state.fx
            result[f"fy_{label}"] = wheel_state.fy
            result[f"fx_linear_{label}"] = wheel_state.fx_linear
            result[f"fy_linear_{label}"] = wheel_state.fy_linear
            result[f"fx_dugoff_{label}"] = wheel_state.fx_dugoff
            result[f"fy_dugoff_{label}"] = wheel_state.fy_dugoff
            result[f"kappa_{label}"] = wheel_state.kappa
            result[f"alpha_{label}"] = wheel_state.alpha
            result[f"delta_v_{label}"] = wheel_state.delta_v
            result[f"dugoff_scale_{label}"] = wheel_state.dugoff_scale
            result[f"ellipse_value_{label}"] = wheel_state.ellipse_value
            result[f"ellipse_scale_{label}"] = wheel_state.ellipse_scale
        return result


ModifiedDugoffTireModel = ModifiedDugoff4WTireModel


__all__ = [
    "ModifiedDugoff4WTireModel",
    "ModifiedDugoff4WTireParameters",
    "ModifiedDugoff4WTireState",
    "ModifiedDugoffTireModel",
    "ModifiedDugoffWheelState",
]
