#!/bin/python3
"""
브레이크 액추에이터 모델
입력: T_brk (모터 토크)
출력: F_clamp (클램핑력)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from vehicle_sim.utils.config_loader import load_param


@dataclass
class BrakeParameters:
    """브레이크 파라미터"""
    tau: float = 0.0          # 시간 상수 [s]
    A_eff: float = 0.0     # 캘리퍼 유효 단면적 [m^2]
    V_d: float = 0.0       # 액추에이터/펌프 변위 (per rev) [m^3/rev]
    p_max_bar: float = 0.0  # 시스템 최대 압력 [bar]


@dataclass
class BrakeState:
    """브레이크 상태 변수"""
    clamp_force: float = 0.0  # F_clamp: 클램핑력 [N]


class BrakeModel:
    """
    브레이크 액추에이터 모델
    입력: T_brk (모터 토크 τ_m)
    출력: F_clamp

    유압/메카트로닉 브레이크의 클램핑력 근사식:
        F_clamp = 2*A_eff*p ≈ (4π*A_eff / V_d) * τ_m
    여기서 V_d는 변위(펌프/스크루) [m^3/rev], τ_m은 모터 토크.

    이 클래스는 클램핑력까지만 모델링하고, 실제 브레이크 모멘트 변환
    (μ_pad·R_rotor 곱)은 상위 drive_model에서 수행한다.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        브레이크 모델 초기화

        Args:
            config_path: Explicit vehicle YAML path or alias from a concrete model wrapper.
        """
        brake_param = load_param('brake', config_path)
        self.params = BrakeParameters(
            tau=float(brake_param.get('tau', 0.1)),
            A_eff=float(brake_param.get('A_eff', 1.2e-3)),
            V_d=float(brake_param.get('V_d', 1.0e-7)),
            p_max_bar=float(brake_param.get('p_max_bar', 200.0))
        )

        # 유효 게인 계산 (F_clamp = clamp_gain * τ_m)
        self._clamp_gain = self._compute_clamp_gain()
        self._max_clamp_force = self._compute_max_clamp_force()
        self.state = BrakeState()

    def update(self, dt: float, T_brk: float) -> float:
        """
        브레이크 액추에이터 업데이트 (1차 지연 시스템)
        입력: T_brk (모터 토크 τ_m)
        출력: F_clamp (클램핑력)

        클램핑력: F_clamp = (4π*A_eff / V_d) * τ_m

        1차 지연 미분방정식: τ * dF/dt + F = F_target
        이산화: F_new = F_old + (dt/τ) * (F_target - F_old)
        """
        # 음수 모터 토크는 클램핑력을 생성하지 않는다고 가정 (필요 시 조정)
        tau_m = max(T_brk, 0.0)
        F_raw = self._clamp_gain * tau_m
        F_target = min(F_raw, self._max_clamp_force)

        # 1차 지연 시스템 업데이트
        # F_new = F_old + (dt/tau) * (F_target - F_old)
        dF = (dt / self.params.tau) * (F_target - self.state.clamp_force)
        F_clamp = self.state.clamp_force + dF

        # 상태 업데이트
        self.state.clamp_force = F_clamp

        return F_clamp

    def _compute_clamp_gain(self) -> float:
        """F_clamp = clamp_gain * τ_m 계산을 위한 게인"""
        if self.params.A_eff <= 0.0:
            raise ValueError("Brake parameter A_eff must be positive.")
        if self.params.V_d <= 0.0:
            raise ValueError("Brake parameter V_d must be positive.")
        return (4.0 * np.pi * self.params.A_eff) / self.params.V_d

    def _compute_max_clamp_force(self) -> float:
        """
        최대 압력 기반 클램핑력 포화값 계산.
        입력 p_max_bar를 Pa로 변환 후 F_max = 2 * A_eff * p_max
        """
        if self.params.p_max_bar <= 0.0:
            return float("inf")
        p_max_pa = self.params.p_max_bar * 1e5  # [Pa]
        return 2.0 * self.params.A_eff * p_max_pa

    def get_state(self) -> Dict:
        """현재 브레이크 상태 조회"""
        return {
            'clamp_force': self.state.clamp_force
        }

    def reset(self) -> None:
        """브레이크 상태 리셋"""
        self.state = BrakeState()


