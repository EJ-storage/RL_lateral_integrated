#!/bin/python3
"""
종방향 타이어 동역학 모델
입력: κ (slip ratio), F_z_tire (타이어 수직력)
출력: F_x_tire (종방향 타이어 힘)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from vehicle_sim.utils.config_loader import load_param


@dataclass
class LongitudinalTireParameters:
    """종방향 타이어 파라미터"""
    C_x: float = 10.0       # 종방향 타이어 강성 계수 (정규화) [-]
    mu: float = 0.9         # 마찰 계수 (건조 노면) [-]
    v_min: float = 0.1      # 최소 속도 (슬립 계산용) [m/s]
    R_eff: float = 0.3      # 타이어 유효 반지름 [m]


@dataclass
class LongitudinalTireState:
    """종방향 타이어 상태 변수"""
    slip_ratio: float = 0.0          # κ: 슬립 비율 [-]
    longitudinal_force: float = 0.0  # F_x_tire: 종방향 타이어 힘 [N]

    V_wheel_x: float = 0.0
    V_wheel: float = 0.0
    V_wheel_x_minus_V_wheel: float = 0.0
    slip_ratio_denom: float = 0.0

##################Stbw Variable##################
@dataclass
class StbwLongitudinalTireParameters:
    C_x: float = 231500
    mu: float = 0.9
    v_min: float = 0.1
    R_eff: float = 0.321
    epsilon: float = 0.1

@dataclass
class StbwLongitudinalTireState:
    slip_ratio_cmd: float = 0.0
    slip_ratio: float = 0.0
    longitudinal_force: float = 0.0
    V_wheel_x: float = 0.0
    V_wheel: float = 0.0
    V_wheel_x_minus_V_wheel: float = 0.0
    slip_ratio_denom: float = 0.0
##################################################

class LongitudinalTireModel:
    """
    종방향 타이어 동역학 모델
    입력: κ (slip ratio), F_z_tire (타이어 수직력)
    출력: F_x_tire (종방향 타이어 힘)

    모델:
    F_x_tire = C_x × F_z_tire × κ
    단, |F_x_tire| ≤ μ × F_z_tire  (마찰 한계)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        종방향 타이어 모델 초기화

        Args:
            config_path: YAML 설정 파일 경로. None이면 기본 vehicle_standard.yaml 사용
        """
        # 타이어 파라미터 로드
        tire_param = load_param('tire', config_path)
        long_param = tire_param.get('longitudinal', {})

        # 차량 스펙에서 휠 반지름 로드
        vehicle_spec = load_param('vehicle_spec', config_path)
        wheel_spec = vehicle_spec.get('wheel', {})

        self.params = LongitudinalTireParameters(
            C_x=float(long_param.get('C_x', 10.0)),
            mu=float(long_param.get('mu', tire_param.get('mu', 0.9))),  # 공통 타이어 마찰 계수 fallback
            v_min=float(long_param.get('v_min', 0.1)),
            R_eff=float(wheel_spec.get('R_eff', 0.3))
        )
        self.state = LongitudinalTireState()

    def calculate_slip_ratio(self, omega_wheel: float, V_wheel_x: float) -> float:
        """
        슬립 비율 계산

        입력:
            - omega_wheel: 휠 각속도 [rad/s]
            - V_wheel_x: 휠 중심 종방향 속도 [m/s]

        출력:
            - κ: 슬립 비율 [-]

        공식:
            κ = (V_wheel - V_x) / |V_x|

        여기서:
            - V_wheel = ω × R_eff (휠 둘레 속도)
            - κ > 0: 구동 슬립 (가속)
            - κ < 0: 제동 슬립 (감속)
            - κ = 0: 완전 구름 (no slip)
        """
        # 휠 둘레 속도 (R_eff는 YAML에서 로드됨)
        V_wheel = omega_wheel * self.params.R_eff

        # 저속에서는 분모가 0에 가까워 슬립이 폭주하므로, 분모를 v_min으로 바닥 처리한다.
        # (κ=0으로 고정하면 정지 상태에서 구동/제동 힘이 0이 되어 차량이 출발/정지하지 못함)
        denom = max(abs(V_wheel_x), self.params.v_min)
        kappa = (V_wheel - V_wheel_x) / denom
        self.state.V_wheel_x = float(V_wheel_x)
        self.state.V_wheel = float(V_wheel)
        self.state.V_wheel_x_minus_V_wheel = float(V_wheel_x - V_wheel)
        self.state.slip_ratio_denom = float(denom)

        # 상태 업데이트
        self.state.slip_ratio = kappa

        return kappa

    def calculate_force(self, kappa: float, F_z_tire: float) -> float:
        """
        종방향 타이어 힘 계산

        입력:
            - kappa: 슬립 비율 [-]
            - F_z_tire: 타이어 수직력 [N]

        출력:
            - F_x_tire: 종방향 타이어 힘 [N]

        모델:
            1. 선형 영역: F_x_tire = C_x × F_z_tire × κ
            2. 마찰 한계: |F_x_tire| ≤ μ × F_z_tire

        특징:
            - 수직력에 비례하는 강성 (정규화된 C_x)
            - F_z_tire가 클수록 더 큰 종방향 힘 발생
        """
        # 1. 수직력 비례 모델
        F_x_tire = self.params.C_x * kappa

        # 2. 마찰 한계 적용 (포화)
        F_x_tire_max = self.params.mu * abs(F_z_tire)
        F_x_tire = np.clip(F_x_tire, -F_x_tire_max, F_x_tire_max)

        # 상태 업데이트
        self.state.longitudinal_force = F_x_tire

        return F_x_tire

    def reset(self) -> None:
        """타이어 모델 상태 리셋"""
        self.state = LongitudinalTireState()

    def get_state(self) -> Dict:
        """현재 종방향 타이어 상태 조회"""
        return {
            "slip_ratio": self.state.slip_ratio,
            "longitudinal_force": self.state.longitudinal_force,
            "V_wheel_x": self.state.V_wheel_x,
            "V_wheel": self.state.V_wheel,
            "V_wheel_x_minus_V_wheel": self.state.V_wheel_x_minus_V_wheel,
            "slip_ratio_denom": self.state.slip_ratio_denom,
        }


class StbwLongitudinalTireModel:
    MAX_ABS_SLIP_RATIO = 5.0

    def __init__(self, config_path: Optional[str]=None):
        tire_param = load_param('tire', config_path)
        long_param = tire_param.get('longitudinal', {})

        vehicle_spec = load_param('vehicle_spec', config_path)
        wheel_spec = vehicle_spec.get('wheel', {})

        self.params = StbwLongitudinalTireParameters(
            C_x=float(long_param.get('C_x', 231500)),
            mu = float(long_param.get('mu', tire_param.get('mu', 0.9))),
            v_min=float(long_param.get('v_min', 0.1)),
            R_eff=float(wheel_spec.get('R_eff', 0.3)),
            epsilon = float(long_param.get('epsilon', 0.1))
        )
        self.state = StbwLongitudinalTireState()

    def _sanitize_slip_ratio(self, value: float, fallback: float = 0.0) -> float:
        if not np.isfinite(value):
            return float(fallback)
        return float(np.clip(value, -self.MAX_ABS_SLIP_RATIO, self.MAX_ABS_SLIP_RATIO))

    def calculate_slip_ratio(self, omega_wheel: float, V_wheel_x: float)->float:
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

        kappa_sl = (3.0*mu*F_z) / C_x

        if k_abs <= kappa_sl:
            Fx = k_sign * (
                C_x * k_abs
                - ( C_x**2 / (3.0 * mu * F_z)) * (k_abs**2)
                + (C_x**3 / (27.0 * (mu * F_z)**2) * (k_abs**3))
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
        self.state = StbwLongitudinalTireState()
    
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
