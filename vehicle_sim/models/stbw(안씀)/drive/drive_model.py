#!/bin/python3
"""
휠 동역학 모델
입력: T_Drv (구동 토크), F_clamp (브레이크 클램핑력), F_x (종방향 타이어 힘)
출력: ω_wheel (휠 각속도)
--> max_wheel_speed 기본값 60.63 rad/s ≈ 579 rpm, R_eff=0.316m일 때 선속도 ≈ 68.9 km/h
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from vehicle_sim.utils.config_loader import load_param


@dataclass
class DriveParameters:
    """휠 동역학 파라미터"""
    J_wheel: float = 0.5   # 휠 관성 모멘트 [kg·m²]
    R_wheel: float = 0.3   # 휠 반지름 [m]
    B_wheel: float = 0.0   # 휠 점성 마찰 계수 [N·m·s/rad] (ω에 비례하는 저항 토크)
    max_wheel_speed: float = 60.63  # 최대 휠 각속도 [rad/s] (config로 주입)


@dataclass
class BrakeTorqueParameters:
    """브레이크 모멘트 환산 파라미터 (클램핑력 → 모멘트)"""
    mu_pad: float = 0.35      # 패드-디스크 마찰계수 [-]
    R_rotor: float = 0.12     # 브레이크 유효 반경 [m]


@dataclass
class DriveState:
    """휠 동역학 상태 변수"""
    wheel_speed: float = 0.0  # ω_wheel: 휠 각속도 [rad/s]


##################Stbw Variable##################
@dataclass
class StbwDriveParameters:
    J_wheel: float = 1.781
    R_wheel: float = 0.321
    B_wheel: float = 0.34
    max_wheel_speed: float = 60.63

@dataclass
class StbwBrakeTorqueParameters:
    mu_pad: float = 0.35
    R_rotor: float = 0.12

@dataclass
class StbwDriveState:
    wheel_speed: float = 0.0
##################################################


class DriveModel:
    """
    휠 동역학 모델
    입력: T_Drv, M_brk, F_x (이전 스텝), direction (전진/후진)
    출력: ω_wheel

    회전 운동 방정식:
    J_wheel * dω/dt = T_Drv - R_wheel * F_x - M_brk - B_wheel * ω

    방향 정의:
    - T_Drv: direction=1 (전진) → 양수, direction=-1 (후진) → 음수
    - M_brk: ω > 0 → 음수 (감속), ω < 0 → 양수 (감속)

    속도 제한:
    - 최대 휠 각속도: ±60.63 rad/s (≈579 rpm, R_eff=0.316 m 기준 선속도 ≈68.9 km/h)
    """

    def __init__(self, config_path: Optional[str] = None, corner_id: Optional[str] = None):
        """
        휠 동역학 모델 초기화

        Args:
            config_path: YAML 설정 파일 경로. None이면 기본 vehicle_standard.yaml 사용
            corner_id: 코너 ID ('FL', 'FR', 'RL', 'RR'). B_wheel 선택에 사용.
        """
        # 차량 스펙에서 휠 파라미터 로드
        vehicle_spec = load_param('vehicle_spec', config_path)
        wheel_spec = vehicle_spec.get('wheel', {})

        # 드라이브 파라미터 로드
        drive_param = load_param('drive', config_path)
        brake_param = load_param('brake', config_path)

        # J_wheel, B_wheel: corner_id에 따라 front/rear 선택
        if corner_id and corner_id[0] == 'F':  # Front wheels (FL, FR)
            J_wheel = float(wheel_spec.get('J_wheel_front', wheel_spec.get('J_wheel', 0.5)))
            B_wheel = float(wheel_spec.get('B_wheel_front', 0.1))
        elif corner_id and corner_id[0] == 'R':  # Rear wheels (RL, RR)
            J_wheel = float(wheel_spec.get('J_wheel_rear', wheel_spec.get('J_wheel', 0.5)))
            B_wheel = float(wheel_spec.get('B_wheel_rear', 0.34))
        else:  # Fallback to legacy single value
            J_wheel = float(wheel_spec.get('J_wheel', 0.5))
            B_wheel = float(wheel_spec.get('B_wheel', 0.0))

        self.params = DriveParameters(
            J_wheel=J_wheel,
            R_wheel=float(wheel_spec.get('R_eff', 0.3)),
            B_wheel=max(B_wheel, 0.0),
            max_wheel_speed=float(drive_param.get('max_wheel_speed', 60.63))
        )
        self.brake_params = BrakeTorqueParameters(
            mu_pad=float(brake_param.get('mu_pad', 0.35)),
            R_rotor=float(brake_param.get('R_rotor', 0.12)),
        )
        self._clamp_to_torque = self._compute_clamp_to_torque_gain()
        self.state = DriveState()

    def update(self, dt: float, T_Drv: float, F_x: float,
               F_clamp: float = 0.0, M_brk_signed: Optional[float] = None,
               direction: int = 1) -> float:
        """
        휠 동역학 업데이트

        입력:
            - T_Drv: 구동 토크 크기 (절댓값) [N·m]
            - F_x: 종방향 타이어 힘 (이전 스텝 값) [N]
            - F_clamp: 브레이크 클램핑력 (절댓값) [N] (M_brk_signed가 None일 때 사용)
            - M_brk_signed: 브레이크 토크 (부호 포함) [N·m] (우선순위 높음, None이면 F_clamp 사용)
            - direction: 전진/후진 방향 (1: 전진, -1: 후진)

        출력:
            - ω_wheel: 휠 각속도 [rad/s]

        브레이크 모멘트 처리:
        - M_brk_signed가 제공되면 직접 사용 (CarMaker 데이터 등)
        - M_brk_signed가 None이면 F_clamp로부터 계산 및 부호 처리

        회전 운동 방정식:
            J * dω/dt = T_Drv_signed - R*F_x + M_brk_signed - B*ω

        방향 처리:
        - T_Drv_signed = direction × T_Drv
        - M_brk_signed: 휠 속도 반대 방향 (음수)

        이산화:
        dω = (dt/J) * (T_Drv_signed - R*F_x + M_brk_signed - B*ω_old)
        ω_new = ω_old + dω

        속도 제한:
        ω_new = clip(ω_new, -60.63, 60.63) [rad/s]
        """
        # 1. 방향을 고려한 구동 토크
        T_Drv_signed = direction * T_Drv

        # 2. 현재 휠 속도
        omega = self.state.wheel_speed

        # 3. 브레이크 모멘트 계산
        if M_brk_signed is not None:
            # 직접 브레이크 토크 사용 (CarMaker 데이터 등)
            pass  # M_brk_signed 그대로 사용
        else:
            # 클램핑력 → 브레이크 모멘트 환산
            F_clamp_eff = max(F_clamp, 0.0)  # 음수 입력은 무효 처리
            M_brk = self._clamp_to_torque * F_clamp_eff

            # 브레이크 토크는 휠 속도 반대 방향으로 작용
            omega0 = 0.7  # [rad/s] 0.2~1.0 권장
            M_brk_signed = -M_brk * np.tanh(omega / omega0)

        # 4. 순 토크 계산
        # 휠 점성 마찰 토크:
        #   M_visc = B*ω (ω와 같은 부호)
        #   순토크에는 -M_visc로 반영되어 결과적으로 -B*ω (항상 휠 속도 반대 방향)
        M_visc = self.params.B_wheel * omega
        T_net = T_Drv_signed - self.params.R_wheel * F_x + M_brk_signed - M_visc

        # 5. 각가속도 계산
        alpha = T_net / self.params.J_wheel

        # 6. 각속도 업데이트 (오일러 적분)
        omega_new = omega + alpha * dt

        # 7. 최대 속도 제한 적용
        omega_limited = self.apply_speed_limits(omega_new)

        # 8. 상태 업데이트
        self.state.wheel_speed = omega_limited

        return omega_limited

    def apply_speed_limits(self, omega: float) -> float:
        """휠 속도 제한 적용 (65.535 km/h = 60.63 rad/s)"""
        return float(np.clip(omega, -self.params.max_wheel_speed, self.params.max_wheel_speed))

    def get_state(self) -> Dict:
        """현재 휠 동역학 상태 조회"""
        return {
            'wheel_speed': self.state.wheel_speed
        }

    def reset(self) -> None:
        """휠 동역학 상태 리셋"""
        self.state = DriveState()

    def _compute_clamp_to_torque_gain(self) -> float:
        """클램핑력 → 브레이크 모멘트 환산 게인"""
        if self.brake_params.R_rotor <= 0.0:
            raise ValueError("Brake parameter R_rotor must be positive.")

        mu = max(self.brake_params.mu_pad, 0.0)
        return mu * self.brake_params.R_rotor


class StbwDriveModel:
    """
    휠 동역학 모델
    입력: T_Drv, M_brk, F_x (이전 스텝), direction (전진/후진)
    출력: ω_wheel

    회전 운동 방정식:
    J_wheel * dω/dt = T_Drv - R_wheel * F_x - M_brk - B_wheel * ω

    방향 정의:
    - T_Drv: direction=1 (전진) → 양수, direction=-1 (후진) → 음수
    - M_brk: ω > 0 → 음수 (감속), ω < 0 → 양수 (감속)

    속도 제한:
    - 최대 휠 각속도: ±60.63 rad/s (≈579 rpm, R_eff=0.316 m 기준 선속도 ≈68.9 km/h)
    """

    def __init__(self, config_path: Optional[str]=None, axle_id: Optional[str]=None):
        """
        휠 동역학 모델 초기화

        Args:
            config_path: YAML 설정 파일 경로. Stbw로 선택
            axle_id: 축 ID ('F', 'R'). B_wheel 선택에 사용.
        """ 

        vehicle_spec = load_param('vehicle_spec', config_path)
        wheel_spec = vehicle_spec.get('wheel', {})

        drive_param = load_param('drive', config_path)
        brake_param = load_param('brake', config_path)


        if axle_id == 'F':
            J_wheel = float(wheel_spec.get('J_wheel_front', 1.781))
            B_wheel = float(wheel_spec.get('B_wheel_front', 0.34))
        elif axle_id == 'R':
            J_wheel = float(wheel_spec.get('J_wheel_rear', 1.781))
            B_wheel = float(wheel_spec.get('B_wheel_rear', 0.34))
        else:
            J_wheel = float(wheel_spec.get('J_wheel', 0.178))
            B_wheel = float(wheel_spec.get('B_wheel', 0.34))

        self.params = StbwDriveParameters(
            J_wheel = J_wheel,
            R_wheel = float(wheel_spec.get('R_eff', 0.321)),
            B_wheel = max(B_wheel, 0.34),
            max_wheel_speed=float(drive_param.get('max_wheel_speed', 60.63))
        )

        self.brake_params = StbwBrakeTorqueParameters(
            mu_pad=float(brake_param.get('mu_pad', 0.35)),
            R_rotor=float(brake_param.get('R_rotor', 0.12))
        )
        self._clamp_to_torque = self._compute_clamp_to_torque_gain()
        self.state = StbwDriveState()

    def update(self, dt: float, T_Drv: float, F_x: float,
               F_clamp: float = 0.0, M_brk_signed: Optional[float]=None,
               direction: int = 1)->float:
        """
        휠 동역학 업데이트

        입력:
            - T_Drv: 구동 토크 크기 (절댓값) [N·m]
            - F_x: 종방향 타이어 힘 (이전 스텝 값) [N]
            - F_clamp: 브레이크 클램핑력 (절댓값) [N] (M_brk_signed가 None일 때 사용)
            - M_brk_signed: 브레이크 토크 (부호 포함) [N·m] (우선순위 높음, None이면 F_clamp 사용)
            - direction: 전진/후진 방향 (1: 전진, -1: 후진)

        출력:
            - ω_wheel: 휠 각속도 [rad/s]

        브레이크 모멘트 처리:
        - M_brk_signed가 제공되면 직접 사용 (CarMaker 데이터 등)
        - M_brk_signed가 None이면 F_clamp로부터 계산 및 부호 처리

        회전 운동 방정식:
            J * dω/dt = T_Drv_signed - R*F_x + M_brk_signed - B*ω

        방향 처리:
        - T_Drv_signed = direction × T_Drv
        - M_brk_signed: 휠 속도 반대 방향 (음수)

        이산화:
        dω = (dt/J) * (T_Drv_signed - R*F_x + M_brk_signed - B*ω_old)
        ω_new = ω_old + dω

        속도 제한:
        ω_new = clip(ω_new, -60.63, 60.63) [rad/s]
        """
                
        T_Drv_signed = direction * T_Drv

        omega = self.state.wheel_speed

        if M_brk_signed is not None:
            pass
        else:
            F_clamp_eff = max(F_clamp, 0.0)
            M_brk = self._clamp_to_torque * F_clamp_eff

            omega0 = 0.7
            M_brk_signed = -M_brk * np.tanh(omega / omega0)

        M_visc = self.params.B_wheel * omega
        T_net = T_Drv_signed - self.params.R_wheel * F_x + M_brk_signed - M_visc

        alpha = T_net/self.params.J_wheel

        omega_new = omega + alpha * dt

        omega_limited = self.apply_speed_limits(omega_new)

        self.state.wheel_speed = omega_limited

        return omega_limited
    

    def apply_speed_limits(self, omega:float)->float:
        return float(np.clip(omega, -self.params.max_wheel_speed, self.params.max_wheel_speed))

    def get_state(self) -> Dict:
        """현재 휠 동역학 상태 조회"""
        return {
            'wheel_speed': self.state.wheel_speed
        }

    def reset(self) -> None:
        """휠 동역학 상태 리셋"""
        self.state = StbwDriveState()

    def _compute_clamp_to_torque_gain(self) -> float:
        """클램핑력 → 브레이크 모멘트 환산 게인"""
        if self.brake_params.R_rotor <= 0.0:
            raise ValueError("Brake parameter R_rotor must be positive.")

        mu = max(self.brake_params.mu_pad, 0.0)
        return mu * self.brake_params.R_rotor
