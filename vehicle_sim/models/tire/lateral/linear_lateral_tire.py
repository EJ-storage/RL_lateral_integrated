#수정중
#!/bin/python3
"""
횡방향 타이어 동역학 모델 (모든 입력/출력은 휠 로컬 좌표계 기준)
입력: α (slip angle), V_wx, V_wy (휠 속도, 휠 프레임)
출력: F_y^lateral (횡방향 타이어 힘, 휠 프레임)
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from vehicle_sim.utils.config_loader import load_param


@dataclass
class LinearTireParameters:
    """횡방향 타이어 파라미터"""
    C_alpha: float = 0.0   # 코너링 강성 [N/rad]
    mu: float = 0.0            # 마찰계수 [-]
    trail: float = 0.0        # 트레일 길이 [m] (얼라이닝 토크 계산용)


@dataclass
class LinearTireState:
    """횡방향 타이어 상태 변수"""
    slip_angle: float = 0.0        # α: 슬립각 [rad]
    lateral_force: float = 0.0     # F_y [N]
    aligning_torque: float = 0.0   # M_z [N*m]


class LinearLateralTireModel:
    """
    횡방향 타이어 동역학 모델
    입력: α, V_wheel_x, V_wheel_y (휠 프레임 속도)
    출력: F_y^lateral
    """

    def __init__(self, parameters: Optional[LinearTireState] = None,
                 config_path: Optional[str] = None):
        """
        횡방향 타이어 모델 초기화

        Args:
            parameters: 파라미터 객체 (직접 주입 시)
            config_path: Explicit vehicle YAML path or alias from a concrete model wrapper.
        """
        if parameters is not None:
            self.params = parameters
        else:
            tire_param = load_param('tire', config_path)
            lateral_param = tire_param.get('lateral', {})
            self.params = LinearTireParameters(
                C_alpha=float(lateral_param.get('C_alpha', LinearTireParameters.C_alpha)),
                mu=float(
                    lateral_param.get(
                        'mu',
                        tire_param.get('mu', LinearTireParameters.mu),
                    )
                ),
                trail=float(lateral_param.get('trail', LinearTireParameters.trail)),
            )
        self.state = LinearTireState()

    def update(self, V_wheel_x: float, V_wheel_y: float,
               F_tire: float) -> float:
        """
        슬립각 → 횡력/얼라이닝 토크 계산 후 상태 갱신

        Args:
            V_wheel_x: 휠 종방향 속도 [m/s]
            V_wheel_y: 휠 횡방향 속도 [m/s]
            F_tire: 수직하중/언스프렁 힘 [N]

        Returns:
            횡방향 타이어 힘 F_y [N]
        """
        alpha = self.calculate_slip_angle(V_wheel_x, V_wheel_y)
        Fy = self.calculate_force(alpha, F_tire)
        M_z = self.calculate_aligning_torque(alpha, F_tire, Fy_override=Fy)

        # 상태 저장
        self.state.slip_angle = alpha
        self.state.lateral_force = Fy
        self.state.aligning_torque = M_z

        return Fy

    def calculate_slip_angle(self, V_wheel_x: float, V_wheel_y: float) -> float:
        """
        슬립 각도 계산
        입력: V_wheel_x, V_wheel_y (휠 속도, 휠 로컬 프레임)
        출력: α (slip angle). 휠 프레임 기준이므로 steering_angle 보정 없음.
        """
        alpha = np.arctan2(V_wheel_y, V_wheel_x)
        return float(alpha)

    def calculate_force(self, alpha: float, F_tire: float) -> float:
        """
        횡방향 타이어 힘 계산
        입력: α (slip angle), F_tire (언스프렁 힘)
        출력: F_y^lateral
        """
        # F_y = -C_alpha * F_z * alpha
        Fy = -self.params.C_alpha * alpha
        Fy_max = abs(self.params.mu * F_tire)
        Fy_limited = float(np.clip(Fy, -Fy_max, Fy_max))
        return Fy_limited

    def calculate_aligning_torque(self, alpha: float, F_tire: float,
                                  Fy_override: Optional[float] = None) -> float:
        """
        얼라이닝 토크 계산
        입력: α (slip angle), F_tire (언스프렁 힘)
        출력: M_z (얼라이닝 토크, 복원 방향)
        """
        Fy = Fy_override if Fy_override is not None else self.calculate_force(alpha, F_tire)
        # trail*Fy가 조향을 복원하는 방향(슬립 감소)으로 작용하도록 부호 설정
        M_z = self.params.trail * Fy
        return float(M_z)

    def get_state(self) -> Dict:
        """현재 횡타이어 상태 조회"""
        return {
            "slip_angle": self.state.slip_angle,
            "lateral_force": self.state.lateral_force,
            "aligning_torque": self.state.aligning_torque,
        }

    def reset(self) -> None:
        """횡타이어 상태 리셋"""
        self.state = LinearTireState()
