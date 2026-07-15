"""
Active Anti-Roll Bar (AARB) Controller
좌우 서스펜션 스트로크 차(Δs)를 입력으로 받아 anti-roll 토크를 생성하는 패시브 ARB 철학 기반 제어기
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ActiveAntiRollBarGains:
    """Active Anti-Roll Bar PD 게인 (축별)"""
    # Front ARB
    k_arb_front: float = 500000.0    # 전륜 ARB 강성 [N/m]
    c_arb_front: float = 2000.0     # 전륜 ARB 댐핑 [N*s/m]

    # Rear ARB
    k_arb_rear: float = 400000.0     # 후륜 ARB 강성 [N/m]
    c_arb_rear: float = 1500.0      # 후륜 ARB 댐핑 [N*s/m]

    # Track width (좌우 간격)
    track_width: float = 1.634      # 트랙 폭 [m]

    # 힘 제한
    max_force: Optional[float] = None  # 최대 액티브 힘 [N] (None이면 서스펜션 모델 제한 사용)


class ActiveAntiRollBarController:
    """
    Active Anti-Roll Bar (AARB) Controller

    패시브 스태빌라이저 바 철학을 따라, 좌우 서스펜션 스트로크 차이를
    입력으로 받아 anti-roll 모멘트를 생성하고 좌우에 힘으로 분배.

    제어 법칙 (축별):
        Δs_front = delta_s_FR - delta_s_FL
        Δs_rear = delta_s_RR - delta_s_RL

        M_arb_front = k_arb_front * Δs_front + c_arb_front * Δs_front_dot
        M_arb_rear = k_arb_rear * Δs_rear + c_arb_rear * Δs_rear_dot

    좌우 힘 분배:
        F_R = +M_arb / track_width
        F_L = -M_arb / track_width

    입력: 각 코너의 delta_s, delta_s_dot
    출력: 코너별 F_arb [N] 또는 T_susp [N*m]
    """

    def __init__(self, gains: Optional[ActiveAntiRollBarGains] = None):
        """
        Active Anti-Roll Bar Controller 초기화

        Args:
            gains: ARB 게인 설정. None이면 기본값 사용
        """
        self.gains = gains if gains is not None else ActiveAntiRollBarGains()

        # 내부 상태 (진단용)
        self.M_arb_front: float = 0.0  # 전륜 ARB 모멘트 [N*m]
        self.M_arb_rear: float = 0.0   # 후륜 ARB 모멘트 [N*m]
        self.delta_s_front: float = 0.0  # 전륜 스트로크 차 [m]
        self.delta_s_rear: float = 0.0   # 후륜 스트로크 차 [m]

    def reset(self) -> None:
        """제어기 상태 리셋"""
        self.M_arb_front = 0.0
        self.M_arb_rear = 0.0
        self.delta_s_front = 0.0
        self.delta_s_rear = 0.0

    def update(
        self,
        delta_s: Dict[str, float],
        delta_s_dot: Dict[str, float],
        susp_models: Optional[Dict] = None,
        output_type: str = "torque",
    ) -> Dict[str, float]:
        """
        좌우 스트로크 차 기반 ARB 힘/토크 계산

        Args:
            delta_s: 각 코너의 서스펜션 스트로크 {corner: delta_s [m]}
                     {"FL": ..., "FR": ..., "RL": ..., "RR": ...}
            delta_s_dot: 각 코너의 스트로크 속도 {corner: delta_s_dot [m/s]}
            susp_models: 서스펜션 모델 딕셔너리 (토크 변환용, 선택)
            output_type: "force" 또는 "torque" (기본: "torque")

        Returns:
            output: 코너별 힘 또는 토크
                    {"FL": value, "FR": value, "RL": value, "RR": value}
                    - output_type="force": [N]
                    - output_type="torque": [N*m]
        """
        # 1. 좌우 스트로크 차 계산
        # Δs_front = delta_s_FR - delta_s_FL (오른쪽이 더 압축되면 양수)
        # Δs_rear = delta_s_RR - delta_s_RL
        self.delta_s_front = delta_s["FL"] - delta_s["FR"]
        self.delta_s_rear = delta_s["RL"] - delta_s["RR"]

        delta_s_dot_front = delta_s_dot["FL"] - delta_s_dot["FR"]
        delta_s_dot_rear = delta_s_dot["RL"] - delta_s_dot["RR"]

        # 2. ARB 모멘트 계산 (PD 제어)
        # M_arb = k * Δs + c * Δs_dot
        self.M_arb_front = (
            self.gains.k_arb_front * self.delta_s_front
            + self.gains.c_arb_front * delta_s_dot_front
        )
        self.M_arb_rear = (
            self.gains.k_arb_rear * self.delta_s_rear
            + self.gains.c_arb_rear * delta_s_dot_rear
        )

        # 3. 좌우 힘 분배
        # F_R = +M_arb / track (오른쪽이 압축되면 오른쪽을 밀어 올림)
        # F_L = -M_arb / track (왼쪽을 눌러서 균형 유지)
        track = self.gains.track_width

        F_arb = {
            "FL": -self.M_arb_front / track,
            "FR": +self.M_arb_front / track,
            "RL": -self.M_arb_rear / track,
            "RR": +self.M_arb_rear / track,
        }

        # 4. 힘 제한 (선택)
        max_force = self.gains.max_force
        if max_force is not None:
            for corner in F_arb:
                F_arb[corner] = float(np.clip(F_arb[corner], -max_force, max_force))

        # 5. 출력 타입에 따라 변환
        if output_type == "force":
            return F_arb
        elif output_type == "torque":
            T_susp = {}
            for corner, F in F_arb.items():
                if susp_models is not None and corner in susp_models:
                    susp_params = susp_models[corner].params
                    T_susp[corner] = self._force_to_torque(F, susp_params)
                else:
                    T_susp[corner] = self._force_to_torque_default(F)
            return T_susp
        else:
            raise ValueError(f"Invalid output_type: {output_type}. Must be 'force' or 'torque'.")

    def _force_to_torque(self, F_act: float, susp_params) -> float:
        """
        액티브 힘을 서스펜션 액추에이터 토크로 변환

        Args:
            F_act: 액티브 힘 [N]
            susp_params: 서스펜션 파라미터 (lead, efficiency, F_active_max)

        Returns:
            T_susp: 서스펜션 토크 [N*m]
        """
        # T = F * lead / (2π * efficiency)
        lead = susp_params.lead if hasattr(susp_params, 'lead') else 0.01
        efficiency = susp_params.efficiency if hasattr(susp_params, 'efficiency') else 0.9

        T_susp = F_act * lead / (2.0 * np.pi * efficiency)
        return float(T_susp)

    def _force_to_torque_default(self, F_act: float) -> float:
        """
        기본 파라미터로 힘을 토크로 변환

        Args:
            F_act: 액티브 힘 [N]

        Returns:
            T_susp: 서스펜션 토크 [N*m]
        """
        # 기본값: lead=0.01 m/rev, efficiency=0.9
        lead = 0.01
        efficiency = 0.9
        T_susp = F_act * lead / (2.0 * np.pi * efficiency)
        return float(T_susp)

    def get_state(self) -> Dict[str, float]:
        """
        현재 제어기 상태 조회

        Returns:
            state: {
                "M_arb_front": 전륜 ARB 모멘트 [N*m],
                "M_arb_rear": 후륜 ARB 모멘트 [N*m],
                "delta_s_front": 전륜 스트로크 차 [m],
                "delta_s_rear": 후륜 스트로크 차 [m]
            }
        """
        return {
            "M_arb_front": self.M_arb_front,
            "M_arb_rear": self.M_arb_rear,
            "delta_s_front": self.delta_s_front,
            "delta_s_rear": self.delta_s_rear,
        }
