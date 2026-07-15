#!/bin/python3
"""
서스펜션 모델 (한 바퀴) - 7-DOF 기반

입력: T_susp, roll, pitch, heave_dev, roll_dot, pitch_dot, heave_dot, z_road
출력: F_sus (차체로), F_z_tire (타이어 수직력)

좌표계:
- heave_dev: 편차 좌표 (입력)
- z_CG = z_CG0 + heave_dev: 절대 좌표 (내부 계산)
- z_u_abs: 휠 중심 절대 높이 (상태 변수, 동역학)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from vehicle_sim.utils.config_loader import load_param


@dataclass
class SuspensionParameters:
    """서스펜션 파라미터"""
    # 기본 강성/댐핑
    K_spring: float = 16000.0   # 스프링 강성 [N/m]
    C_damper_compression: float = 1000.0    # 댐퍼 압축 감쇠 계수 (Push) [N*s/m]
    C_damper_rebound: float = 1000.0        # 댐퍼 리바운드 감쇠 계수 (Pull) [N*s/m]

    # 절대 좌표 기준점
    z_CG0: float = 0.5          # 평형 시 CG 절대 높이 [m]
    L_s0: float = 0.2           # 서스펜션 레스트 길이 [m]
    R_w: float = 0.3            # 타이어 반지름 [m]

    # 스트로크 한계 (위치 제한 방식)
    delta_s_min: float = -0.08  # 최대 압축 한계 [m]
    delta_s_max: float = 0.05   # 최대 신장 한계 [m]

    # 액티브 서스펜션
    lead: float = 0.01          # 볼 스크류 리드 [m/rev]
    efficiency: float = 0.9     # 기계적 효율 [-]
    F_active_max: Optional[float] = 4000.0  # 액티브 힘 포화 한계 [N] (없애려면 None)



@dataclass
class TireVerticalParameters:
    """타이어 수직 파라미터"""
    K_t: float = 200000.0       # 타이어 수직 강성 [N/m]
    C_t: float = 500.0          # 타이어 수직 댐핑 [N*s/m]
    delta_t_max: float = 0.05   # 타이어 최대 압축 한계 [m] (상태 clamp)
    K_hard: float = 1200000.0   # 최대 압축 초과 시 하드닝 강성 [N/m]


@dataclass
class UnsprungParameters:
    """언스프렁 파라미터"""
    m_u: float = 50.0           # 언스프렁 질량 [kg]
    g: float = 9.81             # 중력가속도 [m/s²]


@dataclass
class SprungParameters:
    """차체(스프렁) 파라미터"""
    m_s_corner: float = 450.0   # 코너 차체 질량 [kg]


@dataclass
class SuspensionState:
    """서스펜션 상태 변수"""
    # 서스펜션 힘
    F_active: float = 0.0       # 액티브 힘 [N]
    F_spring: float = 0.0       # 스프링 힘 [N]
    F_damper: float = 0.0       # 댐퍼 힘 [N]
    F_s: float = 0.0            # 총 서스펜션 힘 (차체로) [N]

    # 타이어 힘
    F_z: float = 0.0            # 타이어 수직력 [N]

    # 편차 좌표
    delta_s: float = 0.0        # 서스펜션 스트로크 [m]
    delta_s_dot: float = 0.0    # 서스펜션 스트로크 속도 [m/s]
    delta_t: float = 0.0        # 타이어 압축량 [m]

    # 절대 좌표
    z_body_abs: float = 0.0     # 차체 절대 높이 [m]
    z_u_abs: float = 0.0        # 휠 중심 절대 높이 [m]
    z_u_dot: float = 0.0        # 휠 수직 속도 [m/s]


class SuspensionModel:
    """
    서스펜션 모델 (한 바퀴) - 7-DOF

    동역학:
        m_u × z_u_ddot = F_tire - F_sus - m_u × g

    좌표계:
        - 입력: heave_dev (편차)
        - 내부: z_CG = z_CG0 + heave_dev (절대)
        - 상태: z_u_abs (절대, 동역학)
    """

    def __init__(self, corner_id: str, config_path: Optional[str] = None):
        """
        서스펜션 모델 초기화

        Args:
            corner_id: 바퀴 위치 ('FL', 'FR', 'RL', 'RR')
            config_path: YAML 설정 파일 경로
        """
        if corner_id not in ['FL', 'FR', 'RL', 'RR']:
            raise ValueError(f"Invalid corner_id: {corner_id}")

        self.corner_id = corner_id
        self.config_path = config_path

        # YAML 파라미터 로드
        susp_param = load_param('suspension', config_path)
        tire_param = load_param('tire', config_path)
        unsprung_param = load_param('unsprung', config_path)
        physics_param = load_param('physics', config_path)
        vehicle_spec = load_param('vehicle_spec', config_path)
        vehicle_body = load_param('vehicle_body', config_path)

        geometry = vehicle_spec.get('geometry', {})
        corner_offsets = geometry.get('corner_offsets', {})

        # 평형점 계산 (내부에서 직접 계산)
        z_CG0 = float(susp_param.get('z_CG0', 0.5))
        R_w = float(vehicle_spec.get('wheel', {}).get('R_eff', 0.3))
        K_t = float(tire_param.get('vertical', {}).get('K_t', 200000.0))

        # 스프링 강성 (Front/Rear 구분)
        K_spring_default = float(susp_param.get('K_spring', 25000.0))
        if corner_id in ['FL', 'FR']:
            K_spring = float(susp_param.get('K_spring_front', K_spring_default))
            m_u = float(unsprung_param.get('m_u_front', unsprung_param.get('m_u', 50.0)))
        else:  # RL, RR
            K_spring = float(susp_param.get('K_spring_rear', K_spring_default))
            m_u = float(unsprung_param.get('m_u_rear', unsprung_param.get('m_u', 50.0)))

        m_s = float(vehicle_body.get('m', 1500.0))  # 전체 차체 스프렁 질량
        g = float(physics_param.get('g', 9.81))

        # 하중 분배 비율 (Front/Rear)
        if corner_id in ['FL', 'FR']:
            load_ratio = float(geometry.get('front_load_ratio', 0.5))
        else:  # RL, RR
            load_ratio = float(geometry.get('rear_load_ratio', 0.5))

        # 코너별 스프렁 질량 (좌우 대칭)
        m_s_corner = (m_s * load_ratio) / 2.0

        # 1. 타이어 평형 압축량 (차체 무게 + 휠 무게)
        # F_tire = (m_s_corner + m_u) × g
        delta_t_eq = (m_s_corner + m_u) * g / K_t

        # 2. 평형 시 휠 중심 높이 (타이어가 압축된 상태)
        # delta_t = R_w + z_road - z_u_abs
        # delta_t_eq = R_w - z_u_0 (z_road=0 가정)
        z_u_0 = R_w - delta_t_eq

        # 3. 서스펜션 평형 압축량 (차체 무게만, 양수 = 압축)
        # 평형: F_sus = m_s_corner * g, F_sus = K_spring * delta_s_comp
        delta_s_comp = m_s_corner * g / K_spring

        # 4. 서스펜션 자유 길이 (평형 조건에서 역계산)
        # 평형 시: delta_s_eq = (z_CG0 - z_u_0) - L_s0 = -delta_s_comp
        # 역산: L_s0 = (z_CG0 - z_u_0) + delta_s_comp
        L_s0 = (z_CG0 - z_u_0) + delta_s_comp

        # 댐퍼 계수 (Front/Rear, Compression/Rebound 분리)
        if corner_id in ['FL', 'FR']:
            damper_dict = susp_param.get('damper_front', {})
        else:  # RL, RR
            damper_dict = susp_param.get('damper_rear', {})

        C_damper_compression = float(damper_dict.get('compression', 1000.0))
        C_damper_rebound = float(damper_dict.get('rebound', 1000.0))

        # 서스펜션 파라미터
        self.params = SuspensionParameters(
            K_spring=K_spring,  # 이미 읽은 값 재사용
            C_damper_compression=C_damper_compression,
            C_damper_rebound=C_damper_rebound,
            z_CG0=z_CG0,
            L_s0=L_s0,
            R_w=R_w,
            delta_s_min=float(susp_param.get('delta_s_min', -0.08)),
            delta_s_max=float(susp_param.get('delta_s_max', 0.05)),
            lead=float(susp_param.get('lead', 0.01)),
            efficiency=float(susp_param.get('efficiency', 0.9)),
            F_active_max=float(susp_param.get('F_active_max', SuspensionParameters.F_active_max)),
        )

        # 코너별 CG 기준 오프셋
        offsets_for_corner = corner_offsets.get(corner_id, {})
        self._x_i = float(offsets_for_corner.get('x', 0.0))
        self._y_i = float(offsets_for_corner.get('y', 0.0))

        # 타이어 파라미터
        vert_param = tire_param.get('vertical', {})
        self.tire_params = TireVerticalParameters(
            K_t=float(vert_param.get('K_t', 200000.0)),
            C_t=float(vert_param.get('C_t', 500.0)),
            delta_t_max=float(vert_param.get('delta_t_max', TireVerticalParameters.delta_t_max)),
            K_hard=float(vert_param.get('K_hard', TireVerticalParameters.K_hard)),
        )

        # 언스프렁 파라미터
        self.unsprung_params = UnsprungParameters(
            m_u=m_u,
            g=g
        )

        # 차체 파라미터 (코너별 질량은 하중 분배 비율 적용)
        self.sprung_params = SprungParameters(
            m_s_corner=m_s_corner
        )

        # 전체 스프렁 질량 저장 (참조용)
        self.sprung_params.m_s_total = m_s

        # 평형 시 값 저장 (reset에서 사용)
        self._z_u_0 = z_u_0

        # 평형 상태 계산 및 저장
        self._delta_s_eq = -delta_s_comp  # 압축 방향 (음수)
        self._F_spring_eq = -K_spring * self._delta_s_eq  # = (m_s/4) * g
        self._F_sus_eq = self._F_spring_eq  # 초기에는 damper=0, active=0
        self._delta_t_eq = delta_t_eq
        self._F_tire_eq = K_t * self._delta_t_eq  # = (m_s/4 + m_u) * g
        self._z_body_eq = z_CG0

        # 초기 상태 (평형점으로 설정)
        self.state = SuspensionState(
            z_u_abs=self._z_u_0,
            z_u_dot=0.0,
            F_spring=self._F_spring_eq,
            F_damper=0.0,
            F_s=self._F_sus_eq,
            F_z=self._F_tire_eq,
            delta_s=self._delta_s_eq,
            delta_t=self._delta_t_eq,
            z_body_abs=self._z_body_eq
        )

    def _calculate_body_height(
        self,
        z_CG: float,
        roll: float,
        pitch: float
    ) -> float:
        """
        코너별 차체 절대 높이 계산

        z_body_i_abs = z_CG + roll × y_i + pitch × x_i
        """
        # 올바른 부호: + roll + pitch
        # roll(+) = 왼쪽 올라감 → FL(+y) 높아짐
        # pitch(+) = 앞쪽 올라감 → FL(+x) 높아짐
        z_body_abs = z_CG + roll * self._y_i - pitch * self._x_i
        return float(z_body_abs)

    def _calculate_active_force(self, T_susp: float) -> float:
        """액티브 서스펜션 힘 (볼 스크류)--> 레퍼런스 확인."""
        F_active = (2.0 * np.pi * self.params.efficiency * T_susp) / self.params.lead

        # 힘 포화 한계 (F_active_max가 설정된 경우)
        if self.params.F_active_max is not None:
            F_active = float(np.clip(F_active, -self.params.F_active_max, self.params.F_active_max))

        return float(F_active)


    def _calculate_tire_force(self, delta_t: float, delta_t_dot: float) -> float:
        """타이어 수직력 계산

        delta_t > 0 : 타이어가 노면을 압축하는 양 (contact)
        delta_t <= 0: 이륙 (no contact, F_tire = 0)
        """
        if delta_t <= 0.0:
            # 이륙 - 힘 없음
            return 0.0

        delta_t = float(delta_t)
        delta_t_max = float(self.tire_params.delta_t_max)

        # 선형 구간 + 최대 압축 초과분 하드닝 강성 적용
        delta_t_clip = min(delta_t, delta_t_max)
        excess = max(0.0, delta_t - delta_t_max)

        F_tire = (
            self.tire_params.K_t * delta_t_clip
            + self.tire_params.K_hard * excess
            + self.tire_params.C_t * delta_t_dot
        )

        # 접촉력은 절대 음수 불가 (unilateral contact)
        return max(0.0, float(F_tire))

    def _apply_tire_deflection_limits(self, delta_t: float, delta_t_dot: float, z_u_ddot: float) -> float:
        """타이어 최대 압축 한계 적용 (비활성화 - 하드닝 강성이 물리적으로 처리)

        하드닝 강성(K_hard)이 delta_t_max 초과 시 자동으로 큰 힘을 발생시켜
        더 이상의 압축을 물리적으로 방지하므로, 인위적인 한계 적용은 불필요함.
        """
        # 하드닝 강성 사용 시 클리핑 비활성화
        # at_max_and_compressing = (
        #     delta_t >= self.tire_params.delta_t_max and
        #     delta_t_dot >= 0.0 and
        #     z_u_ddot <= 0.0  # z_u가 아래로 가속 -> delta_t 증가
        # )
        # if at_max_and_compressing:
        #     self.state.z_u_dot = 0.0
        #     return 0.0
        return z_u_ddot

    def _clip_tire_deflection(self, z_road: float) -> float:
        """적분 후 타이어 압축량 클리핑 (비활성화 - 하드닝 강성이 물리적으로 처리)

        하드닝 강성이 delta_t_max 초과분을 물리적으로 처리하므로 클리핑 불필요.
        단, 이륙 상태(delta_t <= 0)는 그대로 반환.

        Returns:
            계산된 delta_t [m]
        """
        delta_t = self.params.R_w + z_road - self.state.z_u_abs

        # 하드닝 강성 사용 시 최대값 클리핑 비활성화
        # if delta_t > self.tire_params.delta_t_max:
        #     self.state.z_u_abs = self.params.R_w + z_road - self.tire_params.delta_t_max
        #     self.state.z_u_dot = 0.0
        #     return float(self.tire_params.delta_t_max)

        if delta_t <= 0.0:
            return 0.0

        return float(delta_t)

    def _apply_stroke_limits(self, z_body_abs: float, delta_s: float, delta_s_dot: float, z_u_ddot: float) -> float:
        """서스펜션 스트로크 한계 적용 (Steering 방식)

        한계에 도달하고 더 나가려는 경우 가속도와 속도를 0으로

        Args:
            z_body_abs: 차체 절대 높이 [m]
            delta_s: 현재 스트로크 [m]
            delta_s_dot: 스트로크 속도 [m/s]
            z_u_ddot: 휠 가속도 [m/s^2]

        Returns:
            수정된 z_u_ddot [m/s^2]
        """
        # Bump stop (과압축): delta_s < delta_s_min
        at_bump_and_compressing = (
            delta_s <= self.params.delta_s_min and
            delta_s_dot <= 0.0 and
            z_u_ddot <= 0.0  # z_u가 아래로 가속 (압축)
        )

        # Rebound stop (과신장): delta_s > delta_s_max
        at_rebound_and_extending = (
            delta_s >= self.params.delta_s_max and
            delta_s_dot >= 0.0 and
            z_u_ddot >= 0.0  # z_u가 위로 가속 (신장)
        )

        if at_bump_and_compressing or at_rebound_and_extending:
            self.state.z_u_dot = 0.0
            return 0.0

        return z_u_ddot

    def _clip_stroke_position(self, z_body_abs: float) -> float:
        """적분 후 스트로크 위치 클리핑 (안전장치)

        Args:
            z_body_abs: 차체 절대 높이 [m]

        Returns:
            클리핑된 delta_s [m]
        """
        delta_s = (z_body_abs - self.state.z_u_abs) - self.params.L_s0

        if delta_s < self.params.delta_s_min:
            self.state.z_u_abs = z_body_abs - self.params.L_s0 - self.params.delta_s_min
            self.state.z_u_dot = 0.0
            return self.params.delta_s_min
        elif delta_s > self.params.delta_s_max:
            self.state.z_u_abs = z_body_abs - self.params.L_s0 - self.params.delta_s_max
            self.state.z_u_dot = 0.0
            return self.params.delta_s_max

        return delta_s

    def update(
        self,
        dt: float,
        T_susp: float,
        X_body: np.ndarray,
        z_road: float = 0.0,
        z_road_dot: float = 0.0
    ) -> Tuple[float, float]:
        """
        서스펜션 동역학 업데이트

        Args:
            dt: 시간 간격 [s]
            T_susp: 서스펜션 토크 [N*m]
            X_body: 차체 상태 벡터 [heave, roll, pitch, heave_dot, roll_dot, pitch_dot]
            z_road: 노면 높이 [m]
            z_road_dot: 노면 속도 [m/s]

        Returns:
            (F_s, F_z): 서스펜션 복원력, 타이어 수직력
        """
        # X_body 배열에서 값 추출
        heave = float(X_body[0])
        roll = float(X_body[1])
        pitch = float(X_body[2])
        heave_dot = float(X_body[3])
        roll_dot = float(X_body[4])
        pitch_dot = float(X_body[5])

        # 1. 절대 좌표 계산
        z_CG = self.params.z_CG0 + heave
        z_body_abs = self._calculate_body_height(z_CG, roll, pitch)

        # 차체 속도
        z_body_dot = heave_dot + roll_dot * self._y_i - pitch_dot * self._x_i

        # 2. 편차 계산
        delta_s = (z_body_abs - self.state.z_u_abs) - self.params.L_s0
        delta_s_dot = z_body_dot - self.state.z_u_dot

        # Tire compression: wheel moving downward (z_u_abs decreases) makes delta_t positive
        delta_t = self.params.R_w + z_road - self.state.z_u_abs
        # delta_t_dot = d/dt(R_w + z_road - z_u_abs) = z_road_dot - z_u_dot
        delta_t_dot = z_road_dot - self.state.z_u_dot

        # 3. 서스펜션 힘
        F_active = self._calculate_active_force(T_susp)
        # Restoring forces pull the body downward (and wheel upward) when delta_s > 0
        F_spring = -self.params.K_spring * delta_s

        # 댐퍼 힘 (비대칭: 압축/리바운드 분리)
        # delta_s_dot < 0: 압축 (compression, push) -> C_damper_compression 사용
        # delta_s_dot > 0: 리바운드 (rebound, pull) -> C_damper_rebound 사용
        if delta_s_dot < 0.0:
            C_damper = self.params.C_damper_compression
        else:
            C_damper = self.params.C_damper_rebound

        F_damper = -C_damper * delta_s_dot

        # 서스펜션 복원력 (스프링 + 댐퍼 + 액티브)
        F_s = F_spring + F_damper + F_active

        # 4. 타이어 수직력
        F_z = self._calculate_tire_force(delta_t, delta_t_dot)

        # 5. 언스프렁 동역학
        # m_u × z_u_ddot = F_z - F_s - m_u × g
        F_gravity = self.unsprung_params.m_u * self.unsprung_params.g
        z_u_ddot = (F_z - F_s - F_gravity) / self.unsprung_params.m_u

        # 타이어 최대 압축 한계 적용 (적분 전)
        z_u_ddot = self._apply_tire_deflection_limits(delta_t, delta_t_dot, z_u_ddot)

        # 스트로크 한계 적용 (적분 전)
        z_u_ddot = self._apply_stroke_limits(z_body_abs, delta_s, delta_s_dot, z_u_ddot)

        # 적분
        self.state.z_u_dot += z_u_ddot * dt
        self.state.z_u_abs += self.state.z_u_dot * dt

        # 스트로크 위치 클리핑 (적분 후 안전장치)
        delta_s = self._clip_stroke_position(z_body_abs)

        # 타이어 압축량 클리핑 (적분 후 안전장치)
        delta_t = self._clip_tire_deflection(z_road)

        # 6. 상태 저장
        self.state.F_active = F_active
        self.state.F_spring = F_spring
        self.state.F_damper = F_damper
        self.state.F_s = F_s
        self.state.F_z = F_z
        self.state.delta_s = delta_s
        self.state.delta_s_dot = delta_s_dot
        self.state.delta_t = delta_t
        self.state.z_body_abs = z_body_abs

        return F_s, F_z

    def get_state(self) -> Dict:
        """현재 서스펜션 상태 조회"""
        return {
            "F_active": self.state.F_active,
            "F_spring": self.state.F_spring,
            "F_damper": self.state.F_damper,
            "F_s": self.state.F_s,
            "F_z": self.state.F_z,
            "delta_s": self.state.delta_s,
            "delta_t": self.state.delta_t,
            "z_body_abs": self.state.z_body_abs,
            "z_u_abs": self.state.z_u_abs,
            "z_u_dot": self.state.z_u_dot,
        }

    def reset(self) -> None:
        """서스펜션 상태 리셋 (평형점으로)"""
        self.state = SuspensionState(
            z_u_abs=self._z_u_0,
            z_u_dot=0.0,
            F_spring=self._F_spring_eq,
            F_damper=0.0,
            F_s=self._F_sus_eq,
            F_z=self._F_tire_eq,
            delta_s=self._delta_s_eq,
            delta_t=self._delta_t_eq,
            z_body_abs=self._z_body_eq
        )
