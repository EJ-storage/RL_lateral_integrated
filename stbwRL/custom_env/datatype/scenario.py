from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import math

import numpy as np

from .reward_function import lateral_accel_limit_for_road_friction


@dataclass
class CurvatureBasedWeaveProfile: # weave 형태의 곡률 생성을 위한 프로파일
    steering_max: float # 최대 조향각
    frequency_hz: float # 조향각 주파수
    steering_offset: float # 조향각 offset
    delay: float = 1.0 

    def __post_init__(self) -> None: # 입력값을 float로 변환하고 유효성 검사 수행
        self.steering_max = float(self.steering_max)
        self.frequency_hz = float(self.frequency_hz)
        self.steering_offset = float(self.steering_offset)
        self.delay = float(self.delay)

        if self.delay < 0.0:
            raise ValueError("Curvature-based weave delay must be non-negative.")
        if self.frequency_hz <= 0.0:
            raise ValueError("Curvature-based weave frequency must be positive.")

    def evaluate(self, t: float) -> Tuple[float, float]: # 운전자 조향각, 운전자 조향각속도 반환
        if t < self.delay: # weave 시작 전 값 (=offset)
            return float(self.steering_offset), 0.0

        local_time = float(t) - self.delay # weave 시작된 후 시간
        omega = 2.0 * math.pi * self.frequency_hz # 각주파수 변환
        phase = omega * local_time # 위상 계산

        offset_component = self.steering_offset # 
        offset_rate = 0.0 

        steering_wheel_angle = self.steering_max * math.sin(phase) + offset_component # 조향각
        steering_wheel_rate = self.steering_max * omega * math.cos(phase) + offset_rate # 조향각속도

        return float(steering_wheel_angle), float(steering_wheel_rate)


@dataclass
class Scenario: # 학습 에피소드에서 사용할 전체 시나리오 조건 저장, 목표 곡률 생성
    initial_speed_mps: float # 초기 속도
    road_friction: float # 도로 마찰 계수
    steering_max: float # 최대 조향각
    frequency_hz: float # 조향각 주파수
    steering_offset: float  # 조향각 offset
    wheelbase: float # 차량 휠베이스
    steering_ratio: float # 조향비 (조향휠, 조향각)
    ay_max_for_target_curvature: float = 6.0 # 횡가속도 기반 곡률 제한
    low_speed_no_saturation_threshold_mps: float = 1.0 # 속도가 1m/s보다 작을 때 곡률 제한 없음
    weave_delay: float = 1.0 # weave 시작 전 지연시간
    target_reference_delay_s: float = 0.03 # 목표 곡률 지연 시정수
    profile: CurvatureBasedWeaveProfile = field(init=False) # 조향 입력 프로파일 객체, Scenario를 생성할 때 외부에서 넣지 않음(자동 생성됨-post_init)
    _previous_reference: Optional[Dict[str, float]] = field(default=None, init=False, repr=False) # 변화율 계산할 때, 1차 지연 계산할 때 사용

    def __post_init__(self) -> None:
        self.initial_speed_mps = float(self.initial_speed_mps)
        self.road_friction = float(self.road_friction)
        self.steering_max = float(self.steering_max)
        self.frequency_hz = float(self.frequency_hz)
        self.steering_offset = float(self.steering_offset)
        self.wheelbase = float(self.wheelbase)
        self.steering_ratio = float(self.steering_ratio)
        self.ay_max_for_target_curvature = float(self.ay_max_for_target_curvature)
        self.low_speed_no_saturation_threshold_mps = float(self.low_speed_no_saturation_threshold_mps)
        self.weave_delay = float(self.weave_delay)
        self.target_reference_delay_s = max(float(self.target_reference_delay_s), 0.0)
        self.profile = CurvatureBasedWeaveProfile(
            steering_max=self.steering_max,
            frequency_hz=self.frequency_hz,
            steering_offset=self.steering_offset,
            delay=self.weave_delay,
        )

    @classmethod # 객체 생성 필요없음
    def sample(cls, config, rng: np.random.Generator) -> "Scenario": # config와 난수 생성기를 사용하여 하나의 시나리오 무작위 생성
        target_curvature = float(config.target_curvature_max) * float(rng.random()) # 목표 곡률 크기 샘플링
        steering_sign = 1.0 if float(rng.random()) - 0.5 >= 0.0 else -1.0 # 1 또는 -1
        signed_target_curvature = steering_sign * target_curvature # 곡률 크기의 부호
        front_wheel_angle = math.atan(signed_target_curvature * float(config.wheelbase)) # 목표 곡률에 대응하는 전륜 조향각 생성
        steering_max = front_wheel_angle * float(config.steering_ratio) # 전륜 조향각을 조향휠 각도로 변환

        min_freq, max_freq = config.curvature_weave_frequency_range # 조향 주파수 불러오기
        min_freq = float(min_freq)
        max_freq = float(max_freq)
        frequency_hz = float(rng.uniform(min_freq, max_freq)) # 조향 주파수 샘플링

        min_offset, max_offset = config.curvature_weave_steering_offset_range # 조향 오프셋 불러오기
        steering_offset = float(rng.uniform(float(min_offset), float(max_offset))) # 조향 오프셋 샘플링

        speed_min, speed_max = config.initial_speed_range_mps # 초기 속도 불러오기
        initial_speed_mps = float(rng.uniform(float(speed_min), float(speed_max))) # 초기 속도 샘플링
        friction_min, friction_max = config.road_friction_range # 도로 마찰계수 불러오기
        road_friction = float(rng.uniform(float(friction_min), float(friction_max))) # 도로 마찰계수 샘플링

        return cls( # 샘플링한 값으로 새로운 Scenario 객체 생성하여 반환 (cls = Scenario)
            initial_speed_mps=initial_speed_mps,
            road_friction=road_friction,
            steering_max=steering_max,
            frequency_hz=frequency_hz,
            steering_offset=steering_offset,
            wheelbase=float(config.wheelbase),
            steering_ratio=float(config.steering_ratio),
            ay_max_for_target_curvature=lateral_accel_limit_for_road_friction(
                config,
                road_friction,
                fallback=float(getattr(config, "ay_max_for_target_curvature", 6.0)),
            ),
            low_speed_no_saturation_threshold_mps=float(config.low_speed_no_saturation_threshold_mps),
            weave_delay=float(config.weave_delay),
            target_reference_delay_s=float(config.target_reference_delay_s),
        )

    def with_steering_max(self, steering_max: float) -> "Scenario": # Scenario 조건 steering_max만 변경하여 새 시나리오 만듦(보상 음수일 때 진폭 1% 줄임)
        return Scenario(
            initial_speed_mps=self.initial_speed_mps,
            road_friction=self.road_friction,
            steering_max=float(steering_max),
            frequency_hz=self.frequency_hz,
            steering_offset=self.steering_offset,
            wheelbase=self.wheelbase,
            steering_ratio=self.steering_ratio,
            ay_max_for_target_curvature=self.ay_max_for_target_curvature,
            low_speed_no_saturation_threshold_mps=self.low_speed_no_saturation_threshold_mps,
            weave_delay=self.weave_delay,
            target_reference_delay_s=self.target_reference_delay_s,
        )

    def reset_reference_history(self) -> None: # reference의 이전 기록 초기화
        self._previous_reference = None

    def get_driver_steering_input(self, t: float) -> Tuple[float, float]: # steering wheel angle, steering wheel rate 반환
        return self.profile.evaluate(float(t))

    def get_raw_target_curvature(self, t: float) -> float: # (raw 곡률)
        steering_wheel_angle = self.get_driver_steering_input(float(t))[0] # 현재 운전자 조향 입력
        front_wheel_angle = steering_wheel_angle / max(abs(self.steering_ratio), 1e-12) # 조향휠 각도를 전륜 조향각으로 변경 
        raw_target_curvature = math.tan(front_wheel_angle) / max(abs(self.wheelbase), 1e-12) # 곡률 계산
        return float(raw_target_curvature)

    def get_target_curvature(self, t: float, vx: float, vy: float) -> float: # (횡가속도 제한 적용)
        raw_target_curvature = self.get_raw_target_curvature(float(t)) # 현재 곡률
        speed = math.hypot(float(vx), float(vy)) # 속도 계산
        if speed < self.low_speed_no_saturation_threshold_mps: # 속도가 1m/s 이하일 때 raw 곡률 사용
            return float(raw_target_curvature)

        curvature_allow = self.ay_max_for_target_curvature / max(speed * speed, 1e-12) # 횡가속도 기반 곡률 제한 적용
        target_curvature = np.clip(raw_target_curvature, -curvature_allow, curvature_allow) 
        return float(target_curvature) # 횡가속도 기반으로 제한한 곡률 반환

    def get_target_reference(self, t: float, vx: float, vy: float, dt: Optional[float] = None) -> Dict[str, float]: # 강화학습 관측에 사용할 최종 reference 계산 (1차 지연)
        raw_target_curvature = self.get_target_curvature(float(t), vx, vy) # 현재 시간과 속도에서 목표 곡률 가져오기

        previous = self._previous_reference # 이전 reference 가져옴
        tau = float(self.target_reference_delay_s) # 1차 지연 시정수 가져옴
        if dt is not None and dt > 0.0 and tau > 0.0 and previous is not None: # 1차 지연 적용
            alpha = 1.0 - math.exp(-float(dt) / tau) 
            target_curvature = (
                previous["target_curvature"]
                + alpha * (raw_target_curvature - previous["target_curvature"])
            )
        else:
            target_curvature = raw_target_curvature

        target_lateral_accel = float(vx) ** 2 * target_curvature 

        if dt is not None and dt > 0.0 and previous is not None:
            target_curvature_dot = (target_curvature - previous["target_curvature"]) / float(dt) # 횡가속도 기반 곡률 제한, 1차 지연까지 적용된 곡률로 곡률 변화율 계산
        else:
            target_curvature_dot = 0.0
        target_lateral_accel_dot = float(vx) ** 2 * target_curvature_dot 

        reference = {
            "target_curvature": float(target_curvature),
            "target_curvature_dot": float(target_curvature_dot),
            "target_lateral_accel": float(target_lateral_accel),
            "target_lateral_accel_dot": float(target_lateral_accel_dot),
        }
        self._previous_reference = dict(reference) # 이전 reference 업데이트
        return reference
