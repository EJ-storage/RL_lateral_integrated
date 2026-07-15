from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import importlib
import math
import sys

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required for sbwRL.customEnv.simulator.CustomEnv. "
        "Install gymnasium before creating the environment."
    ) from exc

from stbwRL.env_config import EnvironmentConfig
from .datatype.history import EnvHistory
from .datatype.observation_function import cal_observation
from .datatype.reward_function import calculate_reward, calculate_sideslip
from .controllers.speed_controller import (
    SpeedPIConfig,
    SpeedPIController,
    SpeedPIOutput,
    target_accel_to_axle_torque,
)
from .scenario_manager import ScenarioManager


def _vehicle_sim_root_for_import(vehicle_model_path: str) -> Path: # 차량 모델 import 경로
    plant_path = Path(vehicle_model_path)
    if plant_path.name == "vehicle_sim":
        return plant_path.parent
    return plant_path


def _import_stbw_vehicle_body(config: EnvironmentConfig): # 차량 모델에서 StbwVehicleBody 클래스 가져옴
    plant_root = _vehicle_sim_root_for_import(config.vehicle_model_path)
    if not plant_root.exists():
        raise ImportError(
            f"vehicle_sim import failed: plant root does not exist: {plant_root}. "
            f"Configured vehicle_model_path={config.vehicle_model_path!r}"
        )

    plant_root_str = str(plant_root)
    if plant_root_str not in sys.path:
        sys.path.insert(0, plant_root_str)

    module_name = "vehicle_sim.stbw_model.vehicle_body.vehicle_body"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(
            f"vehicle_sim import failed while importing {module_name!r} from {plant_root_str!r}: {exc}"
        ) from exc

    vehicle_cls = getattr(module, "StbwVehicleBody", None)
    if vehicle_cls is None:
        raise ImportError(
            f"vehicle_sim import failed: {module_name!r} has no class 'StbwVehicleBody'."
        )

    required_methods = ("reset", "update", "get_outputs")
    missing = [name for name in required_methods if not hasattr(vehicle_cls, name)]
    if missing:
        raise ImportError(
            f"vehicle_sim import failed: StbwVehicleBody is missing required methods {missing}."
        )
    return vehicle_cls


class CustomEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        super().__init__()
        self.config = config if config is not None else EnvironmentConfig()
        self.vehicle_cls = _import_stbw_vehicle_body(self.config)
        self._load_vehicle_wheelbase() # wheelbase 가져오기

        self.rng = np.random.default_rng() # 난수 생성기 만들기
        self.scenario_manager = ScenarioManager(self.config, self.rng) # ScenarioManager 생성
        self.scenario = None # 시나리오 객체 없음
        self.vehicle = None # 차량 객체 없음
        self.history = EnvHistory() # 사용 이력
        self.speed_controller = self._make_speed_controller() # PI 제어기 생성
        self.elapsed_time = 0.0
        self.step_count = 0
        self.steer = 0.0 # 에이전트 명령 적분으로 만든 조향각
        self.steer_dot = 0.0 # 에이전트 명령 적분으로 만든 조향각속도
        self.applied_steer = 0.0 # 속도 기반 조향 스케일 적용 후 실제 차량 모델에 넣는 조향각
        self.applied_steer_dot = 0.0 # 속도 기반 조향 스케일 적용 후 실제 차량 모델에 넣는 조향각속도
        self.previous_applied_steer = 0.0
        self.speed_hold_target_speed_mps = 0.0 # 속도 제어기 목표 속도
        self.speed_hold_output: Optional[SpeedPIOutput] = None # PI 제어기 출력 
        self.speed_hold_drive_torque_nm = 0.0 # 구동 토크
        self.speed_hold_brake_motor_torque = 0.0 # 제동 토크
        self.episode_reward = 0.0

        self.observation_space = spaces.Box( # 관측 공간
            low=-np.inf,
            high=np.inf,
            shape=(self.config.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box( # 행동 공간
            low=-1.0,
            high=1.0,
            shape=(self.config.action_dim,),
            dtype=np.float32,
        )

    def _load_vehicle_wheelbase(self) -> None: # wheelbase 가져오기
        load_param = importlib.import_module("vehicle_sim.utils.config_loader").load_param
        vehicle_spec = load_param("vehicle_spec", self.config.config_path)
        geometry = vehicle_spec.get("geometry", {})
        self.config.wheelbase = float(geometry["L_wheelbase"])

    def update_config(self, config: EnvironmentConfig) -> None: # 커리큘럼 단계 변경 시 환경 설정 교체
        self.config = config # 새로운 단계 설정으로 교체
        self._load_vehicle_wheelbase() # wheelbase 읽어옴
        self.scenario_manager.config = self.config # ScenarioManager에서 config 변경 후 새 시나리오 생성
        self.scenario_manager.current_scenario = None # 기존 사용 시나리오 초기화
        self.speed_controller = self._make_speed_controller() # 속도 제어기 생성

    def _make_speed_controller(self) -> SpeedPIController: # 속도 제어기 생성
        return SpeedPIController(
            SpeedPIConfig(
                kp=float(self.config.speed_hold_kp),
                ki=float(self.config.speed_hold_ki),
                min_target_accel_mps2=float(self.config.speed_hold_min_accel_mps2),
                max_target_accel_mps2=float(self.config.speed_hold_max_accel_mps2),
                integrator_limit_mps=float(self.config.speed_hold_integrator_limit_mps),
                speed_deadband_mps=float(self.config.speed_hold_deadband_mps),
            )
        )

    def _make_vehicle(self): # 차량 객체 생성
        return self.vehicle_cls(config_path=self.config.config_path)

    @staticmethod # 정적 메소드 정의
    def _get_output_value( # 차량 상태값 찾기 (vehicle.get_outputs, vehicle.state)
        outputs: Dict[str, Any],
        state: Any,
        key: str,
    ) -> float:
        if key in outputs: # vehicle 출력 딕셔너리에서 필요한 값 가져오기
            raw_value = outputs[key]
            source = "vehicle.get_outputs()"
        elif state is not None and hasattr(state, key):
            raw_value = getattr(state, key)
            source = "vehicle.state"
        else: # 필요한 값이 없을 때 에러
            output_keys = sorted(str(name) for name in outputs.keys())
            state_type = type(state).__name__ if state is not None else "None"

            raise KeyError(
                f"Required vehicle state '{key}' was not found. "
                f"Available get_outputs() keys={output_keys}, "
                f"state type={state_type}."
            )

        try:
            return float(raw_value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Vehicle state '{key}' from {source} must be convertible "
                f"to float, but received {raw_value!r} "
                f"({type(raw_value).__name__})."
            ) from exc

    def _get_current_state(self) -> Dict[str, float]: # 차량 모델 출력을 딕셔너리로 구성
        if self.vehicle is None:
            raise RuntimeError("Vehicle has not been created.")

        outputs = dict(self.vehicle.get_outputs())
        state_obj = getattr(self.vehicle, "state", None)

        front_steering_angle = self._get_output_value(
            outputs,
            state_obj,
            "front_steering_angle",
        )
        front_steering_rate = self._get_output_value(
            outputs,
            state_obj,
            "front_steering_rate",
        )
        front_road_wheel_angle = self._get_output_value(
            outputs,
            state_obj,
            "front_road_wheel_angle",
        )
        front_road_wheel_rate = self._get_output_value(
            outputs,
            state_obj,
            "front_road_wheel_rate",
        )

        return {
            "t": float(self.elapsed_time),
            "x": self._get_output_value(outputs, state_obj, "x"),
            "y": self._get_output_value(outputs, state_obj, "y"),
            "yaw": self._get_output_value(outputs, state_obj, "yaw"),
            "vx": self._get_output_value(outputs, state_obj, "velocity_x"),
            "vy": self._get_output_value(outputs, state_obj, "velocity_y"),
            "yaw_rate": self._get_output_value(outputs, state_obj, "yaw_rate"),
            "ax": self._get_output_value(outputs, state_obj, "ax"),
            "ay": self._get_output_value(outputs, state_obj, "ay"),
            "steer": front_steering_angle,
            "steer_dot": front_steering_rate,
            "front_steering_angle": front_steering_angle,
            "front_steering_rate": front_steering_rate,
            "front_road_wheel_angle": front_road_wheel_angle,
            "front_road_wheel_rate": front_road_wheel_rate,
        }

    def _vehicle_diagnostics_info(self) -> Dict[str, float]:
        diagnostics: Dict[str, float] = {}

        diagnostic_keys = (
            "F_x_tire",
            "F_y_tire",
            "F_x_tire_raw",
            "F_y_tire_raw",
            "F_z",
            "omega_wheel",
            "friction_circle_limit",
            "friction_circle_usage",
            "friction_circle_usage_raw",
            "friction_circle_scale",
            "road_mu",
            "dugoff_kappa",
            "dugoff_alpha",
            "dugoff_delta_v",
            "dugoff_wheel_linear_speed",
            "dugoff_vx",
            "dugoff_fx_linear",
            "dugoff_fy_linear",
            "dugoff_scale",
            "dugoff_ellipse_scale",
            "dugoff_combined_demand",
        )

        for label, wheel in self.vehicle.iter_wheel_modules():
            wheel_state = wheel.get_state()
            prefix = label.lower()

            for key in diagnostic_keys:
                if key in wheel_state:
                    diagnostics[f"{prefix}_{key}"] = float(wheel_state[key])

            diagnostics[f"{prefix}_friction_circle_saturated"] = float(
                bool(wheel_state.get("friction_circle_saturated", False))
            )

        return diagnostics

    def _set_initial_vehicle_state(
        self,
        initial_speed_mps: float,
    ) -> None:
        self.vehicle.set_state_vector(
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    float(initial_speed_mps),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
        )

        wheel_speed = float(initial_speed_mps) / self._get_wheel_radius_m() # 필요 휠 스피드 계산
        for _, wheel in self.vehicle.iter_wheel_modules():
            wheel.drive.state.wheel_speed = wheel_speed # 휠 스피드 맞춰줌

    def _apply_front_steering_state(self) -> None: # 차량 모델에 조향각, 조향각속도 대입
        if hasattr(self.vehicle, "set_front_steering_state"):
            self.vehicle.set_front_steering_state(self.applied_steer, self.applied_steer_dot)

    def _speed_dependent_steering_scale(self, vx: float) -> float: # 속도 기반 조향 스케일

        threshold = float(self.config.speed_dependent_steering_threshold_mps)
        vx_abs = abs(float(vx))
        if threshold <= 0.0 or vx_abs < threshold:
            return 1.0
        return float((threshold * threshold) / max(vx_abs * vx_abs, 1e-12)) # 속도에 따라 조향 축소

    def _update_applied_steering(self, vx: float) -> None:
        scale = self._speed_dependent_steering_scale(vx) # 현재 속도에서 조향 스케일
        self.previous_applied_steer = float(self.applied_steer)
        self.applied_steer = float( # 조향각 스케일링, min/max 적용
            np.clip(
                float(self.steer) * scale,
                -float(self.config.steer_max),
                float(self.config.steer_max),
            )
        )
        self.applied_steer_dot = float(
            (self.applied_steer - self.previous_applied_steer) / float(self.config.control_dt)
        )

    def _observation_state(self, current_state: Dict[str, float]) -> Dict[str, float]:
        observation_state = dict(current_state)
        observation_state["steer"] = float(self.steer)
        observation_state["steer_dot"] = float(self.steer_dot)
        return observation_state

    def _initialize_steering_from_driver_offset(self) -> None: # 초기 운전자 조향 offset 설정
        steering_wheel_angle, steering_wheel_rate = (
            self.scenario.get_driver_steering_input(0.0)
        ) # 시나리오 시간 0초의 운전자 조향각, 조향각속도 가져옴

        steering_ratio = float(self.config.steering_ratio) # 조향비
        if abs(steering_ratio) <= 1e-12:
            raise ValueError("steering_ratio must be non-zero.")

        target_front_angle = float(steering_wheel_angle) / steering_ratio # 조향각
        target_front_rate = float(steering_wheel_rate) / steering_ratio # 조향각속도

        self.steer = float(
            np.clip(
                target_front_angle,
                -float(self.config.steer_max),
                float(self.config.steer_max),
            )
        ) # 전륜 조향각 범위 제한

        self.steer_dot = float(
            np.clip(
                target_front_rate,
                -float(self.config.steer_dot_max),
                float(self.config.steer_dot_max),
            )
        ) # 전륜 조향각속도 범위 제한

        self.applied_steer = float(self.steer)
        self.applied_steer_dot = float(self.steer_dot)
        self.previous_applied_steer = float(self.applied_steer)

    def _get_vehicle_mass_kg(self) -> float: # 차량 질량
        mass = float(self.vehicle.params.m_total)
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"Invalid vehicle mass: {mass}")
        return mass

    def _get_wheel_radius_m(self) -> float: # 차량 휠 반지름
        _, wheel = next(iter(self.vehicle.iter_wheel_modules()))
        radius = float(wheel.drive.params.R_wheel)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"Invalid wheel radius: {radius}")
        return radius

    def _get_brake_gains( # 휠 브레이크 게인 조회
        self,
    ) -> Tuple[float, float]:
        clamp_gains = []
        torque_gains = []

        for _, wheel in self.vehicle.iter_wheel_modules():
            clamp_gains.append(
                float(wheel.brake._clamp_gain)
            )
            torque_gains.append(
                float(wheel.drive._clamp_to_torque)
            )

        return (
            float(np.mean(clamp_gains)),
            float(np.mean(torque_gains)),
        )

    def _update_speed_hold_command(self, current_state: Dict[str, float]) -> None: # 속도 유지 명령 계산
        if not bool(self.config.enable_speed_hold_pi):
            self.speed_hold_output = None
            self.speed_hold_drive_torque_nm = 0.0
            self.speed_hold_brake_motor_torque = 0.0
            return
        
        current_speed_mps = float(current_state["vx"])

        beta = calculate_sideslip(
            current_state,
            eps=float(self.config.curvature_denominator_eps),
        )
        if (
            current_speed_mps > float(self.config.longitudinal_input_block_min_vx_mps)
            and abs(beta) > float(self.config.longitudinal_input_block_beta_rad)
        ):
            self.speed_hold_output = None
            self.speed_hold_drive_torque_nm = 0.0
            self.speed_hold_brake_motor_torque = 0.0
            return

        self.speed_hold_output = self.speed_controller.update( # 현재 속도와 목표 속도 제어기에 전달
            target_speed_mps=float(self.speed_hold_target_speed_mps),
            current_speed_mps=current_speed_mps,
            dt=float(self.config.control_dt),
        )
        clamp_gain, brake_torque_gain = self._get_brake_gains() # 차량 브레이크 gain 가져옴
        command = target_accel_to_axle_torque( # PI 제어기가 출력한 목표 가속도를 차축 토크 명령으로 변환
            target_accel_mps2=self.speed_hold_output.target_accel_mps2,
            vehicle_mass_kg=self._get_vehicle_mass_kg(),
            wheel_radius_m=self._get_wheel_radius_m(),
            brake_clamp_gain=clamp_gain,
            brake_torque_gain=brake_torque_gain,
            brake_axle_count=len(tuple(self.config.speed_hold_brake_axles)),
        )
        self.speed_hold_drive_torque_nm = float(command.drive_torque_nm) 
        self.speed_hold_brake_motor_torque = float(command.brake_motor_torque)

    def _preroll_initial_vehicle_state(self) -> int: # 초기 차량 모델을 일정 시간 적분해서 차량 상태 초기화
        preroll_time_s = max(float(self.config.initial_state_preroll_time_s), 0.0) # 시간 음수 X
        control_dt = float(self.config.control_dt)
        if preroll_time_s <= 0.0 or control_dt <= 0.0:
            return 0

        preroll_steps = int(round(preroll_time_s / control_dt))
        if preroll_steps <= 0:
            return 0

        episode_target_speed_mps = float(self.speed_hold_target_speed_mps) # 목표 속도 임시 저장
        self.speed_hold_target_speed_mps = float(self.scenario.initial_speed_mps) # preroll 동안 초기 속도 유지하도록 목표 속도를 초기 속도로 바꿈
        self.speed_controller.reset() # 제어기 초기화

        for _ in range(preroll_steps):
            current_state = self._get_current_state() # 현재 차량 상태 읽기
            self._update_applied_steering(current_state["vx"]) # 속도 스케일링 적용 후 조향각 계산
            self._apply_front_steering_state() # 차량 내부 모델 조향각 업데이트
            self._update_speed_hold_command(current_state) # 초기 속도 유지
            self._integrate_vehicle_one_control_step() # substep 수행

        self.elapsed_time = 0.0
        self.step_count = 0
        self.episode_reward = 0.0 # 다시 초기화
        self.speed_hold_target_speed_mps = episode_target_speed_mps # 목표 속도로 복원
        self.speed_controller.reset() # 제어기 초기화
        self.speed_hold_output = None # 토크 명령 초기화 (PI 제어기 출력)
        self.speed_hold_drive_torque_nm = 0.0 # 토크 명령 초기화
        self.speed_hold_brake_motor_torque = 0.0 # 토크 명령 초기화
        self.scenario.reset_reference_history()
        self.history.reset()
        return preroll_steps

    def _speed_hold_info(self) -> Dict[str, float]: # 속도 제어 로그
        output = self.speed_hold_output
        return {
            "speed_hold_enabled": float(bool(self.config.enable_speed_hold_pi)),
            "speed_hold_target_speed_mps": float(self.speed_hold_target_speed_mps),
            "speed_hold_speed_error_mps": 0.0 if output is None else float(output.speed_error_mps),
            "speed_hold_target_accel_mps2": 0.0 if output is None else float(output.target_accel_mps2),
            "speed_hold_raw_target_accel_mps2": 0.0 if output is None else float(output.raw_target_accel_mps2),
            "speed_hold_integrator_state_mps": 0.0 if output is None else float(output.integrator_state_mps),
            "speed_hold_saturated": 0.0 if output is None else float(output.saturated),
            "speed_hold_drive_torque_nm": float(self.speed_hold_drive_torque_nm),
            "speed_hold_brake_motor_torque": float(self.speed_hold_brake_motor_torque),
        }

    def _build_axle_inputs(self) -> Dict[str, Dict[str, float]]: # 전후륜 차축 입력 구성
        road_mu = float(self.scenario.road_friction)
        drive_axle = str(self.config.speed_hold_drive_axle).upper()
        brake_axles = {str(label).upper() for label in self.config.speed_hold_brake_axles}
        return {
            "F": {
                "T_steer": 0.0,
                "T_brk": float(self.speed_hold_brake_motor_torque if "F" in brake_axles else 0.0),
                "T_Drv": float(self.speed_hold_drive_torque_nm if drive_axle == "F" else 0.0),
                "steering_angle": float(self.applied_steer),
                "steering_rate": float(self.applied_steer_dot),
                "road_mu": road_mu,
            },
            "R": {
                "T_steer": 0.0,
                "T_brk": float(self.speed_hold_brake_motor_torque if "R" in brake_axles else 0.0),
                "T_Drv": float(self.speed_hold_drive_torque_nm if drive_axle == "R" else 0.0),
                "road_mu": road_mu,
            },
        }

    def _integrate_vehicle_one_control_step(self) -> None: # 차량 상태 적분 (substep)
        sim_dt = float(self.config.sim_dt)
        control_dt = float(self.config.control_dt)

        num_substeps = int(round(control_dt / sim_dt))

        axle_inputs = self._build_axle_inputs()

        for _ in range(num_substeps):
            self.vehicle.update(
                dt=sim_dt,
                axle_inputs=axle_inputs,
                direction=1,
            )

    def _apply_fixed_target_curvature(self) -> None:
        fixed_target_curvature = getattr(self.config, "fixed_target_curvature_m_inv", None)
        if fixed_target_curvature is None:
            return
        if self.scenario is None:
            raise RuntimeError("Scenario has not been created.")
        front_wheel_angle = math.atan(float(fixed_target_curvature) * float(self.config.wheelbase))
        steering_max = front_wheel_angle * float(self.config.steering_ratio)
        self.scenario = self.scenario.with_steering_max(steering_max)
        self.scenario_manager.current_scenario = self.scenario

    def _scenario_info(self) -> Dict[str, float]:
        return {
            "initial_speed_mps": float(self.scenario.initial_speed_mps),
            "road_friction": float(self.scenario.road_friction),
            "steering_max": float(self.scenario.steering_max),
            "frequency_hz": float(self.scenario.frequency_hz),
            "steering_offset": float(self.scenario.steering_offset),
            "wheelbase": float(self.scenario.wheelbase),
            "steering_ratio": float(self.scenario.steering_ratio),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]: # 환경 초기화
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.scenario_manager.rng = self.rng

        self.scenario = self.scenario_manager.reset() # 새로운 시나리오 or 조향 1% 감소
        self.vehicle = self._make_vehicle() # 차량 객체 생성
        self.vehicle.reset() # 차량 모델 초기화

        self.elapsed_time = 0.0
        self.step_count = 0
        self.steer = 0.0
        self.steer_dot = 0.0
        self.applied_steer = 0.0
        self.applied_steer_dot = 0.0
        self.previous_applied_steer = 0.0 
        self._apply_fixed_target_curvature()
        self.episode_reward = 0.0 
        self.speed_controller.reset()
        speed_hold_target = self.config.speed_hold_target_speed_mps
        if speed_hold_target is None: 
            speed_hold_target = self.scenario.initial_speed_mps
        self.speed_hold_target_speed_mps = float(speed_hold_target)
        self.speed_hold_output = None
        self.speed_hold_drive_torque_nm = 0.0
        self.speed_hold_brake_motor_torque = 0.0
        self.history.reset()

        self._set_initial_vehicle_state(self.scenario.initial_speed_mps) # 차량 초기 속도 설정
        self._initialize_steering_from_driver_offset() # 조향각 초기화
        self._apply_front_steering_state() # 초기 조향 상태 실제 차량 모델에 업데이트
        initial_state_preroll_steps = self._preroll_initial_vehicle_state() # 초기 상태 적분

        current_state = self._get_current_state() # preroll 이후 차량 상태 읽기
        self._update_speed_hold_command(current_state) # 속도 유지 명령 계산
        reference = self.scenario.get_target_reference( # reference 계산
            self.elapsed_time,
            current_state["vx"],
            current_state["vy"],
            dt=None,
        )
        obs = cal_observation(self._observation_state(current_state), self.config, reference) # 초기 관측치 계산

        info = {
            "scenario": self._scenario_info(),
            "reference": dict(reference),
            "road_friction": float(self.scenario.road_friction),
            "initial_state_preroll_time_s": float(self.config.initial_state_preroll_time_s),
            "initial_state_preroll_steps": int(initial_state_preroll_steps),
            **self._speed_hold_info(),
        }
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_action_dim = int(self.config.action_dim)

        if action_array.size != expected_action_dim:
            raise ValueError(
                f"CustomEnv expects a 1D action with shape ({expected_action_dim},), "
                f"got {np.asarray(action).shape}."
            )

        action_value = float(np.clip(action_array[0], -1.0, 1.0))

        dt = float(self.config.control_dt)
        steer_limit = float(self.config.steer_max)
        steer_dot_limit = float(self.config.steer_dot_max)
        steer_ddot_limit = float(self.config.steer_ddot_max)

        previous_steer = float(self.steer)
        previous_steer_dot = float(self.steer_dot)

        steer_ddot_cmd = float(action_value * steer_ddot_limit)

        if (
            previous_steer >= steer_limit
            and previous_steer_dot >= 0.0
            and steer_ddot_cmd > 0.0
        ) or (
            previous_steer <= -steer_limit
            and previous_steer_dot <= 0.0
            and steer_ddot_cmd < 0.0
        ):
            steer_ddot_cmd = 0.0

        self.steer_dot = float(
            np.clip(
                previous_steer_dot + steer_ddot_cmd * dt,
                -steer_dot_limit,
                steer_dot_limit,
            )
        )

        self.steer = float(
            previous_steer + self.steer_dot * dt
        )

        if self.steer >= steer_limit:
            self.steer = steer_limit

            if self.steer_dot > 0.0:
                self.steer_dot = 0.0

        elif self.steer <= -steer_limit:
            self.steer = -steer_limit

            if self.steer_dot < 0.0:
                self.steer_dot = 0.0

        pre_state = self._get_current_state()

        self._update_applied_steering(pre_state["vx"])
        self._apply_front_steering_state()
        self._update_speed_hold_command(pre_state)
        self._integrate_vehicle_one_control_step()

        self.elapsed_time += dt
        self.step_count += 1

        current_state = self._get_current_state()

        reference = self.scenario.get_target_reference(
            self.elapsed_time,
            current_state["vx"],
            current_state["vy"],
            dt=dt,
        )

        agent_state = self._observation_state(current_state)

        obs = cal_observation(
            agent_state,
            self.config,
            reference,
        )

        reward, terminated, _, reward_info = calculate_reward(
            agent_state,
            self.scenario,
            self.config,
            np.array([action_value], dtype=np.float32),
            reference=reference,
        )

        time_limit_reached = (
            self.step_count >= int(self.config.max_steps_per_episode)
            or self.elapsed_time >= float(self.config.max_episode_time)
        )

        truncated = bool(
            (not terminated)
            and time_limit_reached
        )

        self.episode_reward += float(reward)

        info = {
            **reward_info,
            "reference": dict(reference),
            "scenario": self._scenario_info(),
            "elapsed_time": float(self.elapsed_time),
            "step_count": int(self.step_count),
            "steer_ddot_cmd": float(steer_ddot_cmd),
            "internal_steer": float(self.steer),
            "internal_steer_dot": float(self.steer_dot),
            "applied_steer": float(self.applied_steer),
            "applied_steer_dot": float(self.applied_steer_dot),
            "road_friction": float(self.scenario.road_friction),
            "action": float(action_value),
            "t": float(current_state["t"]),
            "x": float(current_state["x"]),
            "y": float(current_state["y"]),
            "yaw": float(current_state["yaw"]),
            "vx": float(current_state["vx"]),
            "vy": float(current_state["vy"]),
            "yaw_rate": float(current_state["yaw_rate"]),
            "ax": float(current_state["ax"]),
            "ay": float(current_state["ay"]),
            "steer": float(current_state["steer"]),
            "steer_dot": float(current_state["steer_dot"]),
            **self._vehicle_diagnostics_info(),
            **self._speed_hold_info(),
        }

        self.history.append(
            t=float(self.elapsed_time),
            x=current_state["x"],
            y=current_state["y"],
            yaw=current_state["yaw"],
            vx=current_state["vx"],
            vy=current_state["vy"],
            yaw_rate=current_state["yaw_rate"],
            steer=current_state["steer"],
            steer_dot=current_state["steer_dot"],
            ax=current_state["ax"],
            ay=current_state["ay"],
            beta=info["beta"],
            curvature=info["curvature"],
            target_curvature=reference["target_curvature"],
            target_curvature_dot=reference["target_curvature_dot"],
            target_lateral_accel=reference["target_lateral_accel"],
            target_lateral_accel_dot=reference["target_lateral_accel_dot"],
            speed_hold_target_speed_mps=info[
                "speed_hold_target_speed_mps"
            ],
            speed_hold_speed_error_mps=info[
                "speed_hold_speed_error_mps"
            ],
            speed_hold_target_accel_mps2=info[
                "speed_hold_target_accel_mps2"
            ],
            speed_hold_raw_target_accel_mps2=info[
                "speed_hold_raw_target_accel_mps2"
            ],
            speed_hold_integrator_state_mps=info[
                "speed_hold_integrator_state_mps"
            ],
            speed_hold_drive_torque_nm=info[
                "speed_hold_drive_torque_nm"
            ],
            speed_hold_brake_motor_torque=info[
                "speed_hold_brake_motor_torque"
            ],
            action=float(action_value),
            reward=float(reward),
            reward_track=info["reward_track"],
            reward_slip=info["reward_slip"],
            reward_used=info["reward_used"],
            terminated=bool(terminated),
            truncated=bool(truncated),
            terminated_reason=info["terminated_reason"],
        )

        if terminated or truncated:
            self.scenario_manager.update_after_episode(
                reward=self.episode_reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )

        return (
            obs,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        pass

    def close(self) -> None:
        return None
