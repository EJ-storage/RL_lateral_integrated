from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stbwRL.env_config import EnvironmentConfig
from stbwRL.training.train_config import PROJECT_ROOT, EnvConfig, TrainConfig

try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecMonitor
except ImportError as exc:
    raise ImportError(
        "stable_baselines3 is required to use sbwRL.training.curriculum."
    ) from exc


KPH_TO_MPS = 1.0 / 3.6
CURRICULUM_ADVANCE_REWARD_THRESHOLD = 1000.0
CURRICULUM_ADVANCE_REWARD_WINDOW_EPISODES = 30
CURRICULUM_ADVANCE_MIN_EPISODES = 30


@dataclass(frozen=True)
class CurriculumStage: # 커리큘럼 단계 구성 정보
    name: str # 단계 이름
    env_overrides: Dict[str, Any] # 환경 구성 수정
    initial_speed_kph: float # 초기 속도
    target_speed_kph: float # 목표 속도
    steering_input: str # 조향 입력 상태 설명하는 문자열 (예: "Straight", "Weave+closed-loop")
    road_friction: float # 도로 마찰 계수
    reward_gate_enabled: bool = True # 에피소드 평균 보상
    reward_threshold: float = CURRICULUM_ADVANCE_REWARD_THRESHOLD # 3000
    reward_window_episodes: int = CURRICULUM_ADVANCE_REWARD_WINDOW_EPISODES # 30
    min_episodes: int = CURRICULUM_ADVANCE_MIN_EPISODES # 30
    description: str = ""

@dataclass(frozen=True)
class CurriculumConfig: # 전체 커리큘럼 설정 클래스
    enabled: bool # 커리큐럼 사용 여부
    stages: Tuple[CurriculumStage, ...] # 커리큘럼 단계 구성 정보


@dataclass(frozen=True)
class CurriculumResumeState: # 커리큘럼 재개 상태 클래스 
    source_path: Path # 학습 재개 파일 경로
    stage_index: int # 현재 단계 인덱스
    completed_steps: int # 완료된 스텝 수
    stage_completed_steps: int # 현재 단계에서 완료된 스텝 수
    stage_episode_count: int # 현재 단계에서 완료된 에피소드 수
    stage_recent_rewards: Tuple[float, ...] # 현재 단계의 최근 보상


CURRICULUM_MUTABLE_ENV_FIELDS = { # 커리큘럼 단계 바뀌면서 바꿀 수 있는 학습 설정, env_overrides를 사용할 수 있는 필드
    "config_path",
    "vehicle_model_path",
    "sim_dt",
    "control_dt",
    "max_episode_time",
    "initial_speed_range_mps",
    "road_friction_range",
    "target_curvature_max",
    "curvature_weave_frequency_range",
    "curvature_weave_steering_offset_range",
    "weave_delay",
    "target_reference_delay_s",
    "initial_state_preroll_time_s",
    "enable_speed_dependent_steering_scaling",
    "speed_dependent_steering_threshold_mps",
    "enable_speed_hold_pi",
    "speed_hold_target_speed_mps",
    "speed_hold_kp",
    "speed_hold_ki",
    "speed_hold_min_accel_mps2",
    "speed_hold_max_accel_mps2",
    "speed_hold_integrator_limit_mps",
    "speed_hold_deadband_mps",
    "speed_hold_drive_axle",
    "speed_hold_brake_axles",
}


def _resolve_project_path(path_value: Optional[str]) -> Optional[Path]: # 파일 경로 해석
    if not path_value:
        return None
    path = Path(path_value) # Path 객체로 변환
    if not path.is_absolute(): # 상대 경로 처리
        path = PROJECT_ROOT / path # 프로젝트 루트 + 상대 경로
    return path


def _fixed_speed_range(kph: float) -> Tuple[float, float]: # km/h -> m/s, (최소, 최대) 최소 = 최대
    speed_mps = float(kph) * KPH_TO_MPS 
    return speed_mps, speed_mps


def _speed_mps(kph: float) -> float: # km/h -> m/s
    return float(kph) * KPH_TO_MPS


def _fixed_range(value: float) -> Tuple[float, float]: # (최소, 최대) 최소 = 최대
    return float(value), float(value)


CURRICULUM_CONFIG = CurriculumConfig(
    enabled=True,
    stages=(
        CurriculumStage(
            name="stage1_straight_high_mu",
            env_overrides={
                "initial_speed_range_mps": _fixed_speed_range(20.0),
                "speed_hold_target_speed_mps": _speed_mps(50.0),
                "road_friction_range": _fixed_range(1.0),
                "target_curvature_max": 0.0,
                "curvature_weave_frequency_range": _fixed_range(0.05),
                "curvature_weave_steering_offset_range": _fixed_range(0.0),
            },
            initial_speed_kph=20.0,
            target_speed_kph=50.0,
            steering_input="Straight",
            road_friction=1.0,
            description="Straight steering input on high-friction road.",
        ),
        CurriculumStage(
            name="stage2_weave_high_mu",
            env_overrides={
                "initial_speed_range_mps": _fixed_speed_range(20.0),
                "speed_hold_target_speed_mps": _speed_mps(50.0),
                "road_friction_range": _fixed_range(1.0),
                "target_curvature_max": 0.05,
                "curvature_weave_frequency_range": (0.01, 0.03),
                "curvature_weave_steering_offset_range": (-0.05, 0.05),
            },
            initial_speed_kph=20.0,
            target_speed_kph=50.0,
            steering_input="Weave profile",
            road_friction=1.0,
            description="Weave profile on high-friction road.",
        ),
        CurriculumStage(
            name="stage3_weave_high_mu_curvature_ramp",
            env_overrides={
                "initial_speed_range_mps": _fixed_speed_range(20.0),
                "speed_hold_target_speed_mps": _speed_mps(50.0),
                "road_friction_range": _fixed_range(1.0),
                "target_curvature_max": 0.08,
                "curvature_weave_frequency_range": (0.01, 0.05),
                "curvature_weave_steering_offset_range": (-0.10, 0.10),
            },
            initial_speed_kph=20.0,
            target_speed_kph=50.0,
            steering_input="Weave profile",
            road_friction=1.0,
            description="High-friction weave with a small increase in curvature and offset range.",
        ),
        CurriculumStage(
            name="stage4_weave_high_mu_frequency_ramp",
            env_overrides={
                "initial_speed_range_mps": _fixed_speed_range(20.0),
                "speed_hold_target_speed_mps": _speed_mps(50.0),
                "road_friction_range": _fixed_range(1.0),
                "target_curvature_max": 0.10,
                "curvature_weave_frequency_range": (0.01, 0.20),
                "curvature_weave_steering_offset_range": (-0.25, 0.25),
            },
            initial_speed_kph=20.0,
            target_speed_kph=50.0,
            steering_input="Weave profile",
            road_friction=1.0,
            description="High-friction weave with a broader frequency range.",
        ),
        CurriculumStage(
            name="stage5_weave_high_mu_offset_ramp",
            env_overrides={
                "initial_speed_range_mps": _fixed_speed_range(20.0),
                "speed_hold_target_speed_mps": _speed_mps(50.0),
                "road_friction_range": _fixed_range(1.0),
                "target_curvature_max": 0.12,
                "curvature_weave_frequency_range": (0.01, 0.50),
                "curvature_weave_steering_offset_range": (-0.50, 0.50),
            },
            initial_speed_kph=20.0,
            target_speed_kph=50.0,
            steering_input="Weave profile",
            road_friction=1.0,
            description="High-friction weave with a moderate steering-offset range.",
        ),
        CurriculumStage(
            name="stage6_weave_high_mu_fast_weave",
            env_overrides={
                "initial_speed_range_mps": _fixed_speed_range(20.0),
                "speed_hold_target_speed_mps": _speed_mps(50.0),
                "road_friction_range": _fixed_range(1.0),
                "target_curvature_max": 0.16,
                "curvature_weave_frequency_range": (0.01, 1.00),
                "curvature_weave_steering_offset_range": (-1.50, 1.50),
            },
            initial_speed_kph=20.0,
            target_speed_kph=50.0,
            steering_input="Weave profile",
            road_friction=1.0,
            description="High-friction weave with faster steering content.",
        ),
        CurriculumStage(
            name="stage7_weave_high_mu_full_range",
            env_overrides={
                "initial_speed_range_mps": _fixed_speed_range(20.0),
                "speed_hold_target_speed_mps": _speed_mps(50.0),
                "road_friction_range": _fixed_range(1.0),
                "target_curvature_max": 0.2,
                "curvature_weave_frequency_range": (0.01, 3.0),
                "curvature_weave_steering_offset_range": (-3.0, 3.0),
            },
            initial_speed_kph=20.0,
            target_speed_kph=50.0,
            steering_input="Weave profile",
            road_friction=1.0,
            description="Original high-friction weave range before reducing road friction.",
        ),
        CurriculumStage(
            name="stage8_weave_mid_mu",
            env_overrides={
                "initial_speed_range_mps": _fixed_speed_range(20.0),
                "speed_hold_target_speed_mps": _speed_mps(50.0),
                "road_friction_range": _fixed_range(0.6),
                "target_curvature_max": 0.2,
                "curvature_weave_frequency_range": (0.01, 3.0),
                "curvature_weave_steering_offset_range": (-3.0, 3.0),
            },
            initial_speed_kph=20.0,
            target_speed_kph=50.0,
            steering_input="Weave profile",
            road_friction=0.6,
            description="Weave profile on medium-friction road.",
        ),
        CurriculumStage(
            name="stage9_weave_low_mu",
            env_overrides={
                "initial_speed_range_mps": _fixed_speed_range(20.0),
                "speed_hold_target_speed_mps": _speed_mps(50.0),
                "road_friction_range": _fixed_range(0.3),
                "target_curvature_max": 0.2,
                "curvature_weave_frequency_range": (0.01, 3.0),
                "curvature_weave_steering_offset_range": (-3.0, 3.0),
            },
            initial_speed_kph=20.0,
            target_speed_kph=50.0,
            steering_input="Weave profile",
            road_friction=0.3,
            description="Weave profile on low-friction road.",
        ),
    ),
)


def compose_env_config(base_env_config: EnvConfig, overrides: Dict[str, Any]) -> EnvConfig: # 기본 학습 환경 설정 전체를 복사, 현재 커리큘럼 단계에서 바꿔야하는 값만 덮어쓰기           
    return replace(base_env_config, **overrides)


def make_environment_config(env_config: EnvConfig) -> EnvironmentConfig: # 학습 설정 객체를 env_config 객체로 변환
    return EnvironmentConfig(
        config_path=env_config.config_path,
        vehicle_model_path=env_config.vehicle_model_path,
        sim_dt=env_config.sim_dt,
        control_dt=env_config.control_dt,
        max_episode_time=env_config.max_episode_time,
        initial_speed_range_mps=env_config.initial_speed_range_mps,
        road_friction_range=env_config.road_friction_range,
        target_curvature_max=env_config.target_curvature_max,
        curvature_weave_frequency_range=env_config.curvature_weave_frequency_range,
        curvature_weave_steering_offset_range=env_config.curvature_weave_steering_offset_range,
        weave_delay=env_config.weave_delay,
        target_reference_delay_s=env_config.target_reference_delay_s,
        initial_state_preroll_time_s=env_config.initial_state_preroll_time_s,
        enable_speed_dependent_steering_scaling=env_config.enable_speed_dependent_steering_scaling,
        speed_dependent_steering_threshold_mps=env_config.speed_dependent_steering_threshold_mps,
        enable_speed_hold_pi=env_config.enable_speed_hold_pi,
        speed_hold_target_speed_mps=env_config.speed_hold_target_speed_mps,
        speed_hold_kp=env_config.speed_hold_kp,
        speed_hold_ki=env_config.speed_hold_ki,
        speed_hold_min_accel_mps2=env_config.speed_hold_min_accel_mps2,
        speed_hold_max_accel_mps2=env_config.speed_hold_max_accel_mps2,
        speed_hold_integrator_limit_mps=env_config.speed_hold_integrator_limit_mps,
        speed_hold_deadband_mps=env_config.speed_hold_deadband_mps,
        speed_hold_drive_axle=env_config.speed_hold_drive_axle,
        speed_hold_brake_axles=env_config.speed_hold_brake_axles,
    )


def serialize_curriculum_config(curriculum_config: CurriculumConfig, total_timesteps: int) -> Dict[str, Any]: # 커리큘럼 설정을 JSON으로 저장할 수 있게 dictionary로 변환
    return {
        "enabled": curriculum_config.enabled,
        "total_timesteps": int(total_timesteps),
        "stages": [asdict(stage) for stage in curriculum_config.stages],
        "notes": [
            "Stage target speed is applied through the speed-hold PI controller; the RL policy still has no longitudinal action.",
            "Weave stages use the reference generator with fixed initial and target speeds.",
        ],
    }


def load_curriculum_resume_state( # 커리큘럼 재개 상태 로딩
    train_config: TrainConfig,
    curriculum_config: CurriculumConfig,
) -> Optional[CurriculumResumeState]:
    state_path = _resolve_project_path(train_config.resume_curriculum_state) # 재개 파일 경로 절대 경로로 변환
    if state_path is None: # 재개 파일 없으면 신규 학습
        return None
    if not state_path.exists(): # 파일 없으면 오류
        raise FileNotFoundError(f"Curriculum resume state file does not exist: {state_path}")
    if not curriculum_config.enabled or not curriculum_config.stages: # 커리큘럼 사용 안하면 재개 상태 무시
        return None

    with state_path.open("r", encoding="utf-8") as fp: # JSON 파일 열고 읽기
        payload = json.load(fp)

    stage_index = int(payload.get("stage_index", 0)) # 재개 상태에서 커리큘럼 단계 인덱스 가져오기, default 0
    stage_index = min(max(stage_index, 0), len(curriculum_config.stages) - 1) # stage_index 범위 제한
    completed_steps = max(int(payload.get("completed_steps", 0)), 0) # 완료된 스텝 수 가져오기, default 0
    stage_completed_steps = max(int(payload.get("stage_completed_steps", 0)), 0) # 완료된 스텝 수 가져오기, default 0
    stage_episode_count = max(int(payload.get("stage_episode_count", 0)), 0) # 완료된 스텝 수 가져오기, default 0

    recent_rewards: List[float] = [] # 보상 저장 리스트
    for value in payload.get("stage_recent_rewards", []): # JSON에 저장된 보상 유효성 검사
        reward = float(value)
        if np.isfinite(reward):
            recent_rewards.append(reward)

    stage = curriculum_config.stages[stage_index] # 보상 리스트가 window 크기보다 크면 마지막 30개만
    if len(recent_rewards) > stage.reward_window_episodes:
        recent_rewards = recent_rewards[-stage.reward_window_episodes:] # recent_rewards[-30:], 리스트 끝에서 30개 선택

    return CurriculumResumeState(
        source_path=state_path,
        stage_index=stage_index,
        completed_steps=completed_steps,
        stage_completed_steps=stage_completed_steps,
        stage_episode_count=stage_episode_count,
        stage_recent_rewards=tuple(recent_rewards),
    )


def validate_curriculum_config( # 설정 유효성 검사
    curriculum_config: CurriculumConfig,
    base_env_config: EnvConfig,
    total_timesteps: int,
    resume_state: Optional[CurriculumResumeState] = None,
) -> None:
    if not curriculum_config.enabled: # 커리큘럼 사용 X
        return
    if not curriculum_config.stages: # 커리큘럼 단계 없을 때
        raise ValueError("Curriculum is enabled, but no stages are configured.")
    
    for index, stage in enumerate(curriculum_config.stages): # 각 단계를 인덱스와 함게 검사
        if not stage.name: # 이름이 빈 문자열일 때
            raise ValueError(f"Curriculum stage {index} must have a name.")
        unknown_fields = set(stage.env_overrides) - set(base_env_config.__dataclass_fields__) # 오타 확인
        if unknown_fields: 
            raise ValueError(f"Curriculum stage {stage.name!r} has unknown env fields: {sorted(unknown_fields)}")
        unsafe_fields = set(stage.env_overrides) - CURRICULUM_MUTABLE_ENV_FIELDS # 바꿀 수 없는 조건 바꿨을 때
        if unsafe_fields:
            raise ValueError(
                f"Curriculum stage {stage.name!r} uses fields that cannot be updated online: {sorted(unsafe_fields)}"
            )
        compose_env_config(base_env_config, stage.env_overrides) 


def resolve_initial_env_config( # 초기 환경 설정
    base_env_config: EnvConfig,
    curriculum_config: CurriculumConfig,
    resume_state: Optional[CurriculumResumeState] = None,
) -> EnvConfig:
    if curriculum_config.enabled and curriculum_config.stages:
        stage_index = 0 if resume_state is None else resume_state.stage_index # 신규 학습은 stage 1부터
        return compose_env_config(base_env_config, curriculum_config.stages[stage_index].env_overrides) 
    return base_env_config


def apply_env_config_to_vec_env(vec_env: VecMonitor, env_config: EnvConfig) -> None:
    vec_env.env_method("update_config", make_environment_config(env_config))


class CurriculumCallback(BaseCallback):
    STATE_WRITE_INTERVAL_STEPS = 5_000

    def __init__(
        self,
        *,
        train_env: VecMonitor,
        eval_env: VecMonitor,
        base_env_config: EnvConfig,
        curriculum_config: CurriculumConfig,
        total_timesteps: int,
        run_dir: Path,
        resume_state: Optional[CurriculumResumeState] = None,
        stage_log_interval_sec: float = 30.0,
        stage_log_interval_steps: int = 1_000,
    ) -> None:
        super().__init__(verbose=0)
        self.train_env = train_env
        self.eval_env = eval_env
        self.base_env_config = base_env_config
        self.curriculum_config = curriculum_config
        self.total_timesteps = max(int(total_timesteps), 1)
        self.run_dir = run_dir
        self.state_path = run_dir / "curriculum_state.json"
        self.resume_state = resume_state
        self.stage_log_interval_sec = max(float(stage_log_interval_sec), 1.0)
        self.stage_log_interval_steps = max(int(stage_log_interval_steps), 1)
        self.last_stage_status_print_time = 0.0
        self.last_stage_status_print_num_timesteps = 0
        self.current_stage_index = -1
        self.current_stage_env_config = base_env_config
        self.stage_start_num_timesteps = 0
        self.last_state_write_num_timesteps = 0
        self.stage_episode_count = 0
        self.stage_recent_rewards: List[float] = []

    def _on_training_start(self) -> None:
        if self.resume_state is None:
            self._apply_stage(0, reason="training_start")
            return

        loaded_model_steps = max(int(self.model.num_timesteps), 0)
        restored_stage_steps = self.resume_state.stage_completed_steps
        if loaded_model_steps > self.resume_state.completed_steps:
            restored_stage_steps += loaded_model_steps - self.resume_state.completed_steps
        restored_stage_steps = min(max(int(restored_stage_steps), 0), loaded_model_steps)
        self._apply_stage(
            self.resume_state.stage_index,
            reason=f"resume_curriculum_state:{self.resume_state.source_path}",
            reset_stage_progress=False,
            stage_completed_steps=restored_stage_steps,
            stage_episode_count=self.resume_state.stage_episode_count,
            stage_recent_rewards=list(self.resume_state.stage_recent_rewards),
        )

    def _on_step(self) -> bool:
        if self.current_stage_index < 0:
            return True
        stage = self.curriculum_config.stages[self.current_stage_index]
        self._collect_finished_episode_rewards(stage)

        stage_steps = self._stage_steps()
        self.logger.record("curriculum/stage_index", float(self.current_stage_index + 1))
        self.logger.record("curriculum/stage_steps", float(stage_steps))
        self.logger.record("curriculum/road_friction", float(stage.road_friction))
        self.logger.record("curriculum/initial_speed_kph", float(stage.initial_speed_kph))
        self.logger.record("curriculum/target_speed_kph", float(stage.target_speed_kph))
        recent_mean_reward = self._recent_mean_reward(stage)
        if recent_mean_reward is not None:
            self.logger.record("curriculum/recent_mean_reward", float(recent_mean_reward))

        should_advance = self._reward_gate_passes(stage)

        if should_advance:
            if self.current_stage_index < len(self.curriculum_config.stages) - 1:
                reason = "reward_gate" if stage.reward_gate_enabled else "stage_step_budget"
                self._apply_stage(self.current_stage_index + 1, reason=reason)
            else:
                self._write_state_file()
                self._maybe_print_stage_status(stage)
                return True

        self._maybe_write_state_file()
        self._maybe_print_stage_status(stage)
        return True

    def _stage_steps(self) -> int:
        return max(int(self.model.num_timesteps) - self.stage_start_num_timesteps, 0)

    def _collect_finished_episode_rewards(self, stage: CurriculumStage) -> None:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            episode_info = info.get("episode")
            if not isinstance(episode_info, dict) or "r" not in episode_info:
                continue
            reward = float(episode_info["r"])
            if not np.isfinite(reward):
                continue
            self.stage_episode_count += 1
            self.stage_recent_rewards.append(reward)
            if len(self.stage_recent_rewards) > stage.reward_window_episodes:
                self.stage_recent_rewards = self.stage_recent_rewards[-stage.reward_window_episodes:]

    def _recent_mean_reward(self, stage: CurriculumStage) -> Optional[float]:
        if len(self.stage_recent_rewards) < stage.reward_window_episodes:
            return None
        return float(np.mean(self.stage_recent_rewards[-stage.reward_window_episodes:]))

    def _reward_gate_passes(self, stage: CurriculumStage) -> bool:
        recent_mean_reward = self._recent_mean_reward(stage)
        return (
            self.stage_episode_count >= stage.min_episodes
            and recent_mean_reward is not None
            and recent_mean_reward > stage.reward_threshold
        )

    def get_status_snapshot(self) -> Dict[str, Any]:
        if self.current_stage_index < 0:
            return {}
        stage = self.curriculum_config.stages[self.current_stage_index]
        return {
            "stage_index": self.current_stage_index + 1,
            "stage_count": len(self.curriculum_config.stages),
            "stage_name": stage.name,
            "stage_steps": self._stage_steps(),
            "stage_episode_count": self.stage_episode_count,
            "min_episodes": stage.min_episodes,
            "reward_gate_enabled": stage.reward_gate_enabled,
            "reward_window_episodes": stage.reward_window_episodes,
            "recent_mean_reward": self._recent_mean_reward(stage),
            "reward_threshold": stage.reward_threshold,
        }

    def _maybe_print_stage_status(self, stage: CurriculumStage) -> None:
        now = time.perf_counter()
        current_steps = int(self.model.num_timesteps)
        elapsed_sec = now - self.last_stage_status_print_time
        elapsed_steps = current_steps - self.last_stage_status_print_num_timesteps
        if elapsed_sec < self.stage_log_interval_sec and elapsed_steps < self.stage_log_interval_steps:
            return

        stage_steps = self._stage_steps()
        recent_mean_reward = self._recent_mean_reward(stage)
        recent_reward_text = "nan" if recent_mean_reward is None else f"{recent_mean_reward:.3f}"
        print(
            "[Curriculum Progress] "
            f"stage={self.current_stage_index + 1}/{len(self.curriculum_config.stages)} "
            f"name={stage.name} | "
            f"stage_steps={stage_steps} | "
            f"global_steps={current_steps} | "
            f"episodes={self.stage_episode_count} | recent_mean_reward={recent_reward_text} | "
            f"initial_speed={stage.initial_speed_kph:.1f}kph | "
            f"target_speed={stage.target_speed_kph:.1f}kph | "
            f"mu={stage.road_friction:.2f} | steering={stage.steering_input}",
            flush=True,
        )
        self.last_stage_status_print_time = now
        self.last_stage_status_print_num_timesteps = current_steps

    def _apply_stage(
        self,
        stage_index: int,
        *,
        reason: str,
        reset_stage_progress: bool = True,
        stage_completed_steps: int = 0,
        stage_episode_count: int = 0,
        stage_recent_rewards: Optional[List[float]] = None,
    ) -> None:
        stage = self.curriculum_config.stages[stage_index]
        stage_env_config = compose_env_config(self.base_env_config, stage.env_overrides)
        apply_env_config_to_vec_env(self.train_env, stage_env_config)
        apply_env_config_to_vec_env(self.eval_env, stage_env_config)
        current_steps = int(self.model.num_timesteps)
        self.current_stage_index = stage_index
        self.current_stage_env_config = stage_env_config
        if reset_stage_progress:
            self.stage_start_num_timesteps = current_steps
            self.stage_episode_count = 0
            self.stage_recent_rewards = []
        else:
            restored_stage_steps = min(max(int(stage_completed_steps), 0), max(current_steps, 0))
            self.stage_start_num_timesteps = current_steps - restored_stage_steps
            self.stage_episode_count = max(int(stage_episode_count), 0)
            self.stage_recent_rewards = [
                float(value)
                for value in (stage_recent_rewards or [])
                if np.isfinite(float(value))
            ]
            if len(self.stage_recent_rewards) > stage.reward_window_episodes:
                self.stage_recent_rewards = self.stage_recent_rewards[-stage.reward_window_episodes:]
        self.last_state_write_num_timesteps = current_steps
        self.last_stage_status_print_time = time.perf_counter()
        self.last_stage_status_print_num_timesteps = current_steps
        self._write_state_file()
        print(
            "[Curriculum] "
            f"stage={stage_index + 1}/{len(self.curriculum_config.stages)} "
            f"name={stage.name} | "
            f"stage_steps={self._stage_steps()} | "
            f"initial_speed={stage.initial_speed_kph:.1f}kph | "
            f"target_speed={stage.target_speed_kph:.1f}kph | "
            f"steering={stage.steering_input} | mu={stage.road_friction:.2f} | "
            f"reason={reason}",
            flush=True,
        )

    def _write_state_file(self) -> None:
        if self.current_stage_index < 0:
            return
        stage = self.curriculum_config.stages[self.current_stage_index]
        payload = {
            "updated_at": datetime.now().isoformat(),
            "stage_index": self.current_stage_index,
            "stage_name": stage.name,
            "completed_steps": int(self.model.num_timesteps),
            "stage_completed_steps": self._stage_steps(),
            "stage_episode_count": self.stage_episode_count,
            "stage_recent_rewards": [float(value) for value in self.stage_recent_rewards],
            "stage_recent_mean_reward": self._recent_mean_reward(stage),
            "env_config": asdict(self.current_stage_env_config),
            "stage": asdict(stage),
        }
        with self.state_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        self.last_state_write_num_timesteps = int(self.model.num_timesteps)

    def _maybe_write_state_file(self) -> None:
        if (int(self.model.num_timesteps) - self.last_state_write_num_timesteps) < self.STATE_WRITE_INTERVAL_STEPS:
            return
        self._write_state_file()

    def _on_training_end(self) -> None:
        self._write_state_file()
