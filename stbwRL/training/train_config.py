from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from stbwRL.env_config import EnvironmentConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENVIRONMENT_DEFAULTS = EnvironmentConfig()


@dataclass
class EnvConfig:
    config_path: str = "stbw"
    vehicle_model_path: str = str(PROJECT_ROOT / "vehicle_sim")
    sim_dt: float = 0.001
    control_dt: float = 0.02
    max_episode_time: float = 30.0
    initial_speed_range_mps: Tuple[float, float] = (15.0, 15.0)
    road_friction_range: Tuple[float, float] = (1.0, 1.0)
    target_curvature_max: float = 0.2
    curvature_weave_frequency_range: Tuple[float, float] = (0.01, 3.0)
    curvature_weave_steering_offset_range: Tuple[float, float] = (-3.0, 3.0)
    weave_delay: float = 1.0
    target_reference_delay_s: float = 0.03
    initial_state_preroll_time_s: float = 0.5
    enable_speed_dependent_steering_scaling: bool = True
    speed_dependent_steering_threshold_mps: float = 8.0
    enable_speed_hold_pi: bool = True
    speed_hold_target_speed_mps: Optional[float] = None
    speed_hold_kp: float = _ENVIRONMENT_DEFAULTS.speed_hold_kp
    speed_hold_ki: float = _ENVIRONMENT_DEFAULTS.speed_hold_ki
    speed_hold_min_accel_mps2: float = -6.0
    speed_hold_max_accel_mps2: float = 4.0
    speed_hold_integrator_limit_mps: float = 8.0
    speed_hold_deadband_mps: float = 0.02
    speed_hold_drive_axle: str = "R"
    speed_hold_brake_axles: Tuple[str, ...] = ("F", "R")


@dataclass
class TrainConfig:
    run_name: Optional[str] = None
    output_dir: str = "runs"
    seed: int = 42
    device: str = "auto"
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 256
    tau: float = 1e-3
    gamma: float = 0.96
    train_freq: int = 50
    gradient_steps: int = 50
    eval_freq: int = 20_000
    save_freq: int = 50_000
    n_eval_episodes: int = 5
    log_interval: int = 10
    progress_log_interval_sec: float = 30.0
    stage_log_interval_steps: int = 1_000
    tracking_log_interval_sec: float = 5.0
    timesteps: int = 90_000
    continue_until_interrupt: bool = True

    resume_model: Optional[str] = None
    resume_replay_buffer: Optional[str] = None
    resume_curriculum_state: Optional[str] = None
    resume_model: Optional[str] = "runs/sac_lateral_20260715_152708/final_model.zip"
    resume_replay_buffer: Optional[str] = "runs/sac_lateral_20260715_152708/final_replay_buffer.pkl"
    resume_curriculum_state: Optional[str] = "runs/sac_lateral_20260715_152708/curriculum_state.json"


ENV_CONFIG = EnvConfig()
TRAIN_CONFIG = TrainConfig()
