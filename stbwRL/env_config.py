from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import math


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
OUTPUT_ROOT = PACKAGE_ROOT / "output"

OBSERVATION_NAMES: Tuple[str, ...] = (
    "vx",
    "vy",
    "steer",
    "steer_dot",
    "yaw_rate",
    "ax",
    "ay",
    "target_curvature",
    "target_curvature_dot",
    "target_lateral_accel",
    "target_lateral_accel_dot",
)

ACTION_NAMES: Tuple[str, ...] = (
    "steer_ddot",
)

OBS_AVG_VALUE: Tuple[float, ...] = (15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
OBS_STD_VALUE: Tuple[float, ...] = (15.0, 10.0, 0.6, 1.25, 1.5, 4.0, 6.0, 0.25, 0.8, 10.0, 40.0)


@dataclass
class EnvironmentConfig:
    config_path: str = "stbw"
    vehicle_model_path: str = str(PROJECT_ROOT / "vehicle_sim")
    sim_dt: float = 0.001
    control_dt: float = 0.02
    max_episode_time: float = 30.0
    gamma: float = 0.96

    obs_avg_value: Tuple[float, ...] = OBS_AVG_VALUE
    obs_std_value: Tuple[float, ...] = OBS_STD_VALUE
    observation_names: Tuple[str, ...] = OBSERVATION_NAMES
    action_names: Tuple[str, ...] = ACTION_NAMES

    wheelbase: float = 2.97
    steering_ratio: float = 17.21
    ay_max_for_target_curvature: float = 6.0
    low_speed_no_saturation_threshold_mps: float = 1.0
    curvature_denominator_eps: float = 1e-6
    enable_speed_dependent_steering_scaling: bool = True
    speed_dependent_steering_threshold_mps: float = 8.0

    enable_speed_hold_pi: bool = True
    speed_hold_target_speed_mps: Optional[float] = None
    speed_hold_kp: float = 0.48
    speed_hold_ki: float = 0.0
    speed_hold_min_accel_mps2: float = -6.0
    speed_hold_max_accel_mps2: float = 4.0
    speed_hold_integrator_limit_mps: float = 8.0
    speed_hold_deadband_mps: float = 0.02
    speed_hold_drive_axle: str = "R" # 후륜구동
    speed_hold_brake_axles: Tuple[str, ...] = ("F", "R")

    steer_ddot_max: float = 240.0 * math.pi / 180.0
    steer_dot_max: float = 60.0 * math.pi / 180.0
    steer_max: float = 30.0 * math.pi / 180.0

    initial_speed_range_mps: Tuple[float, float] = (15.0, 15.0)
    road_friction_range: Tuple[float, float] = (1.0, 1.0)
    target_curvature_max: float = 0.2
    fixed_target_curvature_m_inv: Optional[float] = None
    curvature_weave_frequency_range: Tuple[float, float] = (0.01, 0.35)
    curvature_weave_steering_offset_range: Tuple[float, float] = (-3.0, 3.0)
    weave_delay: float = 1.0
    target_reference_delay_s: float = 0.03
    initial_state_preroll_time_s: float = 0.5

    K_kappa: float = 12500.0
    K_ay: float = 1.25
    K_slip: float = 10.0
    w_slip: float = 0.5
    b_track: float = 1.0
    b_slip: float = 1.5
    R_base: float = 1.0
    P_terminal: float = 1.1
    beta_warn: float = math.pi / 16.0
    beta_term: float = math.pi / 3.0
    curvature_error_lateral_accel_term_mps2: float = 3.0
    curvature_error_lateral_accel_term_by_mu: Tuple[Tuple[float, float], ...] = (
        (0.3, 3.0),
        (0.6, 6.0),
        (1.0, 9.0),
    )
    longitudinal_input_block_beta_rad: float = math.pi / 20.0
    longitudinal_input_block_min_vx_mps: float = 2.0

    @property
    def obs_dim(self) -> int:
        return len(tuple(self.observation_names))

    @property
    def action_dim(self) -> int:
        return len(tuple(self.action_names))

    @property
    def max_steps_per_episode(self) -> int:
        return int(math.floor(self.max_episode_time / self.control_dt))


@dataclass
class SACConfig:
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    tau: float = 1e-3
    gamma: float = 0.96
    train_freq_steps: int = 50
    gradient_steps: int = 100
    learning_starts: int = 100
    net_arch: Tuple[int, ...] = (128, 128)
    paper_net_arch: Tuple[int, ...] = (256, 256)


@dataclass
class FilePath:
    model_dir: str = str(OUTPUT_ROOT / "models")
    model_path: str = str(OUTPUT_ROOT / "models" / "sac_lateral_curvature")
    replay_buffer_path: str = str(OUTPUT_ROOT / "models" / "sac_lateral_curvature_replay_buffer")
    log_dir: str = str(OUTPUT_ROOT / "logs")
    history_dir: str = str(OUTPUT_ROOT / "history")
