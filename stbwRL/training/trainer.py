from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

import numpy as np

from stbwRL.env_config import EnvironmentConfig
from stbwRL.custom_env import CustomEnv
from stbwRL.training.curriculum import (
    CURRICULUM_CONFIG,
    CurriculumCallback,
    CurriculumConfig,
    CurriculumResumeState,
    load_curriculum_resume_state,
    resolve_initial_env_config,
    serialize_curriculum_config,
    validate_curriculum_config,
)
from stbwRL.training.custom_sac_network import SACPolicy
from stbwRL.training.train_config import ENV_CONFIG, TRAIN_CONFIG, PROJECT_ROOT, EnvConfig, TrainConfig

try:
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
except ImportError as exc:
    raise ImportError(
        "stable_baselines3, torch, and tensorboard support are required to run training. "
        "Install them with `pip install stable-baselines3 tensorboard`."
    ) from exc


def format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def resolve_project_path(path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_run_dir(train_config: TrainConfig) -> Path:
    output_dir = resolve_project_path(train_config.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = train_config.run_name or f"sac_lateral_{timestamp}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_environment_config(env_config: EnvConfig) -> EnvironmentConfig:
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


class TrainingTimeCallback(BaseCallback):
    def __init__(self, total_timesteps: int, log_interval_sec: float, continue_until_interrupt: bool = False) -> None:
        super().__init__(verbose=0)
        self.total_timesteps = max(int(total_timesteps), 1)
        self.log_interval_sec = float(log_interval_sec)
        self.continue_until_interrupt = bool(continue_until_interrupt)
        self.start_time = 0.0
        self.last_log_time = 0.0
        self.start_num_timesteps = 0

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.start_num_timesteps = int(self.model.num_timesteps)

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if now - self.last_log_time >= self.log_interval_sec:
            self._print_progress(now)
            self.last_log_time = now
        return True

    def _on_training_end(self) -> None:
        self._print_progress(time.perf_counter(), is_final=True)

    def _print_progress(self, now: float, is_final: bool = False) -> None:
        elapsed = max(now - self.start_time, 0.0)
        completed_steps = max(int(self.model.num_timesteps) - self.start_num_timesteps, 0)
        steps_per_sec = completed_steps / elapsed if elapsed > 1e-9 else 0.0
        prefix = "[Training End]" if is_final else "[Training Progress]"
        if self.continue_until_interrupt:
            print(
                f"{prefix} steps={completed_steps} | elapsed={format_duration(elapsed)} "
                f"| speed={steps_per_sec:7.1f} step/s | mode=until_interrupt"
            )
            return
        progress = min(max(completed_steps / self.total_timesteps, 0.0), 1.0)
        remaining_steps = max(self.total_timesteps - completed_steps, 0)
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 1e-9 else float("inf")
        eta_text = format_duration(eta_seconds) if eta_seconds != float("inf") else "--:--:--"
        print(
            f"{prefix} steps={completed_steps}/{self.total_timesteps} "
            f"({progress * 100.0:5.1f}%) | elapsed={format_duration(elapsed)} "
            f"| speed={steps_per_sec:7.1f} step/s | eta={eta_text}"
        )


class EpisodeRewardCallback(BaseCallback):
    def __init__(self, expected_episode_steps: Optional[int] = None) -> None:
        super().__init__(verbose=0)
        self.expected_episode_steps = expected_episode_steps
        self.episode_rewards = np.zeros(0, dtype=np.float64)
        self.episode_lengths = np.zeros(0, dtype=np.int64)
        self.completed_episodes = 0

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.training_env, "num_envs", 1))
        self.episode_rewards = np.zeros(n_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(n_envs, dtype=np.int64)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True

        rewards_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
        dones_arr = np.asarray(dones, dtype=bool).reshape(-1)
        if self.episode_rewards.size != rewards_arr.size:
            self.episode_rewards = np.zeros(rewards_arr.size, dtype=np.float64)
            self.episode_lengths = np.zeros(rewards_arr.size, dtype=np.int64)

        self.episode_rewards += rewards_arr
        self.episode_lengths += 1

        infos = self.locals.get("infos") or []
        for env_index, done in enumerate(dones_arr):
            if not done:
                continue
            total_reward = float(self.episode_rewards[env_index])
            episode_length = int(self.episode_lengths[env_index])
            if env_index < len(infos):
                episode_info = infos[env_index].get("episode")
                if isinstance(episode_info, dict):
                    total_reward = float(episode_info.get("r", total_reward))
                    episode_length = int(episode_info.get("l", episode_length))

            avg_reward = total_reward / max(episode_length, 1)
            self.completed_episodes += 1
            expected_text = f"/{self.expected_episode_steps}" if self.expected_episode_steps is not None else ""
            print(
                "[Episode] "
                f"episode={self.completed_episodes} | global_step={int(self.model.num_timesteps)} | "
                f"steps={episode_length}{expected_text} | total_reward={total_reward:.3f} | "
                f"avg_reward={avg_reward:.6f}"
            )
            self.logger.record("episode/total_reward", total_reward)
            self.logger.record("episode/avg_reward", avg_reward)
            self.logger.record("episode/length", float(episode_length))
            self.episode_rewards[env_index] = 0.0
            self.episode_lengths[env_index] = 0
        return True


class LateralTrackingDebugCallback(BaseCallback):
    def __init__(
        self,
        log_interval_sec: float,
        curriculum_callback: Optional[CurriculumCallback] = None,
    ) -> None:
        super().__init__(verbose=0)
        self.log_interval_sec = float(log_interval_sec)
        self.curriculum_callback = curriculum_callback
        self.last_log_time = 0.0

    def _on_training_start(self) -> None:
        self.last_log_time = time.perf_counter()

    @staticmethod
    def _fmt(value: Optional[float], precision: int = 4) -> str:
        if value is None:
            return "nan"
        return f"{value:.{precision}f}"

    def _format_curriculum_fields(self) -> List[str]:
        if self.curriculum_callback is None:
            return []
        status = self.curriculum_callback.get_status_snapshot()
        if not status:
            return []

        reward_window_episodes = int(status["reward_window_episodes"])
        gate_label = f"recent_{reward_window_episodes}_reward"
        gate_text = "disabled"
        if bool(status["reward_gate_enabled"]):
            gate_text = (
                f"{gate_label}="
                f"{self._fmt(status['recent_mean_reward'], 3)}/"
                f"{self._fmt(float(status['reward_threshold']), 3)}"
            )

        return [
            f"stage={int(status['stage_index'])}/{int(status['stage_count'])} {status['stage_name']}",
            f"stage_steps={int(status['stage_steps'])}",
            f"episodes={int(status['stage_episode_count'])}/{int(status['min_episodes'])}",
            f"gate={gate_text}",
        ]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True

        def _mean_info(key: str, transform: Optional[Callable[[float], float]] = None) -> Optional[float]:
            values = []
            for info in infos:
                value = info.get(key)
                if value is None:
                    continue
                value = float(value)
                if transform is not None:
                    value = float(transform(value))
                if np.isfinite(value):
                    values.append(value)
            if not values:
                return None
            return float(np.mean(values))

        metrics = {
            "tracking/target_curvature": _mean_info("target_curvature"),
            "tracking/current_curvature": _mean_info("curvature"),
            "tracking/curvature_error": _mean_info("curvature_error"),
            "tracking/abs_curvature_error": _mean_info("curvature_error", abs),
            "stability/beta": _mean_info("beta"),
            "stability/abs_beta": _mean_info("beta", abs),
            "stability/ay": _mean_info("ay"),
            "reward/reward_track": _mean_info("reward_track"),
            "reward/reward_slip": _mean_info("reward_slip"),
            "reward/reward_used": _mean_info("reward_used"),
            "reward/reward_total": _mean_info("reward_total"),
        }
        for name, value in metrics.items():
            if value is not None:
                self.logger.record(name, value)

        now = time.perf_counter()
        if now - self.last_log_time < self.log_interval_sec:
            return True

        fields = self._format_curriculum_fields()
        fields.extend(
            [
                f"kappa_ref={self._fmt(metrics['tracking/target_curvature'])}",
                f"kappa={self._fmt(metrics['tracking/current_curvature'])}",
                f"kappa_err={self._fmt(metrics['tracking/curvature_error'])}",
                f"reward={self._fmt(metrics['reward/reward_total'], 3)}",
            ]
        )
        print("[Tracking] " + " | ".join(fields))
        self.last_log_time = now
        return True


def make_vec_env(env_config: EnvConfig, seed: int) -> VecMonitor:
    environment_config = make_environment_config(env_config)

    def _init() -> CustomEnv:
        env = CustomEnv(environment_config)
        env.reset(seed=seed)
        return env

    return VecMonitor(DummyVecEnv([_init]))


def resolve_device(requested_device: str) -> str:
    device = requested_device.strip().lower()
    print(f"[Python] executable={sys.executable}")
    print(f"[Torch] version={torch.__version__}")
    print(f"[Torch] torch.version.cuda={torch.version.cuda}")
    print(f"[Torch] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Torch] cuda_device_count={torch.cuda.device_count()}")
        print(f"[Torch] cuda_device_name={torch.cuda.get_device_name(0)}")

    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TRAIN_CONFIG.device is set to CUDA, but torch.cuda.is_available() is False."
            )
        resolved = requested_device
    elif device == "cpu":
        resolved = "cpu"
    else:
        raise ValueError(f"Unsupported device setting: {requested_device}")

    print(f"[Torch] requested_device={requested_device}, resolved_device={resolved}")
    return resolved


def create_model(train_config: TrainConfig, train_env: VecMonitor, tensorboard_dir: Path) -> SAC:
    resolved_device = resolve_device(train_config.device)
    resume_model_path = resolve_project_path(train_config.resume_model)
    resume_replay_buffer_path = resolve_project_path(train_config.resume_replay_buffer)

    if resume_model_path:
        print(f"[Resume] model={resume_model_path}")
        model = SAC.load(str(resume_model_path), env=train_env, device=resolved_device)
        model.tensorboard_log = str(tensorboard_dir)
    else:
        model = SAC(
            policy=SACPolicy,
            env=train_env,
            learning_rate=train_config.learning_rate,
            buffer_size=train_config.buffer_size,
            learning_starts=train_config.learning_starts,
            batch_size=train_config.batch_size,
            tau=train_config.tau,
            gamma=train_config.gamma,
            train_freq=train_config.train_freq,
            gradient_steps=train_config.gradient_steps,
            ent_coef="auto",
            tensorboard_log=str(tensorboard_dir),
            device=resolved_device,
            seed=train_config.seed,
            verbose=1,
        )

    if resume_replay_buffer_path:
        print(f"[Resume] replay_buffer={resume_replay_buffer_path}")
        model.load_replay_buffer(str(resume_replay_buffer_path))
    return model


def build_callbacks(
    run_dir: Path,
    train_env: VecMonitor,
    eval_env: VecMonitor,
    env_config: EnvConfig,
    train_config: TrainConfig,
    curriculum_config: CurriculumConfig,
    curriculum_resume_state: Optional[CurriculumResumeState] = None,
) -> CallbackList:
    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    best_dir = run_dir / "best_model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(train_config.save_freq, 1),
        save_path=str(checkpoint_dir),
        name_prefix="sac_lateral",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_dir),
        eval_freq=max(train_config.eval_freq, 1),
        n_eval_episodes=max(train_config.n_eval_episodes, 1),
        deterministic=True,
        render=False,
    )
    expected_episode_steps = (
        int(round(env_config.max_episode_time / env_config.control_dt))
        if env_config.control_dt > 0.0
        else None
    )
    callbacks: List[BaseCallback] = []
    curriculum_callback: Optional[CurriculumCallback] = None
    if curriculum_config.enabled and curriculum_config.stages:
        curriculum_callback = CurriculumCallback(
            train_env=train_env,
            eval_env=eval_env,
            base_env_config=env_config,
            curriculum_config=curriculum_config,
            total_timesteps=train_config.timesteps,
            run_dir=run_dir,
            resume_state=curriculum_resume_state,
            stage_log_interval_sec=train_config.progress_log_interval_sec,
            stage_log_interval_steps=train_config.stage_log_interval_steps,
        )
        callbacks.append(curriculum_callback)
    callbacks.extend([
        checkpoint_callback,
        eval_callback,
        TrainingTimeCallback(
            total_timesteps=train_config.timesteps,
            log_interval_sec=train_config.progress_log_interval_sec,
            continue_until_interrupt=train_config.continue_until_interrupt,
        ),
        EpisodeRewardCallback(expected_episode_steps=expected_episode_steps),
        LateralTrackingDebugCallback(
            log_interval_sec=train_config.tracking_log_interval_sec,
            curriculum_callback=curriculum_callback,
        ),
    ])
    return CallbackList(callbacks)


def save_run_metadata(
    run_dir: Path,
    env_config: EnvConfig,
    train_config: TrainConfig,
    curriculum_config: CurriculumConfig,
) -> None:
    metadata = {
        "created_at": datetime.now().isoformat(),
        "env_config": asdict(env_config),
        "train_config": asdict(train_config),
        "curriculum": serialize_curriculum_config(curriculum_config, train_config.timesteps),
        "observation_order": list(EnvironmentConfig().observation_names),
        "action": "front road wheel steering angular acceleration command, normalized to [-1, 1]",
    }
    with (run_dir / "train_config.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def save_train_summary(run_dir: Path, total_trained_steps: int) -> None:
    summary = {
        "finished_at": datetime.now().isoformat(),
        "total_trained_steps": total_trained_steps,
        "final_model": str(run_dir / "final_model.zip"),
        "final_replay_buffer": str(run_dir / "final_replay_buffer.pkl"),
    }
    with (run_dir / "train_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def validate_train_config(train_config: TrainConfig) -> None:
    if train_config.timesteps <= 0:
        raise ValueError("TRAIN_CONFIG.timesteps must be positive.")
    if train_config.batch_size <= 0:
        raise ValueError("TRAIN_CONFIG.batch_size must be positive.")
    if train_config.buffer_size <= 0:
        raise ValueError("TRAIN_CONFIG.buffer_size must be positive.")


def validate_env_config(env_config: EnvConfig) -> None:
    if env_config.control_dt <= 0.0:
        raise ValueError("ENV_CONFIG.control_dt must be positive.")
    if env_config.sim_dt <= 0.0:
        raise ValueError("ENV_CONFIG.sim_dt must be positive.")
    if env_config.max_episode_time <= 0.0:
        raise ValueError("ENV_CONFIG.max_episode_time must be positive.")


def main() -> int:
    validate_env_config(ENV_CONFIG)
    validate_train_config(TRAIN_CONFIG)
    curriculum_resume_state = load_curriculum_resume_state(TRAIN_CONFIG, CURRICULUM_CONFIG)
    validate_curriculum_config(CURRICULUM_CONFIG, ENV_CONFIG, TRAIN_CONFIG.timesteps, curriculum_resume_state)

    run_dir = build_run_dir(TRAIN_CONFIG)
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(run_dir, ENV_CONFIG, TRAIN_CONFIG, CURRICULUM_CONFIG)

    initial_env_config = resolve_initial_env_config(ENV_CONFIG, CURRICULUM_CONFIG, curriculum_resume_state)
    train_env = make_vec_env(initial_env_config, TRAIN_CONFIG.seed)
    eval_env = make_vec_env(initial_env_config, TRAIN_CONFIG.seed + 10_000)
    model = create_model(TRAIN_CONFIG, train_env, tensorboard_dir)
    callbacks = build_callbacks(
        run_dir,
        train_env,
        eval_env,
        ENV_CONFIG,
        TRAIN_CONFIG,
        CURRICULUM_CONFIG,
        curriculum_resume_state,
    )

    total_timesteps = int(1_000_000_000_000 if TRAIN_CONFIG.continue_until_interrupt else TRAIN_CONFIG.timesteps)
    if TRAIN_CONFIG.continue_until_interrupt:
        print(f"\n[Training] mode=until_interrupt | sb3_total_timesteps={total_timesteps}")
    else:
        print(f"\n[Training] timesteps={TRAIN_CONFIG.timesteps}")

    interrupted_by_user = False
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=TRAIN_CONFIG.log_interval,
            reset_num_timesteps=False,
            tb_log_name=run_dir.name,
        )
    except KeyboardInterrupt:
        interrupted_by_user = True
        print("\n[Training] interrupted_by_user=True | saving_latest_state=True")

    model.save(str(run_dir / "final_model"))
    model.save_replay_buffer(str(run_dir / "final_replay_buffer.pkl"))
    save_train_summary(run_dir, int(model.num_timesteps))

    print("\n[Done]" if not interrupted_by_user else "\n[Stopped]")
    print(f"run_dir: {run_dir}")
    print(f"final_model: {run_dir / 'final_model.zip'}")
    print(f"final_replay_buffer: {run_dir / 'final_replay_buffer.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
