from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

from stbwRL.custom_env import CustomEnv
from stbwRL.env_config import EnvironmentConfig

try:
    from stable_baselines3 import SAC
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "stable_baselines3 is required to run stbwRL/evaluate.py. "
        "Install it with `pip install stable-baselines3`."
    ) from exc


STEERING_LIMIT_HIT_RATIO_THRESHOLD = 0.999
PLOT_SUPTITLE_FONT_SIZE = 22
PLOT_TITLE_FONT_SIZE = 19
PLOT_AXIS_LABEL_FONT_SIZE = 17
PLOT_TICK_LABEL_FONT_SIZE = 16
PLOT_LEGEND_FONT_SIZE = 8


@dataclass
class EvalConfig:
    model_path: Optional[str] = None
    device: str = "auto"
    seed: int = 30
    max_episode_time: Optional[float] = None
    output_dir: Optional[str] = None
    show_plot: bool = True
    save_plots: bool = False
    save_csv: bool = True
    save_summary: bool = True
    deterministic: bool = True
    plot_line_width_scale: float = 1.0
    fixed_action: Optional[Tuple[float, ...]] = None
    evaluation_warmup_time_s: float = 0.0
    initial_speed_kph: Optional[float] = None
    initial_speed_mps: Optional[float] = None
    target_speed_kph: Optional[float] = None
    target_speed_mps: Optional[float] = None
    target_curvature_m_inv: Optional[float] = None
    target_curvature_max_m_inv: Optional[float] = None
    road_friction_mu: Optional[float] = None
    curvature_weave_frequency_hz: Optional[float] = None
    steering_offset_deg: Optional[float] = None
    steering_offset_rad: Optional[float] = None
    weave_delay_s: Optional[float] = None
    target_reference_delay_s: Optional[float] = None
    initial_state_preroll_time_s: Optional[float] = None


# Edit only this block and run this file.
EVAL_CONFIG = EvalConfig(
    model_path=None,
    device="auto",
    seed=30,
    max_episode_time=None,
    output_dir=None,
    show_plot=True,
    save_plots=False,
    save_csv=True,
    save_summary=True,
    deterministic=True,
    fixed_action=None,
    # Evaluation scenario knobs. Leave as None to reuse training metadata.
    initial_speed_kph=20,
    target_speed_kph=50,
    # Signed exact target curvature amplitude [1/m]. Use negative for the opposite direction.
    # If None, the env samples 0..target_curvature_max_m_inv as in training.
    target_curvature_m_inv=0.05,
    target_curvature_max_m_inv=None,
    road_friction_mu=1.0,
    curvature_weave_frequency_hz=0.1,
    steering_offset_deg=None,
    weave_delay_s=None,
)

# Optional evaluation-only overrides. Leave empty to reuse the training metadata.
EVAL_ENV_OVERRIDES: Dict[str, Any] = {}


def configure_matplotlib(show: bool):
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    matplotlib.rcParams.update(
        {
            "font.size": PLOT_TICK_LABEL_FONT_SIZE,
            "figure.titlesize": PLOT_SUPTITLE_FONT_SIZE,
            "axes.titlesize": PLOT_TITLE_FONT_SIZE,
            "axes.labelsize": PLOT_AXIS_LABEL_FONT_SIZE,
            "xtick.labelsize": PLOT_TICK_LABEL_FONT_SIZE,
            "ytick.labelsize": PLOT_TICK_LABEL_FONT_SIZE,
            "legend.fontsize": PLOT_LEGEND_FONT_SIZE,
            "legend.title_fontsize": PLOT_LEGEND_FONT_SIZE,
        }
    )

    import matplotlib.pyplot as plt

    return plt


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = _project_root() / path
    return path.resolve()


def resolve_model_path(model_path: Optional[str]) -> Path:
    if model_path is not None:
        resolved = resolve_project_path(model_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Model file not found: {resolved}")
        return resolved

    runs_dir = _project_root() / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError("runs/ directory not found and EVAL_CONFIG.model_path is not set.")

    candidates = list(runs_dir.rglob("best_model.zip"))
    if not candidates:
        candidates = list(runs_dir.rglob("final_model.zip"))
    if not candidates:
        candidates = list(runs_dir.rglob("*.zip"))
    if not candidates:
        raise FileNotFoundError("No model zip file found under runs/.")

    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def find_run_dir(model_path: Path) -> Optional[Path]:
    for parent in (model_path.parent, *model_path.parents):
        if (parent / "train_config.json").exists():
            return parent
    return None


def load_training_metadata(model_path: Path) -> Dict[str, Any]:
    run_dir = find_run_dir(model_path)
    if run_dir is None:
        return {}

    metadata_path = run_dir / "train_config.json"
    with metadata_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_latest_training_metadata() -> Tuple[Dict[str, Any], Optional[Path]]:
    runs_dir = _project_root() / "runs"
    if not runs_dir.exists():
        return {}, None

    candidates = list(runs_dir.rglob("train_config.json"))
    if not candidates:
        return {}, None

    metadata_path = max(candidates, key=lambda path: path.stat().st_mtime)
    with metadata_path.open("r", encoding="utf-8") as fp:
        return json.load(fp), metadata_path.parent


def _json_sequence_to_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_json_sequence_to_tuple(item) for item in value)
    if isinstance(value, dict):
        return {key: _json_sequence_to_tuple(item) for key, item in value.items()}
    return value


def _resolve_speed_mps(*, mps: Optional[float], kph: Optional[float], name: str) -> Optional[float]:
    if mps is not None and kph is not None:
        raise ValueError(f"Set only one of {name}_mps or {name}_kph.")
    if mps is not None:
        return float(mps)
    if kph is not None:
        return float(kph) / 3.6
    return None


def _resolve_angle_rad(*, rad: Optional[float], deg: Optional[float], name: str) -> Optional[float]:
    if rad is not None and deg is not None:
        raise ValueError(f"Set only one of {name}_rad or {name}_deg.")
    if rad is not None:
        return float(rad)
    if deg is not None:
        return math.radians(float(deg))
    return None


def _apply_eval_env_overrides(env_kwargs: Dict[str, Any], eval_config: EvalConfig) -> None:
    initial_speed_mps = _resolve_speed_mps(
        mps=eval_config.initial_speed_mps,
        kph=eval_config.initial_speed_kph,
        name="initial_speed",
    )
    target_speed_mps = _resolve_speed_mps(
        mps=eval_config.target_speed_mps,
        kph=eval_config.target_speed_kph,
        name="target_speed",
    )

    if initial_speed_mps is not None:
        env_kwargs["initial_speed_range_mps"] = (initial_speed_mps, initial_speed_mps)
    if target_speed_mps is not None:
        env_kwargs["enable_speed_hold_pi"] = True
        env_kwargs["speed_hold_target_speed_mps"] = target_speed_mps
    if eval_config.target_curvature_m_inv is not None:
        target_curvature = float(eval_config.target_curvature_m_inv)
        env_kwargs["fixed_target_curvature_m_inv"] = target_curvature
        env_kwargs["target_curvature_max"] = abs(target_curvature)
    elif eval_config.target_curvature_max_m_inv is not None:
        env_kwargs["fixed_target_curvature_m_inv"] = None
        env_kwargs["target_curvature_max"] = abs(float(eval_config.target_curvature_max_m_inv))
    if eval_config.road_friction_mu is not None:
        road_friction_mu = float(eval_config.road_friction_mu)
        env_kwargs["road_friction_range"] = (road_friction_mu, road_friction_mu)
    if eval_config.curvature_weave_frequency_hz is not None:
        frequency_hz = float(eval_config.curvature_weave_frequency_hz)
        env_kwargs["curvature_weave_frequency_range"] = (frequency_hz, frequency_hz)
    steering_offset_rad = _resolve_angle_rad(
        rad=eval_config.steering_offset_rad,
        deg=eval_config.steering_offset_deg,
        name="steering_offset",
    )
    if steering_offset_rad is not None:
        env_kwargs["curvature_weave_steering_offset_range"] = (steering_offset_rad, steering_offset_rad)
    elif eval_config.target_curvature_m_inv is not None:
        env_kwargs["curvature_weave_steering_offset_range"] = (0.0, 0.0)
    if eval_config.weave_delay_s is not None:
        env_kwargs["weave_delay"] = float(eval_config.weave_delay_s)
    if eval_config.target_reference_delay_s is not None:
        env_kwargs["target_reference_delay_s"] = float(eval_config.target_reference_delay_s)
    if eval_config.initial_state_preroll_time_s is not None:
        env_kwargs["initial_state_preroll_time_s"] = float(eval_config.initial_state_preroll_time_s)


def build_env_config(metadata: Dict[str, Any], eval_config: EvalConfig) -> EnvironmentConfig:
    raw_env_config = dict(metadata.get("env_config", {}))
    allowed_names = {field.name for field in fields(EnvironmentConfig)}
    env_kwargs = {
        key: _json_sequence_to_tuple(value)
        for key, value in raw_env_config.items()
        if key in allowed_names
    }
    env_kwargs.update(EVAL_ENV_OVERRIDES)
    _apply_eval_env_overrides(env_kwargs, eval_config)
    if eval_config.max_episode_time is not None:
        env_kwargs["max_episode_time"] = float(eval_config.max_episode_time)
    return EnvironmentConfig(**env_kwargs)


def default_output_dir(model_path: Optional[Path], output_dir: Optional[str], run_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        path = resolve_project_path(output_dir)
    elif run_dir is not None:
        path = run_dir / "evaluation"
    elif model_path is not None:
        path = model_path.parent / "evaluation"
    else:
        path = _project_root() / "evaluation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_stem(model_path: Optional[Path], fixed_action: Optional[Tuple[float, ...]]) -> str:
    if fixed_action is not None:
        action_text = "_".join(f"{value:+.2f}".replace("+", "p").replace("-", "m") for value in fixed_action)
        return f"fixed_action_{action_text}"
    if model_path is None:
        return "stbw_eval"
    return model_path.stem


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _flatten_numeric(prefix: str, values: Any, row: Dict[str, Any]) -> None:
    if not isinstance(values, dict):
        return
    for key, value in values.items():
        flat_key = f"{prefix}_{key}"
        if isinstance(value, dict):
            _flatten_numeric(flat_key, value, row)
            continue
        if isinstance(value, (str, bytes)):
            continue
        try:
            row[flat_key] = float(value)
        except (TypeError, ValueError):
            continue


def add_observation_columns(row: Dict[str, Any], obs: np.ndarray, env: CustomEnv) -> None:
    obs_norm = np.asarray(obs, dtype=np.float64).reshape(-1)
    avg = np.asarray(env.config.obs_avg_value, dtype=np.float64).reshape(-1)
    std = np.asarray(env.config.obs_std_value, dtype=np.float64).reshape(-1)
    obs_raw = obs_norm * std + avg
    for name, raw_value, norm_value in zip(env.config.observation_names, obs_raw, obs_norm):
        row[f"obs_raw_{name}"] = float(raw_value)
        row[f"obs_norm_{name}"] = float(norm_value)


def build_log_row(
    *,
    step_index: int,
    obs: np.ndarray,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    env: CustomEnv,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "step": int(step_index),
        "time": _safe_float(info.get("elapsed_time", info.get("t", step_index * env.config.control_dt))),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "terminated_reason": str(info.get("terminated_reason", "none")),
    }
    action_array = np.asarray(action, dtype=np.float64).reshape(-1)
    for index, value in enumerate(action_array):
        row[f"action_{index}"] = float(value)
    row["action"] = float(action_array[0]) if action_array.size else float("nan")

    for key, value in info.items():
        if key in {"reference", "scenario"}:
            _flatten_numeric(key, value, row)
            continue
        if isinstance(value, dict):
            continue
        if isinstance(value, (str, bytes)):
            if key == "terminated_reason":
                row[key] = str(value)
            continue
        try:
            row[key] = float(value)
        except (TypeError, ValueError):
            continue

    reference = info.get("reference", {})
    if isinstance(reference, dict):
        for key in ("target_curvature", "target_curvature_dot", "target_lateral_accel", "target_lateral_accel_dot"):
            if key in reference:
                row[key] = _safe_float(reference[key])

    row.setdefault("curvature", row.get("actual_curvature", float("nan")))
    row["actual_curvature"] = row.get("curvature", float("nan"))
    row["curvature_error_abs"] = abs(_safe_float(row.get("curvature_error")))
    row["vx_kph"] = _safe_float(row.get("vx")) * 3.6
    row["target_speed_mps"] = row.get("speed_hold_target_speed_mps", float("nan"))
    row["target_speed_kph"] = _safe_float(row.get("target_speed_mps")) * 3.6
    row["target_ax_mps2"] = row.get("speed_hold_target_accel_mps2", float("nan"))
    row["actual_ax_mps2"] = row.get("ax", float("nan"))
    row["target_ay_mps2"] = row.get("target_lateral_accel", float("nan"))
    row["actual_ay_mps2"] = row.get("ay", float("nan"))

    add_observation_columns(row, obs, env)
    return row


def rollout_episode(
    *,
    model: SAC,
    env: CustomEnv,
    seed: int,
    deterministic: bool,
) -> Tuple[List[Dict[str, Any]], bool, bool]:
    obs, _ = env.reset(seed=seed)
    records: List[Dict[str, Any]] = []
    terminated = False
    truncated = False
    step_index = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)
        records.append(
            build_log_row(
                step_index=step_index,
                obs=next_obs,
                action=np.asarray(action, dtype=np.float64),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
                env=env,
            )
        )
        obs = next_obs
        step_index += 1

    return records, bool(terminated), bool(truncated)


def rollout_fixed_action_episode(
    *,
    env: CustomEnv,
    seed: int,
    fixed_action: Tuple[float, ...],
) -> Tuple[List[Dict[str, Any]], bool, bool]:
    obs, _ = env.reset(seed=seed)
    action = np.asarray(fixed_action, dtype=np.float32).reshape(env.action_space.shape)
    records: List[Dict[str, Any]] = []
    terminated = False
    truncated = False
    step_index = 0

    while not (terminated or truncated):
        next_obs, reward, terminated, truncated, info = env.step(action)
        records.append(
            build_log_row(
                step_index=step_index,
                obs=next_obs,
                action=action,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=info,
                env=env,
            )
        )
        obs = next_obs
        step_index += 1

    return records, bool(terminated), bool(truncated)


def save_csv(records: List[Dict[str, Any]], csv_path: Path) -> None:
    if not records:
        raise ValueError("Cannot save an empty evaluation record list.")
    fieldnames = sorted({key for row in records for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def trim_records_by_time(records: List[Dict[str, Any]], start_time_s: float) -> List[Dict[str, Any]]:
    if not records:
        return []
    start = float(start_time_s)
    trimmed = [row for row in records if _safe_float(row.get("time"), -float("inf")) >= start]
    return trimmed if trimmed else [records[-1]]


def _record_array(records: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_safe_float(row.get(key)) for row in records], dtype=np.float64)


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _finite_rmse(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite * finite)))


def _finite_max_abs(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(np.abs(finite)))


def compute_summary(records: List[Dict[str, Any]], env_config: EnvironmentConfig) -> Dict[str, Any]:
    reward = _record_array(records, "reward")
    curvature_error = _record_array(records, "curvature_error")
    beta = _record_array(records, "beta")
    vx = _record_array(records, "vx")
    speed_error = _record_array(records, "speed_hold_speed_error_mps")
    road_wheel_usage = compute_road_wheel_limit_summary(records)
    return {
        "sample_count": int(len(records)),
        "duration_s": float(_safe_float(records[-1].get("time"))) if records else 0.0,
        "total_reward": float(np.nansum(reward)) if reward.size else 0.0,
        "mean_reward": _finite_mean(reward),
        "curvature_mae_1pm": 1000.0 * _finite_mean(np.abs(curvature_error)),
        "curvature_rmse_1pm": 1000.0 * _finite_rmse(curvature_error),
        "beta_max_abs_rad": _finite_max_abs(beta),
        "beta_warn_rad": float(env_config.beta_warn),
        "beta_term_rad": float(env_config.beta_term),
        "vx_mean_kph": 3.6 * _finite_mean(vx),
        "speed_error_rmse_mps": _finite_rmse(speed_error),
        "road_wheel_limit_check": road_wheel_usage,
    }


def compute_reward_gain_contribution_means(
    records: List[Dict[str, Any]],
    env_config: EnvironmentConfig,
) -> Dict[str, float]:
    curvature_error = _record_array(records, "curvature_error")
    vx = _record_array(records, "vx")
    beta = _record_array(records, "beta")
    beta_excess = np.maximum(np.abs(beta) - float(env_config.beta_warn), 0.0)
    return {
        "K_kappa*dK^2": float(env_config.K_kappa) * _finite_mean(curvature_error * curvature_error),
        "K_ay*dK^2*vx^4": float(env_config.K_ay) * _finite_mean(curvature_error * curvature_error * vx**4),
        "K_slip*beta_excess^2": float(env_config.K_slip) * _finite_mean(beta_excess * beta_excess),
    }


def _finite_stats(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "max": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def compute_road_wheel_limit_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for wheel in ("fl", "fr", "rl", "rr"):
        usage = _record_array(records, f"{wheel}_friction_circle_usage")
        saturated = _record_array(records, f"{wheel}_friction_circle_saturated")
        scale = _record_array(records, f"{wheel}_friction_circle_scale")
        finite_saturated = saturated[np.isfinite(saturated)]
        summary[wheel] = {
            "usage": _finite_stats(usage),
            "saturated_ratio": float(np.mean(finite_saturated > 0.5)) if finite_saturated.size else float("nan"),
            "scale_min": float(np.nanmin(scale)) if np.any(np.isfinite(scale)) else float("nan"),
        }
    return summary


def print_road_wheel_limit_summary(limit_summary: Dict[str, Any]) -> None:
    print("road_wheel_limit_check:")
    for wheel, stats in limit_summary.items():
        usage = stats["usage"]
        print(
            f"  {wheel}: usage_mean={usage['mean']:.4f}, "
            f"usage_p95={usage['p95']:.4f}, usage_max={usage['max']:.4f}, "
            f"saturated_ratio={stats['saturated_ratio']:.3f}, scale_min={stats['scale_min']:.4f}"
        )


def save_summary(summary: Dict[str, Any], summary_path: Path) -> None:
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def _plot_if_finite(axis, time: np.ndarray, values: np.ndarray, **kwargs) -> None:
    finite = np.isfinite(values)
    if not np.any(finite):
        return
    axis.plot(time[finite], values[finite], **kwargs)


def _mean_wheels(records: List[Dict[str, Any]], wheels: Sequence[str], suffix: str) -> np.ndarray:
    arrays = [_record_array(records, f"{wheel}_{suffix}") for wheel in wheels]
    if not arrays:
        return np.full(len(records), np.nan, dtype=np.float64)
    stacked = np.vstack(arrays)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stacked, axis=0)


def _lw(base: float, scale: float) -> float:
    return max(0.2, float(base) * float(scale))


def _plot_friction_circle_axis(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    limit: np.ndarray,
    saturated: np.ndarray,
    title: str,
    line_width_scale: float,
) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.any(finite):
        axis.plot(x[finite], y[finite], color="tab:blue", linewidth=_lw(1.0, line_width_scale), alpha=0.85, label="tire force")
        axis.scatter(x[finite][0], y[finite][0], color="tab:green", s=20, label="start")
        axis.scatter(x[finite][-1], y[finite][-1], color="tab:red", s=20, label="end")

    saturated_finite = finite & np.isfinite(saturated) & (saturated > 0.5)
    if np.any(saturated_finite):
        axis.scatter(
            x[saturated_finite],
            y[saturated_finite],
            color="tab:red",
            s=10,
            alpha=0.75,
            label="saturated samples",
            zorder=3,
        )

    circle_radii = []
    theta = np.linspace(0.0, 2.0 * math.pi, 241)
    finite_limit = limit[np.isfinite(limit) & (limit > 0.0)]
    if finite_limit.size:
        circle_limit = float(np.median(finite_limit))
        circle_radii.append(circle_limit)
        axis.plot(
            circle_limit * np.cos(theta),
            circle_limit * np.sin(theta),
            color="black",
            linewidth=_lw(0.9, line_width_scale),
            linestyle="--",
            alpha=0.65,
            label=f"median limit {circle_limit:.0f} N",
        )

    saturated_limit = limit[saturated_finite & np.isfinite(limit) & (limit > 0.0)]
    if saturated_limit.size:
        saturated_circle_limit = float(np.median(saturated_limit))
        circle_radii.append(saturated_circle_limit)
        axis.plot(
            saturated_circle_limit * np.cos(theta),
            saturated_circle_limit * np.sin(theta),
            color="tab:red",
            linewidth=_lw(1.1, line_width_scale),
            linestyle=":",
            alpha=0.9,
            label=f"saturated median {saturated_circle_limit:.0f} N",
        )

    if np.any(finite):
        force_radius = float(np.nanmax(np.maximum(np.abs(x[finite]), np.abs(y[finite]))))
        if np.isfinite(force_radius) and force_radius > 0.0:
            circle_radii.append(force_radius)
    if circle_radii:
        view_radius = max(circle_radii)
        axis.set_xlim(-1.15 * view_radius, 1.15 * view_radius)
        axis.set_ylim(-1.15 * view_radius, 1.15 * view_radius)

    axis.axhline(0.0, color="black", linewidth=_lw(0.6, line_width_scale), alpha=0.35)
    axis.axvline(0.0, color="black", linewidth=_lw(0.6, line_width_scale), alpha=0.35)
    axis.set_title(title)
    axis.set_xlabel("F_x [N]")
    axis.set_ylabel("F_y [N]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=PLOT_LEGEND_FONT_SIZE)


def plot_records(
    *,
    records: List[Dict[str, Any]],
    env_config: EnvironmentConfig,
    png_path: Path,
    show: bool,
    save: bool,
    line_width_scale: float,
    title: str,
) -> Tuple[Optional[Path], ...]:
    if not records:
        raise ValueError("Cannot plot an empty evaluation record list.")

    plt = configure_matplotlib(show=show)
    time = _record_array(records, "time")
    figures = []

    vx = _record_array(records, "vx")
    vy = _record_array(records, "vy")
    x = _record_array(records, "x")
    y = _record_array(records, "y")
    ax = _record_array(records, "ax")
    ay = _record_array(records, "ay")
    yaw_rate = _record_array(records, "yaw_rate")
    target_curvature = _record_array(records, "target_curvature")
    curvature = _record_array(records, "curvature")
    curvature_error = _record_array(records, "curvature_error")
    curvature_lateral_accel_error = _record_array(records, "curvature_lateral_accel_error")
    curvature_lateral_accel_limit = _record_array(records, "curvature_lateral_accel_limit_mps2")
    beta = _record_array(records, "beta")
    steer = _record_array(records, "steer")
    internal_steer = _record_array(records, "internal_steer")
    applied_steer = _record_array(records, "applied_steer")
    reward_total = _record_array(records, "reward_total")
    reward_track = _record_array(records, "reward_track")
    reward_slip = _record_array(records, "reward_slip")
    reward_used = _record_array(records, "reward_used")
    action = _record_array(records, "action")
    steer_ddot_cmd = _record_array(records, "steer_ddot_cmd")
    target_speed = _record_array(records, "speed_hold_target_speed_mps")
    speed_error = _record_array(records, "speed_hold_speed_error_mps")
    target_ax = _record_array(records, "speed_hold_target_accel_mps2")
    raw_target_ax = _record_array(records, "speed_hold_raw_target_accel_mps2")
    drive_torque = _record_array(records, "speed_hold_drive_torque_nm")
    brake_torque = _record_array(records, "speed_hold_brake_motor_torque")
    target_ay = _record_array(records, "target_lateral_accel")
    target_ay_dot = _record_array(records, "target_lateral_accel_dot")
    target_curvature_dot = _record_array(records, "target_curvature_dot")

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), sharex=False)
    fig.suptitle(f"{title} - Tracking")
    flat_axes = axes.flatten()

    flat_axes[0].plot(x, y, color="tab:blue", linewidth=_lw(1.2, line_width_scale), label="path")
    flat_axes[0].set_title("Vehicle Path")
    flat_axes[0].set_xlabel("x [m]")
    flat_axes[0].set_ylabel("y [m]")
    flat_axes[0].axis("equal")
    flat_axes[0].grid(True, alpha=0.3)
    flat_axes[0].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    _plot_if_finite(flat_axes[1], time, target_curvature, label="target", color="tab:blue", linewidth=_lw(1.2, line_width_scale))
    _plot_if_finite(flat_axes[1], time, curvature, label="actual", color="tab:orange", linewidth=_lw(1.2, line_width_scale))
    flat_axes[1].set_title("Curvature Tracking")
    flat_axes[1].set_ylabel("1/m")
    flat_axes[1].grid(True, alpha=0.3)
    flat_axes[1].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    _plot_if_finite(flat_axes[2], time, curvature_error, label="curvature error", color="tab:red", linewidth=_lw(1.2, line_width_scale))
    flat_axes[2].axhline(0.0, color="black", linewidth=_lw(0.8, line_width_scale), alpha=0.45)
    flat_axes[2].set_title("Curvature Error")
    flat_axes[2].set_ylabel("1/m")
    flat_axes[2].grid(True, alpha=0.3)
    flat_axes[2].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    _plot_if_finite(flat_axes[3], time, vx * 3.6, label="vx", color="tab:blue", linewidth=_lw(1.2, line_width_scale))
    _plot_if_finite(flat_axes[3], time, target_speed * 3.6, label="target", color="tab:orange", linewidth=_lw(1.0, line_width_scale), linestyle="--")
    flat_axes[3].set_title("Speed")
    flat_axes[3].set_ylabel("km/h")
    flat_axes[3].grid(True, alpha=0.3)
    flat_axes[3].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    _plot_if_finite(flat_axes[4], time, beta, label="beta", color="tab:purple", linewidth=_lw(1.2, line_width_scale))
    flat_axes[4].axhline(float(env_config.beta_warn), color="tab:orange", linestyle="--", linewidth=_lw(0.9, line_width_scale), label="warn")
    flat_axes[4].axhline(-float(env_config.beta_warn), color="tab:orange", linestyle="--", linewidth=_lw(0.9, line_width_scale))
    flat_axes[4].axhline(float(env_config.beta_term), color="tab:red", linestyle=":", linewidth=_lw(0.9, line_width_scale), label="term")
    flat_axes[4].axhline(-float(env_config.beta_term), color="tab:red", linestyle=":", linewidth=_lw(0.9, line_width_scale))
    flat_axes[4].set_title("Sideslip")
    flat_axes[4].set_ylabel("rad")
    flat_axes[4].grid(True, alpha=0.3)
    flat_axes[4].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    _plot_if_finite(flat_axes[5], time, ax, label="ax", color="tab:blue", linewidth=_lw(1.0, line_width_scale))
    _plot_if_finite(flat_axes[5], time, ay, label="ay", color="tab:orange", linewidth=_lw(1.0, line_width_scale))
    _plot_if_finite(flat_axes[5], time, target_ay, label="target ay", color="tab:green", linewidth=_lw(0.95, line_width_scale), linestyle="--")
    flat_axes[5].set_title("Acceleration")
    flat_axes[5].set_ylabel("m/s^2")
    flat_axes[5].grid(True, alpha=0.3)
    flat_axes[5].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    _plot_if_finite(flat_axes[6], time, np.degrees(steer), label="front steer", color="tab:blue", linewidth=_lw(1.0, line_width_scale))
    _plot_if_finite(flat_axes[6], time, np.degrees(internal_steer), label="internal", color="tab:orange", linewidth=_lw(0.95, line_width_scale), linestyle="--")
    _plot_if_finite(flat_axes[6], time, np.degrees(applied_steer), label="applied", color="tab:green", linewidth=_lw(0.95, line_width_scale), linestyle=":")
    flat_axes[6].set_title("Steering")
    flat_axes[6].set_ylabel("deg")
    flat_axes[6].grid(True, alpha=0.3)
    flat_axes[6].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    _plot_if_finite(flat_axes[7], time, yaw_rate, label="yaw rate", color="tab:blue", linewidth=_lw(1.0, line_width_scale))
    _plot_if_finite(flat_axes[7], time, curvature_lateral_accel_error, label="|vx^2 * kappa error|", color="tab:red", linewidth=_lw(1.0, line_width_scale))
    _plot_if_finite(flat_axes[7], time, curvature_lateral_accel_limit, label="limit", color="black", linewidth=_lw(0.9, line_width_scale), linestyle="--")
    flat_axes[7].set_title("Yaw / Curvature Error Limit")
    flat_axes[7].grid(True, alpha=0.3)
    flat_axes[7].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    for axis in flat_axes[1:]:
        axis.set_xlabel("time [s]")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figures.append(fig)
    if save:
        fig.savefig(png_path, dpi=200, bbox_inches="tight")

    reward_gain_path = png_path.with_name(f"{png_path.stem}_reward_gain_contribution.png")
    reward_gain_contributions = compute_reward_gain_contribution_means(records, env_config)
    reward_gain_fig, reward_gain_axis = plt.subplots(figsize=(11, 6))
    reward_gain_fig.suptitle(f"{title} - Reward Gain Contributions")
    names = list(reward_gain_contributions.keys())
    values = np.asarray([reward_gain_contributions[name] for name in names], dtype=np.float64)
    bars = reward_gain_axis.bar(names, np.nan_to_num(values, nan=0.0), color=("tab:blue", "tab:red", "tab:purple"))
    for bar, value in zip(bars, values):
        label = f"{value:.6g}" if np.isfinite(value) else "nan"
        reward_gain_axis.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), label, ha="center", va="bottom")
    reward_gain_axis.set_ylabel("mean cost contribution")
    reward_gain_axis.grid(True, axis="y", alpha=0.3)
    reward_gain_axis.tick_params(axis="x", rotation=12)
    reward_gain_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    figures.append(reward_gain_fig)
    if save:
        reward_gain_fig.savefig(reward_gain_path, dpi=200, bbox_inches="tight")

    action_reward_path = png_path.with_name(f"{png_path.stem}_action_reward.png")
    action_fig, action_axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    action_fig.suptitle(f"{title} - Action / Reward")
    action_flat_axes = action_axes.flatten()
    _plot_if_finite(action_flat_axes[0], time, action, label="action norm", color="tab:blue", linewidth=_lw(1.2, line_width_scale))
    _plot_if_finite(action_flat_axes[0], time, steer_ddot_cmd, label="steer ddot cmd", color="tab:orange", linewidth=_lw(1.0, line_width_scale), linestyle="--")
    action_flat_axes[0].set_title("Action")
    action_flat_axes[0].grid(True, alpha=0.3)
    action_flat_axes[0].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    _plot_if_finite(action_flat_axes[1], time, reward_total, label="reward_total", color="tab:blue", linewidth=_lw(1.0, line_width_scale))
    _plot_if_finite(action_flat_axes[1], time, reward_track, label="reward_track", color="tab:green", linewidth=_lw(0.95, line_width_scale))
    _plot_if_finite(action_flat_axes[1], time, reward_slip, label="reward_slip", color="tab:red", linewidth=_lw(0.95, line_width_scale))
    _plot_if_finite(action_flat_axes[1], time, reward_used, label="reward_used", color="tab:purple", linewidth=_lw(0.95, line_width_scale), linestyle="--")
    action_flat_axes[1].set_title("Reward Terms")
    action_flat_axes[1].grid(True, alpha=0.3)
    action_flat_axes[1].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    _plot_if_finite(action_flat_axes[2], time, speed_error, label="speed error", color="tab:red", linewidth=_lw(1.0, line_width_scale))
    _plot_if_finite(action_flat_axes[2], time, target_ax, label="target ax", color="tab:blue", linewidth=_lw(0.95, line_width_scale), linestyle="--")
    action_flat_axes[2].set_title("Speed PI")
    action_flat_axes[2].grid(True, alpha=0.3)
    action_flat_axes[2].legend(fontsize=PLOT_LEGEND_FONT_SIZE)

    _plot_if_finite(action_flat_axes[3], time, drive_torque, label="drive torque", color="tab:green", linewidth=_lw(1.0, line_width_scale))
    _plot_if_finite(action_flat_axes[3], time, brake_torque, label="brake motor torque", color="tab:red", linewidth=_lw(1.0, line_width_scale))
    action_flat_axes[3].set_title("Longitudinal Commands")
    action_flat_axes[3].grid(True, alpha=0.3)
    action_flat_axes[3].legend(fontsize=PLOT_LEGEND_FONT_SIZE)
    for axis in action_flat_axes:
        axis.set_xlabel("time [s]")
    action_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figures.append(action_fig)
    if save:
        action_fig.savefig(action_reward_path, dpi=200, bbox_inches="tight")

    ax_diagnostics_path = png_path.with_name(f"{png_path.stem}_ax_diagnostics.png")
    ax_diag_fig, ax_diag_axes = plt.subplots(5, 2, figsize=(15, 18), sharex=True)
    ax_diag_fig.suptitle(f"{title} - Ax / Reference Diagnostics")
    ax_diag_flat_axes = ax_diag_axes.flatten()
    series = (
        ("Speed", ((vx * 3.6, "vx", "tab:blue"), (target_speed * 3.6, "target", "tab:orange")), "km/h"),
        ("Longitudinal Accel", ((ax, "actual ax", "tab:blue"), (target_ax, "target ax", "tab:orange"), (raw_target_ax, "raw target ax", "tab:green")), "m/s^2"),
        ("Lateral Accel", ((ay, "actual ay", "tab:blue"), (target_ay, "target ay", "tab:orange")), "m/s^2"),
        ("Target Derivatives", ((target_curvature_dot, "target curvature dot", "tab:blue"), (target_ay_dot, "target ay dot", "tab:orange")), ""),
        ("Curvature Error Accel", ((curvature_lateral_accel_error, "error", "tab:red"), (curvature_lateral_accel_limit, "limit", "black")), "m/s^2"),
        ("Speed Error", ((speed_error, "speed error", "tab:red"),), "m/s"),
        ("Body Velocity", ((vx, "vx", "tab:blue"), (vy, "vy", "tab:orange")), "m/s"),
        ("Body Accel", ((ax, "ax", "tab:blue"), (ay, "ay", "tab:orange")), "m/s^2"),
        ("Sideslip", ((beta, "beta", "tab:purple"),), "rad"),
        ("Road Friction", ((_record_array(records, "road_friction"), "mu", "tab:olive"),), ""),
    )
    for axis, (axis_title, axis_series, ylabel) in zip(ax_diag_flat_axes, series):
        for values, label, color in axis_series:
            _plot_if_finite(axis, time, values, label=label, color=color, linewidth=_lw(1.0, line_width_scale))
        axis.set_title(axis_title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
    for axis in ax_diag_flat_axes:
        axis.set_xlabel("time [s]")
    ax_diag_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    figures.append(ax_diag_fig)
    if save:
        ax_diag_fig.savefig(ax_diagnostics_path, dpi=200, bbox_inches="tight")

    friction_circle_path = png_path.with_name(f"{png_path.stem}_friction_circle.png")
    friction_fig, friction_axes = plt.subplots(2, 2, figsize=(13, 10), sharex=False)
    friction_fig.suptitle(f"{title} - Friction Circle")
    for axis, wheel in zip(friction_axes.flatten(), ("fl", "fr", "rl", "rr")):
        _plot_friction_circle_axis(
            axis,
            _record_array(records, f"{wheel}_F_x_tire"),
            _record_array(records, f"{wheel}_F_y_tire"),
            _record_array(records, f"{wheel}_friction_circle_limit"),
            _record_array(records, f"{wheel}_friction_circle_saturated"),
            wheel.upper(),
            line_width_scale,
        )
    friction_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figures.append(friction_fig)
    if save:
        friction_fig.savefig(friction_circle_path, dpi=200, bbox_inches="tight")

    front_slip_diagnostics_path = png_path.with_name(f"{png_path.stem}_front_slip_diagnostics.png")
    front_wheel_speed = _mean_wheels(records, ("fl", "fr"), "dugoff_wheel_linear_speed")
    front_local_vx = _mean_wheels(records, ("fl", "fr"), "dugoff_vx")
    front_delta_v = front_wheel_speed - front_local_vx
    front_kappa = _mean_wheels(records, ("fl", "fr"), "dugoff_kappa")
    front_alpha = _mean_wheels(records, ("fl", "fr"), "dugoff_alpha")
    front_slip_fig, front_slip_axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    front_slip_fig.suptitle(f"{title} - Front Slip Diagnostics")
    _plot_if_finite(front_slip_axes[0], time, front_wheel_speed, label="front omega*R_eff", color="tab:blue", linewidth=_lw(1.2, line_width_scale))
    front_slip_axes[0].set_title("Front Wheel Circumference Speed")
    front_slip_axes[0].set_ylabel("m/s")
    _plot_if_finite(front_slip_axes[1], time, front_local_vx, label="front V_wx_local", color="tab:orange", linewidth=_lw(1.2, line_width_scale))
    front_slip_axes[1].set_title("Front Wheel Local X Velocity")
    front_slip_axes[1].set_ylabel("m/s")
    _plot_if_finite(front_slip_axes[2], time, front_delta_v, label="front omega*R_eff - V_wx_local", color="tab:red", linewidth=_lw(1.2, line_width_scale))
    front_slip_axes[2].axhline(0.0, color="black", linewidth=_lw(0.8, line_width_scale), alpha=0.5)
    front_slip_axes[2].set_title("Front Slip Velocity Delta")
    front_slip_axes[2].set_ylabel("m/s")
    _plot_if_finite(front_slip_axes[3], time, front_kappa, label="front kappa", color="tab:green", linewidth=_lw(1.1, line_width_scale))
    _plot_if_finite(front_slip_axes[3], time, front_alpha, label="front alpha", color="tab:purple", linewidth=_lw(1.0, line_width_scale), linestyle="--")
    front_slip_axes[3].set_title("Front Kappa / Alpha")
    front_slip_axes[3].set_ylabel("rad or ratio")
    for axis in front_slip_axes:
        axis.set_xlabel("time [s]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
    front_slip_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figures.append(front_slip_fig)
    if save:
        front_slip_fig.savefig(front_slip_diagnostics_path, dpi=200, bbox_inches="tight")

    longitudinal_tire_diagnostics_path = png_path.with_name(f"{png_path.stem}_longitudinal_tire_diagnostics.png")
    rear_wheel_speed = _mean_wheels(records, ("rl", "rr"), "dugoff_wheel_linear_speed")
    rear_local_vx = _mean_wheels(records, ("rl", "rr"), "dugoff_vx")
    rear_delta_v = rear_wheel_speed - rear_local_vx
    front_denom = np.maximum(np.abs(front_local_vx), 1.0e-6)
    rear_denom = np.maximum(np.abs(rear_local_vx), 1.0e-6)
    longitudinal_tire_fig, longitudinal_tire_axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    longitudinal_tire_fig.suptitle(f"{title} - Longitudinal Tire Slip Calculation")
    _plot_if_finite(longitudinal_tire_axes[0], time, front_local_vx, label="front V_wheel_x", color="tab:blue", linewidth=_lw(1.2, line_width_scale))
    _plot_if_finite(longitudinal_tire_axes[0], time, rear_local_vx, label="rear V_wheel_x", color="tab:orange", linewidth=_lw(1.0, line_width_scale), linestyle="--")
    longitudinal_tire_axes[0].set_title("V_wheel_x from Tire")
    longitudinal_tire_axes[0].set_ylabel("m/s")
    _plot_if_finite(longitudinal_tire_axes[1], time, front_wheel_speed, label="front V_wheel", color="tab:blue", linewidth=_lw(1.2, line_width_scale))
    _plot_if_finite(longitudinal_tire_axes[1], time, rear_wheel_speed, label="rear V_wheel", color="tab:orange", linewidth=_lw(1.0, line_width_scale), linestyle="--")
    longitudinal_tire_axes[1].set_title("V_wheel = omega_wheel * R_eff")
    longitudinal_tire_axes[1].set_ylabel("m/s")
    _plot_if_finite(longitudinal_tire_axes[2], time, front_delta_v, label="front V_wheel - V_wheel_x", color="tab:red", linewidth=_lw(1.2, line_width_scale))
    _plot_if_finite(longitudinal_tire_axes[2], time, rear_delta_v, label="rear V_wheel - V_wheel_x", color="tab:purple", linewidth=_lw(1.0, line_width_scale), linestyle="--")
    longitudinal_tire_axes[2].axhline(0.0, color="black", linewidth=_lw(0.8, line_width_scale), alpha=0.5)
    longitudinal_tire_axes[2].set_title("Slip Velocity Delta")
    longitudinal_tire_axes[2].set_ylabel("m/s")
    _plot_if_finite(longitudinal_tire_axes[3], time, front_denom, label="front denom", color="tab:green", linewidth=_lw(1.2, line_width_scale))
    _plot_if_finite(longitudinal_tire_axes[3], time, rear_denom, label="rear denom", color="tab:olive", linewidth=_lw(1.0, line_width_scale), linestyle="--")
    longitudinal_tire_axes[3].set_title("Slip Ratio Denominator")
    longitudinal_tire_axes[3].set_ylabel("m/s")
    for axis in longitudinal_tire_axes:
        axis.set_xlabel("time [s]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=PLOT_LEGEND_FONT_SIZE)
    longitudinal_tire_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figures.append(longitudinal_tire_fig)
    if save:
        longitudinal_tire_fig.savefig(longitudinal_tire_diagnostics_path, dpi=200, bbox_inches="tight")

    observation_path = png_path.with_name(f"{png_path.stem}_observations.png")
    observation_names = tuple(str(name) for name in env_config.observation_names)
    observation_cols = 2
    observation_rows = max(1, int(math.ceil(len(observation_names) / observation_cols)))
    observation_fig, observation_axes = plt.subplots(
        observation_rows,
        observation_cols,
        figsize=(14, max(4.0, 2.35 * observation_rows)),
        sharex=True,
        squeeze=False,
    )
    observation_fig.suptitle(f"{title} - Observations")
    observation_flat_axes = observation_axes.flatten()
    for axis, obs_name in zip(observation_flat_axes, observation_names):
        raw_values = _record_array(records, f"obs_raw_{obs_name}")
        norm_values = _record_array(records, f"obs_norm_{obs_name}")
        raw_lines = axis.plot(time, raw_values, label="raw", color="tab:blue", linewidth=_lw(1.1, line_width_scale))
        axis.set_title(obs_name)
        axis.set_ylabel("raw")
        axis.grid(True, alpha=0.3)
        norm_axis = axis.twinx()
        norm_lines = norm_axis.plot(time, norm_values, label="normalized", color="tab:orange", linewidth=_lw(0.95, line_width_scale), linestyle="--")
        norm_axis.axhline(1.0, color="tab:orange", linewidth=_lw(0.7, line_width_scale), alpha=0.25)
        norm_axis.axhline(-1.0, color="tab:orange", linewidth=_lw(0.7, line_width_scale), alpha=0.25)
        norm_axis.set_ylim(-1.1, 1.1)
        norm_axis.set_ylabel("norm")
        lines = raw_lines + norm_lines
        axis.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=PLOT_LEGEND_FONT_SIZE)
    for axis in observation_flat_axes[len(observation_names):]:
        axis.set_visible(False)
    for axis in observation_flat_axes:
        if not axis.get_visible():
            continue
        axis.set_xlabel("time [s]")
        axis.tick_params(axis="x", labelbottom=True)
    observation_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    figures.append(observation_fig)
    if save:
        observation_fig.savefig(observation_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    for figure in figures:
        plt.close(figure)

    if not save:
        return (None, None, None, None, None, None, None)
    return (
        reward_gain_path,
        action_reward_path,
        observation_path,
        ax_diagnostics_path,
        friction_circle_path,
        front_slip_diagnostics_path,
        longitudinal_tire_diagnostics_path,
    )


def main() -> int:
    fixed_action = EVAL_CONFIG.fixed_action
    model_path: Optional[Path] = None
    run_dir: Optional[Path] = None

    if EVAL_CONFIG.model_path is not None:
        model_path = resolve_model_path(EVAL_CONFIG.model_path)
        metadata = load_training_metadata(model_path)
        run_dir = find_run_dir(model_path)
    elif fixed_action is None:
        model_path = resolve_model_path(None)
        metadata = load_training_metadata(model_path)
        run_dir = find_run_dir(model_path)
    else:
        metadata, run_dir = load_latest_training_metadata()

    env_config = build_env_config(metadata, EVAL_CONFIG)
    env = CustomEnv(env_config)
    try:
        if fixed_action is None:
            if model_path is None:
                raise RuntimeError("Model path resolution failed.")
            model = SAC.load(str(model_path), device=EVAL_CONFIG.device)
            records, terminated, truncated = rollout_episode(
                model=model,
                env=env,
                seed=EVAL_CONFIG.seed,
                deterministic=EVAL_CONFIG.deterministic,
            )
        else:
            records, terminated, truncated = rollout_fixed_action_episode(
                env=env,
                seed=EVAL_CONFIG.seed,
                fixed_action=tuple(float(value) for value in fixed_action),
            )

        eval_records = trim_records_by_time(records, start_time_s=EVAL_CONFIG.evaluation_warmup_time_s)
        output_dir = default_output_dir(model_path, EVAL_CONFIG.output_dir, run_dir=run_dir)
        output_stem = build_output_stem(model_path, fixed_action)
        warmup_suffix = f"after_{EVAL_CONFIG.evaluation_warmup_time_s:.1f}s"
        csv_path = output_dir / f"{output_stem}_tracking.csv"
        eval_csv_path = output_dir / f"{output_stem}_tracking_{warmup_suffix}.csv"
        png_path = output_dir / f"{output_stem}_tracking_{warmup_suffix}.png"
        summary_path = output_dir / f"{output_stem}_summary_{warmup_suffix}.json"
        result_title = (
            f"STBW Tracking Evaluation ({model_path.name}, {warmup_suffix})"
            if fixed_action is None and model_path is not None
            else f"STBW Sanity Check (fixed_action={fixed_action}, {warmup_suffix})"
        )

        if EVAL_CONFIG.save_csv:
            save_csv(records, csv_path)
            save_csv(eval_records, eval_csv_path)

        summary = compute_summary(eval_records, env.config)
        summary["terminated"] = bool(terminated)
        summary["truncated"] = bool(truncated)
        summary["evaluation_warmup_time_s"] = float(EVAL_CONFIG.evaluation_warmup_time_s)
        summary["raw_episode_duration_s"] = float(records[-1]["time"])
        summary["evaluated_start_time_s"] = float(eval_records[0]["time"])
        summary["evaluated_end_time_s"] = float(eval_records[-1]["time"])
        summary["raw_sample_count"] = int(len(records))
        summary["evaluated_sample_count"] = int(len(eval_records))
        summary["eval_config"] = asdict(EVAL_CONFIG)
        summary["env_config"] = asdict(env.config)
        if EVAL_CONFIG.save_summary:
            save_summary(summary, summary_path)

        plot_paths = plot_records(
            records=eval_records,
            env_config=env.config,
            png_path=png_path,
            show=EVAL_CONFIG.show_plot,
            save=EVAL_CONFIG.save_plots,
            line_width_scale=EVAL_CONFIG.plot_line_width_scale,
            title=result_title,
        )

        reward_gain_contributions = compute_reward_gain_contribution_means(eval_records, env.config)
        reward_gain_total = float(sum(value for value in reward_gain_contributions.values() if math.isfinite(value)))

        if model_path is not None:
            print(f"model_path: {model_path}")
        if fixed_action is not None:
            print(f"fixed_action: {fixed_action}")
        print(f"evaluation_warmup_time_s: {EVAL_CONFIG.evaluation_warmup_time_s}")
        if EVAL_CONFIG.save_csv:
            print(f"raw_csv_path: {csv_path}")
            print(f"eval_csv_path: {eval_csv_path}")
        if EVAL_CONFIG.save_plots:
            print(f"plot_path: {png_path}")
            labels = (
                "reward_gain_contribution_plot_path",
                "action_reward_plot_path",
                "observation_plot_path",
                "ax_diagnostics_plot_path",
                "friction_circle_plot_path",
                "front_slip_diagnostics_plot_path",
                "longitudinal_tire_diagnostics_plot_path",
            )
            for label, path in zip(labels, plot_paths):
                if path is not None:
                    print(f"{label}: {path}")
        else:
            print("plot_save: disabled")
        print("reward_gain_contribution_mean:")
        for name, value in reward_gain_contributions.items():
            share = 100.0 * float(value) / reward_gain_total if reward_gain_total > 0.0 else 0.0
            print(f"  {name}: {float(value):.6g} ({share:.1f}%)")
        if EVAL_CONFIG.save_summary:
            print(f"summary_path: {summary_path}")
        print_road_wheel_limit_summary(summary["road_wheel_limit_check"])
        print(json.dumps(summary, indent=2))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
