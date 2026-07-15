from typing import Dict, Optional

import numpy as np


def _expected_observation_dim(config) -> int: 
    expected_dim = len(tuple(getattr(config, "observation_names", ())))
    if expected_dim <= 0:
        raise ValueError("config.observation_names must contain at least one entry.")
    return expected_dim


def _observation_names(config, expected_dim: int) -> tuple: # 관측값 개수가 expected_dim과 일치하는지
    names = tuple(getattr(config, "observation_names", ()))
    if names and len(names) != expected_dim:
        raise ValueError(
            "config.observation_names length mismatch: "
            f"expected {expected_dim}, got {len(names)}."
        )
    return names


def _as_numeric_vector(name: str, values) -> np.ndarray: # 입력 형식 통일, 기본 구조 검사
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric 1D vector.") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape {arr.shape}.")
    return arr


def _raise_if_nonfinite(name: str, arr: np.ndarray, names: tuple) -> None:
    invalid_indices = np.flatnonzero(~np.isfinite(arr))
    if invalid_indices.size == 0:
        return

    details = []
    for index in invalid_indices[:5]:
        label = names[index] if index < len(names) else f"index_{index}"
        details.append(f"{label}[{index}]={float(arr[index])!r}")
    suffix = "" if invalid_indices.size <= 5 else f", ... ({invalid_indices.size} total)"
    raise ValueError(f"{name} contains non-finite values: {', '.join(details)}{suffix}.")


def _validate_observation_vector(name: str, values, config) -> np.ndarray:
    expected_dim = _expected_observation_dim(config)
    names = _observation_names(config, expected_dim)
    arr = _as_numeric_vector(name, values)
    if arr.shape != (expected_dim,):
        raise ValueError(
            f"{name} length mismatch: expected shape ({expected_dim},), got {arr.shape}."
        )
    _raise_if_nonfinite(name, arr, names)
    return arr


def _validate_normalization_vector(name: str, values, config) -> np.ndarray:
    arr = _validate_observation_vector(name, values, config)
    if name.endswith("std_value"):
        bad_scale_indices = np.flatnonzero(arr <= 0.0)
        if bad_scale_indices.size:
            expected_dim = _expected_observation_dim(config)
            names = _observation_names(config, expected_dim)
            details = []
            for index in bad_scale_indices[:5]:
                label = names[index] if index < len(names) else f"index_{index}"
                details.append(f"{label}[{index}]={float(arr[index])!r}")
            suffix = "" if bad_scale_indices.size <= 5 else f", ... ({bad_scale_indices.size} total)"
            raise ValueError(f"{name} must contain positive scales: {', '.join(details)}{suffix}.")
    return arr


def build_raw_observation( # 관측값 생성
    current_state: Dict[str, float],
    config,
    reference: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    if reference is None:
        raise ValueError(
            "build_raw_observation requires an explicit reference. "
            "Compute it once in the environment step/reset path and pass it in."
        )

    obs_raw = np.array(
        [
            float(current_state["vx"]),
            float(current_state["vy"]),
            float(current_state["steer"]),
            float(current_state["steer_dot"]),
            float(current_state["yaw_rate"]),
            float(current_state["ax"]),
            float(current_state["ay"]),
            float(reference["target_curvature"]),
            float(reference["target_curvature_dot"]),
            float(reference["target_lateral_accel"]), # 목표 곡률을 속도로 증폭
            float(reference["target_lateral_accel_dot"]), # 목표 곡률 변화율을 속도 제곱으로 증폭
        ],
        dtype=np.float64,
    )
    return _validate_observation_vector("raw observation", obs_raw, config)


def cal_observation(
    current_state: Dict[str, float],
    config,
    reference: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    obs_raw = build_raw_observation(current_state, config, reference)
    avg = _validate_normalization_vector("config.obs_avg_value", config.obs_avg_value, config)
    std = _validate_normalization_vector("config.obs_std_value", config.obs_std_value, config)
    normalized_obs = (obs_raw - avg) / std
    normalized_obs = _validate_observation_vector("normalized observation", normalized_obs, config)
    normalized_obs_float32 = normalized_obs.astype(np.float32)
    _validate_observation_vector("float32 normalized observation", normalized_obs_float32, config)
    return normalized_obs_float32
