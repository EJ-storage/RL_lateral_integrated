from typing import Dict, Optional, Tuple
import math

import numpy as np


def calculate_current_curvature(current_state: Dict[str, float], wheelbase: float, eps: float) -> float: # 현재 차량 상태로부터 실제 차량 곡률 계산
    vx = float(current_state["vx"])
    vy = float(current_state["vy"])
    ax = float(current_state["ax"])
    ay = float(current_state["ay"])
    steer = float(current_state["steer"])
    yaw_rate = float(current_state["yaw_rate"])

    speed = math.hypot(vx, vy)
    denominator = (vx * vx + vy * vy) ** 1.5

    if speed < 1.0 or denominator < eps: # 분모가 0에 가까운 경우 자전거 모델 곡률 수식 사용
        return float(math.tan(steer) / wheelbase)
    # ax/ay are body-frame velocity derivatives. Path curvature is yaw-rate
    # curvature plus the velocity-vector sideslip-rate contribution.
    return float((yaw_rate / speed) + (vx * ay - vy * ax) / denominator)


def calculate_sideslip(current_state: Dict[str, float], eps: float) -> float: # 차량 상태로 부터 sidelip 계산
    vx = float(current_state["vx"])
    vy = float(current_state["vy"])
    denominator = vx
    if abs(denominator) < eps:
        denominator = eps if denominator >= 0.0 else -eps
    return float(vy / denominator)


def lateral_accel_limit_for_road_friction(config, road_friction: float, fallback: float) -> float:
    table = tuple(getattr(config, "curvature_error_lateral_accel_term_by_mu", ()))
    road_friction = float(road_friction)
    fallback = float(fallback)
    if not math.isfinite(road_friction) or not table:
        return fallback

    limits_by_mu = {float(mu): float(limit) for mu, limit in table}
    return limits_by_mu.get(road_friction, fallback)


def curvature_error_lateral_accel_limit_mps2(scenario, config) -> float:
    road_friction = float(getattr(scenario, "road_friction", float("nan")))
    return lateral_accel_limit_for_road_friction(
        config,
        road_friction,
        fallback=float(config.curvature_error_lateral_accel_term_mps2),
    )


def _require_finite_inputs(named_values: Dict[str, float]) -> None:
    invalid = []
    for name, value in named_values.items():
        value_float = float(value)
        if not math.isfinite(value_float):
            invalid.append(f"{name}={value_float!r}")
    if invalid:
        raise ValueError(
            "calculate_reward received non-finite inputs: "
            f"{', '.join(invalid)}."
        )


def calculate_reward( # 보상 계산
    current_state: Dict[str, float],
    scenario, # 사용 안함
    config, # 보상함수 파라미터 가지고 있는 객체
    action, # 현재 에이전트 행동
    reference: Optional[Dict[str, float]] = None, # 환경에서 계산한 reference 딕셔너리
) -> Tuple[float, bool, bool, Dict[str, float]]:
    vx = float(current_state["vx"])
    vy = float(current_state["vy"])
    ax = float(current_state["ax"])
    ay = float(current_state["ay"])
    steer = float(current_state["steer"])
    yaw_rate = float(current_state["yaw_rate"])

    if reference is None: # reference가 없으면 에러 발생
        raise ValueError(
            "calculate_reward requires an explicit reference. "
            "Compute it once in the environment step path and pass it in."
        )

    target_curvature = float(reference["target_curvature"]) # 목표 곡률
    action_value = float(np.asarray(action, dtype=np.float64).reshape(-1)[0]) # action 추출

    _require_finite_inputs(
        {
            "vx": vx,
            "vy": vy,
            "ax": ax,
            "ay": ay,
            "steer": steer,
            "yaw_rate": yaw_rate,
            "target_curvature": target_curvature,
            "action": action_value,
        }
    )
    terminated = False
    truncated = False
    terminated_reason = "none"
    curvature = 0.0
    curvature_error = 0.0
    beta = 0.0
    reward_track = 0.0
    reward_slip = 0.0
    reward_used = 0.0 # 상태 초기값 설정
        

    curvature = calculate_current_curvature( # 현재 차량 상태로부터 실제 차량 곡률 계산
        current_state,
        wheelbase=float(scenario.wheelbase),
        eps=float(config.curvature_denominator_eps),
    )

    beta = calculate_sideslip(# 차량 상태로 부터 sidelip 계산
        current_state,
        eps=float(config.curvature_denominator_eps),
    )

    curvature_error = target_curvature - curvature   # 목표 곡률과 실제 곡률의 차이 계산    

############################################보상 함수###############################################
    reward_track = math.exp( 
        -(
            float(config.K_kappa) * curvature_error * curvature_error
            + float(config.K_ay) * curvature_error * curvature_error * vx**4
        )
    ) - float(config.b_track)
    reward_slip = (
        float(config.w_slip)
        * math.exp(-float(config.K_slip) * (abs(beta) - float(config.beta_warn)) ** 2)
        - float(config.b_slip)
    )

    if abs(beta) <= float(config.beta_warn):
        reward_used = reward_track
    else:
        reward_used = reward_slip

    curvature_lateral_accel_error = abs(vx**2 * curvature_error)
    curvature_lateral_accel_limit = curvature_error_lateral_accel_limit_mps2(scenario, config)
    if curvature_lateral_accel_error > curvature_lateral_accel_limit:
        terminated = True
        terminated_reason = "curvature_lateral_accel_error"
    elif abs(beta) > float(config.beta_term):
        terminated = True
        terminated_reason = "sideslip_limit"

    if terminated:
        reward_total = -float(config.P_terminal) / (1.0 - float(config.gamma))
    else:
        reward_total = float(config.R_base) + float(reward_used)

################################################################################################

    info = {
        "reward_track": float(reward_track),
        "reward_slip": float(reward_slip),
        "reward_used": float(reward_used),
        "reward_total": float(reward_total),
        "curvature": float(curvature),
        "target_curvature": float(target_curvature),
        "curvature_error": float(curvature_error),
        "curvature_lateral_accel_error": float(abs(vx**2 * curvature_error)),
        "curvature_lateral_accel_limit_mps2": float(
            curvature_error_lateral_accel_limit_mps2(scenario, config)
        ),
        "beta": float(beta),
        "ay": float(ay) if math.isfinite(float(ay)) else 0.0,
        "terminated_reason": terminated_reason,
    }
    return float(reward_total), bool(terminated), bool(truncated), info
