from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stbwRL.custom_env.controllers.speed_controller import (  # noqa: E402
    SpeedPIConfig,
    SpeedPIController,
    SpeedPIOutput,
    target_accel_to_axle_torque,
)
from stbwRL.env_config import EnvironmentConfig  # noqa: E402
from vehicle_sim.stbw_model.vehicle_body.vehicle_body import StbwVehicleBody  # noqa: E402


CSV_PATH = PROJECT_ROOT / "vehicle_sim" / "eval_csv" / "weave_30s_31sig_speed2050_mu1_4.csv"
CONFIG_PATH = "stbw"
DRIVE_AXLE = "R"
BRAKE_AXLES = ("F", "R")
ROAD_MU = 1.0
STEER_INPUT_MODE = "average"
USE_SPEED_HOLD_PI = True
TARGET_SPEED_MPS = None
SPEED_HOLD_UPDATE_DT = 0.02
PLOT_BODY_COMPARISON = True
PLOT_WHEEL_OMEGA = True
PLOT_WHEEL_SPEEDS = True
PLOT_WHEEL_LOADS = True
PLOT_WHEEL_VELOCITIES = True
PLOT_WHEEL_SLIP_SPEED_INPUTS = True
PLOT_WHEEL_FORCES = True
PLOT_WHEEL_KAPPA = True

WHEEL_LABELS = ("FL", "FR", "RL", "RR")
BODY_COLUMNS = {
    "vx": "Car.vx",
    "vy": "Car.vy",
    "ax": "Car.ax",
    "ay": "Car.ay",
    "yaw_rate": "Car.YawRate",
}
WHEEL_FORCE_COMPONENTS = ("Fx", "Fy")


@dataclass
class SpeedHoldState:
    controller: SpeedPIController
    target_speed_mps: float
    drive_torque_nm: float = 0.0
    brake_motor_torque: float = 0.0
    last_update_time: Optional[float] = None
    last_output: Optional[SpeedPIOutput] = None


def make_speed_pi_config(config_path: str) -> SpeedPIConfig:
    env_config = EnvironmentConfig(config_path=config_path)
    return SpeedPIConfig(
        kp=float(env_config.speed_hold_kp),
        ki=float(env_config.speed_hold_ki),
        min_target_accel_mps2=float(env_config.speed_hold_min_accel_mps2),
        max_target_accel_mps2=float(env_config.speed_hold_max_accel_mps2),
        integrator_limit_mps=float(env_config.speed_hold_integrator_limit_mps),
        speed_deadband_mps=float(env_config.speed_hold_deadband_mps),
        config_path=None,
    )


def read_csv_columns(csv_path: Path, max_samples: Optional[int] = None) -> Dict[str, np.ndarray]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        names = list(reader.fieldnames)
        rows: List[Mapping[str, str]] = []
        for row in reader:
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break

    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    columns: Dict[str, np.ndarray] = {}
    for name in names:
        try:
            columns[name] = np.array([float(row[name]) for row in rows], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column {name!r} must be numeric.") from exc

    required = [
        "Time",
        "Car.SteerAngleFL",
        "Car.SteerAngleFR",
        *BODY_COLUMNS.values(),
        *(f"Car.WheelSpd_{label}" for label in WHEEL_LABELS),
        *(f"Car.vx{label}" for label in WHEEL_LABELS),
        *(f"Car.vy{label}" for label in WHEEL_LABELS),
        *(f"Car.Fz{label}" for label in WHEEL_LABELS),
    ]
    missing = [name for name in required if name not in columns]
    if missing:
        raise KeyError(f"Missing required CSV columns: {missing}")

    time = columns["Time"]
    if not np.all(np.isfinite(time)):
        raise ValueError("Time column contains non-finite values.")
    if np.any(np.diff(time) < 0.0):
        raise ValueError("Time column must be non-decreasing.")

    return columns


def has_wheel_force_columns(columns: Mapping[str, np.ndarray]) -> bool:
    return all(
        f"Car.{component}{label}" in columns
        for component in WHEEL_FORCE_COMPONENTS
        for label in WHEEL_LABELS
    )


def infer_target_speed_mps(
    csv_path: Path,
    columns: Mapping[str, np.ndarray],
) -> float:
    match = re.search(r"speed(\d{2})(\d{2})", csv_path.stem)
    if match:
        return float(match.group(2)) / 3.6
    return float(columns["Car.vx"][-1])


def front_road_wheel_angle(columns: Mapping[str, np.ndarray], mode: str) -> np.ndarray:
    fl = columns["Car.SteerAngleFL"]
    fr = columns["Car.SteerAngleFR"]

    if mode == "average":
        return 0.5 * (fl + fr)
    if mode == "fl":
        return fl
    if mode == "fr":
        return fr
    if mode == "opposed":
        return 0.5 * (fr - fl)
    raise ValueError(f"Unsupported steer input mode: {mode!r}")


def finite_difference(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=float)
    dt = np.diff(time)
    dv = np.diff(values)
    valid = dt > 1e-12
    result[1:] = np.divide(dv, dt, out=np.zeros_like(dv), where=valid)
    if result.size > 1:
        result[0] = result[1]
    return result


def make_vehicle(columns: Mapping[str, np.ndarray], config_path: str) -> StbwVehicleBody:
    vehicle = StbwVehicleBody(config_path=config_path, drive_axles=DRIVE_AXLE)
    vehicle.reset()
    vehicle.set_state_vector(
        np.array(
            [
                0.0,
                0.0,
                0.0,
                columns["Car.vx"][0],
                columns["Car.vy"][0],
                columns["Car.YawRate"][0],
                columns["Car.ax"][0],
                columns["Car.ay"][0],
            ],
            dtype=float,
        )
    )

    for label in WHEEL_LABELS:
        initial_wheel_speed = float(columns[f"Car.WheelSpd_{label}"][0])
        vehicle.wheels[label].drive.state.wheel_speed = initial_wheel_speed
        vehicle.wheels[label].state.omega_wheel = initial_wheel_speed

    return vehicle


def clip_front_steering(
    vehicle: StbwVehicleBody,
    angle: float,
    rate: float,
) -> Tuple[float, float]:
    steering = vehicle.wheels["FL"].steering
    if steering is None:
        return float(angle), float(rate)
    return (
        float(steering.apply_angle_limits(float(angle))),
        float(steering.apply_rate_limits(float(rate))),
    )


def mean_vehicle_brake_gains(vehicle: StbwVehicleBody) -> Tuple[float, float]:
    clamp_gains = []
    torque_gains = []
    for _, wheel in vehicle.iter_wheel_modules():
        clamp_gains.append(float(wheel.brake._clamp_gain))
        torque_gains.append(float(wheel.drive._clamp_to_torque))
    return float(np.mean(clamp_gains)), float(np.mean(torque_gains))


def update_speed_hold(
    speed_hold: SpeedHoldState,
    vehicle: StbwVehicleBody,
    current_time: float,
    update_dt: float,
) -> None:
    if (
        speed_hold.last_update_time is not None
        and current_time < speed_hold.last_update_time + update_dt - 1e-12
    ):
        return

    if speed_hold.last_update_time is None:
        controller_dt = update_dt
    else:
        controller_dt = max(current_time - speed_hold.last_update_time, 1e-9)

    speed_hold.last_update_time = float(current_time)
    speed_hold.last_output = speed_hold.controller.update(
        target_speed_mps=float(speed_hold.target_speed_mps),
        current_speed_mps=float(vehicle.state.velocity_x),
        dt=float(controller_dt),
    )

    clamp_gain, brake_torque_gain = mean_vehicle_brake_gains(vehicle)
    _, first_wheel = next(iter(vehicle.iter_wheel_modules()))
    command = target_accel_to_axle_torque(
        target_accel_mps2=speed_hold.last_output.target_accel_mps2,
        vehicle_mass_kg=float(vehicle.params.m_total),
        wheel_radius_m=float(first_wheel.drive.params.R_wheel),
        brake_clamp_gain=clamp_gain,
        brake_torque_gain=brake_torque_gain,
        brake_axle_count=len(BRAKE_AXLES),
    )
    speed_hold.drive_torque_nm = float(command.drive_torque_nm)
    speed_hold.brake_motor_torque = float(command.brake_motor_torque)


def axle_inputs(
    road_wheel_angle: float,
    road_wheel_rate: float,
    road_mu: float,
    speed_hold: Optional[SpeedHoldState],
) -> Dict[str, Dict[str, float]]:
    drive_torque = 0.0 if speed_hold is None else speed_hold.drive_torque_nm
    brake_torque = 0.0 if speed_hold is None else speed_hold.brake_motor_torque
    drive_axle = DRIVE_AXLE.upper()
    brake_axles = {label.upper() for label in BRAKE_AXLES}

    return {
        "F": {
            "T_steer": 0.0,
            "T_brk": brake_torque if "F" in brake_axles else 0.0,
            "T_Drv": drive_torque if drive_axle == "F" else 0.0,
            "steering_angle": float(road_wheel_angle),
            "steering_rate": float(road_wheel_rate),
            "road_mu": float(road_mu),
        },
        "R": {
            "T_steer": 0.0,
            "T_brk": brake_torque if "R" in brake_axles else 0.0,
            "T_Drv": drive_torque if drive_axle == "R" else 0.0,
            "road_mu": float(road_mu),
        },
    }


def capture_model_state(vehicle: StbwVehicleBody, time_s: float) -> Dict[str, float]:
    outputs = vehicle.get_outputs()
    vx = float(outputs["velocity_x"])
    vy = float(outputs["velocity_y"])
    yaw_rate = float(outputs["yaw_rate"])
    vx_dot = float(outputs["ax"])
    vy_dot = float(outputs["ay"])
    ax_cg = vx_dot - vy * yaw_rate
    ay_cg = vy_dot + vx * yaw_rate
    row = {
        "time": float(time_s),
        "vx": vx,
        "vy": vy,
        "ax": ax_cg,
        "vx_dot": vx_dot,
        "ay": ay_cg,
        "vy_dot": vy_dot,
        "yaw_rate": yaw_rate,
        "road_wheel_angle": float(outputs["front_road_wheel_angle"]),
        "road_wheel_rate": float(outputs["front_road_wheel_rate"]),
    }

    wheel_velocities = outputs["wheel_velocities"]
    for label in WHEEL_LABELS:
        wheel_state = vehicle.wheels[label].get_state()
        tire_params = vehicle.wheels[label].dugoff_tire.params
        row[f"WheelSpd_{label}"] = float(vehicle.wheels[label].drive.state.wheel_speed)
        row[f"Omega_{label}"] = float(wheel_state["omega_wheel"])
        row[f"Fz{label}"] = float(wheel_state["F_z"])
        row[f"Fx{label}"] = float(wheel_state["F_x_tire"])
        row[f"Fy{label}"] = float(wheel_state["F_y_tire"])
        row[f"Kappa{label}"] = float(wheel_state["dugoff_kappa"])
        row[f"WheelLinearSpeed_{label}"] = float(
            wheel_state["dugoff_wheel_linear_speed"]
        )
        row[f"DugoffVx_{label}"] = float(wheel_state["dugoff_vx"])
        row[f"TireRe_{label}"] = float(
            getattr(tire_params, f"Re_{label}")
        )
        row[f"TireVeps_{label}"] = float(getattr(tire_params, "Veps"))
        row[f"TireKappaMin_{label}"] = float(getattr(tire_params, "kappaMin"))
        row[f"TireKappaMax_{label}"] = float(getattr(tire_params, "kappaMax"))
        row[f"vx{label}"] = float(wheel_velocities[label][0])
        row[f"vy{label}"] = float(wheel_velocities[label][1])

    return row


def dict_rows_to_arrays(rows: Sequence[Mapping[str, float]]) -> Dict[str, np.ndarray]:
    if not rows:
        raise ValueError("No model rows were captured.")
    return {
        key: np.array([float(row[key]) for row in rows], dtype=float)
        for key in rows[0].keys()
    }


def simulate_vehicle(
    columns: Mapping[str, np.ndarray],
    *,
    config_path: str,
    road_mu: float,
    steer_mode: str,
    use_speed_hold: bool,
    target_speed_mps: Optional[float],
    speed_hold_update_dt: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    time = columns["Time"]
    input_angle = front_road_wheel_angle(columns, steer_mode)
    input_rate = finite_difference(input_angle, time)
    vehicle = make_vehicle(columns, config_path)

    if target_speed_mps is None:
        target_speed_mps = float(columns["Car.vx"][-1])

    speed_hold = None
    if use_speed_hold:
        speed_hold = SpeedHoldState(
            controller=SpeedPIController(make_speed_pi_config(config_path)),
            target_speed_mps=float(target_speed_mps),
        )

    rows: List[Dict[str, float]] = []

    angle0, rate0 = clip_front_steering(vehicle, input_angle[0], input_rate[0])
    vehicle.set_front_steering_state(angle0, rate0)
    rows.append(capture_model_state(vehicle, time[0]))

    for i in range(1, len(time)):
        dt = float(time[i] - time[i - 1])
        if dt <= 1e-12:
            rows.append(capture_model_state(vehicle, time[i]))
            continue

        angle, rate = clip_front_steering(vehicle, input_angle[i], input_rate[i])
        vehicle.set_front_steering_state(angle, rate)

        if speed_hold is not None:
            update_speed_hold(
                speed_hold,
                vehicle,
                current_time=float(time[i - 1]),
                update_dt=float(speed_hold_update_dt),
            )

        vehicle.update(
            dt=dt,
            axle_inputs=axle_inputs(angle, rate, road_mu, speed_hold),
            direction=1,
        )
        rows.append(capture_model_state(vehicle, time[i]))

    return dict_rows_to_arrays(rows), input_angle, input_rate


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def print_summary(
    columns: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
    target_speed_mps: Optional[float],
    use_speed_hold: bool,
) -> None:
    print("Replay summary")
    print(f"  samples: {len(columns['Time'])}")
    print(f"  duration: {columns['Time'][-1] - columns['Time'][0]:.3f} s")
    print(f"  speed_hold: {use_speed_hold}")
    if use_speed_hold:
        target = float(columns["Car.vx"][-1] if target_speed_mps is None else target_speed_mps)
        print(f"  target_speed: {target:.6f} m/s ({target * 3.6:.3f} kph)")

    speed_error = model["vx"] - columns["Car.vx"]
    print("  speed comparison: model vx - CarMaker vx")
    print(f"    initial_csv:   {columns['Car.vx'][0]:.6f} m/s ({columns['Car.vx'][0] * 3.6:.3f} kph)")
    print(f"    initial_model: {model['vx'][0]:.6f} m/s ({model['vx'][0] * 3.6:.3f} kph)")
    print(f"    final_csv:     {columns['Car.vx'][-1]:.6f} m/s ({columns['Car.vx'][-1] * 3.6:.3f} kph)")
    print(f"    final_model:   {model['vx'][-1]:.6f} m/s ({model['vx'][-1] * 3.6:.3f} kph)")
    print(f"    final_error:   {speed_error[-1]:.6g} m/s ({speed_error[-1] * 3.6:.6g} kph)")
    print(f"    rmse:          {rmse(model['vx'], columns['Car.vx']):.6g} m/s")
    print(f"    max_abs_error: {np.max(np.abs(speed_error)):.6g} m/s")

    print("  wheel speed comparison: model vs CarMaker [rad/s]")
    for label in WHEEL_LABELS:
        model_speed = model[f"WheelSpd_{label}"]
        csv_speed = columns[f"Car.WheelSpd_{label}"]
        print(
            f"    {label}: "
            f"model first/final/min/max="
            f"{model_speed[0]:.6f}/{model_speed[-1]:.6f}/"
            f"{np.min(model_speed):.6f}/{np.max(model_speed):.6f}, "
            f"csv first/final/min/max="
            f"{csv_speed[0]:.6f}/{csv_speed[-1]:.6f}/"
            f"{np.min(csv_speed):.6f}/{np.max(csv_speed):.6f}"
        )

    if has_wheel_force_columns(columns):
        print("  wheel tire force comparison: model - CarMaker [N]")
        for component in WHEEL_FORCE_COMPONENTS:
            print(f"    {component}:")
            for label in WHEEL_LABELS:
                error = model[f"{component}{label}"] - columns[f"Car.{component}{label}"]
                print(
                    f"      {label}: "
                    f"rmse={rmse(model[f'{component}{label}'], columns[f'Car.{component}{label}']):.6g}, "
                    f"mean_error={np.mean(error):.6g}, "
                    f"final_error={error[-1]:.6g}, "
                    f"max_abs_error={np.max(np.abs(error)):.6g}"
                )
    else:
        print("  wheel tire force comparison: skipped; CSV has no Car.Fx*/Car.Fy* columns")


def plot_pair(
    ax,
    time: np.ndarray,
    csv_values: np.ndarray,
    model_values: np.ndarray,
    *,
    title: str,
    ylabel: str,
    scale: float = 1.0,
) -> None:
    ax.plot(time, csv_values * scale, label="csv", linewidth=1.4)
    ax.plot(time, model_values * scale, label="model", linewidth=1.1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_body_comparison(
    columns: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
    input_angle: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    time = columns["Time"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    fig.suptitle("CSV road wheel replay vs vehicle model")

    ax = axes[0, 0]
    ax.plot(time, np.rad2deg(input_angle), label="csv input", linewidth=1.4)
    ax.plot(
        time,
        np.rad2deg(columns["Car.SteerAngleFL"]),
        label="csv FL raw",
        linewidth=0.8,
        alpha=0.45,
    )
    ax.plot(
        time,
        np.rad2deg(columns["Car.SteerAngleFR"]),
        label="csv FR raw",
        linewidth=0.8,
        alpha=0.45,
    )
    ax.plot(time, np.rad2deg(model["road_wheel_angle"]), label="model applied", linewidth=1.1)
    ax.set_title("front road wheel angle")
    ax.set_ylabel("deg")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plot_pair(axes[0, 1], time, columns["Car.vx"], model["vx"], title="vx", ylabel="m/s")
    plot_pair(axes[1, 0], time, columns["Car.vy"], model["vy"], title="vy", ylabel="m/s")
    plot_pair(axes[1, 1], time, columns["Car.ax"], model["ax"], title="ax (CG)", ylabel="m/s^2")
    plot_pair(axes[2, 0], time, columns["Car.ay"], model["ay"], title="ay (CG)", ylabel="m/s^2")
    plot_pair(
        axes[2, 1],
        time,
        columns["Car.YawRate"],
        model["yaw_rate"],
        title="yaw rate",
        ylabel="deg/s",
        scale=180.0 / math.pi,
    )

    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()


def plot_wheel_grid(
    columns: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
    *,
    csv_prefix: str,
    model_prefix: str,
    title: str,
    ylabel: str,
) -> None:
    import matplotlib.pyplot as plt

    time = columns["Time"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(title)

    for ax, label in zip(axes.flat, WHEEL_LABELS):
        model_values = model[f"{model_prefix}{label}"]
        if model_prefix == "WheelSpd_" and float(np.nanmax(np.abs(model_values))) <= 1.0e-9:
            raise RuntimeError(
                f"Model wheel speed for {label} is all zero before plotting. "
                "Check the wheel-speed capture path."
            )
        plot_pair(
            ax,
            time,
            columns[f"{csv_prefix}{label}"],
            model_values,
            title=label,
            ylabel=ylabel,
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()


def plot_wheel_velocity_grid(
    columns: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
    *,
    component: str,
) -> None:
    import matplotlib.pyplot as plt

    time = columns["Time"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(f"wheel local {component}")

    for ax, label in zip(axes.flat, WHEEL_LABELS):
        plot_pair(
            ax,
            time,
            columns[f"Car.{component}{label}"],
            model[f"{component}{label}"],
            title=label,
            ylabel="m/s",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()


def plot_wheel_slip_speed_inputs(
    columns: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    time = columns["Time"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("wheel speed-derived longitudinal velocity vs vx")

    for ax, label in zip(axes.flat, WHEEL_LABELS):
        tire_re = float(model[f"TireRe_{label}"][0])
        ax.plot(
            time,
            columns[f"Car.WheelSpd_{label}"] * tire_re,
            label="csv Re*WheelSpd",
            linewidth=1.2,
        )
        ax.plot(
            time,
            columns[f"Car.vx{label}"],
            label="csv vx",
            linewidth=1.2,
        )
        ax.plot(
            time,
            model[f"WheelLinearSpeed_{label}"],
            label="model wheel_linear_speed",
            linewidth=1.0,
        )
        ax.plot(
            time,
            model[f"DugoffVx_{label}"],
            label="model vx",
            linewidth=1.0,
        )
        ax.set_title(label)
        ax.set_ylabel("m/s")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()


def plot_wheel_force_grid(
    columns: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
    *,
    component: str,
) -> None:
    import matplotlib.pyplot as plt

    if component not in WHEEL_FORCE_COMPONENTS:
        raise ValueError(f"Unsupported wheel force component: {component!r}")

    required = [f"Car.{component}{label}" for label in WHEEL_LABELS]
    missing = [name for name in required if name not in columns]
    if missing:
        raise KeyError(f"Missing wheel force CSV columns: {missing}")

    time = columns["Time"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(f"wheel tire {component}")

    for ax, label in zip(axes.flat, WHEEL_LABELS):
        plot_pair(
            ax,
            time,
            columns[f"Car.{component}{label}"],
            model[f"{component}{label}"],
            title=label,
            ylabel="N",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()


def calculate_csv_kappa(
    columns: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
    label: str,
) -> np.ndarray:
    tire_re = float(model[f"TireRe_{label}"][0])
    tire_veps = max(float(model[f"TireVeps_{label}"][0]), 1.0e-9)
    kappa_min = float(model[f"TireKappaMin_{label}"][0])
    kappa_max = float(model[f"TireKappaMax_{label}"][0])

    wheel_linear_speed = columns[f"Car.WheelSpd_{label}"] * tire_re
    vx = columns[f"Car.vx{label}"]
    kappa = (wheel_linear_speed - vx) / np.maximum(np.abs(vx), tire_veps)
    if kappa_min < kappa_max:
        kappa = np.clip(kappa, kappa_min, kappa_max)
    return np.asarray(kappa, dtype=float)


def plot_wheel_kappa_grid(
    columns: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    time = columns["Time"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("wheel tire kappa")

    for ax, label in zip(axes.flat, WHEEL_LABELS):
        ax.plot(
            time,
            calculate_csv_kappa(columns, model, label),
            label="csv kappa",
            linewidth=1.4,
        )
        ax.plot(
            time,
            model[f"Kappa{label}"],
            label="model kappa",
            linewidth=1.2,
        )
        ax.set_title(label)
        ax.set_ylabel("kappa [-]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    fig.tight_layout()


def plot_all(
    columns: Mapping[str, np.ndarray],
    model: Mapping[str, np.ndarray],
    input_angle: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    if PLOT_BODY_COMPARISON:
        plot_body_comparison(columns, model, input_angle)

    if PLOT_WHEEL_OMEGA:
        plot_wheel_grid(
            columns,
            model,
            csv_prefix="Car.WheelSpd_",
            model_prefix="Omega_",
            title="wheel omega",
            ylabel="rad/s",
        )

    if PLOT_WHEEL_SPEEDS:
        plot_wheel_grid(
            columns,
            model,
            csv_prefix="Car.WheelSpd_",
            model_prefix="WheelSpd_",
            title="wheel speed",
            ylabel="rad/s",
        )

    if PLOT_WHEEL_LOADS:
        plot_wheel_grid(
            columns,
            model,
            csv_prefix="Car.Fz",
            model_prefix="Fz",
            title="vertical load",
            ylabel="N",
        )

    if PLOT_WHEEL_VELOCITIES:
        plot_wheel_velocity_grid(columns, model, component="vx")
        plot_wheel_velocity_grid(columns, model, component="vy")

    if PLOT_WHEEL_SLIP_SPEED_INPUTS:
        plot_wheel_slip_speed_inputs(columns, model)

    if PLOT_WHEEL_FORCES and has_wheel_force_columns(columns):
        plot_wheel_force_grid(columns, model, component="Fx")
        plot_wheel_force_grid(columns, model, component="Fy")

    if PLOT_WHEEL_KAPPA:
        plot_wheel_kappa_grid(columns, model)

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay CSV front road wheel angle through the local STBW vehicle model "
            "and plot CSV/model outputs without saving figures."
        )
    )
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--config-path", default=CONFIG_PATH)
    parser.add_argument("--road-mu", type=float, default=ROAD_MU)
    parser.add_argument(
        "--steer-mode",
        choices=("average", "fl", "fr", "opposed"),
        default=STEER_INPUT_MODE,
        help=(
            "How to reduce Car.SteerAngleFL/FR to one front road-wheel angle. "
            "average removes symmetric toe in this straight CSV."
        ),
    )
    parser.add_argument("--target-speed-mps", type=float, default=TARGET_SPEED_MPS)
    parser.add_argument("--target-speed-kph", type=float, default=None)
    parser.add_argument("--no-speed-hold", action="store_true")
    parser.add_argument("--speed-hold-update-dt", type=float, default=SPEED_HOLD_UPDATE_DT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    columns = read_csv_columns(args.csv, max_samples=args.max_samples)

    target_speed_mps = args.target_speed_mps
    if args.target_speed_kph is not None:
        target_speed_mps = float(args.target_speed_kph) / 3.6
    elif target_speed_mps is None:
        target_speed_mps = infer_target_speed_mps(args.csv, columns)

    use_speed_hold = USE_SPEED_HOLD_PI and not bool(args.no_speed_hold)
    model, input_angle, _ = simulate_vehicle(
        columns,
        config_path=str(args.config_path),
        road_mu=float(args.road_mu),
        steer_mode=str(args.steer_mode),
        use_speed_hold=use_speed_hold,
        target_speed_mps=target_speed_mps,
        speed_hold_update_dt=float(args.speed_hold_update_dt),
    )

    print_summary(columns, model, target_speed_mps, use_speed_hold)

    if not args.no_show:
        plot_all(columns, model, input_angle)


if __name__ == "__main__":
    main()
