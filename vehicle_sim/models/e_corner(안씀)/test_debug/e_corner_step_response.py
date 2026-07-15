#!/usr/bin/env python3
"""
Simple step/ramp response visualization for a single E-Corner.
Plots suspension/tire forces, steering angle, and wheel speed over time.

Run: python vehicle_sim/models/e_corner/test_debug/e_corner_step_response.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from vehicle_sim.models.e_corner.e_corner import ECorner  # noqa: E402


def simulate_corner() -> dict:
    """Run a simple deterministic scenario for one corner and collect traces."""
    dt = 0.01
    t_end = 6.0
    n_steps = int(t_end / dt)
    time = np.arange(n_steps) * dt

    corner = ECorner(corner_id="FL")

    traces = {
        "time": time,
        "F_s": np.zeros(n_steps),
        "F_x": np.zeros(n_steps),
        "F_y": np.zeros(n_steps),
        "steering_angle": np.zeros(n_steps),
        "omega": np.zeros(n_steps),
        "heave": np.zeros(n_steps),
        "roll": np.zeros(n_steps),
        "pitch": np.zeros(n_steps),
        "T_steer": np.zeros(n_steps),
        "T_brk": np.zeros(n_steps),
        "T_drv": np.zeros(n_steps),
        "T_susp": np.zeros(n_steps),
    }

    # Deterministic input profiles
    heave_base = 0.3
    heave_amp = 0.02
    roll_amp = np.deg2rad(1.5)
    pitch_amp = np.deg2rad(1.0)

    for k, t in enumerate(time):
        T_steer = 1.5 * np.sin(0.5 * t)
        T_drv = 50.0 if t < 2.5 else 10.0
        T_brk = 0.0 if t < 4.0 else 30.0
        T_susp = 120.0 * np.sin(0.8 * t)

        heave = heave_base + heave_amp * np.sin(0.7 * t)
        roll = roll_amp * np.sin(0.4 * t)
        pitch = pitch_amp * np.sin(0.3 * t)

        heave_dot = heave_amp * 0.7 * np.cos(0.7 * t)
        roll_dot = roll_amp * 0.4 * np.cos(0.4 * t)
        pitch_dot = pitch_amp * 0.3 * np.cos(0.3 * t)

        V_wheel_x = 5.0 + 0.5 * np.sin(0.2 * t)
        V_wheel_y = 0.3 * np.sin(0.6 * t)

        # X_body array: [heave, roll, pitch, heave_dot, roll_rate, pitch_rate]
        X_body = np.array([heave, roll, pitch, heave_dot, roll_dot, pitch_dot])

        F_s, F_x, F_y = corner.update(
            dt=dt,
            T_steer=T_steer,
            T_brk=T_brk,
            T_Drv=T_drv,
            T_susp=T_susp,
            V_wheel_x=V_wheel_x,
            V_wheel_y=V_wheel_y,
            X_body=X_body,
        )

        traces["F_s"][k] = F_s
        traces["F_x"][k] = F_x
        traces["F_y"][k] = F_y
        traces["steering_angle"][k] = corner.state.steering_angle
        traces["omega"][k] = corner.state.omega_wheel
        traces["heave"][k] = heave
        traces["roll"][k] = roll
        traces["pitch"][k] = pitch
        traces["T_steer"][k] = T_steer
        traces["T_brk"][k] = T_brk
        traces["T_drv"][k] = T_drv
        traces["T_susp"][k] = T_susp

    return traces


def plot_traces(traces: dict) -> None:
    """Plot collected traces."""
    t = traces["time"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()

    axes[0].plot(t, traces["F_s"], label="F_s (suspension)")
    axes[0].plot(t, traces["F_x"], label="F_x (long)")
    axes[0].plot(t, traces["F_y"], label="F_y (lat)")
    axes[0].set_ylabel("Forces [N]")
    axes[0].legend()

    axes[1].plot(t, traces["steering_angle"], label="Steering angle")
    axes[1].set_ylabel("Angle [rad]")
    axes[1].legend()

    axes[2].plot(t, traces["omega"], label="Wheel speed")
    axes[2].set_ylabel("ω [rad/s]")
    axes[2].legend()

    axes[3].plot(t, traces["heave"], label="Heave")
    axes[3].plot(t, traces["roll"], label="Roll")
    axes[3].plot(t, traces["pitch"], label="Pitch")
    axes[3].set_ylabel("Body states")
    axes[3].legend()

    axes[4].plot(t, traces["T_drv"], label="T_drv")
    axes[4].plot(t, traces["T_brk"], label="T_brk")
    axes[4].plot(t, traces["T_steer"], label="T_steer")
    axes[4].plot(t, traces["T_susp"], label="T_susp")
    axes[4].set_ylabel("Inputs [N*m]")
    axes[4].legend()

    axes[5].plot(t, np.rad2deg(traces["steering_angle"]), label="Steer [deg]")
    axes[5].set_ylabel("Steer [deg]")
    axes[5].set_xlabel("Time [s]")
    axes[5].legend()

    fig.suptitle("E-Corner step/ramp response (FL)")
    fig.tight_layout()
    plt.show()


def main() -> None:
    traces = simulate_corner()
    plot_traces(traces)


if __name__ == "__main__":
    main()
