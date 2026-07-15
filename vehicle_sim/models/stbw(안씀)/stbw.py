"""Integrated E-corner and steer-by-wire axle models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from vehicle_sim.utils.config_loader import load_param
from vehicle_sim.models.drive_layout import drive_axles_label, is_axle_driven

from .drive.brake_model import BrakeModel, StbwBrakeModel
from .drive.drive_model import DriveModel, StbwDriveModel
from .steering.steering_model import SteeringModel, StbwSteeringModel
from .suspension.suspension_model import SuspensionModel
from .tire.lateral.lateral_tire import LateralTireModel, StbwLateralTireModel
from .tire.longitudinal.longitudinal_tire import LongitudinalTireModel, StbwLongitudinalTireModel


@dataclass
class StbwState:
    F_x_tire: float = 0.0
    F_y_tire: float = 0.0
    F_x_tire_raw: float = 0.0
    F_y_tire_raw: float = 0.0
    F_z: float = 0.0
    friction_circle_limit: float = 0.0
    friction_circle_usage: float = 0.0
    friction_circle_usage_raw: float = 0.0
    friction_circle_scale: float = 1.0
    friction_circle_saturated: bool = False
    steering_angle: float = 0.0
    steering_rate: float = 0.0
    omega_wheel: float = 0.0


@dataclass
class StbwParameters:
    axle_id: Optional[str] = None
    axle_name: Optional[str] = None
    config: Optional[Dict] = None
    config_path: Optional[str] = None


class Stbw:
    """Steer-by-wire wheel/axle module used by the RL environment.

    Modified version:
    - Replaces the explicit/semi-explicit longitudinal wheel-tire update with
      an implicit wheel-tire solve.
    - The implicit solve is applied once per axle update.
    - No LPF is used.
    - Longitudinal relaxation length is bypassed; calculated kappa is used
      directly.
    - The longitudinal tire/drive coupling order is implicit.

    Important:
    This class assumes the same imports/classes as your existing file:
        np
        Optional, Tuple, Dict
        load_param
        StbwParameters, StbwState
        StbwSteeringModel, StbwBrakeModel, StbwDriveModel
        StbwLongitudinalTireModel, StbwLateralTireModel
    """

    IMPLICIT_MAX_ITER = 35
    IMPLICIT_TOL_OMEGA = 1e-7
    IMPLICIT_BRACKET_EXPANSIONS = 12

    def __init__(
        self,
        axle_id: Optional[str] = None,
        params: Optional[StbwParameters] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
    ):
        if params is not None:
            if not isinstance(params, StbwParameters):
                raise TypeError("params must be an StbwParameters instance")
            axle_id = axle_id or params.axle_id
            if config is None:
                config = params.config
            if config_path is None:
                config_path = params.config_path

        if axle_id not in ["F", "R", "FL", "FR", "RL", "RR"]:
            raise ValueError(
                f"Invalid axle_id: {axle_id}. Must be one of ['F', 'R', 'FL', 'FR', 'RL', 'RR']"
            )

        self.axle_id = axle_id
        self.axle_group = "F" if str(axle_id).startswith("F") else "R"
        self.is_corner = axle_id in ["FL", "FR", "RL", "RR"]
        self.config = config or {}
        self.config_path = config_path
        self.state = StbwState()
        self.has_steering = self.axle_group == "F"
        self.drive_axles = drive_axles_label(self.config.get("drive_axles", "R"))
        self.has_drive_motor = is_axle_driven(self.axle_group, self.drive_axles)

        vehicle_param = load_param("vehicle_body", config_path)
        physics_param = load_param("physics", config_path)
        geometry_param = load_param("vehicle_spec", config_path)
        geometry_cfg = geometry_param.get("geometry", {})

        self.m = float(vehicle_param.get("m", 1500.0))
        self.g = float(physics_param.get("g", 9.81))
        self.lf = float(geometry_cfg.get("lf", 1.155))
        self.lr = float(geometry_cfg.get("lr", 1.815))
        self.wheelbase = self.lf + self.lr

        if self.has_steering:
            self.F_z_static = self.m * self.g * self.lr / self.wheelbase
            steering_cfg = self._build_steering_config(self.axle_group, self.config, config_path)
            self.steering = StbwSteeringModel(config=steering_cfg, config_path=config_path, axle_id=self.axle_group)
        else:
            self.F_z_static = self.m * self.g * self.lf / self.wheelbase
            self.steering = None
        if self.is_corner:
            self.F_z_static *= 0.5

        self.brake = StbwBrakeModel(config_path=config_path)
        self.drive = StbwDriveModel(config_path=config_path, axle_id=self.axle_group)
        self.longitudinal_tire = StbwLongitudinalTireModel(config_path=config_path)
        self.lateral_tire = StbwLateralTireModel(config_path=config_path)
        self._road_mu_override: Optional[float] = None

    def set_drive_axles(self, drive_axles: str) -> None:
        self.drive_axles = drive_axles_label(drive_axles)
        self.has_drive_motor = is_axle_driven(self.axle_group, self.drive_axles)

    # -------------------------------------------------------------------------
    # Longitudinal tire preview functions
    # -------------------------------------------------------------------------
    # These functions compute trial kappa/Fx without modifying tire state.
    # They are used only inside the implicit solver. After omega_new is solved,
    # the selected kappa/Fx is committed once to the tire state.

    def _clip_slip_ratio(self, kappa: float) -> float:
        max_abs = float(getattr(self.longitudinal_tire, "MAX_ABS_SLIP_RATIO", 5.0))
        if not np.isfinite(kappa):
            return 0.0
        return float(np.clip(kappa, -max_abs, max_abs))

    def _preview_slip_ratio_cmd(self, omega_wheel: float, V_wx_local: float) -> Tuple[float, float, float, float]:
        tire_params = self.longitudinal_tire.params

        R_eff = float(getattr(tire_params, "R_eff", getattr(self.drive.params, "R_wheel", 0.3)))
        v_min = float(getattr(tire_params, "v_min", 0.1))

        V_wheel = float(omega_wheel) * R_eff
        denom = max(abs(float(V_wx_local)), v_min)
        kappa_cmd = (V_wheel - float(V_wx_local)) / denom
        kappa_cmd = self._clip_slip_ratio(kappa_cmd)

        return float(kappa_cmd), float(V_wheel), float(denom), float(R_eff)

    def _preview_kappa(self, kappa_cmd: float) -> float:
        return self._clip_slip_ratio(kappa_cmd)

    def _preview_longitudinal_force_from_kappa(self, kappa: float, F_z: float) -> float:
        """Preview Fx without modifying tire state.

        This mirrors the polynomial/saturated longitudinal tire force form used
        in the user's current StbwLongitudinalTireModel.calculate_force().
        If your calculate_force() implementation is different, keep the same
        state-safe structure but replace only this function body to match it.
        """
        tire_params = self.longitudinal_tire.params

        C_x = float(getattr(tire_params, "C_x", 231500.0))
        mu = self._effective_longitudinal_mu()
        F_z = max(float(F_z), 0.0)

        if not np.isfinite(kappa) or not np.isfinite(C_x) or not np.isfinite(mu) or not np.isfinite(F_z):
            return 0.0
        if C_x <= 1e-9 or mu <= 1e-9 or F_z <= 1e-9:
            return 0.0

        sign = 1.0 if kappa >= 0.0 else -1.0
        k = abs(float(kappa))

        kappa_sl = (3.0 * mu * F_z) / C_x
        if kappa_sl <= 1e-12:
            return 0.0

        if k <= kappa_sl:
            Fx_abs = (
                C_x * k
                - (C_x ** 2.0) / (3.0 * mu * F_z) * (k ** 2.0)
                + (C_x ** 3.0) / (27.0 * (mu * F_z) ** 2.0) * (k ** 3.0)
            )
        else:
            Fx_abs = mu * F_z

        Fx = sign * Fx_abs
        if not np.isfinite(Fx):
            return 0.0
        return float(Fx)

    def _combined_friction_limit(self, F_z: float) -> float:
        longitudinal_mu = self._effective_longitudinal_mu()
        lateral_mu = self._effective_lateral_mu()
        mu = min(longitudinal_mu, lateral_mu)
        if not np.isfinite(mu) or mu <= 0.0:
            return 0.0
        return float(mu * max(float(F_z), 0.0))

    def _effective_longitudinal_mu(self) -> float:
        if self._road_mu_override is not None:
            return float(self._road_mu_override)
        return float(getattr(self.longitudinal_tire.params, "mu", 0.0))

    def _effective_lateral_mu(self) -> float:
        if self._road_mu_override is not None:
            return float(self._road_mu_override)
        return float(getattr(self.lateral_tire.params, "mu", 0.0))

    def _apply_combined_friction_limit(
        self,
        F_x_tire_raw: float,
        F_y_tire_raw: float,
        F_z: float,
    ) -> Tuple[float, float, float, float, float, float, bool]:
        F_x_raw = float(F_x_tire_raw) if np.isfinite(F_x_tire_raw) else 0.0
        F_y_raw = float(F_y_tire_raw) if np.isfinite(F_y_tire_raw) else 0.0
        limit = self._combined_friction_limit(F_z)
        raw_norm = float(np.hypot(F_x_raw, F_y_raw))

        if limit <= 1e-9 or raw_norm <= 1e-9 or not np.isfinite(raw_norm):
            scale = 1.0
            usage_raw = 0.0 if limit <= 1e-9 else raw_norm / limit
            usage = usage_raw
            saturated = False
            return F_x_raw, F_y_raw, float(limit), float(usage), float(usage_raw), scale, saturated

        usage_raw = raw_norm / limit
        scale = min(1.0, limit / raw_norm)
        F_x = F_x_raw * scale
        F_y = F_y_raw * scale
        usage = float(np.hypot(F_x, F_y) / limit)
        saturated = bool(usage_raw > 1.0)
        return (
            float(F_x),
            float(F_y),
            float(limit),
            float(usage),
            float(usage_raw),
            float(scale),
            saturated,
        )

    def _preview_lateral_tire_force(
        self,
        V_wx_local: float,
        V_wy_local: float,
        F_z: float,
    ) -> Tuple[float, float]:
        alpha = self.lateral_tire.calculate_slip_angle(V_wx_local, V_wy_local)
        F_z = abs(float(F_z))
        mu = self._effective_lateral_mu()
        C_alpha = float(getattr(self.lateral_tire.params, "C_alpha", 0.0))
        if F_z <= 1e-6 or mu <= 1e-9 or C_alpha <= 1e-9:
            F_y_tire = 0.0
        else:
            tan_alpha = np.tan(alpha)
            alpha_sl = np.arctan((3.0 * mu * F_z) / C_alpha)
            if abs(alpha) < alpha_sl:
                F_y_tire = (
                    -C_alpha * tan_alpha
                    + (C_alpha ** 2 / (3.0 * mu * F_z)) * abs(tan_alpha) * tan_alpha
                    - (C_alpha ** 3 / (27.0 * mu ** 2 * F_z ** 2)) * (tan_alpha ** 3)
                )
            else:
                F_y_tire = -mu * F_z * np.sign(alpha)
        return float(alpha), float(F_y_tire)

    def _preview_longitudinal_tire_at_omega(
        self,
        dt: float,
        omega_wheel: float,
        V_wx_local: float,
        F_z: float,
    ) -> Tuple[float, float, float, float, float, float]:
        kappa_cmd, V_wheel, denom, R_eff = self._preview_slip_ratio_cmd(
            omega_wheel=omega_wheel,
            V_wx_local=V_wx_local,
        )
        kappa = self._preview_kappa(kappa_cmd)
        F_x_tire = self._preview_longitudinal_force_from_kappa(kappa, F_z)

        return (
            float(F_x_tire),
            float(kappa),
            float(kappa_cmd),
            float(V_wheel),
            float(denom),
            float(R_eff),
        )

    def _commit_longitudinal_tire_state(
        self,
        F_x_tire: float,
        F_x_tire_raw: float,
        combined_force_scale: float,
        kappa: float,
        kappa_cmd: float,
        V_wheel: float,
        V_wx_local: float,
        denom: float,
    ) -> None:
        tire_state = self.longitudinal_tire.state

        # Existing state fields
        if hasattr(tire_state, "slip_ratio_cmd"):
            tire_state.slip_ratio_cmd = float(kappa_cmd)
        if hasattr(tire_state, "slip_ratio"):
            tire_state.slip_ratio = float(kappa)
        if hasattr(tire_state, "longitudinal_force"):
            tire_state.longitudinal_force = float(F_x_tire)

        # Debug/info fields if the state object allows dynamic attributes.
        # These are useful because evaluate.py already tries to log them via info.
        try:
            tire_state.longitudinal_force_raw = float(F_x_tire_raw)
            tire_state.combined_force_scale = float(combined_force_scale)
            tire_state.V_wheel = float(V_wheel)
            tire_state.V_wheel_x = float(V_wx_local)
            tire_state.V_wheel_x_minus_V_wheel = float(V_wx_local - V_wheel)
            tire_state.slip_ratio_denom = float(denom)
            tire_state.denom = float(denom)
        except Exception:
            pass

    def _commit_lateral_tire_state(
        self,
        alpha: float,
        F_y_tire: float,
        F_y_tire_raw: float,
        combined_force_scale: float,
    ) -> None:
        tire_state = self.lateral_tire.state
        M_z = self.lateral_tire.calculate_aligning_torque(F_y_tire)
        tire_state.slip_angle = float(alpha)
        tire_state.slip_angle_dyn = float(alpha)
        tire_state.lateral_force = float(F_y_tire)
        tire_state.aligning_torque = float(M_z)
        try:
            tire_state.lateral_force_raw = float(F_y_tire_raw)
            tire_state.combined_force_scale = float(combined_force_scale)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Drive preview/commit functions
    # -------------------------------------------------------------------------

    def _apply_wheel_speed_limit(self, omega: float) -> float:
        apply_speed_limits = getattr(self.drive, "apply_speed_limits", None)
        if callable(apply_speed_limits):
            return float(apply_speed_limits(float(omega)))

        max_wheel_speed = float(getattr(self.drive.params, "max_wheel_speed", float("inf")))
        if np.isfinite(max_wheel_speed) and max_wheel_speed > 0.0:
            return float(np.clip(float(omega), -max_wheel_speed, max_wheel_speed))
        return float(omega)

    def _preview_signed_brake_torque(self, omega_wheel: float, F_clamp: float) -> Tuple[float, float]:
        """Preview signed brake torque without modifying brake/drive state."""
        clamp_to_torque = float(getattr(self.drive, "_clamp_to_torque", 0.0))
        F_clamp_eff = max(float(F_clamp), 0.0)
        M_brk_abs = clamp_to_torque * F_clamp_eff

        omega0 = 0.7
        M_brk_signed = -M_brk_abs * float(np.tanh(float(omega_wheel) / omega0))
        return float(M_brk_signed), float(M_brk_abs)

    def _preview_wheel_rhs(
        self,
        dt: float,
        omega_old: float,
        omega_candidate: float,
        T_Drv: float,
        F_x_tire: float,
        F_clamp: float,
        direction: int,
    ) -> Tuple[float, float, float, float, float, float, float]:
        drive_params = self.drive.params

        J_wheel = float(getattr(drive_params, "J_wheel", 1.0))
        R_wheel = float(getattr(drive_params, "R_wheel", 0.3))
        B_wheel = float(getattr(drive_params, "B_wheel", 0.0))

        if not np.isfinite(J_wheel) or J_wheel <= 1e-12:
            J_wheel = 1.0

        T_Drv_signed = float(direction) * float(T_Drv)
        T_tire_reaction = R_wheel * float(F_x_tire)
        M_brk_signed, M_brk_abs = self._preview_signed_brake_torque(
            omega_wheel=omega_candidate,
            F_clamp=F_clamp,
        )
        M_visc = B_wheel * float(omega_candidate)

        T_net = T_Drv_signed - T_tire_reaction + M_brk_signed - M_visc
        alpha = T_net / J_wheel

        omega_rhs_unclipped = float(omega_old) + float(dt) * alpha
        omega_rhs = self._apply_wheel_speed_limit(omega_rhs_unclipped)

        return (
            float(omega_rhs),
            float(T_Drv_signed),
            float(T_tire_reaction),
            float(M_brk_signed),
            float(M_brk_abs),
            float(M_visc),
            float(T_net),
        )

    def _implicit_residual(
        self,
        omega_candidate: float,
        dt: float,
        omega_old: float,
        T_Drv: float,
        F_clamp: float,
        V_wx_local: float,
        F_z: float,
        F_y_tire_raw: float,
        direction: int,
    ) -> Tuple[float, Dict[str, float]]:
        F_x_tire_raw, kappa, kappa_cmd, V_wheel, denom, R_eff = self._preview_longitudinal_tire_at_omega(
            dt=dt,
            omega_wheel=omega_candidate,
            V_wx_local=V_wx_local,
            F_z=F_z,
        )
        (
            F_x_tire,
            F_y_tire,
            friction_circle_limit,
            friction_circle_usage,
            friction_circle_usage_raw,
            friction_circle_scale,
            friction_circle_saturated,
        ) = self._apply_combined_friction_limit(
            F_x_tire_raw=F_x_tire_raw,
            F_y_tire_raw=F_y_tire_raw,
            F_z=F_z,
        )

        (
            omega_rhs,
            T_Drv_signed,
            T_tire_reaction,
            M_brk_signed,
            M_brk_abs,
            M_visc,
            T_net,
        ) = self._preview_wheel_rhs(
            dt=dt,
            omega_old=omega_old,
            omega_candidate=omega_candidate,
            T_Drv=T_Drv,
            F_x_tire=F_x_tire,
            F_clamp=F_clamp,
            direction=direction,
        )

        residual = float(omega_candidate) - float(omega_rhs)

        return residual, {
            "omega_rhs": float(omega_rhs),
            "F_x_tire": float(F_x_tire),
            "F_y_tire": float(F_y_tire),
            "F_x_tire_raw": float(F_x_tire_raw),
            "F_y_tire_raw": float(F_y_tire_raw),
            "friction_circle_limit": float(friction_circle_limit),
            "friction_circle_usage": float(friction_circle_usage),
            "friction_circle_usage_raw": float(friction_circle_usage_raw),
            "friction_circle_scale": float(friction_circle_scale),
            "friction_circle_saturated": float(friction_circle_saturated),
            "kappa": float(kappa),
            "kappa_cmd": float(kappa_cmd),
            "V_wheel": float(V_wheel),
            "denom": float(denom),
            "R_eff": float(R_eff),
            "T_Drv_signed": float(T_Drv_signed),
            "T_tire_reaction": float(T_tire_reaction),
            "M_brk_signed": float(M_brk_signed),
            "M_brk_abs": float(M_brk_abs),
            "M_visc": float(M_visc),
            "T_net": float(T_net),
        }

    def _solve_implicit_wheel_tire_step(
        self,
        dt: float,
        omega_old: float,
        T_Drv: float,
        F_clamp: float,
        V_wx_local: float,
        F_z: float,
        F_y_tire_raw: float,
        direction: int,
    ) -> Tuple[float, float, float, float, float, float, float, float, bool]:
        """Solve omega_new and Fx_new consistently for one substep.

        Solves:
            omega_new = omega_old + dt/J * T_net(omega_new, Fx(omega_new))

        with speed limits applied to the RHS exactly as in drive update.
        """

        omega_old = float(omega_old)

        # Initial residual around omega_old.
        r0, info0 = self._implicit_residual(
            omega_candidate=omega_old,
            dt=dt,
            omega_old=omega_old,
            T_Drv=T_Drv,
            F_clamp=F_clamp,
            V_wx_local=V_wx_local,
            F_z=F_z,
            F_y_tire_raw=F_y_tire_raw,
            direction=direction,
        )

        # Explicit RHS is a useful center for the search bracket.
        omega_explicit = float(info0["omega_rhs"])

        # Build an initial bracket. Span is based on the explicit change and a
        # small absolute margin so it works even at small torque.
        delta0 = abs(omega_explicit - omega_old)
        span = max(2.0 * delta0 + 0.5, 1.0)

        lo = min(omega_old, omega_explicit) - span
        hi = max(omega_old, omega_explicit) + span

        r_lo, info_lo = self._implicit_residual(
            omega_candidate=lo,
            dt=dt,
            omega_old=omega_old,
            T_Drv=T_Drv,
            F_clamp=F_clamp,
            V_wx_local=V_wx_local,
            F_z=F_z,
            F_y_tire_raw=F_y_tire_raw,
            direction=direction,
        )
        r_hi, info_hi = self._implicit_residual(
            omega_candidate=hi,
            dt=dt,
            omega_old=omega_old,
            T_Drv=T_Drv,
            F_clamp=F_clamp,
            V_wx_local=V_wx_local,
            F_z=F_z,
            F_y_tire_raw=F_y_tire_raw,
            direction=direction,
        )

        # Expand bracket if needed.
        bracket_found = np.isfinite(r_lo) and np.isfinite(r_hi) and (r_lo == 0.0 or r_hi == 0.0 or r_lo * r_hi <= 0.0)
        for _ in range(self.IMPLICIT_BRACKET_EXPANSIONS):
            if bracket_found:
                break
            span *= 2.0
            lo = min(omega_old, omega_explicit) - span
            hi = max(omega_old, omega_explicit) + span
            r_lo, info_lo = self._implicit_residual(
                omega_candidate=lo,
                dt=dt,
                omega_old=omega_old,
                T_Drv=T_Drv,
                F_clamp=F_clamp,
                V_wx_local=V_wx_local,
                F_z=F_z,
                F_y_tire_raw=F_y_tire_raw,
                direction=direction,
            )
            r_hi, info_hi = self._implicit_residual(
                omega_candidate=hi,
                dt=dt,
                omega_old=omega_old,
                T_Drv=T_Drv,
                F_clamp=F_clamp,
                V_wx_local=V_wx_local,
                F_z=F_z,
                F_y_tire_raw=F_y_tire_raw,
                direction=direction,
            )
            bracket_found = np.isfinite(r_lo) and np.isfinite(r_hi) and (r_lo == 0.0 or r_hi == 0.0 or r_lo * r_hi <= 0.0)

        if bracket_found:
            # Bisection solve.
            best_omega = omega_explicit
            best_info = info0
            best_abs_r = abs(r0) if np.isfinite(r0) else float("inf")

            for _ in range(self.IMPLICIT_MAX_ITER):
                mid = 0.5 * (lo + hi)
                r_mid, info_mid = self._implicit_residual(
                    omega_candidate=mid,
                    dt=dt,
                    omega_old=omega_old,
                    T_Drv=T_Drv,
                    F_clamp=F_clamp,
                    V_wx_local=V_wx_local,
                    F_z=F_z,
                    F_y_tire_raw=F_y_tire_raw,
                    direction=direction,
                )

                if np.isfinite(r_mid) and abs(r_mid) < best_abs_r:
                    best_omega = mid
                    best_info = info_mid
                    best_abs_r = abs(r_mid)

                if (not np.isfinite(r_mid)) or abs(r_mid) < self.IMPLICIT_TOL_OMEGA or abs(hi - lo) < self.IMPLICIT_TOL_OMEGA:
                    break

                if r_lo * r_mid <= 0.0:
                    hi = mid
                    r_hi = r_mid
                else:
                    lo = mid
                    r_lo = r_mid

            omega_new = self._apply_wheel_speed_limit(best_omega)
            final_info = best_info
            solver_mode = 1.0  # bisection

        else:
            # Fallback: damped fixed-point iteration.
            omega_guess = omega_explicit
            final_info = info0
            for _ in range(self.IMPLICIT_MAX_ITER):
                _, trial_info = self._implicit_residual(
                    omega_candidate=omega_guess,
                    dt=dt,
                    omega_old=omega_old,
                    T_Drv=T_Drv,
                    F_clamp=F_clamp,
                    V_wx_local=V_wx_local,
                    F_z=F_z,
                    F_y_tire_raw=F_y_tire_raw,
                    direction=direction,
                )
                omega_rhs = float(trial_info["omega_rhs"])
                omega_next = 0.5 * omega_guess + 0.5 * omega_rhs
                final_info = trial_info
                if abs(omega_next - omega_guess) < self.IMPLICIT_TOL_OMEGA:
                    omega_guess = omega_next
                    break
                omega_guess = omega_next

            omega_new = self._apply_wheel_speed_limit(omega_guess)
            solver_mode = 2.0  # fixed-point fallback

        # Re-evaluate at final omega and commit states once.
        (
            F_x_tire_raw,
            kappa,
            kappa_cmd,
            V_wheel,
            denom,
            R_eff,
        ) = self._preview_longitudinal_tire_at_omega(
            dt=dt,
            omega_wheel=omega_new,
            V_wx_local=V_wx_local,
            F_z=F_z,
        )
        (
            F_x_tire,
            F_y_tire,
            friction_circle_limit,
            friction_circle_usage,
            friction_circle_usage_raw,
            friction_circle_scale,
            friction_circle_saturated,
        ) = self._apply_combined_friction_limit(
            F_x_tire_raw=F_x_tire_raw,
            F_y_tire_raw=F_y_tire_raw,
            F_z=F_z,
        )

        (
            omega_rhs_final,
            T_Drv_signed,
            T_tire_reaction,
            M_brk_signed,
            M_brk_abs,
            M_visc,
            T_net,
        ) = self._preview_wheel_rhs(
            dt=dt,
            omega_old=omega_old,
            omega_candidate=omega_new,
            T_Drv=T_Drv,
            F_x_tire=F_x_tire,
            F_clamp=F_clamp,
            direction=direction,
        )

        # The solved omega is the state update.
        self.drive.state.wheel_speed = float(omega_new)

        # Drive internal debug fields.
        try:
            self.drive.state.T_Drv_signed = float(T_Drv_signed)
            self.drive.state.T_tire_reaction = float(T_tire_reaction)
            self.drive.state.M_brk_signed = float(M_brk_signed)
            self.drive.state.M_brk_abs = float(M_brk_abs)
            self.drive.state.M_visc = float(M_visc)
            self.drive.state.T_net = float(T_net)
            self.drive.state.wheel_alpha = float(T_net / max(float(getattr(self.drive.params, "J_wheel", 1.0)), 1e-12))
            self.drive.state.omega_before = float(omega_old)
            self.drive.state.omega_after = float(omega_new)
            self.drive.state.omega_rhs_final = float(omega_rhs_final)
            self.drive.state.implicit_solver_mode = float(solver_mode)
            self.drive.state.implicit_residual = float(omega_new - omega_rhs_final)
            self.drive.state.speed_limit_hit = float(abs(omega_new - omega_rhs_final) > 1e-6 and abs(omega_new) >= 0.999 * abs(float(getattr(self.drive.params, "max_wheel_speed", 1e99))))
        except Exception:
            pass

        self._commit_longitudinal_tire_state(
            F_x_tire=F_x_tire,
            F_x_tire_raw=F_x_tire_raw,
            combined_force_scale=friction_circle_scale,
            kappa=kappa,
            kappa_cmd=kappa_cmd,
            V_wheel=V_wheel,
            V_wx_local=V_wx_local,
            denom=denom,
        )

        return (
            float(omega_new),
            float(F_x_tire),
            float(F_x_tire_raw),
            float(F_y_tire),
            float(friction_circle_limit),
            float(friction_circle_usage),
            float(friction_circle_usage_raw),
            float(friction_circle_scale),
            bool(friction_circle_saturated),
        )

    # -------------------------------------------------------------------------
    # Main wheel/axle update
    # -------------------------------------------------------------------------

    def update(
        self,
        dt: float,
        T_steer: float,
        T_brk: float,
        T_Drv: float,
        V_wheel_x: float,
        V_wheel_y: float,
        direction: int = 1,
        steering_angle_override: Optional[float] = None,
        steering_rate_override: Optional[float] = None,
        road_mu: Optional[float] = None,
    ) -> Tuple[float, float]:
        self._road_mu_override = None if road_mu is None else float(road_mu)
        if (
            self.has_steering
            and self.steering is not None
            and steering_angle_override is not None
            and steering_rate_override is not None
        ):
            steering_angle = float(steering_angle_override)
            steering_rate = float(steering_rate_override)
            self.steering.state.steering_angle = steering_angle
            self.steering.state.steering_rate = steering_rate
            self.steering.state.steering_torque = 0.0
        elif self.has_steering and self.steering is not None:
            steering_angle, steering_rate = self.steering.update(
                dt,
                T_steer,
                self.steering.state.self_aligning_torque,
            )
        else:
            steering_angle = 0.0
            steering_rate = 0.0

        # Axle capability is owned here at the axle-composition layer.
        # FS/FWD: front axle steers and is driven, rear axle is passive.
        T_Drv = float(T_Drv) if self.has_drive_motor else 0.0

        F_z = self.F_z_static
        c, s = np.cos(steering_angle), np.sin(steering_angle)
        V_wx_local = c * V_wheel_x + s * V_wheel_y
        V_wy_local = -s * V_wheel_x + c * V_wheel_y

        F_y_alpha, F_y_tire_raw = self._preview_lateral_tire_force(
            V_wx_local=V_wx_local,
            V_wy_local=V_wy_local,
            F_z=F_z,
        )

        F_clamp = self.brake.update(dt, T_brk)
        (
            omega_wheel,
            F_x_tire,
            F_x_tire_raw,
            F_y_tire,
            friction_circle_limit,
            friction_circle_usage,
            friction_circle_usage_raw,
            friction_circle_scale,
            friction_circle_saturated,
        ) = self._solve_implicit_wheel_tire_step(
            dt=float(dt),
            omega_old=float(self.drive.state.wheel_speed),
            T_Drv=T_Drv,
            F_clamp=F_clamp,
            V_wx_local=V_wx_local,
            F_z=F_z,
            F_y_tire_raw=F_y_tire_raw,
            direction=direction,
        )
        self._commit_lateral_tire_state(
            alpha=F_y_alpha,
            F_y_tire=F_y_tire,
            F_y_tire_raw=F_y_tire_raw,
            combined_force_scale=friction_circle_scale,
        )

        if self.has_steering and self.steering is not None:
            self.steering.state.self_aligning_torque = self.lateral_tire.state.aligning_torque

        self.state.F_x_tire = float(F_x_tire)
        self.state.F_y_tire = float(F_y_tire)
        self.state.F_x_tire_raw = float(F_x_tire_raw)
        self.state.F_y_tire_raw = float(F_y_tire_raw)
        self.state.F_z = float(F_z)
        self.state.friction_circle_limit = float(friction_circle_limit)
        self.state.friction_circle_usage = float(friction_circle_usage)
        self.state.friction_circle_usage_raw = float(friction_circle_usage_raw)
        self.state.friction_circle_scale = float(friction_circle_scale)
        self.state.friction_circle_saturated = bool(friction_circle_saturated)
        self.state.steering_angle = float(steering_angle)
        self.state.steering_rate = float(steering_rate)
        self.state.omega_wheel = float(omega_wheel)

        return self.state.F_x_tire, self.state.F_y_tire

    def get_state(self) -> Dict:
        friction_circle_limit = float(self.state.friction_circle_limit)
        if not np.isfinite(friction_circle_limit) or friction_circle_limit <= 0.0:
            friction_circle_limit = self._combined_friction_limit(self.state.F_z)

        force_norm = float(np.hypot(self.state.F_x_tire, self.state.F_y_tire))
        raw_force_norm = float(np.hypot(self.state.F_x_tire_raw, self.state.F_y_tire_raw))
        if friction_circle_limit > 1e-9:
            friction_circle_usage = force_norm / friction_circle_limit
            friction_circle_usage_raw = raw_force_norm / friction_circle_limit
        else:
            friction_circle_usage = 0.0
            friction_circle_usage_raw = 0.0

        friction_circle_scale = float(self.state.friction_circle_scale)
        if not np.isfinite(friction_circle_scale):
            friction_circle_scale = min(1.0, friction_circle_limit / max(raw_force_norm, 1e-9))
        friction_circle_saturated = bool(
            self.state.friction_circle_saturated or friction_circle_usage_raw > 1.0
        )

        drive_state = getattr(self.drive, "state", None)

        def d(name: str, default: float = float("nan")) -> float:
            try:
                return float(getattr(drive_state, name, default))
            except (TypeError, ValueError):
                return float(default)

        return {
            "F_x_tire": self.state.F_x_tire,
            "F_y_tire": self.state.F_y_tire,
            "F_x_tire_raw": self.state.F_x_tire_raw,
            "F_y_tire_raw": self.state.F_y_tire_raw,
            "F_z": self.state.F_z,
            "steering_angle": self.state.steering_angle,
            "steering_rate": self.state.steering_rate,
            "omega_wheel": self.state.omega_wheel,
            "friction_circle_limit": float(friction_circle_limit),
            "friction_circle_usage": float(friction_circle_usage),
            "friction_circle_usage_raw": float(friction_circle_usage_raw),
            "friction_circle_scale": float(friction_circle_scale),
            "friction_circle_saturated": bool(friction_circle_saturated),
            "has_drive_motor": bool(self.has_drive_motor),
            "road_mu": float(self._road_mu_override)
            if self._road_mu_override is not None
            else float("nan"),

            # Exact drive internal torque debug fields.
            "T_Drv_signed": d("T_Drv_signed"),
            "T_tire_reaction": d("T_tire_reaction"),
            "M_brk_signed": d("M_brk_signed"),
            "M_brk_abs": d("M_brk_abs"),
            "M_visc": d("M_visc"),
            "T_net": d("T_net"),
            "wheel_alpha": d("wheel_alpha"),
            "omega_before": d("omega_before"),
            "omega_after": d("omega_after"),
            "omega_rhs_final": d("omega_rhs_final"),
            "implicit_solver_mode": d("implicit_solver_mode"),
            "implicit_residual": d("implicit_residual"),
            "speed_limit_hit": d("speed_limit_hit"),
        }

    def reset(self) -> None:
        self.state = StbwState()
        if self.steering is not None:
            self.steering.reset()
        self.brake.reset()
        self.drive.reset()
        self.longitudinal_tire.reset()
        self.lateral_tire.reset()

    @staticmethod
    def _build_steering_config(
        axle_id: str,
        user_config: Optional[Dict],
        config_path: Optional[str],
    ) -> Dict:
        if user_config and "steering" in user_config:
            steering_param = user_config["steering"] or {}
        else:
            steering_param = load_param("steering", config_path)

        if axle_id not in ["F", "R", "FL", "FR", "RL", "RR"]:
            raise ValueError(f"Invalid axle_id: {axle_id}.")

        axle_key = "front" if str(axle_id).startswith("F") else "rear"
        axle_cfg = steering_param.get(axle_key, {}) if isinstance(steering_param, dict) else {}

        steering_cfg = dict(steering_param) if isinstance(steering_param, dict) else {}
        steering_cfg["max_angle_pos"] = axle_cfg.get("max_angle_pos", steering_cfg.get("max_angle_pos", 0.0))
        steering_cfg["max_angle_neg"] = axle_cfg.get("max_angle_neg", steering_cfg.get("max_angle_neg", 0.0))
        return steering_cfg
