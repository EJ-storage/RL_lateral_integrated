"""Integrated steer-by-wire wheel model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from vehicle_sim.models.drive_layout import drive_axles_label, is_axle_driven
from vehicle_sim.utils.config_loader import load_param

from .drive.brake_model import StbwBrakeModel
from .drive.drive_model import StbwDriveModel
from .steering.steering_model import StbwSteeringModel, resolve_stbw_config_path
from .tire.dugoff import ModifiedDugoff4WTireModel
from .tire.lateral.lateral_tire import StbwLateralTireModel
from .tire.longitudinal.longitudinal_tire import StbwLongitudinalTireModel


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
        self.config_path = resolve_stbw_config_path(config_path)
        self.state = StbwState()
        self.has_steering = self.axle_group == "F"
        self.drive_axles = drive_axles_label(self.config.get("drive_axles", "R"))
        self.has_drive_motor = is_axle_driven(self.axle_group, self.drive_axles)
        self._road_mu_override: Optional[float] = None

        vehicle_param = load_param("vehicle_body", self.config_path)
        physics_param = load_param("physics", self.config_path)
        geometry_param = load_param("vehicle_spec", self.config_path)
        geometry_cfg = geometry_param.get("geometry", {})
        self.use_modified_dugoff = True

        self.m = float(vehicle_param.get("m", 0.0))
        self.g = float(physics_param.get("g", 0.0))
        self.h_CG = float(vehicle_param.get("h_CG", 0.0))
        self.lf = float(geometry_cfg.get("lf", 0.0))
        self.lr = float(geometry_cfg.get("lr", 0.0))
        self.wheelbase = self.lf + self.lr
        self.tf = self._track_width_from_geometry(geometry_cfg, "F")
        self.tr = self._track_width_from_geometry(geometry_cfg, "R")

        if self.has_steering:
            self.F_z_static = self.m * self.g * self.lr / self.wheelbase
            steering_cfg = self._build_steering_config(
                self.axle_group,
                self.config,
                self.config_path,
            )
            self.steering = StbwSteeringModel(
                config=steering_cfg,
                config_path=self.config_path,
                axle_id=self.axle_group,
            )
        else:
            self.F_z_static = self.m * self.g * self.lf / self.wheelbase
            self.steering = None

        if self.is_corner:
            self.F_z_static *= 0.5

        self.brake = StbwBrakeModel(config_path=self.config_path)
        self.drive = StbwDriveModel(config_path=self.config_path, axle_id=self.axle_group)
        self.longitudinal_tire = StbwLongitudinalTireModel(config_path=self.config_path)
        self.lateral_tire = StbwLateralTireModel(config_path=self.config_path)
        self.dugoff_tire = ModifiedDugoff4WTireModel(config_path=self.config_path)

    def set_drive_axles(self, drive_axles: str) -> None:
        self.drive_axles = drive_axles_label(drive_axles)
        self.has_drive_motor = is_axle_driven(self.axle_group, self.drive_axles)

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
        ax: float = 0.0,
        ay: float = 0.0,
    ) -> Tuple[float, float]:
        self._road_mu_override = None if road_mu is None else float(road_mu)
        steering_angle, steering_rate = self._update_steering(
            dt=dt,
            T_steer=T_steer,
            steering_angle_override=steering_angle_override,
            steering_rate_override=steering_rate_override,
        )

        T_Drv = float(T_Drv) if self.has_drive_motor else 0.0
        F_z = self._calculate_vertical_load(ax=ax, ay=ay)

        c, s = np.cos(steering_angle), np.sin(steering_angle)
        V_wx_local = c * float(V_wheel_x) + s * float(V_wheel_y)
        V_wy_local = -s * float(V_wheel_x) + c * float(V_wheel_y)

        F_clamp = self.brake.update(dt, T_brk)

        # Same update order as e_corner.py:
        # drive uses the previous tire force, then tire force is recomputed
        # from the newly updated wheel speed.
        omega_wheel = self.drive.update(
            dt,
            T_Drv,
            self.state.F_x_tire,
            F_clamp=F_clamp,
            direction=direction,
        )

        F_x_tire, F_y_tire, dugoff_state = self._calculate_dugoff_force(
            dt=dt,
            omega_wheel=omega_wheel,
            V_wx_local=V_wx_local,
            V_wy_local=V_wy_local,
            F_z=F_z,
        )
        alpha = float(dugoff_state.get("alpha", 0.0))
        M_align = self.lateral_tire.calculate_aligning_torque(F_y_tire)

        self._commit_lateral_state(alpha, F_y_tire, M_align)
        if self.has_steering and self.steering is not None:
            self.steering.state.self_aligning_torque = float(M_align)

        self.state.F_x_tire = float(F_x_tire)
        self.state.F_y_tire = float(F_y_tire)
        self.state.F_x_tire_raw = float(dugoff_state.get("fx_linear", F_x_tire))
        self.state.F_y_tire_raw = float(dugoff_state.get("fy_linear", F_y_tire))
        self.state.F_z = float(F_z)
        self.state.steering_angle = float(steering_angle)
        self.state.steering_rate = float(steering_rate)
        self.state.omega_wheel = float(omega_wheel)
        self._update_friction_diagnostics()

        return self.state.F_x_tire, self.state.F_y_tire

    def get_state(self) -> Dict:
        result = {
            "F_x_tire": self.state.F_x_tire,
            "F_y_tire": self.state.F_y_tire,
            "F_x_tire_raw": self.state.F_x_tire_raw,
            "F_y_tire_raw": self.state.F_y_tire_raw,
            "F_z": self.state.F_z,
            "steering_angle": self.state.steering_angle,
            "steering_rate": self.state.steering_rate,
            "omega_wheel": self.state.omega_wheel,
            "friction_circle_limit": self.state.friction_circle_limit,
            "friction_circle_usage": self.state.friction_circle_usage,
            "friction_circle_usage_raw": self.state.friction_circle_usage_raw,
            "friction_circle_scale": self.state.friction_circle_scale,
            "friction_circle_saturated": self.state.friction_circle_saturated,
            "has_drive_motor": bool(self.has_drive_motor),
            "road_mu": float(self._road_mu_override)
            if self._road_mu_override is not None
            else float("nan"),
        }
        if self.dugoff_tire is not None:
            dugoff_state = self.dugoff_tire.get_wheel_state(self._dugoff_wheel_label())
            result.update(
                {
                    "dugoff_kappa": float(dugoff_state.get("kappa", 0.0)),
                    "dugoff_alpha": float(dugoff_state.get("alpha", 0.0)),
                    "dugoff_delta_v": float(dugoff_state.get("delta_v", 0.0)),
                    "dugoff_wheel_linear_speed": float(
                        dugoff_state.get("wheel_linear_speed", 0.0)
                    ),
                    "dugoff_vx": float(dugoff_state.get("vx", 0.0)),
                    "dugoff_fx_linear": float(dugoff_state.get("fx_linear", 0.0)),
                    "dugoff_fy_linear": float(dugoff_state.get("fy_linear", 0.0)),
                    "dugoff_scale": float(dugoff_state.get("dugoff_scale", 1.0)),
                    "dugoff_ellipse_scale": float(
                        dugoff_state.get("ellipse_scale", 1.0)
                    ),
                    "dugoff_combined_demand": float(
                        dugoff_state.get("combined_demand", 0.0)
                    ),
                }
            )
        return result

    def reset(self) -> None:
        self.state = StbwState()
        if self.steering is not None:
            self.steering.reset()
        self.brake.reset()
        self.drive.reset()
        self.longitudinal_tire.reset()
        self.lateral_tire.reset()
        if self.dugoff_tire is not None:
            self.dugoff_tire.reset()
        self._road_mu_override = None

    def _update_steering(
        self,
        dt: float,
        T_steer: float,
        steering_angle_override: Optional[float],
        steering_rate_override: Optional[float],
    ) -> Tuple[float, float]:
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
            return steering_angle, steering_rate

        if self.has_steering and self.steering is not None:
            return self.steering.update(
                dt,
                T_steer,
                self.steering.state.self_aligning_torque,
            )

        return 0.0, 0.0

    def _calculate_vertical_load(self, ax: float, ay: float) -> float:
        lf = max(float(self.lf), 1.0e-3)
        lr = max(float(self.lr), 1.0e-3)
        tf = max(float(self.tf), 1.0e-3)
        tr = max(float(self.tr), 1.0e-3)
        m = max(float(self.m), 1.0)
        g = max(float(self.g), 0.1)
        h = max(float(self.h_CG), 0.0)
        ax = float(ax)
        ay = float(ay)

        L = lf + lr
        Fzf_static = m * g * (lr / L)
        Fzr_static = m * g * (lf / L)

        dFz_long = m * ax * h / L
        Fzf = Fzf_static - dFz_long
        Fzr = Fzr_static + dFz_long

        mf = m * (lr / L)
        mr = m * (lf / L)

        dFz_lat_f = mf * ay * h / tf
        dFz_lat_r = mr * ay * h / tr

        if self.axle_id == "FL":
            F_z = 0.5 * Fzf - dFz_lat_f
        elif self.axle_id == "FR":
            F_z = 0.5 * Fzf + dFz_lat_f
        elif self.axle_id == "RL":
            F_z = 0.5 * Fzr - dFz_lat_r
        elif self.axle_id == "RR":
            F_z = 0.5 * Fzr + dFz_lat_r
        elif self.axle_group == "F":
            F_z = Fzf
        else:
            F_z = Fzr

        return float(max(F_z, 0.0))

    def _commit_lateral_state(self, alpha: float, F_y_tire: float, M_align: float) -> None:
        tire_state = self.lateral_tire.state
        tire_state.slip_angle = float(alpha)
        if hasattr(tire_state, "slip_angle_dyn"):
            tire_state.slip_angle_dyn = float(alpha)
        tire_state.lateral_force = float(F_y_tire)
        tire_state.aligning_torque = float(M_align)

    def _dugoff_wheel_label(self) -> str:
        if self.axle_id in {"FL", "FR", "RL", "RR"}:
            return str(self.axle_id)
        return "FL" if self.axle_group == "F" else "RL"

    def _calculate_dugoff_force(
        self,
        dt: float,
        omega_wheel: float,
        V_wx_local: float,
        V_wy_local: float,
        F_z: float,
    ) -> Tuple[float, float, Dict[str, float]]:
        if self.dugoff_tire is None:
            raise RuntimeError("Modified Dugoff tire model is not configured.")

        wheel_label = self._dugoff_wheel_label()
        F_x_tire, F_y_tire = self.dugoff_tire.update_wheel(
            dt=dt,
            wheel_label=wheel_label,
            wheelspeed=omega_wheel,
            vx=V_wx_local,
            vy=V_wy_local,
            fz=F_z,
            road_mu=self._road_mu_override,
        )
        dugoff_state = self.dugoff_tire.get_wheel_state(wheel_label)
        self._commit_dugoff_compatibility_state(dugoff_state)
        return float(F_x_tire), float(F_y_tire), dugoff_state

    def _commit_dugoff_compatibility_state(self, dugoff_state: Dict[str, float]) -> None:
        long_state = self.longitudinal_tire.state
        vx = float(dugoff_state.get("vx", 0.0))
        wheel_linear_speed = float(dugoff_state.get("wheel_linear_speed", 0.0))
        kappa = float(dugoff_state.get("kappa", 0.0))
        alpha = float(dugoff_state.get("alpha", 0.0))

        long_state.V_wheel_x = vx
        long_state.V_wheel = wheel_linear_speed
        long_state.V_wheel_x_minus_V_wheel = vx - wheel_linear_speed
        long_state.slip_ratio_denom = max(
            abs(vx),
            float(getattr(self.dugoff_tire.params, "Veps", 1.0e-6))
            if self.dugoff_tire is not None
            else 1.0e-6,
        )
        if hasattr(long_state, "slip_ratio_cmd"):
            long_state.slip_ratio_cmd = kappa
        long_state.slip_ratio = kappa
        long_state.longitudinal_force = float(dugoff_state.get("fx", 0.0))

        lateral_state = self.lateral_tire.state
        lateral_state.slip_angle = alpha
        if hasattr(lateral_state, "slip_angle_dyn"):
            lateral_state.slip_angle_dyn = alpha
        lateral_state.lateral_force = float(dugoff_state.get("fy", 0.0))

    def _friction_circle_limit(self) -> float:
        if self.dugoff_tire is not None:
            dugoff_state = self.dugoff_tire.get_wheel_state(self._dugoff_wheel_label())
            fx_limit = float(dugoff_state.get("fx_limit", 0.0))
            fy_limit = float(dugoff_state.get("fy_limit", 0.0))
            limit = min(fx_limit, fy_limit)
            return float(limit) if np.isfinite(limit) and limit > 0.0 else 0.0
        return 0.0

    def _update_friction_diagnostics(self) -> None:
        if self.dugoff_tire is not None:
            dugoff_state = self.dugoff_tire.get_wheel_state(self._dugoff_wheel_label())
            fx_limit = float(dugoff_state.get("fx_limit", 0.0))
            fy_limit = float(dugoff_state.get("fy_limit", 0.0))
            if fx_limit > 1e-9 and fy_limit > 1e-9:
                usage = float(
                    np.sqrt(
                        (self.state.F_x_tire / fx_limit) ** 2
                        + (self.state.F_y_tire / fy_limit) ** 2
                    )
                )
                usage_raw = float(dugoff_state.get("combined_demand", 0.0))
                limit = min(fx_limit, fy_limit)
            else:
                usage = 0.0
                usage_raw = 0.0
                limit = 0.0

            ellipse_scale = float(dugoff_state.get("ellipse_scale", 1.0))
            self.state.friction_circle_limit = float(limit)
            self.state.friction_circle_usage = float(usage)
            self.state.friction_circle_usage_raw = float(usage_raw)
            self.state.friction_circle_scale = float(ellipse_scale)
            self.state.friction_circle_saturated = bool(
                usage_raw > 1.0 or ellipse_scale < 1.0
            )
            return

    @staticmethod
    def _build_steering_config(
        axle_id: str,
        user_config: Optional[Dict],
        config_path: Optional[str],
    ) -> Dict:
        if user_config and "steering" in user_config:
            steering_param = user_config["steering"] or {}
        else:
            steering_param = load_param("steering", resolve_stbw_config_path(config_path))

        if axle_id not in ["F", "R", "FL", "FR", "RL", "RR"]:
            raise ValueError(f"Invalid axle_id: {axle_id}.")

        axle_key = "front" if str(axle_id).startswith("F") else "rear"
        axle_cfg = steering_param.get(axle_key, {}) if isinstance(steering_param, dict) else {}

        steering_cfg = dict(steering_param) if isinstance(steering_param, dict) else {}
        steering_cfg["max_angle_pos"] = axle_cfg.get(
            "max_angle_pos",
            steering_cfg.get("max_angle_pos", 0.0),
        )
        steering_cfg["max_angle_neg"] = axle_cfg.get(
            "max_angle_neg",
            steering_cfg.get("max_angle_neg", 0.0),
        )
        return steering_cfg

    @staticmethod
    def _track_width_from_geometry(geometry_cfg: Dict, axle_group: str) -> float:
        if axle_group == "F":
            explicit_keys = ("tf", "front_track", "track_front")
            left_label, right_label = "FL", "FR"
        else:
            explicit_keys = ("tr", "rear_track", "track_rear")
            left_label, right_label = "RL", "RR"

        for key in explicit_keys:
            if key in geometry_cfg:
                return float(geometry_cfg[key])

        corner_offsets = geometry_cfg.get("corner_offsets", {})
        left_offset = corner_offsets.get(left_label, {})
        right_offset = corner_offsets.get(right_label, {})
        if "y" in left_offset and "y" in right_offset:
            return abs(float(left_offset["y"]) - float(right_offset["y"]))

        return float(geometry_cfg.get("L_track", geometry_cfg.get("track_width", 0.0)))
