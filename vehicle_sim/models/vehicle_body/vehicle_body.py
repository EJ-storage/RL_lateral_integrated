"""
차체 동역학 모델
차체 3자유도 강체 동역학을 처리
"""

import numpy as np
from typing import Dict, List, Mapping, Optional, Tuple
from dataclasses import dataclass

from vehicle_sim.stbw_model.stbw import Stbw
from vehicle_sim.models.drive_layout import drive_axles_label

@dataclass
class StbwVehicleBodyState:
    x: float = 0.0
    y: float = 0.0
    heave: float = 0.0

    yaw: float = 0.0

    velocity_x: float = 0.0
    velocity_y: float = 0.0

    yaw_rate: float = 0.0

    ax: float = 0.0
    ay: float = 0.0
    throttle_off_time_s: float = 0.0

@dataclass
class StbwVehicleBodyParameters:
    m_total: float = 0.0
    
    Izz: float = 0.0

    a: float = 0.0
    b: float = 0.0
    h_CG: float = 0.0

    g: float = 0.0



def _zero_offsets(labels: Tuple[str, ...]) -> Dict[str, Dict[str, float]]:
    return {label: {"x": 0.0, "y": 0.0} for label in labels}


def _coerce_offsets(
    offsets: Optional[Mapping[str, Mapping[str, float]]],
    labels: Tuple[str, ...],
) -> Dict[str, Dict[str, float]]:
    result = _zero_offsets(labels)
    if offsets is None:
        return result

    for label in labels:
        offset = offsets.get(label, {})
        result[label] = {
            "x": float(offset.get("x", 0.0)),
            "y": float(offset.get("y", 0.0)),
        }
    return result


class _StbwAxleAggregate:
    """Compatibility view that exposes front/rear axle totals over two wheels."""

    def __init__(self, axle_id: str, left: Stbw, right: Stbw):
        self.axle_id = axle_id
        self.left = left
        self.right = right
        self.steering = left.steering
        self.brake = left.brake
        self.drive = left.drive
        self.longitudinal_tire = left.longitudinal_tire
        self.lateral_tire = left.lateral_tire
        self.state = left.state

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()
        self.state = self.left.state
        self.steering = self.left.steering
        self.brake = self.left.brake
        self.drive = self.left.drive
        self.longitudinal_tire = self.left.longitudinal_tire
        self.lateral_tire = self.left.lateral_tire

    @staticmethod
    def _sum_state(states: List[Dict], key: str, default: float = 0.0) -> float:
        return float(sum(float(state.get(key, default)) for state in states))

    @staticmethod
    def _max_state(states: List[Dict], key: str, default: float = 0.0) -> float:
        values = [float(state.get(key, default)) for state in states]
        return float(max(values)) if values else float(default)

    @staticmethod
    def _min_state(states: List[Dict], key: str, default: float = 1.0) -> float:
        values = [float(state.get(key, default)) for state in states]
        return float(min(values)) if values else float(default)

    @staticmethod
    def _avg_attr(objects: List[object], name: str, default: float = 0.0) -> float:
        values = [float(getattr(obj, name, default)) for obj in objects]
        return float(sum(values) / max(len(values), 1))

    def get_state(self) -> Dict:
        states = [self.left.get_state(), self.right.get_state()]
        steering_angle = self._avg_attr([self.left.state, self.right.state], "steering_angle")
        steering_rate = self._avg_attr([self.left.state, self.right.state], "steering_rate")
        omega_wheel = self._avg_attr([self.left.state, self.right.state], "omega_wheel")
        limit = self._sum_state(states, "friction_circle_limit")
        force_norm = float(np.hypot(
            self._sum_state(states, "F_x_tire"),
            self._sum_state(states, "F_y_tire"),
        ))
        raw_force_norm = float(np.hypot(
            self._sum_state(states, "F_x_tire_raw"),
            self._sum_state(states, "F_y_tire_raw"),
        ))
        usage = force_norm / limit if limit > 1e-9 else 0.0
        usage_raw = raw_force_norm / limit if limit > 1e-9 else 0.0
        return {
            "F_x_tire": self._sum_state(states, "F_x_tire"),
            "F_y_tire": self._sum_state(states, "F_y_tire"),
            "F_x_tire_raw": self._sum_state(states, "F_x_tire_raw"),
            "F_y_tire_raw": self._sum_state(states, "F_y_tire_raw"),
            "F_z": self._sum_state(states, "F_z"),
            "steering_angle": steering_angle,
            "steering_rate": steering_rate,
            "omega_wheel": omega_wheel,
            "friction_circle_limit": limit,
            "friction_circle_usage": usage,
            "friction_circle_usage_raw": usage_raw,
            "friction_circle_scale": self._min_state(states, "friction_circle_scale"),
            "friction_circle_saturated": any(bool(state.get("friction_circle_saturated", False)) for state in states),
            "left_state": states[0],
            "right_state": states[1],
        }


class StbwVehicleBody:
    def __init__(
        self,
        parameters: StbwVehicleBodyParameters = None,
        config_path: Optional[str] = None,
        drive_axles: str = "R",
        axle_offsets: Optional[Mapping[str, Mapping[str, float]]] = None,
        corner_offsets: Optional[Mapping[str, Mapping[str, float]]] = None,
    ):
        self.drive_axles = drive_axles_label(drive_axles)
        self.params = parameters if parameters is not None else StbwVehicleBodyParameters()
        self.axle_offsets = _coerce_offsets(axle_offsets, ("F", "R"))
        self.corner_offsets = _coerce_offsets(corner_offsets, ("FL", "FR", "RL", "RR"))

        self.state = StbwVehicleBodyState()
        self.axle_labels: List[str] = ["F", "R"]
        self.wheel_labels: List[str] = ["FL", "FR", "RL", "RR"]
        self.wheels: Dict[str, Stbw] = {
            label: Stbw(
                axle_id=label,
                config={"drive_axles": self.drive_axles},
                config_path=config_path,
            )
            for label in self.wheel_labels
        }
        self.axles: Dict[str, _StbwAxleAggregate] = {
            "F": _StbwAxleAggregate("F", self.wheels["FL"], self.wheels["FR"]),
            "R": _StbwAxleAggregate("R", self.wheels["RL"], self.wheels["RR"]),
        }

    def set_drive_axles(self, drive_axles: str) -> None:
        self.drive_axles = drive_axles_label(drive_axles)
        for wheel in self.wheels.values():
            wheel.set_drive_axles(self.drive_axles)

    def _rotation_matrix(self) -> np.ndarray:
        psi = self.state.yaw

        c_psi, s_psi = np.cos(psi), np.sin(psi)

        return np.array([
            [c_psi, -s_psi],
            [s_psi, c_psi]
        ])

    @staticmethod
    def _axle_label_for_wheel(wheel_label: str) -> str:
        return "F" if str(wheel_label).startswith("F") else "R"

    @staticmethod
    def _left_right_labels(axle_label: str) -> Tuple[str, str]:
        if axle_label == "F":
            return "FL", "FR"
        if axle_label == "R":
            return "RL", "RR"
        raise ValueError(f"Unsupported axle label: {axle_label}")

    def iter_wheel_modules(self):
        return self.wheels.items()

    def set_front_steering_state(self, steering_angle: float, steering_rate: float = 0.0) -> None:
        for label in ("FL", "FR"):
            wheel = self.wheels[label]
            if wheel.steering is None:
                continue
            angle = wheel.steering.apply_angle_limits(float(steering_angle))
            rate = wheel.steering.apply_rate_limits(float(steering_rate))
            wheel.steering.state.steering_angle = angle
            wheel.steering.state.steering_rate = rate
            wheel.state.steering_angle = angle
            wheel.state.steering_rate = rate

    def _split_axle_input_to_wheel(self, wheel_label: str, axle_inputs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        axle_label = self._axle_label_for_wheel(wheel_label)
        axle_input = dict(axle_inputs.get(axle_label, {}))
        wheel_input = dict(axle_inputs.get(wheel_label, {}))
        inputs = {**axle_input, **wheel_input}
        if wheel_label not in axle_inputs:
            inputs["T_brk"] = 0.5 * float(inputs.get("T_brk", 0.0))
            inputs["T_Drv"] = 0.5 * float(inputs.get("T_Drv", 0.0))
        return inputs

    def update(self, dt: float, axle_inputs: Dict[str, Dict[str, float]], direction: int=1) -> None:
        wheel_body_inputs = self.get_wheel_inputs()

        wheel_outputs = {}
        for label in self.wheel_labels:
            wheel = self.wheels[label]
            inputs = self._split_axle_input_to_wheel(label, axle_inputs)

            T_steer = inputs.get("T_steer", 0.0)
            T_brk = inputs.get("T_brk", 0.0)
            T_Drv = inputs.get("T_Drv", 0.0)
            steering_angle = inputs.get("steering_angle")
            steering_rate = inputs.get("steering_rate")
            road_mu = inputs.get("road_mu", inputs.get("tire_mu"))

            V_wheel_x = wheel_body_inputs["wheel_velocities"][label][0]
            V_wheel_y = wheel_body_inputs["wheel_velocities"][label][1]
            ax = wheel_body_inputs["load_transfer_ax"]
            ay = wheel_body_inputs["load_transfer_ay"]

            F_x, F_y = wheel.update(
                dt,
                T_steer,
                T_brk,
                T_Drv,
                V_wheel_x,
                V_wheel_y,
                direction,
                steering_angle_override=steering_angle,
                steering_rate_override=steering_rate,
                road_mu=road_mu,
                ax=ax,
                ay=ay,
            )
            wheel_outputs[label] = (F_x, F_y)

        forces, moments = self.assemble_forces_moments(wheel_outputs)
        forces = np.asarray(forces, dtype=float).copy()

        self._update_dynamics(dt, forces, moments)


    def calculate_accelerations(self, forces: np.ndarray, yaw_moment: float) -> Tuple[np.ndarray, float]:
        Fx = float(forces[0])
        Fy = float(forces[1])

        u = self.state.velocity_x
        v = self.state.velocity_y
        r = self.state.yaw_rate

        m = self.params.m_total
        Iz = self.params.Izz

        ax = Fx / m + v * r
        ay = Fy / m - u * r
        r_dot = yaw_moment / Iz

        linear_acc = np.array([ax, ay], dtype=float)
        yaw_acc = float(r_dot)

        self.state.ax = linear_acc[0]
        self.state.ay = linear_acc[1]

        return linear_acc, yaw_acc

    def get_axle_position(self, axle_idx: int) -> np.ndarray:
        label = self.axle_labels[axle_idx]
        x_i = self.axle_offsets[label]["x"]
        y_i = self.axle_offsets[label]["y"]

        r_body = np.array([x_i, y_i], dtype=float)

        R = self._rotation_matrix()
        r_inertial = R @ r_body
        
        pos_cg = np.array([self.state.x, self.state.y], dtype=float)

        return pos_cg + r_inertial

    def get_wheel_position(self, wheel_label: str) -> np.ndarray:
        x_i = self.corner_offsets[wheel_label]["x"]
        y_i = self.corner_offsets[wheel_label]["y"]

        r_body = np.array([x_i, y_i], dtype=float)
        R = self._rotation_matrix()
        r_inertial = R @ r_body
        pos_cg = np.array([self.state.x, self.state.y], dtype=float)
        return pos_cg + r_inertial
    
    def _body_velocity_at_offset(self, x_i: float, y_i: float) -> np.ndarray:

        u = self.state.velocity_x
        v = self.state.velocity_y
        r = self.state.yaw_rate

        return np.array([
            u-r*y_i,
            v+r*x_i
        ], dtype=float)

    def get_wheel_velocity(self, wheel_label: str, frame: str = "body") -> np.ndarray:
        x_i = self.corner_offsets[wheel_label]["x"]
        y_i = self.corner_offsets[wheel_label]["y"]

        v_wheel = self._body_velocity_at_offset(x_i, y_i)

        if frame == "wheel":
            delta = self.wheels[wheel_label].state.steering_angle
            c, s = np.cos(delta), np.sin(delta)

            v_wx = c*v_wheel[0]+s*v_wheel[1]
            v_wy = -s*v_wheel[0]+c*v_wheel[1]

            return np.array([v_wx, v_wy], dtype=float)
        
        return v_wheel

    def get_axle_velocity(self, axle_idx: int, frame: str = "body") -> np.ndarray:
        label = self.axle_labels[axle_idx]
        left_label, right_label = self._left_right_labels(label)
        left_velocity = self.get_wheel_velocity(left_label, frame=frame)
        right_velocity = self.get_wheel_velocity(right_label, frame=frame)
        return 0.5 * (left_velocity + right_velocity)
    
    def get_state_vector(self) -> np.ndarray:
        return np.array([
            self.state.x, self.state.y, self.state.yaw,
            self.state.velocity_x, self.state.velocity_y, self.state.yaw_rate,
            self.state.ax, self.state.ay,
        ], dtype=float)
    
    def set_state_vector(self, state_vector: np.ndarray) -> None:
        if len(state_vector) != 8:
            raise ValueError(f"State vector must have 8 elements, got {len(state_vector)}")
        
        self.state.x = float(state_vector[0])
        self.state.y = float(state_vector[1])
        self.state.yaw = float(state_vector[2])
        self.state.velocity_x = float(state_vector[3])
        self.state.velocity_y = float(state_vector[4])
        self.state.yaw_rate = float(state_vector[5])
        self.state.ax = float(state_vector[6])
        self.state.ay = float(state_vector[7])

    def reset(self) -> None:
        self.state = StbwVehicleBodyState()

        for wheel in self.wheels.values():
            wheel.reset()

        dummy_inputs = {
            label: {
                "T_steer": 0.0,
                "T_brk": 0.0,
                "T_Drv": 0.0,
                "steering_angle": 0.0,
                "steering_rate": 0.0,
            }
            for label in self.axle_labels
        }
        self.update(dt=0.001, axle_inputs=dummy_inputs)

    
    def assemble_forces_moments(self, axle_outputs: Dict[str, Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        F_total = np.zeros(2, dtype=float)
        Mz_total = 0.0

        for label in self.wheel_labels:
            if label not in axle_outputs:
                continue

            F_x, F_y = axle_outputs[label]

            delta = self.wheels[label].state.steering_angle
            c,s = np.cos(delta), np.sin(delta)

            F_x_body = c*F_x - s*F_y
            F_y_body = s*F_x + c*F_y

            x_i = self.corner_offsets[label]["x"]
            y_i = self.corner_offsets[label]["y"]

            F_total += np.array([F_x_body, F_y_body], dtype=float)

            Mz = x_i * F_y_body - y_i * F_x_body
            Mz_total += Mz
            
        return F_total, float(Mz_total)
        

    def _update_dynamics(self, dt: float, forces: np.ndarray, moments: float)->None:
        linear_acc, yaw_acc = self.calculate_accelerations(forces, moments)

        ax = float(linear_acc[0])
        ay = float(linear_acc[1])

        self.state.velocity_x += ax * dt
        self.state.velocity_y += ay * dt
    
        self.state.yaw_rate += yaw_acc * dt

        self.state.yaw += self.state.yaw_rate * dt

        R = self._rotation_matrix()
        v_body = np.array([self.state.velocity_x, self.state.velocity_y], dtype=float)
        v_inertial = R @ v_body

        self.state.x += v_inertial[0] * dt
        self.state.y += v_inertial[1] * dt

    def get_axle_inputs(self)->Dict:
        axle_velocities={
            label: self.get_axle_velocity(idx, frame="body")
            for idx, label in enumerate(self.axle_labels)
        }

        return {
            "axle_velocities": axle_velocities,
            "velocity_x": self.state.velocity_x,
            "velocity_y": self.state.velocity_y,
            "yaw_rate": self.state.yaw_rate,
        }

    def get_wheel_inputs(self)->Dict:
        wheel_velocities={
            label: self.get_wheel_velocity(label, frame="body")
            for label in self.wheel_labels
        }
        load_transfer_ax = self.state.ax - self.state.velocity_y * self.state.yaw_rate
        load_transfer_ay = self.state.ay + self.state.velocity_x * self.state.yaw_rate

        return {
            "wheel_velocities": wheel_velocities,
            "velocity_x": self.state.velocity_x,
            "velocity_y": self.state.velocity_y,
            "yaw_rate": self.state.yaw_rate,
            "load_transfer_ax": load_transfer_ax,
            "load_transfer_ay": load_transfer_ay,
        }
    
    def get_outputs(self) -> Dict:
        axle_velocities={
            label: self.get_axle_velocity(idx, frame="wheel")
            for idx, label in enumerate(self.axle_labels)
        }
        wheel_velocities={
            label: self.get_wheel_velocity(label, frame="wheel")
            for label in self.wheel_labels
        }

        front_steering = self.axles["F"].steering
        front_state = self.axles["F"].get_state()
        front_road_wheel_angle = float(front_state.get("steering_angle", 0.0))
        front_road_wheel_rate = float(front_state.get("steering_rate", 0.0))
        if front_steering is not None:
            front_steering_wheel_angle = front_steering.road_wheel_angle_to_steering_wheel_angle(
                front_road_wheel_angle
            )
            front_steering_wheel_rate = front_steering.road_wheel_rate_to_steering_wheel_rate(
                front_road_wheel_rate
            )
        else:
            front_steering_wheel_angle = 0.0
            front_steering_wheel_rate = 0.0

        return {
            "velocity_x": self.state.velocity_x,
            "velocity_y": self.state.velocity_y,
            "yaw": self.state.yaw,
            "yaw_rate": self.state.yaw_rate,
            "ax": self.state.ax,
            "ay": self.state.ay,
            "front_steering_angle": front_road_wheel_angle,
            "front_steering_rate": front_road_wheel_rate,
            "front_road_wheel_angle": front_road_wheel_angle,
            "front_road_wheel_rate": front_road_wheel_rate,
            "front_steering_wheel_angle": float(front_steering_wheel_angle),
            "front_steering_wheel_rate": float(front_steering_wheel_rate),
            "axle_velocities": axle_velocities,
            "wheel_velocities": wheel_velocities,
        }


VehicleBody = StbwVehicleBody
