from typing import Dict, List, Optional


HISTORY_FIELDS = (
    "t",
    "x",
    "y",
    "yaw",
    "vx",
    "vy",
    "yaw_rate",
    "steer",
    "steer_dot",
    "ax",
    "ay",
    "beta",
    "curvature",
    "target_curvature",
    "target_curvature_dot",
    "target_lateral_accel",
    "target_lateral_accel_dot",
    "speed_hold_target_speed_mps",
    "speed_hold_speed_error_mps",
    "speed_hold_target_accel_mps2",
    "speed_hold_raw_target_accel_mps2",
    "speed_hold_integrator_state_mps",
    "speed_hold_drive_torque_nm",
    "speed_hold_brake_motor_torque",
    "action",
    "reward",
    "reward_track",
    "reward_slip",
    "reward_used",
    "terminated",
    "truncated",
    "terminated_reason",
)


class EnvHistory:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._data: Dict[str, List[Optional[float]]] = {field: [] for field in HISTORY_FIELDS}

    def append(self, **values) -> None:
        for field in HISTORY_FIELDS:
            self._data[field].append(values.get(field))

    def to_dict(self) -> Dict[str, List[Optional[float]]]:
        return {key: list(value) for key, value in self._data.items()}
