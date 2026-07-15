"""STBW longitudinal tire wrapper backed by the shared Fiala tire model."""

from pathlib import Path
from typing import Optional, Union

from vehicle_sim.models.tire.longitudinal.fiala_longitudinal_tire import (
    FialaLongitudinalTireModel,
    FialaLongitudinalTireParameters,
    FialaLongitudinalTireState,
)
from vehicle_sim.utils.config_loader import load_param


STBW_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "stbw.yaml"


def _resolve_config_path(config_path: Optional[Union[str, Path]]):
    if config_path is None or str(config_path).lower() in {"stbw", "stbw.yaml"}:
        return STBW_CONFIG_PATH
    return config_path


def _load_longitudinal_parameters(
    config_path: Optional[Union[str, Path]],
) -> FialaLongitudinalTireParameters:
    resolved_path = _resolve_config_path(config_path)
    tire_param = load_param("tire", resolved_path)
    long_param = tire_param.get("longitudinal", {})
    vehicle_spec = load_param("vehicle_spec", resolved_path)
    wheel_spec = vehicle_spec.get("wheel", {})
    return FialaLongitudinalTireParameters(
        C_x=float(long_param.get("C_x", FialaLongitudinalTireParameters.C_x)),
        mu=float(long_param.get("mu", tire_param.get("mu", FialaLongitudinalTireParameters.mu))),
        v_min=float(long_param.get("v_min", FialaLongitudinalTireParameters.v_min)),
        R_eff=float(wheel_spec.get("R_eff", FialaLongitudinalTireParameters.R_eff)),
        epsilon=float(long_param.get("epsilon", FialaLongitudinalTireParameters.epsilon)),
    )


class StbwLongitudinalTireModel(FialaLongitudinalTireModel):
    def __init__(
        self,
        parameters: Optional[FialaLongitudinalTireParameters] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__(
            parameters=parameters
            if parameters is not None
            else _load_longitudinal_parameters(config_path)
        )


StbwLongitudinalTireParameters = FialaLongitudinalTireParameters
StbwLongitudinalTireState = FialaLongitudinalTireState
LongitudinalTireModel = StbwLongitudinalTireModel

__all__ = [
    "FialaLongitudinalTireModel",
    "FialaLongitudinalTireParameters",
    "FialaLongitudinalTireState",
    "LongitudinalTireModel",
    "STBW_CONFIG_PATH",
    "StbwLongitudinalTireModel",
    "StbwLongitudinalTireParameters",
    "StbwLongitudinalTireState",
]
