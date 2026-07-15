"""STBW lateral tire wrapper backed by the shared Fiala tire model."""

from pathlib import Path
from typing import Optional, Union

from vehicle_sim.models.tire.lateral.fiala_lateral_tire import (
    FialaLateralTireModel,
    FialaLateralTireParameters,
    FialaLateralTireState,
)
from vehicle_sim.utils.config_loader import load_param


STBW_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "stbw.yaml"


def _resolve_config_path(config_path: Optional[Union[str, Path]]):
    if config_path is None or str(config_path).lower() in {"stbw", "stbw.yaml"}:
        return STBW_CONFIG_PATH
    return config_path


def _load_lateral_parameters(config_path: Optional[Union[str, Path]]) -> FialaLateralTireParameters:
    tire_param = load_param("tire", _resolve_config_path(config_path))
    lateral_param = tire_param.get("lateral", {})
    return FialaLateralTireParameters(
        C_alpha=float(lateral_param.get("C_alpha", FialaLateralTireParameters.C_alpha)),
        mu=float(lateral_param.get("mu", tire_param.get("mu", FialaLateralTireParameters.mu))),
        trail=float(lateral_param.get("trail", FialaLateralTireParameters.trail)),
        vx_epsilon=float(lateral_param.get("vx_epsilon", FialaLateralTireParameters.vx_epsilon)),
    )


class StbwLateralTireModel(FialaLateralTireModel):
    def __init__(
        self,
        parameters: Optional[FialaLateralTireParameters] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__(
            parameters=parameters
            if parameters is not None
            else _load_lateral_parameters(config_path)
        )


StbwLateralTireParameters = FialaLateralTireParameters
StbwLateralTireState = FialaLateralTireState
LateralTireModel = StbwLateralTireModel

__all__ = [
    "FialaLateralTireModel",
    "FialaLateralTireParameters",
    "FialaLateralTireState",
    "LateralTireModel",
    "STBW_CONFIG_PATH",
    "StbwLateralTireModel",
    "StbwLateralTireParameters",
    "StbwLateralTireState",
]
