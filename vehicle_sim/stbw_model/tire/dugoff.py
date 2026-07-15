#!/bin/python3
"""STBW Modified Dugoff tire wrapper backed by stbw.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from vehicle_sim.models.tire.total.dugoff import (
    ModifiedDugoff4WTireModel as BaseModifiedDugoff4WTireModel,
    ModifiedDugoff4WTireParameters,
    ModifiedDugoff4WTireState,
    ModifiedDugoffTireModel as BaseModifiedDugoffTireModel,
    ModifiedDugoffWheelState,
)
from vehicle_sim.utils.config_loader import load_param


STBW_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "stbw.yaml"


def _resolve_config_path(config_path: Optional[Union[str, Path]]):
    if config_path is None or str(config_path).lower() in {"stbw", "stbw.yaml"}:
        return STBW_CONFIG_PATH
    return config_path


def _float_from(mapping: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return value


def _load_dugoff_parameters(
    config_path: Optional[Union[str, Path]],
) -> ModifiedDugoff4WTireParameters:
    tire_param = load_param("tire", _resolve_config_path(config_path))
    dugoff_param = tire_param.get("modified_dugoff", {})
    if not isinstance(dugoff_param, dict):
        dugoff_param = {}

    return ModifiedDugoff4WTireParameters(
        Re_FL=_float_from(dugoff_param, "Re_FL"),
        Re_FR=_float_from(dugoff_param, "Re_FR"),
        Re_RL=_float_from(dugoff_param, "Re_RL"),
        Re_RR=_float_from(dugoff_param, "Re_RR"),
        Ckappa_FL=_float_from(dugoff_param, "Ckappa_FL"),
        Ckappa_FR=_float_from(dugoff_param, "Ckappa_FR"),
        Ckappa_RL=_float_from(dugoff_param, "Ckappa_RL"),
        Ckappa_RR=_float_from(dugoff_param, "Ckappa_RR"),
        Calpha_FL=_float_from(dugoff_param, "Calpha_FL"),
        Calpha_FR=_float_from(dugoff_param, "Calpha_FR"),
        Calpha_RL=_float_from(dugoff_param, "Calpha_RL"),
        Calpha_RR=_float_from(dugoff_param, "Calpha_RR"),
        muX_FL=_float_from(dugoff_param, "muX_FL"),
        muX_FR=_float_from(dugoff_param, "muX_FR"),
        muX_RL=_float_from(dugoff_param, "muX_RL"),
        muX_RR=_float_from(dugoff_param, "muX_RR"),
        muY_FL=_float_from(dugoff_param, "muY_FL"),
        muY_FR=_float_from(dugoff_param, "muY_FR"),
        muY_RL=_float_from(dugoff_param, "muY_RL"),
        muY_RR=_float_from(dugoff_param, "muY_RR"),
        Veps=_float_from(dugoff_param, "Veps"),
        FzMin=_float_from(dugoff_param, "FzMin"),
        kappaMin=_float_from(dugoff_param, "kappaMin"),
        kappaMax=_float_from(dugoff_param, "kappaMax"),
        alphaMax=_float_from(dugoff_param, "alphaMax"),
    )


class StbwModifiedDugoff4WTireModel(BaseModifiedDugoff4WTireModel):
    def __init__(
        self,
        parameters: Optional[ModifiedDugoff4WTireParameters] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__(
            parameters=parameters
            if parameters is not None
            else _load_dugoff_parameters(config_path)
        )


ModifiedDugoff4WTireModel = StbwModifiedDugoff4WTireModel
ModifiedDugoffTireModel = StbwModifiedDugoff4WTireModel


__all__ = [
    "BaseModifiedDugoff4WTireModel",
    "BaseModifiedDugoffTireModel",
    "ModifiedDugoff4WTireModel",
    "ModifiedDugoff4WTireParameters",
    "ModifiedDugoff4WTireState",
    "ModifiedDugoffTireModel",
    "ModifiedDugoffWheelState",
    "STBW_CONFIG_PATH",
    "StbwModifiedDugoff4WTireModel",
]
