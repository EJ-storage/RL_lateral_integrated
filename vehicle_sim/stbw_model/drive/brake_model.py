"""STBW brake model compatibility aliases."""

from pathlib import Path

from vehicle_sim.models.drive.brake_model import (
    BrakeModel,
    BrakeParameters,
    BrakeState,
)


STBW_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "stbw.yaml"


def _resolve_config_path(config_path):
    if config_path is None or str(config_path).lower() in {"stbw", "stbw.yaml"}:
        return STBW_CONFIG_PATH
    return config_path


class StbwBrakeModel(BrakeModel):
    def __init__(self, config_path=None):
        super().__init__(config_path=_resolve_config_path(config_path))


StbwBrakeParameters = BrakeParameters
StbwBrakeState = BrakeState

__all__ = [
    "STBW_CONFIG_PATH",
    "StbwBrakeModel",
    "StbwBrakeParameters",
    "StbwBrakeState",
]
