from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union


DriveAxlesInput = Union[str, Iterable[str]]


def normalize_drive_axles(drive_axles: DriveAxlesInput) -> Tuple[str, ...]:
    if isinstance(drive_axles, str):
        text = drive_axles.strip().upper()
        compact = "".join(char for char in text if char.isalnum())
        aliases = {
            "F": ("F",),
            "FRONT": ("F",),
            "FWD": ("F",),
            "R": ("R",),
            "REAR": ("R",),
            "RWD": ("R",),
            "FR": ("F", "R"),
            "RF": ("F", "R"),
            "AWD": ("F", "R"),
            "4WD": ("F", "R"),
            "ALL": ("F", "R"),
        }
        if compact in aliases:
            return aliases[compact]
        tokens = [token for token in text.replace("+", ",").replace("/", ",").split(",") if token.strip()]
    else:
        tokens = list(drive_axles)

    normalized = []
    for token in tokens:
        token_text = str(token).strip().upper()
        if token_text in {"F", "FRONT", "FWD"}:
            axle = "F"
        elif token_text in {"R", "REAR", "RWD"}:
            axle = "R"
        else:
            raise ValueError("drive_axles must be one of 'F', 'R', or 'FR'.")
        if axle not in normalized:
            normalized.append(axle)

    if not normalized:
        raise ValueError("drive_axles must select at least one axle.")
    return tuple(axle for axle in ("F", "R") if axle in normalized)


def drive_axles_label(drive_axles: DriveAxlesInput) -> str:
    return "".join(normalize_drive_axles(drive_axles))


def resolve_drive_axles(
    drive_axles: Optional[DriveAxlesInput],
    drive_split_front: Optional[float],
    *,
    default: DriveAxlesInput = "R",
) -> str:
    if drive_axles is not None:
        return drive_axles_label(drive_axles)
    if drive_split_front is None:
        return drive_axles_label(default)

    split = float(drive_split_front)
    if split <= 0.0:
        return "R"
    if split >= 1.0:
        return "F"
    return "FR"


def is_axle_driven(axle_label: str, drive_axles: DriveAxlesInput) -> bool:
    axle_group = "F" if str(axle_label).upper().startswith("F") else "R"
    return axle_group in normalize_drive_axles(drive_axles)


def resolve_drive_split_front(
    drive_axles: DriveAxlesInput,
    drive_split_front: Optional[float],
) -> float:
    axles = normalize_drive_axles(drive_axles)
    if axles == ("F",):
        return 1.0
    if axles == ("R",):
        return 0.0

    if drive_split_front is None:
        return 0.5
    split = float(drive_split_front)
    if not 0.0 <= split <= 1.0:
        raise ValueError("drive_split_front must be within [0.0, 1.0].")
    return split
