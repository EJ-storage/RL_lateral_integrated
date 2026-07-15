"""YAML parameter loader for concrete vehicle model configurations.

The common modules under ``vehicle_sim.models`` do not own a default parameter
YAML. Concrete wrappers such as ``vehicle_sim.stbw_model`` must pass their own
configuration file path, or a registered alias that points to that wrapper's
configuration.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import yaml


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
STBW_CONFIG_PATH = PROJECT_ROOT / "stbw_model" / "config" / "stbw.yaml"

YAML_FILE_MAP = {
    "stbw": STBW_CONFIG_PATH,
    "stbw.yaml": STBW_CONFIG_PATH,
}


def load_param(module_name: str, config_path: Optional[Union[str, Path]] = None) -> Dict:
    """Load a top-level parameter block from an explicit vehicle YAML.

    ``config_path=None`` intentionally returns an empty dict. This prevents the
    generic ``vehicle_sim.models`` components from silently reading legacy
    model-local parameter YAML files when they are constructed without a
    concrete vehicle wrapper.
    """
    if config_path is None:
        return {}

    if isinstance(config_path, Path):
        selected_path = config_path
    else:
        config_key = str(config_path)
        selected_path = YAML_FILE_MAP.get(config_key, Path(config_key))

    selected_path = Path(selected_path)
    if not selected_path.exists():
        available_aliases = ", ".join(sorted(YAML_FILE_MAP.keys()))
        raise FileNotFoundError(
            f"Config file not found: {selected_path}\n"
            f"Available aliases: {available_aliases}"
        )

    with open(selected_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    if not isinstance(full_config, dict):
        return {}

    module_config = full_config.get(module_name, {})
    return module_config if isinstance(module_config, dict) else {}
