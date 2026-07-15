"""
Configuration Loader Utility
YAML 설정 파일을 로드하는 공통 유틸리티
"""

import yaml
from pathlib import Path
from typing import Dict, Optional, Union


# =============================================================================
# [YAML 파일 등록 영역]
# -----------------------------------------------------------------------------
# 나중에 새로운 YAML 파일이 추가되면 아래 YAML_FILE_MAP에만 등록하면 됨.
#
# 사용 예:
#   load_param("vehicle")                      -> default 사용
#   load_param("vehicle", "default")          -> 기존 vehicle_standard.yaml 사용
#   load_param("vehicle", "stbw")             -> stbw_vehicle_standard.yaml 사용
#   load_param("vehicle", r"C:\...\abc.yaml") -> 직접 절대경로 사용
#
# 새로운 YAML 추가 방법:
#   1) 아래 YAML_FILE_MAP에 별칭(alias) 추가
#   2) 키는 사용하기 쉬운 이름으로 지정
#   3) 값은 실제 YAML 파일 경로(Path)로 지정
#
# 예시:
#   "test_vehicle": PARAM_DIR / "test_vehicle.yaml"
#   "my_custom": Path(r"C:\Users\...\my_custom.yaml")
#
# 이후 아래처럼 호출 가능:
#   load_param("vehicle", "test_vehicle")
#   load_param("brake", "my_custom")
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
PARAM_DIR = PROJECT_ROOT / "models" / "params"

YAML_FILE_MAP = {
    # 기존 기본 YAML
    "default": PARAM_DIR / "vehicle_standard.yaml",

    # STBW용 YAML
    "stbw": PARAM_DIR / "stbw_vehicle_standard.yaml",
}


def load_param(module_name: str, config_path: Optional[Union[str, Path]] = None) -> Dict:
    """
    차량 모듈별 파라미터 로드

    Args:
        module_name:
            불러올 모듈 이름
            예: 'brake', 'motor', 'suspension', 'steering', 'tire', 'vehicle_body'

        config_path:
            아래 3가지 방식 모두 지원함.

            1) None
               - YAML_FILE_MAP의 'default' 파일을 사용함.

            2) 별칭(alias) 문자열
               - 예: "default", "stbw"
               - YAML_FILE_MAP에 등록된 경로를 사용함.

            3) 실제 파일 경로(str 또는 Path)
               - 예: r"C:\\path\\to\\custom.yaml"
               - 해당 경로의 YAML 파일을 직접 사용함.

    Returns:
        Dict:
            YAML 내부에서 module_name에 해당하는 딕셔너리
            없으면 빈 dict 반환

    Example:
        >>> brake_param = load_param('brake')
        >>> brake_param = load_param('brake', 'default')
        >>> brake_param = load_param('brake', 'stbw')
        >>> brake_param = load_param('brake', r'C:\\path\\to\\custom.yaml')

    [나중에 YAML 파일 추가하는 방법]
    -------------------------------------------------------------------------
    예를 들어 새로운 YAML 파일 test_vehicle.yaml을 추가했다고 가정하면,
    위의 YAML_FILE_MAP에 아래 한 줄만 추가하면 됨.

        YAML_FILE_MAP = {
            "default": PARAM_DIR / "vehicle_standard.yaml",
            "stbw": Path(r"...\\stbw_vehicle_standard.yaml"),
            "test_vehicle": PARAM_DIR / "test_vehicle.yaml",
        }

    그러면 아래처럼 바로 사용할 수 있음.

        load_param("vehicle_body", "test_vehicle")

    [권장 사항]
    -------------------------------------------------------------------------
    1) 자주 쓰는 YAML은 YAML_FILE_MAP에 별칭으로 등록해서 사용
    2) 일회성 파일은 직접 경로를 넣어서 사용
    3) 팀 프로젝트라면 절대경로보다 프로젝트 내부 상대경로 기반 등록이 더 좋음
       예:
           "stbw": PARAM_DIR / "stbw_vehicle_standard.yaml"

    현재는 사용자가 요청한 절대경로를 그대로 반영했지만,
    파일을 프로젝트 내부로 옮길 수 있다면 위와 같이 상대경로 방식이 더 유지보수에 유리함.
    """
    # -------------------------------------------------------------------------
    # config_path가 지정되지 않으면 기본 YAML 사용
    # -------------------------------------------------------------------------
    if config_path is None:
        selected_path = YAML_FILE_MAP["default"]

    else:
        # Path 객체가 들어오면 그대로 사용
        if isinstance(config_path, Path):
            selected_path = config_path

        # 문자열이 들어오면
        else:
            # 1) YAML_FILE_MAP에 등록된 별칭인지 먼저 확인
            if config_path in YAML_FILE_MAP:
                selected_path = YAML_FILE_MAP[config_path]
            # 2) 등록된 별칭이 아니면 실제 파일 경로라고 간주
            else:
                selected_path = Path(config_path)

    selected_path = Path(selected_path)

    if not selected_path.exists():
        available_aliases = ", ".join(YAML_FILE_MAP.keys())
        raise FileNotFoundError(
            f"Config file not found: {selected_path}\n"
            f"Available aliases: {available_aliases}"
        )

    with open(selected_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    if not full_config:
        return {}

    return full_config.get(module_name, {})
