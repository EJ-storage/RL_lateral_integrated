"""
사인파 경로 생성기 - 차량 주행 가능한 x, y 좌표 생성
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def generate_sine_path(
    total_distance: float = 100.0,
    max_curvature: float = 0.1,
    max_curvature_rate: float = 0.01,
    velocity: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    차량이 주행 가능한 사인파 경로의 x, y 좌표 자동 생성

    최대 곡률 및 곡률 변화율 제약을 만족하도록 진폭과 파장을 자동 계산합니다.

    Args:
        total_distance: 총 경로 길이 [m] (기본값: 100.0)
        max_curvature: 최대 허용 곡률 [1/m] (기본값: 0.1, 약 10m 회전반경)
        max_curvature_rate: 최대 허용 곡률 변화율 [1/m²] (기본값: 0.01)
        velocity: 참조 속도 [m/s] (곡률 변화율 계산용, 기본값: 10.0)

    Returns:
        (x, y): x 좌표 배열 [m], y 좌표 배열 [m]

    Note:
        사인파 경로의 곡률 및 곡률 변화율:
        - 곡률: κ(s) ≈ A * k² * sin(k*s)
        - 곡률 최댓값: κ_max = A * k²
        - 곡률 변화율: dκ/ds ≈ A * k³ * cos(k*s)
        - 곡률 변화율 최댓값: dκ/ds_max = A * k³

        제약 조건:
        1. A * k² ≤ max_curvature
        2. A * k³ ≤ max_curvature_rate

        두 제약을 모두 만족하도록 진폭 A와 파수 k를 계산합니다.
    """
    # 점 간격 고정 [m]
    spacing = 0.1

    # 초기 파장 설정: 총 거리의 1/3 (약 3주기)
    wavelength = total_distance / 3.0
    k = 2 * np.pi / wavelength

    # 제약 조건 1: 최대 곡률 제약에서의 진폭
    # A * k² ≤ max_curvature → A ≤ max_curvature / k²
    amplitude_from_curvature = max_curvature / (k**2)

    # 제약 조건 2: 최대 곡률 변화율 제약에서의 진폭
    # A * k³ ≤ max_curvature_rate → A ≤ max_curvature_rate / k³
    amplitude_from_curvature_rate = max_curvature_rate / (k**3)

    # 두 제약 조건 중 더 작은 값을 선택 (더 보수적인 제약)
    amplitude = min(amplitude_from_curvature, amplitude_from_curvature_rate)

    # x 좌표 생성 [m]
    x = np.arange(0, total_distance + spacing, spacing)

    # y 좌표 생성 (사인파) [m]
    y = amplitude * np.sin(k * x)

    return x, y


def plot_path_properties(x: np.ndarray, y: np.ndarray) -> None:
    """
    경로의 곡률 및 곡률 변화율 시각화

    Args:
        x: x 좌표 배열 [m]
        y: y 좌표 배열 [m]
    """
    # 1차 미분 계산 (수치 미분)
    dx = np.gradient(x)
    dy = np.gradient(y)

    # 2차 미분 계산
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # 곡률 계산: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**(3/2)
    curvature = numerator / (denominator + 1e-10)

    # 곡률 변화율 계산
    curvature_rate = np.gradient(curvature)

    # 플롯
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 곡률
    axes[0].plot(x, curvature, 'r-', linewidth=2)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('X [m]')
    axes[0].set_ylabel('Curvature [1/m]')
    axes[0].set_title(f'Path Curvature (Max: {np.max(np.abs(curvature)):.4f} 1/m)')
    axes[0].grid(True, alpha=0.3)

    # 곡률 변화율
    axes[1].plot(x, curvature_rate, 'g-', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('X [m]')
    axes[1].set_ylabel('Curvature Rate [1/m²]')
    axes[1].set_title(f'Curvature Rate (Max: {np.max(np.abs(curvature_rate)):.4f} 1/m²)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
