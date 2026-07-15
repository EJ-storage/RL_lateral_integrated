"""
Trajectory data structures and utilities
"""

from typing import List
from dataclasses import dataclass
import numpy as np


@dataclass
class TrajectoryPoint:
    """트래젝토리의 단일 지점"""
    t: float        # 시간 [s]
    x: float        # x 위치 [m]
    y: float        # y 위치 [m]
    yaw: float      # 요각 [rad]
    yaw_rate: float # 요레이트 [rad/s]
    kappa: float    # 곡률 [1/m]
    v: float        # 속도 [m/s]
    a: float        # 가속도 [m/s^2]


class Trajectory:
    """트래젝토리 포인트 리스트로 구성된 트래젝토리"""

    def __init__(self, points: List[TrajectoryPoint] = None):
        self.points = points if points is not None else []

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

    def append(self, point: TrajectoryPoint):
        """트래젝토리 포인트 추가"""
        self.points.append(point)

    def get_state_at_time(self, time: float) -> TrajectoryPoint:
        """
        주어진 시간에서의 트래젝토리 상태를 반환 (선형 보간)

        Args:
            time: 조회 시간 [s]

        Returns:
            보간된 트래젝토리 포인트
        """
        if len(self.points) == 0:
            raise ValueError("트래젝토리가 비어있습니다")

        if time <= self.points[0].t:
            return self.points[0]

        if time >= self.points[-1].t:
            return self.points[-1]

        # 주변 포인트 찾기
        for i in range(len(self.points) - 1):
            if self.points[i].t <= time <= self.points[i + 1].t:
                p1 = self.points[i]
                p2 = self.points[i + 1]

                # 선형 보간
                alpha = (time - p1.t) / (p2.t - p1.t)

                return TrajectoryPoint(
                    t=time,
                    x=p1.x + alpha * (p2.x - p1.x),
                    y=p1.y + alpha * (p2.y - p1.y),
                    yaw=p1.yaw + alpha * (p2.yaw - p1.yaw),
                    yaw_rate=p1.yaw_rate + alpha * (p2.yaw_rate - p1.yaw_rate),
                    kappa=p1.kappa + alpha * (p2.kappa - p1.kappa),
                    v=p1.v + alpha * (p2.v - p1.v),
                    a=p1.a + alpha * (p2.a - p1.a)
                )

        return self.points[-1]


def path_to_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    dt: float = 0.01,
    initial_velocity: float = 10.0
) -> Trajectory:
    """
    경로(x, y)를 dt 간격의 트래젝토리로 변환

    Args:
        x: x 좌표 배열 [m]
        y: y 좌표 배열 [m]
        dt: 시간 간격 [s] (기본값: 0.01)
        initial_velocity: 초기 속도 [m/s] (기본값: 10.0, 일정 속도 유지)

    Returns:
        생성된 트래젝토리
    """
    trajectory = Trajectory()

    # 1차 미분 계산
    dx = np.gradient(x)
    dy = np.gradient(y)

    # 2차 미분 계산
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # 시간 배열 생성 (dt 간격)
    t = np.arange(len(x)) * dt

    # 일정 속도 및 가속도
    velocity = initial_velocity
    acceleration = 0.0

    for i in range(len(x)):
        # 요각 계산: arctan2(dy, dx)
        yaw = np.arctan2(dy[i], dx[i])

        # 곡률 계산: κ = |dx*d²y - dy*d²x| / (dx² + dy²)^(3/2)
        numerator = abs(dx[i] * ddy[i] - dy[i] * ddx[i])
        denominator = (dx[i]**2 + dy[i]**2)**(3/2)
        kappa = numerator / (denominator + 1e-10)

        # 부호 고려 (왼쪽 회전이 양수)
        cross_product = dx[i] * ddy[i] - dy[i] * ddx[i]
        if cross_product < 0:
            kappa = -kappa

        # 요레이트 계산: ψ_dot = v * κ
        yaw_rate = velocity * kappa

        point = TrajectoryPoint(
            t=t[i],
            x=x[i],
            y=y[i],
            yaw=yaw,
            yaw_rate=yaw_rate,
            kappa=kappa,
            v=velocity,
            a=acceleration
        )

        trajectory.append(point)

    return trajectory
