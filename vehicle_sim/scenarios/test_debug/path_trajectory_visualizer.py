"""
범용 Path → Trajectory 변환 및 애니메이션 시각화 도구

어떤 경로(x, y 배열)든 Trajectory로 변환하고 애니메이션으로 확인할 수 있습니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import sys
from pathlib import Path
from typing import Tuple, Optional

# 상위 디렉토리를 path에 추가
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from vehicle_sim.scenarios import TrajectoryPoint, Trajectory, path_to_trajectory


def visualize_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    path_name: str = "Custom Path",
    velocity: float = 15.0,
    dt: float = 0.01,
    vehicle_length: float = 4.5,
    vehicle_width: float = 2.0,
    frame_skip: int = 5,
    max_curvature_limit: Optional[float] = None
):
    """
    임의의 경로(x, y)를 Trajectory로 변환하고 애니메이션으로 시각화

    Args:
        x: x 좌표 배열 [m]
        y: y 좌표 배열 [m]
        path_name: 경로 이름
        velocity: 차량 속도 [m/s]
        dt: 시간 간격 [s]
        vehicle_length: 차량 길이 [m]
        vehicle_width: 차량 폭 [m]
        frame_skip: 애니메이션 프레임 스킵
        max_curvature_limit: 최대 곡률 제한 표시 [1/m] (None이면 표시 안함)
    """
    print("=" * 70)
    print(f"Path → Trajectory 변환 및 애니메이션: {path_name}")
    print("=" * 70)

    # 1. 경로 정보
    print("\n[1단계] 입력 경로 분석")
    print(f"   - 경로 이름: {path_name}")
    print(f"   - 포인트 수: {len(x)}")
    print(f"   - X 범위: [{x.min():.2f}, {x.max():.2f}] m")
    print(f"   - Y 범위: [{y.min():.2f}, {y.max():.2f}] m")

    # 경로 길이 계산
    path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    print(f"   - 경로 길이: {path_length:.2f} m")

    # 2. Trajectory로 변환
    print(f"\n[2단계] Path → Trajectory 변환")
    print(f"   - 시간 간격 dt: {dt} s")
    print(f"   - 기준 속도: {velocity} m/s")

    trajectory = path_to_trajectory(x, y, dt=dt, initial_velocity=velocity)

    print(f"   ✓ Trajectory 포인트 수: {len(trajectory)}")
    print(f"   ✓ 총 시뮬레이션 시간: {trajectory[-1].t:.2f} s")

    # 3. Trajectory 통계
    print("\n[3단계] Trajectory 분석")
    velocities = np.array([p.v for p in trajectory.points])
    curvatures = np.array([p.kappa for p in trajectory.points])
    yaw_rates = np.array([p.yaw_rate for p in trajectory.points])
    yaws = np.array([p.yaw for p in trajectory.points])
    accels = np.array([p.a for p in trajectory.points])

    print(f"   ✓ 속도 - 평균: {velocities.mean():.2f} m/s, 범위: [{velocities.min():.2f}, {velocities.max():.2f}]")
    print(f"   ✓ 가속도 - 평균: {accels.mean():.2f} m/s², 범위: [{accels.min():.2f}, {accels.max():.2f}]")
    print(f"   ✓ 곡률 - 최대: {np.abs(curvatures).max():.4f} 1/m, 범위: [{curvatures.min():.4f}, {curvatures.max():.4f}]")
    print(f"   ✓ 요레이트 - 최대: {np.abs(yaw_rates).max():.4f} rad/s, 범위: [{yaw_rates.min():.4f}, {yaw_rates.max():.4f}]")
    print(f"   ✓ 요각 - 범위: [{np.degrees(yaws.min()):.2f}°, {np.degrees(yaws.max()):.2f}°]")

    # 4. 애니메이션 생성
    print("\n[4단계] 애니메이션 생성")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{path_name} - Trajectory Animation (v={velocity} m/s)',
                 fontsize=14, fontweight='bold')

    # 서브플롯 1: 경로 애니메이션
    ax1 = axes[0, 0]
    ax1.set_title('Vehicle Path', fontsize=11, fontweight='bold')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 전체 경로 (반투명)
    ax1.plot(x, y, 'b--', alpha=0.3, linewidth=1.5, label='Reference Path')

    # 지나온 경로
    traj_line, = ax1.plot([], [], 'r-', linewidth=2.5, label='Traveled')

    # 차량 표시 (사각형)
    vehicle = Rectangle((0, 0), vehicle_length, vehicle_width,
                        fill=True, facecolor='red', edgecolor='darkred',
                        linewidth=2, alpha=0.8)
    ax1.add_patch(vehicle)

    # 현재 위치 마커
    current_pos, = ax1.plot([], [], 'go', markersize=10, label='Current Position')

    ax1.legend(loc='upper right', fontsize=9)

    # 서브플롯 2: 속도와 가속도
    ax2 = axes[0, 1]
    ax2.set_title('Velocity & Acceleration', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Velocity [m/s]', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.grid(True, alpha=0.3)

    t_vals = [p.t for p in trajectory.points]
    v_vals = [p.v for p in trajectory.points]
    a_vals = [p.a for p in trajectory.points]

    ax2.plot(t_vals, v_vals, 'b-', linewidth=1.5, alpha=0.5, label='Velocity')
    vel_marker, = ax2.plot([], [], 'bo', markersize=8)

    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel('Acceleration [m/s²]', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2_twin.plot(t_vals, a_vals, 'orange', linewidth=1.5, alpha=0.5, label='Acceleration')
    acc_marker, = ax2_twin.plot([], [], 'o', color='orange', markersize=8)

    ax2.legend(loc='upper left', fontsize=9)
    ax2_twin.legend(loc='upper right', fontsize=9)

    # 서브플롯 3: 요각과 요레이트
    ax3 = axes[1, 0]
    ax3.set_title('Yaw Angle & Yaw Rate', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Yaw [rad]', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.grid(True, alpha=0.3)

    yaw_vals = [p.yaw for p in trajectory.points]
    yaw_rate_vals = [p.yaw_rate for p in trajectory.points]

    ax3.plot(t_vals, yaw_vals, 'b-', linewidth=1.5, alpha=0.5, label='Yaw')
    yaw_marker, = ax3.plot([], [], 'bo', markersize=8)

    ax3_twin = ax3.twinx()
    ax3_twin.set_ylabel('Yaw Rate [rad/s]', color='g')
    ax3_twin.tick_params(axis='y', labelcolor='g')
    ax3_twin.plot(t_vals, yaw_rate_vals, 'g-', linewidth=1.5, alpha=0.5, label='Yaw Rate')
    yaw_rate_marker, = ax3_twin.plot([], [], 'go', markersize=8)

    ax3.legend(loc='upper left', fontsize=9)
    ax3_twin.legend(loc='upper right', fontsize=9)

    # 서브플롯 4: 곡률
    ax4 = axes[1, 1]
    ax4.set_title('Path Curvature', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Curvature [1/m]', color='m')
    ax4.tick_params(axis='y', labelcolor='m')
    ax4.grid(True, alpha=0.3)

    kappa_vals = [p.kappa for p in trajectory.points]

    ax4.plot(t_vals, kappa_vals, 'm-', linewidth=1.5, alpha=0.5, label='Curvature')
    kappa_marker, = ax4.plot([], [], 'mo', markersize=8)

    # 최대 곡률 제한선 표시
    if max_curvature_limit is not None:
        ax4.axhline(y=max_curvature_limit, color='r', linestyle='--', alpha=0.5,
                    label=f'Max Limit (±{max_curvature_limit})')
        ax4.axhline(y=-max_curvature_limit, color='r', linestyle='--', alpha=0.5)

    ax4.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    # 애니메이션 업데이트 함수
    def update(frame):
        if frame >= len(trajectory.points):
            return

        point = trajectory.points[frame]

        # 차량 위치 및 방향
        cos_yaw = np.cos(point.yaw)
        sin_yaw = np.sin(point.yaw)

        vehicle_x = point.x - (vehicle_length / 2) * cos_yaw + (vehicle_width / 2) * sin_yaw
        vehicle_y = point.y - (vehicle_length / 2) * sin_yaw - (vehicle_width / 2) * cos_yaw

        vehicle.set_xy((vehicle_x, vehicle_y))
        vehicle.angle = np.degrees(point.yaw)

        # 지나온 경로
        x_history = [p.x for p in trajectory.points[:frame+1]]
        y_history = [p.y for p in trajectory.points[:frame+1]]
        traj_line.set_data(x_history, y_history)

        # 현재 위치
        current_pos.set_data([point.x], [point.y])

        # 마커 업데이트
        vel_marker.set_data([point.t], [point.v])
        acc_marker.set_data([point.t], [point.a])
        yaw_marker.set_data([point.t], [point.yaw])
        yaw_rate_marker.set_data([point.t], [point.yaw_rate])
        kappa_marker.set_data([point.t], [point.kappa])

        return (vehicle, traj_line, current_pos, vel_marker, acc_marker,
                yaw_marker, yaw_rate_marker, kappa_marker)

    # 애니메이션 실행
    total_frames = len(trajectory.points) // frame_skip

    print(f"   ✓ 총 프레임 수: {total_frames}")
    print(f"   ✓ 프레임 스킵: {frame_skip}")
    print("\n애니메이션을 시작합니다... (창을 닫으면 종료)")

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=range(0, len(trajectory.points), frame_skip),
        interval=20,
        blit=False,
        repeat=True
    )

    plt.show()

    print("\n시각화 완료!")
    return trajectory


# ==================== Sine 경로 생성 ====================

def generate_sine_path(
    total_distance: float = 100.0,
    max_curvature: float = 0.1,
    max_curvature_rate: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """사인파 경로 생성"""
    from vehicle_sim.scenarios import paths
    x, y = paths.generate_sine_path(
        total_distance=total_distance,
        max_curvature=max_curvature,
        max_curvature_rate=max_curvature_rate
    )
    return x, y


# ==================== 메인 실행 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Sine Path → Trajectory 시각화 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python path_trajectory_visualizer.py --velocity 15 --distance 100
  python path_trajectory_visualizer.py --velocity 20 --distance 150 --curvature 0.15
        """
    )

    parser.add_argument('--velocity', type=float, default=15.0,
                       help='차량 속도 [m/s]')
    parser.add_argument('--distance', type=float, default=100.0,
                       help='경로 길이 [m]')
    parser.add_argument('--curvature', type=float, default=0.1,
                       help='최대 곡률 [1/m]')
    parser.add_argument('--curvature_rate', type=float, default=0.01,
                       help='최대 곡률 변화율 [1/m²]')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='시간 간격 [s]')

    args = parser.parse_args()

    # Sine 경로 생성
    x, y = generate_sine_path(
        total_distance=args.distance,
        max_curvature=args.curvature,
        max_curvature_rate=args.curvature_rate
    )
    path_name = f"Sine Path ({args.distance}m)"
    max_curv = args.curvature

    # 시각화 실행
    visualize_trajectory(
        x, y,
        path_name=path_name,
        velocity=args.velocity,
        dt=args.dt,
        max_curvature_limit=max_curv
    )
