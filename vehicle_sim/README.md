# Vehicle Dynamics Simulation

차량 동역학 시뮬레이션 프로젝트입니다. E-corner 기반 전기 차량의 동역학을 시뮬레이션합니다.

## 프로젝트 구조

```
vehicle_sim/
├── config/                      # 설정 파일
│   └── vehicle_config.yaml     # 차량 파라미터 (TODO)
├── models/                      # 차량 모델
│   ├── vehicle_body/           # 차체 동역학
│   │   └── vehicle_body.py
│   └── e-corner/               # E-corner 모듈
│       ├── e_corner.py
│       ├── tire/               # 타이어 모델
│       │   ├── tire_model.py
│       │   ├── longitudinal/   # 종방향 타이어
│       │   │   └── longitudinal_tire.py
│       │   └── lateral/        # 횡방향 타이어
│       │       └── lateral_tire.py
│       ├── suspension/         # 서스펜션 모델
│       │   └── suspension_model.py
│       ├── drive/              # 구동 모터 모델
│       │   └── drive_model.py
│       ├── steering/           # 조향 모델
│       │   └── steering_model.py
│       └── config/             # 설정 관리
│           └── corner_config.py
├── controllers/                # 제어기
│   ├── base_controller.py
│   ├── driver_controller.py
│   └── torque_vectoring_controller.py
├── scenarios/                  # 시뮬레이션 시나리오
│   ├── base_scenario.py
│   ├── straight_line_scenario.py
│   ├── constant_radius_scenario.py
│   └── double_lane_change_scenario.py
├── utils/                      # 유틸리티 함수
│   ├── math_utils.py
│   └── coordinate_transform.py
├── simulator.py                # 메인 시뮬레이터
└── main.py                     # 실행 진입점
```

## 주요 기능

### 1. 차량 모델 (models/)
- **VehicleBody**: 6-DOF 강체 동역학 모델
- **ECorner**: 통합 코너 모듈 (타이어 + 서스펜션 + 구동 + 조향)

### 2. 타이어 모델 (models/e-corner/tire/)
- **종방향 타이어**: Magic Formula 기반 종방향 힘 계산
- **횡방향 타이어**: Magic Formula 기반 횡방향 힘 계산
- **복합 슬립**: Similarity method 또는 마찰 타원 방식

### 3. 서스펜션 (models/e-corner/suspension/)
- 스프링-댐퍼 모델
- 비대칭 댐핑
- 범프 스톱 및 이동 제한

### 4. 구동 시스템 (models/e-corner/drive/)
- 전기 모터 동역학
- 토크-속도 특성
- 회생 제동

### 5. 조향 시스템 (models/e-corner/steering/)
- 능동 조향 액추에이터
- 각도 및 속도 제한
- 셀프 얼라이닝 토크 피드백

### 6. 제어기 (controllers/)
- **DriverController**: 운전자 입력을 휠 명령으로 변환
- **TorqueVectoringController**: 토크 배분 제어

### 7. 시나리오 (scenarios/)
- **직선 가속/제동**: 종방향 동역학 테스트
- **정상원 선회**: 횡방향 동역학 테스트
- **더블 레인 체인지**: 과도 응답 테스트

## 사용 방법

### 기본 실행

```python
python main.py
```

### 커스텀 시뮬레이션

```python
from vehicle_sim import VehicleSimulator, SimulatorConfig
from vehicle_sim.controllers import DriverController
from vehicle_sim.scenarios import StraightLineScenario

# 시뮬레이터 생성
config = SimulatorConfig(dt=0.001, max_time=10.0)
simulator = VehicleSimulator(config)

# 시나리오 설정
scenario = StraightLineScenario()
simulator.set_scenario(scenario)

# 제어기 설정
controller = DriverController()
simulator.set_controller(controller)

# 시뮬레이션 실행
results = simulator.run()
```

## 구현해야 할 항목들

### 1. Configuration (최우선)
각 모듈의 파라미터는 config YAML 파일에서 로드됩니다:
- `config/vehicle_config.yaml`: 차량 파라미터 정의 (TODO)
- `corner_config.py`: YAML 로딩 함수 구현 (TODO)

### 2. 함수 구현
각 모듈의 `# TODO:` 주석이 있는 함수들을 구현하세요:
- **타이어 모델**: Magic Formula 구현
- **서스펜션**: 스프링-댐퍼 동역학
- **차체**: Newton-Euler 방정식
- **E-corner**: 힘/모멘트 변환
- **제어기**: PID, Ackermann 조향
- **시나리오**: 입력 프로파일, 성능 평가

## 설계 원칙

1. **파라미터 분리**: 모든 파라미터는 YAML 파일에서 관리
2. **모듈화**: 각 서브시스템이 독립적으로 개발 가능
3. **확장성**: 새로운 제어기나 시나리오 추가 용이
4. **계층적 설계**: 유틸 → 모델 → 제어기 → 시나리오 → 시뮬레이터

## 의존성

```bash
pip install numpy scipy matplotlib pyyaml
```

## 참고 문헌

- Pacejka, H. B. (2012). Tire and Vehicle Dynamics
- Rajamani, R. (2011). Vehicle Dynamics and Control
- Gillespie, T. D. (1992). Fundamentals of Vehicle Dynamics
