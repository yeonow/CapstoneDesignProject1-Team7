---
name: Task
about: 프로젝트 작업을 등록하기 위한 템플릿입니다.
title: "[Task] RSSI 기반 Q-Learning 시뮬레이션 환경 구현"
labels: ""
assignees: ""
---

**## 📌 작업 내용**

Q-Learning 기반 실종자 탐색 알고리즘 학습을 위한 2D Grid Simulation 환경을 구현합니다.

Simulation 내부에서 Agent와 Target의 위치를 관리하고, Agent 이동, 거리 계산, 가상 RSSI 생성, Noise 적용, Step 관리 및 종료 조건을 처리합니다.

Agent와 Target의 좌표는 Simulation 내부의 Ground Truth로만 사용하며 Q-Learning의 State에는 포함하지 않습니다.


**## ✅ 세부 작업**

- [x] `RSSIGridEnv` 클래스 및 기본 설정 구조 구현
- [x] Agent / Target 위치 관리 기능 구현
- [x] `reset()`을 통한 Episode 초기화 기능 구현
- [x] UP / DOWN / LEFT / RIGHT 4방향 이동 기능 구현
- [x] Grid 경계 처리 기능 구현
- [x] Agent와 Target 사이 거리 계산 기능 구현
- [x] 거리 기반 가상 RSSI 생성 기능 구현
- [x] Simulation용 RSSI Noise 적용 기능 구현
- [x] 한 Step에서 생성된 RSSI 값 관리 기능 구현
- [x] Step 수 관리 기능 구현
- [x] Target 도달 및 최대 Step 종료 조건 구현
- [ ] 평가용 `get_info()` 기능 구현
- [ ] `environment.py` 단독 동작 테스트


**## 🎯 완료 조건**

- [x] Agent가 2D Grid 내부에서 정상적으로 이동할 수 있습니다.
- [x] Grid 범위를 벗어나는 이동이 정상적으로 처리됩니다.
- [x] Agent와 Target 사이 거리를 정상적으로 계산할 수 있습니다.
- [x] 거리 기반 가상 Raw RSSI와 Simulation용 Noise를 생성할 수 있습니다.
- [x] 한 번의 이동에서 생성된 동일한 RSSI 값을 State와 Reward 계산에 사용할 수 있습니다.
- [x] Target 도달 또는 최대 Step 도달 시 Episode가 정상적으로 종료됩니다.
- [x] Agent / Target 좌표가 Q-Learning State에 직접 전달되지 않습니다.
- [ ] `environment.py`를 단독으로 실행하여 주요 기능이 정상 동작하는지 최종 확인합니다.


**## 📎 참고 사항**

- Target은 사전에 지정된 스마트폰 Wi-Fi Hotspot으로 가정합니다.
- 1차 구현은 Python 기반 2D Grid Simulation으로 진행합니다.
- Action은 `0 = UP`, `1 = DOWN`, `2 = LEFT`, `3 = RIGHT`로 사용합니다.
- Agent / Target 좌표는 Simulation 내부의 이동 처리, 거리 계산 및 성공 여부 판단을 위한 Ground Truth로만 사용합니다.
- `environment.py`에서는 Raw RSSI 생성까지만 담당합니다.
- RSSI Level, RSSI Trend, State, Reward 계산은 `state_reward.py`에서 담당합니다.
- Q-table, ε-greedy, Q값 업데이트는 `q_learning.py`에서 담당합니다.
- 현재 RSSI 모델은 Log-distance Path Loss 기반의 Simulation용 임시 모델입니다.
- `reference_rssi`, `reference_distance`, `path_loss_exponent`, `noise_std`는 실제 Raspberry Pi 및 Wi-Fi Adapter를 이용한 RSSI 측정 이후 보정할 예정입니다.
- 한 Step에서 RSSI는 한 번만 생성하고 `current_rssi`에 저장하여 State와 Reward 계산에서 동일한 측정값을 사용할 수 있도록 합니다.
- RSSI Gradient 비교 알고리즘은 현재 구현 범위에서 제외합니다.