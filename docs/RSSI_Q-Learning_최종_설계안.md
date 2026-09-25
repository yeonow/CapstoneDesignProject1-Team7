# RSSI 기반 Q-Learning 실종자 탐색 알고리즘 최종 정리본

## 1. 1차 구현 목표

사전에 **Target으로 지정된 스마트폰 Wi-Fi Hotspot 신호**를 대상으로, Raspberry Pi가 수집한 RSSI를 이용해 신호원에 접근하는 탐색 알고리즘을 구현한다.

1차 구현에서는 다음에 초점을 둔다.

- Target 신호는 사전에 알고 있다고 가정
- 실제 스마트폰 좌표는 알고리즘에 제공하지 않음
- Q-Learning으로 다음 이동 방향 결정
- 실제 드론 대신 Raspberry Pi를 사람이 이동시키며 검증
- 먼저 Simulation에서 학습한 뒤 실제 RSSI 환경에서 검증
- 단순 RSSI Gradient 방식과 성능 비교

> 현재 목표는 “불특정 다수의 단말 중 실종자의 스마트폰을 식별하는 문제”보다, **이미 식별된 Target 신호를 이용해 신호원에 접근하는 탐색 알고리즘을 개발하고 검증하는 것**이다.

---

# 2. 전체 처리 흐름

```text
Target Wi-Fi Hotspot ON
        ↓
Raspberry Pi에서 Target RSSI 반복 측정
        ↓
RSSI 대표값 계산
        ↓
State 생성
(RSSI Level + RSSI Trend + Previous Action)
        ↓
Q-Learning
        ↓
Action 선택
(UP / DOWN / LEFT / RIGHT)
        ↓
이동
        ↓
RSSI 재측정
        ↓
Reward 계산
        ↓
Q-Table 업데이트
        ↓
다음 State
        ↓
반복
```

Simulation과 실제 실험 모두 가능한 한 같은 State 생성 방식을 사용한다.

---

# 3. RSSI 처리

RSSI는 한 번 측정한 순간값을 그대로 사용하지 않는다.

기본 처리 과정은 다음과 같다.

```text
같은 위치에서 RSSI 여러 번 측정
        ↓
대표 RSSI 계산
        ↓
이전 위치의 대표 RSSI와 비교
        ↓
RSSI Level + RSSI Trend 생성
```

## 대표값 후보

- 평균
- 중앙값
- 절사 평균
- 이동평균

정확히 어떤 방법을 사용할지는 **실제 Raspberry Pi RSSI 측정 결과를 확인한 뒤 결정**한다.

## 미검출

Target 신호가 탐지되지 않았을 때 임의의 매우 약한 RSSI 값으로 바꾸기보다, 필요하면 `미검출` 상태를 별도로 두는 방안을 검토한다.

---

# 4. Q-Learning 설계

## 4.1 State

1차 구현 State는 다음과 같이 한다.

```text
State =
RSSI Level
+ RSSI Trend
+ Previous Action
```

예시:

```text
RSSI Level      = 약함
RSSI Trend      = 증가
Previous Action = RIGHT
```

## RSSI Level

RSSI를 몇 개의 구간으로 이산화한다.

예시:

```text
0 = 매우 약함
1 = 약함
2 = 보통
3 = 강함
4 = 매우 강함
```

정확한 dBm 경계값은 아직 확정하지 않는다.

**실제 RSSI 데이터 분포를 측정한 뒤 기준값을 결정한다.**

## RSSI Trend

이전 위치의 대표 RSSI와 현재 위치의 대표 RSSI를 비교한다.

```text
-1 = 감소
 0 = 유지
+1 = 증가
```

작은 RSSI 변동을 모두 증가·감소로 처리하지 않기 위해 추후 임계값 `τ`를 둘 수 있다.

```text
ΔRSSI > +τ        → 증가
-τ ≤ ΔRSSI ≤ +τ  → 유지
ΔRSSI < -τ        → 감소
```

`τ` 역시 실제 RSSI 데이터를 확인한 뒤 결정한다.

## Previous Action

직전 이동 방향을 State에 포함한다.

예:

```text
Previous Action = RIGHT
RSSI Trend      = 증가
```

→ 오른쪽으로 이동한 뒤 RSSI가 증가했다는 의미

---

# 5. 좌표 `(x, y)`는 어떻게 사용할 것인가?

Simulation 환경은 Agent와 Target의 실제 위치를 알고 있어야 한다.

하지만 **Q-Learning State에는 현재 좌표와 Target 좌표를 넣지 않는다.**

```text
Simulation 내부
- Agent 좌표 사용
- Target 좌표 사용
- 이동 처리
- 성공 여부 평가

Q-Learning 입력
- RSSI Level
- RSSI Trend
- Previous Action
```

이렇게 분리하는 이유는 실제 Raspberry Pi 실험에서도 Target의 정확한 좌표를 알고리즘에 제공하지 않을 계획이기 때문이다.

Target 좌표는 **Simulation 및 평가용 Ground Truth**로만 사용한다.

---

# 6. Action

1차 구현은 4방향으로 시작한다.

```text
0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
```

처음부터 8방향을 사용하지 않고, 4방향 탐색이 정상적으로 동작하는지 먼저 확인한다.

필요하면 이후 다음과 같이 8방향으로 확장하여 비교한다.

```text
↖ ↑ ↗
←   →
↙ ↓ ↘
```

8방향이 반드시 더 좋은 것은 아니므로 실제 비교 실험으로 판단한다.

---

# 7. Reward

처음에는 Reward를 단순하게 구성한다.

## 1차 Reward

```text
RSSI가 의미 있게 증가 → 양의 보상
RSSI 변화가 작음      → 0
RSSI가 의미 있게 감소 → 음의 보상
이동                  → 작은 비용
```

개념적으로:

```text
Reward = RSSI 변화 보상 - 이동 비용
```

또는 초기 실험에서는 다음과 같은 단순 형태도 시험할 수 있다.

```text
Reward = RSSI_after - RSSI_before
```

## 처음부터 넣지 않을 요소

- 재방문 패널티
- 복잡한 경로 패널티
- 많은 보상 항목

재방문 문제가 실제로 나타난다면 그때 State에 방문 정보를 추가하거나 별도 규칙으로 처리한다.

여러 Reward 항목을 처음부터 동시에 추가하지 않고, **하나씩 추가하며 성능 변화를 확인**한다.

---

# 8. Action 선택

Q-Learning의 Action 선택은 `ε-greedy` 방식으로 한다.

```text
대부분의 경우
→ 현재 Q값이 가장 높은 Action 선택

일부 경우
→ 무작위 Action 선택
```

이를 통해 학습된 행동을 활용하면서도 새로운 방향을 탐색할 수 있다.

정확한 값은 현재 고정하지 않는다.

- 학습률 `α`
- 할인율 `γ`
- 탐험 비율 `ε`
- `ε` 감소 방식

위 값들은 Simulation 결과를 보면서 조정한다.

---

# 9. Simulation 환경

Python 기반 2D Grid 환경을 구현한다.

예:

```text
. . . . . . . .
. S . . . . . .
. . . . . . . .
. . . . . T . .
. . . . . . . .

S = Agent
T = Target
```

초기에는 작은 Grid에서 먼저 정상 작동 여부를 확인한다.

예:

```text
10 × 10
```

이후 필요하면 더 큰 Grid로 확장한다.

## Simulation 내부 정보

환경은 다음을 알고 있다.

- Agent 위치
- Target 위치
- Agent와 Target 사이 거리
- Grid 경계
- RSSI 모델
- Noise

그러나 Target 좌표는 Q-Learning State에 전달하지 않는다.

---

# 10. Simulation의 RSSI 모델

초기 Simulation에서는 거리 기반 RSSI 모델에 Noise를 추가한다.

개념적으로:

```text
RSSI(d) = RSSI(d0) - 10 × n × log10(d / d0) + noise
```

이 모델은 **실제 Wi-Fi 환경을 그대로 재현하는 것이 아니라 초기 Simulation을 만들기 위한 가정**이다.

실제 Raspberry Pi 데이터를 수집한 뒤 다음을 보정한다.

- 거리별 RSSI 변화
- 동일 위치의 RSSI 변동 폭
- Noise 크기
- 장애물에 따른 변화
- 방향에 따른 변화
- 미검출 가능성

---

# 11. Simulation과 Q-Learning 코드 인터페이스

두 파트는 따로 구현해도 되지만 아래 규칙은 반드시 일치해야 한다.

## State 형식

```text
(rssi_level, rssi_trend, previous_action)
```

## Action 번호

```text
0 = UP
1 = DOWN
2 = LEFT
3 = RIGHT
```

## 기본 인터페이스

```python
state = env.reset()

next_state, reward, done = env.step(action)
```

## `reset()`

새 Episode를 초기화한다.

예:

- Agent 시작 위치 설정
- Target 위치 설정
- 이전 RSSI 초기화
- Previous Action 초기화
- Step 수 초기화

## `step(action)`

```text
Action에 따라 Agent 이동
        ↓
새 RSSI 생성
        ↓
새 State 생성
        ↓
Reward 계산
        ↓
종료 여부 판단
        ↓
next_state, reward, done 반환
```

---

# 12. 학습과 평가 분리

## Training

- Q-Table 업데이트
- 여러 시작 위치와 Target 위치 사용
- `ε-greedy`로 탐색
- 반복 Episode 수행

## Validation

- State 구성 검토
- RSSI 구간 기준 검토
- Reward 조정
- `α`, `γ`, `ε` 조정

## Test

- 학습 중지
- Q-Table 고정
- 새로운 조건에서 성능만 측정

최종 평가용 Test 환경에 맞추어 다시 학습하지 않는다.

---

# 13. Raspberry Pi 실제 검증

학습은 주로 PC의 Simulation에서 수행한다.

학습 후 Q-Table을 저장하고 Raspberry Pi에서는 저장된 정책을 이용해 Action을 결정한다.

```text
Simulation에서 Q-Learning 학습
        ↓
Q-Table 저장
        ↓
Raspberry Pi에서 Target RSSI 측정
        ↓
Simulation과 동일한 방식으로 State 생성
        ↓
Q-Table에서 Action 선택
        ↓
사람이 Raspberry Pi 이동
        ↓
RSSI 재측정
        ↓
반복
```

Simulation과 실제 환경에서 **State 구성과 RSSI 이산화 기준을 동일하게 유지**한다.

---

# 14. 종료 조건

## Simulation

Simulation은 Target 위치를 알고 있으므로 다음 조건을 사용할 수 있다.

```text
Target 위치 또는 허용 반경 도달
→ 성공 종료

최대 Step 초과
→ 실패 종료
```

## 실제 환경

실제 환경에서는 Target의 정확한 좌표를 알고리즘이 모르므로 별도 판단이 필요하다.

후보:

```text
RSSI가 일정 기준 이상으로 반복 관측
```

또는

```text
여러 번 측정했을 때 강한 RSSI가 안정적으로 유지
```

정확한 기준은 실제 RSSI 측정 이후 정한다.

따라서 다음을 구분한다.

```text
Simulation
→ 실제 Target 도달 여부 평가 가능

실제 환경
→ RSSI 기반 위치 후보 판정
```

강한 RSSI 한 번만으로 실제 실종자 발견이라고 판단하지 않는다.

---

# 15. 비교 알고리즘

Q-Learning의 효과를 검증하기 위해 최소한 RSSI Gradient 방식과 비교한다.

## RSSI Gradient Search

현재 RSSI 하나만으로 방향을 알 수 있는 것은 아니다.

따라서 최근 이동 결과 또는 주변 방향 탐색을 이용해 RSSI가 증가하는 방향을 찾는다.

예:

```text
RSSI 측정
    ↓
한 방향 이동
    ↓
RSSI 재측정
    ↓
증가하면 해당 방향 유지
감소하면 다른 방향 탐색
```

이 방식은 Q-Learning과 비교하기 위한 Baseline으로 사용한다.

가능하면 Random Search도 추가한다.

```text
Random Search
      VS
RSSI Gradient Search
      VS
Q-Learning
```

---

# 16. 평가 지표

최소 평가 지표:

| 지표 | 의미 |
| --- | --- |
| 탐색 성공률 | 제한 시간 또는 Step 안에 Target 허용 범위에 도달한 비율 |
| 이동 횟수 | Target 접근까지 필요한 Step 수 |
| 이동 거리 | 실제 이동한 전체 거리 |
| 탐색 시간 | RSSI 측정과 이동을 포함한 총 시간 |
| RSSI 측정 횟수 | 탐색 과정에서 필요한 총 측정 횟수 |
| 잘못된 위치 후보 | Target이 아닌 위치를 후보로 판단한 횟수 |
| Noise 강건성 | RSSI 변동이 커질 때 성능이 얼마나 감소하는지 |
| 이동 경로 | Agent가 실제로 이동한 경로 |

성공한 경우의 이동 횟수만 비교하지 않고 **성공률과 실패 사례를 함께 기록**한다.

---

# 17. 실험 비교 조건

모든 알고리즘은 가능한 한 같은 조건에서 평가한다.

- 동일한 시작 위치
- 동일한 Target 위치
- 동일한 Grid
- 동일한 RSSI Noise 조건
- 동일한 최대 Step
- 동일한 RSSI 측정 방식

Q-Learning과 Gradient 방식의 측정 횟수가 다르다면 그 차이도 탐색 비용에 포함한다.

---

# 18. 단계별 구현 순서

```text
1. Target Wi-Fi Hotspot 설정
        ↓
2. Raspberry Pi RSSI 측정 코드 준비
        ↓
3. RSSI 기초 측정
        ↓
4. 2D Grid Simulation 구축
        ↓
5. RSSI Gradient Baseline 구현
        ↓
6. 4방향 Tabular Q-Learning 구현
        ↓
7. Simulation에서 학습 및 비교
        ↓
8. 실측 RSSI를 이용해 Simulation 보정
        ↓
9. Q-Learning 재학습 및 평가
        ↓
10. Raspberry Pi 실제 환경 검증
        ↓
11. 필요할 경우 State / Reward / 8방향 확장
```

RSSI 실측과 Simulation 개발은 가능한 범위에서 병렬로 진행할 수 있다.

---

# 19. 현재 팀이 먼저 맞춰야 할 것

## 팀 전체 합의

- Target은 사전에 지정된 Wi-Fi Hotspot으로 설정
- Q-Learning을 1차 알고리즘으로 사용
- State 기본 구조
- Action 기본 구조
- Reward 기본 원리
- Baseline
- 평가 지표

## Q-Learning 담당 + Simulation 담당

다음 세부 인터페이스를 맞춘다.

```text
State 자료형
Action 번호
reset() 반환값
step(action) 반환값
Reward 전달 방식
done 조건
```

## RSSI 측정 담당과 공유할 것

- RSSI 기록 형식
- 한 위치의 측정 방식
- 거리별 실측 데이터
- Noise 및 변동성
- 미검출 처리 방식

---

# 20. 현재 1차 구현 최종안

| 항목 | 최종 1차안 |
| --- | --- |
| Target | 사전에 지정된 스마트폰 Wi-Fi Hotspot |
| 알고리즘 | **Tabular Q-Learning** |
| RSSI 입력 | 동일 위치에서 여러 번 측정한 대표값 |
| State | **RSSI Level + RSSI Trend + Previous Action** |
| 좌표 | Simulation 내부 및 평가용으로만 사용 |
| Action | **UP / DOWN / LEFT / RIGHT 4방향** |
| Reward | **RSSI 변화 보상 + 작은 이동 비용** |
| Action 선택 | **ε-greedy** |
| Simulation | Python 2D Grid |
| RSSI 모델 | 거리 기반 모델 + Noise → 실측 RSSI로 보정 |
| 학습 | PC Simulation |
| 실제 적용 | 저장된 Q-Table을 Raspberry Pi에서 사용 |
| Baseline | **RSSI Gradient Search** |
| 추가 Baseline | 가능하면 Random Search |
| 평가 | 성공률, 이동 횟수·거리, 탐색 시간, 측정 횟수, Noise 강건성 |

---

# 21. 아직 확정하지 않는 항목

실제 데이터와 1차 구현 결과를 본 뒤 결정한다.

- RSSI Level의 정확한 dBm 구간
- RSSI Trend 임계값 `τ`
- 한 위치에서 RSSI를 몇 번 측정할지
- RSSI 대표값 계산 방식
- `α`
- `γ`
- `ε`
- `ε` 감소 방식
- 이동 비용 크기
- 최대 Step
- 실제 환경 종료 RSSI 기준
- 재방문 패널티 추가 여부
- State에 추가 이력을 넣을지 여부
- 8방향 확장 여부

---

# 22. 현재 연구 질문

> **RSSI 변동성이 존재하는 환경에서 Q-Learning 기반 탐색이 단순 RSSI Gradient 탐색보다 안정적으로 목표 신호원에 접근할 수 있는가?**

Q-Learning을 사용했다는 사실 자체보다, **RSSI 변동 환경에서 어떤 State와 Reward 구성이 적절한지, 그리고 실제 Raspberry Pi 환경에서도 Simulation에서 학습한 탐색 정책이 유효한지**를 실험적으로 검증하는 방향으로 진행한다.
