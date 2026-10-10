# Gaussian Noise 수준에 따른 RSSI Q-Learning 성능 분석

## 1. 목적

RSSI 측정에 추가되는 Gaussian Noise의 표준편차를 `0`, `5`, `10 dBm`으로
변경하면서 Q-Learning 탐색 정책의 성공률, 이동 효율, 누적 보상 변화를 비교했다.
단일 seed의 우연한 결과를 줄이기 위해 각 Noise 수준에서 5개 Random Seed를
독립적으로 실행하고 평균과 표본 표준편차를 계산했다.

## 2. 실험 조건

| 항목 | 값 |
|---|---:|
| Gaussian Noise 표준편차 | 0, 5, 10 dBm |
| Random Seed | 42, 123, 2026, 7777, 10000 |
| Seed별 학습 Episode | 1000 |
| Seed별 평가 Episode | 100 |
| Grid / 최대 Step | 10×10 / 100 |
| 시작 / Target | (0, 0) / (9, 9) |
| 평가 정책 | `training=False` greedy policy |

각 Noise·Seed 조합은 새 환경과 새 Agent로 시작한다. 환경의 Python 전역 RNG와
Agent의 독립 RNG를 같은 seed로 초기화한다. 평가는 학습된 동일 Agent를 사용하며,
별도 환경 RNG를 `seed + 1`로 초기화한다. 평가 중 Q-table 업데이트와 epsilon
decay는 수행하지 않는다.

## 3. 측정 지표

- 성공률
- 전체 Episode 평균 Step 수
- 성공 Episode 평균 Step 수
- 평균 누적 Reward
- Episode 종료 시 Target까지 남은 평균 직선거리
- 평가 실패 횟수

현재 환경은 실제 누적 이동 경로 길이를 기록하지 않는다. 따라서 `steps`는 경계에
막힌 Action까지 포함한 행동 횟수이고, `final_distance`는 이동한 거리가 아니라
Episode 종료 시 Target까지 남은 Euclidean Distance다.

## 4. Noise별 집계 결과

아래 값은 5개 seed 결과의 `평균 ± 표본 표준편차`다.

| Noise std | Train 성공률 | Train 평균 Step | Train 평균 Reward | Eval 성공률 | Eval 평균 Step | Eval 평균 Reward |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9506 ± 0.0340 | 28.45 ± 2.63 | 9.68 ± 0.75 | 1.000 ± 0.000 | 18.60 ± 0.55 | 11.34 ± 0.48 |
| 5 | 0.9612 ± 0.0036 | 29.87 ± 0.66 | 8.86 ± 0.09 | 1.000 ± 0.000 | 22.79 ± 5.81 | 10.10 ± 0.48 |
| 10 | 0.9444 ± 0.0115 | 31.58 ± 0.86 | 7.41 ± 0.18 | 1.000 ± 0.000 | 21.09 ± 1.44 | 9.16 ± 0.33 |

## 5. 해석

### 성공률

평가 성공률은 모든 Noise 조건에서 100%였다. 현재 설정에서는 100 Step이라는 충분한
탐색 한도와 1000회 학습 때문에 성공률이 포화되어, Noise 영향을 구분하는 지표로는
민감도가 낮았다. 학습 중 성공률도 94.4~96.1% 범위로 비교적 안정적이었다.

### 이동 효율

Noise 0에서 평가 평균 Step은 18.60이었고 Noise 5에서는 22.79, Noise 10에서는
21.09였다. Noise가 있는 조건은 Noise 0보다 각각 약 22.5%, 13.4% 더 많은 Action을
사용했다. 완전한 단조 증가는 아니지만 RSSI 관측 Noise가 방향 판단 효율을 낮춘다는
경향을 보여준다.

Noise 5의 평가 Step 표준편차는 5.81로 가장 컸다. Seed 10000에서 평균 33.12 Step이
필요했던 것이 주요 원인으로, 중간 수준 Noise에서도 학습된 정책의 seed 민감도가
커질 수 있음을 보여준다.

### 누적 Reward

평가 평균 Reward는 Noise가 커질수록 `11.34 → 10.10 → 9.16`으로 감소했다.
Noise 0 대비 Noise 5는 약 10.9%, Noise 10은 약 19.3% 낮다. 학습 평균 Reward도
`9.68 → 8.86 → 7.41`로 같은 방향을 보였다. RSSI Trend가 Noise에 의해 뒤집히거나
KEEP으로 판정되면서 긍정 보상이 줄고 추가 이동 비용이 누적된 결과로 해석할 수 있다.

### 거리 지표

평가 Episode는 모두 성공했기 때문에 평가 final distance는 모든 조건에서 0이었다.
학습 final distance는 실패 Episode의 잔여거리를 반영하지만 Noise에 따라 단조롭게
변하지 않았다. 실제 이동 경로 길이가 필요하면 환경에 누적 이동 거리 또는 위치
trajectory 기록을 별도로 추가해야 한다.

## 6. 결론과 제한점

- Noise 0~10 dBm 범위에서 최종 성공률은 유지됐지만, Noise가 있을 때 평균 이동
  Step이 늘고 누적 Reward가 감소해 탐색 효율 저하가 관찰됐다.
- 성공률이 포화되어 있으므로 향후에는 최대 Step을 줄이거나 더 큰 Grid, 무작위 Target,
  더 높은 Noise 조건을 사용하면 정책 강건성 차이를 더 분명하게 볼 수 있다.
- 5개 seed는 단일 seed보다 신뢰도가 높지만 통계적 유의성을 확정하기에는 작다.
  필요하면 seed 수를 20개 이상으로 늘리고 신뢰구간 또는 비모수 검정을 추가해야 한다.
- 결과는 현재 임시 path-loss 모델과 Reward 정의에 대한 시뮬레이션 결과이며 실제 RSSI
  측정 환경으로 일반화하려면 Raspberry Pi/Wi-Fi 실측 파라미터 보정이 필요하다.

## 7. 산출물과 재실행

- Seed별 결과: [`results/noise_sweep_runs.csv`](../results/noise_sweep_runs.csv)
- Noise별 집계: [`results/noise_sweep_summary.csv`](../results/noise_sweep_summary.csv)
- 비교 그래프: [`results/noise_sweep_summary.png`](../results/noise_sweep_summary.png)

```bash
python3 -m src.experiment_noise_sweep \
  --noise-stds 0 5 10 \
  --seeds 42 123 2026 7777 10000 \
  --train-episodes 1000 \
  --eval-episodes 100 \
  --output-dir results

MPLCONFIGDIR=/tmp/codex-matplotlib \
python3 -m src.plot_noise_sweep \
  --input results/noise_sweep_summary.csv \
  --output results/noise_sweep_summary.png
```
