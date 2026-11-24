# 논문용 실험 결과 분석

## 데이터 검증

- algorithm: 검증 통과, 분석 가능
  - Episode 200행, Run 2행 검증 완료.
- noise: 검증 통과, 분석 가능
  - Random의 성공·Step·최종 거리 독립성은 검사하지만, CSV에 경로·RSSI가 없어 경로 및 측정값 자체는 재검증할 수 없습니다.
  - Episode 3000행, Run 30행 검증 완료.

원본 CSV의 필수 열, 자료형, 결측값, 중복, Episode 수, 성공·실패 합, 거리, Seed, 지표 재계산 및 Noise Summary를 검사했다.
오류가 있는 행만 선택적으로 제거하거나 결과를 보정하지 않았다. 검증 통과는 CSV 내부 정합성을 뜻하며 실제 실행 이력의 인증은 아니다.

## Experimental Setup

다음은 **현재 config.py**에서 확인한 값이다. 원본 CSV에는 전체 환경 스냅샷이 없으므로 생성 당시 설정과 동일함을 입증할 수 없다.
- Grid: 10×10; 시작 위치: (0, 0); Target: (9, 9); 최대 Step: 100.
- RSSI 모델: Log-distance path loss; 기준 RSSI -40.0, 기준 거리 1.0, 경로 손실 지수 2.0; 현재 기본 Noise SD 2.0 dB.
- 현재 실행 코드: Q-Learning은 Noise별 새 Q-Table 학습 후 training=False Greedy 평가, Random은 RSSI와 무관한 4방향 균등 선택이다.
- 학습/평가 환경은 분리되며 평가 Noise Seed는 '<eval_seed>:<episode>', Action RNG는 독립적이다. 과거 경로·RSSI·Q-Table은 CSV에 저장되어 있지 않다.
- CSV algorithm Run/train_seed/eval_seed: [(1, 42, 43)].
- CSV algorithm algorithm/train_episodes/eval_episodes: [('q_learning', 500, 100), ('random', 0, 100)].
- CSV noise Run/train_seed/eval_seed: [(1, 42, 43), (2, 125, 126), (3, 2030, 2031), (4, 7783, 7784), (5, 10008, 10009)].
- CSV noise algorithm/train_episodes/eval_episodes: [('q_learning', 1000, 100), ('random', 0, 100)].

## Evaluation Metrics

Success Rate는 성공 횟수/평가 Episode 수이며 표에는 %로 표시한다. Average Steps는 실패와 경계에 막힌 시도를 포함한 실제 Step의 평균이다.
Average Steps on Success는 성공 Episode만의 평균이며 성공이 없는 Run은 결측값이다. Average Final Distance는 종료 시 Ground Truth 거리의 평균이다.
Success/Failure Count는 실제 Episode 수다. Reward는 알고리즘 간 비교 지표에서 제외한다.

## Overall Performance Comparison

| Noise (dB) | Algorithm | Runs | Success / Failure | success_rate_percent | average_steps | average_steps_on_success | average_final_distance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NA | q_learning | 1 | 100 / 0 | 100.0000 ± NA; 95% CI [NA, NA]; n=1 | 18.8300 ± NA; 95% CI [NA, NA]; n=1 | 18.8300 ± NA; 95% CI [NA, NA]; n=1 | 0.0000 ± NA; 95% CI [NA, NA]; n=1 |
| NA | random | 1 | 6 / 94 | 6.0000 ± NA; 95% CI [NA, NA]; n=1 | 98.4800 ± NA; 95% CI [NA, NA]; n=1 | 74.6667 ± NA; 95% CI [NA, NA]; n=1 | 7.1315 ± NA; 95% CI [NA, NA]; n=1 |

Mean ± sample SD; 95% Student's t CI across independent runs. NA = unavailable. Success rate: percent; steps: attempted moves; distance: grid units. Success/failure counts are episode totals. Successful-step metrics exclude runs with no successes. Noise NA denotes the basic comparison, whose historical noise is not recorded in its CSV.

기본 비교의 성공률 차이(Q−Random)는 94.000%p, 평균 이동 횟수 차이(Random−Q)는 79.650 Steps, 평균 최종 거리 차이(Random−Q)는 7.131 grid units다.
기본 비교는 독립 반복 Run이 부족하므로 위 관측 차이에 대한 표본 표준편차나 95% CI를 계산하지 않는다.

## Performance Under Different Noise Levels

| Noise (dB) | Algorithm | Runs | Success / Failure | success_rate_percent | average_steps | average_steps_on_success | average_final_distance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | q_learning | 5 | 500 / 0 | 100.0000 ± 0.0000; 95% CI [100.0000, 100.0000]; n=5 | 18.2000 ± 0.4472; 95% CI [17.6447, 18.7553]; n=5 | 18.2000 ± 0.4472; 95% CI [17.6447, 18.7553]; n=5 | 0.0000 ± 0.0000; 95% CI [0.0000, 0.0000]; n=5 |
| 0.0000 | random | 5 | 29 / 471 | 5.8000 ± 1.9235; 95% CI [3.4116, 8.1884]; n=5 | 98.5520 ± 0.6094; 95% CI [97.7953, 99.3087]; n=5 | 75.0898 ± 4.9196; 95% CI [68.9812, 81.1983]; n=5 | 7.1733 ± 0.1230; 95% CI [7.0205, 7.3260]; n=5 |
| 5.0000 | q_learning | 5 | 500 / 0 | 100.0000 ± 0.0000; 95% CI [100.0000, 100.0000]; n=5 | 23.3800 ± 3.9433; 95% CI [18.4837, 28.2763]; n=5 | 23.3800 ± 3.9433; 95% CI [18.4837, 28.2763]; n=5 | 0.0000 ± 0.0000; 95% CI [0.0000, 0.0000]; n=5 |
| 5.0000 | random | 5 | 29 / 471 | 5.8000 ± 1.9235; 95% CI [3.4116, 8.1884]; n=5 | 98.5520 ± 0.6094; 95% CI [97.7953, 99.3087]; n=5 | 75.0898 ± 4.9196; 95% CI [68.9812, 81.1983]; n=5 | 7.1733 ± 0.1230; 95% CI [7.0205, 7.3260]; n=5 |
| 10.0000 | q_learning | 5 | 500 / 0 | 100.0000 ± 0.0000; 95% CI [100.0000, 100.0000]; n=5 | 20.7120 ± 1.5031; 95% CI [18.8456, 22.5784]; n=5 | 20.7120 ± 1.5031; 95% CI [18.8456, 22.5784]; n=5 | 0.0000 ± 0.0000; 95% CI [0.0000, 0.0000]; n=5 |
| 10.0000 | random | 5 | 29 / 471 | 5.8000 ± 1.9235; 95% CI [3.4116, 8.1884]; n=5 | 98.5520 ± 0.6094; 95% CI [97.7953, 99.3087]; n=5 | 75.0898 ± 4.9196; 95% CI [68.9812, 81.1983]; n=5 | 7.1733 ± 0.1230; 95% CI [7.0205, 7.3260]; n=5 |

Mean ± sample SD; 95% Student's t CI across independent runs. NA = unavailable. Success rate: percent; steps: attempted moves; distance: grid units. Success/failure counts are episode totals. Successful-step metrics exclude runs with no successes. Noise NA denotes the basic comparison, whose historical noise is not recorded in its CSV.

관측된 Q-Learning 결과: Noise 0 dB: 성공률 100.000%, 평균 Step 18.200; Noise 5 dB: 성공률 100.000%, 평균 Step 23.380; Noise 10 dB: 성공률 100.000%, 평균 Step 20.712.
평균 Step은 Noise 순서에 따라 단조롭게 변하지 않았다. Noise 증가가 성능 저하를 일으킨다고 단정할 수 없다.
검증된 Random의 Episode별 성공·Step·최종 거리는 Noise 간 같았다. 경로 자체와 RSSI 값은 CSV만으로 확인할 수 없다.

## Statistical Analysis

| Dataset | Noise (dB) | Metric | Unit | Valid pairs | Q - Random | Sample SD | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| algorithm | NA | success_rate | percentage points | 1 | 94.0000 | NA | [NA, NA] |
| algorithm | NA | average_steps | steps | 1 | -79.6500 | NA | [NA, NA] |
| algorithm | NA | average_steps_on_success | steps | 1 | -55.8367 | NA | [NA, NA] |
| algorithm | NA | average_final_distance | grid units | 1 | -7.1315 | NA | [NA, NA] |
| algorithm | NA | success_count | episodes | 1 | 94.0000 | NA | [NA, NA] |
| algorithm | NA | failure_count | episodes | 1 | -94.0000 | NA | [NA, NA] |
| noise | 0.0000 | success_rate | percentage points | 5 | 94.2000 | 1.9235 | [91.8116, 96.5884] |
| noise | 0.0000 | average_steps | steps | 5 | -80.3520 | 0.7793 | [-81.3197, -79.3843] |
| noise | 0.0000 | average_steps_on_success | steps | 5 | -56.8898 | 4.9613 | [-63.0500, -50.7295] |
| noise | 0.0000 | average_final_distance | grid units | 5 | -7.1733 | 0.1230 | [-7.3260, -7.0205] |
| noise | 0.0000 | success_count | episodes | 5 | 94.2000 | 1.9235 | [91.8116, 96.5884] |
| noise | 0.0000 | failure_count | episodes | 5 | -94.2000 | 1.9235 | [-96.5884, -91.8116] |
| noise | 5.0000 | success_rate | percentage points | 5 | 94.2000 | 1.9235 | [91.8116, 96.5884] |
| noise | 5.0000 | average_steps | steps | 5 | -75.1720 | 4.4407 | [-80.6858, -69.6582] |
| noise | 5.0000 | average_steps_on_success | steps | 5 | -51.7098 | 7.0118 | [-60.4161, -43.0035] |
| noise | 5.0000 | average_final_distance | grid units | 5 | -7.1733 | 0.1230 | [-7.3260, -7.0205] |
| noise | 5.0000 | success_count | episodes | 5 | 94.2000 | 1.9235 | [91.8116, 96.5884] |
| noise | 5.0000 | failure_count | episodes | 5 | -94.2000 | 1.9235 | [-96.5884, -91.8116] |
| noise | 10.0000 | success_rate | percentage points | 5 | 94.2000 | 1.9235 | [91.8116, 96.5884] |
| noise | 10.0000 | average_steps | steps | 5 | -77.8400 | 1.6257 | [-79.8585, -75.8215] |
| noise | 10.0000 | average_steps_on_success | steps | 5 | -54.3778 | 5.2623 | [-60.9118, -47.8437] |
| noise | 10.0000 | average_final_distance | grid units | 5 | -7.1733 | 0.1230 | [-7.3260, -7.0205] |
| noise | 10.0000 | success_count | episodes | 5 | 94.2000 | 1.9235 | [91.8116, 96.5884] |
| noise | 10.0000 | failure_count | episodes | 5 | -94.2000 | 1.9235 | [-96.5884, -91.8116] |

All differences = Q-Learning - Random. Positive success-rate differences favor Q-Learning; negative step/distance differences favor Q-Learning. NA = unavailable; fewer than two pairs have no SD/CI.

모든 통계의 독립 표본 단위는 학습을 새로 수행한 Run이다. Episode 수를 독립 Run 수로 부풀리지 않았다.
평균 신뢰구간은 mean ± t(0.975, n−1) × sample_SD / sqrt(n)이다. 지표별 유효 Run 수를 기록하며 n<2이면 SD와 CI는 결측값이다.
Issue #4 그래프의 오차 막대는 SD이고, 이번 논문용 그래프는 95% CI다. 원본의 단일 Run SD=0 저장 관례는 바꾸지 않고 분석 출력만 결측값으로 처리했다.
대응 차이는 같은 Run·Seed·평가 수를 확인한 쌍의 Q−Random이다. 성공률은 percentage points, Step은 steps, 거리는 grid units이다.
평균 Step은 실패 포함 성공률과 함께 해석해야 하며, 성공 시 Step은 서로 다른 성공 Episode 집합의 조건부 평균이므로 성공 횟수와 함께 읽어야 한다.
Run 간 독립성과 Run 평균의 정규성은 적은 Run으로 충분히 검증할 수 없다. 모든 Run의 성공률이 같으면 t CI 폭은 0이지만 미래 실패 확률이 0이라는 뜻은 아니다.
신뢰구간은 물리적 범위를 벗어나더라도 잘라내지 않았다. 통계적 유의성 검정을 수행하지 않았고 유의하다는 결론을 내리지 않는다.
방법 출처: [NIST: Confidence Limits for the Mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm), [NIST: t Distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm).

## Limitations

- 현재 코드의 Grid·시작·Target 위치는 고정되어 있으며 공간 조건 변화는 검증하지 않았다.
- RSSI는 시뮬레이션 모델이며 Gaussian Noise는 실제 신호의 상관성·차폐·다중경로를 단순화한다.
- Random은 RSSI를 활용하지 않으므로 Noise별 동일 결과는 정책 특성과 일치한다.
- Noise별로 새로 학습한 정책의 성능이며, 한 Noise에서 학습한 정책의 다른 Noise 일반화 실험이 아니다.
- 기본 비교와 Noise 비교는 학습 횟수가 다를 수 있으므로 두 실험의 수치를 같은 학습 조건으로 합치지 않는다.
- 반복 Run이 적거나 성공 Run이 부족하면 불확실성·결측 선택의 영향이 크다. 여러 Noise 지점은 같은 Seed를 공유하므로 서로 독립인 표본으로 합치지 않는다.
- 원본 CSV에 전체 환경 설정·코드 버전·실행 시각·경로가 없어 과거 환경 일치와 완전한 재현을 CSV만으로 보증할 수 없다.

## Conclusion

관측된 차이는 위 표의 수치와 대응 차이에 한정한다. 수치의 방향을 실제 데이터로 확인하며 사전에 특정 알고리즘의 우위를 가정하지 않는다.
환경 메타데이터 보존, 더 많은 독립 Seed, 다른 공간 조건 및 학습/평가 Noise를 교차한 실험이 추가 검증에 필요하다.

## 원본 데이터 출처

- `C:\Users\LG\Documents\종프1\CapstoneDesignProject1-Team7\CapstoneDesignProject1-Team7\results\algorithm_comparison_episodes.csv`: SHA-256 `56e8395b2b1f2380854d78f1737acb43b245c497a79aee168b0ff82d1380c40e`
- `C:\Users\LG\Documents\종프1\CapstoneDesignProject1-Team7\CapstoneDesignProject1-Team7\results\algorithm_comparison_summary.csv`: SHA-256 `7fb3ac69433de6d52eaeebdd1f0f8e95e42726742beb2977c0e58114aaca5382`
- `C:\Users\LG\Documents\종프1\CapstoneDesignProject1-Team7\CapstoneDesignProject1-Team7\results\noise_comparison_episodes.csv`: SHA-256 `37517dda9455f67c019578d0dac34cdbe6d0209af3c719e02b3ee6f50bd5f539`
- `C:\Users\LG\Documents\종프1\CapstoneDesignProject1-Team7\CapstoneDesignProject1-Team7\results\noise_comparison_runs.csv`: SHA-256 `294e16fec84094caffe8889d4810949b175139f6af9def4a34f0ea6c9602d2fa`
- `C:\Users\LG\Documents\종프1\CapstoneDesignProject1-Team7\CapstoneDesignProject1-Team7\results\noise_comparison_summary.csv`: SHA-256 `3cbafc9612ed0b12b902d93294f4ab733febbc85e9d993376d45776898bc6ef7`
