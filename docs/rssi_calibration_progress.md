# RSSI calibration progress

Base: origin/main 5b055fb6c9921b546ef1baec0acd38f0d726608f (2026-11-25).
Worktree: CapstoneDesignProject1-Team7-rssi; branch feature/rssi-realworld-calibration.

## Data policy
User-final scenario1_provisional (90) is primary; strict (89) is sensitivity only.
Scenario2 (24 readings/20 differences) is separate Trend validation. No 20m observations.
Original files copied byte-for-byte to data/rssi_calibration_20261125; SHA-256 checked.

## 현재 상태
- 데이터 검증, 감사 대조, 피팅/민감도, Trend, 환경 선택 및 실제 소규모 smoke 완료.
- 최종 테스트와 변경 범위 검토 완료. 마지막 절차는 새 브랜치 푸시다.
- 아래 checkpoint는 개발 이력이다. 최종 수치/한계/실행 방법은 문서 후반을 따른다.

Do not infer spatial shadow variance from temporal repeats.
Resume from this Worktree; inspect git log and this file before changing any source.
Each completed unit must pass relevant tests and be committed separately.
# Data validation checkpoint

- Baseline full suite: 357 passed. Loader commit: b8e7124.
- Added exact sample coverage, strict subset, movement ordering and selected audit reconciliation.
- Original bytes are immutable; SHA mismatch stops analysis. No low-quality sample is silently removed.

## Fit checkpoint
- Audit reconciliation committed as d1fc3d0 (8 tests passed).
- Reused existing distance_statistics/path_loss; added RSSI-space fit through equation (1).
- Fits aggregate P_rx at 1m and exponent with assumed noise floor -90 dBm; retains direct RSSI fit for comparison.
- Within-distance repeat SD is descriptive, not a spatial shadow-fading estimate. Runtime noise stays 2 dB.

## Environment checkpoint
- Fit commit 86d9c13: 3 tests passed, including known-curve recovery.
- Optional aggregate reference adds a constant offset to equation (2); baseline arithmetic and equation (1) remain unchanged.
- Movement, state/reward, seeds and target selection unchanged. No NLoS transfer inferred from LoS data.

## Factory checkpoint
- Environment commit 0fd213c: 24 baseline/calibrated tests passed.
- Explicit RSSI_MODEL=calibrated plus RSSI_CALIBRATION_FILE loads validated source-bound JSON.
- Both existing Q-Learning and Random factory calls use this setting; no learning code changed.
- LoS, 1m clamp and fitted noise-floor assumptions are checked; default remains baseline.

## Trend checkpoint
- Factory commit 4468203: 27 tests passed.
- Reuses existing ordered legs/transitions; 20 differences, 2 repeats, approach/away summaries.
- Explores 0/1/2/3/4 dB descriptively; keeps existing 2 dB without tuning State/Reward.

## Analysis checkpoint
- Trend commit 4b97cdf: transition and existing-threshold agreement test passed.
- Added read-only-source analysis command, before/after curves, strict sensitivity, trend CSVs and runtime profile.
- All outputs require a new directory; 20m explicitly unobserved/extrapolated. Seeded simulated SD is separate from measured repeat SD.
- 당시 남은 실제 분석/smoke/전체 테스트는 이후 checkpoint에서 완료했다.

## Smoke checkpoint
- Analysis commit 647c509: actual-data output and overwrite protection test passed.
- Actual 90-point fit: reference P_rx=-60.27895489260387 dBm, n=1.583968492442464.
- In-sample RSSI RMSE: baseline 15.36524060741104 dB, calibrated 4.4520562119044556 dB.
- Trend at 2 dB: 7/20 correct (35%); no threshold tuning.
- Added paired 1-run smoke wrapper with try/finally config restoration and metadata; no change to engine or seed rules.

## Measured profile checkpoint
- Smoke commit 7f2b34e: 3 tests passed; full suite afterward: 378 passed.
- Actual smoke: each mode 10 train / 10 eval, one run 42/43, 20x20 / 400 steps / Manhattan split.
- Baseline Q/Random: 1/10 and 6/10 successes; calibrated Q/Random: 2/10 and 6/10. These are debugging outcomes, not performance evidence.
- Saved the actual generated profile under data/rssi_calibration_20261125/calibration_profile.json; test recomputes it from the final 90 samples.
- Original worktree still has exactly 29 deletions and 3 untracked files; all three recorded SHA-256 hashes unchanged.

## 최종 분석: 관측값과 한계

원본 CSV 4개와 README는 `data/rssi_calibration_20261125/`에 있다.
파일명의 provisional은 과거 이름이며 사용자 확정 자료로 사용한다.
감사 기록은 130행, 선택 114행(90+24)이며 최종 측정값/좌표/순서와 일치한다.
선택 113개는 ok, 1개는 low_packet_count이다. strict는 유효 패킷 9개인 10m 표본만 제외한다.
선택 측정의 주파수는 2452 MHz이다. 원본 SHA는 로더 및 배포 프로필에 기록되어 있다.

| 거리(m) | 개수 | 평균 RSSI(dBm) | 반복 표본 SD(dB) |
|---:|---:|---:|---:|
| 1 | 15 | -62.733 | 4.523 |
| 2 | 15 | -66.433 | 4.242 |
| 3 | 15 | -62.667 | 2.690 |
| 5 | 15 | -69.700 | 2.520 |
| 10 | 15 | -77.200 | 2.136 |
| 15 | 15 | -80.333 | 5.119 |

3m 평균이 2m보다 강하므로 단조 감쇠로 모든 관측을 설명할 수 없다.
동일 거리 반복의 pooled SD는 3.715049 dB이다. 공간적 Shadow Fading이나 순간 패킷
변동의 추정치가 아니다. 관측값은 기존 수집 절차의 5초 구간 RSSI 중앙값이다.
독립 Gaussian `NOISE_STD=2.0`은 기존 가정으로 유지하며 시간/공간 상관은 보정하지 않았다.

### 식 (1)/(2)와 대응

식 (2)의 거리 기울기와 Gaussian 1회 생성을 유지하고 1m 절편에 상수 보정을 더한다.
등가 수식은 `P_rx(d)=A-10*n*log10(max(d,1m)/1m)+omega`다.
식 (1)은 `RSSI=10*log10(10^(P_rx/10)+10^(noise_floor/10))`로 유지한다.
**이 RSSI와 측정 RSSI의 제곱 오차**를 최소화하여 A와 n을 구했다.

| 지표 | 최종 90개 | strict 89개 |
|---|---:|---:|
| 종합 1m P_rx A(dBm) | -60.27895489260387 | -60.29101669195445 |
| 거리 감쇠 지수 n | 1.583968492442464 | 1.578005830534379 |
| 관측 RSSI 기준 RMSE(dB) | 4.452056211904456 | 4.471574832794272 |

최종 피팅 R²는 0.665147이다. strict의 n 차이는 약 -0.005963이다.
RMSE는 서로 다른 90개/89개 대상이므로 같은 검증셋 성능 차이로 해석하지 않는다.
기존 순수 RSSI log-distance 회귀도 보존했다: RSSI 절편 -60.299273 dBm,
n=1.567690, RMSE=4.431381 dB. 이 값은 식 (1)을 거치지 않는 별도 기술통계이며
런타임에 적용하는 n과 구분한다.

최종 90개에 대한 noise-free 예측 RMSE는 Baseline 15.365241 dB,
Calibrated 4.452056 dB이다. 모두 in-sample 적합도이며 지역/시간 일반화나 탐색 성능이 아니다.

- noise floor -90 dBm은 미측정 가정이며 보정 수치는 이 값에 조건부다.
- A는 송신 전력, 안테나/기기 효과, 추가 손실의 종합 기준이다.
  실제 송신 전력, eta, multipath를 각각 식별했다고 주장하지 않는다.
- 보정 프로필은 측정 2.452 GHz를 사용한다. Baseline 2.4 GHz 설정은 유지한다.
  종합 절편에 주파수 효과가 포함되므로 다른 주파수의 전이 성능을 검증한 것은 아니다.
- 프로필은 LoS/1m clamp/피팅 noise floor를 검사한다. NLoS 보정 자료는 없다.
- 실측은 1~15m뿐이다. 20m는 연결 단절로 미관측이며 그래프에서 외삽 표시한다.
  20x20 smoke의 먼 좌표(최대 약 26.9m)도 외삽이다. 단절/관측 불가 확률을 구현하지 않았으므로
  실제 해당 거리에서 신호를 관측할 수 있다는 의미가 아니다.
- 단순 회귀의 IID 신뢰구간은 반복의 독립성 가정에 의존하는 참고값으로만 보존한다.

### Trend: 기존 ±2 dB 유지

| 구간 | 접근 일치 | 이탈 일치 | 전체 일치 |
|---|---:|---:|---:|
| Repeat 1 | 1/5 | 2/5 | 3/10 |
| Repeat 2 | 2/5 | 2/5 | 4/10 |
| 합계 | 3/10 | 4/10 | 7/20 (35%) |

UP/KEEP/DOWN은 4/10/6개다. 기존 엄격한 `delta > threshold`, `< -threshold` 규칙을 유지한다.
탐색적 0/1/2/3/4 dB의 일치율은 60/40/35/30/25%다. 동일한 20개 종속 차분을 재사용하므로
0 dB를 최적값으로 선택하지 않는다. 이동 방향 일치율은 위치 정확도나 독립 분류 성능이 아니다.
scenario2의 절대 RSSI가 scenario1과 다르므로 거리 피팅에 합치지 않았다.

### 실제 Smoke 결과

양 모드 각각 새 Q-Table, 10회 학습/알고리즘당 10회 평가, 1 Run(42/43)만 실행했다.
20x20, (0,0), 400 Step, 8방향, LoS, 1m/cell, noise_std=2,
Manhattan 최소거리 5의 기존 2x2 split이다.
두 알고리즘·두 모드의 평가 Target/Seed 일치와 같은 Seed의 채널 재현성을 검증했다.

| 모델 | 알고리즘 | 성공 수 | 전체 평균 종료 Step |
|---|---|---:|---:|
| Baseline | Q-Learning | 1/10 | 374.5 |
| Baseline | Random | 6/10 | 210.4 |
| Calibrated | Q-Learning | 2/10 | 326.9 |
| Calibrated | Random | 6/10 | 210.4 |

통합 동작 확인용이며 보정으로 탐색 성능이 향상됐다는 결론을 내리지 않는다.
State/Reward, 학습률, 할인율, epsilon 감소, Target 함수와 Seed 규칙은 바꾸지 않았다.

## 실행 방법

분석 의존성은 기존 `experiments/field_measurement/analysis/requirements.txt`를 재사용한다.
이번 환경에는 이미 설치되어 있어 패키지를 설치하거나 requirements를 변경하지 않았다.
프로젝트 루트에서 실행한다. 출력 폴더가 이미 있으면 중단하므로 새 이름을 사용한다.

```powershell
python -m src.calibration_analysis --output results/rssi_calibration_review/analysis
python -m src.calibration_smoke --profile data/rssi_calibration_20261125/calibration_profile.json --output results/rssi_calibration_review/smoke
python -m pytest -q -p no:cacheprovider
```

실제 이번 실행의 경로는 `results/rssi_calibration_20261125/`다:

- `analysis/profile.json`, `report.json`: 출처 해시/품질, 피팅/strict/Trend 결과.
- `analysis/distance_statistics.csv`, `model_comparison.csv`: 실측 평균/SD, noise-free 예측,
  같은 거리/Seed로 500회 생성한 시뮬레이션 평균/SD. 후자의 SD는 가정한 2 dB에 기반한다.
- `analysis/trend_transitions.csv`, `trend_accuracy.csv`, `trend_threshold_exploration.csv`.
- `analysis/distance_model_comparison.png` 및 `.pdf`, `movement_trends.png` (PNG 300 dpi).
- `smoke/baseline/` 및 `smoke/calibrated/`: 각각 `algorithm_comparison_episodes.csv`,
  `algorithm_comparison_summary.csv`.
- `smoke/smoke_manifest.json`: config, 프로필/해시, Seed 규칙, 재현성 probe, 대응 Target 검증,
  예비 요약. config의 기본 물리값과 보정된 값을 구분하기 위해 프로필을 함께 읽는다.

기존 `.gitignore`에 따라 새 `results/` 산출물은 로컬에만 있다.
재현에 필요한 원본 CSV, 코드, 실제 보정 프로필은 커밋한다. 기존 논문 결과는 변경하지 않았다.

## 후속 연구에서 보정 선택

같은 Python 프로세스에서 학습 전에 지정한다. Baseline 복귀 시 RSSI_MODEL만 baseline으로 지정한다.

```python
import config
config.RSSI_MODEL = "calibrated"
config.RSSI_CALIBRATION_FILE = "data/rssi_calibration_20261125/calibration_profile.json"
# 이후 기존 compare_algorithms의 공유 build_environment가 보정 모델을 생성한다.
```

후속 연구는 프로필/CSV 해시, noise floor/NOISE_STD, Target 모드/거리 기준, 학습/평가 횟수,
Run Seed와 독립 출력 폴더를 함께 기록해야 한다. 전역 config/RNG를 사용하므로 동일 프로세스의
병렬 스레드 실행은 지원하지 않는다. 팀원 4명의 Run 분담과 100 Run은 별도 작업이다.
실수로 전체 실험을 실행하지 않도록 smoke는 1 Run, 최대 20 Episode로 제한한다.

## 커밋 인수인계

b8e7124 로더 → d1fc3d0 감사 대조 → 86d9c13 피팅 → 0fd213c 환경 →
4468203 팩토리 → 4b97cdf Trend → 647c509 분석/그래프 → 7f2b34e smoke →
da5f284 실제 배포 프로필. 각 커밋은 관련 테스트 후 저장했다.

ba463fa는 Git 자동 줄바꿈 변환에 따른 원본 SHA 불일치를 해결했다.
새 측정 CSV에만 `.gitattributes -text`를 적용하고 실제 Git blob 4개의 SHA가
사용자 원본과 같은지 검사했다. 파일 내용/관측값은 수정하지 않았다.
42142bd는 원본 CSV의 CRLF를 정상 줄 끝으로 인식하도록 해당 파일의 공백 검사만 보완했다.
실제 후행 공백 검사는 유지하며 `git diff --check origin/main...HEAD`가 통과한다.

작업 전 전체 357개, 기능 추가 후 전체 378개가 통과했다.
배포 프로필 재피팅 테스트 추가 후 최종 전체 테스트는 **379 passed (44.42s)**였다.
줄바꿈 보존 변경 후 데이터 테스트 8개와 Git blob SHA 4개 일치를 추가 확인했다.
기존 코드/원본을 다시 만들지 말고 이 Worktree에서 git status/log를 먼저 확인한다.
`git diff --check`와 main 대비 변경 범위를 검토했다. Q-Learning/State/Reward,
Target/Seed 규칙 및 기존 results 파일은 diff가 없다. 현재 남은 완료 절차는 새 브랜치 푸시다.
100 Run, 모든 Noise 조건 연구, 단절 모델/다른 지역 일반화는 이번 범위 밖이다.
