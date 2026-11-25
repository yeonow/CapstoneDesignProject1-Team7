# RSSI calibration progress

Base: origin/main 5b055fb6c9921b546ef1baec0acd38f0d726608f (2026-11-25).
Worktree: CapstoneDesignProject1-Team7-rssi; branch feature/rssi-realworld-calibration.

## Data policy
User-final scenario1_provisional (90) is primary; strict (89) is sensitivity only.
Scenario2 (24 readings/20 differences) is separate Trend validation. No 20m observations.
Original files copied byte-for-byte to data/rssi_calibration_20261125; SHA-256 checked.

## 현재 상태
- 데이터 검증, 감사 대조, 피팅/민감도, Trend, 환경 선택 및 실제 소규모 smoke 완료.
- 최종 테스트와 변경 범위 검토 완료. 새 브랜치의 원격 푸시도 성공했다.
- 아래 checkpoint는 개발 이력이다. 최종 수치/한계/실행 방법은 문서 후반을 따른다.
- 후속 작업: smoke의 Target 후보 거리를 최종 논문 기준인 Chebyshev로 정정했다.
  Manhattan으로 기록된 smoke 내용은 정정 전 이력이며, 현재 결과는 "Chebyshev Target 기준 정정" 절을 따른다.
- 2차 후속: 모든 8방향 실험의 Target 후보 기준을 Chebyshev로 통일했다("8방향 실험 Target 거리 기준 통일" 절).

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

> 정정 전 기록: 아래는 Manhattan 최소거리 조건으로 실행한 과거 smoke이며 수치를 그대로 보존한다.
> 현재 설정(Chebyshev)의 결과는 "Chebyshev Target 기준 정정" 절에 별도로 기록했다.

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

## Chebyshev Target 기준 정정 (후속 작업)

최종 논문의 Target 후보 조건은 시작점 (0,0)에서 **Chebyshev 거리 ≥ 5**다.
smoke에 `TARGET_DISTANCE_METRIC="manhattan"`이 남아 있어 `src/calibration_smoke.py`의 이 값만
`"chebyshev"`로 바꿨다. `src/targets.py`의 거리 함수, 2x2 체커보드 분할, Target RNG/Seed 규칙,
다른 실험의 Manhattan 설정은 변경하지 않았다.
Target 후보 선정에는 Chebyshev 거리를 쓰고, RSSI 거리 감쇠에는 기존과 같이 Agent–Target Euclidean 거리를 쓴다.

### Target 분할 검증 (20x20, split)

| 거리 기준 | 전체 후보 | 학습 | 평가 | 최소 Chebyshev |
|---|---:|---:|---:|---:|
| Chebyshev ≥ 5 (현재) | 375 | 187 | 188 | 5 |
| Manhattan ≥ 5 (정정 전 smoke) | 385 | 193 | 192 | 3 |

Chebyshev 기준에서는 시작점 주변 5x5(25칸)만 제외된다. 학습/평가 중복 없음, 동일 Seed 재현,
Q-Learning·Random 및 Baseline·Calibrated의 평가 Target/Seed 동일성을
`tests/test_calibration_targets.py`에 테스트로 고정했다.

### 실측 범위와 외삽 구간

| 거리 구간 | 근거 | 해석 |
|---|---|---|
| 1m 미만 | 1m clamp | 1m 값을 사용, 근거리는 측정하지 않음 |
| 1–15m | 실측 90개 피팅 | 관측 범위 안의 회귀값 (in-sample) |
| 15m 초과–약 26.9m | 같은 회귀식 외삽 | 관측 없는 예측값, 20x20 대각 (19,19)까지 |

20m는 핫스팟 연결 단절로 측정하지 못했다. 기존 회귀식을 다시 피팅하지 않고 계산한 20m noise-free 예측은
P_rx -80.887 dBm, RSSI -80.384 dBm이다(Baseline RSSI -69.048 dBm). **측정값이 아닌 외삽값**이며
분석 CSV에도 20m는 `observed=False`, 측정 평균/SD 없음으로 기록된다. 20m와 (19,19)에서 2 dB Noise를
포함해도 값은 유한하다(`tests/test_calibration_extrapolation.py`).
단절/관측 불가 확률은 구현하지 않았으므로 이 외삽값은 해당 거리에서 실제로 신호를 수신할 수 있다는
근거가 아니다. 15m 초과 구간의 감쇠가 실측 구간과 같다는 가정 자체를 검증하지 않았다.

### Chebyshev smoke 재실행 결과

정정 전 smoke와 조건이 같고(20x20, (0,0), split, 8방향, 400 Step, LoS, 1m/cell, noise_std=2,
1 Run 42/43, 양 모드 각각 새 Q-Table로 학습 10/평가 10) Target 거리 기준만 Chebyshev다.
출력: `results/rssi_calibration_chebyshev_20261125/smoke/`. 기존 `results/rssi_calibration_20261125/`는 덮어쓰지 않았다.
네 조합(2 모델 × 2 알고리즘)의 평가 Target/Seed 10개가 모두 같고, 모두 평가 집합이며
Chebyshev ≥ 5다(실제 최소 6).

| 모델 | 알고리즘 | 성공 수 | 전체 평균 종료 Step |
|---|---|---:|---:|
| Baseline | Q-Learning | 4/10 | 264.5 |
| Baseline | Random | 1/10 | 398.7 |
| Calibrated | Q-Learning | 8/10 | 184.3 |
| Calibrated | Random | 1/10 | 398.7 |

후보 목록이 바뀌어 같은 Seed라도 평가 Target이 정정 전과 다르므로 정정 전 표와 직접 비교하지 않는다.
Random은 RSSI를 행동 선택에 쓰지 않으므로 두 모델에서 결과가 같다.
1 Run/10 Episode의 동작 확인이며 성능 결론이 아니다. 결과에 따라 알고리즘/파라미터를 조정하지 않았다.

## 8방향 실험 Target 거리 기준 통일 (2차 후속)

사용자 결정에 따라 앞으로의 모든 8방향 탐색 실험은 Target 후보를 Chebyshev 거리 ≥ 5로 선정한다.
위 절의 "다른 실험의 Manhattan 설정은 변경하지 않았다"는 1차 정정 당시의 기록이며, 이후 다음과 같이 통일했다.

- `src/targets.py`: Manhattan 분기를 제거하고 `"chebyshev"` 외의 거리 기준은 `ValueError`로 거부한다.
  `distance_metric` 인자, 후보 순서, 2x2 체커보드 분할, Target RNG/Seed 규칙은 유지한다.
- `src/best_paths_experiment.py`: 실행 설정, 후보 수 검사, 저장 메타데이터를 Chebyshev와 375/187/188로 바꿨다.
  Best Path 선정, 최단 Step 계산, 경로 검증, 그림 생성은 변경하지 않았다.
- `config.py` 주석, `README.md`, `tests/test_targets.py`, `tests/test_best_paths_experiment.py`를 현재 조건에 맞췄다.

| 항목 (20x20, 시작 (0,0), split) | 현재 기준 |
|---|---|
| Target 후보 거리 | Chebyshev ≥ 5 (Manhattan 지정 시 오류) |
| 후보 수 | 375 = 학습 187 + 평가 188 |
| 최단 이동 횟수 | Chebyshev 거리 |
| RSSI 전파 감쇠 거리 | Agent–Target Euclidean 거리 (변경 없음) |

20m 보정 RSSI 예측(-80.384 dBm)과 보정 관련 테스트는 변경 없이 통과한다.
과거 기록은 보존한다. `results/best_paths_8dir/random_target/grid_20x20/`의 Best Path 결과(1,000 Episode, 30 Run)는
Manhattan(385/193/192)으로 실행됐고 다시 실행하지 않았다. 이 문서의 정정 전 Manhattan smoke 기록도 그대로 둔다.
Best Path 동작 확인은 임시 폴더에 2 Run, 학습 2, 평가 8로만 실행했으며 연구 결과가 아니다.

### 3차 후속 정리

- Best Path 기본 출력은 `results/best_paths_8dir/chebyshev_target/grid_20x20/`이다.
  출력 폴더(`--output-dir` 포함)가 이미 있으면 설정 적용·학습·파일 기록 전에 `FileExistsError`로 중단한다.
  과거 Manhattan 결과가 있는 `random_target/` 폴더는 이동·삭제·덮어쓰기 없이 보존한다.
- Best Path 기본 횟수는 최종 논문 조건인 학습 3,000 Episode/Run, 평가 100 Episode/Run, 100 Run이다.
  함수 기본 인자와 CLI 기본값이 같은 상수를 쓴다. 100 Run 전체 실험은 실행하지 않았다.
- `split`은 최종 논문 조건(논문 명칭 Random Target)으로 설명한다. 이전 주석의 "random=본 실험,
  split=보조 실험" 표현만 정정했으며 모드 동작과 전역 기본값 `TARGET_MODE="fixed"`는 그대로다.
- `validate_target_config()`는 fixed를 포함한 모든 모드에서 `"chebyshev"` 외 거리 기준을 거부한다.

## 실행 방법

분석 의존성은 기존 `experiments/field_measurement/analysis/requirements.txt`를 재사용한다.
이번 환경에는 이미 설치되어 있어 패키지를 설치하거나 requirements를 변경하지 않았다.
프로젝트 루트에서 실행한다. 출력 폴더가 이미 있으면 중단하므로 새 이름을 사용한다.

```powershell
python -m src.calibration_analysis --output results/rssi_calibration_review/analysis
python -m src.calibration_smoke --profile data/rssi_calibration_20261125/calibration_profile.json --output results/rssi_calibration_review/smoke
python -m pytest -q -p no:cacheprovider
```

smoke 명령은 현재 Chebyshev 후보 기준으로 실행된다. 정정 후 실행 결과는
`results/rssi_calibration_chebyshev_20261125/smoke/`에 있고 구성은 아래 `smoke/`와 같다.

최초 실행의 경로는 `results/rssi_calibration_20261125/`다(`smoke/`는 정정 전 Manhattan 결과):

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
Target/Seed 규칙 및 기존 results 파일은 diff가 없다.
660e64f까지의 12개 커밋을 `origin/feature/rssi-realworld-calibration`에 푸시했고,
이 완료 기록을 별도 문서 커밋으로 추가한다. PR 생성과 main 병합은 하지 않았다.
요청 범위의 구현/검증은 완료됐다. 로컬 결과는 위 경로에서 확인하고 후속 연구는 별도 범위로 진행한다.
100 Run, 모든 Noise 조건 연구, 단절 모델/다른 지역 일반화는 이번 범위 밖이다.

후속 Chebyshev 정정은 e603ef2 이후 4852194 smoke 거리 기준 → 8bc4c88 Target 분할 테스트 →
6cc1f04 20m 외삽 테스트 → 이 문서 커밋 순서로 저장했다. 작업 전 전체 379개, 이후 **389 passed**다.
회귀 파라미터, 보정 프로필, 원본 CSV, RSSI 수식/Noise/State/Reward, Target/Seed 함수는 변경하지 않았다.

2차 Chebyshev 통일은 9ddec42 이후 84a263f Best Path 실행 조건/메타데이터 → 26167fa Best Path 메타데이터 테스트 →
1ab9d52 Target 테스트 → 9a8a6d0 targets.py Chebyshev 전용 → eb621d6 config 주석 → f9f8b9f README →
이 문서 커밋 순서로 저장했다. 작업 전 전체 389개, 이후 **391 passed**다.
RSSI 수식/Euclidean 거리, 보정 코드/파라미터/프로필, 원본 CSV, Q-Learning/Random, Seed 규칙, 기존 results는 변경하지 않았다.

3차 후속 정리는 471426b 이후 48618c4 기본 출력 경로 → cfab6a7 덮어쓰기 보호 → d7f6dc0 보호 테스트 →
25fd131 Target 모드 설명 → 062e2ce 기본 횟수 → 2cf240a 기본 횟수 테스트 → 85384a3 모든 모드 거리 기준 검증 →
7645c20 거리 기준 거부 테스트 → 2a269c4 README → 이 문서 커밋 순서다. 작업 전 391개, 이후 **409 passed**다.
