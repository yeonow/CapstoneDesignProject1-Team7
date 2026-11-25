# 실측 보정 RSSI 모델의 Level 경계 조정

> **종합설계프로젝트 최종 발표를 위한 추가 개선이다.** 이미 제출한 논문의 실험 조건과 결과를
> 바꾸지 않는다. 논문 결과는 Baseline RSSI 모델과 기존 Level 경계 `[-80, -70, -60, -50]` dBm 기준이다.
>
> 조정 Level은 실측 RSSI로 경계를 직접 최적화한 값이 아니다. 기존 경계가 거리별로 나누던 구분을
> 실측 보정 모델에서도 비슷하게 유지하도록 두 모델의 RSSI 대응으로 옮긴 값이다.

## 배경

실측 보정 모델(1m 종합 수신 전력 약 -60.279 dBm, 경로 손실 지수 약 1.584)은 Baseline보다 같은
거리에서 RSSI가 낮고 거리별 감소 폭이 작다. 기존 경계를 그대로 쓰면 같은 거리의 State Level이
달라지고 일부 Level에 상태가 몰린다.

| 거리 | Baseline noise-free RSSI | Calibrated noise-free RSSI |
|---:|---:|---:|
| 1 m | -43.062220108334394 dBm | -60.27432630646227 dBm |
| 15 m | -66.56439940090905 dBm | -78.58262709035326 dBm |

## 세 가지 설정

| 설정 이름 | `RSSI_MODEL` | `RSSI_LEVEL_MODE` | Level 경계 (dBm) |
|---|---|---|---|
| `baseline_original` | `baseline` | `original` | -80, -70, -60, -50 |
| `calibrated_original` | `calibrated` | `original` | -80, -70, -60, -50 |
| `calibrated_adjusted` | `calibrated` | `calibrated_adjusted` | -89.04901890526463, -81.25897503062048, -73.46893115597632, -65.67888728133218 |

`calibrated_adjusted`는 Calibrated 모델 전용이다. Baseline 모델에서 선택하면 오류로 중단한다.
기본값은 `RSSI_MODEL = "baseline"`, `RSSI_LEVEL_MODE = "original"`이므로 기존 실험은 그대로 재현된다.

## 조정 경계 계산 (`src/rssi_levels.py`)

1. Baseline 모델의 1m, 15m noise-free RSSI를 계산한다(식 (1)의 noise floor 합산 포함).
2. Calibrated 모델의 같은 거리 noise-free RSSI를 계산한다.
3. 두 점을 잇는 선형 대응을 구한다.
   - `slope = (cal_15m - cal_1m) / (base_15m - base_1m) = 0.7790043874644151`
   - `intercept = cal_1m - slope * base_1m = -26.72866790811142`
4. 기존 경계 4개에 `adjusted = intercept + slope * original`을 적용한다.

| 기존 경계 | 조정 경계 |
|---:|---:|
| -80 | -89.04901890526463 |
| -70 | -81.25897503062048 |
| -60 | -73.46893115597632 |
| -50 | -65.67888728133218 |

noise-free RSSI는 실험과 같은 `env_factory.build_environment`로 계산한다. 계산 중 바꾼 config와
전역 random 상태는 원래대로 복원한다. 결과는 진단 실험 manifest
(`results/rssi_level_validation_seed10042_10runs/validation_manifest.json`)의 값과 비트 단위로 같다.
P_rx로 계산하면 최대 0.48 dB 차이가 나므로 사용하지 않는다.

Level 경계만 바뀐다. RSSI 회귀모델과 보정 프로필, Noise 2 dB, Trend ±2 dB와 계산 방식, Reward,
Q-Learning/Random Search, 8방향 이동, 20×20, Chebyshev ≥ 5 Target과 학습/평가 분리, Seed 규칙은 그대로다.

## 설정 방법

```python
import config
config.RSSI_MODEL = "calibrated"
config.RSSI_CALIBRATION_FILE = "data/rssi_calibration_20261125/calibration_profile.json"
config.RSSI_LEVEL_MODE = "calibrated_adjusted"   # 기존 Level은 "original"
```

State를 만들 때마다 `state_reward.rssi_thresholds()`가 현재 모드의 경계를 조회하므로
학습과 평가에 같은 경계가 쓰인다. 조정 경계는 관련 설정값별로 한 번만 계산해 둔다.

## 실행 방법

전용 실행기는 논문 조건(20×20, 시작 (0,0), split, Chebyshev ≥ 5, 400 Step, Noise 2 dB)과
같은 Seed로 선택한 설정을 차례로 실행한다. 출력 폴더가 이미 있으면 시작하지 않는다.

```powershell
# 세 설정 모두 (기본: 학습 3,000 / 평가 100 Episode, 100 Run, Seed 42)
python -m src.rssi_level_experiment --output results/<새 폴더>

# 한 설정만
python -m src.rssi_level_experiment --output results/<새 폴더> --settings calibrated_adjusted
```

출력:

- `<출력>/<설정 이름>/algorithm_comparison_episodes.csv`, `algorithm_comparison_summary.csv`
- `<출력>/level_manifest.json`: 설정별 `RSSI_MODEL`, `RSSI_LEVEL_MODE`, 적용 경계, 조정 근거
  (기준 RSSI, slope, intercept), Trend 임계값, 실행 config, 프로필 SHA-256, Seed 규칙.
  학습 전에 기록하고, 완료 후 세 설정의 Target/Seed 일치 검증 결과와 평균 성공률을 추가한다.

`results/`는 `.gitignore` 대상이므로 결과는 로컬에만 남는다.

## 검증

- `tests/test_rssi_levels.py`: 기존 경계 기본값, 알 수 없는 모드 거부, 조정 경계가 진단 값과 정확히 일치,
  Baseline + 조정 Level 거부, Calibrated + 기존 Level 유지, config/random 상태 복원.
- `tests/test_rssi_level_reproducibility.py`: 조정 경계에서의 Level 구간과 State,
  original 모드가 변경 전 Level 함수와 같은 결과, 조정 모드가 진단 방식(경계 값 직접 지정)과 같은 결과,
  학습·평가 State와 Q-table에 조정 Level 반영, 세 설정의 Target/Seed 및 Random 결과 동일.
- `tests/test_rssi_level_experiment.py`: 실행기 메타데이터, 출력 보호, 단일 설정 실행, CLI 기본값.

### Seed 42, 5 Run 재현 (학습 3,000 / 평가 100 Episode)

`python -m src.rssi_level_experiment --output results/rssi_level_repro_seed42_5runs --runs 5`

| 설정 | 진단 실험 Q-Learning 성공률 | 재현 | episode/summary CSV |
|---|---:|---:|---|
| `baseline_original` | 64.6% (`rssi_precheck_20261125_5runs/baseline`) | 64.6% | 바이트 동일 |
| `calibrated_original` | 48.6% (`rssi_precheck_20261125_5runs/calibrated`) | 48.6% | 바이트 동일 |
| `calibrated_adjusted` | 62.8% (`rssi_diagnosis_20261125_5runs/level_adjusted`) | 62.8% | 바이트 동일 |

Random Search는 RSSI를 쓰지 않으므로 세 설정 모두 34.0%이다. 세 설정의 평가 Target/Seed 일치도 확인했다.
이 PC에서 15 Run(3설정 × 5 Run)은 약 3분 25초가 걸렸다.

## 해석과 한계

- 진단 15 Run 평균은 Calibrated + 기존 Level 53.5%에서 조정 Level 60.3%로 높아졌다.
  그러나 Run별로는 8개 개선, 7개 악화다. 새 Seed 10 Run(Seed 10042)은 5개 개선, 5개 악화이고
  평균은 56.0% → 59.1%다. Seed 42 5 Run의 상승은 2번 Run(11% → 83%)의 영향이 크다.
  **보편적인 성능 개선은 입증되지 않았으며**, 최종 판단은 100 Run 결과로 한다.
- 대응식은 1m와 15m 두 점만 사용한다. 실측은 1–15m이며 그 밖의 거리는 회귀모델 외삽이다.
- Trend 임계값 ±2 dB는 그대로 두었다. 보정 모델은 거리별 RSSI 변화가 작아 Trend 신호도 달라질 수 있으나
  이번 조정 범위가 아니다.

## 최종 100 Run 명령

```powershell
python -m src.rssi_level_experiment --output results/rssi_level_final_seed42_100runs
```

세 설정을 학습 3,000 / 평가 100 Episode, 100 Run(Train Seed 42, 44, …, 240)으로 차례로 실행한다.
5 Run 소요 시간 기준으로 이 PC에서 약 70분이 예상된다. 설정별로 나눠 실행하려면 각기 다른 새 폴더에
`--settings <설정 이름>`을 지정한다. 이때는 설정 간 Target/Seed 일치 검증이 한 실행 안에서만 이뤄지므로
결과를 합칠 때 같은 Seed와 Run 수인지 manifest로 확인한다.
