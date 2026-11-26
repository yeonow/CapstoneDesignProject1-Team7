# CapstoneDesignProject1-Team7

[Raspberry Pi RSSI 실측 실험 안내](experiments/field_measurement/README.md)

## 기본 실행 환경: 실측 보정 환경

`config.py`의 기본값은 최종 실측 보정 실험 조건입니다. 별도 설정 없이 실행하는 모든 실험
(`compare_algorithms`, Best Path, 발표용 경로 등)에 적용됩니다.

| 항목 | 기본값 |
|---|---|
| Grid / 시작 / 최대 Step | 20×20, (0, 0), 400 Step |
| Target | `split`(논문 명칭 Random Target), 체비쇼프 거리 ≥ 5, 학습 187개 / 평가 188개 |
| RSSI 모델 | `RSSI_MODEL = "calibrated"` (`data/rssi_calibration_20261125/calibration_profile.json`: 1m 기준 수신 전력 약 -60.279 dBm, 경로 손실 지수 약 1.584) |
| RSSI Level | `RSSI_LEVEL_MODE = "calibrated_adjusted"`: -89.05, -81.26, -73.47, -65.68 dBm |
| 학습 / 평가 | 3,000 Episode / 100 Episode, Noise 표준편차 2 dB |

제출한 논문 조건(Baseline RSSI 모델과 기존 Level `[-80, -70, -60, -50]` dBm)은
`RSSI_MODEL = "baseline"`, `RSSI_LEVEL_MODE = "original"`을 함께 지정해 실행합니다.
`PATH_LOSS_EXPONENT = 2.0`과 `RSSI_THRESHOLDS`는 Baseline 모델과 조정 Level 계산의 기준값이므로 유지합니다.

## 논문 조건: 8방향 Best Path

프로젝트 루트에서 다음 명령으로 전체 실험을 재현합니다.

```powershell
python -m src.best_paths_experiment
```

전용 모듈은 20×20 Grid, 시작 (0, 0), 내부 `split` 모드(논문 명칭 **Random Target**),
Run당 Q-Learning 3,000 Episode 학습 및 100 Episode 평가, 독립 100 Runs, 최대 400 Step,
Gaussian Noise 표준편차 2 dB를 사용합니다. 각 Run에서 새 정책을 학습하고 greedy 평가합니다.
기존 Random Search의 8방향 균등 무작위 행동은 유지합니다.
RSSI 모델과 Level은 config 기본값(실측 보정 모델 + 조정 Level)을 따르며 JSON의
`rssi_model`, `rssi_level_mode`에 기록합니다.

Target 후보에는 시작점 기준 **체비쇼프 거리 ≥ 5**를 적용하여 375개를 만들고 2×2 블록 체커보드로
학습 187개/평가 188개로 나눕니다. 경로 최단 Step도 같은 **체비쇼프 거리**이며,
RSSI 전파 감쇠는 Agent–Target 사이의 **Euclidean 거리**로 계산합니다.
모든 8방향 실험은 `TARGET_DISTANCE_METRIC = "chebyshev"`만 지원하며 `manhattan`을 지정하면 오류로 중단합니다.
`fixed`, 코드의 `random`, 10×10 실험은 이 모듈에서 실행하지 않습니다.

기본 Seed는 42이고 Run r의 Train Seed는 `42 + 2*(r-1)`, Eval Seed는 `train_seed + 1`입니다.
평가 Target은 `target:<eval_seed>:<episode>`, Noise는 `<eval_seed>:<episode>`로 정하므로
두 알고리즘의 Run/Episode별 Target과 Noise 난수 시퀀스 앞부분이 같습니다.

성공한 실제 평가 Episode 중 `(Path Ratio, Steps, Run, Episode)`가 가장 작은 경로를
알고리즘별로 독립 선정합니다. Representative Path는 생성하지 않습니다.
두 경로의 Target은 다를 수 있으므로 경로 길이를 동일 Target의 성능 비교로 해석하면 안 됩니다.

결과는 기본적으로 `results/best_paths_8dir/chebyshev_target/grid_20x20/`에 저장합니다.
출력 폴더(`--output-dir`로 지정한 경로 포함)가 이미 있으면 학습을 시작하거나 파일을 쓰기 전에
오류로 중단합니다. 기존 결과를 덮어쓰지 않으므로 실행할 때마다 존재하지 않는 새 폴더를 사용합니다.

`results/best_paths_8dir/random_target/grid_20x20/`의 파일은 Target 후보를 맨해튼 거리 기준(385/193/192)으로,
1,000 Episode/30 Runs로 실행했던 **과거 결과**이며(`target_distance_metric: manhattan`) 그대로 보존합니다.
현재 체비쇼프 기준과 기본 횟수로 다시 실행한 결과가 아닙니다.

- `q_learning_best.png`, `random_search_best.png`
- `best_path_comparison.png`, `best_path_comparison.pdf`
- `q_learning_best.json`, `random_search_best.json` (전체 좌표는 `trajectory.path`)
- `experiment_summary.csv` (알고리즘별 전체 평가 집계 및 Best 정보)

기존 형식의 `algorithm_comparison_episodes.csv`와 `algorithm_comparison_summary.csv`도
같은 폴더에 저장합니다. 전체 평가 Step 평균에는 실패를 포함하고 Path Ratio 평균에는 성공만 포함합니다.
기존 `results/path_smoke`는 사용하지 않습니다.
성공이 없으면 JSON `trajectory`는 `null`, `status`는 `no_successful_episode`이며 그래프에 표시합니다.

빠른 확인과 테스트:

```powershell
python -m src.best_paths_experiment --train-episodes 2 --eval-episodes 8 --runs 2 --output-dir results/best_paths_8dir_smoke
python -m pytest -q -p no:cacheprovider
```

`--train-episodes`, `--eval-episodes`, `--runs`, `--seed`, `--output-dir`만 변경할 수 있습니다.
빠른 확인 폴더도 이미 있으면 중단되므로 다시 실행할 때는 새 `--output-dir`를 지정합니다.
논문 전체 실험은 인자를 생략한 기본 명령(3,000 Episode/Run, 100 Episode 평가, 100 Runs)을 사용합니다.

## 발표용 경로 비교 (Target 17, 18)

프로젝트 루트에서 실행합니다.

```powershell
python -m src.presentation_paths
```

기본값은 30 Runs, Run당 Q-Learning 학습 3,000 Episode, 알고리즘별 평가 100 Episode,
Seed 42입니다. 20×20 Grid, 시작점 (0, 0), 8방향 이동, Noise 표준편차 2 dB,
최대 400 Step을 사용합니다. 학습 Target은 체비쇼프 거리 5 이상인 후보 중 기존
2×2 블록 Split 규칙의 학습 집합에서 뽑습니다. (17, 18)은 평가 집합에 속하며
발표용 평가에서만 고정합니다. Q-Learning은 학습된 Q-table로 greedy 행동을 선택합니다.

Run r의 Train Seed는 `seed + 2*(r-1)`, Eval Seed는 `train_seed + 1`입니다.
알고리즘별 행동 RNG는 Run 평가 시작 시 Eval Seed로 초기화하고, 환경 Noise는
Episode마다 동일한 문자열 `<eval_seed>:<episode>`로 초기화합니다.
Random Search의 Train Seed는 같은 Run을 식별하는 데 사용합니다.

`results/presentation_paths/target_17_18/`에 PNG, PDF, JSON 및
`presentation_path_summary.csv`를 생성합니다. 독립 Best Path는 각 알고리즘의
성공 경로에서 (Path Ratio, 실제 Step, Run, Episode) 순으로 선택합니다.
Path Ratio는 실제 Step을 최단 거리 18로 나눈 값입니다.
`paired_comparison.*`은 두 알고리즘이 같은 Run과 Episode에서 성공한 쌍 중 합산
Step이 가장 적은 경우를 보여 줍니다. 합산 Step이 같으면 Run과 Episode 순으로 선택합니다.
JSON에는 실제 좌표가 첫 위치부터 마지막 Step까지 저장되며, 경계에 막힌 Step은
연속된 같은 좌표로 보존됩니다. 성공이 없으면 해당 JSON 경로는 `null`이며 그림에 표시합니다.
CSV는 독립 Best Path 두 건의 발표용 요약입니다.

`--runs`, `--train-episodes`, `--eval-episodes`, `--seed`, `--output-dir`를
지정할 수 있습니다. 테스트용 축소 실행은 별도 폴더를 사용합니다.

```powershell
python -m src.presentation_paths --runs 2 --train-episodes 2 --eval-episodes 8 --output-dir results/presentation_paths_smoke
python -m pytest -q -p no:cacheprovider
```

## 평가 경로 기록 및 시각화

`--plot-paths`로 평가 Episode의 실제 좌표를 기록하고 Grid 및 알고리즘별 Best / Representative Path를 저장합니다. 옵션을 생략하면 기존 실행 방식과 CSV 형식을 유지합니다.

```powershell
python -m src.compare_grid_sizes --target-mode fixed --runs 5 --plot-paths
python -m src.compare_grid_sizes --target-mode random --runs 5 --plot-paths
python -m src.compare_grid_sizes --target-mode split --runs 5 --plot-paths
```

빠른 확인용 실행:

```powershell
python -m src.compare_grid_sizes --grid-sizes 10 --target-mode random --train-episodes 3 --eval-episodes 10 --runs 1 --plot-paths --output-dir results/path_smoke
```

기본 저장 위치는 `results/target_<mode>/grid_<N>x<N>/paths/`입니다. `--output-dir` 지정 시 해당 폴더 아래에 같은 구조로 생성합니다.

- `q_learning_best.json`, `q_learning_representative.json`
- `random_best.json`, `random_representative.json`
- `path_comparison.png`, `path_comparison.pdf`

JSON 최상위에는 Target 모드, Grid 크기, 알고리즘, 선정 종류 및 상태가 있습니다. `trajectory`에는 Run/Episode, 학습·평가·Noise seed, 시작점·Target, 전체 좌표, 성공 여부, Step 수, 최단 Step, Path Ratio, Noise 표준편차 및 막힌 이동 횟수를 저장합니다. Random의 `train_seed`는 짝지어진 Q-Learning Run의 seed이며 Random은 학습하지 않습니다.

Best는 성공 Episode 중 `(Path Ratio, Steps, Run, Episode)` 순으로 선정합니다. Representative는 모든 Run의 성공 Episode를 모아 평균 Path Ratio에 가장 가까운 실제 Episode를 선정하며 동점은 `(Steps, Run, Episode)` 순으로 결정합니다. 좌표를 평균 내거나 경로를 재생성하지 않습니다.

성공이 없으면 `status`는 `no_successful_episode`, `trajectory`는 `null`이며 패널에는 `No successful episode`를 표시합니다. 재실행 시 네 JSON을 모두 덮어씁니다. 시작점이 Target이면 0 Step 경로를 기록하고 정의되지 않는 Path Ratio는 `null`로 저장합니다. 이 Episode는 비율 평균에서 제외하고, 성공 경로가 모두 0 Step이면 Run/Episode 순으로 선정합니다.

막힌 Step도 중복 좌표로 남으므로 좌표 수는 항상 `steps + 1`입니다. 기록은 위치만 읽으며 이동·RSSI 측정·난수 호출을 추가하지 않습니다. 학습 경로는 기록하지 않습니다. 실행 중 실패 경로는 선정 대상에서 제외하고 최종 선정한 네 경로만 파일로 저장합니다.

그래프는 환경과 동일하게 x가 오른쪽, y가 아래쪽으로 증가합니다. 각 Episode의 실제 Target, 방향 화살표와 일부 Step 번호를 표시하며 GUI 창은 열지 않습니다. 하단 성공률·평균 Step·평균 최종 거리는 기존 `calculate_metrics`로 모든 평가 Episode를 집계한 값이고 성공 시 평균 Path Ratio는 기존 `path_ratio_on_success`로 계산합니다. 평균 Step·최종 거리는 실패를 포함하며 비율 평균은 최단 거리가 양수인 성공만 포함합니다.

Run마다 성공 수가 다르면 전체 성공 Episode의 비율 평균은 기존 Grid 요약 CSV의 Run별 비율 평균을 다시 평균 낸 값과 다를 수 있습니다. 경로 선정에도 Episode 전체 평균을 사용합니다. `--no-plot`은 기존 Grid 비교 그래프만 생략하므로 `--plot-paths`와 함께 사용할 수 있습니다.

검증: `python -m pytest -q -p no:cacheprovider`
## 실측 보정 RSSI Level 조정 (종프 최종 발표용 추가 개선)

제출한 논문 결과(Baseline + 기존 Level `[-80, -70, -60, -50]` dBm)는 바꾸지 않습니다.
Baseline + 기존 Level, Calibrated + 기존 Level, Calibrated + 조정 Level을 같은 Seed로 비교합니다.

```powershell
python -m src.rssi_level_experiment --output results/<새 폴더>
```

조정 경계 계산, 설정 방법, 재현 결과와 한계는 [RSSI Level 조정 안내](docs/rssi_level_adjustment.md)를 참고하세요.

## 실험 CSV 검증과 그래프 재생성

학습을 다시 실행하지 않고 기존 CSV를 검증해 그래프를 만들 수 있습니다.
원본 보존, 입력 형식, 성공률/성공 시 Step 해석, SD/95% CI 구분과 실행 명령은
[그래프·실험 결과 인수인계 안내](docs/graph_experiment_quality.md)를 참고하세요.

새 안전한 실행기는 `python -m src.graph_pipeline --help`로 확인합니다.
출력은 `pre_calibration/<새 실행 이름>/` 또는 `post_calibration/<새 실행 이름>/`으로
분리되며, 기존 출력 폴더는 덮어쓰지 않습니다.
