# CapstoneDesignProject1-Team7

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
