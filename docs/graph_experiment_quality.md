# 그래프 검증·재생성·인수인계

이 기능은 기존 CSV를 읽는 분석 작업이다. Q-Learning 재학습, 원본 실험 수치 보정,
실측 RSSI 보정은 수행하지 않는다. 기존 CSV/PNG/PDF는 그대로 보존한다.
검증 통과는 CSV 내부 정합성을 뜻하며 실험 실행 이력이나 물리적 타당성의 인증이 아니다.

## 논문 조건과 데이터의 구분

이번 작업 요청의 기준은 20×20, 8방향, split Target, 시작점 기준 Chebyshev 거리 ≥5,
학습 3,000 Episode/Run, 평가 100 Episode/Run, 조건별 100 Run이다.
Step 비교는 Noise 2 dB에서 최대 200/300/400 Step, Noise 비교는 최대 400 Step에서
2/4/6/8/10 dB를 사용한다. 이 문서는 해당 조건의 결과가 저장소에 모두 있다는 주장이 아니다.

현재 config.py 기본값이나 폴더 이름을 과거 실험의 증거로 사용하면 안 된다.
CSV에 없는 Grid/Noise/Target 분리/이동 규칙은 실험 당시 설정 기록으로 별도 확인한다.
과거 1,000 Episode/30 Run 자료를 3,000 Episode/100 Run 자료로 표시하지 않는다.
Step 비교 도구는 기본적으로 3,000/100/100 횟수를 확인하고 불일치 시 중단한다.
Target 후보의 Manhattan 필터를 사용했던 별도 과거 실험은 이번 Chebyshev 조건과 구분한다.

## 입력 파일과 스키마

| 데이터셋 | 필수 파일 | 내용 |
|---|---|---|
| algorithm | algorithm_comparison_episodes.csv | Run/알고리즘/Episode별 성공, Step, 최종 거리, Seed |
| algorithm | algorithm_comparison_summary.csv | 이름은 summary지만 **Run별** 집계 |
| noise | noise_comparison_episodes.csv | Episode 결과 + noise_std |
| noise | noise_comparison_runs.csv | Run별 집계 + noise_std |
| noise | noise_comparison_summary.csv | Noise·알고리즘별 Run 평균, 표본 SD, 지표별 유효 Run 수 |

검증 대상 알고리즘은 `q_learning`, `random`이다. Greedy/상대 방향 결과와 별도 파일 형식을
임의로 이름만 바꿔 투입하지 않는다. 다른 스키마는 명시적 변환/검토가 필요하다.
Episode 필수 열은 `run, algorithm, episode, train_seed, eval_seed, success, steps, final_distance`다.
Run 및 Noise Summary의 전체 열 정의는 `compare_algorithms.SUMMARY_FIELDS`,
`compare_noise_sweep.NOISE_RUN_FIELDS/NOISE_SUMMARY_FIELDS`를 사용한다.

- 누락/중복 헤더, 행의 열 개수, 손상된 숫자, NaN/무한대, 음수 값은 오류다.
- 빈 성공 시 평균 Step은 정상 결측이다. 0이나 보간값으로 바꾸지 않는다.
- Run/Episode 중복·누락, 알고리즘별 평가 건수, 성공·실패 합, Seed 및 재계산 지표를 검사한다.
- `target_x, target_y`는 함께 있어야 하며 음이 아닌 정수여야 한다. 한쪽만 비면 오류다.
- 좌표가 있으면 알고리즘·Noise 간 동일 Run/Episode Target을 비교한다.
  모든 좌표가 없거나 비어 있는 과거 자료는 해당 검증을 못 했다는 경고를 남긴다.
- 원본을 수정해 오류 행만 없애는 방식으로 그래프를 통과시키지 않는다.

## 통계와 그림의 의미

| 지표/그림 이름 | 단위 | 모집단 |
|---|---|---|
| success_rate_comparison / noise_success_rate | % | 각 Run의 성공 수 ÷ 전체 평가 Episode 수 |
| average_steps_comparison / noise_average_steps | steps | 성공·실패 및 경계에 막힌 행동 모두 포함 |
| average_success_steps_comparison / noise_success_steps | steps | 각 Run에서 성공한 Episode만 포함 |
| final_distance_comparison / noise_final_distance | grid units | 모든 평가 Episode의 종료 거리 |
| step_success_rate | % | 200/300/400 Step별 Run 성공률 |

각 Run을 동일 가중치로 평균한다. 성공 Step은 성공이 없는 Run을 제외한 **Run 평균의 평균**이다.
성공 Episode를 전부 합쳐 계산한 평균과 다를 수 있으므로 유효 Run 수와 성공 횟수를 함께 본다.
고잡음에서 성공한 쉬운 Episode만 남으면 성공 Step이 줄어도 전체 탐색 성능이 좋아졌다고 볼 수 없다.
Q-Learning 성공률이 Random보다 낮은 결과도 그대로 표시한다.

- 표본 SD: Run 간 산포이며 분모는 n−1이다. 성공률 SD 단위는 %p다.
- 95% CI: 평균 ± t(0.975, n−1) × SD / sqrt(n). SD 오차막대와 다르다.
- `graph_pipeline` 및 `plot_paper_results`: **95% Student t CI**.
- `plot_noise_comparison` 및 `plot_step_comparison`: **표본 SD**.
- 유효 Run이 2개 미만이면 새 그림에서는 오차막대를 생략한다.
  기존 원본 Summary의 단일 Run SD=0 저장 관례는 변경하지 않는다.
- CI가 물리 범위를 벗어나더라도 임의로 잘라내지 않는다.
- Noise별 성공 Step 결측은 선의 빈 구간으로 남긴다. NaN 문자열이 있는 손상된 CSV는 거부한다.

## 원본을 보존하는 권장 실행 순서

프로젝트 루트의 PowerShell에서 실행한다. 아래 `path/to/...`는 사용자가 실제 검증할
입력 폴더로 교체할 자리이며, 실험 실행 명령이 아니다. 이름은 실행마다 새로 지정한다.

```powershell
python -m src.graph_pipeline --input-dir "path/to/verified-csvs" --dataset both --output-root graph_outputs --name review-01
```

기본 비교만 있으면 `--dataset algorithm`, Noise 자료만 있으면 `--dataset noise`로 지정한다.
선택한 입력에 오류가 있으면 출력 폴더를 만들기 전에 중단한다.
선택한 데이터셋마다 표 1개와 지표별 PNG/PDF 4쌍을 저장한다. PNG는 300 dpi다.

```text
graph_outputs/pre_calibration/review-01/
  paper_tables/algorithm_performance.csv 또는 noise_performance.csv
  paper_figures/<위 표의 그림 이름>.png 및 .pdf
  graph_manifest.json
```

`graph_manifest.json`에는 입력 경로·크기·SHA-256, 소스 Git SHA·dirty 여부,
관련 소스 파일 해시, 분석 옵션·단위·경고, 산출물 해시를 기록한다.
입력이 생성 도중 바뀌면 오류로 처리한다. 실패한 실행의 부분 출력도 자동 삭제하거나 덮어쓰지 않는다.

기존 CLI도 유지되지만 기존 이름에 저장할 수 있으므로 보존이 필요하면 위 안전한 CLI를 사용한다.
별도 새 경로에 SD Noise 그림만 만들려면 다음처럼 사용할 수 있다.

```powershell
python -m src.plot_noise_comparison --input "path/to/noise_comparison_summary.csv" --output "new-output/noise-sd.png" --percent --dpi 300
```

이 기존 CLI는 자체 provenance JSON/덮어쓰기 보호를 제공하지 않는다.
`--percent`를 생략하면 기존처럼 0~1 비율이며 축에 fraction으로 표시한다.

## Step 비교: 원시 결과와 평균·SD 대조

```powershell
python -m src.plot_step_comparison --step200 "path/to/200steps" --step300 "path/to/300steps" --step400 "path/to/400steps" --output-root graph_outputs --name step-review-01
```

각 폴더에서 algorithm 입력 파일 두 개를 읽고 Episode→Run 정합성을 확인한다.
상한 초과, 실패 종료 Step, 학습/평가 횟수, Run 수, 조건 간 Seed/Target 대응을 검사한다.
`--expected-runs 100 --train-episodes 3000 --eval-episodes 100`이 기본값이다.
다른 횟수의 과거 실험을 검토할 때만 실제 횟수를 명시하고 논문 기준 실험과 구분한다.
성공률 평균과 오차막대는 검증된 Run에서 다시 계산한다.
`step_success_summary.csv`, `step_success_rate.png/.pdf`, `graph_manifest.json`을 새 폴더에 저장한다.
Noise 2 dB 등 CSV로 확인할 수 없는 조건은 그림과 manifest에도 선언값으로 구분한다.

## 팀원 4명의 Run 분담 검사

같은 조건의 Run을 1–25 / 26–50 / 51–75 / 76–100처럼 전역 번호로 나누고,
각 파일에서 Run 1부터 다시 시작하지 않는다. 아래 검사는 실험을 실행하거나 병렬화하지 않는다.

```powershell
python -m src.check_run_shards "worker1.csv" "worker2.csv" "worker3.csv" "worker4.csv" --expected-runs 100 --base-seed 42
```

학습 Seed=base+2×(전역 Run−1), 평가 Seed=학습+1 조건과 중복·누락·평가 건수를 검사한다.
같은 Run의 Seed를 Noise별로 공유하는 것은 허용한다. 다른 Grid/Step/Target 조건은 별도 호출한다.
`--expected-runs` 없이 검사하면 마지막 Run 이후의 누락은 알아낼 수 없다.
이 검사기는 고정 base의 전역 Run 규칙용이다. 기존 Noise 엔진이 지원하는 Run별 서로 다른 base 목록은
자동 변환하지 않으며, 그 실험의 Seed 프로토콜을 별도로 확인해야 한다.

## 실측 보정 전후 및 인수인계

기본 출력 phase는 `pre_calibration`이다. 실제 보정 후 데이터가 확보된 뒤에만
`--phase post_calibration`을 명시한다. phase는 사용자의 선언이며 실측 여부를 인증하지 않는다.
현재 보정 후 수치를 추정하거나 만들지 않는다. 두 phase는 같은 출력 이름을 써도 서로 다른 폴더다.

인수인계 시 원본 CSV, 당시 환경/Target 후보/Seed 기록, Git SHA, 새 그림·표·manifest를 함께 전달한다.
어떤 지표가 실패를 포함하는지, 오차막대가 SD인지 CI인지, 사용한 CLI 옵션을 함께 적는다.
소스가 dirty이면 Git SHA만으로 재현할 수 없으므로 manifest의 소스 해시와 해당 변경도 보존한다.

## 이번 변경의 검증 기록

기준 커밋 `4b8aa84`의 별도 Worktree에서 `python -m pytest -q -p no:cacheprovider`를 실행했다.
변경 전 289개, 코드 변경 후 338개 테스트가 통과했다. 추가 49개는 Target 대응, 그래프 입력,
통계 모집단, 단위/SD/결측 표시, 팀별 Run 분담, 출처 기록, 안전한 CLI와 Step 비교 검증이다.
테스트 수치는 임시 폴더의 가상 데이터이며 기존 논문 결과나 실측 결과가 아니다.
기존 실험 원본을 다시 생성하지 않았으며 학습/State/Reward/환경/Seed 생성 로직은 변경하지 않았다.
