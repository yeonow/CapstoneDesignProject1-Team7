# Experiment / Integration 검증 기록

## 1. 검증 목적

현재 코드에서 Training → 동일 Agent Evaluation → 결과 저장 및 별도의 Reward Sweep이 정상적으로 실행되는지 확인한다. 아래 숫자는 **현재 HEAD 기준 통합 실행 확인용 Smoke Test 결과**이며 프로젝트의 최종 성능이 아니다. Environment / State / Reward / Q-Learning 코드나 설정이 변경되면 정식 실험 전에 다시 검증해야 한다. 튜닝, Baseline 구현, 장시간 성능 비교는 수행하지 않았다.

기존 docs에는 설계안과 설계 문서 비교만 있어 이 실행 검증 기록을 별도로 작성했다. Production 코드와 config.py는 수정하지 않았다.

## 2. 검증 코드 버전

- Branch: `feature/experiment-integration`
- HEAD Commit: `ca1d4e4033278cee5095efa582225b6c1f0827a8`
- 시작 시 작업 트리: `nothing to commit, working tree clean`
- Python: `3.12.10`, Windows PowerShell
- 실제 의존성 버전: numpy `2.5.3`, pytest `9.1.1`, matplotlib `3.11.2`
- 검증 날짜: `2026-10-08` (Asia/Seoul)
- 전체 테스트: `47 passed`

문서 작성 전과 검산 실행 후 모두 전체 테스트 47개가 통과했다.

위 HEAD는 실행한 코드 버전이다. 이 문서의 저장/커밋 이후 HEAD와 구분한다.

## 3. 실행 명령

모든 명령은 프로젝트 루트에서 실행했다. 이번 환경의 Python에는 기본 의존성이 없어 앞 단계에서 설치한 `$env:TEMP\evaluation-integrity-deps`를 사용했다. 아래 환경 변수는 실행 셸에만 적용하며 프로젝트 설정을 바꾸지 않는다. 다른 팀원은 requirements.txt의 의존성이 준비된 Python 환경에서 이 임시 패키지 경로 설정을 생략할 수 있다.

### 전체 테스트

기존 pytest 임시 디렉터리에 접근 권한 문제가 있어 실행마다 새 임시 루트를 사용했다.

```powershell
$env:PYTHONPATH = "$env:TEMP\evaluation-integrity-deps"
$env:PATH = "$env:TEMP\evaluation-integrity-deps\bin;$env:PATH"
$env:PYTEST_DEBUG_TEMPROOT = Join-Path $env:TEMP ('sweep-integrity-pytest-' + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $env:PYTEST_DEBUG_TEMPROOT | Out-Null
pytest -q
```

### 일반 Training의 짧은 실행

`src/experiment.py`의 기본 진입점은 `train()`이다. 학습 결과만 반환하고 `training_results.csv`를 저장하며 평가하지 않는다. CLI Episode 옵션이 없어 이번에는 같은 `train()`을 호출하면서 학습 횟수·로그 간격·저장 경로만 프로세스 내에서 임시 변경했다. 실제 실행한 명령은 다음과 같다.

```powershell
python -c "from unittest.mock import patch; import csv; from pathlib import Path; import config; from src.experiment import train; output=Path('results/integration_smoke_ca1d4e4'); output.mkdir(parents=True,exist_ok=True); ctx=patch.multiple(config,NUM_EPISODES=2,LOG_INTERVAL=1,RESULTS_DIR=str(output)); ctx.start(); results=train(); ctx.stop(); assert len(results)==2; f=(output/'training_results.csv').open(newline='',encoding='utf-8'); rows=list(csv.DictReader(f)); f.close(); assert rows==[{k:str(v) for k,v in r.items()} for r in results]; print('Training CSV verified:',len(rows),'rows')"
```

일반 설정의 모듈 진입점은 `python -m src.experiment`이며, 코드에서만 확인했고 이번에 기본 500회 학습을 실행하지는 않았다.

### Final Training + Evaluation

```powershell
python -m src.final_evaluation --train-episodes 2 --eval-episodes 2 --output results/integration_smoke_ca1d4e4/final_evaluation_results.csv
```

### Reward Sweep

```powershell
python -m src.experiment_sweep --train-episodes 2 --eval-episodes 2 --output results/integration_smoke_ca1d4e4/reward_sweep_results.csv
```

Final/Sweep은 기존 `evaluate_reward_combination()`을 재사용하며 내부에서 `experiment.py`의 생성·Episode·평가 함수를 호출한다. Final은 실행 시점의 config Reward 한 조합만 사용하고 Sweep의 최상 설정을 선택하지 않는다. 일반 Training의 Agent는 Final에 넘기지 않으며, Final 안에서 새로 2회 학습한 동일 Agent를 평가한다.

## 4. 설정 Snapshot 및 Smoke 조건

값은 검증 HEAD의 실제 config.py에서 확인했다. 기본값을 영구 변경하지 않았다.

| 분류 | 설정 | 기본값 |
| --- | --- | --- |
| Environment | GRID_SIZE / MAX_STEPS | 10 / 100 |
| Environment | AGENT_START / TARGET_POSITION | (0, 0) / (9, 9) |
| Q-Learning | ALPHA / GAMMA / EPSILON | 0.1 / 0.9 / 0.9 |
| Q-Learning | EPSILON_MIN / EPSILON_DECAY | 0.05 / 0.995 |
| Experiment | NUM_EPISODES / NUM_EVAL_EPISODES | 500 / 100 |
| Experiment | RANDOM_SEED / LOG_INTERVAL / RESULTS_DIR | 42 / 50 / results |
| RSSI | REFERENCE_RSSI / REFERENCE_DISTANCE | -40.0 / 1.0 |
| RSSI | PATH_LOSS_EXPONENT / NOISE_STD | 2.0 / 2.0 |
| Reward | REWARD_UP / REWARD_KEEP / REWARD_DOWN | 1.0 / 0.0 / -1.0 |
| Reward | MOVE_COST / TERMINAL_REWARD | 0.1 / 10.0 |
| 경계 Reward | BLOCKED_APPLY_MOVE_COST / BLOCKED_TREND_KEEP / BLOCKED_PENALTY | True / False / 0.0 |

config.py에 RSSI_THRESHOLDS / TREND_THRESHOLD는 없다. `src/state_reward.py`의 fallback을 사용하며, 실제 적용 값은 `[-80.0, -70.0, -60.0, -50.0]`과 `2.0 dBm`이다. State는 `(RSSI level, RSSI trend, previous action)`이고 NONE_ACTION은 -1이다. 경계에 막힌 시도도 Step에 포함되며 기본 경계 Reward 설정은 일반 Action과 같은 처리다.

Smoke에서 일반 Training / Final / 각 Sweep 조합 모두 학습 2회, Final / 각 Sweep 조합은 평가 2회로 실행했다. MAX_STEPS 등 나머지 설정은 표의 값 그대로다. 일반 Training만 LOG_INTERVAL=1로 임시 지정했다.

### Sweep 조합

`src/experiment_sweep.py`의 DEFAULT_REWARD_COMBINATIONS에 정의된 다섯 조합을 모두 실행했다. 표에 없는 Reward는 config 기본값을 사용한다.

| 이름 | baseline 대비 변경 |
| --- | --- |
| baseline | 없음 |
| strong_up | reward_up = 2.0 |
| strong_down_penalty | reward_down = -2.0 |
| high_move_cost | move_cost = 0.3 |
| high_terminal_reward | terminal_reward = 20.0 |

baseline은 기본 Reward 조합 이름이며 비교용 Random/Greedy 알고리즘을 의미하지 않는다. 모든 조합에서 reward_keep=0.0이다.

학습 환경의 Python 전역 random과 Agent의 별도 random.Random은 seed 42로 시작한다. 평가 환경의 전역 random은 seed 43으로 다시 초기화된다. Agent의 평가 난수 상태는 학습 후 상태를 이어받으며, greedy 선택에서도 random.choice가 소비된다. 따라서 Q-table/epsilon 불변성과 모든 난수 상태 불변성은 구분한다. 같은 코드·설정·Python/의존성 환경에서 재현성을 확인했으며, 다른 버전 간 동일 결과는 검증하지 않았다.

## 5. 실제 생성 파일

이번 HEAD의 결과는 이전 단계 산출물과 혼동하지 않도록 전용 경로에 저장했다. CSV는 기존 정책대로 같은 파일을 덮어쓰며 timestamp 정책을 새로 추가하지 않았다.

| 경로 | 내용 |
| --- | --- |
| results/integration_smoke_ca1d4e4/training_results.csv | 일반 Training 2 Episode, 2행 |
| results/integration_smoke_ca1d4e4/final_evaluation_results.csv | Final summary, 1행·15열 |
| results/integration_smoke_ca1d4e4/reward_sweep_results.csv | 모든 Reward 조합, 5행·15열 |
| results/integration_smoke_ca1d4e4/episode_verification.json | 아래 검산에서 수집한 Episode 결과·독립 검산·Agent 불변성·설정 Snapshot |

아래 검산은 기존 함수 호출을 관찰해 raw Episode 결과를 별도로 저장한다. 기본 Final/Sweep CSV 자체에는 Episode별 결과가 없으며 Q-table 파일도 자동 저장하지 않는다. results/는 .gitignore 대상이므로 결과 파일은 Git에 포함되지 않는다. 다른 팀원은 위 명령과 아래 검산으로 다시 생성할 수 있다.

## 6. 결과 검산

### 일반 Training

실제 실행된 2 Episode 모두 실패하고 Step은 각각 100이었다. 학습 CSV와 train() 반환값을 모든 열에 대해 비교했다. 출력 Reward는 각각 약 -11.0, -5.0이며 부동소수점 원본값은 CSV에 보존된다.

### Final Evaluation

raw Episode로부터 성공 수·실패 수·평균을 독립 계산하고, 반환 summary 및 CLI CSV에 대조했다.

| 항목 | 현재 HEAD Smoke 결과 / 검산 |
| --- | --- |
| 평가 횟수 | 2 |
| 성공 / 실패 | 1 / 1; 1 + 1 = 2 |
| 성공률 | 1 / 2 = 0.5 |
| 전체 평균 Step | (73 + 100) / 2 = 86.5 |
| 성공 Episode 평균 Step | 73 / 1 = 73.0 |

전체 평균에는 실패 Episode의 100 Step도 포함된다. 성공률 단위는 0~1이다. 성공 0건이면 성공 평균은 Python None, CSV 빈 값이며 성공 시 0 Step으로 해석하면 안 된다. 이번 Final Smoke는 성공 1건이라 결측 경로는 전체 테스트에서 검증됐다.

### Reward Sweep

| 조합 | 평가 횟수 | 성공 / 실패 | 성공률 | 전체 평균 Step | 성공 평균 Step |
| --- | --- | --- | --- | --- | --- |
| baseline | 2 | 1 / 1 | 0.5 | 86.5 | 73.0 |
| strong_up | 2 | 1 / 1 | 0.5 | 92.0 | 84.0 |
| strong_down_penalty | 2 | 2 / 0 | 1.0 | 39.5 | 39.5 |
| high_move_cost | 2 | 2 / 0 | 1.0 | 70.5 | 70.5 |
| high_terminal_reward | 2 | 1 / 1 | 0.5 | 86.5 | 73.0 |

각 행의 이름·5개 Reward 설정·학습/평가 횟수를 정의와 대조했다. 성공률은 0~1이며 성공 수는 평가 횟수−실패 수로 검산했다. 학습 평균 Step/Reward 및 평가 평균은 아래에서 수집한 raw 결과로 전체 5행을 검산했고, CSV와 반환 결과도 모든 열에 대해 비교했다. 순위를 매기거나 가장 좋은 Reward를 채택하지 않았다.

## 7. Episode를 이용한 검산 재실행

아래는 검증용 관찰 코드다. 기존 학습·평가 함수를 호출하며 선택/업데이트 공식이나 seed 설정은 변경하지 않는다. Final 한 조합과 Sweep 다섯 조합을 재실행해 앞서 CLI로 생성한 CSV와의 일치를 확인한다. 각 Agent를 보관해 서로 다른 객체인지 확인하고, 평가 직전/직후의 Q-table key·모든 배열과 epsilon을 비교한다. 통계는 관찰한 raw 결과의 합/건수로 독립 계산한다.

<!-- verification-python -->
```python
import csv
import json
import math
import subprocess
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import config
from src import experiment, experiment_sweep as sweep, final_evaluation as final

output = Path("results/integration_smoke_ca1d4e4")
snapshot = {k: v for k, v in vars(config).items() if k.isupper()}
original_episode = experiment.run_episode
original_evaluate = experiment.evaluate_agent
agents, training, evaluations = [], [], []

def create_agent():
    agent = experiment.build_agent()
    assert all(agent is not previous for previous in agents)
    assert agent.q_table == {} and agent.epsilon == config.EPSILON
    agents.append(agent)
    training.append([])
    return agent

def observe_training(env, agent, reward_config):
    assert agent is agents[-1]
    result = original_episode(env, agent, reward_config=reward_config)
    training[-1].append(result)
    return result

def observe_evaluation(env, agent, num_episodes, reward_config):
    assert agent is agents[-1] and len(training[-1]) == 2
    q_before, epsilon_before = deepcopy(agent.q_table), agent.epsilon
    raw = []
    def observe_episode(*args, **kwargs):
        assert kwargs["training"] is False
        result = original_episode(*args, **kwargs)
        raw.append(result)
        return result
    with patch.object(experiment, "run_episode", side_effect=observe_episode):
        summary = original_evaluate(env, agent, num_episodes, reward_config)
    assert len(raw) == num_episodes == 2
    assert agent.epsilon == epsilon_before and set(agent.q_table) == set(q_before)
    for state in q_before:
        np.testing.assert_array_equal(agent.q_table[state], q_before[state])
    successful = [r["steps"] for r in raw if r["success"]]
    calculated = {
        "success_count": len(successful),
        "failure_count": len(raw) - len(successful),
        "success_rate": len(successful) / len(raw),
        "average_steps": sum(r["steps"] for r in raw) / len(raw),
        "average_steps_on_success": sum(successful) / len(successful) if successful else None,
    }
    for key, value in summary.items():
        assert calculated[key] == value
    evaluations.append({"raw": raw, "calculated": calculated,
                        "q_table_unchanged": True, "epsilon_unchanged": True,
                        "same_training_agent": True, "train_raw": training[-1]})
    return summary

def compare_csv(path, summaries):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows == [{k: "" if v is None else str(v) for k, v in s.items()}
                    for s in summaries]

with patch.object(sweep, "build_agent", side_effect=create_agent), \
     patch.object(sweep, "run_episode", side_effect=observe_training), \
     patch.object(sweep, "evaluate_agent", side_effect=observe_evaluation):
    final_csv = output / "final_evaluation_results.csv"
    sweep_csv = output / "reward_sweep_results.csv"
    before_final, before_sweep = final_csv.read_bytes(), sweep_csv.read_bytes()
    final_summary = final.run_final_experiment(2, 2, final_csv)
    sweep_summaries = sweep.run_sweep(train_episodes=2, eval_episodes=2, output_path=sweep_csv)

assert len(agents) == len(evaluations) == 6
assert len(sweep_summaries) == len(sweep.DEFAULT_REWARD_COMBINATIONS) == 5
assert final_summary["success_count"] == evaluations[0]["calculated"]["success_count"]
assert final_summary["success_count"] + final_summary["failure_count"] == 2
compare_csv(final_csv, [final_summary])
compare_csv(sweep_csv, sweep_summaries)
assert final_csv.read_bytes() == before_final and sweep_csv.read_bytes() == before_sweep
for combination, summary, observation in zip(sweep.DEFAULT_REWARD_COMBINATIONS,
                                            sweep_summaries, evaluations[1:]):
    assert summary["name"] == combination["name"]
    for field in sweep.REWARD_FIELDS:
        assert summary[field] == combination[field]
    for field in ("success_rate", "average_steps", "average_steps_on_success", "failure_count"):
        assert summary["eval_" + field] == observation["calculated"][field]
    raw = observation["train_raw"]
    assert summary["train_episodes"] == summary["eval_episodes"] == len(raw) == 2
    assert summary["train_success_rate"] == sum(r["success"] for r in raw) / len(raw)
    for key, field in (("train_average_steps", "steps"),
                       ("train_average_total_reward", "total_reward")):
        assert math.isclose(summary[key], sum(r[field] for r in raw) / len(raw),
                            rel_tol=1e-12, abs_tol=1e-12)
assert snapshot == {k: v for k, v in vars(config).items() if k.isupper()}
evidence = {
    "verified_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "config": snapshot, "final": final_summary, "sweep": sweep_summaries,
    "observations": evaluations,
    "checks": {"csv_matches_summary": True, "cli_matches_observed_rerun": True,
               "all_agents_distinct": True, "config_unchanged": True},
}
(output / "episode_verification.json").write_text(
    json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Verified: 6 distinct agents, 12 evaluation episodes, all statistics and CSV values")
print(json.dumps([o["raw"] for o in evaluations], indent=2))
```
<!-- /verification-python -->

프로젝트 루트에서 위 문서 코드 블록을 실행한 실제 명령:

```powershell
python -c "from pathlib import Path; text=Path('docs/experiment_integration_validation.md').read_text(encoding='utf-8'); block=text.split('<!-- verification-python -->')[1].split('<!-- /verification-python -->')[0]; exec(compile('\n'.join(block.strip().splitlines()[1:-1]),'<integration verification>','exec'))"
```

## 8. 확인 범위와 미확인 사항

- 코드에서 확인: 기존 세 진입점, config 및 fallback, seed 정책, 통계 의미, 저장 파일명과 덮어쓰기/결측 정책.
- 실제 실행에서 확인: 전체 테스트 통과, 세 종류의 짧은 실행, 지정 Episode 수, Final 및 Sweep의 동일 학습 Agent 평가, 조합별 독립 Agent, 평가 전후 Q-table/epsilon 불변, raw 통계와 CSV 일치, CLI와 관찰 재실행 결과 일치.
- 미확인: 기본 500회 학습/100회 평가의 정식 성능, 다른 seed의 성능 분포, 다른 Python/의존성 버전의 비트 단위 재현성, 실제 RSSI 장비에서의 결과.
- 기존 제한: 평가 중 Agent 난수 상태는 바뀐다. 최종 CSV만으로 모든 환경·학습 설정을 복원할 수 없으므로 검증 HEAD, 위 Snapshot, 명령과 함께 사용한다.
- 이후 Environment / State / Reward / Q-Learning 변경 시 같은 검산을 다시 수행한다. 이번 기록만으로 최종 알고리즘 성능을 주장하지 않는다.

## 9. 최종 Integration Check — 5단계

### 현재 코드 버전과 전체 테스트

- 검증 날짜: 2026-10-10 (Asia/Seoul)
- Branch: `feature/experiment-integration`
- 검증 HEAD: `1a4e95aa76738835dc3fda3d0df3b8540c021d30`
- 시작 작업 트리: clean, Python 3.12.10
- 변경 전 전체 테스트: 총 52개, 통과 52개, 실패 0개, pytest 실행 시간 2.74초.
- 문서 수정 후 최종 전체 테스트: 총 52개, 통과 52개, 실패 0개, pytest 실행 시간 0.60초.
- 실행 명령: 3절의 동일 PowerShell 환경 설정 후 `pytest -q`.

현재 HEAD와 4단계 실행 HEAD `ca1d4e4` 사이의 `git diff -- src config.py`는 비어 있다. 차이는 이 검증 문서와 `tests/test_reproducibility.py` 추가뿐이다. 따라서 **4단계 Smoke/통계 검산 결과를 재사용**했으며 이번에는 별도 CLI Smoke를 반복하지 않았다. 보관된 `episode_verification.json`의 검증 HEAD와 성공한 검산 항목도 확인했다. 새로 실행한 전체 테스트에는 실제 환경/Agent를 사용하는 짧은 학습·평가·Sweep·CSV 테스트가 포함된다. 아래 상태는 현재 코드의 Integration 회귀 검증이며 최종 성능 평가가 아니다.

### Experiment 주요 검증 상태

| 항목 | 상태 | 현재 HEAD의 근거 |
| --- | --- | --- |
| Evaluation Q-value·key 불변 | Pass | test_evaluate_agent_preserves_q_table, 최종 경로의 실제 배열 비교 |
| Evaluation epsilon 불변 | Pass | test_evaluate_agent_preserves_epsilon, 최종 경로의 epsilon 비교 |
| Evaluation Q update·epsilon decay 없음 | Pass | test_evaluation_episode_uses_greedy_policy_without_learning, 최종 경로 호출 검사 |
| Reward Sweep Agent·Q-table·epsilon 독립 | Pass | test_reward_combinations_start_with_independent_agents_and_equal_conditions |
| Reward 외 환경·학습 조건 동일 | Pass | 위 테스트의 실제 객체 설정·Episode 횟수·초기 RNG 검사 |
| 동일 seed 재현성·조합 순서 독립 | Pass | test_sweep_is_reproducible_with_same_seed_and_independent_of_order |
| Sweep 종료 후 config 불변 | Pass | 정상 경로 Snapshot, 학습/평가 예외 시 config 불변 테스트; config를 변경하지 않는 구현 |
| Final 동일 학습 Agent 평가·summary·CSV | Pass | test_final_experiment_reuses_trained_agent_without_learning_in_evaluation |
| 새 Final 실행 초기화·재현성·덮어쓰기 | Pass | test_final_experiment_restarts_reproducibly_and_overwrites_csv |
| 성공/실패 합계·성공률·평균 | Pass | test_evaluate_agent_metrics 및 Final CSV/summary 검사; 4단계 raw 검산 재사용 |
| 성공 0건의 평균 결측 처리 | Pass | Sweep/Final 저장 테스트: None → CSV 빈 값 |
| Action·Q-table·학습 결과·평가 결과 재현성 | Pass | tests/test_reproducibility.py의 5개 테스트 |

전체 평균 Step에는 실패 Episode도 포함한다. `success_count + failure_count = eval_episodes`, `success_rate = success_count / eval_episodes`이며 성공률 단위는 0~1이다. Q-table/epsilon 불변은 난수 상태 불변을 의미하지 않는다. 평가의 `random.choice()`는 Agent RNG를 소비한다.

### Baseline 구현 현황 — 코드에서 확인한 사실

저장소 전체의 파일 목록과 `greedy|baseline|random|RSSI|compare|evaluation|policy` 검색 결과를 확인하고, 각 일치 항목의 구현 및 import 경로를 읽었다. 검색 결과가 있다는 사실만으로 구현 완료로 분류하지 않았다.

| 방법 | 구현 상태 | 위치 | 실제 의미 |
| --- | --- | --- | --- |
| Q-Learning | 구현 완료, 통합 실행 연결됨 | src/q_learning.py의 QLearningAgent; experiment.py, experiment_sweep.py, final_evaluation.py | 실제 tabular Q-learning 학습 후 Q-table greedy 평가 |
| Greedy RSSI baseline | 미구현 | 실행 클래스/함수/runner 없음; 설계안 15절에 RSSI Gradient Search 개념만 있음 | 현재 Q값 최대화 policy는 RSSI를 직접 비교하는 baseline이 아님 |
| Random baseline | 미구현 | 독립 policy/평가 진입점 없음; 설계안에 향후 비교 방식으로 언급 | Agent exploration 및 stub의 random.choice는 baseline이 아님 |

`src/q_learning_stub.py`는 Q-table을 사용하는 임시 epsilon-greedy Agent이고 현재 Experiment에서 import하지 않는다. `experiment_sweep.py`의 baseline은 기본 **Reward 조합 이름**이다. `plot_sweep.py`의 baseline 표시도 해당 조합의 CSV를 그리며 별도 알고리즘을 실행하지 않는다.

설계안의 Gradient 개념은 최근 이동 결과 또는 주변 탐색으로 RSSI 증가 방향을 찾고, 증가하면 방향을 유지하며 감소하면 다른 방향을 탐색하는 예시다. 방향 선택/탐색 비용을 모두 확정한 실행 명세나 구현은 아니다.

### Greedy RSSI 비교 전 규칙과 팀 합의

“확정됨”은 현재 코드 또는 문서에 명시된 범위만 뜻한다. 기존 공통 환경 규칙을 미구현 baseline 자체의 동작으로 단정하지 않는다.

| 항목 | 상태 | 확인한 정의 / 합의할 내용 |
| --- | --- | --- |
| 1. 이동 가능한 방향 | 확정됨: 공통 행동 공간 | 0=UP(y−1), 1=DOWN(y+1), 2=LEFT(x−1), 3=RIGHT(x+1) |
| 2. 방향별 RSSI 획득 | 팀 합의 필요 | 현재 API는 현재 위치 RSSI만 제공; 이동 이력 사용 또는 주변 탐색 중 구체적인 방법 미정 |
| 3. 선택 전 여러 방향 측정 | 팀 합의 필요 | 모든 방향 사전 측정을 보장하는 API/정책 없음 |
| 4. 측정하려면 실제 이동하는지 | 현재 API는 확정됨; baseline 탐색 절차는 합의 필요 | move() 후 새 RSSI 생성; 이웃 좌표의 신호를 무상 조회하는 public API 없음 |
| 5. 측정 후 원위치 복귀 | 팀 합의 필요 | 복귀 허용 여부와 복귀 이동/측정 비용 미정 |
| 6. Step당 측정 횟수 | 현재 Simulation은 확정됨; baseline 예산은 합의 필요 | reset 1회, Action 시도당 1회 생성, get_rssi는 추가 측정 없음 |
| 7. Noise에서 반복 측정 여부 | 팀 합의 필요 | Simulation은 단일 샘플; 설계안은 실측 반복 측정 제안이며 횟수 미정 |
| 8. 반복 측정 대표값 | 도구 구현은 확정됨; 비교 정책은 합의 필요 | get_representative_rssi는 다중 샘플의 중앙값; Experiment는 이를 직접 호출하지 않음. 설계안의 실측 대표값 선택은 미정 |
| 9. RSSI 동률 처리 | 팀 합의 필요 | Q값 동률의 random.choice 규칙을 RSSI baseline으로 자동 적용하지 않음 |
| 10. Grid 경계 | 확정됨: 현재 환경; baseline 선택은 합의 필요 | 막히면 제자리, blocked=True, Step+1, RSSI 재측정. 경계 Action 제외/재선택 여부는 미정 |
| 11. 장애물 | 팀 합의 필요 / 현재 미구현 | 환경에는 Grid 경계만 있고 장애물 Map/회피 API 없음 |
| 12. 신호 미검출 | 팀 합의 필요 / 현재 미구현 | Simulation은 숫자 RSSI 생성; 설계안은 별도 미검출 상태 검토만 명시 |
| 13. 성공 조건 | 확정됨: 현재 Simulation | agent_position == target_position. 설계안의 실환경 위치 후보/허용 범위는 별도 합의 필요 |
| 14. 최대 Step | 확정됨: 공통 비교 조건 | config.MAX_STEPS=100; 모든 방식에 같은 제한 적용 |
| 15. 측정 비용의 Step 포함 | 원칙은 문서에 명시; 계산 방식은 팀 합의 필요 | 설계안 17절은 측정 횟수 차이를 탐색 비용에 포함하도록 함. 별도 측정/왕복 비용 및 지표 저장 방식 미정 |

### Baseline 공정 비교 조건

설계안 17절은 시작 위치·Target·Grid·Noise·최대 Step·측정 방식의 동일성을 요구한다. 아래는 현재 코드에 대응하는 비교 조건이며, 실제 baseline 구현에는 아직 적용/검증하지 않았다.

| 조건 | 현재 코드에서 확인된 기준 |
| --- | --- |
| Grid/Map | GRID_SIZE=10, 장애물 없음 |
| 시작/Target | AGENT_START=(0, 0), TARGET_POSITION=(9, 9) |
| Step 제한 | MAX_STEPS=100, 경계에 막힌 시도도 포함 |
| RSSI/Noise | REFERENCE_RSSI=-40.0, REFERENCE_DISTANCE=1.0, PATH_LOSS_EXPONENT=2.0, NOISE_STD=2.0 |
| 평가 횟수 | NUM_EVAL_EPISODES=100 기본값; Smoke에서는 동일하게 2회 |
| seed | RANDOM_SEED=42 학습, 평가 환경은 43; Python 전역 환경 RNG와 Agent 전용 RNG 구분 |
| 성공 | get_info()['success']: 실제 Target 좌표 도달 |
| 실패 | Episode 종료까지 Target 미도달; 현재 환경에서는 MAX_STEPS 도달 |

Q-Learning의 정보는 현재 RSSI level·이전 대비 trend·직전 Action으로 이루어진 State다. Agent에 위치/Target 좌표/주변 방향 RSSI를 전달하지 않는다. get_info()의 실제 거리/좌표는 환경 종료 판정·결과 집계용이다. Greedy RSSI가 Action 전에 주변 여러 방향을 관측하면 정보와 측정 예산이 달라진다. 이동 Step과 RSSI 측정 횟수의 별도 기록 여부, 측정/복귀 비용 계산, seed별 평가 Episode 구성은 **팀 합의 필요**이며 이번에 지표나 코드를 추가하지 않았다.

동일 seed 출발이 모든 방식에서 동일 Noise 샘플 또는 동일 Action 난수열을 보장하지는 않는다. 경로·측정 횟수에 따라 RNG 소비 순서가 달라지고, 현재 Q-policy는 학습 후 Agent RNG를 이어받는다. Baseline의 독립 RNG·평가 seed 배정 및 반복 seed 목록은 합의 후 검증해야 한다. Random baseline의 경계 Action 포함 여부, 균등 선택 여부, 학습 없는 평가 절차도 구현 전에 합의해야 한다.

### Experiment가 의존하는 실제 인터페이스

#### Environment — src/environment.py

- 클래스: RSSIGridEnv.
- 생성자: grid_size, max_steps, agent_start, target_position, reference_rssi, reference_distance, path_loss_exponent, noise_std.
- reset() → float: 위치·Step·blocked 초기화와 시작 RSSI 1회 생성.
- move(action) → bool: True는 경계에 막힘, False는 실제 이동. Action 시도당 Step+1·RSSI 1회 생성. 잘못된 Action은 ValueError.
- get_rssi() → Optional[float]: 저장 RSSI 반환; 초기 reset 전에는 None 가능. Experiment는 reset/move 이후 숫자값을 사용한다.
- is_done() → bool: Target 도달 또는 최대 Step 도달.
- get_info() → dict: agent_position, target_position, distance, step_count, max_steps, current_rssi, success, done, blocked. Experiment 결과에 직접 필요한 key는 success, step_count, distance.
- 설계안의 env.step(action) 예시는 현재 구현된 인터페이스가 아니다. 실제 연결은 move/get_rssi/is_done/get_info를 사용한다.

#### State / Reward — src/state_reward.py

- make_state(current_rssi, previous_rssi, previous_action) → (level, trend, previous_action), hashable tuple. 첫 previous_rssi=None은 trend=0, 첫 previous_action은 config.NONE_ACTION=-1.
- calculate_reward(current_rssi, previous_rssi, *, reward_up=None, reward_keep=None, reward_down=None, move_cost=None, terminal_reward=None, success=False, blocked=False, blocked_apply_move_cost=None, blocked_trend_keep=None, blocked_penalty=None) → 수치 Reward.
- Experiment는 같은 new_rssi/current_rssi를 State와 Reward에 전달하며, move() 반환값을 blocked로 전달한다. 다섯 Reward 값은 config 또는 Sweep 인자, 경계 옵션은 config/fallback에서 읽는다.
- RSSI_THRESHOLDS/TREND_THRESHOLD는 config에 없으므로 현재 fallback [-80,-70,-60,-50]/2.0 적용. 실제 신호 기반 보정은 미완료.
- get_representative_rssi, get_rssi_level, get_rssi_trend, num_rssi_levels, suggest_thresholds는 보조 기능이다. Experiment가 직접 호출하는 State/Reward 함수는 make_state/calculate_reward이며, 위치별 다중 측정 처리는 현재 실험 경로에 연결되지 않았다.

#### Q-Learning — src/q_learning.py

- QLearningAgent(num_actions=4, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=None). 실제 build_agent()는 config 값을 전달하므로 초기 epsilon은 0.9다.
- choose_action(state, training=True) → int. 학습은 epsilon-greedy, 평가는 training=False로 Q-value greedy 선택.
- update_q(state, action, reward, next_state, done) → 업데이트한 Q값. done=True이면 bootstrap 없이 terminal Reward만 사용.
- decay_epsilon() → 감소 후 epsilon. 학습 Episode마다 호출하며 평가에서는 호출하지 않는다.
- 평가 미등록 State는 저장하지 않는 임시 zero 배열 사용. Q값 동률 후보는 np.isclose로 찾고 Agent 전용 random.choice로 선택한다.
- save_q_table/load_q_table은 구현되어 있지만 현재 Training/Final/Sweep에서는 호출하지 않는다. 자동 모델 보존·복원 및 RNG checkpoint 재개를 지원하는 실행 흐름으로 간주하지 않는다.
- get_q_values/get_q_value/get_best_action은 기존 _ensure_state 경로로 State를 추가할 수 있다. 평가 연결부를 이 API로 교체하면 Q-table key 불변성부터 다시 검증해야 한다.

### 다른 담당 모듈 변경 시 재검증 계약

| 변경 | 반드시 확인할 계약 | 다시 실행할 테스트 |
| --- | --- | --- |
| Action 번호/방향/행동 수 | 0~3의 의미와 num_actions, 경계 시도, previous_action을 Environment·State·Agent에서 함께 맞춤 | tests/test_environment.py, src/test_q_learning.py, tests/test_experiment.py |
| 환경 생성/reset/move/RSSI | 생성자 인자, reset 숫자 RSSI, move bool blocked, 시도당 Step+1과 RSSI 1회, 조회 시 새 샘플 없음 | test_rssi_sampling, test_grid_boundaries, test_episode_transition, test_training_and_csv |
| 성공/done/get_info | Target 도달 및 Step 한도, success/step_count/distance key, 시작부터 done인 경우 추가 Action 없음, terminal Q update | test_target_reached, test_max_steps, test_initially_done, test_terminal_q_update, test_episode_transition |
| State 구조/구간/Trend | hashable tuple과 NONE_ACTION, 한 transition의 State/Reward가 같은 RSSI 사용, 기존 Q-table 재사용 가능 여부 | src/test_state_reward.py, test_episode_transition, Evaluation key 불변 테스트 |
| Reward 인자/경계/terminal | explicit Reward config, blocked 전달 의미, 성공 terminal 보상, Sweep 조합 간 config 누출 없음 | test_explicit_reward_config_and_terminal_reward, test_blocked_options, tests/test_experiment_sweep.py |
| Agent constructor/update/epsilon | config 전달, done update 공식, decay는 학습만, 매 실행 새 Q-table/epsilon | src/test_q_learning.py, test_build_agent, Sweep 독립성 테스트 |
| Evaluation API/greedy 조회 | training=False 전달, Q-table 값·key 및 epsilon 불변, update/decay 없음, 동일 학습 Agent 유지 | tests/test_experiment.py, tests/test_final_evaluation.py |
| RNG/seed 정책 | 환경 전역 random과 Agent random.Random 분리, 학습42/평가환경43, 동일 설정 재현성 | tests/test_reproducibility.py, test_reproducibility_with_same_seed, Sweep 재현성 테스트 |
| 결과 schema/집계/저장 | 전체 평균에 실패 포함, 성공 0건 None→빈 CSV, 합계/성공률 및 summary/CSV 일치, 덮어쓰기 | test_evaluate_agent_metrics, test_run_sweep_saves_csv, tests/test_final_evaluation.py 및 7절 raw 검산 |

표의 함수별 확인 후 전체 `pytest -q`도 다시 실행한다. 7절 raw 검산은 해당 버전의 CSV를 먼저 생성한 뒤 수행한다. 환경/알고리즘 변경 시 과거 CSV와의 숫자 일치를 요구하지 않고 새 raw와 새 summary/CSV의 일치를 확인해야 한다.

### 완료와 남은 작업

**완료 — 현재 HEAD에서 코드 확인 및 자동 테스트로 검증**

- 실제 QLearningAgent 연결, Episode 학습 반복, 별도 greedy Evaluation, 학습 Agent 재사용.
- Reward 조합별 새 Agent/Q-table/epsilon, 동일한 Reward 외 조건, seed 재현성 및 조합 순서 독립.
- Training/Sweep/Final CSV 저장, Final summary, 성공 0건 처리와 summary/CSV 일치.
- Evaluation Q-value·key·epsilon 불변 및 Q update/decay 제외 검증.
- 현재 HEAD 전체 회귀 테스트, 4단계의 실제 Smoke/raw 통계 검산 기록과 재현 명령, Baseline 현황 및 Integration 계약 정리.

**미구현 또는 향후 합의/실측이 필요한 작업**

- Greedy RSSI 및 독립 Random baseline, 두 방식의 공통 평가 실행 경로.
- 측정/주변 탐색/복귀 예산·동률·경계·미검출 정책과 baseline seed 규칙.
- 최종 Reward 선택에 대한 팀 결정: 현재 Final은 config Reward를 사용하고 자동 선택하지 않는다.
- 실제 Raspberry Pi RSSI 측정·대표값 및 level/trend/noise/model 보정, 실제 환경 성능 평가.
- 장애물 모델, 위치 후보/성공 허용 범위 등 실환경과 Simulation의 판정 차이 정리.
- 장시간 성능 비교와 seed별 분포 검증. 현재 Smoke 결과를 최종 성능으로 간주하지 않는다.

이번 5단계에서는 회귀 오류가 없어 Production·설정·기존 테스트를 수정하지 않았고, baseline/새 지표/새 runner도 구현하지 않았다. 이 목록의 남은 항목은 자동으로 착수할 작업이 아니라 팀 협의 대상으로 남긴다.
