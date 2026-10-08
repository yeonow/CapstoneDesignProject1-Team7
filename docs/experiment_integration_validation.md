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
