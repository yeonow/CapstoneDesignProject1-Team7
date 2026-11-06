"""Q-Learning, Greedy Search, Random Search를 같은 8방향 RSSI 환경에서 비교한다.

python -m src.compare_three_algorithms 명령으로 실행한다.
기본 조건: 20×20 Grid, Random Target, 최대 400 Step, Q-Learning 학습 1,000 Episode/Run,
평가 1,000 Episode/알고리즘/Run, 5 Run. 모두 인자로 바꿀 수 있다.

실험 중에만 config의 GRID_SIZE·MAX_STEPS·TARGET_POSITION·TARGET_MODE·TARGET_MIN_DISTANCE를
인자 값으로 바꾸고, 끝나면 원래 값으로 되돌린다.

Seed 규칙은 compare_algorithms와 같다.
- Run r의 학습 Seed: seed + 2*(r-1), 평가 Seed: 학습 Seed + 1
- 평가 Episode e의 Noise Seed: f"{eval_seed}:{e}", Target: targets.evaluation_target(..., eval_seed, e)
- Action RNG: Q-Learning은 tie-break RNG를 eval_seed로 재설정, Random·Greedy는 eval_seed로 생성
따라서 같은 Run·Episode에서 세 알고리즘은 같은 Target과 같은 Noise 시퀀스를 받고,
Q-Learning·Random의 평가 결과는 compare_algorithms와 동일하다.

--algorithms에 q_learning_relative를 추가하면 현재 진행 방향 기준 상대 행동(직진·45°·90°·135° 회전·
되돌아가기, src/relative_actions.py)으로 학습한 Q-Learning도 같은 Seed 규칙으로 함께 비교한다.

평가 중 Agent 좌표를 기록하지만 알고리즘에는 전달하지 않는다(경로 시각화 전용).
저장 파일(<output_dir>/):
- three_algorithm_episodes.csv: Episode별 결과
- three_algorithm_runs.csv: Run·알고리즘별 지표
- three_algorithm_summary.csv: 알고리즘별 Run 평균·표본 표준편차
- three_algorithm_paths.json: 알고리즘별 최적 성공 경로와 대표 경로(실제 좌표)
"""

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path

import config
from src import targets
from src.compare_algorithms import (
    EPISODE_FIELDS,
    SUMMARY_FIELDS,
    _noise_seed,
    _positive_count,
    calculate_metrics,
    train_q_learning_agent,
)
from src.compare_grid_sizes import (
    GRID_SUMMARY_FIELDS,
    aggregate_grid_comparison,
    path_ratio_on_success,
    target_config,
)
from src.experiment import build_environment, run_episode
from src.greedy_agent import GreedySearchAgent
from src.greedy_experiment import run_greedy_episode
from src.random_agent import RandomAgent
from src.random_experiment import run_random_episode
from src.relative_actions import RelativeActionEnv


# q_learning_relative: 현재 진행 방향 기준 상대 행동(src/relative_actions.py)으로 학습한 Q-Learning
ALGORITHMS = ("q_learning", "q_learning_relative", "greedy", "random")
DEFAULT_ALGORITHMS = ("q_learning", "greedy", "random")
Q_ACTION_MODES = {"q_learning": "absolute", "q_learning_relative": "relative"}
DEFAULT_GRID_SIZE = 20
DEFAULT_MAX_STEPS = 400
DEFAULT_TRAIN_EPISODES = 1000
DEFAULT_EVAL_EPISODES = 1000
DEFAULT_RUNS = 5
DEFAULT_TARGET_MODE = "random"

THREE_EPISODE_FIELDS = (*EPISODE_FIELDS, "path_ratio")
THREE_RUN_FIELDS = (
    "target_mode", "grid_size", "max_steps", *SUMMARY_FIELDS, "path_ratio_on_success",
)


class PathRecorder:
    """환경을 감싸 reset·move 후 Agent 좌표를 기록한다. 나머지 동작은 원래 환경에 맡긴다.

    경계에 막힌 Action도 같은 좌표를 한 번 더 기록하므로 len(path) == steps + 1이다.
    """

    def __init__(self, env):
        self._env = env
        self.path = []

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        rssi = self._env.reset(*args, **kwargs)
        self.path = [self._env.agent_position]
        return rssi

    def move(self, action):
        blocked = self._env.move(action)
        self.path.append(self._env.agent_position)
        return blocked


@contextmanager
def experiment_config(grid_size, max_steps, target_mode=None, min_distance=None):
    """블록 안에서만 config의 Grid·Step·Target 설정을 바꾸고, 예외가 나도 되돌린다."""
    previous = (config.GRID_SIZE, config.MAX_STEPS, config.TARGET_POSITION)
    config.GRID_SIZE = grid_size
    config.MAX_STEPS = max_steps
    # fixed 모드에서만 쓰이는 고정 Target. 정의가 없으면 시작점 반대쪽 모서리에 둔다.
    config.TARGET_POSITION = config.GRID_ENV_SETTINGS.get(grid_size, {}).get(
        "target_position", (grid_size - 1, grid_size - 1),
    )
    try:
        with target_config(target_mode, min_distance) as mode:
            yield mode
    finally:
        config.GRID_SIZE, config.MAX_STEPS, config.TARGET_POSITION = previous


def _build_runner(algorithm, env, seed, q_agent):
    """알고리즘별 Episode 실행 함수. Action RNG Seed 규칙은 compare_algorithms와 같다.

    상대 행동 Q-Learning은 환경을 RelativeActionEnv로 감싸고, 그 RNG는 f"heading:{seed}"로 만든다.
    """
    if algorithm in Q_ACTION_MODES:
        if q_agent is None:
            raise ValueError(f"q_agent is required for {algorithm} evaluation.")
        q_agent.random.seed(seed)
        if Q_ACTION_MODES[algorithm] == "relative":
            env = RelativeActionEnv(env, seed=f"heading:{seed}")
        return lambda **kwargs: run_episode(env, q_agent, training=False, **kwargs)
    if algorithm == "random":
        agent = RandomAgent(seed=seed)
        return lambda **kwargs: run_random_episode(env, agent, **kwargs)
    if algorithm == "greedy":
        agent = GreedySearchAgent(seed=seed)
        return lambda **kwargs: run_greedy_episode(env, agent, **kwargs)
    raise ValueError(f"algorithm must be one of {ALGORITHMS}: {algorithm!r}")


def evaluate_algorithm(algorithm, eval_episodes, seed, q_agent=None, noise_std=None):
    """한 알고리즘을 평가해 Episode별 결과와 실제 이동 경로를 반환한다(학습 없음)."""
    count = _positive_count(eval_episodes, "eval_episodes")
    targets.validate_target_config()
    candidates = targets.target_candidates("eval")
    env = PathRecorder(build_environment(noise_std=noise_std))
    runner = _build_runner(algorithm, env, seed, q_agent)
    results = []
    for episode in range(1, count + 1):
        target_kwargs = {} if candidates is None else {
            "target_position": targets.evaluation_target(candidates, seed, episode),
        }
        with _noise_seed(f"{seed}:{episode}"):
            result = runner(**target_kwargs)
        target = env.target_position
        shortest = targets.shortest_path_steps(env.agent_start, target)
        success = bool(result["success"])
        results.append({
            "episode": episode,
            "success": success,
            "steps": int(result["steps"]),
            "final_distance": float(result["final_distance"]),
            "target_x": target[0],
            "target_y": target[1],
            "shortest_path_steps": shortest,
            "path_ratio": result["steps"] / shortest if success and shortest > 0 else None,
            "path": list(env.path),
        })
    return results


def _path_entry(row):
    return {
        "run": row["run"], "episode": row["episode"],
        "target": [row["target_x"], row["target_y"]],
        "steps": row["steps"], "shortest_path_steps": row["shortest_path_steps"],
        "path_ratio": row["path_ratio"],
        "path": [list(position) for position in row["path"]],
    }


def select_paths(episodes, summaries, algorithms=DEFAULT_ALGORITHMS):
    """알고리즘별 최적 성공 경로와 평균 Path Ratio에 가장 가까운 대표 경로를 고른다.

    최적: Path Ratio가 가장 작은 성공 Episode.
    대표: Path Ratio가 요약의 path_ratio_on_success_mean(Run 평균)에 가장 가까운 성공 Episode.
    동률이면 Step이 적은 것, 그다음 앞선 Run·Episode를 고른다. 성공이 없으면 None.
    """
    mean_by_algorithm = {
        row["algorithm"]: row["path_ratio_on_success_mean"] for row in summaries
    }
    selected = {}
    for algorithm in algorithms:
        successes = [
            row for row in episodes
            if row["algorithm"] == algorithm and row["path_ratio"] is not None
        ]
        mean_ratio = mean_by_algorithm.get(algorithm)
        if not successes or mean_ratio is None:
            selected[algorithm] = {"mean_path_ratio": mean_ratio, "best": None, "representative": None}
            continue
        order = lambda row: (row["steps"], row["run"], row["episode"])
        best = min(successes, key=lambda row: (row["path_ratio"], *order(row)))
        representative = min(
            successes, key=lambda row: (abs(row["path_ratio"] - mean_ratio), *order(row)),
        )
        selected[algorithm] = {
            "mean_path_ratio": mean_ratio,
            "best": _path_entry(best),
            "representative": _path_entry(representative),
        }
    return selected


def save_three_algorithm_results(episodes, run_rows, summaries, paths, output_dir):
    """프로젝트 루트 기준 output_dir에 CSV 세 개와 경로 JSON을 저장한다."""
    root = Path(__file__).resolve().parents[1]
    directory = root / output_dir
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    for filename, rows, fields in (
        ("three_algorithm_episodes.csv", episodes, THREE_EPISODE_FIELDS),
        ("three_algorithm_runs.csv", run_rows, THREE_RUN_FIELDS),
        ("three_algorithm_summary.csv", summaries, GRID_SUMMARY_FIELDS),
    ):
        path = directory / filename
        with path.open("w", newline="", encoding="utf-8") as file:
            # 경로 좌표는 CSV에 넣지 않고 JSON에만 저장한다.
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        saved.append(path)
    path = directory / "three_algorithm_paths.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(paths, file, ensure_ascii=False, indent=1)
    saved.append(path)
    return tuple(saved)


def default_output_dir(target_mode, grid_size, max_steps):
    return Path(config.RESULTS_DIR) / (
        f"three_algorithms_{target_mode}_{grid_size}x{grid_size}_{max_steps}steps"
    )


def compare_three_algorithms(
    grid_size=DEFAULT_GRID_SIZE, max_steps=DEFAULT_MAX_STEPS,
    train_episodes=DEFAULT_TRAIN_EPISODES, eval_episodes=DEFAULT_EVAL_EPISODES,
    runs=DEFAULT_RUNS, seed=None, target_mode=DEFAULT_TARGET_MODE,
    target_min_distance=None, output_dir=None, plot=True, algorithms=DEFAULT_ALGORITHMS,
):
    """Run마다 Q-Learning을 새로 학습하고 알고리즘들을 같은 조건에서 평가·저장한다.

    algorithms에 q_learning_relative를 넣으면 상대 행동 Q-Learning도 같은 학습 Seed로 따로 학습한다.
    """
    algorithms = tuple(algorithms)
    if not algorithms or len(set(algorithms)) != len(algorithms):
        raise ValueError("algorithms must be non-empty and must not contain duplicates.")
    unknown = set(algorithms) - set(ALGORITHMS)
    if unknown:
        raise ValueError(f"algorithms must be in {ALGORITHMS}: {sorted(unknown)}")
    train_episodes = _positive_count(train_episodes, "train_episodes")
    eval_episodes = _positive_count(eval_episodes, "eval_episodes")
    runs = _positive_count(runs, "runs")
    _positive_count(grid_size, "grid_size")
    _positive_count(max_steps, "max_steps")
    seed = config.RANDOM_SEED if seed is None else seed

    with experiment_config(grid_size, max_steps, target_mode, target_min_distance) as mode:
        targets.validate_target_config()
        output_dir = (
            default_output_dir(mode, grid_size, max_steps) if output_dir is None else Path(output_dir)
        )
        print(
            f"=== Grid {grid_size}x{grid_size} | actions={len(config.ACTIONS)} | "
            f"start={config.AGENT_START} | target_mode={mode} | max_steps={max_steps} | "
            f"train={train_episodes} | eval={eval_episodes} | runs={runs} ==="
        )
        episodes, run_rows = [], []
        for run in range(1, runs + 1):
            train_seed = seed + 2 * (run - 1)
            eval_seed = train_seed + 1
            print(f"Run {run}/{runs} | train_seed={train_seed} | eval_seed={eval_seed}")
            q_agents = {
                algorithm: train_q_learning_agent(
                    train_episodes, seed=train_seed, action_mode=Q_ACTION_MODES[algorithm],
                )
                for algorithm in algorithms if algorithm in Q_ACTION_MODES
            }
            for algorithm in algorithms:
                results = evaluate_algorithm(
                    algorithm, eval_episodes, eval_seed, q_agents.get(algorithm),
                )
                metadata = {
                    "run": run, "algorithm": algorithm,
                    "train_seed": train_seed, "eval_seed": eval_seed,
                }
                episodes.extend({**metadata, **row} for row in results)
                run_row = {
                    "target_mode": mode, "grid_size": grid_size, "max_steps": max_steps,
                    **metadata,
                    "train_episodes": train_episodes if algorithm in Q_ACTION_MODES else 0,
                    "eval_episodes": eval_episodes,
                    **calculate_metrics(results),
                    "path_ratio_on_success": path_ratio_on_success(results),
                }
                run_rows.append(run_row)
                print(
                    f"{algorithm} | success_rate={run_row['success_rate']:.2%} | "
                    f"average_steps={run_row['average_steps']:.2f} | "
                    f"path_ratio_on_success={run_row['path_ratio_on_success']} | "
                    f"average_final_distance={run_row['average_final_distance']:.3f}"
                )

    summaries = aggregate_grid_comparison(run_rows)
    paths = {
        "grid_size": grid_size, "start": list(config.AGENT_START),
        "target_mode": mode, "max_steps": max_steps,
        "algorithms": select_paths(episodes, summaries, algorithms),
    }
    saved = save_three_algorithm_results(episodes, run_rows, summaries, paths, output_dir)
    for row in summaries:
        print(
            f"{row['algorithm']} | runs={row['run_count']} | "
            f"success_rate={row['success_rate_mean']:.2%} | "
            f"average_steps={row['average_steps_mean']:.2f} | "
            f"path_ratio_on_success={row['path_ratio_on_success_mean']} | "
            f"average_final_distance={row['average_final_distance_mean']:.3f}"
        )
    print("Results saved to:\n" + "\n".join(str(path) for path in saved))
    if plot:
        from src.plot_three_algorithms import make_three_algorithm_report

        make_three_algorithm_report(Path(__file__).resolve().parents[1] / output_dir)
    return {"episodes": episodes, "runs": run_rows, "summary": summaries,
            "paths": paths, "output_dir": output_dir}


def main():
    parser = argparse.ArgumentParser(
        description="Compare Q-Learning, greedy search and random search.",
    )
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--target-mode", choices=targets.TARGET_MODES, default=DEFAULT_TARGET_MODE)
    parser.add_argument("--target-min-distance", type=int, default=config.TARGET_MIN_DISTANCE)
    parser.add_argument(
        "--algorithms", nargs="+", choices=ALGORITHMS, default=list(DEFAULT_ALGORITHMS),
        help="q_learning_relative: Q-Learning with heading-relative actions.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip the figures.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for name in ("grid_size", "max_steps", "train_episodes", "eval_episodes", "runs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0")
    compare_three_algorithms(
        grid_size=args.grid_size, max_steps=args.max_steps,
        train_episodes=args.train_episodes, eval_episodes=args.eval_episodes,
        runs=args.runs, seed=args.seed, target_mode=args.target_mode,
        target_min_distance=args.target_min_distance,
        output_dir=args.output_dir, plot=not args.no_plot, algorithms=args.algorithms,
    )


if __name__ == "__main__":
    main()
