"""기존 비교 CSV를 검증하고 Run 단위 통계·논문용 표·한국어 보고서를 생성한다.

원본 결과를 수정하거나 실험을 자동 실행하지 않는다. 정합성이 깨진 데이터셋은
전체를 분석에서 제외하며, 기본 비교와 Noise 비교의 검증 상태를 별도로 보고한다.
"""

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import fmean, stdev

import config
from src.compare_algorithms import EPISODE_FIELDS, SUMMARY_FIELDS, calculate_metrics
from src.compare_noise_sweep import (
    METRICS, NOISE_EPISODE_FIELDS, NOISE_RUN_FIELDS, NOISE_SUMMARY_FIELDS,
    aggregate_noise_comparison,
)


PAPER_METRICS = ("success_rate_percent", *METRICS[1:])
STAT_FIELDS = ("mean", "std", "ci95_low", "ci95_high", "valid_run_count")
PERFORMANCE_FIELDS = (
    "noise_std", "algorithm", "run_count", "train_episodes", "eval_episodes",
    "success_count_total", "failure_count_total",
    *(f"{metric}_{field}" for metric in PAPER_METRICS for field in STAT_FIELDS),
)
DIFFERENCE_FIELDS = (
    "dataset", "noise_std", "metric", "unit", "direction",
    "q_success_count", "random_success_count", *STAT_FIELDS,
)
PAIR_FIELDS = (
    "dataset", "noise_std", "run", "train_seed", "eval_seed", "metric", "unit",
    "q_value", "random_value", "difference",
)
SOURCE_FILES = (
    "algorithm_comparison_episodes.csv", "algorithm_comparison_summary.csv",
    "noise_comparison_episodes.csv", "noise_comparison_runs.csv", "noise_comparison_summary.csv",
)


@lru_cache(maxsize=None)
def student_t_critical(df):
    """정수 자유도의 t 분포 CDF를 역산해 양측 95% 임계값을 계산한다.

    t=sqrt(df)*tan(theta) 치환으로 밀도 적분을 cos(theta)**(df-1)의
    적분으로 바꾸고, 차수별 점화식과 이분법을 사용한다. 정규분포 근사나
    임계값 표 보간을 사용하지 않으며, 추가 통계 패키지에 의존하지 않는다.
    """
    if isinstance(df, bool) or not isinstance(df, int) or df < 1:
        raise ValueError("자유도는 1 이상의 정수여야 합니다.")
    factor = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(math.pi)

    def cdf(value):
        angle = math.atan(value / math.sqrt(df))
        sine, cosine = math.sin(angle), math.cos(angle)
        power = df - 1
        integral = angle if power % 2 == 0 else sine
        for order in range(2 if power % 2 == 0 else 3, power + 1, 2):
            integral = (sine * cosine ** (order - 1) + (order - 1) * integral) / order
        return 0.5 + factor * integral

    low, high = 0.0, 1.0
    while cdf(high) < 0.975:
        high *= 2
    for _ in range(70):
        middle = (low + high) / 2
        if cdf(middle) < 0.975:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def run_statistics(values):
    """None을 제외한 독립 Run의 평균·표본 표준편차·t 신뢰구간을 계산한다."""
    values = [value for value in values if value is not None]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("통계 입력에 NaN 또는 무한대가 있습니다.")
    count = len(values)
    mean = fmean(values) if count else None
    sd = stdev(values) if count >= 2 else None
    margin = student_t_critical(count - 1) * sd / math.sqrt(count) if count >= 2 else None
    return {
        "mean": mean, "std": sd,
        "ci95_low": mean - margin if margin is not None else None,
        "ci95_high": mean + margin if margin is not None else None,
        "valid_run_count": count,
    }


def read_result_csv(path, required_fields):
    """필수 열과 자료형을 확인한다. 결측 성공 Step 지표만 빈 값을 허용한다."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        missing = set(required_fields) - set(fields)
        if missing or len(fields) != len(set(fields)):
            raise ValueError(f"{path.name}: 필수 열 누락 또는 중복 열: {sorted(missing)}")
        result = []
        for line, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"{path.name}:{line}: 열 개수가 일치하지 않습니다.")
            row = {}
            for key in required_fields:
                value = raw[key]
                if value == "" and key.startswith("average_steps_on_success"):
                    row[key] = None
                elif key == "algorithm":
                    if value not in ("q_learning", "random"):
                        raise ValueError(f"{path.name}:{line}: 알 수 없는 algorithm={value}")
                    row[key] = value
                elif key == "success":
                    if value not in ("True", "False"):
                        raise ValueError(f"{path.name}:{line}: success는 True/False여야 합니다.")
                    row[key] = value == "True"
                else:
                    try:
                        number = float(value)
                        if not math.isfinite(number) or number < 0:
                            raise ValueError
                        integer = key in {
                            "run", "episode", "steps", "train_seed", "eval_seed",
                            "train_episodes", "eval_episodes", "success_count", "failure_count",
                        } or key.endswith("run_count")
                        if integer and not number.is_integer():
                            raise ValueError
                        row[key] = int(number) if integer else number
                    except ValueError:
                        raise ValueError(f"{path.name}:{line}: {key} 값이 잘못되었습니다: {value!r}") from None
            result.append(row)
    if not result:
        raise ValueError(f"{path.name}: 데이터 행이 없습니다.")
    return result


def _same_number(left, right):
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def validate_dataset(input_dir, noise=False):
    """Episode→Run→Summary를 대조하고, 검증된 데이터와 모든 발견 오류를 반환한다."""
    input_dir = Path(input_dir)
    status = {"dataset": "noise" if noise else "algorithm", "valid": False,
              "errors": [], "warnings": [], "episodes": [], "runs": [], "summary": []}
    errors = status["errors"]

    def require(condition, message):
        if not condition:
            errors.append(message)

    try:
        episodes = read_result_csv(input_dir / (
            "noise_comparison_episodes.csv" if noise else "algorithm_comparison_episodes.csv"
        ), NOISE_EPISODE_FIELDS if noise else EPISODE_FIELDS)
        runs = read_result_csv(input_dir / (
            "noise_comparison_runs.csv" if noise else "algorithm_comparison_summary.csv"
        ), NOISE_RUN_FIELDS if noise else SUMMARY_FIELDS)
        summaries = read_result_csv(input_dir / "noise_comparison_summary.csv", NOISE_SUMMARY_FIELDS) if noise else []
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        return status

    def group_key(row):
        return (row.get("noise_std"), row["run"], row["algorithm"])

    grouped, run_map = defaultdict(list), {}
    episode_keys = set()
    for row in episodes:
        key = (*group_key(row), row["episode"])
        require(key not in episode_keys, f"중복 Episode: {key}")
        episode_keys.add(key)
        grouped[group_key(row)].append(row)
        require(row["run"] >= 1 and row["episode"] >= 1, f"Run/Episode 번호 오류: {key}")
        require(row["success"] == (row["final_distance"] == 0), f"성공 여부와 최종 거리 불일치: {key}")
    for row in runs:
        key = group_key(row)
        require(key not in run_map, f"중복 Run: {key}")
        run_map[key] = row
        require(row["run"] >= 1 and row["eval_episodes"] >= 1, f"Run/Episode 수 오류: {key}")
        require(row["train_episodes"] > 0 if row["algorithm"] == "q_learning" else row["train_episodes"] == 0,
                f"학습 Episode 수 오류: {key}")
        require(row["eval_seed"] == row["train_seed"] + 1 and row["train_seed"] <= 2**32 - 1,
                f"학습/평가 Seed 규칙 오류: {key}")
        require(row["success_count"] + row["failure_count"] == row["eval_episodes"], f"성공·실패 합 오류: {key}")
        require(0 <= row["success_rate"] <= 1, f"성공률 범위 오류: {key}")
        rows = grouped.get(key, [])
        require(len(rows) == row["eval_episodes"], f"Run별 Episode 수 불일치: {key}")
        require({ep["episode"] for ep in rows} == set(range(1, row["eval_episodes"] + 1)),
                f"Episode 번호 누락: {key}")
        require(all((ep["train_seed"], ep["eval_seed"]) == (row["train_seed"], row["eval_seed"])
                    for ep in rows), f"Episode/Run Seed 불일치: {key}")
        if rows:
            recomputed = calculate_metrics(rows)
            for metric, value in recomputed.items():
                require(_same_number(row[metric], value), f"재계산 결과 불일치: {key} / {metric}")
    require(set(grouped) == set(run_map), "Episode와 Run 결과의 조합이 일치하지 않습니다.")
    noise_values = {row.get("noise_std") for row in runs}
    seed_by_run = {}
    for level in noise_values:
        level_runs = [row for row in runs if row.get("noise_std") == level]
        level_episodes = [row for row in episodes if row.get("noise_std") == level]
        failure_steps = {row["steps"] for row in level_episodes if not row["success"]}
        # 기존 환경은 실패 시 최대 Step에서 끝나므로 같은 조건의 실패 Step은 같아야 한다.
        require(len(failure_steps) <= 1, f"동일 환경의 실패 종료 Step 불일치: noise={level}")
        if len(failure_steps) == 1:
            inferred_limit = next(iter(failure_steps))
            require(all(row["steps"] <= inferred_limit for row in level_episodes),
                    f"실패 종료 Step으로 확인한 상한 초과: noise={level}")
        ids = {row["run"] for row in level_runs}
        require(ids == set(range(1, max(ids) + 1)), f"Run 번호 누락: noise={level}")
        for algorithm in ("q_learning", "random"):
            subset = [row for row in level_runs if row["algorithm"] == algorithm]
            require({row["run"] for row in subset} == ids, f"알고리즘 Run 누락: {level}/{algorithm}")
            require(len({row["train_seed"] for row in subset}) == len(subset), f"반복 학습 Seed 중복: {level}/{algorithm}")
            require(len({(row["train_episodes"], row["eval_episodes"]) for row in subset}) == 1,
                    f"Run 간 Episode 수 불일치: {level}/{algorithm}")
        for run in ids:
            q, r = run_map.get((level, run, "q_learning")), run_map.get((level, run, "random"))
            if q and r:
                metadata = (q["train_seed"], q["eval_seed"], q["eval_episodes"])
                require(metadata == (r["train_seed"], r["eval_seed"], r["eval_episodes"]), f"대응 Run의 평가 조건 불일치: {level}/{run}")
                require(run not in seed_by_run or seed_by_run[run] == metadata, f"Noise 수준 간 Seed/평가 횟수 변경: run={run}")
                seed_by_run[run] = metadata
    if not noise:
        require(len({row["train_seed"] - 2 * (row["run"] - 1) for row in runs}) == 1,
                "기본 비교의 Run별 Seed 증가 규칙이 일치하지 않습니다.")
    else:
        require(all({row["run"] for row in runs if row["noise_std"] == level} == set(seed_by_run)
                    for level in noise_values), "Noise 수준 간 Run이 누락되었습니다.")
        if not errors:
            expected = aggregate_noise_comparison(runs)
            summary_map = {(row["noise_std"], row["algorithm"]): row for row in summaries}
            require(len(summary_map) == len(summaries), "Noise Summary 중복 행이 있습니다.")
            require(set(summary_map) == {(row["noise_std"], row["algorithm"]) for row in expected}, "Noise Summary 조합이 누락되었습니다.")
            for row in expected:
                stored = summary_map.get((row["noise_std"], row["algorithm"]))
                if stored:
                    for field, value in row.items():
                        if field != "algorithm":
                            require(_same_number(stored[field], value), f"Noise Summary 재계산 불일치: {row['noise_std']}/{row['algorithm']}/{field}")
        baseline = {}
        for row in episodes:
            if row["algorithm"] == "random":
                key = (row["run"], row["episode"])
                outcome = (row["success"], row["steps"], row["final_distance"])
                require(key not in baseline or baseline[key] == outcome, f"Random 결과의 Noise 독립성 위반: {key}")
                baseline[key] = outcome
        status["warnings"].append("Random의 성공·Step·최종 거리 독립성은 검사하지만, CSV에 경로·RSSI가 없어 경로 및 측정값 자체는 재검증할 수 없습니다.")
    status["valid"] = not errors
    if not errors:
        status.update(episodes=episodes, runs=runs, summary=summaries)
    return status


def summarize_runs(runs):
    """알고리즘·Noise별 독립 Run 통계를 계산하며 성공률은 % 단위로 저장한다."""
    grouped = defaultdict(list)
    for row in runs:
        grouped[(row.get("noise_std"), row["algorithm"])].append(row)
    result = []
    for (noise, algorithm), rows in sorted(grouped.items()):
        summary = {
            "noise_std": noise, "algorithm": algorithm, "run_count": len(rows),
            "train_episodes": rows[0]["train_episodes"], "eval_episodes": rows[0]["eval_episodes"],
            "success_count_total": sum(row["success_count"] for row in rows),
            "failure_count_total": sum(row["failure_count"] for row in rows),
        }
        for metric in METRICS:
            name = "success_rate_percent" if metric == "success_rate" else metric
            values = [row[metric] * 100 if metric == "success_rate" else row[metric] for row in rows]
            summary.update({f"{name}_{key}": value for key, value in run_statistics(values).items()})
        result.append(summary)
    return result


def paired_comparisons(runs, dataset):
    """Run·Seed·평가 수가 확인된 쌍만 Q−Random으로 비교한다."""
    grouped = defaultdict(dict)
    for row in runs:
        key = (row.get("noise_std"), row["run"])
        if row["algorithm"] in grouped[key]:
            raise ValueError("대응 분석 입력에 중복 Run이 있습니다.")
        grouped[key][row["algorithm"]] = row
    pairs, differences = [], []
    valid = defaultdict(list)
    for (noise, run), algorithms in sorted(grouped.items()):
        q, r = algorithms.get("q_learning"), algorithms.get("random")
        if not q or not r or any(q[key] != r[key] for key in ("train_seed", "eval_seed", "eval_episodes")):
            continue
        for metric in METRICS:
            if q[metric] is None or r[metric] is None:
                continue
            scale = 100 if metric == "success_rate" else 1
            row = {
                "dataset": dataset, "noise_std": noise, "run": run,
                "train_seed": q["train_seed"], "eval_seed": q["eval_seed"],
                "metric": metric, "unit": "percentage points" if scale == 100 else (
                    "grid units" if metric == "average_final_distance" else "episodes" if metric.endswith("count") else "steps"),
                "q_value": q[metric] * scale, "random_value": r[metric] * scale,
                "difference": (q[metric] - r[metric]) * scale,
            }
            pairs.append(row)
            valid[(noise, metric)].append((row, q["success_count"], r["success_count"]))
    for noise in sorted({row.get("noise_std") for row in runs}):
        for metric in METRICS:
            entries = valid[(noise, metric)]
            differences.append({
                "dataset": dataset, "noise_std": noise, "metric": metric,
                "unit": "percentage points" if metric == "success_rate" else (
                    "grid units" if metric == "average_final_distance" else "episodes" if metric.endswith("count") else "steps"),
                "direction": "Q-Learning - Random",
                "q_success_count": sum(entry[1] for entry in entries),
                "random_success_count": sum(entry[2] for entry in entries),
                **run_statistics([entry[0]["difference"] for entry in entries]),
            })
    return differences, pairs


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format(value):
    return "NA" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def performance_markdown(rows):
    """단위·유효 Run 수·신뢰구간을 포함한 간결한 영어 표를 만든다."""
    headers = ["Noise (dB)", "Algorithm", "Runs", "Success / Failure"] + list(PAPER_METRICS[:4])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        cells = [_format(row["noise_std"]), row["algorithm"], str(row["run_count"]),
                 f"{row['success_count_total']} / {row['failure_count_total']}"]
        for metric in PAPER_METRICS[:4]:
            cells.append(f"{_format(row[f'{metric}_mean'])} ± {_format(row[f'{metric}_std'])}; "
                         f"95% CI [{_format(row[f'{metric}_ci95_low'])}, {_format(row[f'{metric}_ci95_high'])}]; "
                         f"n={row[f'{metric}_valid_run_count']}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("\nMean ± sample SD; 95% Student's t CI across independent runs. NA = unavailable. "
                 "Success rate: percent; steps: attempted moves; distance: grid units. "
                 "Success/failure counts are episode totals. Successful-step metrics exclude runs with no successes. "
                 "Noise NA denotes the basic comparison, whose historical noise is not recorded in its CSV.")
    return "\n".join(lines) + "\n"


def differences_markdown(rows):
    """대응 Run의 차이 방향과 지표별 단위를 명시한 표를 만든다."""
    lines = ["| Dataset | Noise (dB) | Metric | Unit | Valid pairs | Q - Random | Sample SD | 95% CI |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append("| " + " | ".join([
            row["dataset"], _format(row["noise_std"]), row["metric"], row["unit"],
            str(row["valid_run_count"]), _format(row["mean"]), _format(row["std"]),
            f"[{_format(row['ci95_low'])}, {_format(row['ci95_high'])}]",
        ]) + " |")
    lines.append("\nAll differences = Q-Learning - Random. Positive success-rate differences favor Q-Learning; "
                 "negative step/distance differences favor Q-Learning. NA = unavailable; fewer than two pairs have no SD/CI.")
    return "\n".join(lines) + "\n"


def write_report(path, validations, algorithm_rows, noise_rows, differences, input_dir):
    """확인된 CSV 수치와 현재 코드 설정의 출처를 구분해 한국어 보고서를 작성한다."""
    lines = ["# 논문용 실험 결과 분석", "", "## 데이터 검증", ""]
    for status in validations:
        lines.append(f"- {status['dataset']}: {'검증 통과, 분석 가능' if status['valid'] else '검증 실패, 데이터셋 전체 분석 제외'}")
        lines.extend(f"  - {message}" for message in status["errors"] + status["warnings"])
        if status["valid"]:
            lines.append(f"  - Episode {len(status['episodes'])}행, Run {len(status['runs'])}행 검증 완료.")
    lines += ["", "원본 CSV의 필수 열, 자료형, 결측값, 중복, Episode 수, 성공·실패 합, 거리, Seed, 지표 재계산 및 Noise Summary를 검사했다.",
              "오류가 있는 행만 선택적으로 제거하거나 결과를 보정하지 않았다. 검증 통과는 CSV 내부 정합성을 뜻하며 실제 실행 이력의 인증은 아니다.",
              "", "## Experimental Setup", "",
              "다음은 **현재 config.py**에서 확인한 값이다. 원본 CSV에는 전체 환경 스냅샷이 없으므로 생성 당시 설정과 동일함을 입증할 수 없다.",
              f"- Grid: {config.GRID_SIZE}×{config.GRID_SIZE}; 시작 위치: {config.AGENT_START}; Target: {config.TARGET_POSITION}; 최대 Step: {config.MAX_STEPS}.",
              f"- RSSI 모델: Log-distance path loss; 기준 RSSI {config.REFERENCE_RSSI}, 기준 거리 {config.REFERENCE_DISTANCE}, 경로 손실 지수 {config.PATH_LOSS_EXPONENT}; 현재 기본 Noise SD {config.NOISE_STD} dB.",
              "- 현재 실행 코드: Q-Learning은 Noise별 새 Q-Table 학습 후 training=False Greedy 평가, Random은 RSSI와 무관한 4방향 균등 선택이다.",
              "- 학습/평가 환경은 분리되며 평가 Noise Seed는 '<eval_seed>:<episode>', Action RNG는 독립적이다. 과거 경로·RSSI·Q-Table은 CSV에 저장되어 있지 않다."]
    for status in validations:
        if status["valid"]:
            metadata = sorted({(row["run"], row["train_seed"], row["eval_seed"]) for row in status["runs"]})
            counts = sorted({(row["algorithm"], row["train_episodes"], row["eval_episodes"]) for row in status["runs"]})
            lines += [f"- CSV {status['dataset']} Run/train_seed/eval_seed: {metadata}.",
                      f"- CSV {status['dataset']} algorithm/train_episodes/eval_episodes: {counts}."]
    lines += ["", "## Evaluation Metrics", "",
              "Success Rate는 성공 횟수/평가 Episode 수이며 표에는 %로 표시한다. Average Steps는 실패와 경계에 막힌 시도를 포함한 실제 Step의 평균이다.",
              "Average Steps on Success는 성공 Episode만의 평균이며 성공이 없는 Run은 결측값이다. Average Final Distance는 종료 시 Ground Truth 거리의 평균이다.",
              "Success/Failure Count는 실제 Episode 수다. Reward는 알고리즘 간 비교 지표에서 제외한다.",
              "", "## Overall Performance Comparison", "", performance_markdown(algorithm_rows)]
    if not algorithm_rows:
        lines.append("기본 비교 데이터는 분석 불가능하므로 수치를 제시하지 않는다.")
    else:
        by_algorithm = {row["algorithm"]: row for row in algorithm_rows}
        q, r = by_algorithm["q_learning"], by_algorithm["random"]
        lines.append(
            f"기본 비교의 성공률 차이(Q−Random)는 {q['success_rate_percent_mean'] - r['success_rate_percent_mean']:.3f}%p, "
            f"평균 이동 횟수 차이(Random−Q)는 {r['average_steps_mean'] - q['average_steps_mean']:.3f} Steps, "
            f"평균 최종 거리 차이(Random−Q)는 {r['average_final_distance_mean'] - q['average_final_distance_mean']:.3f} grid units다."
        )
        if min(q["run_count"], r["run_count"]) < 2:
            lines.append("기본 비교는 독립 반복 Run이 부족하므로 위 관측 차이에 대한 표본 표준편차나 95% CI를 계산하지 않는다.")
    lines += ["", "## Performance Under Different Noise Levels", "", performance_markdown(noise_rows)]
    if noise_rows:
        q_rows = sorted((row for row in noise_rows if row["algorithm"] == "q_learning"), key=lambda row: row["noise_std"])
        lines.append("관측된 Q-Learning 결과: " + "; ".join(
            f"Noise {row['noise_std']:g} dB: 성공률 {row['success_rate_percent_mean']:.3f}%, 평균 Step {row['average_steps_mean']:.3f}"
            for row in q_rows) + ".")
        means = [row["average_steps_mean"] for row in q_rows]
        if len(means) >= 3 and not (all(a <= b for a, b in zip(means, means[1:])) or all(a >= b for a, b in zip(means, means[1:]))):
            lines.append("평균 Step은 Noise 순서에 따라 단조롭게 변하지 않았다. Noise 증가가 성능 저하를 일으킨다고 단정할 수 없다.")
        lines.append("검증된 Random의 Episode별 성공·Step·최종 거리는 Noise 간 같았다. 경로 자체와 RSSI 값은 CSV만으로 확인할 수 없다.")
    lines += ["", "## Statistical Analysis", "", differences_markdown(differences),
              "모든 통계의 독립 표본 단위는 학습을 새로 수행한 Run이다. Episode 수를 독립 Run 수로 부풀리지 않았다.",
              "평균 신뢰구간은 mean ± t(0.975, n−1) × sample_SD / sqrt(n)이다. 지표별 유효 Run 수를 기록하며 n<2이면 SD와 CI는 결측값이다.",
              "Issue #4 그래프의 오차 막대는 SD이고, 이번 논문용 그래프는 95% CI다. 원본의 단일 Run SD=0 저장 관례는 바꾸지 않고 분석 출력만 결측값으로 처리했다.",
              "대응 차이는 같은 Run·Seed·평가 수를 확인한 쌍의 Q−Random이다. 성공률은 percentage points, Step은 steps, 거리는 grid units이다.",
              "평균 Step은 실패 포함 성공률과 함께 해석해야 하며, 성공 시 Step은 서로 다른 성공 Episode 집합의 조건부 평균이므로 성공 횟수와 함께 읽어야 한다.",
              "Run 간 독립성과 Run 평균의 정규성은 적은 Run으로 충분히 검증할 수 없다. 모든 Run의 성공률이 같으면 t CI 폭은 0이지만 미래 실패 확률이 0이라는 뜻은 아니다.",
              "신뢰구간은 물리적 범위를 벗어나더라도 잘라내지 않았다. 통계적 유의성 검정을 수행하지 않았고 유의하다는 결론을 내리지 않는다.",
              "방법 출처: [NIST: Confidence Limits for the Mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm), "
              "[NIST: t Distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm).",
              "", "## Limitations", "",
              "- 현재 코드의 Grid·시작·Target 위치는 고정되어 있으며 공간 조건 변화는 검증하지 않았다.",
              "- RSSI는 시뮬레이션 모델이며 Gaussian Noise는 실제 신호의 상관성·차폐·다중경로를 단순화한다.",
              "- Random은 RSSI를 활용하지 않으므로 Noise별 동일 결과는 정책 특성과 일치한다.",
              "- Noise별로 새로 학습한 정책의 성능이며, 한 Noise에서 학습한 정책의 다른 Noise 일반화 실험이 아니다.",
              "- 기본 비교와 Noise 비교는 학습 횟수가 다를 수 있으므로 두 실험의 수치를 같은 학습 조건으로 합치지 않는다.",
              "- 반복 Run이 적거나 성공 Run이 부족하면 불확실성·결측 선택의 영향이 크다. 여러 Noise 지점은 같은 Seed를 공유하므로 서로 독립인 표본으로 합치지 않는다.",
              "- 원본 CSV에 전체 환경 설정·코드 버전·실행 시각·경로가 없어 과거 환경 일치와 완전한 재현을 CSV만으로 보증할 수 없다.",
              "", "## Conclusion", "",
              "관측된 차이는 위 표의 수치와 대응 차이에 한정한다. 수치의 방향을 실제 데이터로 확인하며 사전에 특정 알고리즘의 우위를 가정하지 않는다.",
              "환경 메타데이터 보존, 더 많은 독립 Seed, 다른 공간 조건 및 학습/평가 Noise를 교차한 실험이 추가 검증에 필요하다.",
              "", "## 원본 데이터 출처", ""]
    for name in SOURCE_FILES:
        source = Path(input_dir) / name
        try:
            lines.append(f"- `{source}`: SHA-256 `{hashlib.sha256(source.read_bytes()).hexdigest()}`")
        except OSError as exc:
            lines.append(f"- `{source}`: 읽을 수 없음 ({exc})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_paper_results(input_dir=None, output_dir=None):
    """기존 데이터만 분석하고 오류와 분석 가능한 데이터셋을 구분해 저장한다."""
    root = Path(__file__).resolve().parents[1]
    input_dir = root / (config.RESULTS_DIR if input_dir is None else input_dir)
    output_dir = root / (config.RESULTS_DIR if output_dir is None else output_dir)
    validations = [validate_dataset(input_dir), validate_dataset(input_dir, noise=True)]
    algorithm_rows = summarize_runs(validations[0]["runs"])
    noise_rows = summarize_runs(validations[1]["runs"])
    differences, pairs = [], []
    for status in validations:
        stats, details = paired_comparisons(status["runs"], status["dataset"])
        differences.extend(stats)
        pairs.extend(details)
    tables = output_dir / "paper_tables"
    for name, rows, fields, markdown in (
        ("algorithm_performance", algorithm_rows, PERFORMANCE_FIELDS, performance_markdown),
        ("noise_performance", noise_rows, PERFORMANCE_FIELDS, performance_markdown),
        ("algorithm_differences", differences, DIFFERENCE_FIELDS, differences_markdown),
    ):
        _write_csv(tables / f"{name}.csv", rows, fields)
        (tables / f"{name}.md").write_text(markdown(rows), encoding="utf-8")
    _write_csv(tables / "paired_run_differences.csv", pairs, PAIR_FIELDS)
    write_report(output_dir / "paper_analysis_summary.md", validations,
                 algorithm_rows, noise_rows, differences, input_dir)
    for status in validations:
        print(f"{status['dataset']}: {'검증 통과' if status['valid'] else '검증 실패, 분석 제외'}")
        for error in status["errors"]:
            print(f"  {error}")
    if not all(status["valid"] for status in validations):
        print("누락 데이터 재생성 명령(자동 실행하지 않음): python -m src.compare_algorithms / python -m src.compare_noise_sweep")
    print(f"논문용 표: {tables}\n분석 보고서: {output_dir / 'paper_analysis_summary.md'}")
    return {"validation": validations, "algorithm": algorithm_rows, "noise": noise_rows,
            "differences": differences, "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description="Validate and analyze existing experiment CSVs.")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    results = analyze_paper_results(args.input_dir, args.output_dir)
    if not all(status["valid"] for status in results["validation"]):
        parser.exit(1, "일부 데이터셋이 검증에 실패했습니다. 보고서에서 제외 범위를 확인하세요.\n")


if __name__ == "__main__":
    main()
