"""팀원이 나눈 Run CSV의 번호/Seed 충돌을 읽기 전용으로 검사한다. 학습하지 않는다."""

import argparse
from pathlib import Path

from src.analyze_paper_results import read_result_csv


def validate_run_shards(paths, expected_runs=None, base_seed=42):
    """같은 실험 조건의 Q/Random Run CSV를 검사한다. Noise는 조건별로 구분한다.

    Run은 각 파일 내부의 1부터 다시 시작하지 않는 전역 번호여야 한다.
    expected_runs를 지정해야 맨 뒤 Run의 누락까지 검출할 수 있다.
    서로 다른 Grid/Step/Target 조건은 별도 호출한다. CSV에 없는 환경은 추론하지 않는다.
    """
    paths = tuple(Path(path).resolve() for path in paths)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("서로 다른 Run CSV 경로가 필요합니다.")
    if expected_runs is not None and (isinstance(expected_runs, bool) or not isinstance(expected_runs, int) or expected_runs < 1):
        raise ValueError("expected_runs must be a positive integer")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    required = ("run", "algorithm", "train_seed", "eval_seed", "eval_episodes")
    indexed = {}
    for path in paths:
        for row in read_result_csv(path, required, optional_fields=("noise_std",)):
            run = row["run"]
            key = row.get("noise_std"), run, row["algorithm"]
            if key in indexed:
                raise ValueError(f"Run/알고리즘 중복: {key} ({path})")
            if run < 1 or row["eval_episodes"] < 1:
                raise ValueError(f"Run/평가 수 오류: {key}")
            train_seed = base_seed + 2 * (run - 1)
            if train_seed > 2**32 - 1 or (row["train_seed"], row["eval_seed"]) != (train_seed, train_seed + 1):
                raise ValueError(f"전역 Run에 대응하는 Seed 불일치: {key}")
            indexed[key] = row
    levels = {key[0] for key in indexed}
    if None in levels and len(levels) > 1:
        raise ValueError("Noise 열이 있는 CSV와 없는 CSV를 섞을 수 없습니다.")
    all_ids = {key[1] for key in indexed}
    expected = set(range(1, (expected_runs if expected_runs is not None else max(all_ids)) + 1))
    for level in levels:
        for algorithm in ("q_learning", "random"):
            ids = {key[1] for key in indexed if key[0] == level and key[2] == algorithm}
            if ids != expected:
                raise ValueError(f"Run 누락/초과: noise={level}, algorithm={algorithm}")
        for run in expected:
            if indexed[level, run, "q_learning"]["eval_episodes"] != indexed[level, run, "random"]["eval_episodes"]:
                raise ValueError(f"대응 Run 평가 건수 불균형: {level}/{run}")
    if len({row["eval_episodes"] for row in indexed.values()}) != 1:
        raise ValueError("Run/Noise 간 평가 건수가 다릅니다.")
    return {"files": [str(p) for p in paths], "rows": len(indexed),
            "runs_per_condition": len(expected), "conditions": len(levels),
            "base_seed": base_seed, "tail_coverage_checked": expected_runs is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()
    try:
        print(validate_run_shards(args.files, args.expected_runs, args.base_seed))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
