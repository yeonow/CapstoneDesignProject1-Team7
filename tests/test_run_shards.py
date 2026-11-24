"""팀 분담 검사 전용 가상 CSV; 실제 실험이나 Seed 생성 코드는 바꾸지 않는다."""

import pytest

from src.check_run_shards import validate_run_shards


def shard(tmp_path, run):
    path = tmp_path / f"worker_{run}.csv"
    header = "run,algorithm,train_seed,eval_seed,eval_episodes\n"
    body = "".join(f"{run},{a},{42+2*(run-1)},{43+2*(run-1)},100\n" for a in ("q_learning", "random"))
    path.write_text(header + body, encoding="utf-8")
    return path


def test_four_workers_with_global_runs_and_seeds(tmp_path):
    paths = [shard(tmp_path, run) for run in range(1, 5)]
    assert validate_run_shards(paths[::-1], expected_runs=4)["runs_per_condition"] == 4
    with pytest.raises(ValueError, match="누락"):
        validate_run_shards(paths[:-1], expected_runs=4)


def test_restarted_local_numbering_and_changed_seeds_are_rejected(tmp_path):
    first = shard(tmp_path, 1)
    duplicate = tmp_path / "another_worker.csv"
    duplicate.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="중복"):
        validate_run_shards([first, duplicate])
    duplicate.write_text(first.read_text().replace(",42,43,", ",44,45,"), encoding="utf-8")
    with pytest.raises(ValueError, match="Seed"):
        validate_run_shards([duplicate])


def test_unequal_evaluation_counts_and_duplicate_paths(tmp_path):
    path = shard(tmp_path, 1)
    with pytest.raises(ValueError, match="경로"):
        validate_run_shards([path, path])
    path.write_text(path.read_text().replace("random,42,43,100", "random,42,43,99"), encoding="utf-8")
    with pytest.raises(ValueError, match="불균형"):
        validate_run_shards([path])
