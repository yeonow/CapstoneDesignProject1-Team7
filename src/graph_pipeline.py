"""학습 없이 원본 CSV 검증 → Run 통계 → 95% CI 그림 → 출처 JSON을 생성한다."""

import argparse
from pathlib import Path

from src import analyze_paper_results as analysis
from src import plot_paper_results as plotting
from src.graph_artifacts import PHASES, ROOT, capture_inputs, reserve_output, write_manifest


DATASETS = {
    "algorithm": ("algorithm_comparison_episodes.csv", "algorithm_comparison_summary.csv"),
    "noise": ("noise_comparison_episodes.csv", "noise_comparison_runs.csv", "noise_comparison_summary.csv"),
}
SOURCE_PATHS = tuple(ROOT / name for name in (
    "src/graph_pipeline.py", "src/graph_artifacts.py", "src/analyze_paper_results.py",
    "src/plot_paper_results.py", "src/plot_noise_comparison.py",
    "src/compare_algorithms.py", "src/compare_noise_sweep.py",
))


def generate_graphs(input_dir, output_root, name, dataset="both", phase="pre_calibration"):
    """검증 실패면 출력하지 않으며 기존 출력 이름은 재사용하지 않는다.

    상대 경로는 CLI 실행 위치 기준이다. 환경 설정은 CSV만으로 인증할 수 없고
    phase도 사용자 선언이다. 실측 보정 결과를 추정하거나 학습하지 않는다.
    """
    if dataset not in (*DATASETS, "both"):
        raise ValueError("dataset must be algorithm, noise, or both")
    selected = tuple(DATASETS) if dataset == "both" else (dataset,)
    input_dir = Path(input_dir).resolve()
    inputs = capture_inputs(input_dir / filename for kind in selected for filename in DATASETS[kind])
    validations = [analysis.validate_dataset(input_dir, noise=kind == "noise") for kind in selected]
    errors = [f"{v['dataset']}: {message}" for v in validations for message in v["errors"]]
    if errors:
        raise ValueError("입력 검증 실패; 출력하지 않음:\n" + "\n".join(errors))
    directory = reserve_output(output_root, phase, name)
    artifacts = []
    for kind, validation in zip(selected, validations):
        rows = analysis.summarize_runs(validation["runs"])
        table = directory / "paper_tables" / f"{kind}_performance.csv"
        analysis._write_csv(table, rows, analysis.PERFORMANCE_FIELDS)
        artifacts.append(table)
        for metric, basic_name, noise_name, title, ylabel in plotting.FIGURES:
            artifacts.extend(plotting.plot_metric(
                rows, metric, title, ylabel,
                directory / "paper_figures" / (noise_name if kind == "noise" else basic_name),
                noise=kind == "noise",
            ))
    manifest = write_manifest(directory, inputs, {
        "dataset": dataset, "phase": phase, "phase_is_user_declared": True,
        "error_bars": "95% Student t CI across independent runs", "dpi": 300,
        "success_rate_unit": "percent", "step_unit": "steps",
        "validation_warnings": {v["dataset"]: v["warnings"] for v in validations},
    }, artifacts, SOURCE_PATHS)
    return {"output_dir": directory, "manifest": manifest, "artifacts": artifacts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", choices=(*DATASETS, "both"), default="both")
    parser.add_argument("--phase", choices=PHASES, default="pre_calibration")
    args = parser.parse_args()
    try:
        result = generate_graphs(args.input_dir, args.output_root, args.name, args.dataset, args.phase)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Graphs: {result['output_dir']}\nProvenance: {result['manifest']}")


if __name__ == "__main__":
    main()
