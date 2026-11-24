# CapstoneDesignProject1-Team7

## 실험 CSV 검증과 그래프 재생성

학습을 다시 실행하지 않고 기존 CSV를 검증해 그래프를 만들 수 있습니다.
원본 보존, 입력 형식, 성공률/성공 시 Step 해석, SD/95% CI 구분과 실행 명령은
[그래프·실험 결과 인수인계 안내](docs/graph_experiment_quality.md)를 참고하세요.

새 안전한 실행기는 `python -m src.graph_pipeline --help`로 확인합니다.
출력은 `pre_calibration/<새 실행 이름>/` 또는 `post_calibration/<새 실행 이름>/`으로
분리되며, 기존 출력 폴더는 덮어쓰지 않습니다.
