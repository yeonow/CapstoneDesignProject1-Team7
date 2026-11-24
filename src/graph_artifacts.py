"""원본을 덮어쓰지 않는 그림 출력 폴더와 입력/코드/옵션 이력 관리."""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PHASES = ("pre_calibration", "post_calibration")
ROOT = Path(__file__).resolve().parents[1]


def capture_inputs(paths):
    """파일명과 내용 해시를 기록한다. 파일 내용이나 Git 상태는 변경하지 않는다."""
    paths = [Path(path).resolve() for path in paths]
    if len(paths) != len(set(paths)):
        raise ValueError("중복 입력 경로")
    records = []
    for path in paths:
        content = path.read_bytes()
        records.append({"path": str(path), "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    return records


def reserve_output(output_root, phase, name):
    """보정 전/후 폴더를 분리하고 새 실행 이름만 허용한다. 기존 빈 폴더도 거부한다."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("name must be a single safe directory name")
    root = Path(output_root).resolve()
    directory = root / phase / name
    # 기존 symlink/junction을 통해 출력이 지정한 root 밖으로 향하지 않게 한다.
    if not directory.resolve().is_relative_to(root):
        raise ValueError("출력 경로가 지정한 root 밖을 가리킵니다.")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _git_identity():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.PIPE).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.PIPE)
        return {"sha": sha, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def write_manifest(directory, inputs, options, artifacts, source_paths=()):
    """생성 후 입력 불변성을 확인하고 JSON을 배타적으로 저장한다.

    Git SHA만으로 미커밋 코드를 식별할 수 없으므로 dirty와 소스 해시도 기록한다.
    phase와 실험 조건은 호출자가 선언한 값이며 실제 실측 여부의 인증은 아니다.
    """
    directory = Path(directory).resolve()
    if capture_inputs(record["path"] for record in inputs) != inputs:
        raise ValueError("그래프 생성 도중 입력 파일이 변경되었습니다.")
    artifacts = [Path(path).resolve() for path in artifacts]
    if any(not path.is_relative_to(directory) for path in artifacts):
        raise ValueError("산출물은 새 출력 폴더 안에 있어야 합니다.")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_identity(), "inputs": inputs, "options": options,
        "source_files": capture_inputs(source_paths), "artifacts": capture_inputs(artifacts),
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = directory / "graph_manifest.json"
    with path.open("x", encoding="utf-8") as file:
        file.write(text)
    return path
