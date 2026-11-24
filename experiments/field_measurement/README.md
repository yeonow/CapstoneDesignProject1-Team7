# Beacon RSSI 실측 실험

Raspberry Pi 5와 전용 USB Wi-Fi 동글로 스마트폰 2.4 GHz 핫스팟의 Beacon을 수신한다. 이 디렉터리의 도구로 수집, 원본 검증, 데이터 선별, 거리·이동 분석, 기존 Q-Learning 채널 모델과의 비교를 수행한다. 명령은 저장소 루트에서 실행한다.

## 측정 준비

Raspberry Pi OS에서 `python3`, `tshark`, `iw`, `ip`, `nmcli`, `vcgencmd`가 필요하다. `--detach`에는 systemd가 필요하다.

```bash
bash experiments/field_measurement/setup/check_dependencies.sh
sudo python3 -m experiments.field_measurement.setup.radio wlan1
```

- SSH 연결은 이더넷 또는 내장 Wi-Fi를 사용한다. 명령 예시의 `wlan1`을 현장에서 확인한 전용 USB 인터페이스로 바꾼다. USB 버스/장치 번호, 인터페이스의 모드·UP 플래그·주파수, 저전압 플래그를 확인한다. 수집기는 연결된 어댑터의 전환을 거부하고, 한 번에 한 수집 작업만 전용 어댑터를 사용하게 한다.
- 실험용 핫스팟을 2.4 GHz로 설정하고, 예상 채널과 본인 핫스팟의 BSSID를 현장에서 확인한다. 수집 중에는 고정한 BSSID의 Beacon만 캡처한다. 실험 노트에 날짜, Pi와 동글 모델 및 드라이버, 핫스팟 기종·설정, 실제 거리, 송수신기 높이·방향, 장애물, 장비 이동 및 재측정 사유를 기록한다. 식별 정보와 노트는 팀의 비공개 위치에서 관리한다.
- 수집기는 예상 채널을 먼저 검사한다. 불일치하거나 신호가 없으면 후보 채널에서 FCS가 정상이고 DS/HT Primary Channel과 Radiotap 주파수가 일치하는 Beacon을 찾는다. `--scan-channels 1 6 9 11`처럼 후보를 지정할 수 있다. 핫스팟 재시작으로 BSSID가 바뀌면 새 BSSID로 새 작업을 시작한다. 서로 다른 채널의 실측 세트는 각각 보관한다. [802.11 Beacon 필드](https://www.wireshark.org/docs/dfref/w/wlan.html)
- 캡처 중에는 USB 장치번호, 인터페이스 상태, 저전압 상태를 주기적으로 검사한다. 새 저전압 이력 또는 USB 분리·재연결이 감지되면 런을 중단한다. 시작 시 저전압 이력이 이미 있으면 출력으로 알린다. 이력 비트가 이미 설정된 경우 검사 사이에 끝난 저전압은 구분할 수 없으므로 전원 문제를 해결한 뒤 재부팅해서 같은 조건으로 다시 측정한다. [Raspberry Pi 진단 명령](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#vcgencmd)

## 수집 절차

1. **시나리오 1**은 1, 2, 3, 5, 10, 15m에서 거리마다 정지한 상태로 5초 캡처를 15회 실행한다. 한 거리의 작업을 끝내고 다음 거리로 이동한다. 15m 재측정은 장비 위치를 확인할 때까지 잠정으로 표시한다.
2. **시나리오 2**는 10→9→8→7→6→5m 접근과 5→6→7→8→9→10m 이탈을 한 세트로 두 번 반복한다. 각 위치에 도달해 정지한 뒤 5초 1회를 수집한다. 5m 회전점은 접근 마지막과 이탈 처음을 각각 측정한다. 순번은 각 방향에서 1부터 다시 시작한다.

```bash
read -r -p 'Experiment hotspot BSSID: ' hotspot_bssid
out="$HOME/rssi_experiments/session-01"
sudo python3 -m experiments.field_measurement.collect --interface wlan1 \
  --bssid "$hotspot_bssid" --channel 9 --distance 1 --output "$out" --detach

# 시나리오 2, 첫 번째 접근의 첫 지점
sudo python3 -m experiments.field_measurement.collect --interface wlan1 \
  --bssid "$hotspot_bssid" --channel 9 --distance 10 --direction approach \
  --repeat 1 --order 1 --output "$out"
```

- `--detach`는 SSH 터미널과 독립적인 systemd 서비스로 실행한다. 출력된 서비스 이름으로 `sudo systemctl status <unit>`, `sudo journalctl -u <unit>`을 조회한다. 중단은 `sudo systemctl stop <unit>`으로 요청한다. 정상 종료나 종료 요청 시 전용 어댑터를 복원한다. 장비 분리, 강제 종료, 재부팅 뒤에는 다시 인터페이스 상태를 확인한다. [systemd-run](https://www.freedesktop.org/software/systemd/man/latest/systemd-run.html)
- 프레임별로 FCS 검증을 통과하고 지정한 BSSID, Radiotap 주파수, Beacon 채널이 일치하는지 확인한다. 동글이 여러 안테나 값을 기록하면 기존 수집기와 동일하게 첫 번째 값을 사용한다. 5초 캡처에서 적격 Beacon 10개와 첫·끝 적격 프레임 간격 4초를 요구한다. 대표 RSSI는 적격 프레임의 중앙값이다.
- 품질 미달이면 채널을 다시 탐색해 동일 채널에서 한 번 재시도한다. 새 채널이 검출되면 런을 중단하고 원본을 보존한다. 파일은 실제 채널별 디렉터리에 저장한다. `completed.json`이 없는 런은 실패 진단용으로 남으며 새 형식 전처리의 후보에서 제외된다.

## 검증 및 전처리

기존 ZIP 또는 새 수집기의 디렉터리를 입력으로 사용한다. `--channel` 기본값은 기존 실험의 채널 9다. 직접 실행하던 `python experiments/field_measurement/analysis/prepare_field_data.py ...` 명령도 사용할 수 있다.

```powershell
python -m experiments.field_measurement.analysis.prepare_field_data `
  "$env:USERPROFILE\Downloads\rssi_experiments.zip" --channel 9 --15m-run last `
  --output results/rssi-field/prepared-last --verify-captures
python -m experiments.field_measurement.analysis.prepare_field_data `
  "$env:USERPROFILE\Downloads\rssi_experiments.zip" --channel 9 --15m-run first `
  --output results/rssi-field/prepared-first
```

Windows에서 `--verify-captures`를 쓰려면 Wireshark의 TShark를 PATH에 넣는다. 기본 검증은 원본 ZIP CRC, 런별 요약과 총괄 요약의 일치, PCAPNG 파일 존재, 패킷별 FCS·주파수·채널, 프레임 수, 중앙값, 시나리오 순번을 확인한다. `--verify-captures`는 원본 PCAPNG에서도 적격 프레임 번호와 RSSI를 다시 추출하여 CSV와 비교한다.

| 출력 | 내용 |
| --- | --- |
| `scenario1_primary.csv` | 1–10m 적격 표본 |
| `scenario1_provisional.csv` | 15m 선택분을 포함한 계획 표본과 결측 |
| `scenario1_strict_provisional.csv` | 15m 선택분을 포함한 적격 표본 |
| `scenario2.csv` | 각 위치에서 시간순 첫 적격 런 |
| `measurement_audit.csv` | 적격·부적격 시도와 15m 대체 런의 선정 이유 |
| `metadata.json` | 입력 ZIP SHA-256, 채널, 캡처 재디코딩 여부, 15m 선택 |

`--15m-run first|last`는 15m 잠정 민감도 분석의 런을 선택한다. 기존 `field_analysis_prepared.zip`과 비교할 때는 `last`를 사용한다. 15m 재측정 위치가 확인되면 현장 기록과 원본을 함께 검토하고 분석 범위를 다시 결정한다.

## 통계 분석 및 시뮬레이션 비교

분석은 Python 3.10 이상이 필요하다.

```powershell
python -m pip install -r experiments/field_measurement/analysis/requirements.txt
python -m experiments.field_measurement.analysis.analyze_field_data results/rssi-field/prepared-last `
  --output results/rssi-field/primary --seed 42 --draws 1000
python -m experiments.field_measurement.analysis.analyze_field_data results/rssi-field/prepared-last `
  --output results/rssi-field/provisional-last --seed 42 --draws 1000 --include-provisional
python -m experiments.field_measurement.analysis.analyze_field_data results/rssi-field/prepared-first `
  --output results/rssi-field/provisional-first --seed 42 --draws 1000 --include-provisional
```

- **거리·경로 손실.** 거리별 적격 중앙값의 표본 수, 평균, 중앙값, 표준편차, 사분위수, 최솟값과 최댓값을 계산한다. 최소제곱법으로 `RSSI(d) = A − 10n log10(d / 1m)`의 1m 기준 RSSI `A`와 감쇠 지수 `n`을 추정한다. 적합도와 지수의 모형 기반 95% 구간을 JSON에 기록한다.
- **Gaussian 적합성.** 거리별 평균을 뺀 변동의 합동 표준편차와 경로 손실 회귀 잔차를 각각 계산한다. 히스토그램, 정규 Q–Q 그림, Shapiro–Wilk 통계량 및 p값을 제공한다. 같은 거리에서 반복한 표본의 의존성과 RSSI 양자화 때문에 p값은 탐색적으로 해석한다. 이 분석의 변동 단위는 5초 중앙값이며 시뮬레이션의 수신기 noise floor는 별도 설정값이다.
- **이동 방향.** 각 방향 구간 안에서 인접 지점의 `다음 RSSI − 이전 RSSI`를 계산해 프로젝트의 `get_rssi_trend`로 UP·KEEP·DOWN을 분류한다. 기본 임계값은 ±2 dB이며 경계값은 KEEP이다. 접근은 UP, 이탈은 DOWN이 정답이다. 회전점과 반복 간 전이는 제외되어 총 20개 전이를 평가한다.
- **시뮬레이션.** 저장소의 환경 생성기와 `config.py`의 채널 모형을 그대로 사용한다. 거리별 RSSI를 동일한 거리에서 반복 생성하고, 같은 접근·이탈 프로토콜과 동일 Trend 함수로 방향별 정확도 분포를 구한다. 시뮬레이션은 지점마다 단일 추출값을 사용하고 실측값은 5초 중앙값이므로 관측 단위 차이를 감안한다. 보고서의 seed, 반복 수, 채널 파라미터, 입력 해시와 패키지 버전을 재현에 사용한다.

## 테스트와 보관

```powershell
python -m pytest -q -p no:cacheprovider tests src experiments/field_measurement/tests
```

테스트에는 가상 USB 동글·전원 상태, 모니터 복원, Beacon 검증, 채널 변경 시 중단, SSH 독립 실행 인수, 합성 ZIP/디렉터리 전처리부터 PNG 생성까지의 통합 경로를 포함한다.

Pi 현장 검증에서는 실제 USB 동글의 Radiotap/FCS 제공 여부, 모니터 모드 설정·복원, 핫스팟 채널 변경, USB 분리, 저전압 감지, SSH 종료 후 서비스 지속을 각각 확인한다. 실제 결과는 [작업 및 검증 기록](VERIFICATION.md)에 기록한다.

원본 ZIP·PCAPNG, RSSI CSV와 파생 보고서, BSSID, 위치 노트, 네트워크 설정은 팀 비공개 저장소에만 보관한다. 이 문서의 예제 출력 경로인 `results/`는 Git에서 제외된다.
