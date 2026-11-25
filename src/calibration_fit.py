"""거리별 관측 통계와 식 (1)을 보존하는 종합 수신 기준값 피팅."""
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

from experiments.field_measurement.analysis.analyze_field_data import distance_statistics, path_loss


def rssi_from_rx(rx, noise_floor_dbm):
    # 식 (1)의 역방향은 signal+noise 전력 합과 같다. logaddexp로 overflow를 피한다.
    factor = math.log(10) / 10
    return np.logaddexp(np.asarray(rx) * factor, noise_floor_dbm * factor) / factor


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["distance_m"]].append(row["rssi_dbm"])
    groups = dict(sorted(groups.items()))
    statistics = list(distance_statistics(groups))
    # 동일 지점 반복의 분산만 합친 값이다. 공간적 shadow fading 추정치가 아니다.
    within_sd = math.sqrt(sum((len(v) - 1) * np.var(v, ddof=1) for v in groups.values())
                          / (len(rows) - len(groups)))
    return groups, statistics, within_sd


def fit_measurements(rows, noise_floor_dbm=-90.0):
    """관측 RSSI 오차를 최소화한다. 송신 전력/eta를 각각 추정하지 않는다.

    P_rx(d)=A-10*n*log10(d/1m)을 식 (1)에 넣고 RSSI 공간에서 최소제곱.
    식 (2)의 주파수/송신 전력/추가 손실은 1m 종합 기준 A로 흡수된다.
    Noise floor는 측정되지 않았으므로 기존 -90 dBm 가정에 조건부인 추정이다.
    """
    if not math.isfinite(noise_floor_dbm):
        raise ValueError("Noise floor must be finite")
    groups, statistics, within_sd = summarize(rows)
    simple, _ = path_loss(groups)  # 기존 분석의 순수 RSSI log-distance 결과도 함께 보고
    distances = np.array([r["distance_m"] for r in rows])
    observed = np.array([r["rssi_dbm"] for r in rows])
    if np.any(observed <= noise_floor_dbm):
        raise ValueError("Observed RSSI must exceed assumed noise floor")
    def residual(params):
        rx = params[0] - 10 * params[1] * np.log10(distances)
        return rssi_from_rx(rx, noise_floor_dbm) - observed
    fitted = least_squares(residual, [simple["reference_rssi_dbm"], simple["exponent"]],
                           bounds=([-np.inf, 1e-9], [np.inf, np.inf]),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        raise ValueError("Channel calibration failed to converge")
    errors = residual(fitted.x)
    return {
        "reference_rx_power_dbm": float(fitted.x[0]), "path_loss_exponent": float(fitted.x[1]),
        "reference_distance_m": 1.0, "noise_floor_dbm": noise_floor_dbm,
        "sample_count": len(rows), "rmse_db": float(np.sqrt(np.mean(errors ** 2))),
        "r_squared": float(1 - sum(errors ** 2) / sum((observed - observed.mean()) ** 2)),
        "within_distance_repeat_sd_db": within_sd, "distance_statistics": statistics,
        "direct_rssi_log_distance": simple,
        "limits": "Conditional on noise floor; repeated medians, not independent spatial fading. "
                  "Aggregate reference does not identify tx power, eta or multipath separately. "
                  "Valid observations 1-15m; 20m disconnected and unobserved. "
                  "IID confidence interval in direct fit is descriptive only.",
    }
