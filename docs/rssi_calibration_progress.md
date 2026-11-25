# RSSI calibration progress

Base: origin/main 5b055fb6c9921b546ef1baec0acd38f0d726608f (2026-11-25).
Worktree: CapstoneDesignProject1-Team7-rssi; branch feature/rssi-realworld-calibration.

## Data policy
User-final scenario1_provisional (90) is primary; strict (89) is sensitivity only.
Scenario2 (24 readings/20 differences) is separate Trend validation. No 20m observations.
Original files copied byte-for-byte to data/rssi_calibration_20261125; SHA-256 checked.

## Completed
- Read-only typed measurement loader and original source hash verification.

## Pending
- Counts/duplicates/audit/strict-set validation.
- Fit, uncertainty limitations, baseline comparison, separate Trend analysis.
- Explicit calibrated environment selection with unchanged baseline default.
- Small paired smoke experiment, full regression, final documentation and push.

No calibration parameters are decided yet. Do not infer spatial shadow variance from temporal repeats.
Resume from this Worktree; inspect git log and this file before changing any source.
Each completed unit must pass relevant tests and be committed separately.
# Data validation checkpoint

- Baseline full suite: 357 passed. Loader commit: b8e7124.
- Added exact sample coverage, strict subset, movement ordering and selected audit reconciliation.
- Original bytes are immutable; SHA mismatch stops analysis. No low-quality sample is silently removed.

## Fit checkpoint
- Audit reconciliation committed as d1fc3d0 (8 tests passed).
- Reused existing distance_statistics/path_loss; added RSSI-space fit through equation (1).
- Fits aggregate P_rx at 1m and exponent with assumed noise floor -90 dBm; retains direct RSSI fit for comparison.
- Within-distance repeat SD is descriptive, not a spatial shadow-fading estimate. Runtime noise stays 2 dB.

## Environment checkpoint
- Fit commit 86d9c13: 3 tests passed, including known-curve recovery.
- Optional aggregate reference adds a constant offset to equation (2); baseline arithmetic and equation (1) remain unchanged.
- Movement, state/reward, seeds and target selection unchanged. No NLoS transfer inferred from LoS data.

## Factory checkpoint
- Environment commit 0fd213c: 24 baseline/calibrated tests passed.
- Explicit RSSI_MODEL=calibrated plus RSSI_CALIBRATION_FILE loads validated source-bound JSON.
- Both existing Q-Learning and Random factory calls use this setting; no learning code changed.
- LoS, 1m clamp and fitted noise-floor assumptions are checked; default remains baseline.

## Trend checkpoint
- Factory commit 4468203: 27 tests passed.
- Reuses existing ordered legs/transitions; 20 differences, 2 repeats, approach/away summaries.
- Explores 0/1/2/3/4 dB descriptively; keeps existing 2 dB without tuning State/Reward.

## Analysis checkpoint
- Trend commit 4b97cdf: transition and existing-threshold agreement test passed.
- Added read-only-source analysis command, before/after curves, strict sensitivity, trend CSVs and runtime profile.
- All outputs require a new directory; 20m explicitly unobserved/extrapolated. Seeded simulated SD is separate from measured repeat SD.
- Remaining: actual analysis, paired smoke, full suite, preservation check and push.
