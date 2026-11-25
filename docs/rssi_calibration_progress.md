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
