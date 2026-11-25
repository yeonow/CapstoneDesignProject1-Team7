# Final field RSSI measurement data for calibration (2026-11-25)

These are the final measurements provided by the user. File names containing `provisional` are historical file names, not instructions to collect more samples.

- scenario1_provisional.csv: distance-RSSI, 90 rows, 15 measurements at 1,2,3,5,10,15 meters. This is the user-accepted primary set.
- scenario1_strict_provisional.csv: 89 rows, same set excluding a 10-meter observation flagged `low_packet_count` (9 packets). Use only as sensitivity analysis; do not silently replace the primary dataset.
- scenario2.csv: 24 readings, 2 repetitions per direction (approach/away), 6 distances each. 20 adjacent differences total. Keep distinct from scenario1; the two series have different observed RSSI baselines.
- measurement_audit.csv: collection provenance and QC status. Original selected flags must be preserved.

20m was attempted but hotspot connectivity was lost. No valid 20m RSSI measurements are part of the data. Never fabricate or extrapolate 20m as an observation.

Before applying changes to a simulation: inspect the files, identify the exact implementation, preserve historical simulation behavior as a separately selectable baseline, provide model fit diagnostics and limits. Do not silently equate repeated temporal samples with an independent estimate of spatial shadow-fading variance. Do not pool scenario1 and scenario2 into one path-loss fit without an explicitly justified offset or session model.

SHA-256 source file checksums:
- scenario1_provisional.csv: 1d869a86bae7da19feadd2afa4bb9039f5e47b2a279023c3a75ff38d30e40844
- scenario1_strict_provisional.csv: beecdf1aba68311218e980c6d1bb0099b7f41b79f4825ebfd36f48f92bdeba95
- scenario2.csv: db40a41ce0af3002543cb20134b85085dd2f9ea6f6494618352fe898bb0d47aa
- measurement_audit.csv: c5c76d2425db3b0c0bd53db71707d1879122d9d01e66a66e197e7103322bdb33
