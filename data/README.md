# XJTU data

`xjtu_cccv36/` contains the processed data for 55 batteries in six batches.

- `Batch-*/battery-*.csv`: 36 CC/CV features followed by `capacity` in Ah.
- `metadata/Batch-*/battery-*_cycle_map.csv`: original cycle IDs and processed row mapping.
- `feature_manifest.csv`: feature names, definitions and units.

Batch counts are 8, 15, 8, 8, 8 and 8. The corresponding operating conditions
are 2C, 3C, R2.5, R3, RW and Satellite.
