# Figure 03 degC05 Input Data Manifest

- `ceres_monthly_regional_anomalies.nc`: symbolic link to the canonical monthly regional cloud-fraction anomaly dataset.
- `ceres_monthly_regional_climatology.nc`: symbolic link to the canonical monthly regional cloud-fraction climatology dataset.
- `nino34_200207_202302.csv`: local copy of the monthly Nino3.4 index file; this package uses the `nino34_anom` column with a `+/-0.5 C` threshold.

Generated outputs
- `Figure03_degC05_cloud_type_CF_anomaly_42class.csv`: per-cell Delta CF table with bootstrap significance columns.
- `Figure03_degC05_cloud_group_CF_anomaly_summary.csv`: deterministic five-group Delta CF summary.
- `Figure03_degC05_cloud_group_CF_bootstrap.csv`: five-group bootstrap confidence interval summary.
- `Figure03_degC05_validation.txt`: closure, sample-count, definition, and significance checks.
