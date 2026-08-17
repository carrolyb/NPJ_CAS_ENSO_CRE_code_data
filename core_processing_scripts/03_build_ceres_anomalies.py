#!/usr/bin/env python3
"""Build CERES monthly climatology and anomalies."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from enso_cloud.ceres import monthly_anomalies, monthly_climatology, regional_means_from_gridded
from enso_cloud.io_utils import write_csv, write_netcdf
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data_processed/ceres_monthly/ceres_monthly_tropical_pacific.nc"))
    parser.add_argument("--climatology-out", type=Path, default=Path("data_processed/anomalies/ceres_monthly_climatology.nc"))
    parser.add_argument("--anomaly-out", type=Path, default=Path("data_processed/anomalies/ceres_monthly_anomalies.nc"))
    parser.add_argument("--regional-climatology-out", type=Path, default=Path("data_processed/anomalies/ceres_monthly_regional_climatology.nc"))
    parser.add_argument("--regional-anomaly-out", type=Path, default=Path("data_processed/anomalies/ceres_monthly_regional_anomalies.nc"))
    parser.add_argument("--climatology-csv-out", type=Path, default=Path("results/csv/ceres_monthly_climatology.csv"))
    parser.add_argument("--anomaly-csv-out", type=Path, default=Path("results/csv/ceres_monthly_anomalies.csv"))
    parser.add_argument("--regional-climatology-csv-out", type=Path, default=Path("results/csv/ceres_monthly_regional_climatology.csv"))
    parser.add_argument("--regional-anomaly-csv-out", type=Path, default=Path("results/csv/ceres_monthly_regional_anomalies.csv"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ds = xr.open_dataset(args.input)
    climatology = monthly_climatology(ds)
    anomalies = monthly_anomalies(ds, climatology)
    regional_climatology = regional_means_from_gridded(climatology)
    regional_anomalies = regional_means_from_gridded(anomalies)
    write_netcdf(climatology, args.climatology_out, args.overwrite)
    write_netcdf(anomalies, args.anomaly_out, args.overwrite)
    write_netcdf(regional_climatology, args.regional_climatology_out, args.overwrite)
    write_netcdf(regional_anomalies, args.regional_anomaly_out, args.overwrite)
    write_csv(climatology.to_dataframe().reset_index(), args.climatology_csv_out, args.overwrite)
    write_csv(anomalies.to_dataframe().reset_index(), args.anomaly_csv_out, args.overwrite)
    write_csv(regional_climatology.to_dataframe().reset_index(), args.regional_climatology_csv_out, args.overwrite)
    write_csv(regional_anomalies.to_dataframe().reset_index(), args.regional_anomaly_csv_out, args.overwrite)
    print(f"Wrote climatology: {args.climatology_out}")
    print(f"Wrote anomalies: {args.anomaly_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
