#!/usr/bin/env python3
"""Build monthly CERES cloud-type CRE and contribution products."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from enso_cloud.ceres import (
    build_monthly_products,
    list_ceres_files,
    open_ceres_dataset,
    parse_chunks_arg,
    qc_summary,
    regional_means_from_gridded,
    select_region,
)
from enso_cloud.config import DEFAULT_CERES_ROOT, DEFAULT_END, DEFAULT_START
from enso_cloud.io_utils import write_csv, write_netcdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_CERES_ROOT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--chunks", default="time=31,lat=180,lon=360")
    parser.add_argument("--engine", default="h5netcdf", help="xarray backend engine for NetCDF reads.")
    parser.add_argument("--parallel", action="store_true", help="Enable xarray parallel open_mfdataset reads.")
    parser.add_argument(
        "--gridded-out",
        type=Path,
        default=Path("data_processed/ceres_monthly/ceres_monthly_tropical_pacific.nc"),
    )
    parser.add_argument(
        "--regional-out",
        type=Path,
        default=Path("data_processed/ceres_monthly/ceres_monthly_regional.nc"),
    )
    parser.add_argument(
        "--regional-csv-out",
        type=Path,
        default=Path("results/csv/ceres_monthly_regional.csv"),
    )
    parser.add_argument("--qc-out", type=Path, default=Path("results/qc/ceres_monthly_qc.csv"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = list_ceres_files(args.input, args.start, args.end)
    ds = open_ceres_dataset(
        inventory["path"].tolist(),
        parse_chunks_arg(args.chunks),
        engine=args.engine,
        parallel=args.parallel,
    )
    ds = select_region(ds.sel(time=slice(args.start, args.end)), "tropical_pacific")
    monthly = build_monthly_products(ds)
    regional = regional_means_from_gridded(monthly)
    write_netcdf(monthly, args.gridded_out, args.overwrite)
    write_netcdf(regional, args.regional_out, args.overwrite)
    write_csv(regional.to_dataframe().reset_index(), args.regional_csv_out, args.overwrite)
    qc = qc_summary(monthly, regional)
    write_csv(qc, args.qc_out, args.overwrite)
    print(f"Wrote monthly gridded dataset: {args.gridded_out}")
    print(f"Wrote regional dataset: {args.regional_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
