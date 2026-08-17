#!/usr/bin/env python3
"""Aggregate 42 CERES cloud types into physical cloud groups."""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from enso_cloud.analysis import aggregate_cloud_groups, load_group_config
from enso_cloud.config import DEFAULT_CLOUD_GROUP_CONFIG
from enso_cloud.io_utils import write_csv, write_netcdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data_processed/ceres_monthly/ceres_monthly_regional.nc"))
    parser.add_argument("--decomp-input", type=Path, default=Path("data_processed/enso_derived/enso_decomposition.nc"))
    parser.add_argument("--group-config", type=Path, default=DEFAULT_CLOUD_GROUP_CONFIG)
    parser.add_argument("--aggregated-out", type=Path, default=Path("data_processed/enso_derived/ceres_physical_groups.nc"))
    parser.add_argument("--aggregated-csv-out", type=Path, default=Path("results/tables/ceres_physical_groups.csv"))
    parser.add_argument("--aggregated-decomp-out", type=Path, default=Path("data_processed/enso_derived/ceres_physical_groups_decomposition.nc"))
    parser.add_argument("--aggregated-decomp-csv-out", type=Path, default=Path("results/tables/ceres_physical_groups_decomposition.csv"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    groups = load_group_config(args.group_config)
    base = xr.open_dataset(args.input).load()
    decomp = xr.open_dataset(args.decomp_input).load()
    aggregated = aggregate_cloud_groups(base, groups)
    aggregated_decomp = aggregate_cloud_groups(decomp, groups)
    write_netcdf(aggregated, args.aggregated_out, args.overwrite)
    write_csv(aggregated.to_dataframe().reset_index(), args.aggregated_csv_out, args.overwrite)
    write_netcdf(aggregated_decomp, args.aggregated_decomp_out, args.overwrite)
    write_csv(aggregated_decomp.to_dataframe().reset_index(), args.aggregated_decomp_csv_out, args.overwrite)
    print(f"Wrote physical cloud group products: {args.aggregated_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
