#!/usr/bin/env python3
"""Compute ENSO decomposition, regression, and composites for CERES cloud products."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from enso_cloud.ceres import regional_means_from_gridded
from enso_cloud.analysis import (
    composite_table,
    compute_decomposition,
    decomposition_closure_table,
    effect_magnitude_table,
    regression_table,
)
from enso_cloud.io_utils import write_csv, write_netcdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anomaly-input", type=Path, default=Path("data_processed/anomalies/ceres_monthly_anomalies.nc"))
    parser.add_argument("--climatology-input", type=Path, default=Path("data_processed/anomalies/ceres_monthly_climatology.nc"))
    parser.add_argument("--nino-input", type=Path, default=Path("data_processed/anomalies/nino34_200207_202302.csv"))
    parser.add_argument("--predictor-column", default="nino34_anom_std_1991_2020")
    parser.add_argument("--thresholds", default="0.5,1.0")
    parser.add_argument("--decomp-out", type=Path, default=Path("data_processed/enso_derived/enso_decomposition.nc"))
    parser.add_argument("--regression-out", type=Path, default=Path("results/tables/enso_regression.csv"))
    parser.add_argument("--composite-out", type=Path, default=Path("results/tables/enso_composites.csv"))
    parser.add_argument("--closure-qc-out", type=Path, default=Path("results/tables/qc_decomposition_closure.csv"))
    parser.add_argument("--magnitude-qc-out", type=Path, default=Path("results/tables/qc_effect_magnitude.csv"))
    parser.add_argument("--closure-tolerance", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    anomalies = xr.open_dataset(args.anomaly_input)
    climatology = xr.open_dataset(args.climatology_input)
    predictor_df = pd.read_csv(args.nino_input, parse_dates=["date"])
    predictor_time = predictor_df["date"].dt.to_period("M").dt.to_timestamp()
    predictor = xr.DataArray(
        predictor_df[args.predictor_column].to_numpy(),
        dims=("time",),
        coords={"time": predictor_time.to_numpy()},
        name=args.predictor_column,
    )

    decomp_gridded = compute_decomposition(anomalies, climatology)
    decomp = regional_means_from_gridded(decomp_gridded)
    regression = regression_table(
        decomp,
        predictor,
        [name for name in decomp.data_vars if name.endswith("_effect") or name.endswith("_contrib_anom")],
    )
    thresholds = [float(value) for value in args.thresholds.split(",")]
    composites = composite_table(
        decomp,
        predictor,
        [name for name in decomp.data_vars if name.endswith("_effect") or name.endswith("_contrib_anom")],
        thresholds,
    )
    closure_qc = decomposition_closure_table(decomp)
    magnitude_qc = effect_magnitude_table(decomp)

    write_netcdf(decomp, args.decomp_out, args.overwrite)
    write_csv(regression, args.regression_out, args.overwrite)
    write_csv(composites, args.composite_out, args.overwrite)
    write_csv(closure_qc, args.closure_qc_out, args.overwrite)
    write_csv(magnitude_qc, args.magnitude_qc_out, args.overwrite)
    if float(closure_qc["max_abs_residual"].max()) > args.closure_tolerance:
        raise RuntimeError(
            f"Decomposition closure check failed: max_abs_residual={float(closure_qc['max_abs_residual'].max()):.6e} "
            f"> tolerance={args.closure_tolerance:.6e}"
        )
    print(f"Wrote decomposition dataset: {args.decomp_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
