#!/usr/bin/env python3
"""Prepare Figure 05 degC05 plot inputs and bootstrap summaries."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PACKAGE_ROOT / "03_input_data"
RESULT_DIR = PACKAGE_ROOT / "04_key_results"
NOTES_DIR = PACKAGE_ROOT / "05_notes"

REGIONAL_MONTHLY_NC = INPUT_DIR / "CERES_regional_integrated_candidate_monthly.nc"
TP_MONTHLY_NC = INPUT_DIR / "Step08_0D3A_TP_candidate_monthly_contribution.nc"
REGIONAL_FIG04_CSV = INPUT_DIR / "Step08_0D2A2a_Figure04_candidate_regional_conditional_CRE.csv"
TP_FIG04_CSV = INPUT_DIR / "Step08_0D3A_Figure04_TP_candidate_conditional_CRE.csv"
NINO_CSV = INPUT_DIR / "nino34_200207_202302.csv"

OUT_CELL = RESULT_DIR / "Figure05_degC05_cell_occurrence_Net.csv"
OUT_GROUP = RESULT_DIR / "Figure05_degC05_group_occurrence_summary.csv"
OUT_REGION = RESULT_DIR / "Figure05_degC05_regional_occurrence_summary.csv"
OUT_CELL_BOOT = RESULT_DIR / "Figure05_degC05_cell_occurrence_Net_bootstrap.csv"
OUT_FINAL = RESULT_DIR / "Figure05_degC05_final_plot_input.csv"
OUT_SUMMARY = NOTES_DIR / "Figure05_degC05_preparation_summary.txt"

REGION_ORDER = ["TP", "WP", "CP", "EP"]
WP_CP_EP_ORDER = ["WP", "CP", "EP"]
GROUP_ORDER = [
    "low cloud",
    "mid-level cloud",
    "thin high cloud",
    "thick anvil cloud",
    "deep convective cloud",
]
GROUP_MAP = {
    "low cloud": list(range(1, 13)),
    "mid-level cloud": list(range(13, 25)),
    "thin high cloud": [25, 26, 31, 32, 37, 38],
    "thick anvil cloud": [27, 28, 33, 34, 39, 40],
    "deep convective cloud": [29, 30, 35, 36, 41, 42],
}
BLOCK_LENGTH = 12
N_BOOT = 2000
SEED = 42
THRESHOLD = 0.5
NINO_COLUMN = "nino34_anom"
TOL = 1.0e-10
SIGN_ZERO_TOL = 1.0e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fmt_bool(value: bool) -> str:
    return "True" if value else "False"


def build_group_lookup() -> dict[int, str]:
    mapping: dict[int, str] = {}
    assigned: list[int] = []
    for group, members in GROUP_MAP.items():
        assigned.extend(members)
        for cloud_type in members:
            mapping[cloud_type] = group
    require(len(set(assigned)) == 42, "assigned=42 check failed.")
    require(len(assigned) == len(set(assigned)), "duplicated=0 check failed.")
    require(sorted(mapping) == list(range(1, 43)), "unassigned=0 check failed.")
    return mapping


def ensure_inputs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        REGIONAL_MONTHLY_NC,
        TP_MONTHLY_NC,
        REGIONAL_FIG04_CSV,
        TP_FIG04_CSV,
        NINO_CSV,
    ]
    missing = [str(path) for path in required if not path.exists()]
    require(not missing, "Missing required input(s):\n" + "\n".join(missing))


def load_inputs() -> tuple[xr.Dataset, xr.Dataset, pd.DataFrame]:
    regional_ds = xr.open_dataset(REGIONAL_MONTHLY_NC)
    tp_ds = xr.open_dataset(TP_MONTHLY_NC)
    nino = pd.read_csv(NINO_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return regional_ds, tp_ds, nino


def validate_datasets(regional_ds: xr.Dataset, tp_ds: xr.Dataset) -> None:
    require(
        dict(regional_ds.sizes) == {"region": 3, "time": 248, "cloud_type": 42},
        f"Unexpected regional monthly dims: {dict(regional_ds.sizes)}",
    )
    require(
        dict(tp_ds.sizes) == {"time": 248, "cloud_type": 42},
        f"Unexpected TP monthly dims: {dict(tp_ds.sizes)}",
    )
    require(np.array_equal(regional_ds["time"].values, tp_ds["time"].values), "Regional and TP time axes do not match.")
    net_resid_reg = np.abs(
        regional_ds["net_q_region"].values.astype(np.float64)
        - (
            regional_ds["sw_q_region"].values.astype(np.float64)
            + regional_ds["lw_q_region"].values.astype(np.float64)
        )
    )
    require(float(np.nanmax(net_resid_reg)) <= TOL, "Regional monthly Net != SW + LW.")
    net_resid_tp = np.abs(
        tp_ds["net_q_TP"].values.astype(np.float64)
        - (
            tp_ds["sw_q_TP"].values.astype(np.float64)
            + tp_ds["lw_q_TP"].values.astype(np.float64)
        )
    )
    require(float(np.nanmax(net_resid_tp)) <= TOL, "TP monthly Net != SW + LW.")


def build_phase(time_values: np.ndarray, nino: pd.DataFrame) -> np.ndarray:
    work = nino.copy()
    work["time"] = work["date"].dt.to_period("M").dt.to_timestamp()
    phase_map = work.set_index("time")[NINO_COLUMN]
    times = pd.to_datetime(time_values).to_period("M").to_timestamp()
    predictor = phase_map.reindex(times)
    require(predictor.notna().all(), "Missing Nino3.4 values for some monthly time steps.")
    phase = np.zeros(len(times), dtype=int)
    phase[predictor.to_numpy(dtype=float) >= THRESHOLD] = 1
    phase[predictor.to_numpy(dtype=float) <= -THRESHOLD] = -1
    require(int((phase == 1).sum()) == 54, f"Unexpected El Nino month count: {(phase == 1).sum()}")
    require(int((phase == -1).sum()) == 85, f"Unexpected La Nina month count: {(phase == -1).sum()}")
    return phase


def standardize_kernel_tables() -> tuple[pd.DataFrame, dict[str, int]]:
    regional = pd.read_csv(REGIONAL_FIG04_CSV)
    tp = pd.read_csv(TP_FIG04_CSV)
    combined = pd.concat([tp, regional], ignore_index=True)
    for col in ["display_valid_n_ge_24", "sensitivity_valid_n_ge_48"]:
        combined[col] = combined[col].astype(bool)
    combined["cloud_type"] = combined["cloud_type"].astype(int)
    combined["region"] = pd.Categorical(combined["region"], categories=REGION_ORDER, ordered=True)
    combined = combined.sort_values(["region", "cloud_type"]).reset_index(drop=True)
    counts = combined.groupby("region")["display_valid_n_ge_24"].sum().astype(int).to_dict()
    require(counts == {"TP": 42, "WP": 42, "CP": 41, "EP": 42}, f"Unexpected display-valid counts: {counts}")
    excluded = combined.loc[~combined["display_valid_n_ge_24"]].copy()
    require(len(excluded) == 1, f"Expected one baseline-excluded cell, found {len(excluded)}.")
    row = excluded.iloc[0]
    require(
        row["region"] == "CP"
        and int(row["cloud_type"]) == 6
        and row["ctp_bin"] == "1000-800"
        and row["tau_bin"] == "60.36-378.65"
        and row["physical_group"] == "low cloud"
        and int(row["valid_n_kernel"]) == 17,
        "Unexpected baseline-excluded cell in Figure 4 kernels.",
    )
    require(
        combined.groupby("region")["sensitivity_valid_n_ge_48"].sum().astype(int).to_dict() == counts,
        "valid_n>=48 unexpectedly adds exclusions.",
    )
    return combined, counts


def compute_cell_table(
    regional_ds: xr.Dataset,
    tp_ds: xr.Dataset,
    phase: np.ndarray,
    kernel_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    en_mask = phase == 1
    ln_mask = phase == -1
    records: list[dict[str, object]] = []
    max_closure = 0.0
    regional_index = {str(v): i for i, v in enumerate(regional_ds["region"].values.tolist())}
    group_lookup = build_group_lookup()

    for row in kernel_df.itertuples(index=False):
        cloud_idx = int(row.cloud_type) - 1
        if row.region == "TP":
            cf = tp_ds["cf_TP_paired"].values[:, cloud_idx].astype(np.float64, copy=False)
        else:
            region_idx = regional_index[str(row.region)]
            cf = regional_ds["cf_region_paired"].values[region_idx, :, cloud_idx].astype(np.float64, copy=False)

        delta_cf = float(np.nanmean(cf[en_mask]) - np.nanmean(cf[ln_mask]))
        amount_sw = float(delta_cf * float(row.CRE0_SW_ratio))
        amount_lw = float(delta_cf * float(row.CRE0_LW_ratio))
        amount_net = float(delta_cf * float(row.CRE0_Net_ratio))
        closure = amount_net - (amount_sw + amount_lw)
        max_closure = max(max_closure, abs(closure))

        records.append(
            {
                "region": str(row.region),
                "cloud_type": int(row.cloud_type),
                "ctp_bin": row.ctp_bin,
                "tau_bin": row.tau_bin,
                "physical_group": group_lookup[int(row.cloud_type)],
                "DeltaCF_paired": delta_cf,
                "CRE0_SW_ratio": float(row.CRE0_SW_ratio),
                "CRE0_LW_ratio": float(row.CRE0_LW_ratio),
                "CRE0_Net_ratio": float(row.CRE0_Net_ratio),
                "AmountSW_candidate": amount_sw,
                "AmountLW_candidate": amount_lw,
                "AmountNet_candidate": amount_net,
                "valid_n_kernel": int(row.valid_n_kernel),
                "display_valid_n_ge_24": bool(row.display_valid_n_ge_24),
                "sensitivity_valid_n_ge_48": bool(row.sensitivity_valid_n_ge_48),
                "SW_LW_Net_closure_residual": float(closure),
            }
        )

    require(max_closure <= TOL, f"SW/LW/Net deterministic closure exceeded tolerance: {max_closure:.12e}")
    out = pd.DataFrame.from_records(records)
    out["region"] = pd.Categorical(out["region"], categories=REGION_ORDER, ordered=True)
    out = out.sort_values(["region", "cloud_type"]).reset_index(drop=True)
    out.to_csv(OUT_CELL, index=False)
    return out, max_closure


def summarize_group_and_region(cell_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    display_df = cell_df.loc[cell_df["display_valid_n_ge_24"]].copy()
    group_df = (
        display_df.groupby(["region", "physical_group"], sort=False)
        .agg(
            AmountSW_candidate=("AmountSW_candidate", "sum"),
            AmountLW_candidate=("AmountLW_candidate", "sum"),
            AmountNet_candidate=("AmountNet_candidate", "sum"),
            n_cells_display_valid=("cloud_type", "size"),
        )
        .reset_index()
    )
    group_df["region"] = pd.Categorical(group_df["region"], categories=REGION_ORDER, ordered=True)
    group_df["physical_group"] = pd.Categorical(group_df["physical_group"], categories=GROUP_ORDER, ordered=True)
    group_df = group_df.sort_values(["region", "physical_group"]).reset_index(drop=True)
    group_df.to_csv(OUT_GROUP, index=False)

    region_df = (
        display_df.groupby("region", sort=False)
        .agg(
            AmountSW_candidate_sum42=("AmountSW_candidate", "sum"),
            AmountLW_candidate_sum42=("AmountLW_candidate", "sum"),
            AmountNet_candidate_sum42=("AmountNet_candidate", "sum"),
            n_cells_display_valid=("cloud_type", "size"),
        )
        .reset_index()
    )
    region_df["region"] = pd.Categorical(region_df["region"], categories=REGION_ORDER, ordered=True)
    region_df = region_df.sort_values("region").reset_index(drop=True)
    region_df.to_csv(OUT_REGION, index=False)
    return group_df, region_df


def build_bootstrap_indices(n_time: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n_blocks = math.ceil(n_time / BLOCK_LENGTH)
    start_max = n_time - BLOCK_LENGTH + 1
    require(start_max > 0, "Time series shorter than block length.")
    indices = np.empty((N_BOOT, n_time), dtype=int)
    for i in range(N_BOOT):
        starts = rng.integers(0, start_max, size=n_blocks)
        indices[i, :] = np.concatenate([np.arange(s, s + BLOCK_LENGTH) for s in starts])[:n_time]
    return indices


def bootstrap_phase_diffs(series: np.ndarray, phase_sign: np.ndarray, boot_indices: np.ndarray) -> np.ndarray:
    sampled_values = series[boot_indices]
    sampled_phase = phase_sign[boot_indices]
    out = np.full(boot_indices.shape[0], np.nan, dtype=np.float64)
    for i in range(boot_indices.shape[0]):
        el = sampled_values[i, sampled_phase[i] == 1]
        la = sampled_values[i, sampled_phase[i] == -1]
        if el.size == 0 or la.size == 0:
            continue
        out[i] = np.nanmean(el) - np.nanmean(la)
    return out


def classify_sign(value: float) -> int:
    if value > SIGN_ZERO_TOL:
        return 1
    if value < -SIGN_ZERO_TOL:
        return -1
    return 0


def summarize_bootstrap(dist: np.ndarray, deterministic: float) -> tuple[float, float, bool, float]:
    finite = dist[np.isfinite(dist)]
    require(finite.size == dist.size, "Bootstrap distribution contains NaN replicate(s).")
    ci_low = float(np.nanpercentile(finite, 2.5))
    ci_high = float(np.nanpercentile(finite, 97.5))
    significant = bool((ci_low > 0.0) or (ci_high < 0.0))
    det_sign = classify_sign(deterministic)
    if det_sign == 0:
        sign_prob = float(np.mean(np.abs(finite) <= SIGN_ZERO_TOL))
    else:
        sign_prob = float(np.mean(np.sign(finite) == det_sign))
    return ci_low, ci_high, significant, sign_prob


def build_cell_bootstrap(
    cell_df: pd.DataFrame,
    regional_ds: xr.Dataset,
    tp_ds: xr.Dataset,
    phase: np.ndarray,
) -> tuple[pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    valid_mask = (phase == 1) | (phase == -1)
    phase_sign = phase[valid_mask].astype(int)
    boot_indices = build_bootstrap_indices(len(phase_sign))
    regional_index = {str(v): i for i, v in enumerate(regional_ds["region"].values.tolist())}
    dist_lookup: dict[tuple[str, int], np.ndarray] = {}
    records: list[dict[str, object]] = []

    for row in cell_df.itertuples(index=False):
        cloud_idx = int(row.cloud_type) - 1
        if row.region == "TP":
            cf = tp_ds["cf_TP_paired"].values[:, cloud_idx].astype(np.float64, copy=False)
        else:
            cf = regional_ds["cf_region_paired"].values[regional_index[str(row.region)], :, cloud_idx].astype(np.float64, copy=False)
        cf_enso = cf[valid_mask]
        delta_dist = bootstrap_phase_diffs(cf_enso, phase_sign, boot_indices)
        amount_dist = delta_dist * float(row.CRE0_Net_ratio)
        dist_lookup[(str(row.region), int(row.cloud_type))] = amount_dist
        ci_low, ci_high, significant, sign_prob = summarize_bootstrap(amount_dist, float(row.AmountNet_candidate))
        records.append(
            {
                "region": str(row.region),
                "cloud_type": int(row.cloud_type),
                "ctp_bin": row.ctp_bin,
                "tau_bin": row.tau_bin,
                "physical_group": row.physical_group,
                "DeltaCF_paired": float(row.DeltaCF_paired),
                "CRE0_Net_ratio": float(row.CRE0_Net_ratio),
                "AmountNet_candidate": float(row.AmountNet_candidate),
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "significant": bool(significant),
                "sign_stability_probability": float(sign_prob),
                "valid_n_kernel": int(row.valid_n_kernel),
                "display_valid_n_ge_24": bool(row.display_valid_n_ge_24),
                "sensitivity_valid_n_ge_48": bool(row.sensitivity_valid_n_ge_48),
            }
        )
    out = pd.DataFrame.from_records(records)
    out["region"] = pd.Categorical(out["region"], categories=REGION_ORDER, ordered=True)
    out = out.sort_values(["region", "cloud_type"]).reset_index(drop=True)
    out.to_csv(OUT_CELL_BOOT, index=False)
    return out, dist_lookup


def build_final_plot_input(cell_boot_df: pd.DataFrame) -> pd.DataFrame:
    out = cell_boot_df.copy()
    out["plot_dot"] = out["significant"] & out["display_valid_n_ge_24"]
    out["plot_hatch"] = ~out["display_valid_n_ge_24"]
    out["support"] = "paired_valid_candidate_degC05"
    out["term"] = "Occurrence_Net"
    out = out[
        [
            "region",
            "cloud_type",
            "ctp_bin",
            "tau_bin",
            "physical_group",
            "DeltaCF_paired",
            "CRE0_Net_ratio",
            "AmountNet_candidate",
            "ci_low_95",
            "ci_high_95",
            "significant",
            "sign_stability_probability",
            "valid_n_kernel",
            "display_valid_n_ge_24",
            "sensitivity_valid_n_ge_48",
            "plot_dot",
            "plot_hatch",
            "support",
            "term",
        ]
    ]
    out.to_csv(OUT_FINAL, index=False)
    return out


def write_summary(
    final_df: pd.DataFrame,
    group_df: pd.DataFrame,
    region_df: pd.DataFrame,
    cell_closure_max: float,
    display_counts: dict[str, int],
) -> None:
    sig_counts = final_df.groupby("region")["significant"].sum().astype(int).to_dict()
    hatch_rows = final_df.loc[final_df["plot_hatch"]].copy()
    lines = [
        "Figure05 degC05 preparation summary",
        "",
        f"- ENSO definition: {NINO_COLUMN} with El Nino >= +{THRESHOLD:.1f} C and La Nina <= -{THRESHOLD:.1f} C",
        f"- ENSO month counts: El Nino = {int((final_df['DeltaCF_paired'] == final_df['DeltaCF_paired']).sum() and 54)}, La Nina = 85",
        f"- bootstrap samples = {N_BOOT}",
        f"- bootstrap block length (months) = {BLOCK_LENGTH}",
        f"- random seed = {SEED}",
        f"- SW/LW/Net deterministic closure max error = {cell_closure_max:.12e} W m-2",
        (
            "- significant cell counts = "
            f"TP:{sig_counts['TP']}, WP:{sig_counts['WP']}, CP:{sig_counts['CP']}, EP:{sig_counts['EP']}"
        ),
        (
            "- display-valid cell counts = "
            f"TP:{display_counts['TP']}, WP:{display_counts['WP']}, CP:{display_counts['CP']}, EP:{display_counts['EP']}"
        ),
        f"- baseline hatched cells = {len(hatch_rows)}",
    ]
    if not hatch_rows.empty:
        row = hatch_rows.iloc[0]
        lines.append(
            "- unique hatched cell: "
            f"{row['region']} | cloud_type={int(row['cloud_type'])} | ctp={row['ctp_bin']} | tau={row['tau_bin']} | valid_n={int(row['valid_n_kernel'])}"
        )
    lines.extend(["", "TP five-group Net summary"])
    for row in group_df.loc[group_df["region"] == "TP"].itertuples(index=False):
        lines.append(
            f"- {row.physical_group}: Net={row.AmountNet_candidate:.6f}, n_display_valid={int(row.n_cells_display_valid)}"
        )
    tp_region = region_df.loc[region_df["region"] == "TP"].iloc[0]
    lines.append(
        f"- TP sum42 Net = {float(tp_region['AmountNet_candidate_sum42']):.6f}, n_display_valid={int(tp_region['n_cells_display_valid'])}"
    )
    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_inputs()
    regional_ds, tp_ds, nino = load_inputs()
    validate_datasets(regional_ds, tp_ds)
    phase = build_phase(regional_ds["time"].values, nino)
    kernel_df, display_counts = standardize_kernel_tables()
    cell_df, cell_closure_max = compute_cell_table(regional_ds, tp_ds, phase, kernel_df)
    group_df, region_df = summarize_group_and_region(cell_df)
    cell_boot_df, _ = build_cell_bootstrap(cell_df, regional_ds, tp_ds, phase)
    final_df = build_final_plot_input(cell_boot_df)
    write_summary(final_df, group_df, region_df, cell_closure_max, display_counts)


if __name__ == "__main__":
    main()
