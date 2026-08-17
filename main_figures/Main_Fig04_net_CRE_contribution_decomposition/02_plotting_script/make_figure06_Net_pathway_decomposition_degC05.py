#!/usr/bin/env python3
"""Render Figure 06 under the +/-0.5 C ENSO definition using the Figure06 copy.py layout."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PACKAGE_ROOT / "01_final_figure"
INPUT_DIR = PACKAGE_ROOT / "03_input_data"
RESULT_DIR = PACKAGE_ROOT / "04_key_results"
NOTES_DIR = PACKAGE_ROOT / "05_notes"

REGIONAL_MONTHLY_NC = INPUT_DIR / "CERES_regional_integrated_candidate_monthly.nc"
REGIONAL_FIG04_CSV = INPUT_DIR / "Step08_0D2A2a_Figure04_candidate_regional_conditional_CRE.csv"
NINO_CSV = INPUT_DIR / "nino34_200207_202302.csv"
FIG05_GROUP_CSV = PACKAGE_ROOT.parent / "Figure05_occurrence_mediated_Net_degC05" / "04_key_results" / "Figure05_degC05_group_occurrence_summary.csv"
FIG05_REGION_CSV = PACKAGE_ROOT.parent / "Figure05_occurrence_mediated_Net_degC05" / "04_key_results" / "Figure05_degC05_regional_occurrence_summary.csv"
FIG05_METHOD = PACKAGE_ROOT.parent / "Figure05_occurrence_mediated_Net_degC05" / "05_notes" / "Figure05_degC05_method_and_plot_checks.txt"
FIG02_DIRECT_CSV = PACKAGE_ROOT.parent / "Figure02_direct_regional_CRE_degC05" / "04_key_results" / "Figure02_regional_direct_CRE_summary_degC05.csv"

OUT_PNG = FIG_DIR / "Figure06_Net_pathway_decomposition_degC05.png"
OUT_PDF = FIG_DIR / "Figure06_Net_pathway_decomposition_degC05.pdf"
OUT_PLOT = RESULT_DIR / "Figure06_degC05_plot_data.csv"
OUT_GROUP = RESULT_DIR / "Figure06_degC05_group_summary.csv"
OUT_REGION = RESULT_DIR / "Figure06_degC05_regional_summary.csv"
OUT_DIRECT = RESULT_DIR / "Figure06_degC05_direct_summary.csv"
OUT_CAPTION = NOTES_DIR / "Figure06_degC05_caption.md"
OUT_METHOD = NOTES_DIR / "Figure06_degC05_method_and_plot_checks.txt"
OUT_MANIFEST = NOTES_DIR / "Figure06_degC05_input_data_manifest.md"

REGION_ORDER = ["WP", "CP", "EP"]
REGION_TITLE = {
    "WP": "Western Pacific",
    "CP": "Central Pacific",
    "EP": "Eastern Pacific",
}
PANEL_LABELS = {"WP": "a", "CP": "b", "EP": "c", "REG": "d"}
GROUP_ORDER = [
    "low cloud",
    "mid-level cloud",
    "thin high cloud",
    "thick anvil cloud",
    "deep convective cloud",
]
GROUP_LABEL = {
    "low cloud": "Low",
    "mid-level cloud": "Mid-level",
    "thin high cloud": "Thin high",
    "thick anvil cloud": "Thick anvil",
    "deep convective cloud": "Deep\n convective",
}
FOCUS_GROUPS = ["low cloud", "thin high cloud", "thick anvil cloud", "deep convective cloud"]
TERM_ORDER = ["Occurrence", "Conditional-CRE", "Interaction", "Adjustment", "Total"]
COLOR_OCC = "#4C78A8"
COLOR_ADJ = "#F58518"
DIRECT_COLOR = "#000000"
BLOCK_LENGTH = 12
N_BOOT = 2000
SEED = 42
THRESHOLD = 0.5
NINO_COLUMN = "nino34_anom"
TOL = 1.0e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def ensure_inputs() -> None:
    for path in [FIG_DIR, RESULT_DIR, NOTES_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    required = [
        REGIONAL_MONTHLY_NC,
        REGIONAL_FIG04_CSV,
        NINO_CSV,
        FIG05_GROUP_CSV,
        FIG05_REGION_CSV,
        FIG05_METHOD,
        FIG02_DIRECT_CSV,
    ]
    missing = [str(path) for path in required if not path.exists()]
    require(not missing, "Missing required input(s):\n" + "\n".join(missing))


def build_phase(time_values: np.ndarray, nino: pd.DataFrame) -> np.ndarray:
    work = nino.copy()
    work["time"] = work["date"].dt.to_period("M").dt.to_timestamp()
    phase_map = work.set_index("time")[NINO_COLUMN]
    times = pd.to_datetime(time_values).to_period("M").to_timestamp()
    predictor = phase_map.reindex(times)
    require(predictor.notna().all(), "Missing Nino3.4 values for some monthly time steps.")
    phase = np.zeros(len(times), dtype=np.int8)
    predictor_values = predictor.to_numpy(dtype=float)
    phase[predictor_values >= THRESHOLD] = 1
    phase[predictor_values <= -THRESHOLD] = -1
    require(int((phase == 1).sum()) == 54, f"Unexpected El Nino month count: {(phase == 1).sum()}")
    require(int((phase == -1).sum()) == 85, f"Unexpected La Nina month count: {(phase == -1).sum()}")
    return phase


def load_inputs() -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = xr.open_dataset(REGIONAL_MONTHLY_NC)
    fig04 = pd.read_csv(REGIONAL_FIG04_CSV)
    nino = pd.read_csv(NINO_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    fig05_group = pd.read_csv(FIG05_GROUP_CSV)
    fig05_region = pd.read_csv(FIG05_REGION_CSV)
    fig05_group = fig05_group.loc[fig05_group["region"].isin(REGION_ORDER)].copy()
    fig05_region = fig05_region.loc[fig05_region["region"].isin(REGION_ORDER)].copy()
    return ds, fig04, nino, fig05_group, fig05_region


def validate_inputs(ds: xr.Dataset, fig04: pd.DataFrame, fig05_group: pd.DataFrame, fig05_region: pd.DataFrame) -> None:
    require(dict(ds.sizes) == {"region": 3, "time": 248, "cloud_type": 42}, f"Unexpected dataset shape: {dict(ds.sizes)}")
    require(ds.attrs.get("daytime_based") == "True", "Monthly dataset is not tagged as daytime.")
    require(ds.attrs.get("paired_valid_rule") == "True", "Monthly dataset is not tagged as paired-valid.")
    require(ds.attrs.get("all42_joint_strict_mask") == "False", "Monthly dataset unexpectedly uses all-42 joint strict mask.")
    require(ds.attrs.get("cf_zero_contribution") == "0", "Monthly dataset unexpectedly changed cf==0 handling.")
    require(
        ds.attrs.get("regional_contribution_definition") == "area_mean(gridcell_cf_times_gridcell_CRE)",
        "Unexpected contribution definition in monthly dataset.",
    )
    require(len(fig04) == 126, f"Unexpected Figure 4 row count: {len(fig04)}")
    require(len(fig05_group) == 15, f"Unexpected Figure 5 group row count: {len(fig05_group)}")
    require(len(fig05_region) == 3, f"Unexpected Figure 5 regional row count: {len(fig05_region)}")

    net_resid = np.abs(ds["net_q_region"].values - (ds["sw_q_region"].values + ds["lw_q_region"].values))
    require(float(np.nanmax(net_resid)) <= TOL, "Monthly Net != SW + LW.")

    fig04["display_valid_n_ge_24"] = fig04["display_valid_n_ge_24"].astype(bool)
    counts = fig04.groupby("region")["display_valid_n_ge_24"].sum().astype(int).to_dict()
    require(counts == {"WP": 42, "CP": 41, "EP": 42}, f"Unexpected display-valid counts: {counts}")
    excluded = fig04.loc[~fig04["display_valid_n_ge_24"]].copy()
    require(len(excluded) == 1, f"Expected one baseline-excluded cell, found {len(excluded)}.")
    row = excluded.iloc[0]
    require(
        row["region"] == "CP"
        and int(row["cloud_type"]) == 6
        and row["ctp_bin"] == "1000-800"
        and row["tau_bin"] == "60.36-378.65"
        and row["physical_group"] == "low cloud"
        and int(row["valid_n_kernel"]) == 17,
        "Unexpected baseline-excluded cell.",
    )


def phase_diff(series: np.ndarray, phase_sign: np.ndarray) -> float:
    return float(np.nanmean(series[phase_sign == 1]) - np.nanmean(series[phase_sign == -1]))


def build_monthly_terms(ds: xr.Dataset, fig04: pd.DataFrame, phase: np.ndarray) -> pd.DataFrame:
    valid_phase = (phase == 1) | (phase == -1)
    times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    region_lookup = {str(region): i for i, region in enumerate(ds["region"].values.tolist())}
    records: list[dict[str, object]] = []

    ordered = fig04.copy()
    ordered["region"] = pd.Categorical(ordered["region"], categories=REGION_ORDER, ordered=True)
    ordered = ordered.sort_values(["region", "cloud_type"]).reset_index(drop=True)

    for row in ordered.itertuples(index=False):
        r_idx = region_lookup[str(row.region)]
        c_idx = int(row.cloud_type) - 1
        cf = ds["cf_region_paired"].values[r_idx, :, c_idx].astype(np.float64, copy=False)
        q = ds["net_q_region"].values[r_idx, :, c_idx].astype(np.float64, copy=False)
        cre = ds["net_cre_effective"].values[r_idx, :, c_idx].astype(np.float64, copy=False)
        cf0 = float(row.CF0)
        cre0 = float(row.CRE0_Net_ratio)

        occurrence = np.full(q.shape, np.nan, dtype=np.float64)
        conditional = np.full(q.shape, np.nan, dtype=np.float64)
        interaction = np.full(q.shape, np.nan, dtype=np.float64)
        total = np.full(q.shape, np.nan, dtype=np.float64)

        finite = np.isfinite(cf) & np.isfinite(q)
        zero_case = finite & (cf == 0.0) & (q == 0.0)
        positive_case = finite & (cf > 0.0)
        available = zero_case | positive_case

        occurrence[zero_case] = (cf[zero_case] - cf0) * cre0
        conditional[zero_case] = 0.0
        interaction[zero_case] = 0.0
        total[zero_case] = q[zero_case]

        occurrence[positive_case] = (cf[positive_case] - cf0) * cre0
        conditional[positive_case] = cf0 * (cre[positive_case] - cre0)
        interaction[positive_case] = (cf[positive_case] - cf0) * (cre[positive_case] - cre0)
        total[positive_case] = q[positive_case]
        adjustment = conditional + interaction

        q0 = cf0 * cre0
        monthly_residual = np.full(q.shape, np.nan, dtype=np.float64)
        monthly_residual[available] = (total[available] - q0) - (
            occurrence[available] + conditional[available] + interaction[available]
        )
        require(
            float(np.nanmax(np.abs(monthly_residual))) <= TOL if np.isfinite(monthly_residual).any() else True,
            f"Monthly decomposition closure exceeded tolerance for {row.region} cloud_type={row.cloud_type}.",
        )

        for i, month in enumerate(times):
            if not valid_phase[i]:
                continue
            records.append(
                {
                    "region": str(row.region),
                    "cloud_type": int(row.cloud_type),
                    "physical_group": row.physical_group,
                    "display_valid_n_ge_24": bool(row.display_valid_n_ge_24),
                    "valid_n_kernel": int(row.valid_n_kernel),
                    "month": month,
                    "phase_sign": int(phase[i]),
                    "occurrence_t": occurrence[i],
                    "conditional_cre_t": conditional[i],
                    "interaction_t": interaction[i],
                    "adjustment_t": adjustment[i],
                    "total_t": total[i],
                }
            )
    monthly_df = pd.DataFrame.from_records(records)
    require(len(monthly_df) == 126 * (54 + 85), f"Unexpected monthly-term row count: {len(monthly_df)}")
    return monthly_df


def summarize_deterministic(monthly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    display_df = monthly_df.loc[monthly_df["display_valid_n_ge_24"]].copy()

    group_records: list[dict[str, object]] = []
    for (region, group), sub in display_df.groupby(["region", "physical_group"], sort=False):
        monthly_sum = sub.groupby(["month", "phase_sign"], sort=False)[
            ["occurrence_t", "conditional_cre_t", "interaction_t", "adjustment_t", "total_t"]
        ].sum().reset_index()
        group_records.append(
            {
                "region": region,
                "physical_group": group,
                "delta_occurrence": phase_diff(monthly_sum["occurrence_t"].to_numpy(dtype=float), monthly_sum["phase_sign"].to_numpy(dtype=int)),
                "delta_conditional_cre": phase_diff(monthly_sum["conditional_cre_t"].to_numpy(dtype=float), monthly_sum["phase_sign"].to_numpy(dtype=int)),
                "delta_interaction": phase_diff(monthly_sum["interaction_t"].to_numpy(dtype=float), monthly_sum["phase_sign"].to_numpy(dtype=int)),
                "delta_adjustment": phase_diff(monthly_sum["adjustment_t"].to_numpy(dtype=float), monthly_sum["phase_sign"].to_numpy(dtype=int)),
                "delta_total": phase_diff(monthly_sum["total_t"].to_numpy(dtype=float), monthly_sum["phase_sign"].to_numpy(dtype=int)),
                "n_cells_display_valid": int(sub["cloud_type"].nunique()),
            }
        )
    group_df = pd.DataFrame.from_records(group_records)
    group_df["region"] = pd.Categorical(group_df["region"], categories=REGION_ORDER, ordered=True)
    group_df["physical_group"] = pd.Categorical(group_df["physical_group"], categories=GROUP_ORDER, ordered=True)
    group_df = group_df.sort_values(["region", "physical_group"]).reset_index(drop=True)

    regional_df = (
        group_df.groupby("region", sort=False)[
            ["delta_occurrence", "delta_conditional_cre", "delta_interaction", "delta_adjustment", "delta_total", "n_cells_display_valid"]
        ]
        .sum()
        .reset_index()
        .rename(
            columns={
                "delta_occurrence": "delta_occurrence_sum42",
                "delta_conditional_cre": "delta_conditional_cre_sum42",
                "delta_interaction": "delta_interaction_sum42",
                "delta_adjustment": "delta_adjustment_sum42",
                "delta_total": "delta_total_sum42",
            }
        )
    )
    regional_df["region"] = pd.Categorical(regional_df["region"], categories=REGION_ORDER, ordered=True)
    regional_df = regional_df.sort_values("region").reset_index(drop=True)
    return group_df, regional_df


def build_bootstrap_indices(n_time: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n_blocks = math.ceil(n_time / BLOCK_LENGTH)
    start_max = n_time - BLOCK_LENGTH + 1
    require(start_max > 0, "Time series shorter than bootstrap block length.")
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


def summarize_bootstrap(dist: np.ndarray, deterministic: float) -> tuple[float, float, bool]:
    finite = dist[np.isfinite(dist)]
    require(finite.size == dist.size, "Bootstrap distribution contains NaN replicate(s).")
    ci_low = float(np.nanpercentile(finite, 2.5))
    ci_high = float(np.nanpercentile(finite, 97.5))
    significant = bool((ci_low > 0.0) or (ci_high < 0.0))
    return ci_low, ci_high, significant


def summarize_bootstrap_outputs(monthly_df: pd.DataFrame, group_df: pd.DataFrame, regional_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_template = monthly_df[["month", "phase_sign"]].drop_duplicates().sort_values("month").reset_index(drop=True)
    require(len(phase_template) == 54 + 85, f"Unexpected ENSO-month count: {len(phase_template)}")
    phase_sign = phase_template["phase_sign"].to_numpy(dtype=int)
    boot_indices = build_bootstrap_indices(len(phase_template))

    display_df = monthly_df.loc[monthly_df["display_valid_n_ge_24"]].copy()
    group_records: list[dict[str, object]] = []
    regional_records: list[dict[str, object]] = []
    group_dists: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for row in group_df.itertuples(index=False):
        sub = display_df.loc[(display_df["region"] == str(row.region)) & (display_df["physical_group"] == str(row.physical_group))].sort_values("month")
        term_dist = {}
        for term, col in [
            ("Occurrence", "occurrence_t"),
            ("Conditional-CRE", "conditional_cre_t"),
            ("Interaction", "interaction_t"),
            ("Adjustment", "adjustment_t"),
            ("Total", "total_t"),
        ]:
            series = (
                sub.groupby("month", sort=False)[col]
                .sum()
                .reindex(phase_template["month"])
                .to_numpy(dtype=float)
            )
            term_dist[term] = bootstrap_phase_diffs(series, phase_sign, boot_indices)
        group_dists[(str(row.region), str(row.physical_group))] = term_dist

        require(float(np.max(np.abs(term_dist["Total"] - (term_dist["Occurrence"] + term_dist["Conditional-CRE"] + term_dist["Interaction"])))) <= TOL,
                f"Group bootstrap closure failed for {row.region} {row.physical_group}.")
        require(float(np.max(np.abs(term_dist["Adjustment"] - (term_dist["Conditional-CRE"] + term_dist["Interaction"])))) <= TOL,
                f"Group adjustment bootstrap closure failed for {row.region} {row.physical_group}.")

        det_map = {
            "Occurrence": float(row.delta_occurrence),
            "Conditional-CRE": float(row.delta_conditional_cre),
            "Interaction": float(row.delta_interaction),
            "Adjustment": float(row.delta_adjustment),
            "Total": float(row.delta_total),
        }
        for term in TERM_ORDER:
            ci_low, ci_high, significant = summarize_bootstrap(term_dist[term], det_map[term])
            group_records.append(
                {
                    "region": str(row.region),
                    "physical_group": str(row.physical_group),
                    "term": term,
                    "deterministic_estimate": det_map[term],
                    "ci_low_95": ci_low,
                    "ci_high_95": ci_high,
                    "significant": significant,
                    "n_cells_display_valid": int(row.n_cells_display_valid),
                }
            )

    for row in regional_df.itertuples(index=False):
        term_dist = {
            term: sum(group_dists[(str(row.region), group)][term] for group in GROUP_ORDER) for term in TERM_ORDER
        }
        require(float(np.max(np.abs(term_dist["Total"] - (term_dist["Occurrence"] + term_dist["Conditional-CRE"] + term_dist["Interaction"])))) <= TOL,
                f"Regional bootstrap closure failed for {row.region}.")
        require(float(np.max(np.abs(term_dist["Adjustment"] - (term_dist["Conditional-CRE"] + term_dist["Interaction"])))) <= TOL,
                f"Regional adjustment bootstrap closure failed for {row.region}.")

        det_map = {
            "Occurrence": float(row.delta_occurrence_sum42),
            "Conditional-CRE": float(row.delta_conditional_cre_sum42),
            "Interaction": float(row.delta_interaction_sum42),
            "Adjustment": float(row.delta_adjustment_sum42),
            "Total": float(row.delta_total_sum42),
        }
        for term in TERM_ORDER:
            ci_low, ci_high, significant = summarize_bootstrap(term_dist[term], det_map[term])
            regional_records.append(
                {
                    "region": str(row.region),
                    "term": term,
                    "deterministic_estimate": det_map[term],
                    "ci_low_95": ci_low,
                    "ci_high_95": ci_high,
                    "significant": significant,
                    "n_cells_display_valid": int(row.n_cells_display_valid),
                }
            )
    return pd.DataFrame(group_records), pd.DataFrame(regional_records)


def build_direct_df() -> pd.DataFrame:
    direct = pd.read_csv(FIG02_DIRECT_CSV)
    direct = direct.loc[direct["region"].isin(REGION_ORDER)].copy()
    require(len(direct) == 3, f"Expected 3 direct regional rows, found {len(direct)}.")
    direct["region"] = pd.Categorical(direct["region"], categories=REGION_ORDER, ordered=True)
    direct = direct.sort_values("region").reset_index(drop=True)
    direct["significant"] = (direct["delta_net_ci_lower"] > 0.0) | (direct["delta_net_ci_upper"] < 0.0)
    return direct.rename(
        columns={
            "delta_net": "direct_estimate",
            "delta_net_ci_lower": "direct_ci_low_95",
            "delta_net_ci_upper": "direct_ci_high_95",
        }
    )[["region", "direct_estimate", "direct_ci_low_95", "direct_ci_high_95", "significant"]]


def validate_against_references(
    group_df: pd.DataFrame,
    regional_df: pd.DataFrame,
    group_boot: pd.DataFrame,
    regional_boot: pd.DataFrame,
    direct_df: pd.DataFrame,
    fig05_group: pd.DataFrame,
    fig05_region: pd.DataFrame,
) -> None:
    occ_group = group_df[["region", "physical_group", "delta_occurrence"]].merge(
        fig05_group[["region", "physical_group", "AmountNet_candidate"]],
        on=["region", "physical_group"],
        how="inner",
    )
    require(float((occ_group["delta_occurrence"] - occ_group["AmountNet_candidate"]).abs().max()) <= TOL, "Figure05 group occurrence mismatch.")
    occ_region = regional_df[["region", "delta_occurrence_sum42"]].merge(
        fig05_region[["region", "AmountNet_candidate_sum42"]],
        on="region",
        how="inner",
    )
    require(float((occ_region["delta_occurrence_sum42"] - occ_region["AmountNet_candidate_sum42"]).abs().max()) <= TOL, "Figure05 regional occurrence mismatch.")

    direct_ref = pd.read_csv(FIG02_DIRECT_CSV)
    direct_ref = direct_ref.loc[direct_ref["region"].isin(REGION_ORDER), ["region", "delta_net", "delta_net_ci_lower", "delta_net_ci_upper"]]
    merged_direct = direct_df.merge(direct_ref, on="region", how="inner")
    require(float((merged_direct["direct_estimate"] - merged_direct["delta_net"]).abs().max()) <= TOL, "Direct deterministic mismatch.")
    require(float((merged_direct["direct_ci_low_95"] - merged_direct["delta_net_ci_lower"]).abs().max()) <= TOL, "Direct CI-low mismatch.")
    require(float((merged_direct["direct_ci_high_95"] - merged_direct["delta_net_ci_upper"]).abs().max()) <= TOL, "Direct CI-high mismatch.")


def nice_symmetric_limit(max_abs: float) -> float:
    if max_abs <= 0.0:
        return 1.0
    return math.ceil(max_abs * 1.12 * 2.0) / 2.0


def compute_ylim(group_df: pd.DataFrame, regional_df: pd.DataFrame, direct_df: pd.DataFrame) -> tuple[float, float]:
    extrema: list[float] = []
    for _, row in group_df.iterrows():
        extrema.extend([abs(row["delta_occurrence"]), abs(row["delta_adjustment"]), abs(row["delta_total"])])
    for _, row in regional_df.iterrows():
        extrema.extend([abs(row["delta_occurrence_sum42"]), abs(row["delta_adjustment_sum42"]), abs(row["delta_total_sum42"])])
    for _, row in direct_df.iterrows():
        extrema.extend([abs(row["direct_estimate"]), abs(row["direct_ci_low_95"]), abs(row["direct_ci_high_95"])])
    return (-nice_symmetric_limit(max(extrema)), nice_symmetric_limit(max(extrema)))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.titlesize": 15,
            "savefig.dpi": 300,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.005, 1.02, f"({label})", transform=ax.transAxes, ha="left", va="bottom", fontsize=15, fontweight="bold", clip_on=False, zorder=20)


def draw_group_panel(ax: plt.Axes, df: pd.DataFrame, title: str, ylim: tuple[float, float]) -> None:
    x = np.arange(len(GROUP_ORDER))
    occ_x = x - 0.16
    adj_x = x + 0.16
    width = 0.24
    for idx, group in enumerate(GROUP_ORDER):
        row = df.loc[df["physical_group"].astype(str) == group].iloc[0]
        ax.bar(occ_x[idx], row["delta_occurrence"], width=width, color=COLOR_OCC, edgecolor="black", linewidth=0.5, zorder=2)
        ax.bar(adj_x[idx], row["delta_adjustment"], width=width, color=COLOR_ADJ, edgecolor="black", linewidth=0.5, zorder=2)
        marker_face = "black" if bool(row["total_significant"]) else "white"
        ax.errorbar(
            x[idx],
            row["delta_total"],
            yerr=[[row["delta_total"] - row["total_ci_low_95"]], [row["total_ci_high_95"] - row["delta_total"]]],
            fmt="D",
            color="black",
            markerfacecolor=marker_face,
            markeredgecolor="black",
            markersize=6.4,
            elinewidth=1.1,
            capsize=2.8,
            capthick=1.1,
            zorder=4,
        )
    ax.axhline(0.0, color="black", linewidth=0.9, zorder=1)
    ax.set_title(title, loc="center", pad=6)
    ax.set_xticks(x, [GROUP_LABEL[g] for g in GROUP_ORDER])
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=4, width=0.8, pad=3)


def draw_regional_panel(ax: plt.Axes, regional_df: pd.DataFrame, direct_df: pd.DataFrame, ylim: tuple[float, float]) -> None:
    x = np.arange(len(REGION_ORDER))
    occ_x = x - 0.23
    adj_x = x - 0.05
    cand_x = x + 0.11
    direct_x = x + 0.27
    bar_width = 0.14

    for idx, region in enumerate(REGION_ORDER):
        row = regional_df.loc[regional_df["region"].astype(str) == region].iloc[0]
        ax.bar(occ_x[idx], row["delta_occurrence_sum42"], width=bar_width, color=COLOR_OCC, edgecolor="black", linewidth=0.5, zorder=2)
        ax.bar(adj_x[idx], row["delta_adjustment_sum42"], width=bar_width, color=COLOR_ADJ, edgecolor="black", linewidth=0.5, zorder=2)
        marker_face = "black" if bool(row["total_significant"]) else "white"
        ax.errorbar(
            cand_x[idx],
            row["delta_total_sum42"],
            yerr=[[row["delta_total_sum42"] - row["total_ci_low_95"]], [row["total_ci_high_95"] - row["delta_total_sum42"]]],
            fmt="D",
            color="black",
            markerfacecolor=marker_face,
            markeredgecolor="black",
            markersize=6.6,
            elinewidth=1.1,
            capsize=2.8,
            capthick=1.1,
            zorder=4,
        )
        drow = direct_df.loc[direct_df["region"].astype(str) == region].iloc[0]
        dface = DIRECT_COLOR if bool(drow["significant"]) else "white"
        ax.errorbar(
            direct_x[idx],
            drow["direct_estimate"],
            yerr=[[drow["direct_estimate"] - drow["direct_ci_low_95"]], [drow["direct_ci_high_95"] - drow["direct_estimate"]]],
            fmt="o",
            color=DIRECT_COLOR,
            markerfacecolor=dface,
            markeredgecolor=DIRECT_COLOR,
            markersize=5.8,
            elinewidth=1.0,
            capsize=2.6,
            capthick=1.0,
            zorder=4,
        )
    ax.axhline(0.0, color="black", linewidth=0.9, zorder=1)
    panel_label(ax, PANEL_LABELS["REG"])
    ax.set_xticks(x + 0.02, REGION_ORDER)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=4, width=0.8, pad=3)


def build_plot_tables(group_df: pd.DataFrame, regional_df: pd.DataFrame, direct_df: pd.DataFrame) -> pd.DataFrame:
    group_out = group_df.copy()
    group_out["panel"] = group_out["region"].astype(str).map(REGION_TITLE)
    group_out["x_label"] = group_out["physical_group"].astype(str).map(GROUP_LABEL)
    group_out["plot_type"] = "group_occurrence_adjustment_total"
    group_out = group_out.rename(
        columns={
            "delta_occurrence": "occurrence",
            "delta_conditional_cre": "conditional_cre",
            "delta_interaction": "interaction",
            "delta_adjustment": "cre_adjustment",
            "delta_total": "total_pathway",
        }
    )

    regional_out = regional_df.copy()
    regional_out["panel"] = "(d) Regional occurrence, adjustment, and direct response"
    regional_out["x_label"] = regional_out["region"].astype(str)
    regional_out["plot_type"] = "regional_occurrence_adjustment_total"
    regional_out["physical_group"] = ""
    regional_out = regional_out.rename(
        columns={
            "delta_occurrence_sum42": "occurrence",
            "delta_conditional_cre_sum42": "conditional_cre",
            "delta_interaction_sum42": "interaction",
            "delta_adjustment_sum42": "cre_adjustment",
            "delta_total_sum42": "total_pathway",
        }
    )

    direct_out = direct_df.copy()
    direct_out["panel"] = "(d) Regional occurrence, adjustment, and direct response"
    direct_out["plot_type"] = "regional_direct_response"
    direct_out["physical_group"] = ""
    direct_out["x_label"] = direct_out["region"].astype(str)
    for col in ["occurrence", "conditional_cre", "interaction", "cre_adjustment", "total_pathway", "total_ci_low_95", "total_ci_high_95", "total_significant", "adjustment_ci_low_95", "adjustment_ci_high_95", "adjustment_significant", "n_cells_display_valid"]:
        direct_out[col] = np.nan

    group_out["direct_estimate"] = np.nan
    group_out["direct_ci_low_95"] = np.nan
    group_out["direct_ci_high_95"] = np.nan
    group_out["direct_significant"] = np.nan
    regional_out["direct_estimate"] = np.nan
    regional_out["direct_ci_low_95"] = np.nan
    regional_out["direct_ci_high_95"] = np.nan
    regional_out["direct_significant"] = np.nan
    direct_out = direct_out.rename(columns={"significant": "direct_significant"})

    cols = [
        "panel",
        "plot_type",
        "region",
        "physical_group",
        "x_label",
        "occurrence",
        "conditional_cre",
        "interaction",
        "cre_adjustment",
        "total_pathway",
        "total_ci_low_95",
        "total_ci_high_95",
        "total_significant",
        "adjustment_ci_low_95",
        "adjustment_ci_high_95",
        "adjustment_significant",
        "direct_estimate",
        "direct_ci_low_95",
        "direct_ci_high_95",
        "direct_significant",
        "n_cells_display_valid",
    ]
    export_df = pd.concat([group_out[cols], regional_out[cols], direct_out[cols]], ignore_index=True)
    export_df.to_csv(OUT_PLOT, index=False)
    return export_df


def write_caption() -> None:
    text = (
        "Figure 6. Daytime Net cloud radiative effect pathway decomposition across the western, central, and eastern Pacific "
        "using the +/-0.5 C Nino3.4 ENSO definition and the Figure06 copy.py presentation style. Panels (a)-(c) show "
        "El Nino minus La Nina Net contributions for the five physical cloud groups in the western Pacific (WP), central Pacific "
        "(CP), and eastern Pacific (EP), respectively. Blue bars denote occurrence-mediated contributions, and orange bars denote "
        "CRE adjustment, defined as the sum of the conditional-CRE and interaction components. Diamonds indicate total cloud-group "
        "pathway contributions, with filled symbols denoting 95% moving-block-bootstrap confidence intervals that exclude zero. "
        "Panel (d) summarizes the regional occurrence and CRE-adjustment contributions together with the regional total pathways "
        "and the corresponding direct all-sky daytime Net CRE responses. Direct markers are shown for directional comparison only "
        "and are not interpreted as an exact reconstruction target.\n"
    )
    OUT_CAPTION.write_text(text, encoding="utf-8")


def write_manifest() -> None:
    lines = [
        "# Figure06 degC05 input manifest",
        "",
        f"- Monthly regional candidate chain: `{REGIONAL_MONTHLY_NC}`",
        f"- Regional conditional CRE kernel table: `{REGIONAL_FIG04_CSV}`",
        f"- ENSO index file: `{NINO_CSV}`",
        f"- Figure05 degC05 group occurrence summary: `{FIG05_GROUP_CSV}`",
        f"- Figure05 degC05 regional occurrence summary: `{FIG05_REGION_CSV}`",
        f"- Figure02 degC05 direct regional Net summary: `{FIG02_DIRECT_CSV}`",
    ]
    OUT_MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method_checks(group_df: pd.DataFrame, regional_df: pd.DataFrame, direct_df: pd.DataFrame) -> None:
    cp_total_sig = bool(regional_df.loc[regional_df["region"].astype(str) == "CP", "total_significant"].iloc[0])
    cp_direct_sig = bool(direct_df.loc[direct_df["region"].astype(str) == "CP", "significant"].iloc[0])
    wp_total_sig = bool(regional_df.loc[regional_df["region"].astype(str) == "WP", "total_significant"].iloc[0])
    wp_direct_sig = bool(direct_df.loc[direct_df["region"].astype(str) == "WP", "significant"].iloc[0])
    ep_total_sig = bool(regional_df.loc[regional_df["region"].astype(str) == "EP", "total_significant"].iloc[0])
    ep_direct_sig = bool(direct_df.loc[direct_df["region"].astype(str) == "EP", "significant"].iloc[0])
    lines = [
        "Figure06 degC05 method and plot checks",
        "",
        f"- plotting style source = Figure06 candidate Net decomposition final v4 copy.py",
        "- ENSO definition = nino34_anom with El Nino >= +0.5 C and La Nina <= -0.5 C",
        "- monthly source product = daytime candidate integrated chain",
        "- per-cloud-type paired-valid support = True",
        "- no all-42 joint strict mask = True",
        "- cf==0 retained as zero contribution = True",
        "- Net = SW + LW checked in monthly source = True",
        f"- bootstrap block length (months) = {BLOCK_LENGTH}",
        f"- bootstrap samples = {N_BOOT}",
        f"- bootstrap seed = {SEED}",
        "- bootstrap method = joint moving-block bootstrap over the ENSO-month subset",
        "- formal plot regenerated from fixed monthly chain plus fixed +/-0.5 C phase definition",
        "- CRE adjustment is defined as Conditional-CRE + Interaction",
        "- direct markers are retained for directional comparison only",
        "",
        "Reference cross-checks",
        f"- Figure05 degC05 occurrence summaries reused as deterministic occurrence targets = True",
        f"- Figure02 degC05 direct Net summary matched = True",
        "",
        "Interpretation checks",
        f"- CP total and direct Net are both significantly positive = {cp_total_sig and cp_direct_sig}",
        f"- WP total and direct Net are both non-significant = {not wp_total_sig and not wp_direct_sig}",
        f"- EP direct Net is significant while total pathway is non-significant = {ep_direct_sig and not ep_total_sig}",
        "",
        "Output files",
        f"- png: {OUT_PNG}",
        f"- pdf: {OUT_PDF}",
        f"- plot data: {OUT_PLOT}",
        f"- group summary: {OUT_GROUP}",
        f"- regional summary: {OUT_REGION}",
        f"- direct summary: {OUT_DIRECT}",
        f"- caption: {OUT_CAPTION}",
        f"- input manifest: {OUT_MANIFEST}",
    ]
    OUT_METHOD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(group_df: pd.DataFrame, regional_df: pd.DataFrame, direct_df: pd.DataFrame) -> None:
    setup_style()
    ylim = compute_ylim(group_df, regional_df, direct_df)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.1), sharey=True)
    axes = axes.ravel()

    for idx, region in enumerate(REGION_ORDER):
        draw_group_panel(axes[idx], group_df.loc[group_df["region"].astype(str) == region].copy(), REGION_TITLE[region], ylim)
        panel_label(axes[idx], PANEL_LABELS[region])

    draw_regional_panel(axes[3], regional_df, direct_df, ylim)
    axes[0].set_ylabel(r"CRE (W m$^{-2}$)")
    axes[2].set_ylabel(r"CRE (W m$^{-2}$)")

    axes[0].legend(
        handles=[
            mpatches.Patch(facecolor=COLOR_OCC, edgecolor="black", label="Occurrence"),
            mpatches.Patch(facecolor=COLOR_ADJ, edgecolor="black", label="CRE adjustment"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        handlelength=1.4,
        handletextpad=0.5,
        borderaxespad=0.0,
        fontsize=11,
    )
    axes[1].legend(
        handles=[
            mlines.Line2D([], [], color="black", marker="D", linestyle="None", markerfacecolor="black", markeredgecolor="black", markersize=6.2, label="Total (significant)"),
            mlines.Line2D([], [], color="black", marker="D", linestyle="None", markerfacecolor="white", markeredgecolor="black", markersize=6.2, label="Total (not significant)"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
        handlelength=1.0,
        handletextpad=0.5,
        borderaxespad=0.0,
        fontsize=11,
    )
    axes[3].legend(
        handles=[
            mlines.Line2D([], [], color=DIRECT_COLOR, marker="o", linestyle="None", markerfacecolor=DIRECT_COLOR, markeredgecolor=DIRECT_COLOR, markersize=5.8, label="Direct Net CRE (significant)"),
            mlines.Line2D([], [], color=DIRECT_COLOR, marker="o", linestyle="None", markerfacecolor="white", markeredgecolor=DIRECT_COLOR, markersize=5.8, label="Direct Net CRE (not significant)"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        handlelength=1.0,
        handletextpad=0.5,
        borderaxespad=0.0,
        fontsize=11,
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08, wspace=0.12, hspace=0.28)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    plt.close(fig)


def main() -> None:
    ensure_inputs()
    ds, fig04, nino, fig05_group, fig05_region = load_inputs()
    validate_inputs(ds, fig04, fig05_group, fig05_region)
    phase = build_phase(ds["time"].values, nino)
    monthly_df = build_monthly_terms(ds, fig04, phase)
    group_df, regional_df = summarize_deterministic(monthly_df)
    group_boot, regional_boot = summarize_bootstrap_outputs(monthly_df, group_df, regional_df)
    direct_df = build_direct_df()

    group_plot = group_df.merge(
        group_boot.loc[group_boot["term"] == "Total", ["region", "physical_group", "ci_low_95", "ci_high_95", "significant"]].rename(
            columns={"ci_low_95": "total_ci_low_95", "ci_high_95": "total_ci_high_95", "significant": "total_significant"}
        ),
        on=["region", "physical_group"],
        how="inner",
    ).merge(
        group_boot.loc[group_boot["term"] == "Adjustment", ["region", "physical_group", "ci_low_95", "ci_high_95", "significant"]].rename(
            columns={"ci_low_95": "adjustment_ci_low_95", "ci_high_95": "adjustment_ci_high_95", "significant": "adjustment_significant"}
        ),
        on=["region", "physical_group"],
        how="inner",
    )
    regional_plot = regional_df.merge(
        regional_boot.loc[regional_boot["term"] == "Total", ["region", "ci_low_95", "ci_high_95", "significant"]].rename(
            columns={"ci_low_95": "total_ci_low_95", "ci_high_95": "total_ci_high_95", "significant": "total_significant"}
        ),
        on="region",
        how="inner",
    ).merge(
        regional_boot.loc[regional_boot["term"] == "Adjustment", ["region", "ci_low_95", "ci_high_95", "significant"]].rename(
            columns={"ci_low_95": "adjustment_ci_low_95", "ci_high_95": "adjustment_ci_high_95", "significant": "adjustment_significant"}
        ),
        on="region",
        how="inner",
    )

    validate_against_references(group_df, regional_df, group_boot, regional_boot, direct_df, fig05_group, fig05_region)
    group_plot.to_csv(OUT_GROUP, index=False)
    regional_plot.to_csv(OUT_REGION, index=False)
    direct_df.to_csv(OUT_DIRECT, index=False)
    build_plot_tables(group_plot, regional_plot, direct_df)
    make_figure(group_plot, regional_plot, direct_df)
    write_caption()
    write_manifest()
    write_method_checks(group_plot, regional_plot, direct_df)


if __name__ == "__main__":
    main()
