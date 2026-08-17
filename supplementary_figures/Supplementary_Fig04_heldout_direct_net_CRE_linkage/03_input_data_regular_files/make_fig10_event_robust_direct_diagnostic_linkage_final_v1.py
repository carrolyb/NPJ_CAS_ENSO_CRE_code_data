#!/usr/bin/env python3
"""Render Final Figure 10 from fixed audited Step10 outputs only."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "outputs" / "verified" / "figure10_direct_linkage"
FINAL_DIR = BASE_DIR / "final_figures"

STEP10_1A_DIRECT_PATH = BASE_DIR / "Step10_1A_formal_direct_regional_monthly_Net_series.csv"
STEP10_1A_FIG02_CHECK_PATH = BASE_DIR / "Step10_1A_formal_direct_vs_Figure02_composite_reproduction_check.csv"
STEP10_1A_ALIGNED_PATH = BASE_DIR / "Step10_1A_metric_formal_direct_aligned_monthly_series.csv"
STEP10_1A_MODEL_SUMMARY_PATH = BASE_DIR / "Step10_1A_deterministic_model_summary.csv"
STEP10_1A_COLLINEARITY_PATH = BASE_DIR / "Step10_1A_predictor_collinearity_diagnostic.csv"
STEP10_1A_SUMMARY_PATH = BASE_DIR / "Step10_1A_formal_direct_series_and_deterministic_model_screening_summary.txt"

STEP10_1B1_BOOT_PATH = BASE_DIR / "Step10_1B1_model_bootstrap_summary.csv"
STEP10_1B1_INCREMENT_PATH = BASE_DIR / "Step10_1B1_model_increment_bootstrap_summary.csv"
STEP10_1B1_HCTB_PATH = BASE_DIR / "Step10_1B1_HCTB_incremental_coefficient_stability.csv"
STEP10_1B1_SUMMARY_PATH = BASE_DIR / "Step10_1B1_joint_moving_block_bootstrap_summary.txt"

STEP10_1B2A_SPATIAL_PATH = BASE_DIR / "Step10_1B2A_spatial_support_compatibility_audit.txt"
STEP10_1B2A_FOLD_PATH = BASE_DIR / "Step10_1B2A_purged_blocked_CV_fold_inventory.csv"
STEP10_1B2A_SKILL_PATH = BASE_DIR / "Step10_1B2A_purged_blocked_CV_model_skill.csv"
STEP10_1B2A_PRED_PATH = BASE_DIR / "Step10_1B2A_purged_blocked_CV_monthly_predictions.csv"
STEP10_1B2A_INCREMENT_PATH = BASE_DIR / "Step10_1B2A_purged_blocked_CV_model_increment_deterministic.csv"
STEP10_1B2A_BOOT_PATH = BASE_DIR / "Step10_1B2A_purged_blocked_CV_skill_increment_bootstrap.csv"
STEP10_1B2A_SUMMARY_PATH = BASE_DIR / "Step10_1B2A_spatial_support_and_purged_blocked_CV_summary.txt"

STEP10_1B2B_EVENT_INVENTORY_PATH = BASE_DIR / "Step10_1B2B_ENSO_event_inventory.csv"
STEP10_1B2B_LOEO_DETAIL_PATH = BASE_DIR / "Step10_1B2B_leave_one_event_out_refit_stability.csv"
STEP10_1B2B_LOEO_SUMMARY_PATH = BASE_DIR / "Step10_1B2B_leave_one_event_out_refit_summary.csv"
STEP10_1B2B_EVENT_PRED_PATH = BASE_DIR / "Step10_1B2B_event_heldout_monthly_predictions.csv"
STEP10_1B2B_EVENT_SKILL_PATH = BASE_DIR / "Step10_1B2B_event_heldout_skill_by_event.csv"
STEP10_1B2B_POOLED_PATH = BASE_DIR / "Step10_1B2B_pooled_event_heldout_skill.csv"
STEP10_1B2B_BOOT_PATH = BASE_DIR / "Step10_1B2B_event_cluster_bootstrap_skill_increment.csv"
STEP10_1B2B_SUMMARY_PATH = BASE_DIR / "Step10_1B2B_ENSO_event_robustness_and_Figure10_gate_summary.txt"

FIG09_METHOD_PATH = ROOT / "outputs" / "verified" / "figure09_12_new_chain" / "final_figures" / "Figure09_monthly_diagnostic_representativeness_method_and_plot_checks_v2.txt"
FIG09_CAPTION_PATH = ROOT / "outputs" / "verified" / "figure09_12_new_chain" / "final_figures" / "Figure09_monthly_diagnostic_representativeness_caption_v2.md"

OUT_PNG = FINAL_DIR / "Figure10_event_robust_direct_diagnostic_linkage_final_v1.png"
OUT_PDF = FINAL_DIR / "Figure10_event_robust_direct_diagnostic_linkage_final_v1.pdf"
OUT_PLOT_DATA = FINAL_DIR / "Figure10_event_robust_direct_diagnostic_linkage_plot_data_v1.csv"
OUT_CAPTION = FINAL_DIR / "Figure10_event_robust_direct_diagnostic_linkage_caption_v1.md"
OUT_CHECKS = FINAL_DIR / "Figure10_event_robust_direct_diagnostic_linkage_method_and_plot_checks_v1.txt"

REGIONS = ["WP", "CP", "EP"]
REGION_NAMES = {"WP": "Western Pacific", "CP": "Central Pacific", "EP": "Eastern Pacific"}
REGION_COLORS = {
    "WP": {"main": "#2b6cb0", "light": "#8ab6e6"},
    "CP": {"main": "#dd6b20", "light": "#f1b27b"},
    "EP": {"main": "#2f855a", "light": "#86c5a2"},
}
PANEL_LABELS = {
    "WP": "(a)",
    "CP": "(b)",
    "EP": "(c)",
    "blocked_cv": "(d)",
    "event_heldout": "(e)",
    "coef_stability": "(f)",
}
EXPECTED_BLOCKED_CV = {
    "WP": (0.616734, 0.507450, 0.721265),
    "CP": (0.477245, 0.354838, 0.565539),
    "EP": (0.031830, 0.012400, 0.050526),
}
EXPECTED_EVENT_HELDOUT = {
    "WP": (0.628288, 0.451646, 0.792369),
    "CP": (0.437934, 0.295996, 0.596719),
    "EP": (0.027780, 0.003558, 0.057831),
}
EXPECTED_HCTB = {
    "WP": (-49.400666, -54.626488, -43.112247),
    "CP": (-56.360097, -63.759902, -48.691536),
    "EP": (-51.319850, -61.842399, -36.538391),
}
CAPTION_TEXT = (
    "Figure 10. Event-robust diagnostic linkage between cloud-structure metrics and regional direct daytime Net cloud radiative effect variability. "
    "Panels (a)–(c) compare observed regional direct daytime Net CRE anomalies with held-out diagnostic estimates from the full metric model, "
    "M3 = LCSP + HCCF + HCTB, over the western, central, and eastern Pacific fixed regional boxes, respectively. Held-out estimates are obtained "
    "from purged blocked cross-validation using contiguous 12-month test blocks and a 6-month purge window on both sides of each test block. "
    "Panel (d) presents the increase in held-out diagnostic (R^2) from adding HCTB to the comparator model M1 = LCSP + HCCF, based on the purged "
    "blocked cross-validation; error bars denote 95% moving-block-bootstrap confidence intervals. Panel (e) presents the corresponding (R^2) "
    "increments when complete ENSO events are held out; error bars denote 95% event-cluster-bootstrap confidence intervals. Panel (f) reports the "
    "HCTB coefficient in M3, with moving-block-bootstrap confidence intervals and leave-one-ENSO-event-out ranges. HCCF is retained as a total-high-cloud "
    "occurrence comparator, whereas HCTB represents additional high-cloud structural information based on the remapped cloud-group definition. These "
    "results characterize an event-robust diagnostic association with regional direct Net CRE variability and are not interpreted as prediction skill, "
    "causal control, independent validation, or an exact reconstruction of the direct all-sky response."
)


def require_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required audited input is missing: {path}")
    return path


def read_optional_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def assert_close(actual: float, expected: float, label: str, tol: float = 1.0e-6) -> None:
    if not np.isclose(actual, expected, atol=tol, rtol=0.0):
        raise RuntimeError(f"{label} mismatch: actual={actual}, expected={expected}")


def series_range(values: pd.Series) -> tuple[float, float]:
    return float(values.min()), float(values.max())


def main() -> int:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for path in [
        STEP10_1A_DIRECT_PATH,
        STEP10_1A_FIG02_CHECK_PATH,
        STEP10_1A_ALIGNED_PATH,
        STEP10_1A_MODEL_SUMMARY_PATH,
        STEP10_1A_COLLINEARITY_PATH,
        STEP10_1A_SUMMARY_PATH,
        STEP10_1B1_BOOT_PATH,
        STEP10_1B1_INCREMENT_PATH,
        STEP10_1B1_HCTB_PATH,
        STEP10_1B1_SUMMARY_PATH,
        STEP10_1B2A_SPATIAL_PATH,
        STEP10_1B2A_FOLD_PATH,
        STEP10_1B2A_SKILL_PATH,
        STEP10_1B2A_PRED_PATH,
        STEP10_1B2A_INCREMENT_PATH,
        STEP10_1B2A_BOOT_PATH,
        STEP10_1B2A_SUMMARY_PATH,
        STEP10_1B2B_EVENT_INVENTORY_PATH,
        STEP10_1B2B_LOEO_DETAIL_PATH,
        STEP10_1B2B_LOEO_SUMMARY_PATH,
        STEP10_1B2B_EVENT_PRED_PATH,
        STEP10_1B2B_EVENT_SKILL_PATH,
        STEP10_1B2B_POOLED_PATH,
        STEP10_1B2B_BOOT_PATH,
        STEP10_1B2B_SUMMARY_PATH,
    ]:
        require_exists(path)

    fig02_check_df = pd.read_csv(STEP10_1A_FIG02_CHECK_PATH)
    skill_df = pd.read_csv(STEP10_1B2A_SKILL_PATH)
    pred_df = pd.read_csv(STEP10_1B2A_PRED_PATH, parse_dates=["month"])
    inc_df = pd.read_csv(STEP10_1B2A_INCREMENT_PATH)
    inc_boot_df = pd.read_csv(STEP10_1B2A_BOOT_PATH)
    hctb_df = pd.read_csv(STEP10_1B1_HCTB_PATH)
    loeo_df = pd.read_csv(STEP10_1B2B_LOEO_DETAIL_PATH)
    loeo_summary_df = pd.read_csv(STEP10_1B2B_LOEO_SUMMARY_PATH)
    pooled_df = pd.read_csv(STEP10_1B2B_POOLED_PATH)
    event_boot_df = pd.read_csv(STEP10_1B2B_BOOT_PATH)
    event_inventory_df = pd.read_csv(STEP10_1B2B_EVENT_INVENTORY_PATH)
    fold_df = pd.read_csv(STEP10_1B2A_FOLD_PATH)
    spatial_text = STEP10_1B2A_SPATIAL_PATH.read_text(encoding="utf-8")
    cv_summary_text = STEP10_1B2A_SUMMARY_PATH.read_text(encoding="utf-8")
    event_summary_text = STEP10_1B2B_SUMMARY_PATH.read_text(encoding="utf-8")

    if not bool(fig02_check_df["pass_if_exact_same_chain"].all()):
        raise RuntimeError("Figure 2 same-chain reproduction does not pass; stop before plotting Figure 10.")
    if "spatial_support_compatible = True" not in spatial_text:
        raise RuntimeError("Spatial-support compatibility is not confirmed; stop before plotting Figure 10.")
    if "CP passes the held-out core gate = True" not in cv_summary_text:
        raise RuntimeError("Purged blocked-CV core gate is not confirmed; stop before plotting Figure 10.")
    if "Figure 10 allowed to enter formal plotting = True." not in event_summary_text:
        raise RuntimeError("ENSO-event robustness summary does not approve formal plotting; stop before plotting Figure 10.")

    pred_df = pred_df.sort_values(["region", "month"]).reset_index(drop=True)
    if set(pred_df["region"].unique().tolist()) != set(REGIONS):
        raise RuntimeError("Purged blocked-CV prediction regions do not match WP/CP/EP only.")
    if pred_df.shape[0] != 744:
        raise RuntimeError("Purged blocked-CV monthly prediction count mismatch.")
    for region in REGIONS:
        region_pred = pred_df[pred_df["region"] == region]
        if len(region_pred) != 248:
            raise RuntimeError(f"{region} does not have 248 blocked-CV OOF monthly predictions.")

    blocked_cv_records = []
    event_records = []
    coef_records = []

    skill_map = skill_df.set_index(["region", "model_name"])
    inc_map = inc_df.set_index(["region", "comparison"])
    inc_boot_map = inc_boot_df.set_index(["region", "comparison"])
    hctb_map = hctb_df.set_index("region")
    pooled_map = pooled_df.set_index("region")
    event_boot_map = event_boot_df.set_index("region")
    loeo_summary_map = loeo_summary_df.set_index("region")

    for region in REGIONS:
        blocked_det = float(inc_map.loc[(region, "M3_minus_M1"), "delta_OOF_R2"])
        blocked_low = float(inc_boot_map.loc[(region, "M3_minus_M1"), "delta_OOF_R2_ci_low_95"])
        blocked_high = float(inc_boot_map.loc[(region, "M3_minus_M1"), "delta_OOF_R2_ci_high_95"])
        exp_det, exp_low, exp_high = EXPECTED_BLOCKED_CV[region]
        assert_close(blocked_det, exp_det, f"{region} blocked-CV deterministic delta OOF R2")
        assert_close(blocked_low, exp_low, f"{region} blocked-CV delta OOF R2 CI low")
        assert_close(blocked_high, exp_high, f"{region} blocked-CV delta OOF R2 CI high")
        blocked_cv_records.append(
            {
                "panel_group": "blocked_cv_increment",
                "region": region,
                "x_category": region,
                "value": blocked_det,
                "ci_low": blocked_low,
                "ci_high": blocked_high,
            }
        )

        event_det = float(pooled_map.loc[region, "delta_pooled_event_OOF_R2_M3_minus_M1"])
        event_low = float(event_boot_map.loc[region, "delta_pooled_event_OOF_R2_ci_low_95"])
        event_high = float(event_boot_map.loc[region, "delta_pooled_event_OOF_R2_ci_high_95"])
        exp_det, exp_low, exp_high = EXPECTED_EVENT_HELDOUT[region]
        assert_close(event_det, exp_det, f"{region} event-held-out deterministic delta OOF R2")
        assert_close(event_low, exp_low, f"{region} event-held-out delta OOF R2 CI low")
        assert_close(event_high, exp_high, f"{region} event-held-out delta OOF R2 CI high")
        event_records.append(
            {
                "panel_group": "event_heldout_increment",
                "region": region,
                "x_category": region,
                "value": event_det,
                "ci_low": event_low,
                "ci_high": event_high,
            }
        )

        coef_det = float(hctb_map.loc[region, "deterministic_coefficient_HCTB_M3"])
        coef_low = float(hctb_map.loc[region, "coefficient_HCTB_M3_ci_low_95"])
        coef_high = float(hctb_map.loc[region, "coefficient_HCTB_M3_ci_high_95"])
        exp_det, exp_low, exp_high = EXPECTED_HCTB[region]
        assert_close(coef_det, exp_det, f"{region} HCTB deterministic coefficient")
        assert_close(coef_low, exp_low, f"{region} HCTB coefficient CI low")
        assert_close(coef_high, exp_high, f"{region} HCTB coefficient CI high")

        region_loeo = loeo_df[loeo_df["region"] == region]
        loeo_min, loeo_max = series_range(region_loeo["M3_HCTB_coefficient_remaining"])
        if float(loeo_summary_map.loc[region, "fraction_events_HCTB_coefficient_sign_consistent"]) != 1.0:
            raise RuntimeError(f"{region} leave-one-event-out HCTB sign consistency is not 1.0.")
        coef_records.append(
            {
                "panel_group": "hctb_coefficient_stability",
                "region": region,
                "point_estimate": coef_det,
                "moving_block_ci_low": coef_low,
                "moving_block_ci_high": coef_high,
                "loeo_range_low": loeo_min,
                "loeo_range_high": loeo_max,
            }
        )

    all_vals = pd.concat(
        [
            pred_df["Direct_Net_anom_formal"],
            pred_df["predicted_M3"],
        ],
        ignore_index=True,
    )
    lim = float(np.nanmax(np.abs(all_vals.to_numpy(dtype=float))))
    lim = np.ceil((lim * 1.08) / 0.5) * 0.5
    axis_limits = (-lim, lim)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.9))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.87, bottom=0.16, wspace=0.28, hspace=0.34)

    plot_rows = []

    for idx, region in enumerate(REGIONS):
        ax = axes[0, idx]
        color = REGION_COLORS[region]["main"]
        light = REGION_COLORS[region]["light"]
        region_pred = pred_df[pred_df["region"] == region].copy()
        region_pred["panel_group"] = f"scatter_{region}"
        plot_rows.extend(region_pred.to_dict(orient="records"))

        ax.scatter(
            region_pred["Direct_Net_anom_formal"],
            region_pred["predicted_M3"],
            s=20,
            color=color,
            alpha=0.62,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        ax.axline((0, 0), slope=1.0, color="#333333", linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
        ax.axhline(0.0, color="#d0d0d0", linewidth=0.8, zorder=1)
        ax.axvline(0.0, color="#d0d0d0", linewidth=0.8, zorder=1)
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{PANEL_LABELS[region]} {REGION_NAMES[region]} held-out diagnostic linkage", loc="left", pad=8)
        ax.set_xlabel("Observed direct Net CRE anomaly (W m$^{-2}$)")
        if idx == 0:
            ax.set_ylabel("M3 held-out diagnostic estimate (W m$^{-2}$)")
        else:
            ax.set_ylabel("")

        oof_r2 = float(skill_map.loc[(region, "M3"), "OOF_R2"])
        delta_oof_r2 = float(inc_map.loc[(region, "M3_minus_M1"), "delta_OOF_R2"])
        ax.text(
            0.03,
            0.97,
            f"OOF R$^2$ (M3) = {oof_r2:.3f}\n$\\Delta$OOF R$^2$ vs M1 = {delta_oof_r2:+.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": light, "boxstyle": "round,pad=0.25", "alpha": 0.95},
        )

    ax_d = axes[1, 0]
    x = np.arange(len(REGIONS))
    for i, region in enumerate(REGIONS):
        row = blocked_cv_records[i]
        color = REGION_COLORS[region]["main"]
        ax_d.errorbar(
            i,
            row["value"],
            yerr=[[row["value"] - row["ci_low"]], [row["ci_high"] - row["value"]]],
            fmt="o",
            ms=7.5,
            lw=2,
            capsize=4,
            color=color,
            ecolor=color,
            zorder=3,
        )
        plot_rows.append(row)
    ax_d.axhline(0.0, color="#bdbdbd", linewidth=0.9, zorder=1)
    ax_d.set_xticks(x, REGIONS)
    ax_d.set_ylabel("$\\Delta$ OOF R$^2_{M3-M1}$")
    ax_d.set_title(f"{PANEL_LABELS['blocked_cv']} Purged blocked-CV increment", loc="left", pad=8)
    ax_d.set_ylim(-0.03, 0.82)

    ax_e = axes[1, 1]
    for i, region in enumerate(REGIONS):
        row = event_records[i]
        color = REGION_COLORS[region]["main"]
        ax_e.errorbar(
            i,
            row["value"],
            yerr=[[row["value"] - row["ci_low"]], [row["ci_high"] - row["value"]]],
            fmt="o",
            ms=7.5,
            lw=2,
            capsize=4,
            color=color,
            ecolor=color,
            zorder=3,
        )
        plot_rows.append(row)
    ax_e.axhline(0.0, color="#bdbdbd", linewidth=0.9, zorder=1)
    ax_e.set_xticks(x, REGIONS)
    ax_e.set_ylabel("$\\Delta$ OOF R$^2_{M3-M1}$")
    ax_e.set_title(f"{PANEL_LABELS['event_heldout']} ENSO-event-held-out increment", loc="left", pad=8)
    ax_e.set_ylim(-0.03, 0.82)

    ax_f = axes[1, 2]
    y = np.arange(len(REGIONS))
    for i, region in enumerate(REGIONS):
        row = coef_records[i]
        ax_f.plot(
            [row["loeo_range_low"], row["loeo_range_high"]],
            [i, i],
            color="#c8c8c8",
            linewidth=8.0,
            solid_capstyle="round",
            zorder=1,
        )
        ax_f.errorbar(
            row["point_estimate"],
            i,
            xerr=[[row["point_estimate"] - row["moving_block_ci_low"]], [row["moving_block_ci_high"] - row["point_estimate"]]],
            fmt="o",
            ms=6.8,
            lw=2.2,
            capsize=4,
            color="black",
            ecolor="black",
            zorder=3,
        )
        plot_rows.append(row)
    ax_f.axvline(0.0, color="#bdbdbd", linewidth=0.9, zorder=0)
    ax_f.set_yticks(y, REGIONS)
    ax_f.set_xlabel("HCTB coefficient in M3 (W m$^{-2}$ per unit fraction)")
    ax_f.set_title(f"{PANEL_LABELS['coef_stability']} HCTB coefficient stability", loc="left", pad=8)
    ax_f.set_ylim(-0.6, len(REGIONS) - 0.4)
    ax_f.invert_yaxis()

    coef_min = min(row["loeo_range_low"] for row in coef_records)
    coef_max = max(row["loeo_range_high"] for row in coef_records)
    coef_pad = 5.0
    ax_f.set_xlim(coef_min - coef_pad, max(2.0, coef_max + coef_pad))

    legend_handles = [
        mlines.Line2D([], [], color="#333333", linestyle=(0, (4, 3)), linewidth=1.2, label="1:1 reference"),
        mlines.Line2D([], [], color="black", marker="o", linewidth=2.2, markersize=6.5, label="Moving-block 95% CI"),
        mlines.Line2D([], [], color="#c8c8c8", linewidth=8.0, solid_capstyle="round", label="Leave-one-event-out range"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.05))

    fig.suptitle(
        "Figure 10. Event-robust diagnostic linkage between cloud-structure metrics and\nregional direct daytime Net CRE variability",
        fontsize=13,
        y=0.975,
    )

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    plot_data_df = pd.DataFrame(plot_rows)
    plot_data_df.to_csv(OUT_PLOT_DATA, index=False)
    OUT_CAPTION.write_text(CAPTION_TEXT + "\n", encoding="utf-8")

    method_lines = [
        "Figure10 event-robust direct diagnostic linkage method and plot checks v1",
        "",
        "Fixed audited inputs actually used",
        f"- panel (a)-(c) blocked-CV OOF monthly prediction input: {STEP10_1B2A_PRED_PATH}",
        f"- panel (a)-(c) blocked-CV skill summary input: {STEP10_1B2A_SKILL_PATH}",
        f"- panel (a)-(d) blocked-CV deterministic increment input: {STEP10_1B2A_INCREMENT_PATH}",
        f"- panel (d) blocked-CV moving-block-bootstrap CI input: {STEP10_1B2A_BOOT_PATH}",
        f"- panel (e) pooled ENSO-event-held-out skill input: {STEP10_1B2B_POOLED_PATH}",
        f"- panel (e) event-cluster-bootstrap CI input: {STEP10_1B2B_BOOT_PATH}",
        f"- panel (f) moving-block HCTB coefficient input: {STEP10_1B1_HCTB_PATH}",
        f"- panel (f) leave-one-event-out coefficient range input: {STEP10_1B2B_LOEO_DETAIL_PATH}",
        f"- direct-chain boundary input read only: {STEP10_1A_SUMMARY_PATH}",
        f"- formal direct series input read only: {STEP10_1A_DIRECT_PATH}",
        f"- formal direct aligned monthly input read only: {STEP10_1A_ALIGNED_PATH}",
        f"- deterministic model summary input read only: {STEP10_1A_MODEL_SUMMARY_PATH}",
        f"- predictor collinearity input read only: {STEP10_1A_COLLINEARITY_PATH}",
        f"- Figure 2 same-chain reproduction input read only: {STEP10_1A_FIG02_CHECK_PATH}",
        f"- moving-block bootstrap summary input read only: {STEP10_1B1_SUMMARY_PATH}",
        f"- moving-block bootstrap model summary input read only: {STEP10_1B1_BOOT_PATH}",
        f"- moving-block bootstrap increment input read only: {STEP10_1B1_INCREMENT_PATH}",
        f"- spatial-support audit input read only: {STEP10_1B2A_SPATIAL_PATH}",
        f"- blocked-CV fold inventory input read only: {STEP10_1B2A_FOLD_PATH}",
        f"- blocked-CV summary input read only: {STEP10_1B2A_SUMMARY_PATH}",
        f"- ENSO event inventory input read only: {STEP10_1B2B_EVENT_INVENTORY_PATH}",
        f"- leave-one-event-out summary input read only: {STEP10_1B2B_LOEO_SUMMARY_PATH}",
        f"- event-held-out monthly input read only: {STEP10_1B2B_EVENT_PRED_PATH}",
        f"- event-held-out skill-by-event input read only: {STEP10_1B2B_EVENT_SKILL_PATH}",
        f"- ENSO-event robustness summary input read only: {STEP10_1B2B_SUMMARY_PATH}",
        f"- Figure 9 method boundary reference if present: {FIG09_METHOD_PATH if FIG09_METHOD_PATH.exists() else 'not present'}",
        f"- Figure 9 caption boundary reference if present: {FIG09_CAPTION_PATH if FIG09_CAPTION_PATH.exists() else 'not present'}",
        "",
        "Plotting scope and guards",
        "- formal plot generated from audited fixed outputs only.",
        "- no formal direct monthly series recomputed during plotting.",
        "- no HCCF/HCTB/LCSP metrics recomputed during plotting.",
        "- no models refit during plotting.",
        "- no bootstrap, CV or event robustness recomputed during plotting.",
        "- panels (a)-(c) use purged blocked-CV monthly OOF predictions only; no in-sample fit and no event-held-out estimates are used as the scatter source.",
        "- panel (d) uses blocked-CV deterministic increments with moving-block-bootstrap CI from Step10-1B-2A only.",
        "- panel (e) uses pooled ENSO-event-held-out deterministic increments with event-cluster-bootstrap CI from Step10-1B-2B only.",
        "- panel (f) distinguishes moving-block 95% CI from leave-one-event-out coefficient range.",
        "",
        "Approval chain",
        "- Figure 10 approved after formal direct-chain reproduction, spatial-support audit, moving-block bootstrap, purged blocked CV and ENSO-event robustness gates.",
        "- models plotted: M1 = LCSP + HCCF; M3 = LCSP + HCCF + HCTB.",
        "- M2 is not used as the main Figure 10 gate.",
        "",
        "Method boundary",
        "- regions = WP, CP, EP only; TP not plotted.",
        "- formal direct response terminology = fixed regional-box mean direct daytime Net CRE anomaly.",
        "- direct response uses all finite direct-field grid cells within fixed regional boxes.",
        "- separate ocean mask used = False.",
        "- metric/direct spatial support compatible = True.",
        "- anomaly convention = calendar-month climatology removed, not detrended, not standardized as preprocessing.",
        "- HCCF comparator only.",
        "- HCTB uses remapped current six-cell Deep convective definition.",
        "- HCTB current Deep convective members = 29,30,35,36,41,42.",
        "- LCSP equals negative current Low cloud occurrence.",
        "- DCEP excluded.",
        "- no CE/GCE.",
        "- no environmental variables.",
        "- no old regional file.",
        "- no all-42 joint strict mask.",
        "",
        "Blocked-CV design",
        f"- fold inventory rows = {len(fold_df)}.",
        "- blocked CV design: 21 folds, 12-month test blocks except final 8-month block, 6-month purge each side.",
        "",
        "ENSO-event design",
        f"- ENSO event inventory = {int((event_inventory_df['phase'] == 'El Nino').sum())} El Nino events / {int(event_inventory_df.loc[event_inventory_df['phase'] == 'El Nino', 'n_months'].sum())} months, "
        f"{int((event_inventory_df['phase'] == 'La Nina').sum())} La Nina events / {int(event_inventory_df.loc[event_inventory_df['phase'] == 'La Nina', 'n_months'].sum())} months.",
        f"- pooled ENSO-event held-out months = {int(pooled_df['n_heldout_ENSO_months'].iloc[0])}.",
        "",
        "Plotted increment values",
    ]
    for region in REGIONS:
        method_lines.extend(
            [
                f"- {region} blocked-CV delta OOF R2(M3-M1) = {EXPECTED_BLOCKED_CV[region][0]:+.6f}, 95% CI [{EXPECTED_BLOCKED_CV[region][1]:+.6f}, {EXPECTED_BLOCKED_CV[region][2]:+.6f}].",
                f"- {region} ENSO-event-held-out delta OOF R2(M3-M1) = {EXPECTED_EVENT_HELDOUT[region][0]:+.6f}, 95% CI [{EXPECTED_EVENT_HELDOUT[region][1]:+.6f}, {EXPECTED_EVENT_HELDOUT[region][2]:+.6f}].",
                f"- {region} M3 HCTB coefficient = {EXPECTED_HCTB[region][0]:+.6f}, moving-block 95% CI [{EXPECTED_HCTB[region][1]:+.6f}, {EXPECTED_HCTB[region][2]:+.6f}].",
                f"- {region} leave-one-event-out HCTB coefficient sign consistency = {float(loeo_summary_map.loc[region, 'fraction_events_HCTB_coefficient_sign_consistent']):.3f}.",
                f"- {region} leave-one-event-out HCTB coefficient range = [{float(loeo_df.loc[loeo_df['region'] == region, 'M3_HCTB_coefficient_remaining'].min()):+.6f}, "
                f"{float(loeo_df.loc[loeo_df['region'] == region, 'M3_HCTB_coefficient_remaining'].max()):+.6f}].",
            ]
        )
    method_lines.extend(
        [
            "",
            "Outputs",
            f"- plot data output: {OUT_PLOT_DATA}",
            f"- png output: {OUT_PNG}",
            f"- pdf output: {OUT_PDF}",
            f"- caption output: {OUT_CAPTION}",
            f"- method checks output: {OUT_CHECKS}",
            "",
            "Interpretation boundary",
            "- Figure 10 evaluates diagnostic linkage / held-out diagnostic skill only.",
            "- Figure 10 is not independent validation.",
            "- Figure 10 is not prediction skill.",
            "- Figure 10 is not direct-response reconstruction.",
            "- Figure 10 does not establish causal control or complete attribution.",
            "- no prediction skill / control / verified mechanism / independent validation / exact reconstruction / full attribution interpretation.",
            "- direct response must not be described as ocean-only.",
            "- no Figure 11 and no supplementary figures.",
            "- main-text figure count after Figure 10 = 10.",
        ]
    )
    OUT_CHECKS.write_text("\n".join(method_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
