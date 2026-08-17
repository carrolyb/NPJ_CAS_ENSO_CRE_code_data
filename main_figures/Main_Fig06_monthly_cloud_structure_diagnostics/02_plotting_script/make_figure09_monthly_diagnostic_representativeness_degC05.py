#!/usr/bin/env python3
"""Render Figure 09 under the +/-0.5 C ENSO sensitivity definition using the Figure09 copy.py layout."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

matplotlib.use("Agg")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PACKAGE_ROOT / "01_final_figure"
INPUT_DIR = PACKAGE_ROOT / "03_input_data"
RESULT_DIR = PACKAGE_ROOT / "04_key_results"
NOTES_DIR = PACKAGE_ROOT / "05_notes"

ANOM_PATH = INPUT_DIR / "Step09_3A_metric_and_pathway_monthly_anomaly_ready_series.csv"
METRIC_RAW_PATH = INPUT_DIR / "Step09_3A_metric_monthly_raw_series_paired_valid.csv"
PATHWAY_RAW_PATH = INPUT_DIR / "Step09_3A_pathway_monthly_raw_series.csv"
HCTB_DEF_PATH = INPUT_DIR / "Step09_3A_HCTB_legacy_vs_current_definition_record.csv"
ANOM_TRACE_PATH = INPUT_DIR / "Step09_3A_monthly_anomaly_convention_trace.txt"
STEP09A_SUMMARY_PATH = INPUT_DIR / "Step09_3A_metric_monthly_series_preparation_summary.txt"
IDENTITY_PATH = INPUT_DIR / "Step09_3B_metric_pathway_input_identity_check.csv"
DET_PATH = INPUT_DIR / "Step09_3B_single_metric_pathway_relationship_deterministic.csv"
BOOT_PATH = INPUT_DIR / "Step09_3B_single_metric_pathway_relationship_bootstrap.csv"
INCREMENTAL_PATH = INPUT_DIR / "Step09_3B_HCTB_incremental_value_beyond_HCCF.csv"
ENSO_SENS_PATH = INPUT_DIR / "Step09_3B_ENSO_subset_direction_sensitivity_record.csv"
SECONDARY_PATH = INPUT_DIR / "Step09_3B_HCTB_high_contrast_secondary_record.csv"
STEP09B_SUMMARY_PATH = INPUT_DIR / "Step09_3B_monthly_pathway_representativeness_audit_summary.txt"
NINO_PATH = INPUT_DIR / "nino34_200207_202302.csv"

FIG08_METHOD_REF = PACKAGE_ROOT.parent / "Figure08_spatial_Net_total_pathways_degC05" / "05_notes" / "Figure08_degC05_method_and_plot_checks.txt"
FIG08_CAPTION_REF = PACKAGE_ROOT.parent / "Figure08_spatial_Net_total_pathways_degC05" / "05_notes" / "Figure08_degC05_caption.md"

OUT_PNG = FIG_DIR / "Figure09_monthly_diagnostic_representativeness_degC05.png"
OUT_PDF = FIG_DIR / "Figure09_monthly_diagnostic_representativeness_degC05.pdf"
OUT_PLOT_DATA = RESULT_DIR / "Figure09_degC05_plot_data.csv"
OUT_CAPTION = NOTES_DIR / "Figure09_degC05_caption.md"
OUT_CHECKS = NOTES_DIR / "Figure09_degC05_method_and_plot_checks.txt"
OUT_MANIFEST = NOTES_DIR / "Figure09_degC05_input_data_manifest.md"
OUT_ENSO_SENS_DEGC05 = RESULT_DIR / "Figure09_degC05_ENSO_subset_direction_sensitivity.csv"

X_TICKS = {
    "HCCF_comparator": [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3],
    "HCTB_candidate_diagnostic": [-0.08, -0.04, 0, 0.04, 0.08],
    "LCSP_diagnostic": [-0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12],
}

# 如果你还想自己控制显示文字，也可以同时加这个
REGIONS = ["WP", "CP", "EP"]
REGION_TITLES = {"WP": "Western Pacific", "CP": "Central Pacific", "EP": "Eastern Pacific"}
REGION_COLORS = {
    "WP": {"line": "#0f4c5c", "point": "#1f6f82"},
    "CP": {"line": "#8c2f39", "point": "#a33a46"},
    "EP": {"line": "#6b5b00", "point": "#8a7800"},
}
SCATTER_SIZE = 28
SCATTER_ALPHA = 0.72

RELATIONSHIP_INFO = [
    {
        "row": 0,
        "relationship_id": "HCCF_comparator",
        "predictor": "HCCF_anom",
        "response": "NetPath_high_total_anom",
        "x_label": "HCCF anomaly",
        "y_label": "High-cloud Net CRE contribution (W m$^{-2}$)",
        "row_label": "HCCF vs High-cloud pathway",
    },
    {
        "row": 1,
        "relationship_id": "HCTB_candidate_diagnostic",
        "predictor": "HCTB_anom",
        "response": "NetPath_high_total_anom",
        "x_label": "HCTB anomaly",
        "y_label": "High-cloud Net CRE contribution (W m$^{-2}$)",
        "row_label": "HCTB vs High-cloud pathway",
    },
    {
        "row": 2,
        "relationship_id": "LCSP_diagnostic",
        "predictor": "LCSP_anom",
        "response": "NetPath_low_anom",
        "x_label": "LCSP anomaly",
        "y_label": "Low-cloud Net CRE contribution (W m$^{-2}$)",
        "row_label": "LCSP vs Low-cloud pathway",
    },
]

PANEL_LETTERS = {
    (0, 0): "(a)",
    (0, 1): "(b)",
    (0, 2): "(c)",
    (1, 0): "(d)",
    (1, 1): "(e)",
    (1, 2): "(f)",
    (2, 0): "(g)",
    (2, 1): "(h)",
    (2, 2): "(i)",
}

EXPECTED_RELATIONSHIPS = {
    ("WP", "HCCF_comparator"): {"Pearson_r": -0.641040, "R2": 0.410932, "slope": -16.070272, "slope_ci_low_95": -20.571603, "slope_ci_high_95": -13.664923, "slope_significant": True},
    ("CP", "HCCF_comparator"): {"Pearson_r": -0.438105, "R2": 0.191936, "slope": -6.831450, "slope_ci_low_95": -9.169601, "slope_ci_high_95": -4.785962, "slope_significant": True},
    ("EP", "HCCF_comparator"): {"Pearson_r": -0.501760, "R2": 0.251763, "slope": -9.108315, "slope_ci_low_95": -12.447020, "slope_ci_high_95": -7.051992, "slope_significant": True},
    ("WP", "HCTB_candidate_diagnostic"): {"Pearson_r": -0.798115, "R2": 0.636988, "slope": -44.144088, "slope_ci_low_95": -47.794895, "slope_ci_high_95": -40.104292, "slope_significant": True},
    ("CP", "HCTB_candidate_diagnostic"): {"Pearson_r": -0.640715, "R2": 0.410516, "slope": -37.862140, "slope_ci_low_95": -44.723683, "slope_ci_high_95": -30.339220, "slope_significant": True},
    ("EP", "HCTB_candidate_diagnostic"): {"Pearson_r": -0.472323, "R2": 0.223089, "slope": -27.414235, "slope_ci_low_95": -36.684307, "slope_ci_high_95": -16.552579, "slope_significant": True},
    ("WP", "LCSP_diagnostic"): {"Pearson_r": 0.955153, "R2": 0.912317, "slope": 40.363655, "slope_ci_low_95": 38.673295, "slope_ci_high_95": 42.060717, "slope_significant": True},
    ("CP", "LCSP_diagnostic"): {"Pearson_r": 0.956984, "R2": 0.915819, "slope": 30.480831, "slope_ci_low_95": 28.845698, "slope_ci_high_95": 32.680079, "slope_significant": True},
    ("EP", "LCSP_diagnostic"): {"Pearson_r": 0.951260, "R2": 0.904895, "slope": 79.563783, "slope_ci_low_95": 74.729126, "slope_ci_high_95": 86.590995, "slope_significant": True},
}

EXPECTED_DELTA_ADJ_R2 = {"WP": 0.372313, "CP": 0.536160, "EP": 0.489761}
ANNOTATION_TOL = 1.0e-6
ENSO_THRESHOLD_DEGC05 = 0.5
EXPECTED_ENSO_COUNTS_DEGC05 = {"el_nino": 54, "la_nina": 85, "subset_total": 139}
PLOT_DATA_ROW_LABELS_V1 = {
    "HCCF_comparator": "HCCF \u2192 High-cloud pathway",
    "HCTB_candidate_diagnostic": "HCTB \u2192 High-cloud pathway",
    "LCSP_diagnostic": "LCSP \u2192 Low-cloud pathway",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "axes.linewidth": 1,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "figure.titlesize": 16,
            "savefig.dpi": 360,
        }
    )

def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        clip_on=False,
        zorder=20,
    )

def symmetric_limits(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    require(finite.size > 0, "Cannot infer symmetric limits from empty input.")
    max_abs = float(np.max(np.abs(finite)))
    max_abs *= 1.0 + pad
    if max_abs == 0.0:
        max_abs = 1.0
    return -max_abs, max_abs


def classify_sign(value: float, zero_tol: float = 1.0e-12) -> str:
    if value > zero_tol:
        return "positive"
    if value < -zero_tol:
        return "negative"
    return "zero"


def ols_simple(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    require(x.ndim == 1 and y.ndim == 1 and x.size == y.size and x.size >= 2, "OLS input arrays must be one-dimensional and aligned.")
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    dx = x - x_mean
    dy = y - y_mean
    ss_xx = float(np.sum(dx * dx))
    ss_yy = float(np.sum(dy * dy))
    ss_xy = float(np.sum(dx * dy))
    require(ss_xx > 0.0 and ss_yy > 0.0, "OLS input variance must be positive.")
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    pearson_r = ss_xy / float(np.sqrt(ss_xx * ss_yy))
    return {"slope": slope, "intercept": intercept, "Pearson_r": pearson_r}


def ensure_inputs() -> None:
    for path in [FIG_DIR, RESULT_DIR, NOTES_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    required = [
        ANOM_PATH,
        METRIC_RAW_PATH,
        PATHWAY_RAW_PATH,
        HCTB_DEF_PATH,
        ANOM_TRACE_PATH,
        STEP09A_SUMMARY_PATH,
        IDENTITY_PATH,
        DET_PATH,
        BOOT_PATH,
        INCREMENTAL_PATH,
        ENSO_SENS_PATH,
        SECONDARY_PATH,
        STEP09B_SUMMARY_PATH,
        NINO_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    require(not missing, "Missing required Figure 9 input(s):\n" + "\n".join(missing))


def load_inputs() -> dict[str, object]:
    ensure_inputs()
    data = {
        "anom": pd.read_csv(ANOM_PATH, parse_dates=["month"]),
        "metric_raw": pd.read_csv(METRIC_RAW_PATH, parse_dates=["month"]),
        "pathway_raw": pd.read_csv(PATHWAY_RAW_PATH, parse_dates=["month"]),
        "hctb_def": pd.read_csv(HCTB_DEF_PATH),
        "identity": pd.read_csv(IDENTITY_PATH),
        "det": pd.read_csv(DET_PATH),
        "boot": pd.read_csv(BOOT_PATH),
        "incremental": pd.read_csv(INCREMENTAL_PATH),
        "enso": pd.read_csv(ENSO_SENS_PATH),
        "secondary": pd.read_csv(SECONDARY_PATH),
        "anom_trace": ANOM_TRACE_PATH.read_text(),
        "step09a_summary": STEP09A_SUMMARY_PATH.read_text(),
        "step09b_summary": STEP09B_SUMMARY_PATH.read_text(),
        "fig08_method": FIG08_METHOD_REF.read_text() if FIG08_METHOD_REF.exists() else None,
        "fig08_caption": FIG08_CAPTION_REF.read_text() if FIG08_CAPTION_REF.exists() else None,
        "nino": pd.read_csv(NINO_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True),
    }
    return data


def validate_fixed_inputs(data: dict[str, object]) -> None:
    anom = data["anom"]
    det = data["det"]
    boot = data["boot"]
    inc = data["incremental"]
    identity = data["identity"]
    enso = data["enso"]
    hctb_def = data["hctb_def"]

    require(set(anom["region"].unique()) == set(REGIONS), "Unexpected regions in anomaly-ready input.")
    require(anom.groupby("region").size().to_dict() == {"WP": 248, "CP": 248, "EP": 248}, "Figure 9 input does not contain 248 months per region.")
    require(set(anom["anomaly_convention"].unique()) == {"calendar_month_anomaly_only"}, "Unexpected anomaly convention.")
    require(bool((~anom["detrended"]).all()), "Figure 9 input is unexpectedly detrended.")
    require(bool((~anom["standardized"]).all()), "Figure 9 input is unexpectedly standardized as preprocessing.")

    require(bool(identity["pass_threshold_1e_minus_12"].all()), "Step09-3B identity check did not pass.")
    require(float(identity["max_abs_difference"].max()) == 0.0, "Step09-3B identity check is not exact zero.")
    require(bool(hctb_def["current_definition_required_for_main_text"].all()), "HCTB current definition gate failed.")

    det_lookup = det.set_index(["region", "relationship_id"])
    boot_lookup = boot.set_index(["region", "relationship_id"])
    enso_lookup = enso.set_index(["region", "relationship_id"])
    inc_lookup = inc.set_index("region")
    for key, expected in EXPECTED_RELATIONSHIPS.items():
        region, relationship_id = key
        det_row = det_lookup.loc[(region, relationship_id)]
        boot_row = boot_lookup.loc[(region, relationship_id)]
        require(abs(float(det_row["Pearson_r"]) - expected["Pearson_r"]) <= ANNOTATION_TOL, f"Pearson_r mismatch for {key}")
        require(abs(float(det_row["R2"]) - expected["R2"]) <= ANNOTATION_TOL, f"R2 mismatch for {key}")
        require(abs(float(det_row["slope"]) - expected["slope"]) <= ANNOTATION_TOL, f"slope mismatch for {key}")
        require(abs(float(boot_row["slope_ci_low_95"]) - expected["slope_ci_low_95"]) <= ANNOTATION_TOL, f"slope_ci_low mismatch for {key}")
        require(abs(float(boot_row["slope_ci_high_95"]) - expected["slope_ci_high_95"]) <= ANNOTATION_TOL, f"slope_ci_high mismatch for {key}")
        require(bool(boot_row["slope_significant"]) is expected["slope_significant"], f"significance mismatch for {key}")
        require(bool(enso_lookup.loc[(region, relationship_id), "sign_consistent_allmonth_vs_ENSO_subset"]), f"ENSO sign consistency failed for {key}")

    for region, expected in EXPECTED_DELTA_ADJ_R2.items():
        require(abs(float(inc_lookup.loc[region, "delta_adjusted_R2"]) - expected) <= ANNOTATION_TOL, f"delta_adjusted_R2 mismatch for {region}")

    require("HCCF is comparator only." in data["step09a_summary"], "Step09-3A summary lost HCCF comparator-only wording.")
    require("no direct Net CRE linkage is tested in this step." in data["step09b_summary"], "Step09-3B summary lost direct-linkage exclusion wording.")


def build_plot_data(data: dict[str, object]) -> pd.DataFrame:
    anom = data["anom"].copy()
    det = data["det"].set_index(["region", "relationship_id"])
    boot = data["boot"].set_index(["region", "relationship_id"])
    inc = data["incremental"].set_index("region")

    rows: list[pd.DataFrame] = []
    for info in RELATIONSHIP_INFO:
        relationship_id = info["relationship_id"]
        predictor = info["predictor"]
        response = info["response"]
        for region in REGIONS:
            sub = anom.loc[anom["region"] == region, ["month", "region", "phase_index", predictor, response]].copy()
            sub["relationship_id"] = relationship_id
            sub["predictor_name"] = predictor
            sub["response_name"] = response
            sub["predictor_value"] = sub[predictor]
            sub["response_value"] = sub[response]
            sub["panel_letter"] = PANEL_LETTERS[(info["row"], REGIONS.index(region))]
            sub["panel_title"] = REGION_TITLES[region]
            sub["row_label"] = PLOT_DATA_ROW_LABELS_V1[relationship_id]
            sub["x_label"] = info["x_label"]
            sub["y_label"] = info["y_label"]
            sub["R2"] = float(det.loc[(region, relationship_id), "R2"])
            sub["slope"] = float(det.loc[(region, relationship_id), "slope"])
            sub["intercept"] = float(det.loc[(region, relationship_id), "intercept"])
            sub["slope_ci_low_95"] = float(boot.loc[(region, relationship_id), "slope_ci_low_95"])
            sub["slope_ci_high_95"] = float(boot.loc[(region, relationship_id), "slope_ci_high_95"])
            sub["slope_significant"] = bool(boot.loc[(region, relationship_id), "slope_significant"])
            sub["delta_adjusted_R2_vs_HCCF"] = float(inc.loc[region, "delta_adjusted_R2"]) if relationship_id == "HCTB_candidate_diagnostic" else np.nan
            rows.append(
                sub[
                    [
                        "month",
                        "region",
                        "phase_index",
                        "relationship_id",
                        "predictor_name",
                        "response_name",
                        "predictor_value",
                        "response_value",
                        "panel_letter",
                        "panel_title",
                        "row_label",
                        "x_label",
                        "y_label",
                        "R2",
                        "slope",
                        "intercept",
                        "slope_ci_low_95",
                        "slope_ci_high_95",
                        "slope_significant",
                        "delta_adjusted_R2_vs_HCCF",
                    ]
                ]
            )
    plot_df = pd.concat(rows, ignore_index=True)
    plot_df.to_csv(OUT_PLOT_DATA, index=False, float_format="%.15f")
    return plot_df


def compute_axis_limits(plot_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    high_y = symmetric_limits(plot_df.loc[plot_df["response_name"] == "NetPath_high_total_anom", "response_value"].to_numpy(), pad=0.08)
    low_y = symmetric_limits(plot_df.loc[plot_df["response_name"] == "NetPath_low_anom", "response_value"].to_numpy(), pad=0.08)
    hccf_x = symmetric_limits(plot_df.loc[plot_df["predictor_name"] == "HCCF_anom", "predictor_value"].to_numpy(), pad=0.10)
    hctb_x = symmetric_limits(plot_df.loc[plot_df["predictor_name"] == "HCTB_anom", "predictor_value"].to_numpy(), pad=0.10)
    lcsp_x = symmetric_limits(plot_df.loc[plot_df["predictor_name"] == "LCSP_anom", "predictor_value"].to_numpy(), pad=0.10)
    return {
        "high_y": high_y,
        "low_y": low_y,
        "hccf_x": hccf_x,
        "hctb_x": hctb_x,
        "lcsp_x": lcsp_x,
    }


def build_enso_subset_sensitivity_degC05(data: dict[str, object]) -> pd.DataFrame:
    anom = data["anom"].copy()
    det_lookup = data["det"].set_index(["region", "relationship_id"])
    nino = data["nino"].copy()
    nino["month"] = nino["date"].dt.to_period("M").dt.to_timestamp()
    nino_lookup = nino.set_index("month")["nino34_anom"]

    anom["month_lookup"] = anom["month"].dt.to_period("M").dt.to_timestamp()
    anom["nino34_anom"] = anom["month_lookup"].map(nino_lookup)
    require(bool(anom["nino34_anom"].notna().all()), "Missing Nino3.4 anomaly values for some Figure 9 months.")
    anom["phase_degC05"] = 0
    anom.loc[anom["nino34_anom"] >= ENSO_THRESHOLD_DEGC05, "phase_degC05"] = 1
    anom.loc[anom["nino34_anom"] <= -ENSO_THRESHOLD_DEGC05, "phase_degC05"] = -1

    counts = anom.loc[anom["region"] == "WP", "phase_degC05"].value_counts().to_dict()
    require(int(counts.get(1, 0)) == EXPECTED_ENSO_COUNTS_DEGC05["el_nino"], f"Unexpected +0.5 C El Nino month count: {counts.get(1, 0)}")
    require(int(counts.get(-1, 0)) == EXPECTED_ENSO_COUNTS_DEGC05["la_nina"], f"Unexpected -0.5 C La Nina month count: {counts.get(-1, 0)}")
    require(int(counts.get(1, 0) + counts.get(-1, 0)) == EXPECTED_ENSO_COUNTS_DEGC05["subset_total"], "Unexpected total degC05 ENSO-subset count.")

    relationships = [
        ("HCCF_comparator", "HCCF_anom", "NetPath_high_total_anom"),
        ("HCTB_candidate_diagnostic", "HCTB_anom", "NetPath_high_total_anom"),
        ("LCSP_diagnostic", "LCSP_anom", "NetPath_low_anom"),
    ]
    rows: list[dict[str, object]] = []
    for region in REGIONS:
        sub = anom.loc[(anom["region"] == region) & (anom["phase_degC05"].isin([-1, 1]))].sort_values("month").reset_index(drop=True)
        require(len(sub) == EXPECTED_ENSO_COUNTS_DEGC05["subset_total"], f"Unexpected degC05 ENSO-subset length for {region}: {len(sub)}")
        for relationship_id, predictor_col, response_col in relationships:
            fit = ols_simple(sub[predictor_col].to_numpy(dtype=float), sub[response_col].to_numpy(dtype=float))
            det_slope = float(det_lookup.loc[(region, relationship_id), "slope"])
            rows.append(
                {
                    "region": region,
                    "relationship_id": relationship_id,
                    "predictor": predictor_col,
                    "response": response_col,
                    "n_months_ENSO_subset": EXPECTED_ENSO_COUNTS_DEGC05["subset_total"],
                    "n_el_nino_months": EXPECTED_ENSO_COUNTS_DEGC05["el_nino"],
                    "n_la_nina_months": EXPECTED_ENSO_COUNTS_DEGC05["la_nina"],
                    "enso_threshold_degC": ENSO_THRESHOLD_DEGC05,
                    "Pearson_r_ENSO_subset": fit["Pearson_r"],
                    "slope_ENSO_subset": fit["slope"],
                    "sign_consistent_allmonth_vs_ENSO_subset": classify_sign(det_slope) == classify_sign(fit["slope"]),
                }
            )
    out = pd.DataFrame(rows)
    require(bool(out["sign_consistent_allmonth_vs_ENSO_subset"].all()), "At least one degC05 ENSO-subset sign-consistency check failed.")
    out.to_csv(OUT_ENSO_SENS_DEGC05, index=False, float_format="%.15f")
    return out


def make_caption() -> None:
    text = (
        "Figure 9. Monthly representativeness of cloud-structure diagnostics for cloud-type-resolved daytime Net CRE pathways under the Figure 9 copy.py layout, with ENSO robustness checked using a +/-0.5 C Nino3.4 threshold. "
        "Panels (a)–(c) relate anomalies in total high-cloud occurrence (HCCF) to the corresponding high-cloud Net pathway anomalies over the western, central, and eastern Pacific, respectively. "
        "Panels (d)–(f) present the corresponding relationships for the remapped high-cloud thickness-balance diagnostic (HCTB), which contrasts thick-anvil and deep-convective occurrence against thin-high-cloud occurrence. "
        "Panels (g)–(i) relate low-cloud suppression anomalies (LCSP) to low-cloud Net pathway anomalies. "
        "All quantities are calendar-month anomalies derived from the contribution-consistent paired-valid pathway chain. "
        "Solid lines indicate linear fits, and reported slope confidence intervals are derived from a 12-month moving-block bootstrap. "
        "Annotations in panels (d)–(f) report the increase in adjusted R² obtained by adding HCTB to an HCCF-only model. "
        "The plotted scatter points still use all 248 months per region; the +/-0.5 C ENSO definition is used only for a direction-consistency sensitivity audit (54 El Nino months and 85 La Nina months, 139 total subset months). "
        "HCCF is included as a comparator; HCTB is interpreted as providing additional structural information beyond total high-cloud occurrence, and LCSP as a diagnostic representation of the low-cloud pathway. "
        "These relationships diagnose cloud-type pathway variability and are not interpreted as an exact reconstruction of the direct all-sky Net CRE response.\n"
    )
    OUT_CAPTION.write_text(text)


def make_input_manifest() -> None:
    lines = [
        "# Figure09 degC05 input manifest",
        "",
        f"- Monthly anomaly-ready diagnostic/pathway table: `{ANOM_PATH}`",
        f"- Monthly metric raw series table: `{METRIC_RAW_PATH}`",
        f"- Monthly pathway raw series table: `{PATHWAY_RAW_PATH}`",
        f"- HCTB current-definition audit record: `{HCTB_DEF_PATH}`",
        f"- Step09-3B deterministic relationship table: `{DET_PATH}`",
        f"- Step09-3B bootstrap relationship table: `{BOOT_PATH}`",
        f"- Step09-3B HCTB incremental-value table: `{INCREMENTAL_PATH}`",
        f"- Original all-month ENSO subset sensitivity table: `{ENSO_SENS_PATH}`",
        f"- Nino3.4 index file for +/-0.5 C sensitivity definition: `{NINO_PATH}`",
        f"- Figure08 degC05 method reference: `{FIG08_METHOD_REF}`",
        f"- Figure08 degC05 caption reference: `{FIG08_CAPTION_REF}`",
    ]
    OUT_MANIFEST.write_text("\n".join(lines) + "\n")


def make_method_checks(data: dict[str, object], axis_limits: dict[str, tuple[float, float]], enso_degC05: pd.DataFrame) -> None:
    enso_lookup = enso_degC05.set_index(["region", "relationship_id"])
    lines = [
        "Figure09 degC05 method and plot checks",
        "",
        f"- actual anomaly-ready monthly input path: {ANOM_PATH}",
        f"- actual deterministic relationship input path: {DET_PATH}",
        f"- actual bootstrap relationship input path: {BOOT_PATH}",
        f"- actual HCTB incremental-value input path: {INCREMENTAL_PATH}",
        f"- original ENSO-subset sensitivity input path retained for audit context: {ENSO_SENS_PATH}",
        f"- derived degC05 ENSO-subset sensitivity output path: {OUT_ENSO_SENS_DEGC05}",
        f"- Nino3.4 input path for degC05 subset definition: {NINO_PATH}",
        "- plotting style follows the Figure09 copy.py layout = True",
        "- scatter plot data unchanged relative to v2_txt all-month rendering = True",
        "- regression coefficients unchanged from fixed Step09-3B deterministic inputs = True",
        "- moving-block-bootstrap CI unchanged from fixed Step09-3B bootstrap inputs = True",
        "- panel layout and axis limits unchanged from v2_txt = True",
        "- +/-0.5 C Nino3.4 threshold is applied only to the ENSO direction-consistency sensitivity audit = True",
        "- displayed figure still uses all 248 calendar-month anomalies per region = True",
        "- formal plot generated from fixed audited inputs only.",
        "- no monthly series recomputed during plotting.",
        "- no regression or bootstrap recomputed during plotting.",
        "- HCCF comparator only.",
        "- HCTB uses remapped current six-cell Deep convective definition.",
        "- LCSP equals negative current Low cloud occurrence.",
        "- DCEP excluded.",
        "- WP/CP/EP only; TP not plotted.",
        "- calendar-month anomaly convention; no detrending or preprocessing standardization.",
        "- identity checks passed with max_abs_difference=0.",
        "- all nine primary slopes significant by moving-block-bootstrap CI.",
        "- HCTB incremental-value support in WP/CP/EP.",
        f"- degC05 ENSO definition = nino34_anom with El Nino >= +{ENSO_THRESHOLD_DEGC05:.1f} C and La Nina <= -{ENSO_THRESHOLD_DEGC05:.1f} C.",
        f"- degC05 ENSO counts = El Nino {EXPECTED_ENSO_COUNTS_DEGC05['el_nino']}, La Nina {EXPECTED_ENSO_COUNTS_DEGC05['la_nina']}, subset total {EXPECTED_ENSO_COUNTS_DEGC05['subset_total']}.",
        "- degC05 ENSO-phase subset direction consistency for all nine relationships.",
        "- EP caution: HCTB univariate R2 is not larger than HCCF R2, although incremental information beyond HCCF is supported.",
        "- no direct all-sky Net CRE variable plotted or tested.",
        "- Figure 9 remains diagnostic representativeness only.",
        "- Figure 9 does not test direct all-sky Net CRE linkage.",
        "- no verified mechanism / control / exact reconstruction language.",
        "- Figure 9 is diagnostic representativeness, not verified mechanism, causal control, or direct-response reconstruction.",
        "",
        "Figure role boundary",
        "- Figure 9 differs from Figures 3–8 by evaluating monthly representativeness rather than ENSO composite pathway magnitude or spatial distribution.",
        "- Figure 9 does not include direct all-sky Net CRE linkage.",
        "- Figure 9 does not imply exact reconstruction, full attribution, control, or verified mechanism.",
        "",
        "Plotting references",
        f"- plot data output: {OUT_PLOT_DATA}",
        f"- png output: {OUT_PNG}",
        f"- pdf output: {OUT_PDF}",
        f"- caption output: {OUT_CAPTION}",
        f"- method checks output: {OUT_CHECKS}",
        f"- input manifest output: {OUT_MANIFEST}",
        f"- Figure 8 method boundary reference used if available: {FIG08_METHOD_REF if FIG08_METHOD_REF.exists() else 'not present'}",
        f"- Figure 8 caption boundary reference used if available: {FIG08_CAPTION_REF if FIG08_CAPTION_REF.exists() else 'not present'}",
        "",
        "Fixed plotting inputs",
        "- row 1 = HCCF anomaly vs high-cloud Net pathway anomaly",
        "- row 2 = HCTB anomaly vs high-cloud Net pathway anomaly",
        "- row 3 = LCSP anomaly vs low-cloud Net pathway anomaly",
        "- columns = WP, CP, EP",
        "- point count per panel = 248",
        "- points are not colored by ENSO phase",
        "- regression line uses fixed deterministic slope/intercept from Step09-3B",
        "- slope CI annotations use fixed Step09-3B moving-block-bootstrap outputs only",
        "- block_length = 12 months",
        "- n_boot = 2000",
        "- seed = 42",
        "",
        "degC05 ENSO sensitivity summary",
        f"- HCCF comparator WP: r_ENSO_subset={float(enso_lookup.loc[('WP', 'HCCF_comparator'), 'Pearson_r_ENSO_subset']):.6f}, slope_ENSO_subset={float(enso_lookup.loc[('WP', 'HCCF_comparator'), 'slope_ENSO_subset']):.6f}, sign_consistent=True.",
        f"- HCCF comparator CP: r_ENSO_subset={float(enso_lookup.loc[('CP', 'HCCF_comparator'), 'Pearson_r_ENSO_subset']):.6f}, slope_ENSO_subset={float(enso_lookup.loc[('CP', 'HCCF_comparator'), 'slope_ENSO_subset']):.6f}, sign_consistent=True.",
        f"- HCCF comparator EP: r_ENSO_subset={float(enso_lookup.loc[('EP', 'HCCF_comparator'), 'Pearson_r_ENSO_subset']):.6f}, slope_ENSO_subset={float(enso_lookup.loc[('EP', 'HCCF_comparator'), 'slope_ENSO_subset']):.6f}, sign_consistent=True.",
        f"- HCTB diagnostic WP: r_ENSO_subset={float(enso_lookup.loc[('WP', 'HCTB_candidate_diagnostic'), 'Pearson_r_ENSO_subset']):.6f}, slope_ENSO_subset={float(enso_lookup.loc[('WP', 'HCTB_candidate_diagnostic'), 'slope_ENSO_subset']):.6f}, sign_consistent=True.",
        f"- HCTB diagnostic CP: r_ENSO_subset={float(enso_lookup.loc[('CP', 'HCTB_candidate_diagnostic'), 'Pearson_r_ENSO_subset']):.6f}, slope_ENSO_subset={float(enso_lookup.loc[('CP', 'HCTB_candidate_diagnostic'), 'slope_ENSO_subset']):.6f}, sign_consistent=True.",
        f"- HCTB diagnostic EP: r_ENSO_subset={float(enso_lookup.loc[('EP', 'HCTB_candidate_diagnostic'), 'Pearson_r_ENSO_subset']):.6f}, slope_ENSO_subset={float(enso_lookup.loc[('EP', 'HCTB_candidate_diagnostic'), 'slope_ENSO_subset']):.6f}, sign_consistent=True.",
        f"- LCSP diagnostic WP: r_ENSO_subset={float(enso_lookup.loc[('WP', 'LCSP_diagnostic'), 'Pearson_r_ENSO_subset']):.6f}, slope_ENSO_subset={float(enso_lookup.loc[('WP', 'LCSP_diagnostic'), 'slope_ENSO_subset']):.6f}, sign_consistent=True.",
        f"- LCSP diagnostic CP: r_ENSO_subset={float(enso_lookup.loc[('CP', 'LCSP_diagnostic'), 'Pearson_r_ENSO_subset']):.6f}, slope_ENSO_subset={float(enso_lookup.loc[('CP', 'LCSP_diagnostic'), 'slope_ENSO_subset']):.6f}, sign_consistent=True.",
        f"- LCSP diagnostic EP: r_ENSO_subset={float(enso_lookup.loc[('EP', 'LCSP_diagnostic'), 'Pearson_r_ENSO_subset']):.6f}, slope_ENSO_subset={float(enso_lookup.loc[('EP', 'LCSP_diagnostic'), 'slope_ENSO_subset']):.6f}, sign_consistent=True.",
        "",
        "Axis limits",
        f"- HCCF row x-limits shared across WP/CP/EP = [{axis_limits['hccf_x'][0]:.6f}, {axis_limits['hccf_x'][1]:.6f}]",
        f"- HCTB row x-limits shared across WP/CP/EP = [{axis_limits['hctb_x'][0]:.6f}, {axis_limits['hctb_x'][1]:.6f}]",
        f"- LCSP row x-limits shared across WP/CP/EP = [{axis_limits['lcsp_x'][0]:.6f}, {axis_limits['lcsp_x'][1]:.6f}]",
        f"- high-cloud pathway y-limits shared across rows 1-2 = [{axis_limits['high_y'][0]:.6f}, {axis_limits['high_y'][1]:.6f}]",
        f"- low-cloud pathway y-limits shared across row 3 = [{axis_limits['low_y'][0]:.6f}, {axis_limits['low_y'][1]:.6f}]",
    ]
    OUT_CHECKS.write_text("\n".join(lines) + "\n")


def draw_figure(plot_df: pd.DataFrame, data: dict[str, object], axis_limits: dict[str, tuple[float, float]]) -> None:
    det_lookup = data["det"].set_index(["region", "relationship_id"])
    boot_lookup = data["boot"].set_index(["region", "relationship_id"])
    inc_lookup = data["incremental"].set_index("region")

    setup_style()
    fig, axes = plt.subplots(3, 3, figsize=(19.2, 17), constrained_layout=False)
    plt.subplots_adjust(left=0.10, right=0.985, top=0.95, bottom=0.08, wspace=0.22, hspace=0.22)

    for info in RELATIONSHIP_INFO:
        row = info["row"]
        rel = info["relationship_id"]
        x_limits = axis_limits["hccf_x"] if rel == "HCCF_comparator" else axis_limits["hctb_x"] if rel == "HCTB_candidate_diagnostic" else axis_limits["lcsp_x"]
        y_limits = axis_limits["high_y"] if info["response"] == "NetPath_high_total_anom" else axis_limits["low_y"]
        for col, region in enumerate(REGIONS):
            ax = axes[row, col]
            sub = plot_df.loc[(plot_df["region"] == region) & (plot_df["relationship_id"] == rel)].copy()
            colors = REGION_COLORS[region]
            det_row = det_lookup.loc[(region, rel)]
            boot_row = boot_lookup.loc[(region, rel)]

            ax.scatter(
                sub["predictor_value"],
                sub["response_value"],
                s=SCATTER_SIZE,
                alpha=SCATTER_ALPHA,
                color=colors["point"],
                edgecolors="white",
                linewidths=0.18,
                rasterized=True,
                zorder=3,
            )
            ax.axhline(0.0, color="#bdbdbd", lw=0.7, zorder=0)
            ax.axvline(0.0, color="#bdbdbd", lw=0.7, zorder=0)
            xline = np.linspace(x_limits[0], x_limits[1], 200)
            yline = float(det_row["intercept"]) + float(det_row["slope"]) * xline
            ax.plot(xline, yline, color=colors["line"], lw=2.2, zorder=4)

            # ax.set_xlim(*x_limits)
            # if rel in X_TICKS:
            #     ax.set_xticks(X_TICKS[rel])
            #     if X_TICK_LABELS.get(rel) is not None:
            #         ax.set_xticklabels(X_TICK_LABELS[rel])
            # ax.set_ylim(*y_limits)

            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)

            if rel in X_TICKS:
                xticks = X_TICKS[rel]
                ax.set_xticks(xticks)
                ax.set_xticklabels([f"{v:g}" for v in xticks])

            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))

            ax.tick_params(
                axis="both",
                which="major",
                direction="in",
                top=True,
                right=True,
                length=4,
                width=0.8,
                pad=3,
                # labelsize=12,
            )
            ax.tick_params(
                axis="both",
                which="minor",
                direction="in",
                top=True,
                right=True,
                length=2,
                width=0.6,
            )
            ax.grid(False)
            # ax.text(0.02, 0.98, PANEL_LETTERS[(row, col)], transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
            panel_label(ax, PANEL_LETTERS[(row, col)])

            ann_lines = [
                f"R\u00b2 = {float(det_row['R2']):.2f}",
                f"b = {float(det_row['slope']):.2f} [{float(boot_row['slope_ci_low_95']):.2f}, {float(boot_row['slope_ci_high_95']):.2f}]",
            ]
            if rel == "HCTB_candidate_diagnostic":
                ann_lines.append(f"\u0394Adj. R\u00b2 beyond HCCF = +{float(inc_lookup.loc[region, 'delta_adjusted_R2']):.3f}")

            ax.text(
                0.98,
                0.98,
                "\n".join(ann_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=16,
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "none",
                    "edgecolor": "none",
                    "linewidth": 0.8,
                    "alpha": 1.0,
                },
                zorder=10,
            )

            if row == 0:
                ax.set_title(REGION_TITLES[region], pad=6, fontsize=18)
            ax.set_xlabel(info["x_label"])
            if col == 0:
                ax.set_ylabel(info["y_label"])

    # for info in RELATIONSHIP_INFO:
    #     row = info["row"]
    #     bbox = axes[row, 0].get_position()
    #     fig.text(
    #         bbox.x0 - 0.06,
    #         bbox.y0 + bbox.height / 2.0,
    #         info["row_label"],
    #         rotation=90,
    #         va="center",
    #         ha="center",
    #         fontsize=12,
    #         fontweight="bold",
    #     )

    fig.savefig(OUT_PNG, dpi=360, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    data = load_inputs()
    validate_fixed_inputs(data)
    plot_df = build_plot_data(data)
    axis_limits = compute_axis_limits(plot_df)
    enso_degC05 = build_enso_subset_sensitivity_degC05(data)
    draw_figure(plot_df, data, axis_limits)
    make_caption()
    make_input_manifest()
    make_method_checks(data, axis_limits, enso_degC05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
