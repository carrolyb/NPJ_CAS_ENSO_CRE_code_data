#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import StrMethodFormatter

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PACKAGE_ROOT / "01_final_figure"
INPUT_DIR = PACKAGE_ROOT / "03_input_data"
RESULT_DIR = PACKAGE_ROOT / "04_key_results"
NOTES_DIR = PACKAGE_ROOT / "05_notes"

COMPLETE_INPUT = INPUT_DIR / "Step08_0D3A_Figure04_complete_candidate_plot_input.csv"
TP_INPUT = INPUT_DIR / "Step08_0D3A_Figure04_TP_candidate_conditional_CRE.csv"
TP_SUMMARY = NOTES_DIR / "Step08_0D3A_TP_candidate_conditional_CRE_summary.txt"
REGIONAL_INPUT = INPUT_DIR / "Step08_0D2A2a_Figure04_candidate_regional_conditional_CRE.csv"
VALIDN_INVENTORY = INPUT_DIR / "Step08_0D2A2a_candidate_validn_inventory.csv"
REGIONAL_SUMMARY = NOTES_DIR / "Step08_0D2A2a_Figure04_05_candidate_impact_summary.txt"
DIRECT_ALIGNMENT_SUMMARY = NOTES_DIR / "Step08_0D2B2b2_direct_bootstrap_and_pathway_alignment_summary.txt"
METHOD_BOUNDARY = NOTES_DIR / "Figure04_method_and_plot_checks_v2.txt"
DAYTIME_PROVENANCE = NOTES_DIR / "Figure04_daytime_provenance_check.txt"
LEGACY_LAYOUT_SCRIPT = NOTES_DIR / "make_fig04_climatological_conditional_CRE.py"

OUT_PNG = FIG_DIR / "Figure04_conditional_CRE_kernels_degC05.png"
OUT_PDF = FIG_DIR / "Figure04_conditional_CRE_kernels_degC05.pdf"
OUT_DATA = RESULT_DIR / "Figure04_degC05_plot_data.csv"
OUT_CAPTION = NOTES_DIR / "Figure04_degC05_caption.md"
OUT_METHOD = NOTES_DIR / "Figure04_degC05_method_and_plot_checks.txt"

X_EDGES = [0.0, 1.27, 3.55, 9.38, 22.63, 60.36, 378.65]
Y_EDGES = [0, 180, 310, 440, 560, 680, 800, 1000]

PRESS_ORDER = ["180-10", "310-180", "440-310", "560-440", "680-560", "800-680", "1000-800"]
OPT_ORDER = ["0.02-1.27", "1.27-3.55", "3.55-9.38", "9.38-22.63", "22.63-60.36", "60.36-378.65"]
REGION_ORDER = ["TP", "WP", "CP", "EP"]
PHYSICAL_GROUP_ORDER = [
    "low cloud",
    "mid-level cloud",
    "thin high cloud",
    "thick anvil cloud",
    "deep convective cloud",
]
GROUP_COUNTS = {
    "low cloud": 12,
    "mid-level cloud": 12,
    "thin high cloud": 6,
    "thick anvil cloud": 6,
    "deep convective cloud": 6,
}
GROUP_SPECS = [
    {"group": "thin high cloud", "label": "Thin high cloud", "box": (0, 1, 0, 2), "ls": "-"},
    {"group": "thick anvil cloud", "label": "Thick anvil cloud", "box": (2, 3, 0, 2), "ls": "-"},
    {"group": "deep convective cloud", "label": "Deep convective cloud", "box": (4, 5, 0, 2), "ls": "-"},
    {"group": "mid-level cloud", "label": "Mid-level cloud", "box": (0, 5, 3, 4), "ls": "-"},
    {"group": "low cloud", "label": "Low cloud", "box": (0, 5, 5, 6), "ls": "-"},
]
PLOT_COLUMNS = [
    "region",
    "cloud_type",
    "ctp_bin",
    "tau_bin",
    "physical_group",
    "CF0",
    "Q0_SW",
    "Q0_LW",
    "Q0_Net",
    "CRE0_SW_ratio",
    "CRE0_LW_ratio",
    "CRE0_Net_ratio",
    "CRE0_SW_monthlymean",
    "CRE0_LW_monthlymean",
    "CRE0_Net_monthlymean",
    "difference_ratio_minus_monthlymean_Net",
    "valid_n_kernel",
    "display_valid_n_ge_24",
    "sensitivity_valid_n_ge_48",
]
TOL = 1.0e-10

def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.01,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.titlesize": 16,
            "savefig.dpi": 300,
        }
    )


def ensure_inputs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    for path in [
        COMPLETE_INPUT,
        TP_INPUT,
        TP_SUMMARY,
        REGIONAL_INPUT,
        VALIDN_INVENTORY,
        REGIONAL_SUMMARY,
        DIRECT_ALIGNMENT_SUMMARY,
        METHOD_BOUNDARY,
        DAYTIME_PROVENANCE,
        LEGACY_LAYOUT_SCRIPT,
    ]:
        require(path.exists(), f"Missing required input: {path}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    complete = pd.read_csv(COMPLETE_INPUT)
    tp = pd.read_csv(TP_INPUT)
    regional = pd.read_csv(REGIONAL_INPUT)
    validn = pd.read_csv(VALIDN_INVENTORY)
    return complete, tp, regional, validn


def validate_boundaries() -> dict[str, float]:
    tp_summary = TP_SUMMARY.read_text(encoding="utf-8")
    regional_summary = REGIONAL_SUMMARY.read_text(encoding="utf-8")
    direct_summary = DIRECT_ALIGNMENT_SUMMARY.read_text(encoding="utf-8")
    method_text = METHOD_BOUNDARY.read_text(encoding="utf-8")
    provenance_text = DAYTIME_PROVENANCE.read_text(encoding="utf-8")

    require("source_product_is_daytime = True" in provenance_text, "Stop: daytime source not confirmed.")
    require("daytime terminology verified from the source-product chain" in provenance_text, "Stop: daytime provenance wording missing.")
    require("No all-42 joint strict mask is used in this figure." in method_text, "Stop: no-all-42 boundary missing.")
    require("candidate conditional CRE definition = mean(Q)/mean(CF)" in tp_summary, "Stop: TP ratio-of-means definition missing.")
    require("CRE0 is not a simple regional average of gridcell CRE" in tp_summary, "Stop: TP simple-mean prohibition missing.")
    require("CRE0 is not the arithmetic monthly mean of effective CRE" in tp_summary, "Stop: TP monthly-mean prohibition missing.")
    require("per-cloud-type paired-valid support = True" in tp_summary, "Stop: paired-valid boundary missing.")
    require("cf == 0 retained as zero contribution = True" in tp_summary, "Stop: cf==0 boundary missing.")
    require("Net = SW + LW = True" in tp_summary, "Stop: Net=SW+LW boundary missing.")
    require("no all-42 joint strict mask = True" in tp_summary, "Stop: strict-mask prohibition missing in TP summary.")
    require("candidate monthly contribution definition = cosine-area-mean(gridcell_cf_times_gridcell_CRE)" in regional_summary, "Stop: candidate contribution definition missing.")
    require("This is not a simple area-mean CRE and is not the arithmetic monthly mean of effective CRE." in regional_summary, "Stop: regional ratio-of-means boundary missing.")
    require("strict sensitivity valid_n>=48 introduces no newly excluded cells." in regional_summary, "Stop: valid_n>=48 boundary missing.")
    require("Figures 4-7 candidate formal redraw allowed = True" in direct_summary, "Stop: downstream redraw gate is closed.")
    require("not interpreted as an exact reconstruction of the all-sky response" in direct_summary, "Stop: exact-reconstruction prohibition missing.")

    tp_closure_error = parse_summary_float(tp_summary, "TP SW/LW/Net CRE maximum closure error = ")
    require(tp_closure_error <= TOL, f"Stop: TP closure error exceeds tolerance: {tp_closure_error}")
    max_tp_ratio_diff = parse_summary_float(tp_summary, "maximum |CRE0_Net_ratio - CRE0_Net_monthlymean| = ")
    return {
        "tp_closure_error": tp_closure_error,
        "tp_max_ratio_diff": max_tp_ratio_diff,
    }


def parse_summary_float(text: str, prefix: str) -> float:
    for line in text.splitlines():
        if prefix in line:
            value = line.split(prefix, 1)[1].split()[0]
            return float(value)
    raise RuntimeError(f"Could not parse summary value for prefix: {prefix}")


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["region", "ctp_bin", "tau_bin", "physical_group"]:
        out[col] = out[col].astype(str)
    for col in ["display_valid_n_ge_24", "sensitivity_valid_n_ge_48"]:
        if col in out.columns:
            out[col] = out[col].astype(bool)
    out["cloud_type"] = out["cloud_type"].astype(int)
    out["valid_n_kernel"] = out["valid_n_kernel"].astype(int)
    return out


def validate_tables(
    complete: pd.DataFrame,
    tp: pd.DataFrame,
    regional: pd.DataFrame,
    validn: pd.DataFrame,
) -> pd.DataFrame:
    complete = normalize_table(complete)
    tp = normalize_table(tp)
    regional = normalize_table(regional)
    validn = normalize_table(validn)

    require(list(complete.columns) == PLOT_COLUMNS, "Stop: complete candidate plot-input table columns changed.")
    require(len(complete) == 168, f"Stop: complete candidate plot-input row count != 168 ({len(complete)}).")
    require(set(complete["region"]) == set(REGION_ORDER), "Stop: complete candidate table regions are not TP/WP/CP/EP.")
    require(len(tp) == 42, f"Stop: TP candidate row count != 42 ({len(tp)}).")
    require(len(regional) == 126, f"Stop: WP/CP/EP candidate row count != 126 ({len(regional)}).")
    require(len(validn) == 126, f"Stop: valid_n inventory row count != 126 ({len(validn)}).")

    for region in REGION_ORDER:
        sub = complete[complete["region"] == region]
        require(len(sub) == 42, f"Stop: {region} panel does not contain exactly 42 cells.")
        require(sub["cloud_type"].nunique() == 42, f"Stop: {region} does not contain 42 unique cloud types.")
        require(len(sub[["ctp_bin", "tau_bin"]].drop_duplicates()) == 42, f"Stop: {region} does not contain 42 unique CTP/tau cells.")
        require(set(sub["ctp_bin"]) == set(PRESS_ORDER), f"Stop: {region} CTP bins changed.")
        require(set(sub["tau_bin"]) == set(OPT_ORDER), f"Stop: {region} tau bins changed.")

    joined_tp = complete[complete["region"] == "TP"].merge(
        tp,
        on=["region", "cloud_type", "ctp_bin", "tau_bin", "physical_group"],
        suffixes=("_complete", "_tp"),
        how="outer",
        indicator=True,
    )
    require((joined_tp["_merge"] == "both").all(), "Stop: complete table is not identical in keys to the TP candidate input.")
    compare_numeric(joined_tp, ["CRE0_SW_ratio", "CRE0_LW_ratio", "CRE0_Net_ratio", "CF0", "Q0_SW", "Q0_LW", "Q0_Net", "valid_n_kernel"])
    compare_bool(joined_tp, ["display_valid_n_ge_24", "sensitivity_valid_n_ge_48"])

    joined_regional = complete[complete["region"].isin(["WP", "CP", "EP"])].merge(
        regional,
        on=["region", "cloud_type", "ctp_bin", "tau_bin", "physical_group"],
        suffixes=("_complete", "_regional"),
        how="outer",
        indicator=True,
    )
    require((joined_regional["_merge"] == "both").all(), "Stop: complete table is not identical in keys to the regional candidate input.")
    compare_numeric(joined_regional, ["CRE0_SW_ratio", "CRE0_LW_ratio", "CRE0_Net_ratio", "CF0", "Q0_SW", "Q0_LW", "Q0_Net", "valid_n_kernel"])
    compare_bool(joined_regional, ["display_valid_n_ge_24", "sensitivity_valid_n_ge_48"])

    joined_validn = complete[complete["region"].isin(["WP", "CP", "EP"])].merge(
        validn,
        on=["region", "cloud_type", "ctp_bin", "tau_bin", "physical_group", "valid_n_kernel", "display_valid_n_ge_24", "sensitivity_valid_n_ge_48"],
        how="outer",
        indicator=True,
    )
    require((joined_validn["_merge"] == "both").all(), "Stop: valid_n inventory is inconsistent with the complete candidate table.")
    require((~joined_validn["newly_excluded_by_valid_n48"].astype(bool)).all(), "Stop: valid_n>=48 introduces additional exclusions.")

    validate_mapping_closure(complete)
    validate_mask_rules(complete, validn)
    validate_tp_rules(complete)
    validate_net_rule(complete)

    return complete.sort_values(["region", "cloud_type"]).reset_index(drop=True)


def compare_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        left = f"{col}_complete"
        right = left.replace("_complete", "_tp") if f"{col}_tp" in df.columns else f"{col}_regional"
        diff = np.abs(df[left].astype(float) - df[right].astype(float))
        require(np.nanmax(diff) <= TOL, f"Stop: {col} differs between complete input and component input.")


def compare_bool(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        left = f"{col}_complete"
        right = left.replace("_complete", "_tp") if f"{col}_tp" in df.columns else f"{col}_regional"
        require((df[left].astype(bool) == df[right].astype(bool)).all(), f"Stop: {col} differs between complete input and component input.")


def validate_mapping_closure(complete: pd.DataFrame) -> None:
    mapping = complete[["cloud_type", "physical_group", "ctp_bin", "tau_bin"]].drop_duplicates().sort_values("cloud_type")
    require(len(mapping) == 42, "Stop: cloud-type mapping is not closed to 42 unique rows.")
    require(mapping["cloud_type"].tolist() == list(range(1, 43)), "Stop: cloud_type sequence is not 1..42.")
    group_counts = mapping["physical_group"].value_counts().to_dict()
    for group, expected in GROUP_COUNTS.items():
        require(group_counts.get(group, 0) == expected, f"Stop: physical group count mismatch for {group}.")
    require(mapping["cloud_type"].duplicated().sum() == 0, "Stop: duplicated cloud-type assignments found.")
    require(set(mapping["physical_group"]) == set(PHYSICAL_GROUP_ORDER), "Stop: physical-group labels changed.")


def validate_mask_rules(complete: pd.DataFrame, validn: pd.DataFrame) -> None:
    display_false = complete.loc[~complete["display_valid_n_ge_24"]].copy()
    require(len(display_false) == 1, f"Stop: baseline hatch count must equal 1, found {len(display_false)}.")
    row = display_false.iloc[0]
    require(
        row["region"] == "CP"
        and int(row["cloud_type"]) == 6
        and row["ctp_bin"] == "1000-800"
        and row["tau_bin"] == "60.36-378.65"
        and int(row["valid_n_kernel"]) == 17,
        "Stop: CP baseline hatch cell is wrong or missing.",
    )
    for region in ["TP", "WP", "EP"]:
        require(bool((complete.loc[complete["region"] == region, "display_valid_n_ge_24"]).all()), f"Stop: {region} contains an unexpected baseline exclusion.")
    require(bool((complete.loc[complete["region"] == "WP", "sensitivity_valid_n_ge_48"]).all()), "Stop: WP has an unexpected valid_n>=48 exclusion.")
    require(bool((complete.loc[complete["region"] == "EP", "sensitivity_valid_n_ge_48"]).all()), "Stop: EP has an unexpected valid_n>=48 exclusion.")
    require(bool((complete.loc[complete["region"] == "TP", "sensitivity_valid_n_ge_48"]).all()), "Stop: TP has an unexpected valid_n>=48 exclusion.")
    require(bool((validn["newly_excluded_by_valid_n48"] == False).all()), "Stop: valid_n inventory reports additional valid_n>=48 exclusions.")


def validate_tp_rules(complete: pd.DataFrame) -> None:
    tp = complete[complete["region"] == "TP"].copy()
    require(int(tp["display_valid_n_ge_24"].sum()) == 42, "Stop: TP display_valid_n_ge_24 count is not 42.")
    require((~tp["display_valid_n_ge_24"]).sum() == 0, "Stop: TP has baseline-excluded cells.")
    require((~tp["sensitivity_valid_n_ge_48"]).sum() == 0, "Stop: TP has new valid_n>=48 exclusions.")
    target = tp.loc[tp["cloud_type"] == 6].iloc[0]
    require(abs(float(target["difference_ratio_minus_monthlymean_Net"]) - (-4.948686)) < 1.0e-6, "Stop: TP ratio-minus-monthlymean diagnostic changed at cloud_type=6.")


def validate_net_rule(complete: pd.DataFrame) -> None:
    residual = complete["CRE0_Net_ratio"] - (complete["CRE0_SW_ratio"] + complete["CRE0_LW_ratio"])
    require(float(np.nanmax(np.abs(residual))) <= TOL, "Stop: Net != SW + LW in candidate plotting data.")


def make_display_matrix(df: pd.DataFrame, region: str, value_col: str) -> np.ndarray:
    sub = df.loc[df["region"] == region, ["ctp_bin", "tau_bin", value_col, "display_valid_n_ge_24"]].copy()
    sub["value"] = sub[value_col].astype(float)
    if region in {"WP", "CP", "EP"}:
        sub.loc[~sub["display_valid_n_ge_24"].astype(bool), "value"] = np.nan
    matrix = (
        sub.pivot(index="ctp_bin", columns="tau_bin", values="value")
        .reindex(index=PRESS_ORDER, columns=OPT_ORDER)
        .to_numpy(dtype=float)
    )
    return matrix


def add_group_boxes(ax: plt.Axes) -> None:
    for spec in GROUP_SPECS:
        x0, x1, y0, y1 = spec["box"]
        rect = mpatches.Rectangle(
            (x0 - 0.5, y0 - 0.5),
            x1 - x0 + 1,
            y1 - y0 + 1,
            fill=False,
            edgecolor="#666666",
            # lw=1.5,
            ls=spec["ls"],
            # ec="black",
            zorder=3,
        )
        ax.add_patch(rect)


def add_hatch_overlay(ax: plt.Axes, df: pd.DataFrame, region: str) -> None:
    sub = df[(df["region"] == region) & (~df["display_valid_n_ge_24"])][["ctp_bin", "tau_bin"]]
    for row in sub.itertuples(index=False):
        x_idx = OPT_ORDER.index(row.tau_bin)
        y_idx = PRESS_ORDER.index(row.ctp_bin)
        ax.add_patch(
            mpatches.Rectangle(
                (x_idx - 0.5, y_idx - 0.5),
                1.0,
                1.0,
                facecolor="#E6E6E6",
                edgecolor="#666666",
                hatch="///",
                linewidth=0.4,
            )
        )

def add_group_labels_panel_a(ax: plt.Axes) -> None:
    # 低云、中云：放框外右侧
    ax.text(
        5.58, 4.7, "Low cloud",
        fontsize=12, color="black",
        ha="left", va="bottom", fontweight="bold", zorder=5
    )
    ax.text(
        5.58, 2.7, "Mid-level\n  cloud",
        fontsize=12, color="black",
        ha="left", va="bottom", fontweight="bold", zorder=5
    )

    # 三类高云：放对应网格内部偏上
    ax.text(
        0.5, -0.35, "Thin high",
        fontsize=12, color="black",
        ha="center", va="top", fontweight="bold", zorder=5
    )
    ax.text(
        2.5, -0.35, "Thick anvil",
        fontsize=12, color="black",
        ha="center", va="top", fontweight="bold", zorder=5
    )
    ax.text(
        4.5, -0.40, "Deep\nconvective",
        fontsize=12, color="black",
        ha="center", va="top", fontweight="bold", zorder=5
    )

def make_figure(df: pd.DataFrame) -> None:
    panels = [
        ("a", "TP", "CRE0_SW_ratio", "Tropical Pacific"),
        ("b", "TP", "CRE0_LW_ratio", "Tropical Pacific"),
        ("c", "TP", "CRE0_Net_ratio", "Tropical Pacific"),
        ("d", "WP", "CRE0_Net_ratio", "Western Pacific"),
        ("e", "CP", "CRE0_Net_ratio", "Central Pacific"),
        ("f", "EP", "CRE0_Net_ratio", "Eastern Pacific"),
    ]

    tp_sw_abs = float(np.nanmax(np.abs(df.loc[df["region"] == "TP", "CRE0_SW_ratio"].to_numpy(dtype=float))))
    tp_lw_abs = float(np.nanmax(np.abs(df.loc[df["region"] == "TP", "CRE0_LW_ratio"].to_numpy(dtype=float))))
    tp_net_abs = float(np.nanmax(np.abs(df.loc[df["region"] == "TP", "CRE0_Net_ratio"].to_numpy(dtype=float))))
    regional_visible = df[df["region"].isin(["WP", "CP", "EP"]) & df["display_valid_n_ge_24"]]
    regional_net_abs = float(np.nanmax(np.abs(regional_visible["CRE0_Net_ratio"].to_numpy(dtype=float))))

    norms = {
        "a": TwoSlopeNorm(vmin=-tp_sw_abs, vcenter=0.0, vmax=tp_sw_abs),
        "b": TwoSlopeNorm(vmin=-tp_lw_abs, vcenter=0.0, vmax=tp_lw_abs),
        "c": TwoSlopeNorm(vmin=-tp_net_abs, vcenter=0.0, vmax=tp_net_abs),
        "d": TwoSlopeNorm(vmin=-regional_net_abs, vcenter=0.0, vmax=regional_net_abs),
        "e": TwoSlopeNorm(vmin=-regional_net_abs, vcenter=0.0, vmax=regional_net_abs),
        "f": TwoSlopeNorm(vmin=-regional_net_abs, vcenter=0.0, vmax=regional_net_abs),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.8))
    cmap = "RdBu_r"
    images: dict[str, object] = {}

    for ax, (panel, region, value_col, title) in zip(axes.flatten(), panels):
        matrix = make_display_matrix(df, region, value_col)
        image = ax.imshow(matrix, cmap=cmap, norm=norms[panel], aspect="auto")
        images[panel] = image
        ax.set_title(title, pad=6)
        panel_label(ax, panel)

        # ax.set_xticks(np.arange(len(OPT_ORDER)))
        # ax.set_xticklabels(OPT_ORDER, rotation=32, ha="right")
        # ax.set_yticks(np.arange(len(PRESS_ORDER)))
        # ax.set_yticklabels(PRESS_ORDER)
        # ax.set_xlim(-0.5, len(OPT_ORDER) - 0.5)
        # ax.set_ylim(len(PRESS_ORDER) - 0.5, -0.5)
        # ax.set_xticks(np.arange(-0.5, len(OPT_ORDER), 1.0), minor=True)
        # ax.set_yticks(np.arange(-0.5, len(PRESS_ORDER), 1.0), minor=True)
        # ax.grid(which="minor", color="white", linewidth=0.7)
        # ax.tick_params(which="minor", bottom=False, left=False)
        #
        ax.set_xlim(-0.5, len(OPT_ORDER) - 0.5)
        ax.set_ylim(len(PRESS_ORDER) - 0.5, -0.5)

        ax.set_xticks(np.arange(-0.5, len(OPT_ORDER) + 0.5, 1.0))
        ax.set_xticklabels([f"{v:g}" for v in X_EDGES], rotation=0)

        ax.set_yticks(np.arange(-0.5, len(PRESS_ORDER) + 0.5, 1.0))
        ax.set_yticklabels([f"{v:g}" for v in Y_EDGES])

        ax.grid(which="major", color="white", linewidth=0.7)
        ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=4, width=0.8)

        if ax in axes[1]:
            ax.set_xlabel("Optical Depth")
        if ax in axes[:, 0]:
            ax.set_ylabel("Cloud Top Pressure (hPa)")
        add_group_boxes(ax)
        if panel == "c":
            add_group_labels_panel_a(ax)
        if region == "CP":
            add_hatch_overlay(ax, df, region)

    # fig.suptitle("Contribution-consistent daytime conditional CRE kernels", y=0.975, fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.91, bottom=0.18, wspace=0.3, hspace=0.52)

    # cax_a = fig.add_axes([0.905, 0.66, 0.015, 0.18])
    # cax_b = fig.add_axes([0.935, 0.66, 0.015, 0.18])
    # cax_c = fig.add_axes([0.965, 0.66, 0.015, 0.18])
    # cax_net = fig.add_axes([0.935, 0.19, 0.018, 0.32])
    # legend_ax = fig.add_axes([0.895, 0.02, 0.10, 0.14])
    # legend_ax.axis("off")

    # cbar_a = fig.colorbar(images["a"], cax=cax_a)
    # cbar_b = fig.colorbar(images["b"], cax=cax_b)
    # cbar_c = fig.colorbar(images["c"], cax=cax_c)
    # cbar_net = fig.colorbar(images["d"], cax=cax_net)
    #
    # cbar_a.set_label(r"SW (W m$^{-2}$)")
    # cbar_b.set_label(r"LW (W m$^{-2}$)")
    # cbar_c.set_label(r"TP Net (W m$^{-2}$)")
    # cbar_net.set_label(r"WP/CP/EP Net (W m$^{-2}$)")

    # legend_ax = fig.add_axes([0.90, 0.03, 0.09, 0.13])
    # legend_ax.axis("off")

    # 先取各 panel 的位置
    pos_a = axes[0, 0].get_position()
    pos_b = axes[0, 1].get_position()
    pos_c = axes[0, 2].get_position()
    pos_d = axes[1, 0].get_position()
    pos_f = axes[1, 2].get_position()

    # a, b, c 各自下方横向 colorbar
    cax_a = fig.add_axes([pos_a.x0, pos_a.y0 - 0.055, pos_a.width, 0.018])
    cax_b = fig.add_axes([pos_b.x0, pos_b.y0 - 0.055, pos_b.width, 0.018])
    cax_c = fig.add_axes([pos_c.x0, pos_c.y0 - 0.055, pos_c.width, 0.018])

    # d/e/f 公用一个，放在底排中间位置
    shared_left = pos_d.x0 + 0.06
    shared_right = pos_f.x1 - 0.06
    shared_width = shared_right - shared_left
    shared_y = pos_d.y0 - 0.075
    cax_net = fig.add_axes([shared_left, shared_y, shared_width, 0.020])

    cbar_a = fig.colorbar(images["a"], cax=cax_a, orientation="horizontal")
    cbar_b = fig.colorbar(images["b"], cax=cax_b, orientation="horizontal")
    cbar_c = fig.colorbar(images["c"], cax=cax_c, orientation="horizontal")
    cbar_net = fig.colorbar(images["d"], cax=cax_net, orientation="horizontal")

    cbar_a.set_label(r"SW CRE kernel (W m$^{-2}$)", fontsize=16)
    cbar_b.set_label(r"LW CRE kernel (W m$^{-2}$)", fontsize=16)
    cbar_c.set_label(r"Net CRE kernel (W m$^{-2}$)", fontsize=16)
    cbar_net.set_label(r"Net CRE kernel (W m$^{-2}$)", fontsize=16)

    for cbar in [cbar_a, cbar_b, cbar_c, cbar_net]:
        cbar.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        cbar.ax.tick_params(labelsize=15, direction="in", length=4, width=0.8)

    # handles = [
    #     mlines.Line2D([], [], color="black", linestyle="--", linewidth=0.9, label="Low cloud"),
    #     mlines.Line2D([], [], color="black", linestyle="--", linewidth=0.9, label="Mid-level cloud"),
    #     mlines.Line2D([], [], color="black", linestyle="--", linewidth=0.9, label="Thin high cloud"),
    #     mlines.Line2D([], [], color="black", linestyle="--", linewidth=0.9, label="Thick anvil cloud"),
    #     mlines.Line2D([], [], color="black", linestyle="-", linewidth=0.9, label="Deep convective cloud"),
    #     mpatches.Patch(facecolor="#E6E6E6", edgecolor="#666666", hatch="///", label="valid_n < 24"),
    # ]
    # legend_ax.legend(
    #     handles=handles,
    #     loc="upper left",
    #     frameon=False,
    #     fontsize=8,
    #     title="Groups / mask",
    #     title_fontsize=8,
    #     handlelength=2.8,
    # )

    # fig.text(
    #     0.52,
    #     0.055,
    #     "Hatched cells have fewer than 24 valid months for the conditional CRE kernel.",
    #     ha="center",
    #     va="center",
    #     fontsize=8,
    # )

    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def write_caption() -> None:
    caption = (
        "Figure 4. Contribution-consistent climatological daytime conditional cloud radiative effects for the 42 CERES "
        "cloud types. Panels (a)–(c) present the shortwave (SW), longwave (LW), and net cloud radiative effect (CRE) "
        "kernels over the tropical Pacific, respectively. Panels (d)–(f) present the net CRE kernels over the western, "
        "central, and eastern Pacific. The kernels are defined as the climatological integrated cloud-type contribution "
        "divided by the corresponding climatological paired-valid cloud fraction, CRE0 = mean(Q) / mean(CF), where Q is "
        "computed from gridcell cloud fraction multiplied by gridcell conditional CRE before regional aggregation. The "
        "outlined blocks identify the five physical cloud groups used in subsequent pathway diagnostics. Hatching "
        "indicates cells excluded by the baseline valid-sample requirement of valid_n < 24. This packaged redraw is "
        "labeled degC05 for consistency with the current ENSO-0.5C figure-organizing workflow, although the Figure 4 "
        "climatological CRE0 kernels themselves do not depend on the ENSO threshold definition.\n"
    )
    OUT_CAPTION.write_text(caption, encoding="utf-8")


def write_plot_data(df: pd.DataFrame) -> None:
    out = df.copy()
    out["panels_used"] = out["region"].map({"TP": "a,b,c", "WP": "d", "CP": "e", "EP": "f"})
    out["display_value_tp_sw"] = np.where(out["region"] == "TP", out["CRE0_SW_ratio"], np.nan)
    out["display_value_tp_lw"] = np.where(out["region"] == "TP", out["CRE0_LW_ratio"], np.nan)
    out["display_value_tp_net"] = np.where(out["region"] == "TP", out["CRE0_Net_ratio"], np.nan)
    regional_net = np.where(
        (out["region"].isin(["WP", "CP", "EP"])) & out["display_valid_n_ge_24"],
        out["CRE0_Net_ratio"],
        np.nan,
    )
    out["display_value_regional_net"] = regional_net
    out["baseline_hatch"] = ~out["display_valid_n_ge_24"]
    out.to_csv(OUT_DATA, index=False)


def write_method_checks(df: pd.DataFrame, boundary_diag: dict[str, float]) -> None:
    lines = [
        "Figure04 degC05 method and plot checks",
        "",
        "Actual candidate input files",
        f"- complete Figure 4 candidate plot-input table: {COMPLETE_INPUT}",
        f"- TP candidate conditional CRE table: {TP_INPUT}",
        f"- TP candidate summary: {TP_SUMMARY}",
        f"- WP/CP/EP candidate regional conditional CRE table: {REGIONAL_INPUT}",
        f"- candidate valid_n inventory: {VALIDN_INVENTORY}",
        f"- candidate Figure 4/5 impact summary: {REGIONAL_SUMMARY}",
        f"- direct bootstrap and pathway alignment summary: {DIRECT_ALIGNMENT_SUMMARY}",
        f"- Figure 4 method boundary reference: {METHOD_BOUNDARY}",
        f"- Figure 4 daytime provenance reference: {DAYTIME_PROVENANCE}",
        f"- legacy Figure 4 layout script reused for style only: {LEGACY_LAYOUT_SCRIPT}",
        "",
        "Method boundary",
        "- package label = degC05",
        "- Figure 4 CRE0 kernels are climatological and numerically independent of the ENSO threshold definition used for downstream occurrence figures.",
        "- Figure 4 uses candidate ratio-of-means CRE0 = mean(Q)/mean(CF).",
        "- no legacy regional values used for plotting = True",
        "- daytime source confirmed = True",
        "- per-cloud-type paired-valid rule confirmed = True",
        "- cf==0 zero contribution rule confirmed = True",
        "- Net=SW+LW confirmed = True",
        "- no all-42 joint strict mask = True",
        f"- TP closure max error = {boundary_diag['tp_closure_error']:.12e} W m-2",
        f"- TP ratio-minus-monthlymean maximum absolute difference = {boundary_diag['tp_max_ratio_diff']:.6f} W m-2",
        "- candidate WP/CP/EP source files confirmed = True",
        "- TP/WP/EP have no baseline exclusions = True",
        "- CP has exactly one baseline excluded cell: cloud_type=6, valid_n=17 = True",
        "- valid_n>=48 introduces no additional exclusions = True",
        "- five-group mapping closure: assigned=42, duplicated=0, unassigned=0",
        "- Figure 4 supplies reference kernels for pathway diagnostics and is not an all-sky direct CRE reconstruction.",
        "- no candidate values or masks were recomputed during formal plotting.",
        "",
        "Panel and mask status",
        "- panel (a) uses TP CRE0_SW_ratio from the complete candidate input table.",
        "- panel (b) uses TP CRE0_LW_ratio from the complete candidate input table.",
        "- panel (c) uses TP CRE0_Net_ratio from the complete candidate input table.",
        "- panel (d) uses WP CRE0_Net_ratio from the complete candidate input table.",
        "- panel (e) uses CP CRE0_Net_ratio from the complete candidate input table.",
        "- panel (f) uses EP CRE0_Net_ratio from the complete candidate input table.",
        "- masked cells do not display candidate CRE values; the CP low-valid cell is replaced by a gray hatched overlay.",
        "- baseline hatch appears only in panel (e) at CTP=1000-800 hPa, tau=60.36-378.65, cloud_type=6.",
        "- no WP valid_n>=48 hatch added = True",
        "- no TP or EP extra hatch added = True",
        "",
        "Output files",
        f"- png: {OUT_PNG}",
        f"- pdf: {OUT_PDF}",
        f"- plot data: {OUT_DATA}",
        f"- caption: {OUT_CAPTION}",
        f"- method checks: {OUT_METHOD}",
    ]
    OUT_METHOD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    ensure_inputs()

    boundary_diag = validate_boundaries()
    complete, tp, regional, validn = load_inputs()
    plot_df = validate_tables(complete, tp, regional, validn)
    write_plot_data(plot_df)
    make_figure(plot_df)
    write_caption()
    write_method_checks(plot_df, boundary_diag)

    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_DATA}")
    print(f"Saved {OUT_CAPTION}")
    print(f"Saved {OUT_METHOD}")


if __name__ == "__main__":
    main()
