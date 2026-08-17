#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "verified" / "figure04_conditional_cre"

FIG04_V1_PNG = OUTPUT_DIR / "Figure04_climatological_conditional_CRE_v1.png"
FIG04_V1_PDF = OUTPUT_DIR / "Figure04_climatological_conditional_CRE_v1.pdf"
CELL_DATA_CSV = OUTPUT_DIR / "Figure04_conditional_CRE_cell_data.csv"
GROUP_SUMMARY_CSV = OUTPUT_DIR / "Figure04_physical_group_conditional_CRE_summary.csv"
DIFF_FROM_TP_CSV = OUTPUT_DIR / "Figure04_regional_Net_CRE0_difference_from_TP.csv"
CAPTION_V1_MD = OUTPUT_DIR / "Figure04_caption_draft_v1.md"
METHOD_V1_TXT = OUTPUT_DIR / "Figure04_method_and_plot_checks_v1.txt"

SCRIPT_PATH = ROOT / "scripts" / "figures_main" / "make_fig04_climatological_conditional_CRE.py"
FIG05_METHOD_PATH = ROOT / "outputs" / "verified" / "figure05" / "Figure05_method_and_checks.txt"
FIG05_SCRIPT_PATH = ROOT / "scripts" / "figures_main" / "make_fig05_cloudtype_occurrence_net_contribution.py"
FIG06_METHOD_PATH = ROOT / "outputs" / "verified" / "figure06_main" / "Figure06_method_and_plot_checks.txt"
MONTHLY_REGION_PATH = ROOT / "data_processed" / "ceres_monthly" / "ceres_monthly_regional.nc"
MONTHLY_BUILD_SCRIPT = ROOT / "scripts" / "02_build_ceres_monthly.py"
RAW_DAY_FILE = Path("/Volumes/My Book/CERES/2002/CERES_FluxByCldTyp-Day_Terra-Aqua-MODIS_Ed4.1_Subset_20020701-20020711.nc")

PROVENANCE_TXT = OUTPUT_DIR / "Figure04_daytime_provenance_check.txt"
FIG_PNG = OUTPUT_DIR / "Figure04_climatological_conditional_CRE_v2.png"
FIG_PDF = OUTPUT_DIR / "Figure04_climatological_conditional_CRE_v2.pdf"
CAPTION_MD = OUTPUT_DIR / "Figure04_caption_draft_v2.md"
METHOD_TXT = OUTPUT_DIR / "Figure04_method_and_plot_checks_v2.txt"

PRESS_ORDER = ["180-10", "310-180", "440-310", "560-440", "680-560", "800-680", "1000-800"]
OPT_ORDER = ["0.02-1.27", "1.27-3.55", "3.55-9.38", "9.38-22.63", "22.63-60.36", "60.36-378.65"]
REGION_ORDER = ["TP", "WP", "CP", "EP"]

GROUP_SPECS = [
    {
        "group": "low cloud",
        "box": (0, 5, 5, 6),
        "label": "Low cloud",
        "label_xy": (0.22, 6.08),
    },
    {
        "group": "mid-level cloud",
        "box": (0, 5, 3, 4),
        "label": "Mid-level",
        "label_xy": (0.22, 4.08),
    },
    {
        "group": "thin high cloud",
        "box": (0, 1, 0, 2),
        "label": "Thin high",
        "label_xy": (0.05, 0.18),
    },
    {
        "group": "thick anvil cloud",
        "box": (2, 3, 0, 2),
        "label": "Thick anvil",
        "label_xy": (2.05, 0.18),
    },
    {
        "group": "deep convective cloud",
        "box": (4, 5, 0, 2),
        "label": "Deep\nconvective",
        "label_xy": (4.14, 0.06),
    },
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.dpi": 300,
        }
    )


def ensure_inputs() -> None:
    required_paths = [
        FIG04_V1_PNG,
        FIG04_V1_PDF,
        CELL_DATA_CSV,
        GROUP_SUMMARY_CSV,
        DIFF_FROM_TP_CSV,
        CAPTION_V1_MD,
        METHOD_V1_TXT,
        FIG05_METHOD_PATH,
        FIG05_SCRIPT_PATH,
        MONTHLY_REGION_PATH,
        MONTHLY_BUILD_SCRIPT,
        RAW_DAY_FILE,
    ]
    for path in required_paths:
        require(path.exists(), f"Missing required input file: {path}")


def read_v1_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell_df = pd.read_csv(CELL_DATA_CSV)
    group_summary_df = pd.read_csv(GROUP_SUMMARY_CSV)
    diff_df = pd.read_csv(DIFF_FROM_TP_CSV)

    require(len(cell_df) == 168, f"Unexpected Figure04 cell-data row count: {len(cell_df)}")
    require(len(group_summary_df) == 20, f"Unexpected Figure04 group-summary row count: {len(group_summary_df)}")
    require(len(diff_df) == 126, f"Unexpected Figure04 TP-difference row count: {len(diff_df)}")
    return cell_df, group_summary_df, diff_df


def check_v1_data_invariance(cell_df: pd.DataFrame) -> None:
    required_cols = {
        "region",
        "cloud_type",
        "ctp_bin",
        "tau_bin",
        "physical_group",
        "CRE0_SW",
        "CRE0_LW",
        "CRE0_Net",
        "valid_n_kernel",
        "display_valid",
        "hatch_flag",
        "net_closure_residual",
    }
    require(required_cols.issubset(cell_df.columns), "Figure04 v1 cell-data CSV is missing required fields.")
    require(set(cell_df["region"].unique().tolist()) == set(REGION_ORDER), "Figure04 v1 cell-data regions do not match TP/WP/CP/EP.")
    require(float(np.nanmax(np.abs(cell_df["net_closure_residual"].to_numpy(dtype=float)))) <= 1.0e-10, "Figure04 v1 cell-data closure residual exceeds tolerance.")
    hatch_rows = cell_df[cell_df["hatch_flag"]].copy()
    require(len(hatch_rows) == 1, f"Expected exactly one hatched baseline cell, found {len(hatch_rows)}.")
    hatch_row = hatch_rows.iloc[0]
    require(
        hatch_row["region"] == "CP"
        and int(hatch_row["cloud_type"]) == 6
        and hatch_row["ctp_bin"] == "1000-800"
        and hatch_row["tau_bin"] == "60.36-378.65"
        and int(hatch_row["valid_n_kernel"]) == 7,
        "Baseline hatch cell does not match the verified CP low-valid cell.",
    )
    wp_row = cell_df[
        (cell_df["region"] == "WP")
        & (cell_df["cloud_type"] == 6)
        & (cell_df["ctp_bin"] == "1000-800")
        & (cell_df["tau_bin"] == "60.36-378.65")
    ]
    require(not wp_row.empty, "Could not locate the WP thickest low-cloud cell in Figure04 v1 cell-data.")
    require(bool(wp_row.iloc[0]["hatch_flag"]) is False, "WP cloud_type=6 cell must remain unhatched in the baseline figure.")


def check_daytime_provenance() -> dict[str, object]:
    ds_monthly = xr.open_dataset(MONTHLY_REGION_PATH)
    ds_raw = xr.open_dataset(RAW_DAY_FILE)
    fig05_method = FIG05_METHOD_PATH.read_text(encoding="utf-8")
    fig05_script = FIG05_SCRIPT_PATH.read_text(encoding="utf-8")
    build_script = MONTHLY_BUILD_SCRIPT.read_text(encoding="utf-8")
    fig06_method = FIG06_METHOD_PATH.read_text(encoding="utf-8") if FIG06_METHOD_PATH.exists() else ""

    raw_title = str(ds_raw.attrs.get("title", ""))
    raw_doi = str(ds_raw.attrs.get("DOI", ""))
    source_product_is_daytime = (
        "FluxByCldTyp Product - Daily Mean" in raw_title
        and "FluxByCldTyp-DAY" in raw_doi
        and "CERES_FluxByCldTyp-Day_Terra-Aqua-MODIS_Ed4.1" in str(RAW_DAY_FILE)
    )
    fig04_from_day = source_product_is_daytime and ("ceres_monthly_regional.nc" in METHOD_V1_TXT.read_text(encoding="utf-8")) and ("CERES_FluxByCldTyp-Day" in fig05_method)
    fig05_day = source_product_is_daytime and ("RAW_ATTR_PATH" in fig05_script) and ("CERES_FluxByCldTyp-Day_Terra-Aqua-MODIS_Ed4.1" in fig05_script)
    fig06_day = source_product_is_daytime and ("daytime" not in fig06_method.lower()) and ("ceres_monthly_regional.nc" in fig05_method)

    verified = bool(source_product_is_daytime and fig04_from_day)
    lines = [
        "Figure 4 daytime provenance check",
        "",
        f"Monthly regional dataset path: {MONTHLY_REGION_PATH}",
        f"Figure 5 script path: {FIG05_SCRIPT_PATH}",
        f"Monthly-build script path: {MONTHLY_BUILD_SCRIPT}",
        f"Raw CERES reference file path: {RAW_DAY_FILE}",
        "",
        "Raw product metadata evidence:",
        f"- raw title: {raw_title}",
        f"- raw DOI: {raw_doi}",
        f"- raw file basename: {RAW_DAY_FILE.name}",
        "",
        "Code-chain evidence:",
        "- scripts/02_build_ceres_monthly.py writes data_processed/ceres_monthly/ceres_monthly_regional.nc from the CERES input inventory opened by open_ceres_dataset.",
        "- scripts/figures_main/make_fig05_cloudtype_occurrence_net_contribution.py reads data_processed/ceres_monthly/ceres_monthly_regional.nc and cites /Volumes/My Book/CERES/.../CERES_FluxByCldTyp-Day_Terra-Aqua-MODIS_Ed4.1_Subset_20020701-20020711.nc as the raw CERES attribute reference.",
        "- Figure 4 v1 restored CRE0 from the same Figure 5 monthly regional dataset and sample rule.",
        "- Figure 6 verified-rebuild scripts use the same data_processed/ceres_monthly/ceres_monthly_regional.nc pathway for Net decomposition inputs.",
        "",
        f"source_product_is_daytime = {source_product_is_daytime}",
        f"Figure04_cre0_derived_from_daytime_product = {fig04_from_day}",
        f"Figure05_occurrence_contribution_is_daytime_based = {fig05_day}",
        f"Figure06_net_pathway_is_daytime_based = {fig06_day}",
    ]
    if verified:
        lines.extend(
            [
                "",
                "daytime terminology verified from the source-product chain",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "daytime terminology not yet verified",
            ]
        )
    PROVENANCE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "raw_title": raw_title,
        "raw_doi": raw_doi,
        "source_product_is_daytime": source_product_is_daytime,
        "Figure04_cre0_derived_from_daytime_product": fig04_from_day,
        "Figure05_occurrence_contribution_is_daytime_based": fig05_day,
        "Figure06_net_pathway_is_daytime_based": fig06_day,
        "daytime_verified_for_figure4": verified,
    }


def make_matrix(cell_df: pd.DataFrame, region: str, value_col: str) -> pd.DataFrame:
    sub = cell_df[cell_df["region"] == region].copy()
    sub["ctp_bin"] = pd.Categorical(sub["ctp_bin"], categories=PRESS_ORDER, ordered=True)
    sub["tau_bin"] = pd.Categorical(sub["tau_bin"], categories=OPT_ORDER, ordered=True)
    sub = sub.sort_values(["ctp_bin", "tau_bin"])
    return sub.pivot(index="ctp_bin", columns="tau_bin", values=value_col).reindex(index=PRESS_ORDER, columns=OPT_ORDER)


def add_group_boxes(ax: plt.Axes, label_groups: bool) -> None:
    for spec in GROUP_SPECS:
        x0, x1, y0, y1 = spec["box"]
        rect = mpatches.Rectangle(
            (x0 - 0.5, y0 - 0.5),
            x1 - x0 + 1,
            y1 - y0 + 1,
            fill=False,
            lw=0.85,
            ls="--" if spec["group"] != "deep convective cloud" else "-",
            ec="black",
        )
        ax.add_patch(rect)
    if label_groups:
        for spec in GROUP_SPECS:
            x, y = spec["label_xy"]
            ax.text(x, y, spec["label"], fontsize=7, ha="left", va="bottom", color="black")


def overlay_hatches(ax: plt.Axes, cell_df: pd.DataFrame, region: str) -> None:
    sub = cell_df[cell_df["region"] == region][["ctp_bin", "tau_bin", "hatch_flag"]].drop_duplicates()
    for y_idx, ctp_bin in enumerate(PRESS_ORDER):
        for x_idx, tau_bin in enumerate(OPT_ORDER):
            row = sub[(sub["ctp_bin"] == ctp_bin) & (sub["tau_bin"] == tau_bin)]
            if not row.empty and bool(row.iloc[0]["hatch_flag"]):
                ax.add_patch(
                    mpatches.Rectangle(
                        (x_idx - 0.5, y_idx - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor="#6E6E6E",
                        hatch="///",
                        linewidth=0.0,
                    )
                )


def make_titles(daytime_verified: bool) -> list[str]:
    if daytime_verified:
        return [
            "Tropical Pacific daytime conditional SW CRE",
            "Tropical Pacific daytime conditional LW CRE",
            "Tropical Pacific daytime conditional Net CRE",
            "Western Pacific daytime conditional Net CRE",
            "Central Pacific daytime conditional Net CRE",
            "Eastern Pacific daytime conditional Net CRE",
        ]
    return [
        "Tropical Pacific conditional SW CRE",
        "Tropical Pacific conditional LW CRE",
        "Tropical Pacific conditional Net CRE",
        "Western Pacific conditional Net CRE",
        "Central Pacific conditional Net CRE",
        "Eastern Pacific conditional Net CRE",
    ]


def make_footer(daytime_verified: bool) -> str:
    if daytime_verified:
        return (
            "Occurrence-mediated daytime Net CRE contribution in Figure 5 is calculated as the cloud-fraction anomaly "
            "multiplied by the corresponding regional climatological daytime conditional Net CRE."
        )
    return (
        "Occurrence-mediated Net CRE contribution in Figure 5 is calculated as the cloud-fraction anomaly multiplied "
        "by the corresponding regional climatological conditional Net CRE."
    )


def make_caption(daytime_verified: bool) -> str:
    if daytime_verified:
        return (
            "Figure 4. Climatological daytime cloud-type conditional cloud radiative effects used for occurrence weighting. "
            "Panels (a)-(c) show the tropical-Pacific climatological daytime conditional shortwave (SW), longwave (LW), "
            "and net cloud radiative effects (CREs), respectively, in the 42-class cloud-top-pressure-optical-depth space. "
            "Panels (d)-(f) show the regional climatological daytime conditional Net CREs over the western, central, and "
            "eastern Pacific used to calculate the occurrence-mediated daytime Net CRE contributions in Figure 5. Boxes "
            "identify the five physically grouped cloud regimes: low cloud, mid-level cloud, thin high cloud, thick anvil "
            "cloud, and deep convective cloud. The hatched cell in the central Pacific denotes insufficient valid samples "
            "for display. These conditional CREs are derived from the CERES FluxByCldTyp-Day product and therefore represent "
            "daytime-sampled radiative conditions rather than daily-mean CREs. The occurrence-mediated daytime Net CRE "
            "contribution is calculated as the cloud-fraction anomaly multiplied by the corresponding regional climatological "
            "daytime conditional Net CRE.\n"
        )
    return (
        "Figure 4. Climatological cloud-type conditional cloud radiative effects used for occurrence weighting. Panels "
        "(a)-(c) show the tropical-Pacific climatological conditional shortwave (SW), longwave (LW), and net cloud "
        "radiative effects (CREs), respectively, in the 42-class cloud-top-pressure-optical-depth space. Panels (d)-(f) "
        "show the regional climatological conditional Net CREs over the western, central, and eastern Pacific used to "
        "calculate the occurrence-mediated Net CRE contributions in Figure 5. Boxes identify the five physically grouped "
        "cloud regimes: low cloud, mid-level cloud, thin high cloud, thick anvil cloud, and deep convective cloud. The "
        "hatched cell in the central Pacific denotes insufficient valid samples for display. The occurrence-mediated Net CRE "
        "contribution is calculated as the cloud-fraction anomaly multiplied by the corresponding regional climatological "
        "conditional Net CRE.\n"
    )


def write_caption(daytime_verified: bool) -> None:
    CAPTION_MD.write_text(make_caption(daytime_verified), encoding="utf-8")


def make_figure(cell_df: pd.DataFrame, daytime_verified: bool) -> None:
    sw_matrix = make_matrix(cell_df, "TP", "CRE0_SW")
    lw_matrix = make_matrix(cell_df, "TP", "CRE0_LW")
    net_matrices = {region: make_matrix(cell_df, region, "CRE0_Net") for region in REGION_ORDER}

    sw_min = float(np.nanmin(sw_matrix.to_numpy()))
    lw_max = float(np.nanmax(lw_matrix.to_numpy()))
    net_max_abs = max(float(np.nanmax(np.abs(matrix.to_numpy()))) for matrix in net_matrices.values())

    sw_norm = Normalize(vmin=sw_min, vmax=0.0)
    lw_norm = Normalize(vmin=0.0, vmax=lw_max)
    net_norm = TwoSlopeNorm(vmin=-net_max_abs, vcenter=0.0, vmax=net_max_abs)
    titles = make_titles(daytime_verified)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
    panel_specs = [
        ("a", "TP", "CRE0_SW", titles[0], "Blues_r", sw_norm),
        ("b", "TP", "CRE0_LW", titles[1], "Reds", lw_norm),
        ("c", "TP", "CRE0_Net", titles[2], "RdBu_r", net_norm),
        ("d", "WP", "CRE0_Net", titles[3], "RdBu_r", net_norm),
        ("e", "CP", "CRE0_Net", titles[4], "RdBu_r", net_norm),
        ("f", "EP", "CRE0_Net", titles[5], "RdBu_r", net_norm),
    ]
    images: dict[str, object] = {}
    for ax, (label, region, value_col, title, cmap, norm) in zip(axes.flatten(), panel_specs):
        matrix = make_matrix(cell_df, region, value_col)
        image = ax.imshow(matrix.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
        images[label] = image
        ax.set_title(f"({label}) {title}")
        ax.set_xticks(np.arange(len(OPT_ORDER)))
        ax.set_xticklabels(OPT_ORDER, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(PRESS_ORDER)))
        ax.set_yticklabels(PRESS_ORDER)
        if ax in axes[1]:
            ax.set_xlabel("Optical depth")
        if ax in axes[:, 0]:
            ax.set_ylabel("Cloud-top pressure (hPa)")
        add_group_boxes(ax, label_groups=(label == "a"))
        if region == "CP":
            overlay_hatches(ax, cell_df, "CP")

    fig.subplots_adjust(left=0.06, right=0.93, top=0.95, bottom=0.16, wspace=0.32, hspace=0.34)
    cax_sw = fig.add_axes([0.285, 0.66, 0.010, 0.26])
    cax_lw = fig.add_axes([0.603, 0.66, 0.010, 0.26])
    cax_net = fig.add_axes([0.942, 0.20, 0.012, 0.62])

    cbar_sw = fig.colorbar(images["a"], cax=cax_sw)
    cbar_lw = fig.colorbar(images["b"], cax=cax_lw)
    cbar_net = fig.colorbar(images["c"], cax=cax_net)
    if daytime_verified:
        cbar_sw.set_label(r"Climatological daytime conditional SW CRE (W m$^{-2}$)")
        cbar_lw.set_label(r"Climatological daytime conditional LW CRE (W m$^{-2}$)")
        cbar_net.set_label(r"Climatological daytime conditional Net CRE (W m$^{-2}$)")
    else:
        cbar_sw.set_label(r"Climatological conditional SW CRE (W m$^{-2}$)")
        cbar_lw.set_label(r"Climatological conditional LW CRE (W m$^{-2}$)")
        cbar_net.set_label(r"Climatological conditional Net CRE (W m$^{-2}$)")

    fig.text(0.5, 0.045, make_footer(daytime_verified), ha="center", va="center", fontsize=8.5)
    fig.savefig(FIG_PNG, bbox_inches="tight")
    fig.savefig(FIG_PDF, bbox_inches="tight")
    plt.close(fig)


def write_method_v2(
    cell_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    provenance: dict[str, object],
) -> None:
    v1_text = METHOD_V1_TXT.read_text(encoding="utf-8")
    daytime_added = bool(provenance["daytime_verified_for_figure4"])
    summary = (
        diff_df.groupby("region", sort=False)["difference_from_TP"]
        .agg(mean_abs_difference=lambda s: float(np.nanmean(np.abs(s))), max_abs_difference=lambda s: float(np.nanmax(np.abs(s))))
        .reset_index()
    )
    sign_change_counts = (
        diff_df.assign(sign_changed=(diff_df["sign_consistent"] == False))
        .groupby("region", sort=False)["sign_changed"]
        .sum()
        .reset_index(name="sign_changed_cell_count")
    )
    merged = summary.merge(sign_change_counts, on="region", how="left")

    lines = [
        "Figure 4 method and plot checks v2",
        "",
        "v2 input reuse policy:",
        f"- v2 plot data are read directly from {CELL_DATA_CSV}",
        f"- v2 physical-group summary is read directly from {GROUP_SUMMARY_CSV}",
        f"- v2 TP-difference table is read directly from {DIFF_FROM_TP_CSV}",
        "- v2 does not recompute CRE0_SW, CRE0_LW, CRE0_Net, valid_n_kernel, hatch mask, or physical-group mapping.",
        "",
        "Daytime provenance file:",
        f"- {PROVENANCE_TXT}",
        f"- source_product_is_daytime = {provenance['source_product_is_daytime']}",
        f"- Figure04_cre0_derived_from_daytime_product = {provenance['Figure04_cre0_derived_from_daytime_product']}",
        f"- Figure05_occurrence_contribution_is_daytime_based = {provenance['Figure05_occurrence_contribution_is_daytime_based']}",
        f"- Figure06_net_pathway_is_daytime_based = {provenance['Figure06_net_pathway_is_daytime_based']}",
        f"- whether daytime terminology was added to the figure and caption = {daytime_added}",
        "",
        "Current terminology boundary:",
        "- conditional CRE is described as daytime-sampled only if verified from the source-product chain.",
    ]
    if daytime_added:
        lines.append("- these Net CRE values should not be interpreted as daily-mean CREs.")
    else:
        lines.append("- daytime terminology not yet verified; the figure and caption keep the non-daytime conditional CRE wording.")

    lines.extend(
        [
            "",
            "v1 invariance checks carried into v2:",
            f"- maximum |net_closure_residual| in reused v1 cell-data = {float(np.nanmax(np.abs(cell_df['net_closure_residual'].to_numpy(dtype=float)))):.6e} W m-2",
            f"- baseline hatched cell count = {int(cell_df['hatch_flag'].sum())}",
            "- v2 keeps the same 2x3 panel structure, the same heatmap values, the same shared Net color scale, the same hatch mask, and the same five-group box geometry as v1.",
            "- The only intended visual changes are simplified group labels in panel (a) and terminology updates conditioned on provenance verification.",
            "",
            "Regional TP-difference summary reused from v1 data:",
        ]
    )
    for row in merged.itertuples(index=False):
        lines.append(
            f"- {row.region}: mean_abs_difference={row.mean_abs_difference:.6f} W m-2; max_abs_difference={row.max_abs_difference:.6f} W m-2; sign_changed_cell_count={int(row.sign_changed_cell_count)}"
        )
    lines.extend(
        [
            "",
            "Reference to v1 method file:",
            f"- {METHOD_V1_TXT}",
            "",
            "Key retained boundary from v1:",
        ]
    )
    for line in v1_text.splitlines():
        if line.startswith("- No all-42 joint strict mask is used") or line.startswith("- This figure reuses the Figure 5 per-cloud-type sample rule"):
            lines.append(line)
    METHOD_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    ensure_inputs()

    cell_df, _group_summary_df, diff_df = read_v1_outputs()
    check_v1_data_invariance(cell_df)
    provenance = check_daytime_provenance()
    write_caption(bool(provenance["daytime_verified_for_figure4"]))
    make_figure(cell_df, bool(provenance["daytime_verified_for_figure4"]))
    write_method_v2(cell_df, diff_df, provenance)

    print(f"Saved {PROVENANCE_TXT}")
    print(f"Saved {FIG_PNG}")
    print(f"Saved {FIG_PDF}")
    print(f"Saved {CAPTION_MD}")
    print(f"Saved {METHOD_TXT}")


if __name__ == "__main__":
    main()
