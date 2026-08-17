#!/usr/bin/env python3
"""Render Figure 05 degC05 using the Figure05 copy.py layout style."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import StrMethodFormatter


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PACKAGE_ROOT / "01_final_figure"
RESULT_DIR = PACKAGE_ROOT / "04_key_results"
NOTES_DIR = PACKAGE_ROOT / "05_notes"

FINAL_INPUT = RESULT_DIR / "Figure05_degC05_final_plot_input.csv"
PREP_SUMMARY = NOTES_DIR / "Figure05_degC05_preparation_summary.txt"

OUT_PNG = FIG_DIR / "Figure05_occurrence_mediated_Net_degC05.png"
OUT_PDF = FIG_DIR / "Figure05_occurrence_mediated_Net_degC05.pdf"
OUT_PLOT = RESULT_DIR / "Figure05_degC05_plot_data.csv"
OUT_CAPTION = NOTES_DIR / "Figure05_degC05_caption.md"
OUT_METHOD = NOTES_DIR / "Figure05_degC05_method_and_plot_checks.txt"

REGION_ORDER = ["TP", "WP", "CP", "EP"]
PRESS_ORDER = ["180-10", "310-180", "440-310", "560-440", "680-560", "800-680", "1000-800"]
OPT_ORDER = ["0.02-1.27", "1.27-3.55", "3.55-9.38", "9.38-22.63", "22.63-60.36", "60.36-378.65"]
REGION_TITLES = {
    "TP": "Tropical Pacific",
    "WP": "Western Pacific",
    "CP": "Central Pacific",
    "EP": "Eastern Pacific",
}
PANEL_LABELS = {"TP": "a", "WP": "b", "CP": "c", "EP": "d"}
X_EDGES = [0.0, 1.27, 3.55, 9.38, 22.63, 60.36, 378.65]
Y_EDGES = [0, 180, 310, 440, 560, 680, 800, 1000]
GROUP_SPECS = [
    {"group": "thin high cloud", "box": (0, 1, 0, 2)},
    {"group": "thick anvil cloud", "box": (2, 3, 0, 2)},
    {"group": "deep convective cloud", "box": (4, 5, 0, 2)},
    {"group": "mid-level cloud", "box": (0, 5, 3, 4)},
    {"group": "low cloud", "box": (0, 5, 5, 6)},
]
CAPTION_TEXT = (
    "Figure 5. Occurrence-mediated daytime Net cloud radiative effect contributions associated with ENSO-driven "
    "cloud-type reorganization using a +/-0.5 C Nino3.4 definition. Panels (a)-(d) present El Nino minus La Nina "
    "occurrence contributions for the tropical Pacific (TP), western Pacific (WP), central Pacific (CP), and eastern "
    "Pacific (EP), respectively. For each cloud type, the occurrence contribution is calculated as the paired-valid "
    "cloud-fraction anomaly multiplied by the corresponding contribution-consistent climatological daytime Net CRE "
    "kernel, AmountNet = DeltaCF_paired x CRE0_Net, with CRE0_Net = mean(Q_Net) / mean(CF_paired). The outlined "
    "blocks identify the five physical cloud groups used in subsequent pathway diagnostics. Black dots indicate "
    "cloud-type cells whose 95% moving-block-bootstrap confidence intervals exclude zero. Hatching identifies cells "
    "excluded by the baseline valid-sample criterion of valid_n < 24.\n"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.0, 1.01, f"({label})", transform=ax.transAxes, ha="left", va="bottom", fontsize=15, fontweight="bold")


def ensure_inputs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    required = [FINAL_INPUT, PREP_SUMMARY]
    missing = [str(path) for path in required if not path.exists()]
    require(not missing, "Missing required input(s):\n" + "\n".join(missing))


def matrix_from_region(df: pd.DataFrame, region: str, value_col: str) -> pd.DataFrame:
    subset = df.loc[df["region"] == region].copy()
    return (
        subset.assign(
            ctp_bin=pd.Categorical(subset["ctp_bin"], categories=PRESS_ORDER, ordered=True),
            tau_bin=pd.Categorical(subset["tau_bin"], categories=OPT_ORDER, ordered=True),
        )
        .pivot(index="ctp_bin", columns="tau_bin", values=value_col)
        .reindex(index=PRESS_ORDER, columns=OPT_ORDER)
    )


def add_group_boxes(ax: plt.Axes) -> None:
    for spec in GROUP_SPECS:
        x0, x1, y0, y1 = spec["box"]
        ax.add_patch(
            mpatches.Rectangle(
                (x0 - 0.5, y0 - 0.5),
                x1 - x0 + 1,
                y1 - y0 + 1,
                fill=False,
                lw=1.4,
                ls="-",
                ec="black",
                zorder=4,
            )
        )


def add_group_labels_panel_a(ax: plt.Axes) -> None:
    ax.text(5.6, 4.6, "Low cloud", fontsize=12, color="black", ha="left", va="bottom", fontweight="bold", zorder=6)
    ax.text(5.6, 2.8, "Mid-level\n  cloud", fontsize=12, color="black", ha="left", va="bottom", fontweight="bold", zorder=6)
    ax.text(0.55, -0.35, "Thin high", fontsize=11, color="black", ha="center", va="top", fontweight="bold", zorder=6)
    ax.text(2.50, -0.35, "Thick anvil", fontsize=11, color="black", ha="center", va="top", fontweight="bold", zorder=6)
    ax.text(4.50, -0.45, "Deep\nconvective", fontsize=11, color="black", ha="center", va="top", fontweight="bold", zorder=6)


def write_caption() -> None:
    OUT_CAPTION.write_text(CAPTION_TEXT, encoding="utf-8")


def write_method(vmax: float, final_df: pd.DataFrame) -> None:
    sig_counts = final_df.groupby("region")["significant"].sum().astype(int).to_dict()
    hatch_counts = final_df.groupby("region")["plot_hatch"].sum().astype(int).to_dict()
    lines = [
        "Figure05 degC05 method and plot checks",
        "",
        f"- final plot-input path: {FINAL_INPUT}",
        f"- preparation summary: {PREP_SUMMARY}",
        "- plotting style source = Figure05 candidate occurrence Net final v2 copy.py",
        "- ENSO definition = nino34_anom with El Nino >= +0.5 C and La Nina <= -0.5 C",
        "- figure term = occurrence-mediated daytime Net CRE contribution only",
        "- no all-42 joint strict mask = True",
        "- baseline CP hatch retained = True",
        (
            "- significant cell counts = "
            f"TP:{sig_counts['TP']}, WP:{sig_counts['WP']}, CP:{sig_counts['CP']}, EP:{sig_counts['EP']}"
        ),
        (
            "- hatch counts = "
            f"TP:{hatch_counts['TP']}, WP:{hatch_counts['WP']}, CP:{hatch_counts['CP']}, EP:{hatch_counts['EP']}"
        ),
        f"- common symmetric color scale centered at zero = True (vmax={vmax:.6f})",
        f"- png: {OUT_PNG}",
        f"- pdf: {OUT_PDF}",
        f"- plot data: {OUT_PLOT}",
        f"- caption: {OUT_CAPTION}",
    ]
    OUT_METHOD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_inputs()
    final_df = pd.read_csv(FINAL_INPUT)
    for col in ["significant", "display_valid_n_ge_24", "sensitivity_valid_n_ge_48", "plot_dot", "plot_hatch"]:
        final_df[col] = final_df[col].astype(bool)
    final_df = final_df.sort_values(["region", "cloud_type"]).reset_index(drop=True)
    final_df.to_csv(OUT_PLOT, index=False)

    vmax = float(np.nanmax(np.abs(final_df["AmountNet_candidate"].to_numpy(dtype=np.float64))))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "figure.titlesize": 16,
            "savefig.dpi": 240,
        }
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="white")
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(18.8, 9.8), gridspec_kw={"hspace": 0.3, "wspace": 0.35})
    image = None
    for idx, region in enumerate(REGION_ORDER):
        ax = axes.flat[idx]
        values = matrix_from_region(final_df, region, "AmountNet_candidate").to_numpy(dtype=np.float64)
        dot_mask = matrix_from_region(final_df, region, "plot_dot").to_numpy(dtype=bool)
        hatch_mask = matrix_from_region(final_df, region, "plot_hatch").to_numpy(dtype=bool)
        values[hatch_mask] = np.nan
        image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_title(REGION_TITLES[region], pad=6)
        panel_label(ax, PANEL_LABELS[region])
        ax.set_xlim(-0.5, len(OPT_ORDER) - 0.5)
        ax.set_ylim(len(PRESS_ORDER) - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, len(OPT_ORDER) + 0.5, 1.0))
        ax.set_xticklabels([f"{v:g}" for v in X_EDGES], rotation=0)
        ax.set_yticks(np.arange(-0.5, len(PRESS_ORDER) + 0.5, 1.0))
        ax.set_yticklabels([f"{v:g}" for v in Y_EDGES])
        ax.grid(which="major", color="white", linewidth=0.7)
        ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=4, width=0.8, pad=3)
        ax.set_xlabel("Optical Depth")
        if idx % 2 == 0:
            ax.set_ylabel("Cloud Top Pressure (hPa)")
        else:
            ax.set_ylabel("")

        yy_h, xx_h = np.where(hatch_mask)
        for y_idx, x_idx in zip(yy_h, xx_h):
            ax.add_patch(
                mpatches.Rectangle(
                    (x_idx - 0.5, y_idx - 0.5),
                    1,
                    1,
                    facecolor="#e6e6e6",
                    edgecolor="#7f7f7f",
                    hatch="///",
                    linewidth=0.0,
                    alpha=1.0,
                    zorder=3,
                )
            )
        yy, xx = np.where(dot_mask & (~hatch_mask))
        if xx.size > 0:
            ax.scatter(xx, yy, s=18, c="black", marker="o", linewidths=0.0, zorder=5)
        add_group_boxes(ax)
        if region == "TP":
            add_group_labels_panel_a(ax)

    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.10, top=0.96, wspace=0.38, hspace=0.42)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.90, pad=0.045)
    cbar.set_label("Net CRE contribution (W m$^{-2}$)", fontsize=15)
    cbar.ax.tick_params(labelsize=12, direction="in", length=4, width=0.8)
    cbar.ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    write_caption()
    write_method(vmax, final_df)


if __name__ == "__main__":
    main()
