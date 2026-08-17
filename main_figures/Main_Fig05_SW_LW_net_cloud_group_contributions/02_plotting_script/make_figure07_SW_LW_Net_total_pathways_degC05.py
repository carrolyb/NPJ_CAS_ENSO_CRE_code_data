#!/usr/bin/env python3
"""Render Figure 07 under the +/-0.5 C ENSO definition using the Figure07 copy.py layout."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import StrMethodFormatter


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PACKAGE_ROOT / "01_final_figure"
INPUT_DIR = PACKAGE_ROOT / "03_input_data"
RESULT_DIR = PACKAGE_ROOT / "04_key_results"
NOTES_DIR = PACKAGE_ROOT / "05_notes"

REGIONAL_MONTHLY_NC = INPUT_DIR / "CERES_regional_integrated_candidate_monthly.nc"
REGIONAL_FIG04_CSV = INPUT_DIR / "Step08_0D2A2a_Figure04_candidate_regional_conditional_CRE.csv"
NINO_CSV = INPUT_DIR / "nino34_200207_202302.csv"
FIG06_GROUP_CSV = PACKAGE_ROOT.parent / "Figure06_Net_pathway_decomposition_degC05" / "04_key_results" / "Figure06_degC05_group_summary.csv"
FIG06_METHOD = PACKAGE_ROOT.parent / "Figure06_Net_pathway_decomposition_degC05" / "05_notes" / "Figure06_degC05_method_and_plot_checks.txt"
FIG05_METHOD = PACKAGE_ROOT.parent / "Figure05_occurrence_mediated_Net_degC05" / "05_notes" / "Figure05_degC05_method_and_plot_checks.txt"

OUT_PNG = FIG_DIR / "Figure07_SW_LW_Net_total_pathways_degC05.png"
OUT_PDF = FIG_DIR / "Figure07_SW_LW_Net_total_pathways_degC05.pdf"
OUT_PLOT = RESULT_DIR / "Figure07_degC05_plot_data.csv"
OUT_BOOT = RESULT_DIR / "Figure07_degC05_group_SW_LW_Net_total_bootstrap.csv"
OUT_DET = RESULT_DIR / "Figure07_degC05_group_SW_LW_Net_total_summary.csv"
OUT_CAPTION = NOTES_DIR / "Figure07_degC05_caption.md"
OUT_METHOD = NOTES_DIR / "Figure07_degC05_method_and_plot_checks.txt"
OUT_MANIFEST = NOTES_DIR / "Figure07_degC05_input_data_manifest.md"

REGION_ORDER = ["WP", "CP", "EP"]
GROUP_ORDER = [
    "low cloud",
    "mid-level cloud",
    "thin high cloud",
    "thick anvil cloud",
    "deep convective cloud",
]
COMPONENT_ORDER = ["SW", "LW", "Net"]
GROUP_LABELS = {
    "low cloud": "Low",
    "mid-level cloud": "Mid-level",
    "thin high cloud": "Thin high",
    "thick anvil cloud": "Thick anvil",
    "deep convective cloud": "Deep\n convective",
}
PANEL_TITLES = {
    "SW": "SW contribution",
    "LW": "LW contribution",
    "Net": "Net contribution",
}
PANEL_LABELS = {"SW": "a", "LW": "b", "Net": "c"}
BLOCK_LENGTH = 12
N_BOOT = 2000
SEED = 42
THRESHOLD = 0.5
NINO_COLUMN = "nino34_anom"
TOL = 1.0e-10

CAPTION_TEXT = (
    "Figure 7. Cloud-type-resolved daytime shortwave (SW), longwave (LW), and net total-contribution pathways across the "
    "tropical Pacific subregions using the +/-0.5 C Nino3.4 ENSO definition. Panels (a)-(c) present El Nino minus La Nina "
    "group-total contributions for SW, LW, and net cloud radiative effects (CRE), respectively, across the western Pacific "
    "(WP), central Pacific (CP), and eastern Pacific (EP). Rows identify the five physical cloud groups used in the pathway "
    "analysis. Values are derived from the contribution-consistent daytime candidate chain, and black dots indicate group-component "
    "contributions whose 95% moving-block-bootstrap confidence intervals exclude zero. The pathways are interpreted as "
    "cloud-type-resolved diagnostics and not as an exact reconstruction of the direct all-sky CRE response.\n"
)


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
        FIG06_GROUP_CSV,
        FIG06_METHOD,
        FIG05_METHOD,
    ]
    missing = [str(path) for path in required if not path.exists()]
    require(not missing, "Missing required input(s):\n" + "\n".join(missing))


def load_inputs() -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = xr.open_dataset(REGIONAL_MONTHLY_NC)
    fig04 = pd.read_csv(REGIONAL_FIG04_CSV)
    nino = pd.read_csv(NINO_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    fig06_group = pd.read_csv(FIG06_GROUP_CSV)
    return ds, fig04, nino, fig06_group


def validate_inputs(ds: xr.Dataset, fig04: pd.DataFrame, fig06_group: pd.DataFrame) -> None:
    require(dict(ds.sizes) == {"region": 3, "time": 248, "cloud_type": 42}, f"Unexpected dataset shape: {dict(ds.sizes)}")
    require(ds.attrs.get("daytime_based") == "True", "Monthly dataset is not tagged as daytime.")
    require(ds.attrs.get("paired_valid_rule") == "True", "Monthly dataset is not tagged as paired-valid.")
    require(ds.attrs.get("all42_joint_strict_mask") == "False", "Monthly dataset unexpectedly uses all-42 joint strict mask.")
    require(ds.attrs.get("cf_zero_contribution") == "0", "Monthly dataset unexpectedly changed cf==0 handling.")
    require(
        ds.attrs.get("regional_contribution_definition") == "area_mean(gridcell_cf_times_gridcell_CRE)",
        "Unexpected contribution definition in monthly dataset.",
    )
    net_resid = np.abs(ds["net_q_region"].values - (ds["sw_q_region"].values + ds["lw_q_region"].values))
    require(float(np.nanmax(net_resid)) <= TOL, "Monthly Net != SW + LW.")

    fig04["display_valid_n_ge_24"] = fig04["display_valid_n_ge_24"].astype(bool)
    require(len(fig04) == 126, f"Unexpected Figure 4 row count: {len(fig04)}")
    counts = fig04.groupby("region")["display_valid_n_ge_24"].sum().astype(int).to_dict()
    require(counts == {"WP": 42, "CP": 41, "EP": 42}, f"Unexpected display-valid counts: {counts}")

    fig06_group["region"] = pd.Categorical(fig06_group["region"], categories=REGION_ORDER, ordered=True)
    fig06_group["physical_group"] = pd.Categorical(fig06_group["physical_group"], categories=GROUP_ORDER, ordered=True)
    fig06_group = fig06_group.sort_values(["region", "physical_group"]).reset_index(drop=True)
    require(len(fig06_group) == 15, f"Unexpected Figure06 group row count: {len(fig06_group)}")


def build_phase(time_values: np.ndarray, nino: pd.DataFrame) -> np.ndarray:
    work = nino.copy()
    work["time"] = work["date"].dt.to_period("M").dt.to_timestamp()
    phase_map = work.set_index("time")[NINO_COLUMN]
    times = pd.to_datetime(time_values).to_period("M").to_timestamp()
    predictor = phase_map.reindex(times)
    require(predictor.notna().all(), "Missing Nino3.4 values for some monthly time steps.")
    phase = np.zeros(len(times), dtype=np.int8)
    values = predictor.to_numpy(dtype=float)
    phase[values >= THRESHOLD] = 1
    phase[values <= -THRESHOLD] = -1
    require(int((phase == 1).sum()) == 54, f"Unexpected El Nino month count: {(phase == 1).sum()}")
    require(int((phase == -1).sum()) == 85, f"Unexpected La Nina month count: {(phase == -1).sum()}")
    return phase


def phase_diff(series: np.ndarray, phase_sign: np.ndarray) -> float:
    return float(np.nanmean(series[phase_sign == 1]) - np.nanmean(series[phase_sign == -1]))


def build_monthly_component_table(ds: xr.Dataset, fig04: pd.DataFrame, phase: np.ndarray) -> pd.DataFrame:
    valid_phase = (phase == 1) | (phase == -1)
    times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    region_lookup = {str(region): i for i, region in enumerate(ds["region"].values.tolist())}
    ordered = fig04.copy()
    ordered["region"] = pd.Categorical(ordered["region"], categories=REGION_ORDER, ordered=True)
    ordered = ordered.sort_values(["region", "cloud_type"]).reset_index(drop=True)
    records: list[dict[str, object]] = []

    for row in ordered.itertuples(index=False):
        r_idx = region_lookup[str(row.region)]
        c_idx = int(row.cloud_type) - 1
        for component, var_name in [("SW", "sw_q_region"), ("LW", "lw_q_region"), ("Net", "net_q_region")]:
            values = ds[var_name].values[r_idx, :, c_idx].astype(np.float64, copy=False)
            for i, month in enumerate(times):
                if not valid_phase[i]:
                    continue
                records.append(
                    {
                        "region": str(row.region),
                        "cloud_type": int(row.cloud_type),
                        "physical_group": row.physical_group,
                        "display_valid_n_ge_24": bool(row.display_valid_n_ge_24),
                        "month": month,
                        "phase_sign": int(phase[i]),
                        "component": component,
                        "value_t": values[i],
                    }
                )
    out = pd.DataFrame.from_records(records)
    require(len(out) == 126 * 3 * (54 + 85), f"Unexpected monthly component row count: {len(out)}")
    return out


def summarize_deterministic(monthly_df: pd.DataFrame) -> pd.DataFrame:
    display_df = monthly_df.loc[monthly_df["display_valid_n_ge_24"]].copy()
    rows: list[dict[str, object]] = []
    for (region, group, component), sub in display_df.groupby(["region", "physical_group", "component"], sort=False):
        monthly_sum = sub.groupby(["month", "phase_sign"], sort=False)["value_t"].sum().reset_index()
        rows.append(
            {
                "region": region,
                "physical_group": group,
                "component": component,
                "deterministic_estimate": phase_diff(monthly_sum["value_t"].to_numpy(dtype=float), monthly_sum["phase_sign"].to_numpy(dtype=int)),
                "n_cells_display_valid": int(sub["cloud_type"].nunique()),
            }
        )
    det = pd.DataFrame.from_records(rows)
    det["region"] = pd.Categorical(det["region"], categories=REGION_ORDER, ordered=True)
    det["physical_group"] = pd.Categorical(det["physical_group"], categories=GROUP_ORDER, ordered=True)
    det["component"] = pd.Categorical(det["component"], categories=COMPONENT_ORDER, ordered=True)
    det = det.sort_values(["component", "physical_group", "region"]).reset_index(drop=True)
    return det


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


def summarize_bootstrap(monthly_df: pd.DataFrame, det_df: pd.DataFrame) -> pd.DataFrame:
    phase_template = monthly_df[["month", "phase_sign"]].drop_duplicates().sort_values("month").reset_index(drop=True)
    phase_sign = phase_template["phase_sign"].to_numpy(dtype=int)
    require(len(phase_template) == 54 + 85, f"Unexpected ENSO-month count: {len(phase_template)}")
    boot_indices = build_bootstrap_indices(len(phase_template))

    display_df = monthly_df.loc[monthly_df["display_valid_n_ge_24"]].copy()
    rows: list[dict[str, object]] = []

    for row in det_df.itertuples(index=False):
        sub = display_df.loc[
            (display_df["region"] == str(row.region))
            & (display_df["physical_group"] == str(row.physical_group))
            & (display_df["component"] == str(row.component))
        ].copy()
        series = (
            sub.groupby("month", sort=False)["value_t"]
            .sum()
            .reindex(phase_template["month"])
            .to_numpy(dtype=float)
        )
        dist = bootstrap_phase_diffs(series, phase_sign, boot_indices)
        finite = dist[np.isfinite(dist)]
        require(finite.size == dist.size, f"Bootstrap distribution contains NaN replicate(s) for {row.region} {row.physical_group} {row.component}.")
        ci_low = float(np.nanpercentile(finite, 2.5))
        ci_high = float(np.nanpercentile(finite, 97.5))
        significant = bool((ci_low > 0.0) or (ci_high < 0.0))
        rows.append(
            {
                "region": str(row.region),
                "physical_group": str(row.physical_group),
                "component": str(row.component),
                "deterministic_estimate": float(row.deterministic_estimate),
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "significant": significant,
                "sign_stability_probability": float(np.mean(np.sign(finite) == np.sign(float(row.deterministic_estimate)))),
                "n_cells_display_valid": int(row.n_cells_display_valid),
            }
        )
    out = pd.DataFrame.from_records(rows)
    out["region"] = pd.Categorical(out["region"], categories=REGION_ORDER, ordered=True)
    out["physical_group"] = pd.Categorical(out["physical_group"], categories=GROUP_ORDER, ordered=True)
    out["component"] = pd.Categorical(out["component"], categories=COMPONENT_ORDER, ordered=True)
    out = out.sort_values(["component", "physical_group", "region"]).reset_index(drop=True)
    return out


def validate_outputs(boot_df: pd.DataFrame, fig06_group: pd.DataFrame) -> dict[str, float]:
    component_sum = boot_df.pivot(index=["region", "physical_group"], columns="component", values="deterministic_estimate").reset_index()
    require(np.allclose(component_sum["Net"], component_sum["SW"] + component_sum["LW"], atol=TOL, rtol=0.0), "Figure07 deterministic output violates Net = SW + LW.")

    net_df = boot_df.loc[boot_df["component"] == "Net", ["region", "physical_group", "deterministic_estimate", "significant", "n_cells_display_valid"]].copy()
    fig06_group = fig06_group.copy()
    fig06_group["region"] = pd.Categorical(fig06_group["region"], categories=REGION_ORDER, ordered=True)
    fig06_group["physical_group"] = pd.Categorical(fig06_group["physical_group"], categories=GROUP_ORDER, ordered=True)
    merged = net_df.merge(
        fig06_group[["region", "physical_group", "delta_total", "total_significant", "n_cells_display_valid"]],
        on=["region", "physical_group"],
        how="inner",
        suffixes=("_fig07", "_fig06"),
    )
    require(len(merged) == 15, "Figure07 Net rows did not align 1:1 with Figure06 group totals.")
    diff = np.abs(merged["deterministic_estimate"] - merged["delta_total"])
    require(float(diff.max()) <= TOL, f"Figure07 Net panel does not reproduce Figure06 Net totals within tolerance; max={float(diff.max()):.12e}")
    require(bool((merged["significant"] == merged["total_significant"]).all()), "Figure07 Net significance does not match Figure06 totals.")
    require(bool((merged["n_cells_display_valid_fig07"] == merged["n_cells_display_valid_fig06"]).all()), "Figure07 valid-cell counts do not match Figure06.")

    vmax_swlw_raw = float(boot_df.loc[boot_df["component"].isin(["SW", "LW"]), "deterministic_estimate"].abs().max())
    vmax_net_raw = float(boot_df.loc[boot_df["component"] == "Net", "deterministic_estimate"].abs().max())
    require(vmax_swlw_raw > vmax_net_raw, "Expected SW/LW scale to exceed Net scale.")
    return {
        "max_net_diff": float(diff.max()),
        "vmax_swlw_raw": vmax_swlw_raw,
        "vmax_net_raw": vmax_net_raw,
        "vmax_swlw": nice_ceil_symmetric(vmax_swlw_raw),
        "vmax_net": nice_ceil_symmetric(vmax_net_raw),
    }


def nice_ceil_symmetric(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    exponent = math.floor(math.log10(value))
    scaled = value / (10**exponent)
    for step in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        if scaled <= step:
            return step * (10**exponent)
    return 10.0 ** (exponent + 1)


def text_color(value: float, vmax: float) -> str:
    return "white" if abs(value) >= 0.55 * vmax else "black"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.linewidth": 0.8,
            "figure.titlesize": 15,
            "savefig.dpi": 300,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.0, 1.02, f"({label})", transform=ax.transAxes, ha="left", va="bottom", fontsize=15, fontweight="bold")


def make_matrix(df: pd.DataFrame, component: str, value_col: str) -> np.ndarray:
    sub = df.loc[df["component"] == component].copy()
    return (
        sub.pivot(index="physical_group", columns="region", values=value_col)
        .reindex(index=GROUP_ORDER, columns=REGION_ORDER)
        .to_numpy()
    )


def draw_panel(ax: plt.Axes, df: pd.DataFrame, component: str, norm: TwoSlopeNorm, vmax: float) -> plt.AxesImage:
    matrix = make_matrix(df, component, "deterministic_estimate").astype(float)
    sig = make_matrix(df, component, "significant").astype(bool)
    image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_title(PANEL_TITLES[component], pad=6)
    panel_label(ax, PANEL_LABELS[component])

    ax.set_xticks(np.arange(len(REGION_ORDER)))
    ax.set_xticklabels(REGION_ORDER)
    ax.set_yticks(np.arange(len(GROUP_ORDER)))
    ax.set_yticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.set_xticks(np.arange(-0.5, len(REGION_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(GROUP_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=4, width=0.8, pad=3)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=2, width=0.6)

    for i in range(len(GROUP_ORDER)):
        for j in range(len(REGION_ORDER)):
            value = float(matrix[i, j])
            label = f"{value:.2f}".replace("-0.00", "0.00")
            ax.text(j, i + 0.09, label, ha="center", va="center", fontsize=8.4, fontweight="bold", color=text_color(value, vmax), zorder=5)
            if bool(sig[i, j]):
                ax.scatter(j, i - 0.31, s=10, color="black", zorder=6, clip_on=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    return image


def build_plot_data(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df["region"] = plot_df["region"].astype(str)
    plot_df["physical_group"] = plot_df["physical_group"].astype(str)
    plot_df["component"] = plot_df["component"].astype(str)
    plot_df["panel_title"] = plot_df["component"].map(PANEL_TITLES)
    plot_df["group_label"] = plot_df["physical_group"].map(GROUP_LABELS)
    plot_df["display_value"] = plot_df["deterministic_estimate"].map(lambda x: f"{x:.2f}".replace("-0.00", "0.00"))
    plot_df["black_dot"] = plot_df["significant"]
    plot_df["hatch_drawn"] = False
    return plot_df[
        [
            "region",
            "physical_group",
            "group_label",
            "component",
            "panel_title",
            "deterministic_estimate",
            "ci_low_95",
            "ci_high_95",
            "significant",
            "black_dot",
            "hatch_drawn",
            "display_value",
            "sign_stability_probability",
            "n_cells_display_valid",
        ]
    ]


def plot_figure(df: pd.DataFrame, scales: dict[str, float]) -> None:
    setup_style()
    swlw_norm = TwoSlopeNorm(vmin=-scales["vmax_swlw"], vcenter=0.0, vmax=scales["vmax_swlw"])
    net_norm = TwoSlopeNorm(vmin=-scales["vmax_net"], vcenter=0.0, vmax=scales["vmax_net"])

    fig = plt.figure(figsize=(11.8, 5.0), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=6,
        height_ratios=[18.0, 1.15],
        width_ratios=[1, 1, 1, 1, 1, 1],
        left=0.12,
        right=0.97,
        top=0.93,
        bottom=0.16,
        wspace=0.42,
        hspace=0.32,
    )
    ax_sw = fig.add_subplot(gs[0, 0:2])
    ax_lw = fig.add_subplot(gs[0, 2:4])
    ax_net = fig.add_subplot(gs[0, 4:6])
    cax_swlw = fig.add_subplot(gs[1, 0:4])
    cax_net = fig.add_subplot(gs[1, 4:6])

    im_sw = draw_panel(ax_sw, df, "SW", swlw_norm, scales["vmax_swlw"])
    draw_panel(ax_lw, df, "LW", swlw_norm, scales["vmax_swlw"])
    im_net = draw_panel(ax_net, df, "Net", net_norm, scales["vmax_net"])
    for ax in [ax_lw, ax_net]:
        ax.set_yticklabels([])

    cbar_swlw = fig.colorbar(im_sw, cax=cax_swlw, orientation="horizontal")
    cbar_swlw.set_label("SW/LW total contribution (W m$^{-2}$)", fontsize=13)
    cbar_net = fig.colorbar(im_net, cax=cax_net, orientation="horizontal")
    cbar_net.set_label("Net total contribution (W m$^{-2}$)", fontsize=13)
    for cbar in [cbar_swlw, cbar_net]:
        cbar.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        cbar.ax.tick_params(labelsize=13, direction="in", length=3, width=0.8)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def write_caption() -> None:
    OUT_CAPTION.write_text(CAPTION_TEXT, encoding="utf-8")


def write_manifest() -> None:
    lines = [
        "# Figure07 degC05 input manifest",
        "",
        f"- Monthly regional candidate chain: `{REGIONAL_MONTHLY_NC}`",
        f"- Regional conditional CRE / display-valid table: `{REGIONAL_FIG04_CSV}`",
        f"- ENSO index file: `{NINO_CSV}`",
        f"- Figure06 degC05 group-total reference: `{FIG06_GROUP_CSV}`",
    ]
    OUT_MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method(scales: dict[str, float], boot_df: pd.DataFrame) -> None:
    pathway_keys = [
        ("CP", "low cloud"),
        ("CP", "thick anvil cloud"),
        ("CP", "deep convective cloud"),
        ("WP", "low cloud"),
        ("EP", "low cloud"),
    ]
    lines = [
        "Figure07 degC05 method and plot checks",
        "",
        "- plotting style source = Figure07 candidate SW/LW/Net total pathways final v1 copy.py",
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
        "- formal plot regenerated from the fixed monthly chain under the +/-0.5 C ENSO definition",
        "- no hatch drawn because Figure07 is group level",
        "",
        "Reference cross-checks",
        "- Figure07 Net rows reproduce Figure06 degC05 group totals and significance = True",
        "",
        "Scale checks",
        f"- shared SW/LW color scale raw max abs = {scales['vmax_swlw_raw']:.6f}, plot vmax = {scales['vmax_swlw']:.6f}",
        f"- independent Net color scale raw max abs = {scales['vmax_net_raw']:.6f}, plot vmax = {scales['vmax_net']:.6f}",
        f"- Net reproduction max abs difference versus Figure06 degC05 = {scales['max_net_diff']:.12e}",
        "",
        "Pathway checks",
    ]
    for region, group in pathway_keys:
        sub = boot_df.loc[(boot_df["region"].astype(str) == region) & (boot_df["physical_group"].astype(str) == group), ["component", "deterministic_estimate", "significant"]].copy()
        sub = sub.set_index("component")
        lines.append(
            f"- {region} {group}: SW={float(sub.loc['SW', 'deterministic_estimate']):.6f}, "
            f"LW={float(sub.loc['LW', 'deterministic_estimate']):.6f}, "
            f"Net={float(sub.loc['Net', 'deterministic_estimate']):.6f}, "
            f"Net_significant={bool(sub.loc['Net', 'significant'])}"
        )
    lines.extend(
        [
            "",
            "Output files",
            f"- png: {OUT_PNG}",
            f"- pdf: {OUT_PDF}",
            f"- plot data: {OUT_PLOT}",
            f"- bootstrap summary table: {OUT_BOOT}",
            f"- deterministic summary table: {OUT_DET}",
            f"- caption: {OUT_CAPTION}",
            f"- input manifest: {OUT_MANIFEST}",
        ]
    )
    OUT_METHOD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_inputs()
    ds, fig04, nino, fig06_group = load_inputs()
    validate_inputs(ds, fig04, fig06_group)
    phase = build_phase(ds["time"].values, nino)
    monthly_df = build_monthly_component_table(ds, fig04, phase)
    det_df = summarize_deterministic(monthly_df)
    boot_df = summarize_bootstrap(monthly_df, det_df)
    scales = validate_outputs(boot_df, fig06_group)
    det_df.to_csv(OUT_DET, index=False)
    boot_df.to_csv(OUT_BOOT, index=False)
    build_plot_data(boot_df).to_csv(OUT_PLOT, index=False)
    plot_figure(boot_df, scales)
    write_caption()
    write_manifest()
    write_method(scales, boot_df)


if __name__ == "__main__":
    main()
