#!/usr/bin/env python3
"""Generate Figure 02: Regional direct CRE response and robustness using monthly Nino3.4 anomaly +/-0.5 C."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_FIG_DIR = Path("figures_txt/main")
DEFAULT_TABLE_DIR = Path("tables_txt")
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BLOCK_LENGTH = 12
DEFAULT_RANDOM_SEED = 42
NINO_COLUMN = "nino34_anom"
MAIN_THRESHOLD = 0.5
THRESHOLDS = (0.5, 0.75, 1.0)
SMALL = 1.0e-12

DIRECT_VAR_CANDIDATES = {
    "direct_sw_cre": ["sw_allsky_cre", "direct_sw_cre", "sw_cre_direct"],
    "direct_lw_cre": ["lw_allsky_cre", "direct_lw_cre", "lw_cre_direct"],
    "direct_net_cre": ["net_allsky_cre", "direct_net_cre", "net_cre_direct"],
}


@dataclass(frozen=True)
class Region:
    code: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    lon_right_closed: bool


REGIONS = [
    Region("TP", 120.0, 280.0, -15.0, 15.0, True),
    Region("WP", 120.0, 160.0, -15.0, 15.0, False),
    Region("CP", 160.0, 210.0, -15.0, 15.0, False),
    Region("EP", 210.0, 280.0, -15.0, 15.0, True),
]
REGION_ORDER = [region.code for region in REGIONS]
COMPONENT_ORDER = ["sw", "lw", "net"]
COMPONENT_LABELS = {"sw": "SW", "lw": "LW", "net": "Net"}
COMPONENT_COLORS = {"sw": "#35618f", "lw": "#c96f42", "net": "#3f8f6b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_overwrite(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs without --overwrite: {joined}")


def resolve_required_path(project_root: Path, rel: str) -> Path:
    path = (project_root / rel).resolve()
    if path.exists():
        return path
    matches = sorted(project_root.rglob(Path(rel).name))
    if not matches:
        raise FileNotFoundError(f"Required file not found: {rel}")
    return matches[0].resolve()


def select_direct_var(ds: xr.Dataset, aliases: list[str], label: str) -> str:
    for alias in aliases:
        if alias in ds.data_vars:
            return alias
    raise KeyError(f"Could not find {label} in dataset. Tried: {aliases}")


def load_direct_anomaly_dataset(ds_path: Path) -> xr.Dataset:
    ds = xr.open_dataset(ds_path)
    if "time" not in ds.dims or "lat" not in ds.dims or "lon" not in ds.dims:
        raise RuntimeError(f"Dataset must contain time/lat/lon dimensions: {ds_path}")
    if float(ds["lon"].min().item()) < 0.0:
        raise RuntimeError("Longitude coordinate is not 0-360.")
    var_map = {label: select_direct_var(ds, aliases, label) for label, aliases in DIRECT_VAR_CANDIDATES.items()}
    subset = xr.Dataset(
        {
            "direct_sw_cre": ds[var_map["direct_sw_cre"]].astype(np.float64),
            "direct_lw_cre": ds[var_map["direct_lw_cre"]].astype(np.float64),
            "direct_net_cre": ds[var_map["direct_net_cre"]].astype(np.float64),
        }
    )
    if "month" in subset.coords and "time" in subset["month"].dims:
        subset = subset.drop_vars("month")
    return subset.load()


def build_degC05_samples(nino_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nino = pd.read_csv(nino_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    nino["phase"] = pd.NA
    nino.loc[nino["nino34_anom"] >= MAIN_THRESHOLD, "phase"] = "El Nino"
    nino.loc[nino["nino34_anom"] <= -MAIN_THRESHOLD, "phase"] = "La Nina"
    samples = nino.loc[nino["phase"].isin(["El Nino", "La Nina"])].copy()
    samples["month"] = samples["date"].dt.to_period("M").dt.to_timestamp()

    sample_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for phase_name in ["El Nino", "La Nina"]:
        phase_df = samples.loc[samples["phase"] == phase_name].copy()
        if phase_df.empty:
            continue
        phase_df["period_ord"] = phase_df["date"].dt.year * 12 + phase_df["date"].dt.month
        phase_df["gap"] = phase_df["period_ord"].diff().ne(1)
        phase_df.loc[phase_df.index[0], "gap"] = True
        phase_df["event_id"] = phase_df["gap"].cumsum().astype(int)
        for _, row in phase_df.iterrows():
            sample_rows.append(
                {
                    "phase": phase_name,
                    "event_id": int(row["event_id"]),
                    "date": row["date"],
                    "year": int(row["year"]),
                    "month_number": int(row["date"].month),
                    "month": row["date"].to_period("M").to_timestamp(),
                    "season": row["season"],
                    "nino34_sst": float(row["nino34_sst"]),
                    "nino34_anom": float(row["nino34_anom"]),
                    "nino34_anom_std_1981_2010": float(row["nino34_anom_std_1981_2010"]),
                    "nino34_anom_std_1991_2020": float(row["nino34_anom_std_1991_2020"]),
                    "oni_3mon": float(row["oni_3mon"]) if pd.notnull(row["oni_3mon"]) else np.nan,
                }
            )
        for event_id, sub in phase_df.groupby("event_id"):
            event_rows.append(
                {
                    "phase": phase_name,
                    "event_id": int(event_id),
                    "start": sub["date"].min(),
                    "end": sub["date"].max(),
                    "n_months": int(len(sub)),
                }
            )
    sample_df = pd.DataFrame(sample_rows).sort_values(["phase", "event_id", "date"]).reset_index(drop=True)
    event_df = pd.DataFrame(event_rows).sort_values(["phase", "event_id"]).reset_index(drop=True)
    counts = sample_df["phase"].value_counts().to_dict()
    event_counts = event_df["phase"].value_counts().to_dict()
    if counts.get("El Nino", 0) != 54 or counts.get("La Nina", 0) != 85:
        raise RuntimeError(f"Unexpected +/-0.5 C ENSO month counts: {counts}")
    if event_counts.get("El Nino", 0) != 8 or event_counts.get("La Nina", 0) != 12:
        raise RuntimeError(f"Unexpected +/-0.5 C ENSO event counts: {event_counts}")
    return sample_df, event_df


def load_predictor(nino_path: Path) -> pd.DataFrame:
    df = pd.read_csv(nino_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def region_mask(ds: xr.Dataset, region: Region) -> xr.DataArray:
    lat_mask = (ds["lat"] >= region.lat_min) & (ds["lat"] <= region.lat_max)
    if region.lon_right_closed:
        lon_mask = (ds["lon"] >= region.lon_min) & (ds["lon"] <= region.lon_max)
    else:
        lon_mask = (ds["lon"] >= region.lon_min) & (ds["lon"] < region.lon_max)
    return lat_mask & lon_mask


def area_weighted_mean(da: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    subset = da.where(mask, drop=True).astype(np.float64)
    weights = xr.DataArray(
        np.cos(np.deg2rad(subset["lat"].astype(np.float64))),
        dims=("lat",),
        coords={"lat": subset["lat"]},
    )
    return subset.weighted(weights).mean(dim=("lat", "lon"))


def compensation_efficiency(sw: float, lw: float, net: float) -> float:
    denom = abs(sw) + abs(lw)
    if denom < SMALL or sw * lw >= 0.0:
        return np.nan
    return 1.0 - abs(net) / denom


def compute_region_monthly(anom_ds: xr.Dataset) -> pd.DataFrame:
    region_frames: list[pd.DataFrame] = []
    for region in REGIONS:
        mask = region_mask(anom_ds, region)
        frame = pd.DataFrame(
            {
                "month": pd.to_datetime(anom_ds["time"].values),
                "region": region.code,
                "direct_sw_cre": area_weighted_mean(anom_ds["direct_sw_cre"], mask).to_numpy(),
                "direct_lw_cre": area_weighted_mean(anom_ds["direct_lw_cre"], mask).to_numpy(),
                "direct_net_cre": area_weighted_mean(anom_ds["direct_net_cre"], mask).to_numpy(),
            }
        )
        region_frames.append(frame)
    return pd.concat(region_frames, ignore_index=True)


def align_inputs(region_monthly: pd.DataFrame, predictor: pd.DataFrame) -> pd.DataFrame:
    merged = region_monthly.merge(predictor[["month", NINO_COLUMN]], on="month", how="inner", validate="many_to_one")
    counts = merged.groupby("region")["month"].nunique()
    if not counts.eq(len(predictor)).all():
        raise RuntimeError("Regional monthly anomalies and predictor are not aligned over the full study period.")
    return merged.sort_values(["region", "month"]).reset_index(drop=True)


def composite_delta(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    el = frame.loc[frame[NINO_COLUMN] >= threshold]
    la = frame.loc[frame[NINO_COLUMN] <= -threshold]
    if el.empty or la.empty:
        return {
            "delta_sw": np.nan,
            "delta_lw": np.nan,
            "delta_net": np.nan,
            "n_el_nino_months": int(len(el)),
            "n_la_nina_months": int(len(la)),
        }
    sw = float(el["direct_sw_cre"].mean() - la["direct_sw_cre"].mean())
    lw = float(el["direct_lw_cre"].mean() - la["direct_lw_cre"].mean())
    net = float(el["direct_net_cre"].mean() - la["direct_net_cre"].mean())
    return {
        "delta_sw": sw,
        "delta_lw": lw,
        "delta_net": net,
        "n_el_nino_months": int(len(el)),
        "n_la_nina_months": int(len(la)),
    }


def regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan
    xv = x[valid].astype(np.float64)
    yv = y[valid].astype(np.float64)
    x_centered = xv - xv.mean()
    denom = float(np.dot(x_centered, x_centered))
    if denom < SMALL:
        return np.nan
    return float(np.dot(x_centered, yv - yv.mean()) / denom)


def build_block_indices(n_time: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    max_start = n_time - block_length
    if max_start < 0:
        raise ValueError("block_length exceeds sample length")
    n_blocks = math.ceil(n_time / block_length)
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    indices = [np.arange(start, start + block_length, dtype=int) for start in starts]
    return np.concatenate(indices)[:n_time]


def quantile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan
    lower, upper = np.nanpercentile(finite, [2.5, 97.5])
    return float(lower), float(upper)


def compute_bootstrap(region_series: dict[str, pd.DataFrame], bootstrap_samples: int, block_length: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_time = len(next(iter(region_series.values())))
    component_rows: list[dict[str, object]] = []
    regression_rows: list[dict[str, object]] = []

    component_store: dict[str, dict[str, list[float]]] = {
        region: {"delta_sw": [], "delta_lw": [], "delta_net": [], "ce": []} for region in REGION_ORDER
    }
    regression_store: dict[str, list[float]] = {region: [] for region in REGION_ORDER}

    for _ in range(bootstrap_samples):
        sample_index = build_block_indices(n_time, block_length, rng)
        for region in REGION_ORDER:
            boot = region_series[region].iloc[sample_index].reset_index(drop=True)
            delta = composite_delta(boot, MAIN_THRESHOLD)
            ce = compensation_efficiency(delta["delta_sw"], delta["delta_lw"], delta["delta_net"])
            component_store[region]["delta_sw"].append(delta["delta_sw"])
            component_store[region]["delta_lw"].append(delta["delta_lw"])
            component_store[region]["delta_net"].append(delta["delta_net"])
            component_store[region]["ce"].append(ce)
            regression_store[region].append(regression_slope(boot[NINO_COLUMN].to_numpy(), boot["direct_net_cre"].to_numpy()))

    for region in REGION_ORDER:
        row: dict[str, object] = {"region": region, "bootstrap_samples": bootstrap_samples, "block_length_months": block_length}
        for metric in ["delta_sw", "delta_lw", "delta_net", "ce"]:
            values = np.asarray(component_store[region][metric], dtype=np.float64)
            lower, upper = quantile_ci(values)
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
            row[f"{metric}_bootstrap_mean"] = float(np.nanmean(values))
        component_rows.append(row)

        slope_values = np.asarray(regression_store[region], dtype=np.float64)
        slope_lower, slope_upper = quantile_ci(slope_values)
        regression_rows.append(
            {
                "region": region,
                "slope": regression_slope(region_series[region][NINO_COLUMN].to_numpy(), region_series[region]["direct_net_cre"].to_numpy()),
                "ci_lower": slope_lower,
                "ci_upper": slope_upper,
                "bootstrap_mean": float(np.nanmean(slope_values)),
                "bootstrap_std": float(np.nanstd(slope_values, ddof=1)),
                "bootstrap_samples": bootstrap_samples,
                "block_length_months": block_length,
                "n_months": int(len(region_series[region])),
            }
        )

    return pd.DataFrame(component_rows), pd.DataFrame(regression_rows)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "figure.titlesize": 11,
            "savefig.dpi": 220,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.0, 1.03, f"({label})", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")


def plot_figure(
    summary_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    regression_df: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    ax_a, ax_b, ax_c = axes

    summary = summary_df.set_index("region").loc[REGION_ORDER]
    regression = regression_df.set_index("region").loc[REGION_ORDER]

    x = np.arange(len(REGION_ORDER))
    width = 0.22
    for offset, component in zip([-width, 0.0, width], COMPONENT_ORDER):
        metric = f"delta_{component}"
        values = summary[metric].to_numpy()
        lower = summary[f"{metric}_ci_lower"].to_numpy()
        upper = summary[f"{metric}_ci_upper"].to_numpy()
        yerr = np.vstack([values - lower, upper - values])
        ax_a.bar(x + offset, values, width=width, color=COMPONENT_COLORS[component], label=COMPONENT_LABELS[component], edgecolor="white", linewidth=0.5)
        ax_a.errorbar(x + offset, values, yerr=yerr, fmt="none", ecolor="#202020", elinewidth=0.8, capsize=2)
    ax_a.axhline(0.0, color="#333333", linewidth=0.8)
    ax_a.set_xticks(x, REGION_ORDER)
    ax_a.set_ylabel(r"El Niño - La Niña CRE anomaly (W m$^{-2}$)")
    # ax_a.set_title("El Nino minus La Nina direct CRE anomaly")
    ax_a.legend(frameon=False, ncol=3, loc="upper left")
    panel_label(ax_a, "a")

    ce_vals = summary["ce"].to_numpy()
    ce_lower = summary["ce_ci_lower"].to_numpy()
    ce_upper = summary["ce_ci_upper"].to_numpy()
    ce_yerr = np.vstack([ce_vals - ce_lower, ce_upper - ce_vals])
    ax_b.bar(x, ce_vals, width=0.45, color="#7b5ea7", edgecolor="white", linewidth=0.5)
    ax_b.errorbar(x, ce_vals, yerr=ce_yerr, fmt="none", ecolor="#202020", elinewidth=0.8, capsize=2)
    ax_b.set_xticks(x, REGION_ORDER)
    ax_b.set_ylim(0.0, 1.05)
    ax_b.set_ylabel("Compensation Efficiency")
    # ax_b.set_title("Compensation efficiency")
    panel_label(ax_b, "b")

    threshold_matrix = (
        threshold_df.assign(
            threshold_label=threshold_df["threshold_sigma"].map(lambda v: f"\u00b1{v:g}°C")
        )
            .pivot(index="region", columns="threshold_label", values="delta_net")
            .reindex(index=REGION_ORDER, columns=["\u00b10.5°C", "\u00b10.75°C", "\u00b11°C"])
    )
    heat_values = threshold_matrix.to_numpy(dtype=np.float64)
    vmax = float(np.nanmax(np.abs(heat_values)))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
    im = ax_c.imshow(heat_values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_c.set_xticks(np.arange(threshold_matrix.shape[1]), threshold_matrix.columns)
    ax_c.set_yticks(np.arange(threshold_matrix.shape[0]), threshold_matrix.index)
    ax_c.set_title(r"Threshold sensitivity of $\Delta$Net")
    for i in range(threshold_matrix.shape[0]):
        for j in range(threshold_matrix.shape[1]):
            value = heat_values[i, j]
            ax_c.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=11, fontweight="bold", color="#111111")
    # cb = fig.colorbar(im, ax=ax_c, shrink=0.88, pad=0.02)
    # cb.set_label(r"W m$^{-2}$")
    panel_label(ax_c, "c")

    slopes = regression["slope"].to_numpy()
    slope_lower = regression["ci_lower"].to_numpy()
    slope_upper = regression["ci_upper"].to_numpy()
    slope_yerr = np.vstack([slopes - slope_lower, slope_upper - slopes])
    # ax_d.bar(x, slopes, width=0.56, color="#5f8fbe", edgecolor="white", linewidth=0.5)
    # ax_d.errorbar(x, slopes, yerr=slope_yerr, fmt="none", ecolor="#202020", elinewidth=0.8, capsize=2)
    # ax_d.axhline(0.0, color="#333333", linewidth=0.8)
    # ax_d.set_xticks(x, REGION_ORDER)
    # ax_d.set_ylabel(r"W m$^{-2}$ sigma$^{-1}$")
    # ax_d.set_title("Continuous regression of Net CRE anomaly")
    # panel_label(ax_d, "d")

    # for ax in axes.flat:
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("#202020")

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            width=0.9,
            labelsize=11,
        )

    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    package_root = Path(__file__).resolve().parents[1]
    fig_dir = package_root / "01_final_figure"
    table_dir = package_root / "04_key_results"
    input_dir = package_root / "03_input_data"
    notes_dir = package_root / "05_notes"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    png_path = fig_dir / "Figure02_direct_regional_CRE_degC05.png"
    pdf_path = fig_dir / "Figure02_direct_regional_CRE_degC05.pdf"
    summary_path = table_dir / "Figure02_regional_direct_CRE_summary_degC05.csv"
    threshold_path = table_dir / "Figure02_threshold_sensitivity_degC05.csv"
    regression_path = table_dir / "Figure02_regression_bootstrap_degC05.csv"
    sample_export_path = input_dir / "enso_month_samples.csv"
    event_export_path = input_dir / "enso_event_summary.csv"
    method_path = notes_dir / "Figure02_method_and_checks_degC05.txt"
    require_overwrite([png_path, pdf_path, summary_path, threshold_path, regression_path, sample_export_path, event_export_path, method_path], args.overwrite)

    ds_path = resolve_required_path(project_root, "data_processed/anomalies/ceres_monthly_anomalies.nc")
    nino_path = resolve_required_path(project_root, "data_processed/anomalies/nino34_200207_202302.csv")

    anom_ds = load_direct_anomaly_dataset(ds_path)
    main_samples, main_events = build_degC05_samples(nino_path)
    main_samples.to_csv(sample_export_path, index=False, float_format="%.6f", date_format="%Y-%m-%d")
    main_events.to_csv(event_export_path, index=False, date_format="%Y-%m-%d")
    predictor = load_predictor(nino_path)
    region_monthly = compute_region_monthly(anom_ds)
    aligned = align_inputs(region_monthly, predictor)
    region_series = {
        region: aligned.loc[aligned["region"] == region, ["month", "region", "direct_sw_cre", "direct_lw_cre", "direct_net_cre", NINO_COLUMN]]
        .sort_values("month")
        .reset_index(drop=True)
        for region in REGION_ORDER
    }

    summary_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for region in REGION_ORDER:
        base = composite_delta(region_series[region], MAIN_THRESHOLD)
        ce = compensation_efficiency(base["delta_sw"], base["delta_lw"], base["delta_net"])
        summary_rows.append(
            {
                "region": region,
                "delta_sw": base["delta_sw"],
                "delta_lw": base["delta_lw"],
                "delta_net": base["delta_net"],
                "ce": ce,
                "main_threshold_sigma": MAIN_THRESHOLD,
                "n_el_nino_months": base["n_el_nino_months"],
                "n_la_nina_months": base["n_la_nina_months"],
                "n_el_nino_events": int((main_events["phase"] == "El Nino").sum()),
                "n_la_nina_events": int((main_events["phase"] == "La Nina").sum()),
            }
        )
        for threshold in THRESHOLDS:
            result = composite_delta(region_series[region], threshold)
            threshold_rows.append(
                {
                    "region": region,
                    "threshold_sigma": threshold,
                    "delta_net": result["delta_net"],
                    "n_el_nino_months": result["n_el_nino_months"],
                    "n_la_nina_months": result["n_la_nina_months"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    component_bootstrap_df, regression_df = compute_bootstrap(
        region_series=region_series,
        bootstrap_samples=args.bootstrap_samples,
        block_length=args.block_length,
        seed=args.seed,
    )

    summary_df = summary_df.merge(component_bootstrap_df, on="region", how="left", validate="one_to_one")
    threshold_wide = (
        threshold_df.pivot(index="region", columns="threshold_sigma", values="delta_net")
        .rename(columns={0.5: "threshold_0.5_delta_net", 0.75: "threshold_0.75_delta_net", 1.0: "threshold_1.0_delta_net"})
        .reset_index()
    )
    summary_df = summary_df.merge(threshold_wide, on="region", how="left", validate="one_to_one")
    summary_df = summary_df.merge(
        regression_df[["region", "slope", "ci_lower", "ci_upper"]],
        on="region",
        how="left",
        validate="one_to_one",
    ).rename(columns={"slope": "regression_slope", "ci_lower": "regression_ci_lower", "ci_upper": "regression_ci_upper"})

    summary_df = summary_df.set_index("region").loc[REGION_ORDER].reset_index()
    threshold_df = threshold_df.set_index(["region", "threshold_sigma"]).sort_index().reset_index()
    regression_df = regression_df.set_index("region").loc[REGION_ORDER].reset_index()

    summary_df.to_csv(summary_path, index=False, float_format="%.10f")
    threshold_df.to_csv(threshold_path, index=False, float_format="%.10f")
    regression_df.to_csv(regression_path, index=False, float_format="%.10f")
    plot_figure(summary_df, threshold_df, regression_df, png_path, pdf_path)
    method_lines = [
        "Figure02 method and checks (+/-0.5C ENSO definition)",
        "",
        f"- anomaly dataset: {ds_path}",
        f"- predictor file: {nino_path}",
        f"- generated ENSO month file: {sample_export_path}",
        f"- generated ENSO event file: {event_export_path}",
        f"- predictor column used: {NINO_COLUMN}",
        f"- main threshold: +/-{MAIN_THRESHOLD:.2f} C",
        f"- sensitivity thresholds: {', '.join(f'+/-{v:g} C' for v in THRESHOLDS)}",
        f"- El Nino months: {int((main_samples['phase'] == 'El Nino').sum())}",
        f"- La Nina months: {int((main_samples['phase'] == 'La Nina').sum())}",
        f"- El Nino events: {int((main_events['phase'] == 'El Nino').sum())}",
        f"- La Nina events: {int((main_events['phase'] == 'La Nina').sum())}",
        f"- bootstrap samples: {args.bootstrap_samples}",
        f"- block length (months): {args.block_length}",
        f"- random seed: {args.seed}",
    ]
    method_path.write_text("\n".join(method_lines) + "\n", encoding="utf-8")

    print("=== FIGURE 02 REGIONAL DIRECT CRE ROBUSTNESS ===")
    print(f"Input anomaly dataset: {ds_path}")
    print(f"ENSO definition: monthly Nino3.4 anomaly +/-{MAIN_THRESHOLD:.1f} C")
    print(f"Generated ENSO sample file: {sample_export_path}")
    print(f"Nino34 predictor file: {nino_path}")
    print(f"Bootstrap samples: {args.bootstrap_samples}")
    print(f"Moving-block length (months): {args.block_length}")
    print(f"Random seed: {args.seed}")
    for row in summary_df.itertuples():
        print(
            f"{row.region}: SW={row.delta_sw:+.6f}, LW={row.delta_lw:+.6f}, "
            f"Net={row.delta_net:+.6f}, CE={row.ce:.6f}, slope={row.regression_slope:+.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
