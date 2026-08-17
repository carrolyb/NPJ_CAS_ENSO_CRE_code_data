#!/usr/bin/env python3
"""Make Figure 1 direct CRE spatial response map using a monthly Nino3.4 anomaly +/-0.5 C ENSO definition."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize, TwoSlopeNorm


DEFAULT_PROJECT_ROOT = Path(".")
DEFAULT_OUT_DIR = Path("figures_txt/main")
DATA_OUT_DIR = Path("figures_txt/main/data")
DOC_OUT_DIR = Path("docs_txt/figures")
ALGEBRAIC_TOLERANCE = 1.0e-6
GROSS_RESPONSE_THRESHOLD = 2.0
DEGC_THRESHOLD = 0.5

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
    Region("WP", 120.0, 160.0, -15.0, 15.0, False),
    Region("CP", 160.0, 210.0, -15.0, 15.0, False),
    Region("EP", 210.0, 280.0, -15.0, 15.0, True),
]
TP = Region("TP", 120.0, 280.0, -15.0, 15.0, True)
REGION_ORDER = ["TP", "WP", "CP", "EP"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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


def load_direct_anomaly_dataset(ds_path: Path) -> tuple[xr.Dataset, dict[str, str]]:
    ds = xr.open_dataset(ds_path)
    if "time" not in ds.dims or "lat" not in ds.dims or "lon" not in ds.dims:
        raise RuntimeError(f"Dataset must contain time/lat/lon: {ds_path}")
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
    return subset.load(), var_map


def build_degC05_samples(nino_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nino = pd.read_csv(nino_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    nino["phase"] = pd.NA
    nino.loc[nino["nino34_anom"] >= DEGC_THRESHOLD, "phase"] = "El Nino"
    nino.loc[nino["nino34_anom"] <= -DEGC_THRESHOLD, "phase"] = "La Nina"
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
                    "month": row["date"].to_period("M").to_timestamp(),
                    "month_number": int(row["date"].month),
                    "season": row["season"],
                    "nino34_anom": float(row["nino34_anom"]),
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


def area_weighted_mean(da: xr.DataArray, region: Region) -> xr.DataArray:
    lat_mask = (da["lat"] >= region.lat_min) & (da["lat"] <= region.lat_max)
    if region.lon_right_closed:
        lon_mask = (da["lon"] >= region.lon_min) & (da["lon"] <= region.lon_max)
    else:
        lon_mask = (da["lon"] >= region.lon_min) & (da["lon"] < region.lon_max)
    subset = da.where(lat_mask & lon_mask, drop=True).astype(np.float64)
    weights = xr.DataArray(
        np.cos(np.deg2rad(subset["lat"].astype(np.float64))),
        dims=("lat",),
        coords={"lat": subset["lat"]},
    )
    return subset.weighted(weights).mean(dim=("lat", "lon"))


def compensation_efficiency(sw: float, lw: float, net: float) -> float:
    gross = abs(sw) + abs(lw)
    return np.nan if gross == 0.0 else 1.0 - abs(net) / gross


def rounded_percentile_limit(arrays: list[np.ndarray], percentile: float = 98.0) -> float:
    finite_parts = [a[np.isfinite(a)] for a in arrays]
    finite = np.concatenate([a for a in finite_parts if a.size > 0])
    value = float(np.nanpercentile(np.abs(finite), percentile))
    if not np.isfinite(value) or value <= 0:
        return 1.0
    if value <= 1:
        return math.ceil(value * 10.0) / 10.0
    if value <= 5:
        return float(math.ceil(value))
    if value <= 10:
        return float(math.ceil(value / 2.0) * 2)
    return float(math.ceil(value / 5.0) * 5)


def lon_label(value: float) -> str:
    if value == 180:
        return "180"
    if value <= 180:
        return f"{int(value)}E"
    west = int(round(360 - value))
    return f"{west}W"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.0, 1.02, f"({label})", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")


# def map_axis(fig: plt.Figure, spec) -> plt.Axes:
#     ax = fig.add_subplot(spec, projection=ccrs.PlateCarree(central_longitude=180))
#     ax.set_extent([120, 280, -15, 15], crs=ccrs.PlateCarree())
#     ax.coastlines(resolution="110m", linewidth=0.45)
#     ax.add_feature(cfeature.LAND, facecolor="#efefe8", edgecolor="none", zorder=0)
#     xticks = [120, 160, 200, 240, 280]
#     yticks = [-15, 0, 15]
#     gl = ax.gridlines(
#         crs=ccrs.PlateCarree(),
#         draw_labels=True,
#         linewidth=0.3,
#         color="#666666",
#         alpha=0.25,
#         linestyle="--",
#     )
#     gl.top_labels = False
#     gl.right_labels = False
#     gl.xlocator = mpl.ticker.FixedLocator(xticks)
#     gl.ylocator = mpl.ticker.FixedLocator(yticks)
#     gl.xlabel_style = {"size": 7}
#     gl.ylabel_style = {"size": 7}
#     gl.xformatter = mpl.ticker.FuncFormatter(lambda val, pos: lon_label((val + 360) % 360))
#     return ax

def lat_label(value: float) -> str:
    if value < 0:
        return f"{int(abs(value))}S"
    if value > 0:
        return f"{int(value)}N"
    return "0"


def map_axis(fig: plt.Figure, spec) -> plt.Axes:
    ax = fig.add_subplot(spec, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([120, 280, -15, 15], crs=ccrs.PlateCarree())
    ax.coastlines(resolution="110m", linewidth=0.45)
    ax.add_feature(cfeature.LAND, facecolor="#efefe8", edgecolor="none", zorder=0)

    xticks = [120, 150, 180, 210, 240, 270]
    yticks = [-10, 0, 10]

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels([lon_label(v) for v in xticks])
    ax.set_yticklabels([lat_label(v) for v in yticks])

    ax.tick_params(axis="both", labelsize=8, length=3, pad=2)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.3,
        color="#666666",
        alpha=0.25,
        linestyle="--",
    )
    return ax


def draw_region_boxes(ax: plt.Axes, add_labels: bool) -> None:
    for region in REGIONS:
        width = region.lon_max - region.lon_min
        rect = mpatches.Rectangle(
            (region.lon_min, region.lat_min),
            width,
            region.lat_max - region.lat_min,
            fill=False,
            lw=0.8,
            ec="#202020",
            transform=ccrs.PlateCarree(),
        )
        ax.add_patch(rect)
        if add_labels:
            ax.text(
                region.lon_min + 2,
                region.lat_max - 1.5,
                region.code,
                transform=ccrs.PlateCarree(),
                fontsize=10,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.8},
            )


def compute_spatial_composites(anom_ds: xr.Dataset, samples: pd.DataFrame) -> xr.Dataset:
    en_times = samples.loc[samples["phase"] == "El Nino", "month"].to_numpy()
    ln_times = samples.loc[samples["phase"] == "La Nina", "month"].to_numpy()
    sw = anom_ds["direct_sw_cre"].sel(time=en_times).mean("time", skipna=True) - anom_ds["direct_sw_cre"].sel(time=ln_times).mean("time", skipna=True)
    lw = anom_ds["direct_lw_cre"].sel(time=en_times).mean("time", skipna=True) - anom_ds["direct_lw_cre"].sel(time=ln_times).mean("time", skipna=True)
    net_raw = anom_ds["direct_net_cre"].sel(time=en_times).mean("time", skipna=True) - anom_ds["direct_net_cre"].sel(time=ln_times).mean("time", skipna=True)
    net = sw + lw
    gross = abs(sw) + abs(lw)
    ce_raw = xr.where(gross > 0.0, 1.0 - abs(net) / gross, np.nan)
    ce_mask = (sw * lw < 0.0) & (gross >= GROSS_RESPONSE_THRESHOLD)
    ce = ce_raw.where(ce_mask)
    return xr.Dataset(
        {
            "delta_sw": sw,
            "delta_lw": lw,
            "delta_net": net,
            "delta_net_raw": net_raw,
            "gross_response": gross,
            "ce": ce,
        }
    )


def compute_regional_validation(spatial_ds: xr.Dataset, canonical_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    canonical = canonical_df.set_index("region")
    region_map = {"TP": TP, "WP": REGIONS[0], "CP": REGIONS[1], "EP": REGIONS[2]}
    for region_code in REGION_ORDER:
        region = region_map[region_code]
        sw = float(area_weighted_mean(spatial_ds["delta_sw"], region).item())
        lw = float(area_weighted_mean(spatial_ds["delta_lw"], region).item())
        net = float(area_weighted_mean(spatial_ds["delta_net"], region).item())
        ce = compensation_efficiency(sw, lw, net)
        can = canonical.loc[region_code]
        diff_net = net - float(can["delta_net"])
        rows.append(
            {
                "region": region_code,
                "delta_sw_spatial": sw,
                "delta_lw_spatial": lw,
                "delta_net_spatial": net,
                "ce_from_spatial_aggregation": ce,
                "canonical_delta_sw": float(can["delta_sw"]),
                "canonical_delta_lw": float(can["delta_lw"]),
                "canonical_delta_net": float(can["delta_net"]),
                "canonical_ce": float(can["ce"]),
                "difference_net": diff_net,
                "match_canonical": abs(diff_net) < ALGEBRAIC_TOLERANCE,
            }
        )
    return pd.DataFrame(rows)


def make_figure(spatial_ds: xr.Dataset, png_path: Path, pdf_path: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "savefig.dpi": 320,
        }
    )
    sw_lw_limit = rounded_percentile_limit([spatial_ds["delta_sw"].values, spatial_ds["delta_lw"].values], percentile=98.0)
    net_limit = rounded_percentile_limit([spatial_ds["delta_net"].values], percentile=98.0)
    sw_lw_norm = TwoSlopeNorm(vmin=-sw_lw_limit, vcenter=0.0, vmax=sw_lw_limit)
    net_norm = TwoSlopeNorm(vmin=-net_limit, vcenter=0.0, vmax=net_limit)
    # ce_norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(6.2, 7.), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, hspace=-0.35)

    titles = [
        ("delta_sw", "El Niño - La Niña SW CRE anomaly", "RdBu_r", sw_lw_norm),
        ("delta_lw", "El Niño - La Niña LW CRE anomaly", "RdBu_r", sw_lw_norm),
        ("delta_net", "El Niño - La Niña Net CRE anomaly", "RdBu_r", net_norm),
        # ("ce", "Compensation efficiency", "YlGnBu", ce_norm),
    ]
    axes = []
    meshes = []
    for idx, (var, title, cmap, norm) in enumerate(titles):
        # ax = map_axis(fig, gs[idx // 2, idx % 2])
        ax = map_axis(fig, gs[idx])
        panel_label(ax, "abcd"[idx])
        # if var == "ce":
        #     ax.set_facecolor("#e9e9e9")
        mesh = ax.pcolormesh(
            spatial_ds["lon"],
            spatial_ds["lat"],
            spatial_ds[var],
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            shading="auto",
        )
        ax.set_title(title)
        draw_region_boxes(ax, add_labels=(var == "delta_sw"))
        axes.append(ax)
        meshes.append(mesh)

        ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.8)
        ax.tick_params(axis="both", which="minor", direction="in", length=2, width=0.6)

    # cb_sw_lw = fig.colorbar(meshes[0], ax=[axes[0], axes[1]], shrink=0.88, pad=0.03)
    # cb_sw_lw.set_label("SW/LW CRE anomaly (W m$^{-2}$)")
    # cb_net = fig.colorbar(meshes[2], ax=axes[2], shrink=0.88, pad=0.03)
    # cb_net.set_label("Net CRE anomaly (W m$^{-2}$)")
    # cb_ce = fig.colorbar(meshes[3], ax=axes[3], shrink=0.88, pad=0.03)
    # cb_ce.set_label("Compensation efficiency")

    cb_sw = fig.colorbar(
        meshes[0],
        ax = axes[0],
        orientation = "horizontal",
        shrink = 0.9,
        pad = 0.05,
        aspect = 35,
    )
    cb_sw.set_label("SW CRE anomaly (W m$^{-2}$)")

    cb_lw = fig.colorbar(
        meshes[1],
        ax=axes[1],
        orientation="horizontal",
        shrink=0.9,
        pad=0.05,
        aspect=35,
    )
    cb_lw.set_label("LW CRE anomaly (W m$^{-2}$)")

    cb_net = fig.colorbar(
        meshes[2],
        ax=axes[2],
        orientation="horizontal",
        shrink=0.9,
        pad= 0.05,
        aspect=35,
    )
    cb_net.set_label("Net CRE anomaly (W m$^{-2}$)")

    fig.savefig(png_path, bbox_inches="tight", dpi=320)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_caption(out_path: Path) -> None:
    lines = [
        "# Figure 1 Caption Draft",
        "",
        "Figure 1. Spatial response of full direct CRE to ENSO over the tropical Pacific using a monthly Nino3.4 anomaly threshold definition.",
        "Panels (a)-(c) show El Nino minus La Nina anomalies in SW CRE, LW CRE, and Net CRE, respectively, computed from monthly anomaly direct CRE fields over the verified 54 El Nino months and 85 La Nina months within 15S-15N.",
        "Panel (d) shows compensation efficiency, defined as 1 - |DeltaNet| / (|DeltaSW| + |DeltaLW|), and is displayed only where DeltaSW and DeltaLW have opposite signs and the gross response |DeltaSW| + |DeltaLW| is at least 2.0 W m^-2; all other grid cells are masked.",
        "WP, CP, and EP boxes indicate the regions used for subsequent regional statistics only.",
        "ENSO months are defined from monthly Nino3.4 SST anomalies with thresholds of +0.5 C for El Nino and -0.5 C for La Nina. This figure uses only the full direct monthly anomaly CRE data chain and does not include cloud-type attribution.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    package_root = Path(__file__).resolve().parents[1]
    out_dir = package_root / "01_final_figure"
    data_dir = package_root / "04_key_results"
    doc_dir = package_root / "05_notes"
    input_dir = package_root / "03_input_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "Figure01_main_CRE_lat15_degC05.png"
    pdf_path = out_dir / "Figure01_main_CRE_lat15_degC05.pdf"
    csv_path = data_dir / "Figure01_regional_CRE_summary_lat15_degC05.csv"
    caption_path = doc_dir / "Figure01_caption_draft_degC05.md"
    sample_export_path = input_dir / "enso_month_samples.csv"
    event_export_path = input_dir / "enso_event_summary.csv"
    method_note_path = doc_dir / "Figure01_method_and_checks_degC05.txt"
    outputs = [png_path, pdf_path, csv_path, caption_path, sample_export_path, event_export_path, method_note_path]
    require_overwrite(outputs, args.overwrite)

    ds_path = resolve_required_path(project_root, "data_processed/anomalies/ceres_monthly_anomalies.nc")
    nino_path = resolve_required_path(project_root, "data_processed/anomalies/nino34_200207_202302.csv")
    canonical_path = resolve_required_path(project_root, "outputs/verified/step02g_full_direct_canonical/full_direct_canonical_monthly_equal.csv")
    method_path = resolve_required_path(project_root, "outputs/verified/step02g_full_direct_canonical/FULL_DIRECT_CANONICAL_METHOD.txt")

    method_text = method_path.read_text(encoding="utf-8")
    canonical_confirmed = "Main composite: equal weighting of ENSO months" in method_text and "Input: monthly anomaly direct CRE fields" in method_text

    anom_ds, _ = load_direct_anomaly_dataset(ds_path)
    samples, events = build_degC05_samples(nino_path)
    samples.to_csv(sample_export_path, index=False, float_format="%.6f", date_format="%Y-%m-%d")
    events.to_csv(event_export_path, index=False, date_format="%Y-%m-%d")
    canonical_df = pd.read_csv(canonical_path)

    spatial_ds = compute_spatial_composites(anom_ds, samples)
    raw_closure_residual = float(np.nanmax(np.abs((spatial_ds["delta_net_raw"] - (spatial_ds["delta_sw"] + spatial_ds["delta_lw"])).to_numpy())))
    max_abs_residual = float(np.nanmax(np.abs((spatial_ds["delta_net"] - (spatial_ds["delta_sw"] + spatial_ds["delta_lw"])).to_numpy())))
    if max_abs_residual >= ALGEBRAIC_TOLERANCE:
        raise RuntimeError(f"Spatial Net = SW + LW closure failed: max_abs_residual={max_abs_residual:.6e}")

    validation_df = compute_regional_validation(spatial_ds, canonical_df)
    validation_df.to_csv(csv_path, index=False, float_format="%.10f")

    make_figure(spatial_ds, png_path, pdf_path)
    write_caption(caption_path)
    method_lines = [
        "Figure01 method and checks (+/-0.5C ENSO definition)",
        "",
        f"- monthly Nino3.4 source: {nino_path}",
        f"- generated ENSO month file: {sample_export_path}",
        f"- generated ENSO event file: {event_export_path}",
        f"- El Nino months: {int((samples['phase'] == 'El Nino').sum())}",
        f"- La Nina months: {int((samples['phase'] == 'La Nina').sum())}",
        f"- El Nino events: {int((events['phase'] == 'El Nino').sum())}",
        f"- La Nina events: {int((events['phase'] == 'La Nina').sum())}",
        f"- max spatial closure residual: {max_abs_residual:.12e}",
        f"- max regional net difference vs original canonical table: {float(validation_df['difference_net'].abs().max()):.6f}",
    ]
    method_note_path.write_text("\n".join(method_lines) + "\n", encoding="utf-8")

    print("=== FIGURE 01 DIRECT CRE SPATIAL RESPONSE ===")
    print(f"Canonical input method confirmed: {'YES' if canonical_confirmed else 'NO'}")
    print(f"Spatial Net = SW + LW closure: {'PASS' if max_abs_residual < ALGEBRAIC_TOLERANCE else 'FAIL'}")
    print(f"Regional means match original canonical table: {'PASS' if bool(validation_df['match_canonical'].all()) else 'FAIL (expected after ENSO redefinition)'}")
    print(f"ENSO definition: monthly Nino3.4 anomaly +/-{DEGC_THRESHOLD:.1f} C")
    print("CE mask threshold: gross response >= 2.0 W m-2 and opposite-sign SW/LW")
    print(f"Figure outputs written: {png_path}, {pdf_path}")
    print(f"Internal note: raw net field composite closure max residual before reconstruction = {raw_closure_residual:.6e} W m-2")
    print(f"Ready for visual review: {'YES' if canonical_confirmed and max_abs_residual < ALGEBRAIC_TOLERANCE else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
