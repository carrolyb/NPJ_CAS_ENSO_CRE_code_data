#!/usr/bin/env python3
"""Render Figure 08 under the +/-0.5 C ENSO definition using the Figure08 v2 copy.py layout."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import StrMethodFormatter


PROJECT_ROOT = Path("/Volumes/My Book/P3")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from enso_cloud.config import REGIONS


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PACKAGE_ROOT / "01_final_figure"
INPUT_DIR = PACKAGE_ROOT / "03_input_data"
RESULT_DIR = PACKAGE_ROOT / "04_key_results"
NOTES_DIR = PACKAGE_ROOT / "05_notes"

PRIMARY_NC = INPUT_DIR / "ceres_monthly_tropical_pacific.nc"
NINO_CSV = INPUT_DIR / "nino34_200207_202302.csv"
FIG07_BOOT_CSV = PACKAGE_ROOT.parent / "Figure07_SW_LW_Net_total_pathways_degC05" / "04_key_results" / "Figure07_degC05_group_SW_LW_Net_total_bootstrap.csv"
FIG07_METHOD = PACKAGE_ROOT.parent / "Figure07_SW_LW_Net_total_pathways_degC05" / "05_notes" / "Figure07_degC05_method_and_plot_checks.txt"

OUT_PNG = FIG_DIR / "Figure08_spatial_Net_total_pathways_degC05.png"
OUT_PDF = FIG_DIR / "Figure08_spatial_Net_total_pathways_degC05.pdf"
OUT_FINAL_INPUT = RESULT_DIR / "Figure08_degC05_final_plot_input.nc"
OUT_BOOTSTRAP = RESULT_DIR / "Figure08_degC05_spatial_Net_total_bootstrap.nc"
OUT_REGIONAL = RESULT_DIR / "Figure08_degC05_spatial_vs_Figure07_regional_record.csv"
OUT_SIGNIF = RESULT_DIR / "Figure08_degC05_spatial_significance_coverage_summary.csv"
OUT_CAPTION = NOTES_DIR / "Figure08_degC05_caption.md"
OUT_METHOD = NOTES_DIR / "Figure08_degC05_method_and_plot_checks.txt"
OUT_MANIFEST = NOTES_DIR / "Figure08_degC05_input_data_manifest.md"

GROUP_ORDER = [
    "low cloud",
    "mid-level cloud",
    "thin high cloud",
    "thick anvil cloud",
    "deep convective cloud",
]
GROUP_DEFINITIONS = {
    "low cloud": {"press": {"1000-800", "800-680"}, "opt": None},
    "mid-level cloud": {"press": {"680-560", "560-440"}, "opt": None},
    "thin high cloud": {"press": {"440-310", "310-180", "180-10"}, "opt": {"0.02-1.27", "1.27-3.55"}},
    "thick anvil cloud": {"press": {"440-310", "310-180", "180-10"}, "opt": {"3.55-9.38", "9.38-22.63"}},
    "deep convective cloud": {"press": {"440-310", "310-180", "180-10"}, "opt": {"22.63-60.36", "60.36-378.65"}},
}
FOCUS_GROUPS = [
    "low cloud",
    "thin high cloud",
    "thick anvil cloud",
    "deep convective cloud",
]
PANEL_TITLES = {
    "low cloud": "Low cloud",
    "thin high cloud": "Thin high cloud",
    "thick anvil cloud": "Thick anvil cloud",
    "deep convective cloud": "Deep convective cloud",
}
PANEL_LABELS = {
    "low cloud": "a",
    "thin high cloud": "b",
    "thick anvil cloud": "c",
    "deep convective cloud": "d",
}
REGION_ORDER = ["WP", "CP", "EP"]
REGION_MAP = {
    "WP": REGIONS["west_pacific"],
    "CP": REGIONS["central_pacific"],
    "EP": REGIONS["east_pacific"],
}
EXPECTED_TIME = 248
EXPECTED_LAT = 30
EXPECTED_LON = 160
EXPECTED_GRIDCELLS = 4800
BLOCK_LENGTH = 12
N_BOOT = 2000
SEED = 42
THRESHOLD = 0.5
NINO_COLUMN = "nino34_anom"
SIGN_ZERO_TOL = 1.0e-12
CI_LOW_Q = 2.5
CI_HIGH_Q = 97.5
FIXED_VMIN = -15.0
FIXED_VMAX = 15.0
CHUNK_SIZE = 80
TOL = 1.0e-10
DATA_CRS = ccrs.PlateCarree()
MAP_PROJ = ccrs.PlateCarree(central_longitude=180)

CAPTION_TEXT = (
    "Figure 8. Spatial Net total-contribution pathways for the four dominant cloud groups using the +/-0.5 C Nino3.4 "
    "ENSO definition. Panels (a)-(d) show El Nino minus La Nina daytime Net total contributions for low cloud, thin high "
    "cloud, thick anvil cloud, and deep convective cloud, respectively. Stippling marks grid cells whose pointwise 95% "
    "moving-block-bootstrap confidence intervals exclude zero. The plotted fields summarize cloud-group pathway structure and "
    "are not interpreted as an exact reconstruction of the direct all-sky Net CRE response.\n"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def ensure_inputs() -> None:
    for path in [FIG_DIR, RESULT_DIR, NOTES_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    required = [PRIMARY_NC, NINO_CSV, FIG07_BOOT_CSV, FIG07_METHOD]
    missing = [str(path) for path in required if not path.exists()]
    require(not missing, "Missing required input(s):\n" + "\n".join(missing))


def load_inputs() -> tuple[xr.Dataset, pd.DataFrame, pd.DataFrame]:
    ds = xr.open_dataset(PRIMARY_NC)
    nino = pd.read_csv(NINO_CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    fig07 = pd.read_csv(FIG07_BOOT_CSV)
    return ds, nino, fig07


def validate_dataset(ds: xr.Dataset) -> None:
    require(dict(ds.sizes) == {"time": EXPECTED_TIME, "lat": EXPECTED_LAT, "lon": EXPECTED_LON, "cloud_type": 42}, f"Unexpected dataset dimensions: {dict(ds.sizes)}")
    for var in ["net_contrib", "sw_contrib", "lw_contrib", "press_label", "opt_label"]:
        require(var in ds, f"Required variable missing from source dataset: {var}")
    require(float(ds["lon"].values[0]) == 120.5 and float(ds["lon"].values[-1]) == 279.5, "Unexpected longitude axis.")
    require(float(ds["lat"].values[0]) == -14.5 and float(ds["lat"].values[-1]) == 14.5, "Unexpected latitude axis.")


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


def build_group_mapping(ds: xr.Dataset) -> dict[str, np.ndarray]:
    mapping: dict[str, np.ndarray] = {}
    cloud_types = ds["cloud_type"].values.astype(int)
    press_labels = ds["press_label"].values.astype(object)
    opt_labels = ds["opt_label"].values.astype(object)
    assigned: list[int] = []
    for group in GROUP_ORDER:
        spec = GROUP_DEFINITIONS[group]
        press_mask = np.isin(press_labels, list(spec["press"]))
        opt_mask = np.ones_like(press_mask, dtype=bool) if spec["opt"] is None else np.isin(opt_labels, list(spec["opt"]))
        mask = press_mask & opt_mask
        mapping[group] = cloud_types[mask]
        assigned.extend(cloud_types[mask].tolist())
    require(sorted(assigned) == list(range(1, 43)), "Five-group mapping does not close on all 42 cloud types.")
    require(len(assigned) == len(set(assigned)), "Five-group mapping contains duplicate cloud types.")
    return mapping


def build_monthly_group_totals(ds: xr.Dataset, phase: np.ndarray, mapping: dict[str, np.ndarray]) -> xr.Dataset:
    shape = (EXPECTED_TIME, EXPECTED_LAT, EXPECTED_LON, len(GROUP_ORDER))
    sw = np.full(shape, np.nan, dtype=np.float64)
    lw = np.full(shape, np.nan, dtype=np.float64)
    net = np.full(shape, np.nan, dtype=np.float64)
    for g_idx, group in enumerate(GROUP_ORDER):
        cloud_idx = np.array(mapping[group], dtype=int) - 1
        sw[..., g_idx] = ds["sw_contrib"].isel(cloud_type=cloud_idx).sum(dim="cloud_type", skipna=True).values.astype(np.float64)
        lw[..., g_idx] = ds["lw_contrib"].isel(cloud_type=cloud_idx).sum(dim="cloud_type", skipna=True).values.astype(np.float64)
        net[..., g_idx] = sw[..., g_idx] + lw[..., g_idx]
    require(np.nanmax(np.abs(net - (sw + lw))) <= TOL, "Monthly group totals violate Net = SW + LW.")
    out = xr.Dataset(
        data_vars={
            "sw_total_contrib_monthly": (("time", "lat", "lon", "physical_group"), sw),
            "lw_total_contrib_monthly": (("time", "lat", "lon", "physical_group"), lw),
            "net_total_contrib_monthly": (("time", "lat", "lon", "physical_group"), net),
            "phase_index": (("time",), phase.astype(np.int8)),
        },
        coords={
            "time": ds["time"].values,
            "lat": ds["lat"].values,
            "lon": ds["lon"].values,
            "physical_group": GROUP_ORDER,
        },
    )
    out.attrs.update(
        {
            "title": "Figure08 degC05 monthly daytime group total contribution intermediate product",
            "source_file": str(PRIMARY_NC),
            "daytime_based": "True",
            "paired_valid_rule": "True",
            "all42_joint_strict_mask": "False",
            "net_definition": "recomputed_sw_contrib + recomputed_lw_contrib",
            "enso_phase_index_mapping": "-1=La Nina, 0=Neutral/other, 1=El Nino",
            "el_nino_months": int((phase == 1).sum()),
            "la_nina_months": int((phase == -1).sum()),
        }
    )
    return out


def phase_diff(series: np.ndarray, phase_sign: np.ndarray) -> np.ndarray:
    el = np.nanmean(series[phase_sign == 1], axis=0)
    la = np.nanmean(series[phase_sign == -1], axis=0)
    return el - la


def build_bootstrap_indices(n_time: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n_blocks = math.ceil(n_time / BLOCK_LENGTH)
    start_max = n_time - BLOCK_LENGTH + 1
    require(start_max > 0, "Time series shorter than block length.")
    indices = np.empty((N_BOOT, n_time), dtype=np.int16)
    for i in range(N_BOOT):
        starts = rng.integers(0, start_max, size=n_blocks)
        indices[i, :] = np.concatenate([np.arange(s, s + BLOCK_LENGTH) for s in starts])[:n_time]
    return indices


def summarize_bootstrap_distribution(dist: np.ndarray, deterministic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ci_low = np.nanpercentile(dist, CI_LOW_Q, axis=0)
    ci_high = np.nanpercentile(dist, CI_HIGH_Q, axis=0)
    significant = (ci_low > 0.0) | (ci_high < 0.0)
    det_sign = np.sign(np.where(np.abs(deterministic) <= SIGN_ZERO_TOL, 0.0, deterministic))
    sign_prob = np.empty(deterministic.shape, dtype=np.float64)
    pos = det_sign > 0
    neg = det_sign < 0
    zero = det_sign == 0
    if np.any(pos):
        sign_prob[pos] = np.mean(dist[:, pos] > SIGN_ZERO_TOL, axis=0)
    if np.any(neg):
        sign_prob[neg] = np.mean(dist[:, neg] < -SIGN_ZERO_TOL, axis=0)
    if np.any(zero):
        sign_prob[zero] = np.mean(np.abs(dist[:, zero]) <= SIGN_ZERO_TOL, axis=0)
    return ci_low, ci_high, significant, sign_prob


def compute_spatial_bootstrap(monthly_ds: xr.Dataset) -> tuple[xr.DataArray, xr.Dataset, pd.DataFrame]:
    net = monthly_ds["net_total_contrib_monthly"].sel(physical_group=FOCUS_GROUPS).transpose("time", "physical_group", "lat", "lon").astype(np.float64)
    phase = monthly_ds["phase_index"].astype(np.int8)
    enso_mask = (phase == 1) | (phase == -1)
    phase_enso = phase.where(enso_mask, drop=True).values.astype(np.int8)
    net_enso = net.where(enso_mask, drop=True).values.astype(np.float64)
    require(net_enso.shape == (54 + 85, len(FOCUS_GROUPS), EXPECTED_LAT, EXPECTED_LON), f"Unexpected ENSO-only shape: {net_enso.shape}")

    det_values = phase_diff(net_enso, phase_enso)
    det = xr.DataArray(
        det_values,
        coords={"physical_group": FOCUS_GROUPS, "lat": net["lat"].values, "lon": net["lon"].values},
        dims=("physical_group", "lat", "lon"),
        name="delta_net_total_spatial",
    )

    boot_indices = build_bootstrap_indices(len(phase_enso))
    sampled_phase = phase_enso[boot_indices]
    n_groups = len(FOCUS_GROUPS)
    n_cells = EXPECTED_LAT * EXPECTED_LON
    det_flat = det_values.reshape(n_groups, n_cells)
    ci_low = np.empty((n_groups, n_cells), dtype=np.float64)
    ci_high = np.empty((n_groups, n_cells), dtype=np.float64)
    significant = np.empty((n_groups, n_cells), dtype=bool)
    sign_prob = np.empty((n_groups, n_cells), dtype=np.float64)

    for g in range(n_groups):
        group_data = net_enso[:, g, :, :].reshape(len(phase_enso), n_cells)
        require(np.isfinite(group_data).all(), f"Non-finite ENSO values for {FOCUS_GROUPS[g]}.")
        for start in range(0, n_cells, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_cells)
            chunk = group_data[:, start:end]
            sampled_values = chunk[boot_indices]
            el_mask = sampled_phase == 1
            la_mask = sampled_phase == -1
            el_count = el_mask.sum(axis=1)
            la_count = la_mask.sum(axis=1)
            require(np.all(el_count > 0) and np.all(la_count > 0), "Bootstrap replicate lost one ENSO phase.")
            el_sum = np.sum(np.where(el_mask[:, :, None], sampled_values, 0.0), axis=1)
            la_sum = np.sum(np.where(la_mask[:, :, None], sampled_values, 0.0), axis=1)
            dist = (el_sum / el_count[:, None]) - (la_sum / la_count[:, None])
            low, high, sig, prob = summarize_bootstrap_distribution(dist, det_flat[g, start:end])
            ci_low[g, start:end] = low
            ci_high[g, start:end] = high
            significant[g, start:end] = sig
            sign_prob[g, start:end] = prob

    display_mask = np.isfinite(det_values)
    false_mask = np.zeros_like(display_mask, dtype=bool)
    bootstrap_ds = xr.Dataset(
        data_vars={
            "delta_net_total_spatial": (("physical_group", "lat", "lon"), det_values.astype(np.float64)),
            "ci_low_95": (("physical_group", "lat", "lon"), ci_low.reshape(n_groups, EXPECTED_LAT, EXPECTED_LON)),
            "ci_high_95": (("physical_group", "lat", "lon"), ci_high.reshape(n_groups, EXPECTED_LAT, EXPECTED_LON)),
            "significant": (("physical_group", "lat", "lon"), significant.reshape(n_groups, EXPECTED_LAT, EXPECTED_LON).astype(np.int8)),
            "sign_stability_probability": (("physical_group", "lat", "lon"), sign_prob.reshape(n_groups, EXPECTED_LAT, EXPECTED_LON)),
            "display_mask": (("physical_group", "lat", "lon"), display_mask.astype(np.int8)),
            "plot_stipple": (("physical_group", "lat", "lon"), (significant.reshape(n_groups, EXPECTED_LAT, EXPECTED_LON) & display_mask).astype(np.int8)),
            "plot_hatch": (("physical_group", "lat", "lon"), false_mask.astype(np.int8)),
        },
        coords={"physical_group": FOCUS_GROUPS, "lat": net["lat"].values, "lon": net["lon"].values},
    )
    bootstrap_ds.attrs.update(
        {
            "figure": "Figure 8",
            "source_product": "daytime",
            "spatial_metric": "candidate-compatible Net total-contribution pathway",
            "groups": ", ".join(FOCUS_GROUPS),
            "mid_level_cloud_excluded_from_main_panels": "True",
            "enso_definition": "El Nino minus La Nina (+/-0.5 C Nino3.4)",
            "el_nino_months": 54,
            "la_nina_months": 85,
            "bootstrap_method": "joint moving-block bootstrap",
            "block_length_months": BLOCK_LENGTH,
            "n_boot": N_BOOT,
            "random_seed": SEED,
            "confidence_interval": "percentile 95 percent pointwise CI",
            "shared_bootstrap_indices_across_groups_and_gridcells": "True",
            "plot_hatch_used": "False",
            "all42_joint_strict_mask_used": "False",
            "exact_direct_allsky_reconstruction_claim": "False",
        }
    )
    signif_rows = []
    for g_idx, group in enumerate(FOCUS_GROUPS):
        field = det_values[g_idx]
        sig = significant.reshape(n_groups, EXPECTED_LAT, EXPECTED_LON)[g_idx]
        pos = field > SIGN_ZERO_TOL
        neg = field < -SIGN_ZERO_TOL
        signif_rows.append(
            {
                "physical_group": group,
                "n_total_gridcells": EXPECTED_GRIDCELLS,
                "n_display_gridcells": EXPECTED_GRIDCELLS,
                "n_significant_gridcells": int(sig.sum()),
                "significant_fraction": float(sig.sum() / EXPECTED_GRIDCELLS),
                "n_positive_gridcells": int(pos.sum()),
                "n_negative_gridcells": int(neg.sum()),
                "n_significant_positive_gridcells": int((sig & pos).sum()),
                "n_significant_negative_gridcells": int((sig & neg).sum()),
            }
        )
    signif_df = pd.DataFrame(signif_rows)
    return det, bootstrap_ds, signif_df


def build_region_mask(lon: xr.DataArray, lat: xr.DataArray, region_code: str) -> xr.DataArray:
    region = REGION_MAP[region_code]
    lon_mask = (lon >= region.lon_min) & (lon < region.lon_max if region_code != "EP" else lon <= region.lon_max)
    lat_mask = (lat >= region.lat_min) & (lat <= region.lat_max)
    lat_2d, lon_2d = xr.broadcast(lat_mask.astype(bool), lon_mask.astype(bool))
    return lat_2d & lon_2d


def weighted_monthly_region_series(net: xr.DataArray, region_code: str) -> xr.DataArray:
    region_mask = build_region_mask(net["lon"], net["lat"], region_code)
    weights_1d = xr.DataArray(np.cos(np.deg2rad(net["lat"].values)), coords={"lat": net["lat"]}, dims=("lat",))
    weights_2d = weights_1d.broadcast_like(region_mask.astype(np.float64))
    weights = weights_2d.where(region_mask, 0.0)
    denom = weights.sum(dim=("lat", "lon"), skipna=True)
    return ((net * weights).sum(dim=("lat", "lon"), skipna=True) / denom).transpose("time", "physical_group")


def bootstrap_regional_series(monthly_ds: xr.Dataset, fig07_df: pd.DataFrame) -> pd.DataFrame:
    net = monthly_ds["net_total_contrib_monthly"].sel(physical_group=FOCUS_GROUPS).transpose("time", "physical_group", "lat", "lon").astype(np.float64)
    phase = monthly_ds["phase_index"].astype(np.int8)
    enso_mask = (phase == 1) | (phase == -1)
    phase_enso = phase.where(enso_mask, drop=True).values.astype(np.int8)
    boot_indices = build_bootstrap_indices(len(phase_enso))
    sampled_phase = phase_enso[boot_indices]
    fig07 = fig07_df.loc[fig07_df["component"] == "Net"].copy()

    rows: list[dict[str, object]] = []
    regional_monthly = xr.concat(
        [weighted_monthly_region_series(net, region).expand_dims(region=[region]) for region in REGION_ORDER],
        dim="region",
    )

    for region in REGION_ORDER:
        series = regional_monthly.sel(region=region).where(enso_mask, drop=True).transpose("time", "physical_group").values.astype(np.float64)
        require(series.shape == (54 + 85, len(FOCUS_GROUPS)), f"Unexpected regional ENSO series shape for {region}: {series.shape}")
        sampled_values = series[boot_indices]
        el_mask = sampled_phase == 1
        la_mask = sampled_phase == -1
        el_count = el_mask.sum(axis=1)
        la_count = la_mask.sum(axis=1)
        el_sum = np.sum(np.where(el_mask[:, :, None], sampled_values, 0.0), axis=1)
        la_sum = np.sum(np.where(la_mask[:, :, None], sampled_values, 0.0), axis=1)
        dist = (el_sum / el_count[:, None]) - (la_sum / la_count[:, None])
        deterministic = phase_diff(series, phase_enso)
        ci_low, ci_high, significant, sign_prob = summarize_bootstrap_distribution(dist, deterministic)
        for g_idx, group in enumerate(FOCUS_GROUPS):
            ref = fig07.loc[(fig07["region"] == region) & (fig07["physical_group"] == group)].iloc[0]
            rows.append(
                {
                    "region": region,
                    "physical_group": group,
                    "spatial_regional_estimate": float(deterministic[g_idx]),
                    "spatial_ci_low_95": float(ci_low[g_idx]),
                    "spatial_ci_high_95": float(ci_high[g_idx]),
                    "spatial_significant": bool(significant[g_idx]),
                    "spatial_sign_stability_probability": float(sign_prob[g_idx]),
                    "Figure07_candidate_estimate": float(ref["deterministic_estimate"]),
                    "Figure07_ci_low": float(ref["ci_low_95"]),
                    "Figure07_ci_high": float(ref["ci_high_95"]),
                    "Figure07_significant": bool(ref["significant"]),
                    "estimate_difference_spatial_minus_Figure07": float(deterministic[g_idx] - float(ref["deterministic_estimate"])),
                    "sign_consistent": bool(np.sign(deterministic[g_idx]) == np.sign(float(ref["deterministic_estimate"])) or (abs(deterministic[g_idx]) <= SIGN_ZERO_TOL and abs(float(ref["deterministic_estimate"])) <= SIGN_ZERO_TOL)),
                    "significance_consistent": bool(bool(significant[g_idx]) == bool(ref["significant"])),
                }
            )
    return pd.DataFrame(rows)


def build_final_plot_input(bootstrap_ds: xr.Dataset) -> xr.Dataset:
    out = xr.Dataset(
        data_vars={
            "delta_net_total_spatial": bootstrap_ds["delta_net_total_spatial"],
            "ci_low_95": bootstrap_ds["ci_low_95"],
            "ci_high_95": bootstrap_ds["ci_high_95"],
            "plot_stipple": bootstrap_ds["plot_stipple"],
            "display_mask": bootstrap_ds["display_mask"],
            "plot_hatch": bootstrap_ds["plot_hatch"],
        },
        coords={
            "physical_group": bootstrap_ds["physical_group"],
            "lat": bootstrap_ds["lat"],
            "lon": bootstrap_ds["lon"],
        },
    )
    out.attrs.update(
        {
            "final_panels": ", ".join(FOCUS_GROUPS),
            "stippling_meaning": "pointwise 95 percent moving-block-bootstrap confidence interval excludes zero",
            "no_hatching_required": "True",
            "spatial_patterns_correspond_closely_to_integrated_regional_candidate_pathways": "True",
            "exact_direct_allsky_reconstruction_not_claimed": "True",
            "enso_definition": "El Nino minus La Nina (+/-0.5 C Nino3.4)",
        }
    )
    return out


def lon_label(value: float) -> str:
    if value == 180:
        return "180"
    if value < 180:
        return f"{int(value)}E"
    return f"{int(round(360 - value))}W"


def lat_label(value: float) -> str:
    if value < 0:
        return f"{int(abs(value))}S"
    if value > 0:
        return f"{int(value)}N"
    return "0"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.0, 1.03, f"({label})", transform=ax.transAxes, ha="left", va="bottom", fontsize=12, fontweight="bold")


def make_axes(fig: plt.Figure) -> list[plt.Axes]:
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.985, bottom=0.1, top=0.955, wspace=0.10, hspace=-0.4)
    axes = [fig.add_subplot(gs[i, j], projection=MAP_PROJ) for i in range(2) for j in range(2)]
    xticks = [120, 150, 180, 210, 240, 270]
    yticks = [-10, 0, 10]
    for idx, ax in enumerate(axes):
        ax.set_extent([120, 280, -15, 15], crs=DATA_CRS)
        ax.add_feature(cfeature.LAND, facecolor="#efe7d7", edgecolor="none", zorder=0)
        ax.coastlines(resolution="110m", linewidth=0.45, zorder=2)
        ax.set_xticks(xticks, crs=DATA_CRS)
        ax.set_xticklabels([lon_label(v) for v in xticks], fontsize=12)
        ax.set_yticks(yticks, crs=DATA_CRS)
        ax.set_yticklabels([lat_label(v) for v in yticks], fontsize=12)
        ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=4, width=0.8, pad=2)
        if idx % 2 == 1:
            ax.set_yticklabels([])
        if idx < 2:
            ax.set_xticklabels([])
        ax.plot([120, 280], [0, 0], color="#4b5563", linewidth=0.55, linestyle="--", alpha=0.55, transform=DATA_CRS, zorder=3)
        for boundary in [160, 210]:
            ax.plot([boundary, boundary], [-15, 15], color="#111827", linewidth=0.9, transform=DATA_CRS, zorder=3)
        if idx == 0:
            for xpos, label in [(125, "WP"), (165, "CP"), (215, "EP")]:
                ax.text(xpos, 11.0, label, transform=DATA_CRS, ha="center", va="center", fontsize=14, color="#111827", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.60, "pad": 0.6}, zorder=5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#202020")
    return axes


def draw_figure(plot_ds: xr.Dataset) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "figure.titlesize": 12,
            "savefig.dpi": 300,
        }
    )
    fig = plt.figure(figsize=(12.0, 5))
    axes = make_axes(fig)
    norm = TwoSlopeNorm(vmin=FIXED_VMIN, vcenter=0.0, vmax=FIXED_VMAX)
    cmap = plt.get_cmap("RdBu_r")
    mappable = None
    for ax, group in zip(axes, FOCUS_GROUPS):
        sub = plot_ds.sel(physical_group=group)
        mesh = ax.pcolormesh(plot_ds["lon"].values, plot_ds["lat"].values, sub["delta_net_total_spatial"].values, shading="auto", cmap=cmap, norm=norm, transform=DATA_CRS, zorder=1)
        mappable = mesh
        yy, xx = np.where(sub["plot_stipple"].astype(bool).values)
        ax.scatter(plot_ds["lon"].values[xx], plot_ds["lat"].values[yy], s=1.6, c="black", alpha=0.52, linewidths=0.0, transform=DATA_CRS, rasterized=True, zorder=4)
        ax.set_title(PANEL_TITLES[group], fontsize=13, pad=6)
        panel_label(ax, PANEL_LABELS[group])
    require(mappable is not None, "No mappable created for Figure08.")
    cax = fig.add_axes([0.18, 0.16, 0.64, 0.034])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label("Net total contribution (W m$^{-2}$)", fontsize=14)
    cbar.ax.tick_params(labelsize=12, direction="in", length=4, width=0.8)
    cbar.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_caption() -> None:
    OUT_CAPTION.write_text(CAPTION_TEXT, encoding="utf-8")


def write_manifest() -> None:
    lines = [
        "# Figure08 degC05 input manifest",
        "",
        f"- Primary gridded monthly file: `{PRIMARY_NC}`",
        f"- ENSO index file: `{NINO_CSV}`",
        f"- Figure07 degC05 Net bootstrap reference: `{FIG07_BOOT_CSV}`",
    ]
    OUT_MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method(final_ds: xr.Dataset, bootstrap_ds: xr.Dataset, signif_df: pd.DataFrame, regional_df: pd.DataFrame) -> None:
    lines = [
        "Figure08 degC05 method and plot checks",
        "",
        "- plotting style source = Figure08 spatial Net total pathways final v2 copy.py",
        "- ENSO definition = nino34_anom with El Nino >= +0.5 C and La Nina <= -0.5 C",
        "- source product = daytime gridded monthly cloud-type contribution chain",
        "- paired-valid inherited through the gridded candidate-compatible contribution chain",
        "- no all-42 joint strict mask used",
        "- no hatch drawn anywhere",
        "- focus groups = low cloud, thin high cloud, thick anvil cloud, deep convective cloud",
        "- mid-level cloud excluded from the main panels",
        f"- El Nino months = {bootstrap_ds.attrs['el_nino_months']}",
        f"- La Nina months = {bootstrap_ds.attrs['la_nina_months']}",
        f"- bootstrap block length (months) = {BLOCK_LENGTH}",
        f"- bootstrap samples = {N_BOOT}",
        f"- bootstrap seed = {SEED}",
        "- bootstrap method = joint moving-block bootstrap on the ENSO-month subset",
        f"- final display gridcells per panel = {EXPECTED_GRIDCELLS}",
        f"- shared symmetric color scale = [{FIXED_VMIN:.0f}, {FIXED_VMAX:.0f}] W m^-2",
        "",
        "Reference cross-checks",
        f"- Figure07 degC05 method reference read = {FIG07_METHOD.exists()}",
        f"- all regional spatial signs consistent with Figure07 Net = {bool(regional_df['sign_consistent'].all())}",
        f"- all regional spatial significance flags consistent with Figure07 Net = {bool(regional_df['significance_consistent'].all())}",
        "",
        "Pointwise significance coverage",
    ]
    for _, row in signif_df.iterrows():
        lines.append(f"- {row['physical_group']}: significant={int(row['n_significant_gridcells'])}/{int(row['n_total_gridcells'])} ({row['significant_fraction']:.6f})")
    lines.extend(
        [
            "",
            "Output files",
            f"- png: {OUT_PNG}",
            f"- pdf: {OUT_PDF}",
            f"- final plot input: {OUT_FINAL_INPUT}",
            f"- spatial bootstrap: {OUT_BOOTSTRAP}",
            f"- regional record: {OUT_REGIONAL}",
            f"- significance coverage: {OUT_SIGNIF}",
            f"- caption: {OUT_CAPTION}",
            f"- input manifest: {OUT_MANIFEST}",
        ]
    )
    OUT_METHOD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_inputs()
    ds, nino, fig07 = load_inputs()
    validate_dataset(ds)
    phase = build_phase(ds["time"].values, nino)
    mapping = build_group_mapping(ds)
    monthly_ds = build_monthly_group_totals(ds, phase, mapping)
    det, bootstrap_ds, signif_df = compute_spatial_bootstrap(monthly_ds)
    regional_df = bootstrap_regional_series(monthly_ds, fig07)
    final_ds = build_final_plot_input(bootstrap_ds)

    bootstrap_ds.to_netcdf(OUT_BOOTSTRAP)
    final_ds.to_netcdf(OUT_FINAL_INPUT)
    signif_df.to_csv(OUT_SIGNIF, index=False)
    regional_df.to_csv(OUT_REGIONAL, index=False)
    draw_figure(final_ds)
    write_caption()
    write_manifest()
    write_method(final_ds, bootstrap_ds, signif_df, regional_df)


if __name__ == "__main__":
    main()
