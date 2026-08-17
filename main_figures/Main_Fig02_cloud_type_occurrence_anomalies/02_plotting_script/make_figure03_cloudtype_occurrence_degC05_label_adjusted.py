#!/usr/bin/env python3
"""Generate packaged Figure 03 using a +/-0.5 C Nino3.4 definition."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PACKAGE_ROOT / "01_final_figure"
INPUT_DIR = PACKAGE_ROOT / "03_input_data"
RESULT_DIR = PACKAGE_ROOT / "04_key_results"
NOTES_DIR = PACKAGE_ROOT / "05_notes"
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BLOCK_LENGTH = 12
DEFAULT_RANDOM_SEED = 42
NINO_COLUMN = "nino34_anom"
THRESHOLD = 0.5
SMALL = 1.0e-12
X_EDGES = [0.0, 1.27, 3.55, 9.38, 22.63, 60.36, 378.65]
Y_EDGES = [0, 180, 310, 440, 560, 680, 800, 1000]

REGION_ORDER = ["TP", "WP", "CP", "EP"]
REGION_NAME_MAP = {
    "TP": "tropical_pacific",
    "WP": "west_pacific",
    "CP": "central_pacific",
    "EP": "east_pacific",
}
REGION_TITLE_MAP = {
    "TP": "Tropical Pacific",
    "WP": "Western Pacific",
    "CP": "Central Pacific",
    "EP": "Eastern Pacific",
}
PANEL_LABELS = {"TP": "a", "WP": "b", "CP": "c", "EP": "d"}
REGION_SHORT_FROM_NATIVE = {value: key for key, value in REGION_NAME_MAP.items()}
PRESS_ORDER = ["180-10", "310-180", "440-310", "560-440", "680-560", "800-680", "1000-800"]
OPT_ORDER = ["0.02-1.27", "1.27-3.55", "3.55-9.38", "9.38-22.63", "22.63-60.36", "60.36-378.65"]
GROUP_ORDER = ["low_cloud", "mid_level", "thin_high", "thick_anvil", "deep_convective"]
GROUP_LABELS = {
    "low_cloud": "Low cloud",
    "mid_level": "Mid-level\n  cloud",
    "thin_high": "Thin high",
    "thick_anvil": "Thick anvil",
    "deep_convective": "Deep convective",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "figure.titlesize": 15,
            "savefig.dpi": 220,
        }
    )


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


def load_predictor(nino_path: Path) -> pd.DataFrame:
    nino_df = pd.read_csv(nino_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    work = nino_df[["date", NINO_COLUMN]].copy()
    work["time"] = work["date"].dt.to_period("M").dt.to_timestamp()

    el = work.loc[work[NINO_COLUMN] >= THRESHOLD, ["time", NINO_COLUMN]].copy()
    el["phase"] = "El Nino"
    la = work.loc[work[NINO_COLUMN] <= -THRESHOLD, ["time", NINO_COLUMN]].copy()
    la["phase"] = "La Nina"
    monthly_samples = pd.concat([el, la], ignore_index=True).sort_values("time").reset_index(drop=True)

    if len(el) != 54 or len(la) != 85:
        raise RuntimeError(f"Unexpected +/-0.5 C ENSO month counts from predictor: El Nino={len(el)}, La Nina={len(la)}")
    return work


def build_old_group_members() -> dict[str, set[int]]:
    old: dict[str, set[int]] = {}
    old["low_cloud"] = set(range(1, 13))
    old["thin_high"] = set()
    old["thick_anvil"] = set()
    old["deep_convective"] = set()
    for press_index in [4, 5, 6]:
        for opt_index in [0, 1]:
            old["thin_high"].add(press_index * 6 + opt_index + 1)
        for opt_index in [2, 3]:
            old["thick_anvil"].add(press_index * 6 + opt_index + 1)
        old["deep_convective"].add(press_index * 6 + 4 + 1)
    return old


def build_v2_group_members() -> dict[str, set[int]]:
    members: dict[str, set[int]] = {
        "low_cloud": set(),
        "mid_level": set(),
        "thin_high": set(),
        "thick_anvil": set(),
        "deep_convective": set(),
    }
    for press_index in [0, 1]:
        for opt_index in range(6):
            members["low_cloud"].add(press_index * 6 + opt_index + 1)
    for press_index in [2, 3]:
        for opt_index in range(6):
            members["mid_level"].add(press_index * 6 + opt_index + 1)
    for press_index in [4, 5, 6]:
        for opt_index in [0, 1]:
            members["thin_high"].add(press_index * 6 + opt_index + 1)
        for opt_index in [2, 3]:
            members["thick_anvil"].add(press_index * 6 + opt_index + 1)
        for opt_index in [4, 5]:
            members["deep_convective"].add(press_index * 6 + opt_index + 1)
    return members


def closure_check(group_members: dict[str, set[int]]) -> dict[str, object]:
    all_cells = set(range(1, 43))
    assigned_list: list[int] = []
    for group in GROUP_ORDER:
        assigned_list.extend(sorted(group_members[group]))
    assigned = set(assigned_list)
    duplicated = sorted({cell for cell in assigned_list if assigned_list.count(cell) > 1})
    unassigned = sorted(all_cells - assigned)
    return {
        "total_cells": 42,
        "assigned_cells": len(assigned),
        "duplicated_cells": duplicated,
        "unassigned_cells": unassigned,
    }


def axis_frame_from_dataset(regional_anom: xr.Dataset) -> pd.DataFrame:
    cols = ["cloud_type", "press_index", "opt_index", "press_label", "opt_label", "cloud_label", "cloud_long_name"]
    return regional_anom["cf"].to_dataframe(name="cf").reset_index()[cols].drop_duplicates().sort_values("cloud_type").reset_index(drop=True)


def build_group_boxes(group_members: dict[str, set[int]], axis_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    mapping = axis_df.set_index("cloud_type")
    boxes: dict[str, dict[str, float]] = {}
    for group in GROUP_ORDER:
        subset = mapping.loc[sorted(group_members[group])]
        x_values = subset["opt_index"].astype(int).to_numpy()
        y_values = subset["press_label"].map({label: idx for idx, label in enumerate(PRESS_ORDER)}).astype(int).to_numpy()
        boxes[group] = {
            "x": float(x_values.min() - 0.5),
            "y": float(y_values.min() - 0.5),
            "width": float(x_values.max() - x_values.min() + 1),
            "height": float(y_values.max() - y_values.min() + 1),
        }
    return boxes


def build_monthly_cf_actual(regional_anom: xr.Dataset, regional_clim: xr.Dataset, predictor_df: pd.DataFrame) -> pd.DataFrame:
    anom_df = regional_anom["cf"].to_dataframe(name="cf_anom").reset_index()
    clim_df = regional_clim["cf"].to_dataframe(name="cf_clim").reset_index()
    work = anom_df.merge(
        predictor_df[["time", NINO_COLUMN]].set_index("time")[NINO_COLUMN].rename(NINO_COLUMN),
        left_on="time",
        right_index=True,
        how="left",
        validate="many_to_one",
    )
    work["month_num"] = pd.to_datetime(work["time"]).dt.month
    work = work.merge(
        clim_df[["region", "month", "cloud_type", "cf_clim"]],
        left_on=["region", "month_num", "cloud_type"],
        right_on=["region", "month", "cloud_type"],
        how="left",
        validate="many_to_one",
    )
    work["cf_actual"] = work["cf_anom"] + work["cf_clim"]
    work["region"] = work["region"].map(REGION_SHORT_FROM_NATIVE)
    if work["cf_actual"].isna().any():
        raise RuntimeError("Figure03 degC05 CF actual values contain NaN after climatology merge.")
    return work


def build_42class_summary(monthly_cf_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    region_checks: list[dict[str, object]] = []
    for region in REGION_ORDER:
        region_df = monthly_cf_df[monthly_cf_df["region"] == region].copy()
        el_region = region_df[region_df[NINO_COLUMN] >= THRESHOLD]
        la_region = region_df[region_df[NINO_COLUMN] <= -THRESHOLD]
        group_cols = ["cloud_type", "press_index", "opt_index", "press_label", "opt_label", "cloud_label", "cloud_long_name"]
        grouped_el = el_region.groupby(group_cols, as_index=False)["cf_actual"].mean().rename(columns={"cf_actual": "el_nino_cf"})
        grouped_la = la_region.groupby(group_cols, as_index=False)["cf_actual"].mean().rename(columns={"cf_actual": "la_nina_cf"})
        merged = grouped_el.merge(grouped_la, on=group_cols, how="inner", validate="one_to_one")
        merged["delta_cf"] = merged["el_nino_cf"] - merged["la_nina_cf"]
        merged["region"] = region
        rows.extend(merged.to_dict("records"))
        region_checks.append(
            {
                "region": region,
                "delta_sum42": float(math.fsum(np.asarray(merged["delta_cf"].to_numpy(), dtype=np.float64))),
                "n_el_nino_months": int(el_region["time"].nunique()),
                "n_la_nina_months": int(la_region["time"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    out["region"] = pd.Categorical(out["region"], categories=REGION_ORDER, ordered=True)
    out["press_order"] = out["press_label"].map({label: idx for idx, label in enumerate(PRESS_ORDER)})
    out = out.sort_values(["region", "press_order", "opt_index"]).drop(columns=["press_order"]).reset_index(drop=True)
    return out, pd.DataFrame(region_checks).sort_values("region").reset_index(drop=True)


def build_monthly_series(monthly_cf_df: pd.DataFrame, members: dict[str, set[int]], mode: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if mode == "group":
        items = GROUP_ORDER
        label = "cloud_group"
    elif mode == "cell":
        items = sorted(set(range(1, 43)))
        label = "cloud_type"
    else:
        raise ValueError(mode)

    for item in items:
        if mode == "group":
            subset = monthly_cf_df[monthly_cf_df["cloud_type"].isin(members[item])]
            grouped = subset.groupby(["region", "time"], as_index=False)["cf_actual"].sum()
            grouped[label] = item
        else:
            grouped = monthly_cf_df[monthly_cf_df["cloud_type"] == item].groupby(["region", "time"], as_index=False)["cf_actual"].mean()
            grouped[label] = item
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def build_group_point_summary_from_42class(summary42_df: pd.DataFrame, members: dict[str, set[int]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for region in REGION_ORDER:
        subset = summary42_df[summary42_df["region"] == region]
        for group in GROUP_ORDER:
            part = subset[subset["cloud_type"].isin(members[group])]
            rows.append(
                {
                    "region": region,
                    "cloud_group": group,
                    "delta_cf": float(math.fsum(np.asarray(part["delta_cf"].to_numpy(), dtype=np.float64))),
                }
            )
    out = pd.DataFrame(rows)
    out["region"] = pd.Categorical(out["region"], categories=REGION_ORDER, ordered=True)
    out["cloud_group"] = pd.Categorical(out["cloud_group"], categories=GROUP_ORDER, ordered=True)
    return out.sort_values(["region", "cloud_group"]).reset_index(drop=True)


def build_block_indices(n_time: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    max_start = n_time - block_length
    if max_start < 0:
        raise ValueError("block_length exceeds sample length")
    n_blocks = math.ceil(n_time / block_length)
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    blocks = [np.arange(start, start + block_length, dtype=int) for start in starts]
    return np.concatenate(blocks)[:n_time]


def quantile_ci(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan
    lower, upper = np.nanpercentile(finite, [2.5, 97.5])
    return float(lower), float(upper)


def significant_from_ci(ci_low: float, ci_high: float) -> bool:
    return bool(np.isfinite(ci_low) and np.isfinite(ci_high) and ((ci_low > 0.0) or (ci_high < 0.0)))


def bootstrap_delta_series(
    monthly_series_df: pd.DataFrame,
    predictor_df: pd.DataFrame,
    key_col: str,
    keys: list[object],
    bootstrap_samples: int,
    block_length: int,
    seed: int,
) -> pd.DataFrame:
    predictor = predictor_df[["time", NINO_COLUMN]].sort_values("time").reset_index(drop=True)
    predictor_values = predictor[NINO_COLUMN].to_numpy(dtype=np.float64)
    n_time = len(predictor)
    rng = np.random.default_rng(seed)

    series_store: dict[tuple[str, object], np.ndarray] = {}
    point_store: dict[tuple[str, object], dict[str, object]] = {}
    for region in REGION_ORDER:
        for key in keys:
            series_df = monthly_series_df[(monthly_series_df["region"] == region) & (monthly_series_df[key_col] == key)][["time", "cf_actual"]].sort_values("time").reset_index(drop=True)
            merged = predictor.merge(series_df, on="time", how="inner", validate="one_to_one")
            if len(merged) != n_time:
                raise RuntimeError(f"Bootstrap alignment failed for {region} {key_col}={key}.")
            values = merged["cf_actual"].to_numpy(dtype=np.float64)
            series_store[(region, key)] = values
            point_store[(region, key)] = {
                "delta_cf": float(values[predictor_values >= THRESHOLD].mean() - values[predictor_values <= -THRESHOLD].mean()),
                "n_el_nino": int((predictor_values >= THRESHOLD).sum()),
                "n_la_nina": int((predictor_values <= -THRESHOLD).sum()),
            }

    sample_store: dict[tuple[str, object], list[float]] = {(region, key): [] for region in REGION_ORDER for key in keys}
    for _ in range(bootstrap_samples):
        sample_index = build_block_indices(n_time, block_length, rng)
        boot_predictor = predictor_values[sample_index]
        el_mask = boot_predictor >= THRESHOLD
        la_mask = boot_predictor <= -THRESHOLD
        for region in REGION_ORDER:
            for key in keys:
                if not el_mask.any() or not la_mask.any():
                    sample_store[(region, key)].append(np.nan)
                    continue
                values = series_store[(region, key)][sample_index]
                sample_store[(region, key)].append(float(values[el_mask].mean() - values[la_mask].mean()))

    rows: list[dict[str, object]] = []
    for region in REGION_ORDER:
        for key in keys:
            samples = np.asarray(sample_store[(region, key)], dtype=np.float64)
            ci_low, ci_high = quantile_ci(samples)
            point = point_store[(region, key)]
            row = {
                "region": region,
                key_col: key,
                "delta_cf": point["delta_cf"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "significant_95": significant_from_ci(ci_low, ci_high),
                "bootstrap_mean": float(np.nanmean(samples)),
                "bootstrap_samples": int(bootstrap_samples),
                "block_length_months": int(block_length),
                "random_seed": int(seed),
                "n_el_nino": int(point["n_el_nino"]),
                "n_la_nina": int(point["n_la_nina"]),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def build_validation_text(
    closure: dict[str, object],
    region_checks: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    cell_boot_df: pd.DataFrame,
    old_groups: dict[str, set[int]],
    new_groups: dict[str, set[int]],
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    lines.append("Figure03 degC05 validation")
    lines.append("")
    lines.append("1. Five-group mask closure")
    lines.append(f"- total number of 42 cells: {closure['total_cells']}")
    lines.append(f"- number of assigned cells: {closure['assigned_cells']}")
    lines.append(f"- duplicated cells: {len(closure['duplicated_cells'])}")
    lines.append(f"- duplicated cell list: {closure['duplicated_cells']}")
    lines.append(f"- unassigned cells: {len(closure['unassigned_cells'])}")
    lines.append(f"- unassigned cell list: {closure['unassigned_cells']}")
    lines.append("")
    lines.append("2. Regional closure check")
    for region in REGION_ORDER:
        group_sum = float(group_summary_df.loc[group_summary_df["region"] == region, "delta_cf"].sum())
        sum42 = float(region_checks.loc[region_checks["region"] == region, "delta_sum42"].iloc[0])
        diff = group_sum - sum42
        lines.append(f"- {region}: sum_of_five_group_DeltaCF - sum_of_42class_DeltaCF = {diff:.12e}")
    lines.append("")
    lines.append("3. Event sample counts")
    n_el = sorted(region_checks["n_el_nino_months"].astype(int).unique().tolist())
    n_la = sorted(region_checks["n_la_nina_months"].astype(int).unique().tolist())
    lines.append(f"- number of El Nino months: {n_el}")
    lines.append(f"- number of La Nina months: {n_la}")
    lines.append(f"- four regions consistent: {len(n_el) == 1 and len(n_la) == 1}")
    lines.append("")
    lines.append("4. Low-cloud definition audit")
    old_low = sorted(old_groups["low_cloud"])
    new_low = sorted(new_groups["low_cloud"])
    lines.append(f"- old Figure03 low_cloud cells: {old_low}")
    lines.append(f"- new Figure03_degC05 low_cloud cells: {new_low}")
    lines.append(f"- old vs new low_cloud consistent: {old_low == new_low}")
    lines.append("- old Figure03 low_cloud definition came from low_thin + low_thick, which corresponds to CTP > 680 hPa (rows 800-680 and 1000-800).")
    lines.append("- requested alternative CTP > 560 hPa option was not used.")
    lines.append("")
    lines.append("5. Deep-convective definition audit")
    lines.append(f"- old Figure03 deep_convective cell count: {len(old_groups['deep_convective'])}")
    lines.append(f"- new Figure03_degC05 deep_convective cell count: {len(new_groups['deep_convective'])}")
    lines.append("- old Figure03 deep_convective used only tau 22.63-60.36.")
    lines.append("- new Figure03_degC05 deep_convective uses tau >= 22.63, so it includes both the 22.63-60.36 and 60.36-378.65 columns.")
    lines.append("")
    lines.append("6. Bootstrap settings")
    lines.append(f"- bootstrap samples: {args.bootstrap_samples}")
    lines.append(f"- block length (months): {args.block_length}")
    lines.append(f"- random seed: {args.seed}")
    lines.append("")
    lines.append("7. Cell-level significance summary")
    sig_counts = cell_boot_df.groupby("region")["significant_95"].sum()
    for region in REGION_ORDER:
        lines.append(f"- {region}: significant 42-class cells = {int(sig_counts.get(region, 0))}")
    return "\n".join(lines) + "\n"


def plot_main_figure(cf42_df: pd.DataFrame, boxes: dict[str, dict[str, float]], png_path: Path, pdf_path: Path) -> None:
    setup_style()
    vmax = float(np.nanmax(np.abs(cf42_df["delta_cf"].to_numpy(dtype=np.float64))))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(2, 2, figsize=(18.8, 9.2))
    image = None

    for idx, region in enumerate(REGION_ORDER):
        ax = axes.flat[idx]
        subset = cf42_df[cf42_df["region"] == region].copy()
        matrix = (
            subset.assign(
                ctp_bin=pd.Categorical(subset["ctp_bin"], categories=PRESS_ORDER, ordered=True),
                tau_bin=pd.Categorical(subset["tau_bin"], categories=OPT_ORDER, ordered=True),
            )
            .pivot(index="ctp_bin", columns="tau_bin", values="delta_cf")
            .reindex(index=PRESS_ORDER, columns=OPT_ORDER)
        )
        sig = (
            subset.assign(
                ctp_bin=pd.Categorical(subset["ctp_bin"], categories=PRESS_ORDER, ordered=True),
                tau_bin=pd.Categorical(subset["tau_bin"], categories=OPT_ORDER, ordered=True),
            )
            .pivot(index="ctp_bin", columns="tau_bin", values="significant_95")
            .reindex(index=PRESS_ORDER, columns=OPT_ORDER)
        )
        image = ax.imshow(
            matrix.to_numpy(dtype=float),
            cmap="RdBu_r",
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        yy, xx = np.where(sig.to_numpy(dtype=bool))
        if len(xx) > 0:
            ax.scatter(xx, yy, s=13, c="black", marker="o", linewidths=0, zorder=4)
        for group in GROUP_ORDER:
            box = boxes[group]
            rect = mpatches.Rectangle(
                (box["x"], box["y"]),
                box["width"],
                box["height"],
                fill=False,
                linewidth=0.95,
                linestyle="-",
                edgecolor="#666666",
                zorder=3,
            )
            ax.add_patch(rect)
        if region == "TP":
            for group in ["low_cloud", "mid_level"]:
                box = boxes[group]
                ax.text(
                    box["x"] + box["width"] + 0.08,
                    box["y"] + box["height"] / 2.0,
                    GROUP_LABELS[group],
                    fontsize=10,
                    color="black",
                    ha="left",
                    va="center",
                    fontweight="bold",
                )

            high_x0 = min(boxes[group]["x"] for group in ["thin_high", "thick_anvil", "deep_convective"])
            high_x1 = max(boxes[group]["x"] + boxes[group]["width"] for group in ["thin_high", "thick_anvil", "deep_convective"])
            high_y0 = min(boxes[group]["y"] for group in ["thin_high", "thick_anvil", "deep_convective"])
            high_y1 = max(boxes[group]["y"] + boxes[group]["height"] for group in ["thin_high", "thick_anvil", "deep_convective"])
            ax.text(
                high_x1 + 0.08,
                (high_y0 + high_y1) / 2.0,
                "High cloud",
                fontsize=10,
                color="black",
                ha="left",
                va="center",
                fontweight="bold",
            )

            high_label_positions = {
                "thin_high": {"text": "Thin high", "x": 0.5, "y": -0.45},
                "thick_anvil": {"text": "Thick anvil", "x": 2.5, "y": -0.45},
                "deep_convective": {"text": "Deep\nconvective", "x": 4.5, "y": -0.45},
            }
            for group, spec in high_label_positions.items():
                ax.text(
                    spec["x"],
                    spec["y"],
                    spec["text"],
                    fontsize=9,
                    color="black",
                    ha="center",
                    va="top",
                    fontweight="bold",
                    zorder=5,
                )

        ax.set_title(REGION_TITLE_MAP[region])
        panel_label(ax, PANEL_LABELS[region])
        ax.set_xlim(-0.5, len(OPT_ORDER) - 0.5)
        ax.set_ylim(len(PRESS_ORDER) - 0.5, -0.5)
        ax.set_xticks(np.arange(-0.5, len(OPT_ORDER) + 0.5, 1.0))
        ax.set_xticklabels([f"{v:g}" for v in X_EDGES], rotation=0)
        ax.set_yticks(np.arange(-0.5, len(PRESS_ORDER) + 0.5, 1.0))
        ax.set_yticklabels([f"{v:g}" for v in Y_EDGES])
        ax.grid(which="major", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", which="major", direction="in", length=4, width=0.8)
        ax.tick_params(axis="both", which="minor", direction="in", length=2, width=0.6)
        ax.tick_params(axis="x", pad=4)
        ax.tick_params(axis="y", pad=4)
        ax.set_xlabel("Optical Depth")
        if idx % 2 == 0:
            ax.set_ylabel("Cloud Top Pressure (hPa)")
        else:
            ax.set_ylabel("")

    fig.subplots_adjust(bottom=0.05, right=0.8, top=0.92, wspace=0.35, hspace=0.3)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92, pad=0.05)
    cbar.set_label("Cloud Fraction anomaly")
    cbar.ax.tick_params(axis="y", which="both", direction="in")
    # fig.text(0.5, 0.01, "Black dots mark 42-class cells whose 95% moving-block-bootstrap CI excludes zero.", ha="center", va="bottom", fontsize=8)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_group_check(group_boot_df: pd.DataFrame, out_path: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharey=True)
    x = np.arange(len(GROUP_ORDER))
    for idx, region in enumerate(REGION_ORDER):
        ax = axes.flat[idx]
        subset = group_boot_df[group_boot_df["region"] == region].set_index("cloud_group").loc[GROUP_ORDER].reset_index()
        vals = subset["delta_cf"].to_numpy(dtype=np.float64)
        lows = subset["ci_low"].to_numpy(dtype=np.float64)
        highs = subset["ci_high"].to_numpy(dtype=np.float64)
        yerr = np.vstack([vals - lows, highs - vals])
        ax.bar(x, vals, width=0.62, color="#6f9dbb", edgecolor="white", linewidth=0.6)
        ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor="#202020", elinewidth=0.8, capsize=2)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_xticks(x, [GROUP_LABELS[g] for g in GROUP_ORDER], rotation=25, ha="right")
        ax.set_title(f"({chr(97 + idx)}) {REGION_TITLE_MAP[region]}")
        ax.set_ylabel("Cloud Fraction anomaly")
    fig.suptitle("Figure03 degC05 cloud-group Delta CF check", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    bootstrap_samples = DEFAULT_BOOTSTRAP_SAMPLES
    block_length = DEFAULT_BLOCK_LENGTH
    seed = DEFAULT_RANDOM_SEED

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    png_path = FIG_DIR / "Figure03_cloudtype_occurrence_degC05_label_adjusted.png"
    pdf_path = FIG_DIR / "Figure03_cloudtype_occurrence_degC05_label_adjusted.pdf"
    cf42_path = RESULT_DIR / "Figure03_degC05_cloud_type_CF_anomaly_42class_label_adjusted.csv"
    group_summary_path = RESULT_DIR / "Figure03_degC05_cloud_group_CF_anomaly_summary_label_adjusted.csv"
    group_boot_path = RESULT_DIR / "Figure03_degC05_cloud_group_CF_bootstrap_label_adjusted.csv"
    validation_path = NOTES_DIR / "Figure03_degC05_validation_label_adjusted.txt"
    check_png_path = NOTES_DIR / "Figure03_degC05_cloud_group_summary_check_label_adjusted.png"

    regional_anom_path = INPUT_DIR / "ceres_monthly_regional_anomalies.nc"
    regional_clim_path = INPUT_DIR / "ceres_monthly_regional_climatology.nc"
    nino_path = INPUT_DIR / "nino34_200207_202302.csv"
    required_paths = [regional_anom_path, regional_clim_path, nino_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    require(not missing, "Missing required packaged input(s):\n" + "\n".join(missing))

    regional_anom = xr.open_dataset(regional_anom_path).load()
    regional_clim = xr.open_dataset(regional_clim_path).load()
    predictor_df = load_predictor(nino_path)
    axis_df = axis_frame_from_dataset(regional_anom)

    old_groups = build_old_group_members()
    new_groups = build_v2_group_members()
    closure = closure_check(new_groups)
    if closure["assigned_cells"] != 42 or closure["duplicated_cells"] or closure["unassigned_cells"]:
        raise RuntimeError(f"Five-group closure check failed: {closure}")

    monthly_cf_df = build_monthly_cf_actual(regional_anom, regional_clim, predictor_df)
    summary42_df, region_checks = build_42class_summary(monthly_cf_df)
    group_point_summary = build_group_point_summary_from_42class(summary42_df, new_groups)
    group_monthly_df = build_monthly_series(monthly_cf_df, new_groups, "group")
    cell_monthly_df = build_monthly_series(monthly_cf_df, new_groups, "cell")
    group_boot_df = bootstrap_delta_series(
        group_monthly_df,
        predictor_df,
        key_col="cloud_group",
        keys=GROUP_ORDER,
        bootstrap_samples=bootstrap_samples,
        block_length=block_length,
        seed=seed,
    )
    cell_boot_df = bootstrap_delta_series(
        cell_monthly_df,
        predictor_df,
        key_col="cloud_type",
        keys=list(range(1, 43)),
        bootstrap_samples=bootstrap_samples,
        block_length=block_length,
        seed=seed,
    )

    cf42_v2 = summary42_df.merge(
        cell_boot_df[["region", "cloud_type", "ci_low", "ci_high", "significant_95"]],
        on=["region", "cloud_type"],
        how="left",
        validate="one_to_one",
    )
    cf42_v2 = cf42_v2.rename(columns={"press_label": "ctp_bin", "opt_label": "tau_bin"})
    cf42_v2["region"] = pd.Categorical(cf42_v2["region"], categories=REGION_ORDER, ordered=True)
    cf42_v2 = cf42_v2.sort_values(["region", "press_index", "opt_index"]).reset_index(drop=True)

    group_summary = group_point_summary.copy()
    group_summary["region"] = pd.Categorical(group_summary["region"], categories=REGION_ORDER, ordered=True)
    group_summary["cloud_group"] = pd.Categorical(group_summary["cloud_group"], categories=GROUP_ORDER, ordered=True)
    group_summary = group_summary.sort_values(["region", "cloud_group"]).reset_index(drop=True)

    group_boot_out = group_boot_df.merge(group_point_summary, on=["region", "cloud_group"], how="left", suffixes=("_boot", ""))
    group_boot_out = group_boot_out.drop(columns=["delta_cf_boot"])
    group_boot_out["region"] = pd.Categorical(group_boot_out["region"], categories=REGION_ORDER, ordered=True)
    group_boot_out["cloud_group"] = pd.Categorical(group_boot_out["cloud_group"], categories=GROUP_ORDER, ordered=True)
    group_boot_out = group_boot_out.sort_values(["region", "cloud_group"]).reset_index(drop=True)

    boxes = build_group_boxes(new_groups, axis_df)
    plot_main_figure(cf42_v2, boxes, png_path, pdf_path)
    plot_group_check(group_boot_out, check_png_path)

    class ArgsProxy:
        bootstrap_samples = DEFAULT_BOOTSTRAP_SAMPLES
        block_length = DEFAULT_BLOCK_LENGTH
        seed = DEFAULT_RANDOM_SEED
    validation_text = build_validation_text(closure, region_checks, group_summary, cell_boot_df, old_groups, new_groups, ArgsProxy())
    validation_path.write_text(validation_text, encoding="utf-8")

    cf42_v2[["region", "ctp_bin", "tau_bin", "delta_cf", "ci_low", "ci_high", "significant_95"]].to_csv(cf42_path, index=False)
    group_summary.to_csv(group_summary_path, index=False)
    group_boot_out[["region", "cloud_group", "delta_cf", "ci_low", "ci_high", "significant_95", "n_el_nino", "n_la_nina"]].to_csv(group_boot_path, index=False)

    print("Figure03 degC05 generation complete.")
    print(f"Output directory: {PACKAGE_ROOT}")
    print(f"Bootstrap: n={bootstrap_samples}, block_length_months={block_length}, seed={seed}")
    print(f"ENSO months: El Nino={int((predictor_df[NINO_COLUMN] >= THRESHOLD).sum())}, La Nina={int((predictor_df[NINO_COLUMN] <= -THRESHOLD).sum())}")
    print(f"Mask closure: assigned={closure['assigned_cells']}, duplicated={len(closure['duplicated_cells'])}, unassigned={len(closure['unassigned_cells'])}")
    for region in REGION_ORDER:
        group_sum = float(group_summary.loc[group_summary["region"] == region, "delta_cf"].sum())
        sum42 = float(region_checks.loc[region_checks["region"] == region, "delta_sum42"].iloc[0])
        print(f"{region}: five_group_minus_sum42={group_sum - sum42:.12e}")
    print(f"Old low_cloud equals new low_cloud: {sorted(old_groups['low_cloud']) == sorted(new_groups['low_cloud'])}")
    print(f"Old deep_convective equals new deep_convective: {sorted(old_groups['deep_convective']) == sorted(new_groups['deep_convective'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
