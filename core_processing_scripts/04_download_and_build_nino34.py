#!/usr/bin/env python3
"""Download NOAA CPC ERSSTv5 monthly Nino indices and build Nino3.4 products."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from enso_cloud.config import DEFAULT_NINO_URL
from enso_cloud.enso import (
    add_standardized_indices,
    clip_period,
    download_ascii,
    event_counts,
    ordered_columns,
    parse_ascii_table,
)
from enso_cloud.io_utils import ensure_parent, prepare_figure_path, write_csv
DEFAULT_URL = DEFAULT_NINO_URL
DEFAULT_RAW_OUT = Path("data_raw/enso/ersst5.nino.mth.91-20.ascii")
DEFAULT_CSV_OUT = Path("data_processed/anomalies/nino34_monthly.csv")
DEFAULT_START_DATE = "2002-07-15"
DEFAULT_END_DATE = "2023-02-15"
DEFAULT_COUNTS_OUT = Path("results/tables/enso_event_counts.csv")
DEFAULT_FIGURE_OUT = Path("figures/exploratory/nino34_timeseries.png")
DEFAULT_STUDY_FIGURE_OUT = Path("figures/exploratory/nino34_timeseries_study_period.png")
class PlotMargins:
    left = 80
    right = 36
    top = 52
    bottom = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="NOAA CPC ERSSTv5 monthly Nino indices URL.")
    parser.add_argument("--raw-out", type=Path, default=DEFAULT_RAW_OUT, help="Raw ASCII output path.")
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT, help="Processed monthly CSV output path.")
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Research-period clip start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="Research-period clip end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload raw file and overwrite downstream outputs.",
    )
    return parser.parse_args()

def clip_output_path(csv_out: Path, start_date: str, end_date: str) -> Path:
    start_tag = pd.Timestamp(start_date).strftime("%Y%m")
    end_tag = pd.Timestamp(end_date).strftime("%Y%m")
    return csv_out.parent / f"nino34_{start_tag}_{end_tag}.csv"


def format_tick_date(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m")


def pick_tick_positions(dates: pd.Series, target_ticks: int = 7) -> list[int]:
    if dates.empty:
        return []
    if len(dates) <= target_ticks:
        return list(range(len(dates)))
    positions = np.linspace(0, len(dates) - 1, num=target_ticks)
    out = sorted({int(round(pos)) for pos in positions})
    return out


def compute_y_limits(*series_list: Iterable[float]) -> tuple[float, float]:
    values: list[float] = []
    for series in series_list:
        for value in series:
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append(numeric)
    values.extend([-0.5, 0.5])
    if not values:
        return -2.0, 2.0
    ymin = min(values)
    ymax = max(values)
    span = max(ymax - ymin, 1.0)
    pad = span * 0.15
    return ymin - pad, ymax + pad


def value_to_y(value: float, ymin: float, ymax: float, height: int, margins: PlotMargins) -> int:
    if ymax == ymin:
        return margins.top + (height - margins.top - margins.bottom) // 2
    usable = height - margins.top - margins.bottom
    frac = (value - ymin) / (ymax - ymin)
    return int(round(height - margins.bottom - frac * usable))


def index_to_x(index: int, count: int, width: int, margins: PlotMargins) -> int:
    usable = width - margins.left - margins.right
    if count <= 1:
        return margins.left + usable // 2
    return int(round(margins.left + index * usable / (count - 1)))


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    x_values: list[int],
    y_values: list[float],
    ymin: float,
    ymax: float,
    width: int,
    height: int,
    margins: PlotMargins,
    color: tuple[int, int, int],
    line_width: int,
) -> None:
    points: list[tuple[int, int]] = []
    for x, y in zip(x_values, y_values):
        if not math.isfinite(float(y)):
            if len(points) >= 2:
                draw.line(points, fill=color, width=line_width)
            points = []
            continue
        points.append((x, value_to_y(float(y), ymin, ymax, height, margins)))
    if len(points) >= 2:
        draw.line(points, fill=color, width=line_width)


def date_to_x(date_value: pd.Timestamp, dates: pd.Series, x_positions: list[int]) -> int:
    if len(dates) == 1:
        return x_positions[0]
    numeric_dates = dates.astype("int64").to_numpy()
    target = np.int64(date_value.value)
    return int(round(np.interp(target, numeric_dates, np.asarray(x_positions, dtype=float))))


def make_plot(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    out_path: Path,
    overwrite: bool,
    x_min: str | None = None,
    x_max: str | None = None,
) -> None:
    width, height = 1400, 800
    margins = PlotMargins()
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    dates = pd.to_datetime(df["date"])
    if x_min is not None:
        dates = dates[dates >= pd.Timestamp(x_min)]
    if x_max is not None:
        dates = dates[dates <= pd.Timestamp(x_max)]
    plot_df = df.loc[dates.index].reset_index(drop=True)
    dates = pd.to_datetime(plot_df["date"])

    if plot_df.empty:
        raise RuntimeError(f"No records available for plotting range: {x_min} to {x_max}")

    monthly = plot_df["nino34_anom"].astype(float).tolist()
    oni = plot_df["oni_3mon"].astype(float).tolist()
    ymin, ymax = compute_y_limits(monthly, oni)
    x_positions = [index_to_x(i, len(plot_df), width, margins) for i in range(len(plot_df))]

    plot_left = margins.left
    plot_top = margins.top
    plot_right = width - margins.right
    plot_bottom = height - margins.bottom

    shade_start = pd.Timestamp(start_date).replace(day=1)
    shade_end = pd.Timestamp(end_date) + pd.offsets.MonthEnd(0)
    research_mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
    shade_visible_start = max(shade_start, dates.iloc[0])
    shade_visible_end = min(shade_end, dates.iloc[-1] + pd.offsets.MonthEnd(0))
    if research_mask.any() and shade_visible_start <= shade_visible_end:
        shade_x0 = date_to_x(shade_visible_start, dates, x_positions)
        shade_x1 = date_to_x(shade_visible_end, dates, x_positions)
        shade_box = [
            shade_x0,
            plot_top,
            shade_x1,
            plot_bottom,
        ]
        draw.rectangle(shade_box, fill=(240, 240, 210))

    for threshold in (-0.5, 0.0, 0.5):
        y = value_to_y(threshold, ymin, ymax, height, margins)
        color = (170, 170, 170) if threshold else (110, 110, 110)
        draw.line([(plot_left, y), (plot_right, y)], fill=color, width=1)
        if threshold > 0:
            label = "+0.5 degC"
        elif threshold < 0:
            label = "-0.5 degC"
        else:
            label = "0.0"
        draw.text((10, y - 7), label, fill=(80, 80, 80), font=font)

    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline=(0, 0, 0), width=1)

    draw_polyline(
        draw,
        x_positions,
        monthly,
        ymin,
        ymax,
        width,
        height,
        margins,
        color=(31, 119, 180),
        line_width=2,
    )
    draw_polyline(
        draw,
        x_positions,
        oni,
        ymin,
        ymax,
        width,
        height,
        margins,
        color=(214, 39, 40),
        line_width=3,
    )

    tick_positions = pick_tick_positions(dates)
    for idx in tick_positions:
        x = x_positions[idx]
        draw.line([(x, plot_bottom), (x, plot_bottom + 5)], fill=(0, 0, 0), width=1)
        label = format_tick_date(dates.iloc[idx])
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0]
        draw.text((x - label_width // 2, plot_bottom + 10), label, fill=(0, 0, 0), font=font)

    for frac in np.linspace(0.0, 1.0, num=6):
        value = ymin + frac * (ymax - ymin)
        y = value_to_y(value, ymin, ymax, height, margins)
        draw.line([(plot_left - 5, y), (plot_left, y)], fill=(0, 0, 0), width=1)
        draw.text((20, y - 7), f"{value:.1f}", fill=(0, 0, 0), font=font)

    title = "Nino 3.4 anomaly and centered 3-month running mean"
    subtitle = f"NOAA CPC ERSSTv5 monthly Nino indices | Study period shaded: {start_date[:7]} to {end_date[:7]}"
    draw.text((plot_left + 120, 16), title, fill=(0, 0, 0), font=font)
    draw.text((plot_left + 120, 32), subtitle, fill=(70, 70, 70), font=font)
    ylabel = "SST anomaly (degC)"
    draw.text((12, 16), ylabel, fill=(0, 0, 0), font=font)

    legend_y = plot_top + 12
    draw.line([(plot_right - 310, legend_y), (plot_right - 270, legend_y)], fill=(31, 119, 180), width=2)
    draw.text((plot_right - 262, legend_y - 8), "Monthly Nino 3.4 anomaly", fill=(0, 0, 0), font=font)
    draw.line([(plot_right - 310, legend_y + 22), (plot_right - 270, legend_y + 22)], fill=(214, 39, 40), width=3)
    draw.text((plot_right - 262, legend_y + 14), "Centered 3-month running mean", fill=(0, 0, 0), font=font)
    draw.rectangle([plot_right - 310, legend_y + 36, plot_right - 270, legend_y + 52], fill=(240, 240, 210), outline=(200, 200, 170))
    draw.text((plot_right - 262, legend_y + 38), "Study period: 2002-07 to 2023-02", fill=(0, 0, 0), font=font)

    prepare_figure_path(out_path, overwrite)
    image.save(out_path)
    print(f"Wrote figure: {out_path}")


def main() -> int:
    args = parse_args()
    download_ascii(args.url, args.raw_out, args.overwrite)

    monthly = parse_ascii_table(args.raw_out)
    monthly = add_standardized_indices(monthly)
    write_csv(ordered_columns(monthly), args.csv_out, args.overwrite)

    clipped = clip_period(monthly, args.start_date, args.end_date)
    clipped_out = clip_output_path(args.csv_out, args.start_date, args.end_date)
    write_csv(ordered_columns(clipped), clipped_out, args.overwrite)
    write_csv(event_counts(clipped), DEFAULT_COUNTS_OUT, args.overwrite)
    make_plot(monthly, args.start_date, args.end_date, DEFAULT_FIGURE_OUT, args.overwrite)
    make_plot(
        monthly,
        args.start_date,
        args.end_date,
        DEFAULT_STUDY_FIGURE_OUT,
        args.overwrite,
        x_min=args.start_date,
        x_max=args.end_date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
