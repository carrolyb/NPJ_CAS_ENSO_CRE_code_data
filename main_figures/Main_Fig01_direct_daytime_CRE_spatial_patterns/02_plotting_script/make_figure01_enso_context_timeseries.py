#!/usr/bin/env python3
"""Plot the ENSO context time series used by Figure 01 under a monthly Nino3.4 +/-0.5 C definition."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/Volumes/My Book/P3")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NINO34_CSV = ROOT / "data_processed/anomalies/nino34_200207_202302.csv"
SAMPLES_CSV = PACKAGE_ROOT / "03_input_data" / "enso_month_samples.csv"
OUT_DIR = PACKAGE_ROOT / "05_notes"
OUT_PNG = OUT_DIR / "Figure01_ENSO_context_timeseries_degC05.png"
OUT_PDF = OUT_DIR / "Figure01_ENSO_context_timeseries_degC05.pdf"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 320,
        }
    )


def main() -> int:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nino = pd.read_csv(NINO34_CSV, parse_dates=["date"]).sort_values("date")
    samples = pd.read_csv(SAMPLES_CSV, parse_dates=["date"]).sort_values("date")

    sample_phase = samples[["date", "phase"]].drop_duplicates()
    df = nino.merge(sample_phase, on="date", how="left")

    fig, ax = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)

    # Highlight composite months directly on the background.
    ax.fill_between(
        df["date"],
        -2.2,
        2.2,
        where=df["phase"].eq("El Nino"),
        color="#d73027",
        alpha=0.10,
        interpolate=True,
        label="Figure 01 El Niño months (+0.5 C)",
    )
    ax.fill_between(
        df["date"],
        -2.2,
        2.2,
        where=df["phase"].eq("La Nina"),
        color="#4575b4",
        alpha=0.10,
        interpolate=True,
        label="Figure 01 La Niña months (-0.5 C)",
    )

    ax.plot(
        df["date"],
        df["nino34_anom"],
        color="#7a7a7a",
        lw=1.2,
        alpha=0.95,
        label="Monthly Niño 3.4 SST anomaly",
        zorder=2,
    )
    ax.plot(
        df["date"],
        df["oni_3mon"],
        color="#111111",
        lw=1.6,
        label="Centered 3-month mean",
        zorder=3,
    )

    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.axhline(0.5, color="#999999", lw=0.8, ls="--")
    ax.axhline(-0.5, color="#999999", lw=0.8, ls="--")

    ax.set_ylim(-2.2, 2.8)
    ax.set_xlim(df["date"].min(), df["date"].max())
    ax.set_ylabel("Niño 3.4 SST anomaly (°C)")
    ax.set_title("ENSO context for Figure 01", pad=12)
    ax.text(
        0.0,
        1.005,
        "Monthly Niño 3.4 anomalies during July 2002 to February 2023; composite months are highlighted using the monthly +/-0.5 C rule.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color="#d0d0d0", lw=0.6, ls=":")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=8, handlelength=2.6)

    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
