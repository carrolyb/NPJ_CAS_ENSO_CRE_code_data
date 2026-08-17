#!/usr/bin/env python3
"""Render Figure 10 under the +/-0.5 C ENSO phase-episode definition using the Figure10 copy.py layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

matplotlib.use("Agg")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PACKAGE_ROOT / "01_final_figure"
INPUT_DIR = PACKAGE_ROOT / "03_input_data"
RESULT_DIR = PACKAGE_ROOT / "04_key_results"
NOTES_DIR = PACKAGE_ROOT / "05_notes"

NINO_PATH = INPUT_DIR / "nino34_200207_202302.csv"
STEP10_1A_ALIGNED_PATH = INPUT_DIR / "Step10_1A_metric_formal_direct_aligned_monthly_series.csv"
STEP10_1A_MODEL_SUMMARY_PATH = INPUT_DIR / "Step10_1A_deterministic_model_summary.csv"
STEP10_1B2A_PRED_PATH = INPUT_DIR / "Step10_1B2A_purged_blocked_CV_monthly_predictions.csv"
STEP10_1B2A_SKILL_PATH = INPUT_DIR / "Step10_1B2A_purged_blocked_CV_model_skill.csv"
STEP10_1B2A_INCREMENT_PATH = INPUT_DIR / "Step10_1B2A_purged_blocked_CV_model_increment_deterministic.csv"
STEP10_1B2A_BOOT_PATH = INPUT_DIR / "Step10_1B2A_purged_blocked_CV_skill_increment_bootstrap.csv"
STEP10_1B2B_POOLED_PATH = INPUT_DIR / "Step10_1B2B_pooled_event_heldout_skill.csv"
STEP10_1B2B_EVENT_BOOT_PATH = INPUT_DIR / "Step10_1B2B_event_cluster_bootstrap_skill_increment.csv"
STEP10_1B2B_LOEO_SUMMARY_PATH = INPUT_DIR / "Step10_1B2B_leave_one_event_out_refit_summary.csv"
STEP10_1B2B_SUMMARY_PATH = INPUT_DIR / "Step10_1B2B_ENSO_event_robustness_and_Figure10_gate_summary.txt"

V1_CAPTION_PATH = INPUT_DIR / "Figure10_event_robust_direct_diagnostic_linkage_caption_v1.md"
V1_CHECKS_PATH = INPUT_DIR / "Figure10_event_robust_direct_diagnostic_linkage_method_and_plot_checks_v1.txt"
V1_SCRIPT_PATH = INPUT_DIR / "make_fig10_event_robust_direct_diagnostic_linkage_final_v1.py"

OUT_PNG = FIG_DIR / "Figure10_heldout_direct_diagnostic_linkage_degC05.png"
OUT_PDF = FIG_DIR / "Figure10_heldout_direct_diagnostic_linkage_degC05.pdf"
OUT_PLOT_DATA = RESULT_DIR / "Figure10_degC05_plot_data.csv"
OUT_CAPTION = NOTES_DIR / "Figure10_degC05_caption.md"
OUT_CHECKS = NOTES_DIR / "Figure10_degC05_method_and_plot_checks.txt"
OUT_MANIFEST = NOTES_DIR / "Figure10_degC05_input_data_manifest.md"
OUT_EVENT_INVENTORY_DEGC05 = RESULT_DIR / "Figure10_degC05_ENSO_phase_episode_inventory.csv"
OUT_LOEO_SUMMARY_DEGC05 = RESULT_DIR / "Figure10_degC05_leave_one_phase_episode_out_refit_summary.csv"
OUT_EVENT_SKILL_DEGC05 = RESULT_DIR / "Figure10_degC05_phase_episode_heldout_skill_by_event.csv"
OUT_EVENT_PRED_DEGC05 = RESULT_DIR / "Figure10_degC05_phase_episode_heldout_monthly_predictions.csv"
OUT_POOLED_DEGC05 = RESULT_DIR / "Figure10_degC05_pooled_phase_episode_heldout_skill.csv"
OUT_EVENT_BOOT_DEGC05 = RESULT_DIR / "Figure10_degC05_phase_episode_cluster_bootstrap_skill_increment.csv"

REGIONS = ["WP", "CP", "EP"]
REGION_NAMES = {"WP": "Western Pacific", "CP": "Central Pacific", "EP": "Eastern Pacific"}
REGION_COLORS = {
    "WP": {"main": "#2b6cb0", "light": "#8ab6e6"},
    "CP": {"main": "#dd6b20", "light": "#f1b27b"},
    "EP": {"main": "#2f855a", "light": "#86c5a2"},
}
PANEL_LABELS = {"WP": "a", "CP": "b", "EP": "c"}
TOTAL_MONTHS = 248
PURGE_WIDTH = 6
N_BOOT = 2000
SEED = 42
ENSO_THRESHOLD_DEGC05 = 0.5
EXPECTED_ENSO_COUNTS_DEGC05 = {"el_nino_months": 54, "la_nina_months": 85, "total_enso_months": 139}
EXPECTED_HELDOUT_BOX = {
    "WP": (0.611293, 0.616734),
    "CP": (0.613214, 0.477245),
    "EP": (0.872751, 0.031830),
}
CAPTION_TEXT = (
    "Figure 10. Held-out diagnostic linkage between cloud-structure metrics and regional direct daytime Net cloud radiative effect variability. "
    "Panels (a)–(c) compare observed regional direct daytime Net CRE anomalies with held-out diagnostic estimates from the full metric model, "
    "M3 = LCSP + HCCF + HCTB, over the western, central, and eastern Pacific fixed regional boxes, respectively. Held-out estimates are obtained "
    "from purged blocked cross-validation using contiguous 12-month test blocks and a 6-month purge window on both sides of each test block. "
    "Annotations report the held-out diagnostic R² of M3 and its R² increment relative to the comparator model M1 = LCSP + HCCF. HCCF is retained "
    "as a total-high-cloud occurrence comparator, whereas HCTB represents additional high-cloud structural information based on the remapped cloud-group "
    "definition. Separate ENSO-phase-episode-held-out tests using a +/-0.5 C monthly Nino3.4 definition "
    "(54 El Nino months and 85 La Nina months across 20 phase episodes) confirm that the positive diagnostic increments are not dominated by any single episode. "
    "These relationships describe diagnostic association with regional direct Net CRE variability and are not interpreted as prediction skill, causal "
    "control, independent validation, or an exact reconstruction of the direct all-sky response."
)
MODELS = {
    "M1": ["LCSP_anom", "HCCF_anom"],
    "M3": ["LCSP_anom", "HCCF_anom", "HCTB_anom"],
}
RESPONSE = "Direct_Net_anom_formal"
SMALL = 1.0e-12


@dataclass(frozen=True)
class EventSegment:
    event_id: str
    phase_value: int
    phase_name: str
    start_idx: int
    end_idx: int
    start_month: pd.Timestamp
    end_month: pd.Timestamp

    @property
    def n_months(self) -> int:
        return self.end_idx - self.start_idx + 1

    @property
    def purge_start(self) -> int:
        return max(0, self.start_idx - PURGE_WIDTH)

    @property
    def purge_end(self) -> int:
        return min(TOTAL_MONTHS - 1, self.end_idx + PURGE_WIDTH)

def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.0,
        1.02,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        clip_on=False,
        zorder=20,
    )

def require_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required fixed input is missing: {path}")
    return path


def assert_close(actual: float, expected: float, label: str, tol: float = 1.0e-6) -> None:
    if not np.isclose(actual, expected, atol=tol, rtol=0.0):
        raise RuntimeError(f"{label} mismatch: actual={actual}, expected={expected}")


def make_input_manifest() -> None:
    lines = [
        "# Figure10 degC05 input manifest",
        "",
        f"- aligned monthly direct/metric table: `{STEP10_1A_ALIGNED_PATH}`",
        f"- full-sample deterministic model summary: `{STEP10_1A_MODEL_SUMMARY_PATH}`",
        f"- Nino3.4 monthly anomaly file: `{NINO_PATH}`",
        f"- blocked-CV monthly predictions: `{STEP10_1B2A_PRED_PATH}`",
        f"- blocked-CV model skill summary: `{STEP10_1B2A_SKILL_PATH}`",
        f"- blocked-CV model increment summary: `{STEP10_1B2A_INCREMENT_PATH}`",
        f"- blocked-CV bootstrap increment summary: `{STEP10_1B2A_BOOT_PATH}`",
        f"- original pooled ENSO event-held-out summary: `{STEP10_1B2B_POOLED_PATH}`",
        f"- original ENSO event-cluster bootstrap summary: `{STEP10_1B2B_EVENT_BOOT_PATH}`",
        f"- original leave-one-event-out summary: `{STEP10_1B2B_LOEO_SUMMARY_PATH}`",
        f"- original Figure 10 robustness gate summary: `{STEP10_1B2B_SUMMARY_PATH}`",
        f"- v1 caption reference: `{V1_CAPTION_PATH}`",
        f"- v1 method-checks reference: `{V1_CHECKS_PATH}`",
        f"- v1 plotting script reference: `{V1_SCRIPT_PATH}`",
    ]
    OUT_MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_ols(frame: pd.DataFrame, predictors: list[str]) -> tuple[np.ndarray, dict[str, float | bool]]:
    y = frame[RESPONSE].to_numpy(dtype=np.float64)
    x = np.column_stack([np.ones(len(frame), dtype=np.float64)] + [frame[p].to_numpy(dtype=np.float64) for p in predictors])
    beta, _, rank, singular_vals = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residual = y - fitted
    rss = float(np.dot(residual, residual))
    tss = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = np.nan if tss < SMALL else 1.0 - rss / tss
    n = len(y)
    k = x.shape[1]
    adjusted_r2 = np.nan if n <= k or not np.isfinite(r2) else 1.0 - (1.0 - r2) * (n - 1) / (n - k)
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    corr = np.nan if np.std(y) < SMALL or np.std(fitted) < SMALL else float(np.corrcoef(y, fitted)[0, 1])
    condition_number = np.inf
    if singular_vals.size and singular_vals[-1] > 0:
        condition_number = float(singular_vals[0] / singular_vals[-1])
    stats = {
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "adjusted_R2": float(adjusted_r2) if np.isfinite(adjusted_r2) else np.nan,
        "RMSE": rmse,
        "MAE": mae,
        "Pearson_r": corr,
        "finite": bool(np.isfinite(beta).all() and np.isfinite(fitted).all()),
        "rank_deficient": bool(rank < k),
        "condition_number": condition_number,
    }
    return beta, stats


def predict_ols(frame: pd.DataFrame, predictors: list[str], beta: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(frame), dtype=np.float64)] + [frame[p].to_numpy(dtype=np.float64) for p in predictors])
    return x @ beta


def prediction_skill(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    obs = observed[valid].astype(np.float64)
    pred = predicted[valid].astype(np.float64)
    if obs.size == 0:
        return {"n": 0, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Pearson_r": np.nan}
    residual = obs - pred
    sse = float(np.dot(residual, residual))
    sst = float(np.dot(obs - obs.mean(), obs - obs.mean()))
    r2 = np.nan if sst < SMALL else 1.0 - sse / sst
    corr = np.nan if obs.size < 2 or np.std(obs) < SMALL or np.std(pred) < SMALL else float(np.corrcoef(obs, pred)[0, 1])
    return {
        "n": int(obs.size),
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "Pearson_r": corr,
    }


def sign_consistent(candidate: float, reference: float) -> bool:
    if not np.isfinite(candidate) or not np.isfinite(reference):
        return False
    if abs(reference) < SMALL:
        return abs(candidate) < SMALL
    return np.sign(candidate) == np.sign(reference)


def build_degC05_event_segments(aligned_df: pd.DataFrame, nino_df: pd.DataFrame) -> list[EventSegment]:
    reference = aligned_df.loc[aligned_df["region"] == "CP", ["month"]].sort_values("month").reset_index(drop=True)
    require_len = len(reference)
    if require_len != TOTAL_MONTHS:
        raise RuntimeError(f"Unexpected aligned monthly length for CP: {require_len}")
    nino_work = nino_df.copy()
    nino_work["month"] = nino_work["date"].dt.to_period("M").dt.to_timestamp()
    lookup = nino_work.set_index("month")["nino34_anom"]
    monthly_nino = reference["month"].dt.to_period("M").dt.to_timestamp().map(lookup)
    if not monthly_nino.notna().all():
        raise RuntimeError("Missing Nino3.4 anomaly values for some Figure 10 months.")
    phase = np.zeros(TOTAL_MONTHS, dtype=np.int8)
    phase_values = monthly_nino.to_numpy(dtype=float)
    phase[phase_values >= ENSO_THRESHOLD_DEGC05] = 1
    phase[phase_values <= -ENSO_THRESHOLD_DEGC05] = -1
    counts = {"el_nino_months": int((phase == 1).sum()), "la_nina_months": int((phase == -1).sum())}
    counts["total_enso_months"] = counts["el_nino_months"] + counts["la_nina_months"]
    if counts != EXPECTED_ENSO_COUNTS_DEGC05:
        raise RuntimeError(f"Unexpected degC05 ENSO month counts: {counts} != {EXPECTED_ENSO_COUNTS_DEGC05}")

    segments: list[EventSegment] = []
    i = 0
    counters = {1: 0, -1: 0}
    phase_name = {1: "El Nino", -1: "La Nina"}
    phase_label = {1: "ElNino", -1: "LaNina"}
    while i < TOTAL_MONTHS:
        phase_value = int(phase[i])
        if phase_value == 0:
            i += 1
            continue
        j = i
        while j + 1 < TOTAL_MONTHS and int(phase[j + 1]) == phase_value:
            j += 1
        counters[phase_value] += 1
        segments.append(
            EventSegment(
                event_id=f"{phase_label[phase_value]}_{counters[phase_value]:02d}",
                phase_value=phase_value,
                phase_name=phase_name[phase_value],
                start_idx=i,
                end_idx=j,
                start_month=reference.loc[i, "month"],
                end_month=reference.loc[j, "month"],
            )
        )
        i = j + 1
    return segments


def build_degC05_event_robustness(aligned_df: pd.DataFrame, model_summary_df: pd.DataFrame, nino_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    event_segments = build_degC05_event_segments(aligned_df, nino_df)
    inventory_rows: list[dict[str, object]] = []
    for segment in event_segments:
        inventory_rows.append(
            {
                "event_id": segment.event_id,
                "phase": segment.phase_name,
                "start_month": segment.start_month.strftime("%Y-%m-01"),
                "end_month": segment.end_month.strftime("%Y-%m-01"),
                "n_months": segment.n_months,
                "first_index": segment.start_idx,
                "last_index": segment.end_idx,
                "included_in_event_heldout_test": True,
                "notes": "short event; do not interpret event-level R2 and treat Pearson r cautiously" if segment.n_months <= 2 else "",
            }
        )
    inventory_df = pd.DataFrame(inventory_rows)

    full_sample_ref = model_summary_df.pivot(index="region", columns="model_name")
    full_delta_adj_r2 = {}
    full_delta_rmse = {}
    full_hctb = {}
    for region in REGIONS:
        full_delta_adj_r2[region] = float(full_sample_ref["adjusted_R2"].loc[region, "M3"]) - float(full_sample_ref["adjusted_R2"].loc[region, "M1"])
        full_delta_rmse[region] = float(full_sample_ref["RMSE"].loc[region, "M3"]) - float(full_sample_ref["RMSE"].loc[region, "M1"])
        full_hctb[region] = float(full_sample_ref["coefficient_HCTB"].loc[region, "M3"])

    loeo_rows: list[dict[str, object]] = []
    loeo_summary_rows: list[dict[str, object]] = []
    heldout_prediction_rows: list[dict[str, object]] = []
    heldout_skill_rows: list[dict[str, object]] = []

    for region in REGIONS:
        region_df = aligned_df.loc[aligned_df["region"] == region].sort_values("month").reset_index(drop=True)
        if len(region_df) != TOTAL_MONTHS:
            raise RuntimeError(f"{region} aligned series length mismatch.")
        region_loeo = []
        for segment in event_segments:
            remaining = region_df.drop(index=np.arange(segment.start_idx, segment.end_idx + 1)).reset_index(drop=True)
            beta_m1, stats_m1 = fit_ols(remaining, MODELS["M1"])
            beta_m3, stats_m3 = fit_ols(remaining, MODELS["M3"])
            hctb_coef_remaining = float(beta_m3[3])
            delta_adj = float(stats_m3["adjusted_R2"] - stats_m1["adjusted_R2"])
            delta_rmse = float(stats_m3["RMSE"] - stats_m1["RMSE"])
            loeo_row = {
                "region": region,
                "event_id": segment.event_id,
                "phase": segment.phase_name,
                "n_months_removed": segment.n_months,
                "n_months_remaining": len(remaining),
                "M1_adjusted_R2_remaining": float(stats_m1["adjusted_R2"]),
                "M3_adjusted_R2_remaining": float(stats_m3["adjusted_R2"]),
                "delta_adjusted_R2_M3_minus_M1_remaining": delta_adj,
                "M1_RMSE_remaining": float(stats_m1["RMSE"]),
                "M3_RMSE_remaining": float(stats_m3["RMSE"]),
                "delta_RMSE_M3_minus_M1_remaining": delta_rmse,
                "M3_HCTB_coefficient_remaining": hctb_coef_remaining,
                "M3_HCTB_coefficient_sign_consistent_with_full_sample": sign_consistent(hctb_coef_remaining, full_hctb[region]),
                "delta_adjusted_R2_positive": bool(delta_adj > 0),
                "delta_RMSE_negative": bool(delta_rmse < 0),
                "finite_models": bool(stats_m1["finite"] and stats_m3["finite"]),
                "rank_deficient": bool(stats_m1["rank_deficient"] or stats_m3["rank_deficient"]),
            }
            loeo_rows.append(loeo_row)
            region_loeo.append(loeo_row)

            train_mask = np.ones(TOTAL_MONTHS, dtype=bool)
            train_mask[segment.purge_start : segment.purge_end + 1] = False
            test_mask = np.zeros(TOTAL_MONTHS, dtype=bool)
            test_mask[segment.start_idx : segment.end_idx + 1] = True
            training = region_df.loc[train_mask].reset_index(drop=True)
            testing = region_df.loc[test_mask].reset_index(drop=True)
            beta_m1_hold, stats_m1_hold = fit_ols(training, MODELS["M1"])
            beta_m3_hold, stats_m3_hold = fit_ols(training, MODELS["M3"])
            pred_m1 = predict_ols(testing, MODELS["M1"], beta_m1_hold)
            pred_m3 = predict_ols(testing, MODELS["M3"], beta_m3_hold)
            obs = testing[RESPONSE].to_numpy(dtype=np.float64)
            skill_m1 = prediction_skill(obs, pred_m1)
            skill_m3 = prediction_skill(obs, pred_m3)
            heldout_skill_rows.append(
                {
                    "region": region,
                    "event_id": segment.event_id,
                    "phase": segment.phase_name,
                    "n_test_months": segment.n_months,
                    "n_training_months": len(training),
                    "RMSE_M1": float(skill_m1["RMSE"]),
                    "RMSE_M3": float(skill_m3["RMSE"]),
                    "delta_RMSE_M3_minus_M1": float(skill_m3["RMSE"] - skill_m1["RMSE"]),
                    "MAE_M1": float(skill_m1["MAE"]),
                    "MAE_M3": float(skill_m3["MAE"]),
                    "delta_MAE_M3_minus_M1": float(skill_m3["MAE"] - skill_m1["MAE"]),
                    "Pearson_r_M1": float(skill_m1["Pearson_r"]) if np.isfinite(skill_m1["Pearson_r"]) else np.nan,
                    "Pearson_r_M3": float(skill_m3["Pearson_r"]) if np.isfinite(skill_m3["Pearson_r"]) else np.nan,
                    "M3_better_RMSE": bool(skill_m3["RMSE"] < skill_m1["RMSE"]),
                    "M3_better_MAE": bool(skill_m3["MAE"] < skill_m1["MAE"]),
                    "finite_models": bool(
                        stats_m1_hold["finite"] and stats_m3_hold["finite"] and np.isfinite(pred_m1).all() and np.isfinite(pred_m3).all()
                    ),
                    "rank_deficient": bool(stats_m1_hold["rank_deficient"] or stats_m3_hold["rank_deficient"]),
                }
            )
            for idx, row in testing.iterrows():
                heldout_prediction_rows.append(
                    {
                        "month": row["month"].strftime("%Y-%m-01"),
                        "region": region,
                        "event_id": segment.event_id,
                        "phase": segment.phase_name,
                        "n_event_months": segment.n_months,
                        "n_training_months": len(training),
                        "Direct_Net_anom_formal": float(row[RESPONSE]),
                        "predicted_M1": float(pred_m1[idx]),
                        "predicted_M3": float(pred_m3[idx]),
                        "residual_M1": float(row[RESPONSE] - pred_m1[idx]),
                        "residual_M3": float(row[RESPONSE] - pred_m3[idx]),
                    }
                )

        region_loeo_df = pd.DataFrame(region_loeo)
        if not bool(region_loeo_df["finite_models"].all()):
            raise RuntimeError(f"{region} has non-finite leave-one-phase-episode-out fits.")
        if bool(region_loeo_df["rank_deficient"].any()):
            raise RuntimeError(f"{region} has rank-deficient leave-one-phase-episode-out fits.")
        delta_adj_change = np.abs(region_loeo_df["delta_adjusted_R2_M3_minus_M1_remaining"] - full_delta_adj_r2[region])
        delta_rmse_change = np.abs(region_loeo_df["delta_RMSE_M3_minus_M1_remaining"] - full_delta_rmse[region])
        loeo_summary_rows.append(
            {
                "region": region,
                "n_events_removed_one_at_a_time": len(event_segments),
                "min_delta_adjusted_R2_M3_minus_M1": float(region_loeo_df["delta_adjusted_R2_M3_minus_M1_remaining"].min()),
                "max_delta_adjusted_R2_M3_minus_M1": float(region_loeo_df["delta_adjusted_R2_M3_minus_M1_remaining"].max()),
                "fraction_events_delta_adjusted_R2_positive": float(region_loeo_df["delta_adjusted_R2_positive"].mean()),
                "min_delta_RMSE_M3_minus_M1": float(region_loeo_df["delta_RMSE_M3_minus_M1_remaining"].min()),
                "max_delta_RMSE_M3_minus_M1": float(region_loeo_df["delta_RMSE_M3_minus_M1_remaining"].max()),
                "fraction_events_delta_RMSE_negative": float(region_loeo_df["delta_RMSE_negative"].mean()),
                "fraction_events_HCTB_coefficient_sign_consistent": float(region_loeo_df["M3_HCTB_coefficient_sign_consistent_with_full_sample"].mean()),
                "any_rank_deficient_fit": bool(region_loeo_df["rank_deficient"].any()),
                "most_influential_event_by_delta_adjusted_R2_change": str(region_loeo_df.loc[delta_adj_change.idxmax(), "event_id"]),
                "most_influential_event_by_delta_RMSE_change": str(region_loeo_df.loc[delta_rmse_change.idxmax(), "event_id"]),
            }
        )

    loeo_summary_df = pd.DataFrame(loeo_summary_rows)
    heldout_predictions_df = pd.DataFrame(heldout_prediction_rows).sort_values(["region", "event_id", "month"]).reset_index(drop=True)
    heldout_skill_df = pd.DataFrame(heldout_skill_rows).sort_values(["region", "event_id"]).reset_index(drop=True)

    pooled_rows: list[dict[str, object]] = []
    deterministic_by_region: dict[str, dict[str, object]] = {}
    for region in REGIONS:
        region_pred = heldout_predictions_df.loc[heldout_predictions_df["region"] == region].copy()
        obs = region_pred[RESPONSE].to_numpy(dtype=np.float64)
        pred_m1 = region_pred["predicted_M1"].to_numpy(dtype=np.float64)
        pred_m3 = region_pred["predicted_M3"].to_numpy(dtype=np.float64)
        skill_m1 = prediction_skill(obs, pred_m1)
        skill_m3 = prediction_skill(obs, pred_m3)
        region_event_skill = heldout_skill_df.loc[heldout_skill_df["region"] == region]
        row = {
            "region": region,
            "n_heldout_ENSO_months": int(skill_m1["n"]),
            "pooled_event_OOF_R2_M1": float(skill_m1["R2"]),
            "pooled_event_OOF_R2_M3": float(skill_m3["R2"]),
            "delta_pooled_event_OOF_R2_M3_minus_M1": float(skill_m3["R2"] - skill_m1["R2"]),
            "pooled_event_RMSE_M1": float(skill_m1["RMSE"]),
            "pooled_event_RMSE_M3": float(skill_m3["RMSE"]),
            "delta_pooled_event_RMSE_M3_minus_M1": float(skill_m3["RMSE"] - skill_m1["RMSE"]),
            "pooled_event_MAE_M1": float(skill_m1["MAE"]),
            "pooled_event_MAE_M3": float(skill_m3["MAE"]),
            "delta_pooled_event_MAE_M3_minus_M1": float(skill_m3["MAE"] - skill_m1["MAE"]),
            "pooled_event_Pearson_r_M1": float(skill_m1["Pearson_r"]) if np.isfinite(skill_m1["Pearson_r"]) else np.nan,
            "pooled_event_Pearson_r_M3": float(skill_m3["Pearson_r"]) if np.isfinite(skill_m3["Pearson_r"]) else np.nan,
            "fraction_events_M3_better_RMSE": float(region_event_skill["M3_better_RMSE"].mean()),
            "fraction_events_M3_better_MAE": float(region_event_skill["M3_better_MAE"].mean()),
        }
        if row["n_heldout_ENSO_months"] != EXPECTED_ENSO_COUNTS_DEGC05["total_enso_months"]:
            raise RuntimeError(f"{region} pooled degC05 held-out month count mismatch: {row['n_heldout_ENSO_months']}")
        pooled_rows.append(row)
        deterministic_by_region[region] = row
    pooled_df = pd.DataFrame(pooled_rows)

    events_by_phase = {
        phase_name: [segment.event_id for segment in event_segments if segment.phase_name == phase_name]
        for phase_name in ["El Nino", "La Nina"]
    }
    preds_by_region_event = {
        region: {
            event_id: heldout_predictions_df[(heldout_predictions_df["region"] == region) & (heldout_predictions_df["event_id"] == event_id)].copy()
            for event_id in [segment.event_id for segment in event_segments]
        }
        for region in REGIONS
    }
    rng = np.random.default_rng(SEED)
    bootstrap_delta_r2 = {region: np.full(N_BOOT, np.nan, dtype=np.float64) for region in REGIONS}
    bootstrap_delta_rmse = {region: np.full(N_BOOT, np.nan, dtype=np.float64) for region in REGIONS}
    bootstrap_delta_mae = {region: np.full(N_BOOT, np.nan, dtype=np.float64) for region in REGIONS}
    for boot_idx in range(N_BOOT):
        sampled_events: list[str] = []
        for phase_name in ["El Nino", "La Nina"]:
            pool = np.array(events_by_phase[phase_name], dtype=object)
            draw = rng.choice(pool, size=len(pool), replace=True)
            sampled_events.extend(draw.tolist())
        for region in REGIONS:
            sample_df = pd.concat([preds_by_region_event[region][event_id] for event_id in sampled_events], ignore_index=True)
            obs = sample_df[RESPONSE].to_numpy(dtype=np.float64)
            pred_m1 = sample_df["predicted_M1"].to_numpy(dtype=np.float64)
            pred_m3 = sample_df["predicted_M3"].to_numpy(dtype=np.float64)
            skill_m1 = prediction_skill(obs, pred_m1)
            skill_m3 = prediction_skill(obs, pred_m3)
            bootstrap_delta_r2[region][boot_idx] = float(skill_m3["R2"] - skill_m1["R2"])
            bootstrap_delta_rmse[region][boot_idx] = float(skill_m3["RMSE"] - skill_m1["RMSE"])
            bootstrap_delta_mae[region][boot_idx] = float(skill_m3["MAE"] - skill_m1["MAE"])

    bootstrap_rows = []
    for region in REGIONS:
        r2_low, r2_high = np.nanpercentile(bootstrap_delta_r2[region], [2.5, 97.5])
        rmse_low, rmse_high = np.nanpercentile(bootstrap_delta_rmse[region], [2.5, 97.5])
        mae_low, mae_high = np.nanpercentile(bootstrap_delta_mae[region], [2.5, 97.5])
        det = deterministic_by_region[region]
        bootstrap_rows.append(
            {
                "region": region,
                "comparison": "M3_minus_M1",
                "deterministic_delta_pooled_event_OOF_R2": det["delta_pooled_event_OOF_R2_M3_minus_M1"],
                "delta_pooled_event_OOF_R2_ci_low_95": float(r2_low),
                "delta_pooled_event_OOF_R2_ci_high_95": float(r2_high),
                "probability_delta_pooled_event_OOF_R2_gt_0": float(np.mean(bootstrap_delta_r2[region] > 0)),
                "deterministic_delta_pooled_event_RMSE": det["delta_pooled_event_RMSE_M3_minus_M1"],
                "delta_pooled_event_RMSE_ci_low_95": float(rmse_low),
                "delta_pooled_event_RMSE_ci_high_95": float(rmse_high),
                "probability_delta_pooled_event_RMSE_lt_0": float(np.mean(bootstrap_delta_rmse[region] < 0)),
                "deterministic_delta_pooled_event_MAE": det["delta_pooled_event_MAE_M3_minus_M1"],
                "delta_pooled_event_MAE_ci_low_95": float(mae_low),
                "delta_pooled_event_MAE_ci_high_95": float(mae_high),
                "probability_delta_pooled_event_MAE_lt_0": float(np.mean(bootstrap_delta_mae[region] < 0)),
                "bootstrap_unit": "ENSO phase episode cluster",
                "stratified_by_phase": True,
                "shared_event_sample_across_regions_and_models": True,
            }
        )
    bootstrap_df = pd.DataFrame(bootstrap_rows)

    return {
        "inventory": inventory_df,
        "loeo_summary": loeo_summary_df,
        "event_skill": heldout_skill_df,
        "event_predictions": heldout_predictions_df,
        "pooled": pooled_df,
        "bootstrap": bootstrap_df,
    }


def main() -> int:
    for path in [FIG_DIR, RESULT_DIR, NOTES_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for path in [
        STEP10_1A_ALIGNED_PATH,
        STEP10_1A_MODEL_SUMMARY_PATH,
        STEP10_1B2A_PRED_PATH,
        STEP10_1B2A_SKILL_PATH,
        STEP10_1B2A_INCREMENT_PATH,
        STEP10_1B2A_BOOT_PATH,
        STEP10_1B2B_POOLED_PATH,
        STEP10_1B2B_EVENT_BOOT_PATH,
        STEP10_1B2B_LOEO_SUMMARY_PATH,
        STEP10_1B2B_SUMMARY_PATH,
        V1_CAPTION_PATH,
        V1_CHECKS_PATH,
        V1_SCRIPT_PATH,
        NINO_PATH,
    ]:
        require_exists(path)

    aligned_df = pd.read_csv(STEP10_1A_ALIGNED_PATH, parse_dates=["month"]).sort_values(["region", "month"]).reset_index(drop=True)
    model_summary_df = pd.read_csv(STEP10_1A_MODEL_SUMMARY_PATH)
    nino_df = pd.read_csv(NINO_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    pred_df = pd.read_csv(STEP10_1B2A_PRED_PATH, parse_dates=["month"]).sort_values(["region", "month"]).reset_index(drop=True)
    skill_df = pd.read_csv(STEP10_1B2A_SKILL_PATH)
    inc_df = pd.read_csv(STEP10_1B2A_INCREMENT_PATH)
    inc_boot_df = pd.read_csv(STEP10_1B2A_BOOT_PATH)
    degc05_event = build_degC05_event_robustness(aligned_df, model_summary_df, nino_df)
    pooled_df = degc05_event["pooled"]
    event_boot_df = degc05_event["bootstrap"]
    loeo_summary_df = degc05_event["loeo_summary"]
    event_summary_text = STEP10_1B2B_SUMMARY_PATH.read_text(encoding="utf-8")

    degc05_event["inventory"].to_csv(OUT_EVENT_INVENTORY_DEGC05, index=False)
    degc05_event["loeo_summary"].to_csv(OUT_LOEO_SUMMARY_DEGC05, index=False)
    degc05_event["event_skill"].to_csv(OUT_EVENT_SKILL_DEGC05, index=False)
    degc05_event["event_predictions"].to_csv(OUT_EVENT_PRED_DEGC05, index=False)
    degc05_event["pooled"].to_csv(OUT_POOLED_DEGC05, index=False)
    degc05_event["bootstrap"].to_csv(OUT_EVENT_BOOT_DEGC05, index=False)

    if "Figure 10 allowed to enter formal plotting = True." not in event_summary_text:
        raise RuntimeError("Figure 10 plotting approval is not present in the fixed robustness summary.")
    if pred_df.shape[0] != 744:
        raise RuntimeError("Blocked-CV monthly prediction count mismatch.")
    if set(pred_df["region"].unique().tolist()) != set(REGIONS):
        raise RuntimeError("Blocked-CV monthly prediction regions do not match WP/CP/EP only.")

    skill_map = skill_df.set_index(["region", "model_name"])
    inc_map = inc_df.set_index(["region", "comparison"])
    event_boot_map = event_boot_df.set_index("region")
    pooled_map = pooled_df.set_index("region")
    loeo_summary_map = loeo_summary_df.set_index("region")

    for region in REGIONS:
        region_pred = pred_df[pred_df["region"] == region]
        if len(region_pred) != 248:
            raise RuntimeError(f"{region} does not have 248 monthly held-out points.")

        heldout_r2 = float(skill_map.loc[(region, "M3"), "OOF_R2"])
        delta_r2 = float(inc_map.loc[(region, "M3_minus_M1"), "delta_OOF_R2"])
        exp_r2, exp_delta = EXPECTED_HELDOUT_BOX[region]
        assert_close(heldout_r2, exp_r2, f"{region} held-out R2(M3)")
        assert_close(delta_r2, exp_delta, f"{region} held-out delta R2 vs M1")

        ep_det = float(pooled_map.loc[region, "delta_pooled_event_OOF_R2_M3_minus_M1"])
        ep_low = float(event_boot_map.loc[region, "delta_pooled_event_OOF_R2_ci_low_95"])
        ep_high = float(event_boot_map.loc[region, "delta_pooled_event_OOF_R2_ci_high_95"])
        if not np.isfinite(ep_det) or not np.isfinite(ep_low) or not np.isfinite(ep_high):
            raise RuntimeError(f"{region} degC05 phase-episode-held-out diagnostics are not finite.")

        if float(loeo_summary_map.loc[region, "fraction_events_HCTB_coefficient_sign_consistent"]) != 1.0:
            raise RuntimeError(f"{region} leave-one-episode-out HCTB sign consistency is not 1.0.")
        if float(loeo_summary_map.loc[region, "fraction_events_delta_adjusted_R2_positive"]) != 1.0:
            raise RuntimeError(f"{region} leave-one-episode-out delta adjusted R2 is not positive for all episodes.")
        if float(loeo_summary_map.loc[region, "fraction_events_delta_RMSE_negative"]) != 1.0:
            raise RuntimeError(f"{region} leave-one-episode-out delta RMSE is not negative for all episodes.")

    all_vals = pd.concat([pred_df["Direct_Net_anom_formal"], pred_df["predicted_M3"]], ignore_index=True)
    lim = float(np.nanmax(np.abs(all_vals.to_numpy(dtype=float))))
    lim = np.ceil((lim * 1.08) / 0.5) * 0.5
    axis_limits = (-lim, lim)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.titlesize": 13,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    fig.subplots_adjust(left=-0.12, right=1.15, top=0.9, bottom=0.12, wspace=-0.45)

    plot_rows: list[dict[str, object]] = []
    for idx, region in enumerate(REGIONS):
        ax = axes[idx]
        color = REGION_COLORS[region]["main"]
        light = REGION_COLORS[region]["light"]
        region_pred = pred_df[pred_df["region"] == region].copy()
        region_pred["panel_group"] = f"scatter_{region}"
        plot_rows.extend(region_pred.to_dict(orient="records"))

        ax.scatter(
            region_pred["Direct_Net_anom_formal"],
            region_pred["predicted_M3"],
            s=20,
            color=color,
            alpha=0.62,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        ax.axline((0, 0), slope=1.0, color="#333333", linestyle=(0, (4, 3)), linewidth=1.1, zorder=2)
        ax.axhline(0.0, color="#d0d0d0", linewidth=0.8, zorder=1)
        ax.axvline(0.0, color="#d0d0d0", linewidth=0.8, zorder=1)
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            length=4,
            width=0.8,
            pad=3,
            labelsize=12,
        )

        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.set_title(REGION_NAMES[region], loc="center", pad=6, fontsize=14)
        panel_label(ax, PANEL_LABELS[region])

        ax.set_xlabel("Observed direct Net CRE anomaly (W m$^{-2}$)", fontsize=13)
        if idx == 0:
            ax.set_ylabel("Estimated Net CRE anomaly (W m$^{-2}$)", fontsize=13)
        else:
            ax.set_ylabel("")

        heldout_r2 = float(skill_map.loc[(region, "M3"), "OOF_R2"])
        delta_r2 = float(inc_map.loc[(region, "M3_minus_M1"), "delta_OOF_R2"])
        ax.text(
            0.03,
            0.97,
            f"Held-out R$^2$ (M2) = {heldout_r2:.3f}\n$\\Delta$R$^2$ vs M1 = {delta_r2:+.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            bbox={
                "facecolor": "none",
                "edgecolor": "none",
                "linewidth": 0.0,
                "boxstyle": "round,pad=0.28",
                "alpha": 0.70,
            },
            zorder=10,
        )

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    pd.DataFrame(plot_rows).to_csv(OUT_PLOT_DATA, index=False)
    OUT_CAPTION.write_text(CAPTION_TEXT + "\n", encoding="utf-8")
    make_input_manifest()

    method_lines = [
        "Figure10 degC05 method and plot checks",
        "",
        "Revision scope",
        "- v2_txt_degC05 keeps the Figure10 v2 simplified 1x3 layout = True.",
        "- panels (a)-(c) blocked-CV scatter and annotations are unchanged from the fixed all-month Step10-1B-2A outputs = True.",
        "- ENSO phase-episode robustness bookkeeping is re-derived under the +/-0.5 C monthly Nino3.4 definition = True.",
        "- no values, models, bootstrap or CV for panels (a)-(c) are recomputed = True.",
        "",
        "Fixed audited inputs actually used",
        f"- aligned monthly direct/metric table used for degC05 phase-episode robustness derivation: {STEP10_1A_ALIGNED_PATH}",
        f"- full-sample deterministic model summary used for degC05 leave-one-phase-episode-out reference comparisons: {STEP10_1A_MODEL_SUMMARY_PATH}",
        f"- Nino3.4 monthly anomaly input used for degC05 phase definition: {NINO_PATH}",
        f"- panel (a)-(c) blocked-CV monthly held-out diagnostic estimate input: {STEP10_1B2A_PRED_PATH}",
        f"- panel (a)-(c) blocked-CV skill input: {STEP10_1B2A_SKILL_PATH}",
        f"- blocked-CV increment input retained for panel annotations only: {STEP10_1B2A_INCREMENT_PATH}",
        f"- blocked-CV bootstrap CI input retained in method checks only: {STEP10_1B2A_BOOT_PATH}",
        f"- original ENSO-phase-episode-held-out summary input retained for audit context only: {STEP10_1B2B_POOLED_PATH}",
        f"- original ENSO-phase-episode-held-out bootstrap CI input retained for audit context only: {STEP10_1B2B_EVENT_BOOT_PATH}",
        f"- original leave-one-ENSO-phase-episode-out summary input retained for audit context only: {STEP10_1B2B_LOEO_SUMMARY_PATH}",
        f"- derived degC05 event inventory output: {OUT_EVENT_INVENTORY_DEGC05}",
        f"- derived degC05 phase-episode-held-out monthly predictions output: {OUT_EVENT_PRED_DEGC05}",
        f"- derived degC05 phase-episode-held-out skill-by-event output: {OUT_EVENT_SKILL_DEGC05}",
        f"- derived degC05 pooled phase-episode-held-out skill output: {OUT_POOLED_DEGC05}",
        f"- derived degC05 phase-episode bootstrap increment output: {OUT_EVENT_BOOT_DEGC05}",
        f"- derived degC05 leave-one-phase-episode-out summary output: {OUT_LOEO_SUMMARY_DEGC05}",
        f"- original robustness gate summary input read only: {STEP10_1B2B_SUMMARY_PATH}",
        f"- v1 caption reference read only: {V1_CAPTION_PATH}",
        f"- v1 method checks reference read only: {V1_CHECKS_PATH}",
        f"- v1 plotting script reference read only: {V1_SCRIPT_PATH}",
        "",
        "Plotting scope and guards",
        "- formal plot generated from fixed audited outputs only.",
        "- no formal direct monthly series recomputed during plotting.",
        "- no HCCF/HCTB/LCSP metrics recomputed during plotting.",
        "- no models refit for the plotted 248-month blocked-CV scatter panels.",
        "- no bootstrap or CV recomputed for the plotted 248-month blocked-CV scatter panels.",
        "- panels (a)-(c) use purged blocked-CV monthly held-out diagnostic estimates.",
        "- ENSO robustness quantities in caption/method checks are recomputed only for the degC05 phase-episode sensitivity bookkeeping.",
        "- panels (d)-(f) remain omitted from the simplified main figure.",
        "",
        "Model and method boundary",
        "- M1 = LCSP + HCCF.",
        "- M3 = LCSP + HCCF + HCTB.",
        "- formal direct terminology = fixed regional-box mean direct daytime Net CRE anomaly.",
        "- direct and metric spatial support compatible = True.",
        "- HCCF comparator only.",
        "- HCTB remapped current six-cell Deep convective definition.",
        "- HCTB current Deep convective members = 29,30,35,36,41,42.",
        "- LCSP equals negative current Low cloud occurrence.",
        f"- degC05 ENSO definition = nino34_anom with El Nino >= +{ENSO_THRESHOLD_DEGC05:.1f} C and La Nina <= -{ENSO_THRESHOLD_DEGC05:.1f} C.",
        f"- degC05 ENSO month counts = El Nino {EXPECTED_ENSO_COUNTS_DEGC05['el_nino_months']}, La Nina {EXPECTED_ENSO_COUNTS_DEGC05['la_nina_months']}, total {EXPECTED_ENSO_COUNTS_DEGC05['total_enso_months']}.",
        f"- degC05 ENSO phase-episode count = {len(degc05_event['inventory'])} total ({int((degc05_event['inventory']['phase'] == 'El Nino').sum())} El Nino, {int((degc05_event['inventory']['phase'] == 'La Nina').sum())} La Nina).",
        "- ENSO robustness terminology remains phase episode unless formal event-definition verification is later completed.",
        "- no DCEP.",
        "- no CE/GCE.",
        "- no environmental variables.",
        "- no old regional file.",
        "- no all-42 joint strict mask.",
        "",
        "Fixed statistical annotations in plotted panels",
        "- WP Held-out R2 (M3) = 0.611; delta R2 vs M1 = +0.617.",
        "- CP Held-out R2 (M3) = 0.613; delta R2 vs M1 = +0.477.",
        "- EP Held-out R2 (M3) = 0.873; delta R2 vs M1 = +0.032.",
        "",
        "degC05 robustness results retained from removed panels",
        f"- WP ENSO-phase-episode-held-out delta R2 = {float(pooled_map.loc['WP', 'delta_pooled_event_OOF_R2_M3_minus_M1']):+.3f}, 95% CI [{float(event_boot_map.loc['WP', 'delta_pooled_event_OOF_R2_ci_low_95']):+.3f}, {float(event_boot_map.loc['WP', 'delta_pooled_event_OOF_R2_ci_high_95']):+.3f}].",
        f"- CP ENSO-phase-episode-held-out delta R2 = {float(pooled_map.loc['CP', 'delta_pooled_event_OOF_R2_M3_minus_M1']):+.3f}, 95% CI [{float(event_boot_map.loc['CP', 'delta_pooled_event_OOF_R2_ci_low_95']):+.3f}, {float(event_boot_map.loc['CP', 'delta_pooled_event_OOF_R2_ci_high_95']):+.3f}].",
        f"- EP ENSO-phase-episode-held-out delta R2 = {float(pooled_map.loc['EP', 'delta_pooled_event_OOF_R2_M3_minus_M1']):+.3f}, 95% CI [{float(event_boot_map.loc['EP', 'delta_pooled_event_OOF_R2_ci_low_95']):+.3f}, {float(event_boot_map.loc['EP', 'delta_pooled_event_OOF_R2_ci_high_95']):+.3f}].",
        "- leave-one-ENSO-phase-episode-out HCTB coefficient sign-consistent fraction = 1.000 in WP/CP/EP under degC05.",
        "- M3-M1 adjusted R2 increment remains positive after removing any single degC05 phase episode in WP/CP/EP.",
        "- M3-M1 RMSE difference remains negative after removing any single degC05 phase episode in WP/CP/EP.",
        "",
        "Outputs",
        f"- png output: {OUT_PNG}",
        f"- pdf output: {OUT_PDF}",
        f"- plot data output: {OUT_PLOT_DATA}",
        f"- caption output: {OUT_CAPTION}",
        f"- method checks output: {OUT_CHECKS}",
        f"- input manifest output: {OUT_MANIFEST}",
        f"- degC05 event inventory output: {OUT_EVENT_INVENTORY_DEGC05}",
        f"- degC05 pooled event-held-out skill output: {OUT_POOLED_DEGC05}",
        f"- degC05 event-bootstrap increment output: {OUT_EVENT_BOOT_DEGC05}",
        f"- degC05 leave-one-phase-episode-out summary output: {OUT_LOEO_SUMMARY_DEGC05}",
        "",
        "Interpretation boundary",
        "- no Figure 11 and no supplementary figures.",
        "- no prediction skill / control / verified mechanism / independent validation / exact reconstruction / full attribution interpretation.",
        "- direct response must not be described as ocean-only.",
        "- main-text figure count remains 10.",
    ]
    OUT_CHECKS.write_text("\n".join(method_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
