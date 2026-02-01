#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4 Q4 support code:
1) Build the input datasets expected by Q4.ipynb from current project outputs.
2) Run a replay simulation to compare methods and export summary tables.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import numpy as np
import pandas as pd


def _normalize_vote_share(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Clip negative values and normalize per (season, week).
    If the weekly sum is 0 or missing, assign uniform shares.
    """
    v = df[col].astype(float).clip(lower=0.0)
    group_sum = v.groupby([df["season"], df["week"]]).transform("sum")
    group_cnt = v.groupby([df["season"], df["week"]]).transform("count")
    out = v / group_sum.replace(0.0, np.nan)
    out = out.fillna(1.0 / group_cnt.replace(0, np.nan))
    return out


def _fill_rel_ci80(x: pd.DataFrame, df_unc: Optional[pd.DataFrame]) -> pd.Series:
    """
    Fill rel_ci80 with week median -> global median -> 0.5 fallback.
    """
    if "rel_ci80" not in x.columns:
        return pd.Series([0.5] * len(x), index=x.index, dtype=float)

    rel = x["rel_ci80"].astype(float)
    week_median = rel.median(skipna=True)
    if np.isnan(week_median):
        week_median = np.nan

    global_median = np.nan
    if df_unc is not None and "rel_ci80" in df_unc.columns:
        global_median = float(pd.to_numeric(df_unc["rel_ci80"], errors="coerce").median(skipna=True))
    if np.isnan(global_median):
        global_median = 0.5

    rel = rel.fillna(week_median)
    rel = rel.fillna(global_median)
    return rel


def _find_out_dir(t2_dir: Path) -> Path:
    if not t2_dir.exists():
        raise FileNotFoundError(f"Missing t2_1 directory: {t2_dir}")
    for name in os.listdir(t2_dir):
        if name.startswith("out_pmap"):
            return t2_dir / name
    fallback = t2_dir / "out_question"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not find t2_1 out_pmap* or out_question directory.")


def _parse_celebrity(pair_id: str) -> str:
    if not isinstance(pair_id, str):
        return str(pair_id)
    m = re.match(r"^S\d+_(.*?) __ ", pair_id)
    return m.group(1) if m else pair_id


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_input_data(
    data_dir: Path,
    root_dir: Path,
    p_col: str = "p_map",
    extreme_quantile: float = 0.96,
) -> None:
    """
    Build Q4 input files into data_dir using existing project outputs.
    """
    _ensure_dir(data_dir)

    t2_dir = root_dir / "t2_1"
    out_dir = _find_out_dir(t2_dir)

    weekly_metrics_path = out_dir / "weekly_contestant_metrics.csv"
    weekly_comp_path = out_dir / "weekly_method_comparison.csv"
    fan_path = root_dir / "t1" / "fan_vote_estimates.csv"

    if not weekly_metrics_path.exists():
        raise FileNotFoundError(f"Missing {weekly_metrics_path}")
    if not weekly_comp_path.exists():
        raise FileNotFoundError(f"Missing {weekly_comp_path}")
    if not fan_path.exists():
        raise FileNotFoundError(f"Missing {fan_path}")

    weekly = pd.read_csv(weekly_metrics_path)
    weekly_comp = pd.read_csv(weekly_comp_path)
    fan = pd.read_csv(fan_path)

    # ---------------------------
    # Q1_part1_pred_df_active_rows.csv
    # ---------------------------
    pred = weekly.copy()
    if "J_score" in pred.columns:
        pred["judge_total"] = pred["J_score"]
    elif "judge_total" not in pred.columns:
        raise ValueError("weekly_contestant_metrics.csv missing J_score/judge_total")
    if "fan_share" in pred.columns:
        pred["vote_share_hat"] = pred["fan_share"]
    elif "vote_share_hat" not in pred.columns:
        raise ValueError("weekly_contestant_metrics.csv missing fan_share/vote_share_hat")

    # Basic sanity: non-negative judge totals and normalized vote shares
    pred["judge_total"] = pd.to_numeric(pred["judge_total"], errors="coerce").fillna(0.0).clip(lower=0.0)
    pred["vote_share_hat"] = _normalize_vote_share(pred, "vote_share_hat")

    # Keep a compact, explicit column order
    base_cols = [
        "season",
        "week",
        "celebrity_name",
        "pair_id",
        "judge_total",
        "vote_share_hat",
        "judge_rank",
        "fan_rank",
        "rank_total",
        "judge_pct",
        "combined_pct",
        "elim_count",
        "elim_flag",
    ]
    pred_cols = [c for c in base_cols if c in pred.columns]
    pred_out = pred[pred_cols].copy()
    pred_out.to_csv(data_dir / "Q1_part1_pred_df_active_rows.csv", index=False)

    # ---------------------------
    # Q1_part2_elimination_eval_week_df.csv
    # ---------------------------
    week_true = weekly_comp.copy()
    if "elim_count" not in week_true.columns or "actual_eliminated" not in week_true.columns:
        raise ValueError("weekly_method_comparison.csv missing elim_count/actual_eliminated")

    def _to_semicolon_list(x: str) -> str:
        if not isinstance(x, str) or not x.strip():
            return ""
        items = [i.strip() for i in x.split("|") if i.strip()]
        return ";".join(items)

    # Align actual eliminations to performance week.
    # Empirically, model-based predictions line up with the next week's elimination results.
    week_true["actual_eliminated_next"] = (
        week_true.groupby("season")["actual_eliminated"].shift(-1)
    )
    week_true["elim_count_next"] = (
        week_true.groupby("season")["elim_count"].shift(-1)
    )
    week_true["true_k"] = week_true["elim_count_next"].fillna(0).astype(int)
    week_true["true_elims"] = week_true["actual_eliminated_next"].apply(_to_semicolon_list)
    week_true_out = week_true[
        [
            "season",
            "week",
            "true_k",
            "true_elims",
            "actual_eliminated",
            "actual_eliminated_next",
        ]
    ].copy()
    week_true_out.to_csv(data_dir / "Q1_part2_elimination_eval_week_df.csv", index=False)

    # ---------------------------
    # Q3_all_extreme_disagreement_cases.csv
    # ---------------------------
    ext = pred.copy()
    if "judge_rank" not in ext.columns or "fan_rank" not in ext.columns:
        # Recompute if missing
        ext["judge_rank"] = (
            ext.groupby(["season", "week"])["judge_total"]
            .rank(ascending=False, method="min")
        )
        ext["fan_rank"] = (
            ext.groupby(["season", "week"])["vote_share_hat"]
            .rank(ascending=False, method="min")
        )

    # Cap rank_gap by n_active-1 to avoid extreme artifacts
    ext["n_active"] = ext.groupby(["season", "week"])["celebrity_name"].transform("count")
    ext["rank_gap"] = (ext["judge_rank"] - ext["fan_rank"]).abs()
    ext["rank_gap"] = ext["rank_gap"].clip(lower=0.0, upper=(ext["n_active"] - 1).clip(lower=0))

    # Robust quantile (avoid degenerate weeks)
    if not (0.0 < extreme_quantile < 1.0):
        extreme_quantile = 0.96
    q = ext.loc[ext["n_active"] >= 3, "rank_gap"].quantile(extreme_quantile)
    threshold = int(np.ceil(q)) if np.isfinite(q) else 0

    ext["disagreement_direction"] = np.where(
        ext["judge_rank"] > ext["fan_rank"],
        "judges_worse",
        np.where(ext["judge_rank"] < ext["fan_rank"], "fans_worse", "tie"),
    )

    ext_cases = ext[ext["rank_gap"] >= threshold].copy()
    ext_cases["extreme_rule"] = f"rank_gap>= {threshold}"
    ext_cols = [
        "season",
        "week",
        "celebrity_name",
        "pair_id",
        "judge_rank",
        "fan_rank",
        "rank_gap",
        "judge_pct",
        "vote_share_hat",
        "combined_pct",
        "disagreement_direction",
        "extreme_rule",
    ]
    ext_cols = [c for c in ext_cols if c in ext_cases.columns]
    ext_cases[ext_cols].to_csv(
        data_dir / "Q3_all_extreme_disagreement_cases.csv",
        index=False,
    )

    # ---------------------------
    # Q1_part3_uncertainty_unc_result_df.csv
    # ---------------------------
    if p_col not in fan.columns:
        p_col = "p_mean" if "p_mean" in fan.columns else p_col

    unc = fan.copy()
    # Map to Q4 expected column names
    if "rel_ci_width" in unc.columns:
        unc["rel_ci80"] = unc["rel_ci_width"]
    else:
        unc["rel_ci80"] = np.nan
    if "ci_width" in unc.columns:
        unc["ci80_width"] = unc["ci_width"]
    else:
        unc["ci80_width"] = np.nan

    if "p_lo90" in unc.columns and "p_hi90" in unc.columns:
        unc["votes_q10"] = unc["p_lo90"]
        unc["votes_q50"] = unc[p_col] if p_col in unc.columns else unc["p_mean"]
        unc["votes_q90"] = unc["p_hi90"]
    else:
        unc["votes_q10"] = np.nan
        unc["votes_q50"] = unc[p_col] if p_col in unc.columns else np.nan
        unc["votes_q90"] = np.nan

    keep_cols = [
        "season",
        "week",
        "celebrity_name",
        "p_mean",
        "p_map",
        "p_lo90",
        "p_hi90",
        "V_mean",
        "V_map",
        "ci_width",
        "rel_ci_width",
        "rel_ci80",
        "ci80_width",
        "votes_q10",
        "votes_q50",
        "votes_q90",
        "accept_rate",
        "scheme",
        "mean_entropy",
    ]
    keep_cols = [c for c in keep_cols if c in unc.columns]
    unc[keep_cols].to_csv(data_dir / "Q1_part3_uncertainty_unc_result_df.csv", index=False)


def _parse_elim_set(x) -> Set[str]:
    if pd.isna(x):
        return set()
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s2 = s[1:-1].strip()
        if not s2:
            return set()
        parts = [p.strip().strip("'").strip('"') for p in s2.split(",")]
        return set([p for p in parts if p])
    if ";" in s:
        return set([p.strip() for p in s.split(";") if p.strip()])
    if "|" in s:
        return set([p.strip() for p in s.split("|") if p.strip()])
    if "," in s:
        return set([p.strip() for p in s.split(",") if p.strip()])
    return set([s]) if s else set()


def method_rank_elim(df_sw: pd.DataFrame, k: int) -> Tuple[Set[str], pd.DataFrame]:
    x = df_sw.copy()
    x["judge_rank"] = x["judge_total"].rank(ascending=False, method="min")
    x["vote_rank"] = x["vote_share_hat"].rank(ascending=False, method="min")
    x["sum_rank"] = x["judge_rank"] + x["vote_rank"]
    elim = (
        x.sort_values("sum_rank", ascending=False)
        .head(int(k))["celebrity_name"]
        .tolist()
    )
    return set(elim), x


def method_percent_elim(df_sw: pd.DataFrame, k: int) -> Tuple[Set[str], pd.DataFrame]:
    x = df_sw.copy()
    s = x["judge_total"].sum()
    x["judge_pct"] = x["judge_total"] / s if s > 0 else 0.0
    x["vote_pct"] = x["vote_share_hat"]
    x["total_pct"] = x["judge_pct"] + x["vote_pct"]
    elim = (
        x.sort_values("total_pct", ascending=True)
        .head(int(k))["celebrity_name"]
        .tolist()
    )
    return set(elim), x


def _norm01(v: float) -> float:
    v = max(float(v), 0.0)
    return v / (v + 1.0)


def method_dynamic_judgesave_elim(
    df_sw: pd.DataFrame,
    k: int,
    w_base: float = 0.50,
    w_amp: float = 0.30,
    w_clip: Tuple[float, float] = (0.20, 0.80),
) -> Tuple[Set[str], pd.DataFrame]:
    x = df_sw.copy()

    s = x["judge_total"].sum()
    x["judge_pct"] = x["judge_total"] / s if s > 0 else 0.0
    x["vote_pct"] = x["vote_share_hat"]

    mu = x["judge_total"].mean()
    sd = x["judge_total"].std(ddof=0)
    judge_disp = (sd / mu) if (mu is not None and mu > 1e-12) else 0.0
    hhi = float((x["vote_pct"] ** 2).sum())

    disp_n = _norm01(judge_disp)
    hhi_n = _norm01(hhi)

    w_fan = w_base + w_amp * (1.0 - disp_n) - w_amp * (hhi_n)
    w_fan = float(np.clip(w_fan, w_clip[0], w_clip[1]))
    w_jdg = 1.0 - w_fan

    x["w_fan"] = w_fan
    x["w_jdg"] = w_jdg
    x["dynamic_score"] = w_jdg * x["judge_pct"] + w_fan * x["vote_pct"]

    k = int(k)
    if k <= 0:
        return set(), x

    if k == 1:
        bottom2 = x.sort_values("dynamic_score", ascending=True).head(2).copy()
        loser = (
            bottom2.sort_values("judge_total", ascending=True)
            .head(1)["celebrity_name"]
            .iloc[0]
        )
        return set([loser]), x
    elim = (
        x.sort_values("dynamic_score", ascending=True)
        .head(k)["celebrity_name"]
        .tolist()
    )
    return set(elim), x


def method_uncertainty_geometric_elim(
    df_sw: pd.DataFrame,
    k: int,
    df_unc: Optional[pd.DataFrame] = None,
    vote_transform: str = "linear",
    tau_judge: float = 0.1,
    eps: float = 1e-6,
) -> Tuple[Set[str], pd.DataFrame]:
    x = df_sw.copy()
    n = len(x)
    if n == 0:
        return set(), x

    judge_sum = x["judge_total"].sum()
    x["judge_share"] = x["judge_total"] / (judge_sum + eps) if judge_sum > 0 else 1.0 / n

    if vote_transform == "log":
        x["vote_raw"] = x["vote_share_hat"].fillna(0.0)
        x["vote_transformed"] = np.log1p(x["vote_raw"] * 1e6)
        vote_sum = x["vote_transformed"].sum()
        x["fan_share"] = x["vote_transformed"] / (vote_sum + eps) if vote_sum > 0 else 1.0 / n
    else:
        x["fan_share"] = x["vote_share_hat"].fillna(0.0)
        fan_sum = x["fan_share"].sum()
        x["fan_share"] = x["fan_share"] / fan_sum if fan_sum > 0 else 1.0 / n

    if df_unc is not None:
        x = x.merge(
            df_unc[
                [
                    "season",
                    "week",
                    "celebrity_name",
                    "rel_ci80",
                    "ci80_width",
                    "votes_q10",
                    "votes_q50",
                    "votes_q90",
                ]
            ],
            on=["season", "week", "celebrity_name"],
            how="left",
        )
        x["rel_ci80"] = _fill_rel_ci80(x, df_unc)
        u_F = x["rel_ci80"].mean()
        c_F = 1.0 / (1.0 + u_F + eps)
    else:
        c_F = 0.5

    judge_mean = x["judge_total"].mean()
    judge_std = x["judge_total"].std(ddof=0)
    d_J = (judge_std / (judge_mean + eps)) if judge_mean > eps else 0.0
    c_J = d_J / (d_J + tau_judge + eps)

    alpha_t = c_J / (c_J + c_F + eps)

    x["judge_share_safe"] = np.maximum(x["judge_share"], eps)
    x["fan_share_safe"] = np.maximum(x["fan_share"], eps)
    x["geometric_score"] = (
        np.power(x["judge_share_safe"], alpha_t)
        * np.power(x["fan_share_safe"], 1.0 - alpha_t)
    )
    x["c_F"] = c_F
    x["c_J"] = c_J
    x["alpha_t"] = alpha_t

    k = int(k)
    if k <= 0:
        return set(), x

    x_sorted = x.sort_values("geometric_score", ascending=True).copy()
    if k == 1:
        bottom3 = x_sorted.head(3).copy()
        saved = (
            bottom3.sort_values(["geometric_score", "fan_share"], ascending=[False, False])
            .head(1)["celebrity_name"]
            .iloc[0]
        )
        remaining = bottom3[bottom3["celebrity_name"] != saved].copy()
        if len(remaining) > 0:
            loser = (
                remaining.sort_values("geometric_score", ascending=True)
                .head(1)["celebrity_name"]
                .iloc[0]
            )
            return set([loser]), x
        return set([bottom3["celebrity_name"].iloc[-1]]), x

    elim = x_sorted.head(k)["celebrity_name"].tolist()
    return set(elim), x


def run_analysis(data_dir: Path, out_dir: Path) -> None:
    _ensure_dir(out_dir)

    pred_path = data_dir / "Q1_part1_pred_df_active_rows.csv"
    week_path = data_dir / "Q1_part2_elimination_eval_week_df.csv"
    ext_path = data_dir / "Q3_all_extreme_disagreement_cases.csv"
    unc_path = data_dir / "Q1_part3_uncertainty_unc_result_df.csv"

    df_pred = pd.read_csv(pred_path)
    df_week = pd.read_csv(week_path)
    df_ext = pd.read_csv(ext_path)
    df_unc = pd.read_csv(unc_path) if unc_path.exists() else None

    true_elim_col = None
    for c in df_week.columns:
        if "true" in c.lower() and "elim" in c.lower():
            true_elim_col = c
            break
    if true_elim_col is None:
        raise ValueError("df_week missing true elimination column")

    df_week = df_week.copy()
    df_week["true_elim_set"] = df_week[true_elim_col].apply(_parse_elim_set)
    df_ext["season"] = df_ext["season"].astype(int)
    df_ext["week"] = df_ext["week"].astype(int)

    # Normalize vote shares and clip negatives in analysis stage as well
    if "vote_share_hat" in df_pred.columns:
        df_pred["vote_share_hat"] = _normalize_vote_share(df_pred, "vote_share_hat")
    if "judge_total" in df_pred.columns:
        df_pred["judge_total"] = pd.to_numeric(df_pred["judge_total"], errors="coerce").fillna(0.0).clip(lower=0.0)

    ext_names_by_week = (
        df_ext.groupby(["season", "week"])["celebrity_name"]
        .apply(lambda s: set(s.astype(str).tolist()))
        .to_dict()
    )

    rows = []
    debug_rows = []

    for (s, w), g in df_week.groupby(["season", "week"], sort=True):
        k = int(g["true_k"].iloc[0])
        true_set = set(g["true_elim_set"].iloc[0])

        sw = df_pred[(df_pred["season"] == s) & (df_pred["week"] == w)].copy()
        if sw.shape[0] == 0:
            continue

        if "season" not in sw.columns:
            sw["season"] = s
        if "week" not in sw.columns:
            sw["week"] = w

        pred_rank, tab_rank = method_rank_elim(sw, k)
        pred_percent, tab_percent = method_percent_elim(sw, k)
        pred_dynamic, tab_dynamic = method_dynamic_judgesave_elim(sw, k)
        pred_geo, tab_geo = method_uncertainty_geometric_elim(
            sw, k, df_unc=df_unc
        )

        ext_names = ext_names_by_week.get((s, w), set())
        has_extreme = int(len(ext_names) > 0)

        rows.append({
            "season": s,
            "week": w,
            "n_active": int(sw.shape[0]),
            "true_k": k,
            "true_elims": ";".join(sorted(true_set)) if len(true_set) else "",
            "pred_rank": ";".join(sorted(pred_rank)) if len(pred_rank) else "",
            "pred_percent": ";".join(sorted(pred_percent)) if len(pred_percent) else "",
            "pred_dynamic_js": ";".join(sorted(pred_dynamic)) if len(pred_dynamic) else "",
            "pred_uncertainty_geo": ";".join(sorted(pred_geo)) if len(pred_geo) else "",
            "match_rank": int(pred_rank == true_set),
            "match_percent": int(pred_percent == true_set),
            "match_dynamic_js": int(pred_dynamic == true_set),
            "match_uncertainty_geo": int(pred_geo == true_set),
            "has_extreme_disagreement": has_extreme,
            "true_in_extreme": int(len(true_set & ext_names) > 0),
            "rank_in_extreme": int(len(pred_rank & ext_names) > 0),
            "percent_in_extreme": int(len(pred_percent & ext_names) > 0),
            "dynamic_in_extreme": int(len(pred_dynamic & ext_names) > 0),
            "uncertainty_geo_in_extreme": int(len(pred_geo & ext_names) > 0),
        })

        debug_rows.append({
            "season": s,
            "week": w,
            "true_k": k,
            "n_active": int(sw.shape[0]),
            "w_fan": float(tab_dynamic["w_fan"].iloc[0]) if "w_fan" in tab_dynamic.columns else np.nan,
            "w_jdg": float(tab_dynamic["w_jdg"].iloc[0]) if "w_jdg" in tab_dynamic.columns else np.nan,
            "alpha_t_geo": float(tab_geo["alpha_t"].iloc[0]) if "alpha_t" in tab_geo.columns else np.nan,
            "c_F": float(tab_geo["c_F"].iloc[0]) if "c_F" in tab_geo.columns else np.nan,
            "c_J": float(tab_geo["c_J"].iloc[0]) if "c_J" in tab_geo.columns else np.nan,
            "has_extreme": has_extreme,
        })

    replay_df = pd.DataFrame(rows)
    debug_df = pd.DataFrame(debug_rows)

    replay_df.to_csv(out_dir / "q4_replay_df.csv", index=False)
    debug_df.to_csv(out_dir / "q4_debug_df.csv", index=False)

    # Summary metrics
    elim_weeks = replay_df[replay_df["true_k"] > 0].copy()
    extreme_weeks = elim_weeks[elim_weeks["has_extreme_disagreement"] == 1].copy()

    def _hit_rate(df: pd.DataFrame, col: str) -> float:
        if df.empty:
            return float("nan")
        return float(df[col].mean())

    # Prepare ranks for fairness diagnostics
    if "judge_rank" not in df_pred.columns:
        df_pred["judge_rank"] = (
            df_pred.groupby(["season", "week"])["judge_total"]
            .rank(ascending=False, method="min")
        )
    if "fan_rank" not in df_pred.columns:
        df_pred["fan_rank"] = (
            df_pred.groupby(["season", "week"])["vote_share_hat"]
            .rank(ascending=False, method="min")
        )
    df_pred["rank_gap"] = (df_pred["judge_rank"] - df_pred["fan_rank"]).abs()

    pred_lookup = df_pred.set_index(["season", "week", "celebrity_name"])

    def _avg_elim_stats(method_col: str) -> Tuple[float, float, float]:
        jr, fr, gap = [], [], []
        for _, r in replay_df.iterrows():
            names = _parse_elim_set(r[method_col])
            if not names:
                continue
            for name in names:
                key = (r["season"], r["week"], name)
                if key in pred_lookup.index:
                    row = pred_lookup.loc[key]
                    jr.append(float(row["judge_rank"]))
                    fr.append(float(row["fan_rank"]))
                    gap.append(float(row["rank_gap"]))
        if not jr:
            return (float("nan"), float("nan"), float("nan"))
        return (float(np.mean(jr)), float(np.mean(fr)), float(np.mean(gap)))

    summary_rows = []
    for name, match_col, extreme_col, pred_col in [
        ("Rank", "match_rank", "rank_in_extreme", "pred_rank"),
        ("Percent", "match_percent", "percent_in_extreme", "pred_percent"),
        ("Dynamic+JudgeSave", "match_dynamic_js", "dynamic_in_extreme", "pred_dynamic_js"),
        ("Uncertainty+Geometric", "match_uncertainty_geo", "uncertainty_geo_in_extreme", "pred_uncertainty_geo"),
    ]:
        avg_jr, avg_fr, avg_gap = _avg_elim_stats(pred_col)
        summary_rows.append({
            "method": name,
            "match_rate": float(elim_weeks[match_col].mean()) if not elim_weeks.empty else np.nan,
            "extreme_hit_rate": _hit_rate(extreme_weeks, extreme_col),
            "avg_elim_judge_rank": avg_jr,
            "avg_elim_fan_rank": avg_fr,
            "avg_elim_rank_gap": avg_gap,
            "n_weeks": int(elim_weeks.shape[0]),
            "n_extreme_weeks": int(extreme_weeks.shape[0]),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "q4_method_summary.csv", index=False)

    # Disagreement-focused tables
    replay_df["pred_rank_set"] = replay_df["pred_rank"].apply(_parse_elim_set)
    replay_df["pred_percent_set"] = replay_df["pred_percent"].apply(_parse_elim_set)
    replay_df["pred_dynamic_set"] = replay_df["pred_dynamic_js"].apply(_parse_elim_set)
    replay_df["pred_uncertainty_set"] = replay_df["pred_uncertainty_geo"].apply(_parse_elim_set)

    replay_df["rank_vs_percent_diff"] = (replay_df["pred_rank_set"] != replay_df["pred_percent_set"]).astype(int)
    replay_df["geo_vs_rank_diff"] = (replay_df["pred_uncertainty_set"] != replay_df["pred_rank_set"]).astype(int)
    replay_df["geo_vs_percent_diff"] = (replay_df["pred_uncertainty_set"] != replay_df["pred_percent_set"]).astype(int)

    diff_weeks = replay_df[
        (replay_df["true_k"] > 0)
        & (
            (replay_df["rank_vs_percent_diff"] == 1)
            | (replay_df["geo_vs_rank_diff"] == 1)
            | (replay_df["geo_vs_percent_diff"] == 1)
        )
    ].copy()

    focus_weeks = diff_weeks[
        (diff_weeks["has_extreme_disagreement"] == 1)
        & (diff_weeks["geo_vs_rank_diff"] == 1)
    ].copy()

    diff_weeks.to_csv(out_dir / "q4_diff_weeks.csv", index=False)
    focus_weeks.to_csv(out_dir / "q4_focus_weeks.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="f:/MCM2026c", help="Project root directory")
    parser.add_argument("--data-dir", default="f:/MCM2026c/t4/data", help="T4 data output directory")
    parser.add_argument("--out-dir", default="f:/MCM2026c/t4/data", help="T4 analysis output directory")
    parser.add_argument("--build-data", action="store_true", help="Build Q4 input datasets")
    parser.add_argument("--run-analysis", action="store_true", help="Run replay and export summaries")
    args = parser.parse_args()

    root_dir = Path(args.root)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if args.build_data:
        build_input_data(data_dir=data_dir, root_dir=root_dir)
    if args.run_analysis:
        run_analysis(data_dir=data_dir, out_dir=out_dir)
    if not args.build_data and not args.run_analysis:
        # Default behavior: build data then run analysis
        build_input_data(data_dir=data_dir, root_dir=root_dir)
        run_analysis(data_dir=data_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()
