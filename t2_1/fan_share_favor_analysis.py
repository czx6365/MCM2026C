#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 fan_share 的“偏粉丝”稳健性分析（不改动原有主分析逻辑与输出）。

读取：
- weekly_method_comparison.csv
- weekly_contestant_metrics.csv

输出：
- fan_share_favor_summary.csv（新增文件）
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_columns(df: pd.DataFrame, required, df_name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} 缺少字段: {missing}")


def _coerce_int_season_week(df: pd.DataFrame, df_name: str):
    for col in ["season", "week"]:
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_na = df[col].isna().sum()
            if after_na > before_na:
                print(f"[warning] {df_name}.{col} 存在无法转为数值的值，已置为 NaN（请检查数据）")
    return df


def _parse_names(x) -> list:
    if not isinstance(x, str):
        return []
    return [s.strip() for s in x.split("|") if s.strip()]


def _same_elim_set(a, b) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    set_a = {x for x in a.split("|") if x}
    set_b = {x for x in b.split("|") if x}
    return set_a == set_b


def _mean_fan_share(fan_series: pd.Series, season, week, names: list) -> float:
    """
    返回淘汰集合的 fan_share 平均值。
    若任一选手缺失 fan_share（或找不到记录），返回 NaN（不静默丢弃）。
    """
    if not names:
        return np.nan
    vals = []
    for nm in names:
        key = (season, week, nm)
        if key not in fan_series.index:
            return np.nan
        v = fan_series.get(key)
        if pd.isna(v):
            return np.nan
        vals.append(float(v))
    return float(np.mean(vals)) if vals else np.nan


def _summarize(df: pd.DataFrame, scope: str) -> dict:
    delta = df["fan_favor_delta_share"]
    delta_non_na = delta.dropna()
    return {
        "scope": scope,
        "n_weeks": int(len(df)),
        "rank_elim_fan_share_mean": float(df["rank_elim_fan_share"].mean()),
        "rank_elim_fan_share_median": float(df["rank_elim_fan_share"].median()),
        "percent_elim_fan_share_mean": float(df["percent_elim_fan_share"].mean()),
        "percent_elim_fan_share_median": float(df["percent_elim_fan_share"].median()),
        "fan_favor_delta_share_mean": float(delta.mean()),
        "fan_favor_delta_share_median": float(delta.median()),
        "fan_favor_delta_share_pos_share": float((delta_non_na > 0).mean()) if len(delta_non_na) else np.nan,
        "fan_favor_delta_share_non_na": int(len(delta_non_na)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weekly-compare", default="/mnt/data/weekly_method_comparison.csv")
    parser.add_argument("--weekly-contestant", default="/mnt/data/weekly_contestant_metrics.csv")
    parser.add_argument("--out-dir", default="t2_1/out")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]

    def _fallback_path(p: str, local_fallback: str) -> str:
        if Path(p).exists():
            return p
        if p.startswith("/mnt/data/"):
            fallback = root_dir / local_fallback
            if fallback.exists():
                print(f"[info] 使用本地路径替代: {fallback}")
                return str(fallback)
        return p

    args.weekly_compare = _fallback_path(args.weekly_compare, "t2_1/out/weekly_method_comparison.csv")
    args.weekly_contestant = _fallback_path(args.weekly_contestant, "t2_1/out/weekly_contestant_metrics.csv")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wcmp = pd.read_csv(args.weekly_compare)
    wct = pd.read_csv(args.weekly_contestant)

    _ensure_columns(
        wcmp,
        ["season", "week", "rank_eliminated", "percent_eliminated"],
        "weekly_method_comparison",
    )
    _ensure_columns(
        wct,
        ["season", "week", "celebrity_name", "fan_share"],
        "weekly_contestant_metrics",
    )

    wcmp = _coerce_int_season_week(wcmp, "weekly_method_comparison")
    wct = _coerce_int_season_week(wct, "weekly_contestant_metrics")

    fan_series = (
        wct.set_index(["season", "week", "celebrity_name"])["fan_share"]
    )

    # 逐周计算两种方法下淘汰者的 fan_share
    rank_vals = []
    pct_vals = []
    for _, row in wcmp.iterrows():
        season = row["season"]
        week = row["week"]
        rank_names = _parse_names(row["rank_eliminated"])
        pct_names = _parse_names(row["percent_eliminated"])
        rank_vals.append(_mean_fan_share(fan_series, season, week, rank_names))
        pct_vals.append(_mean_fan_share(fan_series, season, week, pct_names))

    wcmp = wcmp.copy()
    wcmp["rank_elim_fan_share"] = rank_vals
    wcmp["percent_elim_fan_share"] = pct_vals
    wcmp["fan_favor_delta_share"] = wcmp["rank_elim_fan_share"] - wcmp["percent_elim_fan_share"]

    # 差异周（Rank vs Percent 淘汰集合不同）
    wcmp["methods_differ"] = ~wcmp.apply(
        lambda r: _same_elim_set(r["rank_eliminated"], r["percent_eliminated"]),
        axis=1
    )

    summary_rows = [
        _summarize(wcmp, "all_weeks"),
        _summarize(wcmp[wcmp["methods_differ"] == True], "diff_weeks"),
    ]

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "fan_share_favor_summary.csv", index=False)

    print(f"DONE: {out_dir / 'fan_share_favor_summary.csv'}")


if __name__ == "__main__":
    main()
