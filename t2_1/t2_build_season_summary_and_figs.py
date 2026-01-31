#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T2-1 跨季对比：Season-level 主表 + 两张论文级图 + “差异周机制解释”统计检验

运行示例：
python t2/t2_build_season_summary_and_figs.py \
  --diff-summary /mnt/data/season_method_diff_summary.csv \
  --week-completeness /mnt/data/week_data_completeness.csv \
  --elim-metrics /mnt/data/elim_rank_metrics.csv \
  --weekly-compare /mnt/data/weekly_method_comparison.csv \
  --weekly-contestant /mnt/data/weekly_contestant_metrics.csv \
  --out-dir t2/out_plus \
  --plot-judge-delta

输出（out-dir 下）：
- season_master_table.csv
- Figure_A_diff_share_vs_season.png
- Figure_B_fan_favor_delta_vs_season.png
- week_mechanism_features.csv
- mechanism_test_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import mannwhitneyu, ttest_ind
except Exception as exc:  # pragma: no cover
    raise ImportError("需要 scipy（用于统计检验）。请先安装：pip install scipy") from exc


# -----------------------------
# 工具函数
# -----------------------------
def _ensure_columns(df: pd.DataFrame, required, df_name: str):
    """检查必需字段是否存在，不存在则抛错。"""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} 缺少字段: {missing}")


def _to_bool(series: pd.Series) -> pd.Series:
    """
    将 0/1、True/False、'True'/'False' 等转换为布尔。
    注意：NaN 不应被静默当成 False（避免污染统计），这里保留 NaN。
    """
    if series.dtype == bool:
        return series
    if np.issubdtype(series.dtype, np.number):
        # 数值型：NaN 保留，非 NaN 转 bool
        out = series.copy()
        mask = out.notna()
        out.loc[mask] = out.loc[mask].astype(int).astype(bool)
        return out
    # 字符串型：NaN 保留
    out = series.copy()
    mask = out.notna()
    out.loc[mask] = (
        out.loc[mask]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "t", "yes", "y"])
    )
    return out


def _coerce_int_season_week(df: pd.DataFrame, df_name: str):
    """确保 season/week 可排序（尽量转成整数）；失败则保持原样但给 warning。"""
    for col in ["season", "week"]:
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_na = df[col].isna().sum()
            if after_na > before_na:
                print(f"[warning] {df_name}.{col} 存在无法转为数值的值，已置为 NaN（请检查数据）")
    return df


# -----------------------------
# 1) Season-level 主表
# -----------------------------
def build_season_master_table(diff_summary: pd.DataFrame,
                              week_completeness: pd.DataFrame,
                              elim_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    构建跨季主表（Season-level evidence table）。

    必备字段见脚本顶部注释。
    """
    wc = week_completeness.copy()
    _ensure_columns(wc, ["season", "week", "has_elim", "any_missing_fan"], "week_data_completeness")
    wc = _coerce_int_season_week(wc, "week_data_completeness")

    wc["has_elim"] = _to_bool(wc["has_elim"])
    wc["any_missing_fan"] = _to_bool(wc["any_missing_fan"])

    # 只在 has_elim==True 的周次统计 coverage
    wc_elim = wc[wc["has_elim"] == True].copy()
    wc_summary = (
        wc_elim.groupby("season", as_index=False)
        .agg(
            weeks_with_elim_total=("week", "count"),
            weeks_missing_fan=("any_missing_fan", lambda s: int((s == True).sum())),
        )
    )
    wc_summary["weeks_used"] = wc_summary["weeks_with_elim_total"] - wc_summary["weeks_missing_fan"]
    wc_summary["coverage"] = (
        wc_summary["weeks_used"] / wc_summary["weeks_with_elim_total"].replace(0, np.nan)
    ).round(3)

    ds = diff_summary.copy()
    _ensure_columns(ds, ["season", "diff_weeks", "diff_share"], "season_method_diff_summary")
    ds = _coerce_int_season_week(ds, "season_method_diff_summary")

    em = elim_metrics.copy()
    _ensure_columns(
        em,
        [
            "season", "week",
            "rank_elim_fan_rank", "percent_elim_fan_rank",
            "rank_elim_judge_rank", "percent_elim_judge_rank",
        ],
        "elim_rank_metrics",
    )
    em = _coerce_int_season_week(em, "elim_rank_metrics")

    # 周级 delta：percent - rank
    em["fan_delta_week"] = em["percent_elim_fan_rank"] - em["rank_elim_fan_rank"]
    em["judge_delta_week"] = em["percent_elim_judge_rank"] - em["rank_elim_judge_rank"]

    em_summary = (
        em.groupby("season", as_index=False)
        .agg(
            fan_favor_delta=("fan_delta_week", "mean"),
            judge_favor_delta=("judge_delta_week", "mean"),
            n_weeks_used_metrics=("week", "nunique"),
        )
    )

    # 合并
    master = wc_summary.merge(ds[["season", "diff_weeks", "diff_share"]], on="season", how="left")
    master = master.merge(em_summary, on="season", how="left")

    # 排序（season 小到大）
    master = master.sort_values("season").reset_index(drop=True)
    return master


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# 2) 作图（论文友好，最终版）
# -----------------------------
def _nice_season_ticks(seasons, max_ticks=18):
    """给 season 生成更易读的 x 轴刻度（自适应稀疏）"""
    seasons = np.array(sorted(set(int(s) for s in seasons if np.isfinite(s))))
    if len(seasons) == 0:
        return []
    # 控制最多显示 max_ticks 个刻度
    step = max(1, int(np.ceil(len(seasons) / max_ticks)))
    return seasons[::step].tolist()


import numpy as np
import matplotlib.pyplot as plt

def plot_diff_share(master, out_path, low_cov_thr=0.5):
    """
    Figure A：diff_share vs season
    - coverage 分档 -> 用不同颜色表示
    - 同档（同大小）同色：图里与图例完全一致
    """

    df = master.sort_values("season").copy()
    df = df[df["season"].notna()].copy()

    seasons = df["season"].astype(int).to_numpy()
    y = df["diff_share"].to_numpy(dtype=float)
    cov = df["coverage"].fillna(0).to_numpy(dtype=float)
    cov = np.clip(cov, 0, 1)

    # --- 1) 定义 coverage 分档（四档） ---
    # 你也可以改阈值：例如 0.25/0.5/0.75/1.0 是“代表值”
    def cov_bin(c):
        if c < 0.375:
            return 0.25
        elif c < 0.625:
            return 0.50
        elif c < 0.875:
            return 0.75
        else:
            return 1.00

    cov_level = np.array([cov_bin(c) for c in cov])

    # --- 2) 每档对应一个固定颜色 + 固定大小 ---
    # 颜色不必手动指定也行，但你想“不同大小不同颜色”就必须明确映射
    # 这里用 matplotlib tab10 的前四个（论文干净）
    level_order = [0.25, 0.50, 0.75, 1.00]
    level_color = {
        0.25: "C0",
        0.50: "C1",
        0.75: "C2",
        1.00: "C3",
    }
    level_size = {   # 同档同大小（满足“同意大小同一颜色”）
        0.25: 70,
        0.50: 140,
        0.75: 240,
        1.00: 360,
    }

    colors = [level_color[l] for l in cov_level]
    sizes  = [level_size[l]  for l in cov_level]

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.scatter(
        seasons, y,
        s=sizes,
        c=colors,
        alpha=0.90,
        edgecolors="white",
        linewidths=0.8
    )

    ax.set_title("Method Disagreement Share by Season", fontsize=14)
    ax.set_xlabel("Season")
    ax.set_ylabel("Disagreement share (rank != percent)")
    ax.set_ylim(-0.02, 1.05)

    # x 轴刻度（每2季一个）
    ticks = sorted(set(seasons.tolist()))[::2]
    ax.set_xticks(ticks)
    ax.grid(True, axis="y", alpha=0.25)

    # --- 3) 图例：每档一个示例点（颜色与大小完全一致）---
    for lv in level_order:
        ax.scatter(
            [], [], s=level_size[lv], c=level_color[lv],
            alpha=0.90, edgecolors="white", linewidths=0.8,
            label=f"coverage={lv:.2f}"
        )
    ax.legend(
        title="Coverage bins (color & size)",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False
    )

    # --- 4) 标注低 coverage 且 diff_share 高的点（可选）---
    # 注意：这里 low_cov_thr 仍按原始 coverage（连续值）判断
    low = df[(df["coverage"] < low_cov_thr) & (df["diff_share"] >= 0.5)]
    for _, r in low.iterrows():
        ax.annotate(
            f"S{int(r['season'])} (cov={r['coverage']:.2f})",
            (int(r["season"]), float(r["diff_share"])),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=9
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fan_favor_delta(master: pd.DataFrame, out_path: Path,
                         low_cov_thr: float = 0.5,
                         plot_judge_delta: bool = True):
    """
    Figure B：fan_favor_delta & judge_favor_delta vs season
    - fan: 圆点 o
    - judge: 叉号 x（不同符号表示）
    - 低 coverage 用星标（只标 fan，避免太乱）
    - 稳健 y 轴：用 5%~95% 分位 + padding
    - 极端点：箭头+注释保留信息
    """
    df = master.sort_values("season").copy()
    df = df[df["season"].notna()].copy()

    seasons = df["season"].astype(int).to_numpy()
    y_fan = df["fan_favor_delta"].to_numpy(dtype=float)

    # judge 可能有 NaN，先取出来
    y_judge = df["judge_favor_delta"].to_numpy(dtype=float) if "judge_favor_delta" in df.columns else None

    fig, ax = plt.subplots(figsize=(10.8, 4.2))

    # fan：圆点
    sc_fan = ax.scatter(
        seasons, y_fan,
        marker="o",
        alpha=0.9,
        edgecolors="white",
        linewidths=0.8,
        label="fan_favor_delta"
    )
    main_color = sc_fan.get_facecolor()[0]

    # judge：叉号（同色，不引入额外编码）
    if plot_judge_delta and (y_judge is not None):
        ax.scatter(
            seasons, y_judge,
            marker="x",
            alpha=0.85,
            linewidths=1.2,
            color=main_color,   # 关键：同色，仅用符号区分
            label="judge_favor_delta"
        )

    # 0 线
    ax.axhline(0, linestyle="--", linewidth=1, color="gray")

    # x 轴刻度（每2季一个）
    ax.set_xticks(sorted(set(seasons.tolist()))[::2])
    ax.tick_params(axis="x", rotation=0)

    ax.grid(True, axis="y", alpha=0.25)

    ax.set_title("Favor Delta by Season", fontsize=14)
    ax.set_xlabel("Season")
    ax.set_ylabel("Delta = (percent) − (rank)")

    # ---- 稳健 y 轴：同时考虑 fan & judge（如果画了 judge）----
    y_all = pd.Series(y_fan).dropna()
    if plot_judge_delta and (y_judge is not None):
        y_all = pd.concat([y_all, pd.Series(y_judge).dropna()], ignore_index=True)

    if len(y_all) >= 8:
        q05, q95 = float(y_all.quantile(0.05)), float(y_all.quantile(0.95))
        pad = 0.20 * max(1e-6, (q95 - q05))
        y_low, y_high = q05 - pad, q95 + pad
        ax.set_ylim(y_low, y_high)

        # 极端点（fan）用箭头标注
        out_hi = df[df["fan_favor_delta"] > y_high]
        out_lo = df[df["fan_favor_delta"] < y_low]
        for _, r in out_hi.iterrows():
            x = int(r["season"]); yv = float(r["fan_favor_delta"])
            ax.annotate(
                f"S{x}\n{yv:.2f}",
                (x, y_high),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="gray")
            )
        for _, r in out_lo.iterrows():
            x = int(r["season"]); yv = float(r["fan_favor_delta"])
            ax.annotate(
                f"S{x}\n{yv:.2f}",
                (x, y_low),
                textcoords="offset points",
                xytext=(0, -18),
                ha="center",
                fontsize=9,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="gray")
            )

    # 低 coverage：星标叠加在 fan 点上
    low = df[df["coverage"] < low_cov_thr]
    if not low.empty:
        ax.scatter(
            low["season"].astype(int),
            low["fan_favor_delta"].astype(float),
            marker="*",
            s=180,
            color=main_color,
            edgecolors="white",
            linewidths=0.8,
            label=f"coverage<{low_cov_thr:g}"
        )

    # legend 图外
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)





# -----------------------------
# 3) 差异周机制解释：构造周级特征 + 检验
# -----------------------------
def compute_week_features(weekly_contestant: pd.DataFrame,
                          weekly_compare: pd.DataFrame,
                          week_completeness: pd.DataFrame) -> pd.DataFrame:
    """
    按 season-week 计算机制特征，并合并 methods_differ。
    关键原则：methods_differ 缺失不应被静默当成 False —— 缺失周不进入检验。
    """
    wc = week_completeness.copy()
    _ensure_columns(wc, ["season", "week", "has_elim"], "week_data_completeness")
    wc = _coerce_int_season_week(wc, "week_data_completeness")
    wc["has_elim"] = _to_bool(wc["has_elim"])
    wc_elim = wc[wc["has_elim"] == True][["season", "week"]].copy()

    wcmp = weekly_compare.copy()
    _ensure_columns(wcmp, ["season", "week"], "weekly_method_comparison")
    wcmp = _coerce_int_season_week(wcmp, "weekly_method_comparison")

    if "methods_differ" not in wcmp.columns:
        _ensure_columns(wcmp, ["rank_eliminated", "percent_eliminated"], "weekly_method_comparison")
        wcmp["methods_differ"] = (wcmp["rank_eliminated"].astype(str) != wcmp["percent_eliminated"].astype(str))

    wcmp["methods_differ"] = _to_bool(wcmp["methods_differ"])
    wcmp = wcmp[["season", "week", "methods_differ"]].copy()

    wct = weekly_contestant.copy()
    _ensure_columns(
        wct,
        [
            "season", "week", "celebrity_name",
            "J_score", "fan_share",
            "rank_total", "combined_pct",
        ],
        "weekly_contestant_metrics",
    )
    wct = _coerce_int_season_week(wct, "weekly_contestant_metrics")

    # 只保留“有淘汰的周”（与 methods_differ 的定义域一致）
    wct = wct.merge(wc_elim, on=["season", "week"], how="inner")

    # 计算周级特征
    rows = []
    for (season, week), g in wct.groupby(["season", "week"]):
        g = g.copy()
        n_active = int(g["celebrity_name"].nunique())

        # fan concentration：top1 - top2
        fan_sorted = g["fan_share"].dropna().sort_values(ascending=False)
        top1_minus_top2 = float(fan_sorted.iloc[0] - fan_sorted.iloc[1]) if len(fan_sorted) >= 2 else np.nan

        # fan HHI：sum(p^2)
        fan_hhi = float((g["fan_share"].dropna() ** 2).sum()) if not g["fan_share"].dropna().empty else np.nan

        # judge closeness：range/mean, std/mean
        j_mean = g["J_score"].mean()
        if pd.isna(j_mean) or j_mean == 0:
            judge_range_norm = np.nan
            judge_std_norm = np.nan
        else:
            judge_range_norm = float((g["J_score"].max() - g["J_score"].min()) / j_mean)
            judge_std_norm = float(g["J_score"].std(ddof=0) / j_mean)

        # margin_rank：rank_total 大者更差；worst - 2nd worst
        rank_sorted = g["rank_total"].dropna().sort_values(ascending=False)
        margin_rank = float(rank_sorted.iloc[0] - rank_sorted.iloc[1]) if len(rank_sorted) >= 2 else np.nan

        # margin_pct：combined_pct 小者更差；2nd worst - worst
        pct_sorted = g["combined_pct"].dropna().sort_values(ascending=True)
        margin_pct = float(pct_sorted.iloc[1] - pct_sorted.iloc[0]) if len(pct_sorted) >= 2 else np.nan

        rows.append(
            {
                "season": int(season) if pd.notna(season) else season,
                "week": int(week) if pd.notna(week) else week,
                "n_active": n_active,
                "fan_top1_minus_top2": top1_minus_top2,
                "fan_hhi": fan_hhi,
                "judge_range_norm": judge_range_norm,
                "judge_std_norm": judge_std_norm,
                "margin_rank": margin_rank,
                "margin_pct": margin_pct,
            }
        )

    features = pd.DataFrame(rows)

    # 合并 methods_differ：用 inner，避免缺失周被当成 False
    before = len(features)
    features = features.merge(wcmp, on=["season", "week"], how="inner")
    after = len(features)
    if after < before:
        print(f"[warning] 有 {before - after} 个周次缺少 methods_differ，将不进入机制检验")

    features["methods_differ"] = _to_bool(features["methods_differ"])
    return features.sort_values(["season", "week"]).reset_index(drop=True)


def mechanism_tests(features: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    对机制特征进行差异检验（methods_differ==1 vs 0）
    - Mann–Whitney U（非参数）为主
    - t-test 为辅
    - effect size：rank-biserial（对称化 + 方向由均值差决定）
    """
    test_features = [
        "fan_top1_minus_top2",
        "fan_hhi",
        "judge_range_norm",
        "judge_std_norm",
        "margin_rank",
        "margin_pct",
    ]

    results = []
    summary_lines = []

    for feat in test_features:
        if feat not in features.columns:
            continue

        x1 = features.loc[features["methods_differ"] == True, feat].dropna()
        x0 = features.loc[features["methods_differ"] == False, feat].dropna()
        n1, n0 = len(x1), len(x0)

        p_mw, p_t, rbc = np.nan, np.nan, np.nan

        if n1 > 0 and n0 > 0:
            if n1 < 10 or n0 < 10:
                print(f"[low power] feature={feat}, n1={n1}, n0={n0}")

            mw = mannwhitneyu(x1, x0, alternative="two-sided")
            p_mw = float(mw.pvalue)

            # 对称化 rank-biserial + 方向
            U = float(mw.statistic)
            U_sym = min(U, n1 * n0 - U)
            rbc_mag = 1.0 - 2.0 * U_sym / (n1 * n0)  # 0~1
            sign = 1.0 if float(np.nanmean(x1)) > float(np.nanmean(x0)) else -1.0
            rbc = sign * rbc_mag

            # t-test（Welch）
            try:
                p_t = float(ttest_ind(x1, x0, equal_var=False).pvalue)
            except Exception:
                p_t = np.nan

        results.append(
            {
                "feature": feat,
                "group1_mean": float(np.nanmean(x1)) if n1 > 0 else np.nan,
                "group0_mean": float(np.nanmean(x0)) if n0 > 0 else np.nan,
                "group1_median": float(np.nanmedian(x1)) if n1 > 0 else np.nan,
                "group0_median": float(np.nanmedian(x0)) if n0 > 0 else np.nan,
                "mannwhitney_u_pvalue": p_mw,
                "ttest_pvalue": p_t,
                "effect_size_rank_biserial": rbc,
                "n_group1": n1,
                "n_group0": n0,
            }
        )

        # 论文式摘要（修复 NaN 判断）
        if (not np.isnan(p_mw)) and (p_mw < 0.05):
            direction = "higher" if float(np.nanmean(x1)) > float(np.nanmean(x0)) else "lower"
            summary_lines.append(f"Difference weeks show significantly {direction} {feat} (MW p={p_mw:.3g}).")

    summary_text = " ".join(summary_lines) if summary_lines else "No significant differences detected at p<0.05."
    return pd.DataFrame(results), summary_text


# -----------------------------
# main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff-summary", default="/mnt/data/season_method_diff_summary.csv")
    parser.add_argument("--week-completeness", default="/mnt/data/week_data_completeness.csv")
    parser.add_argument("--elim-metrics", default="/mnt/data/elim_rank_metrics.csv")
    parser.add_argument("--weekly-compare", default="/mnt/data/weekly_method_comparison.csv")
    parser.add_argument("--weekly-contestant", default="/mnt/data/weekly_contestant_metrics.csv")
    parser.add_argument("--out-dir", default="t2/out_plus")
    parser.add_argument("--plot-judge-delta", action="store_true",
                        help="在 Figure B 叠加 judge_favor_delta（默认不叠加，更论文干净）")
    parser.add_argument("--low-cov-thr", type=float, default=0.5,
                        help="低 coverage 阈值（用于标注/警示）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _fallback_path(p: str, local_fallback: str) -> str:
        """默认 /mnt/data 不存在时，自动回退到本地项目路径。"""
        if Path(p).exists():
            return p
        if p.startswith("/mnt/data/"):
            fallback = Path(local_fallback)
            if fallback.exists():
                print(f"[info] 使用本地路径替代: {fallback}")
                return str(fallback)
        return p

    args.diff_summary = _fallback_path(args.diff_summary, "t2/out/season_method_diff_summary.csv")
    args.week_completeness = _fallback_path(args.week_completeness, "t2/out/week_data_completeness.csv")
    args.elim_metrics = _fallback_path(args.elim_metrics, "t2/out/elim_rank_metrics.csv")
    args.weekly_compare = _fallback_path(args.weekly_compare, "t2/out/weekly_method_comparison.csv")
    args.weekly_contestant = _fallback_path(args.weekly_contestant, "t2/out/weekly_contestant_metrics.csv")

    # 读取数据
    diff_summary = pd.read_csv(args.diff_summary)
    week_completeness = pd.read_csv(args.week_completeness)
    elim_metrics = pd.read_csv(args.elim_metrics)
    weekly_compare = pd.read_csv(args.weekly_compare)
    weekly_contestant = pd.read_csv(args.weekly_contestant)

    # 生成 Season-level 主表
    master = build_season_master_table(diff_summary, week_completeness, elim_metrics)

    # sanity check：weeks_used vs n_weeks_used_metrics 不一致提示
    mismatch = master[
        master["n_weeks_used_metrics"].notna()
        & master["weeks_used"].notna()
        & (master["n_weeks_used_metrics"] != master["weeks_used"])
    ]
    if not mismatch.empty:
        seasons = mismatch["season"].dropna().astype(int).tolist()
        print(f"[warning] elim_rank_metrics 周数与 weeks_used 不一致（可能存在过滤/缺失差异），season={seasons}")

    # 打印 coverage 信息（论文里可直接引用）
    print(f"Total seasons: {master['season'].nunique()}")
    cov = master["coverage"].dropna()
    if not cov.empty:
        print(f"Coverage min/median/max: {cov.min():.3f}/{cov.median():.3f}/{cov.max():.3f}")
    low_cov = master.loc[(master["coverage"].notna()) & (master["coverage"] < args.low_cov_thr), "season"]
    low_cov_list = low_cov.dropna().astype(int).tolist()
    print(f"Coverage < {args.low_cov_thr:g} seasons: {low_cov_list}")

    # 保存主表
    master_out = out_dir / "season_master_table.csv"
    master.to_csv(master_out, index=False)

    # 画图
    plot_diff_share(master, out_dir / "Figure_A_diff_share_vs_season.png", low_cov_thr=args.low_cov_thr)
    plot_fan_favor_delta(
        master,
        out_dir / "Figure_B_fan_favor_delta_vs_season.png",
        low_cov_thr=args.low_cov_thr,
        plot_judge_delta=args.plot_judge_delta,
    )

    # 机制特征表
    features = compute_week_features(weekly_contestant, weekly_compare, week_completeness)
    features_out = out_dir / "week_mechanism_features.csv"
    features.to_csv(features_out, index=False)

    # 机制检验
    test_results, summary_text = mechanism_tests(features)
    test_out = out_dir / "mechanism_test_results.csv"
    test_results.to_csv(test_out, index=False)

    print("Mechanism test summary:")
    print(summary_text)

    # 输出完整性检查
    expected = [
        "season_master_table.csv",
        "Figure_A_diff_share_vs_season.png",
        "Figure_B_fan_favor_delta_vs_season.png",
        "week_mechanism_features.csv",
        "mechanism_test_results.csv",
    ]
    missing = [f for f in expected if not (out_dir / f).exists()]
    if missing:
        print(f"[warning] missing outputs: {missing}")

    print(f"DONE: {out_dir}")


if __name__ == "__main__":
    main()
