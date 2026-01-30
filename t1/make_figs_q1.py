#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # 读取你脚本输出的结果（如路径不同请自行修改）
    fan = pd.read_csv(r"t1\fan_vote_estimates.csv")
    weekly = pd.read_csv(r"t1\weekly_summary.csv")
    out_dir = r"t1\图"
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Figure 1: accept_rate 分布（可辨识性/信息量）
    # -----------------------------
    x = weekly["accept_rate"].dropna().to_numpy()

    plt.figure(figsize=(7, 4))
    plt.hist(x, bins=30)
    plt.xlabel("ABC accept_rate")
    plt.ylabel("Number of weeks")
    plt.title("Distribution of feasible posterior volume (accept_rate)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Fig1_accept_rate_hist.png"), dpi=300)

    # -----------------------------
    # Figure 2: margin vs uncertainty（x=margin, y=week_uncertainty）
    # week_uncertainty = 每周 rel_ci_width 的均值
    # -----------------------------
    agg = (
        fan.groupby(["season", "week"])
           .agg(week_uncertainty=("rel_ci_width", "mean"))
           .reset_index()
    )
    weekly2 = weekly.drop(columns=["week_uncertainty"], errors="ignore").merge(agg, on=["season", "week"], how="left")

    # 只画有 margin_map 的周（k=0 的无淘汰周 margin 为 NaN）
    plot_df = weekly2[~weekly2["margin_map"].isna() & ~weekly2["week_uncertainty"].isna()].copy()

    plt.figure(figsize=(7, 4))
    for sch, sub in plot_df.groupby("scheme"):
        plt.scatter(sub["margin_map"], sub["week_uncertainty"], s=18, alpha=0.75, label=sch)

    # 简单线性回归（全样本）
    if plot_df.shape[0] >= 2:
        x = plot_df["margin_map"].to_numpy()
        y = plot_df["week_uncertainty"].to_numpy()
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ys = coef[0] * xs + coef[1]
        plt.plot(xs, ys, color="black", linewidth=1.5, label="Linear fit")

    plt.xlabel("margin_map (elimination boundary gap)")
    plt.ylabel("Weekly mean relative CI width")
    plt.title("Margin vs uncertainty (weekly mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Fig2_margin_vs_uncertainty.png"), dpi=300)

    # -----------------------------
    # Figure 3: pp_consistency 随周次变化（全局索引）
    # -----------------------------
    if "pp_consistency" in weekly2.columns:
        wk = weekly2.sort_values(["season", "week"]).reset_index(drop=True)
        x = np.arange(len(wk))
        plt.figure(figsize=(10, 4))
        for sch, sub in wk.groupby("scheme"):
            idx = sub.index.to_numpy()
            plt.scatter(idx, sub["pp_consistency"], s=14, alpha=0.75, label=sch)
        plt.xlabel("Global week index (sorted by season, week)")
        plt.ylabel("Posterior predictive consistency")
        plt.title("pp_consistency over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Fig3_pp_consistency_by_week.png"), dpi=300)

        # Heatmap: season x week
        heat = wk.pivot_table(index="season", columns="week", values="pp_consistency", aggfunc="mean")
        plt.figure(figsize=(10, 6))
        data = heat.to_numpy()
        masked = np.ma.masked_invalid(data)
        plt.imshow(masked, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(label="pp_consistency")
        plt.xlabel("Week")
        plt.ylabel("Season")
        plt.title("pp_consistency heatmap (season x week)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Fig4_pp_consistency_heatmap.png"), dpi=300)

    # -----------------------------
    # Figure 5: Rank 均值失配周的 p_mean vs p_map（排序非线性示例）
    # -----------------------------
    mismatch = weekly2[(weekly2["scheme"] == "Rank") & (weekly2["consistency_mean"] == 0)].copy()
    if mismatch.shape[0] == 0:
        print("[提示] 未发现 Rank 下的均值失配周，跳过 Fig5。")
        mismatch = None

    if mismatch is not None:
        # 你当前数据应只有 1 个失配周
        s = int(mismatch.iloc[0]["season"])
        w = int(mismatch.iloc[0]["week"])
        elim_set = str(mismatch.iloc[0]["elim_set"])

        sub = fan[(fan["season"] == s) & (fan["week"] == w)].copy()

        # 让 x 轴更有意义：按 p_mean 从大到小排序
        sub = sub.sort_values("p_mean", ascending=False).reset_index(drop=True)

        labels = sub["celebrity_name"].tolist()
        xpos = np.arange(len(labels))
        width = 0.40

        plt.figure(figsize=(10, 4))
        plt.bar(xpos - width/2, sub["p_mean"].to_numpy(), width=width, label="p_mean")
        plt.bar(xpos + width/2, sub["p_map"].to_numpy(),  width=width, label="p_map")
        plt.xticks(xpos, labels, rotation=45, ha="right")
        plt.ylabel("Fan vote share")
        plt.title(f"Case study: mean vs MAP (Season {s}, Week {w}), elim={elim_set}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Fig5_case_mean_vs_map.png"), dpi=300)

        # （可选加强版）再做一张“fan-rank 的变化图”，更直观解释 mean mismatch
        sub["fan_rank_mean"] = sub["p_mean"].rank(ascending=False, method="average")
        sub["fan_rank_map"]  = sub["p_map"].rank(ascending=False, method="average")

        plt.figure(figsize=(10, 4))
        plt.plot(xpos, sub["fan_rank_mean"].to_numpy(), marker="o", label="fan-rank from p_mean")
        plt.plot(xpos, sub["fan_rank_map"].to_numpy(),  marker="o", label="fan-rank from p_map")
        plt.gca().invert_yaxis()  # 1 = best
        plt.xticks(xpos, labels, rotation=45, ha="right")
        plt.ylabel("Fan rank (1 = highest votes)")
        plt.title(f"Rank nonlinearity: fan-rank shift (Season {s}, Week {w})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Fig5b_case_rank_shift.png"), dpi=300)

    print(f"已输出到 {out_dir}：Fig1_accept_rate_hist.png, Fig2_margin_vs_uncertainty.png, "
          "Fig3_pp_consistency_by_week.png, Fig4_pp_consistency_heatmap.png, "
          "Fig5_case_mean_vs_map.png, Fig5b_case_rank_shift.png")


if __name__ == "__main__":
    main()
