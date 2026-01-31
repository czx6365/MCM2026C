import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ===============================
# 工具函数
# ===============================

def parse_celebrity(pair_id: str) -> str:
    """从 pair_id 中稳健提取 celebrity 名字"""
    if not isinstance(pair_id, str):
        return pair_id
    pair_id = re.sub(r"^S\d+_", "", pair_id)
    return pair_id.split("&")[0].strip()


def assign_rank(df: pd.DataFrame, value_col: str, higher_is_better: bool = True) -> pd.Series:
    """确定性排名（值 + 名字作为 tie-break）"""
    ordered = df.sort_values(
        [value_col, "celebrity_name"],
        ascending=[not higher_is_better, True]
    )
    ranks = pd.Series(range(1, len(ordered) + 1), index=ordered.index)
    return ranks.reindex(df.index)


def select_worst(df: pd.DataFrame, value_col: str, n: int, higher_is_worse: bool) -> list:
    """选出最差的 n 个（论文口径清晰版）"""
    ordered = df.sort_values(
        [value_col, "celebrity_name"],
        ascending=[not higher_is_worse, True]
    )
    return ordered.head(n)["celebrity_name"].tolist()


# ===============================
# 主流程
# ===============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", default="data/process/model_input.csv")
    parser.add_argument("--elims", default="data/process/elimination_events.csv")
    parser.add_argument("--fan", default="t1/fan_vote_estimates.csv")
    parser.add_argument("--p-col", default="p_mean", choices=["p_mean", "p_map"])
    parser.add_argument("--out-dir", default="t2/out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # 读取数据
    # -------------------------------
    model = pd.read_csv(args.model_input)
    fan = pd.read_csv(args.fan)
    elims = pd.read_csv(args.elims)

    model["celebrity_name"] = model["pair_id"].apply(parse_celebrity)

    # -------------------------------
    # 淘汰事件严格对齐到“表演周”
    # elimination_events.week = 公布淘汰周
    # 实际决定淘汰的是 week - 1
    # -------------------------------
    elims_use = elims[[
        "season", "week", "elim_count",
        "eliminated_pair_ids", "notes"
    ]].copy()
    elims_use["week"] = elims_use["week"].astype(int) - 1
    elims_use = elims_use[elims_use["week"] > 0]

    # -------------------------------
    # 合并 fan vote 估计
    # -------------------------------
    fan_use = fan[["season", "week", "celebrity_name", args.p_col]].copy()
    fan_use = fan_use.rename(columns={args.p_col: "fan_share"})

    merged = (
        model
        .merge(fan_use, on=["season", "week", "celebrity_name"], how="left")
        .merge(elims_use, on=["season", "week"], how="left")
    )

    # -------------------------------
    # 周级完整性报告（不参与筛选，仅输出）
    # -------------------------------
    merged["has_elim"] = merged["elim_count"].fillna(0).astype(int) > 0
    week_status = (
        merged.groupby(["season", "week"], as_index=False)
        .agg(
            has_elim=("has_elim", "max"),
            any_missing_fan=("fan_share", lambda s: s.isna().any()),
            n_with_scores=("J_score", lambda s: (s > 0).sum())
        )
    )

    # -------------------------------
    # 周内分析（核心修复点）
    # 候选集合 = 当周 J_score > 0 的选手
    # 不使用 is_active
    # -------------------------------
    weekly_rows = []
    detailed_rows = []

    for (season, week), g in merged.groupby(["season", "week"]):
        g = g.copy()

        elim_n = int(g["elim_count"].fillna(0).iloc[0])
        if elim_n <= 0:
            continue

        # 候选集合：该表演周真实参与者
        g = g[g["J_score"] > 0].copy()
        if g.empty:
            continue

        # fan_share 必须完整
        if g["fan_share"].isna().any():
            continue

        # fan vote 周内归一化（稳健）
        g["fan_share"] = g["fan_share"] / g["fan_share"].sum()

        # ---------------------------
        # Rank method
        # ---------------------------
        g["judge_rank"] = assign_rank(g, "J_score", higher_is_better=True)
        g["fan_rank"] = assign_rank(g, "fan_share", higher_is_better=True)
        g["rank_total"] = g["judge_rank"] + g["fan_rank"]

        # ---------------------------
        # Percent method
        # ---------------------------
        total_j = g["J_score"].sum()
        g["judge_pct"] = g["J_score"] / total_j
        g["combined_pct"] = g["judge_pct"] + g["fan_share"]

        # ---------------------------
        # 淘汰预测
        # ---------------------------
        rank_elims = select_worst(g, "rank_total", elim_n, higher_is_worse=True)
        pct_elims = select_worst(g, "combined_pct", elim_n, higher_is_worse=False)

        rank_bottom_two = select_worst(g, "rank_total", 2, higher_is_worse=True)
        pct_bottom_two = select_worst(g, "combined_pct", 2, higher_is_worse=False)

        def judge_save(bottom_two):
            sub = g[g["celebrity_name"].isin(bottom_two)].copy()
            if sub.empty:
                return None
            # judges save：分低者被淘汰
            return sub.sort_values(
                ["J_score", "celebrity_name"],
                ascending=[True, True]
            ).iloc[0]["celebrity_name"]

        rank_js = judge_save(rank_bottom_two) if elim_n == 1 else ""
        pct_js = judge_save(pct_bottom_two) if elim_n == 1 else ""

        # 实际淘汰人
        actual_names = []
        ids = g["eliminated_pair_ids"].iloc[0]
        if isinstance(ids, str) and ids.strip():
            for pid in ids.split("|"):
                actual_names.append(parse_celebrity(pid.strip()))

        weekly_rows.append({
            "season": season,
            "week": week,
            "elim_count": elim_n,
            "rank_eliminated": "|".join(rank_elims),
            "percent_eliminated": "|".join(pct_elims),
            "rank_bottom_two": "|".join(rank_bottom_two),
            "percent_bottom_two": "|".join(pct_bottom_two),
            "rank_judge_save_elim": rank_js,
            "percent_judge_save_elim": pct_js,
            "actual_eliminated": "|".join(actual_names),
        })

        detailed_rows.append(g)

    weekly = pd.DataFrame(weekly_rows)
    detailed = pd.concat(detailed_rows, ignore_index=True) if detailed_rows else pd.DataFrame()

    # -------------------------------
    # 输出
    # -------------------------------
    weekly.to_csv(out_dir / "weekly_method_comparison.csv", index=False)
    detailed.to_csv(out_dir / "weekly_contestant_metrics.csv", index=False)
    week_status.to_csv(out_dir / "week_data_completeness.csv", index=False)

    if not weekly.empty:
        weekly["methods_differ"] = (
            weekly["rank_eliminated"] != weekly["percent_eliminated"]
        )
        (
            weekly.groupby("season", as_index=False)
            .agg(
                weeks_with_elim=("week", "count"),
                diff_weeks=("methods_differ", "sum")
            )
            .assign(diff_share=lambda d: d["diff_weeks"] / d["weeks_with_elim"])
            .to_csv(out_dir / "season_method_diff_summary.csv", index=False)
        )


if __name__ == "__main__":
    main()
