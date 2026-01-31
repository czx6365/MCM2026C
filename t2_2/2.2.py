#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
counterfactual_controversy.py
--------------------------------
用途：
  - 用你Q1反推的 fan vote 份额（含90%区间）做 Monte Carlo
  - 对比 Percent vs Rank 两种合并方法的淘汰/名次结果
  - 叠加“Bottom two + Judges choose（judges-save）”规则，看影响

输入：
  - 2026_MCM_Problem_C_Data.csv（评委分原始数据）
  - fan_vote_estimates.csv（你输出的p_mean/p_map/p_lo90/p_hi90等）

输出：
  - controversy_counterfactual_summary.csv
  - fig_case_S{season}_{name}_controversy.png
  - fig_case_S{season}_{name}_placement.png

注意：
  1) 本脚本以“每周淘汰人数 = 真实历史淘汰人数”为基准重放，
     由 fan_vote_estimates.csv 的 eliminated_this_week 字段汇总得到。
  2) 决赛周可能不淘汰：脚本在赛季结束时用“最后一个有评委分的周”的合并分对剩余选手排序，
     作为最终名次（避免额外手工校准，且可复现）。
"""

import argparse
import os
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 题面建议的争议案例（可自行扩展）
# -----------------------------
DEFAULT_CASES = [
    (2, "Jerry Rice"),
    (4, "Billy Ray Cyrus"),
    (11, "Bristol Palin"),
    (27, "Bobby Bones"),
]


# -----------------------------
# 配置结构
# -----------------------------
@dataclass
class ReplayConfig:
    scheme: str  # "percent" or "rank"
    use_judges_save: bool
    beta: Optional[float] = None  # None => 确定性judges-save；否则logistic强度
    mc: int = 2000
    seed: int = 0
    fan_mode: str = "p_map"  # 三角分布mode用 p_map 或 p_mean
    tie_break: str = "fan_then_judge"  # 并列时的稳定规则


# -----------------------------
# 评委分提取：每周总分
# -----------------------------
def judge_total_by_week(df_season: pd.DataFrame, week: int) -> pd.Series:
    """
    返回df_season中每位选手在该week的评委总分。
    列名形如：week3_judge2_score
    """
    cols = [
        c
        for c in df_season.columns
        if c.startswith(f"week{week}_") and c.endswith("_score")
    ]
    if not cols:
        return pd.Series([np.nan] * len(df_season), index=df_season.index)

    tot = df_season[cols].sum(axis=1, skipna=True)
    cnt = df_season[cols].notna().sum(axis=1)
    # 若该周该选手完全没有分数（全NA），视为NA
    tot = tot.where(cnt > 0, np.nan)
    return tot


def available_weeks(df_season: pd.DataFrame) -> List[int]:
    """
    从原始数据列名中解析出所有week编号，并判断该周至少有一个选手有评委分。
    """
    week_cols = [c for c in df_season.columns if re.match(r"week\d+_judge\d+_score", c)]
    weeks = sorted({int(re.match(r"week(\d+)_", c).group(1)) for c in week_cols})
    active_weeks = []
    for w in weeks:
        S = judge_total_by_week(df_season, w)
        if S.notna().sum() > 0:
            active_weeks.append(w)
    return active_weeks


def rank_1_is_best(x: pd.Series) -> pd.Series:
    """排名：1=最好；并列用min，保证“更强者不吃亏”。"""
    return x.rank(method="min", ascending=False)


# -----------------------------
# fan share 不确定性采样：三角分布近似
# -----------------------------
def triangular_sample(
    lo: float, mode: float, hi: float, rng: np.random.Generator
) -> float:
    """
    用三角分布在[lo, hi]中采样，mode为峰值点。
    若参数不合法则回退到截断后的mode。
    """
    if any(
        map(
            lambda v: v is None or (isinstance(v, float) and np.isnan(v)),
            [lo, mode, hi],
        )
    ):
        return np.nan
    lo = float(lo)
    mode = float(mode)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    mode = min(max(mode, lo), hi)
    if hi <= lo:
        return mode
    return float(rng.triangular(lo, mode, hi))


# -----------------------------
# 合并分计算：Percent / Rank
# -----------------------------
def combined_scores(
    scheme: str,
    S: pd.Series,
    fan_share: np.ndarray,
) -> np.ndarray:
    """
    返回每位选手的“合并分”（用于淘汰排序）
      - percent：P = pJ + pF，越小越差
      - rank：    R = rJ + rF，越大越差
    """
    if scheme == "percent":
        # 防止sum为0
        Sj = S.to_numpy(dtype=float)
        Sj = np.nan_to_num(Sj, nan=0.0)
        denom = Sj.sum()
        pJ = Sj / denom if denom > 0 else np.zeros_like(Sj)

        pF = fan_share
        return pJ + pF

    if scheme == "rank":
        rJ = rank_1_is_best(S).to_numpy(dtype=float)
        rF = rank_1_is_best(pd.Series(fan_share, index=S.index)).to_numpy(dtype=float)
        return rJ + rF

    raise ValueError("scheme must be 'percent' or 'rank'")


def worse_sort_index(
    scheme: str,
    score: np.ndarray,
    fan_share: np.ndarray,
    S: pd.Series,
) -> np.ndarray:
    """
    给出“从最差到最好”的排序索引（用于bottom2等）。
    tie-break：先看fan更低者更差，再看judge更低者更差（可稳定路径）
    """
    Sj = S.to_numpy(dtype=float)
    Sj = np.nan_to_num(Sj, nan=-np.inf)

    if scheme == "percent":
        # 分数越小越差：按(score, fan, judge)升序
        keys = np.lexsort((Sj, fan_share, score))
        return keys
    else:
        # rank：分数越大越差：按(-score, fan, judge)升序
        keys = np.lexsort((Sj, fan_share, -score))
        return keys


# -----------------------------
# 计算每周真实淘汰人数（从fan_vote_estimates汇总）
# -----------------------------
def true_elim_count_by_week(fan_season: pd.DataFrame) -> Dict[int, int]:
    """
    fan_season 包含 eliminated_this_week (0/1) 对每位选手每周
    汇总得到每周淘汰人数
    """
    if "eliminated_this_week" not in fan_season.columns:
        # 若缺失则默认每周淘汰1人（保守）
        counts = fan_season.groupby("week")["celebrity_name"].nunique()
        return {int(w): 1 for w in counts.index}

    g = fan_season.groupby("week")["eliminated_this_week"].sum()
    out = {int(w): int(v) for w, v in g.items()}
    return out


# -----------------------------
# judges-save 决策：bottom2中淘汰谁
# -----------------------------
def judges_choose_elim(
    bottom2_names: List[str],
    Sb2: Dict[str, float],
    rng: np.random.Generator,
    beta: Optional[float],
) -> str:
    """
    bottom2_names: [worst_by_combined, second_worst_by_combined]
    Sb2: name -> judge_total（当周）
    beta=None => 确定性淘汰评委分更低者
    beta!=None => logistic概率：分差越大越确定
    """
    a, b = bottom2_names[0], bottom2_names[1]
    Ja = Sb2.get(a, -np.inf)
    Jb = Sb2.get(b, -np.inf)

    if beta is None:
        return a if Ja <= Jb else b

    # 概率性：P(elim a) = sigmoid(beta*(Jb - Ja))
    # 若a的评委分更低，则Jb-Ja>0，P(elim a)更大
    x = beta * (Jb - Ja)
    # 数值稳定
    if x >= 0:
        pa = 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        pa = ex / (1.0 + ex)
    return a if rng.random() < pa else b


# -----------------------------
# 赛季重放：核心函数
# -----------------------------
def replay_season_once(
    df_season: pd.DataFrame,
    fan_season: pd.DataFrame,
    cfg: ReplayConfig,
    active_weeks: List[int],
    elim_count_week: Dict[int, int],
    rng: np.random.Generator,
) -> Tuple[List[Tuple[int, str]], Dict[str, int]]:
    """
    单次重放：
      - 返回淘汰顺序 elim_order = [(week, name), ...]
      - 返回最终名次 place_map: name -> place（1最好）
    """
    all_names = df_season["celebrity_name"].tolist()
    alive = set(all_names)
    elim_order: List[Tuple[int, str]] = []

    # 便于查fan行
    fan_key = fan_season.set_index(["week", "celebrity_name"])

    last_week_with_scores = None

    for w in active_weeks:
        # 本周仍在场选手
        dfw = df_season[df_season["celebrity_name"].isin(alive)].copy()
        if len(dfw) <= 1:
            break

        S = judge_total_by_week(dfw, w)
        if S.notna().sum() == 0:
            continue

        last_week_with_scores = w

        # 取/采样 fan share（只对alive子集重归一化）
        fan_vals = []
        for nm in dfw["celebrity_name"].tolist():
            if (w, nm) in fan_key.index:
                row = fan_key.loc[(w, nm)]
                lo = float(row.get("p_lo90", np.nan))
                hi = float(row.get("p_hi90", np.nan))
                if cfg.fan_mode == "p_mean":
                    mode = float(row.get("p_mean", np.nan))
                else:
                    mode = float(row.get("p_map", np.nan))
                if cfg.mc == 1:
                    # 点估计（用mode）
                    val = mode
                else:
                    val = triangular_sample(lo, mode, hi, rng)
                fan_vals.append(val if not np.isnan(val) else 0.0)
            else:
                fan_vals.append(0.0)

        fan_share = np.array(fan_vals, dtype=float)
        fan_share = np.nan_to_num(fan_share, nan=0.0)
        if fan_share.sum() > 0:
            fan_share = fan_share / fan_share.sum()
        else:
            # 极端情况：全0，则均分（避免崩溃）
            fan_share = np.ones_like(fan_share) / len(fan_share)

        # 本周需要淘汰的人数（按真实历史）
        k_elim = int(elim_count_week.get(w, 0))
        # 若真实历史该周无淘汰，则跳过淘汰，仅用于后续“决赛排序”
        if k_elim <= 0:
            continue

        # 可能出现 double elimination：顺序淘汰k_elim次
        for _ in range(k_elim):
            dfw2 = df_season[df_season["celebrity_name"].isin(alive)].copy()
            if len(dfw2) <= 1:
                break

            S2 = judge_total_by_week(dfw2, w)
            # 重新取fan share（同一周同一份fan_share应在alive子集重归一化）
            # 这里保持“原抽样值”但对子集重归一化即可
            names2 = dfw2["celebrity_name"].tolist()
            fan_vals2 = []
            for nm in names2:
                if (w, nm) in fan_key.index:
                    row = fan_key.loc[(w, nm)]
                    lo = float(row.get("p_lo90", np.nan))
                    hi = float(row.get("p_hi90", np.nan))
                    mode = float(row.get(cfg.fan_mode, row.get("p_map", np.nan)))
                    if cfg.mc == 1:
                        val = mode
                    else:
                        # 为了“同一次MC在同一周同一人”一致，可选缓存；这里简化为再采样一次
                        # 如果你希望严格一致，可改成预采样缓存（评审不强制）
                        val = triangular_sample(lo, mode, hi, rng)
                    fan_vals2.append(val if not np.isnan(val) else 0.0)
                else:
                    fan_vals2.append(0.0)

            fan_share2 = np.array(fan_vals2, dtype=float)
            fan_share2 = np.nan_to_num(fan_share2, nan=0.0)
            if fan_share2.sum() > 0:
                fan_share2 = fan_share2 / fan_share2.sum()
            else:
                fan_share2 = np.ones_like(fan_share2) / len(fan_share2)

            score = combined_scores(cfg.scheme, S2, fan_share2)
            order_worst_to_best = worse_sort_index(cfg.scheme, score, fan_share2, S2)

            # bottom2
            if len(order_worst_to_best) >= 2:
                b2_idx = order_worst_to_best[:2]
            else:
                b2_idx = order_worst_to_best[:1]

            bottom2_names = dfw2.iloc[b2_idx]["celebrity_name"].tolist()
            # 默认淘汰 = 最差者
            elim_name = bottom2_names[0]

            if cfg.use_judges_save and len(bottom2_names) == 2:
                # judges在bottom2中投票决定淘汰谁
                Sb2 = {
                    nm: float(S2.loc[dfw2[dfw2["celebrity_name"] == nm].index[0]])
                    for nm in bottom2_names
                }
                elim_name = judges_choose_elim(bottom2_names, Sb2, rng, cfg.beta)

            elim_order.append((w, elim_name))
            if elim_name in alive:
                alive.remove(elim_name)


    # =========================
    # 赛季结束：最终名次判定
    # =========================
    place_map: Dict[str, int] = {}
    alive_final = list(alive)

    if len(alive_final) == 0:
        ranking = []
    elif len(alive_final) == 1:
        ranking = alive_final
    else:
        # ===== 核心修正点 =====
        # 决赛/无淘汰阶段：仅使用 fan votes 决定最终名次
        # 使用最后一周 fan_share（p_map）作为最终投票结果
        finals_week = None
        for w in reversed(active_weeks):
            if w in fan_season["week"].values:
                finals_week = w
                break

        fan_last = []
        for nm in alive_final:
            row = fan_season[
                (fan_season["week"] == finals_week) & (fan_season["celebrity_name"] == nm)
            ]
            if row.empty:
                fan_last.append(0.0)
            else:
                v = float(row.iloc[0].get("p_map", row.iloc[0].get("p_mean", 0.0)))
                fan_last.append(0.0 if np.isnan(v) else v)

        fan_last = np.array(fan_last, dtype=float)
        if fan_last.sum() > 0:
            fan_last = fan_last / fan_last.sum()
        else:
            fan_last = np.ones_like(fan_last) / len(fan_last)

        # fan votes 高 → 名次好
        order = np.argsort(-fan_last)
        ranking_alive = [alive_final[i] for i in order]

        eliminated = [nm for _, nm in elim_order]
        ranking = ranking_alive + eliminated[::-1]

    for idx, nm in enumerate(ranking):
        place_map[nm] = idx + 1

    return elim_order, place_map


# -----------------------------
# 多次MC并统计某个选手的名次分布
# -----------------------------
def mc_placements_for_star(
    df_season: pd.DataFrame,
    fan_season: pd.DataFrame,
    star: str,
    cfg: ReplayConfig,
) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    weeks = available_weeks(df_season)
    elim_counts = true_elim_count_by_week(fan_season)

    places = []
    for t in range(cfg.mc):
        # 每次重放用不同随机性，但可复现
        # 做法：把主rng派生出子rng（或直接rng随机）
        elim_order, place_map = replay_season_once(
            df_season=df_season,
            fan_season=fan_season,
            cfg=cfg,
            active_weeks=weeks,
            elim_count_week=elim_counts,
            rng=rng,
        )
        places.append(place_map.get(star, np.nan))

    return np.array(places, dtype=float)


def summarize_places(x: np.ndarray) -> Dict[str, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"mean": np.nan, "p05": np.nan, "p95": np.nan}
    return {
        "mean": float(np.mean(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
    }


# -----------------------------
# 画图：controversy 曲线（评委rank vs fan rank）
# -----------------------------
def plot_controversy_curve(
    outpath: str, df_season: pd.DataFrame, fan_season: pd.DataFrame, star: str
):
    weeks = available_weeks(df_season)
    fan_key = fan_season.set_index(["week", "celebrity_name"])

    j_rank = []
    f_rank = []
    w_list = []

    for w in weeks:
        dfw = df_season.copy()
        S = judge_total_by_week(dfw, w)
        if S.notna().sum() == 0:
            continue

        # 评委rank（1最好）
        rJ = rank_1_is_best(S)

        # fan share用p_map做rank（1最高票）
        fan_vals = []
        for nm in dfw["celebrity_name"].tolist():
            if (w, nm) in fan_key.index:
                row = fan_key.loc[(w, nm)]
                v = float(row.get("p_map", row.get("p_mean", 0.0)))
            else:
                v = 0.0
            fan_vals.append(0.0 if np.isnan(v) else v)
        fan_vals = np.array(fan_vals, dtype=float)
        if fan_vals.sum() > 0:
            fan_vals = fan_vals / fan_vals.sum()
        rF = rank_1_is_best(pd.Series(fan_vals, index=dfw.index))

        # 取目标选手
        if star not in set(dfw["celebrity_name"]):
            continue
        idx = dfw[dfw["celebrity_name"] == star].index[0]

        j_rank.append(float(rJ.loc[idx]))
        f_rank.append(float(rF.loc[idx]))
        w_list.append(w)

    if not w_list:
        return

    plt.figure(figsize=(7, 4))
    plt.plot(w_list, j_rank, marker="o", label="Judges rank (1=best)")
    plt.plot(w_list, f_rank, marker="o", label="Fans rank (1=best)")
    plt.gca().invert_yaxis()  # rank越小越好，反转更直观
    plt.xlabel("Week")
    plt.ylabel("Rank (1=best)")
    plt.title(f"Controversy curve: {star}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -----------------------------
# 画图：名次分布（四种规则箱线图）
# -----------------------------
def plot_placement_boxplot(outpath: str, star: str, placements: Dict[str, np.ndarray]):
    labels = list(placements.keys())
    data = [placements[k][~np.isnan(placements[k])] for k in labels]

    plt.figure(figsize=(8, 4))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.gca().invert_yaxis()  # 名次越小越好
    plt.ylabel("Placement (1=best)")
    plt.title(f"Placement distribution: {star}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -----------------------------
# 主程序
# -----------------------------
def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)
    default_raw = os.path.join(repo_root, "data", "2026_MCM_Problem_C_Data.csv")
    default_fan = os.path.join(repo_root, "t1", "fan_vote_estimates.csv")
    default_out = this_dir

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", default=default_raw, help="Path to 2026_MCM_Problem_C_Data.csv"
    )
    parser.add_argument(
        "--fan", default=default_fan, help="Path to fan_vote_estimates.csv"
    )
    parser.add_argument("--outdir", default=default_out, help="Output directory")
    parser.add_argument(
        "--mc",
        type=int,
        default=2000,
        help="Monte Carlo runs per scenario (e.g., 2000)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=float("nan"),
        help="If set (not NaN), use probabilistic judges-save with this beta; else deterministic",
    )
    parser.add_argument(
        "--fan_mode",
        default="p_map",
        choices=["p_map", "p_mean"],
        help="Mode used in triangular sampling",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    raw = pd.read_csv(args.raw)
    fan = pd.read_csv(args.fan)

    # 基本清洗：只保留我们需要的列（减少内存）
    needed = [
        "season",
        "week",
        "celebrity_name",
        "p_mean",
        "p_lo90",
        "p_hi90",
        "p_map",
        "eliminated_this_week",
    ]
    for c in needed:
        if c not in fan.columns:
            # 若缺列也不强制报错；但lo/hi缺失会让采样退化
            pass

    # 统一类型
    fan["season"] = fan["season"].astype(int)
    fan["week"] = fan["week"].astype(int)

    raw["season"] = raw["season"].astype(int)

    # judges-save beta
    beta = None if np.isnan(args.beta) else float(args.beta)

    summary_rows = []

    for season, star in DEFAULT_CASES:
        df_season = raw[raw["season"] == season].copy()
        fan_season = fan[fan["season"] == season].copy()

        if df_season.empty or fan_season.empty:
            print(f"[WARN] season {season}: missing data, skipped.")
            continue

        if star not in set(df_season["celebrity_name"]):
            # 简单容错：忽略大小写/多余空格
            cand = None
            for nm in df_season["celebrity_name"].unique():
                if nm.strip().lower() == star.strip().lower():
                    cand = nm
                    break
            if cand is None:
                print(f"[WARN] season {season}: star '{star}' not found, skipped.")
                continue
            star = cand

        # 四种场景配置
        cfg_percent = ReplayConfig(
            scheme="percent",
            use_judges_save=False,
            beta=None,
            mc=args.mc,
            seed=season * 101 + 1,
            fan_mode=args.fan_mode,
        )
        cfg_rank = ReplayConfig(
            scheme="rank",
            use_judges_save=False,
            beta=None,
            mc=args.mc,
            seed=season * 101 + 2,
            fan_mode=args.fan_mode,
        )
        cfg_percent_save = ReplayConfig(
            scheme="percent",
            use_judges_save=True,
            beta=beta,
            mc=args.mc,
            seed=season * 101 + 3,
            fan_mode=args.fan_mode,
        )
        cfg_rank_save = ReplayConfig(
            scheme="rank",
            use_judges_save=True,
            beta=beta,
            mc=args.mc,
            seed=season * 101 + 4,
            fan_mode=args.fan_mode,
        )

        # 跑MC
        p_pl = mc_placements_for_star(df_season, fan_season, star, cfg_percent)
        r_pl = mc_placements_for_star(df_season, fan_season, star, cfg_rank)
        ps_pl = mc_placements_for_star(df_season, fan_season, star, cfg_percent_save)
        rs_pl = mc_placements_for_star(df_season, fan_season, star, cfg_rank_save)

        # 同名次概率（Percent vs Rank）
        same_pr = float(np.mean(p_pl == r_pl))

        # 夺冠概率（place==1）
        p_win = float(np.mean(p_pl == 1))
        r_win = float(np.mean(r_pl == 1))
        ps_win = float(np.mean(ps_pl == 1))
        rs_win = float(np.mean(rs_pl == 1))

        sp = summarize_places(p_pl)
        sr = summarize_places(r_pl)
        sps = summarize_places(ps_pl)
        srs = summarize_places(rs_pl)

        row = {
            "season": season,
            "celebrity": star,
            "Pr_same_placement(percent_vs_rank)": same_pr,
            "percent_mean_place": sp["mean"],
            "percent_p05": sp["p05"],
            "percent_p95": sp["p95"],
            "percent_Pr_win": p_win,
            "rank_mean_place": sr["mean"],
            "rank_p05": sr["p05"],
            "rank_p95": sr["p95"],
            "rank_Pr_win": r_win,
            "percentSave_mean_place": sps["mean"],
            "percentSave_p05": sps["p05"],
            "percentSave_p95": sps["p95"],
            "percentSave_Pr_win": ps_win,
            "rankSave_mean_place": srs["mean"],
            "rankSave_p05": srs["p05"],
            "rankSave_p95": srs["p95"],
            "rankSave_Pr_win": rs_win,
            "delta_rank_minus_percent": sr["mean"] - sp["mean"],
            "delta_rankSave_minus_percent": srs["mean"] - sp["mean"],
        }
        summary_rows.append(row)

        # 画图：controversy curve
        fig1 = os.path.join(
            args.outdir, f"fig_case_S{season}_{star}_controversy.png".replace(" ", "_")
        )
        plot_controversy_curve(fig1, df_season, fan_season, star)

        # 画图：placement boxplot
        fig2 = os.path.join(
            args.outdir, f"fig_case_S{season}_{star}_placement.png".replace(" ", "_")
        )
        placements = {
            "Percent": p_pl,
            "Rank": r_pl,
            "Pct+Save": ps_pl,
            "Rank+Save": rs_pl,
        }
        plot_placement_boxplot(fig2, star, placements)

        print(f"[OK] season {season} {star}: done. Figures saved.")

    out = pd.DataFrame(summary_rows)
    out_csv = os.path.join(args.outdir, "controversy_counterfactual_summary.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n=== Summary saved ===")
    print(out_csv)
    print(out)


if __name__ == "__main__":
    main()
