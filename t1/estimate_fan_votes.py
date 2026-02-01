#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端估计 DWTS 每周 fan vote share 的脚本（2026 MCM Problem C Q1）
升级版：更鲁棒的任务构造 + 更合理的 fallback + p_map 保证一致性
"""

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Tuple, Set, Any, Optional

import numpy as np
import pandas as pd


# -----------------------------
# 基础工具与数据解析
# -----------------------------


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="估计每周 fan vote share (p_{i,t})")
    parser.add_argument("--data_path", type=str, default="/mnt/data/2026_MCM_Problem_C_Data.csv",
                        help="数据文件路径（默认 /mnt/data/2026_MCM_Problem_C_Data.csv）")
    parser.add_argument("--alpha0", type=float, default=5.0, help="Dirichlet 初始先验系数")
    parser.add_argument("--kappa", type=float, default=40.0, help="动态先验强度系数（越大越平滑）")
    parser.add_argument("--n_accept", type=int, default=500, help="每周 ABC 目标接受样本数")
    parser.add_argument("--draw_max", type=int, default=50000, help="每周 ABC 最大抽样次数")
    parser.add_argument("--draw_max_cap", type=int, default=200000, help="fallback 时允许的 draw_max 上限")
    parser.add_argument("--T", type=float, default=1.0, help="总票数 T_t（用于换算 V_mean）")
    parser.add_argument("--seed", type=int, default=1234, help="全局随机种子")
    parser.add_argument("--soft_eps", type=float, default=float("nan"),
                        help="Soft ABC kernel epsilon；若为 NaN 则每周自适应（median distance）")
    parser.add_argument("--soft_delta", type=float, default=1e-12,
                        help="Rank 赛制连续 proxy 中 log(p+delta) 的 delta")
    parser.add_argument("--post28_mode", type=str, default="rank_only",
                        choices=["rank_only"],
                        help="Season>=28 的处理模式（预留接口；当前仅支持 rank_only）")
    return parser.parse_args()


def locate_data_path(user_path: str) -> str:
    """
    定位数据文件：
    1) 若用户给的路径存在，直接用；
    2) 否则尝试当前目录下 data/2026_MCM_Problem_C_Data.csv。
    """
    if os.path.exists(user_path):
        return user_path
    fallback = os.path.join(os.getcwd(), "data", "2026_MCM_Problem_C_Data.csv")
    if os.path.exists(fallback):
        print(f"[提示] 未找到 {user_path}，改用 {fallback}")
        return fallback
    raise FileNotFoundError(f"未找到数据文件：{user_path}")


def extract_week_columns(columns: List[str]) -> Dict[int, List[str]]:
    """从列名中解析 weekX_judgeY_score 的分组。"""
    pattern = re.compile(r"week(\d+)_judge\d+_score")
    week_cols: Dict[int, List[str]] = {}
    for col in columns:
        m = pattern.match(col)
        if m:
            week = int(m.group(1))
            week_cols.setdefault(week, []).append(col)
    return dict(sorted(week_cols.items(), key=lambda x: x[0]))


def scheme_for_season(season: int) -> str:
    """返回赛制类型：Rank 或 Percent。"""
    if season in (1, 2) or season >= 28:
        return "Rank"
    return "Percent"


def stable_seed(base_seed: int, season: int, week: int, tag: str) -> int:
    """稳定的 32-bit seed，避免 Python hash 随机化导致不可复现。"""
    payload = f"{base_seed}|{season}|{week}|{tag}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, "little")


def compute_week_judge_totals(df: pd.DataFrame, week_cols: Dict[int, List[str]]) -> pd.DataFrame:
    """计算每周评委总分 J_{i,t}。"""
    out = df.copy()
    for week, cols in week_cols.items():
        # 评委列可能有 NaN，sum(skipna=True) 会自动忽略
        out[f"week{week}_judge_total"] = out[cols].sum(axis=1, skipna=True, min_count=1)
    return out


def build_week_tasks(df: pd.DataFrame, week_cols: Dict[int, List[str]]) -> List[Dict[str, Any]]:
    """
    生成每个赛季、每周的推断任务（完全数据驱动）：
    - 活跃选手集合 S_{s,t}：当周 judge_total > 0
    - 淘汰集合 E_{s,t} = S_{s,t} \\ S_{s,t+1}
    - 仅保留 t+1 仍有活跃选手的周（排除决赛后一周全 0）
    """
    weeks = sorted(week_cols.keys())
    tasks: List[Dict[str, Any]] = []

    for season, sdf in df.groupby("season"):
        sdf = sdf.reset_index(drop=True)
        names_all = sdf["celebrity_name"].astype(str).tolist()

        # 每周活跃集合（只存 active 名单）
        active_by_week: Dict[int, List[str]] = {}
        for week in weeks:
            total_col = f"week{week}_judge_total"
            totals = sdf[total_col].fillna(0.0).astype(float).to_numpy()
            active_mask = totals > 0.0
            active_names = [names_all[i] for i in range(len(names_all)) if active_mask[i]]
            active_by_week[week] = active_names

        # 生成任务：E_{t}=S_t \ S_{t+1}
        for i, week in enumerate(weeks[:-1]):
            next_week = weeks[i + 1]

            # 若下一周无人活跃，通常是决赛后归零，跳过
            if len(active_by_week[next_week]) == 0:
                continue

            active = active_by_week[week]
            if len(active) == 0:
                continue

            elim = set(active) - set(active_by_week[next_week])

            # 当周评委分：只对 active 选手建 dict（更稳、更省内存）
            total_col = f"week{week}_judge_total"
            totals_series = sdf.set_index("celebrity_name")[total_col].fillna(0.0).astype(float)
            judge_totals = {nm: float(totals_series.get(nm, 0.0)) for nm in active}

            tasks.append({
                "season": int(season),
                "week": int(week),
                "names": active,
                "judge_totals": judge_totals,
                "elim_set": elim,
                "scheme": scheme_for_season(int(season)),
            })

    tasks.sort(key=lambda x: (x["season"], x["week"]))
    return tasks


# -----------------------------
# 赛制计算（纯函数）
# -----------------------------


def rank_average_desc(values: List[float]) -> np.ndarray:
    """并列取平均名次的排名（高者为 1）。"""
    s = pd.Series(values)
    return s.rank(ascending=False, method="average").to_numpy()


def compute_elim_percent(names: List[str], judge_totals: Dict[str, float],
                         p: np.ndarray, k: int) -> Set[str]:
    """Percent 赛制：C_i = J_i/sumJ + p_i，淘汰最小的 k 个。"""
    if k <= 0:
        return set()
    sum_j = sum(judge_totals[n] for n in names)
    if sum_j <= 0:
        sum_j = 1.0
    c_vals = [(n, judge_totals[n] / sum_j + float(p[i])) for i, n in enumerate(names)]
    c_vals.sort(key=lambda x: (x[1], x[0]))  # 加入 name 做稳定排序
    return {n for n, _ in c_vals[:k]}


def compute_elim_rank(names: List[str], judge_totals: Dict[str, float],
                      p: np.ndarray, k: int) -> Set[str]:
    """Rank 赛制：R_i = rankJ_i + rankF_i，淘汰最大 k 个。"""
    if k <= 0:
        return set()
    j_scores = [judge_totals[n] for n in names]
    rj = rank_average_desc(j_scores)
    rf = rank_average_desc(p.tolist())
    r_sum = rj + rf
    pairs = list(zip(names, r_sum))
    pairs.sort(key=lambda x: (x[1], x[0]), reverse=True)  # 大者更差；稳定排序
    return {n for n, _ in pairs[:k]}


def compute_margin_percent(names: List[str], judge_totals: Dict[str, float],
                           p: np.ndarray, k: int) -> float:
    """Percent 赛制的淘汰边际：C_(k+1) - C_(k)。"""
    n = len(names)
    if k <= 0 or k >= n:
        return np.nan
    sum_j = sum(judge_totals[nm] for nm in names)
    if sum_j <= 0:
        sum_j = 1.0
    c_vals = np.array([judge_totals[nm] / sum_j + float(p[i]) for i, nm in enumerate(names)])
    c_sorted = np.sort(c_vals)  # 小到大
    return float(c_sorted[k] - c_sorted[k - 1])


def compute_margin_rank(names: List[str], judge_totals: Dict[str, float],
                        p: np.ndarray, k: int) -> float:
    """Rank 赛制的淘汰边际：R_(k) - R_(k+1)（R 大者淘汰）。"""
    n = len(names)
    if k <= 0 or k >= n:
        return np.nan
    j_scores = [judge_totals[nm] for nm in names]
    rj = rank_average_desc(j_scores)
    rf = rank_average_desc(p.tolist())
    r_sum = rj + rf
    r_sorted = np.sort(r_sum)[::-1]  # 大到小（差到好）
    return float(r_sorted[k - 1] - r_sorted[k])


def zscore(arr: np.ndarray) -> np.ndarray:
    """稳定 z-score，避免 std=0。"""
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mu) / sd


def compute_distance_percent(names: List[str], judge_totals: Dict[str, float],
                             p: np.ndarray, elim_set: Set[str]) -> float:
    """
    Percent 赛制软距离：
    用“淘汰者 vs 幸存者”的成对违反幅度度量：
    d = mean(max(0, C_e - C_safe))，全一致时 d=0。
    """
    if len(elim_set) == 0 or len(elim_set) == len(names):
        return 0.0
    sum_j = sum(judge_totals[nm] for nm in names)
    if sum_j <= 0:
        sum_j = 1.0
    c_vals = np.array([judge_totals[nm] / sum_j + float(p[i]) for i, nm in enumerate(names)], dtype=float)
    elim_mask = np.array([nm in elim_set for nm in names], dtype=bool)
    if not np.any(elim_mask) or np.all(elim_mask):
        return 0.0
    elim_scores = c_vals[elim_mask]
    safe_scores = c_vals[~elim_mask]
    diff = elim_scores[:, None] - safe_scores[None, :]
    violations = diff[diff > 0]
    if violations.size == 0:
        return 0.0
    viol_frac = float(violations.size / diff.size) if diff.size > 0 else 0.0
    mean_v = float(np.mean(violations))
    max_v = float(np.max(violations))
    return float((mean_v + 0.75 * max_v) * (1.0 + viol_frac))


def compute_distance_rank(names: List[str], judge_totals: Dict[str, float],
                          p: np.ndarray, elim_set: Set[str], delta: float) -> float:
    """
    Rank 赛制软距离（rank-gap-aware）：
    以 R_sum = rank(J) + rank(F) 为基准，度量淘汰/幸存之间的成对“rank gap”。
    """
    if len(elim_set) == 0 or len(elim_set) == len(names):
        return 0.0
    j_scores = [judge_totals[nm] for nm in names]
    rj = rank_average_desc(j_scores)
    rf = rank_average_desc(p.tolist())
    r_sum = rj + rf
    elim_mask = np.array([nm in elim_set for nm in names], dtype=bool)
    if not np.any(elim_mask) or np.all(elim_mask):
        return 0.0
    elim_scores = r_sum[elim_mask]
    safe_scores = r_sum[~elim_mask]
    diff = safe_scores[:, None] - elim_scores[None, :]
    violations = diff[diff > 0]
    if violations.size == 0:
        return 0.0
    viol_frac = float(violations.size / diff.size) if diff.size > 0 else 0.0
    mean_v = float(np.mean(violations))
    max_v = float(np.max(violations))
    scale = max(1.0, 2.0 * len(names))
    return float(((mean_v + 0.75 * max_v) * (1.0 + viol_frac)) / scale)


# -----------------------------
# ABC 采样（动态 Dirichlet）
# -----------------------------


def sample_dirichlet(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dirichlet 抽样（保证 alpha > 0）。"""
    alpha = np.maximum(alpha, 1e-6)
    return rng.dirichlet(alpha)


def dirichlet_log_prior(p: np.ndarray, alpha: np.ndarray) -> float:
    """
    计算 Dirichlet 的（去掉常数项的）log prior：
    log pi(p) ∝ sum_i (alpha_i - 1) * log(p_i)
    用于从 accepted 样本中挑一个“最像先验”的 p_map（保证满足淘汰约束）。
    """
    eps = 1e-12
    a = np.maximum(alpha, 1e-6)
    return float(np.sum((a - 1.0) * np.log(p + eps)))


def run_abc_sampling(names: List[str],
                     judge_totals: Dict[str, float],
                     elim_set: Set[str],
                     scheme: str,
                     prior_alpha: np.ndarray,
                     n_accept: int,
                     draw_max: int,
                     rng: np.random.Generator,
                     track_soft: bool,
                     soft_delta: float) -> Tuple[np.ndarray, int, int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    ABC rejection sampling：
    返回 (accepted_samples, accepted, draws, distances, pred_match)。
    """
    n = len(names)
    k = len(elim_set)

    # 若该周无淘汰（k=0），则约束为空：后验=先验
    if k == 0:
        target = min(n_accept, draw_max)
        samples = np.array([sample_dirichlet(prior_alpha, rng) for _ in range(target)])
        if track_soft:
            distances = np.zeros(target, dtype=float)
            pred_match = np.ones(target, dtype=bool)
            return samples, target, target, distances, pred_match
        return samples, target, target, None, None

    accepted = 0
    draws = 0
    samples = []
    distances = [] if track_soft else None
    pred_match = [] if track_soft else None

    while draws < draw_max and accepted < n_accept:
        p = sample_dirichlet(prior_alpha, rng)

        if scheme == "Percent":
            pred = compute_elim_percent(names, judge_totals, p, k)
            if track_soft:
                distances.append(compute_distance_percent(names, judge_totals, p, elim_set))
        else:
            pred = compute_elim_rank(names, judge_totals, p, k)
            if track_soft:
                distances.append(compute_distance_rank(names, judge_totals, p, elim_set, soft_delta))

        draws += 1
        if track_soft:
            pred_match.append(pred == elim_set)
        if pred == elim_set:
            samples.append(p)
            accepted += 1

    if len(samples) == 0:
        dist_arr = np.array(distances, dtype=float) if track_soft else None
        match_arr = np.array(pred_match, dtype=bool) if track_soft else None
        return np.zeros((0, n)), 0, draws, dist_arr, match_arr
    dist_arr = np.array(distances, dtype=float) if track_soft else None
    match_arr = np.array(pred_match, dtype=bool) if track_soft else None
    return np.vstack(samples), accepted, draws, dist_arr, match_arr


def abc_with_fallback(names: List[str],
                      judge_totals: Dict[str, float],
                      elim_set: Set[str],
                      scheme: str,
                      prior_alpha: np.ndarray,
                      n_accept: int,
                      draw_max: int,
                      draw_max_cap: int,
                      rng: np.random.Generator,
                      have_prev: bool,
                      track_soft: bool,
                      soft_delta: float) -> Tuple[np.ndarray, int, int, str, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    更“统计合理”的 fallback：
    1) 正常 prior
    2) 若 0 accepted 且有 prev：优先减小 kappa（放松平滑，让先验更发散）
    3) 再不行用 uniform prior
    4) accepted 太少：适度提高 draw_max（不超过 draw_max_cap）
    5) 仍太少：降低目标 n_accept（保底出 CI）
    返回 (samples, accepted, draws, status, prior_used)
    """
    status = "ok"
    prior_used = prior_alpha.copy()

    # 1) 正常采样
    samples, accepted, draws, distances, pred_match = run_abc_sampling(
        names, judge_totals, elim_set, scheme, prior_used, n_accept, draw_max, rng, track_soft, soft_delta
    )

    # 2) 若完全失败且有 prev：优先放松平滑（更接近“提议分布改良”）
    if accepted == 0 and have_prev:
        status = "fallback_loosen_kappa"
        # 将 prior_alpha 向均匀方向拉近：alpha' = 0.25*alpha + 0.75*1
        prior_used = 0.25 * prior_used + 0.75 * np.ones_like(prior_used)
        samples, accepted, draws, distances, pred_match = run_abc_sampling(
            names, judge_totals, elim_set, scheme, prior_used, n_accept, draw_max, rng, track_soft, soft_delta
        )

    # 3) 仍失败：uniform prior
    if accepted == 0:
        status = "fallback_uniform"
        prior_used = np.ones(len(names))
        samples, accepted, draws, distances, pred_match = run_abc_sampling(
            names, judge_totals, elim_set, scheme, prior_used, n_accept, draw_max, rng, track_soft, soft_delta
        )

    # 4) accepted 太少：提高 draw_max（更像“增加计算预算”而非改变统计含义）
    if accepted < max(10, int(0.1 * n_accept)):
        new_draw_max = min(draw_max_cap, int(draw_max * 2))
        if new_draw_max > draw_max:
            status = "fallback_more_draws"
            samples2, accepted2, draws2, distances2, pred_match2 = run_abc_sampling(
                names, judge_totals, elim_set, scheme, prior_used, n_accept, new_draw_max, rng, track_soft, soft_delta
            )
            # 用更好的结果替换
            if accepted2 > accepted:
                samples, accepted, draws = samples2, accepted2, draws2
                distances, pred_match = distances2, pred_match2
                draw_max = new_draw_max

    # 5) 仍太少：降低 n_accept（至少出个稳定区间）
    if accepted < max(10, int(0.1 * n_accept)):
        status = "fallback_reduce_accept"
        target = min(200, n_accept)
        samples2, accepted2, draws2, distances2, pred_match2 = run_abc_sampling(
            names, judge_totals, elim_set, scheme, prior_used, target, draw_max, rng, track_soft, soft_delta
        )
        if accepted2 > 0:
            samples, accepted, draws = samples2, accepted2, draws2
            distances, pred_match = distances2, pred_match2

    if accepted == 0:
        status = "failed"

    return samples, accepted, draws, status, prior_used, distances, pred_match


# -----------------------------
# 主流程与输出
# -----------------------------


def summarize_posterior(samples: np.ndarray,
                        prior_alpha_used: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """
    输出：
    - posterior mean
    - 90% CI
    - 平均熵
    - p_map：从 accepted 样本里选 prior 最大者（保证满足淘汰约束，适用于 rank 非线性）
    """
    n = samples.shape[1]
    if samples.shape[0] == 0:
        nanv = np.full(n, np.nan)
        return nanv, nanv, nanv, np.nan, nanv

    p_mean = samples.mean(axis=0)
    p_lo = np.quantile(samples, 0.05, axis=0)
    p_hi = np.quantile(samples, 0.95, axis=0)

    # 平均熵（越大越“分散/不确定”）
    eps = 1e-12
    ent = float((-samples * np.log(samples + eps)).sum(axis=1).mean())

    # p_map：挑最符合 prior 的 accepted 样本
    scores = np.array([dirichlet_log_prior(samples[i], prior_alpha_used) for i in range(samples.shape[0])])
    p_map = samples[int(np.argmax(scores))].copy()

    return p_mean, p_lo, p_hi, ent, p_map


def compute_soft_weights(distances: Optional[np.ndarray],
                         eps: float,
                         pred_match: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float]:
    """根据距离计算 soft ABC 权重与 ESS（eps 可基于 match 情况自适应）。"""
    if distances is None or len(distances) == 0:
        return np.array([], dtype=float), float("nan"), 0.0
    use_eps = float(eps)
    if (not np.isfinite(use_eps)) or use_eps <= 0:
        pos = distances[distances > 0]
        pos_scale = float(np.quantile(pos, 0.20)) if len(pos) > 0 else 1e-6
        if (not np.isfinite(pos_scale)) or pos_scale <= 0:
            pos_scale = 1e-6
        if pred_match is not None and len(pred_match) == len(distances):
            weights = 1.0 + 4.0 * pred_match.astype(float)
            order = np.argsort(distances)
            v = distances[order]
            w = weights[order]
            cdf = np.cumsum(w)
            cutoff = 0.5 * cdf[-1]
            idx = int(np.searchsorted(cdf, cutoff))
            wm = float(v[min(idx, len(v) - 1)])
            if (not np.isfinite(wm)) or wm <= 0:
                wm = pos_scale
            match_frac = float(np.mean(pred_match)) if len(pred_match) > 0 else 0.0
            use_eps = max(wm * (1.0 - 0.85 * match_frac), 0.03 * pos_scale, 1e-6)
            use_eps = min(use_eps, pos_scale)
        else:
            use_eps = max(pos_scale, 1e-6)
    w = np.exp(-((distances / use_eps) ** 2))
    w_sum = float(np.sum(w))
    if (not np.isfinite(w_sum)) or w_sum <= 0:
        min_d = float(np.min(distances))
        w = (distances == min_d).astype(float)
        w_sum = float(np.sum(w))
    ess = float((w_sum ** 2) / np.sum(w ** 2)) if w_sum > 0 else 0.0
    return w, use_eps, ess


def compute_pp_consistency(pred_match: Optional[np.ndarray],
                           weights: Optional[np.ndarray],
                           ess: float) -> Tuple[float, float]:
    """Posterior predictive consistency 及其标准误（ESS 近似）。"""
    if pred_match is None or len(pred_match) == 0:
        return np.nan, np.nan
    if weights is None or len(weights) == 0:
        ppc = float(np.mean(pred_match))
        m = len(pred_match)
        se = float(np.sqrt(ppc * (1.0 - ppc) / m)) if m > 0 else np.nan
        return ppc, se
    w_sum = float(np.sum(weights))
    if w_sum <= 0 or (not np.isfinite(w_sum)):
        return np.nan, np.nan
    ppc = float(np.sum(weights * pred_match) / w_sum)
    ess_eff = ess if ess > 1e-12 else float(len(pred_match))
    se = float(np.sqrt(ppc * (1.0 - ppc) / ess_eff)) if ess_eff > 0 else np.nan
    return ppc, se


def predict_elim_and_margin(names: List[str],
                            judge_totals: Dict[str, float],
                            scheme: str,
                            p: np.ndarray,
                            k: int) -> Tuple[Set[str], float]:
    """给定 p，预测淘汰集合并计算 margin。"""
    if scheme == "Percent":
        pred = compute_elim_percent(names, judge_totals, p, k)
        margin = compute_margin_percent(names, judge_totals, p, k)
    else:
        pred = compute_elim_rank(names, judge_totals, p, k)
        margin = compute_margin_rank(names, judge_totals, p, k)
    return pred, margin


def main() -> None:
    args = parse_args()
    data_path = locate_data_path(args.data_path)

    # 读数据
    df = pd.read_csv(data_path)
    week_cols = extract_week_columns(df.columns.tolist())
    if not week_cols:
        print("未识别到 weekX_judgeY_score 列。")
        sys.exit(1)

    df = compute_week_judge_totals(df, week_cols)
    tasks = build_week_tasks(df, week_cols)

    # 输出容器
    est_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    # 动态先验按赛季分开；允许断裂（None）
    prev_p_by_season: Dict[int, Optional[Dict[str, float]]] = {}

    for task in tasks:
        season = task["season"]
        week = task["week"]
        names = task["names"]
        judge_totals = task["judge_totals"]
        elim_set = task["elim_set"]
        scheme = task["scheme"]
        n = len(names)

        # 每周独立 RNG（稳定 seed，避免 Python hash 随机化）
        local_seed = stable_seed(args.seed, season, week, "hard")
        rng = np.random.default_rng(local_seed)

        # 构造动态先验
        prev_p = prev_p_by_season.get(season, None)
        have_prev = isinstance(prev_p, dict)

        if not have_prev:
            prior_alpha = np.full(n, args.alpha0)
            prior_source = "initial"
        else:
            prev_vec = np.array([float(prev_p.get(nm, 0.0)) for nm in names], dtype=float)
            if prev_vec.sum() <= 0:
                prior_alpha = np.full(n, args.alpha0)
                prior_source = "reset_zero_prev"
            else:
                prev_vec = prev_vec / prev_vec.sum()
                prior_alpha = np.maximum(args.kappa * prev_vec, 1e-3)
                prior_source = "smoothed_from_prev"

        # ABC 采样（带更合理 fallback）
        samples, accepted, draws, status, prior_used, distances, pred_match = abc_with_fallback(
            names=names,
            judge_totals=judge_totals,
            elim_set=elim_set,
            scheme=scheme,
            prior_alpha=prior_alpha,
            n_accept=args.n_accept,
            draw_max=args.draw_max,
            draw_max_cap=args.draw_max_cap,
            rng=rng,
            have_prev=have_prev,
            track_soft=True,
            soft_delta=args.soft_delta
        )
        accept_rate = accepted / draws if draws > 0 else 0.0

        # Soft ABC: 距离 -> 权重 -> posterior predictive consistency
        weights, soft_eps, soft_ess = compute_soft_weights(distances, args.soft_eps, pred_match)
        pp_consistency, pp_consistency_se = compute_pp_consistency(pred_match, weights, soft_ess)
        feasible = 1 if (accepted > 0 or (np.isfinite(pp_consistency) and pp_consistency > 0.0)) else 0

        # 后验统计（mean + CI + entropy + p_map）
        if samples.shape[0] == 0:
            p_mean = np.full(n, np.nan)
            p_lo = np.full(n, np.nan)
            p_hi = np.full(n, np.nan)
            mean_entropy = np.nan
            p_map = np.full(n, np.nan)
        else:
            p_mean, p_lo, p_hi, mean_entropy, p_map = summarize_posterior(samples, prior_used)

        # 记录结果（每位选手）
        for i, nm in enumerate(names):
            p_m = float(p_mean[i]) if np.isfinite(p_mean[i]) else np.nan
            p_l = float(p_lo[i]) if np.isfinite(p_lo[i]) else np.nan
            p_h = float(p_hi[i]) if np.isfinite(p_hi[i]) else np.nan
            p_mp = float(p_map[i]) if np.isfinite(p_map[i]) else np.nan

            ci_w = p_h - p_l if np.isfinite(p_h) and np.isfinite(p_l) else np.nan
            rel_w = ci_w / p_m if np.isfinite(ci_w) and np.isfinite(p_m) and p_m > 0 else np.nan

            est_rows.append({
                "season": season,
                "week": week,
                "celebrity_name": nm,

                # posterior mean
                "p_mean": p_m,
                "p_lo90": p_l,
                "p_hi90": p_h,

                # MAP(在 accepted 中挑 prior 最大者)：保证满足淘汰约束
                "p_map": p_mp,

                # 若需要“票数”，设总票数 T
                "V_mean": p_m * args.T if np.isfinite(p_m) else np.nan,
                "V_map": p_mp * args.T if np.isfinite(p_mp) else np.nan,

                "accept_rate": accept_rate,
                "scheme": scheme,
                "eliminated_this_week": 1 if nm in elim_set else 0,

                # 不确定性
                "ci_width": ci_w,
                "rel_ci_width": rel_w,
                "mean_entropy": mean_entropy,
            })

        # 一致性与 margin：同时给 mean 与 map 两套（rank 非线性下很有用）
        k = len(elim_set)
        consistency_mean = np.nan
        margin_mean = np.nan
        consistency_map = np.nan
        margin_map = np.nan

        if np.all(np.isfinite(p_mean)):
            pred_mean, margin_mean = predict_elim_and_margin(names, judge_totals, scheme, p_mean, k)
            consistency_mean = 1.0 if pred_mean == elim_set else 0.0

        if np.all(np.isfinite(p_map)):
            pred_map, margin_map = predict_elim_and_margin(names, judge_totals, scheme, p_map, k)
            consistency_map = 1.0 if pred_map == elim_set else 0.0

        summary_rows.append({
            "season": season,
            "week": week,
            "N_active": n,
            "scheme": scheme,
            "elim_set": ";".join(sorted(list(elim_set))),
            "accept_rate": accept_rate,
            "feasible": feasible,
            "pp_consistency": pp_consistency,
            "pp_consistency_se": pp_consistency_se,
            "soft_eps": soft_eps,
            "soft_ess": soft_ess,
            "soft_draws": int(len(distances)) if distances is not None else 0,

            "consistency_mean": consistency_mean,
            "margin_mean": margin_mean,

            "consistency_map": consistency_map,
            "margin_map": margin_map,

            "n_accept": accepted,
            "draws": draws,
            "status": status,
            "prior_source": prior_source,
        })

        # 更新动态先验：优先用 p_mean（平滑意义更强），若不可用则断裂
        if np.all(np.isfinite(p_mean)):
            prev_p_by_season[season] = {nm: float(p_mean[i]) for i, nm in enumerate(names)}
        else:
            prev_p_by_season[season] = None

    # 输出 CSV
    est_df = pd.DataFrame(est_rows)
    summary_df = pd.DataFrame(summary_rows)

    # 每周不确定性：rel_ci_width 的周均值
    week_uncertainty = (
        est_df.groupby(["season", "week"])["rel_ci_width"]
        .mean()
        .reset_index()
        .rename(columns={"rel_ci_width": "week_uncertainty"})
    )
    summary_df = summary_df.merge(week_uncertainty, on=["season", "week"], how="left")

    # margin vs uncertainty 相关（全局统计）
    corr_df = summary_df[["margin_map", "week_uncertainty"]].dropna()
    pearson = float(corr_df["margin_map"].corr(corr_df["week_uncertainty"], method="pearson")) if len(corr_df) > 1 else float("nan")
    spearman = float(corr_df["margin_map"].corr(corr_df["week_uncertainty"], method="spearman")) if len(corr_df) > 1 else float("nan")
    metrics = {
        "corr_margin_uncertainty_pearson": pearson,
        "corr_margin_uncertainty_spearman": spearman,
        "n_weeks": int(len(corr_df)),
    }
    with open("global_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    est_df.to_csv("fan_vote_estimates.csv", index=False, encoding="utf-8")
    summary_df.to_csv("weekly_summary.csv", index=False, encoding="utf-8")

    # Quick check：每周 p_mean 和为 1（允许数值误差）
    bad_rows = 0
    for (season, week), g in est_df.groupby(["season", "week"]):
        pvals = g["p_mean"].values
        if np.any(np.isnan(pvals)):
            continue
        if np.any(pvals < -1e-8) or abs(pvals.sum() - 1.0) > 1e-6:
            bad_rows += 1
    if bad_rows > 0:
        print(f"[警告] 有 {bad_rows} 个周的 p_mean 未通过归一化检查（可能是数值误差或异常周）。")

    # 终端 summary：pp_consistency 为主指标，consistency_map 仅作“硬一致性/可行性”
    valid_pp = summary_df["pp_consistency"].dropna()
    overall_pp = valid_pp.mean() if len(valid_pp) > 0 else np.nan
    pp_by_scheme = summary_df.groupby("scheme")["pp_consistency"].mean()
    feasible_rate = summary_df["feasible"].mean()

    valid_map = summary_df["consistency_map"].dropna()
    overall_cons_map = valid_map.mean() if len(valid_map) > 0 else np.nan
    cons_map_by_scheme = summary_df.groupby("scheme")["consistency_map"].mean()

    valid_mean = summary_df["consistency_mean"].dropna()
    overall_cons_mean = valid_mean.mean() if len(valid_mean) > 0 else np.nan
    cons_mean_by_scheme = summary_df.groupby("scheme")["consistency_mean"].mean()

    accept_stats = summary_df["accept_rate"].describe()[["mean", "min", "50%", "max"]]
    ci_stats = est_df["ci_width"].describe()[["mean", "min", "50%", "max"]]
    rel_ci_stats = est_df["rel_ci_width"].describe()[["mean", "min", "50%", "max"]]

    margin_map_median = summary_df["margin_map"].median()
    margin_map_min = summary_df["margin_map"].min()

    print("\n=== Summary ===")
    print(f"overall posterior predictive consistency (pp_consistency): {overall_pp:.4f}")
    print("pp_consistency by scheme:")
    for sch, val in pp_by_scheme.items():
        print(f"  {sch}: {val:.4f}")
    print(f"feasible rate (hard or pp>0): {feasible_rate:.4f}")

    print(f"overall consistency (posterior MAP p_map): {overall_cons_map:.4f}")
    print("consistency_map by scheme:")
    for sch, val in cons_map_by_scheme.items():
        print(f"  {sch}: {val:.4f}")

    print(f"\noverall consistency (posterior mean p_mean): {overall_cons_mean:.4f}")
    print("consistency_mean by scheme:")
    for sch, val in cons_mean_by_scheme.items():
        print(f"  {sch}: {val:.4f}")

    print("\naccept_rate stats (mean/min/median/max):")
    print(accept_stats.to_string())

    print("\nCI width stats (mean/min/median/max):")
    print(ci_stats.to_string())

    print("\nRelative CI width stats (mean/min/median/max):")
    print(rel_ci_stats.to_string())

    print("\nmargin_map stats (median/min):")
    print(f"median: {margin_map_median:.6f}")
    print(f"min:    {margin_map_min:.6f}")

    print("\nmargin vs uncertainty correlation (week_uncertainty vs margin_map):")
    print(f"pearson:  {pearson:.4f}")
    print(f"spearman: {spearman:.4f}")

    print("\n输出文件：fan_vote_estimates.csv, weekly_summary.csv, global_metrics.json")


if __name__ == "__main__":
    main()
