#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量敏感性分析脚本：自动运行 estimate_fan_votes.py 并汇总关键指标。

默认扫描：
alpha0 ∈ {1, 5, 20}
kappa  ∈ {10, 40, 120}

输出：
- sensitivity_runs/alpha0=..._kappa=.../  每组参数的一次运行产物（2个CSV + log）
- sensitivity_grid.csv                   汇总指标表（论文可直接引用）
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# 参数解析
# -----------------------------


def parse_list(s: str, cast=float) -> List:
    """把 '1,5,20' 这种字符串解析成列表。"""
    items = [x.strip() for x in s.split(",") if x.strip() != ""]
    return [cast(x) for x in items]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sensitivity grid for alpha0/kappa and summarize results.")
    parser.add_argument("--script", type=str, default="t1/estimate_fan_votes.py",
                        help="要调用的主脚本（默认 t1/estimate_fan_votes.py）")
    parser.add_argument("--data_path", type=str, default="data/2026_MCM_Problem_C_Data.csv",
                        help="数据路径（会透传给主脚本）")
    parser.add_argument("--processed_dir", type=str, default="data/process",
                        help="预处理结果目录（透传给主脚本）")
    parser.add_argument("--alphas", type=str, default="1,5,20",
                        help="alpha0 网格，如 '1,5,20'")
    parser.add_argument("--kappas", type=str, default="10,40,120",
                        help="kappa 网格，如 '10,40,120'")
    parser.add_argument("--n_accept", type=int, default=500, help="每周目标接受样本数")
    parser.add_argument("--draw_max", type=int, default=50000, help="每周最大抽样次数")
    parser.add_argument("--T", type=float, default=10000000, help="总票数缩放 T")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子（透传给主脚本）")
    parser.add_argument("--out_dir", type=str, default="sensitivity_runs",
                        help="批量运行输出目录（默认 sensitivity_runs）")
    parser.add_argument("--timeout", type=int, default=0,
                        help="单次运行超时秒数（0=不限时；Windows上建议不设）")
    return parser.parse_args()


# -----------------------------
# 指标计算
# -----------------------------


def safe_mean(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean()) if len(x) else np.nan


def safe_median(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.median()) if len(x) else np.nan


def summarize_run(weekly_path: Path, fan_path: Path) -> Dict[str, Any]:
    """
    从 weekly_summary.csv 和 fan_vote_estimates.csv 汇总一组参数的指标。
    """
    ws = pd.read_csv(weekly_path)
    fv = pd.read_csv(fan_path)

    # consistency（mean 与 map）
    overall_cons_map = safe_mean(ws["consistency_map"])
    overall_cons_mean = safe_mean(ws["consistency_mean"])

    # 按赛制分组 consistency
    cons_map_by_scheme = ws.groupby("scheme")["consistency_map"].mean().to_dict()
    cons_mean_by_scheme = ws.groupby("scheme")["consistency_mean"].mean().to_dict()

    # accept_rate
    acc_mean = safe_mean(ws["accept_rate"])
    acc_median = safe_median(ws["accept_rate"])
    acc_min = float(ws["accept_rate"].min()) if len(ws) else np.nan
    acc_max = float(ws["accept_rate"].max()) if len(ws) else np.nan

    # CI（逐选手逐周）
    ci_mean = safe_mean(fv["ci_width"])
    ci_median = safe_median(fv["ci_width"])
    ci_min = float(fv["ci_width"].min()) if len(fv) else np.nan
    ci_max = float(fv["ci_width"].max()) if len(fv) else np.nan

    rel_ci_mean = safe_mean(fv["rel_ci_width"])
    rel_ci_median = safe_median(fv["rel_ci_width"])
    rel_ci_min = float(fv["rel_ci_width"].min()) if len(fv) else np.nan
    rel_ci_max = float(fv["rel_ci_width"].max()) if len(fv) else np.nan

    # margin_map（逐周）
    margin_map_median = safe_median(ws["margin_map"])
    margin_map_min = float(ws["margin_map"].min()) if ws["margin_map"].notna().any() else np.nan

    # 均值失配次数（通常只出现在 Rank）
    mean_mismatch = int(((ws["consistency_mean"] == 0) & (ws["scheme"] == "Rank")).sum())

    # 无淘汰周计数（elim_set 为空字符串）
    no_elim_weeks = int((ws["elim_set"].fillna("") == "").sum())

    # 输出字典
    out = {
        "overall_consistency_map": overall_cons_map,
        "overall_consistency_mean": overall_cons_mean,
        "consistency_map_rank": float(cons_map_by_scheme.get("Rank", np.nan)),
        "consistency_map_percent": float(cons_map_by_scheme.get("Percent", np.nan)),
        "consistency_mean_rank": float(cons_mean_by_scheme.get("Rank", np.nan)),
        "consistency_mean_percent": float(cons_mean_by_scheme.get("Percent", np.nan)),

        "accept_rate_mean": acc_mean,
        "accept_rate_median": acc_median,
        "accept_rate_min": acc_min,
        "accept_rate_max": acc_max,

        "ci_width_mean": ci_mean,
        "ci_width_median": ci_median,
        "ci_width_min": ci_min,
        "ci_width_max": ci_max,

        "rel_ci_width_mean": rel_ci_mean,
        "rel_ci_width_median": rel_ci_median,
        "rel_ci_width_min": rel_ci_min,
        "rel_ci_width_max": rel_ci_max,

        "margin_map_median": margin_map_median,
        "margin_map_min": margin_map_min,

        "rank_mean_mismatch_weeks": mean_mismatch,
        "no_elimination_weeks": no_elim_weeks,
        "num_weeks_evaluated": int(len(ws)),
        "num_rows_fan": int(len(fv)),
    }
    return out


# -----------------------------
# 主流程：批量运行 + 复制产物 + 汇总
# -----------------------------


def run_one(script: str, data_path: str, alpha0: float, kappa: float,
            n_accept: int, draw_max: int, T: float, seed: int,
            run_dir: Path, timeout: int = 0, processed_dir: str = "") -> Tuple[int, str]:
    """
    运行一次主脚本，返回 (returncode, log_text)。
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, script,
        "--data_path", data_path,
        "--alpha0", str(alpha0),
        "--kappa", str(kappa),
        "--n_accept", str(n_accept),
        "--draw_max", str(draw_max),
        "--T", str(T),
        "--seed", str(seed),
    ]
    if processed_dir:
        cmd += ["--processed_dir", processed_dir]

    # 在当前工作目录运行（主脚本会生成 fan_vote_estimates.csv / weekly_summary.csv）
    try:
        if timeout and timeout > 0:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        else:
            p = subprocess.run(cmd, capture_output=True, text=True)
        log_text = (p.stdout or "") + "\n" + (p.stderr or "")
        return p.returncode, log_text
    except subprocess.TimeoutExpired as e:
        return 124, f"[TIMEOUT]\n{e}"


def main():
    args = parse_args()
    alphas = parse_list(args.alphas, float)
    kappas = parse_list(args.kappas, float)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    grid_rows: List[Dict[str, Any]] = []

    # 记录当前目录下输出文件名（主脚本默认写这里）
    fan_name = "fan_vote_estimates.csv"
    week_name = "weekly_summary.csv"

    for a in alphas:
        for k in kappas:
            tag = f"alpha0={a}_kappa={k}"
            run_dir = out_root / tag
            print(f"\n[RUN] {tag}")

            # 运行主脚本
            rc, log_text = run_one(
                script=args.script,
                data_path=args.data_path,
                alpha0=a,
                kappa=k,
                n_accept=args.n_accept,
                draw_max=args.draw_max,
                T=args.T,
                seed=args.seed,
                run_dir=run_dir,
                timeout=args.timeout,
                processed_dir=args.processed_dir
            )

            # 保存日志
            (run_dir / "run.log").write_text(log_text, encoding="utf-8")

            # 若失败，记录并继续
            if rc != 0:
                print(f"[FAIL] returncode={rc}. See {run_dir/'run.log'}")
                grid_rows.append({
                    "alpha0": a, "kappa": k,
                    "returncode": rc,
                    "status": "failed",
                })
                continue

            # 检查输出文件是否存在
            fan_src = Path(fan_name)
            week_src = Path(week_name)
            if (not fan_src.exists()) or (not week_src.exists()):
                print(f"[FAIL] missing output CSV. See log: {run_dir/'run.log'}")
                grid_rows.append({
                    "alpha0": a, "kappa": k,
                    "returncode": 2,
                    "status": "missing_output",
                })
                continue

            # 复制输出到该参数子目录，避免覆盖
            fan_dst = run_dir / fan_name
            week_dst = run_dir / week_name
            shutil.copyfile(fan_src, fan_dst)
            shutil.copyfile(week_src, week_dst)

            # 汇总指标
            metrics = summarize_run(week_dst, fan_dst)
            row = {"alpha0": a, "kappa": k, "returncode": 0, "status": "ok"}
            row.update(metrics)
            row.update({
                "consistency_map": metrics.get("overall_consistency_map"),
                "consistency_mean": metrics.get("overall_consistency_mean"),
                "accept_rate_median": metrics.get("accept_rate_median"),
                "CI_width_mean": metrics.get("ci_width_mean"),
                "CI_width_median": metrics.get("ci_width_median"),
                "rel_CI_width_mean": metrics.get("rel_ci_width_mean"),
                "rel_CI_width_median": metrics.get("rel_ci_width_median"),
                "margin_median": metrics.get("margin_map_median"),
                "margin_min": metrics.get("margin_map_min"),
            })
            grid_rows.append(row)

            print(f"[OK] consistency_map={metrics['overall_consistency_map']:.4f}, "
                  f"consistency_mean={metrics['overall_consistency_mean']:.4f}, "
                  f"acc_median={metrics['accept_rate_median']:.4f}, "
                  f"rel_ci_median={metrics['rel_ci_width_median']:.4f}")

    # 写出汇总表
    grid_df = pd.DataFrame(grid_rows)
    grid_df = grid_df.sort_values(["alpha0", "kappa"]).reset_index(drop=True)
    out_csv = Path("sensitivity_grid.csv")
    grid_df.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n[DONE] 已生成 sensitivity_grid.csv")
    print("你可以在论文中用它做：参数网格稳健性表 / heatmap / 结论句。")


if __name__ == "__main__":
    main()
