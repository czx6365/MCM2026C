#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

coef_path = os.path.join(OUT_DIR, "coef_compare_all.csv")
perf_path = os.path.join(OUT_DIR, "performance_weeks_model.csv")
pro_j_path = os.path.join(OUT_DIR, "pro_effects_judges_std.csv")
pro_f_path = os.path.join(OUT_DIR, "pro_effects_fans_std.csv")
qs_path = os.path.join(OUT_DIR, "quick_summary.txt")

coef = pd.read_csv(coef_path)
perf = pd.read_csv(perf_path)
pro_j = pd.read_csv(pro_j_path)
pro_f = pd.read_csv(pro_f_path)

# -----------------------------
# 工具：从 "C(pro_s)[T.Name]" 提取 Name
# -----------------------------
def term_to_name(term: str) -> str:
    m = re.search(r"\[T\.(.*)\]", str(term))
    return m.group(1) if m else str(term)

# ============================================================
# Figure 1: Judges vs Fans 系数对比散点图（方向是否一致）
# ============================================================
x = coef["judge_coef_std"].to_numpy()
y = coef["fan_coef_std"].to_numpy()
labels = coef["term"].astype(str).to_list()

plt.figure(figsize=(6.5, 6.0))
plt.scatter(x, y)
lim = np.nanmax(np.abs(np.r_[x, y])) * 1.2 if np.isfinite(np.nanmax(np.abs(np.r_[x, y]))) else 1
plt.plot([-lim, lim], [-lim, lim], linewidth=1)  # 45°线
plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.xlabel("Judges effect (standardized coef)")
plt.ylabel("Fans effect (standardized coef)")
plt.title("Do factors impact judges and fans in the same way?")

# 标注点（避免太拥挤：这里只标注全部；变量多时可只标注方向相反/显著者）
for xi, yi, lab in zip(x, y, labels):
    if np.isfinite(xi) and np.isfinite(yi):
        plt.text(xi, yi, lab, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Fig1_coef_judges_vs_fans.png"), dpi=300)
plt.close()



def normalize_pro_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    把 pro_effects_*.csv 读出来的各种列名格式统一成：
    pro, coef_std
    """
    df = df.copy()
    cols = list(df.columns)

    # 情况1：第一列是索引(参数名)，第二列是系数
    if "coef_std" not in df.columns and len(cols) >= 2:
        df = df.rename(columns={cols[1]: "coef_std"})

    # 情况2：只有一列，直接当 coef_std
    if "coef_std" not in df.columns and len(cols) == 1:
        df = df.rename(columns={cols[0]: "coef_std"})

    # pro 名称来自第一列（参数名）
    df["pro"] = df.iloc[:, 0].apply(term_to_name)

    out = df[["pro", "coef_std"]].copy()
    # 清理：去掉基准/OtherPro（可选）
    out = out[out["pro"].ne("OtherPro")]
    # 去重（以防 CSV 中重复）
    out = out.drop_duplicates(subset=["pro"])
    return out

pro_j = normalize_pro_df(pro_j)  # columns: pro, coef_std
pro_f = normalize_pro_df(pro_f)

# 只保留两边都出现的 pro，避免一边缺失
common_pro = sorted(set(pro_j["pro"]).intersection(set(pro_f["pro"])))
pj = pro_j.set_index("pro").loc[common_pro].rename(columns={"coef_std": "judge_coef"})
pf = pro_f.set_index("pro").loc[common_pro].rename(columns={"coef_std": "fan_coef"})

merged = pj.join(pf, how="inner")  # index=pro, columns=[judge_coef, fan_coef]

# ---- 选同一批 pro：按绝对值取极端者（更公平）----
K = 12  # 你可调 10/12/15，12 通常更饱满
top_f = merged["fan_coef"].abs().sort_values(ascending=False).head(K).index
top_j = merged["judge_coef"].abs().sort_values(ascending=False).head(K).index
selected = sorted(set(top_f).union(set(top_j)))

plot_df = merged.loc[selected].copy()

# ---- 统一排序：建议按 fan_coef 从大到小（也可换成 abs）----
plot_df = plot_df.sort_values("fan_coef", ascending=True)  # barh 从下往上，ascending=True 更直观

# ---- 画图：左右共享同一 y 顺序 ----
plt.figure(figsize=(12.5, 6.5))

# 左：Judges
ax1 = plt.subplot(1, 2, 1)
ax1.barh(plot_df.index, plot_df["judge_coef"])
ax1.axvline(0, linewidth=0.8)
ax1.set_title("Pro dancer effects on Judges (std)")
ax1.set_xlabel("Effect vs baseline")
ax1.grid(axis="x", linewidth=0.5, alpha=0.3)  # 网格更易读

# 右：Fans
ax2 = plt.subplot(1, 2, 2, sharey=ax1)  # sharey：保证顺序一致
ax2.barh(plot_df.index, plot_df["fan_coef"])
ax2.axvline(0, linewidth=0.8)
ax2.set_title("Pro dancer effects on Fans (std)")
ax2.set_xlabel("Effect vs baseline")
ax2.grid(axis="x", linewidth=0.5, alpha=0.3)

# ---- 可选：把关键数字写在图里（增强论文叙事）----
# 建议从 quick_summary.txt 读取 corr 和 ΔR2（你已经在文件里写了）
try:
    qs = open(os.path.join(OUT_DIR, "quick_summary.txt"), "r", encoding="utf-8").read()
    m_corr = re.search(r"corr\s*=\s*([-\d\.eE]+)", qs)
    m_drj  = re.search(r"ΔR2\s*\(Judges,\s*add pro\)\s*:\s*([-\d\.eE]+)", qs)
    m_drf  = re.search(r"ΔR2\s*\(Fans,\s*add pro\)\s*:\s*([-\d\.eE]+)", qs)
    corr_txt = m_corr.group(1) if m_corr else "NA"
    drj_txt  = m_drj.group(1) if m_drj else "NA"
    drf_txt  = m_drf.group(1) if m_drf else "NA"

    # 在右图角落写注释
    ax2.text(
        0.02, 0.02,
        f"ΔR² add-pro: Judges {drj_txt}, Fans {drf_txt}\n"
        f"corr(pro effects) ≈ {corr_txt}",
        transform=ax2.transAxes,
        fontsize=9,
        va="bottom",
        ha="left"
    )
except Exception:
    pass

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Fig2_pro_effects_judges_vs_fans_aligned.png"), dpi=300)
plt.close()

