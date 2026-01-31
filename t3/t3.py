#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCM 2026 C：Q3 影响分析（Pro dancer + 明星特征）
- 读取官方数据 + 你们估计的 fan vote（p_mean, CI）
- 构造“周粒度”数据：judge_total / judge_percent / fan_logit
- 主模型：
  (1) Judges：OLS + season/week 控制 + pro 固定效应 + 特征；pair_id 聚类稳健SE
  (2) Fans：WLS（用CI宽度加权）+ season/week 控制 + pro 固定效应 + 特征；pair_id 聚类稳健SE
  对比：同一特征在 judges vs fans 上的标准化系数与显著性
- 配套模型（表现）：
  season-level：weeks_survived ~ mean judges + mean fans + 特征 + pro
  用增量R^2量化 pro/特征对“走多远”的额外贡献
输出：
  outputs/coef_compare.csv
  outputs/pro_effects_judges_std.csv
  outputs/pro_effects_fans_std.csv
  outputs/performance_weeks_model.csv
"""

import os
import re
import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "2026_MCM_Problem_C_Data.csv")
FAN_PATH = os.path.join(HERE, "..", "t1", "fan_vote_estimates.csv")
OUT_DIR = os.path.join(HERE, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1) 读取数据
# -----------------------------
df = pd.read_csv(DATA_PATH)
fan = pd.read_csv(FAN_PATH)

# -----------------------------
# 2) 将“宽表 judges scores”改成“长表：每周一行”
# -----------------------------
base_cols = [
    "season", "celebrity_name", "ballroom_partner",
    "celebrity_industry", "celebrity_homestate", "celebrity_homecountry/region",
    "celebrity_age_during_season", "placement", "results"
]
base = df[base_cols].copy()

rows = []
for w in range(1, 12):  # 官方数据最多 week1..week11
    wcols = [c for c in df.columns if c.startswith(f"week{w}_judge") and c.endswith("_score")]
    if not wcols:
        continue
    total = df[wcols].sum(axis=1, skipna=True)
    # 如果该周所有 judge 分都是 NA，则把 sum(=0) 修正为 NA
    all_na = df[wcols].isna().all(axis=1)
    total = total.mask(all_na, np.nan)

    tmp = base.copy()
    tmp["week"] = w
    tmp["judge_total"] = total
    rows.append(tmp)

weekly = pd.concat(rows, ignore_index=True)

# 重命名列，避免斜杠影响公式解析
weekly = weekly.rename(columns={
    "celebrity_homecountry/region": "homecountry",
    "celebrity_age_during_season": "age",
    "celebrity_industry": "industry"
})

# 每个“赛季-明星”作为一个参赛单元（pair_id）
weekly["pair_id"] = weekly["season"].astype(int).astype(str) + "__" + weekly["celebrity_name"]

# active：judge_total > 0 表示仍在比赛（题面说明淘汰后为0）
weekly["active"] = weekly["judge_total"].fillna(0) > 0

# judge_percent：同一 season-week 内（仅 active）占总分比例
weekly["judge_percent"] = np.nan
for (s, w), g in weekly[weekly["active"]].groupby(["season", "week"]):
    denom = g["judge_total"].sum()
    weekly.loc[g.index, "judge_percent"] = g["judge_total"] / denom if denom > 0 else np.nan

# -----------------------------
# 3) 合并 fan vote estimates（你们的 p_mean + CI）
# -----------------------------
fan_use = fan[[
    "season", "week", "celebrity_name",
    "p_mean", "p_lo90", "p_hi90", "ci_width", "rel_ci_width", "accept_rate", "scheme"
]].copy()

dat = weekly.merge(fan_use, on=["season", "week", "celebrity_name"], how="left")

# -----------------------------
# 4) 类别特征做 Top-K 合并（避免 dummy 过多）
# -----------------------------
def topk(series: pd.Series, k: int = 10) -> pd.Series:
    vc = series.value_counts(dropna=False)
    top = set(vc.head(k).index)
    return series.where(series.isin(top), other="Other")

dat["industry_s"] = topk(dat["industry"].fillna("Unknown"), k=10)
dat["homecountry_s"] = topk(dat["homecountry"].fillna("Unknown"), k=10)

# pro dancer 同样做 Top-K（按 active 周的出现频次）
pro_counts = dat.loc[dat["active"], "ballroom_partner"].value_counts()
keep_pro = set(pro_counts[pro_counts >= 25].index)  # 阈值可调
dat["pro_s"] = dat["ballroom_partner"].where(dat["ballroom_partner"].isin(keep_pro), other="OtherPro")

# -----------------------------
# 5) fan_logit + 5A 加权（CI 越窄权重越大）
# -----------------------------
eps = 1e-6
p = dat["p_mean"].clip(eps, 1 - eps)
dat["fan_logit"] = np.log(p / (1 - p))

# 权重：1 / ci_width^2；为防极端值影响，做 99% 截断
w_raw = 1.0 / (dat["ci_width"].fillna(dat["ci_width"].median()) ** 2)
dat["fan_w"] = w_raw
wcap = dat.loc[dat["p_mean"].notna(), "fan_w"].quantile(0.99)
dat["fan_w"] = dat["fan_w"].clip(upper=wcap)

# -----------------------------
# 6) 主模型：Judges vs Fans（同一组解释变量）
# -----------------------------
jud = dat[dat["active"] & dat["judge_total"].notna()].copy()
fan_dat = dat[dat["active"] & dat["p_mean"].notna()].copy()

# 标准化（便于“同一变量”跨模型比较）
jud["judge_z"] = (jud["judge_total"] - jud["judge_total"].mean()) / jud["judge_total"].std()
fan_dat["fan_z"] = (fan_dat["fan_logit"] - fan_dat["fan_logit"].mean()) / fan_dat["fan_logit"].std()

jud["age_z"] = (jud["age"] - jud["age"].mean()) / jud["age"].std()
fan_dat["age_z"] = (fan_dat["age"] - fan_dat["age"].mean()) / fan_dat["age"].std()

# (A) Judges：OLS + pair_id 聚类稳健SE
f_j = "judge_z ~ C(season) + C(week) + age_z + C(industry_s) + C(homecountry_s) + C(pro_s)"
m_j = smf.ols(f_j, data=jud).fit(cov_type="cluster", cov_kwds={"groups": jud["pair_id"]})

# (B) Fans：WLS（加权） + pair_id 聚类稳健SE
f_f = "fan_z ~ C(season) + C(week) + age_z + C(industry_s) + C(homecountry_s) + C(pro_s)"
m_f_raw = smf.wls(f_f, data=fan_dat, weights=fan_dat["fan_w"]).fit()
m_f = m_f_raw.get_robustcov_results(cov_type="cluster", groups=fan_dat["pair_id"])

# -----------------------------
# 7) 对比：同一项在 judges vs fans 的系数与显著性
# -----------------------------
def extract_term(res, term: str):
    """
    statsmodels 的 OLS（pandas Series）与 robust WLS（numpy array）取值方式不同
    """
    if hasattr(res, "params") and isinstance(res.params, pd.Series):
        coef = float(res.params.get(term, np.nan))
        pval = float(res.pvalues.get(term, np.nan))
        return coef, pval
    else:
        names = res.model.exog_names
        if term not in names:
            return np.nan, np.nan
        idx = names.index(term)
        return float(res.params[idx]), float(res.pvalues[idx])

# -----------------------------
# 7) 对比：加入“所有”industry dummy（可选 homecountry）
# -----------------------------
def extract_term(res, term: str):
    """
    statsmodels 的 OLS（pandas Series）与 robust WLS（numpy array）取值方式不同
    返回：(coef, pval)
    """
    if hasattr(res, "params") and isinstance(res.params, pd.Series):
        coef = float(res.params.get(term, np.nan))
        pval = float(res.pvalues.get(term, np.nan))
        return coef, pval
    else:
        names = res.model.exog_names
        if term not in names:
            return np.nan, np.nan
        idx = names.index(term)
        return float(res.params[idx]), float(res.pvalues[idx])

def collect_terms_from_ols(res_ols, prefixes):
    """从 OLS 的 params index 里收集所有匹配前缀的 term"""
    terms = []
    for t in res_ols.params.index.astype(str).tolist():
        for p in prefixes:
            if t.startswith(p):
                terms.append(t)
                break
    return sorted(set(terms))

def collect_terms_from_robust(res_rb, prefixes):
    """从 robust results 的 exog_names 里收集所有匹配前缀的 term"""
    terms = []
    for t in res_rb.model.exog_names:
        for p in prefixes:
            if str(t).startswith(p):
                terms.append(str(t))
                break
    return sorted(set(terms))

# 你想要“全部 industry dummy”，prefix 就是这个：
prefixes = ["C(industry_s)[T."]

# 可选：如果你也想把所有 homecountry 点都画出来，把下面一行解除注释
# prefixes.append("C(homecountry_s)[T.")

# 我们也把 age_z 放进去（连续变量）
must_include = ["age_z"]

# 从两个模型里分别收集，再做并集
terms_j = collect_terms_from_ols(m_j, prefixes)
terms_f = collect_terms_from_robust(m_f, prefixes)
terms_all = sorted(set(terms_j).union(set(terms_f)).union(set(must_include)))

rows_cmp = []
for t in terms_all:
    cj, pj = extract_term(m_j, t)
    cf, pf = extract_term(m_f, t)

    rows_cmp.append({
        "term": t,
        "judge_coef_std": cj,
        "judge_p": pj,
        "fan_coef_std": cf,
        "fan_p": pf,
        "same_direction": (np.sign(cj) == np.sign(cf)) if (np.isfinite(cj) and np.isfinite(cf)) else np.nan,
        "judge_sig_05": (pj < 0.05) if np.isfinite(pj) else np.nan,
        "fan_sig_05": (pf < 0.05) if np.isfinite(pf) else np.nan,
    })

cmp_all_df = pd.DataFrame(rows_cmp)

# 兼容你原来的文件（仍然输出 coef_compare.csv：只放 age_z + 若干代表项也行）
# 这里我建议：
# - coef_compare.csv：保留你原来的“故事版少量点”
# - coef_compare_all.csv：给画“证据链”大散点用
cmp_all_df.to_csv(os.path.join(OUT_DIR, "coef_compare_all.csv"), index=False)

# 如果你想让原 coef_compare.csv 也升级成“全量”，就把下面这一行替换你原来的输出：
# cmp_all_df.to_csv(os.path.join(OUT_DIR, "coef_compare.csv"), index=False)


# -----------------------------
# 8) Pro dancer 影响：提取 pro 固定效应（标准化尺度）
# -----------------------------
pro_j = m_j.params.filter(like="C(pro_s)").sort_values()
pro_j.to_csv(os.path.join(OUT_DIR, "pro_effects_judges_std.csv"), header=["coef_std"])

# fans 的 params 可能是 ndarray，需用 exog_names 对齐
pro_f = {}
for name, coef in zip(m_f.model.exog_names, m_f.params):
    if name.startswith("C(pro_s)"):
        pro_f[name] = coef
pro_f = pd.Series(pro_f).sort_values()
pro_f.to_csv(os.path.join(OUT_DIR, "pro_effects_fans_std.csv"), header=["coef_std"])

# -----------------------------
# 9) 配套模型：表现（weeks_survived）= “走多远”
# -----------------------------
active = dat[dat["active"]].copy()
agg = active.groupby(
    ["pair_id", "season", "celebrity_name", "pro_s", "industry_s", "homecountry_s", "age", "placement"],
    as_index=False
).agg(
    weeks_survived=("week", "max"),
    mean_judge_percent=("judge_percent", "mean"),
    mean_p=("p_mean", "mean"),
)

# 标准化连续变量
agg["age_z"] = (agg["age"] - agg["age"].mean()) / agg["age"].std()
agg["mj_z"] = (agg["mean_judge_percent"] - agg["mean_judge_percent"].mean()) / agg["mean_judge_percent"].std()
agg["mp_z"] = (agg["mean_p"] - agg["mean_p"].mean()) / agg["mean_p"].std()

# 完整模型：控制 judges & fans 的平均表现，再看特征与 pro 的增量贡献
perf_full = smf.ols(
    "weeks_survived ~ C(season) + mj_z + mp_z + age_z + C(industry_s) + C(homecountry_s) + C(pro_s)",
    data=agg
).fit(cov_type="HC3")

perf_base = smf.ols(
    "weeks_survived ~ C(season) + mj_z + mp_z",
    data=agg
).fit(cov_type="HC3")

perf_traits = smf.ols(
    "weeks_survived ~ C(season) + mj_z + mp_z + age_z + C(industry_s) + C(homecountry_s)",
    data=agg
).fit(cov_type="HC3")

out_perf = pd.DataFrame([{
    "R2_base_only_scores": perf_base.rsquared,
    "R2_add_traits": perf_traits.rsquared,
    "R2_add_pro": perf_full.rsquared,
    "delta_R2_traits": perf_traits.rsquared - perf_base.rsquared,
    "delta_R2_pro": perf_full.rsquared - perf_traits.rsquared,
    "effect_mj_z_weeks": perf_full.params.get("mj_z", np.nan),
    "effect_mp_z_weeks": perf_full.params.get("mp_z", np.nan),
    "effect_age_z_weeks": perf_full.params.get("age_z", np.nan),
}])

out_perf.to_csv(os.path.join(OUT_DIR, "performance_weeks_model.csv"), index=False)

# -----------------------------
# 10) 额外：pro 对 judges vs fans 是否一致（相关性）
# -----------------------------
def term_to_name(term: str) -> str:
    m = re.search(r"\[T\.(.*)\]", term)
    return m.group(1) if m else term

pj = pro_j.copy()
pf = pro_f.copy()
pj.index = [term_to_name(i) for i in pj.index]
pf.index = [term_to_name(i) for i in pf.index]
common = pj.index.intersection(pf.index)

corr = np.nan
if len(common) >= 3:
    corr = float(np.corrcoef(pj.loc[common], pf.loc[common])[0, 1])

with open(os.path.join(OUT_DIR, "quick_summary.txt"), "w", encoding="utf-8") as f:
    f.write("=== Quick Summary ===\n")
    f.write(f"Judges model R2 (std y): {m_j.rsquared:.4f}\n")
    f.write(f"Fans   model R2 (std y, WLS): {m_f_raw.rsquared:.4f}\n")
    f.write("\n-- Incremental R2 of adding pro (weekly) --\n")
    # 计算 weekly 模型加入 pro 的增量R2
    m_j0 = smf.ols(
        "judge_z ~ C(season) + C(week) + age_z + C(industry_s) + C(homecountry_s)",
        data=jud
    ).fit(cov_type="cluster", cov_kwds={"groups": jud["pair_id"]})
    m_f0_raw = smf.wls(
        "fan_z ~ C(season) + C(week) + age_z + C(industry_s) + C(homecountry_s)",
        data=fan_dat, weights=fan_dat["fan_w"]
    ).fit()
    f.write(f"ΔR2 (Judges, add pro): {m_j.rsquared - m_j0.rsquared:.4f}\n")
    f.write(f"ΔR2 (Fans,   add pro): {m_f_raw.rsquared - m_f0_raw.rsquared:.4f}\n")
    f.write("\n-- Pro effects correlation (judges vs fans, standardized) --\n")
    f.write(f"corr = {corr}\n")

print("Done. Outputs saved to:", OUT_DIR)
print("Key files:",
      "coef_compare.csv, pro_effects_judges_std.csv, pro_effects_fans_std.csv, performance_weeks_model.csv, quick_summary.txt",
      sep="\n- ")
