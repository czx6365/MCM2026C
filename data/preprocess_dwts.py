# -*- coding: utf-8 -*-
"""
DWTS (2026 MCM Problem C) 预处理脚本 —— 评审级鲁棒版本
核心修复：
1) “每季有效周 (observed weeks)”：只在该 season 实际存在评分信息的 week 上建 roster / elimination
2) pair_id 全局唯一：默认加 season 前缀，并做字符串清洗
3) 质量检查按“有效周序列”输出，不再用全局 max_week 盲目 reindex

运行：
python preprocess_dwts.py
"""

import json
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 0. 小工具函数
# =========================
def _normalize_season(series: pd.Series) -> pd.Series:
    """将 season 字段统一转为整数（允许缺失）。"""
    extracted = series.astype(str).str.extract(r"(\d+)")
    season_num = pd.to_numeric(extracted[0], errors="coerce")
    if season_num.isna().any():
        warnings.warn("season 列存在无法解析的值，将保留为缺失。")
    return season_num.astype("Int64")


def _clean_text(s: pd.Series) -> pd.Series:
    """清洗字符串：去首尾空白、压缩多空格、去掉不可见字符。"""
    out = s.astype(str)
    out = out.str.replace(r"[\u200b\u200c\u200d\ufeff]", "", regex=True)  # 零宽字符
    out = out.str.strip()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out


def _sanitize_pair_id(season: pd.Series, pair_id: pd.Series) -> pd.Series:
    """
    让 pair_id 更鲁棒：
    - 清洗
    - 替换分隔符，避免后续 split/保存出问题
    - 加 season 前缀，确保跨季唯一
    """
    pid = _clean_text(pair_id)
    pid = pid.str.replace("|", " ", regex=False)
    pid = pid.str.replace("/", " __ ", regex=False)
    pid = pid.str.replace(r"\s+", " ", regex=True).str.strip()
    s = season.astype("Int64").astype(str)
    return ("S" + s + "_" + pid).str.replace(r"\s+", " ", regex=True).str.strip()


# =========================
# 1. 读取数据
# =========================
def load_data(path: Path) -> pd.DataFrame:
    """读取 CSV，并打印 head / columns / dtypes 摘要。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件: {path}")

    df = pd.read_csv(
        path,
        na_values=["N/A", "NA", "na", "n/a", "", "NULL", "null"],
        keep_default_na=True,
        low_memory=False,
    )

    print("=== head() ===")
    print(df.head())
    print("=== columns ===")
    print(list(df.columns))
    print("=== dtypes ===")
    print(df.dtypes)
    return df


# =========================
# 2. 自动识别 schema
# =========================
def _find_season_col(df: pd.DataFrame) -> str | None:
    """自动识别 season 列。"""
    season_candidates = [c for c in df.columns if re.search(r"season", c, re.I)]
    if season_candidates:
        for c in season_candidates:
            if re.fullmatch(r"season", c, re.I):
                return c

        # 轻微打分选择“更像 season 的列”
        priority_patterns = [
            (r"season_number|season_no|season_id", 2.5),
            (r"^season_", 2.0),
            (r"_season$", 2.0),
        ]

        best = None
        best_score = -1e9
        for c in season_candidates:
            season_num = _normalize_season(df[c])
            ratio = season_num.notna().mean()

            name_score = 0.0
            for pat, w in priority_patterns:
                if re.search(pat, c, re.I):
                    name_score += w
            if re.search(r"age|during", c, re.I):
                name_score -= 2.0

            score = name_score + ratio
            if score > best_score:
                best_score = score
                best = c
        return best

    # 若列名无 season，尝试从列值推断（不建议，但做兜底）
    best = None
    best_ratio = 0.0
    for c in df.columns:
        series = df[c].dropna()
        if series.empty:
            continue
        as_str = series.astype(str)
        ratio = as_str.str.contains(r"\bS?\d+\b", case=False, regex=True).mean()
        if ratio > best_ratio and ratio >= 0.8:
            best = c
            best_ratio = ratio
    return best


def _choose_best_id_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """在候选中挑选唯一性较高的列。"""
    best = None
    best_ratio = 0.0
    for c in candidates:
        series = df[c]
        if series.isna().all():
            continue
        ratio = series.nunique(dropna=True) / max(len(series), 1)
        if ratio > best_ratio and ratio >= 0.7:
            best = c
            best_ratio = ratio
    return best


def _find_celebrity_and_pro_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """寻找明星与舞者列，用于生成 pair_id。"""
    celeb_patterns = [
        (r"celebrity", 3),
        (r"celeb", 3),
        (r"celebrity_name", 4),
        (r"name", 2),
        (r"star", 1),
    ]
    pro_patterns = [
        (r"ballroom_partner", 4),
        (r"partner", 3),
        (r"pro", 3),
        (r"ballroom", 2),
        (r"dancer", 1),
        (r"professional", 1),
    ]

    def _score(col: str, patterns: list[tuple[str, int]]) -> int:
        score = 0
        for pat, w in patterns:
            if re.search(pat, col, re.I):
                score += w
        return score

    celeb_best, celeb_score = None, -1
    pro_best, pro_score = None, -1

    for c in df.columns:
        cs = _score(c, celeb_patterns)
        if cs > celeb_score:
            celeb_best, celeb_score = c, cs
        ps = _score(c, pro_patterns)
        if ps > pro_score:
            pro_best, pro_score = c, ps

    return celeb_best, pro_best


def detect_schema(df: pd.DataFrame) -> dict:
    """识别 season / pair_id / judge 分数列，并打印识别结果。"""
    # --- season 列 ---
    season_col = _find_season_col(df)
    if season_col is None:
        raise ValueError("无法定位 season 列，请检查数据列名。")
    season_series = _normalize_season(df[season_col])

    # --- pair_id 列 ---
    pair_candidates = [
        c
        for c in df.columns
        if re.search(r"(pair|couple|team|id)", c, re.I) and c != season_col
    ]
    pair_id_col = _choose_best_id_col(df, pair_candidates)

    pair_id_method = None
    celeb_col = None
    pro_col = None

    if pair_id_col is not None:
        pair_id_series = _clean_text(df[pair_id_col])
        pair_id_method = "existing_col"
    else:
        # 退而求其次：明星名 + 专业舞者名
        celeb_col, pro_col = _find_celebrity_and_pro_cols(df)
        if celeb_col and pro_col:
            celeb = _clean_text(df[celeb_col].where(df[celeb_col].notna(), ""))
            pro = _clean_text(df[pro_col].where(df[pro_col].notna(), ""))
            pair_id_series = (celeb + " / " + pro).str.strip()
            pair_id_series = pair_id_series.str.replace(r"^/|/$", "", regex=True).str.strip()
            pair_id_method = "generated_celebrity_pro"
            warnings.warn(f"未找到唯一 pair_id 列，使用 {celeb_col} + {pro_col} 生成 pair_id。")
            if (pair_id_series == "").any():
                warnings.warn("生成的 pair_id 存在空值，请检查姓名字段缺失情况。")
        else:
            pair_id_series = season_series.astype("Int64").astype(str) + "_row" + df.index.astype(str)
            pair_id_method = "generated_index"
            warnings.warn("无法识别 pair_id，将使用 season+行号 生成。")

    # --- judges score 列 ---
    score_col_map: dict[str, tuple[int, int]] = {}
    weeks = set()
    judges = set()

    # 允许 week1_judge2_score / Week 1 Judge 2 Score 等多种写法
    pattern = re.compile(r"week\s*([0-9]+).*judge\s*([0-9]+).*score", re.I)

    for col in df.columns:
        m = pattern.search(col)
        if m:
            week = int(m.group(1))
            judge_id = int(m.group(2))
            if week < 1 or judge_id < 1:
                warnings.warn(f"解析到异常 week/judge 编号: {col}")
                continue
            score_col_map[col] = (week, judge_id)
            weeks.add(week)
            judges.add(judge_id)

    if not score_col_map:
        raise ValueError("未能解析到任何 judge-score 列，请检查列名。")

    week_min, week_max = min(weeks), max(weeks)

    schema_info = {
        "season_col": season_col,
        "pair_id_col": pair_id_col,
        "pair_id_method": pair_id_method,
        "celebrity_col": celeb_col,
        "pro_col": pro_col,
        "score_cols_count": len(score_col_map),
        "week_range": [week_min, week_max],
        "judge_ids": sorted(judges),
    }
    print("=== detect_schema ===")
    print(json.dumps(schema_info, ensure_ascii=False, indent=2))

    return {
        "season_col": season_col,
        "season_series": season_series,
        "pair_id_col": pair_id_col,
        "pair_id_series_raw": pair_id_series,
        "pair_id_method": pair_id_method,
        "celebrity_col": celeb_col,
        "pro_col": pro_col,
        "score_col_map": score_col_map,
        "score_cols": list(score_col_map.keys()),
        "week_min": week_min,
        "week_max": week_max,
        "weeks": sorted(weeks),
        "judge_ids": sorted(judges),
    }


# =========================
# 3. 宽转长 + 汇总
# =========================
def build_long_judge_scores(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """宽表转长表：每行 = 1 个评委在某周对某 pair 的分数。"""
    score_cols = schema["score_cols"]

    # 将 score 列统一转为数值（非数值 -> NaN）
    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")

    long_df = df.melt(
        id_vars=["season", "pair_id"],
        value_vars=score_cols,
        var_name="score_col",
        value_name="judge_score",
    )

    week_map = {k: v[0] for k, v in schema["score_col_map"].items()}
    judge_map = {k: v[1] for k, v in schema["score_col_map"].items()}

    long_df["week"] = long_df["score_col"].map(week_map).astype(int)
    long_df["judge_id"] = long_df["score_col"].map(judge_map).astype(int)
    long_df.drop(columns=["score_col"], inplace=True)

    long_df["judge_score"] = long_df["judge_score"].astype(float)
    long_df["is_score_present"] = long_df["judge_score"].notna()

    # 负分检查
    if (long_df["judge_score"] < 0).any():
        warnings.warn("检测到负分，请检查原始数据。")

    return long_df[["season", "week", "pair_id", "judge_id", "judge_score", "is_score_present"]]


def build_judges_totals(long_df: pd.DataFrame) -> pd.DataFrame:
    """按 week 汇总评委总分与有效评委数。"""
    totals = (
        long_df.groupby(["season", "week", "pair_id"], as_index=False)
        .agg(
            J_score=("judge_score", lambda s: s.sum(min_count=1)),  # 全 NaN -> NaN
            num_judges_scored=("judge_score", "count"),  # 非 NaN 个数
        )
        .sort_values(["season", "week", "pair_id"])
    )
    totals["J_score_is_missing"] = totals["num_judges_scored"] == 0
    totals["J_score"] = totals["J_score"].astype(float)
    return totals


def infer_season_observed_weeks(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    关键：识别“每季有效周”
    week_exists=True 当且仅当该 season-week 至少出现过一个非 NaN 的 judge_score
    """
    sw = (
        long_df.groupby(["season", "week"], as_index=False)["is_score_present"]
        .any()
        .rename(columns={"is_score_present": "week_exists"})
    )
    return sw


def filter_to_observed_weeks(judges_totals: pd.DataFrame, season_week_exists: pd.DataFrame) -> pd.DataFrame:
    """只保留每季真实存在的周，杜绝“尾部一串 0”的假周污染。"""
    merged = judges_totals.merge(season_week_exists, on=["season", "week"], how="left")
    merged["week_exists"] = merged["week_exists"].fillna(False)
    filtered = merged[merged["week_exists"]].drop(columns=["week_exists"])
    return filtered


# =========================
# 4. active roster（按“有效周”）
# =========================
def build_active_roster(judges_totals_observed: pd.DataFrame) -> pd.DataFrame:
    """
    推断每个 pair 在每周是否仍在赛（只在 observed weeks 上定义）。
    last_active_week：最后一个 (num_judges_scored>0 AND J_score>0) 的周
    """
    totals = judges_totals_observed.copy()

    active_mask = (totals["num_judges_scored"] > 0) & (totals["J_score"] > 0)
    last_active = (
        totals.loc[active_mask]
        .groupby(["season", "pair_id"])["week"]
        .max()
        .reset_index(name="last_active_week")
    )

    all_pairs = totals.groupby(["season", "pair_id"], as_index=False).size()[["season", "pair_id"]]
    last_active = all_pairs.merge(last_active, on=["season", "pair_id"], how="left")

    if last_active["last_active_week"].isna().any():
        # 理论上不该发生（DWTS 每对至少一周有分），但做提示
        warnings.warn("存在 pair 从未出现正分，last_active_week 无法确定。")

    roster = totals.merge(last_active, on=["season", "pair_id"], how="left")

    # week <= last_active_week => active
    roster["is_active"] = roster["last_active_week"].notna() & (roster["week"] <= roster["last_active_week"])

    roster["last_active_week"] = roster["last_active_week"].astype("Int64")
    roster["elim_week"] = roster["last_active_week"]
    roster["active_reason"] = np.where(roster["last_active_week"].notna(), "J_score>0", "unknown_no_positive_scores")

    return roster[["season", "week", "pair_id", "is_active", "elim_week", "last_active_week", "active_reason"]]


# =========================
# 5. elimination events（按“有效周”）
# =========================
def infer_elimination_events(active_roster: pd.DataFrame, include_finale: bool = True) -> pd.DataFrame:
    """
    识别每周淘汰事件类型与被淘汰选手。
    注意：week 序列是“该 season 的 observed weeks”，不会出现虚假的尾部周。
    """
    events: list[dict] = []

    for season, g in active_roster.groupby("season"):
        # 用该季“存在的周”排序（不一定从 1 连续，但通常是连续）
        weeks = sorted(g["week"].unique())
        if not weeks:
            continue
        max_week = max(weeks)

        # 构建每周 active 集合
        week_active: dict[int, set] = {}
        for w in weeks:
            week_active[w] = set(g.loc[(g["week"] == w) & (g["is_active"]), "pair_id"].tolist())

        prev_week = None
        for w in weeks:
            if (not include_finale) and (w == max_week):
                continue

            if prev_week is None:
                events.append(
                    {
                        "season": int(season),
                        "week": int(w),
                        "elim_count": 0,
                        "eliminated_pair_ids": "",
                        "notes": "season_start",
                    }
                )
                prev_week = w
                continue

            prev_set = week_active[prev_week]
            cur_set = week_active[w]

            prev_count = len(prev_set)
            cur_count = len(cur_set)

            eliminated = sorted(prev_set - cur_set)
            added = sorted(cur_set - prev_set)

            raw_drop = prev_count - cur_count
            elim_count = int(raw_drop) if raw_drop > 0 else 0
            eliminated_ids = "|".join(eliminated) if elim_count > 0 else ""

            if cur_count < prev_count:
                if elim_count == 1:
                    notes = "single_elimination"
                elif elim_count == 2:
                    notes = "double_elimination"
                else:
                    # 只有在“有效周”下还 drop>2 才真正值得关注
                    notes = "anomaly_drop>2"
            elif cur_count == prev_count:
                notes = "no_elimination"
                if added:
                    notes = f"return_or_revival:new={'|'.join(added)}"
            else:
                notes = f"return_or_revival:new={'|'.join(added)}"
                elim_count = 0
                eliminated_ids = ""

            # finale：若最后一周出现多名“同时结束”，不要当异常；打标即可
            if w == max_week and (prev_count > 0) and (cur_count == 0 or elim_count > 0):
                notes = notes + ";finale_multi_exit"

            events.append(
                {
                    "season": int(season),
                    "week": int(w),
                    "elim_count": int(elim_count),
                    "eliminated_pair_ids": eliminated_ids,
                    "notes": notes,
                }
            )
            prev_week = w

    return pd.DataFrame(events)[["season", "week", "elim_count", "eliminated_pair_ids", "notes"]]


# =========================
# 6. QA & 输出
# =========================
def save_outputs(
    judges_long: pd.DataFrame,
    judges_totals: pd.DataFrame,
    active_roster: pd.DataFrame,
    elimination_events: pd.DataFrame,
    output_dir: Path,
    save_parquet: bool = False,
) -> dict:
    """保存 CSV（可选 parquet）。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "judges_scores_long": output_dir / "judges_scores_long.csv",
        "judges_totals": output_dir / "judges_totals.csv",
        "active_roster": output_dir / "active_roster.csv",
        "elimination_events": output_dir / "elimination_events.csv",
    }

    judges_long.to_csv(paths["judges_scores_long"], index=False)
    judges_totals.to_csv(paths["judges_totals"], index=False)
    active_roster.to_csv(paths["active_roster"], index=False)
    elimination_events.to_csv(paths["elimination_events"], index=False)

    if save_parquet:
        try:
            judges_long.to_parquet(output_dir / "judges_scores_long.parquet", index=False)
            judges_totals.to_parquet(output_dir / "judges_totals.parquet", index=False)
            active_roster.to_parquet(output_dir / "active_roster.parquet", index=False)
            elimination_events.to_parquet(output_dir / "elimination_events.parquet", index=False)
        except Exception as exc:
            warnings.warn(f"parquet 输出失败: {exc}")

    return paths


def export_model_inputs(judges_totals: pd.DataFrame, active_roster: pd.DataFrame, output_dir: Path) -> Path:
    """输出模型输入表：season, week, pair_id, J_score, is_active, elim_flag。"""
    merged = active_roster.merge(judges_totals, on=["season", "week", "pair_id"], how="left")
    merged = merged[["season", "week", "pair_id", "J_score", "is_active"]].sort_values(["season", "pair_id", "week"])

    # elim_flag：本周不再 active 且上周 active（按 observed weeks 的顺序）
    merged["prev_active"] = merged.groupby(["season", "pair_id"])["is_active"].shift(1)
    merged["elim_flag"] = (merged["prev_active"] == True) & (merged["is_active"] == False)
    merged["elim_flag"] = merged["elim_flag"].fillna(False)
    merged.drop(columns=["prev_active"], inplace=True)

    out = output_dir / "model_input.csv"
    merged.to_csv(out, index=False)
    return out


def run_quality_checks(
    judges_long: pd.DataFrame,
    judges_totals: pd.DataFrame,
    active_roster: pd.DataFrame,
    elimination_events: pd.DataFrame,
    season_week_exists: pd.DataFrame,
    output_paths: dict,
) -> dict:
    """执行质量检查并保存 JSON 报告。"""
    report: dict = {"per_season": {}, "data_consistency": {}, "output_files": {}}

    print("=== 质量检查 ===")
    # 每季按“有效周”输出 active_counts（不再用 range(1,max_week)）
    for season, g in active_roster.groupby("season"):
        weeks = sorted(g["week"].unique())
        total_pairs = int(g["pair_id"].nunique())

        active_counts = (
            g[g["is_active"]]
            .groupby("week")["pair_id"]
            .nunique()
            .reindex(weeks, fill_value=0)
            .tolist()
        )

        anomalies = []
        prev = None
        for w, count in zip(weeks, active_counts):
            if prev is not None:
                if count > prev:
                    anomalies.append({"week": int(w), "type": "count_increase", "delta": int(count - prev)})
                if prev - count > 2:
                    anomalies.append({"week": int(w), "type": "drop_gt_2", "delta": int(prev - count)})
            prev = count

        neg_weeks = (
            judges_long[(judges_long["season"] == season) & (judges_long["judge_score"] < 0)]["week"]
            .dropna()
            .unique()
            .tolist()
        )
        if neg_weeks:
            anomalies.append({"week": [int(x) for x in neg_weeks], "type": "negative_scores"})

        report["per_season"][str(int(season))] = {
            "total_pairs": total_pairs,
            "observed_weeks": [int(x) for x in weeks],
            "active_counts": [int(x) for x in active_counts],
            "anomalies": anomalies,
        }

        print(
            f"Season {int(season)}: total_pairs={total_pairs}, observed_weeks={weeks}, "
            f"active_counts={active_counts}"
        )
        if anomalies:
            print(f"  anomalies: {anomalies}")

    # 数据一致性检查
    merged = active_roster.merge(judges_totals, on=["season", "week", "pair_id"], how="left")

    # 1) active 且 J_score == 0 但有评委打分（理论上不应出现）
    cond_zero = merged["is_active"] & (merged["num_judges_scored"] > 0) & (merged["J_score"] == 0)
    zero_count = int(cond_zero.sum())
    zero_samples = merged.loc[cond_zero, ["season", "week", "pair_id"]].head(5).to_dict(orient="records")

    # 2) 出局后仍出现正分（说明 last_active_week 推断错误）
    last_week = merged["last_active_week"].fillna(-1)
    cond_after_elim = (merged["week"] > last_week) & (merged["J_score"] > 0)
    after_elim_count = int(cond_after_elim.sum())
    after_elim_samples = merged.loc[cond_after_elim, ["season", "week", "pair_id"]].head(5).to_dict(orient="records")

    # 3) 缺分周比例（该数据看起来为 0，但逻辑保留）
    cond_missing = merged["is_active"] & (merged["num_judges_scored"] == 0)
    missing_count = int(cond_missing.sum())
    active_rows = int(merged["is_active"].sum())
    missing_ratio = missing_count / active_rows if active_rows else 0.0

    # 4) “无效周”统计（这项是本次修复的关键证据）
    invalid_weeks = season_week_exists[~season_week_exists["week_exists"]]
    invalid_weeks_count = int(len(invalid_weeks))

    report["data_consistency"] = {
        "active_with_zero_score_count": zero_count,
        "active_with_zero_score_samples": zero_samples,
        "positive_score_after_elim_count": after_elim_count,
        "positive_score_after_elim_samples": after_elim_samples,
        "missing_score_weeks_active_count": missing_count,
        "missing_score_weeks_active_ratio": missing_ratio,
        "invalid_season_week_rows_in_long": invalid_weeks_count,
    }

    print(f"active 内 J_score==0 且 num_judges_scored>0 的数量: {zero_count}")
    print(f"出局后仍有正分的数量: {after_elim_count}")
    print(f"缺分周(仍 active) 数量: {missing_count}, 比例: {missing_ratio:.4f}")
    print(f"无效 season-week（全体 NaN 的 week）数量（用于证明已截断假周）: {invalid_weeks_count}")

    # 输出文件存在性与行数
    for key, value in output_paths.items():
        if isinstance(value, Path):
            report["output_files"][key] = {"path": str(value), "exists": value.exists()}

    report["output_files"]["rows_summary"] = {
        "judges_scores_long": int(len(judges_long)),
        "judges_totals": int(len(judges_totals)),
        "active_roster": int(len(active_roster)),
        "elimination_events": int(len(elimination_events)),
    }

    return report


# =========================
# 7. 主程序
# =========================
def main() -> None:
    logging.basicConfig(level=logging.INFO)

    script_dir = Path(__file__).resolve().parent

    # 你可以改成绝对路径；默认按你当前工程结构读取
    DATA_PATH = script_dir / "data" / "2026_MCM_Problem_C_Data.csv"

    df = load_data(DATA_PATH)
    schema = detect_schema(df)

    # 标准化 season / pair_id
    df["season"] = schema["season_series"]
    df = df[df["season"].notna()].copy()  # 丢弃无法解析 season 的行（否则后续 groupby 乱）
    df["season"] = df["season"].astype("Int64")

    raw_pid = schema["pair_id_series_raw"]
    raw_pid = raw_pid.loc[df.index]  # 对齐过滤后的 df
    df["pair_id"] = _sanitize_pair_id(df["season"], raw_pid)

    # 唯一性检查（评审很看重）
    dup = pd.DataFrame({"season": df["season"], "pair_id": df["pair_id"]}).duplicated().sum()
    if dup > 0:
        warnings.warn(f"检测到 season+pair_id 仍有重复（{dup} 条），建议检查姓名字段是否存在完全相同组合。")

    print(
        "解析到的 week-judge-score 列数: "
        f"{len(schema['score_cols'])}, 全局最大 week: {schema['week_max']}, "
        f"季数: {df['season'].nunique()}, pair 数: {df['pair_id'].nunique()}"
    )

    judges_long = build_long_judge_scores(df, schema)
    judges_totals = build_judges_totals(judges_long)

    # —— 关键修复：每季有效周识别 + 过滤 —— #
    season_week_exists = infer_season_observed_weeks(judges_long)
    judges_totals_obs = filter_to_observed_weeks(judges_totals, season_week_exists)

    active_roster = build_active_roster(judges_totals_obs)
    elimination_events = infer_elimination_events(active_roster, include_finale=True)

    output_dir = script_dir / "data" / "process"
    output_paths = save_outputs(judges_long, judges_totals_obs, active_roster, elimination_events, output_dir)

    model_input_path = export_model_inputs(judges_totals_obs, active_roster, output_dir)
    output_paths["model_input"] = model_input_path

    report = run_quality_checks(
        judges_long,
        judges_totals_obs,
        active_roster,
        elimination_events,
        season_week_exists,
        output_paths,
    )

    report_path = output_dir / "preprocess_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"质量检查报告已保存: {report_path}")


if __name__ == "__main__":
    main()
