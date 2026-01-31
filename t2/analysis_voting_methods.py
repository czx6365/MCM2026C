import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_celebrity(pair_id: str) -> str:
    if not isinstance(pair_id, str):
        return pair_id
    m = re.match(r"^S\d+_(.*?) __ ", pair_id)
    return m.group(1) if m else pair_id


def assign_rank(df: pd.DataFrame, value_col: str, higher_is_better: bool = True) -> pd.Series:
    # Deterministic ranking: sort by value then name
    ascending = not higher_is_better
    ordered = df.sort_values([value_col, "celebrity_name"], ascending=[ascending, True])
    ranks = pd.Series(range(1, len(ordered) + 1), index=ordered.index)
    return ranks.reindex(df.index)


def select_bottom(df: pd.DataFrame, value_col: str, n: int, lower_is_worse: bool = True) -> list:
    # lower_is_worse=True -> sort ascending (lowest) for percent method
    ascending = lower_is_worse
    ordered = df.sort_values([value_col, "celebrity_name"], ascending=[ascending, True])
    return ordered.head(n)["celebrity_name"].tolist()


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

    model = pd.read_csv(args.model_input)
    fan = pd.read_csv(args.fan)
    elims = pd.read_csv(args.elims)

    model["celebrity_name"] = model["pair_id"].apply(parse_celebrity)

    # Only weeks where contestants are active (performed that week)
    model = model[model["is_active"] == True].copy()

    # Merge fan vote estimates
    fan_use = fan[["season", "week", "celebrity_name", args.p_col]].copy()
    fan_use = fan_use.rename(columns={args.p_col: "fan_share"})
    merged = model.merge(fan_use, on=["season", "week", "celebrity_name"], how="left")

    # Attach elimination info
    # NOTE: elimination_events.week refers to the "elimination result week", while scores/fan votes
    # are tied to the performance week. We shift elimination week by -1 to align with performance data.
    elims_shift = elims[["season", "week", "elim_count", "eliminated_pair_ids", "notes"]].copy()
    elims_shift["week"] = elims_shift["week"].astype(int) - 1
    elims_shift = elims_shift[elims_shift["week"] > 0]
    merged = merged.merge(elims_shift, on=["season", "week"], how="left")

    # Identify weeks with elimination and complete fan data
    merged["has_elim"] = merged["elim_count"].fillna(0).astype(int) > 0
    week_status = (
        merged.groupby(["season", "week"], as_index=False)
        .agg(has_elim=("has_elim", "max"),
             any_missing_fan=("fan_share", lambda s: s.isna().any()),
             n_active=("celebrity_name", "count"))
    )

    # Per-week computations
    weekly_rows = []
    detailed_rows = []
    for (season, week), g in merged.groupby(["season", "week"]):
        g = g.copy()
        has_elim = int(g["elim_count"].fillna(0).iloc[0]) > 0
        if not has_elim:
            continue
        # Skip weeks with missing fan data
        if g["fan_share"].isna().any():
            continue

        # Ranks
        g["judge_rank"] = assign_rank(g, "J_score", higher_is_better=True)
        g["fan_rank"] = assign_rank(g, "fan_share", higher_is_better=True)
        g["rank_total"] = g["judge_rank"] + g["fan_rank"]

        # Percent method
        total_j = g["J_score"].sum()
        g["judge_pct"] = g["J_score"] / total_j if total_j > 0 else np.nan
        g["combined_pct"] = g["judge_pct"] + g["fan_share"]

        # Elimination selection
        elim_n = int(g["elim_count"].iloc[0])
        rank_elims = select_bottom(g, "rank_total", elim_n, lower_is_worse=False)
        pct_elims = select_bottom(g, "combined_pct", elim_n, lower_is_worse=True)

        # Bottom two and judge-save (only for single elimination weeks)
        rank_bottom_two = select_bottom(g, "rank_total", 2, lower_is_worse=False)
        pct_bottom_two = select_bottom(g, "combined_pct", 2, lower_is_worse=True)

        def judge_save(bottom_two):
            sub = g[g["celebrity_name"].isin(bottom_two)].copy()
            ordered = sub.sort_values(["J_score", "celebrity_name"], ascending=[True, True])
            return ordered.iloc[0]["celebrity_name"] if len(ordered) else None

        rank_judge_save = judge_save(rank_bottom_two) if elim_n == 1 else None
        pct_judge_save = judge_save(pct_bottom_two) if elim_n == 1 else None

        # Actual eliminated names
        actual_ids = g["eliminated_pair_ids"].iloc[0]
        actual_names = []
        if isinstance(actual_ids, str) and actual_ids.strip():
            for pid in actual_ids.split("|"):
                actual_names.append(parse_celebrity(pid.strip()))

        weekly_rows.append({
            "season": season,
            "week": week,
            "elim_count": elim_n,
            "rank_eliminated": "|".join(rank_elims),
            "percent_eliminated": "|".join(pct_elims),
            "rank_bottom_two": "|".join(rank_bottom_two),
            "percent_bottom_two": "|".join(pct_bottom_two),
            "rank_judge_save_elim": rank_judge_save or "",
            "percent_judge_save_elim": pct_judge_save or "",
            "actual_eliminated": "|".join(actual_names),
        })

        detailed_rows.append(g)

    weekly = pd.DataFrame(weekly_rows)
    detailed = pd.concat(detailed_rows, ignore_index=True) if detailed_rows else pd.DataFrame()

    # Save detailed per-contestant metrics for elimination weeks
    detailed_out = detailed[[
        "season", "week", "celebrity_name", "pair_id", "J_score", "fan_share",
        "judge_rank", "fan_rank", "rank_total", "judge_pct", "combined_pct",
        "elim_count", "elim_flag"
    ]].copy()

    detailed_out.to_csv(out_dir / "weekly_contestant_metrics.csv", index=False)
    weekly.to_csv(out_dir / "weekly_method_comparison.csv", index=False)

    # Summary: differences between rank and percent
    if not weekly.empty:
        weekly["methods_differ"] = weekly["rank_eliminated"] != weekly["percent_eliminated"]
        season_summary = (
            weekly.groupby("season", as_index=False)
            .agg(weeks_with_elim=("week", "count"),
                 diff_weeks=("methods_differ", "sum"))
        )
        season_summary["diff_share"] = season_summary["diff_weeks"] / season_summary["weeks_with_elim"]
        season_summary.to_csv(out_dir / "season_method_diff_summary.csv", index=False)

        # Fan vs judge favoring metric
        elim_metrics = []
        for _, row in weekly.iterrows():
            season = row["season"]
            week = row["week"]
            g = detailed[(detailed["season"] == season) & (detailed["week"] == week)]
            def avg_rank(elim_list, col):
                if not elim_list:
                    return np.nan
                return g[g["celebrity_name"].isin(elim_list)][col].mean()
            rank_elims = row["rank_eliminated"].split("|") if row["rank_eliminated"] else []
            pct_elims = row["percent_eliminated"].split("|") if row["percent_eliminated"] else []
            elim_metrics.append({
                "season": season,
                "week": week,
                "rank_elim_fan_rank": avg_rank(rank_elims, "fan_rank"),
                "percent_elim_fan_rank": avg_rank(pct_elims, "fan_rank"),
                "rank_elim_judge_rank": avg_rank(rank_elims, "judge_rank"),
                "percent_elim_judge_rank": avg_rank(pct_elims, "judge_rank"),
            })
        elim_metrics = pd.DataFrame(elim_metrics)
        elim_metrics.to_csv(out_dir / "elim_rank_metrics.csv", index=False)

    # Week completeness report
    week_status.to_csv(out_dir / "week_data_completeness.csv", index=False)

    # Controversy case summary (can extend this list as needed)
    cases = [
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (11, "Bristol Palin"),
        (27, "Bobby Bones"),
    ]

    if not weekly.empty and not detailed.empty:
        weekly_rank = weekly.set_index(["season", "week"])["rank_eliminated"]
        weekly_pct = weekly.set_index(["season", "week"])["percent_eliminated"]
        weekly_rank_js = weekly.set_index(["season", "week"])["rank_judge_save_elim"]
        weekly_pct_js = weekly.set_index(["season", "week"])["percent_judge_save_elim"]

        rows = []
        for season, name in cases:
            g = detailed[(detailed["season"] == season) & (detailed["celebrity_name"] == name)].copy()
            if g.empty:
                continue
            g = g.sort_values("week")

            def first_week(pred_series, match_fn):
                for _, r in g.iterrows():
                    key = (r["season"], r["week"])
                    val = pred_series.get(key, "")
                    if match_fn(val):
                        return r["week"]
                return None

            rank_week = first_week(weekly_rank, lambda v: isinstance(v, str) and name in v.split("|"))
            pct_week = first_week(weekly_pct, lambda v: isinstance(v, str) and name in v.split("|"))
            rank_js_week = first_week(weekly_rank_js, lambda v: isinstance(v, str) and v == name)
            pct_js_week = first_week(weekly_pct_js, lambda v: isinstance(v, str) and v == name)

            w = weekly[weekly["season"] == season].copy()
            w["rank_bottom_two_has"] = (
                w["rank_bottom_two"].fillna("").str.split("|").apply(lambda xs: name in xs)
            )
            w["percent_bottom_two_has"] = (
                w["percent_bottom_two"].fillna("").str.split("|").apply(lambda xs: name in xs)
            )
            rank_bottom_weeks = w[w["rank_bottom_two_has"]]["week"].tolist()
            pct_bottom_weeks = w[w["percent_bottom_two_has"]]["week"].tolist()

            rows.append({
                "season": season,
                "celebrity_name": name,
                "rank_elim_week": rank_week,
                "percent_elim_week": pct_week,
                "rank_judge_save_week": rank_js_week,
                "percent_judge_save_week": pct_js_week,
                "rank_bottom_two_weeks": ",".join(str(x) for x in rank_bottom_weeks),
                "percent_bottom_two_weeks": ",".join(str(x) for x in pct_bottom_weeks),
                "weeks_competed_in_data": g["week"].max(),
            })

        pd.DataFrame(rows).to_csv(out_dir / "controversy_summary.csv", index=False)


if __name__ == "__main__":
    main() 

