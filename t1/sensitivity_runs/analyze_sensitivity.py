#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze sensitivity grid and create paper-ready outputs.

Outputs:
- Table S1 (CSV + Markdown) with key metrics
- Fig S1: kappa vs consistency_mean (lines by alpha0)
- Fig S2: kappa vs rel_CI_width_median (lines by alpha0)
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sensitivity grid and create Table S1/Fig S1/Fig S2.")
    parser.add_argument("--grid", type=str, default="sensitivity_grid.csv",
                        help="Path to sensitivity_grid.csv")
    parser.add_argument("--out_dir", type=str, default="sensitivity_outputs",
                        help="Output directory for tables/figures")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    return parser.parse_args()


def resolve_grid(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    script_dir = Path(__file__).resolve().parent
    alt = script_dir / path_str
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Grid file not found: {path_str}")


def ensure_column(df: pd.DataFrame, target: str, fallbacks: List[str]) -> None:
    if target in df.columns:
        return
    for col in fallbacks:
        if col in df.columns:
            df[target] = df[col]
            return
    raise KeyError(f"Missing required column: {target} (fallbacks tried: {fallbacks})")


def prepare_grid(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()

    # Keep only successful runs when status is available.
    if "status" in out.columns:
        out = out[out["status"] == "ok"].copy()

    # Canonical columns with fallbacks for older grids.
    ensure_column(out, "consistency_map", ["overall_consistency_map"])
    ensure_column(out, "consistency_mean", ["overall_consistency_mean"])
    ensure_column(out, "accept_rate_median", [])
    ensure_column(out, "rel_CI_width_median", ["rel_ci_width_median"])
    ensure_column(out, "margin_median", ["margin_map_median"])

    # Coerce to numeric (safe for plotting and sorting).
    for col in ["alpha0", "kappa", "consistency_map", "consistency_mean",
                "accept_rate_median", "rel_CI_width_median", "margin_median"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values(["alpha0", "kappa"]).reset_index(drop=True)
    return out


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_line, sep_line]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_table_s1(grid: pd.DataFrame, out_dir: Path) -> None:
    cols = [
        "alpha0",
        "kappa",
        "consistency_map",
        "consistency_mean",
        "accept_rate_median",
        "rel_CI_width_median",
        "margin_median",
    ]
    table = grid[cols].copy()

    out_csv = out_dir / "table_S1.csv"
    table.to_csv(out_csv, index=False, encoding="utf-8")

    # Markdown version with rounding for easy paper pasting.
    md_table = table.copy()
    md_table["alpha0"] = md_table["alpha0"].map(lambda x: f"{x:.0f}")
    md_table["kappa"] = md_table["kappa"].map(lambda x: f"{x:.0f}")
    for col in [
        "consistency_map",
        "consistency_mean",
        "accept_rate_median",
        "rel_CI_width_median",
        "margin_median",
    ]:
        md_table[col] = md_table[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    out_md = out_dir / "table_S1.md"
    out_md.write_text(dataframe_to_markdown(md_table), encoding="utf-8")


def plot_lines(grid: pd.DataFrame, y_col: str, ylabel: str, title: str, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for alpha0 in sorted(grid["alpha0"].dropna().unique()):
        subset = grid[grid["alpha0"] == alpha0].sort_values("kappa")
        ax.plot(subset["kappa"], subset[y_col], marker="o", label=f"alpha0={alpha0:.0f}")

    ax.set_xlabel("kappa")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="alpha0", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    grid_path = resolve_grid(args.grid)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(grid_path)
    grid = prepare_grid(grid)

    write_table_s1(grid, out_dir)

    plot_lines(
        grid=grid,
        y_col="consistency_mean",
        ylabel="Consistency (mean)",
        title="Fig S1: Consistency vs kappa",
        out_path=out_dir / "fig_S1_consistency_vs_kappa.png",
        dpi=args.dpi,
    )

    plot_lines(
        grid=grid,
        y_col="rel_CI_width_median",
        ylabel="Median Relative CI Width",
        title="Fig S2: Relative CI Width vs kappa",
        out_path=out_dir / "fig_S2_rel_ci_vs_kappa.png",
        dpi=args.dpi,
    )

    print("[OK] Wrote Table S1 and Fig S1/Fig S2 outputs to:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
