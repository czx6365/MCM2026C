#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析敏感性分析结果，生成总结报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # 读取敏感性分析汇总表
    grid = pd.read_csv("sensitivity_grid.csv")
    
    print("=" * 80)
    print("敏感性分析结果总结")
    print("=" * 80)
    
    # 1. 一致性分析
    print("\n【1. 一致性分析】")
    print(f"所有参数组合的 consistency_map = {grid['overall_consistency_map'].unique()}")
    print(f"consistency_mean 范围: {grid['overall_consistency_mean'].min():.4f} - {grid['overall_consistency_mean'].max():.4f}")
    
    # 找出有失配的参数组合
    mismatch = grid[grid['overall_consistency_mean'] < 1.0]
    if len(mismatch) > 0:
        print(f"\n发现 {len(mismatch)} 组参数存在 consistency_mean < 1.0:")
        for _, row in mismatch.iterrows():
            print(f"  alpha0={row['alpha0']:.1f}, kappa={row['kappa']:.1f}: "
                  f"consistency_mean={row['overall_consistency_mean']:.4f}, "
                  f"Rank={row['consistency_mean_rank']:.4f}, "
                  f"Percent={row['consistency_mean_percent']:.4f}")
    
    # 2. 接受率分析
    print("\n【2. ABC接受率分析】")
    print(f"平均接受率范围: {grid['accept_rate_mean'].min():.4f} - {grid['accept_rate_mean'].max():.4f}")
    print(f"中位数接受率范围: {grid['accept_rate_median'].min():.4f} - {grid['accept_rate_median'].max():.4f}")
    print(f"最小接受率范围: {grid['accept_rate_min'].min():.6f} - {grid['accept_rate_min'].max():.6f}")
    
    # 按alpha0和kappa分组分析
    print("\n按 alpha0 分组:")
    for alpha0 in sorted(grid['alpha0'].unique()):
        subset = grid[grid['alpha0'] == alpha0]
        print(f"  alpha0={alpha0:.1f}: 平均接受率={subset['accept_rate_mean'].mean():.4f} "
              f"(范围: {subset['accept_rate_mean'].min():.4f}-{subset['accept_rate_mean'].max():.4f})")
    
    print("\n按 kappa 分组:")
    for kappa in sorted(grid['kappa'].unique()):
        subset = grid[grid['kappa'] == kappa]
        print(f"  kappa={kappa:.1f}: 平均接受率={subset['accept_rate_mean'].mean():.4f} "
              f"(范围: {subset['accept_rate_mean'].min():.4f}-{subset['accept_rate_mean'].max():.4f})")
    
    # 3. 不确定性分析（CI宽度）
    print("\n【3. 不确定性分析（CI宽度）】")
    print(f"平均CI宽度范围: {grid['ci_width_mean'].min():.4f} - {grid['ci_width_mean'].max():.4f}")
    print(f"中位数CI宽度范围: {grid['ci_width_median'].min():.4f} - {grid['ci_width_median'].max():.4f}")
    
    print("\n按 kappa 对CI宽度的影响:")
    for kappa in sorted(grid['kappa'].unique()):
        subset = grid[grid['kappa'] == kappa]
        print(f"  kappa={kappa:.1f}: 平均CI宽度={subset['ci_width_mean'].mean():.4f} "
              f"(范围: {subset['ci_width_mean'].min():.4f}-{subset['ci_width_mean'].max():.4f})")
    
    # 4. 相对CI宽度分析
    print("\n【4. 相对不确定性分析（相对CI宽度）】")
    print(f"平均相对CI宽度范围: {grid['rel_ci_width_mean'].min():.4f} - {grid['rel_ci_width_mean'].max():.4f}")
    print(f"中位数相对CI宽度范围: {grid['rel_ci_width_median'].min():.4f} - {grid['rel_ci_width_median'].max():.4f}")
    
    # 5. 淘汰边界稳健性（margin）
    print("\n【5. 淘汰边界稳健性（margin_map）】")
    print(f"中位数margin范围: {grid['margin_map_median'].min():.6f} - {grid['margin_map_median'].max():.6f}")
    print(f"最小margin范围: {grid['margin_map_min'].min():.6f} - {grid['margin_map_min'].max():.6f}")
    
    # 6. 参数敏感性总结
    print("\n【6. 参数敏感性总结】")
    
    # alpha0的影响
    alpha0_effect = grid.groupby('alpha0').agg({
        'overall_consistency_mean': 'mean',
        'accept_rate_mean': 'mean',
        'ci_width_mean': 'mean',
        'rel_ci_width_mean': 'mean'
    })
    print("\nalpha0 的影响（平均值）:")
    print(alpha0_effect.to_string())
    
    # kappa的影响
    kappa_effect = grid.groupby('kappa').agg({
        'overall_consistency_mean': 'mean',
        'accept_rate_mean': 'mean',
        'ci_width_mean': 'mean',
        'rel_ci_width_mean': 'mean'
    })
    print("\nkappa 的影响（平均值）:")
    print(kappa_effect.to_string())
    
    # 7. 最优参数推荐
    print("\n【7. 参数选择建议】")
    
    # 综合考虑：一致性高、不确定性低、接受率合理
    # 计算综合得分（一致性权重0.5，相对CI宽度权重0.3，接受率权重0.2）
    grid['score'] = (
        0.5 * grid['overall_consistency_mean'] +
        0.3 * (1.0 / (1.0 + grid['rel_ci_width_mean'])) +  # 相对CI越小越好
        0.2 * grid['accept_rate_mean']
    )
    
    best = grid.loc[grid['score'].idxmax()]
    print(f"\n综合最优参数组合:")
    print(f"  alpha0 = {best['alpha0']:.1f}")
    print(f"  kappa = {best['kappa']:.1f}")
    print(f"  consistency_mean = {best['overall_consistency_mean']:.4f}")
    print(f"  consistency_map = {best['overall_consistency_map']:.4f}")
    print(f"  平均接受率 = {best['accept_rate_mean']:.4f}")
    print(f"  平均CI宽度 = {best['ci_width_mean']:.4f}")
    print(f"  平均相对CI宽度 = {best['rel_ci_width_mean']:.4f}")
    
    # 8. 可视化（如果可能）
    print("\n【8. 生成可视化图表】")
    out_dir = "图"
    os.makedirs(out_dir, exist_ok=True)
    
    # 图1: alpha0 vs kappa 对一致性的影响
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 准备数据（pivot table）
    pivot_cons = grid.pivot(index='alpha0', columns='kappa', values='overall_consistency_mean')
    pivot_ci = grid.pivot(index='alpha0', columns='kappa', values='ci_width_mean')
    pivot_rel_ci = grid.pivot(index='alpha0', columns='kappa', values='rel_ci_width_mean')
    pivot_acc = grid.pivot(index='alpha0', columns='kappa', values='accept_rate_mean')
    
    # 子图1: 一致性
    im1 = axes[0, 0].imshow(pivot_cons.values, aspect='auto', cmap='RdYlGn', vmin=0.99, vmax=1.0)
    axes[0, 0].set_xticks(range(len(pivot_cons.columns)))
    axes[0, 0].set_xticklabels([f"{k:.0f}" for k in pivot_cons.columns])
    axes[0, 0].set_yticks(range(len(pivot_cons.index)))
    axes[0, 0].set_yticklabels([f"{a:.0f}" for a in pivot_cons.index])
    axes[0, 0].set_xlabel('kappa')
    axes[0, 0].set_ylabel('alpha0')
    axes[0, 0].set_title('Consistency (mean)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 子图2: CI宽度
    im2 = axes[0, 1].imshow(pivot_ci.values, aspect='auto', cmap='YlOrRd')
    axes[0, 1].set_xticks(range(len(pivot_ci.columns)))
    axes[0, 1].set_xticklabels([f"{k:.0f}" for k in pivot_ci.columns])
    axes[0, 1].set_yticks(range(len(pivot_ci.index)))
    axes[0, 1].set_yticklabels([f"{a:.0f}" for a in pivot_ci.index])
    axes[0, 1].set_xlabel('kappa')
    axes[0, 1].set_ylabel('alpha0')
    axes[0, 1].set_title('Mean CI Width')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 子图3: 相对CI宽度
    im3 = axes[1, 0].imshow(pivot_rel_ci.values, aspect='auto', cmap='YlOrRd')
    axes[1, 0].set_xticks(range(len(pivot_rel_ci.columns)))
    axes[1, 0].set_xticklabels([f"{k:.0f}" for k in pivot_rel_ci.columns])
    axes[1, 0].set_yticks(range(len(pivot_rel_ci.index)))
    axes[1, 0].set_yticklabels([f"{a:.0f}" for a in pivot_rel_ci.index])
    axes[1, 0].set_xlabel('kappa')
    axes[1, 0].set_ylabel('alpha0')
    axes[1, 0].set_title('Mean Relative CI Width')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 子图4: 接受率
    im4 = axes[1, 1].imshow(pivot_acc.values, aspect='auto', cmap='viridis')
    axes[1, 1].set_xticks(range(len(pivot_acc.columns)))
    axes[1, 1].set_xticklabels([f"{k:.0f}" for k in pivot_acc.columns])
    axes[1, 1].set_yticks(range(len(pivot_acc.index)))
    axes[1, 1].set_yticklabels([f"{a:.0f}" for a in pivot_acc.index])
    axes[1, 1].set_xlabel('kappa')
    axes[1, 1].set_ylabel('alpha0')
    axes[1, 1].set_title('Mean Accept Rate')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sensitivity_heatmap.png"), dpi=300)
    print(f"  已保存: {out_dir}/sensitivity_heatmap.png")
    
    # 图2: kappa对CI宽度的影响（线图）
    fig, ax = plt.subplots(figsize=(10, 6))
    for alpha0 in sorted(grid['alpha0'].unique()):
        subset = grid[grid['alpha0'] == alpha0].sort_values('kappa')
        ax.plot(subset['kappa'], subset['ci_width_mean'], marker='o', label=f'alpha0={alpha0:.1f}')
    ax.set_xlabel('kappa')
    ax.set_ylabel('Mean CI Width')
    ax.set_title('Effect of kappa on uncertainty (CI width)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sensitivity_kappa_vs_ci.png"), dpi=300)
    print(f"  已保存: {out_dir}/sensitivity_kappa_vs_ci.png")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()

