# MCM 2026 Problem C: DWTS 投票与淘汰机制分析

本项目围绕《Dancing with the Stars》历史赛季数据，构建了一套从评委分数和淘汰结果反推粉丝投票、比较不同淘汰规则、分析争议案例与角色效应、并设计改进淘汰方法的完整研究流程。代码和结果按题目拆分为 `t1` 到 `t4` 四个主模块，适合直接用于论文写作、结果复核和后续扩展。

## 项目回答的问题

- `Q1 / t1`：在粉丝投票不公开的情况下，如何反推每周每位选手的 `fan vote share`，并量化不确定性？
- `Q2 / t2_1 + t2_2`：`Rank` 合并法和 `Percent` 合并法有何差异？哪一种更贴近粉丝偏好？`Judges Save` 会带来什么影响？
- `Q3 / t3`：职业舞者和明星特征对评委评分与粉丝投票的影响是否一致？谁更决定“能走多远”？
- `Q4 / t4`：如果重新设计淘汰方法，哪种方法在整体预测和极端分歧周里表现更好？

## 数据范围

- 原始数据：[`data/2026_MCM_Problem_C_Data.csv`](data/2026_MCM_Problem_C_Data.csv)
- 覆盖赛季：34 个赛季
- 处理后的 season-specific `pair_id`：421 个
- 唯一明星数：408 个
- 预处理后 `season-week-pair` 记录：4199 条
- 识别出的淘汰事件：303 次
- 预处理阶段剔除的无效 `season-week` 块：39 个

## 核心结论

- `Q1`：基于 `ABC + Soft ABC + 动态 Dirichlet 先验` 的反推模型在 301 个评估周上的平均 `pp_consistency` 为 `0.6248`，`p_map` 的可行性一致率为 `1.0000`，`p_mean` 一致率为 `0.9967`。`Rank` 赛制下的不确定性高于 `Percent` 赛制。
- `Q1`：敏感性分析表明，`alpha0` 和 `kappa` 的不同组合不会改变“模型可行”这一结论；更大的 `kappa` 能显著缩小相对置信区间。
- `Q2`：`Rank` 与 `Percent` 在部分周次会给出不同淘汰结果；在基于 `fan_share` 重新筛出的 17 个真正“翻盘周”中，`Percent` 方法 100% 淘汰了粉丝票份额更低的选手，说明它在竞争激烈、边际很小的场景下更直接反映粉丝偏好幅度。
- `Q2`：争议案例的反事实模拟表明，`Judges Save` 会系统性削弱部分高人气争议选手的优势，尤其对 `Jerry Rice`、`Bristol Palin`、`Bobby Bones` 这类案例更明显。
- `Q3`：评委模型 `R^2 = 0.6553`，粉丝模型 `R^2 = 0.4843`；职业舞者对粉丝投票的增量解释力 (`0.0445`) 明显高于对评委评分的增量解释力 (`0.0179`)。两侧舞者效应相关系数仅 `0.0328`，几乎不一致。
- `Q4`：在 188 个有淘汰的周中，`Percent` 方法整体匹配率最高 (`76.60%`)；在 67 个极端分歧周中，`Uncertainty+Geometric` 的极端命中率最高 (`20.90%`)。

## 目录结构

- [`data/`](data)：原始数据、预处理脚本与中间表。
- [`t1/`](t1)：Q1 反推粉丝投票份额，含主结果、全局指标与敏感性分析。
- [`t2_1/`](t2_1)：Q2.1 `Rank` vs `Percent` 的跨季比较、机制检验与图表。
- [`t2_2/`](t2_2)：Q2.2 争议选手反事实模拟与案例图。
- [`t3/`](t3)：Q3 职业舞者与明星特征对评委/粉丝影响的回归分析。
- [`t4/`](t4)：Q4 重放模拟与淘汰方法对比。
- [`t2/`](t2)：较早的 notebook 版本与中间导出，可视为实验草稿区。
- [`人气/`](人气)：基于 Wikipedia pageviews 的外部人气代理变量采集脚本。

## 主要输出文件

- `data/process/model_input.csv`：后续模块统一使用的周级建模输入表。
- `t1/fan_vote_estimates.csv`：每位选手每周的 `p_mean / p_map / CI` 等估计结果。
- `t1/weekly_summary.csv`：Q1 的周级一致性、不确定性与采样状态汇总。
- `t2_1/out_question/weekly_method_comparison.csv`：`Rank` 与 `Percent` 的逐周淘汰对比。
- `t2_1/fan_share_favor_summary.csv`：基于 `fan_share` 的“偏粉丝”稳健性结论。
- `t2_2/controversy_counterfactual_summary.csv`：争议案例的 Monte Carlo 反事实汇总。
- `t3/outputs/coef_compare_all.csv`：评委侧与粉丝侧系数对比。
- `t3/outputs/performance_weeks_model.csv`：`weeks_survived` 模型及增量 `R^2`。
- `t4/outputs/q4_method_summary.csv`：四种淘汰方法的总对比表。

## 环境依赖

建议使用 Python 3.11+ 或 3.12。核心依赖如下：

```bash
pip install numpy pandas matplotlib statsmodels scipy requests jupyter
```

可选依赖：

```bash
pip install scienceplots
```

## 推荐运行顺序

### 1. 预处理原始数据

```bash
python data/preprocess_dwts.py
```

输出到 `data/process/`，生成：

- `judges_scores_long.csv`
- `judges_totals.csv`
- `active_roster.csv`
- `elimination_events.csv`
- `model_input.csv`
- `preprocess_report.json`

### 2. 运行 Q1：反推粉丝投票份额

从 `t1` 目录运行，避免结果写到仓库根目录：

```bash
cd t1
python estimate_fan_votes.py --data_path ../data/2026_MCM_Problem_C_Data.csv
cd ..
```

主要输出：

- `t1/fan_vote_estimates.csv`
- `t1/weekly_summary.csv`
- `t1/global_metrics.json`

敏感性分析结果已随仓库附带在 `t1/sensitivity_runs/` 与 `t1/sensitivity_runs/sensitivity_outputs/` 中；若要重新生成，建议先检查 `run_sensitivity.py` 与当前 `estimate_fan_votes.py` 的参数接口是否一致，再批量运行。

### 3. 运行 Q2.1：比较 Rank 与 Percent

```bash
python t2_1/analysis_voting_methods.py --model_input data/process/model_input.csv --elims data/process/elimination_events.csv --fan t1/fan_vote_estimates.csv --p-col p_mean --out-dir t2_1/out_question
python t2_1/fan_share_favor_analysis.py --weekly-compare t2_1/out_question/weekly_method_comparison.csv --weekly-contestant t2_1/out_question/weekly_contestant_metrics.csv --out-dir t2_1
python t2_1/t2_build_season_summary_and_figs.py --diff-summary t2_1/out_question/season_method_diff_summary.csv --week-completeness t2_1/out_question/week_data_completeness.csv --elim-metrics t2_1/out_question/elim_rank_metrics.csv --weekly-compare t2_1/out_question/weekly_method_comparison.csv --weekly-contestant t2_1/out_question/weekly_contestant_metrics.csv --out-dir t2_1/out_question --plot-judge-delta
```

### 4. 运行 Q2.2：争议案例反事实模拟

```bash
python t2_2/2.2.py --outdir t2_2
```

### 5. 运行 Q3：职业舞者与明星特征影响分析

```bash
python t3/t3.py
```

输出到 `t3/outputs/`。

### 6. 运行 Q4：重放模拟与淘汰方法设计

```bash
python t4/t4.py --root . --data-dir t4/data --out-dir t4/outputs
```

说明：

- 该步骤会先构造 Q4 所需输入，再执行 replay analysis。
- 依赖 `t1/` 与 `t2_1/` 的已有结果文件。

## 各模块摘要

### `t1`：粉丝投票反推

- 核心方法：`Hard ABC + Soft ABC + 动态先验 + MAP 选择`
- 重点文件：`estimate_fan_votes.py`
- 当前仓库中已有结果：2659 条选手-周估计记录、301 个周级汇总结果
- 结果解释建议：后续分析优先使用 `p_map`，趋势展示可参考 `p_mean`

### `t2_1`：赛制比较

- 核心问题：`Rank` 与 `Percent` 是否会改变淘汰结果，以及哪种更偏向粉丝
- 重点文件：`analysis_voting_methods.py`、`fan_share_favor_analysis.py`、`t2_build_season_summary_and_figs.py`
- 代表性发现：`Percent` 在整体上略偏粉丝，但真正重要的是它在分歧周里系统性地更贴近粉丝投票幅度

### `t2_2`：争议案例

- 核心方法：使用 `p_map / p_mean / CI` 做 Monte Carlo 重放
- 当前案例：`Jerry Rice`、`Billy Ray Cyrus`、`Bristol Palin`、`Bobby Bones`
- 输出包括每个案例的争议曲线图和名次分布图

### `t3`：角色与特征影响

- 核心方法：`OLS / WLS + cluster-robust SE`
- 评委侧与粉丝侧分别建模，再比较同一变量在两侧的方向和显著性
- 额外构建 `weeks_survived` 模型，量化“表现”“职业舞者”“明星特征”的增量贡献

### `t4`：淘汰方法设计

- 比较方法：`Rank`、`Percent`、`Dynamic+JudgeSave`、`Uncertainty+Geometric`
- 核心输出：`q4_method_summary.csv`、`q4_replay_df.csv`、`q4_diff_weeks.csv`、`q4_focus_weeks.csv`
- 推荐结论：标准场景优先 `Percent`，极端分歧场景可考虑 `Uncertainty+Geometric`

## 补充说明

- 仓库内部分中文总结 Markdown 的编码不完全统一，终端下可能出现乱码；论文写作或结果核对时，建议优先以 `.py` 脚本和 `.csv` 输出为准。
- `t2/` 与部分压缩包、草稿文件属于历史实验痕迹，不影响主流程复现。
- `人气/` 目录提供了一个独立的外部人气抓取思路，目前未纳入主分析管线。

## 参考阅读顺序

如果你只是想快速理解项目，建议按下面顺序看：

1. 本 README
2. `data/process/preprocess_report.json`
3. `t1/weekly_summary.csv`
4. `t2_1/fan_share_favor_summary.csv`
5. `t3/outputs/quick_summary.txt`
6. `t4/outputs/q4_method_summary.csv`
