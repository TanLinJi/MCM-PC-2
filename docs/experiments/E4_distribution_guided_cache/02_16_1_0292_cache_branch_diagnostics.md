# 02_16_1：02_9_2 Cache Branch 与 GPA Replacement 诊断

日期：2026-06-18

状态：已完成，结果已分析。

## 1. 实验目的

本实验不提出新方法，也不修改 `02_9_2` 的预测逻辑。目标是解释 `02_9_2` 为什么有效、以及它在哪些地方出错。

需要回答的问题：

```text
1. 02_9_2 的收益主要来自哪个分支？
2. GPA local cache 是帮助 final logits，还是在某些扰动上放大错误？
3. text distribution gate 到底是在拒绝错误伪标签，还是误拒绝正确低熵样本？
4. 下一步应该改 replacement gate、local cache logits，还是 posterior correction？
```

## 2. 基本设置

```text
实验编号：02_16_1
数据集：ModelNet-C
扰动等级：severity=2
扰动类型：add_global, add_local, dropout_global, dropout_local, rotate, scale, jitter
backbone：ULIP
载体：02_9_2 / E4-C-A0+E1-textdist-only
最终分类器：manual_full
最终 logits：Point-Cache voting，不变
```

严格复用 `02_9_2` 设置：

```text
E4_TEXT_DIST_PROMPT_SOURCE=manualfull_llm_dynamic_init
E4_TEXT_SCORE_WEIGHT=0.15
E4_SCORE_NORM_MODE=running_zscore
dynamic prompt count=10
manual_full:LLM = 0.75:0.25
prompt cache=results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json
```

## 3. 诊断内容

本实验额外记录以下分支的单独准确率：

```text
zero_shot_text_proto_dot：只用 manual_full zero-shot 文本原型点积
global_entropy_cache：只用 global entropy cache logits
gpa_global_cache_diag：只用 GPA global cache logits，诊断项，不参与正式 02_9_2 final logits
gpa_local_cache：只用 GPA-controlled local cache logits
negative_cache_penalty：只看 negative cache penalty logits
positive_cache_total：global entropy cache + GPA local cache
cache_total_signed：global entropy cache + GPA local cache - negative cache
final_logits：正式 02_9_2 final logits
norm_fusion_offline：离线归一化融合诊断项，不参与正式预测
```

GPA replacement event 额外记录：

```text
sample_index
class_index / pred
target
pseudo_label_correct
decision
new_entropy / old_entropy
new_visual_score / old_visual_score
new_text_score / old_text_score
new_joint_score / old_joint_score
joint_score_margin
```

## 4. 已实现文件

诊断模型：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/runners/E7_entropy_energy_alignment_multicache/model_e7_b3_diag_0292_textdist_cache_branch_diagnostics.py
```

说明：该模型是已有的 `02_9_2` 分支诊断实现，本次只补充了 GPA event 中的 `target` 与 `pseudo_label_correct` 字段。

02_16_1 runner 入口：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/runners/E4_distribution_guided_cache/run_02_16_1_ulip_modelnetc_s2_0292_cache_branch_diagnostics.py
```

运行脚本：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache/02_16_1_ulip_modelnetc_s2_0292_cache_branch_diagnostics.sh
```

## 5. 执行命令

在 `mcmpc` 环境中执行：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/02_16_1_ulip_modelnetc_s2_0292_cache_branch_diagnostics.sh 0
```

其中 `0` 表示当前单张 4090 的物理 GPU 0。

运行前可做 dry-run：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/02_16_1_ulip_modelnetc_s2_0292_cache_branch_diagnostics.sh 0 --dry-run
```

## 6. 结果保存位置

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/02_16_1_ulip_modelnetc_s2_0292_cache_branch_diagnostics/
```

核心输出：

```text
summary.csv
logs/
e7_b3_diag_0292_stats/
wandb/
```

`e7_b3_diag_0292_stats/` 中每个 corruption 会输出：

```text
*_e7_b3_diag_0292_stats.json
*_e7_b3_diag_0292_samples.jsonl
gpa_replacement_events_*.jsonl
```

## 7. 运行后分析重点

第一层：看 final accuracy 是否复现 `02_9_2` S2 水平。

```text
02_9_2 S2 average = 54.71
```

第二层：看分支贡献。

```text
final_logits 是否主要由 zero-shot 提升？
cache_total_signed 是否本身强？
gpa_local_cache 是否在 add_local / jitter 上强，但在 rotate / add_global 上弱？
negative_cache_penalty 是否有稳定贡献？
```

第三层：看 GPA replacement 的伪标签正确性。

```text
accepted/replace 中 pseudo_label_correct 的比例
reject_entropy 中 pseudo_label_correct 的比例
reject_joint 中 pseudo_label_correct 的比例
```

如果大量正确样本被 `reject_joint` 拒绝，下一步应做低熵可信回退。

如果大量错误样本被 `replace` 接收，下一步应做更强的一致性 veto。

如果 `gpa_local_cache` 分支本身经常低于 zero-shot，下一步应改 local cache logits 的使用方式，而不是继续调 text gate。

## 8. 完整性检查

运行结果目录：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/02_16_1_ulip_modelnetc_s2_0292_cache_branch_diagnostics/
```

完整性：

| 检查项 | 结果 |
|---|---:|
| `summary.csv` 数据行 | 7 |
| severity=2 corruption 数量 | 7 |
| `*_e7_b3_diag_0292_stats.json` 数量 | 7 |
| `*_e7_b3_diag_0292_samples.jsonl` 数量 | 7 |
| `gpa_replacement_events_*.jsonl` 数量 | 7 |

结论：本次诊断结果完整，可以用于分析 `02_9_2` 的 S2 行为。

## 9. 与 E0 / 02_9_2 的最终结果对比

| Corruption | E0 Point-Cache | 02_9_2 | 02_16_1 | 02_16_1 - E0 | 02_16_1 - 02_9_2 |
|---|---:|---:|---:|---:|---:|
| add_global | 47.81 | 47.89 | 48.06 | +0.25 | +0.17 |
| add_local | 46.68 | 50.85 | 50.93 | +4.25 | +0.08 |
| dropout_global | 59.20 | 59.12 | 59.12 | -0.08 | +0.00 |
| dropout_local | 56.69 | 57.21 | 57.46 | +0.77 | +0.25 |
| rotate | 62.07 | 61.30 | 60.98 | -1.09 | -0.32 |
| scale | 55.23 | 55.92 | 55.83 | +0.60 | -0.09 |
| jitter | 50.32 | 50.65 | 50.49 | +0.17 | -0.16 |
| **Average** | **54.00** | **54.71** | **54.70** | **+0.70** | **-0.01** |

结论：

1. `02_16_1` 平均准确率为 `54.6957`，基本复现 `02_9_2` 的 S2 平均 `54.7057`。
2. 因为 `02_16_1` 不改变预测逻辑，最终精度接近 `02_9_2` 是预期结果。
3. 细微差异来自诊断 wrapper / 浮点顺序 / 日志统计路径，不构成新方法收益。
4. 因此，本实验的价值不在最终精度，而在分支分解和 GPA replacement event 诊断。

## 10. 分支贡献分析

逐 corruption 分支准确率：

| Corruption | final | zero-shot | global cache | GPA global diag | GPA local | positive cache | signed cache | norm fusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| add_global | 48.06 | 33.63 | 42.59 | 43.68 | 41.94 | 44.98 | 46.03 | 47.45 |
| add_local | 50.93 | 43.96 | 44.61 | 49.19 | 45.95 | 47.77 | 48.50 | 50.77 |
| dropout_global | 59.12 | 54.70 | 50.85 | 52.39 | 50.08 | 51.90 | 51.99 | 57.17 |
| dropout_local | 57.46 | 51.13 | 49.68 | 51.62 | 50.00 | 51.62 | 51.62 | 56.89 |
| rotate | 60.98 | 55.27 | 52.47 | 52.31 | 49.31 | 52.51 | 52.59 | 59.56 |
| scale | 55.83 | 50.61 | 47.73 | 47.65 | 46.15 | 48.34 | 48.42 | 53.04 |
| jitter | 50.49 | 43.56 | 44.61 | 46.31 | 45.30 | 46.72 | 47.12 | 49.72 |

S2 聚合分支准确率：

| Branch | Correct / Total | Acc |
|---|---:|---:|
| zero-shot text prototype dot | 8215 / 17276 | 47.55 |
| global entropy cache | 8207 / 17276 | 47.51 |
| GPA global cache diag | 8469 / 17276 | 49.02 |
| GPA local cache | 8113 / 17276 | 46.96 |
| positive cache total | 8486 / 17276 | 49.12 |
| cache total signed | 8546 / 17276 | 49.47 |
| final logits | 9449 / 17276 | 54.69 |
| norm fusion offline | 9245 / 17276 | 53.51 |

关键观察：

1. cache-only 分支不够强。`cache_total_signed=49.47`，明显低于 `final_logits=54.69`。
2. `02_9_2` 的收益不是某个缓存分支单独变强，而是 zero-shot prior 与缓存投票互补。
3. `norm_fusion_offline=53.51` 低于正式 final，说明简单做归一化融合不是直接解法。
4. `GPA global cache diag=49.02` 高于 `GPA local cache=46.96`，而正式 `02_9_2` final 只使用 GPA local，不使用 GPA global logits。这是后续改 final score 的重要线索。
5. GPA local 分支作为独立分类器偏弱，说明当前 GPA evidence 的使用方式可能不充分。

## 11. Zero-shot 与 Cache 的互补关系

S2 聚合统计：

| 指标 | 数量 |
|---|---:|
| zero-shot correct total | 8215 |
| zero-shot correct but final wrong | 605 |
| zero-shot correct but cache wrong | 1846 |
| zero-shot wrong total | 9061 |
| zero-shot wrong but final correct | 1839 |
| zero-shot wrong but cache correct | 2177 |
| zero-shot/cache agree total | 9231 |
| zero-shot/cache disagree total | 8045 |
| zero-shot/cache agree and final correct | 6369 |
| zero-shot/cache disagree and final correct | 3080 |

解释：

1. final 修正了 `1839 / 9061 = 20.3%` 的 zero-shot 错误样本。
2. final 也破坏了 `605 / 8215 = 7.4%` 的 zero-shot 正确样本。
3. cache 单独不稳定：zero-shot 正确时，cache 错了 1846 个；zero-shot 错误时，cache 对了 2177 个。
4. 因此不能简单增强 cache 权重。更合理的方向是做更精细的写入门控或残差式校正。

## 12. GPA Replacement Event 诊断

S2 聚合的伪标签正确率：

| Phase | Decision | Count | Pseudo-label correct | Correct rate |
|---|---|---:|---:|---:|
| build | add_not_full | 769 | 358 | 46.55 |
| build | replace | 500 | 346 | 69.20 |
| build | reject_entropy | 9604 | 3801 | 39.58 |
| build | reject_joint | 6403 | 3731 | 58.27 |
| test | add_not_full | 8 | 3 | 37.50 |
| test | replace | 385 | 273 | 70.91 |
| test | reject_entropy | 13893 | 5947 | 42.81 |
| test | reject_joint | 2990 | 1992 | 66.62 |

关键观察：

1. `replace` 是相对精准的：test 阶段 accepted replacement 的伪标签正确率为 `70.91%`。
2. 但 `reject_joint` 拒绝了大量正确伪标签：test 阶段 `2990` 个 joint-rejected 样本中，`1992` 个伪标签是正确的，正确率 `66.62%`。
3. 这说明当前 joint gate 不是单纯在拦截错误样本，它也明显误拒绝了很多本可进入 GPA/local cache 的高质量样本。
4. `reject_entropy` 更像粗筛：test 阶段 `42.81%` 的 reject entropy 样本伪标签正确，说明 entropy gate 也会错杀，但其错误率仍明显高于 joint reject，优先级低于修 joint gate。

test 阶段 `replace` 与 `reject_joint` 的逐 corruption 正确率：

| Corruption | replace correct rate | reject_joint correct rate |
|---|---:|---:|
| add_global | 72.97 | 72.90 |
| add_local | 55.36 | 58.85 |
| dropout_global | 82.76 | 76.43 |
| dropout_local | 70.15 | 61.31 |
| rotate | 68.33 | 73.61 |
| scale | 74.19 | 60.38 |
| jitter | 73.33 | 64.74 |

解释：

1. 在 `add_global`、`dropout_global`、`rotate` 中，`reject_joint` 的正确率非常高，说明严格 joint gate 明显过保守。
2. `add_local` 的 accepted replacement 正确率最低，说明局部异常点场景仍需要保护，不能完全放开 joint gate。
3. 这支持“低熵可信回退 / 软化 joint gate”，但不支持“无条件放宽所有 replacement”。

## 13. Joint Reject 样本的数值特征

test 阶段 `reject_joint` 的聚合均值：

| Metric | All reject_joint | Correct pseudo-label | Wrong pseudo-label |
|---|---:|---:|---:|
| new_entropy | 0.2654 | 0.1743 | 0.4473 |
| new_visual_score | -1.0256 | -0.9685 | -1.1398 |
| new_text_score | -2.4561 | -2.4308 | -2.5065 |
| new_joint_score | -0.1879 | -0.0761 | -0.4110 |
| joint_score_margin | -0.9158 | -0.8329 | -1.0812 |

解释：

1. joint-rejected 但伪标签正确的样本，平均 entropy 明显更低：`0.1743` vs `0.4473`。
2. 正确样本的 visual score 也更好：`-0.9685` vs `-1.1398`。
3. text score 方向基本一致，但区分度不如 entropy 与 visual score 稳定。
4. 这说明被 joint gate 拒绝的正确样本往往不是“低质量样本”，而是“相对被替换样本 joint score 不够高”的样本。

## 14. 结论与下一步

本实验给出的核心结论：

1. `02_16_1` 成功复现 `02_9_2` 的 S2 表现，因此诊断结果可信。
2. `02_9_2` 的最终收益来自 zero-shot prior 与 cache evidence 的互补，不来自单个 cache 分支。
3. 当前 GPA replacement 的 `replace` 决策较精准，但 joint gate 明显过保守。
4. 大量 joint-rejected 样本伪标签正确，并且具有低熵、更好的视觉分布分数特征。
5. 继续调 E1 prompt、text weight 或简单融合权重，不能解决这个根问题。

后续优先方向：

```text
在 02_9_2 的 GPA replacement 中加入低熵、视觉合理的 trusted fallback。
```

该方向应满足：

1. 保留 entropy gate，不完全放开 replacement。
2. 当 `entropy_new < entropy_old` 但 `joint_score_new < joint_score_old` 时，不直接拒绝。
3. 若新样本满足低熵且视觉分布分数不差，可允许其进入 GPA/local cache。
4. 先在 S2 诊断，再考虑 all35。

同时应保留第二条候选方向：

```text
把 GPA global cache diag 作为校准后的 residual evidence 引入 final logits。
```

原因是 `GPA global cache diag=49.02` 明显强于 `GPA local cache=46.96`，而当前正式 final 没有使用这一分支。
