# 02_14_1：02_9_2 载体替换 E1-33 Prompt 后的 ModelNet-C All35 验证

日期：2026-06-18

## 1. 实验目的

本实验基于当前最好的 `02_9_2` E4 载体，只替换其中用于 text distribution 的 E1 LLM prompt 配置，验证固定后的 E1-33 prompt 设置在完整 ModelNet-C all35 上是否能继续提升或保持 `02_9_2` 的效果。

注意：本实验不是重新设计 E4，也不是把 E1 文本原型作为最终分类器。最终 classifier/logits 仍然使用 `manual_full`，E1 LLM 描述只用于构建 cache replacement 阶段的 text distribution。

## 2. 基本实验设置

| 项目 | 设置 |
|---|---|
| 实验编号 | `02_14_1` |
| 数据集 | ModelNet-C full |
| 评测范围 | 35 个设置，7 corruption x 5 severity |
| 骨干模型 | ULIP |
| 载体实验 | `02_9_2` |
| 方法 | E4-C-A0+E1-textdist-only |
| 最终分类器/logits | `manual_full`，不变 |
| E1 描述用途 | 只用于 text distribution，不直接作为最终分类器 |
| `E4_TEXT_SCORE_WEIGHT` | `0.15` |
| `E4_SCORE_NORM_MODE` | `running_zscore` |
| 硬件设置 | 单张 RTX 4090 |
| all35 调度方式 | 单 worker 顺序运行 35 个任务 |

## 3. 与原始 02_9_2 的唯一核心差异

原始 `02_9_2` all35 使用旧的 10 条 LLM prompt 配置：

```text
prompt_cache_dir = results/E1_text_prototype_enhancement/shared_prompts
prompt_cache_file = modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json
dynamic_prompt_count = 10
llm_prompt_mode = multiview_2d3d
prompt_static_weight = 0.75
prompt_dynamic_weight = 0.25
```

本实验替换为当前固定的 E1-33 配置：

```text
prompt_cache_dir = llm
prompt_cache_file = modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json
dynamic_prompt_count = 15
llm_prompt_mode = image10_pointcloud5
prompt_static_weight = 0.60
prompt_dynamic_weight = 0.40
```

除此之外，`02_9_2` 的 E4 载体设置保持不变：

```text
E4-C-A0+E1-textdist-only
manual_full final classifier/logits
E4_TEXT_SCORE_WEIGHT = 0.15
E4_SCORE_NORM_MODE = running_zscore
```

## 4. 相关文件

Runner：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/runners/E4_distribution_guided_cache/modelnetc_all35_02_9_2/launch_02_9_2_modelnetc_all35_e1_33_prompt_single4090.py
```

Worker：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/runners/E4_distribution_guided_cache/modelnetc_all35_02_9_2/worker_02_9_2_modelnetc_all35.py
```

启动脚本：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache/02_14_1_ulip_modelnetc_all35_single4090_e4_c_a0_e1_textdist_only_tw0p15_score_norm_e1_33_prompt_manual60_llm40.sh
```

Prompt JSON：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json
```

## 5. 执行命令

在当前 `mcmpc` 环境中执行：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/02_14_1_ulip_modelnetc_all35_single4090_e4_c_a0_e1_textdist_only_tw0p15_score_norm_e1_33_prompt_manual60_llm40.sh 0
```

说明：

```text
0 = 使用物理 GPU 0
当前脚本按单张 RTX 4090 设计，只启动一个 worker
35 个 ModelNet-C 任务会在同一张卡上顺序执行
```

## 6. 结果保存位置

结果目录：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/02_14_1_all35_ulip_modelnetc_zs_global_local_e4_c_a0_e1_textdist_only_tw0p15_score_norm_e1_33_prompt_single4090/
```

预期生成文件：

```text
summary.csv
all35_table.csv
all35_table.md
all35_table.html
tasks_worker0.json
worker0_stdout.log
logs/worker0/
wandb_worker0/
```

其中 `all35_table.md` 是最方便直接对比的表格，`summary.csv` 记录每个 corruption/severity 的单项结果。

## 7. 对比对象

主要对比：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/02_9_2_all35_ulip_modelnetc_zs_global_local_e4_c_a0_e1_textdist_only_tw0p15_score_norm_dual_t4/
```

原始 `02_9_2` all35 平均结果：

| 指标 | 02_9_2 |
|---|---:|
| S0 平均 | 59.93 |
| S1 平均 | 56.70 |
| S2 平均 | 54.71 |
| S3 平均 | 50.26 |
| S4 平均 | 44.51 |
| all35 平均 | 53.22 |

辅助对比：

```text
E0 baseline all35
E1_36 manual60_llm40 zero-shot text prototype all35
```

但本实验最关键的问题是：在完全保留 `02_9_2` 载体时，仅将 text distribution prompt 从旧 10 条/0.75:0.25 替换为 E1-33 的 15 条/0.60:0.40，all35 是否优于原始 `02_9_2`。

## 8. 完整性检查

代码与脚本已完成静态检查：

```text
bash -n 通过
python -m py_compile 通过
dry-run 通过
```

dry-run 已确认：

```text
Prompt cache: 40 classes, 15 prompts per class
Task count: 35
Execution model: one worker process on one GPU; all 35 tasks run sequentially
```

完整 all35 实验已手动执行完成。

| 检查项 | 当前值 | 期望值 | 状态 |
|---|---:|---:|---|
| `summary.csv` 数据行 | 35 | 35 | 正常 |
| 唯一 corruption/severity 设置 | 35 | 35 | 正常 |
| `status=done` 行数 | 35 | 35 | 正常 |
| `all35_table.md` | 已生成 | 已生成 | 正常 |

`gpa_stats/` 下当前有 70 个统计文件，说明每个设置生成了多份 GPA 统计输出；这不影响 `summary.csv` 与 all35 汇总表的完整性判断。

## 9. 02_14_1 All35 结果

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 55.63 | 51.01 | 47.93 | 45.58 | 37.76 | 47.58 |
| add_local | 58.06 | 53.40 | 50.16 | 46.60 | 44.77 | 50.60 |
| dropout_global | 61.35 | 58.06 | 58.55 | 58.71 | 53.12 | 57.96 |
| dropout_local | 63.61 | 58.75 | 56.44 | 51.78 | 43.96 | 54.91 |
| rotate | 62.36 | 61.99 | 60.90 | 57.62 | 48.54 | 58.28 |
| scale | 58.43 | 58.79 | 56.16 | 53.81 | 54.34 | 56.31 |
| jitter | 59.52 | 54.82 | 50.12 | 39.55 | 29.13 | 46.63 |
| **Average** | **59.85** | **56.69** | **54.32** | **50.52** | **44.52** | **53.18** |

## 10. 与原始 02_9_2 的对比

核心结论：

```text
02_14_1 all35 = 53.18
02_9_2  all35 = 53.22
变化 = -0.04
```

这说明：在完全保留 `02_9_2` 载体时，将 text distribution prompt 从旧 10 条/0.75:0.25 替换为 E1-33 的 15 条/0.60:0.40，没有带来 all35 提升，整体基本持平但略低。

逐 severity 对比：

| 指标 | 02_9_2 | 02_14_1 | 变化 |
|---|---:|---:|---:|
| S0 平均 | 59.93 | 59.85 | -0.08 |
| S1 平均 | 56.70 | 56.69 | -0.01 |
| S2 平均 | 54.71 | 54.32 | -0.39 |
| S3 平均 | 50.26 | 50.52 | +0.26 |
| S4 平均 | 44.51 | 44.52 | +0.01 |
| all35 平均 | 53.22 | 53.18 | -0.04 |

逐 corruption 平均对比：

| Corruption | 02_9_2 | 02_14_1 | 变化 |
|---|---:|---:|---:|
| add_global | 47.49 | 47.58 | +0.09 |
| add_local | 50.84 | 50.60 | -0.24 |
| dropout_global | 58.27 | 57.96 | -0.31 |
| dropout_local | 55.33 | 54.91 | -0.42 |
| rotate | 58.00 | 58.28 | +0.28 |
| scale | 56.19 | 56.31 | +0.12 |
| jitter | 46.43 | 46.63 | +0.20 |

逐项差值如下，数值为 `02_14_1 - 02_9_2`：

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| add_global | -0.12 | +0.44 | +0.04 | +0.24 | -0.12 | +0.09 |
| add_local | -0.41 | -0.53 | -0.69 | -0.16 | +0.56 | -0.24 |
| dropout_global | -0.32 | -0.57 | -0.57 | +0.28 | -0.36 | -0.31 |
| dropout_local | -0.57 | +0.36 | -0.77 | -0.45 | -0.69 | -0.42 |
| rotate | +0.04 | +0.52 | -0.40 | +0.69 | +0.57 | +0.28 |
| scale | +0.41 | -0.16 | +0.24 | +0.20 | -0.12 | +0.12 |
| jitter | +0.40 | -0.16 | -0.53 | +1.02 | +0.24 | +0.20 |
| **Average** | **-0.08** | **-0.01** | **-0.39** | **+0.26** | **+0.01** | **-0.04** |

最明显正向单点：

| Setting | 变化 |
|---|---:|
| jitter_S3 | +1.02 |
| rotate_S3 | +0.69 |
| rotate_S4 | +0.57 |
| add_local_S4 | +0.56 |
| rotate_S1 | +0.52 |

最明显负向单点：

| Setting | 变化 |
|---|---:|
| dropout_local_S2 | -0.77 |
| add_local_S2 | -0.69 |
| dropout_local_S4 | -0.69 |
| dropout_global_S1 | -0.57 |
| dropout_global_S2 | -0.57 |

## 11. 与 E0 Baseline 的对比

虽然 `02_14_1` 没有超过原始 `02_9_2`，但仍然高于 E0 Point-Cache baseline。

| 指标 | E0 baseline | 02_14_1 | 变化 |
|---|---:|---:|---:|
| S0 平均 | 59.04 | 59.85 | +0.81 |
| S1 平均 | 57.10 | 56.69 | -0.41 |
| S2 平均 | 54.00 | 54.32 | +0.32 |
| S3 平均 | 50.44 | 50.52 | +0.08 |
| S4 平均 | 44.49 | 44.52 | +0.03 |
| all35 平均 | 53.01 | 53.18 | +0.17 |

逐 corruption 平均对比：

| Corruption | E0 baseline | 02_14_1 | 变化 |
|---|---:|---:|---:|
| add_global | 48.43 | 47.58 | -0.85 |
| add_local | 49.16 | 50.60 | +1.44 |
| dropout_global | 57.83 | 57.96 | +0.13 |
| dropout_local | 55.08 | 54.91 | -0.17 |
| rotate | 59.08 | 58.28 | -0.80 |
| scale | 56.30 | 56.31 | +0.01 |
| jitter | 45.22 | 46.63 | +1.41 |

主要收益仍来自 `add_local` 和 `jitter`；主要负向仍集中在 `add_global` 和 `rotate`。这与原始 `02_9_2` 的收益结构一致。

## 12. 与 E1_36 纯文本原型的关系

`E1_36` 是同一套 E1-33 prompt 配置在 zero-shot text prototype 层面的 all35 验证，all35 平均为 49.36。`02_14_1` 为 53.18，比 `E1_36` 高 +3.82。

这个差距说明：

1. E1-33 prompt 作为最终 zero-shot classifier 并不强；
2. 但放进 `02_9_2` 的 text distribution 后，cache 机制仍能维持接近原 `02_9_2` 的性能；
3. E1 prompt 对 E4 的作用更像 cache replacement 的文本分布约束，而不是最终分类器增强。

## 13. 结论

本实验的算法变量控制是清楚的：相对原始 `02_9_2`，唯一核心方法差异是 text distribution prompt 配置从旧 10 条/0.75:0.25 替换为 E1-33 的 15 条/0.60:0.40。执行环境从原来的双 worker/双卡记录改为单张 RTX 4090、单 worker 顺序运行；每个 corruption/severity 任务仍是独立评测。

结论：

1. `02_14_1` 不应替代原始 `02_9_2` 作为当前 ModelNet-C all35 最优配置，因为 all35 低了 0.04。
2. E1-33 prompt 配置没有在 E4 text distribution 中带来稳定增益，尤其 S2 平均下降 -0.39。
3. 该配置改善了 `rotate`、`scale`、`jitter`，但损失了 `dropout_global`、`dropout_local`、`add_local` 的中等强度表现。
4. 后续若继续沿用 `02_9_2` 载体，优先保留原始旧 prompt text distribution 配置；E1-33 prompt 可以作为候选分支记录，而不是主线替换。
