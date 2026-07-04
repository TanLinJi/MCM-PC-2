# 02_15_2：E1-33 Fused Prototype Text Gate 权重 0.10

日期：2026-06-18

状态：已运行，S2 结果已记录。

## 1. 实验目的

`02_15_1` 将 E1-33 fused prototype 接入 GPA cache replacement gate，修复了 `02_14_1` 的一部分负迁移，但 S2 平均结果仍低于 `02_9_2`。

关键观察：

```text
02_15_1 S2 average = 54.43
02_9_2  S2 average = 54.71
02_14_1 S2 average = 54.32
E0      S2 average = 54.00
```

逐扰动上，`02_15_1` 对 `add_global`、`dropout_global` 有收益，但对 `add_local`、`jitter` 明显下降。因此本实验只降低 fused prototype text gate 强度，验证 `E4_TEXT_SCORE_WEIGHT=0.10` 是否能减少局部扰动上的过强文本门控。

## 2. 基本设置

```text
实验编号：02_15_2
数据集：ModelNet-C
扰动等级：severity=2
扰动类型：add_global, add_local, dropout_global, dropout_local, rotate, scale, jitter
backbone：ULIP
载体：02_9_2
最终分类器：manual_full
最终 logits：Point-Cache voting，不变
cache：global entropy cache + GPA local cache + negative cache，保持 02_9_2
score normalization：running_zscore
```

## 3. E1-33 文本设置

```text
文本门控模式：E4_TEXT_GATE_MODE=fused_prototype
E4_TEXT_SCORE_WEIGHT=0.10
E4_TEXT_PROTO_SCORE_SCALE=1.0
prompt cache：llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json
dynamic prompt count：15
prompt composition：10 image-style + 5 pointcloud-style
manual_full:LLM = 0.60:0.40
```

本实验和 `02_15_1` 的唯一实验变量：

```text
02_15_1：E4_TEXT_SCORE_WEIGHT=0.15
02_15_2：E4_TEXT_SCORE_WEIGHT=0.10
```

## 4. 脚本

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E4_distribution_guided_cache/02_15_2_ulip_modelnetc_s2_e1_33_fused_prototype_text_gate_tw0p10_manual60_llm40.sh
```

## 5. 执行命令

在 `mcmpc` 环境中执行：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/02_15_2_ulip_modelnetc_s2_e1_33_fused_prototype_text_gate_tw0p10_manual60_llm40.sh 0
```

其中 `0` 表示使用当前单张 4090 环境中的物理 GPU 0。

运行前可做 dry-run 检查：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/02_15_2_ulip_modelnetc_s2_e1_33_fused_prototype_text_gate_tw0p10_manual60_llm40.sh 0 --dry-run
```

## 6. 结果保存位置

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/02_15_2_ulip_modelnetc_s2_e4_c_a0_e1_33_fused_prototype_text_gate_tw0p10_manual60_llm40/
```

核心文件：

```text
summary.csv
logs/
gpa_stats/
wandb/
```

## 7. 结果记录

运行命令：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/02_15_2_ulip_modelnetc_s2_e1_33_fused_prototype_text_gate_tw0p10_manual60_llm40.sh 0
```

结果目录：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/02_15_2_ulip_modelnetc_s2_e4_c_a0_e1_33_fused_prototype_text_gate_tw0p10_manual60_llm40/
```

逐扰动结果：

| corruption | E0 | 02_9_2 | 02_14_1 | 02_15_1 tw0.15 | 02_15_2 tw0.10 | 02_15_2 - 02_15_1 | 02_15_2 - 02_9_2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| add_global | 47.81 | 47.89 | 47.93 | 48.82 | 48.18 | -0.64 | +0.29 |
| add_local | 46.68 | 50.85 | 50.16 | 49.15 | 49.96 | +0.81 | -0.89 |
| dropout_global | 59.20 | 59.12 | 58.55 | 59.48 | 58.95 | -0.53 | -0.17 |
| dropout_local | 56.69 | 57.21 | 56.44 | 57.01 | 56.12 | -0.89 | -1.09 |
| rotate | 62.07 | 61.30 | 60.90 | 61.02 | 61.67 | +0.65 | +0.37 |
| scale | 55.23 | 55.92 | 56.16 | 55.55 | 55.51 | -0.04 | -0.41 |
| jitter | 50.32 | 50.65 | 50.12 | 49.96 | 50.93 | +0.97 | +0.28 |

S2 平均：

```text
E0      = 54.00
02_9_2  = 54.71
02_14_1 = 54.32
02_15_1 = 54.43
02_15_2 = 54.47
```

平均差值：

```text
02_15_2 - E0      = +0.47
02_15_2 - 02_14_1 = +0.15
02_15_2 - 02_15_1 = +0.05
02_15_2 - 02_9_2  = -0.23
```

GPA 诊断统计：

```text
02_15_2 total test replacement = 388
02_15_1 total test replacement = 376
02_9_2  total test replacement = 381

02_15_2 total joint-score reject = 2926
02_15_1 total joint-score reject = 2638
02_9_2  total joint-score reject = 3051

02_15_2 total entropy reject = 13954
02_15_1 total entropy reject = 14254
02_9_2  total entropy reject = 13836
```

结论：

```text
0.10 相比 0.15 略优，主要恢复 add_local、rotate、jitter。
但 0.10 同时牺牲 add_global、dropout_global、dropout_local。
整体仍低于原始最好 02_9_2，因此不能作为最终配置。
```

## 8. 判定标准

优先比较对象：

```text
02_15_1：fused prototype text gate, text_weight=0.15
02_9_2：原始最好 E4 S2 配置
02_14_1：E1-33 prompt-level distribution 替换实验
E0：baseline
```

`02_15_2` 相比 `02_15_1` 确实恢复了 `add_local` 和 `jitter`，说明 `0.15` 的 fused prototype gate 对局部扰动偏强。

但 `02_15_2` 仍低于 `02_9_2`，且 `dropout_local` 明显下降。因此下一步不建议直接跑完整 all35。更合理的下一步是在 S2 上测试 `0.125`，观察是否能在 `0.10` 的局部恢复和 `0.15` 的 global/dropout 收益之间取得更好的平衡。
