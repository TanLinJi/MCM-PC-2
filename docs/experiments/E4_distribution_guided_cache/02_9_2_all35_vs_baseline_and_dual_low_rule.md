# 02_9_2 All35 结果分析与低熵低能量回退规则

日期：2026-06-11

## 1. 本文目的

本文记录当前 DPC-Point 主线在 ModelNet-C 全部 35 个损坏设置上的结果，并与原始 Point-Cache baseline 对比。

当前实验配置：

| 项目 | 内容 |
|---|---|
| 方法 | E4-C-A0+E1-textdist-only |
| 文本分布权重（text distribution weight） | `E4_TEXT_SCORE_WEIGHT=0.15` |
| 分数归一化（score normalization） | `running_zscore` |
| 骨干模型（backbone） | ULIP |
| 数据集（dataset） | ModelNet-C |
| 损坏类型（corruption） | 7 类 |
| 损坏强度（severity） | S0-S4 |
| 总设置数 | 35 |

对比 baseline 文档：

```text
/root/autodl-tmp/MCM-PC-2/docs/experiments/E0_baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local.md
```

当前结果目录：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E4_distribution_guided_cache/02_9_2_all35_ulip_modelnetc_zs_global_local_e4_c_a0_e1_textdist_only_tw0p15_score_norm_dual_t4/
```

完整性检查：

| 检查项 | 当前值 | 期望值 | 状态 |
|---|---:|---:|---|
| `summary.csv` 数据行 | 35 | 35 | 正常 |
| 唯一设置数 | 35 | 35 | 正常 |
| `status=done` 行数 | 35 | 35 | 正常 |
| `gpa_stats` 文件数 | 35 | 35 | 正常 |

结论：这一轮 all35 结果完整，可以作为当前 DPC-Point 主线的有效记录。

## 2. 当前 02_9_2 All35 结果

| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 55.75 | 50.57 | 47.89 | 45.34 | 37.88 | 47.49 |
| add_local | 58.47 | 53.93 | 50.85 | 46.76 | 44.21 | 50.84 |
| dropout_global | 61.67 | 58.63 | 59.12 | 58.43 | 53.48 | 58.27 |
| dropout_local | 64.18 | 58.39 | 57.21 | 52.23 | 44.65 | 55.33 |
| rotate | 62.32 | 61.47 | 61.30 | 56.93 | 47.97 | 58.00 |
| scale | 58.02 | 58.95 | 55.92 | 53.61 | 54.46 | 56.19 |
| jitter | 59.12 | 54.98 | 50.65 | 38.53 | 28.89 | 46.43 |
| **Average** | **59.93** | **56.70** | **54.71** | **50.26** | **44.51** | **53.22** |

## 3. 与原始 Point-Cache 的整体对比

| 指标 | 原始 Point-Cache | 当前 DPC-Point | 变化 |
|---|---:|---:|---:|
| S0 平均 | 59.04 | 59.93 | +0.90 |
| S1 平均 | 57.10 | 56.70 | -0.40 |
| S2 平均 | 54.00 | 54.71 | +0.71 |
| S3 平均 | 50.44 | 50.26 | -0.18 |
| S4 平均 | 44.49 | 44.51 | +0.02 |
| all35 平均 | 53.01 | 53.22 | +0.21 |

结论：

1. 当前 DPC-Point 相比原始 Point-Cache 在 all35 上小幅提升 `+0.21`。
2. 在 severity=2 上提升 `+0.71`，这个指标更接近原论文常用对比设置。
3. 当前结果证明分布引导原型缓存（distribution-guided prototype cache）有有效信号，但优势还不够稳定。
4. 当前收益有明显选择性：对局部扰动和噪声更有效，对 `rotate` 和 `add_global` 有负作用。

## 4. 按损坏类型分析

| Corruption | Baseline Avg | DPC-Point Avg | 变化 |
|---|---:|---:|---:|
| add_global | 48.43 | 47.49 | -0.94 |
| add_local | 49.16 | 50.84 | +1.69 |
| dropout_global | 57.83 | 58.27 | +0.44 |
| dropout_local | 55.08 | 55.33 | +0.25 |
| rotate | 59.08 | 58.00 | -1.08 |
| scale | 56.30 | 56.19 | -0.11 |
| jitter | 45.22 | 46.43 | +1.22 |

正向结果主要来自：

- `add_local`
- `jitter`
- `dropout_global`
- `dropout_local`

负向结果主要来自：

- `rotate`
- `add_global`

解释：当前联合分布筛选规则（joint distribution filtering）能挡住一部分不可靠样本，但在某些场景中也会挡掉原始 Point-Cache 会接受的高质量样本。

## 5. 逐项差值

表中数值为：

```text
当前 DPC-Point 准确率 - 原始 Point-Cache 准确率
```

| Corruption | S0 | S1 | S2 | S3 | S4 | 平均变化 |
|---|---:|---:|---:|---:|---:|---:|
| add_global | +0.36 | -2.19 | +0.08 | -0.85 | -2.11 | -0.94 |
| add_local | +2.07 | +2.03 | +4.17 | +0.20 | -0.04 | +1.69 |
| dropout_global | +0.00 | -0.32 | -0.08 | +0.57 | +2.02 | +0.44 |
| dropout_local | +2.02 | -0.08 | +0.52 | -0.97 | -0.24 | +0.25 |
| rotate | -0.56 | -1.90 | -0.77 | -0.61 | -1.58 | -1.08 |
| scale | -0.33 | +0.40 | +0.69 | -0.56 | -0.73 | -0.11 |
| jitter | +2.72 | -0.73 | +0.33 | +0.97 | +2.80 | +1.22 |
| **Average** | **+0.90** | **-0.40** | **+0.71** | **-0.18** | **+0.02** | **+0.21** |

最明显的正向单点：

| Setting | Baseline | DPC-Point | 变化 |
|---|---:|---:|---:|
| add_local_S2 | 46.68 | 50.85 | +4.17 |
| jitter_S4 | 26.09 | 28.89 | +2.80 |
| jitter_S0 | 56.40 | 59.12 | +2.72 |
| add_local_S0 | 56.40 | 58.47 | +2.07 |
| add_local_S1 | 51.90 | 53.93 | +2.03 |
| dropout_local_S0 | 62.16 | 64.18 | +2.02 |
| dropout_global_S4 | 51.46 | 53.48 | +2.02 |

最明显的负向单点：

| Setting | Baseline | DPC-Point | 变化 |
|---|---:|---:|---:|
| add_global_S1 | 52.76 | 50.57 | -2.19 |
| add_global_S4 | 39.99 | 37.88 | -2.11 |
| rotate_S1 | 63.37 | 61.47 | -1.90 |
| rotate_S4 | 49.55 | 47.97 | -1.58 |
| dropout_local_S3 | 53.20 | 52.23 | -0.97 |

## 6. 术语说明

| 代码名 | 中文解释 |
|---|---|
| `pc_feats` | 点云全局特征（point-cloud global features） |
| `patch_centers` | 局部 patch 聚类中心特征（local patch cluster-center features） |
| `clip_logits` | CLIP/ULIP 分类 logits（classification logits） |
| `entropy` | 熵 / 预测不确定性（entropy / prediction uncertainty） |
| `scaled_entropy` | Point-Cache 风格缩放熵（Point-Cache scaled entropy） |
| `energy` | 能量 / 样本似然代理量（energy / likelihood proxy） |
| `GPA cache` | 全局原型对齐缓存（Global Prototype-Alignment cache） |
| `local cache` | 局部缓存（local cache） |
| `joint_score` | 联合分布分数（joint distribution score） |

注意：

- 熵（entropy）理论上不应为负，数值越小表示模型越确定。
- 能量（energy）可以为负。本文使用 `energy = -logsumexp(clip_logits)`，数值越小、越负，表示能量越低。
- 当前代码中的 `scaled_entropy` 不是严格的 0 到 1 归一化熵，而是沿用 Point-Cache 的缩放方式。

## 7. 当前问题

当前 DPC-Point 的缓存写入规则可以概括为：

```text
先要求当前样本比缓存中最差样本熵更低；
再要求当前样本的联合分布分数更高。
```

这个规则对局部噪声、局部缺失等场景有帮助，但也可能过于严格。

clean 结果也支持这个判断：

| Setting | Accuracy |
|---|---:|
| 原始 Point-Cache clean | 64.18 |
| 当前 02_9_2 clean | 63.86 |
| 变化 | -0.32 |

因此下一步不应继续简单增大文本权重（text weight），也不应直接改最终 logits 融合。更合适的是：只在 GPA/local cache 写入阶段增加可信样本回退。

## 8. 方案A：低熵低能量可信样本回退

本节是已确认的实现规则。

规则名称：

```text
低熵 + 低能量可信样本回退
dual-low trusted fallback
```

核心思想：

```text
如果当前测试样本同时低熵、低能量，
并且它比当前类别 GPA cache 中最差样本熵更低，
则认为它是可信样本。

可信样本不再受联合分布分数门控限制，
而是按原始 Point-Cache 的低熵替换规则写入 GPA/local cache。
```

该规则仍符合测试时适应（Test-Time Adaptation, TTA）设定，因为它只使用当前样本输出、在线能量统计和当前缓存状态，不使用 clean/corruption 标签。

## 9. 具体更新流程

对当前样本计算：

```text
点云全局特征（pc_feats）
局部 patch 聚类中心特征（patch_centers）
分类 logits（clip_logits）
熵（entropy）
能量（energy）
```

能量计算为：

```text
energy = -logsumexp(clip_logits)
```

如果该类别的 GPA cache 未满：

```text
直接加入 GPA cache 和 local cache。
```

如果该类别的 GPA cache 已满：

第一步，找到该类别缓存中熵最高的样本：

```text
worst_sample = 该类别 GPA cache 中 entropy 最大的样本
```

第二步，保留原始低熵门控：

```text
如果当前样本 entropy >= worst_sample entropy：
    拒绝当前样本
```

第三步，判断当前样本是否可信：

```text
低熵条件：scaled_entropy <= tau_entropy
低能量条件：energy_z <= tau_energy_z
可信条件：低熵条件 and 低能量条件
```

第四步，执行替换：

```text
如果当前样本可信：
    直接替换 worst_sample
    不检查 joint_score
否则：
    继续使用当前 DPC 规则
    只有 joint_score(current) > joint_score(worst) 才替换
```

## 10. 低熵条件

第一版使用当前 Point-Cache 代码里的缩放熵（scaled entropy）：

```text
scaled_entropy = entropy / log2(num_classes)
```

默认阈值：

```text
tau_entropy = 0.10
```

建议小规模消融：

```text
tau_entropy in {0.08, 0.10, 0.12}
```

## 11. 低能量条件

能量（energy）定义为：

```text
energy = -logsumexp(clip_logits)
```

实现时使用 `clip_logits.float()` 计算，避免半精度带来的数值误差。

在线能量统计（online energy statistics）维护：

```text
running_mean_energy
running_std_energy
running_count_energy
```

当前样本的能量 z 分数：

```text
energy_z = (energy - running_mean_energy) / running_std_energy
```

默认低能量条件：

```text
energy_z <= -0.5
```

含义：当前样本的能量至少比历史平均能量低 0.5 个标准差。

## 12. 能量统计 warmup

为了避免刚开始统计不稳定，加入 warmup：

```text
energy_stats_min_count = 64
```

在已经看到的样本数少于 64 时：

```text
不启用可信样本回退；
继续使用当前 DPC 规则。
```

建议后续比较：

```text
energy_stats_min_count in {32, 64}
```

## 13. 缓存内容是否保存 energy

第一版不强制保存缓存中样本的能量（energy）。

原因：

1. 当前方法是训练免参数更新的测试时适应（training-free TTA），模型参数不会变化。
2. 缓存排序和替换目标仍然由熵（entropy）决定。
3. 方案A只需要判断当前样本是否低能量，不需要比较缓存中旧样本的能量。

因此实现上：

```text
缓存仍主要保存特征和熵。
当前样本实时计算 energy。
在线统计记录历史 energy 分布。
事件日志可以记录当前样本 energy，便于后续分析。
```

## 14. 推荐初始配置与实现入口

旧的 E4-C 原文件不加入新规则，保证 `02_9_2` 已完成实验可复现。

A1 新文件和 `05_*` 新脚本默认启用方案A：

```text
模型文件：
Point-Cache/runners/E4_distribution_guided_cache/model_e4_c_a1_dual_low_trusted_fallback_accepted_history_text_visual_distribution_guided_gpa.py

severity=2 runner：
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a1_dual_low_trusted_fallback_ulip_modelnetc_s2_accepted_history_text_visual_distribution_guided_gpa.py

clean runner：
Point-Cache/runners/E4_distribution_guided_cache/run_e4_c_a1_dual_low_trusted_fallback_ulip_modelnetc_clean.py
```

默认环境变量：

```text
E4_TRUSTED_FALLBACK=dual_low_entropy_energy
E4_TRUSTED_ENTROPY_THRESHOLD=0.10
E4_TRUSTED_ENERGY_Z_THRESHOLD=-0.5
E4_TRUSTED_ENERGY_MIN_COUNT=64
```

启动脚本：

```text
ModelNet-C severity=2：
Point-Cache/scripts/E4_distribution_guided_cache/05_1_ulip_modelnetc_s2_zs_global_local_e4_c_a1_dual_low_trusted_fallback_tw0p15_ent0p10_energyzm0p5_min64.sh

ModelNet-C clean：
Point-Cache/scripts/E4_distribution_guided_cache/05_2_clean_ulip_modelnetc_clean_zs_global_local_e4_c_a1_dual_low_trusted_fallback_tw0p15_ent0p10_energyzm0p5_min64.sh
```

第一轮验证：

1. clean
2. ModelNet-C severity=2

成功标准：

| 指标 | 目标 |
|---|---|
| clean | 接近或超过原始 Point-Cache clean `64.18` |
| ModelNet-C S2 平均 | 不低于当前 `54.71` |
| add_global / rotate | 缓解当前负增益 |
| add_local / jitter | 尽量保留当前正增益 |

## 15. 研究判断

当前 `02_9_2` 说明分布引导原型缓存（distribution-guided prototype cache）是有效的，但硬性联合分布筛选会带来副作用。

低熵低能量可信样本回退的作用是：

```text
对不确定样本继续使用 DPC 的分布净化；
对高可信样本保留原始 Point-Cache 的低熵适应能力。
```

如果该规则有效，论文叙事可以收敛为：

```text
DPC-Point uses distribution-guided cache purification for uncertain shifted samples,
while preserving entropy-based adaptation for trusted samples.
```

中文表述：

```text
DPC-Point 对不确定、可能发生分布偏移的样本使用分布引导缓存净化；
同时，对低熵低能量的可信样本保留原始 Point-Cache 的低熵适应能力。
```
