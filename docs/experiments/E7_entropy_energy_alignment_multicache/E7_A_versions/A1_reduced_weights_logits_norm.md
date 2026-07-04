# E7-A1：降低缓存权重与 Logits Norm 诊断

日期：2026-06-12  
状态：已完成

---

## 1. 实验目的

E7-A1 用于验证 E7-A0 的下降是否来自“正向缓存得分过强”。

如果缓存过强，那么降低缓存权重后应当：

```text
降低错误改写零样本预测的比例；
提高或至少稳定最终准确率。
```

---

## 2. 运行脚本

脚本位置：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_2_ulip_modelnetc_s2_zs_global_e7_a1_entropy_energy_alignment_cache_h0p6_e0p6_a0p9_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh
```

运行命令：

```bash
cd Point-Cache
bash scripts/E7_entropy_energy_alignment_multicache/00_2_ulip_modelnetc_s2_zs_global_e7_a1_entropy_energy_alignment_cache_h0p6_e0p6_a0p9_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh 0
```

结果目录：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_2_ulip_modelnetc_s2_zs_global_e7_a1_entropy_energy_alignment_cache_h0p6_e0p6_a0p9_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

---

## 3. 参数设置

| 参数 | 数值 | 含义 |
|---|---:|---|
| `alpha_ZS` | 1.0 | 零样本得分（zero-shot logits）权重 |
| `alpha_H` | 0.6 | 熵缓存得分（entropy cache logits）权重 |
| `alpha_E` | 0.6 | 能量缓存得分（energy cache logits）权重 |
| `alpha_A` | 0.9 | 对齐缓存得分（alignment cache logits）权重 |
| `beta_H` | 3.0 | 熵缓存相似度温度 |
| `beta_E` | 3.0 | 能量缓存相似度温度 |
| `beta_A` | 3.0 | 对齐缓存相似度温度 |

最终得分：

```text
S_final = S_zs + 0.6*S_H + 0.6*S_E + 0.9*S_A
```

---

## 4. 新增诊断

E7-A1 添加 logits norm（得分向量范数）诊断：

| 诊断项 | 含义 |
|---|---|
| `test_zs_logits_norm_mean` | 零样本得分平均范数 |
| `test_entropy_logits_norm_mean` | 熵缓存得分平均范数 |
| `test_energy_logits_norm_mean` | 能量缓存得分平均范数 |
| `test_alignment_logits_norm_mean` | 对齐缓存得分平均范数 |
| `test_positive_cache_total_logits_norm_mean` | 三个正向缓存合计得分平均范数 |
| `test_final_logits_norm_mean` | 最终得分平均范数 |

---

## 5. 验证目标

1. 降低缓存权重是否能减少错误预测改写。
2. 缓存总范数是否显著低于零样本得分范数。
3. 如果准确率仍下降，说明问题可能不是尺度，而是缓存证据方向不稳定。

---

## 6. 结果

| 方法 | ModelNet-C S2 平均准确率 |
|---|---:|
| Zero-shot | 47.68 |
| 原始 Global Cache | 52.66 |
| E7-A0 | 53.31 |
| E7-A1 | 50.97 |

逐项对比 A0：

| corruption | A0 | A1 | 差值 |
|---|---:|---:|---:|
| add_global_2 | 48.50 | 42.38 | -6.12 |
| add_local_2 | 47.57 | 46.64 | -0.93 |
| dropout_global_2 | 57.66 | 56.44 | -1.22 |
| dropout_local_2 | 56.16 | 53.53 | -2.63 |
| rotate_2 | 61.14 | 58.71 | -2.43 |
| scale_2 | 53.04 | 52.27 | -0.77 |
| jitter_2 | 49.11 | 46.84 | -2.27 |

---

## 7. 诊断结果

A1 的平均 logits norm：

| 项目 | 平均范数 |
|---|---:|
| `S_zs` | 33.48 |
| `S_H` | 2.44 |
| `S_E` | 2.44 |
| `S_A` | 2.19 |
| `S_cache` | 7.04 |
| `S_final` | 36.83 |

`S_cache / S_zs` 平均约为 `0.216`。这说明 A1 的缓存总证据已经明显弱于零样本证据。

其他诊断：

| 指标 | 数值 |
|---|---:|
| 平均预测改变率 | 约 10.7% |
| 熵缓存/能量缓存预测一致率 | 约 81.4% |
| 对齐缓存测试阶段触发率 | 约 1.13% |

---

## 8. 结果分析

A1 说明 E7-A0 的问题不只是“缓存太强”。降低缓存权重后，预测确实更少偏离零样本预测，但准确率反而进一步下降。

这表示 A0 中一部分强缓存干预是有正收益的，A1 把这部分纠错能力削弱了。同时，A1 的 logits norm 显示缓存证据已经不强，但准确率仍然低于原始 Global Cache，说明关键问题更可能是：

```text
缓存证据的方向质量不稳定，而不是缓存证据尺度过大。
```

熵缓存和能量缓存的一致率很高，说明两者并没有形成足够互补的证据。对齐缓存触发率较低，说明它没有有效承担在线筛选和修正作用。

---

## 9. 下一步计划状态

已基于 A1 进入 A2：在 A1 权重基础上添加样本级门控（sample-wise gating），尝试保留缓存纠错能力，同时降低不可信缓存证据的影响。
