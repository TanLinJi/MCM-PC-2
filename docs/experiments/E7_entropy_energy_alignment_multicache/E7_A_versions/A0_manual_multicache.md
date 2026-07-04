# E7-A0：原始手动权重多缓存

日期：2026-06-12  
状态：已完成

---

## 1. 实验目的

E7-A0 是第一版熵-能量-对齐多缓存（Entropy-Energy-Alignment Multi-Cache）。

它验证的问题是：

```text
在不使用局部缓存（local cache）的情况下，
熵缓存、能量缓存和后置对齐缓存能否替代原 Point-Cache 的全局+局部缓存结构。
```

---

## 2. 运行脚本

脚本位置：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_1_ulip_modelnetc_s2_zs_global_e7_a_entropy_energy_alignment_cache_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh
```

运行命令：

```bash
cd Point-Cache
bash scripts/E7_entropy_energy_alignment_multicache/00_1_ulip_modelnetc_s2_zs_global_e7_a_entropy_energy_alignment_cache_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh 0
```

结果目录：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_1_ulip_modelnetc_s2_zs_global_e7_a_entropy_energy_alignment_cache_manual_full_clip_manualfull_llm_dynamic_init_textdist
```

---

## 3. 参数设置

| 参数 | 数值 | 含义 |
|---|---:|---|
| `alpha_ZS` | 1.0 | 零样本得分（zero-shot logits）权重 |
| `alpha_H` | 2.0 | 熵缓存得分（entropy cache logits）权重 |
| `alpha_E` | 2.0 | 能量缓存得分（energy cache logits）权重 |
| `alpha_A` | 2.0 | 对齐缓存得分（alignment cache logits）权重 |
| `beta_H` | 3.0 | 熵缓存相似度温度 |
| `beta_E` | 3.0 | 能量缓存相似度温度 |
| `beta_A` | 3.0 | 对齐缓存相似度温度 |
| 熵缓存容量 | 5 / class | 每类最多 5 个熵缓存样本 |
| 能量缓存容量 | 5 / class | 每类最多 5 个能量缓存样本 |
| 对齐缓存容量 | 3 / class | 每类最多 3 个对齐缓存样本 |

最终得分：

```text
S_final = S_zs + 2.0*S_H + 2.0*S_E + 2.0*S_A
```

---

## 4. 验证目标

1. E7-A0 是否能超过原始 global-only cache。
2. E7-A0 是否能接近或超过当前 anchor `02_9_2`。
3. 对齐缓存是否有足够触发率。
4. 取消局部缓存（local cache）后，哪些 corruption 损失最大。

---

## 5. 结果

| 方法 | ModelNet-C S2 平均准确率 |
|---|---:|
| Zero-shot | 47.68 |
| 原始 Global Cache | 52.66 |
| 原始 Global + Local | 54.00 |
| `02_9_2` | 54.71 |
| E7-A0 | 53.31 |

逐项对比 `02_9_2`：

| corruption | E7-A0 | `02_9_2` | 差值 |
|---|---:|---:|---:|
| add_global_2 | 48.50 | 47.89 | +0.61 |
| add_local_2 | 47.57 | 50.85 | -3.28 |
| dropout_global_2 | 57.66 | 59.12 | -1.46 |
| dropout_local_2 | 56.16 | 57.21 | -1.05 |
| rotate_2 | 61.14 | 61.30 | -0.16 |
| scale_2 | 53.04 | 55.92 | -2.88 |
| jitter_2 | 49.11 | 50.65 | -1.54 |

---

## 6. 结果分析

E7-A0 不是完全失败。它在不使用局部缓存（local cache）的情况下，S2 平均准确率达到 `53.31`，超过原始 global-only cache 的 `52.66`，说明熵缓存、能量缓存和对齐缓存的结构确实提供了一些额外信息。

但 E7-A0 没有超过完整 Point-Cache，也没有超过当前 anchor `02_9_2`。主要损失集中在：

```text
add_local_2: -3.28
scale_2: -2.88
```

这说明局部缓存或原有局部结构在某些 corruption 下仍有重要作用。E7-A0 目前只能说明“全局多缓存结构有潜力”，还不能作为主方案。

诊断中，对齐缓存测试阶段触发率较低，约 `0.93% - 1.34%`，说明对齐缓存没有成为一个活跃的在线筛选模块，更多依赖预构建阶段形成的缓存状态。

---

## 7. 下一步计划状态

已基于 A0 进入 A1：降低缓存权重并添加 logits norm（得分向量范数）诊断，验证问题是否来自缓存得分过强。
