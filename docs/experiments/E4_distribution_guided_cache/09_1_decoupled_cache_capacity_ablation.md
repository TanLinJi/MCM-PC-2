# 09_1：02_9_2 缓存容量解耦消融计划

日期：2026-06-30

状态：代码已实现；ModelNet-C S2 容量消融、all35 完整验证和 clean 验证均已完成。当前主配置确定为 `e3_g3_l3_n5 = (3,3,3,5)`。

## 1. 实验目的

本实验基于 `02_9_2 / E4-C-A0+E1-textdist-only`，目标是对四类缓存的样本容量做可控消融：

```text
1. 全局熵缓存容量是否过小或过大？
2. GPA 全局缓存容量是否应该小于当前 K=3，以减少初始加入噪声？
3. 局部缓存容量是否应与 GPA 全局缓存容量保持一致？
4. 负缓存容量从 2 调到 1 或 3 时，是否会减少误伤或增加有效抑制？
```

本实验不提出新的 gate，不改 text distribution score，不改最终 logits 融合规则。它只把原先耦合在一起的容量变量显式拆开，作为容量消融和复现检查。

## 2. 与 02_9_2 的关系

`09_1` 必须严格继承 `02_9_2` 的主体设置：

```text
载体：E4-C-A0+E1-textdist-only
数据集：ModelNet-C
扰动等级：severity=2
backbone：ULIP
最终分类器：manual_full
最终 logits：沿用 02_9_2 Point-Cache voting
E4_TEXT_DIST_PROMPT_SOURCE=manualfull_llm_dynamic_init
E4_TEXT_SCORE_WEIGHT=0.15
E4_SCORE_NORM_MODE=running_zscore
dynamic prompt count=10
manual_full:LLM = 0.75:0.25
```

默认容量设置应与 `02_9_2` 完全一致：

```text
entropy_cap = 3
gpa_cap = 3
local_cap = 3
neg_cap = 2
local_centers = 3
```

因此，`09_1-default` 是代码拆分后的复现检查。若默认设置不能复现 `02_9_2` 的 S2 结果，则优先排查容量解耦实现，而不是分析新结论。

参考 `02_9_2` S2 结果：

| Corruption | 02_9_2 Acc |
|---|---:|
| add_global | 48.06 |
| add_local | 50.85 |
| dropout_global | 59.12 |
| dropout_local | 57.46 |
| rotate | 60.94 |
| scale | 55.83 |
| jitter | 50.49 |
| **Average** | **54.68** |

## 3. 容量定义

本文中的“缓存容量”统一定义为：

```text
每类最多缓存多少个样本。
```

不把局部 KMeans 中心数量计入缓存容量。

| 名称 | 含义 | 当前 02_9_2 |
|---|---|---:|
| `entropy_cap` | 全局熵缓存每类样本容量 | 3 |
| `gpa_cap` | GPA 全局缓存每类样本容量 | 3 |
| `local_cap` | 局部缓存每类样本容量 | 3 |
| `neg_cap` | 负缓存每类样本容量 | 2 |
| `local_centers` | 每个样本的 KMeans 局部中心数量，原 `n_cluster` | 3 |

建议环境变量命名：

```text
E4_ENTROPY_CAP
E4_GPA_CAP
E4_LOCAL_CAP
E4_NEG_CAP
E4_LOCAL_CENTERS
```

其中 `local_centers` 只控制每个进入局部缓存的样本保存多少个局部中心，不作为容量消融主变量。

## 4. GPA 与局部缓存关系

当前 `02_9_2` 的语义是：

```text
只有进入 GPA cache 的样本，才会把它的 patch_centers 写入 local cache。
```

代码逻辑上，`gpa_cache` 与 `gpa_local_cache` 是成对加入、成对替换、成对排序的。因此主实验保持：

```text
local_cap = gpa_cap
```

原因：

1. 这符合当前 DPC-Point 方法定义。
2. local cache 存的是同一批 GPA accepted samples 的局部特征。
3. 若 `local_cap > gpa_cap`，当前语义下没有额外样本来源。
4. 若 `local_cap < gpa_cap`，它变成“只保留 GPA 样本子集的局部特征”的额外诊断，不属于第一批主容量消融。

实现时可以保留 `local_cap` 变量用于显式记录和后续诊断，但第一批正式实验只跑 `local_cap = gpa_cap`。

## 5. 历史诊断依据

`02_16_1` 诊断显示，当前 `K=3` 时正缓存最大容量为 `40 * 3 = 120`。各扰动最终填充如下：

| Corruption | Entropy total | GPA total | Local total |
|---|---:|---:|---:|
| add_global | 91 | 91 | 91 |
| add_local | 107 | 107 | 107 |
| dropout_global | 116 | 116 | 116 |
| dropout_local | 118 | 118 | 118 |
| rotate | 118 | 118 | 118 |
| scale | 120 | 120 | 120 |
| jitter | 107 | 107 | 107 |

解释：

1. `dropout_global/dropout_local/rotate/scale` 基本接近或达到满容量。
2. `add_global/add_local/jitter` 未满，瓶颈更可能是 gate 拒绝，而不是容量本身。
3. 容量增大不一定有效，尤其是 GPA/cache-local 这一路。

GPA event 伪标签正确率：

| 事件 | 数量 | 正确率 |
|---|---:|---:|
| 初次加入 `add_not_full` | 777 | 46.5% |
| 满后替换 `replace` | 885 | 69.9% |
| 熵门控拒绝 `reject_entropy` | 23497 | 41.5% |
| 联合分数拒绝 `reject_joint` | 9393 | 60.9% |

解释：

1. GPA 初始加入阶段较 noisy。
2. 满后 replacement 更干净。
3. 盲目增大 `gpa_cap/local_cap` 可能让更多 noisy initial samples 进入缓存。
4. 因此 `gpa_cap=2` 是高优先级实验，`gpa_cap=4` 是必要反证实验。

## 6. 第一批容量设置

第一批只做 ModelNet-C severity=2 的 7 类扰动。

| 设置名 | entropy_cap | gpa_cap | local_cap | neg_cap | local_centers | 目的 |
|---|---:|---:|---:|---:|---:|---|
| `default` | 3 | 3 | 3 | 2 | 3 | 复现 02_9_2，验证解耦代码无额外行为变化 |
| `entropy2` | 2 | 3 | 3 | 2 | 3 | 检查全局熵缓存变小是否减少噪声 |
| `entropy4` | 4 | 3 | 3 | 2 | 3 | 检查全局熵缓存轻微增大是否提升覆盖 |
| `gpa2` | 3 | 2 | 2 | 2 | 3 | 重点实验：减少 GPA/local 初始噪声 |
| `gpa4` | 3 | 4 | 4 | 2 | 3 | 反证实验：验证 GPA/local 变大是否引入噪声 |
| `neg1` | 3 | 3 | 3 | 1 | 3 | 检查负缓存容量降低是否减少误伤 |
| `neg3` | 3 | 3 | 3 | 3 | 3 | 检查负缓存容量增加是否提升有效抑制 |

优先执行顺序：

```text
default -> gpa2 -> entropy4 -> neg1 -> neg3 -> entropy2 -> gpa4
```

## 7. 不纳入第一批的设置

`local_centers` 的消融已经以 `n_cluster=2..7` 跑过。它控制每个样本的 KMeans 局部中心数量，不是本文定义的缓存样本容量。

第一批不重复跑 `local_centers`，只固定：

```text
local_centers = 3
```

`local_cap != gpa_cap` 暂不作为主实验设置。如果后续需要，可以单独设计为 local-subset diagnostic，而不是容量主表的一部分。

## 8. 已实现文件

已新增文件，避免覆盖历史实验：

```text
runners/E4_distribution_guided_cache/model_e4_c_a0_e1_decoupled_cache_capacity.py
runners/E4_distribution_guided_cache/run_e4_c_a0_e1_decoupled_cache_capacity_ulip_modelnetc_s2.py
scripts/E4_distribution_guided_cache/09_run_e4_c_a0_e1_decoupled_cache_capacity_common.sh
scripts/E4_distribution_guided_cache/09_1_ulip_modelnetc_s2_decoupled_cache_capacity_ablation.sh
```

计划结果目录模式：

```text
results/E4_distribution_guided_cache/09_1_<setting>_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist/
```

其中 `<setting>` 对应：

```text
default
entropy2
entropy4
gpa2
gpa4
neg1
neg3
```

## 9. 结果记录要求

每个设置至少保存：

```text
summary.csv
logs/
gpa_stats/
```

`gpa_stats/*.json` 中需要显式记录：

```text
entropy_cap
gpa_cap
local_cap
neg_cap
local_centers
```

运行后分析表至少包括：

| 指标 | 说明 |
|---|---|
| final accuracy | 每个 corruption 的最终准确率 |
| average accuracy | 7 类扰动平均 |
| cache totals | 四类缓存最终填充量 |
| update counts | add / replace / reject 统计 |
| GPA event correctness | add / replace / reject 的伪标签正确率 |
| 与 default 差值 | 判断容量变化是否有效 |

## 10. 判定标准

`default` 设置：

```text
目标：复现 02_9_2 S2 average ~= 54.68。
若单项或平均差异明显超过 0.1 pp，需要先检查代码拆分。
```

容量消融：

```text
若 gpa2 提升或持平，说明减少 GPA/local 初始噪声有价值。
若 gpa4 下降，支持 GPA/local 容量不宜盲目增大。
若 entropy4 只提升 rotate/add_global 但平均不提升，说明全局熵容量适合作为扰动相关调节项。
若 neg1 优于 neg3，说明负缓存存在误伤风险。
若 neg3 优于 neg1，说明当前负缓存覆盖不足。
```

## 11. 第一批结果

完成时间：2026-06-30

所有设置均完成 ModelNet-C severity=2 的 7 类扰动。结果目录：

```text
results/E4_distribution_guided_cache/09_1_<setting>_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist/
```

### 11.1 Accuracy 汇总

| Setting | entropy_cap | gpa_cap | local_cap | neg_cap | add_global | add_local | dropout_global | dropout_local | rotate | scale | jitter | Average | vs default |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` | 3 | 3 | 3 | 2 | 48.06 | 50.85 | 59.12 | 57.46 | 60.94 | 55.83 | 50.49 | 54.68 | +0.00 |
| `gpa2` | 3 | 2 | 2 | 2 | 49.07 | 49.80 | 58.27 | 56.65 | 61.51 | 54.62 | 50.65 | 54.37 | -0.31 |
| `gpa4` | 3 | 4 | 4 | 2 | 47.49 | 50.45 | 58.91 | 57.33 | 60.98 | 57.05 | 50.61 | 54.69 | +0.01 |
| `entropy2` | 2 | 3 | 3 | 2 | 49.27 | 50.00 | 59.08 | 56.77 | 60.62 | 55.75 | 51.34 | 54.69 | +0.01 |
| `entropy4` | 4 | 3 | 3 | 2 | 48.26 | 50.00 | 58.51 | 57.46 | 61.22 | 56.00 | 50.49 | 54.56 | -0.12 |
| `neg1` | 3 | 3 | 3 | 1 | 47.20 | 50.69 | 59.20 | 57.29 | 60.94 | 55.79 | 50.49 | 54.51 | -0.16 |
| `neg3` | 3 | 3 | 3 | 3 | 48.62 | 50.93 | 59.16 | 57.37 | 61.10 | 55.75 | 50.49 | 54.77 | +0.10 |

### 11.2 缓存最终填充量

表中为 7 类扰动上的平均缓存样本数。容量单位为“每类最多缓存多少个样本”。

| Setting | entropy total avg | GPA total avg | local total avg | visual dist classes avg |
|---|---:|---:|---:|---:|
| `default` | 111.00 | 111.00 | 111.00 | 37.43 |
| `gpa2` | 111.00 | 74.57 | 74.57 | 37.43 |
| `gpa4` | 111.00 | 147.43 | 147.43 | 37.43 |
| `entropy2` | 74.57 | 111.00 | 111.00 | 37.43 |
| `entropy4` | 147.43 | 111.00 | 111.00 | 37.43 |
| `neg1` | 111.00 | 111.00 | 111.00 | 37.43 |
| `neg3` | 111.00 | 111.00 | 111.00 | 37.43 |

### 11.3 结论

1. `09_1-default` 复现了当前单独重跑的 `02_9_2`：平均 `54.68`，可作为本轮容量消融基准。
2. `neg3` 是本轮最优，平均 `54.77`，比 default 高 `+0.10`。提升主要来自 `add_global`、`add_local`、`rotate`，但幅度较小。
3. `neg1` 明显低于 default，平均 `54.51`。负缓存从 2 减到 1 会削弱有效抑制，尤其 `add_global` 下降明显。
4. `gpa2` 平均 `54.37`，低于 default `-0.31`。减少 GPA/local 容量虽然提升了 `add_global` 和 `rotate`，但损伤 `add_local/dropout/scale`，说明 GPA/local 容量 2 不适合作为默认。
5. `gpa4` 平均 `54.69`，几乎等于 default。它显著提升 `scale`，但损伤 `add_global` 和 `add_local`，说明单纯增大 GPA/local 容量没有稳定收益。
6. `entropy2` 平均 `54.69`，几乎等于 default。它提升 `add_global` 和 `jitter`，但损伤 `dropout_local/rotate`。
7. `entropy4` 平均 `54.56`，低于 default。全局熵缓存增大到 4 不稳定，尤其损伤 `dropout_global/add_local`。

本轮建议：若只选一个后续候选，优先考虑 `neg3`；若追求稳健主线，仍保留 `default`，因为 `neg3` 的平均提升只有 `+0.10`，需要在 clean、其他 severity 或 all35 上再验证。

## 12. 组合容量追加实验

### 12.1 `e2_g4_l4_n3`

容量设置：

```text
entropy_cap = 2
gpa_cap = 4
local_cap = 4
neg_cap = 3
local_centers = 3
```

结果目录：

```text
results/E4_distribution_guided_cache/09_1_e2_g4_l4_n3_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist/
```

| Corruption | default | e2_g4_l4_n3 | vs default |
|---|---:|---:|---:|
| add_global | 48.06 | 48.01 | -0.05 |
| add_local | 50.85 | 50.61 | -0.24 |
| dropout_global | 59.12 | 59.60 | +0.48 |
| dropout_local | 57.46 | 57.13 | -0.33 |
| rotate | 60.94 | 60.74 | -0.20 |
| scale | 55.83 | 55.51 | -0.32 |
| jitter | 50.49 | 50.97 | +0.48 |
| **Average** | **54.68** | **54.65** | **-0.03** |

缓存填充量符合预期：

| Setting | entropy total avg | GPA total avg | local total avg | visual dist classes avg |
|---|---:|---:|---:|---:|
| default | 111.00 | 111.00 | 111.00 | 37.43 |
| e2_g4_l4_n3 | 74.57 | 147.43 | 147.43 | 37.43 |

结论：

1. `e2_g4_l4_n3` 平均 `54.65`，低于 default `-0.03`，低于当前最优 `neg3` 的 `54.77`。
2. 该组合只明显提升 `dropout_global` 和 `jitter`，但损伤 `add_local/dropout_local/rotate/scale`。
3. `entropy2`、`gpa4`、`neg3` 的单独收益没有叠加，说明这些容量变化之间存在相互抵消。
4. 当前不建议把 `(2,4,4,3)` 作为主线默认配置。

### 12.2 `e2_g3_l3_n3` 与 `e3_g4_l4_n3`

追加完成时间：2026-06-30

容量设置：

| Setting | entropy_cap | gpa_cap | local_cap | neg_cap | local_centers |
|---|---:|---:|---:|---:|---:|
| `e2_g3_l3_n3` | 2 | 3 | 3 | 3 | 3 |
| `e3_g4_l4_n3` | 3 | 4 | 4 | 3 | 3 |

结果目录：

```text
results/E4_distribution_guided_cache/09_1_e2_g3_l3_n3_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist/
results/E4_distribution_guided_cache/09_1_e3_g4_l4_n3_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist/
```

| Setting | add_global | add_local | dropout_global | dropout_local | rotate | scale | jitter | Average | vs default | vs neg3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` | 48.06 | 50.85 | 59.12 | 57.46 | 60.94 | 55.83 | 50.49 | 54.679 | +0.000 | -0.096 |
| `neg3` | 48.62 | 50.93 | 59.16 | 57.37 | 61.10 | 55.75 | 50.49 | 54.774 | +0.096 | +0.000 |
| `e2_g3_l3_n3` | 49.59 | 50.20 | 59.12 | 56.77 | 60.66 | 55.83 | 51.26 | 54.776 | +0.097 | +0.001 |
| `e3_g4_l4_n3` | 47.93 | 50.45 | 58.67 | 57.46 | 60.90 | 56.93 | 50.61 | 54.707 | +0.029 | -0.067 |
| `e2_g4_l4_n3` | 48.01 | 50.61 | 59.60 | 57.13 | 60.74 | 55.51 | 50.97 | 54.653 | -0.026 | -0.121 |

缓存填充量：

| Setting | entropy total avg | GPA total avg | local total avg | visual dist classes avg |
|---|---:|---:|---:|---:|
| `default` | 111.00 | 111.00 | 111.00 | 37.43 |
| `neg3` | 111.00 | 111.00 | 111.00 | 37.43 |
| `e2_g3_l3_n3` | 74.57 | 111.00 | 111.00 | 37.43 |
| `e3_g4_l4_n3` | 111.00 | 147.43 | 147.43 | 37.43 |
| `e2_g4_l4_n3` | 74.57 | 147.43 | 147.43 | 37.43 |

结论：

1. `e2_g3_l3_n3` 平均 `54.776`，与 `neg3` 的 `54.774` 基本打平，只高 `+0.001`，不能视为显著提升。
2. `e2_g3_l3_n3` 的收益主要来自 `add_global` 和 `jitter`，但明显牺牲 `add_local/dropout_local/rotate`。
3. `e3_g4_l4_n3` 平均 `54.707`，比 default 高 `+0.029`，但低于 `neg3` `-0.067`。它主要提升 `scale`，同时损伤 `add_global/add_local/dropout_global/rotate`。
4. `e2_g4_l4_n3` 同时使用 entropy2、gpa4/local4、neg3，平均反而低于 default，说明三种容量变化不能简单叠加。
5. 当前最稳健的候选仍是 `neg3`；若只看两位小数，`e2_g3_l3_n3` 与 `neg3` 都是 `54.78`，需要更多 severity 或 all35 才能判断是否存在真实差异。

### 12.3 `e2_g3_l3_n4` 与 `e3_g3_l3_n4`

追加完成时间：2026-06-30

容量设置：

| Setting | entropy_cap | gpa_cap | local_cap | neg_cap | local_centers |
|---|---:|---:|---:|---:|---:|
| `e3_g3_l3_n4` | 3 | 3 | 3 | 4 | 3 |
| `e2_g3_l3_n4` | 2 | 3 | 3 | 4 | 3 |

结果目录：

```text
results/E4_distribution_guided_cache/09_1_e3_g3_l3_n4_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist/
results/E4_distribution_guided_cache/09_1_e2_g3_l3_n4_ulip_modelnetc_s2_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist/
```

| Setting | add_global | add_local | dropout_global | dropout_local | rotate | scale | jitter | Average | vs default | vs neg3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` | 48.06 | 50.85 | 59.12 | 57.46 | 60.94 | 55.83 | 50.49 | 54.679 | +0.000 | -0.096 |
| `neg3` | 48.62 | 50.93 | 59.16 | 57.37 | 61.10 | 55.75 | 50.49 | 54.774 | +0.096 | +0.000 |
| `e3_g3_l3_n4` | 49.07 | 50.85 | 59.08 | 57.29 | 60.98 | 55.75 | 50.53 | 54.793 | +0.114 | +0.019 |
| `e2_g3_l3_n3` | 49.59 | 50.20 | 59.12 | 56.77 | 60.66 | 55.83 | 51.26 | 54.776 | +0.097 | +0.001 |
| `e2_g3_l3_n4` | 50.08 | 50.24 | 59.08 | 56.89 | 60.62 | 55.88 | 51.09 | 54.840 | +0.161 | +0.066 |

缓存填充量：

| Setting | entropy total avg | GPA total avg | local total avg | visual dist classes avg |
|---|---:|---:|---:|---:|
| `default` | 111.00 | 111.00 | 111.00 | 37.43 |
| `neg3` | 111.00 | 111.00 | 111.00 | 37.43 |
| `e3_g3_l3_n4` | 111.00 | 111.00 | 111.00 | 37.43 |
| `e2_g3_l3_n3` | 74.57 | 111.00 | 111.00 | 37.43 |
| `e2_g3_l3_n4` | 74.57 | 111.00 | 111.00 | 37.43 |

负缓存 test 阶段统计：

| Setting | test_neg_add | test_neg_replace | test_neg_reject |
|---|---:|---:|---:|
| `default` | 504 | 788 | 3159 |
| `neg3` | 741 | 932 | 2778 |
| `e3_g3_l3_n4` | 966 | 1006 | 2479 |
| `e2_g3_l3_n3` | 741 | 932 | 2778 |
| `e2_g3_l3_n4` | 966 | 1006 | 2479 |

结论：

1. `e3_g3_l3_n4` 平均 `54.793`，高于 `neg3` `+0.019`，说明负缓存从 3 增到 4 仍有一点收益，但幅度很小。
2. `e2_g3_l3_n4` 平均 `54.840`，是当前 09_1 容量消融中最高的 S2 平均，比 default 高 `+0.161`，比 `neg3` 高 `+0.066`。
3. `e2_g3_l3_n4` 的主要收益来自 `add_global`：`50.08`，比 default 高 `+2.02`；同时 `jitter` 高于 default `+0.60`。
4. 代价也很明确：`e2_g3_l3_n4` 损伤 `add_local/dropout_local/rotate`，所以它更像是偏向 add_global/jitter 的容量组合，而不是全面提升。
5. 负缓存容量增大到 4 后，test 阶段负缓存接收更多样本：`test_neg_add/test_neg_replace/test_neg_reject = 966/1006/2479`，与 `neg3` 的 `741/932/2778` 相比，说明负缓存覆盖继续增强。
6. 当前最值得继续验证的候选变为 `e2_g3_l3_n4`；但平均提升仍只有 `+0.16`，需要 clean、其他 severity 或 all35 验证稳定性。

## 13. 诊断统计扩展

更新时间：2026-06-30

为解释容量消融结果，`09_1` 代码临时加入诊断统计。所有诊断代码均带有发布前删除标记：

```text
DIAG_ONLY_REMOVE_FOR_RELEASE
```

新增统计不会改变缓存更新规则或最终 logits，只写入 `gpa_stats/*.json` 和 GPA event jsonl。

### 13.1 新增 JSON 标记

`gpa_stats/*.json` 顶层新增：

```text
diagnostic_marker = DIAG_ONLY_REMOVE_FOR_RELEASE
diagnostic_only_remove_for_release = true
diagnostic_stats_version = cache_quality_and_prediction_transitions_v1
```

### 13.2 新增 cache quality 统计

`gpa_stats/*.json` 新增：

```text
diagnostic_cache_quality.entropy_cache
diagnostic_cache_quality.gpa_cache
diagnostic_cache_quality.gpa_local_cache
diagnostic_cache_quality.negative_cache
```

每个缓存记录：

```text
total
pred_correct
pred_wrong
unknown
pred_correct_rate
pred_wrong_rate
by_class
```

其中负缓存的 `pred_correct` 可视为潜在误伤风险，`pred_wrong` 可视为潜在有效抑制来源。

### 13.3 新增 diag_* 计数

`stats` 中新增 `diag_*` 前缀的计数，包括：

```text
diag_<phase>_<cache>_<decision>_pred_correct
diag_<phase>_<cache>_<decision>_pred_wrong
diag_<phase>_neg_<decision>_potential_misfire
diag_<phase>_neg_<decision>_potential_helpful
```

以及最终预测阶段转换：

```text
diag_test_pred_zero_shot_*
diag_test_pred_after_entropy_*
diag_test_pred_after_local_*
diag_test_pred_after_negative_*
diag_test_pred_final_*
diag_test_transition_*_right_to_wrong
diag_test_transition_*_wrong_to_right
diag_test_transition_*_right_to_right
diag_test_transition_*_wrong_to_wrong
```

这些统计用于判断某个缓存分支是把错误样本修正为正确，还是把正确样本误伤为错误。

### 13.4 GPA event 扩展

`gpa_replacement_events_*.jsonl` 中每条事件新增：

```text
diag_marker
new_pred
new_target
new_pred_correct
old_pred
old_target
old_pred_correct
```

这用于分析 GPA add / replace / reject 中新旧样本的伪标签质量。

## 14. All35 完整验证入口

S2 容量消融完成后，选择两组候选设置跑 ModelNet-C all35 完整验证：

| 设置名 | entropy_cap | gpa_cap | local_cap | neg_cap | 选择原因 |
|---|---:|---:|---:|---:|---|
| `e3_g3_l3_n5` | 3 | 3 | 3 | 5 | 相对 default 更均衡，验证负缓存容量增大是否在 all35 上稳定有效 |
| `e2_g3_l3_n5` | 2 | 3 | 3 | 5 | S2 平均最高，验证该收益是否能推广到 all35 |

新增 all35 入口：

```text
runners/E4_distribution_guided_cache/modelnetc_all35_09_1_decoupled_cache_capacity/launch_09_1_modelnetc_all35_decoupled_cache_capacity.py
runners/E4_distribution_guided_cache/modelnetc_all35_09_1_decoupled_cache_capacity/worker_09_1_modelnetc_all35_decoupled_cache_capacity.py
scripts/E4_distribution_guided_cache/09_1_ulip_modelnetc_all35_decoupled_cache_capacity_ablation.sh
```

运行命令：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/E4_distribution_guided_cache/09_1_ulip_modelnetc_all35_decoupled_cache_capacity_ablation.sh 0
```

其中最后的 `0` 是单张 4090 的物理 GPU id；如果环境里只有一张卡，也可以不写这个参数，默认使用 `0`。

结果目录：

```text
results/E4_distribution_guided_cache/09_1_e3_g3_l3_n5_all35_ulip_modelnetc_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist/
results/E4_distribution_guided_cache/09_1_e2_g3_l3_n5_all35_ulip_modelnetc_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist/
```

每个目录会保存：

```text
summary.csv
summary_worker*.csv
all35_table.csv
all35_table.md
all35_table.html
logs/
gpa_stats/
```

## 15. 结果记录：All35 与 clean 验证

日期：2026-07-01

本节记录 `09_1` 容量消融后进入完整验证阶段的结论。候选设置为：

```text
e3_g3_l3_n5 = entropy_cap=3, gpa_cap=3, local_cap=3, neg_cap=5
e2_g3_l3_n5 = entropy_cap=2, gpa_cap=3, local_cap=3, neg_cap=5
```

### 15.1 All35 结果

结果文件：

```text
results/E4_distribution_guided_cache/09_1_e3_g3_l3_n5_all35_ulip_modelnetc_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist/all35_table.csv
results/E4_distribution_guided_cache/09_1_e2_g3_l3_n5_all35_ulip_modelnetc_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist/all35_table.csv
```

整体结果：

| 配置 | S0 | S1 | S2 | S3 | S4 | All35 Avg |
|---|---:|---:|---:|---:|---:|---:|
| `02_9_2` | 59.93 | 56.70 | 54.71 | 50.26 | 44.51 | 53.22 |
| `e3_g3_l3_n5` | 59.85 | 56.63 | 54.85 | 50.70 | 45.00 | 53.41 |
| `e2_g3_l3_n5` | 59.35 | 56.58 | 54.91 | 50.06 | 44.52 | 53.09 |

结论：

```text
e3_g3_l3_n5 是 all35 上最优且更稳的设置。
e2_g3_l3_n5 虽然 S2 最高，但不能推广到 all35。
```

`e2_g3_l3_n5` 的 S2 平均为 54.91，高于 `e3_g3_l3_n5` 的 54.85；但它在 S0、S3 和 S4 上明显更弱，最终 all35 平均只有 53.09，低于 `02_9_2` 的 53.22。因此，`entropy_cap=2` 更像是 severity=2 上的局部最优，不适合作为主配置。

### 15.2 与 PointCache all35 对比

PointCache 对比来源：

```text
docs/experiments/E0_baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local.md
results/E0_baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local/summary.csv
```

整体对比：

| 方法 | 配置 | S2 Avg | All35 Avg |
|---|---|---:|---:|
| PointCache | `zs_global_local` | 54.00 | 53.01 |
| DPC-Point | `e3_g3_l3_n5` | 54.85 | 53.41 |
| DPC-Point | `e2_g3_l3_n5` | 54.91 | 53.09 |

主配置相对 PointCache：

```text
S2 Avg:    54.85 - 54.00 = +0.85
All35 Avg: 53.41 - 53.01 = +0.40
```

按 corruption 平均对比：

| Corruption | PointCache | DPC `e3_g3_l3_n5` | 差值 |
|---|---:|---:|---:|
| add_global | 48.43 | 49.03 | +0.61 |
| add_local | 49.16 | 50.71 | +1.56 |
| dropout_global | 57.83 | 58.09 | +0.26 |
| dropout_local | 55.08 | 55.44 | +0.36 |
| rotate | 59.08 | 58.15 | -0.93 |
| scale | 56.30 | 56.10 | -0.20 |
| jitter | 45.22 | 46.32 | +1.10 |
| **Average** | **53.01** | **53.41** | **+0.40** |

解释：

```text
DPC-Point 的主要优势来自 add_local 和 jitter。
rotate 是当前主配置相对 PointCache 的主要短板。
scale 略低于 PointCache，但差距较小。
```

因此，论文或报告中的主表建议使用 `e3_g3_l3_n5` 作为 DPC-Point 结果，与 PointCache 的完整 `zs_global_local` baseline 对比。

### 15.3 clean ModelNet 验证

clean 验证使用与 `02_9_2_clean` 相同的数据口径：

```text
data/modelnet_c/clean.h5
```

结果文件：

```text
results/E4_distribution_guided_cache/09_1_e3_g3_l3_n5_clean_ulip_modelnetc_clean_zs_global_local_e4_c_a0_e1_decoupled_cache_capacity_tw0p15_score_norm_manual_full_clip_manualfull_llm_dynamic_init_textdist/summary.csv
```

clean 结果：

| 方法 | clean Acc |
|---|---:|
| Zero-shot | 56.77 |
| PointCache Global | 62.12 |
| PointCache Global + Local | 64.18 |
| `02_9_2 clean` | 63.86 |
| DPC-Point `e3_g3_l3_n5 clean` | 64.14 |

关键差值：

| 对比 | 差值 |
|---|---:|
| DPC `e3_g3_l3_n5` vs Zero-shot | +7.37 |
| DPC `e3_g3_l3_n5` vs PointCache Global | +2.02 |
| DPC `e3_g3_l3_n5` vs PointCache Global + Local | -0.04 |
| DPC `e3_g3_l3_n5` vs `02_9_2 clean` | +0.28 |

clean 诊断显示：

| 阶段 | Acc |
|---|---:|
| zero-shot stage | 56.40 |
| after entropy cache | 61.18 |
| after local cache | 64.18 |
| after negative cache / final | 64.14 |

解释：

```text
clean 上主要收益来自正缓存，尤其是 local cache。
neg_cap=5 没有明显破坏 clean 性能；最终只比 after local cache 低约 0.04。
```

### 15.4 最终选择

综合 S2、all35、PointCache 对比和 clean 验证：

```text
最终主配置：e3_g3_l3_n5 = (3,3,3,5)
不采用为主配置：e2_g3_l3_n5 = (2,3,3,5)
```

选择理由：

1. `e3_g3_l3_n5` 在 all35 上达到 53.41，高于 PointCache 53.01，也高于 `02_9_2` 53.22。
2. `e3_g3_l3_n5` 在 clean 上达到 64.14，基本持平 PointCache clean 64.18，没有明显牺牲干净集性能。
3. `e2_g3_l3_n5` 虽然 S2 更高，但 all35 平均低于 `02_9_2`，说明 `entropy_cap=2` 不够稳。
4. 负缓存容量从 2 增大到 5 是有效方向；但全局熵缓存容量仍应保持 3。
