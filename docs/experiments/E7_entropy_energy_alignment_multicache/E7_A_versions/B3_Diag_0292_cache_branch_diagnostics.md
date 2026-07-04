# E7-B3-Diag-02_9_2：基于当前最好方案的缓存分支诊断

日期：2026-06-16
状态：已实现，待运行与分析

---

## 0. 实验目的

本实验是 B3 之前的诊断实验，不改变正式预测结果。

上一版 B3-Diag 使用 A4 fixed 作为载体，`add_global_2` 结果为 43.56%（1075/2468），基本复现 A4 fixed 的 43.52%（1074/2468）。这说明低结果主要来自 A4 fixed 载体本身，而不是诊断代码改变了预测。

因此，本实验改用当前最好方案 `02_9_2` 作为载体，诊断 `02_9_2` 中各个得分分支到底在帮助还是破坏最终预测。

核心问题：

```text
02_9_2 中，零样本文本原型点积得分、全局熵缓存、GPA 局部缓存、负缓存分别贡献了什么？
```

---

## 1. 载体与边界

载体：

```text
02_9_2: E4-C-A0+E1-textdist-only, text_weight=0.15, running_zscore
```

严格边界：

1. 不修改 `02_9_2` 的正式预测公式。
2. 不修改缓存更新规则。
3. 不修改文本端处理：最终分类器仍使用 `manual_full`，E1 LLM 文本只用于文本分布（text distribution）。
4. 不使用真实标签参与测试时适应（Test-Time Adaptation, TTA）。
5. 真实标签只用于离线诊断统计。

---

## 2. 02_9_2 的得分分支

02_9_2 的正式最终得分为：

```text
S_final = S_zs + S_H + S_L - S_N
```

其中：

| 符号 | 中文含义 | 英文 | 说明 |
|---|---|---|---|
| `S_zs` | 零样本文本原型点积得分 | zero-shot text prototype dot-product logits | `clip_logits`，点云特征与文本原型点积 |
| `S_H` | 全局熵缓存投票得分 | global entropy cache voting logits | 原始 Point-Cache 风格的正全局缓存 |
| `S_L` | GPA 控制的局部缓存投票得分 | GPA-controlled local cache voting logits | 只有进入 GPA 缓存的样本，其局部特征才进入局部缓存 |
| `S_N` | 负缓存惩罚得分 | negative cache penalty logits | 满足负缓存条件的样本形成惩罚项，最终以减法进入 |
| `S_final` | 最终融合得分 | final logits | 正式预测只使用这一项 |

另外，为了诊断 GPA 缓存本身，本实验额外计算：

```text
S_GPA = GPA 全局缓存投票得分
```

注意：`S_GPA` 只用于诊断，不参与 02_9_2 正式预测。02_9_2 正式预测中，GPA 缓存通过控制局部缓存 `S_L` 间接起作用。

---

## 3. 不适用项

02_9_2 没有以下结构：

| 项目 | 是否存在于 02_9_2 | 说明 |
|---|---|---|
| 候选池（candidate pool） | 否 | 候选池是 E7-A4 之后引入的结构 |
| 能量缓存（energy cache） | 否 | 02_9_2 不维护单独能量缓存 |
| 对齐核心缓存（alignment core cache） | 否 | 02_9_2 使用 GPA 缓存，不是 E7-A4 的对齐核心缓存 |

因此本实验不会伪造候选池或能量缓存统计。对应字段会明确标记为“不适用（not applicable）”。

---

## 4. 必须记录的正确数/总数

每个测试样本记录以下分支的单独预测结果：

| 诊断项 | 中文解释 |
|---|---|
| `zero_shot_text_proto_dot` | 只用零样本文本原型点积得分 `S_zs` 做预测 |
| `global_entropy_cache` | 只用全局熵缓存投票得分 `S_H` 做预测 |
| `gpa_global_cache_diag` | 只用 GPA 全局缓存诊断得分 `S_GPA` 做预测 |
| `gpa_local_cache` | 只用 GPA 控制的局部缓存投票得分 `S_L` 做预测 |
| `negative_cache_penalty` | 只用负缓存惩罚得分 `S_N` 做预测 |
| `positive_cache_total` | 只用正缓存总得分 `S_H + S_L` 做预测 |
| `cache_total_signed` | 只用带符号缓存总得分 `S_H + S_L - S_N` 做预测 |
| `final_logits` | 使用正式最终得分 `S_final` 做预测 |
| `norm_fusion_offline` | 离线归一化融合诊断，不参与正式预测 |

准确率必须同时写成：

```text
xx.xx%（correct/total）
```

例如：

```text
零样本文本原型点积得分（zero-shot text prototype dot-product logits）：
33.63%（830/2468）
```

这句话的含义是：只用原始零样本点积得分 `clip_logits` 做预测，2468 个样本里预测对了 830 个，准确率是 33.63%。

---

## 5. 帮助与破坏统计

记录以下对比：

```text
zs_correct_final_wrong：S_zs 正确但 S_final 错误
zs_wrong_final_correct：S_zs 错误但 S_final 正确
zs_correct_cache_wrong：S_zs 正确但 cache_total_signed 错误
zs_wrong_cache_correct：S_zs 错误但 cache_total_signed 正确
```

目的：

1. 判断缓存是否经常把零样本正确预测改错。
2. 判断缓存是否能救回零样本错误预测。
3. 判断下一步 B3 应该做归一化融合、门控融合，还是转向分布得分。

---

## 6. 范数与分类间隔

记录每个分支的 logits 范数（logits norm）和分类间隔（margin）：

```text
S_zs norm / margin
S_H norm / margin
S_GPA norm / margin
S_L norm / margin
S_N norm / margin
S_H + S_L norm / margin
S_H + S_L - S_N norm / margin
S_final norm / margin
```

这些统计用于判断：

1. 某个缓存分支是否过强。
2. 缓存分支在和零样本冲突时是否可靠。
3. 不同分支的数值尺度是否可以直接相加。

---

## 7. 输出文件

模型实现：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/model_e7_b3_diag_0292_textdist_cache_branch_diagnostics.py
```

扰动数据集 runner：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/run_e7_b3_diag_0292_ulip_modelnetc_s2_cache_branch_diagnostics.py
```

clean runner：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/run_e7_b3_diag_0292_ulip_modelnetc_clean_cache_branch_diagnostics.py
```

扰动数据集脚本：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_11_ulip_modelnetc_s2_zs_global_local_e7_b3_diag_0292_cache_branch_diagnostics.sh
```

clean 脚本：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_12_clean_ulip_modelnetc_clean_zs_global_local_e7_b3_diag_0292_cache_branch_diagnostics.sh
```

统计文件：

```text
<cor_type>_e7_b3_diag_0292_stats.json
<cor_type>_e7_b3_diag_0292_samples.jsonl
```

---

## 8. 运行命令

从 `Point-Cache` 目录运行：

```bash
bash scripts/E7_entropy_energy_alignment_multicache/00_11_ulip_modelnetc_s2_zs_global_local_e7_b3_diag_0292_cache_branch_diagnostics.sh 0
```

clean 版本：

```bash
bash scripts/E7_entropy_energy_alignment_multicache/00_12_clean_ulip_modelnetc_clean_zs_global_local_e7_b3_diag_0292_cache_branch_diagnostics.sh 0
```

默认保存逐样本轻量诊断：

```text
E7_B3_DIAG_SAVE_SAMPLES=1
E7_B3_DIAG_SAVE_RAW_LOGITS=0
```

---

## 9. 后续判断

如果 `norm_fusion_offline` 明显优于 `final_logits`，B3 优先做归一化融合。

如果 `cache_total_signed` 单独准确率低，但与 `S_zs` 一致时最终预测明显更可靠，B3 优先做门控融合。

如果 `S_H`、`S_L`、`S_N` 的尺度差异明显，B3 不应继续直接相加，而应先做统一尺度的分支融合。

---

## 10. S2 结果分析

日期：2026-06-16

结果目录：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_11_ulip_modelnetc_s2_zs_global_local_e7_b3_diag_0292_cache_branch_diagnostics/
```

完整性：

| 项目 | 数值 |
|---|---:|
| S2 扰动类型数 | 7/7 |
| 总样本数 | 17276 |
| 状态 | 全部完成 |

### 10.1 正式准确率复现情况

| 方法 | S2 平均 |
|---|---:|
| 原始 Point-Cache global+local baseline | 54.00%（约 9329/17276） |
| 02_9_2 原始结果 | 54.71%（约 9451/17276） |
| B3-Diag-02_9_2 正式最终得分 | 54.69%（9449/17276） |

结论：B3-Diag-02_9_2 基本复现 02_9_2，诊断逻辑没有改变正式预测。

### 10.2 分支单独预测结果

| 分支 | S2 汇总 |
|---|---:|
| 零样本文本原型点积得分（zero-shot text prototype dot-product logits） | 47.55%（8215/17276） |
| 全局熵缓存投票得分（global entropy cache voting logits） | 47.51%（8207/17276） |
| GPA 全局缓存诊断得分（GPA global cache diagnostic voting logits） | 49.02%（8469/17276） |
| GPA 控制的局部缓存投票得分（GPA-controlled local cache voting logits） | 46.96%（8113/17276） |
| 正缓存总得分（positive cache total logits） | 49.12%（8486/17276） |
| 带符号缓存总得分（signed cache total logits） | 49.47%（8546/17276） |
| 最终融合得分（final logits） | 54.69%（9449/17276） |
| 离线归一化融合诊断得分（offline normalized fusion diagnostic logits） | 53.51%（9245/17276） |

结论：

1. 缓存分支单独预测不如最终融合得分。
2. GPA 全局缓存诊断得分是最强的单个缓存分支，但它在 02_9_2 中只用于诊断，正式公式没有直接加入它。
3. 离线归一化融合低于正式最终得分，不支持下一步直接做简单样本内 z-score 融合。

### 10.3 缓存帮助与破坏

| 指标 | 数值 |
|---|---:|
| `S_zs` 正确但 `S_final` 错误 | 605 |
| `S_zs` 错误但 `S_final` 正确 | 1839 |
| 净收益 | +1234 |

这里的净收益正好对应：

```text
9449 - 8215 = 1234
```

说明最终融合相对零样本文本原型点积得分的提升，主要来自缓存救回零样本错误预测。

### 10.4 一致与冲突分组

| 分组 | 样本数 | 零样本正确率 | 缓存正确率 | 最终正确率 |
|---|---:|---:|---:|---:|
| `argmax(S_zs) == argmax(S_cache)` | 9231 | 69.00%（6369/9231） | 69.00%（6369/9231） | 69.00%（6369/9231） |
| `argmax(S_zs) != argmax(S_cache)` | 8045 | 22.95%（1846/8045） | 27.06%（2177/8045） | 38.28%（3080/8045） |

结论：

1. 一致组天然可靠。
2. 冲突组并不应该简单回退到零样本，因为最终融合在冲突组里明显好于零样本和缓存单独预测。
3. 这否定了“冲突就回退到 zero-shot”的简单门控方案。

### 10.5 下一步方向

优先方向：

1. 不做简单归一化融合，因为 `norm_fusion_offline` 低于正式 `final_logits`。
2. 不做简单冲突回退，因为冲突组中 `S_final` 仍明显优于 `S_zs`。
3. 重点考虑把 GPA 全局缓存诊断得分 `S_GPA` 作为一个小权重分支加入正式公式，但需要先做权重扫描，避免和全局熵缓存重复计分。

建议下一步实验：

```text
B3：02_9_2 + GPA 全局缓存小权重分支
S_final = S_zs + S_H + S_L - S_N + lambda * S_GPA
```

第一版建议只做小范围权重：

```text
lambda in {0.1, 0.2, 0.3}
```

更稳妥的做法是先重新运行一次 B3-Diag-02_9_2，并设置：

```bash
E7_B3_DIAG_SAVE_RAW_LOGITS=1
```

这样可以离线扫描 `lambda`，不需要反复跑点云编码器。
