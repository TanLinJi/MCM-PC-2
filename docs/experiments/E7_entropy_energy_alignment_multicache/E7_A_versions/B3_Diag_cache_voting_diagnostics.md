# E7-B3-Diag：缓存投票得分诊断

日期：2026-06-15
状态：已实现，待运行与分析

---

## 0. 简要总结

B3-Diag 是 B3 之前的诊断实验，只记录逐样本得分关系，不改变最终预测结果。

本实验要回答一个核心问题：

```text
当前缓存投票得分（cache voting score）到底是在帮助 zero-shot，还是在破坏 zero-shot？
```

B3-Diag 不做以下改动：

1. 不去掉对齐核心缓存（alignment core cache）。
2. 不改变候选池（candidate pool）更新规则。
3. 不改变熵缓存（entropy cache）和能量缓存（energy cache）更新规则。
4. 不引入文本分布得分（text distribution score）。
5. 不改变最终预测公式。

它只在现有预测流程中额外记录诊断量，为后续 B3 的“缓存投票融合方式改进”提供证据。

---

## 1. 命名说明

本实验编号为：

```text
E7-B3-Diag
```

后续编号规则：

| 编号 | 含义 |
|---|---|
| B3-Diag | 只诊断缓存投票，不改预测 |
| B3 | 只改缓存投票融合方式，不改缓存结构 |
| B4 | 再研究去掉对齐核心缓存 |

不再使用 `B3-1`、`B3-2` 这类编号。

---

## 2. 实验背景

B2 结果显示，单纯改变候选池 top1 晋升路径没有带来收益。

| 方法 | S2 平均准确率 |
|---|---:|
| 02_9_2 | 54.71%（9451/17276） |
| A4 fixed | 53.18%（9188/17276） |
| B1 | 52.87%（9133/17276） |
| B2 | 52.51%（9071/17276） |

B2 还显示：

| 指标 | A4 fixed | B2 |
|---|---:|---:|
| 候选池进入样本伪标签正确率 | 61.45%（4233/6888） | 59.69%（5116/8571） |
| 对齐核心缓存进入样本伪标签正确率 | 62.98%（769/1221） | 65.88%（751/1140） |
| 测试阶段当前样本进入真正缓存比例 | 0.89%（154/17276） | 0.60%（104/17276） |

这些结果说明：

1. B2 的 top1 晋升样本更可靠，但样本数减少。
2. 更可靠的对齐核心缓存没有转化成更高最终准确率。
3. 需要直接诊断缓存投票得分，而不是继续盲目调整候选池或缓存容量。

---

## 3. 载体选择

B3-Diag 的主载体使用：

```text
E7-A4 fixed
```

不使用 B2 作为主载体。

原因：

```text
B2 已经改变了候选池到对齐核心缓存的晋升路径。
如果在 B2 上诊断缓存投票，会把“缓存内容变化”和“缓存投票融合问题”混在一起。
```

因此，B3-Diag 应该基于 A4 fixed：

```text
候选池 -> 对齐核心缓存 -> 熵缓存 / 能量缓存
```

这样后续 B3 才能只改变融合方式，而不改变缓存结构。

---

## 4. 严格 TTA 边界

B3-Diag 仍然是免训练测试时适应（training-free Test-Time Adaptation, training-free TTA）：

1. 不更新点云编码器（point encoder）。
2. 不更新文本编码器（text encoder）。
3. 不更新文本原型（text prototypes）。
4. 不使用真实标签参与任何测试时适应决策。
5. 不反向传播（backpropagation-free）。
6. 真实标签只用于离线诊断统计。

---

## 5. 当前 A4 得分形式

A4 的最终得分为：

```text
S_final = S_zs + S_A + S_H + S_E
```

其中：

| 符号 | 中文含义 | 英文 |
|---|---|---|
| `S_zs` | 零样本得分 | zero-shot logits |
| `S_A` | 对齐核心缓存投票得分 | alignment core cache voting logits |
| `S_H` | 熵缓存投票得分 | entropy cache voting logits |
| `S_E` | 能量缓存投票得分 | energy cache voting logits |

定义缓存总投票得分：

```text
S_cache = S_A + S_H + S_E
```

B3-Diag 不改变这个公式，只额外记录 `S_zs`、`S_A`、`S_H`、`S_E`、`S_cache` 和 `S_final` 的关系。

---

## 6. 必须记录的逐样本诊断量

对每个测试样本，记录以下信息：

| 字段 | 中文含义 |
|---|---|
| `target` | 真实标签，只用于离线分析 |
| `pred_zs` | `argmax(S_zs)` |
| `pred_A` | `argmax(S_A)`，如果对齐核心缓存为空或全零则标记 invalid |
| `pred_H` | `argmax(S_H)`，如果熵缓存为空或全零则标记 invalid |
| `pred_E` | `argmax(S_E)`，如果能量缓存为空或全零则标记 invalid |
| `pred_cache` | `argmax(S_cache)`，如果缓存总得分全零则标记 invalid |
| `pred_final` | `argmax(S_final)` |
| `correct_zs` | `pred_zs == target` |
| `correct_A` | `pred_A == target` |
| `correct_H` | `pred_H == target` |
| `correct_E` | `pred_E == target` |
| `correct_cache` | `pred_cache == target` |
| `correct_final` | `pred_final == target` |
| `zs_cache_agree` | `pred_zs == pred_cache` |
| `zs_final_changed` | `pred_zs != pred_final` |
| `cache_norm` | `||S_cache||_2` |
| `zs_norm` | `||S_zs||_2` |
| `final_norm` | `||S_final||_2` |
| `zs_margin` | `S_zs` 的 top1-top2 分类间隔 |
| `cache_margin` | `S_cache` 的 top1-top2 分类间隔 |

注意：

```text
真实标签 target 只用于统计 correct_*，不能进入任何缓存更新或预测决策。
```

---

## 7. 必须汇总的诊断统计

### 7.1 单项准确率

每个 corruption 和 clean 都要记录：

```text
S_zs 准确率：xx.xx%（正确数/总数）
S_A 准确率：xx.xx%（正确数/有效样本数）
S_H 准确率：xx.xx%（正确数/有效样本数）
S_E 准确率：xx.xx%（正确数/有效样本数）
S_cache 准确率：xx.xx%（正确数/有效样本数）
S_final 准确率：xx.xx%（正确数/总数）
```

如果某个缓存得分全零，该样本不计入该缓存单项准确率分母。

### 7.2 缓存帮助与破坏

必须记录：

```text
zs_correct_final_wrong：S_zs 正确但 S_final 错误，x/y
zs_wrong_final_correct：S_zs 错误但 S_final 正确，x/y
zs_correct_cache_wrong：S_zs 正确但 S_cache 预测错误，x/y
zs_wrong_cache_correct：S_zs 错误但 S_cache 预测正确，x/y
```

这些统计用于判断缓存投票是净帮助还是净破坏。

### 7.3 一致性分组

按 `pred_zs` 和 `pred_cache` 是否一致分组：

```text
argmax(S_zs) == argmax(S_cache) 时：
    S_final 准确率：xx.xx%（正确数/总数）

argmax(S_zs) != argmax(S_cache) 时：
    S_final 准确率：xx.xx%（正确数/总数）
```

如果不一致组准确率明显低，说明后续 B3 应该做门控（gating）或降低缓存权重。

### 7.4 范数与 margin 诊断

记录：

```text
S_zs logits norm 均值/最大值
S_cache logits norm 均值/最大值
S_final logits norm 均值/最大值
zs_margin 均值/分位数
cache_margin 均值/分位数
```

重点看：

```text
S_cache 是否在某些 corruption 中过强；
S_cache margin 大但预测错误的样本是否很多；
S_zs margin 小时 S_cache 是否更容易帮忙。
```

---

## 8. 可选离线归一化诊断

B3-Diag 不改变预测，但可以离线计算以下候选融合结果：

```text
S_norm_final = norm(S_zs) + norm(S_cache)
```

这里的 `norm()` 只能用于离线诊断，不影响正式预测。

第一版建议使用样本内 z-score：

```text
norm(S) = (S - mean(S)) / (std(S) + eps)
```

需要记录：

```text
norm(S_zs) + norm(S_cache) 的离线准确率：xx.xx%（正确数/总数）
```

如果这个离线结果明显好于 A4，说明 B3 可以尝试只改融合方式。

---

## 9. 判断标准

B3-Diag 不以准确率提升为目标，因为它不改变正式预测。

它的成功标准是能回答以下问题：

1. `S_cache` 单独预测是否高于 `S_zs`。
2. `S_cache` 与 `S_zs` 一致时是否更可靠。
3. `S_cache` 与 `S_zs` 冲突时是否经常破坏正确预测。
4. `norm(S_zs) + norm(S_cache)` 的离线结果是否优于原始 `S_final`。
5. 对齐核心缓存、熵缓存、能量缓存中哪一项最常帮忙或最常破坏。

如果诊断显示：

```text
S_cache 经常把正确的 S_zs 改错；
且 norm(S_zs) + norm(S_cache) 离线更好；
```

则 B3 应优先尝试归一化融合。

如果诊断显示：

```text
只有 zs/cache 一致时结果可靠；
```

则 B3 应优先尝试门控融合。

如果诊断显示：

```text
S_cache 单项准确率很低；
且冲突时几乎总是错误；
```

则 B3 不应继续增强缓存投票，而应准备 B4/B5 的结构或分布得分路线。

---

## 10. 实验输出

结果目录建议：

```text
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_9_ulip_modelnetc_s2_zs_global_e7_b3_diag_cache_voting_diagnostics_a4fixed/
Point-Cache/results/E7_entropy_energy_alignment_multicache/00_10_clean_ulip_modelnetc_clean_zs_global_e7_b3_diag_cache_voting_diagnostics_a4fixed/
```

脚本命名建议：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_9_ulip_modelnetc_s2_zs_global_e7_b3_diag_cache_voting_diagnostics_a4fixed.sh
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_10_clean_ulip_modelnetc_clean_zs_global_e7_b3_diag_cache_voting_diagnostics_a4fixed.sh
```

统计文件：

```text
<cor_type>_e7_b3_diag_stats.json
<cor_type>_e7_b3_diag_samples.jsonl
```

例如扰动数据集会生成 `add_global_2_e7_b3_diag_stats.json`，clean 版本会生成 `clean_e7_b3_diag_stats.json`。

其中：

| 文件 | 内容 |
|---|---|
| `e7_b3_diag_stats.json` | 汇总统计 |
| `e7_b3_diag_samples.jsonl` | 逐样本诊断，可通过环境变量控制是否保存 |

默认可以保存汇总统计；逐样本 JSONL 如果文件过大，可以只在需要时开启。

### 10.1 已实现代码

模型实现：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/model_e7_b3_diag_cache_voting_diagnostics.py
```

扰动数据集 runner：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/run_e7_b3_diag_ulip_modelnetc_s2_cache_voting_diagnostics.py
```

干净数据集 runner：

```text
Point-Cache/runners/E7_entropy_energy_alignment_multicache/run_e7_b3_diag_ulip_modelnetc_clean_cache_voting_diagnostics.py
```

扰动数据集脚本：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_9_ulip_modelnetc_s2_zs_global_e7_b3_diag_cache_voting_diagnostics_a4fixed.sh
```

干净数据集脚本：

```text
Point-Cache/scripts/E7_entropy_energy_alignment_multicache/00_10_clean_ulip_modelnetc_clean_zs_global_e7_b3_diag_cache_voting_diagnostics_a4fixed.sh
```

### 10.2 运行命令

从 `Point-Cache` 目录运行：

```bash
bash scripts/E7_entropy_energy_alignment_multicache/00_9_ulip_modelnetc_s2_zs_global_e7_b3_diag_cache_voting_diagnostics_a4fixed.sh 0
```

clean 版本：

```bash
bash scripts/E7_entropy_energy_alignment_multicache/00_10_clean_ulip_modelnetc_clean_zs_global_e7_b3_diag_cache_voting_diagnostics_a4fixed.sh 0
```

默认保存轻量逐样本诊断：

```text
E7_B3_DIAG_SAVE_SAMPLES=1
E7_B3_DIAG_SAVE_RAW_LOGITS=0
```

如果需要保存完整 logits，可临时设置：

```bash
E7_B3_DIAG_SAVE_RAW_LOGITS=1 bash scripts/E7_entropy_energy_alignment_multicache/00_9_ulip_modelnetc_s2_zs_global_e7_b3_diag_cache_voting_diagnostics_a4fixed.sh 0
```

---

## 11. 与后续实验的关系

B3-Diag 完成后：

```text
B3：只改缓存投票融合方式，不改缓存结构。
B4：再研究去掉对齐核心缓存。
```

B3 不应引入文本分布得分，也不应去掉对齐核心缓存。

B4 才讨论：

```text
去掉对齐核心缓存；
候选池不删除；
候选池维护分布；
熵缓存和能量缓存从候选池复制样本。
```

这样可以保证每个实验只改变一个核心变量。
