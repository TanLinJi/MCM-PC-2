# E4 到 E5 研究叙事增量归档

更新时间：2026-06-10  
项目根目录：`/root/autodl-tmp/MCM-PC-2`  
核心代码库：`/root/autodl-tmp/MCM-PC-2/Point-Cache`

---

## 1. 当前定位

这份文档补充 `e0_e3_research_narrative.md` 之后的研究进展，重点记录 E4 和 E5 的阶段性判断。

E3 的核心结论是：单中心 GPA Cache 对添加型噪声有一定帮助，但对几何变化和多形态类别不稳定。因此 E4 转向“类别分布引导的缓存替换”，E5 则尝试吸收 ADAPT/PGA 的“文本先验 + 高置信 test-time bank + Gaussian alignment”思想。

---

## 2. E4 当前关键结论

E4-A/B/C 的探索说明：

1. 只用当前 cache snapshot 的视觉分布过窄，`shot_capacity=3` 无法表达足够多样性。
2. accepted-history visual distribution 比 current-cache-only 更合理，因为它保留了历史上被 cache 准入规则认可过的多样样本。
3. 文本分布直接强行支配最终分类会导致下降；更稳的做法是让 E1 prompt 信息只进入 text distribution。
4. 当前最强 severity=2 结果是 `02_9_2`：

```text
E4-C-A0+E1-textdist-only
E4_TEXT_SCORE_WEIGHT=0.15
ModelNet-C severity=2 七类 corruption 平均准确率约 54.71
```

但 `02_9_2` 在 clean 上为 `63.86`，低于原始 Point-Cache clean `64.18`。这说明后续方法必须同时关注 clean 和 corrupted，不能只优化 corruption 平均。

---

## 3. E5-A 阶段判断

E5-A 参考 ADAPT/PGA，引入独立 StatsBank、delayed update、shared covariance 和 standalone GDA diagnostics。

已完成结果：

| corruption | 原始 Point-Cache/E4 分支 | standalone GDA | 结论 |
|---|---:|---:|---|
| add_global_2 | 47.89 | 47.40 | 负向 |
| add_local_2 | 50.85 | 49.39 | 负向 |

阶段判断：

```text
E5-A 作为 standalone GDA 方向失败。
失败点不是 shared covariance 本身，而是把 GDA 当成独立分类器或最终强专家的迁移方式不适合当前 Point-Cache/E4-C 框架。
```

原因包括：

1. Point-Cache/E4 已经有强 final logits，GDA 证据和 cache 证据高度相关，直接叠加容易重复计数。
2. StatsBank 使用 pseudo-label，早期错误会污染 Gaussian 均值和协方差。
3. shared covariance 更适合作为稳定度量，不等于 standalone GDA 分类器一定更强。

---

## 4. E5 后续修正路线

E5 后续主线调整为：

```text
E5-B：text-prior posterior prototype residual
E5-C：shared covariance 只作为 cache replacement metric
E5-D：基于有效样本数的动态 text-visual fusion weight
```

详细设计文档：

```text
docs/experiments/E5_adapt_inspired_gaussian_alignment_cache/E5_BCD_posterior_prototype_residual_design.md
```

核心思想是：不要再把 GDA 当成新分类器，而是把 ADAPT/PGA 中最有价值的部分迁移为“文本先验锚定的 posterior prototype 修正”。

下一步优先级：

1. E5-B0：只构建 posterior prototype 和 residual diagnostics，不改最终预测。
2. E5-B1：在 `02_9_2` 基础上加入 posterior residual final logits，并同时输出 original 与多个 gamma 结果。
3. E5-C1：如果 E5-B 有信号，再用 shared covariance Mahalanobis metric 改 cache replacement。
4. E5-D1：最后把固定 text weight `0.15` 改为有效样本数驱动的动态融合权重。

截至 2026-06-10，E5-B0/B1 已新增独立实现：

```text
Point-Cache/runners/E5_adapt_inspired_gaussian_alignment_cache/model_e5_b0_b1_posterior_prototype_residual.py
Point-Cache/runners/E5_adapt_inspired_gaussian_alignment_cache/run_e5_b0_b1_ulip_modelnetc_s2_posterior_prototype_residual.py
Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache/01_run_e5_b0_b1_ulip_modelnetc_s2_common.sh
Point-Cache/scripts/E5_adapt_inspired_gaussian_alignment_cache/01_1_ulip_modelnetc_s2_zs_global_local_e5_b0_b1_posterior_prototype_residual_manual_full_clip_manualfull_llm_dynamic_init_textdist.sh
```

该实现保持 `02_9_2` 的 E4-C-A0+E1-textdist-only 设置不变，新增独立 delayed-update StatsBank，并在同一次实验中输出 original 与多个 posterior residual gamma 的准确率。

---

## 5. 当前可信边界

当前 E4/E5 结论仍主要来自 ULIP + ModelNet-C severity=2。后续论文级结论必须补：

```text
ModelNet-C all severities / all corruptions
ModelNet-C clean
ScanObjectNN-C
ShapeNet-C baseline 复现后再迁移
更多 backbone
不同测试流顺序
运行成本和统计诊断
```

在这些补完前，`02_9_2` 可以作为当前最佳工程基线，但不能直接作为最终论文完整 benchmark。
