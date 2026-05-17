# Experiments

按 stage / 实验编号组织，所有人类可读的实验设计、说明、结果总结都在这里。

> **原始数据放哪？** `.log`、`.json`、`.csv` 等运行日志保留在 `/root/autodl-tmp/MCM-PC-2/Point-Cache/logs/recur-pc/<run_id>/`，不复制到本目录。本目录只放说明文档和已经归纳过的 summary。

---

## 子目录索引

### `stages/` — 阶段总结

线性时间线，每个 stage 一个 .md。读这个能快速掌握「我们做过什么 + 为什么」。

| 文件 | 阶段 | 主题 |
|---|---|---|
| `stage0_tpe_zero_shot_text_methods_summary.md` | Stage 0 | E0-TPE 4 种 zero-shot 文本方法对比 |
| `stage0_tpe_spherical_text_anchor_corrected.md` | Stage 0 | 球面文本锚点（vMF）修订版 |
| `stage1_baseline_repro.md` | Stage 1 | Point-Cache baseline 复现（ULIP-2） |
| `stage2_emr.md` | Stage 2 | E2-EMR Entropy-Margin Reliability |
| `stage3_glc.md` | Stage 3 | E3-GLC Global-Local Consistency |
| `stage4_canc_diag.md` | Stage 4 | E4-CANC 诊断 + v0/v1 |

### `e0_tpe/` — Text Prototype Enhancement（零样本阶段）

| 文件 | 用途 |
|---|---|
| `text_prototype_enhancement.md` | 实验设计与基本结论 |
| `spherical_text_anchor_report.html` | 球面文本锚点详细报告（41 KB） |

### `e2_emr/` — Entropy-Margin Reliability cache admission

| 文件 | 用途 |
|---|---|
| `e2_emr_admission.md` | 实验说明 + Reliability Score 公式 |
| `e2_emr_admission.html` | 同上，HTML 版（含 8 节解析、Δ 可视化） |

### `e3_glc/` — Global-Local Consistency reliability

| 文件 | 用途 |
|---|---|
| `e3_glc_consistency.md` | E3-GLC 实验设计与失败复盘 |
| `e3_glc_v1_math_derivation.html` | E3-GLC-v1 实验计划与数学推导 |

### `e4_canc/` — Confusion / Conflict-Aware Negative Cache

| 文件 | 用途 |
|---|---|
| `e4_canc_overview.md` | E4-CANC 总览（核心想法 + 与 E3 的关系） |
| `e4_canc_v0_conservative.md` | E4-CANC-v0 Conservative 版本 |
| `e4_canc_v0_rule_explanation.html` | v0 规则解释与阶段分析 |
| `e4_canc_diag_math_derivation.html` | E4-CANC-DIAG 数学推导（修订版） |

### `pointcache_repro/` — Point-Cache 复现资料

| 文件 | 用途 |
|---|---|
| `commands.md` | ZS / Global Cache / Hierarchical 三套复现命令 |
| `reproduction_notes.md` | Point-Cache Fig 1a 复现说明，含 corruption 类型表 |
| `project_structure.md` | Point-Cache 论文章节 ↔ 代码模块映射 |

---

## 顶层文件

| 文件 | 用途 |
|---|---|
| `experiment_summary.md` | 跨阶段实验汇总（baseline / E2 / E3 / E4 数字一览） |
| `repro_log.md` | 复现操作日志 |

---

## 推荐阅读顺序

按时间线和实验编号顺序读最直观：

1. `pointcache_repro/project_structure.md` — 先理解 Point-Cache 代码骨架
2. `pointcache_repro/commands.md` — 看复现命令
3. `stages/stage1_baseline_repro.md` — Stage 1 复现
4. `stages/stage2_emr.md` + `e2_emr/e2_emr_admission.md`
5. `stages/stage3_glc.md` + `e3_glc/e3_glc_consistency.md`
6. `stages/stage4_canc_diag.md` + `e4_canc/e4_canc_overview.md` + `e4_canc/e4_canc_v0_conservative.md`
7. `stages/stage0_tpe_*` + `e0_tpe/*` — Stage 0 文本端探索
8. `experiment_summary.md` — 数字汇总

## 新增实验怎么记录

1. **新建 stage**：`stages/stage<N>_<topic>.md`，遵循模板
2. **新建实验子目录**（如果实验独立）：`e<N>_<short_name>/<short_name>.md`
3. **runner 跑出新数字**：更新 `experiment_summary.md` 的对应行；更新 `project/progress_log.md`
4. **如果是 quick test 失败**：记录在对应 stage 的 .md 末尾「失败复盘」段，不另开文件

## 命名规范

- 实验编号：`e<n>_<short_name>`（小写下划线，如 `e2_emr`、`e4_canc`）
- 实验子文件名：`<eN>_<aspect>.md/.html`（如 `e4_canc_v0_conservative.md`）
- HTML 报告：日期前缀放 `docs/reports/` 而非这里；本目录的 .html 是与对应 .md 等价的可视化版本
