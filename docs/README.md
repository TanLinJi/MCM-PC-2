# MCM-PC-2 Documentation

> **项目**：MCM-PC: Reliability-Aware Multi-Cache Matrix for Test-Time Adaptation of 3D Point Cloud Vision-Language Models
>
> **路径**：`/root/autodl-tmp/MCM-PC-2/docs/`

`docs/` 是本项目所有人类可读文档的统一入口。代码、原始日志、模型权重、数据集留在 `Point-Cache/` 等代码目录，不放这里。

---

## 目录结构

```
docs/
├── README.md                      ← 本文件，目录索引
├── project/                       ← 项目管理：偏好规则、术语、进度
├── proposals/                     ← 创新点、研究方案、路线规划
├── paper/                         ← 论文章节草稿
├── experiments/                   ← 实验设计、结果、阶段总结
├── reports/                       ← 自包含 HTML 报告
├── context/windsurf/              ← AI 对话归档、上下文恢复
├── archive/                       ← 旧版本方案、参考代码副本
└── assets/figures/                ← 图、流程图、图源文件
```

---

## 一、`project/` — 项目管理

| 文件 | 用途 |
|---|---|
| `user_preferences.md` | 用户工作偏好与项目规则（commit 规则、命名规则、跑分规则） |
| `glossary.md` | 术语表（D/P/F/G/W 编号、anchor pollution、EMR/GLC/CANC 等中英文对照） |
| `progress_log.md` | 关键里程碑数字日志 |

## 二、`proposals/` — 创新点和方案

| 文件 | 用途 |
|---|---|
| `core_innovations.md` | 4 个核心创新点（Multi-Cache Matrix / Compactness-Margin / Global-Local Consistency / Adaptive Fusion） |
| `auxiliary_innovation_3.md` | 辅助创新 3：Confusion-Aware Negative Cache 详细推导 |
| `matrix_idea_v0.md` | 3×2 缓存矩阵最初想法草稿（Entropy/Align/Negative × Global/Local） |
| `ideas_log.md` | 想法零碎记录 |
| `top_conference_proposal_v1.html` | 顶会论文方案 v1（41 KB，含 14-18 周路线） |
| `project_roadmap_v1.html` | 项目 roadmap v1（25 KB，7 节）|

## 三、`paper/` — 论文章节草稿

| 文件 | 章节 | 状态 |
|---|---|---|
| `0_outline.md` | 大纲、章节依赖、写作时间预算 | v0.1 |
| `abstract.md` | 摘要 + 流程图骨架 | 占位 |
| `1_introduction.md` | 引言 | 草稿 |
| `2_related_work.md` | 相关工作（3D TTA / 零样本 / 缓存）| v0.1 草稿 |
| `3_method.md` | 方法 | 占位 |

## 四、`experiments/` — 实验

详见 `experiments/README.md`。粗略：

| 子目录/文件 | 内容 |
|---|---|
| `stages/` | stage0~4 实验阶段 .md 归档（按 baseline → EMR → GLC → CANC 顺序） |
| `e0_tpe/` | 文本原型增强（zero-shot 阶段） |
| `e2_emr/` | Entropy-Margin Reliability cache admission |
| `e3_glc/` | Global-Local Consistency reliability |
| `e4_canc/` | Confusion-Aware / Conflict-Aware Negative Cache（含 v0 / v1 / DIAG） |
| `pointcache_repro/` | Point-Cache 复现说明、命令、项目结构 |
| `experiment_summary.md` | 跨阶段实验总结 |
| `repro_log.md` | 复现日志 |

## 五、`reports/` — HTML 报告

| 文件 | 类型 |
|---|---|
| `2026-05-15_top_conference_progress.html` | 顶会论文阶段进展报告（浅色主题） |
| `2026-05-17_global_roadmap_v2.html` | 全局路线图 v2（39 KB，最详细） |
| `2026-05-17_task_specification.html` | 任务说明书 v1.0（暗黑主题，含 6 周路线 / 风险 / 立即行动） |

## 六、`context/windsurf/`

`conversation_archive.md/.html` — Cascade 对话归档，恢复上下文用。

## 七、`archive/`

| 文件 | 用途 |
|---|---|
| `README.md` | MCM-PC 项目原 docs 目录设计规范（参考） |
| `mcp3d_framework_proposal.md` | MCP-3D 论文方案（旧版，MCM-PC 项目分支） |
| `reference_code/model_with_hierarchical_caches.py` | Point-Cache 原版 hierarchical cache runner 副本（参考用） |

## 八、`assets/figures/`

| 文件 | 引用位置 |
|---|---|
| `confusion_aware_negative_cache.png` | `proposals/core_innovations.md` 创新点示意 |
| `e2_emr_delta_chart.png` | `experiments/e2_emr/e2_emr_admission.md` Δ 可视化 |

---

## 命名规范

- **目录名**：英文小写 + 下划线
- **MD 文件名**：英文小写 + 下划线（如 `auxiliary_innovation_3.md`）
- **HTML 报告**：日期前缀 + 描述（如 `2026-05-17_task_specification.html`）
- **方案 HTML**：版本后缀（如 `top_conference_proposal_v1.html`）
- **实验编号**：`e<n>_<short_name>`（如 `e2_emr`、`e4_canc`）
- **stage 编号**：`stage<n>_<topic>.md`（如 `stage4_canc_diag.md`）

## 推荐阅读顺序

新读者按下面顺序读，30 分钟摸清项目：

1. `project/user_preferences.md` — 工作规则
2. `project/glossary.md` — 术语和实验编号
3. `reports/2026-05-17_task_specification.html` — 当前局势 + 6 周路线
4. `proposals/core_innovations.md` — 核心创新点
5. `experiments/stages/` 全部（按编号顺序）
6. `reports/2026-05-17_global_roadmap_v2.html` — 全局路线图
7. `paper/0_outline.md` — 论文大纲和写作进度

## 新增文档怎么放

- 新论文章节 / 摘要 / 段落：`paper/`
- 新实验数据汇总（不是原始日志）：`experiments/<eN_topic>/`；原始 .log/.csv/.json 留在 `Point-Cache/logs/`
- 新创新点或方法草稿：`proposals/`
- 新 HTML 自包含报告：`reports/`，加日期前缀
- 新对话归档：`context/windsurf/`
- 新图源（svg / png / mmd）：`assets/figures/`
- 旧版方案、参考代码：`archive/`

## 维护规则

- 文件命名变更必须用 `git mv` 保留 history
- 重复或失效文件移到 `archive/` 而非直接删除（除非确认是空文件或日志副本）
- 每次结构调整在本文件 changelog 末尾追加一行

---

## Changelog

- **2026-05-17** v1.0：初始结构落盘。归档了 `docs/new/` 中所有用户上传的资料，按 8 大类目录重新分类，统一英文命名。同时合并 E2-EMR 三份重复、两份代号说明，删除日志副本（原版在 `Point-Cache/logs/recur-pc/`）。
