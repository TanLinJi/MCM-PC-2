# MCM-PC 总实验登记表

最后更新：2026-06-03

本文档是 MCM-PC 项目的正式实验总登记表，用于统一记录 E0、E1、E2 等实验的编号、名称、目标、状态、文档路径、脚本路径、结果路径和下一步工作。

本文档不替代每个实验自己的日志和分析文档。每个正式实验仍应维护独立的：

- log.md：实验过程日志；
- analysis.md：实验结果分析。

旧探索实验已经归档到：

    docs/experiments/archive/legacy_pre_mcmpc_restart/

旧实验不占用新的 E0-E5 编号。

## 1. 正式实验序列

| 编号 | 实验名称 | 目标 | 当前状态 |
|---|---|---|---|
| E0 | Point-Cache baseline 复现与分析 | 复现 Point-Cache baseline，并分析 baseline 行为 | 复现与结果分析已完成 |
| E1 | 文本原型增强 | 研究点云语义模板与 DeepSeek 动态生成提示词对文本原型的影响 | 文档初始化完成，代码未开始 |
| E2 | 局部缓存可靠性门控 | 根据可靠性动态控制 local cache 贡献 | 计划中 |
| E3 | 冲突感知负向抑制 | 将 global-local 冲突作为保守负向证据 | 计划中 |
| E4 | 可靠性感知多缓存矩阵 | 构建完整 MCM-PC 方法 | 计划中 |
| E5 | 消融实验与可视化 | 完成消融、案例分析和论文图表 | 计划中 |

## 2. E0：Point-Cache baseline 复现与分析

### 目标

复现 Point-Cache baseline 结果，并将其作为后续 MCM-PC 实验的比较基础。

E0 包括 baseline 复现和 baseline 结果分析两部分。

### 实验范围

Backbone：

- ULIP
- ULIP-2
- OpenShape
- Uni3D

数据集与设置：

- ModelNet clean
- ModelNet-C all35
- ScanObjNN clean hardest
- ScanObjNN-C hardest all35

方法：

- zero-shot
- zero-shot + global cache
- zero-shot + global cache + local cache

### 当前状态

baseline 复现已经完成。

baseline 结果分析已经完成，相关结果文档已整理在 `docs/experiments/E0_baseline/`。

当前高层观察：

- global cache 是主要且相对稳定的增益来源；
- local cache 提供辅助增益，但并不总是稳定；
- 一些 backbone 和损坏类型下，local cache 的额外贡献较弱，甚至可能为负；
- 这些观察为后续可靠性感知 cache 设计提供动机。

### 文档路径

已有资料：

- docs/experiments/E0_baseline/
- docs/experiments/pointcache_repro/
- docs/experiments/repro_log.md

计划整理为：

- docs/experiments/E0_baseline/log.md
- docs/experiments/E0_baseline/analysis.md

### 结果路径

- Point-Cache/results/baseline/

### 下一步

维护 E0 baseline 结果文档，并作为后续 E1-E5 实验的对照基准。

## 3. E1：文本原型增强

### 目标

研究 Point-Cache 的文本原型是否可以通过点云语义相关模板和 DeepSeek 动态生成的类别级描述得到增强。

### 动机

Point-Cache 的视觉侧可以通过 global cache 和 local cache 动态适应测试数据流，但文本侧主要依赖固定模板。

原始完整手工模板集合中包含大量 2D 图像风格模板，例如 photo、blurry photo、painting、cropped photo 等。这些模板对 3D 点云识别未必最合适。

E1 因此先研究文本端增强，再进入后续 cache 机制修改。

### 术语说明

| 名称 | 中文含义 | 说明 |
|---|---|---|
| manual_full | 原始完整手工模板集合 | Point-Cache baseline 默认使用的完整固定模板集合，作为 E0 兼容对照 |
| manual_3d | 点云/3D 相关手工模板子集 | 从 manual_full 中筛选出的点云语义相关模板集合 |
| deepseek_static | DeepSeek 离线固定描述集合 | 提前生成并保存为 JSON，实验时只读取 |
| deepseek_dynamic_init | DeepSeek 实验初始化动态描述集合 | 实验开始时根据候选类别名称生成，生成后冻结 |
| manual3d_deepseek_dynamic_init | 点云手工模板与 DeepSeek 动态描述融合 | E1 的主要候选方法 |

### 动态提示词规则

E1 使用 dynamic-init，不使用 dynamic-online。

允许：

- 在测试流开始前，根据数据集候选类别名称生成提示词；
- 保存生成结果；
- 使用冻结后的文本原型进行推理。

不允许：

- 对每个测试样本调用 LLM；
- 使用测试样本真实标签；
- 将测试点云内容暴露给 LLM；
- 根据单个样本预测结果生成提示词；
- 在测试流中持续更新提示词。

### 主要候选方法

E1 的主要候选方法是：

    manual3d_deepseek_dynamic_init

推荐形式是分支级加权融合：

    text_prototype =
        static_weight * mean(manual_3d embeddings)
        + dynamic_weight * mean(dynamic DeepSeek embeddings)

默认设置：

- static branch weight：0.75
- dynamic branch weight：0.25
- dynamic prompt count：25 prompts per class

### 计划阶段

阶段 1：zero-shot 文本原型对比。

比较：

- manual_full
- manual_3d
- deepseek_dynamic_init
- manual3d_deepseek_dynamic_init

阶段 2：Point-Cache 文本原型对比。

如果阶段 1 出现有意义趋势，再扩展到：

- zero-shot
- zero-shot + global cache
- zero-shot + global cache + local cache

### 第一批优先实验设置

- ULIP × ModelNet-C all35
- ULIP-2 × ModelNet-C all35
- Uni3D × ScanObjNN-C hardest all35

其中 all35 表示 7 类基础损坏类型 × 5 个损坏强度。

### 文档路径

- docs/decisions/D002_prompt_source_policy.md
- docs/experiments/E1_text_prototype_enhancement/log.md
- docs/experiments/E1_text_prototype_enhancement/analysis.md

### 脚本路径

- Point-Cache/scripts/E1_text_prototype_enhancement/

### 结果路径

- Point-Cache/results/E1_text_prototype_enhancement/

### 当前状态

E1 决策文档与中文实验文档已经初始化。

尚未修改代码。

### 下一步

在进入代码修改前，检查并更新 .gitignore，确保 DeepSeek API key 本地文件不会被提交。

## 4. E2：局部缓存可靠性门控

### 目标

根据可靠性动态控制 local cache 的贡献。

### 动机

E0 表明 local cache 在部分设置中有用，但在部分设置中贡献较弱或不稳定。因此 local cache 不应始终以固定权重融合。

### 初始想法

将固定 local cache 融合改为样本级可靠性权重：

    final = zs + alpha_g * global + r_l(x) * alpha_l * local

其中 r_l(x) 可以由 agreement、margin、entropy、prototype distance 等可靠性信号决定。

### 状态

计划中。

### 下一步

在 E1 文本端实验完成或边界清晰后再启动。

## 5. E3：冲突感知负向抑制

### 目标

将 global-local conflict 作为不可靠正向证据的信号，对可疑类别进行保守抑制。

### 动机

早期探索实验表明，global-local conflict 可能提示错误，但 local top-1 不应被直接当作修正后的伪标签。

### 初始想法

conflict 应作为负向证据，而不是直接的正向标签修正。

### 状态

计划中。

## 6. E4：可靠性感知多缓存矩阵

### 目标

构建完整 MCM-PC 框架，动态校准多个证据源。

### 候选证据源

- text prototype branch
- zero-shot logits
- global cache
- local cache
- conflict 或 negative suppression branch

### 状态

计划中。

## 7. E5：消融实验与可视化

### 目标

准备最终消融实验、案例分析和论文图表。

### 候选分析

- prompt source 消融；
- static vs dynamic prompt 消融；
- local reliability 消融；
- conflict suppression 消融；
- full matrix fusion 消融；
- 按损坏类型分析；
- cache reliability 可视化；
- failure case 可视化。

### 状态

计划中。

## 8. 相关已有文档

以下文档属于项目级资料，不由本文档替代。

项目规则与状态：

- docs/project/user_preferences.md
- docs/project/progress_log.md
- docs/project/glossary.md
- docs/project/project_tree.md

早期方案与想法：

- docs/proposals/core_innovations.md
- docs/proposals/ideas_log.md
- docs/proposals/matrix_idea_v0.md
- docs/proposals/auxiliary_innovation_3.md

旧论文笔记：

- docs/paper/0_outline.md
- docs/paper/abstract.md
- docs/paper/1_introduction.md
- docs/paper/2_related_work.md
- docs/paper/3_method.md

归档旧探索实验：

- docs/experiments/archive/legacy_pre_mcmpc_restart/

## 9. 论文草稿关联

完整 ICASSP 论文草稿维护在：

    paper/ICASSP/

每个实验应逐步贡献到论文草稿：

- motivation；
- method design；
- experiment tables；
- ablations；
- visualizations；
- limitations；
- references。

论文不应等所有实验完成后才开始写。
