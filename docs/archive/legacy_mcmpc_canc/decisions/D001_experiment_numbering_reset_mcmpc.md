# D001: 实验编号重置与文档规则

## 日期

2026-06-02

## 背景

项目之前包含若干探索性实验，包括早期文本 prompt 增强、entropy/margin 可靠性、全局-局部冲突分析，以及负缓存尝试。这些实验有助于理解失败模式，但它们不再属于正式的 MCM-PC 实验序列。

当前项目已经重新围绕论文方向组织：

**MCM-PC: Reliability-Aware Multi-Cache Matrix for Test-Time Adaptation of 3D Point Cloud Vision-Language Models**

为避免历史探索性实验与新的正式实验序列混淆，实验编号被重置。

## 决策

正式实验序列定义如下：

- **E0**：Point-Cache baseline 的复现与基线结果分析。
- **E1**：面向点云感知与动态生成 prompt 的文本原型增强。
- **E2**：可靠性门控的局部缓存。
- **E3**：冲突感知的负向抑制。
- **E4**：可靠性感知的多缓存矩阵。
- **E5**：消融实验与可视化。

历史探索性实验会被归档，不占用新的实验编号。

## 实验编号规则

1. E0 仅指 Point-Cache baseline 的复现与基线结果分析。
2. 新的 MCM-PC 方法实验从 E1 开始。
3. 归档的历史实验不得影响新的实验编号。
4. 新的实验目录、脚本和文档应遵循新的编号方案。
5. 基线结果必须保持可复现，且不应被 E1 或更晚的实验覆盖。

## 文档规则

每个正式实验都必须包含两个同步文档：

1. **实验日志**
   记录精确的命令、脚本、配置、数据集、骨干模型、检查点、prompt 来源、运行备注、错误、修复以及 git commit 信息。

2. **实验分析**
   总结量化结果，将其与 E0 基线比较，识别收益或失败，并解释该实验是否支持 MCM-PC 假设。

此外，完整的论文手稿应在整个项目过程中持续维护和更新。

## 论文草稿规则

完整的 ICASSP 论文草稿应单独维护在：

`paper/ICASSP/`

论文草稿应与实验同步更新，包括动机、相关工作、方法、实验、消融、图表、局限性和参考文献。实验发现应逐步转化为适合论文使用的写作，而不是只停留在笔记层面。

## 推荐文档结构

每个实验请使用如下结构：

`docs/experiments/E*_name/`

至少包含：

- `log.md`
- `analysis.md`

对于 E0，请使用：

`docs/experiments/E0_baseline/`

对于归档的历史实验，请使用：

`docs/experiments/archive/legacy_pre_mcmpc_restart/`

## 立即下一步

在这份决策文档确认后，应将历史实验文档移动到：

`docs/experiments/archive/legacy_pre_mcmpc_restart/`

随后应在以下位置初始化 ICASSP 论文工作区：

`paper/ICASSP/`
