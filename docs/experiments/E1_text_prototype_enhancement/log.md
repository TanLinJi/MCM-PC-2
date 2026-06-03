# E1：文本原型增强实验日志

## 基本信息

- 实验编号：E1
- 实验名称：文本原型增强（Text Prototype Enhancement）
- 开始日期：2026-06-03
- 论文方向：MCM-PC：面向 3D 点云视觉语言模型测试时适应的可靠性感知多缓存矩阵方法

## 术语速查表

为避免后续实验记录混乱，本实验中所有提示词来源名称统一解释如下。

| 名称 | 中文含义 | 在本实验中的作用 |
|---|---|---|
| `manual_full` | 原始完整手工模板集合 | Point-Cache baseline 默认使用的完整固定模板集合，作为 E0 兼容对照，不能删除或改写。 |
| `manual_3d` | 点云/3D 相关手工模板子集 | 从 `manual_full` 中筛选出的更适合点云识别的模板集合，用于验证去掉 2D 图像风格模板是否有帮助。 |
| `deepseek_static` | DeepSeek 离线固定描述集合 | 预先用 DeepSeek 生成并保存为 JSON 的类别描述，实验时只读取，不动态调用 API。 |
| `deepseek_dynamic_init` | DeepSeek 实验初始化动态描述集合 | 每次实验启动时，根据当前数据集候选类别名称调用 DeepSeek 生成类别级点云描述；生成后固定用于整个测试流。 |
| `manual3d_deepseek_dynamic_init` | 点云手工模板与 DeepSeek 动态描述融合 | 将 `manual_3d` 分支和 `deepseek_dynamic_init` 分支进行加权融合，是 E1 的主要候选方法。 |
| `prompt source` | 提示词来源 | 指构造文本原型时使用哪一种文本模板或类别描述来源。 |
| `text prototype` | 文本原型 | 每个类别的文本特征表示，由该类别的多条提示词编码后平均或加权融合得到。 |
| `dynamic-init` | 初始化阶段动态生成 | 在实验开始、测试流开始之前生成提示词；测试过程中不再更新。 |
| `dynamic-online` | 测试流中在线动态生成 | 每来一个测试样本或根据测试过程持续调用 LLM 生成提示词；E1 不采用这种方式。 |
| `all35` | 35 个损坏设置 | 7 类基础损坏类型 × 5 个损坏强度，不译为“腐败”。 |
| `corruption` | 损坏类型/扰动类型 | 点云鲁棒性实验中的数据损坏或扰动，例如 jitter、scale、dropout_global 等。 |

## 实验目标

E1 研究 Point-Cache 的文本原型构造方式是否可以通过点云语义相关模板和 DeepSeek 动态生成的类别级描述得到增强。

E1 只处理文本端，不先修改 global cache 或 local cache 机制。

## 目录约定

文档目录：

    docs/experiments/E1_text_prototype_enhancement/

脚本目录：

    Point-Cache/scripts/E1_text_prototype_enhancement/

结果目录：

    Point-Cache/results/E1_text_prototype_enhancement/

## 提示词来源详细说明

### manual_full：原始完整手工模板集合

`manual_full` 指 Point-Cache 原始完整手工模板集合。

它是 E0 baseline 兼容的文本设置，必须保持不变。该集合中既包含少量点云相关模板，也包含大量 2D 图像风格模板，例如 photo、blurry photo、painting、cropped photo 等。

### manual_3d：点云/3D 相关手工模板子集

`manual_3d` 指从 `manual_full` 中筛选出的点云/3D 相关模板子集。

保留原则：

- 保留与 point cloud、3D、object、shape、model、geometry、scene 等语义相关的模板；
- 移除明显 2D 图像风格模板，例如 photo、image、picture、painting、sketch、cartoon、blurry、cropped、black-and-white 等。

### deepseek_static：DeepSeek 离线固定描述集合

`deepseek_static` 指提前使用 DeepSeek 生成并保存为 JSON 的固定类别描述。

实验时只读取 JSON，不再调用 API。该设置主要用于可复现对照。

### deepseek_dynamic_init：DeepSeek 实验初始化动态描述集合

`deepseek_dynamic_init` 指每次实验启动时，根据当前数据集的候选类别名称调用 DeepSeek 生成类别级点云描述。

该设置只允许使用候选类别名称，不允许使用测试样本真实标签、测试点云内容或单个样本预测结果。生成后的提示词会被保存，并在整个测试流中固定使用。

### manual3d_deepseek_dynamic_init：点云手工模板与 DeepSeek 动态描述融合

`manual3d_deepseek_dynamic_init` 指 `manual_3d` 分支和 `deepseek_dynamic_init` 分支的融合版本。

这是 E1 的主要候选方法。

## 计划阶段

### 阶段 1：Zero-shot 文本原型对比

先不启用 cache，只比较不同提示词来源对 zero-shot 推理的影响。

计划比较：

- `manual_full`
- `manual_3d`
- `deepseek_dynamic_init`
- `manual3d_deepseek_dynamic_init`

### 阶段 2：Point-Cache 文本原型对比

如果阶段 1 出现有意义趋势，再扩展到 Point-Cache 设置：

- zero-shot
- zero-shot + global cache
- zero-shot + global cache + local cache

## 第一批优先实验设置

- ULIP × ModelNet-C all35
- ULIP-2 × ModelNet-C all35
- Uni3D × ScanObjNN-C hardest all35

其中 all35 表示 7 类基础损坏类型 × 5 个损坏强度。

## 实现日志

### 2026-06-03

初始化 E1 中文实验日志。

当前尚未修改代码。

## 运行命令

待代码实现后补充。

## 实验结果

待实验运行后补充。

## 问题与修复

按实验过程持续补充。

## Git 记录

按每次相关提交持续补充。
