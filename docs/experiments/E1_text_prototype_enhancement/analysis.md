# E1：文本原型增强实验分析

## 术语速查表

为避免后续分析中出现缩写歧义，本实验统一使用以下术语。

| 名称 | 中文含义 | 在本实验中的作用 |
|---|---|---|
| `manual_full` | 原始完整手工模板集合 | Point-Cache baseline 默认使用的完整固定模板集合，作为 E0 兼容对照，不能删除或改写。 |
| `manual_3d` | 点云/3D 相关手工模板子集 | 从 `manual_full` 中筛选出的更适合点云识别的模板集合，用于验证去掉 2D 图像风格模板是否有帮助。 |
| `llm_static` | LLM 离线固定描述集合 | 预先用 LLM 生成并保存为 JSON 的类别描述，实验时只读取，不动态调用 API。 |
| `llm_dynamic_init` | LLM 实验初始化动态描述集合 | 每次实验启动时，根据当前数据集候选类别名称调用 LLM 生成类别级点云描述；生成后固定用于整个测试流。 |
| `manual3d_llm_dynamic_init` | 点云手工模板与 LLM 动态描述融合 | 将 `manual_3d` 分支和 `llm_dynamic_init` 分支进行加权融合，是 E1 的主要候选方法。 |
| `prompt source` | 提示词来源 | 指构造文本原型时使用哪一种文本模板或类别描述来源。 |
| `text prototype` | 文本原型 | 每个类别的文本特征表示，由该类别的多条提示词编码后平均或加权融合得到。 |
| `dynamic-init` | 初始化阶段动态生成 | 在实验开始、测试流开始之前生成提示词；测试过程中不再更新。 |
| `dynamic-online` | 测试流中在线动态生成 | 每来一个测试样本或根据测试过程持续调用 LLM 生成提示词；E1 不采用这种方式。 |
| `all35` | 35 个损坏设置 | 7 类基础损坏类型 × 5 个损坏强度，不译为“腐败”。 |
| `corruption` | 损坏类型/扰动类型 | 点云鲁棒性实验中的数据损坏或扰动，例如 jitter、scale、dropout_global 等。 |

## 研究问题

Point-Cache 的文本原型是否可以通过点云语义相关模板和 LLM 动态生成的类别级描述得到增强？

更具体地说，E1 关注以下问题：

1. 原始完整手工模板集合 `manual_full` 中包含大量 2D 图像风格模板，这些模板是否会稀释点云几何语义？
2. 从原始模板中筛选出的点云/3D 相关模板 `manual_3d` 是否优于 `manual_full`？
3. 在实验初始化阶段由 LLM 根据候选类别名称生成的类别级描述，是否能提升文本原型质量？
4. `manual_3d` 与 LLM 动态生成描述的融合是否能进一步提升 zero-shot 和 Point-Cache 结果？

## 核心假设

Point-Cache 的视觉侧可以通过 global cache 和 local cache 动态适应测试数据流，但文本侧主要依赖固定模板。

如果原始模板中存在大量 2D image-style prompt，那么这些模板可能并不适合 3D 点云识别。使用点云语义更明确的模板，或者结合 LLM 生成的类别级点云描述，可能会得到更好的文本原型。

## 对比设置

### manual_full：原始完整手工模板集合

含义：

    Point-Cache 原始完整手工模板集合，也是 E0 baseline 兼容的文本端设置。

作用：

    作为所有 E1 文本端方法的基础对照。

预期：

    能复现 Point-Cache 原始文本原型行为。

### manual_3d：点云/3D 相关手工模板子集

含义：

    从 manual_full 中筛选出的点云/3D 相关模板子集。

作用：

    验证移除 2D 图像风格模板是否能提升点云文本原型。

预期：

    如果 2D 图像风格模板对点云任务有干扰，则 manual_3d 可能优于 manual_full。

### llm_dynamic_init：LLM 实验初始化动态描述集合

含义：

    实验启动时，根据候选类别名称调用 LLM 生成类别级点云描述，生成后固定使用。

作用：

    验证动态大模型类别描述是否能提升文本原型语义质量。

预期：

    对固定模板过于泛化或语义不足的类别可能有帮助。

### manual3d_llm_dynamic_init：点云手工模板与 LLM 动态描述融合

含义：

    manual_3d 分支与 llm_dynamic_init 分支的加权融合。

作用：

    作为 E1 主要候选方法，结合稳定的点云手工模板先验和动态大模型语义扩展。

预期：

    如果两类文本信息互补，该设置应优于单独使用 manual_3d 或 llm_dynamic_init。

## 计划指标

### Zero-shot 阶段

- all35 平均准确率；
- 按基础损坏类型统计准确率；
- 按损坏强度统计准确率；
- 相对 `manual_full` 的提升或下降；
- 不同 backbone 和数据集上的稳定性。

### Point-Cache 阶段

- zero-shot 准确率；
- zero-shot + global cache 准确率；
- zero-shot + global cache + local cache 准确率；
- global gain；
- local extra gain；
- 相对 E0 baseline 的总提升。

## 第一批优先实验设置

1. ULIP × ModelNet-C all35
2. ULIP-2 × ModelNet-C all35
3. Uni3D × ScanObjNN-C hardest all35

选择原因：

- ULIP 是较弱 backbone，便于观察文本端增强是否有效；
- ULIP-2 可作为中间强度对照；
- Uni3D × ScanObjNN-C hardest 更接近强 backbone + 真实扫描损坏场景。

## 成功标准

E1 被认为有继续推进价值，如果满足以下任一情况：

- `manual_3d` 优于 `manual_full`；
- `llm_dynamic_init` 优于 `manual_full` 或 `manual_3d`；
- `manual3d_llm_dynamic_init` 优于 `manual_full`；
- zero-shot 增益能够传递到 Point-Cache global 或 hierarchical 设置；
- hard 损坏类型或 ScanObjNN-C hardest 上出现稳定提升。

## 当前状态

规划阶段。

暂无实验结果。

## 分析记录

待阶段 1 实验完成后补充。

## 可写入论文的结论

待实验结果确认后补充。
