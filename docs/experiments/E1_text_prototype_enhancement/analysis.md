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


## 2026-06-03 阶段性结果与方向修正

### 已完成最小测试

已完成：

    ULIP × ModelNet-C severity=2 × zero-shot × manual_full

结果与 E0 baseline 完全一致，说明 E1 新增的 `--prompt-source manual_full` 路径没有破坏原始 Point-Cache 文本原型构造流程。

随后完成：

    ULIP × ModelNet-C severity=2 × zero-shot × manual_3d

结果明显低于 `manual_full`。

### 关键现象

`manual_3d` 相比 `manual_full` 明显下降，说明直接删除 2D 图像风格模板并不可行。

这表明：

1. Point-Cache 原始 `manual_full` 中虽然包含许多 2D image-style prompts；
2. 但这些 prompts 可能与 ULIP 的 CLIP-style 文本空间高度匹配；
3. 它们为文本原型提供了重要的视觉语义锚点；
4. 仅保留点云/3D 相关模板会导致文本原型偏离预训练文本空间，从而降低 zero-shot 性能。

### 结论修正

原始假设：

    删除 2D 图像风格模板，只保留点云/3D 模板，可能提升点云文本原型。

当前结果不支持该假设。

修正后的假设：

    manual_full 中的 2D/视觉语义模板具有稳定的 CLIP-style 语义锚定作用，不应删除。
    E1 应保留 manual_full，并使用 LLM 生成同时包含 2D 视觉语义和 3D 点云几何语义的多视角描述作为补充。

### E1 新主线

E1 后续主线调整为：

    manual_full + multi-view LLM descriptions

其中 multi-view LLM descriptions 指 LLM 生成的类别描述需要同时覆盖：

- 2D 视觉语义：常见外观、可识别部件、视觉类别身份；
- 3D 点云几何：整体形状、结构部件、对称性、空间布局、几何特征。

### 设置角色调整

`manual_full`：

    不仅是 baseline 对照，也是后续融合方法中的稳定视觉语义先验。

`manual_3d`：

    从主方法候选降级为诊断消融，用于证明简单过滤 2D 模板是有害的。

`llm_dynamic_init`：

    需要从“纯点云几何描述”调整为“视觉语义 + 点云几何”的多视角 LLM 描述。

`manual3d_llm_dynamic_init`：

    保留为诊断对照，但不再作为主方法。

`manualfull_llm_dynamic_init`：

    新增为 E1 主方法候选，表示原始完整手工模板分支与多视角 LLM 描述分支的加权融合。

### 后续实验优先级

接下来优先验证：

1. `llm_dynamic_init`：使用多视角 LLM 描述，观察纯 LLM 文本原型是否优于 `manual_3d`；
2. `manualfull_llm_dynamic_init`：融合 `manual_full` 和多视角 LLM 描述，观察是否能在不破坏 CLIP-style 语义锚点的前提下提升 zero-shot；
3. 暂缓将 `manual3d_llm_dynamic_init` 作为主实验，只保留为消融。


## 2026-06-03：manual_3d 方向终止

`manual_3d` 的 severity=2 结果显著低于 `manual_full`，说明简单删除 2D 图像风格模板会损害 ULIP 的文本原型。

因此，`manual_3d` 不再作为 E1 主线或后续默认对照。

保留结论：

    manual_3d 是一个失败但有价值的诊断消融。
    它证明了原始 manual_full 中的 2D/视觉语义模板虽然看似不符合点云直觉，
    但对于 CLIP-style 文本空间具有重要的语义锚定作用。

后续分析集中于：

- `manual_full`：原始完整手工模板集合；
- `llm_dynamic_init`：多视角 LLM 类别描述；
- `manualfull_llm_dynamic_init`：manual_full 与多视角 LLM 描述融合。

