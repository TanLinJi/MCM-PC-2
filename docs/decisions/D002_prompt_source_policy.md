# D002: 面向 E1 文本原型增强的 Prompt 来源策略

## 日期

2026-06-03

## 背景

Point-Cache 通过类别名和 prompt 模板构造文本原型。其视觉侧适应是动态的，因为全局缓存和局部缓存会随着在线测试样本不断更新。然而，它的文本侧表示主要仍依赖固定的手工 prompt 集合。

原始手工 prompt 集合包含许多 2D 图像风格模板，例如 photo-style、blurry-photo-style、cropped-photo-style 和 painting-style prompts。这些模板有助于复现原始 Point-Cache baseline，但对 3D 点云识别未必最优。

因此，E1 先聚焦于文本侧处理，再去修改缓存机制。

## E1 定义

E1 定义为：

**结合点云感知与动态生成 prompt 的文本原型增强**

目标是研究：通过点云感知模板和 LLM 生成的类别级描述，文本原型是否可以得到提升。

## 术语定义

为避免歧义，这里统一定义 E1 中使用的所有 prompt 来源名称。

### manual_full

`manual_full` 表示 Point-Cache 使用的原始完整手工 prompt 集合。

中文理解：`manual_full` 指 Point-Cache 原始完整手工模板集合，也就是 baseline 默认使用的固定 prompt templates。它既包含点云相关模板，也包含大量 2D 图像风格模板，例如 photo-style 或 painting-style prompts。

这是与 E0 兼容的 baseline prompt 来源，必须保持不变。

### manual_3d

`manual_3d` 表示从 `manual_full` 中筛选出的点云感知子集。

中文理解：`manual_3d` 指从原始手工模板中筛选出的点云 / 3D 相关模板子集。

筛选原则：

- 保留与 point cloud、3D、object、shape、model、geometry 或 scene 相关的 prompt；
- 去除明显属于图像风格的 prompt，例如 photo、image、picture、painting、sketch、cartoon、blurry、cropped 或 black-and-white 描述。

`manual_3d` 不是 `manual_full` 的替代品，它只是 E1 的一个对比设置。

### llm_static

`llm_static` 表示在实验前由 LLM 生成并保存为 JSON 的静态 prompt 来源。

中文理解：`llm_static` 指提前用 LLM 生成并保存为 JSON 的固定文本描述。实验时只读取，不再动态调用 API。

这种设置有利于复现，也便于与现有 Point-Cache 风格的固定 LLM prompts 进行比较。

### llm_dynamic_init

`llm_dynamic_init` 表示在每次实验开始时生成的 LLM prompts。

中文理解：`llm_dynamic_init` 指在每次实验启动时，根据当前数据集的候选类别名称调用 LLM 生成文本描述。生成后会保存并冻结，在整个测试流中不再更新。

这里的生成是类别级别的，不是样本级别的。

允许使用的信息：

- 数据集候选类别名。

禁止使用的信息：

- 测试样本的 ground-truth 标签；
- 测试点云内容；
- 单个测试样本的预测结果。

### manual3d_llm_dynamic_init

`manual3d_llm_dynamic_init` 表示将 `manual_3d` 和 `llm_dynamic_init` 组合起来的混合 prompt 来源。

中文理解：`manual3d_llm_dynamic_init` 指“点云手工模板分支 + LLM 动态生成描述分支”的融合版本。

这是主要的 E1 方法候选。

## 核心决策

原始 Point-Cache 的 prompt 模板不能被删除。

它们会作为与 E0 兼容的 baseline 文本设置保留。

E1 会引入额外的 prompt 来源用于受控比较，而不是直接覆盖原有模板列表。

## 动态 Prompt 规则

E1 使用的是 dynamic-init prompt 生成方式，而不是 dynamic-online prompt 生成方式。

E1 中允许的做法：

1. 在推理前读取数据集候选类别名；
2. 调用 LLM 生成类别级别的点云感知描述；
3. 将生成的 prompts 保存到实验结果目录；
4. 构建文本原型；
5. 使用冻结后的文本原型进行 zero-shot 或 Point-Cache 推理。

E1 中不允许的做法：

1. 不要对每个测试样本都调用 LLM；
2. 不要基于单个测试样本生成 prompt；
3. 不要使用 ground-truth 标签；
4. 不要向 LLM 暴露点云内容；
5. 不要在测试流中更新 prompt。

dynamic-online prompt 生成留作未来工作。

## E1 的主要融合形式

更推荐的 E1 形式是分支级加权融合，而不是强行规定 prompt 字符串的总数量。

默认形式：

    text_prototype =
        static_weight * mean(manual_3d embeddings)
        + dynamic_weight * mean(dynamic LLM embeddings)

默认设置：

- 静态分支权重：0.75
- 动态分支权重：0.25
- 动态 prompt 数量：每个类别 25 个 prompts

这样可以避免强行要求 `manual_3d` 固定包含某个数量的模板。

## API Key 策略

LLM API key 绝不能提交到仓库中。

为了方便，项目使用一个固定的本地 key 文件：

    Point-Cache/llm/secrets/llm_api_key.txt

这个文件仅限本地使用，必须被 git 忽略。

文件格式为单行真实 API key，例如：

    sk-xxxxxxxxxxxxxxxx

实现中应使用 `strip()` 读取该文件，以去掉首尾空白字符。

实现中应按以下顺序读取 API key：

1. 如果 `Point-Cache/llm/secrets/llm_api_key.txt` 存在，则从该文件读取；
2. 否则读取环境变量 `LLM_API_KEY`；
3. 如果两者都不存在且动态 prompt 已启用，则抛出错误。

本项目不使用示例 key 文件。

## 可复现性规则

生成后的 prompts 必须保存到对应实验结果目录下，例如：

    Point-Cache/results/mcmpc/E1_tpe/prompts_used.json

保存的 prompt 文件应包含：

- prompt 来源；
- LLM 提供方；
- LLM 模型名；
- 生成时间；
- prompt 预算；
- 类别名；
- 生成的 prompts；
- 生成配置。

这样可以保证动态 prompts 在生成后仍然可审计、可复现。

## E1 实验顺序

### 阶段 1：zero-shot prompt 比较

比较以下设置：

- manual_full；
- manual_3d；
- llm_dynamic_init；
- manual3d_llm_dynamic_init。

### 阶段 2：Point-Cache prompt 比较

如果阶段 1 显示出有意义的趋势，则进一步扩展到：

- zero-shot；
- zero-shot + global cache；
- zero-shot + global cache + local cache。

## 初始优先级设置

E1 的初始设置如下：

- ULIP × ModelNet-C all35；
- ULIP-2 × ModelNet-C all35；
- Uni3D × ScanObjNN-C hardest all35。

## 与 E0 的关系

E0 仍然是 Point-Cache baseline 的复现与分析。

`manual_full` 是与 E0 兼容的文本设置。

E1 不能覆盖 E0 的结果。


## 2026-06-03 方向更新：从 2D 过滤转向多视角 Prompt 增强

早期 E1 的 severity=2 结果显示，在 ULIP × ModelNet-C 的 zero-shot 评估中，`manual_3d` 明显比 `manual_full` 更差。

这说明，直接移除 2D 图像风格 prompt 并不是 E1 的好主方向。

虽然 `manual_full` 中很多模板看起来像 2D 图像 prompt，但它们仍然为 ULIP 提供了有用的 CLIP-style 视觉语义锚点。因此，`manual_full` 不仅应该作为与 E0 兼容的 baseline 被保留，也应该被视为一个重要且稳定的文本先验。

### 更新后的决策

E1 不应再把 `manual_3d` 当作 `manual_full` 的主要替代方案。

相反，E1 应当遵循以下方向：

- 保留 `manual_full` 作为稳定的 CLIP-style 视觉语义锚点；
- 将 LLM 生成的描述作为额外的语义扩展分支；
- 让 LLM 描述具有多视角特征，同时覆盖 2D 视觉语义和 3D 点云几何。

### 更新后的 prompt 来源角色

`manual_full`：

- 原始完整手工 prompt 集合；
- 与 E0 兼容的 baseline；
- 稳定的 CLIP-style 视觉语义先验；
- 应当被保留。

`manual_3d`：

- 经过筛选的 3D-only 手工 prompt 子集；
- 现在被视为诊断性消融，而不是主方法；
- 用于说明简单移除 2D 图像风格 prompt 是有害的。

`llm_dynamic_init`：

- 应从纯点云几何描述修订为多视角类别描述；
- 生成的描述应同时包含常见视觉外观和 3D 几何结构。

`manual3d_llm_dynamic_init`：

- 保留为诊断性设置；
- 不再作为主要的 E1 候选。

`manualfull_llm_dynamic_init`：

- 新的主要 E1 候选；
- 融合 `manual_full` 与多视角的 LLM 生成类别描述；
- 旨在在加入 3D 几何知识的同时保留 CLIP-style 视觉锚点。

### 更新后的 LLM 生成原则

LLM 应生成同时包含以下两类信息的描述：

1. 对图文模型通常有用的视觉语义线索，例如常见物体外观、可识别部件和类别级视觉身份；
2. 3D 点云线索，例如形状、结构、部件布局、对称性、空间排列和几何属性。

LLM 不应只生成纯 photo templates，也不应只生成纯点云几何描述。

更推荐的描述格式是将视觉身份与 3D 结构结合成完整句子。

示例：

    飞机在视觉上通常由机身、机翼和尾翼来识别，而其点云通常会呈现出细长的机身、左右对称的机翼结构以及后部稳定翼。

### 更新后的主要假设

修订后的 E1 假设是：

`manual_full` 提供稳定的 CLIP-style 视觉语义锚点，而 LLM 生成的多视角描述提供额外的 3D 几何语义。两者结合后，可能比仅使用 2D 风格手工模板或仅使用纯 3D 描述生成更强的文本原型。


## 2026-06-03 更新：取消 `manual_3d` 的主方向地位

在 ULIP × ModelNet-C 的 severity=2 zero-shot 测试之后，`manual_3d` 被发现明显劣于 `manual_full`。

决策：

- `manual_3d` 不再作为活跃的 E1 prompt 来源；
- `manual3d_llm_dynamic_init` 不再作为活跃的 E1 候选；
- 现有的 `manual_3d` 结果只保留为负面诊断证据；
- 后续 E1 实验应重点关注：
  - `manual_full`
  - `llm_dynamic_init`
  - `manualfull_llm_dynamic_init`

现在 E1 的主要方向变为：

    manual_full + multi-view LLM descriptions

其中 `manual_full` 保留 CLIP-style 视觉语义锚点，而 LLM 分支则补充同时包含 2D 视觉语义和 3D 点云几何的多视角描述。

