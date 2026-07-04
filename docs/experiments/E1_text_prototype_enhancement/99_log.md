# E1：文本原型增强实验日志

## 基本信息

- 实验编号：E1
- 实验名称：文本原型增强（Text Prototype Enhancement）
- 开始日期：2026-06-03
- 当前论文方向：DPC-Point：Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation
- 当前角色：E1 作为 DPC-Point 的 text-distribution prior 来源；直接替换最终文本分类器不是当前主线。

## 术语速查表

为避免后续实验记录混乱，本实验中所有提示词来源名称统一解释如下。

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

## 实验目标

E1 研究 Point-Cache 的文本原型构造方式是否可以通过点云语义相关模板和 LLM 动态生成的类别级描述得到增强。

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

### llm_static：LLM 离线固定描述集合

`llm_static` 指提前使用 LLM 生成并保存为 JSON 的固定类别描述。

实验时只读取 JSON，不再调用 API。该设置主要用于可复现对照。

### llm_dynamic_init：LLM 实验初始化动态描述集合

`llm_dynamic_init` 指每次实验启动时，根据当前数据集的候选类别名称调用 LLM 生成类别级点云描述。

该设置只允许使用候选类别名称，不允许使用测试样本真实标签、测试点云内容或单个样本预测结果。生成后的提示词会被保存，并在整个测试流中固定使用。

### manual3d_llm_dynamic_init：点云手工模板与 LLM 动态描述融合

`manual3d_llm_dynamic_init` 指 `manual_3d` 分支和 `llm_dynamic_init` 分支的融合版本。

这是 E1 的主要候选方法。

## 计划阶段

### 阶段 1：Zero-shot 文本原型对比

先不启用 cache，只比较不同提示词来源对 zero-shot 推理的影响。

计划比较：

- `manual_full`
- `manual_3d`
- `llm_dynamic_init`
- `manual3d_llm_dynamic_init`

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

添加 E1 文本原型增强的命令行参数入口，默认 `prompt_source=manual_full`，不改变 E0 baseline 行为。

新增参数包括：

- `--prompt-source`
- `--llm-api-key-file`
- `--llm-model`
- `--dynamic-prompt-count`
- `--prompt-static-weight`
- `--prompt-dynamic-weight`

## 代码改动记录

### 2026-06-03：支持 manual_full / manual_3d 提示词来源

本次改动：

- 在 `datasets/templates.py` 中新增 `manual_3d_prompts`，由原始 `text_prompts` 自动筛选得到；
- 新增 `datasets/prompt_utils.py`，统一根据 `--prompt-source` 选择提示词来源；
- 修改 `modelnet40.py`、`modelnet_c.py`、`scanobjnn.py`、`sonn_c.py`，使其使用 `get_prompt_template(...)`；
- 当前已支持：
  - `manual_full`
  - `manual_3d`
- LLM 相关来源暂时只保留参数入口，后续实现。

默认 `--prompt-source manual_full`，因此不改变 E0 baseline 默认行为。

## 运行命令

待代码实现后补充。

## 代码改动记录补充

### 2026-06-03：重构 clip_classifier 文本原型构造接口

本次改动：

- 将 `clip_classifier()` 中的文本构造、文本编码和原型平均逻辑拆成辅助函数；
- 当前继续支持原始 list 模板格式，因此 `manual_full` 和 `manual_3d` 均可正常工作；
- 预留 dict 格式提示词支持，用于后续 LLM 生成的类别级描述；
- 默认路径不改变 E0 baseline 行为。

## 实验结果

待实验运行后补充。

## 问题与修复

按实验过程持续补充。

## Git 记录

按每次相关提交持续补充。

### 2026-06-03：支持两路文本原型加权融合

本次改动：

- 在 `clip_classifier()` 中增加“两路文本原型加权融合”能力；
- 第一路用于 `manual_3d` 点云手工模板；
- 第二路用于后续 LLM 生成的类别级描述；
- 当前只是增加融合接口，不调用 LLM API；
- 默认 `manual_full` 和 `manual_3d` 的原始模板平均逻辑不变，因此不影响 E0 baseline 默认行为。

融合形式：

    final_text_prototype =
        static_weight * manual_3d_text_prototype
        + dynamic_weight * llm_text_prototype

默认计划：

- static_weight = 0.75
- dynamic_weight = 0.25

### 2026-06-03：实现通用 LLM 初始化阶段动态提示词生成

本次改动：

- 新增 `Point-Cache/llm/llm_prompt_generator.py`；
- 统一使用通用 LLM 命名，不再把 E1 方法绑定到 DeepSeek；
- 当前默认 provider 是 `deepseek`，默认模型是 `deepseek-v4-pro`；
- API key 固定读取 `Point-Cache/llm/secrets/llm_api_key.txt`；
- API key 文件格式为单行 `sk-xxx`；
- 支持根据数据集候选类别名称生成类别级点云描述；
- 生成结果会保存到 `Point-Cache/results/E1_text_prototype_enhancement/prompts/`；
- 再次运行时，如果缓存 JSON 已存在，默认直接读取，不重复调用 API；
- 新增 `--force-regenerate-prompts`，用于强制重新生成。

注意：

- E1 只根据候选类别名称生成提示词；
- 不使用测试样本真实标签；
- 不使用测试点云内容；
- 不根据单个样本预测结果生成提示词。


### 2026-06-03：重命名 E1 动态提示词生成模块

为避免与 Point-Cache 原始离线提示词生成脚本 `llm_generate_prompts.py` 混淆，将新增的通用 LLM 动态提示词生成模块重命名为：

    Point-Cache/llm/e1_dynamic_prompt_generator.py

说明：

- `llm_generate_prompts.py` 是 Point-Cache 原始代码中的旧脚本，主要用于离线生成固定 GPT / PointLLM prompt JSON；
- `e1_dynamic_prompt_generator.py` 是 E1 新增模块，用于实验初始化阶段根据候选类别名称动态生成类别级点云描述；
- 两者用途不同，后续 E1 统一使用 `e1_dynamic_prompt_generator.py`。

### 2026-06-03：新增 ULIP × ModelNet-C all35 zero-shot 的 manual_full 脚本

本次新增 E1 第一组实际实验脚本：

- `Point-Cache/scripts/E1_text_prototype_enhancement/01_run_ulip_modelnetc_all35_zs_prompt_common.sh`
- `Point-Cache/scripts/E1_text_prototype_enhancement/01_1_ulip_modelnetc_all35_zs_manual_full_single_gpu.sh`

该实验用于验证：

- E1 的 `--prompt-source manual_full` 能够正常进入 runner；
- `manual_full` 作为 Point-Cache 原始完整手工模板集合，能够作为 E1 的 E0-compatible 对照；
- 结果保存到 `Point-Cache/results/E1_text_prototype_enhancement/01_1_ulip_modelnetc_all35_zs_manual_full/`。

运行命令：

    bash Point-Cache/scripts/E1_text_prototype_enhancement/01_1_ulip_modelnetc_all35_zs_manual_full_single_gpu.sh 0

### 2026-06-03：修复 clip_classifier 缺失辅助函数问题

问题：

- 运行 `manual_full` zero-shot 脚本时，`clip_classifier()` 报错：
  `NameError: name '_build_prompt_texts' is not defined`

原因：

- 前面重构 `clip_classifier()` 时，函数体已经切换到新的文本构造接口；
- 但 `_build_prompt_texts()`、`_lookup_class_prompts()`、`_encode_texts_as_prototype()` 三个辅助函数没有成功写入 `utils.py`。

修复：

- 在 `Point-Cache/utils/utils.py` 中补充上述三个辅助函数；
- `manual_full` 和 `manual_3d` 使用 list 模板；
- `llm_dynamic_init` 使用按类别名称保存的提示词字典。


### 2026-06-03：新增 ULIP × ModelNet-C severity=2 zero-shot 文本消融脚本

本次新增 severity=2 最小测试 runner：

- `Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py`

该 runner 只跑 7 个 severity=2 损坏设置：

- add_global_2
- add_local_2
- dropout_global_2
- dropout_local_2
- rotate_2
- scale_2
- jitter_2

本次新增 4 个脚本，对应 E1 文本端的 4 个不同假设：

- `00_1_ulip_modelnetc_s2_zs_manual_full_smoke.sh`
  - 目的：验证 E1 新接口不破坏 Point-Cache 原始完整手工模板 baseline。
- `00_2_ulip_modelnetc_s2_zs_manual_3d_smoke.sh`
  - 目的：验证删除明显 2D 图像风格模板、只保留点云/3D相关手工模板是否有帮助。
- `01_2_ulip_modelnetc_s2_zs_dynamic_llm_descriptions_single_gpu.sh`
  - 目的：验证实验初始化阶段由 LLM 根据候选类别名称生成点云描述是否有帮助。
- `01_3_ulip_modelnetc_s2_zs_fuse_manual3d_llm_single_gpu.sh`
  - 目的：验证点云手工模板分支与 LLM 动态描述分支加权融合是否有帮助。


### 2026-06-03：根据 manual_3d 结果修正 E1 实验方向

已完成 `manual_full` 和 `manual_3d` 的 ULIP × ModelNet-C severity=2 zero-shot 最小测试。

观察：

- `manual_full` 与 E0 baseline 完全一致，说明 E1 新接口没有破坏原始文本原型路径；
- `manual_3d` 明显低于 `manual_full`，说明简单删除 2D 图像风格模板不可行。

方向修正：

- 不再把 `manual_3d` 作为 E1 主方法；
- `manual_3d` 改为诊断消融，用于说明简单过滤 2D 模板会损害 ULIP 文本原型；
- 后续 E1 主线改为保留 `manual_full` 的 CLIP-style 视觉语义锚点，同时引入 LLM 生成的多视角类别描述；
- LLM 描述应同时包含：
  - 2D 视觉语义；
  - 3D 点云几何结构；
- 新增主候选方向：`manualfull_llm_dynamic_init`。

下一步需要修改：

- LLM 生成提示词，使其从纯 3D 几何描述调整为“视觉语义 + 点云几何”的多视角描述；
- `prompt_utils.py` 和 `clip_classifier()`，支持 `manualfull_llm_dynamic_init`；
- E1 severity=2 脚本，增加 `manualfull_llm_dynamic_init` 对照。


### 2026-06-03：移除 manual_3d 主线

根据 severity=2 最小实验结果，`manual_3d` 明显低于 `manual_full`。

处理：

- `manual_3d` 不再作为后续 active prompt source；
- `manual3d_llm_dynamic_init` 不再作为后续 active candidate；
- 已经得到的 `manual_3d` 结果保留为失败证据；
- 后续 E1 只继续关注：
  - `manual_full`
  - `llm_dynamic_init`
  - `manualfull_llm_dynamic_init`

当前主线：

    保留 manual_full 的 CLIP-style 视觉语义锚点，
    同时引入 LLM 生成的 2D 视觉语义 + 3D 点云几何多视角描述。


### 2026-06-03：E1 融合方案取得阶段性正结果

完成 ULIP × ModelNet-C severity=2 zero-shot 下的关键文本方案对比。

当前结果：

- 原始完整手工模板 manual_full：47.68
- 删除 2D 模板后的 manual_3d：35.63
- 只使用 LLM 生成的类别级多视角描述 llm_dynamic_init：39.30
- 原始完整手工模板与 LLM 描述融合 manualfull_llm_dynamic_init：48.88

关键结论：

- 简单删除 2D 模板不可行；
- 只用大模型描述不能替代原始模板；
- 原始完整手工模板与LLM 生成的类别级多视角描述融合后，平均准确率超过 baseline 1.20；
- 这是 E1 当前第一个有效正结果。

已新增阶段性报告：

    docs/experiments/E1_text_prototype_enhancement/e1_prompt_fusion_stage_report.md

### 2026-06-03：完成 E1 smoke test 脚本命名规范化

本次调整：

- smoke test 脚本统一改为 `00_*` 编号；
- 文件名不再包含 `single_gpu`；
- 默认单卡运行，但用户仍可通过脚本最后一个参数选择物理 GPU；
- 四种核心方法统一为：
  - `manual_full`
  - `manual_3d`
  - `llm_only`
  - `manual_full_llm_fusion`
- 新增/保留 `manual_3d` smoke test 脚本，用于复现失败消融；
- 脚本 README 已同步更新。

当前规范化后的核心脚本：

- `00_0_llm_prompt_generation_smoke.sh`
- `00_run_ulip_modelnetc_s2_zs_text_method_smoke_common.sh`
- `00_1_ulip_modelnetc_s2_zs_manual_full_smoke.sh`
- `00_2_ulip_modelnetc_s2_zs_manual_3d_smoke.sh`
- `00_3_ulip_modelnetc_s2_zs_llm_only_smoke.sh`
- `00_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_smoke.sh`

### 2026-06-03：补充 E1 smoke test 独立分析文档

本次新增 smoke test 分析目录：

    docs/experiments/E1_text_prototype_enhancement/

新增文档：

- `00_1_manual_full_smoke_analysis.md`
- `00_2_manual_3d_smoke_analysis.md`
- `00_3_llm_only_smoke_analysis.md`
- `00_4_manual_full_llm_fusion_smoke_analysis.md`
- `00_smoke_test_summary.md`

这些文档分别对应当前四个 E1 smoke test 方法，并记录每个方法的实验目的、结果、对比和结论。

### 2026-06-03：更新 E1 总文档中的旧命名

本次同步更新了 E1 相关总文档中的旧脚本名、旧结果目录名和旧公开方法名。

统一后的公开方法名：

- `manual_full`
- `manual_3d`
- `llm_only`
- `manual_full_llm_fusion`

其中 `manual_full_llm_fusion` 用于强调该方法是原始完整手工模板文本原型与 LLM 描述文本原型的加权融合，而不是简单追加 LLM 文本。

更新范围：

- `docs/decisions/D002_prompt_source_policy.md`
- `docs/experiments/E1_text_prototype_enhancement/log.md`
- `docs/experiments/E1_text_prototype_enhancement/10_historical_analysis_and_lessons.md`
- `docs/experiments/E1_text_prototype_enhancement/e1_prompt_fusion_stage_report.md`
- `docs/experiments/E1_text_prototype_enhancement/e1_experiment_standardization_plan.md`
- `docs/experiments/experiment_registry.md`
- `Point-Cache/scripts/E1_text_prototype_enhancement/README.md`

### 2026-06-03：建立 E1 共享 LLM prompt 缓存

为避免后续权重消融重复调用 LLM API，将 smoke test 中已经生成完成的 ModelNet-C LLM prompt 复制到共享缓存目录：

    Point-Cache/results/E1_text_prototype_enhancement/shared_prompts/

当前共享 prompt 文件：

    modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json

后续权重消融脚本应统一使用该共享 prompt 缓存目录作为 `--prompt-cache-dir`，确保不同融合权重使用完全相同的 LLM 描述。

### 2026-06-03：新增 E1-S1 融合权重消融计划与脚本

本次新增 E1-S1 融合权重消融计划：

    docs/experiments/E1_text_prototype_enhancement/01_s2_fusion_weight_ablation_plan.md

新增脚本：

- `01_run_ulip_modelnetc_s2_zs_fusion_weight_ablation_common.sh`
- `01_1_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w090_010.sh`
- `01_2_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w085_015.sh`
- `01_3_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w075_025.sh`
- `01_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w050_050.sh`

这些脚本统一读取共享 LLM prompt 缓存：

    Point-Cache/results/E1_text_prototype_enhancement/shared_prompts/

目的：

- 不重复调用 LLM API；
- 保证所有权重实验使用相同 LLM 描述；
- 只考察融合权重对结果的影响。

### 2026-06-03：完成 E1-S1 融合权重消融分析

完成 `manual_full_llm_fusion` 在 ULIP × ModelNet-C severity=2 zero-shot 设置下的四组融合权重对比：

- 0.90:0.10：48.41
- 0.85:0.15：48.62
- 0.75:0.25：48.88
- 0.50:0.50：48.37

baseline zero-shot `manual_full` 为 47.68。

主要结论：

- 四种融合权重全部超过 baseline；
- 0.75:0.25 平均准确率最高；
- 0.85:0.15 更稳健；
- 0.50:0.50 说明 LLM 权重过高会在部分噪声类损坏上带来下降；
- 后续 all35 完整验证建议优先比较 0.75:0.25 和 0.85:0.15。

新增分析文档：

    docs/experiments/E1_text_prototype_enhancement/01_s2_fusion_weight_ablation_analysis.md

### 2026-06-03：完成 E1-S1 融合权重消融分析

完成 `manual_full_llm_fusion` 在 ULIP × ModelNet-C severity=2 zero-shot 设置下的四组融合权重对比：

- 0.90:0.10：48.41
- 0.85:0.15：48.62
- 0.75:0.25：48.88
- 0.50:0.50：48.37

baseline zero-shot `manual_full` 为 47.68。

主要结论：

- 四种融合权重全部超过 baseline；
- 0.75:0.25 平均准确率最高；
- 0.85:0.15 更稳健；
- 0.50:0.50 说明 LLM 权重过高会在部分噪声类损坏上带来下降；
- 后续 all35 完整验证建议优先比较 0.75:0.25 和 0.85:0.15。

新增分析文档：

    docs/experiments/E1_text_prototype_enhancement/01_s2_fusion_weight_ablation_analysis.md
