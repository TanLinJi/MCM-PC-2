# E1 文本原型增强阶段性报告：从失败消融到有效融合

更新日期：2026-06-03

## 1. 阶段性结论

E1 的目标是研究 Point-Cache 文本原型构造方式是否可以通过更合理的类别文本描述得到增强。

截至当前阶段，ULIP × ModelNet-C severity=2 zero-shot 最小实验已经完成四组关键对比：

1. 原始完整手工模板；
2. 删除 2D 图像风格模板后的 3D 手工模板；
3. 只使用大模型动态生成的多视角类别描述；
4. 原始完整手工模板与大模型多视角类别描述加权融合。

实验结果表明：

- 简单删除 2D 图像风格模板会显著降低性能；
- 只使用大模型生成描述也不能替代原始手工模板；
- 原始完整手工模板与大模型多视角类别描述融合后，首次超过原始 baseline；
- 当前融合方案在 severity=2 平均准确率上比原始完整手工模板提高 +1.20。

因此，E1 的有效方向已经从“替换原始模板”修正为：

> 保留 Point-Cache 原始完整手工模板作为稳定的 CLIP-style 视觉语义锚点，同时引入大模型生成的多视角类别描述作为补充语义分支。

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 测试范围 | severity=2 |
| 损坏类型数量 | 7 |
| 任务 | zero-shot classification |
| 结果目录 | Point-Cache/results/E1_text_prototype_enhancement/ |
| 主要 runner | Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py |

severity=2 包含 7 个损坏类型：

- add_global_2
- add_local_2
- dropout_global_2
- dropout_local_2
- rotate_2
- scale_2
- jitter_2

## 3. 文本方案说明

### 3.1 原始完整手工模板：manual_full

代码名：manual_full

含义：

Point-Cache 原始完整手工模板集合，即 E0 baseline 默认使用的文本模板。

作用：

- 作为 E1 新接口的正确性验证；
- 作为当前所有文本增强方案的 baseline；
- 在后续融合方案中提供稳定的 CLIP-style 视觉语义锚点。

### 3.2 删除 2D 模板后的 3D 手工模板：manual_3d

代码名：manual_3d

含义：

从原始完整手工模板中删除明显 2D 图像风格模板，只保留点云/3D 相关模板。

当前结论：

该方向失败。实验显示 manual_3d 显著低于 manual_full，说明原始模板中的 2D/image-style prompts 虽然看起来不符合点云直觉，但对 ULIP 的 CLIP-style 文本空间具有重要锚定作用。

后续处理：

- 不再作为 active 主线；
- 不再继续扩展 manual3d_llm_dynamic_init；
- 保留结果作为失败但有价值的诊断证据。

### 3.3 只使用大模型动态描述：llm_dynamic_init

代码名：llm_dynamic_init

含义：

实验初始化阶段，根据候选类别名称，由大模型生成每类 10 条多视角类别描述。

当前生成策略：

- 每类 10 条；
- 部分描述偏 2D 视觉语义；
- 部分描述偏 3D 点云几何；
- 少量描述连接视觉外观与几何结构；
- 不再要求每一条句子都同时包含 2D 和 3D 信息。

当前结论：

只使用大模型描述优于 manual_3d，但仍明显低于 manual_full。这说明大模型描述具备一定补充价值，但不能直接替代原始手工模板。

### 3.4 原始模板与大模型描述融合：manualfull_llm_dynamic_init

代码名：manualfull_llm_dynamic_init

含义：

将原始完整手工模板分支与大模型多视角描述分支分别编码为文本原型，然后进行加权融合。

当前默认权重：

- manual_full 权重：0.75
- LLM 权重：0.25

当前结论：

这是目前 E1 中最有效的方案。它在 severity=2 平均准确率上首次超过 manual_full，说明大模型描述不适合替代原始模板，但适合作为补充语义分支。

## 4. 当前结果对比

| 方法 | 中文含义 | 平均准确率 |
|---|---|---:|
| manual_full | 原始完整手工模板 | 47.68 |
| manual_3d | 删除 2D 模板后的 3D 手工模板 | 35.63 |
| llm_dynamic_init | 只用大模型多视角描述 | 39.30 |
| manualfull_llm_dynamic_init | 原始模板 + 大模型描述融合 | 48.88 |

相对 manual_full：

| 方法 | 平均差值 |
|---|---:|
| manual_3d | -12.05 |
| llm_dynamic_init | -8.38 |
| manualfull_llm_dynamic_init | +1.20 |

## 5. 分损坏类型结果

| 损坏类型 | manual_full | manual_3d | llm_dynamic_init | manualfull_llm_dynamic_init |
|---|---:|---:|---:|---:|
| add_global | 34.00 | 19.49 | 26.66 | 33.55 |
| add_local | 43.92 | 31.16 | 34.44 | 44.61 |
| dropout_global | 54.70 | 45.91 | 50.69 | 57.01 |
| dropout_local | 50.57 | 38.94 | 43.92 | 53.44 |
| rotate | 55.19 | 41.82 | 46.23 | 56.36 |
| scale | 50.89 | 41.37 | 44.21 | 52.76 |
| jitter | 44.49 | 30.75 | 28.97 | 44.45 |
| 平均 | 47.68 | 35.63 | 39.30 | 48.88 |

融合方案相对 manual_full 的分项变化：

| 损坏类型 | 差值 |
|---|---:|
| add_global | -0.45 |
| add_local | +0.69 |
| dropout_global | +2.31 |
| dropout_local | +2.87 |
| rotate | +1.17 |
| scale | +1.87 |
| jitter | -0.04 |
| 平均 | +1.20 |

融合方案在 7 个损坏类型中有 5 个超过 manual_full，仅在 add_global 和 jitter 上略低，且下降幅度很小。

## 6. 关键发现

### 6.1 不能简单删除 2D 图像风格模板

manual_3d 明显低于 manual_full，说明 2D/image-style prompts 虽然不是点云描述，但它们与 ULIP 的文本编码器语义空间高度匹配，能够提供稳定的视觉语义锚点。

这解释了为什么“看起来更符合点云”的手工模板反而效果更差。

### 6.2 大模型描述不能直接替代原始模板

llm_dynamic_init 平均准确率为 39.30，明显低于 manual_full 的 47.68。

这说明大模型生成描述虽然包含类别细节和几何语义，但单独使用时不足以匹配 ULIP 原有的 CLIP-style 文本空间。

### 6.3 大模型描述适合作为补充语义分支

manualfull_llm_dynamic_init 达到 48.88，超过原始 baseline 1.20。

这说明大模型描述的价值在于补充，而不是替代。原始模板提供稳定语义中心，大模型描述提供额外类别知识和几何语义，两者融合后得到更强文本原型。

## 7. 已完成的主要代码改动

### 7.1 参数入口

文件：

- Point-Cache/utils/utils.py

新增与 E1 文本原型增强相关的参数：

- --prompt-source
- --llm-api-key-file
- --llm-provider
- --llm-model
- --dynamic-prompt-count
- --prompt-static-weight
- --prompt-dynamic-weight
- --prompt-cache-dir
- --llm-api-base-url
- --llm-temperature
- --llm-prompt-mode
- --force-regenerate-prompts
- --llm-max-retries
- --llm-repair-retries

当前 active prompt source：

- manual_full
- llm_dynamic_init
- manualfull_llm_dynamic_init

manual_3d 已不再作为后续 active 主线。

### 7.2 数据集模板选择逻辑

新增文件：

- Point-Cache/datasets/prompt_utils.py

作用：

根据 --prompt-source 返回不同文本模板来源。

当前支持：

- manual_full
- llm_dynamic_init
- manualfull_llm_dynamic_init

修改的数据集文件：

- Point-Cache/datasets/modelnet40.py
- Point-Cache/datasets/modelnet_c.py
- Point-Cache/datasets/scanobjnn.py
- Point-Cache/datasets/sonn_c.py

关键修复：

LLM 动态描述必须在读取完候选类别名称后再生成。因此模板初始化位置已经调整为读取 classnames 后执行。

### 7.3 LLM 动态描述生成模块

新增文件：

- Point-Cache/llm/e1_dynamic_prompt_generator.py

作用：

- 从本地固定 API key 文件读取 key；
- 调用 OpenAI-compatible LLM API；
- 根据候选类别名称生成类别级描述；
- 将生成结果保存为 JSON 缓存；
- 支持部分缓存续跑；
- 支持自动重试、修复式生成和兜底描述；
- 支持 10 条多视角描述生成策略。

API key 固定路径：

- Point-Cache/llm/secrets/llm_api_key.txt

该文件被 .gitignore 忽略，不提交。

### 7.4 文本原型构造

文件：

- Point-Cache/utils/utils.py

修改点：

- 将原始 clip_classifier() 拆分为更清晰的文本构造与文本编码逻辑；
- 支持手工模板列表；
- 支持按类别名称保存的大模型描述字典；
- 支持两路文本原型加权融合。

融合形式：

final_text_prototype =
static_weight * manual_full_text_prototype
+ dynamic_weight * llm_text_prototype

当前默认：

- static_weight = 0.75
- dynamic_weight = 0.25

## 8. 已完成的主要脚本

脚本目录：

- Point-Cache/scripts/E1_text_prototype_enhancement/

关键脚本：

- 00_smoke_test_llm_dynamic_prompt.sh
  - 最小 API 测试，不跑模型，只验证 API key、LLM 调用、缓存保存逻辑。

- 01_run_ulip_modelnetc_s2_zs_prompt_ablation_common.sh
  - ULIP × ModelNet-C severity=2 zero-shot 文本消融公共脚本。

- 01_0_ulip_modelnetc_s2_zs_baseline_manual_full_single_gpu.sh
  - 验证 E1 新接口不破坏原始完整手工模板 baseline。

- 01_2_ulip_modelnetc_s2_zs_dynamic_multiview_llm_descriptions_single_gpu.sh
  - 只使用大模型多视角类别描述。

- 01_4_ulip_modelnetc_s2_zs_fuse_manualfull_multiview_llm_single_gpu.sh
  - 原始完整手工模板与大模型多视角描述融合，是当前 E1 主候选方法。

## 9. 当前结果文件位置

原始完整手工模板结果：

- Point-Cache/results/E1_text_prototype_enhancement/01_0_ulip_modelnetc_s2_zs_baseline_manual_full/

只用大模型多视角描述结果：

- Point-Cache/results/E1_text_prototype_enhancement/01_2_ulip_modelnetc_s2_zs_dynamic_multiview_llm_descriptions/

原始模板 + 大模型描述融合结果：

- Point-Cache/results/E1_text_prototype_enhancement/01_4_ulip_modelnetc_s2_zs_fuse_manualfull_multiview_llm/

大模型生成描述缓存：

- Point-Cache/results/E1_text_prototype_enhancement/01_2_ulip_modelnetc_s2_zs_dynamic_multiview_llm_descriptions/prompts/
- Point-Cache/results/E1_text_prototype_enhancement/01_4_ulip_modelnetc_s2_zs_fuse_manualfull_multiview_llm/prompts/

这些结果目录被 .gitignore 忽略，不进入 Git。

## 10. 阶段性论文表述

当前阶段可以形成如下论文观点：

> 原始 Point-Cache 文本模板虽然包含大量 2D image-style prompts，但这些模板与 ULIP 的 CLIP-style 文本空间高度匹配，能够提供稳定的视觉语义锚点。直接过滤这些模板会导致明显性能下降。与此同时，LLM 生成的多视角类别描述能够提供额外的类别知识和点云几何语义，但单独使用时不足以替代原始模板。将原始完整手工模板与 LLM 多视角描述进行加权融合，可以在 ModelNet-C severity=2 zero-shot 设置下获得稳定提升。

## 11. 下一步计划

后续优先进行权重消融，寻找更优的融合比例。

候选权重：

| manual_full 权重 | LLM 权重 |
|---:|---:|
| 0.90 | 0.10 |
| 0.85 | 0.15 |
| 0.75 | 0.25 |
| 0.50 | 0.50 |

当前已完成的是：

- manual_full : LLM = 0.75 : 0.25

由于纯 LLM 描述低于 manual_full，推测更小的 LLM 权重可能更稳定，例如 0.85:0.15 或 0.90:0.10。

权重消融完成后，再选择最佳权重运行完整 all35 设置。
