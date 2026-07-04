# E1 文本原型增强实验规范化计划

更新日期：2026-06-03

## 1. 规范化目标

E1 当前已经完成了一个重要的最小验证：在 ULIP × ModelNet-C × severity=2 × zero-shot 设置下，原始完整手工模板与大模型类别描述的融合方法超过了原始 baseline。

但是，当前 E1 的脚本、结果目录和文档命名还存在以下问题：

1. 部分脚本名称包含 `single_gpu`，但实际脚本已经支持用户传入 GPU 编号，因此不应再把 `single_gpu` 写入文件名；
2. 部分实验名称仍使用早期命名，例如 `llm_only`、`manual_full_llm_fusion`，不够直观；
3. 结果目录中存在早期 all35 试跑结果，会干扰当前 smoke test 结果理解；
4. `manual_3d` 虽然已经证明效果较差，但仍应作为失败消融被保留，而不是作为后续 active 主线；
5. 每个实验结果应有对应分析文档，避免只有 summary.csv 而缺少解释；
6. E1 当前只是最小验证，不应与完整正式实验混淆。

因此，E1 需要进行一次命名、目录、结果和文档规范化。

## 2. 当前 E1 的四种文本方法

E1 统一保留四种文本原型构建方法。

| 方法编号 | 中文名称 | 建议实验名 | 当前/内部含义 |
|---|---|---|---|
| M1 | 原始完整手工模板 | `manual_full` | Point-Cache 原始完整文本模板，也是 E0 baseline 的 zero-shot 文本设置 |
| M2 | 删除 2D 图像风格模板后的 3D 手工模板 | `manual_3d` | 失败消融，用于证明不能简单删除 2D 图像风格提示词 |
| M3 | 只使用大模型类别描述 | `llm_only` | 只使用大模型生成的类别级多视角描述 |
| M4 | 原始完整手工模板与 LLM 描述融合 | `manual_full_llm_fusion` | 当前 E1 主方法。分别构造 manual_full 文本原型和 LLM 描述文本原型，再进行加权融合 |

说明：

- `manual_full` 是 baseline，不是新方法；
- `manual_3d` 只作为失败消融保留，不作为后续主线；
- `llm_only` 用于验证大模型描述是否可以独立替代手工模板；
- `manual_full_llm_fusion` 是当前最重要的方法名称，强调该方法不是简单追加 LLM 文本，而是将原始手工模板文本原型与 LLM 描述文本原型进行加权融合；
- 代码内部可暂时保留已有参数名，但脚本、结果目录和文档应逐步使用更直观的方法名。

## 3. 当前 smoke test 的定义

当前已经完成的 E1 实验应被定义为 smoke test，而不是完整验证。

smoke test 的含义：

- 用一个较小实验设置快速验证想法是否成立；
- 主要用于检查代码路径、方法趋势和初步有效性；
- 不能替代完整实验；
- 可以作为后续正式实验设计依据。

当前 E1 smoke test 设置为：

| 项目 | 设置 |
|---|---|
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 损坏强度 | severity=2 |
| 损坏类型 | 7 类 |
| 任务设置 | zero-shot |
| 对比方法 | manual_full、manual_3d、llm_only、manual_full_llm_fusion |

当前 smoke test 不包含：

- clean ModelNet40；
- clean ScanObjectNN；
- ScanObjectNN-C；
- ModelNet-C 的 all35 完整损坏设置；
- global cache；
- global cache + local cache；
- 多 backbone 对比。

## 4. 完整验证应如何理解

用户提出的完整验证方向基本正确，但需要更准确地表述。

完整验证不应简单写成“四个数据集都跑 7×5 损坏类型”，因为 clean 数据集没有损坏类型。

更准确的完整验证矩阵应为：

### 4.1 数据集维度

| 数据集类型 | 数据集 | 是否有 7×5 损坏类型 |
|---|---|---|
| clean | ModelNet40 | 否 |
| corruption | ModelNet-C | 是 |
| clean | ScanObjectNN | 否 |
| corruption | ScanObjectNN-C | 是 |

### 4.2 任务设置维度

与 E0 baseline 对齐，至少包含三种设置：

| 设置名称 | 含义 |
|---|---|
| zero-shot | 只使用文本原型，不使用缓存 |
| zero-shot + global cache | 文本原型 + 全局缓存 |
| zero-shot + global cache + local cache | 文本原型 + 全局缓存 + 局部缓存 |

### 4.3 文本方法维度

| 方法 | 是否继续 |
|---|---|
| manual_full | 是 |
| manual_3d | 只作为失败消融，不建议完整大规模跑 |
| llm_only | 作为对照，可选择性跑 |
| manual_full_llm_fusion | 是，主方法 |

### 4.4 是否有必要全部跑完

不建议现在立即全部跑完。

原因：

1. 完整矩阵代价很大；
2. manual_3d 已经明显失败，不值得在所有设置下大规模复跑；
3. llm_only 明显低于 manual_full，主要作为对照；
4. 当前最有价值的是 manual_full_llm_fusion；
5. 在完整 all35 之前，应先做融合权重消融，确定更优的 LLM 权重。

建议采用分阶段验证：

| 阶段 | 目的 | 是否当前执行 |
|---|---|---|
| S0 smoke test | 验证方向是否成立 | 已完成 |
| S1 权重消融 | 找到更优融合比例 | 下一步优先 |
| S2 ModelNet-C all35 | 在完整损坏设置下验证主方法 | 权重确定后执行 |
| S3 cache 设置验证 | 验证 global cache 和 local cache 下是否仍有效 | all35 有效后执行 |
| S4 跨数据集验证 | 扩展到 ScanObjectNN / ScanObjectNN-C | 后续执行 |
| S5 多 backbone 验证 | 扩展到 ULIP2 / OpenShape / Uni3D | 资源允许时执行 |

## 5. 建议的脚本命名规范

### 5.1 不再使用 single_gpu

以后脚本名不再包含：

    single_gpu

原因：

- 默认就是单卡运行；
- 用户可以通过脚本最后一个参数选择物理 GPU 编号；
- 文件名中写 single_gpu 会造成冗余。

### 5.2 smoke test 脚本从 00 编号

当前 E1 severity=2 最小验证属于 smoke test，因此统一使用 `00_*` 编号。

建议脚本目录：

    Point-Cache/scripts/E1_text_prototype_enhancement/

建议保留脚本：

| 脚本 | 作用 |
|---|---|
| `00_0_llm_prompt_generation_smoke.sh` | 大模型描述生成最小测试 |
| `00_run_ulip_modelnetc_s2_zs_text_method_smoke_common.sh` | smoke test 公共脚本 |
| `00_1_ulip_modelnetc_s2_zs_manual_full_smoke.sh` | 原始完整手工模板 |
| `00_2_ulip_modelnetc_s2_zs_manual_3d_smoke.sh` | 删除 2D 模板后的 3D 手工模板 |
| `00_3_ulip_modelnetc_s2_zs_llm_only_smoke.sh` | 只使用大模型类别描述 |
| `00_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_smoke.sh` | 原始完整模板 + 大模型描述融合 |

说明：

- `00_2` 是失败消融，但建议保留脚本或至少保留文档记录；
- 后续如果不想继续运行 `manual_3d`，可以将脚本移动到 archive，但当前阶段建议先保留，方便复现已完成结果。

## 6. 建议的结果目录命名规范

结果目录应与脚本一一对应。

建议结果目录：

    Point-Cache/results/E1_text_prototype_enhancement/

应保留：

| 结果目录 | 对应方法 |
|---|---|
| `00_1_ulip_modelnetc_s2_zs_manual_full_smoke/` | manual_full |
| `00_2_ulip_modelnetc_s2_zs_manual_3d_smoke/` | manual_3d |
| `00_3_ulip_modelnetc_s2_zs_llm_only_smoke/` | llm_only |
| `00_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_smoke/` | manual_full_llm_fusion |

应清理或归档：

| 当前目录 | 处理建议 |
|---|---|
| `01_1_ulip_modelnetc_all35_zs_manual_full/` | 早期 all35 试跑，不属于当前 smoke test，建议删除或移入 archive |
| 旧命名的结果目录 | 重命名为 00_* smoke test 规范命名 |

## 7. 建议的文档组织

目录：

    docs/experiments/E1_text_prototype_enhancement/

文档保留原则：

- 不删除已有 `log.md`、`analysis.md` 和 `e1_prompt_fusion_stage_report.md`；
- `log.md` 用于记录完整实验过程、报错、修复和阶段节点；
- `analysis.md` 用于沉淀实验分析和方法判断；
- `e1_prompt_fusion_stage_report.md` 用于保留当前阶段性正结果；
- `e1_experiment_standardization_plan.md` 用于约束后续脚本、结果目录和文档命名；
- 如果旧文档中存在与当前规范冲突的表述，优先选择“修订和追加说明”，不直接删除历史记录。

建议结构：

| 文件或目录 | 作用 |
|---|---|
| `log.md` | E1 过程日志 |
| `analysis.md` | E1 总体分析 |
| `e1_experiment_standardization_plan.md` | 本规范化计划 |
| `e1_prompt_fusion_stage_report.md` | 当前融合正结果阶段性报告 |
| `smoke_tests/` | 每个 smoke test 的独立分析文档 |

建议新增：

    docs/experiments/E1_text_prototype_enhancement/

其中每个实验一个分析文件：

| 文件 | 对应实验 |
|---|---|
| `00_1_manual_full_smoke_analysis.md` | manual_full |
| `00_2_manual_3d_smoke_analysis.md` | manual_3d |
| `00_3_llm_only_smoke_analysis.md` | llm_only |
| `00_4_manual_full_llm_fusion_smoke_analysis.md` | manual_full_llm_fusion |
| `00_smoke_test_summary.md` | 四种方法总表与结论 |

## 8. 当前 smoke test 结果

| 方法 | 平均准确率 | 结论 |
|---|---:|---|
| manual_full | 47.68 | baseline，说明 E1 新接口没有破坏原始模板路径 |
| manual_3d | 35.63 | 明显下降，证明不能简单删除 2D 图像风格模板 |
| llm_only | 39.30 | 优于 manual_3d，但不能替代 manual_full |
| manual_full_llm_fusion | 48.88 | 当前最优，超过 baseline 1.20 |

核心结论：

    大模型描述不适合直接替代原始手工模板，
    但适合作为补充语义分支与原始模板融合。

## 9. 后续任务计划

### Task 1：写入本规范化计划

状态：待完成。

产物：

    docs/experiments/E1_text_prototype_enhancement/e1_experiment_standardization_plan.md

### Task 2：重命名 smoke test 脚本

目标：

- 删除文件名中的 `single_gpu`；
- 使用 `00_*_smoke` 编号；
- 只保留四种核心方法和一个公共脚本；
- 删除旧命名脚本。

### Task 3：清理结果目录

目标：

- 结果目录与脚本一一对应；
- 删除或归档早期 all35 试跑目录；
- 将已有正式 smoke test 结果重命名为 00_* 规范命名。

### Task 4：补充每个 smoke test 的独立分析文档

目标：

在以下目录中写入每个实验的结果分析：

    docs/experiments/E1_text_prototype_enhancement/

包括：

- manual_full；
- manual_3d；
- llm_only；
- manual_full_llm_fusion；
- 四方法总表。

### Task 5：更新总文档

目标：

同步更新：

- `docs/experiments/E1_text_prototype_enhancement/log.md`
- `docs/experiments/E1_text_prototype_enhancement/10_historical_analysis_and_lessons.md`
- `docs/experiments/experiment_registry.md`
- `Point-Cache/scripts/E1_text_prototype_enhancement/README.md`

### Task 6：检查代码和脚本

目标：

- Python 文件通过 py_compile；
- shell 脚本通过 bash -n；
- 不检查 results，因为 results 被 .gitignore 忽略。

### Task 7：Git 提交

建议提交信息：

    docs: standardize E1 smoke test organization

## 10. 当前暂不做的事情

以下事情暂时不做：

1. 不立即跑完整 all35；
2. 不立即跑四个数据集；
3. 不立即跑 global cache 和 local cache；
4. 不再扩展 manual_3d；
5. 不把 results 加入 Git；
6. 不把 API key 加入 Git；
7. 不把大模型生成 prompt JSON 加入 Git。

## 11. 下一步优先级

规范化完成后，下一步实验优先级为：

1. 对 `manual_full_llm_fusion` 做融合权重消融；
2. 选择最佳权重；
3. 用最佳权重跑 ModelNet-C all35 zero-shot；
4. 若 all35 仍有效，再扩展到 cache 设置；
5. 最后再考虑跨数据集和多 backbone。

## 12. LLM 生成描述的长期存放位置与命名规范

当前 Point-Cache 原始代码中，`Point-Cache/llm/` 目录已经包含若干离线 prompt bank 文件，例如：

- `mn40_gpt4_prompts.json`
- `mn40_gpt35_prompts.json`
- `mn40_pointllm_prompts.json`
- `sonn_gpt4_prompts.json`
- `sonn_gpt35_prompts.json`
- `sonn_pointllm_prompts.json`

这些文件属于 Point-Cache 原始 prompt 资源，命名中通常包含：

- 数据集简称，例如 `mn40`、`sonn`；
- 生成模型简称，例如 `gpt4`、`gpt35`、`pointllm`；
- prompt 文件类型，例如 `prompts.json`。

E1 当前由 LLM 动态生成的类别描述与 Point-Cache 原始 prompt bank 存在差异：

1. E1 生成描述服务于当前实验流程；
2. E1 prompt JSON 包含生成模型、prompt mode、类别列表、缓存状态等实验元数据；
3. E1 生成结果目前属于实验产物，而不是稳定的内置 prompt bank；
4. 后续可能使用不同大模型生成描述，例如 DeepSeek、Qwen、GPT、Kimi 等。

因此，当前阶段不立即将 E1 生成的 prompt JSON 固定放入 `Point-Cache/llm/`。

当前原则：

- 实验运行中生成的 prompt JSON 继续放在 `Point-Cache/results/E1_text_prototype_enhancement/.../prompts/`；
- 该目录属于实验结果目录，被 `.gitignore` 忽略，不提交；
- 等 E1 方法稳定后，再从结果目录中选择最终版本进行归档；
- 若需要归档，应新建独立目录，而不是混入 Point-Cache 原始 prompt 文件中。

建议的长期归档目录：

    Point-Cache/llm/e1_prompt_bank/

建议命名规则：

    {dataset}_{llm_name}_{prompt_mode}_{prompt_count}_prompts.json

其中：

- `dataset` 表示数据集，例如 `modelnet40`、`modelnet_c`、`scanobjnn`、`scanobjnn_c`；
- `llm_name` 表示生成模型，例如 `ds_v4pro`、`gpt4o`、`qwen_max`；
- `prompt_mode` 表示描述策略，例如 `mixed_2d3d`；
- `prompt_count` 表示每类描述数量，例如 `10`。

当前 DeepSeek V4 Pro 可暂定命名为：

    ds_v4pro

例如：

    modelnet_c_ds_v4pro_mixed_2d3d_10_prompts.json

该任务暂不立即执行。它应在 E1 smoke test 规范化、脚本命名规范化、结果目录清理和分析文档补齐之后再讨论。

## 13. 更新后的任务队列

当前任务优先级调整为：

1. 完成 smoke test 脚本命名规范化；
2. 清理 E1 smoke test 结果目录，使结果目录与脚本一一对应；
3. 为每个 smoke test 补充独立分析文档；
4. 更新 `log.md`、`analysis.md`、`experiment_registry.md` 和脚本 README；
5. 完成代码与脚本检查；
6. Git 提交；
7. 再讨论 LLM prompt bank 是否归档到 `Point-Cache/llm/e1_prompt_bank/`，以及最终命名规范。

## 14. 脚本说明与运行方式文档要求

E1 脚本目录下的 `README.md` 必须明确说明每个脚本的用途和运行方式。

至少需要包含：

1. 当前阶段是否为 smoke test；
2. 每个脚本对应的文本方法；
3. 每个脚本的实验目的；
4. 每个脚本的运行命令；
5. 每个脚本对应的结果目录；
6. 是否会调用 LLM API；
7. 是否会读取已有 prompt 缓存；
8. 如何指定物理 GPU；
9. 如何修改融合权重；
10. 哪些脚本通常由用户直接运行，哪些脚本只是公共脚本。

当前需要维护的脚本说明文件：

    Point-Cache/scripts/E1_text_prototype_enhancement/README.md

该 README 是 E1 smoke test 的主要复现实验入口说明。


## 15. E1 共享 LLM prompt 缓存

为避免后续权重消融和完整实验重复调用 LLM API，E1 采用共享 prompt 缓存机制。

当前共享缓存目录：

    Point-Cache/llm/e1_prompt_bank/

当前共享 prompt 文件：

    Point-Cache/llm/e1_prompt_bank/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json

说明：

- 该文件由 smoke test 中已生成的 LLM prompt 复制而来；
- 后续权重消融应统一读取该共享 prompt；
- 不同权重实验不应重新生成 LLM prompt；
- 旧 `Point-Cache/results/E1_text_prototype_enhancement/shared_prompts/` 仅作为历史结果追溯；
- 当前可归档 prompt bank 固定为 `Point-Cache/llm/e1_prompt_bank/`；
- 长期归档时，当前 DeepSeek V4 Pro 可命名为 `ds_v4pro`，例如 `modelnet_c_ds_v4pro_mixed_2d3d_10_prompts.json`。

当前阶段为了兼容代码自动读取逻辑，共享缓存文件仍保留生成器默认文件名：

    modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json

后续如果改成 `ds_v4pro` 命名，需要同步修改 prompt 读取逻辑或增加显式 prompt 文件参数。
