# E1 Experiment Plan

更新日期：2026-06-18

## 目标

E1 的核心问题是：LLM 生成的类别描述能否作为 `manual_full` 的补充分支，提升 3D zero-shot 文本原型质量。

E1 的变量包括：

- LLM 描述数量；
- image-style 与 pointcloud-style 描述比例；
- `manual_full:LLM` 融合权重；
- 数据集迁移能力。

## E0 对照

E1 不运行 baseline。所有 baseline 均来自 E0：

| 对照项 | 来源 | 在 E1 中的用途 |
|---|---|---|
| `manual_full` | E0 | 主 baseline |
| `manual_3d` | E0 | 解释手工模板中 3D 子集的作用 |
| `llm_only` | E0 或历史消融 | 判断 LLM 是否只能作为补充分支 |

E1 的实验对象只有 `manual_full + LLM` 融合路线。

## 当前正式候选配置

当前 E1 正式候选配置固定为 E1_36：

```text
manual_full = 0.60
LLM = 0.40
15 prompts/class = 10 image + 5 pointcloud
```

E1_36 在 ModelNet-C full 上达到 49.36，高于 E1_20 的 48.81 和 E0 baseline 的 46.85。后续 ModelNet、ScanObjectNN、ScanObjectNN-C 验证默认使用该配置。完整配置见 `current_candidate_config.md`。

## 实验编号

只有需要执行代码、脚本或 LLM 生成命令的项目才分配 E1 实验编号。文档整理、命名规范、模板审计等准备任务不占用实验编号。

| 编号范围 | 阶段 |
|---|---|
| E1_01-E1_09 | LLM 描述生成 |
| E1_10-E1_19 | ModelNet-C 全量融合权重消融 |
| E1_20-E1_29 | ModelNet-C 全量 prompt 数量与组成消融 |
| E1_40-E1_49 | ModelNet 验证 |
| E1_50-E1_59 | ScanObjectNN 验证 |
| E1_60-E1_69 | ScanObjectNN-C 验证 |

跨数据集汇总与最终配置冻结是分析/管理任务，不占用实验编号，除非后续需要专门执行统计脚本。

## 当前 ModelNet-C 全量实验

E1 不单独在某个 corruption 或某个 severity 上报告实验准确率。ModelNet-C 相关实验直接运行完整 ModelNet-C，即所有 corruption 与 severity 组合。

当前 ModelNet-C 全量设置按 35 个 corruption-severity 组合理解：

```text
7 corruption types x 5 severities = 35 evaluations
```

### Prompt 生成

| 编号 | 描述文件 | 内容 |
|---|---|---|
| E1_01 | `modelnet_c_llm_descriptions_deepseek_v4pro_10_prompts_4_image_4_pointcloud_2_bridge.json` | 每类 10 条：4 image + 4 pointcloud + 2 bridge |
| E1_02 | `modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json` | 每类 15 条：10 image + 5 pointcloud |
| E1_03 | `modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_5_image_10_pointcloud.json` | 每类 15 条：5 image + 10 pointcloud |
| E1_04 | `modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_12_image_3_pointcloud.json` | 每类 15 条：12 image + 3 pointcloud |

### 融合权重消融

固定 prompt：E1_01 的 10 prompts。

| 编号 | `manual_full:LLM` |
|---|---|
| E1_10 | 90:10 |
| E1_11 | 85:15 |
| E1_12 | 80:20 |
| E1_13 | 75:25 |
| E1_14 | 50:50 |

### Prompt 数量与组成消融

当前 E1_13 full 显示 `manual75_llm25` 优于 `manual90_llm10`，因此 prompt 数量与组成消融固定使用 `manual75_llm25`，隔离 prompt 数量和 prompt 类型比例的影响。

| 编号 | prompt 设置 | 默认权重 |
|---|---|---|
| E1_20 | 15 prompts, 10 image + 5 pointcloud | 75:25 |
| E1_21 | 15 prompts, 5 image + 10 pointcloud | 75:25 |
| E1_22 severity2 diagnostic | 15 prompts, 12 image + 3 pointcloud | 75:25 |

E1_13 作为 10 prompts, 4 image + 4 pointcloud + 2 bridge, 75:25 的数量/组成对照。

## 四数据集验证

在 ModelNet-C 全量实验中选出 1-2 个最佳 E1 配置后，再验证 ModelNet、ScanObjectNN 和 ScanObjectNN-C。

| 阶段 | 数据集 | 目的 |
|---|---|---|
| E1_40-E1_49 | ModelNet | 验证 clean ModelNet 泛化 |
| E1_50-E1_59 | ScanObjectNN | 验证真实扫描 clean 泛化 |
| E1_60-E1_69 | ScanObjectNN-C | 验证真实扫描 corruption 泛化 |

当前 E1_40 使用 E1_36 候选配置在 clean ModelNet 上做单次验证。为保证与 E0 baseline 对齐，底层 loader 仍使用 `modelnet_c`，数据文件为 `Point-Cache/data/modelnet_c/clean.h5`。
