# E1 Naming And Paths

更新日期：2026-06-17

## 固定路径

```text
Point-Cache/docs/E1_text_prototype_enhancement/
Point-Cache/scripts/E1_text_prototype_enhancement/
Point-Cache/runners/E1_text_prototype_enhancement/
Point-Cache/results/E1_text_prototype_enhancement/
Point-Cache/llm/
```

## 术语规范

| 不使用 | 使用 |
|---|---|
| `s2` / `severity2` | `modelnet_c_full` |
| `2d_to_3d_2_to_1` | `10_image_5_pointcloud` |
| `2d_to_3d_1_to_2` | `5_image_10_pointcloud` |
| `w90_10` | `manual90_llm10` |
| `w80_20` | `manual80_llm20` |
| `shared_prompts` | `Point-Cache/llm/` 下的正式 LLM description JSON |

## 脚本目录

```text
Point-Cache/scripts/E1_text_prototype_enhancement/
├── common/
├── prompt_generation/
├── fusion_weight_ablation/
├── prompt_composition_ablation/
├── modelnet_c_full_validation/
├── modelnet_clean_validation/
├── scanobjnn/
└── scanobjnn_c/
```

父目录表达实验类型。只有需要执行代码的脚本保留 E1 实验编号；文档整理、模板审计和命名规范不占用编号。

## 实验说明规范

给出任何实验执行命令前，必须先写基本实验设置：

```text
实验编号
实验目的
数据集与评估范围
Backbone
任务设置
文本原型方法
LLM prompt 文件与组成
融合权重
结果目录
```

每个需要运行代码的实验必须有一个独立实验文档，命名使用：

```text
E1_xx_{dataset}_{scope}_{setting}.md
```

示例：

```text
E1_10_modelnet_c_full_manual90_llm10.md
E1_13_modelnet_c_severity2_manual75_llm25.md
```

`result_summary.md` 只记录结果索引和总体表格，不写单实验的完整分析。

## LLM 描述文件

LLM 描述文件直接放在 `Point-Cache/llm/`。

```text
modelnet_c_llm_descriptions_deepseek_v4pro_10_prompts_4_image_4_pointcloud_2_bridge.json
modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json
modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_5_image_10_pointcloud.json
modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_12_image_3_pointcloud.json
```

命名字段顺序：

```text
{dataset}_llm_descriptions_{provider_model}_{prompt_count}_prompts_{composition}.json
```

## 脚本文件名

Prompt 生成脚本：

```text
prompt_generation/
├── E1_01_10_prompts_4_image_4_pointcloud_2_bridge.sh
├── E1_02_15_prompts_10_image_5_pointcloud.sh
├── E1_03_15_prompts_5_image_10_pointcloud.sh
└── E1_04_15_prompts_12_image_3_pointcloud.sh
```

融合权重脚本：

```text
fusion_weight_ablation/
├── E1_10_modelnet_c_full_manual90_llm10.sh
├── E1_11_modelnet_c_full_manual85_llm15.sh
├── E1_12_modelnet_c_full_manual80_llm20.sh
├── E1_13_modelnet_c_full_manual75_llm25.sh
├── E1_13_modelnet_c_severity2_manual75_llm25.sh
└── E1_14_modelnet_c_full_manual50_llm50.sh
```

Prompt 组成脚本：

```text
prompt_composition_ablation/
├── E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25.sh
├── E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25.sh
├── E1_21_modelnet_c_full_15_prompts_5_image_10_pointcloud_manual75_llm25.sh
└── E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25.sh
```

跨数据集验证脚本：

```text
modelnet_c_full_validation/
└── E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40.sh

modelnet_clean_validation/
└── E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40.sh
```

## 结果目录

```text
Point-Cache/results/E1_text_prototype_enhancement/
├── E1_10_modelnet_c_full_manual90_llm10/
├── E1_11_modelnet_c_full_manual85_llm15/
├── E1_12_modelnet_c_full_manual80_llm20/
├── E1_13_modelnet_c_full_manual75_llm25/
├── E1_14_modelnet_c_full_manual50_llm50/
├── E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25/
├── E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25/
├── E1_21_modelnet_c_full_15_prompts_5_image_10_pointcloud_manual75_llm25/
├── E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25/
├── E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40/
└── E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40/
```
