# E1 Text Prototype Enhancement

更新日期：2026-06-18

## 研究边界

E1 只研究在 E0 baseline 之上的文本原型增强：

- 是否用 LLM 类别描述增强 `manual_full` 手工模板；
- LLM 描述数量应该是多少；
- LLM 描述中 image-style 与 pointcloud-style 的比例应该是多少；
- `manual_full:LLM` 融合权重应该是多少；
- 最佳 E1 设置能否推广到 ModelNet、ModelNet-C、ScanObjectNN 和 ScanObjectNN-C。

E1 不重新定义、不重新编号、不重新运行 E0 baseline。`manual_full`、`manual_3d`、`llm_only` 等 baseline 结果只在 E1 文档中作为 E0 引用值出现。

## 正式目录

```text
Point-Cache/docs/E1_text_prototype_enhancement/
Point-Cache/scripts/E1_text_prototype_enhancement/
Point-Cache/runners/E1_text_prototype_enhancement/
Point-Cache/results/E1_text_prototype_enhancement/
Point-Cache/llm/
```

LLM 生成描述直接保存到 `Point-Cache/llm/`，不再放入 E1 结果目录或额外 prompt bank 子目录。

## 文档入口

| 文件 | 作用 |
|---|---|
| `experiment_plan.md` | E1 完整实验方案 |
| `naming_and_paths.md` | 文件、脚本、结果和 prompt 命名规范 |
| `worklist.md` | 当前一步一步推进的执行清单 |
| `manual_template_audit.md` | `manual_full` 模板审计记录 |
| `prompt_generation_protocol.md` | LLM 描述生成规则 |
| `current_candidate_config.md` | 当前正式候选配置，固定 E1_36 作为后续跨数据集验证默认设置 |
| `modelnet_c_protocol.md` | ModelNet-C 实验协议 |
| `modelnet_c_full_fusion_protocol.md` | ModelNet-C full 融合权重公共协议 |
| `E1_10_modelnet_c_full_manual90_llm10.md` | E1_10 单实验设置、命令、结果路径和分析 |
| `E1_13_modelnet_c_severity2_manual75_llm25.md` | E1_13 severity=2 诊断单实验设置、命令和结果记录 |
| `E1_13_modelnet_c_full_manual75_llm25.md` | E1_13 full 单实验设置、命令和待分析记录 |
| `E1_20_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual75_llm25.md` | E1_20 prompt 数量与组成消融设置、命令和待分析记录 |
| `E1_21_modelnet_c_severity2_15_prompts_5_image_10_pointcloud_manual75_llm25.md` | E1_21 severity=2 prompt 组成诊断设置、命令和待分析记录 |
| `E1_22_modelnet_c_severity2_15_prompts_12_image_3_pointcloud_manual75_llm25.md` | E1_22 severity=2 prompt 组成诊断设置、命令和待分析记录 |
| `E1_23_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual73_llm27.md` | E1_23 severity=2 融合权重微调诊断设置、命令和结果记录 |
| `E1_24_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual74_llm26.md` | E1_24 severity=2 融合权重微调诊断设置、命令和结果记录 |
| `E1_33_modelnet_c_severity2_15_prompts_10_image_5_pointcloud_manual60_llm40.md` | E1_33 当前采用版本设置、命令、结果和采用理由 |
| `E1_36_modelnet_c_full_15_prompts_10_image_5_pointcloud_manual60_llm40.md` | E1_36 使用 E1_33 设置运行完整 ModelNet-C 的实验说明 |
| `E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40.md` | E1_40 使用 E1_36 候选配置运行 clean ModelNet 的实验说明 |
| `modelnet_protocol.md` | ModelNet 实验协议 |
| `scanobjnn_protocol.md` | ScanObjectNN 实验协议 |
| `scanobjnn_c_protocol.md` | ScanObjectNN-C 实验协议 |
| `result_summary.md` | 正式重跑后的 E1 结果索引，不写长篇单实验分析 |

## 当前执行规则

1. 每一步只完成一个 worklist 项。
2. 跑实验命令由用户手动执行。
3. 我完成当前步骤后停止，等待用户确认，再进入下一步。
4. 任何命名、路径或脚本重写都先记录到文档，再进入实现。
