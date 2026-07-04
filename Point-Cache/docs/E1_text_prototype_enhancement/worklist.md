# E1 Worklist

更新日期：2026-06-18

## 执行规则

1. 跑代码和实验命令由用户手动执行。
2. 我每次只完成一个 worklist 项。
3. 每个 worklist 项完成后必须等待用户确认，才能进入下一项。
4. E1 不重跑 E0 baseline，只引用 E0 baseline。
5. 只有需要执行代码、脚本或 LLM 生成命令的项目才分配 E1 实验编号。

## 准备任务，不占用实验编号

| 状态 | 工作项 | 产物 |
|---|---|---|
| done | 建立新版 E1 文档、命名和路径规范 | `Point-Cache/docs/E1_text_prototype_enhancement/` |
| done | 审计 `manual_full` 模板组成 | `manual_template_audit.md` |
| done | 固定 E1_36 为当前正式候选配置 | `current_candidate_config.md` |

## 代码运行项，使用实验编号

| 编号 | 状态 | 工作项 | 产物 |
|---|---|---|---|
| E1_01 | done | 准备并由用户执行 10 prompts 生成：4 image + 4 pointcloud + 2 bridge | prompt JSON + generation script |
| E1_02 | done | 准备并由用户执行 15 prompts 生成：10 image + 5 pointcloud | prompt JSON + generation script |
| E1_03 | done | 准备并由用户执行 15 prompts 生成：5 image + 10 pointcloud | prompt JSON + generation script |
| E1_04 | done | 准备并由用户执行 15 prompts 生成：12 image + 3 pointcloud | prompt JSON + generation script |
| E1_10 | done | 准备并由用户执行 ModelNet-C 全量权重消融 90:10 | fusion result |
| E1_11 | pending | 准备并由用户执行 ModelNet-C 全量权重消融 85:15 | fusion result |
| E1_12 | pending | 准备并由用户执行 ModelNet-C 全量权重消融 80:20 | fusion result |
| E1_13 | done | 准备并由用户执行 ModelNet-C 全量权重消融 75:25 | fusion result |
| E1_14 | pending | 准备并由用户执行 ModelNet-C 全量权重消融 50:50 | fusion result |
| E1_13 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 诊断脚本 75:25 | diagnostic result |
| E1_20 | done | 准备并由用户执行 ModelNet-C 全量 prompt 消融：15 prompts, 10 image + 5 pointcloud, 75:25 | composition result |
| E1_21 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 prompt 诊断：15 prompts, 5 image + 10 pointcloud, 75:25 | diagnostic result |
| E1_21 | pending | 准备并由用户执行 ModelNet-C 全量 prompt 消融：15 prompts, 5 image + 10 pointcloud, 75:25 | composition result |
| E1_22 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 prompt 诊断：15 prompts, 12 image + 3 pointcloud, 75:25 | diagnostic result |
| E1_23 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重微调诊断：15 prompts, 10 image + 5 pointcloud, 73:27 | diagnostic result |
| E1_24 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重微调诊断：15 prompts, 10 image + 5 pointcloud, 74:26 | diagnostic result |
| E1_25 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重微调诊断：15 prompts, 10 image + 5 pointcloud, 73.5:26.5 | diagnostic result |
| E1_26 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重微调诊断：15 prompts, 10 image + 5 pointcloud, 72.5:27.5 | diagnostic result |
| E1_27 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重微调诊断：15 prompts, 10 image + 5 pointcloud, 72:28 | diagnostic result |
| E1_28 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重微调诊断：15 prompts, 10 image + 5 pointcloud, 71.5:28.5 | diagnostic result |
| E1_29 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重微调诊断：15 prompts, 10 image + 5 pointcloud, 71:29 | diagnostic result |
| E1_30 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重微调诊断：15 prompts, 10 image + 5 pointcloud, 70:30 | diagnostic result |
| E1_31 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重微调诊断：15 prompts, 10 image + 5 pointcloud, 69:31 | diagnostic result |
| E1_32 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重微调诊断：15 prompts, 10 image + 5 pointcloud, 68:32 | diagnostic result |
| E1_33 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重远点诊断：15 prompts, 10 image + 5 pointcloud, 60:40 | diagnostic result |
| E1_34 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重远点诊断：15 prompts, 10 image + 5 pointcloud, 55:45 | diagnostic result |
| E1_35 severity2 diagnostic | done | 准备并由用户执行 ModelNet-C severity=2 权重细化诊断：15 prompts, 10 image + 5 pointcloud, 57.5:42.5 | diagnostic result |
| E1_36 | done | 准备并由用户执行 ModelNet-C full 验证：15 prompts, 10 image + 5 pointcloud, 60:40 | full validation result |
| E1_40 | done | 准备并由用户执行 clean ModelNet 验证：15 prompts, 10 image + 5 pointcloud, 60:40 | clean ModelNet verification result |
| E1_50 | pending | 准备并由用户执行 ScanObjectNN 验证 | verification result |
| E1_60 | pending | 准备并由用户执行 ScanObjectNN-C 验证 | verification result |

## 汇总任务，不占用实验编号

| 状态 | 工作项 | 产物 |
|---|---|---|
| done | 拆分 E1 结果记录格式：一个实验一个文档，`result_summary.md` 只保留索引 | `E1_10_modelnet_c_full_manual90_llm10.md`, `E1_13_modelnet_c_severity2_manual75_llm25.md`, `result_summary.md` |
| pending | 汇总四数据集结果并冻结最终配置 | `result_summary.md` |
