# E1-S1：manual_full_llm_fusion 融合权重消融计划

更新日期：2026-06-16

## 1. 实验目的

E1 smoke test 已经表明：

- `manual_full` 平均准确率为 47.68；
- `llm_only` 平均准确率为 39.30；
- `manual_full_llm_fusion` 在 0.75:0.25 权重下达到 48.88，超过 baseline 1.20。

这说明 LLM 生成描述不适合直接替代原始手工模板，但适合作为补充语义分支与原始完整手工模板融合。

当前问题是：

    manual_full : LLM = 0.75 : 0.25

不一定是最优权重。

由于 `llm_only` 明显低于 `manual_full`，LLM 分支权重过高可能会引入文本原型偏移。因此需要进行融合权重消融，寻找更稳定的融合比例。

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| 阶段 | E1-S1 |
| 实验类型 | 融合权重消融 |
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 损坏强度 | severity=2 |
| 损坏类型数量 | 7 |
| 任务设置 | zero-shot |
| 方法 | manual_full_llm_fusion |
| LLM 描述数量 | 每类 10 条 |
| LLM prompt 来源 | `Point-Cache/llm/e1_prompt_bank/`，不重新生成 |

## 3. 共享 LLM prompt 缓存

本阶段所有权重实验统一使用同一份 LLM 生成描述：

    Point-Cache/llm/e1_prompt_bank/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json

原因：

- 避免不同权重实验重复调用 LLM API；
- 避免不同权重使用不同随机生成描述导致结果不可比；
- 保证权重消融只考察融合权重变化，而不引入 prompt 内容变化。

## 4. 候选权重

| 实验编号 | manual_full 权重 | LLM 权重 | 说明 |
|---|---:|---:|---|
| 01_1 | 0.90 | 0.10 | 更保守地引入 LLM 描述 |
| 01_2 | 0.85 | 0.15 | 中等保守融合 |
| 01_5 | 0.80 | 0.20 | 2026-06-16 补充的中间权重 |
| 01_3 | 0.75 | 0.25 | smoke test 中已取得正结果的默认融合权重 |
| 01_4 | 0.50 | 0.50 | 较高 LLM 权重，用于观察是否产生文本原型偏移 |

## 5. 预期判断

如果 0.90:0.10 或 0.85:0.15 优于 0.75:0.25，说明 LLM 描述应作为弱补充分支，而不是较强替代分支。

如果 0.75:0.25 仍然最好，说明当前补充强度合适。

如果 0.50:0.50 明显下降，说明 LLM 权重过高会削弱 manual_full 的稳定语义锚点。

## 6. 脚本命名

脚本目录：

    Point-Cache/scripts/E1_text_prototype_enhancement/

公共脚本：

    01_run_ulip_modelnetc_s2_zs_fusion_weight_ablation_common.sh

权重消融脚本：

| 脚本 | 权重 |
|---|---|
| 01_1_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w090_010.sh | 0.90:0.10 |
| 01_2_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w085_015.sh | 0.85:0.15 |
| 01_5_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020.sh | 0.80:0.20 |
| 01_3_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w075_025.sh | 0.75:0.25 |
| 01_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w050_050.sh | 0.50:0.50 |

## 7. 结果目录

结果统一保存到：

    Point-Cache/results/E1_text_prototype_enhancement/

对应结果目录：

| 实验编号 | 结果目录 |
|---|---|
| 01_1 | 01_1_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w090_010 |
| 01_2 | 01_2_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w085_015 |
| 01_5 | 01_5_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020 |
| 01_3 | 01_3_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w075_025 |
| 01_4 | 01_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w050_050 |

## 8. 当前不做的事情

本阶段暂不做：

- ModelNet-C all35；
- global cache；
- local cache；
- ScanObjectNN / ScanObjectNN-C；
- 多 backbone；
- 重新生成 LLM prompt。

上述内容应在最佳权重确定后再进行。
