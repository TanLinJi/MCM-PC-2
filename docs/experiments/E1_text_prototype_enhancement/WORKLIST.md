# E1 Restart Worklist

更新日期：2026-06-16

## 执行原则

1. 先整理代码、命名、prompt bank 和文档，再跑新实验。
2. 先做 ULIP + ModelNet-C severity=2 zero-shot 消融，再扩展 all35。
3. LLM prompt cache 固定保存在 `Point-Cache/llm/e1_prompt_bank/`。
4. 新实验默认复用缓存；只有 prompt cache 缺失时，才运行生成脚本。
5. 每个实验必须检查 `summary.csv` 行数、`status=done`、cor_type、日志路径和平均准确率。

## 当前清单

| ID | 状态 | 工作项 | 产物 |
|---|---|---|---|
| E1-R0 | done | 盘点 E1 runner/scripts/results/docs | 本 worklist 与 README |
| E1-R1 | done | 统计 `manual_full` 中 2D image-style 模板数量 | `02_prompt_bank_and_template_audit.md` |
| E1-R2 | done | 将 prompt bank 迁移到 `Point-Cache/llm/e1_prompt_bank` | 3 个已有 JSON 缓存副本 |
| E1-R3 | done | 新增 `manual_full:LLM = 0.80:0.20` 脚本 | `01_5_...w080_020.sh` |
| E1-R4 | done | 新增 15 prompts 2D:3D 比例生成脚本 | `02_1_generate...2to1.sh`, `02_2_generate...1to2.sh` |
| E1-R5 | done | 新增 15 prompts 比例消融评估脚本 | `02_3_...p15_2d3d_2to1.sh`, `02_4_...p15_2d3d_1to2.sh` |
| E1-R6 | done | 运行 `0.80:0.20` S2 zero-shot 权重消融 | Avg 48.84, `results/E1_text_prototype_enhancement/01_5_.../summary.csv` |
| E1-R7 | pending | 生成 15 prompts, 2D:3D = 2:1 | `llm/e1_prompt_bank/modelnet_c_*_multiview_2d3d_2to1_15_prompts.json` |
| E1-R8 | pending | 生成 15 prompts, 2D:3D = 1:2 | `llm/e1_prompt_bank/modelnet_c_*_multiview_2d3d_1to2_15_prompts.json` |
| E1-R9 | pending | 跑 15 prompts 2D:3D = 2:1 S2 zero-shot | `02_3_.../summary.csv` |
| E1-R10 | pending | 跑 15 prompts 2D:3D = 1:2 S2 zero-shot | `02_4_.../summary.csv` |
| E1-R11 | pending | 汇总新增 S2 消融结果，选择 all35 候选 | 新分析文档 |
| E1-R12 | pending | 设计并实现 ModelNet-C all35 zero-shot runner/script | all35 runner + scripts |

## 推荐运行顺序

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache

export E1_PYTHON_CMD="conda run -n mcmpc python"

# 1. 新增中间权重消融，复用已有 10 prompts
bash scripts/E1_text_prototype_enhancement/01_5_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020.sh 0

# 2. 生成 15 prompts 两种比例
bash scripts/E1_text_prototype_enhancement/02_1_generate_modelnetc_llm_prompts_p15_2d3d_2to1.sh
bash scripts/E1_text_prototype_enhancement/02_2_generate_modelnetc_llm_prompts_p15_2d3d_1to2.sh

# 3. 运行 15 prompts 比例消融
bash scripts/E1_text_prototype_enhancement/02_3_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020_p15_2d3d_2to1.sh 0
bash scripts/E1_text_prototype_enhancement/02_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020_p15_2d3d_1to2.sh 0
```
