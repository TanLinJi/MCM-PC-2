# E1 Text Prototype Enhancement

更新日期：2026-06-16

## 当前定位

E1 研究文本原型增强，不修改 Point-Cache 的缓存机制。当前重启原则是：

```text
manual_full 是最终文本原型的稳定锚点。
LLM 描述只能作为补充语义分支或后续 text distribution prior。
不要用 LLM-only 或文本 prompt 高斯分布直接替代最终分类器。
```

## 当前代码与结果位置

| 类型 | 路径 |
|---|---|
| Runner | `Point-Cache/runners/E1_text_prototype_enhancement/` |
| Scripts | `Point-Cache/scripts/E1_text_prototype_enhancement/` |
| Results | `Point-Cache/results/E1_text_prototype_enhancement/` |
| LLM prompt bank | `Point-Cache/llm/e1_prompt_bank/` |
| Docs | `docs/experiments/E1_text_prototype_enhancement/` |

## 当前应读文件

| 文件 | 作用 |
|---|---|
| `WORKLIST.md` | E1 重启工作清单和执行顺序 |
| `02_prompt_bank_and_template_audit.md` | manual_full 模板统计、LLM prompt bank 规范 |
| `03_restart_ablation_plan.md` | 新增 0.80:0.20、15 prompts、2D/3D 比例消融计划 |
| `00_s2_zero_shot_smoke_summary.md` | 已完成 S2 smoke test 汇总 |
| `01_s2_fusion_weight_ablation_analysis.md` | 已完成融合权重消融结果 |
| `99_log.md` | 历史实验日志 |

## 已完成关键结果

ULIP + ModelNet-C severity=2 + zero-shot：

| 方法 | 平均准确率 | 结论 |
|---|---:|---|
| `manual_full` | 47.68 | 原始完整手工模板 baseline |
| `manual_3d` | 35.63 | 失败消融，不能删除 2D/CLIP-style 模板 |
| `llm_only` | 39.30 | LLM 描述不能独立替代 manual_full |
| `manual_full_llm_fusion` 0.75:0.25 | 48.88 | 当前 E1 已完成 S2 最优 |

融合权重已完成：

| manual_full:LLM | 平均准确率 |
|---|---:|
| 0.90:0.10 | 48.41 |
| 0.85:0.15 | 48.62 |
| 0.80:0.20 | 48.84 |
| 0.75:0.25 | 48.88 |
| 0.50:0.50 | 48.37 |

## 下一步

1. 生成每类 15 条 LLM 描述，分别测试 2D:3D = 2:1 和 1:2。
2. 对比 `10 prompts / 15 prompts`，固定融合权重优先使用 `0.80:0.20`。
3. S2 消融稳定后，再进入 ModelNet-C all35 zero-shot。

本机建议使用：

```bash
export E1_PYTHON_CMD="conda run -n mcmpc python"
```
