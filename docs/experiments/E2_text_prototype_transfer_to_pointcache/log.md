# E2 实验日志：文本原型增强向 Point-Cache 缓存流程的传递验证

## 2026-06-04：建立 E2 实验计划

E2 实验名称：

    E2_text_prototype_transfer_to_pointcache

中文名称：

    E2：文本原型增强向 Point-Cache 缓存流程的传递验证

E2 目标：

    验证 E1 中 `manual_full_llm_fusion` 带来的文本原型收益，
    能否传递到 Point-Cache 的 global cache 和 full Point-Cache 流程中。

当前 E2 定位为 smoke test，只做：

- ULIP；
- ModelNet-C；
- severity=2；
- zs_global；
- zs_global_local；
- manual_full；
- manual_full_llm_fusion。

暂不做：

- all35；
- 多数据集；
- 多 backbone；
- manual_3d；
- llm_only；
- 重新生成 LLM prompt。

## 2026-06-04：执行 E2 准备检查：runner 兼容性

新增检查报告：

    docs/experiments/E2_text_prototype_transfer_to_pointcache/checks/runner_compatibility_check.md

检查目标：

- 确认 E1 的 `prompt-source` 机制是否能进入 Point-Cache global cache 和 hierarchical cache runner；
- 确认是否已有 runner 支持 `zs_global` 与 `zs_global_local`；
- 确认 E1 shared prompt cache 是否完整；
- 为后续编写 E2 smoke test 脚本做准备。

## 2026-06-04：修正 E2 检查项命名规则

根据实验管理规范，runner 兼容性检查不再作为独立实验编号。

调整内容：

- 将 `runner_checks/00_runner_compatibility_check.md` 改为 `checks/runner_compatibility_check.md`；
- 明确只有真正运行模型并产生结果目录的脚本才使用实验编号；
- runner 检查、环境检查、prompt 缓存检查、脚本语法检查等均归为准备工作，不进入实验编号体系。

## 2026-06-04：完成 E2-1 smoke test 脚本编写

E2-1 是真正的实验脚本准备步骤，不包括 runner 兼容性检查。

新增脚本目录：

    Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/

新增公共脚本：

    00_run_ulip_modelnetc_s2_cache_transfer_common.sh

新增四个 smoke test 脚本：

- `00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke.sh`
- `00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke.sh`
- `00_3_ulip_modelnetc_s2_zs_global_local_manual_full_smoke.sh`
- `00_4_ulip_modelnetc_s2_zs_global_local_manual_full_llm_fusion_smoke.sh`

这些脚本用于验证 E1 的 `manual_full_llm_fusion` 文本原型收益能否传递到 Point-Cache 的 global cache 和 full Point-Cache 设置中。

说明：

- runner 兼容性检查属于准备检查，不作为实验编号；
- 只有会运行模型并产生结果目录的脚本才使用实验编号。

## 2026-06-04：记录 E2 global cache 对比分析

新增文档：

    docs/experiments/E2_text_prototype_transfer_to_pointcache/smoke_tests/00_global_cache_transfer_analysis.md

记录内容：

- E2 global cache 两组实验结果；
- 与 E1 zero-shot 结果的纵向对比；
- `manual_full_llm_fusion` 在 global cache 下相对 `manual_full` 的提升；
- `add_global` 异常下降现象；
- 对“global cache 可能放大初始伪标签偏差”的解释；
- 后续 full Point-Cache 实验注意事项。

## 2026-06-04：完成 E2 smoke test 完整分析文档

新增完整分析文档：

    docs/experiments/E2_text_prototype_transfer_to_pointcache/smoke_tests/00_e2_smoke_test_full_analysis.md

新增结果汇总文档：

    docs/experiments/E2_text_prototype_transfer_to_pointcache/results_summary.md

记录内容：

- E2 四组 smoke test 的实验设置；
- E2 对应脚本和结果目录；
- E2 运行过程中的 cache_type 修复；
- zero-shot、global cache、full Point-Cache 三阶段对比；
- manual_full 与 manual_full_llm_fusion 的整体准确率对比；
- 文本融合收益在 Point-Cache 流程中的传递情况；
- add_global 特殊现象的暂时性记录；
- 后续 ModelNet-C all35 full Point-Cache 验证建议。

当前结论：

    manual_full_llm_fusion 在 zero-shot、global cache 和 full Point-Cache 三个阶段均优于 manual_full，
    说明 E1 的文本原型融合收益能够传递到 Point-Cache 缓存流程中。
