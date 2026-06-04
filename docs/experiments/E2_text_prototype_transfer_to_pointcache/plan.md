# E2：文本原型增强向 Point-Cache 缓存流程的传递验证

更新日期：2026-06-04

## 1. 实验名称

英文目录名：

    E2_text_prototype_transfer_to_pointcache

中文名称：

    E2：文本原型增强向 Point-Cache 缓存流程的传递验证

## 2. 实验背景

E1 已经完成文本原型增强的最小验证。

E1 的核心结论是：

    原始完整手工模板 manual_full 不能被 LLM 生成描述直接替代；
    但 LLM 生成的类别级多视角描述可以作为补充语义分支；
    将 manual_full 文本原型与 LLM 描述文本原型进行加权融合，可以提升 zero-shot 分类性能。

E1 当前最优平均性能来自：

    manual_full : LLM = 0.75 : 0.25

对应方法名：

    manual_full_llm_fusion

E1-S1 权重消融结果表明：

| 方法 | manual_full 权重 | LLM 权重 | 平均准确率 |
|---|---:|---:|---:|
| manual_full | 1.00 | 0.00 | 47.68 |
| manual_full_llm_fusion | 0.90 | 0.10 | 48.41 |
| manual_full_llm_fusion | 0.85 | 0.15 | 48.62 |
| manual_full_llm_fusion | 0.75 | 0.25 | 48.88 |
| manual_full_llm_fusion | 0.50 | 0.50 | 48.37 |

因此，E1 证明了文本原型融合在 zero-shot 设置下是有效的。

但 Point-Cache 的完整流程不仅包括 zero-shot 文本原型，还包括测试时动态缓存，包括：

- global cache；
- local cache；
- global cache + local cache 的完整 Point-Cache。

因此，E2 需要回答的问题是：

    E1 中得到的文本原型融合收益，能否传递到 Point-Cache 的缓存增强流程中？

## 3. 实验目标

E2 的目标不是继续做文本消融，而是进行纵向流程验证。

具体目标：

1. 验证 `manual_full_llm_fusion` 在 `zs_global` 设置下是否仍然优于 `manual_full`；
2. 验证 `manual_full_llm_fusion` 在 `zs_global_local` 设置下是否仍然优于 `manual_full`；
3. 判断 E1 的文本收益是否会被缓存机制保留、放大或抵消；
4. 打通从文本原型增强到 Point-Cache 缓存增强的完整实验链路。

## 4. E2 与 E1 的关系

E1 是横向实验，主要比较不同文本原型构造方式。

E1 包括：

| 方法 | 作用 |
|---|---|
| manual_full | 原始完整手工模板 baseline |
| manual_3d | 删除 2D 图像风格模板后的失败消融 |
| llm_only | 只使用 LLM 生成描述 |
| manual_full_llm_fusion | 原始完整手工模板与 LLM 描述融合 |

E2 是纵向实验，主要验证 E1 的最优文本方法能否进入 Point-Cache 流程。

E2 只保留两种文本方法：

| 方法 | 作用 |
|---|---|
| manual_full | Point-Cache 原始文本 baseline |
| manual_full_llm_fusion | E1 当前主方法 |

E2 暂时不继续使用：

| 方法 | 原因 |
|---|---|
| manual_3d | E1 已证明明显失败 |
| llm_only | E1 已证明不能替代 manual_full |
| 其他权重 | E1-S1 已完成权重消融，当前先使用 0.75:0.25 |

## 5. E2 当前最小验证设置

当前 E2 先做 smoke test，不直接进入完整 all35。

| 项目 | 设置 |
|---|---|
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 损坏强度 | severity=2 |
| 损坏类型数量 | 7 |
| 文本方法 | manual_full、manual_full_llm_fusion |
| 缓存设置 | zs_global、zs_global_local |
| 是否重新生成 LLM prompt | 否 |
| LLM prompt 来源 | E1 共享 prompt 缓存 |

说明：

E2 不重复跑 `zs`，因为 E1 已经完成 zero-shot 对比。E2 分析时直接引用 E1 中的 zero-shot 结果作为起点。

## 6. E2 当前实验矩阵

当前只跑 4 组最小纵向验证：

| 编号 | 设置 | 文本方法 | 目的 |
|---|---|---|---|
| 00_1 | zs_global | manual_full | 原始文本模板 + global cache |
| 00_2 | zs_global | manual_full_llm_fusion | 验证文本融合收益能否传递到 global cache |
| 00_3 | zs_global_local | manual_full | 原始文本模板 + 完整 Point-Cache |
| 00_4 | zs_global_local | manual_full_llm_fusion | 验证文本融合收益能否传递到完整 Point-Cache |

其中：

- `zs_global` 表示 zero-shot + global cache；
- `zs_global_local` 表示 zero-shot + global cache + local cache；
- `manual_full_llm_fusion` 使用 E1 当前默认权重 0.75:0.25。

## 7. 共享 LLM prompt 缓存

E2 必须复用 E1 已生成的共享 prompt，不重新调用 LLM API。

共享 prompt 目录：

    Point-Cache/results/E1_text_prototype_enhancement/shared_prompts/

共享 prompt 文件：

    Point-Cache/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json

原因：

1. 避免重复消耗 API token；
2. 保证 E1 和 E2 使用同一份 LLM 描述；
3. 避免 LLM 重新生成带来的随机差异；
4. 让 E2 只验证缓存流程是否能继承文本收益，而不是同时改变文本内容。

## 8. 目录规范

### 8.1 文档目录

    docs/experiments/E2_text_prototype_transfer_to_pointcache/

建议文件：

| 文件 | 作用 |
|---|---|
| plan.md | E2 实验计划 |
| log.md | E2 实验过程记录 |
| analysis.md | E2 总体分析 |
| smoke_tests/ | 每个 E2 smoke test 的独立分析 |
| results_summary.md | E2 结果汇总 |

### 8.2 脚本目录

    Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/

建议脚本：

| 脚本 | 作用 |
|---|---|
| 00_run_ulip_modelnetc_s2_cache_transfer_common.sh | E2 公共脚本 |
| 00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke.sh | manual_full + global cache |
| 00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke.sh | manual_full_llm_fusion + global cache |
| 00_3_ulip_modelnetc_s2_zs_global_local_manual_full_smoke.sh | manual_full + full Point-Cache |
| 00_4_ulip_modelnetc_s2_zs_global_local_manual_full_llm_fusion_smoke.sh | manual_full_llm_fusion + full Point-Cache |

### 8.3 结果目录

    Point-Cache/results/E2_text_prototype_transfer_to_pointcache/

建议结果目录：

| 结果目录 | 对应实验 |
|---|---|
| 00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke/ | manual_full + global cache |
| 00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke/ | manual_full_llm_fusion + global cache |
| 00_3_ulip_modelnetc_s2_zs_global_local_manual_full_smoke/ | manual_full + full Point-Cache |
| 00_4_ulip_modelnetc_s2_zs_global_local_manual_full_llm_fusion_smoke/ | manual_full_llm_fusion + full Point-Cache |

## 9. 方法命名规范

E2 公开方法名：

| 公开方法名 | 内部 prompt source | 含义 |
|---|---|---|
| manual_full | manual_full | Point-Cache 原始完整手工模板 |
| manual_full_llm_fusion | manualfull_llm_dynamic_init | manual_full 文本原型与 LLM 描述文本原型加权融合 |

E2 缓存设置名：

| 名称 | 含义 |
|---|---|
| zs_global | zero-shot + global cache |
| zs_global_local | zero-shot + global cache + local cache |

E2 不使用：

| 名称 | 原因 |
|---|---|
| add_llm | 容易误解成简单追加文本 |
| dynamic_multiview_llm_descriptions | 已改为 llm_only，但 E2 不使用 llm_only |
| fuse_manualfull_multiview_llm | 旧命名，已统一为 manual_full_llm_fusion |

## 10. 预期结果与判断标准

### 10.1 global cache 判断

如果：

    manual_full_llm_fusion + zs_global
    >
    manual_full + zs_global

说明 E1 的文本原型收益可以传递到 global cache。

### 10.2 full Point-Cache 判断

如果：

    manual_full_llm_fusion + zs_global_local
    >
    manual_full + zs_global_local

说明 E1 的文本原型收益可以传递到完整 Point-Cache。

### 10.3 可能出现的三种结果

#### 情况 A：global cache 和 full Point-Cache 都提升

这是最理想情况。

说明：

    文本原型融合不仅提升 zero-shot，
    也能稳定提升 Point-Cache 的缓存增强流程。

后续可进入 ModelNet-C all35 完整验证。

#### 情况 B：global cache 提升，但 full Point-Cache 不提升

说明：

    文本收益能传递到全局缓存，
    但在局部缓存融合时可能被抵消。

后续需要分析 local cache 与文本原型之间是否存在冲突。

#### 情况 C：global cache 和 full Point-Cache 都不提升

说明：

    E1 文本收益主要作用于 zero-shot 分支，
    但缓存分支可能主导最终预测，使文本收益被覆盖。

后续需要考虑缓存权重、文本分支权重或融合位置。

## 11. 当前不做的事情

E2 当前不做以下内容：

1. 不跑 all35；
2. 不跑 clean ModelNet40；
3. 不跑 ScanObjectNN；
4. 不跑 ScanObjectNN-C；
5. 不跑多 backbone；
6. 不继续做 manual_3d；
7. 不继续做 llm_only；
8. 不重新生成 LLM prompt；
9. 不继续扩展权重消融；
10. 不把 results 加入 Git。

## 12. 完整验证的后续扩展

如果 E2 smoke test 有效，后续可以进入 E3：

    E3_modelnetc_all35_text_prototype_transfer

E3 可扩展为：

| 阶段 | 设置 |
|---|---|
| E3-1 | ModelNet-C all35 zero-shot |
| E3-2 | ModelNet-C all35 zs_global |
| E3-3 | ModelNet-C all35 zs_global_local |

如果 E3 仍然有效，再扩展到：

| 阶段 | 内容 |
|---|---|
| E4 | ScanObjectNN / ScanObjectNN-C |
| E5 | 多 backbone，例如 ULIP2、OpenShape、Uni3D |
| E6 | 完整论文主表实验 |

## 13. E2 任务计划

### Task E2-0：建立 E2 文档和目录

状态：进行中。

产物：

    docs/experiments/E2_text_prototype_transfer_to_pointcache/plan.md

### 准备检查：确认现有 runner 是否支持 prompt-source 与 cache method 同时运行

目标：

确认 E1 新增的 prompt-source 机制是否能被 Point-Cache 的 global cache 和 local cache runner 正确调用。

重点检查：

- runner 是否调用 `clip_classifier(args, classnames, template, clip_model)`；
- dataset 是否能返回 E1 prompt template；
- cache 设置是否不会覆盖 prompt-source；
- prompt-cache-dir 是否可以指向 E1 shared_prompts；
- full Point-Cache 是否会重复构造文本原型。

### Task E2-2：编写 E2 公共脚本

目标：

创建：

    Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/00_run_ulip_modelnetc_s2_cache_transfer_common.sh

该公共脚本应支持：

- 传入 EXP_ID；
- 传入 baseline method；
- 传入 prompt source；
- 传入方法说明；
- 传入 GPU 编号；
- 指定 E2 结果目录；
- 指定 E1 shared prompt 缓存；
- 设置融合权重 0.75:0.25。

### Task E2-3：编写四个 E2 smoke test 脚本

目标：

创建四个脚本：

- `00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke.sh`
- `00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke.sh`
- `00_3_ulip_modelnetc_s2_zs_global_local_manual_full_smoke.sh`
- `00_4_ulip_modelnetc_s2_zs_global_local_manual_full_llm_fusion_smoke.sh`

### Task E2-4：先运行 global cache 两组

建议先跑：

| 脚本 | GPU |
|---|---|
| 00_1 manual_full + zs_global | GPU 0 |
| 00_2 manual_full_llm_fusion + zs_global | GPU 1 |

如果两张卡空闲，可以并行运行。

### Task E2-5：再运行 full Point-Cache 两组

再跑：

| 脚本 | GPU |
|---|---|
| 00_3 manual_full + zs_global_local | GPU 0 |
| 00_4 manual_full_llm_fusion + zs_global_local | GPU 1 |

### Task E2-6：写 E2 结果分析

分析重点：

1. E1 zero-shot 提升是否在 global cache 中保留；
2. E1 zero-shot 提升是否在 full Point-Cache 中保留；
3. global cache 与 local cache 是否放大或抵消文本收益；
4. 是否值得进入 all35。

### Task E2-7：Git 提交

建议提交信息：

    feat: add E2 text prototype transfer to Point-Cache plan and scripts

## 14. 实验编号与检查项命名规则

为避免实验管理混乱，E2 之后统一采用以下规则：

### 14.1 会编号的内容

只有真正运行模型并产生实验结果的脚本和结果目录才使用实验编号。

例如：

- `00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke.sh`
- `00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke.sh`
- `00_3_ulip_modelnetc_s2_zs_global_local_manual_full_smoke.sh`
- `00_4_ulip_modelnetc_s2_zs_global_local_manual_full_llm_fusion_smoke.sh`

这些脚本会产生对应结果目录，因此需要编号。

### 14.2 不编号的内容

以下内容不作为实验编号：

- runner 兼容性检查；
- 环境检查；
- prompt 缓存完整性检查；
- 结果目录清理；
- 脚本语法检查；
- 文档同步检查；
- 代码静态检查。

这些内容统一放在：

    docs/experiments/E2_text_prototype_transfer_to_pointcache/checks/

例如：

    checks/runner_compatibility_check.md

### 14.3 原则

实验编号只表示“可复现的实验运行单元”。

检查项属于准备工作和维护工作，不进入实验编号体系。

## 15. 术语修正：纵向实验与横向实验

为避免后续实验管理混乱，当前项目中统一采用以下定义：

- 纵向实验：只围绕所提方法做最小验证，目的是把方法链路跑通，证明方向成立；
- 横向实验：完整验证或大规模验证，包括 all35、多个数据集、多个 cache 设置、多个 backbone、多个方法对比等。

因此，E2 当前 smoke test 属于纵向最小验证；后续 ModelNet-C all35、跨数据集、跨 backbone 和完整方法矩阵属于横向完整验证。

