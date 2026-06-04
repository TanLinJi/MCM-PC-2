# E2 Smoke Test 完整分析：文本原型增强向 Point-Cache 缓存流程的传递验证

更新日期：2026-06-04

## 1. 实验名称

英文名称：

    E2_text_prototype_transfer_to_pointcache

中文名称：

    E2：文本原型增强向 Point-Cache 缓存流程的传递验证

## 2. 实验背景

E1 已经完成文本原型增强方向的最小验证，核心结论是：

    LLM 生成的类别描述不能直接替代 Point-Cache 原始完整手工模板，
    但可以作为补充语义分支，与原始完整手工模板文本原型进行加权融合。

E1 中当前主方法为：

    manual_full_llm_fusion

其含义是：

    manual_full 文本原型与 LLM 描述文本原型进行加权融合。

当前默认融合权重为：

    manual_full : LLM = 0.75 : 0.25

E1 zero-shot 阶段结果为：

| 方法 | 设置 | 平均准确率 |
|---|---|---:|
| manual_full | zero-shot | 47.68 |
| manual_full_llm_fusion | zero-shot | 48.88 |

因此，E1 证明了文本原型融合在 zero-shot 设置下能够带来提升。

但是，Point-Cache 的完整流程并不只依赖 zero-shot 文本原型，还会进一步利用测试时动态缓存，包括：

- global cache；
- local cache；
- global cache + local cache 的完整 Point-Cache 流程。

因此，E2 需要回答的问题是：

    E1 中通过文本原型融合获得的 zero-shot 收益，
    能否传递到 Point-Cache 的缓存增强流程中？

## 3. 实验目标

E2 不是继续做横向文本消融，而是进行纵向流程验证。

具体目标包括：

1. 验证 `manual_full_llm_fusion` 在 `zs_global` 设置下是否仍然优于 `manual_full`；
2. 验证 `manual_full_llm_fusion` 在 `zs_global_local` 设置下是否仍然优于 `manual_full`；
3. 分析文本原型融合收益在 zero-shot、global cache、full Point-Cache 三个阶段中的变化；
4. 判断是否值得进入后续 ModelNet-C all35 的横向完整验证。

## 4. 实验设置

| 项目 | 设置 |
|---|---|
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 损坏强度 | severity=2 |
| 损坏类型数量 | 7 |
| 文本方法 | manual_full、manual_full_llm_fusion |
| 缓存设置 | zs_global、zs_global_local |
| LLM prompt 来源 | E1 shared prompt cache |
| 是否重新生成 LLM prompt | 否 |
| manual_full_llm_fusion 权重 | manual_full:LLM = 0.75:0.25 |
| 结果目录 | Point-Cache/results/E2_text_prototype_transfer_to_pointcache/ |

说明：

E2 不重新生成 LLM prompt，而是复用 E1 已经生成好的共享 prompt 缓存：

    Point-Cache/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json

这样可以保证：

1. E1 与 E2 使用相同的 LLM 类别描述；
2. 不重复消耗 API token；
3. 不引入 LLM 重新生成带来的随机差异；
4. E2 只验证文本原型收益能否传递到缓存流程，而不是同时改变文本内容。

## 5. E2 四组 smoke test

E2 当前最小验证共运行四组实验：

| 编号 | cache 设置 | 文本方法 | 作用 |
|---|---|---|---|
| 00_1 | zs_global | manual_full | 原始完整手工模板 + global cache |
| 00_2 | zs_global | manual_full_llm_fusion | 文本融合 + global cache |
| 00_3 | zs_global_local | manual_full | 原始完整手工模板 + full Point-Cache |
| 00_4 | zs_global_local | manual_full_llm_fusion | 文本融合 + full Point-Cache |

其中：

- `zs_global` 表示 zero-shot + global cache；
- `zs_global_local` 表示 zero-shot + global cache + local cache，也就是完整 Point-Cache 设置。

## 6. 对应脚本与结果目录

### 6.1 00_1：manual_full + zs_global

脚本：

    Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke.sh

结果目录：

    Point-Cache/results/E2_text_prototype_transfer_to_pointcache/00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke/

### 6.2 00_2：manual_full_llm_fusion + zs_global

脚本：

    Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke.sh

结果目录：

    Point-Cache/results/E2_text_prototype_transfer_to_pointcache/00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke/

### 6.3 00_3：manual_full + zs_global_local

脚本：

    Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/00_3_ulip_modelnetc_s2_zs_global_local_manual_full_smoke.sh

结果目录：

    Point-Cache/results/E2_text_prototype_transfer_to_pointcache/00_3_ulip_modelnetc_s2_zs_global_local_manual_full_smoke/

### 6.4 00_4：manual_full_llm_fusion + zs_global_local

脚本：

    Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/00_4_ulip_modelnetc_s2_zs_global_local_manual_full_llm_fusion_smoke.sh

结果目录：

    Point-Cache/results/E2_text_prototype_transfer_to_pointcache/00_4_ulip_modelnetc_s2_zs_global_local_manual_full_llm_fusion_smoke/

## 7. 实验过程中的关键修复

在运行 `zs_global_local` 两组实验时，曾出现如下错误：

    ValueError: not enough values to unpack (expected 6, got 5)

原因是：

    E2 公共脚本中将 --cache-type 固定写成了 global。

但是：

- `zs_global` 应使用 `--cache-type global`；
- `zs_global_local` 应使用 `--cache-type hierarchical`。

`zs_global_local` 会进入 hierarchical cache runner，该 runner 需要 `get_logits()` 返回 6 个值，包括 `patch_centers`。如果仍然使用 `cache-type=global`，则 `get_logits()` 只返回 5 个值，从而触发报错。

已修复为根据 `CACHE_METHOD` 自动选择 `CACHE_TYPE`：

    zs_global       -> CACHE_TYPE=global
    zs_global_local -> CACHE_TYPE=hierarchical

该问题属于脚本参数配置问题，不是方法本身的问题，也不是 LLM prompt 的问题。

## 8. 总体结果对比

| 阶段 | 设置 | 文本方法 | 平均准确率 | 相对 E1 zs manual_full |
|---|---|---|---:|---:|
| E1 | zero-shot | manual_full | 47.68 | 0.00 |
| E1 | zero-shot | manual_full_llm_fusion | 48.88 | +1.20 |
| E2 | zs_global | manual_full | 52.66 | +4.98 |
| E2 | zs_global | manual_full_llm_fusion | 53.18 | +5.50 |
| E2 | zs_global_local | manual_full | 54.00 | +6.32 |
| E2 | zs_global_local | manual_full_llm_fusion | 54.21 | +6.53 |

当前最高结果为：

    manual_full_llm_fusion + zs_global_local = 54.21

相对于 E1 zero-shot baseline：

    54.21 - 47.68 = +6.53

相对于原始完整 Point-Cache 设置：

    54.21 - 54.00 = +0.21

因此，E2 的总体结论是：

    manual_full_llm_fusion 在 zero-shot、global cache 和 full Point-Cache 三个阶段中，
    平均准确率均高于 manual_full。

这说明 E1 中得到的文本原型融合收益可以传递到 Point-Cache 缓存流程中。

## 9. 分损坏类型完整结果

| 损坏类型 | zs manual_full | zs fusion | global manual_full | global fusion | full manual_full | full fusion |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 34.00 | 33.55 | 46.07 | 41.65 | 47.81 | 42.87 |
| add_local | 43.92 | 44.61 | 47.24 | 48.26 | 46.68 | 48.26 |
| dropout_global | 54.70 | 57.01 | 57.05 | 59.28 | 59.20 | 60.37 |
| dropout_local | 50.57 | 53.44 | 54.86 | 56.04 | 56.69 | 57.86 |
| rotate | 55.19 | 56.36 | 59.81 | 61.43 | 62.07 | 63.29 |
| scale | 50.89 | 52.76 | 53.97 | 55.67 | 55.23 | 55.55 |
| jitter | 44.49 | 44.45 | 49.64 | 49.92 | 50.32 | 51.26 |
| 平均 | 47.68 | 48.88 | 52.66 | 53.18 | 54.00 | 54.21 |

## 10. 文本融合相对 manual_full 的增益变化

| 损坏类型 | zero-shot 阶段 | global cache 阶段 | full Point-Cache 阶段 |
|---|---:|---:|---:|
| add_global | -0.45 | -4.42 | -4.94 |
| add_local | +0.69 | +1.02 | +1.58 |
| dropout_global | +2.31 | +2.23 | +1.17 |
| dropout_local | +2.87 | +1.18 | +1.17 |
| rotate | +1.17 | +1.62 | +1.22 |
| scale | +1.87 | +1.70 | +0.32 |
| jitter | -0.04 | +0.28 | +0.94 |
| 平均 | +1.20 | +0.52 | +0.21 |

这个表说明：

1. 在 zero-shot 阶段，文本融合带来的平均提升为 +1.20；
2. 在 global cache 阶段，文本融合带来的平均提升为 +0.52；
3. 在 full Point-Cache 阶段，文本融合带来的平均提升为 +0.21。

因此，文本融合收益在进入缓存流程后仍然存在，但边际贡献逐渐变小。

这说明：

    Point-Cache 的 cache 分支是主要增益来源；
    manual_full_llm_fusion 是在 cache 增强基础上的附加增益来源。

## 11. cache 本身带来的收益

### 11.1 manual_full 下的 cache 收益

| 对比 | 平均准确率变化 |
|---|---:|
| zero-shot -> global cache | 52.66 - 47.68 = +4.98 |
| global cache -> full Point-Cache | 54.00 - 52.66 = +1.34 |
| zero-shot -> full Point-Cache | 54.00 - 47.68 = +6.32 |

### 11.2 manual_full_llm_fusion 下的 cache 收益

| 对比 | 平均准确率变化 |
|---|---:|
| zero-shot -> global cache | 53.18 - 48.88 = +4.30 |
| global cache -> full Point-Cache | 54.21 - 53.18 = +1.03 |
| zero-shot -> full Point-Cache | 54.21 - 48.88 = +5.33 |

这说明：

1. global cache 对两种文本方法都有显著提升；
2. local cache 在 global cache 的基础上继续带来提升；
3. 完整 Point-Cache 的主要增益来自 cache 机制本身；
4. 文本融合在 cache 增益之外提供额外但较小的提升。

## 12. 对 E2 结果的整体理解

E2 的关键不是观察某一个损坏类型，而是验证整体分类准确率是否沿着纵向流程保持正收益。

从整体平均准确率看：

    manual_full_llm_fusion 在三个阶段都优于 manual_full。

具体为：

| 阶段 | manual_full | manual_full_llm_fusion | 提升 |
|---|---:|---:|---:|
| zero-shot | 47.68 | 48.88 | +1.20 |
| global cache | 52.66 | 53.18 | +0.52 |
| full Point-Cache | 54.00 | 54.21 | +0.21 |

因此可以得到 E2 的主要结论：

    E1 的文本原型融合收益能够传递到 Point-Cache 缓存流程中。
    这种收益在 zero-shot 阶段最明显，在 global cache 和 full Point-Cache 中仍然保持正向，
    但随着缓存机制增强，文本分支的边际贡献被压缩。

这是合理的，因为 Point-Cache 在 zero-shot 预测基础上进一步引入测试时样本缓存，最终预测不再只由文本原型决定，而是由文本原型、global cache 和 local cache 共同影响。

## 13. 关于 add_global 的说明

在当前结果中，`add_global` 是一个比较特殊的现象。

`manual_full_llm_fusion` 在 `add_global` 上低于 `manual_full`：

| 阶段 | 差值 |
|---|---:|
| zero-shot | -0.45 |
| global cache | -4.42 |
| full Point-Cache | -4.94 |

这个现象需要记录，但当前不是 E2 的主要分析重点。

原因是：

1. E2 当前目标是验证文本原型融合收益是否能够在整体分类准确率上向 Point-Cache 流程传递；
2. 从平均准确率看，`manual_full_llm_fusion` 在 zero-shot、global cache 和 full Point-Cache 三个阶段均优于 `manual_full`；
3. 7 个损坏类型中，多数损坏类型在 fusion 设置下是提升的；
4. `add_global` 更像是一个特殊损坏类型上的异常现象，后续在完整改进 Point-Cache 或进一步分析伪标签/缓存构建机制时再单独研究。

因此，当前文档中对 `add_global` 的定位是：

    作为特殊现象记录；
    不作为否定 E2 主结论的依据；
    后续在 full Point-Cache 完整改进阶段再回头分析。

## 14. 对 add_global 现象的暂时解释

虽然当前不把 `add_global` 作为重点，但仍可以记录一个暂时性解释，供后续分析使用。

Point-Cache 的 cache 不是用真实标签构建的，而是在测试时利用模型自己的预测结果构建：

    1. 模型先对测试样本进行 zero-shot 预测；
    2. 根据预测结果和置信度筛选样本；
    3. 将点云特征作为 key；
    4. 将模型预测类别作为 value，也就是伪标签；
    5. 后续样本查询 cache，并将 cache logits 与 zero-shot logits 结合。

因此，如果文本原型变化导致某些样本的初始预测或置信度结构发生变化，那么进入 cache 的伪标签也会发生变化。

如果这种变化是正确的，cache 会放大正收益。

如果这种变化是错误的，cache 也可能放大负收益。

当前 `add_global` 上可能存在这种情况：

    manual_full_llm_fusion 在 add_global 上改变了部分初始预测或置信度；
    global/local cache 又依赖这些初始预测来构建缓存；
    因此某些不利的初始伪标签差异可能在缓存流程中被放大。

不过，这只是当前阶段的合理推测。由于 E2 关注的是整体分类准确率和纵向流程是否有效，因此该问题暂不展开，只作为后续分析方向保留。

## 15. E2 当前结论

E2 smoke test 当前可以得到以下结论：

1. E2 四组实验均已完成，结果有效；
2. `manual_full_llm_fusion` 在 zero-shot 阶段相对 `manual_full` 提升 +1.20；
3. `manual_full_llm_fusion` 在 global cache 阶段相对 `manual_full` 提升 +0.52；
4. `manual_full_llm_fusion` 在 full Point-Cache 阶段相对 `manual_full` 提升 +0.21；
5. 这说明 E1 的文本原型融合收益能够传递到 Point-Cache 的缓存流程中；
6. 随着 global cache 和 local cache 引入，cache 分支成为主要增益来源，文本融合的边际贡献变小；
7. 当前最优结果为 `manual_full_llm_fusion + zs_global_local`，平均准确率为 54.21；
8. `add_global` 是特殊现象，应记录但不作为当前重点；
9. 当前结果支持进入 ModelNet-C all35 的进一步横向完整验证。

## 16. 下一步建议

当前 E2 smoke test 已经跑通，且整体结果为正。

下一步建议：

1. 先将 E2 当前文档、脚本和结果分析提交到 Git；
2. 然后进入横向完整验证；
3. 优先验证 full Point-Cache 设置，而不是立即扩展所有横向组合。

建议下一阶段实验为：

    ModelNet-C all35
    manual_full vs manual_full_llm_fusion
    zs_global_local

原因：

1. `zs_global_local` 是完整 Point-Cache 设置；
2. 当前 E2 smoke test 中 `manual_full_llm_fusion + zs_global_local` 仍然优于原始完整 Point-Cache；
3. 如果 all35 下仍然保持提升，则 E2 的结论会更加稳固；
4. 暂时不必重新跑 manual_3d、llm_only 或更多权重消融，因为这些属于横向实验，不是当前纵向流程验证的重点。
