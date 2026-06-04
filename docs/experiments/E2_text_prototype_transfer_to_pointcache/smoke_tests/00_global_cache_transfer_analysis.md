# E2 Global Cache Smoke Test 分析：文本原型融合收益向 global cache 的传递验证

更新日期：2026-06-04

## 1. 实验目的

E2 的目标是验证 E1 中得到的文本原型增强收益，是否能够传递到 Point-Cache 的缓存增强流程中。

E1 已经证明：

- `manual_full` 是 Point-Cache 原始完整手工模板；
- `manual_full_llm_fusion` 是当前 E1 的主方法；
- 在 zero-shot 设置下，`manual_full_llm_fusion` 相比 `manual_full` 有提升。

E2 不再继续做横向文本消融，而是做纵向流程验证：

    文本原型增强
        ↓
    zero-shot 提升
        ↓
    global cache 是否还能保留这种提升？
        ↓
    full Point-Cache 是否还能保留这种提升？

本文档记录 E2 中 global cache 两组 smoke test 的结果分析。

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| 实验阶段 | E2 |
| 实验名称 | E2_text_prototype_transfer_to_pointcache |
| Backbone | ULIP |
| 数据集 | ModelNet-C |
| 损坏强度 | severity=2 |
| 损坏类型数量 | 7 |
| cache 设置 | zs_global |
| 对比方法 1 | manual_full |
| 对比方法 2 | manual_full_llm_fusion |
| LLM prompt 来源 | E1 shared prompt cache |
| 融合权重 | manual_full:LLM = 0.75:0.25 |

## 3. 对应脚本与结果目录

### 3.1 manual_full + global cache

脚本：

    Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke.sh

结果目录：

    Point-Cache/results/E2_text_prototype_transfer_to_pointcache/00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke/

对应设置：

    zs_global + manual_full

### 3.2 manual_full_llm_fusion + global cache

脚本：

    Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke.sh

结果目录：

    Point-Cache/results/E2_text_prototype_transfer_to_pointcache/00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke/

对应设置：

    zs_global + manual_full_llm_fusion

## 4. 总体结果对比

| 设置 | 文本方法 | 平均准确率 |
|---|---|---:|
| E1 zero-shot baseline | manual_full | 47.68 |
| E1 zero-shot fusion | manual_full_llm_fusion | 48.88 |
| E2 global cache baseline | manual_full + zs_global | 52.66 |
| E2 global cache fusion | manual_full_llm_fusion + zs_global | 53.18 |

## 5. 核心结论

在 global cache 设置下：

    manual_full_llm_fusion + zs_global = 53.18
    manual_full + zs_global = 52.66

因此：

    提升 = 53.18 - 52.66 = +0.52

这说明 E1 中得到的文本原型融合收益可以传递到 Point-Cache 的 global cache 流程中。

不过，相比 zero-shot 阶段的提升：

    E1 zero-shot 提升 = 48.88 - 47.68 = +1.20

global cache 阶段的提升变为：

    E2 global cache 提升 = 53.18 - 52.66 = +0.52

因此当前结论应表述为：

    文本原型融合收益能够部分传递到 global cache；
    global cache 没有完全抵消文本收益；
    但传递后的增益幅度小于 zero-shot 阶段。

## 6. 分损坏类型结果对比

| 损坏类型 | E1 zs manual_full | E1 zs fusion | E2 global manual_full | E2 global fusion | fusion 相对 global baseline |
|---|---:|---:|---:|---:|---:|
| add_global | 34.00 | 33.55 | 46.07 | 41.65 | -4.42 |
| add_local | 43.92 | 44.61 | 47.24 | 48.26 | +1.02 |
| dropout_global | 54.70 | 57.01 | 57.05 | 59.28 | +2.23 |
| dropout_local | 50.57 | 53.44 | 54.86 | 56.04 | +1.18 |
| rotate | 55.19 | 56.36 | 59.81 | 61.43 | +1.62 |
| scale | 50.89 | 52.76 | 53.97 | 55.67 | +1.70 |
| jitter | 44.49 | 44.45 | 49.64 | 49.92 | +0.28 |
| 平均 | 47.68 | 48.88 | 52.66 | 53.18 | +0.52 |

## 7. 分损坏类型观察

从分损坏类型结果看，`manual_full_llm_fusion + zs_global` 在 7 个损坏类型中有 6 个超过 `manual_full + zs_global`。

提升的损坏类型包括：

- add_local：+1.02
- dropout_global：+2.23
- dropout_local：+1.18
- rotate：+1.62
- scale：+1.70
- jitter：+0.28

唯一明显下降的是：

- add_global：-4.42

因此，global cache 的整体结论是积极的，但 `add_global` 是一个需要重点关注的异常项。

如果去掉 `add_global`，其余 6 个损坏类型的平均提升约为：

    (+1.02 + 2.23 + 1.18 + 1.62 + 1.70 + 0.28) / 6 = +1.34

这说明 `manual_full_llm_fusion` 在 global cache 中并不是普遍失效，而是在多数损坏类型中仍然有效，只是被 `add_global` 的明显下降拉低了总体平均提升。

## 8. global cache 本身的收益

### 8.1 manual_full 下的 global cache 收益

    manual_full + zero-shot = 47.68
    manual_full + global cache = 52.66

提升：

    52.66 - 47.68 = +4.98

### 8.2 manual_full_llm_fusion 下的 global cache 收益

    manual_full_llm_fusion + zero-shot = 48.88
    manual_full_llm_fusion + global cache = 53.18

提升：

    53.18 - 48.88 = +4.30

这说明 global cache 对两种文本原型都有明显增益。

不过，global cache 对 `manual_full` 的增益是 +4.98，而对 `manual_full_llm_fusion` 的增益是 +4.30。一个合理解释是：

    manual_full_llm_fusion 已经在 zero-shot 阶段修正了一部分错误，
    因此留给 global cache 继续提升的空间相对变小。

## 9. add_global 异常现象分析

`add_global` 是当前最明显的异常项。

| 设置 | add_global 准确率 |
|---|---:|
| E1 zs manual_full | 34.00 |
| E1 zs fusion | 33.55 |
| E2 global manual_full | 46.07 |
| E2 global fusion | 41.65 |

在 zero-shot 阶段：

    E1 zs fusion - E1 zs manual_full = 33.55 - 34.00 = -0.45

这个下降很小。

但加入 global cache 后：

    E2 global fusion - E2 global manual_full = 41.65 - 46.07 = -4.42

这个下降被明显放大。

这说明在 `add_global` 上，问题可能不是文本原型本身造成了很大下降，而是文本原型导致的初始预测差异被 global cache 放大了。

## 10. “缓存放大初始伪标签偏差”是什么意思？

Point-Cache 的 global cache 不是用真实标签建立的，而是在测试时根据模型自己的预测结果动态建立的。

简化来说，global cache 的流程可以理解为：

    1. 模型先对一个测试样本做 zero-shot 预测；
    2. 如果这个预测足够可信，就把这个样本写入 cache；
    3. cache 里保存：
       - key：点云特征；
       - value：模型自己预测出来的类别，也就是伪标签；
    4. 后续新样本会查询 cache；
    5. cache 的结果会参与最终分类。

所以，global cache 的质量依赖于初始 zero-shot 预测。

对于 `manual_full`：

    zero-shot 文本原型是原始完整手工模板；
    它会产生一组初始预测和置信度；
    global cache 基于这些预测来写入伪标签。

对于 `manual_full_llm_fusion`：

    zero-shot 文本原型发生了变化；
    它融合了 LLM 生成描述；
    因此初始预测分布和置信度也可能发生变化；
    global cache 写入的伪标签也会随之变化。

如果这种变化是正确的，那么 cache 会放大正收益。

例如：

    原来 manual_full 把某些样本预测错；
    manual_full_llm_fusion 把它们预测对；
    global cache 把这些正确伪标签写进去；
    后续样本受益；
    最终准确率提高。

但如果这种变化是错误的，那么 cache 也可能放大负收益。

例如：

    原来 manual_full 在 add_global 上某些样本预测正确；
    manual_full_llm_fusion 由于文本原型变化，把它们预测错；
    如果这些错误预测又具有较高置信度；
    global cache 可能把错误伪标签写进去；
    后续相似样本查询 cache 时受到错误伪标签影响；
    最终 add_global 准确率下降被放大。

这就是“manual_full_llm_fusion 改变了初始 zero-shot 预测分布；global cache 又依赖初始预测结果、伪标签和置信度来构建缓存；因此某些早期错误可能被缓存机制放大”的含义。

更直白地说：

    global cache 会记住模型认为可靠的历史预测。
    如果记住的是正确预测，它会帮忙；
    如果记住的是错误预测，它也可能帮倒忙。

在当前实验中，`add_global` 很可能就是这种情况。

## 11. 当前 E2 global cache 结论

当前 E2 global cache smoke test 得出以下结论：

1. `manual_full_llm_fusion + zs_global` 平均准确率为 53.18；
2. `manual_full + zs_global` 平均准确率为 52.66；
3. 文本原型融合在 global cache 下仍然带来 +0.52 的提升；
4. 说明 E1 的文本收益可以部分传递到 Point-Cache global cache；
5. 7 个损坏类型中有 6 个提升，说明该趋势不是单个损坏类型偶然造成的；
6. `add_global` 出现明显下降，说明 cache 机制可能会放大部分初始伪标签偏差；
7. 后续需要继续运行 full Point-Cache 两组实验，观察 local cache 是否会缓解、放大或抵消这一现象。

## 12. 下一步计划

下一步进入 E2 full Point-Cache smoke test：

| 编号 | 设置 | 文本方法 |
|---|---|---|
| 00_3 | zs_global_local | manual_full |
| 00_4 | zs_global_local | manual_full_llm_fusion |

需要注意：

`zs_global_local` 必须使用：

    --cache-type hierarchical

如果误用：

    --cache-type global

会导致 hierarchical cache runner 期望 6 个返回值，但 `get_logits()` 只返回 5 个值，从而报错：

    ValueError: not enough values to unpack (expected 6, got 5)

该问题已定位为 E2 公共脚本中的 cache_type 设置错误，应通过根据 `CACHE_METHOD` 自动选择 `CACHE_TYPE` 修复：

    zs_global       -> CACHE_TYPE=global
    zs_global_local -> CACHE_TYPE=hierarchical
