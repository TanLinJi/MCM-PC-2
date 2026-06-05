# E3-V1-A 诊断记录：顺序式 GPA Cache + GPA-only center

更新日期：2026-06-05

## 1. 当前实验设置

实验名称：

    E3_global_prototype_alignment_cache

当前版本：

    E3-V1-A

含义：

    顺序式 Global Prototype-Alignment Cache
    + GPA-only center
    + manual_full
    + zs_global_local

对比对象：

    E2 00_3：manual_full + 原始 full Point-Cache

## 2. 当前结果

| 损坏类型 | E2 原始 full Point-Cache | E3-V1-A | 差值 |
|---|---:|---:|---:|
| add_global | 47.81 | 50.36 | +2.55 |
| add_local | 46.68 | 48.22 | +1.54 |
| dropout_global | 59.20 | 56.12 | -3.08 |
| dropout_local | 56.69 | 56.81 | +0.12 |
| rotate | 62.07 | 60.09 | -1.98 |
| scale | 55.23 | 54.13 | -1.10 |
| jitter | 50.32 | 48.34 | -1.98 |
| 平均 | 54.00 | 53.44 | -0.56 |

## 3. 总体判断

当前 E3-V1-A 不是正向结果。

虽然 add_global 和 add_local 有明显提升，但平均准确率低于 E2 原始 full Point-Cache。

这说明当前组合：

    顺序式 GPA Cache
    + GPA-only center
    + 当前准入规则
    + 不修改最终预测加权公式

不能作为当前主方法。

## 4. 核心问题：GPA Cache 未满时没有形成更严格筛选

当前顺序式逻辑为：

    Global Entropy Cache
        ->
    Global Prototype-Alignment Cache
        ->
    Local Cache

其中样本必须先通过 Global Entropy Cache，才有资格进入 GPA Cache。

但是当前 GPA Cache 的未满阶段规则是：

    如果 GPA Cache 未满，
    则已经通过 Global Entropy Cache 准入的样本会直接进入 GPA Cache。

因此，在 GPA Cache 未满时：

    Global Entropy Cache 和 GPA Cache 的准入条件本质上几乎一样。

这会导致：

    预构建阶段 GPA-controlled Local Cache 的样本数量
    与 Global Entropy Cache 的样本数量几乎一致。

也就是说，当前 GPA Cache 没有在预构建阶段发挥足够的筛选作用。

## 5. 删除 min_center_size 的原因

此前设计中存在：

    min_center_size

用于表示 GPA Cache 中样本数量达到一定规模后再形成原型中心。

但在当前代码逻辑中：

    GPA Cache 未满时直接加入；
    GPA Cache 满后才启用距离约束。

因此，min_center_size 实际不改变准入逻辑。

它既不能阻止 GPA Cache 在未满阶段直接加入样本，也不能使距离约束更早生效。

所以当前版本中删除 min_center_size，避免文档和代码产生误导。

## 6. 当前缓存未满时的准入条件

### 6.1 Global Entropy Cache 未满

如果预测类别 c 的 Global Entropy Cache 未满：

    样本直接进入 Global Entropy Cache；
    然后按熵从低到高排序。

只有当 Global Entropy Cache 已满时，才会比较：

    新样本熵 < 当前最高熵样本熵

### 6.2 Global Prototype-Alignment Cache 未满

在当前顺序式 E3-V1-A 中：

    只有已经进入或替换 Global Entropy Cache 的样本，
    才有资格进入 GPA Cache。

如果该类 GPA Cache 未满：

    样本直接进入 GPA Cache；
    对应 local patch centers 写入 GPA-controlled Local Cache。

只有当 GPA Cache 已满时，才启用：

    新样本熵 < GPA Cache 中最高熵样本熵
    且
    新样本到 GPA 原型中心的距离
    <
    GPA Cache 中最高熵样本到 GPA 原型中心的距离

## 7. GPA Cache 状态延续与统计保存修正

本阶段确认并修正了代码中关于 GPA Cache 状态命名不一致的问题。

之前文件中曾出现：

    runtime_gpa_cache

该变量名容易误导为：

    预构建阶段的 GPA Cache 和正式测试阶段的 GPA Cache 是两个不同缓存。

当前设计要求：

    预构建阶段形成的 GPA Cache，
    在正式测试阶段继续沿用并更新。

因此代码已统一为：

    gpa_cache

当前关键状态应为：

    build_cache_in_advance 返回 entropy_cache、gpa_cache、gpa_local_cache、stats；
    run_test_tda 接收 entropy_cache、gpa_cache、gpa_local_cache、build_stats；
    在线阶段继续更新同一个 gpa_cache；
    _save_gpa_stats 保存真实的 gpa_cache。

对应源码检查应包含：

    return entropy_cache, gpa_cache, gpa_local_cache, stats
    entropy_cache, gpa_cache, gpa_local_cache, build_stats = build_cache_in_advance(...)
    _save_gpa_stats(..., gpa_cache, ...)

## 8. 必须新增替换事件日志

后续实验必须保存替换事件日志：

    gpa_replacement_events_<cor_type>.jsonl

每次发生替换或距离拒绝时，记录：

- phase；
- class_index；
- new_entropy；
- old_entropy；
- new_distance；
- old_distance；
- decision。

其中 decision 至少包括：

- replace；
- reject_entropy；
- reject_distance。

该日志用于分析：

    高熵样本是否一定距离更远；
    距离约束是否真的发挥作用；
    被拒绝样本和被替换样本之间的熵/距离关系。

## 9. 下一步方案

当前不继续把 E3-V1-A 作为主方法。

下一步固定“顺序式关系”不变，优先测试两个中心来源：

| 名称 | 原型中心来源 |
|---|---|
| Center-B | Entropy-only center |
| Center-C | Entropy+GPA union center |

两组实验需要同时补充 gpa_stats 和 replacement event 日志。

如果 Center-B / Center-C 仍不理想，则进入：

    E3-V2：并列式 GPA Cache

也就是：

    Global Entropy Cache 和 GPA Cache 并列更新，
    二者互不替换，
    再重新设计 local cache 来源和最终加权方式。
