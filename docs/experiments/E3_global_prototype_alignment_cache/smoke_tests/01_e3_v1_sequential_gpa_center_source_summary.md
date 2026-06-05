# E3-V1 归总文档：顺序式 GPA Cache 中心来源消融

更新日期：2026-06-05

## 1. 阶段定位

E3-V1 主要验证：

    在完整 Point-Cache 设置中，
    采用顺序式 Global Prototype-Alignment Cache，
    是否能够通过控制 local cache 的样本来源提升整体分类准确率。

其中顺序式关系为：

    Global Entropy Cache
        ->
    Global Prototype-Alignment Cache
        ->
    GPA-controlled Local Cache

也就是说：

1. 样本先按照原始 Point-Cache 低熵规则尝试进入 Global Entropy Cache；
2. 只有通过 Global Entropy Cache 准入的样本，才有资格进入 GPA Cache；
3. 只有进入 GPA Cache 的样本，其 local patch centers 才进入 local cache；
4. 当前 E3-V1 不修改最终预测加权公式。

## 2. 保留内容

E3-V1 相关代码、脚本和实验结果全部保留。

原因：

1. E3-V1 是一个完整的负结果消融；
2. 它验证了“顺序式 GPA Cache + 当前准入规则”不是当前最佳方向；
3. 它为 E3-V2 并列式方案提供了明确动机；
4. 它有助于论文和实验记录中说明方法探索过程。

保留对象包括：

### 2.1 代码

    Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_gpa.py
    Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_gpa_entropy_only_center.py
    Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_gpa_entropy_gpa_union_center.py

### 2.2 Runner

    Point-Cache/runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_gpa.py
    Point-Cache/runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_gpa_entropy_only_center.py
    Point-Cache/runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_gpa_entropy_gpa_union_center.py

### 2.3 脚本

    Point-Cache/scripts/E3_global_prototype_alignment_cache/00_1_ulip_modelnetc_s2_zs_global_local_gpa_manual_full_smoke.sh
    Point-Cache/scripts/E3_global_prototype_alignment_cache/01_1_ulip_modelnetc_s2_zs_global_local_gpa_entropy_only_center_manual_full.sh
    Point-Cache/scripts/E3_global_prototype_alignment_cache/01_2_ulip_modelnetc_s2_zs_global_local_gpa_entropy_gpa_union_center_manual_full.sh

### 2.4 结果

    Point-Cache/results/E3_global_prototype_alignment_cache/

## 3. E3-V1 对比对象

E3-V1 的主要对比对象是 E2 中的原始 full Point-Cache：

    E2 00_3：manual_full + zs_global_local

其平均准确率为：

    54.00

## 4. E3-V1 三种中心来源

E3-V1 共测试了三种原型中心来源。

| 名称 | 中心来源 | 说明 |
|---|---|---|
| Center-A | GPA-only center | 只使用 GPA Cache 中样本计算类别中心 |
| Center-B | Entropy-only center | 只使用 Global Entropy Cache 中样本计算类别中心 |
| Center-C | Entropy+GPA union center | 使用 Global Entropy Cache 和 GPA Cache 的并集计算类别中心 |

三种方案都保持：

- 顺序式 GPA Cache；
- manual_full 文本模板；
- zs_global_local；
- 不修改最终预测加权公式。

## 5. 总体结果

| 方法 | 中心来源 | 平均准确率 | 相对 E2 baseline |
|---|---|---:|---:|
| E2 原始 full Point-Cache | 无 GPA | 54.00 | 0.00 |
| E3-V1-A | GPA-only center | 53.44 | -0.56 |
| E3-V1-B | Entropy-only center | 52.43 | -1.57 |
| E3-V1-C | Entropy+GPA union center | 53.01 | -0.99 |

结论：

    E3-V1 三种中心来源均未超过 E2 原始 full Point-Cache baseline。

## 6. 分损坏类型结果

| 损坏类型 | E2 baseline | GPA-only | Entropy-only | Entropy+GPA union |
|---|---:|---:|---:|---:|
| add_global | 47.81 | 50.36 | 48.14 | 47.45 |
| add_local | 46.68 | 48.22 | 46.43 | 47.69 |
| dropout_global | 59.20 | 56.12 | 56.93 | 57.09 |
| dropout_local | 56.69 | 56.81 | 56.04 | 56.81 |
| rotate | 62.07 | 60.09 | 58.83 | 60.01 |
| scale | 55.23 | 54.13 | 53.08 | 53.04 |
| jitter | 50.32 | 48.34 | 47.53 | 48.95 |
| 平均 | 54.00 | 53.44 | 52.43 | 53.01 |

## 7. E3-V1 主要发现

### 7.1 顺序式 GPA Cache 不是当前最佳方向

三种中心来源均低于 E2 baseline，说明问题不只是中心来源，而是顺序式 GPA Cache 的结构本身可能存在限制。

当前顺序式结构中：

    GPA Cache 依赖 Global Entropy Cache 的准入。

这使得 GPA Cache 很难成为真正独立的缓存分支。

### 7.2 GPA Cache 未满阶段筛选作用不足

在当前实现中：

    如果 GPA Cache 未满，
    已经通过 Global Entropy Cache 准入的样本会直接进入 GPA Cache。

因此未满阶段：

    Global Entropy Cache 和 GPA Cache 的准入规则本质上几乎一样。

这导致预构建阶段经常出现：

    entropy cache total ≈ gpa cache total ≈ gpa local cache total

说明 GPA Cache 在预构建阶段并没有形成足够强的筛选作用。

### 7.3 单纯更换中心来源无法解决问题

Center-B 和 Center-C 的实验结果表明：

    使用 Entropy-only center 或 Entropy+GPA union center
    并不能解决 E3-V1 的整体下降问题。

因此继续在顺序式框架下只更换中心来源，价值已经不大。

## 8. E3-V1 当前结论

E3-V1 得到如下结论：

    顺序式 GPA Cache 在当前设置下不能稳定提升完整 Point-Cache。
    三种中心来源均未超过原始 full Point-Cache baseline。
    问题更可能来自缓存关系和准入机制，而不是单一中心来源。

因此，E3 下一步进入：

    E3-V2：并列式 Global Prototype-Alignment Cache

## 9. E3-V2 的动机

E3-V2 要解决 E3-V1 的核心问题：

    GPA Cache 不应只是 Global Entropy Cache 的附属层。

因此 E3-V2 改为：

    Global Entropy Cache 和 GPA Cache 并列更新、互不依赖、互不替换。

新样本到来后：

    一路尝试更新 Global Entropy Cache；
    另一路尝试更新 GPA Cache。

这样 GPA Cache 才能形成相对独立的候选集合。

## 10. 后续保留问题

E3-V1 不再作为主方法继续扩展，但保留如下后续分析价值：

1. 分析 gpa_replacement_events，观察高熵和距离是否一致；
2. 分析 GPA Cache 的替换和拒绝比例；
3. 与 E3-V2 并列式方案对照；
4. 判断 Point-Cache 中 local cache 更需要“中心紧致性”还是“结构覆盖度”。
