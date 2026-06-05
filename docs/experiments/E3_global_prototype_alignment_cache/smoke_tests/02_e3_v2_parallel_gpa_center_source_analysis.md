# E3-V2 归总分析：并列式 GPA Cache 中心来源消融

更新日期：2026-06-05

## 1. 实验定位

本实验属于：

    E3_global_prototype_alignment_cache

当前阶段：

    E3-V2：并列式 Global Prototype-Alignment Cache

E3-V2 的核心目标是验证：

    将 Global Entropy Cache 与 Global Prototype-Alignment Cache 从顺序式关系改为并列式关系后，
    是否能够提升完整 Point-Cache 的整体分类准确率。

E3-V2 仍然保持以下变量不变：

- 数据集：ModelNet-C
- 损坏强度：severity=2
- backbone：ULIP
- 文本方法：manual_full
- cache 设置：zs_global_local
- global cache logits：仍使用 Global Entropy Cache
- local cache：仍由 GPA Cache 控制
- 最终预测加权公式：暂不修改

唯一核心变化是：

    Global Entropy Cache 和 GPA Cache 并列更新，互不依赖，互不替换。

## 2. E3-V1 与 E3-V2 的区别

### 2.1 E3-V1：顺序式 GPA Cache

E3-V1 的关系为：

    Global Entropy Cache
        ->
    Global Prototype-Alignment Cache
        ->
    GPA-controlled Local Cache

也就是说：

    样本必须先通过 Global Entropy Cache 准入，
    才有资格进入 GPA Cache。

这种设计的问题是：

    GPA Cache 很容易成为 Global Entropy Cache 的附属层，
    尤其在 GPA Cache 未满时，
    通过 Global Entropy Cache 的样本会直接进入 GPA Cache。

因此，E3-V1 中 GPA Cache 没有形成足够独立的候选集合。

### 2.2 E3-V2：并列式 GPA Cache

E3-V2 的关系为：

    Global Entropy Cache
        与
    Global Prototype-Alignment Cache

并列更新。

新样本到来后：

    一路尝试更新 Global Entropy Cache；
    另一路独立尝试更新 GPA Cache。

两者互不依赖、互不替换。

这样做的动机是：

    让 GPA Cache 不再只是 Global Entropy Cache 的后置筛选层，
    而是形成一个相对独立的原型对齐缓存分支。

## 3. E3-V2 三种中心来源

E3-V2 继续保留三种原型中心来源消融。

| 方法编号 | 关系 | 中心来源 | 说明 |
|---|---|---|---|
| E3-V2-A | 并列式 | GPA-only center | 只使用 GPA Cache 中的样本计算类别中心 |
| E3-V2-B | 并列式 | Entropy-only center | 只使用 Global Entropy Cache 中的样本计算类别中心 |
| E3-V2-C | 并列式 | Entropy+GPA union center | 使用 Global Entropy Cache 与 GPA Cache 的并集计算类别中心 |

其中 E3-V2-C 是当前最重要的方案，因为它同时利用：

1. Global Entropy Cache 中较稳定的低熵样本；
2. GPA Cache 中经过原型对齐机制筛选的样本。

## 4. 总体结果

E2 原始 full Point-Cache baseline：

    manual_full + zs_global_local = 54.00

E3-V2 三组结果如下：

| 方法 | 缓存关系 | 中心来源 | 平均准确率 | 相对 E2 baseline |
|---|---|---:|---:|---:|
| E2 原始 full Point-Cache | 原始 | 无 GPA | 54.00 | 0.00 |
| E3-V2-A | 并列式 | GPA-only center | 53.70 | -0.30 |
| E3-V2-B | 并列式 | Entropy-only center | 53.15 | -0.85 |
| E3-V2-C | 并列式 | Entropy+GPA union center | 54.04 | +0.04 |

结论：

    E3-V2-C 是当前唯一超过 E2 baseline 的 GPA 方案。

虽然提升幅度很小：

    54.04 - 54.00 = +0.04

但它说明：

    并列式 GPA Cache + Entropy+GPA union center 是当前最有潜力的方向。

## 5. E3-V1 与 E3-V2 总体对比

| 方法 | 关系 | 中心来源 | 平均准确率 |
|---|---|---|---:|
| E2 baseline | 原始 full Point-Cache | 无 GPA | 54.00 |
| E3-V1-A | 顺序式 | GPA-only center | 53.44 |
| E3-V1-B | 顺序式 | Entropy-only center | 52.43 |
| E3-V1-C | 顺序式 | Entropy+GPA union center | 53.01 |
| E3-V2-A | 并列式 | GPA-only center | 53.70 |
| E3-V2-B | 并列式 | Entropy-only center | 53.15 |
| E3-V2-C | 并列式 | Entropy+GPA union center | 54.04 |

从 V1 到 V2 的提升：

| 中心来源 | V1 顺序式 | V2 并列式 | 提升 |
|---|---:|---:|---:|
| GPA-only center | 53.44 | 53.70 | +0.26 |
| Entropy-only center | 52.43 | 53.15 | +0.72 |
| Entropy+GPA union center | 53.01 | 54.04 | +1.03 |

结论：

    在三种中心来源下，并列式均优于顺序式。

这说明：

    E3-V1 的主要问题并不只是中心来源，
    而是 GPA Cache 作为 Global Entropy Cache 后置层时独立性不足。

## 6. 分损坏类型对比

| 损坏类型 | E2 baseline | V2-A GPA-only | V2-B Entropy-only | V2-C Entropy+GPA union |
|---|---:|---:|---:|---:|
| add_global | 47.81 | 48.58 | 47.37 | 46.84 |
| add_local | 46.68 | 48.34 | 48.14 | 50.49 |
| dropout_global | 59.20 | 57.78 | 57.78 | 58.31 |
| dropout_local | 56.69 | 55.92 | 56.65 | 56.04 |
| rotate | 62.07 | 60.98 | 59.76 | 61.67 |
| scale | 55.23 | 55.59 | 54.46 | 55.06 |
| jitter | 50.32 | 48.74 | 47.89 | 49.88 |
| 平均 | 54.00 | 53.70 | 53.15 | 54.04 |

## 7. 对 V2-A：GPA-only center 的分析

E3-V2-A 的结果为：

    53.70

相对 E2 baseline：

    -0.30

相对 E3-V1-A：

    +0.26

说明：

    即使中心来源不变，只把顺序式改为并列式，也能带来一定改善。

这说明并列式关系本身是有意义的。

但是，GPA-only center 仍然低于 baseline，原因可能是：

1. GPA Cache 初始阶段仍然存在冷启动问题；
2. GPA Cache 自身样本数量较少，当前每类容量 K=3，中心容易受少数样本影响；
3. GPA-only center 可能过度依赖 GPA Cache 自身，缺少 Global Entropy Cache 提供的稳定低熵分布参考。

因此，GPA-only center 不适合作为当前主方案。

## 8. 对 V2-B：Entropy-only center 的分析

E3-V2-B 的结果为：

    53.15

相对 E2 baseline：

    -0.85

相对 E3-V1-B：

    +0.72

说明：

    并列式关系依然带来提升，
    但单独使用 Global Entropy Cache 作为中心来源效果仍然不好。

这说明：

    Entropy Cache 的中心虽然更稳定，
    但它并不一定能代表 GPA Cache 希望捕获的“原型对齐”样本分布。

Entropy-only center 的问题可能是：

1. 它忽略了 GPA Cache 自身的对齐样本；
2. 它本质上更接近原始 Point-Cache 的低熵缓存逻辑；
3. 它无法充分利用新增 GPA Cache 的结构信息。

因此，Entropy-only center 也不适合作为当前主方案。

## 9. 对 V2-C：Entropy+GPA union center 的重点分析

E3-V2-C 的结果为：

    54.04

相对 E2 baseline：

    +0.04

相对 E3-V1-C：

    +1.03

相对 E3-V2-A：

    +0.34

相对 E3-V2-B：

    +0.89

这是当前 E3 中最好的结果。

E3-V2-C 的优势在于：

    它同时利用了 Global Entropy Cache 和 GPA Cache。

其中：

- Global Entropy Cache 提供低熵、较稳定的测试样本分布；
- GPA Cache 提供经过原型对齐约束的候选样本；
- 两者联合形成视觉原型中心，可以缓解单一缓存中心不稳定的问题。

因此，E3-V2-C 支持如下判断：

    GPA Cache 不应依附于 Entropy Cache 顺序更新；
    同时，原型中心也不应只依赖单一缓存；
    Global Entropy Cache 与 GPA Cache 的联合中心更稳定。

## 10. E3-V2-C 的局限性

虽然 E3-V2-C 超过 baseline，但提升幅度很小：

    +0.04

而且从分项结果看，主要提升来自：

    add_local: 50.49 vs 46.68，提升 +3.81

但是在多数损坏类型上仍然低于 E2 baseline：

| 损坏类型 | V2-C - E2 |
|---|---:|
| add_global | -0.97 |
| dropout_global | -0.89 |
| dropout_local | -0.65 |
| rotate | -0.40 |
| scale | -0.17 |
| jitter | -0.44 |

这说明：

    当前 E3-V2-C 虽然方向正确，但稳定性仍然不足。
    它还不能作为最终主方法，只能说明并列式 GPA + union center 有潜力。

## 11. 为什么并列式优于顺序式

顺序式的主要问题是：

    GPA Cache 依赖 Global Entropy Cache 的准入。

这种结构会导致：

1. GPA Cache 的候选样本受 Entropy Cache 限制；
2. GPA Cache 在未满阶段与 Entropy Cache 区分不明显；
3. GPA Cache 很难形成独立的原型对齐分支。

并列式解决了这个问题：

    GPA Cache 与 Global Entropy Cache 同步更新、互不依赖。

因此：

    GPA Cache 可以形成相对独立的候选集合；
    Entropy Cache 则继续提供稳定低熵样本；
    两者可以互补，而不是前后依赖。

这解释了为什么三种中心来源下，V2 都比 V1 更好。

## 12. 为什么 Entropy+GPA union center 最好

三种中心来源代表三种不同假设：

### 12.1 GPA-only center

假设：

    GPA Cache 自己可以形成可靠中心。

结果：

    53.70，低于 baseline。

说明：

    GPA Cache 自身可能仍然受到初始化样本和 K 较小的影响。

### 12.2 Entropy-only center

假设：

    Global Entropy Cache 更稳定，因此用它作为中心更好。

结果：

    53.15，低于 baseline。

说明：

    Entropy Cache 虽然稳定，但它不能充分表达 GPA Cache 中对齐样本的分布。

### 12.3 Entropy+GPA union center

假设：

    Global Entropy Cache 和 GPA Cache 可以互补。

结果：

    54.04，略高于 baseline。

说明：

    联合中心能同时利用低熵稳定性和原型对齐样本的信息，
    是当前三个中心来源中最合理的选择。

## 13. 当前阶段结论

E3-V2 的阶段性结论为：

    并列式 GPA Cache 明显优于顺序式 GPA Cache；
    Entropy+GPA union center 是当前最佳中心来源；
    E3-V2-C 首次超过 E2 原始 full Point-Cache baseline；
    但提升幅度较小，仍需继续改进 GPA Cache 初始化机制。

因此，E3 的下一阶段不应继续单纯更换中心来源，而应转向：

    GPA Cache 初始化机制改进。

## 14. 后续改进方向

后续重点是解决 GPA Cache 的初始化问题。

当前 GPA Cache 的问题是：

    缓存未满时，前 K 个样本仍然较容易直接进入 GPA Cache；
    这些样本会影响原型中心和 local cache；
    如果早期样本不够可靠，会影响后续替换轨迹。

后续可以尝试三类初始化改进。

### 14.1 Init-A：先用 Global Entropy Cache 初始化中心

思路：

    先构建 Global Entropy Cache；
    用 Global Entropy Cache 计算初始视觉中心；
    再启用 GPA Cache 的距离约束。

优点：

    避免 GPA Cache 完全冷启动。

问题：

    最终中心来源仍需要结合实验判断。

### 14.2 Init-B：延迟 local cache 写入

思路：

    GPA Cache 可以先收集候选；
    但在筛选完成前，不立即将局部特征写入 local cache；
    只有最终进入 GPA Cache 的 K 个样本，其局部特征才写入 local cache。

目的：

    避免早期未经充分筛选的样本污染 local cache。

### 14.3 Init-C：候选池初始化

思路：

    每类先收集 2K 或 3K 个候选样本；
    使用 Global Entropy Cache + GPA candidate pool 计算联合中心；
    根据熵和距离筛出 K 个样本进入 GPA Cache；
    只有这 K 个样本的局部特征进入 local cache。

推荐优先级：

    Init-C 最高。

原因：

    它直接解决“前 K 个样本无条件进入 GPA Cache”的问题，
    并且可以沿用 E3-V2-C 证明有效的 Entropy+GPA union center 思想。

## 15. 下一步建议

下一步进入：

    E3-V3：GPA Cache 初始化机制改进

优先实现：

    Init-C：候选池初始化
    中心来源：Global Entropy Cache + GPA candidate pool union center

暂时不引入文本原型。

原因：

    当前 E3-V2-C 已经说明视觉缓存内部还有改进空间；
    应先解决 GPA Cache 初始化问题，
    再考虑引入文本原型作为语义锚点。
