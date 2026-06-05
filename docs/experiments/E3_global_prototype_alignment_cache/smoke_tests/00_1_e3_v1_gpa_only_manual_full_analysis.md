# E3-V1 初版结果分析：顺序式 GPA Cache + GPA-only center + manual_full

更新日期：2026-06-05

## 1. 实验定位

本实验属于：

    E3_global_prototype_alignment_cache

当前方法版本：

    E3-V1：顺序式 Global Prototype-Alignment Cache

当前原型中心来源：

    Center-A：GPA-only center

当前文本方法：

    manual_full

当前 cache 设置：

    zs_global_local

对应结果目录：

    Point-Cache/results/E3_global_prototype_alignment_cache/00_1_ulip_modelnetc_s2_zs_global_local_gpa_manual_full_smoke/

对比对象为 E2 中的原始完整 Point-Cache：

    E2 00_3：manual_full + zs_global_local

E2 对比结果：

    manual_full + 原始 full Point-Cache = 54.00

## 2. 当前 E3-V1 初版实验结果

| 损坏类型 | E2 原始 full Point-Cache | E3-V1 GPA-only center | 差值 |
|---|---:|---:|---:|
| add_global | 47.81 | 50.36 | +2.55 |
| add_local | 46.68 | 48.22 | +1.54 |
| dropout_global | 59.20 | 56.12 | -3.08 |
| dropout_local | 56.69 | 56.81 | +0.12 |
| rotate | 62.07 | 60.09 | -1.98 |
| scale | 55.23 | 54.13 | -1.10 |
| jitter | 50.32 | 48.34 | -1.98 |
| 平均 | 54.00 | 53.44 | -0.56 |

## 3. 总体结论

当前 E3-V1 初版结果不是正向的。

平均准确率：

    E2 原始 full Point-Cache：54.00
    E3-V1 GPA-only center：53.44

下降：

    53.44 - 54.00 = -0.56

因此，当前组合：

    顺序式 GPA Cache
    + GPA-only center
    + 当前准入规则
    + 不修改最终预测加权公式

不能作为当前主方法。

## 4. 结果不是完全无效，而是不稳定

虽然平均结果下降，但并不是所有损坏类型都下降。

提升明显的损坏类型：

| 损坏类型 | 提升 |
|---|---:|
| add_global | +2.55 |
| add_local | +1.54 |
| dropout_local | +0.12 |

下降明显的损坏类型：

| 损坏类型 | 下降 |
|---|---:|
| dropout_global | -3.08 |
| rotate | -1.98 |
| jitter | -1.98 |
| scale | -1.10 |

这说明 GPA Cache 的思想在部分损坏类型上可能有效，但当前实现方式不稳定，不能稳定提升整体平均准确率。

## 5. 当前发现的关键问题：GPA Cache 在预构建阶段筛选作用不足

从日志看，预构建阶段经常出现：

    entropy cache total ≈ gpa local cache total

这说明：

    几乎所有进入 Global Entropy Cache 的样本，
    同时也进入了 GPA-controlled Local Cache。

这与最初希望的目标不完全一致。

最初目标是：

    GPA Cache 应该比 Global Entropy Cache 更严格，
    从而只允许更可靠、更靠近原型中心的样本贡献 local cache。

但当前代码中，GPA Cache 的启动和未满阶段过于宽松：

1. GPA Cache 未形成中心前，直接按低熵准入积累；
2. GPA Cache 已形成中心但未满时，仍然直接加入；
3. 只有 GPA Cache 满了以后，才启用低熵 + 距离约束。

由于当前每类缓存容量 K 较小，很多类别还没经历足够多的满后替换，预构建阶段就结束了。因此距离约束没有充分发挥筛选作用。

## 6. 当前缓存未满时的真实准入规则

### 6.1 Global Entropy Cache 未满时

如果预测类别 c 的 Global Entropy Cache 未满：

    当前样本直接进入该类 Global Entropy Cache；
    然后按熵从低到高排序。

只有当 Global Entropy Cache 已满后，才会判断：

    新样本熵是否低于该类缓存中当前最高熵样本。

### 6.2 Global Prototype-Alignment Cache 未满时

当前 E3-V1 的规则为：

    只有成功进入或替换 Global Entropy Cache 的样本，
    才有资格继续尝试进入 GPA Cache。

如果该类别 GPA Cache 样本数不足 min_center_size：

    直接进入 GPA Cache，用于建立初始中心。

如果该类别 GPA Cache 已形成中心，但仍未满：

    当前代码仍然直接加入 GPA Cache。

如果该类别 GPA Cache 已满：

    才启用严格规则：
        新样本熵更低；
        且新样本到原型中心的距离
        小于 GPA Cache 中最高熵样本到中心的距离。

因此，当前未满阶段过于宽松，是导致 GPA local cache 数量接近 entropy cache 数量的重要原因。

## 7. gpa_stats 缺失问题

当前结果目录下未找到：

    gpa_stats/

这说明当前运行没有成功保存 GPA 统计文件。

这不必然等价于 GPA Cache 状态延续 bug 未修复，但说明统计保存逻辑需要检查。

后续必须保证每个 corruption 保存 GPA 统计信息，至少包括：

1. Global Entropy Cache 每类数量；
2. GPA Cache 每类数量；
3. GPA-controlled Local Cache 每类数量；
4. GPA 加入次数；
5. GPA 替换次数；
6. GPA 因熵拒绝次数；
7. GPA 因距离拒绝次数；
8. 每次发生替换时，新样本和被替换样本的熵与距离。

其中第 8 点需要单独保存为替换事件日志。

建议文件名：

    gpa_replacement_events_<cor_type>.jsonl

每一行记录一次替换事件：

    {
      "phase": "build" 或 "test",
      "class_index": 类别编号,
      "new_entropy": 新样本熵,
      "old_entropy": 被替换样本熵,
      "new_distance": 新样本到原型中心距离,
      "old_distance": 被替换样本到原型中心距离,
      "decision": "replace"
    }

如果发生距离拒绝，也应记录：

    {
      "phase": "build" 或 "test",
      "class_index": 类别编号,
      "new_entropy": 新样本熵,
      "old_entropy": 当前最高熵样本熵,
      "new_distance": 新样本到原型中心距离,
      "old_distance": 当前最高熵样本到原型中心距离,
      "decision": "reject_distance"
    }

这样后续才能分析熵和距离是否一致，以及高熵样本是否真的更远。

## 8. 当前判断

当前结果支持如下判断：

### 判断 1：顺序式 GPA-only center 不是最佳方案

当前平均准确率下降 0.56，说明当前组合不能作为主方法。

### 判断 2：需要优先做原型中心来源消融

当前 GPA-only center 可能不够稳定。

下一步固定“顺序式关系”不变，只改变原型中心来源，测试：

| 名称 | 原型中心来源 | 目的 |
|---|---|---|
| Center-B | Entropy-only center | 使用 Global Entropy Cache 计算中心，验证更稳定中心是否有效 |
| Center-C | Entropy+GPA union center | 使用 Global Entropy Cache 与 GPA Cache 并集计算中心，验证融合中心是否有效 |

说明：

    Center-B 和 Center-C 都需要保持其他变量不变，
    仍然采用顺序式关系和当前 full Point-Cache 设置。

### 判断 3：需要后续测试 MCP-style 并列式方案

当前顺序式关系可能限制了 GPA Cache 的作用。

后续应实现：

    Entropy Cache 与 GPA Cache 并列更新，
    二者互不替换，
    再进一步设计它们如何影响 local cache 或 global logits。

即使 Center-B / Center-C 有改善，MCP-style 并列式方案仍应作为后续对照实验。

### 判断 4：需要后续测试距离优先准入规则

当前规则是：

    熵优先 + 最高熵样本距离校验

但高熵不一定距离大，因此后续应测试：

    距离优先 + 小于当前最大距离

该消融暂时不做，写入后续计划。

## 9. 下一步工作顺序

建议下一步按如下顺序推进：

1. 修复 GPA 统计保存；
2. 增加 replacement event 日志；
3. 新增 Center-B：Entropy-only center runner；
4. 新增 Center-C：Entropy+GPA union center runner；
5. 两张卡分别跑 Center-B 和 Center-C；
6. 对比 E2 原始 full Point-Cache、E3 Center-A、Center-B、Center-C；
7. 决定是否进入 MCP-style 并列式方案。

## 10. 当前阶段结论

E3-V1 初版结果虽然不是正向，但非常有诊断价值。

它说明：

    仅采用顺序式 GPA Cache + GPA-only center，
    并不能稳定提升 Point-Cache 的整体分类准确率。

主要问题可能不是“原型对齐思想无效”，而是：

1. GPA-only center 在 K 较小时不够稳定；
2. GPA Cache 未满阶段过于宽松；
3. 当前规则没有在预构建阶段形成足够筛选；
4. local cache 可能不仅需要靠近中心，也需要保持结构覆盖度。

因此，E3 不应停止，而应进入中心来源消融和并列式缓存方案验证。
