# E3-V3-C：候选池距离初始化 GPA-Cache 消融实验说明

## 1. 实验背景

当前 E3 的目标是在 Point-Cache 的 hierarchical cache 框架中引入 GPA-Cache，即 Global Prototype-Alignment Cache，用原型距离约束改善进入 local cache 的样本质量。

此前 E3-V2-C，即“并列式 GPA-Cache + Entropy/GPA 联合中心”方法，在 ULIP × ModelNet-C severity=2 × 7 corruption smoke test 上取得 54.04 的平均准确率，略高于 E2 原始 full Point-Cache baseline 的 54.00。这个结果说明并列式 GPA-Cache 和原型中心约束方向有一定潜力，但提升非常小。

进一步分析后认为，当前主要问题是 GPA-Cache 的初始化机制仍然不稳定：当某一类 GPA-Cache 未满时，前 K 个样本会直接进入 GPA-Cache。这些早期样本会强烈影响 GPA-Center 和 GPA-controlled local cache。如果前 K 个样本存在伪标签错误、类内结构不典型，或者只覆盖某个局部模式，就会污染后续的 GPA-Cache 更新和 local cache 写入。

因此，E3-V3-C 尝试用候选池初始化替代“前 K 个样本直接进入”的机制。

## 2. 实验目的

E3-V3-C 主要验证以下问题：

1. 每类先收集 2K 个候选样本，再从中选择 K 个样本初始化 GPA-Cache，是否比前 K 个样本直接进入更稳定；
2. 原型中心只由候选池构造，是否能避免 Entropy Cache 对 GPA-Cache 的过度干预；
3. GPA-Cache 满后的更新是否可以只依赖距离，而不再依赖熵；
4. GPA-Cache 替换后，同步更新 local cache 和 GPA-Center 是否能保持全局缓存与局部缓存一致；
5. 该机制能否超过 E3-V2-C 的 54.04，以及 E2 原始 full Point-Cache baseline 的 54.00。

## 3. 消融实验矩阵

E3-V3-C 不是单一实验，而是一组围绕“候选池初始化 GPA-Cache”的消融实验。当前消融矩阵包含两个维度：

1. 原型中心构造方式；
2. GPA-Cache 满后的在线更新规则。

### 3.1 原型中心构造方式

| 编号 | 中心来源 | 具体含义 | 实验目的 |
|---|---|---|---|
| C1 | Candidate-only center | 只使用该类别 2K 个候选池样本构造临时原型中心 | 验证纯候选池几何中心是否足够稳定，避免 Entropy Cache 影响 |
| C2 | Candidate + Entropy center | 同时使用该类别 2K 个候选池样本和 Entropy Cache 中样本构造临时原型中心 | 验证 Entropy Cache 是否可以作为候选池中心的稳定锚点 |

当前首先实现和运行的是 **C1**，即原型中心只来源于该类别 2K 个候选池样本。

### 3.2 GPA-Cache 满后的在线更新规则

| 编号 | 是否使用熵 | 替换条件 | 替换对象 | 具体含义 |
|---|---|---|---|---|
| U-a1 | 使用熵 | 新样本熵更低，且新样本到中心的距离小于 GPA-Cache 中最高熵样本到中心的距离 | GPA-Cache 中最高熵样本 | 最接近 MCP Align Cache 风格，低熵作为门控，距离作为二次筛选 |
| U-a2 | 使用熵 | 新样本熵更低，且新样本到中心的距离小于 GPA-Cache 中最远样本到中心的距离 | GPA-Cache 中离中心最远样本 | 保留低熵门控，但替换对象改为破坏类内紧凑性的最远样本 |
| U-b | 不使用熵 | 新样本到中心的距离小于 GPA-Cache 中最远样本到中心的距离 | GPA-Cache 中离中心最远样本 | 纯距离更新规则，直接优化 GPA-Cache 的类内紧凑性 |

当前首先实现和运行的是 **U-b**，即 GPA-Cache 满后不再使用熵，只比较距离。

## 4. 当前首跑版本

当前首跑版本为：

```text
E3-V3-C1-Ub
= Candidate-only center
+ Distance-only update
```

完整中文名：

```text
候选池距离初始化 GPA-Cache：候选池中心 + 无熵距离更新
```

完整英文名：

```text
Candidate-pool distance initialization with candidate-only center and distance-only update
```

## 5. 当前首跑版本的具体规则

对于每个类别 c：

1. 测试样本到来后，通过 zero-shot 分支得到预测类别 c；
2. 如果类别 c 的 GPA-Cache 尚未初始化，则将该样本加入 `gpa_candidate_pool[c]`；
3. 当 `gpa_candidate_pool[c]` 达到 2K 个样本后，触发初始化；
4. 使用这 2K 个候选样本的 global feature 均值构造临时中心；
5. 计算每个候选样本到临时中心的 cosine distance；
6. 选择距离临时中心最近的 K 个样本进入正式 GPA-Cache；
7. 这 K 个样本对应的 patch centers 同步写入 GPA-controlled local cache；
8. 使用正式进入 GPA-Cache 的 K 个样本重新计算正式 GPA-Center。

GPA-Cache 初始化完成后，在线更新规则为：

```text
如果新样本到当前 GPA-Center 的距离
小于 GPA-Cache 中离 GPA-Center 最远样本的距离，
则用新样本替换该最远样本。
```

即：

```text
d_new < d_far
```

其中：

```text
d_new = distance(new_sample, gpa_center[c])
d_far = max_i distance(gpa_cache[c][i], gpa_center[c])
```

该规则不使用熵，也不再使用“最高熵样本”作为替换对象。

## 6. Local Cache 同步规则

GPA-Cache 与 GPA-controlled local cache 必须保持一一对应关系：

```text
gpa_cache[c][i]       对应第 i 个全局样本
gpa_local_cache[c][i] 对应该样本的 patch centers
```

当 GPA-Cache 发生替换时：

```text
gpa_cache[c][idx_far] = new_global_item
gpa_local_cache[c][idx_far] = new_local_item
```

这样可以避免 global GPA-Cache 已经更新，而 local cache 仍保留旧样本局部特征的问题。

## 7. GPA-Center 更新规则

每次 GPA-Cache 发生替换后，必须立即重算该类别的 GPA-Center：

```text
gpa_center[c] = mean(features of gpa_cache[c])
```

不能延迟更新，否则下一批样本会使用旧中心计算距离，导致更新规则失真。

## 8. 最终预测公式

当前实验暂时不调整最终预测公式，也不调整 local cache 权重。仍保持此前 E3 的基本预测结构：

```text
final_logits = zero-shot logits
             + global entropy cache logits
             + GPA-controlled local cache logits
             - negative cache logits
```

其中：

1. global cache logits 仍来自原始 Entropy Cache；
2. local cache logits 来自 GPA-controlled local cache；
3. negative cache 仍沿用原始 Point-Cache 逻辑；
4. 当前不引入 GPA global logits；
5. 当前不降低 local cache 权重。

后续如果发现 local cache 贡献过强，再单独做 local cache 权重消融。

## 9. 后续消融优先级

如果 E3-V3-C1-Ub 效果不好，后续建议按以下顺序继续消融：

| 优先级 | 实验编号 | 中心来源 | 更新规则 | 目的 |
|---|---|---|---|---|
| 1 | E3-V3-C1-Ub | 2K 候选池 | 无熵，替换最远样本 | 当前首跑，验证纯距离候选池初始化 |
| 2 | E3-V3-C1-Ua2 | 2K 候选池 | 低熵 + 替换最远样本 | 检查是否需要恢复熵门控，同时保留距离紧凑性优化 |
| 3 | E3-V3-C1-Ua1 | 2K 候选池 | 低熵 + 替换最高熵样本 | 对齐此前 E3-V2-C/MCP 风格规则 |
| 4 | E3-V3-C2-Ub | 2K 候选池 + Entropy Cache | 无熵，替换最远样本 | 检查 Entropy Cache 是否适合作为中心锚点 |
| 5 | E3-V3-C2-Ua2 | 2K 候选池 + Entropy Cache | 低熵 + 替换最远样本 | 验证中心锚点和熵门控的组合效果 |

## 10. 日志检查项

实验运行时需要重点检查：

1. 每类是否在候选池达到 2K 后才初始化 GPA-Cache；
2. 初始化时是否只用 2K candidate pool 构造临时中心；
3. 是否选择距离临时中心最近的 K 个样本进入 GPA-Cache；
4. GPA-Cache 满后是否只使用距离更新；
5. 替换对象是否是“离 GPA-Center 最远的样本”，不是“最高熵样本”；
6. 每次替换后 local cache 是否同步替换；
7. 每次替换后 GPA-Center 是否立即更新；
8. 最终预测权重是否保持不变。

推荐日志关键词：

```text
[E3-V3-C1-Ub] init class=...
[E3-V3-C1-Ub] update class=... d_new=... d_far=... replaced=True/False
[E3-V3-C1-Ub] center updated class=...
[E3-V3-C1-Ub] cache totals: entropy=..., gpa=..., gpa_local=..., candidate_left=...
```

## 11. 预期结果与失败解释

如果 E3-V3-C1-Ub 优于 E3-V2-C，则说明候选池距离初始化与纯距离更新可以缓解前 K 个样本污染问题，并增强 GPA-controlled local cache 的可靠性。

如果 E3-V3-C1-Ub 低于 E3-V2-C，但优于 E3-V3-B，则说明候选池初始化有一定价值，但完全去掉熵可能导致伪标签错误样本以紧凑簇形式进入 GPA-Cache。

如果 E3-V3-C1-Ub 明显低于 E2 baseline，则说明纯距离规则可能过度中心化，损失 local cache 多样性，后续需要回到带熵的 U-a1/U-a2 更新规则，或加入 diversity 约束。

## 12. 当前文件路径约定

本说明文档放在项目级 docs 目录下：

```text
/root/autodl-tmp/MCM-PC-2/docs/experiments/E3_global_prototype_alignment_cache/initialization_strategies/
```

代码、脚本和结果仍然放在 Point-Cache 目录下：

```text
代码：
/root/autodl-tmp/MCM-PC-2/Point-Cache/runners/E3_global_prototype_alignment_cache/

脚本：
/root/autodl-tmp/MCM-PC-2/Point-Cache/scripts/E3_global_prototype_alignment_cache/

结果：
/root/autodl-tmp/MCM-PC-2/Point-Cache/results/E3_global_prototype_alignment_cache/
```
