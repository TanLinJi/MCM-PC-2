# E3-V3-C1-Ub 实验结果分析：候选池中心 + 无熵距离更新

## 1. 实验基本信息

实验编号：

```text
E3-V3-C1-Ub
```

完整方法名：

```text
候选池距离初始化 GPA-Cache：候选池中心 + 无熵距离更新
```

英文名：

```text
Candidate-pool distance initialization with candidate-only center and distance-only update
```

对应脚本：

```text
Point-Cache/scripts/E3_global_prototype_alignment_cache/04_1_ulip_modelnetc_s2_zs_global_local_parallel_gpa_candidate_pool_distance_c1_ub_manual_full.sh
```

对应结果目录：

```text
Point-Cache/results/E3_global_prototype_alignment_cache/04_1_ulip_modelnetc_s2_zs_global_local_parallel_gpa_candidate_pool_distance_c1_ub_manual_full
```

实验设置：

```text
Backbone: ULIP
Dataset: ModelNet-C
Severity: 2
Corruptions: add_global, add_local, dropout_global, dropout_local, rotate, scale, jitter
Prompt source: manual_full
Cache setting: zs_global_local
GPA candidate multiplier: 2
shot_capacity K: 3
n_cluster m: 3
alpha: 4.0
beta: 3.0
```

## 2. 方法回顾

E3-V3-C1-Ub 属于 E3-V3-C 候选池初始化 GPA-Cache 消融实验。

该方法包含两个核心设计：

### 2.1 C1：Candidate-only center

每个类别先收集 2K 个候选样本。候选样本来自 zero-shot 预测为该类别的测试样本。

当某个类别的候选池达到 2K 后，只使用这 2K 个候选样本的 global feature 均值构造临时原型中心：

```text
candidate_center_c = mean(features of 2K candidate samples predicted as class c)
```

然后计算每个候选样本到该临时中心的距离，选择距离最近的 K 个样本进入正式 GPA-Cache。

当前 C1 版本不把 Entropy Cache 混入临时中心，目的是保持候选池距离初始化的独立性，避免重新引入低熵偏置。

### 2.2 U-b：Distance-only update

GPA-Cache 初始化完成并达到 K 个样本后，在线更新不再使用熵门控。

对于新样本 x，如果它的 zero-shot 预测类别为 c，则计算：

```text
d_new = distance(x, gpa_center[c])
d_far = max_i distance(gpa_cache[c][i], gpa_center[c])
```

如果：

```text
d_new < d_far
```

则用新样本替换当前 GPA-Cache 中离 GPA-Center 最远的样本。

该规则不使用“最高熵样本”作为替换对象，也不要求新样本熵更低。

每次替换后，同步替换对应 local cache，并立即重算该类别 GPA-Center。

## 3. 实验结果

### 3.1 分 corruption 结果

| Corruption | E3-V3-C1-Ub | E2 baseline | Δ vs E2 | E3-V2-C | Δ vs E3-V2-C | E3-V3-B | Δ vs E3-V3-B |
|---|---:|---:|---:|---:|---:|---:|---:|
| add_global | 50.57 | 47.81 | +2.76 | 46.84 | +3.73 | 47.41 | +3.16 |
| add_local | 49.47 | 46.68 | +2.79 | 50.49 | -1.02 | 48.78 | +0.69 |
| dropout_global | 56.85 | 59.20 | -2.35 | 58.31 | -1.46 | 57.70 | -0.85 |
| dropout_local | 55.51 | 56.69 | -1.18 | 56.04 | -0.53 | 56.04 | -0.53 |
| rotate | 60.29 | 62.07 | -1.78 | 61.67 | -1.38 | 59.93 | +0.36 |
| scale | 52.55 | 55.23 | -2.68 | 55.06 | -2.51 | 54.13 | -1.58 |
| jitter | 48.46 | 50.32 | -1.86 | 49.88 | -1.42 | 48.74 | -0.28 |
| **Average** | **53.39** | **54.00** | **-0.61** | **54.04** | **-0.66** | **53.25** | **+0.14** |

### 3.2 整体对比

E3-V3-C1-Ub 的平均准确率为：

```text
53.39
```

与关键方法对比：

```text
E2 manual_full + 原始 full Point-Cache baseline: 54.00
E3-V2-C 并列式 GPA-Cache + Entropy/GPA 联合中心: 54.04
E3-V3-B Entropy-bootstrap 初始化: 53.25
E3-V3-C1-Ub 候选池中心 + 无熵距离更新: 53.39
```

因此：

```text
E3-V3-C1-Ub 比 E3-V3-B 高 +0.14；
E3-V3-C1-Ub 比 E2 baseline 低 -0.61；
E3-V3-C1-Ub 比 E3-V2-C 低 -0.66。
```

## 4. 结果分析

### 4.1 候选池初始化方向没有被完全否定

E3-V3-C1-Ub 比 E3-V3-B 高 0.14。这说明相较于直接用 Entropy Cache 启动 GPA-Cache，候选池距离初始化稍微更合理。

E3-V3-B 的主要问题是 GPA-Cache 初始化过度依赖 Entropy Cache，导致 GPA-Cache 与 Entropy Cache 高度重合，削弱了 GPA-Cache 作为独立原型对齐缓存的作用。

E3-V3-C1-Ub 改为只用 2K candidate pool 构造临时中心，并从候选池中选择距离中心最近的 K 个样本进入 GPA-Cache，因此它在一定程度上恢复了 GPA-Cache 的独立性。

但这个收益很弱，平均只提升 0.14，不能说明当前版本已经有效。

### 4.2 纯距离更新对 additive corruption 有明显帮助

E3-V3-C1-Ub 在 add_global 和 add_local 上提升明显：

```text
add_global: 47.81 -> 50.57, +2.76 vs E2
add_local : 46.68 -> 49.47, +2.79 vs E2
```

这说明纯距离候选池初始化和距离更新能够在 additive noise 场景下筛出更紧凑的类内样本。对于加点扰动，原始点云结构仍然相对保留，距离中心最近的样本可能确实更接近类别主体结构，因此 GPA-controlled local cache 对这类扰动有帮助。

### 4.3 纯距离更新在 dropout、scale、jitter、rotate 上不稳定

E3-V3-C1-Ub 在多数非 additive corruption 上下降：

```text
dropout_global: -2.35 vs E2
dropout_local : -1.18 vs E2
rotate        : -1.78 vs E2
scale         : -2.68 vs E2
jitter        : -1.86 vs E2
```

这说明纯距离规则可能过度追求类内紧凑性，导致 GPA-Cache 越来越集中在某个中心模式附近。这样会降低 local cache 的覆盖度和多样性。

对于 dropout、scale、jitter、rotate 这类扰动，同一类别的样本可能在特征空间中形成多个子模式。纯距离更新会倾向于保留最靠近单一中心的样本，而不是保留多个代表性模式。因此，GPA-local-cache 可能变得更干净但更窄，最终导致整体鲁棒性下降。

### 4.4 E3-V2-C 的“熵 + 距离”仍然更稳

E3-V2-C 平均准确率为 54.04，仍然高于 E3-V3-C1-Ub 的 53.39。

这说明此前 E3-V2-C 中的低熵门控仍然有作用。虽然 E3-V2-C 仍存在 GPA-Cache 前 K 个样本直接进入的问题，但它在满后更新时使用低熵和距离双重约束，因此比纯距离更新更稳。

E3-V3-C1-Ub 的结果说明：

```text
候选池初始化可以保留；
但 GPA-Cache 满后的无熵距离更新过于激进。
```

## 5. 当前结论

E3-V3-C1-Ub 没有达到预期目标。

该方法相较 E3-V3-B 有轻微提升，说明候选池初始化方向仍有价值；但它低于 E2 baseline 和 E3-V2-C，说明“无熵、纯距离替换最远样本”的在线更新规则不够稳定。

当前最合理的判断是：

```text
候选池距离初始化可以继续保留；
但 GPA-Cache 满后的更新规则需要恢复熵门控。
```

因此，下一步优先做：

```text
E3-V3-C1-Ua2：
Candidate-only center
+ low-entropy gate
+ replace farthest-to-center sample
```

同时，为了完整对照此前 E3-V2-C/MCP 风格规则，也应并行做：

```text
E3-V3-C1-Ua1：
Candidate-only center
+ low-entropy gate
+ replace highest-entropy sample
```

## 6. 下一步实验计划

### 6.1 E3-V3-C1-Ua2

完整含义：

```text
候选池距离初始化 GPA-Cache：候选池中心 + 低熵门控 + 替换最远样本
```

更新规则：

```text
如果新样本熵低于当前 GPA-Cache 中最高熵样本，
并且新样本到 GPA-Center 的距离小于当前 GPA-Cache 中最远样本到 GPA-Center 的距离，
则用新样本替换当前 GPA-Cache 中离 GPA-Center 最远的样本。
```

目的：

```text
验证 E3-V3-C1-Ub 的下降是否来自完全去掉熵门控。
```

### 6.2 E3-V3-C1-Ua1

完整含义：

```text
候选池距离初始化 GPA-Cache：候选池中心 + 低熵门控 + 替换最高熵样本
```

更新规则：

```text
如果新样本熵低于当前 GPA-Cache 中最高熵样本，
并且新样本到 GPA-Center 的距离小于该最高熵样本到 GPA-Center 的距离，
则用新样本替换当前 GPA-Cache 中最高熵样本。
```

目的：

```text
对齐此前 E3-V2-C 和 MCP Align Cache 风格规则，判断在候选池初始化后，替换最高熵样本是否比替换最远样本更稳。
```

## 7. 记录结论

当前应将 E3-V3-C1-Ub 记录为一个中间负结果：

```text
E3-V3-C1-Ub 比 Entropy-bootstrap 初始化略好，但仍低于 E2 和 E3-V2-C。
它说明候选池初始化有潜力，但纯距离在线更新不够稳。
后续需要恢复熵门控，并比较替换最远样本与替换最高熵样本两种规则。
```
