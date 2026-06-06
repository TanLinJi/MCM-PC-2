# E3-V3-C1 系列结果分析：候选池单中心方法对加噪有效，但对几何结构变化失效

## 1. 分析背景

当前 E3 实验的目标是在 Point-Cache 的 hierarchical cache 框架中引入 GPA-Cache，即 Global Prototype-Alignment Cache，用原型距离约束改善进入 local cache 的样本质量。

此前围绕 GPA-Cache 初始化问题，已经完成以下相关实验：

```text
E3-V2-C：
并列式 GPA-Cache + Entropy/GPA 联合中心
平均准确率：54.04

E3-V3-B：
Entropy-bootstrap 初始化 GPA-Cache
平均准确率：53.25

E3-V3-C1-Ub：
候选池中心 + 无熵距离更新
平均准确率：53.39

E3-V3-C1-Ua2：
候选池中心 + 低熵门控 + 替换最远样本
平均准确率：53.68

E3-V3-C1-Ua1：
候选池中心 + 低熵门控 + 替换最高熵样本
平均准确率：54.02
```

其中，E3-V3-C1 系列的共同特点是：

```text
C1：
类别原型中心只来源于每类 2K candidate pool；
不使用 Entropy Cache 参与初始化中心构造。
```

不同点在于 GPA-Cache 满后的更新规则：

```text
U-b：
无熵，只看距离，替换离 GPA-Center 最远样本。

U-a2：
低熵门控 + 距离约束，替换离 GPA-Center 最远样本。

U-a1：
低熵门控 + 距离约束，替换 GPA-Cache 中最高熵样本。
```

本轮分析的核心现象是：

```text
E3-V3-C1 系列方法的正收益几乎都集中在 add_global 和 add_local；
而在 dropout_global、dropout_local、rotate、scale、jitter 上大多下降。
```

该现象说明，当前候选池单中心 GPA 机制更像是一种“去外点噪声”的缓存净化策略，而不是一种能够覆盖复杂几何结构变化的通用鲁棒缓存策略。

---

## 2. 实验结果汇总

### 2.1 关键平均准确率

| 方法 | 平均准确率 | 相对 E2 baseline 54.00 | 相对 E3-V2-C 54.04 |
|---|---:|---:|---:|
| E3-V3-B：Entropy-bootstrap 初始化 | 53.25 | -0.75 | -0.79 |
| E3-V3-C1-Ub：候选池中心 + 无熵距离更新 | 53.39 | -0.61 | -0.65 |
| E3-V3-C1-Ua2：候选池中心 + 低熵门控 + 替换最远样本 | 53.68 | -0.32 | -0.36 |
| E3-V3-C1-Ua1：候选池中心 + 低熵门控 + 替换最高熵样本 | 54.02 | +0.02 | -0.02 |
| E3-V2-C：并列式 GPA-Cache + Entropy/GPA 联合中心 | 54.04 | +0.04 | 0.00 |

从整体平均准确率看，C1 系列内部呈现出明确趋势：

```text
E3-V3-C1-Ua1 > E3-V3-C1-Ua2 > E3-V3-C1-Ub > E3-V3-B
```

这说明：

```text
1. 候选池初始化比简单 Entropy-bootstrap 初始化略好；
2. 完全去掉熵门控会导致不稳定；
3. 恢复低熵门控后结果提升；
4. 替换最高熵样本比替换最远样本更稳。
```

### 2.2 分 corruption 结果

| corruption | E2 baseline | E3-V2-C | E3-V3-B | C1-Ub | C1-Ua2 | C1-Ua1 |
|---|---:|---:|---:|---:|---:|---:|
| add_global | 47.81 | 46.84 | 47.41 | 50.57 | 50.81 | 50.53 |
| add_local | 46.68 | 50.49 | 48.78 | 49.47 | 50.04 | 50.57 |
| dropout_global | 59.20 | 58.31 | 57.70 | 56.85 | 56.85 | 56.85 |
| dropout_local | 56.69 | 56.04 | 56.04 | 55.51 | 54.82 | 56.32 |
| rotate | 62.07 | 61.67 | 59.93 | 60.29 | 60.70 | 61.02 |
| scale | 55.23 | 55.06 | 54.13 | 52.55 | 53.36 | 53.36 |
| jitter | 50.32 | 49.88 | 48.74 | 48.46 | 49.15 | 49.51 |
| **Average** | **54.00** | **54.04** | **53.25** | **53.39** | **53.68** | **54.02** |

E3-V3-C1-Ua1 是当前 C1 系列最好的版本，但它的正收益仍然主要来自 add_global 和 add_local：

```text
add_global: 50.53 vs E2 47.81, +2.72
add_local : 50.57 vs E2 46.68, +3.89
```

而在其他 corruption 上：

```text
dropout_global: 56.85 vs E2 59.20, -2.35
dropout_local : 56.32 vs E2 56.69, -0.37
rotate        : 61.02 vs E2 62.07, -1.05
scale         : 53.36 vs E2 55.23, -1.87
jitter        : 49.51 vs E2 50.32, -0.81
```

因此，E3-V3-C1-Ua1 的平均值能接近 E2 baseline，主要依赖 add_global 和 add_local 上的大幅提升；非 additive corruption 仍然整体低于 baseline。

---

## 3. add_global 和 add_local 代表什么

在 ModelNet-C 的 7 种 corruption 中：

```text
add_global     ：添加全局外点 / 全局离群点
add_local      ：添加局部外点 / 局部离群点
dropout_global ：删除全局结构
dropout_local  ：删除局部部件
rotate         ：旋转
scale          ：缩放
jitter         ：点坐标抖动
```

其中，add_global 和 add_local 的共同本质是：

```text
在原始点云上额外加入不属于物体主体结构的噪声点。
```

它们与其他 corruption 的本质区别在于：

```text
add_global / add_local：
    原始物体主体结构仍然存在，只是额外加入噪声点或离群点。

dropout_global / dropout_local：
    原始物体的一部分结构真实消失。

rotate / scale：
    物体整体几何姿态或尺度发生系统性变化。

jitter：
    大量点的位置发生扰动，局部几何细节变得不稳定。
```

因此，add_global 和 add_local 更接近“噪声污染”问题；而 dropout、rotate、scale、jitter 更接近“结构变化 / 几何变化 / 局部特征失稳”问题。

---

## 4. 为什么 C1 系列对 add_global / add_local 有正收益

### 4.1 C1 初始化本质上是在选择类内中心样本

E3-V3-C1 的初始化规则是：

```text
每类先收集 2K 个候选样本；
用这 2K 个候选样本的 global feature 均值构造 candidate center；
计算候选样本到 candidate center 的距离；
选择距离最近的 K 个进入 GPA-Cache；
这些样本对应的 patch centers 写入 GPA-controlled local cache。
```

该规则天然偏向：

```text
离类别中心更近的样本；
形态更平均的样本；
受离群噪声影响更小的样本；
global feature 更靠近类内主簇的样本。
```

因此，C1 机制本质上是在从候选样本中筛选“类内中心样本”。

### 4.2 对 add_global / add_local，中心样本往往也是去噪样本

add_global 和 add_local 只是增加外点，不会删除原始主体结构。

如果某个样本被外点污染严重，它的 global feature 或 local patch centers 往往会被拉离类别主体中心。

此时，C1 的“选择离中心最近的 K 个样本”会自然过滤掉被外点拉偏的样本，保留主体结构更稳定、噪声影响更小的样本。

因此，在 add_global / add_local 下，C1 方法等价于一种隐式去噪机制：

```text
外点越多、污染越严重，样本越可能偏离类别中心；
距离中心筛选会排除这些偏离样本；
进入 GPA-Cache 和 local cache 的样本更干净。
```

这解释了为什么 C1 系列在 add_global 和 add_local 上稳定产生正收益。

### 4.3 GPA-controlled local cache 对 additive noise 更容易发挥作用

Point-Cache 的 local cache 依赖点云局部 patch centers。

在 add_global / add_local 中，虽然加入了外点，但物体主体结构仍然存在，因此测试样本的主体 local features 仍然可以和缓存中的干净 local features 匹配。

也就是说：

```text
additive noise 场景下：
    查询样本的主体结构仍然存在；
    GPA-local-cache 中的中心样本 local features 更干净；
    local cache 检索能够增强正确类别。
```

因此，C1 系列在 additive corruption 上提升明显。

---

## 5. 为什么 C1 系列对 dropout 失效

### 5.1 dropout 的问题不是多了噪声，而是少了结构

dropout_global 和 dropout_local 的本质是：

```text
点云的一部分真实结构被删除。
```

这与 add_global / add_local 完全不同。

additive corruption 是“额外多了一些不该有的点”，但 dropout 是“原来该有的点没了”。

C1 机制可以筛掉外点污染严重的样本，但无法恢复已经缺失的结构。

### 5.2 干净 local cache 可能与缺失结构的查询样本不匹配

C1 选择的通常是更接近类别平均形态的样本。

这些样本的 local cache 可能更完整、更中心化。

但 dropout 测试样本的局部结构已经缺失。

当查询样本缺失某些局部结构时，local cache 中完整样本的 patch centers 不一定能与其正确匹配。

因此：

```text
additive corruption：
    查询样本主体结构还在，可以匹配干净 local cache。

dropout corruption：
    查询样本结构缺失，和完整 local cache 存在局部 mismatch。
```

这会导致 GPA-controlled local cache 在 dropout 下不但无法提供稳定增益，甚至可能增强错误类别。

### 5.3 单中心筛选还可能削弱 dropout 所需的结构多样性

dropout 样本可能呈现多种缺失模式。

例如同一类物体，可能缺失顶部、底部、局部边缘或主体的一部分。

如果 GPA-Cache 只保留最靠近单一中心的样本，那么 local cache 主要覆盖完整且平均的结构，而不能覆盖多种缺失模式。

因此，C1 单中心机制对 dropout 的底层缺陷是：

```text
它提高了缓存样本的中心性；
但没有提高对缺失结构模式的覆盖。
```

---

## 6. 为什么 C1 系列对 rotate / scale 失效

### 6.1 rotate / scale 是系统性几何变换

rotate 和 scale 不是局部噪声，而是对整个点云进行几何变换：

```text
rotate：整体姿态改变；
scale ：整体尺度改变。
```

如果点云编码器对这些变换不是完全不变，那么同一类别样本在 feature space 中会发生整体偏移。

### 6.2 离中心最近不等于适应几何变换

C1 的核心假设是：

```text
离类别中心越近，样本越可靠。
```

但在 rotate / scale 下，测试流中的样本可能整体分布偏离原始中心。

此时，离候选池单中心最近的样本不一定最能代表当前变换模式。

换句话说：

```text
对于 additive noise：
    类别中心接近干净主体结构，距离中心近通常代表噪声少。

对于 rotate / scale：
    类别中心可能代表标准姿态或平均尺度；
    但当前样本可能形成新的变换子簇；
    离原中心近不一定代表适合当前查询分布。
```

因此，C1 的单中心 compactness 假设在几何变换下不可靠。

### 6.3 需要的是多模式覆盖，而不是单中心紧凑

rotate / scale 可能使同一类别形成多个子分布。

此时，缓存应该覆盖多个模式，而不是只压缩到一个中心附近。

C1 系列方法会让 GPA-Cache 更紧凑，但这种紧凑性可能以牺牲多样性为代价。

因此，rotate / scale 下的底层问题是：

```text
单中心 GPA-Cache 追求类内紧凑；
但几何变换需要类内多模式覆盖。
```

---

## 7. 为什么 C1 系列对 jitter 失效

### 7.1 jitter 破坏局部几何细节

jitter 会对点坐标加入抖动，导致局部几何关系变得不稳定。

Point-Cache 的 local cache 依赖 patch centers。

如果输入点的位置发生抖动，那么 patch centers 的稳定性也会下降。

### 7.2 C1 改的是缓存样本，不是查询样本

C1 系列改进的是：

```text
哪些样本可以进入 GPA-Cache；
哪些样本的 local patch centers 可以进入 GPA-controlled local cache。
```

但它没有改变：

```text
当前 jitter 查询样本的 patch centers 是否稳定；
local affinity 是否应该自适应降权；
局部特征是否需要做去噪或平滑。
```

因此，即使 local cache 中的样本更中心、更干净，jitter 后的查询样本 local features 仍然可能不稳定，导致 local cache 检索失效。

### 7.3 local cache 可能放大 jitter 下的错误相似度

由于 jitter 扰动直接影响局部 patch features，local cache 的相似度计算可能变得不可靠。

如果仍然用固定 alpha/beta 将 local cache logits 加到最终预测中，就可能放大错误匹配。

因此，jitter 下的问题不是简单换一批更中心的 local cache 就能解决，而是需要考虑：

```text
query 端局部特征可靠性；
local cache 权重自适应；
或局部结构鲁棒化。
```

---

## 8. 为什么 Ua1 比 Ua2 更好

### 8.1 Ua2 的规则

E3-V3-C1-Ua2 使用：

```text
低熵门控 + 替换离中心最远样本
```

即新样本需要满足：

```text
curr_entropy < highest_entropy_in_gpa_cache
and
curr_distance_to_center < farthest_distance_to_center
```

如果满足，则替换当前 GPA-Cache 中离中心最远的样本。

### 8.2 Ua1 的规则

E3-V3-C1-Ua1 使用：

```text
低熵门控 + 替换最高熵样本
```

即新样本需要满足：

```text
curr_entropy < highest_entropy_in_gpa_cache
and
curr_distance_to_center < distance(highest_entropy_item, center)
```

如果满足，则替换当前 GPA-Cache 中最高熵样本。

### 8.3 最远样本不一定是坏样本

Ua2 看起来更符合“提高类内紧凑性”的目标，但实验结果比 Ua1 差。

原因可能是：

```text
离中心最远的样本不一定是坏样本；
它可能代表该类别的另一个有效结构模式。
```

例如：

```text
另一种姿态；
另一种尺度；
另一种局部结构；
另一种 corruption 形态；
类内边缘但仍然正确的样本。
```

Ua2 删除最远样本，可能会删掉有用的类内多样性。

这会让 GPA-local-cache 更窄，进一步加剧 dropout、rotate、scale、jitter 上的覆盖不足。

### 8.4 最高熵样本更可能是不可靠样本

相比之下，Ua1 删除最高熵样本。

最高熵样本代表模型自己也不确定，通常更可能是不稳定或污染项。

因此，Ua1 更像是：

```text
保留低熵可靠性；
用距离约束辅助判断；
但不主动删除类内边缘模式。
```

这比 Ua2 更保守，也更稳定。

这解释了为什么：

```text
E3-V3-C1-Ua1: 54.02
E3-V3-C1-Ua2: 53.68
```

---

## 9. 当前设计的底层问题总结

C1 系列结果说明，当前方法的底层机制是：

```text
通过候选池单中心筛选，保留离类别中心最近的样本；
通过 GPA-controlled local cache，把这些中心样本的局部特征用于后续检索。
```

这个机制适合：

```text
外点噪声污染；
主体结构仍然存在；
坏样本会被噪声拉离中心；
离中心近大概率意味着更干净。
```

因此它对 add_global / add_local 有明显帮助。

但这个机制不适合：

```text
结构缺失；
整体几何变换；
局部扰动；
类内多模式分布。
```

因为这些问题需要的不是单中心紧凑，而是：

```text
多模式覆盖；
结构多样性；
query 端局部特征可靠性判断；
local cache 权重自适应。
```

因此，C1 系列的核心局限是：

```text
它提高了 GPA-Cache 的中心紧凑性；
但牺牲了 local cache 的覆盖度和多样性。
```

---

## 10. 当前研究判断

当前可以形成以下结论：

```text
1. 候选池初始化方向没有被否定；
2. 低熵门控仍然必要；
3. 替换最高熵样本比替换最远样本更稳；
4. E3-V3-C1-Ua1 是当前 C1 系列最优版本；
5. C1 系列的收益主要来自 add_global / add_local；
6. 当前单中心 GPA 方法更像是去噪策略，而不是通用几何鲁棒策略；
7. 对几何结构变化，需要考虑 Entropy Cache 中心锚点、多中心原型或 diversity-aware 选择。
```

---

## 11. 后续方向：为什么要看 C2-Ua1

C2-Ua1 的设计是：

```text
C2：
初始化临时中心同时使用 2K candidate pool 和 Entropy Cache；

Ua1：
GPA-Cache 满后使用低熵门控 + 替换最高熵样本。
```

它的目的不是简单追求更紧凑，而是验证：

```text
Entropy Cache 能否为 candidate pool 的中心构造提供更稳定的锚点；
在保留 Ua1 稳定更新规则的同时，改善 C1 单中心候选池中心不稳的问题。
```

如果 C2-Ua1 比 C1-Ua1 好，说明：

```text
candidate-only center 可能不够稳；
Entropy Cache 作为中心锚点是有价值的。
```

如果 C2-Ua1 仍然只在 add_global / add_local 上提升，而在几何结构变化上仍然下降，则说明：

```text
问题不是中心锚点是否稳定；
而是单中心 GPA 本身无法覆盖几何多模式。
```

此时后续应转向：

```text
多中心 GPA-Cache；
diversity-aware candidate selection；
local cache 权重自适应；
或 corruption-aware local reliability 判断。
```

---

## 12. 简短结论

E3-V3-C1 系列实验说明：候选池单中心 GPA-Cache 对 add_global 和 add_local 这类外点噪声具有明显正收益，因为它能够筛掉被外点拉离类别中心的样本，使进入 GPA-local-cache 的局部特征更干净。

但对于 dropout、rotate、scale、jitter 等几何结构变化，该方法失效或下降，因为这些 corruption 改变的是结构完整性、几何模式或局部特征稳定性，单中心紧凑性筛选会牺牲类内多样性和 local cache 覆盖度。

当前最优 C1 版本为 E3-V3-C1-Ua1，说明低熵门控和替换最高熵样本仍然比纯距离或替换最远样本更稳。

下一步应查看 C2-Ua1，判断 Entropy Cache 作为中心锚点是否能改善 candidate-only center 的不稳定问题。
