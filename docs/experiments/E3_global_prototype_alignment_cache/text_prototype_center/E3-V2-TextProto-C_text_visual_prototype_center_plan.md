# E3-V2-TextProto-C：引入 Text Prototype 的文本-视觉联合原型中心实验说明

## 1. 实验命名调整

本实验原本临时记为：

    E3-V2-C-T
    或
    E3-V2-C-T1

但是这个名字没有充分体现“引入 Text Prototype”这一核心改动。

由于本实验最重要的新变量是：

    在 E3-V2-C 的视觉 GPA 原型中心中，引入类别 Text Prototype 作为语义锚点。

因此，建议正式命名为：

    E3-V2-TextProto-C

其中：

    E3：
        表示该实验仍属于 E3 的 Global Prototype-Alignment Cache 主线。

    V2：
        表示沿用 E3-V2 的并列式 GPA-Cache 框架。

    TextProto：
        明确表示引入 Text Prototype。

    C：
        表示沿用 E3-V2-C 的中心构造与替换思想，即使用联合中心进行 GPA-Cache 更新判断。

第一版可进一步记为：

    E3-V2-TextProto-C1

其中 C1 表示第一版文本-视觉联合中心权重设置。

如果后续做权重消融，可以命名为：

    E3-V2-TextProto-C1-w0.9v0.1t
    E3-V2-TextProto-C1-w0.8v0.2t
    E3-V2-TextProto-C1-w0.7v0.3t
    E3-V2-TextProto-C1-w0.5v0.5t

其中：

    v 表示 visual prototype 权重；
    t 表示 text prototype 权重。

---

## 2. 实验背景

当前 E3 阶段已经完成了围绕 GPA-Cache 的多组实验。

E3 阶段目前最好的版本是：

    E3-V2-C：
        并列式 GPA-Cache
        + Entropy Cache 与 GPA-Cache 联合视觉中心
        + 低熵门控
        + 替换 GPA-Cache 中最高熵样本

E3-V2-C 的平均准确率为：

    54.04

E3-V3 系列尝试解决 GPA-Cache 初始化问题，包括：

    Entropy-bootstrap 初始化；
    Candidate pool 初始化；
    Candidate-only center；
    Candidate+Entropy center；
    无熵距离更新；
    低熵门控 + 替换最远样本；
    低熵门控 + 替换最高熵样本。

但是，E3-V3 系列没有超过 E3-V2-C：

    E3-V3-C1-Ua1：54.02
    E3-V3-C2-Ua1：54.00

这说明，当前主要瓶颈可能不在于 candidate pool 初始化，而在于原型中心本身的表达方式。

此前 E4-A 尝试引入类别概率分布，但 E4-A 当前版本更差。主要原因是类别分布由很少量、早期进入 GPA-Cache 的伪标签样本建立，方差估计不稳定，分布裁判反而放大了早期 cache 偏差。

因此，当前先回到 E3-V2-C 的稳定框架，在原有视觉中心的基础上，引入 Text Prototype 作为语义锚点，构造文本-视觉联合原型中心。

---

## 3. 本实验的核心变化

E3-V2-C 的原始中心只来自视觉测试样本：

    visual_center_c = mean(EntropyCache[c] ∪ GPACache[c])

其中：

    EntropyCache[c]：
        原始 Global Entropy Cache 中类别 c 的低熵全局特征。

    GPACache[c]：
        GPA-Cache 中类别 c 的全局特征。

E3-V2-TextProto-C 在此基础上引入类别文本原型：

    text_center_c = Text Prototype[c]

最终中心为：

    text_visual_center_c = normalize(
        w_visual * visual_center_c + w_text * text_center_c
    )

也就是说：

    视觉原型负责适应当前测试流；
    文本原型负责提供稳定类别语义锚点；
    最终 GPA 原型中心由视觉原型和 Text Prototype 共同组成。

---

## 4. Text Prototype 来自哪里

当前 Point-Cache / ULIP 推理中已经存在类别文本特征。

在代码里，分类时使用的文本类别特征通常表现为：

    clip_weights

每个类别 c 对应一个文本特征向量。

这个文本特征由 prompt 经过文本编码器得到。

第一版 E3-V2-TextProto-C 建议使用：

    prompt_source = manual_full

也就是说，先不改变文本 prompt 来源，只使用现有 E2 / E3 实验中已经验证过的 manual_full 文本原型。

这样做的原因是：

    1. 控制变量；
    2. 不同时引入 LLM prompt 变化；
    3. 只验证“Text Prototype 参与 GPA 中心构造”是否有效；
    4. 便于直接与 E3-V2-C 比较。

后续如果该方向有效，再考虑：

    manual_full + LLM 文本描述；
    E1/E2 中已经验证过的文本增强版本；
    不同 Text Prototype 权重；
    Text Prototype 质量对 GPA 中心的影响。

---

## 5. GPA-Cache 初始化规则

E3-V2-TextProto-C 第一版完全沿用 E3-V2-C 的初始化规则。

对于预测类别 c：

    如果 GPA-Cache[c] 未满：
        新样本直接进入 GPA-Cache[c]；
        该样本对应的 local patch features 进入 GPA-local-cache[c]。

也就是说：

    不使用 candidate pool；
    不改变前 K 个样本直接进入的初始化逻辑；
    不改变 local cache 同步逻辑。

这样做的原因是：

    E3-V2-C 是当前最优版本；
    E3-V3 的 candidate pool 初始化没有超过 E3-V2-C；
    当前实验只验证 Text Prototype 中心是否有帮助；
    不希望同时改变初始化机制，导致结果难以解释。

---

## 6. GPA-Cache 满后替换规则

E3-V2-TextProto-C 第一版也沿用 E3-V2-C 的替换对象：

    替换 GPA-Cache[c] 中最高熵样本。

具体规则如下。

对于新样本 x_new，zero-shot 预测类别为 c。

如果 GPA-Cache[c] 已满：

    1. 找到 GPA-Cache[c] 中最高熵样本 x_high；

    2. 构造视觉中心：
           visual_center_c = mean(EntropyCache[c] ∪ GPACache[c])

    3. 取得文本中心：
           text_center_c = Text Prototype[c]

    4. 构造文本-视觉联合中心：
           center_c = normalize(
               w_visual * visual_center_c + w_text * text_center_c
           )

    5. 计算两个距离：
           d_new  = distance(x_new, center_c)
           d_high = distance(x_high, center_c)

    6. 如果：
           entropy(x_new) < entropy(x_high)
           and
           d_new < d_high

       则：
           用 x_new 替换 x_high；
           同步替换 GPA-local-cache[c]；
           替换后重新计算 visual_center_c；
           后续判断继续使用更新后的文本-视觉联合中心。

否则：

    不更新 GPA-Cache；
    不更新 GPA-local-cache。

因此，该方法的替换规则可以概括为：

    低熵门控
    + 更靠近文本-视觉联合中心
    + 替换最高熵样本

---

## 7. 与 E3-V2-C 的唯一核心区别

E3-V2-C：

    center_c = visual_center_c
             = mean(EntropyCache[c] ∪ GPACache[c])

    替换条件：
        新样本低熵
        且新样本比最高熵样本更靠近视觉中心

E3-V2-TextProto-C：

    center_c = normalize(
        w_visual * visual_center_c + w_text * text_center_c
    )

    替换条件：
        新样本低熵
        且新样本比最高熵样本更靠近文本-视觉联合中心

除此之外，其他部分暂时不改：

    GPA-Cache 初始化不改；
    替换最高熵样本不改；
    低熵门控不改；
    local cache 同步不改；
    最终预测公式不改；
    prompt_source 第一版不改。

---

## 8. 0.7 visual + 0.3 text 的根据是什么

需要明确记录：

    0.7 visual + 0.3 text 不是 Point-Cache 原文、MCP 原文或 BayesMM 原文中给出的固定权重。

它不是原文超参数，也不是已有论文直接规定的数值。

这是一个用于第一版实验的保守启发式初值。

选择该权重的理由如下。

### 8.1 视觉中心是当前已经验证过的主干

E3-V2-C 的视觉联合中心已经取得当前 E3 阶段最好的结果：

    E3-V2-C：54.04

因此，第一版不应该让 Text Prototype 占主导。

如果文本权重过大，例如 0.5 或更高，可能会把中心过度拉向静态文本语义空间，削弱视觉 cache 对当前测试分布的适应能力。

所以第一版应该让视觉中心占主要权重。

### 8.2 Text Prototype 的作用是语义锚点，而不是替代视觉中心

Text Prototype 的作用是：

    防止视觉中心被噪声样本或异常测试流带偏；
    给视觉中心提供一个稳定的类别语义方向；
    在视觉 cache 不稳定时起到轻量约束。

它不是用来完全替代视觉原型。

因此，文本权重应该足够小，避免主导中心；但也不能太小，否则几乎没有影响。

### 8.3 0.3 是一个能产生可观察影响但不至于过强的起点

如果使用：

    0.9 visual + 0.1 text

Text Prototype 影响可能太弱，不容易观察到区别。

如果使用：

    0.5 visual + 0.5 text

Text Prototype 可能过强，尤其在文本原型与测试视觉分布存在 domain gap 时，可能拉偏中心。

因此，第一版选择：

    0.7 visual + 0.3 text

其含义是：

    视觉中心仍然主导；
    Text Prototype 提供较明显但不压倒视觉中心的语义锚点。

这只是第一版探索权重，不应在论文中写成固定理论值。

### 8.4 这个权重需要后续消融

如果第一版有正向迹象，必须进一步做权重消融。

建议消融矩阵：

    E3-V2-TextProto-C-w0.9v0.1t：
        visual 0.9 + text 0.1

    E3-V2-TextProto-C-w0.8v0.2t：
        visual 0.8 + text 0.2

    E3-V2-TextProto-C-w0.7v0.3t：
        visual 0.7 + text 0.3

    E3-V2-TextProto-C-w0.5v0.5t：
        visual 0.5 + text 0.5

如果 0.9 / 0.1 或 0.8 / 0.2 更好，说明 Text Prototype 只能轻量约束。

如果 0.7 / 0.3 最好，说明文本语义锚点需要较明显参与。

如果 0.5 / 0.5 下降，说明 Text Prototype 过强会干扰当前视觉 cache 中心。

因此，0.7 / 0.3 的定位是：

    第一版启发式探索点；
    不是原文结论；
    必须通过消融实验确认。

---

## 9. 为什么 Text Prototype 可能有帮助

E3 的单中心视觉原型方法存在一个问题：

    中心完全来自测试流视觉样本。

如果测试流中存在噪声、伪标签错误、几何变化或局部分布偏移，视觉中心可能被带偏。

Text Prototype 提供的是一个固定语义锚点。

它可能带来以下作用：

    1. 抑制视觉中心漂移；
    2. 增强类别语义一致性；
    3. 减少错误视觉样本对中心的影响；
    4. 在视觉原型不稳定时提供稳定方向；
    5. 使 GPA-Cache 的替换规则不完全依赖测试流视觉统计。

因此，该方法本质上不是“用文本替代视觉”，而是：

    用 Text Prototype 约束视觉原型中心。

---

## 10. 可能风险

该方法也有风险。

### 10.1 Text Prototype 和视觉分布可能存在偏差

Text Prototype 来自 prompt 编码，视觉中心来自测试点云特征。

虽然二者在同一多模态空间中，但它们的分布可能并不完全一致。

如果文本权重太大，可能会把中心拉离当前视觉测试分布。

### 10.2 Text Prototype 已经参与 zero-shot logits

最终预测中原本就有 zero-shot logits，而 zero-shot logits 已经使用 Text Prototype。

如果 GPA 中心也使用 Text Prototype，可能存在一定程度的重复利用文本信息。

但需要注意：

    Text Prototype 在这里不是直接参与最终 logits；
    它只参与 GPA-Cache 的样本更新判断；
    作用是控制哪些样本进入 GPA-local-cache。

因此，这种重复使用是可以接受的，但需要通过实验验证是否有副作用。

### 10.3 prompt 质量会影响 Text Prototype

如果 manual_full 文本原型不够好，Text Prototype 可能不是可靠锚点。

如果第一版没有收益，后续可以考虑：

    使用 E1/E2 验证过的 LLM 文本增强原型；
    使用 manual + LLM 混合文本原型；
    或降低文本权重。

---

## 11. 预期结果与分析方式

主要比较对象：

    E2 baseline：54.00
    E3-V2-C：54.04
    E3-V3-C1-Ua1：54.02
    E3-V3-C2-Ua1：54.00

重点观察：

    add_global
    add_local
    dropout_global
    dropout_local
    rotate
    scale
    jitter

如果 E3-V2-TextProto-C 在 dropout、rotate、scale、jitter 上超过 E3-V2-C，说明 Text Prototype 语义锚点对几何变化有帮助。

如果只在 add_global / add_local 上提升，说明该方法仍然主要起到去噪作用，没有解决几何结构变化问题。

如果整体下降，可能说明：

    文本权重过大；
    Text Prototype 与视觉分布存在偏差；
    直接融合向量不如融合距离稳定；
    manual_full Text Prototype 不够强；
    GPA 中心中重复使用文本信息带来副作用。

---

## 12. 第一版实验定义

建议第一版实验为：

    E3-V2-TextProto-C-w0.7v0.3t

设置：

    visual_center = mean(EntropyCache[c] ∪ GPACache[c])

    text_center = Text Prototype[c]

    final_center = normalize(
        0.7 * visual_center + 0.3 * text_center
    )

    GPA-Cache 初始化：
        沿用 E3-V2-C，未满直接进入。

    GPA-Cache 满后：
        找到最高熵样本；
        如果新样本熵更低，且更靠近 final_center；
        则替换最高熵样本。

    local cache：
        与 GPA-Cache 同步替换。

    最终预测公式：
        暂时不改。

    prompt_source：
        manual_full。

---

## 13. 后续扩展

如果 E3-V2-TextProto-C-w0.7v0.3t 有正收益，可以继续做：

    1. 文本权重消融：
        0.9 visual + 0.1 text
        0.8 visual + 0.2 text
        0.7 visual + 0.3 text
        0.5 visual + 0.5 text

    2. Text Prototype 来源消融：
        manual_full
        manual + LLM
        LLM-only

    3. 融合方式消融：
        向量融合：
            center = normalize(w_v * visual + w_t * text)

        距离融合：
            distance = w_v * distance(x, visual)
                     + w_t * distance(x, text)

    4. 与 E4 类别分布方法结合：
        Text Prototype 作为分布先验或语义锚点。

---

## 14. 简短结论

E3-V2-TextProto-C 是在当前最优 E3-V2-C 框架上引入 Text Prototype 语义锚点的实验。它不改变 GPA-Cache 初始化、不改变替换最高熵样本、不改变最终预测公式，只把原来的纯视觉联合中心升级为文本-视觉联合中心。

0.7 visual + 0.3 text 不是原文固定权重，而是第一版保守启发式初值。其依据是：E3-V2-C 的视觉中心已经验证有效，因此应保持视觉主导；Text Prototype 只作为语义锚点参与，权重不宜过大。该权重需要后续通过 0.9/0.1、0.8/0.2、0.7/0.3、0.5/0.5 等消融实验验证。

该实验的目标是判断：Text Prototype 能否抑制纯视觉 GPA 中心漂移，从而改善 cache 更新质量，尤其是在 dropout、rotate、scale、jitter 等几何结构变化场景下是否优于 E3-V2-C。
