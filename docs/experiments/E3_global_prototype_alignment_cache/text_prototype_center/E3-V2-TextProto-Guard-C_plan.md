# E3-V2-TextProto-Guard-C 实验说明：视觉去噪与文本语义保护的双分支 GPA-Cache 更新

## 1. 当前阶段背景

E3_global_prototype_alignment_cache 已经完成了多组实验，包括：

    E3-V1：
        顺序式 GPA-Cache。

    E3-V2：
        并列式 GPA-Cache。

    E3-V3：
        围绕 GPA-Cache 初始化问题进行候选池、熵启动、中心来源等消融。

    E4：
        尝试引入类别概率分布，但当前 E4-A 由于小样本分布不稳定、历史累计分布与当前 cache 不一致等问题，平均效果下降。

    E3-V2-TextProto-C：
        回到 E3-V2-C，在视觉中心中引入 Text Prototype，构造文本-视觉联合中心。

目前最稳定的 E3 基线仍然是：

    E3-V2-C：
        并列式 GPA-Cache
        + Entropy Cache 与 GPA-Cache 联合视觉中心
        + 低熵门控
        + 替换 GPA-Cache 中最高熵样本

平均准确率：

    E3-V2-C：54.04

Text Prototype 向量融合实验已经完成以下权重：

    E3-V2-TextProto-C-w0.9v0.1t
    E3-V2-TextProto-C-w0.8v0.2t
    E3-V2-TextProto-C-w0.7v0.3t

其中修复 gpa_cache 清空问题后，w0.9v0.1t 的平均准确率为：

    53.98

它非常接近 E3-V2-C，但仍未超过 E3-V2-C。

---

## 2. 已发现的关键问题

### 2.1 不能再简单复制旧 runner 文件

此前多次通过复制旧文件再字符串替换生成新实验代码，导致旧问题被重新带入新实验。

已经出现过的问题包括：

    1. 预构建后的 gpa_cache 被清空；
    2. GPA-Cache 与 GPA-local-cache 脱节；
    3. build 阶段和 test 阶段函数签名不一致；
    4. helper 函数被调用但没有定义；
    5. 新增函数内部递归误调用自己；
    6. wrapper 放到更深目录后 sys.path 不正确；
    7. 修复过的问题在新实验中重复出现。

因此，从 E3-V2-TextProto-Guard-C 开始，代码实现原则是：

    不直接 cp 旧实验文件再批量 replace；
    尽量新写干净 runner；
    如果必须参考旧逻辑，只参考算法结构；
    所有关键函数、缓存更新、build/test 调用都显式写清楚；
    每次都要检查 gpa_cache 和 gpa_local_cache 是否一致；
    每次都要检查 build 后 gpa_cache 是否被保留；
    文档、代码、执行命令分开给出。

### 2.2 Text Prototype 向量融合的收益与损失抵消

E3-V2-TextProto-C 的向量融合方式为：

    final_center = normalize(
        w_visual * visual_center + w_text * text_center
    )

其中：

    visual_center = mean(EntropyCache[c] ∪ GPACache[c])
    text_center   = Text Prototype[c]

实验结果说明：

    Text Prototype 对 dropout_global、dropout_local 等结构缺失类损坏有帮助；
    但直接向量融合会拉动视觉中心，伤害 add_local、rotate、scale、jitter 等依赖视觉去噪或当前视觉分布适应的损坏类型。

修复后 w0.9v0.1t 的表现为：

    add_global：
        优于 E3-V2-C。

    dropout_global：
        略优于 E3-V2-C。

    dropout_local：
        明显优于 E3-V2-C。

    add_local、rotate、scale、jitter：
        低于 E3-V2-C。

因此，当前结论不是 Text Prototype 无效，而是：

    Text Prototype 不适合直接以固定比例融合进视觉中心；
    它更适合作为替换判断时的辅助语义保护条件。

---

## 3. 新实验命名

本实验建议正式命名为：

    E3-V2-TextProto-Guard-C

含义：

    E3：
        仍属于 Global Prototype-Alignment Cache 主线。

    V2：
        沿用 E3-V2 的并列式 GPA-Cache 框架。

    TextProto：
        明确表示引入 Text Prototype。

    Guard：
        表示 Text Prototype 不再直接融合进中心，而是作为语义保护条件参与更新判断。

    C：
        表示仍然继承 E3-V2-C 的 Entropy/GPA 联合视觉中心思想。

第一版建议命名为：

    E3-V2-TextProto-Guard-C-rho0.05

其中：

    rho0.05 表示视觉距离允许 5% 的轻微退化，用来给 Text Prototype 语义保护分支留出空间。

后续如果做消融，可以命名为：

    E3-V2-TextProto-Guard-C-rho0.02
    E3-V2-TextProto-Guard-C-rho0.05
    E3-V2-TextProto-Guard-C-rho0.10

---

## 4. 文件路径与脚本命名规范

### 4.1 文档路径

所有 E3 实验说明文档放在：

    docs/experiments/E3_global_prototype_alignment_cache/

本实验属于 Text Prototype 中心方向，因此文档放在：

    docs/experiments/E3_global_prototype_alignment_cache/text_prototype_center/

当前说明文档为：

    docs/experiments/E3_global_prototype_alignment_cache/text_prototype_center/E3-V2-TextProto-Guard-C_plan.md

### 4.2 runner 路径

当前 E3 的 runner 主要放在：

    Point-Cache/runners/E3_global_prototype_alignment_cache/

Text Prototype 相关 runner 放在子目录：

    Point-Cache/runners/E3_global_prototype_alignment_cache/text_prototype_center/

Guard 实验建议后续代码文件命名为：

    model_with_hierarchical_caches_textproto_guard_center.py
    run_e3_ulip_modelnetc_s2_textproto_guard_center.py

注意：

    后续实现时不要直接复制旧 runner 后批量替换；
    应该新写一个干净版本，显式实现 Guard 逻辑。

### 4.3 script 路径

Text Prototype 相关脚本放在：

    Point-Cache/scripts/E3_global_prototype_alignment_cache/text_prototype_center/

已有向量融合脚本编号为：

    00_1：w0.7v0.3t
    00_2：w0.9v0.1t
    00_3：w0.8v0.2t

因此 Guard 实验建议使用新的编号段：

    01_1_ulip_modelnetc_s2_zs_global_local_e3_v2_textproto_guard_c_rho0.05_manual_full.sh

如果后续做 rho 消融：

    01_1：rho0.05
    01_2：rho0.02
    01_3：rho0.10

公共脚本建议命名为：

    01_run_ulip_modelnetc_s2_textproto_guard_common.sh

### 4.4 result 路径

结果仍然放在：

    Point-Cache/results/E3_global_prototype_alignment_cache/

第一版结果目录建议为：

    Point-Cache/results/E3_global_prototype_alignment_cache/01_1_ulip_modelnetc_s2_zs_global_local_e3_v2_textproto_guard_c_rho0.05_manual_full/

---

## 5. 方法核心思想

E3-V2-TextProto-Guard-C 的核心思想是：

    不再把 Text Prototype 和 visual prototype 强行融合成一个中心；
    而是保留两个独立判断：

        visual_center 负责视觉去噪；
        text_center 负责语义保护。

因此，Text Prototype 的角色从：

    参与构造中心

改变为：

    参与替换判断时的语义保护条件。

这样可以避免 Text Prototype 直接拉偏视觉中心，同时仍然让它在结构缺失场景中发挥作用。

---

## 6. 原型定义

对于类别 c：

### 6.1 视觉原型中心

    visual_center_c = mean(EntropyCache[c] ∪ GPACache[c])

其中：

    EntropyCache[c]：
        Global Entropy Cache 中类别 c 的低熵全局特征。

    GPACache[c]：
        GPA-Cache 中类别 c 的全局特征。

该中心来自当前测试流，具有在线适应能力。

它负责：

    视觉去噪；
    保留当前测试分布适应性；
    维护 E3-V2-C 已经验证有效的视觉中心逻辑。

### 6.2 文本原型中心

    text_center_c = Text Prototype[c]

当前代码中通常对应：

    text_center_c = clip_weights[:, c]

需要注意：

    如果上游 clip_weights 已经由多个 prompt 文本特征平均得到，那么 clip_weights[:, c] 就是类别 c 的 Text Prototype Center；
    如果上游没有平均，则后续需要显式检查并修正文本原型构造方式。

Text Prototype 是固定的，不随测试流变化。

它负责：

    提供类别语义锚点；
    抑制视觉中心被结构缺失样本带偏；
    在 dropout、scale 等视觉结构不完整场景中提供辅助判断。

### 6.3 不再构造融合中心

本实验不使用：

    final_center = normalize(
        w_visual * visual_center + w_text * text_center
    )

也就是说，Guard 方法没有一个被文本拉动后的共同中心。

它保留两个中心：

    visual_center_c
    text_center_c

并分别计算距离。

---

## 7. GPA-Cache 初始化规则

E3-V2-TextProto-Guard-C 第一版完全沿用 E3-V2-C 的初始化方式。

对于预测类别 c：

    如果 GPA-Cache[c] 未满：
        新样本直接进入 GPA-Cache[c]；
        对应 local patch features 进入 GPA-local-cache[c]。

初始化阶段不使用 Text Prototype 约束。

原因是：

    E3-V2-C 是当前最稳的 E3 版本；
    E3-V3 候选池初始化没有超过 E3-V2-C；
    当前实验只想验证 Text Prototype Guard 是否能改善替换阶段；
    不希望同时改变初始化机制导致结果难以解释。

---

## 8. GPA-Cache 满后的替换对象

替换对象仍然沿用 E3-V2-C：

    x_high = GPA-Cache[c] 中最高熵样本

也就是说，优先替换当前缓存中模型最不确定的样本。

原因是：

    E3-V3-C1-Ua1 说明替换最高熵样本比替换最远样本更稳；
    最远样本不一定是坏样本，它可能只是类别中的正常几何变化；
    最高熵样本更可能是不可靠样本。

---

## 9. 距离定义

对于新样本 x_new 和最高熵样本 x_high，分别计算：

    d_visual_new  = distance(x_new,  visual_center_c)
    d_visual_high = distance(x_high, visual_center_c)

    d_text_new    = distance(x_new,  text_center_c)
    d_text_high   = distance(x_high, text_center_c)

距离沿用当前 E3 中的 cosine distance 形式。

---

## 10. 替换规则：双分支 Guard

### 10.1 总门控：低熵

首先必须满足：

    entropy(x_new) < entropy(x_high)

即新样本必须比当前最高熵缓存样本更可靠。

如果不满足该条件，直接拒绝。

### 10.2 分支 A：视觉去噪分支

分支 A 完全保留 E3-V2-C 原始更新能力：

    d_visual_new < d_visual_high

如果新样本在视觉中心上更好，则直接允许替换，不强制要求它也更靠近 Text Prototype。

完整条件：

    entropy(x_new) < entropy(x_high)
    and
    d_visual_new < d_visual_high

这个分支负责：

    add_global；
    add_local；
    jitter；
    以及其他依赖当前视觉分布去噪的情况。

### 10.3 分支 B：文本语义保护分支

对于结构缺失或尺度变化，样本在视觉中心上可能没有明显更近，甚至略微更远。

此时允许一个文本保护通道：

    d_visual_new <= d_visual_high * (1 + rho_visual)
    and
    d_text_new < d_text_high

其中：

    rho_visual 是视觉距离退化容忍比例。

第一版设：

    rho_visual = 0.05

含义：

    新样本的视觉距离最多允许比旧样本差 5%；
    但前提是它必须更靠近 Text Prototype。

完整条件：

    entropy(x_new) < entropy(x_high)
    and
    d_visual_new <= d_visual_high * 1.05
    and
    d_text_new < d_text_high

这个分支负责：

    dropout_global；
    dropout_local；
    scale；
    以及其他结构缺失或视觉中心判断不够友好的情况。

### 10.4 最终替换条件

最终替换规则为：

    replace =
        entropy(x_new) < entropy(x_high)
        and
        (
            d_visual_new < d_visual_high
            or
            (
                d_visual_new <= d_visual_high * (1 + rho_visual)
                and
                d_text_new < d_text_high
            )
        )

如果 replace 为 True：

    用 x_new 替换 x_high；
    同步替换 GPA-local-cache[c]；
    更新后后续 visual_center_c 由新的 EntropyCache[c] ∪ GPACache[c] 重新计算。

否则：

    不更新 GPA-Cache；
    不更新 GPA-local-cache。

---

## 11. 该方法为什么可能同时处理结构缺失与视觉去噪

### 11.1 保留视觉去噪能力

E3-V2-C 的视觉中心来自当前测试流，因此对 add_global、add_local、jitter 这类视觉扰动更敏感。

Guard 方法保留分支 A：

    d_visual_new < d_visual_high

这意味着只要新样本视觉上更干净，就可以替换，不会因为 Text Prototype 距离不够好而被拒绝。

因此它保留 E3-V2-C 的视觉去噪能力。

### 11.2 引入结构缺失保护

Text Prototype 在之前向量融合实验中已经显示出对 dropout_global、dropout_local 有一定正向作用。

Guard 方法不再让 Text Prototype 直接拉偏视觉中心，而是通过分支 B 提供一个语义保护通道：

    视觉不明显变差
    +
    文本更一致

这样，结构缺失样本如果在视觉中心上没有明显优势，但在文本语义上更符合类别，也有机会替换掉最高熵样本。

### 11.3 避免直接向量融合的副作用

向量融合会直接改变中心：

    final_center = normalize(w_visual * visual + w_text * text)

这会伤害当前视觉中心对测试流的适应性。

Guard 方法不改变 visual_center，只把 Text Prototype 作为附加判断，因此更安全。

---

## 12. 与已有实验的关系

### 12.1 与 E3-V2-C 的关系

E3-V2-C：

    低熵
    +
    视觉距离更近
    +
    替换最高熵样本

E3-V2-TextProto-Guard-C：

    低熵
    +
    (
        视觉距离更近
        or
        视觉距离不明显变差且文本距离更近
    )
    +
    替换最高熵样本

因此，Guard 方法是在 E3-V2-C 的基础上增加一个文本保护分支，而不是替换 E3-V2-C 的视觉规则。

### 12.2 与 E3-V2-TextProto-C 向量融合的关系

E3-V2-TextProto-C 向量融合：

    直接构造文本-视觉融合中心。

E3-V2-TextProto-Guard-C：

    不构造融合中心；
    视觉中心和文本中心分开计算距离；
    Text Prototype 只作为辅助约束。

因此，Guard 方法针对的是向量融合“拉偏视觉中心”的问题。

### 12.3 与 E4-A 的关系

E4-A 尝试用类别分布替代中心距离，但小样本在线分布不稳定。

Guard 方法不再估计类别方差或分布，只使用固定 Text Prototype 作为语义保护，因此比 E4-A 更简单、更稳。

---

## 13. 第一版实验设置

建议第一版：

    E3-V2-TextProto-Guard-C-rho0.05

设置：

    visual_center:
        mean(EntropyCache[c] ∪ GPACache[c])

    text_center:
        Text Prototype[c]

    rho_visual:
        0.05

    GPA-Cache 初始化:
        沿用 E3-V2-C，未满直接进入。

    GPA-Cache 满后:
        替换最高熵样本。

    替换规则:
        低熵
        +
        (
            视觉距离更近
            or
            视觉距离最多变差 5% 且文本距离更近
        )

    local cache:
        与 GPA-Cache 同步替换。

    最终预测公式:
        暂时不改。

    prompt_source:
        manual_full。

---

## 14. 需要记录的统计信息

为了判断 Guard 机制是否真正工作，需要记录以下事件：

    gpa_replace_visual_branch
    gpa_replace_text_guard_branch
    gpa_reject_entropy
    gpa_reject_visual_and_text_guard

同时记录每次替换或拒绝时的距离：

    d_visual_new
    d_visual_high
    d_text_new
    d_text_high
    visual_ratio = d_visual_new / d_visual_high
    text_margin = d_text_high - d_text_new

这样可以分析：

    add_global 是否主要通过 visual branch 替换；
    dropout_global 是否更多通过 text guard branch 替换；
    text guard branch 是否真的贡献了结构缺失场景下的改进；
    text guard branch 是否过宽，导致视觉去噪被破坏。

---

## 15. 预期结果

如果该方法有效，预期表现为：

    1. 平均准确率超过 E3-V2-C 或至少不低于 E3-V2-C；
    2. add_global、add_local、jitter 不应明显低于 E3-V2-C；
    3. dropout_global、dropout_local 保留 Text Prototype 带来的正收益；
    4. scale 不应明显下降；
    5. 替换事件中，visual branch 和 text guard branch 都应有合理贡献。

如果结果不好，可能原因包括：

    1. rho_visual 太大，Text Guard 过宽，伤害视觉去噪；
    2. rho_visual 太小，Text Guard 几乎不起作用；
    3. text_center 与视觉空间仍存在偏差；
    4. clip_weights 不是预期的类别文本平均原型；
    5. Text Prototype 对当前 corruption 的帮助有限。

---

## 16. 后续消融

如果 rho0.05 有正向迹象，可以继续做：

    rho0.02：
        更保守的 Text Guard。

    rho0.10：
        更宽松的 Text Guard。

也可以继续做：

    距离融合版本：
        d_final = w_visual * d_visual + w_text * d_text

    Text Prototype 来源消融：
        manual_full
        manual + LLM
        LLM-only

    分支消融：
        只保留 visual branch；
        只保留 text guard branch；
        visual branch + text guard branch。

---

## 17. 简短结论

E3-V2-TextProto-Guard-C 的核心是：

    不让 Text Prototype 直接改变视觉中心；
    而是保留 E3-V2-C 的视觉去噪分支；
    同时增加一个“视觉不明显变差 + 文本更一致”的语义保护分支。

它试图同时解决两个问题：

    视觉中心负责 add_global、add_local、jitter 等视觉去噪；
    Text Prototype 负责 dropout、scale 等结构缺失或语义稳定性问题。

第一版建议做：

    E3-V2-TextProto-Guard-C-rho0.05

这是当前从 E4-A 失败和 TextProto 向量融合实验中得到的最合理下一步。
