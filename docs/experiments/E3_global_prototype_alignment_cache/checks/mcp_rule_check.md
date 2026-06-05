# E3 准备检查：MCP Entropy Cache 与 Align Cache 规则确认

更新日期：2026-06-04

## 1. 检查目的

本文档用于记录 MCP 中 Entropy Cache 与 Align Cache 的真实更新规则，并说明 E3 如何借鉴这些规则。

注意：

    checks/ 目录中的内容属于准备检查，不作为实验编号。

## 2. MCP 中 Entropy Cache 的更新规则

MCP 中 Entropy Cache 是低熵缓存。

其基本逻辑为：

    如果该类别 Entropy Cache 未满：
        加入当前样本。

    如果该类别 Entropy Cache 已满：
        找到该类别 Entropy Cache 中当前最高熵样本；
        如果新样本熵更低；
        则替换 Entropy Cache 中这个最高熵样本。

替换只发生在 Entropy Cache 自己内部。

## 3. MCP 中 Align Cache 的更新规则

MCP 中 Align Cache 是更严格的对齐缓存。

其基本逻辑为：

    如果该类别 Align Cache 未满：
        加入当前样本。

    如果该类别 Align Cache 已满：
        找到该类别 Align Cache 中当前最高熵样本；
        如果新样本熵更低；
        并且新样本到原型中心的距离
        小于这个最高熵样本到原型中心的距离；
        则替换 Align Cache 中这个最高熵样本。

注意：

- Align Cache 的低熵比较对象是 Align Cache 自己内部的最高熵样本；
- Align Cache 的距离比较对象也是 Align Cache 自己内部的最高熵样本；
- Align Cache 替换的是 Align Cache 自己内部的最高熵样本；
- 不是替换 Entropy Cache 中的样本。

## 4. Entropy Cache 与 Align Cache 的关系

MCP 中 Entropy Cache 和 Align Cache 是并列更新的两个缓存。

新样本到来后，可以分别尝试更新：

    Entropy Cache
    Align Cache

两者不会互相替换内部样本。

因此，一个样本可能：

1. 只进入 Entropy Cache；
2. 只进入 Align Cache；
3. 同时进入两个缓存；
4. 两个缓存都不进入。

## 5. E3 与 MCP 的关系

E3 内部说明文档中可以记录与 MCP 的对应关系，但论文中不主动强调与 MCP 的关系。

E3-V1 不是完整复现 MCP，而是采用适合 Point-Cache 的顺序式改造：

    Global Entropy Cache
        ↓
    Global Prototype-Alignment Cache
        ↓
    Local Cache

也就是说：

- Global Entropy Cache 保留原始 Point-Cache 的低熵更新逻辑；
- GPA Cache 是新增的更严格缓存；
- 只有进入 GPA Cache 的样本，其局部特征才写入 Local Cache；
- GPA Cache 自己维护类别原型中心；
- 当前最小验证阶段暂不修改最终预测加权公式。

## 6. 后续计划

后续需要继续验证：

1. MCP-style 并列更新方案；
2. 不同原型中心来源；
3. 不同 GPA Cache 准入规则；
4. 是否引入文本原型中心；
5. 是否修改最终预测加权公式。
