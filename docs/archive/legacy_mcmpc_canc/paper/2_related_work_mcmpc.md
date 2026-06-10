# §2 Related Work (草稿 v0.1, 2026-05-10)

> **状态**：早期草稿，每个 [需补 X] 占位符等具体实验/citation 完成后填。
>
> **更新触发**：每次完成一个相关工作的对照实验，回来扩写对应段落。

---

## 2.1 测试时适配在 3D 识别中的应用 (Test-Time Adaptation for 3D Recognition)

3D 识别系统在部署后常面临训练分布之外的扰动——扫描设备噪声、采样点缺失、尺度归一化误差等。**测试时适配** (Test-Time Adaptation, TTA) 旨在不改变模型权重的前提下，于推理阶段自适应地调整模型行为。

早期 3D TTA 方法 (T3A [需补 ref]、TENT-3D [需补 ref]、MEMO-3D [需补 ref]) 多由二维 TTA 直接迁移，依赖批次统计量重校准或熵最小化损失。这些方法假设有同分布的小批次数据，与真实部署中的样本流式到达不符。**Point-Cache** [需补 ref, 2024] 提出基于缓存 (cache) 的免训练 TTA：维护历史样本的特征库，新样本通过 K 近邻 (K-Nearest Neighbors, KNN) 检索得到软标签作为辅助信号。其层级化变体 (hierarchical) 进一步将缓存按全局/局部特征分组，在 ModelNet-C 上取得 76.59% 的 35-setting 平均准确率。

**局限**：上述方法的样本-cache 关联完全建立在**特征余弦相似度**上。我们在复现实验中观察到一个未被以往工作单独讨论的现象 (本文 §4.1, F1)：**在 scale 类全局形变下，Point-Cache hierarchical TTA 反而比零样本基线退化 -0.40pp** (5 个 severity 中 4 个为负增益)。这一现象的机制在于 scale 是整体特征流形的偏移，特征空间的近邻检索难以识别"被同样污染"的历史样本——反而把它们当作"同类"用于投票，加重错误。本文 C1 (ICP-CD 几何距离) 通过引入与特征空间正交的几何信号弥补这一盲区。

[需补：W2.5 P5 完成后追加跨方法 (global cache vs hierarchical) / 跨 backbone (OpenShape vs ULIP-2) 的退化证据，确认现象普遍性]

---

## 2.2 视觉-语言预训练驱动的 3D 零样本分类 (VLP-based 3D Zero-Shot Classification)

将 **CLIP** (Contrastive Language-Image Pre-training [需补 ref, 2021]) 的图文对齐范式扩展到 3D 是近年的主要技术路径。代表性工作包括 PointCLIP [需补 ref]、ULIP / ULIP-2 [需补 ref]、OpenShape [需补 ref]、Uni3D [需补 ref] 等。这些方法在多视角投影、跨模态蒸馏等技术上各有侧重，但**类原型** (class prototype) **的构造方式高度一致**——将文本模板字符串 (如 `"a point cloud of a {class_name}"`) 输入文本编码器，得到一个单点向量作为该类的锚点。

**局限**：单一文本向量难以表达类内的几何多样性。"椅子"涵盖办公椅、扶手椅、餐椅、沙发椅等数千种几何变体；以单一向量作锚点意味着所有变体共享同一个"类中心"，与真实分布存在系统性偏差。在跨域场景 (合成训练 → 真实扫描测试) 下，这种偏差被进一步放大。

本文 C2 (vMF 文本锚点) 借鉴二维领域的 BayesMM [需补 ref] 思想：对每类生成多条文本变体 (paraphrase)，将其特征视为球面上的样本，用 **冯·米塞斯-费舍尔分布** (von Mises-Fisher distribution, **vMF**) 的最大后验估计 (Maximum a Posteriori, MAP) 拟合一个有信心区间的分布锚点，而非单点向量。

---

## 2.3 缓存与记忆架构在 TTA 中的设计 (Cache/Memory Architectures in TTA)

近期免训练 TTA 方法对"缓存如何组织"展开了多种探索。**TDA** (Training-free Dynamic Adapter [需补 ref, 2024]) 设计正/负双缓存——正缓存收集高置信样本作为类原型，负缓存收集低置信样本作为不确定性参考。Point-Cache 的 hierarchical 变体按特征层级 (全局 vs 局部) 分缓存。**DMN** (Dual Memory Network [需补 ref]) 引入门控机制动态调度多缓存。

**局限**：现有缓存均为**一维分组**——要么按"信心程度"分 (TDA)，要么按"特征层级"分 (Point-Cache hierarchical)，缺乏二者的正交组合。在 3D TTA 场景下，"信心程度"和"信号源"是两条独立的判别维度——一个低置信样本可能在全局形状上模糊但局部几何上明确 (例：椅子腿断裂的椅子)，反之亦然。

本文 C3 (2×3 记忆矩阵) 显式建模这两个维度：行向量为信心 (高/低)，列向量为信号源 (全局特征 / 局部特征 / 几何 ICP-CD)。每格独立维护一组样本，通过 z-score 归一化实现跨格融合。该设计将 C1 的几何信号与 C2 的文本锚点统一到一个正交结构中。

---

## 局限 → 贡献 对应表 (写作 anchor)

| Related Work 局限 | 本文贡献 | 实证依据 |
|---|---|---|
| §2.1 特征余弦在 scale 类全局形变下失效 | C1 ICP-CD 几何距离 | F1 (-0.40pp), W4 主实验 |
| §2.2 单一文本锚点缺乏多样性 | C2 vMF 文本锚点 | W3 vMF vs 单点 ablation |
| §2.3 缓存维度单一 (信心 ⊻ 层级) | C3 2×3 记忆矩阵 (信心 × 信号源) | W5 6-cell leave-one-out |

---

## 待办

- [ ] 补 7 处 [需补 ref] 引用
- [ ] §2.1 末段补 P5 跨条件实验数字
- [ ] §2.3 末段考虑加 BayesMM 的 cache 设计 (如果它有的话) 作为对比
- [ ] 与 §3.5 (2×3 矩阵) 的术语对齐 (避免说 "高/低置信" 在 §2.3 但又说 "正/负缓存" 在 §3.5)
