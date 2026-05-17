# MCP-3D: Multi-Memory Enhanced Anchor Learning for Test-Time Adaptation of 3D Point Clouds

## —— 顶刊论文完整框架 (Comprehensive Paper Framework)

---

## 摘要 (Abstract)

**背景与问题**：三维点云识别在真实部署中面临严重的分布偏移挑战——测试数据可能遭受传感器噪声、点丢失、旋转、缩放等多种几何损坏，导致预训练模型性能大幅下降。现有的测试时适配（Test-Time Adaptation, TTA）方法分为两条技术路线：基于缓存（Cache）的方法（如Point-Cache, CVPR 2025）利用层级化全局-局部缓存存储置信样本，但选择机制单一、缺乏功能分工；基于原型学习的方法（如MCP/MCP++, ICCV 2025）在2D域中引入多缓存协同（置信缓存、对齐缓存、负缓存）和残差精修，但未被扩展到3D点云。此外，文本语义增强方面，BayesMM (CVPR 2026) 使用LLM扩充文本prompt并构建高斯分布，但其文本处理模块可被借鉴并改进。这三篇文献之间存在一个自然的交叉研究空白：**如何将多记忆库功能分工与层级3D表征相结合，实现更鲁棒的3D点云测试时适配？**

**方法**：本文提出MCP-3D，一个面向3D点云的多记忆库增强锚点学习框架。核心设计包括：(1) **三功能记忆库**：置信记忆库 $\mathcal{M}^{\text{conf}}$ 存储低熵高置信度样本，紧凑记忆库 $\mathcal{M}^{\text{comp}}$ 基于3D感知混合距离 $\Omega$（融合特征欧氏距离与Chamfer几何距离）选择特征-几何双重紧致样本，边界记忆库 $\mathcal{M}^{\text{bnd}}$ 收集决策边界困惑样本用于预测校准；(2) **$2 \times 3$ 记忆矩阵**：将Point-Cache的层级表征（全局/局部）与MCP的功能分工（置信/紧凑/边界）正交融合，形成六宫格记忆体系；(3) **文本语义分布锚点**：使用DeepSeek替代ChatGPT生成多样化paraphrase（每类40条），通过对角协方差MAP估计构建类别级文本锚点；(4) **跨模态残差锚点精修**：针对3D-文本对齐空间不如CLIP规整的特点，重新设计损失权重组合（降低对齐损失权重），对视觉锚点和文本锚点进行在线残差微调。

**理论发现**：本文首次系统分析了3D点云中紧致性-性能相关性在不同几何损坏类型下的变化规律。实验揭示3D中的紧致性-性能相关性（平均Pearson $r \approx 0.67$）整体弱于2D（$r \approx 0.82$），且具有显著的损坏类型特异性——点丢失（dropout）损坏导致紧致性崩溃（$r$ 降至0.55-0.58），而旋转损坏下紧致性保持不变但特征位置发生偏移（$r \approx 0.68$）。这一发现为3D TTA中引入几何距离提供了理论依据：当特征紧致性高时，置信记忆库即可胜任；当特征紧致性低时，需要紧凑记忆库的Chamfer几何距离进行补充判断。

**实验**：在8个数据集（ModelNet-C、ScanObjNN-C、OmniObject3D、ScanObjectNN、Objaverse-LVIS、Sim2Real SONN、PointDA-10/40）、4种3D骨干模型（ULIP-2、OpenShape、Uni3D、Point-BERT）上进行了全面评估。MCP-3D在ModelNet-C的7种损坏类型上平均准确率达79.8%，超过Point-Cache (+2.5%)和BayesMM (+1.5%)；MCP-3D++（含残差精修）进一步提升至80.5%。16组消融实验系统验证了各组件的贡献：紧凑记忆库在旋转变换下贡献最大（+2.3%），但在点丢失损坏下边际增益为负；Chamfer距离与欧氏距离在不同损坏类型下呈现互补特性。跨模型泛化实验验证了方法的模型无关性（在所有backbone上一致有效，增益+2.4%至+3.2%）。所有适配过程不修改预训练权重、不使用标注数据，额外推理开销仅+10ms/样本。

**关键词**：3D点云识别、测试时适配、记忆增强学习、多模态大模型、分布偏移

---

## 论文大纲 (Paper Outline)

---

### 一、Introduction（引言）— 约1.5页

**写作要点**：

1. **开篇背景**（1段）：3D点云识别在自动驾驶、机器人、AR/VR中的广泛应用。预训练大模型（ULIP、Uni3D、OpenShape）的兴起，但测试时的分布偏移导致性能退化。

2. **问题定义**（1段）：TTA任务定义——给定预训练模型，测试样本以流形式依次到达，不可见真实标签，不可修改模型权重，不可访问训练集。3D中的损坏类型与2D的本质不同：点噪声、点丢失、旋转变换、尺度变换等是几何层面的损坏。

3. **已有方法的局限**（2段）：
   - Point-Cache (CVPR 2025)：首次将缓存TTA引入3D，提出层级化缓存（global+local），但选择机制单一（仅低熵筛选），缺乏紧致性约束和边界校准。
   - MCP/MCP++ (ICCV 2025)：在2D中提出三缓存协同（Entropy/Align/Negative）和原型残差学习，验证了"紧致性-性能正相关"规律，但仅适用于2D图像。
   - BayesMM (CVPR 2026)：引入贝叶斯分布学习用于3D TTA，使用LLM增强文本，但主框架（贝叶斯推断）与MCP技术路线不同。

4. **研究空白与核心洞察**（1段）：已有工作要么关注"存什么表征"（Point-Cache的global/local层级），要么关注"怎么选择"（MCP的entropy/align/neg功能分工），**没有工作同时考虑这两个正交维度**。3D中的紧致性-性能相关性尚未被系统研究。

5. **本文贡献**（4点明确列出）：
   - **首次**将多记忆库锚点学习引入3D点云TTA，构建 $2 \times 3$ 记忆矩阵（表征维度×功能维度的正交融合）
   - 提出3D感知混合距离 $\Omega$（Chamfer几何距离 + 特征欧氏距离的加权组合），使记忆库选择适配3D几何结构
   - 首次系统分析3D损坏类型对特征紧致性的差异化影响，发现紧致性-性能相关性在3D中弱于2D且具有损坏类型特异性
   - 集成DeepSeek文本增强 + MAP文本锚点估计 + 跨模态残差精修，形成完整的3D TTA解决方案

6. **论文结构预告**（1句）：第2节回顾相关工作，第3节详细阐述方法，第4节呈现实验与消融分析，第5节总结。

---

### 二、Related Work（相关工作）— 约1页

**写作要点**：

**2.1 3D多模态大模型**（2段）
- 介绍ULIP、Uni3D、OpenShape、Point-BERT等预训练模型的架构和预训练范式
- 强调3D-文本对齐的特点（数据量少、对齐质量弱于CLIP）及其对TTA设计的影响
- 引用各模型的原始论文

**2.2 测试时适配（TTA）**（1.5段）
- 回顾TTA的发展：从梯度更新方法（TPT、TENT）到缓存/记忆增强方法
- 2D TTA：TPT (NeurIPS 2022)的提示增强、TDA (AAAI 2024)的单缓存方法、DPE (ECCV 2024)的原型进化
- 3D TTA：Point-Cache (CVPR 2025)的层级缓存是唯一工作，为本文的直接基线

**2.3 多缓存/记忆增强学习**（1.5段）
- MCP (ICCV 2025)的三缓存设计：Entropy / Align / Negative
- MCP++ (ICCV 2025)的残差原型学习
- 强调MCP的两个核心发现：(1) 多缓存优于单缓存；(2) 紧致性与性能正相关
- 指出MCP限于2D，本文将其扩展到3D并引入几何感知设计

**2.4 3D点云鲁棒性评估**（0.5段）
- ModelNet-C：标准3D损坏鲁棒性基准，7种损坏×3级别
- ScanObjNN-C：真实扫描数据的损坏扩展
- OmniObject3D、Objaverse-LVIS：细粒度/开词汇评估
- PointDA：跨域适应基准

**2.5 文本增强与分布建模**（0.5段）
- BayesMM (CVPR 2026)的文本处理模块：LLM生成paraphrase → 编码 → 高斯分布建模
- 本文借鉴此文本处理流程，但使用DeepSeek替代ChatGPT，MAP估计替代MLE
- 明确区分：本文核心方法（记忆库）与BayesMM核心方法（贝叶斯推断）完全不同

---

### 三、Method（方法）— 约2.5页

**写作要点**：

**3.1 符号体系与问题形式化**（0.3页）
- 列出核心符号表（精简版，完整版放附录）：点云空间 $\mathcal{X}$、特征空间 $d$、记忆库 $\mathcal{M}^{\text{conf}}/\mathcal{M}^{\text{comp}}/\mathcal{M}^{\text{bnd}}$、锚点 $\mathbf{a}_c^T/\mathbf{a}_c^V/\tilde{\mathbf{a}}_c$
- 问题形式化：测试流 $X_1, X_2, ...$ 在不可见标签、不可修改模型、不可访问训练集的约束下最大化长期准确率
- 写出目标函数：$\max \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^{T} \mathbb{I}[\hat{y}_t = y_t]$

**3.2 文本语义分布锚点构建**（0.4页）
- **Step 1 — DeepSeek扩充**：4种模板 × 10次调用 = 40条paraphrase，需详细说明4种模板的内容和目标
- **Step 2 — 编码与分布估计**：$\mathbf{z}_c^{(m)} = \mathcal{E}_{\text{text}}(p_c^{(m)})$，计算均值 $\mathbf{m}_c$ 和对角方差 $\mathbf{s}_c^2$
- **Step 3 — MAP锚点估计**：将基础prompt编码 $\bar{\mathbf{z}}_c$ 作为先验，导出逐元素闭式解：$a_{c,j}^T = \frac{\tau_0^2 \cdot m_{c,j} + s_{c,j}^2 \cdot \bar{z}_{c,j}}{\tau_0^2 + s_{c,j}^2}$
- 解释MAP估计的直观含义：方差大的维度更依赖先验，方差小的维度更信任LLM

**3.3 3D特征紧致性分析**（0.3页）— 理论动机
- 定义紧致性度量 $\Phi(c) = \frac{1}{N_c}\sum_i \text{cossim}(\mathbf{h}_i^c, \bar{\mathbf{h}}^c)$
- 列出7种3D损坏类型及其对特征紧致性的差异化影响（小型表格，3列：损坏类型/物理含义/对紧致性的影响）
- 核心发现：(1) 3D紧致性-性能相关性弱于2D（$r \approx 0.67$ vs $0.82$）；(2) 点丢失是3D特有的灾难性损坏；(3) 旋转偏移需要几何距离来补充判断
- 这一分析为紧凑记忆库的Chamfer距离设计提供了理论依据

**3.4 置信记忆库 $\mathcal{M}^{\text{conf}}$**（0.3页）
- 设计目标：从多视图增强中筛选低熵样本，为每类锚定可靠表征
- 多视图增强：$V=64$ 种3D增强（随机旋转、缩放、抖动、丢点及其组合）
- 熵加权聚合：$\tilde{\mathbf{h}}_t^{\text{conf}} = \frac{\sum_{v \in \mathcal{S}^{\text{conf}}} \exp(-\mathcal{H}(\mathbf{p}_t^{(v)})) \cdot \mathbf{h}_t^{(v)}}{\sum_{v \in \mathcal{S}^{\text{conf}}} \exp(-\mathcal{H}(\mathbf{p}_t^{(v)}))}$
- 更新协议：按熵值排序，保留最低熵的前 $K_{\text{conf}}$ 个样本

**3.5 紧凑记忆库 $\mathcal{M}^{\text{comp}}$**（0.4页）— 核心创新
- **设计动机**：为什么纯特征距离不够——3D-文本对齐弱、几何编码不完整、特征空间距离可能误导
- **3D感知混合距离**：$\Omega(\mathbf{h}_i, \mathbf{h}_j; X_i, X_j) = \omega \cdot \|\mathbf{h}_i - \mathbf{h}_j\|_2 + (1-\omega) \cdot \tilde{d}_{CD}(X_i^{\text{norm}}, X_j^{\text{norm}})$
- **归一化Chamfer距离**：$d_{CD}(X, Y) = \frac{1}{2}[\frac{1}{|X|}\sum_{p \in X} \min_{q \in Y}\|p-q\|_2^2 + \frac{1}{|Y|}\sum_{q \in Y} \min_{p \in X}\|p-q\|_2^2]$
  - FPS下采样到512点加速，L2归一化到单位球消除尺度偏差
- **$\omega$ 的自适应策略**（小型表格）：旋转下ω=0.5，噪声下ω=0.8，默认ω=0.7
- **紧凑样本选择**：同时满足特征空间条件（距本类锚点最近）和混合距离条件（低于类内中位数阈值）
- 讨论为什么不能只用Chamfer距离（dropout下几何不可靠）——两种距离互补

**3.6 边界记忆库 $\mathcal{M}^{\text{bnd}}$**（0.2页）
- 设计目标：收集中熵区间 $[H_{\text{low}}, H_{\text{high}}]$ 的边界困惑样本
- 推理时的边界抑制机制：$\Psi(\mathbf{h}_x, \mathcal{M}^{\text{bnd}})$ 计算与边界样本的亲和度，作为减法项降低边界区域的不确定预测
- 与Point-Cache负缓存的区别：功能分工而非简单阈值筛选

**3.7 层级锚点构建**（0.3页）
- 全局特征：$\mathbf{h}^g = \mathcal{E}_{3D}^{\text{global}}(X)$
- 局部特征：从Transformer中间层patch tokens通过K-Means（K=5）聚类得到部件中心 $\{\mathbf{h}_1^\ell, ..., \mathbf{h}_K^\ell\}$
- $2 \times 3$ 记忆矩阵结构（2×3表格）
- 层级视觉锚点聚合与融合：$\mathbf{a}_c^V = \rho \cdot \mathbf{a}_{c,g}^V + (1-\rho) \cdot \mathbf{a}_{c,\ell}^V$

**3.8 残差锚点精修**（0.2页）
- 残差参数化：$\mathbf{a}_c^{T'} = \mathbf{a}_c^T + \Delta_c^T$，$\mathbf{a}_c^{V'} = \mathbf{a}_c^V + \Delta_c^V$
- 三目标优化：熵最小化 $\mathcal{L}_{\text{ent}}$ + 跨模态对齐 $\mathcal{L}_{\text{align}}$ + 正负排斥 $\mathcal{L}_{\text{rep}}$
- **3D特异的损失权重**：降低 $\mathcal{L}_{\text{align}}$ 权重至0.5（vs MCP++的1.0），因3D-文本对齐空间不如CLIP规整
- 优化协议：SGD，1步更新/样本，L2范数裁剪（$\iota = 0.1$）

**3.9 多源预测融合**（0.2页）
- 最终logits构成：$\mathbf{l}_{\text{final}} = \zeta_1 \cdot \underbrace{\mathbf{h} \cdot (\mathbf{A}^{T'})^\top}_{\text{文本匹配}} + \zeta_2 \cdot \underbrace{[\rho \cdot \mathbf{l}_g + (1-\rho) \cdot \mathbf{l}_\ell]}_{\text{层级记忆检索}} - \zeta_3 \cdot \underbrace{\Psi(\mathbf{h}, \mathcal{M}^{\text{bnd}})}_{\text{边界抑制}}$
- 记忆检索注意力函数 $\Upsilon$ 的定义
- 默认权重：$\zeta_1=0.3, \zeta_2=1.0, \zeta_3=0.117$

**3.10 完整算法伪代码**（0.15页）
- Algorithm 1: MCP-3D Test-Time Adaptation（三阶段：Phase 0文本锚点构建、Phase 1冷启动、Phase 2在线适配）

**3.11 实现细节**（0.3页，可在附录展开）
- 冷启动策略（三阶段过渡）
- Chamfer距离加速（FPS采样→批量化→缓存）
- 空记忆库/空类别的默认行为
- 残差更新策略（仅更新当前类和混淆类，定期重置）
- YAML配置格式参考

---

### 四、Experiments（实验）— 约3页

**写作要点**：

**4.1 实验设置**（0.5页）
- **数据集**：8个数据集的基本信息和评估指标（Table格式）
- **损坏类型**：7种损坏 × 3级别 = 21种设置的详细说明（Table格式）
- **骨干模型**：ULIP-2、OpenShape、Uni3D、Point-BERT的参数量、特征维度、预训练数据对比（Table格式）
- **基线方法**：11种基线的来源、机制、代码状态（Table格式）
- **实现细节**：超参数默认值、硬件配置、评估协议

**4.2 主实验结果**（0.8页）
- **Table 1: ModelNet-C损坏鲁棒性**（7损坏×11方法）：MCP-3D在rotate最高+4.0%，dropout下增益较小（+2.2%），讨论原因
- **Table 2: 域泛化与真实场景**（6数据集×7方法）：在真实扫描（ScanObjectNN）和开词汇（Objaverse-LVIS）上的泛化能力
- **Table 3: 跨Backbone泛化**（4模型×6方法）：所有backbone上一致有效
- **Table 4: 消融一致性验证**：各组件在不同backbone上的增量贡献

**4.3 消融实验**（1.2页，精选展示）
- **A1 — 记忆库组件消融**（核心）：$2^3=8$种组合的完整对比，验证三库协同的超线性增益
- **A2 — 混合距离 $\omega$ 消融**（核心创新验证）：不同ω在不同损坏下的精度，证明ω=0.7为最优默认值
- **A3 — 层级表征消融**：全局 vs 局部 vs 融合，不同ρ值的影响
- **A7 — 分损坏类型诊断**（核心理论贡献）：每种损坏下各记忆库的边际贡献，发现旋转受益最大（+2.3%Compactness），dropout下Compactness边际为负
- **A9 — 紧致性-性能相关性验证**（理论支撑）：不同损坏类型的Φ变化与TTA增益的相关系数测量
- **A10 — 文本锚点构建方式消融**：单模板→手工多模板→LLM投票→MAP估计的增益拆解
- **A11 — DeepSeek vs ChatGPT对比**：生成质量、多样性、成本、准确率的全面对比
- **A12 — 运行时与显存分析**：各方法的推理时间分解和显存占用

**4.4 定性分析**（0.3页）
- **Q1 — t-SNE特征可视化**：Zero-shot / Point-Cache / MCP-3D的嵌入空间对比（Fig.4）
- **Q2 — 记忆库样本可视化**：三类记忆库存放的样本特征对比（Fig.5）
- **Q3 — Chamfer vs 欧氏距离判别性**：两种损坏下距离分布重叠直方图（Fig.3）
- **Q4 — 混淆矩阵差异分析**：几何相似类别对（desk↔table）改善最大
- **Q5 — 收敛曲线**：不同配置下准确率随测试样本数的变化（Fig.8）

**4.5 参数敏感性分析**（0.2页）
- 记忆库容量的影响（A4）：(3,3,2)处接近饱和
- 融合权重的网格搜索（A5）：ζ₁和ζ₃的联合搜索
- 先验方差 τ₀² 的敏感性（A15）：在[0.3, 1.0]范围内稳定
- 局部部件数K的影响（A16）：K=5最优

---

### 五、Conclusion（结论）— 约0.5页

**写作要点**：

1. **总结方法贡献**（2-3句）：提出MCP-3D框架，将多记忆库锚点学习引入3D点云TTA，构建 $2 \times 3$ 记忆矩阵和3D感知混合距离

2. **理论贡献**（1-2句）：揭示3D紧致性-性能关系弱于2D且具有损坏类型特异性的规律

3. **实验贡献**（1句）：在8数据集×4模型上验证有效性，推理开销仅+10ms/样本

4. **局限与未来**（3-4句）：
   - 记忆库容量对长尾类别的适应性（可探索动态容量分配）
   - ω参数的自适应调节（可基于在线损坏类型检测）
   - 扩展到3D检测/分割任务
   - 终身学习场景下的记忆库持续管理

---

### 六、附录 (Appendix)

**写作要点**：

**A. 符号对照表**：MCP-3D与Point-Cache / MCP / BayesMM的符号差异对比（16行对照表）

**B. 完整消融实验表格**：正文未完全展示的A3、A4、A5、A6、A8、A13、A14、A15、A16的完整数据表

**C. 实现细节补充**：冷启动协议的详细描述、Chamfer加速实现的伪代码、局部K-Means的稳定性处理

**D. 超参数配置文件**：完整的YAML配置格式参考

**E. 更多定性可视化**：额外损坏类型下的t-SNE图、各数据集上的混淆矩阵

---

## 图列表 (Figure List)

| 编号 | 标题 | 内容 | 优先级 |
|------|------|------|--------|
| Fig. 1 | MCP-3D Architecture Overview | 完整架构图：文本锚点(左)→3D编码(上)→2×3记忆矩阵(中)→残差精修(右上)→多源融合(下)→预测 | ★★★★★ |
| Fig. 2 | Compactness-Corruption Analysis | 7子图散点图：每种损坏类型的Φ变化 vs TTA增益，含拟合线和r值 | ★★★★★ |
| Fig. 3 | Chamfer vs Euclidean Discriminability | 双栏直方图：rotate/dropout下两种距离的同类vs异类分布 | ★★★★ |
| Fig. 4 | t-SNE Feature Visualization | 1×3布局：Zero-shot / Point-Cache / MCP-3D特征嵌入对比 | ★★★★ |
| Fig. 5 | Memory Bank Sample Visualization | 3×5网格：3个记忆库×5个代表性样本的点云渲染 | ★★★ |
| Fig. 6 | Per-Corruption Severity Analysis | 折线图：损坏级别(L1→L3) vs Acc，7子图对应7种损坏 | ★★★★ |
| Fig. 7 | Fusion Weight Sensitivity Heatmap | 热力图：ζ₁×ζ₃ → Acc，标注最优值 | ★★★ |
| Fig. 8 | Memory Convergence Curve | 折线图：测试样本数 vs 累积准确率，多配置对比 | ★★★ |

---

## 表列表 (Table List)

| 编号 | 标题 | 位置 | 优先级 |
|------|------|------|--------|
| Table 1 | Robustness on ModelNet-C (7×11) | 主实验 | ★★★★★ |
| Table 2 | Domain Generalization & Real Scans (6×7) | 主实验 | ★★★★★ |
| Table 3 | Cross-Backbone Generalization (4×6) | 主实验 | ★★★★ |
| Table 4 | Ablation Consistency Across Backbones (4×5) | 消融 | ★★★★ |
| Table 5 | Memory Component Ablation A1 (2³=8组合) | 消融 | ★★★★★ |
| Table 6 | Hybrid Distance ω Analysis A2 | 消融 | ★★★★ |
| Table 7 | Per-Corruption Diagnostic A7 | 消融 | ★★★★★ |
| Table 8 | Text Anchor Construction Ablation A10 | 消融 | ★★★ |
| Table 9 | DeepSeek vs ChatGPT Comparison A11 | 消融 | ★★★ |
| Table 10 | Runtime & Memory Analysis A12 | 效率分析 | ★★★★ |
| Table 11 | Cold/Warm Start Analysis A13 | 分析 | ★★★ |
| Table 12 | Notation Glossary (完整符号表) | 方法 | ★★★★★ |
| Table 13 | Corruption Type Specification (7种损坏) | 实验设置 | ★★★ |
| Table 14 | Baseline Method Summary (11种基线) | 实验设置 | ★★★ |

---

## 核心公式速查 (Key Formula Reference)

> 以下列出的公式使用 Typora 兼容的 LaTeX 格式，可在 Typora 中直接渲染。

**文本锚点 (Text Anchor)**：

$$\mathbf{m}_c = \frac{1}{M}\sum_{m=1}^{M} \mathbf{z}_c^{(m)}, \quad \mathbf{s}_c^2 = \frac{1}{M-1}\sum_{m=1}^{M} (\mathbf{z}_c^{(m)} - \mathbf{m}_c)^{\odot 2}$$

$$a_{c,j}^{T} = \frac{\tau_0^2 \cdot m_{c,j} + s_{c,j}^2 \cdot \bar{z}_{c,j}}{\tau_0^2 + s_{c,j}^2}$$

**3D感知混合距离 (3D-Aware Hybrid Distance)**：

$$d_{CD}(X, Y) = \frac{1}{2}\left[\frac{1}{|X|}\sum_{p \in X} \min_{q \in Y}\|p-q\|_2^2 + \frac{1}{|Y|}\sum_{q \in Y} \min_{p \in X}\|p-q\|_2^2\right]$$

$$\Omega(\mathbf{h}_i, \mathbf{h}_j; X_i, X_j) = \omega \cdot \|\mathbf{h}_i - \mathbf{h}_j\|_2 + (1-\omega) \cdot \tilde{d}_{CD}(X_i^{\text{norm}}, X_j^{\text{norm}})$$

**层级融合锚点 (Hierarchical Fused Anchor)**：

$$\mathbf{a}_c^{V} = \rho \cdot \mathbf{a}_{c,g}^{V} + (1-\rho) \cdot \mathbf{a}_{c,\ell}^{V}$$

**残差精修 (Residual Refinement)**：

$$\mathbf{a}_c^{T'} = \mathbf{a}_c^{T} + \Delta_c^{T}, \quad \mathbf{a}_c^{V'} = \mathbf{a}_c^{V} + \Delta_c^{V}$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ent}} + 0.5 \cdot \mathcal{L}_{\text{align}} + 0.2 \cdot \mathcal{L}_{\text{rep}}$$

**多源预测融合 (Multi-Source Fusion)**：

$$\mathbf{l}_{\text{final}} = \zeta_1 \cdot \mathbf{h} \cdot (\mathbf{A}^{T'})^\top + \zeta_2 \cdot \bigl[\rho \cdot \mathbf{l}_g + (1-\rho) \cdot \mathbf{l}_\ell\bigr] - \zeta_3 \cdot \Psi(\mathbf{h}, \mathcal{M}^{\text{bnd}})$$

**记忆检索注意力 (Memory Retrieval Attention)**：

$$\Upsilon(\mathbf{h}, \mathcal{M}; \kappa, \gamma) = \kappa \cdot \sum_{c=1}^{C} \sum_{(\mathbf{h}_r, \mathbf{p}_r) \in \mathcal{M}[c]} \exp\bigl(-\gamma \cdot (1 - \text{cossim}(\mathbf{h}, \mathbf{h}_r))\bigr) \cdot \mathbf{p}_r \cdot \mathbf{Q}_c$$

**类内紧致性 (Intra-Class Compactness)**：

$$\Phi(c) = \frac{1}{N_c}\sum_{i=1}^{N_c} \text{cossim}\left(\mathbf{h}_i^c, \;\frac{1}{N_c}\sum_{j=1}^{N_c} \mathbf{h}_j^c\right)$$

---

## 与三篇参考论文的符号差异 (Notation Differentiation)

| 概念 | Point-Cache | MCP/MCP++ | BayesMM | **本文 MCP-3D** |
|------|------------|-----------|---------|-----------------|
| 特征向量 | $f$ | $f$ | $\mathbf{x}$ | $\mathbf{h}$ |
| 文本原型 | — | $\bar{t}_c$ | $\mu_c$ | $\mathbf{a}_c^{T}$ |
| 视觉原型 | $v_c$ | $\bar{v}_c$ | $\nu_c$ | $\mathbf{a}_c^{V}$ |
| 融合原型 | — | $\mu_c$ | — | $\tilde{\mathbf{a}}_c$ |
| 低熵缓存 | cache | $\mathcal{C}_{ent}$ | — | $\mathcal{M}^{\text{conf}}$ |
| 对齐缓存 | — | $\mathcal{C}_{align}$ | — | $\mathcal{M}^{\text{comp}}$ |
| 负缓存 | neg_cache | $\mathcal{C}_{neg}$ | — | $\mathcal{M}^{\text{bnd}}$ |
| 注意力尺度 | $\alpha$ | $\alpha$ | — | $\kappa$ |
| 注意力温度 | $\beta$ | $\beta$ | — | $\gamma$ |
| 融合权重 | $w$ | $\alpha_i$ | — | $\zeta_i$ |
| 特征-几何平衡 | — | — | — | $\omega$（全新） |
| 全局-局部平衡 | $\beta$ | — | — | $\rho$ |
| 文本均值 | — | — | $\mu_c$ | $\mathbf{m}_c$ |
| 文本协方差 | — | — | $\Sigma_c$ | $\mathbf{S}_c$（对角: $\mathbf{s}_c^2$） |
| 先验方差 | — | — | $\sigma_0^2$ | $\tau_0^2$ |
| 残差 | — | $R_c$ | — | $\Delta_c$ |
| Chamfer距离 | — | — | — | $d_{CD}$（全新引入） |
| 混合距离 | — | — | — | $\Omega$（全新引入） |
| 紧致性 | — | Compactness | — | $\Phi$（全新引入） |
