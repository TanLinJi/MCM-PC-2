# 01_3_ulip_modelnet_clean_zs_global_local

## 1. 实验名称

ULIP × ModelNet clean × Zero-shot + Global Cache + Local Cache。

本实验是 01_ulip_modelnet_clean 实验组中的第三个子实验，也是 clean ModelNet 上的完整 Point-Cache 方法。

| 项目 | 内容 |
|---|---|
| 实验编号 | 01_3_ulip_modelnet_clean_zs_global_local |
| 实验组 | 01_ulip_modelnet_clean |
| Backbone | ULIP |
| 数据集 | ModelNet clean |
| 方法 | Zero-shot + Global Cache + Local Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 是 |
| 方法类型 | Hierarchical Cache / 完整 Point-Cache |
| 运行范围 | clean 数据，只运行 1 个数据文件 |

---

## 2. 实验目的

本实验用于复现 ULIP 在 clean ModelNet 数据上加入完整 Point-Cache 后的结果。

它的作用有三个：

| 目的 | 说明 |
|---|---|
| 复现 clean Hierarchical Cache baseline | 对齐原论文 Table 1 中 ULIP + Hierarchical Cache 的 clean ModelNet 结果 |
| 验证完整 Point-Cache 的效果 | 与 01_1 Zero-shot 对比，观察完整方法的总提升 |
| 分析 Local Cache 的额外贡献 | 与 01_2 Global Cache 对比，观察 Local Cache 是否进一步提升 |

---

## 3. 数据说明

本实验实际读取的数据文件是：

| 项目 | 内容 |
|---|---|
| 数据集参数 | modelnet_c |
| corruption 类型 | clean |
| 实际数据文件 | Point-Cache/data/modelnet_c/clean.h5 |
| 点数 | 1024 |
| 类别数 | 40 |

注意：虽然代码中的 dataset 参数是 modelnet_c，但本实验读取的是 clean.h5，因此它是 clean ModelNet 数据，不是 corrupted 数据。

---

## 4. 实验脚本与结果路径

| 项目 | 路径 |
|---|---|
| 方法脚本 | Point-Cache/scripts/baseline/01_3_ulip_modelnet_clean_zs_global_local_single_gpu.sh |
| 结果目录 | Point-Cache/results/baseline/01_3_ulip_modelnet_clean_zs_global_local |
| 结果汇总 | Point-Cache/results/baseline/01_3_ulip_modelnet_clean_zs_global_local/summary.csv |
| 日志目录 | Point-Cache/results/baseline/01_3_ulip_modelnet_clean_zs_global_local/logs |

---

## 5. 运行命令

使用第一张 T4：

| 命令 |
|---|
| cd /root/autodl-tmp/MCM-PC-2/Point-Cache |
| bash scripts/baseline/01_3_ulip_modelnet_clean_zs_global_local_single_gpu.sh 0 |

使用第二张 T4：

| 命令 |
|---|
| cd /root/autodl-tmp/MCM-PC-2/Point-Cache |
| bash scripts/baseline/01_3_ulip_modelnet_clean_zs_global_local_single_gpu.sh 1 |

本次实际运行使用的是单张 T4。

---

## 6. 方法说明：Zero-shot + Global Cache + Local Cache

本实验使用完整 Point-Cache，也就是 Hierarchical Cache。

完整方法由三部分组成：

| 组成部分 | 说明 |
|---|---|
| Zero-shot logits | ULIP 原始点云-文本相似度预测 |
| Global Cache logits | 基于全局点云特征的缓存检索结果 |
| Local Cache logits | 基于局部 patch 特征聚类后的缓存检索结果 |

最终预测可以理解为：

| 预测来源 | 作用 |
|---|---|
| Zero-shot | 提供原始语义分类能力 |
| Global Cache | 利用测试流中可靠样本的整体形状信息 |
| Local Cache | 利用局部几何细节补充全局特征不足 |

---

## 7. 与 01_2 的区别

| 实验 | 使用 Global Cache | 使用 Local Cache | 说明 |
|---|---|---|---|
| 01_2 | 是 | 否 | 只使用全局点云特征缓存 |
| 01_3 | 是 | 是 | 同时使用全局缓存和局部缓存 |

因此，`01_3 - 01_2` 的差值可以理解为 Local Cache 在 clean ModelNet 上的额外贡献。

---

## 8. 关键参数

| 参数 | 数值 | 说明 |
|---|---:|---|
| shot_capacity | 3 | 每个类别最多缓存 3 个样本 |
| n_cluster | 3 | 每个点云的局部 patch 聚类数 |
| alpha | 4.0 | cache logits 的融合权重 |
| beta | 3.0 | cache attention / affinity 的锐度系数 |
| npoints | 1024 | 每个点云输入点数 |
| cache_type | hierarchical | 使用完整层级缓存 |

---

## 9. 模型与权重加载记录

运行日志中出现：

| 日志项 | 数值 | 含义 |
|---|---:|---|
| clip_model.state_dict | 150 | 当前文本模型参数项数量 |
| pretrain_slip_sd | 150 | SLIP 预训练权重参数项数量 |
| lm3d_model.state_dict | 161 | 当前 ULIP 点云模型参数项数量 |
| pretrain_point_sd | 161 | ULIP 点云预训练权重参数项数量 |

说明当前模型结构与预训练权重基本匹配。

模型参数量记录为：

| 模块 | 参数量 |
|---|---:|
| clip_params | 63,428,097 |
| lm3d_params | 22,252,032 |
| total_params | 85,680,129 |

---

## 10. cache 构建记录

运行日志中出现：

| 日志项 | 数值 | 含义 |
|---|---:|---|
| cfg positive shot_capacity | 3 | 每个类别最多存 3 个 positive cache 样本 |
| n_cluster in KMeans | 3 | 每个点云聚类成 3 个局部特征中心 |
| alpha | 4.0 | cache logits 权重 |
| beta | 3.0 | cache attention sharpness |
| len(pos_cache) | 40 | 40 个类别均建立 global positive cache 入口 |
| len(pos_local_cache) | 40 | 40 个类别均建立 local positive cache 入口 |

注意：

| 项目 | 解释 |
|---|---|
| len(pos_cache): 40 | 表示 global cache 有 40 个类别入口，不是只缓存 40 个样本 |
| len(pos_local_cache): 40 | 表示 local cache 有 40 个类别入口，不是只缓存 40 个局部特征 |
| shot_capacity = 3 | 每个类别最多缓存 3 个可靠样本 |
| n_cluster = 3 | 每个样本的局部 patch 被聚类成 3 个局部中心 |

---

## 11. 实验完成情况

| 项目 | 结果 |
|---|---:|
| 应运行数据文件 | 1 |
| 实际完成数据文件 | 1 |
| failed 数量 | 0 |
| failed_parse_acc 数量 | 0 |
| 最终状态 | 完成 |

---

## 12. 与原论文对齐结果

| 方法 | 当前复现 Acc | 原论文 Acc | Diff |
|---|---:|---:|---:|
| ZS + Global + Local Cache | 64.18 | 64.22 | -0.04 |

分析：

1. 当前复现的 ZS + Global + Local Cache 准确率为 64.18。
2. 原论文 Table 1 中 ULIP + Hierarchical Cache 在 clean ModelNet 上为 64.22。
3. 当前复现与原论文仅相差 -0.04。
4. 这个误差非常小，可以认为本实验复现成功。

---

## 13. 与 Zero-shot 的对比

| 方法 | Acc | 相比 Zero-shot |
|---|---:|---:|
| Zero-shot | 56.77 | 0.00 |
| ZS + Global + Local Cache | 64.18 | +7.41 |

分析：

1. 完整 Point-Cache 在 clean ModelNet 上带来 +7.41 的提升。
2. 说明即使没有 corruption，测试流缓存信息仍然能显著增强 ULIP 的识别能力。
3. 这个提升比单独 Global Cache 的 +5.35 更高。
4. 因此，完整层级缓存结构在 clean 数据上也是有效的。

---

## 14. 与 Global Cache 的对比

| 方法 | Acc | 相比 Global Cache |
|---|---:|---:|
| ZS + Global Cache | 62.12 | 0.00 |
| ZS + Global + Local Cache | 64.18 | +2.06 |

分析：

1. Local Cache 在 Global Cache 基础上额外带来 +2.06 的提升。
2. 这个提升说明局部 patch 信息在 clean ModelNet 上也具有补充价值。
3. Global Cache 提供整体形状信息，Local Cache 进一步补充局部几何细节。
4. 因此，Hierarchical Cache 的 coarse-to-fine 设计在 clean 数据上也成立。

---

## 15. 三种方法横向对比

| 方法 | Acc | 相比 Zero-shot | 相比上一阶段 |
|---|---:|---:|---:|
| Zero-shot | 56.77 | 0.00 | - |
| ZS + Global Cache | 62.12 | +5.35 | +5.35 |
| ZS + Global + Local Cache | 64.18 | +7.41 | +2.06 |

分析：

1. Zero-shot 是原始基线。
2. Global Cache 是主要提升来源。
3. Local Cache 在 Global Cache 基础上继续提升。
4. 三种方法呈现稳定递增趋势：Zero-shot < Global Cache < Global + Local Cache。

---

## 16. 运行过程中的 accuracy 解释

日志中会出现多次 accuracy，例如：

| 类型 | 含义 |
|---|---|
| 中间 accuracy | 测试过程中每隔一定样本数打印的累计平均准确率 |
| Final accuracy | 全部测试样本跑完后的最终准确率 |

本实验真正记录的是最后一行 Final accuracy：

| 指标 | 数值 |
|---|---:|
| Final TDA test accuracy | 64.18 |

日志中打印的是 `TDA's test accuracy`，这是原始 runner 里的命名。  
在本实验记录中，它对应的是 `ZS + Global + Local Cache` 的最终准确率。

---

## 17. 关键观察

| 观察 | 说明 |
|---|---|
| Hierarchical Cache 复现高度对齐 | 当前复现 64.18，原论文 64.22，仅差 -0.04 |
| 完整 Point-Cache 表现最好 | 64.18 高于 Zero-shot 和 Global Cache |
| Global Cache 是主要提升来源 | Zero-shot 到 Global 提升 +5.35 |
| Local Cache 继续提升 | Global 到 Hierarchical 额外提升 +2.06 |
| clean 数据上局部信息有效 | Local Cache 在没有 corruption 时也有明显收益 |
| 结果可作为后续 corrupted 实验对照 | clean 上趋势正确，说明方法链路基本正常 |

---

## 18. 与 02 组实验的关系

| 实验组 | 数据 | 方法趋势 |
|---|---|---|
| 01_ulip_modelnet_clean | clean ModelNet | Zero-shot < Global < Hierarchical |
| 02_ulip_modelnetc_corruptions_all35 | ModelNet-C corrupted all35 | Zero-shot < Global < Hierarchical |

说明：

1. clean 数据和 corrupted 数据上，Point-Cache 的方法趋势一致。
2. Global Cache 都是主要提升来源。
3. Local Cache 都能继续提升，但在 corrupted all35 中 Local Cache 的额外提升更依赖 corruption 类型。
4. clean 数据上 Local Cache 额外提升 +2.06，比 ModelNet-C all35 的平均 +0.93 更明显。

---

## 19. 与后续 MCM-PC 改进的关系

本实验提供了一个重要参照：

| 现象 | 含义 |
|---|---|
| clean 上 Local Cache 提升明显 | 局部特征本身是有价值的 |
| corrupted all35 上 Local Cache 平均提升变小 | 损坏会影响局部特征可靠性 |
| clean 与 corrupted 的 Local Cache 收益差异明显 | 后续 MCM-PC 可以重点研究局部可靠性判断 |
| Global + Local 固定融合仍有效 | 但仍有可能通过动态融合进一步改进 |

因此，后续 MCM-PC 方向可以考虑：

| 方向 | 说明 |
|---|---|
| local reliability estimation | 判断局部缓存是否可靠 |
| conflict-aware fusion | 当 global 和 local 冲突时动态调整权重 |
| corruption-aware weighting | 根据样本损坏程度调整 local cache 贡献 |
| negative / suppression mechanism | 对不可靠缓存或错误伪标签进行抑制 |

---

## 20. 阶段性结论

本实验可以记录为：

ULIP × ModelNet clean × ZS + Global + Local Cache 复现成功。

主要依据如下：

1. 实验成功运行完成。
2. 最终准确率为 64.18。
3. 与原论文 64.22 仅差 -0.04。
4. 相比 Zero-shot 提升 +7.41。
5. 相比 Global Cache 进一步提升 +2.06。
6. 数据、权重、Global Cache、Local Cache 构建和日志解析均正常。
7. 完整 Point-Cache 在 clean ModelNet 上复现成功。

---

## 21. 后续记录

至此，01 组文档已经基本补全：

| 文档 | 状态 |
|---|---|
| 01_ulip_modelnet_clean_summary.md | 已补充 |
| 01_1_ulip_modelnet_clean_zs.md | 已补充 |
| 01_2_ulip_modelnet_clean_zs_global.md | 已补充 |
| 01_3_ulip_modelnet_clean_zs_global_local.md | 当前补充 |

下一步建议检查文档和脚本状态，然后进行一次 Git 分类型提交。

