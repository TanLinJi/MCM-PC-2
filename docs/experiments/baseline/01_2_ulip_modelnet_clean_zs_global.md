# 01_2_ulip_modelnet_clean_zs_global

## 1. 实验名称

ULIP × ModelNet clean × Zero-shot + Global Cache。

本实验是 01_ulip_modelnet_clean 实验组中的第二个子实验。

| 项目 | 内容 |
|---|---|
| 实验编号 | 01_2_ulip_modelnet_clean_zs_global |
| 实验组 | 01_ulip_modelnet_clean |
| Backbone | ULIP |
| 数据集 | ModelNet clean |
| 方法 | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |
| 方法类型 | Point-Cache Global Cache |
| 运行范围 | clean 数据，只运行 1 个数据文件 |

---

## 2. 实验目的

本实验用于复现 ULIP 在 clean ModelNet 数据上加入 Global Cache 后的结果。

它的作用有三个：

| 目的 | 说明 |
|---|---|
| 复现 clean Global Cache baseline | 对齐原论文 Table 1 中 ULIP + Global Cache 的 clean ModelNet 结果 |
| 验证全局缓存收益 | 与 01_1 Zero-shot 对比，观察 Global Cache 是否提升准确率 |
| 为完整 Point-Cache 提供前置对照 | 后续 01_3 会在本实验基础上进一步加入 Local Cache |

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
| 方法脚本 | Point-Cache/scripts/baseline/01_2_ulip_modelnet_clean_zs_global_single_gpu.sh |
| 结果目录 | Point-Cache/results/baseline/01_2_ulip_modelnet_clean_zs_global |
| 结果汇总 | Point-Cache/results/baseline/01_2_ulip_modelnet_clean_zs_global/summary.csv |
| 日志目录 | Point-Cache/results/baseline/01_2_ulip_modelnet_clean_zs_global/logs |

---

## 5. 运行命令

使用第一张 T4：

| 命令 |
|---|
| cd /root/autodl-tmp/MCM-PC-2/Point-Cache |
| bash scripts/baseline/01_2_ulip_modelnet_clean_zs_global_single_gpu.sh 0 |

使用第二张 T4：

| 命令 |
|---|
| cd /root/autodl-tmp/MCM-PC-2/Point-Cache |
| bash scripts/baseline/01_2_ulip_modelnet_clean_zs_global_single_gpu.sh 1 |

本次实际运行使用的是单张 T4。

---

## 6. 方法说明：Zero-shot + Global Cache

本实验在 ULIP 原始 Zero-shot 推理基础上加入 Global Cache。

Global Cache 的基本流程可以简化为：

| 步骤 | 说明 |
|---|---|
| 1 | 对当前测试样本先进行 zero-shot 预测 |
| 2 | 计算预测分布的 entropy，用 entropy 衡量置信度 |
| 3 | 将低 entropy、高置信度样本的全局点云特征存入缓存 |
| 4 | 每个类别最多缓存固定数量的高质量样本 |
| 5 | 后续样本到来时，检索与其相似的全局缓存特征 |
| 6 | 将 Global Cache logits 与 zero-shot logits 融合，得到最终预测 |

本实验只使用全局点云特征缓存，不使用局部 patch 特征缓存。

---

## 7. 关键参数

| 参数 | 数值 | 说明 |
|---|---:|---|
| shot_capacity | 3 | 每个类别最多缓存 3 个样本 |
| n_cluster | 3 | 日志会打印该参数，但本实验不使用 Local Cache |
| alpha | 4.0 | cache logits 的融合权重 |
| beta | 3.0 | cache attention / affinity 的锐度系数 |
| npoints | 1024 | 每个点云输入点数 |
| cache_type | global | 使用 Global Cache |

---

## 8. 模型与权重加载记录

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

## 9. cache 构建记录

运行日志中出现：

| 日志项 | 数值 | 含义 |
|---|---:|---|
| cfg positive shot_capacity | 3 | 每个类别最多存 3 个 positive cache 样本 |
| n_cluster in KMeans | 3 | 局部聚类数量参数，本实验虽然打印但不使用 local cache |
| alpha | 4.0 | cache logits 权重 |
| beta | 3.0 | cache attention sharpness |
| len(pos_cache) | 40 | 40 个类别均建立 positive cache 入口 |

注意：`len(pos_cache): 40` 表示有 40 个类别入口，不是只缓存了 40 个样本。

---

## 10. 实验完成情况

| 项目 | 结果 |
|---|---:|
| 应运行数据文件 | 1 |
| 实际完成数据文件 | 1 |
| failed 数量 | 0 |
| failed_parse_acc 数量 | 0 |
| 最终状态 | 完成 |

---

## 11. 与原论文对齐结果

| 方法 | 当前复现 Acc | 原论文 Acc | Diff |
|---|---:|---:|---:|
| ZS + Global Cache | 62.12 | 62.12 | +0.00 |

分析：

1. 当前复现的 ZS + Global Cache 准确率为 62.12。
2. 原论文 Table 1 中 ULIP + Global Cache 在 clean ModelNet 上也是 62.12。
3. 当前复现与原论文完全一致。
4. 因此，本实验可以认为复现成功。

---

## 12. 与 Zero-shot 的对比

| 方法 | Acc | 相比 Zero-shot |
|---|---:|---:|
| Zero-shot | 56.77 | 0.00 |
| ZS + Global Cache | 62.12 | +5.35 |

分析：

1. Global Cache 在 clean ModelNet 上带来 +5.35 的提升。
2. 即使没有 corruption，测试流中的高置信度样本仍然能为后续预测提供有用信息。
3. 说明 Global Cache 不仅用于鲁棒性修复，也可以提升 clean 数据上的整体识别准确率。
4. 该提升是 01 组实验中最主要的提升来源。

---

## 13. 与 Hierarchical Cache 的关系

| 方法 | Acc | 相比 Global Cache |
|---|---:|---:|
| ZS + Global Cache | 62.12 | 0.00 |
| ZS + Global + Local Cache | 64.18 | +2.06 |

分析：

1. 01_3 在本实验基础上进一步加入 Local Cache。
2. Local Cache 在 clean ModelNet 上额外提升 +2.06。
3. 因此，Global Cache 是基础提升来源，Local Cache 是进一步补充。
4. clean 数据上二者都有效，说明 Point-Cache 的 hierarchical design 在非 corrupted 场景中也有意义。

---

## 14. 运行过程中的 accuracy 解释

日志中会出现多次 accuracy，例如：

| 类型 | 含义 |
|---|---|
| 中间 accuracy | 测试过程中每隔一定样本数打印的累计平均准确率 |
| Final accuracy | 全部测试样本跑完后的最终准确率 |

本实验真正记录的是最后一行 Final accuracy：

| 指标 | 数值 |
|---|---:|
| Final TDA test accuracy | 62.12 |

日志中打印的是 `TDA's test accuracy`，这是原始 runner 里的命名。  
在本实验记录中，它对应的是 `ZS + Global Cache` 的最终准确率。

---

## 15. 关键观察

| 观察 | 说明 |
|---|---|
| Global Cache 复现完全对齐 | 当前复现 62.12，与原论文 62.12 一致 |
| 相比 Zero-shot 提升明显 | 从 56.77 提升到 62.12，提升 +5.35 |
| clean 数据上缓存仍然有效 | 说明缓存机制不只是对 corruption 有用 |
| Global Cache 是主要提升来源 | 后续 Hierarchical Cache 的提升是在此基础上继续增加 |
| 结果可作为 01_3 的直接对照 | 用于衡量 Local Cache 的额外贡献 |

---

## 16. 阶段性结论

本实验可以记录为：

ULIP × ModelNet clean × ZS + Global Cache 复现成功。

主要依据如下：

1. 实验成功运行完成。
2. 最终准确率为 62.12。
3. 与原论文 62.12 完全一致。
4. 相比 Zero-shot 提升 +5.35。
5. 数据、权重、缓存构建和日志解析均正常。
6. 该结果可作为 clean ModelNet 上 Hierarchical Cache 的直接对照。

---

## 17. 后续记录

下一步应补充：

| 顺序 | 文档 |
|---:|---|
| 1 | 01_3_ulip_modelnet_clean_zs_global_local.md |

