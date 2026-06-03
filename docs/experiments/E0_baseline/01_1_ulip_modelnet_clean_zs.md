# 01_1_ulip_modelnet_clean_zs

## 1. 实验名称

ULIP × ModelNet clean × Zero-shot。

本实验是 01_ulip_modelnet_clean 实验组中的第一个子实验。

| 项目 | 内容 |
|---|---|
| 实验编号 | 01_1_ulip_modelnet_clean_zs |
| 实验组 | 01_ulip_modelnet_clean |
| Backbone | ULIP |
| 数据集 | ModelNet clean |
| 方法 | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |
| 是否使用 test-time adaptation | 否 |
| 运行范围 | clean 数据，只运行 1 个数据文件 |

---

## 2. 实验目的

本实验用于复现 ULIP 在 clean ModelNet 数据上的原始 Zero-shot 结果。

它的作用有三个：

| 目的 | 说明 |
|---|---|
| 建立 clean baseline | 后续 Global Cache 和 Hierarchical Cache 都要与本实验比较 |
| 检查数据与权重是否正常 | 如果 clean Zero-shot 明显异常，后续 cache 结果也不可信 |
| 对齐原论文 Table 1 | 原论文中 ULIP clean ModelNet Zero-shot 为 56.16 |

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
| 方法脚本 | Point-Cache/scripts/baseline/01_1_ulip_modelnet_clean_zs_single_gpu.sh |
| 结果目录 | Point-Cache/results/baseline/01_1_ulip_modelnet_clean_zs |
| 结果汇总 | Point-Cache/results/baseline/01_1_ulip_modelnet_clean_zs/summary.csv |
| 日志目录 | Point-Cache/results/baseline/01_1_ulip_modelnet_clean_zs/logs |

---

## 5. 运行命令

使用第一张 T4：

| 命令 |
|---|
| cd /root/autodl-tmp/MCM-PC-2/Point-Cache |
| bash scripts/baseline/01_1_ulip_modelnet_clean_zs_single_gpu.sh 0 |

使用第二张 T4：

| 命令 |
|---|
| cd /root/autodl-tmp/MCM-PC-2/Point-Cache |
| bash scripts/baseline/01_1_ulip_modelnet_clean_zs_single_gpu.sh 1 |

本次实际运行使用的是单张 T4。

---

## 6. 方法说明：Zero-shot

本实验使用 ULIP 原始 Zero-shot 推理。

Zero-shot 的流程可以简化理解为：

| 步骤 | 说明 |
|---|---|
| 1 | 使用 ULIP 点云编码器提取点云全局特征 |
| 2 | 使用文本编码器提取 40 个 ModelNet 类别的文本特征 |
| 3 | 计算点云特征和文本特征之间的相似度 |
| 4 | 选择相似度最高的类别作为预测类别 |
| 5 | 在整个测试集上统计 Top-1 accuracy |

本实验不构建缓存，也不会利用测试流中前面样本的信息。

因此，它是最原始的 clean baseline。

---

## 7. 模型与权重加载记录

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

## 8. 实验完成情况

| 项目 | 结果 |
|---|---:|
| 应运行数据文件 | 1 |
| 实际完成数据文件 | 1 |
| failed 数量 | 0 |
| failed_parse_acc 数量 | 0 |
| 最终状态 | 完成 |

---

## 9. 与原论文对齐结果

| 方法 | 当前复现 Acc | 原论文 Acc | Diff |
|---|---:|---:|---:|
| Zero-shot | 56.77 | 56.16 | +0.61 |

分析：

1. 当前复现的 Zero-shot 准确率为 56.77。
2. 原论文 Table 1 中 ULIP clean ModelNet Zero-shot 为 56.16。
3. 当前复现比原论文高 +0.61。
4. 这个差异很小，可以认为 clean Zero-shot 复现成功。

---

## 10. 运行过程中的 accuracy 解释

日志中会出现多次 accuracy，例如：

| 类型 | 含义 |
|---|---|
| 中间 accuracy | 测试过程中每隔一定样本数打印的累计平均准确率 |
| Final accuracy | 全部测试样本跑完后的最终准确率 |

本实验真正记录的是最后一行 Final accuracy：

| 指标 | 数值 |
|---|---:|
| Final Zero-shot test accuracy | 56.77 |

---

## 11. 与后续实验的关系

本实验是 01 组的基础对照。

| 后续实验 | 方法 | 比较方式 |
|---|---|---|
| 01_2_ulip_modelnet_clean_zs_global | ZS + Global Cache | Global - ZS |
| 01_3_ulip_modelnet_clean_zs_global_local | ZS + Global + Local Cache | Hier - ZS 和 Hier - Global |

在 clean ModelNet 上，后续两个实验的结果为：

| 方法 | Acc | 相比本实验提升 |
|---|---:|---:|
| Zero-shot | 56.77 | 0.00 |
| ZS + Global Cache | 62.12 | +5.35 |
| ZS + Global + Local Cache | 64.18 | +7.41 |

---

## 12. 关键观察

| 观察 | 说明 |
|---|---|
| Zero-shot 结果正常 | 当前复现 56.77，与原论文 56.16 接近 |
| 权重加载正常 | 文本模型和点云模型参数项数量均匹配 |
| clean 数据读取正常 | 实际读取 data/modelnet_c/clean.h5 |
| 可作为 sanity check | 后续 cache 和 corruption 实验均建立在此基础上 |
| 未使用测试流信息 | Zero-shot 对每个样本独立预测，不进行缓存更新 |

---

## 13. 阶段性结论

本实验可以记录为：

ULIP × ModelNet clean × Zero-shot 复现成功。

主要依据如下：

1. 实验成功运行完成。
2. 最终准确率为 56.77。
3. 与原论文 56.16 只差 +0.61。
4. 数据、权重、类别读取均正常。
5. 该结果可作为 clean ModelNet 上 Global Cache 和 Hierarchical Cache 的对照基线。

---

## 14. 后续记录

下一步应补充：

| 顺序 | 文档 |
|---:|---|
| 1 | 01_2_ulip_modelnet_clean_zs_global.md |
| 2 | 01_3_ulip_modelnet_clean_zs_global_local.md |

