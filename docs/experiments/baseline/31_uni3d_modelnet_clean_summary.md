# 31_uni3d_modelnet_clean_summary

## 1. 实验组目的

本总文档汇总 Uni3D 在 ModelNet clean 上的三组 baseline 复现实验。

31 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | Uni3D |
| Dataset | ModelNet clean |
| 数据集参数 | modelnet_c |
| 实际数据文件 | data/modelnet_c/clean.h5 |
| 输入点数 | 1024 |
| Uni3D point encoder checkpoint | weights/uni3d/modelnet40/model.pt |
| Uni3D text encoder checkpoint | weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin |

三组实验分别为：

| 实验编号 | 方法 | 作用 |
|---|---|---|
| 31_1_uni3d_modelnet_clean_zs | Zero-shot | 无缓存基础对照 |
| 31_2_uni3d_modelnet_clean_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 31_3_uni3d_modelnet_clean_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 及 Local Cache 额外影响 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| Uni3D 在 ModelNet clean 上的 Zero-shot 基础性能是多少？ | 由 31_1 给出 |
| Global Cache 是否有效？ | 比较 31_2 - 31_1 |
| Local Cache 是否有额外贡献？ | 比较 31_3 - 31_2 |
| 三个结果是否与原文对齐？ | 分别与原文 Point-Cache Table 1 对比 |
| Uni3D checkpoint 是否会影响复现结果？ | 记录旧 checkpoint 偏低与 ModelNet40 checkpoint 对齐的特殊情况 |
| 后续 Uni3D 实验应如何设置 checkpoint？ | 31 / 32 使用 modelnet40 checkpoint，33 / 34 使用 scanobjnn checkpoint |

---

## 2. 当前实现方式

31 组是 clean 单文件实验，因此使用普通 bash 脚本执行，不使用 all35 优化 runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/31_run_uni3d_modelnet_clean_common.sh |
| 31_1 脚本 | Point-Cache/scripts/baseline/31_1_uni3d_modelnet_clean_zs_single_gpu.sh |
| 31_2 脚本 | Point-Cache/scripts/baseline/31_2_uni3d_modelnet_clean_zs_global_single_gpu.sh |
| 31_3 脚本 | Point-Cache/scripts/baseline/31_3_uni3d_modelnet_clean_zs_global_local_single_gpu.sh |
| checkpoint 下载脚本 | Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |

31 组每个子实验只测试一个文件：

data/modelnet_c/clean.h5

因此每个子实验的 summary.csv 应只有 1 行，每个 logs 目录应只有 1 个 log 文件。

---

## 3. Uni3D checkpoint 特殊情况记录

本组实验出现了一个重要复现问题：最初使用服务器原有的 Uni3D checkpoint 时，31 组结果整体偏低。

最初使用的 checkpoint 为：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

该 checkpoint 可以正常加载，脚本也可以正常运行，31_3 中也确实构建了 global cache 和 local cache：

| 检查项 | 结果 |
|---|---|
| 是否进入 hierarchical runner | 是 |
| 是否构建 global cache | 是 |
| 是否构建 local cache | 是 |
| pos_cache 数量 | 40 |
| pos_local_cache 数量 | 40 |

但是旧 checkpoint 下 31 组结果明显低于原文：

| 方法 | 旧 checkpoint 结果 | 原文结果 | Diff |
|---|---:|---:|---:|
| Zero-shot | 80.11 | 81.81 | -1.70 |
| ZS + Global | 81.56 | 83.14 | -1.58 |
| ZS + Global + Local | 81.60 | 83.87 | -2.27 |

随后下载 Uni3D-g 的 ModelNet40 checkpoint：

weights/uni3d/modelnet40/model.pt

重新运行 31 组后，结果与原文高度对齐：

| 方法 | ModelNet40 checkpoint 结果 | 原文结果 | Diff |
|---|---:|---:|---:|
| Zero-shot | 81.85 | 81.81 | +0.04 |
| ZS + Global | 83.23 | 83.14 | +0.09 |
| ZS + Global + Local | 83.71 | 83.87 | -0.16 |

因此，本次复现实验确认：**Uni3D checkpoint 与数据集设置是否匹配，会直接影响结果是否对齐原文。**

后续规则：

| 实验组 | 数据设置 | 应使用 checkpoint |
|---|---|---|
| 31 | Uni3D × ModelNet clean | weights/uni3d/modelnet40/model.pt |
| 32 | Uni3D × ModelNet-C all35 | weights/uni3d/modelnet40/model.pt |
| 33 | Uni3D × ScanObjNN clean hardest | weights/uni3d/scanobjnn/model.pt |
| 34 | Uni3D × ScanObjNN-C hardest all35 | weights/uni3d/scanobjnn/model.pt |

`weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt` 可以作为额外 ablation 或参考 checkpoint，但不应作为正式 baseline 复现 checkpoint。

该特殊情况也提示：此前 Uni3D 相关实验若出现稳定偏低或 Local Cache 增益异常，应优先回查 checkpoint 路径，而不是先怀疑 runner 或 cache 逻辑。

---

## 4. Uni3D checkpoint 下载脚本

本次新增 checkpoint 下载脚本：

Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh

该脚本用于下载 Uni3D-g 相关官方 checkpoints，包括：

| 文件 | 用途 |
|---|---|
| weights/uni3d/modelnet40/model.pt | ModelNet / ModelNet-C |
| weights/uni3d/scanobjnn/model.pt | ScanObjNN / ScanObjNN-C hardest |
| weights/uni3d/model.pt | Uni3D-g 通用权重，备用 |
| weights/uni3d/lvis/model.pt | LVIS / OmniObject3D 等开放类别设置备用 |

脚本默认使用 Hugging Face 镜像：

https://hf-mirror.com

如果服务器可直接访问 Hugging Face，可以改用官方 endpoint。

需要注意：这些 checkpoint 文件体积很大，不应提交到 Git。

---

## 5. 三个实验的输出完整性检查

| 实验编号 | 方法 | summary 行数 | cor_type 唯一数 | log_path 唯一数 | logs 文件数 | status | 状态 |
|---|---|---:|---:|---:|---:|---|---|
| 31_1_uni3d_modelnet_clean_zs | Zero-shot | 1 | 1 | 1 | 1 | done | 完成 |
| 31_2_uni3d_modelnet_clean_zs_global | ZS + Global | 1 | 1 | 1 | 1 | done | 完成 |
| 31_3_uni3d_modelnet_clean_zs_global_local | ZS + Global + Local | 1 | 1 | 1 | 1 | done | 完成 |

说明：

1. 31 组每个子实验都只对应 `data/modelnet_c/clean.h5` 一个测试文件，因此 summary 行数应为 1。
2. 三个子实验均为 `status=done`。
3. 每个子实验都有唯一 log_path，说明结果和日志可以一一对应。
4. 每个 logs 目录均只有 1 个 log 文件，没有旧日志或重复日志残留。
5. 执行完整性正常并不等于结果正常；结果是否正常还需要与原文参考值对比。

---

## 6. 核心结果总表

| 实验编号 | 方法 | 当前复现值 | 原文参考值 | Diff = 当前 - 原文 | 是否对齐 |
|---|---|---:|---:|---:|---|
| 31_1_uni3d_modelnet_clean_zs | Zero-shot | 81.85 | 81.81 | +0.04 | 高度对齐 |
| 31_2_uni3d_modelnet_clean_zs_global | ZS + Global Cache | 83.23 | 83.14 | +0.09 | 高度对齐 |
| 31_3_uni3d_modelnet_clean_zs_global_local | ZS + Global + Local Cache | 83.71 | 83.87 | -0.16 | 高度接近 |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | -0.01 |
| MAE | 0.10 |
| Max Abs Diff | 0.16 |

分析：

31 组三个子实验均完成，并且三种方法的绝对结果整体与原文 Point-Cache Table 1 高度对齐。

其中 31_1 和 31_2 几乎完全对齐原文，差异分别只有 +0.04 和 +0.09。31_3 比原文低 -0.16，仍然属于很小差异。

因此，31 组可以认为是有效复现结果，不需要重跑。

---

## 7. 方法间变化分析

### 7.1 当前复现变化

| 比较 | 当前复现值 | 变化 |
|---|---:|---:|
| 31_1 Zero-shot | 81.85 | — |
| 31_2 ZS + Global | 83.23 | +1.38 over 31_1 |
| 31_3 ZS + Global + Local | 83.71 | +0.48 over 31_2 |
| 31_3 ZS + Global + Local | 83.71 | +1.86 over 31_1 |

### 7.2 原文变化

| 比较 | 原文值 | 变化 |
|---|---:|---:|
| Zero-shot | 81.81 | — |
| + Global Cache | 83.14 | +1.33 over Zero-shot |
| + Hierarchical Cache | 83.87 | +0.73 over Global Cache |
| + Hierarchical Cache | 83.87 | +2.06 over Zero-shot |

### 7.3 当前变化与原文变化对比

| 变化来源 | 原文变化 | 当前复现变化 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | +1.33 | +1.38 | +0.05 |
| Local Cache extra over Global | +0.73 | +0.48 | -0.25 |
| Full Point-Cache over Zero-shot | +2.06 | +1.86 | -0.20 |

分析：

当前复现中的方法趋势为：

Zero-shot < ZS + Global Cache < ZS + Global Cache + Local Cache

这个趋势与原文一致。

Global Cache 的相对增益与原文高度一致：原文 +1.33，当前 +1.38。Local Cache 的额外增益略弱：原文 +0.73，当前 +0.48，但方向正确，最终完整方法与原文只差 -0.16。

因此，31 组不仅绝对数值对齐，方法间增益结构也基本对齐。

---

## 8. 方法贡献分解

以当前复现结果为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

83.71 - 81.85 = +1.86

其中：

| 贡献来源 | Accuracy 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +1.38 | 约 74.2% |
| Local Cache | +0.48 | 约 25.8% |
| 完整 Point-Cache | +1.86 | 100.00% |

以原文结果为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

83.87 - 81.81 = +2.06

其中：

| 贡献来源 | Accuracy 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +1.33 | 约 64.6% |
| Local Cache | +0.73 | 约 35.4% |
| 完整 Point-Cache | +2.06 | 100.00% |

分析：

当前复现中，Global Cache 是主要提升来源，Local Cache 有额外正贡献但幅度较小。该结构与原文一致，只是当前 Local Cache 占比略低。

这与 21 组 OpenShape × ModelNet clean 不同。OpenShape 在 ModelNet clean 上使用 cache 后略降，而 Uni3D 在 ModelNet clean 上使用 Global Cache 和 Local Cache 后均有正增益。说明不同 backbone 对 cache 的响应不同。

---

## 9. 与旧 checkpoint 的完整对比

| 方法 | 旧 pc_encoder checkpoint | ModelNet40 checkpoint | 原文 | 新旧差值 |
|---|---:|---:|---:|---:|
| Zero-shot | 80.11 | 81.85 | 81.81 | +1.74 |
| ZS + Global | 81.56 | 83.23 | 83.14 | +1.67 |
| ZS + Global + Local | 81.60 | 83.71 | 83.87 | +2.11 |

旧 checkpoint：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

正式 checkpoint：

weights/uni3d/modelnet40/model.pt

分析：

旧 checkpoint 下三个实验整体低于原文 1.5 到 2.3 个百分点。切换到 ModelNet40 checkpoint 后，三个实验全部对齐原文。

因此，本次实验可以明确记录为：31 组最初偏低并非脚本未运行、cache 未构建或随机波动，而是 Uni3D point encoder checkpoint 与 ModelNet40 数据设置不匹配造成的。

这个特殊情况应作为后续 Uni3D 实验的重要经验：**Uni3D baseline 必须记录并核对 checkpoint 路径。**

---

## 10. 与 01 / 11 / 21 组 ModelNet clean 的关系

31 组可以与前面 ULIP、ULIP-2、OpenShape 的 ModelNet clean 结果进行横向比较。

| Backbone | Zero-shot | ZS + Global | ZS + Global + Local |
|---|---:|---:|---:|
| ULIP | 约 56 | 约 62 | 约 64 |
| ULIP-2 | 72.20 | 73.99 | 74.35 |
| OpenShape | 84.72 | 84.48 | 84.00 |
| Uni3D | 81.85 | 83.23 | 83.71 |

分析：

Uni3D 在 ModelNet clean 上明显强于 ULIP 和 ULIP-2，略低于 OpenShape。完整 Point-Cache 下，Uni3D 为 83.71，OpenShape 为 84.00，二者非常接近。

不同 backbone 对 cache 的响应不同：

| Backbone | clean 上 cache 现象 |
|---|---|
| OpenShape | clean 上 cache 略降 |
| Uni3D | clean 上 Global 和 Local 均有正增益 |
| ULIP / ULIP-2 | cache 通常有明显正增益 |

这说明 cache 在 clean setting 上是否提升，取决于 backbone 的表征特性和 zero-shot 起点，不能一概而论。

---

## 11. 对后续 32 组的意义

31 组是 Uni3D × ModelNet clean，后续 32 组将进入：

Uni3D × ModelNet-C all35

31 组为 32 组提供 clean reference：

| 后续比较 | 目的 |
|---|---|
| 32_1 vs 31_1 | 观察 ModelNet-C corruption 相比 clean 对 Uni3D Zero-shot 的影响 |
| 32_2 vs 31_2 | 观察 corruption 相比 clean 对 Global Cache 的影响 |
| 32_3 vs 31_3 | 观察 corruption 相比 clean 对完整 Point-Cache 的影响 |
| 32_2 - 32_1 | 评估 Global Cache 在 Uni3D × ModelNet-C 上是否提升鲁棒性 |
| 32_3 - 32_2 | 评估 Local Cache 在 Uni3D × ModelNet-C 上是否继续有额外贡献 |

后续 32 组必须继续使用：

weights/uni3d/modelnet40/model.pt

不能再使用：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

否则可能再次出现稳定偏低，并导致与原文不对齐。

---

## 12. 对后续 33 / 34 组的意义

31 / 32 组属于 ModelNet 系列，因此使用 ModelNet40 checkpoint。33 / 34 组属于 ScanObjNN 系列，因此应使用：

weights/uni3d/scanobjnn/model.pt

具体规则：

| 组别 | 数据设置 | checkpoint |
|---|---|---|
| 31 | Uni3D × ModelNet clean | weights/uni3d/modelnet40/model.pt |
| 32 | Uni3D × ModelNet-C all35 | weights/uni3d/modelnet40/model.pt |
| 33 | Uni3D × ScanObjNN clean hardest | weights/uni3d/scanobjnn/model.pt |
| 34 | Uni3D × ScanObjNN-C hardest all35 | weights/uni3d/scanobjnn/model.pt |

这是后续 D 组 Uni3D 实验必须遵守的设置。

---

## 13. 对后续 MCM-PC 的启发

当前 31 组实验对后续方法设计和实验管理有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| Uni3D 对 checkpoint 选择敏感 | checkpoint 路径必须写进文档和脚本 |
| 旧 checkpoint 下结果稳定偏低 | 不能只看脚本成功就判定复现正常 |
| ModelNet40 checkpoint 下结果高度对齐 | 数据集对应 checkpoint 是正式复现前提 |
| Global Cache 在 Uni3D clean 上提升 +1.38 | 全局缓存对 Uni3D 有稳定正增益 |
| Local Cache 额外提升 +0.48 | 局部缓存有额外贡献，但弱于 Global |
| OpenShape clean 上 cache 略降，而 Uni3D clean 上 cache 提升 | backbone 对 cache 的响应具有差异 |
| 31 组是 32 组 clean reference | 后续 ModelNet-C 分析必须基于正确 clean 结果 |

这对 MCM-PC 很重要：后续如果使用 Uni3D 作为 backbone，必须先确认 checkpoint 是否与数据集一致，否则实验结论可能被错误 checkpoint 干扰。

---

## 14. 阶段性结论

31 组 Uni3D × ModelNet clean baseline 已完成。

主要结论如下：

1. 三个子实验均完成，summary.csv 行数均为 1，status 均为 done。
2. 当前正式结果均使用 weights/uni3d/modelnet40/model.pt。
3. 31_1 Zero-shot 当前复现值为 81.85，原文参考值为 81.81，差异 +0.04。
4. 31_2 ZS + Global 当前复现值为 83.23，原文参考值为 83.14，差异 +0.09。
5. 31_3 ZS + Global + Local 当前复现值为 83.71，原文参考值为 83.87，差异 -0.16。
6. 三个结果整体与原文高度对齐，31 组可以作为有效复现结果归档。
7. 当前方法趋势为 Zero-shot < Global Cache < Global + Local Cache，与原文一致。
8. Global Cache 相比 Zero-shot 当前提升 +1.38，与原文 +1.33 高度一致。
9. Local Cache 在 Global Cache 基础上当前额外提升 +0.48，略低于原文 +0.73，但方向正确。
10. 完整 Point-Cache 相比 Zero-shot 当前提升 +1.86，接近原文 +2.06。
11. 旧 checkpoint weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt 下结果整体偏低，31_3 曾低于原文 -2.27。
12. 切换到 ModelNet40 checkpoint 后，31 组三个结果全部对齐原文。
13. 该特殊情况说明 Uni3D checkpoint 选择是复现实验的关键变量。
14. 31 / 32 组后续应统一使用 weights/uni3d/modelnet40/model.pt。
15. 33 / 34 组后续应统一使用 weights/uni3d/scanobjnn/model.pt。
16. checkpoint 下载脚本已记录在 Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh。

---

## 15. 运行命令汇总

31_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/31_1_uni3d_modelnet_clean_zs_single_gpu.sh 0

31_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/31_2_uni3d_modelnet_clean_zs_global_single_gpu.sh 0

31_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/31_3_uni3d_modelnet_clean_zs_global_local_single_gpu.sh 1

---

## 16. 检查命令汇总

31_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/31_1_uni3d_modelnet_clean_zs/summary.csv | wc -l

tail -n +2 results/baseline/31_1_uni3d_modelnet_clean_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/31_1_uni3d_modelnet_clean_zs/logs -maxdepth 1 -name '*.log' | wc -l

31_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/31_2_uni3d_modelnet_clean_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/31_2_uni3d_modelnet_clean_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/31_2_uni3d_modelnet_clean_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

31_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/31_3_uni3d_modelnet_clean_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/31_3_uni3d_modelnet_clean_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/31_3_uni3d_modelnet_clean_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
