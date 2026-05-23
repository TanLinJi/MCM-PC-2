# 33_uni3d_scanobjnn_clean_hardest_summary

## 1. 实验组目的

本总文档汇总 Uni3D 在 ScanObjNN clean hardest split 上的三组 baseline 复现实验。

33 组实验固定：

| 维度 | 内容 |
|---|---|
| Backbone | Uni3D |
| Dataset | ScanObjNN clean hardest |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 输入点数 | 1024 |
| Uni3D point encoder checkpoint | weights/uni3d/scanobjnn/model.pt |
| Uni3D text encoder checkpoint | weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin |

三组实验分别为：

| 实验编号 | 方法 | 作用 |
|---|---|---|
| 33_1_uni3d_scanobjnn_clean_hardest_zs | Zero-shot | 无缓存基础对照 |
| 33_2_uni3d_scanobjnn_clean_hardest_zs_global | Zero-shot + Global Cache | 验证全局缓存增益 |
| 33_3_uni3d_scanobjnn_clean_hardest_zs_global_local | Zero-shot + Global Cache + Local Cache | 验证完整 Point-Cache 及 Local Cache 额外影响 |

本实验组的核心目标是回答：

| 问题 | 说明 |
|---|---|
| Uni3D 在 ScanObjNN clean hardest 上的 Zero-shot 基础性能是多少？ | 由 33_1 给出 |
| Global Cache 是否有效？ | 比较 33_2 - 33_1 |
| Local Cache 是否有额外贡献？ | 比较 33_3 - 33_2 |
| 三个结果是否与原文对齐？ | 分别与原文 Point-Cache Table 7 对比 |
| Uni3D checkpoint 是否会影响复现结果？ | 记录旧 checkpoint 偏低与 ScanObjNN checkpoint 对齐的特殊情况 |
| 后续 34 组应如何设置 checkpoint？ | 34 组继续使用 weights/uni3d/scanobjnn/model.pt |

---

## 2. 当前实现方式

33 组是 clean 单文件实验，因此使用普通 bash 脚本执行，不使用 all35 优化 runner。

| 项目 | 路径或名称 |
|---|---|
| 公共脚本 | Point-Cache/scripts/baseline/33_run_uni3d_scanobjnn_clean_hardest_common.sh |
| 33_1 脚本 | Point-Cache/scripts/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs_single_gpu.sh |
| 33_2 脚本 | Point-Cache/scripts/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global_single_gpu.sh |
| 33_3 脚本 | Point-Cache/scripts/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh |
| checkpoint 下载脚本 | Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh |
| 结果根目录 | Point-Cache/results/baseline/ |
| 文档目录 | docs/experiments/baseline/ |

33 组每个子实验只测试一个文件：

data/sonn_c/hardest/clean.h5

因此每个子实验的 summary.csv 应只有 1 行，每个 logs 目录应只有 1 个 log 文件。

---

## 3. Uni3D checkpoint 特殊情况记录

本组实验出现了一个重要复现问题：最初 33 组公共脚本没有优先使用 ScanObjNN 对应 checkpoint，导致结果整体明显偏低。

错误运行时，公共脚本的 checkpoint 候选列表大致为：

| 候选顺序 | checkpoint |
|---:|---|
| 1 | weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt |
| 2 | weights/uni3d/model.pt |
| 3 | weights/uni3d/uni3d_g_ensembled_model.pt |

由于服务器中 `weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt` 存在，脚本自动优先选择了该旧 checkpoint，而不是 ScanObjNN 对应 checkpoint。

错误 checkpoint 下的结果为：

| 方法 | 旧 checkpoint 结果 | 原文结果 | Diff |
|---|---:|---:|---:|
| Zero-shot | 41.33 | 46.04 | -4.71 |
| ZS + Global | 43.79 | 50.28 | -6.49 |
| ZS + Global + Local | 45.49 | 51.13 | -5.64 |

这说明旧 checkpoint 虽然可以正常加载、脚本也能跑通，但不适合作为 Uni3D × ScanObjNN clean hardest 的正式复现 checkpoint。

修正后，将 ScanObjNN checkpoint 加入候选列表第一位：

weights/uni3d/scanobjnn/model.pt

修正后 checkpoint 候选逻辑应优先使用：

| 实验组 | 数据设置 | 应使用 checkpoint |
|---|---|---|
| 31 | Uni3D × ModelNet clean | weights/uni3d/modelnet40/model.pt |
| 32 | Uni3D × ModelNet-C all35 | weights/uni3d/modelnet40/model.pt |
| 33 | Uni3D × ScanObjNN clean hardest | weights/uni3d/scanobjnn/model.pt |
| 34 | Uni3D × ScanObjNN-C hardest all35 | weights/uni3d/scanobjnn/model.pt |

修正后重新运行 33 组，结果恢复正常：

| 方法 | ScanObjNN checkpoint 结果 | 原文结果 | Diff |
|---|---:|---:|---:|
| Zero-shot | 45.63 | 46.04 | -0.41 |
| ZS + Global | 50.03 | 50.28 | -0.25 |
| ZS + Global + Local | 51.98 | 51.13 | +0.85 |

因此，本次复现实验再次确认：**Uni3D checkpoint 与数据集设置是否匹配，会直接影响结果是否对齐原文。**

`weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt` 可以作为额外 ablation 或参考 checkpoint，但不应作为正式 baseline 复现 checkpoint。

---

## 4. Uni3D checkpoint 下载脚本

checkpoint 下载脚本记录在：

Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh

该脚本用于下载 Uni3D-g 相关官方 checkpoints，包括：

| 文件 | 用途 |
|---|---|
| weights/uni3d/modelnet40/model.pt | ModelNet / ModelNet-C |
| weights/uni3d/scanobjnn/model.pt | ScanObjNN / ScanObjNN-C hardest |
| weights/uni3d/model.pt | Uni3D-g 通用权重，备用 |
| weights/uni3d/lvis/model.pt | LVIS / OmniObject3D 等开放类别设置备用 |

本次 33 组正式使用：

weights/uni3d/scanobjnn/model.pt

这些 checkpoint 文件体积很大，不应提交到 Git。

---

## 5. 三个实验的输出完整性检查

| 实验编号 | 方法 | summary 行数 | cor_type 唯一数 | log_path 唯一数 | logs 文件数 | status | 状态 |
|---|---|---:|---:|---:|---:|---|---|
| 33_1_uni3d_scanobjnn_clean_hardest_zs | Zero-shot | 1 | 1 | 1 | 1 | done | 完成 |
| 33_2_uni3d_scanobjnn_clean_hardest_zs_global | ZS + Global | 1 | 1 | 1 | 1 | done | 完成 |
| 33_3_uni3d_scanobjnn_clean_hardest_zs_global_local | ZS + Global + Local | 1 | 1 | 1 | 1 | done | 完成 |

说明：

1. 33 组每个子实验都只对应 `data/sonn_c/hardest/clean.h5` 一个测试文件，因此 summary 行数应为 1。
2. 三个子实验均为 `status=done`。
3. 每个子实验都有唯一 log_path，说明结果和日志可以一一对应。
4. 每个 logs 目录均只有 1 个 log 文件，没有旧日志或重复日志残留。
5. 日志中已确认实际 checkpoint 为 `weights/uni3d/scanobjnn/model.pt`。
6. 执行完整性正常并不等于结果正常；结果是否正常还需要与原文参考值对比。

---

## 6. 核心结果总表

| 实验编号 | 方法 | 当前复现值 | 原文参考值 | Diff = 当前 - 原文 | 是否对齐 |
|---|---|---:|---:|---:|---|
| 33_1_uni3d_scanobjnn_clean_hardest_zs | Zero-shot | 45.63 | 46.04 | -0.41 | 接近 |
| 33_2_uni3d_scanobjnn_clean_hardest_zs_global | ZS + Global Cache | 50.03 | 50.28 | -0.25 | 高度接近 |
| 33_3_uni3d_scanobjnn_clean_hardest_zs_global_local | ZS + Global + Local Cache | 51.98 | 51.13 | +0.85 | 略高但可接受 |

补充统计：

| 指标 | 数值 |
|---|---:|
| Mean Diff | +0.06 |
| MAE | 0.50 |
| Max Abs Diff | 0.85 |

分析：

33 组三个子实验均完成，并且三种方法的绝对结果整体与原文 Point-Cache Table 7 接近。

其中 33_1 和 33_2 与原文非常接近，差异分别为 -0.41 和 -0.25。33_3 比原文高 +0.85，略高但仍可接受。

因此，33 组可以认为是有效复现结果，不需要重跑。

---

## 7. 方法间变化分析

### 7.1 当前复现变化

| 比较 | 当前复现值 | 变化 |
|---|---:|---:|
| 33_1 Zero-shot | 45.63 | — |
| 33_2 ZS + Global | 50.03 | +4.40 over 33_1 |
| 33_3 ZS + Global + Local | 51.98 | +1.95 over 33_2 |
| 33_3 ZS + Global + Local | 51.98 | +6.35 over 33_1 |

### 7.2 原文变化

| 比较 | 原文值 | 变化 |
|---|---:|---:|
| Zero-shot | 46.04 | — |
| + Global Cache | 50.28 | +4.24 over Zero-shot |
| + Hierarchical Cache | 51.13 | +0.85 over Global Cache |
| + Hierarchical Cache | 51.13 | +5.09 over Zero-shot |

### 7.3 当前变化与原文变化对比

| 变化来源 | 原文变化 | 当前复现变化 | Diff |
|---|---:|---:|---:|
| Global Cache over Zero-shot | +4.24 | +4.40 | +0.16 |
| Local Cache extra over Global | +0.85 | +1.95 | +1.10 |
| Full Point-Cache over Zero-shot | +5.09 | +6.35 | +1.26 |

分析：

当前复现中的方法趋势为：

Zero-shot < ZS + Global Cache < ZS + Global Cache + Local Cache

这个趋势与原文一致。

Global Cache 的相对增益与原文高度一致：原文 +4.24，当前 +4.40。Local Cache 的额外增益当前更强：原文 +0.85，当前 +1.95。也正是因为 Local Cache 额外增益更强，33_3 最终结果比原文高 +0.85。

因此，33 组不仅绝对数值基本对齐，方法间增益结构也合理；只是 33_3 的 Local extra 相比原文偏高，需要在文档中明确记录。

---

## 8. 方法贡献分解

以当前复现结果为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

51.98 - 45.63 = +6.35

其中：

| 贡献来源 | Accuracy 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +4.40 | 约 69.3% |
| Local Cache | +1.95 | 约 30.7% |
| 完整 Point-Cache | +6.35 | 100.00% |

以原文结果为准，完整 Point-Cache 相比 Zero-shot 的总提升为：

51.13 - 46.04 = +5.09

其中：

| 贡献来源 | Accuracy 提升 | 占完整提升比例 |
|---|---:|---:|
| Global Cache | +4.24 | 约 83.3% |
| Local Cache | +0.85 | 约 16.7% |
| 完整 Point-Cache | +5.09 | 100.00% |

分析：

当前复现中，Global Cache 仍然是主要提升来源，但 Local Cache 的额外贡献也很明显，占完整提升约 30.7%。

与原文相比，当前 Local Cache 占比更高。这导致 33_3 略高于原文，但从方法趋势和最终结果看，仍属于有效复现。

---

## 9. 与旧 checkpoint 的完整对比

| 方法 | 旧 pc_encoder checkpoint | ScanObjNN checkpoint | 原文 | 新旧差值 |
|---|---:|---:|---:|---:|
| Zero-shot | 41.33 | 45.63 | 46.04 | +4.30 |
| ZS + Global | 43.79 | 50.03 | 50.28 | +6.24 |
| ZS + Global + Local | 45.49 | 51.98 | 51.13 | +6.49 |

旧 checkpoint：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

正式 checkpoint：

weights/uni3d/scanobjnn/model.pt

分析：

旧 checkpoint 下三个实验整体低于原文 4.7 到 6.5 个百分点。切换到 ScanObjNN checkpoint 后，三个实验整体恢复到原文附近。

因此，本次实验可以明确记录为：33 组最初偏低并非脚本未运行、cache 未构建或随机波动，而是 Uni3D point encoder checkpoint 与 ScanObjNN 数据设置不匹配造成的。

这个特殊情况应作为后续 Uni3D 实验的重要经验：**Uni3D baseline 必须记录并核对 checkpoint 路径。**

---

## 10. 与 03 / 13 / 23 组 ScanObjNN clean hardest 的关系

33 组可以与前面 ULIP、ULIP-2、OpenShape 的 ScanObjNN clean hardest 结果进行横向比较。

| Backbone | Zero-shot | ZS + Global | ZS + Global + Local |
|---|---:|---:|---:|
| ULIP | 29.08 | 32.20 | 32.48 |
| ULIP-2 | 33.31 | 39.38 | 待记录 |
| OpenShape | 41.88 | 41.95 | 43.82 |
| Uni3D | 45.63 | 50.03 | 51.98 |

分析：

Uni3D 在 ScanObjNN clean hardest 上明显强于 ULIP、ULIP-2 和 OpenShape。尤其在完整 Point-Cache 下，Uni3D 为 51.98，OpenShape 为 43.82，差距为 +8.16。

这说明 Uni3D 的 ScanObjNN 对应 checkpoint 对真实扫描 hardest split 有明显优势。与 ModelNet clean 上 OpenShape 略高于 Uni3D 的情况不同，ScanObjNN clean hardest 上 Uni3D 是当前最强 backbone。

---

## 11. 与 31 组 ModelNet clean 的关系

31 组是 Uni3D 在 ModelNet clean 上的结果；33 组是 Uni3D 在 ScanObjNN clean hardest 上的结果。

| 方法 | 31 组 ModelNet clean | 33 组 ScanObjNN clean hardest | 变化 |
|---|---:|---:|---:|
| Zero-shot | 81.85 | 45.63 | -36.22 |
| ZS + Global | 83.23 | 50.03 | -33.20 |
| ZS + Global + Local | 83.71 | 51.98 | -31.73 |

分析：

ScanObjNN clean hardest 远难于 ModelNet clean。即使使用完整 Point-Cache，Uni3D 在 ModelNet clean 上为 83.71，但在 ScanObjNN clean hardest 上只有 51.98。

这说明真实扫描 hardest split 与 CAD clean setting 之间存在巨大难度差异。后续 34 组 ScanObjNN-C hardest 会在此基础上进一步叠加 corruption，因此是更关键的困难设置。

---

## 12. 对后续 34 组的意义

34 组将进入：

Uni3D × ScanObjNN-C hardest all35

33 组为 34 组提供 clean reference：

| 后续比较 | 目的 |
|---|---|
| 34_1 vs 33_1 | 观察 ScanObjNN-C corruption 相比 clean hardest 对 Uni3D Zero-shot 的影响 |
| 34_2 vs 33_2 | 观察 corruption 相比 clean hardest 对 Global Cache 的影响 |
| 34_3 vs 33_3 | 观察 corruption 相比 clean hardest 对完整 Point-Cache 的影响 |
| 34_2 - 34_1 | 评估 Global Cache 在 Uni3D × ScanObjNN-C hardest 上是否提升鲁棒性 |
| 34_3 - 34_2 | 评估 Local Cache 在 Uni3D × ScanObjNN-C hardest 上是否继续有额外贡献 |

后续 34 组必须继续使用：

weights/uni3d/scanobjnn/model.pt

不能使用：

weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

也不能使用：

weights/uni3d/modelnet40/model.pt

否则可能再次出现稳定偏低或与原文不对齐。

---

## 13. 当前结果意义分析

33 组结果说明：

| 观察 | 解释 |
|---|---|
| Zero-shot = 45.63 | Uni3D 在 ScanObjNN clean hardest 上的基础性能 |
| ZS + Global = 50.03 | Global Cache 有明确正增益 |
| ZS + Global + Local = 51.98 | 完整 Point-Cache 最好 |
| Global extra = +4.40 | Global Cache 是主要提升来源 |
| Local extra = +1.95 | Local Cache 有明显额外贡献 |
| 33_1 / 33_2 与原文接近 | 脚本和 checkpoint 设置基本正确 |
| 33_3 高于原文 +0.85 | 完整方法略高，但可接受 |
| 旧 checkpoint 明显偏低 | checkpoint 是关键复现变量 |

33 组是一个非常关键的实验组。它说明 Uni3D 在真实扫描 clean hardest 上明显强于前面 backbone，同时也说明 Point-Cache 的 Global Cache 和 Local Cache 均有贡献。

---

## 14. 对后续 MCM-PC 的启发

当前 33 组实验对后续方法设计和实验管理有以下启发：

| 观察 | 对 MCM-PC 的启发 |
|---|---|
| Uni3D 对 checkpoint 选择极其敏感 | checkpoint 路径必须写进脚本和文档 |
| ScanObjNN 系列必须用 scanobjnn checkpoint | 33 / 34 组统一使用 weights/uni3d/scanobjnn/model.pt |
| 旧 pc_encoder checkpoint 会导致明显偏低 | 不应作为正式 baseline checkpoint |
| Global Cache 提供 +4.40 | 全局缓存是稳定主模块 |
| Local Cache 提供 +1.95 | 局部缓存也有较强价值 |
| Uni3D 在 ScanObjNN clean hardest 上强于 OpenShape | 说明 backbone 与数据集适配性非常重要 |
| 33 组是 34 组 clean reference | 后续 ScanObjNN-C 分析必须基于正确 clean 结果 |

这对 MCM-PC 很重要：后续如果使用 Uni3D 作为 backbone，必须先确认 checkpoint 是否与数据集一致，否则实验结论可能被错误 checkpoint 干扰。

---

## 15. 阶段性结论

33 组 Uni3D × ScanObjNN clean hardest baseline 已完成。

主要结论如下：

1. 三个子实验均完成，summary.csv 行数均为 1，status 均为 done。
2. 当前正式结果均使用 weights/uni3d/scanobjnn/model.pt。
3. 33_1 Zero-shot 当前复现值为 45.63，原文参考值为 46.04，差异 -0.41。
4. 33_2 ZS + Global 当前复现值为 50.03，原文参考值为 50.28，差异 -0.25。
5. 33_3 ZS + Global + Local 当前复现值为 51.98，原文参考值为 51.13，差异 +0.85。
6. 三个结果整体与原文接近，33 组可以作为有效复现结果归档。
7. 当前方法趋势为 Zero-shot < Global Cache < Global + Local Cache，与原文一致。
8. Global Cache 相比 Zero-shot 当前提升 +4.40，与原文 +4.24 高度一致。
9. Local Cache 在 Global Cache 基础上当前额外提升 +1.95，高于原文 +0.85。
10. 完整 Point-Cache 相比 Zero-shot 当前提升 +6.35，高于原文 +5.09。
11. 旧 checkpoint weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt 下结果整体偏低，33_2 曾低于原文 -6.49。
12. 切换到 ScanObjNN checkpoint 后，33 组三个结果全部恢复到原文附近。
13. 该特殊情况说明 Uni3D checkpoint 选择是复现实验的关键变量。
14. 33 / 34 组后续应统一使用 weights/uni3d/scanobjnn/model.pt。
15. 31 / 32 组应继续使用 weights/uni3d/modelnet40/model.pt。
16. checkpoint 下载脚本已记录在 Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh。
17. 33 组可作为后续 34 组 Uni3D × ScanObjNN-C hardest all35 的 clean reference。

---

## 16. 运行命令汇总

33_1 Zero-shot：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs_single_gpu.sh 0

33_2 Zero-shot + Global Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global_single_gpu.sh 0

33_3 Zero-shot + Global Cache + Local Cache：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh 1

---

## 17. 检查命令汇总

33_1：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs/summary.csv | wc -l

tail -n +2 results/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/33_1_uni3d_scanobjnn_clean_hardest_zs/logs -maxdepth 1 -name '*.log' | wc -l

33_2：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

33_3：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local/summary.csv | wc -l

tail -n +2 results/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local/logs -maxdepth 1 -name '*.log' | wc -l
