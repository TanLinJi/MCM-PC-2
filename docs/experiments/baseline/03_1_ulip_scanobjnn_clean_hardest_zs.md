# 03_1_ulip_scanobjnn_clean_hardest_zs

## 1. 实验目的

本实验用于复现 ULIP 在 ScanObjNN clean hardest 上的 Zero-shot 结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 03_1_ulip_scanobjnn_clean_hardest_zs |
| Backbone | ULIP |
| Dataset | ScanObjNN clean hardest |
| Method | Zero-shot |
| 是否使用 Global Cache | 否 |
| 是否使用 Local Cache | 否 |

本实验的作用是获得 ULIP 在真实扫描点云 clean hardest split 上的无缓存基础性能。该结果后续会作为 03_2 和 03_3 的对照基线，但本文件只记录 03_1 本身，不展开整个 03 组的综合分析。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 方法 | Zero-shot |
| 方法简写 | zs |
| 原始核心 runner | runners/zs_infer.py |
| 方法脚本 | Point-Cache/scripts/baseline/03_1_ulip_scanobjnn_clean_hardest_zs_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/03_run_ulip_scanobjnn_clean_hardest_common.sh |
| 输入点数 | 1024 |
| 权重 | weights/ulip/pointbert_ulip1.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| GPU | 单张 Tesla T4 |

本实验使用 `sonn_c` 作为 dataset 参数，`sonn_variant=hardest`，并指定 `cor_type=clean`。实际读取文件为：

data/sonn_c/hardest/clean.h5

---

## 3. 输出结构

输出目录：

Point-Cache/results/baseline/03_1_ulip_scanobjnn_clean_hardest_zs/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

03_1_ulip_scanobjnn_clean_hardest_zs_clean_YYYYMMDD_HHMMSS.log

因为本实验只测试 `clean.h5`，所以期望只有 1 行 summary 记录和 1 个对应 log。

---

## 4. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 1 | 1 | clean 实验只包含 1 个测试文件 |
| summary 中唯一 cor_type 数 | 1 | 1 | 只测试 clean |
| summary 中唯一 log_path 数 | 1 | 1 | 每次运行对应 1 个 log |
| status=done 数 | 1 | 1 | 脚本执行成功 |
| 当前复现 accuracy | 29.08 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，03_1 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 5. 当前结果表

| 实验编号 | Dataset | Variant | File | Method | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 03_1_ulip_scanobjnn_clean_hardest_zs | sonn_c | hardest | data/sonn_c/hardest/clean.h5 | Zero-shot | 29.08 | done |

该结果表示：ULIP 在 ScanObjNN hardest clean 数据上的 Zero-shot 准确率为 29.08。

---

## 6. 与原文结果对比

原文中与本实验直接相关的参考值包括：

| 来源 | 原文值 | 说明 |
|---|---:|---|
| Figure 1(b) | 29.39 | ScanObjNN clean，ULIP Zero-shot |
| Supplementary Table 7 | 29.29 | S-PB T50-RS-C hardest split，ULIP Original Data |

当前复现结果为 29.08。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Figure 1(b) clean ScanObjNN | 29.39 | 29.08 | -0.31 | 0.31 |
| Table 7 Original Data / SONN | 29.29 | 29.08 | -0.21 | 0.21 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与 Table 7 的差异 | -0.21 |
| 与 Figure 1(b) 的差异 | -0.31 |
| 最大绝对差异 | 0.31 |
| 平均绝对差异 | 0.26 |

分析：

当前复现结果 29.08 与原文 Supplementary Table 7 的 29.29 相差 -0.21，与 Figure 1(b) 的 29.39 相差 -0.31。误差很小，可以认为 03_1 的 Zero-shot clean hardest 结果与原文基本对齐。

更严谨地说：03_1 不只是脚本跑通，而且数值也与原文 ScanObjNN hardest clean / Original Data 的参考结果基本一致。

---

## 7. 结果含义分析

03_1 的准确率为 29.08，说明 ULIP 在 ScanObjNN hardest clean 数据上的 Zero-shot 基础性能较低。

这与 ScanObjNN hardest 的数据特点有关：

| 可能原因 | 说明 |
|---|---|
| 真实扫描数据 | ScanObjNN 来自真实扫描，不是规则 CAD 模型 |
| 几何不完整 | 对象可能存在遮挡、缺失和扫描不完整 |
| 背景与噪声 | hardest split 中样本更复杂 |
| 类别混淆 | 真实物体局部结构更容易相似 |
| 无测试时自适应 | Zero-shot 不利用测试流分布信息 |

因此，本实验的意义不是追求高数值，而是建立 ULIP 在真实扫描 clean hardest 设置下的无缓存基线。

---

## 8. 与前序子实验的关系

在 03 组内部，03_1 是第一个子实验，因此没有前序 03 组子实验可供对比。

它后续将作为以下实验的基线：

| 后续实验 | 对比方式 |
|---|---|
| 03_2_ulip_scanobjnn_clean_hardest_zs_global | 与 03_1 比较，评估 Global Cache 增益 |
| 03_3_ulip_scanobjnn_clean_hardest_zs_global_local | 与 03_1 和 03_2 比较，评估完整 Point-Cache 及 Local Cache 额外增益 |

具体方法间增益不在本文件展开，应放在 03_2、03_3 子实验文档及 03 组 summary 文档中。

---

## 9. 阶段性结论

本实验完成了 ULIP × ScanObjNN clean hardest 的 Zero-shot baseline 复现。

主要结论如下：

1. 03_1 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 29.08。
3. 原文 Supplementary Table 7 中 ULIP 在 S-PB T50-RS-C hardest split 的 Original Data 上为 29.29。
4. 当前复现结果与 Table 7 的差异为 -0.21，与 Figure 1(b) 的差异为 -0.31。
5. 因此，03_1 不仅运行成功，而且数值与原文基本对齐。
6. 该实验是 03_2 和 03_3 的基础对照，不在本文件中展开完整 03 组方法间对比。

---

## 10. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/03_1_ulip_scanobjnn_clean_hardest_zs_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/03_1_ulip_scanobjnn_clean_hardest_zs_single_gpu.sh 1

---

## 11. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/03_1_ulip_scanobjnn_clean_hardest_zs/summary.csv | wc -l

tail -n +2 results/baseline/03_1_ulip_scanobjnn_clean_hardest_zs/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/03_1_ulip_scanobjnn_clean_hardest_zs/logs -maxdepth 1 -name '*.log' | wc -l

cat results/baseline/03_1_ulip_scanobjnn_clean_hardest_zs/summary.csv
