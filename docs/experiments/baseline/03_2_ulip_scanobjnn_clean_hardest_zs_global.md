# 03_2_ulip_scanobjnn_clean_hardest_zs_global

## 1. 实验目的

本实验用于复现 ULIP 在 ScanObjNN clean hardest 上使用 Zero-shot + Global Cache 的结果。

本实验只关注一个子实验：

| 项目 | 内容 |
|---|---|
| 实验编号 | 03_2_ulip_scanobjnn_clean_hardest_zs_global |
| Backbone | ULIP |
| Dataset | ScanObjNN clean hardest |
| Method | Zero-shot + Global Cache |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |

本实验是在 03_1 Zero-shot 的基础上加入 Global Cache，用于验证全局缓存是否能够提升 ULIP 在 ScanObjNN clean hardest 上的识别性能。

本文件只记录 03_2 本身，并与前序子实验 03_1 进行对比。完整 03 组的三方法总表和 Global / Local 贡献分解应放在 03 组 summary 文档中。

---

## 2. 实验设置

| 项目 | 内容 |
|---|---|
| Backbone | ULIP |
| 数据集参数 | sonn_c |
| 数据集变体 | hardest |
| 数据目录 | data/sonn_c |
| 实际数据文件 | data/sonn_c/hardest/clean.h5 |
| 方法 | Zero-shot + Global Cache |
| 方法简写 | zs_global |
| 原始核心 runner | runners/model_with_global_cache.py |
| 方法脚本 | Point-Cache/scripts/baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global_single_gpu.sh |
| 公共脚本 | Point-Cache/scripts/baseline/03_run_ulip_scanobjnn_clean_hardest_common.sh |
| cache_type | global |
| 是否使用 Global Cache | 是 |
| 是否使用 Local Cache | 否 |
| 输入点数 | 1024 |
| Global Cache shot_capacity | 3 |
| Global Cache alpha | 4.0 |
| Global Cache beta | 3.0 |
| 权重 | weights/ulip/pointbert_ulip1.pt |
| 文本编码器权重 | weights/ulip/slip_base_100ep.pt |
| GPU | 单张 Tesla T4 |

本实验使用 `sonn_c` 作为 dataset 参数，`sonn_variant=hardest`，并指定 `cor_type=clean`。实际读取文件为：

data/sonn_c/hardest/clean.h5

---

## 3. 方法说明

03_2 与 03_1 的区别在于：03_1 只使用 zero-shot logits，而 03_2 在此基础上加入 Global Cache。

| 子实验 | Zero-shot logits | Global Cache | Local Cache |
|---|---:|---:|---:|
| 03_1 | 是 | 否 | 否 |
| 03_2 | 是 | 是 | 否 |

Global Cache 的基本作用是：在测试时动态缓存高置信度样本的全局点云特征及其伪标签，再用后续测试样本与缓存特征之间的相似度生成 cache logits，并与 zero-shot logits 融合。

因此，本实验重点观察：

| 观察点 | 说明 |
|---|---|
| 03_2 是否高于 03_1 | 判断 Global Cache 是否带来有效增益 |
| 03_2 是否接近原文值 | 判断复现是否与 Point-Cache 原始结果对齐 |
| 增益幅度是否接近原文 | 判断 Global Cache 的实际作用是否稳定 |

---

## 4. 输出结构

输出目录：

Point-Cache/results/baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global/

输出内容：

| 文件或文件夹 | 说明 |
|---|---|
| summary.csv | clean 单文件实验的准确率汇总，共 1 行 |
| logs/ | 当前实验的完整运行日志 |
| wandb/ | wandb offline 日志 |

log 命名规则：

03_2_ulip_scanobjnn_clean_hardest_zs_global_clean_YYYYMMDD_HHMMSS.log

因为本实验只测试 `clean.h5`，所以期望只有 1 行 summary 记录和 1 个对应 log。

---

## 5. 当前结果检查

| 检查项 | 当前值 | 期望值 | 说明 |
|---|---:|---:|---|
| summary.csv 行数 | 1 | 1 | clean 实验只包含 1 个测试文件 |
| summary 中唯一 cor_type 数 | 1 | 1 | 只测试 clean |
| summary 中唯一 log_path 数 | 1 | 1 | 每次运行对应 1 个 log |
| status=done 数 | 1 | 1 | 脚本执行成功 |
| 当前复现 accuracy | 32.20 | 需与原文比较 | 不能只根据脚本成功判断结果是否正常 |

从执行完整性看，03_2 脚本执行成功，summary.csv 生成正常，结果记录完整。

---

## 6. 当前结果表

| 实验编号 | Dataset | Variant | File | Method | Accuracy | Status |
|---|---|---|---|---|---:|---|
| 03_2_ulip_scanobjnn_clean_hardest_zs_global | sonn_c | hardest | data/sonn_c/hardest/clean.h5 | Zero-shot + Global Cache | 32.20 | done |

该结果表示：ULIP 在 ScanObjNN hardest clean 数据上加入 Global Cache 后，准确率为 32.20。

---

## 7. 与原文结果对比

原文中与本实验直接相关的参考值为：

| 来源 | 原文值 | 说明 |
|---|---:|---|
| Supplementary Table 7 | 32.37 | S-PB T50-RS-C hardest split，ULIP + Global Cache |

当前复现结果为 32.20。

| 对比对象 | 原文值 | 当前复现值 | Diff = 当前 - 原文 | Abs Diff |
|---|---:|---:|---:|---:|
| Table 7 ULIP + Global Cache | 32.37 | 32.20 | -0.17 | 0.17 |

补充统计：

| 指标 | 数值 |
|---|---:|
| 与 Table 7 的差异 | -0.17 |
| 绝对差异 | 0.17 |

分析：

当前复现结果 32.20 与原文 Supplementary Table 7 的 32.37 相差 -0.17。误差很小，可以认为 03_2 的 Zero-shot + Global Cache 结果与原文基本对齐。

更严谨地说：03_2 不只是脚本跑通，而且数值也与原文 ScanObjNN hardest clean / ULIP + Global Cache 的参考结果基本一致。

---

## 8. 与前序实验 03_1 的对比

03_1 是本实验的直接前序子实验，方法为 Zero-shot，不使用缓存。

| 实验编号 | 方法 | 当前复现值 |
|---|---|---:|
| 03_1_ulip_scanobjnn_clean_hardest_zs | Zero-shot | 29.08 |
| 03_2_ulip_scanobjnn_clean_hardest_zs_global | Zero-shot + Global Cache | 32.20 |

Global Cache 带来的增益为：

| 比较 | 当前复现增益 |
|---|---:|
| 03_2 - 03_1 | +3.12 |

原文中对应的增益为：

| 比较 | 原文增益 |
|---|---:|
| Global Cache - Original Data | 32.37 - 29.29 = +3.08 |

对比：

| 指标 | 原文 | 当前复现 | Diff |
|---|---:|---:|---:|
| Global Cache 增益 | +3.08 | +3.12 | +0.04 |

分析：

当前复现中，Global Cache 将准确率从 29.08 提升到 32.20，提升 +3.12。该提升幅度与原文的 +3.08 几乎一致。

因此，03_2 说明 Global Cache 在 ScanObjNN clean hardest 上确实有效，而且当前复现不仅最终准确率接近原文，增益幅度也与原文基本一致。

---

## 9. 结果含义分析

ScanObjNN clean hardest 是真实扫描数据，即使没有 synthetic corruption，也存在明显 domain gap。03_1 的 Zero-shot 准确率只有 29.08，说明单纯依赖文本原型和点云编码器进行独立预测时，ULIP 在该设置下性能较低。

03_2 加入 Global Cache 后提升到 32.20，说明测试流中的全局特征缓存可以提供额外的分布信息，对真实扫描域偏移有一定缓解作用。

| 观察 | 含义 |
|---|---|
| 03_2 高于 03_1 | Global Cache 对 ScanObjNN clean hardest 有效 |
| 当前增益 +3.12 | 与原文增益 +3.08 基本一致 |
| 当前值 32.20 接近原文 32.37 | 复现结果可信 |
| 仍然只有 32.20 | 真实扫描 hardest split 仍然困难 |

这说明 Global Cache 是 ScanObjNN clean hardest 上的主要有效模块之一，但它并没有完全解决真实扫描数据带来的困难。

---

## 10. 与后续子实验的关系

03_2 是 03_3 的直接前序实验。

| 后续实验 | 对比方式 |
|---|---|
| 03_3_ulip_scanobjnn_clean_hardest_zs_global_local | 与 03_2 比较，评估 Local Cache 在 Global Cache 基础上的额外增益 |

本文件不展开 03_3 的实际结果。03_3 的数值及 Local Cache 额外贡献应记录在 03_3 子实验文档和 03 组 summary 文档中。

---

## 11. 阶段性结论

本实验完成了 ULIP × ScanObjNN clean hardest 的 Zero-shot + Global Cache baseline 复现。

主要结论如下：

1. 03_2 脚本执行成功，summary.csv 正常生成，status=done。
2. 当前复现准确率为 32.20。
3. 原文 Supplementary Table 7 中 ULIP 在 S-PB T50-RS-C hardest split 上的 +Global Cache 结果为 32.37。
4. 当前复现结果与原文差异为 -0.17，可以认为复现基本对齐。
5. 相比 03_1 Zero-shot 的 29.08，03_2 提升到 32.20，增益为 +3.12。
6. 当前 Global Cache 增益 +3.12 与原文增益 +3.08 基本一致。
7. 因此，Global Cache 在 ScanObjNN clean hardest 上的作用得到复现验证。
8. 本实验是 03_3 分析 Local Cache 额外贡献的直接对照。

---

## 12. 运行命令

使用第一张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global_single_gpu.sh 0

使用第二张 T4：

cd /root/autodl-tmp/MCM-PC-2/Point-Cache
bash scripts/baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global_single_gpu.sh 1

---

## 13. 检查命令

cd /root/autodl-tmp/MCM-PC-2/Point-Cache

tail -n +2 results/baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global/summary.csv | wc -l

tail -n +2 results/baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global/summary.csv | cut -d',' -f15 | sort -u | wc -l

find results/baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global/logs -maxdepth 1 -name '*.log' | wc -l

cat results/baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global/summary.csv
