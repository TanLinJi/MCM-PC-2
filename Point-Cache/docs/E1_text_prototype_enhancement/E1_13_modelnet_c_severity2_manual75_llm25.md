# E1_13 ModelNet-C Severity2 Manual75 LLM25 Diagnostic

更新日期：2026-06-17

## 基本实验设置

| 项目 | 设置 |
|---|---|
| 实验编号 | E1_13 severity2 diagnostic |
| Result `exp_id` | `E1_13_modelnet_c_severity2_manual75_llm25` |
| 实验目的 | 在同一新版 prompt JSON 下，诊断 `manual_full:LLM = 75:25` 的 severity=2 表现 |
| 数据集与评估范围 | ModelNet-C severity=2，7 evaluations = 7 corruption types x 1 severity |
| Backbone | ULIP |
| 任务设置 | zero-shot |
| 文本原型方法 | `manual_full + LLM` weighted fusion |
| LLM prompt | 10 prompts/class = 4 image + 4 pointcloud + 2 bridge |
| 融合权重 | `manual_full = 0.75`, `LLM = 0.25` |
| Prompt JSON | `Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_10_prompts_4_image_4_pointcloud_2_bridge.json` |
| 结果目录 | `Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_severity2_manual75_llm25/` |

说明：这是诊断性平行实验，不替代 E1_13 的 ModelNet-C full 主线实验。它只用于隔离比较同一新版 prompt 下 `manual90_llm10` 与 `manual75_llm25` 在 severity=2 上的差异。

编号记录：当前诊断实验沿用 `E1_13` 权重槽，并在编号后显式加上 `severity2 diagnostic`。正式 ModelNet-C full 的 `manual75_llm25` 主线实验记录为 `E1_13_modelnet_c_full_manual75_llm25`，已完成并单独成文档。

## 执行命令

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="conda run -n mcmpc python"
bash scripts/E1_text_prototype_enhancement/fusion_weight_ablation/E1_13_modelnet_c_severity2_manual75_llm25.sh 0
```

## 输出位置

```text
Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_severity2_manual75_llm25/
Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_severity2_manual75_llm25/summary.csv
Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_severity2_manual75_llm25/logs/
```

## 完成检查

| 检查项 | 结果 |
|---|---|
| `summary.csv` 数据行数 | 7 |
| `status` | 全部 `done` |
| severity | 全部为 `2` |
| corruption 覆盖 | 7 类 |
| 平均准确率 | 48.90 |

## 对比范围

所有对比都限制在 ModelNet-C `severity=2`，即 7 个 corruption 类型。

| 对比对象 | 路径 | 说明 |
|---|---|---|
| E1_13 severity2 diagnostic | `Point-Cache/results/E1_text_prototype_enhancement/E1_13_modelnet_c_severity2_manual75_llm25/summary.csv` | 新版 prompt，`manual75_llm25` |
| 旧 `0.75:0.25` | `Point-Cache/results/E1_text_prototype_enhancement/01_3_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w075_025/summary.csv` | 旧 prompt，`manual75_llm25` |
| E1_10 severity=2 子集 | `Point-Cache/results/E1_text_prototype_enhancement/E1_10_modelnet_c_full_w90_10/summary.csv` | 新版 prompt，`manual90_llm10`，只取 severity=2 |
| E0 baseline severity=2 子集 | `Point-Cache/results/E0_baseline/02_1_ulip_modelnetc_corruptions_all35_zs/summary.csv` | ULIP zero-shot baseline，只取 severity=2 |

## 总体对比

| 设置 | severity=2 平均准确率 | 与 E1_13 差值 |
|---|---:|---:|
| E1_13 `manual75_llm25`, 新版 prompt | 48.90 | 0.00 |
| 旧 `0.75:0.25`, 旧 prompt | 48.88 | +0.02 |
| E1_10 `manual90_llm10`, 新版 prompt | 48.41 | +0.49 |
| E0 baseline | 47.68 | +1.22 |

## 逐 Corruption 对比

| corruption | E1_13 | 旧 0.75:0.25 | diff | E1_10 severity=2 | diff | E0 severity=2 | diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| add_global | 35.33 | 33.55 | +1.78 | 35.05 | +0.28 | 34.00 | +1.33 |
| add_local | 44.49 | 44.61 | -0.12 | 44.12 | +0.37 | 43.92 | +0.57 |
| dropout_global | 57.13 | 57.01 | +0.12 | 55.92 | +1.21 | 54.70 | +2.43 |
| dropout_local | 51.82 | 53.44 | -1.62 | 51.70 | +0.12 | 50.57 | +1.25 |
| rotate | 56.16 | 56.36 | -0.20 | 55.75 | +0.41 | 55.19 | +0.97 |
| scale | 53.28 | 52.76 | +0.52 | 51.82 | +1.46 | 50.89 | +2.39 |
| jitter | 44.12 | 44.45 | -0.33 | 44.53 | -0.41 | 44.49 | -0.37 |

## 分析记录

与旧 `0.75:0.25` 对比：

1. 两者权重相同，都是 `manual_full:LLM = 0.75:0.25`，但 prompt JSON 不同。
2. E1_13 平均准确率为 48.90，旧结果为 48.88，整体只高 +0.02，几乎持平。
3. E1_13 在 `add_global`、`dropout_global`、`scale` 上更好，在 `add_local`、`dropout_local`、`rotate`、`jitter` 上更低。
4. 最大正增益是 `add_global` (+1.78)，最大下降是 `dropout_local` (-1.62)。
5. 因为 prompt 文本不同，这个对比不能解释为权重效果，只能说明新版 prompt 在同权重下总体没有明显损失。

与 E1_10 `manual90_llm10` 对比：

1. E1_13 和 E1_10 使用同一个新版 prompt JSON、同一 backbone、同一 ModelNet-C severity=2 范围；主要差异是融合权重。
2. E1_13 平均 48.90，高于 E1_10 severity=2 子集的 48.41，差值为 +0.49。
3. E1_13 在 7 类 corruption 中有 6 类高于 E1_10，只有 `jitter` 低 -0.41。
4. 提升最明显的是 `scale` (+1.46) 和 `dropout_global` (+1.21)。
5. 在 severity=2 诊断范围内，提高 LLM 权重到 25% 比 10% 更好；但该结论目前只适用于 severity=2，不能直接替代 ModelNet-C full 结论。

与 E0 baseline 对比：

1. E1_13 平均 48.90，高于 E0 severity=2 baseline 的 47.68，差值为 +1.22。
2. E1_13 在 7 类 corruption 中有 6 类高于 E0，只有 `jitter` 低 -0.37。
3. 对 E0 的提升主要来自 `dropout_global` (+2.43)、`scale` (+2.39)、`add_global` (+1.33)、`dropout_local` (+1.25)。
4. `jitter` 是当前设置的主要负例，说明更高 LLM 权重可能会牺牲随机抖动扰动下的稳定性。

## 当前结论

E1_13 severity2 diagnostic 证明，在新版 prompt 下，`manual75_llm25` 在 ModelNet-C severity=2 上优于 `manual90_llm10` 和 E0 baseline；但它相对旧 `0.75:0.25` 基本持平。

正式 full 结果已记录在 `E1_13_modelnet_c_full_manual75_llm25.md`。该 full 结果确认 `manual75_llm25` 在完整 ModelNet-C 上也优于 `manual90_llm10`。
