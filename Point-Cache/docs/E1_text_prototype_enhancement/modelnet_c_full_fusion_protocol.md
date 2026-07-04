# ModelNet-C Full Fusion Protocol

更新日期：2026-06-17

## 范围

E1 的 ModelNet-C 实验直接运行完整 ModelNet-C：

```text
7 corruption types x 5 severities = 35 evaluations
```

不单独报告某个 severity 的消融结论。

## 公共脚本

```text
Point-Cache/scripts/E1_text_prototype_enhancement/common/run_modelnet_c_fusion.sh
```

公共脚本负责：

1. 检查 prompt JSON 是否存在；
2. 调用正式 ModelNet-C full runner；
3. 固定 `--modelnet-c-severities all`；
4. 保存结果到 `Point-Cache/results/E1_text_prototype_enhancement/`。

注：公共脚本默认运行 `MODELNET_C_SEVERITIES=all`。若显式设置 `MODELNET_C_SEVERITIES=2`，则只运行 severity=2，用于诊断性对比。

## 单实验文档

每个实验的基本设置、执行命令、结果目录、检查项和结果分析都写入独立文档：

| 实验 | 文档 |
|---|---|
| E1_10 ModelNet-C full `manual90_llm10` | `E1_10_modelnet_c_full_manual90_llm10.md` |
| E1_13 ModelNet-C severity=2 `manual75_llm25` diagnostic | `E1_13_modelnet_c_severity2_manual75_llm25.md` |
| E1_13 ModelNet-C full `manual75_llm25` | `E1_13_modelnet_c_full_manual75_llm25.md` |

后续 E1_11、E1_12、E1_14、E1_20 和 E1_21 也按同样规则新增独立实验文档。
