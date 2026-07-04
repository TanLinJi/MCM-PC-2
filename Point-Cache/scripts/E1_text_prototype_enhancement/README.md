# E1：文本原型增强实验脚本说明

本目录存放 E1：文本原型增强实验相关脚本。

## 1. 当前阶段定位

当前阶段是 smoke test，也就是最小验证实验。

当前 smoke test 设置：

- Backbone：ULIP
- 数据集：ModelNet-C
- 损坏强度：severity=2
- 损坏类型：7 类
- 任务设置：zero-shot
- 默认运行方式：单卡运行
- GPU 选择方式：通过脚本最后一个参数指定物理 GPU 编号

示例：

    bash 某个脚本.sh 0

表示使用物理 GPU 0。

注意：

- 文件名不再使用 `single_gpu`；
- 默认就是单卡运行；
- 后续如果需要多卡或批量调度，再单独建立批量脚本。

## 2. 四种核心文本方法

| 编号 | 方法名 | prompt source | 含义 | 当前作用 |
|---|---|---|---|---|
| 00_1 | manual_full | manual_full | Point-Cache 原始完整手工模板 | baseline，对齐 E0 zero-shot |
| 00_2 | manual_3d | manual_3d | 删除 2D 图像风格模板后的 3D 手工模板 | 失败消融，证明不能简单删除 2D 模板 |
| 00_3 | llm_only | llm_dynamic_init | 只使用 LLM 生成的类别级多视角描述 | 对照实验，验证 LLM 描述能否独立替代手工模板 |
| 00_4 | manual_full_llm_fusion | manualfull_llm_dynamic_init | 原始完整手工模板文本原型与 LLM 描述文本原型加权融合 | 当前主方法 |

## 3. 脚本列表、作用和运行方式

### 3.1 LLM prompt 生成最小测试

脚本：

    00_0_llm_prompt_generation_smoke.sh

作用：

- 不跑分类模型；
- 只测试 LLM API key 是否能读取；
- 测试 LLM API 是否能调用；
- 测试生成的类别描述是否能保存到 prompt 缓存；
- 用于排查 API key、网络、缓存路径和生成器逻辑问题。

运行方式：

    bash Point-Cache/scripts/E1_text_prototype_enhancement/00_0_llm_prompt_generation_smoke.sh

如果需要强制重新调用 API，而不是读取已有缓存：

    FORCE_REGENERATE=1 bash Point-Cache/scripts/E1_text_prototype_enhancement/00_0_llm_prompt_generation_smoke.sh

说明：

- 一般不建议频繁使用 `FORCE_REGENERATE=1`，因为会消耗 API token；
- 正常情况下应优先读取缓存。

### 3.2 smoke test 公共脚本

脚本：

    00_run_ulip_modelnetc_s2_zs_text_method_smoke_common.sh

作用：

- 所有 ULIP × ModelNet-C severity=2 zero-shot 文本方法 smoke test 的公共入口；
- 负责设置环境变量、结果目录、runner 路径、数据集、模型权重和 prompt-source 参数；
- 通常不直接手动运行，而由下面四个方法脚本调用。

参数格式：

    bash 00_run_ulip_modelnetc_s2_zs_text_method_smoke_common.sh EXP_ID PROMPT_SOURCE METHOD_FULL PURPOSE [GPU]

参数含义：

| 参数 | 含义 |
|---|---|
| EXP_ID | 实验结果目录名 |
| PROMPT_SOURCE | 文本方法代码名 |
| METHOD_FULL | 方法完整说明 |
| PURPOSE | 实验目的说明 |
| GPU | 物理 GPU 编号，可省略，默认 0 |

一般用户不需要直接运行该公共脚本。

### 3.3 manual_full smoke test

脚本：

    00_1_ulip_modelnetc_s2_zs_manual_full_smoke.sh

对应方法：

    manual_full

作用：

- 使用 Point-Cache 原始完整手工模板；
- 验证 E1 新增 prompt-source 接口是否破坏原始 zero-shot baseline；
- 该实验应与 E0 中 ULIP × ModelNet-C severity=2 zero-shot 的结果一致。

运行方式：

    bash Point-Cache/scripts/E1_text_prototype_enhancement/00_1_ulip_modelnetc_s2_zs_manual_full_smoke.sh 0

结果目录：

    Point-Cache/results/E1_text_prototype_enhancement/00_1_ulip_modelnetc_s2_zs_manual_full_smoke/

当前已知结果：

    平均准确率：47.68

### 3.4 manual_3d smoke test

脚本：

    00_2_ulip_modelnetc_s2_zs_manual_3d_smoke.sh

对应方法：

    manual_3d

作用：

- 从原始手工模板中删除明显 2D 图像风格模板；
- 仅保留更偏 3D 形状、几何结构和点云语义的模板；
- 用作失败消融实验；
- 用于验证“不能简单删除 2D 图像风格模板”这一判断。

运行方式：

    bash Point-Cache/scripts/E1_text_prototype_enhancement/00_2_ulip_modelnetc_s2_zs_manual_3d_smoke.sh 0

结果目录：

    Point-Cache/results/E1_text_prototype_enhancement/00_2_ulip_modelnetc_s2_zs_manual_3d_smoke/

当前已知结果：

    平均准确率：35.63

实验结论：

- 显著低于 manual_full；
- 说明 2D 图像风格模板对于 ULIP 的 CLIP-style 文本空间具有语义锚定作用；
- 后续不再把 manual_3d 作为主方法，只保留为失败消融。

### 3.5 llm_only smoke test

脚本：

    00_3_ulip_modelnetc_s2_zs_llm_only_smoke.sh

对应方法：

    llm_only

内部 prompt source：

    llm_dynamic_init

作用：

- 不使用 Point-Cache 原始手工模板；
- 只使用 LLM 根据类别名称生成的类别级多视角描述；
- 每个类别默认生成 10 条描述；
- 描述集合中包含一部分 2D 视觉语义描述和一部分 3D 点云几何描述；
- 用于验证 LLM 生成描述能否独立替代原始手工模板。

运行方式：

    LLM_TEMPERATURE=0.3 bash Point-Cache/scripts/E1_text_prototype_enhancement/00_3_ulip_modelnetc_s2_zs_llm_only_smoke.sh 0

结果目录：

    Point-Cache/results/E1_text_prototype_enhancement/00_3_ulip_modelnetc_s2_zs_llm_only_smoke/

LLM prompt 缓存目录：

    Point-Cache/llm/e1_prompt_bank/

当前已知结果：

    平均准确率：39.30

实验结论：

- 高于 manual_3d；
- 低于 manual_full；
- 说明 LLM 生成描述有一定语义价值，但不能直接替代原始完整手工模板。

### 3.6 manual_full_llm_fusion smoke test

脚本：

    00_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_smoke.sh

对应方法：

    manual_full_llm_fusion

内部 prompt source：

    manualfull_llm_dynamic_init

作用：

- 同时使用 Point-Cache 原始完整手工模板和 LLM 生成的类别级多视角描述；
- 先分别构造两个文本原型：
  - manual_full 文本原型；
  - LLM 描述文本原型；
- 再进行加权融合；
- 这是当前 E1 的主方法。

默认融合权重：

    manual_full : LLM = 0.75 : 0.25

运行方式：

    LLM_TEMPERATURE=0.3 bash Point-Cache/scripts/E1_text_prototype_enhancement/00_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_smoke.sh 0

如果要修改融合权重，例如 manual_full:LLM = 0.85:0.15：

    PROMPT_STATIC_WEIGHT=0.85 PROMPT_DYNAMIC_WEIGHT=0.15 LLM_TEMPERATURE=0.3 bash Point-Cache/scripts/E1_text_prototype_enhancement/00_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_smoke.sh 0

结果目录：

    Point-Cache/results/E1_text_prototype_enhancement/00_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_smoke/

LLM prompt 缓存目录：

    Point-Cache/llm/e1_prompt_bank/

当前已知结果：

    平均准确率：48.88

实验结论：

- 超过 manual_full baseline 1.20；
- 说明 LLM 描述不适合替代原始模板，但适合作为补充语义分支；
- 原始完整手工模板提供稳定 CLIP-style 视觉语义锚点；
- LLM 描述补充类别级视觉语义和 3D 点云几何信息。

## 4. 结果目录规范

结果统一保存到：

    Point-Cache/results/E1_text_prototype_enhancement/

当前规范化后的 smoke test 结果目录应为：

| 目录 | 对应方法 |
|---|---|
| `00_1_ulip_modelnetc_s2_zs_manual_full_smoke/` | manual_full |
| `00_2_ulip_modelnetc_s2_zs_manual_3d_smoke/` | manual_3d |
| `00_3_ulip_modelnetc_s2_zs_llm_only_smoke/` | llm_only |
| `00_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_smoke/` | manual_full_llm_fusion |

说明：

- `results/` 被 `.gitignore` 忽略；
- 结果文件不提交到 Git；
- summary.csv 可用于分析，但正式分析文档应放到 `docs/experiments/E1_text_prototype_enhancement/` 下。

## 5. LLM API Key

本地 API key 固定放在：

    Point-Cache/llm/secrets/llm_api_key.txt

文件格式：

    sk-xxx

说明：

- 文件只包含一行真实 API key；
- 该文件被 `.gitignore` 忽略；
- 不能提交到 Git。

## 6. LLM prompt 缓存说明

LLM 生成的类别描述会保存为 JSON 缓存。

作用：

- 避免重复调用 API；
- 节省 token；
- 保证重复实验时使用相同描述；
- 支持续跑，已经生成的类别不会重新生成。

E1 重启后的 prompt 缓存统一放在：

    Point-Cache/llm/e1_prompt_bank/

旧的 `results/E1_text_prototype_enhancement/shared_prompts/` 只作为历史结果追溯。

## 7. 当前结论

当前 smoke test 已显示：

- manual_3d 明显低于 manual_full，说明不能简单删除 2D 图像风格模板；
- llm_only 高于 manual_3d，但仍低于 manual_full，说明 LLM 描述不能直接替代原始模板；
- manual_full_llm_fusion 超过 manual_full，说明 LLM 描述适合作为补充语义分支与原始模板融合。

## 8. 下一步实验

规范化完成后，下一步优先进行融合权重消融。

候选权重：

| manual_full 权重 | LLM 权重 |
|---:|---:|
| 0.90 | 0.10 |
| 0.85 | 0.15 |
| 0.75 | 0.25 |
| 0.50 | 0.50 |

权重消融完成后，再选择最佳权重运行 ModelNet-C all35 zero-shot 完整验证。

## 9. E1-S1：融合权重消融脚本

E1-S1 用于寻找 `manual_full_llm_fusion` 的更优融合比例。

公共脚本：

    01_run_ulip_modelnetc_s2_zs_fusion_weight_ablation_common.sh

权重消融脚本：

| 脚本 | manual_full 权重 | LLM 权重 | 作用 |
|---|---:|---:|---|
| `01_1_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w090_010.sh` | 0.90 | 0.10 | 更保守地引入 LLM 描述 |
| `01_2_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w085_015.sh` | 0.85 | 0.15 | 中等保守融合 |
| `01_3_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w075_025.sh` | 0.75 | 0.25 | smoke test 中已取得正结果的默认权重 |
| `01_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w050_050.sh` | 0.50 | 0.50 | 检查较高 LLM 权重是否导致文本原型偏移 |
| `01_5_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020.sh` | 0.80 | 0.20 | 补充 0.85:0.15 与 0.75:0.25 之间的中间权重 |

运行示例：

    bash Point-Cache/scripts/E1_text_prototype_enhancement/01_1_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w090_010.sh 0

说明：

- 权重消融统一读取共享 prompt 缓存；
- 不应重新生成 LLM prompt；
- 共享 prompt 缓存路径为：
  `Point-Cache/llm/e1_prompt_bank/`

## 10. E1-S2：LLM 描述数量与 2D/3D 比例消融

本阶段补充每类 15 条 LLM 描述，并测试两种 2D/3D 比例：

| 脚本 | 作用 |
|---|---|
| `02_1_generate_modelnetc_llm_prompts_p15_2d3d_2to1.sh` | 生成 15 条描述，2D:3D = 2:1 |
| `02_2_generate_modelnetc_llm_prompts_p15_2d3d_1to2.sh` | 生成 15 条描述，2D:3D = 1:2 |
| `02_3_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020_p15_2d3d_2to1.sh` | 评估 15 条描述，2D:3D = 2:1 |
| `02_4_ulip_modelnetc_s2_zs_manual_full_llm_fusion_w080_020_p15_2d3d_1to2.sh` | 评估 15 条描述，2D:3D = 1:2 |

这两个 15 prompts 消融默认使用：

    manual_full:LLM = 0.80:0.20

## 11. 当前候选配置跨数据集验证

E1_36 已固定为当前正式候选配置：

    15 prompts/class = 10 image + 5 pointcloud
    manual_full:LLM = 0.60:0.40

clean ModelNet 验证脚本：

    modelnet_clean_validation/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40.sh

运行方式：

    cd /root/autodl-tmp/MCM-PC-2/Point-Cache
    bash scripts/E1_text_prototype_enhancement/modelnet_clean_validation/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40.sh 0

结果目录：

    Point-Cache/results/E1_text_prototype_enhancement/E1_40_modelnet_clean_15_prompts_10_image_5_pointcloud_manual60_llm40/
