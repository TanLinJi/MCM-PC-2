# Prompt Generation Protocol

更新日期：2026-06-17

## 保存位置

所有 E1 LLM 描述直接保存到：

```text
Point-Cache/llm/
```

## ModelNet-C 描述文件

```text
modelnet_c_llm_descriptions_deepseek_v4pro_10_prompts_4_image_4_pointcloud_2_bridge.json
modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json
modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_5_image_10_pointcloud.json
modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_12_image_3_pointcloud.json
```

## 描述类型

| 类型 | 含义 |
|---|---|
| image | common visual appearance, recognizable parts, image-level cues |
| pointcloud | 3D geometry, structure, symmetry, spatial layout, point distribution |
| bridge | connect image-level semantics with 3D geometric structure |

## 生成规则

1. 每个类别生成固定数量英文完整句子。
2. 每条描述至少 8 个英文词。
3. 不使用测试点云、测试标签或 corruption 信息。
4. 生成完成后必须检查类别数、每类条数和 JSON metadata。
5. 生成命令由用户手动执行。
6. 如果网络中断，可重复同一条命令；生成器会读取已完成的 partial JSON，只补缺失类别。

## E1_01

目标：

```text
10 prompts = 4 image + 4 pointcloud + 2 bridge
```

脚本：

```text
Point-Cache/scripts/E1_text_prototype_enhancement/prompt_generation/E1_01_10_prompts_4_image_4_pointcloud_2_bridge.sh
```

执行命令：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="conda run -n mcmpc python"
bash scripts/E1_text_prototype_enhancement/prompt_generation/E1_01_10_prompts_4_image_4_pointcloud_2_bridge.sh
```

输出保存位置：

```text
Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_10_prompts_4_image_4_pointcloud_2_bridge.json
```

完整性检查，2026-06-16：

```text
dataset_name: modelnet_c
llm_prompt_mode: image4_pointcloud4_bridge2
dynamic_prompt_count: 10
class_names: 40
completed_classes: 40
prompt_counts: [10]
failed_classes: []
missing_or_bad_classes: []
```

## E1_02

目标：

```text
15 prompts = 10 image + 5 pointcloud
```

脚本：

```text
Point-Cache/scripts/E1_text_prototype_enhancement/prompt_generation/E1_02_15_prompts_10_image_5_pointcloud.sh
```

执行命令：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="conda run -n mcmpc python"
bash scripts/E1_text_prototype_enhancement/prompt_generation/E1_02_15_prompts_10_image_5_pointcloud.sh
```

输出保存位置：

```text
Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_10_image_5_pointcloud.json
```

完整性检查，2026-06-16：

```text
dataset_name: modelnet_c
llm_prompt_mode: image10_pointcloud5
dynamic_prompt_count: 15
class_names: 40
completed_classes: 40
prompt_counts: [15]
failed_classes: []
missing_or_bad_classes: []
```

## E1_03

目标：

```text
15 prompts = 5 image + 10 pointcloud
```

脚本：

```text
Point-Cache/scripts/E1_text_prototype_enhancement/prompt_generation/E1_03_15_prompts_5_image_10_pointcloud.sh
```

执行命令：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="conda run -n mcmpc python"
bash scripts/E1_text_prototype_enhancement/prompt_generation/E1_03_15_prompts_5_image_10_pointcloud.sh
```

输出保存位置：

```text
Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_5_image_10_pointcloud.json
```

完整性检查，2026-06-17：

```text
dataset_name: modelnet_c
llm_prompt_mode: image5_pointcloud10
dynamic_prompt_count: 15
class_names: 40
completed_classes: 40
prompt_counts: [15]
failed_classes: []
missing_or_bad_classes: []
```

## E1_04

目标：

```text
15 prompts = 12 image + 3 pointcloud
```

脚本：

```text
Point-Cache/scripts/E1_text_prototype_enhancement/prompt_generation/E1_04_15_prompts_12_image_3_pointcloud.sh
```

执行命令：

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
export E1_PYTHON_CMD="python"
bash scripts/E1_text_prototype_enhancement/prompt_generation/E1_04_15_prompts_12_image_3_pointcloud.sh
```

输出保存位置：

```text
Point-Cache/llm/modelnet_c_llm_descriptions_deepseek_v4pro_15_prompts_12_image_3_pointcloud.json
```

完整性检查，2026-06-17：

```text
dataset_name: modelnet_c
llm_prompt_mode: image12_pointcloud3
dynamic_prompt_count: 15
class_names: 40
completed_classes: 40
prompt_counts: [15]
failed_classes: []
missing_or_bad_classes: []
```
