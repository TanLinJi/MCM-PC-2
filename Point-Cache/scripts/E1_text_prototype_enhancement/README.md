# E1：文本原型增强实验脚本

本目录存放 E1：文本原型增强实验相关脚本。

## 实验目标

E1 研究 Point-Cache 的文本原型是否可以通过以下方式增强：

- `manual_full`：Point-Cache 原始完整手工模板集合；
- `manual_3d`：从原始模板中筛选出的点云/3D 相关模板子集；
- `llm_dynamic_init`：实验初始化阶段由 LLM 根据候选类别名称生成类别级点云描述；
- `manual3d_llm_dynamic_init`：`manual_3d` 与 `llm_dynamic_init` 的文本原型加权融合。

## 脚本命名规则

脚本按阶段编号：

- `00_*`：环境、API、最小功能验证；
- `01_*`：zero-shot prompt-source 对比；
- `02_*`：Point-Cache global cache 对比；
- `03_*`：Point-Cache hierarchical cache 对比。

## 结果目录

E1 结果统一保存到：

    Point-Cache/results/E1_text_prototype_enhancement/

其中 LLM 动态生成提示词缓存保存到：

    Point-Cache/results/E1_text_prototype_enhancement/prompts/

## API Key

本地 API key 固定放在：

    Point-Cache/llm/secrets/llm_api_key.txt

该文件应只包含一行真实 API key，例如：

    sk-xxx

该文件被 `.gitignore` 忽略，不能提交。


## 2026-06-03 方向修正

severity=2 最小实验显示，`manual_3d` 明显低于 `manual_full`。

因此，E1 后续脚本方向调整为：

- `manual_full`：继续作为 E0-compatible baseline 和稳定视觉语义锚点；
- `manual_3d`：作为诊断消融，说明简单删除 2D 图像风格模板不可行；
- `llm_dynamic_init`：后续应改为生成“2D 视觉语义 + 3D 点云几何”的多视角描述；
- `manual3d_llm_dynamic_init`：保留为消融，不作为主方法；
- `manualfull_llm_dynamic_init`：新增为主候选方法，融合原始完整手工模板和多视角 LLM 描述。

后续新增脚本命名应体现该目的，例如：

    01_4_ulip_modelnetc_s2_zs_fuse_manualfull_multiview_llm_single_gpu.sh


## manual_3d 方向终止

severity=2 最小实验显示，`manual_3d` 明显低于 `manual_full`。

因此，本目录后续不再维护 `manual_3d` 相关 active 脚本。

后续 active 脚本集中于：

- `baseline_manual_full`
- `dynamic_multiview_llm_descriptions`
- `fuse_manualfull_multiview_llm`

`manual_3d` 的历史结果仅用于说明“简单删除 2D 图像风格模板不可行”。

