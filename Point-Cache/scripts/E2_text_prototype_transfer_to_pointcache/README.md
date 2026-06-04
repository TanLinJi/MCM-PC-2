# E2：文本原型增强向 Point-Cache 缓存流程的传递验证脚本

## 1. 当前阶段定位

E2 当前是 smoke test，用于验证 E1 中得到的文本原型融合收益能否传递到 Point-Cache 的缓存增强流程。

当前设置：

- Backbone：ULIP
- 数据集：ModelNet-C
- 损坏强度：severity=2
- 损坏类型：7 类
- 文本方法：manual_full、manual_full_llm_fusion
- 缓存设置：zs_global、zs_global_local
- 默认单卡运行，可通过脚本最后一个参数选择物理 GPU 编号

## 2. 脚本列表

| 脚本 | 设置 | 文本方法 | 作用 |
|---|---|---|---|
| `00_run_ulip_modelnetc_s2_cache_transfer_common.sh` | common | common | E2 公共脚本 |
| `00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke.sh` | zs_global | manual_full | 原始模板 + global cache |
| `00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke.sh` | zs_global | manual_full_llm_fusion | 文本融合 + global cache |
| `00_3_ulip_modelnetc_s2_zs_global_local_manual_full_smoke.sh` | zs_global_local | manual_full | 原始模板 + 完整 Point-Cache |
| `00_4_ulip_modelnetc_s2_zs_global_local_manual_full_llm_fusion_smoke.sh` | zs_global_local | manual_full_llm_fusion | 文本融合 + 完整 Point-Cache |

## 3. 运行方式

使用物理 GPU 0 运行 manual_full + global cache：

    bash Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke.sh 0

使用物理 GPU 1 运行 manual_full_llm_fusion + global cache：

    bash Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke.sh 1

使用物理 GPU 0 运行 manual_full + 完整 Point-Cache：

    bash Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/00_3_ulip_modelnetc_s2_zs_global_local_manual_full_smoke.sh 0

使用物理 GPU 1 运行 manual_full_llm_fusion + 完整 Point-Cache：

    bash Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache/00_4_ulip_modelnetc_s2_zs_global_local_manual_full_llm_fusion_smoke.sh 1

## 4. 结果目录

结果保存到：

    Point-Cache/results/E2_text_prototype_transfer_to_pointcache/

对应目录：

- `00_1_ulip_modelnetc_s2_zs_global_manual_full_smoke/`
- `00_2_ulip_modelnetc_s2_zs_global_manual_full_llm_fusion_smoke/`
- `00_3_ulip_modelnetc_s2_zs_global_local_manual_full_smoke/`
- `00_4_ulip_modelnetc_s2_zs_global_local_manual_full_llm_fusion_smoke/`

## 5. LLM prompt 缓存

E2 不重新生成 LLM prompt，统一复用 E1 共享缓存：

    Point-Cache/results/E1_text_prototype_enhancement/shared_prompts/modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json

这样可以保证 E1 与 E2 使用完全相同的 LLM 类别描述，避免重复消耗 API token，也避免随机生成差异。

## 6. 判断标准

如果：

    manual_full_llm_fusion + zs_global
    >
    manual_full + zs_global

说明 E1 文本收益可以传递到 global cache。

如果：

    manual_full_llm_fusion + zs_global_local
    >
    manual_full + zs_global_local

说明 E1 文本收益可以传递到完整 Point-Cache。
