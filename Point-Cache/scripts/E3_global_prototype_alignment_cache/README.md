# E3：全局原型对齐缓存脚本说明

## 1. 当前阶段

E3 当前实现 E3-V1：顺序式 Global Prototype-Alignment Cache。

当前只跑完整 Point-Cache 设置：

    zs_global_local

并对比 E2 中已有的原始 full Point-Cache 结果。

## 2. 脚本列表

| 脚本 | 文本方法 | 作用 |
|---|---|---|
| `00_run_ulip_modelnetc_s2_gpa_common.sh` | common | E3 公共脚本 |
| `00_1_ulip_modelnetc_s2_zs_global_local_gpa_manual_full_smoke.sh` | manual_full | 不使用 E2 文本融合，验证 GPA Cache 本身是否有效 |
| `00_2_ulip_modelnetc_s2_zs_global_local_gpa_manual_full_llm_fusion_smoke.sh` | manual_full_llm_fusion | 使用 E2 文本融合，验证 GPA Cache 是否兼容文本融合 |

## 3. 运行方式

GPU 0 运行 manual_full：

    bash Point-Cache/scripts/E3_global_prototype_alignment_cache/00_1_ulip_modelnetc_s2_zs_global_local_gpa_manual_full_smoke.sh 0

GPU 1 运行 manual_full_llm_fusion：

    bash Point-Cache/scripts/E3_global_prototype_alignment_cache/00_2_ulip_modelnetc_s2_zs_global_local_gpa_manual_full_llm_fusion_smoke.sh 1

## 4. 结果目录

结果保存到：

    Point-Cache/results/E3_global_prototype_alignment_cache/

对应目录：

- `00_1_ulip_modelnetc_s2_zs_global_local_gpa_manual_full_smoke/`
- `00_2_ulip_modelnetc_s2_zs_global_local_gpa_manual_full_llm_fusion_smoke/`

## 5. GPA 统计文件

每个 corruption 会额外保存 GPA 统计信息到：

    Point-Cache/results/E3_global_prototype_alignment_cache/<EXP_ID>/gpa_stats/

统计内容包括：

- Global Entropy Cache 每类数量；
- GPA Cache 每类数量；
- GPA-controlled Local Cache 每类数量；
- GPA 启动、加入、替换、拒绝次数；
- 最终准确率。

## 6. E3-01：中心来源消融脚本

E3-01 固定顺序式 GPA Cache 关系不变，只改变 GPA 原型中心来源。

公共脚本：

    01_run_ulip_modelnetc_s2_gpa_center_source_ablation_common.sh

实验脚本：

| 脚本 | 中心来源 | 文本方法 | 作用 |
|---|---|---|---|
| `01_1_ulip_modelnetc_s2_zs_global_local_gpa_entropy_only_center_manual_full.sh` | Entropy-only center | manual_full | 使用 Global Entropy Cache 计算 GPA 原型中心 |
| `01_2_ulip_modelnetc_s2_zs_global_local_gpa_entropy_gpa_union_center_manual_full.sh` | Entropy+GPA union center | manual_full | 使用 Global Entropy Cache 与 GPA Cache 并集计算 GPA 原型中心 |

运行方式：

    bash Point-Cache/scripts/E3_global_prototype_alignment_cache/01_1_ulip_modelnetc_s2_zs_global_local_gpa_entropy_only_center_manual_full.sh 0

    bash Point-Cache/scripts/E3_global_prototype_alignment_cache/01_2_ulip_modelnetc_s2_zs_global_local_gpa_entropy_gpa_union_center_manual_full.sh 1

说明：

- 这两组实验只改变中心来源；
- 仍然采用顺序式 GPA Cache；
- 仍然使用 manual_full；
- 仍然运行 zs_global_local；
- 结果用于和 E2 00_3 以及 E3-V1-A GPA-only center 对比。

## 7. E3-02：并列式 GPA Cache 中心来源消融脚本

E3-02 将 Global Entropy Cache 与 GPA Cache 改为并列更新。

公共脚本：

    02_run_ulip_modelnetc_s2_parallel_gpa_center_source_ablation_common.sh

实验脚本：

| 脚本 | 关系 | 中心来源 | 更新规则 | 文本方法 |
|---|---|---|---|---|
| `02_1_ulip_modelnetc_s2_zs_global_local_parallel_gpa_gpa_only_center_manual_full.sh` | 并列式 | GPA-only center | 低熵 + 替换最高熵样本 | manual_full |
| `02_2_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_only_center_manual_full.sh` | 并列式 | Entropy-only center | 低熵 + 替换最高熵样本 | manual_full |
| `02_3_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_gpa_union_center_manual_full.sh` | 并列式 | Entropy+GPA union center | 低熵 + 替换最高熵样本 | manual_full |
| `02_4_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_gpa_union_center_replace_farthest_manual_full.sh` | 并列式 | Entropy+GPA union center | 低熵 + 替换最远样本 | manual_full |

运行方式：

    bash Point-Cache/scripts/E3_global_prototype_alignment_cache/02_1_ulip_modelnetc_s2_zs_global_local_parallel_gpa_gpa_only_center_manual_full.sh 0

    bash Point-Cache/scripts/E3_global_prototype_alignment_cache/02_2_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_only_center_manual_full.sh 1

    bash Point-Cache/scripts/E3_global_prototype_alignment_cache/02_3_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_gpa_union_center_manual_full.sh 0

    bash Point-Cache/scripts/E3_global_prototype_alignment_cache/02_4_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_gpa_union_center_replace_farthest_manual_full.sh 0

说明：

- `02_3` 对应 E3-V2-C，即 `2+C+a`：并列式 + Entropy/GPA union center + 低熵门控 + 替换最高熵样本；
- `02_4` 对应 E3-V2-Cb，即 `2+C+b`：并列式 + Entropy/GPA union center + 低熵门控 + 替换离中心最远样本；
- `02_4` 会在 `gpa_replacement_events_*.jsonl` 中记录最高熵样本、最远样本、二者是否重合，以及缓存内每个样本的熵和距离，用于后续统计熵与原型距离的关系。

## 8. E3-03：GPA Cache 候选池初始化脚本

E3-03 用于验证 GPA Cache 初始化机制改进。

当前脚本：

| 脚本 | 方法 | 作用 |
|---|---|---|
| `03_1_ulip_modelnetc_s2_zs_global_local_parallel_gpa_candidate_pool_init_manual_full.sh` | E3-V3-A | 并列式 GPA Cache + 候选池初始化 |

运行方式：

    bash Point-Cache/scripts/E3_global_prototype_alignment_cache/03_1_ulip_modelnetc_s2_zs_global_local_parallel_gpa_candidate_pool_init_manual_full.sh 0

默认候选池倍率：

    GPA_CANDIDATE_MULTIPLIER=2

即：

    每类先收集 2K 个候选样本，
    再根据 Entropy Cache 与 GPA candidate pool 联合中心筛出 K 个进入 GPA Cache。
