# E3 实验日志：全局原型对齐缓存

## 2026-06-04：建立 E3 实验计划

E3 实验名称：

    E3_global_prototype_alignment_cache

中文名称：

    E3：全局原型对齐缓存

当前 E3-V1 采用顺序式 GPA Cache 方案：

- 先按原始 Point-Cache 低熵规则更新 Global Entropy Cache；
- 再尝试进入 Global Prototype-Alignment Cache；
- GPA Cache 自己维护类别原型中心；
- GPA Cache 未形成中心前，先按低熵准入积累初始样本；
- GPA Cache 形成中心后，启用低熵 + 原型距离约束；
- 只有进入 GPA Cache 的样本，其 local patch centers 才写入 Local Cache；
- 当前最小验证阶段暂不修改最终预测加权公式。

当前计划先跑两组：

| 编号 | 文本方法 | cache 设置 |
|---|---|---|
| 00_1 | manual_full | zs_global_local |
| 00_2 | manual_full_llm_fusion | zs_global_local |

## 2026-06-04：实现 E3-V1 顺序式 GPA Cache runner 和脚本

新增 E3 专用 runner：

    Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_gpa.py
    Point-Cache/runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_gpa.py

新增 E3 脚本目录：

    Point-Cache/scripts/E3_global_prototype_alignment_cache/

新增脚本：

- `00_run_ulip_modelnetc_s2_gpa_common.sh`
- `00_1_ulip_modelnetc_s2_zs_global_local_gpa_manual_full_smoke.sh`
- `00_2_ulip_modelnetc_s2_zs_global_local_gpa_manual_full_llm_fusion_smoke.sh`

当前实现：

- Global Entropy Cache 保留原始 Point-Cache 低熵逻辑；
- GPA Cache 采用 GPA-only center；
- GPA Cache 未形成中心前先按低熵准入积累；
- GPA Cache 形成中心后使用低熵 + 距离约束；
- 只有进入 GPA Cache 的样本写入 local cache；
- 当前最小验证阶段不修改最终预测加权公式。

## 2026-06-05：记录 E3-V1 GPA-only center 初版负结果

E3-V1 00_1 已完成：

    manual_full + zs_global_local + 顺序式 GPA Cache + GPA-only center

平均准确率：

    53.44

对比 E2 原始 full Point-Cache：

    manual_full + zs_global_local = 54.00

下降：

    -0.56

该结果说明当前顺序式 GPA-only center 不是最佳方案。

同时发现：

- 预构建阶段 GPA-controlled Local Cache 数量与 Global Entropy Cache 数量几乎一致；
- 当前规则在 GPA Cache 未满阶段过于宽松；
- 结果目录中未找到 gpa_stats，需要检查统计保存逻辑；
- 后续必须记录 GPA 替换事件，包括新旧样本的熵和距离。

下一步计划：

1. 修复 GPA 统计保存；
2. 增加 `gpa_replacement_events_<cor_type>.jsonl`；
3. 实现 Center-B：Entropy-only center；
4. 实现 Center-C：Entropy+GPA union center；
5. 两张卡分别运行 Center-B 和 Center-C。

## 2026-06-05：修正 E3-V1-A 诊断与 GPA Cache 状态命名

本次确认：

- 当前顺序式 GPA Cache 在未满阶段没有形成更严格筛选；
- 只要样本进入 Global Entropy Cache，且 GPA Cache 未满，就会进入 GPA Cache；
- 因此 min_center_size 在当前逻辑中没有实际作用，已从代码和脚本中删除；
- 当前 E3-V1-A 平均准确率为 53.44，低于 E2 原始 full Point-Cache 的 54.00；
- 代码中曾存在 runtime_gpa_cache 命名遗留，容易误导为预构建 GPA Cache 与正式测试 GPA Cache 分离；
- 当前已统一为 gpa_cache，确保预构建阶段形成的 GPA Cache 在正式测试阶段继续沿用并更新；
- 后续需要新增 `gpa_replacement_events_<cor_type>.jsonl`，记录替换或拒绝时新旧样本的熵和距离。

详细诊断文档：

    docs/experiments/E3_global_prototype_alignment_cache/smoke_tests/00_1_e3_v1_gpa_only_manual_full_diagnosis.md

## 2026-06-05：新增 E3 中心来源消融代码

新增两个 E3 中心来源变体：

### Center-B：Entropy-only center

模型文件：

    Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_gpa_entropy_only_center.py

Runner：

    Point-Cache/runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_gpa_entropy_only_center.py

脚本：

    Point-Cache/scripts/E3_global_prototype_alignment_cache/01_1_ulip_modelnetc_s2_zs_global_local_gpa_entropy_only_center_manual_full.sh

### Center-C：Entropy+GPA union center

模型文件：

    Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_gpa_entropy_gpa_union_center.py

Runner：

    Point-Cache/runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_gpa_entropy_gpa_union_center.py

脚本：

    Point-Cache/scripts/E3_global_prototype_alignment_cache/01_2_ulip_modelnetc_s2_zs_global_local_gpa_entropy_gpa_union_center_manual_full.sh

本次中心来源消融固定以下变量不变：

- 仍然采用顺序式 GPA Cache；
- 仍然采用 manual_full；
- 仍然运行 zs_global_local；
- 仍然暂不修改最终预测加权公式；
- 只改变 GPA 原型中心来源。

目标：

    判断 E3-V1-A 下降是否主要来自 GPA-only center 不稳定。

## 2026-06-05：完成 E3-V1 归总并进入 E3-V2 计划

E3-V1 顺序式 GPA Cache 已完成三种中心来源消融：

- GPA-only center：53.44
- Entropy-only center：52.43
- Entropy+GPA union center：53.01

对比 E2 原始 full Point-Cache：

- E2 baseline：54.00

结论：

    E3-V1 三种中心来源均未超过 baseline。
    继续在顺序式关系下更换中心来源意义不大。

新增归总文档：

    docs/experiments/E3_global_prototype_alignment_cache/smoke_tests/01_e3_v1_sequential_gpa_center_source_summary.md

下一步进入 E3-V2：

    并列式 Global Prototype-Alignment Cache

E3-V2 仍保留三种中心来源消融：

- GPA-only center
- Entropy-only center
- Entropy+GPA union center

## 2026-06-05：实现 E3-V2 并列式 GPA Cache 三种中心来源

新增 E3-V2 三个模型文件：

- `model_with_hierarchical_caches_parallel_gpa_gpa_only_center.py`
- `model_with_hierarchical_caches_parallel_gpa_entropy_only_center.py`
- `model_with_hierarchical_caches_parallel_gpa_entropy_gpa_union_center.py`

新增 E3-V2 三个 runner：

- `run_e3_ulip_modelnetc_s2_parallel_gpa_gpa_only_center.py`
- `run_e3_ulip_modelnetc_s2_parallel_gpa_entropy_only_center.py`
- `run_e3_ulip_modelnetc_s2_parallel_gpa_entropy_gpa_union_center.py`

新增 E3-V2 脚本：

- `02_1_ulip_modelnetc_s2_zs_global_local_parallel_gpa_gpa_only_center_manual_full.sh`
- `02_2_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_only_center_manual_full.sh`
- `02_3_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_gpa_union_center_manual_full.sh`

E3-V2 核心变化：

    Global Entropy Cache 和 GPA Cache 并列更新，
    GPA Cache 不再依赖 Global Entropy Cache 准入。
