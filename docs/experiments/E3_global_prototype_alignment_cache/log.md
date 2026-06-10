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

## 2026-06-05：记录 E3-V3 GPA Cache 初始化改进方案

为解决 E3-V2-C 中仍然存在的 GPA Cache 初始化不稳定问题，记录三类初始化改进方案：

- Init-A：先用 Global Entropy Cache 初始化中心；
- Init-B：延迟 local cache 写入；
- Init-C：候选池初始化。

当前建议优先实现 Init-C：

    每类先收集 2K 或 3K 个候选样本；
    使用 Global Entropy Cache + GPA candidate pool 构造联合中心；
    根据熵和距离筛出 K 个样本进入 GPA Cache；
    只有最终选中的 K 个样本的局部特征进入 local cache。

详细文档：

    docs/experiments/E3_global_prototype_alignment_cache/initialization_strategies/e3_v3_gpa_cache_initialization_strategies.md

## 2026-06-05：实现 E3-V3-A 候选池初始化

新增模型文件：

    Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_parallel_gpa_candidate_pool_init.py

新增 runner：

    Point-Cache/runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_parallel_gpa_candidate_pool_init.py

新增脚本：

    Point-Cache/scripts/E3_global_prototype_alignment_cache/03_1_ulip_modelnetc_s2_zs_global_local_parallel_gpa_candidate_pool_init_manual_full.sh

方法设定：

    parallel GPA
    + candidate pool initialization
    + Entropy Cache and GPA candidate pool union center
    + candidate pool size = 2K
    + final GPA size = K
    + local cache 只写入最终筛出的 K 个样本

## 2026-06-06：暂停 Init-C 第一版候选池初始化

E3-V3-A Init-C 第一版在 add_global_2 上出现异常下降。

预构建阶段显示候选池机制生效：

- entropy cache total = 89
- gpa cache total = 87
- gpa local cache total = 87
- gpa candidate pool total = 2

但正式测试阶段累计准确率明显偏低，实验被手动停止。

当前判断：

    不能直接说明 Init-C 思路失败，
    但第一版实现或筛选策略过于激进，需要暂停并诊断。

详细记录：

    docs/experiments/E3_global_prototype_alignment_cache/initialization_strategies/03_1_init_c_candidate_pool_failed_attempt_analysis.md

下一步转向更保守的 Init-A：

    Entropy-bootstrap initialization with Entropy+GPA union center。

## 2026-06-06：新增 E3-V3-C 候选池距离初始化 GPA-Cache 消融说明

新增文档：

`docs/experiments/E3_global_prototype_alignment_cache/initialization_strategies/E3-V3-C_candidate_pool_distance_initialization_ablation.md`

当前首跑版本：

`E3-V3-C1-Ub = Candidate-only center + Distance-only update`

该版本使用每类 2K 候选池构造临时中心，选择距离中心最近的 K 个样本进入 GPA-Cache。GPA-Cache 满后，不使用熵，只根据新样本是否比当前缓存中离中心最远样本更近来决定是否替换。每次替换后必须同步更新 local cache，并立即重算 GPA-Center。

最终预测公式和 local cache 权重暂时不改。

## 2026-06-06：E3-V3-C1-Ub 结果分析完成

新增结果分析文档：

`docs/experiments/E3_global_prototype_alignment_cache/initialization_strategies/E3-V3-C1-Ub_result_analysis.md`

E3-V3-C1-Ub，即“候选池中心 + 无熵距离更新方法”，在 ULIP × ModelNet-C severity=2 × 7 corruption smoke test 上取得 53.39 平均准确率。该结果高于 E3-V3-B Entropy-bootstrap 初始化方法的 53.25，但低于 E2 原始 full Point-Cache baseline 的 54.00，也低于 E3-V2-C 的 54.04。

结论：候选池初始化方向没有被完全否定，但无熵纯距离更新规则不够稳定。下一步同时进行 E3-V3-C1-Ua2 和 E3-V3-C1-Ua1。

## 2026-06-06：记录 E3-V3-C1 系列“去噪有效、几何变化失效”分析

新增分析文档：

`docs/experiments/E3_global_prototype_alignment_cache/analysis/E3-V3-C1_series_result_analysis_noise_vs_geometry.md`

核心结论：

E3-V3-C1 系列方法的正收益主要集中在 add_global 和 add_local。该现象说明候选池单中心 GPA-Cache 更像是一种外点噪声过滤机制：add_global 和 add_local 会向点云中加入离群点，但原始物体主体结构仍然存在，因此距离类别中心最近的样本更可能是噪声影响较小的样本，进入 GPA-local-cache 后能够提升 additive corruption 下的鲁棒性。

但该机制对 dropout、rotate、scale、jitter 等 corruption 不稳定。原因是这些 corruption 改变的是结构完整性、整体几何分布或局部 patch 稳定性，而不是简单加入外点噪声。单中心 GPA 追求类内紧凑性，可能牺牲 local cache 的覆盖度和多样性，因此对几何结构变化失效。

当前 C1 系列最优版本为 E3-V3-C1-Ua1，平均准确率 54.02，基本追平 E2 baseline，但仍略低于 E3-V2-C 的 54.04。下一步查看 C2-Ua1，即中心来源改为 candidate pool + Entropy Cache，更新规则保留 Ua1。

## 2026-06-07：补充 E3 单中心原型方法的适用边界分析

更新文档：

`docs/experiments/E3_global_prototype_alignment_cache/analysis/E3_overall_result_analysis_and_E4_motivation.md`

补充重要发现：

E3 阶段的多种改进，包括 GPA-only center、Entropy-only center、Entropy/GPA union center、Entropy-bootstrap initialization、Candidate-only center、Candidate+Entropy center、替换最高熵样本、替换最远样本，本质上大多仍然属于“改进单中心原型”的方法。

这类方法对 add_global 和 add_local 这类添加型外点噪声有效，因为原始物体主体结构仍然存在，外点会把受污染严重的样本特征拉离类别主体中心，因此选择更靠近原型中心的样本可以起到过滤外点噪声、净化 cache 的作用。

但面对 dropout、rotate、scale、jitter 等几何结构变化时，单中心原型方法会失效或收益不稳定。原因是这些损坏改变的是结构完整性、整体几何分布、局部 patch 稳定性或类内结构模式。此时样本远离中心不一定代表它是错误样本或脏样本，也可能代表该类别在几何变化下的正常模式。

该发现可以作为后续论文中的一个重要分析点：单中心原型对齐方法适合处理添加型噪声，但难以覆盖几何结构变化。E4 将据此引入类别概率分布，用“是否符合类别分布”替代“是否靠近单一中心”作为 cache 更新判断标准。

## 2026-06-07：提出 E3-V2-TextProto-C 文本原型增强中心实验

新增说明文档：

`docs/experiments/E3_global_prototype_alignment_cache/text_prototype_center/E3-V2-TextProto-C_text_visual_prototype_center_plan.md`

核心思想：

回到当前 E3 阶段最优的 E3-V2-C 框架，在原有 Entropy Cache + GPA-Cache 构造的视觉联合中心基础上，引入类别 Text Prototype 作为语义锚点，构造文本-视觉联合中心。

第一版建议命名：

    E3-V2-TextProto-C-w0.7v0.3t

第一版设置：

    visual_center = mean(EntropyCache[c] ∪ GPACache[c])
    text_center = Text Prototype[c]
    final_center = normalize(0.7 * visual_center + 0.3 * text_center)

GPA-Cache 初始化、低熵门控、替换最高熵样本、local cache 同步和最终预测公式均暂时沿用 E3-V2-C。

需要特别说明：0.7 visual + 0.3 text 不是原文固定权重，而是第一版保守启发式初值。原因是 E3-V2-C 的视觉中心已经验证有效，应保持视觉主导；Text Prototype 只作为语义锚点参与。后续需要通过不同文本权重进行消融验证。

## 2026-06-08：补充 E3-V2-TextProto-Guard-C 实验计划

新增说明文档：

`docs/experiments/E3_global_prototype_alignment_cache/text_prototype_center/E3-V2-TextProto-Guard-C_plan.md`

当前结论：

E4-A 的小样本类别分布方法不稳定；E3-V2-TextProto-C 的文本-视觉向量融合虽然在 dropout_global、dropout_local 等结构缺失场景中有帮助，但会拉偏视觉中心，伤害 add_local、rotate、scale、jitter 等依赖视觉去噪的场景。因此，下一步不再把 Text Prototype 与 visual center 直接融合成一个中心，而是提出 E3-V2-TextProto-Guard-C。

E3-V2-TextProto-Guard-C 的核心规则：

    entropy_new < entropy_high
    and
    (
        d_visual_new < d_visual_high
        or
        (
            d_visual_new <= d_visual_high * (1 + rho_visual)
            and
            d_text_new < d_text_high
        )
    )

第一版设置：

    rho_visual = 0.05

该方法保留 E3-V2-C 的视觉去噪分支，同时增加 Text Prototype 语义保护分支，目标是同时保留 add_global/add_local/jitter 的视觉去噪能力，并继承 Text Prototype 在 dropout/scale 上的结构缺失收益。

实现代码时需要注意：

    不再直接复制旧 runner 后字符串替换；
    应新写干净版本或显式重构公共逻辑；
    必须检查 build 后 gpa_cache 不被清空；
    必须检查 GPA-Cache 和 GPA-local-cache 同步；
    必须记录 visual branch 与 text guard branch 的替换事件。

## 2026-06-09：补做 E3-V2-Cb，即 2+C+b 更新规则消融

当前复盘发现，E3-V2 已完成以下组合：

```text
2 + A + a
2 + B + a
2 + C + a
```

但尚未在原始 E3-V2 并列式设置下补做：

```text
2 + C + b
```

其中：

```text
2 = 并列式 GPA Cache；
C = Entropy Cache 与 GPA Cache 并集构造原型中心；
b = 低熵门控 + 新样本距离小于当前最远样本距离 + 替换最远样本。
```

因此新增独立补实验：

```text
E3-V2-Cb
```

对应代码：

```text
Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_parallel_gpa_entropy_gpa_union_center_replace_farthest.py
Point-Cache/runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_parallel_gpa_entropy_gpa_union_center_replace_farthest.py
```

对应脚本：

```text
Point-Cache/scripts/E3_global_prototype_alignment_cache/02_4_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_gpa_union_center_replace_farthest_manual_full.sh
```

结果目录：

```text
Point-Cache/results/E3_global_prototype_alignment_cache/02_4_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_gpa_union_center_replace_farthest_manual_full
```

本次实现保持以下内容不变：

- 仍然使用 ULIP；
- 仍然使用 ModelNet-C severity=2 的 7 corruption smoke test；
- 仍然使用 manual_full；
- 仍然使用 zs_global_local；
- 仍然使用 Global Entropy Cache 计算 global logits；
- 仍然使用 GPA-controlled local cache 计算 local logits；
- 最终 logits 公式不变。

本次只改变 GPA-Cache 满后的替换规则：

```text
E3-V2-C / 2+C+a:
    entropy_new < entropy_high
    and
    d_new < d_high
    -> replace highest-entropy sample

E3-V2-Cb / 2+C+b:
    entropy_new < entropy_high
    and
    d_new < d_far
    -> replace farthest-to-center sample
```

其中：

```text
entropy_high = GPA-Cache[c] 中最高熵样本的熵；
d_high       = 最高熵样本到原型中心的距离；
d_far        = GPA-Cache[c] 中最远样本到原型中心的距离；
d_new        = 新样本到原型中心的距离。
```

本次补实验同时增强了事件日志，`gpa_replacement_events_*.jsonl` 会记录：

- 新样本熵；
- 新样本到中心距离；
- 当前最高熵样本索引、熵、距离；
- 当前最远样本索引、熵、距离；
- 最高熵样本是否同时也是最远样本；
- 当前 GPA-Cache 内所有样本的熵和距离。

这样后续可以直接统计：

```text
高熵样本是否通常离原型中心更远？
低熵样本是否通常离原型中心更近？
替换最高熵样本和替换最远样本是否在多数情况下等价？
```

注意：

旧 E3-V2-C 代码中存在一个历史实现风险：测试阶段注释写“继承预构建 GPA global cache”，但实际曾将 `gpa_cache` 置空，仅保留 `gpa_local_cache`。对于 `2+C+b`，如果 GPA global cache 和 GPA-local-cache 不一一对应，则“替换最远样本”的索引不可解释。因此本次 E3-V2-Cb 明确继承 build 阶段的 GPA global cache，并在替换前检查 GPA-Cache 与 GPA-local-cache 长度一致。
