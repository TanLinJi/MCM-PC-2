# MCM-PC 项目协作规范、用户偏好、路径命名、实验记录与 Git 工作流完整归档

更新时间：2026-06-06  
项目根目录：`/root/autodl-tmp/MCM-PC-2`  
核心代码库：`/root/autodl-tmp/MCM-PC-2/Point-Cache`  
默认运行环境：`conda activate mcmpc`  
默认语言：中文  

---

## 0. 本文件目的

这份文档记录项目推进过程中的协作规则、用户偏好、路径命名、实验编号、文档要求、脚本要求、Git 习惯和常见 bug 处理方式。它不是实验内容总结，而是“如何继续和用户协作推进这个项目”的操作手册。

目标是：任何接手者不需要重新读全部聊天记录，也能知道：用户希望如何命名实验；文件应该放在哪里；文档应该写到什么程度；什么时候能 git，什么时候不能 git；出 bug 时应该如何解释和修复；哪些事情用户已经明确不喜欢；哪些实验只是检查，不应该单独编号；当前项目最重要的上下文是什么。

---

## 1. 用户沟通偏好

### 1.1 默认中文

用户默认使用中文交流。除非用户明确要求英文，否则回复、注释、文档说明都应使用中文。

### 1.2 用户希望“解释清楚背景”

用户不满意只给结论。用户明确指出：提到问题时，必须讲问题背景；提到解决方案时，必须讲为什么这样解决；讲实验时，必须讲实验目的、设定、路径、结果、结论和下一步；不能只写“结果不好”，要解释为什么可能不好；不能只写“等筛选完成”，必须说什么时候筛选、怎么筛选、判定准则是什么。

因此所有后续回答都应遵循：

```text
背景 -> 问题 -> 方案 -> 实现 -> 预期 -> 失败怎么办
```

### 1.3 用户不喜欢“挤牙膏式回答”

用户曾明确批评：问到 Init-B 时，回答像挤牙膏，问什么才说什么。后续介绍任何方案时必须一次性讲完整：触发条件、数据结构、更新规则、筛选准则、与其他方案区别、预期收益、风险、如果失败如何分析。

### 1.4 用户希望及时承认错误

如果之前判断错了，要直接承认。例如：runtime_gpa_cache 的问题；Init-A 补丁没有写进去；参数错位导致 `pred` 变成 list；`entropy_cache[pred][-1]` 取错导致 `_loss_value` 收到 list。不要强行解释成“实验现象”。先定位 bug，再给修复。

### 1.5 用户更看重整体平均准确率

用户强调：某一个 corruption 上的特殊现象可以记录，但当前关注整体分类准确率，不要过度围绕某个损坏类型展开主结论。例如 add_global 或 add_local 可作为线索，但不能替代平均准确率判断。

---

## 2. 实验推进总原则

### 2.1 先纵向跑通，再横向补实验

这是用户多次明确的核心原则。当前 E0-E3 的主线属于纵向推进：

```text
E0 复现 Point-Cache
E1 验证文本增强 zero-shot
E2 验证文本收益迁移到 Point-Cache
E3 改造缓存构造，引入 GPA Cache
```

当前多数实验只是 smoke test。跑通主线后，再回头补横向：全 corruption、全 severity、更多 backbone、更多数据集、文本模板数量消融、cache setting 消融、权重消融、初始化消融、negative cache 消融、多中心原型消融。后续回复中必须持续提醒这个原则，不要把 smoke test 结果说成最终完整结论。

### 2.2 检查不单独作为实验编号

用户明确说过：以后这种检查的，不单独作为实验编号。例如 `py_compile`、`bash -n`、grep 检查、smoke run 的语法验证、单 corruption 快速试跑。这些可以作为文档记录，但不应新建 E 编号。实验编号只给真正的实验设定。

### 2.3 实验命名必须体现任务含义

用户非常重视命名。命名应能从文件名看出阶段编号、backbone、数据集、severity、cache setting、方法核心、文本来源、是否为 smoke/full。例如：

```text
03_2_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_bootstrap_init_manual_full.sh
```

这个名字体现：E3 第 03 组、ULIP、ModelNet-C、severity=2、zs_global_local、parallel GPA、entropy bootstrap init、manual_full。

### 2.4 不要滥用英文术语

用户曾说不要一上来就是英文名，要先解释中文含义，再给英文名。例如先说“全局原型对齐缓存”，再说英文 `Global Prototype-Alignment Cache`，简称 GPA Cache。

---

## 3. 项目目录规范

根目录：

```text
/root/autodl-tmp/MCM-PC-2
```

所有命令默认从这里开始：

```bash
cd /root/autodl-tmp/MCM-PC-2
```

Conda 环境：

```bash
conda activate mcmpc
```

Point-Cache 根目录：

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache
```

实验 runner 目录：

```text
Point-Cache/runners/E1_text_prototype_enhancement
Point-Cache/runners/E2_text_prototype_transfer_to_pointcache
Point-Cache/runners/E3_global_prototype_alignment_cache
```

实验脚本目录：

```text
Point-Cache/scripts/E1_text_prototype_enhancement
Point-Cache/scripts/E2_text_prototype_transfer_to_pointcache
Point-Cache/scripts/E3_global_prototype_alignment_cache
```

结果目录：

```text
Point-Cache/results/E1_text_prototype_enhancement
Point-Cache/results/E2_text_prototype_transfer_to_pointcache
Point-Cache/results/E3_global_prototype_alignment_cache
```

文档目录：

```text
docs/experiments/<experiment_name>
docs/experiments/E3_global_prototype_alignment_cache/smoke_tests
docs/experiments/E3_global_prototype_alignment_cache/initialization_strategies
```

---

## 4. 文档规范

### 4.0 长期维护入口

当前项目新增两个长期维护入口：

```text
docs/experiments/narrative/e0_e3_research_narrative.md
docs/project/conventions/user_preferences_workflow_conventions.md
```

后续每当实验进展、bug 修复、结果判断、下一步研究计划发生变化时，应同步更新 `docs/experiments/narrative/` 下的实验叙事文档。每当用户提出新的协作偏好、命名规则、脚本规范、HTML 风格、Git 规则或分析要求时，应同步更新 `docs/project/conventions/` 下的规范文档。

如果单个文件过长，可以在同一目录下按阶段或主题拆分，并更新该目录的 `README.md` 索引。

### 4.1 每个阶段至少要有三个文档

建议每个实验目录维护：

```text
plan.md
log.md
analysis.md
```

`plan.md` 记录计划、设计、待做事项；`log.md` 按日期记录做了什么、改了什么、跑了什么；`analysis.md` 记录结果分析、对比表、结论、失败原因。

### 4.2 特殊问题要单独建文档

例如：

```text
smoke_tests/02_e3_v2_parallel_gpa_center_source_analysis.md
initialization_strategies/e3_v3_gpa_cache_initialization_strategies.md
initialization_strategies/03_1_init_c_candidate_pool_failed_attempt_analysis.md
```

用户希望这些文档能让陌生人读懂，所以不要只写“见聊天记录”。

### 4.3 文档必须包含的内容

每个实验说明文档至少包含：实验背景、研究问题、与上一个实验的关系、方法设计、文件路径、脚本路径、参数设置、实验结果、与 baseline 对比、分 corruption 表、结果解释、失败或异常、后续补实验计划、当前结论的可信边界。

### 4.4 文档不能只写“结果”

不能只写“Init-C 失败”。应该写清楚：Init-C 试图解决前 K 个样本无筛选进入 GPA Cache 的问题；它通过候选池收集 2K 个样本并按熵和距离筛 K 个；当前第一版 add_global_2 准确率异常下降；这可能来自 local cache 覆盖不足、筛选规则过于中心化或状态机复杂；因此暂时暂停，不否定方向，转向更保守的 Init-A。

### 4.5 文档要写“后续怎么补”

每个阶段都要区分：当前 smoke test 已做什么；完整实验还欠什么；未来怎么补。

### 4.6 HTML 说明文档风格

以后用户要求编写的所有说明类 HTML 文档，默认采用：

```text
docs/reports/2026-05-17_task_specification.html
```

的暗色侧边栏风格，包括整体布局、色调、目录导航、卡片密度和信息组织方式。除非用户明确要求更换风格，不再新建另一套浅色或营销式页面风格。

---

## 5. Git 工作流偏好

### 5.1 不要擅自提交

用户明确说过：以后先不要着急提交，会提醒。不要主动说“现在提交”；不要在没有用户要求时给 git commit/push 命令；可以建议“检查没问题后再由你决定是否 git”。只有用户说“来 git 一下”“先提交”“push”时，才给 git 命令。

### 5.2 提交前必须检查

常规检查：

```bash
python -m py_compile <runner/model files>
bash -n <script files>
git status --short
git diff --cached --name-only
```

### 5.3 提交信息应清楚

例如：

```bash
git commit -m "feat: add and analyze E3 V2 parallel GPA ablation"
```

不要用含糊 commit message。

### 5.4 结果文件一般不提交，文档和代码提交

除非用户明确要求，通常提交 runner、scripts、docs、README。大型结果目录、logs、模型权重、数据文件不应随意提交。

---

## 6. 脚本规范

脚本最后一个参数通常是物理 GPU：

```bash
bash xxx.sh 0
bash xxx.sh 1
```

脚本内部用：

```bash
"${1:-0}"
```

E3 V2 使用公共脚本：

```text
02_run_ulip_modelnetc_s2_parallel_gpa_center_source_ablation_common.sh
```

具体实验脚本调用公共脚本并传入 EXP_ID、RUNNER、PROMPT_SOURCE、METHOD_FULL、PURPOSE、GPU。这种结构用户认可。

如果某次失败后要重新跑，建议：

```bash
rm -rf Point-Cache/results/<exp>/<exp_id>
```

避免旧日志混入新结果。

用户有两张卡，可以在不同终端跑多个脚本。注意每个脚本指定不同物理 GPU，不要让两个脚本写同一个 result dir，summary 和 log 文件名要带 EXP_ID 和 timestamp。

---

## 7. 命名规范

实验阶段命名：

```text
E1_text_prototype_enhancement
E2_text_prototype_transfer_to_pointcache
E3_global_prototype_alignment_cache
```

文本方法命名：

```text
manual_full
manual_3d
llm_only
manual_full_llm_fusion
```

用户不希望用 `manual_full_add_llm`，因为这不体现融合。

GPA 方法命名：

```text
sequential_gpa
parallel_gpa
gpa_only_center
entropy_only_center
entropy_gpa_union_center
candidate_pool_init
entropy_bootstrap_init
delayed_local_cache_writing
```

E3 版本含义：

```text
E3-V1：顺序式 GPA Cache
E3-V2：并列式 GPA Cache
E3-V3：GPA Cache 初始化机制改进
V3-A：candidate pool initialization（Init-C，已暂停）
V3-B：entropy bootstrap initialization（Init-A，正在调试）
```

---

## 8. 重要代码文件

E3 V2 稳定基础文件：

```text
Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_parallel_gpa_entropy_gpa_union_center.py
Point-Cache/runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_parallel_gpa_entropy_gpa_union_center.py
```

Init-C 文件：

```text
Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_parallel_gpa_candidate_pool_init.py
Point-Cache/runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_parallel_gpa_candidate_pool_init.py
Point-Cache/scripts/E3_global_prototype_alignment_cache/03_1_ulip_modelnetc_s2_zs_global_local_parallel_gpa_candidate_pool_init_manual_full.sh
```

状态：第一版出现准确率异常下降，已暂停。

Init-A 文件：

```text
Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_parallel_gpa_entropy_bootstrap_init.py
Point-Cache/runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_parallel_gpa_entropy_bootstrap_init.py
Point-Cache/scripts/E3_global_prototype_alignment_cache/03_2_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_bootstrap_init_manual_full.sh
```

状态：正在调试。最新应修复：

```python
worst_ent = _loss_value(entropy_cache[pred][-1][1])
```

而不是：

```python
worst_ent = _loss_value(entropy_cache[pred][-1])
```

---

## 9. 常见 bug 与处理规范

### 9.1 `ValueError: not enough values to unpack (expected 6, got 5)`

Point-Cache hierarchical cache runner 需要 `get_logits` 返回 6 个值：`pc_feats, patch_centers, clip_logits, loss, prob_map, pred`。如果文本增强路径返回 5 个值，就会报错。解决是统一 `get_logits` 返回协议，确保 hierarchical cache 需要的局部特征和概率图都存在。

### 9.2 `TypeError: unhashable type: 'list'`

多次发生在 `_update_gpa_cache` 调用参数错位。例如函数签名有 `gpa_candidate_pool`，但调用时漏掉它，导致 `pred <- global_item`，此时 `pred` 是 list，用作 dict key 就报 unhashable。解决：检查函数定义和所有调用，确保参数顺序一致，用 grep：

```bash
grep -n "_update_gpa_cache(" -A14 <file>
```

### 9.3 `IndexError: list index out of range`

Init-C 中出现 `gpa_cache[pred]` 是空列表，但代码以为 formal GPA 已形成。解决：不能只判断 `pred in gpa_cache`，还要判断 `len(gpa_cache.get(pred, [])) > 0`。空 formal cache 应回收到 candidate pool 或直接 reject，并记录 stats。

### 9.4 `_loss_value` 收到 list

Init-A 中 `entropy_cache[pred][-1]` 是整个 item `[pc_feats, loss]`，不是 loss。解决：取第二项：

```python
_loss_value(entropy_cache[pred][-1][1])
```

### 9.5 补丁没写进文件

曾经 patch 脚本在 `p.write_text` 之前 raise，导致前面看似改了但文件没有写入。解决：patch 后必须 grep 检查方法标识；必须 `py_compile`；必须 grep helper 函数是否存在；如果正则不可靠，先打印真实函数结构再整体替换。

---

## 10. 用户关于“方案说明”的要求

以后介绍任何方案，必须包含：这个方案解决哪个旧问题；旧问题为什么会发生；新方案的数据结构；什么时候触发；怎么判断样本是否进入缓存；如果熵和距离冲突怎么办；local cache 怎么同步；full cache 后怎么替换；失败可能说明什么；需要记录哪些 stats。

示例：Init-B 不能只说“等候选样本筛选完成”，必须说：候选池达到 2K 时筛选；build 结束时若候选数 >= K 则退化筛选；筛选按 `entropy_rank + distance_rank`；只将选出的 K 个 local item 写入 local cache。

---

## 11. 结果分析规范

每个分析文档必须有总对比表。用户多次要求“特别是总对比表”。每个分析文档都应给一个全局表，如“方法 / 关系 / 中心来源 / 平均准确率 / 相对 baseline”。还必须有分项表，至少按 `add_global, add_local, dropout_global, dropout_local, rotate, scale, jitter` 展开。

不要只看单项。如果某个 corruption 特别好或坏，可以记录，但不能替代整体平均。失败也要写成结果说明：失败时的设定、已经生效的机制、异常数值、可能原因、为什么暂停、下一步替代方案。Init-C 第一版就是典型例子。

---

## 12. 当前用户明确的研究判断

1. E3 不应该一开始就引入 BayesMM 做法。
2. 当前先不引入文本原型，先解决 GPA Cache 初始化。
3. 文本原型可作为后续 Center-D。
4. 如果视觉内部还没做好，不要急着加文本原型掩盖问题。
5. E3 最小测试阶段可以不改最终预测公式，但后续可能需要改。
6. 顺序式 GPA 效果不好，已转向并列式。
7. 并列式 union center 当前最好，但收益太小。
8. Init-C 第一版暂停，不直接否定候选池思想。
9. Init-A 当前优先调试。
10. 之后可能做 Init-B，且必须讲清楚筛选机制。

---

## 13. 当前必须避免的错误

不要把 smoke test 写成最终结论；不要只给代码不给背景；不要只说“筛选”不定义筛选规则；不要在用户没提醒时 git；不要忽略 local cache 与 global/GPA cache 的同步；不要在参数错位时继续分析准确率；不要把 bug 当成实验失败；不要过度分析单一 corruption；不要一上来用英文名替代中文解释；不要说“我们已经修复”但不做 grep/py_compile 检查。

---

## 14. 当前下一步操作建议

当前最直接下一步是修复 Init-A：

```bash
cd /root/autodl-tmp/MCM-PC-2

python - <<'PY'
from pathlib import Path
p = Path("Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_parallel_gpa_entropy_bootstrap_init.py")
text = p.read_text(encoding="utf-8")
text = text.replace(
    "worst_ent = _loss_value(entropy_cache[pred][-1])",
    "worst_ent = _loss_value(entropy_cache[pred][-1][1])",
)
p.write_text(text, encoding="utf-8")
print("patched", p)
PY

python -m py_compile \
  Point-Cache/runners/E3_global_prototype_alignment_cache/model_with_hierarchical_caches_parallel_gpa_entropy_bootstrap_init.py \
  Point-Cache/runners/E3_global_prototype_alignment_cache/run_e3_ulip_modelnetc_s2_parallel_gpa_entropy_bootstrap_init.py
```

然后清理结果目录并跑：

```bash
rm -rf Point-Cache/results/E3_global_prototype_alignment_cache/03_2_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_bootstrap_init_manual_full

bash Point-Cache/scripts/E3_global_prototype_alignment_cache/03_2_ulip_modelnetc_s2_zs_global_local_parallel_gpa_entropy_bootstrap_init_manual_full.sh 0
```

观察 `entropy cache total`、`gpa cache total`、`gpa local cache total` 和 `add_global_2 final accuracy`。如果不崩，再跑完整 7 corruption。

---

## 15. 专家自查：陌生专家读完还可能问什么？

### 15.1 为什么实验文件名里 `03_2` 是 Init-A，而 `03_1` 是 Init-C？

因为 E3-V3 中先尝试了 Init-C 候选池初始化，编号为 `03_1`；该第一版异常下降后暂停，再尝试更保守的 Init-A，编号为 `03_2`。编号保留历史，不回滚重命名，以免破坏结果目录和文档一致性。

### 15.2 为什么现在不直接删除 Init-C 文件？

因为 Init-C 失败分析有价值，代码也可作为后续改进参考。但要在文档中标记“第一版暂停”，避免误用。

### 15.3 如果后续读者想继续 Init-C，该怎么做？

不要直接跑当前第一版。先检查 selected/rejected 事件日志；比较 local cache 覆盖；加入多样性约束；或改成只用候选池初始化 center，不减少 local cache 覆盖；重新设计 fallback。

### 15.4 如果后续读者想做论文写作，应注意什么？

论文里不要直接写“参考 MCP 源码”。可以写成：受低熵样本不一定保证类内紧凑性的启发，引入基于原型距离的缓存更新约束。内部文档可以详细写 MCP 对我们的启发。

### 15.5 当前文档是否已经覆盖用户指出的两个问题？

已覆盖：每个问题补充了背景和解决办法；明确写了当前是纵向 smoke test，后续要回补横向实验；对 Init-A/B/C 细节、触发条件、筛选准则和失败分析做了补充；加入了“专家自查”部分。

---

## 16. 最终提醒

这个项目的协作关键不是只给命令，而是让实验逻辑可追溯。任何新实验都应在跑之前说明：为什么做、怎么做、和已有实验有什么区别、如果成功说明什么、如果失败说明什么。跑完后必须写文档，再考虑 git。
