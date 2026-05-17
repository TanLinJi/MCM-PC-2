# MCM-PC 用户偏好与工作规则总结

生成时间：2026-05-15 07:28:38

> 本文档整理当前 MCM-PC / Point-Cache 项目中已经形成的用户偏好、沟通规则、实验规则、Git 规则、脚本命名规则、文档归档规则和后续协作注意事项。  
> 后续所有实验、代码修改、文档总结和论文写作都应优先遵守本文档。

---

## 1. 总体协作偏好

### 1.1 需要主动推进，但不能跳步

用户希望助手能主动判断下一步应该做什么，但不能在关键实验、关键修改前直接跳过分析。

尤其在方法设计时，需要遵循：

1. 先解释为什么要做；
2. 再给出数学或逻辑依据；
3. 再给出具体文件名和命令；
4. 先 quick test；
5. 再 full run；
6. 跑完后先分析结果；
7. 再归档和 commit。

不应该直接给一个未经解释的大段代码，让用户不知道为什么要这样做。

### 1.2 不要反复索要已经有的信息

用户明确表示：如果代码、文件、路径、实验背景在当前项目中已经存在，助手不应该反复向用户索要。

后续应优先根据已有上下文判断，例如：

- Point-Cache 路径；
- runner 文件位置；
- scripts/recur-pc 位置；
- logs/recur-pc 位置；
- 已经运行过的实验；
- 当前 branch；
- 已有实验编号；
- 服务器 GPU/CPU 环境；
- 已经形成的实验结论。

如果确实缺少关键信息，才简短询问。

### 1.3 解释要直接、具体、能落地

用户更喜欢：

- 直接告诉他“改哪个文件、哪个位置、怎么改”；
- 给出完整命令；
- 给出完整文件；
- 给出下一步操作顺序；
- 给出明确判断：该不该跑、该不该 commit、该不该进入下一阶段。

不喜欢：

- 只讲概念；
- 只说“你可以考虑”；
- 解释过于空泛；
- 输出杂乱无章；
- 不区分实验类型、文档类型、代码类型。

---

## 2. 项目方向与论文总体偏好

### 2.1 当前项目主线

项目暂定方向：

**MCM-PC / Multi-Cache Matrix for 3D Point Cloud Test-Time Adaptation**

更具体地说，当前项目从 Point-Cache 出发，目标是构建一个面向 3D 点云视觉-语言模型的 test-time adaptation / test-time generalization 方法。

当前最重要的路线已经逐渐从：

```text
positive cache admission 改进
```

转向：

```text
global-local conflict-aware negative / boundary suppression
```

### 2.2 未来完整框架仍然是 Multi-Cache Matrix

用户强调：不能只总结到 E4 或 E6，后续还有完整的 Multi-Cache Matrix。

因此后续文档中应始终区分：

```text
当前阶段：E4-CANC conflict-aware negative suppression
长期目标：MCM-PC Multi-Cache Matrix
```

Multi-Cache Matrix 未来应包含：

- global positive cache；
- local positive cache；
- negative cache；
- boundary cache；
- conflict cache；
- reliability cache；
- text prototype uncertainty；
- semantic confusion / pairwise boundary matrix。

### 2.3 论文写作偏好

当产生新的论文想法或核心卖点时，用户希望助手：

1. 用英文写出论文表述；
2. 明确标注应该放在哪个部分；
3. 例如：
   - Abstract；
   - Introduction；
   - Related Work；
   - Method；
   - Experiments；
   - Ablation；
   - Limitation。

示例格式：

```markdown
### [Method]
...
```

或：

```markdown
### [Experiment / Ablation]
...
```

---

## 3. 实验编号与命名规则

### 3.1 E 是 Experiment，不是 Stage

为了避免文档混乱，后续应明确：

```text
Stage = 项目大阶段
E = 具体实验编号
```

例如：

| 编号 | 含义 |
|---|---|
| E1-BASE | 原始 Point-Cache baseline reproduction |
| E2-EMR | Entropy-Margin Reliability |
| E3-GLC | Global-Local Consistency |
| E4-CANC | Conflict-Aware Negative Cache |
| E5-MCM | 未来 Multi-Cache Matrix 完整框架 |

### 3.2 必须补上 E1

之前文档中直接从 E2 开始，会让用户困惑。

后续统一定义：

```text
E1-BASE = Original Point-Cache Baseline Reproduction
```

E1 不是新方法，而是原始 Point-Cache hierarchical baseline 的复现实验，是所有后续方法的对照组。

### 3.3 实验名称必须清楚

每一个实验都要有正式名称。

例如：

| 实验 | 正式名称 |
|---|---|
| E1-BASE | Original Point-Cache Hierarchical Cache Baseline |
| E2-EMR | Entropy-Margin Reliability |
| E3-GLC-v0 | Naive Global-Local Consistency Reliability |
| E3-GLC-v1 | Non-expansive Global-Local Consistency Gate |
| E4-CANC-DIAG-v2 | Conflict Signal Validation |
| E4-CANC-v0 | Conservative Conflict-Aware Negative Cache |
| E4-CANC-v1 | Conflict-Gated Negative Cache |
| E4-CANC-v2 | Soft Conflict Suppression |

---

## 4. 脚本命名规则

用户明确要求：实验脚本命名必须统一，不要同一个实验出现多种命名风格。

### 4.1 推荐命名格式

```text
run_<experiment>_<method/module>_<dataset>_<corruption_range>_<gpu_mode>.sh
```

例如：

```text
run_e4_canc_v1_hierarchical_modelnetc_all35_corruptions_dual_gpu.sh
run_e4_canc_v2_hierarchical_modelnetc_suffix2_corruptions_dual_gpu.sh
run_e1_base_hierarchical_modelnetc_all35_corruptions_dual_gpu.sh
```

### 4.2 命名中必须体现的信息

脚本名应体现：

1. 实验编号；
2. 方法版本；
3. 模型或模块；
4. 数据集；
5. corruption 范围；
6. GPU 设置。

例如：

```text
e4_canc_v1
hierarchical
modelnetc
all35_corruptions
dual_gpu
```

### 4.3 不要混用旧命名风格

用户不喜欢出现类似：

```text
run_e3_glc_modelnetc_all_corruptions_dual_gpu.sh
run_reliability_v2_glc_hierarchical_modelnetc_all_corruptions_gpu1.sh
```

这种同一个阶段命名风格不统一的问题。

后续应统一前缀和结构。

---

## 5. 日志与结果保存规则

### 5.1 日志必须保存

每个实验脚本都必须保存：

- 每个 corruption 的 `.log`；
- 每个 corruption 的 `.json`，如果 runner 支持；
- 汇总 `.csv`；
- 最终 combined summary。

日志统一放在：

```text
Point-Cache/logs/recur-pc/
```

### 5.2 logs 不能提交 Git

`logs/` 是实验输出，不应提交到 Git。

提交前必须检查：

```bash
git diff --cached --name-only | grep 'Point-Cache/logs' && echo "ERROR: logs staged" || echo "OK: no logs staged"
```

### 5.3 结果分析前必须确认 summary 有效

尤其 all35 实验必须检查：

1. 是否 35 行；
2. 是否全部 `status=OK`；
3. 是否没有错误 suffix；
4. 是否包含 `_0` 到 `_4`；
5. 是否没有 `_5`。

推荐检查命令：

```bash
cut -d, -f3 "$SUMMARY" | grep '_5' && echo "ERROR: still has _5" || echo "OK: no _5"
cut -d, -f3 "$SUMMARY" | grep '_0' | wc -l
```

---

## 6. ModelNet-C corruption 后缀映射规则

这个是当前项目中非常重要的坑。

### 6.1 正确映射

本地文件名后缀与论文 severity 的对应关系是：

| 本地后缀 | 论文理解 severity |
|---|---:|
| `_0` | severity 1 |
| `_1` | severity 2 |
| `_2` | severity 3 |
| `_3` | severity 4 |
| `_4` | severity 5 |

因此，正确 all35 是：

```text
7 corruption types × local suffix _0.._4 = 35
```

### 6.2 错误写法

错误写法是：

```text
_1, _2, _3, _4, _5
```

因为 `_5` 不存在，且会漏掉 `_0`。

### 6.3 文档表述必须谨慎

不能再随便写：

```text
add_global_2 = severity 2
```

应写：

```text
add_global_2 是本地后缀 _2 setting，对应论文 severity 3。
```

或者在工程阶段直接写：

```text
suffix _2
```

等论文表格再映射回 severity 3。

---

## 7. Git 提交规则

用户明确要求：**不同类型的改动不要混在一个 commit 里。**

### 7.1 推荐拆分方式

| 改动类型 | commit 示例 |
|---|---|
| runner 代码 | `exp: add e4 canc v2 soft conflict suppression runner` |
| 运行脚本 | `exp: add e4 canc v2 suffix2 runner script` |
| 文档归档 | `docs: record e4 canc v1 corrected all35 results` |
| 方法笔记 | `docs: update mcm-pc method roadmap` |
| 脚本修复 | `exp: fix e4 canc v1 all35 corruption suffix mapping` |

### 7.2 提交前检查

提交前应先看：

```bash
git status --short
git diff --cached --name-only
```

如果只想提交 runner，就只 add runner。

如果只想提交 scripts，就只 add scripts。

### 7.3 不要把文档、runner、脚本混成一个 commit

例如不要把：

```text
docs/stages/stage4_canc_diag.md
runners/model_with_hierarchical_caches_e4_canc_v1.py
scripts/recur-pc/run_e4_canc_v1_...
```

全部塞进一个 commit。

应拆成：

```text
exp: add e4 canc v1 runner
exp: add e4 canc v1 runner scripts
docs: record e4 canc v1 results
```

---

## 8. 实验流程规则

### 8.1 不要直接 full run

每个新 runner 或新脚本都要先：

1. `python -m py_compile runner.py`
2. `bash -n script.sh`
3. `QUICK_TEST=1 bash script.sh`
4. 检查 quick test summary
5. 再 full run。

### 8.2 quick test 的作用

quick test 不是为了证明方法有效，而是为了确认：

- runner 能跑；
- import 没问题；
- 参数没问题；
- summary 能生成；
- log 能保存；
- GPU 设置没问题；
- 没有 wandb 报错。

### 8.3 full run 后先分析，再 commit 文档

实验结果跑完后：

1. 先发 summary；
2. 分析；
3. 写阶段文档；
4. 再 commit 文档。

不要先写结论再分析。

---

## 9. 数学推导偏好

用户明确要求：**不要每次都靠实验试错，方法设计前要从数学上分析。**

### 9.1 需要严谨推导

尤其是涉及：

- reliability score；
- entropy；
- margin；
- conflict；
- candidate rule；
- negative cache；
- soft suppression；

都要先分析数学含义和可能漏洞。

### 9.2 不能把 diagnostic signal 直接当 method

例如 E4-DIAG 证明：

```text
global-local conflict 能显著提高 global pseudo-label 错误概率
```

但这不等于：

```text
hard negative cache insertion 一定提升准确率
```

必须区分：

```text
diagnostic validity
```

和：

```text
intervention effectiveness
```

### 9.3 需要注意 base-rate fallacy

不能只看：

```math
P(W=1|I_D=1) > P(W=1)
```

还要看绝对 precision，例如：

```math
P(W=1|I_D=1) > tau_err
```

否则可能只是相对提高，但样本中大部分仍是正确预测。

### 9.4 不能把 local alternative 当作 pseudo-label

E4-CANC-DIAG-v2 已经形成重要结论：

```text
global-local conflict 能发现 global pseudo-label 错误，
但 local alternative class 的正确率很低。
```

因此后续不能写：

```math
k_l^* = y
```

也不能把：

```math
k_l^*
```

作为 positive pseudo-label。

正确方向是：

```math
hat_y_g should be suppressed
```

也就是 negative / boundary suppression。

---

## 10. 当前已形成的重要实验结论

### 10.1 E2/E3 的结论

E2-EMR、E3-GLC-v0、E3-GLC-v1 都没有超过 E1 baseline。

这说明：

```text
positive cache admission 改进目前不稳定
```

后续不应继续把 GLC 用作 positive cache promotion 主线。

### 10.2 E4-CANC-DIAG-v2 的结论

E4-CANC-DIAG-v2 证明：

```text
global-local conflict 在 ModelNet-C 7 个 corruption 上均显著提高 global pseudo-label 出错概率。
```

但：

```text
local alternative class 正确率很低。
```

因此 E4 后续方向应该是：

```text
negative / boundary suppression
```

而不是：

```text
local alternative positive correction
```

### 10.3 E4-CANC-v0 的结论

E4-CANC-v0 使用：

```math
I_neg^v0 = I_H or (I_M and I_D)
```

结果：

```text
安全，但过于保守。
```

平均只提升约 `+0.02`，真正新增 negative samples 约 `0.78%`。

### 10.4 E4-CANC-v1 的结论

E4-CANC-v1 使用：

```math
I_neg^v1 = I_H or (I_D and p_g^(1) > tau_p)
```

修正后的 all35 结果：

```text
35 个 corruption 全部 OK。
```

它增加了 negative intervention，但 hard negative insertion 没有明显转化为强 accuracy gain。

当前判断：

```text
conflict signal 是强的，但 hard cache insertion 的利用方式可能不够优。
```

### 10.5 E4-CANC-v2 的动机

E4-CANC-v2 方向是：

```text
Soft Conflict Suppression
```

不把 conflict samples 长期写入 negative cache，而是对当前样本的 global predicted class 做软抑制：

```math
z'_{hat_y_g} = z_{hat_y_g} - alpha r(x)
```

其中 `r(x)` 是连续 conflict risk score。

---

## 11. 服务器环境规则

当前服务器环境：

| 资源 | 配置 |
|---|---|
| CPU | 24 核 |
| 内存 | 94 GB |
| GPU | 2 × Tesla T4 |
| 系统盘 `/` | 约 12T/15T，速度一般 |
| 数据盘 `/root/autodl-tmp` | 约 12T/15T，速度快 |

项目主要路径：

```text
/root/autodl-tmp/MCM-PC-2
/root/autodl-tmp/MCM-PC-2/Point-Cache
```

数据盘适合：

- 数据；
- 权重；
- 高 IO 实验；
- MCM-PC-2 项目主体。

---

## 12. 双卡实验规则

由于 Point-Cache runner 一次处理一个 corruption，双卡并行方式不是拆单个 corruption，而是：

```text
GPU0 跑一部分 corruption
GPU1 跑另一部分 corruption
```

例如 all35：

```text
GPU0: 18 corruptions
GPU1: 17 corruptions
```

每个 worker 分别写自己的 summary，最后合并。

---

## 13. 文档偏好

用户喜欢把阶段性分析写成：

- HTML 文档；
- Markdown 文档；
- 阶段归档；
- 论文笔记。

文档应包含：

1. 全局思路；
2. 实验代号解释；
3. 为什么做；
4. 数学公式；
5. 实验结果表；
6. 成功和失败分析；
7. 后续计划；
8. 论文写作素材。

---

## 14. HTML 文档偏好

用户希望 HTML 文档：

- 内容详细；
- 结构清晰；
- 有目录；
- 不要只写一句话时间线；
- 成功和失败都要写；
- 不能提前使用未解释的代号；
- 先讲全局方法思路，再讲当前阶段；
- 后续 Multi-Cache Matrix 也要写入，不要只总结到当前 E4。

### 14.1 HTML 视觉风格固定模板

从 2026-05-17 起，用户要求后续所有由助手编写的说明文档类 HTML，统一采用下面这个文件的视觉风格和页面结构：

```text
/root/autodl-tmp/MCM-PC-2/docs/reports/2026-05-17_task_specification.html
```

后续新 HTML 报告应复用该文件的主要风格要素：

1. 深色主题；
2. 左侧 sticky 目录；
3. `doc-header` 文档头；
4. `tldr` 摘要块；
5. `dashboard` 指标卡片；
6. 深色表格、pill、callout、formula、compare、roadmap 等组件；
7. 中文说明为主，必要位置保留英文论文标题、方法名和实验名。

除非用户明确要求换风格，否则不要再使用浅色卡片式 HTML 报告模板。

---

## 15. 沟通注意事项

### 15.1 回答中要及时纠正错误

如果之前分析有误，例如：

```text
把 suffix _2 误认为 severity 2
```

必须明确纠正，而不是含糊带过。

### 15.2 不要过度自信

如果缺少 baseline all35，就不能说：

```text
E4-CANC-v1 超过 baseline all35
```

只能说：

```text
E4-CANC-v1 all35 自身结果是 ...
还需要 E1-BASE all35 进行公平对比。
```

### 15.3 用户常常需要下一步命令

当用户说：

```text
下一步应该干什么
```

应给出：

1. 当前判断；
2. 下一步任务；
3. 完整命令；
4. 检查命令；
5. 何时把结果发回来；
6. 是否需要 commit。

---

## 16. 当前最近的下一步优先级

截至本文档生成时，当前优先级是：

1. 等 E1-BASE all35 跑完；
2. 上传 `summary_e1_base_dual_gpu_all35.csv`；
3. 与 E4-CANC-v1 corrected all35 做公平对比；
4. 同时可以继续 E4-CANC-v2 suffix `_2` quick/full test；
5. 如果 v2 有希望，再考虑 all35；
6. 结果稳定后再写阶段文档和 commit。

---

## 17. 给未来助手的简短规则

如果后续继续这个项目，请优先遵守：

```text
1. 不要重复问已有路径和实验背景。
2. 每个新实验必须有正式编号和名字。
3. 脚本命名必须统一。
4. 先 quick test，再 full run。
5. logs 不提交 Git。
6. commit 按类型拆分。
7. all35 使用 _0.._4，不使用 _5。
8. E1 是 baseline reproduction。
9. E2/E3 是 positive admission 探索，已不作为主线。
10. E4 是当前主线：conflict-aware negative / soft suppression。
11. local alternative 不能当 pseudo-label。
12. 新方法先做数学推导，再写代码。
13. 文档要详细，不要只写一句话时间线。
14. 没有公平 baseline 时，不要声称方法超过 baseline。
15. 阶段性结果要及时总结并归档。
16. 说明文档类 HTML 统一采用 `docs/reports/2026-05-17_task_specification.html` 的深色目录式风格。
```
