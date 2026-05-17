# MCM-PC 文档目录说明

`docs/` 是本项目的人类可读文档入口。代码、数据、权重、原始运行日志仍保留在各自的代码目录中；只有报告、笔记、论文草稿、决策记录、参考资料等放在这里。

## 目录分类

| 目录 | 放什么 | 例子 |
|---|---|---|
| `project/` | 项目管理、SOP、导航、里程碑追踪 | `NAVIGATION.md`, `MILESTONE_SOP.md`, `progress.txt` |
| `proposals/` | 研究方案、可行性分析、方法设计草稿 | `MCP3D_full_proposal_v2.md` |
| `decisions/` | 已锁定决策、设计理由、失败复盘 | `D19_design_rationale.md`, `D20_p1_post_mortem.md` |
| `concepts/` | 概念学习笔记和解释材料 | `00_overview.md`, `05_mcp_three_caches.html` |
| `paper/` | 论文大纲和正文草稿 | `00_outline.md`, `01_introduction.md` |
| `experiments/` | 可阅读的实验总结、里程碑报告 | `fig1a_summary.md`, `p1/P1_full_drift.md` |
| `reports/` | 自包含 HTML 报告或复习页面 | `2026-05-10_review_session.html` |
| `context/windsurf/` | 对话归档、恢复上下文、偏好和下一步 | `INDEX.md`, `next_steps.md`, `decisions.md` |
| `references/papers/` | 参考论文 PDF | Point-Cache、MCP、BayesMM 等论文 |
| `assets/figures/` | 图、图源文件、流程图源文件 | `.svg`, `.mmd` |

## 以后新增文档怎么放

- 新论文正文或章节草稿：放到 `paper/`。
- 新实验总结：放到 `experiments/<milestone>/`；原始日志继续留在 `Point-Cache/logs/` 或 `Point-Cache/results/`。
- 新概念解释：放到 `concepts/`；如果是长篇可视化讲义，可以做成自包含 HTML。
- 新方法决策、失败假设、复盘：放到 `decisions/`。
- 新会话恢复信息、偏好、下一步计划：放到 `context/windsurf/`。
- 新 HTML 报告：放到 `reports/`。
- 新图表和 Mermaid / SVG 源文件：放到 `assets/figures/`。
- 新参考论文 PDF：放到 `references/papers/`。

## 当前推荐阅读顺序

1. `project/NAVIGATION.md`
2. `project/progress.txt`
3. `decisions/D20_p1_post_mortem.md`
4. `experiments/fig1a_summary.md`
5. `experiments/p1/P1_full_drift.md`
6. `experiments/p1/P1_pollution_sim.md`
7. `paper/01_introduction.md`
