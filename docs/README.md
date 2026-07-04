# DPC-Point Documentation

> Project: **DPC-Point: Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation**
>
> Path: `/root/autodl-tmp/MCM-PC-2/docs/`

`docs/` is the human-readable documentation home for the current DPC-Point
paper line. Code, raw logs, datasets, checkpoints, and result artifacts remain
under `Point-Cache/` and related code directories.

## Current Reading Order

For the active ICASSP 2027 paper direction, read these files first:

1. `proposals/current_direction.md` - current DPC-Point positioning, method anchor, evidence, and near-term work.
2. `experiments/narrative/e4_e5_research_narrative.md` - current E4/E5 research narrative and evidence boundary.
3. `experiments/E4_distribution_guided_cache/02_9_text_weight_ablation_analysis.md` - selected DPC anchor setting.
4. `experiments/E5_adapt_inspired_gaussian_alignment_cache/E5_BCD_posterior_prototype_residual_design.md` - exploratory posterior/GDA branch.
5. `experiments/experiment_registry.md` - experiment status registry.
6. `paper/0_outline.md` - paper outline, contribution plan, and required result tables.
7. `project/glossary.md` - current terminology and archived-term boundary.
8. `project/conventions/user_preferences_workflow_conventions.md` - collaboration, experiment, and documentation rules.

## Directory Map

| Directory | Current Role |
|---|---|
| `paper/` | Current DPC-Point paper outline and section drafts |
| `proposals/` | Current research direction and method-positioning notes |
| `experiments/` | Experiment plans, analyses, registry, and long-form research narrative |
| `project/` | Project rules, glossary, progress log, and navigation aids |
| `decisions/` | Locked project decisions, including the rename to DPC-Point |
| `reports/` | Current self-contained HTML reports |
| `references/` | Paper PDFs and reference material |
| `assets/figures/` | Figures and figure sources used by docs or paper drafts |
| `context/windsurf/` | Conversation/context archive |
| `archive/` | Historical material that should not drive the current paper |

## Current Paper State

The active method is the E4 distribution-guided prototype cache:

```text
E4-C-A0+E1-textdist-only
E4_TEXT_SCORE_WEIGHT=0.15
```

The current strongest complete severity-2 anchor is:

```text
02_9_2
ULIP + ModelNet-C severity=2
Avg acc: 54.70595045
```

This is a working anchor, not yet a final paper benchmark. The paper still needs
all35, clean tradeoff, ScanObjNN-C, ShapeNet-C, and backbone-transfer evidence.

## Archive Boundary

The current DPC-Point line supersedes the older MCM-PC / Multi-Cache Matrix /
CANC / MCP-3D framing. Those documents are preserved under:

```text
docs/archive/legacy_mcmpc_canc/
```

Early pre-restart experiments are preserved under:

```text
docs/experiments/archive/legacy_pre_mcmpc_restart/
```

Archived files are retained for traceability only. If an archived idea becomes
active again, summarize it in a current DPC-Point document instead of editing the
old archived file.

## Documentation Rules

- New paper sections go under `paper/`.
- New method proposals or positioning notes go under `proposals/`.
- New experiment summaries go under `experiments/<experiment_name>/`; raw logs and result files stay in `Point-Cache/`.
- New long-form research narratives go under `experiments/narrative/`.
- New collaboration rules or user preferences go under `project/conventions/`.
- Deprecated plans should be moved to `archive/`, not deleted, unless they are empty placeholders.
- High-level indexes must use DPC-Point terminology and should not point readers to archived MCM-PC/CANC plans as current work.

## Changelog

- **2026-06-10** v2.0: Adopted DPC-Point as the current paper title and reorganized the docs entrypoint around the E4 distribution-guided prototype cache line. Archived old MCM-PC/CANC paper and proposal material under `docs/archive/legacy_mcmpc_canc/`.
- **2026-06-06** v1.3: Added long-term experiment narrative and project convention directories.
- **2026-06-02** v1.2: Expanded project tree documentation.
- **2026-06-02** v1.1: Added `project/project_tree.md`.
- **2026-05-17** v1.0: Initial docs structure.
