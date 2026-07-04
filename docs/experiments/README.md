# DPC-Point Experiments

This directory records experiment documentation for the current paper line:

**DPC-Point: Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation**

The current goal is to improve Point-Cache by replacing confidence-only cache
replacement with distribution-guided prototype cache purification.

## Current Experiment Sequence

| ID | Name | Role in Current Paper | Status |
|---|---|---|---|
| E0 | Point-Cache baseline reproduction | Baseline and failure-analysis foundation | Complete / maintained |
| E1 | Text prototype enhancement | Prompt-distribution source; direct classifier replacement is not the final path | Complete as supporting evidence |
| E2 | Text prototype transfer to Point-Cache | Tested direct prompt transfer into Point-Cache | Complete; not current main method |
| E3 | Global prototype alignment cache | Explored visual prototype alignment and cache replacement variants | Exploratory; motivates E4 |
| E4 | Distribution-guided cache | Current main DPC-Point method | Active main line |
| E5 | ADAPT/PGA-inspired Gaussian alignment cache | Posterior/GDA-inspired exploratory branch | Exploratory; secondary unless it beats E4 cleanly |

## Current Main Anchor

```text
02_9_2
E4-C-A0+E1-textdist-only
E4_TEXT_SCORE_WEIGHT=0.15
ULIP + ModelNet-C severity=2
Avg acc: 54.70595045
```

The E4 anchor keeps the original Point-Cache final classifier and final logits
formula, but changes prototype/cache replacement with a joint visual-history and
text-distribution score.

## Required Paper Evidence

The current paper is not complete until the following evidence is available:

| Evidence | Minimum Requirement |
|---|---|
| ModelNet-C robustness | ULIP all35, baseline vs DPC-Point |
| Clean tradeoff | clean baseline vs DPC-Point |
| Cross-dataset robustness | ScanObjNN-C hardest and ShapeNet-C where feasible |
| Backbone transfer | at least one additional backbone, preferably ULIP-2 or OpenShape |
| Ablation | visual-history score, textdist score, text weight, normalization |
| Cost/diagnostics | runtime, memory, cache replacement behavior, order sensitivity |

## Documentation Rule

Each active experiment directory should keep human-readable records:

```text
docs/experiments/<experiment_name>/log.md
docs/experiments/<experiment_name>/analysis.md
```

Raw logs, CSV/JSON outputs, checkpoints, datasets, and generated result folders
stay under `Point-Cache/`.

Update `experiments/narrative/e4_e5_research_narrative.md` whenever a result
changes the current research judgment.

## Archived Legacy Experiments

Early exploratory experiments before the formal restart are archived under:

```text
docs/experiments/archive/legacy_pre_mcmpc_restart/
```

Those files are historical context and do not define the current E0-E5 DPC-Point
sequence.
