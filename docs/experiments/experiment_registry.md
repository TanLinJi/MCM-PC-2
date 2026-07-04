# DPC-Point Experiment Registry

Last updated: 2026-06-10

Current paper title:

**DPC-Point: Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation**

This registry tracks the current experiment state. It does not replace each
experiment's own log, analysis, or result files.

## Current Registry

| ID | Name | Role | Status | Key Doc / Result |
|---|---|---|---|---|
| E0 | Point-Cache baseline reproduction | Reproduction foundation and comparison baseline | Complete; maintained | `docs/experiments/E0_baseline/`, `docs/experiments/pointcache_repro/` |
| E1 | Text prototype enhancement | Source of semantic prompt distributions; direct text-classifier replacement is not the main method | Restarting from prompt/weight ablations | `docs/experiments/E1_text_prototype_enhancement/README.md` |
| E2 | Text prototype transfer to Point-Cache | Tests whether E1 prompts improve Point-Cache directly | Complete; not selected as final framing | `docs/experiments/E2_text_prototype_transfer_to_pointcache/results_summary.md` |
| E3 | Global prototype alignment cache | Explores prototype alignment and cache replacement ideas | Exploratory; partly superseded by E4 | `docs/experiments/E3_global_prototype_alignment_cache/analysis.md` |
| E4 | Distribution-guided cache | Current DPC-Point main method | Active main line | `docs/experiments/E4_distribution_guided_cache/02_9_text_weight_ablation_analysis.md` |
| E5 | ADAPT/PGA-inspired Gaussian alignment cache | Secondary exploratory branch for posterior/residual cache alignment | Exploratory | `docs/experiments/E5_adapt_inspired_gaussian_alignment_cache/E5_BCD_posterior_prototype_residual_design.md` |

## Selected Current Anchor

```text
Run ID: 02_9_2
Method: E4-C-A0+E1-textdist-only
Text score weight: 0.15
Backbone: ULIP
Dataset: ModelNet-C severity=2
Average accuracy: 54.70595045
Clean accuracy: 63.86
```

Reference comparison:

```text
Original Point-Cache clean: 64.18
Point-Cache style full severity-2 baseline: 54.00
```

Interpretation: E4 is promising on severity-2 robustness, but the clean tradeoff
and all35 results must be reported before the method is paper-ready.

## Current Next Work

1. Finish ModelNet-C all35 for `02_9_2`.
2. Report clean vs corrupted tradeoff against the original Point-Cache baseline.
3. Run ScanObjNN-C hardest and ShapeNet-C comparisons.
4. Add one additional backbone beyond ULIP.
5. Keep E5 as exploratory unless it beats E4 without clean regression.

## Archive Boundary

Legacy MCM-PC / Multi-Cache Matrix / CANC planning files are archived under:

```text
docs/archive/legacy_mcmpc_canc/
```

Early pre-restart experiments are archived under:

```text
docs/experiments/archive/legacy_pre_mcmpc_restart/
```
