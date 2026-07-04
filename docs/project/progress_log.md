# Progress Log

This log records high-level project milestones. Detailed experiment evidence is
kept in `docs/experiments/` and `Point-Cache/`.

## 2026-06-10 - Project Rename And Current Paper Line

Current paper title:

**DPC-Point: Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation**

The active paper direction is no longer the broad MCM-PC / Multi-Cache Matrix /
CANC framing. The old paper and proposal files were archived under
`docs/archive/legacy_mcmpc_canc/`.

## 2026-06-10 - Current E4 Anchor

Selected current method anchor:

```text
02_9_2
E4-C-A0+E1-textdist-only
E4_TEXT_SCORE_WEIGHT=0.15
ULIP + ModelNet-C severity=2
Avg acc: 54.70595045
Clean acc: 63.86
```

Reference comparison:

```text
Point-Cache style full severity-2 baseline: 54.00
Original Point-Cache clean: 64.18
```

Interpretation: the method has a useful severity-2 robustness signal, but the
paper still needs all35, clean tradeoff, cross-dataset, and backbone-transfer
evidence.

## 2026-06-08 - E4/E5 Narrative Consolidation

E4 became the current main line: distribution-guided replacement of the
prototype/cache entry, using accepted visual history and a semantic text
distribution prior. E5 remains exploratory unless posterior/residual GDA results
beat E4 without clean regression.

Current narrative:

```text
docs/experiments/narrative/e4_e5_research_narrative.md
```

## 2026-05-13 - Baseline Reproduction

Early reproduction milestone:

| Date | Method | Backbone | Dataset | Corruption | Accuracy | Delta vs Zero-shot |
|---|---|---|---|---|---:|---:|
| 2026-05-13 | Zero-shot | ULIP-2 | ModelNet-C | add_global_2 | 65.19 | - |
| 2026-05-13 | Global Cache | ULIP-2 | ModelNet-C | add_global_2 | 67.06 | +1.87 |
| 2026-05-13 | Hierarchical Cache | ULIP-2 | ModelNet-C | add_global_2 | 68.15 | +2.96 |

This confirmed the Point-Cache reproduction setup and motivated later cache
reliability analysis.
