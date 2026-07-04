# DPC-Point Paper Outline

Working title:

**DPC-Point: Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation**

Target: ICASSP 2027  
Full paper deadline: 2026-09-16  
Current status: method direction selected; benchmark evidence still incomplete.

## Central Thesis

Online point cloud cache adaptation should not admit or replace cache samples
using confidence alone. DPC-Point uses trusted visual history and semantic text
prompt distributions to guide prototype cache replacement, reducing cache
pollution under test-time corruptions.

## Proposed Contributions

1. We analyze Point-Cache under corrupted point cloud streams and identify that
   confidence-based cache update can admit confidently wrong or distributionally
   inconsistent samples.
2. We propose a distribution-guided prototype cache replacement rule that uses
   accepted-history visual statistics rather than a narrow current-cache snapshot.
3. We introduce a decoupled semantic text distribution prior: LLM-generated
   descriptions guide cache purification but do not replace the stable base text
   classifier.
4. We evaluate robustness across ModelNet-C, ScanObjNN-C, clean data, ablations,
   and multiple 3D vision-language backbones.

## Section Plan

| Section | Content | Current Status |
|---|---|---|
| Abstract | Problem, method, core results | Skeleton |
| 1 Introduction | Motivation, cache pollution, DPC-Point contributions | Skeleton |
| 2 Related Work | Point cloud TTA, 3D vision-language models, cache adaptation, text priors | Skeleton |
| 3 Method | Baseline Point-Cache, visual distribution, text distribution, replacement rule | Skeleton |
| 4 Experiments | Main results, clean/corruption tradeoff, ablations, efficiency | Blocked by all35 and extra backbones |
| 5 Discussion | Failure cases, clean regression, pseudo-label risk, limitations | Pending |
| 6 Conclusion | Summary | Pending |

## Required Result Tables

| Table | Required Evidence | Status |
|---|---|---|
| Main ModelNet-C | ULIP all35, baseline vs DPC-Point | Running / incomplete |
| Main ScanObjNN-C | hardest all35 or severity-2 minimum with baseline | Partial severity-2 |
| Clean robustness | clean baseline vs DPC-Point | Current `02_9_2` clean available |
| Backbone transfer | ULIP-2 or OpenShape at minimum | Not complete |
| Ablation | accepted-history, textdist, text weight, score normalization | Severity-2 partial |
| Efficiency | runtime / memory vs Point-Cache | Not complete |

## Current Best Result Anchor

Current strongest complete severity-2 run:

```text
02_9_2
E4-C-A0+E1-textdist-only
E4_TEXT_SCORE_WEIGHT=0.15
ULIP + ModelNet-C severity=2
Avg acc: 54.70595045
```

This is a working anchor, not yet a final paper benchmark.
