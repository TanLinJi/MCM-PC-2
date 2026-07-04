# DPC-Point Current Direction

Current paper title:

**DPC-Point: Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation**

## Positioning

DPC-Point is the current paper direction for improving Point-Cache. The method
focuses on distribution-guided prototype cache purification during online
test-time adaptation for point cloud vision-language models.

This replaces the broader MCM-PC / Multi-Cache Matrix framing for the current
ICASSP 2027 submission. Multi-Cache Matrix and CANC documents are archived under
`docs/archive/legacy_mcmpc_canc/`.

## Core Claim

Point-Cache stores low-entropy online test samples and reuses them as global and
local cache evidence. Low entropy is useful, but it is not enough under point
cloud corruptions: confidently wrong samples can enter the cache and later
pollute local prototype evidence.

DPC-Point changes the cache update question from:

```text
Is this sample more confident?
```

to:

```text
Is this sample more consistent with the class distribution implied by trusted
visual history and semantic text prompts?
```

## Current Main Method

The current main method is:

```text
E4-C-A0+E1-textdist-only
E4_TEXT_SCORE_WEIGHT=0.15
```

It keeps the original Point-Cache final classifier and final logits formula, but
changes GPA/prototype cache replacement:

```text
replace if entropy(new) < entropy(old)
and joint_score(new, class) > joint_score(old, class)

joint_score =
    normalized accepted-history visual distribution score
    + lambda_t * normalized prompt-level text distribution score
```

The E1 LLM descriptions are not used to replace the final text classifier. They
are used only to build the text distribution prior for cache replacement.

## Evidence So Far

Current strongest complete ModelNet-C severity-2 result:

| Setting | Avg Acc |
|---|---:|
| Point-Cache style full baseline on ULIP severity-2 | 54.00 |
| DPC-Point current setting `02_9_2` | 54.706 |

The current clean result is:

| Setting | Clean Acc |
|---|---:|
| Original Point-Cache `ZS + Global + Local Cache` | 64.18 |
| `02_9_2` clean | 63.86 |

This means the current method is promising but not yet paper-complete. The final
paper must report clean robustness tradeoffs, not only corrupted severity-2
averages.

## Immediate Paper-Critical Work

1. Finish ModelNet-C all35 for the current `02_9_2` setting.
2. Compare against Point-Cache on clean, severity-2, and all35.
3. Run ScanObjNN-C hardest with fair baseline comparison.
4. Add at least one more backbone after ULIP, preferably ULIP-2 or OpenShape.
5. Keep E5 posterior residual as a secondary exploratory line unless it clearly
   beats `02_9_2` without hurting clean.
