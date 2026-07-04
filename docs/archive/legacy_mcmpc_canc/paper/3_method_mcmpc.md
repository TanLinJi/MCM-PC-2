# 3 Method Draft

## 3.1 Problem Setup

Given a frozen 3D vision-language model, each test point cloud produces a global
feature, local part features, and zero-shot logits against text prototypes. The
test stream is processed online. No source training data and no gradient update
are used.

The goal is to improve robustness under distribution shift by updating
test-time caches without admitting unreliable prototype evidence.

## 3.2 Point-Cache Baseline

The baseline prediction combines:

```text
final logits =
    zero-shot text logits
    + global positive cache logits
    + local positive cache logits
    - negative cache logits
```

Positive cache updates are mainly confidence or entropy driven. DPC-Point keeps
this final logit form unchanged for the current main method and changes the
prototype/GPA cache replacement criterion.

## 3.3 Accepted-History Visual Distribution

For each predicted class, DPC-Point maintains a visual distribution using only
samples that have been accepted by trusted positive-cache mechanisms. Unlike a
current-cache snapshot, accepted history can preserve more class diversity when
the active cache capacity is small.

For class `c`, the current implementation stores:

```text
count_c, mean_c, diagonal variance_c
```

The visual distribution score for feature `x` is:

```text
s_v(x, c) = - mean((x - mean_c)^2 / (var_c + eps))
```

## 3.4 Prompt-Level Text Distribution

For each class, prompt-level text embeddings define a fixed semantic
distribution. The current strongest setting uses the base `manual_full` prompts
for the final classifier, while LLM-generated E1 descriptions are included only
in the text distribution used for replacement scoring.

The text distribution score has the same diagonal form:

```text
s_t(x, c) = - mean((x - text_mean_c)^2 / (text_var_c + eps))
```

## 3.5 Distribution-Guided Prototype Cache Replacement

When the GPA/prototype cache for class `c` is full, DPC-Point compares a new
candidate with the highest-entropy existing cache entry. The replacement rule is:

```text
replace if
    entropy(new) < entropy(old)
and
    joint_score(new, c) > joint_score(old, c)
```

where:

```text
joint_score(x, c) =
    norm(s_v(x, c)) + lambda_t * norm(s_t(x, c))
```

The current default is:

```text
E4_SCORE_NORM_MODE=running_zscore
E4_TEXT_SCORE_WEIGHT=0.15
```

## 3.6 Decoupled Semantic Prior

DPC-Point deliberately separates two roles of text:

1. final classifier text prototypes;
2. prompt-level semantic distribution for cache purification.

The current evidence indicates that directly replacing the final classifier with
LLM prompt fusion can be unstable, while using LLM descriptions only as a text
distribution prior improves cache replacement more safely.

## 3.7 Optional E5 Extension

The E5 posterior prototype residual branch is exploratory. It uses a delayed
StatsBank to estimate text-prior posterior prototypes and reports multiple
residual gamma values in one pass. It is not part of the main DPC-Point method
unless it improves over `02_9_2` without clean-data regression.
