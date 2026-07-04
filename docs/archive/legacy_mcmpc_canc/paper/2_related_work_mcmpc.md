# 2 Related Work Draft

## Point Cloud Test-Time Adaptation

Test-time adaptation aims to improve model behavior during inference without
retraining on source data. For point cloud recognition, this setting is
especially relevant because deployment data often contains missing points,
sampling artifacts, global noise, local perturbations, rotation, and scale
shifts. Cache-based methods are attractive because they are training-free and
can work with online test streams.

Point-Cache is the main baseline for this project. It maintains global and local
caches of online point cloud features and fuses cache logits with zero-shot text
logits. DPC-Point builds directly on this setting and focuses on a narrower
question: how should prototype cache entries be admitted or replaced under
distribution shift?

## 3D Vision-Language Recognition

Models such as ULIP, ULIP-2, OpenShape, and Uni3D align point cloud features
with text features, enabling zero-shot point cloud classification. Their
performance depends not only on the 3D encoder but also on the text prototype
construction strategy. Prior work often averages prompt embeddings into a
single class vector. DPC-Point instead uses prompt-level text embeddings as a
semantic distribution for cache replacement while keeping the final classifier
stable.

## Cache Reliability and Prototype Pollution

Confidence-based cache admission is simple but vulnerable to confidently wrong
pseudo-labels. Once a wrong or unrepresentative sample enters the cache, it may
act as a polluted prototype for later samples. This issue is amplified in local
point cloud caches because local part evidence can be noisy or incomplete under
corruptions. DPC-Point treats cache replacement as a distribution-consistency
problem rather than a confidence-only ranking problem.

## Textual Priors and Distributional Modeling

Recent work on multimodal Bayesian distribution learning and probabilistic
Gaussian alignment suggests that text and visual evidence can be modeled as
distributions rather than single points. DPC-Point adopts this insight in a
lightweight way: prompt-level text embeddings define a fixed semantic
distribution, while accepted online visual samples define a dynamic visual
distribution. These distributions guide cache purification rather than becoming
a standalone classifier.

## Current Citation Gaps

- Point-Cache CVPR 2025.
- ULIP / ULIP-2.
- OpenShape.
- Uni3D.
- Training-free dynamic adapter / cache-based TTA references.
- BayesMM / multimodal Bayesian distribution learning.
- ADAPT / probabilistic Gaussian alignment.
