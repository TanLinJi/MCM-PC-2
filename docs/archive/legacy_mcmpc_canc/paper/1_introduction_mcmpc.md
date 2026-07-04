# 1 Introduction Draft

Large 3D vision-language models enable open-vocabulary point cloud recognition
by aligning point cloud features with text prototypes. In deployment, however,
test point clouds are rarely clean: missing points, local perturbations, global
noise, rotation, scale shifts, and sensor artifacts can move features away from
their clean training distribution. Because retraining is often unavailable,
test-time adaptation has become a practical way to improve robustness with only
online test samples.

Point-Cache is a representative training-free approach. It dynamically stores
online point cloud features in global and local caches, then combines the
zero-shot text logits with cache retrieval logits. This design is efficient and
plug-and-play, but its cache update relies heavily on confidence or entropy.
Under distribution shift, confidence is not the same as reliability. A corrupted
sample may be confidently assigned to the wrong class, admitted into the cache,
and later used as misleading prototype evidence.

DPC-Point addresses this cache-pollution problem by making cache replacement
distribution-aware. The key idea is simple: a new sample should not replace an
existing prototype-cache entry merely because it has lower entropy; it should
also be more consistent with the class distribution implied by previously
accepted visual evidence and semantic text prompts.

Our current implementation keeps the strong Point-Cache prediction formula
unchanged and modifies the GPA/prototype cache replacement rule. The visual
distribution is estimated from accepted-history samples rather than the current
cache snapshot, which is too narrow when the cache capacity is small. The text
distribution is built from prompt-level embeddings, including LLM-generated
descriptions, but these descriptions are used only for cache purification and
not to replace the base text classifier.

Contributions:

1. We identify confidence-only cache update as a source of prototype cache
   pollution for robust point cloud test-time adaptation.
2. We propose distribution-guided prototype cache replacement using
   accepted-history visual statistics and prompt-level semantic text
   distributions.
3. We show that decoupling the semantic distribution prior from the final text
   classifier is important: text prompts can guide cache purification without
   destabilizing the base zero-shot classifier.
4. We provide an empirical evaluation plan covering corrupted and clean point
   cloud benchmarks, ablations, and backbone transfer.
