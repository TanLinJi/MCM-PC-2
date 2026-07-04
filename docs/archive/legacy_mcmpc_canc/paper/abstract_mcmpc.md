# Abstract Draft

Large 3D vision-language models exhibit strong zero-shot recognition ability on
point clouds, yet their deployment remains challenged by distribution shifts.
Cache-based adaptation provides an efficient online solution, but relying on a
single prediction-confidence criterion for cache updates can introduce
unrepresentative samples and gradually contaminate the prototype cache. To this
end, we propose **DPC-Point**, a Distribution-Guided Prototype Cache framework
that builds cache entries through class-wise probabilistic distribution
modeling. For each class, DPC-Point maintains a visual distribution from
accepted historical features and a semantic distribution from textual embeddings
of class descriptions. LLM-generated class descriptions are used to strengthen
the semantic distribution with richer category-level semantics. Given an
incoming sample, DPC-Point computes a
visual-semantic distribution score that measures its consistency with the
predicted class distribution. Cache entries are then updated only when the
candidate is both confident and distributionally consistent, yielding a more
representative prototype cache for online adaptation. The resulting framework is
source-free, training-free, backpropagation-free, and directly applicable to
streaming point cloud recognition without modifying the frozen prediction model
or accessing future test samples.
Experiments on distribution-shifted point cloud benchmarks show that DPC-Point
improves robustness, with the current ULIP result on ModelNet-C severity-2
reaching 54.71% average accuracy.

## Notes

This is the current ICASSP-style draft. Before submission, replace the final
result sentence with the complete all35, clean, cross-dataset, and multi-backbone
numbers.
