> We formulate test-time evidence in point cloud recognition as a cache matrix, where rows correspond to structural levels, including global object features, local part features, and semantic textual anchors, while columns correspond to different reliability roles, including confident evidence, compact prototype evidence, and boundary-aware negative evidence.





> Unlike previous hierarchical cache designs where local fingerprints are stored once the global sample is admitted, we explicitly evaluate the agreement between global object-level predictions and local part-level evidence. This global-local consistency serves as a structural reliability signal for cache admission, replacement, and logit fusion.



> Since normalized multimodal embeddings lie on a hypersphere, we optionally model textual prompt embeddings with a von Mises-Fisher(vMF) distribution rather than a Euclidean Gaussian. The estimated concentration parameter provides a semantic reliability score, which is used to modulate the contribution of textual priors during evidence fusion.

