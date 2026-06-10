Existing point-cloud cache methods mainly store high-confidence test samples and reuse them as online evidence for prediction refinement. Although such a strategy is efficient and effective, confidence alone is not a sufficient indicator of cache reliability under test-time distribution shifts. In corrupted or cross-domain point clouds, a sample may be confidently misclassified, globally plausible but locally inconsistent, or close to multiple confusing categories in the embedding space. As a result, the cached samples may not be semantically reliable, geometrically compact, locally consistent, or discriminatively separated from boundary classes.

To address these limitations, we propose a reliability-aware Multi-Cache Matrix framework for test-time point cloud adaptation. Instead of maintaining a single type of high-confidence cache, our framework organizes test-time evidence across global and local structural levels, as well as different reliability roles, including confident evidence, compact prototype evidence, and boundary-aware negative evidence. This design enables more stable, compact, and discriminative adaptation for open-vocabulary 3D point cloud recognition.



> We propose a Multi-Cache Matrix framework for test-time point cloud adaptation, which organizes online test-time evidence across structural levels and reliability roles. Unlike conventional hierarchical caches that store only high-confidence global and local fingerprints, our framework explicitly separates confident, compact, and boundary evidence for both global object features and local part features.

## 







> We argue that entropy alone is insufficient for test-time cache construction in 3D point cloud recognition. Under geometric corruptions, a sample can be confidently misclassified, compact within a wrong cluster, or globally correct but locally inconsistent. Therefore, reliable cache construction should jointly consider confidence, intra-class compactness, inter-class margin, and global-local structural consistency.

> We introduce a unified reliability criterion that jointly considers prediction entropy, intra-class compactness, inter-class angular margin, and global-local structural consistency. This criterion reduces the risk of admitting confidently wrong or locally inconsistent samples into the cache under severe point cloud corruptions.

> We further design a reliability-gated evidence fusion strategy that adaptively balances textual priors, global cache retrieval, local part retrieval, and negative boundary evidence for each test sample. This avoids manually fixed logit fusion and enables robust adaptation under different corruption patterns.



**Gaussian textual priors ignore the hyperspherical geometry of normalized multimodal embeddings.**