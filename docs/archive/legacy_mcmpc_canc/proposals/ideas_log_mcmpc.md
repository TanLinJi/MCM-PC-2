> We reveal that hierarchical point-cloud caches suffer from class-wise geometric dispersion and local-part prototype drift under test-time distribution shifts. To address this, we propose a compactness-aware multimodal hierarchical cache framework that jointly models global/local point features, text-derived semantic distributions, and negative uncertainty evidence for robust open-vocabulary 3D test-time adaptation.

MCM-PC: Multi-Cache Multimodal Point-Cache



`模块 A：Textual Distribution Prototype，只参考 BayesMM 的文本处理`

对每个类别，不只用一个 prompt，而是用多个 LLM 生成的 prompt。比如：

“a 3D object of a chair”
 “a point cloud model of a chair”
 “a scanned 3D chair with back and legs”
 “a geometric shape representing a chair”

然后用文本编码器得到多个文本特征，计算均值和协方差。这个模块的作用不是完整复现 BayesMM，而是给每个类别一个**更稳定的文本原型先验**。BayesMM 原文就是用 paraphrases 建立文本分布，之后通过均值、协方差和 MAP 得到文本原型