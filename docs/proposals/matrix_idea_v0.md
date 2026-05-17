| 层级   | Entropy Cache          | Align Cache                             | Negative Cache   |
| ------ | ---------------------- | --------------------------------------- | ---------------- |
| Global | 低熵全局点云特征       | 靠近 global prototype center 的样本     | 高不确定全局样本 |
| Local  | 低熵 patch / part 特征 | 靠近 local-part prototype center 的样本 | 局部不确定样本   |



创新点：

**贡献 1：发现问题。**
 现有 Point-Cache 虽然使用 global/local 层次缓存，但仍主要依赖低熵伪标签；在点云 corruption、局部缺失、旋转、jitter 等情况下，低熵样本未必形成紧致类内分布，容易造成 prototype drift。

**贡献 2：提出方法。**
 提出一种 compactness-aware hierarchical multi-cache framework，在 global 和 local 两个层级分别构建 entropy cache、align cache 和 negative cacshe，用文本分布原型作为 semantic anchor，提升 test-time point cloud adaptation 的稳定性。

**贡献 3：实验验证。**
 在 ModelNet-C、ScanObjectNN-C、ModelNet40、ScanObjectNN hardest split、OmniObject3D、Objaverse-LVIS 上，对 ULIP、ULIP-2、OpenShape、Uni3D 做实验，证明该方法在 robustness、generalization、memory、throughput 上优于 Point-Cache，并尽量接近或超过 BayesMM。