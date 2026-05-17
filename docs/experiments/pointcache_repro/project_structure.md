

# 第 0 层：Point-Cache 论文一句话

> **训练免费**的点云 test-time adaptation：测试样本流式进来时，**动态维护两层（全局 + 局部）× 两类（正例 + 反例） = 4 个缓存**，把缓存特征当 "类原型" 修正 zero-shot CLIP-3D logits。

它的两大核心创新对应：
- **Hierarchical**：除了整体点云特征，还把点云**聚成 5 个 patch**，每个 patch 当独立证据存进 local cache。
- **Positive + Negative**：低熵样本 → 正例 cache（"这个特征长得像 chair"）；中等熵样本 → 反例 cache（"这个特征'不太像' chair, 而是 chair/table 之间"）。

---

# 第 1 层：代码目录 ↔ 论文模块

| 论文章节 / 模块                      | 对应代码                                                     | 一句话作用                                           |
| ------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------- |
| §3 总框架 / Algorithm 1              | [runners/model_with_hierarchical_caches.py](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_hierarchical_caches.py:0:0-0:0) (主)<br/>[runners/model_with_global_cache.py](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:0:0-0:0) (简化版) | 训练-时-适配主循环                                   |
| §3.1 3D Encoder backbone             | [models/openshape/ppta.py](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/models/openshape/ppta.py:0:0-0:0)<br/>`models/uni3d/`、`models/ulip/` | OpenShape / Uni3D / ULIP 三个 backbone 的封装        |
| §3.1 Patch 聚类（Hierarchical 关键） | [models/openshape/ppta.py](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/models/openshape/ppta.py:0:0-0:0) 的 `cluster_patches` (KMeans, n_cluster=5) | 把 256 个 transformer token 聚成 5 个 patch          |
| §3.2 文本特征 (CLIP-text)            | [utils/utils.py](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/utils/utils.py:0:0-0:0) 的 [clip_classifier()](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/utils/utils.py:205:0-237:23) | 用 80 个 ImageNet-style prompts 平均得到每类文本嵌入 |
| §3.2 Zero-shot logits                | [utils/utils.py](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/utils/utils.py:0:0-0:0) 的 [get_openshape_logits()](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/utils/utils.py:338:0-376:73) ([get_logits](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/utils/utils.py:416:0-424:90) 分发器) | `100. * pc_feats @ clip_weights`                     |
| §3.3 Cache build / update            | [model_with_hierarchical_caches.py](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_hierarchical_caches.py:0:0-0:0) 的 [build_cache_in_advance()](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_hierarchical_caches.py:15:0-66:29) + [update_cache()](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:62:0-86:28) | 按预测类别 + entropy 维护字典 cache                  |
| §3.3 Cache logits（Tip-Adapter 核）  | [compute_cache_logits()](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:89:0-122:31) + [compute_local_cache_logits()](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_hierarchical_caches.py:145:0-172:37) | 公式 `exp(-β(1-affinity))` 的实现                    |
| §3.3 Final logits 融合               | [run_test_tda()](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:125:0-177:42) 里的 `final_logits = clip_logits + pos - neg` | 三路相加（zero-shot + 正例 - 反例）                  |
| §3.3 Negative cache entropy gate     | [run_test_tda()](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:125:0-177:42) 里的 `if pos_enabled / if neg_enabled and prop_entropy in (lower, upper)` | "中等不确定性"才加入 neg cache                       |
| 表 4 / 表 5 实验脚本                 | [scripts/eval_model_with_hierarchical_caches.sh](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/scripts/eval_model_with_hierarchical_caches.sh:0:0-0:0) 等 | 用 pueue 批量调度 21+ setting                        |
| 数据集加载                           | `datasets/data_modelnet_c.py` 等<br/>分发在 [utils/utils.py](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/utils/utils.py:0:0-0:0) 的 [build_test_data_loader()](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/utils/utils.py:462:0-537:60) | h5 → DataLoader (batch_size=1)                       |
| 配置超参 (α, β, shot_capacity)       | [configs/modelnet_c.yaml](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/configs/modelnet_c.yaml:0:0-0:0) 等 17 个 yaml | 每个数据集独立调过的超参                             |

---

# 第 2 层：一次 TTA forward 的完整数据流

下面这张流程图就是 [run_test_tda](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:125:0-177:42) 主循环跑**第 i 个**测试样本时**实际**发生的事：

```
            [输入] pc (1, 1024, 3),  rgb (1, 1024, 3)
                       │
                       ▼
       ┌───────────────────────────────────┐
       │  3D Encoder (OpenShape PointBERT) │   ← models/openshape/ppta.py
       │  • 256-token transformer           │
       │  • CLS token       → pc_feats      │   (1, 1280)   全局特征
       │  • 256 patch tokens → KMeans(k=5)  │   (5, 1280)   5 个 patch 中心
       │                    → patch_centers │
       └──────────────┬────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
  pc_feats (1, 1280)            patch_centers (5, 1280)
        │                             │
        ▼                             │
  ┌─────────────┐                     │
  │ CLIP text   │ ← clip_weights      │
  │  similarity │   (1280, 40)        │
  │  × 100      │                     │
  └──────┬──────┘                     │
         │                            │
         ▼                            │
  clip_logits (1, 40) = "zero-shot"   │
         │                            │
         ▼                            │
  loss = softmax_entropy(clip_logits) │
  prop_entropy = loss / log2(40)      │
  pred = argmax(clip_logits)          │
         │                            │
         ▼                            │
  ┌────────────────────────────────────────────────┐
  │           [Cache 更新阶段]                     │
  │ if pred 还有空位 OR loss 比当前最大的小:       │
  │   pos_cache[pred].append([pc_feats, loss])     │
  │   pos_local_cache[pred].append([patch_centers, │
  │                                  loss])        │
  │ if 0.2 < prop_entropy < 0.5:                   │
  │   neg_cache[pred].append([pc_feats, loss,      │
  │                            prob_map])          │
  └────────────────────────────────────────────────┘
         │
         ▼
  ┌────────────────────────────────────────────────┐
  │           [Logits 融合阶段]                    │
  │ final = clip_logits.clone()                    │
  │                                                │
  │ ① 正例全局   final += compute_cache_logits(    │
  │              pc_feats, pos_cache, α=2.0,β=3.0) │
  │                                                │
  │ ② 正例局部   final += compute_local_cache_     │
  │              logits(patch_centers,             │
  │              pos_local_cache, α=2.0, β=3.0)    │
  │                                                │
  │ ③ 反例全局   final -= compute_cache_logits(    │
  │              pc_feats, neg_cache, α=0.117,β=1) │
  │              （注意是减号！）                  │
  └────────────────────────────────────────────────┘
         │
         ▼
  pred_final = argmax(final)
  acc.append(pred_final == target)
```

**重点观察**：
- **三路 logits 加减**就是论文 Fig. 2 那张图的全部
- 同一个样本既参与 cache 更新，又用更新前的 cache 算 logits（自匹配的弱点会被很多样本平均掉）
- `batch_size=1` 是必须的，因为 cache 是 sequential 状态

---

# 第 3 层：核心公式 ↔ 代码行号对应

下面把论文里你能在 PDF 上看到的几个关键公式逐个翻成代码：

## 3.1 Zero-shot logits（论文 Eq. 1）

```
            f_pc · W_text
logits_zs = ─────────────  × 100
            ‖f_pc‖ ‖W_text‖
```

代码（`@/root/autodl-tmp/MCM-PC/Point-Cache/utils/utils.py:349-352`）：

```@/root/autodl-tmp/MCM-PC/Point-Cache/utils/utils.py:349-352
        pc_feats = lm3d_model(xyz, feat)
        pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True)
        # 100 times by jerry
        clip_logits = 100. * pc_feats @ clip_weights
```

`100.` 就是 CLIP 的 `logit_scale`，注释 `100 times by jerry` 是原作者写的。

## 3.2 Cache logits（论文 Eq. 4，**Tip-Adapter style**）

```
                                 ┌                      ┐
logits_cache = α · exp ─β·(1-A)  │  cache_values one-hot │
                                 └                      ┘
其中 A = f_query · K_cacheᵀ      (cosine 因为已 L2-normalized)
```

代码（`@/root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:120-123`）：

```@/root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:120-123
    affinity = pc_feats @ cache_keys
    # cache_logits: (1, n_cls)
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits
```

把式子展开：`(-1)*(β - β·A) = -β·(1-A) = β·A - β` → `exp(β·A - β) = exp(-β)·exp(β·A)`。也就是说：
- **affinity** = 当前样本 query 特征 · 缓存里所有 key 的余弦相似度（一行 n_cls × k_shot 列）
- **β** 控制相似度的"温度"（β 大 → 只有非常像的 cache item 起作用）
- **α** 控制 cache logits 在 final logits 里的权重
- **cache_values** = one-hot 类标签 → 把"affinity 加权和"分发到对应类别上

这就是 Tip-Adapter (ECCV'22) 的核心思想，Point-Cache 直接复用，**没有改公式**，只是把 image features 换成 3D point features，并加了 hierarchical 和 positive/negative 两个变体。

## 3.3 Local（patch）cache logits（论文新增，§3.3）

代码（`@/root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_hierarchical_caches.py:170-172`）：

```@/root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_hierarchical_caches.py:170-172
    affinity = patch_centers.mean(dim=0, keepdim=True) @ local_cache_keys
    # local_cache_logits: (1, n_cls)
    local_cache_logits = ((-1) * (beta - beta * affinity)).exp() @ local_cache_values
```

注意两个**关键差别**：
- **query 端**：把当前样本的 5 个 patch_centers **取平均** 当 query（不是单独算 5 次）
- **key 端**：cache 里**每个 cache item 存了 5 个 patch_centers**，所以 cache_keys 维度是 `(emb_dim, 5 × n_cls × k_shot)`，比 global cache 大 5 倍
- **value 端**：每个 patch 都标该样本的类标签，所以同一个标签被复制了 5 次

直觉解释：**patch 提供"局部线索"**——比如 chair 的椅腿、椅背 patch 长得像 chair；当全局特征被腐蚀（add_global 加了大量噪声点）扰乱时，局部 patch 仍然保有信息。

## 3.4 Negative cache（论文 §3.3 末段）

注意 negative cache 的 value **不是 one-hot**，而是 `prob_map`（softmax 后的连续分布），并且做**阈值掩码**：

```@/root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:109-112
    if neg_mask_thresholds:
        # (n_cls * k_shot, emb_dim)
        cache_values = torch.cat(cache_values, dim=0)
        cache_values = ((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).half().cuda()
```

Mask 阈值 `(0.03, 1.0)` 意思是：保留概率落在 (0.03, 1.0) 的类别为 1，其它为 0。然后这些 1 在最终 `final_logits -= cache_logits` 里**被减掉**——告诉模型"这些类的可能性要降低"。

## 3.5 Cache 更新策略（论文 §3.3 中段）

代码（`@/root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_hierarchical_caches.py:212-216`）：

```@/root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_hierarchical_caches.py:212-216
        if pos_enabled:
            update_cache(pos_cache, pos_local_cache, pred, [pc_feats, patch_centers, loss], pos_params['shot_capacity'])

        if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
            update_cache(neg_cache, neg_local_cache, pred, [pc_feats, None, loss, prob_map], neg_params['shot_capacity'], True)
```

两个准入门槛：

| Cache        | 准入条件                                                     | 论文直觉                                                     |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **positive** | 来一个就尝试入 cache（满了就跟最大 entropy 那个比，更小的替换它） | 选最自信的样本当类原型                                       |
| **negative** | `0.2 < prop_entropy < 0.5`（中等不确定）                     | 不要太自信（已经是好正例了），也不要太混乱（信噪比低）；中等不确定的 prob_map 形如 [0.4 chair, 0.3 table, 0.1 desk, ...]，提供"它**不**是哪个类"的信息 |

[update_cache](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:62:0-86:28) 内部还有个**淘汰机制**：cache 满了之后，新样本必须 entropy **更低**才能挤掉 cache 里 entropy 最大的那个（按 entropy 升序排）。

---

# 第 4 层：超参语义 → 配置文件

`@/root/autodl-tmp/MCM-PC/Point-Cache/configs/modelnet_c.yaml`：

```yaml
positive:
  enabled: True
  shot_capacity: 3      # 每个类最多存 3 个样本
  alpha: 2.0            # 正例 cache logits 在 final 里的权重
  beta: 3.0             # 正例 cache 相似度温度

negative:
  enabled: True
  shot_capacity: 2      # 每个类最多 2 个 neg
  alpha: 0.117          # 反例权重很小（反例信号弱，多了会伤性能）
  beta: 1.0             # 反例温度低（β 越小相似度衰减越慢，affinity 不那么严格）
  entropy_threshold:
    lower: 0.2          # prop_entropy 下界（太自信不入 neg）
    upper: 0.5          # prop_entropy 上界（太混乱不入 neg）
  mask_threshold:
    lower: 0.03         # prob_map 下界（小于此值的类不视为"反例类"）
    upper: 1.0
```

每个数据集都独立调过（[configs/sonn_c.yaml](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/configs/sonn_c.yaml:0:0-0:0)、[configs/snv2_c.yaml](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/configs/snv2_c.yaml:0:0-0:0) 等里的 α, β 不一样），属于**论文报告数前的网格搜索结果**。这一点之后做 ablation 时要小心：α/β 在不同 corruption 上不一定最优。

---

# 第 5 层：和论文图表的具体对照

| 论文里                                           | 代码里                                                       |
| ------------------------------------------------ | ------------------------------------------------------------ |
| **Fig. 2 整体架构图**                            | [run_test_tda](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:125:0-177:42) 主循环就是这张图 |
| **Fig. 3 hierarchical cache 示意**               | KMeans 聚 5 个 patch (`cluster_patches` in [ppta.py](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/models/openshape/ppta.py:0:0-0:0)) |
| **Algorithm 1**                                  | [build_cache_in_advance](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_hierarchical_caches.py:15:0-66:29) + [update_cache](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:62:0-86:28) + [compute_cache_logits](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/runners/model_with_global_cache.py:89:0-122:31) 三个函数串起来 |
| **Table 4 ModelNet-C zero-shot + cache**         | 21 行 = 7 corruption × 3 severity，每行跑一次 [eval_model_with_hierarchical_caches.sh](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/scripts/eval_model_with_hierarchical_caches.sh:0:0-0:0) |
| **Table 5 ScanObjectNN-C**                       | 同上，换 dataset = sonn_c                                    |
| **Table 7 ablation: pos / neg / global / local** | 改 yaml 里 `enabled: True/False` 或改 `--cache-type`         |
| **Table 8 各超参 sensitivity**                   | 改 yaml 里 α/β 的扫描                                        |

---

# 你接下来可能想问的具体问题

挑几个最有价值的：

1. **"5 个 patch 是怎么从 256 个 transformer token 聚出来的？KMeans 怎么定义距离？"** → 看 [models/openshape/ppta.py](cci:7://file:///root/autodl-tmp/MCM-PC/Point-Cache/models/openshape/ppta.py:0:0-0:0) `cluster_patches` 函数（用 sklearn KMeans on token features）
2. **"CLIP text prompts 是哪 80 个 template？"** → [clip_classifier()](cci:1://file:///root/autodl-tmp/MCM-PC/Point-Cache/utils/utils.py:205:0-237:23) 里读 `template`，每个数据集 80 句模板（看 `datasets/templates.py`）
3. **"为什么 build cache 要预先扫一遍 test set？不能完全 online 吗？"** → 论文 §3.3 末段：完全 online 时前 200 个样本 cache 还很空，logits 修正几乎无效；预先 build 是 warm-start，但**预先 build 用的是 zero-shot pred**（弱标签），所以引入了一点 noise——这是论文 Fig. 5 ablation 的对象
4. **"OpenShape backbone 的 256 个 token 是哪 256 个？为什么是 256？"** → `PointPatchTransformer.__init__` 里 `patches=256`，对应 PointNet++ Set Abstraction 把 1024 点降到 256 个 group
5. **"为什么 final = zero-shot + pos - neg，不是别的融合方式？"** → 这是 Tip-Adapter 沿用的设计；论文 Table 8 里有"只用 pos"、"只用 neg"、"用乘法替代加法" 的消融
6. **"21 个 setting 里哪个 corruption 最有挑战？"** → 论文 Table 4：`rotate_2`（旋转）和 `dropout_global_2`（全局丢点）最难，加 cache 涨幅最大；`scale_2`（缩放）最容易，加 cache 涨幅最小

随便挑哪个继续？或者你想现在先跑那个 single-setting global cache 命令把 TTA 实际跑一次，再继续问？



### 运行入口脚本

```bash
/root/autodl-tmp/MCM-PC/Point-Cache/scripts/eval_zs_infer.sh  			# 普通 zero-shot 推理，不用 cache
/root/autodl-tmp/MCM-PC/Point-Cache/scripts/eval_model_with_global_cache.sh			# 只用 global cache
/root/autodl-tmp/MCM-PC/Point-Cache/scripts/eval_model_with_hierarchical_caches.sh   # 用 hierarchical cache，也就是 global + local

对应三个文件：
python runners/zs_infer.py
python runners/model_with_global_cache.py
python runners/model_with_hierarchical_caches.py
```

