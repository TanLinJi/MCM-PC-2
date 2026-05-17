## 图1复现

### 相关知识

以OpenShape列为例:

| 柱子           | 颜色                                                         | 含义                                        | 数据规模                |
| :------------- | :----------------------------------------------------------- | :------------------------------------------ | :---------------------- |
| **绿** = 84.56 | clean ModelNet40 zero-shot                                   | 1 次                                        | 一个数                  |
| **橙** = 73.49 | ModelNet-C zero-shot 全 corruption 平均                      | **7 corruption × 5 severity = 35 次**取均值 | 35 次跑出 35 个数取平均 |
| **紫** = 76.59 | ModelNet-C + Point-Cache **hierarchical** 全 corruption 平均 | 35 次取均值                                 | 35 次跑出 35 个数取平均 |



**7 种损坏类型 × 5 个严重等级**：

| 损坏类型         | 含义                 | severity 0-4 大致幅度          |
| :--------------- | :------------------- | :----------------------------- |
| `add_global`     | 整个空间均匀加噪声点 | +10 ~ +50 个噪声点             |
| `add_local`      | 在物体周围加噪声点   | +100 ~ +500 个噪声点           |
| `dropout_global` | 随机丢点             | 剩 768 / 640 / 512 / 384 / 256 |
| `dropout_local`  | 局部丢点（割掉一块） | 剩 924 ~ 524                   |
| `jitter`         | 每个点加小高斯扰动   | 扰动方差从小到大               |
| `rotate`         | 整体绕轴旋转         | 旋转角度从小到大               |
| `scale`          | 整体缩放             | 缩放因子从小到大               |

**5 个严重等级** = severity 0、1、2、3、4（"轻 → 重"），文件名后缀就是这个数字。

ModelNet-C benchmark [Ren et al. 2022] 的标准评测协议就是 **35 setting 算术平均**。所以 bar 2 的橙柱（73.49）= 35 个 zero-shot 数字的平均。



Figure 1(a) 有 3 根柱子，按论文图例颜色：

| 命名      | 含义                                                | 论文期望（OpenShape 列） |
| :-------- | :-------------------------------------------------- | :----------------------- |
| **bar 1** | 绿柱：ModelNet40 clean zero-shot                    | 84.56                    |
| **bar 2** | 橙柱：ModelNet-C zero-shot 35-mean                  | **73.49** ← 你刚跑的     |
| **bar 3** | 紫柱：ModelNet-C + Point-Cache hierarchical 35-mean | 76.59                    |



### OpenShape列

#### ModelNet, Zero-shot

##### 命令：

```bash
cd /root/autodl-tmp/MCM-PC/Point-Cache

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
python runners/zs_infer.py \
    --config configs \   # 超参数所在目录
    --lm3d openshape \   # 选那个3D多模态backbone, 可选值：openshape/uni3d/ulip
    --cache-type global \	# 使用哪种cache模型，可选值：global/local/hierarchical/vis
    --ckpt_path weights/openshape/openshape-pointbert-vitg14-rgb/model.pt \  # 3D backbone的权重路径
    --dataset modelnet40 \	# 数据集
    --npoints 1024 \		# 输入点云的点数，1024 / 4096 / 16384（OmniObject3D 需要）
    --oshape-version vitg14 \  # OpenShape 用哪个 CLIP-text backbone， OpenShape 专属（--lm3d openshape 才用）
    --wandb-log
```

##### 结果:

```bash
---- ***Final*** Zero-shot test accuracy: 83.27. ----

wandb: 
wandb: Run history:
wandb: Averaged test accuracy █▁▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:             modelnet40 ▁
wandb: 
wandb: Run summary:
wandb: Averaged test accuracy 83.2658
wandb:             modelnet40 83.2658
wandb: 
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /root/autodl-tmp/MCM-PC/Point-Cache/wandb/offline-run-20260510_115449-vgvrjtse
wandb: Find logs at: ./wandb/offline-run-20260510_115449-vgvrjtse/logs
```

`实验值：83.2658，论文值：84.56`



#### ModelNet-c，Zero-shot

##### 命令：

```bash
cd /root/autodl-tmp/MCM-PC
bash Point-Cache/scripts/repro_fig1a_bar2_zs_corruption.sh
```

##### 结果：

```
                  sev=0    sev=1    sev=2    sev=3    sev=4   |  per-cor mean
add_global        78.85    74.72    71.47    69.57    68.15   |   72.55
add_local         74.84    70.87    67.50    64.71    63.37   |   68.26
dropout_global    83.75    83.06    81.24    78.61    63.57   |   78.05
dropout_local     80.47    77.96    73.46    68.03    60.70   |   72.12
jitter            79.29    71.39    59.76    45.75    32.66   |   57.77   ← 最脆弱
rotate            84.12    83.71    82.54    79.21    72.33   |   80.38   ← 最稳健
scale             80.23    78.81    78.57    77.92    76.82   |   ~78.4
```

```
=== fig1a_bar2_20260510_122450  (35 runs) ===
  add_global_0                       78.85
  add_global_1                       74.72
  add_global_2                       71.47
  add_global_3                       69.57
  add_global_4                       68.15
  add_local_0                        74.84
  add_local_1                        70.87
  add_local_2                        67.50
  add_local_3                        64.71
  add_local_4                        63.37
  dropout_global_0                   83.75
  dropout_global_1                   83.06
  dropout_global_2                   81.24
  dropout_global_3                   78.61
  dropout_global_4                   63.57
  dropout_local_0                    80.47
  dropout_local_1                    77.96
  dropout_local_2                    73.46
  dropout_local_3                    68.03
  dropout_local_4                    60.70
  jitter_0                           79.29
  jitter_1                           71.39
  jitter_2                           59.76
  jitter_3                           45.75
  jitter_4                           32.66
  rotate_0                           84.12
  rotate_1                           83.71
  rotate_2                           82.54
  rotate_3                           79.21
  rotate_4                           72.33
  scale_0                            80.23
  scale_1                            78.81
  scale_2                            78.57
  scale_3                            77.92
  scale_4                            76.82

  >>> mean over 35 runs: 72.51
```

`实验值：72.51；论文值：73.49；`

 

#### ModelNet-c，ZS+PointCache

##### smoke test

```
cd /root/autodl-tmp/MCM-PC
bash Point-Cache/scripts/repro_fig1a_bar3_tta.sh
```

```bash
(mcmpc) root@autodl-container-4a7f45b869-e38322e3:~/autodl-tmp/MCM-PC/Point-Cache# WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
python runners/model_with_hierarchical_caches.py \
    --config configs --lm3d openshape --cache-type hierarchical \
    --ckpt_path weights/openshape/openshape-pointbert-vitg14-rgb/model.pt \
    --dataset modelnet_c --cor_type add_global_2 \
    --npoints 1024 --oshape-version vitg14 --wandb-log \
    > logs/fig1a_bar3_smoketest/add_global_2.log 2>&1
(mcmpc) root@autodl-container-4a7f45b869-e38322e3:~/autodl-tmp/MCM-PC/Point-Cache# grep -E 'Final|Error|Traceback' logs/fig1a_bar3_smoketest/add_global_2.log
---- ***Final*** TDA's test accuracy: 74.80. ----
```

##### 命令：

```bash
cd /root/autodl-tmp/MCM-PC
bash Point-Cache/scripts/repro_fig1a_bar3_tta.sh
```

##### 结果：

```
[fig1a-bar3] all 35 jobs done at Sun May 10 19:05:56 CST 2026
[fig1a-bar3] running summary...

Log root: /root/autodl-tmp/MCM-PC/Point-Cache/logs/fig1a_bar3_20260510_152446

=== fig1a_bar3_20260510_152446  (35 runs) ===
  add_global_0                       79.90
  add_global_1                       76.22
  add_global_2                       74.80
  add_global_3                       71.35
  add_global_4                       70.91
  add_local_0                        77.43
  add_local_1                        75.45
  add_local_2                        73.82
  add_local_3                        72.12
  add_local_4                        70.95
  dropout_global_0                   83.10
  dropout_global_1                   84.00
  dropout_global_2                   82.25
  dropout_global_3                   80.11
  dropout_global_4                   65.19
  dropout_local_0                    82.25
  dropout_local_1                    80.31
  dropout_local_2                    76.66
  dropout_local_3                    71.96
  dropout_local_4                    66.86
  jitter_0                           80.06
  jitter_1                           75.08
  jitter_2                           68.80
  jitter_3                           54.74
  jitter_4                           45.06
  rotate_0                           84.97
  rotate_1                           83.79
  rotate_2                           82.82
  rotate_3                           79.34
  rotate_4                           73.87
  scale_0                            79.70
  scale_1                            79.17
  scale_2                            78.16
  scale_3                            77.19
  scale_4                            76.13

  >>> mean over 35 runs: 75.27
```

`实验值：75.27；论文值：76.59；`



`最有信息量的发现 — TTA 在哪里涨/在哪里反而退化`

Per-corruption 涨幅（bar 3 mean − bar 2 mean）：

| corruption     | bar 2 mean | bar 3 mean | Δ         | 解读                           |
| :------------- | :--------- | :--------- | :-------- | :----------------------------- |
| jitter         | 57.77      | **64.75**  | **+6.98** | TTA 拯救最多（OpenShape 软肋） |
| add_local      | 68.26      | 73.95      | +5.69     | 第二大涨幅                     |
| dropout_local  | 72.12      | 75.61      | +3.49     | 合理                           |
| add_global     | 72.55      | 74.64      | +2.09     | 中规中矩                       |
| dropout_global | 78.05      | 78.93      | +0.88     | 本来就好                       |
| rotate         | 80.38      | 80.96      | +0.58     | 几乎无增益                     |
| **scale**      | **78.47**  | **78.07**  | **-0.40** | ⚠️ **TTA 反而退化**             |

两个关键观察

1. **jitter 上 +6.98pp**：Point-Cache 的 hierarchical TTA 已经把"特征不可靠"的问题缓解了大半。MCP-3D 想在这里继续 attack，要拿出"几何距离 (ICP-CD) 比特征余弦更稳健"的具体证据 — 这是 W2.5 P1 探针实验该回答的。
2. **scale 上 −0.40pp**：**这是 MCP-3D 最 promising 的 attack point**。Point-Cache 的 cache 检索基于特征相似度，但 scale 扰动后特征会偏移整片 manifold（不是局部噪声），cache 里的 historic key 反而误导预测。**ICP-CD 几何距离能矫正这种全局扰动**——这给 W2.5 P2/P4 探针实验明确了切入点。