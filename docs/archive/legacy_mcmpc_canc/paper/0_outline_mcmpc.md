# MCP-3D 论文章节大纲与写作进度

> **路径**：`/root/autodl-tmp/MCM-PC/docs/paper/`
>
> **原则 (decisions.md D12)**：每完成一个实验里程碑立即写对应段落，不延后。

## 章节列表

| 章节 | 内容 | 触发条件 | 状态 | 字数目标 |
|---|---|---|---|---|
| §1 Introduction | 动机 + 贡献概览 | W2.5 P1+P2 完成 | ⏳ blocked | ~1000 |
| §2 Related Work | 3D 测试时适配 / 零样本 / 缓存架构 | 已有 F1-F5 | 🟡 **v0.1 草稿** | ~1500 |
| §3 Method | C1 ICP-CD / C2 vMF / C3 2×3 矩阵 | W3-W5 各自完成 | ⏳ blocked | ~3000 |
| §4 Experiments | ModelNet-C / ScanObjectNN-C / 消融 | W6-W10 完成 | ⏳ blocked | ~2500 |
| §5 Discussion | 失败案例 / 紧致度诊断 / 局限 | W11-W12 完成 | ⏳ blocked | ~1500 |
| §6 Conclusion | 总结 | W12 完成 | ⏳ blocked | ~500 |

## 触发 → 写作映射表

```
W2.5 P1 完成 → §1 motivation 段：填入 F3 (rotate +0.58pp) + P1 旋转鲁棒性测量
W2.5 P2 完成 → §1 motivation 段：填入 F1 (scale -0.40pp) + P2 紧致度相关性 r
W2.5 P5 完成 → §2.1 末段：填入跨方法 scale 退化证据
W3 完成     → §3.3 vMF 锚点
W4 完成     → §3.4 ICP-CD 距离
W5 完成     → §3.5 2×3 矩阵
W6 主实验   → §4.1 ModelNet-C 主结果
W7 真实场景 → §4.2 ScanObjectNN-C
W8 消融     → §4.3 消融
W9 紧致度诊断 → §5.1 诊断分析
W10 BayesMM 对照 → §5.2 跨域对比
W11-12      → §5.3 局限 + §6 结论
W13-14      → 整合 + abstract + intro 收尾
W15-16      → 投稿润色 (AAAI)
```

## 写作时间预算

- 每周 3-5 小时纯写作，与实验并行
- 每段写完即 commit：`docs(paper): write §X.Y first draft`
- 每章节写完打 tag：`paper-§X-draft`

## 章节依赖图

```
        §2 Related Work (NOW, v0.1)
              ↓ 引用
        §3 Method ← W3/W4/W5 里程碑
              ↓
        §4 Experiments ← W6-W10
              ↓ 反向回填
        §1 Introduction
              ↓
        §5 Discussion → §6 Conclusion
```

## 当前进度
- §2 Related Work draft v0.1 (本次 2026-05-10 落盘) - 见 `02_related_work.md`
