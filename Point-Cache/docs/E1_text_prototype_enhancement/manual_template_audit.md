# Manual Template Audit

更新日期：2026-06-16

## 审计对象

`manual_full` 对应 `Point-Cache/datasets/templates.py` 中的 `text_prompts`。

E1 使用该集合作为静态文本原型分支：

```text
manual_full + LLM
```

E1 不把 `manual_3d` 作为主实验路线；`manual_3d` 只用于理解 `manual_full` 中真正 3D / pointcloud 风格模板的比例。

该审计是准备任务，不占用 E1 实验编号。

## 统计结果

| 项目 | 数量 | 说明 |
|---|---:|---|
| `manual_full` 总模板数 | 64 | `text_prompts` 全量 |
| image-style 模板数，源码关键词口径 | 50 | 命中 `photo`、`image`、`picture`、`painting`、`sketch`、`cartoon` 等关键词 |
| image-style 模板数，语义扩展口径 | 53 | 在源码关键词口径基础上，把 3 条 `itap` 也视为拍照语义 |
| 非 image-style 模板数，源码关键词口径 | 14 | 未命中 image-style 关键词 |
| `manual_3d` 模板数，源码规则 | 3 | 命中 3D/style 关键词且不命中 image-style 关键词 |

结论：按项目当前源码关键词口径，`manual_full` 中 2D image-style 模板有 **50** 条；如果把 `itap` 也视为 image-style，扩展口径为 **53** 条。

## 源码规则

`manual_3d` 的源码筛选逻辑是：

```text
命中 _3D_STYLE_PROMPT_KEYWORDS
并且不命中 _IMAGE_STYLE_PROMPT_KEYWORDS
```

在当前 `manual_full` 中，最终只保留 3 条：

```text
a point cloud model of {}.
There is a {} in the scene.
There is the {} in the scene.
```

这说明 `manual_full` 虽然是 3D zero-shot 的默认手工模板集合，但其文本表面形式高度偏向 2D image / photo 风格。

## 非 image-style 模板清单，源码关键词口径

```text
a point cloud model of {}.
There is a {} in the scene.
There is the {} in the scene.
itap of a {}.
itap of my {}.
itap of the {}.
a plastic {}.
the plastic {}.
a toy {}.
the toy {}.
a plushie {}.
the plushie {}.
an embroidered {}.
the embroidered {}.
```

其中 `itap` 是 “I took a picture” 的缩写，语义上仍偏 image / photo。因此 E1 后续记录结果时默认报告两套口径：

- 严格源码关键词口径：50 条 image-style；
- 语义扩展口径：53 条 image-style。

## 对 E1 的启发

1. `manual_full` 已经大量依赖 image-style 语言，因此 E1 的 LLM 描述不能只继续堆叠 image-style 描述。
2. 15 prompts 消融必须显式比较：
   - 10 image + 5 pointcloud；
   - 5 image + 10 pointcloud。
3. 如果 5 image + 10 pointcloud 优于 10 image + 5 pointcloud，说明 LLM 的价值更可能来自补充点云几何语义。
4. 如果 10 image + 5 pointcloud 更优，说明 CLIP/ULIP 的文本侧仍更受益于 image-level 类别语义。
5. `manual_3d` 只有 3 条模板，表达能力过窄，不适合作为 E1 主分支替代 `manual_full`。
