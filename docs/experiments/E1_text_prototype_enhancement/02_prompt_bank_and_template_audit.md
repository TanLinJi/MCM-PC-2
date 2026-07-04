# E1 Prompt Bank And Template Audit

更新日期：2026-06-16

## 1. LLM Prompt Bank 规范

E1 重启后，LLM 生成描述的 canonical 保存位置为：

```text
Point-Cache/llm/e1_prompt_bank/
```

已迁移的已有缓存：

| 数据集 | 文件 |
|---|---|
| ModelNet-C | `modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json` |
| ShapeNet-C / SNV2-C | `snv2_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json` |
| ScanObjNN-C / SONN-C | `sonn_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json` |

旧 results 下的 `shared_prompts/` 暂时保留，只作为历史结果追溯，不再作为新实验默认路径。

## 2. manual_full 模板统计

当前 `manual_full` 来自：

```text
Point-Cache/datasets/templates.py
```

统计口径使用代码中的 image-style keyword 集合：

```text
photo, image, picture, painting, sketch, cartoon, drawing, rendering, screenshot, view, camera
```

按这个口径：

| 项 | 数量 |
|---|---:|
| `manual_full` 总模板数 | 64 |
| 命中 image-style / 2D keyword 的模板数 | 50 |
| 未命中 image-style keyword 的模板数 | 14 |
| 当前 `manual_3d` 严格过滤后保留模板数 | 3 |

解释：

```text
manual_full 中约 78.1% 的模板是 image-style / 2D 风格。
manual_3d 只剩 3 条，语义覆盖过窄。
```

这与 E1 结果一致：`manual_3d = 35.63`，明显低于 `manual_full = 47.68`。因此 2D/CLIP-style prompt 不是冗余噪声，而是 ULIP 文本空间的重要语义锚点。

## 3. 2D Image-Style 模板列表

以下 50 条命中 image-style keyword：

1. `a photo of a {} in the scene.`
2. `a photo of the {} in the scene.`
3. `a photo of one {} in the scene.`
4. `a photo of a {}.`
5. `a photo of my {}.`
6. `a photo of the {}.`
7. `a photo of one {}.`
8. `a photo of many {}.`
9. `a good photo of a {}.`
10. `a good photo of the {}.`
11. `a bad photo of a {}.`
12. `a bad photo of the {}.`
13. `a photo of a nice {}.`
14. `a photo of the nice {}.`
15. `a photo of a cool {}.`
16. `a photo of the cool {}.`
17. `a photo of a weird {}.`
18. `a photo of the weird {}.`
19. `a photo of a small {}.`
20. `a photo of the small {}.`
21. `a photo of a large {}.`
22. `a photo of the large {}.`
23. `a photo of a clean {}.`
24. `a photo of the clean {}.`
25. `a photo of a dirty {}.`
26. `a photo of the dirty {}.`
27. `a bright photo of a {}.`
28. `a bright photo of the {}.`
29. `a dark photo of a {}.`
30. `a dark photo of the {}.`
31. `a photo of a hard to see {}.`
32. `a photo of the hard to see {}.`
33. `a low resolution photo of a {}.`
34. `a low resolution photo of the {}.`
35. `a cropped photo of a {}.`
36. `a cropped photo of the {}.`
37. `a close-up photo of a {}.`
38. `a close-up photo of the {}.`
39. `a jpeg corrupted photo of a {}.`
40. `a jpeg corrupted photo of the {}.`
41. `a blurry photo of a {}.`
42. `a blurry photo of the {}.`
43. `a pixelated photo of a {}.`
44. `a pixelated photo of the {}.`
45. `a black and white photo of the {}.`
46. `a black and white photo of a {}`
47. `a cartoon {}.`
48. `the cartoon {}.`
49. `a painting of the {}.`
50. `a painting of a {}.`
