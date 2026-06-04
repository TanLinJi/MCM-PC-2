# E2 Runner 兼容性检查

检查目标：

确认当前代码是否支持：

1. `manual_full` 与 `manual_full_llm_fusion` 两种文本方法；
2. `zs_global` 与 `zs_global_local` 两种 Point-Cache 设置；
3. E1 的共享 LLM prompt 缓存；
4. 在 Point-Cache 缓存 runner 中正确调用 E1 的文本原型构造逻辑。

## 自动检查结果

- [PASS] utils.py E1 prompt arguments: all required prompt arguments found
- [PASS] prompt_utils.py prompt source support: manual_full and manualfull_llm_dynamic_init supported
- [PASS] modelnet_c.py uses get_prompt_template with classnames: get_prompt_template and self.classnames found
- [PASS] Point-Cache/runners/model_with_global_cache.py calls dataset loader and clip_classifier: build_test_data_loader=True, clip_classifier=True
- [PASS] Point-Cache/runners/model_with_hierarchical_caches.py calls dataset loader and clip_classifier: build_test_data_loader=True, clip_classifier=True
- [PASS] Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py method support: zs_global=True, zs_global_local=True, prompt_source_ref=False, clip_classifier=True
- [PASS] Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py method support: zs_global=True, zs_global_local=True, prompt_source_ref=False, clip_classifier=True
- [PASS] E1 shared prompt cache complete: class_count=40, missing=[]

## 初步判断

所有关键检查均通过。下一步可以直接编写 E2 smoke test 脚本。

## 关键源码检索

### 1. prompt-source / LLM fusion 相关

Point-Cache/utils/utils.py:66:        '--prompt-source',
Point-Cache/utils/utils.py:68:        default='manual_full',
Point-Cache/utils/utils.py:70:            'manual_full',
Point-Cache/utils/utils.py:74:            'manualfull_llm_dynamic_init',
Point-Cache/utils/utils.py:103:        '--prompt-static-weight',
Point-Cache/utils/utils.py:109:        '--prompt-dynamic-weight',
Point-Cache/utils/utils.py:115:        '--prompt-cache-dir',
Point-Cache/utils/utils.py:326:       所有类别共用的手工模板，例如 manual_full / manual_3d。
Point-Cache/datasets/prompt_utils.py:12:    manual_full:
Point-Cache/datasets/prompt_utils.py:18:    manualfull_llm_dynamic_init:
Point-Cache/datasets/prompt_utils.py:19:        manual_full 分支 + LLM 动态描述分支加权融合。
Point-Cache/datasets/prompt_utils.py:22:    prompt_source = getattr(cfg, "prompt_source", "manual_full")
Point-Cache/datasets/prompt_utils.py:24:    if prompt_source == "manual_full":
Point-Cache/datasets/prompt_utils.py:39:                "manualfull_llm_dynamic_init",
Point-Cache/datasets/prompt_utils.py:56:        if prompt_source == "manualfull_llm_dynamic_init":

### 2. cache runner 调用文本原型的位置

Point-Cache/runners/model_with_global_cache.py:205:    test_loader, classnames, template = build_test_data_loader(args, dataset_name, args.data_root, preprocess)
Point-Cache/runners/model_with_global_cache.py:210:    clip_weights = clip_classifier(args, classnames, template, clip_model)
Point-Cache/runners/model_with_hierarchical_caches.py:257:    test_loader, classnames, template = build_test_data_loader(args, dataset_name, args.data_root, preprocess)
Point-Cache/runners/model_with_hierarchical_caches.py:262:    clip_weights = clip_classifier(args, classnames, template, clip_model)
Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py:23:    build_test_data_loader,
Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py:141:        clip_weights_state["clip_weights"] = clip_classifier(args, classnames, template, clip_model)
Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py:192:                test_loader, classnames, template = build_test_data_loader(args, args.dataset, args.data_root, None)
Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py:23:    build_test_data_loader,
Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py:141:        clip_weights_state["clip_weights"] = clip_classifier(args, classnames, template, clip_model)
Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py:192:                test_loader, classnames, template = build_test_data_loader(args, args.dataset, args.data_root, None)

### 3. zs / zs_global / zs_global_local 支持情况

Point-Cache/runners/model_with_hierarchical_caches_e4_canc_v1.py:29:from model_with_hierarchical_caches import (
Point-Cache/runners/model_with_hierarchical_caches_e4_canc_diag_v2.py:25:from model_with_hierarchical_caches import (
Point-Cache/runners/model_with_hierarchical_caches_e4_canc_diag.py:31:from model_with_hierarchical_caches import (
Point-Cache/runners/model_with_hierarchical_caches_e4_canc_v0.py:29:from model_with_hierarchical_caches import (
Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py:29:from runners.model_with_global_cache import run_test_tda as run_global_cache
Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py:30:from runners.model_with_hierarchical_caches import run_test_tda as run_hierarchical_cache
Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py:63:    baseline_parser.add_argument("--baseline-method", required=True, choices=["zs", "zs_global", "zs_global_local"])
Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py:64:    baseline_parser.add_argument("--baseline-method-full", required=True)
Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py:204:                elif baseline_method == "zs_global":
Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py:206:                elif baseline_method == "zs_global_local":
Point-Cache/runners/E1_text_prototype_enhancement/run_e1_ulip_modelnetc_s2_zs_prompt_ablation.py:291:        if method in ["zs_global", "zs_global_local"]:
Point-Cache/runners/baseline/run_openshape_modelnetc_corruptions_all35.py:29:from runners.model_with_global_cache import run_test_tda as run_global_cache
Point-Cache/runners/baseline/run_openshape_modelnetc_corruptions_all35.py:30:from runners.model_with_hierarchical_caches import run_test_tda as run_hierarchical_cache
Point-Cache/runners/baseline/run_openshape_modelnetc_corruptions_all35.py:63:    baseline_parser.add_argument("--baseline-method", required=True, choices=["zs", "zs_global", "zs_global_local"])
Point-Cache/runners/baseline/run_openshape_modelnetc_corruptions_all35.py:64:    baseline_parser.add_argument("--baseline-method-full", required=True)
Point-Cache/runners/baseline/run_openshape_modelnetc_corruptions_all35.py:204:                elif baseline_method == "zs_global":
Point-Cache/runners/baseline/run_openshape_modelnetc_corruptions_all35.py:206:                elif baseline_method == "zs_global_local":
Point-Cache/runners/baseline/run_openshape_modelnetc_corruptions_all35.py:291:        if method in ["zs_global", "zs_global_local"]:
Point-Cache/runners/baseline/run_uni3d_modelnetc_corruptions_all35.py:29:from runners.model_with_global_cache import run_test_tda as run_global_cache
Point-Cache/runners/baseline/run_uni3d_modelnetc_corruptions_all35.py:30:from runners.model_with_hierarchical_caches import run_test_tda as run_hierarchical_cache
Point-Cache/runners/baseline/run_uni3d_modelnetc_corruptions_all35.py:63:    baseline_parser.add_argument("--baseline-method", required=True, choices=["zs", "zs_global", "zs_global_local"])
Point-Cache/runners/baseline/run_uni3d_modelnetc_corruptions_all35.py:64:    baseline_parser.add_argument("--baseline-method-full", required=True)
Point-Cache/runners/baseline/run_uni3d_modelnetc_corruptions_all35.py:204:                elif baseline_method == "zs_global":
Point-Cache/runners/baseline/run_uni3d_modelnetc_corruptions_all35.py:206:                elif baseline_method == "zs_global_local":
Point-Cache/runners/baseline/run_uni3d_modelnetc_corruptions_all35.py:291:        if method in ["zs_global", "zs_global_local"]:
Point-Cache/runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py:29:from runners.model_with_global_cache import run_test_tda as run_global_cache
Point-Cache/runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py:30:from runners.model_with_hierarchical_caches import run_test_tda as run_hierarchical_cache
Point-Cache/runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py:63:    baseline_parser.add_argument("--baseline-method", required=True, choices=["zs", "zs_global", "zs_global_local"])
Point-Cache/runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py:64:    baseline_parser.add_argument("--baseline-method-full", required=True)
Point-Cache/runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py:205:                elif baseline_method == "zs_global":
Point-Cache/runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py:207:                elif baseline_method == "zs_global_local":
Point-Cache/runners/baseline/run_ulip2_scanobjnnc_hardest_corruptions_all35.py:294:        if method in ["zs_global", "zs_global_local"]:
Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py:29:from runners.model_with_global_cache import run_test_tda as run_global_cache
Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py:30:from runners.model_with_hierarchical_caches import run_test_tda as run_hierarchical_cache
Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py:63:    baseline_parser.add_argument("--baseline-method", required=True, choices=["zs", "zs_global", "zs_global_local"])
Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py:64:    baseline_parser.add_argument("--baseline-method-full", required=True)
Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py:204:                elif baseline_method == "zs_global":
Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py:206:                elif baseline_method == "zs_global_local":
Point-Cache/runners/baseline/run_ulip_modelnetc_corruptions_all35.py:291:        if method in ["zs_global", "zs_global_local"]:
Point-Cache/runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py:29:from runners.model_with_global_cache import run_test_tda as run_global_cache
Point-Cache/runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py:30:from runners.model_with_hierarchical_caches import run_test_tda as run_hierarchical_cache
Point-Cache/runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py:63:    baseline_parser.add_argument("--baseline-method", required=True, choices=["zs", "zs_global", "zs_global_local"])
Point-Cache/runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py:64:    baseline_parser.add_argument("--baseline-method-full", required=True)
Point-Cache/runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py:205:                elif baseline_method == "zs_global":
Point-Cache/runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py:207:                elif baseline_method == "zs_global_local":
Point-Cache/runners/baseline/run_ulip_scanobjnnc_hardest_corruptions_all35.py:294:        if method in ["zs_global", "zs_global_local"]:
Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py:29:from runners.model_with_global_cache import run_test_tda as run_global_cache
Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py:30:from runners.model_with_hierarchical_caches import run_test_tda as run_hierarchical_cache
Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py:63:    baseline_parser.add_argument("--baseline-method", required=True, choices=["zs", "zs_global", "zs_global_local"])
Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py:64:    baseline_parser.add_argument("--baseline-method-full", required=True)
Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py:204:                elif baseline_method == "zs_global":
Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py:206:                elif baseline_method == "zs_global_local":
Point-Cache/runners/baseline/run_uni3d_scanobjnnc_hardest_corruptions_all35.py:321:        if method in ["zs_global", "zs_global_local"]:
Point-Cache/runners/baseline/run_ulip2_modelnetc_corruptions_all35.py:29:from runners.model_with_global_cache import run_test_tda as run_global_cache
Point-Cache/runners/baseline/run_ulip2_modelnetc_corruptions_all35.py:30:from runners.model_with_hierarchical_caches import run_test_tda as run_hierarchical_cache
Point-Cache/runners/baseline/run_ulip2_modelnetc_corruptions_all35.py:63:    baseline_parser.add_argument("--baseline-method", required=True, choices=["zs", "zs_global", "zs_global_local"])
Point-Cache/runners/baseline/run_ulip2_modelnetc_corruptions_all35.py:64:    baseline_parser.add_argument("--baseline-method-full", required=True)
Point-Cache/runners/baseline/run_ulip2_modelnetc_corruptions_all35.py:204:                elif baseline_method == "zs_global":
Point-Cache/runners/baseline/run_ulip2_modelnetc_corruptions_all35.py:206:                elif baseline_method == "zs_global_local":
Point-Cache/runners/baseline/run_ulip2_modelnetc_corruptions_all35.py:291:        if method in ["zs_global", "zs_global_local"]:
Point-Cache/runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py:29:from runners.model_with_global_cache import run_test_tda as run_global_cache
Point-Cache/runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py:30:from runners.model_with_hierarchical_caches import run_test_tda as run_hierarchical_cache
Point-Cache/runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py:63:    baseline_parser.add_argument("--baseline-method", required=True, choices=["zs", "zs_global", "zs_global_local"])
Point-Cache/runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py:64:    baseline_parser.add_argument("--baseline-method-full", required=True)
Point-Cache/runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py:205:                elif baseline_method == "zs_global":
Point-Cache/runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py:207:                elif baseline_method == "zs_global_local":
Point-Cache/runners/baseline/run_openshape_scanobjnnc_hardest_corruptions_all35.py:294:        if method in ["zs_global", "zs_global_local"]:
Point-Cache/scripts/eval_model_with_hierarchical_caches_mem.sh:26:    python runners/model_with_hierarchical_caches_mem.py \
Point-Cache/scripts/eval_model_with_hierarchical_caches_mem.sh:44:    python runners/model_with_hierarchical_caches_mem.py \
Point-Cache/scripts/eval_model_with_hierarchical_caches_mem.sh:58:    python runners/model_with_hierarchical_caches_mem.py \
Point-Cache/scripts/eval_model_with_global_cache_ablate_seed.sh:27:    python runners/model_with_global_cache_ablate_seed.py \
Point-Cache/scripts/eval_model_with_global_cache_ablate_seed.sh:47:    python runners/model_with_global_cache_ablate_seed.py \
Point-Cache/scripts/eval_model_with_global_cache_ablate_seed.sh:63:    python runners/model_with_global_cache_ablate_seed.py \
Point-Cache/scripts/eval_model_with_global_cache_mem.sh:26:    python runners/model_with_global_cache_mem.py \
Point-Cache/scripts/eval_model_with_global_cache_mem.sh:44:    python runners/model_with_global_cache_mem.py \
Point-Cache/scripts/eval_model_with_global_cache_mem.sh:58:    python runners/model_with_global_cache_mem.py \
Point-Cache/scripts/E0_baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global_single_gpu.sh:8:  "33_2_uni3d_scanobjnn_clean_hardest_zs_global" \
Point-Cache/scripts/E0_baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/33_2_uni3d_scanobjnn_clean_hardest_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/12_run_ulip2_modelnetc_corruptions_all35_common.sh:46:  --baseline-method "${METHOD}" \
Point-Cache/scripts/E0_baseline/12_run_ulip2_modelnetc_corruptions_all35_common.sh:47:  --baseline-method-full "${METHOD_FULL}" \
Point-Cache/scripts/E0_baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh:8:  "32_3_uni3d_modelnetc_corruptions_all35_zs_global_local" \
Point-Cache/scripts/E0_baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/32_3_uni3d_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/31_3_uni3d_modelnet_clean_zs_global_local_single_gpu.sh:8:  "31_3_uni3d_modelnet_clean_zs_global_local" \
Point-Cache/scripts/E0_baseline/31_3_uni3d_modelnet_clean_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/31_3_uni3d_modelnet_clean_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh:8:  "14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local" \
Point-Cache/scripts/E0_baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/14_3_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh:8:  "02_3_ulip_modelnetc_corruptions_all35_zs_global_local" \
Point-Cache/scripts/E0_baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/02_3_ulip_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh:8:  "24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global" \
Point-Cache/scripts/E0_baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/24_2_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/31_2_uni3d_modelnet_clean_zs_global_single_gpu.sh:8:  "31_2_uni3d_modelnet_clean_zs_global" \
Point-Cache/scripts/E0_baseline/31_2_uni3d_modelnet_clean_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/31_2_uni3d_modelnet_clean_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/14_run_ulip2_scanobjnnc_hardest_corruptions_all35_common.sh:46:  --baseline-method "${METHOD}" \
Point-Cache/scripts/E0_baseline/14_run_ulip2_scanobjnnc_hardest_corruptions_all35_common.sh:47:  --baseline-method-full "${METHOD_FULL}" \
Point-Cache/scripts/E0_baseline/24_run_openshape_scanobjnnc_hardest_corruptions_all35_common.sh:47:  --baseline-method "${METHOD}" \
Point-Cache/scripts/E0_baseline/24_run_openshape_scanobjnnc_hardest_corruptions_all35_common.sh:48:  --baseline-method-full "${METHOD_FULL}" \
Point-Cache/scripts/E0_baseline/02_run_ulip_modelnetc_corruptions_all35_common.sh:44:  --baseline-method "${METHOD}" \
Point-Cache/scripts/E0_baseline/02_run_ulip_modelnetc_corruptions_all35_common.sh:45:  --baseline-method-full "${METHOD_FULL}" \
Point-Cache/scripts/E0_baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh:8:  "12_3_ulip2_modelnetc_corruptions_all35_zs_global_local" \
Point-Cache/scripts/E0_baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/12_3_ulip2_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/21_2_openshape_modelnet_clean_zs_global_single_gpu.sh:8:  "21_2_openshape_modelnet_clean_zs_global" \
Point-Cache/scripts/E0_baseline/21_2_openshape_modelnet_clean_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/21_2_openshape_modelnet_clean_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh:8:  "04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global" \
Point-Cache/scripts/E0_baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/04_2_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh:8:  "14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global" \
Point-Cache/scripts/E0_baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/14_2_ulip2_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/11_3_ulip2_modelnet_clean_zs_global_local_single_gpu.sh:8:  "11_3_ulip2_modelnet_clean_zs_global_local" \
Point-Cache/scripts/E0_baseline/11_3_ulip2_modelnet_clean_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/11_3_ulip2_modelnet_clean_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global_single_gpu.sh:8:  "32_2_uni3d_modelnetc_corruptions_all35_zs_global" \
Point-Cache/scripts/E0_baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/32_2_uni3d_modelnetc_corruptions_all35_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global_single_gpu.sh:8:  "02_2_ulip_modelnetc_corruptions_all35_zs_global" \
Point-Cache/scripts/E0_baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/02_2_ulip_modelnetc_corruptions_all35_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/04_run_ulip_scanobjnnc_hardest_corruptions_all35_common.sh:46:  --baseline-method "${METHOD}" \
Point-Cache/scripts/E0_baseline/04_run_ulip_scanobjnnc_hardest_corruptions_all35_common.sh:47:  --baseline-method-full "${METHOD_FULL}" \
Point-Cache/scripts/E0_baseline/21_3_openshape_modelnet_clean_zs_global_local_single_gpu.sh:8:  "21_3_openshape_modelnet_clean_zs_global_local" \
Point-Cache/scripts/E0_baseline/21_3_openshape_modelnet_clean_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/21_3_openshape_modelnet_clean_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh:8:  "03_3_ulip_scanobjnn_clean_hardest_zs_global_local" \
Point-Cache/scripts/E0_baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/03_3_ulip_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh:8:  "34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global" \
Point-Cache/scripts/E0_baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh:8:  "22_3_openshape_modelnetc_corruptions_all35_zs_global_local" \
Point-Cache/scripts/E0_baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/22_3_openshape_modelnetc_corruptions_all35_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global_single_gpu.sh:8:  "03_2_ulip_scanobjnn_clean_hardest_zs_global" \
Point-Cache/scripts/E0_baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/03_2_ulip_scanobjnn_clean_hardest_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh:8:  "33_3_uni3d_scanobjnn_clean_hardest_zs_global_local" \
Point-Cache/scripts/E0_baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/33_3_uni3d_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh:8:  "24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local" \
Point-Cache/scripts/E0_baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/24_3_openshape_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global_single_gpu.sh:8:  "23_2_openshape_scanobjnn_clean_hardest_zs_global" \
Point-Cache/scripts/E0_baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/23_2_openshape_scanobjnn_clean_hardest_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh:8:  "23_3_openshape_scanobjnn_clean_hardest_zs_global_local" \
Point-Cache/scripts/E0_baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/23_3_openshape_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/32_run_uni3d_modelnetc_corruptions_all35_common.sh:85:  --baseline-method "${METHOD}" \
Point-Cache/scripts/E0_baseline/32_run_uni3d_modelnetc_corruptions_all35_common.sh:86:  --baseline-method-full "${METHOD_FULL}" \
Point-Cache/scripts/E0_baseline/22_run_openshape_modelnetc_corruptions_all35_common.sh:47:  --baseline-method "${METHOD}" \
Point-Cache/scripts/E0_baseline/22_run_openshape_modelnetc_corruptions_all35_common.sh:48:  --baseline-method-full "${METHOD_FULL}" \
Point-Cache/scripts/E0_baseline/01_3_ulip_modelnet_clean_zs_global_local_single_gpu.sh:7:EXP_ID="01_3_ulip_modelnet_clean_zs_global_local"
Point-Cache/scripts/E0_baseline/01_3_ulip_modelnet_clean_zs_global_local_single_gpu.sh:8:METHOD="zs_global_local"
Point-Cache/scripts/E0_baseline/01_3_ulip_modelnet_clean_zs_global_local_single_gpu.sh:10:RUNNER="runners/model_with_hierarchical_caches.py"
Point-Cache/scripts/E0_baseline/01_2_ulip_modelnet_clean_zs_global_single_gpu.sh:7:EXP_ID="01_2_ulip_modelnet_clean_zs_global"
Point-Cache/scripts/E0_baseline/01_2_ulip_modelnet_clean_zs_global_single_gpu.sh:8:METHOD="zs_global"
Point-Cache/scripts/E0_baseline/01_2_ulip_modelnet_clean_zs_global_single_gpu.sh:10:RUNNER="runners/model_with_global_cache.py"
Point-Cache/scripts/E0_baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh:8:  "13_3_ulip2_scanobjnn_clean_hardest_zs_global_local" \
Point-Cache/scripts/E0_baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/13_3_ulip2_scanobjnn_clean_hardest_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/34_run_uni3d_scanobjnnc_hardest_corruptions_all35_common.sh:85:  --baseline-method "${METHOD}" \
Point-Cache/scripts/E0_baseline/34_run_uni3d_scanobjnnc_hardest_corruptions_all35_common.sh:86:  --baseline-method-full "${METHOD_FULL}" \
Point-Cache/scripts/E0_baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global_single_gpu.sh:8:  "12_2_ulip2_modelnetc_corruptions_all35_zs_global" \
Point-Cache/scripts/E0_baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/12_2_ulip2_modelnetc_corruptions_all35_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global_single_gpu.sh:8:  "13_2_ulip2_scanobjnn_clean_hardest_zs_global" \
Point-Cache/scripts/E0_baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/13_2_ulip2_scanobjnn_clean_hardest_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh:8:  "34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local" \
Point-Cache/scripts/E0_baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/E0_baseline/11_2_ulip2_modelnet_clean_zs_global_single_gpu.sh:8:  "11_2_ulip2_modelnet_clean_zs_global" \
Point-Cache/scripts/E0_baseline/11_2_ulip2_modelnet_clean_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/11_2_ulip2_modelnet_clean_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global_single_gpu.sh:8:  "22_2_openshape_modelnetc_corruptions_all35_zs_global" \
Point-Cache/scripts/E0_baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global_single_gpu.sh:9:  "zs_global" \
Point-Cache/scripts/E0_baseline/22_2_openshape_modelnetc_corruptions_all35_zs_global_single_gpu.sh:11:  "runners/model_with_global_cache.py" \
Point-Cache/scripts/E0_baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh:8:  "04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local" \
Point-Cache/scripts/E0_baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh:9:  "zs_global_local" \
Point-Cache/scripts/E0_baseline/04_3_ulip_scanobjnnc_hardest_corruptions_all35_zs_global_local_single_gpu.sh:11:  "runners/model_with_hierarchical_caches.py" \
Point-Cache/scripts/eval_model_with_hierarchical_caches_ablate_seed.sh:27:    python runners/model_with_hierarchical_caches_ablate_seed.py \
Point-Cache/scripts/eval_model_with_hierarchical_caches_ablate_seed.sh:47:    python runners/model_with_hierarchical_caches_ablate_seed.py \
Point-Cache/scripts/eval_model_with_hierarchical_caches_ablate_seed.sh:63:    python runners/model_with_hierarchical_caches_ablate_seed.py \
Point-Cache/scripts/eval_model_with_hierarchical_caches_ent.sh:26:    python runners/model_with_hierarchical_caches_ent.py \
Point-Cache/scripts/eval_model_with_hierarchical_caches_ent.sh:44:    python runners/model_with_hierarchical_caches_ent.py \
Point-Cache/scripts/eval_model_with_hierarchical_caches_ent.sh:58:    python runners/model_with_hierarchical_caches_ent.py \
Point-Cache/scripts/E1_text_prototype_enhancement/01_run_ulip_modelnetc_s2_zs_fusion_weight_ablation_common.sh:62:  --baseline-method "zs" \
Point-Cache/scripts/E1_text_prototype_enhancement/01_run_ulip_modelnetc_s2_zs_fusion_weight_ablation_common.sh:63:  --baseline-method-full "Fusion weight ablation: manual_full=${STATIC_WEIGHT}, LLM=${DYNAMIC_WEIGHT}" \
Point-Cache/scripts/E1_text_prototype_enhancement/00_run_ulip_modelnetc_s2_zs_text_method_smoke_common.sh:43:  --baseline-method "zs" \
Point-Cache/scripts/E1_text_prototype_enhancement/00_run_ulip_modelnetc_s2_zs_text_method_smoke_common.sh:44:  --baseline-method-full "${METHOD_FULL}" \
Point-Cache/scripts/eval_model_with_global_cache.sh:26:    python runners/model_with_global_cache.py \
Point-Cache/scripts/eval_model_with_global_cache.sh:45:    python runners/model_with_global_cache.py \
Point-Cache/scripts/eval_model_with_global_cache.sh:60:    python runners/model_with_global_cache.py \
Point-Cache/scripts/eval_model_with_hierarchical_caches.sh:26:    python runners/model_with_hierarchical_caches.py \
Point-Cache/scripts/eval_model_with_hierarchical_caches.sh:45:    python runners/model_with_hierarchical_caches.py \
Point-Cache/scripts/eval_model_with_hierarchical_caches.sh:60:    python runners/model_with_hierarchical_caches.py \
Point-Cache/scripts/recur-pc/run_hierarchical_ulip2_modelnetc_add_global2.sh:27:CUDA_VISIBLE_DEVICES="${GPU_ID}" python runners/model_with_hierarchical_caches.py \
Point-Cache/scripts/recur-pc/run_e4_canc_v1_hierarchical_modelnetc_all_corruptions_dual_gpu.sh:36:RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches_e4_canc_v1.py"
Point-Cache/scripts/recur-pc/run_e4_canc_v0_hierarchical_modelnetc_all_corruptions_dual_gpu.sh:36:RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches_e4_canc_v0.py"
Point-Cache/scripts/recur-pc/run_global_ulip2_modelnetc_add_global2.sh:27:CUDA_VISIBLE_DEVICES="${GPU_ID}" python runners/model_with_global_cache.py \
Point-Cache/scripts/recur-pc/run_e4_canc_v1_hierarchical_modelnetc_all35_corruptions_dual_gpu.sh:36:RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches_e4_canc_v1.py"
Point-Cache/scripts/recur-pc/run_e4_canc_diag_hierarchical_modelnetc_all_corruptions_dual_gpu.sh:40:RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches_e4_canc_diag.py"
Point-Cache/scripts/recur-pc/run_e3_glc_v1_hierarchical_modelnetc_all_corruptions_dual_gpu.sh:39:RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches_e3_glc_v1.py"
Point-Cache/scripts/recur-pc/run_reliability_v1_hierarchical_ulip2_modelnetc_add_global2.sh:59:CUDA_VISIBLE_DEVICES="${GPU_ID}" python runners/model_with_hierarchical_caches.py \
Point-Cache/scripts/recur-pc/run_e3_glc_v0_hierarchical_modelnetc_all_corruptions_dual_gpu.sh:31:RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches_e3_glc_v0.py"
Point-Cache/scripts/recur-pc/run_reliability_v1_hierarchical_modelnetc_all_corruptions_gpu1.sh:23:RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches_reliability_v1.py"
Point-Cache/scripts/recur-pc/run_reliability_v1_hierarchical_modelnetc_all_corruptions_gpu1.sh:65:  echo "Point-Cache/runners/model_with_hierarchical_caches_reliability_v1.py"
Point-Cache/scripts/recur-pc/run_e1_base_hierarchical_modelnetc_all35_corruptions_dual_gpu.sh:45:RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches.py"
Point-Cache/scripts/recur-pc/run_e3_glc_v0_hierarchical_modelnetc_all_corruptions_gpu1.sh:24:RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches_e3_glc_v0.py"
Point-Cache/scripts/recur-pc/run_e4_canc_diag_v2_hierarchical_modelnetc_all_corruptions_dual_gpu.sh:40:RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches_e4_canc_diag_v2.py"
Point-Cache/scripts/recur-pc/run_baseline_hierarchical_modelnetc_all_corruptions_gpu0.sh:23:RUNNER="${POINT_CACHE_ROOT}/runners/model_with_hierarchical_caches.py"
Point-Cache/scripts/eval_model_with_hierarchical_caches_speed.sh:26:    python runners/model_with_hierarchical_caches_speed.py \
Point-Cache/scripts/eval_model_with_hierarchical_caches_speed.sh:44:    python runners/model_with_hierarchical_caches_speed.py \
Point-Cache/scripts/eval_model_with_hierarchical_caches_speed.sh:58:    python runners/model_with_hierarchical_caches_speed.py \
