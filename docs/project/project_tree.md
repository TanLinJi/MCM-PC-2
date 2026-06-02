# MCM-PC-2 项目目录树

> 说明：这是“总览 + 叶子样例”版。大目录只列代表项；同类文件很多时保留约 3 个样例并注明省略。

## 根目录

```text
MCM-PC-2/
├── .gitignore
├── ICCV25-MCP/
├── Point-Cache/
├── docs/
├── mcm_pc/
└── paper_notes/
```

## ICCV25-MCP

```text
ICCV25-MCP/
├── LICENSE
├── README.md
├── clip/
├── configs/
├── datasets/
├── docs/
├── gpt3_prompts/
├── mcp_runner.py
├── requirements.txt
├── scripts/
├── util/
└── utils.py
```

### ICCV25-MCP/clip

```text
clip/
├── __init__.py
├── bpe_simple_vocab_16e6.txt.gz
├── clip.py
├── model.py
└── simple_tokenizer.py
```

### ICCV25-MCP/configs

```text
configs/
├── aircraft.yaml
├── imagenet.yaml
├── oxford_pets.yaml
└── ...（其余 12 个 yaml 省略）
```

### ICCV25-MCP/datasets

```text
datasets/
├── aircraft.py
├── imagenet.py
├── imagenetv2.py
└── ...（其余数据集模块省略）
```

### ICCV25-MCP/gpt3_prompts

```text
gpt3_prompts/
├── CuPL_prompts_caltech101.json
├── CuPL_prompts_imagenet.json
├── CuPL_prompts_ucf101.json
└── ...（其余 prompt 省略）
```

### ICCV25-MCP/util

```text
util/
├── __init__.py
└── tools.py
```

### ICCV25-MCP/scripts

```text
scripts/
├── run_cd_benchmark_rn50.sh
├── run_cd_benchmark_vit.sh
├── run_ood_benchmark_rn50.sh
└── run_ood_benchmark_vit.sh
```

### ICCV25-MCP/docs

```text
docs/
└── mcp.png
```

## Point-Cache

```text
Point-Cache/
├── assets/
├── clip/
├── configs/
├── data/
├── datasets/
├── env.yaml
├── llm/
├── logs/
├── models/
├── notebook/
├── results/
├── runners/
├── scripts/
├── utils/
├── wandb/
└── weights/
```

### Point-Cache/assets

```text
assets/
├── architecture.png
└── motivation.png
```

### Point-Cache/clip

```text
clip/
├── __init__.py
├── bpe_simple_vocab_16e6.txt.gz
├── clip.py
├── model.py
└── simple_tokenizer.py
```

### Point-Cache/configs

```text
configs/
├── modelnet40.yaml
├── scanobjnn.yaml
├── sonn_c.yaml
└── ...（其余 13 个 yaml 省略）
```

### Point-Cache/data

```text
data/
├── modelnet40/
│   ├── classnames.txt
│   ├── test_pc.npy
│   └── test_split.json
├── modelnet40_c/
├── modelnet_c/
├── objaverse_lvis/
├── omniobject3d/
├── scanobjnn/
├── shapenet_c/
└── sonn_c/
```

### Point-Cache/datasets

```text
datasets/
├── augmix_ops.py
├── modelnet40.py
├── scanobjnn.py
└── ...（其余 15 个模块省略）
```

### Point-Cache/llm

```text
llm/
├── da_mn10_gpt35_prompts.json
├── llm_generate_prompts.py
├── mn40_pointllm_prompts.json
└── ...（其余 prompt 省略）
```

### Point-Cache/logs

```text
logs/
├── debug-internal.log
├── debug.log
├── latest-run
└── offline-run-20260514_163522-uju3meb1/
```

### Point-Cache/models

```text
models/
├── openshape/
├── ulip/
└── uni3d/
```

#### Point-Cache/models/openshape

```text
openshape/
├── __init__.py
├── config.yaml
├── pointnet_util.py
├── ppta.py
└── __pycache__/
```

#### Point-Cache/models/ulip

```text
ulip/
├── __init__.py
├── pointbert/
│   ├── PointTransformer_8192point.yaml
│   ├── checkpoint.py
│   ├── dvae.py
│   ├── logger.py
│   ├── misc.py
│   └── point_encoder.py
├── text_encoder.py
├── ulip_model.py
└── __pycache__/
```

#### Point-Cache/models/uni3d

```text
uni3d/
├── __init__.py
├── __pycache__/
└── point_encoder.py
```

### Point-Cache/notebook

```text
notebook/
├── check_cache_threshold.ipynb
├── clip/
├── generate_pc_cls_images.py
├── gifs/
├── images/
├── try_diffusion_models.ipynb
└── visualize.ipynb
```

#### Point-Cache/notebook/clip

```text
clip/
├── .DS_Store
├── __init__.py
├── bpe_simple_vocab_16e6.txt.gz
├── clip.py
├── model.py
└── simple_tokenizer.py
```

#### Point-Cache/notebook/gifs

```text
gifs/
├── os_guitar.gif
├── ulip2_table.gif
├── uni3d_calculator.gif
└── ...（其余 gif 省略）
```

#### Point-Cache/notebook/images

```text
images/
├── ablate_alpha_in_cache.pdf
├── acc_on_clean_and_corrupted_mn40.pdf
├── omni3d_calculator_4096.pdf
└── ...（其余可视化文件省略）
```

### Point-Cache/results

```text
results/
└── baseline/
```

#### Point-Cache/results/baseline

```text
baseline/
├── 01_1_ulip_modelnet_clean_zs/
├── 11_1_ulip2_modelnet_clean_zs/
├── 21_1_openshape_modelnet_clean_zs/
├── 31_1_uni3d_modelnet_clean_zs/
└── ...（其余结果目录省略）
```

### Point-Cache/runners

```text
runners/
├── baseline/
└── recur-pc/
```

#### Point-Cache/runners/baseline

```text
baseline/
├── run_openshape_modelnetc_corruptions_all35.py
├── run_ulip_modelnetc_corruptions_all35.py
├── run_ulip2_modelnetc_corruptions_all35.py
├── run_uni3d_modelnetc_corruptions_all35.py
└── ...（其余运行入口省略）
```

#### Point-Cache/runners/recur-pc

```text
recur-pc/
├── run_baseline_hierarchical_modelnetc_all_corruptions_gpu0.sh
├── run_e0_tpe_v1_lite_zs_modelnetc_severity2_corruptions_dual_gpu.sh
├── run_e3_glc_v1_hierarchical_modelnetc_all_corruptions_dual_gpu.sh
└── ...（其余 17 个脚本省略）
```

### Point-Cache/scripts

```text
scripts/
├── baseline/
├── data_download_scripts/
├── eval_model_with_global_cache.sh
├── eval_model_with_hierarchical_caches.sh
├── eval_zs_infer.sh
├── param_count.sh
├── record_adaptation_acc.sh
├── record_adaptation_logits.sh
└── recur-pc/
```

#### Point-Cache/scripts/baseline

```text
baseline/
├── 00_check_baseline_data_paths.sh
├── 01_1_ulip_modelnet_clean_zs_single_gpu.sh
├── 21_1_openshape_modelnet_clean_zs_single_gpu.sh
├── 31_1_uni3d_modelnet_clean_zs_single_gpu.sh
└── ...（其余 60+ 个同类脚本省略）
```

#### Point-Cache/scripts/data_download_scripts

```text
data_download_scripts/
├── download_mc.py
├── download_modelnet40.py
├── download_omniobject3d.py
├── download_uni3d_checkpoints.sh
└── ...（其余下载脚本省略）
```

#### Point-Cache/scripts/recur-pc

```text
recur-pc/
├── run_baseline_hierarchical_modelnetc_all_corruptions_gpu0.sh
├── run_e0_tpe_v1_lite_zs_modelnetc_severity2_corruptions_dual_gpu.sh
├── run_e4_canc_v1_hierarchical_modelnetc_all_corruptions_dual_gpu.sh
└── ...（其余脚本省略）
```

### Point-Cache/utils

```text
utils/
├── __init__.py
├── check_img_text_acc.py
├── compute_mean_and_std.py
├── debug.py
├── find_class_pc.py
├── generate_pc_view_labels.py
├── mv_utils_zs.py
├── utils.py
└── visualize.py
```

### Point-Cache/wandb

```text
wandb/
├── debug-internal.log
├── debug.log
├── latest-run
└── offline-run-20260514_163522-uju3meb1/
```

### Point-Cache/weights

```text
weights/
├── download_openshape_weights.py
├── download_uni3d_weights.py
├── openshape/
├── ulip/
├── ulip2/
└── uni3d/
```

#### Point-Cache/weights/openshape

```text
openshape/
├── open_clip_pytorch_model/
│   └── vit-bigG-14/
│       ├── laion2b_s39b_b160k.bin
│       └── open_clip_pytorch_model.bin
└── openshape-pointbert-vitg14-rgb/
    └── model.pt
```

#### Point-Cache/weights/ulip

```text
ulip/
├── image-text-encoder/
│   └── slip_base_100ep.pt
├── point-encoder/
│   └── pointbert_ulip1.pt
├── pointbert_ulip1.pt
├── pointbert_ulip2.pt
└── slip_base_100ep.pt
```

#### Point-Cache/weights/ulip2

```text
ulip2/
├── image-text-encoder/
│   └── slip_base_100ep.pt
├── point-encoder/
│   └── pointbert_ulip2.pt
└── pointbert_ulip2.pt
```

#### Point-Cache/weights/uni3d

```text
uni3d/
├── _hf_download/
│   ├── .cache/
│   └── modelzoo/
└── ...（缓存层其余内容省略）
```

## docs

```text
docs/
├── README.md
├── archive/
├── assets/
├── context/
├── experiments/
├── paper/
├── project/
├── proposals/
└── reports/
```

### docs/archive

```text
archive/
├── README.md
├── mcp3d_framework_proposal.md
└── reference_code/
    └── model_with_hierarchical_caches.py
```

### docs/assets/figures

```text
figures/
├── confusion_aware_negative_cache.png
└── e2_emr_delta_chart.png
```

### docs/context/windsurf

```text
windsurf/
├── conversation_archive.html
└── conversation_archive.md
```

### docs/experiments

```text
experiments/
├── README.md
├── baseline/
├── baseline.zip
├── e0_tpe/
├── e2_emr/
├── e3_glc/
├── e4_canc/
├── experiment_summary.md
├── pointcache_repro/
├── repro_log.md
└── stages/
```

#### docs/experiments/baseline

```text
baseline/
├── 01_1_ulip_modelnet_clean_zs.md
├── 01_2_ulip_modelnet_clean_zs_global.md
├── 11_1_ulip2_modelnet_clean_zs.md
├── 21_1_openshape_modelnet_clean_zs.md
├── 31_1_uni3d_modelnet_clean_zs.md
└── ...（其余 30+ 个同类文件省略）
```

#### docs/experiments/e0_tpe

```text
e0_tpe/
├── spherical_text_anchor_report.html
└── text_prototype_enhancement.md
```

#### docs/experiments/e2_emr

```text
e2_emr/
├── e2_emr_admission.html
└── e2_emr_admission.md
```

#### docs/experiments/e3_glc

```text
e3_glc/
├── e3_glc_consistency.md
└── e3_glc_v1_math_derivation.html
```

#### docs/experiments/e4_canc

```text
e4_canc/
├── e4_canc_diag_math_derivation.html
├── e4_canc_overview.md
├── e4_canc_v0_conservative.md
└── e4_canc_v0_rule_explanation.html
```

#### docs/experiments/pointcache_repro

```text
pointcache_repro/
├── commands.md
├── project_structure.md
└── reproduction_notes.md
```

#### docs/experiments/stages

```text
stages/
├── stage0_tpe_spherical_text_anchor_corrected.md
├── stage0_tpe_zero_shot_text_methods_summary.md
├── stage1_baseline_repro.md
├── stage2_emr.md
├── stage3_glc.md
└── stage4_canc_diag.md
```

### docs/paper

```text
paper/
├── 0_outline.md
├── abstract.md
├── 1_introduction.md
├── 2_related_work.md
└── 3_method.md
```

### docs/project

```text
project/
├── glossary.md
├── progress_log.md
├── project_tree.md
└── user_preferences.md
```

### docs/proposals

```text
proposals/
├── auxiliary_innovation_3.md
├── core_innovations.md
├── ideas_log.md
├── matrix_idea_v0.md
├── project_roadmap_v1.html
└── top_conference_proposal_v1.html
```

### docs/reports

```text
reports/
├── 2026-05-17_global_roadmap_v2.html
├── 2026-05-17_task_specification.html
└── mcm_pc_top_conference_progress_20260515.html
```

## paper_notes

```text
paper_notes/
├── idea_bank.md
├── mcm_pc_method_ideas.md
├── method_notes.md
└── proposal_notes.md
```

## mcm_pc

```text
mcm_pc/
└── __init__.py
```

## Changelog

- **2026-06-02** v1.1：新增 `project/project_tree.md`，整理当前仓库的目录树。
- **2026-05-17** v1.0：初始结构落盘。归档了 `docs/new/` 中所有用户上传的资料，按 8 大类目录重新分类，统一英文命名。同时合并 E2-EMR 三份重复、两份代号说明，删除日志副本（原版在 `Point-Cache/logs/recur-pc/`）。
