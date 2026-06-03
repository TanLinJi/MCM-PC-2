# MCM-PC Experiment Registry

Last updated: 2026-06-03

This document is the central registry for the formal MCM-PC experiment sequence.

It records:

- experiment ID;
- experiment name;
- experiment goal;
- current status;
- log and analysis document paths;
- result paths;
- related project documents;
- next actions.

This registry does not replace detailed experiment logs or analysis documents. Each formal experiment should still maintain its own `log.md` and `analysis.md`.

## 1. Formal Experiment Sequence

| ID | Experiment Name | Goal | Status |
|---|---|---|---|
| E0 | Point-Cache Baseline Reproduction and Analysis | Reproduce Point-Cache baselines and analyze baseline behavior | Reproduction completed; analysis in progress |
| E1 | Text Prototype Enhancement | Study point-cloud-aware templates and dynamically generated DeepSeek prompts | Planning |
| E2 | Reliability-Gated Local Cache | Dynamically control local cache contribution based on reliability | Planned |
| E3 | Conflict-Aware Negative Suppression | Use global-local conflict as conservative negative evidence | Planned |
| E4 | Reliability-Aware Multi-Cache Matrix | Build the full MCM-PC framework | Planned |
| E5 | Ablation Studies and Visualization | Prepare ablations, case studies, and paper figures | Planned |

## 2. E0: Point-Cache Baseline Reproduction and Analysis

### Goal

Reproduce the Point-Cache baseline results and use them as the foundation for all later MCM-PC experiments.

E0 includes both baseline reproduction and baseline result analysis.

### Scope

Backbones:

- ULIP
- ULIP-2
- OpenShape
- Uni3D

Datasets and settings:

- ModelNet clean
- ModelNet-C all35
- ScanObjNN clean hardest
- ScanObjNN-C hardest all35

Methods:

- zero-shot
- zero-shot + global cache
- zero-shot + global cache + local cache

### Current Status

Reproduction completed.

Baseline analysis is in progress.

Current high-level observations:

- Global cache is the main stable source of improvement.
- Local cache provides auxiliary gains, but its contribution is not always stable.
- Some backbone and corruption settings show weak or negative local-cache contribution.
- These observations motivate later reliability-aware cache design.

### Documents

Existing documents:

- docs/experiments/baseline/
- docs/experiments/baseline.zip
- docs/experiments/pointcache_repro/
- docs/experiments/repro_log.md
- docs/experiments/experiment_summary.md

Planned formal E0 documents:

- docs/experiments/E0_baseline/log.md
- docs/experiments/E0_baseline/analysis.md

### Result Paths

- Point-Cache/results/baseline/

### Next Action

Create formal E0 baseline analysis documents based on existing baseline result markdown files.

## 3. E1: Text Prototype Enhancement

### Goal

Study whether Point-Cache text prototypes can be improved by using point-cloud-aware manual templates and dynamically generated DeepSeek prompts.

### Motivation

Point-Cache dynamically adapts the visual side through global and local caches, but its text prototypes are mainly constructed from fixed manual templates.

The original manual prompt ensemble contains many 2D image-style templates, such as photo-style, blurry-photo-style, cropped-photo-style, and painting-style prompts. These prompts may not be optimal for 3D point cloud recognition.

E1 studies text-side enhancement before modifying the cache mechanism.

### Planned Prompt Sources

- manual_full: original Point-Cache manual prompt ensemble; E0-compatible baseline.
- manual_3d: point-cloud-aware subset filtered from manual_full.
- deepseek_static: pre-generated DeepSeek prompts saved as JSON.
- deepseek_dynamic_init: DeepSeek prompts generated at experiment initialization based only on candidate class names.
- manual3d_deepseek_dynamic_init: branch-level fusion of manual_3d and deepseek_dynamic_init.

### Dynamic Prompt Rule

E1 uses dynamic-init prompt generation.

Allowed:

- generate prompts before the test stream starts;
- use only dataset candidate class names;
- save generated prompts to the experiment result directory;
- freeze generated prompts during inference.

Not allowed in E1:

- do not call LLM for each test sample;
- do not use ground-truth labels;
- do not expose test point cloud content to the LLM;
- do not generate prompts based on individual sample predictions.

### Main Candidate

The main E1 method candidate is:

    manual3d_deepseek_dynamic_init

Preferred formulation:

    text_prototype =
        static_weight * mean(manual_3d embeddings)
        + dynamic_weight * mean(dynamic DeepSeek embeddings)

Default setting:

- static branch weight: 0.75
- dynamic branch weight: 0.25
- dynamic prompt count: 25 prompts per class

### Planned Experimental Stages

Stage 1: zero-shot prompt comparison.

Compare:

- manual_full
- manual_3d
- deepseek_dynamic_init
- manual3d_deepseek_dynamic_init

Stage 2: Point-Cache prompt comparison.

If Stage 1 shows a meaningful trend, extend to:

- zero-shot
- zero-shot + global cache
- zero-shot + global cache + local cache

### Priority Settings

Initial candidates:

- ULIP × ModelNet-C all35
- ULIP-2 × ModelNet-C all35
- Uni3D × ScanObjNN-C hardest all35

### Documents

Planned:

- docs/decisions/D002_prompt_source_policy.md
- docs/experiments/E1_tpe/log.md
- docs/experiments/E1_tpe/analysis.md

### Result Paths

Planned:

- Point-Cache/results/mcmpc/E1_tpe/

### Next Action

Write D002 prompt-source policy before modifying code.

## 4. E2: Reliability-Gated Local Cache

### Goal

Dynamically control local cache contribution according to reliability.

### Motivation

E0 shows that local cache is useful in some settings but weak or unstable in others. Therefore, local cache should not always be fused with a fixed weight.

### Initial Idea

Replace fixed local cache fusion with a sample-wise reliability weight:

    final = zs + alpha_g * global + r_l(x) * alpha_l * local

where `r_l(x)` is determined by reliability signals such as agreement, margin, entropy, and possibly prototype distance.

### Status

Planned.

### Next Action

Start after E1 text-side experiments are completed or clearly bounded.

## 5. E3: Conflict-Aware Negative Suppression

### Goal

Use global-local conflict as a signal of unreliable positive evidence and suppress suspicious classes conservatively.

### Motivation

Earlier exploratory experiments suggested that global-local conflict may indicate failure, but local top-1 should not be blindly trusted as a corrected pseudo-label.

### Initial Idea

Conflict should be used as negative evidence, not direct positive correction.

### Status

Planned.

## 6. E4: Reliability-Aware Multi-Cache Matrix

### Goal

Build the full MCM-PC framework by dynamically calibrating multiple evidence sources.

### Candidate Evidence Sources

- text prototype branch;
- zero-shot logits;
- global cache;
- local cache;
- conflict or negative suppression branch.

### Status

Planned.

## 7. E5: Ablation Studies and Visualization

### Goal

Prepare final ablations, case studies, and paper figures.

### Candidate Analyses

- prompt source ablation;
- static vs dynamic prompt ablation;
- local reliability ablation;
- conflict suppression ablation;
- full matrix fusion ablation;
- per-corruption analysis;
- cache reliability visualization;
- failure case visualization.

### Status

Planned.

## 8. Related Existing Documents

The following existing documents are project-level references and should not be overwritten by this registry.

Project rules and status:

- docs/project/user_preferences.md
- docs/project/progress_log.md
- docs/project/glossary.md
- docs/project/project_tree.md

Early proposals and ideas:

- docs/proposals/core_innovations.md
- docs/proposals/ideas_log.md
- docs/proposals/matrix_idea_v0.md
- docs/proposals/auxiliary_innovation_3.md

Old paper notes:

- docs/paper/0_outline.md
- docs/paper/abstract.md
- docs/paper/1_introduction.md
- docs/paper/2_related_work.md
- docs/paper/3_method.md

Reports:

- docs/reports/

Archived legacy experiments:

- docs/experiments/archive/legacy_pre_mcmpc_restart/

## 9. Paper Draft Connection

The complete ICASSP paper draft should be maintained under:

    paper/ICASSP/

Each experiment should gradually contribute to the paper draft:

- motivation;
- method design;
- experiment tables;
- ablations;
- visualizations;
- limitations;
- references.

The paper should not be written only after all experiments are finished.
