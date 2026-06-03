# Experiments

This directory records the formal experiment sequence and experiment documentation for the MCM-PC project.

Current paper direction:

**MCM-PC: Reliability-Aware Multi-Cache Matrix for Test-Time Adaptation of 3D Point Cloud Vision-Language Models**

## 1. Formal Experiment Sequence

After the experiment numbering reset, the formal experiment sequence is defined as follows.

| ID | Name | Purpose | Status |
|---|---|---|---|
| E0 | Point-Cache Baseline Reproduction and Analysis | Reproduce Point-Cache baselines and analyze baseline behavior | In progress |
| E1 | Text Prototype Enhancement | Study point-cloud-aware templates and dynamically generated LLM prompts | Planned |
| E2 | Reliability-Gated Local Cache | Dynamically control local cache contribution based on reliability | Planned |
| E3 | Conflict-Aware Negative Suppression | Use unreliable global-local conflict as negative evidence | Planned |
| E4 | Reliability-Aware Multi-Cache Matrix | Build the full MCM-PC framework | Planned |
| E5 | Ablation Studies and Visualization | Conduct ablations, case studies, and paper figures | Planned |

## 2. E0: Baseline Reproduction and Analysis

E0 includes the completed Point-Cache baseline reproduction and the corresponding result analysis.

Current E0 assets include:

- `E0_baseline/`: Point-Cache baseline 复现结果与分析文档。
- `pointcache_repro/`: reproduction notes and commands.
- `repro_log.md`: reproduction log.

E0 is not a new MCM-PC method experiment. It is the foundation for all later comparisons.

## 3. Experiment Documentation Rule

Each formal experiment after E0 should have its own directory:

    docs/experiments/E*_name/

Each experiment directory should include at least:

    log.md
    analysis.md

The two documents have different responsibilities:

- `log.md` records commands, scripts, configurations, checkpoints, datasets, backbones, prompt sources, runtime notes, errors, fixes, and git commits.
- `analysis.md` summarizes quantitative results, compares them with E0, identifies gains or failures, and explains whether the experiment supports the MCM-PC hypothesis.

The complete ICASSP paper draft is maintained separately under:

    paper/ICASSP/

The paper draft should be updated alongside experiments, rather than being written only after all experiments are finished.

## 4. Archived Legacy Experiments

Early exploratory experiments before the formal MCM-PC restart have been archived under:

    docs/experiments/archive/legacy_pre_mcmpc_restart/

These legacy experiments do not occupy or affect the new E0-E5 numbering.

Archived legacy items include:

- early text prototype enhancement attempts;
- entropy/margin reliability experiments;
- global-local consistency experiments;
- conservative negative cache attempts;
- old staged experiment notes.

## 5. Current Next Step

The next formal work is:

    E1: Text Prototype Enhancement

Before modifying code, the E1 plan and prompt-source policy should be documented.
