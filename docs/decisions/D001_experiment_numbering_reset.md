# D001: Experiment Numbering Reset and Documentation Rules

## Date

2026-06-02

## Background

The project previously contained several exploratory experiments, including early text prompt enhancement, entropy/margin reliability, global-local conflict analysis, and negative cache attempts. These experiments were useful for understanding failure modes, but they are no longer part of the formal MCM-PC experiment sequence.

The current project is now reorganized around the paper direction:

**MCM-PC: Reliability-Aware Multi-Cache Matrix for Test-Time Adaptation of 3D Point Cloud Vision-Language Models**

To avoid confusion between historical exploratory experiments and the new formal experiment sequence, the experiment numbering is reset.

## Decision

The formal experiment sequence is defined as follows:

- **E0**: Point-Cache baseline reproduction and baseline result analysis.
- **E1**: Text Prototype Enhancement with point-cloud-aware and dynamically generated prompts.
- **E2**: Reliability-Gated Local Cache.
- **E3**: Conflict-Aware Negative Suppression.
- **E4**: Reliability-Aware Multi-Cache Matrix.
- **E5**: Ablation studies and visualization.

Historical exploratory experiments are archived and do not occupy the new experiment numbering.

## Experiment Numbering Rules

1. E0 refers only to Point-Cache baseline reproduction and baseline result analysis.
2. New MCM-PC method experiments start from E1.
3. Archived legacy experiments must not affect the new experiment numbering.
4. New experiment directories, scripts, and documents should follow the new numbering scheme.
5. Baseline results must remain reproducible and should not be overwritten by E1 or later experiments.

## Documentation Rules

Each formal experiment must include two synchronized documents:

1. **Experiment Log**  
   Records the exact commands, scripts, configuration, dataset, backbone, checkpoint, prompt source, runtime notes, errors, fixes, and git commit information.

2. **Experiment Analysis**  
   Summarizes the quantitative results, compares them with E0 baseline, identifies gains or failures, and explains whether the experiment supports the MCM-PC hypothesis.

In addition, the complete paper manuscript should be maintained and updated throughout the project.

## Paper Draft Rule

The complete ICASSP paper draft should be maintained separately under:

`paper/ICASSP/`

The paper draft should be updated alongside experiments, including motivation, related work, method, experiments, ablations, figures, tables, limitations, and references. Experimental findings should be gradually converted into paper-ready writing rather than remaining only as notes.

## Recommended Document Layout

For each experiment, use the following structure:

`docs/experiments/E*_name/`

with at least:

- `log.md`
- `analysis.md`

For E0, use:

`docs/experiments/E0_baseline/`

For archived legacy experiments, use:

`docs/experiments/archive/legacy_pre_mcmpc_restart/`

## Immediate Next Step

After this decision document is committed, legacy experiment documents should be moved into:

`docs/experiments/archive/legacy_pre_mcmpc_restart/`

The ICASSP paper workspace should then be initialized under:

`paper/ICASSP/`
