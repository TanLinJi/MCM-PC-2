# DPC-Point Workflow Conventions

Project root:

```text
/root/autodl-tmp/MCM-PC-2
```

Core codebase:

```text
/root/autodl-tmp/MCM-PC-2/Point-Cache
```

## Current Canonical Files

| Purpose | File |
|---|---|
| Project docs index | `docs/README.md` |
| Current method direction | `docs/proposals/current_direction.md` |
| Experiment registry | `docs/experiments/experiment_registry.md` |
| Research narrative | `docs/experiments/narrative/e4_e5_research_narrative.md` |
| Paper outline | `docs/paper/0_outline.md` |
| Glossary | `docs/project/glossary.md` |

## Workflow

1. Explain the hypothesis before changing code or launching a long run.
2. Run quick checks before full benchmark runs when feasible.
3. Save experiment logs and summaries under `Point-Cache/results/...`.
4. Summarize completed experiments in `docs/experiments/...`.
5. Update the narrative when a result changes the research direction.
6. Archive obsolete plans under `docs/archive/`, not by deleting them.

## Training-Free TTA Boundary

DPC-Point is a training-free Test-Time Adaptation (training-free TTA) project.
During test-time adaptation, do not update point encoder parameters, text encoder
parameters, text prototypes, classifier heads, or any other model parameters.
Online adaptation may update only caches, cache statistics, distribution
statistics, diagnostic counters, scoring weights, and cache replacement/update
rules.

## Experiment Documentation Rule

Use one document per experiment version. For example, an experiment family can
have a folder containing separate files for A0, A1, A2, and later variants.

Before creating scripts or launching a new experiment version, create or update
the corresponding document with:

- what the experiment is prepared to do;
- the exact parameter values;
- the meaning of each parameter;
- the hypothesis or mechanism being tested;
- the planned comparison baseline;
- the script path and relative run command.

After results are available, write both the result analysis and the next-step
plan into the corresponding document. During discussion, critically evaluate
both the user's proposed logic and Codex's proposed logic; do not assume either
side's plan is correct without checking for failure modes.

For alignment-cache correctness diagnostics, "alignment sample correctness"
means the cumulative historical zero-shot pseudo-label correctness of every
sample that has ever entered or replaced the alignment cache. It is not the
correctness of only the final cache snapshot. Ground-truth labels can be used
for this offline diagnostic only and must not enter Test-Time Adaptation (TTA)
decisions.

Diagnostic-only statistics and logging added for internal validation must be
clearly marked as diagnostic logic. Once the full experiment line is validated,
remove or disable these diagnostic-only paths from final method code and keep
them only in archived experiment/debug variants.

## Current Method Anchor

```text
02_9_2
E4-C-A0+E1-textdist-only
E4_TEXT_SCORE_WEIGHT=0.15
```

This setting is the current DPC-Point anchor but still needs all35,
clean/corruption tradeoff analysis, ScanObjNN-C, and extra-backbone validation.

## Archive Boundary

MCM-PC, Multi-Cache Matrix, CANC, and MCP-3D are historical framings. Their
documents live under `docs/archive/legacy_mcmpc_canc/` and should not be used as
current instructions unless explicitly reactivated.
