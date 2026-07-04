# Current Project Preferences

This file records the current DPC-Point working rules. Older MCM-PC / CANC
preferences are archived under:

```text
docs/archive/legacy_mcmpc_canc/project/
```

## Current Paper Direction

Use the title:

```text
DPC-Point: Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation
```

Avoid using MCM-PC, Multi-Cache Matrix, or CANC as the active paper title or
main contribution unless explicitly reviving that line.

## Experiment Policy

- The project setting is training-free Test-Time Adaptation (training-free TTA):
  test-time adaptation must not update point encoder, text encoder, text
  prototypes, classifier heads, or any other model parameters. Allowed online
  changes are limited to caches, cache statistics, distribution statistics,
  diagnostic counters, scoring weights, and replacement/update rules.
- Keep Point-Cache as the primary baseline.
- Treat `02_9_2` as the current DPC-Point anchor, not as final paper evidence.
- Do not claim paper-level completion before all35, clean, ScanObjNN-C, and at
  least one extra backbone are checked.
- Preserve raw logs, CSV, JSON, weights, and datasets in `Point-Cache/`.
- Keep human-readable summaries under `docs/experiments/`.
- Use one document per experiment version, for example separate documents for
  A0, A1, and A2 under an experiment-family folder.
- Before starting each new experiment version, create or update that experiment
  document with what the experiment will test, exact parameters, parameter
  meanings, the hypothesis/target, the script path, and the relative run
  command.
- After an experiment finishes, write both the result analysis and the next-step
  plan into the corresponding document. During discussion, critically evaluate
  both the user's proposed logic and Codex's proposed logic; do not assume either
  side's plan is correct without checking for failure modes.
- When analyzing alignment-cache sample correctness, use cumulative historical
  correctness for every sample that has ever entered or replaced the alignment
  cache, not only the samples currently remaining in the cache. Ground-truth
  labels may be used only for offline diagnostics and must not affect TTA
  updates or predictions.
- Diagnostic-only statistics and logging added for method verification must be
  clearly marked as diagnostic logic. After the full experiment line is validated,
  remove or disable these diagnostic-only paths from the final method code and
  keep them only in archived experiment/debug variants.

## Writing Policy

- Paper text should be written in English.
- Project management and analysis notes can remain Chinese.
- In Chinese explanations, introduce technical terms in Chinese first, followed
  by the English term in parentheses. For example, write "有资格样本
  (eligible sample)" instead of using "eligible" alone.
- When a method becomes current, update `docs/README.md`,
  `docs/experiments/experiment_registry.md`, and `docs/paper/0_outline.md`.
- Archive obsolete framing instead of deleting it.

## Naming Policy

- Use `DPC-Point` for the project/paper.
- Use E4 names for implementation lineage when needed, for example
  `E4-C-A0+E1-textdist-only`.
- In paper prose, translate E4 implementation names into method language:
  "distribution-guided prototype cache replacement".
