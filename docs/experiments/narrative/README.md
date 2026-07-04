# Experiment Narrative

This directory keeps long-form research narratives for the current DPC-Point
project. These files explain why each experiment was run, what changed the
research direction, which results are reliable, and what remains incomplete.

## Current Files

| File | Purpose |
|---|---|
| `e0_e3_research_narrative.md` | Historical E0-E3 route from baseline reproduction through text/prototype transfer and early alignment attempts |
| `e4_e5_research_narrative.md` | Current E4 distribution-guided cache and E5 ADAPT/PGA-inspired branch narrative |

## Maintenance Rules

- Update these narratives after any result or bug fix that changes the research judgment.
- Keep raw logs, CSV/JSON outputs, checkpoints, and datasets under `Point-Cache/`.
- Each narrative should state background, setup, paths, results, comparison, failure reason, evidence boundary, and next action.
- If a narrative grows too long, split it by experiment phase and update this README.
