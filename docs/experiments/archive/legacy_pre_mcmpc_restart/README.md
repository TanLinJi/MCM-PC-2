# Legacy Experiments Before The Current DPC-Point Line

This directory archives early exploratory experiments conducted before the
current DPC-Point experiment sequence was consolidated.

Current paper title:

**DPC-Point: Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation**

## Archived Material

| Directory | Historical Contents |
|---|---|
| `e0_tpe/` | Early text prototype enhancement attempts |
| `e2_emr/` | Entropy/margin reliability experiments |
| `e3_glc/` | Global-local consistency experiments |
| `e4_canc/` | Conservative/conflict-aware negative cache attempts |
| `stages/` | Old stage-style experiment notes |

These files were useful for understanding failure modes, but they do not define
the current DPC-Point experiment registry.

## Current Canonical Sequence

Use `docs/experiments/experiment_registry.md` for the active sequence:

- E0: Point-Cache baseline reproduction.
- E1: Text prototype enhancement as supporting text-distribution evidence.
- E2: Text prototype transfer to Point-Cache.
- E3: Global prototype alignment cache exploration.
- E4: Distribution-guided cache, the current DPC-Point main line.
- E5: ADAPT/PGA-inspired Gaussian alignment cache, exploratory.

## Archive Rule

Do not edit archived experiment notes to match current terminology. If an old
result becomes relevant again, summarize it in a current experiment document and
cite this archive as historical context.
