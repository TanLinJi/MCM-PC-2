# D003: Rename Current Paper Direction to DPC-Point

Date: 2026-06-10

## Decision

The current paper direction is renamed to:

```text
DPC-Point: Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation
```

## Rationale

The current strongest evidence supports distribution-guided prototype cache
replacement, not the broader Multi-Cache Matrix framework. The DPC-Point title
matches the active E4-C-A0+E1-textdist-only method line:

- accepted-history visual distribution;
- prompt-level text distribution;
- distribution-guided GPA/prototype cache replacement;
- unchanged base final classifier and Point-Cache logit formula.

## Consequence

MCM-PC, Multi-Cache Matrix, CANC, and MCP-3D are archived as historical research
directions under:

```text
docs/archive/legacy_mcmpc_canc/
```

Future paper drafts, current experiment registry entries, and project summaries
should use DPC-Point terminology.
