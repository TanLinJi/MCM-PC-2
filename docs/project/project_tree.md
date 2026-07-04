# DPC-Point Project Tree

Current paper title:

**DPC-Point: Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation**

This file is a current navigation snapshot. The repository directory is still
named `MCM-PC-2` for continuity, but the active paper and documentation line is
DPC-Point.

## Root

```text
/root/autodl-tmp/MCM-PC-2/
├── ICCV25-MCP/              # external/reference code and mcp3d material
├── Point-Cache/             # active code, scripts, logs, results, datasets
├── docs/                    # human-readable project documentation
├── mcm_pc/                  # historical package placeholder
└── paper_notes/             # paper notes placeholder
```

## Active Code Areas

```text
Point-Cache/
├── configs/
├── data/
├── datasets/
├── llm/
├── models/
├── results/
├── runners/
│   ├── E4_distribution_guided_cache/
│   └── E5_adapt_inspired_gaussian_alignment_cache/
├── scripts/
│   ├── E4_distribution_guided_cache/
│   └── E5_adapt_inspired_gaussian_alignment_cache/
└── utils/
```

Current method work is concentrated in the E4/E5 runner and script folders.
Generated results and raw logs should remain under `Point-Cache/`.

## Current Docs

```text
docs/
├── README.md
├── proposals/
│   └── current_direction.md
├── paper/
│   ├── 0_outline.md
│   ├── abstract.md
│   ├── 1_introduction.md
│   ├── 2_related_work.md
│   └── 3_method.md
├── experiments/
│   ├── README.md
│   ├── experiment_registry.md
│   ├── E0_baseline/
│   ├── E1_text_prototype_enhancement/
│   ├── E2_text_prototype_transfer_to_pointcache/
│   ├── E3_global_prototype_alignment_cache/
│   ├── E4_distribution_guided_cache/
│   ├── E5_adapt_inspired_gaussian_alignment_cache/
│   ├── narrative/
│   └── pointcache_repro/
├── project/
│   ├── glossary.md
│   ├── progress_log.md
│   ├── project_tree.md
│   ├── user_preferences.md
│   └── conventions/
├── decisions/
├── references/
├── reports/
├── assets/figures/
└── archive/
```

## Archive Areas

```text
docs/archive/
├── README.md
├── legacy_mcmpc_canc/
├── mcp3d_framework_proposal.md
└── reference_code/

docs/experiments/archive/
└── legacy_pre_mcmpc_restart/
```

The archive contains historical MCM-PC / Multi-Cache Matrix / CANC / MCP-3D
material and early pre-restart experiments. It is retained for traceability, not
as the current paper plan.

## Canonical Entry Points

For active work, start from:

1. `docs/README.md`
2. `docs/proposals/current_direction.md`
3. `docs/experiments/experiment_registry.md`
4. `docs/experiments/narrative/e4_e5_research_narrative.md`
5. `docs/paper/0_outline.md`
