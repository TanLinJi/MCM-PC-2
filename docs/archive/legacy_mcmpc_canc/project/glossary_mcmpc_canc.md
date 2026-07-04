# DPC-Point Glossary

## Current Project Name

**DPC-Point** = **Distribution-Guided Prototype Cache for Robust Point Cloud
Test-Time Adaptation**

This is the current paper name and project framing.

## Core Terms

| Term | Meaning |
|---|---|
| Point-Cache | CVPR 2025 baseline that performs training-free point cloud test-time adaptation with global/local caches |
| DPC | Distribution-Guided Prototype Cache |
| GPA Cache | Global prototype-alignment cache branch used in the E3/E4 implementation line |
| Accepted-history visual distribution | Class distribution estimated from samples accepted by trusted positive-cache updates |
| Text distribution | Prompt-level semantic distribution built from manual prompts and optional frozen LLM descriptions |
| Textdist-only | E1 LLM descriptions are used only for text distribution scoring, not for the final classifier |
| Cache purification | Replacing or admitting cache entries using distribution consistency, not confidence alone |
| Prototype pollution | Wrong or unrepresentative samples entering the cache and misleading later predictions |
| Running z-score | Online normalization used to align visual and text score scales before fusion |

## Current Main Experiment

```text
02_9_2
E4-C-A0+E1-textdist-only
E4_TEXT_SCORE_WEIGHT=0.15
```

This is the current DPC-Point anchor setting.

## Historical Terms

The following terms are historical and should not be used as the current paper
scope:

| Historical Term | Current Status |
|---|---|
| MCM-PC | Archived broad framing |
| Multi-Cache Matrix | Archived long-term idea, not current ICASSP paper scope |
| CANC / Conflict-Aware Negative Cache | Archived earlier experiment line |
| MCP-3D | Archived earlier proposal title |

Historical files are preserved in `docs/archive/legacy_mcmpc_canc/`.
