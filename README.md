# MCM-PC-2

This repository is the working project space for **DPC-Point: Distribution-Guided Prototype Cache for Robust Point Cloud Test-Time Adaptation**.

The project builds on the Point-Cache codebase and keeps experiment logs, paper notes, and DPC-Point documentation in one workspace. The active method line is under `Point-Cache/runners/E4_distribution_guided_cache/` and `Point-Cache/scripts/E4_distribution_guided_cache/`.

## Project Layout

| Path | Role |
|---|---|
| `Point-Cache/` | Main codebase, datasets, checkpoints, runners, scripts, and result folders |
| `docs/` | DPC-Point experiment notes, paper notes, result records, and project documentation |
| `docs/experiments/E4_distribution_guided_cache/` | Current E4/DPC-Point experiment records |
| `ICCV25-MCP/` | Related paper/project material |
| `paper_notes/` | Working notes |

For the documentation entrypoint, see:

```text
docs/README.md
```

## Current Experiment Line

The current DPC-Point experiment line keeps the Point-Cache online inference protocol and adds:

- decoupled cache capacities;
- text-visual distribution-guided prototype alignment;
- explicit final-score fusion:

```text
y = y_zs + alpha_g * y_g + alpha_l * y_l - alpha_n * y_n
```

The exact cache-capacity combinations and final-score weights are intentionally kept in script-level arrays, not in this top-level README:

```text
COMBINATIONS
FINAL_SCORE_WEIGHTS
```

Edit those arrays in the corresponding script when running new ablations.

## Main Commands

Run commands from:

```bash
cd /root/autodl-tmp/MCM-PC-2/Point-Cache
```

ULIP on ModelNet-C all35:

```bash
bash scripts/E4_distribution_guided_cache/09_2_ulip_modelnetc_all35_explicit_final_score_ablation.sh 0
```

ULIP on ScanObjNN-C hardest all35:

```bash
bash scripts/E4_distribution_guided_cache/09_2_ulip_scanobjnnc_hardest_all35_explicit_final_score_best.sh 0
```

Backbone transfer on ScanObjNN-C hardest all35:

```bash
bash scripts/E4_distribution_guided_cache/09_2_ulip2_scanobjnnc_hardest_all35_explicit_final_score_best.sh 0
bash scripts/E4_distribution_guided_cache/09_2_openshape_scanobjnnc_hardest_all35_explicit_final_score_best.sh 0
bash scripts/E4_distribution_guided_cache/09_2_uni3d_scanobjnnc_hardest_all35_explicit_final_score_best.sh 0
```

The last argument is the physical GPU id.

## Selected Results

Selected ULIP results recorded so far:

| Dataset / protocol | ULIP | PointCache | DPC-Point |
|---|---:|---:|---:|
| ModelNet-C all35 | - | - | 53.71 |
| ScanObjNN-C hardest, severity 2 | 23.97 | 28.42 | 30.53 |
| ScanObjNN-C hardest, all35 | 23.66 | 27.51 | 29.67 |

Notes:

- The ScanObjNN-C severity-2 row is aligned with Table 7 in the Point-Cache paper.
- The original Point-Cache paper reports ScanObjNN-C hardest at severity 2, not all35.
- The ScanObjNN-C all35 row uses the local Point-Cache all35 reproduction, with severity-2 cells calibrated to the paper values for ULIP and PointCache.

Detailed records:

```text
docs/experiments/E4_distribution_guided_cache/09_2_current_best_all35_record.md
docs/experiments/E4_distribution_guided_cache/09_2_explicit_final_score_weight_ablation.md
docs/experiments/E4_distribution_guided_cache/09_2_scanobjnnc_best_config_eval.md
docs/experiments/E4_distribution_guided_cache/09_2_scanobjnnc_vs_pointcache_paper_all35_table.md
```

## Result Locations

Experiment outputs are saved under:

```text
Point-Cache/results/E4_distribution_guided_cache/
```

Important files inside each result directory:

```text
summary.csv
final_score_weight_summary.csv
logs/
gpa_stats/
```

## Development Notes

- Keep raw results and logs under `Point-Cache/results/`.
- Keep experiment analysis and paper-facing summaries under `docs/experiments/`.
- Prefer adding new experiment scripts over modifying old scripts when changing protocols.
- Do not regenerate LLM prompts during benchmark runs; reuse the shared prompt cache under `Point-Cache/results/E1_text_prototype_enhancement/shared_prompts/`.

## Git Helper

Before committing, inspect the exact staged files:

```bash
cd /root/autodl-tmp/MCM-PC-2
git status --short
git diff --cached --stat
```

