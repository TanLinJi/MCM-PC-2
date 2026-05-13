# Stage 0: Repository Initialization

Date: 2026-05-13

## Completed
- Created GitHub repository: TanLinJi/MCM-PC-2.
- Initialized local Git repository under `~/autodl-tmp/MCM-PC-2`.
- Set `main` as the clean initial workspace branch.
- Removed large files, datasets, `.pt` files, and Python cache files from Git tracking.
- Confirmed that `Point-Cache/data/` is ignored and not tracked.
- Created and pushed the `baseline-repro` branch.
- Added research logging directories:
  - `docs/`
  - `paper_notes/`
  - `mcm_pc/`

## Current branch
`baseline-repro`

## Next goal
Reproduce the original Point-Cache zero-shot, global cache, and hierarchical cache baselines.
## Stage 1.1: Zero-shot Baseline Reproduction

Date: 2026-05-13

### Setting
- Method: Zero-shot inference
- Backbone: ULIP-2
- Dataset: ModelNet-C
- Corruption: add_global_2
- Number of points: 1024
- GPU: Tesla T4
- Script: `Point-Cache/scripts/recur-pc/run_zs_ulip2_modelnetc_add_global2.sh`

### Result
- Final zero-shot accuracy: 65.19

### Notes
- Fixed an unconditional `wandb.log()` call in `Point-Cache/runners/zs_infer.py`.
- The experiment runs without enabling WandB logging.
