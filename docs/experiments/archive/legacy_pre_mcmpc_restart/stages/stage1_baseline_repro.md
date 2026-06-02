# Stage 1: Point-Cache Baseline Reproduction

## Goal

Reproduce the original Point-Cache baselines on the same minimal setting before modifying the code.

## Common Setting

| Item | Value |
|---|---|
| Backbone | ULIP-2 |
| Dataset | ModelNet-C |
| Corruption | add_global_2 |
| Points | 1024 |
| GPU | Tesla T4 |
| Branch | baseline-repro |

## Results

| Stage | Method | Accuracy | Delta vs Zero-shot |
|---|---|---:|---:|
| 1.1 | Zero-shot | 65.19 | - |
| 1.2 | Global Cache | 67.06 | +1.87 |
| 1.3 | Hierarchical Cache | 68.15 | +2.96 |

## Scripts

| Method | Script |
|---|---|
| Zero-shot | `Point-Cache/scripts/recur-pc/run_zs_ulip2_modelnetc_add_global2.sh` |
| Global Cache | `Point-Cache/scripts/recur-pc/run_global_ulip2_modelnetc_add_global2.sh` |
| Hierarchical Cache | `Point-Cache/scripts/recur-pc/run_hierarchical_ulip2_modelnetc_add_global2.sh` |

## Notes

- Fixed unconditional `wandb.log()` calls so experiments can run without WandB.
- Zero-shot baseline reproduced successfully.
- Global Cache baseline reproduced successfully.
- Hierarchical Cache baseline reproduced successfully.
- Verified trend: Hierarchical Cache > Global Cache > Zero-shot.
