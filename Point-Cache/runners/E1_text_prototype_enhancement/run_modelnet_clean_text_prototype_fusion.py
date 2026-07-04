#!/usr/bin/env python
"""Official E1 entry point for ULIP x clean ModelNet text prototype fusion."""

import argparse
import csv
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None

POINT_CACHE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POINT_CACHE_ROOT))

from utils.utils import (
    get_arguments,
    set_random_seed,
    load_models,
    build_test_data_loader,
    clip_classifier,
)

from runners.zs_infer import infer as run_zero_shot


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    baseline_parser = argparse.ArgumentParser(add_help=False)
    baseline_parser.add_argument("--baseline-exp-id", required=True)
    baseline_parser.add_argument("--baseline-method", required=True, choices=["zs"])
    baseline_parser.add_argument("--baseline-method-full", required=True)
    baseline_parser.add_argument("--baseline-gpu", default="0")
    baseline_parser.add_argument("--baseline-result-root", default="results/E1_text_prototype_enhancement")
    baseline_parser.add_argument("--modelnet-clean-data-root", default="data/modelnet_c")
    baseline_parser.add_argument("--modelnet-clean-cor-type", default="clean")

    baseline_args, remaining = baseline_parser.parse_known_args()

    old_argv = sys.argv
    sys.argv = [old_argv[0]] + remaining
    base_args = get_arguments()
    sys.argv = old_argv

    return baseline_args, base_args


def make_wandb_run_name(args):
    if args.lm3d == "openshape":
        prefix = f"[zs_infer-manual-prompts]/global_feat/{args.lm3d}-{args.oshape_version}"
    elif args.lm3d == "ulip":
        prefix = f"[zs_infer-manual-prompts]/global_feat/{args.ulip_version}"
    else:
        prefix = f"[zs_infer-manual-prompts]/global_feat/{args.lm3d}"

    return f"{prefix}/modelnet_clean-{args.npoints}/{args.cor_type}"


def write_summary_header(summary_file):
    with summary_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_id",
            "dataset",
            "data_root",
            "corruption",
            "severity",
            "cor_type",
            "file",
            "sonn_variant",
            "backbone",
            "method",
            "method_full",
            "acc",
            "status",
            "gpu",
            "log_path",
        ])


def append_summary_row(summary_file, row):
    with summary_file.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    baseline_args, args = parse_args()

    if args.wandb and wandb is None:
        print("[E1] wandb is not installed; disabling wandb logging for this run.")
        args.wandb = False

    exp_id = baseline_args.baseline_exp_id
    method = baseline_args.baseline_method
    method_full = baseline_args.baseline_method_full
    physical_gpu = baseline_args.baseline_gpu

    project_root = Path("/root/autodl-tmp/MCM-PC-2")
    pc_root = project_root / "Point-Cache"

    result_root = pc_root / baseline_args.baseline_result_root
    run_dir = result_root / exp_id
    log_dir = run_dir / "logs"
    wandb_dir = run_dir / "wandb"
    summary_file = run_dir / "summary.csv"

    log_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir.mkdir(parents=True, exist_ok=True)

    args.dataset = "modelnet_c"
    args.modelnet_c_root = baseline_args.modelnet_clean_data_root
    args.cor_type = baseline_args.modelnet_clean_cor_type

    data_root = args.modelnet_c_root
    data_file = Path(data_root) / f"{args.cor_type}.h5"
    corruption = "clean"
    severity = "-"
    backbone = "ULIP"
    sonn_variant = "-"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{exp_id}_{args.cor_type}_{timestamp}.log"

    write_summary_header(summary_file)

    init_log_buffer = io.StringIO()

    with log_file.open("w") as lf:
        with redirect_stdout(Tee(sys.__stdout__, lf, init_log_buffer)), redirect_stderr(Tee(sys.__stderr__, lf, init_log_buffer)):
            print("============================================================")
            print("E1 ULIP clean ModelNet zero-shot text prototype runner")
            print(f"EXP_ID: {exp_id}")
            print(f"Method: {method_full}")
            print(f"Physical GPU: {physical_gpu}")
            print("Internal device: 0")
            print("Dataset label: modelnet_clean")
            print(f"Loader dataset: {args.dataset}")
            print(f"Data root: {data_root}")
            print(f"Data file: {data_file}")
            print(f"cor_type: {args.cor_type}")
            print(f"Result dir: {run_dir}")
            print(f"Log file: {log_file}")
            print("============================================================")

            if not data_file.exists():
                append_summary_row(summary_file, [
                    exp_id,
                    args.dataset,
                    data_root,
                    corruption,
                    severity,
                    args.cor_type,
                    str(data_file),
                    sonn_variant,
                    backbone,
                    method,
                    method_full,
                    "",
                    "missing_file",
                    physical_gpu,
                    str(log_file),
                ])
                raise FileNotFoundError(f"Missing file: {data_file}")

            run = None
            try:
                set_random_seed(args.seed)

                clip_model, lm3d_model = load_models(args)

                print(f"Processing {args.dataset} dataset as clean ModelNet.")
                test_loader, classnames, template = build_test_data_loader(args, args.dataset, args.data_root, None)
                print(f">>> classnames: {classnames}")

                clip_weights = clip_classifier(args, classnames, template, clip_model)

                if args.wandb:
                    run = wandb.init(
                        project="Point-TDA",
                        name=make_wandb_run_name(args),
                        reinit=True,
                    )

                acc = run_zero_shot(args, lm3d_model, test_loader, clip_weights)

                if args.wandb:
                    wandb.log({"modelnet_clean": float(acc)})
                    if run is not None:
                        run.finish()

                append_summary_row(summary_file, [
                    exp_id,
                    args.dataset,
                    data_root,
                    corruption,
                    severity,
                    args.cor_type,
                    str(data_file),
                    sonn_variant,
                    backbone,
                    method,
                    method_full,
                    f"{float(acc):.2f}",
                    "done",
                    physical_gpu,
                    str(log_file),
                ])

                print("============================================================")
                print(f"DONE: modelnet_clean, acc={float(acc):.2f}")
                print(f"summary: {summary_file}")
                print("============================================================")

            except Exception:
                if run is not None:
                    run.finish()
                append_summary_row(summary_file, [
                    exp_id,
                    args.dataset,
                    data_root,
                    corruption,
                    severity,
                    args.cor_type,
                    str(data_file),
                    sonn_variant,
                    backbone,
                    method,
                    method_full,
                    "",
                    "failed",
                    physical_gpu,
                    str(log_file),
                ])
                traceback.print_exc()
                raise

    print()
    print("============================================================")
    print(f"Clean ModelNet run finished: {exp_id}")
    print(f"summary: {summary_file}")
    print("============================================================")
    print(summary_file.read_text())


if __name__ == "__main__":
    main()
