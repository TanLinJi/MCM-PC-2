#!/usr/bin/env python
# 本 runner 用于在单个 Python 进程中复现 Uni3D × ScanObjNN-C hardest 全 35 个损坏设置的一种 baseline 方法，避免 35 次重复加载模型。
import argparse
import csv
import gc
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

import torch
import wandb

POINT_CACHE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POINT_CACHE_ROOT))

from utils.utils import (
    get_arguments,
    set_random_seed,
    load_models,
    build_test_data_loader,
    clip_classifier,
    get_config_file,
)

from runners.zs_infer import infer as run_zero_shot
from runners.model_with_global_cache import run_test_tda as run_global_cache
from runners.model_with_hierarchical_caches import run_test_tda as run_hierarchical_cache


CORRUPTIONS = [
    "add_global",
    "add_local",
    "dropout_global",
    "dropout_local",
    "rotate",
    "scale",
    "jitter",
]

SEVERITIES = [0, 1, 2, 3, 4]


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
    baseline_parser.add_argument("--baseline-method", required=True, choices=["zs", "zs_global", "zs_global_local"])
    baseline_parser.add_argument("--baseline-method-full", required=True)
    baseline_parser.add_argument("--baseline-gpu", default="0")
    baseline_parser.add_argument("--baseline-result-root", default="results/baseline")

    baseline_args, remaining = baseline_parser.parse_known_args()

    old_argv = sys.argv
    sys.argv = [old_argv[0]] + remaining
    base_args = get_arguments()
    sys.argv = old_argv

    return baseline_args, base_args


def make_wandb_run_name(args, baseline_method):
    dataset_name = args.dataset

    if baseline_method == "zs":
        if args.lm3d == "openshape":
            prefix = f"[zs_infer-manual-prompts]/global_feat/{args.lm3d}-{args.oshape_version}"
        elif args.lm3d == "ulip":
            prefix = f"[zs_infer-manual-prompts]/global_feat/{args.ulip_version}"
        else:
            prefix = f"[zs_infer-manual-prompts]/global_feat/{args.lm3d}"
    else:
        if args.lm3d == "openshape":
            prefix = f"[test-manual-prompts]/{args.cache_type}_cache/{args.lm3d}-{args.oshape_version}"
        elif args.lm3d == "ulip":
            prefix = f"[test-manual-prompts]/{args.cache_type}_cache/{args.ulip_version}"
        else:
            prefix = f"[test-manual-prompts]/{args.cache_type}_cache/{args.lm3d}"

    if "_c" in dataset_name and "sonn" in dataset_name:
        return f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}/{args.cor_type}"
    if "_c" in dataset_name:
        return f"{prefix}/{dataset_name}-{args.npoints}/{args.cor_type}"
    if "scanobjnn" in dataset_name or "scanobjectnn" in dataset_name:
        return f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}"
    if "sim2real_sonn" in dataset_name:
        return f"{prefix}/{dataset_name}-{args.sim2real_type}-{args.npoints}"
    if "pointda" in dataset_name:
        return f"{prefix}/{dataset_name}-{args.npoints}"
    return f"{prefix}/{dataset_name}-{args.npoints}"


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


def build_clip_weights_once(args, clip_model, clip_weights_state, classnames, template):
    if clip_weights_state["clip_weights"] is None:
        clip_weights_state["classnames"] = list(classnames)
        clip_weights_state["template"] = template
        clip_weights_state["clip_weights"] = clip_classifier(args, classnames, template, clip_model)
    else:
        if list(classnames) != clip_weights_state["classnames"]:
            raise RuntimeError(
                "Classnames changed inside ScanObjNN-C hardest all35 loop. "
                "clip_weights must be rebuilt, but this runner expects fixed ScanObjNN-C hardest classes."
            )

    return clip_weights_state["clip_weights"]


def run_one_corruption(
    args,
    baseline_method,
    method_full,
    cfg,
    clip_model,
    lm3d_model,
    clip_weights_state,
    exp_id,
    corruption,
    severity,
    data_root,
    log_file,
    init_log_text,
):
    cor_type = f"{corruption}_{severity}"
    args.cor_type = cor_type

    with log_file.open("w") as lf:
        lf.write(init_log_text)
        lf.flush()

        with redirect_stdout(Tee(sys.__stdout__, lf)), redirect_stderr(Tee(sys.__stderr__, lf)):
            run = None
            test_loader = None

            try:
                print("============================================================")
                print(f"Running {exp_id}")
                print(f"method={method_full}")
                print(f"corruption={corruption}")
                print(f"severity={severity}")
                print(f"cor_type={cor_type}")
                print(f"data_file={Path(data_root) / (cor_type + '.h5')}")
                print(f"log_file={log_file}")
                print("============================================================")

                set_random_seed(args.seed)

                print(f"Processing {args.dataset} dataset.")
                test_loader, classnames, template = build_test_data_loader(args, args.dataset, args.data_root, None)
                print(f">>> classnames: {classnames}")

                clip_weights = build_clip_weights_once(args, clip_model, clip_weights_state, classnames, template)

                if args.wandb:
                    run_name = make_wandb_run_name(args, baseline_method)
                    run_config = cfg if cfg is not None else None
                    run = wandb.init(project="Point-TDA", config=run_config, name=run_name, reinit=True)

                if baseline_method == "zs":
                    acc = run_zero_shot(args, lm3d_model, test_loader, clip_weights)
                elif baseline_method == "zs_global":
                    acc = run_global_cache(args, cfg["positive"], cfg["negative"], test_loader, lm3d_model, clip_weights)
                elif baseline_method == "zs_global_local":
                    acc = run_hierarchical_cache(args, cfg["positive"], cfg["negative"], test_loader, lm3d_model, clip_weights)
                else:
                    raise ValueError(f"Unsupported baseline_method: {baseline_method}")

                if args.wandb:
                    wandb.log({f"{args.dataset}": float(acc)})
                    if run is not None:
                        run.finish()

                print("============================================================")
                print(f"DONE: {cor_type}, acc={float(acc):.2f}")
                print("============================================================")

                return float(acc)

            except Exception:
                print("ERROR: current cor_type failed.")
                traceback.print_exc()
                if run is not None:
                    run.finish()
                raise

            finally:
                try:
                    if test_loader is not None:
                        del test_loader
                except Exception:
                    pass

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def main():
    baseline_args, args = parse_args()

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

    args.dataset = "sonn_c"
    args.sonn_variant = "hardest"

    # SONN_C loader expects shape_names.txt under data/sonn_c,
    # and reads h5 from data/sonn_c/{sonn_variant}/{cor_type}.h5.
    args.sonn_c_root = "data/sonn_c"
    args.data_root = args.sonn_c_root

    # Summary/log metadata should point to the actual h5 directory.
    data_root = "data/sonn_c/hardest"
    backbone = "Uni3D"
    sonn_variant = "hardest"

    expected_dataset = "sonn_c"
    expected_loader_root = "data/sonn_c"
    expected_summary_data_root = "data/sonn_c/hardest"
    expected_sonn_variant = "hardest"
    expected_ckpt = "weights/uni3d/scanobjnn/model.pt"

    if args.dataset != expected_dataset:
        raise RuntimeError(f"Unexpected dataset for 34 group: {args.dataset}; expected {expected_dataset}")
    if args.sonn_c_root != expected_loader_root:
        raise RuntimeError(f"Unexpected args.sonn_c_root for 34 group: {args.sonn_c_root}; expected {expected_loader_root}")
    if args.data_root != expected_loader_root:
        raise RuntimeError(f"Unexpected args.data_root for 34 group: {args.data_root}; expected {expected_loader_root}")
    if data_root != expected_summary_data_root:
        raise RuntimeError(f"Unexpected summary data_root for 34 group: {data_root}; expected {expected_summary_data_root}")
    if args.sonn_variant != expected_sonn_variant:
        raise RuntimeError(f"Unexpected args.sonn_variant for 34 group: {args.sonn_variant}; expected {expected_sonn_variant}")
    if sonn_variant != expected_sonn_variant:
        raise RuntimeError(f"Unexpected summary sonn_variant for 34 group: {sonn_variant}; expected {expected_sonn_variant}")
    if getattr(args, "ckpt_path", None) != expected_ckpt:
        raise RuntimeError(
            f"Unexpected Uni3D checkpoint for 34 group: {getattr(args, 'ckpt_path', None)}; "
            f"expected {expected_ckpt}"
        )

    write_summary_header(summary_file)

    init_log_buffer = io.StringIO()

    with redirect_stdout(Tee(sys.__stdout__, init_log_buffer)), redirect_stderr(Tee(sys.__stderr__, init_log_buffer)):
        print("============================================================")
        print("Optimized Uni3D ScanObjNN-C hardest all35 baseline runner")
        print(f"EXP_ID: {exp_id}")
        print(f"Method: {method_full}")
        print(f"Physical GPU: {physical_gpu}")
        print("Internal device: 0")
        print(f"Dataset: {args.dataset}")
        print(f"Data root: {data_root}")
        print(f"Result dir: {run_dir}")
        print("Model will be loaded once, then 35 cor_type values will be evaluated.")
        print("============================================================")

        set_random_seed(args.seed)

        clip_model, lm3d_model = load_models(args)

        cfg = None
        if method in ["zs_global", "zs_global_local"]:
            cfg = get_config_file(args, args.config, args.dataset)
            print("\nRunning dataset configurations:")
            print(cfg, "\n")

    init_log_text = init_log_buffer.getvalue()

    clip_weights_state = {
        "clip_weights": None,
        "classnames": None,
        "template": None,
    }

    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            cor_type = f"{corruption}_{severity}"
            data_file = Path(data_root) / f"{cor_type}.h5"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{exp_id}_{cor_type}_{timestamp}.log"

            if not data_file.exists():
                append_summary_row(summary_file, [
                    exp_id,
                    args.dataset,
                    data_root,
                    corruption,
                    severity,
                    cor_type,
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

            try:
                acc = run_one_corruption(
                    args=args,
                    baseline_method=method,
                    method_full=method_full,
                    cfg=cfg,
                    clip_model=clip_model,
                    lm3d_model=lm3d_model,
                    clip_weights_state=clip_weights_state,
                    exp_id=exp_id,
                    corruption=corruption,
                    severity=severity,
                    data_root=data_root,
                    log_file=log_file,
                    init_log_text=init_log_text,
                )

                append_summary_row(summary_file, [
                    exp_id,
                    args.dataset,
                    data_root,
                    corruption,
                    severity,
                    cor_type,
                    str(data_file),
                    sonn_variant,
                    backbone,
                    method,
                    method_full,
                    f"{acc:.2f}",
                    "done",
                    physical_gpu,
                    str(log_file),
                ])

            except Exception:
                append_summary_row(summary_file, [
                    exp_id,
                    args.dataset,
                    data_root,
                    corruption,
                    severity,
                    cor_type,
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
                raise

    print()
    print("============================================================")
    print(f"All 35 runs finished: {exp_id}")
    print(f"summary: {summary_file}")
    print("============================================================")
    print(summary_file.read_text())


if __name__ == "__main__":
    main()
