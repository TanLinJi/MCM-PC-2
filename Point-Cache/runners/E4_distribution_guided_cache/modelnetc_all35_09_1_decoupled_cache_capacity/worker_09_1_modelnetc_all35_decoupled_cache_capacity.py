#!/usr/bin/env python
import argparse
import gc
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import torch

POINT_CACHE_ROOT = Path(__file__).resolve().parents[3]
if str(POINT_CACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(POINT_CACHE_ROOT))

from utils.utils import get_arguments, get_config_file, load_models, set_random_seed
from runners.E4_distribution_guided_cache.run_e4_c_a0_e1_decoupled_cache_capacity_ulip_modelnetc_s2 import (
    Tee,
    append_summary_row,
    run_one_corruption,
    write_summary_header,
)


def _env_int(name, default):
    raw = os.environ.get(name, None)
    if raw is None or str(raw).strip() == "":
        return int(default)
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def parse_args():
    worker_parser = argparse.ArgumentParser(add_help=False)
    worker_parser.add_argument("--worker-id", required=True)
    worker_parser.add_argument("--tasks-json", required=True)
    worker_parser.add_argument("--baseline-exp-id", required=True)
    worker_parser.add_argument("--baseline-method", required=True, choices=["zs", "zs_global", "zs_global_local"])
    worker_parser.add_argument("--baseline-method-full", required=True)
    worker_parser.add_argument("--baseline-gpu", required=True)
    worker_parser.add_argument("--baseline-result-root", default="results/E4_distribution_guided_cache")

    worker_args, remaining = worker_parser.parse_known_args()

    old_argv = sys.argv
    sys.argv = [old_argv[0]] + remaining
    base_args = get_arguments()
    sys.argv = old_argv

    return worker_args, base_args


def load_tasks(tasks_json):
    with Path(tasks_json).open("r", encoding="utf-8") as f:
        tasks = json.load(f)

    normalized = []
    for item in tasks:
        normalized.append({
            "corruption": str(item["corruption"]),
            "severity": int(item["severity"]),
        })
    return normalized


def main():
    worker_args, args = parse_args()

    exp_id = worker_args.baseline_exp_id
    method = worker_args.baseline_method
    method_full = worker_args.baseline_method_full
    physical_gpu = worker_args.baseline_gpu
    worker_id = worker_args.worker_id
    tasks = load_tasks(worker_args.tasks_json)

    pc_root = POINT_CACHE_ROOT
    result_root = pc_root / worker_args.baseline_result_root
    run_dir = result_root / exp_id
    log_dir = run_dir / "logs" / f"worker{worker_id}"
    summary_file = run_dir / f"summary_worker{worker_id}.csv"

    log_dir.mkdir(parents=True, exist_ok=True)

    args.dataset = "modelnet_c"
    args.modelnet_c_root = "data/modelnet_c"
    args.baseline_exp_id = exp_id
    args.baseline_result_root = str(result_root)
    args.baseline_method_full = method_full
    args.e4_variant = "09_1-E4-C-A0+E1-textdist-only-decoupled-cache-capacity-all35"
    args.e4_text_dist_prompt_source = os.environ.get("E4_TEXT_DIST_PROMPT_SOURCE", args.prompt_source)
    args.e4_dist_eps = float(os.environ.get("E4_DIST_EPS", "1e-4"))
    args.e4_dist_min_var = float(os.environ.get("E4_DIST_MIN_VAR", "1e-4"))
    args.e4_text_dist_eps = float(os.environ.get("E4_TEXT_DIST_EPS", str(args.e4_dist_eps)))
    args.e4_text_dist_min_var = float(os.environ.get("E4_TEXT_DIST_MIN_VAR", str(args.e4_dist_min_var)))
    args.e4_text_score_weight = float(os.environ.get("E4_TEXT_SCORE_WEIGHT", "0.15"))
    args.e4_score_norm_mode = os.environ.get("E4_SCORE_NORM_MODE", "running_zscore")
    args.e4_score_norm_min_count = int(os.environ.get("E4_SCORE_NORM_MIN_COUNT", "8"))
    args.e4_score_norm_eps = float(os.environ.get("E4_SCORE_NORM_EPS", "1e-6"))
    args.e4_score_norm_clip = float(os.environ.get("E4_SCORE_NORM_CLIP", "0"))
    args.e4_entropy_cap = _env_int("E4_ENTROPY_CAP", args.k_shot)
    args.e4_gpa_cap = _env_int("E4_GPA_CAP", args.k_shot)
    args.e4_local_cap = _env_int("E4_LOCAL_CAP", args.e4_gpa_cap)
    args.e4_neg_cap = _env_int("E4_NEG_CAP", 2)
    args.n_cluster = _env_int("E4_LOCAL_CENTERS", args.n_cluster)

    data_root = args.modelnet_c_root
    backbone = "ULIP"
    sonn_variant = "-"

    write_summary_header(summary_file)

    init_log_buffer = io.StringIO()

    with redirect_stdout(Tee(sys.__stdout__, init_log_buffer)), redirect_stderr(Tee(sys.__stderr__, init_log_buffer)):
        print("============================================================")
        print("09_1 all35 worker: decoupled cache-capacity on ModelNet-C")
        print(f"EXP_ID: {exp_id}")
        print(f"Worker: {worker_id}")
        print(f"Physical GPU: {physical_gpu}")
        print("Internal device: 0")
        print(f"Task count: {len(tasks)}")
        print(f"Dataset: {args.dataset}")
        print(f"Data root: {data_root}")
        print(f"Result dir: {run_dir}")
        print(f"Worker summary: {summary_file}")
        print("Variant: 09_1-E4-C-A0+E1-textdist-only-decoupled-cache-capacity-all35")
        print(f"Clip prompt source: {args.prompt_source}")
        print(f"Text distribution prompt source: {args.e4_text_dist_prompt_source}")
        print(f"E4_DIST_EPS: {args.e4_dist_eps}")
        print(f"E4_DIST_MIN_VAR: {args.e4_dist_min_var}")
        print(f"E4_TEXT_DIST_EPS: {args.e4_text_dist_eps}")
        print(f"E4_TEXT_DIST_MIN_VAR: {args.e4_text_dist_min_var}")
        print(f"E4_TEXT_SCORE_WEIGHT: {args.e4_text_score_weight}")
        print(f"E4_SCORE_NORM_MODE: {args.e4_score_norm_mode}")
        print(f"E4_SCORE_NORM_MIN_COUNT: {args.e4_score_norm_min_count}")
        print(f"E4_SCORE_NORM_EPS: {args.e4_score_norm_eps}")
        print(f"E4_SCORE_NORM_CLIP: {args.e4_score_norm_clip}")
        print(f"E4_ENTROPY_CAP: {args.e4_entropy_cap}")
        print(f"E4_GPA_CAP: {args.e4_gpa_cap}")
        print(f"E4_LOCAL_CAP: {args.e4_local_cap}")
        print(f"E4_NEG_CAP: {args.e4_neg_cap}")
        print(f"E4_LOCAL_CENTERS: {args.n_cluster}")
        print("The model is loaded once in this worker, then assigned corruptions run sequentially.")
        print("============================================================")

        set_random_seed(args.seed)

        clip_model, lm3d_model = load_models(args)

        cfg = None
        if method in ["zs_global", "zs_global_local"]:
            cfg = get_config_file(args, args.config, args.dataset)
            cfg["decoupled_cache_capacity"] = {
                "entropy_cap": int(args.e4_entropy_cap),
                "gpa_cap": int(args.e4_gpa_cap),
                "local_cap": int(args.e4_local_cap),
                "neg_cap": int(args.e4_neg_cap),
                "local_centers": int(args.n_cluster),
            }
            print("\nRunning dataset configurations:")
            print(cfg, "\n")

    init_log_text = init_log_buffer.getvalue()

    clip_weights_state = {
        "clip_weights": None,
        "classnames": None,
        "template": None,
        "text_dist": None,
    }

    for task in tasks:
        corruption = task["corruption"]
        severity = int(task["severity"])
        cor_type = f"{corruption}_{severity}"
        data_file = Path(data_root) / f"{cor_type}.h5"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{exp_id}_worker{worker_id}_{cor_type}_{timestamp}.log"

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
                f"{float(acc):.2f}",
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
            traceback.print_exc()
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    print("============================================================")
    print(f"Worker {worker_id} finished {len(tasks)} tasks.")
    print(f"summary: {summary_file}")
    print("============================================================")
    print(summary_file.read_text())


if __name__ == "__main__":
    main()

