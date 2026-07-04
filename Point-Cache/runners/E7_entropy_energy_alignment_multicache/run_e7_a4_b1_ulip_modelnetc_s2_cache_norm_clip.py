#!/usr/bin/env python
"""
E7-A4-B1 runner: ULIP + ModelNet-C severity=2, cache-norm-clipped candidate-pool alignment-core cache.

设计文档：
docs/experiments/E7_entropy_energy_alignment_multicache/E7_A_versions/A4_B1_cache_norm_clip.md

与 E7-A0/A1/A2/A3 runner 的关键差异：
1. cache_type 固定为 "global"（E7 只用全局点云特征，不使用 local cache）；
2. 路由到 E7-A4-B1 的缓存总得分范数裁剪模型 run_test_tda；
3. 零样本分类器 clip_weights 来自 manual_full（--prompt-source manual_full）；
4. text_dist 来自 E1 LLM 描述，仅用于缓存替换打分，不替换最终分类器。

本文件不修改任何已有 E4/E5/A4 文件。
"""

import os
import argparse
import csv
import gc
import io
import sys
import traceback
from copy import copy
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path

import torch
import wandb

POINT_CACHE_ROOT = Path(__file__).resolve().parents[2]
if str(POINT_CACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(POINT_CACHE_ROOT))

import clip
import open_clip

from utils.utils import (
    get_arguments,
    set_random_seed,
    load_models,
    build_test_data_loader,
    clip_classifier,
    get_config_file,
    _build_prompt_texts,
    _is_weighted_prompt_fusion,
)

from datasets.prompt_utils import get_prompt_template
from runners.zs_infer import infer as run_zero_shot
from runners.E7_entropy_energy_alignment_multicache.model_e7_a4_b1_cache_norm_clip import (
    run_test_tda as run_e7_a4_b1_cache_norm_clip,
)


CORRUPTIONS = [
    "add_global",
    "add_local",
    "dropout_global",
    "dropout_local",
    "rotate",
    "scale",
    "jitter",
]

SEVERITIES = [2]


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
    baseline_parser.add_argument("--baseline-method", required=True, choices=["zs", "zs_global"])
    baseline_parser.add_argument("--baseline-method-full", required=True)
    baseline_parser.add_argument("--baseline-gpu", default="0")
    baseline_parser.add_argument("--baseline-result-root", default="results/E7_entropy_energy_alignment_multicache")

    baseline_args, remaining = baseline_parser.parse_known_args()

    old_argv = sys.argv
    sys.argv = [old_argv[0]] + remaining
    base_args = get_arguments()
    sys.argv = old_argv

    return baseline_args, base_args


def make_wandb_run_name(args):
    dataset_name = args.dataset

    if args.lm3d == "openshape":
        prefix = f"[E7-test-manual-prompts]/{args.cache_type}_cache/{args.lm3d}-{args.oshape_version}"
    elif args.lm3d == "ulip":
        prefix = f"[E7-test-manual-prompts]/{args.cache_type}_cache/{args.ulip_version}"
    else:
        prefix = f"[E7-test-manual-prompts]/{args.cache_type}_cache/{args.lm3d}"

    if "_c" in dataset_name:
        return f"{prefix}/{dataset_name}-{args.npoints}/{args.cor_type}"
    return f"{prefix}/{dataset_name}-{args.npoints}"


def write_summary_header(summary_file):
    with summary_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_id", "dataset", "data_root", "corruption", "severity", "cor_type",
            "file", "sonn_variant", "backbone", "method", "method_full",
            "acc", "status", "gpu", "log_path",
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
                "Classnames changed inside ModelNet-C severity=2 loop. "
                "clip_weights must be rebuilt, but this runner expects fixed ModelNet-C classes."
            )
    return clip_weights_state["clip_weights"]


def _tokenize_texts(args, texts):
    if args.lm3d in ["uni3d", "ulip"]:
        return clip.tokenize(texts).cuda()
    if args.lm3d == "openshape":
        return open_clip.tokenizer.tokenize(texts).cuda()
    raise ValueError(f"Unsupported lm3d type: {args.lm3d}")


@torch.no_grad()
def _encode_prompt_embeddings(args, clip_model, texts):
    tokenized_texts = _tokenize_texts(args, texts)
    class_embeddings = clip_model.encode_text(tokenized_texts)
    class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return class_embeddings.float()


def _distribution_from_prompt_embeddings(class_embeddings, min_var):
    mean = class_embeddings.mean(dim=0, keepdim=True)
    if class_embeddings.size(0) <= 1:
        var = torch.ones_like(mean) * min_var
    else:
        var = class_embeddings.var(dim=0, unbiased=True, keepdim=True).clamp_min(min_var)
    return {"count": int(class_embeddings.size(0)), "mean": mean.detach(), "var": var.detach()}


def _distribution_from_weighted_prompt_embeddings(class_embeddings, weights, min_var):
    weights = weights.to(device=class_embeddings.device, dtype=class_embeddings.dtype).view(-1, 1)
    weights = weights / weights.sum().clamp_min(1e-12)
    mean = (class_embeddings * weights).sum(dim=0, keepdim=True)
    if class_embeddings.size(0) <= 1:
        var = torch.ones_like(mean) * min_var
    else:
        centered = class_embeddings - mean
        var = (centered.pow(2) * weights).sum(dim=0, keepdim=True).clamp_min(min_var)
    return {"count": int(class_embeddings.size(0)), "mean": mean.detach(), "var": var.detach()}


def _build_weighted_fusion_prompt_distribution(args, clip_model, classname, template, min_var):
    static_weight = float(template.get("static_weight", 0.75))
    dynamic_weight = float(template.get("dynamic_weight", 0.25))

    static_texts = _build_prompt_texts(classname, template["static_template"])
    dynamic_texts = _build_prompt_texts(classname, template["dynamic_template"])

    static_embeddings = _encode_prompt_embeddings(args, clip_model, static_texts)
    dynamic_embeddings = _encode_prompt_embeddings(args, clip_model, dynamic_texts)

    class_embeddings = torch.cat([static_embeddings, dynamic_embeddings], dim=0)

    static_weights = torch.full(
        (static_embeddings.size(0),),
        static_weight / float(max(static_embeddings.size(0), 1)),
        device=class_embeddings.device, dtype=class_embeddings.dtype,
    )
    dynamic_weights = torch.full(
        (dynamic_embeddings.size(0),),
        dynamic_weight / float(max(dynamic_embeddings.size(0), 1)),
        device=class_embeddings.device, dtype=class_embeddings.dtype,
    )
    weights = torch.cat([static_weights, dynamic_weights], dim=0)

    return _distribution_from_weighted_prompt_embeddings(class_embeddings, weights, min_var)


@torch.no_grad()
def build_text_distribution_once(args, clip_model, clip_weights_state, classnames, template):
    if clip_weights_state.get("text_dist") is not None:
        if list(classnames) != clip_weights_state["classnames"]:
            raise RuntimeError(
                "Classnames changed inside ModelNet-C severity=2 loop. "
                "text_dist must be rebuilt, but this runner expects fixed ModelNet-C classes."
            )
        return clip_weights_state["text_dist"]

    min_var = float(os.environ.get("E7_TEXT_DIST_MIN_VAR", os.environ.get("E7_DIST_MIN_VAR", "1e-4")))
    text_dist = {}

    for class_index, classname in enumerate(classnames):
        if _is_weighted_prompt_fusion(template):
            text_dist[int(class_index)] = _build_weighted_fusion_prompt_distribution(
                args, clip_model, classname, template, min_var,
            )
        else:
            texts = _build_prompt_texts(classname, template)
            class_embeddings = _encode_prompt_embeddings(args, clip_model, texts)
            text_dist[int(class_index)] = _distribution_from_prompt_embeddings(class_embeddings, min_var)

    clip_weights_state["text_dist"] = text_dist
    return text_dist


def build_text_distribution_template(args, classnames, fallback_template):
    text_dist_prompt_source = getattr(args, "e7_text_dist_prompt_source", args.prompt_source)
    if text_dist_prompt_source == args.prompt_source:
        return fallback_template

    text_args = copy(args)
    text_args.prompt_source = text_dist_prompt_source
    return get_prompt_template(text_args, classnames, dataset_name=args.dataset)


def run_one_corruption(
    args, baseline_method, method_full, cfg, clip_model, lm3d_model,
    clip_weights_state, exp_id, corruption, severity, data_root, log_file, init_log_text,
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
                text_template = (
                    build_text_distribution_template(args, classnames, template)
                    if clip_weights_state.get("text_dist") is None
                    else template
                )
                text_dist = build_text_distribution_once(args, clip_model, clip_weights_state, classnames, text_template)
                print(f"[E7-A4-B1] clip (final classifier) prompt source: {args.prompt_source}")
                print(f"[E7-A4-B1] text distribution prompt source: {args.e7_text_dist_prompt_source}")
                print(f"[E7-A4-B1] text_dist classes: {len(text_dist)}")

                if args.wandb:
                    run_name = make_wandb_run_name(args)
                    run_config = cfg if cfg is not None else None
                    run = wandb.init(project="Point-TDA", config=run_config, name=run_name, reinit=True)

                if baseline_method == "zs":
                    acc = run_zero_shot(args, lm3d_model, test_loader, clip_weights)
                elif baseline_method == "zs_global":
                    acc = run_e7_a4_b1_cache_norm_clip(
                        args, cfg["positive"], cfg["negative"], test_loader,
                        lm3d_model, clip_weights, text_dist=text_dist,
                    )
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

    # E7 只用全局点云特征：强制 cache_type=global。
    args.cache_type = "global"

    args.dataset = "modelnet_c"
    args.modelnet_c_root = "data/modelnet_c"
    args.baseline_exp_id = exp_id
    args.baseline_result_root = str(result_root)
    args.baseline_method_full = method_full
    args.e7_variant = "E7-A4-B1-cache-norm-clip"
    args.e7_text_dist_prompt_source = os.environ.get("E7_TEXT_DIST_PROMPT_SOURCE", args.prompt_source)

    data_root = args.modelnet_c_root
    backbone = "ULIP"
    sonn_variant = "-"

    write_summary_header(summary_file)

    init_log_buffer = io.StringIO()

    with redirect_stdout(Tee(sys.__stdout__, init_log_buffer)), redirect_stderr(Tee(sys.__stderr__, init_log_buffer)):
        print("============================================================")
        print("E7-A4-B1 ULIP ModelNet-C severity=2 Cache-Norm-Clipped Candidate-Pool Alignment-Core runner")
        print(f"EXP_ID: {exp_id}")
        print(f"Method: {method_full}")
        print(f"Physical GPU: {physical_gpu}")
        print("Internal device: 0")
        print(f"Dataset: {args.dataset}")
        print(f"Data root: {data_root}")
        print(f"Result dir: {run_dir}")
        print("Variant: E7-A4-B1 cache-norm-clipped candidate-pool alignment-core cache (global feature only, no local cache)")
        print(f"Final classifier prompt source: {args.prompt_source}")
        print(f"Text distribution prompt source: {args.e7_text_dist_prompt_source}")
        print(f"cache_type (forced): {args.cache_type}")
        print("Model will be loaded once, then 7 severity=2 cor_type values will be evaluated.")
        print("============================================================")

        set_random_seed(args.seed)

        clip_model, lm3d_model = load_models(args)

        cfg = None
        if method in ["zs_global"]:
            cfg = get_config_file(args, args.config, args.dataset)
            print("\nRunning dataset configurations:")
            print(cfg, "\n")

    init_log_text = init_log_buffer.getvalue()

    clip_weights_state = {
        "clip_weights": None,
        "classnames": None,
        "template": None,
        "text_dist": None,
    }

    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            cor_type = f"{corruption}_{severity}"
            data_file = Path(data_root) / f"{cor_type}.h5"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{exp_id}_{cor_type}_{timestamp}.log"

            if not data_file.exists():
                append_summary_row(summary_file, [
                    exp_id, args.dataset, data_root, corruption, severity, cor_type,
                    str(data_file), sonn_variant, backbone, method, method_full,
                    "", "missing_file", physical_gpu, str(log_file),
                ])
                raise FileNotFoundError(f"Missing file: {data_file}")

            try:
                acc = run_one_corruption(
                    args=args, baseline_method=method, method_full=method_full, cfg=cfg,
                    clip_model=clip_model, lm3d_model=lm3d_model, clip_weights_state=clip_weights_state,
                    exp_id=exp_id, corruption=corruption, severity=severity,
                    data_root=data_root, log_file=log_file, init_log_text=init_log_text,
                )

                append_summary_row(summary_file, [
                    exp_id, args.dataset, data_root, corruption, severity, cor_type,
                    str(data_file), sonn_variant, backbone, method, method_full,
                    f"{acc:.2f}", "done", physical_gpu, str(log_file),
                ])

            except Exception:
                append_summary_row(summary_file, [
                    exp_id, args.dataset, data_root, corruption, severity, cor_type,
                    str(data_file), sonn_variant, backbone, method, method_full,
                    "", "failed", physical_gpu, str(log_file),
                ])
                raise

    print()
    print("============================================================")
    print(f"All 7 severity=2 runs finished: {exp_id}")
    print(f"summary: {summary_file}")
    print("============================================================")
    print(summary_file.read_text())


if __name__ == "__main__":
    main()
