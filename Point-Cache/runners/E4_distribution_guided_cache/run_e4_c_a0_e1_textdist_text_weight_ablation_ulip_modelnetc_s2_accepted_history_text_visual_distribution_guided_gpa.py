#!/usr/bin/env python
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
from runners.model_with_global_cache import run_test_tda as run_global_cache
from runners.E4_distribution_guided_cache import model_e4_c_accepted_history_text_visual_distribution_guided_gpa as e4_model


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
DEFAULT_TEXT_SCORE_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20]


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
    baseline_parser.add_argument("--baseline-result-root", default="results/E0_baseline")

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
            "text_score_weight",
            "acc",
            "status",
            "gpu",
            "log_path",
        ])


def append_summary_row(summary_file, row):
    with summary_file.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def parse_text_score_weights(raw_value):
    if raw_value is None or str(raw_value).strip() == "":
        return list(DEFAULT_TEXT_SCORE_WEIGHTS)

    weights = []
    for item in str(raw_value).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        weight = float(item)
        if weight < 0:
            raise ValueError(f"E4 text score weight must be non-negative, got {weight}")
        weights.append(weight)

    if not weights:
        raise ValueError("E4_TEXT_SCORE_WEIGHT_LIST did not contain any valid weights.")

    seen = set()
    unique_weights = []
    for weight in weights:
        key = f"{weight:.8f}"
        if key in seen:
            continue
        seen.add(key)
        unique_weights.append(weight)
    return unique_weights


def format_text_score_weight(weight):
    return f"{float(weight):.2f}"


def text_score_weight_slug(weight):
    return format_text_score_weight(weight).replace(".", "_")


def write_summary_files_header(summary_file, summary_files_by_weight):
    write_summary_header(summary_file)
    for path in summary_files_by_weight.values():
        write_summary_header(path)


def rename_gpa_stats_for_weight(args, cor_type, text_score_weight):
    result_root = Path(args.baseline_result_root)
    exp_id = args.baseline_exp_id
    stats_dir = result_root / exp_id / "gpa_stats"
    if not stats_dir.exists():
        return

    slug = text_score_weight_slug(text_score_weight)
    rename_pairs = [
        (
            stats_dir / f"{cor_type}_gpa_stats.json",
            stats_dir / f"{cor_type}_text_weight_{slug}_gpa_stats.json",
        ),
        (
            stats_dir / f"gpa_replacement_events_{cor_type}.jsonl",
            stats_dir / f"gpa_replacement_events_{cor_type}_text_weight_{slug}.jsonl",
        ),
    ]

    for src, dst in rename_pairs:
        if not src.exists():
            continue
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        print(f"[E4-C-A0+E1-textdist text-weight ablation] Saved weight-specific stats: {dst}")


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

    return {
        "count": int(class_embeddings.size(0)),
        "mean": mean.detach(),
        "var": var.detach(),
    }


def _distribution_from_weighted_prompt_embeddings(class_embeddings, weights, min_var):
    weights = weights.to(device=class_embeddings.device, dtype=class_embeddings.dtype).view(-1, 1)
    weights = weights / weights.sum().clamp_min(1e-12)

    mean = (class_embeddings * weights).sum(dim=0, keepdim=True)

    if class_embeddings.size(0) <= 1:
        var = torch.ones_like(mean) * min_var
    else:
        centered = class_embeddings - mean
        var = (centered.pow(2) * weights).sum(dim=0, keepdim=True).clamp_min(min_var)

    return {
        "count": int(class_embeddings.size(0)),
        "mean": mean.detach(),
        "var": var.detach(),
    }


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
        device=class_embeddings.device,
        dtype=class_embeddings.dtype,
    )
    dynamic_weights = torch.full(
        (dynamic_embeddings.size(0),),
        dynamic_weight / float(max(dynamic_embeddings.size(0), 1)),
        device=class_embeddings.device,
        dtype=class_embeddings.dtype,
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

    min_var = float(os.environ.get("E4_TEXT_DIST_MIN_VAR", os.environ.get("E4_DIST_MIN_VAR", "1e-4")))
    text_dist = {}

    for class_index, classname in enumerate(classnames):
        if _is_weighted_prompt_fusion(template):
            text_dist[int(class_index)] = _build_weighted_fusion_prompt_distribution(
                args,
                clip_model,
                classname,
                template,
                min_var,
            )
        else:
            texts = _build_prompt_texts(classname, template)
            class_embeddings = _encode_prompt_embeddings(args, clip_model, texts)
            text_dist[int(class_index)] = _distribution_from_prompt_embeddings(class_embeddings, min_var)

    clip_weights_state["text_dist"] = text_dist
    return text_dist


def build_text_distribution_template(args, classnames, fallback_template):
    text_dist_prompt_source = getattr(args, "e4_text_dist_prompt_source", args.prompt_source)

    if text_dist_prompt_source == args.prompt_source:
        return fallback_template

    text_args = copy(args)
    text_args.prompt_source = text_dist_prompt_source
    return get_prompt_template(text_args, classnames, dataset_name=args.dataset)


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
                text_template = (
                    build_text_distribution_template(args, classnames, template)
                    if clip_weights_state.get("text_dist") is None
                    else template
                )
                text_dist = build_text_distribution_once(args, clip_model, clip_weights_state, classnames, text_template)
                print(f"[E4-C-A0+E1-textdist] clip prompt source: {args.prompt_source}")
                print(f"[E4-C-A0+E1-textdist] text distribution prompt source: {args.e4_text_dist_prompt_source}")
                print(f"[E4-C] text_dist classes: {len(text_dist)}")

                results = []
                for text_score_weight in args.e4_text_score_weights:
                    run = None
                    e4_model.TEXT_SCORE_WEIGHT = float(text_score_weight)
                    args.e4_text_score_weight = float(text_score_weight)

                    print("------------------------------------------------------------")
                    print(
                        "[E4-C-A0+E1-textdist text-weight ablation] "
                        f"E4_TEXT_SCORE_WEIGHT={format_text_score_weight(text_score_weight)}"
                    )
                    print("------------------------------------------------------------")

                    set_random_seed(args.seed)

                    if args.wandb:
                        run_name = make_wandb_run_name(args, baseline_method)
                        run_name = f"{run_name}/text_weight_{format_text_score_weight(text_score_weight)}"
                        run_config = dict(cfg) if isinstance(cfg, dict) else cfg
                        run = wandb.init(project="Point-TDA", config=run_config, name=run_name, reinit=True)

                    if baseline_method == "zs":
                        acc = run_zero_shot(args, lm3d_model, test_loader, clip_weights)
                    elif baseline_method == "zs_global":
                        acc = run_global_cache(args, cfg["positive"], cfg["negative"], test_loader, lm3d_model, clip_weights)
                    elif baseline_method == "zs_global_local":
                        acc = e4_model.run_test_tda(
                            args,
                            cfg["positive"],
                            cfg["negative"],
                            test_loader,
                            lm3d_model,
                            clip_weights,
                            text_dist=text_dist,
                        )
                    else:
                        raise ValueError(f"Unsupported baseline_method: {baseline_method}")

                    if args.wandb:
                        wandb.log({
                            f"{args.dataset}": float(acc),
                            "E4_TEXT_SCORE_WEIGHT": float(text_score_weight),
                        })
                        if run is not None:
                            run.finish()

                    print("------------------------------------------------------------")
                    print(
                        f"DONE: {cor_type}, "
                        f"text_weight={format_text_score_weight(text_score_weight)}, "
                        f"acc={float(acc):.2f}"
                    )
                    print("------------------------------------------------------------")

                    rename_gpa_stats_for_weight(args, cor_type, text_score_weight)
                    results.append((float(text_score_weight), float(acc)))

                print("============================================================")
                print(f"DONE: {cor_type}, evaluated {len(results)} text weights.")
                print("============================================================")

                return results

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

    args.dataset = "modelnet_c"
    args.modelnet_c_root = "data/modelnet_c"
    args.baseline_exp_id = exp_id
    args.baseline_result_root = str(result_root)
    args.baseline_method_full = method_full
    args.e4_variant = "E4-C-A0+E1-textdist-text-weight-ablation"
    args.e4_text_dist_prompt_source = os.environ.get("E4_TEXT_DIST_PROMPT_SOURCE", args.prompt_source)
    args.e4_dist_eps = float(os.environ.get("E4_DIST_EPS", "1e-4"))
    args.e4_dist_min_var = float(os.environ.get("E4_DIST_MIN_VAR", "1e-4"))
    args.e4_text_dist_eps = float(os.environ.get("E4_TEXT_DIST_EPS", str(args.e4_dist_eps)))
    args.e4_text_dist_min_var = float(os.environ.get("E4_TEXT_DIST_MIN_VAR", str(args.e4_dist_min_var)))
    args.e4_text_score_weights = parse_text_score_weights(os.environ.get("E4_TEXT_SCORE_WEIGHT_LIST"))
    args.e4_text_score_weight = float(args.e4_text_score_weights[0])
    args.e4_score_norm_mode = os.environ.get("E4_SCORE_NORM_MODE", "none")
    args.e4_score_norm_min_count = int(os.environ.get("E4_SCORE_NORM_MIN_COUNT", "8"))
    args.e4_score_norm_eps = float(os.environ.get("E4_SCORE_NORM_EPS", "1e-6"))
    args.e4_score_norm_clip = float(os.environ.get("E4_SCORE_NORM_CLIP", "0"))

    data_root = args.modelnet_c_root
    backbone = "ULIP"
    sonn_variant = "-"
    summary_files_by_weight = {
        weight: run_dir / f"summary_text_weight_{text_score_weight_slug(weight)}.csv"
        for weight in args.e4_text_score_weights
    }

    write_summary_files_header(summary_file, summary_files_by_weight)

    init_log_buffer = io.StringIO()

    with redirect_stdout(Tee(sys.__stdout__, init_log_buffer)), redirect_stderr(Tee(sys.__stderr__, init_log_buffer)):
        print("============================================================")
        print("E4-C-A0+E1-textdist text-weight ablation ULIP ModelNet-C severity=2 runner")
        print(f"EXP_ID: {exp_id}")
        print(f"Method: {method_full}")
        print(f"Physical GPU: {physical_gpu}")
        print("Internal device: 0")
        print(f"Dataset: {args.dataset}")
        print(f"Data root: {data_root}")
        print(f"Result dir: {run_dir}")
        print("Variant: E4-C-A0+E1-textdist-text-weight-ablation")
        print(f"Clip prompt source: {args.prompt_source}")
        print(f"Text distribution prompt source: {args.e4_text_dist_prompt_source}")
        print(f"E4_DIST_EPS: {args.e4_dist_eps}")
        print(f"E4_DIST_MIN_VAR: {args.e4_dist_min_var}")
        print(f"E4_TEXT_DIST_EPS: {args.e4_text_dist_eps}")
        print(f"E4_TEXT_DIST_MIN_VAR: {args.e4_text_dist_min_var}")
        print(f"E4_TEXT_SCORE_WEIGHT_LIST: {[format_text_score_weight(w) for w in args.e4_text_score_weights]}")
        print(f"E4_SCORE_NORM_MODE: {args.e4_score_norm_mode}")
        print(f"E4_SCORE_NORM_MIN_COUNT: {args.e4_score_norm_min_count}")
        print(f"E4_SCORE_NORM_EPS: {args.e4_score_norm_eps}")
        print(f"E4_SCORE_NORM_CLIP: {args.e4_score_norm_clip}")
        print("Model will be loaded once, then 7 severity=2 cor_type values will be evaluated for each text weight.")
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
        "text_dist": None,
    }

    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            cor_type = f"{corruption}_{severity}"
            data_file = Path(data_root) / f"{cor_type}.h5"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{exp_id}_{cor_type}_{timestamp}.log"

            if not data_file.exists():
                for text_score_weight in args.e4_text_score_weights:
                    row = [
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
                        format_text_score_weight(text_score_weight),
                        "",
                        "missing_file",
                        physical_gpu,
                        str(log_file),
                    ]
                    append_summary_row(summary_file, row)
                    append_summary_row(summary_files_by_weight[text_score_weight], row)
                raise FileNotFoundError(f"Missing file: {data_file}")

            try:
                results = run_one_corruption(
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

                for text_score_weight, acc in results:
                    row = [
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
                        format_text_score_weight(text_score_weight),
                        f"{acc:.2f}",
                        "done",
                        physical_gpu,
                        str(log_file),
                    ]
                    append_summary_row(summary_file, row)
                    append_summary_row(summary_files_by_weight[text_score_weight], row)

            except Exception:
                for text_score_weight in args.e4_text_score_weights:
                    row = [
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
                        format_text_score_weight(text_score_weight),
                        "",
                        "failed",
                        physical_gpu,
                        str(log_file),
                    ]
                    append_summary_row(summary_file, row)
                    append_summary_row(summary_files_by_weight[text_score_weight], row)
                raise

    print()
    print("============================================================")
    print(f"All 7 severity=2 runs finished for {len(args.e4_text_score_weights)} text weights: {exp_id}")
    print(f"summary: {summary_file}")
    for weight, path in summary_files_by_weight.items():
        print(f"summary text_weight={format_text_score_weight(weight)}: {path}")
    print("============================================================")
    print(summary_file.read_text())


if __name__ == "__main__":
    main()
