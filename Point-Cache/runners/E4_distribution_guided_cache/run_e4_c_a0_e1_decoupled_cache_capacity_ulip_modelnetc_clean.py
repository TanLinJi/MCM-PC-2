#!/usr/bin/env python
import csv
import gc
import io
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import torch

POINT_CACHE_ROOT = Path(__file__).resolve().parents[2]
if str(POINT_CACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(POINT_CACHE_ROOT))

from utils.utils import (  # noqa: E402
    build_test_data_loader,
    get_config_file,
    load_models,
    set_random_seed,
)
from runners.zs_infer import infer as run_zero_shot  # noqa: E402
from runners.model_with_global_cache import run_test_tda as run_global_cache  # noqa: E402
from runners.E4_distribution_guided_cache.model_e4_c_a0_e1_decoupled_cache_capacity import (  # noqa: E402
    run_test_tda as run_hierarchical_cache,
)
from runners.E4_distribution_guided_cache.run_e4_c_a0_e1_decoupled_cache_capacity_ulip_modelnetc_s2 import (  # noqa: E402
    Tee,
    build_clip_weights_once,
    build_text_distribution_once,
    build_text_distribution_template,
    parse_args,
)


DATASET = "modelnet_c"
DATA_ROOT = "data/modelnet_c"
COR_TYPE = "clean"
DATA_FILE = Path(DATA_ROOT) / "clean.h5"


def _env_int(name, default):
    raw = os.environ.get(name, None)
    if raw is None or str(raw).strip() == "":
        return int(default)
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def write_summary_header(summary_file):
    with summary_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_id",
            "dataset",
            "data_root",
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


def run_clean_modelnet(
    args,
    baseline_method,
    method_full,
    cfg,
    clip_model,
    lm3d_model,
    clip_weights_state,
    exp_id,
    data_root,
    log_file,
    init_log_text,
):
    args.cor_type = COR_TYPE

    with log_file.open("w") as lf:
        lf.write(init_log_text)
        lf.flush()

        with redirect_stdout(Tee(sys.__stdout__, lf)), redirect_stderr(Tee(sys.__stderr__, lf)):
            test_loader = None

            try:
                print("============================================================")
                print(f"Running {exp_id}")
                print(f"method={method_full}")
                print(f"dataset={DATASET}")
                print(f"cor_type={COR_TYPE}")
                print(f"data_file={DATA_FILE}")
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
                print("[09_1-clean] clip prompt source:", args.prompt_source)
                print("[09_1-clean] text distribution prompt source:", args.e4_text_dist_prompt_source)
                print("[09_1-clean] text_dist classes:", len(text_dist))

                if baseline_method == "zs":
                    acc = run_zero_shot(args, lm3d_model, test_loader, clip_weights)
                elif baseline_method == "zs_global":
                    acc = run_global_cache(args, cfg["positive"], cfg["negative"], test_loader, lm3d_model, clip_weights)
                elif baseline_method == "zs_global_local":
                    acc = run_hierarchical_cache(
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

                print("============================================================")
                print(f"DONE: {COR_TYPE}, acc={float(acc):.2f}")
                print("============================================================")

                return float(acc)

            except Exception:
                print("ERROR: clean ModelNet run failed.")
                traceback.print_exc()
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
    summary_file = run_dir / "summary.csv"

    log_dir.mkdir(parents=True, exist_ok=True)

    args.dataset = DATASET
    args.modelnet_c_root = DATA_ROOT
    args.data_root = DATA_ROOT
    args.cor_type = COR_TYPE
    args.baseline_exp_id = exp_id
    args.baseline_result_root = str(result_root)
    args.baseline_method_full = method_full
    args.e4_variant = "09_1-E4-C-A0+E1-textdist-only-decoupled-cache-capacity-clean"
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
        print("09_1 decoupled cache-capacity on clean ModelNet-C runner")
        print(f"EXP_ID: {exp_id}")
        print(f"Method: {method_full}")
        print(f"Physical GPU: {physical_gpu}")
        print("Internal device: 0")
        print(f"Dataset: {args.dataset}")
        print(f"Data root: {data_root}")
        print(f"Data file: {DATA_FILE}")
        print(f"Result dir: {run_dir}")
        print("Variant: 09_1-E4-C-A0+E1-textdist-only-decoupled-cache-capacity-clean")
        print(f"Clip prompt source: {args.prompt_source}")
        print(f"Text distribution prompt source: {args.e4_text_dist_prompt_source}")
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

    if not DATA_FILE.exists():
        append_summary_row(summary_file, [
            exp_id,
            args.dataset,
            data_root,
            COR_TYPE,
            str(DATA_FILE),
            sonn_variant,
            backbone,
            method,
            method_full,
            "",
            "missing_file",
            physical_gpu,
            "",
        ])
        raise FileNotFoundError(f"Missing file: {DATA_FILE}")

    clip_weights_state = {
        "clip_weights": None,
        "classnames": None,
        "template": None,
        "text_dist": None,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{exp_id}_{COR_TYPE}_{timestamp}.log"

    try:
        acc = run_clean_modelnet(
            args=args,
            baseline_method=method,
            method_full=method_full,
            cfg=cfg,
            clip_model=clip_model,
            lm3d_model=lm3d_model,
            clip_weights_state=clip_weights_state,
            exp_id=exp_id,
            data_root=data_root,
            log_file=log_file,
            init_log_text=init_log_text,
        )

        append_summary_row(summary_file, [
            exp_id,
            args.dataset,
            data_root,
            COR_TYPE,
            str(DATA_FILE),
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
            COR_TYPE,
            str(DATA_FILE),
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
    print(f"Clean ModelNet-C run finished: {exp_id}")
    print(f"summary: {summary_file}")
    print("============================================================")
    print(summary_file.read_text())


if __name__ == "__main__":
    main()

