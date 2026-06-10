"""E4-CANC-DIAG-v2: Conflict Signal Validation.

This runner does NOT change Point-Cache prediction, positive cache, negative
cache, or final logits. It reproduces the original hierarchical cache pipeline
and only records diagnostic statistics.

Compared with E4-CANC-DIAG-v1, this version adds:
1. D_gl quantiles.
2. Multi-threshold sweep for conflict detection.
3. Correct safe-candidate precision:
       I_safe = I_H OR (I_M AND I_D)
4. Both mean-local and max-part disagreement diagnostics.
"""

import os
import sys
import json
import math

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *

from model_with_hierarchical_caches import (
    build_cache_in_advance,
    update_cache,
    compute_cache_logits,
    compute_local_cache_logits,
)


# ============================================================
# Diagnostic thresholds
# ============================================================

TAU_M = 0.20
TAU_P = 0.40

TAU_D_LIST = [
    0.0,
    1e-7,
    5e-7,
    1e-6,
    5e-6,
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    5e-3,
    1e-2,
    5e-2,
]

MIN_CONFLICT_RATE_FOR_BEST = 1.0  # percent


# ============================================================
# Utility functions
# ============================================================

def _safe_rate(num, den):
    if den == 0:
        return None
    return 100.0 * float(num) / float(den)


def _to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _to_int(x):
    if torch.is_tensor(x):
        return int(x.detach().cpu().item())
    return int(x)


def _percentile(values, q):
    if not values:
        return None

    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return xs[lo]

    weight = pos - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def _fmt_key_tau(tau):
    return f"{tau:.0e}" if tau < 0.001 else f"{tau:g}"


@torch.no_grad()
def compute_global_diagnostics(loss, prob_map, clip_weights):
    entropy = get_entropy(loss, clip_weights)
    entropy = _to_float(entropy)

    probs = prob_map.float()
    top2 = torch.topk(probs, k=2, dim=1).values[0]

    p1 = float(top2[0].detach().cpu().item())
    p2 = float(top2[1].detach().cpu().item())
    margin = p1 - p2

    return {
        "entropy": entropy,
        "p1": p1,
        "p2": p2,
        "margin": margin,
    }


@torch.no_grad()
def compute_local_conflict_diagnostics(patch_centers, clip_weights, global_pred):
    """Compute mean-local and max-part disagreement.

    Mean-local:
        D_mean = [max_{k != y_g} p_l_mean(k) - p_l_mean(y_g)]_+

    Max-part:
        D_max = max_r [max_{k != y_g} p_{l,r}(k) - p_{l,r}(y_g)]_+
    """
    global_pred = int(global_pred)

    local_logits = patch_centers @ clip_weights
    local_probs_parts = torch.softmax(local_logits.float(), dim=-1)

    # ---------- mean-local ----------
    mean_probs = local_probs_parts.mean(dim=0)

    mean_other_probs = mean_probs.clone()
    mean_other_probs[global_pred] = -1.0

    mean_alt_prob, mean_alt_cls = torch.max(mean_other_probs, dim=0)
    mean_global_prob = mean_probs[global_pred]

    d_mean = torch.clamp(mean_alt_prob - mean_global_prob, min=0.0)

    # ---------- max-part ----------
    part_global_probs = local_probs_parts[:, global_pred]

    part_other_probs = local_probs_parts.clone()
    part_other_probs[:, global_pred] = -1.0

    part_alt_probs, part_alt_cls = torch.max(part_other_probs, dim=1)
    part_conflicts = torch.clamp(part_alt_probs - part_global_probs, min=0.0)

    d_max, best_part_idx = torch.max(part_conflicts, dim=0)
    max_alt_cls = part_alt_cls[best_part_idx]

    return {
        "d_mean": float(d_mean.detach().cpu().item()),
        "alt_mean": int(mean_alt_cls.detach().cpu().item()),
        "d_max": float(d_max.detach().cpu().item()),
        "alt_max": int(max_alt_cls.detach().cpu().item()),
    }


def compute_sweep_stats(records, mode, tau):
    """Compute diagnostics for one threshold.

    mode:
        "mean" uses d_mean / alt_mean
        "max" uses d_max / alt_max
    """
    d_key = "d_mean" if mode == "mean" else "d_max"
    alt_key = "alt_mean" if mode == "mean" else "alt_max"

    n = len(records)
    if n == 0:
        return {}

    I_D = [r[d_key] > tau for r in records]
    I_H = [r["I_H"] for r in records]
    I_M = [r["I_M"] for r in records]
    W = [r["global_wrong"] for r in records]

    conflict_count = sum(I_D)
    conflict_wrong = sum(int(d and w) for d, w in zip(I_D, W))
    alt_correct = sum(int(d and r[alt_key] == r["target"]) for d, r in zip(I_D, records))

    conflict_boundary = [d and m for d, m in zip(I_D, I_M)]
    conflict_boundary_count = sum(conflict_boundary)
    conflict_boundary_wrong = sum(int(cb and w) for cb, w in zip(conflict_boundary, W))

    candidate_or = [h or m or d for h, m, d in zip(I_H, I_M, I_D)]
    candidate_or_count = sum(candidate_or)
    candidate_or_wrong = sum(int(c and w) for c, w in zip(candidate_or, W))

    candidate_safe = [h or (m and d) for h, m, d in zip(I_H, I_M, I_D)]
    candidate_safe_count = sum(candidate_safe)
    candidate_safe_wrong = sum(int(c and w) for c, w in zip(candidate_safe, W))

    return {
        "tau": tau,
        "conflict_rate": _safe_rate(conflict_count, n),
        "wrong_given_conflict": _safe_rate(conflict_wrong, conflict_count),
        "local_alt_correct_given_conflict": _safe_rate(alt_correct, conflict_count),

        "conflict_boundary_rate": _safe_rate(conflict_boundary_count, n),
        "wrong_given_conflict_boundary": _safe_rate(conflict_boundary_wrong, conflict_boundary_count),

        "candidate_or_rate": _safe_rate(candidate_or_count, n),
        "wrong_given_candidate_or": _safe_rate(candidate_or_wrong, candidate_or_count),

        "candidate_safe_rate": _safe_rate(candidate_safe_count, n),
        "wrong_given_candidate_safe": _safe_rate(candidate_safe_wrong, candidate_safe_count),
    }


def choose_best_threshold(sweep, key="wrong_given_conflict", min_rate=1.0):
    """Choose threshold with highest precision under non-trivial coverage."""
    best = None

    for item in sweep:
        precision = item.get(key)
        rate = item.get("conflict_rate", None)

        if precision is None or rate is None:
            continue

        if rate < min_rate:
            continue

        if best is None:
            best = item
        elif precision > best.get(key, -1):
            best = item
        elif precision == best.get(key, -1) and rate > best.get("conflict_rate", -1):
            best = item

    return best or {}


def choose_best_safe_threshold(sweep, min_rate=1.0):
    best = None

    for item in sweep:
        precision = item.get("wrong_given_candidate_safe")
        rate = item.get("candidate_safe_rate", None)

        if precision is None or rate is None:
            continue

        if rate < min_rate:
            continue

        if best is None:
            best = item
        elif precision > best.get("wrong_given_candidate_safe", -1):
            best = item
        elif precision == best.get("wrong_given_candidate_safe", -1) and rate > best.get("candidate_safe_rate", -1):
            best = item

    return best or {}


# ============================================================
# Main diagnostic loop
# ============================================================

@torch.no_grad()
def run_test_tda_diag_v2(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights):
    pos_enabled, neg_enabled = pos_cfg["enabled"], neg_cfg["enabled"]

    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ["shot_capacity", "alpha", "beta"]}
        pos_cache, pos_local_cache = build_cache_in_advance(
            args,
            test_loader,
            lm3d_model,
            clip_weights,
            pos_params["shot_capacity"],
        )
        print("len(pos_cache):", len(pos_cache))
        print("len(pos_local_cache):", len(pos_local_cache))
    else:
        pos_params = {}
        pos_cache, pos_local_cache = {}, {}

    if neg_enabled:
        neg_params = {
            k: neg_cfg[k]
            for k in ["shot_capacity", "alpha", "beta", "entropy_threshold", "mask_threshold"]
        }
    else:
        neg_params = {}

    neg_cache, neg_local_cache = {}, {}
    accuracies = []

    records = []

    final_correct = 0
    global_wrong_count = 0

    tau_l = neg_cfg["entropy_threshold"]["lower"]
    tau_u = neg_cfg["entropy_threshold"]["upper"]

    feasible_boundary = TAU_M > max(0.0, 2.0 * TAU_P - 1.0)

    print(f"[E4-DIAG-v2] TAU_M={TAU_M}, TAU_P={TAU_P}")
    print(f"[E4-DIAG-v2] TAU_D_LIST={TAU_D_LIST}")
    print(f"[E4-DIAG-v2] Boundary feasibility: {feasible_boundary}")

    if not feasible_boundary:
        print("[E4-DIAG-v2][WARNING] Boundary candidate may be empty under current thresholds.")

    for i, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()

        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(
            args, feature, lm3d_model, clip_weights
        )

        target = target.cuda()
        target_int = _to_int(target)
        global_pred = _to_int(pred)

        global_diag = compute_global_diagnostics(loss, prob_map, clip_weights)
        local_diag = compute_local_conflict_diagnostics(patch_centers, clip_weights, global_pred)

        entropy = global_diag["entropy"]
        p1 = global_diag["p1"]
        margin = global_diag["margin"]

        I_H = tau_l < entropy < tau_u
        I_M = (margin < TAU_M) and (p1 > TAU_P)

        global_wrong = global_pred != target_int
        global_wrong_count += int(global_wrong)

        records.append({
            "target": target_int,
            "global_pred": global_pred,
            "global_wrong": global_wrong,

            "entropy": entropy,
            "p1": p1,
            "margin": margin,

            "I_H": I_H,
            "I_M": I_M,

            "d_mean": local_diag["d_mean"],
            "alt_mean": local_diag["alt_mean"],
            "d_max": local_diag["d_max"],
            "alt_max": local_diag["alt_max"],
        })

        # ---------------- original cache update ----------------
        if pos_enabled:
            update_cache(
                pos_cache,
                pos_local_cache,
                pred,
                [pc_feats, patch_centers, loss],
                pos_params["shot_capacity"],
            )

        if (
            neg_enabled
            and neg_params["entropy_threshold"]["lower"]
            < entropy
            < neg_params["entropy_threshold"]["upper"]
        ):
            update_cache(
                neg_cache,
                neg_local_cache,
                pred,
                [pc_feats, None, loss, prob_map],
                neg_params["shot_capacity"],
                include_prob_map=True,
            )

        # ---------------- original final logits ----------------
        final_logits = clip_logits.clone()

        if pos_enabled and pos_cache:
            final_logits += compute_cache_logits(
                pc_feats,
                pos_cache,
                pos_params["alpha"],
                pos_params["beta"],
                clip_weights,
            )
            final_logits += compute_local_cache_logits(
                patch_centers,
                pos_local_cache,
                pos_params["alpha"],
                pos_params["beta"],
                clip_weights,
            )

        if neg_enabled and neg_cache:
            final_logits -= compute_cache_logits(
                pc_feats,
                neg_cache,
                neg_params["alpha"],
                neg_params["beta"],
                clip_weights,
                (
                    neg_params["mask_threshold"]["lower"],
                    neg_params["mask_threshold"]["upper"],
                ),
            )

        acc = cls_acc(final_logits, target)
        accuracies.append(acc)
        final_correct += int(torch.argmax(final_logits, dim=1).item() == target_int)

        if i % args.print_freq == 0:
            print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    n = len(records)
    final_acc = sum(accuracies) / len(accuracies)

    d_mean_values = [r["d_mean"] for r in records]
    d_max_values = [r["d_max"] for r in records]
    entropy_values = [r["entropy"] for r in records]
    margin_values = [r["margin"] for r in records]

    sweep_mean = [compute_sweep_stats(records, "mean", tau) for tau in TAU_D_LIST]
    sweep_max = [compute_sweep_stats(records, "max", tau) for tau in TAU_D_LIST]

    best_mean = choose_best_threshold(sweep_mean, key="wrong_given_conflict", min_rate=MIN_CONFLICT_RATE_FOR_BEST)
    best_max = choose_best_threshold(sweep_max, key="wrong_given_conflict", min_rate=MIN_CONFLICT_RATE_FOR_BEST)

    best_safe_mean = choose_best_safe_threshold(sweep_mean, min_rate=MIN_CONFLICT_RATE_FOR_BEST)
    best_safe_max = choose_best_safe_threshold(sweep_max, min_rate=MIN_CONFLICT_RATE_FOR_BEST)

    count_H = sum(int(r["I_H"]) for r in records)
    count_M = sum(int(r["I_M"]) for r in records)
    count_HM = sum(int(r["I_H"] and r["I_M"]) for r in records)

    summary = {
        "n": n,
        "accuracy": final_acc,
        "final_error_rate": _safe_rate(n - final_correct, n),
        "global_error_rate": _safe_rate(global_wrong_count, n),

        "tau_m": TAU_M,
        "tau_p": TAU_P,
        "boundary_feasible": feasible_boundary,

        "mean_entropy": sum(entropy_values) / max(n, 1),
        "mean_global_margin": sum(margin_values) / max(n, 1),
        "mean_d_mean": sum(d_mean_values) / max(n, 1),
        "mean_d_max": sum(d_max_values) / max(n, 1),

        "d_mean_p50": _percentile(d_mean_values, 50),
        "d_mean_p90": _percentile(d_mean_values, 90),
        "d_mean_p95": _percentile(d_mean_values, 95),
        "d_mean_p99": _percentile(d_mean_values, 99),
        "d_mean_p999": _percentile(d_mean_values, 99.9),

        "d_max_p50": _percentile(d_max_values, 50),
        "d_max_p90": _percentile(d_max_values, 90),
        "d_max_p95": _percentile(d_max_values, 95),
        "d_max_p99": _percentile(d_max_values, 99),
        "d_max_p999": _percentile(d_max_values, 99.9),

        "uncertain_rate": _safe_rate(count_H, n),
        "boundary_rate": _safe_rate(count_M, n),
        "overlap_HM_rate": _safe_rate(count_HM, n),

        "best_mean_tau": best_mean.get("tau"),
        "best_mean_conflict_rate": best_mean.get("conflict_rate"),
        "best_mean_wrong_given_conflict": best_mean.get("wrong_given_conflict"),
        "best_mean_local_alt_correct": best_mean.get("local_alt_correct_given_conflict"),

        "best_max_tau": best_max.get("tau"),
        "best_max_conflict_rate": best_max.get("conflict_rate"),
        "best_max_wrong_given_conflict": best_max.get("wrong_given_conflict"),
        "best_max_local_alt_correct": best_max.get("local_alt_correct_given_conflict"),

        "best_safe_mean_tau": best_safe_mean.get("tau"),
        "best_safe_mean_rate": best_safe_mean.get("candidate_safe_rate"),
        "best_safe_mean_wrong": best_safe_mean.get("wrong_given_candidate_safe"),

        "best_safe_max_tau": best_safe_max.get("tau"),
        "best_safe_max_rate": best_safe_max.get("candidate_safe_rate"),
        "best_safe_max_wrong": best_safe_max.get("wrong_given_candidate_safe"),

        "sweep_mean": sweep_mean,
        "sweep_max": sweep_max,
    }

    print("---- ***Final*** TDA's test accuracy: {:.2f}. ----\n".format(final_acc))
    print("E4_CANC_DIAG_V2_SUMMARY " + json.dumps(summary, sort_keys=True))

    out_path = os.environ.get("E4_DIAG_OUT", "")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"[E4-DIAG-v2] Summary JSON saved to: {out_path}")

    return final_acc, summary


# ============================================================
# Entry
# ============================================================

def main():
    args = get_arguments()
    set_random_seed(args.seed)

    config_path = args.config

    clip_model, lm3d_model = load_models(args)
    preprocess = None

    dataset_name = args.dataset
    print(f"Processing {dataset_name} dataset.")

    cfg = get_config_file(args, config_path, dataset_name)
    print("\nRunning dataset configurations:")
    print(cfg, "\n")

    test_loader, classnames, template = build_test_data_loader(
        args, dataset_name, args.data_root, preprocess
    )

    print(f">>> classnames:", classnames)

    clip_weights = clip_classifier(args, classnames, template, clip_model)

    run_test_tda_diag_v2(args, cfg["positive"], cfg["negative"], test_loader, lm3d_model, clip_weights)


if __name__ == "__main__":
    main()
