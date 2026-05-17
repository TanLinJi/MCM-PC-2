"""E4-CANC-DIAG: Global-Local Conflict Diagnosis.

This runner does NOT change Point-Cache prediction, positive cache, negative
cache, or final logits. It reproduces the original hierarchical cache pipeline
and only records diagnostic statistics for global-local conflict.

Goal:
    Validate whether global-local conflict is a useful signal for detecting
    wrong global pseudo-labels before implementing Confusion-Aware Negative Cache.

Key diagnostics:
    P(global wrong)
    P(global wrong | conflict)
    P(global wrong | conflict and boundary)
    P(local alternative correct | conflict)
"""

import os
import sys
import json
import math
import wandb

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *

# Import original Point-Cache hierarchical-cache operators.
# This keeps the prediction pipeline identical to the baseline runner.
from model_with_hierarchical_caches import (
    build_cache_in_advance,
    update_cache,
    compute_cache_logits,
    compute_local_cache_logits,
)


# ============================================================
# Diagnostic thresholds
# ============================================================

TAU_D = 0.05   # conflict threshold for D_gl
TAU_M = 0.20   # small global margin threshold
TAU_P = 0.40   # top-1 probability must be above this for boundary candidate

# Original negative-cache entropy interval is read from config:
# neg_cfg['entropy_threshold']['lower'] and ['upper']


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


@torch.no_grad()
def compute_global_diagnostics(loss, prob_map, clip_weights):
    """Compute global entropy, top probabilities, and global margin."""
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
    """Compute mean-local and max-part global-local disagreement.

    Mean-local disagreement:
        D_mean = [ max_{k != y_g} p_l_mean(k) - p_l_mean(y_g) ]_+

    Max-part disagreement:
        D_max = max_r [ max_{k != y_g} p_{l,r}(k) - p_{l,r}(y_g) ]_+

    Returns:
        mean conflict score, mean alternative class,
        max-part conflict score, max-part alternative class.
    """
    global_pred = int(global_pred)

    local_logits = patch_centers @ clip_weights
    local_probs_parts = torch.softmax(local_logits.float(), dim=-1)

    # ---------- mean-local version ----------
    mean_probs = local_probs_parts.mean(dim=0)

    mean_other_probs = mean_probs.clone()
    mean_other_probs[global_pred] = -1.0

    mean_alt_prob, mean_alt_cls = torch.max(mean_other_probs, dim=0)
    mean_global_prob = mean_probs[global_pred]
    d_mean = torch.clamp(mean_alt_prob - mean_global_prob, min=0.0)

    # ---------- max-part version ----------
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


# ============================================================
# Main diagnostic loop
# ============================================================

@torch.no_grad()
def run_test_tda_diag(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights):
    """Run original hierarchical cache pipeline and collect E4 diagnostics."""

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

    # ---------- diagnostic counters ----------
    n = 0
    final_correct = 0
    global_wrong = 0

    count_H = 0
    count_M = 0
    count_D_mean = 0
    count_D_max = 0

    count_HM = 0
    count_HD_mean = 0
    count_MD_mean = 0
    count_HMD_mean = 0

    count_HD_max = 0
    count_MD_max = 0
    count_HMD_max = 0

    count_or_mean = 0
    count_safe_mean = 0
    count_or_max = 0
    count_safe_max = 0

    wrong_D_mean = 0
    wrong_D_max = 0
    wrong_MD_mean = 0
    wrong_MD_max = 0

    local_alt_correct_D_mean = 0
    local_alt_correct_D_max = 0

    # For checking average diagnostic magnitudes
    sum_entropy = 0.0
    sum_margin = 0.0
    sum_d_mean = 0.0
    sum_d_max = 0.0

    tau_l = neg_cfg["entropy_threshold"]["lower"]
    tau_u = neg_cfg["entropy_threshold"]["upper"]

    # Feasibility check for boundary condition:
    # M_g < TAU_M and p1 > TAU_P is non-empty only if TAU_M > 2*TAU_P - 1.
    feasible_boundary = TAU_M > max(0.0, 2.0 * TAU_P - 1.0)
    print(f"[E4-DIAG] TAU_D={TAU_D}, TAU_M={TAU_M}, TAU_P={TAU_P}")
    print(f"[E4-DIAG] Boundary feasibility: {feasible_boundary}")

    if not feasible_boundary:
        print("[E4-DIAG][WARNING] Boundary candidate may be empty under current thresholds.")

    for i, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()

        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(
            args, feature, lm3d_model, clip_weights
        )

        target = target.cuda()
        target_int = _to_int(target)
        global_pred = _to_int(pred)

        # ---------------- diagnostics before cache logits ----------------
        global_diag = compute_global_diagnostics(loss, prob_map, clip_weights)
        local_diag = compute_local_conflict_diagnostics(patch_centers, clip_weights, global_pred)

        entropy = global_diag["entropy"]
        p1 = global_diag["p1"]
        margin = global_diag["margin"]

        d_mean = local_diag["d_mean"]
        d_max = local_diag["d_max"]
        alt_mean = local_diag["alt_mean"]
        alt_max = local_diag["alt_max"]

        I_H = (tau_l < entropy < tau_u)
        I_M = (margin < TAU_M) and (p1 > TAU_P)
        I_D_mean = d_mean > TAU_D
        I_D_max = d_max > TAU_D

        W = global_pred != target_int

        n += 1
        global_wrong += int(W)

        count_H += int(I_H)
        count_M += int(I_M)
        count_D_mean += int(I_D_mean)
        count_D_max += int(I_D_max)

        count_HM += int(I_H and I_M)

        count_HD_mean += int(I_H and I_D_mean)
        count_MD_mean += int(I_M and I_D_mean)
        count_HMD_mean += int(I_H and I_M and I_D_mean)

        count_HD_max += int(I_H and I_D_max)
        count_MD_max += int(I_M and I_D_max)
        count_HMD_max += int(I_H and I_M and I_D_max)

        count_or_mean += int(I_H or I_M or I_D_mean)
        count_safe_mean += int(I_H or (I_M and I_D_mean))

        count_or_max += int(I_H or I_M or I_D_max)
        count_safe_max += int(I_H or (I_M and I_D_max))

        wrong_D_mean += int(I_D_mean and W)
        wrong_D_max += int(I_D_max and W)
        wrong_MD_mean += int(I_M and I_D_mean and W)
        wrong_MD_max += int(I_M and I_D_max and W)

        local_alt_correct_D_mean += int(I_D_mean and alt_mean == target_int)
        local_alt_correct_D_max += int(I_D_max and alt_max == target_int)

        sum_entropy += entropy
        sum_margin += margin
        sum_d_mean += d_mean
        sum_d_max += d_max

        # ---------------- original negative cache update ----------------
        prop_entropy_value = entropy

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
            < prop_entropy_value
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

    final_acc = sum(accuracies) / len(accuracies)

    summary = {
        "n": n,
        "accuracy": final_acc,
        "final_error_rate": _safe_rate(n - final_correct, n),
        "global_error_rate": _safe_rate(global_wrong, n),

        "tau_d": TAU_D,
        "tau_m": TAU_M,
        "tau_p": TAU_P,
        "boundary_feasible": feasible_boundary,

        "mean_entropy": sum_entropy / max(n, 1),
        "mean_global_margin": sum_margin / max(n, 1),
        "mean_d_mean": sum_d_mean / max(n, 1),
        "mean_d_max": sum_d_max / max(n, 1),

        "uncertain_rate": _safe_rate(count_H, n),
        "boundary_rate": _safe_rate(count_M, n),

        "conflict_mean_rate": _safe_rate(count_D_mean, n),
        "conflict_max_rate": _safe_rate(count_D_max, n),

        "wrong_given_conflict_mean": _safe_rate(wrong_D_mean, count_D_mean),
        "wrong_given_conflict_max": _safe_rate(wrong_D_max, count_D_max),

        "wrong_given_safe_mean": _safe_rate(wrong_MD_mean, count_MD_mean),
        "wrong_given_safe_max": _safe_rate(wrong_MD_max, count_MD_max),

        "local_alt_correct_given_conflict_mean": _safe_rate(local_alt_correct_D_mean, count_D_mean),
        "local_alt_correct_given_conflict_max": _safe_rate(local_alt_correct_D_max, count_D_max),

        "candidate_or_mean_rate": _safe_rate(count_or_mean, n),
        "candidate_safe_mean_rate": _safe_rate(count_safe_mean, n),
        "candidate_or_max_rate": _safe_rate(count_or_max, n),
        "candidate_safe_max_rate": _safe_rate(count_safe_max, n),

        "overlap_HM_rate": _safe_rate(count_HM, n),
        "overlap_HD_mean_rate": _safe_rate(count_HD_mean, n),
        "overlap_MD_mean_rate": _safe_rate(count_MD_mean, n),
        "overlap_HMD_mean_rate": _safe_rate(count_HMD_mean, n),
        "overlap_HD_max_rate": _safe_rate(count_HD_max, n),
        "overlap_MD_max_rate": _safe_rate(count_MD_max, n),
        "overlap_HMD_max_rate": _safe_rate(count_HMD_max, n),
    }

    print("---- ***Final*** TDA's test accuracy: {:.2f}. ----\n".format(final_acc))
    print("E4_CANC_DIAG_SUMMARY " + json.dumps(summary, sort_keys=True))

    out_path = os.environ.get("E4_DIAG_OUT", "")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"[E4-DIAG] Summary JSON saved to: {out_path}")

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

    # Do not initialize wandb in diagnostic runner.
    run_test_tda_diag(args, cfg["positive"], cfg["negative"], test_loader, lm3d_model, clip_weights)


if __name__ == "__main__":
    main()
