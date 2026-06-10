"""E4-CANC-v1: Conservative Conflict-Aware Negative Cache.

This runner modifies only the negative-cache admission rule.

Original Point-Cache negative cache:
    I_neg = I_H

E4-CANC-v1:
    I_neg = I_H OR (I_D AND p_g_top1 > tau_p)

where:
    I_H: original entropy-based negative candidate
    I_D: mean-local global-local conflict candidate
    p_g_top1 > tau_p: global prediction is not completely low-confidence

Important:
    The local alternative class is NOT used as a pseudo-label.
    Conflict is used only to suppress the global predicted class.
"""

import os
import sys
import json
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
# E4-CANC-v1 hyperparameters
# ============================================================

TAU_D = 5e-5
TAU_M = 0.20
TAU_P = 0.40


# ============================================================
# Utilities
# ============================================================

def _to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def _to_int(x):
    if torch.is_tensor(x):
        return int(x.detach().cpu().item())
    return int(x)


def _safe_rate(num, den):
    if den == 0:
        return None
    return 100.0 * float(num) / float(den)


@torch.no_grad()
def compute_global_stats(loss, prob_map, clip_weights):
    """Compute entropy, top-1 probability, and top-1/top-2 margin."""
    entropy = get_entropy(loss, clip_weights)
    entropy = _to_float(entropy)

    probs = prob_map.float()
    top2 = torch.topk(probs, k=2, dim=1).values[0]

    p1 = float(top2[0].detach().cpu().item())
    p2 = float(top2[1].detach().cpu().item())
    margin = p1 - p2

    return entropy, p1, margin


@torch.no_grad()
def compute_mean_local_conflict(patch_centers, clip_weights, global_pred):
    """Compute mean-local global-local disagreement score.

    D_gl_mean(x) =
        [ max_{k != y_g} p_l_mean(k) - p_l_mean(y_g) ]_+

    Args:
        patch_centers: Tensor of shape (n_cluster, emb_dim).
        clip_weights: Tensor of shape (emb_dim, num_classes).
        global_pred: Global predicted class index.

    Returns:
        d_mean: Python float.
        alt_cls: Mean-local alternative class index.
    """
    global_pred = int(global_pred)

    local_logits = patch_centers @ clip_weights
    local_probs_parts = torch.softmax(local_logits.float(), dim=-1)
    mean_probs = local_probs_parts.mean(dim=0)

    other_probs = mean_probs.clone()
    other_probs[global_pred] = -1.0

    alt_prob, alt_cls = torch.max(other_probs, dim=0)
    global_prob = mean_probs[global_pred]

    d_mean = torch.clamp(alt_prob - global_prob, min=0.0)

    return float(d_mean.detach().cpu().item()), int(alt_cls.detach().cpu().item())


# ============================================================
# Main E4-CANC-v1 loop
# ============================================================

@torch.no_grad()
def run_test_tda_e4_canc_v1(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights):
    """Run hierarchical cache with E4 conservative negative-cache admission."""

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

    tau_l = neg_cfg["entropy_threshold"]["lower"]
    tau_u = neg_cfg["entropy_threshold"]["upper"]

    feasible_boundary = TAU_M > max(0.0, 2.0 * TAU_P - 1.0)
    print(f"[E4-CANC-v1] TAU_D={TAU_D}, TAU_M={TAU_M}, TAU_P={TAU_P}")
    print(f"[E4-CANC-v1] Boundary feasibility: {feasible_boundary}")

    if not feasible_boundary:
        print("[E4-CANC-v1][WARNING] Boundary candidate may be empty under current thresholds.")

    # Diagnostics
    n = 0
    final_correct = 0
    base_neg_count = 0
    e4_extra_count = 0
    e4_total_count = 0
    conflict_count = 0
    boundary_count = 0
    conflict_confident_count = 0

    for i, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()

        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(
            args, feature, lm3d_model, clip_weights
        )

        target = target.cuda()
        global_pred = _to_int(pred)

        entropy, p1, margin = compute_global_stats(loss, prob_map, clip_weights)
        d_mean, alt_cls = compute_mean_local_conflict(patch_centers, clip_weights, global_pred)

        I_H = tau_l < entropy < tau_u
        I_M = (margin < TAU_M) and (p1 > TAU_P)
        I_D = d_mean > TAU_D

        I_E4_extra = I_D and (p1 > TAU_P)
        I_neg_e4 = I_H or I_E4_extra

        n += 1
        base_neg_count += int(I_H)
        boundary_count += int(I_M)  # kept for diagnostics only
        conflict_count += int(I_D)
        conflict_confident_count += int(I_E4_extra)
        e4_extra_count += int((not I_H) and I_E4_extra)
        e4_total_count += int(I_neg_e4)

        # ---------------- positive cache: unchanged ----------------
        if pos_enabled:
            update_cache(
                pos_cache,
                pos_local_cache,
                pred,
                [pc_feats, patch_centers, loss],
                pos_params["shot_capacity"],
            )

        # ---------------- negative cache: E4-CANC-v1 change ----------------
        if neg_enabled and I_neg_e4:
            update_cache(
                neg_cache,
                neg_local_cache,
                pred,
                [pc_feats, None, loss, prob_map],
                neg_params["shot_capacity"],
                include_prob_map=True,
            )

        # ---------------- final logits: unchanged ----------------
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
        final_correct += int(torch.argmax(final_logits, dim=1).item() == int(target.detach().cpu().item()))

        if i % args.print_freq == 0:
            print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    final_acc = sum(accuracies) / len(accuracies)

    summary = {
        "n": n,
        "accuracy": final_acc,
        "final_error_rate": _safe_rate(n - final_correct, n),

        "tau_d": TAU_D,
        "tau_m": TAU_M,
        "tau_p": TAU_P,
        "boundary_feasible": feasible_boundary,

        "base_negative_rate": _safe_rate(base_neg_count, n),
        "e4_total_negative_rate": _safe_rate(e4_total_count, n),
        "e4_extra_negative_rate": _safe_rate(e4_extra_count, n),

        "boundary_rate": _safe_rate(boundary_count, n),
        "conflict_rate": _safe_rate(conflict_count, n),
        "conflict_confident_rate": _safe_rate(conflict_confident_count, n),
    }

    print("---- ***Final*** TDA's test accuracy: {:.2f}. ----\n".format(final_acc))
    print("E4_CANC_V1_SUMMARY " + json.dumps(summary, sort_keys=True))

    out_path = os.environ.get("E4_CANC_OUT", "")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"[E4-CANC-v1] Summary JSON saved to: {out_path}")

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

    run_test_tda_e4_canc_v1(
        args,
        cfg["positive"],
        cfg["negative"],
        test_loader,
        lm3d_model,
        clip_weights,
    )


if __name__ == "__main__":
    main()
