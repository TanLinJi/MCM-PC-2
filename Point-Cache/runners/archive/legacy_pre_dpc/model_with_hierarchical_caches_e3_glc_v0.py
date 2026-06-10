'''Hierarchical caches: global + local.

Stage 3 modification:
Reliability Score v2 with Global-Local Consistency for positive cache admission.

Original Point-Cache mainly keeps low-entropy samples in the positive cache.
This version keeps the hierarchical cache pipeline unchanged, but ranks and
updates positive cache entries by a reliability score:

    R(x) = -H(x) + lambda_margin * M(x) + lambda_glc * C_gl(x)

where H(x) is the normalized entropy, M(x) is the top-1 / top-2
probability margin, and C_gl(x) measures whether local part evidence
supports the global prediction.

Negative cache keeps the original entropy / probability-map based logic.
'''

import os
import sys
import wandb
import operator

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *


# ============================================================
# Stage 2: Reliability Score v1 utilities
# ============================================================

@torch.no_grad()
def compute_prediction_margin(prob_map):
    """Compute the top-1 minus top-2 probability margin.

    Args:
        prob_map: Tensor of shape (B, C), usually B = 1.

    Returns:
        margin: Python float. Larger margin means the model is less confused
                between the best and the second-best classes.
    """
    top2_probs = torch.topk(prob_map, k=2, dim=1).values
    margin = top2_probs[:, 0] - top2_probs[:, 1]
    return margin.item()


@torch.no_grad()
def compute_global_local_consistency(patch_centers, clip_weights, pred):
    """Compute global-local consistency score.

    The global branch predicts class `pred`. We then ask whether local
    part features also support this predicted class.

    Implementation:
        1. Compute local part logits by comparing each part feature with
           text prototypes.
        2. Convert local logits to probabilities.
        3. Average local probabilities over parts.
        4. Use the averaged local probability of the global predicted class
           as the consistency score.

    Args:
        patch_centers: Tensor of shape (n_cluster, emb_dim).
        clip_weights: Tensor of shape (emb_dim, num_classes).
        pred: Global predicted class index.

    Returns:
        glc_score: Python float in [0, 1]. Higher means local evidence agrees
                   more strongly with the global prediction.
    """
    local_logits = patch_centers @ clip_weights
    local_probs = torch.softmax(local_logits.float(), dim=-1)
    local_probs = local_probs.mean(dim=0)

    if torch.is_tensor(pred):
        pred = int(pred.item())
    else:
        pred = int(pred)

    glc_score = local_probs[pred]
    return glc_score.item()


@torch.no_grad()
def compute_reliability_score(
    loss,
    prob_map,
    clip_weights,
    patch_centers=None,
    pred=None,
    lambda_margin=1.0,
    lambda_glc=1.0,
):
    """Compute reliability score for positive cache admission.

    Original Point-Cache uses entropy as the main quality criterion.
    E2-EMR additionally considers the top-1/top-2 classification margin.
    E3-GLC further adds a global-local consistency term:

        R(x) = -H(x) + lambda_margin * M(x) + lambda_glc * C_gl(x)

    where:
        H(x): normalized entropy returned by get_entropy(loss, clip_weights)
        M(x): top-1 probability - top-2 probability
        C_gl(x): local evidence assigned to the global predicted class

    Args:
        loss: Entropy-like scalar returned by get_logits().
        prob_map: Class probability map of shape (B, C).
        clip_weights: Text classifier weights.
        patch_centers: Local part features of shape (n_cluster, emb_dim).
        pred: Global predicted class index.
        lambda_margin: Weight for margin term.
        lambda_glc: Weight for global-local consistency term.

    Returns:
        reliability: Python float. Higher is better.
        entropy: Python float.
        margin: Python float.
        glc_score: Python float.
    """
    entropy = get_entropy(loss, clip_weights)
    if torch.is_tensor(entropy):
        entropy = entropy.item()

    margin = compute_prediction_margin(prob_map)

    if patch_centers is not None and pred is not None:
        glc_score = compute_global_local_consistency(patch_centers, clip_weights, pred)
    else:
        glc_score = 0.0

    reliability = -entropy + lambda_margin * margin + lambda_glc * glc_score

    return reliability, entropy, margin, glc_score


# ============================================================
# Cache construction and update
# ============================================================

@torch.no_grad()
def build_cache_in_advance(args, test_loader, lm3d_model, clip_weights, shot_capacity, include_prob_map=False):
    """Build initial cache before online test-time adaptation.

    For positive cache:
        include_prob_map = False
        cache item = [feature, loss, reliability, entropy, margin]
        local item = [patch_centers, loss, reliability, entropy, margin]
        Sorting key = reliability, descending.

    For negative cache:
        include_prob_map = True
        cache item = [feature, loss, prob_map]
        Sorting key = loss / entropy, ascending.

    In the current hierarchical pipeline, this function is mainly used to build
    positive global/local caches in advance. The negative cache is updated
    online in run_test_tda().
    """
    if include_prob_map:
        print('*' * 10, 'Building [global] neg. cache ...', '*' * 10, '\n')
    else:
        print('*' * 10, 'Building [global] and [local] pos. cache ...', '*' * 10, '\n')

    cache, local_cache = {}, {}

    for pc, _, _, rgb in test_loader:
        # pc:  (1, n, 3)
        # rgb: (1, n, 3)
        feature = torch.cat([pc, rgb], dim=-1).half()

        # pc_feats:      (1, emb_dim)
        # patch_centers: (n_cluster, emb_dim)
        # clip_logits:   (1, n_cls)
        # loss:          scalar
        # prob_map:      (1, n_cls)
        # pred:          scalar class index
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(
            args, feature, lm3d_model, clip_weights
        )

        if include_prob_map:
            # Keep original negative-cache format.
            item = [pc_feats, loss, prob_map]
            local_item = [patch_centers, loss, prob_map]
        else:
            # Stage 3 change: positive cache uses reliability score.
            reliability, entropy, margin, glc_score = compute_reliability_score(
                loss,
                prob_map,
                clip_weights,
                patch_centers=patch_centers,
                pred=pred,
                lambda_margin=1.0,
                lambda_glc=1.0,
            )
            item = [pc_feats, loss, reliability, entropy, margin, glc_score]
            local_item = [patch_centers, loss, reliability, entropy, margin, glc_score]

        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
                local_cache[pred].append(local_item)
            else:
                if include_prob_map:
                    # Original rule for negative cache: lower entropy is better.
                    if loss < cache[pred][-1][1]:
                        cache[pred][-1] = item
                        local_cache[pred][-1] = local_item
                else:
                    # New rule for positive cache: higher reliability is better.
                    if reliability > cache[pred][-1][2]:
                        cache[pred][-1] = item
                        local_cache[pred][-1] = local_item
        else:
            cache[pred] = [item]
            local_cache[pred] = [local_item]

        if include_prob_map:
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
            local_cache[pred] = sorted(local_cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(2), reverse=True)
            local_cache[pred] = sorted(local_cache[pred], key=operator.itemgetter(2), reverse=True)

        cache_num = sum(len(cache[key]) for key in cache)
        num_classes = clip_logits.size(1)
        full_num = shot_capacity * num_classes

        if cache_num == full_num:
            if include_prob_map:
                print('*' * 10, 'Building [global] neg. cache is Done!', '*' * 10, '\n')
            else:
                print('*' * 10, 'Building [global] and [local] pos. cache is Done!', '*' * 10, '\n')
            break

    return cache, local_cache


@torch.no_grad()
def update_cache(cache, local_cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache during online test-time adaptation.

    Positive cache input:
        features_loss = [pc_feats, patch_centers, loss, reliability, entropy, margin, glc_score]
        cache item    = [pc_feats, loss, reliability, entropy, margin, glc_score]
        local item    = [patch_centers, loss, reliability, entropy, margin, glc_score]

    Negative cache input:
        features_loss = [pc_feats, None, loss, prob_map]
        cache item    = [pc_feats, loss, prob_map]

    For positive cache, the worst item is the one with the lowest reliability.
    For negative cache, the original entropy-based ordering is preserved.
    """
    if include_prob_map:
        # Negative cache keeps original format.
        item = [features_loss[0], features_loss[2], features_loss[3]]
    else:
        item = [
            features_loss[0],  # pc_feats
            features_loss[2],  # loss
            features_loss[3],  # reliability
            features_loss[4],  # entropy
            features_loss[5],  # margin
            features_loss[6],  # glc_score
            features_loss[6],  # glc_score
        ]
        local_item = [
            features_loss[1],  # patch_centers
            features_loss[2],  # loss
            features_loss[3],  # reliability
            features_loss[4],  # entropy
            features_loss[5],  # margin
            features_loss[6],  # glc_score
        ]

    if pred in cache:
        if len(cache[pred]) < shot_capacity:
            cache[pred].append(item)
            if not include_prob_map:
                local_cache[pred].append(local_item)
        else:
            if include_prob_map:
                current_loss = features_loss[2]
                worst_loss = cache[pred][-1][1]

                if current_loss < worst_loss:
                    cache[pred][-1] = item
            else:
                current_reliability = features_loss[3]
                worst_reliability = cache[pred][-1][2]

                if current_reliability > worst_reliability:
                    cache[pred][-1] = item
                    local_cache[pred][-1] = local_item

        if include_prob_map:
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(2), reverse=True)
            local_cache[pred] = sorted(local_cache[pred], key=operator.itemgetter(2), reverse=True)
    else:
        cache[pred] = [item]
        if not include_prob_map:
            local_cache[pred] = [local_item]


# ============================================================
# Cache retrieval logits
# ============================================================

@torch.no_grad()
def compute_cache_logits(pc_feats, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using global positive or negative cache."""
    cache_keys = []
    cache_values = []

    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            cache_keys.append(item[0])
            if neg_mask_thresholds:
                # Negative cache: item[2] is prob_map.
                cache_values.append(item[2])
            else:
                # Positive cache: value is the pseudo class label.
                cache_values.append(class_index)

    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)

    if neg_mask_thresholds:
        cache_values = torch.cat(cache_values, dim=0)
        cache_values = ((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).half().cuda()
    else:
        cache_values = F.one_hot(
            torch.Tensor(cache_values).to(torch.int64),
            num_classes=clip_weights.size(1)
        ).half().cuda()

    affinity = pc_feats @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits


@torch.no_grad()
def compute_local_cache_logits(patch_centers, local_cache, alpha, beta, clip_weights):
    """Compute logits using local positive cache."""
    local_cache_keys = []
    local_cache_values = []

    for class_index in sorted(local_cache.keys()):
        for item in local_cache[class_index]:
            local_cache_keys.append(item[0])
            n_cluster = item[0].shape[0]
            local_cache_values.append([class_index] * n_cluster)

    local_cache_keys = torch.cat(local_cache_keys, dim=0).permute(1, 0)

    local_cache_values = F.one_hot(
        torch.Tensor(local_cache_values).to(torch.int64),
        num_classes=clip_weights.size(1)
    ).half().cuda()
    local_cache_values = local_cache_values.view(-1, clip_weights.size(1))

    affinity = patch_centers.mean(dim=0, keepdim=True) @ local_cache_keys
    local_cache_logits = ((-1) * (beta - beta * affinity)).exp() @ local_cache_values
    return alpha * local_cache_logits


# ============================================================
# Main test-time adaptation loop
# ============================================================

@torch.no_grad()
def run_test_tda(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights):
    """Run hierarchical cache-based test-time adaptation.

    Stage 3 changes are restricted to positive cache admission and update.
    The final prediction formula is unchanged from Point-Cache:

        final_logits = zero-shot logits
                     + global positive cache logits
                     + local positive cache logits
                     - negative cache logits
    """
    pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']

    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        pos_cache, pos_local_cache = build_cache_in_advance(
            args,
            test_loader,
            lm3d_model,
            clip_weights,
            pos_params['shot_capacity']
        )
        print('len(pos_cache):', len(pos_cache))
        print('len(pos_local_cache):', len(pos_local_cache))
    else:
        pos_params = {}
        pos_cache, pos_local_cache = {}, {}

    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}
    else:
        neg_params = {}

    neg_cache, neg_local_cache = {}, {}
    accuracies = []

    for i, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()

        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(
            args, feature, lm3d_model, clip_weights
        )

        target = target.cuda()
        prop_entropy = get_entropy(loss, clip_weights)
        if torch.is_tensor(prop_entropy):
            prop_entropy_value = prop_entropy.item()
        else:
            prop_entropy_value = prop_entropy

        if pos_enabled:
            reliability, entropy, margin, glc_score = compute_reliability_score(
                loss,
                prob_map,
                clip_weights,
                patch_centers=patch_centers,
                pred=pred,
                lambda_margin=1.0,
                lambda_glc=1.0,
            )
            update_cache(
                pos_cache,
                pos_local_cache,
                pred,
                [pc_feats, patch_centers, loss, reliability, entropy, margin, glc_score],
                pos_params['shot_capacity']
            )

        if (
            neg_enabled
            and neg_params['entropy_threshold']['lower'] < prop_entropy_value < neg_params['entropy_threshold']['upper']
        ):
            update_cache(
                neg_cache,
                neg_local_cache,
                pred,
                [pc_feats, None, loss, prob_map],
                neg_params['shot_capacity'],
                include_prob_map=True
            )

        final_logits = clip_logits.clone()

        if pos_enabled and pos_cache:
            final_logits += compute_cache_logits(
                pc_feats,
                pos_cache,
                pos_params['alpha'],
                pos_params['beta'],
                clip_weights
            )
            final_logits += compute_local_cache_logits(
                patch_centers,
                pos_local_cache,
                pos_params['alpha'],
                pos_params['beta'],
                clip_weights
            )

        if neg_enabled and neg_cache:
            final_logits -= compute_cache_logits(
                pc_feats,
                neg_cache,
                neg_params['alpha'],
                neg_params['beta'],
                clip_weights,
                (
                    neg_params['mask_threshold']['lower'],
                    neg_params['mask_threshold']['upper']
                )
            )

        acc = cls_acc(final_logits, target)
        accuracies.append(acc)

        if args.wandb:
            wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

        if i % args.print_freq == 0:
            print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    print("---- ***Final*** TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))
    return sum(accuracies) / len(accuracies)


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

    print(f'>>> classnames:', classnames)

    clip_weights = clip_classifier(args, classnames, template, clip_model)

    if args.wandb:
        if args.lm3d == 'openshape':
            prefix = f"[test-manual-prompts]/{args.cache_type}_cache/{args.lm3d}-{args.oshape_version}"
        elif args.lm3d == 'ulip':
            prefix = f"[test-manual-prompts]/{args.cache_type}_cache/{args.ulip_version}"
        else:
            prefix = f"[test-manual-prompts]/{args.cache_type}_cache/{args.lm3d}"

        if '_c' in dataset_name and 'sonn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}/{args.cor_type}"
        elif '_c' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}/{args.cor_type}"
        elif 'scanobjnn' in dataset_name or 'scanobjectnn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}"
        elif 'sim2real_sonn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sim2real_type}-{args.npoints}"
        elif 'pointda' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}"
        else:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}"

        run = wandb.init(project="Point-TDA", config=cfg, name=run_name)

    acc = run_test_tda(args, cfg['positive'], cfg['negative'], test_loader, lm3d_model, clip_weights)

    if args.wandb:
        wandb.log({f"{dataset_name}": acc})
        run.finish()


if __name__ == "__main__":
    main()