"""
MCP3D: Multi-Cache Enhanced Prototype Learning for Test-Time Adaptation
       of 3D Point Cloud Vision-Language Models.

Adapted from ICCV25-MCP (https://github.com/CenturyChen/MCP).
Replaces the 2D CLIP image encoder with Point-Cache's 3D encoders
(ULIP, ULIP-2, OpenShape, Uni3D) while preserving the multi-cache
architecture (entropy cache, align cache, negative cache) and the
MCP++ residual learning framework.
"""

import os
import sys
import random
import math
import argparse
import operator
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# --- Point-Cache imports (3D model loading, data loading, text classifier) ---
_PC_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'Point-Cache')
sys.path.insert(0, _PC_ROOT)
from utils.utils import (
    set_random_seed,
    get_entropy,
    softmax_entropy,
    avg_entropy,
    cls_acc,
    load_models,
    get_logits,
    clip_classifier,
    build_test_data_loader,
)

# For MCP config file reading (different from Point-Cache's get_config_file)
import yaml

# --- Reuse ICCV25-MCP's util tools (AverageMeter, ProgressMeter, accuracy) ---
_MCP_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _MCP_ROOT)
from util.tools import AverageMeter, ProgressMeter, Summary, accuracy

# ---------------------------------------------------------------------------
# Cache update functions (adapted from ICCV25-MCP, feature-format agnostic)
# ---------------------------------------------------------------------------

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update entropy / negative cache with new features and loss per class.

    Args:
        cache: dict[int, list[tuple]], class_idx -> list of (feature, loss, [prob_map])
        pred: int, predicted class index
        features_loss: list containing [feature_tensor, loss_scalar, (optional prob_map)]
        shot_capacity: int, max number of cached items per class
        include_prob_map: bool, whether the third element (prob_map) is present
    """
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def update_align_cache(align_cache, pred, features_loss, shot_capacity, cen, res_text_feat, all_classes, cache_keys=None):
    """Update align cache with samples closest to their class center (low entropy)."""
    with torch.no_grad():
        text_feat = res_text_feat[:, pred]
        class_center = update_class_center(cen, pred, text_feat, cache_keys, all_classes)
        feature = features_loss[0][:1]
        curr_entropy = features_loss[1]
        feat_dist = torch.norm(feature - class_center)
        if pred not in align_cache:
            align_cache[pred] = []
        if len(align_cache[pred]) < shot_capacity:
            align_cache[pred].append((feature, curr_entropy))
            align_cache[pred].sort(key=lambda x: float(x[1]))
            return
        worst_feat, worst_ent = align_cache[pred][-1]
        if curr_entropy < worst_ent:
            worst_dist = torch.norm(worst_feat - class_center)
            if feat_dist < worst_dist:
                align_cache[pred][-1] = (feature, curr_entropy)
                align_cache[pred].sort(key=lambda x: float(x[1]))


def update_class_center(center, pseudo_label, text_center, cache_keys, all_classes):
    """Compute updated class center combining text and cached visual features."""
    if pseudo_label in all_classes and cache_keys is not None:
        index = all_classes.index(pseudo_label)
        existing_class_center = cache_keys[:, index]
        new_class_center = center * existing_class_center + (1 - center) * text_center
    else:
        new_class_center = text_center
    return new_class_center


def update_cache_joint(cache_memory, cache_keys, ent_cache, align_cache, ent_pred, align_pred):
    """Merge entropy cache and align cache into unified cache memory."""
    total_shot = cache_memory.size(1)
    update_classes = set()
    if ent_pred is not None:
        update_classes.add(int(ent_pred))
    if align_pred is not None:
        update_classes.add(int(align_pred))

    for cls in update_classes:
        write_idx = 0
        if align_cache and cls in align_cache:
            for (feature, _) in align_cache[cls]:
                if write_idx >= total_shot:
                    break
                if feature.dim() == 2:
                    feature = feature.squeeze(0)
                cache_memory[cls, write_idx, :] = feature
                write_idx += 1
        if ent_cache and cls in ent_cache and write_idx < total_shot:
            for (feature, _) in ent_cache[cls]:
                if write_idx >= total_shot:
                    break
                if feature.dim() == 2:
                    feature = feature.squeeze(0)
                cache_memory[cls, write_idx, :] = feature
                write_idx += 1

        if write_idx < total_shot:
            cache_memory[cls, write_idx:, :].zero_()
        if write_idx > 0:
            new_proto = cache_memory[cls, :write_idx, :].mean(dim=0)
            cache_keys[:, cls] = new_proto


def shrink_cache_keys_and_values(cache_keys):
    """Prune empty cache slots, return active keys / values / class indices."""
    num_classes = cache_keys.size(1)
    nonzero_mask = cache_keys.abs().sum(dim=0) != 0
    selected_idxs = nonzero_mask.nonzero(as_tuple=True)[0]
    pos_cache_keys = cache_keys.index_select(1, selected_idxs).contiguous()
    cache_values_new = F.one_hot(selected_idxs, num_classes=num_classes).to(
        dtype=cache_keys.dtype, device=cache_keys.device)
    all_classes = selected_idxs.tolist()
    return pos_cache_keys, cache_values_new, all_classes


# ---------------------------------------------------------------------------
# Logit computation helpers
# ---------------------------------------------------------------------------

def compute_cache_logits(point_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using entropy, align or negative cache.

    Args:
        point_features: (1, dim) or (1, n_patches, dim) 3D features
        cache: dict, class_idx -> list of (feature, loss, [prob_map])
        alpha, beta: scalar weights
        clip_weights: (dim, n_cls) text classifier
        neg_mask_thresholds: optional (low, high) tuple for negative cache masking
    """
    with torch.no_grad():
        cache_keys_list = []
        cache_values_list = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys_list.append(item[0])
                if neg_mask_thresholds:
                    cache_values_list.append(item[2])
                else:
                    cache_values_list.append(class_index)

        if not cache_keys_list:
            return torch.zeros(1, clip_weights.size(1), device=clip_weights.device)

        cache_keys_t = torch.cat(cache_keys_list, dim=0).permute(1, 0)

        if neg_mask_thresholds:
            cache_values_t = torch.cat(cache_values_list, dim=0)
            cache_values_t = (
                ((cache_values_t > neg_mask_thresholds[0]) & (cache_values_t < neg_mask_thresholds[1]))
                .type(torch.int8)
                .to(dtype=clip_weights.dtype, device=clip_weights.device)
            )
        else:
            cache_values_t = F.one_hot(
                torch.tensor(cache_values_list, dtype=torch.int64),
                num_classes=clip_weights.size(1)
            ).to(dtype=clip_weights.dtype, device=clip_weights.device)

        # Handle multi-patch features: use mean pooling over patches
        if point_features.dim() == 3:
            point_features = point_features.mean(dim=1)  # (1, dim)

        affinity = point_features @ cache_keys_t
        cache_logits = ((-1) * (beta - beta * affinity)).exp().to(cache_values_t.dtype) @ cache_values_t
        return alpha * cache_logits


def compute_cache_key_logits(point_features, cache_keys, cache_values, alpha, beta):
    """Compute logits from compact cache key/value matrices."""
    if point_features.dim() == 3:
        point_features = point_features.mean(dim=1)
    affinity = point_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits


def get_cache_pred(point_features, cache_memory, global_text_feat):
    """Compute prediction using adaptive similarity between positive caches and text features."""
    feat = point_features[:1].to(dtype=cache_memory.dtype, device=cache_memory.device, non_blocking=True)
    if feat.dim() == 3:
        feat = feat.mean(dim=1)
    global_text_feat = global_text_feat.to(dtype=cache_memory.dtype, device=cache_memory.device, non_blocking=True)

    cached_image_feat = torch.cat((cache_memory, global_text_feat), dim=1)
    cached_image_feat_KV = cached_image_feat / cached_image_feat.norm(dim=-1, keepdim=True)
    cached_image_feat_KV[cached_image_feat.sum(-1) == 0] = 0

    similarity_score = (feat * cached_image_feat_KV).sum(-1)
    similarity_score = torch.exp(-5.5 * (-similarity_score + 1))
    adaptive_image_feat = (cached_image_feat_KV * similarity_score.unsqueeze(-1)).sum(1)
    adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
    logits = 100. * adaptive_image_feat @ feat.unsqueeze(-1)
    logits = logits[:, :, 0]
    return logits.softmax(dim=1)


def align_neg_keys(point_features, pos_classes, neg_cache):
    """Align negative cache features to match positive class dimensions."""
    with torch.no_grad():
        aligned_neg_keys = []
        for class_idx in pos_classes:
            if class_idx in neg_cache:
                num_items = len(neg_cache[class_idx])
                class_prototype = torch.zeros_like(point_features)
                for item in neg_cache[class_idx]:
                    class_prototype += item[0] / num_items
                aligned_neg_keys.append(class_prototype)
            else:
                aligned_neg_keys.append(torch.zeros_like(point_features))
        aligned_neg_keys = torch.cat(aligned_neg_keys, dim=0).permute(1, 0)
        return aligned_neg_keys


def select_confident_samples(prob):
    """Select top-10% lowest-entropy samples from augmented views (2D only).
    For 3D single-view, simply returns the prediction and entropy.
    """
    if prob.size(0) == 1:
        # Single-view 3D case: no multi-view selection needed
        init_pred = prob
        align_pred = int(init_pred[0].argmax())
        aug_loss = -(init_pred[0] * (init_pred[0].clamp_min(1e-8).log())).sum()
        return init_pred, align_pred, aug_loss
    # Multi-view case
    batch_entropy = -(prob * torch.log(prob + 1e-6)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size(0) * 0.1)]
    init_pred = prob[idx].mean(0, keepdim=True)
    align_pred = int(init_pred[0].argmax())
    aug_loss = -(init_pred[0] * (init_pred[0].clamp_min(1e-8).log())).sum()
    return init_pred, align_pred, aug_loss


# ---------------------------------------------------------------------------
# MCP++ residual modules (unchanged from ICCV25-MCP)
# ---------------------------------------------------------------------------

class PositiveCacheResidue(nn.Module):
    def __init__(self, pos_cache_keys):
        super().__init__()
        self.feat_dim, self.cache_size = pos_cache_keys.shape
        self.residual = nn.Parameter(
            torch.zeros([self.feat_dim, self.cache_size],
                        dtype=pos_cache_keys.dtype, device=pos_cache_keys.device),
            requires_grad=True
        )

    def forward(self, x):
        new_pos_cache_keys = x.clone() + self.residual
        new_pos_cache_keys = F.normalize(new_pos_cache_keys, dim=0)
        return new_pos_cache_keys


class TextResidue(nn.Module):
    def __init__(self, clip_weights):
        super().__init__()
        self.feat_dim, self.class_num = clip_weights.shape
        self.residual = nn.Parameter(
            torch.zeros([self.feat_dim, self.class_num],
                        dtype=clip_weights.dtype, device=clip_weights.device),
            requires_grad=True
        )

    def forward(self, x):
        x = F.normalize(x, dim=0)
        new_clip_weights = x + self.residual
        new_clip_weights = F.normalize(new_clip_weights, dim=0)
        return new_clip_weights


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def loss_negative_positive(v_positive, v_negative):
    cosine_similarity = F.cosine_similarity(v_positive, v_negative, dim=1)
    epsilon = 1e-7
    loss = -torch.log(1 - cosine_similarity + epsilon).mean()
    return loss


# ---------------------------------------------------------------------------
# 3D feature extraction (replaces get_clip_logits from 2D MCP)
# ---------------------------------------------------------------------------

def get_3d_logits(args, feat, lm3d_model, clip_weights):
    """Extract 3D features and compute logits using Point-Cache's model.

    Args:
        args: argument namespace (must have .lm3d, .cache_type, .p_thres)
        feat: (1, N, 6) tensor of [xyz, rgb] point cloud
        lm3d_model: the loaded 3D model
        clip_weights: (emb_dim, n_cls) text classifier

    Returns:
        point_features: (1, emb_dim) or (1, n_patches, emb_dim)
        clip_logits: (1, n_cls)
        loss: scalar entropy loss
        pred: int, predicted class
        global_feat: (1, emb_dim) or (1, n_patches, emb_dim) - for cache storage
        prob_map: (1, n_cls) softmax probability
    """
    # Determine cache type for Point-Cache's get_logits
    if args.cache_type not in ['global', 'local', 'hierarchical']:
        # Default to hierarchical for maximum info
        args.cache_type = 'hierarchical'

    if args.cache_type == 'local':
        patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feat, lm3d_model, clip_weights)
        point_features = patch_centers  # (1, n_patches, dim)
        global_feat = patch_centers
    elif args.cache_type == 'global':
        pc_feats, clip_logits, loss, prob_map, pred = get_logits(args, feat, lm3d_model, clip_weights)
        point_features = pc_feats  # (1, dim)
        global_feat = pc_feats
    elif args.cache_type == 'hierarchical':
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feat, lm3d_model, clip_weights)
        point_features = pc_feats  # (1, dim) global for classification
        global_feat = patch_centers  # (1, n_patches, dim) local for cache
    else:
        raise ValueError(f'Unknown cache_type: {args.cache_type}')

    return point_features, clip_logits, loss, pred, global_feat, prob_map


# ---------------------------------------------------------------------------
# Main MCP3D test-time adaptation loop
# ---------------------------------------------------------------------------

def run_test_mcp3d(args, pos_cfg, neg_cfg, lr_cfg, test_loader, lm3d_model, clip_weights, dataset_name, classnames):
    """Run MCP test-time adaptation on 3D point cloud data.

    This is the core adaptation loop, structurally following ICCV25-MCP's
    run_test_mcp but using 3D point encoders instead of CLIP image encoder.
    """
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top1_cache = AverageMeter('AccCache@1', ':6.2f', Summary.AVERAGE)
    top1_pro = AverageMeter('AccPro@1', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(len(test_loader), [top1, top1_cache, top1_pro], prefix='Test: ')
    pred_vanilla, pred_cache, pred_pro, labels = [], [], [], []
    entro_cache, neg_cache, align_cache = {}, {}, {}
    n_cls = len(classnames)

    # Unpack hyperparameters
    pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ['align_shot', 'entropy_shot', 'alpha', 'beta']}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

    pos_cache_keys, all_classes = None, []
    feat_dim = clip_weights.shape[0]
    cache_memory = torch.zeros(
        (n_cls, pos_params['entropy_shot'] + pos_params['align_shot'], feat_dim),
        dtype=clip_weights.dtype, device=clip_weights.device
    )
    cache_keys = torch.zeros(
        (feat_dim, n_cls), dtype=clip_weights.dtype, device=clip_weights.device
    )
    is_res = args.res.lower() == "true"
    clip_weights_global = clip_weights.clone()

    num_avg = 0

    # --- Test-time adaptation loop ---
    for i, (pc, target, _, rgb) in enumerate(tqdm(test_loader, desc='Processed test samples: ')):
        # Build 6-channel point cloud input (xyz + rgb)
        feat = torch.cat([pc, rgb], dim=-1).half()
        target = target.cuda()

        if is_res:
            clip_weights_local = clip_weights_global.clone().detach()
            text_residue = TextResidue(clip_weights_local)
            new_clip_weights = text_residue(clip_weights_local)

        # Extract 3D features and compute initial logits
        point_features, clip_logits, loss, ent_pred, global_feat, prob_map = get_3d_logits(
            args, feat, lm3d_model,
            new_clip_weights if is_res else clip_weights
        )

        with torch.no_grad():
            prop_entropy = get_entropy(loss, clip_weights)
            init_pred, align_pred, aug_loss = select_confident_samples(prob_map)

        # --- Positive cache updates ---
        if pos_enabled:
            # Entropy cache
            update_cache(entro_cache, ent_pred, [point_features, loss], pos_params['entropy_shot'])
            # Align cache
            update_align_cache(
                align_cache, align_pred,
                [global_feat, aug_loss],
                pos_params['align_shot'], args.cen,
                clip_weights, all_classes, pos_cache_keys
            )
            with torch.no_grad():
                update_cache_joint(cache_memory, cache_keys, entro_cache, align_cache, ent_pred, align_pred)
            pos_cache_keys, pos_cache_values, all_classes = shrink_cache_keys_and_values(cache_keys)

        # --- Negative cache updates ---
        neg_logits = 100. * point_features @ clip_weights
        if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
            neg_logits = neg_logits + compute_cache_logits(
                point_features, entro_cache, pos_params['alpha'], pos_params['beta'], clip_weights
            )
            neg_loss = softmax_entropy(neg_logits)
            neg_entropy = get_entropy(neg_loss, clip_weights)
            neg_pred = int(neg_logits.topk(1, 1, True, True)[1].t()[0])
            if neg_enabled and neg_params['entropy_threshold']['lower'] < neg_entropy < neg_params['entropy_threshold']['upper']:
                neg_map = neg_logits.softmax(-1)
                update_cache(neg_cache, neg_pred, [point_features, neg_loss, neg_map], neg_params['shot_capacity'], True)
            elif neg_entropy <= neg_params['entropy_threshold']['lower']:
                update_cache(entro_cache, neg_pred, [point_features, neg_loss], pos_params['entropy_shot'])

        final_logits = clip_logits.clone()

        # --- MCP++ residual learning ---
        if is_res:
            pos_cache_residue = PositiveCacheResidue(pos_cache_keys)
            neg_cache_keys = align_neg_keys(point_features, all_classes, neg_cache)
            if args.tta_steps > 0:
                optimizer = torch.optim.AdamW([
                    {'params': text_residue.parameters(), 'lr': lr_cfg['text'], 'eps': 1e-3, 'weight_decay': 1e-1},
                    {'params': pos_cache_residue.parameters(), 'lr': lr_cfg['image'], 'eps': 1e-3, 'weight_decay': 1e-1}
                ])
                for j in range(args.tta_steps):
                    new_clip_weights = text_residue(clip_weights_local)
                    if pos_enabled:
                        new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                        final_logits = final_logits + compute_cache_key_logits(
                            point_features, new_pos_cache_keys, pos_cache_values,
                            pos_params['alpha'], pos_params['beta']
                        )
                    if neg_enabled and neg_cache:
                        final_logits = final_logits - compute_cache_logits(
                            point_features, neg_cache,
                            neg_params['alpha'], neg_params['beta'],
                            new_clip_weights,
                            (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper'])
                        )
                    entropy_loss = avg_entropy(final_logits)
                    pos2neg_loss = loss_negative_positive(new_pos_cache_keys.T, neg_cache_keys.T)
                    img2text_loss = F.cross_entropy(
                        (point_features @ new_clip_weights[:, all_classes]).float(),
                        torch.tensor([all_classes.index(align_pred)], device=point_features.device)
                    ) if all_classes else torch.tensor(0.0, device=point_features.device)

                    lamda, gamma = 0.5, 0.2
                    loss_total = entropy_loss + lamda * img2text_loss + gamma * pos2neg_loss

                    optimizer.zero_grad()
                    if j == args.tta_steps - 1:
                        loss_total.backward()
                    else:
                        loss_total.backward(retain_graph=True)
                    optimizer.step()

            pos_cache_residue.eval()
            text_residue.eval()
            with torch.no_grad():
                new_clip_weights = text_residue(clip_weights_local)
                new_img_text = 100. * point_features @ new_clip_weights
                new_img_text = new_img_text.softmax(dim=-1)
                confi_logits, _, _ = select_confident_samples(new_img_text)

        # --- Final logit composition ---
        with torch.no_grad():
            if pos_enabled and (entro_cache or align_cache):
                if is_res:
                    new_pos_cache_keys = pos_cache_residue(pos_cache_keys)
                    final_logits = final_logits + compute_cache_key_logits(
                        point_features, new_pos_cache_keys, pos_cache_values,
                        pos_params['alpha'], pos_params['beta']
                    )
                else:
                    final_logits = final_logits + compute_cache_key_logits(
                        point_features, pos_cache_keys, pos_cache_values,
                        pos_params['alpha'], pos_params['beta']
                    )
            if neg_enabled and neg_cache:
                final_logits = final_logits - compute_cache_logits(
                    point_features, neg_cache,
                    neg_params['alpha'], neg_params['beta'],
                    clip_weights,
                    (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper'])
                )
            final_logits = final_logits.softmax(-1)
            img_pro_pred = final_logits

            if is_res:
                img_text_pred = confi_logits
            else:
                img_text_pred = init_pred

            global_text_feat = clip_weights.clone().unsqueeze(1).permute(2, 1, 0).to(clip_weights.device)
            img_global_pred = get_cache_pred(global_feat, cache_memory, global_text_feat)

        # --- Cumulative averaging for MCP++ ---
        if is_res:
            fin_loss = avg_entropy(final_logits)
            if get_entropy(fin_loss, clip_weights) < 0.1:
                num_avg += 1
                clip_weights_global = (
                    clip_weights_global * (num_avg / (num_avg + 1)) +
                    new_clip_weights * (1 / (num_avg + 1))
                )

        # --- Record predictions ---
        with torch.no_grad():
            pred_vanilla.append(img_text_pred)
            pred_cache.append(img_global_pred)
            pred_pro.append(img_pro_pred)
            labels.append(target)

            acc1, _ = accuracy(img_text_pred, target, topk=(1, 5))
            acc1_global, _ = accuracy(img_global_pred, target, topk=(1, 5))
            acc1_pro, _ = accuracy(img_pro_pred, target, topk=(1, 5))
            top1.update(acc1[0], 1)
            top1_cache.update(acc1_global[0], 1)
            top1_pro.update(acc1_pro[0], 1)

        torch.cuda.empty_cache()
        if i % 1000 == 0:
            progress.display(i)

    # --- Post-loop weight search ---
    with torch.no_grad():
        progress.display_summary()
        pred_vanilla = torch.cat(pred_vanilla, dim=0)
        pred_cache = torch.cat(pred_cache, dim=0)
        pred_pro = torch.cat(pred_pro, dim=0)
        labels = torch.cat(labels, dim=0)

        weight_search = True
        if weight_search:
            beta1_list = [1.0]
            beta2_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
            beta3_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
            print('-' * 20)
            print('Starting weight search...')
            best_acc, best_beta2, best_beta3 = 0., 0., 0.
            for beta1 in beta1_list:
                for beta2 in beta2_list:
                    for beta3 in beta3_list:
                        logits = pred_vanilla * beta1 + pred_cache * beta2 + pred_pro * beta3
                        acc, _ = accuracy(logits, labels, topk=(1, 5))
                        acc = acc.item()
                        if acc > best_acc:
                            print(f'New best: beta1={beta1:.4f}, beta2={beta2:.4f}, beta3={beta3:.4f}, Acc={acc:.2f}')
                            best_acc, best_beta1, best_beta2, best_beta3 = acc, beta1, beta2, beta3
            print(f"Searched Acc: {best_acc:.2f} with beta1 {best_beta1:.3f}, dynamic {best_beta2:.3f}, static {best_beta3:.3f}")

    return [best_acc, best_beta1, best_beta2, best_beta3]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_arguments():
    parser = argparse.ArgumentParser(description='MCP3D: Multi-Cache Prototype Learning for 3D Point Clouds')

    # --- MCP config ---
    parser.add_argument('--config', dest='config', required=True,
                        help='Path to MCP config directory (yaml files per dataset).')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True,
                        help='Datasets separated by slash, e.g. modelnet_c/scanobjnn')
    parser.add_argument('--data-root', dest='data_root', type=str, default='./data/',
                        help='Path to the datasets directory.')

    # --- 3D model settings (from Point-Cache) ---
    parser.add_argument('--lm3d', default='ulip', type=str,
                        choices=['ulip', 'ulip2', 'openshape', 'uni3d'],
                        help='Which large multi-modal 3D model to use.')
    parser.add_argument('--cache-type', type=str, default='hierarchical',
                        choices=['global', 'local', 'hierarchical'],
                        help='Feature granularity from the 3D model.')

    # --- MCP hyperparameters ---
    parser.add_argument('--cen', default=0.8, type=float, help='Center weight for class center update.')
    parser.add_argument('--tta-steps', default=1, type=int,
                        help='TTA steps for MCP++ residual learning.')
    parser.add_argument('--res', default='False', type=str,
                        help='Enable MCP++ residual learning (True/False).')

    # --- ULIP / ULIP-2 settings ---
    parser.add_argument('--ulip-version', type=str, default='ulip2', choices=['ulip1', 'ulip2'])
    parser.add_argument('--slip-ckpt-path', type=str,
                        default='weights/ulip/slip_base_100ep.pt')
    parser.add_argument('--pc-depth', type=int, default=12)
    parser.add_argument('--num-head', type=int, default=6)
    parser.add_argument('--encoder-dim', type=int, default=256)

    # --- OpenShape settings ---
    parser.add_argument('--oshape-version', type=str, default='vitg14', choices=['vitg14', 'vitl14'])
    parser.add_argument('--npoints', default=1024, type=int)

    # --- Uni3D settings ---
    parser.add_argument('--pc-model', type=str, default='eva_giant_patch14_560')
    parser.add_argument('--clip-model', type=str, default='EVA02-E-14-plus')
    parser.add_argument('--pretrained', default='weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin', type=str)
    parser.add_argument('--ckpt-path', default='weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt', type=str)
    parser.add_argument('--drop-path-rate', default=0.0, type=float)

    # --- Shared 3D model settings ---
    parser.add_argument('--pc-feat-dim', type=int, default=768)
    parser.add_argument('--group-size', type=int, default=32)
    parser.add_argument('--num-group', type=int, default=512)
    parser.add_argument('--pc-encoder-dim', type=int, default=512)
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--patch-dropout', type=float, default=0.)

    # --- General ---
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--p-thres', type=float, default=0.1)
    parser.add_argument('--print-freq', type=int, default=500)
    parser.add_argument('--sonn-variant', type=str, default='hardest')
    parser.add_argument('--cor-type', type=str, default='add_global_2')
    parser.add_argument('--k-shot', type=int, default=3)
    parser.add_argument('--n-cluster', type=int, default=3)

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_random_seed(args.seed)

    # Load 3D model + CLIP text model via Point-Cache's loader
    print(f'Loading 3D model: {args.lm3d} ...')
    clip_model, lm3d_model = load_models(args)
    lm3d_model.eval()

    # Run MCP3D on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} dataset with {args.lm3d}.")
        print(f"{'='*60}")

        # Load dataset config (MCP hyperparameters)
        cfg = load_mcp_config(args.config, dataset_name)
        print("Dataset configurations:\n", cfg)

        # Build test data loader (Point-Cache native)
        test_loader, classnames, template = build_test_data_loader(args, dataset_name, args.data_root, None)
        print(f"Number of classes: {len(classnames)}")

        # Build text classifier (CLIP text encoder)
        clip_weights = clip_classifier(args, classnames, template, clip_model)
        clip_weights = clip_weights.half().to(device)

        start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[INFO] Experiment started at {start_timestamp}\n")

        results = run_test_mcp3d(
            args, cfg['positive'], cfg['negative'], cfg['learning_rate'],
            test_loader, lm3d_model, clip_weights, dataset_name, classnames
        )
        method_name = "MCP++" if args.res.lower() == "true" else "MCP"
        print(f"\n=> {method_name} Acc. on testset [{dataset_name}]: {results[0]:.2f}")


# ---------------------------------------------------------------------------
# MCP config file reader (different format from Point-Cache's get_config_file)
# ---------------------------------------------------------------------------

def load_mcp_config(config_dir, dataset_name):
    """Load MCP hyperparameter config for a given dataset."""
    config_file = os.path.join(config_dir, f"{dataset_name}.yaml")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"MCP config file not found: {config_file}")
    with open(config_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return cfg


if __name__ == "__main__":
    main()
