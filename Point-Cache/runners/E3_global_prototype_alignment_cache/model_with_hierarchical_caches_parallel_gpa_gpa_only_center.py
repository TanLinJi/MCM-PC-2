"""
E3: Global Prototype-Alignment Cache for Point-Cache.

当前实现是 E3-V2-A-parallel-gpa-cache-gpa-only-center：并列式 GPA Cache。

核心思想：
1. 保留原始 Point-Cache 的 Global Entropy Cache，仍用于 global cache logits；
2. 新增 Global Prototype-Alignment Cache，简称 GPA Cache；
3. GPA Cache 自己维护每个类别的全局原型中心；
4. GPA Cache 未形成中心前，先按低熵准入积累初始样本；
5. GPA Cache 形成中心后，再启用“低熵 + 原型距离更近”的严格更新；
6. 只有进入 GPA Cache 的样本，其 patch_centers 才写入 local cache；
7. 当前最小验证阶段暂不修改最终预测加权公式。
"""

import os
import sys
import json
import time
import operator
from pathlib import Path
from collections import defaultdict

import wandb
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import *  # noqa: F401,F403

CENTER_SOURCE_LABEL = "GPA-only center"
GPA_VARIANT_NAME = "E3-V2-A-parallel-gpa-cache-gpa-only-center"


def _loss_value(loss):
    """Convert entropy tensor/scalar to python float for sorting and logging."""
    if torch.is_tensor(loss):
        return float(loss.detach().float().cpu().item())
    return float(loss)


def _clone_item(item):
    """Keep tensor references, but make list structure independent."""
    return list(item)



def _get_gpa_stats_enabled():
    return os.environ.get("GPA_SAVE_STATS", "1") != "0"


def _normalize_center(center):
    return center / center.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _compute_gpa_center(gpa_cache, pred):
    """
    GPA-only center:
    类别 pred 的原型中心 = GPA Cache[pred] 中全局特征均值。
    """
    if pred not in gpa_cache or len(gpa_cache[pred]) == 0:
        return None

    feats = [item[0] for item in gpa_cache[pred]]
    center = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)
    return _normalize_center(center)


def _feature_distance_to_center(feat, center):
    """
    使用欧氏距离。Point-Cache 中 pc_feats 已经归一化，因此该距离等价于一种余弦距离的单调变换。
    """
    return float(torch.norm(feat - center, p=2).detach().float().cpu().item())


def _sort_cache_by_entropy(cache, pred):
    cache[pred] = sorted(cache[pred], key=lambda x: _loss_value(x[1]))


def _sort_local_cache_by_entropy(local_cache, pred):
    local_cache[pred] = sorted(local_cache[pred], key=lambda x: _loss_value(x[1]))


def _update_entropy_cache(cache, pred, item, shot_capacity, stats, phase):
    """
    原始 Point-Cache 风格的 Global Entropy Cache 更新。

    返回:
        accepted: 当前样本是否成功进入或替换 Global Entropy Cache。
    """
    if pred in cache:
        if len(cache[pred]) < shot_capacity:
            cache[pred].append(item)
            _sort_cache_by_entropy(cache, pred)
            stats[f"{phase}_entropy_add"] += 1
            return True

        worst_ent = _loss_value(cache[pred][-1][1])
        curr_ent = _loss_value(item[1])

        if curr_ent < worst_ent:
            cache[pred][-1] = item
            _sort_cache_by_entropy(cache, pred)
            stats[f"{phase}_entropy_replace"] += 1
            return True

        stats[f"{phase}_entropy_reject"] += 1
        return False

    cache[pred] = [item]
    stats[f"{phase}_entropy_add"] += 1
    return True


def _update_negative_cache(cache, pred, item, shot_capacity, stats, phase):
    """
    Negative cache 保持原始 Point-Cache 的低熵排序替换逻辑。
    item = [pc_feats, loss, prob_map]
    """
    if pred in cache:
        if len(cache[pred]) < shot_capacity:
            cache[pred].append(item)
            _sort_cache_by_entropy(cache, pred)
            stats[f"{phase}_neg_add"] += 1
            return True

        worst_ent = _loss_value(cache[pred][-1][1])
        curr_ent = _loss_value(item[1])

        if curr_ent < worst_ent:
            cache[pred][-1] = item
            _sort_cache_by_entropy(cache, pred)
            stats[f"{phase}_neg_replace"] += 1
            return True

        stats[f"{phase}_neg_reject"] += 1
        return False

    cache[pred] = [item]
    stats[f"{phase}_neg_add"] += 1
    return True



def _update_gpa_cache(gpa_cache, gpa_local_cache, pred, global_item, local_item, shot_capacity, stats, phase, event_records=None):
    """
    更新 Global Prototype-Alignment Cache，并同步控制 local cache。

    global_item = [pc_feats, loss]
    local_item  = [patch_centers, loss]

    当前 E3-V2 并列式规则：

    1. GPA Cache 未满时：
       - GPA Cache 与 Global Entropy Cache 并列更新，
         样本会独立尝试进入 GPA Cache；
       - 此时不启用距离约束。

    2. GPA Cache 已满时：
       - 找到 GPA Cache 中当前最高熵样本；
       - 如果新样本熵更低；
       - 并且新样本到 GPA 原型中心的距离
         小于这个最高熵样本到 GPA 原型中心的距离；
       - 则替换该最高熵样本。

    说明：
    当前版本保留该规则用于记录初版负结果。
    后续 Center-B / Center-C 或并列式方案需要重新设计未满阶段的距离准入。
    """
    if pred not in gpa_cache:
        gpa_cache[pred] = []
        gpa_local_cache[pred] = []

    curr_ent = _loss_value(global_item[1])

    def record_event(decision, old_entropy=None, new_distance=None, old_distance=None):
        if event_records is None:
            return
        event_records.append({
            "phase": phase,
            "class_index": int(pred),
            "decision": decision,
            "new_entropy": float(curr_ent),
            "old_entropy": None if old_entropy is None else float(old_entropy),
            "new_distance": None if new_distance is None else float(new_distance),
            "old_distance": None if old_distance is None else float(old_distance),
        })

    # GPA Cache 未满：直接加入。
    # 注意：该函数只会在样本已经通过 Global Entropy Cache 准入后被调用。
    if len(gpa_cache[pred]) < shot_capacity:
        gpa_cache[pred].append(global_item)
        gpa_local_cache[pred].append(local_item)
        _sort_cache_by_entropy(gpa_cache, pred)
        _sort_local_cache_by_entropy(gpa_local_cache, pred)
        stats[f"{phase}_gpa_add_not_full"] += 1
        return True

    # GPA Cache 已满：启用 GPA-only center 的距离约束。
    center = _compute_gpa_center(gpa_cache, pred)

    if center is None:
        stats[f"{phase}_gpa_no_center_reject"] += 1
        return False

    worst_global_item = gpa_cache[pred][-1]
    worst_ent = _loss_value(worst_global_item[1])

    curr_dist = _feature_distance_to_center(global_item[0], center)
    worst_dist = _feature_distance_to_center(worst_global_item[0], center)

    if curr_ent >= worst_ent:
        stats[f"{phase}_gpa_reject_entropy"] += 1
        record_event(
            decision="reject_entropy",
            old_entropy=worst_ent,
            new_distance=curr_dist,
            old_distance=worst_dist,
        )
        return False

    if curr_dist < worst_dist:
        gpa_cache[pred][-1] = global_item
        gpa_local_cache[pred][-1] = local_item
        _sort_cache_by_entropy(gpa_cache, pred)
        _sort_local_cache_by_entropy(gpa_local_cache, pred)
        stats[f"{phase}_gpa_replace"] += 1
        record_event(
            decision="replace",
            old_entropy=worst_ent,
            new_distance=curr_dist,
            old_distance=worst_dist,
        )
        return True

    stats[f"{phase}_gpa_reject_distance"] += 1
    record_event(
        decision="reject_distance",
        old_entropy=worst_ent,
        new_distance=curr_dist,
        old_distance=worst_dist,
    )
    return False


def _summarize_cache(cache):
    return {str(k): len(v) for k, v in sorted(cache.items(), key=lambda kv: kv[0])}


def _save_gpa_stats(args, stats, entropy_cache, gpa_cache, gpa_local_cache, acc=None, event_records=None):
    if not _get_gpa_stats_enabled():
        return

    result_root = getattr(args, "baseline_result_root", None)
    exp_id = getattr(args, "baseline_exp_id", None)

    if not result_root or not exp_id:
        return

    out_dir = Path(result_root) / exp_id / "gpa_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exp_id": exp_id,
        "cor_type": getattr(args, "cor_type", None),
        "gpa_variant": GPA_VARIANT_NAME,
        "center_source": CENTER_SOURCE_LABEL,
        "final_acc": acc,
        "stats": dict(stats),
        "entropy_cache_class_counts": _summarize_cache(entropy_cache),
        "gpa_cache_class_counts": _summarize_cache(gpa_cache),
        "gpa_local_cache_class_counts": _summarize_cache(gpa_local_cache),
        "entropy_cache_total": int(sum(len(v) for v in entropy_cache.values())),
        "gpa_cache_total": int(sum(len(v) for v in gpa_cache.values())),
        "gpa_local_cache_total": int(sum(len(v) for v in gpa_local_cache.values())),
    }

    filename = f"{getattr(args, 'cor_type', 'unknown')}_gpa_stats.json"
    with (out_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if event_records is not None:
        event_filename = f"gpa_replacement_events_{getattr(args, 'cor_type', 'unknown')}.jsonl"
        event_path = out_dir / event_filename
        with event_path.open("w", encoding="utf-8") as f:
            for event in event_records:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"[E3-GPA] Saved GPA replacement events to {event_path}")

    print(f"[E3-GPA] Saved GPA stats to {out_dir / filename}")


@torch.no_grad()
def build_cache_in_advance(args, test_loader, lm3d_model, clip_weights, shot_capacity, include_prob_map=False):
    """
    Build cache in advance.

    include_prob_map=False:
        构建 Global Entropy Cache + GPA Cache + GPA-controlled Local Cache。

    include_prob_map=True:
        构建 Negative Cache。此时不构建 local cache。
    """
    stats = defaultdict(int)

    if include_prob_map:
        print("*" * 10, "Building [global] neg. cache ...", "*" * 10, "\n")
        neg_cache = {}

        for pc, _, _, rgb in test_loader:
            feature = torch.cat([pc, rgb], dim=-1).half()
            pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

            item = [pc_feats, loss, prob_map]
            _update_negative_cache(neg_cache, pred, item, shot_capacity, stats, "build")

            cache_num = sum(len(neg_cache[key]) for key in neg_cache)
            num_classes = clip_logits.size(1)
            full_num = shot_capacity * num_classes

            if cache_num == full_num:
                print("*" * 10, "Building [global] neg. cache is Done!", "*" * 10, "\n")
                break

        return neg_cache, {}, stats

    print("*" * 10, "Building [global entropy] and [GPA-controlled local] pos. cache ...", "*" * 10, "\n")

    entropy_cache = {}
    gpa_cache = {}
    gpa_local_cache = {}
    gpa_event_records = []

    for pc, _, _, rgb in test_loader:
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        global_item = [pc_feats, loss]
        local_item = [patch_centers, loss]

        _update_entropy_cache(entropy_cache, pred, global_item, shot_capacity, stats, "build")

        _update_gpa_cache(
            gpa_cache,
            gpa_local_cache,
            pred,
            global_item,
            local_item,
            shot_capacity,
            stats,
            "build",
            gpa_event_records,
        )
        cache_num = sum(len(entropy_cache[key]) for key in entropy_cache)
        num_classes = clip_logits.size(1)
        full_num = shot_capacity * num_classes

        if cache_num == full_num:
            print("*" * 10, "Building [global entropy] cache is full.", "*" * 10, "\n")
            break

    return entropy_cache, gpa_cache, gpa_local_cache, stats, gpa_event_records


@torch.no_grad()
def compute_cache_logits(pc_feats, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative global cache."""
    cache_keys = []
    cache_values = []

    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            cache_keys.append(item[0])
            if neg_mask_thresholds:
                cache_values.append(item[2])
            else:
                cache_values.append(class_index)

    if not cache_keys:
        return torch.zeros_like(pc_feats @ clip_weights)

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
    """Compute logits using GPA-controlled positive local cache."""
    local_cache_keys = []
    local_cache_values = []

    for class_index in sorted(local_cache.keys()):
        for item in local_cache[class_index]:
            local_cache_keys.append(item[0])
            n_cluster = item[0].shape[0]
            local_cache_values.append([class_index] * n_cluster)

    if not local_cache_keys:
        return torch.zeros((1, clip_weights.size(1)), device=patch_centers.device, dtype=patch_centers.dtype)

    local_cache_keys = torch.cat(local_cache_keys, dim=0).permute(1, 0)

    local_cache_values = F.one_hot(
        torch.Tensor(local_cache_values).to(torch.int64),
        num_classes=clip_weights.size(1)
    ).half().cuda()
    local_cache_values = local_cache_values.view(-1, clip_weights.size(1))

    affinity = patch_centers.mean(dim=0, keepdim=True) @ local_cache_keys
    local_cache_logits = ((-1) * (beta - beta * affinity)).exp() @ local_cache_values
    return alpha * local_cache_logits


@torch.no_grad()
def run_test_tda(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights):
    """
    E3-V1 test-time adaptation.

    Global logits:
        使用 Global Entropy Cache，即原始 Point-Cache 的正全局缓存。

    Local logits:
        使用 GPA-controlled Local Cache，即只有进入 GPA Cache 的样本贡献局部特征。

    Negative cache:
        保持原始 Point-Cache 逻辑。
    """
    entropy_cache, gpa_cache, gpa_local_cache, build_stats, gpa_event_records = build_cache_in_advance(
        args, test_loader, lm3d_model, clip_weights, pos_cfg["shot_capacity"]
    )

    print("[E3-GPA] len(entropy_cache):", len(entropy_cache))
    print("[E3-GPA] len(gpa_cache):", len(gpa_cache))
    print("[E3-GPA] len(gpa_local_cache):", len(gpa_local_cache))
    print("[E3-GPA] entropy cache total:", sum(len(v) for v in entropy_cache.values()))
    print("[E3-GPA] gpa cache total:", sum(len(v) for v in gpa_cache.values()))
    print("[E3-GPA] gpa local cache total:", sum(len(v) for v in gpa_local_cache.values()))

    neg_cache = {}
    gpa_cache = {}
    gpa_cache_stats = defaultdict(int)
    for k, v in build_stats.items():
        gpa_cache_stats[k] += v

    # 继承预构建阶段的 GPA global cache 状态。
    # 这样测试阶段的 GPA 原型中心由预构建阶段已有 GPA 样本继续维护，
    # 而不是从空缓存重新开始。
    for k in gpa_local_cache.keys():
        gpa_cache.setdefault(k, [])

    accuracies = []

    pos_enabled, neg_enabled = pos_cfg["enabled"], neg_cfg["enabled"]

    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ["shot_capacity", "alpha", "beta"]}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ["shot_capacity", "alpha", "beta", "entropy_threshold", "mask_threshold"]}

    for i, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()

        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)

        if pos_enabled:
            global_item = [pc_feats, loss]
            local_item = [patch_centers, loss]

            _update_entropy_cache(
                entropy_cache, pred, global_item, pos_params["shot_capacity"], gpa_cache_stats, "test"
            )

            # gpa_cache 是在线阶段的 GPA 状态；
            # gpa_local_cache 则是实际参与 local logits 的 local cache。
            _update_gpa_cache(
                gpa_cache,
                gpa_local_cache,
                pred,
                global_item,
                local_item,
                pos_params["shot_capacity"],
                gpa_cache_stats,
                "test",
                gpa_event_records,
            )
        if neg_enabled and neg_params["entropy_threshold"]["lower"] < prop_entropy < neg_params["entropy_threshold"]["upper"]:
            _update_negative_cache(
                neg_cache,
                pred,
                [pc_feats, loss, prob_map],
                neg_params["shot_capacity"],
                gpa_cache_stats,
                "test"
            )

        final_logits = clip_logits.clone()

        if pos_enabled and entropy_cache:
            final_logits += compute_cache_logits(
                pc_feats,
                entropy_cache,
                pos_params["alpha"],
                pos_params["beta"],
                clip_weights
            )

            if gpa_local_cache:
                final_logits += compute_local_cache_logits(
                    patch_centers,
                    gpa_local_cache,
                    pos_params["alpha"],
                    pos_params["beta"],
                    clip_weights
                )

        if neg_enabled and neg_cache:
            final_logits -= compute_cache_logits(
                pc_feats,
                neg_cache,
                neg_params["alpha"],
                neg_params["beta"],
                clip_weights,
                (neg_params["mask_threshold"]["lower"], neg_params["mask_threshold"]["upper"])
            )

        acc = cls_acc(final_logits, target)
        accuracies.append(acc)
        wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

        if i % args.print_freq == 0:
            print("---- E3-GPA test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    final_acc = sum(accuracies) / len(accuracies)
    print("---- ***Final*** E3-GPA test accuracy: {:.2f}. ----\n".format(final_acc))

    _save_gpa_stats(args, gpa_cache_stats, entropy_cache, gpa_cache, gpa_local_cache, final_acc, gpa_event_records)

    return final_acc
