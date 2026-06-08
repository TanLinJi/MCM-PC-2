"""
E3: Global Prototype-Alignment Cache for Point-Cache.

当前实现是 E3-V2-TextProto-C，即在 E3-V2-C 的视觉 union center 上加入固定类别文本原型。

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
from pathlib import Path
from collections import defaultdict

import wandb
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import *  # noqa: F401,F403

CENTER_SOURCE_LABEL = "TextProto+Entropy+GPA union center"
TEXT_PROTO_VISUAL_WEIGHT = float(os.environ.get("TEXT_PROTO_VISUAL_WEIGHT", "0.7"))
TEXT_PROTO_TEXT_WEIGHT = float(os.environ.get("TEXT_PROTO_TEXT_WEIGHT", "0.3"))
GPA_VARIANT_NAME = f"E3-V2-TextProto-C-w{TEXT_PROTO_VISUAL_WEIGHT:g}v{TEXT_PROTO_TEXT_WEIGHT:g}t"


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



def _collect_unique_features(*caches, pred):
    """
    从多个 cache 中收集类别 pred 的全局特征，并尽量避免同一 tensor 被重复计入。

    说明：
    当前顺序式方案中，同一个样本可能同时存在于 Global Entropy Cache 和 GPA Cache。
    如果直接拼接两个 cache，可能会重复计算同一个样本。
    因此这里优先用 tensor 的 data_ptr 去重。
    """
    feats = []
    seen = set()

    for cache in caches:
        if cache is None or pred not in cache:
            continue

        for item in cache[pred]:
            feat = item[0]
            key = (
                int(feat.data_ptr()),
                tuple(feat.shape),
                str(feat.device),
                str(feat.dtype),
            )
            if key in seen:
                continue
            seen.add(key)
            feats.append(feat)

    return feats


def _compute_gpa_center(gpa_cache, pred, entropy_cache=None):
    """
    Center-C: Entropy+GPA union center.

    类别 pred 的 GPA 原型中心由两个缓存的并集计算：

        Global Entropy Cache[pred]
        +
        Global Prototype-Alignment Cache[pred]

    这样做的目的：
    - 保持顺序式 GPA Cache 关系不变；
    - 只改变“原型中心来源”；
    - 验证同时利用低熵缓存和原型对齐缓存是否能得到更稳定中心。
    """
    feats = _collect_unique_features(entropy_cache, gpa_cache, pred=pred)

    if not feats:
        return None

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

def _get_text_prototype(clip_weights, pred, ref_feat):
    """
    取得类别 pred 对应的 Text Prototype。

    clip_weights 通常形状为 [D, C]，每一列对应一个类别的文本原型。
    返回形状与 ref_feat 对齐的 [1, D]。
    """
    if clip_weights is None:
        return None

    pred = int(pred)

    if clip_weights.dim() != 2:
        return None

    if pred < 0 or pred >= clip_weights.size(1):
        return None

    text_proto = clip_weights[:, pred].detach()

    if text_proto.dim() == 1:
        text_proto = text_proto.unsqueeze(0)

    text_proto = text_proto.to(device=ref_feat.device, dtype=ref_feat.dtype)
    text_proto = text_proto / text_proto.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return text_proto


def _compute_text_visual_center(gpa_cache, pred, entropy_cache, clip_weights, ref_feat):
    """
    E3-V2-TextProto-C：
    构造 Text Prototype + visual prototype 的联合中心。

    visual_center = mean(EntropyCache[c] ∪ GPACache[c])
    text_center   = Text Prototype[c]

    final_center = normalize(
        w_visual * visual_center + w_text * text_center
    )

    如果 text prototype 不可用，则退化为原 E3-V2-C 的 visual center。
    """
    visual_center = _compute_gpa_center(gpa_cache, pred, entropy_cache=entropy_cache)

    if visual_center is None:
        return None

    text_center = _get_text_prototype(clip_weights, pred, ref_feat)

    if text_center is None:
        return visual_center

    center = TEXT_PROTO_VISUAL_WEIGHT * visual_center + TEXT_PROTO_TEXT_WEIGHT * text_center
    center = center / center.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return center


def _update_gpa_cache(entropy_cache, gpa_cache, gpa_local_cache, pred, global_item, local_item, shot_capacity, stats, phase, event_records=None, clip_weights=None):
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

    # GPA Cache 未满：直接加入。该规则沿用 E3-V2-C，并不要求样本已进入 Global Entropy Cache。
    if len(gpa_cache[pred]) < shot_capacity:
        gpa_cache[pred].append(global_item)
        gpa_local_cache[pred].append(local_item)
        _sort_cache_by_entropy(gpa_cache, pred)
        _sort_local_cache_by_entropy(gpa_local_cache, pred)
        stats[f"{phase}_gpa_add_not_full"] += 1
        return True

    # GPA Cache 已满：启用 TextProto + visual union center 的距离约束。
    center = _compute_text_visual_center(gpa_cache, pred, entropy_cache, clip_weights, global_item[0])

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
        "text_proto_visual_weight": float(TEXT_PROTO_VISUAL_WEIGHT),
        "text_proto_text_weight": float(TEXT_PROTO_TEXT_WEIGHT),
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
        print(f"[E3-V2-TextProto-C] Saved GPA replacement events to {event_path}")

    print(f"[E3-V2-TextProto-C] Saved GPA stats to {out_dir / filename}")


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
            entropy_cache,
            gpa_cache,
            gpa_local_cache,
            pred,
            global_item,
            local_item,
            shot_capacity,
            stats,
            "build",
            gpa_event_records,
            clip_weights=clip_weights,
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
    E3-V2-TextProto-C test-time adaptation.

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

    print("[E3-V2-TextProto-C] len(entropy_cache):", len(entropy_cache))
    print("[E3-V2-TextProto-C] len(gpa_cache):", len(gpa_cache))
    print("[E3-V2-TextProto-C] len(gpa_local_cache):", len(gpa_local_cache))
    print("[E3-V2-TextProto-C] entropy cache total:", sum(len(v) for v in entropy_cache.values()))
    print("[E3-V2-TextProto-C] gpa cache total:", sum(len(v) for v in gpa_cache.values()))
    print("[E3-V2-TextProto-C] gpa local cache total:", sum(len(v) for v in gpa_local_cache.values()))

    neg_cache = {}
    gpa_cache_stats = defaultdict(int)
    for k, v in build_stats.items():
        gpa_cache_stats[k] += v

    # 继承预构建阶段的 GPA global/local cache 状态。
    # 否则 TextProto+visual center 在测试阶段会从空 GPA cache 重新开始，
    # 与预构建得到的 GPA-controlled local cache 失去对应关系。
    missing_local_classes = sorted(set(gpa_cache.keys()) - set(gpa_local_cache.keys()))
    missing_global_classes = sorted(set(gpa_local_cache.keys()) - set(gpa_cache.keys()))
    if missing_local_classes or missing_global_classes:
        print(
            "[E3-V2-TextProto-C][WARNING] GPA/global-local cache class mismatch: "
            f"missing_local={missing_local_classes}, missing_global={missing_global_classes}"
        )

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

            # gpa_cache 继承预构建状态并在线更新；
            # gpa_local_cache 是与之同步、实际参与 local logits 的 local cache。
            _update_gpa_cache(
                entropy_cache,
                gpa_cache,
                gpa_local_cache,
                pred,
                global_item,
                local_item,
                pos_params["shot_capacity"],
                gpa_cache_stats,
                "test",
                gpa_event_records,
                clip_weights=clip_weights,
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
            print("---- E3-V2-TextProto-C test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    final_acc = sum(accuracies) / len(accuracies)
    print("---- ***Final*** E3-V2-TextProto-C test accuracy: {:.2f}. ----\n".format(final_acc))

    _save_gpa_stats(args, gpa_cache_stats, entropy_cache, gpa_cache, gpa_local_cache, final_acc, gpa_event_records)

    return final_acc
