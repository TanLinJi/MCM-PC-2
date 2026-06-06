"""
E3: Global Prototype-Alignment Cache for Point-Cache.

当前实现是 E3-V3-C1-Ub：候选池距离初始化 GPA Cache。

核心思想：
1. 保留原始 Point-Cache 的 Global Entropy Cache，仍用于 global cache logits；
2. 新增 Global Prototype-Alignment Cache，简称 GPA Cache；
3. GPA Cache 自己维护每个类别的全局原型中心；
4. GPA Cache 未形成中心前，先按低熵准入积累初始样本；
5. GPA Cache 形成中心后，只启用“距离更近”的无熵更新；
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

CENTER_SOURCE_LABEL = "Candidate-only center"
GPA_VARIANT_NAME = "E3-V3-C1-Ub-candidate-pool-distance-initialization"


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




def _get_gpa_candidate_multiplier():
    """
    GPA 候选池倍率。

    默认 2，表示每类先收集 2K 个候选样本，再筛出 K 个进入正式 GPA Cache。
    """
    return int(os.environ.get("GPA_CANDIDATE_MULTIPLIER", "2"))


def _get_gpa_candidate_capacity(shot_capacity):
    return max(int(shot_capacity), int(_get_gpa_candidate_multiplier()) * int(shot_capacity))


def _make_gpa_candidate(global_item, local_item):
    return {
        "global_item": global_item,
        "local_item": local_item,
        "entropy": _loss_value(global_item[1]),
    }


def _compute_candidate_union_center(entropy_cache, candidate_pool, pred):
    """
    E3-V3-C1-Ub 初始化中心：

        center = mean(GPA candidate pool[pred])

    注意：当前 C1 版本不使用 Entropy Cache 构造候选池临时中心。
    entropy_cache 参数仅为保持函数签名兼容，实际不参与中心计算。
    """
    feats = []

    if candidate_pool is not None and pred in candidate_pool:
        feats.extend([cand["global_item"][0] for cand in candidate_pool[pred]])

    if not feats:
        return None

    center = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)
    return _normalize_center(center)


def _rank_select_gpa_candidates(entropy_cache, gpa_candidate_pool, pred, shot_capacity):
    """
    E3-V3-C1-Ub：从 2K 候选池中只按距离筛出 K 个样本。

    规则：
        1. 临时中心只由 gpa_candidate_pool[pred] 构造；
        2. 不使用 entropy rank；
        3. 选择距离临时中心最近的 K 个样本进入 GPA-Cache。
    """
    candidates = gpa_candidate_pool.get(pred, [])
    if len(candidates) == 0:
        return [], []

    center = _compute_candidate_union_center(None, gpa_candidate_pool, pred)
    if center is None:
        return [], []

    rows = []
    for idx, cand in enumerate(candidates):
        entropy = _loss_value(cand["global_item"][1])
        distance = _feature_distance_to_center(cand["global_item"][0], center)
        rows.append({
            "idx": idx,
            "entropy": entropy,
            "distance": distance,
            "candidate": cand,
        })

    rows = sorted(rows, key=lambda x: (x["distance"], x["entropy"]))

    for rank, row in enumerate(rows):
        row["entropy_rank"] = -1
        row["distance_rank"] = rank
        row["rank_score"] = rank

    selected = rows[:shot_capacity]
    rejected = rows[shot_capacity:]

    return selected, rejected


def _sort_gpa_cache_and_local_together(gpa_cache, gpa_local_cache, pred):
    pairs = list(zip(gpa_cache[pred], gpa_local_cache[pred]))
    pairs = sorted(pairs, key=lambda pair: _loss_value(pair[0][1]))
    gpa_cache[pred] = [pair[0] for pair in pairs]
    gpa_local_cache[pred] = [pair[1] for pair in pairs]


def _record_candidate_event(event_records, phase, pred, decision, row):
    if event_records is None:
        return

    event_records.append({
        "phase": phase,
        "class_index": int(pred),
        "decision": decision,
        "center_source": CENTER_SOURCE_LABEL,
        "new_entropy": float(row["entropy"]),
        "old_entropy": None,
        "new_distance": float(row["distance"]),
        "old_distance": None,
        "entropy_rank": int(row["entropy_rank"]),
        "distance_rank": int(row["distance_rank"]),
        "rank_score": int(row["rank_score"]),
    })


def _finalize_gpa_candidate_pool(
    entropy_cache,
    gpa_cache,
    gpa_local_cache,
    gpa_candidate_pool,
    pred,
    shot_capacity,
    stats,
    phase,
    event_records=None,
    min_candidates=None,
):
    """
    将候选池筛选成正式 GPA Cache。

    条件：
        候选数量 >= min_candidates

    第一版：
        - 正常触发时 min_candidates = 2K；
        - build 阶段结束后补充触发时 min_candidates = K。
    """
    if pred not in gpa_candidate_pool:
        return False

    candidates = gpa_candidate_pool[pred]
    if min_candidates is None:
        min_candidates = shot_capacity

    if len(candidates) < min_candidates:
        stats[f"{phase}_gpa_candidate_not_enough"] += 1
        return False

    if len(candidates) < shot_capacity:
        stats[f"{phase}_gpa_candidate_less_than_k"] += 1
        return False

    selected, rejected = _rank_select_gpa_candidates(
        entropy_cache,
        gpa_candidate_pool,
        pred,
        shot_capacity,
    )

    if len(selected) == 0:
        stats[f"{phase}_gpa_candidate_finalize_failed"] += 1
        return False

    gpa_cache[pred] = [row["candidate"]["global_item"] for row in selected]
    gpa_local_cache[pred] = [row["candidate"]["local_item"] for row in selected]
    _sort_gpa_cache_and_local_together(gpa_cache, gpa_local_cache, pred)

    stats[f"{phase}_gpa_candidate_finalize"] += 1
    stats[f"{phase}_gpa_candidate_selected"] += len(selected)
    stats[f"{phase}_gpa_candidate_rejected"] += len(rejected)

    for row in selected:
        _record_candidate_event(event_records, phase, pred, "candidate_selected", row)
    for row in rejected:
        _record_candidate_event(event_records, phase, pred, "candidate_rejected", row)

    del gpa_candidate_pool[pred]
    return True



def _update_gpa_cache(
    entropy_cache,
    gpa_cache,
    gpa_local_cache,
    gpa_candidate_pool,
    pred,
    global_item,
    local_item,
    shot_capacity,
    stats,
    phase,
    event_records=None,
):
    """
    E3-V3-C1-Ub：候选池距离初始化 + 无熵距离更新。

    初始化阶段：
        如果该类 GPA-Cache 尚未正式建立，则样本先进入 GPA candidate pool；
        当候选池达到 2K 时，只用 candidate pool 构造临时中心；
        选择距离临时中心最近的 K 个样本进入正式 GPA-Cache；
        只有这 K 个样本的局部特征进入 GPA-controlled local cache。

    正式阶段：
        不使用熵门控；
        计算当前 GPA-Cache 中所有样本到 GPA-Center 的距离；
        如果新样本比当前最远样本更近，则替换该最远样本；
        同步替换 local cache；
        替换后立即触发中心重算检查，下一次更新会基于最新 GPA-Cache 计算中心。
    """
    if pred not in gpa_cache or len(gpa_cache.get(pred, [])) == 0:
        if pred in gpa_cache and len(gpa_cache.get(pred, [])) == 0:
            stats[f"{phase}_gpa_empty_formal_cache_recovered"] += 1
            gpa_cache.pop(pred, None)
            gpa_local_cache.pop(pred, None)

        if pred not in gpa_candidate_pool:
            gpa_candidate_pool[pred] = []

        gpa_candidate_pool[pred].append(_make_gpa_candidate(global_item, local_item))
        stats[f"{phase}_gpa_candidate_add"] += 1

        candidate_capacity = _get_gpa_candidate_capacity(shot_capacity)

        if len(gpa_candidate_pool[pred]) >= candidate_capacity:
            return _finalize_gpa_candidate_pool(
                entropy_cache,
                gpa_cache,
                gpa_local_cache,
                gpa_candidate_pool,
                pred,
                shot_capacity,
                stats,
                phase,
                event_records,
                min_candidates=candidate_capacity,
            )

        return False

    curr_ent = _loss_value(global_item[1])

    def record_event(decision, farthest_index=None, old_entropy=None, new_distance=None, old_distance=None):
        if event_records is None:
            return
        event_records.append({
            "phase": phase,
            "class_index": int(pred),
            "decision": decision,
            "center_source": CENTER_SOURCE_LABEL,
            "update_rule": "distance_only_replace_farthest",
            "farthest_index": None if farthest_index is None else int(farthest_index),
            "new_entropy": float(curr_ent),
            "old_entropy": None if old_entropy is None else float(old_entropy),
            "new_distance": None if new_distance is None else float(new_distance),
            "old_distance": None if old_distance is None else float(old_distance),
        })

    center = _compute_gpa_center(gpa_cache, pred, entropy_cache=None)

    if center is None:
        stats[f"{phase}_gpa_no_center_reject"] += 1
        return False

    if pred not in gpa_local_cache or len(gpa_local_cache.get(pred, [])) != len(gpa_cache[pred]):
        raise RuntimeError(
            f"GPA cache/local cache length mismatch for class {pred}: "
            f"gpa={len(gpa_cache.get(pred, []))}, "
            f"local={len(gpa_local_cache.get(pred, []))}"
        )

    curr_dist = _feature_distance_to_center(global_item[0], center)

    cached_distances = [
        _feature_distance_to_center(old_item[0], center)
        for old_item in gpa_cache[pred]
    ]
    farthest_idx = max(range(len(cached_distances)), key=lambda idx: cached_distances[idx])
    farthest_dist = cached_distances[farthest_idx]
    farthest_ent = _loss_value(gpa_cache[pred][farthest_idx][1])

    if curr_dist < farthest_dist:
        gpa_cache[pred][farthest_idx] = global_item
        gpa_local_cache[pred][farthest_idx] = local_item
        stats[f"{phase}_gpa_replace_distance_only"] += 1

        record_event(
            decision="replace_farthest_distance_only",
            farthest_index=farthest_idx,
            old_entropy=farthest_ent,
            new_distance=curr_dist,
            old_distance=farthest_dist,
        )

        _ = _compute_gpa_center(gpa_cache, pred, entropy_cache=None)
        stats[f"{phase}_gpa_center_update_after_replace"] += 1
        return True

    stats[f"{phase}_gpa_reject_distance_only"] += 1

    record_event(
        decision="reject_distance_only",
        farthest_index=farthest_idx,
        old_entropy=farthest_ent,
        new_distance=curr_dist,
        old_distance=farthest_dist,
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
        print(f"[E3-V3-C1-Ub] Saved GPA replacement events to {event_path}")

    print(f"[E3-V3-C1-Ub] Saved GPA stats to {out_dir / filename}")


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
    gpa_candidate_pool = {}
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
            gpa_candidate_pool,
            pred,
            global_item,
            local_item,
            shot_capacity,
            stats,
            "build",
            gpa_event_records,
        )
        entropy_cache_num = sum(len(entropy_cache[key]) for key in entropy_cache)
        gpa_cache_num = sum(len(gpa_cache[key]) for key in gpa_cache)
        num_classes = clip_logits.size(1)
        full_num = shot_capacity * num_classes

        if entropy_cache_num == full_num and stats["build_entropy_cache_full_once"] == 0:
            print("*" * 10, "Building [global entropy] cache is full; continue for GPA candidate initialization.", "*" * 10, "\n")
            stats["build_entropy_cache_full_once"] += 1

        if gpa_cache_num == full_num:
            print("*" * 10, "Building [E3-V3-C1-Ub GPA/local] cache is full.", "*" * 10, "\n")
            break

    # C1-Ub 严格要求候选池达到 2K 才初始化，不在 build 结束时用 K 个候选做补充初始化。
    # 未达到 2K 的类别会把候选池状态带入 test 阶段继续收集。
    return entropy_cache, gpa_cache, gpa_local_cache, stats, gpa_event_records, gpa_candidate_pool


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
    entropy_cache, gpa_cache, gpa_local_cache, build_stats, gpa_event_records, gpa_candidate_pool = build_cache_in_advance(
        args, test_loader, lm3d_model, clip_weights, pos_cfg["shot_capacity"]
    )

    print("[E3-V3-C1-Ub] len(entropy_cache):", len(entropy_cache))
    print("[E3-V3-C1-Ub] len(gpa_cache):", len(gpa_cache))
    print("[E3-V3-C1-Ub] len(gpa_local_cache):", len(gpa_local_cache))
    print("[E3-V3-C1-Ub] entropy cache total:", sum(len(v) for v in entropy_cache.values()))
    print("[E3-V3-C1-Ub] gpa cache total:", sum(len(v) for v in gpa_cache.values()))
    print("[E3-V3-C1-Ub] gpa local cache total:", sum(len(v) for v in gpa_local_cache.values()))
    print("[E3-V3-C1-Ub] gpa candidate pool total:", sum(len(v) for v in gpa_candidate_pool.values()))

    neg_cache = {}
    gpa_cache_stats = defaultdict(int)
    for k, v in build_stats.items():
        gpa_cache_stats[k] += v

    # 注意：必须继承 build 阶段已经初始化好的 GPA global cache。
    # 不能把 gpa_cache 重置为空，否则 GPA-local-cache 会与 GPA-Cache 脱节，
    # 且候选池初始化得到的样本不会真正参与后续 GPA-Center 更新。
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
                entropy_cache,
                gpa_cache,
                gpa_local_cache,
                gpa_candidate_pool,
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
            print("---- E3-V3-C1-Ub test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    final_acc = sum(accuracies) / len(accuracies)
    print("---- ***Final*** E3-V3-C1-Ub test accuracy: {:.2f}. ----\n".format(final_acc))

    _save_gpa_stats(args, gpa_cache_stats, entropy_cache, gpa_cache, gpa_local_cache, final_acc, gpa_event_records)

    return final_acc
