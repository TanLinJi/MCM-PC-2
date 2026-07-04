"""
E7-A: Entropy-Energy-Alignment Multi-Cache for Point-Cache.

设计文档：docs/experiments/E7_entropy_energy_alignment_multicache/README.md

核心思想：
1. 不使用 Point-Cache 的局部缓存（local cache），只使用全局点云特征 pc_feats；
2. 零样本分类器使用 manual_full 文本原型（clip_weights），构造 S_zs，全程不变；
3. 并行维护熵缓存（entropy cache）和能量缓存（energy cache）；
4. 对齐缓存（alignment cache）后置汇合：仅当同一样本同时被熵缓存和能量缓存接受
   （acceptance event）时，才有资格进入对齐缓存；
5. 对齐缓存与上游缓存之间没有级联删除，进入后只由自身容量和替换逻辑管理；
6. 每个缓存维护自己的 accepted-history 原型分布（diagonal Gaussian），用于替换打分；
7. E1 的 LLM 文本描述只用于各缓存替换时的 text distribution 打分，不进入最终 logits；
8. 最终 logits：S_final = alpha_zs * S_zs + alpha_H * S_H + alpha_E * S_E + alpha_A * S_A。

本文件不修改任何已有 E4/E5 文件，是 E7 的独立实现。
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from collections import defaultdict

import wandb
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import *  # noqa: F401,F403

E7_VARIANT_NAME = "E7-A-entropy-energy-alignment-multi-cache"


# ============================================================
# E7 超参数（环境变量读取，独立于 E4/E5）
# ============================================================

ENTROPY_CAPACITY = int(os.environ.get("E7_ENTROPY_CAPACITY", "5"))
ENERGY_CAPACITY = int(os.environ.get("E7_ENERGY_CAPACITY", "5"))
ALIGNMENT_CAPACITY = int(os.environ.get("E7_ALIGNMENT_CAPACITY", "3"))

ALPHA_ZS = float(os.environ.get("E7_ALPHA_ZS", "1.0"))
ALPHA_ENTROPY = float(os.environ.get("E7_ALPHA_ENTROPY", "2.0"))
ALPHA_ENERGY = float(os.environ.get("E7_ALPHA_ENERGY", "2.0"))
ALPHA_ALIGNMENT = float(os.environ.get("E7_ALPHA_ALIGNMENT", "2.0"))

BETA_ENTROPY = float(os.environ.get("E7_BETA_ENTROPY", "3.0"))
BETA_ENERGY = float(os.environ.get("E7_BETA_ENERGY", "3.0"))
BETA_ALIGNMENT = float(os.environ.get("E7_BETA_ALIGNMENT", "3.0"))

# 分布打分参数（沿用 E4 的对角高斯式打分）
DIST_EPS = float(os.environ.get("E7_DIST_EPS", "1e-4"))
DIST_MIN_VAR = float(os.environ.get("E7_DIST_MIN_VAR", "1e-4"))
TEXT_DIST_EPS = float(os.environ.get("E7_TEXT_DIST_EPS", str(DIST_EPS)))
TEXT_DIST_MIN_VAR = float(os.environ.get("E7_TEXT_DIST_MIN_VAR", str(DIST_MIN_VAR)))

# 替换打分中的文本分布权重（沿用 02_9_2 的 0.15）
TEXT_SCORE_WEIGHT = float(os.environ.get("E7_TEXT_SCORE_WEIGHT", "0.15"))

# 在线分数归一化（沿用 02_9_2 的 running_zscore）
SCORE_NORM_MODE = os.environ.get("E7_SCORE_NORM_MODE", "running_zscore").strip().lower()
SCORE_NORM_MIN_COUNT = int(os.environ.get("E7_SCORE_NORM_MIN_COUNT", "8"))
SCORE_NORM_EPS = float(os.environ.get("E7_SCORE_NORM_EPS", "1e-6"))
SCORE_NORM_CLIP = float(os.environ.get("E7_SCORE_NORM_CLIP", "0"))

# 对齐缓存最小可用门控：未达到前不参与最终 logits
ALIGNMENT_MIN_TOTAL = int(os.environ.get("E7_ALIGNMENT_MIN_TOTAL", "0"))

GATED_FUSION_ENABLED = os.environ.get("E7_GATED_FUSION", "0") == "1"
GATE_AGREE = float(os.environ.get("E7_GATE_AGREE", "1.0"))
GATE_CORRECT = float(os.environ.get("E7_GATE_CORRECT", "1.2"))
GATE_FALLBACK = float(os.environ.get("E7_GATE_FALLBACK", "0.2"))
GATE_ZS_MARGIN_MAX = float(os.environ.get("E7_GATE_ZS_MARGIN_MAX", "5.0"))
GATE_CACHE_MARGIN_MIN = float(os.environ.get("E7_GATE_CACHE_MARGIN_MIN", "1.0"))
GATE_SIM_MIN = float(os.environ.get("E7_GATE_SIM_MIN", "0.60"))

if SCORE_NORM_MODE not in {"none", "running_zscore"}:
    raise ValueError(f"Unsupported E7_SCORE_NORM_MODE: {SCORE_NORM_MODE}")


def _get_stats_enabled():
    return os.environ.get("GPA_SAVE_STATS", "1") != "0"


# ============================================================
# 基础数值工具
# ============================================================

def _ctrl_value(value):
    """把熵/能量张量或标量转成 python float，用于排序和比较。"""
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def _feature_float(feat):
    return feat.detach().float()


def _feature_key(feat):
    x = feat.detach()
    storage = x.untyped_storage() if hasattr(x, "untyped_storage") else x.storage()
    return (int(x.data_ptr()), int(storage.data_ptr()), tuple(x.shape), str(x.device))


def _compute_energy(clip_logits):
    """energy = -logsumexp(clip_logits)，越低越自信。"""
    return float((-torch.logsumexp(clip_logits, dim=1)).detach().float().cpu().item())


# ============================================================
# 在线分数归一化（每个缓存独立维护一份 state）
# ============================================================

def _make_score_norm_state():
    return {
        "visual": {"count": 0, "mean": 0.0, "m2": 0.0},
        "text": {"count": 0, "mean": 0.0, "m2": 0.0},
    }


def _running_std(entry):
    count = int(entry["count"])
    if count < 2:
        return None
    return (float(entry["m2"]) / float(count - 1)) ** 0.5


def _score_for_joint(score_norm_state, modality, raw_score):
    if raw_score is None:
        return None, False
    if SCORE_NORM_MODE == "none" or score_norm_state is None:
        return float(raw_score), False

    entry = score_norm_state[modality]
    count = int(entry["count"])
    std = _running_std(entry)
    if count < SCORE_NORM_MIN_COUNT or std is None or std < SCORE_NORM_EPS:
        return float(raw_score), False

    score = (float(raw_score) - float(entry["mean"])) / (std + SCORE_NORM_EPS)
    if SCORE_NORM_CLIP > 0:
        score = max(min(score, SCORE_NORM_CLIP), -SCORE_NORM_CLIP)
    return float(score), True


def _score_norm_ready(score_norm_state, modalities):
    if SCORE_NORM_MODE == "none" or score_norm_state is None:
        return False
    for modality in modalities:
        entry = score_norm_state[modality]
        count = int(entry["count"])
        std = _running_std(entry)
        if count < SCORE_NORM_MIN_COUNT or std is None or std < SCORE_NORM_EPS:
            return False
    return True


def _update_running_score(entry, value):
    value = float(value)
    count_old = int(entry["count"])
    count_new = count_old + 1

    if count_old == 0:
        entry["count"] = 1
        entry["mean"] = value
        entry["m2"] = 0.0
        return

    delta = value - float(entry["mean"])
    mean_new = float(entry["mean"]) + delta / float(count_new)
    delta2 = value - mean_new

    entry["count"] = count_new
    entry["mean"] = mean_new
    entry["m2"] = float(entry["m2"]) + delta * delta2


def _update_score_norm_state(score_norm_state, score):
    if SCORE_NORM_MODE == "none" or score_norm_state is None or score is None:
        return 0
    updates = 0
    for modality in ("visual", "text"):
        value = score.get(modality)
        if value is None:
            continue
        _update_running_score(score_norm_state[modality], value)
        updates += 1
    return updates


def _summarize_score_norm_state(score_norm_state):
    if score_norm_state is None:
        return {}
    summary = {}
    for modality, entry in score_norm_state.items():
        std = _running_std(entry)
        summary[modality] = {
            "count": int(entry["count"]),
            "mean": float(entry["mean"]),
            "std": None if std is None else float(std),
        }
    return summary


def _record_logit_norm(stats, phase, name, logits):
    value = float(torch.norm(logits.detach().float(), p=2).cpu().item())
    stats[f"{phase}_{name}_logits_norm_sum"] += value
    stats[f"{phase}_{name}_logits_norm_count"] += 1
    max_key = f"{phase}_{name}_logits_norm_max"
    stats[max_key] = max(float(stats.get(max_key, 0.0)), value)


def _finalize_logit_norm_stats(stats, phase):
    names = [
        "zs",
        "entropy",
        "energy",
        "alignment",
        "positive_cache_total",
        "final",
    ]
    for name in names:
        count_key = f"{phase}_{name}_logits_norm_count"
        sum_key = f"{phase}_{name}_logits_norm_sum"
        mean_key = f"{phase}_{name}_logits_norm_mean"
        count = int(stats.get(count_key, 0))
        if count > 0:
            stats[mean_key] = float(stats.get(sum_key, 0.0)) / float(count)


def _top1_margin(logits):
    topk = logits.detach().float().topk(2, dim=1).values
    return float((topk[:, 0] - topk[:, 1]).cpu().item())


def _record_scalar_stat(stats, phase, name, value):
    value = float(value)
    stats[f"{phase}_{name}_sum"] += value
    stats[f"{phase}_{name}_count"] += 1
    max_key = f"{phase}_{name}_max"
    stats[max_key] = max(float(stats.get(max_key, 0.0)), value)


def _finalize_scalar_stats(stats, phase):
    names = [
        "gate_value",
        "cache_similarity",
        "zs_margin",
        "cache_margin",
    ]
    for name in names:
        count_key = f"{phase}_{name}_count"
        sum_key = f"{phase}_{name}_sum"
        mean_key = f"{phase}_{name}_mean"
        count = int(stats.get(count_key, 0))
        if count > 0:
            stats[mean_key] = float(stats.get(sum_key, 0.0)) / float(count)


def _finalize_alignment_zs_correctness_stats(stats, phase):
    """
    离线诊断：累计统计进入对齐缓存相关样本的 zero-shot 伪标签正确率。

    这些指标只用于实验分析；真实标签不参与 TTA 的缓存更新、替换或最终预测。
    """
    for name in ("alignment_eligible_zs", "alignment_entered_zs"):
        total = int(stats.get(f"{phase}_{name}_total", 0))
        correct = int(stats.get(f"{phase}_{name}_correct", 0))
        if total > 0:
            stats[f"{phase}_{name}_acc"] = float(correct) / float(total) * 100.0


def _finalize_alignment_zs_correctness_all_stats(stats):
    """汇总 build + test 的历史累计正确率。"""
    for name in ("alignment_eligible_zs", "alignment_entered_zs"):
        total = int(stats.get(f"build_{name}_total", 0)) + int(stats.get(f"test_{name}_total", 0))
        correct = int(stats.get(f"build_{name}_correct", 0)) + int(stats.get(f"test_{name}_correct", 0))
        stats[f"all_{name}_total"] = total
        stats[f"all_{name}_correct"] = correct
        if total > 0:
            stats[f"all_{name}_acc"] = float(correct) / float(total) * 100.0


def _record_alignment_zs_correctness(stats, phase, entered_entropy, entered_energy, entered_alignment, zs_correct):
    if entered_entropy and entered_energy:
        stats[f"{phase}_alignment_eligible_zs_total"] += 1
        stats[f"{phase}_alignment_eligible_zs_correct"] += int(zs_correct)
    if entered_alignment:
        stats[f"{phase}_alignment_entered_zs_total"] += 1
        stats[f"{phase}_alignment_entered_zs_correct"] += int(zs_correct)


# ============================================================
# Accepted-history 视觉分布（每个缓存维护一份）
# ============================================================

def _update_history_distribution(history_dist, pred, feat, stats=None, phase=None, cache_tag=None):
    """
    累计被该缓存接受过的可信视觉样本（accepted-history）。

    只有样本成功进入或替换该缓存后才调用这里。
    同一 tensor 在同一缓存历史中只计一次。
    """
    pred = int(pred)
    key = _feature_key(feat)

    if pred not in history_dist:
        history_dist[pred] = {"count": 0, "mean": None, "m2": None, "seen": set()}

    entry = history_dist[pred]
    if key in entry["seen"]:
        return False

    x = _feature_float(feat)
    entry["seen"].add(key)

    if int(entry["count"]) == 0:
        entry["count"] = 1
        entry["mean"] = x.clone()
        entry["m2"] = torch.zeros_like(x)
    else:
        count_old = int(entry["count"])
        count_new = count_old + 1
        mean_old = entry["mean"]
        delta = x - mean_old
        mean_new = mean_old + delta / float(count_new)
        delta2 = x - mean_new

        entry["m2"] = entry["m2"] + delta * delta2
        entry["mean"] = mean_new
        entry["count"] = count_new

    if stats is not None and phase is not None and cache_tag is not None:
        stats[f"{phase}_{cache_tag}_history_update"] += 1

    return True


def _history_distribution(history_dist, pred):
    pred = int(pred)
    if pred not in history_dist:
        return None
    entry = history_dist[pred]
    count = int(entry["count"])
    if count < 2:
        return None
    var = (entry["m2"] / float(max(count - 1, 1))).clamp_min(DIST_MIN_VAR)
    return {"count": count, "mean": entry["mean"], "var": var}


def _text_distribution(text_dist, pred, ref_feat):
    if text_dist is None:
        return None
    pred = int(pred)
    if pred not in text_dist:
        return None
    entry = text_dist[pred]
    return {
        "count": int(entry["count"]),
        "mean": entry["mean"].to(device=ref_feat.device, dtype=ref_feat.dtype),
        "var": entry["var"].to(device=ref_feat.device, dtype=ref_feat.dtype),
    }


def _distribution_score_from_entry(entry, feat, eps):
    if entry is None or int(entry["count"]) < 2:
        return None
    x = _feature_float(feat).to(device=entry["mean"].device, dtype=entry["mean"].dtype)
    raw = torch.mean(((x - entry["mean"]) ** 2) / (entry["var"] + eps))
    return float((-raw).detach().cpu().item())


def _joint_distribution_score(history_dist, text_dist, pred, feat, score_norm_state):
    """
    计算缓存替换用的 text-visual joint score。

    visual_score 来自该缓存自己的 accepted-history 分布；
    text_score 来自固定的 prompt-level text distribution。
    """
    visual_entry = _history_distribution(history_dist, pred)
    text_entry = _text_distribution(text_dist, pred, feat)

    visual_score = _distribution_score_from_entry(visual_entry, feat, DIST_EPS)
    text_score = _distribution_score_from_entry(text_entry, feat, TEXT_DIST_EPS)

    if visual_score is None:
        return None

    if text_score is None:
        use_norm = _score_norm_ready(score_norm_state, ("visual",))
        visual_joint, visual_norm = (
            _score_for_joint(score_norm_state, "visual", visual_score)
            if use_norm else (float(visual_score), False)
        )
        text_joint = None
        text_norm = False
        joint_score = visual_joint
    else:
        use_norm = _score_norm_ready(score_norm_state, ("visual", "text"))
        if use_norm:
            visual_joint, visual_norm = _score_for_joint(score_norm_state, "visual", visual_score)
            text_joint, text_norm = _score_for_joint(score_norm_state, "text", text_score)
        else:
            visual_joint, visual_norm = float(visual_score), False
            text_joint, text_norm = float(text_score), False
        joint_score = visual_joint + TEXT_SCORE_WEIGHT * text_joint

    return {
        "joint": float(joint_score),
        "visual": float(visual_score),
        "text": None if text_score is None else float(text_score),
        "visual_for_joint": float(visual_joint),
        "text_for_joint": None if text_joint is None else float(text_joint),
        "visual_score_normalized": bool(visual_norm),
        "text_score_normalized": bool(text_norm),
        "visual_count": 0 if visual_entry is None else int(visual_entry["count"]),
        "text_count": 0 if text_entry is None else int(text_entry["count"]),
    }


def _observe_score_norm(score_norm_state, curr_score, worst_score, stats, phase, cache_tag):
    norm_updates = _update_score_norm_state(score_norm_state, curr_score)
    norm_updates += _update_score_norm_state(score_norm_state, worst_score)
    if norm_updates:
        stats[f"{phase}_{cache_tag}_score_norm_update"] += norm_updates
        stats[f"{phase}_{cache_tag}_score_norm_observed_pairs"] += 1


# ============================================================
# 缓存项约定
#   entropy item : {"feat": (1,D), "label": int, "ctrl": entropy_float}
#   energy  item : {"feat": (1,D), "label": int, "ctrl": energy_float}
#   align   item : {"feat": (1,D), "label": int}
# ============================================================

def _sort_by_ctrl_ascending(cache, pred):
    """按控制量升序排序，最差（最高熵/能量）的样本排在末尾。"""
    cache[pred] = sorted(cache[pred], key=lambda it: it["ctrl"])


def _update_ctrl_cache(
    cache,
    history_dist,
    text_dist,
    score_norm_state,
    pred,
    item,
    capacity,
    stats,
    phase,
    cache_tag,
):
    """
    熵缓存 / 能量缓存的通用更新。

    返回 accepted: 当前样本是否成功进入或替换该缓存（acceptance event）。

    规则（设计文档 9.2 / 9.3）：
        未满 -> 直接加入；
        已满 -> 新样本控制量低于当前最差，且 joint_score 更高，才替换最差样本。
    """
    if pred not in cache:
        cache[pred] = []

    if len(cache[pred]) < capacity:
        cache[pred].append(item)
        _sort_by_ctrl_ascending(cache, pred)
        stats[f"{phase}_{cache_tag}_add_not_full"] += 1
        _update_history_distribution(history_dist, pred, item["feat"], stats, phase, cache_tag)
        return True

    worst_item = cache[pred][-1]
    worst_ctrl = worst_item["ctrl"]
    curr_ctrl = item["ctrl"]

    if curr_ctrl >= worst_ctrl:
        stats[f"{phase}_{cache_tag}_reject_ctrl"] += 1
        return False

    curr_score = _joint_distribution_score(history_dist, text_dist, pred, item["feat"], score_norm_state)
    worst_score = _joint_distribution_score(history_dist, text_dist, pred, worst_item["feat"], score_norm_state)

    if curr_score is None or worst_score is None:
        # 分布尚不可用时，退回纯控制量门控：控制量更低即替换。
        cache[pred][-1] = item
        _sort_by_ctrl_ascending(cache, pred)
        stats[f"{phase}_{cache_tag}_replace_ctrl_only"] += 1
        _update_history_distribution(history_dist, pred, item["feat"], stats, phase, cache_tag)
        return True

    _observe_score_norm(score_norm_state, curr_score, worst_score, stats, phase, cache_tag)

    if curr_score["joint"] > worst_score["joint"]:
        cache[pred][-1] = item
        _sort_by_ctrl_ascending(cache, pred)
        stats[f"{phase}_{cache_tag}_replace_joint"] += 1
        _update_history_distribution(history_dist, pred, item["feat"], stats, phase, cache_tag)
        return True

    stats[f"{phase}_{cache_tag}_reject_joint"] += 1
    return False


def _update_alignment_cache(
    align_cache,
    align_history_dist,
    text_dist,
    score_norm_state,
    pred,
    item,
    capacity,
    stats,
    phase,
):
    """
    对齐缓存更新（设计文档 9.4）。

    只有当前样本同时被熵缓存和能量缓存接受时才会调用本函数。

    规则：
        未满 -> 直接加入；
        已满 -> 新样本比该类对齐缓存中分布匹配最差的样本更符合 joint 分布，才替换它。

    非级联删除：对齐缓存条目独立于熵缓存/能量缓存，上游替换不影响这里。
    """
    cache_tag = "alignment"
    if pred not in align_cache:
        align_cache[pred] = []

    if len(align_cache[pred]) < capacity:
        align_cache[pred].append(item)
        stats[f"{phase}_{cache_tag}_add_not_full"] += 1
        _update_history_distribution(align_history_dist, pred, item["feat"], stats, phase, cache_tag)
        return True

    curr_score = _joint_distribution_score(align_history_dist, text_dist, pred, item["feat"], score_norm_state)
    if curr_score is None:
        stats[f"{phase}_{cache_tag}_reject_no_distribution"] += 1
        return False

    # 找到当前类对齐缓存中 joint 最差（最低）的已有样本。
    worst_idx = None
    worst_score = None
    for idx, existing in enumerate(align_cache[pred]):
        existing_score = _joint_distribution_score(
            align_history_dist, text_dist, pred, existing["feat"], score_norm_state
        )
        if existing_score is None:
            continue
        if worst_score is None or existing_score["joint"] < worst_score["joint"]:
            worst_idx = idx
            worst_score = existing_score

    if worst_idx is None or worst_score is None:
        stats[f"{phase}_{cache_tag}_reject_no_distribution"] += 1
        return False

    _observe_score_norm(score_norm_state, curr_score, worst_score, stats, phase, cache_tag)

    if curr_score["joint"] > worst_score["joint"]:
        align_cache[pred][worst_idx] = item
        stats[f"{phase}_{cache_tag}_replace_joint"] += 1
        _update_history_distribution(align_history_dist, pred, item["feat"], stats, phase, cache_tag)
        return True

    stats[f"{phase}_{cache_tag}_reject_joint"] += 1
    return False


def _process_sample_caches(
    caches,
    history_dists,
    score_norm_states,
    text_dist,
    pred,
    pc_feats,
    entropy_value,
    energy_value,
    capacities,
    stats,
    phase,
):
    """
    对单个样本并行更新熵缓存和能量缓存，再做同步判断推导对齐缓存。

    返回 (entered_entropy, entered_energy, entered_alignment)。

    entered_alignment 表示样本实际加入或替换了对齐缓存；如果只是同时通过
    熵缓存和能量缓存、但被对齐缓存自身规则拒绝，则为 False。
    """
    entropy_item = {"feat": pc_feats, "label": int(pred), "ctrl": _ctrl_value(entropy_value)}
    energy_item = {"feat": pc_feats, "label": int(pred), "ctrl": _ctrl_value(energy_value)}

    entered_entropy = _update_ctrl_cache(
        caches["entropy"], history_dists["entropy"], text_dist, score_norm_states["entropy"],
        pred, entropy_item, capacities["entropy"], stats, phase, "entropy",
    )
    entered_energy = _update_ctrl_cache(
        caches["energy"], history_dists["energy"], text_dist, score_norm_states["energy"],
        pred, energy_item, capacities["energy"], stats, phase, "energy",
    )

    entered_alignment = False
    if entered_entropy and entered_energy:
        align_item = {"feat": pc_feats, "label": int(pred)}
        entered_alignment = _update_alignment_cache(
            caches["alignment"], history_dists["alignment"], text_dist, score_norm_states["alignment"],
            pred, align_item, capacities["alignment"], stats, phase,
        )
        stats[f"{phase}_alignment_eligible"] += 1

    return entered_entropy, entered_energy, entered_alignment


# ============================================================
# 缓存得分（相似度加权投票）
# ============================================================

@torch.no_grad()
def compute_cache_vote_logits(pc_feats, cache, alpha, beta, clip_weights):
    """
    S(c) = alpha * sum_i exp(-beta*(1 - feat·feat_i)) * 1[label_i == c]
    """
    cache_keys = []
    cache_values = []

    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            cache_keys.append(item["feat"])
            cache_values.append(item["label"])

    if not cache_keys:
        return torch.zeros_like(pc_feats @ clip_weights)

    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
    cache_values = F.one_hot(
        torch.Tensor(cache_values).to(torch.int64),
        num_classes=clip_weights.size(1),
    ).half().cuda()

    affinity = pc_feats @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits


@torch.no_grad()
def compute_max_cache_similarity(pc_feats, caches):
    cache_keys = []
    for cache in caches:
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item["feat"])

    if not cache_keys:
        return 0.0

    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
    similarity = pc_feats @ cache_keys
    return float(similarity.max().detach().float().cpu().item())


def _compute_gate_value(clip_logits, cache_logits, cache_similarity, stats, phase):
    zs_pred = int(clip_logits.topk(1, dim=1)[1].item())
    cache_pred = int(cache_logits.topk(1, dim=1)[1].item()) if cache_logits.abs().sum() > 0 else -1
    zs_margin = _top1_margin(clip_logits)
    cache_margin = _top1_margin(cache_logits) if cache_logits.abs().sum() > 0 else 0.0

    if cache_pred == zs_pred:
        gate_value = GATE_AGREE
        stats[f"{phase}_gate_agree_count"] += 1
    elif (
        zs_margin <= GATE_ZS_MARGIN_MAX
        and cache_margin >= GATE_CACHE_MARGIN_MIN
        and cache_similarity >= GATE_SIM_MIN
    ):
        gate_value = GATE_CORRECT
        stats[f"{phase}_gate_correct_count"] += 1
    else:
        gate_value = GATE_FALLBACK
        stats[f"{phase}_gate_fallback_count"] += 1

    _record_scalar_stat(stats, phase, "gate_value", gate_value)
    _record_scalar_stat(stats, phase, "cache_similarity", cache_similarity)
    _record_scalar_stat(stats, phase, "zs_margin", zs_margin)
    _record_scalar_stat(stats, phase, "cache_margin", cache_margin)
    return float(gate_value)


# ============================================================
# 缓存构建与测试
# ============================================================

@torch.no_grad()
def build_cache_in_advance(args, test_loader, lm3d_model, clip_weights, capacities, text_dist=None):
    """
    预构建三个缓存，直到熵缓存填满（与 E4/Point-Cache 的 build 触发条件一致）。
    """
    print("*" * 10, "Building E7 entropy/energy/alignment caches ...", "*" * 10, "\n")

    caches = {"entropy": {}, "energy": {}, "alignment": {}}
    history_dists = {"entropy": {}, "energy": {}, "alignment": {}}
    score_norm_states = {
        "entropy": _make_score_norm_state(),
        "energy": _make_score_norm_state(),
        "alignment": _make_score_norm_state(),
    }
    stats = defaultdict(int)

    for pc, target, _, rgb in test_loader:
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        energy_value = _compute_energy(clip_logits)
        zs_correct = int(pred == int(target.detach().cpu().item()))

        entered_entropy, entered_energy, entered_alignment = _process_sample_caches(
            caches, history_dists, score_norm_states, text_dist,
            pred, pc_feats, loss, energy_value, capacities, stats, "build",
        )
        _record_alignment_zs_correctness(
            stats, "build", entered_entropy, entered_energy, entered_alignment, zs_correct,
        )

        cache_num = sum(len(caches["entropy"][key]) for key in caches["entropy"])
        num_classes = clip_logits.size(1)
        full_num = capacities["entropy"] * num_classes

        if cache_num == full_num:
            print("*" * 10, "E7 entropy cache is full. Build done.", "*" * 10, "\n")
            break

    return caches, history_dists, score_norm_states, stats


def _summarize_cache(cache):
    return {str(k): len(v) for k, v in sorted(cache.items(), key=lambda kv: kv[0])}


def _summarize_history_distribution(history_dist):
    summary = {}
    for c in sorted(history_dist.keys()):
        entry = _history_distribution(history_dist, c)
        if entry is not None:
            var = entry["var"].detach().float().cpu()
            summary[str(c)] = {
                "count": int(entry["count"]),
                "var_mean": float(var.mean().item()),
                "var_min": float(var.min().item()),
                "var_max": float(var.max().item()),
            }
        else:
            summary[str(c)] = {
                "count": int(history_dist[c]["count"]),
                "var_mean": None, "var_min": None, "var_max": None,
            }
    return summary


def _save_e7_stats(args, stats, caches, history_dists, score_norm_states, acc=None):
    if not _get_stats_enabled():
        return

    result_root = getattr(args, "baseline_result_root", None)
    exp_id = getattr(args, "baseline_exp_id", None)
    if not result_root or not exp_id:
        return

    out_dir = Path(result_root) / exp_id / "e7_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exp_id": exp_id,
        "cor_type": getattr(args, "cor_type", None),
        "e7_variant": E7_VARIANT_NAME,
        "uses_local_cache": False,
        "text_distribution_role": "cache_replacement_scoring_only",
        "final_text_classifier_source": getattr(args, "prompt_source", None),
        "capacities": {
            "entropy": ENTROPY_CAPACITY,
            "energy": ENERGY_CAPACITY,
            "alignment": ALIGNMENT_CAPACITY,
        },
        "alphas": {
            "zs": ALPHA_ZS, "entropy": ALPHA_ENTROPY,
            "energy": ALPHA_ENERGY, "alignment": ALPHA_ALIGNMENT,
        },
        "betas": {
            "entropy": BETA_ENTROPY, "energy": BETA_ENERGY, "alignment": BETA_ALIGNMENT,
        },
        "e7_text_score_weight": float(TEXT_SCORE_WEIGHT),
        "e7_score_norm_mode": SCORE_NORM_MODE,
        "e7_dist_eps": float(DIST_EPS),
        "e7_dist_min_var": float(DIST_MIN_VAR),
        "alignment_min_total": int(ALIGNMENT_MIN_TOTAL),
        "gated_fusion": {
            "enabled": bool(GATED_FUSION_ENABLED),
            "gate_agree": float(GATE_AGREE),
            "gate_correct": float(GATE_CORRECT),
            "gate_fallback": float(GATE_FALLBACK),
            "zs_margin_max": float(GATE_ZS_MARGIN_MAX),
            "cache_margin_min": float(GATE_CACHE_MARGIN_MIN),
            "cache_similarity_min": float(GATE_SIM_MIN),
        },
        "final_acc": acc,
        "stats": dict(stats),
        "entropy_cache_class_counts": _summarize_cache(caches["entropy"]),
        "energy_cache_class_counts": _summarize_cache(caches["energy"]),
        "alignment_cache_class_counts": _summarize_cache(caches["alignment"]),
        "entropy_cache_total": int(sum(len(v) for v in caches["entropy"].values())),
        "energy_cache_total": int(sum(len(v) for v in caches["energy"].values())),
        "alignment_cache_total": int(sum(len(v) for v in caches["alignment"].values())),
        "entropy_history_summary": _summarize_history_distribution(history_dists["entropy"]),
        "energy_history_summary": _summarize_history_distribution(history_dists["energy"]),
        "alignment_history_summary": _summarize_history_distribution(history_dists["alignment"]),
        "score_norm_summary": {
            k: _summarize_score_norm_state(v) for k, v in score_norm_states.items()
        },
    }

    filename = f"{getattr(args, 'cor_type', 'unknown')}_e7_stats.json"
    with (out_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[E7-A] Saved E7 stats to {out_dir / filename}")


@torch.no_grad()
def run_test_tda(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights, text_dist=None):
    """
    E7-A test-time adaptation.

    最终 logits：
        S_final = alpha_zs * S_zs + alpha_H * S_H + alpha_E * S_E + alpha_A * S_A
    其中 S_zs = clip_logits（manual_full 文本原型，100x 缩放），不被文本分布替换。
    """
    capacities = {
        "entropy": ENTROPY_CAPACITY,
        "energy": ENERGY_CAPACITY,
        "alignment": ALIGNMENT_CAPACITY,
    }

    caches, history_dists, score_norm_states, build_stats = build_cache_in_advance(
        args, test_loader, lm3d_model, clip_weights, capacities, text_dist=text_dist
    )

    print("[E7-A] entropy cache total:", sum(len(v) for v in caches["entropy"].values()))
    print("[E7-A] energy cache total:", sum(len(v) for v in caches["energy"].values()))
    print("[E7-A] alignment cache total:", sum(len(v) for v in caches["alignment"].values()))
    print("[E7-A] text distribution classes:", 0 if text_dist is None else len(text_dist))
    print("[E7-A] score norm mode:", SCORE_NORM_MODE)
    print("[E7-A] alphas: zs={} H={} E={} A={}".format(ALPHA_ZS, ALPHA_ENTROPY, ALPHA_ENERGY, ALPHA_ALIGNMENT))
    print("[E7-A] gated fusion enabled:", GATED_FUSION_ENABLED)
    if GATED_FUSION_ENABLED:
        print(
            "[E7-A] gates: agree={} correct={} fallback={} zs_margin_max={} "
            "cache_margin_min={} sim_min={}".format(
                GATE_AGREE, GATE_CORRECT, GATE_FALLBACK,
                GATE_ZS_MARGIN_MAX, GATE_CACHE_MARGIN_MIN, GATE_SIM_MIN,
            )
        )

    stats = defaultdict(int)
    for k, v in build_stats.items():
        stats[k] += v

    accuracies = []
    zs_changed = 0
    total_seen = 0
    cache_agreement_HE = 0

    for i, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        target = target.cuda()
        energy_value = _compute_energy(clip_logits)
        zs_correct = int(pred == int(target.detach().cpu().item()))

        # 先更新缓存（沿用 Point-Cache/E4 的 update-then-logits 约定）。
        entered_entropy, entered_energy, entered_alignment = _process_sample_caches(
            caches, history_dists, score_norm_states, text_dist,
            pred, pc_feats, loss, energy_value, capacities, stats, "test",
        )
        _record_alignment_zs_correctness(
            stats, "test", entered_entropy, entered_energy, entered_alignment, zs_correct,
        )

        s_h = compute_cache_vote_logits(pc_feats, caches["entropy"], ALPHA_ENTROPY, BETA_ENTROPY, clip_weights)
        s_e = compute_cache_vote_logits(pc_feats, caches["energy"], ALPHA_ENERGY, BETA_ENERGY, clip_weights)

        align_total = sum(len(v) for v in caches["alignment"].values())
        s_a = torch.zeros_like(clip_logits)
        if align_total > ALIGNMENT_MIN_TOTAL:
            s_a = compute_cache_vote_logits(pc_feats, caches["alignment"], ALPHA_ALIGNMENT, BETA_ALIGNMENT, clip_weights)

        positive_cache_logits = s_h + s_e + s_a
        if GATED_FUSION_ENABLED:
            cache_similarity = compute_max_cache_similarity(
                pc_feats,
                [caches["entropy"], caches["energy"], caches["alignment"]],
            )
            gate_value = _compute_gate_value(
                ALPHA_ZS * clip_logits,
                positive_cache_logits,
                cache_similarity,
                stats,
                "test",
            )
            final_logits = ALPHA_ZS * clip_logits.clone() + gate_value * positive_cache_logits
        else:
            final_logits = ALPHA_ZS * clip_logits.clone() + positive_cache_logits

        _record_logit_norm(stats, "test", "zs", ALPHA_ZS * clip_logits)
        _record_logit_norm(stats, "test", "entropy", s_h)
        _record_logit_norm(stats, "test", "energy", s_e)
        _record_logit_norm(stats, "test", "alignment", s_a)
        _record_logit_norm(stats, "test", "positive_cache_total", positive_cache_logits)
        _record_logit_norm(stats, "test", "final", final_logits)

        # 诊断：熵缓存和能量缓存对当前样本的投票预测是否一致。
        h_pred = int(s_h.topk(1, dim=1)[1].item()) if s_h.abs().sum() > 0 else -1
        e_pred = int(s_e.topk(1, dim=1)[1].item()) if s_e.abs().sum() > 0 else -2
        if h_pred == e_pred:
            cache_agreement_HE += 1

        final_pred = int(final_logits.topk(1, dim=1)[1].item())
        if final_pred != int(pred):
            zs_changed += 1
        total_seen += 1

        acc = cls_acc(final_logits, target)
        accuracies.append(acc)
        wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

        if i % args.print_freq == 0:
            print("---- E7-A test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    final_acc = sum(accuracies) / len(accuracies)
    print("---- ***Final*** E7-A test accuracy: {:.2f}. ----\n".format(final_acc))

    stats["test_zs_vs_final_pred_change"] = int(zs_changed)
    stats["test_total_seen"] = int(total_seen)
    stats["test_cache_agreement_HE"] = int(cache_agreement_HE)
    _finalize_logit_norm_stats(stats, "test")
    _finalize_scalar_stats(stats, "test")
    _finalize_alignment_zs_correctness_stats(stats, "build")
    _finalize_alignment_zs_correctness_stats(stats, "test")
    _finalize_alignment_zs_correctness_all_stats(stats)

    _save_e7_stats(args, stats, caches, history_dists, score_norm_states, final_acc)

    return final_acc
