"""
E7-A4-B1: Candidate-Pool Alignment-Core Cache with Cache-Norm Clipping.

设计文档：
docs/experiments/E7_entropy_energy_alignment_multicache/E7_A_versions/A4_B1_cache_norm_clip.md

核心约束：
1. 免训练测试时适应（training-free TTA），不更新模型参数；
2. 预测端使用 manual_full 文本原型，E1 LLM 描述只作为 text distribution；
3. 候选池不参与最终得分，不维护正式分布；
4. 对齐核心缓存是主可信缓存；
5. 熵缓存和能量缓存只接收已经进入/替换对齐核心缓存的样本；
6. 当前样本先用旧缓存打分，再更新缓存；若进入真正缓存，再用新缓存打分并平均；
7. 对 S_cache = S_A + S_H + S_E 做 L2 范数裁剪，第一版 cap=20。
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import *  # noqa: F401,F403

from runners.E7_entropy_energy_alignment_multicache.model_e7_a_entropy_energy_alignment_cache import (
    _compute_energy,
    _ctrl_value,
    _feature_key,
    _finalize_logit_norm_stats,
    _history_distribution,
    _joint_distribution_score,
    _make_score_norm_state,
    _observe_score_norm,
    _record_logit_norm,
    _summarize_history_distribution,
    _summarize_score_norm_state,
    _top1_margin,
    _update_history_distribution,
    compute_cache_vote_logits,
)

E7_VARIANT_NAME = "E7-A4-B1-cache-norm-clip"


# ============================================================
# A4 超参数
# ============================================================

CANDIDATE_CAPACITY = int(os.environ.get("E7_CANDIDATE_CAPACITY", "8"))
ALIGNMENT_CAPACITY = int(os.environ.get("E7_ALIGNMENT_CAPACITY", "4"))
ENTROPY_CAPACITY = int(os.environ.get("E7_ENTROPY_CAPACITY", "3"))
ENERGY_CAPACITY = int(os.environ.get("E7_ENERGY_CAPACITY", "3"))

ALPHA_ZS = float(os.environ.get("E7_ALPHA_ZS", "1.0"))
ALPHA_ALIGNMENT = float(os.environ.get("E7_ALPHA_ALIGNMENT", "2.0"))
ALPHA_ENTROPY = float(os.environ.get("E7_ALPHA_ENTROPY", "2.0"))
ALPHA_ENERGY = float(os.environ.get("E7_ALPHA_ENERGY", "2.0"))

BETA_ALIGNMENT = float(os.environ.get("E7_BETA_ALIGNMENT", "3.0"))
BETA_ENTROPY = float(os.environ.get("E7_BETA_ENTROPY", "3.0"))
BETA_ENERGY = float(os.environ.get("E7_BETA_ENERGY", "3.0"))

ALIGNMENT_MIN_TOTAL = int(os.environ.get("E7_ALIGNMENT_MIN_TOTAL", "0"))
RELIABILITY_EPS = float(os.environ.get("E7_A4_RELIABILITY_EPS", "1e-6"))
SCORE_AVG_OLD_WEIGHT = float(os.environ.get("E7_A4_SCORE_OLD_WEIGHT", "0.5"))
SCORE_AVG_NEW_WEIGHT = float(os.environ.get("E7_A4_SCORE_NEW_WEIGHT", "0.5"))
CACHE_NORM_CAP = float(os.environ.get("E7_CACHE_NORM_CAP", "20.0"))
CACHE_NORM_EPS = float(os.environ.get("E7_CACHE_NORM_EPS", "1e-6"))

TEXT_SCORE_WEIGHT = float(os.environ.get("E7_TEXT_SCORE_WEIGHT", "0.15"))
SCORE_NORM_MODE = os.environ.get("E7_SCORE_NORM_MODE", "running_zscore").strip().lower()


def _get_stats_enabled():
    return os.environ.get("GPA_SAVE_STATS", "1") != "0"


# ============================================================
# 可靠性与候选池工具
# ============================================================

def _finite_or(value, fallback):
    value = float(value)
    return value if math.isfinite(value) else float(fallback)


def _make_candidate_item(pc_feats, pred, entropy_value, energy_value, margin_value):
    return {
        "feat": pc_feats,
        "label": int(pred),
        "entropy": _finite_or(_ctrl_value(entropy_value), 1e9),
        "energy": _finite_or(_ctrl_value(energy_value), 1e9),
        "margin": _finite_or(float(margin_value), -1e9),
    }


def _normalize_metric(values, lower_is_better):
    values = [float(v) for v in values]
    v_min = min(values)
    v_max = max(values)
    denom = v_max - v_min
    if abs(denom) <= RELIABILITY_EPS:
        return [1.0 for _ in values]
    if lower_is_better:
        return [(v_max - v) / (denom + RELIABILITY_EPS) for v in values]
    return [(v - v_min) / (denom + RELIABILITY_EPS) for v in values]


def _reliability_metrics(items):
    """在给定小集合内部计算 q_H/q_E/q_M、瓶颈可靠性 B 和理想点接近度 C。"""
    if not items:
        return []

    q_h = _normalize_metric([it["entropy"] for it in items], lower_is_better=True)
    q_e = _normalize_metric([it["energy"] for it in items], lower_is_better=True)
    q_m = _normalize_metric([it["margin"] for it in items], lower_is_better=False)

    metrics = []
    for h, e, m in zip(q_h, q_e, q_m):
        bottleneck = min(h, e, m)
        d_pos = math.sqrt((1.0 - h) ** 2 + (1.0 - e) ** 2 + (1.0 - m) ** 2)
        d_neg = math.sqrt(h ** 2 + e ** 2 + m ** 2)
        closeness = d_neg / (d_pos + d_neg + RELIABILITY_EPS)
        metrics.append({
            "q_h": float(h),
            "q_e": float(e),
            "q_m": float(m),
            "B": float(bottleneck),
            "C": float(closeness),
        })
    return metrics


def _better_reliability(metric_a, metric_b):
    """字典序比较：先 B，再 C。"""
    if metric_a["B"] > metric_b["B"] + RELIABILITY_EPS:
        return True
    if abs(metric_a["B"] - metric_b["B"]) <= RELIABILITY_EPS:
        return metric_a["C"] > metric_b["C"] + RELIABILITY_EPS
    return False


def _rank_indices_by_reliability(items, new_index=None):
    metrics = _reliability_metrics(items)

    def key(idx):
        # B/C 越大越好；完全并列时保留旧样本，拒绝新样本。
        is_new = 1 if new_index is not None and idx == new_index else 0
        return (-metrics[idx]["B"], -metrics[idx]["C"], is_new, idx)

    return sorted(range(len(items)), key=key), metrics


def _diag_correct(diag_state, item):
    if diag_state is None:
        return None
    return diag_state.get("zs_correct", {}).get(_feature_key(item["feat"]))


def _record_candidate_acceptance(stats, phase, item, metric=None, diag_state=None, diag_values=None):
    stats[f"{phase}_candidate_entered_zs_total"] += 1
    correct = _diag_correct(diag_state, item)
    if correct is not None:
        stats[f"{phase}_candidate_entered_zs_correct"] += int(correct)
    if metric is not None:
        _record_metric_scalar(stats, phase, "candidate_B", metric["B"])
        _record_metric_scalar(stats, phase, "candidate_C", metric["C"])
    _record_item_diag(diag_values, phase, "candidate_history", item, metric, correct)


def _record_alignment_acceptance(stats, phase, item, diag_state=None):
    stats[f"{phase}_alignment_core_entered_zs_total"] += 1
    correct = _diag_correct(diag_state, item)
    if correct is not None:
        stats[f"{phase}_alignment_core_entered_zs_correct"] += int(correct)


def _record_metric_scalar(stats, phase, name, value):
    value = float(value)
    stats[f"{phase}_{name}_sum"] += value
    stats[f"{phase}_{name}_count"] += 1
    stats[f"{phase}_{name}_max"] = max(float(stats.get(f"{phase}_{name}_max", 0.0)), value)


def _append_diag_value(diag_values, key, value):
    if diag_values is None:
        return
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(value):
        diag_values[key].append(value)


def _record_item_diag(diag_values, phase, group, item, metric=None, zs_correct=None, extra=None):
    if diag_values is None:
        return

    prefixes = (f"{phase}_{group}", f"all_{group}")
    values = {
        "entropy": item.get("entropy"),
        "energy": item.get("energy"),
        "margin": item.get("margin"),
    }
    if metric is not None:
        values["B"] = metric.get("B")
        values["C"] = metric.get("C")
    if zs_correct is not None:
        values["zs_correct"] = int(zs_correct)
    if extra:
        values.update(extra)

    for prefix in prefixes:
        for name, value in values.items():
            if value is not None:
                _append_diag_value(diag_values, f"{prefix}_{name}", value)


def _summarize_values(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return {"count": 0}

    values.sort()
    n = len(values)
    mean = sum(values) / float(n)
    var = sum((v - mean) ** 2 for v in values) / float(n) if n > 1 else 0.0

    def quantile(q):
        if n == 1:
            return values[0]
        pos = q * float(n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return values[lo]
        weight = pos - float(lo)
        return values[lo] * (1.0 - weight) + values[hi] * weight

    return {
        "count": int(n),
        "mean": float(mean),
        "std": float(math.sqrt(var)),
        "min": float(values[0]),
        "p25": float(quantile(0.25)),
        "p50": float(quantile(0.50)),
        "p75": float(quantile(0.75)),
        "p90": float(quantile(0.90)),
        "p95": float(quantile(0.95)),
        "max": float(values[-1]),
    }


def _summarize_diag_values(diag_values):
    if not diag_values:
        return {}
    return {key: _summarize_values(values) for key, values in sorted(diag_values.items())}


def _update_candidate_pool(candidate_pools, pred, item, stats, phase, diag_state=None, diag_values=None):
    """
    每类候选池容量为 8。

    未满：直接加入。
    已满：临时形成 9 个样本，按 (B desc, C desc) 保留前 8 个。

    返回 current_kept, current_metric: 当前样本是否成功进入候选池及其 B/C 指标。
    """
    pred = int(pred)
    if pred not in candidate_pools:
        candidate_pools[pred] = []

    pool = candidate_pools[pred]
    if len(pool) < CANDIDATE_CAPACITY:
        pool.append(item)
        metrics = _reliability_metrics(pool)
        metric = metrics[-1] if metrics else None
        stats[f"{phase}_candidate_add_not_full"] += 1
        _record_candidate_acceptance(stats, phase, item, metric, diag_state, diag_values)
        return True, metric

    tmp_items = pool + [item]
    new_index = len(tmp_items) - 1
    ranked, metrics = _rank_indices_by_reliability(tmp_items, new_index=new_index)
    keep_indices = set(ranked[:CANDIDATE_CAPACITY])

    if new_index not in keep_indices:
        stats[f"{phase}_candidate_reject"] += 1
        correct = _diag_correct(diag_state, item)
        _record_item_diag(diag_values, phase, "candidate_rejected", item, metrics[new_index], correct)
        return False, metrics[new_index]

    removed_index = ranked[-1]
    candidate_pools[pred] = [tmp_items[idx] for idx in ranked[:CANDIDATE_CAPACITY]]
    stats[f"{phase}_candidate_replace"] += 1
    stats[f"{phase}_candidate_removed_old"] += int(removed_index != new_index)
    _record_candidate_acceptance(stats, phase, item, metrics[new_index], diag_state, diag_values)
    return True, metrics[new_index]


# ============================================================
# 对齐核心缓存、熵缓存、能量缓存更新
# ============================================================

def _as_alignment_item(item):
    return {
        "feat": item["feat"],
        "label": int(item["label"]),
        "entropy": float(item["entropy"]),
        "energy": float(item["energy"]),
        "margin": float(item["margin"]),
    }


def _sort_ctrl_cache(cache, pred):
    cache[pred] = sorted(cache[pred], key=lambda it: it["ctrl"])


def _initialize_alignment_from_candidates(caches, history_dists, pred, stats, phase, diag_state=None):
    pred = int(pred)
    if pred in caches["alignment"] and len(caches["alignment"][pred]) > 0:
        return []
    if len(caches["candidate"].get(pred, [])) < CANDIDATE_CAPACITY:
        return []

    pool = caches["candidate"][pred]
    ranked, _ = _rank_indices_by_reliability(pool)
    selected = [_as_alignment_item(pool[idx]) for idx in ranked[:ALIGNMENT_CAPACITY]]

    caches["alignment"][pred] = []
    for align_item in selected:
        caches["alignment"][pred].append(align_item)
        _update_history_distribution(
            history_dists["alignment"], pred, align_item["feat"], stats, phase, "alignment_core",
        )
        _record_alignment_acceptance(stats, phase, align_item, diag_state)

    stats[f"{phase}_alignment_core_init_from_candidate"] += len(selected)
    return selected


def _update_alignment_core(
    caches, history_dists, score_norm_states, text_dist, pred, item, stats, phase,
    diag_state=None, diag_values=None,
):
    """
    当前样本成功进入候选池后，才尝试更新对齐核心缓存。

    返回 accepted_items, current_entered_alignment。
    accepted_items 包含本次新进入/替换对齐核心缓存的所有样本；初始化时可能有多个。
    """
    pred = int(pred)
    accepted_items = []

    # 首次候选池满 8 时，用候选池 top4 初始化对齐核心缓存。
    init_items = _initialize_alignment_from_candidates(caches, history_dists, pred, stats, phase, diag_state)
    if init_items:
        accepted_items.extend(init_items)
        current_key = _feature_key(item["feat"])
        current_entered = any(_feature_key(it["feat"]) == current_key for it in init_items)
        return accepted_items, current_entered

    # 候选池未满时，只收集候选样本，不能提前写入对齐核心缓存。
    if len(caches["candidate"].get(pred, [])) < CANDIDATE_CAPACITY:
        stats[f"{phase}_alignment_core_wait_candidate_full"] += 1
        return [], False

    if pred not in caches["alignment"]:
        caches["alignment"][pred] = []

    align_item = _as_alignment_item(item)
    if len(caches["alignment"][pred]) < ALIGNMENT_CAPACITY:
        caches["alignment"][pred].append(align_item)
        _update_history_distribution(
            history_dists["alignment"], pred, align_item["feat"], stats, phase, "alignment_core",
        )
        _record_alignment_acceptance(stats, phase, align_item, diag_state)
        stats[f"{phase}_alignment_core_add_not_full"] += 1
        return [align_item], True

    tmp_items = caches["alignment"][pred] + [align_item]
    metrics = _reliability_metrics(tmp_items)
    new_idx = len(tmp_items) - 1
    new_metric = metrics[new_idx]

    worst_idx = None
    worst_metric = None
    for idx in range(len(caches["alignment"][pred])):
        metric = metrics[idx]
        if worst_metric is None or _better_reliability(worst_metric, metric):
            worst_idx = idx
            worst_metric = metric

    if worst_idx is None or worst_metric is None:
        stats[f"{phase}_alignment_core_reject_reliability"] += 1
        return [], False

    if not _better_reliability(new_metric, worst_metric):
        stats[f"{phase}_alignment_core_reject_reliability"] += 1
        correct = _diag_correct(diag_state, item)
        _record_item_diag(diag_values, phase, "alignment_reject_reliability", item, new_metric, correct)
        return [], False

    curr_score = _joint_distribution_score(
        history_dists["alignment"], text_dist, pred, align_item["feat"], score_norm_states["alignment"],
    )
    worst_item = caches["alignment"][pred][worst_idx]
    worst_score = _joint_distribution_score(
        history_dists["alignment"], text_dist, pred, worst_item["feat"], score_norm_states["alignment"],
    )

    if curr_score is None or worst_score is None:
        stats[f"{phase}_alignment_core_reject_no_distribution"] += 1
        stats[f"{phase}_alignment_core_reject_distribution"] += 1
        correct = _diag_correct(diag_state, item)
        _record_item_diag(diag_values, phase, "alignment_reject_distribution", item, new_metric, correct)
        return [], False

    _observe_score_norm(score_norm_states["alignment"], curr_score, worst_score, stats, phase, "alignment_core")

    if curr_score["joint"] > worst_score["joint"]:
        caches["alignment"][pred][worst_idx] = align_item
        _update_history_distribution(
            history_dists["alignment"], pred, align_item["feat"], stats, phase, "alignment_core",
        )
        _record_alignment_acceptance(stats, phase, align_item, diag_state)
        stats[f"{phase}_alignment_core_replace"] += 1
        _record_metric_scalar(stats, phase, "alignment_core_B", new_metric["B"])
        _record_metric_scalar(stats, phase, "alignment_core_C", new_metric["C"])
        return [align_item], True

    stats[f"{phase}_alignment_core_reject_distribution"] += 1
    correct = _diag_correct(diag_state, item)
    _record_item_diag(
        diag_values, phase, "alignment_reject_distribution", item, new_metric, correct,
        extra={
            "current_joint_score": curr_score.get("joint"),
            "worst_joint_score": worst_score.get("joint"),
            "current_visual_score": curr_score.get("visual"),
            "worst_visual_score": worst_score.get("visual"),
            "current_text_score": curr_score.get("text"),
            "worst_text_score": worst_score.get("text"),
        },
    )
    return [], False


def _update_ctrl_cache_conservative(
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
    熵/能量缓存只接收对齐核心缓存接受过的样本。

    未满：直接加入；
    已满：控制量更低，且 joint distribution score 更高，才替换。
    分布不可用时保守拒绝满缓存替换。
    """
    pred = int(pred)
    if pred not in cache:
        cache[pred] = []

    if len(cache[pred]) < capacity:
        cache[pred].append(item)
        _sort_ctrl_cache(cache, pred)
        stats[f"{phase}_{cache_tag}_add"] += 1
        _update_history_distribution(history_dist, pred, item["feat"], stats, phase, cache_tag)
        return True

    worst_item = cache[pred][-1]
    if item["ctrl"] >= worst_item["ctrl"]:
        stats[f"{phase}_{cache_tag}_reject_ctrl"] += 1
        return False

    curr_score = _joint_distribution_score(history_dist, text_dist, pred, item["feat"], score_norm_state)
    worst_score = _joint_distribution_score(history_dist, text_dist, pred, worst_item["feat"], score_norm_state)

    if curr_score is None or worst_score is None:
        stats[f"{phase}_{cache_tag}_reject_no_distribution"] += 1
        stats[f"{phase}_{cache_tag}_reject_distribution"] += 1
        return False

    _observe_score_norm(score_norm_state, curr_score, worst_score, stats, phase, cache_tag)

    if curr_score["joint"] > worst_score["joint"]:
        cache[pred][-1] = item
        _sort_ctrl_cache(cache, pred)
        stats[f"{phase}_{cache_tag}_replace"] += 1
        _update_history_distribution(history_dist, pred, item["feat"], stats, phase, cache_tag)
        return True

    stats[f"{phase}_{cache_tag}_reject_distribution"] += 1
    return False


def _update_entropy_energy_from_alignment(caches, history_dists, score_norm_states, text_dist, pred, accepted_items, stats, phase):
    entered_entropy_items = []
    entered_energy_items = []

    for align_item in accepted_items:
        entropy_item = {
            "feat": align_item["feat"],
            "label": int(align_item["label"]),
            "ctrl": float(align_item["entropy"]),
        }
        energy_item = {
            "feat": align_item["feat"],
            "label": int(align_item["label"]),
            "ctrl": float(align_item["energy"]),
        }

        if _update_ctrl_cache_conservative(
            caches["entropy"], history_dists["entropy"], text_dist, score_norm_states["entropy"],
            pred, entropy_item, ENTROPY_CAPACITY, stats, phase, "entropy",
        ):
            entered_entropy_items.append(align_item)

        if _update_ctrl_cache_conservative(
            caches["energy"], history_dists["energy"], text_dist, score_norm_states["energy"],
            pred, energy_item, ENERGY_CAPACITY, stats, phase, "energy",
        ):
            entered_energy_items.append(align_item)

    return entered_entropy_items, entered_energy_items


def _process_sample_updates(
    caches,
    history_dists,
    score_norm_states,
    text_dist,
    pred,
    pc_feats,
    entropy_value,
    energy_value,
    margin_value,
    zs_correct,
    stats,
    phase,
    diag_state,
    diag_values=None,
):
    """
    返回 update_info，说明当前样本是否进入真正缓存。
    """
    item = _make_candidate_item(pc_feats, pred, entropy_value, energy_value, margin_value)
    current_key = _feature_key(pc_feats)
    diag_state.setdefault("zs_correct", {})[current_key] = int(zs_correct)

    candidate_kept, candidate_metric = _update_candidate_pool(
        caches["candidate"], pred, item, stats, phase, diag_state, diag_values,
    )
    if not candidate_kept:
        return {
            "candidate_kept": False,
            "current_entered_alignment": False,
            "current_entered_entropy": False,
            "current_entered_energy": False,
            "entered_true_cache": False,
        }

    accepted_alignment_items, current_entered_alignment = _update_alignment_core(
        caches, history_dists, score_norm_states, text_dist, pred, item, stats, phase,
        diag_state, diag_values,
    )

    entered_entropy_items, entered_energy_items = _update_entropy_energy_from_alignment(
        caches, history_dists, score_norm_states, text_dist, pred, accepted_alignment_items, stats, phase,
    )

    current_entered_entropy = any(_feature_key(it["feat"]) == current_key for it in entered_entropy_items)
    current_entered_energy = any(_feature_key(it["feat"]) == current_key for it in entered_energy_items)
    entered_true_cache = bool(current_entered_alignment or current_entered_entropy or current_entered_energy)

    if entered_true_cache:
        stats[f"{phase}_entered_true_cache"] += 1
        correct = _diag_correct(diag_state, item)
        _record_item_diag(
            diag_values, phase, "entered_alignment_entropy_energy_cache",
            item, candidate_metric, correct,
        )
    elif candidate_kept:
        stats[f"{phase}_candidate_only"] += 1
        correct = _diag_correct(diag_state, item)
        _record_item_diag(diag_values, phase, "candidate_only", item, candidate_metric, correct)

    return {
        "candidate_kept": True,
        "current_entered_alignment": bool(current_entered_alignment),
        "current_entered_entropy": bool(current_entered_entropy),
        "current_entered_energy": bool(current_entered_energy),
        "entered_true_cache": entered_true_cache,
    }


# ============================================================
# 打分、统计、保存
# ============================================================

def _apply_cache_norm_clip(cache_logits):
    norm_before_tensor = torch.norm(cache_logits.detach().float(), p=2)
    norm_before = float(norm_before_tensor.cpu().item())
    clipped = cache_logits
    did_clip = False

    if CACHE_NORM_CAP > 0 and norm_before > CACHE_NORM_CAP:
        scale = cache_logits.new_tensor(CACHE_NORM_CAP) / (
            norm_before_tensor.to(device=cache_logits.device, dtype=cache_logits.dtype)
            + cache_logits.new_tensor(CACHE_NORM_EPS)
        )
        clipped = cache_logits * scale
        did_clip = True

    norm_after = float(torch.norm(clipped.detach().float(), p=2).cpu().item())
    return clipped, {
        "before": norm_before,
        "after": norm_after,
        "clipped": bool(did_clip),
    }


def _record_cache_norm_clip_stats(stats, phase, tag, clip_info):
    prefixes = [f"{phase}_cache_norm"]
    if tag:
        prefixes.append(f"{phase}_{tag}_cache_norm")

    for prefix in prefixes:
        stats[f"{prefix}_before_sum"] += float(clip_info["before"])
        stats[f"{prefix}_before_count"] += 1
        stats[f"{prefix}_before_max"] = max(
            float(stats.get(f"{prefix}_before_max", 0.0)),
            float(clip_info["before"]),
        )
        stats[f"{prefix}_after_sum"] += float(clip_info["after"])
        stats[f"{prefix}_after_count"] += 1
        stats[f"{prefix}_after_max"] = max(
            float(stats.get(f"{prefix}_after_max", 0.0)),
            float(clip_info["after"]),
        )
        stats[f"{prefix}_clip_count"] += int(clip_info["clipped"])


def _finalize_cache_norm_clip_stats(stats, phase):
    for prefix in (
        f"{phase}_cache_norm",
        f"{phase}_old_cache_norm",
        f"{phase}_new_cache_norm",
    ):
        before_count = int(stats.get(f"{prefix}_before_count", 0))
        after_count = int(stats.get(f"{prefix}_after_count", 0))
        if before_count > 0:
            stats[f"{prefix}_before_mean"] = (
                float(stats.get(f"{prefix}_before_sum", 0.0)) / float(before_count)
            )
            stats[f"{prefix}_clip_rate"] = (
                float(stats.get(f"{prefix}_clip_count", 0)) / float(before_count)
            )
        if after_count > 0:
            stats[f"{prefix}_after_mean"] = (
                float(stats.get(f"{prefix}_after_sum", 0.0)) / float(after_count)
            )


def _finalize_extra_logit_norm_stats(stats, phase, names):
    for name in names:
        count_key = f"{phase}_{name}_logits_norm_count"
        sum_key = f"{phase}_{name}_logits_norm_sum"
        mean_key = f"{phase}_{name}_logits_norm_mean"
        count = int(stats.get(count_key, 0))
        if count > 0:
            stats[mean_key] = float(stats.get(sum_key, 0.0)) / float(count)


def _compute_e7_a4_b1_logits(pc_feats, caches, clip_logits, clip_weights):
    s_a = torch.zeros_like(clip_logits)
    align_total = sum(len(v) for v in caches["alignment"].values())
    if align_total > ALIGNMENT_MIN_TOTAL:
        s_a = compute_cache_vote_logits(pc_feats, caches["alignment"], ALPHA_ALIGNMENT, BETA_ALIGNMENT, clip_weights)

    s_h = compute_cache_vote_logits(pc_feats, caches["entropy"], ALPHA_ENTROPY, BETA_ENTROPY, clip_weights)
    s_e = compute_cache_vote_logits(pc_feats, caches["energy"], ALPHA_ENERGY, BETA_ENERGY, clip_weights)
    raw_cache_logits = s_a + s_h + s_e
    clipped_cache_logits, clip_info = _apply_cache_norm_clip(raw_cache_logits)
    final_logits = ALPHA_ZS * clip_logits.clone() + clipped_cache_logits
    return final_logits, s_a, s_h, s_e, raw_cache_logits, clipped_cache_logits, clip_info


def _record_acc_counter(stats, phase, name, logits, target):
    pred = int(logits.detach().float().topk(1, dim=1)[1].item())
    tgt = int(target.detach().cpu().item())
    stats[f"{phase}_{name}_total"] += 1
    stats[f"{phase}_{name}_correct"] += int(pred == tgt)


def _finalize_acc_counter(stats, phase, names):
    for name in names:
        total = int(stats.get(f"{phase}_{name}_total", 0))
        correct = int(stats.get(f"{phase}_{name}_correct", 0))
        if total > 0:
            stats[f"{phase}_{name}_acc"] = float(correct) / float(total) * 100.0


def _finalize_zs_correctness(stats):
    for phase in ("build", "test"):
        for name in ("candidate_entered_zs", "alignment_core_entered_zs"):
            total = int(stats.get(f"{phase}_{name}_total", 0))
            correct = int(stats.get(f"{phase}_{name}_correct", 0))
            if total > 0:
                stats[f"{phase}_{name}_acc"] = float(correct) / float(total) * 100.0

    for name in ("candidate_entered_zs", "alignment_core_entered_zs"):
        total = int(stats.get(f"build_{name}_total", 0)) + int(stats.get(f"test_{name}_total", 0))
        correct = int(stats.get(f"build_{name}_correct", 0)) + int(stats.get(f"test_{name}_correct", 0))
        stats[f"all_{name}_total"] = total
        stats[f"all_{name}_correct"] = correct
        if total > 0:
            stats[f"all_{name}_acc"] = float(correct) / float(total) * 100.0


def _finalize_metric_scalars(stats, phase):
    for name in ("candidate_B", "candidate_C", "alignment_core_B", "alignment_core_C"):
        count = int(stats.get(f"{phase}_{name}_count", 0))
        if count > 0:
            stats[f"{phase}_{name}_mean"] = float(stats.get(f"{phase}_{name}_sum", 0.0)) / float(count)


def _summarize_cache(cache):
    return {str(k): len(v) for k, v in sorted(cache.items(), key=lambda kv: kv[0])}


def _save_e7_a4_stats(args, stats, caches, history_dists, score_norm_states, acc=None, diag_values=None):
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
        "text_distribution_role": "replacement_scoring_only",
        "final_text_classifier_source": getattr(args, "prompt_source", None),
        "text_prediction_decoupled": True,
        "capacities": {
            "candidate": CANDIDATE_CAPACITY,
            "alignment_core": ALIGNMENT_CAPACITY,
            "entropy": ENTROPY_CAPACITY,
            "energy": ENERGY_CAPACITY,
        },
        "alphas": {
            "zs": ALPHA_ZS,
            "alignment": ALPHA_ALIGNMENT,
            "entropy": ALPHA_ENTROPY,
            "energy": ALPHA_ENERGY,
        },
        "betas": {
            "alignment": BETA_ALIGNMENT,
            "entropy": BETA_ENTROPY,
            "energy": BETA_ENERGY,
        },
        "score_average": {
            "old_weight": SCORE_AVG_OLD_WEIGHT,
            "new_weight": SCORE_AVG_NEW_WEIGHT,
        },
        "cache_norm_clip": {
            "cap": CACHE_NORM_CAP,
            "eps": CACHE_NORM_EPS,
        },
        "e7_text_score_weight": float(TEXT_SCORE_WEIGHT),
        "e7_score_norm_mode": SCORE_NORM_MODE,
        "final_acc": acc,
        "stats": dict(stats),
        "diag_value_summary": _summarize_diag_values(diag_values),
        "candidate_pool_class_counts": _summarize_cache(caches["candidate"]),
        "alignment_core_class_counts": _summarize_cache(caches["alignment"]),
        "entropy_cache_class_counts": _summarize_cache(caches["entropy"]),
        "energy_cache_class_counts": _summarize_cache(caches["energy"]),
        "candidate_pool_total": int(sum(len(v) for v in caches["candidate"].values())),
        "alignment_core_total": int(sum(len(v) for v in caches["alignment"].values())),
        "entropy_cache_total": int(sum(len(v) for v in caches["entropy"].values())),
        "energy_cache_total": int(sum(len(v) for v in caches["energy"].values())),
        "alignment_history_summary": _summarize_history_distribution(history_dists["alignment"]),
        "entropy_history_summary": _summarize_history_distribution(history_dists["entropy"]),
        "energy_history_summary": _summarize_history_distribution(history_dists["energy"]),
        "score_norm_summary": {
            k: _summarize_score_norm_state(v) for k, v in score_norm_states.items()
        },
    }

    if os.environ.get("E7_SAVE_DIAG_VALUES_RAW", "1") != "0" and diag_values:
        payload["diag_values"] = {
            key: [float(v) for v in values if math.isfinite(float(v))]
            for key, values in sorted(diag_values.items())
        }

    filename = f"{getattr(args, 'cor_type', 'unknown')}_e7_a4_b1_stats.json"
    with (out_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[E7-A4-B1] Saved stats to {out_dir / filename}")


# ============================================================
# 构建与测试
# ============================================================

@torch.no_grad()
def build_cache_in_advance(args, test_loader, lm3d_model, clip_weights, text_dist=None):
    print("*" * 10, "Building E7-A4-B1 candidate/alignment-core caches ...", "*" * 10, "\n")

    caches = {"candidate": {}, "alignment": {}, "entropy": {}, "energy": {}}
    history_dists = {"alignment": {}, "entropy": {}, "energy": {}}
    score_norm_states = {
        "alignment": _make_score_norm_state(),
        "entropy": _make_score_norm_state(),
        "energy": _make_score_norm_state(),
    }
    diag_state = {"zs_correct": {}}
    diag_values = defaultdict(list)
    stats = defaultdict(int)

    for pc, target, _, rgb in test_loader:
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        entropy_value = loss
        energy_value = _compute_energy(clip_logits)
        margin_value = _top1_margin(clip_logits)
        zs_correct = int(pred == int(target.detach().cpu().item()))

        _process_sample_updates(
            caches, history_dists, score_norm_states, text_dist,
            pred, pc_feats, entropy_value, energy_value, margin_value, zs_correct, stats, "build", diag_state,
            diag_values,
        )

        align_total = sum(len(v) for v in caches["alignment"].values())
        num_classes = clip_logits.size(1)
        if align_total >= ALIGNMENT_CAPACITY * num_classes:
            print("*" * 10, "E7-A4-B1 alignment core cache is full. Build done.", "*" * 10, "\n")
            break

    return caches, history_dists, score_norm_states, diag_state, diag_values, stats


@torch.no_grad()
def run_test_tda(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights, text_dist=None):
    """
    E7-A4-B1 test-time adaptation.

    S_old 使用旧缓存；
    当前样本若进入真正缓存，则 S_final = 0.5*S_old + 0.5*S_new；
    否则 S_final = S_old。
    """
    caches, history_dists, score_norm_states, diag_state, diag_values, build_stats = build_cache_in_advance(
        args, test_loader, lm3d_model, clip_weights, text_dist=text_dist,
    )

    print("[E7-A4-B1] candidate pool total:", sum(len(v) for v in caches["candidate"].values()))
    print("[E7-A4-B1] alignment core total:", sum(len(v) for v in caches["alignment"].values()))
    print("[E7-A4-B1] entropy cache total:", sum(len(v) for v in caches["entropy"].values()))
    print("[E7-A4-B1] energy cache total:", sum(len(v) for v in caches["energy"].values()))
    print("[E7-A4-B1] final classifier prompt source:", getattr(args, "prompt_source", None))
    print("[E7-A4-B1] text distribution classes:", 0 if text_dist is None else len(text_dist))
    print("[E7-A4-B1] cache norm cap:", CACHE_NORM_CAP)
    print("[E7-A4-B1] capacities: candidate={} alignment={} entropy={} energy={}".format(
        CANDIDATE_CAPACITY, ALIGNMENT_CAPACITY, ENTROPY_CAPACITY, ENERGY_CAPACITY,
    ))
    print("[E7-A4-B1] alphas: zs={} A={} H={} E={}".format(
        ALPHA_ZS, ALPHA_ALIGNMENT, ALPHA_ENTROPY, ALPHA_ENERGY,
    ))

    stats = defaultdict(int)
    for k, v in build_stats.items():
        stats[k] += v

    accuracies = []
    zs_changed = 0
    total_seen = 0

    for i, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        target = target.cuda()
        entropy_value = loss
        energy_value = _compute_energy(clip_logits)
        margin_value = _top1_margin(clip_logits)
        zs_correct = int(pred == int(target.detach().cpu().item()))

        (
            old_logits,
            old_s_a,
            old_s_h,
            old_s_e,
            old_cache_raw,
            old_cache_clipped,
            old_clip_info,
        ) = _compute_e7_a4_b1_logits(pc_feats, caches, clip_logits, clip_weights)
        _record_cache_norm_clip_stats(stats, "test", "old", old_clip_info)
        _record_acc_counter(stats, "test", "score_old", old_logits, target)

        update_info = _process_sample_updates(
            caches, history_dists, score_norm_states, text_dist,
            pred, pc_feats, entropy_value, energy_value, margin_value, zs_correct, stats, "test", diag_state,
            diag_values,
        )

        if update_info["entered_true_cache"]:
            (
                new_logits,
                new_s_a,
                new_s_h,
                new_s_e,
                new_cache_raw,
                new_cache_clipped,
                new_clip_info,
            ) = _compute_e7_a4_b1_logits(pc_feats, caches, clip_logits, clip_weights)
            _record_cache_norm_clip_stats(stats, "test", "new", new_clip_info)
            final_logits = SCORE_AVG_OLD_WEIGHT * old_logits + SCORE_AVG_NEW_WEIGHT * new_logits
            _record_acc_counter(stats, "test", "score_new", new_logits, target)
            _record_acc_counter(stats, "test", "score_avg", final_logits, target)
            s_a = SCORE_AVG_OLD_WEIGHT * old_s_a + SCORE_AVG_NEW_WEIGHT * new_s_a
            s_h = SCORE_AVG_OLD_WEIGHT * old_s_h + SCORE_AVG_NEW_WEIGHT * new_s_h
            s_e = SCORE_AVG_OLD_WEIGHT * old_s_e + SCORE_AVG_NEW_WEIGHT * new_s_e
            cache_raw = SCORE_AVG_OLD_WEIGHT * old_cache_raw + SCORE_AVG_NEW_WEIGHT * new_cache_raw
            cache_clipped = SCORE_AVG_OLD_WEIGHT * old_cache_clipped + SCORE_AVG_NEW_WEIGHT * new_cache_clipped
        else:
            final_logits = old_logits
            _record_acc_counter(stats, "test", "score_avg", final_logits, target)
            s_a, s_h, s_e = old_s_a, old_s_h, old_s_e
            cache_raw = old_cache_raw
            cache_clipped = old_cache_clipped

        _record_logit_norm(stats, "test", "zs", ALPHA_ZS * clip_logits)
        _record_logit_norm(stats, "test", "alignment", s_a)
        _record_logit_norm(stats, "test", "entropy", s_h)
        _record_logit_norm(stats, "test", "energy", s_e)
        _record_logit_norm(stats, "test", "positive_cache_total_before_clip", cache_raw)
        _record_logit_norm(stats, "test", "positive_cache_total_after_clip", cache_clipped)
        _record_logit_norm(stats, "test", "positive_cache_total", cache_clipped)
        _record_logit_norm(stats, "test", "final", final_logits)

        final_pred = int(final_logits.topk(1, dim=1)[1].item())
        if final_pred != int(pred):
            zs_changed += 1
        total_seen += 1

        acc = cls_acc(final_logits, target)
        accuracies.append(acc)
        wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

        if i % args.print_freq == 0:
            print("---- E7-A4-B1 test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    final_acc = sum(accuracies) / len(accuracies)
    print("---- ***Final*** E7-A4-B1 test accuracy: {:.2f}. ----\n".format(final_acc))

    stats["test_zs_vs_final_pred_change"] = int(zs_changed)
    stats["test_total_seen"] = int(total_seen)
    _finalize_logit_norm_stats(stats, "test")
    _finalize_extra_logit_norm_stats(
        stats, "test", ("positive_cache_total_before_clip", "positive_cache_total_after_clip"),
    )
    _finalize_cache_norm_clip_stats(stats, "test")
    _finalize_metric_scalars(stats, "build")
    _finalize_metric_scalars(stats, "test")
    _finalize_zs_correctness(stats)
    _finalize_acc_counter(stats, "test", ("score_old", "score_new", "score_avg"))

    _save_e7_a4_stats(args, stats, caches, history_dists, score_norm_states, final_acc, diag_values)
    return final_acc
