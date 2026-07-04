"""
E4-C-A0+E1 Distribution-Final Score for Point-Cache.

核心思想：
1. 保留原始 Point-Cache 的 Global Entropy Cache，但不再用其投票得分做最终分类；
2. 保留 Global Prototype-Alignment Cache，简称 GPA Cache；
3. 文本端使用每类 prompt-level embeddings 构建固定 text distribution；
4. 视觉端只累计曾被正缓存接受过的可信样本，构建 accepted-history visual distribution；
5. GPA Cache 未满时沿用 E3-V2-C / E4-A 的直接加入规则；
6. GPA Cache 满后使用“低熵 + text-visual joint score 更高”替换最高熵样本；
7. 只有进入 GPA Cache 的样本，其 patch_centers 才写入 local cache；
8. 最终预测不再使用 zero-shot 原型点积或缓存投票，全部改为分布得分。
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

CENTER_SOURCE_LABEL = "Accepted-history text-visual class-wise distribution score"
GPA_VARIANT_NAME = "E4-C-A0-E1-distribution-final-score"


def _loss_value(loss):
    """Convert entropy tensor/scalar to python float for sorting and logging."""
    if torch.is_tensor(loss):
        return float(loss.detach().float().cpu().item())
    return float(loss)


def _get_gpa_stats_enabled():
    return os.environ.get("GPA_SAVE_STATS", "1") != "0"


def _sort_cache_by_entropy(cache, pred):
    cache[pred] = sorted(cache[pred], key=lambda x: _loss_value(x[1]))


def _sort_local_cache_by_entropy(local_cache, pred):
    local_cache[pred] = sorted(local_cache[pred], key=lambda x: _loss_value(x[1]))


def _sort_gpa_pair_by_entropy(gpa_cache, gpa_local_cache, pred):
    pairs = sorted(
        zip(gpa_cache[pred], gpa_local_cache[pred]),
        key=lambda pair: _loss_value(pair[0][1]),
    )
    gpa_cache[pred] = [pair[0] for pair in pairs]
    gpa_local_cache[pred] = [pair[1] for pair in pairs]


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




# ============================================================
# E4-C：Accepted-History Text-Visual 类别概率分布辅助函数
# ============================================================

DIST_EPS = float(os.environ.get("E4_DIST_EPS", "1e-4"))
DIST_MIN_VAR = float(os.environ.get("E4_DIST_MIN_VAR", "1e-4"))
TEXT_DIST_EPS = float(os.environ.get("E4_TEXT_DIST_EPS", str(DIST_EPS)))
TEXT_DIST_MIN_VAR = float(os.environ.get("E4_TEXT_DIST_MIN_VAR", str(DIST_MIN_VAR)))
TEXT_SCORE_WEIGHT = float(os.environ.get("E4_TEXT_SCORE_WEIGHT", "0.1"))
SCORE_NORM_MODE = os.environ.get("E4_SCORE_NORM_MODE", "none").strip().lower()
SCORE_NORM_MIN_COUNT = int(os.environ.get("E4_SCORE_NORM_MIN_COUNT", "8"))
SCORE_NORM_EPS = float(os.environ.get("E4_SCORE_NORM_EPS", "1e-6"))
SCORE_NORM_CLIP = float(os.environ.get("E4_SCORE_NORM_CLIP", "0"))

FINAL_TEXT_DIST_WEIGHT = float(os.environ.get("E4_FINAL_TEXT_DIST_WEIGHT", "1.0"))
FINAL_VISUAL_DIST_WEIGHT = float(os.environ.get("E4_FINAL_VISUAL_DIST_WEIGHT", "1.0"))
FINAL_DIST_NORM_MODE = os.environ.get("E4_FINAL_DIST_NORM_MODE", "per_sample_zscore").strip().lower()
FINAL_DIST_NORM_EPS = float(os.environ.get("E4_FINAL_DIST_NORM_EPS", "1e-6"))
FINAL_MISSING_SCORE_MARGIN = float(os.environ.get("E4_FINAL_MISSING_SCORE_MARGIN", "10.0"))

if SCORE_NORM_MODE not in {"none", "running_zscore"}:
    raise ValueError(f"Unsupported E4_SCORE_NORM_MODE: {SCORE_NORM_MODE}")

if FINAL_DIST_NORM_MODE not in {"none", "per_sample_zscore", "per_sample_minmax"}:
    raise ValueError(f"Unsupported E4_FINAL_DIST_NORM_MODE: {FINAL_DIST_NORM_MODE}")


def _feature_float(feat):
    return feat.detach().float()


def _feature_key(feat):
    x = feat.detach()
    storage = x.untyped_storage() if hasattr(x, "untyped_storage") else x.storage()
    return (int(x.data_ptr()), int(storage.data_ptr()), tuple(x.shape), str(x.device))


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


def _update_visual_distribution(visual_dist, pred, feat, stats=None, phase=None, reason=None):
    """
    累计被正缓存接受过的可信视觉样本。

    样本只有在成功进入或替换 Global Entropy Cache / GPA-Cache 后才会调用这里。
    被两个正缓存同时接受的同一 tensor 只计一次；完全没有进入正缓存的候选样本不参与分布。
    """
    pred = int(pred)
    key = _feature_key(feat)

    if pred not in visual_dist:
        visual_dist[pred] = {
            "count": 0,
            "mean": None,
            "m2": None,
            "seen": set(),
        }

    entry = visual_dist[pred]
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

    if stats is not None and phase is not None:
        stats[f"{phase}_visual_dist_update"] += 1
        if reason is not None:
            stats[f"{phase}_visual_dist_update_{reason}"] += 1

    return True


def _visual_distribution(visual_dist, pred):
    pred = int(pred)
    if pred not in visual_dist:
        return None

    entry = visual_dist[pred]
    count = int(entry["count"])
    if count < 2:
        return None

    var = (entry["m2"] / float(max(count - 1, 1))).clamp_min(DIST_MIN_VAR)
    return {
        "count": count,
        "mean": entry["mean"],
        "var": var,
    }


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


def _joint_distribution_score(visual_dist, text_dist, pred, feat, score_norm_state=None):
    """
    计算 E4-C 的 text-visual joint score。

    visual_dist 来自被正缓存接受过的可信历史样本；
    text_dist 来自固定 prompt-level embeddings。
    """
    visual_entry = _visual_distribution(visual_dist, pred)
    text_entry = _text_distribution(text_dist, pred, feat)

    visual_score = _distribution_score_from_entry(visual_entry, feat, DIST_EPS)
    text_score = _distribution_score_from_entry(text_entry, feat, TEXT_DIST_EPS)

    if visual_score is None:
        return None

    if text_score is None:
        use_normalized_scores = _score_norm_ready(score_norm_state, ("visual",))
        visual_joint_score, visual_score_normalized = (
            _score_for_joint(score_norm_state, "visual", visual_score)
            if use_normalized_scores
            else (float(visual_score), False)
        )
        text_joint_score = None
        text_score_normalized = False
        joint_score = visual_joint_score
    else:
        use_normalized_scores = _score_norm_ready(score_norm_state, ("visual", "text"))
        if use_normalized_scores:
            visual_joint_score, visual_score_normalized = _score_for_joint(score_norm_state, "visual", visual_score)
            text_joint_score, text_score_normalized = _score_for_joint(score_norm_state, "text", text_score)
        else:
            visual_joint_score, visual_score_normalized = float(visual_score), False
            text_joint_score, text_score_normalized = float(text_score), False
        joint_score = visual_joint_score + TEXT_SCORE_WEIGHT * text_joint_score

    return {
        "joint": float(joint_score),
        "visual": float(visual_score),
        "text": None if text_score is None else float(text_score),
        "visual_for_joint": float(visual_joint_score),
        "text_for_joint": None if text_joint_score is None else float(text_joint_score),
        "visual_score_normalized": bool(visual_score_normalized),
        "text_score_normalized": bool(text_score_normalized),
        "visual_count": 0 if visual_entry is None else int(visual_entry["count"]),
        "text_count": 0 if text_entry is None else int(text_entry["count"]),
    }


def _summarize_distribution_from_entry(entry):
    if entry is None:
        return None

    var = entry["var"].detach().float().cpu()
    return {
        "count": int(entry["count"]),
        "var_mean": float(var.mean().item()),
        "var_min": float(var.min().item()),
        "var_max": float(var.max().item()),
    }


def _summarize_visual_distribution(visual_dist):
    summary = {}

    for c in sorted(visual_dist.keys()):
        entry = _visual_distribution(visual_dist, c)
        if entry is not None:
            summary[str(c)] = _summarize_distribution_from_entry(entry)
        else:
            count = int(visual_dist[c]["count"])
            summary[str(c)] = {"count": int(count), "var_mean": None, "var_min": None, "var_max": None}

    return summary


def _summarize_text_distribution(text_dist):
    if text_dist is None:
        return {}

    summary = {}
    for c, entry in sorted(text_dist.items(), key=lambda kv: kv[0]):
        summary[str(c)] = _summarize_distribution_from_entry(entry)
    return summary


def _update_gpa_cache(
    gpa_cache,
    gpa_local_cache,
    visual_dist,
    text_dist,
    score_norm_state,
    pred,
    global_item,
    local_item,
    shot_capacity,
    stats,
    phase,
    event_records=None,
):
    """
    E4-C：Accepted-History Text-Visual 类别概率分布引导的 GPA-Cache 更新。

    沿用 E3-V2-C / E4-A：
        1. 未满直接加入 GPA-Cache；
        2. 满后替换最高熵样本；
        3. 保留低熵门控。

    改动：
        用曾被正缓存接受过的历史可信视觉分布，以及固定 prompt 文本分布共同计算 joint score。
    """
    if pred not in gpa_cache:
        gpa_cache[pred] = []
        gpa_local_cache[pred] = []

    curr_ent = _loss_value(global_item[1])

    def record_event(decision, old_entropy=None, new_score=None, old_score=None):
        if event_records is None:
            return
        event_records.append({
            "phase": phase,
            "class_index": int(pred),
            "decision": decision,
            "update_rule": "low_entropy_gate_accepted_history_text_visual_joint_score_replace_highest_entropy",
            "new_entropy": float(curr_ent),
            "old_entropy": None if old_entropy is None else float(old_entropy),
            "new_joint_score": None if new_score is None else float(new_score["joint"]),
            "old_joint_score": None if old_score is None else float(old_score["joint"]),
            "new_visual_score": None if new_score is None else float(new_score["visual"]),
            "old_visual_score": None if old_score is None else float(old_score["visual"]),
            "new_text_score": None if new_score is None or new_score["text"] is None else float(new_score["text"]),
            "old_text_score": None if old_score is None or old_score["text"] is None else float(old_score["text"]),
            "new_visual_score_for_joint": None if new_score is None else float(new_score["visual_for_joint"]),
            "old_visual_score_for_joint": None if old_score is None else float(old_score["visual_for_joint"]),
            "new_text_score_for_joint": None if new_score is None or new_score["text_for_joint"] is None else float(new_score["text_for_joint"]),
            "old_text_score_for_joint": None if old_score is None or old_score["text_for_joint"] is None else float(old_score["text_for_joint"]),
            "new_visual_score_normalized": None if new_score is None else bool(new_score["visual_score_normalized"]),
            "old_visual_score_normalized": None if old_score is None else bool(old_score["visual_score_normalized"]),
            "new_text_score_normalized": None if new_score is None else bool(new_score["text_score_normalized"]),
            "old_text_score_normalized": None if old_score is None else bool(old_score["text_score_normalized"]),
            "score_norm_mode": SCORE_NORM_MODE,
            "joint_score_margin": None if new_score is None or old_score is None else float(new_score["joint"] - old_score["joint"]),
            "visual_count": None if new_score is None else int(new_score["visual_count"]),
            "text_count": None if new_score is None else int(new_score["text_count"]),
        })

    if len(gpa_cache[pred]) < shot_capacity:
        gpa_cache[pred].append(global_item)
        gpa_local_cache[pred].append(local_item)

        _sort_gpa_pair_by_entropy(gpa_cache, gpa_local_cache, pred)

        stats[f"{phase}_gpa_add_not_full"] += 1
        stats[f"{phase}_gpa_add_not_full_accepted_history_text_visual_distribution"] += 1
        _update_visual_distribution(visual_dist, pred, global_item[0], stats, phase, "gpa_add")
        record_event(decision="add_not_full_accepted_history_text_visual_distribution")
        return True

    worst_global_item = gpa_cache[pred][-1]
    worst_ent = _loss_value(worst_global_item[1])

    curr_score = _joint_distribution_score(visual_dist, text_dist, pred, global_item[0], score_norm_state)
    worst_score = _joint_distribution_score(visual_dist, text_dist, pred, worst_global_item[0], score_norm_state)

    if curr_score is None or worst_score is None:
        stats[f"{phase}_gpa_reject_no_accepted_history_text_visual_distribution"] += 1
        record_event(decision="reject_no_accepted_history_text_visual_distribution", old_entropy=worst_ent, new_score=curr_score, old_score=worst_score)
        return False

    if curr_ent >= worst_ent:
        stats[f"{phase}_gpa_reject_entropy"] += 1
        stats[f"{phase}_gpa_reject_entropy_accepted_history_text_visual_distribution"] += 1
        record_event(decision="reject_entropy_accepted_history_text_visual_distribution", old_entropy=worst_ent, new_score=curr_score, old_score=worst_score)
        return False

    norm_updates = _update_score_norm_state(score_norm_state, curr_score)
    norm_updates += _update_score_norm_state(score_norm_state, worst_score)
    if norm_updates:
        stats[f"{phase}_score_norm_update"] += norm_updates
        stats[f"{phase}_score_norm_observed_pairs"] += 1

    if curr_score["joint"] > worst_score["joint"]:
        gpa_cache[pred][-1] = global_item
        gpa_local_cache[pred][-1] = local_item

        _sort_gpa_pair_by_entropy(gpa_cache, gpa_local_cache, pred)

        stats[f"{phase}_gpa_replace_accepted_history_text_visual_distribution"] += 1
        _update_visual_distribution(visual_dist, pred, global_item[0], stats, phase, "gpa_replace")
        record_event(decision="replace_accepted_history_text_visual_distribution", old_entropy=worst_ent, new_score=curr_score, old_score=worst_score)
        return True

    stats[f"{phase}_gpa_reject_accepted_history_text_visual_distribution"] += 1
    record_event(decision="reject_accepted_history_text_visual_distribution", old_entropy=worst_ent, new_score=curr_score, old_score=worst_score)
    return False


def _summarize_cache(cache):
    return {str(k): len(v) for k, v in sorted(cache.items(), key=lambda kv: kv[0])}


def _save_gpa_stats(
    args,
    stats,
    entropy_cache,
    gpa_cache,
    gpa_local_cache,
    visual_dist=None,
    text_dist=None,
    score_norm_state=None,
    acc=None,
    event_records=None,
):
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
        "e4_variant": "E4-C",
        "distribution_scope": "text_visual_distribution",
        "visual_distribution_scope": "accepted_positive_cache_history",
        "visual_distribution_history_policy": "accumulate_samples_accepted_by_entropy_or_gpa_cache_only",
        "text_distribution_enabled": True,
        "e4_dist_eps": float(DIST_EPS),
        "e4_dist_min_var": float(DIST_MIN_VAR),
        "e4_text_dist_eps": float(TEXT_DIST_EPS),
        "e4_text_dist_min_var": float(TEXT_DIST_MIN_VAR),
        "e4_text_score_weight": float(TEXT_SCORE_WEIGHT),
        "e4_score_norm_mode": SCORE_NORM_MODE,
        "e4_score_norm_min_count": int(SCORE_NORM_MIN_COUNT),
        "e4_score_norm_eps": float(SCORE_NORM_EPS),
        "e4_score_norm_clip": float(SCORE_NORM_CLIP),
        "final_score_type": "distribution_only",
        "final_score_uses_clip_logits": False,
        "final_score_uses_cache_voting_logits": False,
        "e4_final_text_dist_weight": float(FINAL_TEXT_DIST_WEIGHT),
        "e4_final_visual_dist_weight": float(FINAL_VISUAL_DIST_WEIGHT),
        "e4_final_dist_norm_mode": FINAL_DIST_NORM_MODE,
        "e4_final_dist_norm_eps": float(FINAL_DIST_NORM_EPS),
        "e4_final_missing_score_margin": float(FINAL_MISSING_SCORE_MARGIN),
        "score_normalization_summary": _summarize_score_norm_state(score_norm_state),
        "center_source": CENTER_SOURCE_LABEL,
        "final_acc": acc,
        "stats": dict(stats),
        "entropy_cache_class_counts": _summarize_cache(entropy_cache),
        "gpa_cache_class_counts": _summarize_cache(gpa_cache),
        "gpa_local_cache_class_counts": _summarize_cache(gpa_local_cache),
        "entropy_cache_total": int(sum(len(v) for v in entropy_cache.values())),
        "gpa_cache_total": int(sum(len(v) for v in gpa_cache.values())),
        "gpa_local_cache_total": int(sum(len(v) for v in gpa_local_cache.values())),
        "visual_distribution_num_classes": 0 if visual_dist is None else int(len(visual_dist)),
        "visual_distribution_summary": {} if visual_dist is None else _summarize_visual_distribution(visual_dist),
        "text_distribution_num_classes": 0 if text_dist is None else int(len(text_dist)),
        "text_distribution_summary": _summarize_text_distribution(text_dist),
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
        print(f"[E4-C] Saved GPA replacement events to {event_path}")

    print(f"[E4-C] Saved GPA stats to {out_dir / filename}")


@torch.no_grad()
def build_cache_in_advance(args, test_loader, lm3d_model, clip_weights, shot_capacity, include_prob_map=False, text_dist=None):
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
    visual_dist = {}
    score_norm_state = _make_score_norm_state()

    for pc, _, _, rgb in test_loader:
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        global_item = [pc_feats, loss]
        local_item = [patch_centers, loss]

        # E4-C：GPA 使用已被正缓存接受过的历史可信 visual_dist 评分。
        _update_gpa_cache(
            gpa_cache,
            gpa_local_cache,
            visual_dist,
            text_dist,
            score_norm_state,
            pred,
            global_item,
            local_item,
            shot_capacity,
            stats,
            "build",
            gpa_event_records,
        )

        entropy_accepted = _update_entropy_cache(entropy_cache, pred, global_item, shot_capacity, stats, "build")
        if entropy_accepted:
            _update_visual_distribution(visual_dist, pred, global_item[0], stats, "build", "entropy_accept")

        cache_num = sum(len(entropy_cache[key]) for key in entropy_cache)
        num_classes = clip_logits.size(1)
        full_num = shot_capacity * num_classes

        if cache_num == full_num:
            print("*" * 10, "Building [global entropy] cache is full.", "*" * 10, "\n")
            break

    return entropy_cache, gpa_cache, gpa_local_cache, visual_dist, score_norm_state, stats, gpa_event_records


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


def _scores_to_tensor(scores, ref_logits):
    valid_scores = [float(s) for s in scores if s is not None]
    if not valid_scores:
        return torch.zeros_like(ref_logits), torch.zeros_like(ref_logits, dtype=torch.bool)

    fill_value = min(valid_scores) - FINAL_MISSING_SCORE_MARGIN
    values = [fill_value if s is None else float(s) for s in scores]
    logits = torch.tensor(values, device=ref_logits.device, dtype=ref_logits.dtype).view_as(ref_logits)
    valid_mask = torch.tensor(
        [s is not None for s in scores],
        device=ref_logits.device,
        dtype=torch.bool,
    ).view_as(ref_logits)
    return logits, valid_mask


def _normalize_final_dist_logits(logits, valid_mask=None):
    if FINAL_DIST_NORM_MODE == "none":
        return logits

    if valid_mask is not None and valid_mask.any():
        active = logits[valid_mask]
    else:
        active = logits.view(-1)

    if active.numel() <= 1:
        return torch.zeros_like(logits)

    if FINAL_DIST_NORM_MODE == "per_sample_minmax":
        min_v = active.min()
        max_v = active.max()
        denom = (max_v - min_v).clamp_min(FINAL_DIST_NORM_EPS)
        return (logits - min_v) / denom

    mean = active.mean()
    std = active.std(unbiased=False).clamp_min(FINAL_DIST_NORM_EPS)
    return (logits - mean) / std


def compute_text_distribution_logits(pc_feats, text_dist, ref_logits):
    scores = []
    for class_index in range(ref_logits.size(1)):
        text_entry = _text_distribution(text_dist, class_index, pc_feats)
        scores.append(_distribution_score_from_entry(text_entry, pc_feats, TEXT_DIST_EPS))
    return _scores_to_tensor(scores, ref_logits)


def compute_visual_distribution_logits(pc_feats, visual_dist, ref_logits):
    scores = []
    for class_index in range(ref_logits.size(1)):
        visual_entry = _visual_distribution(visual_dist, class_index)
        scores.append(_distribution_score_from_entry(visual_entry, pc_feats, DIST_EPS))
    return _scores_to_tensor(scores, ref_logits)


def compute_distribution_final_logits(pc_feats, visual_dist, text_dist, ref_logits):
    text_logits, text_valid = compute_text_distribution_logits(pc_feats, text_dist, ref_logits)
    visual_logits, visual_valid = compute_visual_distribution_logits(pc_feats, visual_dist, ref_logits)

    text_norm = _normalize_final_dist_logits(text_logits, text_valid)
    visual_norm = _normalize_final_dist_logits(visual_logits, visual_valid)
    final_logits = FINAL_TEXT_DIST_WEIGHT * text_norm + FINAL_VISUAL_DIST_WEIGHT * visual_norm

    return final_logits, {
        "text_raw": text_logits,
        "visual_raw": visual_logits,
        "text_norm": text_norm,
        "visual_norm": visual_norm,
        "text_valid_count": int(text_valid.sum().detach().cpu().item()),
        "visual_valid_count": int(visual_valid.sum().detach().cpu().item()),
    }


def _record_acc_counter(stats, phase, name, logits, target):
    pred = int(logits.detach().float().topk(1, dim=1)[1].item())
    tgt = int(target.detach().cpu().item())
    stats[f"{phase}_{name}_correct"] += int(pred == tgt)
    stats[f"{phase}_{name}_total"] += 1


def _finalize_acc_counter(stats, phase, names):
    for name in names:
        total = int(stats.get(f"{phase}_{name}_total", 0))
        correct = int(stats.get(f"{phase}_{name}_correct", 0))
        if total > 0:
            stats[f"{phase}_{name}_acc"] = float(correct) / float(total) * 100.0


@torch.no_grad()
def run_test_tda(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights, text_dist=None):
    """
    E4-C distribution-final test-time adaptation.

    Global Entropy Cache 和 GPA-controlled Local Cache 仍用于维护可信视觉历史分布，
    但最终分类不再使用 global/local/negative cache voting logits。
    最终分类只使用 text distribution score 和 accepted-history visual distribution score。
    """
    entropy_cache, gpa_cache, gpa_local_cache, visual_dist, score_norm_state, build_stats, gpa_event_records = build_cache_in_advance(
        args, test_loader, lm3d_model, clip_weights, pos_cfg["shot_capacity"], text_dist=text_dist
    )

    print("[E4-C] len(entropy_cache):", len(entropy_cache))
    print("[E4-C] len(gpa_cache):", len(gpa_cache))
    print("[E4-C] len(gpa_local_cache):", len(gpa_local_cache))
    print("[E4-C] entropy cache total:", sum(len(v) for v in entropy_cache.values()))
    print("[E4-C] gpa cache total:", sum(len(v) for v in gpa_cache.values()))
    print("[E4-C] gpa local cache total:", sum(len(v) for v in gpa_local_cache.values()))
    print("[E4-C] visual distribution classes:", len(visual_dist))
    print("[E4-C] text distribution classes:", 0 if text_dist is None else len(text_dist))
    print("[E4-C] score norm mode:", SCORE_NORM_MODE)
    print("[E4-C] score norm state:", _summarize_score_norm_state(score_norm_state))
    print("[E4-C-DistFinal] final score type: distribution_only")
    print("[E4-C-DistFinal] final text dist weight:", FINAL_TEXT_DIST_WEIGHT)
    print("[E4-C-DistFinal] final visual dist weight:", FINAL_VISUAL_DIST_WEIGHT)
    print("[E4-C-DistFinal] final dist norm mode:", FINAL_DIST_NORM_MODE)

    neg_cache = {}
    gpa_cache_stats = defaultdict(int)
    for k, v in build_stats.items():
        gpa_cache_stats[k] += v

    # E4-C 保留 build 阶段形成的 EntropyCache、GPA-Cache、GPA-local-cache 和
    # accepted-history visual_dist。visual_dist 只累计曾被正缓存接受过的样本。
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

            _update_gpa_cache(
                gpa_cache,
                gpa_local_cache,
                visual_dist,
                text_dist,
                score_norm_state,
                pred,
                global_item,
                local_item,
                pos_params["shot_capacity"],
                gpa_cache_stats,
                "test",
                gpa_event_records,
            )

            entropy_accepted = _update_entropy_cache(
                entropy_cache, pred, global_item, pos_params["shot_capacity"], gpa_cache_stats, "test"
            )
            if entropy_accepted:
                _update_visual_distribution(visual_dist, pred, global_item[0], gpa_cache_stats, "test", "entropy_accept")
        final_logits, dist_info = compute_distribution_final_logits(
            pc_feats,
            visual_dist,
            text_dist,
            clip_logits,
        )

        _record_acc_counter(gpa_cache_stats, "test", "clip_proto", clip_logits, target)
        _record_acc_counter(gpa_cache_stats, "test", "text_dist_norm", dist_info["text_norm"], target)
        _record_acc_counter(gpa_cache_stats, "test", "visual_dist_norm", dist_info["visual_norm"], target)
        _record_acc_counter(gpa_cache_stats, "test", "distribution_final", final_logits, target)
        gpa_cache_stats["test_text_dist_valid_count_sum"] += int(dist_info["text_valid_count"])
        gpa_cache_stats["test_visual_dist_valid_count_sum"] += int(dist_info["visual_valid_count"])
        gpa_cache_stats["test_distribution_final_seen"] += 1

        acc = cls_acc(final_logits, target)
        accuracies.append(acc)
        wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

        if i % args.print_freq == 0:
            print("---- E4-C-DistFinal test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    final_acc = sum(accuracies) / len(accuracies)
    _finalize_acc_counter(
        gpa_cache_stats,
        "test",
        ("clip_proto", "text_dist_norm", "visual_dist_norm", "distribution_final"),
    )

    print("---- ***Final*** E4-C-DistFinal test accuracy: {:.2f}. ----\n".format(final_acc))

    _save_gpa_stats(
        args,
        gpa_cache_stats,
        entropy_cache,
        gpa_cache,
        gpa_local_cache,
        visual_dist,
        text_dist,
        score_norm_state,
        final_acc,
        gpa_event_records,
    )

    return final_acc
