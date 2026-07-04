"""
09_2: Explicit final-score variant of 02_9_2 / 09_1.

核心思想：
1. 保留原始 Point-Cache 的 Global Entropy Cache，仍用于 global cache logits；
2. 保留 Global Prototype-Alignment Cache，简称 GPA Cache；
3. 文本端使用每类 prompt-level embeddings 构建固定 text distribution；
4. 视觉端只累计曾被正缓存接受过的可信样本，构建 accepted-history visual distribution；
5. GPA Cache 未满时沿用 E3-V2-C / E4-A 的直接加入规则；
6. GPA Cache 满后使用“低熵 + text-visual joint score 更高”替换最高熵样本；
7. 只有进入 GPA Cache 的样本，其 patch_centers 才写入 local cache；
8. 仅新增显式最终得分公式，不修改缓存更新、gate 或 score：
   y = y_zs + alpha_g * y_g + alpha_l * y_l - alpha_n * y_n。
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import *  # noqa: F401,F403

CENTER_SOURCE_LABEL = "Accepted-history text-visual class-wise distribution score"
GPA_VARIANT_NAME = "09_2-explicit-final-score-e4-c-a0-e1-textdist-only"
DIAG_MARKER = "DIAG_ONLY_REMOVE_FOR_RELEASE"


def _loss_value(loss):
    """Convert entropy tensor/scalar to python float for sorting and logging."""
    if torch.is_tensor(loss):
        return float(loss.detach().float().cpu().item())
    return float(loss)


# DIAG_ONLY_REMOVE_FOR_RELEASE_START
# These helpers are for capacity-ablation diagnosis only. They record cache
# pseudo-label quality and prediction-stage transitions, without changing any
# cache update rule or final logits. Remove this block for release code.
def _diag_label_int(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        x = value.detach()
        if x.numel() == 0:
            return None
        return int(x.view(-1)[0].cpu().item())
    return int(value)


def _diag_meta(pred, target):
    pred_int = _diag_label_int(pred)
    target_int = _diag_label_int(target)
    if pred_int is None or target_int is None:
        return None
    return {
        "diag_marker": DIAG_MARKER,
        "pred": pred_int,
        "target": target_int,
        "pred_correct": bool(pred_int == target_int),
    }


def _diag_attach_item(item, pred, target):
    meta = _diag_meta(pred, target)
    if meta is None:
        return item
    return list(item) + [meta]


def _diag_item_meta(item):
    if isinstance(item, (list, tuple)) and item:
        maybe_meta = item[-1]
        if isinstance(maybe_meta, dict) and maybe_meta.get("diag_marker") == DIAG_MARKER:
            return maybe_meta
    return None


def _diag_record_cache_decision(stats, phase, cache_name, decision, pred, target):
    key = f"diag_{phase}_{cache_name}_{decision}"
    stats[f"{key}_total"] += 1

    meta = _diag_meta(pred, target)
    if meta is None:
        stats[f"{key}_unknown"] += 1
        return

    if meta["pred_correct"]:
        stats[f"{key}_pred_correct"] += 1
        if cache_name == "neg":
            stats[f"{key}_potential_misfire"] += 1
    else:
        stats[f"{key}_pred_wrong"] += 1
        if cache_name == "neg":
            stats[f"{key}_potential_helpful"] += 1


def _diag_record_replaced_old_item(stats, phase, cache_name, old_item):
    key = f"diag_{phase}_{cache_name}_replace_old"
    meta = _diag_item_meta(old_item)
    if meta is None:
        stats[f"{key}_unknown"] += 1
        return

    if meta["pred_correct"]:
        stats[f"{key}_pred_correct"] += 1
        if cache_name == "neg":
            stats[f"{key}_potential_misfire"] += 1
    else:
        stats[f"{key}_pred_wrong"] += 1
        if cache_name == "neg":
            stats[f"{key}_potential_helpful"] += 1


def _diag_logits_pred(logits):
    return int(logits.detach().float().argmax(dim=1).view(-1)[0].cpu().item())


def _diag_record_prediction_stage(stats, phase, stage, logits, target):
    pred = _diag_logits_pred(logits)
    target_int = _diag_label_int(target)
    key = f"diag_{phase}_pred_{stage}"
    stats[f"{key}_total"] += 1

    if target_int is None:
        stats[f"{key}_unknown"] += 1
        return pred

    if pred == target_int:
        stats[f"{key}_correct"] += 1
    else:
        stats[f"{key}_wrong"] += 1
    return pred


def _diag_record_prediction_transition(stats, phase, transition, before_pred, after_pred, target):
    target_int = _diag_label_int(target)
    key = f"diag_{phase}_transition_{transition}"
    stats[f"{key}_total"] += 1

    if target_int is None:
        stats[f"{key}_unknown"] += 1
        return

    before_correct = before_pred == target_int
    after_correct = after_pred == target_int

    if before_correct and after_correct:
        suffix = "right_to_right"
    elif before_correct and not after_correct:
        suffix = "right_to_wrong"
    elif not before_correct and after_correct:
        suffix = "wrong_to_right"
    else:
        suffix = "wrong_to_wrong"

    stats[f"{key}_{suffix}"] += 1
    if before_pred != after_pred:
        stats[f"{key}_pred_changed"] += 1


def _diag_summarize_cache_quality(cache):
    total = 0
    correct = 0
    wrong = 0
    unknown = 0
    by_class = {}

    for class_index, items in sorted(cache.items(), key=lambda kv: kv[0]):
        class_summary = {"total": 0, "pred_correct": 0, "pred_wrong": 0, "unknown": 0}
        for item in items:
            total += 1
            class_summary["total"] += 1
            meta = _diag_item_meta(item)
            if meta is None:
                unknown += 1
                class_summary["unknown"] += 1
            elif meta["pred_correct"]:
                correct += 1
                class_summary["pred_correct"] += 1
            else:
                wrong += 1
                class_summary["pred_wrong"] += 1

        by_class[str(class_index)] = class_summary

    return {
        "diag_marker": DIAG_MARKER,
        "total": int(total),
        "pred_correct": int(correct),
        "pred_wrong": int(wrong),
        "unknown": int(unknown),
        "pred_correct_rate": None if total == 0 else float(correct / total),
        "pred_wrong_rate": None if total == 0 else float(wrong / total),
        "by_class": by_class,
    }
# DIAG_ONLY_REMOVE_FOR_RELEASE_END


def _get_gpa_stats_enabled():
    return os.environ.get("GPA_SAVE_STATS", "1") != "0"


def _read_int_env(name, default):
    raw = os.environ.get(name, None)
    if raw is None or str(raw).strip() == "":
        return int(default)
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _attr_or_env_int(args, attr_name, env_name, default):
    value = getattr(args, attr_name, None)
    if value is not None:
        return int(value)
    return _read_int_env(env_name, default)


def _resolve_cache_caps(args, pos_cfg, neg_cfg):
    """
    Resolve explicit 09_2 cache capacities.

    Capacity means samples per class. local_centers is the KMeans center count
    per local sample and is recorded separately from cache capacity.
    """
    default_pos_cap = int(pos_cfg["shot_capacity"])
    default_neg_cap = int(neg_cfg["shot_capacity"])

    entropy_cap = _attr_or_env_int(args, "e4_entropy_cap", "E4_ENTROPY_CAP", default_pos_cap)
    gpa_cap = _attr_or_env_int(args, "e4_gpa_cap", "E4_GPA_CAP", default_pos_cap)
    local_cap = _attr_or_env_int(args, "e4_local_cap", "E4_LOCAL_CAP", gpa_cap)
    neg_cap = _attr_or_env_int(args, "e4_neg_cap", "E4_NEG_CAP", default_neg_cap)
    local_centers = _attr_or_env_int(args, "n_cluster", "E4_LOCAL_CENTERS", 3)

    for name, value in {
        "entropy_cap": entropy_cap,
        "gpa_cap": gpa_cap,
        "neg_cap": neg_cap,
        "local_centers": local_centers,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if local_cap < 0:
        raise ValueError(f"local_cap must be non-negative, got {local_cap}")

    return {
        "entropy_cap": entropy_cap,
        "gpa_cap": gpa_cap,
        "local_cap": local_cap,
        "neg_cap": neg_cap,
        "local_centers": local_centers,
    }


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


def _update_entropy_cache(cache, pred, item, shot_capacity, stats, phase, target=None):
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
            _diag_record_cache_decision(stats, phase, "entropy", "add", pred, target)
            return True

        worst_ent = _loss_value(cache[pred][-1][1])
        curr_ent = _loss_value(item[1])

        if curr_ent < worst_ent:
            _diag_record_replaced_old_item(stats, phase, "entropy", cache[pred][-1])
            cache[pred][-1] = item
            _sort_cache_by_entropy(cache, pred)
            stats[f"{phase}_entropy_replace"] += 1
            _diag_record_cache_decision(stats, phase, "entropy", "replace", pred, target)
            return True

        stats[f"{phase}_entropy_reject"] += 1
        _diag_record_cache_decision(stats, phase, "entropy", "reject", pred, target)
        return False

    cache[pred] = [item]
    stats[f"{phase}_entropy_add"] += 1
    _diag_record_cache_decision(stats, phase, "entropy", "add", pred, target)
    return True


def _update_negative_cache(cache, pred, item, shot_capacity, stats, phase, target=None):
    """
    Negative cache 保持原始 Point-Cache 的低熵排序替换逻辑。
    item = [pc_feats, loss, prob_map]
    """
    if pred in cache:
        if len(cache[pred]) < shot_capacity:
            cache[pred].append(item)
            _sort_cache_by_entropy(cache, pred)
            stats[f"{phase}_neg_add"] += 1
            _diag_record_cache_decision(stats, phase, "neg", "add", pred, target)
            return True

        worst_ent = _loss_value(cache[pred][-1][1])
        curr_ent = _loss_value(item[1])

        if curr_ent < worst_ent:
            _diag_record_replaced_old_item(stats, phase, "neg", cache[pred][-1])
            cache[pred][-1] = item
            _sort_cache_by_entropy(cache, pred)
            stats[f"{phase}_neg_replace"] += 1
            _diag_record_cache_decision(stats, phase, "neg", "replace", pred, target)
            return True

        stats[f"{phase}_neg_reject"] += 1
        _diag_record_cache_decision(stats, phase, "neg", "reject", pred, target)
        return False

    cache[pred] = [item]
    stats[f"{phase}_neg_add"] += 1
    _diag_record_cache_decision(stats, phase, "neg", "add", pred, target)
    return True


def _update_local_cache_from_gpa_accept(local_cache, pred, local_item, local_cap, stats, phase):
    """
    Update local cache only after the corresponding sample is accepted by GPA.

    When local_cap == gpa_cap, this produces the same local sample set as the
    paired 02_9_2 update. Different local_cap values are retained for diagnostic
    experiments, but first-batch 09_2 settings keep local_cap == gpa_cap.
    """
    if local_cap <= 0:
        stats[f"{phase}_local_reject_capacity_zero"] += 1
        return False

    if pred in local_cache:
        if len(local_cache[pred]) < local_cap:
            local_cache[pred].append(local_item)
            _sort_local_cache_by_entropy(local_cache, pred)
            stats[f"{phase}_local_add"] += 1
            return True

        worst_ent = _loss_value(local_cache[pred][-1][1])
        curr_ent = _loss_value(local_item[1])

        if curr_ent < worst_ent:
            local_cache[pred][-1] = local_item
            _sort_local_cache_by_entropy(local_cache, pred)
            stats[f"{phase}_local_replace"] += 1
            return True

        stats[f"{phase}_local_reject"] += 1
        return False

    local_cache[pred] = [local_item]
    stats[f"{phase}_local_add"] += 1
    return True




# ============================================================
# E4-C：Accepted-History Text-Visual 类别概率分布辅助函数
# ============================================================

DIST_EPS = float(os.environ.get("E4_DIST_EPS", "1e-4"))
DIST_MIN_VAR = float(os.environ.get("E4_DIST_MIN_VAR", "1e-4"))
TEXT_DIST_EPS = float(os.environ.get("E4_TEXT_DIST_EPS", str(DIST_EPS)))
TEXT_DIST_MIN_VAR = float(os.environ.get("E4_TEXT_DIST_MIN_VAR", str(DIST_MIN_VAR)))
TEXT_SCORE_WEIGHT = float(os.environ.get("E4_TEXT_SCORE_WEIGHT", "0.1"))
TEXT_GATE_MODE = os.environ.get("E4_TEXT_GATE_MODE", "distribution").strip().lower()
TEXT_PROTO_SCORE_SCALE = float(os.environ.get("E4_TEXT_PROTO_SCORE_SCALE", "1.0"))
SCORE_NORM_MODE = os.environ.get("E4_SCORE_NORM_MODE", "none").strip().lower()
SCORE_NORM_MIN_COUNT = int(os.environ.get("E4_SCORE_NORM_MIN_COUNT", "8"))
SCORE_NORM_EPS = float(os.environ.get("E4_SCORE_NORM_EPS", "1e-6"))
SCORE_NORM_CLIP = float(os.environ.get("E4_SCORE_NORM_CLIP", "0"))

if TEXT_GATE_MODE not in {"distribution", "fused_prototype"}:
    raise ValueError(f"Unsupported E4_TEXT_GATE_MODE: {TEXT_GATE_MODE}")

if SCORE_NORM_MODE not in {"none", "running_zscore"}:
    raise ValueError(f"Unsupported E4_SCORE_NORM_MODE: {SCORE_NORM_MODE}")


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
    result = {
        "count": int(entry["count"]),
        "mean": entry["mean"].to(device=ref_feat.device, dtype=ref_feat.dtype),
        "var": entry["var"].to(device=ref_feat.device, dtype=ref_feat.dtype),
    }
    if "prototype" in entry and entry["prototype"] is not None:
        prototype = entry["prototype"]
        if not torch.is_tensor(prototype):
            prototype = torch.as_tensor(prototype)
        result["prototype"] = prototype.to(device=ref_feat.device, dtype=ref_feat.dtype)
    if "prototype_source" in entry:
        result["prototype_source"] = entry["prototype_source"]
    return result


def _distribution_score_from_entry(entry, feat, eps):
    if entry is None or int(entry["count"]) < 2:
        return None

    x = _feature_float(feat).to(device=entry["mean"].device, dtype=entry["mean"].dtype)
    raw = torch.mean(((x - entry["mean"]) ** 2) / (entry["var"] + eps))
    return float((-raw).detach().cpu().item())


def _prototype_score_from_entry(entry, feat):
    if entry is None or "prototype" not in entry or entry["prototype"] is None:
        return None

    prototype = entry["prototype"]
    if prototype.dim() == 1:
        prototype = prototype.view(1, -1)

    x = _feature_float(feat).to(device=prototype.device, dtype=prototype.dtype)
    if x.dim() == 1:
        x = x.view(1, -1)

    if x.size(-1) != prototype.size(-1):
        return None

    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    prototype = prototype / prototype.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    score = (x * prototype).sum(dim=-1).mean() * TEXT_PROTO_SCORE_SCALE
    return float(score.detach().cpu().item())


def _joint_distribution_score(visual_dist, text_dist, pred, feat, score_norm_state=None):
    """
    计算 E4-C 的 text-visual joint score。

    visual_dist 来自被正缓存接受过的可信历史样本；
    text_dist 默认来自固定 prompt-level embeddings；当 E4_TEXT_GATE_MODE=fused_prototype
    时，文本分数改用 E1 branch-wise fused prototype cosine score。
    """
    visual_entry = _visual_distribution(visual_dist, pred)
    text_entry = _text_distribution(text_dist, pred, feat)

    visual_score = _distribution_score_from_entry(visual_entry, feat, DIST_EPS)
    if TEXT_GATE_MODE == "fused_prototype":
        text_score = _prototype_score_from_entry(text_entry, feat)
        text_score_kind = "fused_prototype_cosine"
    else:
        text_score = _distribution_score_from_entry(text_entry, feat, TEXT_DIST_EPS)
        text_score_kind = "prompt_distribution"

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
        "text_gate_mode": TEXT_GATE_MODE,
        "text_score_kind": text_score_kind,
        "text_proto_score_scale": float(TEXT_PROTO_SCORE_SCALE),
    }


def _summarize_distribution_from_entry(entry):
    if entry is None:
        return None

    var = entry["var"].detach().float().cpu()
    summary = {
        "count": int(entry["count"]),
        "var_mean": float(var.mean().item()),
        "var_min": float(var.min().item()),
        "var_max": float(var.max().item()),
    }
    prototype = entry.get("prototype")
    if prototype is not None:
        if not torch.is_tensor(prototype):
            prototype = torch.as_tensor(prototype)
        proto = prototype.detach().float().cpu()
        summary["has_prototype"] = True
        summary["prototype_norm"] = float(proto.norm(dim=-1).mean().item())
        summary["prototype_source"] = entry.get("prototype_source")
    else:
        summary["has_prototype"] = False
    return summary


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
    gpa_cap,
    local_cap,
    stats,
    phase,
    event_records=None,
    target=None,
):
    """
    E4-C：Accepted-History Text-Visual 类别概率分布引导的 GPA-Cache 更新。

    沿用 E3-V2-C / E4-A：
        1. GPA 未满直接加入 GPA-Cache；
        2. 满后替换最高熵样本；
        3. 保留低熵门控。

    改动：
        用曾被正缓存接受过的历史可信视觉分布，以及固定 prompt 文本分布共同计算 joint score。
        local cache 只在 GPA 接受样本后更新，并由 local_cap 独立限制容量。
    """
    if pred not in gpa_cache:
        gpa_cache[pred] = []
        gpa_local_cache[pred] = []

    curr_ent = _loss_value(global_item[1])

    def record_event(decision, old_entropy=None, new_score=None, old_score=None, old_item=None):
        if event_records is None:
            return
        new_meta = _diag_meta(pred, target)
        old_meta = _diag_item_meta(old_item)
        event_records.append({
            "phase": phase,
            "class_index": int(pred),
            "decision": decision,
            "diag_marker": DIAG_MARKER,
            "new_pred": None if new_meta is None else int(new_meta["pred"]),
            "new_target": None if new_meta is None else int(new_meta["target"]),
            "new_pred_correct": None if new_meta is None else bool(new_meta["pred_correct"]),
            "old_pred": None if old_meta is None else int(old_meta["pred"]),
            "old_target": None if old_meta is None else int(old_meta["target"]),
            "old_pred_correct": None if old_meta is None else bool(old_meta["pred_correct"]),
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
            "text_gate_mode": TEXT_GATE_MODE,
            "text_score_kind": None if new_score is None else new_score.get("text_score_kind"),
            "text_proto_score_scale": float(TEXT_PROTO_SCORE_SCALE),
            "score_norm_mode": SCORE_NORM_MODE,
            "joint_score_margin": None if new_score is None or old_score is None else float(new_score["joint"] - old_score["joint"]),
            "visual_count": None if new_score is None else int(new_score["visual_count"]),
            "text_count": None if new_score is None else int(new_score["text_count"]),
            "gpa_cap": int(gpa_cap),
            "local_cap": int(local_cap),
        })

    if len(gpa_cache[pred]) < gpa_cap:
        gpa_cache[pred].append(global_item)
        _sort_cache_by_entropy(gpa_cache, pred)
        _update_local_cache_from_gpa_accept(gpa_local_cache, pred, local_item, local_cap, stats, phase)

        stats[f"{phase}_gpa_add_not_full"] += 1
        stats[f"{phase}_gpa_add_not_full_accepted_history_text_visual_distribution"] += 1
        _diag_record_cache_decision(stats, phase, "gpa", "add_not_full", pred, target)
        _update_visual_distribution(visual_dist, pred, global_item[0], stats, phase, "gpa_add")
        record_event(decision="add_not_full_accepted_history_text_visual_distribution")
        return True

    worst_global_item = gpa_cache[pred][-1]
    worst_ent = _loss_value(worst_global_item[1])

    curr_score = _joint_distribution_score(visual_dist, text_dist, pred, global_item[0], score_norm_state)
    worst_score = _joint_distribution_score(visual_dist, text_dist, pred, worst_global_item[0], score_norm_state)

    if curr_score is None or worst_score is None:
        stats[f"{phase}_gpa_reject_no_accepted_history_text_visual_distribution"] += 1
        _diag_record_cache_decision(stats, phase, "gpa", "reject_no_distribution", pred, target)
        record_event(
            decision="reject_no_accepted_history_text_visual_distribution",
            old_entropy=worst_ent,
            new_score=curr_score,
            old_score=worst_score,
            old_item=worst_global_item,
        )
        return False

    if curr_ent >= worst_ent:
        stats[f"{phase}_gpa_reject_entropy"] += 1
        stats[f"{phase}_gpa_reject_entropy_accepted_history_text_visual_distribution"] += 1
        _diag_record_cache_decision(stats, phase, "gpa", "reject_entropy", pred, target)
        record_event(
            decision="reject_entropy_accepted_history_text_visual_distribution",
            old_entropy=worst_ent,
            new_score=curr_score,
            old_score=worst_score,
            old_item=worst_global_item,
        )
        return False

    norm_updates = _update_score_norm_state(score_norm_state, curr_score)
    norm_updates += _update_score_norm_state(score_norm_state, worst_score)
    if norm_updates:
        stats[f"{phase}_score_norm_update"] += norm_updates
        stats[f"{phase}_score_norm_observed_pairs"] += 1

    if curr_score["joint"] > worst_score["joint"]:
        _diag_record_replaced_old_item(stats, phase, "gpa", worst_global_item)
        gpa_cache[pred][-1] = global_item
        _sort_cache_by_entropy(gpa_cache, pred)
        _update_local_cache_from_gpa_accept(gpa_local_cache, pred, local_item, local_cap, stats, phase)

        stats[f"{phase}_gpa_replace_accepted_history_text_visual_distribution"] += 1
        _diag_record_cache_decision(stats, phase, "gpa", "replace", pred, target)
        _update_visual_distribution(visual_dist, pred, global_item[0], stats, phase, "gpa_replace")
        record_event(
            decision="replace_accepted_history_text_visual_distribution",
            old_entropy=worst_ent,
            new_score=curr_score,
            old_score=worst_score,
            old_item=worst_global_item,
        )
        return True

    stats[f"{phase}_gpa_reject_accepted_history_text_visual_distribution"] += 1
    _diag_record_cache_decision(stats, phase, "gpa", "reject_joint", pred, target)
    record_event(
        decision="reject_accepted_history_text_visual_distribution",
        old_entropy=worst_ent,
        new_score=curr_score,
        old_score=worst_score,
        old_item=worst_global_item,
    )
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
    capacity_summary=None,
    neg_cache=None,
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
        "diagnostic_marker": DIAG_MARKER,
        "diagnostic_only_remove_for_release": True,
        "diagnostic_stats_version": "cache_quality_and_prediction_transitions_v1",
        "e4_text_gate_mode": TEXT_GATE_MODE,
        "e4_dist_eps": float(DIST_EPS),
        "e4_dist_min_var": float(DIST_MIN_VAR),
        "e4_text_dist_eps": float(TEXT_DIST_EPS),
        "e4_text_dist_min_var": float(TEXT_DIST_MIN_VAR),
        "e4_text_score_weight": float(TEXT_SCORE_WEIGHT),
        "e4_text_proto_score_scale": float(TEXT_PROTO_SCORE_SCALE),
        "e4_score_norm_mode": SCORE_NORM_MODE,
        "e4_score_norm_min_count": int(SCORE_NORM_MIN_COUNT),
        "e4_score_norm_eps": float(SCORE_NORM_EPS),
        "e4_score_norm_clip": float(SCORE_NORM_CLIP),
        "capacity_summary": {} if capacity_summary is None else dict(capacity_summary),
        "entropy_cap": None if capacity_summary is None else int(capacity_summary["entropy_cap"]),
        "gpa_cap": None if capacity_summary is None else int(capacity_summary["gpa_cap"]),
        "local_cap": None if capacity_summary is None else int(capacity_summary["local_cap"]),
        "neg_cap": None if capacity_summary is None else int(capacity_summary["neg_cap"]),
        "local_centers": None if capacity_summary is None else int(capacity_summary["local_centers"]),
        "score_normalization_summary": _summarize_score_norm_state(score_norm_state),
        "center_source": CENTER_SOURCE_LABEL,
        "final_acc": acc,
        "stats": dict(stats),
        "entropy_cache_class_counts": _summarize_cache(entropy_cache),
        "gpa_cache_class_counts": _summarize_cache(gpa_cache),
        "gpa_local_cache_class_counts": _summarize_cache(gpa_local_cache),
        "neg_cache_class_counts": {} if neg_cache is None else _summarize_cache(neg_cache),
        "entropy_cache_total": int(sum(len(v) for v in entropy_cache.values())),
        "gpa_cache_total": int(sum(len(v) for v in gpa_cache.values())),
        "gpa_local_cache_total": int(sum(len(v) for v in gpa_local_cache.values())),
        "neg_cache_total": 0 if neg_cache is None else int(sum(len(v) for v in neg_cache.values())),
        "diagnostic_cache_quality": {
            "diag_marker": DIAG_MARKER,
            "entropy_cache": _diag_summarize_cache_quality(entropy_cache),
            "gpa_cache": _diag_summarize_cache_quality(gpa_cache),
            "gpa_local_cache": _diag_summarize_cache_quality(gpa_local_cache),
            "negative_cache": _diag_summarize_cache_quality({} if neg_cache is None else neg_cache),
        },
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
def build_cache_in_advance(
    args,
    test_loader,
    lm3d_model,
    clip_weights,
    entropy_cap,
    gpa_cap,
    local_cap,
    include_prob_map=False,
    neg_cap=None,
    text_dist=None,
):
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
        if neg_cap is None:
            raise ValueError("neg_cap must be provided when include_prob_map=True")
        neg_cache = {}

        for pc, target, _, rgb in test_loader:
            feature = torch.cat([pc, rgb], dim=-1).half()
            pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

            item = _diag_attach_item([pc_feats, loss, prob_map], pred, target)
            _update_negative_cache(neg_cache, pred, item, neg_cap, stats, "build", target=target)

            cache_num = sum(len(neg_cache[key]) for key in neg_cache)
            num_classes = clip_logits.size(1)
            full_num = neg_cap * num_classes

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

    for pc, target, _, rgb in test_loader:
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        global_item = _diag_attach_item([pc_feats, loss], pred, target)
        local_item = _diag_attach_item([patch_centers, loss], pred, target)

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
            gpa_cap,
            local_cap,
            stats,
            "build",
            gpa_event_records,
            target=target,
        )

        entropy_accepted = _update_entropy_cache(
            entropy_cache, pred, global_item, entropy_cap, stats, "build", target=target
        )
        if entropy_accepted:
            _update_visual_distribution(visual_dist, pred, global_item[0], stats, "build", "entropy_accept")

        entropy_cache_num = sum(len(entropy_cache[key]) for key in entropy_cache)
        gpa_cache_num = sum(len(gpa_cache[key]) for key in gpa_cache)
        local_cache_num = sum(len(gpa_local_cache[key]) for key in gpa_local_cache)
        num_classes = clip_logits.size(1)
        entropy_full_num = entropy_cap * num_classes
        gpa_full_num = gpa_cap * num_classes
        local_full_num = local_cap * num_classes

        if (
            entropy_cache_num >= entropy_full_num
            and gpa_cache_num >= gpa_full_num
            and local_cache_num >= local_full_num
        ):
            print("*" * 10, "Building decoupled positive caches is Done!", "*" * 10, "\n")
            break

    return entropy_cache, gpa_cache, gpa_local_cache, visual_dist, score_norm_state, stats, gpa_event_records


def _default_final_score_weights(pos_cfg, neg_cfg):
    return [{
        "name": "ag2p0_al2p0_an0p117",
        "alpha_g": float(pos_cfg["alpha"]),
        "alpha_l": float(pos_cfg["alpha"]),
        "alpha_n": float(neg_cfg["alpha"]),
    }]


def _resolve_final_score_weights(args, pos_cfg, neg_cfg):
    weights = getattr(args, "e4_final_score_weights", None)
    if weights:
        return weights
    return _default_final_score_weights(pos_cfg, neg_cfg)


@torch.no_grad()
def compute_cache_score(pc_feats, cache, beta, clip_weights, neg_mask_thresholds=None):
    """Compute an unweighted global-cache score term: y_g or y_n."""
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
    cache_score = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return cache_score


@torch.no_grad()
def compute_local_cache_score(patch_centers, local_cache, beta, clip_weights):
    """Compute an unweighted GPA-controlled local-cache score term: y_l."""
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
    local_cache_score = ((-1) * (beta - beta * affinity)).exp() @ local_cache_values
    return local_cache_score


@torch.no_grad()
def run_test_tda(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights, text_dist=None):
    """
    E4-C test-time adaptation.

    Global logits:
        使用 Global Entropy Cache，即原始 Point-Cache 的正全局缓存。

    Local logits:
        使用 GPA-controlled Local Cache，即只有进入 GPA Cache 的样本贡献局部特征。

    Negative cache:
        保持原始 Point-Cache 逻辑。
    """
    capacity_summary = _resolve_cache_caps(args, pos_cfg, neg_cfg)
    entropy_cap = capacity_summary["entropy_cap"]
    gpa_cap = capacity_summary["gpa_cap"]
    local_cap = capacity_summary["local_cap"]
    neg_cap = capacity_summary["neg_cap"]

    print("[09_2] explicit final-score cache capacities:", capacity_summary)

    entropy_cache, gpa_cache, gpa_local_cache, visual_dist, score_norm_state, build_stats, gpa_event_records = build_cache_in_advance(
        args,
        test_loader,
        lm3d_model,
        clip_weights,
        entropy_cap,
        gpa_cap,
        local_cap,
        text_dist=text_dist,
    )

    print("[E4-C] len(entropy_cache):", len(entropy_cache))
    print("[E4-C] len(gpa_cache):", len(gpa_cache))
    print("[E4-C] len(gpa_local_cache):", len(gpa_local_cache))
    print("[E4-C] entropy cache total:", sum(len(v) for v in entropy_cache.values()))
    print("[E4-C] gpa cache total:", sum(len(v) for v in gpa_cache.values()))
    print("[E4-C] gpa local cache total:", sum(len(v) for v in gpa_local_cache.values()))
    print("[E4-C] visual distribution classes:", len(visual_dist))
    print("[E4-C] text distribution classes:", 0 if text_dist is None else len(text_dist))
    print("[E4-C] text gate mode:", TEXT_GATE_MODE)
    print("[E4-C] text proto score scale:", TEXT_PROTO_SCORE_SCALE)
    print("[E4-C] score norm mode:", SCORE_NORM_MODE)
    print("[E4-C] score norm state:", _summarize_score_norm_state(score_norm_state))

    neg_cache = {}
    gpa_cache_stats = defaultdict(int)
    for k, v in build_stats.items():
        gpa_cache_stats[k] += v

    # E4-C 保留 build 阶段形成的 EntropyCache、GPA-Cache、GPA-local-cache 和
    # accepted-history visual_dist。visual_dist 只累计曾被正缓存接受过的样本。
    final_score_weights = _resolve_final_score_weights(args, pos_cfg, neg_cfg)
    if not final_score_weights:
        raise ValueError("At least one final-score weight setting is required.")

    primary_weight = final_score_weights[0]
    accuracies_by_weight = {weight["name"]: [] for weight in final_score_weights}
    capacity_summary["final_score_formula"] = "y = y_zs + alpha_g * y_g + alpha_l * y_l - alpha_n * y_n"
    capacity_summary["final_score_weights"] = final_score_weights
    capacity_summary["primary_final_score_weight"] = primary_weight

    print("[E4-C] explicit final-score formula: y = y_zs + alpha_g * y_g + alpha_l * y_l - alpha_n * y_n")
    print("[E4-C] final-score weight settings:")
    for weight in final_score_weights:
        print(
            "  {name}: alpha_g={alpha_g}, alpha_l={alpha_l}, alpha_n={alpha_n}".format(
                **weight
            )
        )

    pos_enabled, neg_enabled = pos_cfg["enabled"], neg_cfg["enabled"]

    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ["beta"]}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ["beta", "entropy_threshold", "mask_threshold"]}

    for i, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()

        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)

        if pos_enabled:
            global_item = _diag_attach_item([pc_feats, loss], pred, target)
            local_item = _diag_attach_item([patch_centers, loss], pred, target)

            _update_gpa_cache(
                gpa_cache,
                gpa_local_cache,
                visual_dist,
                text_dist,
                score_norm_state,
                pred,
                global_item,
                local_item,
                gpa_cap,
                local_cap,
                gpa_cache_stats,
                "test",
                gpa_event_records,
                target=target,
            )

            entropy_accepted = _update_entropy_cache(
                entropy_cache, pred, global_item, entropy_cap, gpa_cache_stats, "test", target=target
            )
            if entropy_accepted:
                _update_visual_distribution(visual_dist, pred, global_item[0], gpa_cache_stats, "test", "entropy_accept")
        if neg_enabled and neg_params["entropy_threshold"]["lower"] < prop_entropy < neg_params["entropy_threshold"]["upper"]:
            _update_negative_cache(
                neg_cache,
                pred,
                _diag_attach_item([pc_feats, loss, prob_map], pred, target),
                neg_cap,
                gpa_cache_stats,
                "test",
                target=target,
            )

        y_zs = clip_logits.clone()
        y_g = torch.zeros_like(y_zs)
        y_l = torch.zeros_like(y_zs)
        y_n = torch.zeros_like(y_zs)

        diag_logits = y_zs.clone()
        zero_pred = _diag_record_prediction_stage(gpa_cache_stats, "test", "zero_shot", diag_logits, target)
        prev_stage_name = "zero_shot"
        prev_pred = zero_pred

        if pos_enabled and entropy_cache:
            y_g = compute_cache_score(
                pc_feats,
                entropy_cache,
                pos_params["beta"],
                clip_weights
            )
            diag_logits = diag_logits + primary_weight["alpha_g"] * y_g
            entropy_pred = _diag_record_prediction_stage(gpa_cache_stats, "test", "after_entropy", diag_logits, target)
            _diag_record_prediction_transition(
                gpa_cache_stats, "test", f"{prev_stage_name}_to_after_entropy", prev_pred, entropy_pred, target
            )
            prev_stage_name = "after_entropy"
            prev_pred = entropy_pred

            if gpa_local_cache:
                y_l = compute_local_cache_score(
                    patch_centers,
                    gpa_local_cache,
                    pos_params["beta"],
                    clip_weights
                )
                diag_logits = diag_logits + primary_weight["alpha_l"] * y_l
                local_pred = _diag_record_prediction_stage(gpa_cache_stats, "test", "after_local", diag_logits, target)
                _diag_record_prediction_transition(
                    gpa_cache_stats, "test", f"{prev_stage_name}_to_after_local", prev_pred, local_pred, target
                )
                prev_stage_name = "after_local"
                prev_pred = local_pred

        if neg_enabled and neg_cache:
            y_n = compute_cache_score(
                pc_feats,
                neg_cache,
                neg_params["beta"],
                clip_weights,
                (neg_params["mask_threshold"]["lower"], neg_params["mask_threshold"]["upper"])
            )
            diag_logits = diag_logits - primary_weight["alpha_n"] * y_n
            neg_pred = _diag_record_prediction_stage(gpa_cache_stats, "test", "after_negative", diag_logits, target)
            _diag_record_prediction_transition(
                gpa_cache_stats, "test", f"{prev_stage_name}_to_after_negative", prev_pred, neg_pred, target
            )
            prev_stage_name = "after_negative"
            prev_pred = neg_pred

        final_pred = _diag_record_prediction_stage(gpa_cache_stats, "test", "final", diag_logits, target)
        _diag_record_prediction_transition(
            gpa_cache_stats, "test", "zero_shot_to_final", zero_pred, final_pred, target
        )

        for weight in final_score_weights:
            final_logits = (
                y_zs
                + weight["alpha_g"] * y_g
                + weight["alpha_l"] * y_l
                - weight["alpha_n"] * y_n
            )
            acc = cls_acc(final_logits, target)
            accuracies_by_weight[weight["name"]].append(acc)

        if i % args.print_freq == 0:
            primary_acc = sum(accuracies_by_weight[primary_weight["name"]]) / len(accuracies_by_weight[primary_weight["name"]])
            print("---- E4-C primary test accuracy: {:.2f}. ----\n".format(primary_acc))

    final_score_results = []
    for weight in final_score_weights:
        values = accuracies_by_weight[weight["name"]]
        final_score_results.append({
            "name": weight["name"],
            "alpha_g": weight["alpha_g"],
            "alpha_l": weight["alpha_l"],
            "alpha_n": weight["alpha_n"],
            "acc": sum(values) / len(values),
        })

    primary_result = final_score_results[0]
    final_acc = primary_result["acc"]
    print("---- ***Final*** E4-C primary test accuracy: {:.2f}. ----\n".format(final_acc))
    print("---- Final-score weight sweep results ----")
    for result in final_score_results:
        print(
            "{name}: alpha_g={alpha_g}, alpha_l={alpha_l}, alpha_n={alpha_n}, acc={acc:.2f}".format(
                **result
            )
        )

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
        capacity_summary,
        neg_cache=neg_cache,
    )

    return {
        "primary_acc": final_acc,
        "primary_weight": primary_weight,
        "weight_results": final_score_results,
    }
