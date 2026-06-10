"""
E4-C-A0-c2: Z-score Accepted-History Text-Visual GPA Cache with Prior-Corrected GPA Global Evidence.

核心思想：
1. 保留原始 Point-Cache 的 Global Entropy Cache，仍用于 global cache logits；
2. 保留 Global Prototype-Alignment Cache，简称 GPA Cache；
3. 文本端使用每类 prompt-level embeddings 构建固定 text distribution；
4. 视觉端只累计曾被正缓存接受过的可信样本，构建 accepted-history visual distribution；
5. GPA Cache 未满时沿用 E3-V2-C / E4-A 的直接加入规则；
6. GPA Cache 满后使用“低熵 + text-visual joint score 更高”替换最高熵样本；
7. 只有进入 GPA Cache 的样本，其 patch_centers 才写入 local cache；
8. 同一次测试同时输出原始最终 logits、raw GPA global logits，以及 prior-corrected GPA global evidence。
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

CENTER_SOURCE_LABEL = "Accepted-history text-visual z-score distribution score with prior-corrected GPA global logits comparison"
GPA_VARIANT_NAME = "E4-C-A0-c2-zscore-accepted-history-text-visual-prior-corrected-gpa-global-logits"
ORIGINAL_FORMULA = "original_formula"
RAW_GPA_GLOBAL_FORMULA = "raw_gpa_global_logits"


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
# E4-C-A0-c2：Accepted-History Text-Visual 类别概率分布辅助函数
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
GPA_GLOBAL_EPS = float(os.environ.get("E4_GPA_GLOBAL_EPS", os.environ.get("GPA_GLOBAL_EPS", "1e-12")))
GPA_GLOBAL_PRIOR = os.environ.get("E4_GPA_GLOBAL_PRIOR", os.environ.get("GPA_GLOBAL_PRIOR", "uniform")).strip().lower()
GPA_GLOBAL_BETA_OVERRIDE = os.environ.get("E4_GPA_GLOBAL_BETA", os.environ.get("GPA_GLOBAL_BETA", "")).strip()
GPA_GLOBAL_GAMMAS_RAW = os.environ.get("E4_GPA_GLOBAL_GAMMAS", os.environ.get("GPA_GLOBAL_GAMMAS", "0.1,0.25,0.5"))

if SCORE_NORM_MODE not in {"none", "running_zscore"}:
    raise ValueError(f"Unsupported E4_SCORE_NORM_MODE: {SCORE_NORM_MODE}")

if GPA_GLOBAL_PRIOR != "uniform":
    raise ValueError(f"Unsupported E4_GPA_GLOBAL_PRIOR: {GPA_GLOBAL_PRIOR}. Use 'uniform' for E4-C-A0-c2.")


def _parse_gamma_list(raw):
    gammas = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        value = float(text)
        if value < 0:
            raise ValueError(f"E4_GPA_GLOBAL_GAMMAS must be non-negative, got {value}")
        gammas.append(value)
    if not gammas:
        raise ValueError("E4_GPA_GLOBAL_GAMMAS must contain at least one value")
    return tuple(gammas)


GPA_GLOBAL_GAMMAS = _parse_gamma_list(GPA_GLOBAL_GAMMAS_RAW)


def _gamma_label(gamma):
    return f"{float(gamma):g}"


def _prior_corrected_formula_label(gamma):
    return f"prior_corrected_gpa_global_gamma_{_gamma_label(gamma)}"


def get_formula_labels():
    return [ORIGINAL_FORMULA, RAW_GPA_GLOBAL_FORMULA] + [
        _prior_corrected_formula_label(gamma) for gamma in GPA_GLOBAL_GAMMAS
    ]


def get_formula_summary_slug(formula):
    if formula == ORIGINAL_FORMULA:
        return "original"
    if formula == RAW_GPA_GLOBAL_FORMULA:
        return "raw_gpa_global"
    prefix = "prior_corrected_gpa_global_gamma_"
    if formula.startswith(prefix):
        return "prior_corrected_gamma_" + formula[len(prefix):]
    return formula.replace(".", "_")


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
    计算 E4-C-A0-c2 的 text-visual joint score。

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
    E4-C-A0-c2：Accepted-History Text-Visual 类别概率分布引导的 GPA-Cache 更新。

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


def _gpa_global_beta(default_beta):
    if GPA_GLOBAL_BETA_OVERRIDE:
        return float(GPA_GLOBAL_BETA_OVERRIDE)
    return float(default_beta)


@torch.no_grad()
def compute_prior_corrected_gpa_global_evidence(pc_feats, gpa_cache, beta, clip_weights):
    """
    Compute e_G(c, x) = log(r_c(x) + eps) - log(m(x) + eps).

    r_c(x) is the class-size-normalized GPA global kernel density:
        r_c(x) = mean_{g_i in G_c} exp(beta * (cos(x, g_i) - 1)).

    With a uniform prior pi_c = 1 / C:
        m(x) = sum_j pi_j r_j(x).

    The returned evidence is log(q_G(c|x) / pi_c), so larger values support
    class c and can be added to logits with a positive gamma.
    """
    num_classes = clip_weights.size(1)
    pc_feats_float = pc_feats.detach().float()
    evidence = torch.zeros((pc_feats.size(0), num_classes), device=pc_feats.device, dtype=torch.float32)

    if not gpa_cache:
        stats = {
            "num_classes": int(num_classes),
            "nonempty_classes": 0,
            "empty_classes": int(num_classes),
            "normalizer_min": None,
            "normalizer_max": None,
        }
        return evidence, stats

    density = torch.zeros_like(evidence)
    nonempty_classes = 0

    for class_index in range(num_classes):
        entries = gpa_cache.get(class_index, [])
        if not entries:
            continue

        keys = torch.cat([item[0] for item in entries], dim=0).detach().float().permute(1, 0)
        affinity = pc_feats_float @ keys
        kernel = torch.exp(float(beta) * (affinity - 1.0))
        density[:, class_index] = kernel.mean(dim=1)
        nonempty_classes += 1

    prior = 1.0 / float(num_classes)
    normalizer = density.sum(dim=1, keepdim=True) * prior
    evidence = torch.log(density.clamp_min(GPA_GLOBAL_EPS)) - torch.log(normalizer.clamp_min(GPA_GLOBAL_EPS))

    stats = {
        "num_classes": int(num_classes),
        "nonempty_classes": int(nonempty_classes),
        "empty_classes": int(num_classes - nonempty_classes),
        "normalizer_min": float(normalizer.detach().float().min().cpu().item()),
        "normalizer_max": float(normalizer.detach().float().max().cpu().item()),
        "density_min": float(density.detach().float().min().cpu().item()),
        "density_max": float(density.detach().float().max().cpu().item()),
    }
    return evidence, stats


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
        "e4_variant": "E4-C-A0-c2",
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
        "gpa_global_evidence_mode": "prior_corrected_log_likelihood_ratio",
        "gpa_global_prior": GPA_GLOBAL_PRIOR,
        "gpa_global_eps": float(GPA_GLOBAL_EPS),
        "gpa_global_beta_override": None if not GPA_GLOBAL_BETA_OVERRIDE else float(GPA_GLOBAL_BETA_OVERRIDE),
        "gpa_global_gammas": [float(gamma) for gamma in GPA_GLOBAL_GAMMAS],
        "formula_labels": get_formula_labels(),
        "score_normalization_summary": _summarize_score_norm_state(score_norm_state),
        "center_source": CENTER_SOURCE_LABEL,
        "final_acc": acc,
        "final_acc_original": None if not isinstance(acc, dict) else acc.get("original"),
        "final_acc_raw_gpa_global": None if not isinstance(acc, dict) else acc.get("raw_gpa_global"),
        "final_acc_prior_corrected": {
            _gamma_label(gamma): None if not isinstance(acc, dict) else acc.get(_prior_corrected_formula_label(gamma))
            for gamma in GPA_GLOBAL_GAMMAS
        },
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
        print(f"[E4-C-A0-c2] Saved GPA replacement events to {event_path}")

    print(f"[E4-C-A0-c2] Saved GPA stats to {out_dir / filename}")


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

        # E4-C-A0-c2：GPA 使用已被正缓存接受过的历史可信 visual_dist 评分。
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


@torch.no_grad()
def run_test_tda(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights, text_dist=None):
    """
    E4-C-A0-c2 test-time adaptation.

    Global logits:
        使用 Global Entropy Cache，即原始 Point-Cache 的正全局缓存。

    Local logits:
        使用 GPA-controlled Local Cache，即只有进入 GPA Cache 的样本贡献局部特征。

    Negative cache:
        保持原始 Point-Cache 逻辑。
    """
    entropy_cache, gpa_cache, gpa_local_cache, visual_dist, score_norm_state, build_stats, gpa_event_records = build_cache_in_advance(
        args, test_loader, lm3d_model, clip_weights, pos_cfg["shot_capacity"], text_dist=text_dist
    )

    print("[E4-C-A0-c2] len(entropy_cache):", len(entropy_cache))
    print("[E4-C-A0-c2] len(gpa_cache):", len(gpa_cache))
    print("[E4-C-A0-c2] len(gpa_local_cache):", len(gpa_local_cache))
    print("[E4-C-A0-c2] entropy cache total:", sum(len(v) for v in entropy_cache.values()))
    print("[E4-C-A0-c2] gpa cache total:", sum(len(v) for v in gpa_cache.values()))
    print("[E4-C-A0-c2] gpa local cache total:", sum(len(v) for v in gpa_local_cache.values()))
    print("[E4-C-A0-c2] visual distribution classes:", len(visual_dist))
    print("[E4-C-A0-c2] text distribution classes:", 0 if text_dist is None else len(text_dist))
    print("[E4-C-A0-c2] score norm mode:", SCORE_NORM_MODE)
    print("[E4-C-A0-c2] score norm state:", _summarize_score_norm_state(score_norm_state))
    print("[E4-C-A0-c2] GPA global prior:", GPA_GLOBAL_PRIOR)
    print("[E4-C-A0-c2] GPA global eps:", GPA_GLOBAL_EPS)
    print("[E4-C-A0-c2] GPA global beta override:", GPA_GLOBAL_BETA_OVERRIDE or "positive.beta")
    print("[E4-C-A0-c2] GPA global gammas:", ", ".join(_gamma_label(gamma) for gamma in GPA_GLOBAL_GAMMAS))

    neg_cache = {}
    gpa_cache_stats = defaultdict(int)
    for k, v in build_stats.items():
        gpa_cache_stats[k] += v

    # E4-C-A0-c2 保留 build 阶段形成的 EntropyCache、GPA-Cache、GPA-local-cache 和
    # accepted-history visual_dist。visual_dist 只累计曾被正缓存接受过的样本。
    formula_labels = get_formula_labels()
    accuracies = {formula: [] for formula in formula_labels}

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
        if neg_enabled and neg_params["entropy_threshold"]["lower"] < prop_entropy < neg_params["entropy_threshold"]["upper"]:
            _update_negative_cache(
                neg_cache,
                pred,
                [pc_feats, loss, prob_map],
                neg_params["shot_capacity"],
                gpa_cache_stats,
                "test"
            )

        final_logits_original = clip_logits.clone()

        if pos_enabled and entropy_cache:
            final_logits_original += compute_cache_logits(
                pc_feats,
                entropy_cache,
                pos_params["alpha"],
                pos_params["beta"],
                clip_weights
            )

            if gpa_local_cache:
                final_logits_original += compute_local_cache_logits(
                    patch_centers,
                    gpa_local_cache,
                    pos_params["alpha"],
                    pos_params["beta"],
                    clip_weights
                )

        final_logits_by_formula = {ORIGINAL_FORMULA: final_logits_original.clone()}
        final_logits_raw_gpa_global = final_logits_original.clone()

        if pos_enabled and gpa_cache:
            final_logits_raw_gpa_global += compute_cache_logits(
                pc_feats,
                gpa_cache,
                pos_params["alpha"],
                pos_params["beta"],
                clip_weights
            )

            gpa_global_beta = _gpa_global_beta(pos_params["beta"])
            gpa_evidence, gpa_evidence_stats = compute_prior_corrected_gpa_global_evidence(
                pc_feats,
                gpa_cache,
                gpa_global_beta,
                clip_weights,
            )
            gpa_cache_stats["test_gpa_global_evidence_eval"] += 1
            gpa_cache_stats["test_gpa_global_evidence_empty_classes_total"] += int(gpa_evidence_stats["empty_classes"])
            gpa_cache_stats["test_gpa_global_evidence_nonempty_classes_total"] += int(gpa_evidence_stats["nonempty_classes"])

            for gamma in GPA_GLOBAL_GAMMAS:
                final_logits_by_formula[_prior_corrected_formula_label(gamma)] = (
                    final_logits_original + float(gamma) * gpa_evidence
                )

        if neg_enabled and neg_cache:
            neg_logits = compute_cache_logits(
                pc_feats,
                neg_cache,
                neg_params["alpha"],
                neg_params["beta"],
                clip_weights,
                (neg_params["mask_threshold"]["lower"], neg_params["mask_threshold"]["upper"])
            )
            final_logits_original -= neg_logits
            final_logits_raw_gpa_global -= neg_logits
            for formula in list(final_logits_by_formula.keys()):
                final_logits_by_formula[formula] -= neg_logits

        final_logits_by_formula[ORIGINAL_FORMULA] = final_logits_original
        final_logits_by_formula[RAW_GPA_GLOBAL_FORMULA] = final_logits_raw_gpa_global

        for formula in formula_labels:
            logits = final_logits_by_formula.get(formula)
            if logits is None:
                logits = final_logits_original
            accuracies[formula].append(cls_acc(logits, target))

        wandb.log({
            f"Averaged test accuracy/{formula}": sum(values) / len(values)
            for formula, values in accuracies.items()
        }, commit=True)

        if i % args.print_freq == 0:
            print("---- E4-C-A0-c2 original test accuracy: {:.2f}. ----".format(
                sum(accuracies[ORIGINAL_FORMULA]) / len(accuracies[ORIGINAL_FORMULA])
            ))
            print("---- E4-C-A0-c2 raw GPA global test accuracy: {:.2f}. ----".format(
                sum(accuracies[RAW_GPA_GLOBAL_FORMULA]) / len(accuracies[RAW_GPA_GLOBAL_FORMULA])
            ))
            for gamma in GPA_GLOBAL_GAMMAS:
                formula = _prior_corrected_formula_label(gamma)
                print("---- E4-C-A0-c2 prior-corrected gamma={} test accuracy: {:.2f}. ----".format(
                    _gamma_label(gamma),
                    sum(accuracies[formula]) / len(accuracies[formula])
                ))
            print()

    final_acc = {
        formula: float(sum(values) / len(values)) for formula, values in accuracies.items()
    }
    final_acc["original"] = final_acc[ORIGINAL_FORMULA]
    final_acc["raw_gpa_global"] = final_acc[RAW_GPA_GLOBAL_FORMULA]
    print("---- ***Final*** E4-C-A0-c2 original test accuracy: {:.2f}. ----".format(final_acc[ORIGINAL_FORMULA]))
    print("---- ***Final*** E4-C-A0-c2 raw GPA global test accuracy: {:.2f}. ----".format(final_acc[RAW_GPA_GLOBAL_FORMULA]))
    for gamma in GPA_GLOBAL_GAMMAS:
        formula = _prior_corrected_formula_label(gamma)
        print("---- ***Final*** E4-C-A0-c2 prior-corrected gamma={} test accuracy: {:.2f}. ----".format(
            _gamma_label(gamma),
            final_acc[formula],
        ))
    print()

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
