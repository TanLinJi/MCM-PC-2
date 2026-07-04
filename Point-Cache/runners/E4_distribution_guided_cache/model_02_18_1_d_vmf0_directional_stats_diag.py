"""
02_18_1 D-vMF-0: direction-distribution diagnostics on the 02_9_2 carrier.

核心思想：
1. 载体完全沿用 02_9_2：E4-C-A0+E1-textdist-only, text_weight=0.15；
2. 不改变缓存更新规则；
3. 不改变正式最终预测公式；
4. 保留 02_16_1 的逐分支诊断；
5. 额外维护 D-vMF 方向分布统计，只用于诊断，不参与最终得分。
"""

import hashlib
import math
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
GPA_VARIANT_NAME = "02_18_1-D-vMF-0-directional-stats-diagnostics"

B3_DIAG_SAVE_SAMPLES = os.environ.get("E7_B3_DIAG_SAVE_SAMPLES", "1") != "0"
B3_DIAG_SAVE_RAW_LOGITS = os.environ.get("E7_B3_DIAG_SAVE_RAW_LOGITS", "0") == "1"
B3_DIAG_EPS = float(os.environ.get("E7_B3_DIAG_EPS", "1e-6"))
D_VMF_EPS = float(os.environ.get("D_VMF_EPS", "1e-6"))
D_VMF_TEXT_PRIOR_WEIGHT = float(os.environ.get("D_VMF_TEXT_PRIOR_WEIGHT", "1.0"))

BRANCH_DESCRIPTIONS = {
    "zero_shot_text_proto_dot": {
        "zh": "零样本文本原型点积得分",
        "en": "zero-shot text prototype dot-product logits",
    },
    "global_entropy_cache": {
        "zh": "全局熵缓存投票得分",
        "en": "global entropy cache voting logits",
    },
    "gpa_global_cache_diag": {
        "zh": "GPA 全局缓存诊断得分",
        "en": "GPA global cache diagnostic voting logits",
    },
    "gpa_local_cache": {
        "zh": "GPA 控制的局部缓存投票得分",
        "en": "GPA-controlled local cache voting logits",
    },
    "negative_cache_penalty": {
        "zh": "负缓存惩罚得分",
        "en": "negative cache penalty logits",
    },
    "negative_cache_signed": {
        "zh": "负缓存带符号贡献得分",
        "en": "signed negative cache contribution logits",
    },
    "positive_cache_total": {
        "zh": "正缓存总得分",
        "en": "positive cache total logits",
    },
    "cache_total_signed": {
        "zh": "带符号缓存总得分",
        "en": "signed cache total logits",
    },
    "final_logits": {
        "zh": "最终融合得分",
        "en": "final logits",
    },
    "norm_fusion_offline": {
        "zh": "离线归一化融合诊断得分",
        "en": "offline normalized fusion diagnostic logits",
    },
}


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


# ============================================================
# 02_18_1：沿用 02_16_1 的缓存分支诊断辅助函数
# ============================================================

def _target_int(target):
    if torch.is_tensor(target):
        return int(target.detach().cpu().item())
    return int(target)


def _logits_valid(logits):
    if logits is None:
        return False
    x = logits.detach().float()
    if not torch.isfinite(x).all():
        return False
    return float(x.abs().sum().cpu().item()) > B3_DIAG_EPS


def _logits_pred(logits):
    return int(logits.detach().float().topk(1, dim=1)[1].item())


def _top1_margin_from_logits(logits):
    if logits is None or logits.size(1) < 2:
        return 0.0
    top2 = logits.detach().float().topk(2, dim=1).values
    return float((top2[:, 0] - top2[:, 1]).detach().cpu().item())


def _sample_zscore_logits(logits):
    x = logits.detach().float()
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, unbiased=False, keepdim=True)
    return (x - mean) / (std + B3_DIAG_EPS)


def _append_diag_value(diag_values, key, value):
    if diag_values is None:
        return
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if torch.isfinite(torch.tensor(value)):
        diag_values[key].append(value)


def _summarize_values(values):
    values = [float(v) for v in values if torch.isfinite(torch.tensor(float(v)))]
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
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        weight = pos - float(lo)
        return values[lo] * (1.0 - weight) + values[hi] * weight

    return {
        "count": int(n),
        "mean": float(mean),
        "std": float(var ** 0.5),
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


# ============================================================
# 02_18_1 D-vMF-0：方向分布原型诊断辅助函数
# ============================================================

def _make_d_vmf_state(clip_weights, n_prior=D_VMF_TEXT_PRIOR_WEIGHT):
    """Create class-wise directional statistics on the normalized feature sphere."""
    mu_t = clip_weights.detach().float().permute(1, 0).contiguous()
    mu_t = F.normalize(mu_t, dim=1, eps=D_VMF_EPS)
    return {
        "mu_t": mu_t,
        "a_v": torch.zeros_like(mu_t),
        "n_v": torch.zeros(mu_t.size(0), device=mu_t.device, dtype=torch.float32),
        "w2_v": torch.zeros(mu_t.size(0), device=mu_t.device, dtype=torch.float32),
        "seen_feature_hashes": set(),
        "n_prior": float(n_prior),
        "eps": float(D_VMF_EPS),
    }


def _d_vmf_feature_hash(feat):
    """Hash a normalized feature so build/test duplicate updates can be detected."""
    x = feat.detach().float().view(-1).cpu()
    x = torch.round(x * 10000.0).to(torch.int16).numpy().tobytes()
    return hashlib.sha1(x).hexdigest()


def _d_vmf_entropy_weight(loss, num_classes):
    entropy_raw = _loss_value(loss)
    max_entropy = math.log(max(int(num_classes), 2))
    entropy_norm = entropy_raw / max(max_entropy, D_VMF_EPS)
    entropy_norm = max(0.0, min(1.0, float(entropy_norm)))
    weight = max(0.0, 1.0 - entropy_norm)
    return float(entropy_raw), float(entropy_norm), float(weight)


def _d_vmf_all_mu_q(state):
    eps = float(state["eps"])
    n_prior = float(state["n_prior"])
    mu_t = state["mu_t"]
    a_v = state["a_v"]
    n_v = state["n_v"]
    w2_v = state["w2_v"]

    mu = F.normalize(mu_t * n_prior + a_v, dim=1, eps=eps)
    a_norm = torch.norm(a_v, p=2, dim=1)
    r_v = torch.where(n_v > eps, a_norm / n_v.clamp_min(eps), torch.zeros_like(n_v))
    n_eff = torch.where(w2_v > eps, n_v.pow(2) / (w2_v + eps), torch.zeros_like(n_v))
    maturity_base = torch.clamp(n_eff - 1.0, min=0.0)
    maturity = torch.where(
        maturity_base > 0.0,
        maturity_base / (maturity_base + n_prior + eps),
        torch.zeros_like(maturity_base),
    )
    q = maturity * r_v
    cos_mu_text = (mu * mu_t).sum(dim=1)
    stats = {
        "n_v": n_v,
        "w2_v": w2_v,
        "n_eff": n_eff,
        "R_V": r_v,
        "maturity": maturity,
        "q_dir": q,
        "cos_mu_v_mu_t": cos_mu_text,
    }
    return mu, q, stats


def _d_vmf_scores(state, feat, beta):
    mu, q, class_stats = _d_vmf_all_mu_q(state)
    x = F.normalize(feat.detach().float(), dim=1, eps=state["eps"]).to(mu.device)
    cos = x @ mu.t()
    kernel = torch.exp(-float(beta) * (1.0 - cos))
    scores = kernel * q.view(1, -1)
    return scores, class_stats


def _make_d_vmf_sample_diag(state, feat, pred, loss, target, beta):
    scores, class_stats = _d_vmf_scores(state, feat, beta)
    num_classes = scores.size(1)
    pred = int(pred)
    target = int(target)

    entropy_raw, entropy_norm, weight = _d_vmf_entropy_weight(loss, num_classes)
    scores_cpu = scores.detach().float().cpu().view(-1)
    pred_score = float(scores_cpu[pred].item())
    top_values, top_indices = torch.topk(scores_cpu, k=min(2, num_classes))
    top1_score = float(top_values[0].item())
    top1_class = int(top_indices[0].item())
    top2_score = float(top_values[1].item()) if num_classes > 1 else 0.0
    top2_class = int(top_indices[1].item()) if num_classes > 1 else None

    if num_classes > 1:
        mask = torch.ones(num_classes, dtype=torch.bool)
        mask[pred] = False
        best_other = float(scores_cpu[mask].max().item())
    else:
        best_other = 0.0

    pred_stats = {
        key: value[pred].detach().float().cpu().item()
        for key, value in class_stats.items()
    }
    q_nonzero_classes = int((class_stats["q_dir"] > D_VMF_EPS).sum().detach().cpu().item())

    return {
        "H_zs": float(entropy_raw),
        "H_zs_norm": float(entropy_norm),
        "w_t": float(weight),
        "S_dir_pred": float(pred_score),
        "S_dir_top1": float(top1_score),
        "S_dir_top2": float(top2_score),
        "S_dir_margin": float(pred_score - best_other),
        "S_dir_top1_class": int(top1_class),
        "S_dir_top2_class": top2_class,
        "S_dir_top1_correct": int(top1_class == target),
        "q_dir_pred": float(pred_stats["q_dir"]),
        "R_V_pred": float(pred_stats["R_V"]),
        "n_V_pred": float(pred_stats["n_v"]),
        "n_eff_pred": float(pred_stats["n_eff"]),
        "cos_mu_V_pred_mu_T_pred": float(pred_stats["cos_mu_v_mu_t"]),
        "q_dir_nonzero_classes": int(q_nonzero_classes),
        "q_dir_pred_nonzero": int(pred_stats["q_dir"] > D_VMF_EPS),
    }


def _update_d_vmf_state(state, pred, feat, loss, num_classes, stats, phase, reason):
    feature_hash = _d_vmf_feature_hash(feat)
    if feature_hash in state["seen_feature_hashes"]:
        stats[f"{phase}_d_vmf_skip_seen"] += 1
        return False, "seen"

    entropy_raw, entropy_norm, weight = _d_vmf_entropy_weight(loss, num_classes)
    state["seen_feature_hashes"].add(feature_hash)

    if weight <= D_VMF_EPS:
        stats[f"{phase}_d_vmf_skip_zero_weight"] += 1
        return False, "zero_weight"

    c = int(pred)
    x = F.normalize(feat.detach().float(), dim=1, eps=state["eps"]).view(-1).to(state["a_v"].device)
    state["a_v"][c] += float(weight) * x
    state["n_v"][c] += float(weight)
    state["w2_v"][c] += float(weight) ** 2

    stats[f"{phase}_d_vmf_update"] += 1
    stats[f"{phase}_d_vmf_update_{reason}"] += 1
    return True, "updated"


def _summarize_d_vmf_state(state):
    if state is None:
        return {}

    summary = {}
    _, _, class_stats = _d_vmf_all_mu_q(state)
    q_nonzero = int((class_stats["q_dir"] > D_VMF_EPS).sum().detach().cpu().item())
    n_nonzero = int((class_stats["n_v"] > D_VMF_EPS).sum().detach().cpu().item())

    for c in range(state["mu_t"].size(0)):
        entry = {
            key: value[c].detach().float().cpu().item()
            for key, value in class_stats.items()
        }
        summary[str(c)] = {
            "n_V": float(entry["n_v"]),
            "w2_V": float(entry["w2_v"]),
            "n_eff": float(entry["n_eff"]),
            "R_V": float(entry["R_V"]),
            "maturity": float(entry["maturity"]),
            "q_dir": float(entry["q_dir"]),
            "cos_mu_V_mu_T": float(entry["cos_mu_v_mu_t"]),
        }

    return {
        "n_prior": float(state["n_prior"]),
        "num_classes": int(state["mu_t"].size(0)),
        "num_seen_feature_hashes": int(len(state["seen_feature_hashes"])),
        "classes_with_visual_weight": int(n_nonzero),
        "classes_with_nonzero_q_dir": int(q_nonzero),
        "per_class": summary,
    }


def _record_d_vmf_diag_values(stats, diag_values, diag):
    stats["test_d_vmf_sample_total"] += 1
    stats["test_d_vmf_q_pred_nonzero_total"] += int(diag["q_dir_pred_nonzero"])
    stats["test_d_vmf_score_pred_nonzero_total"] += int(abs(float(diag["S_dir_pred"])) > D_VMF_EPS)
    stats["test_d_vmf_top1_total"] += 1
    stats["test_d_vmf_top1_correct"] += int(diag["S_dir_top1_correct"])

    for key in (
        "H_zs",
        "H_zs_norm",
        "w_t",
        "S_dir_pred",
        "S_dir_top1",
        "S_dir_top2",
        "S_dir_margin",
        "q_dir_pred",
        "R_V_pred",
        "n_V_pred",
        "n_eff_pred",
        "cos_mu_V_pred_mu_T_pred",
        "q_dir_nonzero_classes",
    ):
        _append_diag_value(diag_values, f"test_d_vmf_{key}", diag[key])

    suffix = "zs_correct" if diag.get("zs_correct") else "zs_wrong"
    for key in ("S_dir_margin", "q_dir_pred", "R_V_pred", "n_eff_pred", "w_t"):
        _append_diag_value(diag_values, f"test_d_vmf_{suffix}_{key}", diag[key])


def _record_branch_counter(stats, branch, logits, target, valid=True):
    if not valid:
        return None, None
    pred = _logits_pred(logits)
    correct = int(pred == int(target))
    stats[f"test_branch_{branch}_total"] += 1
    stats[f"test_branch_{branch}_correct"] += correct
    return pred, correct


def _finalize_branch_counters(stats):
    for branch in BRANCH_DESCRIPTIONS:
        total = int(stats.get(f"test_branch_{branch}_total", 0))
        correct = int(stats.get(f"test_branch_{branch}_correct", 0))
        if total > 0:
            stats[f"test_branch_{branch}_acc"] = float(correct) / float(total) * 100.0

    total_seen = int(stats.get("test_branch_zero_shot_text_proto_dot_total", 0))
    cache_valid = int(stats.get("test_branch_cache_total_signed_total", 0))
    if total_seen > 0:
        stats["test_branch_cache_total_signed_valid_rate"] = float(cache_valid) / float(total_seen)

    zs_correct_total = int(stats.get("test_zs_correct_total", 0))
    if zs_correct_total > 0:
        stats["test_zs_correct_final_wrong_rate"] = (
            float(stats.get("test_zs_correct_final_wrong", 0)) / float(zs_correct_total)
        )
        stats["test_zs_correct_cache_wrong_rate"] = (
            float(stats.get("test_zs_correct_cache_wrong", 0)) / float(zs_correct_total)
        )

    zs_wrong_total = int(stats.get("test_zs_wrong_total", 0))
    if zs_wrong_total > 0:
        stats["test_zs_wrong_final_correct_rate"] = (
            float(stats.get("test_zs_wrong_final_correct", 0)) / float(zs_wrong_total)
        )
        stats["test_zs_wrong_cache_correct_rate"] = (
            float(stats.get("test_zs_wrong_cache_correct", 0)) / float(zs_wrong_total)
        )

    d_vmf_total = int(stats.get("test_d_vmf_sample_total", 0))
    if d_vmf_total > 0:
        stats["test_d_vmf_q_pred_nonzero_rate"] = (
            float(stats.get("test_d_vmf_q_pred_nonzero_total", 0)) / float(d_vmf_total)
        )
        stats["test_d_vmf_score_pred_nonzero_rate"] = (
            float(stats.get("test_d_vmf_score_pred_nonzero_total", 0)) / float(d_vmf_total)
        )

    d_vmf_top1_total = int(stats.get("test_d_vmf_top1_total", 0))
    if d_vmf_top1_total > 0:
        stats["test_d_vmf_top1_acc"] = (
            float(stats.get("test_d_vmf_top1_correct", 0)) / float(d_vmf_top1_total) * 100.0
        )


def _record_cache_branch_diagnostics(
    stats,
    diag_values,
    diag_samples,
    sample_index,
    target,
    branch_logits,
    d_vmf_diag=None,
):
    target = int(target)
    valid = {
        name: (True if name in {"zero_shot_text_proto_dot", "final_logits", "norm_fusion_offline"} else _logits_valid(logits))
        for name, logits in branch_logits.items()
    }

    preds = {}
    corrects = {}
    for name, logits in branch_logits.items():
        pred, correct = _record_branch_counter(stats, name, logits, target, valid[name])
        preds[name] = pred
        corrects[name] = correct

    pred_zs = preds["zero_shot_text_proto_dot"]
    correct_zs = corrects["zero_shot_text_proto_dot"]
    correct_final = corrects["final_logits"]
    correct_cache = corrects["cache_total_signed"]

    if d_vmf_diag is not None:
        d_vmf_diag["zs_correct"] = int(bool(correct_zs))
        d_vmf_diag["final_correct"] = int(bool(correct_final))
        _record_d_vmf_diag_values(stats, diag_values, d_vmf_diag)

    if correct_zs:
        stats["test_zs_correct_total"] += 1
        stats["test_zs_correct_final_wrong"] += int(not correct_final)
        if correct_cache is not None:
            stats["test_zs_correct_cache_wrong"] += int(not correct_cache)
    else:
        stats["test_zs_wrong_total"] += 1
        stats["test_zs_wrong_final_correct"] += int(correct_final)
        if correct_cache is not None:
            stats["test_zs_wrong_cache_correct"] += int(correct_cache)

    if preds["cache_total_signed"] is not None:
        agree = int(pred_zs == preds["cache_total_signed"])
        stats["test_zs_cache_agree_total"] += agree
        stats["test_zs_cache_disagree_total"] += int(not agree)
        if agree:
            stats["test_zs_cache_agree_final_total"] += 1
            stats["test_zs_cache_agree_final_correct"] += int(correct_final)
        else:
            stats["test_zs_cache_disagree_final_total"] += 1
            stats["test_zs_cache_disagree_final_correct"] += int(correct_final)

    for name, logits in branch_logits.items():
        if valid[name]:
            x = logits.detach().float()
            _append_diag_value(diag_values, f"test_{name}_norm", torch.norm(x, p=2).cpu().item())
            _append_diag_value(diag_values, f"test_{name}_margin", _top1_margin_from_logits(x))

    if diag_samples is not None and B3_DIAG_SAVE_SAMPLES:
        item = {
            "sample_index": int(sample_index),
            "target": int(target),
        }
        for name in branch_logits:
            item[f"valid_{name}"] = int(valid[name])
            item[f"pred_{name}"] = None if preds[name] is None else int(preds[name])
            item[f"correct_{name}"] = None if corrects[name] is None else int(corrects[name])
        if d_vmf_diag is not None:
            item["d_vmf"] = {
                key: (int(value) if isinstance(value, bool) else value)
                for key, value in d_vmf_diag.items()
            }
        if B3_DIAG_SAVE_RAW_LOGITS:
            for name, logits in branch_logits.items():
                item[f"{name}_logits"] = [float(v) for v in logits.detach().float().cpu().view(-1).tolist()]
        diag_samples.append(item)


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
    target=None,
    sample_index=None,
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
    target_value = None if target is None else _target_int(target)
    pseudo_label_correct = None if target_value is None else int(int(pred) == int(target_value))

    def record_event(decision, old_entropy=None, new_score=None, old_score=None):
        if event_records is None:
            return
        event_records.append({
            "phase": phase,
            "sample_index": None if sample_index is None else int(sample_index),
            "class_index": int(pred),
            "target": target_value,
            "pseudo_label_correct": pseudo_label_correct,
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


def _save_b3_diag_0292_stats(
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
    diag_values=None,
    diag_samples=None,
    d_vmf_state=None,
):
    if not _get_gpa_stats_enabled():
        return

    result_root = getattr(args, "baseline_result_root", None)
    exp_id = getattr(args, "baseline_exp_id", None)

    if not result_root or not exp_id:
        return

    out_dir = Path(result_root) / exp_id / "d_vmf0_directional_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    branch_summary = {}
    for name, desc in BRANCH_DESCRIPTIONS.items():
        total = int(stats.get(f"test_branch_{name}_total", 0))
        correct = int(stats.get(f"test_branch_{name}_correct", 0))
        branch_summary[name] = {
            "zh": desc["zh"],
            "en": desc["en"],
            "correct": correct,
            "total": total,
            "acc": None if total == 0 else float(correct) / float(total) * 100.0,
            "format": "not_applicable" if total == 0 else f"{float(correct) / float(total) * 100.0:.2f}%（{correct}/{total}）",
        }

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exp_id": exp_id,
        "cor_type": getattr(args, "cor_type", None),
        "gpa_variant": GPA_VARIANT_NAME,
        "e4_variant": "02_18_1 D-vMF-0 diagnostics on 02_9_2 carrier",
        "b3_diag": {
            "carrier": "02_9_2",
            "changes_prediction": False,
            "save_samples": bool(B3_DIAG_SAVE_SAMPLES),
            "save_raw_logits": bool(B3_DIAG_SAVE_RAW_LOGITS),
            "eps": float(B3_DIAG_EPS),
            "not_applicable": {
                "candidate_pool": "02_9_2 does not use candidate pool.",
                "energy_cache": "02_9_2 does not maintain a separate energy cache.",
                "alignment_core_cache": "02_9_2 uses GPA cache, not E7-A4 alignment core cache.",
            },
        },
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
        "score_normalization_summary": _summarize_score_norm_state(score_norm_state),
        "center_source": CENTER_SOURCE_LABEL,
        "final_acc": acc,
        "stats": dict(stats),
        "branch_descriptions": BRANCH_DESCRIPTIONS,
        "branch_summary": branch_summary,
        "diag_value_summary": _summarize_diag_values(diag_values),
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
        "d_vmf": {
            "enabled": True,
            "changes_prediction": False,
            "changes_cache_update": False,
            "carrier_timing": "02_9_2 legacy prebuild plus test loop",
            "text_prior_weight": float(D_VMF_TEXT_PRIOR_WEIGHT),
            "eps": float(D_VMF_EPS),
            "beta_source": "positive cache beta",
            "support_kernel": "q_dir[c] * exp(-beta * (1 - cosine(x, mu_V[c])))",
            "update_policy": "update once when a sample is accepted by Global Entropy Cache or GPA Cache",
            "entropy_weight": "w_t = clamp(1 - H_zs / ln(C), 0, 1)",
            "state_summary": _summarize_d_vmf_state(d_vmf_state),
        },
    }

    filename = f"{getattr(args, 'cor_type', 'unknown')}_d_vmf0_directional_stats.json"
    with (out_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if event_records is not None:
        event_filename = f"gpa_replacement_events_{getattr(args, 'cor_type', 'unknown')}.jsonl"
        event_path = out_dir / event_filename
        with event_path.open("w", encoding="utf-8") as f:
            for event in event_records:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"[02_18_1 D-vMF-0] Saved GPA replacement events to {event_path}")

    if B3_DIAG_SAVE_SAMPLES and diag_samples:
        sample_path = out_dir / f"{getattr(args, 'cor_type', 'unknown')}_d_vmf0_samples.jsonl"
        with sample_path.open("w", encoding="utf-8") as f:
            for item in diag_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[02_18_1 D-vMF-0] Saved sample diagnostics to {sample_path}")

    print(f"[02_18_1 D-vMF-0] Saved stats to {out_dir / filename}")


@torch.no_grad()
def build_cache_in_advance(
    args,
    test_loader,
    lm3d_model,
    clip_weights,
    shot_capacity,
    include_prob_map=False,
    text_dist=None,
    d_vmf_state=None,
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

    for sample_index, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        global_item = [pc_feats, loss]
        local_item = [patch_centers, loss]

        # E4-C：GPA 使用已被正缓存接受过的历史可信 visual_dist 评分。
        gpa_accepted = _update_gpa_cache(
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
            target=target,
            sample_index=sample_index,
        )

        entropy_accepted = _update_entropy_cache(entropy_cache, pred, global_item, shot_capacity, stats, "build")
        if entropy_accepted:
            _update_visual_distribution(visual_dist, pred, global_item[0], stats, "build", "entropy_accept")

        if d_vmf_state is not None and (gpa_accepted or entropy_accepted):
            reason = "gpa_or_entropy_accept" if gpa_accepted and entropy_accepted else (
                "gpa_accept" if gpa_accepted else "entropy_accept"
            )
            _update_d_vmf_state(
                d_vmf_state,
                pred,
                global_item[0],
                global_item[1],
                clip_logits.size(1),
                stats,
                "build",
                reason,
            )

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
    E4-C test-time adaptation.

    Global logits:
        使用 Global Entropy Cache，即原始 Point-Cache 的正全局缓存。

    Local logits:
        使用 GPA-controlled Local Cache，即只有进入 GPA Cache 的样本贡献局部特征。

    Negative cache:
        保持原始 Point-Cache 逻辑。
    """
    d_vmf_state = _make_d_vmf_state(clip_weights)
    entropy_cache, gpa_cache, gpa_local_cache, visual_dist, score_norm_state, build_stats, gpa_event_records = build_cache_in_advance(
        args,
        test_loader,
        lm3d_model,
        clip_weights,
        pos_cfg["shot_capacity"],
        text_dist=text_dist,
        d_vmf_state=d_vmf_state,
    )

    print("[02_18_1 D-vMF-0] carrier: 02_9_2 / E4-C-A0+E1-textdist-only")
    print("[02_18_1 D-vMF-0] prediction/cache logic: unchanged")
    print("[02_18_1 D-vMF-0] len(entropy_cache):", len(entropy_cache))
    print("[02_18_1 D-vMF-0] len(gpa_cache):", len(gpa_cache))
    print("[02_18_1 D-vMF-0] len(gpa_local_cache):", len(gpa_local_cache))
    print("[02_18_1 D-vMF-0] entropy cache total:", sum(len(v) for v in entropy_cache.values()))
    print("[02_18_1 D-vMF-0] gpa cache total:", sum(len(v) for v in gpa_cache.values()))
    print("[02_18_1 D-vMF-0] gpa local cache total:", sum(len(v) for v in gpa_local_cache.values()))
    print("[02_18_1 D-vMF-0] visual distribution classes:", len(visual_dist))
    print("[02_18_1 D-vMF-0] text distribution classes:", 0 if text_dist is None else len(text_dist))
    print("[02_18_1 D-vMF-0] score norm mode:", SCORE_NORM_MODE)
    print("[02_18_1 D-vMF-0] score norm state:", _summarize_score_norm_state(score_norm_state))
    print("[02_18_1 D-vMF-0] direction state:", _summarize_d_vmf_state(d_vmf_state))
    print("[02_18_1 D-vMF-0] save sample diagnostics:", B3_DIAG_SAVE_SAMPLES)
    print("[02_18_1 D-vMF-0] save raw logits:", B3_DIAG_SAVE_RAW_LOGITS)

    neg_cache = {}
    gpa_cache_stats = defaultdict(int)
    for k, v in build_stats.items():
        gpa_cache_stats[k] += v

    # E4-C 保留 build 阶段形成的 EntropyCache、GPA-Cache、GPA-local-cache 和
    # accepted-history visual_dist。visual_dist 只累计曾被正缓存接受过的样本。
    accuracies = []
    diag_values = defaultdict(list)
    diag_samples = []

    pos_enabled, neg_enabled = pos_cfg["enabled"], neg_cfg["enabled"]

    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ["shot_capacity", "alpha", "beta"]}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ["shot_capacity", "alpha", "beta", "entropy_threshold", "mask_threshold"]}

    for i, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()

        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)

        target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)
        d_vmf_diag = _make_d_vmf_sample_diag(
            d_vmf_state,
            pc_feats,
            pred,
            loss,
            _target_int(target),
            pos_cfg["beta"],
        )
        d_vmf_diag["positive_cache_accepted"] = 0
        d_vmf_diag["direction_stats_updated"] = 0
        d_vmf_diag["direction_stats_update_status"] = "not_attempted"
        d_vmf_diag["direction_stats_update_reason"] = None

        gpa_accepted = False
        entropy_accepted = False
        if pos_enabled:
            global_item = [pc_feats, loss]
            local_item = [patch_centers, loss]

            gpa_accepted = _update_gpa_cache(
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
                target=target,
                sample_index=i,
            )

            entropy_accepted = _update_entropy_cache(
                entropy_cache, pred, global_item, pos_params["shot_capacity"], gpa_cache_stats, "test"
            )
            if entropy_accepted:
                _update_visual_distribution(visual_dist, pred, global_item[0], gpa_cache_stats, "test", "entropy_accept")
            if gpa_accepted or entropy_accepted:
                update_reason = "gpa_or_entropy_accept" if gpa_accepted and entropy_accepted else (
                    "gpa_accept" if gpa_accepted else "entropy_accept"
                )
                updated, update_status = _update_d_vmf_state(
                    d_vmf_state,
                    pred,
                    global_item[0],
                    global_item[1],
                    clip_logits.size(1),
                    gpa_cache_stats,
                    "test",
                    update_reason,
                )
                d_vmf_diag["positive_cache_accepted"] = 1
                d_vmf_diag["direction_stats_updated"] = int(updated)
                d_vmf_diag["direction_stats_update_status"] = update_status
                d_vmf_diag["direction_stats_update_reason"] = update_reason
        if neg_enabled and neg_params["entropy_threshold"]["lower"] < prop_entropy < neg_params["entropy_threshold"]["upper"]:
            _update_negative_cache(
                neg_cache,
                pred,
                [pc_feats, loss, prob_map],
                neg_params["shot_capacity"],
                gpa_cache_stats,
                "test"
            )

        zs_logits = clip_logits.clone()
        entropy_logits = torch.zeros_like(clip_logits)
        gpa_global_logits = torch.zeros_like(clip_logits)
        local_logits = torch.zeros_like(clip_logits)
        negative_logits = torch.zeros_like(clip_logits)

        final_logits = zs_logits.clone()

        if pos_enabled and entropy_cache:
            entropy_logits = compute_cache_logits(
                pc_feats,
                entropy_cache,
                pos_params["alpha"],
                pos_params["beta"],
                clip_weights
            )
            final_logits += entropy_logits

            if gpa_cache:
                gpa_global_logits = compute_cache_logits(
                    pc_feats,
                    gpa_cache,
                    pos_params["alpha"],
                    pos_params["beta"],
                    clip_weights
                )

            if gpa_local_cache:
                local_logits = compute_local_cache_logits(
                    patch_centers,
                    gpa_local_cache,
                    pos_params["alpha"],
                    pos_params["beta"],
                    clip_weights
                )
                final_logits += local_logits

        if neg_enabled and neg_cache:
            negative_logits = compute_cache_logits(
                pc_feats,
                neg_cache,
                neg_params["alpha"],
                neg_params["beta"],
                clip_weights,
                (neg_params["mask_threshold"]["lower"], neg_params["mask_threshold"]["upper"])
            )
            final_logits -= negative_logits

        positive_cache_logits = entropy_logits + local_logits
        signed_negative_logits = -negative_logits
        signed_cache_logits = positive_cache_logits + signed_negative_logits
        valid_signed_cache = _logits_valid(signed_cache_logits)
        norm_cache_logits = _sample_zscore_logits(signed_cache_logits) if valid_signed_cache else torch.zeros_like(zs_logits).float()
        norm_fusion_logits = _sample_zscore_logits(zs_logits) + norm_cache_logits

        branch_logits = {
            "zero_shot_text_proto_dot": zs_logits,
            "global_entropy_cache": entropy_logits,
            "gpa_global_cache_diag": gpa_global_logits,
            "gpa_local_cache": local_logits,
            "negative_cache_penalty": negative_logits,
            "negative_cache_signed": signed_negative_logits,
            "positive_cache_total": positive_cache_logits,
            "cache_total_signed": signed_cache_logits,
            "final_logits": final_logits,
            "norm_fusion_offline": norm_fusion_logits,
        }
        _record_cache_branch_diagnostics(
            gpa_cache_stats,
            diag_values,
            diag_samples,
            i,
            _target_int(target),
            branch_logits,
            d_vmf_diag,
        )

        acc = cls_acc(final_logits, target)
        accuracies.append(acc)
        wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

        if i % args.print_freq == 0:
            print("---- 02_18_1 D-vMF-0 test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    final_acc = sum(accuracies) / len(accuracies)
    print("---- ***Final*** 02_18_1 D-vMF-0 test accuracy: {:.2f}. ----\n".format(final_acc))
    _finalize_branch_counters(gpa_cache_stats)

    _save_b3_diag_0292_stats(
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
        diag_values,
        diag_samples,
        d_vmf_state,
    )

    return final_acc
