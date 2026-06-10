"""
E5-A0/A1: ADAPT-inspired StatsBank + shared-covariance GDA diagnostics.

This file intentionally keeps the E4-C-A0+E1-textdist-only prediction path
unchanged and adds an independent E5 diagnostic branch:
1. Point-Cache/E4-C caches and final logits remain the original baseline path.
2. A separate online StatsBank stores high-confidence manual_full predictions.
3. The current sample is evaluated by GDA before it can update StatsBank.
4. Shared-covariance GDA is recorded as standalone diagnostics only.
"""

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
GPA_VARIANT_NAME = "E5-baseline-E4-C-accepted-history-text-visual-distribution-guided-gpa-cache"
E5_VARIANT_NAME = "E5-A0-A1-adapt-inspired-statsbank-shared-covariance-gda-diagnostics"


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


# ============================================================
# E5-A0/A1：independent StatsBank + shared-covariance GDA
# ============================================================

E5_STATSBANK_CAPACITY = int(os.environ.get("E5_STATSBANK_CAPACITY", "16"))
E5_GDA_ALPHA = float(os.environ.get("E5_GDA_ALPHA", "0.9"))
E5_GDA_MIN_TOTAL = int(os.environ.get("E5_GDA_MIN_TOTAL", "8"))
E5_GDA_MIN_CLASSES = int(os.environ.get("E5_GDA_MIN_CLASSES", "2"))
E5_GDA_NORM_EPS = float(os.environ.get("E5_GDA_NORM_EPS", "1e-6"))
E5_GDA_NORM_CLIP = float(os.environ.get("E5_GDA_NORM_CLIP", "3.0"))
E5_SAVE_SAMPLE_DIAGNOSTICS = os.environ.get("E5_SAVE_SAMPLE_DIAGNOSTICS", "1") != "0"


def _parse_float_list(raw, default):
    if raw is None or not str(raw).strip():
        return tuple(default)

    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return tuple(values) if values else tuple(default)


E5_GDA_OVERRIDE_THRESHOLDS = _parse_float_list(
    os.environ.get("E5_GDA_OVERRIDE_THRESHOLDS"),
    (0.50, 0.75, 1.00, 1.25, 1.50, 2.00),
)


class _E5StatsBank:
    """Per-class bounded bank for GDA statistics, independent from Point-Cache."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.bank = defaultdict(list)
        self.version = 0

    def add(self, pred, feat, confidence, entropy, sample_index):
        pred = int(pred)
        confidence = float(confidence)
        entropy = float(entropy)
        x = feat.detach().float().view(-1).clone()

        item = {
            "feat": x,
            "confidence": confidence,
            "entropy": entropy,
            "sample_index": int(sample_index),
        }

        bucket = self.bank[pred]
        if len(bucket) < self.capacity:
            bucket.append(item)
            bucket.sort(key=lambda v: v["confidence"], reverse=True)
            self.version += 1
            return True, "add"

        worst_idx = min(range(len(bucket)), key=lambda idx: bucket[idx]["confidence"])
        if confidence <= float(bucket[worst_idx]["confidence"]):
            return False, "reject_low_confidence"

        bucket[worst_idx] = item
        bucket.sort(key=lambda v: v["confidence"], reverse=True)
        self.version += 1
        return True, "replace"

    def total(self):
        return int(sum(len(items) for items in self.bank.values()))

    def class_count(self):
        return int(sum(1 for items in self.bank.values() if items))

    def counts(self):
        return {str(k): len(v) for k, v in sorted(self.bank.items(), key=lambda kv: kv[0])}

    def samples(self):
        feats = []
        labels = []
        confidences = []

        for class_index, items in sorted(self.bank.items(), key=lambda kv: kv[0]):
            for item in items:
                feats.append(item["feat"])
                labels.append(int(class_index))
                confidences.append(float(item["confidence"]))

        if not feats:
            return None, None, None

        return (
            torch.stack(feats, dim=0),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(confidences, dtype=torch.float32),
        )

    def summary(self):
        conf_values = []
        entropy_values = []
        for items in self.bank.values():
            for item in items:
                conf_values.append(float(item["confidence"]))
                entropy_values.append(float(item["entropy"]))

        return {
            "capacity_per_class": int(self.capacity),
            "total": self.total(),
            "num_classes": self.class_count(),
            "class_counts": self.counts(),
            "confidence_min": None if not conf_values else float(min(conf_values)),
            "confidence_max": None if not conf_values else float(max(conf_values)),
            "confidence_mean": None if not conf_values else float(sum(conf_values) / len(conf_values)),
            "entropy_min": None if not entropy_values else float(min(entropy_values)),
            "entropy_max": None if not entropy_values else float(max(entropy_values)),
            "entropy_mean": None if not entropy_values else float(sum(entropy_values) / len(entropy_values)),
            "version": int(self.version),
        }


def _make_e5_diag_state():
    return {
        "counts": defaultdict(int),
        "scalars": {},
        "samples": [],
        "model_builds": [],
        "last_model_summary": None,
        "gated_overrides": {
            f"{threshold:g}": defaultdict(int)
            for threshold in E5_GDA_OVERRIDE_THRESHOLDS
        },
    }


def _update_e5_scalar(state, name, value):
    if value is None:
        return

    value = float(value)
    entry = state["scalars"].setdefault(
        name,
        {"count": 0, "mean": 0.0, "m2": 0.0, "min": None, "max": None},
    )

    count_old = int(entry["count"])
    count_new = count_old + 1
    if count_old == 0:
        entry["count"] = 1
        entry["mean"] = value
        entry["m2"] = 0.0
        entry["min"] = value
        entry["max"] = value
        return

    delta = value - float(entry["mean"])
    mean_new = float(entry["mean"]) + delta / float(count_new)
    delta2 = value - mean_new

    entry["count"] = count_new
    entry["mean"] = mean_new
    entry["m2"] = float(entry["m2"]) + delta * delta2
    entry["min"] = min(float(entry["min"]), value)
    entry["max"] = max(float(entry["max"]), value)


def _summarize_e5_scalars(state):
    summary = {}
    for name, entry in sorted(state["scalars"].items()):
        count = int(entry["count"])
        std = None
        if count > 1:
            std = (float(entry["m2"]) / float(count - 1)) ** 0.5
        summary[name] = {
            "count": count,
            "mean": float(entry["mean"]),
            "std": None if std is None else float(std),
            "min": None if entry["min"] is None else float(entry["min"]),
            "max": None if entry["max"] is None else float(entry["max"]),
        }
    return summary


def _record_e5_gated_overrides(state, target, original_pred, gda_result):
    target = int(target)
    original_pred = int(original_pred)
    original_correct = bool(original_pred == target)

    for threshold in E5_GDA_OVERRIDE_THRESHOLDS:
        key = f"{threshold:g}"
        entry = state["gated_overrides"][key]
        entry["total"] += 1

        use_gda = False
        simulated_pred = original_pred

        if gda_result.get("available", False):
            margin = gda_result.get("norm_one_vs_rest_margin")
            gda_pred = int(gda_result["raw_pred"])
            if margin is not None and float(margin) >= float(threshold) and gda_pred != original_pred:
                use_gda = True
                simulated_pred = gda_pred
                entry["overrides"] += 1

        simulated_correct = bool(simulated_pred == target)
        if simulated_correct:
            entry["correct"] += 1

        if use_gda:
            if not original_correct and simulated_correct:
                entry["fixes"] += 1
            elif original_correct and not simulated_correct:
                entry["breaks"] += 1
            elif not original_correct and not simulated_correct:
                entry["wrong_to_wrong_overrides"] += 1
            else:
                entry["correct_to_correct_overrides"] += 1


def _summarize_e5_gated_overrides(state):
    summary = {}
    for threshold in E5_GDA_OVERRIDE_THRESHOLDS:
        key = f"{threshold:g}"
        entry = dict(state["gated_overrides"].get(key, {}))
        total = int(entry.get("total", 0))
        overrides = int(entry.get("overrides", 0))
        correct = int(entry.get("correct", 0))
        fixes = int(entry.get("fixes", 0))
        breaks = int(entry.get("breaks", 0))
        summary[key] = {
            "threshold": float(threshold),
            "total": total,
            "correct": correct,
            "acc": None if total == 0 else 100.0 * correct / total,
            "overrides": overrides,
            "override_rate": None if total == 0 else 100.0 * overrides / total,
            "fixes": fixes,
            "breaks": breaks,
            "net_fixes": fixes - breaks,
            "wrong_to_wrong_overrides": int(entry.get("wrong_to_wrong_overrides", 0)),
            "correct_to_correct_overrides": int(entry.get("correct_to_correct_overrides", 0)),
        }
    return summary


def _e5_text_prior_means(clip_weights):
    means = clip_weights.detach().float().permute(1, 0).contiguous()
    return means / means.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _e5_build_gda_model(stats_bank, clip_weights, device):
    total = stats_bank.total()
    num_classes_with_samples = stats_bank.class_count()

    if total < E5_GDA_MIN_TOTAL:
        return None, {
            "available": False,
            "skip_reason": "insufficient_total",
            "total": int(total),
            "num_classes": int(num_classes_with_samples),
        }
    if num_classes_with_samples < E5_GDA_MIN_CLASSES:
        return None, {
            "available": False,
            "skip_reason": "insufficient_classes",
            "total": int(total),
            "num_classes": int(num_classes_with_samples),
        }

    feats, labels, confidences = stats_bank.samples()
    if feats is None:
        return None, {
            "available": False,
            "skip_reason": "empty_bank",
            "total": 0,
            "num_classes": 0,
        }

    feats = feats.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device)
    confidences = confidences.to(device=device, dtype=torch.float32).clamp_min(1e-6)

    text_prior = _e5_text_prior_means(clip_weights).to(device=device, dtype=torch.float32)
    class_means = text_prior.clone()
    visual_counts = torch.zeros(text_prior.size(0), device=device, dtype=torch.long)

    for class_index in labels.unique(sorted=True):
        mask = labels == class_index
        class_feats = feats[mask]
        class_weights = confidences[mask].view(-1, 1)
        visual_mean = (class_feats * class_weights).sum(dim=0) / class_weights.sum().clamp_min(1e-12)
        class_means[int(class_index)] = E5_GDA_ALPHA * visual_mean + (1.0 - E5_GDA_ALPHA) * text_prior[int(class_index)]
        visual_counts[int(class_index)] = int(mask.sum().item())

    sample_means = class_means[labels]
    centered = feats - sample_means
    n_total = int(feats.size(0))
    d = int(feats.size(1))
    cov = centered.t().matmul(centered) / float(max(n_total - 1, 1))
    trace_cov = float(torch.trace(cov).detach().cpu().item())

    if trace_cov <= E5_GDA_NORM_EPS:
        return None, {
            "available": False,
            "skip_reason": "degenerate_covariance",
            "total": int(total),
            "num_classes": int(num_classes_with_samples),
            "trace_cov": trace_cov,
        }

    eye = torch.eye(d, device=device, dtype=torch.float32)
    shrinkage_matrix = float(max(n_total - 1, 1)) * cov + trace_cov * eye

    inverse_method = "solve"
    try:
        inv_cov = float(d) * torch.linalg.solve(shrinkage_matrix, eye)
    except RuntimeError:
        inverse_method = "pinv"
        inv_cov = float(d) * torch.linalg.pinv(shrinkage_matrix)

    inv_cov = 0.5 * (inv_cov + inv_cov.t())

    diag = torch.diagonal(shrinkage_matrix).detach()
    summary = {
        "available": True,
        "total": int(total),
        "num_classes": int(num_classes_with_samples),
        "feature_dim": int(d),
        "alpha": float(E5_GDA_ALPHA),
        "trace_cov": trace_cov,
        "shrinkage_diag_min": float(diag.min().cpu().item()),
        "shrinkage_diag_max": float(diag.max().cpu().item()),
        "shrinkage_diag_mean": float(diag.mean().cpu().item()),
        "inverse_fro_norm": float(torch.linalg.norm(inv_cov).detach().cpu().item()),
        "inverse_method": inverse_method,
        "visual_class_counts": {str(i): int(v) for i, v in enumerate(visual_counts.detach().cpu().tolist()) if int(v) > 0},
    }

    model = {
        "means": class_means,
        "inv_cov": inv_cov,
        "summary": summary,
    }
    return model, summary


def _e5_samplewise_zscore(scores):
    mean = scores.mean()
    std = scores.std(unbiased=False)
    norm = (scores - mean) / std.clamp_min(E5_GDA_NORM_EPS)
    if E5_GDA_NORM_CLIP > 0:
        norm = norm.clamp(min=-E5_GDA_NORM_CLIP, max=E5_GDA_NORM_CLIP)
    return norm


def _e5_one_vs_rest_margin(scores, class_index):
    class_index = int(class_index)
    if scores.numel() <= 1:
        return None
    mask = torch.ones(scores.numel(), device=scores.device, dtype=torch.bool)
    mask[class_index] = False
    rest_log_mean_exp = torch.logsumexp(scores[mask], dim=0) - math.log(int(mask.sum().item()))
    return float((scores[class_index] - rest_log_mean_exp).detach().cpu().item())


def _e5_compute_gda_result(gda_model, feat, target):
    if gda_model is None:
        return {"available": False}

    x = feat.detach().float().view(-1).to(device=gda_model["means"].device)
    means = gda_model["means"]
    inv_cov = gda_model["inv_cov"]

    means_inv = means.matmul(inv_cov)
    linear = means_inv.matmul(x)
    quadratic = (means_inv * means).sum(dim=1)
    raw_scores = linear - 0.5 * quadratic
    norm_scores = _e5_samplewise_zscore(raw_scores)

    raw_top2 = torch.topk(raw_scores, k=min(2, raw_scores.numel()), dim=0)
    norm_top2 = torch.topk(norm_scores, k=min(2, norm_scores.numel()), dim=0)

    raw_pred = int(raw_top2.indices[0].detach().cpu().item())
    norm_pred = int(norm_top2.indices[0].detach().cpu().item())
    target = int(target)

    raw_margin = None
    norm_margin = None
    if raw_top2.values.numel() > 1:
        raw_margin = float((raw_top2.values[0] - raw_top2.values[1]).detach().cpu().item())
    if norm_top2.values.numel() > 1:
        norm_margin = float((norm_top2.values[0] - norm_top2.values[1]).detach().cpu().item())

    return {
        "available": True,
        "raw_pred": raw_pred,
        "norm_pred": norm_pred,
        "raw_correct": bool(raw_pred == target),
        "norm_correct": bool(norm_pred == target),
        "raw_top1_score": float(raw_top2.values[0].detach().cpu().item()),
        "norm_top1_score": float(norm_top2.values[0].detach().cpu().item()),
        "raw_top1_top2_margin": raw_margin,
        "norm_top1_top2_margin": norm_margin,
        "raw_one_vs_rest_margin": _e5_one_vs_rest_margin(raw_scores, raw_pred),
        "norm_one_vs_rest_margin": _e5_one_vs_rest_margin(norm_scores, norm_pred),
        "target_raw_score": float(raw_scores[target].detach().cpu().item()),
        "target_norm_score": float(norm_scores[target].detach().cpu().item()),
        "raw_score_mean": float(raw_scores.mean().detach().cpu().item()),
        "raw_score_std": float(raw_scores.std(unbiased=False).detach().cpu().item()),
        "norm_score_mean": float(norm_scores.mean().detach().cpu().item()),
        "norm_score_std": float(norm_scores.std(unbiased=False).detach().cpu().item()),
    }


def _e5_record_diagnostic(state, sample_index, target, original_pred, gda_result, model_summary):
    counts = state["counts"]
    counts["samples_total"] += 1

    target = int(target)
    original_pred = int(original_pred)
    if original_pred == target:
        counts["original_correct"] += 1

    if model_summary is not None:
        state["last_model_summary"] = model_summary

    _record_e5_gated_overrides(state, target, original_pred, gda_result)

    if not gda_result.get("available", False):
        counts["gda_skipped"] += 1
        return

    counts["gda_available"] += 1
    if gda_result["raw_correct"]:
        counts["gda_raw_correct"] += 1
    if gda_result["norm_correct"]:
        counts["gda_norm_correct"] += 1
    if int(gda_result["raw_pred"]) == original_pred:
        counts["gda_raw_agree_with_original"] += 1
    if int(gda_result["norm_pred"]) == original_pred:
        counts["gda_norm_agree_with_original"] += 1

    for name in (
        "raw_top1_score",
        "norm_top1_score",
        "raw_top1_top2_margin",
        "norm_top1_top2_margin",
        "raw_one_vs_rest_margin",
        "norm_one_vs_rest_margin",
        "target_raw_score",
        "target_norm_score",
        "raw_score_mean",
        "raw_score_std",
        "norm_score_mean",
        "norm_score_std",
    ):
        _update_e5_scalar(state, name, gda_result.get(name))

    if E5_SAVE_SAMPLE_DIAGNOSTICS:
        state["samples"].append({
            "sample_index": int(sample_index),
            "target": target,
            "original_pred": original_pred,
            "gda_raw_pred": int(gda_result["raw_pred"]),
            "gda_norm_pred": int(gda_result["norm_pred"]),
            "original_correct": bool(original_pred == target),
            "gda_raw_correct": bool(gda_result["raw_correct"]),
            "gda_norm_correct": bool(gda_result["norm_correct"]),
            "gda_raw_agree_with_original": bool(int(gda_result["raw_pred"]) == original_pred),
            "gda_norm_agree_with_original": bool(int(gda_result["norm_pred"]) == original_pred),
            "raw_top1_top2_margin": gda_result.get("raw_top1_top2_margin"),
            "norm_top1_top2_margin": gda_result.get("norm_top1_top2_margin"),
            "raw_one_vs_rest_margin": gda_result.get("raw_one_vs_rest_margin"),
            "norm_one_vs_rest_margin": gda_result.get("norm_one_vs_rest_margin"),
        })


def _save_e5_diagnostics(args, state, stats_bank, final_acc):
    result_root = getattr(args, "baseline_result_root", None)
    exp_id = getattr(args, "baseline_exp_id", None)
    if not result_root or not exp_id:
        return

    out_dir = Path(result_root) / exp_id / "e5_gda_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = dict(state["counts"])
    total = int(counts.get("samples_total", 0))
    available = int(counts.get("gda_available", 0))

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exp_id": exp_id,
        "cor_type": getattr(args, "cor_type", None),
        "e5_variant": E5_VARIANT_NAME,
        "baseline_reference": "E4-C-A0+E1-textdist-only 02_9_2",
        "protocol_note": (
            "Point-Cache/E4-C baseline keeps its original warmup/update order; "
            "only the independent E5 StatsBank/GDA branch uses delayed update."
        ),
        "final_original_pointcache_acc": None if final_acc is None else float(final_acc),
        "statsbank_capacity": int(E5_STATSBANK_CAPACITY),
        "gda_alpha": float(E5_GDA_ALPHA),
        "gda_min_total": int(E5_GDA_MIN_TOTAL),
        "gda_min_classes": int(E5_GDA_MIN_CLASSES),
        "gda_norm": {
            "mode": "sample_wise_class_zscore",
            "eps": float(E5_GDA_NORM_EPS),
            "clip": float(E5_GDA_NORM_CLIP),
        },
        "counts": counts,
        "derived_metrics": {
            "original_acc_from_counts": None if total == 0 else 100.0 * counts.get("original_correct", 0) / total,
            "gda_raw_acc_available_only": None if available == 0 else 100.0 * counts.get("gda_raw_correct", 0) / available,
            "gda_norm_acc_available_only": None if available == 0 else 100.0 * counts.get("gda_norm_correct", 0) / available,
            "gda_raw_agreement_available_only": None if available == 0 else 100.0 * counts.get("gda_raw_agree_with_original", 0) / available,
            "gda_norm_agreement_available_only": None if available == 0 else 100.0 * counts.get("gda_norm_agree_with_original", 0) / available,
            "gda_available_ratio": None if total == 0 else 100.0 * available / total,
        },
        "scalar_summary": _summarize_e5_scalars(state),
        "gated_override_diagnostics": _summarize_e5_gated_overrides(state),
        "statsbank_summary": stats_bank.summary(),
        "last_gda_model_summary": state["last_model_summary"],
        "model_builds": state["model_builds"],
    }

    filename = f"{getattr(args, 'cor_type', 'unknown')}_e5_gda_stats.json"
    with (out_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if E5_SAVE_SAMPLE_DIAGNOSTICS:
        sample_filename = f"gda_sample_diagnostics_{getattr(args, 'cor_type', 'unknown')}.jsonl"
        sample_path = out_dir / sample_filename
        with sample_path.open("w", encoding="utf-8") as f:
            for record in state["samples"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[E5-A0/A1] Saved sample GDA diagnostics to {sample_path}")

    print(f"[E5-A0/A1] Saved GDA diagnostics to {out_dir / filename}")


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


@torch.no_grad()
def run_test_tda(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights, text_dist=None):
    """
    E5-A0/A1 test-time adaptation with GDA diagnostics.

    Baseline global logits:
        使用 Global Entropy Cache，即原始 Point-Cache 的正全局缓存。

    Baseline local logits:
        使用 GPA-controlled Local Cache，即只有进入 GPA Cache 的样本贡献局部特征。

    Baseline negative cache:
        保持原始 Point-Cache 逻辑。

    E5 diagnostics:
        使用独立 StatsBank，当前样本先诊断、后进入 StatsBank。
        GDA 不参与 final logits，本函数返回的仍是原始 Point-Cache/E4-C 准确率。
    """
    entropy_cache, gpa_cache, gpa_local_cache, visual_dist, score_norm_state, build_stats, gpa_event_records = build_cache_in_advance(
        args, test_loader, lm3d_model, clip_weights, pos_cfg["shot_capacity"], text_dist=text_dist
    )

    print("[E5-A0/A1] baseline len(entropy_cache):", len(entropy_cache))
    print("[E5-A0/A1] baseline len(gpa_cache):", len(gpa_cache))
    print("[E5-A0/A1] baseline len(gpa_local_cache):", len(gpa_local_cache))
    print("[E5-A0/A1] baseline entropy cache total:", sum(len(v) for v in entropy_cache.values()))
    print("[E5-A0/A1] baseline gpa cache total:", sum(len(v) for v in gpa_cache.values()))
    print("[E5-A0/A1] baseline gpa local cache total:", sum(len(v) for v in gpa_local_cache.values()))
    print("[E5-A0/A1] baseline visual distribution classes:", len(visual_dist))
    print("[E5-A0/A1] text distribution classes:", 0 if text_dist is None else len(text_dist))
    print("[E5-A0/A1] E4 score norm mode:", SCORE_NORM_MODE)
    print("[E5-A0/A1] E4 score norm state:", _summarize_score_norm_state(score_norm_state))
    print("[E5-A0/A1] StatsBank capacity:", E5_STATSBANK_CAPACITY)
    print("[E5-A0/A1] GDA alpha:", E5_GDA_ALPHA)
    print("[E5-A0/A1] GDA min total/classes:", E5_GDA_MIN_TOTAL, E5_GDA_MIN_CLASSES)
    print("[E5-A0/A1] GDA normalization: sample-wise class z-score, clip=", E5_GDA_NORM_CLIP)

    neg_cache = {}
    gpa_cache_stats = defaultdict(int)
    for k, v in build_stats.items():
        gpa_cache_stats[k] += v

    # E4-C 保留 build 阶段形成的 EntropyCache、GPA-Cache、GPA-local-cache 和
    # accepted-history visual_dist。visual_dist 只累计曾被正缓存接受过的样本。
    accuracies = []
    e5_state = _make_e5_diag_state()
    stats_bank = _E5StatsBank(E5_STATSBANK_CAPACITY)
    cached_gda_model = None
    cached_gda_summary = None
    cached_bank_version = -1

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

        original_pred = int(final_logits.topk(1, dim=1, largest=True, sorted=True)[1].t()[0])

        if cached_bank_version != stats_bank.version:
            cached_gda_model, cached_gda_summary = _e5_build_gda_model(stats_bank, clip_weights, pc_feats.device)
            cached_bank_version = stats_bank.version
            if cached_gda_summary is not None:
                if len(e5_state["model_builds"]) < 20 or i % max(int(args.print_freq), 1) == 0:
                    sampled_summary = dict(cached_gda_summary)
                    sampled_summary["sample_index"] = int(i)
                    sampled_summary["statsbank_version"] = int(stats_bank.version)
                    e5_state["model_builds"].append(sampled_summary)

        gda_result = _e5_compute_gda_result(cached_gda_model, pc_feats, int(target.detach().cpu().item()))
        _e5_record_diagnostic(
            e5_state,
            i,
            int(target.detach().cpu().item()),
            original_pred,
            gda_result,
            cached_gda_summary,
        )

        statsbank_confidence = max(0.0, 1.0 - float(prop_entropy))
        accepted, statsbank_decision = stats_bank.add(
            pred,
            pc_feats,
            statsbank_confidence,
            prop_entropy,
            i,
        )
        e5_state["counts"][f"statsbank_{statsbank_decision}"] += 1
        if accepted:
            e5_state["counts"]["statsbank_accept"] += 1

        wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

        if i % args.print_freq == 0:
            print("---- E5-A0/A1 original Point-Cache test accuracy: {:.2f}. ----".format(sum(accuracies) / len(accuracies)))
            print(
                "---- E5-A0/A1 GDA available/skipped: {}/{}; StatsBank total/classes: {}/{}. ----\n".format(
                    e5_state["counts"].get("gda_available", 0),
                    e5_state["counts"].get("gda_skipped", 0),
                    stats_bank.total(),
                    stats_bank.class_count(),
                )
            )

    final_acc = sum(accuracies) / len(accuracies)
    print("---- ***Final*** E5-A0/A1 original Point-Cache test accuracy: {:.2f}. ----".format(final_acc))
    available = int(e5_state["counts"].get("gda_available", 0))
    if available > 0:
        gda_raw_acc = 100.0 * e5_state["counts"].get("gda_raw_correct", 0) / available
        gda_norm_acc = 100.0 * e5_state["counts"].get("gda_norm_correct", 0) / available
        gda_agree = 100.0 * e5_state["counts"].get("gda_norm_agree_with_original", 0) / available
        print("---- ***Final*** E5-A1 standalone raw GDA accuracy on available samples: {:.2f}. ----".format(gda_raw_acc))
        print("---- ***Final*** E5-A1 standalone normalized GDA accuracy on available samples: {:.2f}. ----".format(gda_norm_acc))
        print("---- ***Final*** E5-A1 normalized GDA agreement with original: {:.2f}. ----\n".format(gda_agree))
    else:
        print("---- ***Final*** E5-A1 standalone GDA unavailable for all samples. ----\n")

    gated_summary = _summarize_e5_gated_overrides(e5_state)
    best_gated = None
    for item in gated_summary.values():
        if item["acc"] is None:
            continue
        if best_gated is None or item["acc"] > best_gated["acc"]:
            best_gated = item
    if best_gated is not None:
        print(
            "---- ***Final*** E5-A1 best gated override diagnostic: "
            "threshold={:.2f}, acc={:.2f}, overrides={}, net_fixes={}. ----\n".format(
                best_gated["threshold"],
                best_gated["acc"],
                best_gated["overrides"],
                best_gated["net_fixes"],
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
    )
    _save_e5_diagnostics(args, e5_state, stats_bank, final_acc)

    return final_acc
