"""
E4: Distribution-Guided GPA Cache for Point-Cache.

当前实现是 E4-A-distribution-guided-gpa-cache。

核心思想：
1. 保留原始 Point-Cache 的 Global Entropy Cache，仍用于 global cache logits；
2. 新增 Global Prototype-Alignment Cache，简称 GPA Cache；
3. 对进入 GPA Cache 的视觉全局特征维护每类在线对角分布；
4. GPA Cache 未满时沿用 E3-V2-C 的直接加入规则，并更新类别分布；
5. GPA Cache 满后使用“低熵 + 更符合类别分布”的规则替换最高熵样本；
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

CENTER_SOURCE_LABEL = "Class-wise distribution score"
GPA_VARIANT_NAME = "E4-A-distribution-guided-gpa-cache"


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
# E4-A：类别概率分布辅助函数
# ============================================================

DIST_EPS = float(os.environ.get("E4_DIST_EPS", "1e-4"))
DIST_MIN_VAR = float(os.environ.get("E4_DIST_MIN_VAR", "1e-4"))


def _feature_float(feat):
    return feat.detach().float()


def _update_class_distribution(class_dist, pred, feat, stats=None, phase=None, reason=None):
    """
    E4-A：在线更新每个类别的特征分布。

    每个类别维护：
        count：用于分布统计的样本数量；
        mean ：类别平均特征；
        m2   ：Welford 在线方差累计量。

    注意：
        当前第一版只在样本进入 GPA-Cache 或成功替换 GPA-Cache 时更新分布，
        避免大量未进入 cache 的伪标签样本污染类别分布。
    """
    pred = int(pred)
    x = _feature_float(feat)

    if pred not in class_dist:
        class_dist[pred] = {
            "count": 1,
            "mean": x.clone(),
            "m2": torch.zeros_like(x),
        }
    else:
        entry = class_dist[pred]
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
        stats[f"{phase}_dist_update"] += 1
        if reason is not None:
            stats[f"{phase}_dist_update_{reason}"] += 1


def _class_distribution_var(entry):
    """
    返回类别分布的对角方差。

    count <= 1 时，方差不可稳定估计，因此使用最小方差兜底。
    """
    count = int(entry["count"])

    if count <= 1:
        return torch.ones_like(entry["mean"]) * DIST_MIN_VAR

    var = entry["m2"] / float(max(count - 1, 1))
    return var.clamp_min(DIST_MIN_VAR)


def _distribution_score(class_dist, pred, feat):
    """
    计算样本对类别分布的符合度。

    score 越大，说明样本越符合该类别分布。

    当前第一版：
        score = - mean((x - mean_c)^2 / (var_c + eps))

    这个 score 不是最终分类概率，只是 GPA-Cache 更新时使用的判断指标。
    """
    pred = int(pred)

    if pred not in class_dist:
        return None

    entry = class_dist[pred]

    if int(entry["count"]) < 2:
        return None

    x = _feature_float(feat)
    mean = entry["mean"]
    var = _class_distribution_var(entry)

    raw = torch.mean(((x - mean) ** 2) / (var + DIST_EPS))
    return float((-raw).detach().cpu().item())


def _summarize_class_distribution(class_dist):
    """
    保存每个类别的分布统计，便于后续分析分布是否塌缩或被污染。
    """
    summary = {}

    for c, entry in sorted(class_dist.items(), key=lambda kv: kv[0]):
        var = _class_distribution_var(entry).detach().float().cpu()

        summary[str(c)] = {
            "count": int(entry["count"]),
            "var_mean": float(var.mean().item()),
            "var_min": float(var.min().item()),
            "var_max": float(var.max().item()),
        }

    return summary


def _update_gpa_cache(entropy_cache, gpa_cache, gpa_local_cache, class_dist, pred, global_item, local_item, shot_capacity, stats, phase, event_records=None):
    """
    E4-A：类别概率分布引导的 GPA-Cache 更新。

    沿用 E3-V2-C：
        1. 未满直接加入 GPA-Cache；
        2. 满后替换最高熵样本；
        3. 保留低熵门控。

    改动：
        用“更符合类别分布”替代“距离中心更近”。
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
            "update_rule": "low_entropy_gate_distribution_score_replace_highest_entropy",
            "new_entropy": float(curr_ent),
            "old_entropy": None if old_entropy is None else float(old_entropy),
            "new_distribution_score": None if new_score is None else float(new_score),
            "old_distribution_score": None if old_score is None else float(old_score),
            "score_margin": None if new_score is None or old_score is None else float(new_score - old_score),
            "dist_count": int(class_dist.get(int(pred), {}).get("count", 0)),
        })

    if len(gpa_cache[pred]) < shot_capacity:
        gpa_cache[pred].append(global_item)
        gpa_local_cache[pred].append(local_item)

        _sort_cache_by_entropy(gpa_cache, pred)
        _sort_local_cache_by_entropy(gpa_local_cache, pred)

        _update_class_distribution(class_dist, pred, global_item[0], stats, phase, "cache_add")

        stats[f"{phase}_gpa_add_not_full"] += 1
        stats[f"{phase}_gpa_add_not_full_distribution"] += 1
        record_event(decision="add_not_full_distribution")
        return True

    worst_global_item = gpa_cache[pred][-1]
    worst_ent = _loss_value(worst_global_item[1])

    curr_score = _distribution_score(class_dist, pred, global_item[0])
    worst_score = _distribution_score(class_dist, pred, worst_global_item[0])

    if curr_score is None or worst_score is None:
        stats[f"{phase}_gpa_reject_no_distribution"] += 1
        record_event(decision="reject_no_distribution", old_entropy=worst_ent, new_score=curr_score, old_score=worst_score)
        return False

    if curr_ent >= worst_ent:
        stats[f"{phase}_gpa_reject_entropy"] += 1
        stats[f"{phase}_gpa_reject_entropy_distribution"] += 1
        record_event(decision="reject_entropy_distribution", old_entropy=worst_ent, new_score=curr_score, old_score=worst_score)
        return False

    if curr_score > worst_score:
        gpa_cache[pred][-1] = global_item
        gpa_local_cache[pred][-1] = local_item

        _sort_cache_by_entropy(gpa_cache, pred)
        _sort_local_cache_by_entropy(gpa_local_cache, pred)

        _update_class_distribution(class_dist, pred, global_item[0], stats, phase, "cache_replace")

        stats[f"{phase}_gpa_replace_distribution"] += 1
        record_event(decision="replace_distribution", old_entropy=worst_ent, new_score=curr_score, old_score=worst_score)
        return True

    stats[f"{phase}_gpa_reject_distribution"] += 1
    record_event(decision="reject_distribution", old_entropy=worst_ent, new_score=curr_score, old_score=worst_score)
    return False


def _summarize_cache(cache):
    return {str(k): len(v) for k, v in sorted(cache.items(), key=lambda kv: kv[0])}


def _save_gpa_stats(args, stats, entropy_cache, gpa_cache, gpa_local_cache, class_dist=None, acc=None, event_records=None):
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
        "e4_variant": "E4-A",
        "distribution_scope": "visual_cache_features_only",
        "text_distribution_enabled": False,
        "e4_dist_eps": float(DIST_EPS),
        "e4_dist_min_var": float(DIST_MIN_VAR),
        "center_source": CENTER_SOURCE_LABEL,
        "final_acc": acc,
        "stats": dict(stats),
        "entropy_cache_class_counts": _summarize_cache(entropy_cache),
        "gpa_cache_class_counts": _summarize_cache(gpa_cache),
        "gpa_local_cache_class_counts": _summarize_cache(gpa_local_cache),
        "entropy_cache_total": int(sum(len(v) for v in entropy_cache.values())),
        "gpa_cache_total": int(sum(len(v) for v in gpa_cache.values())),
        "gpa_local_cache_total": int(sum(len(v) for v in gpa_local_cache.values())),
        "class_distribution_num_classes": 0 if class_dist is None else int(len(class_dist)),
        "class_distribution_summary": {} if class_dist is None else _summarize_class_distribution(class_dist),
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
        print(f"[E4-A] Saved GPA replacement events to {event_path}")

    print(f"[E4-A] Saved GPA stats to {out_dir / filename}")


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
    class_dist = {}

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
            class_dist,
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

    return entropy_cache, gpa_cache, gpa_local_cache, class_dist, stats, gpa_event_records


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
    E4-A test-time adaptation.

    Global logits:
        使用 Global Entropy Cache，即原始 Point-Cache 的正全局缓存。

    Local logits:
        使用 GPA-controlled Local Cache，即只有进入 GPA Cache 的样本贡献局部特征。

    Negative cache:
        保持原始 Point-Cache 逻辑。
    """
    entropy_cache, gpa_cache, gpa_local_cache, class_dist, build_stats, gpa_event_records = build_cache_in_advance(
        args, test_loader, lm3d_model, clip_weights, pos_cfg["shot_capacity"]
    )

    print("[E4-A] len(entropy_cache):", len(entropy_cache))
    print("[E4-A] len(gpa_cache):", len(gpa_cache))
    print("[E4-A] len(gpa_local_cache):", len(gpa_local_cache))
    print("[E4-A] entropy cache total:", sum(len(v) for v in entropy_cache.values()))
    print("[E4-A] gpa cache total:", sum(len(v) for v in gpa_cache.values()))
    print("[E4-A] gpa local cache total:", sum(len(v) for v in gpa_local_cache.values()))
    print("[E4-A] class distribution classes:", len(class_dist))

    neg_cache = {}
    gpa_cache_stats = defaultdict(int)
    for k, v in build_stats.items():
        gpa_cache_stats[k] += v

    # E4-A 保留 build 阶段形成的 GPA-Cache、GPA-local-cache 和类别分布。
    # 这样分布裁判与实际 local cache 来源保持一致。
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

            # gpa_cache、gpa_local_cache 和 class_dist 都继承预构建状态并在线更新。
            _update_gpa_cache(
                entropy_cache,
                gpa_cache,
                gpa_local_cache,
                class_dist,
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
            print("---- E4-A test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    final_acc = sum(accuracies) / len(accuracies)
    print("---- ***Final*** E4-A test accuracy: {:.2f}. ----\n".format(final_acc))

    _save_gpa_stats(args, gpa_cache_stats, entropy_cache, gpa_cache, gpa_local_cache, class_dist, final_acc, gpa_event_records)

    return final_acc
