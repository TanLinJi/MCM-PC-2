"""
E3-V2-TextProto-Guard-C

Clean implementation for Text Prototype Guard.

Key invariants:
1. Text center is fixed:
       text_center_c = clip_weights[:, c]

2. Visual center is always recomputed from CURRENT caches:
       visual_center_c = mean(current EntropyCache[c] ∪ current GPACache[c])

3. Replaced / evicted historical GPA samples do NOT participate unless they are
   still present in a current cache.

4. No Welford accumulation, no historical distribution.

5. build-stage gpa_cache is preserved into test stage. There is no test-stage
   gpa_cache reset.
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import wandb
import torch

POINT_CACHE_ROOT = Path(__file__).resolve().parents[3]
if str(POINT_CACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(POINT_CACHE_ROOT))

from runners.E3_global_prototype_alignment_cache import (
    model_with_hierarchical_caches_parallel_gpa_entropy_gpa_union_center as core,
)

CENTER_SOURCE_LABEL = "Current Entropy+GPA visual center with fixed TextProto guard"
GPA_VARIANT_NAME = "E3-V2-TextProto-Guard-C-rho0.05"

TEXT_PROTO_GUARD_RHO = float(os.environ.get("TEXT_PROTO_GUARD_RHO", "0.05"))


def _loss_value(loss):
    return core._loss_value(loss)


def _get_stats_enabled():
    return os.environ.get("GPA_SAVE_STATS", "1") != "0"


def _summarize_cache(cache):
    return {str(k): len(v) for k, v in sorted(cache.items(), key=lambda kv: kv[0])}


def _get_text_center(clip_weights, pred, ref_feat):
    """
    Fixed Text Prototype center.

    clip_weights normally has shape [D, C], where column c is the already-built
    class text prototype center. This function does not update or accumulate it.
    """
    if clip_weights is None:
        return None

    pred = int(pred)

    if clip_weights.dim() != 2:
        return None

    if pred < 0 or pred >= clip_weights.size(1):
        return None

    text_center = clip_weights[:, pred].detach()

    if text_center.dim() == 1:
        text_center = text_center.unsqueeze(0)

    text_center = text_center.to(device=ref_feat.device, dtype=ref_feat.dtype)
    text_center = text_center / text_center.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return text_center


def _compute_current_visual_center(entropy_cache, gpa_cache, pred):
    """
    Current visual center only.

    This uses only samples that are currently present in EntropyCache[pred] or
    GPACache[pred]. There is no historical accumulation.
    """
    return core._compute_gpa_center(gpa_cache, int(pred), entropy_cache=entropy_cache)


def _feature_distance(feat, center):
    return core._feature_distance_to_center(feat, center)


def _sort_gpa_and_local_by_entropy(gpa_cache, gpa_local_cache, pred):
    """
    Sort GPA global and local caches as paired items, so they cannot become
    misaligned after add / replace.
    """
    pred = int(pred)

    paired = list(zip(gpa_cache[pred], gpa_local_cache[pred]))
    paired = sorted(paired, key=lambda pair: _loss_value(pair[0][1]))

    gpa_cache[pred] = [pair[0] for pair in paired]
    gpa_local_cache[pred] = [pair[1] for pair in paired]


def _assert_cache_pair_aligned(gpa_cache, gpa_local_cache, pred, where):
    pred = int(pred)

    if pred not in gpa_cache or pred not in gpa_local_cache:
        raise RuntimeError(
            f"[{where}] GPA-Cache and GPA-local-cache key mismatch for class={pred}: "
            f"gpa_has={pred in gpa_cache}, local_has={pred in gpa_local_cache}"
        )

    if len(gpa_cache[pred]) != len(gpa_local_cache[pred]):
        raise RuntimeError(
            f"[{where}] GPA-Cache and GPA-local-cache length mismatch for class={pred}: "
            f"len(gpa_cache)={len(gpa_cache[pred])}, "
            f"len(gpa_local_cache)={len(gpa_local_cache[pred])}"
        )


def _safe_ratio(numerator, denominator):
    denominator = float(denominator)
    if denominator == 0.0:
        return None
    return float(numerator) / denominator


def _update_gpa_cache_guard(
    entropy_cache,
    gpa_cache,
    gpa_local_cache,
    pred,
    global_item,
    local_item,
    shot_capacity,
    stats,
    phase,
    event_records=None,
    clip_weights=None,
):
    """
    Guard update rule.

    Initialization:
        if GPACache[c] is not full:
            add current sample to GPACache[c] and GPALocalCache[c]

    Full cache:
        x_high = highest-entropy sample in current GPACache[c]

        replace if:

            entropy_new < entropy_high
            and
            (
                d_visual_new < d_visual_high
                or
                (
                    d_visual_new <= d_visual_high * (1 + rho)
                    and
                    d_text_new < d_text_high
                )
            )

    Notes:
        - visual center is recomputed from current EntropyCache and current GPACache.
        - text center is fixed from clip_weights.
        - no historical accumulated samples are used.
    """
    pred = int(pred)

    if pred not in gpa_cache:
        gpa_cache[pred] = []
        gpa_local_cache[pred] = []

    _assert_cache_pair_aligned(gpa_cache, gpa_local_cache, pred, f"{phase}_before_update")

    curr_ent = _loss_value(global_item[1])

    def record_event(
        decision,
        branch=None,
        old_entropy=None,
        d_visual_new=None,
        d_visual_high=None,
        d_text_new=None,
        d_text_high=None,
    ):
        if event_records is None:
            return

        visual_ratio = None
        if d_visual_new is not None and d_visual_high is not None:
            visual_ratio = _safe_ratio(d_visual_new, d_visual_high)

        text_margin = None
        if d_text_new is not None and d_text_high is not None:
            text_margin = float(d_text_high) - float(d_text_new)

        event_records.append({
            "phase": phase,
            "class_index": int(pred),
            "decision": decision,
            "branch": branch,
            "update_rule": "low_entropy_visual_or_textproto_guard",
            "rho_visual": float(TEXT_PROTO_GUARD_RHO),
            "new_entropy": float(curr_ent),
            "old_entropy": None if old_entropy is None else float(old_entropy),
            "d_visual_new": None if d_visual_new is None else float(d_visual_new),
            "d_visual_high": None if d_visual_high is None else float(d_visual_high),
            "d_text_new": None if d_text_new is None else float(d_text_new),
            "d_text_high": None if d_text_high is None else float(d_text_high),
            "visual_ratio": visual_ratio,
            "text_margin": text_margin,
        })

    if len(gpa_cache[pred]) < shot_capacity:
        gpa_cache[pred].append(global_item)
        gpa_local_cache[pred].append(local_item)

        _sort_gpa_and_local_by_entropy(gpa_cache, gpa_local_cache, pred)
        _assert_cache_pair_aligned(gpa_cache, gpa_local_cache, pred, f"{phase}_after_add")

        stats[f"{phase}_gpa_add_not_full"] += 1
        stats[f"{phase}_gpa_add_not_full_guard"] += 1

        record_event(decision="add_not_full_guard", branch="init")
        return True

    visual_center = _compute_current_visual_center(entropy_cache, gpa_cache, pred)

    if visual_center is None:
        stats[f"{phase}_gpa_no_visual_center_reject"] += 1
        record_event(decision="reject_no_visual_center")
        return False

    text_center = _get_text_center(clip_weights, pred, global_item[0])

    worst_global_item = gpa_cache[pred][-1]
    worst_ent = _loss_value(worst_global_item[1])

    d_visual_new = _feature_distance(global_item[0], visual_center)
    d_visual_high = _feature_distance(worst_global_item[0], visual_center)

    d_text_new = None
    d_text_high = None

    if text_center is not None:
        d_text_new = _feature_distance(global_item[0], text_center)
        d_text_high = _feature_distance(worst_global_item[0], text_center)

    if curr_ent >= worst_ent:
        stats[f"{phase}_gpa_reject_entropy"] += 1
        stats[f"{phase}_gpa_reject_entropy_guard"] += 1

        record_event(
            decision="reject_entropy_guard",
            branch="entropy",
            old_entropy=worst_ent,
            d_visual_new=d_visual_new,
            d_visual_high=d_visual_high,
            d_text_new=d_text_new,
            d_text_high=d_text_high,
        )
        return False

    visual_branch = d_visual_new < d_visual_high

    text_guard_branch = False
    if d_text_new is not None and d_text_high is not None:
        text_guard_branch = (
            d_visual_new <= d_visual_high * (1.0 + float(TEXT_PROTO_GUARD_RHO))
            and
            d_text_new < d_text_high
        )

    if visual_branch or text_guard_branch:
        branch = "visual_branch" if visual_branch else "text_guard_branch"

        gpa_cache[pred][-1] = global_item
        gpa_local_cache[pred][-1] = local_item

        _sort_gpa_and_local_by_entropy(gpa_cache, gpa_local_cache, pred)
        _assert_cache_pair_aligned(gpa_cache, gpa_local_cache, pred, f"{phase}_after_replace")

        stats[f"{phase}_gpa_replace_guard"] += 1
        stats[f"{phase}_gpa_replace_{branch}"] += 1

        record_event(
            decision="replace_guard",
            branch=branch,
            old_entropy=worst_ent,
            d_visual_new=d_visual_new,
            d_visual_high=d_visual_high,
            d_text_new=d_text_new,
            d_text_high=d_text_high,
        )
        return True

    stats[f"{phase}_gpa_reject_visual_and_text_guard"] += 1

    record_event(
        decision="reject_visual_and_text_guard",
        branch="reject_guard",
        old_entropy=worst_ent,
        d_visual_new=d_visual_new,
        d_visual_high=d_visual_high,
        d_text_new=d_text_new,
        d_text_high=d_text_high,
    )
    return False


def _save_guard_stats(args, stats, entropy_cache, gpa_cache, gpa_local_cache, acc=None, event_records=None):
    if not _get_stats_enabled():
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
        "rho_visual": float(TEXT_PROTO_GUARD_RHO),
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
        print(f"[E3-V2-TextProto-Guard-C] Saved GPA replacement events to {event_path}")

    print(f"[E3-V2-TextProto-Guard-C] Saved GPA stats to {out_dir / filename}")


@torch.no_grad()
def build_cache_in_advance(args, test_loader, lm3d_model, clip_weights, shot_capacity):
    """
    Build current caches.

    Returned gpa_cache and gpa_local_cache are the exact objects that will be
    used and updated in run_test_tda. They must not be cleared after this point.
    """
    print("*" * 10, "Building [global entropy] and [TextProto-Guard GPA/local] pos. cache ...", "*" * 10, "\n")

    entropy_cache = {}
    gpa_cache = {}
    gpa_local_cache = {}
    stats = defaultdict(int)
    event_records = []

    for pc, _, _, rgb in test_loader:
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = core.get_logits(args, feature, lm3d_model, clip_weights)

        pred = int(pred)
        global_item = [pc_feats, loss]
        local_item = [patch_centers, loss]

        core._update_entropy_cache(
            entropy_cache,
            pred,
            global_item,
            shot_capacity,
            stats,
            "build",
        )

        _update_gpa_cache_guard(
            entropy_cache=entropy_cache,
            gpa_cache=gpa_cache,
            gpa_local_cache=gpa_local_cache,
            pred=pred,
            global_item=global_item,
            local_item=local_item,
            shot_capacity=shot_capacity,
            stats=stats,
            phase="build",
            event_records=event_records,
            clip_weights=clip_weights,
        )

        cache_num = sum(len(entropy_cache[key]) for key in entropy_cache)
        num_classes = clip_logits.size(1)
        full_num = shot_capacity * num_classes

        if cache_num == full_num:
            print("*" * 10, "Building [global entropy] cache is full.", "*" * 10, "\n")
            break

    return entropy_cache, gpa_cache, gpa_local_cache, stats, event_records


@torch.no_grad()
def run_test_tda(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights):
    """
    Main TTA loop.

    Critical design:
        gpa_cache returned by build_cache_in_advance is preserved.
        This function never resets gpa_cache after prebuilding.
    """
    entropy_cache, gpa_cache, gpa_local_cache, build_stats, event_records = build_cache_in_advance(
        args,
        test_loader,
        lm3d_model,
        clip_weights,
        pos_cfg["shot_capacity"],
    )

    print("[E3-V2-TextProto-Guard-C] len(entropy_cache):", len(entropy_cache))
    print("[E3-V2-TextProto-Guard-C] len(gpa_cache):", len(gpa_cache))
    print("[E3-V2-TextProto-Guard-C] len(gpa_local_cache):", len(gpa_local_cache))
    print("[E3-V2-TextProto-Guard-C] entropy cache total:", sum(len(v) for v in entropy_cache.values()))
    print("[E3-V2-TextProto-Guard-C] gpa cache total:", sum(len(v) for v in gpa_cache.values()))
    print("[E3-V2-TextProto-Guard-C] gpa local cache total:", sum(len(v) for v in gpa_local_cache.values()))

    neg_cache = {}

    gpa_cache_stats = defaultdict(int)
    for k, v in build_stats.items():
        gpa_cache_stats[k] += v

    accuracies = []

    pos_enabled = pos_cfg["enabled"]
    neg_enabled = neg_cfg["enabled"]

    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ["shot_capacity", "alpha", "beta"]}

    if neg_enabled:
        neg_params = {
            k: neg_cfg[k]
            for k in ["shot_capacity", "alpha", "beta", "entropy_threshold", "mask_threshold"]
        }

    for i, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()

        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = core.get_logits(
            args,
            feature,
            lm3d_model,
            clip_weights,
        )

        pred = int(pred)
        target = target.cuda()
        prop_entropy = core.get_entropy(loss, clip_weights)

        if pos_enabled:
            global_item = [pc_feats, loss]
            local_item = [patch_centers, loss]

            core._update_entropy_cache(
                entropy_cache,
                pred,
                global_item,
                pos_params["shot_capacity"],
                gpa_cache_stats,
                "test",
            )

            _update_gpa_cache_guard(
                entropy_cache=entropy_cache,
                gpa_cache=gpa_cache,
                gpa_local_cache=gpa_local_cache,
                pred=pred,
                global_item=global_item,
                local_item=local_item,
                shot_capacity=pos_params["shot_capacity"],
                stats=gpa_cache_stats,
                phase="test",
                event_records=event_records,
                clip_weights=clip_weights,
            )

        if neg_enabled and neg_params["entropy_threshold"]["lower"] < prop_entropy < neg_params["entropy_threshold"]["upper"]:
            core._update_negative_cache(
                neg_cache,
                pred,
                [pc_feats, loss, prob_map],
                neg_params["shot_capacity"],
                gpa_cache_stats,
                "test",
            )

        final_logits = clip_logits.clone()

        if pos_enabled and entropy_cache:
            final_logits += core.compute_cache_logits(
                pc_feats,
                entropy_cache,
                pos_params["alpha"],
                pos_params["beta"],
                clip_weights,
            )

            if gpa_local_cache:
                final_logits += core.compute_local_cache_logits(
                    patch_centers,
                    gpa_local_cache,
                    pos_params["alpha"],
                    pos_params["beta"],
                    clip_weights,
                )

        if neg_enabled and neg_cache:
            final_logits -= core.compute_cache_logits(
                pc_feats,
                neg_cache,
                neg_params["alpha"],
                neg_params["beta"],
                clip_weights,
                (
                    neg_params["mask_threshold"]["lower"],
                    neg_params["mask_threshold"]["upper"],
                ),
            )

        acc = core.cls_acc(final_logits, target)
        accuracies.append(acc)

        wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

        if i % args.print_freq == 0:
            print("---- E3-V2-TextProto-Guard-C test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

    final_acc = sum(accuracies) / len(accuracies)

    print("---- ***Final*** E3-V2-TextProto-Guard-C test accuracy: {:.2f}. ----\n".format(final_acc))

    _save_guard_stats(
        args=args,
        stats=gpa_cache_stats,
        entropy_cache=entropy_cache,
        gpa_cache=gpa_cache,
        gpa_local_cache=gpa_local_cache,
        acc=final_acc,
        event_records=event_records,
    )

    return final_acc
