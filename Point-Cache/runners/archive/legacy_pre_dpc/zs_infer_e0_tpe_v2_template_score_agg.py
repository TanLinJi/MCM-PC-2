import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *


@torch.no_grad()
def build_template_weights(args, classnames, template, clip_model):
    """
    Build per-template text features.

    Output:
        template_weights: [num_classes, num_templates, dim]

    Difference from original Point-Cache:
        Original Point-Cache first averages the 64 template embeddings into
        one text prototype per class.

        Here we keep all 64 template embeddings and aggregate template logits
        after comparing each template with the point-cloud feature.
    """
    all_class_embeddings = []

    for classname in classnames:
        cname = classname.replace('_', ' ')
        texts = [t.format(cname) for t in template]

        if args.lm3d == 'uni3d' or args.lm3d == 'ulip':
            text_tokens = clip.tokenize(texts).cuda()
        elif args.lm3d == 'openshape':
            text_tokens = open_clip.tokenizer.tokenize(texts).cuda()
        else:
            raise ValueError(f"Unsupported lm3d type: {args.lm3d}")

        class_embeddings = clip_model.encode_text(text_tokens)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        all_class_embeddings.append(class_embeddings)

    template_weights = torch.stack(all_class_embeddings, dim=0).cuda()
    return template_weights


@torch.no_grad()
def encode_global_point_feature(args, feature, lm3d_model):
    """
    Encode only the global point-cloud feature.

    This runner is for zero-shot diagnosis, so it must not call hierarchical
    local feature paths or cache logic.
    """
    feat = feature.cuda()

    if args.lm3d == 'uni3d':
        pc_feats = lm3d_model.encode_pc(feat)
    elif args.lm3d == 'openshape':
        xyz = feat[:, :, :3]
        pc_feats = lm3d_model(xyz, feat)
    elif args.lm3d == 'ulip':
        xyz = feat[:, :, :3]
        pc_feats = lm3d_model(xyz)
    else:
        raise ValueError(f"Unsupported lm3d type: {args.lm3d}")

    pc_feats = pc_feats / pc_feats.norm(dim=-1, keepdim=True)
    return pc_feats


def aggregate_template_logits(template_logits, agg_mode="trimmean10", trim_ratio=0.10):
    """
    Args:
        template_logits: [batch, num_classes, num_templates]

    Returns:
        class_logits: [batch, num_classes]
    """
    if agg_mode == "meanlogit":
        return template_logits.mean(dim=-1)

    if agg_mode == "median":
        return template_logits.median(dim=-1).values

    if agg_mode in ["trimmean", "trimmean10"]:
        num_templates = template_logits.shape[-1]
        k = int(num_templates * trim_ratio)

        if k <= 0 or 2 * k >= num_templates:
            return template_logits.mean(dim=-1)

        sorted_logits, _ = torch.sort(template_logits, dim=-1)
        kept_logits = sorted_logits[:, :, k:num_templates - k]
        return kept_logits.mean(dim=-1)

    raise ValueError(f"Unsupported agg_mode: {agg_mode}")


@torch.no_grad()
def infer(args, lm3d_model, test_loader, template_weights):
    print(">>> In function `infer`: E0-TPE-v2 template score aggregation zero-shot inference")

    if args.cache_type != "global":
        raise ValueError(
            f"This runner is for zero-shot global diagnosis only. "
            f"Expected --cache-type global, but got {args.cache_type}."
        )

    agg_mode = os.environ.get("TPE_AGG_MODE", "trimmean10")
    trim_ratio = float(os.environ.get("TPE_TRIM_RATIO", "0.10"))

    print(f"[E0-TPE-v2] aggregation mode: {agg_mode}")
    print(f"[E0-TPE-v2] trim ratio: {trim_ratio}")

    accuracies = []

    for i, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()
        target = target.cuda()

        pc_feats = encode_global_point_feature(args, feature, lm3d_model)

        # template_weights: [C, M, D]
        # pc_feats: [B, D]
        # template_logits: [B, C, M]
        template_logits = 100.0 * torch.einsum("bd,cmd->bcm", pc_feats, template_weights)

        clip_logits = aggregate_template_logits(
            template_logits,
            agg_mode=agg_mode,
            trim_ratio=trim_ratio,
        )

        acc = cls_acc(clip_logits, target)
        accuracies.append(acc)

        if i % args.print_freq == 0:
            print("---- E0-TPE-v2 zero-shot test accuracy: {:.2f}. ----\n".format(
                sum(accuracies) / len(accuracies)
            ))

    final_acc = sum(accuracies) / len(accuracies)
    print("---- ***Final*** E0-TPE-v2 zero-shot test accuracy: {:.2f}. ----\n".format(final_acc))
    return final_acc


def main(args):
    print(">>> In function `main`: E0-TPE-v2 template score aggregation")

    clip_model, lm3d_model = load_models(args)
    preprocess = None

    dataset_name = args.dataset
    print(f"Processing {dataset_name} dataset.")

    test_loader, classnames, template = build_test_data_loader(
        args, dataset_name, args.data_root, preprocess
    )

    print(f">>> [{dataset_name}] classnames: {classnames}\n")

    template_weights = build_template_weights(args, classnames, template, clip_model)
    print(f"[E0-TPE-v2] template_weights shape: {tuple(template_weights.shape)}")

    zs_acc = infer(args, lm3d_model, test_loader, template_weights)
    return zs_acc


if __name__ == "__main__":
    args = get_arguments()
    set_random_seed(args.seed)
    main(args)
