import os
import sys
import wandb

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *


def infer(args, lm3d_model, test_loader, clip_weights):
    # assert args.cache_type == 'local', f'Local cache is expected, but got {args.cache_type}!'
    
    print('>>> In function `run`: zero-shot inference')
    
    accuracies = []
    for i, (pc, target, _, rgb) in enumerate(test_loader):
        # pc: (1, n, 3)     rgb: (1, n, 3)
        feature = torch.cat([pc, rgb], dim=-1).half()
        target = target.cuda()
        
        # pc_feats: (1, emb_dim)
        # patch_centers: (5, emb_dim)
        # clip_logits: (1, n_cls)
        # loss: a scalar
        # prob_map: (1, n_cls)
        # pred: a scalar, class index
        if args.cache_type == 'local':
            patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)
        elif args.cache_type == 'global':
            pc_feats, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)
        elif args.cache_type == 'hierarchical':
            pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(args, feature, lm3d_model, clip_weights)
        else:
            raise ValueError(f'The choice from [local, global, hierarchical] is expected, but got {args.cache_type}!')
            
        acc = cls_acc(clip_logits, target)
        accuracies.append(acc)
        
        # 修复wandb报错
        if args.wandb:
            wandb.log({"Averaged test accuracy": sum(accuracies)/len(accuracies)}, commit=True)

        if i % args.print_freq == 0:
            print("---- Zero-shot test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
        
    print("---- ***Final*** Zero-shot test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies))) 
    return sum(accuracies)/len(accuracies)



@torch.no_grad()
def clip_classifier_e0_tpe_v1_lite(args, classnames, template, clip_model):
    """
    E0-TPE-v1-lite: Lightweight Text Prototype Shrinkage.

    This function keeps Point-Cache's original 64 templates and text encoder.
    The only change is how the final text prototype is constructed from
    the 64 normalized prompt embeddings.

    Original Point-Cache:
        text prototype = normalize(mean(64 normalized prompt embeddings))

    E0-TPE-v1-lite:
        text prototype = normalize((1 - R_bar) * base_prompt + R_bar * mean_direction)

    where R_bar is the mean resultant length, measuring template consistency.
    """
    clip_weights = []

    r_bar_values = []
    base_mean_cos_values = []
    proto_shift_cos_values = []

    for classname in classnames:
        cname = classname.replace('_', ' ')
        texts = [t.format(cname) for t in template]

        if args.lm3d == 'uni3d' or args.lm3d == 'ulip':
            text_tokens = clip.tokenize(texts).cuda()
        elif args.lm3d == 'openshape':
            text_tokens = open_clip.tokenizer.tokenize(texts).cuda()
        else:
            raise ValueError(f"Unsupported lm3d type: {args.lm3d}")

        # [num_templates, dim]
        class_embeddings = clip_model.encode_text(text_tokens)

        # Normalize each prompt embedding onto the unit hypersphere.
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        # Point-Cache's original 64-template mean direction.
        mean_vec = class_embeddings.mean(dim=0)
        r_bar = mean_vec.norm().clamp(min=1e-6, max=1.0)
        mean_direction = mean_vec / r_bar

        # Base prompt: the first Point-Cache template,
        # usually "a point cloud model of {}."
        base_direction = class_embeddings[0]
        base_direction = base_direction / base_direction.norm()

        # Lightweight shrinkage:
        # R_bar close to 1 -> trust original mean direction.
        # R_bar smaller    -> shrink toward the base prompt direction.
        tpe_proto = (1.0 - r_bar) * base_direction + r_bar * mean_direction
        tpe_proto = tpe_proto / tpe_proto.norm()

        clip_weights.append(tpe_proto)

        r_bar_values.append(float(r_bar.detach().cpu()))
        base_mean_cos_values.append(float((base_direction * mean_direction).sum().detach().cpu()))
        proto_shift_cos_values.append(float((tpe_proto * mean_direction).sum().detach().cpu()))

    clip_weights = torch.stack(clip_weights, dim=1).cuda()

    r_bar_tensor = torch.tensor(r_bar_values)
    base_mean_tensor = torch.tensor(base_mean_cos_values)
    proto_shift_tensor = torch.tensor(proto_shift_cos_values)

    print("[E0-TPE-v1-lite] Text prototype shrinkage enabled.")
    print("[E0-TPE-v1-lite] R_bar mean/min/max: "
          f"{r_bar_tensor.mean().item():.6f} / "
          f"{r_bar_tensor.min().item():.6f} / "
          f"{r_bar_tensor.max().item():.6f}")
    print("[E0-TPE-v1-lite] base-mean cosine mean/min/max: "
          f"{base_mean_tensor.mean().item():.6f} / "
          f"{base_mean_tensor.min().item():.6f} / "
          f"{base_mean_tensor.max().item():.6f}")
    print("[E0-TPE-v1-lite] tpe-vs-original cosine mean/min/max: "
          f"{proto_shift_tensor.mean().item():.6f} / "
          f"{proto_shift_tensor.min().item():.6f} / "
          f"{proto_shift_tensor.max().item():.6f}")

    return clip_weights


def main(args):
    print('>>> In function `main`')
    
    clip_model, lm3d_model = load_models(args)
    
    preprocess = None
    
    # Run TDA on each dataset
    dataset_name = args.dataset
    print('>>> In loop `for`')
    
    print(f"Processing {dataset_name} dataset.")
    
    test_loader, classnames, template = build_test_data_loader(args, dataset_name, args.data_root, preprocess)
    
    print(f'>>> {[dataset_name]} classnames: {classnames} \n')
    
    # `clip_weights` are text features of shape (emb_dim, n_cls)
    clip_weights = clip_classifier_e0_tpe_v1_lite(args, classnames, template, clip_model)
    
    if args.wandb:
        if args.lm3d == 'openshape':
            prefix = f"[zs_infer-manual-prompts]/global_feat/{args.lm3d}-{args.oshape_version}"
        elif args.lm3d == 'ulip':
            prefix = f"[zs_infer-manual-prompts]/global_feat/{args.ulip_version}"
        else:
            prefix = f"[zs_infer-manual-prompts]/global_feat/{args.lm3d}"
        
        if '_c' in dataset_name and 'sonn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}/{args.cor_type}"
        elif '_c' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}/{args.cor_type}"
        elif 'scanobjnn' in dataset_name or 'scanobjectnn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sonn_variant}-{args.npoints}"
        elif 'sim2real_sonn' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.sim2real_type}-{args.npoints}"
        elif 'pointda' in dataset_name:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}"
        else:
            run_name = f"{prefix}/{dataset_name}-{args.npoints}"
        
        run = wandb.init(project="Point-TDA", name=run_name)

    zs_acc = infer(args, lm3d_model, test_loader, clip_weights)
    
    if args.wandb:
        wandb.log({f"{dataset_name}": zs_acc})
        run.finish()
        
        
if __name__ == '__main__':
    args = get_arguments()
    # Set random seed
    set_random_seed(args.seed)
    
    main(args)
