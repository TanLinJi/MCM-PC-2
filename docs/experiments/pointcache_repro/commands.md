GPU_ID=1 bash scripts/recur-pc/run_zs_ulip2_modelnetc_add_global2.sh先跑 zero-shot，使用 ULIP-2 + ModelNet-C + add_global_2：

```bash
cd ~/autodl-tmp/MCM-PC-2/Point-Cache

CUDA_VISIBLE_DEVICES=0 python runners/zs_infer.py \
  --config configs \
  --lm3d ulip \
  --cache-type global \
  --ckpt_path weights/ulip/pointbert_ulip2.pt \
  --dataset modelnet_c \
  --sonn_variant obj_only \
  --cor_type add_global_2 \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --ulip-version ulip2
  
  
```

```
cd /root/autodl-tmp/MCM-PC-2/Point-Cache


```

这里我先去掉了：

```
--wandb-log
```

原因是你现在是复现阶段，不需要先接入 wandb，避免登录或网络问题影响跑通。



再跑 global cache：

```bash
CUDA_VISIBLE_DEVICES=0 python runners/model_with_global_cache.py \
  --config configs \
  --lm3d ulip \
  --cache-type global \
  --ckpt_path weights/ulip/pointbert_ulip2.pt \
  --dataset modelnet_c \
  --sonn_variant obj_only \
  --cor_type add_global_2 \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --ulip-version ulip2
```



再跑 hierarchical cache：

```bash
CUDA_VISIBLE_DEVICES=0 python runners/model_with_hierarchical_caches.py \
  --config configs \
  --lm3d ulip \
  --cache-type hierarchical \
  --ckpt_path weights/ulip/pointbert_ulip2.pt \
  --dataset modelnet_c \
  --sonn_variant obj_only \
  --cor_type add_global_2 \
  --npoints 1024 \
  --sim2real_type so_obj_only_9 \
  --ulip-version ulip2
```