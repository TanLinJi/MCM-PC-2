# Point-Cache Baseline 复现问题总结与实验管理规范

## 1. 文档目的

本文档用于系统总结本轮 Point-Cache baseline 复现过程中出现的所有主要问题，而不仅仅是最后 34 组的问题。

本轮复现覆盖的实验范围为：

| Backbone | ModelNet clean | ModelNet-C all35 | ScanObjNN clean hardest | ScanObjNN-C hardest all35 |
|---|---|---|---|---|
| ULIP | 01 | 02 | 03 | 04 |
| ULIP-2 | 11 | 12 | 13 | 14 |
| OpenShape | 21 | 22 | 23 | 24 |
| Uni3D | 31 | 32 | 33 | 34 |

其中，当前重点复现和归档的主线是 Point-Cache baseline，包括：

| 方法 | 含义 |
|---|---|
| Zero-shot | 无缓存基础推理 |
| Zero-shot + Global Cache | 加入全局缓存 |
| Zero-shot + Global Cache + Local Cache | 加入全局缓存和局部缓存，即完整 Point-Cache / Hierarchical Cache |

本文档的目标有四个：

1. 记录复现过程中出现过的所有重要问题；
2. 解释每个问题为什么发生；
3. 总结最终修复方式；
4. 固化后续 MCM-PC / Point-Cache 实验必须遵守的检查规范。

这份文档应作为后续实验前的必读文件，尤其适用于以下情况：

| 场景 | 为什么需要看本文档 |
|---|---|
| 复制已有 runner 写新实验 | 防止残留旧数据集路径和 summary 字段 |
| 使用 Uni3D backbone | 防止 checkpoint 选错 |
| 跑 all35 corruption 实验 | 防止 summary metadata 错误 |
| 判断结果是否可归档 | 防止只看 accuracy，不看 metadata |
| 准备 Git commit | 防止 results、weights、临时备份文件被提交 |
| 设计 MCM-PC 新方法 | 明确 baseline 中最困难的 corruption 和 failure modes |

---

## 2. 本轮复现中的主要问题总览

本轮 baseline 复现中出现的问题大致可以分为八类：

| 问题类别 | 典型表现 | 影响 |
|---|---|---|
| checkpoint 选择错误 | Uni3D 使用了通用 pc_encoder checkpoint | 结果稳定偏低，无法对齐原文 |
| runner 复制残留 | 从 32 组复制到 34 组后残留 modelnet_c | summary metadata 错误 |
| loader root 与 summary root 混淆 | 把 data/sonn_c/hardest 当作 SONN_C loader root | 找不到 shape_names.txt |
| common 脚本参数残留 | 34 common 中仍有 --dataset modelnet_c | 代码可读性差，后续易误判 |
| summary 与 logs 清理不彻底 | 旧日志、重复 summary 行残留 | 行数和 log_path 不干净 |
| 旧 summary 与新 summary 混淆 | 上传的是旧 summary，服务器当前结果已更新 | 分析结论混乱 |
| 结果判断标准不完整 | 只看脚本是否跑完或 accuracy 是否接近原文 | 错误结果可能被归档 |
| Git 提交粒度与排除规则 | results、weights、备份文件可能进入暂存区 | 仓库污染、复现记录混乱 |

最重要的总原则是：

    脚本执行成功不等于复现正确。
    accuracy 接近原文也不等于复现正确。
    summary metadata 正确性与 accuracy 正确性同等重要。

---

## 3. 问题一：Uni3D checkpoint 选择错误

### 3.1 问题背景

在 Uni3D 相关实验中，最早的关键问题是 checkpoint 选择。

服务器中原本存在一个 Uni3D checkpoint：

    weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

这个文件可以正常加载，也可以跑出结果，因此一开始容易被误认为是正式复现 checkpoint。

但实际复现 Point-Cache 原文结果时，Uni3D checkpoint 并不是通用的，而是与数据集相关。

### 3.2 错误表现

在 31 组或 33 组早期尝试中，如果使用通用 pc_encoder checkpoint，结果会稳定低于原文。

例如，在 33 组 Uni3D × ScanObjNN clean hardest 中，旧 checkpoint 下的结果为：

| 实验 | 方法 | 旧 checkpoint 结果 | 原文参考 | 差异 |
|---|---|---:|---:|---:|
| 33_1 | Zero-shot | 41.33 | 46.04 | -4.71 |
| 33_2 | ZS + Global | 43.79 | 50.28 | -6.49 |
| 33_3 | ZS + Global + Local | 45.49 | 51.13 | -5.64 |

这说明旧 checkpoint 并不是正式复现所需 checkpoint。

### 3.3 正确 checkpoint 规则

最终确认，Uni3D checkpoint 必须按数据集设置区分：

| 实验组 | 数据设置 | 正式 checkpoint |
|---|---|---|
| 31 | Uni3D × ModelNet clean | weights/uni3d/modelnet40/model.pt |
| 32 | Uni3D × ModelNet-C all35 | weights/uni3d/modelnet40/model.pt |
| 33 | Uni3D × ScanObjNN clean hardest | weights/uni3d/scanobjnn/model.pt |
| 34 | Uni3D × ScanObjNN-C hardest all35 | weights/uni3d/scanobjnn/model.pt |

不能用于正式复现的 checkpoint：

    weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

除非明确作为 ablation 或额外对比，否则不能把它作为 baseline checkpoint。

### 3.4 后续规范

以后所有 Uni3D 实验必须在以下位置明确记录 checkpoint：

| 位置 | 要求 |
|---|---|
| shell 脚本 | CKPT_PATH 写清楚 |
| Python runner | fail-fast 检查 expected_ckpt |
| log 输出 | 打印实际 ckpt_path |
| summary 文档 | 写入 checkpoint 路径 |
| 组内 summary | 解释 checkpoint 对结果的影响 |
| Git 文档 | 记录 checkpoint 下载脚本 |

建议在 runner 中加入：

    expected_ckpt = "weights/uni3d/scanobjnn/model.pt"
    if getattr(args, "ckpt_path", None) != expected_ckpt:
        raise RuntimeError(
            f"Unexpected Uni3D checkpoint: {getattr(args, 'ckpt_path', None)}; "
            f"expected {expected_ckpt}"
        )

---

## 4. 问题二：checkpoint 下载脚本最初缺失

### 4.1 问题背景

在发现 Uni3D 需要 dataset-specific checkpoint 后，需要下载：

| checkpoint | 用途 |
|---|---|
| weights/uni3d/modelnet40/model.pt | ModelNet / ModelNet-C |
| weights/uni3d/scanobjnn/model.pt | ScanObjNN / ScanObjNN-C hardest |

最初服务器无法直接访问 HuggingFace，出现网络不可达问题。

### 4.2 问题表现

直接下载时曾出现：

    Network is unreachable

后续通过镜像源解决：

    HF_ENDPOINT=https://hf-mirror.com

### 4.3 最终规范

已将 Uni3D checkpoint 下载脚本放入：

    Point-Cache/scripts/data_download_scripts/download_uni3d_checkpoints.sh

该脚本的作用是记录并复现 checkpoint 下载过程。

后续注意：

1. checkpoint 文件体积很大，不应提交到 Git；
2. 下载脚本可以提交；
3. 文档中应记录 checkpoint 路径和 sha256；
4. 每次使用 Uni3D 前，应检查 checkpoint 是否存在；
5. 不能因为服务器中已有某个 .pt 文件，就默认它是正式 checkpoint。

推荐检查命令：

    cd /root/autodl-tmp/MCM-PC-2/Point-Cache

    find weights/uni3d -maxdepth 5 -type f \( -name "*.pt" -o -name "*.bin" \) \
      -printf "%p\t%s bytes\n" | sort

    sha256sum \
      weights/uni3d/modelnet40/model.pt \
      weights/uni3d/scanobjnn/model.pt \
      weights/uni3d/open_clip_pytorch_model/laion2b_s9b_b144k.bin \
      weights/uni3d/pc_encoder/uni3d_g_ensembled_model.pt

---

## 5. 问题三：all35 runner 复制时只检查了 shell，漏查 Python 内部字段

### 5.1 问题背景

all35 实验与 clean 单文件实验不同。

clean 单文件实验通常结构较简单：

| 类型 | 结构 |
|---|---|
| clean 实验 | 单个 shell 脚本调用原始 runner |
| 输出 | 1 行 summary，1 个 log |

all35 实验结构更复杂：

| 类型 | 结构 |
|---|---|
| all35 实验 | common shell + Python optimized runner |
| 输出 | 35 行 summary，35 个 log |
| 特点 | Python 内部循环 35 个 cor_type |

all35 runner 内部会自己写 summary，因此它内部的 metadata 字段非常重要。

### 5.2 问题表现

在 34 组中，最初从 32 组 ModelNet-C all35 runner 复制生成：

    run_uni3d_scanobjnnc_hardest_corruptions_all35.py

但复制后只重点检查了：

| 已检查 | 实际不足 |
|---|---|
| common shell 的 CKPT_PATH | 不够 |
| 脚本能否运行 | 不够 |
| accuracy 是否接近原文 | 不够 |
| logs 数量是否正确 | 不够 |

漏查了 Python runner 内部：

    args.dataset
    args.data_root
    args.sonn_variant
    data_root
    data_file
    append_summary_row
    summary.csv 写入字段

这导致 runner 内部仍残留 32 组 ModelNet-C 字段。

### 5.3 后续规范

以后复制 all35 runner 时，必须执行以下检查：

    cd /root/autodl-tmp/MCM-PC-2/Point-Cache

    grep -nE "args.dataset|args.data_root|args.sonn_variant|data_root|data_file|summary|append_summary_row|modelnet_c|sonn_c|ckpt" \
      runners/baseline/<runner>.py

不能只检查 shell 脚本。

---

## 6. 问题四：34 组从 32 组复制后残留 ModelNet-C metadata

### 6.1 问题背景

34 组是：

    Uni3D × ScanObjNN-C hardest all35

但它是从 32 组：

    Uni3D × ModelNet-C all35

复制 runner 而来。

32 组中的字段本来是正确的：

    args.modelnet_c_root = "data/modelnet_c"
    data_root = args.modelnet_c_root
    sonn_variant = "-"

但复制到 34 组后，这些字段变成错误残留。

### 6.2 错误表现

错误 summary 曾显示：

    dataset = sonn_c
    data_root = data/modelnet_c
    file = data/modelnet_c/{cor_type}.h5
    sonn_variant = -

这非常危险，因为 dataset 已经是 sonn_c，看起来像部分正确，但 data_root、file 和 sonn_variant 仍然是错的。

### 6.3 正确 metadata

34 组最终正确 metadata 为：

    dataset = sonn_c
    data_root = data/sonn_c/hardest
    file = data/sonn_c/hardest/{cor_type}.h5
    sonn_variant = hardest
    backbone = Uni3D

### 6.4 后续规范

不能只看 dataset 字段。必须同时检查：

| 字段 | 34 组正确值 |
|---|---|
| dataset | sonn_c |
| data_root | data/sonn_c/hardest |
| file | data/sonn_c/hardest/{cor_type}.h5 |
| sonn_variant | hardest |
| backbone | Uni3D |
| method | zs / zs_global / zs_global_local |

---

## 7. 问题五：ScanObjNN-C 的 loader root 与 summary root 不同

### 7.1 问题背景

修复 34 组 metadata 时，曾经把 loader root 也设置成：

    data/sonn_c/hardest

这导致运行时报错：

    FileNotFoundError: data/sonn_c/hardest/shape_names.txt

### 7.2 根本原因

SONN_C 数据集类需要从：

    data/sonn_c/shape_names.txt

读取类别名。

但 corrupted h5 文件位于：

    data/sonn_c/hardest/{cor_type}.h5

因此对于 ScanObjNN-C hardest：

| 路径类型 | 正确值 | 用途 |
|---|---|---|
| loader root | data/sonn_c | 数据集类读取 shape_names.txt |
| args.data_root | data/sonn_c | build_test_data_loader 使用 |
| actual h5 root | data/sonn_c/hardest | corruption h5 文件所在目录 |
| summary data_root | data/sonn_c/hardest | summary 中记录实际 h5 目录 |
| file | data/sonn_c/hardest/{cor_type}.h5 | 当前 cor_type 的实际文件 |

### 7.3 正确代码逻辑

34 runner 中正确逻辑应为：

    args.dataset = "sonn_c"
    args.sonn_variant = "hardest"

    # SONN_C loader 使用
    args.sonn_c_root = "data/sonn_c"
    args.data_root = args.sonn_c_root

    # summary / log 记录实际 h5 文件目录
    data_root = "data/sonn_c/hardest"
    sonn_variant = "hardest"
    data_file = Path(data_root) / f"{cor_type}.h5"

### 7.4 后续规范

以后遇到 ScanObjNN / ScanObjNN-C，必须先查看 dataset class，而不是凭路径名称推断。

需要检查：

    Point-Cache/datasets/sonn_c.py
    Point-Cache/utils/utils.py

确认数据集类如何使用：

    args.data_root
    args.sonn_variant
    shape_names.txt
    h5 file path

---

## 8. 问题六：common 脚本中残留 --dataset modelnet_c

### 8.1 问题背景

34 runner 修正后，common shell 中仍然残留：

    --dataset modelnet_c

虽然 Python runner 内部已经覆盖为：

    args.dataset = "sonn_c"

所以最终 summary 没有再错，但这个参数残留会误导后续阅读和维护。

### 8.2 最终修复

34 common 脚本最终修改为：

    --dataset sonn_c \
    --sonn_variant hardest \
    --cor_type add_global_0 \

并保留：

    CKPT_PATH="weights/uni3d/scanobjnn/model.pt"

### 8.3 后续规范

即使某个参数在 runner 内部会被覆盖，shell 脚本也不能保留错误值。

原因：

1. 容易误导下一次复制；
2. 容易误导代码审查；
3. 容易造成 runner 修改后重新生效；
4. 文档和脚本不一致会降低可信度。

---

## 9. 问题七：summary / logs 旧文件残留与重复结果

### 9.1 问题背景

在一些 all35 实验中，出现过重复日志或旧日志残留。

例如某些实验中 summary 是 35 行，但 logs 目录中存在 70 个 log。这通常来自：

1. 同一实验重复跑过；
2. 新 summary 覆盖了旧 summary，但 logs 没清理；
3. 两张 GPU 分别跑过同一批 cor_type；
4. 旧结果目录没有删除干净。

### 9.2 影响

这种情况会造成：

| 现象 | 风险 |
|---|---|
| logs 文件数多于 summary 行数 | 无法一一对应 |
| 同一 cor_type 有多个 log | 不知道哪个是正式结果 |
| summary 中 log_path 唯一，但 logs 目录有旧文件 | 文档归档不干净 |
| 旧结果和新结果混在一起 | 后续分析容易误判 |

### 9.3 后续规范

all35 实验正式归档前必须检查：

    cd /root/autodl-tmp/MCM-PC-2/Point-Cache

    for exp in <exp_list>
    do
      echo "============================================================"
      echo "$exp"

      echo "summary 行数："
      tail -n +2 "results/baseline/${exp}/summary.csv" | wc -l

      echo "唯一 cor_type 数："
      tail -n +2 "results/baseline/${exp}/summary.csv" | cut -d',' -f6 | sort -u | wc -l

      echo "唯一 log_path 数："
      tail -n +2 "results/baseline/${exp}/summary.csv" | cut -d',' -f15 | sort -u | wc -l

      echo "logs 文件数："
      find "results/baseline/${exp}/logs" -maxdepth 1 -name '*.log' | wc -l

      echo "status 统计："
      tail -n +2 "results/baseline/${exp}/summary.csv" | cut -d',' -f13 | sort | uniq -c

      echo "summary 前两行："
      head -2 "results/baseline/${exp}/summary.csv"
    done

正式归档标准：

| 实验类型 | summary 行数 | 唯一 cor_type | 唯一 log_path | logs 文件数 | status |
|---|---:|---:|---:|---:|---|
| clean 单文件 | 1 | 1 | 1 | 1 | 1 done |
| all35 | 35 | 35 | 35 | 35 | 35 done |

---

## 10. 问题八：旧 summary 与服务器当前 summary 混淆

### 10.1 问题背景

在 34 组排查后期，一度出现：

1. 上传的 summary34-1 / summary34-2 显示旧 metadata；
2. 服务器当前 summary 已经是正确 metadata；
3. 34_3 summary 正确；
4. 因此误以为 34_1 / 34_2 仍然错误。

后续通过服务器直接检查确认，当前服务器结果已经正确。

### 10.2 后续规范

以后如果上传文件和服务器当前结果不一致，优先以服务器当前文件为准。

确认命令：

    cd /root/autodl-tmp/MCM-PC-2/Point-Cache

    for exp in <exp_list>
    do
      echo "============================================================"
      echo "$exp"
      echo "summary 修改时间："
      stat -c "%y %n" "results/baseline/${exp}/summary.csv"

      echo "summary 前两行："
      head -2 "results/baseline/${exp}/summary.csv"
    done

如果需要再次上传，应明确从当前服务器路径复制：

    results/baseline/<exp>/summary.csv

不要上传旧下载文件。

---

## 11. 问题九：没有先做 fail-fast，导致错误配置能跑完整组

### 11.1 问题背景

34 组早期 runner 没有强制检查：

    expected_dataset
    expected_loader_root
    expected_summary_data_root
    expected_sonn_variant
    expected_ckpt

因此即使 metadata 不正确，程序仍然可以跑完整 35 个 setting。

### 11.2 后续规范

all35 runner 必须加入 fail-fast 检查。

34 组推荐检查如下：

    expected_dataset = "sonn_c"
    expected_loader_root = "data/sonn_c"
    expected_summary_data_root = "data/sonn_c/hardest"
    expected_sonn_variant = "hardest"
    expected_ckpt = "weights/uni3d/scanobjnn/model.pt"

    if args.dataset != expected_dataset:
        raise RuntimeError(...)
    if args.sonn_c_root != expected_loader_root:
        raise RuntimeError(...)
    if args.data_root != expected_loader_root:
        raise RuntimeError(...)
    if data_root != expected_summary_data_root:
        raise RuntimeError(...)
    if args.sonn_variant != expected_sonn_variant:
        raise RuntimeError(...)
    if sonn_variant != expected_sonn_variant:
        raise RuntimeError(...)
    if getattr(args, "ckpt_path", None) != expected_ckpt:
        raise RuntimeError(...)

这样只要配置不符合预期，就会在第一时间报错，而不是静默跑完整组。

---

## 12. 问题十：文档和 Git 提交粒度需要固定

### 12.1 文档组织规范

当前 baseline 复现实验文档放在：

    docs/experiments/baseline

每个子实验单独一个文档，组内再有一个 summary 文档。

例如 34 组：

    docs/experiments/baseline/34_1_uni3d_scanobjnnc_hardest_corruptions_all35_zs.md
    docs/experiments/baseline/34_2_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global.md
    docs/experiments/baseline/34_3_uni3d_scanobjnnc_hardest_corruptions_all35_zs_global_local.md
    docs/experiments/baseline/34_uni3d_scanobjnnc_hardest_corruptions_all35_summary.md

本文档作为问题复盘文档：

    docs/experiments/baseline/baseline_reproduction_issue_summary.md

### 12.2 Git 提交规范

用户要求 Git 提交按变更类型拆分，不要混在一起。

推荐拆分：

| commit 类型 | 内容 |
|---|---|
| runner / scripts | 实验脚本、common 脚本、runner |
| docs | 实验文档 |
| issue summary | 问题复盘文档 |
| data download scripts | checkpoint 下载脚本 |
| 不提交 | results、weights、wandb、logs、备份文件 |

提交前必须检查：

    cd /root/autodl-tmp/MCM-PC-2

    git status --short

    git status --short | grep -E "Point-Cache/results|Point-Cache/weights" || echo "OK: no results or weights in status output"

    find Point-Cache/runners/baseline -maxdepth 1 -name "*.bak*" -print

如果有备份文件：

    find Point-Cache/runners/baseline -maxdepth 1 -name "*.bak*" -delete

提交脚本和文档应分开：

    git commit -m "Add Uni3D ScanObjNN-C hardest all35 baseline scripts"
    git commit -m "Document Uni3D ScanObjNN-C hardest all35 baseline results"
    git commit -m "Document baseline reproduction issues"

---

## 13. 后续 all35 runner 复制标准流程

以后复制 all35 runner 时，必须遵守以下流程。

### 13.1 第一步：明确实验矩阵

写清楚：

| 字段 | 内容 |
|---|---|
| exp group | 例如 34 |
| backbone | 例如 Uni3D |
| dataset | 例如 sonn_c |
| data root | 例如 data/sonn_c |
| actual h5 root | 例如 data/sonn_c/hardest |
| variant | 例如 hardest |
| checkpoint | 例如 weights/uni3d/scanobjnn/model.pt |
| methods | zs / zs_global / zs_global_local |
| cor_type 范围 | 7 × 5 = 35 |

---

### 13.2 第二步：检查 common shell

    grep -nE "RUNNER|CKPT_PATH|dataset|sonn_variant|cor_type|CACHE_TYPE|METHOD|EXP_ID" \
      Point-Cache/scripts/baseline/<common_script>.sh

必须确认：

1. runner 是目标 runner；
2. checkpoint 正确；
3. dataset 正确；
4. sonn_variant 正确；
5. 没有旧 dataset 残留；
6. cache_type 与方法一致。

---

### 13.3 第三步：检查 Python runner 内部字段

    grep -nE "args.dataset|args.data_root|args.sonn_variant|data_root|data_file|append_summary_row|modelnet_c|sonn_c|checkpoint|ckpt" \
      Point-Cache/runners/baseline/<runner>.py

必须确认：

| 字段 | 必须检查 |
|---|---|
| args.dataset | 是否目标数据集 |
| args.data_root | 是否 loader 需要的 root |
| args.sonn_variant | 是否正确 |
| data_root | 是否 summary 需要的 actual h5 root |
| data_file | 是否指向实际 h5 |
| append_summary_row | 所有分支是否一致 |
| ckpt_path | 是否正确 |
| 禁止旧字段 | 旧数据集路径不能残留 |

---

### 13.4 第四步：检查 summary 写入分支

    python - <<'PY'
    from pathlib import Path

    p = Path("Point-Cache/runners/baseline/<runner>.py")
    lines = p.read_text(encoding="utf-8").splitlines()

    for i, line in enumerate(lines, start=1):
        if "append_summary_row(summary_file" in line:
            print("=" * 90)
            print(f"append_summary_row at line {i}")
            start = max(1, i - 8)
            end = min(len(lines), i + 35)
            for j in range(start, end + 1):
                print(f"{j:04d}: {lines[j-1]}")
    PY

必须确认 missing_file、done、failed 三个分支都写同样 metadata。

---

### 13.5 第五步：语法和禁止项检查

    python -m py_compile Point-Cache/runners/baseline/<runner>.py

    grep -nE "data/modelnet_c|modelnet_c_root|sonn_variant = \"-\"|wrong_checkpoint" \
      Point-Cache/runners/baseline/<runner>.py || echo "OK: no forbidden remnants"

---

### 13.6 第六步：先跑一个方法验证

不要一口气跑三个方法。先跑最基础的 Zero-shot：

    bash scripts/baseline/<exp>_zs_single_gpu.sh 0

跑完立刻检查：

    head -2 results/baseline/<exp>/summary.csv

确认 metadata 完全正确后，再跑后续方法。

---

## 14. 结果有效性的最终判定标准

以后判断一个实验是否可以归档，必须同时满足以下条件。

### 14.1 clean 单文件实验

| 检查项 | 标准 |
|---|---|
| summary 行数 | 1 |
| log_path 唯一数 | 1 |
| logs 文件数 | 1 |
| status | 1 done |
| metadata | 正确 |
| checkpoint | 正确 |
| accuracy | 与原文或预期对齐 |

### 14.2 all35 实验

| 检查项 | 标准 |
|---|---|
| summary 行数 | 35 |
| cor_type 唯一数 | 35 |
| log_path 唯一数 | 35 |
| logs 文件数 | 35 |
| status | 35 done |
| metadata | 正确 |
| checkpoint | 正确 |
| severity=2 | 与原文对齐 |
| all35 | 作为扩展统计 |
| corruption-wise | 分析困难类型 |
| severity-wise | 分析退化趋势 |

---

## 15. 对 MCM-PC 方法设计的启发

本轮 baseline 复现不仅完成了原文结果对齐，也暴露了后续方法实验的重点。

### 15.1 Global Cache 是稳定主模块

在 34 组中：

| 比较 | S2 Gain | all35 Gain |
|---|---:|---:|
| Global - ZS | +4.56 | +4.32 |

Global Cache 对所有 corruption 均为正增益。

因此后续 MCM-PC 不应轻易削弱 Global Cache，而应考虑在其基础上增加可靠性判断。

### 15.2 Local Cache 有额外贡献，但不是所有 corruption 都稳定正向

34 组中：

| 比较 | S2 Gain | all35 Gain |
|---|---:|---:|
| Local extra over Global | +1.82 | +1.20 |

但 corruption-wise 观察显示：

| corruption | Local extra |
|---|---:|
| dropout_local | +2.74 |
| jitter | +1.87 |
| dropout_global | +1.51 |
| add_local | -0.50 |

说明 Local Cache 对 dropout / jitter 有价值，但对 add_local 可能不稳定。

后续 MCM-PC 可考虑：

    reliability-aware local cache
    corruption-aware local cache weighting
    global-local consistency filtering
    negative / suppression mechanism
    pseudo-label risk control

### 15.3 后续重点困难 setting

完整 Point-Cache 后最低 setting：

| cor_type | Accuracy |
|---|---:|
| jitter_4 | 25.29 |
| jitter_3 | 29.32 |
| dropout_local_4 | 30.08 |
| dropout_global_4 | 34.04 |
| dropout_local_3 | 34.42 |
| jitter_2 | 35.60 |

这些 setting 是后续 MCM-PC 的优先观察目标。

---

## 16. 最终总结

本轮 baseline 复现中，最重要的问题不是某一次脚本报错，而是实验工程管理流程暴露出不足。

核心教训如下：

1. checkpoint 必须按数据集匹配，尤其是 Uni3D；
2. all35 runner 复制时必须检查 Python 内部字段；
3. shell 脚本正确不代表 runner 内部正确；
4. loader root 与 summary root 可能不同；
5. summary metadata 正确性与 accuracy 正确性同等重要；
6. 旧 summary 与服务器当前 summary 必须区分；
7. fail-fast 检查必须写入 runner；
8. 不能一口气跑完整组，必须先用一个方法验证 metadata；
9. 文档必须同时记录 severity=2 和 all35；
10. Git 提交必须拆分脚本、文档、问题总结，不提交 results 和 weights。

34 组最终已经修正并完成，整个 Point-Cache baseline 大复现阶段也已经完成。后续 MCM-PC 方法实验应基于当前干净 baseline，并严格沿用本文档中的检查流程。

