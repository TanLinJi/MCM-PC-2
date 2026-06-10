#!/usr/bin/env python
import argparse
import csv
import html
import json
import os
import subprocess
import sys
from pathlib import Path


POINT_CACHE_ROOT = Path(__file__).resolve().parents[3]

CORRUPTIONS = [
    "add_global",
    "add_local",
    "dropout_global",
    "dropout_local",
    "rotate",
    "scale",
    "jitter",
]
SEVERITIES = [0, 1, 2, 3, 4]
DEFAULT_EXP_ID = (
    "02_9_2_all35_ulip_modelnetc_zs_global_local_"
    "e4_c_a0_e1_textdist_only_tw0p15_score_norm_dual_t4"
)
METHOD_FULL = (
    "02-9-2 all35: E4-C-A0+E1-textdist-only on ModelNet-C; "
    "manual_full final classifier/logits, E1 cached descriptions only for text distribution, "
    "text_weight=0.15, running_zscore score normalization"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the current best 02-9-2 setting on all 35 ModelNet-C corruptions with independent workers."
    )
    parser.add_argument("--exp-id", default=DEFAULT_EXP_ID)
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU ids. Default: 0,1")
    parser.add_argument("--result-root", default="results/E4_distribution_guided_cache")
    parser.add_argument("--text-weight", default="0.15")
    parser.add_argument("--print-freq", default="500")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wandb", action="store_true", help="Do not pass --wandb-log to workers.")
    return parser.parse_args()


def prompt_cache_is_complete(cache_path, class_file, required_prompt_count=10):
    if not cache_path.exists():
        raise FileNotFoundError(f"shared E1 LLM prompt cache not found: {cache_path}")
    if not class_file.exists():
        raise FileNotFoundError(f"ModelNet-C class file not found: {class_file}")

    with cache_path.open("r", encoding="utf-8") as f:
        saved = json.load(f)

    prompts = saved.get("prompts", saved)
    if not isinstance(prompts, dict):
        raise RuntimeError(f"invalid prompt cache format: {cache_path}")

    with class_file.open("r", encoding="utf-8") as f:
        classnames = [line.strip() for line in f if line.strip()]

    missing = []
    short = []
    for classname in classnames:
        clean_name = classname.replace("_", " ")
        class_prompts = prompts.get(clean_name)
        if class_prompts is None:
            missing.append(clean_name)
        elif len(class_prompts) < required_prompt_count:
            short.append((clean_name, len(class_prompts)))

    if missing or short:
        lines = ["shared E1 prompt cache is incomplete; stop to avoid changing the 02-9-2 setting."]
        if missing:
            lines.append("Missing classes: " + ", ".join(missing))
        if short:
            lines.append("Classes with too few prompts: " + ", ".join(f"{name}:{count}" for name, count in short))
        raise RuntimeError("\n".join(lines))

    return len(classnames)


def build_tasks():
    return [
        {"corruption": corruption, "severity": severity}
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
    ]


def split_tasks(tasks, worker_count):
    chunks = [[] for _ in range(worker_count)]
    for index, task in enumerate(tasks):
        chunks[index % worker_count].append(task)
    return chunks


def verify_data_files(pc_root, tasks):
    missing = []
    data_root = pc_root / "data" / "modelnet_c"
    for task in tasks:
        cor_type = f"{task['corruption']}_{task['severity']}"
        path = data_root / f"{cor_type}.h5"
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing ModelNet-C files:\n" + "\n".join(missing))


def write_tasks(run_dir, task_chunks):
    task_files = []
    for worker_id, tasks in enumerate(task_chunks):
        path = run_dir / f"tasks_worker{worker_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
        task_files.append(path)
    return task_files


def build_worker_command(args, worker_id, gpu_id, task_file):
    worker_script = Path(__file__).resolve().parent / "worker_02_9_2_modelnetc_all35.py"
    cmd = [
        args.python,
        str(worker_script),
        "--worker-id",
        str(worker_id),
        "--tasks-json",
        str(task_file),
        "--baseline-exp-id",
        args.exp_id,
        "--baseline-method",
        "zs_global_local",
        "--baseline-method-full",
        METHOD_FULL,
        "--baseline-gpu",
        str(gpu_id),
        "--baseline-result-root",
        args.result_root,
        "--config",
        "configs",
        "--lm3d",
        "ulip",
        "--cache-type",
        "hierarchical",
        "--prompt-source",
        "manual_full",
        "--prompt-cache-dir",
        "results/E1_text_prototype_enhancement/shared_prompts",
        "--llm-provider",
        "deepseek",
        "--llm-model",
        "deepseek-v4-pro",
        "--llm-api-key-file",
        "llm/secrets/llm_api_key.txt",
        "--llm-api-base-url",
        "https://api.deepseek.com/chat/completions",
        "--llm-temperature",
        "0.3",
        "--llm-prompt-mode",
        "multiview_2d3d",
        "--dynamic-prompt-count",
        "10",
        "--prompt-static-weight",
        "0.75",
        "--prompt-dynamic-weight",
        "0.25",
        "--ckpt_path",
        "weights/ulip/pointbert_ulip1.pt",
        "--slip-ckpt-path",
        "weights/ulip/slip_base_100ep.pt",
        "--dataset",
        "modelnet_c",
        "--sonn_variant",
        "hardest",
        "--cor_type",
        "add_global_0",
        "--npoints",
        "1024",
        "--sim2real_type",
        "so_obj_only_9",
        "--oshape-version",
        "vitg14",
        "--ulip-version",
        "ulip1",
        "--device",
        "0",
        "--print-freq",
        str(args.print_freq),
    ]
    if not args.no_wandb:
        cmd.append("--wandb-log")
    return cmd


def build_worker_env(args, run_dir, worker_id, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["WANDB_MODE"] = "offline"
    env["WANDB_DIR"] = str(run_dir / f"wandb_worker{worker_id}")
    env["WANDB_SILENT"] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    env["GPA_SAVE_STATS"] = env.get("GPA_SAVE_STATS", "1")
    env["E4_DIST_EPS"] = env.get("E4_DIST_EPS", "1e-4")
    env["E4_DIST_MIN_VAR"] = env.get("E4_DIST_MIN_VAR", "1e-4")
    env["E4_TEXT_DIST_EPS"] = env.get("E4_TEXT_DIST_EPS", env["E4_DIST_EPS"])
    env["E4_TEXT_DIST_MIN_VAR"] = env.get("E4_TEXT_DIST_MIN_VAR", env["E4_DIST_MIN_VAR"])
    env["E4_SCORE_NORM_MODE"] = env.get("E4_SCORE_NORM_MODE", "running_zscore")
    env["E4_SCORE_NORM_MIN_COUNT"] = env.get("E4_SCORE_NORM_MIN_COUNT", "8")
    env["E4_SCORE_NORM_EPS"] = env.get("E4_SCORE_NORM_EPS", "1e-6")
    env["E4_SCORE_NORM_CLIP"] = env.get("E4_SCORE_NORM_CLIP", "0")
    env["E4_TEXT_DIST_PROMPT_SOURCE"] = env.get("E4_TEXT_DIST_PROMPT_SOURCE", "manualfull_llm_dynamic_init")
    env["E4_TEXT_SCORE_WEIGHT"] = str(args.text_weight)
    return env


def run_workers(args, run_dir, task_files, gpu_ids):
    processes = []
    log_handles = []

    for worker_id, (gpu_id, task_file) in enumerate(zip(gpu_ids, task_files)):
        cmd = build_worker_command(args, worker_id, gpu_id, task_file)
        env = build_worker_env(args, run_dir, worker_id, gpu_id)
        stdout_log = run_dir / f"worker{worker_id}_stdout.log"
        handle = stdout_log.open("w", encoding="utf-8")
        log_handles.append(handle)

        print(f"[launcher] worker{worker_id}: physical GPU {gpu_id}, tasks={task_file}")
        print(f"[launcher] worker{worker_id} stdout: {stdout_log}")
        processes.append((
            worker_id,
            subprocess.Popen(
                cmd,
                cwd=str(POINT_CACHE_ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            ),
            stdout_log,
        ))

    failed = []
    for worker_id, process, stdout_log in processes:
        returncode = process.wait()
        if returncode != 0:
            failed.append((worker_id, returncode, stdout_log))

    for handle in log_handles:
        handle.close()

    if failed:
        messages = []
        for worker_id, returncode, stdout_log in failed:
            messages.append(f"worker{worker_id} failed with code {returncode}. log: {stdout_log}")
            try:
                tail = stdout_log.read_text(encoding="utf-8").splitlines()[-80:]
                messages.extend("  " + line for line in tail)
            except Exception:
                pass
        raise RuntimeError("\n".join(messages))


def read_worker_rows(run_dir):
    rows = []
    for path in sorted(run_dir.glob("summary_worker*.csv")):
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
    rows.sort(key=lambda row: (CORRUPTIONS.index(row["corruption"]), int(row["severity"])))
    return rows


def write_summary_csv(run_dir, rows):
    summary_file = run_dir / "summary.csv"
    if not rows:
        return summary_file

    fieldnames = list(rows[0].keys())
    with summary_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_file


def rounded_mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return round(sum(values) / len(values) + 1e-8, 2)


def build_table(rows):
    acc = {}
    status = {}
    for row in rows:
        key = (row["corruption"], int(row["severity"]))
        status[key] = row["status"]
        if row["status"] == "done" and row["acc"]:
            acc[key] = float(row["acc"])
        else:
            acc[key] = None

    table = []
    for corruption in CORRUPTIONS:
        values = [acc.get((corruption, severity)) for severity in SEVERITIES]
        table.append({
            "corruption": corruption,
            "values": values,
            "avg": rounded_mean(values),
        })

    severity_avgs = [
        rounded_mean([acc.get((corruption, severity)) for corruption in CORRUPTIONS])
        for severity in SEVERITIES
    ]
    overall_avg = rounded_mean([
        acc.get((corruption, severity))
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
    ])

    return table, severity_avgs, overall_avg, status


def format_cell(value):
    if value is None:
        return ""
    return f"{value:.2f}"


def write_table_csv(run_dir, table, severity_avgs, overall_avg):
    path = run_dir / "all35_table.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Corruption", "S0", "S1", "S2", "S3", "S4", "Avg(S0-S4)"])
        for row in table:
            writer.writerow([row["corruption"]] + [format_cell(v) for v in row["values"]] + [format_cell(row["avg"])])
        writer.writerow(["Average"] + [format_cell(v) for v in severity_avgs] + [format_cell(overall_avg)])
    return path


def write_table_md(run_dir, table, severity_avgs, overall_avg):
    path = run_dir / "all35_table.md"
    lines = [
        "| Corruption | S0 | S1 | S2 | S3 | S4 | Avg(S0-S4) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            "| "
            + " | ".join([row["corruption"]] + [format_cell(v) for v in row["values"]] + [format_cell(row["avg"])])
            + " |"
        )
    lines.append(
        "| "
        + " | ".join(["Average"] + [format_cell(v) for v in severity_avgs] + [format_cell(overall_avg)])
        + " |"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_table_html(run_dir, table, severity_avgs, overall_avg):
    path = run_dir / "all35_table.html"
    rows = []
    for row in table:
        cells = [html.escape(row["corruption"])] + [format_cell(v) for v in row["values"]] + [format_cell(row["avg"])]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")

    avg_cells = ["Average"] + [format_cell(v) for v in severity_avgs] + [format_cell(overall_avg)]
    rows.append("<tr class=\"avg\">" + "".join(f"<td>{cell}</td>" for cell in avg_cells) + "</tr>")

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>02-9-2 ModelNet-C All35 Results</title>
  <style>
    body {{
      margin: 0;
      background: #1f1f1f;
      color: #d0d0d0;
      font-family: Arial, Helvetica, sans-serif;
    }}
    main {{
      max-width: 980px;
      margin: 32px auto;
      padding: 0 20px;
    }}
    h1 {{
      margin: 0 0 18px;
      color: #eeeeee;
      font-size: 24px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #202020;
      font-size: 20px;
      line-height: 1.45;
    }}
    th, td {{
      padding: 10px 16px;
      border-bottom: 1px solid #454545;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: #d6d6d6;
      font-weight: 700;
      border-bottom-color: #8a8a8a;
    }}
    tr.avg td {{
      font-weight: 700;
      color: #d8d8d8;
      border-bottom: 0;
    }}
    .meta {{
      margin: 0 0 16px;
      color: #a8a8a8;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>02-9-2 ModelNet-C All35 Results</h1>
    <p class="meta">E4-C-A0+E1-textdist-only, text weight 0.15, running z-score normalization.</p>
    <table>
      <thead>
        <tr>
          <th>Corruption</th>
          <th>S0</th>
          <th>S1</th>
          <th>S2</th>
          <th>S3</th>
          <th>S4</th>
          <th>Avg(S0-S4)</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")
    return path


def merge_and_write_tables(run_dir):
    rows = read_worker_rows(run_dir)
    summary_file = write_summary_csv(run_dir, rows)
    table, severity_avgs, overall_avg, _status = build_table(rows)
    csv_table = write_table_csv(run_dir, table, severity_avgs, overall_avg)
    md_table = write_table_md(run_dir, table, severity_avgs, overall_avg)
    html_table = write_table_html(run_dir, table, severity_avgs, overall_avg)
    return summary_file, csv_table, md_table, html_table


def main():
    args = parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise SystemExit("No GPU id was provided.")

    pc_root = POINT_CACHE_ROOT
    run_dir = pc_root / args.result_root / args.exp_id

    prompt_cache = pc_root / "results" / "E1_text_prototype_enhancement" / "shared_prompts" / "modelnet_c_deepseek_deepseek-v4-pro_multiview_2d3d_10_prompts.json"
    class_file = pc_root / "data" / "modelnet_c" / "shape_names.txt"
    class_count = prompt_cache_is_complete(prompt_cache, class_file)

    tasks = build_tasks()
    verify_data_files(pc_root, tasks)
    task_chunks = split_tasks(tasks, len(gpu_ids))

    print("============================================================")
    print("02-9-2 ModelNet-C all35 dual-GPU launcher")
    print(f"Point-Cache root: {pc_root}")
    print(f"EXP_ID: {args.exp_id}")
    print(f"Result dir: {run_dir}")
    print(f"Physical GPUs: {', '.join(gpu_ids)}")
    print(f"Prompt cache verified: {class_count} classes, 10 prompts per class")
    print(f"Task split: {', '.join(str(len(chunk)) for chunk in task_chunks)}")
    print("Execution model: one independent worker process per GPU; each worker loads the model once.")
    print("============================================================")

    if args.dry_run:
        for worker_id, gpu_id in enumerate(gpu_ids):
            task_file = run_dir / f"tasks_worker{worker_id}.json"
            print(f"[dry-run] worker{worker_id}, gpu={gpu_id}, tasks={task_file}")
            print(" ".join(build_worker_command(args, worker_id, gpu_id, task_file)))
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    task_files = write_tasks(run_dir, task_chunks)
    run_workers(args, run_dir, task_files, gpu_ids)
    summary_file, csv_table, md_table, html_table = merge_and_write_tables(run_dir)

    print("============================================================")
    print("All workers finished. Merged result files:")
    print(f"summary: {summary_file}")
    print(f"csv table: {csv_table}")
    print(f"markdown table: {md_table}")
    print(f"html table: {html_table}")
    print("============================================================")
    print(md_table.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
