#!/usr/bin/env python
"""
Benchmark orchestrator.

Spawns one subprocess per (method, label_rate, seed) combo via bench_eval.py.
- stdout/stderr of each run is tee'd to benchmark_results/logs/<run_id>.log
- per-run metrics dumped to benchmark_results/results/<run_id>.json
- aggregated table written to benchmark_results/results/summary.csv

Edit RUNS below to control which configurations are evaluated. The defaults
benchmark the five-line story from the paper discussion:
  supervised baselines (DCNN, DeepSense, R-GRU) vs LIMU-BERT-X-pretrained GRU.
"""
import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
# Per-run output dirs (logs/results/plots) and the saved/ checkpoint subfolder are
# derived from --run_id inside main(); nothing is created at import time.


# ---------------------------------------------------------------------------
# CONFIG MATRIX
# ---------------------------------------------------------------------------
# Each run is a dict of kwargs forwarded to bench_eval.py. Add or remove freely.
# "tag" is a human-readable identifier used in the CSV / plots.
#
# Available "method" values (see models.py:fetch_classifier):
#   supervised:  gru, dcnn, deepsense, attn, cnn2, cnn1, lstm
#   bert:        base_gru, base_cnn, base_attn, base_lstm
# ---------------------------------------------------------------------------

DATASET = "jetson_leg"
DATASET_VERSION = "10_20_both_xyz_pocket"
MODEL_VERSION = "v3"
TRAINING_RATE = 0.8
SEEDS = [3431, 42, 2026]
DATASET = "jetson_leg"
DATASET_VERSION = "10_20_both_0103_xyz_both"
MODEL_VERSION = "v3"
TRAINING_RATE = 0.8
SEEDS = [3431, 42, 2026]
# LABEL_RATES = [0.002 , 0.005, 0.01, 0.02, 0.1, 0.2]   # paper standard
# LABEL_RATES = [0.01 , 0.02, 0.04, 0.05, 0.1, 0.2]     # 2x single subject
# LABEL_RATES = [0.02, 0.05, 0.08, 0.1, 0.2, 0.3]       # single subject
LABEL_RATES = [0.005 , 0.01, 0.02, 0.03, 0.05, 0.1]     # 4x single subject

# Which label column to predict. 0 = activity for camargo (see
# dataset/data_config.json:86). Do NOT use -1 here: in some configs that index
# matches a "_label_index: -1" sentinel and yields label_num=0 → CUDA assert.
LABEL_INDEX = 0

# Balanced label sampling for the labeled TRAIN pool. When 1, balance=1 means the
# labeled subsample is balanced by activity class (per-class budget), and that budget
# is spread as evenly as possible across subjects on multi-subject datasets (e.g. a
# 7-window class budget over 2 subjects -> 4 + 3). Single-subject configs fall back to
# class-only balance automatically (see utils.prepare_classifier_dataset).
# Per-run override: add "balance": 0/1 to a RUNS entry.
BALANCE = 1

# Path to the LIMU-BERT-X foundation-model checkpoint to use.
# Adjust if you want a different pretrained file.
LIMU_BERTX_CKPT = os.path.join(
    "saved", "pretrain_base_" + DATASET + "_" + DATASET_VERSION, "limu_bert_x_9cls_dapt_5e-4_3200_seed3431.pt"
)

# model_version convention:
#   supervised mode      → "<classifier_version>"            e.g. "v1", "v2"
#   bert / bert_separated → "<bert_version>_<classifier_version>"  e.g. "v3_v1"
# The classifier_version part picks an entry in config/classifier.json
# (gru_v1 / gru_v2 / dcnn_v1 / deepsense_v1 / ...). To swap a head per-row,
# just edit the string below — every entry has its own model_version field.
RUNS = [
    # --- Supervised baselines (no pretraining) ---
    # {"tag": "DCNN",        "mode": "supervised", "method": "dcnn",      "model_version": "v1"},
    # {"tag": "DeepSense",   "mode": "supervised", "method": "deepsense", "model_version": "v1"},
    # {"tag": "R-GRU",       "mode": "supervised", "method": "gru",       "model_version": "v3"},
    # --- LIMU-BERT-X foundation model + GRU head ---
    # bert_version pinned to v3 so joint runs match the separated path and
    # inference/test_csv.py. classifier_version swappable (v1 = paper-ish,
    # v2 = with dropout).
    # {"tag": "LIMU-BERT-X+GRU (frozen)",
    #  "mode": "bert", "method": "base_gru", "model_version": "v3_v3",
    #  "pretrain_model": LIMU_BERTX_CKPT, "frozen_bert": 1},
    {"tag": "LIMU-BERT-X+GRU (finetune)",
     "mode": "bert", "method": "base_gru", "model_version": "v3_v3",
     "pretrain_model": LIMU_BERTX_CKPT, "frozen_bert": 0},
    # Same finetune path, but effective lr = 1e-4 * lr_scale = 1e-3.
    {"tag": "LIMU-BERT-X+GRU (finetune-high-lr)",
     "mode": "bert", "method": "base_gru", "model_version": "v3_v3",
     "pretrain_model": LIMU_BERTX_CKPT, "frozen_bert": 0, "lr_scale": 10},
    # Separated mode: BERT runs in eval/no_grad as a frozen feature extractor
    # (same as inference/test_csv.py), embeddings are cached in memory, then a
    # standalone GRU head is trained via classifier.classify_embeddings.
    {"tag": "LIMU-BERT-X+GRU (separated)",
     "mode": "bert_separated", "method": "gru", "model_version": "v3",
     "pretrain_model": LIMU_BERTX_CKPT},
]


def run_id(tag, label_rate, seed):
    safe = tag.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
    return f"{safe}__lr{label_rate}__seed{seed}"


def run_one(run_cfg, label_rate, seed, log_dir, result_dir, save_dir_name,
            gpu=None, dry=False):
    rid = run_id(run_cfg["tag"], label_rate, seed)
    log_path = os.path.join(log_dir, rid + ".log")
    json_path = os.path.join(result_dir, rid + ".json")

    cmd = [
        sys.executable, os.path.join(HERE, "bench_eval.py"),
        "--mode", run_cfg["mode"],
        "--method", run_cfg["method"],
        "--model_version", run_cfg.get("model_version", MODEL_VERSION),
        "--dataset", run_cfg.get("dataset", DATASET),
        "--dataset_version", run_cfg.get("dataset_version", DATASET_VERSION),
        "--label_rate", str(label_rate),
        "--training_rate", str(run_cfg.get("training_rate", TRAINING_RATE)),
        "--seed", str(seed),
        "--save_model", rid,
        "--save_dir", save_dir_name,
        "--out_json", json_path,
        "--balance", str(run_cfg.get("balance", BALANCE)),
        "--label_index", str(run_cfg.get("label_index", LABEL_INDEX)),
    ]
    if run_cfg.get("pretrain_model"):
        cmd += ["--pretrain_model", run_cfg["pretrain_model"]]
    if run_cfg["mode"] == "bert":
        cmd += ["--frozen_bert", str(run_cfg.get("frozen_bert", 1))]
    # Optional LR schedule overrides (default = no warmup, no cosine).
    if "warmup_epochs" in run_cfg:
        cmd += ["--warmup_epochs", str(run_cfg["warmup_epochs"])]
    if "cosine_decay" in run_cfg:
        cmd += ["--cosine_decay", str(int(bool(run_cfg["cosine_decay"])))]
    if "cosine_eta_min" in run_cfg:
        cmd += ["--cosine_eta_min", str(run_cfg["cosine_eta_min"])]
    if "early_stop_patience" in run_cfg:
        cmd += ["--early_stop_patience", str(run_cfg["early_stop_patience"])]
    if "lr_scale" in run_cfg:
        cmd += ["--lr_scale", str(run_cfg["lr_scale"])]
    if gpu is not None:
        cmd += ["--gpu", gpu]

    header = (
        f"=== {rid} ===\n"
        f"start: {dt.datetime.now().isoformat()}\n"
        f"cmd:   {' '.join(cmd)}\n"
        "----------------------------------------------------------------\n"
    )
    print(header, end="", flush=True)
    if dry:
        return None

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(header)
        logf.flush()
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        ret = proc.wait()
        footer = f"----------------------------------------------------------------\nreturncode: {ret}\nend: {dt.datetime.now().isoformat()}\n"
        logf.write(footer)
    print(footer, end="")

    if ret != 0 or not os.path.exists(json_path):
        return {"tag": run_cfg["tag"], "label_rate": label_rate, "seed": seed,
                "acc": None, "f1": None, "status": f"failed(rc={ret})",
                "log": log_path}

    with open(json_path, "r") as f:
        result = json.load(f)
    return {
        "tag": run_cfg["tag"],
        "method": result.get("method"),
        "mode": result.get("mode"),
        "pretrain_model": result.get("pretrain_model"),
        "frozen_bert": result.get("frozen_bert"),
        "recipe": result.get("recipe"),
        "dataset": result.get("dataset"),
        "dataset_version": result.get("dataset_version"),
        "label_rate": label_rate,
        "seed": seed,
        "acc": result.get("acc"),
        "f1": result.get("f1"),
        "status": "ok",
        "log": log_path,
        "json": json_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True,
                    help="Manual Run ID naming this benchmark (e.g. 'Run51_20260706_test'). "
                         "Outputs go to benchmark_results/history/<run_id>/{logs,results,plots} "
                         "and checkpoints to saved/bench_<run_id>/. Must not contain path separators.")
    ap.add_argument("--no_plot", action="store_true",
                    help="Skip auto-generating plots at the end of the sweep.")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--dry", action="store_true", help="Print the commands without running.")
    ap.add_argument("--only", default=None,
                    help="Comma-separated substrings; only runs whose tag matches at least one will execute.")
    ap.add_argument("--label_rates", default=None,
                    help="Comma-separated override (e.g. '0.01,0.1,1.0').")
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated override (e.g. '3431,42').")
    ap.add_argument("--model_version", default=None,
                    help="Override MODEL_VERSION for every run in this invocation. "
                         "BERT mode wants <bert_v>_<classifier_v>, e.g. 'v3_v1'.")
    args = ap.parse_args()

    run_id_name = args.run_id.strip()
    if not run_id_name or os.path.sep in run_id_name or "/" in run_id_name:
        raise SystemExit(f"--run_id must be a non-empty folder name without path separators, got {args.run_id!r}")
    run_dir = os.path.join(HERE, "history", run_id_name)
    log_dir = os.path.join(run_dir, "logs")
    result_dir = os.path.join(run_dir, "results")
    plot_dir = os.path.join(run_dir, "plots")
    for d in (log_dir, result_dir, plot_dir):
        os.makedirs(d, exist_ok=True)
    save_dir_name = "bench_" + run_id_name

    label_rates = LABEL_RATES if args.label_rates is None else [float(x) for x in args.label_rates.split(",")]
    seeds = SEEDS if args.seeds is None else [int(x) for x in args.seeds.split(",")]
    runs = RUNS
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        runs = [r for r in RUNS if any(k in r["tag"] for k in keys)]
    if args.model_version is not None:
        runs = [{**r, "model_version": args.model_version} for r in runs]

    summary_path = os.path.join(result_dir, "summary.csv")
    write_header = not os.path.exists(summary_path)
    if not write_header:
        print(f"WARNING: {summary_path} already exists — run_id {run_id_name!r} was used "
              f"before; new rows will be APPENDED to the existing summary.csv.")
    fieldnames = ["tag", "method", "mode", "pretrain_model", "frozen_bert", "recipe",
                  "dataset", "dataset_version", "label_rate", "seed", "acc", "f1", "status",
                  "log", "json"]
    with open(summary_path, "a", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for run_cfg in runs:
            for lr in label_rates:
                for seed in seeds:
                    row = run_one(run_cfg, lr, seed, log_dir, result_dir, save_dir_name,
                                  gpu=args.gpu, dry=args.dry)
                    if row is None:
                        continue
                    writer.writerow(row)
                    csvf.flush()
    print(f"\nSummary CSV: {summary_path}")
    print(f"Run dir:     {run_dir}")

    if not args.dry and not args.no_plot:
        try:
            subprocess.run([sys.executable, os.path.join(HERE, "plot_benchmark.py"),
                            "--run_dir", run_dir], check=True)
        except Exception as e:  # plotting must never fail a completed sweep
            print(f"WARNING: plot_benchmark.py failed ({e}); results are still in {result_dir}.")


if __name__ == "__main__":
    main()
