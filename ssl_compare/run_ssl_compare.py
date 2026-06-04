#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single merged-WARMSTART orchestrator.

ONE warmstart run on the foundation model (limu_bert_x): foundation init +
FULL lr (1e-3) + FULL epochs (1200) of in-domain SSL on the four datasets
MERGED into a single pretraining pool, then that ONE checkpoint is evaluated
ONLY on camargo 8cls:

  warmstart   foundation init, aggressive full lr/epochs re-training
              pretrained on camargo + molinaro + scherpereel + scherpereel_exo at once.
              (Contrast with DAPT's gentle lr=1e-4/epochs=300 short adaptation.)

Each dataset's last-10% test split is held out of the merge (same seed as
bench_eval), so camargo's test split is never seen during warmstart. A
supervised R-GRU row (no pretraining) is added as the yardstick -- the whole
question is whether the merged-warmstart ckpt beats it on camargo.

Pipeline:
  Phase 1 (pretrain): one call to ssl_compare/pretrain_ssl.py --mode warmstart with
                      --merge listing all four datasets. Emits one per-seed checkpoint:
                      saved/pretrain_base_<host>/warmstart_merged4_seed<seed>.pt
  Phase 2 (eval):     for each (row, label_rate, seed) on camargo, call
                      benchmark_results/bench_eval.py. The SSL rows point at the
                      single merged checkpoint.
  Phase 3 (report):   aggregate F1 to mean +/- std per (row, label_rate).

Run from the repo root:

  python ssl_compare/run_ssl_compare.py --gpu 0
  python ssl_compare/run_ssl_compare.py --dry                 # print commands only
  python ssl_compare/run_ssl_compare.py --skip_pretrain       # reuse existing ckpt
"""
import argparse
import csv
import datetime as dt
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
BENCH_EVAL = os.path.join(REPO_ROOT, "benchmark_results", "bench_eval.py")
PRETRAIN_SSL = os.path.join(HERE, "pretrain_ssl.py")
LOG_DIR = os.path.join(HERE, "logs")
RESULT_DIR = os.path.join(HERE, "results")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_VERSION = "v3"                       # BERT config version (base_v3)
TRAINING_RATE = 0.8
SEEDS = [3431]   # single fixed seed: pretrain and eval both use it -> aligned split, no leakage
LABEL_RATES = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
AUGMENT = 0                                # rotation+noise, held constant
BATCH_SIZE = 128                           # pretrain batch (config default is 128); bigger = faster

# Datasets MERGED into the single warmstart pretraining pool. (dataset, version).
# All four are concatenated (foundation init + full lr/epochs in-domain SSL), but
# the resulting ckpt is only EVALUATED on camargo below.
MERGE_DATASETS = [
    ("camargo",         "10_20_dense_8cls"),
    ("molinaro",        "10_20_both"),
    ("scherpereel",     "10_20_both"),
    ("scherpereel_exo", "10_20_both"),
]
MERGE_SPEC = ",".join("%s:%s" % (d, v) for d, v in MERGE_DATASETS)

# Datasets the merged warmstart ckpt is evaluated on. (dataset, version, label_index).
# Only camargo 8cls per the experiment scope.
EVAL_DATASETS = [
    ("camargo", "10_20_dense_8cls", 0),
]

# The merged ckpt borrows one dataset's dir for save location + model config
# (model dims are dataset-independent: all are (N, 20, 6) -> base_v3 feature_num=6).
HOST_DATASET = "camargo"
HOST_VERSION = "10_20_dense_8cls"
MERGE_TAG = "warmstart_merged4"   # -> saved/pretrain_base_<host>/warmstart_merged4_seed<seed>.pt
HOST_DIR = os.path.join("saved", "pretrain_base_%s_%s" % (HOST_DATASET, HOST_VERSION))
FOUNDATION_CKPT = os.path.join(HOST_DIR, "limu_bert_x")   # no .pt; loaders re-append it

# Warmstart recipe (foundation init, FULL lr + FULL epochs -> aggressive re-training,
# unlike DAPT's gentle lr=1e-4/epochs=300). Matches pretrain_ssl.py MODE_DEFAULTS["warmstart"].
WARMSTART = {"lr": 1e-3, "epochs": 400}

# Downstream evaluation paths for the warmstart ckpt.
#   "bert_separated" = frozen BERT feature extractor + standalone GRU head (probe).
#                      tag = "warmstart".
#   "bert"           = joint finetune (BERT + head co-trained). tag = "warmstart_ft".
EVAL_MODES = ["bert_separated", "bert"]

# Supervised yardstick (no pretraining). Set to None to drop it.
SUPERVISED_REF = {"tag": "R-GRU", "method": "gru", "model_version": "v3"}


def merged_ckpt_stem(seed):
    """Per-seed merged-DAPT checkpoint path (no .pt); bench_eval re-appends .pt."""
    return os.path.join(HOST_DIR, "%s_seed%d" % (MERGE_TAG, seed))


def stream_subprocess(cmd, log_path, dry):
    header = (
        "=== %s ===\nstart: %s\ncmd:   %s\n%s\n"
        % (os.path.basename(log_path), dt.datetime.now().isoformat(),
           " ".join(cmd), "-" * 64)
    )
    print(header, flush=True)
    if dry:
        return 0
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(header + "\n")
        logf.flush()
        # Force the child to unbuffered stdout/stderr so per-epoch prints stream
        # out immediately. Without this, Python block-buffers stdout when it's a
        # pipe (not a tty) and the parent loop appears to hang.
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace", env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        ret = proc.wait()
        logf.write("\n%s\nreturncode: %d\nend: %s\n"
                   % ("-" * 64, ret, dt.datetime.now().isoformat()))
    return ret


# ---------------------------------------------------------------------------
# Phase 1: one merged DAPT pretrain across all datasets (all seeds)
# ---------------------------------------------------------------------------
def pretrain_phase(gpu, dry, skip_existing):
    seeds_arg = ",".join(str(s) for s in SEEDS)
    cmd = [
        sys.executable, "-u", PRETRAIN_SSL,
        MODEL_VERSION, HOST_DATASET, HOST_VERSION,
        "--mode", "warmstart",
        "--merge", MERGE_SPEC,
        "--out_name", MERGE_TAG,
        "--seeds", seeds_arg,
        "--training_rate", str(TRAINING_RATE),
        "--lr", repr(WARMSTART["lr"]),
        "--epochs", str(WARMSTART["epochs"]),
        "--batch_size", str(BATCH_SIZE),
        "--augment", str(AUGMENT),
        "-f", FOUNDATION_CKPT,
    ]
    if skip_existing:
        cmd += ["--skip_existing"]
    if gpu is not None:
        cmd += ["-g", gpu]
    log_path = os.path.join(LOG_DIR, "pretrain_%s.log" % MERGE_TAG)
    ret = stream_subprocess(cmd, log_path, dry)
    if ret != 0 and not dry:
        print("!! merged DAPT pretrain failed (rc=%d); see %s" % (ret, log_path))


# ---------------------------------------------------------------------------
# Phase 2: evaluate the merged ckpt on each dataset via bench_eval.py
# ---------------------------------------------------------------------------
def eval_one(dataset, version, label_index, tag, eval_cmd_extra,
             label_rate, seed, gpu, dry, skip_existing=False):
    rid = "%s__%s__%s__lr%s__seed%d" % (
        dataset, version, tag.replace(" ", "_"), label_rate, seed)
    log_path = os.path.join(LOG_DIR, "eval_" + rid + ".log")
    json_path = os.path.join(RESULT_DIR, "eval_" + rid + ".json")
    base = {"dataset": dataset, "dataset_version": version,
            "tag": tag, "label_rate": label_rate, "seed": seed}
    if skip_existing and os.path.exists(json_path):
        with open(json_path, "r") as f:
            r = json.load(f)
        print("=== skip existing eval: %s ===" % rid)
        return dict(base, acc=r.get("acc"), f1=r.get("f1"), status="ok")
    cmd = [
        sys.executable, "-u", BENCH_EVAL,
        "--dataset", dataset,
        "--dataset_version", version,
        "--label_rate", str(label_rate),
        "--training_rate", str(TRAINING_RATE),
        "--seed", str(seed),
        "--label_index", str(label_index),
        "--balance", "1",
        "--save_model", "sslcmp_" + rid,
        "--out_json", json_path,
    ] + eval_cmd_extra
    if gpu is not None:
        cmd += ["--gpu", gpu]

    ret = stream_subprocess(cmd, log_path, dry)
    if dry:
        return None
    if ret != 0 or not os.path.exists(json_path):
        return dict(base, acc=None, f1=None, status="failed(rc=%d)" % ret)
    with open(json_path, "r") as f:
        r = json.load(f)
    return dict(base, acc=r.get("acc"), f1=r.get("f1"), status="ok")


def build_eval_rows():
    """Rows are dataset-independent: every SSL row points at the single merged ckpt.

      bert_separated -> tag = "warmstart"     (frozen BERT + GRU probe)
      bert           -> tag = "warmstart_ft"  (joint finetune)
    Plus an optional supervised R-GRU yardstick.
    """
    rows = []
    for eval_mode in EVAL_MODES:
        if eval_mode == "bert_separated":
            rows.append(("warmstart", lambda seed: [
                "--mode", "bert_separated", "--method", "gru",
                "--model_version", MODEL_VERSION,
                "--pretrain_model", merged_ckpt_stem(seed),
            ]))
        elif eval_mode == "bert":
            rows.append(("warmstart_ft", lambda seed: [
                "--mode", "bert", "--method", "base_gru",
                "--model_version", MODEL_VERSION + "_" + MODEL_VERSION,
                "--pretrain_model", merged_ckpt_stem(seed), "--frozen_bert", "0",
            ]))
        else:
            sys.exit("Unknown EVAL_MODES entry: %s" % eval_mode)
    if SUPERVISED_REF is not None:
        ref = SUPERVISED_REF
        rows.append((ref["tag"], lambda seed: [
            "--mode", "supervised", "--method", ref["method"],
            "--model_version", ref["model_version"],
        ]))
    return rows


def eval_phase(label_rates, gpu, dry, skip_existing=False):
    rows = build_eval_rows()
    results = []
    for dataset, version, label_index in EVAL_DATASETS:
        for tag, extra_fn in rows:
            for lr in label_rates:
                for seed in SEEDS:
                    row = eval_one(dataset, version, label_index, tag, extra_fn(seed),
                                   lr, seed, gpu, dry, skip_existing=skip_existing)
                    if row is not None:
                        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Phase 3: aggregate + report
# ---------------------------------------------------------------------------
def aggregate(results, label_rates):
    summary_path = os.path.join(RESULT_DIR, "ssl_compare_warmstart_merged_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=[
            "dataset", "dataset_version", "tag", "label_rate", "seed", "acc", "f1", "status"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # one mean +/- std F1 table per dataset
    datasets = []
    for r in results:
        key = (r["dataset"], r["dataset_version"])
        if key not in datasets:
            datasets.append(key)

    for dataset, version in datasets:
        sub = [r for r in results if r["dataset"] == dataset and r["dataset_version"] == version]
        tags = []
        for r in sub:
            if r["tag"] not in tags:
                tags.append(r["tag"])
        print("\n======== %s_%s : F1 (mean +/- std over seeds) ========" % (dataset, version))
        print("%-12s" % "label_rate" + "".join("%-22s" % t for t in tags))
        for lr in label_rates:
            cells = []
            for t in tags:
                f1s = [x["f1"] for x in sub
                       if x["tag"] == t and x["label_rate"] == lr and x["f1"] is not None]
                if f1s:
                    mean = sum(f1s) / len(f1s)
                    std = math.sqrt(sum((v - mean) ** 2 for v in f1s) / len(f1s))
                    cells.append("%-22s" % ("%.3f +/- %.3f" % (mean, std)))
                else:
                    cells.append("%-22s" % "-")
            print("%-12s" % str(lr) + "".join(cells))
    print("\nSummary CSV: %s" % summary_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--dry", action="store_true", help="Print commands without running.")
    ap.add_argument("--skip_pretrain", action="store_true", help="Reuse existing merged checkpoint.")
    ap.add_argument("--skip_eval", action="store_true", help="Only run the pretrain phase.")
    ap.add_argument("--skip_existing_ckpt", action="store_true",
                    help="During pretrain, skip seeds whose ckpt already exists.")
    ap.add_argument("--skip_existing_eval", action="store_true",
                    help="During eval, reuse rows whose output JSON already exists.")
    ap.add_argument("--label_rates", default=None, help="Comma-separated override.")
    args = ap.parse_args()

    label_rates = LABEL_RATES if args.label_rates is None else [float(x) for x in args.label_rates.split(",")]

    if not args.skip_pretrain:
        pretrain_phase(args.gpu, args.dry, args.skip_existing_ckpt)
    else:
        print("Skipping pretrain phase (reusing existing merged checkpoint).")

    if args.skip_eval:
        print("Skipping eval phase.")
        return

    results = eval_phase(label_rates, args.gpu, args.dry,
                         skip_existing=args.skip_existing_eval)
    if not args.dry and results:
        aggregate(results, label_rates)


if __name__ == "__main__":
    main()
