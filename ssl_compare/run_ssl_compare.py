#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Three-way in-domain SSL comparison orchestrator.

Pretrains, then evaluates, three foundation-model recipes on Camargo and writes a
single comparison table:

  scratch    from-scratch in-domain SSL   (random init,      full lr,   full epochs)
  warmstart  warm-start in-domain SSL     (foundation init,  full lr,   full epochs)
  dapt       naive DAPT                   (foundation init,  gentle lr, few epochs)

Optionally adds a supervised R-GRU row (no pretraining) as the yardstick -- the
whole question is whether ANY of the SSL recipes beats it (see project notes).

Pipeline:
  Phase 1 (pretrain): for each mode, call ssl_compare/pretrain_ssl.py once. It emits
                      one per-seed checkpoint: saved/pretrain_base_<ds>_<ver>/<mode>_seed<seed>.pt
  Phase 2 (eval):     for each (row, label_rate, seed), call benchmark_results/bench_eval.py.
                      SSL rows run --mode bert_separated with the SEED-MATCHING checkpoint
                      (so the held-out test split was never seen during pretraining).
  Phase 3 (report):   aggregate F1 to mean +/- std per (row, label_rate) and dump CSV.

Edit the CONFIG block below to change dataset / seeds / label_rates / per-mode
hyperparameters. Run from the repo root:

  python ssl_compare/run_ssl_compare.py --gpu 0
  python ssl_compare/run_ssl_compare.py --dry                 # print commands only
  python ssl_compare/run_ssl_compare.py --skip_pretrain       # reuse existing ckpts
  python ssl_compare/run_ssl_compare.py --only warmstart,dapt # subset of rows
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
DATASET = "camargo"
DATASET_VERSION = "10_20_dense_8cls"
MODEL_VERSION = "v3"                       # BERT config version (base_v3)
TRAINING_RATE = 0.8
SEEDS = [3431]   # single fixed seed: pretrain and eval both use it -> aligned split, no leakage
LABEL_RATES = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
LABEL_INDEX = 0                            # 0 = activity for camargo
AUGMENT = 1                                # rotation+noise, held constant across SSL modes

PRETRAIN_DIR = os.path.join("saved", "pretrain_base_" + DATASET + "_" + DATASET_VERSION)
FOUNDATION_CKPT = os.path.join(PRETRAIN_DIR, "limu_bert_x")   # no .pt; loaders re-append it

# Downstream evaluation path for the SSL rows. "bert_separated" = BERT as a frozen
# feature extractor + standalone GRU head (closest to a representation-quality probe
# and matches inference/test_csv.py). Swap to "bert" for joint finetune.
EVAL_MODE = "bert_separated"

# Per-mode pretraining recipe. lr/epochs forwarded to pretrain_ssl.py (which also
# has these as defaults); use_init decides whether -f FOUNDATION_CKPT is passed.
SSL_MODES = {
    "scratch":   {"lr": 1e-3, "epochs": 1200, "use_init": False},
    "warmstart": {"lr": 1e-3, "epochs": 1200, "use_init": True},
    "dapt":      {"lr": 1e-4, "epochs": 300,  "use_init": True},
}

# Supervised yardstick (no pretraining). Set to None to drop it.
SUPERVISED_REF = {"tag": "R-GRU", "method": "gru", "model_version": "v3"}


def ckpt_stem(mode, seed):
    """Per-seed checkpoint path (no .pt); bench_eval / loaders re-append .pt."""
    return os.path.join(PRETRAIN_DIR, "%s_seed%d" % (mode, seed))


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
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        ret = proc.wait()
        logf.write("\n%s\nreturncode: %d\nend: %s\n"
                   % ("-" * 64, ret, dt.datetime.now().isoformat()))
    return ret


# ---------------------------------------------------------------------------
# Phase 1: pretrain each SSL mode (one call per mode -> all seeds)
# ---------------------------------------------------------------------------
def pretrain_phase(modes, gpu, dry, skip_existing):
    seeds_arg = ",".join(str(s) for s in SEEDS)
    for mode in modes:
        spec = SSL_MODES[mode]
        cmd = [
            sys.executable, PRETRAIN_SSL,
            MODEL_VERSION, DATASET, DATASET_VERSION,
            "--mode", mode,
            "--seeds", seeds_arg,
            "--training_rate", str(TRAINING_RATE),
            "--lr", repr(spec["lr"]),
            "--epochs", str(spec["epochs"]),
            "--augment", str(AUGMENT),
        ]
        if spec["use_init"]:
            cmd += ["-f", FOUNDATION_CKPT]
        if skip_existing:
            cmd += ["--skip_existing"]
        if gpu is not None:
            cmd += ["-g", gpu]
        log_path = os.path.join(LOG_DIR, "pretrain_%s.log" % mode)
        ret = stream_subprocess(cmd, log_path, dry)
        if ret != 0 and not dry:
            print("!! pretrain mode=%s failed (rc=%d); see %s" % (mode, ret, log_path))


# ---------------------------------------------------------------------------
# Phase 2: evaluate each row via bench_eval.py
# ---------------------------------------------------------------------------
def eval_one(tag, eval_cmd_extra, label_rate, seed, gpu, dry):
    rid = "%s__lr%s__seed%d" % (tag.replace(" ", "_"), label_rate, seed)
    log_path = os.path.join(LOG_DIR, "eval_" + rid + ".log")
    json_path = os.path.join(RESULT_DIR, "eval_" + rid + ".json")
    cmd = [
        sys.executable, BENCH_EVAL,
        "--dataset", DATASET,
        "--dataset_version", DATASET_VERSION,
        "--label_rate", str(label_rate),
        "--training_rate", str(TRAINING_RATE),
        "--seed", str(seed),
        "--label_index", str(LABEL_INDEX),
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
        return {"tag": tag, "label_rate": label_rate, "seed": seed,
                "acc": None, "f1": None, "status": "failed(rc=%d)" % ret}
    with open(json_path, "r") as f:
        r = json.load(f)
    return {"tag": tag, "label_rate": label_rate, "seed": seed,
            "acc": r.get("acc"), "f1": r.get("f1"), "status": "ok"}


def build_eval_rows(modes):
    """Each row -> (tag, function(seed) -> extra bench_eval args)."""
    rows = []
    for mode in modes:
        if EVAL_MODE == "bert_separated":
            rows.append((mode, lambda seed, m=mode: [
                "--mode", "bert_separated", "--method", "gru",
                "--model_version", MODEL_VERSION,
                "--pretrain_model", ckpt_stem(m, seed),
            ]))
        else:  # joint finetune
            rows.append((mode, lambda seed, m=mode: [
                "--mode", "bert", "--method", "base_gru",
                "--model_version", MODEL_VERSION + "_" + MODEL_VERSION,
                "--pretrain_model", ckpt_stem(m, seed), "--frozen_bert", "0",
            ]))
    if SUPERVISED_REF is not None:
        ref = SUPERVISED_REF
        rows.append((ref["tag"], lambda seed: [
            "--mode", "supervised", "--method", ref["method"],
            "--model_version", ref["model_version"],
        ]))
    return rows


def eval_phase(modes, label_rates, gpu, dry):
    rows = build_eval_rows(modes)
    results = []
    for tag, extra_fn in rows:
        for lr in label_rates:
            for seed in SEEDS:
                row = eval_one(tag, extra_fn(seed), lr, seed, gpu, dry)
                if row is not None:
                    results.append(row)
    return results


# ---------------------------------------------------------------------------
# Phase 3: aggregate + report
# ---------------------------------------------------------------------------
def aggregate(results, label_rates):
    summary_path = os.path.join(RESULT_DIR, "ssl_compare_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["tag", "label_rate", "seed", "acc", "f1", "status"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # mean +/- std F1 per (tag, label_rate)
    tags = []
    for r in results:
        if r["tag"] not in tags:
            tags.append(r["tag"])
    print("\n================ F1 (mean +/- std over seeds) ================")
    head = "%-12s" % "label_rate" + "".join("%-22s" % t for t in tags)
    print(head)
    for lr in label_rates:
        cells = []
        for t in tags:
            f1s = [x["f1"] for x in results
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
    ap.add_argument("--skip_pretrain", action="store_true", help="Reuse existing checkpoints.")
    ap.add_argument("--skip_eval", action="store_true", help="Only run the pretrain phase.")
    ap.add_argument("--skip_existing_ckpt", action="store_true",
                    help="During pretrain, skip seeds whose ckpt already exists.")
    ap.add_argument("--only", default=None,
                    help="Comma-separated SSL modes to include (default: all of scratch,warmstart,dapt).")
    ap.add_argument("--label_rates", default=None, help="Comma-separated override.")
    args = ap.parse_args()

    modes = list(SSL_MODES.keys())
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        modes = [m for m in modes if m in keys]
        if not modes:
            sys.exit("--only matched no known modes %s" % list(SSL_MODES.keys()))
    label_rates = LABEL_RATES if args.label_rates is None else [float(x) for x in args.label_rates.split(",")]

    if not args.skip_pretrain:
        pretrain_phase(modes, args.gpu, args.dry, args.skip_existing_ckpt)
    else:
        print("Skipping pretrain phase (reusing existing checkpoints).")

    if args.skip_eval:
        print("Skipping eval phase.")
        return

    results = eval_phase(modes, label_rates, args.gpu, args.dry)
    if not args.dry and results:
        aggregate(results, label_rates)


if __name__ == "__main__":
    main()
