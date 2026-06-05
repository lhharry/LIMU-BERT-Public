#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Subject-grouped 5-fold CV driver for the in-domain SSL comparison.

`run_ssl_compare.py` is NOT wired to loop folds (README lines 93-96): the grouped
capability lives only on `pretrain_ssl.py` + `bench_eval.py`. This script is that
fold loop. For every fold it (re)builds the full, publishable (B) matrix (paper line
(B) = "reuse the phone-HAR foundation weight limu_bert_x"):

  Phase 1  pretrain, per fold, per SSL mode -> one ckpt each (on the 4 datasets MERGED,
           ONLY camargo's test fold held out):
             scratch_merged4_fold<F>_seed<SEED>.pt    (random init,     full lr 1e-3)
             warmstart_merged4_fold<F>_seed<SEED>.pt  (foundation init, full lr 1e-3)
             dapt_merged4_fold<F>_seed<SEED>.pt       (foundation init, gentle lr 1e-4)

  Phase 2  downstream eval, per fold, per label_rate (9 rows):
             <mode> / <mode>_ft   for mode in {scratch, warmstart, dapt}
                                  bert_separated probe + bert joint-finetune
             foundation / _ft     reuse limu_bert_x DIRECTLY, no in-domain SSL ((B) protagonist)
             R-GRU                supervised, no pretraining (the yardstick)

  Phase 3  aggregate F1/acc to mean +/- std OVER FOLDS per (tag, label_rate).

EVERY pretrain and EVERY finetune subprocess streams its complete per-epoch
training log to its own dedicated file under logs_groupcv/ (unbuffered, line by
line -- nothing is merged or truncated), mirroring the Run5 logs.

Run from the repo root:

  python ssl_compare/run_groupcv.py --gpu 0                  # full 5-fold matrix
  python ssl_compare/run_groupcv.py --gpu 0 --dry            # print every command, run nothing
  python ssl_compare/run_groupcv.py --gpu 0 --folds 1,2,3,4  # only the missing folds
  python ssl_compare/run_groupcv.py --gpu 0 \
      --skip_existing_ckpt --skip_existing_eval              # resume without redoing finished work
  python ssl_compare/run_groupcv.py --gpu 0 --only scratch   # one SSL mode
  python ssl_compare/run_groupcv.py --aggregate_only         # just rebuild the summary from existing JSONs
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
LOG_DIR = os.path.join(HERE, "logs_groupcv")
RESULT_DIR = os.path.join(HERE, "results_groupcv")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CONFIG  (edit here)
# ---------------------------------------------------------------------------
MODEL_VERSION = "v3"
TRAINING_RATE = 0.8
SEED = 3431               # single fixed model seed (pretrain & eval share it -> aligned split)
SPLIT_SEED = 3431         # fixed fold partition seed, independent of model seed
N_FOLDS = 5
FOLDS = [0, 1, 2, 3, 4]   # default: all folds; override with --folds
LABEL_RATES = [0.01, 0.02, 0.05]   # matches the Run5 pilot; override with --label_rates
AUGMENT = 0               # held constant across modes (Run5 used 0)
BATCH_SIZE = 512          # pretrain batch. Keep IDENTICAL across all folds (don't mix with old 128 ckpts).
NUM_WORKERS = 4           # DataLoader workers for pretrain + downstream training loaders
GROUP_LABEL_INDEX = 1     # camargo subject id column

# The 4 datasets merged into ONE SSL pool; only the holdout's test fold is excluded.
MERGE_DATASETS = [
    ("camargo",         "10_20_dense_8cls"),
    ("molinaro",        "10_20_both"),
    ("scherpereel",     "10_20_both"),
    ("scherpereel_exo", "10_20_both"),
]
MERGE_SPEC = ",".join("%s:%s" % (d, v) for d, v in MERGE_DATASETS)
HOLDOUT_DATASET = "camargo"

# Host dataset = where ckpts are saved + which model config is used (dims are
# dataset-independent: all (N, 20, 6) -> base_v3 feature_num=6). Also the eval target.
HOST_DATASET = "camargo"
HOST_VERSION = "10_20_dense_8cls"
EVAL_LABEL_INDEX = 0
HOST_DIR = os.path.join("saved", "pretrain_base_%s_%s" % (HOST_DATASET, HOST_VERSION))
FOUNDATION_CKPT = os.path.join(HOST_DIR, "limu_bert_x")   # no .pt; loaders re-append it

# Per-mode SSL recipe. Under paper line (B) "reuse the foundation weight limu_bert_x":
#   foundation-only (no SSL here) + dapt (gentle adapt) are the (B) protagonists vs R-GRU;
#   warmstart (foundation init + full-lr relearn) is (B)'s ablation; scratch is the
#   "no-foundation in-domain SSL" control. scratch=1200 so its pretext actually converges
#   (the 800ep Run5 curve was still descending -> would be an under-trained, unfair control).
SSL_MODES = {
    "scratch":   {"lr": 1e-3, "epochs": 1200, "use_foundation": False},
    "warmstart": {"lr": 1e-3, "epochs": 400,  "use_foundation": True},
    "dapt":      {"lr": 1e-4, "epochs": 300,  "use_foundation": True},
}

# Supervised yardstick (no pretraining). Set to None to drop it.
SUPERVISED_REF = {"tag": "R-GRU", "method": "gru", "model_version": MODEL_VERSION}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ckpt_stem(mode, fold):
    """SSL ckpt path WITHOUT .pt (bench_eval re-appends). Matches pretrain_ssl naming."""
    return os.path.join(HOST_DIR, "%s_merged4_fold%d_seed%d" % (mode, fold, SEED))


def stream_subprocess(cmd, log_path, dry):
    """Run cmd, streaming child stdout/stderr line-by-line to BOTH console and its
    own complete log file. PYTHONUNBUFFERED + bufsize=1 keep per-epoch prints live
    and guarantee the full training log is captured (same recipe that produced the
    Run5 logs)."""
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
# Phase 1: pretrain one (mode, fold)
# ---------------------------------------------------------------------------
def pretrain_one(mode, fold, gpu, dry, skip_existing):
    spec = SSL_MODES[mode]
    log_path = os.path.join(LOG_DIR, "pretrain_%s_merged4_fold%d.log" % (mode, fold))
    cmd = [
        sys.executable, "-u", PRETRAIN_SSL,
        MODEL_VERSION, HOST_DATASET, HOST_VERSION,
        "--mode", mode,
        "--merge", MERGE_SPEC,
        "--holdout_dataset", HOLDOUT_DATASET,
        "--out_name", "%s_merged4" % mode,
        "--seeds", str(SEED),
        "--training_rate", str(TRAINING_RATE),
        "--lr", repr(spec["lr"]),
        "--epochs", str(spec["epochs"]),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", str(NUM_WORKERS),
        "--augment", str(AUGMENT),
        "--split", "group",
        "--group_label_index", str(GROUP_LABEL_INDEX),
        "--fold_id", str(fold),
        "--n_folds", str(N_FOLDS),
        "--split_seed", str(SPLIT_SEED),
    ]
    if spec["use_foundation"]:
        cmd += ["-f", FOUNDATION_CKPT]
    if skip_existing:
        cmd += ["--skip_existing"]
    if gpu is not None:
        cmd += ["-g", gpu]
    ret = stream_subprocess(cmd, log_path, dry)
    if ret != 0 and not dry:
        print("!! pretrain %s fold%d failed (rc=%d); see %s" % (mode, fold, ret, log_path))
    return ret


# ---------------------------------------------------------------------------
# Phase 2: one downstream eval
# ---------------------------------------------------------------------------
def eval_one(name, tag, extra, fold, label_rate, gpu, dry, skip_existing):
    """name -> file stem; tag -> summary grouping key."""
    rid = "%s_fold%d_lr%s" % (name, fold, label_rate)
    log_path = os.path.join(LOG_DIR, "eval_%s.log" % rid)
    json_path = os.path.join(RESULT_DIR, "eval_%s.json" % rid)
    base = {"tag": tag, "label_rate": label_rate, "fold": fold}
    if skip_existing and os.path.exists(json_path):
        with open(json_path, "r") as f:
            r = json.load(f)
        print("=== skip existing eval: %s ===" % rid, flush=True)
        return dict(base, acc=r.get("acc"), f1=r.get("f1"), status="ok")
    cmd = [
        sys.executable, "-u", BENCH_EVAL,
        "--dataset", HOST_DATASET,
        "--dataset_version", HOST_VERSION,
        "--label_rate", str(label_rate),
        "--training_rate", str(TRAINING_RATE),
        "--seed", str(SEED),
        "--label_index", str(EVAL_LABEL_INDEX),
        "--balance", "1",
        "--split", "group",
        "--group_label_index", str(GROUP_LABEL_INDEX),
        "--fold_id", str(fold),
        "--n_folds", str(N_FOLDS),
        "--split_seed", str(SPLIT_SEED),
        "--num_workers", str(NUM_WORKERS),
        "--save_model", "groupcv_" + rid,
        "--out_json", json_path,
    ] + extra
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


def eval_rows_for_fold(fold):
    """(name, tag, extra-args) for every downstream row of one fold."""
    rows = []
    for mode in SSL_MODES:
        stem = ckpt_stem(mode, fold)
        # representation-quality probe (frozen BERT + GRU head)
        rows.append((
            "%s_merged4" % mode, mode,
            ["--mode", "bert_separated", "--method", "gru",
             "--model_version", MODEL_VERSION, "--pretrain_model", stem],
        ))
        # joint finetune (BERT + GRU co-trained) -- comparable to R-GRU
        rows.append((
            "%s_ft_merged4" % mode, "%s_ft" % mode,
            ["--mode", "bert", "--method", "base_gru",
             "--model_version", MODEL_VERSION + "_" + MODEL_VERSION,
             "--pretrain_model", stem, "--frozen_bert", "0"],
        ))
    # foundation-only: reuse the phone-HAR foundation weight DIRECTLY, no in-domain SSL.
    # Purest test of paper line (B). Not in SSL_MODES -> never pretrained here; the ckpt
    # (FOUNDATION_CKPT) is fold-independent but evaluated per fold's held-out subjects.
    rows.append((
        "foundation", "foundation",
        ["--mode", "bert_separated", "--method", "gru",
         "--model_version", MODEL_VERSION, "--pretrain_model", FOUNDATION_CKPT],
    ))
    rows.append((
        "foundation_ft", "foundation_ft",
        ["--mode", "bert", "--method", "base_gru",
         "--model_version", MODEL_VERSION + "_" + MODEL_VERSION,
         "--pretrain_model", FOUNDATION_CKPT, "--frozen_bert", "0"],
    ))
    if SUPERVISED_REF is not None:
        ref = SUPERVISED_REF
        rows.append((
            "rgru", ref["tag"],
            ["--mode", "supervised", "--method", ref["method"],
             "--model_version", ref["model_version"]],
        ))
    return rows


# ---------------------------------------------------------------------------
# Phase 3: aggregate over folds
# ---------------------------------------------------------------------------
def aggregate(results, folds, label_rates):
    raw_path = os.path.join(RESULT_DIR, "groupcv_raw.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as csvf:
        w = csv.DictWriter(csvf, fieldnames=["tag", "label_rate", "fold", "acc", "f1", "status"])
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["tag"], x["label_rate"], x["fold"])):
            w.writerow(r)

    tags = []
    for r in results:
        if r["tag"] not in tags:
            tags.append(r["tag"])

    mean_path = os.path.join(RESULT_DIR, "groupcv_meanstd.csv")
    with open(mean_path, "w", newline="", encoding="utf-8") as csvf:
        w = csv.writer(csvf)
        w.writerow(["metric", "tag", "label_rate", "mean", "std", "n_folds"])
        for metric in ("f1", "acc"):
            print("\n======== %s : %s (mean +/- std over %d folds) ========"
                  % (HOST_DATASET, metric.upper(), len(folds)))
            print("%-12s" % "label_rate" + "".join("%-20s" % t for t in tags))
            for lr in label_rates:
                cells = []
                for t in tags:
                    vals = [x[metric] for x in results
                            if x["tag"] == t and x["label_rate"] == lr and x[metric] is not None]
                    if vals:
                        m = sum(vals) / len(vals)
                        s = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
                        cells.append("%-20s" % ("%.3f +/- %.3f" % (m, s)))
                        w.writerow([metric, t, lr, "%.4f" % m, "%.4f" % s, len(vals)])
                    else:
                        cells.append("%-20s" % "-")
                print("%-12s" % str(lr) + "".join(cells))
    print("\nRaw per-fold CSV : %s" % raw_path)
    print("Mean+/-std CSV   : %s" % mean_path)


def load_existing_results(folds, label_rates):
    """Rebuild the results list purely from JSONs on disk (for --aggregate_only)."""
    results = []
    for fold in folds:
        for name, tag, _ in eval_rows_for_fold(fold):
            for lr in label_rates:
                jp = os.path.join(RESULT_DIR, "eval_%s_fold%d_lr%s.json" % (name, fold, lr))
                if not os.path.exists(jp):
                    continue
                with open(jp, "r") as f:
                    r = json.load(f)
                results.append({"tag": tag, "label_rate": lr, "fold": fold,
                                "acc": r.get("acc"), "f1": r.get("f1"), "status": "ok"})
    return results


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Grouped 5-fold CV driver for ssl_compare")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--dry", action="store_true", help="Print every command, run nothing.")
    ap.add_argument("--folds", default=None, help="Comma-separated fold ids (default: all 5).")
    ap.add_argument("--label_rates", default=None, help="Comma-separated override.")
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of SSL modes (scratch,warmstart).")
    ap.add_argument("--skip_pretrain", action="store_true", help="Skip Phase 1 entirely.")
    ap.add_argument("--skip_eval", action="store_true", help="Skip Phase 2 entirely.")
    ap.add_argument("--skip_existing_ckpt", action="store_true",
                    help="Phase 1: skip a (mode,fold) whose ckpt already exists.")
    ap.add_argument("--skip_existing_eval", action="store_true",
                    help="Phase 2: reuse a row whose output JSON already exists.")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="Skip pretrain+eval; rebuild the summary from existing JSONs.")
    args = ap.parse_args()

    folds = FOLDS if args.folds is None else [int(x) for x in args.folds.split(",")]
    label_rates = LABEL_RATES if args.label_rates is None else [float(x) for x in args.label_rates.split(",")]
    modes = list(SSL_MODES) if args.only is None else [m.strip() for m in args.only.split(",")]
    for m in modes:
        if m not in SSL_MODES:
            sys.exit("Unknown SSL mode: %s (have %s)" % (m, list(SSL_MODES)))

    if args.aggregate_only:
        results = load_existing_results(folds, label_rates)
        if results:
            aggregate(results, folds, label_rates)
        else:
            print("No result JSONs found under %s" % RESULT_DIR)
        return

    # Phase 1: pretrain every (mode, fold) first so a single GPU stays busy and
    # all ckpts exist before any eval reads them.
    if not args.skip_pretrain:
        for fold in folds:
            for mode in modes:
                pretrain_one(mode, fold, args.gpu, args.dry, args.skip_existing_ckpt)
    else:
        print("Skipping pretrain phase.")

    if args.skip_eval:
        print("Skipping eval phase.")
        return

    # Phase 2: downstream eval for every (fold, row, label_rate).
    results = []
    for fold in folds:
        for name, tag, extra in eval_rows_for_fold(fold):
            # an R-GRU row uses no SSL mode; SSL rows are filtered by --only
            mode_of_row = name.split("_")[0]
            if mode_of_row in SSL_MODES and mode_of_row not in modes:
                continue
            for lr in label_rates:
                row = eval_one(name, tag, extra, fold, lr, args.gpu, args.dry,
                               args.skip_existing_eval)
                if row is not None:
                    results.append(row)

    if not args.dry and results:
        aggregate(results, folds, label_rates)


if __name__ == "__main__":
    main()
