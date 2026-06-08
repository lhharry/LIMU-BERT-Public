"""Visualize 3-fold grouped-CV results (fold0 = Run6, fold1/fold2 = Run7).

Generates three figures, all averaged over the 3 folds:
  1. Pretrain loss curves (train + test) for warmstart / scratch / dapt.
  2. Averaged confusion matrices for foundation_ft / dapt_ft / scratch_ft /
     warmstart_ft / R-GRU at label_rate = 0.05.
  3. Accuracy and macro-F1 vs label_rate for all 9 tags
     (ft = solid line, non-ft = dashed line, R-GRU = baseline).

This script ONLY reads files that already exist on disk:
  - pretrain logs            -> loss curves
  - eval_*.json              -> confusion matrices
  - groupcv_raw.csv          -> acc / f1 vs label_rate
It never fabricates data. Any missing file raises FileNotFoundError.
"""

import os
import re
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "history")

RUN6 = os.path.join(HISTORY, "Run6_20260605", "ssl_compare")
RUN7 = os.path.join(HISTORY, "Run7_20260607_3fold")

# fold_id -> {log dir, results dir, raw csv}
FOLDS = {
    0: {
        "log": os.path.join(RUN6, "logs_groupcv"),
        "res": os.path.join(RUN6, "results_groupcv"),
    },
    1: {
        "log": os.path.join(RUN7, "logs_groupcv"),
        "res": os.path.join(RUN7, "results_groupcv"),
    },
    2: {
        "log": os.path.join(RUN7, "logs_groupcv"),
        "res": os.path.join(RUN7, "results_groupcv"),
    },
}

RAW_CSVS = [
    os.path.join(RUN6, "results_groupcv", "groupcv_raw.csv"),
    os.path.join(RUN7, "results_groupcv", "groupcv_raw.csv"),
]

OUT_DIR = os.path.join(HISTORY, "figs_run67_3fold")

# Camargo 10_20_dense_8cls activity names (from dataset/camargo_v2.py ACTIVITY_NAMES)
CLASS_NAMES = ["stand", "walk", "turn", "jog",
               "rampascent", "rampdescent", "stairascent", "stairdescent"]

# Pretrain methods for the loss-curve figure
SSL_METHODS = ["warmstart", "scratch", "dapt"]

# label_rate used for the confusion-matrix figure
CM_LABEL_RATE = "0.05"

# Confusion-matrix targets: (display tag, eval-filename stub)
CM_TARGETS = [
    ("foundation_ft", "foundation_ft"),
    ("dapt_ft", "dapt_ft_merged4"),
    ("scratch_ft", "scratch_ft_merged4"),
    ("warmstart_ft", "warmstart_ft_merged4"),
    ("R-GRU", "rgru"),
]

# Tags for the acc/f1-vs-label_rate figure and their plot family colours
FAMILY_COLOR = {"foundation": "C0", "dapt": "C1", "scratch": "C2", "warmstart": "C3"}
ALL_TAGS = ["foundation", "foundation_ft", "dapt", "dapt_ft",
            "scratch", "scratch_ft", "warmstart", "warmstart_ft", "R-GRU"]

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s*:\s*Average Loss\s+([\d.]+)\.\s*Test Loss\s+([\d.]+)"
)


def _require(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("Required input file is missing: %s" % path)
    return path


# --------------------------------------------------------------------------
# 1. Loss curves
# --------------------------------------------------------------------------
def parse_loss_log(path):
    """Return {epoch: (train_loss, test_loss)} parsed from a pretrain log."""
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = EPOCH_RE.search(line)
            if m:
                ep = int(m.group(1))
                out[ep] = (float(m.group(2)), float(m.group(3)))
    if not out:
        raise ValueError("No epoch/loss lines parsed from %s" % path)
    return out


def collect_loss(method):
    """Average loss curves over the 3 folds for one pretrain method.

    Returns (epochs, train_mean, train_std, test_mean, test_std).
    Only epochs present in ALL folds are kept.
    """
    per_fold = {}
    for fold, dirs in FOLDS.items():
        path = _require(os.path.join(
            dirs["log"], "pretrain_%s_merged4_fold%d.log" % (method, fold)))
        per_fold[fold] = parse_loss_log(path)

    common = set.intersection(*[set(d.keys()) for d in per_fold.values()])
    epochs = np.array(sorted(common))

    train = np.array([[per_fold[f][e][0] for f in FOLDS] for e in epochs])
    test = np.array([[per_fold[f][e][1] for f in FOLDS] for e in epochs])
    return (epochs,
            train.mean(axis=1), train.std(axis=1),
            test.mean(axis=1), test.std(axis=1))


def plot_loss(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    titles = ["Pretrain Train Loss (3-fold mean)",
              "Pretrain Test Loss (3-fold mean)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    for i, method in enumerate(SSL_METHODS):
        ep, tr_m, tr_s, te_m, te_s = collect_loss(method)
        c = "C%d" % i
        axes[0].plot(ep, tr_m, color=c, label=method)
        axes[0].fill_between(ep, tr_m - tr_s, tr_m + tr_s, color=c, alpha=0.18)
        axes[1].plot(ep, te_m, color=c, label=method)
        axes[1].fill_between(ep, te_m - te_s, te_m + te_s, color=c, alpha=0.18)

    for ax in axes:
        ax.legend(title="Pretrain method")
    fig.suptitle("Self-supervised pretraining loss "
                 "(warmstart / scratch / dapt, shaded = ±std over 3 folds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("[saved] %s" % out_path)


# --------------------------------------------------------------------------
# 2. Confusion matrices
# --------------------------------------------------------------------------
def load_confmat(stub, fold, label_rate):
    path = _require(os.path.join(
        FOLDS[fold]["res"],
        "eval_%s_fold%d_lr%s.json" % (stub, fold, label_rate)))
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cm = np.array(data["confusion_matrix"], dtype=float)
    n = len(CLASS_NAMES)
    if cm.shape != (n, n):
        raise ValueError("Unexpected confusion-matrix shape %s in %s"
                         % (cm.shape, path))
    return cm


def avg_confmat(stub, label_rate):
    """Sum the per-fold confusion matrices, then row-normalize to recall."""
    total = sum(load_confmat(stub, f, label_rate) for f in FOLDS)
    row_sums = total.sum(axis=1, keepdims=True)
    norm = np.zeros_like(total)
    np.divide(total, row_sums, out=norm, where=row_sums != 0)
    return total, norm


def plot_confmats(out_path):
    n = len(CM_TARGETS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.8 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (tag, stub) in zip(axes, CM_TARGETS):
        _, norm = avg_confmat(stub, CM_LABEL_RATE)
        im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title("%s @ label_rate=%s" % (tag, CM_LABEL_RATE))
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = norm[i, j]
                ax.text(j, i, "%.2f" % v, ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Row-normalized confusion matrices "
                 "(3-fold summed, label_rate=%s)" % CM_LABEL_RATE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("[saved] %s" % out_path)


# --------------------------------------------------------------------------
# 3. Accuracy / F1 vs label_rate
# --------------------------------------------------------------------------
def load_raw_rows():
    rows = []
    for path in RAW_CSVS:
        _require(path)
        with open(path, "r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("status") != "ok":
                    continue
                rows.append({
                    "tag": r["tag"],
                    "label_rate": float(r["label_rate"]),
                    "fold": int(r["fold"]),
                    "acc": float(r["acc"]),
                    "f1": float(r["f1"]),
                })
    if not rows:
        raise ValueError("No rows loaded from raw CSVs")
    return rows


def aggregate(rows, tag, metric):
    """Return (sorted label_rates, means, stds) over folds for one tag/metric."""
    by_rate = {}
    for r in rows:
        if r["tag"] == tag:
            by_rate.setdefault(r["label_rate"], []).append(r[metric])
    rates = sorted(by_rate)
    means = np.array([np.mean(by_rate[x]) for x in rates])
    stds = np.array([np.std(by_rate[x]) for x in rates])
    return np.array(rates), means, stds


def tag_style(tag):
    if tag == "R-GRU":
        return dict(color="black", linestyle="-", marker="o",
                    linewidth=2.2, label="R-GRU (baseline)")
    if tag.endswith("_ft"):
        return dict(color=FAMILY_COLOR[tag[:-3]], linestyle="-", marker="o",
                    linewidth=1.8, label=tag)
    return dict(color=FAMILY_COLOR[tag], linestyle="--", marker="s",
                linewidth=1.8, label=tag)


def plot_curves(out_path):
    rows = load_raw_rows()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metrics = [("acc", "Accuracy"), ("f1", "Macro-F1")]

    all_rates = sorted({r["label_rate"] for r in rows})
    for ax, (metric, nice) in zip(axes, metrics):
        for tag in ALL_TAGS:
            rates, means, stds = aggregate(rows, tag, metric)
            st = tag_style(tag)
            ax.errorbar(rates, means, yerr=stds, capsize=3, markersize=5, **st)
        ax.set_title("%s vs label_rate (3-fold mean ± std)" % nice)
        ax.set_xlabel("label_rate")
        ax.set_ylabel(nice)
        ax.set_xticks(all_rates)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Downstream %s on Camargo (grouped 3-fold CV): "
                 "ft = solid, non-ft = dashed" % "performance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("[saved] %s" % out_path)


# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_loss(os.path.join(OUT_DIR, "fig1_pretrain_loss_3fold.png"))
    plot_confmats(os.path.join(OUT_DIR, "fig2_confusion_matrices_lr0.05_3fold.png"))
    plot_curves(os.path.join(OUT_DIR, "fig3_acc_f1_vs_label_rate_3fold.png"))
    print("Done. Figures in: %s" % OUT_DIR)


if __name__ == "__main__":
    main()
