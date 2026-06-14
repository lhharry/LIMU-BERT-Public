"""DAPT ablation: compare three dapt variants under identical 3-fold grouped CV.

All variants share the same recipe (lr=1e-4, epochs=300, merge4, foundation init,
split_seed=3431); they differ in ONE pretraining knob each:

  baseline   dapt, mask 0.15/0.8, augment=0   (fold0=Run6, fold1/2=Run7)
  mask0.35   dapt, mask 0.35/0.95, augment=0  (Run8_20260608_maskconfig)
  +aug       dapt, mask 0.15/0.8, augment=1   (Run9_20260608_AUG1)

Three 3-fold-averaged figures are produced, comparing frozen probe (`dapt`) and
finetune (`dapt_ft`) against the supervised `R-GRU` baseline:
  1. dapt pretraining loss (train + test)
  2. row-normalized confusion matrices at label_rate=0.05
  3. accuracy / macro-F1 vs label_rate

This script ONLY reads files that already exist on disk; it never fabricates data.
Any missing file raises FileNotFoundError.
"""

import os
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_run67_3fold import EPOCH_RE, parse_loss_log, CLASS_NAMES, _require

# --------------------------------------------------------------------------
# Paths / variant sources
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "history")

RUN6 = os.path.join(HISTORY, "Run6_20260605", "ssl_compare")
RUN7 = os.path.join(HISTORY, "Run7_20260607_3fold")
RUN8 = os.path.join(HISTORY, "Run8_20260608_maskconfig")
RUN9 = os.path.join(HISTORY, "Run9_20260608_AUG1")


def _dirs(base):
    return {"log": os.path.join(base, "logs_groupcv"),
            "res": os.path.join(base, "results_groupcv")}


# baseline spans Run6 (fold0) + Run7 (fold1/2); Run8/Run9 hold all 3 folds each.
BASELINE_FOLDS = {0: _dirs(RUN6), 1: _dirs(RUN7), 2: _dirs(RUN7)}
RUN8_FOLDS = {f: _dirs(RUN8) for f in (0, 1, 2)}
RUN9_FOLDS = {f: _dirs(RUN9) for f in (0, 1, 2)}

# Each variant: display label, plot colour, per-fold dirs, raw-csv list.
VARIANTS = [
    {"label": "dapt (mask0.15, no-aug)", "color": "C0", "folds": BASELINE_FOLDS,
     "raw_csvs": [os.path.join(RUN6, "results_groupcv", "groupcv_raw.csv"),
                  os.path.join(RUN7, "results_groupcv", "groupcv_raw.csv")]},
    {"label": "dapt (mask0.35)", "color": "C1", "folds": RUN8_FOLDS,
     "raw_csvs": [os.path.join(RUN8, "results_groupcv", "groupcv_raw.csv")]},
    {"label": "dapt (+aug)", "color": "C2", "folds": RUN9_FOLDS,
     "raw_csvs": [os.path.join(RUN9, "results_groupcv", "groupcv_raw.csv")]},
]

# R-GRU baseline is dataset/fold-deterministic -> take it from the baseline run.
RGRU_FOLDS = BASELINE_FOLDS
RGRU_CSVS = VARIANTS[0]["raw_csvs"]

CM_LABEL_RATE = "0.05"
LABEL_RATES = [0.01, 0.02, 0.05]
OUT_DIR = os.path.join(HISTORY, "figs_dapt_ablation")


# --------------------------------------------------------------------------
# Loaders (parameterized by variant source)
# --------------------------------------------------------------------------
def collect_loss(variant):
    """3-fold-averaged dapt loss. Returns epochs, train mean/std, test mean/std."""
    per_fold = {}
    for fold, d in variant["folds"].items():
        path = _require(os.path.join(d["log"], "pretrain_dapt_merged4_fold%d.log" % fold))
        per_fold[fold] = parse_loss_log(path)
    common = set.intersection(*[set(v.keys()) for v in per_fold.values()])
    epochs = np.array(sorted(common))
    folds = sorted(per_fold)
    train = np.array([[per_fold[f][e][0] for f in folds] for e in epochs])
    test = np.array([[per_fold[f][e][1] for f in folds] for e in epochs])
    return (epochs, train.mean(1), train.std(1), test.mean(1), test.std(1))


def load_confmat(folds, stub, fold, label_rate):
    path = _require(os.path.join(
        folds[fold]["res"], "eval_%s_fold%d_lr%s.json" % (stub, fold, label_rate)))
    with open(path, "r", encoding="utf-8") as fh:
        cm = np.array(json.load(fh)["confusion_matrix"], dtype=float)
    n = len(CLASS_NAMES)
    if cm.shape != (n, n):
        raise ValueError("Unexpected confusion-matrix shape %s in %s" % (cm.shape, path))
    return cm


def avg_confmat(folds, stub, label_rate):
    """Sum per-fold confusion matrices, then row-normalize to recall."""
    total = sum(load_confmat(folds, stub, f, label_rate) for f in folds)
    row_sums = total.sum(axis=1, keepdims=True)
    norm = np.zeros_like(total)
    np.divide(total, row_sums, out=norm, where=row_sums != 0)
    return norm


def load_raw_rows(raw_csvs):
    rows = []
    for path in raw_csvs:
        _require(path)
        with open(path, "r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("status") != "ok":
                    continue
                rows.append({"tag": r["tag"], "label_rate": float(r["label_rate"]),
                             "fold": int(r["fold"]), "acc": float(r["acc"]),
                             "f1": float(r["f1"])})
    if not rows:
        raise ValueError("No rows loaded from %s" % raw_csvs)
    return rows


def aggregate(rows, tag, metric):
    """(sorted label_rates, means, stds) over folds for one tag/metric."""
    by_rate = {}
    for r in rows:
        if r["tag"] == tag:
            by_rate.setdefault(r["label_rate"], []).append(r[metric])
    rates = sorted(by_rate)
    means = np.array([np.mean(by_rate[x]) for x in rates])
    stds = np.array([np.std(by_rate[x]) for x in rates])
    return np.array(rates), means, stds


# --------------------------------------------------------------------------
# Figure 1: pretrain loss
# --------------------------------------------------------------------------
def plot_loss(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, title in zip(axes, ["DAPT Train Loss (3-fold mean)",
                                "DAPT Test Loss (3-fold mean)"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    for v in VARIANTS:
        ep, tr_m, tr_s, te_m, te_s = collect_loss(v)
        c = v["color"]
        axes[0].plot(ep, tr_m, color=c, label=v["label"])
        axes[0].fill_between(ep, tr_m - tr_s, tr_m + tr_s, color=c, alpha=0.18)
        axes[1].plot(ep, te_m, color=c, label=v["label"])
        axes[1].fill_between(ep, te_m - te_s, te_m + te_s, color=c, alpha=0.18)
    for ax in axes:
        ax.legend()
    fig.suptitle("DAPT pretraining loss (shaded = +/-std over 3 folds)\n"
                 "NOTE: mask ratio / augmentation change the pretext difficulty -> "
                 "absolute loss is NOT directly comparable; judge by downstream metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("[saved] %s" % out_path)


# --------------------------------------------------------------------------
# Figure 2: confusion matrices @ label_rate 0.05
# --------------------------------------------------------------------------
def _draw_cm(ax, norm, title):
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = norm[i, j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=6)
    return im


def plot_confmats(out_path):
    ncols = 4
    fig, axes = plt.subplots(2, ncols, figsize=(5.0 * ncols, 9.6))
    # Row 0: dapt_ft per variant + R-GRU
    for k, v in enumerate(VARIANTS):
        norm = avg_confmat(v["folds"], "dapt_ft_merged4", CM_LABEL_RATE)
        im = _draw_cm(axes[0, k], norm, "%s  [finetune]" % v["label"])
    rgru = avg_confmat(RGRU_FOLDS, "rgru", CM_LABEL_RATE)
    im = _draw_cm(axes[0, 3], rgru, "R-GRU (baseline)")
    # Row 1: frozen dapt per variant
    for k, v in enumerate(VARIANTS):
        norm = avg_confmat(v["folds"], "dapt_merged4", CM_LABEL_RATE)
        _draw_cm(axes[1, k], norm, "%s  [frozen probe]" % v["label"])
    axes[1, 3].axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.suptitle("Row-normalized confusion matrices (3-fold summed, label_rate=%s)"
                 % CM_LABEL_RATE)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[saved] %s" % out_path)


# --------------------------------------------------------------------------
# Figure 3: acc / f1 vs label_rate
# --------------------------------------------------------------------------
def plot_curves(out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metrics = [("acc", "Accuracy"), ("f1", "Macro-F1")]

    variant_rows = [(v, load_raw_rows(v["raw_csvs"])) for v in VARIANTS]
    rgru_rows = load_raw_rows(RGRU_CSVS)

    for ax, (metric, nice) in zip(axes, metrics):
        for v, rows in variant_rows:
            c = v["color"]
            r, m, s = aggregate(rows, "dapt_ft", metric)
            ax.errorbar(r, m, yerr=s, color=c, linestyle="-", marker="o",
                        capsize=3, markersize=5, label="%s [ft]" % v["label"])
            r, m, s = aggregate(rows, "dapt", metric)
            ax.errorbar(r, m, yerr=s, color=c, linestyle="--", marker="s",
                        capsize=3, markersize=5, label="%s [frozen]" % v["label"])
        r, m, s = aggregate(rgru_rows, "R-GRU", metric)
        ax.errorbar(r, m, yerr=s, color="black", linestyle="-", marker="^",
                    linewidth=2.2, capsize=3, markersize=6, label="R-GRU (baseline)")
        ax.set_title("%s vs label_rate (3-fold mean +/- std)" % nice)
        ax.set_xlabel("label_rate")
        ax.set_ylabel(nice)
        ax.set_xticks(LABEL_RATES)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("DAPT ablation on Camargo (grouped 3-fold CV): "
                 "ft = solid, frozen probe = dashed")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("[saved] %s" % out_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_loss(os.path.join(OUT_DIR, "fig1_dapt_loss.png"))
    plot_confmats(os.path.join(OUT_DIR, "fig2_dapt_confusion_lr0.05.png"))
    plot_curves(os.path.join(OUT_DIR, "fig3_dapt_acc_f1.png"))
    print("Done. Figures in: %s" % OUT_DIR)


if __name__ == "__main__":
    main()
