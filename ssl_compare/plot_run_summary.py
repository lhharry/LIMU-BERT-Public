#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize one ssl_compare run: pretrain loss curves + downstream F1 comparison.

Layout (single figure):

  Left  : MLM train/test loss vs epoch for every pretrain_<mode>.log found
  Right : downstream F1 vs label_rate, one line per tag in ssl_compare_summary.csv

By default reads the live `ssl_compare/{logs,results}` directories. Point --run_dir
at a `history/RunN_.../` snapshot to plot an archived run.

Usage (from repo root):
  python ssl_compare/plot_run_summary.py
  python ssl_compare/plot_run_summary.py --run_dir ssl_compare/history/Run2_20260528_beforeAUG
  python ssl_compare/plot_run_summary.py --run_dir ssl_compare/history/Run2_20260528_beforeAUG \
      --out ssl_compare/history/Run2_20260528_beforeAUG/run_summary.png
  python ssl_compare/plot_run_summary.py --metric acc      # plot acc instead of f1
  python ssl_compare/plot_run_summary.py --log_y_loss      # log scale on loss axis
"""
import argparse
import csv
import glob
import os
import re
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EPOCH_RE = re.compile(
    r"^Epoch\s+(\d+)/(\d+)\s*:\s*Average Loss\s*([0-9.eE+-]+)\.\s*Test Loss\s*([0-9.eE+-]+)"
)


def parse_pretrain_log(path):
    """Return (epochs, train_losses, test_losses) — lists of equal length."""
    epochs, train, test = [], [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = EPOCH_RE.match(line)
            if m:
                epochs.append(int(m.group(1)))
                train.append(float(m.group(3)))
                test.append(float(m.group(4)))
    return epochs, train, test


def load_pretrain_curves(log_dir):
    """Map mode_name -> (epochs, train, test) for every pretrain_*.log in log_dir."""
    out = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "pretrain_*.log"))):
        name = os.path.splitext(os.path.basename(path))[0].replace("pretrain_", "")
        e, tr, te = parse_pretrain_log(path)
        if e:
            out[name] = (e, tr, te)
        else:
            print("  (no epoch lines parsed from %s)" % path)
    return out


def load_summary(csv_path, metric):
    """Map tag -> sorted list of (label_rate, mean_metric) over seeds."""
    by_tag = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            try:
                lr = float(row["label_rate"])
                val = float(row[metric])
            except (TypeError, ValueError, KeyError):
                continue
            by_tag.setdefault(row["tag"], {}).setdefault(lr, []).append(val)
    out = {}
    for tag, lr_map in by_tag.items():
        points = sorted((lr, sum(vs) / len(vs)) for lr, vs in lr_map.items())
        out[tag] = points
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Plot ssl_compare pretrain curves + F1 summary.")
    p.add_argument("--run_dir", default=HERE,
                   help="Directory containing logs/ and results/ subfolders. Default: "
                        "ssl_compare/ (the live run). For an archived run, point this at "
                        "ssl_compare/history/RunN_.../.")
    p.add_argument("--metric", choices=("f1", "acc"), default="f1",
                   help="Which metric to plot on the right panel.")
    p.add_argument("--log_y_loss", action="store_true",
                   help="Use log scale on the loss y-axis (helps when curves span an order of magnitude).")
    p.add_argument("--no_test_loss", action="store_true",
                   help="Plot only train loss (less clutter when curves overlap).")
    p.add_argument("--out", default=None,
                   help="Save figure to this path instead of showing it.")
    p.add_argument("--title", default=None,
                   help="Override the figure suptitle (default: derived from --run_dir).")
    return p.parse_args()


def main():
    args = parse_args()
    log_dir = os.path.join(args.run_dir, "logs")
    res_dir = os.path.join(args.run_dir, "results")
    csv_path = os.path.join(res_dir, "ssl_compare_summary.csv")
    if not os.path.isdir(log_dir):
        sys.exit("logs/ not found under %s" % args.run_dir)
    if not os.path.exists(csv_path):
        sys.exit("ssl_compare_summary.csv not found under %s" % res_dir)

    curves = load_pretrain_curves(log_dir)
    summary = load_summary(csv_path, args.metric)

    # Stable color mapping shared between both panels (so 'scratch' is the same
    # color in the loss curve and the F1 line).
    tag_color = {}
    palette = plt.get_cmap("tab10")
    all_tags = list(curves.keys()) + [t for t in summary if t not in curves]
    for i, t in enumerate(all_tags):
        tag_color[t] = palette(i % 10)

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4.6))

    # ---- left: pretrain loss curves ----
    if not curves:
        ax_loss.text(0.5, 0.5, "no pretrain_*.log found", ha="center", va="center",
                     transform=ax_loss.transAxes)
    else:
        for name, (epochs, train, test) in curves.items():
            c = tag_color.get(name, None)
            ax_loss.plot(epochs, train, color=c, linewidth=1.6, label="%s (train)" % name)
            if not args.no_test_loss:
                ax_loss.plot(epochs, test, color=c, linewidth=1.0, linestyle="--",
                             alpha=0.8, label="%s (test)" % name)
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("MLM loss (MSE)")
        ax_loss.set_title("Pretrain loss")
        if args.log_y_loss:
            ax_loss.set_yscale("log")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(fontsize=8, loc="upper right")

    # ---- right: downstream metric vs label_rate ----
    if not summary:
        ax_f1.text(0.5, 0.5, "no rows in ssl_compare_summary.csv",
                   ha="center", va="center", transform=ax_f1.transAxes)
    else:
        for tag, points in summary.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            c = tag_color.get(tag, None)
            # Dashed line + open marker for the supervised yardstick so it visually
            # separates from the SSL rows.
            is_ref = tag.lower().startswith(("r-gru", "supervised"))
            ax_f1.plot(xs, ys, marker="o" if not is_ref else "s",
                       linestyle="--" if is_ref else "-",
                       color=c, label=tag, linewidth=1.8, markersize=5,
                       markerfacecolor=("white" if is_ref else c))
        ax_f1.set_xscale("log")
        ax_f1.set_xlabel("label_rate (log scale)")
        ax_f1.set_ylabel(args.metric.upper())
        ax_f1.set_title("Downstream %s vs label_rate" % args.metric.upper())
        ax_f1.grid(True, alpha=0.3, which="both")
        ax_f1.legend(fontsize=8, loc="lower right")

    run_name = os.path.basename(os.path.normpath(args.run_dir)) or "ssl_compare"
    fig.suptitle(args.title or ("ssl_compare run: %s" % run_name), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        fig.savefig(args.out, dpi=160, bbox_inches="tight")
        print("Saved figure -> %s" % args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
