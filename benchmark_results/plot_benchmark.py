#!/usr/bin/env python
"""
Read benchmark_results/results/summary.csv and produce:
  - plots/labelrate_vs_f1.png
  - plots/labelrate_vs_acc.png
  - plots/confusion_<tag>_lr<rate>_seed<seed>.png  (one per run with a JSON)

Aggregates seeds: mean line with std band.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Defaults preserve the legacy behavior (benchmark_results/{results,plots}).
# main() overrides these from --run_dir so plots land in a run's own folder.
RESULT_DIR = os.path.join(HERE, "results")
PLOT_DIR = os.path.join(HERE, "plots")


def load_summary():
    path = os.path.join(RESULT_DIR, "summary.csv")
    if not os.path.exists(path):
        raise SystemExit(f"summary.csv not found at {path}. Run run_benchmark.py first.")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("status") != "ok":
                continue
            try:
                r["label_rate"] = float(r["label_rate"])
                r["acc"] = float(r["acc"]) if r["acc"] else None
                r["f1"] = float(r["f1"]) if r["f1"] else None
                r["seed"] = int(r["seed"])
            except (TypeError, ValueError):
                continue
            rows.append(r)
    return rows


def plot_curve(rows, metric, out_path, ylabel):
    by_tag = defaultdict(lambda: defaultdict(list))  # tag -> lr -> [vals]
    for r in rows:
        if r.get(metric) is None:
            continue
        by_tag[r["tag"]][r["label_rate"]].append(r[metric])

    if not by_tag:
        print(f"No data for metric={metric}; skipping.")
        return

    plt.figure(figsize=(7, 5))
    for tag, lr_dict in sorted(by_tag.items()):
        lrs = sorted(lr_dict.keys())
        means = [np.mean(lr_dict[lr]) for lr in lrs]
        stds = [np.std(lr_dict[lr]) for lr in lrs]
        line, = plt.plot(lrs, means, marker="o", label=tag)
        plt.fill_between(
            lrs,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color=line.get_color(),
            alpha=0.15,
        )
    plt.xscale("log")
    plt.xlabel("Label rate (log scale)")
    plt.ylabel(ylabel)
    plt.title(f"Camargo benchmark: label rate vs {ylabel}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def plot_confusion_matrices(rows):
    for r in rows:
        json_path = r.get("json")
        if not json_path or not os.path.exists(json_path):
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        matrix = np.array(data.get("confusion_matrix", []))
        if matrix.size == 0:
            continue
        tag_safe = r["tag"].replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        out = os.path.join(
            PLOT_DIR,
            f"confusion_{tag_safe}_lr{r['label_rate']}_seed{r['seed']}.png",
        )
        plt.figure(figsize=(5, 4))
        norm = matrix.astype(float) / matrix.sum(axis=1, keepdims=True).clip(min=1)
        plt.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f"{r['tag']}\nlr={r['label_rate']} seed={r['seed']} f1={r['f1']:.3f}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()


def main():
    global RESULT_DIR, PLOT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None,
                    help="Run folder to read/write (results from <run_dir>/results, "
                         "plots to <run_dir>/plots). Defaults to benchmark_results/{results,plots}.")
    args = ap.parse_args()
    if args.run_dir:
        RESULT_DIR = os.path.join(args.run_dir, "results")
        PLOT_DIR = os.path.join(args.run_dir, "plots")
    os.makedirs(PLOT_DIR, exist_ok=True)

    rows = load_summary()
    plot_curve(rows, "f1", os.path.join(PLOT_DIR, "labelrate_vs_f1.png"), "Macro F1")
    plot_curve(rows, "acc", os.path.join(PLOT_DIR, "labelrate_vs_acc.png"), "Accuracy")
    plot_confusion_matrices(rows)


if __name__ == "__main__":
    main()
